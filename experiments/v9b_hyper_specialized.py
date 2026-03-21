#!/usr/bin/env python3
"""
Iterative Retrieval v9b - Hyper-Specialized Verification

v9 showed that 6 general verification methods all converge at +3.3% because
they all catch the same single failure (answer not in context). The other 11
failures need SPECIALIZED interventions.

Root cause analysis of 12/30 baseline failures:
  5x Extraction picks wrong entity from correct paragraph
  2x Over-verbose (already pass relaxed EM)
  2x Retrieval miss on hop 2
  1x Refusal despite answer present
  1x Near-miss entity name
  1x Wrong granularity (fixed by v9 AIC)

Key insight: the hop 2 RELATION tells us the expected answer TYPE. "mother" →
person, "headquarters location" → city, "record label" → organization, etc.
This signal is FREE from the decomposition and can constrain extraction.

Instead of asking "is this answer valid?" (general), ask "is this a PERSON
NAME from the context?" (specialized). The relation→type mapping catches
cases where the answer IS in context but is the wrong entity.

Configs test each specialization independently, then combine:
  0: baseline
  1: answer_type_constraint   - Add "(Answer should be a {type})" to prompt
  2: relation_aware_extract   - Specialized extraction prompt per relation
  3: entity_type_filter       - Extract all entities, filter by expected type
  4: question_echo            - Echo the question type in extraction prompt
  5: contrastive_extract      - "X, not Y" style prompt
  6: answer_position_bias     - Prefer last-mentioned matching entity
  7: combined_best            - Best specializations stacked
  8: combined_plus_aic        - Combined + answer-in-context from v9
"""

import sys
import time
import json
import re
import string
import argparse
import requests
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple


# ── Metrics ──────────────────────────────────────────────────────────

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = set(pred_tokens) & set(truth_tokens)
    if not common or not pred_tokens or not truth_tokens:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)

def exact_match(prediction, ground_truth, aliases=None):
    pred = normalize_answer(prediction)
    if pred == normalize_answer(ground_truth):
        return True
    for alias in (aliases or []):
        if pred == normalize_answer(alias):
            return True
    return False

def relaxed_match(prediction, ground_truth, aliases=None):
    pred = normalize_answer(prediction)
    gold = normalize_answer(ground_truth)
    if pred == gold or gold in pred or pred in gold:
        return True
    for alias in (aliases or []):
        a = normalize_answer(alias)
        if pred == a or a in pred or pred in a:
            return True
    return False

def best_f1(prediction, ground_truth, aliases=None):
    scores = [f1_score(prediction, ground_truth)]
    for alias in (aliases or []):
        scores.append(f1_score(prediction, alias))
    return max(scores)


# ── LLM ──────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434"

def ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32, retries=2):
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens}
                },
                timeout=300
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            if attempt < retries:
                time.sleep(5)
            else:
                raise

def extract_short_answer(response):
    answer = response.split('\n')[0].strip()
    for prefix in ["the answer is", "answer:", "a:", "the "]:
        if answer.lower().startswith(prefix):
            answer = answer[len(prefix):].strip()
    answer = answer.rstrip('.').strip('[]')
    if len(answer) > 80:
        for sep in [',', ';', '(', ' - ', ' who ', ' which ']:
            if sep in answer:
                answer = answer[:answer.index(sep)].strip()
                break
    return answer


# ── Question Formatting ──────────────────────────────────────────────

RELATION_MAP = {
    "performer": "Who performed {subject}?",
    "author": "Who is the author of {subject}?",
    "spouse": "Who is the spouse of {subject}?",
    "child": "Who is the child of {subject}?",
    "father": "Who is the father of {subject}?",
    "mother": "Who is the mother of {subject}?",
    "place of birth": "Where was {subject} born?",
    "headquarters location": "Where is {subject} headquartered?",
    "record label": "What record label is {subject} signed to?",
    "educated at": "Where was {subject} educated?",
    "employer": "Who employs {subject}?",
    "country": "What country is {subject} in?",
    "genre": "What genre is {subject}?",
    "award received": "What award did {subject} receive?",
    "founded by": "Who founded {subject}?",
    "owned by": "Who owns {subject}?",
    "manufacturer": "Who manufactured {subject}?",
    "distributed by": "Who distributed {subject}?",
    "producer": "Who produced {subject}?",
    "director": "Who directed {subject}?",
    "notable work": "What is a notable work by {subject}?",
    "instrument": "What instrument does {subject} play?",
    "has part": "Who is a member of {subject}?",
    "capital": "What is the capital of {subject}?",
    "shares border with": "What borders {subject}?",
}

def format_hop_question(hop_question, previous_answers):
    q = hop_question
    for i, answer in enumerate(previous_answers, 1):
        q = q.replace(f"#{i}", answer)
    if ">>" in q:
        parts = q.split(">>")
        subject = parts[0].strip()
        relation = parts[1].strip().lower()
        for key, template in RELATION_MAP.items():
            if key in relation:
                return template.format(subject=subject)
        if "located in" in relation or "administrative territorial" in relation:
            return f"What administrative region is {subject} located in?"
        return f"What is the {relation} of {subject}?"
    return q


# ── Decomposition ───────────────────────────────────────────────────

DECOMPOSE_TEMPLATE = """Break this complex question into exactly 2 simple sub-questions. The first should identify a key entity, the second should find the final answer about that entity.

Use #1 to reference the answer from sub-question 1.

Examples:

Question: Who is the spouse of the Green performer?
1. Who performed Green?
2. Who is the spouse of #1?

Question: Who founded the company that distributed the film UHF?
1. What company distributed the film UHF?
2. Who founded #1?

Question: Where is Ulrich Walter's employer headquartered?
1. Who employs Ulrich Walter?
2. Where is #1 headquartered?

Question: In what county is William W. Blair's birthplace located?
1. Where was William W. Blair born?
2. What county is #1 located in?

Question: What league does the team that plays in Stadio Ciro Vigorito play for?
1. What team plays in Stadio Ciro Vigorito?
2. What league does #1 play for?

Question: Which company owns the manufacturer of Learjet 60?
1. Who manufactured Learjet 60?
2. Which company owns #1?

Question: What is the capital of the country where the Yenisei River originates?
1. In what country does the Yenisei River originate?
2. What is the capital of #1?

Question: Who directed the film that features the song "Let It Go"?
1. What film features the song "Let It Go"?
2. Who directed #1?

Question: {question}
1."""


def decompose_with_qwen(question):
    response = ask_model(
        DECOMPOSE_TEMPLATE.format(question=question),
        model="qwen2.5:7b", temperature=0.1, max_tokens=80
    )
    full = "1. " + response
    sub_questions = []
    for line in full.split('\n'):
        line = line.strip()
        m = re.match(r'^\d+\.\s+(.+)$', line)
        if m:
            q = m.group(1).strip()
            q = q.rstrip('?').strip() + '?'
            sub_questions.append(q)
    if len(sub_questions) >= 2:
        return sub_questions[:2]
    elif len(sub_questions) == 1:
        return [sub_questions[0], question]
    else:
        return [question]


# ── Embedding Retrieval ──────────────────────────────────────────────

class EmbeddingRetriever:
    def __init__(self):
        self.model = None

    def _load_model(self):
        if self.model is None:
            print("  Loading BGE embeddings...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    def retrieve(self, query, paragraphs, top_k=3):
        self._load_model()
        query_text = f"Represent this sentence for searching relevant passages: {query}"
        para_texts = [f"{p['title']} {p['paragraph_text']}" for p in paragraphs]
        query_emb = self.model.encode([query_text], normalize_embeddings=True)
        para_embs = self.model.encode(para_texts, normalize_embeddings=True)
        sims = np.dot(para_embs, query_emb.T).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(int(idx), float(sims[idx])) for idx in top_indices]

    def get_all_scores(self, query, paragraphs):
        self._load_model()
        query_text = f"Represent this sentence for searching relevant passages: {query}"
        para_texts = [f"{p['title']} {p['paragraph_text']}" for p in paragraphs]
        query_emb = self.model.encode([query_text], normalize_embeddings=True)
        para_embs = self.model.encode(para_texts, normalize_embeddings=True)
        return np.dot(para_embs, query_emb.T).flatten()


# ── Extraction Prompt ────────────────────────────────────────────────

PROMPT_8SHOT = """Answer the question using the context below. Give ONLY the specific name, place, or fact. One or two words maximum.

Context: [Steve Hillage] Green is the fourth studio album by British progressive rock musician Steve Hillage.
Question: Who performed Green?
Answer: Steve Hillage

Context: [Miquette Giraudy] Miquette Giraudy is a keyboard player, best known for her work with her partner Steve Hillage.
Question: Who is the spouse of Steve Hillage?
Answer: Miquette Giraudy

Context: [Orion Pictures] The film was distributed by Orion Pictures, founded by Mike Medavoy and four other executives.
Question: Who founded Orion Pictures?
Answer: Mike Medavoy

Context: [Canyon, Texas] Canyon is a city in and the county seat of Randall County, Texas, United States.
Question: What administrative region is Canyon located in?
Answer: Randall County

Context: [Daniel Webster] Daniel Webster married Caroline LeRoy in 1829. Their son Fletcher Webster served in the Union Army.
Question: Who is the child of Daniel Webster?
Answer: Fletcher Webster

Context: [Michelle Phillips] Michelle Phillips is an American singer. Her daughter Chynna Phillips is also a singer.
Question: Who is the child of Michelle Phillips?
Answer: Chynna Phillips

Context: [Lawrence Sanders] Lawrence Sanders was an American author known for The Anderson Tapes and The Timothy Files.
Question: Who is the author of The Timothy Files?
Answer: Lawrence Sanders

Context: [Shaun Tan] Shaun Tan won the Academy Award for Best Animated Short Film for The Lost Thing.
Question: What award did Shaun Tan receive?
Answer: Academy Award for Best Animated Short Film

Context: {context}
Question: {question}
Answer:"""


# ══════════════════════════════════════════════════════════════════════
# ANSWER TYPE SYSTEM — The core specialization
# ══════════════════════════════════════════════════════════════════════

# Map relation keywords → expected answer type + extraction constraint
RELATION_TO_TYPE = {
    # People
    "spouse": ("person name", "a person's full name"),
    "child": ("person name", "a person's full name"),
    "father": ("person name", "a person's full name"),
    "mother": ("person name", "a person's full name"),
    "performer": ("person name", "a person's or band's name"),
    "author": ("person name", "a person's name"),
    "founded by": ("person name", "a person's name"),
    "owned by": ("person or organization name", "a person's or organization's name"),
    "employer": ("organization name", "an organization or company name"),
    "producer": ("person name", "a person's name"),
    "director": ("person name", "a person's name"),
    "manufacturer": ("organization name", "a company name"),
    "distributed by": ("organization name", "a company name"),
    "has part": ("person name", "a person's name"),
    # Places
    "place of birth": ("place name", "a city or town name"),
    "headquarters location": ("city name", "a city name"),
    "country": ("country name", "a country name"),
    "capital": ("city name", "a city name"),
    "shares border with": ("place name", "a county, state, or country name"),
    "administrative territorial": ("administrative region", "a state, county, or province name"),
    "located in": ("administrative region", "a state, county, or province name"),
    # Things
    "record label": ("record label name", "a record label name"),
    "genre": ("genre name", "a music or literary genre"),
    "award received": ("award name", "an award name"),
    "notable work": ("work title", "a title of a book, film, play, or work"),
    "instrument": ("instrument name", "a musical instrument"),
    "educated at": ("institution name", "a school, college, or university name"),
    # Abstract
    "followed by": ("entity name", "the name of the successor entity"),
    "movement": ("concept or movement name", "the name of a movement or ideology"),
}


def infer_answer_type(hop_question, original_question=""):
    """Infer expected answer type from the hop question's relation or wording."""
    q_lower = hop_question.lower()

    # Check >> relation format first (most precise)
    if ">>" in hop_question:
        relation = hop_question.split(">>")[1].strip().lower()
        for key, (short, long) in RELATION_TO_TYPE.items():
            if key in relation:
                return short, long

    # Fall back to question word patterns
    for key, (short, long) in RELATION_TO_TYPE.items():
        if key in q_lower:
            return short, long

    # Generic question word heuristics
    if q_lower.startswith("who"):
        return "person name", "a person's name"
    if "where" in q_lower or "what city" in q_lower or "what country" in q_lower:
        return "place name", "a place name"
    if "what county" in q_lower or "what state" in q_lower or "what province" in q_lower:
        return "administrative region", "an administrative region name"
    if q_lower.startswith("when") or "what year" in q_lower:
        return "date or year", "a date or year"

    # Also check the original multi-hop question for clues
    if original_question:
        oq = original_question.lower()
        if "who" in oq:
            return "person name", "a person's name"
        if "where" in oq:
            return "place name", "a place name"

    return "", ""


def check_answer_in_context(answer, context):
    if not answer or not context:
        return False
    return normalize_answer(answer) in normalize_answer(context)


# ══════════════════════════════════════════════════════════════════════
# SHARED HOP HELPERS
# ══════════════════════════════════════════════════════════════════════

def _prepare_hop_question(hop, previous_answers):
    hop_q = hop["question"]
    for j, ans in enumerate(previous_answers, 1):
        hop_q = hop_q.replace(f"#{j}", ans)
    if ">>" in hop_q:
        hop_q = format_hop_question(hop_q, [])
    return hop_q


def _build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices):
    gold_idx = gold_decomp[i]["paragraph_support_idx"] if i < len(gold_decomp) else -1
    gold_answer = gold_decomp[i]["answer"] if i < len(gold_decomp) else "N/A"
    em = exact_match(answer, gold_answer) if gold_answer != "N/A" else None
    return {
        "hop": i + 1, "question": hop_q, "gold_answer": gold_answer,
        "predicted": answer, "em": em,
        "gold_retrieved": gold_idx in retrieved_indices,
    }


def _retrieve_and_format(hop_q, paragraphs, retriever, top_k=3, query_expansion=None):
    """Retrieve paragraphs and format context string."""
    query = hop_q
    if query_expansion:
        query = f"{hop_q} {query_expansion}"
    results = retriever.retrieve(query, paragraphs, top_k=top_k)
    retrieved_indices = [idx for idx, _ in results]
    context = "\n\n".join(
        f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
        for idx, _ in results
    )
    return context, results, retrieved_indices


# ══════════════════════════════════════════════════════════════════════
# PIPELINE CONFIGS
# ══════════════════════════════════════════════════════════════════════

# ── Config 0: Baseline ──────────────────────────────────────────────

def pipeline_baseline(sample, retriever, auto_hops, stats):
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        context, results, retrieved_indices = _retrieve_and_format(hop_q, paragraphs, retriever)

        prompt = PROMPT_8SHOT.format(context=context, question=hop_q)
        response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
        answer = extract_short_answer(response)

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 1: Answer Type Constraint ────────────────────────────────
# Appends "(Answer should be a {type})" to the question in the prompt.

def pipeline_answer_type_constraint(sample, retriever, auto_hops, stats):
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        context, results, retrieved_indices = _retrieve_and_format(hop_q, paragraphs, retriever)

        short_type, _ = infer_answer_type(hop["question"], sample["question"])
        constrained_q = hop_q
        if short_type:
            constrained_q = f"{hop_q} (Answer should be a {short_type})"
            stats["typed"] += 1
        else:
            stats["untyped"] += 1

        prompt = PROMPT_8SHOT.format(context=context, question=constrained_q)
        response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
        answer = extract_short_answer(response)

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 2: Relation-Aware Extraction ──────────────────────────────
# Completely replaces the extraction prompt with a relation-specific one.

RELATION_EXTRACT_PROMPT = """Extract the {long_type} from the context that answers the question. Give ONLY the specific name. One or two words maximum. Do not explain.

Context: {context}
Question: {question}
{long_type}:"""

def pipeline_relation_aware_extract(sample, retriever, auto_hops, stats):
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        context, results, retrieved_indices = _retrieve_and_format(hop_q, paragraphs, retriever)

        _, long_type = infer_answer_type(hop["question"], sample["question"])
        if long_type:
            stats["specialized"] += 1
            prompt = RELATION_EXTRACT_PROMPT.format(
                context=context, question=hop_q, long_type=long_type
            )
        else:
            stats["generic"] += 1
            prompt = PROMPT_8SHOT.format(context=context, question=hop_q)

        response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
        answer = extract_short_answer(response)

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 3: Entity Type Filter ─────────────────────────────────────
# Extract multiple candidate answers, filter by expected type using LLM.

def pipeline_entity_type_filter(sample, retriever, auto_hops, stats):
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        context, results, retrieved_indices = _retrieve_and_format(hop_q, paragraphs, retriever)

        short_type, _ = infer_answer_type(hop["question"], sample["question"])

        # Primary extraction
        prompt = PROMPT_8SHOT.format(context=context, question=hop_q)
        response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
        answer = extract_short_answer(response)

        if short_type:
            # Check if the answer matches expected type
            type_check_prompt = (
                f"Is '{answer}' a {short_type}? Answer only YES or NO."
            )
            check = ask_model(type_check_prompt, model="qwen2.5:7b", temperature=0.0, max_tokens=4)
            first_char = check.strip().lower()[:1]

            if first_char == 'n':
                stats["filtered"] += 1
                # Re-extract with type constraint
                constrained_prompt = PROMPT_8SHOT.format(
                    context=context,
                    question=f"{hop_q} (Answer must be a {short_type})"
                )
                response2 = ask_model(constrained_prompt, model="qwen2.5:7b", temperature=0.0, max_tokens=32)
                answer = extract_short_answer(response2)
            else:
                stats["passed_filter"] += 1
        else:
            stats["no_type"] += 1

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 4: Question Echo ──────────────────────────────────────────
# Repeats the answer type expectation at the end of the prompt.

def pipeline_question_echo(sample, retriever, auto_hops, stats):
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        context, results, retrieved_indices = _retrieve_and_format(hop_q, paragraphs, retriever)

        short_type, _ = infer_answer_type(hop["question"], sample["question"])

        prompt = PROMPT_8SHOT.format(context=context, question=hop_q)
        if short_type:
            prompt += f"\n(Remember: answer with a {short_type})"
            stats["echoed"] += 1

        response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
        answer = extract_short_answer(response)

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 5: Contrastive Extract ────────────────────────────────────
# "Extract the X, not the Y" — prevents confusion between entity types.

CONTRASTIVE_MAP = {
    "person name": "not a place, organization, or title",
    "city name": "not a person, country, or organization",
    "country name": "not a city, person, or organization",
    "place name": "not a person or organization",
    "administrative region": "not a country, city, or person",
    "organization name": "not a person or place",
    "record label name": "not a person, song, or album title",
    "work title": "not a person's name or an organization",
    "institution name": "not a person's name or degree",
    "award name": "not a person or organization",
    "entity name": "not a person if asking about a company, not a company if asking about a person",
}

def pipeline_contrastive_extract(sample, retriever, auto_hops, stats):
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        context, results, retrieved_indices = _retrieve_and_format(hop_q, paragraphs, retriever)

        short_type, _ = infer_answer_type(hop["question"], sample["question"])
        contrast = CONTRASTIVE_MAP.get(short_type, "")

        if short_type and contrast:
            constrained_q = f"{hop_q} (Answer with a {short_type}, {contrast})"
            stats["contrastive"] += 1
        else:
            constrained_q = hop_q
            stats["plain"] += 1

        prompt = PROMPT_8SHOT.format(context=context, question=constrained_q)
        response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
        answer = extract_short_answer(response)

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 6: Answer Position Bias ───────────────────────────────────
# When the answer type is known, prefer the LAST matching entity mentioned
# in the context (counters first-mention bias seen in Dill Records case).

def pipeline_answer_position_bias(sample, retriever, auto_hops, stats):
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        context, results, retrieved_indices = _retrieve_and_format(hop_q, paragraphs, retriever)

        short_type, long_type = infer_answer_type(hop["question"], sample["question"])

        # Primary extraction
        prompt = PROMPT_8SHOT.format(context=context, question=hop_q)
        response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
        answer = extract_short_answer(response)

        # If we have a type, also ask for ALL entities of that type
        if short_type and long_type:
            list_prompt = (
                f"List every {long_type} mentioned in this text. "
                f"One per line, nothing else.\n\n"
                f"Text: {context}\n\n"
                f"List of {short_type}s:"
            )
            list_response = ask_model(list_prompt, model="qwen2.5:7b", temperature=0.0, max_tokens=100)
            entities = [e.strip().lstrip('- •0123456789.)')
                       for e in list_response.split('\n') if e.strip()]
            entities = [e for e in entities if e and len(e) > 1]

            if entities:
                stats["listed"] += 1
                # Check if primary answer is in the list
                norm_answer = normalize_answer(answer)
                norm_entities = [normalize_answer(e) for e in entities]
                if norm_answer not in norm_entities and entities:
                    # Primary answer not recognized as correct type — use last entity
                    answer = entities[-1]
                    stats["reselected"] += 1
            else:
                stats["no_entities"] += 1
        else:
            stats["no_type"] += 1

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 7: Combined Best ──────────────────────────────────────────
# Type constraint + contrastive + greedy decoding. No extra LLM calls.

def pipeline_combined_best(sample, retriever, auto_hops, stats):
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        context, results, retrieved_indices = _retrieve_and_format(hop_q, paragraphs, retriever)

        short_type, long_type = infer_answer_type(hop["question"], sample["question"])
        contrast = CONTRASTIVE_MAP.get(short_type, "")

        if short_type and contrast:
            constrained_q = f"{hop_q} (Answer with a {short_type}, {contrast})"
        elif short_type:
            constrained_q = f"{hop_q} (Answer should be a {short_type})"
        else:
            constrained_q = hop_q

        prompt = PROMPT_8SHOT.format(context=context, question=constrained_q)
        # Greedy decoding for determinism
        response = ask_model(prompt, model="qwen2.5:7b", temperature=0.0, max_tokens=32)
        answer = extract_short_answer(response)

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 8: Combined + AIC ────────────────────────────────────────
# Best specialization + answer-in-context fallback from v9.

def pipeline_combined_plus_aic(sample, retriever, auto_hops, stats):
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        context, results, retrieved_indices = _retrieve_and_format(hop_q, paragraphs, retriever)

        short_type, long_type = infer_answer_type(hop["question"], sample["question"])
        contrast = CONTRASTIVE_MAP.get(short_type, "")

        if short_type and contrast:
            constrained_q = f"{hop_q} (Answer with a {short_type}, {contrast})"
        elif short_type:
            constrained_q = f"{hop_q} (Answer should be a {short_type})"
        else:
            constrained_q = hop_q

        prompt = PROMPT_8SHOT.format(context=context, question=constrained_q)
        response = ask_model(prompt, model="qwen2.5:7b", temperature=0.0, max_tokens=32)
        answer = extract_short_answer(response)

        # AIC check
        if not check_answer_in_context(answer, context):
            stats["aic_triggered"] += 1
            # Re-extract with even tighter constraint
            retry_prompt = PROMPT_8SHOT.format(context=context, question=hop_q)
            response2 = ask_model(retry_prompt, model="qwen2.5:7b", temperature=0.0, max_tokens=16)
            new_answer = extract_short_answer(response2)
            if check_answer_in_context(new_answer, context):
                answer = new_answer
                stats["aic_replaced"] += 1
            else:
                # Try k+1 fallback
                results_ext = retriever.retrieve(hop_q, paragraphs, top_k=4)
                if len(results_ext) > 3:
                    extra_idx = results_ext[3][0]
                    extra_ctx = context + f"\n\n[{paragraphs[extra_idx]['title']}] {paragraphs[extra_idx]['paragraph_text']}"
                    prompt3 = PROMPT_8SHOT.format(context=extra_ctx, question=constrained_q)
                    response3 = ask_model(prompt3, model="qwen2.5:7b", temperature=0.0, max_tokens=16)
                    new_answer2 = extract_short_answer(response3)
                    if check_answer_in_context(new_answer2, extra_ctx):
                        answer = new_answer2
                        stats["aic_replaced"] += 1
                    else:
                        stats["aic_kept"] += 1
                else:
                    stats["aic_kept"] += 1
        else:
            stats["aic_passed"] += 1

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ══════════════════════════════════════════════════════════════════════
# CONFIG REGISTRY
# ══════════════════════════════════════════════════════════════════════

CONFIGS = [
    {"id": 0, "name": "baseline",
     "fn": pipeline_baseline,
     "description": "Standard v5 pipeline, no specialization"},
    {"id": 1, "name": "answer_type_constraint",
     "fn": pipeline_answer_type_constraint,
     "description": "Append '(Answer should be a {type})' to prompt"},
    {"id": 2, "name": "relation_aware_extract",
     "fn": pipeline_relation_aware_extract,
     "description": "Specialized extraction prompt per relation type"},
    {"id": 3, "name": "entity_type_filter",
     "fn": pipeline_entity_type_filter,
     "description": "Extract then verify answer matches expected type"},
    {"id": 4, "name": "question_echo",
     "fn": pipeline_question_echo,
     "description": "Echo answer type reminder after prompt"},
    {"id": 5, "name": "contrastive_extract",
     "fn": pipeline_contrastive_extract,
     "description": "'Answer with X, not Y' contrastive constraint"},
    {"id": 6, "name": "answer_position_bias",
     "fn": pipeline_answer_position_bias,
     "description": "List all typed entities, counter first-mention bias"},
    {"id": 7, "name": "combined_best",
     "fn": pipeline_combined_best,
     "description": "Type constraint + contrastive + greedy, no extra LLM"},
    {"id": 8, "name": "combined_plus_aic",
     "fn": pipeline_combined_plus_aic,
     "description": "Combined best + answer-in-context fallback"},
]


# ══════════════════════════════════════════════════════════════════════
# TEST RUNNER
# ══════════════════════════════════════════════════════════════════════

def test_one_config(config, samples, retriever, all_auto_decomps):
    fn = config["fn"]
    em_scores, rem_scores, f1_scores, latencies = [], [], [], []
    per_question = []
    stats = defaultdict(int)  # Each config tracks its own stats

    for sample in samples:
        auto_sub_qs = all_auto_decomps[sample["id"]]
        auto_hops = [{"question": sq} for sq in auto_sub_qs]

        t0 = time.time()
        pred, hops = fn(sample, retriever, auto_hops, stats)
        lat = (time.time() - t0) * 1000

        answer = sample["answer"]
        aliases = sample.get("answer_aliases", [])
        em = exact_match(pred, answer, aliases)
        rem = relaxed_match(pred, answer, aliases)
        f1 = best_f1(pred, answer, aliases)

        em_scores.append(em)
        rem_scores.append(rem)
        f1_scores.append(f1)
        latencies.append(lat)
        per_question.append({
            "id": sample["id"], "prediction": pred, "answer": answer,
            "em": em, "relaxed_em": rem, "f1": f1,
        })

    n = len(samples)
    return {
        "em": sum(em_scores) / n * 100,
        "relaxed_em": sum(rem_scores) / n * 100,
        "f1": sum(f1_scores) / n,
        "latency_ms": sum(latencies) / n,
        "per_question": per_question,
        "specialization_stats": stats,
    }


def _save(output_path, results, n_configs, n_samples):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "metadata": {
            "experiment": "v9b_hyper_specialized",
            "n_configs": n_configs,
            "n_completed": len(results),
            "n_samples": n_samples,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "configs": results,
    }
    with open(output, "w") as f:
        json.dump(data, f, indent=2, default=str)


def run_experiment(limit=30, output_path="results/v9b_hyper_specialized.json",
                   resume=True, configs_to_run=None):
    from datasets import load_dataset

    print("Loading MuSiQue validation set...")
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    samples = [s for s in ds if s.get("answerable", True)][:limit]
    print(f"Testing on {len(samples)} answerable questions\n")

    retriever = EmbeddingRetriever()
    retriever._load_model()

    # Phase 1: Decomposition
    print("=" * 70)
    print("PHASE 1: DECOMPOSITION")
    print("=" * 70)

    all_auto_decomps = {}
    for i, sample in enumerate(samples):
        auto_sub_qs = decompose_with_qwen(sample["question"])
        all_auto_decomps[sample["id"]] = auto_sub_qs
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] decomposed")
    print(f"  Decomposed {len(samples)} questions")

    # Check answer type coverage
    typed, untyped = 0, 0
    for sample in samples:
        for sq in all_auto_decomps[sample["id"]]:
            short, _ = infer_answer_type(sq, sample["question"])
            if short:
                typed += 1
            else:
                untyped += 1
    print(f"  Type inference: {typed}/{typed+untyped} hops typed ({typed/(typed+untyped)*100:.0f}%)")

    # Load previous results
    output = Path(output_path)
    existing_results = {}
    if resume and output.exists():
        with open(output) as f:
            data = json.load(f)
        existing_results = data.get("configs", {})
        print(f"  Resuming: {len(existing_results)} configs already completed")

    # Phase 2: Run configs
    configs = CONFIGS
    if configs_to_run is not None:
        configs = [c for c in CONFIGS if c["id"] in configs_to_run]

    print(f"\n{'=' * 70}")
    print(f"PHASE 2: TESTING {len(configs)} HYPER-SPECIALIZED CONFIGS")
    print("=" * 70)

    baseline_em = None

    for config in configs:
        key = f"{config['id']}_{config['name']}"

        if key in existing_results:
            result = existing_results[key]
            if config["id"] == 0:
                baseline_em = result["em"]
            print(f"  [cached] Config {config['id']}: {config['name']:35s} EM={result['em']:5.1f}%")
            continue

        print(f"\n  Running Config {config['id']}: {config['name']}...")
        print(f"  {config['description']}")

        t0 = time.time()
        try:
            result = test_one_config(config, samples, retriever, all_auto_decomps)
        except Exception as e:
            result = {"em": 0, "relaxed_em": 0, "f1": 0, "latency_ms": 0,
                     "error": str(e), "per_question": [], "specialization_stats": {}}
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

        elapsed = time.time() - t0
        existing_results[key] = result

        if config["id"] == 0:
            baseline_em = result["em"]

        delta = f" ({result['em'] - baseline_em:+.1f}%)" if baseline_em is not None and config["id"] != 0 else ""
        ss = result.get("specialization_stats", {})
        stats_str = "  " + " ".join(f"{k}={v}" for k, v in ss.items()) if ss else ""

        print(f"  -> EM={result['em']:5.1f}%{delta}  "
              f"relEM={result['relaxed_em']:5.1f}%  "
              f"F1={result['f1']:.3f}  "
              f"lat={result['latency_ms']:.0f}ms  [{elapsed:.0f}s]"
              f"{stats_str}")

        _save(output_path, existing_results, len(CONFIGS), len(samples))

    _save(output_path, existing_results, len(CONFIGS), len(samples))

    # Summary
    print(f"\n{'=' * 70}")
    print(f"FINAL SUMMARY: v9b Hyper-Specialized")
    print(f"{'=' * 70}")

    if baseline_em is None:
        baseline_em = existing_results.get("0_baseline", {}).get("em", 60.0)

    print(f"\nBaseline EM: {baseline_em:.1f}%\n")
    print(f"{'Cfg':<4} {'Name':<35} {'EM':>6} {'Delta':>7} {'relEM':>6} "
          f"{'F1':>6} {'Lat':>7}")
    print("-" * 80)

    for config in CONFIGS:
        key = f"{config['id']}_{config['name']}"
        r = existing_results.get(key)
        if not r:
            continue
        delta = r["em"] - baseline_em
        marker = "+" if delta > 0.5 else ("-" if delta < -0.5 else "=")
        print(f"[{marker}] {config['id']:<2} {config['name']:<35} {r['em']:5.1f}% "
              f"{delta:+5.1f}%  {r['relaxed_em']:5.1f}% {r['f1']:.3f} "
              f"{r['latency_ms']:6.0f}ms")

    # Per-question diff against baseline
    print(f"\n{'=' * 70}")
    print("PER-QUESTION DIFF (configs that changed answers vs baseline)")
    print("=" * 70)

    base_pq = {q["id"]: q for q in existing_results.get("0_baseline", {}).get("per_question", [])}
    for config in CONFIGS:
        if config["id"] == 0:
            continue
        key = f"{config['id']}_{config['name']}"
        r = existing_results.get(key, {})
        changes = []
        for q in r.get("per_question", []):
            bq = base_pq.get(q["id"])
            if bq and q["prediction"] != bq["prediction"]:
                direction = ""
                if q["em"] and not bq["em"]:
                    direction = "FIXED"
                elif not q["em"] and bq["em"]:
                    direction = "BROKE"
                elif q["em"] == bq["em"]:
                    direction = "DIFF"
                changes.append((q["id"], bq["prediction"], q["prediction"], q["answer"], direction))
        if changes:
            delta = r["em"] - baseline_em
            print(f"\n  Config {config['id']} ({config['name']}) [{delta:+.1f}%]:")
            for qid, old, new, gold, direction in changes:
                print(f"    [{direction:5s}] {qid[:25]:25s}  '{old}' → '{new}'  (gold: '{gold}')")


def main():
    parser = argparse.ArgumentParser(description="v9b: Hyper-Specialized Verification")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", type=str, default="results/v9b_hyper_specialized.json")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config IDs to run (e.g. '0,1,7')")
    args = parser.parse_args()

    configs_to_run = None
    if args.configs:
        configs_to_run = [int(x) for x in args.configs.split(",")]

    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        assert any("qwen2.5:7b" in m for m in models), "qwen2.5:7b not found"
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    start = time.time()
    run_experiment(
        limit=args.limit,
        output_path=args.output,
        resume=not args.no_resume,
        configs_to_run=configs_to_run,
    )
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
