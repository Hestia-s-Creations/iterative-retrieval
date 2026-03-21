#!/usr/bin/env python3
"""
Iterative Retrieval v9c - Fundamentally Different Approaches

v9/v9b proved: parameter tuning and verification are exhausted at +3.3%.
The bottleneck is extraction — the model has the right paragraph but picks
the wrong entity ~40% of the time. This is a SELECTION problem, not a
GENERATION problem. We've been using generative approaches for a
discriminative task.

6 fundamentally different paradigms:

  0: baseline             — Standard Decompose→Retrieve→Extract (generative)
  1: extractive_span      — NER + embedding similarity, zero LLM extraction
  2: no_decomposition     — Skip decomposition, retrieve for full question
  3: reverse_pipeline     — Retrieve→Extract candidates→Decompose to verify
  4: pairwise_ranking     — Extract candidates, LLM picks from multiple choice
  5: embedding_only       — Pure BGE, no LLM anywhere (even decomposition)
  6: two_model_split      — Qwen decomposes, NER+BGE extracts
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
from collections import defaultdict
from typing import List, Tuple


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

    def score_candidates(self, question, candidates):
        """Score candidate answers against a question using embedding similarity."""
        self._load_model()
        if not candidates:
            return []
        q_text = f"Represent this sentence for searching relevant passages: {question}"
        q_emb = self.model.encode([q_text], normalize_embeddings=True)
        c_embs = self.model.encode(candidates, normalize_embeddings=True)
        scores = np.dot(c_embs, q_emb.T).flatten()
        return list(zip(candidates, scores.tolist()))

    def encode_text(self, text):
        self._load_model()
        return self.model.encode([text], normalize_embeddings=True)[0]


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
# NER — Extract named entities / noun phrases from text
# ══════════════════════════════════════════════════════════════════════

def extract_noun_phrases(text):
    """Extract candidate entities using regex-based NER.

    Captures: capitalized multi-word names, quoted titles, parenthetical
    names, and standalone capitalized words. No external NER model needed.
    """
    candidates = set()

    # Paragraph titles (already bracketed in our format)
    for m in re.finditer(r'\[([^\]]+)\]', text):
        candidates.add(m.group(1).strip())

    # Quoted titles
    for m in re.finditer(r'"([^"]+)"', text):
        candidates.add(m.group(1).strip())

    # Capitalized multi-word sequences (proper nouns)
    # Match 1-6 capitalized words in sequence, allowing small connectors
    for m in re.finditer(
        r'(?<![.\d])\b([A-Z][a-zà-ÿ]+(?:\s+(?:of|the|de|von|van|di|del|la|le|des|d\'|und|and|for|in|at|on|by)\s+)?'
        r'(?:[A-Z][a-zà-ÿ]+(?:\s+(?:of|the|de|von|van|di|del|la|le|des|d\'|und|and|for|in|at|on|by)\s+)?){0,5})',
        text
    ):
        name = m.group(0).strip()
        if len(name) > 1 and name not in ('The', 'A', 'An', 'In', 'On', 'At', 'By', 'For', 'And', 'Or'):
            candidates.add(name)

    # Also grab individual capitalized words (single-word entities)
    for m in re.finditer(r'\b([A-Z][a-zà-ÿ]{2,})\b', text):
        word = m.group(1)
        if word not in ('The', 'This', 'That', 'These', 'Those', 'There', 'Here',
                        'After', 'Before', 'During', 'While', 'Where', 'When',
                        'Which', 'What', 'How', 'Who', 'His', 'Her', 'Its',
                        'They', 'She', 'Was', 'Were', 'Has', 'Had', 'Have',
                        'Are', 'Being', 'Been', 'Also', 'Other', 'Some',
                        'Many', 'Most', 'Such', 'Much', 'Very', 'Not'):
            candidates.add(word)

    # Parenthetical names: "born Name Name" or "(Name Name)"
    for m in re.finditer(r'\(([^)]+)\)', text):
        inner = m.group(1).strip()
        if re.match(r'^[A-Z]', inner) and len(inner) < 60:
            candidates.add(inner)

    # Years and dates (useful for "when" questions)
    for m in re.finditer(r'\b(\d{4})\b', text):
        candidates.add(m.group(1))

    return list(candidates)


# ══════════════════════════════════════════════════════════════════════
# SHARED HELPERS
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


def _retrieve_and_format(hop_q, paragraphs, retriever, top_k=3):
    results = retriever.retrieve(hop_q, paragraphs, top_k=top_k)
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
    """Standard Decompose→Retrieve→Extract (generative)."""
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


# ── Config 1: Extractive Span Selection ──────────────────────────────
# NER extracts all candidate entities from retrieved paragraphs.
# BGE scores each candidate against the hop question.
# Highest-scoring candidate wins. Zero LLM calls for extraction.

def pipeline_extractive_span(sample, retriever, auto_hops, stats):
    """NER + embedding similarity extraction. No LLM for answer selection."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        context, results, retrieved_indices = _retrieve_and_format(hop_q, paragraphs, retriever)

        # Extract all candidate entities from retrieved paragraphs
        candidates = extract_noun_phrases(context)
        stats["total_candidates"] += len(candidates)

        if candidates:
            # Score each candidate against the hop question
            scored = retriever.score_candidates(hop_q, candidates)
            scored.sort(key=lambda x: x[1], reverse=True)
            answer = scored[0][0]
            stats["extracted"] += 1
        else:
            # Fallback: use paragraph title
            top_idx = results[0][0] if results else 0
            answer = paragraphs[top_idx]["title"]
            stats["fallback_title"] += 1

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 2: No Decomposition ──────────────────────────────────────
# Skip decomposition entirely. Retrieve for the full question, extract
# from top-k paragraphs in a single pass. Tests whether decomposition
# is actually helping or just adding error propagation.

def pipeline_no_decomposition(sample, retriever, auto_hops, stats):
    """Single-pass: retrieve for full question, extract directly."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    question = sample["question"]

    # Retrieve for the full multi-hop question
    context, results, retrieved_indices = _retrieve_and_format(
        question, paragraphs, retriever, top_k=5
    )

    prompt = PROMPT_8SHOT.format(context=context, question=question)
    response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
    answer = extract_short_answer(response)

    hop_details = [{
        "hop": 1, "question": question, "gold_answer": sample["answer"],
        "predicted": answer, "em": exact_match(answer, sample["answer"],
                                                sample.get("answer_aliases", [])),
        "gold_retrieved": any(
            gold_decomp[j]["paragraph_support_idx"] in retrieved_indices
            for j in range(len(gold_decomp))
        ),
    }]

    return answer, hop_details


# ── Config 3: Reverse Pipeline ──────────────────────────────────────
# Retrieve for full question first, extract candidate entities,
# THEN use decomposition to verify/refine.

def pipeline_reverse(sample, retriever, auto_hops, stats):
    """Retrieve→Extract candidates→Decompose to verify."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    question = sample["question"]

    # Step 1: Broad retrieval for full question
    context_full, results_full, indices_full = _retrieve_and_format(
        question, paragraphs, retriever, top_k=5
    )

    # Step 2: Extract candidate answers from broad context
    candidates = extract_noun_phrases(context_full)
    stats["initial_candidates"] += len(candidates)

    if not candidates:
        # Fallback to generative
        prompt = PROMPT_8SHOT.format(context=context_full, question=question)
        response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
        answer = extract_short_answer(response)
        stats["generative_fallback"] += 1
    else:
        # Step 3: Use decomposition to score candidates
        # For each candidate, check: does hop 1 question retrieve a paragraph
        # mentioning this candidate? (verification by retrieval)
        hop1_q = _prepare_hop_question(auto_hops[0], [])

        # Score candidates against hop 2 question (where the final answer lives)
        if len(auto_hops) >= 2:
            # First get hop 1 answer via standard pipeline
            context1, results1, _ = _retrieve_and_format(hop1_q, paragraphs, retriever)
            prompt1 = PROMPT_8SHOT.format(context=context1, question=hop1_q)
            resp1 = ask_model(prompt1, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
            hop1_answer = extract_short_answer(resp1)

            # Now score candidates against hop 2 question with hop 1 answer filled in
            hop2_q = _prepare_hop_question(auto_hops[1], [hop1_answer])
            scored = retriever.score_candidates(hop2_q, candidates)
        else:
            scored = retriever.score_candidates(question, candidates)

        scored.sort(key=lambda x: x[1], reverse=True)
        answer = scored[0][0]
        stats["candidate_selected"] += 1

    hop_details = [{
        "hop": 1, "question": question, "gold_answer": sample["answer"],
        "predicted": answer,
        "em": exact_match(answer, sample["answer"], sample.get("answer_aliases", [])),
        "gold_retrieved": any(
            gold_decomp[j]["paragraph_support_idx"] in indices_full
            for j in range(len(gold_decomp))
        ),
    }]

    return answer, hop_details


# ── Config 4: Pairwise Ranking (Multiple Choice) ────────────────────
# Standard decomposition + retrieval, but extraction is multiple choice.
# Extract all entities, present as options A/B/C/D, LLM picks.

MC_PROMPT = """Given the context, which option best answers the question? Reply with ONLY the letter (A, B, C, etc.).

Context: {context}
Question: {question}

Options:
{options}

Answer:"""

def pipeline_pairwise_ranking(sample, retriever, auto_hops, stats):
    """Decompose→Retrieve→Multiple-choice selection."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        context, results, retrieved_indices = _retrieve_and_format(hop_q, paragraphs, retriever)

        # Extract candidates from context
        candidates = extract_noun_phrases(context)

        # Deduplicate by normalized form, keep originals
        seen = set()
        unique_candidates = []
        for c in candidates:
            norm = normalize_answer(c)
            if norm not in seen and len(c) > 1:
                seen.add(norm)
                unique_candidates.append(c)

        if len(unique_candidates) >= 2:
            # Score candidates to get top-N most relevant
            scored = retriever.score_candidates(hop_q, unique_candidates)
            scored.sort(key=lambda x: x[1], reverse=True)
            top_candidates = [c for c, s in scored[:8]]  # Top 8 candidates

            # Format as multiple choice
            letters = "ABCDEFGH"
            options = "\n".join(f"{letters[j]}. {c}" for j, c in enumerate(top_candidates))

            prompt = MC_PROMPT.format(context=context, question=hop_q, options=options)
            response = ask_model(prompt, model="qwen2.5:7b", temperature=0.0, max_tokens=4)

            # Parse letter
            letter = response.strip().upper()[:1]
            if letter in letters[:len(top_candidates)]:
                idx = letters.index(letter)
                answer = top_candidates[idx]
                stats["mc_selected"] += 1
            else:
                # Fallback to highest-scored candidate
                answer = top_candidates[0]
                stats["mc_fallback_top"] += 1
        else:
            # Too few candidates, use generative extraction
            prompt = PROMPT_8SHOT.format(context=context, question=hop_q)
            response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
            answer = extract_short_answer(response)
            stats["mc_fallback_gen"] += 1

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 5: Embedding Only (No LLM Anywhere) ──────────────────────
# Pure BGE pipeline. Decomposition via template matching, retrieval via
# BGE, extraction via entity-question embedding similarity.
# Tests the floor: how good can you get with ZERO LLM calls?

def template_decompose(question):
    """Rule-based decomposition without LLM. Heuristic only."""
    q = question.strip()

    # Pattern: "What/Who is the X of the Y of Z?"
    # Try to split on relational markers
    patterns = [
        # "Who is the spouse of the performer of Green?"
        (r"(?:who|what)\s+(?:is|was|are|were)\s+(?:the\s+)?(\w+(?:\s+\w+)?)\s+of\s+(?:the\s+)?(.+?)(?:\s+of\s+(.+))?[?]?$",
         lambda m: [
             f"What is the {m.group(2)} of {m.group(3)}?" if m.group(3) else m.group(2) + "?",
             f"What is the {m.group(1)} of #1?"
         ]),
        # "Where is X's Y headquartered?"
        (r"where\s+(?:is|was)\s+(.+?)(?:'s|'s)\s+(\w+)\s+(\w+)[?]?$",
         lambda m: [
             f"What is the {m.group(2)} of {m.group(1)}?",
             f"Where is #1 {m.group(3)}?"
         ]),
    ]

    for pat, builder in patterns:
        m = re.match(pat, q, re.IGNORECASE)
        if m:
            try:
                return builder(m)
            except Exception:
                pass

    # Fallback: no decomposition
    return [question]


def pipeline_embedding_only(sample, retriever, auto_hops, stats):
    """Pure embedding pipeline. Zero LLM calls."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    question = sample["question"]

    # Try template decomposition
    template_hops = template_decompose(question)

    if len(template_hops) >= 2:
        stats["template_decomposed"] += 1
        hops = [{"question": q} for q in template_hops]
    else:
        stats["no_decomp"] += 1
        hops = [{"question": question}]

    previous_answers = []
    hop_details = []

    for i, hop in enumerate(hops):
        hop_q = _prepare_hop_question(hop, previous_answers)

        # Retrieve
        context, results, retrieved_indices = _retrieve_and_format(hop_q, paragraphs, retriever)

        # Extract candidates via NER
        candidates = extract_noun_phrases(context)

        if candidates:
            # Score against question
            scored = retriever.score_candidates(hop_q, candidates)
            scored.sort(key=lambda x: x[1], reverse=True)
            answer = scored[0][0]
        else:
            # Use paragraph title as answer
            top_idx = results[0][0] if results else 0
            answer = paragraphs[top_idx]["title"]

        # For multi-hop: build hop detail against gold if available
        if i < len(gold_decomp):
            hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        else:
            hop_details.append({
                "hop": i + 1, "question": hop_q,
                "gold_answer": sample["answer"] if i == len(hops) - 1 else "N/A",
                "predicted": answer, "em": None, "gold_retrieved": False,
            })
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 6: Two-Model Split ────────────────────────────────────────
# Qwen does decomposition (where it's strong).
# NER + BGE does extraction (where generative LLM is weak).
# Best of both worlds.

def pipeline_two_model_split(sample, retriever, auto_hops, stats):
    """Qwen decomposes, NER+BGE extracts. LLM for decomposition only."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        context, results, retrieved_indices = _retrieve_and_format(hop_q, paragraphs, retriever)

        # Extract candidates via NER
        candidates = extract_noun_phrases(context)
        stats["total_candidates"] += len(candidates)

        if candidates:
            # Score each candidate against the question
            scored = retriever.score_candidates(hop_q, candidates)
            scored.sort(key=lambda x: x[1], reverse=True)
            answer = scored[0][0]
            stats["ner_extracted"] += 1
        else:
            # Rare fallback: use generative
            prompt = PROMPT_8SHOT.format(context=context, question=hop_q)
            response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
            answer = extract_short_answer(response)
            stats["llm_fallback"] += 1

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ══════════════════════════════════════════════════════════════════════
# CONFIG REGISTRY
# ══════════════════════════════════════════════════════════════════════

CONFIGS = [
    {"id": 0, "name": "baseline",
     "fn": pipeline_baseline,
     "description": "Standard Decompose→Retrieve→Extract (generative)"},
    {"id": 1, "name": "extractive_span",
     "fn": pipeline_extractive_span,
     "description": "NER + BGE similarity, zero LLM extraction"},
    {"id": 2, "name": "no_decomposition",
     "fn": pipeline_no_decomposition,
     "description": "Skip decomposition, retrieve for full question"},
    {"id": 3, "name": "reverse_pipeline",
     "fn": pipeline_reverse,
     "description": "Retrieve→Extract candidates→Decompose to verify"},
    {"id": 4, "name": "pairwise_ranking",
     "fn": pipeline_pairwise_ranking,
     "description": "Decompose→Retrieve→Multiple-choice LLM selection"},
    {"id": 5, "name": "embedding_only",
     "fn": pipeline_embedding_only,
     "description": "Pure BGE, zero LLM calls anywhere"},
    {"id": 6, "name": "two_model_split",
     "fn": pipeline_two_model_split,
     "description": "Qwen decomposes, NER+BGE extracts"},
]


# ══════════════════════════════════════════════════════════════════════
# TEST RUNNER
# ══════════════════════════════════════════════════════════════════════

def test_one_config(config, samples, retriever, all_auto_decomps):
    fn = config["fn"]
    em_scores, rem_scores, f1_scores, latencies = [], [], [], []
    per_question = []
    stats = defaultdict(int)

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
        "stats": dict(stats),
    }


def _save(output_path, results, n_configs, n_samples):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "metadata": {
            "experiment": "v9c_fundamentals",
            "n_configs": n_configs,
            "n_completed": len(results),
            "n_samples": n_samples,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "configs": results,
    }
    with open(output, "w") as f:
        json.dump(data, f, indent=2, default=str)


def run_experiment(limit=30, output_path="results/v9c_fundamentals.json",
                   resume=True, configs_to_run=None):
    from datasets import load_dataset

    print("Loading MuSiQue validation set...")
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    samples = [s for s in ds if s.get("answerable", True)][:limit]
    print(f"Testing on {len(samples)} answerable questions\n")

    retriever = EmbeddingRetriever()
    retriever._load_model()

    # Phase 1: Decomposition (needed by configs that use it)
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

    # Load previous
    output = Path(output_path)
    existing_results = {}
    if resume and output.exists():
        with open(output) as f:
            data = json.load(f)
        existing_results = data.get("configs", {})
        print(f"  Resuming: {len(existing_results)} configs already completed")

    # Phase 2: Run
    configs = CONFIGS
    if configs_to_run is not None:
        configs = [c for c in CONFIGS if c["id"] in configs_to_run]

    print(f"\n{'=' * 70}")
    print(f"PHASE 2: TESTING {len(configs)} FUNDAMENTAL APPROACHES")
    print("=" * 70)

    baseline_em = None

    for config in configs:
        key = f"{config['id']}_{config['name']}"

        if key in existing_results:
            result = existing_results[key]
            if config["id"] == 0:
                baseline_em = result["em"]
            print(f"  [cached] Config {config['id']}: {config['name']:30s} EM={result['em']:5.1f}%")
            continue

        print(f"\n  Running Config {config['id']}: {config['name']}...")
        print(f"  {config['description']}")

        t0 = time.time()
        try:
            result = test_one_config(config, samples, retriever, all_auto_decomps)
        except Exception as e:
            result = {"em": 0, "relaxed_em": 0, "f1": 0, "latency_ms": 0,
                     "error": str(e), "per_question": [], "stats": {}}
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

        elapsed = time.time() - t0
        existing_results[key] = result

        if config["id"] == 0:
            baseline_em = result["em"]

        delta = f" ({result['em'] - baseline_em:+.1f}%)" if baseline_em is not None and config["id"] != 0 else ""
        ss = result.get("stats", {})
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
    print(f"FINAL SUMMARY: v9c Fundamentally Different Approaches")
    print(f"{'=' * 70}")

    if baseline_em is None:
        baseline_em = existing_results.get("0_baseline", {}).get("em", 60.0)

    print(f"\nBaseline EM: {baseline_em:.1f}%\n")
    print(f"{'Cfg':<4} {'Name':<30} {'EM':>6} {'Delta':>7} {'relEM':>6} "
          f"{'F1':>6} {'Lat':>7} {'LLM calls':>10}")
    print("-" * 85)

    llm_calls = {
        0: "2/sample",
        1: "0 extract",
        2: "1/sample",
        3: "~1/sample",
        4: "2+1/hop",
        5: "0 total",
        6: "decomp only",
    }

    for config in CONFIGS:
        key = f"{config['id']}_{config['name']}"
        r = existing_results.get(key)
        if not r:
            continue
        delta = r["em"] - baseline_em
        marker = "+" if delta > 0.5 else ("-" if delta < -0.5 else "=")
        lc = llm_calls.get(config["id"], "?")
        print(f"[{marker}] {config['id']:<2} {config['name']:<30} {r['em']:5.1f}% "
              f"{delta:+5.1f}%  {r['relaxed_em']:5.1f}% {r['f1']:.3f} "
              f"{r['latency_ms']:6.0f}ms  {lc:>10}")

    # Per-question diff
    print(f"\n{'=' * 70}")
    print("PER-QUESTION DIFF vs BASELINE")
    print("=" * 70)

    base_pq = {q["id"]: q for q in existing_results.get("0_baseline", {}).get("per_question", [])}
    for config in CONFIGS:
        if config["id"] == 0:
            continue
        key = f"{config['id']}_{config['name']}"
        r = existing_results.get(key, {})
        fixed, broke, diff = [], [], []
        for q in r.get("per_question", []):
            bq = base_pq.get(q["id"])
            if bq and (q["em"] != bq["em"] or q["prediction"] != bq["prediction"]):
                if q["em"] and not bq["em"]:
                    fixed.append((q["id"], bq["prediction"], q["prediction"], q["answer"]))
                elif not q["em"] and bq["em"]:
                    broke.append((q["id"], bq["prediction"], q["prediction"], q["answer"]))
                elif q["prediction"] != bq["prediction"]:
                    diff.append((q["id"], bq["prediction"], q["prediction"], q["answer"]))

        delta = r.get("em", 0) - baseline_em
        print(f"\n  Config {config['id']} ({config['name']}) [{delta:+.1f}%]: "
              f"fixed={len(fixed)} broke={len(broke)} diff={len(diff)}")
        for qid, old, new, gold in fixed:
            print(f"    [FIXED] '{old}' → '{new}'  (gold: '{gold}')")
        for qid, old, new, gold in broke:
            print(f"    [BROKE] '{old}' → '{new}'  (gold: '{gold}')")


def main():
    parser = argparse.ArgumentParser(description="v9c: Fundamentally Different Approaches")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", type=str, default="results/v9c_fundamentals.json")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--configs", type=str, default=None)
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
