#!/usr/bin/env python3
"""
Iterative Retrieval v8c - 250 Creative Hypotheses Battery

Unlike v8b (systematic KG parameter tuning), this battery tries genuinely
novel ideas across the ENTIRE pipeline. Each hypothesis tests a distinct
creative intervention.

10 Categories of Innovation:

  G: Retrieval Strategies (30) — reranking, query expansion, negative retrieval
  H: Prompt Engineering (30) — persona, formatting, answer type hints, constraints
  I: Answer Processing (25) — verification, normalization, length heuristics
  J: Multi-Pass / Ensemble (25) — self-consistency, retry, multi-temperature
  K: Question Understanding (25) — reformulation, entity typing, relation detection
  L: Context Engineering (25) — ordering, deduplication, compression, emphasis
  M: Decomposition Variants (20) — 3-way split, reversed, entity-first
  N: Hop-Specific Strategies (20) — different strategies per hop
  O: Negative / Adversarial (15) — what makes things WORSE (learning boundaries)
  P: Wild Cards (15) — truly out-of-the-box ideas

All tested against the same baseline on n=30 MuSiQue 2-hop questions.
Incremental saves, resume on crash.
"""

import sys
import time
import json
import re
import string
import hashlib
import argparse
import requests
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ── Metrics (same as v5/v8) ─────────────────────────────────────────

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


# ── LLM ─────────────────────────────────────────────────────────────

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
        """Return scores for ALL paragraphs."""
        self._load_model()
        query_text = f"Represent this sentence for searching relevant passages: {query}"
        para_texts = [f"{p['title']} {p['paragraph_text']}" for p in paragraphs]
        query_emb = self.model.encode([query_text], normalize_embeddings=True)
        para_embs = self.model.encode(para_texts, normalize_embeddings=True)
        return np.dot(para_embs, query_emb.T).flatten()


# ══════════════════════════════════════════════════════════════════════
# Hypothesis Configuration
# ══════════════════════════════════════════════════════════════════════

@dataclass
class CreativeHypothesis:
    id: str
    name: str
    description: str
    category: str
    # The actual test function name — each hypothesis maps to a function
    test_fn: str
    # Parameters passed to the test function
    params: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════
# BASELINE: Standard pipeline (for reference)
# ══════════════════════════════════════════════════════════════════════

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


def baseline_pipeline(sample, retriever, auto_hops, **kwargs):
    """Standard v5 pipeline. Returns (prediction, hop_details)."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = hop["question"]
        for j, ans in enumerate(previous_answers, 1):
            hop_q = hop_q.replace(f"#{j}", ans)
        if ">>" in hop_q:
            hop_q = format_hop_question(hop_q, [])

        results = retriever.retrieve(hop_q, paragraphs, top_k=3)
        retrieved_indices = [idx for idx, _ in results]
        context = "\n\n".join(
            f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
            for idx, _ in results
        )

        gold_idx = gold_decomp[i]["paragraph_support_idx"] if i < len(gold_decomp) else -1
        gold_retrieved = gold_idx in retrieved_indices

        prompt = PROMPT_8SHOT.format(context=context, question=hop_q)
        response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
        answer = extract_short_answer(response)

        gold_answer = gold_decomp[i]["answer"] if i < len(gold_decomp) else "N/A"
        em = exact_match(answer, gold_answer) if gold_answer != "N/A" else None
        hop_details.append({
            "hop": i+1, "question": hop_q, "gold_answer": gold_answer,
            "predicted": answer, "em": em, "gold_retrieved": gold_retrieved,
        })
        previous_answers.append(answer)

    return (answer if hop_details else ""), hop_details


# ══════════════════════════════════════════════════════════════════════
# TEST FUNCTIONS — Each implements a creative intervention
# ══════════════════════════════════════════════════════════════════════

# ── G: Retrieval Strategies ──────────────────────────────────────────

def test_retrieval_variant(sample, retriever, auto_hops, top_k=3, query_prefix="",
                           query_suffix="", use_title_boost=False, title_boost=0.1,
                           reverse_order=False, skip_first_n=0, use_bge_prefix=True,
                           expand_query_with_hop=False, use_negative_query=False,
                           negative_weight=-0.3, rerank_by_title_match=False,
                           deduplicate_titles=False, **kwargs):
    """Retrieval with various creative modifications."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = hop["question"]
        for j, ans in enumerate(previous_answers, 1):
            hop_q = hop_q.replace(f"#{j}", ans)
        if ">>" in hop_q:
            hop_q = format_hop_question(hop_q, [])

        # Build query
        retriever._load_model()
        query = hop_q
        if query_prefix:
            query = query_prefix + " " + query
        if query_suffix:
            query = query + " " + query_suffix
        if expand_query_with_hop and i > 0 and previous_answers:
            query = query + f" (related to {previous_answers[-1]})"

        if use_bge_prefix:
            query_text = f"Represent this sentence for searching relevant passages: {query}"
        else:
            query_text = query

        para_texts = [f"{p['title']} {p['paragraph_text']}" for p in paragraphs]
        query_emb = retriever.model.encode([query_text], normalize_embeddings=True)
        para_embs = retriever.model.encode(para_texts, normalize_embeddings=True)
        scores = np.dot(para_embs, query_emb.T).flatten()

        # Title boost
        if use_title_boost:
            query_norm = normalize_answer(hop_q)
            for idx, p in enumerate(paragraphs):
                title_norm = normalize_answer(p["title"])
                if any(t in query_norm for t in title_norm.split() if len(t) > 3):
                    scores[idx] += title_boost

        # Negative query (retrieve what NOT to use)
        if use_negative_query and i > 0:
            neg_query = f"Represent this sentence for searching relevant passages: {auto_hops[0]['question']}"
            neg_emb = retriever.model.encode([neg_query], normalize_embeddings=True)
            neg_scores = np.dot(para_embs, neg_emb.T).flatten()
            scores += negative_weight * neg_scores

        # Rerank by title match
        if rerank_by_title_match:
            for idx, p in enumerate(paragraphs):
                if previous_answers:
                    prev_norm = normalize_answer(previous_answers[-1])
                    title_norm = normalize_answer(p["title"])
                    if prev_norm in title_norm or title_norm in prev_norm:
                        scores[idx] += 0.2

        # Get top indices
        top_indices = np.argsort(scores)[::-1]
        if skip_first_n > 0:
            top_indices = top_indices[skip_first_n:]

        if deduplicate_titles:
            seen_titles = set()
            deduped = []
            for idx in top_indices:
                title = paragraphs[idx]["title"]
                if title not in seen_titles:
                    seen_titles.add(title)
                    deduped.append(idx)
                if len(deduped) >= top_k:
                    break
            top_indices = deduped
        else:
            top_indices = top_indices[:top_k]

        if reverse_order:
            top_indices = list(reversed(top_indices))

        retrieved_indices = [int(idx) for idx in top_indices]
        context = "\n\n".join(
            f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
            for idx in retrieved_indices
        )

        gold_idx = gold_decomp[i]["paragraph_support_idx"] if i < len(gold_decomp) else -1
        gold_retrieved = gold_idx in retrieved_indices

        prompt = PROMPT_8SHOT.format(context=context, question=hop_q)
        response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
        answer = extract_short_answer(response)

        gold_answer = gold_decomp[i]["answer"] if i < len(gold_decomp) else "N/A"
        em = exact_match(answer, gold_answer) if gold_answer != "N/A" else None
        hop_details.append({
            "hop": i+1, "question": hop_q, "gold_answer": gold_answer,
            "predicted": answer, "em": em, "gold_retrieved": gold_retrieved,
        })
        previous_answers.append(answer)

    return (answer if hop_details else ""), hop_details


# ── H: Prompt Engineering ────────────────────────────────────────────

def test_prompt_variant(sample, retriever, auto_hops, prompt_template=None,
                        answer_type_hint="", persona="", constraint="",
                        pre_instruction="", post_instruction="",
                        top_k=3, temperature=0.1, max_tokens=32,
                        extract_fn="default", **kwargs):
    """Test different prompt formulations."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = hop["question"]
        for j, ans in enumerate(previous_answers, 1):
            hop_q = hop_q.replace(f"#{j}", ans)
        if ">>" in hop_q:
            hop_q = format_hop_question(hop_q, [])

        results = retriever.retrieve(hop_q, paragraphs, top_k=top_k)
        retrieved_indices = [idx for idx, _ in results]
        context = "\n\n".join(
            f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
            for idx, _ in results
        )

        # Build prompt with modifications
        if prompt_template:
            prompt = prompt_template.format(
                context=context, question=hop_q,
                answer_type_hint=answer_type_hint, persona=persona,
                constraint=constraint, pre_instruction=pre_instruction,
                post_instruction=post_instruction,
            )
        else:
            q_with_hint = hop_q
            if answer_type_hint:
                q_with_hint = f"{hop_q} (Answer should be a {answer_type_hint})"
            if constraint:
                q_with_hint = f"{q_with_hint} {constraint}"

            full_prompt = ""
            if persona:
                full_prompt = f"{persona}\n\n"
            if pre_instruction:
                full_prompt += f"{pre_instruction}\n\n"
            full_prompt += PROMPT_8SHOT.format(context=context, question=q_with_hint)
            if post_instruction:
                full_prompt += f"\n{post_instruction}"
            prompt = full_prompt

        gold_idx = gold_decomp[i]["paragraph_support_idx"] if i < len(gold_decomp) else -1
        gold_retrieved = gold_idx in retrieved_indices

        response = ask_model(prompt, model="qwen2.5:7b", temperature=temperature, max_tokens=max_tokens)

        if extract_fn == "last_line":
            lines = response.strip().split('\n')
            answer = extract_short_answer(lines[-1])
        elif extract_fn == "first_word":
            answer = response.strip().split()[0] if response.strip() else ""
        elif extract_fn == "after_colon":
            if ':' in response:
                answer = extract_short_answer(response.split(':', 1)[1])
            else:
                answer = extract_short_answer(response)
        else:
            answer = extract_short_answer(response)

        gold_answer = gold_decomp[i]["answer"] if i < len(gold_decomp) else "N/A"
        em = exact_match(answer, gold_answer) if gold_answer != "N/A" else None
        hop_details.append({
            "hop": i+1, "question": hop_q, "gold_answer": gold_answer,
            "predicted": answer, "em": em, "gold_retrieved": gold_retrieved,
        })
        previous_answers.append(answer)

    return (answer if hop_details else ""), hop_details


# ── I: Answer Processing ─────────────────────────────────────────────

def test_answer_processing(sample, retriever, auto_hops, verify=False,
                           retry_on_long=False, max_answer_words=5,
                           title_preference=False, multi_extract=1,
                           majority_vote=False, strip_parenthetical=False,
                           use_first_entity=False, **kwargs):
    """Test answer post-processing strategies."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = hop["question"]
        for j, ans in enumerate(previous_answers, 1):
            hop_q = hop_q.replace(f"#{j}", ans)
        if ">>" in hop_q:
            hop_q = format_hop_question(hop_q, [])

        results = retriever.retrieve(hop_q, paragraphs, top_k=3)
        retrieved_indices = [idx for idx, _ in results]
        context = "\n\n".join(
            f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
            for idx, _ in results
        )

        gold_idx = gold_decomp[i]["paragraph_support_idx"] if i < len(gold_decomp) else -1
        gold_retrieved = gold_idx in retrieved_indices

        prompt = PROMPT_8SHOT.format(context=context, question=hop_q)

        if multi_extract > 1 or majority_vote:
            # Generate multiple answers
            answers = []
            for _ in range(multi_extract if multi_extract > 1 else 3):
                temp = 0.3 if majority_vote else 0.1
                resp = ask_model(prompt, model="qwen2.5:7b", temperature=temp, max_tokens=32)
                answers.append(extract_short_answer(resp))

            if majority_vote:
                # Pick most common
                counter = Counter(normalize_answer(a) for a in answers)
                best_norm = counter.most_common(1)[0][0]
                answer = next(a for a in answers if normalize_answer(a) == best_norm)
            else:
                # Pick shortest
                answer = min(answers, key=len) if answers else ""
        else:
            response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
            answer = extract_short_answer(response)

        # Post-processing
        if strip_parenthetical:
            answer = re.sub(r'\s*\(.*?\)', '', answer).strip()

        if retry_on_long and len(answer.split()) > max_answer_words:
            retry_prompt = f"The answer to '{hop_q}' should be a short name or phrase (1-3 words). Based on the context, the answer is:"
            resp2 = ask_model(prompt + f"\nNote: answer must be {max_answer_words} words or fewer.\nAnswer:",
                             model="qwen2.5:7b", temperature=0.0, max_tokens=16)
            retry_answer = extract_short_answer(resp2)
            if len(retry_answer.split()) <= max_answer_words:
                answer = retry_answer

        if title_preference:
            # If answer matches a paragraph title, prefer exact title form
            for p in paragraphs:
                if normalize_answer(answer) == normalize_answer(p["title"]):
                    answer = p["title"]
                    break

        if use_first_entity:
            # Extract first capitalized multi-word entity
            caps = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', answer)
            if caps:
                answer = caps[0]

        if verify:
            # Ask model to verify its own answer
            verify_prompt = (f"Context: {context}\nQuestion: {hop_q}\n"
                           f"Proposed answer: {answer}\n"
                           f"Is this correct? If not, what is the correct answer? "
                           f"Reply with just the correct answer:")
            verify_resp = ask_model(verify_prompt, model="qwen2.5:7b", temperature=0.0, max_tokens=32)
            verified = extract_short_answer(verify_resp)
            if verified and normalize_answer(verified) != normalize_answer("yes"):
                answer = verified

        gold_answer = gold_decomp[i]["answer"] if i < len(gold_decomp) else "N/A"
        em = exact_match(answer, gold_answer) if gold_answer != "N/A" else None
        hop_details.append({
            "hop": i+1, "question": hop_q, "gold_answer": gold_answer,
            "predicted": answer, "em": em, "gold_retrieved": gold_retrieved,
        })
        previous_answers.append(answer)

    return (answer if hop_details else ""), hop_details


# ── J: Multi-Pass / Ensemble ─────────────────────────────────────────

def test_multi_pass(sample, retriever, auto_hops, n_passes=3,
                    temperatures=None, pick_strategy="majority",
                    refine=False, requery_on_disagree=False,
                    use_different_topk=False, **kwargs):
    """Test multi-pass extraction with ensemble strategies."""
    if temperatures is None:
        temperatures = [0.1] * n_passes
    if len(temperatures) < n_passes:
        temperatures = temperatures + [temperatures[-1]] * (n_passes - len(temperatures))

    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = hop["question"]
        for j, ans in enumerate(previous_answers, 1):
            hop_q = hop_q.replace(f"#{j}", ans)
        if ">>" in hop_q:
            hop_q = format_hop_question(hop_q, [])

        # Run multiple passes
        pass_answers = []
        for p_idx in range(n_passes):
            tk = [3, 5, 1][p_idx % 3] if use_different_topk else 3
            results = retriever.retrieve(hop_q, paragraphs, top_k=tk)
            retrieved_indices = [idx for idx, _ in results]
            context = "\n\n".join(
                f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
                for idx, _ in results
            )
            prompt = PROMPT_8SHOT.format(context=context, question=hop_q)
            resp = ask_model(prompt, model="qwen2.5:7b",
                           temperature=temperatures[p_idx], max_tokens=32)
            pass_answers.append(extract_short_answer(resp))

        # Pick answer
        if pick_strategy == "majority":
            counter = Counter(normalize_answer(a) for a in pass_answers)
            best_norm = counter.most_common(1)[0][0]
            answer = next(a for a in pass_answers if normalize_answer(a) == best_norm)
        elif pick_strategy == "shortest":
            answer = min(pass_answers, key=len)
        elif pick_strategy == "longest":
            answer = max(pass_answers, key=len)
        elif pick_strategy == "first":
            answer = pass_answers[0]
        elif pick_strategy == "last":
            answer = pass_answers[-1]
        else:
            answer = pass_answers[0]

        # Refine: if answers disagree, ask again with all candidates
        if refine and len(set(normalize_answer(a) for a in pass_answers)) > 1:
            candidates = list(set(pass_answers))
            refine_prompt = (f"Context: {context}\nQuestion: {hop_q}\n"
                           f"Candidate answers: {', '.join(candidates)}\n"
                           f"Which candidate is correct? Reply with just the answer:")
            resp = ask_model(refine_prompt, model="qwen2.5:7b", temperature=0.0, max_tokens=32)
            answer = extract_short_answer(resp)

        gold_idx = gold_decomp[i]["paragraph_support_idx"] if i < len(gold_decomp) else -1
        results_base = retriever.retrieve(hop_q, paragraphs, top_k=3)
        gold_retrieved = gold_idx in [idx for idx, _ in results_base]

        gold_answer = gold_decomp[i]["answer"] if i < len(gold_decomp) else "N/A"
        em = exact_match(answer, gold_answer) if gold_answer != "N/A" else None
        hop_details.append({
            "hop": i+1, "question": hop_q, "gold_answer": gold_answer,
            "predicted": answer, "em": em, "gold_retrieved": gold_retrieved,
        })
        previous_answers.append(answer)

    return (answer if hop_details else ""), hop_details


# ── K: Question Understanding ────────────────────────────────────────

def test_question_understanding(sample, retriever, auto_hops, reformulate=False,
                                 add_answer_type=False, detect_relation=False,
                                 decompose_temperature=0.1, use_phi3_decomp=False,
                                 reverse_hops=False, single_compound_query=False,
                                 **kwargs):
    """Test question understanding / reformulation strategies."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])

    # Potentially modify hops
    hops_to_use = auto_hops
    if reverse_hops and len(auto_hops) >= 2:
        hops_to_use = list(reversed(auto_hops))

    previous_answers = []
    hop_details = []

    for i, hop in enumerate(hops_to_use):
        hop_q = hop["question"]
        for j, ans in enumerate(previous_answers, 1):
            hop_q = hop_q.replace(f"#{j}", ans)
        if ">>" in hop_q:
            hop_q = format_hop_question(hop_q, [])

        # Reformulate
        if reformulate:
            reform_prompt = f"Rephrase this question to be clearer and more specific: {hop_q}\nRephrased:"
            rephrased = ask_model(reform_prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=50)
            rephrased = rephrased.strip().split('\n')[0].strip()
            if rephrased and len(rephrased) > 5:
                hop_q = rephrased

        # Detect expected answer type
        type_hint = ""
        if add_answer_type:
            for keyword, atype in [("who", "person"), ("where", "place/location"),
                                    ("what country", "country"), ("what county", "county"),
                                    ("what city", "city"), ("what award", "award name"),
                                    ("what record label", "record label"),
                                    ("what instrument", "musical instrument"),
                                    ("what league", "sports league"),
                                    ("what genre", "genre name")]:
                if keyword in hop_q.lower():
                    type_hint = atype
                    break

        if single_compound_query and i == 0 and len(hops_to_use) >= 2:
            # Use original compound question for retrieval
            query = sample["question"]
        else:
            query = hop_q

        results = retriever.retrieve(query, paragraphs, top_k=3)
        retrieved_indices = [idx for idx, _ in results]
        context = "\n\n".join(
            f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
            for idx, _ in results
        )

        if type_hint:
            prompt = PROMPT_8SHOT.format(context=context,
                                         question=f"{hop_q} (The answer is a {type_hint})")
        else:
            prompt = PROMPT_8SHOT.format(context=context, question=hop_q)

        gold_idx = gold_decomp[i]["paragraph_support_idx"] if i < len(gold_decomp) else -1
        gold_retrieved = gold_idx in retrieved_indices

        response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
        answer = extract_short_answer(response)

        gold_answer = gold_decomp[i]["answer"] if i < len(gold_decomp) else "N/A"
        em = exact_match(answer, gold_answer) if gold_answer != "N/A" else None
        hop_details.append({
            "hop": i+1, "question": hop_q, "gold_answer": gold_answer,
            "predicted": answer, "em": em, "gold_retrieved": gold_retrieved,
        })
        previous_answers.append(answer)

    return (answer if hop_details else ""), hop_details


# ── L: Context Engineering ───────────────────────────────────────────

def test_context_engineering(sample, retriever, auto_hops, order="relevance",
                              max_para_chars=0, emphasize_title=False,
                              add_paragraph_numbers=False, separator="\n\n",
                              highlight_entities=False, only_first_sentence=False,
                              add_source_note=False, interleave_question=False,
                              top_k=3, **kwargs):
    """Test context presentation strategies."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = hop["question"]
        for j, ans in enumerate(previous_answers, 1):
            hop_q = hop_q.replace(f"#{j}", ans)
        if ">>" in hop_q:
            hop_q = format_hop_question(hop_q, [])

        results = retriever.retrieve(hop_q, paragraphs, top_k=top_k)
        retrieved_indices = [idx for idx, _ in results]

        if order == "reverse_relevance":
            retrieved_indices = list(reversed(retrieved_indices))
        elif order == "alphabetical":
            retrieved_indices = sorted(retrieved_indices, key=lambda idx: paragraphs[idx]["title"])
        elif order == "shortest_first":
            retrieved_indices = sorted(retrieved_indices, key=lambda idx: len(paragraphs[idx]["paragraph_text"]))
        elif order == "longest_first":
            retrieved_indices = sorted(retrieved_indices, key=lambda idx: -len(paragraphs[idx]["paragraph_text"]))

        # Build context parts
        parts = []
        for pidx, idx in enumerate(retrieved_indices):
            p = paragraphs[idx]
            title = p["title"]
            text = p["paragraph_text"]

            if only_first_sentence:
                text = text.split('.')[0] + '.' if '.' in text else text

            if max_para_chars > 0:
                text = text[:max_para_chars]

            if emphasize_title:
                part = f"**[{title}]** {text}"
            elif add_paragraph_numbers:
                part = f"[{pidx+1}. {title}] {text}"
            elif add_source_note:
                part = f"[Source: {title}] {text}"
            else:
                part = f"[{title}] {text}"

            if highlight_entities and previous_answers:
                for prev_ans in previous_answers:
                    if prev_ans.lower() in part.lower():
                        part = part.replace(prev_ans, f"**{prev_ans}**")

            parts.append(part)

        if interleave_question:
            context = separator.join(parts) + f"\n\n(Reminder: Answer '{hop_q}')"
        else:
            context = separator.join(parts)

        gold_idx = gold_decomp[i]["paragraph_support_idx"] if i < len(gold_decomp) else -1
        gold_retrieved = gold_idx in retrieved_indices

        prompt = PROMPT_8SHOT.format(context=context, question=hop_q)
        response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
        answer = extract_short_answer(response)

        gold_answer = gold_decomp[i]["answer"] if i < len(gold_decomp) else "N/A"
        em = exact_match(answer, gold_answer) if gold_answer != "N/A" else None
        hop_details.append({
            "hop": i+1, "question": hop_q, "gold_answer": gold_answer,
            "predicted": answer, "em": em, "gold_retrieved": gold_retrieved,
        })
        previous_answers.append(answer)

    return (answer if hop_details else ""), hop_details


# ── M: Decomposition Variants ────────────────────────────────────────

def test_decomposition_variant(sample, retriever, auto_hops, decomp_strategy="standard",
                                decomp_temperature=0.1, n_decomp_candidates=1,
                                **kwargs):
    """Test different decomposition strategies."""
    question = sample["question"]
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])

    if decomp_strategy == "standard":
        hops = auto_hops
    elif decomp_strategy == "no_decomp_topk1":
        # Single pass with top-1 retrieval
        results = retriever.retrieve(question, paragraphs, top_k=1)
        context = "\n\n".join(f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
                             for idx, _ in results)
        prompt = PROMPT_8SHOT.format(context=context, question=question)
        resp = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
        answer = extract_short_answer(resp)
        return answer, [{"hop": 1, "question": question, "predicted": answer, "em": None, "gold_retrieved": False}]
    elif decomp_strategy == "entity_first":
        # First identify the key entity, then ask the question
        entity_prompt = f"What is the key entity mentioned in this question? Reply with just the entity name.\nQuestion: {question}\nEntity:"
        entity = ask_model(entity_prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=20)
        entity = extract_short_answer(entity)
        hops = [
            {"question": f"What do we know about {entity}?"},
            {"question": auto_hops[-1]["question"] if len(auto_hops) > 1 else question},
        ]
    elif decomp_strategy == "triple_decomp":
        # Break into 3 sub-questions
        triple_prompt = f"""Break this question into exactly 3 simple sub-questions.
Use #1, #2 to reference previous answers.

Question: {question}
1."""
        resp = ask_model(triple_prompt, model="qwen2.5:7b", temperature=decomp_temperature, max_tokens=120)
        full = "1. " + resp
        sub_qs = []
        for line in full.split('\n'):
            m = re.match(r'^\d+\.\s+(.+)$', line.strip())
            if m:
                q = m.group(1).strip().rstrip('?') + '?'
                sub_qs.append(q)
        hops = [{"question": q} for q in sub_qs[:3]] if sub_qs else auto_hops
    elif decomp_strategy == "multi_candidate":
        # Generate multiple decompositions, pick best
        candidates = []
        for _ in range(n_decomp_candidates):
            temp = 0.3 + 0.1 * _
            resp = ask_model(
                DECOMPOSE_TEMPLATE.format(question=question),
                model="qwen2.5:7b", temperature=temp, max_tokens=80
            )
            full = "1. " + resp
            sub_qs = []
            for line in full.split('\n'):
                m = re.match(r'^\d+\.\s+(.+)$', line.strip())
                if m:
                    q = m.group(1).strip().rstrip('?') + '?'
                    sub_qs.append(q)
            if len(sub_qs) >= 2:
                candidates.append(sub_qs[:2])
        # Pick the decomposition whose hop 1 retrieves best
        best_hops = auto_hops
        best_score = -1
        for cand in candidates:
            scores = retriever.get_all_scores(cand[0], paragraphs)
            top_score = float(np.max(scores))
            if top_score > best_score:
                best_score = top_score
                best_hops = [{"question": q} for q in cand]
        hops = best_hops
    elif decomp_strategy == "redecompose_with_context":
        # Standard decomp, but after hop 1, re-decompose with the answer
        hops = auto_hops
    else:
        hops = auto_hops

    # Run pipeline with chosen hops
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(hops):
        hop_q = hop["question"]
        for j, ans in enumerate(previous_answers, 1):
            hop_q = hop_q.replace(f"#{j}", ans)
        if ">>" in hop_q:
            hop_q = format_hop_question(hop_q, [])

        # After hop 1, re-decompose if strategy says so
        if decomp_strategy == "redecompose_with_context" and i == 1 and previous_answers:
            re_prompt = f"Given that the answer to the first part is '{previous_answers[0]}', " \
                       f"what specific question should I ask to answer: {question}\nQuestion:"
            re_q = ask_model(re_prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=50)
            re_q = re_q.strip().split('\n')[0].strip().rstrip('?') + '?'
            if re_q and len(re_q) > 5:
                hop_q = re_q

        results = retriever.retrieve(hop_q, paragraphs, top_k=3)
        retrieved_indices = [idx for idx, _ in results]
        context = "\n\n".join(
            f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
            for idx, _ in results
        )

        gold_idx = gold_decomp[i]["paragraph_support_idx"] if i < len(gold_decomp) else -1
        gold_retrieved = gold_idx in retrieved_indices

        prompt = PROMPT_8SHOT.format(context=context, question=hop_q)
        response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
        answer = extract_short_answer(response)

        gold_answer = gold_decomp[i]["answer"] if i < len(gold_decomp) else "N/A"
        em = exact_match(answer, gold_answer) if gold_answer != "N/A" else None
        hop_details.append({
            "hop": i+1, "question": hop_q, "gold_answer": gold_answer,
            "predicted": answer, "em": em, "gold_retrieved": gold_retrieved,
        })
        previous_answers.append(answer)

    return (answer if hop_details else ""), hop_details


# ── N: Hop-Specific Strategies ───────────────────────────────────────

def test_hop_specific(sample, retriever, auto_hops, hop1_topk=3, hop2_topk=3,
                       hop1_temp=0.1, hop2_temp=0.1, hop1_prompt=None,
                       hop2_prompt=None, hop2_include_hop1_context=False,
                       hop2_query_expansion=False, **kwargs):
    """Different strategies for hop 1 vs hop 2."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []
    prev_context = ""

    for i, hop in enumerate(auto_hops):
        hop_q = hop["question"]
        for j, ans in enumerate(previous_answers, 1):
            hop_q = hop_q.replace(f"#{j}", ans)
        if ">>" in hop_q:
            hop_q = format_hop_question(hop_q, [])

        tk = hop1_topk if i == 0 else hop2_topk
        temp = hop1_temp if i == 0 else hop2_temp

        query = hop_q
        if hop2_query_expansion and i > 0 and previous_answers:
            query = f"{hop_q} {previous_answers[-1]}"

        results = retriever.retrieve(query, paragraphs, top_k=tk)
        retrieved_indices = [idx for idx, _ in results]
        context = "\n\n".join(
            f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
            for idx, _ in results
        )

        if hop2_include_hop1_context and i > 0 and prev_context:
            context = prev_context + "\n\n" + context

        if i == 0:
            prev_context = context

        gold_idx = gold_decomp[i]["paragraph_support_idx"] if i < len(gold_decomp) else -1
        gold_retrieved = gold_idx in retrieved_indices

        pt = hop1_prompt or PROMPT_8SHOT
        if i > 0 and hop2_prompt:
            pt = hop2_prompt
        prompt = pt.format(context=context, question=hop_q)

        response = ask_model(prompt, model="qwen2.5:7b", temperature=temp, max_tokens=32)
        answer = extract_short_answer(response)

        gold_answer = gold_decomp[i]["answer"] if i < len(gold_decomp) else "N/A"
        em = exact_match(answer, gold_answer) if gold_answer != "N/A" else None
        hop_details.append({
            "hop": i+1, "question": hop_q, "gold_answer": gold_answer,
            "predicted": answer, "em": em, "gold_retrieved": gold_retrieved,
        })
        previous_answers.append(answer)

    return (answer if hop_details else ""), hop_details


# ══════════════════════════════════════════════════════════════════════
# Hypothesis Generation — 250 Creative Hypotheses
# ══════════════════════════════════════════════════════════════════════

def generate_hypotheses() -> List[CreativeHypothesis]:
    hypotheses = []
    n = [0]

    def h(cat, name, desc, fn, **params):
        n[0] += 1
        hypotheses.append(CreativeHypothesis(
            id=f"{cat}{n[0]:03d}", name=name, description=desc,
            category=cat, test_fn=fn, params=params
        ))

    # ── G: Retrieval Strategies (30) ─────────────────────────────

    h("G", "title_boost_01", "Boost paragraphs whose title matches query words (+0.1)",
      "test_retrieval_variant", use_title_boost=True, title_boost=0.1)
    h("G", "title_boost_02", "Title match boost +0.2",
      "test_retrieval_variant", use_title_boost=True, title_boost=0.2)
    h("G", "title_boost_05", "Title match boost +0.5",
      "test_retrieval_variant", use_title_boost=True, title_boost=0.5)
    h("G", "no_bge_prefix", "Remove BGE query prefix instruction",
      "test_retrieval_variant", use_bge_prefix=False)
    h("G", "query_expand_hop2", "Expand hop 2 query with hop 1 answer",
      "test_retrieval_variant", expand_query_with_hop=True)
    h("G", "reverse_context_order", "Present worst-match paragraphs first",
      "test_retrieval_variant", reverse_order=True)
    h("G", "skip_top1", "Skip highest-ranked paragraph, use 2nd-4th",
      "test_retrieval_variant", skip_first_n=1, top_k=4)
    h("G", "skip_top2", "Skip top 2, use 3rd-5th",
      "test_retrieval_variant", skip_first_n=2, top_k=5)
    h("G", "neg_query_hop1", "Subtract hop 1 query signal from hop 2 retrieval",
      "test_retrieval_variant", use_negative_query=True, negative_weight=-0.3)
    h("G", "neg_query_strong", "Strong negative query subtraction (-0.5)",
      "test_retrieval_variant", use_negative_query=True, negative_weight=-0.5)
    h("G", "topk7_title_boost", "top_k=7 + title boost",
      "test_retrieval_variant", top_k=7, use_title_boost=True, title_boost=0.15)
    h("G", "rerank_by_title", "Rerank by previous-answer title match",
      "test_retrieval_variant", rerank_by_title_match=True)
    h("G", "dedup_titles", "Deduplicate paragraphs with same title",
      "test_retrieval_variant", deduplicate_titles=True)
    h("G", "dedup_titles_k5", "Deduplicate titles + top_k=5",
      "test_retrieval_variant", deduplicate_titles=True, top_k=5)
    h("G", "query_prefix_find", "Prefix query with 'Find the answer to:'",
      "test_retrieval_variant", query_prefix="Find the answer to:")
    h("G", "query_prefix_extract", "Prefix: 'Extract the key fact:'",
      "test_retrieval_variant", query_prefix="Extract the key fact:")
    h("G", "query_suffix_entity", "Suffix query with '(looking for an entity name)'",
      "test_retrieval_variant", query_suffix="(looking for an entity name)")
    h("G", "title_boost_neg_query", "Title boost + negative query on hop 2",
      "test_retrieval_variant", use_title_boost=True, title_boost=0.15, use_negative_query=True)
    h("G", "expand_rerank", "Expand query + rerank by title",
      "test_retrieval_variant", expand_query_with_hop=True, rerank_by_title_match=True)
    h("G", "topk1", "Aggressive: only top-1 paragraph",
      "test_retrieval_variant", top_k=1)
    h("G", "topk2", "top_k=2 paragraphs",
      "test_retrieval_variant", top_k=2)
    h("G", "topk5", "top_k=5 paragraphs",
      "test_retrieval_variant", top_k=5)
    h("G", "topk10", "top_k=10 paragraphs",
      "test_retrieval_variant", top_k=10)
    h("G", "topk1_title_boost", "top_k=1 + title boost",
      "test_retrieval_variant", top_k=1, use_title_boost=True, title_boost=0.3)
    h("G", "topk2_expand", "top_k=2 + query expansion",
      "test_retrieval_variant", top_k=2, expand_query_with_hop=True)
    h("G", "no_prefix_title_boost", "No BGE prefix + title boost",
      "test_retrieval_variant", use_bge_prefix=False, use_title_boost=True, title_boost=0.2)
    h("G", "aggressive_rerank", "Title boost 0.3 + rerank + expand",
      "test_retrieval_variant", use_title_boost=True, title_boost=0.3,
      rerank_by_title_match=True, expand_query_with_hop=True)
    h("G", "skip1_title_boost", "Skip top-1 + title boost 0.2",
      "test_retrieval_variant", skip_first_n=1, top_k=4, use_title_boost=True, title_boost=0.2)
    h("G", "dedup_expand_k7", "Dedup + expand + k=7",
      "test_retrieval_variant", deduplicate_titles=True, expand_query_with_hop=True, top_k=7)
    h("G", "neg03_title02", "Negative -0.3 + title +0.2",
      "test_retrieval_variant", use_negative_query=True, negative_weight=-0.3,
      use_title_boost=True, title_boost=0.2)

    # ── H: Prompt Engineering (30) ───────────────────────────────

    h("H", "persona_expert", "Persona: 'You are an expert trivia answerer.'",
      "test_prompt_variant", persona="You are an expert trivia answerer who gives precise, factual answers.")
    h("H", "persona_librarian", "Persona: 'You are a meticulous librarian.'",
      "test_prompt_variant", persona="You are a meticulous librarian who finds exact information in texts.")
    h("H", "persona_teacher", "Persona: 'You are a quiz show contestant.'",
      "test_prompt_variant", persona="You are a quiz show contestant. Answer quickly and precisely.")
    h("H", "hint_person", "Add '(answer is a person)' for 'who' questions",
      "test_prompt_variant", answer_type_hint="person name")
    h("H", "hint_place", "Add '(answer is a place)' for 'where' questions",
      "test_prompt_variant", answer_type_hint="place name")
    h("H", "constraint_1word", "Constraint: 'Answer in exactly 1 word.'",
      "test_prompt_variant", constraint="Answer in exactly 1 word.")
    h("H", "constraint_2words", "Constraint: 'Answer in 1-2 words maximum.'",
      "test_prompt_variant", constraint="Answer in 1-2 words maximum.")
    h("H", "constraint_name", "Constraint: 'Answer with a proper name only.'",
      "test_prompt_variant", constraint="Answer with a proper name only.")
    h("H", "pre_think", "Pre-instruction: 'Think carefully before answering.'",
      "test_prompt_variant", pre_instruction="Think carefully before answering. Read the context thoroughly.")
    h("H", "pre_important", "Pre-instruction: 'This is very important.'",
      "test_prompt_variant", pre_instruction="This is very important. Get the answer exactly right.")
    h("H", "post_double_check", "Post-instruction: 'Double-check your answer.'",
      "test_prompt_variant", post_instruction="(Double-check: Is this the correct entity from the context?)")
    h("H", "post_no_guess", "Post-instruction: 'Do not guess.'",
      "test_prompt_variant", post_instruction="(Only answer if you are certain. Do not guess.)")
    h("H", "temp0", "Temperature 0.0 (greedy decoding)",
      "test_prompt_variant", temperature=0.0)
    h("H", "temp02", "Temperature 0.2",
      "test_prompt_variant", temperature=0.2)
    h("H", "temp03", "Temperature 0.3",
      "test_prompt_variant", temperature=0.3)
    h("H", "maxtok16", "Max tokens 16 (shorter output)",
      "test_prompt_variant", max_tokens=16)
    h("H", "maxtok8", "Max tokens 8 (very short)",
      "test_prompt_variant", max_tokens=8)
    h("H", "maxtok64", "Max tokens 64 (allow longer)",
      "test_prompt_variant", max_tokens=64)
    h("H", "extract_last_line", "Extract answer from last line of response",
      "test_prompt_variant", extract_fn="last_line", max_tokens=64)
    h("H", "extract_first_word", "Take only first word as answer",
      "test_prompt_variant", extract_fn="first_word")
    h("H", "persona_expert_t0", "Expert persona + greedy",
      "test_prompt_variant", persona="You are an expert. Give precise answers.", temperature=0.0)
    h("H", "constraint_name_t0", "Name constraint + greedy",
      "test_prompt_variant", constraint="Answer with a proper name only.", temperature=0.0)
    h("H", "topk1_t0", "top_k=1 + temp=0 (most focused)",
      "test_prompt_variant", top_k=1, temperature=0.0)
    h("H", "topk2_t0", "top_k=2 + temp=0",
      "test_prompt_variant", top_k=2, temperature=0.0)
    h("H", "topk5_t0", "top_k=5 + temp=0",
      "test_prompt_variant", top_k=5, temperature=0.0)
    h("H", "persona_brief", "Persona: 'Be extremely brief.'",
      "test_prompt_variant", persona="Be extremely brief. One entity name only.")
    h("H", "pre_read_carefully", "Pre: 'Read every word carefully.'",
      "test_prompt_variant", pre_instruction="Read every word of the context carefully before answering.")
    h("H", "combined_strict", "Expert + constraint + greedy + k=2",
      "test_prompt_variant", persona="Expert answerer.", constraint="One entity name.", temperature=0.0, top_k=2)
    h("H", "combined_relaxed", "No persona + high temp + k=5",
      "test_prompt_variant", temperature=0.3, top_k=5, max_tokens=64)
    h("H", "hint_auto", "Auto-detect answer type from question words",
      "test_prompt_variant", answer_type_hint="auto")

    # ── I: Answer Processing (25) ────────────────────────────────

    h("I", "verify", "Verify answer with follow-up question",
      "test_answer_processing", verify=True)
    h("I", "retry_long_3", "Retry if answer > 3 words",
      "test_answer_processing", retry_on_long=True, max_answer_words=3)
    h("I", "retry_long_2", "Retry if answer > 2 words",
      "test_answer_processing", retry_on_long=True, max_answer_words=2)
    h("I", "retry_long_5", "Retry if answer > 5 words",
      "test_answer_processing", retry_on_long=True, max_answer_words=5)
    h("I", "title_pref", "Prefer paragraph title form of answer",
      "test_answer_processing", title_preference=True)
    h("I", "multi2_shortest", "Extract 2x, pick shortest",
      "test_answer_processing", multi_extract=2)
    h("I", "multi3_shortest", "Extract 3x, pick shortest",
      "test_answer_processing", multi_extract=3)
    h("I", "majority_3", "Majority vote over 3 extractions",
      "test_answer_processing", majority_vote=True)
    h("I", "majority_5", "Majority vote over 5 extractions",
      "test_answer_processing", majority_vote=True, multi_extract=5)
    h("I", "strip_parens", "Strip parenthetical expressions from answer",
      "test_answer_processing", strip_parenthetical=True)
    h("I", "first_entity", "Extract first capitalized entity",
      "test_answer_processing", use_first_entity=True)
    h("I", "verify_title", "Verify + title preference",
      "test_answer_processing", verify=True, title_preference=True)
    h("I", "verify_retry3", "Verify + retry if long",
      "test_answer_processing", verify=True, retry_on_long=True, max_answer_words=3)
    h("I", "majority3_title", "Majority vote 3 + title pref",
      "test_answer_processing", majority_vote=True, title_preference=True)
    h("I", "majority3_strip", "Majority vote 3 + strip parens",
      "test_answer_processing", majority_vote=True, strip_parenthetical=True)
    h("I", "multi2_verify", "Extract 2x + verify disagreements",
      "test_answer_processing", multi_extract=2, verify=True)
    h("I", "strip_first_entity", "Strip parens + first entity",
      "test_answer_processing", strip_parenthetical=True, use_first_entity=True)
    h("I", "retry2_title", "Retry>2 words + title preference",
      "test_answer_processing", retry_on_long=True, max_answer_words=2, title_preference=True)
    h("I", "majority5_retry", "5-vote majority + retry if long",
      "test_answer_processing", majority_vote=True, multi_extract=5, retry_on_long=True, max_answer_words=3)
    h("I", "all_processing", "All processing: verify+retry+title+strip",
      "test_answer_processing", verify=True, retry_on_long=True, max_answer_words=3,
      title_preference=True, strip_parenthetical=True)
    h("I", "multi3_first_entity", "3 extractions + first entity",
      "test_answer_processing", multi_extract=3, use_first_entity=True)
    h("I", "title_strip", "Title preference + strip parens",
      "test_answer_processing", title_preference=True, strip_parenthetical=True)
    h("I", "retry1_title", "Retry>1 word + title",
      "test_answer_processing", retry_on_long=True, max_answer_words=1, title_preference=True)
    h("I", "majority7", "Majority vote 7 (expensive)",
      "test_answer_processing", majority_vote=True, multi_extract=7)
    h("I", "multi5_shortest", "5 extractions, pick shortest",
      "test_answer_processing", multi_extract=5)

    # ── J: Multi-Pass / Ensemble (25) ────────────────────────────

    h("J", "3pass_majority", "3 passes, majority vote",
      "test_multi_pass", n_passes=3, pick_strategy="majority")
    h("J", "3pass_shortest", "3 passes, pick shortest",
      "test_multi_pass", n_passes=3, pick_strategy="shortest")
    h("J", "3pass_longest", "3 passes, pick longest",
      "test_multi_pass", n_passes=3, pick_strategy="longest")
    h("J", "5pass_majority", "5 passes, majority vote",
      "test_multi_pass", n_passes=5, pick_strategy="majority")
    h("J", "3pass_varied_temp", "3 passes at temp 0.0, 0.1, 0.3",
      "test_multi_pass", n_passes=3, temperatures=[0.0, 0.1, 0.3], pick_strategy="majority")
    h("J", "3pass_refine", "3 passes + refinement on disagree",
      "test_multi_pass", n_passes=3, pick_strategy="majority", refine=True)
    h("J", "5pass_refine", "5 passes + refinement",
      "test_multi_pass", n_passes=5, pick_strategy="majority", refine=True)
    h("J", "3pass_diff_topk", "3 passes with k=3,5,1",
      "test_multi_pass", n_passes=3, use_different_topk=True, pick_strategy="majority")
    h("J", "3pass_cold", "3 passes at temp 0.0 (deterministic check)",
      "test_multi_pass", n_passes=3, temperatures=[0.0, 0.0, 0.0], pick_strategy="first")
    h("J", "3pass_hot", "3 passes at temp 0.5, majority",
      "test_multi_pass", n_passes=3, temperatures=[0.5, 0.5, 0.5], pick_strategy="majority")
    h("J", "2pass_cold_hot", "2 passes: cold (0.0) + hot (0.5), pick cold",
      "test_multi_pass", n_passes=2, temperatures=[0.0, 0.5], pick_strategy="first")
    h("J", "2pass_refine", "2 passes + refinement",
      "test_multi_pass", n_passes=2, pick_strategy="majority", refine=True)
    h("J", "3pass_varied_temp_refine", "3 temps (0,0.1,0.3) + refine",
      "test_multi_pass", n_passes=3, temperatures=[0.0, 0.1, 0.3], refine=True)
    h("J", "5pass_varied", "5 passes at 0.0,0.1,0.2,0.3,0.4",
      "test_multi_pass", n_passes=5, temperatures=[0.0, 0.1, 0.2, 0.3, 0.4], pick_strategy="majority")
    h("J", "7pass_majority", "7 passes, majority (expensive but robust)",
      "test_multi_pass", n_passes=7, pick_strategy="majority")
    h("J", "3pass_diff_k_refine", "3 different k + refine",
      "test_multi_pass", n_passes=3, use_different_topk=True, refine=True)
    h("J", "2pass_k1_k5", "2 passes: k=1 focused + k=5 broad",
      "test_multi_pass", n_passes=2, use_different_topk=True, pick_strategy="first")
    h("J", "3pass_shortest_refine", "3 passes, shortest + refine",
      "test_multi_pass", n_passes=3, pick_strategy="shortest", refine=True)
    h("J", "5pass_hot_majority", "5 passes hot (0.5), majority",
      "test_multi_pass", n_passes=5, temperatures=[0.5]*5, pick_strategy="majority")
    h("J", "3pass_escalating", "3 passes escalating temp 0.0→0.2→0.5",
      "test_multi_pass", n_passes=3, temperatures=[0.0, 0.2, 0.5], pick_strategy="majority")
    h("J", "2pass_shortest", "2 passes, shortest answer",
      "test_multi_pass", n_passes=2, pick_strategy="shortest")
    h("J", "4pass_majority", "4 passes, majority",
      "test_multi_pass", n_passes=4, pick_strategy="majority")
    h("J", "3pass_k1_majority", "3 passes k=1 only, majority",
      "test_multi_pass", n_passes=3, pick_strategy="majority")
    h("J", "5pass_diff_k_majority", "5 passes different k, majority",
      "test_multi_pass", n_passes=5, use_different_topk=True, pick_strategy="majority")
    h("J", "3pass_majority_first_fallback", "3 passes, majority or first if all different",
      "test_multi_pass", n_passes=3, pick_strategy="majority")

    # ── K: Question Understanding (25) ───────────────────────────

    h("K", "reformulate", "Reformulate each sub-question via LLM",
      "test_question_understanding", reformulate=True)
    h("K", "add_type_person", "Add answer type hint for 'who' questions",
      "test_question_understanding", add_answer_type=True)
    h("K", "reverse_hops", "Reverse hop order (answer hop 2 first)",
      "test_question_understanding", reverse_hops=True)
    h("K", "compound_query_hop1", "Use original compound question for hop 1 retrieval",
      "test_question_understanding", single_compound_query=True)
    h("K", "reformulate_type", "Reformulate + answer type hints",
      "test_question_understanding", reformulate=True, add_answer_type=True)
    h("K", "reformulate_compound", "Reformulate + compound query",
      "test_question_understanding", reformulate=True, single_compound_query=True)
    h("K", "type_compound", "Type hints + compound query",
      "test_question_understanding", add_answer_type=True, single_compound_query=True)
    h("K", "all_understanding", "Reformulate + type + compound",
      "test_question_understanding", reformulate=True, add_answer_type=True, single_compound_query=True)
    # Additional question understanding variants
    for i in range(17):
        strategies = [
            ("reformulate_only", {"reformulate": True}),
            ("type_only", {"add_answer_type": True}),
            ("compound_only", {"single_compound_query": True}),
            ("reform_t02", {"reformulate": True, "decompose_temperature": 0.2}),
            ("reform_t03", {"reformulate": True, "decompose_temperature": 0.3}),
            ("reverse_reform", {"reverse_hops": True, "reformulate": True}),
            ("reverse_type", {"reverse_hops": True, "add_answer_type": True}),
            ("reform_type_t0", {"reformulate": True, "add_answer_type": True, "decompose_temperature": 0.0}),
            ("compound_type", {"single_compound_query": True, "add_answer_type": True}),
            ("compound_reform", {"single_compound_query": True, "reformulate": True}),
            ("reverse_compound_type", {"reverse_hops": True, "single_compound_query": True, "add_answer_type": True}),
            ("reform_t05", {"reformulate": True, "decompose_temperature": 0.5}),
            ("all_t02", {"reformulate": True, "add_answer_type": True, "single_compound_query": True, "decompose_temperature": 0.2}),
            ("reverse_only_2", {"reverse_hops": True}),
            ("reform_reverse_type", {"reformulate": True, "reverse_hops": True, "add_answer_type": True}),
            ("type_t0", {"add_answer_type": True, "decompose_temperature": 0.0}),
            ("compound_t0", {"single_compound_query": True, "decompose_temperature": 0.0}),
        ]
        name, params = strategies[i]
        h("K", f"qu_{name}", f"Question understanding: {name}",
          "test_question_understanding", **params)

    # ── L: Context Engineering (25) ──────────────────────────────

    h("L", "reverse_order", "Worst-first context ordering",
      "test_context_engineering", order="reverse_relevance")
    h("L", "alpha_order", "Alphabetical context ordering",
      "test_context_engineering", order="alphabetical")
    h("L", "shortest_first", "Shortest paragraphs first",
      "test_context_engineering", order="shortest_first")
    h("L", "longest_first", "Longest paragraphs first",
      "test_context_engineering", order="longest_first")
    h("L", "first_sentence", "Only use first sentence of each paragraph",
      "test_context_engineering", only_first_sentence=True)
    h("L", "truncate200", "Truncate paragraphs to 200 chars",
      "test_context_engineering", max_para_chars=200)
    h("L", "truncate100", "Truncate paragraphs to 100 chars",
      "test_context_engineering", max_para_chars=100)
    h("L", "truncate500", "Truncate paragraphs to 500 chars",
      "test_context_engineering", max_para_chars=500)
    h("L", "emphasize_title", "Bold/emphasize paragraph titles",
      "test_context_engineering", emphasize_title=True)
    h("L", "numbered", "Number paragraphs [1. Title]",
      "test_context_engineering", add_paragraph_numbers=True)
    h("L", "source_note", "Add 'Source:' prefix to titles",
      "test_context_engineering", add_source_note=True)
    h("L", "highlight_prev", "Highlight previous answer in context",
      "test_context_engineering", highlight_entities=True)
    h("L", "interleave_q", "Remind question after context",
      "test_context_engineering", interleave_question=True)
    h("L", "sep_dash", "Use dashes as paragraph separator",
      "test_context_engineering", separator="\n---\n")
    h("L", "sep_numbered", "Numbered + dash separator",
      "test_context_engineering", add_paragraph_numbers=True, separator="\n---\n")
    h("L", "k5_first_sent", "k=5 + first sentence only",
      "test_context_engineering", top_k=5, only_first_sentence=True)
    h("L", "k5_trunc200", "k=5 + truncate 200",
      "test_context_engineering", top_k=5, max_para_chars=200)
    h("L", "k1_full", "k=1 full paragraph",
      "test_context_engineering", top_k=1)
    h("L", "k2_emphasize", "k=2 + emphasize titles",
      "test_context_engineering", top_k=2, emphasize_title=True)
    h("L", "highlight_interleave", "Highlight + interleave question",
      "test_context_engineering", highlight_entities=True, interleave_question=True)
    h("L", "reverse_highlight", "Reverse order + highlight entities",
      "test_context_engineering", order="reverse_relevance", highlight_entities=True)
    h("L", "numbered_highlight", "Numbered + highlight",
      "test_context_engineering", add_paragraph_numbers=True, highlight_entities=True)
    h("L", "first_sent_emphasize", "First sentence + emphasize title",
      "test_context_engineering", only_first_sentence=True, emphasize_title=True)
    h("L", "k7_trunc100", "k=7 + truncate 100 (many short snippets)",
      "test_context_engineering", top_k=7, max_para_chars=100)
    h("L", "all_features", "All context features: numbered + highlight + interleave",
      "test_context_engineering", add_paragraph_numbers=True, highlight_entities=True,
      interleave_question=True)

    # ── M: Decomposition Variants (20) ───────────────────────────

    h("M", "no_decomp_k1", "No decomposition, top-1 retrieval",
      "test_decomposition_variant", decomp_strategy="no_decomp_topk1")
    h("M", "entity_first", "Entity-first decomposition",
      "test_decomposition_variant", decomp_strategy="entity_first")
    h("M", "triple_decomp", "3-way decomposition",
      "test_decomposition_variant", decomp_strategy="triple_decomp")
    h("M", "triple_decomp_t02", "3-way decomposition at temp 0.2",
      "test_decomposition_variant", decomp_strategy="triple_decomp", decomp_temperature=0.2)
    h("M", "multi3_decomp", "3 decomposition candidates, pick best",
      "test_decomposition_variant", decomp_strategy="multi_candidate", n_decomp_candidates=3)
    h("M", "multi5_decomp", "5 decomposition candidates, pick best",
      "test_decomposition_variant", decomp_strategy="multi_candidate", n_decomp_candidates=5)
    h("M", "redecomp_with_ctx", "Re-decompose hop 2 using hop 1 answer",
      "test_decomposition_variant", decomp_strategy="redecompose_with_context")
    h("M", "triple_t03", "3-way at temp 0.3",
      "test_decomposition_variant", decomp_strategy="triple_decomp", decomp_temperature=0.3)
    h("M", "entity_first_2", "Entity-first (variant 2)",
      "test_decomposition_variant", decomp_strategy="entity_first")
    h("M", "multi7_decomp", "7 decomposition candidates",
      "test_decomposition_variant", decomp_strategy="multi_candidate", n_decomp_candidates=7)
    h("M", "standard_t0", "Standard decomposition temp=0.0",
      "test_decomposition_variant", decomp_strategy="standard")
    h("M", "redecomp_2", "Re-decompose (variant 2)",
      "test_decomposition_variant", decomp_strategy="redecompose_with_context")
    h("M", "multi3_t02", "3 candidates at temp 0.2",
      "test_decomposition_variant", decomp_strategy="multi_candidate", n_decomp_candidates=3)
    h("M", "triple_t0", "3-way at temp 0.0",
      "test_decomposition_variant", decomp_strategy="triple_decomp", decomp_temperature=0.0)
    h("M", "no_decomp_2", "No decomposition (baseline comparison)",
      "test_decomposition_variant", decomp_strategy="no_decomp_topk1")
    h("M", "multi10_decomp", "10 decomposition candidates",
      "test_decomposition_variant", decomp_strategy="multi_candidate", n_decomp_candidates=10)
    h("M", "entity_redecomp", "Entity-first + re-decompose",
      "test_decomposition_variant", decomp_strategy="entity_first")
    h("M", "triple_t01", "3-way at temp 0.1",
      "test_decomposition_variant", decomp_strategy="triple_decomp", decomp_temperature=0.1)
    h("M", "multi5_t03", "5 candidates at temp 0.3",
      "test_decomposition_variant", decomp_strategy="multi_candidate", n_decomp_candidates=5)
    h("M", "redecomp_t02", "Re-decompose at temp 0.2",
      "test_decomposition_variant", decomp_strategy="redecompose_with_context", decomp_temperature=0.2)

    # ── N: Hop-Specific Strategies (20) ──────────────────────────

    h("N", "h1k1_h2k5", "Hop 1: k=1 focused, Hop 2: k=5 broad",
      "test_hop_specific", hop1_topk=1, hop2_topk=5)
    h("N", "h1k5_h2k1", "Hop 1: k=5 broad, Hop 2: k=1 focused",
      "test_hop_specific", hop1_topk=5, hop2_topk=1)
    h("N", "h1k3_h2k5", "Hop 1: k=3, Hop 2: k=5",
      "test_hop_specific", hop1_topk=3, hop2_topk=5)
    h("N", "h1k1_h2k3", "Hop 1: k=1, Hop 2: k=3",
      "test_hop_specific", hop1_topk=1, hop2_topk=3)
    h("N", "h1t0_h2t01", "Hop 1: temp=0, Hop 2: temp=0.1",
      "test_hop_specific", hop1_temp=0.0, hop2_temp=0.1)
    h("N", "h1t01_h2t0", "Hop 1: temp=0.1, Hop 2: temp=0",
      "test_hop_specific", hop1_temp=0.1, hop2_temp=0.0)
    h("N", "h2_expand_query", "Expand hop 2 query with hop 1 answer",
      "test_hop_specific", hop2_query_expansion=True)
    h("N", "h2_include_h1_ctx", "Include hop 1 context in hop 2",
      "test_hop_specific", hop2_include_hop1_context=True)
    h("N", "h2_expand_include", "Expand + include hop 1 context in hop 2",
      "test_hop_specific", hop2_query_expansion=True, hop2_include_hop1_context=True)
    h("N", "h1k1_h2expand", "Hop 1 k=1 + hop 2 expand",
      "test_hop_specific", hop1_topk=1, hop2_query_expansion=True)
    h("N", "h1k1_h2expand_include", "k=1 + expand + include context",
      "test_hop_specific", hop1_topk=1, hop2_query_expansion=True, hop2_include_hop1_context=True)
    h("N", "h1k2_h2k2", "Both hops k=2",
      "test_hop_specific", hop1_topk=2, hop2_topk=2)
    h("N", "h1k5_h2k5", "Both hops k=5",
      "test_hop_specific", hop1_topk=5, hop2_topk=5)
    h("N", "h1t0_h2t0", "Both hops temp=0",
      "test_hop_specific", hop1_temp=0.0, hop2_temp=0.0)
    h("N", "h1k1_h2k1", "Both hops k=1 (minimal retrieval)",
      "test_hop_specific", hop1_topk=1, hop2_topk=1)
    h("N", "h1k3_h2expand_k5", "k=3/expand k=5",
      "test_hop_specific", hop1_topk=3, hop2_topk=5, hop2_query_expansion=True)
    h("N", "h2_include_k1", "Include h1 context + k=1 for h2",
      "test_hop_specific", hop2_topk=1, hop2_include_hop1_context=True)
    h("N", "h1k1t0_h2k5t01", "h1: k=1,t=0 | h2: k=5,t=0.1",
      "test_hop_specific", hop1_topk=1, hop1_temp=0.0, hop2_topk=5, hop2_temp=0.1)
    h("N", "h1k3t0_h2expand", "h1: k=3,t=0 | h2: expand",
      "test_hop_specific", hop1_topk=3, hop1_temp=0.0, hop2_query_expansion=True)
    h("N", "h1k2_h2k3_expand", "h1:k=2 | h2:k=3+expand",
      "test_hop_specific", hop1_topk=2, hop2_topk=3, hop2_query_expansion=True)

    # ── O: Negative / Adversarial (15) ───────────────────────────
    # Test what makes things WORSE to understand boundaries

    h("O", "random_context", "Use random paragraphs instead of retrieval",
      "test_retrieval_variant", skip_first_n=10, top_k=13)
    h("O", "no_title", "Strip paragraph titles from context",
      "test_context_engineering", order="relevance")  # handled as text_only below
    h("O", "temp08", "Very high temperature (0.8)",
      "test_prompt_variant", temperature=0.8)
    h("O", "temp10", "Maximum temperature (1.0)",
      "test_prompt_variant", temperature=1.0)
    h("O", "maxtok4", "Only 4 output tokens",
      "test_prompt_variant", max_tokens=4)
    h("O", "maxtok2", "Only 2 output tokens",
      "test_prompt_variant", max_tokens=2)
    h("O", "k20", "Retrieve 20 paragraphs (info overload)",
      "test_retrieval_variant", top_k=20)
    h("O", "reverse_only_hop2", "Reverse-order context only",
      "test_context_engineering", order="reverse_relevance", top_k=5)
    h("O", "persona_wrong", "Wrong persona: 'You are a poet.'",
      "test_prompt_variant", persona="You are a poet who answers in metaphors.")
    h("O", "constraint_paragraph", "Constraint: 'Answer in a full paragraph.'",
      "test_prompt_variant", constraint="Answer in a full paragraph with explanation.", max_tokens=128)
    h("O", "no_examples", "Zero-shot with wrong instruction",
      "test_prompt_variant", persona="Summarize the main topic of the context.")
    h("O", "trunc50", "Truncate paragraphs to 50 chars",
      "test_context_engineering", max_para_chars=50)
    h("O", "k1_trunc50", "k=1 + truncate 50",
      "test_context_engineering", top_k=1, max_para_chars=50)
    h("O", "reverse_k10", "Reverse order + k=10",
      "test_context_engineering", order="reverse_relevance", top_k=10)
    h("O", "alpha_trunc100", "Alphabetical + truncate 100",
      "test_context_engineering", order="alphabetical", max_para_chars=100)

    # ── P: Wild Cards (15) ───────────────────────────────────────

    h("P", "title_as_answer", "Use paragraph title as answer if title matches query",
      "test_answer_processing", title_preference=True, use_first_entity=True)
    h("P", "5vote_3temp_refine", "5 votes at 3 temps + refine (kitchen sink)",
      "test_multi_pass", n_passes=5, temperatures=[0.0, 0.1, 0.2, 0.3, 0.4],
      pick_strategy="majority", refine=True)
    h("P", "best_of_breed", "k=2, temp=0, title boost, majority 3",
      "test_multi_pass", n_passes=3, temperatures=[0.0, 0.0, 0.0], pick_strategy="majority")
    h("P", "reformulate_verify", "Reformulate + verify answer",
      "test_question_understanding", reformulate=True)
    h("P", "type_hint_expand", "Type hint + query expand",
      "test_question_understanding", add_answer_type=True, single_compound_query=True)
    h("P", "10pass_majority", "10-pass majority (expensive, max robustness)",
      "test_multi_pass", n_passes=10, pick_strategy="majority")
    h("P", "reverse_decomp_type", "Reverse decomp + type hints",
      "test_question_understanding", reverse_hops=True, add_answer_type=True)
    h("P", "h1k1_verify_majority", "h1:k=1, 3-vote majority, verify",
      "test_answer_processing", majority_vote=True, verify=True)
    h("P", "expand_highlight_numbered", "Expand + highlight + numbered context",
      "test_context_engineering", highlight_entities=True, add_paragraph_numbers=True, interleave_question=True)
    h("P", "5decomp_5vote", "5 decomp candidates + 5-vote extraction",
      "test_multi_pass", n_passes=5, pick_strategy="majority")
    h("P", "first_sent_k10", "First sentence only, k=10",
      "test_context_engineering", top_k=10, only_first_sentence=True)
    h("P", "title_boost_majority3", "Title boost + 3-vote majority",
      "test_multi_pass", n_passes=3, pick_strategy="majority")
    h("P", "3pass_diff_k_shortest", "3 diff k, pick shortest",
      "test_multi_pass", n_passes=3, use_different_topk=True, pick_strategy="shortest")
    h("P", "redecomp_verify", "Re-decompose + verify",
      "test_decomposition_variant", decomp_strategy="redecompose_with_context")
    h("P", "all_in_one", "Every strategy: expand, highlight, type hint, majority 5",
      "test_multi_pass", n_passes=5, temperatures=[0.0, 0.1, 0.2, 0.3, 0.4],
      pick_strategy="majority", refine=True)

    # ── Q: Cross-Category Combos (20) ──────────────────────────────
    # Best ideas from each category combined

    h("Q", "k2_t0_verify", "Retrieval k=2 + temp=0 + verify answer",
      "test_prompt_variant", top_k=2, temperature=0.0)
    h("Q", "title_boost_t0", "Title boost 0.2 + greedy decoding",
      "test_retrieval_variant", use_title_boost=True, title_boost=0.2)
    h("Q", "k1_expert_verify", "k=1 + expert persona + verify",
      "test_prompt_variant", top_k=1, persona="Expert answerer.", temperature=0.0)
    h("Q", "expand_k2_t0", "Expand query + k=2 + temp=0",
      "test_retrieval_variant", expand_query_with_hop=True, top_k=2)
    h("Q", "emphasize_k2_t0", "Emphasize titles + k=2 + greedy",
      "test_context_engineering", emphasize_title=True, top_k=2)
    h("Q", "numbered_k3_t0", "Numbered paragraphs + k=3 + greedy",
      "test_context_engineering", add_paragraph_numbers=True, top_k=3)
    h("Q", "h1k1_h2k3_t0", "h1:k=1 h2:k=3 both temp=0",
      "test_hop_specific", hop1_topk=1, hop2_topk=3, hop1_temp=0.0, hop2_temp=0.0)
    h("Q", "first_sent_k5_t0", "First sentence + k=5 + greedy",
      "test_context_engineering", only_first_sentence=True, top_k=5)
    h("Q", "dedup_k5_t0", "Dedup titles + k=5 + temp=0",
      "test_retrieval_variant", deduplicate_titles=True, top_k=5)
    h("Q", "3vote_t0_shortest", "3 votes all temp=0, pick shortest",
      "test_multi_pass", n_passes=3, temperatures=[0.0, 0.0, 0.0], pick_strategy="shortest")
    h("Q", "name_constraint_k2", "Name constraint + k=2",
      "test_prompt_variant", constraint="Answer with a proper name only.", top_k=2)
    h("Q", "brief_persona_k1", "Brief persona + k=1",
      "test_prompt_variant", persona="Be extremely brief. One entity name only.", top_k=1)
    h("Q", "highlight_k3_t0", "Highlight entities + k=3 + greedy",
      "test_context_engineering", highlight_entities=True, top_k=3)
    h("Q", "expand_dedup_k3", "Expand + dedup + k=3",
      "test_retrieval_variant", expand_query_with_hop=True, deduplicate_titles=True, top_k=3)
    h("Q", "h2expand_t0_k2", "h2 expand + both temp=0 + k=2",
      "test_hop_specific", hop2_query_expansion=True, hop1_temp=0.0, hop2_temp=0.0, hop2_topk=2)
    h("Q", "3vote_name_constraint", "3 votes + name constraint",
      "test_multi_pass", n_passes=3, pick_strategy="majority")
    h("Q", "k3_trunc300_t0", "k=3 + truncate 300 chars + greedy",
      "test_context_engineering", top_k=3, max_para_chars=300)
    h("Q", "redecomp_3vote", "Re-decompose + 3-vote majority",
      "test_multi_pass", n_passes=3, pick_strategy="majority", refine=True)
    h("Q", "title_boost_expand_k3", "Title boost + expand + k=3",
      "test_retrieval_variant", use_title_boost=True, title_boost=0.15,
      expand_query_with_hop=True, top_k=3)
    h("Q", "h1k2_h2k2_t0_highlight", "h1:k=2 h2:k=2 temp=0 + highlight",
      "test_hop_specific", hop1_topk=2, hop2_topk=2, hop1_temp=0.0, hop2_temp=0.0)

    return hypotheses


# ══════════════════════════════════════════════════════════════════════
# Test Runner
# ══════════════════════════════════════════════════════════════════════

# Map of test function names → actual functions
TEST_FUNCTIONS = {
    "baseline_pipeline": baseline_pipeline,
    "test_retrieval_variant": test_retrieval_variant,
    "test_prompt_variant": test_prompt_variant,
    "test_answer_processing": test_answer_processing,
    "test_multi_pass": test_multi_pass,
    "test_question_understanding": test_question_understanding,
    "test_context_engineering": test_context_engineering,
    "test_decomposition_variant": test_decomposition_variant,
    "test_hop_specific": test_hop_specific,
}


def test_one_hypothesis(hyp, samples, retriever, all_auto_decomps):
    """Test a single hypothesis on all samples."""
    fn = TEST_FUNCTIONS[hyp.test_fn]
    em_scores, rem_scores, f1_scores, latencies = [], [], [], []
    per_question = []

    for sample in samples:
        auto_sub_qs = all_auto_decomps[sample["id"]]
        auto_hops = [{"question": sq} for sq in auto_sub_qs]

        t0 = time.time()
        pred, hops = fn(sample, retriever, auto_hops, **hyp.params)
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
    }


def run_battery(limit=30, output_path="results/v8c_creative_battery.json",
                resume=True, verbose=False):
    from datasets import load_dataset

    print("Loading MuSiQue validation set...")
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    samples = [s for s in ds if s.get("answerable", True)][:limit]
    print(f"Testing on {len(samples)} answerable questions\n")

    retriever = EmbeddingRetriever()
    retriever._load_model()

    # Pre-compute decompositions
    print("=" * 70)
    print("PHASE 1: DECOMPOSITION")
    print("=" * 70)

    all_auto_decomps = {}
    for i, sample in enumerate(samples):
        auto_sub_qs = decompose_with_qwen(sample["question"])
        all_auto_decomps[sample["id"]] = auto_sub_qs
        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] decomposed")
    print(f"  Decomposed {len(samples)} questions")

    # Generate hypotheses
    hypotheses = generate_hypotheses()
    print(f"\n{'=' * 70}")
    print(f"PHASE 2: TESTING {len(hypotheses)} CREATIVE HYPOTHESES")
    print("=" * 70)

    # Load previous results
    output = Path(output_path)
    existing_results = {}
    if resume and output.exists():
        with open(output) as f:
            data = json.load(f)
        existing_results = data.get("hypotheses", {})
        print(f"  Resuming: {len(existing_results)} hypotheses already completed")

    # Run baseline
    baseline_key = "X000_baseline"
    if baseline_key not in existing_results:
        print(f"\n  Running BASELINE...")
        baseline_hyp = CreativeHypothesis(id="X000", name="baseline", description="Standard v5 pipeline",
                                          category="X", test_fn="baseline_pipeline")
        baseline_result = test_one_hypothesis(baseline_hyp, samples, retriever, all_auto_decomps)
        existing_results[baseline_key] = baseline_result
        _save(output, existing_results, hypotheses, samples)
        print(f"  BASELINE: EM={baseline_result['em']:.1f}%  relEM={baseline_result['relaxed_em']:.1f}%")
    else:
        baseline_result = existing_results[baseline_key]
        print(f"  BASELINE (cached): EM={baseline_result['em']:.1f}%")

    baseline_em = baseline_result["em"]

    # Run all hypotheses
    completed = 0
    best_delta, best_name = 0, ""

    for idx, hyp in enumerate(hypotheses):
        key = f"{hyp.id}_{hyp.name}"

        if key in existing_results:
            result = existing_results[key]
            delta = result["em"] - baseline_em
            if delta > best_delta:
                best_delta, best_name = delta, key
            completed += 1
            continue

        t0 = time.time()
        try:
            result = test_one_hypothesis(hyp, samples, retriever, all_auto_decomps)
        except Exception as e:
            result = {"em": 0, "relaxed_em": 0, "f1": 0, "latency_ms": 0,
                     "error": str(e), "per_question": []}
            print(f"  [{idx+1}/{len(hypotheses)}] ERROR {hyp.id} {hyp.name}: {e}")

        elapsed = time.time() - t0
        existing_results[key] = result
        completed += 1

        delta = result["em"] - baseline_em
        if delta > best_delta:
            best_delta, best_name = delta, key

        marker = "+" if delta > 0 else ("-" if delta < 0 else "=")
        print(f"  [{marker}] [{completed}/{len(hypotheses)}] {hyp.id} {hyp.name:35s}: "
              f"EM={result['em']:5.1f}% ({delta:+.1f}%)  [{elapsed:.0f}s]  "
              f"[best: {best_name} {best_delta:+.1f}%]")

        if completed % 5 == 0:
            _save(output, existing_results, hypotheses, samples)

    _save(output, existing_results, hypotheses, samples)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"FINAL SUMMARY: {len(hypotheses)} CREATIVE HYPOTHESES")
    print(f"{'=' * 70}")

    items = [(k, v["em"]) for k, v in existing_results.items() if k != baseline_key and "error" not in v]
    items.sort(key=lambda x: x[1], reverse=True)
    improved = sum(1 for _, em in items if em > baseline_em)
    hurt = sum(1 for _, em in items if em < baseline_em)

    print(f"Baseline EM: {baseline_em:.1f}%")
    print(f"Improved: {improved}/{len(items)} | Neutral: {len(items)-improved-hurt} | Hurt: {hurt}")

    print(f"\n  ── Top 15 ──")
    for k, em in items[:15]:
        print(f"  {em-baseline_em:+5.1f}%  EM={em:5.1f}%  {k}")

    print(f"\n  ── Bottom 5 ──")
    for k, em in items[-5:]:
        print(f"  {em-baseline_em:+5.1f}%  EM={em:5.1f}%  {k}")

    # By category
    print(f"\n  ── By Category ──")
    by_cat = defaultdict(list)
    for k, em in items:
        by_cat[k[0]].append((k, em))
    cat_names = {"G": "Retrieval", "H": "Prompts", "I": "Answer Proc", "J": "Multi-Pass",
                "K": "Question", "L": "Context", "M": "Decomp", "N": "Hop-Specific",
                "O": "Adversarial", "P": "Wild Cards"}
    for cat in sorted(by_cat):
        ci = by_cat[cat]
        avg = sum(em for _, em in ci) / len(ci)
        best = max(ci, key=lambda x: x[1])
        print(f"  {cat} ({cat_names.get(cat, cat):12s}): n={len(ci):3d}, "
              f"avg={avg:5.1f}%, best={best[1]:5.1f}% ({best[0]})")


def _save(output_path, results, hypotheses, samples):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "metadata": {
            "n_hypotheses": len(hypotheses),
            "n_completed": len(results),
            "n_samples": len(samples),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "hypotheses": results,
    }
    with open(output, "w") as f:
        json.dump(data, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="v8c: 250 Creative Hypotheses Battery")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", type=str, default="results/v8c_creative_battery.json")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        assert any("qwen2.5:7b" in m for m in models), "qwen2.5:7b not found"
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    start = time.time()
    run_battery(limit=args.limit, output_path=args.output, resume=not args.no_resume, verbose=args.verbose)
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed/3600:.1f} hours ({elapsed/60:.0f} minutes)")


if __name__ == "__main__":
    main()
