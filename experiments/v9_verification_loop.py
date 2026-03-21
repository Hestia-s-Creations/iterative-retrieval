#!/usr/bin/env python3
"""
Iterative Retrieval v9 - Verification Loop Experiment

500 hypotheses (v8b + v8c) proved the current pipeline is near-optimal for
parameter tuning. The bottleneck is extraction quality — the model has the
right paragraph but picks the wrong answer ~40% of the time. Errors are
stochastic (ensemble helps +3.3%) not systematic.

Critical lesson from v8c: A naive verify loop scored 33.3% EM (-26.7%)
because Qwen 7B never says literal "yes", so every answer got replaced.
v9 uses discriminative (KEEP/DISCARD) signals, not substitutive prompts.

Architecture: Decompose → Retrieve → Extract → VERIFY → Chain
                                                  ↓
                                          keep / re-extract / fallback

9 Configs:
  0: baseline          — no verification
  1: aic               — answer-in-context check + greedy retry
  2: entailment        — TRUE/FALSE prompt, k+1 fallback
  3: consistency       — dual-temp extract, majority on disagree
  4: retrieval_verify  — BGE score check on answer entity
  5: confidence_gated  — heuristic gate + entailment on suspicious only
  6: asym_plus_aic     — asymmetric k (h1=3, h2=5+expand) + AIC
  7: multicandidate_consistency — 5-candidate decomp + consistency verify
  8: kitchen_sink      — all combined
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
from collections import Counter
from typing import Dict, List, Optional, Tuple


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
# VERIFICATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def check_answer_in_context(answer, context):
    """Check if the extracted answer appears somewhere in the context.

    FREE (string op). Catches hallucinated entities not found in any paragraph.
    ~9/12 baseline failures produce answers not in context at all.
    """
    if not answer or not context:
        return False
    return normalize_answer(answer) in normalize_answer(context)


REFUSAL_PATTERNS = [
    "not mentioned", "not found", "not specified", "not stated",
    "cannot determine", "cannot be determined", "no information",
    "not enough information", "unknown", "n/a", "none",
    "not clear", "unclear", "insufficient",
]

def is_suspicious_answer(answer, context, question):
    """Heuristic gate: flag answers that are likely wrong.

    Only ~35% of answers get flagged, so expensive verification is targeted.
    Flags: empty, too long (>4 words), refusal patterns, not-in-context.
    """
    if not answer or not answer.strip():
        return True
    # Too long — likely grabbed a whole sentence
    if len(answer.split()) > 4:
        return True
    # Refusal patterns
    ans_lower = answer.lower()
    for pat in REFUSAL_PATTERNS:
        if pat in ans_lower:
            return True
    # Not in context — most valuable signal
    if not check_answer_in_context(answer, context):
        return True
    return False


def verify_entailment(answer, question, context):
    """Ask model: does the context support this answer? TRUE or FALSE.

    Key fix vs v8c: prompt ends with "TRUE or FALSE:" to constrain output.
    Parse FIRST CHARACTER: f/n → reject. Anything else → keep.
    """
    prompt = (
        f"Passage: {context}\n"
        f"Claim: The answer to '{question}' is '{answer}'.\n"
        f"TRUE or FALSE:"
    )
    response = ask_model(prompt, model="qwen2.5:7b", temperature=0.0, max_tokens=8)
    first_char = response.strip().lower()[:1] if response.strip() else ""
    # f = FALSE, n = NO → reject
    if first_char in ('f', 'n'):
        return False
    # t = TRUE, y = YES, anything else → keep (conservative: don't reject on parse failure)
    return True


def verify_retrieval_grounded(answer, paragraphs, retriever):
    """Check if the answer entity is semantically grounded in any paragraph.

    Uses BGE embeddings (already loaded). Zero LLM calls.
    Threshold 0.4: below suggests the answer is hallucinated.
    """
    if not answer or not answer.strip():
        return False
    scores = retriever.get_all_scores(answer, paragraphs)
    return float(np.max(scores)) >= 0.4


# ── Fallback Chain ───────────────────────────────────────────────────

def fallback_reextract(question, context, paragraphs, retriever, top_k=3):
    """Re-extract with tighter constraints, then try k+1 paragraph.

    Returns (new_answer, method) or (None, None) if no better answer found.
    """
    # Attempt 1: re-extract with greedy decoding, shorter max tokens
    prompt = PROMPT_8SHOT.format(context=context, question=question)
    response = ask_model(prompt, model="qwen2.5:7b", temperature=0.0, max_tokens=16)
    new_answer = extract_short_answer(response)
    if new_answer and check_answer_in_context(new_answer, context):
        return new_answer, "reextract_greedy"

    # Attempt 2: get the next-ranked paragraph (k+1) and extract from it
    results = retriever.retrieve(question, paragraphs, top_k=top_k + 1)
    if len(results) > top_k:
        extra_idx = results[top_k][0]
        extra_context = f"[{paragraphs[extra_idx]['title']}] {paragraphs[extra_idx]['paragraph_text']}"
        combined = context + "\n\n" + extra_context
        prompt2 = PROMPT_8SHOT.format(context=combined, question=question)
        response2 = ask_model(prompt2, model="qwen2.5:7b", temperature=0.0, max_tokens=16)
        new_answer2 = extract_short_answer(response2)
        if new_answer2 and check_answer_in_context(new_answer2, combined):
            return new_answer2, "kplus1"

    return None, None


# ══════════════════════════════════════════════════════════════════════
# PIPELINE CONFIGS (0-8)
# ══════════════════════════════════════════════════════════════════════

def _run_hop(hop_q, paragraphs, retriever, top_k=3, temperature=0.1,
             max_tokens=32, query_expansion=None):
    """Shared hop execution: retrieve + extract. Returns (answer, context, results)."""
    query = hop_q
    if query_expansion:
        query = f"{hop_q} {query_expansion}"
    results = retriever.retrieve(query, paragraphs, top_k=top_k)
    retrieved_indices = [idx for idx, _ in results]
    context = "\n\n".join(
        f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
        for idx, _ in results
    )
    prompt = PROMPT_8SHOT.format(context=context, question=hop_q)
    response = ask_model(prompt, model="qwen2.5:7b", temperature=temperature, max_tokens=max_tokens)
    answer = extract_short_answer(response)
    return answer, context, results


def _prepare_hop_question(hop, previous_answers):
    """Substitute previous answers into hop question."""
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


# ── Config 0: Baseline ──────────────────────────────────────────────

def pipeline_baseline(sample, retriever, auto_hops, stats):
    """Standard v5 pipeline. No verification."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        answer, context, results = _run_hop(hop_q, paragraphs, retriever)
        retrieved_indices = [idx for idx, _ in results]
        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 1: Answer-in-Context Check ────────────────────────────────

def pipeline_aic(sample, retriever, auto_hops, stats):
    """answer-in-context check + greedy re-extract on failure."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        answer, context, results = _run_hop(hop_q, paragraphs, retriever)
        retrieved_indices = [idx for idx, _ in results]

        # Verify: is the answer in the context?
        if not check_answer_in_context(answer, context):
            stats["triggered"] += 1
            new_answer, method = fallback_reextract(hop_q, context, paragraphs, retriever)
            if new_answer:
                stats["replaced"] += 1
                answer = new_answer
            else:
                stats["kept_original"] += 1
        else:
            stats["passed"] += 1

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 2: Entailment Verification ────────────────────────────────

def pipeline_entailment(sample, retriever, auto_hops, stats):
    """TRUE/FALSE entailment check, k+1 fallback on rejection."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        answer, context, results = _run_hop(hop_q, paragraphs, retriever)
        retrieved_indices = [idx for idx, _ in results]

        # Entailment check
        if not verify_entailment(answer, hop_q, context):
            stats["triggered"] += 1
            new_answer, method = fallback_reextract(hop_q, context, paragraphs, retriever)
            if new_answer:
                stats["replaced"] += 1
                answer = new_answer
            else:
                stats["kept_original"] += 1
        else:
            stats["passed"] += 1

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 3: Consistency Verification ───────────────────────────────

def pipeline_consistency(sample, retriever, auto_hops, stats):
    """Dual-temp extract, majority vote when they disagree."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)

        # Primary extraction (temp=0.1)
        answer1, context, results = _run_hop(hop_q, paragraphs, retriever, temperature=0.1)
        retrieved_indices = [idx for idx, _ in results]

        # Second extraction (temp=0.0, greedy)
        prompt = PROMPT_8SHOT.format(context=context, question=hop_q)
        response2 = ask_model(prompt, model="qwen2.5:7b", temperature=0.0, max_tokens=32)
        answer2 = extract_short_answer(response2)

        if normalize_answer(answer1) == normalize_answer(answer2):
            # They agree — high confidence
            answer = answer1
            stats["passed"] += 1
        else:
            stats["triggered"] += 1
            # Third extraction to break tie
            response3 = ask_model(prompt, model="qwen2.5:7b", temperature=0.2, max_tokens=32)
            answer3 = extract_short_answer(response3)

            # Majority vote
            candidates = [answer1, answer2, answer3]
            normalized = [normalize_answer(a) for a in candidates]
            counts = Counter(normalized)
            winner_norm = counts.most_common(1)[0][0]

            # Pick the first candidate matching the winner
            for cand, norm in zip(candidates, normalized):
                if norm == winner_norm:
                    answer = cand
                    break

            if normalize_answer(answer) != normalize_answer(answer1):
                stats["replaced"] += 1
            else:
                stats["kept_original"] += 1

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 4: Retrieval-Grounded Verification ────────────────────────

def pipeline_retrieval_verify(sample, retriever, auto_hops, stats):
    """BGE score check on answer entity. Zero LLM verification calls."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        answer, context, results = _run_hop(hop_q, paragraphs, retriever)
        retrieved_indices = [idx for idx, _ in results]

        if not verify_retrieval_grounded(answer, paragraphs, retriever):
            stats["triggered"] += 1
            new_answer, method = fallback_reextract(hop_q, context, paragraphs, retriever)
            if new_answer and verify_retrieval_grounded(new_answer, paragraphs, retriever):
                stats["replaced"] += 1
                answer = new_answer
            else:
                stats["kept_original"] += 1
        else:
            stats["passed"] += 1

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 5: Confidence-Gated Verification ──────────────────────────

def pipeline_confidence_gated(sample, retriever, auto_hops, stats):
    """Heuristic gate + entailment only on suspicious answers (~35%)."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        answer, context, results = _run_hop(hop_q, paragraphs, retriever)
        retrieved_indices = [idx for idx, _ in results]

        if is_suspicious_answer(answer, context, hop_q):
            stats["triggered"] += 1
            # Suspicious — run entailment check
            if not verify_entailment(answer, hop_q, context):
                new_answer, method = fallback_reextract(hop_q, context, paragraphs, retriever)
                if new_answer:
                    stats["replaced"] += 1
                    answer = new_answer
                else:
                    stats["kept_original"] += 1
            else:
                # Entailment says it's fine despite being suspicious
                stats["kept_original"] += 1
        else:
            stats["passed"] += 1

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 6: Asymmetric Retrieval + AIC ─────────────────────────────

def pipeline_asym_plus_aic(sample, retriever, auto_hops, stats):
    """Asymmetric k (h1=3, h2=5+query expansion) + answer-in-context check.

    Combines the two best +6.7% findings from v8c:
    - N196_h1k3_h2expand_k5 (asymmetric retrieval)
    - AIC verification (config 1 of v9)
    """
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)

        # Asymmetric: hop 1 gets k=3, hop 2 gets k=5 + query expansion
        if i == 0:
            answer, context, results = _run_hop(hop_q, paragraphs, retriever, top_k=3)
        else:
            expansion = previous_answers[-1] if previous_answers else None
            answer, context, results = _run_hop(
                hop_q, paragraphs, retriever, top_k=5, query_expansion=expansion
            )
        retrieved_indices = [idx for idx, _ in results]

        # AIC check
        if not check_answer_in_context(answer, context):
            stats["triggered"] += 1
            top_k = 3 if i == 0 else 5
            new_answer, method = fallback_reextract(hop_q, context, paragraphs, retriever, top_k=top_k)
            if new_answer:
                stats["replaced"] += 1
                answer = new_answer
            else:
                stats["kept_original"] += 1
        else:
            stats["passed"] += 1

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 7: Multi-Candidate Decomposition + Consistency ────────────

def decompose_multi_candidate(question, n, retriever, paragraphs):
    """Generate n decomposition candidates, pick the one with best hop-1 retrieval."""
    candidates = []
    for c in range(n):
        temp = 0.2 + 0.1 * c
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

    if not candidates:
        return None

    # Pick candidate whose hop 1 question retrieves the highest-scoring paragraph
    best_hops = None
    best_score = -1
    for cand in candidates:
        scores = retriever.get_all_scores(cand[0], paragraphs)
        top_score = float(np.max(scores))
        if top_score > best_score:
            best_score = top_score
            best_hops = cand
    return best_hops


def pipeline_multicandidate_consistency(sample, retriever, auto_hops, stats):
    """5-candidate decomposition + consistency verification on each hop."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    # Multi-candidate decomposition
    best_decomp = decompose_multi_candidate(
        sample["question"], 5, retriever, paragraphs
    )
    if best_decomp:
        hops = [{"question": q} for q in best_decomp]
    else:
        hops = auto_hops

    for i, hop in enumerate(hops):
        hop_q = _prepare_hop_question(hop, previous_answers)

        # Primary extraction
        answer1, context, results = _run_hop(hop_q, paragraphs, retriever, temperature=0.1)
        retrieved_indices = [idx for idx, _ in results]

        # Consistency: second extraction
        prompt = PROMPT_8SHOT.format(context=context, question=hop_q)
        response2 = ask_model(prompt, model="qwen2.5:7b", temperature=0.0, max_tokens=32)
        answer2 = extract_short_answer(response2)

        if normalize_answer(answer1) == normalize_answer(answer2):
            answer = answer1
            stats["passed"] += 1
        else:
            stats["triggered"] += 1
            # Disagree — use AIC to pick the better one
            a1_in = check_answer_in_context(answer1, context)
            a2_in = check_answer_in_context(answer2, context)
            if a2_in and not a1_in:
                answer = answer2
                stats["replaced"] += 1
            elif a1_in and not a2_in:
                answer = answer1
                stats["kept_original"] += 1
            else:
                # Both in or both out — prefer greedy (answer2)
                answer = answer2
                if normalize_answer(answer) != normalize_answer(answer1):
                    stats["replaced"] += 1
                else:
                    stats["kept_original"] += 1

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 8: Kitchen Sink ───────────────────────────────────────────

def pipeline_kitchen_sink(sample, retriever, auto_hops, stats):
    """All verification combined: multi-candidate decomp + asymmetric retrieval
    + AIC + consistency + confidence-gated entailment."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    # Multi-candidate decomposition (5 candidates)
    best_decomp = decompose_multi_candidate(
        sample["question"], 5, retriever, paragraphs
    )
    if best_decomp:
        hops = [{"question": q} for q in best_decomp]
    else:
        hops = auto_hops

    for i, hop in enumerate(hops):
        hop_q = _prepare_hop_question(hop, previous_answers)

        # Asymmetric retrieval
        if i == 0:
            top_k = 3
            answer1, context, results = _run_hop(hop_q, paragraphs, retriever, top_k=3, temperature=0.1)
        else:
            top_k = 5
            expansion = previous_answers[-1] if previous_answers else None
            answer1, context, results = _run_hop(
                hop_q, paragraphs, retriever, top_k=5, temperature=0.1, query_expansion=expansion
            )
        retrieved_indices = [idx for idx, _ in results]

        # Consistency: second extraction (greedy)
        prompt = PROMPT_8SHOT.format(context=context, question=hop_q)
        response2 = ask_model(prompt, model="qwen2.5:7b", temperature=0.0, max_tokens=32)
        answer2 = extract_short_answer(response2)

        if normalize_answer(answer1) == normalize_answer(answer2):
            # They agree — use AIC as final check
            answer = answer1
            if not check_answer_in_context(answer, context):
                stats["triggered"] += 1
                # Consistent but hallucinated — try fallback
                new_answer, method = fallback_reextract(hop_q, context, paragraphs, retriever, top_k=top_k)
                if new_answer:
                    stats["replaced"] += 1
                    answer = new_answer
                else:
                    stats["kept_original"] += 1
            else:
                stats["passed"] += 1
        else:
            stats["triggered"] += 1
            # Disagree — pick best via AIC
            a1_in = check_answer_in_context(answer1, context)
            a2_in = check_answer_in_context(answer2, context)

            if a2_in and not a1_in:
                answer = answer2
            elif a1_in and not a2_in:
                answer = answer1
            else:
                # Both in context (or both out) — use confidence-gated entailment
                if is_suspicious_answer(answer1, context, hop_q):
                    if verify_entailment(answer2, hop_q, context):
                        answer = answer2
                    elif verify_entailment(answer1, hop_q, context):
                        answer = answer1
                    else:
                        # Both fail — try fallback
                        new_answer, method = fallback_reextract(hop_q, context, paragraphs, retriever, top_k=top_k)
                        answer = new_answer if new_answer else answer1
                else:
                    answer = answer1

            if normalize_answer(answer) != normalize_answer(answer1):
                stats["replaced"] += 1
            else:
                stats["kept_original"] += 1

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ══════════════════════════════════════════════════════════════════════
# CONFIG REGISTRY
# ══════════════════════════════════════════════════════════════════════

CONFIGS = [
    {"id": 0, "name": "baseline",                   "fn": pipeline_baseline,
     "description": "Standard v5 pipeline, no verification"},
    {"id": 1, "name": "aic",                         "fn": pipeline_aic,
     "description": "Answer-in-context check + greedy retry"},
    {"id": 2, "name": "entailment",                  "fn": pipeline_entailment,
     "description": "TRUE/FALSE entailment prompt, k+1 fallback"},
    {"id": 3, "name": "consistency",                  "fn": pipeline_consistency,
     "description": "Dual-temp extract, majority vote on disagree"},
    {"id": 4, "name": "retrieval_verify",             "fn": pipeline_retrieval_verify,
     "description": "BGE score check on answer entity (zero LLM)"},
    {"id": 5, "name": "confidence_gated",             "fn": pipeline_confidence_gated,
     "description": "Heuristic gate + entailment on suspicious only"},
    {"id": 6, "name": "asym_plus_aic",                "fn": pipeline_asym_plus_aic,
     "description": "Asymmetric k (h1=3,h2=5+expand) + AIC"},
    {"id": 7, "name": "multicandidate_consistency",   "fn": pipeline_multicandidate_consistency,
     "description": "5-candidate decomp + consistency verify"},
    {"id": 8, "name": "kitchen_sink",                 "fn": pipeline_kitchen_sink,
     "description": "All combined: multi-decomp + asym + AIC + consistency + entailment"},
]


# ══════════════════════════════════════════════════════════════════════
# TEST RUNNER
# ══════════════════════════════════════════════════════════════════════

def test_one_config(config, samples, retriever, all_auto_decomps):
    """Test a single config on all samples. Returns results dict with verify_stats."""
    fn = config["fn"]
    em_scores, rem_scores, f1_scores, latencies = [], [], [], []
    per_question = []

    # Aggregate verification stats across all samples
    stats = {"triggered": 0, "replaced": 0, "kept_original": 0, "passed": 0}

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
    total_hops = stats["triggered"] + stats["passed"]
    return {
        "em": sum(em_scores) / n * 100,
        "relaxed_em": sum(rem_scores) / n * 100,
        "f1": sum(f1_scores) / n,
        "latency_ms": sum(latencies) / n,
        "per_question": per_question,
        "verify_stats": {
            "total_hops": total_hops,
            "triggered": stats["triggered"],
            "triggered_pct": stats["triggered"] / total_hops * 100 if total_hops else 0,
            "replaced": stats["replaced"],
            "replaced_pct": stats["replaced"] / total_hops * 100 if total_hops else 0,
            "kept_original": stats["kept_original"],
            "kept_pct": stats["kept_original"] / total_hops * 100 if total_hops else 0,
            "passed": stats["passed"],
            "passed_pct": stats["passed"] / total_hops * 100 if total_hops else 0,
        },
    }


def _save(output_path, results, n_configs, n_samples):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "metadata": {
            "experiment": "v9_verification_loop",
            "n_configs": n_configs,
            "n_completed": len(results),
            "n_samples": n_samples,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "configs": results,
    }
    with open(output, "w") as f:
        json.dump(data, f, indent=2, default=str)


def run_experiment(limit=30, output_path="results/v9_verification_loop.json",
                   resume=True, configs_to_run=None):
    from datasets import load_dataset

    print("Loading MuSiQue validation set...")
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    samples = [s for s in ds if s.get("answerable", True)][:limit]
    print(f"Testing on {len(samples)} answerable questions\n")

    retriever = EmbeddingRetriever()
    retriever._load_model()

    # Phase 1: Pre-compute decompositions
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

    # Load previous results
    output = Path(output_path)
    existing_results = {}
    if resume and output.exists():
        with open(output) as f:
            data = json.load(f)
        existing_results = data.get("configs", {})
        print(f"  Resuming: {len(existing_results)} configs already completed")

    # Phase 2: Run configs
    print(f"\n{'=' * 70}")
    print(f"PHASE 2: TESTING {len(CONFIGS)} VERIFICATION CONFIGS")
    print("=" * 70)

    # Filter configs if specific ones requested
    configs = CONFIGS
    if configs_to_run is not None:
        configs = [c for c in CONFIGS if c["id"] in configs_to_run]

    baseline_em = None

    for config in configs:
        key = f"{config['id']}_{config['name']}"

        if key in existing_results:
            result = existing_results[key]
            if config["id"] == 0:
                baseline_em = result["em"]
            print(f"  [cached] Config {config['id']}: {config['name']:35s} "
                  f"EM={result['em']:5.1f}%")
            continue

        print(f"\n  Running Config {config['id']}: {config['name']}...")
        print(f"  {config['description']}")

        t0 = time.time()
        try:
            result = test_one_config(config, samples, retriever, all_auto_decomps)
        except Exception as e:
            result = {"em": 0, "relaxed_em": 0, "f1": 0, "latency_ms": 0,
                     "error": str(e), "per_question": [], "verify_stats": {}}
            print(f"  ERROR: {e}")

        elapsed = time.time() - t0
        existing_results[key] = result

        if config["id"] == 0:
            baseline_em = result["em"]

        delta = f" ({result['em'] - baseline_em:+.1f}%)" if baseline_em is not None and config["id"] != 0 else ""
        vs = result.get("verify_stats", {})
        verify_info = ""
        if vs.get("total_hops", 0) > 0:
            verify_info = (f"  [verify: triggered={vs['triggered_pct']:.0f}%, "
                          f"replaced={vs['replaced_pct']:.0f}%, "
                          f"kept={vs['kept_pct']:.0f}%]")

        print(f"  -> EM={result['em']:5.1f}%{delta}  "
              f"relEM={result['relaxed_em']:5.1f}%  "
              f"F1={result['f1']:.3f}  "
              f"lat={result['latency_ms']:.0f}ms  "
              f"[{elapsed:.0f}s]{verify_info}")

        _save(output_path, existing_results, len(CONFIGS), len(samples))

    _save(output_path, existing_results, len(CONFIGS), len(samples))

    # Summary
    print(f"\n{'=' * 70}")
    print(f"FINAL SUMMARY: v9 Verification Loop")
    print(f"{'=' * 70}")

    if baseline_em is None:
        baseline_em = existing_results.get("0_baseline", {}).get("em", 60.0)

    print(f"\nBaseline EM: {baseline_em:.1f}%\n")
    print(f"{'Config':<5} {'Name':<35} {'EM':>6} {'Delta':>7} {'relEM':>6} "
          f"{'F1':>6} {'Lat(ms)':>8} {'Trig%':>6} {'Repl%':>6}")
    print("-" * 95)

    for config in CONFIGS:
        key = f"{config['id']}_{config['name']}"
        r = existing_results.get(key)
        if not r:
            continue
        delta = r["em"] - baseline_em
        vs = r.get("verify_stats", {})
        trig = f"{vs.get('triggered_pct', 0):.0f}%" if vs else "-"
        repl = f"{vs.get('replaced_pct', 0):.0f}%" if vs else "-"
        marker = "+" if delta > 0.5 else ("-" if delta < -0.5 else "=")
        print(f"[{marker}] {config['id']:<3} {config['name']:<35} {r['em']:5.1f}% "
              f"{delta:+5.1f}%  {r['relaxed_em']:5.1f}% {r['f1']:.3f} "
              f"{r['latency_ms']:7.0f}  {trig:>6} {repl:>6}")

    # Verification criteria check
    print(f"\n{'=' * 70}")
    print("VERIFICATION CRITERIA")
    print("=" * 70)

    all_ems = {config["name"]: existing_results.get(f"{config['id']}_{config['name']}", {}).get("em", 0)
               for config in CONFIGS}

    c1 = abs(all_ems.get("baseline", 0) - 60.0) < 5.0
    print(f"  1. Baseline ~60%: {all_ems.get('baseline', 0):.1f}% {'PASS' if c1 else 'FAIL'}")

    below = [n for n, em in all_ems.items() if em < baseline_em - 0.5 and n != "baseline"]
    c2 = len(below) == 0
    print(f"  2. No config below baseline: {len(below)} below {'PASS' if c2 else 'FAIL'}")
    if below:
        for n in below:
            print(f"     - {n}: {all_ems[n]:.1f}%")

    best_non_baseline = max((em for n, em in all_ems.items() if n != "baseline"), default=0)
    c3 = best_non_baseline > 63.3
    print(f"  3. Any config >63.3%: best={best_non_baseline:.1f}% {'PASS' if c3 else 'FAIL'}")

    ks_em = all_ems.get("kitchen_sink", 0)
    c4 = ks_em >= 70.0
    print(f"  4. Kitchen sink >=70%: {ks_em:.1f}% {'PASS' if c4 else 'ASPIRATIONAL'}")

    c5 = all(existing_results.get(f"{c['id']}_{c['name']}", {}).get("verify_stats")
             for c in CONFIGS if c["id"] != 0)
    print(f"  5. verify_stats logged: {'PASS' if c5 else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser(description="v9: Verification Loop Experiment")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", type=str, default="results/v9_verification_loop.json")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config IDs to run (e.g. '0,1,6')")
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
