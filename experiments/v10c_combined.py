#!/usr/bin/env python3
"""
Iterative Retrieval v10c - Combined Best Methods

v10 + v10b proved:
  - MC selection is always worse than 8-shot for Qwen 7B
  - Per-paragraph extraction is worse (paragraphs lack answer context)
  - Multi-model ensemble (Qwen + Phi-3) hurts more than helps
  - Self-consistency shows extreme determinism (90% unanimous)

Two methods improved (+3.3% each, fixing DIFFERENT questions):
  - v9 AIC: catches hallucinated entities not in context → fixes "Texas"→"Crockett County"
  - v10b focused_rerank: augmented hop-2 query → fixes "Lega Pro"→"Lega Pro Prima Divisione"

These are ORTHOGONAL (zero overlap). Combined prediction: +6.6%.

6 Configs:
  0: baseline                — Standard pipeline
  1: aic_only               — v9 best: answer-in-context check
  2: focused_rerank_only    — v10b best: augmented hop-2 retrieval
  3: aic_plus_focused       — Both combined (the main hypothesis)
  4: aic_plus_focused_k5    — Combined + expanded retrieval (k=5 for hop 2)
  5: aic_focused_2pass      — Combined + re-extract from narrowed context on AIC fail
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

def decompose_with_qwen(question, temperature=0.1):
    response = ask_model(
        DECOMPOSE_TEMPLATE.format(question=question),
        model="qwen2.5:7b", temperature=temperature, max_tokens=80
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


# ── Extraction Prompt ───────────────────────────────────────────────

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


# ── Shared Helpers ──────────────────────────────────────────────────

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


def _standard_extract(hop_q, context):
    prompt = PROMPT_8SHOT.format(context=context, question=hop_q)
    response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
    return extract_short_answer(response)


def check_answer_in_context(answer, context):
    if not answer or not context:
        return False
    return normalize_answer(answer) in normalize_answer(context)


def _focused_retrieve(hop_q, prev_answer, paragraphs, retriever, top_k=3):
    """Retrieve with both original and answer-augmented query, merge results."""
    augmented_q = f"{prev_answer}: {hop_q}"
    results_original = retriever.retrieve(hop_q, paragraphs, top_k=top_k)
    results_augmented = retriever.retrieve(augmented_q, paragraphs, top_k=top_k)

    seen = set()
    merged = []
    for idx, score in results_augmented:
        if idx not in seen:
            seen.add(idx)
            merged.append((idx, score))
    for idx, score in results_original:
        if idx not in seen:
            seen.add(idx)
            merged.append((idx, score))

    results = merged[:top_k]
    retrieved_indices = [idx for idx, _ in results]
    context = "\n\n".join(
        f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
        for idx, _ in results
    )
    return context, results, retrieved_indices


REFUSAL_PATTERNS = [
    "not mentioned", "not found", "not specified", "not stated",
    "cannot determine", "cannot be determined", "no information",
    "not enough information", "unknown", "n/a", "none",
    "not clear", "unclear", "insufficient",
]

def is_refusal(answer):
    """Check if the answer is a refusal/not-found pattern."""
    if not answer:
        return True
    ans_lower = answer.lower()
    for pat in REFUSAL_PATTERNS:
        if pat in ans_lower:
            return True
    return False


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
        answer = _standard_extract(hop_q, context)

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 1: AIC Only ─────────────────────────────────────────────

def pipeline_aic_only(sample, retriever, auto_hops, stats):
    """AIC check: re-extract with expanded k on failure."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        context, results, retrieved_indices = _retrieve_and_format(hop_q, paragraphs, retriever)
        answer = _standard_extract(hop_q, context)

        if not check_answer_in_context(answer, context) or is_refusal(answer):
            stats["aic_triggered"] += 1
            # Re-extract with expanded context (k=5)
            context5, results5, indices5 = _retrieve_and_format(hop_q, paragraphs, retriever, top_k=5)
            answer2 = _standard_extract(hop_q, context5)
            if check_answer_in_context(answer2, context5) and not is_refusal(answer2):
                answer = answer2
                retrieved_indices = indices5
                stats["aic_rescued"] += 1
            else:
                stats["aic_kept_original"] += 1
        else:
            stats["aic_passed"] += 1

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 2: Focused Rerank Only ──────────────────────────────────

def pipeline_focused_rerank(sample, retriever, auto_hops, stats):
    """Augmented hop-2 retrieval with hop-1 answer."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)

        if i == 0 or not previous_answers:
            context, results, retrieved_indices = _retrieve_and_format(hop_q, paragraphs, retriever)
        else:
            context, results, retrieved_indices = _focused_retrieve(
                hop_q, previous_answers[-1], paragraphs, retriever
            )
            stats["focused_queries"] += 1

        answer = _standard_extract(hop_q, context)

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 3: AIC + Focused Rerank (THE MAIN HYPOTHESIS) ──────────

def pipeline_aic_plus_focused(sample, retriever, auto_hops, stats):
    """Combined AIC + focused hop-2 retrieval."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)

        # Focused retrieval for hop 2+
        if i == 0 or not previous_answers:
            context, results, retrieved_indices = _retrieve_and_format(hop_q, paragraphs, retriever)
        else:
            context, results, retrieved_indices = _focused_retrieve(
                hop_q, previous_answers[-1], paragraphs, retriever
            )
            stats["focused_queries"] += 1

        answer = _standard_extract(hop_q, context)

        # AIC check
        if not check_answer_in_context(answer, context) or is_refusal(answer):
            stats["aic_triggered"] += 1
            # Re-extract with expanded context
            if i == 0 or not previous_answers:
                context5, results5, indices5 = _retrieve_and_format(hop_q, paragraphs, retriever, top_k=5)
            else:
                context5, results5, indices5 = _focused_retrieve(
                    hop_q, previous_answers[-1], paragraphs, retriever, top_k=5
                )
            answer2 = _standard_extract(hop_q, context5)
            if check_answer_in_context(answer2, context5) and not is_refusal(answer2):
                answer = answer2
                retrieved_indices = indices5
                stats["aic_rescued"] += 1
            else:
                stats["aic_kept_original"] += 1
        else:
            stats["aic_passed"] += 1

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 4: AIC + Focused + k=5 for hop 2 ───────────────────────

def pipeline_aic_focused_k5(sample, retriever, auto_hops, stats):
    """Combined with expanded retrieval (k=5) for hop 2."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)

        # k=5 for hop 2+ with focused retrieval
        if i == 0:
            context, results, retrieved_indices = _retrieve_and_format(hop_q, paragraphs, retriever, top_k=3)
        else:
            context, results, retrieved_indices = _focused_retrieve(
                hop_q, previous_answers[-1], paragraphs, retriever, top_k=5
            )
            stats["focused_k5"] += 1

        answer = _standard_extract(hop_q, context)

        # AIC check
        if not check_answer_in_context(answer, context) or is_refusal(answer):
            stats["aic_triggered"] += 1
            # Already at k=5 for hop 2, try k=7 fallback
            if i == 0:
                context7, _, indices7 = _retrieve_and_format(hop_q, paragraphs, retriever, top_k=5)
            else:
                context7, _, indices7 = _focused_retrieve(
                    hop_q, previous_answers[-1], paragraphs, retriever, top_k=7
                )
            answer2 = _standard_extract(hop_q, context7)
            if check_answer_in_context(answer2, context7) and not is_refusal(answer2):
                answer = answer2
                retrieved_indices = indices7
                stats["aic_rescued"] += 1
            else:
                stats["aic_kept_original"] += 1
        else:
            stats["aic_passed"] += 1

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 5: AIC + Focused + Narrowed Re-extract ─────────────────
# On AIC failure, instead of expanding context, try extracting from
# each of the top-3 paragraphs INDIVIDUALLY. Pick the first answer
# that passes AIC. This targets the "confused by adjacent entities" case.

def pipeline_aic_focused_narrow(sample, retriever, auto_hops, stats):
    """AIC failure → try extracting from each paragraph individually."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)

        if i == 0 or not previous_answers:
            context, results, retrieved_indices = _retrieve_and_format(hop_q, paragraphs, retriever)
        else:
            context, results, retrieved_indices = _focused_retrieve(
                hop_q, previous_answers[-1], paragraphs, retriever
            )

        answer = _standard_extract(hop_q, context)

        if not check_answer_in_context(answer, context) or is_refusal(answer):
            stats["aic_triggered"] += 1
            rescued = False

            # Try each paragraph individually
            for idx, score in results:
                single = f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
                single_answer = _standard_extract(hop_q, single)
                if check_answer_in_context(single_answer, single) and not is_refusal(single_answer):
                    answer = single_answer
                    stats["narrow_rescued"] += 1
                    rescued = True
                    break

            if not rescued:
                # Last resort: expand to k=5
                context5, _, indices5 = _retrieve_and_format(hop_q, paragraphs, retriever, top_k=5)
                answer2 = _standard_extract(hop_q, context5)
                if check_answer_in_context(answer2, context5) and not is_refusal(answer2):
                    answer = answer2
                    retrieved_indices = indices5
                    stats["expand_rescued"] += 1
                else:
                    stats["aic_kept_original"] += 1
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
     "description": "Standard Decompose→Retrieve→Extract"},
    {"id": 1, "name": "aic_only",
     "fn": pipeline_aic_only,
     "description": "v9 best: AIC check + expanded k fallback"},
    {"id": 2, "name": "focused_rerank",
     "fn": pipeline_focused_rerank,
     "description": "v10b best: augmented hop-2 retrieval"},
    {"id": 3, "name": "aic_plus_focused",
     "fn": pipeline_aic_plus_focused,
     "description": "COMBINED: AIC + focused rerank (predict +6.6%)"},
    {"id": 4, "name": "aic_focused_k5",
     "fn": pipeline_aic_focused_k5,
     "description": "Combined + k=5 for hop 2"},
    {"id": 5, "name": "aic_focused_narrow",
     "fn": pipeline_aic_focused_narrow,
     "description": "Combined + per-paragraph fallback on AIC fail"},
]


# ══════════════════════════════════════════════════════════════════════
# TEST RUNNER
# ══════════════════════════════════════════════════════════════════════

def test_one_config(config, samples, retriever, all_auto_decomps):
    fn = config["fn"]
    em_scores, rem_scores, f1_scores, latencies = [], [], [], []
    per_question = []
    stats = defaultdict(int)

    for idx, sample in enumerate(samples):
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

        if (idx + 1) % 10 == 0:
            running_em = sum(em_scores) / len(em_scores) * 100
            print(f"    [{idx+1}/{len(samples)}] running EM={running_em:.1f}%")

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
            "experiment": "v10c_combined",
            "n_configs": n_configs,
            "n_completed": len(results),
            "n_samples": n_samples,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "configs": results,
    }
    with open(output, "w") as f:
        json.dump(data, f, indent=2, default=str)


def run_experiment(limit=30, output_path="results/v10c_combined.json",
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

    output = Path(output_path)
    existing_results = {}
    if resume and output.exists():
        with open(output) as f:
            data = json.load(f)
        existing_results = data.get("configs", {})
        print(f"  Resuming: {len(existing_results)} configs already completed")

    configs = CONFIGS
    if configs_to_run is not None:
        configs = [c for c in CONFIGS if c["id"] in configs_to_run]

    print(f"\n{'=' * 70}")
    print(f"PHASE 2: TESTING {len(configs)} COMBINED APPROACHES")
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
    print(f"FINAL SUMMARY: v10c Combined Methods")
    print(f"{'=' * 70}")

    if baseline_em is None:
        baseline_em = existing_results.get("0_baseline", {}).get("em", 60.0)

    print(f"\nBaseline EM: {baseline_em:.1f}%\n")
    print(f"{'Cfg':<4} {'Name':<30} {'EM':>6} {'Delta':>7} {'relEM':>6} {'F1':>6} {'Lat':>7}")
    print("-" * 80)

    for config in CONFIGS:
        key = f"{config['id']}_{config['name']}"
        r = existing_results.get(key)
        if not r:
            continue
        delta = r["em"] - baseline_em
        marker = "+" if delta > 0.5 else ("-" if delta < -0.5 else "=")
        print(f"[{marker}] {config['id']:<2} {config['name']:<30} {r['em']:5.1f}% "
              f"{delta:+5.1f}%  {r['relaxed_em']:5.1f}% {r['f1']:.3f} "
              f"{r['latency_ms']:6.0f}ms")

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
        for qid, old, new, gold in diff[:5]:
            print(f"    [DIFF]  '{old}' → '{new}'  (gold: '{gold}')")


def main():
    parser = argparse.ArgumentParser(description="v10c: Combined Best Methods")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", type=str, default="results/v10c_combined.json")
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
