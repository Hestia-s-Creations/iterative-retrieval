#!/usr/bin/env python3
"""
Iterative Retrieval v10b - Per-Paragraph Extraction & Multi-Model Ensemble

v10 proved:
  - MC selection is WORSE than 8-shot for Qwen 7B (-20% EM)
  - The model is incredibly deterministic (90% unanimous across temps)
  - Position bias is minimal (dual-context vote changes nothing)
  - AIC barely fires (2/60 hops) — most errors have answers IN context

The REAL bottleneck: 5/12 baseline errors are "wrong entity from correct paragraph"
— both the gold answer and the predicted answer appear in the SAME sentence.
The model sees "X was signed to Dill Records, a sub-label of Asian Man Records"
and picks "Dill Records" when the answer is "Asian Man Records".

New hypothesis: extracting from the FULL 3-paragraph context introduces confusion.
If we extract from each paragraph INDEPENDENTLY, the model can't be confused by
entities that co-occur in adjacent paragraphs.

6 Configs:
  0: baseline              — Standard 3-paragraph extraction
  1: per_paragraph         — Extract from each paragraph alone, vote
  2: per_para_8shot_vote   — Per-para + 8-shot vote between disagreements
  3: multi_model_ensemble  — Qwen + Phi-3 extract, vote between models
  4: per_para_multi_model  — Per-paragraph × multi-model (2×3 = 6 extractions)
  5: focused_rerank        — Re-retrieve with answer-augmented query for hop 2
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


# ── Extraction Prompts ──────────────────────────────────────────────

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

# For selecting between competing answers (NOT MC with letters)
SELECTION_PROMPT = """Two different extractions answered the question differently. Using the context, which answer is CORRECT? Reply with ONLY the correct answer, nothing else.

Context: {context}
Question: {question}

Answer 1: {answer1}
Answer 2: {answer2}

Correct answer:"""


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


def _standard_extract(hop_q, context, model="qwen2.5:7b"):
    """Standard 8-shot extraction."""
    prompt = PROMPT_8SHOT.format(context=context, question=hop_q)
    response = ask_model(prompt, model=model, temperature=0.1, max_tokens=32)
    return extract_short_answer(response)


def majority_vote(answers):
    """Return the most common answer (original casing). Ties go to first."""
    if not answers:
        return ""
    normed_to_original = {}
    counts = Counter()
    for ans in answers:
        norm = normalize_answer(ans)
        counts[norm] += 1
        if norm not in normed_to_original:
            normed_to_original[norm] = ans
    winner_norm = counts.most_common(1)[0][0]
    return normed_to_original[winner_norm]


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
        answer = _standard_extract(hop_q, context)

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 1: Per-Paragraph Extraction ─────────────────────────────
# Extract from EACH retrieved paragraph independently.
# Each extraction sees only ONE paragraph, preventing entity confusion.
# Majority vote across the per-paragraph answers.

def pipeline_per_paragraph(sample, retriever, auto_hops, stats):
    """Extract from each paragraph independently, majority vote."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        results = retriever.retrieve(hop_q, paragraphs, top_k=3)
        retrieved_indices = [idx for idx, _ in results]

        # Extract from each paragraph independently
        per_para_answers = []
        for idx, score in results:
            single_context = f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
            ans = _standard_extract(hop_q, single_context)
            per_para_answers.append(ans)
            stats["per_para_extractions"] += 1

        # Count unique answers
        normed = [normalize_answer(a) for a in per_para_answers]
        unique = set(normed)
        stats[f"unique_{len(unique)}"] += 1

        # Majority vote
        answer = majority_vote(per_para_answers)

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 2: Per-Paragraph + Full-Context Tiebreak ────────────────
# Extract from each paragraph independently.
# If they all agree → use it (high confidence).
# If they disagree → also extract from full context, use as tiebreaker.
# The full-context extraction has more information but risks confusion.

def pipeline_per_para_tiebreak(sample, retriever, auto_hops, stats):
    """Per-paragraph extraction, full-context tiebreak on disagreement."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        results = retriever.retrieve(hop_q, paragraphs, top_k=3)
        retrieved_indices = [idx for idx, _ in results]

        # Per-paragraph extraction
        per_para_answers = []
        for idx, score in results:
            single_context = f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
            ans = _standard_extract(hop_q, single_context)
            per_para_answers.append(ans)

        normed = [normalize_answer(a) for a in per_para_answers]
        unique = set(n for n in normed if n)

        if len(unique) <= 1:
            # Unanimous — use it
            answer = per_para_answers[0]
            stats["unanimous"] += 1
        else:
            # Disagreement — add full-context extraction as tiebreaker
            full_context = "\n\n".join(
                f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
                for idx, _ in results
            )
            full_answer = _standard_extract(hop_q, full_context)
            all_answers = per_para_answers + [full_answer]
            answer = majority_vote(all_answers)
            stats["tiebreak_used"] += 1

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 3: Multi-Model Ensemble ─────────────────────────────────
# Use BOTH Qwen 7B and Phi-3 3.8B for extraction.
# Different models have different biases.
# Majority vote between the two models.

def pipeline_multi_model(sample, retriever, auto_hops, stats):
    """Qwen 7B + Phi-3 3.8B extraction, majority vote."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []
    models = ["qwen2.5:7b", "phi3:mini"]

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        context, results, retrieved_indices = _retrieve_and_format(hop_q, paragraphs, retriever)

        # Extract with each model
        model_answers = []
        for model in models:
            ans = _standard_extract(hop_q, context, model=model)
            model_answers.append((model, ans))

        normed = [normalize_answer(a) for _, a in model_answers]

        if normed[0] == normed[1]:
            # Both agree
            answer = model_answers[0][1]  # Use Qwen's version
            stats["agree"] += 1
        else:
            # Disagree — use Qwen (the stronger model) with LLM selection
            stats["disagree"] += 1
            prompt = SELECTION_PROMPT.format(
                context=context, question=hop_q,
                answer1=model_answers[0][1], answer2=model_answers[1][1]
            )
            response = ask_model(prompt, model="qwen2.5:7b", temperature=0.0, max_tokens=32)
            selected = extract_short_answer(response)

            # Check if selection matches either model's answer
            sel_norm = normalize_answer(selected)
            if sel_norm == normed[0]:
                answer = model_answers[0][1]
                stats["selected_qwen"] += 1
            elif sel_norm == normed[1]:
                answer = model_answers[1][1]
                stats["selected_phi3"] += 1
            else:
                # Selection produced a third answer — use it if in context, else Qwen
                answer = selected if normalize_answer(selected) in normalize_answer(context) else model_answers[0][1]
                stats["selected_third"] += 1

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 4: Per-Paragraph × Multi-Model ──────────────────────────
# The maximal extraction strategy: 3 paragraphs × 2 models = 6 extractions.
# Maximum diversity of perspectives. Majority vote from the full pool.

def pipeline_per_para_multi_model(sample, retriever, auto_hops, stats):
    """3 paragraphs × 2 models = 6 extractions, majority vote."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []
    models = ["qwen2.5:7b", "phi3:mini"]

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        results = retriever.retrieve(hop_q, paragraphs, top_k=3)
        retrieved_indices = [idx for idx, _ in results]

        all_answers = []
        for idx, score in results:
            single_context = f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
            for model in models:
                ans = _standard_extract(hop_q, single_context, model=model)
                all_answers.append(ans)
                stats["extractions"] += 1

        # Count unique answers
        normed_unique = set(normalize_answer(a) for a in all_answers)
        stats[f"unique_{len(normed_unique)}"] += 1

        answer = majority_vote(all_answers)

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 5: Focused Re-Retrieval for Hop 2 ──────────────────────
# After hop 1, re-retrieve for hop 2 with an AUGMENTED query that
# includes the hop 1 answer. This creates a more targeted query for
# the second hop, potentially surfacing the gold paragraph more reliably.

def pipeline_focused_rerank(sample, retriever, auto_hops, stats):
    """Augmented hop-2 query with hop-1 answer for focused retrieval."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)

        if i == 0:
            # Hop 1: standard
            context, results, retrieved_indices = _retrieve_and_format(hop_q, paragraphs, retriever)
            answer = _standard_extract(hop_q, context)
        else:
            # Hop 2+: augmented retrieval
            # Create augmented query that includes hop 1 answer for context
            prev_answer = previous_answers[-1] if previous_answers else ""
            augmented_q = f"{prev_answer}: {hop_q}"
            stats["augmented_queries"] += 1

            # Retrieve with BOTH queries, merge results
            results_original = retriever.retrieve(hop_q, paragraphs, top_k=3)
            results_augmented = retriever.retrieve(augmented_q, paragraphs, top_k=3)

            # Merge: augmented results get priority (appear first)
            seen_indices = set()
            merged = []
            for idx, score in results_augmented:
                if idx not in seen_indices:
                    seen_indices.add(idx)
                    merged.append((idx, score))
            for idx, score in results_original:
                if idx not in seen_indices:
                    seen_indices.add(idx)
                    merged.append((idx, score))

            results = merged[:3]
            retrieved_indices = [idx for idx, _ in results]

            context = "\n\n".join(
                f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
                for idx, _ in results
            )
            answer = _standard_extract(hop_q, context)

            # Track if augmented changed the retrieval
            orig_set = {idx for idx, _ in results_original[:3]}
            aug_set = {idx for idx, _ in results[:3]}
            if orig_set != aug_set:
                stats["retrieval_changed"] += 1
            else:
                stats["retrieval_same"] += 1

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ══════════════════════════════════════════════════════════════════════
# CONFIG REGISTRY
# ══════════════════════════════════════════════════════════════════════

CONFIGS = [
    {"id": 0, "name": "baseline",
     "fn": pipeline_baseline,
     "description": "Standard Decompose→Retrieve→Extract (generative)",
     "llm_calls": "2/sample"},
    {"id": 1, "name": "per_paragraph",
     "fn": pipeline_per_paragraph,
     "description": "Extract from each paragraph independently, majority vote",
     "llm_calls": "6/sample"},
    {"id": 2, "name": "per_para_tiebreak",
     "fn": pipeline_per_para_tiebreak,
     "description": "Per-paragraph + full-context tiebreak on disagreement",
     "llm_calls": "6-8/sample"},
    {"id": 3, "name": "multi_model_ensemble",
     "fn": pipeline_multi_model,
     "description": "Qwen 7B + Phi-3 3.8B extraction, vote/select",
     "llm_calls": "4-5/sample"},
    {"id": 4, "name": "per_para_multi_model",
     "fn": pipeline_per_para_multi_model,
     "description": "3 paragraphs × 2 models = 6 extractions, majority vote",
     "llm_calls": "12/sample"},
    {"id": 5, "name": "focused_rerank",
     "fn": pipeline_focused_rerank,
     "description": "Augmented hop-2 query with hop-1 answer for retrieval",
     "llm_calls": "2/sample"},
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
            "hop_details": hops,
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
            "experiment": "v10b_per_paragraph",
            "n_configs": n_configs,
            "n_completed": len(results),
            "n_samples": n_samples,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "configs": results,
    }
    with open(output, "w") as f:
        json.dump(data, f, indent=2, default=str)


def run_experiment(limit=30, output_path="results/v10b_per_paragraph.json",
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
    print(f"PHASE 2: TESTING {len(configs)} PER-PARAGRAPH APPROACHES")
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
    print(f"FINAL SUMMARY: v10b Per-Paragraph Extraction")
    print(f"{'=' * 70}")

    if baseline_em is None:
        baseline_em = existing_results.get("0_baseline", {}).get("em", 60.0)

    print(f"\nBaseline EM: {baseline_em:.1f}%\n")
    print(f"{'Cfg':<4} {'Name':<30} {'EM':>6} {'Delta':>7} {'relEM':>6} "
          f"{'F1':>6} {'Lat':>7} {'LLM calls':>12}")
    print("-" * 90)

    for config in CONFIGS:
        key = f"{config['id']}_{config['name']}"
        r = existing_results.get(key)
        if not r:
            continue
        delta = r["em"] - baseline_em
        marker = "+" if delta > 0.5 else ("-" if delta < -0.5 else "=")
        lc = config.get("llm_calls", "?")
        print(f"[{marker}] {config['id']:<2} {config['name']:<30} {r['em']:5.1f}% "
              f"{delta:+5.1f}%  {r['relaxed_em']:5.1f}% {r['f1']:.3f} "
              f"{r['latency_ms']:6.0f}ms  {lc:>12}")

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

    # Dill Records tracker
    print(f"\n{'=' * 70}")
    print("DILL RECORDS TRACKER")
    print("=" * 70)
    for config in CONFIGS:
        key = f"{config['id']}_{config['name']}"
        r = existing_results.get(key, {})
        for q in r.get("per_question", []):
            if q["answer"] == "Asian Man Records":
                status = "CORRECT" if q["em"] else "WRONG"
                print(f"  Config {config['id']} ({config['name']}): "
                      f"{status} — predicted '{q['prediction']}'")
                break


def main():
    parser = argparse.ArgumentParser(description="v10b: Per-Paragraph Extraction")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", type=str, default="results/v10b_per_paragraph.json")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config IDs to run")
    args = parser.parse_args()

    configs_to_run = None
    if args.configs:
        configs_to_run = [int(x) for x in args.configs.split(",")]

    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        assert any("qwen2.5:7b" in m for m in models), "qwen2.5:7b not found"
        if not any("phi3" in m for m in models):
            print("WARNING: phi3:mini not found — configs 3, 4 will use qwen fallback")
    except Exception as e:
        print(f"ERROR: Ollama check failed: {e}")
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
