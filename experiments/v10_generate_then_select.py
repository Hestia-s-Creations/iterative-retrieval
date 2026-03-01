#!/usr/bin/env python3
"""
Iterative Retrieval v10 - Generate-then-Select

The insight: Recognition is easier than recall. v9c proved MC ranking FIXED the
hardest case (Dill Records → Asian Man Records) that no other method could solve.
It failed overall (6.7% EM) because regex NER generated garbage candidate lists.

v10 fixes this: use the LLM to GENERATE candidates, then use MC to SELECT.
Two-phase extraction replaces single-shot generation.

The cognitive science parallel: humans are much better at recognizing the right
answer from a list than producing it from scratch. Same applies to LLMs —
generation and discrimination are different capabilities.

8 Configs:
  0: baseline                 — Standard Decompose→Retrieve→Extract
  1: llm_candidates_mc        — Generate candidate list, MC selection
  2: generate_then_select     — Two-prompt: list answers, then pick best
  3: aic_gated_mc             — AIC check → MC only when answer not in context
  4: self_consistency_pool    — 3x extraction at different temps → MC from pool
  5: dual_context_vote        — Extract with forward + reversed context, vote
  6: pipeline_self_consistency — 3 full pipeline runs, majority vote final answer
  7: hybrid_all               — AIC gate → self-consistency pool → MC tiebreaker
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


# ── Candidate Generation Prompt ─────────────────────────────────────

CANDIDATE_LIST_PROMPT = """Read the context and list ALL possible answers to the question. Include every name, place, organization, or fact that could plausibly answer it. One per line, numbered.

Context: {context}
Question: {question}

Possible answers:
1."""

MC_SELECT_PROMPT = """Given the context, which option best answers the question? Reply with ONLY the letter.

Context: {context}
Question: {question}

{options}

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
    """Standard 8-shot extraction. Returns raw short answer."""
    prompt = PROMPT_8SHOT.format(context=context, question=hop_q)
    response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
    return extract_short_answer(response)


def check_answer_in_context(answer, context):
    """FREE check: is the answer string present in the context?"""
    if not answer or not context:
        return False
    return normalize_answer(answer) in normalize_answer(context)


def generate_candidate_list(question, context):
    """Ask LLM to list all plausible answers from the context.

    Returns deduplicated list of candidates. This is the KEY innovation:
    LLM-generated candidates vs regex NER candidates.
    """
    prompt = CANDIDATE_LIST_PROMPT.format(context=context, question=question)
    response = ask_model(prompt, model="qwen2.5:7b", temperature=0.3, max_tokens=120)

    candidates = []
    seen_norm = set()
    for line in ("1. " + response).split('\n'):
        line = line.strip()
        m = re.match(r'^\d+[.)]\s*(.+)$', line)
        if m:
            candidate = m.group(1).strip().rstrip('.')
            # Clean up common LLM artifacts
            candidate = re.sub(r'\s*\(.*?\)\s*$', '', candidate).strip()
            if candidate and len(candidate) < 80:
                norm = normalize_answer(candidate)
                if norm and norm not in seen_norm:
                    seen_norm.add(norm)
                    candidates.append(candidate)

    return candidates


def mc_select(question, context, candidates):
    """Present candidates as MC options and let LLM pick.

    Returns selected candidate or None on parse failure.
    """
    if not candidates:
        return None

    letters = "ABCDEFGHIJ"
    n = min(len(candidates), 10)
    options = "\n".join(f"{letters[i]}. {candidates[i]}" for i in range(n))

    prompt = MC_SELECT_PROMPT.format(context=context, question=question, options=options)
    response = ask_model(prompt, model="qwen2.5:7b", temperature=0.0, max_tokens=4)

    letter = response.strip().upper()[:1]
    if letter in letters[:n]:
        idx = letters.index(letter)
        return candidates[idx]
    return None


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


# ── Config 1: LLM Candidates + MC Selection ────────────────────────
# Phase 1: Generate answer normally (recall)
# Phase 2: Generate candidate list (high recall, enumerative)
# Phase 3: MC selection from candidates + original answer
# The original answer is always included as a candidate.

def pipeline_llm_candidates_mc(sample, retriever, auto_hops, stats):
    """Generate + LLM candidate list + MC selection."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        context, results, retrieved_indices = _retrieve_and_format(hop_q, paragraphs, retriever)

        # Phase 1: Standard extraction
        baseline_answer = _standard_extract(hop_q, context)

        # Phase 2: Generate candidate list
        candidates = generate_candidate_list(hop_q, context)
        stats["total_candidates"] += len(candidates)

        # Ensure baseline answer is in the candidate list
        baseline_norm = normalize_answer(baseline_answer)
        if baseline_norm and not any(normalize_answer(c) == baseline_norm for c in candidates):
            candidates.insert(0, baseline_answer)

        if len(candidates) >= 2:
            # Phase 3: MC selection
            selected = mc_select(hop_q, context, candidates)
            if selected:
                answer = selected
                stats["mc_selected"] += 1
            else:
                answer = baseline_answer
                stats["mc_parse_fail"] += 1
        else:
            answer = baseline_answer
            stats["too_few_candidates"] += 1

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 2: Generate-then-Select (replace extraction entirely) ───
# No standard extraction at all. Two-phase:
# 1. List all possible answers (generation with high recall)
# 2. Pick the best one (discrimination)

def pipeline_generate_then_select(sample, retriever, auto_hops, stats):
    """Two-prompt extraction: list candidates, then select best."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        context, results, retrieved_indices = _retrieve_and_format(hop_q, paragraphs, retriever)

        # Phase 1: Generate candidates only
        candidates = generate_candidate_list(hop_q, context)
        stats["total_candidates"] += len(candidates)

        if len(candidates) >= 2:
            # Phase 2: MC selection
            selected = mc_select(hop_q, context, candidates)
            if selected:
                answer = selected
                stats["mc_selected"] += 1
            else:
                answer = candidates[0]  # First candidate as fallback
                stats["mc_parse_fail"] += 1
        elif len(candidates) == 1:
            answer = candidates[0]
            stats["single_candidate"] += 1
        else:
            # No candidates generated — fall back to standard extraction
            answer = _standard_extract(hop_q, context)
            stats["no_candidates_fallback"] += 1

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 3: AIC-Gated MC ─────────────────────────────────────────
# Run normal pipeline. If answer passes AIC (answer in context), keep it.
# Only trigger candidate generation + MC when AIC fails.
# Combines the FREE AIC gate (v9 best) with the MC fix (v9c breakthrough).

def pipeline_aic_gated_mc(sample, retriever, auto_hops, stats):
    """AIC check → MC selection only when answer not in context."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        context, results, retrieved_indices = _retrieve_and_format(hop_q, paragraphs, retriever)

        # Standard extraction first
        answer = _standard_extract(hop_q, context)

        # AIC gate
        if check_answer_in_context(answer, context):
            stats["aic_pass"] += 1
            # Answer is in context — trust it
        else:
            stats["aic_fail"] += 1
            # Answer NOT in context — suspicious, try MC
            candidates = generate_candidate_list(hop_q, context)
            stats["rescue_candidates"] += len(candidates)

            # Include original answer as option
            baseline_norm = normalize_answer(answer)
            if baseline_norm and not any(normalize_answer(c) == baseline_norm for c in candidates):
                candidates.append(answer)

            if len(candidates) >= 2:
                selected = mc_select(hop_q, context, candidates)
                if selected:
                    answer = selected
                    stats["mc_rescued"] += 1
                else:
                    stats["mc_parse_fail"] += 1
            else:
                stats["no_candidates"] += 1

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 4: Self-Consistency Extraction Pool ─────────────────────
# Run extraction 3 times at different temperatures.
# If all agree → keep. If disagree → MC from the pool of unique answers.

def pipeline_self_consistency_pool(sample, retriever, auto_hops, stats):
    """3x extraction at temps [0.1, 0.5, 0.8] → MC from disagreements."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []
    temps = [0.1, 0.5, 0.8]

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        context, results, retrieved_indices = _retrieve_and_format(hop_q, paragraphs, retriever)

        # Extract at multiple temperatures
        candidates = []
        seen_norm = set()
        for temp in temps:
            prompt = PROMPT_8SHOT.format(context=context, question=hop_q)
            response = ask_model(prompt, model="qwen2.5:7b", temperature=temp, max_tokens=32)
            ans = extract_short_answer(response)
            norm = normalize_answer(ans)
            if norm and norm not in seen_norm:
                seen_norm.add(norm)
                candidates.append(ans)

        stats["unique_per_hop"] += len(candidates)

        if len(candidates) == 1:
            # All agree — strong signal
            answer = candidates[0]
            stats["unanimous"] += 1
        elif len(candidates) == 2:
            # Two different answers — check if one appears more
            # Re-extract at temp=0.1 (tiebreaker)
            prompt = PROMPT_8SHOT.format(context=context, question=hop_q)
            response = ask_model(prompt, model="qwen2.5:7b", temperature=0.0, max_tokens=32)
            tiebreaker = extract_short_answer(response)
            # Vote: which candidate matches the tiebreaker?
            tb_norm = normalize_answer(tiebreaker)
            if tb_norm == normalize_answer(candidates[0]):
                answer = candidates[0]
                stats["tiebreak_first"] += 1
            elif tb_norm == normalize_answer(candidates[1]):
                answer = candidates[1]
                stats["tiebreak_second"] += 1
            else:
                # Tiebreaker is a third answer — use MC
                if tb_norm not in seen_norm:
                    candidates.append(tiebreaker)
                selected = mc_select(hop_q, context, candidates)
                answer = selected if selected else candidates[0]
                stats["tiebreak_mc"] += 1
        else:
            # 3 different answers — MC selection
            selected = mc_select(hop_q, context, candidates)
            if selected:
                answer = selected
                stats["mc_3way"] += 1
            else:
                answer = candidates[0]
                stats["mc_3way_fail"] += 1

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 5: Dual-Context Vote ────────────────────────────────────
# Extract with paragraphs in original order AND reversed order.
# Position bias means the model attends more to earlier context.
# If reversing changes the answer, the model wasn't confident.
# When they disagree, use MC.

def pipeline_dual_context_vote(sample, retriever, auto_hops, stats):
    """Extract with forward + reversed context ordering, vote on disagree."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        results = retriever.retrieve(hop_q, paragraphs, top_k=3)
        retrieved_indices = [idx for idx, _ in results]

        # Forward context
        context_fwd = "\n\n".join(
            f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
            for idx, _ in results
        )
        answer_fwd = _standard_extract(hop_q, context_fwd)

        # Reversed context
        context_rev = "\n\n".join(
            f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
            for idx, _ in reversed(results)
        )
        answer_rev = _standard_extract(hop_q, context_rev)

        if normalize_answer(answer_fwd) == normalize_answer(answer_rev):
            # Both agree — strong signal
            answer = answer_fwd
            stats["agree"] += 1
        else:
            # Disagreement — position bias present
            stats["disagree"] += 1
            candidates = [answer_fwd, answer_rev]

            # Also generate candidate list for MC
            extra = generate_candidate_list(hop_q, context_fwd)
            seen = {normalize_answer(c) for c in candidates}
            for c in extra:
                if normalize_answer(c) not in seen:
                    candidates.append(c)
                    seen.add(normalize_answer(c))

            if len(candidates) >= 2:
                selected = mc_select(hop_q, context_fwd, candidates)
                if selected:
                    answer = selected
                    stats["mc_resolved"] += 1
                else:
                    answer = answer_fwd
                    stats["mc_fail_fwd"] += 1
            else:
                answer = answer_fwd

        hop_details.append(_build_hop_detail(i, hop_q, answer, gold_decomp, retrieved_indices))
        previous_answers.append(answer)

    return (previous_answers[-1] if previous_answers else ""), hop_details


# ── Config 6: Pipeline Self-Consistency ────────────────────────────
# Run the FULL pipeline 3 times with different decompositions (temp=0.7).
# Majority vote on the final answer.
# Attacks the decomposition quality bottleneck.

def pipeline_self_consistency(sample, retriever, auto_hops, stats):
    """3 full pipeline runs with different decompositions, majority vote."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    question = sample["question"]

    final_answers = []

    for run_idx in range(3):
        # Generate fresh decomposition for runs 1 and 2 (run 0 uses cached)
        if run_idx == 0:
            hops = auto_hops
        else:
            sub_qs = decompose_with_qwen(question, temperature=0.7)
            hops = [{"question": sq} for sq in sub_qs]

        previous_answers = []
        for i, hop in enumerate(hops):
            hop_q = _prepare_hop_question(hop, previous_answers)
            context, results, retrieved_indices = _retrieve_and_format(hop_q, paragraphs, retriever)
            answer = _standard_extract(hop_q, context)
            previous_answers.append(answer)

        final = previous_answers[-1] if previous_answers else ""
        final_answers.append(final)

    # Majority vote
    normed = [normalize_answer(a) for a in final_answers]
    counter = Counter(normed)
    winner_norm, count = counter.most_common(1)[0]

    if count >= 2:
        stats["majority"] += 1
        # Find original-cased version
        for a, n in zip(final_answers, normed):
            if n == winner_norm:
                answer = a
                break
    else:
        # All different — MC from the 3 candidates
        stats["no_majority"] += 1
        # Retrieve for the original question to get context for MC
        context, _, _ = _retrieve_and_format(question, paragraphs, retriever, top_k=5)
        selected = mc_select(question, context, final_answers)
        if selected:
            answer = selected
            stats["mc_tiebreak"] += 1
        else:
            answer = final_answers[0]  # First run as fallback
            stats["fallback_first"] += 1

    # Build hop details from the winning run (approximate)
    hop_details = [{
        "hop": 1, "question": question, "gold_answer": sample["answer"],
        "predicted": answer,
        "em": exact_match(answer, sample["answer"], sample.get("answer_aliases", [])),
        "gold_retrieved": True,  # Approximate — we ran 3 pipelines
    }]

    return answer, hop_details


# ── Config 7: Hybrid All ───────────────────────────────────────────
# The best of everything:
# 1. Standard extraction
# 2. AIC gate (FREE)
# 3. If AIC fails → self-consistency pool (3 temps)
# 4. If pool disagrees → MC from LLM-generated candidates + pool
# Progressive escalation: cheap checks first, expensive only when needed.

def pipeline_hybrid_all(sample, retriever, auto_hops, stats):
    """Progressive escalation: extract → AIC → self-consistency → MC."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(auto_hops):
        hop_q = _prepare_hop_question(hop, previous_answers)
        context, results, retrieved_indices = _retrieve_and_format(hop_q, paragraphs, retriever)

        # Level 1: Standard extraction
        answer = _standard_extract(hop_q, context)

        # Level 2: AIC gate (FREE)
        if check_answer_in_context(answer, context):
            stats["l2_pass"] += 1
        else:
            stats["l2_fail"] += 1

            # Level 3: Self-consistency (2 more extractions)
            prompt = PROMPT_8SHOT.format(context=context, question=hop_q)
            alt_answers = [answer]
            seen = {normalize_answer(answer)}
            for temp in [0.5, 0.8]:
                resp = ask_model(prompt, model="qwen2.5:7b", temperature=temp, max_tokens=32)
                alt = extract_short_answer(resp)
                norm = normalize_answer(alt)
                if norm and norm not in seen:
                    seen.add(norm)
                    alt_answers.append(alt)

            # Check if any alternative passes AIC
            aic_pass = [a for a in alt_answers if check_answer_in_context(a, context)]
            if len(aic_pass) == 1:
                answer = aic_pass[0]
                stats["l3_aic_rescue"] += 1
            elif len(aic_pass) > 1:
                # Multiple AIC-passing answers — MC to pick
                selected = mc_select(hop_q, context, aic_pass)
                answer = selected if selected else aic_pass[0]
                stats["l3_mc_aic"] += 1
            else:
                # Level 4: Nothing passes AIC — generate candidates + MC
                candidates = generate_candidate_list(hop_q, context)
                # Add all extraction attempts
                for a in alt_answers:
                    norm = normalize_answer(a)
                    if norm and not any(normalize_answer(c) == norm for c in candidates):
                        candidates.append(a)

                if len(candidates) >= 2:
                    selected = mc_select(hop_q, context, candidates)
                    answer = selected if selected else candidates[0]
                    stats["l4_mc_full"] += 1
                else:
                    stats["l4_no_candidates"] += 1

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
    {"id": 1, "name": "llm_candidates_mc",
     "fn": pipeline_llm_candidates_mc,
     "description": "Generate + LLM candidate list + MC selection",
     "llm_calls": "4/hop"},
    {"id": 2, "name": "generate_then_select",
     "fn": pipeline_generate_then_select,
     "description": "Two-prompt: list candidates → MC pick (no 8-shot)",
     "llm_calls": "3/hop"},
    {"id": 3, "name": "aic_gated_mc",
     "fn": pipeline_aic_gated_mc,
     "description": "AIC gate → MC rescue only when answer not in context",
     "llm_calls": "2-4/hop"},
    {"id": 4, "name": "self_consistency_pool",
     "fn": pipeline_self_consistency_pool,
     "description": "3x temp extraction → MC from disagreements",
     "llm_calls": "3-5/hop"},
    {"id": 5, "name": "dual_context_vote",
     "fn": pipeline_dual_context_vote,
     "description": "Forward + reversed context → vote on disagreement",
     "llm_calls": "2-4/hop"},
    {"id": 6, "name": "pipeline_self_consistency",
     "fn": pipeline_self_consistency,
     "description": "3 full pipeline runs (different decomps) → majority vote",
     "llm_calls": "9/sample"},
    {"id": 7, "name": "hybrid_all",
     "fn": pipeline_hybrid_all,
     "description": "Progressive: extract → AIC → self-consistency → MC",
     "llm_calls": "2-6/hop"},
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

        # Progress
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
            "experiment": "v10_generate_then_select",
            "n_configs": n_configs,
            "n_completed": len(results),
            "n_samples": n_samples,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "configs": results,
    }
    with open(output, "w") as f:
        json.dump(data, f, indent=2, default=str)


def run_experiment(limit=30, output_path="results/v10_generate_then_select.json",
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
    print(f"PHASE 2: TESTING {len(configs)} GENERATE-THEN-SELECT APPROACHES")
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
    print(f"FINAL SUMMARY: v10 Generate-then-Select")
    print(f"{'=' * 70}")

    if baseline_em is None:
        baseline_em = existing_results.get("0_baseline", {}).get("em", 60.0)

    print(f"\nBaseline EM: {baseline_em:.1f}%\n")
    print(f"{'Cfg':<4} {'Name':<30} {'EM':>6} {'Delta':>7} {'relEM':>6} "
          f"{'F1':>6} {'Lat':>7} {'LLM calls':>10}")
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
        for qid, old, new, gold in diff[:5]:
            print(f"    [DIFF]  '{old}' → '{new}'  (gold: '{gold}')")

    # Dill Records tracker
    print(f"\n{'=' * 70}")
    print("DILL RECORDS TRACKER (the case only MC has ever fixed)")
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
    parser = argparse.ArgumentParser(description="v10: Generate-then-Select")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", type=str, default="results/v10_generate_then_select.json")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config IDs to run (e.g., '0,1,3')")
    args = parser.parse_args()

    configs_to_run = None
    if args.configs:
        configs_to_run = [int(x) for x in args.configs.split(",")]

    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        assert any("qwen2.5:7b" in m for m in models), "qwen2.5:7b not found"
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
