#!/usr/bin/env python3
"""
Iterative Retrieval Experiment for Tiny Mind.

THE HYPOTHESIS:
Multi-hop QA fails because the MODEL tries to chain reasoning in a single pass.
If the SYSTEM decomposes into single-hop questions and chains results,
Phi-3 (95% accurate on 1-hop) should perform much better.

CONFIGURATIONS:
1. SINGLE_PASS: Standard approach - full question + all 20 paragraphs
2. GOLD_ORACLE: Gold decomposition + gold supporting paragraph per hop
3. GOLD_DECOMP_LOCAL: Gold decomposition + retrieve from 20 paragraphs per hop
4. GOLD_DECOMP_DB: Gold decomposition + retrieve from Tiny Mind DB per hop

This tests whether the ARCHITECTURE (system-level chaining) beats the MODEL
(single-pass multi-hop reasoning). If GOLD_ORACLE >> SINGLE_PASS, the thesis holds.
"""

import sys
import time
import json
import re
import string
import argparse
import requests
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict

import numpy as np



# ── Metrics ──────────────────────────────────────────────────────────

def normalize_answer(s):
    """Lower text, remove punctuation, articles, extra whitespace."""
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


def best_f1(prediction, ground_truth, aliases=None):
    scores = [f1_score(prediction, ground_truth)]
    for alias in (aliases or []):
        scores.append(f1_score(prediction, alias))
    return max(scores)


# ── LLM Interface ───────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434"

def ask_phi3(prompt, temperature=0.1, max_tokens=64):
    """Ask Phi-3 a question. Returns the response text."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": "phi3:mini",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            },
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as e:
        return f"ERROR: {e}"


def extract_short_answer(response):
    """Extract a short factual answer from model response."""
    # Take first line, strip common prefixes
    answer = response.split('\n')[0].strip()
    # Remove "The answer is", "Answer:", etc.
    for prefix in ["the answer is", "answer:", "a:", "the "]:
        if answer.lower().startswith(prefix):
            answer = answer[len(prefix):].strip()
    # Remove trailing period
    answer = answer.rstrip('.')
    # If still long, take first clause
    if len(answer) > 100:
        for sep in [',', ';', '(', ' - ']:
            if sep in answer:
                answer = answer[:answer.index(sep)].strip()
                break
    return answer


# ── Hop Question Formatting ─────────────────────────────────────────

def format_hop_question(hop_question, previous_answers):
    """
    Convert MuSiQue hop notation to natural language.

    MuSiQue uses:
    - "Green >> performer" → "Who/what is the performer of Green?"
    - "#1 >> spouse" → "Who/what is the spouse of [answer from hop 1]?"
    - "#2 >> country" → "What country is [answer from hop 2] in?"
    """
    q = hop_question

    # Substitute #N references with previous answers
    for i, answer in enumerate(previous_answers, 1):
        q = q.replace(f"#{i}", answer)

    # Convert "X >> Y" to natural question
    if ">>" in q:
        parts = q.split(">>")
        subject = parts[0].strip()
        relation = parts[1].strip()
        q = f"What is the {relation} of {subject}?"

    return q


# ── Experiment Configurations ────────────────────────────────────────

def run_single_pass(sample, verbose=False):
    """Config 1: Standard single-pass. All paragraphs + full question."""
    question = sample["question"]

    # Build context from all 20 paragraphs
    paras = sample["paragraphs"]
    context_parts = []
    for para in paras:
        title = para.get("title", "")
        text = para.get("paragraph_text", "")
        context_parts.append(f"[{title}] {text}")

    # Limit context to avoid overloading Phi-3
    context = "\n\n".join(context_parts)
    if len(context) > 3000:
        context = context[:3000] + "..."

    prompt = f"""Answer the question using ONLY the provided context. Give a brief, direct answer with just the name or fact requested. Do not explain.

Context:
{context}

Question: {question}
Answer:"""

    response = ask_phi3(prompt, temperature=0.1, max_tokens=64)
    answer = extract_short_answer(response)

    if verbose:
        print(f"    [SINGLE_PASS] Raw: {response[:80]}")
        print(f"    [SINGLE_PASS] Extracted: {answer}")

    return answer


def run_gold_oracle(sample, verbose=False):
    """Config 2: Gold decomposition + gold supporting paragraphs per hop."""
    decomposition = sample["question_decomposition"]
    paragraphs = sample["paragraphs"]

    previous_answers = []
    hop_details = []

    for i, hop in enumerate(decomposition):
        hop_q = format_hop_question(hop["question"], previous_answers)
        support_idx = hop["paragraph_support_idx"]
        support_para = paragraphs[support_idx]
        context = f"[{support_para['title']}] {support_para['paragraph_text']}"

        prompt = f"""Extract the answer from the context. Give ONLY the name, number, or fact. No explanation.

Context: {context}

Question: {hop_q}
Answer:"""

        response = ask_phi3(prompt, temperature=0.1, max_tokens=32)
        answer = extract_short_answer(response)

        hop_details.append({
            "hop": i + 1,
            "question": hop_q,
            "gold_answer": hop["answer"],
            "predicted": answer,
            "em": exact_match(answer, hop["answer"]),
            "f1": f1_score(answer, hop["answer"]),
            "support_title": support_para["title"],
        })

        if verbose:
            status = "✓" if hop_details[-1]["em"] else "✗"
            print(f"    [ORACLE] Hop {i+1}: {hop_q[:50]}...")
            print(f"      {status} Expected: {hop['answer']}, Got: {answer}")

        # Use model's answer for chaining (not gold)
        previous_answers.append(answer)

    return previous_answers[-1] if previous_answers else "", hop_details


def run_gold_oracle_gold_chain(sample, verbose=False):
    """Config 2b: Gold decomposition + gold paragraphs + GOLD answer chaining.

    This uses the gold answer for each hop to feed into the next hop,
    isolating per-hop extraction accuracy from error cascading.
    """
    decomposition = sample["question_decomposition"]
    paragraphs = sample["paragraphs"]

    previous_answers_gold = []
    previous_answers_pred = []
    hop_details = []

    for i, hop in enumerate(decomposition):
        # Use GOLD answers for substitution (no error cascade)
        hop_q = format_hop_question(hop["question"], previous_answers_gold)
        support_idx = hop["paragraph_support_idx"]
        support_para = paragraphs[support_idx]
        context = f"[{support_para['title']}] {support_para['paragraph_text']}"

        prompt = f"""Extract the answer from the context. Give ONLY the name, number, or fact. No explanation.

Context: {context}

Question: {hop_q}
Answer:"""

        response = ask_phi3(prompt, temperature=0.1, max_tokens=32)
        answer = extract_short_answer(response)

        hop_details.append({
            "hop": i + 1,
            "question": hop_q,
            "gold_answer": hop["answer"],
            "predicted": answer,
            "em": exact_match(answer, hop["answer"]),
            "f1": f1_score(answer, hop["answer"]),
        })

        if verbose:
            status = "✓" if hop_details[-1]["em"] else "✗"
            print(f"    [ORACLE-GOLD] Hop {i+1}: {hop_q[:50]}...")
            print(f"      {status} Expected: {hop['answer']}, Got: {answer}")

        previous_answers_gold.append(hop["answer"])  # Gold for chaining
        previous_answers_pred.append(answer)  # Actual prediction

    return previous_answers_pred[-1] if previous_answers_pred else "", hop_details


def run_gold_decomp_local(sample, verbose=False):
    """Config 3: Gold decomposition + retrieve from the 20 sample paragraphs per hop.

    Uses simple keyword matching to find the most relevant paragraph
    for each hop question from the sample's 20 paragraphs.
    """
    decomposition = sample["question_decomposition"]
    paragraphs = sample["paragraphs"]

    previous_answers = []
    hop_details = []

    for i, hop in enumerate(decomposition):
        hop_q = format_hop_question(hop["question"], previous_answers)

        # Simple retrieval: score each paragraph by keyword overlap
        query_words = set(normalize_answer(hop_q).split())
        scored = []
        for j, para in enumerate(paragraphs):
            text = para.get("paragraph_text", "")
            title = para.get("title", "")
            para_words = set(normalize_answer(f"{title} {text}").split())
            overlap = len(query_words & para_words)
            scored.append((overlap, j, para))

        # Take top 3 paragraphs
        scored.sort(reverse=True, key=lambda x: x[0])
        top_paras = scored[:3]

        context_parts = []
        for _, _, para in top_paras:
            context_parts.append(f"[{para['title']}] {para['paragraph_text']}")
        context = "\n\n".join(context_parts)

        # Check if gold paragraph was retrieved
        gold_idx = hop["paragraph_support_idx"]
        retrieved_indices = [s[1] for s in top_paras]
        gold_retrieved = gold_idx in retrieved_indices

        prompt = f"""Extract the answer from the context. Give ONLY the name, number, or fact. No explanation.

Context: {context}

Question: {hop_q}
Answer:"""

        response = ask_phi3(prompt, temperature=0.1, max_tokens=32)
        answer = extract_short_answer(response)

        hop_details.append({
            "hop": i + 1,
            "question": hop_q,
            "gold_answer": hop["answer"],
            "predicted": answer,
            "em": exact_match(answer, hop["answer"]),
            "f1": f1_score(answer, hop["answer"]),
            "gold_retrieved": gold_retrieved,
            "retrieved_indices": retrieved_indices,
        })

        if verbose:
            status = "✓" if hop_details[-1]["em"] else "✗"
            retrieval_status = "✓" if gold_retrieved else "✗"
            print(f"    [LOCAL] Hop {i+1}: {hop_q[:50]}...")
            print(f"      {status} Expected: {hop['answer']}, Got: {answer} (retrieval: {retrieval_status})")

        previous_answers.append(answer)

    return previous_answers[-1] if previous_answers else "", hop_details


# ── Main Experiment ──────────────────────────────────────────────────

@dataclass
class ConfigResult:
    name: str
    em_scores: list = field(default_factory=list)
    f1_scores: list = field(default_factory=list)
    latencies: list = field(default_factory=list)
    hop_details: list = field(default_factory=list)  # per-question hop breakdowns
    per_question: list = field(default_factory=list)

    @property
    def em_rate(self):
        return sum(self.em_scores) / len(self.em_scores) * 100 if self.em_scores else 0

    @property
    def mean_f1(self):
        return sum(self.f1_scores) / len(self.f1_scores) if self.f1_scores else 0

    @property
    def mean_latency(self):
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0


def run_experiment(limit=50, verbose=True):
    """Run all configurations on MuSiQue validation set."""
    from datasets import load_dataset

    print("Loading MuSiQue validation set...")
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")

    # Filter to answerable questions only
    samples = [s for s in ds if s.get("answerable", True)][:limit]
    print(f"Testing on {len(samples)} answerable questions\n")

    configs = {
        "SINGLE_PASS": ConfigResult("SINGLE_PASS"),
        "GOLD_ORACLE": ConfigResult("GOLD_ORACLE"),
        "GOLD_ORACLE_GOLD_CHAIN": ConfigResult("GOLD_ORACLE_GOLD_CHAIN"),
        "GOLD_DECOMP_LOCAL": ConfigResult("GOLD_DECOMP_LOCAL"),
    }

    for i, sample in enumerate(samples):
        question = sample["question"]
        answer = sample["answer"]
        aliases = sample.get("answer_aliases", [])
        n_hops = len(sample.get("question_decomposition", []))

        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(samples)}] ({n_hops}-hop) {question}")
        print(f"Expected: {answer}")
        if aliases:
            print(f"Aliases: {aliases}")
        print(f"{'='*70}")

        # Config 1: Single Pass
        print("\n  >> SINGLE_PASS")
        t0 = time.time()
        pred_sp = run_single_pass(sample, verbose=verbose)
        lat_sp = (time.time() - t0) * 1000
        em_sp = exact_match(pred_sp, answer, aliases)
        f1_sp = best_f1(pred_sp, answer, aliases)
        configs["SINGLE_PASS"].em_scores.append(em_sp)
        configs["SINGLE_PASS"].f1_scores.append(f1_sp)
        configs["SINGLE_PASS"].latencies.append(lat_sp)
        configs["SINGLE_PASS"].per_question.append({
            "id": sample["id"], "predicted": pred_sp,
            "em": em_sp, "f1": f1_sp, "n_hops": n_hops
        })
        status = "✓" if em_sp else "✗"
        print(f"  {status} Predicted: {pred_sp} (F1={f1_sp:.2f}, {lat_sp:.0f}ms)")

        # Config 2: Gold Oracle (model chaining)
        print("\n  >> GOLD_ORACLE")
        t0 = time.time()
        pred_go, hops_go = run_gold_oracle(sample, verbose=verbose)
        lat_go = (time.time() - t0) * 1000
        em_go = exact_match(pred_go, answer, aliases)
        f1_go = best_f1(pred_go, answer, aliases)
        configs["GOLD_ORACLE"].em_scores.append(em_go)
        configs["GOLD_ORACLE"].f1_scores.append(f1_go)
        configs["GOLD_ORACLE"].latencies.append(lat_go)
        configs["GOLD_ORACLE"].hop_details.append(hops_go)
        configs["GOLD_ORACLE"].per_question.append({
            "id": sample["id"], "predicted": pred_go,
            "em": em_go, "f1": f1_go, "n_hops": n_hops,
            "hop_details": hops_go
        })
        status = "✓" if em_go else "✗"
        print(f"  {status} Predicted: {pred_go} (F1={f1_go:.2f}, {lat_go:.0f}ms)")

        # Config 2b: Gold Oracle with Gold Chain (no error cascade)
        print("\n  >> GOLD_ORACLE_GOLD_CHAIN")
        t0 = time.time()
        pred_gc, hops_gc = run_gold_oracle_gold_chain(sample, verbose=verbose)
        lat_gc = (time.time() - t0) * 1000
        em_gc = exact_match(pred_gc, answer, aliases)
        f1_gc = best_f1(pred_gc, answer, aliases)
        configs["GOLD_ORACLE_GOLD_CHAIN"].em_scores.append(em_gc)
        configs["GOLD_ORACLE_GOLD_CHAIN"].f1_scores.append(f1_gc)
        configs["GOLD_ORACLE_GOLD_CHAIN"].latencies.append(lat_gc)
        configs["GOLD_ORACLE_GOLD_CHAIN"].hop_details.append(hops_gc)
        configs["GOLD_ORACLE_GOLD_CHAIN"].per_question.append({
            "id": sample["id"], "predicted": pred_gc,
            "em": em_gc, "f1": f1_gc, "n_hops": n_hops,
            "hop_details": hops_gc
        })
        status = "✓" if em_gc else "✗"
        print(f"  {status} Predicted: {pred_gc} (F1={f1_gc:.2f}, {lat_gc:.0f}ms)")

        # Config 3: Gold Decomposition + Local Retrieval
        print("\n  >> GOLD_DECOMP_LOCAL")
        t0 = time.time()
        pred_gl, hops_gl = run_gold_decomp_local(sample, verbose=verbose)
        lat_gl = (time.time() - t0) * 1000
        em_gl = exact_match(pred_gl, answer, aliases)
        f1_gl = best_f1(pred_gl, answer, aliases)
        configs["GOLD_DECOMP_LOCAL"].em_scores.append(em_gl)
        configs["GOLD_DECOMP_LOCAL"].f1_scores.append(f1_gl)
        configs["GOLD_DECOMP_LOCAL"].latencies.append(lat_gl)
        configs["GOLD_DECOMP_LOCAL"].hop_details.append(hops_gl)
        configs["GOLD_DECOMP_LOCAL"].per_question.append({
            "id": sample["id"], "predicted": pred_gl,
            "em": em_gl, "f1": f1_gl, "n_hops": n_hops,
            "hop_details": hops_gl
        })
        status = "✓" if em_gl else "✗"
        print(f"  {status} Predicted: {pred_gl} (F1={f1_gl:.2f}, {lat_gl:.0f}ms)")

        # Running totals
        print(f"\n  --- Running Totals ({i+1} questions) ---")
        for name, cfg in configs.items():
            print(f"    {name:25s}: EM={cfg.em_rate:5.1f}%, F1={cfg.mean_f1:.3f}")

    return configs, samples


def analyze_results(configs, samples):
    """Deep analysis of results."""
    print("\n" + "=" * 70)
    print("ITERATIVE RETRIEVAL EXPERIMENT - FINAL RESULTS")
    print("=" * 70)

    print(f"\n{'Config':<28s} {'EM':>6s} {'F1':>6s} {'Latency':>10s}")
    print("-" * 55)
    for name, cfg in configs.items():
        print(f"{name:<28s} {cfg.em_rate:5.1f}% {cfg.mean_f1:.3f} {cfg.mean_latency:8.0f}ms")

    # Improvement analysis
    sp_em = configs["SINGLE_PASS"].em_rate
    go_em = configs["GOLD_ORACLE"].em_rate
    gc_em = configs["GOLD_ORACLE_GOLD_CHAIN"].em_rate
    gl_em = configs["GOLD_DECOMP_LOCAL"].em_rate

    print(f"\n--- Improvement over SINGLE_PASS ({sp_em:.1f}% EM) ---")
    for name in ["GOLD_ORACLE", "GOLD_ORACLE_GOLD_CHAIN", "GOLD_DECOMP_LOCAL"]:
        cfg = configs[name]
        delta = cfg.em_rate - sp_em
        print(f"  {name}: {'+' if delta >= 0 else ''}{delta:.1f}% EM")

    # Per-hop analysis for GOLD_ORACLE
    print("\n--- Per-Hop Accuracy (GOLD_ORACLE) ---")
    hop_em = defaultdict(list)
    hop_f1 = defaultdict(list)
    for question_hops in configs["GOLD_ORACLE"].hop_details:
        for hop in question_hops:
            hop_em[hop["hop"]].append(hop["em"])
            hop_f1[hop["hop"]].append(hop["f1"])

    for h in sorted(hop_em.keys()):
        em = sum(hop_em[h]) / len(hop_em[h]) * 100
        f1 = sum(hop_f1[h]) / len(hop_f1[h])
        print(f"  Hop {h}: EM={em:.1f}%, F1={f1:.3f} (n={len(hop_em[h])})")

    # Per-hop analysis for GOLD_ORACLE_GOLD_CHAIN
    print("\n--- Per-Hop Accuracy (GOLD_ORACLE_GOLD_CHAIN - no error cascade) ---")
    hop_em = defaultdict(list)
    hop_f1 = defaultdict(list)
    for question_hops in configs["GOLD_ORACLE_GOLD_CHAIN"].hop_details:
        for hop in question_hops:
            hop_em[hop["hop"]].append(hop["em"])
            hop_f1[hop["hop"]].append(hop["f1"])

    for h in sorted(hop_em.keys()):
        em = sum(hop_em[h]) / len(hop_em[h]) * 100
        f1 = sum(hop_f1[h]) / len(hop_f1[h])
        print(f"  Hop {h}: EM={em:.1f}%, F1={f1:.3f} (n={len(hop_em[h])})")

    # Retrieval analysis for GOLD_DECOMP_LOCAL
    print("\n--- Retrieval Analysis (GOLD_DECOMP_LOCAL) ---")
    total_hops = 0
    gold_retrieved = 0
    for question_hops in configs["GOLD_DECOMP_LOCAL"].hop_details:
        for hop in question_hops:
            total_hops += 1
            if hop.get("gold_retrieved", False):
                gold_retrieved += 1

    if total_hops > 0:
        print(f"  Gold paragraph retrieved: {gold_retrieved}/{total_hops} ({gold_retrieved/total_hops*100:.1f}%)")

    # Error cascade analysis
    print("\n--- Error Cascade Analysis ---")
    cascade_failures = 0
    total_multi_hop = 0
    for q_hops in configs["GOLD_ORACLE"].hop_details:
        if len(q_hops) > 1:
            total_multi_hop += 1
            # Check if first hop was right but final was wrong
            if q_hops[0]["em"] and not q_hops[-1]["em"]:
                cascade_failures += 1

    if total_multi_hop > 0:
        print(f"  First hop correct → final wrong: {cascade_failures}/{total_multi_hop} ({cascade_failures/total_multi_hop*100:.1f}%)")

    # By n_hops
    print("\n--- By Number of Hops ---")
    for name in configs:
        print(f"\n  {name}:")
        by_hops = defaultdict(list)
        for pq in configs[name].per_question:
            by_hops[pq["n_hops"]].append(pq)
        for n in sorted(by_hops.keys()):
            qs = by_hops[n]
            em = sum(q["em"] for q in qs) / len(qs) * 100
            f1 = sum(q["f1"] for q in qs) / len(qs)
            print(f"    {n}-hop: EM={em:.1f}%, F1={f1:.3f} (n={len(qs)})")

    return {
        "configs": {name: {
            "em": cfg.em_rate,
            "f1": cfg.mean_f1,
            "latency_ms": cfg.mean_latency,
            "per_question": cfg.per_question,
        } for name, cfg in configs.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Iterative Retrieval Experiment")
    parser.add_argument("--limit", type=int, default=30, help="Number of questions")
    parser.add_argument("--output", type=str, default="evaluation/results/iterative_retrieval.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Verify Ollama is running
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if "phi3:mini" not in models:
            print("ERROR: phi3:mini not loaded. Run: ollama pull phi3:mini")
            sys.exit(1)
        print(f"Ollama OK. Models: {models}")
    except Exception as e:
        print(f"ERROR: Ollama not responding at {OLLAMA_URL}: {e}")
        sys.exit(1)

    configs, samples = run_experiment(
        limit=args.limit,
        verbose=not args.quiet
    )

    results = analyze_results(configs, samples)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
