#!/usr/bin/env python3
"""
Iterative Retrieval v4 - Rule-based decomposition (no gold decompositions).

The key insight from v3: Phi-3 can't decompose questions (6.7% EM).
But MuSiQue questions follow rigid patterns. We can parse them instead.

Changes from v3:
1. RULE_DECOMPOSE: Rule-based question decomposition using NLP patterns
2. Uses a larger LLM (qwen2.5:7b) for decomposition ONLY, Phi-3 for extraction
3. Tests the full autonomous pipeline: decompose → retrieve → extract → chain
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
from dataclasses import dataclass, field



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


# ── LLM ─────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434"

def ask_model(prompt, model="phi3:mini", temperature=0.1, max_tokens=32, retries=2):
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
                timeout=300  # 5 min timeout for model swapping under memory pressure
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            if attempt < retries:
                print(f"    [RETRY] {model} timed out, attempt {attempt+2}/{retries+1}...")
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

def format_hop_question(hop_question, previous_answers):
    q = hop_question
    for i, answer in enumerate(previous_answers, 1):
        q = q.replace(f"#{i}", answer)

    if ">>" in q:
        parts = q.split(">>")
        subject = parts[0].strip()
        relation = parts[1].strip().lower()

        relation_map = {
            "performer": f"Who performed {subject}?",
            "author": f"Who is the author of {subject}?",
            "spouse": f"Who is the spouse of {subject}?",
            "child": f"Who is the child of {subject}?",
            "father": f"Who is the father of {subject}?",
            "mother": f"Who is the mother of {subject}?",
            "place of birth": f"Where was {subject} born?",
            "headquarters location": f"Where is {subject} headquartered?",
            "record label": f"What record label is {subject} signed to?",
            "educated at": f"Where was {subject} educated?",
            "employer": f"Who employs {subject}?",
            "country": f"What country is {subject} in?",
            "genre": f"What genre is {subject}?",
            "award received": f"What award did {subject} receive?",
            "founded by": f"Who founded {subject}?",
            "owned by": f"Who owns {subject}?",
            "manufacturer": f"Who manufactured {subject}?",
            "distributed by": f"Who distributed {subject}?",
            "producer": f"Who produced {subject}?",
            "director": f"Who directed {subject}?",
            "notable work": f"What is a notable work by {subject}?",
            "instrument": f"What instrument does {subject} play?",
            "has part": f"Who is a member of {subject}?",
            "capital": f"What is the capital of {subject}?",
            "shares border with": f"What borders {subject}?",
        }
        for key, template in relation_map.items():
            if key in relation:
                return template
        if "located in" in relation or "administrative territorial" in relation:
            return f"What administrative region is {subject} located in?"
        return f"What is the {relation} of {subject}?"
    return q


FEW_SHOT_TEMPLATE = """Answer the question using the context below. Give ONLY the specific name, place, or fact asked for. Be as concise as possible - just the core answer, no extra details.

Examples:

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

Now answer this question:

Context: {context}
Question: {question}
Answer:"""


# ── Qwen2.5 Decomposition ───────────────────────────────────────────

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

Question: {question}
1."""


def decompose_with_qwen(question):
    """Use Qwen2.5 7B to decompose — larger model for planning, smaller for extraction."""
    response = ask_model(
        DECOMPOSE_TEMPLATE.format(question=question),
        model="qwen2.5:7b",
        temperature=0.1,
        max_tokens=80
    )

    # Parse: response starts after "1. " (already in template)
    full = "1. " + response
    sub_questions = []
    for line in full.split('\n'):
        line = line.strip()
        m = re.match(r'^\d+\.\s+(.+)$', line)
        if m:
            q = m.group(1).strip()
            # Clean up common artifacts
            q = q.rstrip('?').strip() + '?'
            sub_questions.append(q)

    # Only return first 2 (we expect 2-hop)
    if len(sub_questions) >= 2:
        return sub_questions[:2]
    elif len(sub_questions) == 1:
        return [sub_questions[0], question]  # Fallback
    else:
        return [question]  # Total fallback


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


# ── Configs ──────────────────────────────────────────────────────────

def run_qwen_decompose(sample, retriever, verbose=False):
    """Qwen decomposes, Phi-3 extracts. Best of both worlds."""
    question = sample["question"]
    paragraphs = sample["paragraphs"]

    # Step 1: Qwen decomposes
    sub_questions = decompose_with_qwen(question)

    if verbose:
        print(f"    [QWEN] Decomposition: {sub_questions}")

    # Step 2: Phi-3 answers each sub-question iteratively
    previous_answers = []
    hop_details = []

    for i, sq in enumerate(sub_questions):
        # Substitute #N references
        hop_q = sq
        for j, ans in enumerate(previous_answers, 1):
            hop_q = hop_q.replace(f"#{j}", ans)

        results = retriever.retrieve(hop_q, paragraphs, top_k=3)
        retrieved_indices = [idx for idx, _ in results]
        context_parts = [f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
                        for idx, _ in results]
        context = "\n\n".join(context_parts)

        # Check if gold paragraph was retrieved (for analysis)
        gold_decomp = sample.get("question_decomposition", [])
        gold_idx = gold_decomp[i]["paragraph_support_idx"] if i < len(gold_decomp) else -1
        gold_retrieved = gold_idx in retrieved_indices

        prompt = FEW_SHOT_TEMPLATE.format(context=context, question=hop_q)
        response = ask_model(prompt, model="phi3:mini", temperature=0.1, max_tokens=32)
        raw_answer = extract_short_answer(response)

        # Compare with gold if available
        gold_answer = gold_decomp[i]["answer"] if i < len(gold_decomp) else "N/A"
        em = exact_match(raw_answer, gold_answer) if gold_answer != "N/A" else False

        hop_details.append({
            "hop": i + 1, "question": hop_q,
            "gold_answer": gold_answer,
            "predicted": raw_answer,
            "em": em,
            "gold_retrieved": gold_retrieved,
        })

        if verbose:
            s = "+" if em else "-"
            retr = "R" if gold_retrieved else "X"
            print(f"    [{s}{retr}] Hop {i+1}: {hop_q[:60]}...")
            print(f"        Got: {raw_answer} (gold: {gold_answer})")

        previous_answers.append(raw_answer)

    final = raw_answer if hop_details else ""
    return final, hop_details, sub_questions


def run_gold_decomp_embed(sample, retriever, verbose=False):
    """Gold decomposition + embedding retrieval (v2 baseline for comparison)."""
    decomposition = sample["question_decomposition"]
    paragraphs = sample["paragraphs"]

    previous_answers = []
    hop_details = []

    for i, hop in enumerate(decomposition):
        hop_q = format_hop_question(hop["question"], previous_answers)

        results = retriever.retrieve(hop_q, paragraphs, top_k=3)
        retrieved_indices = [idx for idx, _ in results]
        context_parts = [f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
                        for idx, _ in results]
        context = "\n\n".join(context_parts)

        gold_idx = hop["paragraph_support_idx"]
        gold_retrieved = gold_idx in retrieved_indices

        prompt = FEW_SHOT_TEMPLATE.format(context=context, question=hop_q)
        response = ask_model(prompt, model="phi3:mini", temperature=0.1, max_tokens=32)
        raw_answer = extract_short_answer(response)

        em = exact_match(raw_answer, hop["answer"])
        hop_details.append({
            "hop": i + 1, "question": hop_q,
            "gold_answer": hop["answer"],
            "predicted": raw_answer,
            "em": em,
            "gold_retrieved": gold_retrieved,
        })

        if verbose:
            s = "+" if em else "-"
            print(f"    [{s}] Hop {i+1}: {hop_q[:60]}...")
            print(f"        Got: {raw_answer} (gold: {hop['answer']})")

        previous_answers.append(raw_answer)

    final = raw_answer if hop_details else ""
    return final, hop_details


# ── Main ─────────────────────────────────────────────────────────────

@dataclass
class ConfigResult:
    name: str
    em_scores: list = field(default_factory=list)
    relaxed_em_scores: list = field(default_factory=list)
    f1_scores: list = field(default_factory=list)
    latencies: list = field(default_factory=list)
    hop_details: list = field(default_factory=list)
    per_question: list = field(default_factory=list)

    @property
    def em_rate(self):
        return sum(self.em_scores) / len(self.em_scores) * 100 if self.em_scores else 0
    @property
    def relaxed_em_rate(self):
        return sum(self.relaxed_em_scores) / len(self.relaxed_em_scores) * 100 if self.relaxed_em_scores else 0
    @property
    def mean_f1(self):
        return sum(self.f1_scores) / len(self.f1_scores) if self.f1_scores else 0
    @property
    def mean_latency(self):
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0


def run_experiment(limit=30, verbose=True):
    from datasets import load_dataset

    print("Loading MuSiQue validation set...")
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    samples = [s for s in ds if s.get("answerable", True)][:limit]
    print(f"Testing on {len(samples)} answerable questions")

    retriever = EmbeddingRetriever()

    configs = {
        "GOLD_EMBED": ConfigResult("GOLD_EMBED"),
        "QWEN_DECOMPOSE": ConfigResult("QWEN_DECOMPOSE"),
    }

    for i, sample in enumerate(samples):
        question = sample["question"]
        answer = sample["answer"]
        aliases = sample.get("answer_aliases", [])
        n_hops = len(sample.get("question_decomposition", []))

        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(samples)}] ({n_hops}-hop) {question}")
        print(f"Expected: {answer}")
        print(f"{'='*70}")

        # 1. Gold decomposition + embedding retrieval (v2 baseline)
        t0 = time.time()
        pred, hops = run_gold_decomp_embed(sample, retriever, verbose=verbose)
        lat = (time.time() - t0) * 1000
        em = exact_match(pred, answer, aliases)
        rem = relaxed_match(pred, answer, aliases)
        f1 = best_f1(pred, answer, aliases)
        configs["GOLD_EMBED"].em_scores.append(em)
        configs["GOLD_EMBED"].relaxed_em_scores.append(rem)
        configs["GOLD_EMBED"].f1_scores.append(f1)
        configs["GOLD_EMBED"].latencies.append(lat)
        configs["GOLD_EMBED"].hop_details.append(hops)
        configs["GOLD_EMBED"].per_question.append({
            "id": sample["id"], "predicted": pred, "em": em,
            "relaxed_em": rem, "f1": f1, "n_hops": n_hops
        })
        s = "+" if em else ("~" if rem else "-")
        print(f"  [{s}] GOLD_EMBED: {pred} (F1={f1:.2f})")

        # 2. Qwen decomposes, Phi-3 extracts
        t0 = time.time()
        pred, hops, sub_qs = run_qwen_decompose(sample, retriever, verbose=verbose)
        lat = (time.time() - t0) * 1000
        em = exact_match(pred, answer, aliases)
        rem = relaxed_match(pred, answer, aliases)
        f1 = best_f1(pred, answer, aliases)
        configs["QWEN_DECOMPOSE"].em_scores.append(em)
        configs["QWEN_DECOMPOSE"].relaxed_em_scores.append(rem)
        configs["QWEN_DECOMPOSE"].f1_scores.append(f1)
        configs["QWEN_DECOMPOSE"].latencies.append(lat)
        configs["QWEN_DECOMPOSE"].hop_details.append(hops)
        configs["QWEN_DECOMPOSE"].per_question.append({
            "id": sample["id"], "predicted": pred, "em": em,
            "relaxed_em": rem, "f1": f1, "n_hops": n_hops,
            "sub_questions": sub_qs,
        })
        s = "+" if em else ("~" if rem else "-")
        print(f"  [{s}] QWEN_DECOMP: {pred} (F1={f1:.2f})")

        # Running totals
        print(f"\n  --- Running ({i+1} questions) ---")
        for name, cfg in configs.items():
            print(f"    {name:25s}: EM={cfg.em_rate:5.1f}% relEM={cfg.relaxed_em_rate:5.1f}% F1={cfg.mean_f1:.3f}")

    return configs, samples


def analyze_results(configs, samples):
    print("\n" + "=" * 70)
    print("ITERATIVE RETRIEVAL v4 - FINAL RESULTS")
    print("=" * 70)

    print(f"\n{'Config':<26s} {'EM':>6s} {'relEM':>6s} {'F1':>6s} {'Lat':>8s}")
    print("-" * 60)
    for name, cfg in configs.items():
        print(f"{name:<26s} {cfg.em_rate:5.1f}% {cfg.relaxed_em_rate:5.1f}% {cfg.mean_f1:.3f} {cfg.mean_latency:6.0f}ms")

    # Compare with v2 baselines
    v2_path = Path("evaluation/results/iterative_retrieval_v2_30.json")
    if v2_path.exists():
        with open(v2_path) as f:
            v2 = json.load(f)
        print("\n  --- vs v2 Baselines ---")
        for v2_name in ["SINGLE_PASS_v2", "ORACLE_v2", "EMBED_RETRIEVAL"]:
            if v2_name in v2["configs"]:
                v2_em = v2["configs"][v2_name]["em"]
                print(f"    v2 {v2_name}: {v2_em:.1f}% EM")
        for name, cfg in configs.items():
            print(f"    v4 {name}: {cfg.em_rate:.1f}% EM")

    # Decomposition quality analysis
    if "QWEN_DECOMPOSE" in configs:
        print("\n  --- Decomposition Quality ---")
        gold_match = 0
        total = 0
        for q_hops in configs["QWEN_DECOMPOSE"].hop_details:
            for h in q_hops:
                if h.get("gold_retrieved"):
                    gold_match += 1
                total += 1
        if total > 0:
            print(f"    Gold paragraph retrieved: {gold_match}/{total} ({gold_match/total*100:.1f}%)")

        # Per-hop accuracy
        hop1_correct = sum(1 for hops in configs["QWEN_DECOMPOSE"].hop_details
                          if len(hops) > 0 and hops[0].get("em", False))
        hop2_correct = sum(1 for hops in configs["QWEN_DECOMPOSE"].hop_details
                          if len(hops) > 1 and hops[1].get("em", False))
        n = len(configs["QWEN_DECOMPOSE"].hop_details)
        print(f"    Hop 1 EM: {hop1_correct}/{n} ({hop1_correct/n*100:.1f}%)")
        print(f"    Hop 2 EM: {hop2_correct}/{n} ({hop2_correct/n*100:.1f}%)")

    results = {
        "configs": {name: {
            "em": cfg.em_rate,
            "relaxed_em": cfg.relaxed_em_rate,
            "f1": cfg.mean_f1,
            "latency_ms": cfg.mean_latency,
            "per_question": cfg.per_question,
        } for name, cfg in configs.items()},
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Iterative Retrieval v4")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", type=str, default="evaluation/results/iterative_retrieval_v4.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"Ollama models: {models}")
        assert "phi3:mini" in models, "phi3:mini not found"
        assert "qwen2.5:7b" in models, "qwen2.5:7b not found - run: ollama pull qwen2.5:7b"
    except AssertionError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR connecting to Ollama: {e}")
        sys.exit(1)

    configs, samples = run_experiment(limit=args.limit, verbose=not args.quiet)
    results = analyze_results(configs, samples)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
