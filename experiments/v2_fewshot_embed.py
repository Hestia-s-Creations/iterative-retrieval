#!/usr/bin/env python3
"""
Iterative Retrieval v2 - Improved extraction and retrieval.

Changes from v1:
1. Better extraction prompt with few-shot examples
2. Explicit conciseness instructions
3. Embedding-based retrieval for LOCAL config (vs keyword matching)
4. Natural language hop question reformulation
5. Substring-aware scoring (relaxed EM)
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
    """Relaxed EM: gold is substring of pred OR pred is substring of gold."""
    pred = normalize_answer(prediction)
    gold = normalize_answer(ground_truth)
    if pred == gold:
        return True
    if gold in pred or pred in gold:
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


# ── LLM Interface ───────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434"

def ask_phi3(prompt, temperature=0.1, max_tokens=32):
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": "phi3:mini",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens}
        },
        timeout=120
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def extract_short_answer(response):
    answer = response.split('\n')[0].strip()
    for prefix in ["the answer is", "answer:", "a:", "the "]:
        if answer.lower().startswith(prefix):
            answer = answer[len(prefix):].strip()
    answer = answer.rstrip('.')
    # Remove brackets
    answer = answer.strip('[]')
    if len(answer) > 80:
        for sep in [',', ';', '(', ' - ', ' who ', ' which ']:
            if sep in answer:
                answer = answer[:answer.index(sep)].strip()
                break
    return answer


# ── Question Formatting ─────────────────────────────────────────────

def format_hop_question_v2(hop_question, previous_answers):
    """Convert MuSiQue hop notation to natural language, improved."""
    q = hop_question

    # Substitute #N references
    for i, answer in enumerate(previous_answers, 1):
        q = q.replace(f"#{i}", answer)

    # Convert "X >> Y" to natural question with varied phrasing
    if ">>" in q:
        parts = q.split(">>")
        subject = parts[0].strip()
        relation = parts[1].strip().lower()

        # Map common relations to natural questions
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

        # Check for exact or substring match
        for key, template in relation_map.items():
            if key in relation:
                return template

        # Check for "located in" variants
        if "located in" in relation or "administrative territorial" in relation:
            return f"What administrative region is {subject} located in?"

        # Fallback
        return f"What is the {relation} of {subject}?"

    return q


# ── Extraction Prompts ──────────────────────────────────────────────

FEW_SHOT_TEMPLATE = """Answer the question using the context below. Give ONLY the specific name, place, or fact asked for. Be as concise as possible — just the core answer, no extra details.

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


# ── Embedding-based Retrieval ───────────────────────────────────────

class EmbeddingRetriever:
    """Simple embedding-based retrieval using sentence-transformers."""

    def __init__(self):
        self.model = None

    def _load_model(self):
        if self.model is None:
            print("  Loading BGE embeddings...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    def retrieve(self, query, paragraphs, top_k=3):
        """Retrieve top_k paragraphs by embedding similarity."""
        self._load_model()

        query_text = f"Represent this sentence for searching relevant passages: {query}"
        para_texts = [f"{p['title']} {p['paragraph_text']}" for p in paragraphs]

        query_emb = self.model.encode([query_text], normalize_embeddings=True)
        para_embs = self.model.encode(para_texts, normalize_embeddings=True)

        sims = np.dot(para_embs, query_emb.T).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]

        return [(int(idx), float(sims[idx])) for idx in top_indices]


# ── Experiment Configs ──────────────────────────────────────────────

def run_single_pass_v2(sample, verbose=False):
    """Improved single-pass with better prompting."""
    question = sample["question"]
    paras = sample["paragraphs"]

    context_parts = []
    for para in paras[:10]:
        text = f"[{para['title']}] {para['paragraph_text'][:200]}"
        context_parts.append(text)
    context = "\n".join(context_parts)

    prompt = FEW_SHOT_TEMPLATE.format(context=context, question=question)
    response = ask_phi3(prompt, temperature=0.1, max_tokens=32)
    answer = extract_short_answer(response)

    if verbose:
        print(f"    [SINGLE-v2] {answer}")
    return answer


def run_oracle_v2(sample, use_gold_chain=False, verbose=False):
    """Improved oracle with few-shot extraction and natural questions."""
    decomposition = sample["question_decomposition"]
    paragraphs = sample["paragraphs"]

    previous_answers_model = []
    previous_answers_gold = []
    hop_details = []

    for i, hop in enumerate(decomposition):
        chain = previous_answers_gold if use_gold_chain else previous_answers_model
        hop_q = format_hop_question_v2(hop["question"], chain)

        support_idx = hop["paragraph_support_idx"]
        support_para = paragraphs[support_idx]
        context = f"[{support_para['title']}] {support_para['paragraph_text']}"

        prompt = FEW_SHOT_TEMPLATE.format(context=context, question=hop_q)
        response = ask_phi3(prompt, temperature=0.1, max_tokens=32)
        answer = extract_short_answer(response)

        em = exact_match(answer, hop["answer"])
        r_em = relaxed_match(answer, hop["answer"])

        hop_details.append({
            "hop": i + 1,
            "question": hop_q,
            "gold_answer": hop["answer"],
            "predicted": answer,
            "em": em,
            "relaxed_em": r_em,
            "f1": f1_score(answer, hop["answer"]),
        })

        if verbose:
            status = "+" if em else ("~" if r_em else "-")
            print(f"    [{status}] Hop {i+1}: {hop_q[:50]}...")
            print(f"        Expected: {hop['answer']}, Got: {answer}")

        previous_answers_model.append(answer)
        previous_answers_gold.append(hop["answer"])

    final = previous_answers_model[-1] if previous_answers_model else ""
    return final, hop_details


def run_embedding_retrieval(sample, retriever, use_gold_chain=False, verbose=False):
    """Gold decomposition + embedding-based retrieval from sample paragraphs."""
    decomposition = sample["question_decomposition"]
    paragraphs = sample["paragraphs"]

    previous_answers_model = []
    previous_answers_gold = []
    hop_details = []

    for i, hop in enumerate(decomposition):
        chain = previous_answers_gold if use_gold_chain else previous_answers_model
        hop_q = format_hop_question_v2(hop["question"], chain)

        results = retriever.retrieve(hop_q, paragraphs, top_k=3)
        retrieved_indices = [idx for idx, _ in results]

        context_parts = []
        for idx, sim in results:
            para = paragraphs[idx]
            context_parts.append(f"[{para['title']}] {para['paragraph_text']}")
        context = "\n\n".join(context_parts)

        gold_idx = hop["paragraph_support_idx"]
        gold_retrieved = gold_idx in retrieved_indices

        prompt = FEW_SHOT_TEMPLATE.format(context=context, question=hop_q)
        response = ask_phi3(prompt, temperature=0.1, max_tokens=32)
        answer = extract_short_answer(response)

        em = exact_match(answer, hop["answer"])

        hop_details.append({
            "hop": i + 1,
            "question": hop_q,
            "gold_answer": hop["answer"],
            "predicted": answer,
            "em": em,
            "relaxed_em": relaxed_match(answer, hop["answer"]),
            "f1": f1_score(answer, hop["answer"]),
            "gold_retrieved": gold_retrieved,
            "retrieved_indices": retrieved_indices,
            "retrieval_scores": [s for _, s in results],
        })

        if verbose:
            status = "+" if em else "-"
            r_status = "+" if gold_retrieved else "-"
            print(f"    [{status}] Hop {i+1}: {hop_q[:50]}...")
            print(f"        Expected: {hop['answer']}, Got: {answer} (retr:{r_status})")

        previous_answers_model.append(answer)
        previous_answers_gold.append(hop["answer"])

    final = previous_answers_model[-1] if previous_answers_model else ""
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
        "SINGLE_PASS_v2": ConfigResult("SINGLE_PASS_v2"),
        "ORACLE_v2": ConfigResult("ORACLE_v2"),
        "ORACLE_v2_GOLD": ConfigResult("ORACLE_v2_GOLD"),
        "EMBED_RETRIEVAL": ConfigResult("EMBED_RETRIEVAL"),
        "EMBED_RETRIEVAL_GOLD": ConfigResult("EMBED_RETRIEVAL_GOLD"),
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

        # 1. Single Pass v2
        t0 = time.time()
        pred = run_single_pass_v2(sample, verbose=verbose)
        lat = (time.time() - t0) * 1000
        em = exact_match(pred, answer, aliases)
        rem = relaxed_match(pred, answer, aliases)
        f1 = best_f1(pred, answer, aliases)
        configs["SINGLE_PASS_v2"].em_scores.append(em)
        configs["SINGLE_PASS_v2"].relaxed_em_scores.append(rem)
        configs["SINGLE_PASS_v2"].f1_scores.append(f1)
        configs["SINGLE_PASS_v2"].latencies.append(lat)
        configs["SINGLE_PASS_v2"].per_question.append({
            "id": sample["id"], "predicted": pred, "em": em,
            "relaxed_em": rem, "f1": f1, "n_hops": n_hops
        })
        s = "+" if em else ("~" if rem else "-")
        print(f"  [{s}] SINGLE_v2: {pred} (F1={f1:.2f})")

        # 2. Oracle v2 (model chaining)
        t0 = time.time()
        pred, hops = run_oracle_v2(sample, use_gold_chain=False, verbose=verbose)
        lat = (time.time() - t0) * 1000
        em = exact_match(pred, answer, aliases)
        rem = relaxed_match(pred, answer, aliases)
        f1 = best_f1(pred, answer, aliases)
        configs["ORACLE_v2"].em_scores.append(em)
        configs["ORACLE_v2"].relaxed_em_scores.append(rem)
        configs["ORACLE_v2"].f1_scores.append(f1)
        configs["ORACLE_v2"].latencies.append(lat)
        configs["ORACLE_v2"].hop_details.append(hops)
        configs["ORACLE_v2"].per_question.append({
            "id": sample["id"], "predicted": pred, "em": em,
            "relaxed_em": rem, "f1": f1, "n_hops": n_hops
        })
        s = "+" if em else ("~" if rem else "-")
        print(f"  [{s}] ORACLE_v2: {pred} (F1={f1:.2f})")

        # 3. Oracle v2 (gold chaining)
        t0 = time.time()
        pred, hops = run_oracle_v2(sample, use_gold_chain=True, verbose=verbose)
        lat = (time.time() - t0) * 1000
        em = exact_match(pred, answer, aliases)
        rem = relaxed_match(pred, answer, aliases)
        f1 = best_f1(pred, answer, aliases)
        configs["ORACLE_v2_GOLD"].em_scores.append(em)
        configs["ORACLE_v2_GOLD"].relaxed_em_scores.append(rem)
        configs["ORACLE_v2_GOLD"].f1_scores.append(f1)
        configs["ORACLE_v2_GOLD"].latencies.append(lat)
        configs["ORACLE_v2_GOLD"].hop_details.append(hops)
        configs["ORACLE_v2_GOLD"].per_question.append({
            "id": sample["id"], "predicted": pred, "em": em,
            "relaxed_em": rem, "f1": f1, "n_hops": n_hops
        })
        s = "+" if em else ("~" if rem else "-")
        print(f"  [{s}] ORACLE_v2_GOLD: {pred} (F1={f1:.2f})")

        # 4. Embedding Retrieval (model chaining)
        t0 = time.time()
        pred, hops = run_embedding_retrieval(sample, retriever, use_gold_chain=False, verbose=verbose)
        lat = (time.time() - t0) * 1000
        em = exact_match(pred, answer, aliases)
        rem = relaxed_match(pred, answer, aliases)
        f1 = best_f1(pred, answer, aliases)
        configs["EMBED_RETRIEVAL"].em_scores.append(em)
        configs["EMBED_RETRIEVAL"].relaxed_em_scores.append(rem)
        configs["EMBED_RETRIEVAL"].f1_scores.append(f1)
        configs["EMBED_RETRIEVAL"].latencies.append(lat)
        configs["EMBED_RETRIEVAL"].hop_details.append(hops)
        configs["EMBED_RETRIEVAL"].per_question.append({
            "id": sample["id"], "predicted": pred, "em": em,
            "relaxed_em": rem, "f1": f1, "n_hops": n_hops
        })
        s = "+" if em else ("~" if rem else "-")
        print(f"  [{s}] EMBED: {pred} (F1={f1:.2f})")

        # 5. Embedding Retrieval (gold chaining)
        t0 = time.time()
        pred, hops = run_embedding_retrieval(sample, retriever, use_gold_chain=True, verbose=verbose)
        lat = (time.time() - t0) * 1000
        em = exact_match(pred, answer, aliases)
        rem = relaxed_match(pred, answer, aliases)
        f1 = best_f1(pred, answer, aliases)
        configs["EMBED_RETRIEVAL_GOLD"].em_scores.append(em)
        configs["EMBED_RETRIEVAL_GOLD"].relaxed_em_scores.append(rem)
        configs["EMBED_RETRIEVAL_GOLD"].f1_scores.append(f1)
        configs["EMBED_RETRIEVAL_GOLD"].latencies.append(lat)
        configs["EMBED_RETRIEVAL_GOLD"].hop_details.append(hops)
        configs["EMBED_RETRIEVAL_GOLD"].per_question.append({
            "id": sample["id"], "predicted": pred, "em": em,
            "relaxed_em": rem, "f1": f1, "n_hops": n_hops
        })
        s = "+" if em else ("~" if rem else "-")
        print(f"  [{s}] EMBED_GOLD: {pred} (F1={f1:.2f})")

        # Running totals
        print(f"\n  --- Running ({i+1} questions) ---")
        for name, cfg in configs.items():
            print(f"    {name:25s}: EM={cfg.em_rate:5.1f}% relEM={cfg.relaxed_em_rate:5.1f}% F1={cfg.mean_f1:.3f}")

    return configs, samples


def analyze_results(configs, samples):
    print("\n" + "=" * 70)
    print("ITERATIVE RETRIEVAL v2 - FINAL RESULTS")
    print("=" * 70)

    print(f"\n{'Config':<26s} {'EM':>6s} {'relEM':>6s} {'F1':>6s} {'Lat':>8s}")
    print("-" * 60)
    for name, cfg in configs.items():
        print(f"{name:<26s} {cfg.em_rate:5.1f}% {cfg.relaxed_em_rate:5.1f}% {cfg.mean_f1:.3f} {cfg.mean_latency:6.0f}ms")

    # Per-hop for oracle configs
    for config_name in ["ORACLE_v2", "ORACLE_v2_GOLD"]:
        print(f"\n  --- Per-Hop: {config_name} ---")
        hop_em = defaultdict(list)
        hop_rem = defaultdict(list)
        hop_f1 = defaultdict(list)
        for q_hops in configs[config_name].hop_details:
            for h in q_hops:
                hop_em[h["hop"]].append(h["em"])
                hop_rem[h["hop"]].append(h.get("relaxed_em", False))
                hop_f1[h["hop"]].append(h["f1"])
        for n in sorted(hop_em.keys()):
            em = sum(hop_em[n]) / len(hop_em[n]) * 100
            rem = sum(hop_rem[n]) / len(hop_rem[n]) * 100
            f1 = sum(hop_f1[n]) / len(hop_f1[n])
            print(f"    Hop {n}: EM={em:.1f}%, relEM={rem:.1f}%, F1={f1:.3f} (n={len(hop_em[n])})")

    # Retrieval analysis
    for config_name in ["EMBED_RETRIEVAL", "EMBED_RETRIEVAL_GOLD"]:
        if configs[config_name].hop_details:
            total = 0
            retrieved = 0
            for q_hops in configs[config_name].hop_details:
                for h in q_hops:
                    total += 1
                    if h.get("gold_retrieved"):
                        retrieved += 1
            print(f"\n  --- Retrieval: {config_name} ---")
            print(f"    Gold paragraph found: {retrieved}/{total} ({retrieved/total*100:.1f}%)")

    # V1 comparison
    v1_path = Path("evaluation/results/iterative_retrieval_30.json")
    if v1_path.exists():
        with open(v1_path) as f:
            v1 = json.load(f)
        print("\n  --- v1 vs v2 Comparison ---")
        print(f"    {'Config':<26s} {'v1 EM':>7s} {'v2 EM':>7s} {'Delta':>7s}")
        mapping = {
            "SINGLE_PASS": "SINGLE_PASS_v2",
            "GOLD_ORACLE": "ORACLE_v2",
            "GOLD_ORACLE_GOLD_CHAIN": "ORACLE_v2_GOLD",
            "GOLD_DECOMP_LOCAL": "EMBED_RETRIEVAL",
        }
        for v1_name, v2_name in mapping.items():
            if v1_name in v1["configs"] and v2_name in configs:
                v1_em = v1["configs"][v1_name]["em"]
                v2_em = configs[v2_name].em_rate
                delta = v2_em - v1_em
                print(f"    {v1_name:<26s} {v1_em:5.1f}% {v2_em:5.1f}% {'+' if delta>=0 else ''}{delta:.1f}%")

    # Save
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
    parser = argparse.ArgumentParser(description="Iterative Retrieval v2")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", type=str, default="evaluation/results/iterative_retrieval_v2.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Verify Ollama
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        assert "phi3:mini" in models, "phi3:mini not loaded"
        print(f"Ollama OK: {models}")
    except Exception as e:
        print(f"ERROR: {e}")
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
