#!/usr/bin/env python3
"""
Multi-Benchmark Iterative Retrieval Evaluation.

Tests the system-level decomposition approach across multiple benchmarks,
not just MuSiQue. This prevents gaming a single benchmark and validates
that the approach generalizes.

Benchmarks:
1. HotpotQA (distractor) - 2-hop, 10 paragraphs provided
2. MuSiQue - 2-4 hop, 20 paragraphs provided
3. TriviaQA - 1-hop baseline (sanity check)

For each benchmark, tests:
- SINGLE_PASS: All context + question → model
- ITERATIVE: System decomposes → retrieve per hop → extract → chain

For HotpotQA, we test both gold decomposition (from supporting facts)
and the SINGLE_PASS approach.
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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


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

def ask_phi3(prompt, temperature=0.1, max_tokens=32, retries=2):
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": "phi3:mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens}
                },
                timeout=180
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            if attempt < retries:
                print(f"    [RETRY] phi3:mini timed out, attempt {attempt+2}/{retries+1}...")
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


FEW_SHOT_TEMPLATE = """Answer the question using the context below. Give ONLY the specific name, place, or fact asked for. Be as concise as possible - just the core answer, no extra details.

Examples:

Context: [Steve Hillage] Green is the fourth studio album by British progressive rock musician Steve Hillage.
Question: Who performed Green?
Answer: Steve Hillage

Context: [Orion Pictures] The film was distributed by Orion Pictures, founded by Mike Medavoy and four other executives.
Question: Who founded Orion Pictures?
Answer: Mike Medavoy

Context: [Canyon, Texas] Canyon is a city in and the county seat of Randall County, Texas, United States.
Question: What county is Canyon in?
Answer: Randall County

Now answer this question:

Context: {context}
Question: {question}
Answer:"""


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
        """paragraphs: list of {"title": str, "text": str}"""
        self._load_model()
        query_text = f"Represent this sentence for searching relevant passages: {query}"
        para_texts = [f"{p['title']} {p['text']}" for p in paragraphs]
        query_emb = self.model.encode([query_text], normalize_embeddings=True)
        para_embs = self.model.encode(para_texts, normalize_embeddings=True)
        sims = np.dot(para_embs, query_emb.T).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(int(idx), float(sims[idx])) for idx in top_indices]


# ── HotpotQA Processing ─────────────────────────────────────────────

def process_hotpotqa_sample(sample):
    """Convert HotpotQA sample to standardized format."""
    paragraphs = []
    # HotpotQA context is a list of [title, [sentences]]
    for title, sentences in zip(sample["context"]["title"], sample["context"]["sentences"]):
        text = " ".join(sentences)
        paragraphs.append({"title": title, "text": text})

    # Supporting facts tell us which paragraphs are relevant
    support_titles = set(sample["supporting_facts"]["title"])
    support_indices = [i for i, p in enumerate(paragraphs) if p["title"] in support_titles]

    return {
        "question": sample["question"],
        "answer": sample["answer"],
        "aliases": [],
        "paragraphs": paragraphs,
        "support_indices": support_indices,
        "type": sample.get("type", "unknown"),
        "level": sample.get("level", "unknown"),
    }


def run_hotpotqa_single_pass(processed, verbose=False):
    """All paragraphs + question → single model call."""
    context_parts = [f"[{p['title']}] {p['text']}" for p in processed["paragraphs"]]
    context = "\n\n".join(context_parts)
    prompt = FEW_SHOT_TEMPLATE.format(context=context, question=processed["question"])
    response = ask_phi3(prompt, temperature=0.1, max_tokens=32)
    return extract_short_answer(response)


def run_hotpotqa_iterative(processed, retriever, verbose=False):
    """Iterative: retrieve relevant paragraphs, extract from each, combine."""
    question = processed["question"]
    paragraphs = processed["paragraphs"]

    # Step 1: Retrieve top paragraphs for the question
    results = retriever.retrieve(question, paragraphs, top_k=3)
    context_parts = [f"[{paragraphs[idx]['title']}] {paragraphs[idx]['text']}"
                    for idx, _ in results]
    context = "\n\n".join(context_parts)

    # Check retrieval quality
    retrieved_indices = {idx for idx, _ in results}
    support_found = sum(1 for si in processed["support_indices"] if si in retrieved_indices)
    retrieval_quality = support_found / len(processed["support_indices"]) if processed["support_indices"] else 0

    prompt = FEW_SHOT_TEMPLATE.format(context=context, question=question)
    response = ask_phi3(prompt, temperature=0.1, max_tokens=32)
    answer = extract_short_answer(response)

    if verbose:
        print(f"    Retrieved {len(results)} paragraphs, {support_found}/{len(processed['support_indices'])} supporting")

    return answer, retrieval_quality


def run_hotpotqa_gold_context(processed, verbose=False):
    """Only gold supporting paragraphs → model."""
    context_parts = [f"[{processed['paragraphs'][i]['title']}] {processed['paragraphs'][i]['text']}"
                    for i in processed["support_indices"]]
    context = "\n\n".join(context_parts)
    prompt = FEW_SHOT_TEMPLATE.format(context=context, question=processed["question"])
    response = ask_phi3(prompt, temperature=0.1, max_tokens=32)
    return extract_short_answer(response)


# ── TriviaQA Processing ─────────────────────────────────────────────

def process_triviaqa_sample(sample):
    """Convert TriviaQA sample to standardized format.

    TriviaQA uses columnar format: search_results is a dict with lists,
    not a list of dicts. E.g. search_results["title"] = [title1, title2, ...]
    """
    answer = sample["answer"]["value"]
    aliases = sample["answer"].get("aliases", [])

    # TriviaQA RC has search_results in columnar format
    paragraphs = []
    sr = sample.get("search_results", {})
    if sr and sr.get("search_context"):
        titles = sr.get("title", [])
        contexts = sr.get("search_context", [])
        for i in range(min(5, len(contexts))):
            ctx = contexts[i] if i < len(contexts) else ""
            title = titles[i] if i < len(titles) else ""
            if ctx:
                paragraphs.append({
                    "title": title,
                    "text": ctx[:500]
                })

    # Also try entity_pages (also columnar)
    if not paragraphs:
        ep = sample.get("entity_pages", {})
        if ep and ep.get("wiki_context"):
            titles = ep.get("title", [])
            contexts = ep.get("wiki_context", [])
            for i in range(min(5, len(contexts))):
                ctx = contexts[i] if i < len(contexts) else ""
                title = titles[i] if i < len(titles) else ""
                if ctx:
                    paragraphs.append({
                        "title": title,
                        "text": ctx[:500]
                    })

    return {
        "question": sample["question"],
        "answer": answer,
        "aliases": aliases,
        "paragraphs": paragraphs,
    }


def run_triviaqa_single_pass(processed, verbose=False):
    """All context + question → single model call."""
    if processed["paragraphs"]:
        context_parts = [f"[{p['title']}] {p['text']}" for p in processed["paragraphs"][:3]]
        context = "\n\n".join(context_parts)
        prompt = FEW_SHOT_TEMPLATE.format(context=context, question=processed["question"])
    else:
        # No context - parametric only
        prompt = f"Answer concisely: {processed['question']}\nAnswer:"
    response = ask_phi3(prompt, temperature=0.1, max_tokens=32)
    return extract_short_answer(response)


# ── MuSiQue Processing (reuse from v2) ──────────────────────────────

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


def run_musique_single_pass(sample, verbose=False):
    """All 20 paragraphs + question → model."""
    paragraphs = sample["paragraphs"]
    context_parts = [f"[{p['title']}] {p['paragraph_text']}" for p in paragraphs]
    context = "\n\n".join(context_parts)
    prompt = FEW_SHOT_TEMPLATE.format(context=context, question=sample["question"])
    response = ask_phi3(prompt, temperature=0.1, max_tokens=32)
    return extract_short_answer(response)


def run_musique_iterative(sample, retriever, verbose=False):
    """Gold decomposition + embedding retrieval + model extraction."""
    decomposition = sample["question_decomposition"]
    paragraphs = sample["paragraphs"]
    # Convert to standardized format for retriever
    std_paragraphs = [{"title": p["title"], "text": p["paragraph_text"]} for p in paragraphs]

    previous_answers = []
    for i, hop in enumerate(decomposition):
        hop_q = format_hop_question(hop["question"], previous_answers)
        results = retriever.retrieve(hop_q, std_paragraphs, top_k=3)
        context_parts = [f"[{std_paragraphs[idx]['title']}] {std_paragraphs[idx]['text']}"
                        for idx, _ in results]
        context = "\n\n".join(context_parts)
        prompt = FEW_SHOT_TEMPLATE.format(context=context, question=hop_q)
        response = ask_phi3(prompt, temperature=0.1, max_tokens=32)
        answer = extract_short_answer(response)
        previous_answers.append(answer)
        if verbose:
            print(f"    Hop {i+1}: {hop_q[:50]}... -> {answer} (gold: {hop['answer']})")

    return previous_answers[-1] if previous_answers else ""


def run_musique_oracle(sample, verbose=False):
    """Gold decomposition + gold paragraphs."""
    decomposition = sample["question_decomposition"]
    paragraphs = sample["paragraphs"]

    previous_answers = []
    for i, hop in enumerate(decomposition):
        hop_q = format_hop_question(hop["question"], previous_answers)
        gold_idx = hop["paragraph_support_idx"]
        p = paragraphs[gold_idx]
        context = f"[{p['title']}] {p['paragraph_text']}"
        prompt = FEW_SHOT_TEMPLATE.format(context=context, question=hop_q)
        response = ask_phi3(prompt, temperature=0.1, max_tokens=32)
        answer = extract_short_answer(response)
        previous_answers.append(answer)
        if verbose:
            em = "+" if normalize_answer(answer) == normalize_answer(hop["answer"]) else "-"
            print(f"    [{em}] Hop {i+1}: {answer} (gold: {hop['answer']})")

    return previous_answers[-1] if previous_answers else ""


# ── Main Runner ──────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    name: str
    config: str
    em_scores: list = field(default_factory=list)
    relaxed_em_scores: list = field(default_factory=list)
    f1_scores: list = field(default_factory=list)
    latencies: list = field(default_factory=list)

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


def run_hotpotqa(limit=30, verbose=True):
    """Run HotpotQA evaluation."""
    from datasets import load_dataset
    print("\n" + "="*70)
    print("HOTPOTQA EVALUATION")
    print("="*70)

    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    # Filter to comparison and bridge types
    samples = list(ds)[:limit]
    print(f"Testing {len(samples)} HotpotQA questions")

    retriever = EmbeddingRetriever()

    results = {
        "single_pass": BenchmarkResult("hotpotqa", "single_pass"),
        "embed_retrieval": BenchmarkResult("hotpotqa", "embed_retrieval"),
        "gold_context": BenchmarkResult("hotpotqa", "gold_context"),
    }
    retrieval_qualities = []

    for i, sample in enumerate(samples):
        processed = process_hotpotqa_sample(sample)
        answer = processed["answer"]
        q_type = processed["type"]

        if verbose:
            print(f"\n  [{i+1}/{len(samples)}] ({q_type}) {processed['question'][:60]}...")
            print(f"  Expected: {answer}")

        # Single pass
        t0 = time.time()
        pred = run_hotpotqa_single_pass(processed, verbose)
        lat = (time.time() - t0) * 1000
        em = exact_match(pred, answer)
        rem = relaxed_match(pred, answer)
        f1 = f1_score(pred, answer)
        results["single_pass"].em_scores.append(em)
        results["single_pass"].relaxed_em_scores.append(rem)
        results["single_pass"].f1_scores.append(f1)
        results["single_pass"].latencies.append(lat)

        # Embedding retrieval
        t0 = time.time()
        pred_emb, rq = run_hotpotqa_iterative(processed, retriever, verbose)
        lat = (time.time() - t0) * 1000
        em_emb = exact_match(pred_emb, answer)
        rem_emb = relaxed_match(pred_emb, answer)
        f1_emb = f1_score(pred_emb, answer)
        results["embed_retrieval"].em_scores.append(em_emb)
        results["embed_retrieval"].relaxed_em_scores.append(rem_emb)
        results["embed_retrieval"].f1_scores.append(f1_emb)
        results["embed_retrieval"].latencies.append(lat)
        retrieval_qualities.append(rq)

        # Gold context
        t0 = time.time()
        pred_gold = run_hotpotqa_gold_context(processed, verbose)
        lat = (time.time() - t0) * 1000
        em_gold = exact_match(pred_gold, answer)
        rem_gold = relaxed_match(pred_gold, answer)
        f1_gold = f1_score(pred_gold, answer)
        results["gold_context"].em_scores.append(em_gold)
        results["gold_context"].relaxed_em_scores.append(rem_gold)
        results["gold_context"].f1_scores.append(f1_gold)
        results["gold_context"].latencies.append(lat)

        if verbose:
            sp = "+" if em else "-"
            ep = "+" if em_emb else "-"
            gp = "+" if em_gold else "-"
            print(f"    [{sp}] Single: {pred}")
            print(f"    [{ep}] Embed:  {pred_emb} (retrieval: {rq:.0%})")
            print(f"    [{gp}] Gold:   {pred_gold}")

    # Summary
    print(f"\nHotpotQA Results (n={len(samples)}):")
    avg_rq = sum(retrieval_qualities) / len(retrieval_qualities) if retrieval_qualities else 0
    print(f"  Avg retrieval quality: {avg_rq:.1%}")
    for name, r in results.items():
        print(f"  {name:20s}: EM={r.em_rate:5.1f}% relEM={r.relaxed_em_rate:5.1f}% F1={r.mean_f1:.3f}")

    return results


def run_musique_bench(limit=30, verbose=True):
    """Run MuSiQue evaluation."""
    from datasets import load_dataset
    print("\n" + "="*70)
    print("MUSIQUE EVALUATION")
    print("="*70)

    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    samples = [s for s in ds if s.get("answerable", True)][:limit]
    print(f"Testing {len(samples)} MuSiQue questions")

    retriever = EmbeddingRetriever()

    results = {
        "single_pass": BenchmarkResult("musique", "single_pass"),
        "embed_retrieval": BenchmarkResult("musique", "embed_retrieval"),
        "oracle": BenchmarkResult("musique", "oracle"),
    }

    for i, sample in enumerate(samples):
        answer = sample["answer"]
        aliases = sample.get("answer_aliases", [])
        n_hops = len(sample.get("question_decomposition", []))

        if verbose:
            print(f"\n  [{i+1}/{len(samples)}] ({n_hops}-hop) {sample['question'][:60]}...")
            print(f"  Expected: {answer}")

        # Single pass
        t0 = time.time()
        pred = run_musique_single_pass(sample, verbose)
        lat = (time.time() - t0) * 1000
        em = exact_match(pred, answer, aliases)
        rem = relaxed_match(pred, answer, aliases)
        f1 = best_f1(pred, answer, aliases)
        results["single_pass"].em_scores.append(em)
        results["single_pass"].relaxed_em_scores.append(rem)
        results["single_pass"].f1_scores.append(f1)
        results["single_pass"].latencies.append(lat)

        # Embedding retrieval (gold decomposition)
        t0 = time.time()
        pred_emb = run_musique_iterative(sample, retriever, verbose)
        lat = (time.time() - t0) * 1000
        em_emb = exact_match(pred_emb, answer, aliases)
        rem_emb = relaxed_match(pred_emb, answer, aliases)
        f1_emb = best_f1(pred_emb, answer, aliases)
        results["embed_retrieval"].em_scores.append(em_emb)
        results["embed_retrieval"].relaxed_em_scores.append(rem_emb)
        results["embed_retrieval"].f1_scores.append(f1_emb)
        results["embed_retrieval"].latencies.append(lat)

        # Oracle (gold decomposition + gold paragraphs)
        t0 = time.time()
        pred_oracle = run_musique_oracle(sample, verbose)
        lat = (time.time() - t0) * 1000
        em_oracle = exact_match(pred_oracle, answer, aliases)
        rem_oracle = relaxed_match(pred_oracle, answer, aliases)
        f1_oracle = best_f1(pred_oracle, answer, aliases)
        results["oracle"].em_scores.append(em_oracle)
        results["oracle"].relaxed_em_scores.append(rem_oracle)
        results["oracle"].f1_scores.append(f1_oracle)
        results["oracle"].latencies.append(lat)

        if verbose:
            sp = "+" if em else "-"
            ep = "+" if em_emb else "-"
            op = "+" if em_oracle else "-"
            print(f"    [{sp}] Single: {pred}")
            print(f"    [{ep}] Embed:  {pred_emb}")
            print(f"    [{op}] Oracle: {pred_oracle}")

    print(f"\nMuSiQue Results (n={len(samples)}):")
    for name, r in results.items():
        print(f"  {name:20s}: EM={r.em_rate:5.1f}% relEM={r.relaxed_em_rate:5.1f}% F1={r.mean_f1:.3f}")

    return results


def run_triviaqa_bench(limit=30, verbose=True):
    """Run TriviaQA evaluation (1-hop sanity check)."""
    from datasets import load_dataset
    print("\n" + "="*70)
    print("TRIVIAQA EVALUATION")
    print("="*70)

    ds = load_dataset("trivia_qa", "rc", split="validation")
    samples = list(ds)[:limit]
    print(f"Testing {len(samples)} TriviaQA questions")

    results = {
        "single_pass": BenchmarkResult("triviaqa", "single_pass"),
    }

    for i, sample in enumerate(samples):
        processed = process_triviaqa_sample(sample)
        answer = processed["answer"]
        aliases = processed["aliases"]

        if verbose and i % 5 == 0:
            print(f"  [{i+1}/{len(samples)}] {processed['question'][:60]}...")

        t0 = time.time()
        pred = run_triviaqa_single_pass(processed, verbose)
        lat = (time.time() - t0) * 1000
        em = exact_match(pred, answer, aliases)
        rem = relaxed_match(pred, answer, aliases)
        f1 = best_f1(pred, answer, aliases)
        results["single_pass"].em_scores.append(em)
        results["single_pass"].relaxed_em_scores.append(rem)
        results["single_pass"].f1_scores.append(f1)
        results["single_pass"].latencies.append(lat)

    print(f"\nTriviaQA Results (n={len(samples)}):")
    for name, r in results.items():
        print(f"  {name:20s}: EM={r.em_rate:5.1f}% relEM={r.relaxed_em_rate:5.1f}% F1={r.mean_f1:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-Benchmark Iterative Retrieval")
    parser.add_argument("--limit", type=int, default=30, help="Questions per benchmark")
    parser.add_argument("--output", type=str, default="evaluation/results/multi_benchmark.json")
    parser.add_argument("--benchmarks", nargs="+", default=["hotpotqa", "musique", "triviaqa"],
                       choices=["hotpotqa", "musique", "triviaqa"])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Verify Ollama
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        assert "phi3:mini" in models
        print(f"Ollama OK: {models}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    all_results = {}

    if "triviaqa" in args.benchmarks:
        tqa_results = run_triviaqa_bench(limit=args.limit, verbose=not args.quiet)
        all_results["triviaqa"] = {name: {
            "em": r.em_rate, "relaxed_em": r.relaxed_em_rate,
            "f1": r.mean_f1, "latency_ms": r.mean_latency,
        } for name, r in tqa_results.items()}

    if "hotpotqa" in args.benchmarks:
        hqa_results = run_hotpotqa(limit=args.limit, verbose=not args.quiet)
        all_results["hotpotqa"] = {name: {
            "em": r.em_rate, "relaxed_em": r.relaxed_em_rate,
            "f1": r.mean_f1, "latency_ms": r.mean_latency,
        } for name, r in hqa_results.items()}

    if "musique" in args.benchmarks:
        mqa_results = run_musique_bench(limit=args.limit, verbose=not args.quiet)
        all_results["musique"] = {name: {
            "em": r.em_rate, "relaxed_em": r.relaxed_em_rate,
            "f1": r.mean_f1, "latency_ms": r.mean_latency,
        } for name, r in mqa_results.items()}

    # Final summary
    print("\n" + "="*70)
    print("MULTI-BENCHMARK SUMMARY")
    print("="*70)
    print(f"\n{'Benchmark':<12s} {'Config':<20s} {'EM':>6s} {'relEM':>6s} {'F1':>6s}")
    print("-" * 60)
    for bench, configs in all_results.items():
        for config, metrics in configs.items():
            print(f"{bench:<12s} {config:<20s} {metrics['em']:5.1f}% {metrics['relaxed_em']:5.1f}% {metrics['f1']:.3f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
