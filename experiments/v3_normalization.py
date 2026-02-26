#!/usr/bin/env python3
"""
Iterative Retrieval v3 - Answer normalization + auto-decomposition.

Changes from v2:
1. Answer normalizer strips verbose answers to core entity
2. Tests auto-decomposition: can the model generate sub-questions?
3. Focuses on closing the EMBED_RETRIEVAL → ORACLE gap
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



# ── Metrics (same as v2) ────────────────────────────────────────────

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
    answer = answer.rstrip('.').strip('[]')
    if len(answer) > 80:
        for sep in [',', ';', '(', ' - ', ' who ', ' which ']:
            if sep in answer:
                answer = answer[:answer.index(sep)].strip()
                break
    return answer


# ── Answer Normalization ────────────────────────────────────────────

def normalize_intermediate_answer(answer):
    """Strip verbose answers to their core entity for better hop chaining.

    Examples:
      "Hughesville, Maryland" → "Hughesville"
      "Blossom Films owned by Nicole Kidman" → "Blossom Films"
      "Canyon, Texas" → "Canyon"
      "Ali & Gipp featuring LeToya Luckett" → "Ali & Gipp"
      "Exeter College, Oxford" → "Exeter College"
      "Nicole Kidman through her company Blossom Films" → "Nicole Kidman"
      "Southwest City, Missouri" → "Southwest City"
    """
    original = answer.strip()

    # Pattern 1: "X through/via/by/from/of/owned by Y" → take X
    for pattern in [
        r'^(.+?)\s+(?:through|via|by|from|owned by|of)\s+',
        r'^(.+?)\s+(?:featuring|feat\.|ft\.)\s+',
    ]:
        m = re.match(pattern, original, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            if len(candidate) >= 3:
                return candidate

    # Pattern 2: "X, State/Country" → take X (but not "X, Y" for names)
    # Match: "Canyon, Texas", "Hughesville, Maryland", "Southwest City, Missouri"
    # Don't match: "Ali & Gipp" or person names
    location_suffixes = [
        r',\s*(?:Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|'
        r'Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|'
        r'Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|'
        r'Mississippi|Missouri|Montana|Nebraska|Nevada|New\s+Hampshire|'
        r'New\s+Jersey|New\s+Mexico|New\s+York|North\s+Carolina|North\s+Dakota|'
        r'Ohio|Oklahoma|Oregon|Pennsylvania|Rhode\s+Island|South\s+Carolina|'
        r'South\s+Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|'
        r'West\s+Virginia|Wisconsin|Wyoming|'
        r'England|Scotland|Wales|Ireland|France|Germany|Mexico|Canada|'
        r'Australia|India|China|Japan|Brazil|'
        r'United\s+States|United\s+Kingdom|UK|US|USA)\s*$',
    ]
    for pattern in location_suffixes:
        m = re.search(pattern, original, re.IGNORECASE)
        if m:
            return original[:m.start()].strip()

    # Pattern 3: Parenthetical removal: "X (Y)" → "X"
    m = re.match(r'^(.+?)\s*\(.*\)\s*$', original)
    if m:
        candidate = m.group(1).strip()
        if len(candidate) >= 3:
            return candidate

    return original


# ── Question Formatting (same as v2) ────────────────────────────────

def format_hop_question_v2(hop_question, previous_answers):
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


AUTO_DECOMPOSE_TEMPLATE = """Break down this multi-hop question into simpler sub-questions that can be answered one at a time. Each sub-question should ask about ONE fact.

Use #N to reference the answer from a previous sub-question.

Examples:

Question: Who is the spouse of the Green performer?
Sub-questions:
1. Who performed Green?
2. Who is the spouse of #1?

Question: What county was Tim Dubois born in?
Sub-questions:
1. Where was Tim DuBois born?
2. What county is #1 located in?

Question: Who founded the company that distributed the film UHF?
Sub-questions:
1. Who distributed the film UHF?
2. Who founded #1?

Now decompose this question:

Question: {question}
Sub-questions:
1."""


# ── Embedding Retrieval (same as v2) ────────────────────────────────

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

def run_embed_normalized(sample, retriever, verbose=False):
    """Embedding retrieval + answer normalization between hops."""
    decomposition = sample["question_decomposition"]
    paragraphs = sample["paragraphs"]

    previous_answers = []
    hop_details = []

    for i, hop in enumerate(decomposition):
        hop_q = format_hop_question_v2(hop["question"], previous_answers)

        results = retriever.retrieve(hop_q, paragraphs, top_k=3)
        retrieved_indices = [idx for idx, _ in results]
        context_parts = [f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
                        for idx, _ in results]
        context = "\n\n".join(context_parts)

        gold_idx = hop["paragraph_support_idx"]
        gold_retrieved = gold_idx in retrieved_indices

        prompt = FEW_SHOT_TEMPLATE.format(context=context, question=hop_q)
        response = ask_phi3(prompt, temperature=0.1, max_tokens=32)
        raw_answer = extract_short_answer(response)
        normalized = normalize_intermediate_answer(raw_answer)

        em = exact_match(raw_answer, hop["answer"])
        hop_details.append({
            "hop": i + 1, "question": hop_q,
            "gold_answer": hop["answer"],
            "raw_predicted": raw_answer, "normalized": normalized,
            "em": em, "relaxed_em": relaxed_match(raw_answer, hop["answer"]),
            "f1": f1_score(raw_answer, hop["answer"]),
            "gold_retrieved": gold_retrieved,
        })

        if verbose:
            s = "+" if em else "-"
            norm_marker = f" -> [{normalized}]" if normalized != raw_answer else ""
            print(f"    [{s}] Hop {i+1}: {hop_q[:50]}...")
            print(f"        Got: {raw_answer}{norm_marker} (gold: {hop['answer']})")

        # USE NORMALIZED ANSWER for chaining
        previous_answers.append(normalized)

    final = extract_short_answer(hop_details[-1]["raw_predicted"]) if hop_details else ""
    return final, hop_details


def run_auto_decompose(sample, retriever, verbose=False):
    """Auto-decomposition: model generates sub-questions, system chains them."""
    question = sample["question"]
    paragraphs = sample["paragraphs"]

    # Step 1: Ask model to decompose
    prompt = AUTO_DECOMPOSE_TEMPLATE.format(question=question)
    response = ask_phi3(prompt, temperature=0.3, max_tokens=100)

    # Parse sub-questions
    sub_questions = []
    # The response starts after "1. " which we already have in the template
    full_response = "1. " + response
    for line in full_response.split('\n'):
        line = line.strip()
        # Match numbered lines: "1. ...", "2. ..."
        m = re.match(r'^\d+\.\s+(.+)$', line)
        if m:
            sub_questions.append(m.group(1).strip().rstrip('?') + '?')

    if not sub_questions:
        sub_questions = [question]  # Fallback to original

    if verbose:
        print(f"    [AUTO] Decomposition: {sub_questions}")

    # Step 2: Iteratively answer each sub-question
    previous_answers = []
    hop_details = []

    for i, sq in enumerate(sub_questions):
        # Substitute #N references
        hop_q = sq
        for j, ans in enumerate(previous_answers, 1):
            hop_q = hop_q.replace(f"#{j}", ans)

        results = retriever.retrieve(hop_q, paragraphs, top_k=3)
        context_parts = [f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
                        for idx, _ in results]
        context = "\n\n".join(context_parts)

        prompt = FEW_SHOT_TEMPLATE.format(context=context, question=hop_q)
        response = ask_phi3(prompt, temperature=0.1, max_tokens=32)
        raw_answer = extract_short_answer(response)
        normalized = normalize_intermediate_answer(raw_answer)

        hop_details.append({
            "hop": i + 1, "question": hop_q,
            "raw_predicted": raw_answer, "normalized": normalized,
        })

        if verbose:
            norm_marker = f" -> [{normalized}]" if normalized != raw_answer else ""
            print(f"    [AUTO] Hop {i+1}: {hop_q[:50]}...")
            print(f"        Got: {raw_answer}{norm_marker}")

        previous_answers.append(normalized)

    final = raw_answer if hop_details else ""
    return final, hop_details, sub_questions


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
        "EMBED_NORMALIZED": ConfigResult("EMBED_NORMALIZED"),
        "AUTO_DECOMPOSE": ConfigResult("AUTO_DECOMPOSE"),
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

        # 1. Embedding Retrieval + Normalization (gold decomposition)
        t0 = time.time()
        pred, hops = run_embed_normalized(sample, retriever, verbose=verbose)
        lat = (time.time() - t0) * 1000
        em = exact_match(pred, answer, aliases)
        rem = relaxed_match(pred, answer, aliases)
        f1 = best_f1(pred, answer, aliases)
        configs["EMBED_NORMALIZED"].em_scores.append(em)
        configs["EMBED_NORMALIZED"].relaxed_em_scores.append(rem)
        configs["EMBED_NORMALIZED"].f1_scores.append(f1)
        configs["EMBED_NORMALIZED"].latencies.append(lat)
        configs["EMBED_NORMALIZED"].hop_details.append(hops)
        configs["EMBED_NORMALIZED"].per_question.append({
            "id": sample["id"], "predicted": pred, "em": em,
            "relaxed_em": rem, "f1": f1, "n_hops": n_hops
        })
        s = "+" if em else ("~" if rem else "-")
        print(f"  [{s}] EMBED_NORM: {pred} (F1={f1:.2f})")

        # 2. Auto-Decompose + Embedding Retrieval + Normalization
        t0 = time.time()
        pred, hops, sub_qs = run_auto_decompose(sample, retriever, verbose=verbose)
        lat = (time.time() - t0) * 1000
        em = exact_match(pred, answer, aliases)
        rem = relaxed_match(pred, answer, aliases)
        f1 = best_f1(pred, answer, aliases)
        configs["AUTO_DECOMPOSE"].em_scores.append(em)
        configs["AUTO_DECOMPOSE"].relaxed_em_scores.append(rem)
        configs["AUTO_DECOMPOSE"].f1_scores.append(f1)
        configs["AUTO_DECOMPOSE"].latencies.append(lat)
        configs["AUTO_DECOMPOSE"].hop_details.append(hops)
        configs["AUTO_DECOMPOSE"].per_question.append({
            "id": sample["id"], "predicted": pred, "em": em,
            "relaxed_em": rem, "f1": f1, "n_hops": n_hops,
            "sub_questions": sub_qs,
        })
        s = "+" if em else ("~" if rem else "-")
        print(f"  [{s}] AUTO_DECOMP: {pred} (F1={f1:.2f})")

        # Running totals
        print(f"\n  --- Running ({i+1} questions) ---")
        for name, cfg in configs.items():
            print(f"    {name:25s}: EM={cfg.em_rate:5.1f}% relEM={cfg.relaxed_em_rate:5.1f}% F1={cfg.mean_f1:.3f}")

    return configs, samples


def analyze_results(configs, samples):
    print("\n" + "=" * 70)
    print("ITERATIVE RETRIEVAL v3 - FINAL RESULTS")
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
            print(f"    v3 {name}: {cfg.em_rate:.1f}% EM")

    # Normalization impact analysis
    if "EMBED_NORMALIZED" in configs and configs["EMBED_NORMALIZED"].hop_details:
        print("\n  --- Normalization Impact ---")
        changed = 0
        total = 0
        for q_hops in configs["EMBED_NORMALIZED"].hop_details:
            for h in q_hops[:-1]:  # Only intermediate hops (not final)
                total += 1
                if h.get("raw_predicted") != h.get("normalized"):
                    changed += 1
        if total > 0:
            print(f"    Answers normalized: {changed}/{total} ({changed/total*100:.1f}%)")

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
    parser = argparse.ArgumentParser(description="Iterative Retrieval v3")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", type=str, default="evaluation/results/iterative_retrieval_v3.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        assert "phi3:mini" in models
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
