#!/usr/bin/env python3
"""
v5 Scale Validation — Run key configs on 100 MuSiQue questions.

Addresses the N=30 caveat. Tests only essential configs to save time:
  1. single_pass:  No decomposition baseline
  2. auto_qwen:    Fully autonomous (THE KEY CLAIM)
  3. gold_qwen:    Gold decomposition reference

Expected runtime: ~20-30 min for 100 questions × 3 configs.
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


# ── Prompts ──────────────────────────────────────────────────────────

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

PROMPT_SINGLE_PASS = """Answer this question using the provided context. Give ONLY the answer - a name, place, or fact. No explanation.

Context:
{context}

Question: {question}
Answer:"""


# ── Decomposition ────────────────────────────────────────────────────

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
        model="qwen2.5:7b",
        temperature=0.1,
        max_tokens=80
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


# ── Retrieval ────────────────────────────────────────────────────────

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


# ── Pipeline ─────────────────────────────────────────────────────────

def run_iterative(sample, retriever, decomposition, top_k=3):
    """Run iterative pipeline with given decomposition."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(decomposition):
        hop_q = hop["question"]
        for j, ans in enumerate(previous_answers, 1):
            hop_q = hop_q.replace(f"#{j}", ans)
        if ">>" in hop_q:
            hop_q = format_hop_question(hop_q, [])

        results = retriever.retrieve(hop_q, paragraphs, top_k=top_k)
        retrieved_indices = [idx for idx, _ in results]
        context_parts = [f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
                        for idx, _ in results]
        context = "\n\n".join(context_parts)

        gold_idx = gold_decomp[i]["paragraph_support_idx"] if i < len(gold_decomp) else -1
        gold_retrieved = gold_idx in retrieved_indices

        prompt = PROMPT_8SHOT.format(context=context, question=hop_q)
        response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
        answer = extract_short_answer(response)

        gold_answer = gold_decomp[i]["answer"] if i < len(gold_decomp) else "N/A"
        hop_details.append({
            "hop": i + 1, "question": hop_q,
            "gold_answer": gold_answer,
            "predicted": answer,
            "em": exact_match(answer, gold_answer) if gold_answer != "N/A" else None,
            "gold_retrieved": gold_retrieved,
        })
        previous_answers.append(answer)

    return answer if hop_details else "", hop_details


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="v5 Scale Validation")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=str, default="results/v5_scale_100.json")
    args = parser.parse_args()

    from datasets import load_dataset

    # Check Ollama
    r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    models = [m["name"] for m in r.json().get("models", [])]
    print(f"Ollama models: {models}")

    print(f"Loading MuSiQue validation set (limit={args.limit})...")
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    samples = [s for s in ds if s.get("answerable", True)][:args.limit]
    print(f"Testing on {len(samples)} answerable questions\n")

    retriever = EmbeddingRetriever()
    retriever._load_model()

    # Results tracking
    results = {
        "single_pass": {"em": [], "rem": [], "f1": [], "per_q": []},
        "gold_qwen": {"em": [], "rem": [], "f1": [], "per_q": [], "hops": []},
        "auto_qwen": {"em": [], "rem": [], "f1": [], "per_q": [], "hops": []},
    }

    start = time.time()

    for i, sample in enumerate(samples):
        question = sample["question"]
        answer = sample["answer"]
        aliases = sample.get("answer_aliases", [])
        gold_decomp = sample["question_decomposition"]

        # 1. Single pass
        all_ctx = "\n\n".join(
            f"[{p['title']}] {p['paragraph_text']}" for p in sample["paragraphs"][:10]
        )
        prompt = PROMPT_SINGLE_PASS.format(context=all_ctx, question=question)
        response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
        pred = extract_short_answer(response)
        em = exact_match(pred, answer, aliases)
        rem = relaxed_match(pred, answer, aliases)
        f1 = best_f1(pred, answer, aliases)
        results["single_pass"]["em"].append(em)
        results["single_pass"]["rem"].append(rem)
        results["single_pass"]["f1"].append(f1)
        results["single_pass"]["per_q"].append({"q": question, "pred": pred, "ans": answer, "em": em})

        sp_mark = "+" if em else "-"

        # 2. Gold decomp + Qwen
        gold_hops = [{"question": hop["question"]} for hop in gold_decomp]
        pred_g, hops_g = run_iterative(sample, retriever, gold_hops)
        em_g = exact_match(pred_g, answer, aliases)
        rem_g = relaxed_match(pred_g, answer, aliases)
        f1_g = best_f1(pred_g, answer, aliases)
        results["gold_qwen"]["em"].append(em_g)
        results["gold_qwen"]["rem"].append(rem_g)
        results["gold_qwen"]["f1"].append(f1_g)
        results["gold_qwen"]["hops"].append(hops_g)
        results["gold_qwen"]["per_q"].append({"q": question, "pred": pred_g, "ans": answer, "em": em_g})

        g_mark = "+" if em_g else "-"

        # 3. Auto decomp + Qwen
        sub_qs = decompose_with_qwen(question)
        auto_hops = [{"question": sq} for sq in sub_qs]
        pred_a, hops_a = run_iterative(sample, retriever, auto_hops)
        em_a = exact_match(pred_a, answer, aliases)
        rem_a = relaxed_match(pred_a, answer, aliases)
        f1_a = best_f1(pred_a, answer, aliases)
        results["auto_qwen"]["em"].append(em_a)
        results["auto_qwen"]["rem"].append(rem_a)
        results["auto_qwen"]["f1"].append(f1_a)
        results["auto_qwen"]["hops"].append(hops_a)
        results["auto_qwen"]["per_q"].append({"q": question, "pred": pred_a, "ans": answer, "em": em_a, "sub_qs": sub_qs})

        a_mark = "+" if em_a else "-"

        # Progress
        n = i + 1
        sp_em = sum(results["single_pass"]["em"]) / n * 100
        g_em = sum(results["gold_qwen"]["em"]) / n * 100
        a_em = sum(results["auto_qwen"]["em"]) / n * 100
        elapsed = time.time() - start
        eta = elapsed / n * (len(samples) - n)

        print(f"[{n:3d}/{len(samples)}] [{sp_mark}{g_mark}{a_mark}] SP={sp_em:5.1f}% Gold={g_em:5.1f}% Auto={a_em:5.1f}% | {question[:50]}... ETA:{eta/60:.0f}m")

    elapsed = time.time() - start

    # Final summary
    n = len(samples)
    print(f"\n{'=' * 70}")
    print(f"SCALE VALIDATION: {n} MuSiQue questions")
    print(f"{'=' * 70}")

    for name in ["single_pass", "gold_qwen", "auto_qwen"]:
        r = results[name]
        em = sum(r["em"]) / n * 100
        rem = sum(r["rem"]) / n * 100
        f1 = sum(r["f1"]) / n
        print(f"  {name:15s}: EM={em:5.1f}%  relEM={rem:5.1f}%  F1={f1:.3f}")

    gap = sum(results["gold_qwen"]["em"]) / n * 100 - sum(results["auto_qwen"]["em"]) / n * 100
    print(f"\n  Auto-decomp gap: {gap:+.1f}% EM")
    print(f"  Total time: {elapsed/60:.1f} min ({elapsed/n:.1f}s per question)")

    # Per-hop analysis
    for name in ["gold_qwen", "auto_qwen"]:
        hop1_em = sum(1 for h in results[name]["hops"] if len(h) > 0 and h[0].get("em"))
        hop2_em = sum(1 for h in results[name]["hops"] if len(h) > 1 and h[1].get("em"))
        hop1_r = sum(1 for h in results[name]["hops"] if len(h) > 0 and h[0].get("gold_retrieved"))
        hop2_r = sum(1 for h in results[name]["hops"] if len(h) > 1 and h[1].get("gold_retrieved"))
        print(f"  {name}: Hop1 EM={hop1_em}/{n} ({hop1_em/n*100:.1f}%) Retr={hop1_r}/{n} ({hop1_r/n*100:.1f}%) | Hop2 EM={hop2_em}/{n} ({hop2_em/n*100:.1f}%) Retr={hop2_r}/{n} ({hop2_r/n*100:.1f}%)")

    # Save
    output = {
        "metadata": {
            "n_samples": n,
            "elapsed_seconds": elapsed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": {name: {
            "em": sum(r["em"]) / n * 100,
            "relaxed_em": sum(r["rem"]) / n * 100,
            "f1": sum(r["f1"]) / n,
        } for name, r in results.items()},
        "per_question": {name: r["per_q"] for name, r in results.items()},
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
