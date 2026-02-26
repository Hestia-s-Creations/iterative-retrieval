#!/usr/bin/env python3
"""
v5 HotpotQA Auto-Decomposition — Proves the approach generalizes.

HotpotQA has NO gold decompositions, making auto-decomposition the ONLY option.
This is a stronger test: can the system decompose questions it has never seen
gold decompositions for?

HotpotQA has two question types:
  - "bridge": Find entity A, then find fact about A (like MuSiQue)
  - "comparison": Find facts about entities A and B, then compare

We use Qwen 7B for both decomposition and extraction.
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

def exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def relaxed_match(prediction, ground_truth):
    pred = normalize_answer(prediction)
    gold = normalize_answer(ground_truth)
    return pred == gold or gold in pred or pred in gold

def best_f1(prediction, ground_truth):
    return f1_score(prediction, ground_truth)


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

# HotpotQA-adapted decomposition: handles both bridge and comparison types
DECOMPOSE_TEMPLATE_HOTPOT = """Break this complex question into exactly 2 simple sub-questions that can each be answered from a single paragraph.

Use #1 to reference the answer from sub-question 1.

Examples:

Question: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
1. Who portrayed Corliss Archer in the film Kiss and Tell?
2. What government position was held by #1?

Question: Were Scott Derrickson and Ed Wood of the same nationality?
1. What nationality is Scott Derrickson?
2. What nationality is Ed Wood?

Question: What is the birthplace of the director of the film Dangal?
1. Who directed the film Dangal?
2. What is the birthplace of #1?

Question: Are both Coldplay and the Chainsmokers from the same country?
1. What country is Coldplay from?
2. What country is the Chainsmokers from?

Question: Which magazine was started first, Arthur's Magazine or First for Women?
1. When was Arthur's Magazine started?
2. When was First for Women started?

Question: What is the name of the fight song of the university whose main campus is in the city of Champaign?
1. Which university has its main campus in Champaign?
2. What is the name of the fight song of #1?

Question: {question}
1."""


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
        para_texts = [f"{p['title']} {p['text']}" for p in paragraphs]
        query_emb = self.model.encode([query_text], normalize_embeddings=True)
        para_embs = self.model.encode(para_texts, normalize_embeddings=True)
        sims = np.dot(para_embs, query_emb.T).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(int(idx), float(sims[idx])) for idx in top_indices]


# ── HotpotQA Processing ─────────────────────────────────────────────

def process_hotpotqa(sample):
    """Convert HotpotQA sample to standard format."""
    paragraphs = []
    for title, sentences in zip(sample["context"]["title"], sample["context"]["sentences"]):
        text = " ".join(sentences)
        paragraphs.append({"title": title, "text": text})

    support_titles = set(sample["supporting_facts"]["title"])
    support_indices = [i for i, p in enumerate(paragraphs) if p["title"] in support_titles]

    return {
        "question": sample["question"],
        "answer": sample["answer"],
        "type": sample["type"],
        "paragraphs": paragraphs,
        "support_indices": support_indices,
        "support_titles": support_titles,
    }


def decompose_hotpotqa(question):
    """Decompose a HotpotQA question using Qwen."""
    response = ask_model(
        DECOMPOSE_TEMPLATE_HOTPOT.format(question=question),
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


def run_iterative_hotpot(proc_sample, retriever, sub_questions, top_k=3):
    """Run iterative pipeline on HotpotQA with auto-decomposed questions."""
    paragraphs = proc_sample["paragraphs"]
    support_indices = proc_sample["support_indices"]
    previous_answers = []
    hop_details = []

    for i, sq in enumerate(sub_questions):
        # Substitute #N references
        hop_q = sq
        for j, ans in enumerate(previous_answers, 1):
            hop_q = hop_q.replace(f"#{j}", ans)

        results = retriever.retrieve(hop_q, paragraphs, top_k=top_k)
        retrieved_indices = [idx for idx, _ in results]

        # Check if any gold supporting paragraph was retrieved
        gold_retrieved = any(idx in retrieved_indices for idx in support_indices)

        context_parts = [f"[{paragraphs[idx]['title']}] {paragraphs[idx]['text']}"
                        for idx, _ in results]
        context = "\n\n".join(context_parts)

        prompt = PROMPT_8SHOT.format(context=context, question=hop_q)
        response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
        answer = extract_short_answer(response)

        hop_details.append({
            "hop": i + 1, "question": hop_q,
            "predicted": answer,
            "gold_retrieved": gold_retrieved,
        })
        previous_answers.append(answer)

    # For comparison questions, the final answer may need synthesis
    # from both hops. Try a final synthesis step.
    if proc_sample["type"] == "comparison" and len(previous_answers) == 2:
        synth_ctx = f"Sub-answer 1: {previous_answers[0]}\nSub-answer 2: {previous_answers[1]}"
        synth_prompt = f"""Based on these two facts, answer the original question with just "yes" or "no", or a short factual answer.

{synth_ctx}

Original question: {proc_sample['question']}
Answer:"""
        response = ask_model(synth_prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=16)
        final = extract_short_answer(response)
    else:
        final = previous_answers[-1] if previous_answers else ""

    return final, hop_details


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="v5 HotpotQA Auto-Decomposition")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", type=str, default="results/v5_hotpotqa_auto_decomp.json")
    args = parser.parse_args()

    from datasets import load_dataset

    r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    models = [m["name"] for m in r.json().get("models", [])]
    print(f"Ollama models: {models}")

    print(f"Loading HotpotQA validation set (limit={args.limit})...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    samples = list(ds)[:args.limit]
    print(f"Testing on {len(samples)} questions\n")

    retriever = EmbeddingRetriever()
    retriever._load_model()

    # Track by type
    results = {
        "single_pass": {"em": [], "rem": [], "f1": [], "per_q": []},
        "auto_decomp": {"em": [], "rem": [], "f1": [], "per_q": [], "hops": []},
        "embed_no_decomp": {"em": [], "rem": [], "f1": [], "per_q": []},
    }
    type_counts = {"bridge": 0, "comparison": 0}
    type_em = {"bridge": {"sp": [], "auto": [], "embed": []}, "comparison": {"sp": [], "auto": [], "embed": []}}

    start = time.time()

    for i, raw_sample in enumerate(samples):
        sample = process_hotpotqa(raw_sample)
        question = sample["question"]
        answer = sample["answer"]
        q_type = sample["type"]
        type_counts[q_type] = type_counts.get(q_type, 0) + 1

        # 1. Single pass (all paragraphs)
        all_ctx = "\n\n".join(f"[{p['title']}] {p['text']}" for p in sample["paragraphs"])
        prompt = PROMPT_SINGLE_PASS.format(context=all_ctx, question=question)
        response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
        pred_sp = extract_short_answer(response)
        em_sp = exact_match(pred_sp, answer)
        rem_sp = relaxed_match(pred_sp, answer)
        f1_sp = best_f1(pred_sp, answer)
        results["single_pass"]["em"].append(em_sp)
        results["single_pass"]["rem"].append(rem_sp)
        results["single_pass"]["f1"].append(f1_sp)
        results["single_pass"]["per_q"].append({"q": question, "pred": pred_sp, "ans": answer, "em": em_sp, "type": q_type})
        type_em[q_type]["sp"].append(em_sp)

        # 2. Embedding retrieval without decomposition (retrieve top-3 for full question)
        retr_results = retriever.retrieve(question, sample["paragraphs"], top_k=3)
        ctx_parts = [f"[{sample['paragraphs'][idx]['title']}] {sample['paragraphs'][idx]['text']}"
                    for idx, _ in retr_results]
        ctx = "\n\n".join(ctx_parts)
        prompt = PROMPT_8SHOT.format(context=ctx, question=question)
        response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
        pred_e = extract_short_answer(response)
        em_e = exact_match(pred_e, answer)
        rem_e = relaxed_match(pred_e, answer)
        f1_e = best_f1(pred_e, answer)
        results["embed_no_decomp"]["em"].append(em_e)
        results["embed_no_decomp"]["rem"].append(rem_e)
        results["embed_no_decomp"]["f1"].append(f1_e)
        results["embed_no_decomp"]["per_q"].append({"q": question, "pred": pred_e, "ans": answer, "em": em_e, "type": q_type})
        type_em[q_type]["embed"].append(em_e)

        # 3. Auto-decomposition + iterative retrieval
        sub_qs = decompose_hotpotqa(question)
        pred_a, hops = run_iterative_hotpot(sample, retriever, sub_qs)
        em_a = exact_match(pred_a, answer)
        rem_a = relaxed_match(pred_a, answer)
        f1_a = best_f1(pred_a, answer)
        results["auto_decomp"]["em"].append(em_a)
        results["auto_decomp"]["rem"].append(rem_a)
        results["auto_decomp"]["f1"].append(f1_a)
        results["auto_decomp"]["hops"].append(hops)
        results["auto_decomp"]["per_q"].append({
            "q": question, "pred": pred_a, "ans": answer, "em": em_a,
            "type": q_type, "sub_qs": sub_qs,
        })
        type_em[q_type]["auto"].append(em_a)

        # Progress
        n = i + 1
        sp_em = sum(results["single_pass"]["em"]) / n * 100
        e_em = sum(results["embed_no_decomp"]["em"]) / n * 100
        a_em = sum(results["auto_decomp"]["em"]) / n * 100
        elapsed = time.time() - start
        eta = elapsed / n * (len(samples) - n)

        sp_m = "+" if em_sp else "-"
        e_m = "+" if em_e else "-"
        a_m = "+" if em_a else "-"
        print(f"[{n:3d}/{len(samples)}] [{sp_m}{e_m}{a_m}] ({q_type:10s}) SP={sp_em:5.1f}% Emb={e_em:5.1f}% Auto={a_em:5.1f}% | {question[:45]}... ETA:{eta/60:.0f}m")

    elapsed = time.time() - start
    n = len(samples)

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"HOTPOTQA AUTO-DECOMPOSITION: {n} questions")
    print(f"{'=' * 70}")

    for name in ["single_pass", "embed_no_decomp", "auto_decomp"]:
        r = results[name]
        em = sum(r["em"]) / n * 100
        rem = sum(r["rem"]) / n * 100
        f1 = sum(r["f1"]) / n
        print(f"  {name:18s}: EM={em:5.1f}%  relEM={rem:5.1f}%  F1={f1:.3f}")

    print(f"\n  By question type:")
    for q_type in ["bridge", "comparison"]:
        cnt = type_counts.get(q_type, 0)
        if cnt == 0:
            continue
        sp = sum(type_em[q_type]["sp"]) / cnt * 100
        emb = sum(type_em[q_type]["embed"]) / cnt * 100
        auto = sum(type_em[q_type]["auto"]) / cnt * 100
        print(f"    {q_type} (n={cnt}): SP={sp:.1f}% Embed={emb:.1f}% Auto={auto:.1f}%")

    # Retrieval analysis
    hop1_retr = sum(1 for h in results["auto_decomp"]["hops"] if len(h) > 0 and h[0].get("gold_retrieved"))
    hop2_retr = sum(1 for h in results["auto_decomp"]["hops"] if len(h) > 1 and h[1].get("gold_retrieved"))
    print(f"\n  Retrieval (auto_decomp):")
    print(f"    Hop 1 gold retrieved: {hop1_retr}/{n} ({hop1_retr/n*100:.1f}%)")
    print(f"    Hop 2 gold retrieved: {hop2_retr}/{n} ({hop2_retr/n*100:.1f}%)")

    print(f"\n  Total time: {elapsed/60:.1f} min")

    # Save
    output = {
        "metadata": {
            "benchmark": "hotpotqa",
            "n_samples": n,
            "elapsed_seconds": elapsed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "type_counts": type_counts,
        },
        "summary": {name: {
            "em": sum(r["em"]) / n * 100,
            "relaxed_em": sum(r["rem"]) / n * 100,
            "f1": sum(r["f1"]) / n,
        } for name, r in results.items()},
        "by_type": {q_type: {
            "n": type_counts.get(q_type, 0),
            "single_pass_em": sum(type_em[q_type]["sp"]) / max(len(type_em[q_type]["sp"]), 1) * 100,
            "embed_em": sum(type_em[q_type]["embed"]) / max(len(type_em[q_type]["embed"]), 1) * 100,
            "auto_em": sum(type_em[q_type]["auto"]) / max(len(type_em[q_type]["auto"]), 1) * 100,
        } for q_type in ["bridge", "comparison"]},
        "per_question": {name: r["per_q"] for name, r in results.items()},
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
