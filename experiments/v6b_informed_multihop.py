#!/usr/bin/env python3
"""
v6b: Informed Multi-Hop — Decomposition quality when hop count is given.

v6 showed that "blind" auto-decomposition (model decides hop count) fails.
This experiment isolates DECOMPOSITION QUALITY by telling the model exactly
how many sub-questions to produce.

This tests: Can Qwen 7B decompose 3-4 hop questions as well as it does 2-hop
when given the correct hop count?

Three conditions:
  1. single_pass: All paragraphs + full question (baseline)
  2. gold_qwen: Gold decomposition + Qwen extraction (ceiling)
  3. informed_auto: "Break into exactly N sub-questions" + Qwen extraction

If informed_auto ≈ gold_qwen, the bottleneck is hop count prediction.
If informed_auto << gold_qwen, decomposition quality degrades with complexity.
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

# Hop-count-specific decomposition templates
DECOMPOSE_2HOP = """Break this complex question into exactly 2 simple sub-questions. The first should identify a key entity, the second should find the final answer about that entity.

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

Question: Which company owns the manufacturer of Learjet 60?
1. Who manufactured Learjet 60?
2. Which company owns #1?

Question: What is the capital of the country where the Yenisei River originates?
1. In what country does the Yenisei River originate?
2. What is the capital of #1?

Question: Who directed the film that features the song "Let It Go"?
1. What film features the song "Let It Go"?
2. Who directed #1?

Question: What award did the author of The Red Tree receive?
1. Who is the author of The Red Tree?
2. What award did #1 receive?

Question: {question}
1."""

DECOMPOSE_3HOP = """Break this complex question into exactly 3 simple sub-questions. Each should find one fact, building on previous answers.

Use #1, #2 to reference answers from earlier sub-questions.

Examples:

Question: What is the birthplace of the man who voices Stan on the series that includes the episode The Hobbit?
1. What series includes the episode The Hobbit?
2. Who voices Stan on #1?
3. What is the birthplace of #2?

Question: In what county is the birthplace of the child of the woman who portrayed Corliss Archer?
1. Who portrayed Corliss Archer?
2. Who is the child of #1?
3. In what county was #2 born?

Question: What record label is the performer of the theme song for the series created by Matt Groening on?
1. What series was created by Matt Groening?
2. Who performs the theme song for #1?
3. What record label is #2 on?

Question: Who is the president of the country that established the Truth and Friendship Commission with Indonesia?
1. What commission was established with Indonesia?
2. What country established #1 with Indonesia?
3. Who is the president of #2?

Question: {question}
1."""

DECOMPOSE_4HOP = """Break this complex question into exactly 4 simple sub-questions. Each should find one fact, building on previous answers.

Use #1, #2, #3 to reference answers from earlier sub-questions.

Examples:

Question: Who is the president of the newly declared independent country that established the Truth and Friendship Commission with the country containing the airport that includes Lion Air?
1. What airport includes Lion Air?
2. What country contains #1?
3. What country established the Truth and Friendship Commission with #2?
4. Who is the president of #3?

Question: Where was the director of the film that won Best Picture at the ceremony hosted by Billy Crystal born?
1. What ceremony was hosted by Billy Crystal?
2. What film won Best Picture at #1?
3. Who directed #2?
4. Where was #3 born?

Question: What was the limit on games per developer for the console whose successor had an advantage in its processor over the Genesis?
1. What console's successor had a processor advantage over the Genesis?
2. What was the successor console to #1?
3. What advantage did #2's processor have over the Genesis?
4. What was the limit on games per developer for #1?

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
        para_texts = [f"{p['title']} {p['paragraph_text']}" for p in paragraphs]
        query_emb = self.model.encode([query_text], normalize_embeddings=True)
        para_embs = self.model.encode(para_texts, normalize_embeddings=True)
        sims = np.dot(para_embs, query_emb.T).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(int(idx), float(sims[idx])) for idx in top_indices]


# ── Hop Notation ─────────────────────────────────────────────────────

RELATION_TO_QUESTION = {
    "performer": "Who performed {}?",
    "spouse": "Who is the spouse of {}?",
    "founded by": "Who founded {}?",
    "founder": "Who founded {}?",
    "place of birth": "Where was {} born?",
    "birthplace": "Where was {} born?",
    "country": "What country is {} in?",
    "country of origin": "What country is {} from?",
    "headquarters location": "Where is {} headquartered?",
    "employer": "Who employs {}?",
    "manufacturer": "Who manufactured {}?",
    "owned by": "Who owns {}?",
    "developer": "Who developed {}?",
    "author": "Who is the author of {}?",
    "director": "Who directed {}?",
    "capital": "What is the capital of {}?",
    "located in": "Where is {} located?",
    "child": "Who is the child of {}?",
    "parent": "Who is the parent of {}?",
    "part of": "What is {} part of?",
    "member of": "What is {} a member of?",
    "genre": "What genre is {}?",
    "award received": "What award did {} receive?",
    "language": "What language does {} speak?",
    "continent": "What continent is {} on?",
    "league": "What league does {} play in?",
    "shares border with": "What borders {}?",
    "educated at": "Where was {} educated?",
    "screenwriter": "Who wrote the screenplay for {}?",
    "producer": "Who produced {}?",
    "cast member": "Who is a cast member of {}?",
    "narrative location": "Where is {} set?",
    "publication date": "When was {} published?",
    "lyrics by": "Who wrote the lyrics for {}?",
    "composer": "Who composed {}?",
    "record label": "What record label is {} on?",
    "has part": "What is a part of {}?",
}

def format_hop_question(hop_notation, previous_answers=None):
    if ">>" not in hop_notation:
        return hop_notation
    parts = hop_notation.split(">>")
    subject = parts[0].strip()
    relation = parts[1].strip() if len(parts) > 1 else ""
    if previous_answers:
        for j, ans in enumerate(previous_answers, 1):
            subject = subject.replace(f"#{j}", ans)
    relation_lower = relation.lower().strip()
    if relation_lower in RELATION_TO_QUESTION:
        return RELATION_TO_QUESTION[relation_lower].format(subject)
    else:
        return f"What is the {relation} of {subject}?"


# ── Decomposition ───────────────────────────────────────────────────

DECOMPOSE_TEMPLATES = {
    2: DECOMPOSE_2HOP,
    3: DECOMPOSE_3HOP,
    4: DECOMPOSE_4HOP,
}

def decompose_informed(question, n_hops):
    """Decompose with the correct hop count specified in the template."""
    template = DECOMPOSE_TEMPLATES[n_hops]
    response = ask_model(
        template.format(question=question),
        model="qwen2.5:7b",
        temperature=0.1,
        max_tokens=200
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

    # Enforce exact hop count: take first n_hops, pad if needed
    if len(sub_questions) >= n_hops:
        return sub_questions[:n_hops]
    elif len(sub_questions) >= 1:
        # Pad with the original question as fallback
        while len(sub_questions) < n_hops:
            sub_questions.append(question)
        return sub_questions
    else:
        return [question] * n_hops


# ── Pipeline ────────────────────────────────────────────────────────

def run_pipeline(sample, retriever, decomposition, model="qwen2.5:7b",
                 top_k=3, verbose=False):
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(decomposition):
        hop_q = hop if isinstance(hop, str) else hop["question"]
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
        response = ask_model(prompt, model=model, temperature=0.1, max_tokens=32)
        answer = extract_short_answer(response)

        gold_answer = gold_decomp[i]["answer"] if i < len(gold_decomp) else "N/A"
        em = exact_match(answer, gold_answer) if gold_answer != "N/A" else None

        hop_details.append({
            "hop": i + 1, "question": hop_q,
            "gold_answer": gold_answer, "predicted": answer,
            "em": em, "gold_retrieved": gold_retrieved,
        })
        previous_answers.append(answer)

    final = answer if hop_details else ""
    return final, hop_details


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="v6b: Informed Multi-Hop Decomposition")
    parser.add_argument("--limit-per-hop", type=int, default=30)
    parser.add_argument("--output", type=str, default="results/v6b_informed_multihop.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from datasets import load_dataset

    r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    models = [m["name"] for m in r.json().get("models", [])]
    print(f"Ollama models: {models}")

    print("Loading MuSiQue validation set...")
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")

    by_hops = {2: [], 3: [], 4: []}
    for s in ds:
        if not s.get("answerable", True):
            continue
        n_hops = len(s["question_decomposition"])
        if n_hops in by_hops:
            by_hops[n_hops].append(s)

    for h in [2, 3, 4]:
        n = min(args.limit_per_hop, len(by_hops[h]))
        print(f"  {h}-hop: {len(by_hops[h])} available, using {n}")

    retriever = EmbeddingRetriever()
    retriever._load_model()

    # Results: (hop_count, config) -> lists
    results = {}
    for h in [2, 3, 4]:
        for c in ["single_pass", "gold_qwen", "informed_auto"]:
            results[(h, c)] = {
                "em": [], "rem": [], "f1": [], "per_q": [], "hops": [],
                "decomp_quality": [],
            }

    start = time.time()
    total_done = 0
    total_q = sum(min(args.limit_per_hop, len(by_hops[h])) for h in [2, 3, 4])

    for hop_count in [2, 3, 4]:
        samples = by_hops[hop_count][:args.limit_per_hop]
        print(f"\n{'='*70}")
        print(f"  {hop_count}-HOP QUESTIONS (n={len(samples)})")
        print(f"{'='*70}")

        for i, sample in enumerate(samples):
            question = sample["question"]
            answer = sample["answer"]
            aliases = sample.get("answer_aliases", [])
            gold_decomp = sample["question_decomposition"]

            # 1. Single pass
            all_ctx = "\n\n".join(f"[{p['title']}] {p['paragraph_text']}"
                                 for p in sample["paragraphs"])
            prompt = PROMPT_SINGLE_PASS.format(context=all_ctx, question=question)
            response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
            pred_sp = extract_short_answer(response)
            em_sp = exact_match(pred_sp, answer, aliases)
            rem_sp = relaxed_match(pred_sp, answer, aliases)
            f1_sp = best_f1(pred_sp, answer, aliases)
            results[(hop_count, "single_pass")]["em"].append(em_sp)
            results[(hop_count, "single_pass")]["rem"].append(rem_sp)
            results[(hop_count, "single_pass")]["f1"].append(f1_sp)

            # 2. Gold decomp + Qwen
            pred_g, hops_g = run_pipeline(sample, retriever, gold_decomp, verbose=args.verbose)
            em_g = exact_match(pred_g, answer, aliases)
            rem_g = relaxed_match(pred_g, answer, aliases)
            f1_g = best_f1(pred_g, answer, aliases)
            results[(hop_count, "gold_qwen")]["em"].append(em_g)
            results[(hop_count, "gold_qwen")]["rem"].append(rem_g)
            results[(hop_count, "gold_qwen")]["f1"].append(f1_g)
            results[(hop_count, "gold_qwen")]["hops"].append(hops_g)

            # 3. Informed auto decomp
            auto_sub_qs = decompose_informed(question, hop_count)
            auto_decomp = [{"question": q} for q in auto_sub_qs]
            pred_a, hops_a = run_pipeline(sample, retriever, auto_decomp, verbose=args.verbose)
            em_a = exact_match(pred_a, answer, aliases)
            rem_a = relaxed_match(pred_a, answer, aliases)
            f1_a = best_f1(pred_a, answer, aliases)
            results[(hop_count, "informed_auto")]["em"].append(em_a)
            results[(hop_count, "informed_auto")]["rem"].append(rem_a)
            results[(hop_count, "informed_auto")]["f1"].append(f1_a)
            results[(hop_count, "informed_auto")]["hops"].append(hops_a)
            results[(hop_count, "informed_auto")]["decomp_quality"].append({
                "n_gold": len(gold_decomp),
                "n_auto": len(auto_sub_qs),
                "match": len(auto_sub_qs) == len(gold_decomp),
            })

            total_done += 1
            n = i + 1
            sp_em = sum(results[(hop_count, "single_pass")]["em"]) / n * 100
            g_em = sum(results[(hop_count, "gold_qwen")]["em"]) / n * 100
            a_em = sum(results[(hop_count, "informed_auto")]["em"]) / n * 100
            elapsed = time.time() - start
            eta = elapsed / total_done * (total_q - total_done)

            sp_m = "+" if em_sp else "-"
            g_m = "+" if em_g else "-"
            a_m = "+" if em_a else "-"
            print(f"  [{n:3d}/{len(samples)}] [{sp_m}{g_m}{a_m}] SP={sp_em:5.1f}% G={g_em:5.1f}% A={a_em:5.1f}% | {question[:45]}... ETA:{eta/60:.0f}m")

    elapsed = time.time() - start

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"INFORMED MULTI-HOP RESULTS")
    print(f"{'='*70}")

    print(f"\n  Exact Match:")
    print(f"  {'Config':15s} | {'2-hop':>10s} | {'3-hop':>10s} | {'4-hop':>10s}")
    print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for config in ["single_pass", "gold_qwen", "informed_auto"]:
        row = f"  {config:15s} |"
        for h in [2, 3, 4]:
            r = results[(h, config)]
            n = len(r["em"])
            em = sum(r["em"]) / n * 100 if n else 0
            row += f" {em:5.1f}% ({n:2d}) |"
        print(row)

    print(f"\n  Relaxed EM:")
    print(f"  {'Config':15s} | {'2-hop':>10s} | {'3-hop':>10s} | {'4-hop':>10s}")
    print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for config in ["single_pass", "gold_qwen", "informed_auto"]:
        row = f"  {config:15s} |"
        for h in [2, 3, 4]:
            r = results[(h, config)]
            n = len(r["rem"])
            em = sum(r["rem"]) / n * 100 if n else 0
            row += f" {em:5.1f}%      |"
        print(row)

    # Auto-decomp gap
    print(f"\n  Auto-decomp gap (gold - informed_auto):")
    for h in [2, 3, 4]:
        g = sum(results[(h, "gold_qwen")]["em"]) / len(results[(h, "gold_qwen")]["em"]) * 100
        a = sum(results[(h, "informed_auto")]["em"]) / len(results[(h, "informed_auto")]["em"]) * 100
        print(f"    {h}-hop: {g-a:+.1f}% (gold={g:.1f}%, auto={a:.1f}%)")

    # Per-hop retrieval
    print(f"\n  Per-hop retrieval (gold paragraph found):")
    for h in [2, 3, 4]:
        for config in ["gold_qwen", "informed_auto"]:
            all_hops = results[(h, config)]["hops"]
            if not all_hops:
                continue
            max_hop = max(len(hd) for hd in all_hops)
            hop_retr = []
            for hi in range(max_hop):
                found = sum(1 for hd in all_hops if hi < len(hd) and hd[hi].get("gold_retrieved"))
                total = sum(1 for hd in all_hops if hi < len(hd))
                if total > 0:
                    hop_retr.append(f"H{hi+1}:{found}/{total}")
            print(f"    {h}-hop {config}: {', '.join(hop_retr)}")

    # Comparison with v6 blind
    print(f"\n  Comparison: blind vs informed auto-decomposition")
    print(f"  (v6 blind results would show here if you ran both)")

    print(f"\n  Total time: {elapsed/60:.1f} min")

    # ── Save ─────────────────────────────────────────────────────────
    output = {
        "metadata": {
            "experiment": "v6b_informed_multihop",
            "limit_per_hop": args.limit_per_hop,
            "elapsed_seconds": elapsed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": {},
    }

    for h in [2, 3, 4]:
        output["summary"][f"{h}_hop"] = {}
        for config in ["single_pass", "gold_qwen", "informed_auto"]:
            r = results[(h, config)]
            n = len(r["em"])
            output["summary"][f"{h}_hop"][config] = {
                "em": sum(r["em"]) / n * 100 if n else 0,
                "relaxed_em": sum(r["rem"]) / n * 100 if n else 0,
                "f1": sum(r["f1"]) / n if n else 0,
                "n": n,
            }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
