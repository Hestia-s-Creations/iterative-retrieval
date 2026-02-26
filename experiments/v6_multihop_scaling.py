#!/usr/bin/env python3
"""
v6: Multi-Hop Scaling — Does decomposition benefit scale with hop count?

Hypothesis: If system decomposition takes 2-hop from 16%→42% (n=100),
it should help EVEN MORE on 3-4 hop where single-pass should be near 0%.

Tests 2/3/4-hop MuSiQue questions with:
  1. single_pass: All paragraphs + full question → model
  2. gold_qwen: Gold decomposition + Qwen 7B extraction
  3. auto_qwen: Auto decomposition + Qwen 7B extraction

Auto-decomposition uses a variable-hop template that lets the model decide
how many sub-questions are needed (tests decomposition quality on harder questions).
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
from collections import defaultdict, Counter
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

# Variable-hop decomposition template — doesn't specify the number of hops
DECOMPOSE_TEMPLATE_MULTI = """Break this complex question into simple sub-questions that can each be answered from a single paragraph. Each sub-question should find one fact.

Use #1, #2, #3 to reference answers from earlier sub-questions.

Examples:

Question: Who is the spouse of the Green performer?
1. Who performed Green?
2. Who is the spouse of #1?

Question: What is the birthplace of the man who voices Stan on the series that includes the episode The Hobbit?
1. What series includes the episode The Hobbit?
2. Who voices Stan on #1?
3. What is the birthplace of #2?

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

Question: In what county is the birthplace of the child of the woman who portrayed Corliss Archer?
1. Who portrayed Corliss Archer?
2. Who is the child of #1?
3. Where was #2 born?

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
    """Convert MuSiQue hop notation (X >> Y) to natural language question."""
    if ">>" not in hop_notation:
        return hop_notation

    parts = hop_notation.split(">>")
    subject = parts[0].strip()
    relation = parts[1].strip() if len(parts) > 1 else ""

    # Substitute #N references
    if previous_answers:
        for j, ans in enumerate(previous_answers, 1):
            subject = subject.replace(f"#{j}", ans)

    # Map relation to question template
    relation_lower = relation.lower().strip()
    if relation_lower in RELATION_TO_QUESTION:
        return RELATION_TO_QUESTION[relation_lower].format(subject)
    else:
        return f"What is the {relation} of {subject}?"


# ── Decomposition ───────────────────────────────────────────────────

def decompose_with_qwen(question, max_hops=6):
    """Use Qwen 7B to decompose a question into variable number of sub-questions."""
    response = ask_model(
        DECOMPOSE_TEMPLATE_MULTI.format(question=question),
        model="qwen2.5:7b",
        temperature=0.1,
        max_tokens=200  # More tokens for 3-4 hop decompositions
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
        return sub_questions[:max_hops]
    elif len(sub_questions) == 1:
        return [sub_questions[0], question]
    else:
        return [question]


# ── Pipeline Runner ──────────────────────────────────────────────────

def run_pipeline(sample, retriever, decomposition, model="qwen2.5:7b",
                 top_k=3, use_gold_context=False, verbose=False):
    """Run iterative pipeline with given decomposition (variable hops)."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(decomposition):
        # Get this hop's question, substitute previous answers
        hop_q = hop if isinstance(hop, str) else hop["question"]
        for j, ans in enumerate(previous_answers, 1):
            hop_q = hop_q.replace(f"#{j}", ans)

        # Handle >> relation format if present
        if ">>" in hop_q:
            hop_q = format_hop_question(hop_q, [])

        # Retrieve context
        if use_gold_context and i < len(gold_decomp):
            gold_idx = gold_decomp[i]["paragraph_support_idx"]
            p = paragraphs[gold_idx]
            context = f"[{p['title']}] {p['paragraph_text']}"
            retrieved_indices = [gold_idx]
        else:
            results = retriever.retrieve(hop_q, paragraphs, top_k=top_k)
            retrieved_indices = [idx for idx, _ in results]
            context_parts = [f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
                           for idx, _ in results]
            context = "\n\n".join(context_parts)

        # Check gold paragraph retrieval
        gold_idx = gold_decomp[i]["paragraph_support_idx"] if i < len(gold_decomp) else -1
        gold_retrieved = gold_idx in retrieved_indices

        # Extract answer
        prompt = PROMPT_8SHOT.format(context=context, question=hop_q)
        response = ask_model(prompt, model=model, temperature=0.1, max_tokens=32)
        answer = extract_short_answer(response)

        # Compare with gold
        gold_answer = gold_decomp[i]["answer"] if i < len(gold_decomp) else "N/A"
        em = exact_match(answer, gold_answer) if gold_answer != "N/A" else None

        hop_details.append({
            "hop": i + 1,
            "question": hop_q,
            "gold_answer": gold_answer,
            "predicted": answer,
            "em": em,
            "gold_retrieved": gold_retrieved,
        })

        if verbose:
            s = "+" if em else "-"
            r = "R" if gold_retrieved else "X"
            print(f"      [{s}{r}] Hop {i+1}: {hop_q[:55]}...")
            print(f"           Got: {answer} | Gold: {gold_answer}")

        previous_answers.append(answer)

    final = answer if hop_details else ""
    return final, hop_details


# ── Main Experiment ──────────────────────────────────────────────────

@dataclass
class HopResult:
    hop_count: int
    config: str
    em_scores: list = field(default_factory=list)
    relaxed_em_scores: list = field(default_factory=list)
    f1_scores: list = field(default_factory=list)
    per_question: list = field(default_factory=list)
    hop_details: list = field(default_factory=list)
    decomp_quality: list = field(default_factory=list)

    @property
    def em_rate(self):
        return sum(self.em_scores) / len(self.em_scores) * 100 if self.em_scores else 0
    @property
    def relaxed_em_rate(self):
        return sum(self.relaxed_em_scores) / len(self.relaxed_em_scores) * 100 if self.relaxed_em_scores else 0
    @property
    def mean_f1(self):
        return sum(self.f1_scores) / len(self.f1_scores) if self.f1_scores else 0


def main():
    parser = argparse.ArgumentParser(description="v6: Multi-Hop Scaling Experiment")
    parser.add_argument("--limit-per-hop", type=int, default=30,
                       help="Number of questions per hop count (default 30)")
    parser.add_argument("--output", type=str, default="results/v6_multihop_scaling.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from datasets import load_dataset

    # Check Ollama
    r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    models = [m["name"] for m in r.json().get("models", [])]
    print(f"Ollama models: {models}")
    assert any("qwen" in m for m in models), "qwen2.5:7b not found"

    # Load dataset and separate by hop count
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
        print(f"  {h}-hop: {len(by_hops[h])} available, using {min(args.limit_per_hop, len(by_hops[h]))}")

    retriever = EmbeddingRetriever()
    retriever._load_model()

    # Results by hop count and config
    results = {}
    for h in [2, 3, 4]:
        for c in ["single_pass", "gold_qwen", "auto_qwen"]:
            results[(h, c)] = HopResult(h, c)

    start = time.time()
    total_done = 0
    total_questions = sum(min(args.limit_per_hop, len(by_hops[h])) for h in [2, 3, 4])

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
            results[(hop_count, "single_pass")].em_scores.append(em_sp)
            results[(hop_count, "single_pass")].relaxed_em_scores.append(rem_sp)
            results[(hop_count, "single_pass")].f1_scores.append(f1_sp)
            results[(hop_count, "single_pass")].per_question.append({
                "q": question, "pred": pred_sp, "ans": answer, "em": em_sp,
            })

            # 2. Gold decomposition + Qwen extract
            pred_g, hops_g = run_pipeline(
                sample, retriever, gold_decomp, model="qwen2.5:7b",
                top_k=3, verbose=args.verbose
            )
            em_g = exact_match(pred_g, answer, aliases)
            rem_g = relaxed_match(pred_g, answer, aliases)
            f1_g = best_f1(pred_g, answer, aliases)
            results[(hop_count, "gold_qwen")].em_scores.append(em_g)
            results[(hop_count, "gold_qwen")].relaxed_em_scores.append(rem_g)
            results[(hop_count, "gold_qwen")].f1_scores.append(f1_g)
            results[(hop_count, "gold_qwen")].hop_details.append(hops_g)
            results[(hop_count, "gold_qwen")].per_question.append({
                "q": question, "pred": pred_g, "ans": answer, "em": em_g,
                "hops": hops_g,
            })

            # 3. Auto decomposition + Qwen extract
            auto_sub_qs = decompose_with_qwen(question)
            auto_decomp = [{"question": q} for q in auto_sub_qs]
            pred_a, hops_a = run_pipeline(
                sample, retriever, auto_decomp, model="qwen2.5:7b",
                top_k=3, verbose=args.verbose
            )
            em_a = exact_match(pred_a, answer, aliases)
            rem_a = relaxed_match(pred_a, answer, aliases)
            f1_a = best_f1(pred_a, answer, aliases)
            results[(hop_count, "auto_qwen")].em_scores.append(em_a)
            results[(hop_count, "auto_qwen")].relaxed_em_scores.append(rem_a)
            results[(hop_count, "auto_qwen")].f1_scores.append(f1_a)
            results[(hop_count, "auto_qwen")].hop_details.append(hops_a)
            results[(hop_count, "auto_qwen")].per_question.append({
                "q": question, "pred": pred_a, "ans": answer, "em": em_a,
                "sub_qs": auto_sub_qs, "n_auto_hops": len(auto_sub_qs),
                "hops": hops_a,
            })

            # Decomposition quality
            results[(hop_count, "auto_qwen")].decomp_quality.append({
                "n_gold": len(gold_decomp),
                "n_auto": len(auto_sub_qs),
                "hop_count_match": len(auto_sub_qs) == len(gold_decomp),
            })

            total_done += 1
            n = i + 1
            sp_em = results[(hop_count, "single_pass")].em_rate
            g_em = results[(hop_count, "gold_qwen")].em_rate
            a_em = results[(hop_count, "auto_qwen")].em_rate
            elapsed = time.time() - start
            eta = elapsed / total_done * (total_questions - total_done)

            sp_m = "+" if em_sp else "-"
            g_m = "+" if em_g else "-"
            a_m = "+" if em_a else "-"
            h_auto = len(auto_sub_qs)
            print(f"  [{n:3d}/{len(samples)}] [{sp_m}{g_m}{a_m}] {hop_count}h(→{h_auto}a) SP={sp_em:5.1f}% G={g_em:5.1f}% A={a_em:5.1f}% | {question[:40]}... ETA:{eta/60:.0f}m")

    elapsed = time.time() - start

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"MULTI-HOP SCALING RESULTS")
    print(f"{'='*70}")

    # Summary table
    print(f"\n  {'Config':15s} | {'2-hop':>10s} | {'3-hop':>10s} | {'4-hop':>10s}")
    print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for config in ["single_pass", "gold_qwen", "auto_qwen"]:
        row = f"  {config:15s} |"
        for h in [2, 3, 4]:
            r = results[(h, config)]
            n = len(r.em_scores)
            row += f" {r.em_rate:5.1f}% ({n:2d}) |"
        print(row)

    # Relaxed EM
    print(f"\n  Relaxed EM:")
    print(f"  {'Config':15s} | {'2-hop':>10s} | {'3-hop':>10s} | {'4-hop':>10s}")
    print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for config in ["single_pass", "gold_qwen", "auto_qwen"]:
        row = f"  {config:15s} |"
        for h in [2, 3, 4]:
            r = results[(h, config)]
            row += f" {r.relaxed_em_rate:5.1f}%      |"
        print(row)

    # Decomposition quality
    print(f"\n  Auto-decomposition quality:")
    for h in [2, 3, 4]:
        dq = results[(h, "auto_qwen")].decomp_quality
        if dq:
            hop_match = sum(1 for d in dq if d["hop_count_match"]) / len(dq) * 100
            avg_auto = sum(d["n_auto"] for d in dq) / len(dq)
            print(f"    {h}-hop: {hop_match:.0f}% correct hop count, avg auto hops: {avg_auto:.1f}")

    # Per-hop retrieval analysis
    print(f"\n  Per-hop retrieval (gold paragraph found):")
    for h in [2, 3, 4]:
        for config in ["gold_qwen", "auto_qwen"]:
            all_hops = results[(h, config)].hop_details
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

    # Decomp benefit
    print(f"\n  Decomposition benefit (auto EM - single_pass EM):")
    for h in [2, 3, 4]:
        sp = results[(h, "single_pass")].em_rate
        auto = results[(h, "auto_qwen")].em_rate
        delta = auto - sp
        sign = "+" if delta >= 0 else ""
        print(f"    {h}-hop: {sign}{delta:.1f}% ({sp:.1f}% → {auto:.1f}%)")

    print(f"\n  Total time: {elapsed/60:.1f} min")

    # ── Save ─────────────────────────────────────────────────────────
    output = {
        "metadata": {
            "experiment": "v6_multihop_scaling",
            "limit_per_hop": args.limit_per_hop,
            "elapsed_seconds": elapsed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": {},
        "decomp_quality": {},
    }

    for h in [2, 3, 4]:
        output["summary"][f"{h}_hop"] = {}
        for config in ["single_pass", "gold_qwen", "auto_qwen"]:
            r = results[(h, config)]
            output["summary"][f"{h}_hop"][config] = {
                "em": r.em_rate,
                "relaxed_em": r.relaxed_em_rate,
                "f1": r.mean_f1,
                "n": len(r.em_scores),
            }
        # Decomp quality for this hop count
        dq = results[(h, "auto_qwen")].decomp_quality
        if dq:
            output["decomp_quality"][f"{h}_hop"] = {
                "hop_count_match_rate": sum(1 for d in dq if d["hop_count_match"]) / len(dq) * 100,
                "avg_auto_hops": sum(d["n_auto"] for d in dq) / len(dq),
                "details": dq,
            }

    # Per-question results
    output["per_question"] = {}
    for h in [2, 3, 4]:
        output["per_question"][f"{h}_hop"] = {}
        for config in ["single_pass", "gold_qwen", "auto_qwen"]:
            output["per_question"][f"{h}_hop"][config] = results[(h, config)].per_question

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
