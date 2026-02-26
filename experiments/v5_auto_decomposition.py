#!/usr/bin/env python3
"""
Iterative Retrieval v5 - Full Auto-Decomposition Experiment

The question: Can Qwen 2.5 7B close the auto-decomposition gap?

Current state:
  - Gold decomp + Qwen extract = 56.7% EM (validated, best autonomous)
  - Gold decomp + Phi-3 extract = 30.0% EM (v2 baseline)
  - Auto decomp + Phi-3 extract = untested (v4, never run)
  - Auto decomp + Qwen extract = untested (THIS IS THE KEY TEST)

If Qwen can decompose AND extract well, we have a fully autonomous pipeline
with no gold labels at any step. That's the difference between "interesting
research" and "autonomous agent that actually works."

Tests 6 configs:
  1. gold_phi3:      Gold decomp + Phi-3 4-shot extract (v2 reference)
  2. gold_qwen:      Gold decomp + Qwen 8-shot extract (56.7% reference)
  3. auto_phi3:      Qwen decomp + Phi-3 4-shot extract
  4. auto_qwen:      Qwen decomp + Qwen 8-shot extract  ← THE KEY TEST
  5. auto_qwen_gold_ctx: Qwen decomp + Qwen extract + gold context (ceiling)
  6. single_pass:    No decomposition baseline
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


# ── Prompt Templates ─────────────────────────────────────────────────

PROMPT_4SHOT = """Answer the question using the context below. Give ONLY the specific name, place, or fact asked for. Be as concise as possible - just the core answer, no extra details.

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


# ── Qwen Decomposition ──────────────────────────────────────────────

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
    """Use Qwen2.5 7B to decompose a multi-hop question into sub-questions."""
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


# ── Pipeline Runner ──────────────────────────────────────────────────

def run_pipeline(sample, retriever, decomposition, prompt_template, model,
                 top_k=3, use_gold_context=False, verbose=False):
    """
    Run the iterative retrieval pipeline with a given decomposition.

    Args:
        sample: MuSiQue sample
        retriever: EmbeddingRetriever instance
        decomposition: List of {"question": str} dicts (gold or auto-generated)
        prompt_template: Extraction prompt
        model: Ollama model name for extraction
        top_k: Number of paragraphs to retrieve
        use_gold_context: If True, use gold paragraph instead of retrieval
        verbose: Print per-hop details
    """
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    for i, hop in enumerate(decomposition):
        # Get this hop's question, substitute previous answers
        hop_q = hop["question"]
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
        prompt = prompt_template.format(context=context, question=hop_q)
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


# ── Decomposition Quality Analysis ──────────────────────────────────

def analyze_decomposition(auto_sub_qs, gold_decomp, previous_answers=None):
    """Compare auto decomposition against gold. Returns quality metrics."""
    gold_qs = []
    prev = previous_answers or []
    for hop in gold_decomp:
        q = format_hop_question(hop["question"], prev)
        gold_qs.append(q)
        prev.append(hop["answer"])

    metrics = {
        "n_auto": len(auto_sub_qs),
        "n_gold": len(gold_qs),
        "hop_count_match": len(auto_sub_qs) == len(gold_qs),
        "per_hop": [],
    }

    for i in range(min(len(auto_sub_qs), len(gold_qs))):
        auto_norm = normalize_answer(auto_sub_qs[i])
        gold_norm = normalize_answer(gold_qs[i])

        # Token overlap
        auto_tokens = set(auto_norm.split())
        gold_tokens = set(gold_norm.split())
        overlap = auto_tokens & gold_tokens
        precision = len(overlap) / len(auto_tokens) if auto_tokens else 0
        recall = len(overlap) / len(gold_tokens) if gold_tokens else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics["per_hop"].append({
            "auto": auto_sub_qs[i],
            "gold": gold_qs[i],
            "token_f1": f1,
            "exact_match": auto_norm == gold_norm,
        })

    return metrics


# ── Main Experiment ──────────────────────────────────────────────────

@dataclass
class ConfigResult:
    name: str
    desc: str
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
    print(f"Testing on {len(samples)} answerable questions\n")

    retriever = EmbeddingRetriever()
    retriever._load_model()

    # ── Phase 1: Decomposition Quality ────────────────────────────
    print("=" * 70)
    print("PHASE 1: DECOMPOSITION QUALITY (Qwen 7B vs Gold)")
    print("=" * 70)

    decomp_results = []
    all_auto_decomps = {}  # Cache for Phase 2

    for i, sample in enumerate(samples):
        question = sample["question"]
        gold_decomp = sample["question_decomposition"]

        # Get Qwen's decomposition
        t0 = time.time()
        auto_sub_qs = decompose_with_qwen(question)
        decomp_time = (time.time() - t0) * 1000

        # Cache for reuse
        all_auto_decomps[sample["id"]] = auto_sub_qs

        # Analyze quality
        quality = analyze_decomposition(auto_sub_qs, gold_decomp)
        quality["question"] = question
        quality["decomp_time_ms"] = decomp_time
        quality["auto_questions"] = auto_sub_qs
        decomp_results.append(quality)

        # Print
        gold_q1 = format_hop_question(gold_decomp[0]["question"], [])
        gold_q2 = format_hop_question(gold_decomp[1]["question"], [gold_decomp[0]["answer"]]) if len(gold_decomp) > 1 else "N/A"

        if verbose:
            hop_f1s = [h["token_f1"] for h in quality["per_hop"]]
            avg_f1 = sum(hop_f1s) / len(hop_f1s) if hop_f1s else 0
            marker = "+" if avg_f1 > 0.5 else ("~" if avg_f1 > 0.2 else "-")
            print(f"  [{marker}] Q{i+1}: {question[:60]}...")
            print(f"       Auto: {auto_sub_qs}")
            print(f"       Gold: [{gold_q1}, {gold_q2}]")
            if quality["per_hop"]:
                f1_str = ', '.join(f"{h['token_f1']:.2f}" for h in quality["per_hop"])
                print(f"       F1: {f1_str}")

    # Decomposition summary
    hop_count_matches = sum(1 for d in decomp_results if d["hop_count_match"])
    all_hop_f1s = [h["token_f1"] for d in decomp_results for h in d["per_hop"]]
    hop1_f1s = [d["per_hop"][0]["token_f1"] for d in decomp_results if d["per_hop"]]
    hop2_f1s = [d["per_hop"][1]["token_f1"] for d in decomp_results if len(d["per_hop"]) > 1]
    hop1_exact = sum(1 for d in decomp_results if d["per_hop"] and d["per_hop"][0]["exact_match"])
    hop2_exact = sum(1 for d in decomp_results if len(d["per_hop"]) > 1 and d["per_hop"][1]["exact_match"])
    avg_decomp_time = sum(d["decomp_time_ms"] for d in decomp_results) / len(decomp_results)

    print(f"\n  --- Decomposition Quality Summary ---")
    print(f"  Hop count matches: {hop_count_matches}/{len(samples)} ({hop_count_matches/len(samples)*100:.1f}%)")
    print(f"  Hop 1 token F1:    {sum(hop1_f1s)/len(hop1_f1s):.3f} (exact match: {hop1_exact}/{len(samples)})")
    print(f"  Hop 2 token F1:    {sum(hop2_f1s)/len(hop2_f1s):.3f} (exact match: {hop2_exact}/{len(samples)})")
    print(f"  Mean decomp time:  {avg_decomp_time:.0f}ms")

    # ── Phase 2: Full Pipeline Comparison ─────────────────────────
    print(f"\n{'=' * 70}")
    print("PHASE 2: FULL PIPELINE (6 configs × 30 questions)")
    print("=" * 70)

    configs = {
        "gold_phi3": ConfigResult("gold_phi3", "Gold decomp + Phi-3 4-shot (v2 ref)"),
        "gold_qwen": ConfigResult("gold_qwen", "Gold decomp + Qwen 8-shot (56.7% ref)"),
        "auto_phi3": ConfigResult("auto_phi3", "Qwen decomp + Phi-3 4-shot"),
        "auto_qwen": ConfigResult("auto_qwen", "Qwen decomp + Qwen 8-shot ← KEY TEST"),
        "auto_qwen_gold_ctx": ConfigResult("auto_qwen_gold_ctx", "Qwen decomp + Qwen + gold ctx (ceiling)"),
        "single_pass": ConfigResult("single_pass", "No decomposition baseline"),
    }

    config_specs = {
        "gold_phi3": {
            "decomp": "gold", "prompt": PROMPT_4SHOT,
            "model": "phi3:mini", "top_k": 3, "gold_ctx": False,
        },
        "gold_qwen": {
            "decomp": "gold", "prompt": PROMPT_8SHOT,
            "model": "qwen2.5:7b", "top_k": 3, "gold_ctx": False,
        },
        "auto_phi3": {
            "decomp": "auto", "prompt": PROMPT_4SHOT,
            "model": "phi3:mini", "top_k": 3, "gold_ctx": False,
        },
        "auto_qwen": {
            "decomp": "auto", "prompt": PROMPT_8SHOT,
            "model": "qwen2.5:7b", "top_k": 3, "gold_ctx": False,
        },
        "auto_qwen_gold_ctx": {
            "decomp": "auto", "prompt": PROMPT_8SHOT,
            "model": "qwen2.5:7b", "top_k": 3, "gold_ctx": True,
        },
    }

    for i, sample in enumerate(samples):
        question = sample["question"]
        answer = sample["answer"]
        aliases = sample.get("answer_aliases", [])
        gold_decomp = sample["question_decomposition"]
        n_hops = len(gold_decomp)

        print(f"\n{'─' * 70}")
        print(f"[{i+1}/{len(samples)}] ({n_hops}-hop) {question}")
        print(f"Expected: {answer}")

        # Prepare decompositions
        gold_hops = [{"question": hop["question"]} for hop in gold_decomp]
        auto_sub_qs = all_auto_decomps[sample["id"]]
        auto_hops = [{"question": sq} for sq in auto_sub_qs]

        # Run each config
        for cfg_name, spec in config_specs.items():
            decomp = gold_hops if spec["decomp"] == "gold" else auto_hops

            t0 = time.time()
            pred, hops = run_pipeline(
                sample, retriever,
                decomposition=decomp,
                prompt_template=spec["prompt"],
                model=spec["model"],
                top_k=spec["top_k"],
                use_gold_context=spec["gold_ctx"],
                verbose=(verbose and cfg_name in ("auto_qwen", "gold_qwen")),
            )
            lat = (time.time() - t0) * 1000

            em = exact_match(pred, answer, aliases)
            rem = relaxed_match(pred, answer, aliases)
            f1 = best_f1(pred, answer, aliases)

            configs[cfg_name].em_scores.append(em)
            configs[cfg_name].relaxed_em_scores.append(rem)
            configs[cfg_name].f1_scores.append(f1)
            configs[cfg_name].latencies.append(lat)
            configs[cfg_name].hop_details.append(hops)
            configs[cfg_name].per_question.append({
                "id": sample["id"], "question": question,
                "prediction": pred, "answer": answer,
                "em": em, "relaxed_em": rem, "f1": f1,
                "hops": hops,
            })

            s = "+" if em else ("~" if rem else "-")
            print(f"  [{s}] {cfg_name:22s}: {pred[:40]}")

        # Single pass: all paragraphs, no decomposition
        t0 = time.time()
        all_ctx = "\n\n".join(
            f"[{p['title']}] {p['paragraph_text']}" for p in sample["paragraphs"][:10]
        )
        prompt = PROMPT_SINGLE_PASS.format(context=all_ctx, question=question)
        response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32)
        pred = extract_short_answer(response)
        lat = (time.time() - t0) * 1000

        em = exact_match(pred, answer, aliases)
        rem = relaxed_match(pred, answer, aliases)
        f1 = best_f1(pred, answer, aliases)
        configs["single_pass"].em_scores.append(em)
        configs["single_pass"].relaxed_em_scores.append(rem)
        configs["single_pass"].f1_scores.append(f1)
        configs["single_pass"].latencies.append(lat)
        configs["single_pass"].per_question.append({
            "id": sample["id"], "question": question,
            "prediction": pred, "answer": answer,
            "em": em, "relaxed_em": rem, "f1": f1,
        })
        s = "+" if em else ("~" if rem else "-")
        print(f"  [{s}] {'single_pass':22s}: {pred[:40]}")

        # Running totals
        if (i + 1) % 5 == 0 or i == len(samples) - 1:
            print(f"\n  ── Running totals ({i+1}/{len(samples)}) ──")
            for name, cfg in configs.items():
                if cfg.em_scores:
                    print(f"    {name:22s}: EM={cfg.em_rate:5.1f}%  relEM={cfg.relaxed_em_rate:5.1f}%  F1={cfg.mean_f1:.3f}")

    return configs, samples, decomp_results


def print_final_results(configs, samples, decomp_results):
    print(f"\n{'=' * 70}")
    print("FINAL RESULTS: AUTO-DECOMPOSITION EXPERIMENT")
    print(f"{'=' * 70}")
    print(f"Dataset: MuSiQue | N={len(samples)} | All 2-hop answerable questions")

    # Main results table
    print(f"\n{'Config':<24s} {'Decomp':<6s} {'Extract':<10s} {'EM':>6s} {'relEM':>6s} {'F1':>6s} {'Lat':>7s}")
    print("─" * 70)
    order = ["single_pass", "gold_phi3", "gold_qwen", "auto_phi3", "auto_qwen", "auto_qwen_gold_ctx"]
    for name in order:
        cfg = configs[name]
        decomp = "N/A" if "single" in name else ("gold" if "gold" in name else "auto")
        model = "qwen7b" if "qwen" in name or "single" in name else "phi3"
        print(f"{name:<24s} {decomp:<6s} {model:<10s} {cfg.em_rate:5.1f}% {cfg.relaxed_em_rate:5.1f}% {cfg.mean_f1:.3f} {cfg.mean_latency:5.0f}ms")

    # Key comparisons
    print(f"\n  ── Key Comparisons ──")
    gold_qwen_em = configs["gold_qwen"].em_rate
    auto_qwen_em = configs["auto_qwen"].em_rate
    gap = gold_qwen_em - auto_qwen_em
    print(f"  Auto-decomposition gap (Qwen):  {gap:+.1f}% EM ({gold_qwen_em:.1f}% gold → {auto_qwen_em:.1f}% auto)")

    gold_phi3_em = configs["gold_phi3"].em_rate
    auto_phi3_em = configs["auto_phi3"].em_rate
    gap2 = gold_phi3_em - auto_phi3_em
    print(f"  Auto-decomposition gap (Phi-3): {gap2:+.1f}% EM ({gold_phi3_em:.1f}% gold → {auto_phi3_em:.1f}% auto)")

    sp_em = configs["single_pass"].em_rate
    print(f"  Single-pass → Auto+Qwen:        {auto_qwen_em - sp_em:+.1f}% EM ({sp_em:.1f}% → {auto_qwen_em:.1f}%)")
    print(f"  Extractor upgrade (auto decomp): {auto_qwen_em - auto_phi3_em:+.1f}% EM (Phi-3 → Qwen)")

    # Per-hop analysis for auto_qwen
    print(f"\n  ── Per-Hop Analysis (auto_qwen) ──")
    for cfg_name in ["gold_qwen", "auto_qwen"]:
        hop1_em = sum(1 for hops in configs[cfg_name].hop_details
                      if len(hops) > 0 and hops[0].get("em", False))
        hop2_em = sum(1 for hops in configs[cfg_name].hop_details
                      if len(hops) > 1 and hops[1].get("em", False))
        hop1_retr = sum(1 for hops in configs[cfg_name].hop_details
                        if len(hops) > 0 and hops[0].get("gold_retrieved", False))
        hop2_retr = sum(1 for hops in configs[cfg_name].hop_details
                        if len(hops) > 1 and hops[1].get("gold_retrieved", False))
        n = len(configs[cfg_name].hop_details)
        print(f"  {cfg_name}:")
        print(f"    Hop 1: EM={hop1_em}/{n} ({hop1_em/n*100:.1f}%), Gold retrieved={hop1_retr}/{n} ({hop1_retr/n*100:.1f}%)")
        print(f"    Hop 2: EM={hop2_em}/{n} ({hop2_em/n*100:.1f}%), Gold retrieved={hop2_retr}/{n} ({hop2_retr/n*100:.1f}%)")

    # Error cascade analysis
    print(f"\n  ── Error Cascade (auto_qwen: where does it fail?) ──")
    n_total = len(configs["auto_qwen"].per_question)
    n_correct = sum(1 for q in configs["auto_qwen"].per_question if q["em"])

    # Decomposition failure → wrong retrieval → wrong extraction
    decomp_ok_retr_ok = 0
    decomp_ok_retr_fail = 0
    decomp_fail = 0
    for j, q in enumerate(configs["auto_qwen"].per_question):
        hops = q.get("hops", [])
        if not hops:
            decomp_fail += 1
            continue
        # Check if decomposition led to correct retrieval
        all_retrieved = all(h.get("gold_retrieved", False) for h in hops)
        # Check decomposition quality
        dq = decomp_results[j] if j < len(decomp_results) else None
        if dq and dq["per_hop"]:
            avg_f1 = sum(h["token_f1"] for h in dq["per_hop"]) / len(dq["per_hop"])
            if avg_f1 < 0.3:
                decomp_fail += 1
            elif all_retrieved:
                decomp_ok_retr_ok += 1
            else:
                decomp_ok_retr_fail += 1
        elif all_retrieved:
            decomp_ok_retr_ok += 1
        else:
            decomp_ok_retr_fail += 1

    print(f"  Correct:                    {n_correct}/{n_total} ({n_correct/n_total*100:.1f}%)")
    print(f"  Decomp quality OK + retr OK:{decomp_ok_retr_ok}/{n_total}")
    print(f"  Decomp quality OK + retr fail:{decomp_ok_retr_fail}/{n_total}")
    print(f"  Decomp quality poor:        {decomp_fail}/{n_total}")

    return {
        "decomposition_quality": {
            "hop_count_match_rate": sum(1 for d in decomp_results if d["hop_count_match"]) / len(decomp_results),
            "hop1_token_f1": sum(d["per_hop"][0]["token_f1"] for d in decomp_results if d["per_hop"]) / len(decomp_results),
            "hop2_token_f1": sum(d["per_hop"][1]["token_f1"] for d in decomp_results if len(d["per_hop"]) > 1) / len(decomp_results),
            "mean_decomp_time_ms": sum(d["decomp_time_ms"] for d in decomp_results) / len(decomp_results),
            "per_question": decomp_results,
        },
        "pipeline_results": {name: {
            "em": cfg.em_rate,
            "relaxed_em": cfg.relaxed_em_rate,
            "f1": cfg.mean_f1,
            "latency_ms": cfg.mean_latency,
            "per_question": cfg.per_question,
        } for name, cfg in configs.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="v5: Full Auto-Decomposition Experiment")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", type=str, default="results/v5_auto_decomposition.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Check Ollama
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"Ollama models: {models}")
        assert any("phi3" in m for m in models), "phi3:mini not found"
        assert any("qwen2.5:7b" in m for m in models), "qwen2.5:7b not found"
    except AssertionError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR connecting to Ollama: {e}")
        sys.exit(1)

    start = time.time()
    configs, samples, decomp_results = run_experiment(
        limit=args.limit, verbose=not args.quiet
    )
    elapsed = time.time() - start

    results = print_final_results(configs, samples, decomp_results)
    results["metadata"] = {
        "n_samples": len(samples),
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to: {output_path}")
    print(f"Total time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
