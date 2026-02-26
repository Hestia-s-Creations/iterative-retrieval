#!/usr/bin/env python3
"""
Validate top hypothesis battery results on full 30-question MuSiQue set.
Screens the top autonomous configs (no gold context/chain) found in the
100-hypothesis battery.
"""

import sys
import time
import json
import re
import string
import requests
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional

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

def ask_model(prompt, model="phi3:mini", temperature=0.1, max_tokens=32,
              repeat_penalty=1.1, top_p=0.9, retries=2):
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "repeat_penalty": repeat_penalty,
                        "top_p": top_p,
                    }
                },
                timeout=300
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            if attempt < retries:
                print(f"      [RETRY] {model} timeout, attempt {attempt+2}...")
                time.sleep(5)
            else:
                return "[ERROR: timeout]"


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


# ── Embedding Retrieval ──────────────────────────────────────────────

class EmbeddingRetriever:
    def __init__(self):
        self.model = None

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    def retrieve(self, query, paragraphs, top_k=3, query_prefix="Represent this sentence for searching relevant passages: "):
        self._load_model()
        query_text = f"{query_prefix}{query}"
        para_texts = [f"{p['title']} {p['paragraph_text']}" for p in paragraphs]
        query_emb = self.model.encode([query_text], normalize_embeddings=True)
        para_embs = self.model.encode(para_texts, normalize_embeddings=True)
        sims = np.dot(para_embs, query_emb.T).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(int(idx), float(sims[idx])) for idx in top_indices]


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
    "member of": "What is {subject} a member of?",
    "followed by": "What succeeded {subject}?",
    "occupant": "What team plays at {subject}?",
    "league": "What league does {subject} play in?",
    "movement": "What is the goal of {subject}?",
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


# ── Prompt Templates ────────────────────────────────────────────────

PROMPT_BASELINE = """Extract the answer to the question from the context. Give ONLY the answer, nothing else.

Examples:
Context: [Exeter College] Exeter College is one of the oldest colleges at the University of Oxford.
Question: Where was J.R.R. Tolkien educated?
Extracted answer: Exeter College

Context: [Nicole Kidman] Kidman founded Blossom Films in 2010.
Question: Who founded Blossom Films?
Extracted answer: Nicole Kidman

Context: [Southwest City] Southwest City is a city in McDonald County, Missouri, United States.
Question: What county is Southwest City in?
Extracted answer: McDonald County

Context: [Paramore] Paramore is an American rock band from Franklin, Tennessee.
Question: What genre is Paramore?
Extracted answer: rock

Context: {context}
Question: {question}
Extracted answer:"""

PROMPT_EIGHT_SHOT = """Extract the answer to the question from the context. Give ONLY the answer, nothing else.

Examples:
Context: [Exeter College] Exeter College is one of the oldest colleges at the University of Oxford.
Question: Where was J.R.R. Tolkien educated?
Extracted answer: Exeter College

Context: [Nicole Kidman] Kidman founded Blossom Films in 2010.
Question: Who founded Blossom Films?
Extracted answer: Nicole Kidman

Context: [Southwest City] Southwest City is a city in McDonald County, Missouri, United States.
Question: What county is Southwest City in?
Extracted answer: McDonald County

Context: [Paramore] Paramore is an American rock band from Franklin, Tennessee.
Question: What genre is Paramore?
Extracted answer: rock

Context: [The Beatles] The Beatles were an English rock band formed in Liverpool in 1960.
Question: Where were The Beatles formed?
Extracted answer: Liverpool

Context: [Marie Curie] Marie Curie won the Nobel Prize in Physics in 1903 and in Chemistry in 1911.
Question: What award did Marie Curie receive?
Extracted answer: Nobel Prize

Context: [Amazon] Amazon was founded by Jeff Bezos in 1994 in his garage in Bellevue, Washington.
Question: Who founded Amazon?
Extracted answer: Jeff Bezos

Context: [SpaceX] SpaceX designs, manufactures and launches advanced rockets and spacecraft for NASA.
Question: Who does SpaceX work for?
Extracted answer: NASA

Context: {context}
Question: {question}
Extracted answer:"""

PROMPT_QUOTE_BASED = """Find the answer to the question by quoting directly from the context. Copy the exact words.

Context: {context}
Question: {question}
Direct quote answer:"""

PROMPT_NEGATIVE = """Answer the question from the context. DO NOT include extra details, locations, descriptions, or parenthetical information. Just the core answer.

Examples:
- WRONG: "Exeter College, Oxford" → RIGHT: "Exeter College"
- WRONG: "Blossom Films owned by Nicole Kidman" → RIGHT: "Blossom Films"
- WRONG: "Southwest City, Missouri" → RIGHT: "Southwest City"

Context: {context}
Question: {question}
Answer:"""

PROMPT_INSTRUCTION = """You are a precise fact extraction system. Your job is to extract EXACTLY the answer from the given context. Rules:
1. Answer with ONLY the entity name, nothing else
2. Do NOT add location, description, or context
3. Keep it as SHORT as possible
4. If unsure, give your best guess from the context

Context: {context}
Question: {question}
ANSWER:"""


# ── Post-Processors ─────────────────────────────────────────────────

def postprocess_strip_location(answer):
    pattern = r',\s*(?:Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New\s+Hampshire|New\s+Jersey|New\s+Mexico|New\s+York|North\s+Carolina|North\s+Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode\s+Island|South\s+Carolina|South\s+Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West\s+Virginia|Wisconsin|Wyoming|England|Scotland|Wales|Ireland|France|Germany|Mexico|Canada|Australia|India|China|Japan|United\s+States|United\s+Kingdom|UK|US|USA)\s*$'
    m = re.search(pattern, answer, re.IGNORECASE)
    if m:
        return answer[:m.start()].strip()
    return answer

def postprocess_strip_parenthetical(answer):
    m = re.match(r'^(.+?)\s*\(.*\)\s*$', answer)
    if m and len(m.group(1).strip()) >= 3:
        return m.group(1).strip()
    return answer

def postprocess_first_entity(answer):
    for sep in [',', ';', ' and ', ' & ', ' featuring ', ' feat.']:
        if sep in answer:
            candidate = answer[:answer.index(sep)].strip()
            if len(candidate) >= 2:
                return candidate
    return answer

def postprocess_strip_qualifiers(answer):
    for pattern in [
        r'^(.+?)\s+(?:through|via|by|from|owned by|of)\s+',
        r'^(.+?)\s+(?:featuring|feat\.|ft\.)\s+',
    ]:
        m = re.match(pattern, answer, re.IGNORECASE)
        if m and len(m.group(1).strip()) >= 3:
            return m.group(1).strip()
    return answer

def postprocess_combined(answer):
    answer = postprocess_strip_qualifiers(answer)
    answer = postprocess_strip_location(answer)
    answer = postprocess_strip_parenthetical(answer)
    answer = postprocess_first_entity(answer)
    return answer


# ── Pipeline ─────────────────────────────────────────────────────────

def run_pipeline(sample, retriever, prompt_template=PROMPT_BASELINE,
                 model="phi3:mini", temperature=0.1, max_tokens=32,
                 top_k=3, repeat_penalty=1.1, top_p=0.9,
                 query_prefix="Represent this sentence for searching relevant passages: ",
                 use_gold_context=False, use_gold_chain=False,
                 answer_postprocess=None, context_format="bracket",
                 n_votes=1):
    decomposition = sample["question_decomposition"]
    if isinstance(decomposition, str):
        decomposition = json.loads(decomposition)
    paragraphs = sample["paragraphs"]
    previous_answers = []

    for i, hop in enumerate(decomposition):
        hop_q = format_hop_question(hop["question"], previous_answers)

        if use_gold_context:
            gold_idx = hop["paragraph_support_idx"]
            p = paragraphs[gold_idx]
            if context_format == "plain":
                context = p['paragraph_text']
            else:
                context = f"[{p['title']}] {p['paragraph_text']}"
        else:
            results = retriever.retrieve(hop_q, paragraphs, top_k=top_k,
                                         query_prefix=query_prefix)
            parts = []
            for j, (idx, score) in enumerate(results):
                p = paragraphs[idx]
                if context_format == "plain":
                    parts.append(p['paragraph_text'])
                else:
                    parts.append(f"[{p['title']}] {p['paragraph_text']}")
            context = "\n\n".join(parts)

        answers = []
        for v in range(n_votes):
            t = temperature if n_votes == 1 else max(0.3, temperature + v * 0.1)
            prompt = prompt_template.format(context=context, question=hop_q)
            response = ask_model(prompt, model=model, temperature=t,
                                max_tokens=max_tokens, repeat_penalty=repeat_penalty,
                                top_p=top_p)
            answer = extract_short_answer(response)
            if answer_postprocess:
                answer = answer_postprocess(answer)
            answers.append(answer)

        if n_votes > 1:
            from collections import Counter
            normalized_answers = [normalize_answer(a) for a in answers]
            counts = Counter(normalized_answers)
            best = counts.most_common(1)[0][0]
            for a, na in zip(answers, normalized_answers):
                if na == best:
                    answer = a
                    break
        else:
            answer = answers[0]

        if use_gold_chain:
            previous_answers.append(hop["answer"])
        else:
            previous_answers.append(answer)

    return answer


# ── Configs to Validate ──────────────────────────────────────────────

CONFIGS = {
    "baseline_v2": {
        "desc": "Phi-3 baseline 4-shot top-3 (v2 reference)",
        "kwargs": {"prompt_template": PROMPT_BASELINE, "model": "phi3:mini", "top_k": 3}
    },
    "H75_qwen_8shot": {
        "desc": "Qwen 7B + 8-shot + top-3 (BEST autonomous)",
        "kwargs": {"prompt_template": PROMPT_EIGHT_SHOT, "model": "qwen2.5:7b", "top_k": 3}
    },
    "H82_qwen_8shot_top2": {
        "desc": "Qwen 7B + 8-shot + top-2",
        "kwargs": {"prompt_template": PROMPT_EIGHT_SHOT, "model": "qwen2.5:7b", "top_k": 2}
    },
    "H92_qwen_neg_top2_greedy_pp": {
        "desc": "Qwen 7B + negative + top-2 + greedy + postprocess",
        "kwargs": {"prompt_template": PROMPT_NEGATIVE, "model": "qwen2.5:7b", "top_k": 2,
                   "temperature": 0.0, "answer_postprocess": postprocess_combined}
    },
    "H21_phi3_top1": {
        "desc": "Phi-3 + top-1 (less noise)",
        "kwargs": {"prompt_template": PROMPT_BASELINE, "model": "phi3:mini", "top_k": 1}
    },
    "H11_phi3_quote": {
        "desc": "Phi-3 + quote-based prompt",
        "kwargs": {"prompt_template": PROMPT_QUOTE_BASED, "model": "phi3:mini", "top_k": 3}
    },
    "H30_phi3_plain_context": {
        "desc": "Phi-3 + plain context (no titles)",
        "kwargs": {"prompt_template": PROMPT_BASELINE, "model": "phi3:mini", "top_k": 3,
                   "context_format": "plain"}
    },
    "H87_phi3_neg_top1_greedy_pp": {
        "desc": "Phi-3 + negative + top-1 + temp=0.01 + postprocess",
        "kwargs": {"prompt_template": PROMPT_NEGATIVE, "model": "phi3:mini", "top_k": 1,
                   "temperature": 0.01, "answer_postprocess": postprocess_combined}
    },
    # Gold context references for comparison
    "oracle_phi3_gold": {
        "desc": "Phi-3 + gold context (oracle reference)",
        "kwargs": {"prompt_template": PROMPT_BASELINE, "model": "phi3:mini", "use_gold_context": True}
    },
    "oracle_qwen_gold": {
        "desc": "Qwen 7B + gold context (oracle reference)",
        "kwargs": {"prompt_template": PROMPT_EIGHT_SHOT, "model": "qwen2.5:7b", "use_gold_context": True}
    },
}


def main():
    from datasets import load_dataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", default="evaluation/results/validation_top_configs.json")
    parser.add_argument("--configs", nargs="*", default=None, help="Specific configs to run")
    args = parser.parse_args()

    print("Loading MuSiQue...")
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    samples = [s for s in ds if s.get("answerable", True)][:args.limit]
    print(f"Validating on {len(samples)} questions")

    print("Loading embeddings...")
    retriever = EmbeddingRetriever()
    retriever._load_model()

    configs_to_run = args.configs or list(CONFIGS.keys())
    results = {}
    start_time = time.time()

    for config_name in configs_to_run:
        if config_name not in CONFIGS:
            print(f"Unknown config: {config_name}")
            continue

        cfg = CONFIGS[config_name]
        print(f"\n  {config_name}: {cfg['desc']}")
        print(f"    ", end="", flush=True)

        em_scores, rem_scores, f1_scores, latencies = [], [], [], []
        errors = 0
        per_question = []

        for i, sample in enumerate(samples):
            answer = sample["answer"]
            aliases = sample.get("answer_aliases", [])
            q = sample["question"]

            try:
                t0 = time.time()
                pred = run_pipeline(sample, retriever, **cfg["kwargs"])
                lat = (time.time() - t0) * 1000

                em = exact_match(pred, answer, aliases)
                rem = relaxed_match(pred, answer, aliases)
                f1 = best_f1(pred, answer, aliases)

                em_scores.append(em)
                rem_scores.append(rem)
                f1_scores.append(f1)
                latencies.append(lat)

                per_question.append({
                    "question": q,
                    "prediction": pred,
                    "answer": answer,
                    "em": em,
                    "relaxed_em": rem,
                    "f1": f1,
                })

                print("+" if em else ("~" if rem else "."), end="", flush=True)

            except Exception as e:
                errors += 1
                em_scores.append(False)
                rem_scores.append(False)
                f1_scores.append(0.0)
                latencies.append(0)
                per_question.append({"question": q, "error": str(e)})
                print("X", end="", flush=True)

        em_rate = sum(em_scores) / len(em_scores) * 100
        rem_rate = sum(rem_scores) / len(rem_scores) * 100
        mean_f1 = np.mean(f1_scores)
        mean_lat = np.mean(latencies)

        print(f"\n    EM={em_rate:5.1f}% relEM={rem_rate:5.1f}% F1={mean_f1:.3f} lat={mean_lat:.0f}ms")

        results[config_name] = {
            "description": cfg["desc"],
            "em": em_rate,
            "relaxed_em": rem_rate,
            "f1": float(mean_f1),
            "latency_ms": float(mean_lat),
            "errors": errors,
            "n_samples": len(samples),
            "per_question": per_question,
        }

        # Save incrementally
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)

    total = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"VALIDATION RESULTS ({len(samples)} questions, {total/60:.1f} min)")
    print(f"{'='*70}")
    print(f"{'Config':<35s} {'EM':>6s} {'relEM':>6s} {'F1':>6s} {'Lat':>7s}")
    print("-" * 70)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["em"]):
        print(f"{name:<35s} {r['em']:5.1f}% {r['relaxed_em']:5.1f}% {r['f1']:.3f} {r['latency_ms']:6.0f}ms")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
