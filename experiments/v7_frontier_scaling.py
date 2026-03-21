#!/usr/bin/env python3
"""
Iterative Retrieval v7 - Frontier Model Scaling Experiment

THE QUESTION: Does system-level decomposition help frontier models too?

If system architecture is "the other half of intelligence", the effect should
be orthogonal to model capability. We test:

  1. claude_single_pass:   Claude extracts from all paragraphs, no decomposition
  2. claude_system_auto:   Qwen decomposes → Claude extracts per hop → chain
  3. claude_system_gold:   Gold decomposition → Claude extracts per hop → chain
  4. qwen_system_auto:     Qwen decomposes → Qwen extracts (v5 reference: 60.0%)

If Claude single-pass < Claude system, then system architecture helps at ALL scales.
If Claude single-pass > Claude system, large models have internalized decomposition.
Either result is publishable.

Uses Claude Code OAuth for authentication. No API key needed.

Models tested:
  - Claude Sonnet 4.6 (claude-sonnet-4-6)
  - Claude Opus 4.6 (claude-opus-4-6) [optional, expensive]
  - Qwen 2.5 7B via Ollama (reference)
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


# ── Claude API Client ────────────────────────────────────────────────

CREDENTIALS_FILE = Path.home() / ".claude" / ".credentials.json"

def get_claude_oauth_token():
    """Read OAuth token from Claude Code credentials."""
    if not CREDENTIALS_FILE.exists():
        raise RuntimeError("No Claude Code credentials found. Run 'claude login' first.")

    with open(CREDENTIALS_FILE) as f:
        creds = json.load(f)

    oauth = creds.get("claudeAiOauth", {})
    token = oauth.get("accessToken")
    if not token:
        raise RuntimeError("No OAuth token in credentials file.")

    expires_at = oauth.get("expiresAt", 0)
    if expires_at and (expires_at / 1000) < time.time():
        raise RuntimeError("OAuth token expired. Re-authenticate with Claude Code.")

    return token


def _init_claude_client():
    """Initialize Anthropic client with OAuth."""
    import anthropic
    token = get_claude_oauth_token()
    return anthropic.Anthropic(
        auth_token=token,
        default_headers={"anthropic-beta": "oauth-2025-04-20"},
    )

_claude_client = None

def get_claude_client():
    global _claude_client
    if _claude_client is None:
        _claude_client = _init_claude_client()
    return _claude_client


def ask_claude(prompt, model="claude-sonnet-4-6", max_tokens=64, temperature=0.0, retries=2):
    """Call Claude API for extraction. Returns text response."""
    client = get_claude_client()

    for attempt in range(retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            error_str = str(e)
            if "rate_limit" in error_str.lower() or "429" in error_str:
                wait = 10 * (attempt + 1)
                print(f"    [RATE LIMIT] Waiting {wait}s (attempt {attempt+1}/{retries+1})...")
                time.sleep(wait)
            elif "overloaded" in error_str.lower() or "529" in error_str:
                wait = 15 * (attempt + 1)
                print(f"    [OVERLOADED] Waiting {wait}s (attempt {attempt+1}/{retries+1})...")
                time.sleep(wait)
            elif attempt < retries:
                print(f"    [RETRY] Claude error: {error_str[:80]}... (attempt {attempt+2}/{retries+1})")
                time.sleep(5)
            else:
                raise


# ── Ollama Client (for Qwen reference) ──────────────────────────────

OLLAMA_URL = "http://localhost:11434"

def ask_ollama(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32, retries=2):
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
                print(f"    [RETRY] Ollama timed out, attempt {attempt+2}/{retries+1}...")
                time.sleep(5)
            else:
                raise


# ── Unified Ask Function ─────────────────────────────────────────────

def ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=32):
    """Route to Claude or Ollama based on model name."""
    if model.startswith("claude-"):
        return ask_claude(prompt, model=model, max_tokens=max_tokens, temperature=float(temperature))
    else:
        return ask_ollama(prompt, model=model, temperature=temperature, max_tokens=max_tokens)


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

# 8-shot for per-hop extraction (same as v5 Qwen prompt)
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


# ── Qwen Decomposition (reused from v5) ─────────────────────────────

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
    """Use Qwen 2.5 7B to decompose a multi-hop question into sub-questions."""
    response = ask_ollama(
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
    """Run the iterative retrieval pipeline."""
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

        gold_idx = gold_decomp[i]["paragraph_support_idx"] if i < len(gold_decomp) else -1
        gold_retrieved = gold_idx in retrieved_indices

        # Extract answer
        prompt = prompt_template.format(context=context, question=hop_q)
        response = ask_model(prompt, model=model, max_tokens=32)
        answer = extract_short_answer(response)

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
class ConfigResult:
    name: str
    desc: str
    em_scores: list = field(default_factory=list)
    relaxed_em_scores: list = field(default_factory=list)
    f1_scores: list = field(default_factory=list)
    latencies: list = field(default_factory=list)
    hop_details: list = field(default_factory=list)
    per_question: list = field(default_factory=list)
    api_cost_estimate: float = 0.0

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


def run_experiment(limit=30, claude_model="claude-sonnet-4-6", verbose=True):
    from datasets import load_dataset

    print(f"Loading MuSiQue validation set...")
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    samples = [s for s in ds if s.get("answerable", True)][:limit]
    print(f"Testing on {len(samples)} answerable questions")
    print(f"Claude model: {claude_model}")
    print()

    # Verify Claude auth
    print("Verifying Claude authentication...")
    try:
        test = ask_claude("Say 'ok'", model=claude_model, max_tokens=4)
        print(f"  Claude responding: {test}")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    # Verify Ollama (for Qwen reference + decomposition)
    print("Verifying Ollama...")
    try:
        test = ask_ollama("Say 'ok'", model="qwen2.5:7b", max_tokens=4)
        print(f"  Qwen responding: {test}")
    except Exception as e:
        print(f"  WARNING: Ollama not available, skipping Qwen reference: {e}")

    retriever = EmbeddingRetriever()
    retriever._load_model()

    # ── Phase 1: Decompose all questions with Qwen ────────────────
    print(f"\n{'=' * 70}")
    print("PHASE 1: DECOMPOSITION (Qwen 7B for all configs)")
    print("=" * 70)

    all_auto_decomps = {}
    for i, sample in enumerate(samples):
        question = sample["question"]
        t0 = time.time()
        auto_sub_qs = decompose_with_qwen(question)
        dt = (time.time() - t0) * 1000
        all_auto_decomps[sample["id"]] = auto_sub_qs
        if verbose:
            print(f"  [{i+1:3d}] {question[:60]}...")
            print(f"        → {auto_sub_qs} ({dt:.0f}ms)")

    # ── Phase 2: Run All Configs ──────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"PHASE 2: EXTRACTION (4 configs × {len(samples)} questions)")
    print("=" * 70)

    configs = {
        "claude_single_pass": ConfigResult(
            "claude_single_pass",
            f"{claude_model} single-pass (no decomposition)"
        ),
        "claude_system_auto": ConfigResult(
            "claude_system_auto",
            f"Qwen decomp → {claude_model} extract per hop"
        ),
        "claude_system_gold": ConfigResult(
            "claude_system_gold",
            f"Gold decomp → {claude_model} extract per hop"
        ),
        "qwen_system_auto": ConfigResult(
            "qwen_system_auto",
            "Qwen decomp → Qwen extract (v5 reference: 60.0%)"
        ),
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

        gold_hops = [{"question": hop["question"]} for hop in gold_decomp]
        auto_sub_qs = all_auto_decomps[sample["id"]]
        auto_hops = [{"question": sq} for sq in auto_sub_qs]

        # ── Config 1: Claude single-pass ──────────────────────────
        t0 = time.time()
        all_ctx = "\n\n".join(
            f"[{p['title']}] {p['paragraph_text']}" for p in sample["paragraphs"][:10]
        )
        prompt = PROMPT_SINGLE_PASS.format(context=all_ctx, question=question)
        response = ask_model(prompt, model=claude_model, max_tokens=32)
        pred = extract_short_answer(response)
        lat = (time.time() - t0) * 1000

        em = exact_match(pred, answer, aliases)
        rem = relaxed_match(pred, answer, aliases)
        f1 = best_f1(pred, answer, aliases)
        configs["claude_single_pass"].em_scores.append(em)
        configs["claude_single_pass"].relaxed_em_scores.append(rem)
        configs["claude_single_pass"].f1_scores.append(f1)
        configs["claude_single_pass"].latencies.append(lat)
        configs["claude_single_pass"].per_question.append({
            "id": sample["id"], "question": question,
            "prediction": pred, "answer": answer,
            "em": em, "relaxed_em": rem, "f1": f1,
        })
        s = "+" if em else ("~" if rem else "-")
        print(f"  [{s}] claude_single_pass:    {pred[:45]}")

        # ── Config 2: Claude system (auto decomp) ────────────────
        t0 = time.time()
        pred, hops = run_pipeline(
            sample, retriever,
            decomposition=auto_hops,
            prompt_template=PROMPT_8SHOT,
            model=claude_model,
            top_k=3, use_gold_context=False,
            verbose=verbose,
        )
        lat = (time.time() - t0) * 1000

        em = exact_match(pred, answer, aliases)
        rem = relaxed_match(pred, answer, aliases)
        f1 = best_f1(pred, answer, aliases)
        configs["claude_system_auto"].em_scores.append(em)
        configs["claude_system_auto"].relaxed_em_scores.append(rem)
        configs["claude_system_auto"].f1_scores.append(f1)
        configs["claude_system_auto"].latencies.append(lat)
        configs["claude_system_auto"].hop_details.append(hops)
        configs["claude_system_auto"].per_question.append({
            "id": sample["id"], "question": question,
            "prediction": pred, "answer": answer,
            "em": em, "relaxed_em": rem, "f1": f1,
            "hops": hops,
        })
        s = "+" if em else ("~" if rem else "-")
        print(f"  [{s}] claude_system_auto:    {pred[:45]}")

        # ── Config 3: Claude system (gold decomp) ────────────────
        t0 = time.time()
        pred, hops = run_pipeline(
            sample, retriever,
            decomposition=gold_hops,
            prompt_template=PROMPT_8SHOT,
            model=claude_model,
            top_k=3, use_gold_context=False,
            verbose=False,
        )
        lat = (time.time() - t0) * 1000

        em = exact_match(pred, answer, aliases)
        rem = relaxed_match(pred, answer, aliases)
        f1 = best_f1(pred, answer, aliases)
        configs["claude_system_gold"].em_scores.append(em)
        configs["claude_system_gold"].relaxed_em_scores.append(rem)
        configs["claude_system_gold"].f1_scores.append(f1)
        configs["claude_system_gold"].latencies.append(lat)
        configs["claude_system_gold"].hop_details.append(hops)
        configs["claude_system_gold"].per_question.append({
            "id": sample["id"], "question": question,
            "prediction": pred, "answer": answer,
            "em": em, "relaxed_em": rem, "f1": f1,
            "hops": hops,
        })
        s = "+" if em else ("~" if rem else "-")
        print(f"  [{s}] claude_system_gold:    {pred[:45]}")

        # ── Config 4: Qwen system (auto decomp, v5 reference) ────
        t0 = time.time()
        pred, hops = run_pipeline(
            sample, retriever,
            decomposition=auto_hops,
            prompt_template=PROMPT_8SHOT,
            model="qwen2.5:7b",
            top_k=3, use_gold_context=False,
            verbose=False,
        )
        lat = (time.time() - t0) * 1000

        em = exact_match(pred, answer, aliases)
        rem = relaxed_match(pred, answer, aliases)
        f1 = best_f1(pred, answer, aliases)
        configs["qwen_system_auto"].em_scores.append(em)
        configs["qwen_system_auto"].relaxed_em_scores.append(rem)
        configs["qwen_system_auto"].f1_scores.append(f1)
        configs["qwen_system_auto"].latencies.append(lat)
        configs["qwen_system_auto"].hop_details.append(hops)
        configs["qwen_system_auto"].per_question.append({
            "id": sample["id"], "question": question,
            "prediction": pred, "answer": answer,
            "em": em, "relaxed_em": rem, "f1": f1,
            "hops": hops,
        })
        s = "+" if em else ("~" if rem else "-")
        print(f"  [{s}] qwen_system_auto:      {pred[:45]}")

        # Running totals every 5 questions
        if (i + 1) % 5 == 0 or i == len(samples) - 1:
            print(f"\n  ── Running totals ({i+1}/{len(samples)}) ──")
            for name, cfg in configs.items():
                if cfg.em_scores:
                    print(f"    {name:24s}: EM={cfg.em_rate:5.1f}%  relEM={cfg.relaxed_em_rate:5.1f}%  F1={cfg.mean_f1:.3f}  lat={cfg.mean_latency:.0f}ms")

    return configs, samples


def print_final_results(configs, samples, claude_model):
    print(f"\n{'=' * 70}")
    print("FINAL RESULTS: FRONTIER MODEL SCALING EXPERIMENT")
    print(f"{'=' * 70}")
    print(f"Dataset: MuSiQue | N={len(samples)} | 2-hop answerable")
    print(f"Claude model: {claude_model}")
    print(f"Qwen model: qwen2.5:7b (local)")

    # Main results table
    print(f"\n{'Config':<26s} {'Extractor':<20s} {'EM':>6s} {'relEM':>6s} {'F1':>6s} {'Lat':>7s}")
    print("─" * 75)
    for name in ["claude_single_pass", "claude_system_auto", "claude_system_gold", "qwen_system_auto"]:
        cfg = configs[name]
        extractor = claude_model if "claude" in name else "qwen2.5:7b"
        print(f"{name:<26s} {extractor:<20s} {cfg.em_rate:5.1f}% {cfg.relaxed_em_rate:5.1f}% {cfg.mean_f1:.3f} {cfg.mean_latency:5.0f}ms")

    # Key comparisons
    sp = configs["claude_single_pass"].em_rate
    sys_auto = configs["claude_system_auto"].em_rate
    sys_gold = configs["claude_system_gold"].em_rate
    qwen_ref = configs["qwen_system_auto"].em_rate

    print(f"\n  ── Key Findings ──")
    print(f"  System benefit for Claude:     {sys_auto - sp:+.1f}% EM (single-pass {sp:.1f}% → system {sys_auto:.1f}%)")
    print(f"  Auto-decomp gap (Claude):      {sys_gold - sys_auto:+.1f}% EM (gold {sys_gold:.1f}% vs auto {sys_auto:.1f}%)")
    print(f"  Claude vs Qwen (system):       {sys_auto - qwen_ref:+.1f}% EM ({sys_auto:.1f}% vs {qwen_ref:.1f}%)")
    print(f"  v5 reference (Qwen auto):      {qwen_ref:.1f}% EM (expected ~60%)")

    # The key question
    print(f"\n  ── THE KEY QUESTION ──")
    if sys_auto > sp:
        gain = sys_auto - sp
        print(f"  SYSTEM HELPS CLAUDE: +{gain:.1f}% EM")
        print(f"  → System architecture is orthogonal to model capability")
        print(f"  → Even frontier models benefit from externalized cognition")
    elif sp > sys_auto:
        gap = sp - sys_auto
        print(f"  CLAUDE DOESN'T NEED SYSTEM: +{gap:.1f}% EM for single-pass")
        print(f"  → Large models have partially internalized decomposition")
        print(f"  → System architecture is a SUBSTITUTE for model capability")
    else:
        print(f"  TIED: {sp:.1f}% EM both ways")
        print(f"  → Need more data to distinguish")

    # Per-hop analysis for Claude system configs
    for cfg_name in ["claude_system_auto", "claude_system_gold"]:
        if configs[cfg_name].hop_details:
            hop1_em = sum(1 for hops in configs[cfg_name].hop_details
                          if len(hops) > 0 and hops[0].get("em", False))
            hop2_em = sum(1 for hops in configs[cfg_name].hop_details
                          if len(hops) > 1 and hops[1].get("em", False))
            n = len(configs[cfg_name].hop_details)
            print(f"\n  {cfg_name}:")
            print(f"    Hop 1 EM: {hop1_em}/{n} ({hop1_em/n*100:.1f}%)")
            print(f"    Hop 2 EM: {hop2_em}/{n} ({hop2_em/n*100:.1f}%)")

    return {
        "metadata": {
            "claude_model": claude_model,
            "n_samples": len(samples),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": {name: {
            "em": cfg.em_rate,
            "relaxed_em": cfg.relaxed_em_rate,
            "f1": cfg.mean_f1,
            "latency_ms": cfg.mean_latency,
            "per_question": cfg.per_question,
        } for name, cfg in configs.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="v7: Frontier Model Scaling Experiment")
    parser.add_argument("--limit", type=int, default=30,
                        help="Number of questions to test (default: 30)")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6",
                        choices=["claude-sonnet-4-6", "claude-opus-4-6"],
                        help="Claude model to test")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (default: results/v7_frontier_{model}.json)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.output is None:
        model_short = args.model.replace("claude-", "").replace("-", "_")
        args.output = f"results/v7_frontier_{model_short}.json"

    start = time.time()
    configs, samples = run_experiment(
        limit=args.limit,
        claude_model=args.model,
        verbose=not args.quiet,
    )
    elapsed = time.time() - start

    results = print_final_results(configs, samples, args.model)
    results["metadata"]["elapsed_seconds"] = elapsed

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to: {output_path}")
    print(f"Total time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
