#!/usr/bin/env python3
"""
Hypothesis Battery: 100 hypotheses tested against the iterative retrieval system.

Strategy:
- Screen each hypothesis on 15 MuSiQue questions (fast)
- Compare against baseline (v2 EMBED_RETRIEVAL approach)
- Report effect size and statistical significance
- Flag promising hypotheses for full validation

Categories:
  1-20:  Prompt engineering (extraction prompt variations)
  21-40: Retrieval parameters (top_k, query formatting, reranking)
  41-55: Model parameters (temperature, max_tokens, repeat_penalty)
  56-70: Context formatting (paragraph presentation, ordering)
  71-85: Answer processing (extraction, normalization, voting)
  86-100: Architecture (hop strategies, multi-model, ensembles)
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
from typing import List, Dict, Any, Callable, Optional, Tuple
import traceback


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


# ── Hypothesis Definition ────────────────────────────────────────────

@dataclass
class Hypothesis:
    id: int
    name: str
    category: str
    description: str
    run_fn: Callable  # (sample, retriever, config) -> predicted_answer


@dataclass
class HypothesisResult:
    id: int
    name: str
    category: str
    em_scores: list = field(default_factory=list)
    relaxed_em_scores: list = field(default_factory=list)
    f1_scores: list = field(default_factory=list)
    latencies: list = field(default_factory=list)
    errors: int = 0

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


# ── Prompt Templates ─────────────────────────────────────────────────

# Baseline (v2)
PROMPT_BASELINE = """Answer the question using the context below. Give ONLY the specific name, place, or fact asked for. Be as concise as possible - just the core answer, no extra details.

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

# Prompt variant: Zero-shot (no examples)
PROMPT_ZERO_SHOT = """Answer the question using ONLY the context below. Give the shortest possible answer - just the name, place, or fact.

Context: {context}
Question: {question}
Answer:"""

# Prompt variant: 2-shot
PROMPT_TWO_SHOT = """Answer the question using the context below. Give ONLY the specific answer.

Context: [Steve Hillage] Green is the fourth studio album by British progressive rock musician Steve Hillage.
Question: Who performed Green?
Answer: Steve Hillage

Context: [Orion Pictures] The film was distributed by Orion Pictures, founded by Mike Medavoy and four other executives.
Question: Who founded Orion Pictures?
Answer: Mike Medavoy

Context: {context}
Question: {question}
Answer:"""

# Prompt variant: 8-shot (more examples)
PROMPT_EIGHT_SHOT = """Answer the question using the context below. Give ONLY the specific name, place, or fact. One or two words maximum.

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

# Prompt: Instruction emphasis
PROMPT_INSTRUCTION = """INSTRUCTION: Read the context carefully and extract the EXACT answer to the question. Your answer must be a single entity name - no explanations, no sentences, no qualifiers.

Context: {context}

Question: {question}

ANSWER (entity name only):"""

# Prompt: Chain of thought then answer
PROMPT_COT_THEN_ANSWER = """Read the context and answer the question.

Context: {context}

Question: {question}

Think step by step, then give your final answer after "ANSWER:".

Reasoning:"""

# Prompt: Role-based
PROMPT_ROLE = """You are a precise fact extraction system. Given a context passage and a question, extract the exact answer from the context. Return ONLY the answer - a name, place, date, or fact. No extra words.

Context: {context}
Question: {question}
Extracted answer:"""

# Prompt: Structured output
PROMPT_JSON = 'Extract the answer from the context. Return only the answer value.\n\nContext: {context}\nQuestion: {question}\n\n{{"answer": "'

# Prompt: Negative instruction
PROMPT_NEGATIVE = """Answer the question from the context. DO NOT include extra details, locations, descriptions, or parenthetical information. Just the core answer.

Examples:
- WRONG: "Exeter College, Oxford" → RIGHT: "Exeter College"
- WRONG: "Blossom Films owned by Nicole Kidman" → RIGHT: "Blossom Films"
- WRONG: "Southwest City, Missouri" → RIGHT: "Southwest City"

Context: {context}
Question: {question}
Answer:"""

# Prompt: Repeat question
PROMPT_REPEAT = """Context: {context}

Question: {question}
Based on the above context, {question}
Answer:"""

# Prompt: Quote-based
PROMPT_QUOTE = """Find the answer to the question in the context. Quote the shortest relevant phrase.

Context: {context}
Question: {question}
Quoted answer: \""""


# ── Generic Pipeline Runner ──────────────────────────────────────────

def run_iterative_pipeline(sample, retriever, prompt_template=PROMPT_BASELINE,
                           model="phi3:mini", temperature=0.1, max_tokens=32,
                           top_k=3, query_prefix="Represent this sentence for searching relevant passages: ",
                           repeat_penalty=1.1, top_p=0.9,
                           use_gold_context=False, use_gold_chain=False,
                           answer_postprocess=None, n_votes=1,
                           reverse_hops=False, bidirectional=False,
                           include_question_in_context=False,
                           context_format="bracket", max_context_chars=None):
    """Generic iterative pipeline with configurable parameters."""
    decomposition = sample["question_decomposition"]
    paragraphs = sample["paragraphs"]

    if reverse_hops:
        decomposition = list(reversed(decomposition))

    previous_answers = []

    for i, hop in enumerate(decomposition):
        hop_q = format_hop_question(hop["question"], previous_answers)

        if use_gold_context:
            gold_idx = hop["paragraph_support_idx"]
            p = paragraphs[gold_idx]
            if context_format == "bracket":
                context = f"[{p['title']}] {p['paragraph_text']}"
            elif context_format == "titled":
                context = f"Title: {p['title']}\nContent: {p['paragraph_text']}"
            elif context_format == "plain":
                context = p['paragraph_text']
            elif context_format == "numbered":
                context = f"1. [{p['title']}] {p['paragraph_text']}"
            else:
                context = f"[{p['title']}] {p['paragraph_text']}"
        else:
            results = retriever.retrieve(hop_q, paragraphs, top_k=top_k,
                                        query_prefix=query_prefix)
            parts = []
            for j, (idx, score) in enumerate(results):
                p = paragraphs[idx]
                if context_format == "bracket":
                    parts.append(f"[{p['title']}] {p['paragraph_text']}")
                elif context_format == "titled":
                    parts.append(f"Title: {p['title']}\nContent: {p['paragraph_text']}")
                elif context_format == "plain":
                    parts.append(p['paragraph_text'])
                elif context_format == "numbered":
                    parts.append(f"{j+1}. [{p['title']}] {p['paragraph_text']}")
                elif context_format == "scored":
                    parts.append(f"[{p['title']}] (relevance: {score:.2f}) {p['paragraph_text']}")
                else:
                    parts.append(f"[{p['title']}] {p['paragraph_text']}")
            context = "\n\n".join(parts)

        if max_context_chars and len(context) > max_context_chars:
            context = context[:max_context_chars]

        if include_question_in_context:
            context = f"Question: {hop_q}\n\n{context}"

        # Handle voting
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
            # Majority vote
            from collections import Counter
            normalized_answers = [normalize_answer(a) for a in answers]
            counts = Counter(normalized_answers)
            best = counts.most_common(1)[0][0]
            # Find original (non-normalized) version
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


# ── Answer Post-Processors ───────────────────────────────────────────

def postprocess_strip_location(answer):
    """Strip US state/country suffixes."""
    pattern = r',\s*(?:Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New\s+Hampshire|New\s+Jersey|New\s+Mexico|New\s+York|North\s+Carolina|North\s+Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode\s+Island|South\s+Carolina|South\s+Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West\s+Virginia|Wisconsin|Wyoming|England|Scotland|Wales|Ireland|France|Germany|Mexico|Canada|Australia|India|China|Japan|United\s+States|United\s+Kingdom|UK|US|USA)\s*$'
    m = re.search(pattern, answer, re.IGNORECASE)
    if m:
        return answer[:m.start()].strip()
    return answer

def postprocess_strip_parenthetical(answer):
    """Remove parenthetical content."""
    m = re.match(r'^(.+?)\s*\(.*\)\s*$', answer)
    if m and len(m.group(1).strip()) >= 3:
        return m.group(1).strip()
    return answer

def postprocess_first_entity(answer):
    """Take just the first entity (before commas, semicolons, etc)."""
    for sep in [',', ';', ' and ', ' & ', ' featuring ', ' feat.']:
        if sep in answer:
            candidate = answer[:answer.index(sep)].strip()
            if len(candidate) >= 2:
                return candidate
    return answer

def postprocess_strip_qualifiers(answer):
    """Strip 'owned by', 'through', 'via' phrases."""
    for pattern in [
        r'^(.+?)\s+(?:through|via|by|from|owned by|of)\s+',
        r'^(.+?)\s+(?:featuring|feat\.|ft\.)\s+',
    ]:
        m = re.match(pattern, answer, re.IGNORECASE)
        if m and len(m.group(1).strip()) >= 3:
            return m.group(1).strip()
    return answer

def postprocess_combined(answer):
    """All post-processing combined."""
    answer = postprocess_strip_qualifiers(answer)
    answer = postprocess_strip_location(answer)
    answer = postprocess_strip_parenthetical(answer)
    answer = postprocess_first_entity(answer)
    return answer

def postprocess_title_case(answer):
    """Force title case (proper nouns)."""
    return answer.title()

def postprocess_lowercase(answer):
    """Force lowercase."""
    return answer.lower()


# ── Build All 100 Hypotheses ─────────────────────────────────────────

def build_hypotheses(retriever):
    hypotheses = []
    h_id = 0

    # ══════════════════════════════════════════════════════════════════
    # CATEGORY 1: PROMPT ENGINEERING (H1-H20)
    # ══════════════════════════════════════════════════════════════════

    def make_prompt_fn(template, **kwargs):
        def fn(sample, ret, cfg=None):
            return run_iterative_pipeline(sample, ret, prompt_template=template, **kwargs)
        return fn

    h_id += 1  # H1
    hypotheses.append(Hypothesis(h_id, "baseline_4shot", "prompt",
        "Baseline: 4-shot extraction prompt (v2 approach)",
        make_prompt_fn(PROMPT_BASELINE)))

    h_id += 1  # H2
    hypotheses.append(Hypothesis(h_id, "zero_shot", "prompt",
        "Zero-shot: no examples, just instructions",
        make_prompt_fn(PROMPT_ZERO_SHOT)))

    h_id += 1  # H3
    hypotheses.append(Hypothesis(h_id, "two_shot", "prompt",
        "2-shot: fewer examples for less prompt length",
        make_prompt_fn(PROMPT_TWO_SHOT)))

    h_id += 1  # H4
    hypotheses.append(Hypothesis(h_id, "eight_shot", "prompt",
        "8-shot: more examples for better pattern learning",
        make_prompt_fn(PROMPT_EIGHT_SHOT)))

    h_id += 1  # H5
    hypotheses.append(Hypothesis(h_id, "instruction_emphasis", "prompt",
        "Instruction emphasis: CAPS and explicit constraints",
        make_prompt_fn(PROMPT_INSTRUCTION)))

    h_id += 1  # H6
    hypotheses.append(Hypothesis(h_id, "cot_then_answer", "prompt",
        "Chain-of-thought then extract answer after ANSWER:",
        make_prompt_fn(PROMPT_COT_THEN_ANSWER, max_tokens=100)))

    h_id += 1  # H7
    hypotheses.append(Hypothesis(h_id, "role_based", "prompt",
        "Role: 'You are a precise fact extraction system'",
        make_prompt_fn(PROMPT_ROLE)))

    h_id += 1  # H8
    hypotheses.append(Hypothesis(h_id, "json_format", "prompt",
        "JSON-structured output format",
        make_prompt_fn(PROMPT_JSON)))

    h_id += 1  # H9
    hypotheses.append(Hypothesis(h_id, "negative_instruction", "prompt",
        "Negative examples: 'DO NOT include extra details'",
        make_prompt_fn(PROMPT_NEGATIVE)))

    h_id += 1  # H10
    hypotheses.append(Hypothesis(h_id, "repeat_question", "prompt",
        "Repeat the question twice in the prompt",
        make_prompt_fn(PROMPT_REPEAT)))

    h_id += 1  # H11
    hypotheses.append(Hypothesis(h_id, "quote_based", "prompt",
        "Ask model to quote from context",
        make_prompt_fn(PROMPT_QUOTE)))

    h_id += 1  # H12
    hypotheses.append(Hypothesis(h_id, "baseline_gold_context", "prompt",
        "Baseline prompt with gold context (upper bound)",
        make_prompt_fn(PROMPT_BASELINE, use_gold_context=True)))

    h_id += 1  # H13
    hypotheses.append(Hypothesis(h_id, "baseline_gold_chain", "prompt",
        "Baseline with gold answer chaining (isolate extraction)",
        make_prompt_fn(PROMPT_BASELINE, use_gold_chain=True)))

    h_id += 1  # H14
    hypotheses.append(Hypothesis(h_id, "baseline_gold_both", "prompt",
        "Gold context + gold chain (theoretical ceiling)",
        make_prompt_fn(PROMPT_BASELINE, use_gold_context=True, use_gold_chain=True)))

    # Prompt: vary max_tokens
    h_id += 1  # H15
    hypotheses.append(Hypothesis(h_id, "max_tokens_8", "prompt",
        "Very short max_tokens=8 (force brevity)",
        make_prompt_fn(PROMPT_BASELINE, max_tokens=8)))

    h_id += 1  # H16
    hypotheses.append(Hypothesis(h_id, "max_tokens_16", "prompt",
        "Short max_tokens=16",
        make_prompt_fn(PROMPT_BASELINE, max_tokens=16)))

    h_id += 1  # H17
    hypotheses.append(Hypothesis(h_id, "max_tokens_64", "prompt",
        "Longer max_tokens=64 (allow elaboration)",
        make_prompt_fn(PROMPT_BASELINE, max_tokens=64)))

    h_id += 1  # H18
    hypotheses.append(Hypothesis(h_id, "max_tokens_128", "prompt",
        "Long max_tokens=128",
        make_prompt_fn(PROMPT_BASELINE, max_tokens=128)))

    # System message variations
    h_id += 1  # H19
    hypotheses.append(Hypothesis(h_id, "instruction_8shot", "prompt",
        "Instruction emphasis + 8-shot combined",
        make_prompt_fn(PROMPT_EIGHT_SHOT)))

    h_id += 1  # H20
    hypotheses.append(Hypothesis(h_id, "negative_8shot_gold", "prompt",
        "Negative instruction + gold context",
        make_prompt_fn(PROMPT_NEGATIVE, use_gold_context=True)))

    # ══════════════════════════════════════════════════════════════════
    # CATEGORY 2: RETRIEVAL PARAMETERS (H21-H40)
    # ══════════════════════════════════════════════════════════════════

    h_id += 1  # H21
    hypotheses.append(Hypothesis(h_id, "top_k_1", "retrieval",
        "Retrieve only top-1 paragraph",
        make_prompt_fn(PROMPT_BASELINE, top_k=1)))

    h_id += 1  # H22
    hypotheses.append(Hypothesis(h_id, "top_k_2", "retrieval",
        "Retrieve top-2 paragraphs",
        make_prompt_fn(PROMPT_BASELINE, top_k=2)))

    h_id += 1  # H23
    hypotheses.append(Hypothesis(h_id, "top_k_5", "retrieval",
        "Retrieve top-5 paragraphs",
        make_prompt_fn(PROMPT_BASELINE, top_k=5)))

    h_id += 1  # H24
    hypotheses.append(Hypothesis(h_id, "top_k_7", "retrieval",
        "Retrieve top-7 paragraphs",
        make_prompt_fn(PROMPT_BASELINE, top_k=7)))

    h_id += 1  # H25
    hypotheses.append(Hypothesis(h_id, "top_k_10", "retrieval",
        "Retrieve top-10 paragraphs (all context)",
        make_prompt_fn(PROMPT_BASELINE, top_k=10)))

    h_id += 1  # H26
    hypotheses.append(Hypothesis(h_id, "no_query_prefix", "retrieval",
        "No BGE query prefix (raw query embedding)",
        make_prompt_fn(PROMPT_BASELINE, query_prefix="")))

    h_id += 1  # H27
    hypotheses.append(Hypothesis(h_id, "query_prefix_qa", "retrieval",
        "QA-style query prefix",
        make_prompt_fn(PROMPT_BASELINE, query_prefix="Answer this question: ")))

    h_id += 1  # H28
    hypotheses.append(Hypothesis(h_id, "query_prefix_search", "retrieval",
        "Search-style query prefix",
        make_prompt_fn(PROMPT_BASELINE, query_prefix="Search for: ")))

    h_id += 1  # H29
    hypotheses.append(Hypothesis(h_id, "context_format_titled", "retrieval",
        "Context format: 'Title: X\\nContent: Y'",
        make_prompt_fn(PROMPT_BASELINE, context_format="titled")))

    h_id += 1  # H30
    hypotheses.append(Hypothesis(h_id, "context_format_plain", "retrieval",
        "Context format: plain text (no title)",
        make_prompt_fn(PROMPT_BASELINE, context_format="plain")))

    h_id += 1  # H31
    hypotheses.append(Hypothesis(h_id, "context_format_numbered", "retrieval",
        "Context format: numbered paragraphs",
        make_prompt_fn(PROMPT_BASELINE, context_format="numbered")))

    h_id += 1  # H32
    hypotheses.append(Hypothesis(h_id, "context_format_scored", "retrieval",
        "Context format: include relevance score",
        make_prompt_fn(PROMPT_BASELINE, context_format="scored")))

    h_id += 1  # H33
    hypotheses.append(Hypothesis(h_id, "max_context_500", "retrieval",
        "Truncate context to 500 chars",
        make_prompt_fn(PROMPT_BASELINE, max_context_chars=500)))

    h_id += 1  # H34
    hypotheses.append(Hypothesis(h_id, "max_context_1000", "retrieval",
        "Truncate context to 1000 chars",
        make_prompt_fn(PROMPT_BASELINE, max_context_chars=1000)))

    h_id += 1  # H35
    hypotheses.append(Hypothesis(h_id, "max_context_2000", "retrieval",
        "Truncate context to 2000 chars",
        make_prompt_fn(PROMPT_BASELINE, max_context_chars=2000)))

    h_id += 1  # H36
    hypotheses.append(Hypothesis(h_id, "question_in_context", "retrieval",
        "Include question text above context",
        make_prompt_fn(PROMPT_BASELINE, include_question_in_context=True)))

    h_id += 1  # H37
    hypotheses.append(Hypothesis(h_id, "top1_gold_context", "retrieval",
        "Top-1 retrieval + gold context comparison",
        make_prompt_fn(PROMPT_BASELINE, top_k=1, use_gold_context=True)))

    h_id += 1  # H38
    hypotheses.append(Hypothesis(h_id, "top5_8shot", "retrieval",
        "Top-5 retrieval + 8-shot prompt",
        make_prompt_fn(PROMPT_EIGHT_SHOT, top_k=5)))

    h_id += 1  # H39
    hypotheses.append(Hypothesis(h_id, "top1_instruction", "retrieval",
        "Top-1 retrieval + instruction emphasis",
        make_prompt_fn(PROMPT_INSTRUCTION, top_k=1)))

    h_id += 1  # H40
    hypotheses.append(Hypothesis(h_id, "top2_negative", "retrieval",
        "Top-2 retrieval + negative instruction",
        make_prompt_fn(PROMPT_NEGATIVE, top_k=2)))

    # ══════════════════════════════════════════════════════════════════
    # CATEGORY 3: MODEL PARAMETERS (H41-H55)
    # ══════════════════════════════════════════════════════════════════

    h_id += 1  # H41
    hypotheses.append(Hypothesis(h_id, "temp_0.0", "model_params",
        "Temperature 0.0 (greedy decoding)",
        make_prompt_fn(PROMPT_BASELINE, temperature=0.0)))

    h_id += 1  # H42
    hypotheses.append(Hypothesis(h_id, "temp_0.01", "model_params",
        "Temperature 0.01 (near-greedy)",
        make_prompt_fn(PROMPT_BASELINE, temperature=0.01)))

    h_id += 1  # H43
    hypotheses.append(Hypothesis(h_id, "temp_0.3", "model_params",
        "Temperature 0.3 (moderate creativity)",
        make_prompt_fn(PROMPT_BASELINE, temperature=0.3)))

    h_id += 1  # H44
    hypotheses.append(Hypothesis(h_id, "temp_0.5", "model_params",
        "Temperature 0.5",
        make_prompt_fn(PROMPT_BASELINE, temperature=0.5)))

    h_id += 1  # H45
    hypotheses.append(Hypothesis(h_id, "temp_0.7", "model_params",
        "Temperature 0.7 (high creativity)",
        make_prompt_fn(PROMPT_BASELINE, temperature=0.7)))

    h_id += 1  # H46
    hypotheses.append(Hypothesis(h_id, "temp_1.0", "model_params",
        "Temperature 1.0 (maximum entropy)",
        make_prompt_fn(PROMPT_BASELINE, temperature=1.0)))

    h_id += 1  # H47
    hypotheses.append(Hypothesis(h_id, "repeat_penalty_1.0", "model_params",
        "Repeat penalty 1.0 (no penalty)",
        make_prompt_fn(PROMPT_BASELINE, repeat_penalty=1.0)))

    h_id += 1  # H48
    hypotheses.append(Hypothesis(h_id, "repeat_penalty_1.3", "model_params",
        "Repeat penalty 1.3 (high penalty)",
        make_prompt_fn(PROMPT_BASELINE, repeat_penalty=1.3)))

    h_id += 1  # H49
    hypotheses.append(Hypothesis(h_id, "repeat_penalty_1.5", "model_params",
        "Repeat penalty 1.5 (very high)",
        make_prompt_fn(PROMPT_BASELINE, repeat_penalty=1.5)))

    h_id += 1  # H50
    hypotheses.append(Hypothesis(h_id, "top_p_0.5", "model_params",
        "Top-p 0.5 (nucleus sampling)",
        make_prompt_fn(PROMPT_BASELINE, top_p=0.5)))

    h_id += 1  # H51
    hypotheses.append(Hypothesis(h_id, "top_p_0.7", "model_params",
        "Top-p 0.7",
        make_prompt_fn(PROMPT_BASELINE, top_p=0.7)))

    h_id += 1  # H52
    hypotheses.append(Hypothesis(h_id, "top_p_0.95", "model_params",
        "Top-p 0.95",
        make_prompt_fn(PROMPT_BASELINE, top_p=0.95)))

    h_id += 1  # H53
    hypotheses.append(Hypothesis(h_id, "top_p_1.0", "model_params",
        "Top-p 1.0 (no nucleus filtering)",
        make_prompt_fn(PROMPT_BASELINE, top_p=1.0)))

    h_id += 1  # H54
    hypotheses.append(Hypothesis(h_id, "greedy_short", "model_params",
        "Greedy + short output (temp=0, max_tokens=16)",
        make_prompt_fn(PROMPT_BASELINE, temperature=0.0, max_tokens=16)))

    h_id += 1  # H55
    hypotheses.append(Hypothesis(h_id, "optimal_params_guess", "model_params",
        "Best guess params: temp=0.01, top_p=0.7, max=16, rp=1.0",
        make_prompt_fn(PROMPT_BASELINE, temperature=0.01, top_p=0.7, max_tokens=16, repeat_penalty=1.0)))

    # ══════════════════════════════════════════════════════════════════
    # CATEGORY 4: ANSWER PROCESSING (H56-H70)
    # ══════════════════════════════════════════════════════════════════

    h_id += 1  # H56
    hypotheses.append(Hypothesis(h_id, "postprocess_location", "answer_proc",
        "Strip location suffixes (state/country)",
        make_prompt_fn(PROMPT_BASELINE, answer_postprocess=postprocess_strip_location)))

    h_id += 1  # H57
    hypotheses.append(Hypothesis(h_id, "postprocess_parenthetical", "answer_proc",
        "Remove parenthetical content",
        make_prompt_fn(PROMPT_BASELINE, answer_postprocess=postprocess_strip_parenthetical)))

    h_id += 1  # H58
    hypotheses.append(Hypothesis(h_id, "postprocess_first_entity", "answer_proc",
        "Take only first entity (before comma/and)",
        make_prompt_fn(PROMPT_BASELINE, answer_postprocess=postprocess_first_entity)))

    h_id += 1  # H59
    hypotheses.append(Hypothesis(h_id, "postprocess_qualifiers", "answer_proc",
        "Strip 'owned by', 'through', 'via' phrases",
        make_prompt_fn(PROMPT_BASELINE, answer_postprocess=postprocess_strip_qualifiers)))

    h_id += 1  # H60
    hypotheses.append(Hypothesis(h_id, "postprocess_combined", "answer_proc",
        "All post-processing combined",
        make_prompt_fn(PROMPT_BASELINE, answer_postprocess=postprocess_combined)))

    h_id += 1  # H61
    hypotheses.append(Hypothesis(h_id, "postprocess_title_case", "answer_proc",
        "Force title case on answers",
        make_prompt_fn(PROMPT_BASELINE, answer_postprocess=postprocess_title_case)))

    h_id += 1  # H62
    hypotheses.append(Hypothesis(h_id, "postprocess_combined_gold", "answer_proc",
        "All post-processing + gold context",
        make_prompt_fn(PROMPT_BASELINE, use_gold_context=True, answer_postprocess=postprocess_combined)))

    h_id += 1  # H63
    hypotheses.append(Hypothesis(h_id, "vote_3x", "answer_proc",
        "3-vote majority on extraction",
        make_prompt_fn(PROMPT_BASELINE, n_votes=3)))

    h_id += 1  # H64
    hypotheses.append(Hypothesis(h_id, "vote_5x", "answer_proc",
        "5-vote majority on extraction",
        make_prompt_fn(PROMPT_BASELINE, n_votes=5)))

    h_id += 1  # H65
    hypotheses.append(Hypothesis(h_id, "vote_3x_gold", "answer_proc",
        "3-vote majority + gold context",
        make_prompt_fn(PROMPT_BASELINE, n_votes=3, use_gold_context=True)))

    h_id += 1  # H66
    hypotheses.append(Hypothesis(h_id, "negative_postprocess", "answer_proc",
        "Negative instruction + combined post-processing",
        make_prompt_fn(PROMPT_NEGATIVE, answer_postprocess=postprocess_combined)))

    h_id += 1  # H67
    hypotheses.append(Hypothesis(h_id, "8shot_postprocess", "answer_proc",
        "8-shot + combined post-processing",
        make_prompt_fn(PROMPT_EIGHT_SHOT, answer_postprocess=postprocess_combined)))

    h_id += 1  # H68
    hypotheses.append(Hypothesis(h_id, "instruction_postprocess", "answer_proc",
        "Instruction emphasis + combined post-processing",
        make_prompt_fn(PROMPT_INSTRUCTION, answer_postprocess=postprocess_combined)))

    h_id += 1  # H69
    hypotheses.append(Hypothesis(h_id, "vote_3x_8shot", "answer_proc",
        "3-vote + 8-shot prompt",
        make_prompt_fn(PROMPT_EIGHT_SHOT, n_votes=3)))

    h_id += 1  # H70
    hypotheses.append(Hypothesis(h_id, "greedy_postprocess", "answer_proc",
        "Greedy decoding + combined post-processing",
        make_prompt_fn(PROMPT_BASELINE, temperature=0.0, answer_postprocess=postprocess_combined)))

    # ══════════════════════════════════════════════════════════════════
    # CATEGORY 5: ARCHITECTURE / STRATEGY (H71-H85)
    # ══════════════════════════════════════════════════════════════════

    h_id += 1  # H71
    hypotheses.append(Hypothesis(h_id, "single_pass", "architecture",
        "Single pass: all 20 paragraphs at once (baseline comparison)",
        lambda s, r, c=None: run_iterative_pipeline(s, r, top_k=20)))

    h_id += 1  # H72
    hypotheses.append(Hypothesis(h_id, "reverse_hops", "architecture",
        "Reverse hop order (answer hop 2 first)",
        make_prompt_fn(PROMPT_BASELINE, reverse_hops=True)))

    # Qwen 7B as extractor
    h_id += 1  # H73
    hypotheses.append(Hypothesis(h_id, "qwen7b_extractor", "architecture",
        "Qwen 2.5 7B as extractor instead of Phi-3",
        make_prompt_fn(PROMPT_BASELINE, model="qwen2.5:7b")))

    h_id += 1  # H74
    hypotheses.append(Hypothesis(h_id, "qwen7b_gold", "architecture",
        "Qwen 2.5 7B + gold context",
        make_prompt_fn(PROMPT_BASELINE, model="qwen2.5:7b", use_gold_context=True)))

    h_id += 1  # H75
    hypotheses.append(Hypothesis(h_id, "qwen7b_8shot", "architecture",
        "Qwen 2.5 7B + 8-shot prompt",
        make_prompt_fn(PROMPT_EIGHT_SHOT, model="qwen2.5:7b")))

    h_id += 1  # H76
    hypotheses.append(Hypothesis(h_id, "qwen7b_gold_chain", "architecture",
        "Qwen 2.5 7B + gold context + gold chain (ceiling)",
        make_prompt_fn(PROMPT_BASELINE, model="qwen2.5:7b", use_gold_context=True, use_gold_chain=True)))

    # Combo strategies
    h_id += 1  # H77
    hypotheses.append(Hypothesis(h_id, "best_prompt_top1", "architecture",
        "8-shot + top-1 + greedy + postprocess",
        make_prompt_fn(PROMPT_EIGHT_SHOT, top_k=1, temperature=0.0, answer_postprocess=postprocess_combined)))

    h_id += 1  # H78
    hypotheses.append(Hypothesis(h_id, "best_prompt_top2", "architecture",
        "8-shot + top-2 + greedy + postprocess",
        make_prompt_fn(PROMPT_EIGHT_SHOT, top_k=2, temperature=0.0, answer_postprocess=postprocess_combined)))

    h_id += 1  # H79
    hypotheses.append(Hypothesis(h_id, "negative_top2_greedy", "architecture",
        "Negative + top-2 + greedy + postprocess",
        make_prompt_fn(PROMPT_NEGATIVE, top_k=2, temperature=0.0, answer_postprocess=postprocess_combined)))

    h_id += 1  # H80
    hypotheses.append(Hypothesis(h_id, "instruction_top1_greedy", "architecture",
        "Instruction + top-1 + greedy",
        make_prompt_fn(PROMPT_INSTRUCTION, top_k=1, temperature=0.0)))

    h_id += 1  # H81
    hypotheses.append(Hypothesis(h_id, "8shot_vote3_top2", "architecture",
        "8-shot + 3-vote + top-2",
        make_prompt_fn(PROMPT_EIGHT_SHOT, top_k=2, n_votes=3)))

    h_id += 1  # H82
    hypotheses.append(Hypothesis(h_id, "qwen_8shot_top2", "architecture",
        "Qwen 7B + 8-shot + top-2",
        make_prompt_fn(PROMPT_EIGHT_SHOT, model="qwen2.5:7b", top_k=2)))

    h_id += 1  # H83
    hypotheses.append(Hypothesis(h_id, "qwen_negative_top2", "architecture",
        "Qwen 7B + negative prompt + top-2",
        make_prompt_fn(PROMPT_NEGATIVE, model="qwen2.5:7b", top_k=2)))

    h_id += 1  # H84
    hypotheses.append(Hypothesis(h_id, "qwen_instruction_greedy", "architecture",
        "Qwen 7B + instruction + greedy",
        make_prompt_fn(PROMPT_INSTRUCTION, model="qwen2.5:7b", temperature=0.0)))

    h_id += 1  # H85
    hypotheses.append(Hypothesis(h_id, "qwen_vote3_gold", "architecture",
        "Qwen 7B + 3-vote + gold context",
        make_prompt_fn(PROMPT_BASELINE, model="qwen2.5:7b", n_votes=3, use_gold_context=True)))

    # ══════════════════════════════════════════════════════════════════
    # CATEGORY 6: COMBINED / BEST GUESS (H86-H100)
    # ══════════════════════════════════════════════════════════════════

    h_id += 1  # H86
    hypotheses.append(Hypothesis(h_id, "phi3_optimized_v1", "combined",
        "Phi-3: 8shot + top-2 + temp=0 + rp=1.0 + postprocess",
        make_prompt_fn(PROMPT_EIGHT_SHOT, top_k=2, temperature=0.0, repeat_penalty=1.0, answer_postprocess=postprocess_combined)))

    h_id += 1  # H87
    hypotheses.append(Hypothesis(h_id, "phi3_optimized_v2", "combined",
        "Phi-3: negative + top-1 + temp=0.01 + postprocess",
        make_prompt_fn(PROMPT_NEGATIVE, top_k=1, temperature=0.01, answer_postprocess=postprocess_combined)))

    h_id += 1  # H88
    hypotheses.append(Hypothesis(h_id, "phi3_optimized_v3", "combined",
        "Phi-3: instruction + top-3 + temp=0 + max=16",
        make_prompt_fn(PROMPT_INSTRUCTION, top_k=3, temperature=0.0, max_tokens=16)))

    h_id += 1  # H89
    hypotheses.append(Hypothesis(h_id, "phi3_optimized_v4", "combined",
        "Phi-3: 4shot + top-2 + temp=0 + max=16 + postprocess",
        make_prompt_fn(PROMPT_BASELINE, top_k=2, temperature=0.0, max_tokens=16, answer_postprocess=postprocess_combined)))

    h_id += 1  # H90
    hypotheses.append(Hypothesis(h_id, "phi3_optimized_gold", "combined",
        "Phi-3: 8shot + gold_ctx + temp=0 + postprocess",
        make_prompt_fn(PROMPT_EIGHT_SHOT, use_gold_context=True, temperature=0.0, answer_postprocess=postprocess_combined)))

    h_id += 1  # H91
    hypotheses.append(Hypothesis(h_id, "qwen_optimized_v1", "combined",
        "Qwen: 8shot + top-2 + temp=0 + postprocess",
        make_prompt_fn(PROMPT_EIGHT_SHOT, model="qwen2.5:7b", top_k=2, temperature=0.0, answer_postprocess=postprocess_combined)))

    h_id += 1  # H92
    hypotheses.append(Hypothesis(h_id, "qwen_optimized_v2", "combined",
        "Qwen: negative + top-2 + temp=0 + postprocess",
        make_prompt_fn(PROMPT_NEGATIVE, model="qwen2.5:7b", top_k=2, temperature=0.0, answer_postprocess=postprocess_combined)))

    h_id += 1  # H93
    hypotheses.append(Hypothesis(h_id, "qwen_optimized_v3", "combined",
        "Qwen: instruction + top-1 + temp=0 + max=16",
        make_prompt_fn(PROMPT_INSTRUCTION, model="qwen2.5:7b", top_k=1, temperature=0.0, max_tokens=16)))

    h_id += 1  # H94
    hypotheses.append(Hypothesis(h_id, "qwen_optimized_gold", "combined",
        "Qwen: 8shot + gold_ctx + temp=0 + postprocess",
        make_prompt_fn(PROMPT_EIGHT_SHOT, model="qwen2.5:7b", use_gold_context=True, temperature=0.0, answer_postprocess=postprocess_combined)))

    h_id += 1  # H95
    hypotheses.append(Hypothesis(h_id, "qwen_optimized_gold_chain", "combined",
        "Qwen: 8shot + gold_ctx + gold_chain + temp=0 + postprocess",
        make_prompt_fn(PROMPT_EIGHT_SHOT, model="qwen2.5:7b", use_gold_context=True, use_gold_chain=True, temperature=0.0, answer_postprocess=postprocess_combined)))

    h_id += 1  # H96
    hypotheses.append(Hypothesis(h_id, "phi3_vote5_gold", "combined",
        "Phi-3: 5-vote + gold context + 8shot",
        make_prompt_fn(PROMPT_EIGHT_SHOT, n_votes=5, use_gold_context=True)))

    h_id += 1  # H97
    hypotheses.append(Hypothesis(h_id, "phi3_everything_kitchen_sink", "combined",
        "Phi-3: 8shot + top-2 + vote3 + temp0 + postprocess + rp1.0",
        make_prompt_fn(PROMPT_EIGHT_SHOT, top_k=2, n_votes=3, temperature=0.0, repeat_penalty=1.0, answer_postprocess=postprocess_combined)))

    h_id += 1  # H98
    hypotheses.append(Hypothesis(h_id, "qwen_everything_kitchen_sink", "combined",
        "Qwen: 8shot + top-2 + vote3 + temp0 + postprocess",
        make_prompt_fn(PROMPT_EIGHT_SHOT, model="qwen2.5:7b", top_k=2, n_votes=3, temperature=0.0, answer_postprocess=postprocess_combined)))

    h_id += 1  # H99
    hypotheses.append(Hypothesis(h_id, "qwen_max_config", "combined",
        "Qwen: 8shot + gold_ctx + vote5 + temp0 + postprocess (MAX)",
        make_prompt_fn(PROMPT_EIGHT_SHOT, model="qwen2.5:7b", use_gold_context=True, n_votes=5, temperature=0.0, answer_postprocess=postprocess_combined)))

    h_id += 1  # H100
    hypotheses.append(Hypothesis(h_id, "phi3_max_config", "combined",
        "Phi-3: 8shot + gold_ctx + vote5 + temp0 + postprocess (MAX)",
        make_prompt_fn(PROMPT_EIGHT_SHOT, use_gold_context=True, n_votes=5, temperature=0.0, answer_postprocess=postprocess_combined)))

    return hypotheses


# ── Main Runner ──────────────────────────────────────────────────────

def run_battery(limit=15, output_path="evaluation/results/hypothesis_battery.json",
                start_h=1, end_h=100, verbose=False):
    from datasets import load_dataset

    print("Loading MuSiQue validation set...")
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    samples = [s for s in ds if s.get("answerable", True)][:limit]
    print(f"Screening on {len(samples)} questions")

    print("Loading embeddings...")
    retriever = EmbeddingRetriever()
    retriever._load_model()

    hypotheses = build_hypotheses(retriever)
    print(f"Built {len(hypotheses)} hypotheses (testing H{start_h}-H{end_h})")

    results = {}
    start_time = time.time()

    for h in hypotheses:
        if h.id < start_h or h.id > end_h:
            continue

        hr = HypothesisResult(h.id, h.name, h.category)
        h_start = time.time()
        print(f"\n  H{h.id:3d} [{h.category:12s}] {h.name}: ", end="", flush=True)

        for i, sample in enumerate(samples):
            answer = sample["answer"]
            aliases = sample.get("answer_aliases", [])

            try:
                t0 = time.time()
                pred = h.run_fn(sample, retriever)
                lat = (time.time() - t0) * 1000

                em = exact_match(pred, answer, aliases)
                rem = relaxed_match(pred, answer, aliases)
                f1 = best_f1(pred, answer, aliases)

                hr.em_scores.append(em)
                hr.relaxed_em_scores.append(rem)
                hr.f1_scores.append(f1)
                hr.latencies.append(lat)

                if verbose:
                    s = "+" if em else ("~" if rem else ".")
                    print(s, end="", flush=True)
                else:
                    print("+" if em else ".", end="", flush=True)

            except Exception as e:
                hr.errors += 1
                hr.em_scores.append(False)
                hr.relaxed_em_scores.append(False)
                hr.f1_scores.append(0.0)
                hr.latencies.append(0)
                print("X", end="", flush=True)
                if verbose:
                    print(f"\n      ERROR: {e}")

        h_elapsed = time.time() - h_start
        print(f"  EM={hr.em_rate:5.1f}% relEM={hr.relaxed_em_rate:5.1f}% F1={hr.mean_f1:.3f} ({h_elapsed:.0f}s)")

        results[f"H{h.id}"] = {
            "id": h.id,
            "name": h.name,
            "category": h.category,
            "description": h.description,
            "em": hr.em_rate,
            "relaxed_em": hr.relaxed_em_rate,
            "f1": hr.mean_f1,
            "latency_ms": hr.mean_latency,
            "errors": hr.errors,
            "n_samples": len(samples),
        }

        # Save incrementally
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)

    total_elapsed = time.time() - start_time

    # Print summary
    print("\n" + "=" * 80)
    print("HYPOTHESIS BATTERY - FINAL RESULTS")
    print("=" * 80)
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"\n{'H#':>4s} {'Name':<35s} {'Cat':<12s} {'EM':>6s} {'relEM':>6s} {'F1':>6s}")
    print("-" * 80)

    # Sort by EM
    sorted_results = sorted(results.values(), key=lambda x: x["em"], reverse=True)
    for r in sorted_results:
        marker = " ***" if r["em"] >= 40 else (" ** " if r["em"] >= 30 else "")
        print(f"H{r['id']:3d} {r['name']:<35s} {r['category']:<12s} {r['em']:5.1f}% {r['relaxed_em']:5.1f}% {r['f1']:.3f}{marker}")

    # Category summary
    print("\n" + "-" * 60)
    print("Category Averages:")
    cats = defaultdict(list)
    for r in results.values():
        cats[r["category"]].append(r["em"])
    for cat, ems in sorted(cats.items(), key=lambda x: -np.mean(x[1])):
        print(f"  {cat:<15s}: avg EM = {np.mean(ems):5.1f}% (best = {max(ems):5.1f}%, n={len(ems)})")

    # Top 10
    print("\n" + "-" * 60)
    print("TOP 10 HYPOTHESES:")
    for i, r in enumerate(sorted_results[:10]):
        print(f"  {i+1}. H{r['id']:3d} {r['name']:<35s} EM={r['em']:5.1f}% F1={r['f1']:.3f}")

    print(f"\nResults saved to: {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="100-Hypothesis Battery Test")
    parser.add_argument("--limit", type=int, default=15, help="Questions per hypothesis (screening)")
    parser.add_argument("--output", type=str, default="evaluation/results/hypothesis_battery.json")
    parser.add_argument("--start", type=int, default=1, help="Start hypothesis ID")
    parser.add_argument("--end", type=int, default=100, help="End hypothesis ID")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Verify Ollama
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"Ollama models: {models}")
        assert "phi3:mini" in models, "phi3:mini not found"
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    run_battery(limit=args.limit, output_path=args.output,
                start_h=args.start, end_h=args.end, verbose=args.verbose)


if __name__ == "__main__":
    main()
