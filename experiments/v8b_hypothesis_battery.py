#!/usr/bin/env python3
"""
Iterative Retrieval v8b - 250-Hypothesis Knowledge Graph Battery

Systematic exploration of the KG parameter space. Each hypothesis is tested
independently against the same baseline on n=30 MuSiQue 2-hop questions.

v8 findings (baseline to beat):
  - baseline_auto:    63.3% EM (no KG)
  - kg_context_only:  53.3% EM (-10%, context enrichment HURTS)
  - kg_bridging_only: 63.3% EM (neutral)
  - kg_retrieval_only: 63.3% EM (neutral)

250 hypotheses across 6 categories:
  A: Context Enrichment (65) — max_triples, confidence, hop filter, format, relations
  B: Hop Bridging (45) — boost, fuzzy threshold, neighbors, conditional activation
  C: Retrieval Augmentation (40) — alpha, entity matching, score function
  D: Pipeline Parameters (35) — top_k, temperature, max_tokens, context format
  E: Extraction Prompt Variations (35) — n-shot, KG-specific prompts, instructions
  F: Cross-Category Combinations (30) — best of each group combined

Features:
  - Incremental saves after each hypothesis (resume on crash)
  - Reuses v8's KG extraction cache
  - Pre-computes decompositions and embeddings once
  - ~6 hours overnight for full battery
"""

import sys
import time
import json
import re
import string
import hashlib
import argparse
import requests
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple


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


# ── Decomposition ───────────────────────────────────────────────────

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
        model="qwen2.5:7b", temperature=0.1, max_tokens=80
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


# ══════════════════════════════════════════════════════════════════════
# KG Infrastructure (from v8)
# ══════════════════════════════════════════════════════════════════════

ENTITY_TYPES = {"PERSON", "PLACE", "ORGANIZATION", "WORK", "EVENT"}
CORE_RELATIONS = {
    "spouse_of", "child_of", "parent_of", "employer_of", "headquartered_in",
    "founded_by", "owned_by", "performer_of", "author_of", "director_of",
    "manufacturer_of", "located_in", "borders", "capital_of", "member_of",
}

# Relation subsets for filtering experiments
PERSON_RELATIONS = {"spouse_of", "child_of", "parent_of", "employer_of", "author_of", "director_of", "performer_of"}
PLACE_RELATIONS = {"located_in", "headquartered_in", "borders", "capital_of"}
ORG_RELATIONS = {"founded_by", "owned_by", "employer_of", "headquartered_in", "member_of"}
OWNERSHIP_RELATIONS = {"owned_by", "founded_by", "manufacturer_of"}
FAMILY_RELATIONS = {"spouse_of", "child_of", "parent_of"}
CREATION_RELATIONS = {"performer_of", "author_of", "director_of", "manufacturer_of"}

REL_DISPLAY = {
    "spouse_of": "spouse of", "child_of": "child of", "parent_of": "parent of",
    "employer_of": "employer of", "headquartered_in": "headquartered in",
    "founded_by": "founded by", "owned_by": "owned by",
    "performer_of": "performer of", "author_of": "author of",
    "director_of": "director of", "manufacturer_of": "manufacturer of",
    "located_in": "located in", "borders": "borders",
    "capital_of": "capital of", "member_of": "member of",
}


def _build_extraction_prompt(title: str, text: str) -> str:
    return (
        'Extract structured facts (triples) from this paragraph. For each fact, provide:\n'
        '- subject: entity name\n'
        '- subject_type: PERSON, PLACE, ORGANIZATION, WORK, or EVENT\n'
        '- relation: one of [spouse_of, child_of, parent_of, employer_of, headquartered_in, '
        'founded_by, owned_by, performer_of, author_of, director_of, manufacturer_of, '
        'located_in, borders, capital_of, member_of]\n'
        '- object: entity name\n'
        '- object_type: PERSON, PLACE, ORGANIZATION, WORK, or EVENT\n'
        '- confidence: 0.0 to 1.0\n\n'
        'Output JSON array only. Max 10 triples. Only extract facts explicitly stated.\n\n'
        'Example input: "[Steve Hillage] Green is the fourth studio album by British progressive rock musician Steve Hillage."\n'
        'Example output: [{"subject": "Steve Hillage", "subject_type": "PERSON", "relation": "performer_of", "object": "Green", "object_type": "WORK", "confidence": 0.95}]\n\n'
        'Example input: "[Orion Pictures] The film was distributed by Orion Pictures, founded by Mike Medavoy and four other executives."\n'
        'Example output: [{"subject": "Orion Pictures", "subject_type": "ORGANIZATION", "relation": "founded_by", "object": "Mike Medavoy", "object_type": "PERSON", "confidence": 0.9}]\n\n'
        'Example input: "[Canyon, Texas] Canyon is a city in and the county seat of Randall County, Texas, United States."\n'
        'Example output: [{"subject": "Canyon", "subject_type": "PLACE", "relation": "located_in", "object": "Randall County", "object_type": "PLACE", "confidence": 0.95}, '
        '{"subject": "Canyon", "subject_type": "PLACE", "relation": "located_in", "object": "Texas", "object_type": "PLACE", "confidence": 0.9}]\n\n'
        'Example input: "[Miquette Giraudy] Miquette Giraudy is a keyboard player, best known for her work with her partner Steve Hillage."\n'
        'Example output: [{"subject": "Miquette Giraudy", "subject_type": "PERSON", "relation": "spouse_of", "object": "Steve Hillage", "object_type": "PERSON", "confidence": 0.85}]\n\n'
        f'Paragraph: [{title}] {text}\nOutput:'
    )


@dataclass
class KGTriple:
    subject: str
    subject_type: str
    relation: str
    object: str
    object_type: str
    source_paragraph_idx: int
    confidence: float


class KnowledgeGraph:
    def __init__(self):
        self.entities: Dict[str, dict] = {}
        self.edges: List[KGTriple] = []
        self.paragraph_index: Dict[int, Set[str]] = defaultdict(set)
        self.entity_paragraphs: Dict[str, Set[int]] = defaultdict(set)

    def _normalize(self, name: str) -> str:
        return normalize_answer(name)

    def add_triple(self, triple: KGTriple):
        self.edges.append(triple)
        for name, etype in [(triple.subject, triple.subject_type), (triple.object, triple.object_type)]:
            norm = self._normalize(name)
            if norm not in self.entities:
                self.entities[norm] = {"name": name, "type": etype, "aliases": set()}
            self.entities[norm]["aliases"].add(name)
            self.paragraph_index[triple.source_paragraph_idx].add(norm)
            self.entity_paragraphs[norm].add(triple.source_paragraph_idx)

    def find_entity(self, name: str) -> Optional[str]:
        norm = self._normalize(name)
        return norm if norm in self.entities else None

    def fuzzy_find(self, name: str, threshold: float = 0.6) -> Optional[str]:
        norm = self._normalize(name)
        if norm in self.entities:
            return norm
        best, best_score = None, 0.0
        for key in self.entities:
            if not key or not norm:
                continue
            if norm in key or key in norm:
                score = min(len(norm), len(key)) / max(len(norm), len(key))
                if score > best_score and score >= threshold:
                    best_score, best = score, key
            else:
                nt, kt = set(norm.split()), set(key.split())
                if nt and kt:
                    score = len(nt & kt) / max(len(nt), len(kt))
                    if score > best_score and score >= threshold:
                        best_score, best = score, key
        return best

    def get_neighbors(self, entity_name: str) -> List[Tuple[str, KGTriple]]:
        norm = self._normalize(entity_name)
        neighbors = []
        for edge in self.edges:
            sn, on = self._normalize(edge.subject), self._normalize(edge.object)
            if sn == norm:
                neighbors.append((on, edge))
            elif on == norm:
                neighbors.append((sn, edge))
        return neighbors

    def get_triples_for_entity(self, name: str, max_triples: int = 5,
                                confidence_threshold: float = 0.0,
                                relation_filter: Optional[Set[str]] = None) -> List[KGTriple]:
        norm = self._normalize(name)
        triples = []
        for edge in self.edges:
            if self._normalize(edge.subject) == norm or self._normalize(edge.object) == norm:
                if edge.confidence >= confidence_threshold:
                    if relation_filter is None or edge.relation in relation_filter:
                        triples.append(edge)
        triples.sort(key=lambda t: t.confidence, reverse=True)
        return triples[:max_triples]

    def get_paragraphs_for_entity(self, name: str) -> Set[int]:
        norm = self._normalize(name)
        return self.entity_paragraphs.get(norm, set())


class KGExtractor:
    def __init__(self, cache_dir: str = "kg_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, List[dict]] = {}
        self._load_cache()

    def _load_cache(self):
        path = self.cache_dir / "paragraph_triples.json"
        if path.exists():
            with open(path) as f:
                self.cache = json.load(f)

    def save_cache(self):
        with open(self.cache_dir / "paragraph_triples.json", "w") as f:
            json.dump(self.cache, f)

    @staticmethod
    def _hash(title, text):
        return hashlib.sha256(f"{title}|||{text}".encode()).hexdigest()[:16]

    def extract(self, title, text, idx) -> List[KGTriple]:
        h = self._hash(title, text)
        if h not in self.cache:
            prompt = _build_extraction_prompt(title, text)
            try:
                resp = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=512)
                self.cache[h] = self._parse(resp)
            except Exception:
                self.cache[h] = []
        return [KGTriple(
            subject=t["subject"], subject_type=t.get("subject_type", "PERSON"),
            relation=t["relation"], object=t["object"],
            object_type=t.get("object_type", "PERSON"),
            source_paragraph_idx=idx, confidence=t.get("confidence", 0.8),
        ) for t in self.cache[h]]

    def _parse(self, response: str) -> List[dict]:
        response = response.strip()
        for attempt in [
            lambda: json.loads(response),
            lambda: json.loads(re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL).group(1)),
            lambda: json.loads(response[response.find('['):response.rfind(']')+1]),
        ]:
            try:
                data = attempt()
                if isinstance(data, list):
                    return self._validate(data)
            except (json.JSONDecodeError, AttributeError, ValueError):
                continue
        return []

    def _validate(self, data: list) -> List[dict]:
        rel_map = {
            "spouse": "spouse_of", "married_to": "spouse_of",
            "child": "child_of", "son_of": "child_of", "daughter_of": "child_of",
            "parent": "parent_of", "father_of": "parent_of", "mother_of": "parent_of",
            "employer": "employer_of", "works_for": "employer_of",
            "headquarters": "headquartered_in",
            "founded": "founded_by", "creator": "founded_by",
            "owned": "owned_by", "owner": "owned_by",
            "performer": "performer_of", "performed": "performer_of",
            "author": "author_of", "written_by": "author_of", "wrote": "author_of",
            "director": "director_of", "directed": "director_of",
            "manufacturer": "manufacturer_of", "made_by": "manufacturer_of",
            "location": "located_in", "in": "located_in",
            "border": "borders", "adjacent_to": "borders",
            "capital": "capital_of", "member": "member_of", "part_of": "member_of",
        }
        valid = []
        for item in data[:10]:
            if not isinstance(item, dict):
                continue
            if not all(k in item for k in ("subject", "relation", "object")):
                continue
            rel = item["relation"].lower().strip().replace(" ", "_")
            rel = rel_map.get(rel, rel)
            if rel not in CORE_RELATIONS:
                continue
            st = item.get("subject_type", "PERSON").upper()
            ot = item.get("object_type", "PERSON").upper()
            if st not in ENTITY_TYPES: st = "PERSON"
            if ot not in ENTITY_TYPES: ot = "PERSON"
            valid.append({
                "subject": str(item["subject"]).strip(),
                "subject_type": st, "relation": rel,
                "object": str(item["object"]).strip(),
                "object_type": ot,
                "confidence": float(item.get("confidence", 0.8)),
            })
        return valid

    def build_graph(self, paragraphs, confidence_threshold=0.0):
        kg = KnowledgeGraph()
        for idx, p in enumerate(paragraphs):
            for t in self.extract(p["title"], p["paragraph_text"], idx):
                if t.confidence >= confidence_threshold:
                    kg.add_triple(t)
        return kg


# ══════════════════════════════════════════════════════════════════════
# Hypothesis Configuration System
# ══════════════════════════════════════════════════════════════════════

@dataclass
class HypothesisConfig:
    """Every tunable parameter for one hypothesis test."""
    id: str
    name: str
    description: str
    category: str

    # Context enrichment
    use_kg_context: bool = False
    max_context_triples: int = 5
    context_confidence_threshold: float = 0.7
    context_hop_filter: str = "both"      # "hop1", "hop2", "both"
    context_format: str = "structured"    # "structured", "natural", "entities_only", "inline", "verbose"
    context_relation_filter: Optional[str] = None  # key into RELATION_SUBSETS
    context_prompt_position: str = "before_context"  # "before_context", "after_context", "system"

    # Bridging
    use_kg_bridging: bool = False
    bridge_boost: float = 0.15
    bridge_fuzzy_threshold: float = 0.6
    bridge_include_neighbors: bool = True
    bridge_conditional_threshold: Optional[float] = None  # Only bridge when best_embed < this
    bridge_entity_processing: str = "full"  # "full", "normalized", "first_token"

    # Retrieval augmentation
    use_kg_retrieval: bool = False
    retrieval_alpha: float = 0.3
    retrieval_entity_threshold: float = 0.5
    retrieval_score_function: str = "fractional"  # "binary", "fractional", "confidence_weighted"

    # Pipeline parameters
    top_k: int = 3
    extraction_temperature: float = 0.1
    extraction_max_tokens: int = 32
    context_window: str = "full"  # "full", "title_only", "text_only", "truncated_200"

    # Prompt template
    prompt_key: str = "8shot"  # Key into PROMPTS dict

    # KG extraction threshold (applied when building per-question graph)
    kg_confidence_threshold: float = 0.0


# ── Prompt Templates ─────────────────────────────────────────────────

PROMPTS = {}

PROMPTS["8shot"] = """Answer the question using the context below. Give ONLY the specific name, place, or fact. One or two words maximum.

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

PROMPTS["4shot"] = """Answer the question using the context below. Give ONLY the specific name, place, or fact. Be as concise as possible - just the core answer, no extra details.

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

PROMPTS["2shot"] = """Answer the question using ONLY the context. Give the specific name, place, or fact. One or two words.

Context: [Steve Hillage] Green is the fourth studio album by British progressive rock musician Steve Hillage.
Question: Who performed Green?
Answer: Steve Hillage

Context: [Canyon, Texas] Canyon is a city in and the county seat of Randall County, Texas, United States.
Question: What administrative region is Canyon located in?
Answer: Randall County

Context: {context}
Question: {question}
Answer:"""

PROMPTS["0shot"] = """Answer the question using ONLY the provided context. Give ONLY the answer - one or two words maximum. No explanation.

Context: {context}
Question: {question}
Answer:"""

PROMPTS["10shot"] = """Answer the question using the context below. Give ONLY the specific name, place, or fact. One or two words maximum.

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

Context: [Niassa Province] Niassa Province borders Cabo Delgado Province to the east and Zambezia Province to the south.
Question: What province borders Niassa to the east?
Answer: Cabo Delgado Province

Context: [Learjet 60] The Learjet 60 is a mid-size business jet manufactured by Bombardier Aerospace, a subsidiary of Bombardier Inc.
Question: Who manufactured Learjet 60?
Answer: Bombardier Aerospace

Context: {context}
Question: {question}
Answer:"""

PROMPTS["strict"] = """Extract the answer from the context. Reply with ONLY the exact entity name. Nothing else.

Context: {context}
Question: {question}
Answer:"""

PROMPTS["cot"] = """Answer the question using the context below. Think step by step, then give the final answer on the last line starting with "Answer:".

Context: {context}
Question: {question}
Reasoning:"""

PROMPTS["8shot_kg_before"] = """Answer the question using the known facts and context below. Give ONLY the specific name, place, or fact. One or two words maximum.

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

{kg_facts}Context: {context}
Question: {question}
Answer:"""

PROMPTS["8shot_kg_after"] = """Answer the question using the context below. Give ONLY the specific name, place, or fact. One or two words maximum.

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
{kg_facts}Question: {question}
Answer:"""

PROMPTS["kg_only"] = """Use ONLY the following facts to answer the question. Give the specific name, place, or fact. One or two words.

{kg_facts}
Question: {question}
Answer:"""

PROMPTS["kg_prioritize"] = """Answer the question using the facts and context below. PRIORITIZE the structured facts over the paragraph text. Give ONLY the specific answer. One or two words.

{kg_facts}Context: {context}
Question: {question}
Answer:"""

PROMPTS["kg_hint"] = """Answer the question using the context below. Give ONLY the specific name, place, or fact. One or two words maximum.

Hint - these related facts may help:
{kg_facts}
Context: {context}
Question: {question}
Answer:"""


RELATION_SUBSETS = {
    "person": PERSON_RELATIONS,
    "place": PLACE_RELATIONS,
    "org": ORG_RELATIONS,
    "ownership": OWNERSHIP_RELATIONS,
    "family": FAMILY_RELATIONS,
    "creation": CREATION_RELATIONS,
    "no_location": CORE_RELATIONS - PLACE_RELATIONS,
    "no_family": CORE_RELATIONS - FAMILY_RELATIONS,
}


# ══════════════════════════════════════════════════════════════════════
# Hypothesis Generation — 250 Hypotheses
# ══════════════════════════════════════════════════════════════════════

def generate_hypotheses() -> List[HypothesisConfig]:
    hypotheses = []
    n = [0]  # mutable counter

    def h(cat, name, desc, **kwargs):
        n[0] += 1
        hypotheses.append(HypothesisConfig(
            id=f"{cat}{n[0]:03d}", name=name, description=desc, category=cat, **kwargs
        ))

    # ── GROUP A: Context Enrichment (65 hypotheses) ──────────────

    # A1-A7: Max triples sweep
    for mt in [1, 2, 3, 4, 7, 10, 15]:
        h("A", f"ctx_max{mt}", f"Context enrichment with max {mt} triples",
          use_kg_context=True, max_context_triples=mt)

    # A8-A14: Confidence threshold sweep
    for ct in [0.0, 0.3, 0.5, 0.6, 0.8, 0.9, 0.95]:
        h("A", f"ctx_conf{ct}", f"Context enrichment with confidence >= {ct}",
          use_kg_context=True, context_confidence_threshold=ct)

    # A15-A17: Hop filter
    for hf in ["hop1", "hop2", "both"]:
        h("A", f"ctx_hop_{hf}", f"Context enrichment only on {hf}",
          use_kg_context=True, context_hop_filter=hf)

    # A18-A22: Context format
    for fmt in ["structured", "natural", "entities_only", "inline", "verbose"]:
        h("A", f"ctx_fmt_{fmt}", f"Context format: {fmt}",
          use_kg_context=True, context_format=fmt)

    # A23-A30: Relation type filtering
    for rel_key in ["person", "place", "org", "ownership", "family", "creation", "no_location", "no_family"]:
        h("A", f"ctx_rel_{rel_key}", f"Context with only {rel_key} relations",
          use_kg_context=True, context_relation_filter=rel_key)

    # A31-A33: Prompt position
    for pos in ["before_context", "after_context", "system"]:
        h("A", f"ctx_pos_{pos}", f"KG facts placed {pos}",
          use_kg_context=True, context_prompt_position=pos)

    # A34-A39: Max triples × hop2 only (most promising combo from v8 analysis)
    for mt in [1, 2, 3, 4, 5, 7]:
        h("A", f"ctx_h2_max{mt}", f"Hop2-only context, max {mt} triples",
          use_kg_context=True, max_context_triples=mt, context_hop_filter="hop2")

    # A40-A45: Max triples × high confidence
    for mt in [1, 2, 3, 5, 7, 10]:
        h("A", f"ctx_hiconf_max{mt}", f"High-confidence ({'>'}=0.9) context, max {mt}",
          use_kg_context=True, max_context_triples=mt, context_confidence_threshold=0.9)

    # A46-A51: Max triples × hop1 only
    for mt in [1, 2, 3, 5, 7, 10]:
        h("A", f"ctx_h1_max{mt}", f"Hop1-only context, max {mt}",
          use_kg_context=True, max_context_triples=mt, context_hop_filter="hop1")

    # A52-A55: Entities only format × different counts
    for mt in [3, 5, 10, 15]:
        h("A", f"ctx_ent_max{mt}", f"Entities-only format, max {mt}",
          use_kg_context=True, max_context_triples=mt, context_format="entities_only")

    # A56-A59: Natural language format × different counts
    for mt in [1, 3, 5, 10]:
        h("A", f"ctx_nat_max{mt}", f"Natural language format, max {mt}",
          use_kg_context=True, max_context_triples=mt, context_format="natural")

    # A60-A65: Specific prompt templates for KG context
    for pk in ["8shot_kg_before", "8shot_kg_after", "kg_only", "kg_prioritize", "kg_hint", "strict"]:
        h("A", f"ctx_prompt_{pk}", f"Context with {pk} prompt template",
          use_kg_context=True, prompt_key=pk)

    # ── GROUP B: Hop Bridging (45 hypotheses) ────────────────────

    # B1-B10: Boost sweep
    for boost in [0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
        h("B", f"bridge_boost{boost}", f"Bridge boost = {boost}",
          use_kg_bridging=True, bridge_boost=boost)

    # B11-B17: Fuzzy threshold sweep
    for ft in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        h("B", f"bridge_fuzz{ft}", f"Bridge fuzzy threshold = {ft}",
          use_kg_bridging=True, bridge_fuzzy_threshold=ft)

    # B18-B19: Neighbor inclusion
    h("B", "bridge_no_neighbors", "Bridge without KG neighbor expansion",
      use_kg_bridging=True, bridge_include_neighbors=False)
    h("B", "bridge_with_neighbors", "Bridge with KG neighbor expansion",
      use_kg_bridging=True, bridge_include_neighbors=True)

    # B20-B25: Conditional bridging (only when embedding is weak)
    for ct in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        h("B", f"bridge_cond{ct}", f"Bridge only when best embed < {ct}",
          use_kg_bridging=True, bridge_conditional_threshold=ct)

    # B26-B28: Bridge entity processing
    for ep in ["full", "normalized", "first_token"]:
        h("B", f"bridge_ent_{ep}", f"Bridge entity processing: {ep}",
          use_kg_bridging=True, bridge_entity_processing=ep)

    # B29-B34: Top-k with bridging
    for tk in [1, 2, 4, 5, 7, 10]:
        h("B", f"bridge_topk{tk}", f"Bridge with top_k={tk}",
          use_kg_bridging=True, top_k=tk)

    # B35-B38: Bridge + high boost + specific settings
    h("B", "bridge_aggressive", "Aggressive bridge: boost=0.5, fuzz=0.4, neighbors",
      use_kg_bridging=True, bridge_boost=0.5, bridge_fuzzy_threshold=0.4)
    h("B", "bridge_conservative", "Conservative bridge: boost=0.05, fuzz=0.8, no neighbors",
      use_kg_bridging=True, bridge_boost=0.05, bridge_fuzzy_threshold=0.8, bridge_include_neighbors=False)
    h("B", "bridge_medium", "Medium bridge: boost=0.1, fuzz=0.6",
      use_kg_bridging=True, bridge_boost=0.1, bridge_fuzzy_threshold=0.6)
    h("B", "bridge_topk1_boost05", "Bridge top_k=1, boost=0.5",
      use_kg_bridging=True, top_k=1, bridge_boost=0.5)

    # B39-B45: Bridge + different conditional + boost combos
    for boost, cond in [(0.1, 0.5), (0.2, 0.5), (0.3, 0.4), (0.5, 0.4), (0.1, 0.6), (0.3, 0.6), (0.5, 0.3)]:
        h("B", f"bridge_b{boost}_c{cond}", f"Bridge boost={boost}, conditional<{cond}",
          use_kg_bridging=True, bridge_boost=boost, bridge_conditional_threshold=cond)

    # ── GROUP C: Retrieval Augmentation (40 hypotheses) ──────────

    # C1-C12: Alpha sweep
    for alpha in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.4, 0.5, 0.6, 0.7, 0.9]:
        h("C", f"retr_alpha{alpha}", f"KG retrieval alpha = {alpha}",
          use_kg_retrieval=True, retrieval_alpha=alpha)

    # C13-C18: Entity matching threshold
    for et in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        h("C", f"retr_entthresh{et}", f"Retrieval entity match threshold = {et}",
          use_kg_retrieval=True, retrieval_entity_threshold=et)

    # C19-C21: Score function
    for sf in ["binary", "fractional", "confidence_weighted"]:
        h("C", f"retr_score_{sf}", f"Retrieval KG score function: {sf}",
          use_kg_retrieval=True, retrieval_score_function=sf)

    # C22-C27: Alpha × top_k
    for alpha, tk in [(0.1, 1), (0.1, 5), (0.3, 1), (0.3, 5), (0.5, 1), (0.5, 5)]:
        h("C", f"retr_a{alpha}_k{tk}", f"Retrieval alpha={alpha}, top_k={tk}",
          use_kg_retrieval=True, retrieval_alpha=alpha, top_k=tk)

    # C28-C33: Alpha × score function combos
    for alpha, sf in [(0.1, "binary"), (0.1, "confidence_weighted"),
                       (0.3, "binary"), (0.3, "confidence_weighted"),
                       (0.5, "binary"), (0.5, "confidence_weighted")]:
        h("C", f"retr_a{alpha}_{sf[:4]}", f"Retrieval alpha={alpha}, score={sf}",
          use_kg_retrieval=True, retrieval_alpha=alpha, retrieval_score_function=sf)

    # C34-C40: Retrieval + KG confidence filtering
    for alpha, conf in [(0.1, 0.5), (0.1, 0.8), (0.3, 0.5), (0.3, 0.8),
                         (0.5, 0.5), (0.5, 0.8), (0.7, 0.8)]:
        h("C", f"retr_a{alpha}_kg{conf}", f"Retrieval alpha={alpha}, KG conf>={conf}",
          use_kg_retrieval=True, retrieval_alpha=alpha, kg_confidence_threshold=conf)

    # ── GROUP D: Pipeline Parameters (35 hypotheses) ─────────────

    # D1-D6: Top-k sweep (no KG)
    for tk in [1, 2, 4, 5, 7, 10]:
        h("D", f"topk{tk}", f"Retrieval top_k={tk} (no KG)",
          top_k=tk)

    # D7-D11: Temperature sweep
    for temp in [0.0, 0.05, 0.2, 0.3, 0.5]:
        h("D", f"temp{temp}", f"Extraction temperature={temp}",
          extraction_temperature=temp)

    # D12-D16: Max tokens sweep
    for mt in [8, 16, 24, 48, 64]:
        h("D", f"maxtok{mt}", f"Extraction max_tokens={mt}",
          extraction_max_tokens=mt)

    # D17-D20: Context window format
    for cw in ["full", "title_only", "text_only", "truncated_200"]:
        h("D", f"ctx_win_{cw}", f"Context window: {cw}",
          context_window=cw)

    # D21-D25: Prompt template variations (no KG)
    for pk in ["4shot", "2shot", "0shot", "10shot", "strict"]:
        h("D", f"prompt_{pk}", f"Prompt template: {pk}",
          prompt_key=pk)

    # D26: Chain-of-thought
    h("D", "prompt_cot", "Chain-of-thought extraction",
      prompt_key="cot", extraction_max_tokens=128)

    # D27-D31: Top-k × temperature combos
    for tk, temp in [(1, 0.0), (2, 0.0), (5, 0.0), (1, 0.2), (5, 0.2)]:
        h("D", f"k{tk}_t{temp}", f"top_k={tk}, temp={temp}",
          top_k=tk, extraction_temperature=temp)

    # D32-D35: Top-k × prompt combos
    for tk, pk in [(1, "strict"), (2, "4shot"), (5, "10shot"), (1, "0shot")]:
        h("D", f"k{tk}_{pk}", f"top_k={tk}, prompt={pk}",
          top_k=tk, prompt_key=pk)

    # ── GROUP E: KG-Specific Prompts (35 hypotheses) ─────────────

    # E1-E6: KG context prompts × max_triples
    for pk, mt in [("kg_hint", 1), ("kg_hint", 3), ("kg_hint", 5),
                    ("kg_prioritize", 1), ("kg_prioritize", 3), ("kg_prioritize", 5)]:
        h("E", f"kgp_{pk}_m{mt}", f"KG prompt={pk}, max={mt}",
          use_kg_context=True, prompt_key=pk, max_context_triples=mt)

    # E7-E12: KG-only extraction (no paragraph context!)
    for mt in [1, 3, 5, 7, 10, 15]:
        h("E", f"kgonly_m{mt}", f"KG-only extraction (no context), max={mt}",
          use_kg_context=True, prompt_key="kg_only", max_context_triples=mt)

    # E13-E18: KG context + different confidence × format combos
    for conf, fmt in [(0.5, "natural"), (0.5, "structured"), (0.8, "natural"),
                       (0.8, "structured"), (0.9, "natural"), (0.9, "entities_only")]:
        h("E", f"kgp_c{conf}_{fmt[:3]}", f"KG conf>={conf}, format={fmt}",
          use_kg_context=True, context_confidence_threshold=conf, context_format=fmt)

    # E19-E24: Hop2-only KG context with different prompts
    for pk in ["8shot_kg_before", "8shot_kg_after", "kg_hint", "kg_prioritize", "strict", "4shot"]:
        h("E", f"h2_{pk}", f"Hop2-only KG context, prompt={pk}",
          use_kg_context=True, context_hop_filter="hop2",
          prompt_key=pk if "kg" in pk else "8shot_kg_before")

    # E25-E30: Hop1-only KG context with relation filtering
    for rel_key in ["person", "place", "creation", "ownership", "family", "no_location"]:
        h("E", f"h1_rel_{rel_key}", f"Hop1-only context, relations={rel_key}",
          use_kg_context=True, context_hop_filter="hop1", context_relation_filter=rel_key)

    # E31-E35: Context with strict entity matching (only exact matches in question)
    for mt, conf in [(1, 0.9), (2, 0.9), (3, 0.8), (5, 0.7), (1, 0.95)]:
        h("E", f"strict_m{mt}_c{conf}", f"Strict context: max={mt}, conf>={conf}",
          use_kg_context=True, max_context_triples=mt, context_confidence_threshold=conf,
          context_format="structured")

    # ── GROUP F: Cross-Category Combinations (30 hypotheses) ─────

    # F1-F5: Bridge + Context combos
    h("F", "bridge_ctx1", "Bridge + 1 triple context",
      use_kg_bridging=True, use_kg_context=True, max_context_triples=1)
    h("F", "bridge_ctx2_h2", "Bridge + 2 triple hop2-only context",
      use_kg_bridging=True, use_kg_context=True, max_context_triples=2, context_hop_filter="hop2")
    h("F", "bridge_ctx3_nat", "Bridge + 3 triple natural context",
      use_kg_bridging=True, use_kg_context=True, max_context_triples=3, context_format="natural")
    h("F", "bridge_ctx1_hiconf", "Bridge + 1 triple high-conf context",
      use_kg_bridging=True, use_kg_context=True, max_context_triples=1, context_confidence_threshold=0.9)
    h("F", "bridge_b03_ctx2", "Bridge boost=0.3 + 2 triple context",
      use_kg_bridging=True, bridge_boost=0.3, use_kg_context=True, max_context_triples=2)

    # F6-F10: Retrieval + Context combos
    h("F", "retr01_ctx1", "Retrieval alpha=0.1 + 1 triple context",
      use_kg_retrieval=True, retrieval_alpha=0.1, use_kg_context=True, max_context_triples=1)
    h("F", "retr02_ctx3", "Retrieval alpha=0.2 + 3 triple context",
      use_kg_retrieval=True, retrieval_alpha=0.2, use_kg_context=True, max_context_triples=3)
    h("F", "retr01_ctx1_h2", "Retrieval alpha=0.1 + 1 triple hop2 context",
      use_kg_retrieval=True, retrieval_alpha=0.1, use_kg_context=True,
      max_context_triples=1, context_hop_filter="hop2")
    h("F", "retr03_ctx2_hiconf", "Retrieval alpha=0.3 + 2 triple high-conf",
      use_kg_retrieval=True, retrieval_alpha=0.3, use_kg_context=True,
      max_context_triples=2, context_confidence_threshold=0.9)
    h("F", "retr05_ctx1_nat", "Retrieval alpha=0.5 + 1 natural triple",
      use_kg_retrieval=True, retrieval_alpha=0.5, use_kg_context=True,
      max_context_triples=1, context_format="natural")

    # F11-F15: Retrieval + Bridge combos
    h("F", "retr01_bridge", "Retrieval alpha=0.1 + bridge",
      use_kg_retrieval=True, retrieval_alpha=0.1, use_kg_bridging=True)
    h("F", "retr02_bridge", "Retrieval alpha=0.2 + bridge",
      use_kg_retrieval=True, retrieval_alpha=0.2, use_kg_bridging=True)
    h("F", "retr01_bridge_cond05", "Retrieval + conditional bridge (<0.5)",
      use_kg_retrieval=True, retrieval_alpha=0.1, use_kg_bridging=True, bridge_conditional_threshold=0.5)
    h("F", "retr03_bridge_b03", "Retrieval alpha=0.3 + bridge boost=0.3",
      use_kg_retrieval=True, retrieval_alpha=0.3, use_kg_bridging=True, bridge_boost=0.3)
    h("F", "retr01_bridge_nonn", "Retrieval + bridge no neighbors",
      use_kg_retrieval=True, retrieval_alpha=0.1, use_kg_bridging=True, bridge_include_neighbors=False)

    # F16-F20: All three KG features with tuning
    h("F", "full_minimal", "All KG features, minimal: alpha=0.1, boost=0.05, 1 triple",
      use_kg_retrieval=True, retrieval_alpha=0.1, use_kg_bridging=True, bridge_boost=0.05,
      use_kg_context=True, max_context_triples=1)
    h("F", "full_conservative", "All KG, conservative: alpha=0.1, boost=0.1, 2 triples conf>0.9",
      use_kg_retrieval=True, retrieval_alpha=0.1, use_kg_bridging=True, bridge_boost=0.1,
      use_kg_context=True, max_context_triples=2, context_confidence_threshold=0.9)
    h("F", "full_h2only", "All KG, context hop2-only: alpha=0.1, boost=0.15, 3 triples h2",
      use_kg_retrieval=True, retrieval_alpha=0.1, use_kg_bridging=True,
      use_kg_context=True, max_context_triples=3, context_hop_filter="hop2")
    h("F", "full_natural", "All KG, natural format: alpha=0.2, boost=0.1, 3 natural triples",
      use_kg_retrieval=True, retrieval_alpha=0.2, use_kg_bridging=True, bridge_boost=0.1,
      use_kg_context=True, max_context_triples=3, context_format="natural")
    h("F", "full_entonly", "All KG, entities-only: alpha=0.1, boost=0.1, 5 entities",
      use_kg_retrieval=True, retrieval_alpha=0.1, use_kg_bridging=True, bridge_boost=0.1,
      use_kg_context=True, max_context_triples=5, context_format="entities_only")

    # F21-F25: Pipeline param + KG combos
    h("F", "k5_bridge", "top_k=5 + bridge",
      top_k=5, use_kg_bridging=True)
    h("F", "k1_ctx1", "top_k=1 + 1 triple context",
      top_k=1, use_kg_context=True, max_context_triples=1)
    h("F", "k5_retr01", "top_k=5 + retrieval alpha=0.1",
      top_k=5, use_kg_retrieval=True, retrieval_alpha=0.1)
    h("F", "k2_bridge_ctx1", "top_k=2 + bridge + 1 triple",
      top_k=2, use_kg_bridging=True, use_kg_context=True, max_context_triples=1)
    h("F", "k1_t0_ctx1", "top_k=1, temp=0, 1 triple context",
      top_k=1, extraction_temperature=0.0, use_kg_context=True, max_context_triples=1)

    # F26-F30: Aggressive vs minimal combos
    h("F", "aggressive_all", "Aggressive: alpha=0.5, boost=0.5, 10 triples, k=5",
      use_kg_retrieval=True, retrieval_alpha=0.5, use_kg_bridging=True, bridge_boost=0.5,
      use_kg_context=True, max_context_triples=10, top_k=5)
    h("F", "whisper", "Whisper: alpha=0.01, boost=0.01, 1 triple conf>0.95",
      use_kg_retrieval=True, retrieval_alpha=0.01, use_kg_bridging=True, bridge_boost=0.01,
      use_kg_context=True, max_context_triples=1, context_confidence_threshold=0.95)
    h("F", "bridge_only_k5_b03", "Bridge-focused: k=5, boost=0.3, no context/retrieval",
      top_k=5, use_kg_bridging=True, bridge_boost=0.3)
    h("F", "ctx_only_strict", "Context-focused: strict prompt, 1 triple, conf>0.9",
      use_kg_context=True, prompt_key="strict", max_context_triples=1,
      context_confidence_threshold=0.9)
    h("F", "retr_only_binary05", "Retrieval-focused: binary score, alpha=0.5, k=5",
      use_kg_retrieval=True, retrieval_alpha=0.5, retrieval_score_function="binary", top_k=5)

    return hypotheses


# ══════════════════════════════════════════════════════════════════════
# Pipeline Runner (parameterized by HypothesisConfig)
# ══════════════════════════════════════════════════════════════════════

def format_context(paragraphs, indices, context_window="full"):
    """Format retrieved paragraphs according to context_window setting."""
    parts = []
    for idx in indices:
        p = paragraphs[idx]
        if context_window == "title_only":
            parts.append(f"[{p['title']}]")
        elif context_window == "text_only":
            parts.append(p["paragraph_text"])
        elif context_window == "truncated_200":
            text = p["paragraph_text"][:200]
            parts.append(f"[{p['title']}] {text}")
        else:  # "full"
            parts.append(f"[{p['title']}] {p['paragraph_text']}")
    return "\n\n".join(parts)


def format_kg_facts(triples: List[KGTriple], fmt: str = "structured") -> str:
    """Format KG triples according to the specified format."""
    if not triples:
        return ""

    if fmt == "structured":
        lines = ["Known facts:"]
        for t in triples:
            rel = REL_DISPLAY.get(t.relation, t.relation.replace("_", " "))
            lines.append(f"- {t.subject} ({rel}) {t.object}")
        return "\n".join(lines) + "\n\n"

    elif fmt == "natural":
        lines = ["Known facts:"]
        templates = {
            "spouse_of": "{s} is married to {o}",
            "child_of": "{s} is a child of {o}",
            "parent_of": "{s} is a parent of {o}",
            "employer_of": "{s} employs {o}",
            "headquartered_in": "{s} is headquartered in {o}",
            "founded_by": "{s} was founded by {o}",
            "owned_by": "{s} is owned by {o}",
            "performer_of": "{s} performed {o}",
            "author_of": "{s} wrote {o}",
            "director_of": "{s} directed {o}",
            "manufacturer_of": "{s} manufactured {o}",
            "located_in": "{s} is located in {o}",
            "borders": "{s} borders {o}",
            "capital_of": "{s} is the capital of {o}",
            "member_of": "{s} is a member of {o}",
        }
        for t in triples:
            tmpl = templates.get(t.relation, "{s} is related to {o}")
            lines.append(f"- {tmpl.format(s=t.subject, o=t.object)}")
        return "\n".join(lines) + "\n\n"

    elif fmt == "entities_only":
        entities = set()
        for t in triples:
            entities.add(t.subject)
            entities.add(t.object)
        return "Related entities: " + ", ".join(sorted(entities)) + "\n\n"

    elif fmt == "inline":
        facts = []
        for t in triples:
            rel = REL_DISPLAY.get(t.relation, t.relation.replace("_", " "))
            facts.append(f"{t.subject} ({rel}) {t.object}")
        return "Facts: " + "; ".join(facts) + "\n\n"

    elif fmt == "verbose":
        lines = ["The following structured knowledge may help answer the question:"]
        for t in triples:
            rel = REL_DISPLAY.get(t.relation, t.relation.replace("_", " "))
            lines.append(f"  - Entity \"{t.subject}\" (type: {t.subject_type}) has relationship "
                        f"\"{rel}\" with entity \"{t.object}\" (type: {t.object_type}) "
                        f"[confidence: {t.confidence:.0%}]")
        return "\n".join(lines) + "\n\n"

    return ""


def get_relevant_triples(kg, question, retrieved_indices, config):
    """Get KG triples relevant to the current question, filtered by config."""
    question_norm = normalize_answer(question)
    matched_entities = set()

    for entity_name in kg.entities:
        if entity_name in question_norm:
            matched_entities.add(entity_name)
        elif question_norm:
            nt, qt = set(entity_name.split()), set(question_norm.split())
            if nt and len(nt & qt) >= min(2, len(nt)):
                matched_entities.add(entity_name)

    # Also from retrieved paragraphs
    for idx in retrieved_indices:
        matched_entities.update(kg.paragraph_index.get(idx, set()))

    rel_filter = None
    if config.context_relation_filter:
        rel_filter = RELATION_SUBSETS.get(config.context_relation_filter)

    seen = set()
    triples = []
    for ent in matched_entities:
        for t in kg.get_triples_for_entity(ent, max_triples=3,
                                            confidence_threshold=config.context_confidence_threshold,
                                            relation_filter=rel_filter):
            key = (kg._normalize(t.subject), t.relation, kg._normalize(t.object))
            if key not in seen:
                seen.add(key)
                triples.append(t)

    triples.sort(key=lambda t: t.confidence, reverse=True)
    return triples[:config.max_context_triples]


def run_hypothesis(config: HypothesisConfig, sample, retriever, kg, auto_hops):
    """Run a single hypothesis configuration on one sample. Returns (prediction, hop_details)."""
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []

    prompt_template = PROMPTS.get(config.prompt_key, PROMPTS["8shot"])
    uses_kg_facts = "{kg_facts}" in prompt_template

    for i, hop in enumerate(auto_hops):
        hop_q = hop["question"]
        for j, ans in enumerate(previous_answers, 1):
            hop_q = hop_q.replace(f"#{j}", ans)
        if ">>" in hop_q:
            hop_q = format_hop_question(hop_q, [])

        # ── Retrieval ──
        # Determine if we should bridge this hop
        do_bridge = (config.use_kg_bridging and i > 0 and previous_answers)

        # Check conditional bridging
        if do_bridge and config.bridge_conditional_threshold is not None:
            # Get baseline embedding score
            base_results = retriever.retrieve(hop_q, paragraphs, top_k=1)
            best_embed = base_results[0][1] if base_results else 0
            if best_embed >= config.bridge_conditional_threshold:
                do_bridge = False

        if do_bridge:
            # Bridge retrieval
            bridge_entity = previous_answers[-1]
            if config.bridge_entity_processing == "normalized":
                bridge_entity = normalize_answer(bridge_entity)
            elif config.bridge_entity_processing == "first_token":
                bridge_entity = bridge_entity.split()[0] if bridge_entity.split() else bridge_entity

            retriever._load_model()
            query_text = f"Represent this sentence for searching relevant passages: {hop_q}"
            para_texts = [f"{p['title']} {p['paragraph_text']}" for p in paragraphs]
            query_emb = retriever.model.encode([query_text], normalize_embeddings=True)
            para_embs = retriever.model.encode(para_texts, normalize_embeddings=True)
            scores = np.dot(para_embs, query_emb.T).flatten()

            # Apply bridge boost
            norm = kg._normalize(bridge_entity)
            bridged = set()
            ent = kg.find_entity(bridge_entity) or kg.fuzzy_find(bridge_entity, config.bridge_fuzzy_threshold)
            if ent:
                bridged.update(kg.entity_paragraphs.get(ent, set()))
            if config.bridge_include_neighbors:
                search_name = ent if ent else norm
                for neighbor_name, edge in kg.get_neighbors(bridge_entity):
                    bridged.update(kg.entity_paragraphs.get(neighbor_name, set()))

            for idx in bridged:
                if idx < len(scores):
                    scores[idx] += config.bridge_boost

            top_indices = np.argsort(scores)[::-1][:config.top_k]
            retrieved_indices = [int(idx) for idx in top_indices]

        elif config.use_kg_retrieval:
            # KG-augmented retrieval
            retriever._load_model()
            query_text = f"Represent this sentence for searching relevant passages: {hop_q}"
            para_texts = [f"{p['title']} {p['paragraph_text']}" for p in paragraphs]
            query_emb = retriever.model.encode([query_text], normalize_embeddings=True)
            para_embs = retriever.model.encode(para_texts, normalize_embeddings=True)
            embed_scores = np.dot(para_embs, query_emb.T).flatten()

            # KG entity scores
            query_norm = normalize_answer(hop_q)
            query_entities = set()
            for ent_name in kg.entities:
                if ent_name in query_norm:
                    query_entities.add(ent_name)
                else:
                    et, qt = set(ent_name.split()), set(query_norm.split())
                    if et and len(et & qt) >= min(2, len(et)):
                        query_entities.add(ent_name)

            kg_scores = np.zeros(len(paragraphs))
            for idx in range(len(paragraphs)):
                para_ents = kg.paragraph_index.get(idx, set())
                if query_entities and para_ents:
                    if config.retrieval_score_function == "binary":
                        kg_scores[idx] = 1.0 if any(
                            qe in pe or pe in qe for qe in query_entities for pe in para_ents
                        ) else 0.0
                    elif config.retrieval_score_function == "confidence_weighted":
                        matching_triples = [e for e in kg.edges
                                          if e.source_paragraph_idx == idx
                                          and (kg._normalize(e.subject) in query_entities
                                               or kg._normalize(e.object) in query_entities)]
                        if matching_triples:
                            kg_scores[idx] = max(t.confidence for t in matching_triples)
                    else:  # fractional
                        matches = sum(1 for qe in query_entities
                                     if any(qe in pe or pe in qe for pe in para_ents))
                        kg_scores[idx] = matches / len(query_entities)

            combined = (1 - config.retrieval_alpha) * embed_scores + config.retrieval_alpha * kg_scores
            top_indices = np.argsort(combined)[::-1][:config.top_k]
            retrieved_indices = [int(idx) for idx in top_indices]
        else:
            # Standard embedding retrieval
            results = retriever.retrieve(hop_q, paragraphs, top_k=config.top_k)
            retrieved_indices = [idx for idx, _ in results]

        gold_idx = gold_decomp[i]["paragraph_support_idx"] if i < len(gold_decomp) else -1
        gold_retrieved = gold_idx in retrieved_indices

        # ── Context formatting ──
        context = format_context(paragraphs, retrieved_indices, config.context_window)

        # ── KG context enrichment ──
        kg_facts_str = ""
        should_add_context = config.use_kg_context
        if should_add_context and config.context_hop_filter == "hop1" and i > 0:
            should_add_context = False
        if should_add_context and config.context_hop_filter == "hop2" and i == 0:
            should_add_context = False

        if should_add_context:
            triples = get_relevant_triples(kg, hop_q, retrieved_indices, config)
            if triples:
                kg_facts_str = format_kg_facts(triples, config.context_format)

        # ── Build prompt ──
        if uses_kg_facts:
            prompt = prompt_template.format(
                context=context, question=hop_q,
                kg_facts=kg_facts_str if kg_facts_str else ""
            )
        elif kg_facts_str and config.use_kg_context:
            # Non-KG prompt template but we need to inject facts
            if config.context_prompt_position == "after_context":
                prompt = PROMPTS["8shot_kg_after"].format(
                    context=context, question=hop_q, kg_facts=kg_facts_str)
            elif config.context_prompt_position == "system":
                prompt = kg_facts_str + prompt_template.format(context=context, question=hop_q)
            else:  # before_context (default)
                prompt = PROMPTS["8shot_kg_before"].format(
                    context=context, question=hop_q, kg_facts=kg_facts_str)
        else:
            # No KG context or no triples found
            if uses_kg_facts:
                prompt = prompt_template.format(
                    context=context, question=hop_q, kg_facts="")
            else:
                prompt = prompt_template.format(context=context, question=hop_q)

        # ── Extract answer ──
        response = ask_model(
            prompt, model="qwen2.5:7b",
            temperature=config.extraction_temperature,
            max_tokens=config.extraction_max_tokens
        )

        if config.prompt_key == "cot":
            # Extract answer from CoT response
            lines = response.strip().split('\n')
            answer_line = ""
            for line in reversed(lines):
                if line.strip().lower().startswith("answer:"):
                    answer_line = line.strip()[7:].strip()
                    break
            answer = extract_short_answer(answer_line) if answer_line else extract_short_answer(lines[-1])
        else:
            answer = extract_short_answer(response)

        gold_answer = gold_decomp[i]["answer"] if i < len(gold_decomp) else "N/A"
        em = exact_match(answer, gold_answer) if gold_answer != "N/A" else None

        hop_details.append({
            "hop": i + 1, "question": hop_q, "gold_answer": gold_answer,
            "predicted": answer, "em": em, "gold_retrieved": gold_retrieved,
        })
        previous_answers.append(answer)

    final = answer if hop_details else ""
    return final, hop_details


# ══════════════════════════════════════════════════════════════════════
# Main Experiment Loop
# ══════════════════════════════════════════════════════════════════════

def run_battery(limit=30, cache_dir="kg_cache", output_path="results/v8b_hypothesis_battery.json",
                resume=True, verbose=False):
    from datasets import load_dataset

    print("Loading MuSiQue validation set...")
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    samples = [s for s in ds if s.get("answerable", True)][:limit]
    print(f"Testing on {len(samples)} answerable questions\n")

    retriever = EmbeddingRetriever()
    retriever._load_model()

    # ── Pre-compute decompositions ────────────────────────────────
    print("=" * 70)
    print("PHASE 1: DECOMPOSITION")
    print("=" * 70)

    all_auto_decomps = {}
    for i, sample in enumerate(samples):
        t0 = time.time()
        auto_sub_qs = decompose_with_qwen(sample["question"])
        all_auto_decomps[sample["id"]] = auto_sub_qs
        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] decomposed ({(time.time()-t0)*1000:.0f}ms)")
    print(f"  Decomposed {len(samples)} questions")

    # ── Build KGs ─────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("PHASE 2: KNOWLEDGE GRAPH EXTRACTION")
    print("=" * 70)

    kg_extractor = KGExtractor(cache_dir=cache_dir)
    all_kgs = {}
    for i, sample in enumerate(samples):
        all_kgs[sample["id"]] = kg_extractor.build_graph(sample["paragraphs"])
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] KG built")
    kg_extractor.save_cache()
    print(f"  Built {len(all_kgs)} knowledge graphs ({len(kg_extractor.cache)} cached paragraphs)")

    # ── Generate hypotheses ───────────────────────────────────────
    hypotheses = generate_hypotheses()
    print(f"\n{'=' * 70}")
    print(f"PHASE 3: TESTING {len(hypotheses)} HYPOTHESES")
    print("=" * 70)

    # Load previous results for resume
    output = Path(output_path)
    existing_results = {}
    if resume and output.exists():
        with open(output) as f:
            data = json.load(f)
        existing_results = data.get("hypotheses", {})
        print(f"  Resuming: {len(existing_results)} hypotheses already completed")

    # Run baseline first
    baseline_config = HypothesisConfig(id="B000", name="baseline", description="No KG (v5 reference)", category="X")
    baseline_key = "B000_baseline"

    if baseline_key not in existing_results:
        print(f"\n  Running BASELINE...")
        baseline_result = _test_one(baseline_config, samples, retriever, all_kgs, all_auto_decomps)
        existing_results[baseline_key] = baseline_result
        _save_incremental(output, existing_results, hypotheses, samples)
        print(f"  BASELINE: EM={baseline_result['em']:.1f}%  relEM={baseline_result['relaxed_em']:.1f}%  "
              f"F1={baseline_result['f1']:.3f}")
    else:
        baseline_result = existing_results[baseline_key]
        print(f"  BASELINE (cached): EM={baseline_result['em']:.1f}%")

    baseline_em = baseline_result["em"]

    # Run all hypotheses
    completed = 0
    improved = 0
    hurt = 0
    best_delta = 0
    best_name = ""

    for idx, hyp in enumerate(hypotheses):
        key = f"{hyp.id}_{hyp.name}"

        if key in existing_results:
            result = existing_results[key]
            delta = result["em"] - baseline_em
            if delta > 0: improved += 1
            elif delta < 0: hurt += 1
            if delta > best_delta:
                best_delta = delta
                best_name = key
            completed += 1
            continue

        t0 = time.time()
        try:
            result = _test_one(hyp, samples, retriever, all_kgs, all_auto_decomps)
        except Exception as e:
            result = {"em": 0, "relaxed_em": 0, "f1": 0, "latency_ms": 0,
                     "error": str(e), "per_question": []}
            print(f"  [{idx+1}/{len(hypotheses)}] ERROR {hyp.id} {hyp.name}: {e}")

        elapsed = time.time() - t0
        existing_results[key] = result
        completed += 1

        delta = result["em"] - baseline_em
        if delta > 0: improved += 1
        elif delta < 0: hurt += 1
        if delta > best_delta:
            best_delta = delta
            best_name = key

        marker = "+" if delta > 0 else ("-" if delta < 0 else "=")
        print(f"  [{marker}] [{completed}/{len(hypotheses)}] {hyp.id} {hyp.name:30s}: "
              f"EM={result['em']:5.1f}% ({delta:+.1f}%)  [{elapsed:.0f}s]  "
              f"[best: {best_name} {best_delta:+.1f}%]")

        # Incremental save every 5 hypotheses
        if completed % 5 == 0:
            _save_incremental(output, existing_results, hypotheses, samples)

    # Final save
    _save_incremental(output, existing_results, hypotheses, samples)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"FINAL SUMMARY: {len(hypotheses)} HYPOTHESES TESTED")
    print(f"{'=' * 70}")
    print(f"Baseline EM: {baseline_em:.1f}%")
    print(f"Improved:    {improved}/{len(hypotheses)} ({improved/len(hypotheses)*100:.1f}%)")
    print(f"Neutral:     {len(hypotheses)-improved-hurt}/{len(hypotheses)}")
    print(f"Hurt:        {hurt}/{len(hypotheses)} ({hurt/len(hypotheses)*100:.1f}%)")

    # Top 10
    sorted_results = sorted(
        [(k, v) for k, v in existing_results.items() if k != baseline_key],
        key=lambda x: x[1]["em"], reverse=True
    )

    print(f"\n  ── Top 10 Hypotheses ──")
    for k, v in sorted_results[:10]:
        delta = v["em"] - baseline_em
        print(f"  {delta:+5.1f}%  EM={v['em']:5.1f}%  {k}")

    print(f"\n  ── Bottom 10 Hypotheses ──")
    for k, v in sorted_results[-10:]:
        delta = v["em"] - baseline_em
        print(f"  {delta:+5.1f}%  EM={v['em']:5.1f}%  {k}")

    # By category
    print(f"\n  ── By Category ──")
    for cat in ["A", "B", "C", "D", "E", "F"]:
        cat_results = [(k, v) for k, v in existing_results.items()
                       if k.startswith(cat) and k != baseline_key]
        if cat_results:
            ems = [v["em"] for _, v in cat_results]
            avg_em = sum(ems) / len(ems)
            best = max(cat_results, key=lambda x: x[1]["em"])
            cat_names = {"A": "Context", "B": "Bridging", "C": "Retrieval",
                        "D": "Pipeline", "E": "KG Prompts", "F": "Combos"}
            print(f"  {cat} ({cat_names.get(cat, cat):10s}): n={len(cat_results):3d}, "
                  f"avg={avg_em:5.1f}%, best={best[1]['em']:5.1f}% ({best[0]})")


def _test_one(config, samples, retriever, all_kgs, all_auto_decomps):
    """Test a single hypothesis on all samples."""
    em_scores, rem_scores, f1_scores, latencies = [], [], [], []
    per_question = []

    for sample in samples:
        auto_sub_qs = all_auto_decomps[sample["id"]]
        auto_hops = [{"question": sq} for sq in auto_sub_qs]
        kg = all_kgs[sample["id"]]

        # Build KG with appropriate confidence threshold
        if config.kg_confidence_threshold > 0:
            # Rebuild with higher threshold
            filtered_kg = KnowledgeGraph()
            for edge in kg.edges:
                if edge.confidence >= config.kg_confidence_threshold:
                    filtered_kg.add_triple(edge)
            use_kg = filtered_kg
        else:
            use_kg = kg

        t0 = time.time()
        pred, hops = run_hypothesis(config, sample, retriever, use_kg, auto_hops)
        lat = (time.time() - t0) * 1000

        answer = sample["answer"]
        aliases = sample.get("answer_aliases", [])
        em = exact_match(pred, answer, aliases)
        rem = relaxed_match(pred, answer, aliases)
        f1 = best_f1(pred, answer, aliases)

        em_scores.append(em)
        rem_scores.append(rem)
        f1_scores.append(f1)
        latencies.append(lat)
        per_question.append({
            "id": sample["id"], "prediction": pred, "answer": answer,
            "em": em, "relaxed_em": rem, "f1": f1,
        })

    n = len(samples)
    return {
        "em": sum(em_scores) / n * 100,
        "relaxed_em": sum(rem_scores) / n * 100,
        "f1": sum(f1_scores) / n,
        "latency_ms": sum(latencies) / n,
        "per_question": per_question,
    }


def _save_incremental(output_path, results, hypotheses, samples):
    """Save results incrementally."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "metadata": {
            "n_hypotheses": len(hypotheses),
            "n_completed": len(results),
            "n_samples": len(samples),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "hypotheses": results,
    }

    with open(output, "w") as f:
        json.dump(data, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="v8b: 250-Hypothesis KG Battery")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", type=str, default="results/v8b_hypothesis_battery.json")
    parser.add_argument("--cache-dir", type=str, default="kg_cache")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Verify Ollama
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        assert any("qwen2.5:7b" in m for m in models), "qwen2.5:7b not found"
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    start = time.time()
    run_battery(
        limit=args.limit,
        cache_dir=args.cache_dir,
        output_path=args.output,
        resume=not args.no_resume,
        verbose=args.verbose,
    )
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed/3600:.1f} hours ({elapsed/60:.0f} minutes)")


if __name__ == "__main__":
    main()
