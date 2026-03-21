#!/usr/bin/env python3
"""
Iterative Retrieval v8 - Knowledge Graph Integration Experiment

THE QUESTION: Can a Knowledge Graph built from extracted triples improve
multi-hop QA beyond embedding-only retrieval?

Current pipeline (v5): Decompose → Retrieve (BGE top-3) → Extract (Qwen 7B) → Chain
Best result: 60.0% EM on MuSiQue 2-hop (n=30, fully autonomous).

The bottleneck is NOT retrieval (96% gold paragraph recall). It's extraction
quality — the model has the right paragraph but picks the wrong answer.
The KG must help extraction, not just retrieval.

Three integration points tested:
  1. KG Context Enrichment:   Add structured triples to extraction prompt (+3-7%)
  2. KG Hop Bridging:         Use KG edges to find paragraphs for hop 2 (+2-5%)
  3. KG Retrieval Augmentation: Boost embedding scores with entity matches (+0-2%)

7 configs tested (ablation):
  0. baseline_auto:      No KG (v5 reference, expected ~60.0%)
  1. kg_context_only:    Context enrichment only
  2. kg_bridging_only:   Hop bridging only
  3. kg_retrieval_only:  Retrieval augmentation only
  4. kg_bridge_context:  Bridging + context
  5. kg_full:            All three KG features
  6. kg_full_gold_decomp: All three + gold decomposition (ceiling)
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

# Context-enriched prompt: structured KG facts injected before the paragraph
PROMPT_8SHOT_KG = """Answer the question using the known facts and context below. Give ONLY the specific name, place, or fact. One or two words maximum.

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


# ══════════════════════════════════════════════════════════════════════
# Section 2: KG Triple Extraction & Graph
# ══════════════════════════════════════════════════════════════════════

ENTITY_TYPES = {"PERSON", "PLACE", "ORGANIZATION", "WORK", "EVENT"}

CORE_RELATIONS = {
    "spouse_of", "child_of", "parent_of", "employer_of", "headquartered_in",
    "founded_by", "owned_by", "performer_of", "author_of", "director_of",
    "manufacturer_of", "located_in", "borders", "capital_of", "member_of",
}

def _build_extraction_prompt(title: str, text: str) -> str:
    """Build KG extraction prompt. Separate function to avoid .format() escaping issues with JSON."""
    return (
        'Extract structured facts (triples) from this paragraph. For each fact, provide:\n'
        '- subject: entity name\n'
        '- subject_type: PERSON, PLACE, ORGANIZATION, WORK, or EVENT\n'
        '- relation: one of [spouse_of, child_of, parent_of, employer_of, headquartered_in, '
        'founded_by, owned_by, performer_of, author_of, director_of, manufacturer_of, '
        'located_in, borders, capital_of, member_of]\n'
        '- object: entity name\n'
        '- object_type: PERSON, PLACE, ORGANIZATION, WORK, or EVENT\n'
        '- confidence: 0.0 to 1.0\n'
        '\n'
        'Output JSON array only. Max 10 triples. Only extract facts explicitly stated.\n'
        '\n'
        'Example input: "[Steve Hillage] Green is the fourth studio album by British progressive rock musician Steve Hillage."\n'
        'Example output: [{"subject": "Steve Hillage", "subject_type": "PERSON", "relation": "performer_of", "object": "Green", "object_type": "WORK", "confidence": 0.95}]\n'
        '\n'
        'Example input: "[Orion Pictures] The film was distributed by Orion Pictures, founded by Mike Medavoy and four other executives."\n'
        'Example output: [{"subject": "Orion Pictures", "subject_type": "ORGANIZATION", "relation": "founded_by", "object": "Mike Medavoy", "object_type": "PERSON", "confidence": 0.9}]\n'
        '\n'
        'Example input: "[Canyon, Texas] Canyon is a city in and the county seat of Randall County, Texas, United States."\n'
        'Example output: [{"subject": "Canyon", "subject_type": "PLACE", "relation": "located_in", "object": "Randall County", "object_type": "PLACE", "confidence": 0.95}, '
        '{"subject": "Canyon", "subject_type": "PLACE", "relation": "located_in", "object": "Texas", "object_type": "PLACE", "confidence": 0.9}]\n'
        '\n'
        'Example input: "[Miquette Giraudy] Miquette Giraudy is a keyboard player, best known for her work with her partner Steve Hillage."\n'
        'Example output: [{"subject": "Miquette Giraudy", "subject_type": "PERSON", "relation": "spouse_of", "object": "Steve Hillage", "object_type": "PERSON", "confidence": 0.85}]\n'
        '\n'
        f'Paragraph: [{title}] {text}\n'
        'Output:'
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

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@dataclass
class Entity:
    name: str
    entity_type: str
    aliases: Set[str] = field(default_factory=set)
    paragraph_indices: Set[int] = field(default_factory=set)


class KnowledgeGraph:
    """In-memory graph built from extracted triples."""

    def __init__(self):
        self.entities: Dict[str, Entity] = {}       # normalized_name → Entity
        self.edges: List[KGTriple] = []
        self.paragraph_index: Dict[int, Set[str]] = defaultdict(set)  # para_idx → entity names
        self.entity_paragraphs: Dict[str, Set[int]] = defaultdict(set)  # entity → para_idxs

    def _normalize(self, name: str) -> str:
        return normalize_answer(name)

    def add_triple(self, triple: KGTriple):
        self.edges.append(triple)

        # Register subject
        subj_norm = self._normalize(triple.subject)
        if subj_norm not in self.entities:
            self.entities[subj_norm] = Entity(
                name=triple.subject, entity_type=triple.subject_type
            )
        ent = self.entities[subj_norm]
        ent.aliases.add(triple.subject)
        ent.paragraph_indices.add(triple.source_paragraph_idx)
        self.paragraph_index[triple.source_paragraph_idx].add(subj_norm)
        self.entity_paragraphs[subj_norm].add(triple.source_paragraph_idx)

        # Register object
        obj_norm = self._normalize(triple.object)
        if obj_norm not in self.entities:
            self.entities[obj_norm] = Entity(
                name=triple.object, entity_type=triple.object_type
            )
        ent = self.entities[obj_norm]
        ent.aliases.add(triple.object)
        ent.paragraph_indices.add(triple.source_paragraph_idx)
        self.paragraph_index[triple.source_paragraph_idx].add(obj_norm)
        self.entity_paragraphs[obj_norm].add(triple.source_paragraph_idx)

    def find_entity(self, name: str) -> Optional[Entity]:
        norm = self._normalize(name)
        return self.entities.get(norm)

    def fuzzy_find(self, name: str, threshold: float = 0.6) -> Optional[Entity]:
        """Find entity by normalized substring match."""
        norm = self._normalize(name)
        if norm in self.entities:
            return self.entities[norm]

        # Try substring matching
        best_match = None
        best_score = 0.0
        for key, ent in self.entities.items():
            if not key or not norm:
                continue
            # Bidirectional substring
            if norm in key or key in norm:
                overlap = min(len(norm), len(key)) / max(len(norm), len(key))
                if overlap > best_score and overlap >= threshold:
                    best_score = overlap
                    best_match = ent
            else:
                # Token overlap (like relaxed_match)
                norm_tokens = set(norm.split())
                key_tokens = set(key.split())
                if norm_tokens and key_tokens:
                    overlap = len(norm_tokens & key_tokens) / max(len(norm_tokens), len(key_tokens))
                    if overlap > best_score and overlap >= threshold:
                        best_score = overlap
                        best_match = ent
        return best_match

    def get_neighbors(self, entity_name: str) -> List[Tuple[str, KGTriple]]:
        """Get all entities connected to this one with their edge."""
        norm = self._normalize(entity_name)
        neighbors = []
        for edge in self.edges:
            subj_norm = self._normalize(edge.subject)
            obj_norm = self._normalize(edge.object)
            if subj_norm == norm:
                neighbors.append((obj_norm, edge))
            elif obj_norm == norm:
                neighbors.append((subj_norm, edge))
        return neighbors

    def get_triples_for_entity(self, name: str, max_triples: int = 5) -> List[KGTriple]:
        """Get triples involving this entity, sorted by confidence."""
        norm = self._normalize(name)
        triples = []
        for edge in self.edges:
            if self._normalize(edge.subject) == norm or self._normalize(edge.object) == norm:
                triples.append(edge)
        triples.sort(key=lambda t: t.confidence, reverse=True)
        return triples[:max_triples]

    def get_paragraphs_for_entity(self, name: str) -> Set[int]:
        """Get all paragraph indices where this entity appears."""
        norm = self._normalize(name)
        return self.entity_paragraphs.get(norm, set())

    def stats(self) -> dict:
        return {
            "entities": len(self.entities),
            "edges": len(self.edges),
            "paragraphs_covered": len(self.paragraph_index),
        }


class KGExtractor:
    """LLM-based triple extraction with disk caching."""

    def __init__(self, cache_dir: str = "kg_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, List[dict]] = {}
        self._load_cache()
        self.stats = {"extracted": 0, "cached": 0, "failed": 0}

    def _cache_path(self) -> Path:
        return self.cache_dir / "paragraph_triples.json"

    def _load_cache(self):
        path = self._cache_path()
        if path.exists():
            with open(path) as f:
                self.cache = json.load(f)
            print(f"  KG cache loaded: {len(self.cache)} paragraphs cached")

    def save_cache(self):
        with open(self._cache_path(), "w") as f:
            json.dump(self.cache, f, indent=2)

    @staticmethod
    def _hash_paragraph(title: str, text: str) -> str:
        content = f"{title}|||{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def extract(self, title: str, text: str, paragraph_idx: int) -> List[KGTriple]:
        """Extract triples from a paragraph, using cache if available."""
        content_hash = self._hash_paragraph(title, text)

        if content_hash in self.cache:
            self.stats["cached"] += 1
            raw_triples = self.cache[content_hash]
            return [KGTriple(
                subject=t["subject"],
                subject_type=t.get("subject_type", "PERSON"),
                relation=t["relation"],
                object=t["object"],
                object_type=t.get("object_type", "PERSON"),
                source_paragraph_idx=paragraph_idx,
                confidence=t.get("confidence", 0.8),
            ) for t in raw_triples if t.get("confidence", 0.8) >= 0.7]

        # Call LLM for extraction
        prompt = _build_extraction_prompt(title, text)
        try:
            response = ask_model(prompt, model="qwen2.5:7b", temperature=0.1, max_tokens=512)
            triples_raw = self._parse_response(response)
            self.cache[content_hash] = triples_raw
            self.stats["extracted"] += 1

            return [KGTriple(
                subject=t["subject"],
                subject_type=t.get("subject_type", "PERSON"),
                relation=t["relation"],
                object=t["object"],
                object_type=t.get("object_type", "PERSON"),
                source_paragraph_idx=paragraph_idx,
                confidence=t.get("confidence", 0.8),
            ) for t in triples_raw if t.get("confidence", 0.8) >= 0.7]
        except Exception as e:
            self.stats["failed"] += 1
            self.cache[content_hash] = []
            return []

    def _parse_response(self, response: str) -> List[dict]:
        """Parse LLM JSON response into list of triple dicts."""
        # Try to find JSON array in the response
        response = response.strip()

        # Try direct parse
        try:
            data = json.loads(response)
            if isinstance(data, list):
                return self._validate_triples(data)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, list):
                    return self._validate_triples(data)
            except json.JSONDecodeError:
                pass

        # Try finding array boundaries
        start = response.find('[')
        end = response.rfind(']')
        if start >= 0 and end > start:
            try:
                data = json.loads(response[start:end + 1])
                if isinstance(data, list):
                    return self._validate_triples(data)
            except json.JSONDecodeError:
                pass

        return []

    def _validate_triples(self, data: list) -> List[dict]:
        """Filter to valid triples only."""
        valid = []
        for item in data[:10]:  # Max 10 per paragraph
            if not isinstance(item, dict):
                continue
            if not all(k in item for k in ("subject", "relation", "object")):
                continue
            # Normalize relation to our set
            rel = item["relation"].lower().strip().replace(" ", "_")
            if rel not in CORE_RELATIONS:
                # Try to map common variations
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
                    "capital": "capital_of",
                    "member": "member_of", "part_of": "member_of",
                }
                rel = rel_map.get(rel, rel)
            if rel not in CORE_RELATIONS:
                continue

            # Normalize entity types
            subj_type = item.get("subject_type", "PERSON").upper()
            obj_type = item.get("object_type", "PERSON").upper()
            if subj_type not in ENTITY_TYPES:
                subj_type = "PERSON"
            if obj_type not in ENTITY_TYPES:
                obj_type = "PERSON"

            valid.append({
                "subject": str(item["subject"]).strip(),
                "subject_type": subj_type,
                "relation": rel,
                "object": str(item["object"]).strip(),
                "object_type": obj_type,
                "confidence": float(item.get("confidence", 0.8)),
            })
        return valid

    def build_graph(self, paragraphs: list) -> KnowledgeGraph:
        """Build KG from a list of paragraphs (each has 'title' and 'paragraph_text')."""
        kg = KnowledgeGraph()
        for idx, para in enumerate(paragraphs):
            triples = self.extract(para["title"], para["paragraph_text"], idx)
            for triple in triples:
                kg.add_triple(triple)
        return kg


# ══════════════════════════════════════════════════════════════════════
# Section 3: KG Retrieval Augmentation
# ══════════════════════════════════════════════════════════════════════

class KGRetriever(EmbeddingRetriever):
    """Extends embedding retrieval with KG entity matching boost."""

    def __init__(self, kg: KnowledgeGraph, alpha: float = 0.3):
        super().__init__()
        self.kg = kg
        self.alpha = alpha

    def retrieve(self, query, paragraphs, top_k=3):
        """Retrieve with KG entity matching boost."""
        self._load_model()

        # Standard embedding scores
        query_text = f"Represent this sentence for searching relevant passages: {query}"
        para_texts = [f"{p['title']} {p['paragraph_text']}" for p in paragraphs]
        query_emb = self.model.encode([query_text], normalize_embeddings=True)
        para_embs = self.model.encode(para_texts, normalize_embeddings=True)
        embed_scores = np.dot(para_embs, query_emb.T).flatten()

        # KG entity matching scores
        query_entities = self._extract_query_entities(query)
        kg_scores = np.zeros(len(paragraphs))

        for idx in range(len(paragraphs)):
            para_entities = self.kg.paragraph_index.get(idx, set())
            if query_entities and para_entities:
                # Fraction of query entities found in paragraph's KG entities
                matches = sum(1 for qe in query_entities
                              if any(qe in pe or pe in qe for pe in para_entities))
                kg_scores[idx] = matches / len(query_entities) if query_entities else 0

        # Combine: embedding dominates, KG is tiebreaker
        combined = (1 - self.alpha) * embed_scores + self.alpha * kg_scores

        top_indices = np.argsort(combined)[::-1][:top_k]
        return [(int(idx), float(combined[idx])) for idx in top_indices]

    def _extract_query_entities(self, query: str) -> Set[str]:
        """Extract entity-like terms from a query using simple heuristics."""
        entities = set()
        # Use the KG's own entities as a lookup
        query_norm = normalize_answer(query)
        for entity_name in self.kg.entities:
            if entity_name in query_norm or query_norm in entity_name:
                entities.add(entity_name)
            else:
                # Token overlap check
                ent_tokens = set(entity_name.split())
                query_tokens = set(query_norm.split())
                if ent_tokens and len(ent_tokens & query_tokens) >= min(2, len(ent_tokens)):
                    entities.add(entity_name)
        return entities


# ══════════════════════════════════════════════════════════════════════
# Section 4: KG Hop Bridging
# ══════════════════════════════════════════════════════════════════════

class KGBridger:
    """Uses KG edges to find paragraphs connected to hop 1 answer for hop 2."""

    def __init__(self, kg: KnowledgeGraph, boost: float = 0.15):
        self.kg = kg
        self.boost = boost

    def get_bridged_paragraphs(self, entity_name: str) -> Set[int]:
        """Find paragraph indices connected to an entity via KG edges."""
        bridged = set()

        # Direct lookup
        ent = self.kg.find_entity(entity_name)
        if ent:
            bridged.update(ent.paragraph_indices)

        # Fuzzy lookup
        if not ent:
            ent = self.kg.fuzzy_find(entity_name)
            if ent:
                bridged.update(ent.paragraph_indices)

        # Also get paragraphs from neighbors (1-hop in KG)
        norm = self.kg._normalize(entity_name)
        for neighbor_name, edge in self.kg.get_neighbors(entity_name):
            neighbor_ent = self.kg.entities.get(neighbor_name)
            if neighbor_ent:
                bridged.update(neighbor_ent.paragraph_indices)

        return bridged

    def bridge_retrieve(self, query: str, paragraphs: list, retriever: EmbeddingRetriever,
                        bridge_entity: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """Retrieve with bridging boost for paragraphs connected to bridge_entity."""
        retriever._load_model()

        # Standard embedding scores
        query_text = f"Represent this sentence for searching relevant passages: {query}"
        para_texts = [f"{p['title']} {p['paragraph_text']}" for p in paragraphs]
        query_emb = retriever.model.encode([query_text], normalize_embeddings=True)
        para_embs = retriever.model.encode(para_texts, normalize_embeddings=True)
        embed_scores = np.dot(para_embs, query_emb.T).flatten()

        # Bridge boost
        bridged_paras = self.get_bridged_paragraphs(bridge_entity)
        for idx in bridged_paras:
            if idx < len(embed_scores):
                embed_scores[idx] += self.boost

        top_indices = np.argsort(embed_scores)[::-1][:top_k]
        return [(int(idx), float(embed_scores[idx])) for idx in top_indices]


# ══════════════════════════════════════════════════════════════════════
# Section 5: KG Context Enrichment
# ══════════════════════════════════════════════════════════════════════

class KGContextEnricher:
    """Adds relevant KG triples to extraction prompts as structured facts."""

    def __init__(self, kg: KnowledgeGraph, max_triples: int = 5):
        self.kg = kg
        self.max_triples = max_triples

    def get_relevant_triples(self, question: str, context_entities: Optional[Set[str]] = None) -> List[KGTriple]:
        """Find triples relevant to the current question."""
        relevant = []

        # Extract entities mentioned in the question
        question_norm = normalize_answer(question)
        matched_entities = set()

        for entity_name in self.kg.entities:
            if entity_name in question_norm:
                matched_entities.add(entity_name)
            elif question_norm:
                # Check token overlap
                ent_tokens = set(entity_name.split())
                q_tokens = set(question_norm.split())
                if ent_tokens and len(ent_tokens & q_tokens) >= min(2, len(ent_tokens)):
                    matched_entities.add(entity_name)

        # Also include entities from context if provided
        if context_entities:
            matched_entities.update(context_entities)

        # Get triples for matched entities
        seen = set()
        for entity_name in matched_entities:
            for triple in self.kg.get_triples_for_entity(entity_name, max_triples=3):
                key = (self.kg._normalize(triple.subject), triple.relation,
                       self.kg._normalize(triple.object))
                if key not in seen:
                    seen.add(key)
                    relevant.append(triple)

        # Sort by confidence, limit
        relevant.sort(key=lambda t: t.confidence, reverse=True)
        return relevant[:self.max_triples]

    def format_facts(self, triples: List[KGTriple]) -> str:
        """Format triples as human-readable facts for prompt injection."""
        if not triples:
            return ""

        # Convert relation names to readable format
        rel_display = {
            "spouse_of": "spouse of",
            "child_of": "child of",
            "parent_of": "parent of",
            "employer_of": "employer of",
            "headquartered_in": "headquartered in",
            "founded_by": "founded by",
            "owned_by": "owned by",
            "performer_of": "performer of",
            "author_of": "author of",
            "director_of": "director of",
            "manufacturer_of": "manufacturer of",
            "located_in": "located in",
            "borders": "borders",
            "capital_of": "capital of",
            "member_of": "member of",
        }

        lines = ["Known facts:"]
        for triple in triples:
            rel = rel_display.get(triple.relation, triple.relation.replace("_", " "))
            lines.append(f"- {triple.subject} ({rel}) {triple.object}")
        lines.append("")  # blank line before context
        return "\n".join(lines) + "\n"


# ══════════════════════════════════════════════════════════════════════
# Section 6: Pipeline Runner (modified from v5 with KG hooks)
# ══════════════════════════════════════════════════════════════════════

def run_pipeline(sample, retriever, decomposition, model="qwen2.5:7b",
                 top_k=3, use_gold_context=False, verbose=False,
                 kg=None, use_kg_retrieval=False, use_kg_bridging=False,
                 use_kg_context=False, kg_retriever=None, kg_bridger=None,
                 kg_enricher=None):
    """
    Run the iterative retrieval pipeline with optional KG integration.

    KG features (each independently toggleable):
      - use_kg_retrieval: Boost embedding scores with KG entity matches
      - use_kg_bridging: Use KG edges to boost hop 2 paragraph retrieval
      - use_kg_context: Add relevant KG triples to extraction prompt
    """
    paragraphs = sample["paragraphs"]
    gold_decomp = sample.get("question_decomposition", [])
    previous_answers = []
    hop_details = []
    kg_stats = {"bridge_hits": 0, "context_triples_added": 0, "retrieval_boosted": 0}

    for i, hop in enumerate(decomposition):
        hop_q = hop["question"]
        for j, ans in enumerate(previous_answers, 1):
            hop_q = hop_q.replace(f"#{j}", ans)

        if ">>" in hop_q:
            hop_q = format_hop_question(hop_q, [])

        # ── Retrieve context ──────────────────────────────────────
        if use_gold_context and i < len(gold_decomp):
            gold_idx = gold_decomp[i]["paragraph_support_idx"]
            p = paragraphs[gold_idx]
            context = f"[{p['title']}] {p['paragraph_text']}"
            retrieved_indices = [gold_idx]
        elif use_kg_bridging and i > 0 and previous_answers and kg_bridger:
            # Hop 2+: use KG bridging with previous answer as bridge entity
            bridge_entity = previous_answers[-1]
            results = kg_bridger.bridge_retrieve(
                hop_q, paragraphs, retriever, bridge_entity, top_k=top_k
            )
            retrieved_indices = [idx for idx, _ in results]

            # Track if bridging found a paragraph that embedding alone wouldn't
            embed_only = retriever.retrieve(hop_q, paragraphs, top_k=top_k)
            embed_indices = {idx for idx, _ in embed_only}
            bridged_paras = kg_bridger.get_bridged_paragraphs(bridge_entity)
            new_from_bridge = set(retrieved_indices) - embed_indices
            if new_from_bridge & bridged_paras:
                kg_stats["bridge_hits"] += 1

            context_parts = [f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
                           for idx, _ in results]
            context = "\n\n".join(context_parts)
        elif use_kg_retrieval and kg_retriever:
            # KG-augmented retrieval
            results = kg_retriever.retrieve(hop_q, paragraphs, top_k=top_k)
            retrieved_indices = [idx for idx, _ in results]
            kg_stats["retrieval_boosted"] += 1
            context_parts = [f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
                           for idx, _ in results]
            context = "\n\n".join(context_parts)
        else:
            results = retriever.retrieve(hop_q, paragraphs, top_k=top_k)
            retrieved_indices = [idx for idx, _ in results]
            context_parts = [f"[{paragraphs[idx]['title']}] {paragraphs[idx]['paragraph_text']}"
                           for idx, _ in results]
            context = "\n\n".join(context_parts)

        gold_idx = gold_decomp[i]["paragraph_support_idx"] if i < len(gold_decomp) else -1
        gold_retrieved = gold_idx in retrieved_indices

        # ── Extract answer (with optional KG context enrichment) ──
        if use_kg_context and kg_enricher and kg:
            # Get entities from retrieved paragraphs for more targeted lookup
            context_entities = set()
            for idx in retrieved_indices:
                context_entities.update(kg.paragraph_index.get(idx, set()))

            triples = kg_enricher.get_relevant_triples(hop_q, context_entities)
            kg_facts = kg_enricher.format_facts(triples)

            if triples:
                kg_stats["context_triples_added"] += len(triples)
                prompt = PROMPT_8SHOT_KG.format(
                    kg_facts=kg_facts, context=context, question=hop_q
                )
            else:
                # No relevant triples found — fall back to standard prompt
                prompt = PROMPT_8SHOT.format(context=context, question=hop_q)
        else:
            prompt = PROMPT_8SHOT.format(context=context, question=hop_q)

        response = ask_model(prompt, model=model, temperature=0.1, max_tokens=32)
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
    return final, hop_details, kg_stats


# ══════════════════════════════════════════════════════════════════════
# Section 7: Experiment Runner with All 7 Configs
# ══════════════════════════════════════════════════════════════════════

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
    kg_stats: dict = field(default_factory=lambda: {"bridge_hits": 0, "context_triples_added": 0, "retrieval_boosted": 0})

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


def run_experiment(limit=30, cache_dir="kg_cache", verbose=True, config_filter=None):
    from datasets import load_dataset

    print("Loading MuSiQue validation set...")
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    samples = [s for s in ds if s.get("answerable", True)][:limit]
    print(f"Testing on {len(samples)} answerable questions\n")

    retriever = EmbeddingRetriever()
    retriever._load_model()

    # ── Phase 1: Decompose all questions with Qwen ────────────────
    print("=" * 70)
    print("PHASE 1: DECOMPOSITION (Qwen 7B)")
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

    # ── Phase 2: Build KG for each question ───────────────────────
    print(f"\n{'=' * 70}")
    print("PHASE 2: KNOWLEDGE GRAPH EXTRACTION")
    print("=" * 70)

    kg_extractor = KGExtractor(cache_dir=cache_dir)
    all_kgs = {}  # sample_id → KnowledgeGraph

    for i, sample in enumerate(samples):
        t0 = time.time()
        kg = kg_extractor.build_graph(sample["paragraphs"])
        dt = (time.time() - t0) * 1000
        all_kgs[sample["id"]] = kg

        if verbose and (i + 1) % 10 == 0:
            stats = kg.stats()
            print(f"  [{i+1:3d}/{len(samples)}] KG: {stats['entities']} entities, "
                  f"{stats['edges']} edges ({dt:.0f}ms)")

    # Save cache after extraction
    kg_extractor.save_cache()
    print(f"\n  KG extraction stats: {kg_extractor.stats}")
    print(f"  Cache saved: {len(kg_extractor.cache)} paragraphs")

    # ── Phase 3: Run All Configs ──────────────────────────────────
    print(f"\n{'=' * 70}")
    print("PHASE 3: PIPELINE EVALUATION (7 configs)")
    print("=" * 70)

    # Config definitions
    config_defs = {
        "baseline_auto": {
            "desc": "No KG (v5 reference ~60%)",
            "decomp": "auto", "gold_ctx": False,
            "kg_retrieval": False, "kg_bridging": False, "kg_context": False,
        },
        "kg_context_only": {
            "desc": "KG context enrichment only",
            "decomp": "auto", "gold_ctx": False,
            "kg_retrieval": False, "kg_bridging": False, "kg_context": True,
        },
        "kg_bridging_only": {
            "desc": "KG hop bridging only",
            "decomp": "auto", "gold_ctx": False,
            "kg_retrieval": False, "kg_bridging": True, "kg_context": False,
        },
        "kg_retrieval_only": {
            "desc": "KG retrieval augmentation only",
            "decomp": "auto", "gold_ctx": False,
            "kg_retrieval": True, "kg_bridging": False, "kg_context": False,
        },
        "kg_bridge_context": {
            "desc": "KG bridging + context",
            "decomp": "auto", "gold_ctx": False,
            "kg_retrieval": False, "kg_bridging": True, "kg_context": True,
        },
        "kg_full": {
            "desc": "All KG features (auto decomp)",
            "decomp": "auto", "gold_ctx": False,
            "kg_retrieval": True, "kg_bridging": True, "kg_context": True,
        },
        "kg_full_gold_decomp": {
            "desc": "All KG features (gold decomp, ceiling)",
            "decomp": "gold", "gold_ctx": False,
            "kg_retrieval": True, "kg_bridging": True, "kg_context": True,
        },
    }

    # Filter configs if requested
    if config_filter:
        config_defs = {k: v for k, v in config_defs.items() if k in config_filter}

    configs = {name: ConfigResult(name, spec["desc"]) for name, spec in config_defs.items()}

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

        # Get this question's KG
        kg = all_kgs[sample["id"]]

        # Build KG components once per question
        kg_ret = KGRetriever(kg, alpha=0.3)
        kg_ret.model = retriever.model  # Share embedding model
        kg_bridge = KGBridger(kg, boost=0.15)
        kg_enrich = KGContextEnricher(kg, max_triples=5)

        for cfg_name, spec in config_defs.items():
            decomp = gold_hops if spec["decomp"] == "gold" else auto_hops

            t0 = time.time()
            pred, hops, q_kg_stats = run_pipeline(
                sample, retriever,
                decomposition=decomp,
                model="qwen2.5:7b",
                top_k=3,
                use_gold_context=spec["gold_ctx"],
                verbose=(verbose and cfg_name == "kg_full"),
                kg=kg,
                use_kg_retrieval=spec["kg_retrieval"],
                use_kg_bridging=spec["kg_bridging"],
                use_kg_context=spec["kg_context"],
                kg_retriever=kg_ret if spec["kg_retrieval"] else None,
                kg_bridger=kg_bridge if spec["kg_bridging"] else None,
                kg_enricher=kg_enrich if spec["kg_context"] else None,
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

            # Accumulate KG stats
            for k in q_kg_stats:
                configs[cfg_name].kg_stats[k] = configs[cfg_name].kg_stats.get(k, 0) + q_kg_stats[k]

            s = "+" if em else ("~" if rem else "-")
            print(f"  [{s}] {cfg_name:22s}: {pred[:40]}")

        # Running totals every 5 questions
        if (i + 1) % 5 == 0 or i == len(samples) - 1:
            print(f"\n  ── Running totals ({i+1}/{len(samples)}) ──")
            for name, cfg in configs.items():
                if cfg.em_scores:
                    print(f"    {name:22s}: EM={cfg.em_rate:5.1f}%  relEM={cfg.relaxed_em_rate:5.1f}%  F1={cfg.mean_f1:.3f}")

    return configs, samples, all_kgs, kg_extractor


def print_final_results(configs, samples, all_kgs, kg_extractor):
    print(f"\n{'=' * 70}")
    print("FINAL RESULTS: KNOWLEDGE GRAPH INTEGRATION EXPERIMENT")
    print(f"{'=' * 70}")
    print(f"Dataset: MuSiQue | N={len(samples)} | 2-hop answerable")
    print(f"Extractor: qwen2.5:7b | Retrieval: BGE top-3")

    # KG stats
    total_entities = sum(kg.stats()["entities"] for kg in all_kgs.values())
    total_edges = sum(kg.stats()["edges"] for kg in all_kgs.values())
    avg_entities = total_entities / len(all_kgs) if all_kgs else 0
    avg_edges = total_edges / len(all_kgs) if all_kgs else 0
    print(f"KG extraction cache: {len(kg_extractor.cache)} paragraphs")
    print(f"Avg per question: {avg_entities:.1f} entities, {avg_edges:.1f} edges")
    print(f"KG extraction stats: {kg_extractor.stats}")

    # Main results table
    print(f"\n{'Config':<24s} {'KG-R':>4s} {'KG-B':>4s} {'KG-C':>4s} {'EM':>6s} {'relEM':>6s} {'F1':>6s} {'Lat':>7s}")
    print("─" * 70)

    order = ["baseline_auto", "kg_context_only", "kg_bridging_only", "kg_retrieval_only",
             "kg_bridge_context", "kg_full", "kg_full_gold_decomp"]
    for name in order:
        if name not in configs:
            continue
        cfg = configs[name]
        spec_lookup = {
            "baseline_auto": ("", "", ""),
            "kg_context_only": ("", "", "yes"),
            "kg_bridging_only": ("", "yes", ""),
            "kg_retrieval_only": ("yes", "", ""),
            "kg_bridge_context": ("", "yes", "yes"),
            "kg_full": ("yes", "yes", "yes"),
            "kg_full_gold_decomp": ("yes", "yes", "yes"),
        }
        r, b, c = spec_lookup.get(name, ("", "", ""))
        print(f"{name:<24s} {r:>4s} {b:>4s} {c:>4s} {cfg.em_rate:5.1f}% {cfg.relaxed_em_rate:5.1f}% {cfg.mean_f1:.3f} {cfg.mean_latency:5.0f}ms")

    # Key comparisons
    baseline_em = configs["baseline_auto"].em_rate if "baseline_auto" in configs else 0

    print(f"\n  ── Key Comparisons vs Baseline ({baseline_em:.1f}% EM) ──")
    for name in order[1:]:
        if name not in configs:
            continue
        cfg = configs[name]
        delta = cfg.em_rate - baseline_em
        marker = "+" if delta > 0 else ""
        print(f"  {name:24s}: {marker}{delta:.1f}% EM")

    # KG-specific metrics
    print(f"\n  ── KG Metrics ──")
    for name in order:
        if name not in configs:
            continue
        cfg = configs[name]
        ks = cfg.kg_stats
        if any(v > 0 for v in ks.values()):
            print(f"  {name:24s}: bridges={ks.get('bridge_hits',0)}, "
                  f"context_triples={ks.get('context_triples_added',0)}, "
                  f"retrieval_boosts={ks.get('retrieval_boosted',0)}")

    # Per-hop analysis
    print(f"\n  ── Per-Hop Analysis ──")
    for cfg_name in ["baseline_auto", "kg_full"]:
        if cfg_name not in configs:
            continue
        cfg = configs[cfg_name]
        if not cfg.hop_details:
            continue
        hop1_em = sum(1 for hops in cfg.hop_details
                      if len(hops) > 0 and hops[0].get("em", False))
        hop2_em = sum(1 for hops in cfg.hop_details
                      if len(hops) > 1 and hops[1].get("em", False))
        hop1_retr = sum(1 for hops in cfg.hop_details
                        if len(hops) > 0 and hops[0].get("gold_retrieved", False))
        hop2_retr = sum(1 for hops in cfg.hop_details
                        if len(hops) > 1 and hops[1].get("gold_retrieved", False))
        n = len(cfg.hop_details)
        print(f"  {cfg_name}:")
        print(f"    Hop 1: EM={hop1_em}/{n} ({hop1_em/n*100:.1f}%), Gold retrieved={hop1_retr}/{n} ({hop1_retr/n*100:.1f}%)")
        print(f"    Hop 2: EM={hop2_em}/{n} ({hop2_em/n*100:.1f}%), Gold retrieved={hop2_retr}/{n} ({hop2_retr/n*100:.1f}%)")

    # THE KEY QUESTION
    best_kg_name = max(
        [n for n in order[1:] if n in configs and n != "kg_full_gold_decomp"],
        key=lambda n: configs[n].em_rate,
        default=None
    )

    print(f"\n  ── THE KEY QUESTION ──")
    if best_kg_name:
        best_kg = configs[best_kg_name]
        delta = best_kg.em_rate - baseline_em
        if delta > 0:
            print(f"  KG HELPS: Best={best_kg_name} at {best_kg.em_rate:.1f}% EM ({delta:+.1f}% over baseline)")
            print(f"  → Knowledge graph enrichment improves extraction quality")
        elif delta == 0:
            print(f"  KG NEUTRAL: Best KG config = baseline at {baseline_em:.1f}% EM")
            print(f"  → KG doesn't hurt but doesn't measurably help at n={len(samples)}")
        else:
            print(f"  KG HURTS: Best={best_kg_name} at {best_kg.em_rate:.1f}% EM ({delta:.1f}% vs baseline)")
            print(f"  → KG features add noise; extraction already adequate")

    return {
        "metadata": {
            "n_samples": len(samples),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "kg_cache_size": len(kg_extractor.cache),
            "kg_extraction_stats": kg_extractor.stats,
            "avg_entities_per_question": sum(kg.stats()["entities"] for kg in all_kgs.values()) / len(all_kgs) if all_kgs else 0,
            "avg_edges_per_question": sum(kg.stats()["edges"] for kg in all_kgs.values()) / len(all_kgs) if all_kgs else 0,
        },
        "results": {name: {
            "em": cfg.em_rate,
            "relaxed_em": cfg.relaxed_em_rate,
            "f1": cfg.mean_f1,
            "latency_ms": cfg.mean_latency,
            "kg_stats": cfg.kg_stats,
            "per_question": cfg.per_question,
        } for name, cfg in configs.items()},
    }


# ══════════════════════════════════════════════════════════════════════
# Section 8: CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="v8: Knowledge Graph Integration Experiment")
    parser.add_argument("--limit", type=int, default=30,
                        help="Number of questions to test (default: 30)")
    parser.add_argument("--output", type=str, default="results/v8_knowledge_graph.json",
                        help="Output file path")
    parser.add_argument("--cache-dir", type=str, default="kg_cache",
                        help="Directory for KG triple extraction cache")
    parser.add_argument("--config", type=str, default=None,
                        help="Comma-separated config names to run (default: all)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config_filter = None
    if args.config:
        config_filter = set(args.config.split(","))
        # Always include baseline for comparison
        config_filter.add("baseline_auto")

    # Check Ollama
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"Ollama models: {models}")
        assert any("qwen2.5:7b" in m for m in models), "qwen2.5:7b not found"
    except AssertionError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR connecting to Ollama: {e}")
        sys.exit(1)

    start = time.time()
    configs, samples, all_kgs, kg_extractor = run_experiment(
        limit=args.limit,
        cache_dir=args.cache_dir,
        verbose=not args.quiet,
        config_filter=config_filter,
    )
    elapsed = time.time() - start

    results = print_final_results(configs, samples, all_kgs, kg_extractor)
    results["metadata"]["elapsed_seconds"] = elapsed

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to: {output_path}")
    print(f"Total time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
