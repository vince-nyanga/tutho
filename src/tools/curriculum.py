import json
import re
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TopicResult:
    code: str
    name: str
    level: str
    knowledge_components: list[dict]
    parent_name: str | None = None
    exam_weight: float | None = None
    teaching_notes: list[str] | None = None
    common_misconceptions: list[str] | None = None


_STOP_WORDS = frozenset({
    "the", "and", "for", "are", "with", "that", "this", "from", "not",
    "but", "you", "all", "can", "has", "was", "one", "our", "out",
    "use", "her", "each", "which", "their", "will", "other", "about",
    "many", "then", "them", "these", "some", "would", "make", "like",
    "been", "have", "into", "more", "when", "very", "what", "how",
    "need", "help", "want", "learn", "study", "teach", "know",
})


def _tokenize(text: str) -> set[str]:
    words = set(re.findall(r'[a-z]{3,}', text.lower()))
    return words - _STOP_WORDS


def _trigrams(text: str) -> set[str]:
    """Generate trigrams from text, similar to Postgres pg_trgm."""
    t = f"  {text.lower()}  "
    return {t[i:i+3] for i in range(len(t) - 2)}


def _trigram_similarity(a: str, b: str) -> float:
    """Compute trigram similarity (0-1), like Postgres similarity()."""
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class CurriculumStore:
    def __init__(self, curriculum_dir: str = "curriculum"):
        self._nodes: dict[str, dict] = {}
        self._kcs: dict[str, dict] = {}
        self._data: list[dict] = []
        self._grade_subject_nodes: dict[tuple[int, str], list[str]] = {}
        self._node_keywords: dict[str, set[str]] = {}
        self._node_search_text: dict[str, str] = {}
        self._load(curriculum_dir)

    def _load(self, curriculum_dir: str):
        for path in Path(curriculum_dir).glob("*.json"):
            with open(path, "r") as f:
                data = json.load(f)
                self._data.append(data)
                grade = data["grade"]
                subject = data["subject"]
                key = (grade, subject)
                if key not in self._grade_subject_nodes:
                    self._grade_subject_nodes[key] = []
                self._index_nodes(data.get("nodes", []), parent_name=None, grade_subject_key=key)

    def get_topic(self, grade: int, subject: str, topic_query: str) -> TopicResult | None:
        key = (grade, subject)
        node_codes = self._grade_subject_nodes.get(key, [])

        if not node_codes:
            return None

        # Direct code match — only valid if the node has KCs (i.e. it's a topic, not a unit)
        query_upper = topic_query.strip().upper()
        if query_upper in self._nodes:
            node = self._nodes[query_upper]
            if query_upper in node_codes and node.get("knowledge_components"):
                return self._node_to_result(node)

        query_lower = topic_query.lower()
        query_words = query_lower.split()

        best_match = None
        best_score = 0

        for code in node_codes:
            node = self._nodes[code]
            score = self._score_match(node, query_words, query_lower)
            if score > best_score:
                best_match = node
                best_score = score

        if not best_match or best_score < 7:
            return None

        return self._node_to_result(best_match)

    def get_prerequisites(self, kc_code: str) -> list[dict]:
        kc = self._kcs.get(kc_code)
        if not kc:
            return []

        prereqs = []
        for prereq_code in kc.get("kc_prerequisites", []):
            prereq_kc = self._kcs.get(prereq_code)
            if prereq_kc:
                prereqs.append({
                    "code": prereq_kc["code"],
                    "description": prereq_kc.get("description"),
                    "node_name": prereq_kc.get("_parent_name")
                })
        return prereqs

    def get_node_by_code(self, code: str) -> dict | None:
        return self._nodes.get(code)

    def get_kc_by_code(self, code: str) -> dict | None:
        return self._kcs.get(code)

    def get_available_curriculum(self) -> dict[int, list[str]]:
        mapping = {}
        for data in self._data:
            grade = data["grade"]
            subject = data["subject"]
            if grade not in mapping:
                mapping[grade] = []
            if subject not in mapping[grade]:
                mapping[grade].append(subject)
        return mapping

    def get_topic_list(self, grade: int, subject: str) -> list[dict]:
        key = (grade, subject)
        node_codes = self._grade_subject_nodes.get(key, [])
        return [
            {"code": code, "name": self._nodes[code]["name"]}
            for code in node_codes
        ]

    def _node_to_result(self, node: dict) -> TopicResult:
        return TopicResult(
            code=node["code"],
            name=node["name"],
            level=node["level"],
            knowledge_components=node.get("knowledge_components", []),
            parent_name=node.get("parent_name"),
            exam_weight=node.get("exam_weight"),
            teaching_notes=[
                kc["teaching_notes"]
                for kc in node.get("knowledge_components", [])
                if kc.get("teaching_notes")
            ],
            common_misconceptions=[
                m
                for kc in node.get("knowledge_components", [])
                for m in kc.get("common_misconceptions", [])
            ]
        )

    def _index_nodes(self, nodes: list[dict], parent_name: str | None = None, grade_subject_key: tuple = None):
        for node in nodes:
            node["parent_name"] = parent_name

            # Always register in the node lookup for direct code access
            self._nodes[node["code"]] = node

            # Only add to the searchable index if this node has KCs (i.e. it's a topic, not a unit).
            # This prevents unit-level nodes from polluting the search index and ensures
            # get_topic() can never return a hollow TopicResult with empty KCs.
            if grade_subject_key and node.get("knowledge_components"):
                self._grade_subject_nodes[grade_subject_key].append(node["code"])

            keywords = _tokenize(node["name"])
            search_parts = [node["name"]]
            if parent_name:
                keywords |= _tokenize(parent_name)
                search_parts.append(parent_name)

            for kc in node.get("knowledge_components", []):
                self._kcs[kc["code"]] = kc
                kc["_node_code"] = node["code"]
                kc["_parent_name"] = node.get("name")
                keywords |= _tokenize(kc.get("description", ""))
                keywords |= _tokenize(kc.get("curriculum_statement", ""))
                search_parts.append(kc.get("description", ""))
                search_parts.append(kc.get("curriculum_statement", ""))
                for m in kc.get("common_misconceptions", []):
                    keywords |= _tokenize(m)
                    search_parts.append(m)

            self._node_keywords[node["code"]] = keywords
            self._node_search_text[node["code"]] = " ".join(search_parts)

            if "children" in node:
                self._index_nodes(node["children"], node.get("name"), grade_subject_key)

    def _score_match(self, node: dict, query_words: list[str], query_lower: str) -> float:
        name_lower = node["name"].lower()
        code = node["code"]

        # Exact name match
        if query_lower == name_lower:
            return 100

        query_tokens = _tokenize(query_lower)
        if not query_tokens:
            return 0

        # Trigram similarity against name (high weight, like pg_trgm)
        name_sim = _trigram_similarity(query_lower, name_lower)

        # Trigram similarity against parent unit name (students often use unit-level terms)
        parent_name = (node.get("parent_name") or "").lower()
        parent_sim = _trigram_similarity(query_lower, parent_name) if parent_name else 0

        # Trigram similarity against full search text (description, curriculum_statement, etc.)
        search_text = self._node_search_text.get(code, "")
        text_sim = _trigram_similarity(query_lower, search_text)

        # Token overlap for exact word matches
        name_tokens = _tokenize(name_lower)
        node_keywords = self._node_keywords.get(code, set())

        name_token_hits = len(query_tokens & name_tokens)
        keyword_token_hits = len(query_tokens & node_keywords)

        # Weighted combination
        score = (
            name_sim * 40           # trigram match on topic name
            + parent_sim * 20       # trigram match on parent unit name
            + text_sim * 10         # trigram match on descriptions
            + name_token_hits * 8   # exact word in name
            + keyword_token_hits * 3  # exact word in descriptions/statements
        )

        # Penalize when no query tokens appear in the node's indexed text.
        # Prevents spurious matches from coincidental trigram overlap.
        if not (query_tokens & node_keywords):
            score *= 0.3

        return score