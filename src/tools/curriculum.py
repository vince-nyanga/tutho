import json
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


class CurriculumStore:
    def __init__(self, curriculum_dir: str = "curriculum"):
        self._nodes: dict[str, dict] = {}
        self._kcs: dict[str, dict] = {}
        self._data: list[dict] = []
        self._grade_subject_nodes: dict[tuple[int, str], list[str]] = {}
        self.load(curriculum_dir)

    def load(self, curriculum_dir: str):
        for path in Path(curriculum_dir).glob("*.json"):
            if path.name == "schema.json":
                continue
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

        # Direct code match first
        query_upper = topic_query.strip().upper()
        if query_upper in self._nodes:
            node = self._nodes[query_upper]
            if query_upper in node_codes:
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

        if not best_match:
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
        topics = []
        for code in node_codes:
            node = self._nodes[code]
            if node.get("knowledge_components"):
                topics.append({
                    "code": code,
                    "name": node["name"],
                })
        return topics

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
            self._nodes[node["code"]] = node
            if grade_subject_key:
                self._grade_subject_nodes[grade_subject_key].append(node["code"])

            for kc in node.get("knowledge_components", []):
                self._kcs[kc["code"]] = kc
                kc["_node_code"] = node["code"]
                kc["_parent_name"] = node.get("name")

            if "children" in node:
                self._index_nodes(node["children"], node.get("name"), grade_subject_key)

    def _score_match(self, node: dict, query_words: list[str], query_lower: str) -> int:
        score = 0
        name_lower = node["name"].lower()

        # Exact name match gets a big bonus
        if query_lower == name_lower:
            return 100

        # Full query appears as substring of name
        if query_lower in name_lower:
            score += 10

        # Name appears as substring of query
        if name_lower in query_lower:
            score += 8

        # Individual word matches in name
        for word in query_words:
            if len(word) <= 2:
                continue
            if word in name_lower:
                score += 3
            # Partial match: "sequence" matches "sequences"
            elif any(word in part or part in word for part in name_lower.split()):
                score += 2

        # KC description matches (lower weight)
        for kc in node.get("knowledge_components", []):
            desc_lower = kc.get("description", "").lower()
            for word in query_words:
                if len(word) <= 2:
                    continue
                if word in desc_lower:
                    score += 1

        return score