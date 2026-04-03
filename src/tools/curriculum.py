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
    def __init(self, curriculum_dir: str = "curriculum"):
        self._nodes: dict[str, dict] = {}
        self._kcs: dict[str, dict] = {}
        self._data: list[dict] = []


    def load(self, curriculum_dir: str):
        for path in Path(curriculum_dir).glob("*.json"):
            if path.name == "schema.json":
                continue
            with open(path, "r") as f:
                data = json.load(f)
                self._data.append(data)
                self._index_nodes(data.get("nodes", []))

    def get_topic(self, grade: int, subject: str, topic_query: str) -> TopicResult | None:
        query_words = topic_query.lower().split()

        best_match = None
        best_score = 0

        for data in self._data:
            if data["grade"] != grade or data["subject"] != subject:
                continue
            for code, node in self._nodes.items():
                score = self._score_match(node, query_words)
                if score > best_score:
                    best_match = node
                    best_score = score

        if not best_match:
            return None

        return TopicResult(
            code=best_match["code"],
            name=best_match["name"],
            level=best_match["level"],
            knowledge_components=best_match.get("knowledge_components", []),
            parent_name=best_match.get("parent_name"),
            exam_weight=best_match.get("exam_weight"),
            teaching_notes=[
                kc["teaching_notes"]
                for kc in best_match.get("knowledge_components", [])
                if kc.get("teaching_notes")
            ],
            common_misconceptions=[
               m
               for kc in best_match.get("knowledge_components", [])
                for m in kc.get("common_misconceptions", [])
            ]
        )

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


    def _index_nodes(self, nodes: list[dict], parent_name: str | None = None):
        for node in nodes:
            node["parent_name"] = parent_name
            self._nodes[node["code"]] = node
            for kc in node.get("knowledge_components", []):
                self._kcs[kc["code"]] = kc
                kc["_node_code"] = node["code"]
                kc["_parent_name"] = node.get("name")

            if "children" in node:
                self._index_nodes(node["children"], node.get("name"))


    def _score_match(self, node: dict, query_words: list[str]) -> int:
        # Very rudimentary for now
        score = 0
        name_lower = node["name"].lower()

        for word in query_words:
            if word in name_lower:
                score += 2

        for kc in node.get("knowledge_components", []):
            desc_lower = kc["description"].lower()
            for word in query_words:
                if word in desc_lower:
                    score += 1

        return score
