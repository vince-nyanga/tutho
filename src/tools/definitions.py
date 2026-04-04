import json
from dataclasses import asdict
from typing import Callable
from pydantic import BaseModel, Field

from src.tools.curriculum import CurriculumStore
from src.mastery import update_mastery, get_all_mastery


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, dict] = {}
        self._handlers: dict[str, Callable] = {}

    def register(self, tool_name: str, description: str, params_model: type[BaseModel], handler: Callable):
        self._tools[tool_name] = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": description,
                "parameters": params_model.model_json_schema(),
            }
        }
        self._handlers[tool_name] = handler

    def get_tools(self, names: list[str] | None = None) -> list[dict]:
        if names is None:
            return list(self._tools.values())
        else:
            return [self._tools[name] for name in names if name in self._tools]

    def execute(self, tool_name: str, args_json: str):
        handler = self._handlers[tool_name]
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}
        args = json.loads(args_json)
        return handler(**args)


# --- Param Models ---

class GetTopicParams(BaseModel):
    grade: int = Field(..., description="Grade level")
    subject: str = Field(..., description="Subject name")
    topic_query: str = Field(..., description="What the student is asking about")


class AssessResponseParams(BaseModel):
    kc_code: str = Field(..., description="The knowledge component code being assessed")
    is_correct: bool = Field(..., description="Whether the student's response was correct")


class GetProgressParams(BaseModel):
    pass


def create_learning_registry(curriculum: CurriculumStore, phone_hash: str = None) -> ToolRegistry:
    registry = ToolRegistry()

    def handle_get_topic(grade, subject, topic_query):
        result = curriculum.get_topic(grade, subject, topic_query)
        return asdict(result) if result else {"error": "Topic not found"}

    def handle_assess_response(kc_code, is_correct):
        if not phone_hash:
            return {"error": "No student session available"}
        kc = curriculum.get_kc_by_code(kc_code)
        slip = kc.get("slip_rate", 0.1) if kc else 0.1
        lr = kc.get("learning_rate", 0.15) if kc else 0.15
        p_l0 = kc.get("default_p_l0", 0.1) if kc else 0.1
        mastery = update_mastery(phone_hash, kc_code, is_correct, slip, lr, p_l0)
        return {
            "kc_code": kc_code,
            "mastery": mastery.p_mastery,
            "level": mastery.level,
            "attempts": mastery.attempts,
            "correct": mastery.correct,
        }

    def handle_get_progress():
        if not phone_hash:
            return {"error": "No student session available"}
        all_mastery = get_all_mastery(phone_hash)
        if not all_mastery:
            return {"message": "No progress tracked yet.", "topics": []}
        topics = []
        for m in all_mastery:
            topics.append({
                "kc_code": m.kc_code,
                "mastery": m.p_mastery,
                "level": m.level,
                "attempts": m.attempts,
                "accuracy": f"{m.correct / m.attempts:.0%}" if m.attempts > 0 else "N/A",
            })
        topics.sort(key=lambda t: t["mastery"])
        weakest = [t for t in topics if t["level"] in ("not_started", "developing")]
        return {
            "total_topics": len(topics),
            "weakest_areas": weakest[:3],
            "topics": topics,
        }

    registry.register(
        tool_name="get_topic",
        description="Get a topic from the curriculum",
        params_model=GetTopicParams,
        handler=handle_get_topic,
    )

    registry.register(
        tool_name="assess_response",
        description="Record whether the student answered correctly and update their mastery level for a knowledge component. Call this after the student attempts a question or demonstrates understanding.",
        params_model=AssessResponseParams,
        handler=handle_assess_response,
    )

    registry.register(
        tool_name="get_progress",
        description="Get the student's mastery progress across all topics they have worked on. Use this when the student asks how they are doing or what to study next.",
        params_model=GetProgressParams,
        handler=handle_get_progress,
    )

    return registry