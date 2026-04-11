from dataclasses import asdict
from pydantic import BaseModel, Field

from src.tools.curriculum import CurriculumStore
from src.tools.registry import ToolRegistry
from src.db import update_mastery


class GetTopicsParams(BaseModel):
    grade: int = Field(..., description="Grade level (e.g. 12)")
    subject: str = Field(..., description="Subject name (e.g. Mathematics)")


class GetTopicParams(BaseModel):
    grade: int = Field(..., description="Grade level")
    subject: str = Field(..., description="Subject name")
    topic_query: str = Field(..., description="What the student is asking about")


class AssessResponseParams(BaseModel):
    kc_code: str = Field(..., description="The knowledge component code being assessed")
    is_correct: bool = Field(..., description="Whether the student's response was correct")


def create_learning_registry(curriculum: CurriculumStore, phone_hash: str = None) -> ToolRegistry:
    registry = ToolRegistry()

    def handle_get_topics(grade, subject):
        topics = curriculum.get_topic_list(grade, subject)
        if not topics:
            return {"error": "No topics found for this grade and subject"}
        return {"topics": [{"name": t["name"], "code": t["code"]} for t in topics]}

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

    registry.register(
        tool_name="get_topics",
        description="Look up what topics are available in the CAPS curriculum.",
        params_model=GetTopicsParams,
        handler=handle_get_topics,
    )

    registry.register(
        tool_name="get_topic",
        description="Fetch full details for a specific topic including teaching notes, misconceptions, and knowledge components.",
        params_model=GetTopicParams,
        handler=handle_get_topic,
    )

    registry.register(
        tool_name="assess_response",
        description="Record whether the student's answer was correct. This updates their mastery score.",
        params_model=AssessResponseParams,
        handler=handle_assess_response,
    )

    return registry
