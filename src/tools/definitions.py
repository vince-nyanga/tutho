from dataclasses import asdict
from typing import Annotated

from pydantic import Field

from src.tools.curriculum import CurriculumStore
from src.tools.registry import ToolRegistry
from src.db import update_mastery


def create_learning_registry(curriculum: CurriculumStore, phone_hash: str = None) -> ToolRegistry:
    registry = ToolRegistry()

    def handle_get_topics(
        grade: Annotated[int, Field(description="Grade level (e.g. 12)")],
        subject: Annotated[str, Field(description="Subject name (e.g. Mathematics)")],
    ):
        """Look up what topics are available in the CAPS curriculum."""
        topics = curriculum.get_topic_list(grade, subject)
        if not topics:
            return {"error": "No topics found for this grade and subject"}
        return {"topics": [{"name": t["name"], "code": t["code"]} for t in topics]}

    def handle_get_topic(
        grade: Annotated[int, Field(description="Grade level")],
        subject: Annotated[str, Field(description="Subject name")],
        topic_query: Annotated[str, Field(description="What the student is asking about")],
    ):
        """Fetch full details for a specific topic including teaching notes, misconceptions, and knowledge components."""
        result = curriculum.get_topic(grade, subject, topic_query)
        return asdict(result) if result else {"error": "Topic not found"}

    def handle_assess_response(
        kc_code: Annotated[str, Field(description="The knowledge component code being assessed")],
        is_correct: Annotated[bool, Field(description="Whether the student's response was correct")],
    ):
        """Record whether the student's answer was correct. This updates their mastery score."""
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

    registry.register("get_topics", handle_get_topics)
    registry.register("get_topic", handle_get_topic)
    registry.register("assess_response", handle_assess_response)

    return registry
