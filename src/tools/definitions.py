import json
from dataclasses import asdict
from typing import Callable
from pydantic import BaseModel, Field

from src.tools.curriculum import CurriculumStore

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


# --- Learning Tools ---

class GetTopicParams(BaseModel):
    grade: int = Field(..., description="Grade level")
    subject: str = Field(..., description="Subject name")
    topic_query: str = Field(..., description="What the student is asking about")


def create_learning_registry(curriculum: CurriculumStore) -> ToolRegistry:
    registry = ToolRegistry()

    def handle_get_topic(grade, subject, topic_query):
        result = curriculum.get_topic(grade, subject, topic_query)
        return asdict(result) if result else {"error": "Topic not found"}

    registry.register(
        tool_name="get_topic",
        description="Get a topic from the curriculum",
        params_model=GetTopicParams,
        handler=handle_get_topic)

    return registry