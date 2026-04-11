import json
from typing import Callable
from pydantic import BaseModel


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
        return [self._tools[name] for name in names if name in self._tools]

    def execute(self, tool_name: str, args_json: str):
        handler = self._handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}
        args = json.loads(args_json)
        return handler(**args)
