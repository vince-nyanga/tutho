import inspect
import json
from typing import Callable, get_type_hints

from pydantic import create_model
from pydantic.fields import FieldInfo


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, dict] = {}
        self._handlers: dict[str, Callable] = {}

    def register(self, tool_name: str, handler: Callable):
        hints = get_type_hints(handler, include_extras=True)
        sig = inspect.signature(handler)
        fields = {}
        for name, param in sig.parameters.items():
            annotation = hints.get(name)
            if hasattr(annotation, "__metadata__"):
                base_type = annotation.__args__[0]
                field_info = next(
                    (m for m in annotation.__metadata__ if isinstance(m, FieldInfo)),
                    ...,
                )
                fields[name] = (base_type, field_info)
            else:
                fields[name] = (annotation, ...)

        model = create_model(f"{tool_name}_params", **fields)
        self._tools[tool_name] = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": inspect.getdoc(handler) or "",
                "parameters": model.model_json_schema(),
            },
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
