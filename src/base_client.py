from typing import Protocol, runtime_checkable

@runtime_checkable
class ModelClient(Protocol):
    async def chat(
            self,
            system_prompt: str,
            messages: list[dict],
            tools: list[dict],
            tool_choice: str | dict | None = None) -> object:
        ...
