from typing import Protocol, runtime_checkable

@runtime_checkable
class ModelClient(Protocol):
    async def classify(
            self,
            system_prompt: str,
            user_message: str,
            history: list[dict] | None = None) -> dict:
        ...

    async def chat_with_tools(
            self,
            system_prompt: str,
            messages: list[dict],
            tools: list[dict]) -> object:
        ...