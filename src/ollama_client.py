import json

from openai import AsyncOpenAI
from src.base_client import ModelClient

class OllamaClient(ModelClient):
    def __init__(self, model: str = "gemma4:e4b", base_url: str = "http://localhost:8080/v1"):
        self.model = model
        self.client = AsyncOpenAI(base_url=base_url, api_key="sk-no-api")

    async def classify(self, system_prompt: str, user_message: str, history: list[dict] = None) -> dict:
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history[-4:])

        messages.append({"role": "user", "content": user_message})
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=1,
            top_p=95,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    async def chat_with_tools(self, system_prompt: str, messages: list[dict], tools: list[dict], tool_choice: str | dict | None = None) -> object:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *messages
            ],
            "temperature": 1,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice and tools:
            kwargs["tool_choice"] = tool_choice

        responses = await self.client.chat.completions.create(**kwargs)
        return responses.choices[0].message