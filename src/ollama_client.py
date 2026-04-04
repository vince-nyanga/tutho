import json

from openai import AsyncOpenAI
from src.base_client import ModelClient

class OllamaClient(ModelClient):
    def __init__(self, model: str = "gemma4:e4b", base_url: str = "http://localhost:11434/v1"):
        self.model = model
        self.client = AsyncOpenAI(base_url=base_url, api_key="ollama")

    async def classify(self, system_prompt: str, user_message: str, history: list[dict] = None) -> dict:
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history[-4:])

        messages.append({"role": "user", "content": user_message})
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    async def chat_with_tools(self, system_prompt: str, messages: list[dict], tools: list[dict]) -> object:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *messages
            ],
            "temperature": 0,
        }
        if tools:
            kwargs["tools"] = tools

        responses = await self.client.chat.completions.create(**kwargs)
        return responses.choices[0].message