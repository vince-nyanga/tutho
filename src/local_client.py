from openai import AsyncOpenAI
class LocalClient:
    def __init__(self, model: str = "gemma4:e4b", base_url: str = "http://localhost:8080/v1"):
        self.model = model
        self.client = AsyncOpenAI(base_url=base_url, api_key="sk-no-api")

    async def chat(self, system_prompt: str, messages: list[dict], tools: list[dict], tool_choice: str | dict | None = None) -> object:
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
