import json
from logging import getLogger

logger = getLogger(__name__)

from src.base_client import ModelClient

class TransformersClient(ModelClient):
    def __init__(self, model_name: str = "google/gemma-4-E2B-it"):
        self.model_name = model_name
        self.pipe = None

    def classify(
            self,
            system_prompt: str,
            user_message: str,
            history: list[dict] | None = None) -> dict:

        self._load_model()

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history[-4:])
        messages.append({"role": "user", "content": user_message})

        output = self.pipe(
            messages,
            max_new_tokens=256,
            return_full_text=False
        )

        text = output[0]["generated_text"]
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())


    def chat_with_tools(
            self,
            system_prompt: str,
            messages: list[dict],
            tools: list[dict]) -> object:

        self._load_model()
        output = self.pipe(
            [{"role": "system", "content": system_prompt}, messages],
            tools=tools,
            max_new_tokens=1024,
            return_full_text=False
        )
        return _TransformersMessage(output[0]["generated_text"])


    def _load_model(self):
        logger.info(f"Loading model {self.model_name}")
        if self.pipe is not None:
            logger.info("Model already loaded")
            return

        try:
            import spaces
            import torch
            from transformers import pipeline
            self.pipe = pipeline(
                "any-to-any",
                model=self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            logger.info(f"Model loaded")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}. {e}")
            raise RuntimeError(f"Failed to load model {self.model_name}. {e}")


class _TransformersMessage:
    """Wraps transformers tool-call output to match the OpenAI message interface."""

    def __init__(self, content):
        self.content = None
        self.tool_calls = None

        if isinstance(content, list):
            # Tool calls come back as a list of dicts with "name" and "arguments"
            self.tool_calls = [_TransformersToolCall(tc) for tc in content]
        else:
            self.content = content


class _TransformersToolCall:
    def __init__(self, data: dict):
        self.id = f"call_{data.get('name', 'tool')}"
        self.function = _TransformersFunction(data)


class _TransformersFunction:
    def __init__(self, data: dict):
        self.name = data.get("name", "")
        args = data.get("arguments", {})
        self.arguments = json.dumps(args) if isinstance(args, dict) else args