import json
from logging import getLogger

logger = getLogger(__name__)

from src.base_client import ModelClient

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(fn): return fn

_pipe = None

@spaces.GPU
def _run_inference(model_name, messages, tools=None, max_new_tokens=512):
    global _pipe
    if _pipe is None:
        import torch
        from transformers import pipeline
        logger.info(f"Loading model {model_name}")
        _pipe = pipeline(
            "any-to-any",
            model=model_name,
            dtype=torch.float16,
            device_map="auto"
        )
        logger.info("Model loaded")
    if tools:
        return _pipe(messages, tools=tools, max_new_tokens=max_new_tokens, return_full_text=False)
    return _pipe(messages, max_new_tokens=max_new_tokens, return_full_text=False)


class TransformersClient(ModelClient):
    def __init__(self, model_name: str = "google/gemma-4-E2B-it"):
        self.model_name = model_name

    async def classify(self, system_prompt, user_message, history=None) -> dict:
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history[-4:])
        messages.append({"role": "user", "content": user_message})
        output = _run_inference(self.model_name, messages, max_new_tokens=256)
        text = output[0]["generated_text"].strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())

    async def chat_with_tools(self, system_prompt, messages, tools) -> object:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        output = _run_inference(self.model_name, full_messages, tools=tools, max_new_tokens=1024)
        return _TransformersMessage(output[0]["generated_text"])


class _TransformersMessage:
    def __init__(self, content):
        self.content = None
        self.tool_calls = None
        if isinstance(content, list):
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