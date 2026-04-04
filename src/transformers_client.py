import re
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

_model = None
_processor = None

@spaces.GPU
def _run_inference(model_name, messages, tools=None, max_new_tokens=512):
    global _model, _processor
    if _model is None:
        import torch
        from transformers import AutoProcessor, AutoModelForMultimodalLM
        logger.info(f"Loading model {model_name}")
        _model = AutoModelForMultimodalLM.from_pretrained(
            model_name, dtype="auto", device_map="auto"
        )
        _processor = AutoProcessor.from_pretrained(model_name)
        logger.info("Model loaded")

    if tools:
        text = _processor.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = _processor(text=text, return_tensors="pt").to(_model.device)
    else:
        text = _processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = _processor(text=text, return_tensors="pt").to(_model.device)

    input_len = inputs["input_ids"].shape[-1]
    output = _model.generate(**inputs, max_new_tokens=max_new_tokens)
    new_tokens = output[0][input_len:]
    raw_text = _processor.decode(new_tokens, skip_special_tokens=False)
    logger.info(f"Raw output: {raw_text[:500]}")
    return raw_text


def _extract_tool_calls(text):
    def cast(v):
        try:
            return int(v)
        except:
            try:
                return float(v)
            except:
                return {'true': True, 'false': False}.get(v.lower(), v.strip("'\""))

    return [{
        "name": name,
        "arguments": {
            k: cast((v1 or v2).strip())
            for k, v1, v2 in re.findall(r'(\w+):(?:<\|"\|>(.*?)<\|"\|>|([^,}]*))', args)
        }
    } for name, args in re.findall(
        r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", text, re.DOTALL
    )]


class TransformersClient(ModelClient):
    def __init__(self, model_name: str = "google/gemma-4-E4B-it"):
        self.model_name = model_name

    async def classify(self, system_prompt, user_message, history=None) -> dict:
        messages = []
        if history:
            messages.extend(history[-4:])
        messages.append({
            "role": "user",
            "content": f"{system_prompt}\n\nMessage to classify: {user_message}\n\nRespond with JSON only. No explanation."
        })
        raw_text = _run_inference(self.model_name, messages, max_new_tokens=1024)

        # Clean special tokens
        for token in ["<end_of_turn>", "<eos>", "<turn|>"]:
            raw_text = raw_text.replace(token, "")
        text = raw_text.strip()

        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON found in model output: {text}")
        return json.loads(text[start:end])

    async def chat_with_tools(self, system_prompt, messages, tools) -> object:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        raw_text = _run_inference(self.model_name, full_messages, tools=tools if tools else None, max_new_tokens=2048)
        return self._parse_response(raw_text)

    def _parse_response(self, raw_text: str) -> object:
        tool_calls = _extract_tool_calls(raw_text)

        if tool_calls:
            logger.info(f"Parsed tool calls: {tool_calls}")
            return _TransformersMessage(tool_calls)

        # No tool calls, extract text content
        for token in ["<end_of_turn>", "<eos>", "<turn|>", "<|tool_response>"]:
            raw_text = raw_text.replace(token, "")
        return _TransformersMessage(raw_text.strip())


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