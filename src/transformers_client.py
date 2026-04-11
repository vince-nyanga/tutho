import re
import json
from logging import getLogger

logger = getLogger(__name__)


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
    else:
        text = _processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = _processor(text=text, return_tensors="pt").to(_model.device)
    input_len = inputs["input_ids"].shape[-1]

    output = _model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )

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

    results = []
    for name, args in re.findall(
        r"(?:<\|tool_call>)?call:(\w+)\{(.*?)\}<tool_call\|>", text, re.DOTALL
    ):
        # Try parsing as JSON first (handles {"key": "value"} format)
        try:
            parsed = json.loads("{" + args + "}")
            results.append({"name": name, "arguments": parsed})
            continue
        except (json.JSONDecodeError, ValueError):
            pass

        # Fall back to Gemma-specific format (key:value without quotes)
        arguments = {
            k: cast((v1 or v2).strip())
            for k, v1, v2 in re.findall(r'(\w+):(?:<\|"\|>(.*?)<\|"\|>|([^,}]*))', args)
        }
        results.append({"name": name, "arguments": arguments})

    return results


class TransformersClient:
    def __init__(self, model_name: str = "google/gemma-4-E4B-it"):
        self.model_name = model_name

    async def chat(self, system_prompt, messages, tools, tool_choice=None) -> object:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        raw_text = _run_inference(
            self.model_name, full_messages,
            tools=tools if tools else None,
            max_new_tokens=2048
        )
        return self._parse_response(raw_text)

    def _parse_response(self, raw_text: str) -> object:
        tool_calls = _extract_tool_calls(raw_text)

        if tool_calls:
            logger.info(f"Parsed tool calls: {tool_calls}")
            # Strip tool call tags from remaining text to get content
            cleaned = re.sub(r'(?:<\|tool_call>)?call:\w+\{.*?\}<tool_call\|>', '', raw_text, flags=re.DOTALL).strip()
            for token in ["<end_of_turn>", "<eos>", "<turn|>", "<|tool_response>"]:
                cleaned = cleaned.replace(token, "")
            cleaned = re.sub(r'<\|.*?\|>', '', cleaned).strip()
            return _TransformersMessage(tool_calls, cleaned if cleaned else None)

        for token in ["<end_of_turn>", "<eos>", "<turn|>", "<|tool_response>"]:
            raw_text = raw_text.replace(token, "")
        raw_text = re.sub(r'<\|.*?\|>', '', raw_text).strip()
        return _TransformersMessage(raw_text)


class _TransformersMessage:
    def __init__(self, content, text=None):
        self.content = None
        self.tool_calls = None
        if isinstance(content, list):
            self.tool_calls = [_TransformersToolCall(tc) for tc in content]
            self.content = text
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