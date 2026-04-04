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
_model = None
_tokenizer = None

@spaces.GPU
def _run_inference(model_name, messages, tools=None, max_new_tokens=512):
    global _pipe, _model, _tokenizer
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
        _model = _pipe.model
        _tokenizer = _pipe.tokenizer
        logger.info("Model loaded")

    if tools:
        inputs = _tokenizer.apply_chat_template(
            messages,
            tools=tools,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).to(_model.device)

        input_len = inputs["input_ids"].shape[-1]
        output = _model.generate(**inputs, max_new_tokens=max_new_tokens)
        new_tokens = output[0][input_len:]
        raw_text = _tokenizer.decode(new_tokens, skip_special_tokens=False)
        logger.info(f"Raw tool output: {raw_text[:500]}")
        return raw_text

    converted = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        converted.append({"role": msg["role"], "content": content})

    return _pipe(converted, max_new_tokens=max_new_tokens, return_full_text=False)


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
        output = _run_inference(self.model_name, messages, max_new_tokens=1024)

        text = output[0]["generated_text"].strip()
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

        if tools:
            raw_text = _run_inference(self.model_name, full_messages, tools=tools, max_new_tokens=2048)
            return self._parse_tool_response(raw_text)
        else:
            output = _run_inference(self.model_name, full_messages, max_new_tokens=2048)
            content = output[0]["generated_text"]
            if isinstance(content, str):
                content = content.strip().replace("<turn|>", "").strip()
            return _TransformersMessage(content)

    def _parse_tool_response(self, raw_text: str) -> object:
        logger.info(f"Parsing tool response: {raw_text[:500]}")

        for token in ["<end_of_turn>", "<eos>", "<turn|>"]:
            raw_text = raw_text.replace(token, "")
        raw_text = raw_text.strip()

        if "```tool_code" in raw_text:
            tool_calls = []
            parts = raw_text.split("```tool_code")
            text_content = parts[0].strip()

            for part in parts[1:]:
                end = part.find("```")
                if end != -1:
                    json_str = part[:end].strip()
                else:
                    json_str = part.strip()

                try:
                    tc_data = json.loads(json_str)
                    tool_calls.append(tc_data)
                    logger.info(f"Parsed tool call: {tc_data}")
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool call JSON: {json_str}")

            if tool_calls:
                return _TransformersMessage(tool_calls)

            if text_content:
                return _TransformersMessage(text_content)

        if raw_text.startswith("{") or raw_text.startswith("["):
            try:
                data = json.loads(raw_text)
                if isinstance(data, dict) and "name" in data:
                    return _TransformersMessage([data])
                if isinstance(data, list):
                    return _TransformersMessage(data)
            except json.JSONDecodeError:
                pass

        return _TransformersMessage(raw_text)


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