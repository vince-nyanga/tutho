import json
import re
import os
from enum import Enum
from jinja2 import Environment, FileSystemLoader
from src.tools.curriculum import CurriculumStore
from src.tools.definitions import create_learning_registry
from src.tools.registry import ToolRegistry
from src.db import get_mastery
from logging import getLogger


class Intent(str, Enum):
    LEARN = "learn"
    ANSWER = "answer"
    GREETING = "greeting"
    OFF_TOPIC = "off_topic"

logger = getLogger(__name__)


class Router:
    def __init__(self, client, curriculum: CurriculumStore):
        self.client = client
        self.curriculum = curriculum
        self.templates = Environment(loader=FileSystemLoader("src/prompts"))

    async def handle_message(self, message: str, session: dict, history: list[dict] = None) -> str:
        if history is None:
            history = []
        classification = await self._classify(message, session, history)

        if not classification.get("grade") and session.get("grade"):
            classification["grade"] = session["grade"]

        if not classification.get("subject") and session.get("subject"):
            classification["subject"] = session["subject"]

        try:
            intent = Intent(classification.get("intent", "greeting"))
        except ValueError:
            logger.warning(f"Unknown intent '{classification.get('intent')}', falling back to learn")
            intent = Intent.LEARN
        logger.info(f"Routing: {intent.value}")

        match intent:
            case Intent.LEARN:
                return await self._handle_learn(message, classification, session, history)
            case Intent.ANSWER:
                return await self._handle_answer(message, classification, session, history)
            case Intent.GREETING:
                return await self._handle_greeting(message, classification, session)
            case Intent.OFF_TOPIC:
                return await self._handle_off_topic(message, classification, session)

    async def _classify(self, message: str, session: dict, history: list[dict] = None) -> dict:
        grade = session.get("grade", 12)
        subject = session.get("subject", "Mathematics")

        template = self.templates.get_template("classifier.j2")
        prompt = template.render(
            session_grade=grade,
            session_subject=subject,
            current_topic=session.get("current_topic"),
            current_kc=session.get("current_kc"),
        )

        registry = create_learning_registry(self.curriculum, session.get("phone_hash"))
        tools = registry.get_tools(names=["get_topics", "get_topic"])

        messages = [{"role": "user", "content": message}]
        response = await self.client.chat(prompt, messages, tools)

        if response.tool_calls:
            logger.info(f"Classifier tools: {[tc.function.name for tc in response.tool_calls]}")
            result = await self._execute_tool_loop(response, prompt, messages, tools, registry)
            classification = self._extract_json(result)
        else:
            classification = self._extract_json(response.content)

        logger.info(f"Classification: {classification}")

        # Update session from classification
        if classification.get("kc_code"):
            session["current_kc"] = classification["kc_code"]
        if classification.get("topic"):
            session["current_topic"] = classification["topic"]

        return classification

    def _extract_json(self, text: str) -> dict:
        if not text:
            return {"intent": "greeting", "subject": None, "grade": None, "topic": None}
        cleaned = re.sub(r'<\|.*?\|>', '', text).strip()
        start = cleaned.find('{')
        end = cleaned.rfind('}') + 1
        if start >= 0 and end > start:
            try:
                result = json.loads(cleaned[start:end])
                # Normalize string "None"/"null" to actual None
                for key in result:
                    if isinstance(result[key], str) and result[key].lower() in ("none", "null"):
                        result[key] = None
                return result
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse classifier JSON: {cleaned[start:end]}")
        logger.warning(f"No JSON found in classifier output: {text[:200]}")
        return {"intent": "greeting", "subject": None, "grade": None, "topic": None}

    async def _handle_learn(self, message: str, classification: dict, session: dict, history: list[dict] = None) -> str:
        topic = None
        topic_name = classification.get("topic") or session.get("current_topic")
        grade = classification.get("grade") or session.get("current_grade", 12)
        subject = classification.get("subject") or session.get("current_subject", "Mathematics")

        try:
            grade = int(grade)
        except (TypeError, ValueError):
            grade = 12

        if topic_name and grade and subject:
            topic = self.curriculum.get_topic(
                grade=grade,
                subject=subject,
                topic_query=topic_name
            )

        if topic:
            session["current_topic"] = topic_name
            session["current_grade"] = grade
            session["current_subject"] = subject
            if topic.knowledge_components:
                session["current_kc"] = topic.knowledge_components[0]["code"]

        phone_hash = session.get("phone_hash")

        logger.info(f"Topic: {topic.code if topic else 'None'}, KC: {session.get('current_kc')}")

        student_mastery = []
        if topic and phone_hash:
            for kc in topic.knowledge_components:
                m = get_mastery(phone_hash, kc["code"], kc.get("default_p_l0", 0.1))
                student_mastery.append({
                    "code": kc["code"],
                    "description": kc.get("description", ""),
                    "mastery": m.p_mastery,
                    "level": m.level,
                    "attempts": m.attempts,
                })

        template = self.templates.get_template("tutor.j2")
        system_prompt = template.render(
            grade=grade,
            subject=subject,
            language=session.get("language"),
            language_name=session.get("language_name"),
            topic=topic,
            student_mastery=student_mastery,
        )

        learning_registry = create_learning_registry(self.curriculum, phone_hash)
        tools = learning_registry.get_tools(names=["get_topics", "get_topic"])

        messages = list(history) if history else [{"role": "user", "content": message}]

        response = await self.client.chat(system_prompt, messages, tools)

        if response.tool_calls:
            return await self._execute_tool_loop(response, system_prompt, messages, tools, learning_registry)

        return response.content

    async def _handle_answer(self, message: str, classification: dict, session: dict, history: list[dict] = None) -> str:
        grade = classification.get("grade") or session.get("current_grade", 12)
        subject = classification.get("subject") or session.get("current_subject", "Mathematics")
        kc_code = classification.get("kc_code") or session.get("current_kc")
        phone_hash = session.get("phone_hash")

        logger.info(f"Assessing answer for KC: {kc_code}")

        template = self.templates.get_template("answer.j2")
        system_prompt = template.render(
            grade=grade,
            subject=subject,
            language=session.get("language"),
            language_name=session.get("language_name"),
            kc_code=kc_code,
        )

        learning_registry = create_learning_registry(self.curriculum, phone_hash)
        tools = learning_registry.get_tools(names=["assess_response"])

        messages = list(history[-4:]) if history else [{"role": "user", "content": message}]

        response = await self.client.chat(system_prompt, messages, tools)

        if response.tool_calls:
            return await self._execute_tool_loop(response, system_prompt, messages, tools, learning_registry)

        logger.warning("assess_response was NOT called by the model")
        return response.content

    async def _handle_greeting(self, message: str, classification: dict, session: dict) -> str:
        template = self.templates.get_template("greeting.j2")
        system_prompt = template.render(
            session_grade=session.get("grade"),
            session_subject=session.get("subject"),
            language=session.get("language"),
            language_name=session.get("language_name"),
        )
        response = await self.client.chat(system_prompt, [{"role": "user", "content": message}], tools=[])
        return response.content

    async def _handle_off_topic(self, message: str, classification: dict, session: dict) -> str:
        template = self.templates.get_template("off_topic.j2")
        system_prompt = template.render(
            session_grade=session.get("grade"),
            session_subject=session.get("subject"),
            language=session.get("language"),
            language_name=session.get("language_name"),
        )
        response = await self.client.chat(system_prompt, [{"role": "user", "content": message}], tools=[])
        return response.content

    async def _execute_tool_loop(self, response, system_prompt: str, messages: list[dict], tools: list[dict],
                                 registry: ToolRegistry = None) -> str:
        if registry is None:
            registry = create_learning_registry(self.curriculum)

        tool_calls_data = []
        tool_responses_data = []

        for tool_call in response.tool_calls:
            args = json.loads(tool_call.function.arguments) if isinstance(tool_call.function.arguments,
                                                                          str) else tool_call.function.arguments
            logger.info(f"Tool: {tool_call.function.name}({args})")
            try:
                result = registry.execute(tool_call.function.name, tool_call.function.arguments)
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                result = {"error": str(e)}

            tool_calls_data.append({
                "id": tool_call.id or f"call_{tool_call.function.name}",
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments if isinstance(tool_call.function.arguments, str) else json.dumps(args)
                }
            })
            tool_responses_data.append({
                "role": "tool",
                "tool_call_id": tool_call.id or f"call_{tool_call.function.name}",
                "content": json.dumps(result, default=str)
            })

        messages.append({
            "role": "assistant",
            "tool_calls": tool_calls_data,
        })
        messages.extend(tool_responses_data)

        follow_up_response = await self.client.chat(system_prompt, messages, tools)

        if follow_up_response.tool_calls:
            return await self._execute_tool_loop(follow_up_response, system_prompt, messages, tools, registry)

        return follow_up_response.content
