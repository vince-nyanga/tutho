import json
import os
from jinja2 import Environment, FileSystemLoader
from src.base_client import ModelClient
from src.tools.curriculum import CurriculumStore
from src.tools.definitions import create_learning_registry, ToolRegistry
from src.mastery import get_mastery
from logging import getLogger

logger = getLogger(__name__)


class Router:
    def __init__(self, client: ModelClient, curriculum: CurriculumStore):
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

        match classification["intent"]:
            case "learn":
                return await self._handle_learn(message, classification, session, history)
            case "practice":
                return await self._handle_learn(message, classification, session, history)
            case "follow_up":
                return await self._handle_learn(message, classification, session, history)
            case "exam_prep":
                return await self._handle_exam_prep(message, classification, session)
            case "progress":
                return await self._handle_progress(message, classification, session)
            case "greeting":
                return await self._handle_greeting(message, classification, session)
            case "off_topic":
                return await self._handle_off_topic(message, classification, session)
            case _:
                return await self._handle_greeting(message, classification, session)

    async def _classify(self, message: str, session: dict, history: list[dict] = None) -> dict:
        template = self.templates.get_template("classifier.j2")
        prompt = template.render(
            session_grade=session.get("grade"),
            session_subject=session.get("subject"),
            current_topic=session.get("topic"),
            available_curriculum=self.curriculum.get_available_curriculum(),
        )

        return await self.client.classify(prompt, message, history)

    async def _handle_learn(self, message: str, classification: dict, session: dict, history: list[dict] = None) -> str:
        topic = None

        logger.info(f"Classification: {classification}")
        logger.info(f"Language: {session.get('language')}")
        logger.info(f"Language Name: {session.get('language_name')}")

        # Coerce grade to int and fall back to session for follow-ups
        topic_name = classification.get("topic") or session.get("current_topic")
        grade = classification.get("grade") or session.get("current_grade", 12)
        subject = classification.get("subject") or session.get("current_subject", "mathematics")

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

        # Save topic context to session for follow-up messages
        if topic:
            session["current_topic"] = topic_name
            session["current_grade"] = grade
            session["current_subject"] = subject

        phone_hash = session.get("phone_hash")

        logger.info(f"Topic found: {topic.code if topic else 'None'}")
        logger.info(f"KC codes: {[kc['code'] for kc in topic.knowledge_components] if topic else []}")
        logger.info(f"Phone hash: {phone_hash}")

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

        logger.info(f"Student mastery: {student_mastery}")

        template = self.templates.get_template("tutor.j2")
        system_prompt = template.render(
            grade=grade,
            subject=subject,
            language=session.get("language"),
            language_name=session.get("language_name"),
            topic=topic,
            student_mastery=student_mastery,
        )

        logger.info(f"System prompt (first 500 chars): {system_prompt[:500]}")

        learning_registry = create_learning_registry(self.curriculum, phone_hash)
        tools = learning_registry.get_tools()

        logger.info(f"Tools provided: {[t['function']['name'] for t in tools]}")

        messages = [*history, {"role": "user", "content": message}]

        response = await self.client.chat_with_tools(system_prompt, messages, tools)

        logger.info(f"Tool calls: {[tc.function.name for tc in response.tool_calls] if response.tool_calls else 'None'}")

        if response.tool_calls:
            return await self._execute_tool_loop(response, system_prompt, messages, tools, learning_registry)

        return response.content

    async def _handle_practice(self, message: str, classification: dict, session: dict) -> str:
        pass

    async def _handle_exam_prep(self, message: str, classification: dict, session: dict) -> str:
        pass

    async def _handle_progress(self, message: str, classification: dict, session: dict) -> str:
        pass

    async def _handle_greeting(self, message: str, classification: dict, session: dict) -> str:
        template = self.templates.get_template("greeting.j2")
        system_prompt = template.render(
            session_grade=session.get("grade"),
            session_subject=session.get("subject"),
            language=session.get("language"),
            language_name=session.get("language_name"),
        )
        response = await self.client.chat_with_tools(system_prompt, [{"role": "user", "content": message}], tools=[])
        return response.content

    async def _handle_off_topic(self, message: str, classification: dict, session: dict) -> str:
        template = self.templates.get_template("off_topic.j2")
        system_prompt = template.render(
            session_grade=session.get("grade"),
            session_subject=session.get("subject"),
            language=session.get("language"),
            language_name=session.get("language_name"),
        )
        response = await self.client.chat_with_tools(system_prompt, [{"role": "user", "content": message}], tools=[])
        return response.content


async def _execute_tool_loop(self, response, system_prompt: str, messages: list[dict], tools: list[dict],
                             registry: ToolRegistry = None) -> str:
    if registry is None:
        registry = create_learning_registry(self.curriculum)

    tool_calls_data = []
    tool_responses_data = []

    for tool_call in response.tool_calls:
        logger.info(f"Executing tool: {tool_call.function.name} with args: {tool_call.function.arguments}")
        args = json.loads(tool_call.function.arguments) if isinstance(tool_call.function.arguments,
                                                                      str) else tool_call.function.arguments
        result = registry.execute(tool_call.function.name, tool_call.function.arguments)
        logger.info(f"Tool result: {json.dumps(result, default=str)[:200]}")

        tool_calls_data.append({
            "function": {
                "name": tool_call.function.name,
                "arguments": args
            }
        })
        tool_responses_data.append({
            "name": tool_call.function.name,
            "response": result
        })

    messages.append({
        "role": "assistant",
        "tool_calls": tool_calls_data,
        "tool_responses": tool_responses_data,
    })

    follow_up_response = await self.client.chat_with_tools(system_prompt, messages, tools)

    logger.info(
        f"Follow-up tool calls: {[tc.function.name for tc in follow_up_response.tool_calls] if follow_up_response.tool_calls else 'None'}")

    if follow_up_response.tool_calls:
        return await self._execute_tool_loop(follow_up_response, system_prompt, messages, tools, registry)

    return follow_up_response.content