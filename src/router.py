import json
import re
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
        grade = session.get("grade", 12)
        subject = session.get("subject", "Mathematics")

        template = self.templates.get_template("classifier.j2")
        prompt = template.render(
            session_grade=grade,
            session_subject=subject,
            current_topic=session.get("current_topic"),
            available_curriculum=self.curriculum.get_available_curriculum(),
        )

        # Give classifier just the get_topics tool
        registry = create_learning_registry(self.curriculum, session.get("phone_hash"))
        all_tools = registry.get_tools()
        classifier_tools = [t for t in all_tools if t["function"]["name"] == "get_topics"]

        logger.info(f"Classifier tools: {[t['function']['name'] for t in classifier_tools]}")

        messages = [{"role": "user", "content": message}]
        response = await self.client.chat_with_tools(prompt, messages, classifier_tools)

        logger.info(f"Classifier tool calls: {[tc.function.name for tc in response.tool_calls] if response.tool_calls else 'None'}")

        # If model called get_topics, execute it and get final answer
        if response.tool_calls:
            result = await self._execute_tool_loop(response, prompt, messages, classifier_tools, registry)
            return self._extract_json(result)

        # No tool call — parse JSON directly from response
        return self._extract_json(response.content)

    def _extract_json(self, text: str) -> dict:
        """Extract JSON classification from model output text."""
        if not text:
            return {"intent": "greeting", "subject": None, "grade": None, "topic": None}
        # Strip any special tokens
        cleaned = re.sub(r'<\|.*?\|>', '', text).strip()
        start = cleaned.find('{')
        end = cleaned.rfind('}') + 1
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse classifier JSON: {cleaned[start:end]}")
        logger.warning(f"No JSON found in classifier output: {text[:200]}")
        return {"intent": "greeting", "subject": None, "grade": None, "topic": None}

    async def _handle_learn(self, message: str, classification: dict, session: dict, history: list[dict] = None) -> str:
        topic = None

        logger.info(f"Classification: {classification}")
        logger.info(f"Language: {session.get('language')}")