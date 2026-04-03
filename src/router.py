import json
from jinja2 import Environment, FileSystemLoader
from src.ollama_client import OllamaClient
from src.tools.curriculum import CurriculumStore

class Router:
    def __init__(self, client: OllamaClient, curriculum: CurriculumStore):
        self.client = client
        self.curriculum = curriculum
        self.templates = Environment(loader=FileSystemLoader("src/prompts"))


    async def handle_message(self, message: str, session: dict) -> str:
        classification = await self._classify(message, session)

        if not classification.get("grade") and session.get("grade"):
            classification["grade"] = session["grade"]

        if not classification.get("subject") and session.get("subject"):
            classification["subject"] = session["subject"]


        match classification["intent"]:
            case "learn":
                return await self._handle_learn(message, classification, session)
            case "practice":
                return await self._handle_practice(message, classification, session)
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


    async def _classify(self, message: str, session: dict) -> dict:
        template = self.templates.get_template("classifier.j2")
        prompt = template.render(
            session_grade = session.get("grade"),
            session_subject = session.get("subject"),
            current_topic = session.get("topic"),
            available_curriculum=self.curriculum.get_available_curriculum(),
        )

        return await self.client.classify(prompt, message)

    async def _handle_learn(self, message: str, classification: dict, session: dict) -> str:
        pass

    async def _handle_practice(self, message: str, classification: dict, session: dict) -> str:
        pass

    async def _handle_exam_prep(self, message: str, classification: dict, session: dict) -> str:
        pass

    async def _handle_progress(self, message: str, classification: dict, session: dict) -> str:
        pass

    async def _handle_greeting(self, message: str, session: dict) -> str:
        pass

    async def _handle_off_topic(self, message: str, session: dict) -> str:
        pass