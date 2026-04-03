import json
from jinja2 import Environment, FileSystemLoader
from src.ollama_client import OllamaClient
from src.tools.curriculum import CurriculumStore
from src.tools.definitions import create_learning_registry

class Router:
    def __init__(self, client: OllamaClient, curriculum: CurriculumStore):
        self.client = client
        self.curriculum = curriculum
        self.templates = Environment(loader=FileSystemLoader("src/prompts"))
        self.learning_registry = create_learning_registry(curriculum)

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
        topic = None

        if classification.get("topic") and classification.get("grade") and classification.get("subject"):
            topic = self.curriculum.get_topic(
                grade=classification["grade"],
                subject=classification["subject"],
                topic_query=classification["topic"]
            )

        template = self.templates.get_template("tutor.j2")
        system_prompt = template.render(
            grade=classification["grade"],
            subject=classification["subject"],
            language=session.get("language"),
            language_name=session.get("language_name"),
            topic=topic
        )

        tools = self.learning_registry.get_tools()

        messages = [{"role": "user", "content": message}]

        response = await self.client.chat_with_tools(system_prompt, messages, tools)

        if response.tool_calls:
            return await self._execute_tool_loop(response, system_prompt, messages, tools)

        return response.content

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

    async def _execute_tool_loop(self, response, system_prompt: str, messages: list[dict], tools: list[dict]) -> str:
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in response.tool_calls
            ]
        })

        for tool_call in response.tool_calls:
            result = self.learning_registry.execute(tool_call.function.name, tool_call.function.arguments)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str)
            })

        follow_up_response = await self.client.chat_with_tools(system_prompt, messages, tools)

        if follow_up_response.tool_calls:
            return await self._execute_tool_loop(follow_up_response, system_prompt, messages, tools)

        return follow_up_response.content
