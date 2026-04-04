import gradio as gr
import asyncio
import os
from fastapi import FastAPI
from gradio.routes import mount_gradio_app
from src.tools.curriculum import CurriculumStore
from src.router import Router
from src.transformers_client import TransformersClient
from src.server import create_server

curriculum = CurriculumStore()
client = TransformersClient(os.getenv("HF_MODEL", "google/gemma-4-E2B-it"))
router = Router(client, curriculum)

LANGUAGE_NAMES = {
    "en": "English",
    "zu": "isiZulu",
    "xh": "isiXhosa",
    "st": "Sesotho",
    "tn": "Setswana",
    "af": "Afrikaans",
}


async def chat(message, history, grade, subject, language):
    session = {
        "grade": int(grade),
        "subject": subject,
        "language": language,
        "language_name": LANGUAGE_NAMES.get(language, "English"),
    }
    response = await router.handle_message(message, session, history=history)
    return response


demo = gr.ChatInterface(
    fn=chat,
    additional_inputs=[
        gr.Dropdown(choices=["10", "11", "12"], value="12", label="Grade"),
        gr.Dropdown(choices=["Mathematics"], value="Mathematics", label="Subject"),
        gr.Dropdown(
            choices=[
                ("English", "en"),
                ("isiZulu", "zu"),
                ("isiXhosa", "xh"),
                ("Sesotho", "st"),
                ("Setswana", "tn"),
                ("Afrikaans", "af"),
            ],
            value="en",
            label="Language"
        ),
    ],
    title="Thuto AI",
    description="AI tutor for South African students (CAPS curriculum)",
    type="messages",
)

fastapi_app = create_server(curriculum, router)
app = mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    demo.launch()