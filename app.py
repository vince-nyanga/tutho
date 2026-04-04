import gradio as gr
import os
from logging import getLogger
from fastapi import Form
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from src.tools.curriculum import CurriculumStore
from src.router import Router
from src.transformers_client import TransformersClient
from src.server import init_db, get_session, save_session, parse_command, LANGUAGE_NAMES, hash_phone

logger = getLogger(__name__)

curriculum = CurriculumStore()
client = TransformersClient(os.getenv("HF_MODEL", "google/gemma-4-E2B-it"))
router = Router(client, curriculum)

init_db()


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


with gr.WebhooksServer(ui=demo) as webhooks_server:
    @webhooks_server.post("/webhook/whatsapp")
    async def whatsapp_webhook(From: str = Form(...), Body: str = Form(...)):
        logger.info(f"Incoming WhatsApp from {hash_phone(From)}: {Body.strip()}")
        twiml = MessagingResponse()
        twiml.message(f"Echo: {Body.strip()}")
        return Response(content=str(twiml), media_type="application/xml")


if __name__ == "__main__":
    webhooks_server.launch()