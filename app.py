import gradio as gr
import os
import asyncio
from logging import getLogger
from fastapi import Form, Request
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from huggingface_hub import WebhooksServer
from src.tools.curriculum import CurriculumStore
from src.router import Router
from src.transformers_client import TransformersClient
from src.server import init_db, get_session, save_session, parse_command, LANGUAGE_NAMES, hash_phone

logger = getLogger(__name__)

curriculum = CurriculumStore()
client = TransformersClient(os.getenv("HF_MODEL", "google/gemma-4-E2B-it"))
router = Router(client, curriculum)

init_db()

# Warm up model on startup
async def warmup():
    logger.info("Warming up model...")
    try:
        await client.classify("You are a classifier. Return JSON only: {\"intent\": \"greeting\"}", "hello")
        logger.info("Model warmed up.")
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")

asyncio.run(warmup())


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

app = WebhooksServer(ui=demo)


@app.add_webhook("/whatsapp")
async def whatsapp_webhook(request: Request):
    form = await request.form()
    phone = form.get("From", "")
    message = form.get("Body", "").strip()
    logger.info(f"Incoming WhatsApp from {hash_phone(phone)}: {message}")

    session = get_session(phone)
    twiml = MessagingResponse()

    if parse_command(message, session):
        save_session(session)
        logger.info(f"Command handled: {message}")
        twiml.message("Done! ✓")
        return Response(content=str(twiml), media_type="application/xml")

    try:
        response_text = await router.handle_message(message, session, history=session["history"])
        logger.info(f"Response: {response_text[:100]}...")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        response_text = "Sorry, something went wrong. Please try again."

    session["history"].append({"role": "user", "content": message})
    session["history"].append({"role": "assistant", "content": response_text})
    save_session(session)
    twiml.message(response_text)
    return Response(content=str(twiml), media_type="application/xml")


if __name__ == "__main__":
    app.launch()