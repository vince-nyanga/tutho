
import logging
logging.basicConfig(level=logging.INFO)

import gradio as gr
import os
import requests
from logging import getLogger
from requests.auth import HTTPBasicAuth
from fastapi import Request, BackgroundTasks
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

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")


async def process_and_reply(phone, message, message_sid):
    logger.info(f"Background task started for {hash_phone(phone)}")
    try:
        session = get_session(phone)

        if parse_command(message, session):
            reply_text = (
                f"Updated! Grade: {session['grade']}, "
                f"Subject: {session['subject']}, "
                f"Language: {session['language_name']}"
            )
        else:
            reply_text = await router.handle_message(
                message, session, history=session["history"]
            )
            session["history"].append({"role": "user", "content": message})
            session["history"].append({"role": "assistant", "content": reply_text})

        save_session(session)

        logger.info(f"Sending reply to {hash_phone(phone)}: {reply_text[:100]}...")
        url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
        resp = requests.post(
            url,
            data={
                "From": TWILIO_WHATSAPP_NUMBER,
                "To": phone,
                "Body": reply_text,
            },
            auth=HTTPBasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
        )
        logger.info(f"Twilio send response: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        logger.error(f"Error in background task: {e}", exc_info=True)


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
        gr.Dropdown(choices=["7", "8", "9", "10", "11", "12"], value="10", label="Grade"),
        gr.Dropdown(choices=["Mathematics", "Physical Sciences"], value="Mathematics", label="Subject"),
        gr.Dropdown(
            choices=[("English", "en"), ("isiZulu", "zu"), ("Sesotho", "st"),
                     ("Setswana", "tn"), ("isiXhosa", "xh"), ("Afrikaans", "af")],
            value="en", label="Language",
        ),
    ],
    title="Thuto AI",
    description="AI tutor for South African CAPS curriculum",
    type="messages",
)
demo.queue()

app = WebhooksServer(ui=demo)


@app.add_webhook("/whatsapp")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    form = await request.form()
    phone = form.get("From", "")
    message = form.get("Body", "")
    message_sid = form.get("MessageSid", "")

    logger.info(f"WhatsApp webhook from {hash_phone(phone)}: {message[:50]}")

    background_tasks.add_task(process_and_reply, phone, message, message_sid)

    twiml = MessagingResponse()
    return Response(content=str(twiml), media_type="application/xml")


if __name__ == "__main__":
    app.launch(ssr_mode=False)