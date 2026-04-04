import gradio as gr
import os
import requests
from logging import getLogger
from requests.auth import HTTPBasicAuth
from fastapi import Form, Request, BackgroundTasks
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioClient
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


def send_typing_indicator(phone: str, message_sid: str):
    try:
        messaging_service_sid = os.getenv("TWILIO_MESSAGING_SERVICE_SID")
        if not messaging_service_sid:
            logger.warning("No TWILIO_MESSAGING_SERVICE_SID set, skipping typing indicator")
            return
        requests.post(
            f"https://messaging.twilio.com/v2/Services/{messaging_service_sid}/Indicators/Typing",
            data={
                "From": os.getenv("TWILIO_WHATSAPP_NUMBER"),
                "To": phone,
                "MessageSid": message_sid,
            },
            auth=HTTPBasicAuth(
                os.getenv("TWILIO_ACCOUNT_SID"),
                os.getenv("TWILIO_AUTH_TOKEN"),
            ),
        )
        logger.info("Typing indicator sent")
    except Exception as e:
        logger.warning(f"Typing indicator failed: {e}")


async def process_and_reply(phone: str, message: str, message_sid: str):
    send_typing_indicator(phone, message_sid)

    session = get_session(phone)
    logger.info(f"Processing message: {message}")
    try:
        response_text = await router.handle_message(message, session, history=session["history"])
        logger.info(f"Response: {response_text[:100]}...")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        response_text = "Sorry, something went wrong. Please try again."

    session["history"].append({"role": "user", "content": message})
    session["history"].append({"role": "assistant", "content": response_text})
    save_session(session)

    twilio_client = TwilioClient(
        os.getenv("TWILIO_ACCOUNT_SID"),
        os.getenv("TWILIO_AUTH_TOKEN"),
    )
    twilio_client.messages.create(
        from_=os.getenv("TWILIO_WHATSAPP_NUMBER"),
        to=phone,
        body=response_text,
    )
    logger.info("Reply sent via Twilio API")


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
            label="Language",
        ),
    ],
    title="Thuto AI",
    description="AI tutor for South African students (CAPS curriculum)",
    type="messages",
)

app = WebhooksServer(ui=demo)


@app.add_webhook("/whatsapp")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    form = await request.form()
    phone = form.get("From", "")
    message = form.get("Body", "").strip()
    message_sid = form.get("MessageSid", "")
    logger.info(f"Incoming WhatsApp from {hash_phone(phone)}: {message}")

    session = get_session(phone)
    twiml = MessagingResponse()

    if parse_command(message, session):
        save_session(session)
        logger.info(f"Command handled: {message}")
        twiml.message("Done! ✓")
        return Response(content=str(twiml), media_type="application/xml")

    background_tasks.add_task(process_and_reply, phone, message, message_sid)
    return Response(content=str(twiml), media_type="application/xml")


if __name__ == "__main__":
    app.launch()