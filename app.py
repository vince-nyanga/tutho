import gradio as gr
import os
from logging import getLogger
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


async def chat(message, history, grade, subject, language):
    session = {
        "grade": int(grade),
        "subject": subject,
        "language": language,
        "language_name": LANGUAGE_NAMES.get(language, "English"),
    }
    response = await router.handle_message(message, session, history=history)
    return response


async def process_and_reply(phone: str, message: str):
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
        os.getenv("TWILIO_AUTH_TOKEN")
    )
    twilio_client.messages.create(
        from_=os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886"),
        to=phone,
        body=response_text
    )
    logger.info("Reply sent via Twilio API")


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
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
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

    background_tasks.add_task(process_and_reply, phone, message)
    return Response(content=str(twiml), media_type="application/xml")


if __name__ == "__main__":
    app.launch()