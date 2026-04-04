import gradio as gr
import os
from fastapi import Form
from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse
from src.tools.curriculum import CurriculumStore
from src.router import Router
from src.transformers_client import TransformersClient
from src.server import init_db, get_session, save_session, parse_command, LANGUAGE_NAMES

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


@demo.app.post("/webhook/whatsapp", response_class=PlainTextResponse)
async def whatsapp_webhook(From: str = Form(...), Body: str = Form(...)):
    session = get_session(From)
    message = Body.strip()
    twiml = MessagingResponse()

    if parse_command(message, session):
        save_session(session)
        twiml.message("Done! ✓")
        return str(twiml)

    try:
        response_text = await router.handle_message(message, session, history=session["history"])
    except Exception as e:
        response_text = "Sorry, something went wrong. Please try again."

    session["history"].append({"role": "user", "content": message})
    session["history"].append({"role": "assistant", "content": response_text})
    save_session(session)
    twiml.message(response_text)
    return str(twiml)


if __name__ == "__main__":
    demo.launch()