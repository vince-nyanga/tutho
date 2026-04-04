
import gradio as gr
import asyncio
import os
from src.tools.curriculum import CurriculumStore
from src.router import Router
from src.transformers_client import TransformersClient


curriculum = CurriculumStore()
client = TransformersClient(os.getenv("HF_MODEL", "google/gemma-4-E2B-it"))
router = Router(client, curriculum)


async def chat(message, history, grade, subject, language):
    session = {
        "grade": int(grade),
        "subject": subject,
        "language": language,
    }
    response = await router.handle_message(message, session, history=history)
    return response


with gr.Blocks(title="Thuto AI") as demo:
    gr.Markdown("# Thuto AI\nAI tutor for South African students (CAPS curriculum)")

    grade = gr.Dropdown(choices=["10", "11", "12"], value="12", label="Grade")
    subject = gr.Dropdown(choices=["Mathematics"], value="Mathematics", label="Subject")
    language = gr.Dropdown(
        choices=[("English", "en"), ("isiZulu", "zu"), ("Afrikaans", "af")],
        value="en",
        label="Language"
    )

    gr.ChatInterface(
        fn=lambda msg, hist: asyncio.run(chat(msg, hist, grade.value, subject.value, language.value)),
        chatbot=gr.Chatbot(height=500, type="messages"),
        textbox=gr.Textbox(placeholder="Ask me anything about your studies...", scale=7),
        type="messages",
    )


if __name__ == "__main__":
    demo.launch()