import streamlit as st
import asyncio
from src.ollama_client import OllamaClient
from src.router import Router
from src.tools.curriculum import CurriculumStore

st.title("Tutho AI")
@st.cache_resource
def init_router():
    client = OllamaClient()
    curriculum = CurriculumStore()
    return Router(client, curriculum)

router = init_router()

if "messages" not in st.session_state:
    st.session_state["messages"] = []


if "user_session" not in st.session_state:
    st.session_state.user_session = {
        "grade": 12,
        "subject": None,
        "language": "zu",
        "language_name": "IsiZulu",
        "topic": None
    }

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What do you want to learn today?"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Proocessing..."):
            response = asyncio.run(
              router.handle_message(prompt, st.session_state.user_session)
            )

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
