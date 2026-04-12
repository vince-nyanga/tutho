import logging
logging.basicConfig(level=logging.INFO)

import asyncio
import os
import streamlit as st
from src.router import Router
from src.tools.curriculum import CurriculumStore
from src.db import init_db

st.set_page_config(page_title="Thuto AI", page_icon="📚")
st.title("Thuto AI")


def get_model_client():
    backend = os.getenv("MODEL_BACKEND", "ollama")
    if backend == "transformers":
        from src.transformers_client import TransformersClient
        return TransformersClient(os.getenv("HF_MODEL", "google/gemma-4-E4B-it"))
    from src.local_client import LocalClient
    return LocalClient(os.getenv("OLLAMA_MODEL", "gemma4:e4b"))


@st.cache_resource
def init_router():
    init_db()
    client = get_model_client()
    curriculum = CurriculumStore()
    return Router(client, curriculum)


router = init_router()

with st.sidebar:
    if st.button("Clear Chat"):
        st.session_state["messages"] = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "user_session" not in st.session_state:
    st.session_state.user_session = {
        "grade": 12,
        "subject": "Mathematics",
        "current_topic": None,
        "phone_hash": "local_dev_user",
    }

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What do you want to learn today?"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = asyncio.run(
                router.handle_message(
                    prompt,
                    st.session_state["user_session"],
                    history=st.session_state["messages"],
                )
            )

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})