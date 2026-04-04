import json
import sqlite3
import hashlib
import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse
from logging import getLogger

logger = getLogger(__name__)

DB_PATH = "sessions.db"

LANGUAGE_NAMES = {
    "en": "English",
    "zu": "isiZulu",
    "xh": "isiXhosa",
    "st": "Sesotho",
    "tn": "Setswana",
    "af": "Afrikaans",
}


def hash_phone(phone: str) -> str:
    return hashlib.sha256(phone.encode()).hexdigest()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            phone_hash TEXT PRIMARY KEY,
            history TEXT DEFAULT '[]',
            grade INTEGER DEFAULT 12,
            subject TEXT DEFAULT 'Mathematics',
            language TEXT DEFAULT 'en',
            language_name TEXT DEFAULT 'English'
        )
    """)
    conn.commit()
    conn.close()


def get_session(phone: str) -> dict:
    phone_hash = hash_phone(phone)
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT * FROM sessions WHERE phone_hash = ?", (phone_hash,)
    ).fetchone()
    conn.close()
    if row:
        return {
            "phone_hash": row[0],
            "history": json.loads(row[1]),
            "grade": row[2],
            "subject": row[3],
            "language": row[4],
            "language_name": row[5],
        }
    return {
        "phone_hash": phone_hash,
        "history": [],
        "grade": 12,
        "subject": "Mathematics",
        "language": "en",
        "language_name": "English",
    }


def save_session(session: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO sessions (phone_hash, history, grade, subject, language, language_name)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(phone_hash) DO UPDATE SET
            history=excluded.history,
            grade=excluded.grade,
            subject=excluded.subject,
            language=excluded.language,
            language_name=excluded.language_name
    """, (
        session["phone_hash"],
        json.dumps(session["history"][-20:]),
        session["grade"],
        session["subject"],
        session["language"],
        session["language_name"],
    ))
    conn.commit()
    conn.close()


def parse_command(message: str, session: dict) -> bool:
    """Handle special commands like /lang zu or /grade 11. Returns True if handled."""
    if not message.startswith("/"):
        return False

    parts = message.strip().split()
    command = parts[0].lower()

    if command == "/lang" and len(parts) == 2:
        lang_code = parts[1].lower()
        if lang_code in LANGUAGE_NAMES:
            session["language"] = lang_code
            session["language_name"] = LANGUAGE_NAMES[lang_code]
            return True

    if command == "/grade" and len(parts) == 2:
        try:
            session["grade"] = int(parts[1])
            return True
        except ValueError:
            pass

    if command == "/reset":
        session["history"] = []
        return True

    return False


def create_server(curriculum, router) -> FastAPI:
    init_db()
    server = FastAPI()

    @server.post("/webhook/whatsapp", response_class=PlainTextResponse)
    async def whatsapp_webhook(
        From: str = Form(...),
        Body: str = Form(...),
    ):
        phone = From
        message = Body.strip()
        logger.info(f"Incoming message from {hash_phone(phone)}: {message}")

        session = get_session(phone)
        history = session["history"]

        twiml = MessagingResponse()

        # Handle commands
        if parse_command(message, session):
            save_session(session)
            twiml.message("Done! ✓")
            return str(twiml)

        try:
            response_text = await router.handle_message(message, session, history=history)
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            response_text = "Sorry, something went wrong. Please try again."

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response_text})
        session["history"] = history
        save_session(session)

        twiml.message(response_text)
        return str(twiml)

    return server