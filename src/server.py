import json
import hashlib
import os
from logging import getLogger

logger = getLogger(__name__)

TURSO_DATABASE_URL = os.environ.get("TURSO_DATABASE_URL", "")
TURSO_AUTH_TOKEN = os.environ.get("TURSO_AUTH_TOKEN", "")

if TURSO_DATABASE_URL and TURSO_AUTH_TOKEN:
    import libsql_experimental as dblib
    logger.info("Using Turso database")
else:
    import sqlite3 as dblib
    logger.info("Using local SQLite database")

DB_PATH = "thuto.db"

LANGUAGE_NAMES = {
    "en": "English",
    "zu": "isiZulu",
    "xh": "isiXhosa",
    "st": "Sesotho",
    "tn": "Setswana",
    "af": "Afrikaans",
}


def get_connection():
    if TURSO_DATABASE_URL and TURSO_AUTH_TOKEN:
        conn = dblib.connect(DB_PATH, sync_url=TURSO_DATABASE_URL, auth_token=TURSO_AUTH_TOKEN)
        conn.sync()
        return conn
    else:
        return dblib.connect(DB_PATH)


def hash_phone(phone: str) -> str:
    return hashlib.sha256(phone.encode()).hexdigest()


def init_db():
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            phone_hash TEXT PRIMARY KEY,
            history TEXT DEFAULT '[]',
            grade INTEGER DEFAULT 12,
            subject TEXT DEFAULT 'Mathematics',
            language TEXT DEFAULT 'en',
            language_name TEXT DEFAULT 'English'
        );

        CREATE TABLE IF NOT EXISTS mastery (
            phone_hash TEXT,
            kc_code TEXT,
            p_mastery REAL DEFAULT 0.1,
            attempts INTEGER DEFAULT 0,
            correct INTEGER DEFAULT 0,
            PRIMARY KEY (phone_hash, kc_code)
        );
    """)
    conn.commit()
    conn.close()
    logger.info(f"Database initialized (turso={bool(TURSO_DATABASE_URL)})")


def get_session(phone: str) -> dict:
    phone_hash = hash_phone(phone)
    conn = get_connection()
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
    conn = get_connection()
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