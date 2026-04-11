LANGUAGE_NAMES = {
    "en": "English",
    "zu": "isiZulu",
    "xh": "isiXhosa",
    "st": "Sesotho",
    "tn": "Setswana",
    "af": "Afrikaans",
}


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
        session["current_topic"] = None
        session["current_grade"] = None
        session["current_subject"] = None
        session["current_kc"] = None
        return True

    return False
