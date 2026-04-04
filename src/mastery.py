import sqlite3
from dataclasses import dataclass
from logging import getLogger

logger = getLogger(__name__)

DB_PATH = "/data/thuto.db"

MASTERY_LEVELS = {
    "not_started": (0.0, 0.30),
    "developing": (0.30, 0.60),
    "proficient": (0.60, 0.85),
    "mastered": (0.85, 1.0),
}

GUESS_RATES = {
    "open": 0.05,
    "mcq": 0.25,
    "conversation": 0.10,
}

@dataclass
class KCMastery:
    phone_hash: str
    kc_code: str
    p_mastery: float
    attempts: int
    correct: int

    @property
    def level(self) -> str:
        for label, (low, high) in MASTERY_LEVELS.items():
            if low <= self.p_mastery < high:
                return label
        return "mastered"


def get_mastery(phone_hash: str, kc_code: str, default_p_l0: float = 0.10) -> KCMastery:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT p_mastery, attempts, correct FROM mastery WHERE phone_hash = ? AND kc_code = ?",
        (phone_hash, kc_code),
    ).fetchone()
    conn.close()
    if row:
        return KCMastery(phone_hash, kc_code, row[0], row[1], row[2])
    return KCMastery(phone_hash, kc_code, default_p_l0, 0, 0)

def get_all_mastery(phone_hash: str) -> list[KCMastery]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT kc_code, p_mastery, attempts, correct FROM mastery WHERE phone_hash = ?",
        (phone_hash,),
    ).fetchall()
    conn.close()
    return [KCMastery(phone_hash, row[0], row[1], row[2], row[3]) for row in rows]

def save_mastery(mastery: KCMastery):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO mastery (phone_hash, kc_code, p_mastery, attempts, correct)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(phone_hash, kc_code) DO UPDATE SET
            p_mastery=excluded.p_mastery,
            attempts=excluded.attempts,
            correct=excluded.correct
    """, (
        mastery.phone_hash,
        mastery.kc_code,
        mastery.p_mastery,
        mastery.attempts,
        mastery.correct,
    ))
    conn.commit()
    conn.close()


def bkt_update(
    prior: float,
    is_correct: bool,
    slip_rate: float = 0.1,
    learning_rate: float = 0.15,
    question_type: str = "conversation",
) -> float:
    p_s = slip_rate
    p_g = GUESS_RATES.get(question_type, 0.10)
    p_t = learning_rate

    if is_correct:
        p_correct_known = 1 - p_s
        p_correct_unknown = p_g
        posterior = (prior * p_correct_known) / (
            prior * p_correct_known + (1 - prior) * p_correct_unknown
        )
    else:
        p_incorrect_known = p_s
        p_incorrect_unknown = 1 - p_g
        posterior = (prior * p_incorrect_known) / (
            prior * p_incorrect_known + (1 - prior) * p_incorrect_unknown
        )

    updated = posterior + (1 - posterior) * p_t
    return round(min(updated, 0.99), 4)


def update_mastery(phone_hash: str, kc_code: str, is_correct: bool,
                   slip_rate: float = 0.1, learning_rate: float = 0.15,
                   default_p_l0: float = 0.1, question_type: str = "conversation") -> KCMastery:
    mastery = get_mastery(phone_hash, kc_code, default_p_l0)
    mastery.p_mastery = bkt_update(
        mastery.p_mastery, is_correct, slip_rate, learning_rate, question_type
    )
    mastery.attempts += 1
    if is_correct:
        mastery.correct += 1
    save_mastery(mastery)
    logger.info(f"Mastery updated: {kc_code} -> {mastery.p_mastery:.4f} ({mastery.level})")
    return mastery