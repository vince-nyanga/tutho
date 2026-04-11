from dataclasses import dataclass

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
