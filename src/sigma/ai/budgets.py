"""Token budgets per task class (spec §46)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class TaskClass(StrEnum):
    DETERMINISTIC = "deterministic"
    RETRIEVAL = "retrieval"
    SCREEN = "screen"
    SPECIALIST = "specialist"
    DEBATE = "debate"
    COMMITTEE = "committee"
    CAPITAL_DECISION = "capital_decision"


@dataclass(frozen=True)
class Budget:
    max_input_tokens: int
    max_output_tokens: int


BUDGETS: dict[TaskClass, Budget] = {
    TaskClass.DETERMINISTIC: Budget(0, 0),
    TaskClass.RETRIEVAL: Budget(2_000, 500),
    TaskClass.SCREEN: Budget(4_000, 800),
    TaskClass.SPECIALIST: Budget(24_000, 3_000),
    TaskClass.DEBATE: Budget(16_000, 3_000),
    TaskClass.COMMITTEE: Budget(40_000, 6_000),
    TaskClass.CAPITAL_DECISION: Budget(80_000, 10_000),
}


def estimate_tokens(text: str) -> int:
    """Cheap, slightly conservative token estimate (3.5 chars/token; real English + JSON averages ~4).

    Used only for pre-flight budget checks; the ledger records the provider's actual counts.
    """
    return math.ceil(len(text) / 3.5) if text else 0
