"""Train / validation / holdout periods, and the lock on the holdout.

The holdout is for ONE final run of a frozen methodology. Touching it any earlier would let its
results steer design choices, which turns it into more training data. ``HoldoutGuard`` refuses to
hand holdout dates to any experiment unless the caller says why, and records every such request.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


class HoldoutLocked(RuntimeError):
    pass


@dataclass(frozen=True)
class Splits:
    train_end: date = date(2020, 12, 31)
    validation_end: date = date(2023, 12, 31)

    def period_of(self, d: date) -> str:
        if d <= self.train_end:
            return "train"
        return "validation" if d <= self.validation_end else "holdout"


class HoldoutGuard:
    def __init__(self, splits: Splits, ledger: Path):
        self.splits, self.ledger = splits, Path(ledger)

    def require(self, dates: Iterable[date], unlock: str | None = None) -> None:
        held = sorted(d for d in dates if self.splits.period_of(d) == "holdout")
        if not held:
            return
        if not unlock:
            raise HoldoutLocked(
                f"{len(held)} dates fall in the sealed holdout (after {self.splits.validation_end}); "
                "pass unlock='<reason>' only for the final frozen-model run")
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.open("a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now().isoformat(timespec='seconds')}\t{unlock}\t"
                     f"{held[0]}..{held[-1]} ({len(held)} dates)\n")
