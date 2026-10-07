"""AI cost accounting (spec §45).

The in-memory ledger is used by tests and by the worker as a write-through buffer; the Postgres
implementation (``sigma.data.repositories.llm_calls``) persists the same records to ``llm_calls``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime


@dataclass(frozen=True)
class LLMCallRecord:
    task_id: str
    agent: str
    task_class: str
    role: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    latency_ms: int
    est_cost_usd: float
    price_verified: bool
    success: bool
    error_class: str | None = None
    fallback_from: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class InMemoryLedger:
    def __init__(self, clock: Callable[[], datetime] | None = None):
        self._records: list[LLMCallRecord] = []
        self._clock = clock or (lambda: datetime.now(UTC))

    def record(self, rec: LLMCallRecord) -> None:
        self._records.append(rec)

    @property
    def records(self) -> list[LLMCallRecord]:
        return list(self._records)

    def spent_on(self, day: date | None = None) -> float:
        day = day or self._clock().date()
        return sum(r.est_cost_usd for r in self._records if r.created_at.date() == day)

    def by_agent(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = defaultdict(
            lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "failures": 0}
        )
        for r in self._records:
            row = out[r.agent]
            row["calls"] += 1
            row["input_tokens"] += r.input_tokens
            row["output_tokens"] += r.output_tokens
            row["cost_usd"] += r.est_cost_usd
            row["failures"] += 0 if r.success else 1
        return dict(out)

    def fallback_count(self) -> int:
        return sum(1 for r in self._records if r.fallback_from and r.success)
