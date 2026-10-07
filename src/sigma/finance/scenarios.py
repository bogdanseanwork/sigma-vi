"""Bear / base / bull scenario analysis with explicit probabilities (spec §15)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

_PROB_TOL = 1e-6


@dataclass(frozen=True)
class Scenario:
    name: str
    probability: float
    value_per_share: float


@dataclass(frozen=True)
class ScenarioAnalysis:
    expected_value: float
    expected_return: float
    downside_return: float       # worst scenario vs price
    upside_return: float         # best scenario vs price
    prob_loss: float
    reward_to_risk: float        # probability-weighted gains / probability-weighted losses (inf if no loss)


def analyze(scenarios: Sequence[Scenario], price: float) -> ScenarioAnalysis:
    if price <= 0:
        raise ValueError("price must be positive")
    if not scenarios:
        raise ValueError("need at least one scenario")
    if any(not 0 <= s.probability <= 1 for s in scenarios):
        raise ValueError("each probability must be within [0, 1]")
    total = sum(s.probability for s in scenarios)
    if abs(total - 1.0) > _PROB_TOL:
        raise ValueError(f"scenario probabilities must sum to 1 (got {total:.6f})")

    ev = sum(s.probability * s.value_per_share for s in scenarios)
    gains = sum(s.probability * max(s.value_per_share - price, 0.0) for s in scenarios)
    losses = sum(s.probability * max(price - s.value_per_share, 0.0) for s in scenarios)
    values = [s.value_per_share for s in scenarios]
    return ScenarioAnalysis(
        expected_value=ev,
        expected_return=ev / price - 1.0,
        downside_return=min(values) / price - 1.0,
        upside_return=max(values) / price - 1.0,
        prob_loss=sum(s.probability for s in scenarios if s.value_per_share < price),
        reward_to_risk=float("inf") if losses == 0 else gains / losses,
    )
