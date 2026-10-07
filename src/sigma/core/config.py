"""Configuration and credential discovery.

Reports which integrations are usable *by key presence only* — values are never returned,
printed or logged. Run ``python -m sigma.core.config`` for a status table.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class Integration(StrEnum):
    MASSIVE = "massive"
    ALPHA_VANTAGE = "alpha_vantage"
    SEC_EDGAR = "sec_edgar"
    FRED = "fred"
    EXA = "exa"
    ALPACA_PAPER = "alpaca_paper"
    POSTGRES = "postgres"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"


# Every integration lists the env vars that must all be non-empty for it to be usable.
REQUIRED_ENV: dict[Integration, tuple[str, ...]] = {
    Integration.MASSIVE: ("MASSIVE_API_KEY",),
    Integration.ALPHA_VANTAGE: ("ALPHA_VANTAGE_API_KEY",),
    Integration.SEC_EDGAR: ("SEC_EDGAR_USER_AGENT",),
    Integration.FRED: ("FRED_API_KEY",),
    Integration.EXA: ("EXA_API_KEY",),
    Integration.ALPACA_PAPER: ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"),
    Integration.POSTGRES: ("DATABASE_URL",),
    Integration.ANTHROPIC: ("ANTHROPIC_API_KEY",),
    Integration.OPENAI: ("OPENAI_API_KEY",),
    Integration.GEMINI: ("GEMINI_API_KEY",),
}


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class IntegrationStatus:
    integration: Integration
    available: bool
    missing: tuple[str, ...]  # names of missing env vars — never values


@dataclass(frozen=True)
class Settings:
    statuses: dict[Integration, IntegrationStatus]
    live_trading_enabled: bool
    alpaca_paper: bool
    daily_llm_budget_usd: float
    per_task_max_usd: float

    def available(self, integration: Integration) -> bool:
        return self.statuses[integration].available

    @property
    def model_providers(self) -> list[str]:
        return [
            i.value
            for i in (Integration.ANTHROPIC, Integration.OPENAI, Integration.GEMINI)
            if self.available(i)
        ]


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    statuses: dict[Integration, IntegrationStatus] = {}
    for integration, names in REQUIRED_ENV.items():
        missing = tuple(n for n in names if not (env.get(n) or "").strip())
        statuses[integration] = IntegrationStatus(integration, not missing, missing)

    alpaca_paper = env.get("ALPACA_PAPER", "true").strip().lower() != "false"
    live = _truthy(env.get("SIGMA_LIVE_TRADING"))
    if live and alpaca_paper:
        # Contradictory flags resolve to the safe side.
        live = False
    return Settings(
        statuses=statuses,
        live_trading_enabled=live,
        alpaca_paper=alpaca_paper,
        daily_llm_budget_usd=float(env.get("SIGMA_DAILY_LLM_BUDGET_USD", "25")),
        per_task_max_usd=float(env.get("SIGMA_PER_TASK_MAX_USD", "2")),
    )


def format_status(settings: Settings) -> str:
    lines = ["integration      status       missing env vars", "-" * 60]
    for status in settings.statuses.values():
        flag = "available" if status.available else "UNAVAILABLE"
        lines.append(f"{status.integration.value:<16} {flag:<12} {', '.join(status.missing)}")
    lines.append("-" * 60)
    lines.append(f"live trading enabled: {settings.live_trading_enabled}  (paper: {settings.alpaca_paper})")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    sys.stdout.write(format_status(load_settings()) + "\n")
