"""Configuration and credential discovery.

Reports which integrations are usable *by key presence only* — values are never returned,
printed or logged. Run ``python -m sigma.core.config`` for a status table.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


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
    OLLAMA = "ollama"


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
    Integration.OLLAMA: ("OLLAMA_API_BASE",),  # local models; no key, just the address
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
            for i in (Integration.ANTHROPIC, Integration.OPENAI, Integration.GEMINI, Integration.OLLAMA)
            if self.available(i)
        ]


DEFAULT_DOTENV = Path(__file__).resolve().parents[3] / ".env"

# Values copied verbatim from .env.example are not real credentials.
_PLACEHOLDER_MARKERS = ("you@example.com", "user:password@host")


def read_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env parser: KEY=VALUE lines, '#' comment lines, optional 'export ', optional quotes."""
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key.strip()] = value
    return out


def export_dotenv(
    path: Path | None = None, environ: MutableMapping[str, str] | None = None
) -> list[str]:
    """Copy non-empty .env values into the process environment (existing variables win).

    Third-party clients such as LiteLLM read credentials from the environment, so this runs once at
    startup. Returns the variable *names* added — never values.
    """
    target = os.environ if environ is None else environ
    added = []
    for key, value in read_dotenv(path or DEFAULT_DOTENV).items():
        if value and key not in target:
            target[key] = value
            added.append(key)
    return added


def _is_set(value: str | None) -> bool:
    v = (value or "").strip()
    return bool(v) and not any(marker in v for marker in _PLACEHOLDER_MARKERS)


def load_settings(
    env: Mapping[str, str] | None = None,
    *,
    dotenv_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Settings from ``env`` if given; otherwise the project .env overlaid by the process environment."""
    if env is None:
        merged = read_dotenv(dotenv_path or DEFAULT_DOTENV)
        merged.update(os.environ if environ is None else environ)
        env = merged
    statuses: dict[Integration, IntegrationStatus] = {}
    for integration, names in REQUIRED_ENV.items():
        missing = tuple(n for n in names if not _is_set(env.get(n)))
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
        daily_llm_budget_usd=float(env.get("SIGMA_DAILY_LLM_BUDGET_USD", "0")),
        per_task_max_usd=float(env.get("SIGMA_PER_TASK_MAX_USD", "0")),
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
