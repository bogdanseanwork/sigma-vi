"""Secret redaction.

Anything that might reach a log line, an exception message, a prompt or a database row passes
through :func:`redact` first. Spec §5: API keys must never appear in prompts, logs, source,
database records or commits.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

REDACTED = "[REDACTED]"

# Environment variables whose *values* are secrets. Kept in one place so config and redaction agree.
SECRET_ENV_VARS: tuple[str, ...] = (
    "DATABASE_URL",
    "MASSIVE_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
    "DALOOPA_API_KEY",
    "FRED_API_KEY",
    "EXA_API_KEY",
    "ALPACA_API_KEY_ID",
    "ALPACA_API_SECRET_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
)

_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"),                 # Anthropic
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}"),           # OpenAI
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),                    # Google
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}"),          # bearer tokens
    re.compile(r"(?i)(api[_-]?key|apikey|secret|token|password)(\s*[=:]\s*)([^\s&'\"]{6,})"),
    re.compile(r"postgres(?:ql)?://[^:\s/]+:[^@\s]+@"),        # credentials inside DSNs
)

_MIN_LITERAL_LEN = 8  # don't redact trivially short env values like "true"


def _env_secret_values(env: Mapping[str, str] | None = None) -> Iterable[str]:
    env = os.environ if env is None else env
    for name in SECRET_ENV_VARS:
        value = env.get(name)
        if value and len(value) >= _MIN_LITERAL_LEN:
            yield value


def redact(text: str, env: Mapping[str, str] | None = None) -> str:
    """Return ``text`` with known secret values and secret-shaped strings replaced."""
    if not text:
        return text
    out = text
    for value in sorted(_env_secret_values(env), key=len, reverse=True):
        out = out.replace(value, REDACTED)
    for pat in _PATTERNS:
        if pat.groups >= 3:
            out = pat.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", out)
        elif "postgres" in pat.pattern:
            out = pat.sub(lambda m: m.group(0).split("://")[0] + f"://{REDACTED}@", out)
        else:
            out = pat.sub(REDACTED, out)
    return out


def redact_obj(obj: Any, env: Mapping[str, str] | None = None) -> Any:
    """Recursively redact strings inside dicts/lists (for JSON payloads bound for logs or the DB)."""
    if isinstance(obj, str):
        return redact(obj, env)
    if isinstance(obj, Mapping):
        return {k: redact_obj(v, env) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return type(obj)(redact_obj(v, env) for v in obj)
    return obj
