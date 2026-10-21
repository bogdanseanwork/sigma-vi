"""LiteLLM-backed :class:`~sigma.ai.router.ModelClient`.

Maps LiteLLM's provider-normalised exceptions onto the router's error classes. LiteLLM reads
provider keys from the environment (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY); keys are
never passed through SIGMA code. Install with ``pip install sigma-vi[ai]``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from sigma.ai.roles import ModelSpec
from sigma.ai.router import (
    Completion,
    ContextTooLong,
    Message,
    ProviderUnavailable,
    QuotaExceeded,
    RateLimited,
)
from sigma.core.config import export_dotenv
from sigma.core.security import redact

_QUOTA_HINTS = ("quota", "billing", "insufficient", "usage limit", "credit")
_BUSY_HINTS = ("503", "overloaded", "high demand", "unavailable", "try again later")
_BUSY_RETRY_AFTER_S = 20.0  # temporary provider overload: wait this long before the router retries
_OLLAMA_TIMEOUT_ENV = "SIGMA_OLLAMA_TIMEOUT_S"
_OLLAMA_TIMEOUT_DEFAULT_S = "900"  # a 20B model on CPU can take many minutes per reply


def effective_timeout(provider: str, timeout_s: float) -> float:
    """Local models get a much longer timeout than hosted ones."""
    if provider != "ollama":
        return timeout_s
    try:
        local = float(os.environ.get(_OLLAMA_TIMEOUT_ENV, _OLLAMA_TIMEOUT_DEFAULT_S))
    except ValueError:
        local = float(_OLLAMA_TIMEOUT_DEFAULT_S)
    return max(timeout_s, local)


class LiteLLMClient:
    def __init__(self) -> None:
        try:
            import litellm
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("litellm is not installed; pip install 'sigma-vi[ai]'") from e
        self._litellm = litellm
        export_dotenv()  # make .env keys (e.g. GEMINI_API_KEY) visible to LiteLLM

    def complete(
        self, model: ModelSpec, messages: Sequence[Message], max_output_tokens: int, timeout_s: float
    ) -> Completion:
        lt = self._litellm
        extra: dict[str, Any] = {}
        if model.provider == "ollama":
            extra["api_base"] = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
        try:
            resp: Any = lt.completion(
                model=model.id,
                **extra,
                messages=[dict(m) for m in messages],
                max_tokens=max_output_tokens,
                timeout=effective_timeout(model.provider, timeout_s),
                num_retries=0,  # retries are the router's job
            )
        except lt.exceptions.ContextWindowExceededError as e:
            raise ContextTooLong(redact(str(e))) from None
        except lt.exceptions.RateLimitError as e:
            msg = str(e).lower()
            if any(h in msg for h in _QUOTA_HINTS):
                raise QuotaExceeded(redact(str(e))) from None
            raise RateLimited(redact(str(e))) from None
        except lt.exceptions.ServiceUnavailableError as e:
            # 503 "high demand" spikes are temporary: back off and retry rather than give up on the model
            raise RateLimited(redact(str(e)), retry_after_s=_BUSY_RETRY_AFTER_S) from None
        except lt.exceptions.InternalServerError as e:
            msg = str(e).lower()
            if any(h in msg for h in _BUSY_HINTS):
                raise RateLimited(redact(str(e)), retry_after_s=_BUSY_RETRY_AFTER_S) from None
            raise ProviderUnavailable(redact(str(e))) from None
        except (
            lt.exceptions.AuthenticationError,
            lt.exceptions.PermissionDeniedError,
            lt.exceptions.NotFoundError,  # unknown/retired model id in config/models.toml → fall back
            lt.exceptions.APIConnectionError,
            lt.exceptions.Timeout,
        ) as e:
            raise ProviderUnavailable(redact(str(e))) from None

        usage = getattr(resp, "usage", None)
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
        return Completion(
            text=resp.choices[0].message.content or "",
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cached_tokens=cached,
        )
