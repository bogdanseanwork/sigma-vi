"""LiteLLM-backed :class:`~sigma.ai.router.ModelClient`.

Maps LiteLLM's provider-normalised exceptions onto the router's error classes. LiteLLM reads
provider keys from the environment (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY); keys are
never passed through SIGMA code. Install with ``pip install sigma-vi[ai]``.
"""

from __future__ import annotations

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
from sigma.core.security import redact

_QUOTA_HINTS = ("quota", "billing", "insufficient", "usage limit", "credit")


class LiteLLMClient:
    def __init__(self) -> None:
        try:
            import litellm
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("litellm is not installed; pip install 'sigma-vi[ai]'") from e
        self._litellm = litellm

    def complete(
        self, model: ModelSpec, messages: Sequence[Message], max_output_tokens: int, timeout_s: float
    ) -> Completion:  # pragma: no cover — exercised against live providers on the deployment host
        lt = self._litellm
        try:
            resp: Any = lt.completion(
                model=model.id,
                messages=[dict(m) for m in messages],
                max_tokens=max_output_tokens,
                timeout=timeout_s,
                num_retries=0,  # retries are the router's job
            )
        except lt.exceptions.ContextWindowExceededError as e:
            raise ContextTooLong(redact(str(e))) from None
        except lt.exceptions.RateLimitError as e:
            msg = str(e).lower()
            if any(h in msg for h in _QUOTA_HINTS):
                raise QuotaExceeded(redact(str(e))) from None
            raise RateLimited(redact(str(e))) from None
        except (
            lt.exceptions.AuthenticationError,
            lt.exceptions.PermissionDeniedError,
            lt.exceptions.NotFoundError,  # unknown/retired model id in config/models.toml → fall back

            lt.exceptions.ServiceUnavailableError,
            lt.exceptions.APIConnectionError,
            lt.exceptions.Timeout,
            lt.exceptions.InternalServerError,
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
