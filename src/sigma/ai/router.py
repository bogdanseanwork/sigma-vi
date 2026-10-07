"""Model router with budgets, circuit breakers and checkpointed failover (spec §5, §46).

Flow for one task:

1. If the checkpoint store already holds a completed result for ``task_id`` → return it. No call.
2. Estimate input tokens; if above the task-class budget, compress (if a compressor is supplied) or
   refuse. Spending more than the budget is never silent.
3. Walk the role's fallback chain. Skip models whose provider has no credentials, whose circuit is
   open, whose context window is too small, or whose estimated cost breaches the per-task or daily
   ceiling.
4. Call the model. Classify failures:
   - RateLimited      → back off and retry the same model (bounded), then move on
   - QuotaExceeded    → open the provider's circuit for a long cooldown, move on
   - ProviderUnavailable / timeout → count toward the circuit breaker, move on
   - ContextTooLong   → compress once and retry the same model, else move on
5. Every attempt (success or failure) is written to the ledger, with ``fallback_from`` set when a
   previous model in the chain failed.
6. On success the result is checkpointed as completed. If every model fails the task is
   checkpointed as ``parked`` with the failure trail, and :class:`AllModelsFailed` is raised so the
   orchestrator can resume it later without redoing sibling tasks.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from sigma.ai.budgets import BUDGETS, TaskClass, estimate_tokens
from sigma.ai.ledger import InMemoryLedger, LLMCallRecord
from sigma.ai.roles import ModelRegistry, ModelSpec, Role
from sigma.core.security import redact

Message = Mapping[str, str]  # {"role": "system"|"user"|"assistant", "content": "..."}


# ---------------------------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------------------------
class ProviderError(Exception):
    """Base class for errors raised by a ModelClient."""


class RateLimited(ProviderError):
    def __init__(self, msg: str = "rate limited", retry_after_s: float | None = None):
        super().__init__(msg)
        self.retry_after_s = retry_after_s


class QuotaExceeded(ProviderError):
    """Billing / subscription / usage-limit exhaustion. Not worth retrying soon."""


class ProviderUnavailable(ProviderError):
    """Outage, 5xx, timeout, auth failure."""


class ContextTooLong(ProviderError):
    pass


class BudgetExceeded(Exception):
    pass


class AllModelsFailed(Exception):
    def __init__(self, task_id: str, attempts: list[dict[str, Any]]):
        super().__init__(f"all models failed for task {task_id}: {[a['error_class'] for a in attempts]}")
        self.task_id = task_id
        self.attempts = attempts


# ---------------------------------------------------------------------------------------------
# Collaborators
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0


class ModelClient(Protocol):
    def complete(
        self, model: ModelSpec, messages: Sequence[Message], max_output_tokens: int, timeout_s: float
    ) -> Completion: ...


class CheckpointStore(Protocol):
    def load(self, task_id: str) -> dict[str, Any] | None: ...

    def save(self, task_id: str, state: dict[str, Any]) -> None: ...


class InMemoryCheckpointStore:
    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def load(self, task_id: str) -> dict[str, Any] | None:
        state = self._data.get(task_id)
        return dict(state) if state else None

    def save(self, task_id: str, state: dict[str, Any]) -> None:
        self._data[task_id] = dict(state)


@dataclass
class _Circuit:
    failures: int = 0
    open_until: datetime | None = None


@dataclass(frozen=True)
class RouterPolicy:
    max_rate_limit_retries: int = 2
    base_backoff_s: float = 2.0
    max_backoff_s: float = 30.0
    breaker_failure_threshold: int = 3
    breaker_cooldown: timedelta = timedelta(minutes=5)
    quota_cooldown: timedelta = timedelta(hours=1)
    timeout_s: float = 120.0
    per_task_max_usd: float = 2.0
    daily_budget_usd: float = 25.0


@dataclass(frozen=True)
class RouterResult:
    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    from_checkpoint: bool
    fallbacks: list[str] = field(default_factory=list)


Compressor = Callable[[Sequence[Message], int], Sequence[Message]]


class ModelRouter:
    def __init__(
        self,
        registry: ModelRegistry,
        client: ModelClient,
        *,
        available_providers: set[str],
        ledger: InMemoryLedger | None = None,
        checkpoints: CheckpointStore | None = None,
        policy: RouterPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.registry = registry
        self.client = client
        self.available_providers = set(available_providers)
        self.ledger = ledger or InMemoryLedger()
        self.checkpoints = checkpoints or InMemoryCheckpointStore()
        self.policy = policy or RouterPolicy()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._sleep = sleep
        self._circuits: dict[str, _Circuit] = {}

    # -- circuit breaker ------------------------------------------------------------------------
    def _circuit(self, provider: str) -> _Circuit:
        return self._circuits.setdefault(provider, _Circuit())

    def provider_open(self, provider: str) -> bool:
        c = self._circuit(provider)
        if c.open_until and self._clock() < c.open_until:
            return True
        if c.open_until and self._clock() >= c.open_until:
            c.open_until, c.failures = None, 0  # half-open: allow a probe
        return False

    def _record_failure(self, provider: str, cooldown: timedelta | None = None) -> None:
        c = self._circuit(provider)
        c.failures += 1
        if cooldown is not None:
            c.open_until = self._clock() + cooldown
        elif c.failures >= self.policy.breaker_failure_threshold:
            c.open_until = self._clock() + self.policy.breaker_cooldown

    def _record_success(self, provider: str) -> None:
        self._circuits[provider] = _Circuit()

    # -- main entry -----------------------------------------------------------------------------
    def run(
        self,
        *,
        task_id: str,
        agent: str,
        role: Role,
        task_class: TaskClass,
        messages: Sequence[Message],
        compressor: Compressor | None = None,
    ) -> RouterResult:
        if task_class is TaskClass.DETERMINISTIC:
            raise ValueError("deterministic tasks must not call a model (spec §6)")

        state = self.checkpoints.load(task_id)
        if state and state.get("status") == "completed":
            return RouterResult(
                text=state["text"], model=state["model"], provider=state["provider"],
                input_tokens=0, output_tokens=0, cost_usd=0.0, from_checkpoint=True,
                fallbacks=list(state.get("fallbacks", [])),
            )

        budget = BUDGETS[task_class]
        messages = self._fit_budget(messages, budget.max_input_tokens, compressor)
        est_in = estimate_tokens("".join(m["content"] for m in messages))

        attempts: list[dict[str, Any]] = list((state or {}).get("attempts", []))
        failed_models: list[str] = []
        for spec in self.registry.chain(role):
            skip = self._skip_reason(spec, est_in, budget.max_output_tokens)
            if skip:
                attempts.append({"model": spec.id, "error_class": f"skipped:{skip}"})
                continue

            current = messages
            compressed_for_context = False
            retries = 0
            while True:
                started = time.monotonic()
                try:
                    out = self.client.complete(spec, current, budget.max_output_tokens, self.policy.timeout_s)
                except RateLimited as e:
                    self._log(task_id, agent, task_class, role, spec, est_in, 0, started, False,
                              "rate_limited", failed_models)
                    if retries < self.policy.max_rate_limit_retries:
                        retries += 1
                        delay = e.retry_after_s or min(
                            self.policy.base_backoff_s * 2 ** (retries - 1), self.policy.max_backoff_s
                        )
                        self._sleep(delay)
                        continue
                    self._record_failure(spec.provider)
                    attempts.append({"model": spec.id, "error_class": "rate_limited"})
                    break
                except QuotaExceeded:
                    self._log(task_id, agent, task_class, role, spec, est_in, 0, started, False,
                              "quota_exceeded", failed_models)
                    self._record_failure(spec.provider, cooldown=self.policy.quota_cooldown)
                    attempts.append({"model": spec.id, "error_class": "quota_exceeded"})
                    break
                except ContextTooLong:
                    self._log(task_id, agent, task_class, role, spec, est_in, 0, started, False,
                              "context_too_long", failed_models)
                    if compressor and not compressed_for_context:
                        compressed_for_context = True
                        current = compressor(current, max(est_in // 2, 1))
                        continue
                    attempts.append({"model": spec.id, "error_class": "context_too_long"})
                    break
                except ProviderUnavailable as e:
                    self._log(task_id, agent, task_class, role, spec, est_in, 0, started, False,
                              "unavailable", failed_models)
                    self._record_failure(spec.provider)
                    attempts.append({"model": spec.id, "error_class": "unavailable",
                                     "detail": redact(str(e))[:200]})
                    break

                # success
                cost = spec.cost_usd(out.input_tokens, out.output_tokens, out.cached_tokens)
                self._record_success(spec.provider)
                self.ledger.record(LLMCallRecord(
                    task_id=task_id, agent=agent, task_class=task_class.value, role=role.value,
                    provider=spec.provider, model=spec.id, input_tokens=out.input_tokens,
                    output_tokens=out.output_tokens, cached_tokens=out.cached_tokens,
                    latency_ms=int((time.monotonic() - started) * 1000), est_cost_usd=cost,
                    price_verified=spec.price_verified, success=True,
                    fallback_from=failed_models[-1] if failed_models else None,
                ))
                self.checkpoints.save(task_id, {
                    "status": "completed", "text": out.text, "model": spec.id,
                    "provider": spec.provider, "fallbacks": failed_models, "attempts": attempts,
                })
                return RouterResult(
                    text=out.text, model=spec.id, provider=spec.provider,
                    input_tokens=out.input_tokens, output_tokens=out.output_tokens,
                    cost_usd=cost, from_checkpoint=False, fallbacks=list(failed_models),
                )
            failed_models.append(spec.id)

        self.checkpoints.save(task_id, {"status": "parked", "attempts": attempts})
        raise AllModelsFailed(task_id, attempts)

    # -- helpers --------------------------------------------------------------------------------
    def _fit_budget(
        self, messages: Sequence[Message], max_in: int, compressor: Compressor | None
    ) -> Sequence[Message]:
        est = estimate_tokens("".join(m["content"] for m in messages))
        if est <= max_in:
            return messages
        if compressor is None:
            raise BudgetExceeded(f"input ~{est} tokens exceeds budget {max_in} and no compressor given")
        compressed = compressor(messages, max_in)
        est2 = estimate_tokens("".join(m["content"] for m in compressed))
        if est2 > max_in:
            raise BudgetExceeded(f"compressed input ~{est2} tokens still exceeds budget {max_in}")
        return compressed

    def _skip_reason(self, spec: ModelSpec, est_in: int, max_out: int) -> str | None:
        if spec.provider not in self.available_providers:
            return "no_credentials"
        if self.provider_open(spec.provider):
            return "circuit_open"
        if est_in + max_out > spec.context_window:
            return "context_window"
        worst_cost = spec.cost_usd(est_in, max_out)
        if worst_cost > self.policy.per_task_max_usd:
            return "per_task_cost_ceiling"
        if self.ledger.spent_on(self._clock().date()) + worst_cost > self.policy.daily_budget_usd:
            return "daily_cost_ceiling"
        return None

    def _log(
        self, task_id: str, agent: str, task_class: TaskClass, role: Role, spec: ModelSpec,
        est_in: int, out_tokens: int, started: float, success: bool, error_class: str,
        failed_models: list[str],
    ) -> None:
        self.ledger.record(LLMCallRecord(
            task_id=task_id, agent=agent, task_class=task_class.value, role=role.value,
            provider=spec.provider, model=spec.id, input_tokens=0, output_tokens=out_tokens,
            cached_tokens=0, latency_ms=int((time.monotonic() - started) * 1000), est_cost_usd=0.0,
            price_verified=spec.price_verified, success=success, error_class=error_class,
            fallback_from=failed_models[-1] if failed_models else None,
        ))
