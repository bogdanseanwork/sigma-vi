import unittest
from datetime import UTC, datetime, timedelta

from sigma.ai.budgets import TaskClass
from sigma.ai.ledger import InMemoryLedger
from sigma.ai.roles import ModelRegistry, ModelSpec, Role
from sigma.ai.router import (
    AllModelsFailed,
    BudgetExceeded,
    Completion,
    ContextTooLong,
    InMemoryCheckpointStore,
    ModelRouter,
    ProviderUnavailable,
    QuotaExceeded,
    RateLimited,
    RouterPolicy,
)


def spec(key, provider, ctx=200_000, pin=1.0, pout=5.0):
    return ModelSpec(key, f"{provider}/{key}", provider, ctx, pin, pout, True)


def registry():
    models = {
        "a": spec("a", "anthropic"),
        "o": spec("o", "openai"),
        "g": spec("g", "gemini"),
    }
    chains = {r: ["a", "o", "g"] for r in Role}
    return ModelRegistry(models, chains)


class ScriptedClient:
    """Returns/raises according to a per-model script; records calls."""

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []

    def complete(self, model, messages, max_output_tokens, timeout_s):
        self.calls.append(model.key)
        step = self.script[model.key].pop(0)
        if isinstance(step, Exception):
            raise step
        return Completion(step, input_tokens=100, output_tokens=50)


class Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 7, 12, tzinfo=UTC)

    def __call__(self):
        return self.now


MSG = [{"role": "user", "content": "analyse revenue quality"}]

# Tests of the paid path opt out of the $0 defaults explicitly.
PAID = {"free_only": False, "per_task_max_usd": 2.0, "daily_budget_usd": 25.0}


def make_router(client, providers=("anthropic", "openai", "gemini"), **kw):
    clock = kw.pop("clock", Clock())
    sleeps = kw.pop("sleeps", [])
    return ModelRouter(
        registry(), client, available_providers=set(providers),
        ledger=kw.pop("ledger", InMemoryLedger(clock)),
        checkpoints=kw.pop("checkpoints", InMemoryCheckpointStore()),
        policy=kw.pop("policy", RouterPolicy(**PAID)), clock=clock, sleep=sleeps.append,
    )


class RouterTests(unittest.TestCase):
    def run_task(self, router, task_id="t1", **kw):
        return router.run(task_id=task_id, agent="fundamental", role=Role.RESEARCH,
                          task_class=kw.pop("task_class", TaskClass.SPECIALIST),
                          messages=kw.pop("messages", MSG), **kw)

    def test_primary_success(self):
        c = ScriptedClient({"a": ["ok"]})
        r = self.run_task(make_router(c))
        self.assertEqual((r.text, r.provider, r.fallbacks), ("ok", "anthropic", []))
        self.assertAlmostEqual(r.cost_usd, (100 * 1.0 + 50 * 5.0) / 1e6)

    def test_quota_falls_back_and_opens_circuit(self):
        c = ScriptedClient({"a": [QuotaExceeded("usage limit")], "o": ["from openai", "again"]})
        router = make_router(c)
        r = self.run_task(router)
        self.assertEqual(r.provider, "openai")
        self.assertEqual(r.fallbacks, ["anthropic/a"])
        self.assertTrue(router.provider_open("anthropic"))
        # Second task skips anthropic entirely while circuit is open.
        r2 = self.run_task(router, task_id="t2")
        self.assertEqual(r2.provider, "openai")
        self.assertEqual(c.calls, ["a", "o", "o"])
        success = [x for x in router.ledger.records if x.success]
        self.assertEqual(success[0].fallback_from, "anthropic/a")

    def test_circuit_half_opens_after_cooldown(self):
        clock = Clock()
        c = ScriptedClient({"a": [QuotaExceeded(), "back"], "o": ["x"]})
        router = make_router(c, clock=clock)
        self.run_task(router)
        clock.now += timedelta(hours=1, seconds=1)
        self.assertEqual(self.run_task(router, task_id="t2").provider, "anthropic")

    def test_failed_half_open_probe_reopens_circuit_immediately(self):
        clock = Clock()
        down = [ProviderUnavailable()] * 3
        c = ScriptedClient({"a": [*down, ProviderUnavailable()], "o": ["x"] * 5})
        router = make_router(c, clock=clock)
        for i in range(3):  # three outages trip the breaker
            self.run_task(router, task_id=f"t{i}")
        self.assertTrue(router.provider_open("anthropic"))
        clock.now += RouterPolicy().breaker_cooldown + timedelta(seconds=1)
        self.run_task(router, task_id="probe")  # half-open probe to anthropic fails once
        self.assertEqual(c.calls.count("a"), 4)
        self.assertTrue(router.provider_open("anthropic"))  # one failed probe is enough to reopen

    def test_daily_budget_ceiling_skips_models(self):
        clock = Clock()
        ledger = InMemoryLedger(clock)
        from sigma.ai.ledger import LLMCallRecord
        ledger.record(LLMCallRecord("old", "x", "specialist", "RESEARCH", "anthropic", "m", 0, 0, 0, 0,
                                    est_cost_usd=24.9999, price_verified=True, success=True,
                                    created_at=clock.now))
        router = make_router(ScriptedClient({}), clock=clock, ledger=ledger,
                             policy=RouterPolicy(**PAID))
        with self.assertRaises(AllModelsFailed) as ctx:
            self.run_task(router)
        self.assertTrue(all("daily_cost_ceiling" in a["error_class"] for a in ctx.exception.attempts))

    def test_rate_limit_retries_then_succeeds(self):
        sleeps = []
        c = ScriptedClient({"a": [RateLimited(), RateLimited(retry_after_s=7), "ok"]})
        r = self.run_task(make_router(c, sleeps=sleeps))
        self.assertEqual(r.provider, "anthropic")
        self.assertEqual(sleeps, [2.0, 7])

    def test_rate_limit_exhausted_moves_on(self):
        c = ScriptedClient({"a": [RateLimited()] * 3, "o": ["ok"]})
        self.assertEqual(self.run_task(make_router(c)).provider, "openai")

    def test_context_too_long_compresses_once(self):
        compressed = []

        def compressor(msgs, target):
            compressed.append(target)
            return [{"role": "user", "content": "short"}]

        c = ScriptedClient({"a": [ContextTooLong(), "ok"]})
        r = self.run_task(make_router(c), compressor=compressor)
        self.assertEqual(r.provider, "anthropic")
        self.assertEqual(len(compressed), 1)

    def test_missing_credentials_skipped(self):
        c = ScriptedClient({"g": ["ok"]})
        r = self.run_task(make_router(c, providers=("gemini",)))
        self.assertEqual(r.provider, "gemini")
        self.assertEqual(c.calls, ["g"])

    def test_all_fail_parks_task_and_resume_does_not_redo_completed(self):
        store = InMemoryCheckpointStore()
        c = ScriptedClient({
            "a": [ProviderUnavailable("503 api_key=SHOULD_NOT_LEAK_123")],
            "o": [ProviderUnavailable()],
            "g": [ProviderUnavailable()],
        })
        router = make_router(c, checkpoints=store)
        with self.assertRaises(AllModelsFailed) as ctx:
            self.run_task(router)
        self.assertEqual(store.load("t1")["status"], "parked")
        self.assertNotIn("SHOULD_NOT_LEAK_123", str(ctx.exception.attempts))

        # Resume: provider recovered.
        c.script["a"] = ["recovered"]
        r = self.run_task(router)
        self.assertEqual(r.text, "recovered")
        # Re-running a completed task returns the checkpoint without a model call.
        n = len(c.calls)
        r2 = self.run_task(router)
        self.assertTrue(r2.from_checkpoint)
        self.assertEqual(len(c.calls), n)

    def test_budget_enforced(self):
        big = [{"role": "user", "content": "x" * 200_000}]
        with self.assertRaises(BudgetExceeded):
            self.run_task(make_router(ScriptedClient({})), messages=big)

    def test_cost_ceilings(self):
        pricey = RouterPolicy(**{**PAID, "per_task_max_usd": 0.000001})
        with self.assertRaises(AllModelsFailed) as ctx:
            self.run_task(make_router(ScriptedClient({}), policy=pricey))
        self.assertTrue(all("per_task_cost_ceiling" in a["error_class"] for a in ctx.exception.attempts))

    def test_deterministic_tasks_refused(self):
        with self.assertRaises(ValueError):
            self.run_task(make_router(ScriptedClient({})), task_class=TaskClass.DETERMINISTIC)


class FreeOnlyTests(unittest.TestCase):
    def free_registry(self):
        models = {
            "paid": ModelSpec("paid", "anthropic/x", "anthropic", 200_000, 3.0, 15.0, True, free=False),
            "unpriced": ModelSpec("unpriced", "openai/y", "openai", 200_000, 0.0, 0.0, False, free=False),
            "gem": ModelSpec("gem", "gemini/z", "gemini", 1_000_000, 0.0, 0.0, True, free=True),
        }
        return ModelRegistry(models, {r: ["paid", "unpriced", "gem"] for r in Role})

    def router(self, client, free_only=True):
        return ModelRouter(self.free_registry(), client,
                           available_providers={"anthropic", "openai", "gemini"},
                           policy=RouterPolicy(**{**PAID, "free_only": free_only}), sleep=lambda s: None)

    def run_task(self, router):
        return router.run(task_id="t", agent="a", role=Role.RESEARCH, task_class=TaskClass.SPECIALIST,
                          messages=MSG)

    def test_free_only_skips_paid_and_unpriced_models(self):
        c = ScriptedClient({"gem": ["ok"]})
        r = self.run_task(self.router(c))
        self.assertEqual(r.provider, "gemini")
        self.assertEqual(c.calls, ["gem"])
        self.assertEqual(r.cost_usd, 0.0)

    def test_defaults_are_zero_cost(self):
        p = RouterPolicy()
        self.assertEqual((p.free_only, p.per_task_max_usd, p.daily_budget_usd), (True, 0.0, 0.0))

    def test_free_models_run_under_zero_dollar_ceilings(self):
        c = ScriptedClient({"gem": ["ok"]})
        router = ModelRouter(self.free_registry(), c, available_providers={"gemini"}, sleep=lambda s: None)
        self.assertEqual(self.run_task(router).provider, "gemini")

    def test_free_only_with_no_free_model_parks_task_instead_of_spending(self):
        reg = self.free_registry()
        reg.chains = {r: ["paid", "unpriced"] for r in Role}
        router = ModelRouter(reg, ScriptedClient({}), available_providers={"anthropic", "openai"},
                             sleep=lambda s: None)
        with self.assertRaises(AllModelsFailed) as ctx:
            self.run_task(router)
        self.assertEqual({a["error_class"] for a in ctx.exception.attempts}, {"skipped:not_free"})

    def test_unverified_price_never_counts_as_free_when_paid_allowed(self):
        c = ScriptedClient({"paid": ["ok"]})
        router = self.router(c, free_only=False)
        r = self.run_task(router)
        self.assertEqual(r.provider, "anthropic")  # verified paid model is allowed when free_only is off
        c2 = ScriptedClient({"gem": ["ok"]})
        reg = self.free_registry()
        reg.chains = {r: ["unpriced", "gem"] for r in Role}
        router2 = ModelRouter(reg, c2, available_providers={"openai", "gemini"},
                              policy=RouterPolicy(**PAID), sleep=lambda s: None)
        self.assertEqual(self.run_task(router2).provider, "gemini")
        self.assertNotIn("unpriced", c2.calls)


class RegistryTests(unittest.TestCase):
    def test_default_config_loads_and_covers_all_roles(self):
        reg = ModelRegistry.from_toml()
        for role in Role:
            self.assertTrue(reg.chain(role))
        # Critic chain starts with a different provider family than the judge (cross-model diversity).
        self.assertNotEqual(reg.chain(Role.CRITIC)[0].provider, reg.chain(Role.JUDGE)[0].provider)

    def test_default_config_is_zero_cost(self):
        reg = ModelRegistry.from_toml()
        for role in Role:
            for spec_ in reg.chain(role):
                self.assertTrue(spec_.free, f"{role}: {spec_.id} is not marked free")
                self.assertEqual((spec_.input_per_mtok, spec_.output_per_mtok), (0.0, 0.0), spec_.id)

    def test_default_config_has_a_local_fallback_for_every_role(self):
        reg = ModelRegistry.from_toml()
        for role in Role:
            self.assertIn("ollama", {s.provider for s in reg.chain(role)}, role)

    def test_unknown_model_rejected(self):
        with self.assertRaises(ValueError):
            ModelRegistry({}, {r: ["missing"] for r in Role})


if __name__ == "__main__":
    unittest.main()
