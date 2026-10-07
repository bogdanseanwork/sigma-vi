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


def make_router(client, providers=("anthropic", "openai", "gemini"), **kw):
    clock = kw.pop("clock", Clock())
    sleeps = kw.pop("sleeps", [])
    return ModelRouter(
        registry(), client, available_providers=set(providers),
        ledger=kw.pop("ledger", InMemoryLedger(clock)),
        checkpoints=kw.pop("checkpoints", InMemoryCheckpointStore()),
        policy=kw.pop("policy", RouterPolicy()), clock=clock, sleep=sleeps.append,
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
        pricey = RouterPolicy(per_task_max_usd=0.000001)
        with self.assertRaises(AllModelsFailed) as ctx:
            self.run_task(make_router(ScriptedClient({}), policy=pricey))
        self.assertTrue(all("per_task_cost_ceiling" in a["error_class"] for a in ctx.exception.attempts))

    def test_deterministic_tasks_refused(self):
        with self.assertRaises(ValueError):
            self.run_task(make_router(ScriptedClient({})), task_class=TaskClass.DETERMINISTIC)


class RegistryTests(unittest.TestCase):
    def test_default_config_loads_and_covers_all_roles(self):
        reg = ModelRegistry.from_toml()
        for role in Role:
            self.assertTrue(reg.chain(role))
        # Critic chain starts with a different provider family than the judge (cross-model diversity).
        self.assertNotEqual(reg.chain(Role.CRITIC)[0].provider, reg.chain(Role.JUDGE)[0].provider)

    def test_unknown_model_rejected(self):
        with self.assertRaises(ValueError):
            ModelRegistry({}, {r: ["missing"] for r in Role})


if __name__ == "__main__":
    unittest.main()
