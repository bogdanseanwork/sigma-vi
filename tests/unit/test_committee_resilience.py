"""Committee resilience: output room, 503 handling, local timeouts, clean stop, progress."""

import json
import os
import sys
import types
import unittest
from unittest import mock

from sigma.ai.budgets import BUDGETS, TaskClass
from sigma.ai.litellm_client import LiteLLMClient, effective_timeout
from sigma.ai.roles import ModelSpec
from sigma.ai.router import AllModelsFailed, ProviderUnavailable, RateLimited, RouterPolicy
from sigma.committee import pipeline
from sigma.committee.run import describe_failure


class RecordingAsker:
    def __init__(self):
        self.task_classes = []

    def ask(self, task_id, agent, role, task_class, system, user):
        self.task_classes.append(task_class)
        return json.dumps({"vote": "own", "confidence": 70, "expected_return_24m_pct": 10,
                           "main_disagreement": "x", "evidence": ["a"]})


class VoteBudgetTests(unittest.TestCase):
    def test_vote_uses_specialist_class_with_room_for_thinking_tokens(self):
        asker = RecordingAsker()
        pipeline.vote(asker, "AAA", "pkt", "growth")
        self.assertEqual(asker.task_classes, [TaskClass.SPECIALIST])
        self.assertGreaterEqual(BUDGETS[TaskClass.SPECIALIST].max_output_tokens, 3_000)

    def test_progress_is_logged_every_25_votes_and_at_the_end(self):
        asker, lines = RecordingAsker(), []
        packets = {f"S{i}": "p" for i in range(10)}
        pipeline.run_votes(asker, packets, ["growth"] * 7, log=lines.append)   # 70 votes
        self.assertEqual(lines, ["  votes: 25/70", "  votes: 50/70", "  votes: 70/70"])

    def test_exact_multiple_is_logged_once(self):
        lines = []
        pipeline.run_votes(RecordingAsker(), {"A": "p"}, ["growth"] * 25, log=lines.append)
        self.assertEqual(lines, ["  votes: 25/25"])


class PolicyTests(unittest.TestCase):
    def test_default_rate_limit_retries(self):
        self.assertEqual(RouterPolicy().max_rate_limit_retries, 4)


class FailureTextTests(unittest.TestCase):
    def test_message_lists_each_attempt_on_one_line(self):
        exc = AllModelsFailed("vote:AAA:growth", [
            {"model": "gemini/x", "error_class": "unavailable", "detail": "line1\nline2   " + "z" * 300},
            {"model": "ollama/y", "error_class": "rate_limited"},
        ])
        text = describe_failure(exc)
        lines = text.splitlines()
        self.assertIn("vote:AAA:growth", lines[0])
        self.assertTrue(lines[1].startswith("  - gemini/x: line1 line2 zzz"))
        self.assertEqual(len(lines[1]), len("  - gemini/x: ") + 160)
        self.assertEqual(lines[2], "  - ollama/y: rate_limited")
        self.assertTrue(lines[-1].startswith("Nothing finished is lost. Fix the cause"))
        self.assertTrue(lines[-1].endswith("run the same command again."))


def _fake_litellm(raises):
    mod = types.ModuleType("litellm")

    class _Base(Exception):
        pass

    names = ["ContextWindowExceededError", "RateLimitError", "AuthenticationError", "PermissionDeniedError",
             "NotFoundError", "ServiceUnavailableError", "APIConnectionError", "Timeout", "InternalServerError"]
    mod.exceptions = types.SimpleNamespace(**{n: type(n, (_Base,), {}) for n in names})
    mod.calls = []

    def completion(**kw):
        mod.calls.append(kw)
        if raises is not None:
            raise raises(mod)
        raise AssertionError("no scripted outcome")

    mod.completion = completion
    return mod


class ClientMappingTests(unittest.TestCase):
    def _run(self, make_exc, provider="gemini", timeout=120.0):
        mod = _fake_litellm(make_exc)
        spec = ModelSpec("m", f"{provider}/m", provider, 1000, 0.0, 0.0, True)
        with mock.patch.dict(sys.modules, {"litellm": mod}), mock.patch("sigma.ai.litellm_client.export_dotenv"):
            client = LiteLLMClient()
            try:
                client.complete(spec, [{"role": "user", "content": "hi"}], 100, timeout)
            except Exception as e:
                return e, mod
        return None, mod

    def test_503_high_demand_is_rate_limited_with_retry_after(self):
        e, _ = self._run(lambda m: m.exceptions.ServiceUnavailableError("503 This model is currently experiencing high demand"))
        self.assertIsInstance(e, RateLimited)
        self.assertEqual(e.retry_after_s, 20.0)

    def test_internal_server_error_503_is_rate_limited_but_other_500_is_unavailable(self):
        e, _ = self._run(lambda m: m.exceptions.InternalServerError("code 503 overloaded"))
        self.assertIsInstance(e, RateLimited)
        e, _ = self._run(lambda m: m.exceptions.InternalServerError("boom"))
        self.assertIsInstance(e, ProviderUnavailable)
        self.assertNotIsInstance(e, RateLimited)

    def test_auth_and_connection_errors_stay_unavailable(self):
        for name in ("AuthenticationError", "PermissionDeniedError", "NotFoundError", "APIConnectionError", "Timeout"):
            e, _ = self._run(lambda m, n=name: getattr(m.exceptions, n)("x"))
            self.assertIsInstance(e, ProviderUnavailable, name)

    def test_ollama_gets_long_timeout(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SIGMA_OLLAMA_TIMEOUT_S", None)
            self.assertEqual(effective_timeout("ollama", 120.0), 900.0)
            self.assertEqual(effective_timeout("gemini", 120.0), 120.0)
        with mock.patch.dict(os.environ, {"SIGMA_OLLAMA_TIMEOUT_S": "50"}):
            self.assertEqual(effective_timeout("ollama", 120.0), 120.0)
        with mock.patch.dict(os.environ, {"SIGMA_OLLAMA_TIMEOUT_S": "junk"}):
            self.assertEqual(effective_timeout("ollama", 120.0), 900.0)

    def test_timeout_passed_to_litellm(self):
        _, mod = self._run(lambda m: m.exceptions.Timeout("t"), provider="ollama")
        self.assertGreaterEqual(mod.calls[0]["timeout"], 900.0)


if __name__ == "__main__":
    unittest.main()
