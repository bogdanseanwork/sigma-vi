import unittest

from sigma.core.config import Integration, format_status, load_settings
from sigma.core.security import REDACTED, redact, redact_obj


class RedactionTests(unittest.TestCase):
    def test_env_secret_values_are_removed(self):
        env = {"OPENAI_API_KEY": "abc123-very-secret-value"}
        out = redact("calling with abc123-very-secret-value now", env)
        self.assertNotIn("abc123-very-secret-value", out)
        self.assertIn(REDACTED, out)

    def test_shaped_secrets_are_removed_without_env(self):
        samples = [
            "key sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAA",
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345",
            "https://x.com/q?apikey=ZZZZZZZZZZZZ&symbol=IBM",
            "postgresql://alice:hunter2pass@db.neon.tech/sigma",
        ]
        for s in samples:
            out = redact(s, env={})
            self.assertIn(REDACTED, out, s)
        self.assertNotIn("hunter2pass", redact(samples[3], env={}))
        self.assertIn("symbol=IBM", redact(samples[2], env={}))

    def test_short_values_not_redacted(self):
        self.assertEqual(redact("paper=true", {"OPENAI_API_KEY": "true"}), "paper=true")

    def test_nested(self):
        env = {"FRED_API_KEY": "fredkey-1234567890"}
        obj = {"a": ["x fredkey-1234567890"], "b": 3}
        self.assertEqual(redact_obj(obj, env), {"a": [f"x {REDACTED}"], "b": 3})


class ConfigTests(unittest.TestCase):
    def test_presence_only(self):
        s = load_settings({"OPENAI_API_KEY": "x" * 20, "ALPACA_API_KEY_ID": "id"})
        self.assertTrue(s.available(Integration.OPENAI))
        self.assertFalse(s.available(Integration.ANTHROPIC))
        self.assertFalse(s.available(Integration.ALPACA_PAPER))
        self.assertEqual(s.statuses[Integration.ALPACA_PAPER].missing, ("ALPACA_API_SECRET_KEY",))
        self.assertEqual(s.model_providers, ["openai"])

    def test_status_text_never_contains_values(self):
        secret = "s" * 40
        text = format_status(load_settings({"OPENAI_API_KEY": secret}))
        self.assertNotIn(secret, text)

    def test_live_trading_defaults_off_and_contradiction_resolves_safe(self):
        self.assertFalse(load_settings({}).live_trading_enabled)
        self.assertFalse(load_settings({"SIGMA_LIVE_TRADING": "true"}).live_trading_enabled)
        s = load_settings({"SIGMA_LIVE_TRADING": "true", "ALPACA_PAPER": "false"})
        self.assertTrue(s.live_trading_enabled)


if __name__ == "__main__":
    unittest.main()
