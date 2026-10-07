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


class DotenvTests(unittest.TestCase):
    def write(self, text):
        import tempfile
        from pathlib import Path
        d = tempfile.mkdtemp()
        p = Path(d) / ".env"
        p.write_text(text, encoding="utf-8")
        return p

    def test_reads_values_comments_quotes_and_export(self):
        from sigma.core.config import read_dotenv
        p = self.write('# comment\nFRED_API_KEY=abc123\n\nexport EXA_API_KEY="q v"\nEMPTY=\nX=a#b\n')
        expected = {"FRED_API_KEY": "abc123", "EXA_API_KEY": "q v", "EMPTY": "", "X": "a#b"}
        self.assertEqual(read_dotenv(p), expected)

    def test_missing_file_is_empty(self):
        from pathlib import Path

        from sigma.core.config import read_dotenv
        self.assertEqual(read_dotenv(Path("/nonexistent/.env")), {})

    def test_load_settings_reads_dotenv_and_environment_wins(self):
        p = self.write("FRED_API_KEY=fromfile\nEXA_API_KEY=fromfile\n")
        s = load_settings(dotenv_path=p, environ={"EXA_API_KEY": "fromenv"})
        self.assertTrue(s.available(Integration.FRED))
        self.assertTrue(s.available(Integration.EXA))

    def test_template_placeholders_count_as_missing(self):
        p = self.write('DATABASE_URL=postgresql://user:password@host/sigma_vi?sslmode=require\n'
                       'SEC_EDGAR_USER_AGENT="SIGMA VI research you@example.com"\n')
        s = load_settings(dotenv_path=p, environ={})
        self.assertFalse(s.available(Integration.POSTGRES))
        self.assertFalse(s.available(Integration.SEC_EDGAR))

    def test_export_dotenv_fills_environment_without_overriding(self):
        from sigma.core.config import export_dotenv
        p = self.write("GEMINI_API_KEY=fromfile\nFRED_API_KEY=fromfile\nEMPTY=\n")
        environ = {"FRED_API_KEY": "already-set"}
        added = export_dotenv(p, environ)
        self.assertEqual(environ, {"FRED_API_KEY": "already-set", "GEMINI_API_KEY": "fromfile"})
        self.assertEqual(added, ["GEMINI_API_KEY"])  # names only, never values

    def test_utf8_bom_tolerated(self):
        from sigma.core.config import read_dotenv
        p = self.write("﻿FRED_API_KEY=abc\n")
        self.assertEqual(read_dotenv(p), {"FRED_API_KEY": "abc"})


if __name__ == "__main__":
    unittest.main()
