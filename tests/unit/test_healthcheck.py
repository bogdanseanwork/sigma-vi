import unittest

from sigma.core.healthcheck import CheckResult, format_results, run_checks


class HealthcheckTests(unittest.TestCase):
    def test_pass_fail_and_skip(self):
        def ok():
            return "200 OK"

        def boom():
            raise RuntimeError("HTTP 401 unauthorized")

        results = run_checks({"fred": ok, "exa": boom, "alpaca": None})
        self.assertEqual([(r.name, r.status) for r in results],
                         [("fred", "OK"), ("exa", "FAIL"), ("alpaca", "SKIPPED")])
        self.assertIn("401", results[1].detail)

    def test_errors_are_redacted_and_truncated(self):
        secret = "supersecretkey-1234567890"

        def leak():
            raise RuntimeError(f"bad url https://x.test/?api_key={secret} " + "x" * 500)

        [r] = run_checks({"fred": leak}, env={"FRED_API_KEY": secret})
        self.assertNotIn(secret, r.detail)
        self.assertLessEqual(len(r.detail), 200)

    def test_format_has_one_line_per_check_and_summary(self):
        text = format_results([CheckResult("fred", "OK", "series found"),
                               CheckResult("exa", "FAIL", "HTTP 401")])
        self.assertIn("fred", text)
        self.assertIn("1 of 2 checks passed", text)


if __name__ == "__main__":
    unittest.main()
