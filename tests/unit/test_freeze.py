"""Freezing the methodology, and the one-shot holdout gate."""

import json
import tempfile
import unittest
from pathlib import Path

from sigma import freeze


class Base(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.dir = Path(self._d.name)
        self.frozen = self.dir / "frozen_model.json"
        self.ledger = self.dir / "holdout_ledger.txt"

    def tearDown(self):
        self._d.cleanup()


class ManifestTests(Base):
    def test_manifest_covers_weights_filters_costs_splits_and_code(self):
        m = freeze.manifest()
        for key in ("weights", "categories", "plausible", "filters", "top_n", "cost_bps", "splits",
                    "criteria", "code"):
            self.assertIn(key, m)
        self.assertAlmostEqual(sum(m["weights"].values()), 1.0)
        self.assertTrue(all(len(h) == 64 for h in m["code"].values()))

    def test_digest_is_stable_and_changes_with_any_input(self):
        m = freeze.manifest()
        self.assertEqual(freeze.digest(m), freeze.digest(json.loads(json.dumps(m))))
        m2 = json.loads(json.dumps(m))
        m2["weights"]["Growth"] += 0.01
        self.assertNotEqual(freeze.digest(m), freeze.digest(m2))


class FreezeTests(Base):
    def test_freeze_then_check_passes_until_something_changes(self):
        freeze.freeze(self.frozen, note="test")
        ok, diffs = freeze.check(self.frozen)
        self.assertTrue(ok, diffs)
        doc = json.loads(self.frozen.read_text())
        doc["manifest"]["weights"]["Growth"] += 0.05           # someone tunes a weight afterwards
        self.frozen.write_text(json.dumps(doc))
        ok, diffs = freeze.check(self.frozen)
        self.assertFalse(ok)
        self.assertTrue(any("weights" in d for d in diffs))

    def test_a_frozen_model_cannot_be_overwritten_silently(self):
        freeze.freeze(self.frozen, note="first")
        with self.assertRaises(freeze.AlreadyFrozen):
            freeze.freeze(self.frozen, note="second")
        freeze.freeze(self.frozen, note="second", replace=True)   # deliberate, and recorded
        self.assertEqual(json.loads(self.frozen.read_text())["note"], "second")

    def test_check_without_a_freeze_fails(self):
        ok, diffs = freeze.check(self.frozen)
        self.assertFalse(ok)
        self.assertIn("not frozen", diffs[0])


class GateTests(Base):
    def test_holdout_needs_a_freeze(self):
        with self.assertRaises(freeze.GateClosed) as cm:
            freeze.open_gate(self.frozen, self.ledger, "final")
        self.assertIn("freeze", str(cm.exception))

    def test_holdout_needs_the_model_unchanged_since_the_freeze(self):
        freeze.freeze(self.frozen, note="x")
        doc = json.loads(self.frozen.read_text())
        doc["manifest"]["top_n"] = 30
        self.frozen.write_text(json.dumps(doc))
        with self.assertRaises(freeze.GateClosed) as cm:
            freeze.open_gate(self.frozen, self.ledger, "final")
        self.assertIn("changed", str(cm.exception))

    def test_it_opens_once_and_records_it(self):
        freeze.freeze(self.frozen, note="x")
        freeze.open_gate(self.frozen, self.ledger, "final frozen-model run")
        self.assertIn("FINAL", self.ledger.read_text())
        with self.assertRaises(freeze.GateClosed) as cm:
            freeze.open_gate(self.frozen, self.ledger, "again")
        self.assertIn("already", str(cm.exception))

    def test_a_second_look_is_possible_but_labelled_as_contaminated(self):
        freeze.freeze(self.frozen, note="x")
        freeze.open_gate(self.frozen, self.ledger, "first")
        freeze.open_gate(self.frozen, self.ledger, "second", second_look=True)
        self.assertIn("SECOND LOOK", self.ledger.read_text())


class JudgeTests(unittest.TestCase):
    def test_pass_fail_against_preregistered_criteria(self):
        result = {"cagr_net": 0.12, "cagr_universe": 0.08, "ic": 0.03, "ic_t": 1.5,
                  "random_percentile": 97.0, "worst_dd": -0.30, "dd_limit": -0.40}
        rows = freeze.judge(result)
        self.assertTrue(all(r["passed"] for r in rows), rows)
        result["cagr_net"] = 0.05
        rows = {r["criterion"]: r["passed"] for r in freeze.judge(result)}
        self.assertFalse(rows["net CAGR beats the equal-weight universe"])

    def test_missing_numbers_fail_rather_than_pass(self):
        rows = freeze.judge({})
        self.assertFalse(any(r["passed"] for r in rows))


if __name__ == "__main__":
    unittest.main()


class HoldoutAssessTests(unittest.TestCase):
    def test_assess_runs_only_on_holdout_dates_and_judges(self):
        import numpy as np
        import pandas as pd

        from sigma import holdout
        from sigma.backtest import splits
        rng = np.random.default_rng(0)
        dates = [__import__("datetime").date(y, m, 28) for y in (2022, 2024, 2025) for m in (3, 6, 9, 12)]
        rows = []
        for d in dates:
            n = 120
            sc = rng.random(n)
            rows.append(pd.DataFrame({"date": d, "symbol": [f"S{i}" for i in range(n)],
                                      "sector": rng.choice(list("AB"), n), "sigma_score": sc,
                                      "fwd": 0.2 * sc + 0.05 * rng.standard_normal(n), "ended": False}))
        data = pd.concat(rows, ignore_index=True)
        bench = pd.DataFrame({"SPY": 0.02}, index=sorted(data["date"].unique()))
        with tempfile.TemporaryDirectory() as d:
            out = holdout.assess(data, bench, splits.Splits(), n=20, trials=200, unlock="test",
                                 ledger=Path(d) / "l.txt")
        self.assertEqual(len(out["dates"]), 8)                      # the 2022 rows are validation
        self.assertTrue(all(v["passed"] for v in out["verdict"]), out["verdict"])
        self.assertIn("PASS", holdout.render(out, "abc" * 30, "test"))
