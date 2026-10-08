"""Weights, confidence, portfolio analysis and the report's honesty about stages that were not run."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sigma import final


def book(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "final_score": rng.uniform(70, 100, n), "confidence": rng.uniform(30, 80, n),
        "thesis_strength": rng.uniform(30, 80, n), "vol": rng.uniform(0.2, 0.8, n),
        "sector": rng.choice(["Tech", "Health", "Energy", "Fin", "Ind"], n),
        "market_cap": rng.uniform(1e9, 5e11, n)}, index=[f"S{i:02d}" for i in range(n)])


class WeightTests(unittest.TestCase):
    def test_weights_sum_to_one_and_respect_every_limit(self):
        df = book()
        w = final.conviction_weights(final.conviction_raw(df), df["sector"])
        self.assertAlmostEqual(w.sum(), 1.0, places=9)
        self.assertTrue((w <= final.MAX_WEIGHT + 1e-9).all())
        self.assertTrue((w >= final.MIN_WEIGHT - 1e-9).all())
        self.assertTrue((w.groupby(df["sector"]).sum() <= final.MAX_SECTOR + 1e-9).all())

    def test_a_dominant_name_is_capped_not_dropped(self):
        raw = pd.Series([1000.0] + [1.0] * 39, index=[f"S{i}" for i in range(40)])
        w = final.conviction_weights(raw, pd.Series(["A", "B", "C", "D", "E"] * 8, index=raw.index))
        self.assertAlmostEqual(w.iloc[0], final.MAX_WEIGHT, places=6)
        self.assertGreater(w.iloc[1:].min(), 0)

    def test_sector_limit_binds_when_one_sector_dominates(self):
        raw = pd.Series([10.0] * 20 + [1.0] * 20, index=[f"S{i}" for i in range(40)])
        sector = pd.Series(["Tech"] * 20 + ["Other", "Other2", "Other3", "Other4"] * 5, index=raw.index)
        w = final.conviction_weights(raw, sector)
        self.assertLessEqual(w[sector == "Tech"].sum(), final.MAX_SECTOR + 1e-9)
        self.assertAlmostEqual(w.sum(), 1.0, places=9)

    def test_impossible_limits_raise(self):
        with self.assertRaises(ValueError):
            final.conviction_weights(pd.Series([1.0] * 10), pd.Series(["A"] * 10), cap=0.05)

    def test_higher_risk_means_lower_weight_all_else_equal(self):
        df = book(40).assign(final_score=90.0, confidence=60.0, thesis_strength=60.0, vol=0.3)
        df.loc["S00", "vol"] = 0.9
        w = final.conviction_weights(final.conviction_raw(df), pd.Series(["A", "B", "C", "D", "E"] * 8, index=df.index))
        self.assertLess(w["S00"], w["S01"])

    def test_a_calm_quarter_does_not_make_a_stock_look_riskless(self):
        df = book(5).assign(vol=0.01)
        raw = final.conviction_raw(df)
        self.assertTrue(np.isfinite(raw).all())


class ConfidenceTests(unittest.TestCase):
    def test_is_bounded_and_capped_by_model_reliability(self):
        best = final.confidence_score(1.0, 1.0, 100, 6)
        self.assertLessEqual(best, 75.0)               # reliability 0.5 -> never reads as certain
        self.assertEqual(final.confidence_score(1.0, 1.0, 100, 6, model_reliability=1.0), 100.0)
        self.assertEqual(final.confidence_score(0, 0, 0, 0), 0.0)

    def test_monotone_in_agreement(self):
        self.assertLess(final.confidence_score(0.9, 0.2, 60, 6), final.confidence_score(0.9, 0.9, 60, 6))


class AnalysisTests(unittest.TestCase):
    def test_exposures_sum_to_one(self):
        df = book().assign(w=1 / 40)
        e = final.exposures(df, "w")
        self.assertAlmostEqual(e["sector"].sum(), 1.0)
        self.assertAlmostEqual(e["market cap"].sum(), 1.0)

    def test_perfectly_correlated_names_are_one_risk_factor(self):
        rng = np.random.default_rng(0)
        base = rng.standard_normal(300) * 0.01
        r = pd.DataFrame({f"S{i}": base for i in range(5)})
        s = final.risk_stats(r, pd.Series(0.2, index=r.columns))
        self.assertAlmostEqual(s["avg_pairwise_corr"], 1.0, places=6)
        self.assertAlmostEqual(s["effective_risk_factors"], 1.0, places=3)
        self.assertAlmostEqual(s["effective_names"], 5.0)

    def test_independent_names_diversify(self):
        rng = np.random.default_rng(1)
        r = pd.DataFrame(rng.standard_normal((2000, 8)) * 0.01, columns=[f"S{i}" for i in range(8)])
        s = final.risk_stats(r, pd.Series(1 / 8, index=r.columns))
        self.assertLess(abs(s["avg_pairwise_corr"]), 0.1)
        self.assertGreater(s["effective_risk_factors"], 6.0)
        self.assertLess(s["vol"], 0.01 * np.sqrt(252) * 0.5)


class ReportTests(unittest.TestCase):
    def setUp(self):
        n = 22
        syms = [f"S{i:02d}" for i in range(30)]
        screen = pd.DataFrame({"symbol": syms, "name": "Co", "sector": ["Tech", "Health", "Energy", "Fin", "Ind"] * 6,
                               "market_cap": 5e9, "coverage": 0.9, "low_volatility": -0.4})
        table = pd.DataFrame({"final_score": np.linspace(95, 80, 30), "sigma_score": np.linspace(94, 79, 30),
                              "committee_adj": 1.0, "debate_adj": 1.0, "tournament_adj": 0.0,
                              "red_penalty": -1.0, "agreement": 0.8, "exp_return_24m_pct": 20.0,
                              "n_votes": 6}, index=pd.Index(screen["symbol"], name="symbol"))
        debates = {s: {"confidence": 70, "thesis_strength": 60, "surviving_criticisms": ["valuation"],
                       "transcript": [("Bull case", "Strong growth.")]} for s in screen["symbol"]}
        self.top, self.watch = final.assemble(table, screen, debates, top_n=n, extra=2)

    def test_assemble_splits_and_weights(self):
        self.assertEqual(len(self.top), 22)
        self.assertEqual(len(self.watch), 2)
        self.assertAlmostEqual(self.top["w_conviction"].sum(), 1.0, places=9)
        self.assertAlmostEqual(self.top["w_equal"].sum(), 1.0)

    def test_too_few_sectors_relaxes_the_sector_limit_and_says_so(self):
        scr = pd.DataFrame({"symbol": [f"S{i:02d}" for i in range(30)], "name": "Co",
                            "sector": ["Tech", "Health"] * 15, "market_cap": 5e9, "coverage": 0.9,
                            "low_volatility": -0.4})
        tbl = pd.DataFrame({"final_score": np.linspace(95, 80, 30), "sigma_score": 90.0, "committee_adj": 0.0,
                            "debate_adj": 0.0, "tournament_adj": 0.0, "red_penalty": 0.0, "agreement": 0.5,
                            "exp_return_24m_pct": 10.0, "n_votes": 6}, index=pd.Index(scr["symbol"], name="symbol"))
        top, _ = final.assemble(tbl, scr, {}, top_n=30, extra=0)
        self.assertGreater(top.attrs["sector_cap"], final.MAX_SECTOR)
        self.assertAlmostEqual(top["w_conviction"].sum(), 1.0, places=9)
        with tempfile.TemporaryDirectory() as d:
            text = final.render(top, top.iloc[0:0], final.exposures(top, "w_conviction"), None, Path(d))
        self.assertIn("relaxed from 30%", text)

    def test_missing_stages_are_reported_as_not_run_never_omitted(self):
        with tempfile.TemporaryDirectory() as d:
            text = final.render(self.top, self.watch, final.exposures(self.top, "w_conviction"), None, Path(d))
        self.assertIn("EXPERIMENTAL", text)
        self.assertIn("Sealed holdout opened: **NO - not run**", text)
        self.assertIn("Methodology frozen before the holdout: **NO**", text)
        self.assertEqual(text.count("**NOT RUN.**"), len(final.STAGE_FILES))

    def test_present_stage_text_is_included_verbatim(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "holdout_report.txt").write_text("2 of 4 criteria passed.", encoding="utf-8")
            text = final.render(self.top, self.watch, final.exposures(self.top, "w_conviction"), None, Path(d))
        self.assertIn("2 of 4 criteria passed.", text)
        self.assertIn("Sealed holdout opened: **yes**", text)


if __name__ == "__main__":
    unittest.main()
