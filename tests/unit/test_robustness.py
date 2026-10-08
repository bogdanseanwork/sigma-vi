"""E2 robustness tools on synthetic panels where the right answer is known."""

import math
import unittest
from datetime import date

import numpy as np
import pandas as pd

from sigma.backtest import robustness as rb

CATS = ["Growth", "Valuation", "Momentum"]
WEIGHTS = {"Growth": 0.5, "Valuation": 0.3, "Momentum": 0.2}


def period_of(d):
    return "train" if d <= date(2020, 12, 31) else "validation"


def make_panel(signal_cat=None, n_dates=12, n=300, seed=1, signal=0.05):
    """Random panel; if ``signal_cat`` is given that category truly predicts returns."""
    rng = np.random.default_rng(seed)
    dates = [date(2018 + i // 4, 3 * (i % 4) + 3, 28) for i in range(n_dates)]
    frames = []
    for d in dates:
        cats = {f"cat_{c}": rng.standard_normal(n) for c in CATS}
        fwd = rng.standard_normal(n) * 0.10
        if signal_cat:
            fwd = fwd + signal * cats[f"cat_{signal_cat}"]
        df = pd.DataFrame({"date": d, "symbol": [f"S{i}" for i in range(n)],
                           "sector": rng.choice(["A", "B", "C"], n), "fwd": fwd, **cats})
        df["sigma_score"] = rb.composite_from(df, WEIGHTS).rank(pct=True) * 100
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


class MultipleTestingTests(unittest.TestCase):
    def test_p_value_matches_a_known_t(self):
        self.assertAlmostEqual(rb.p_from_t(2.228, 11), 0.05, places=3)  # t(0.975, df=10)

    def test_holm_is_monotone_capped_and_no_smaller_than_raw(self):
        p = [0.001, 0.02, 0.04, 0.5]
        adj = rb.holm(p)
        self.assertEqual(adj[0], 0.004)
        self.assertTrue(all(a >= b for a, b in zip(adj, p, strict=True)))
        self.assertTrue(all(a <= 1 for a in adj))

    def test_bh_is_less_strict_than_holm(self):
        p = [0.001, 0.02, 0.04, 0.5]
        self.assertTrue(all(b <= h + 1e-12 for b, h in zip(rb.benjamini_hochberg(p), rb.holm(p), strict=True)))

    def test_nan_p_values_pass_through(self):
        out = rb.holm([0.01, float("nan")])
        self.assertTrue(math.isnan(out[1]))

    def test_ledger_flags_many_tests(self):
        led = rb.correct({"a": (3.0, 12), "b": (0.5, 12), "c": (2.1, 12)})
        self.assertGreater(led.loc["c", "p_holm"], led.loc["c", "p"])


class CompositeTests(unittest.TestCase):
    def test_missing_category_is_rescaled_not_zeroed(self):
        df = pd.DataFrame({"cat_Growth": [1.0, 1.0], "cat_Valuation": [1.0, np.nan], "cat_Momentum": [1.0, 1.0]})
        comp = rb.composite_from(df, WEIGHTS)
        self.assertAlmostEqual(comp.iloc[1], 1.0)  # all available scores are 1, so the rescaled mean is 1

    def test_too_little_coverage_gives_no_score(self):
        df = pd.DataFrame({"cat_Growth": [np.nan], "cat_Valuation": [np.nan], "cat_Momentum": [1.0]})
        self.assertTrue(rb.composite_from(df, WEIGHTS).isna().all())  # only 20% of weight present


class FactorIcTests(unittest.TestCase):
    def test_finds_the_real_signal_and_not_the_noise(self):
        panel = make_panel("Growth", signal=0.08)
        res = rb.ic_by_period(panel, ["cat_Growth", "cat_Valuation"], period_of)
        self.assertGreater(res["train"].loc["cat_Growth", "t"], 4)
        self.assertLess(abs(res["train"].loc["cat_Valuation", "t"]), 3)

    def test_step_thins_overlapping_dates(self):
        panel = make_panel(n_dates=12)
        res = rb.ic_by_period(panel, ["cat_Growth"], period_of, step=4)
        total = sum(int(v.loc["cat_Growth", "n"]) for v in res.values())
        self.assertEqual(total, 3)


class AblationTests(unittest.TestCase):
    def test_dropping_the_real_signal_hurts_most(self):
        panel = make_panel("Growth", signal=0.08, n_dates=12)
        ab = rb.ablation(panel, WEIGHTS, period_of, n=40)
        tr = ab[ab.period == "train"].set_index("dropped")
        self.assertLess(tr.loc["Growth", "d_ic"], tr.loc["Valuation", "d_ic"])
        self.assertLess(tr.loc["Growth", "d_ic"], 0)
        self.assertEqual(tr.loc["(none dropped)", "d_ic"], 0.0)


class RandomMatchedTests(unittest.TestCase):
    def test_real_edge_beats_nearly_all_random_portfolios(self):
        panel = make_panel("Growth", signal=0.10, n_dates=12)
        panel["sigma_score"] = panel["cat_Growth"]
        res = rb.random_matched(panel, n=40, trials=300)
        self.assertGreater(res["percentile"], 95)

    def test_no_edge_sits_in_the_middle(self):
        panel = make_panel(None, n_dates=12)
        panel["sigma_score"] = np.random.default_rng(3).random(len(panel))
        res = rb.random_matched(panel, n=40, trials=400)
        self.assertTrue(5 < res["percentile"] < 95)

    def test_same_seed_same_answer(self):
        panel = make_panel("Growth", n_dates=4)
        a = rb.random_matched(panel, n=20, trials=50, seed=5)
        b = rb.random_matched(panel, n=20, trials=50, seed=5)
        self.assertEqual(a, b)


class PerturbationTests(unittest.TestCase):
    def test_zero_perturbation_changes_nothing(self):
        panel = make_panel("Growth", n_dates=4)
        res = rb.perturbation(panel, WEIGHTS, pct=0.0, n=40, trials=5)
        self.assertAlmostEqual(res["mean_overlap"], 1.0)

    def test_bigger_perturbation_changes_more_names(self):
        panel = make_panel("Growth", n_dates=4)
        small = rb.perturbation(panel, WEIGHTS, pct=0.05, n=40, trials=30)
        large = rb.perturbation(panel, WEIGHTS, pct=0.50, n=40, trials=30)
        self.assertGreater(small["mean_overlap"], large["mean_overlap"])


class RegimeTests(unittest.TestCase):
    def test_splits_up_and_down_quarters(self):
        panel = make_panel("Growth", signal=0.08, n_dates=12)
        panel["sigma_score"] = panel["cat_Growth"]
        dates = sorted(panel["date"].unique())
        market = pd.Series([0.05, -0.04] * 6, index=dates)
        out = rb.regimes(panel, market, n=40).set_index("regime")
        self.assertEqual(out.loc["market up", "quarters"], 6)
        self.assertEqual(out.loc["market down", "quarters"], 6)
        self.assertGreater(out.loc["market up", "excess"], 0)


if __name__ == "__main__":
    unittest.main()
