"""Simulation engine on histories with known properties."""

import unittest

import numpy as np

from sigma import simulate as sim


def history(n=60, mean=0.01, sd=0.05, bench_mean=0.007, seed=0, ppy=12):
    rng = np.random.default_rng(seed)
    bench = bench_mean + 0.04 * rng.standard_normal(n)
    strat = mean + 0.6 * (bench - bench_mean) + sd * rng.standard_normal(n)
    return sim.History(np.column_stack([strat, bench]), ppy)


def cfg(**kw):
    base = {"paths": 4000, "years": 2.0, "periods_per_year": 12, "seed": 1,
            "cost_bps_median": 0.0, "cost_bps_sigma": 0.0}
    base.update(kw)
    return sim.Config(**base)


class MetricTests(unittest.TestCase):
    def test_known_path(self):
        # +10%, -50%, +20%: nav 1.1, 0.55, 0.66 -> max drawdown 50%, terminal -34%
        p = np.array([[[0.10, 0.0], [-0.50, 0.0], [0.20, 0.0]]])
        m = sim.path_metrics(p, ppy=3)
        self.assertAlmostEqual(float(m["terminal"][0]), -0.34)
        self.assertAlmostEqual(float(m["max_dd"][0]), 0.50)
        self.assertAlmostEqual(float(m["cagr"][0]), -0.34)        # 3 steps at 3 per year = 1 year
        self.assertAlmostEqual(float(m["excess"][0]), -0.34)

    def test_costs_lower_the_strategy_but_not_the_benchmark(self):
        rng = np.random.default_rng(0)
        base = np.zeros((10, 12, 2))
        out = sim.apply_costs(rng, base, cfg(cost_bps_median=30.0, cost_bps_sigma=0.0, turnover_per_year=2.0))
        self.assertTrue((out[:, :, 0] < 0).all())
        self.assertTrue((out[:, :, 1] == 0).all())
        # 2 turns a year x 2 sides x 30 bp = 1.2% a year
        self.assertAlmostEqual(float(out[0, :, 0].sum()), -0.012, places=6)


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.h = history()
        self.rng = np.random.default_rng(3)

    def test_shapes(self):
        for name, g in sim._GEN.items():
            out = g(np.random.default_rng(3), 50, 24, self.h)
            self.assertEqual(out.shape, (50, 24, 2), name)
            self.assertTrue(np.isfinite(out).all(), name)

    def test_bootstrap_only_uses_historical_values(self):
        out = sim.gen_bootstrap(self.rng, 100, 24, self.h)
        hist_vals = set(np.round(self.h.returns[:, 0], 12))
        self.assertTrue(set(np.round(out[:, :, 0].ravel(), 12)) <= hist_vals)

    def test_bootstrap_blocks_continue_in_order(self):
        idx = sim._stationary_indices(np.random.default_rng(0), 200, 40, 60, mean_block=8.0)
        steps_forward = ((idx[:, 1:] - idx[:, :-1]) % 60 == 1).mean()
        self.assertGreater(steps_forward, 0.7)   # ~1 - 1/8 of steps continue a block

    def test_student_t_matches_fitted_moments_and_has_fat_tails(self):
        out = sim.gen_student_t(np.random.default_rng(0), 20000, 12, self.h).reshape(-1, 2)
        self.assertAlmostEqual(out[:, 0].mean(), self.h.returns[:, 0].mean(), places=2)
        self.assertAlmostEqual(out[:, 0].std(), self.h.returns[:, 0].std(ddof=1), delta=0.004)
        z = (out[:, 0] - out[:, 0].mean()) / out[:, 0].std()
        self.assertGreater(float((z**4).mean()), 3.5)       # normal would be 3

    def test_uncertainty_is_wider_than_fixed_parameters(self):
        a = sim.gen_student_t(np.random.default_rng(0), 5000, 24, self.h)
        b = sim.gen_uncertainty(np.random.default_rng(0), 5000, 24, self.h)
        ta, tb = np.prod(1 + a[:, :, 0], axis=1), np.prod(1 + b[:, :, 0], axis=1)
        self.assertGreater(tb.std(), ta.std())

    def test_stress_is_worse_than_plain_bootstrap(self):
        a = sim.path_metrics(sim.gen_bootstrap(np.random.default_rng(0), 3000, 24, self.h), 12)
        b = sim.path_metrics(sim.gen_stress(np.random.default_rng(0), 3000, 24, self.h), 12)
        self.assertGreater(np.median(b["max_dd"]), np.median(a["max_dd"]))

    def test_regime_states_follow_history(self):
        _, trans, params = sim._regimes(self.h)
        self.assertTrue(np.allclose(trans.sum(axis=1), 1.0))
        self.assertLess(params[1][0][1], params[0][0][1])     # benchmark mean is lower when stressed


class RunTests(unittest.TestCase):
    def test_ledger_counts_every_requested_path(self):
        res = sim.run(history(), cfg(paths=3000))
        self.assertEqual(res.completed, 3000)
        self.assertEqual(len(res.metrics["terminal"]), 3000)
        self.assertEqual(sum(res.ledger.values()), 3000)

    def test_reproducible_from_seed(self):
        a = sim.run(history(), cfg(paths=2000, seed=9))
        b = sim.run(history(), cfg(paths=2000, seed=9))
        c = sim.run(history(), cfg(paths=2000, seed=10))
        np.testing.assert_array_equal(a.metrics["terminal"], b.metrics["terminal"])
        self.assertFalse(np.array_equal(a.metrics["terminal"], c.metrics["terminal"]))

    def test_a_strategy_with_edge_beats_the_benchmark_more_often_than_not(self):
        res = sim.run(history(mean=0.02, bench_mean=0.005), cfg(paths=4000))
        self.assertLess(sim.summarise(res.metrics)["p_underperform"], 0.4)

    def test_a_losing_strategy_shows_a_high_probability_of_loss(self):
        res = sim.run(history(mean=-0.01), cfg(paths=4000))
        self.assertGreater(sim.summarise(res.metrics)["p_loss"], 0.6)

    def test_extreme_fat_tails_never_produce_impossible_returns(self):
        wild = sim.History(np.random.default_rng(0).standard_normal((40, 2)) * 0.30, 12)
        res = sim.run(wild, cfg(paths=4000))
        self.assertTrue(np.isfinite(res.metrics["cagr"]).all())
        self.assertEqual(res.completed, 4000)

    def test_costs_reduce_median_return(self):
        a = sim.summarise(sim.run(history(), cfg(paths=3000)).metrics)["cagr_median"]
        b = sim.summarise(sim.run(history(), cfg(paths=3000, cost_bps_median=100.0)).metrics)["cagr_median"]
        self.assertLess(b, a)

    def test_report_states_the_limitation_when_under_a_million(self):
        h = history()
        res = sim.run(h, cfg(paths=2000))
        text = sim.render(res, h)
        self.assertIn("fewer than 1,000,000 paths", text)
        self.assertIn("2,000 of 2,000", text)

    def test_convergence_flags_and_se(self):
        res = sim.run(history(), cfg(paths=40000))
        conv = sim.convergence(res)
        self.assertEqual(len(conv), 4)
        self.assertTrue((conv["std_error"] > 0).all())

    def test_multiple_workers_give_the_same_answer(self):
        a = sim.run(history(), cfg(paths=1000, workers=1))
        b = sim.run(history(), cfg(paths=1000, workers=2))
        np.testing.assert_array_equal(a.metrics["terminal"], b.metrics["terminal"])


class HistoryTests(unittest.TestCase):
    def test_holdout_rows_excluded_unless_asked(self):
        import tempfile
        from pathlib import Path

        import pandas as pd
        df = pd.DataFrame({"date": pd.date_range("2022-01-31", periods=6, freq="ME"),
                           "strategy": 0.01, "benchmark": 0.005,
                           "period": ["validation"] * 3 + ["holdout"] * 3})
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "h.csv"
            df.to_csv(p, index=False)
            self.assertEqual(len(sim.load_history(str(p)).returns), 3)
            self.assertEqual(len(sim.load_history(str(p), use_holdout=True).returns), 6)
            self.assertEqual(sim.load_history(str(p)).periods_per_year, 12)


if __name__ == "__main__":
    unittest.main()
