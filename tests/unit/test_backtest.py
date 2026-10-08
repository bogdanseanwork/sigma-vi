"""Point-in-time backtest mechanics: splits, forward returns, portfolios, costs, information coefficients."""

import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from sigma.backtest import analysis, engine, splits

D = date.fromisoformat


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.s = splits.Splits(train_end=D("2020-12-31"), validation_end=D("2023-12-31"))

    def test_periods(self):
        self.assertEqual(self.s.period_of(D("2019-06-30")), "train")
        self.assertEqual(self.s.period_of(D("2021-03-31")), "validation")
        self.assertEqual(self.s.period_of(D("2024-01-02")), "holdout")

    def test_holdout_is_locked_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            g = splits.HoldoutGuard(self.s, Path(d) / "holdout_ledger.txt")
            g.require([D("2022-03-31")])  # fine
            with self.assertRaises(splits.HoldoutLocked):
                g.require([D("2022-03-31"), D("2024-03-29")])
            self.assertFalse((Path(d) / "holdout_ledger.txt").exists())

    def test_opening_the_holdout_is_recorded_with_the_reason(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = Path(d) / "holdout_ledger.txt"
            g = splits.HoldoutGuard(self.s, ledger)
            g.require([D("2024-03-29")], unlock="final frozen-model run v1")
            self.assertIn("final frozen-model run v1", ledger.read_text())
            self.assertIn("2024-03-29", ledger.read_text())


class CalendarTests(unittest.TestCase):
    def test_last_trading_day_of_each_quarter(self):
        days = pd.bdate_range("2023-01-02", "2023-12-29")
        got = engine.rebalance_dates([d.date() for d in days], D("2023-01-01"), D("2023-12-31"), "Q")
        self.assertEqual(got, [D("2023-03-31"), D("2023-06-30"), D("2023-09-29"), D("2023-12-29")])

    def test_monthly(self):
        days = pd.bdate_range("2023-01-02", "2023-03-31")
        self.assertEqual(len(engine.rebalance_dates([d.date() for d in days], D("2023-01-01"),
                                                     D("2023-03-31"), "M")), 3)

    def test_incomplete_final_period_is_dropped(self):
        days = pd.bdate_range("2023-01-02", "2023-05-10")
        got = engine.rebalance_dates([d.date() for d in days], D("2023-01-01"), D("2023-05-10"), "Q")
        self.assertEqual(got, [D("2023-03-31")])


def wide(prices):
    idx = pd.to_datetime(list(prices["dates"]))
    return pd.DataFrame({k: pd.Series(v, index=idx) for k, v in prices.items() if k != "dates"})


class ForwardReturnTests(unittest.TestCase):
    def test_simple_forward_return_between_rebalance_dates(self):
        w = wide({"dates": ["2023-03-31", "2023-06-30"], "AAA": [10.0, 11.0], "BBB": [20.0, 19.0]})
        fr = engine.forward_returns(w, [D("2023-03-31"), D("2023-06-30")])
        self.assertAlmostEqual(fr.loc[D("2023-03-31"), "AAA"], 0.10)
        self.assertAlmostEqual(fr.loc[D("2023-03-31"), "BBB"], -0.05)
        self.assertNotIn(D("2023-06-30"), fr.index)  # last date has no next period

    def test_step_gives_a_longer_horizon_return(self):
        w = wide({"dates": ["2023-03-31", "2023-06-30", "2023-09-29"], "AAA": [10.0, 11.0, 12.1]})
        dates = [D("2023-03-31"), D("2023-06-30"), D("2023-09-29")]
        fr = engine.forward_returns(w, dates, step=2)
        self.assertAlmostEqual(fr.loc[D("2023-03-31"), "AAA"], 0.21)
        self.assertEqual(list(fr.index), [D("2023-03-31")])

    def test_stock_that_stops_trading_is_valued_at_its_last_price_and_flagged(self):
        w = wide({"dates": ["2023-03-31", "2023-05-15", "2023-06-30"], "AAA": [10.0, np.nan, np.nan],
                  "BBB": [10.0, 10.0, 10.0]})
        w.loc["2023-03-31", "AAA"] = 10.0
        w.loc["2023-05-15", "AAA"] = 4.0
        fr, ended = engine.forward_returns(w, [D("2023-03-31"), D("2023-06-30")], with_flags=True)
        self.assertAlmostEqual(fr.loc[D("2023-03-31"), "AAA"], -0.60)
        self.assertTrue(ended.loc[D("2023-03-31"), "AAA"])
        self.assertFalse(ended.loc[D("2023-03-31"), "BBB"])

    def test_stock_without_a_start_price_has_no_return(self):
        w = wide({"dates": ["2023-03-31", "2023-06-30"], "NEW": [np.nan, 12.0]})
        fr = engine.forward_returns(w, [D("2023-03-31"), D("2023-06-30")])
        self.assertTrue(np.isnan(fr.loc[D("2023-03-31"), "NEW"]))


class PriceMatrixTests(unittest.TestCase):
    def test_duplicate_bars_do_not_break_the_matrix(self):
        from sigma.backtest import run
        px = pd.DataFrame({"symbol": ["A", "A", "A", "B"],
                           "date": [D("2023-01-03"), D("2023-01-03"), D("2023-01-04"), D("2023-01-03")],
                           "close": [1.0, 2.0, 3.0, 4.0]})
        w = run.price_matrix(px)
        self.assertEqual(w.loc["2023-01-03", "A"], 2.0)
        self.assertEqual(w.shape, (2, 2))


class PortfolioTests(unittest.TestCase):
    def panel(self):
        rows = []
        for d, rets in {D("2023-03-31"): [0.10, 0.05, -0.02, -0.10], D("2023-06-30"): [0.04, 0.02, 0.01, -0.03]}.items():
            for i, r in enumerate(rets):
                rows.append({"date": d, "symbol": f"S{i}", "sigma_score": 100 - 10 * i, "fwd": r})
        return pd.DataFrame(rows)

    def test_top_n_equal_weight_gross_return(self):
        out = engine.run_portfolio(self.panel(), n=2, cost_bps=0)
        self.assertAlmostEqual(out.loc[D("2023-03-31"), "gross"], 0.075)
        self.assertAlmostEqual(out.loc[D("2023-06-30"), "gross"], 0.03)

    def test_first_period_pays_to_build_the_portfolio_then_only_for_changes(self):
        out = engine.run_portfolio(self.panel(), n=2, cost_bps=50)
        self.assertAlmostEqual(out.loc[D("2023-03-31"), "turnover"], 1.0)   # all new
        self.assertAlmostEqual(out.loc[D("2023-06-30"), "turnover"], 0.0)   # same names
        self.assertAlmostEqual(out.loc[D("2023-03-31"), "net"], 0.075 - 0.0050)
        self.assertAlmostEqual(out.loc[D("2023-06-30"), "net"], 0.03)

    def test_turnover_counts_fractions_replaced(self):
        p = self.panel()
        p.loc[p["date"] == D("2023-06-30"), "sigma_score"] = [10, 20, 30, 40]   # reversal: S3,S2 now top
        out = engine.run_portfolio(p, n=2, cost_bps=0)
        self.assertAlmostEqual(out.loc[D("2023-06-30"), "turnover"], 1.0)

    def test_missing_forward_return_is_skipped_not_zeroed(self):
        p = self.panel()
        p.loc[(p["date"] == D("2023-03-31")) & (p["symbol"] == "S0"), "fwd"] = np.nan
        out = engine.run_portfolio(p, n=2, cost_bps=0)
        self.assertAlmostEqual(out.loc[D("2023-03-31"), "gross"], 0.05)

    def test_universe_benchmark_is_equal_weight_of_everything_scored(self):
        b = engine.universe_benchmark(self.panel())
        self.assertAlmostEqual(b.loc[D("2023-03-31")], (0.10 + 0.05 - 0.02 - 0.10) / 4)

    def test_cost_presets_ordered(self):
        c = engine.COST_BPS
        self.assertLess(c["optimistic"], c["base"])
        self.assertLess(c["base"], c["pessimistic"])


class AnalysisTests(unittest.TestCase):
    def test_perfect_ranking_has_information_coefficient_one(self):
        rows = [{"date": D("2023-03-31"), "symbol": f"S{i}", "sigma_score": i, "fwd": i * 0.01}
                for i in range(30)]
        ic = analysis.information_coefficients(pd.DataFrame(rows), ["sigma_score"])
        self.assertAlmostEqual(ic.loc[D("2023-03-31"), "sigma_score"], 1.0)

    def test_reversed_ranking_is_minus_one_and_small_samples_are_skipped(self):
        rows = [{"date": D("2023-03-31"), "symbol": f"S{i}", "x": -i, "fwd": i * 0.01} for i in range(30)]
        rows += [{"date": D("2023-06-30"), "symbol": f"S{i}", "x": i, "fwd": i} for i in range(3)]
        ic = analysis.information_coefficients(pd.DataFrame(rows), ["x"], min_names=20)
        self.assertAlmostEqual(ic.loc[D("2023-03-31"), "x"], -1.0)
        self.assertNotIn(D("2023-06-30"), ic.index)

    def test_ic_summary(self):
        s = analysis.ic_summary(pd.Series([0.05, 0.03, 0.07, 0.01, 0.04]))
        self.assertAlmostEqual(s["mean"], 0.04)
        self.assertEqual(s["n"], 5)
        self.assertGreater(s["t"], 2)
        self.assertEqual(s["hit"], 1.0)

    def test_quintile_spread(self):
        rows = [{"date": D("2023-03-31"), "symbol": f"S{i}", "sigma_score": i, "fwd": i / 100}
                for i in range(50)]
        q = analysis.quintile_returns(pd.DataFrame(rows), "sigma_score")
        self.assertEqual(list(q.columns), [1, 2, 3, 4, 5])
        self.assertTrue(q.iloc[0].is_monotonic_increasing)

    def test_performance_stats(self):
        r = pd.Series([0.05, -0.02, 0.03, 0.04], index=[D("2023-03-31"), D("2023-06-30"), D("2023-09-29"),
                                                        D("2023-12-29")])
        s = analysis.performance(r, periods_per_year=4)
        self.assertAlmostEqual(s["total_return"], 1.05 * 0.98 * 1.03 * 1.04 - 1)
        self.assertAlmostEqual(s["cagr"], (1.05 * 0.98 * 1.03 * 1.04) - 1)  # exactly one year
        self.assertLess(s["max_drawdown"], 0)
        self.assertEqual(s["periods"], 4)


if __name__ == "__main__":
    unittest.main()


def synthetic_panel(signal=True, periods=12, names=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = [d.date() for d in pd.date_range("2019-03-31", periods=periods, freq="QE")]
    rows = []
    for d in dates:
        score = rng.permutation(names).astype(float)
        noise = rng.normal(0, 0.05, names)
        fwd = (score / names * 0.10 if signal else 0.0) + noise
        for i in range(names):
            rows.append({"date": d, "symbol": f"S{i}", "sigma_score": score[i], "cat_Growth": score[i],
                         "cat_Valuation": rng.normal(), "fwd": fwd[i], "ended": False})
    return pd.DataFrame(rows)


class EvaluateTests(unittest.TestCase):
    def evaluate(self, panel, **kw):
        from sigma.backtest import run
        bench = pd.DataFrame({"SPY": 0.02}, index=sorted(panel["date"].unique()))
        return run.evaluate(panel, bench, n=10, splits=splits.Splits(D("2020-12-31"), D("2023-12-31")), **kw)

    def test_a_real_signal_shows_positive_ic_and_beats_the_universe(self):
        res = self.evaluate(synthetic_panel(signal=True))
        ic = res["ic"]["train"].loc["sigma_score"]
        self.assertGreater(ic["mean"], 0.3)
        self.assertGreater(res["portfolio"]["train"]["base"]["cagr"], res["universe"]["train"]["cagr"])

    def test_no_signal_shows_no_edge(self):
        res = self.evaluate(synthetic_panel(signal=False, periods=12, seed=3))
        self.assertLess(abs(res["ic"]["train"].loc["sigma_score", "mean"]), 0.15)

    def test_costs_lower_returns_in_the_order_of_the_presets(self):
        res = self.evaluate(synthetic_panel(signal=True))["portfolio"]["train"]
        self.assertGreater(res["optimistic"]["cagr"], res["base"]["cagr"])
        self.assertGreater(res["base"]["cagr"], res["pessimistic"]["cagr"])

    def test_holdout_dates_are_refused(self):
        p = synthetic_panel(periods=20)  # runs into 2023-2024 and beyond
        p2 = pd.concat([p, p.assign(date=D("2024-06-28"))])
        with self.assertRaises(splits.HoldoutLocked):
            self.evaluate(p2)

    def test_render_mentions_the_universe_and_every_benchmark(self):
        from sigma.backtest import run
        res = self.evaluate(synthetic_panel())
        text = run.render(res, top_n=10)
        for needle in ("train", "validation", "SPY", "universe", "pessimistic", "sigma_score"):
            self.assertIn(needle, text)
