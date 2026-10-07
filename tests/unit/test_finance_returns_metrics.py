"""Returns and risk metrics. Expected values are hand-derived (shown inline)."""

import math
import unittest

import numpy as np

from sigma.finance import metrics as m
from sigma.finance import returns as r


class ReturnsTests(unittest.TestCase):
    def test_simple_and_log_returns(self):
        p = [100.0, 110.0, 99.0]
        np.testing.assert_allclose(r.simple_returns(p), [0.10, -0.10])
        np.testing.assert_allclose(r.log_returns(p), [math.log(1.1), math.log(0.9)])

    def test_cumulative_return(self):
        # 1.1 * 0.9 - 1 = -0.01
        self.assertAlmostEqual(r.cumulative_return([0.10, -0.10]), -0.01)

    def test_cagr_from_values(self):
        # doubling over 5 years: 2 ** (1/5) - 1
        self.assertAlmostEqual(r.cagr(100, 200, 5), 0.1486983550, places=9)

    def test_cagr_from_returns_matches_values(self):
        rets = np.full(252 * 2, (1.21 ** (1 / 504)) - 1)  # exactly +10%/yr for 2 years
        self.assertAlmostEqual(r.cagr_from_returns(rets, 252), 0.10, places=10)

    def test_cagr_total_loss(self):
        self.assertEqual(r.cagr(100, 0, 3), -1.0)

    def test_cagr_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            r.cagr(0, 100, 1)
        with self.assertRaises(ValueError):
            r.cagr(100, 120, 0)

    def test_annualized_volatility(self):
        x = [0.01, -0.01, 0.02, 0.0]
        # sample std (ddof=1) = sqrt(5e-4 / 3)
        self.assertAlmostEqual(r.annualized_volatility(x, 252), math.sqrt(5e-4 / 3) * math.sqrt(252))

    def test_total_returns_reinvest_dividend_on_ex_date(self):
        # ex-date at index 2: (99 + 1) / 110 - 1; other days are price-only
        tr = r.total_returns([100.0, 110.0, 99.0], {2: 1.0})
        np.testing.assert_allclose(tr, [0.10, 100.0 / 110.0 - 1])

    def test_total_returns_reject_dividend_on_first_bar(self):
        with self.assertRaises(ValueError):
            r.total_returns([100.0, 101.0], {0: 1.0})

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            r.simple_returns([100, float("nan"), 101])


class MetricTests(unittest.TestCase):
    X = (0.01, -0.01, 0.02, 0.0)

    def test_sharpe(self):
        # mean 0.005, sample std sqrt(5e-4/3) → daily 0.3872983; * sqrt(252)
        expected = 0.005 / math.sqrt(5e-4 / 3) * math.sqrt(252)
        self.assertAlmostEqual(m.sharpe(self.X, periods_per_year=252), expected)

    def test_sharpe_with_risk_free(self):
        rf_annual = 0.0252
        rf_daily = (1 + rf_annual) ** (1 / 252) - 1
        ex = np.array(self.X) - rf_daily
        expected = ex.mean() / ex.std(ddof=1) * math.sqrt(252)
        self.assertAlmostEqual(m.sharpe(self.X, risk_free_annual=rf_annual), expected)

    def test_sharpe_zero_vol_is_nan(self):
        self.assertTrue(math.isnan(m.sharpe([0.01, 0.01, 0.01])))

    def test_sortino(self):
        # downside deviation (MAR 0) = sqrt(mean([0, 1e-4, 0, 0])) = 0.005
        self.assertAlmostEqual(m.sortino(self.X), 0.005 / 0.005 * math.sqrt(252))

    def test_max_drawdown_and_recovery(self):
        nav = [100, 120, 90, 95, 130]
        dd = m.drawdown_stats(nav)
        self.assertAlmostEqual(dd.max_drawdown, -0.25)
        self.assertEqual((dd.peak_index, dd.trough_index, dd.recovery_index), (1, 2, 4))
        self.assertEqual(dd.recovery_periods, 2)

    def test_drawdown_never_recovered(self):
        dd = m.drawdown_stats([100, 80, 90])
        self.assertAlmostEqual(dd.max_drawdown, -0.20)
        self.assertIsNone(dd.recovery_index)

    def test_max_drawdown_from_returns_includes_start(self):
        # starting value counts as a peak: a first-day -10% is a 10% drawdown
        self.assertAlmostEqual(m.max_drawdown_from_returns([-0.10, 0.05]), -0.10)

    def test_calmar(self):
        rets = [0.10, -0.20, 0.30]
        years = 3 / 252
        growth = 1.1 * 0.8 * 1.3
        expected = (growth ** (1 / years) - 1) / 0.20
        self.assertAlmostEqual(m.calmar(rets, 252), expected)

    def test_beta_alpha_correlation(self):
        rng = np.random.default_rng(7)
        bench = rng.normal(0.0004, 0.01, 1000)
        asset = 0.0001 + 2.0 * bench
        self.assertAlmostEqual(m.beta(asset, bench), 2.0, places=10)
        self.assertAlmostEqual(m.correlation(asset, bench), 1.0, places=10)
        self.assertAlmostEqual(m.alpha_annualized(asset, bench, 252), 0.0001 * 252, places=10)

    def test_capture_ratios(self):
        bench = np.array([0.02, -0.02, 0.01, -0.01])
        asset = np.array([0.01, -0.01, 0.005, -0.005])  # half of benchmark
        self.assertAlmostEqual(m.downside_capture(asset, bench), 0.5, places=2)
        self.assertAlmostEqual(m.upside_capture(asset, bench), 0.5, places=2)

    def test_var_es_historical(self):
        x = np.arange(-50, 50) / 1000.0  # -0.050 .. 0.049, 100 obs
        # 5% left tail = 5 worst observations: -0.050..-0.046; ES = mean = -0.048
        self.assertAlmostEqual(m.expected_shortfall(x, 0.95), -0.048)
        self.assertAlmostEqual(m.value_at_risk(x, 0.95), -0.046)

    def test_hit_rate(self):
        self.assertAlmostEqual(m.hit_rate([0.1, -0.1, 0.0, 0.2]), 0.5)

    def test_turnover(self):
        w0 = {"A": 0.5, "B": 0.5}
        w1 = {"A": 0.3, "B": 0.5, "C": 0.2}
        # one-way turnover = 0.5 * (0.2 + 0 + 0.2) = 0.2
        self.assertAlmostEqual(m.turnover(w0, w1), 0.2)

    def test_brier(self):
        self.assertAlmostEqual(m.brier_score([0.7, 0.2], [1, 0]), (0.09 + 0.04) / 2)

    def test_summary_keys(self):
        s = m.performance_summary([0.01, -0.005, 0.007, 0.002] * 100,
                                  benchmark=[0.005, -0.002, 0.003, 0.0] * 100)
        for key in ("cagr", "volatility", "sharpe", "sortino", "calmar", "max_drawdown", "beta", "alpha",
                    "hit_rate", "es_97_5", "downside_capture"):
            self.assertIn(key, s)


if __name__ == "__main__":
    unittest.main()
