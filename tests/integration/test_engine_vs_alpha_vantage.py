"""Reconcile the deterministic engine with an independent implementation on real data.

Inputs (fixtures): Massive split-adjusted daily closes and cash dividends for AAPL and SPY, 2025.
Reference: Alpha Vantage ANALYTICS_FIXED_WINDOW over the same range, computed server-side.

Findings encoded below (2026-10-07):
* AV returns are TOTAL returns (dividends reinvested). With ``total_returns`` our mean and
  cumulative returns match AV to ~1e-10.
* AV STDDEV is the population std (ddof=0). The engine uses sample std (ddof=1) everywhere; the two
  differ by sqrt(n/(n-1)) and match exactly once that factor is applied.
* AV MAX_DRAWDOWN is the worst run of CONSECUTIVE down closes, not peak-to-trough. It understates
  risk (AAPL 2025: -23.0% vs the true -30.2% peak-to-trough from 24 Feb to 8 Apr), so SIGMA never
  consumes it for risk limits; the engine's peak-to-trough figure is checked by hand below.
"""

import csv
import math
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

from sigma.finance import metrics as m
from sigma.finance import returns as r

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

ALPHA_VANTAGE = {  # retrieved 2026-10-07, range 2025-01-02..2025-12-31, close
    "AAPL": {"mean": 0.0006601952546375526, "stddev": 0.02040831499425894,
             "max_drawdown": -0.22988967796685889, "cumulative_return": 0.11982515627336676},
    "SPY": {"mean": 0.0007395921904658089, "stddev": 0.012273734941869967,
            "max_drawdown": -0.12052717352795295, "cumulative_return": 0.1800499199593235},
    "correlation": 0.7536035878,
}


def _rows(name):
    with open(FIXTURES / name) as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def load_prices():
    rows = _rows("aapl_spy_2025_daily_close.csv")
    return {t: np.array([float(x[t.lower()]) for x in rows]) for t in ("AAPL", "SPY")}


def load_dividends():
    out = defaultdict(dict)
    for x in _rows("aapl_spy_2025_dividends.csv"):
        out[x["ticker"]][int(x["ex_index"])] = float(x["cash_amount"])
    return out


def worst_consecutive_decline(prices):
    worst, start = 0.0, 0
    for i in range(1, len(prices)):
        if prices[i] >= prices[i - 1]:
            start = i
        worst = min(worst, prices[i] / prices[start] - 1)
    return worst


class EngineReconcilesWithAlphaVantage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = load_prices()
        divs = load_dividends()
        cls.tr = {t: r.total_returns(p, divs[t]) for t, p in cls.prices.items()}

    def test_mean_total_return(self):
        for t in ("AAPL", "SPY"):
            self.assertAlmostEqual(self.tr[t].mean(), ALPHA_VANTAGE[t]["mean"], places=12, msg=t)

    def test_cumulative_total_return(self):
        for t in ("AAPL", "SPY"):
            self.assertAlmostEqual(r.cumulative_return(self.tr[t]), ALPHA_VANTAGE[t]["cumulative_return"],
                                   places=10, msg=t)

    def test_stddev_differs_only_by_bessel_correction(self):
        for t in ("AAPL", "SPY"):
            x = self.tr[t]
            population = x.std(ddof=1) * math.sqrt((x.size - 1) / x.size)
            self.assertAlmostEqual(population, ALPHA_VANTAGE[t]["stddev"], places=12, msg=t)

    def test_correlation(self):
        self.assertAlmostEqual(m.correlation(self.tr["AAPL"], self.tr["SPY"]),
                               ALPHA_VANTAGE["correlation"], places=9)

    def test_av_drawdown_is_a_losing_streak_measure(self):
        for t in ("AAPL", "SPY"):
            self.assertAlmostEqual(worst_consecutive_decline(self.prices[t]),
                                   ALPHA_VANTAGE[t]["max_drawdown"], places=12, msg=t)

    def test_engine_drawdown_is_true_peak_to_trough(self):
        dd = m.drawdown_stats(self.prices["AAPL"])
        self.assertEqual((self.prices["AAPL"][dd.peak_index], self.prices["AAPL"][dd.trough_index]),
                         (247.10, 172.42))
        self.assertAlmostEqual(dd.max_drawdown, 172.42 / 247.10 - 1)
        self.assertLess(dd.max_drawdown, ALPHA_VANTAGE["AAPL"]["max_drawdown"])


if __name__ == "__main__":
    unittest.main()
