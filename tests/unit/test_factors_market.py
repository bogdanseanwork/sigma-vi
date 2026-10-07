"""Price-based features and the SIC-to-sector map."""

import math
import unittest
from datetime import date

import numpy as np
import pandas as pd

from sigma.factors import market, sectors


def bars(symbol, closes, volume=1000, start="2023-01-02"):
    days = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"symbol": symbol, "date": [d.date() for d in days],
                         "close": closes, "volume": volume})


class MarketFeatureTests(unittest.TestCase):
    def setUp(self):
        # 300 trading days of +0.1%/day compounding, then a 20% drop and partial recovery.
        up = list(100 * 1.001 ** np.arange(300))
        self.closes = up + [up[-1] * 0.8, up[-1] * 0.9]
        self.adj = bars("AAA", self.closes)
        self.raw = bars("AAA", [c * 2 for c in self.closes], volume=500)  # pre-split raw prices
        self.as_of = self.adj["date"].iloc[-1]

    def feats(self, **kw):
        return market.features(self.adj, self.raw, self.as_of, **kw).loc["AAA"]

    def test_momentum_skips_the_latest_month(self):
        c = self.closes
        expected = c[-1 - 21] / c[-1 - 252] - 1
        self.assertAlmostEqual(self.feats()["mom_12_1"], expected, places=10)

    def test_dollar_volume_and_price_use_raw_prices(self):
        f = self.feats()
        self.assertAlmostEqual(f["price"], self.closes[-1] * 2)
        self.assertAlmostEqual(f["adv_usd"], float(np.median([c * 2 * 500 for c in self.closes[-63:]])))

    def test_max_drawdown_over_last_year(self):
        self.assertAlmostEqual(self.feats()["max_dd_1y"], 0.2, places=10)

    def test_volatility_is_annualised(self):
        r = np.diff(np.log(self.closes[-253:]))
        self.assertAlmostEqual(self.feats()["vol_1y"], float(np.std(r, ddof=1) * math.sqrt(252)))

    def test_future_bars_are_ignored(self):
        cut = self.adj["date"].iloc[200]
        f = market.features(self.adj, self.raw, cut).loc["AAA"]
        self.assertEqual(f["last_date"], cut)
        self.assertTrue(math.isnan(f["mom_12_1"]))  # not enough history yet: missing, not invented

    def test_symbols_with_no_recent_bar_are_dropped(self):
        later = date(2030, 1, 1)
        self.assertNotIn("AAA", market.features(self.adj, self.raw, later, max_stale_days=7).index)


class SectorTests(unittest.TestCase):
    def test_known_codes(self):
        cases = {3674: "Information Technology", 7372: "Information Technology", 2834: "Health Care",
                 1311: "Energy", 6022: "Financials", 6798: "Real Estate", 4911: "Utilities",
                 2080: "Consumer Staples", 5961: "Consumer Discretionary", 4813: "Communication Services",
                 3711: "Consumer Discretionary", 3721: "Industrials", 2800: "Materials", 6770: "Shell"}
        for sic, want in cases.items():
            self.assertEqual(sectors.sector_for_sic(sic), want, sic)

    def test_unknown(self):
        self.assertEqual(sectors.sector_for_sic(None), "Unknown")
        self.assertEqual(sectors.sector_for_sic(9999), "Other")


if __name__ == "__main__":
    unittest.main()
