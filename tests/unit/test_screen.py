"""Provisional screen: joins, investability filters, funnel counts, ranking."""

import unittest
from datetime import date

import numpy as np
import pandas as pd

from sigma import screen
from tests.unit.test_factors_scoring import company

FUND_COLS = [k for k in company() if k not in {"price", "mom_12_1", "mom_6_1", "vol_1y", "max_dd_1y"}]


def make(n=30):
    rng = np.random.default_rng(1)
    uni, mkt, fund, sec = [], [], [], []
    for i in range(n):
        sym, cik = f"S{i:02d}", 1000 + i
        c = company(revenue_1y=1000 / (1 + rng.uniform(-0.2, 0.5)), mom_12_1=rng.normal(0.1, 0.3))
        uni.append({"symbol": sym, "kind": "common", "cik": cik, "name": f"Co {i}", "exchange": "NYSE"})
        mkt.append({"symbol": sym, "last_date": date(2026, 10, 6), "price": c["price"], "adv_usd": 5e6,
                    "mom_12_1": c["mom_12_1"], "mom_6_1": c["mom_6_1"], "vol_1y": c["vol_1y"],
                    "max_dd_1y": c["max_dd_1y"]})
        fund.append({"cik": cik, "fiscal_end": date(2026, 6, 30)} | {k: c[k] for k in FUND_COLS})
        sec.append({"cik": cik, "sic": 7372, "sector": "Information Technology", "sic_description": "x"})
    return pd.DataFrame(uni), pd.DataFrame(mkt).set_index("symbol"), \
        pd.DataFrame(fund).set_index("cik"), pd.DataFrame(sec)


class ScreenTests(unittest.TestCase):
    def setUp(self):
        self.uni, self.mkt, self.fund, self.sec = make()
        self.f = screen.Filters(min_market_cap=1e3, min_adv_usd=1e6, min_price=2.0, stage1=10, stage2=5)

    def run_screen(self):
        return screen.build_screen(self.fund, self.mkt, self.uni, self.sec, self.f)

    def test_ranked_with_stages(self):
        ranked, funnel = self.run_screen()
        self.assertEqual(len(ranked), 30)
        self.assertTrue(ranked["sigma_score"].is_monotonic_decreasing)
        self.assertEqual(list(ranked["rank"][:3]), [1, 2, 3])
        self.assertEqual((ranked["stage"] == "advance").sum(), 5)
        self.assertEqual((ranked["stage"] == "candidate").sum(), 5)
        self.assertEqual(funnel[-1], ("scored", 30))

    def test_filters_are_counted(self):
        self.mkt.loc["S00", "adv_usd"] = 10.0           # illiquid
        self.sec.loc[self.sec["cik"] == 1001, "sector"] = "Shell"   # SPAC
        self.fund = self.fund.drop(index=1002)          # no current filings
        ranked, funnel = self.run_screen()
        steps = dict(funnel)
        self.assertEqual(steps["has current SEC filings"], 29)
        self.assertEqual(steps["not a shell / SPAC"], 28)
        self.assertEqual(steps["liquid (median daily $ volume)"], 27)
        self.assertNotIn("S00", set(ranked["symbol"]))

    def test_one_listing_per_company_keeps_the_most_liquid(self):
        extra = self.uni.iloc[[3]].assign(symbol="S03B")
        self.uni = pd.concat([self.uni, extra], ignore_index=True)
        self.mkt.loc["S03B"] = self.mkt.loc["S03"]
        self.mkt.loc["S03B", "adv_usd"] = 9e9
        ranked, _ = self.run_screen()
        self.assertIn("S03B", set(ranked["symbol"]))
        self.assertNotIn("S03", set(ranked["symbol"]))

    def test_non_common_excluded(self):
        self.uni.loc[0, "kind"] = "etf"
        ranked, _ = self.run_screen()
        self.assertNotIn("S00", set(ranked["symbol"]))


class ReportTests(unittest.TestCase):
    def test_report_is_labelled_and_lists_the_top(self):
        uni, mkt, fund, sec = make()
        f = screen.Filters(min_market_cap=1e3, min_adv_usd=1e6, stage1=10, stage2=5)
        ranked, funnel = screen.build_screen(fund, mkt, uni, sec, f)
        text = screen.report(ranked, funnel, date(2026, 10, 6), f, top=3)
        self.assertIn("EXPERIMENTAL, NOT VALIDATED", text)
        self.assertIn("not scored", text)
        self.assertIn(ranked["symbol"].iloc[0], text)
        self.assertIn("Information Technology", text)


if __name__ == "__main__":
    unittest.main()
