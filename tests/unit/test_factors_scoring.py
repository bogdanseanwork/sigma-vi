"""Factor definitions, normalisation and the provisional 0-100 score."""

import math
import unittest

import numpy as np
import pandas as pd

from sigma.factors import scoring as S


def company(**kw):
    base = dict(revenue=1000.0, revenue_1y=800.0, revenue_3y=500.0, gross_profit=600.0, cost_of_revenue=None,
                operating_income=200.0, operating_income_1y=150.0, net_income=150.0, net_income_1y=100.0,
                cfo=220.0, cfo_1y=150.0, capex=40.0, capex_1y=30.0, sbc=20.0, dna=50.0, interest_expense=10.0,
                total_assets=2000.0, total_assets_1y=1800.0, equity=1000.0, equity_1y=900.0,
                cash=300.0, short_term_investments=0.0, debt_current=100.0, debt_noncurrent=400.0,
                current_assets=900.0, current_liabilities=450.0, shares=100.0, shares_1y=102.0,
                price=30.0, mom_12_1=0.25, mom_6_1=0.1, vol_1y=0.3, max_dd_1y=0.2)
    base.update(kw)
    return base


class FactorTests(unittest.TestCase):
    def setUp(self):
        self.f = S.compute_factors(pd.DataFrame([company()])).iloc[0]

    def test_growth(self):
        self.assertAlmostEqual(self.f["rev_growth"], 0.25)
        self.assertAlmostEqual(self.f["rev_cagr_3y"], 2 ** (1 / 3) - 1)
        self.assertAlmostEqual(self.f["op_income_growth"], 50 / 150)

    def test_valuation_uses_market_cap_and_enterprise_value(self):
        mcap = 3000.0
        ev = mcap + 500 - 300
        self.assertAlmostEqual(self.f["earnings_yield"], 150 / mcap)
        self.assertAlmostEqual(self.f["fcf_yield"], 180 / mcap)
        self.assertAlmostEqual(self.f["ebitda_to_ev"], 250 / ev)

    def test_lower_is_better_factors_are_sign_flipped(self):
        self.assertAlmostEqual(self.f["low_volatility"], -0.3)
        self.assertAlmostEqual(self.f["share_count_change"], -(100 / 102 - 1))
        self.assertAlmostEqual(self.f["net_debt_to_ebitda"], -(200 / 250))

    def test_growth_from_a_negative_base_is_measured_against_its_size(self):
        f = S.compute_factors(pd.DataFrame([company(operating_income=50.0, operating_income_1y=-100.0)])).iloc[0]
        self.assertAlmostEqual(f["op_income_growth"], 1.5)

    def test_undefined_ratios_are_missing_not_zero(self):
        f = S.compute_factors(pd.DataFrame([company(revenue_3y=None, equity=-50.0, equity_1y=-60.0)])).iloc[0]
        self.assertTrue(math.isnan(f["rev_cagr_3y"]))
        self.assertTrue(math.isnan(f["roe"]))

    def test_negative_ebitda_with_debt_is_penalised_not_skipped(self):
        f = S.compute_factors(pd.DataFrame([company(operating_income=-100.0, dna=10.0)])).iloc[0]
        self.assertEqual(f["net_debt_to_ebitda"], S.WORST_LEVERAGE)


class NormalisationTests(unittest.TestCase):
    def test_winsorize(self):
        s = pd.Series(list(range(100)) + [10_000])
        w = S.winsorize(s, 0.01)
        self.assertLess(w.max(), 10_000)
        self.assertEqual(w.iloc[50], 50)

    def test_sector_z_is_centered_within_each_sector(self):
        df = pd.DataFrame({"x": list(range(20)) + list(range(100, 120)),
                           "sector": ["A"] * 20 + ["B"] * 20})
        z = S.sector_z(df[["x"]], df["sector"], min_group=5)
        self.assertAlmostEqual(z["x"][df["sector"] == "A"].mean(), 0)
        self.assertAlmostEqual(z["x"][df["sector"] == "B"].mean(), 0)
        self.assertAlmostEqual(z.loc[0, "x"], z.loc[20, "x"])

    def test_small_sectors_use_the_whole_universe(self):
        df = pd.DataFrame({"x": list(range(20)) + [1000.0], "sector": ["A"] * 20 + ["Tiny"]})
        z = S.sector_z(df[["x"]], df["sector"], min_group=5)
        self.assertGreater(z.loc[20, "x"], 1)  # compared with everyone, not just itself


class ScoreTests(unittest.TestCase):
    def universe(self, n=40):
        rng = np.random.default_rng(0)
        rows = [company(revenue_1y=1000 / (1 + g), mom_12_1=m, vol_1y=v)
                for g, m, v in zip(rng.uniform(-0.2, 0.6, n), rng.normal(0.1, 0.3, n), rng.uniform(0.15, 0.8, n),
                                   strict=True)]
        df = pd.DataFrame(rows)
        df["sector"] = ["A", "B"] * (n // 2)
        return df

    def test_weights_exclude_categories_without_free_data_and_sum_to_one(self):
        w = S.effective_weights()
        self.assertNotIn("Earnings Revisions", w)
        self.assertNotIn("Catalysts", w)
        self.assertAlmostEqual(sum(w.values()), 1.0)
        self.assertAlmostEqual(w["Growth"], 0.15 / 0.80)

    def test_scores_are_0_to_100_and_reward_the_better_company(self):
        df = self.universe()
        best = company(revenue_1y=500.0, operating_income_1y=50.0, net_income_1y=40.0, mom_12_1=0.9, vol_1y=0.12, max_dd_1y=0.05)
        worst = company(revenue_1y=2000.0, mom_12_1=-0.6, vol_1y=1.2, max_dd_1y=0.7)
        df = pd.concat([df, pd.DataFrame([best | {"sector": "A"}, worst | {"sector": "A"}])], ignore_index=True)
        out = S.score(df)
        self.assertTrue(out["sigma_score"].between(0, 100).all())
        self.assertGreater(out["sigma_score"].iloc[-2], out["sigma_score"].iloc[-1])
        self.assertEqual(out["sigma_score"].idxmax(), len(df) - 2)

    def test_thin_coverage_gets_no_score(self):
        df = self.universe()
        empty = {k: None for k in company()} | {"sector": "A", "mom_12_1": 0.3}
        df = pd.concat([df, pd.DataFrame([empty])], ignore_index=True)
        out = S.score(df)
        self.assertTrue(math.isnan(out["sigma_score"].iloc[-1]))
        self.assertLess(out["coverage"].iloc[-1], S.MIN_COVERAGE)

    def test_category_columns_reported(self):
        out = S.score(self.universe())
        for cat in S.effective_weights():
            self.assertIn(f"cat_{cat}", out.columns)


if __name__ == "__main__":
    unittest.main()
