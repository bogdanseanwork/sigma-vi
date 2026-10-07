"""Ratios, valuation, scenarios, return engine. Expected values hand-derived inline."""

import math
import unittest

from sigma.finance import ratios as ra
from sigma.finance import return_engine as re_
from sigma.finance import scenarios as sc
from sigma.finance import valuation as v


class RatioTests(unittest.TestCase):
    def test_fcf_accepts_either_capex_sign(self):
        self.assertEqual(ra.free_cash_flow(500, 120), 380)
        self.assertEqual(ra.free_cash_flow(500, -120), 380)

    def test_margins_and_conversion(self):
        self.assertAlmostEqual(ra.margin(380, 1000), 0.38)
        self.assertAlmostEqual(ra.fcf_conversion(380, 400), 0.95)
        self.assertTrue(math.isnan(ra.margin(1, 0)))

    def test_roic(self):
        # NOPAT = 200 * (1 - 0.25) = 150; / 1000
        self.assertAlmostEqual(ra.roic(ebit=200, tax_rate=0.25, invested_capital=1000), 0.15)

    def test_growth_series(self):
        g = ra.yoy_growth([100, 110, 99])
        self.assertAlmostEqual(g[0], 0.10)
        self.assertAlmostEqual(g[1], -0.10)

    def test_growth_from_negative_base_is_nan(self):
        self.assertTrue(math.isnan(ra.yoy_growth([-10, 5])[0]))

    def test_working_capital_divergence_flags(self):
        # receivables +30% vs sales +10% → flagged at 10pp threshold
        self.assertTrue(ra.divergence_flag(item_growth=0.30, sales_growth=0.10))
        self.assertFalse(ra.divergence_flag(item_growth=0.15, sales_growth=0.10))

    def test_net_debt_and_dilution(self):
        self.assertEqual(ra.net_debt(debt=300, cash=500), -200)
        # 100 → 110 shares over 2 years: sqrt(1.1) - 1
        self.assertAlmostEqual(ra.annual_dilution(100, 110, 2), math.sqrt(1.1) - 1)


class MultipleTests(unittest.TestCase):
    def test_enterprise_value(self):
        self.assertEqual(v.enterprise_value(market_cap=1000, debt=200, cash=50, minority_interest=10), 1160)

    def test_multiples(self):
        self.assertAlmostEqual(v.pe(price=50, eps=2.5), 20)
        self.assertTrue(math.isnan(v.pe(price=50, eps=-1)))
        self.assertAlmostEqual(v.ev_to(ev=1160, metric=116), 10)
        self.assertAlmostEqual(v.fcf_yield(fcf=50, market_cap=1000), 0.05)
        # PEG: P/E 20 / growth 25% → 0.8
        self.assertAlmostEqual(v.peg(pe_ratio=20, eps_growth=0.25), 0.8)

    def test_wacc_and_capm(self):
        ke = v.capm_cost_of_equity(risk_free=0.04, beta=1.2, equity_risk_premium=0.05)
        self.assertAlmostEqual(ke, 0.10)
        # 0.8 * 0.10 + 0.2 * 0.05 * (1 - 0.2) = 0.088
        self.assertAlmostEqual(v.wacc(equity_value=800, debt_value=200, cost_of_equity=0.10,
                                      cost_of_debt=0.05, tax_rate=0.20), 0.088)


class DCFTests(unittest.TestCase):
    FCF = (100.0, 110.0, 121.0)

    def test_gordon_dcf(self):
        res = v.dcf(self.FCF, discount_rate=0.10, terminal_growth=0.02)
        # each explicit FCF discounts to 90.909..., total 272.727...
        self.assertAlmostEqual(res.pv_explicit, 300 / 1.1, places=6)
        tv = 121 * 1.02 / 0.08
        self.assertAlmostEqual(res.terminal_value, tv)
        self.assertAlmostEqual(res.pv_terminal, tv / 1.1**3)
        self.assertAlmostEqual(res.enterprise_value, 300 / 1.1 + tv / 1.1**3)
        self.assertAlmostEqual(res.terminal_share, res.pv_terminal / res.enterprise_value)

    def test_per_share(self):
        res = v.dcf(self.FCF, discount_rate=0.10, terminal_growth=0.02)
        ps = v.equity_value_per_share(res.enterprise_value, net_debt=200, diluted_shares=10)
        self.assertAlmostEqual(ps, (res.enterprise_value - 200) / 10)

    def test_exit_multiple(self):
        res = v.dcf(self.FCF, discount_rate=0.10, exit_multiple=15, terminal_metric=150)
        self.assertAlmostEqual(res.terminal_value, 2250)
        self.assertAlmostEqual(res.pv_terminal, 2250 / 1.331)

    def test_mid_year_convention(self):
        res = v.dcf(self.FCF, discount_rate=0.10, terminal_growth=0.02, mid_year=True)
        self.assertAlmostEqual(res.pv_explicit, (300 / 1.1) * math.sqrt(1.1), places=6)

    def test_rejects_growth_at_or_above_discount(self):
        with self.assertRaises(ValueError):
            v.dcf(self.FCF, discount_rate=0.05, terminal_growth=0.05)

    def test_requires_exactly_one_terminal_method(self):
        with self.assertRaises(ValueError):
            v.dcf(self.FCF, discount_rate=0.1)
        with self.assertRaises(ValueError):
            v.dcf(self.FCF, discount_rate=0.1, terminal_growth=0.02, exit_multiple=10, terminal_metric=1)


class ReverseDCFTests(unittest.TestCase):
    def test_recovers_implied_fcf_growth(self):
        kw = dict(base_fcf=100.0, years=10, discount_rate=0.09, terminal_growth=0.025,
                  net_debt=-500.0, diluted_shares=50.0)
        price = v.price_from_fcf_growth(growth=0.14, **kw)
        res = v.reverse_dcf_fcf_growth(price=price, **kw)
        self.assertAlmostEqual(res.implied_growth, 0.14, places=8)
        self.assertTrue(res.converged)

    def test_recovers_implied_revenue_growth_with_margin_ramp(self):
        kw = dict(base_revenue=1000.0, start_fcf_margin=0.10, target_fcf_margin=0.25, years=10,
                  discount_rate=0.10, terminal_growth=0.03, net_debt=0.0, diluted_shares=100.0)
        price = v.price_from_revenue_growth(revenue_growth=0.18, **kw)
        res = v.reverse_dcf_revenue_growth(price=price, **kw)
        self.assertAlmostEqual(res.implied_growth, 0.18, places=8)

    def test_unreachable_price_reports_not_converged(self):
        res = v.reverse_dcf_fcf_growth(price=1e12, base_fcf=1.0, years=5, discount_rate=0.1,
                                       terminal_growth=0.02, net_debt=0, diluted_shares=1)
        self.assertFalse(res.converged)


class ScenarioTests(unittest.TestCase):
    def test_probability_weighted(self):
        scenarios = [sc.Scenario("bear", 0.25, 60.0), sc.Scenario("base", 0.50, 100.0),
                     sc.Scenario("bull", 0.25, 150.0)]
        a = sc.analyze(scenarios, price=90.0)
        self.assertAlmostEqual(a.expected_value, 102.5)
        self.assertAlmostEqual(a.expected_return, 102.5 / 90 - 1)
        self.assertAlmostEqual(a.downside_return, 60 / 90 - 1)
        self.assertAlmostEqual(a.upside_return, 150 / 90 - 1)
        # upside/downside skew: (E[gain|gain] weighted) / (E[loss|loss] weighted)
        gains = 0.5 * (100 - 90) + 0.25 * (150 - 90)
        losses = 0.25 * (90 - 60)
        self.assertAlmostEqual(a.reward_to_risk, gains / losses)
        self.assertAlmostEqual(a.prob_loss, 0.25)

    def test_probabilities_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            sc.analyze([sc.Scenario("a", 0.5, 1), sc.Scenario("b", 0.4, 2)], price=1)


class ReturnEngineTests(unittest.TestCase):
    def test_exact_decomposition(self):
        d = re_.decompose(revenue_growth=0.10, margin_start=0.10, margin_end=0.11, share_change=-0.02,
                          multiple_start=20, multiple_end=18, shareholder_yield=0.01, years=1)
        eps_growth = 1.1 * 1.1 / 0.98 - 1
        self.assertAlmostEqual(d.eps_growth, eps_growth)
        price_return = (1 + eps_growth) * 0.9 - 1
        self.assertAlmostEqual(d.price_return, price_return)
        self.assertAlmostEqual(d.total_return, price_return + 0.01)
        # log contributions sum exactly to log price return
        self.assertAlmostEqual(sum(d.log_contributions.values()), math.log(1 + price_return))

    def test_annualizes_over_years(self):
        d = re_.decompose(revenue_growth=0.10, margin_start=0.2, margin_end=0.2, share_change=0.0,
                          multiple_start=15, multiple_end=15, shareholder_yield=0.0, years=3)
        self.assertAlmostEqual(d.price_return, 0.10)


if __name__ == "__main__":
    unittest.main()
