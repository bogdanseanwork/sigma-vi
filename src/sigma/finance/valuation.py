"""Valuation: enterprise value, multiples, cost of capital, DCF and reverse DCF (spec §12, §15).

Reverse DCF answers the mandatory question "what does today's price already require?" by solving
for the growth rate that equates a DCF to the market price, holding every other input fixed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from scipy.optimize import brentq

# ---------------------------------------------------------------------------------------------
# Enterprise value and multiples
# ---------------------------------------------------------------------------------------------


def enterprise_value(
    market_cap: float, debt: float, cash: float, minority_interest: float = 0.0, preferred: float = 0.0
) -> float:
    return market_cap + debt - cash + minority_interest + preferred


def pe(price: float, eps: float) -> float:
    """P/E; NaN when EPS is non-positive (a negative P/E is not a valuation)."""
    return math.nan if eps <= 0 else price / eps


def ev_to(ev: float, metric: float) -> float:
    """EV / EBITDA, EV / EBIT, EV / Sales — NaN when the denominator is non-positive."""
    return math.nan if metric <= 0 else ev / metric


def fcf_yield(fcf: float, market_cap: float) -> float:
    return math.nan if market_cap <= 0 else fcf / market_cap


def peg(pe_ratio: float, eps_growth: float) -> float:
    """P/E divided by growth in percentage points (0.25 → 25). NaN for non-positive growth."""
    if eps_growth <= 0 or not math.isfinite(pe_ratio):
        return math.nan
    return pe_ratio / (eps_growth * 100.0)


def capm_cost_of_equity(risk_free: float, beta: float, equity_risk_premium: float) -> float:
    return risk_free + beta * equity_risk_premium


def wacc(
    equity_value: float, debt_value: float, cost_of_equity: float, cost_of_debt: float, tax_rate: float
) -> float:
    total = equity_value + debt_value
    if total <= 0:
        raise ValueError("equity_value + debt_value must be positive")
    return (equity_value / total) * cost_of_equity + (debt_value / total) * cost_of_debt * (1 - tax_rate)


# ---------------------------------------------------------------------------------------------
# DCF
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DCFResult:
    pv_explicit: float
    terminal_value: float        # undiscounted, at end of final explicit year
    pv_terminal: float
    enterprise_value: float

    @property
    def terminal_share(self) -> float:
        """Fraction of EV coming from the terminal value — high values mean fragile valuations."""
        return self.pv_terminal / self.enterprise_value if self.enterprise_value else math.nan


def dcf(
    fcfs: Sequence[float],
    discount_rate: float,
    *,
    terminal_growth: float | None = None,
    exit_multiple: float | None = None,
    terminal_metric: float | None = None,
    mid_year: bool = False,
) -> DCFResult:
    """Unlevered DCF on explicit free cash flows for years 1..N.

    Terminal value: Gordon growth on FCF_N (``terminal_growth``) or ``exit_multiple * terminal_metric``
    — exactly one. The terminal value is discounted from the end of year N in both conventions.
    """
    if not fcfs:
        raise ValueError("need at least one explicit-period FCF")
    if discount_rate <= -1:
        raise ValueError("discount_rate must exceed -100%")
    gordon = terminal_growth is not None
    multiple = exit_multiple is not None or terminal_metric is not None
    if gordon == multiple:
        raise ValueError(
            "specify exactly one terminal method: terminal_growth OR exit_multiple+terminal_metric"
        )
    if multiple and (exit_multiple is None or terminal_metric is None):
        raise ValueError("exit multiple method needs both exit_multiple and terminal_metric")

    n = len(fcfs)
    shift = 0.5 if mid_year else 0.0
    pv_explicit = sum(cf / (1 + discount_rate) ** (t - shift) for t, cf in enumerate(fcfs, start=1))

    if gordon:
        assert terminal_growth is not None
        if terminal_growth >= discount_rate:
            raise ValueError("terminal_growth must be below discount_rate")
        tv = fcfs[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
    else:
        assert exit_multiple is not None and terminal_metric is not None
        tv = exit_multiple * terminal_metric
    pv_tv = tv / (1 + discount_rate) ** n
    return DCFResult(pv_explicit, tv, pv_tv, pv_explicit + pv_tv)


def equity_value_per_share(enterprise_value_: float, net_debt: float, diluted_shares: float) -> float:
    if diluted_shares <= 0:
        raise ValueError("diluted_shares must be positive")
    return (enterprise_value_ - net_debt) / diluted_shares


# ---------------------------------------------------------------------------------------------
# Reverse DCF
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReverseDCFResult:
    implied_growth: float        # NaN when no growth rate in the search range reproduces the price
    converged: bool
    search_range: tuple[float, float]
    price: float


def _fcf_path_growth(base_fcf: float, growth: float, years: int) -> list[float]:
    return [base_fcf * (1 + growth) ** t for t in range(1, years + 1)]


def price_from_fcf_growth(
    *, growth: float, base_fcf: float, years: int, discount_rate: float, terminal_growth: float,
    net_debt: float, diluted_shares: float,
) -> float:
    res = dcf(_fcf_path_growth(base_fcf, growth, years), discount_rate, terminal_growth=terminal_growth)
    return equity_value_per_share(res.enterprise_value, net_debt, diluted_shares)


def _fcf_path_revenue(
    base_revenue: float, growth: float, start_margin: float, target_margin: float, years: int
) -> list[float]:
    out = []
    for t in range(1, years + 1):
        m = start_margin + (target_margin - start_margin) * t / years  # linear ramp to target by year N
        out.append(base_revenue * (1 + growth) ** t * m)
    return out


def price_from_revenue_growth(
    *, revenue_growth: float, base_revenue: float, start_fcf_margin: float, target_fcf_margin: float,
    years: int, discount_rate: float, terminal_growth: float, net_debt: float, diluted_shares: float,
) -> float:
    fcfs = _fcf_path_revenue(base_revenue, revenue_growth, start_fcf_margin, target_fcf_margin, years)
    res = dcf(fcfs, discount_rate, terminal_growth=terminal_growth)
    return equity_value_per_share(res.enterprise_value, net_debt, diluted_shares)


def _solve(price: float, fn: Callable[[float], float], lo: float, hi: float) -> ReverseDCFResult:
    f_lo, f_hi = fn(lo) - price, fn(hi) - price
    if not (math.isfinite(f_lo) and math.isfinite(f_hi)) or f_lo * f_hi > 0:
        return ReverseDCFResult(math.nan, False, (lo, hi), price)
    g = brentq(lambda x: fn(x) - price, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200)
    return ReverseDCFResult(float(g), True, (lo, hi), price)


def reverse_dcf_fcf_growth(
    *, price: float, base_fcf: float, years: int, discount_rate: float, terminal_growth: float,
    net_debt: float, diluted_shares: float, search_range: tuple[float, float] = (-0.5, 1.0),
) -> ReverseDCFResult:
    """Constant annual FCF growth over ``years`` implied by ``price``."""
    return _solve(
        price,
        lambda g: price_from_fcf_growth(
            growth=g, base_fcf=base_fcf, years=years, discount_rate=discount_rate,
            terminal_growth=terminal_growth, net_debt=net_debt, diluted_shares=diluted_shares,
        ),
        *search_range,
    )


def reverse_dcf_revenue_growth(
    *, price: float, base_revenue: float, start_fcf_margin: float, target_fcf_margin: float, years: int,
    discount_rate: float, terminal_growth: float, net_debt: float, diluted_shares: float,
    search_range: tuple[float, float] = (-0.5, 1.0),
) -> ReverseDCFResult:
    """Revenue CAGR implied by ``price`` given an FCF-margin path ramping linearly to a target."""
    return _solve(
        price,
        lambda g: price_from_revenue_growth(
            revenue_growth=g, base_revenue=base_revenue, start_fcf_margin=start_fcf_margin,
            target_fcf_margin=target_fcf_margin, years=years, discount_rate=discount_rate,
            terminal_growth=terminal_growth, net_debt=net_debt, diluted_shares=diluted_shares,
        ),
        *search_range,
    )
