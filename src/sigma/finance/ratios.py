"""Financial-statement ratios and quality flags (spec §15)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise


def free_cash_flow(cfo: float, capex: float) -> float:
    """FCF = CFO - CapEx. CapEx is accepted with either sign (filings report it as an outflow)."""
    return cfo - abs(capex)


def margin(numerator: float, revenue: float) -> float:
    return math.nan if revenue == 0 else numerator / revenue


def fcf_conversion(fcf: float, net_income: float) -> float:
    return math.nan if net_income <= 0 else fcf / net_income


def roic(ebit: float, tax_rate: float, invested_capital: float) -> float:
    """NOPAT / invested capital, where invested capital = debt + equity - excess cash (caller's choice)."""
    if invested_capital <= 0:
        return math.nan
    return ebit * (1.0 - tax_rate) / invested_capital


def yoy_growth(values: Sequence[float]) -> list[float]:
    """Period-over-period growth. A non-positive base yields NaN rather than a misleading sign."""
    out: list[float] = []
    for prev, cur in pairwise(values):
        out.append(math.nan if prev <= 0 else cur / prev - 1.0)
    return out


def divergence_flag(item_growth: float, sales_growth: float, threshold: float = 0.10) -> bool:
    """True when inventory/receivables grow materially faster than sales (spec §15 balance-sheet flags)."""
    return (item_growth - sales_growth) > threshold


def net_debt(debt: float, cash: float) -> float:
    return debt - cash


def annual_dilution(shares_start: float, shares_end: float, years: float) -> float:
    if shares_start <= 0 or years <= 0:
        raise ValueError("shares_start and years must be positive")
    return (shares_end / shares_start) ** (1.0 / years) - 1.0
