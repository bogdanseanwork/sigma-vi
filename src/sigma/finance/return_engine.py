"""Return engine (spec §18): stock return ≈ EPS growth + shareholder yield + valuation change.

EPS = revenue * net margin / diluted shares, so annualised price return decomposes exactly in logs:

    ln(1 + price_return) = ln(1 + revenue_growth)
                         + ln(margin_end / margin_start) / T
                         - ln(1 + share_change)
                         + ln(multiple_end / multiple_start) / T

``shareholder_yield`` is added on top. To avoid double counting, pass dividends (and buybacks only if
``share_change`` was NOT already reduced by them).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ReturnDecomposition:
    eps_growth: float            # annualised
    multiple_change: float       # annualised
    price_return: float          # annualised
    total_return: float          # price_return + shareholder_yield
    log_contributions: dict[str, float]


def decompose(
    *,
    revenue_growth: float,
    margin_start: float,
    margin_end: float,
    share_change: float,
    multiple_start: float,
    multiple_end: float,
    shareholder_yield: float,
    years: float,
) -> ReturnDecomposition:
    """All growth inputs are annual rates; margins and multiples are start/end-of-horizon levels."""
    if years <= 0:
        raise ValueError("years must be positive")
    if min(margin_start, margin_end, multiple_start, multiple_end) <= 0:
        raise ValueError("margins and multiples must be positive for a multiplicative decomposition")
    if revenue_growth <= -1 or share_change <= -1:
        raise ValueError("growth rates must exceed -100%")

    contrib = {
        "revenue": math.log1p(revenue_growth),
        "margin": math.log(margin_end / margin_start) / years,
        "share_count": -math.log1p(share_change),
        "multiple": math.log(multiple_end / multiple_start) / years,
    }
    eps_growth = math.exp(contrib["revenue"] + contrib["margin"] + contrib["share_count"]) - 1
    multiple_change = math.exp(contrib["multiple"]) - 1
    price_return = math.exp(sum(contrib.values())) - 1
    return ReturnDecomposition(
        eps_growth=eps_growth,
        multiple_change=multiple_change,
        price_return=price_return,
        total_return=price_return + shareholder_yield,
        log_contributions=contrib,
    )
