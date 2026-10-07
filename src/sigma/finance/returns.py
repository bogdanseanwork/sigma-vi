"""Return arithmetic. Simple vs log returns are explicit in every name."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

ArrayLike = Sequence[float] | NDArray[np.float64]


def as_array(x: ArrayLike, name: str = "values") -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf; clean or align data before computing")
    return arr


def simple_returns(prices: ArrayLike) -> NDArray[np.float64]:
    p = as_array(prices, "prices")
    if np.any(p <= 0):
        raise ValueError("prices must be positive")
    return p[1:] / p[:-1] - 1.0


def total_returns(prices: ArrayLike, dividends_by_index: Mapping[int, float]) -> NDArray[np.float64]:
    """Daily total returns with cash dividends reinvested at the ex-date close.

    ``prices`` are split-adjusted, NOT dividend-adjusted closes. ``dividends_by_index`` maps the
    position of each ex-date in ``prices`` to the per-share cash amount. Price-only returns understate
    performance by roughly the dividend yield (≈1.4 pp for SPY in 2025), so backtests use this.
    """
    p = as_array(prices, "prices")
    if np.any(p <= 0):
        raise ValueError("prices must be positive")
    d = np.zeros_like(p)
    for i, amount in dividends_by_index.items():
        if not 1 <= i < p.size:
            raise ValueError(f"dividend index {i} must fall on a bar after the first price")
        d[i] += amount
    return (p[1:] + d[1:]) / p[:-1] - 1.0


def log_returns(prices: ArrayLike) -> NDArray[np.float64]:
    p = as_array(prices, "prices")
    if np.any(p <= 0):
        raise ValueError("prices must be positive")
    return np.diff(np.log(p))


def cumulative_return(simple: ArrayLike) -> float:
    r = as_array(simple, "returns")
    return float(np.prod(1.0 + r) - 1.0)


def cagr(start_value: float, end_value: float, years: float) -> float:
    if start_value <= 0:
        raise ValueError("start_value must be positive")
    if years <= 0:
        raise ValueError("years must be positive")
    if end_value <= 0:
        return -1.0
    return (end_value / start_value) ** (1.0 / years) - 1.0


def cagr_from_returns(simple: ArrayLike, periods_per_year: int = 252) -> float:
    r = as_array(simple, "returns")
    if r.size == 0:
        raise ValueError("need at least one return")
    growth = float(np.prod(1.0 + r))
    return cagr(1.0, growth, r.size / periods_per_year)


def annualized_volatility(simple: ArrayLike, periods_per_year: int = 252) -> float:
    r = as_array(simple, "returns")
    if r.size < 2:
        return math.nan
    return float(r.std(ddof=1) * math.sqrt(periods_per_year))


def per_period_rate(annual_rate: float, periods_per_year: int = 252) -> float:
    """Geometric de-annualisation, e.g. a 4% annual risk-free rate to a daily rate."""
    return (1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0
