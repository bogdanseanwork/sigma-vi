"""Performance and risk metrics (spec §21, §35). Definitions are in docs/VALIDATION.md §7."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from sigma.finance.returns import (
    ArrayLike,
    annualized_volatility,
    as_array,
    cagr_from_returns,
    per_period_rate,
)


def _excess(returns: ArrayLike, risk_free_annual: float, periods_per_year: int) -> np.ndarray:
    r = as_array(returns, "returns")
    return r - per_period_rate(risk_free_annual, periods_per_year) if risk_free_annual else r


def sharpe(returns: ArrayLike, risk_free_annual: float = 0.0, periods_per_year: int = 252) -> float:
    ex = _excess(returns, risk_free_annual, periods_per_year)
    if ex.size < 2:
        return math.nan
    sd = ex.std(ddof=1)
    if sd == 0 or not np.isfinite(sd) or sd < 1e-15:
        return math.nan
    return float(ex.mean() / sd * math.sqrt(periods_per_year))


def downside_deviation(returns: ArrayLike, mar_per_period: float = 0.0) -> float:
    r = as_array(returns, "returns")
    shortfall = np.minimum(r - mar_per_period, 0.0)
    return float(math.sqrt(np.mean(shortfall**2)))


def sortino(returns: ArrayLike, risk_free_annual: float = 0.0, periods_per_year: int = 252) -> float:
    ex = _excess(returns, risk_free_annual, periods_per_year)
    dd = downside_deviation(ex)
    if dd == 0:
        return math.nan
    return float(ex.mean() / dd * math.sqrt(periods_per_year))


@dataclass(frozen=True)
class DrawdownStats:
    max_drawdown: float          # negative fraction, e.g. -0.25
    peak_index: int
    trough_index: int
    recovery_index: int | None   # first index at/above the prior peak, None if never recovered

    @property
    def recovery_periods(self) -> int | None:
        return None if self.recovery_index is None else self.recovery_index - self.trough_index


def drawdown_stats(nav: ArrayLike) -> DrawdownStats:
    v = as_array(nav, "nav")
    if v.size == 0 or np.any(v <= 0):
        raise ValueError("nav must be non-empty and positive")
    running_peak = np.maximum.accumulate(v)
    dd = v / running_peak - 1.0
    trough = int(np.argmin(dd))
    if dd[trough] == 0:
        return DrawdownStats(0.0, trough, trough, trough)
    peak = int(np.argmax(v[: trough + 1]))
    after = np.nonzero(v[trough:] >= v[peak])[0]
    recovery = int(trough + after[0]) if after.size else None
    return DrawdownStats(float(dd[trough]), peak, trough, recovery)


def nav_from_returns(returns: ArrayLike, start: float = 1.0) -> np.ndarray:
    r = as_array(returns, "returns")
    return np.concatenate([[start], start * np.cumprod(1.0 + r)])


def max_drawdown_from_returns(returns: ArrayLike) -> float:
    return drawdown_stats(nav_from_returns(returns)).max_drawdown


def calmar(returns: ArrayLike, periods_per_year: int = 252) -> float:
    mdd = max_drawdown_from_returns(returns)
    if mdd == 0:
        return math.nan
    return cagr_from_returns(returns, periods_per_year) / abs(mdd)


def _aligned(asset: ArrayLike, bench: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    a, b = as_array(asset, "asset"), as_array(bench, "benchmark")
    if a.shape != b.shape:
        raise ValueError("asset and benchmark returns must be aligned and equal length")
    return a, b


def beta(asset: ArrayLike, bench: ArrayLike) -> float:
    a, b = _aligned(asset, bench)
    var = b.var(ddof=1)
    return math.nan if var == 0 else float(np.cov(a, b, ddof=1)[0, 1] / var)


def alpha_annualized(asset: ArrayLike, bench: ArrayLike, periods_per_year: int = 252) -> float:
    """OLS intercept of asset on benchmark, scaled by periods (Jensen's alpha with rf already netted)."""
    a, b = _aligned(asset, bench)
    bt = beta(a, b)
    return float((a.mean() - bt * b.mean()) * periods_per_year)


def correlation(asset: ArrayLike, bench: ArrayLike) -> float:
    a, b = _aligned(asset, bench)
    return float(np.corrcoef(a, b)[0, 1])


def downside_capture(asset: ArrayLike, bench: ArrayLike) -> float:
    """Mean asset return over mean benchmark return on benchmark-down periods (arithmetic)."""
    a, b = _aligned(asset, bench)
    mask = b < 0
    return math.nan if not mask.any() else float(a[mask].mean() / b[mask].mean())


def upside_capture(asset: ArrayLike, bench: ArrayLike) -> float:
    a, b = _aligned(asset, bench)
    mask = b > 0
    return math.nan if not mask.any() else float(a[mask].mean() / b[mask].mean())


def _tail(returns: ArrayLike, confidence: float) -> np.ndarray:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    r = np.sort(as_array(returns, "returns"))
    k = max(1, math.ceil(round(r.size * (1 - confidence), 10)))
    return r[:k]


def value_at_risk(returns: ArrayLike, confidence: float = 0.95) -> float:
    """Historical VaR as a (negative) return: the boundary of the worst (1-c) tail."""
    return float(_tail(returns, confidence)[-1])


def expected_shortfall(returns: ArrayLike, confidence: float = 0.975) -> float:
    """Historical ES / CVaR: mean of the worst (1-c) fraction of returns."""
    return float(_tail(returns, confidence).mean())


def hit_rate(returns: ArrayLike) -> float:
    r = as_array(returns, "returns")
    return math.nan if r.size == 0 else float(np.mean(r > 0))


def turnover(old_weights: Mapping[str, float], new_weights: Mapping[str, float]) -> float:
    """One-way turnover: half the sum of absolute weight changes."""
    keys = set(old_weights) | set(new_weights)
    return 0.5 * sum(abs(new_weights.get(k, 0.0) - old_weights.get(k, 0.0)) for k in keys)


def brier_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    p, o = np.asarray(probabilities, float), np.asarray(outcomes, float)
    if p.shape != o.shape or p.size == 0:
        raise ValueError("probabilities and outcomes must be equal-length and non-empty")
    if np.any((p < 0) | (p > 1)) or not np.all(np.isin(o, (0.0, 1.0))):
        raise ValueError("probabilities must be in [0,1] and outcomes in {0,1}")
    return float(np.mean((p - o) ** 2))


def performance_summary(
    returns: ArrayLike,
    benchmark: ArrayLike | None = None,
    risk_free_annual: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    r = as_array(returns, "returns")
    out = {
        "cagr": cagr_from_returns(r, periods_per_year),
        "volatility": annualized_volatility(r, periods_per_year),
        "sharpe": sharpe(r, risk_free_annual, periods_per_year),
        "sortino": sortino(r, risk_free_annual, periods_per_year),
        "calmar": calmar(r, periods_per_year),
        "max_drawdown": max_drawdown_from_returns(r),
        "hit_rate": hit_rate(r),
        "var_95": value_at_risk(r, 0.95),
        "es_97_5": expected_shortfall(r, 0.975),
    }
    if benchmark is not None:
        b = as_array(benchmark, "benchmark")
        out |= {
            "beta": beta(r, b),
            "alpha": alpha_annualized(r, b, periods_per_year),
            "correlation": correlation(r, b),
            "downside_capture": downside_capture(r, b),
            "upside_capture": upside_capture(r, b),
            "excess_cagr": out["cagr"] - cagr_from_returns(b, periods_per_year),
        }
    return out
