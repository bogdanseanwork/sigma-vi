"""Does the score predict returns? Information coefficients, quintiles, performance statistics."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


def information_coefficients(panel: pd.DataFrame, columns: Sequence[str], min_names: int = 20
                             ) -> pd.DataFrame:
    """Rank correlation between each column and the forward return, per rebalance date."""
    rows = {}
    for d, g in panel.groupby("date", sort=True):
        g = g.dropna(subset=["fwd"])
        if len(g) < min_names:
            continue
        r = g["fwd"].rank()
        rows[d] = {c: g[c].rank().corr(r) if g[c].notna().sum() >= min_names else np.nan for c in columns}
    return pd.DataFrame.from_dict(rows, orient="index")


def ic_summary(ic: pd.Series) -> dict[str, float]:
    x = ic.dropna()
    n = len(x)
    sd = float(x.std(ddof=1)) if n > 1 else math.nan
    return {"n": n, "mean": float(x.mean()) if n else math.nan, "std": sd,
            "t": float(x.mean() / (sd / math.sqrt(n))) if n > 1 and sd > 0 else math.nan,
            "hit": float((x > 0).mean()) if n else math.nan}


def quintile_returns(panel: pd.DataFrame, score: str, buckets: int = 5) -> pd.DataFrame:
    """Mean forward return by score quintile (1 = lowest) for each date."""
    rows = {}
    for d, g in panel.groupby("date", sort=True):
        g = g.dropna(subset=[score, "fwd"])
        if len(g) < buckets * 5:
            continue
        q = pd.qcut(g[score].rank(method="first"), buckets, labels=range(1, buckets + 1))
        rows[d] = g.groupby(q, observed=True)["fwd"].mean()
    return pd.DataFrame.from_dict(rows, orient="index")


def performance(returns: pd.Series, periods_per_year: int = 4, risk_free_annual: float = 0.0
                ) -> dict[str, float]:
    r = returns.dropna().to_numpy(dtype=float)
    if len(r) == 0:
        return {"periods": 0}
    nav = np.cumprod(1 + r)
    total = float(nav[-1] - 1)
    years = len(r) / periods_per_year
    peak = np.maximum.accumulate(np.concatenate([[1.0], nav]))
    dd = np.concatenate([[1.0], nav]) / peak - 1
    ex = r - risk_free_annual / periods_per_year
    sd = float(np.std(r, ddof=1)) if len(r) > 1 else math.nan
    down = np.minimum(ex, 0)
    dsd = float(np.sqrt(np.mean(down**2)))
    return {
        "periods": len(r), "total_return": total, "cagr": float((1 + total) ** (1 / years) - 1),
        "volatility": sd * math.sqrt(periods_per_year) if sd == sd else math.nan,
        "sharpe": float(ex.mean() / sd * math.sqrt(periods_per_year)) if sd and sd > 0 else math.nan,
        "sortino": float(ex.mean() / dsd * math.sqrt(periods_per_year)) if dsd > 0 else math.nan,
        "max_drawdown": float(dd.min()), "hit_rate": float((r > 0).mean()),
    }
