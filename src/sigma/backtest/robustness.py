"""Is the result real, or luck and overfitting? (milestone E2)

Every function takes a panel with one row per (date, symbol): ``sigma_score``, ``cat_*`` category
scores, individual factor columns, ``sector`` and ``fwd`` (return to the next rebalance). Nothing here
touches the holdout; the caller decides which dates go in.

* factor-level rank IC, with a multiple-testing correction because many factors are tried;
* category ablation: drop one category, re-rank, see what changes;
* matched random portfolios: how often does a blind pick with the same sector mix do as well?
* weight perturbation: do small changes to the mandate's weights swap the picks?
* regimes: does the score work in down markets as well as up markets?
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from sigma.backtest import analysis, engine
from sigma.factors.scoring import MIN_COVERAGE

PERIODS_PER_YEAR = 4


# ---------------------------------------------------------------------------------------------
# Multiple testing
# ---------------------------------------------------------------------------------------------
def p_from_t(t: float, n: int) -> float:
    """Two-sided p-value of a t statistic from ``n`` observations."""
    if t != t or n < 2:
        return math.nan
    return float(2 * stats.t.sf(abs(t), df=n - 1))


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls the chance of ANY false positive)."""
    p = np.array(pvals, dtype=float)
    ok = ~np.isnan(p)
    out = np.full(len(p), np.nan)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.maximum.accumulate((m - np.arange(m)) * p[ok][idx])
    vals = np.empty(m)
    vals[idx] = np.minimum(adj, 1.0)
    out[ok] = vals
    return list(out)


def benjamini_hochberg(pvals: Sequence[float]) -> list[float]:
    """BH adjusted p-values (controls the share of false discoveries among those called real)."""
    p = np.array(pvals, dtype=float)
    ok = ~np.isnan(p)
    out = np.full(len(p), np.nan)
    m = ok.sum()
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    vals = np.empty(m)
    vals[order] = np.minimum(adj, 1.0)
    out[ok] = vals
    return list(out)


def correct(tests: Mapping[str, tuple[float, int]]) -> pd.DataFrame:
    """Ledger of every IC test run: raw t and p, and p after Holm and Benjamini-Hochberg."""
    names = list(tests)
    p = [p_from_t(*tests[k]) for k in names]
    return pd.DataFrame({"t": [tests[k][0] for k in names], "n": [tests[k][1] for k in names],
                         "p": p, "p_holm": holm(p), "p_bh": benjamini_hochberg(p)}, index=names)


# ---------------------------------------------------------------------------------------------
# Factor-level IC
# ---------------------------------------------------------------------------------------------
def ic_by_period(panel: pd.DataFrame, columns: Sequence[str], period_of: Callable[[date], str],
                 fwd: str = "fwd", step: int = 1) -> dict[str, pd.DataFrame]:
    """Mean rank IC, t, hit rate and n per column, for each period. ``step`` > 1 keeps every
    ``step``-th date, so that overlapping long-horizon returns are not counted as independent."""
    work = panel.rename(columns={fwd: "fwd"}) if fwd != "fwd" else panel
    dates = sorted(work["date"].unique())[::step]
    work = work[work["date"].isin(dates)]
    ic = analysis.information_coefficients(work, columns)
    out = {}
    for per in ("train", "validation", "holdout"):
        sel = [d for d in ic.index if period_of(d) == per]
        if sel:
            out[per] = pd.DataFrame({c: analysis.ic_summary(ic.loc[sel, c]) for c in columns}).T
    return out


# ---------------------------------------------------------------------------------------------
# Composite from categories (so categories can be dropped or re-weighted)
# ---------------------------------------------------------------------------------------------
def composite_from(panel: pd.DataFrame, weights: Mapping[str, float], min_coverage: float = MIN_COVERAGE
                   ) -> pd.Series:
    """Weighted mean of the category scores a company has, rescaled over those available."""
    num = pd.Series(0.0, index=panel.index)
    cover = pd.Series(0.0, index=panel.index)
    total = sum(weights.values())
    for cat, w in weights.items():
        c = panel[f"cat_{cat}"]
        num += c.fillna(0.0) * w / total
        cover += c.notna() * w / total
    return (num / cover).where(cover >= min_coverage)


def _top_n_gross(panel: pd.DataFrame, score: str, n: int) -> pd.Series:
    return engine.run_portfolio(panel, n=n, cost_bps=0.0, score=score)["gross"]


def _cagr(r: np.ndarray | pd.Series, ppy: int = PERIODS_PER_YEAR) -> float:
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    if not len(r):
        return math.nan
    return float(np.prod(1 + r) ** (ppy / len(r)) - 1)


def _in_period(panel: pd.DataFrame, period_of: Callable[[date], str], per: str) -> pd.DataFrame:
    keep = panel["date"].map(lambda d: period_of(d) == per)
    return panel[keep]


# ---------------------------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------------------------
def ablation(panel: pd.DataFrame, weights: Mapping[str, float], period_of: Callable[[date], str],
             n: int = 40) -> pd.DataFrame:
    """Re-rank without each category in turn. Rows: (period, dropped category); columns: mean rank IC
    and the top-n gross CAGR, plus their change versus keeping everything."""
    rows = []
    variants: dict[str, dict[str, float]] = {"(none dropped)": dict(weights)}
    for cat in weights:
        variants[cat] = {c: w for c, w in weights.items() if c != cat}
    for per in ("train", "validation"):
        sub = _in_period(panel, period_of, per)
        if sub.empty:
            continue
        base: dict[str, float] = {}
        for name, w in variants.items():
            s = sub.assign(_score=composite_from(sub, w))
            ic = analysis.information_coefficients(s, ["_score"])["_score"]
            cagr = _cagr(_top_n_gross(s, "_score", n))
            if name == "(none dropped)":
                base = {"ic": float(ic.mean()), "cagr": cagr}
            rows.append({"period": per, "dropped": name, "ic": float(ic.mean()), "cagr": cagr,
                         "d_ic": float(ic.mean()) - base["ic"], "d_cagr": cagr - base["cagr"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Matched random portfolios
# ---------------------------------------------------------------------------------------------
def random_matched(panel: pd.DataFrame, n: int = 40, trials: int = 5000, seed: int = 7,
                   score: str = "sigma_score") -> dict[str, Any]:
    """Compare the top-n portfolio with ``trials`` blind portfolios holding the same number of names
    from each sector at every date. Gross of costs on both sides. ``percentile`` is the share of
    random portfolios the real one beat; 95+ is what a real edge would look like."""
    rng = np.random.default_rng(seed)
    dates = sorted(panel["date"].unique())
    real, rand = [], np.zeros((trials, len(dates)))
    for j, d in enumerate(dates):
        g = panel[panel["date"] == d].dropna(subset=["fwd"])
        picks = g.dropna(subset=[score]).nlargest(n, score)
        real.append(float(picks["fwd"].mean()))
        tot = np.zeros(trials)
        cnt = 0
        for sector, k in picks.groupby("sector").size().items():
            pool = g.loc[g["sector"] == sector, "fwd"].to_numpy()
            k = min(int(k), len(pool))
            if k == 0:
                continue
            idx = np.argsort(rng.random((trials, len(pool))), axis=1)[:, :k]
            tot += pool[idx].sum(axis=1)
            cnt += k
        rand[:, j] = tot / max(cnt, 1)
    real_cagr = _cagr(real)
    cagrs = np.prod(1 + rand, axis=1) ** (PERIODS_PER_YEAR / len(dates)) - 1
    return {"sigma_cagr": real_cagr, "random_mean": float(cagrs.mean()),
            "random_p05": float(np.percentile(cagrs, 5)), "random_p95": float(np.percentile(cagrs, 95)),
            "percentile": float((cagrs < real_cagr).mean() * 100), "trials": trials, "dates": len(dates)}


# ---------------------------------------------------------------------------------------------
# Weight perturbation
# ---------------------------------------------------------------------------------------------
def perturbation(panel: pd.DataFrame, weights: Mapping[str, float], pct: float, n: int = 40,
                 trials: int = 200, seed: int = 11) -> dict[str, float]:
    """Multiply each category weight by a random factor within +/-``pct`` and rebuild the top n.
    Reports how much of the base portfolio survives and the spread of CAGR."""
    rng = np.random.default_rng(seed)
    dates = sorted(panel["date"].unique())
    groups = {d: panel[panel["date"] == d] for d in dates}
    base_sets = {}
    base_comp = composite_from(panel, weights)
    for d, g in groups.items():
        c = base_comp.loc[g.index].dropna()
        base_sets[d] = set(g.loc[c.nlargest(n).index, "symbol"])
    base_cagr = _cagr(_top_n_gross(panel.assign(_score=base_comp), "_score", n))
    overlaps, cagrs = [], []
    for _ in range(trials):
        w = {c: v * rng.uniform(1 - pct, 1 + pct) for c, v in weights.items()}
        comp = composite_from(panel, w)
        cagrs.append(_cagr(_top_n_gross(panel.assign(_score=comp), "_score", n)))
        ov = []
        for d, g in groups.items():
            c = comp.loc[g.index].dropna()
            picks = set(g.loc[c.nlargest(n).index, "symbol"])
            ov.append(len(picks & base_sets[d]) / max(len(base_sets[d]), 1))
        overlaps.append(float(np.mean(ov)))
    return {"pct": pct, "mean_overlap": float(np.mean(overlaps)), "min_overlap": float(np.min(overlaps)),
            "base_cagr": base_cagr, "cagr_p05": float(np.percentile(cagrs, 5)),
            "cagr_p95": float(np.percentile(cagrs, 95)), "trials": trials}


# ---------------------------------------------------------------------------------------------
# Regimes (labels use the realised market return: descriptive, not something a trader could know)
# ---------------------------------------------------------------------------------------------
def regimes(panel: pd.DataFrame, market: pd.Series, n: int = 40) -> pd.DataFrame:
    """Top-n gross return, universe return and mean IC in quarters when the market rose vs fell."""
    port = engine.run_portfolio(panel, n=n, cost_bps=0.0)["gross"]
    uni = engine.universe_benchmark(panel)
    ic = analysis.information_coefficients(panel, ["sigma_score"])["sigma_score"]
    rows = []
    for label, mask in (("market up", market > 0), ("market down", market <= 0)):
        dates = [d for d in market.index[mask] if d in port.index]
        if not dates:
            continue
        rows.append({"regime": label, "quarters": len(dates), "top_n": float(port.loc[dates].mean()),
                     "universe": float(uni.reindex(dates).mean()),
                     "excess": float((port.loc[dates] - uni.reindex(dates)).mean()),
                     "ic": float(ic.reindex(dates).mean())})
    return pd.DataFrame(rows)
