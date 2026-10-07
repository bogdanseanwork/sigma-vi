"""Factor definitions and the provisional SIGMA score (0-100).

EXPERIMENTAL: the categories and starting weights come from the selection mandate's Phase 2 table
and are hypotheses, not validated results. Backtests (milestone E) decide what survives.

Every factor is signed so that higher is better. Each is winsorised at 1%/99% across the universe,
converted to a z-score within its sector (whole universe for sectors too small to compare), and
clipped at +/-3. A category score is the mean of its available factor z-scores; the composite is the
weighted mean of available categories. Companies with too little coverage get no score rather than
a guess. Categories with no free point-in-time data (earnings revisions; catalysts, which need the
research agents) are listed as not scored, and the remaining weights are rescaled to sum to 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPEC_WEIGHTS: dict[str, float] = {
    "Growth": 0.15, "Earnings Revisions": 0.10, "Profitability": 0.10, "Quality / Moat": 0.10,
    "Valuation": 0.15, "Free Cash Flow": 0.10, "Catalysts": 0.10, "Momentum": 0.05,
    "Management": 0.05, "Financial Health": 0.05, "Risk": 0.05,
}
NOT_SCORED: dict[str, str] = {
    "Earnings Revisions": "no free source keeps point-in-time analyst estimate history",
    "Catalysts": "needs the research agents (milestone H)",
}
CATEGORIES: dict[str, list[str]] = {
    "Growth": ["rev_growth", "rev_cagr_3y", "op_income_growth", "net_income_growth"],
    "Profitability": ["gross_margin", "op_margin", "roe", "roa"],
    "Quality / Moat": ["roic", "accruals", "cash_conversion", "margin_trend"],
    "Valuation": ["earnings_yield", "fcf_yield", "ebitda_to_ev", "sales_to_ev"],
    "Free Cash Flow": ["fcf_margin", "fcf_growth", "sbc_intensity"],
    "Momentum": ["mom_12_1", "mom_6_1"],
    "Management": ["share_count_change"],  # capital-allocation proxy: dilution vs buybacks
    "Financial Health": ["net_debt_to_ebitda", "interest_coverage", "current_ratio"],
    "Risk": ["low_volatility", "low_drawdown"],
}
# Lenders' and insurers' operating cash flow mixes in loan and policy flows, so free-cash-flow and
# accrual measures say little about them; they are scored on the other factors instead.
NOT_FOR_FINANCIALS = ("fcf_yield", "fcf_margin", "fcf_growth", "cash_conversion", "accruals")
# Ratios outside these ranges are filing/tagging artefacts or pass-through businesses (metal traders,
# brokers), not information. They are treated as missing, which the coverage threshold then handles.
PLAUSIBLE: dict[str, tuple[float, float]] = {
    "earnings_yield": (-1.0, 0.6), "fcf_yield": (-1.0, 0.6), "ebitda_to_ev": (-1.0, 1.0),
    "sales_to_ev": (0.0, 20.0), "gross_margin": (-1.0, 1.0), "op_margin": (-5.0, 1.0),
    "roe": (-5.0, 5.0), "roa": (-2.0, 2.0), "roic": (-5.0, 5.0), "fcf_margin": (-5.0, 1.0),
    "rev_growth": (-3.0, 4.0), "cash_conversion": (-20.0, 20.0),
}
MIN_EV_TO_MCAP = 0.1      # EV below 10% of market cap usually means debt was not tagged: skip EV ratios
MIN_COVERAGE = 0.6        # share of available weight a company needs to be scored
WORST_LEVERAGE = -10.0    # net debt with negative EBITDA: worse than any finite ratio
MAX_COVERAGE_RATIO = 50.0
TAX_RATE = 0.21


def effective_weights() -> dict[str, float]:
    live = {c: w for c, w in SPEC_WEIGHTS.items() if c not in NOT_SCORED}
    total = sum(live.values())
    return {c: w / total for c, w in live.items()}


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").astype(float)


def _div(a: pd.Series, b: pd.Series, positive_denominator: bool = True) -> pd.Series:
    ok = b > 0 if positive_denominator else b != 0
    return (a / b).where(ok)


def _rel_change(x: pd.Series, x0: pd.Series) -> pd.Series:
    return ((x - x0) / x0.abs()).where(x0 != 0)


def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    n = {c: _num(df, c) for c in df.columns}
    g = lambda c: n.get(c, pd.Series(np.nan, index=df.index))  # noqa: E731
    z0 = lambda c: g(c).fillna(0.0)  # noqa: E731 — balance items a company may simply not report
    rev, rev1, rev3 = g("revenue"), g("revenue_1y"), g("revenue_3y")
    oi, oi1, ni, ni1 = g("operating_income"), g("operating_income_1y"), g("net_income"), g("net_income_1y")
    fcf, fcf1 = g("cfo") - z0("capex"), g("cfo_1y") - g("capex_1y").fillna(0.0)
    mcap = g("price") * g("shares")
    debt = g("debt").fillna(z0("debt_current") + z0("debt_noncurrent")) if "debt" in df else \
        z0("debt_current") + z0("debt_noncurrent")
    debt = debt.fillna(0.0)
    net_debt = debt - z0("cash") - z0("short_term_investments")
    ev = (mcap + net_debt).where(mcap + net_debt >= MIN_EV_TO_MCAP * mcap)
    ebitda = oi + z0("dna")
    gross = g("gross_profit").fillna(rev - g("cost_of_revenue"))
    avg_eq = pd.concat([g("equity"), g("equity_1y").fillna(g("equity"))], axis=1).mean(axis=1)
    assets = g("total_assets")
    avg_assets = pd.concat([assets, g("total_assets_1y").fillna(assets)], axis=1).mean(axis=1)
    invested = g("equity") + net_debt

    leverage = (-net_debt / ebitda).where(ebitda > 0)
    leverage = leverage.mask((ebitda <= 0) & (net_debt > 0), WORST_LEVERAGE)
    interest = g("interest_expense")
    coverage = _div(oi, interest).clip(upper=MAX_COVERAGE_RATIO)
    coverage = coverage.mask(((interest.isna()) | (interest == 0)) & (oi > 0), MAX_COVERAGE_RATIO)

    out = pd.DataFrame(index=df.index)
    out["rev_growth"] = np.log(_div(rev, rev1).where(rev > 0))  # log: a tiny base can't dominate
    out["rev_cagr_3y"] = (_div(rev, rev3) ** (1 / 3) - 1).where(rev > 0)
    out["op_income_growth"] = _rel_change(oi, oi1)
    out["net_income_growth"] = _rel_change(ni, ni1)
    out["gross_margin"] = _div(gross, rev)
    out["op_margin"] = _div(oi, rev)
    out["roe"] = _div(ni, avg_eq)
    out["roa"] = _div(ni, avg_assets)
    out["roic"] = _div(oi * (1 - TAX_RATE), invested)
    out["accruals"] = -_div(ni - g("cfo"), g("total_assets"))
    out["cash_conversion"] = _div(g("cfo"), ni)
    out["margin_trend"] = _div(oi, rev) - _div(oi1, rev1)
    out["earnings_yield"] = _div(ni, mcap)
    out["fcf_yield"] = _div(fcf, mcap)
    out["ebitda_to_ev"] = _div(ebitda, ev)
    out["sales_to_ev"] = _div(rev, ev)
    out["fcf_margin"] = _div(fcf, rev)
    out["fcf_growth"] = _rel_change(fcf, fcf1)
    out["sbc_intensity"] = -_div(g("sbc"), rev)
    out["mom_12_1"] = g("mom_12_1")
    out["mom_6_1"] = g("mom_6_1")
    out["share_count_change"] = -(_div(g("shares"), g("shares_1y")) - 1)
    out["net_debt_to_ebitda"] = leverage
    out["interest_coverage"] = coverage
    out["current_ratio"] = _div(g("current_assets"), g("current_liabilities"))
    out["low_volatility"] = -g("vol_1y")
    out["low_drawdown"] = -g("max_dd_1y")
    out["market_cap"] = mcap
    for col, (lo, hi) in PLAUSIBLE.items():
        out[col] = out[col].where(out[col].between(lo, hi))
    if "sector" in df:
        out.loc[df["sector"] == "Financials", list(NOT_FOR_FINANCIALS)] = np.nan
    return out.replace([np.inf, -np.inf], np.nan)


def winsorize(s: pd.Series, q: float = 0.01) -> pd.Series:
    if s.notna().sum() < 3:
        return s
    return s.clip(s.quantile(q), s.quantile(1 - q))


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0


def sector_z(df: pd.DataFrame, sector: pd.Series, min_group: int = 10) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        x = df[col].astype(float)
        whole = _z(x)
        z = whole.copy()
        for _, idx in x.groupby(sector).groups.items():
            part = x.loc[idx]
            if part.notna().sum() >= min_group:
                z.loc[idx] = _z(part)
        out[col] = z
    return out


def score(df: pd.DataFrame, sector_col: str = "sector") -> pd.DataFrame:
    """Factors, category scores, coverage and the 0-100 ``sigma_score`` for each row of ``df``."""
    factors = compute_factors(df)
    cols = [f for fs in CATEGORIES.values() for f in fs]
    wins = factors[cols].apply(winsorize)
    z = sector_z(wins, df[sector_col].fillna("Unknown")).clip(-3, 3)
    weights = effective_weights()
    out = factors.copy()
    num = pd.Series(0.0, index=df.index)
    cover = pd.Series(0.0, index=df.index)
    for cat, fs in CATEGORIES.items():
        c = z[fs].mean(axis=1, skipna=True)
        out[f"cat_{cat}"] = c
        has = c.notna()
        num += (c.fillna(0.0) * weights[cat])
        cover += has * weights[cat]
    out["coverage"] = cover
    composite = (num / cover).where(cover >= MIN_COVERAGE)
    out["composite"] = composite
    ranks = composite.rank(method="average")
    n = composite.notna().sum()
    out["sigma_score"] = ((ranks - 1) / (n - 1) * 100) if n > 1 else ranks * 0 + 50
    return out
