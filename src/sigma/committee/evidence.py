"""One compact, factual evidence packet per company. Agents see only this - no invented facts.

Everything comes from SIGMA's own point-in-time data (screen row, filings index, estimate snapshots).
If a number is missing it is left out, never guessed; agents are told to say so when evidence is thin.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import pandas as pd

from sigma.factors.scoring import CATEGORIES

LABELS = {
    "rev_growth": "revenue growth (log, 1y)", "rev_cagr_3y": "revenue CAGR 3y", "op_income_growth": "operating income growth",
    "net_income_growth": "net income growth", "gross_margin": "gross margin", "op_margin": "operating margin",
    "roe": "ROE", "roa": "ROA", "roic": "ROIC", "accruals": "accruals (signed, higher=better)",
    "cash_conversion": "cash conversion", "margin_trend": "margin trend", "earnings_yield": "earnings yield",
    "fcf_yield": "FCF yield", "ebitda_to_ev": "EBITDA / EV", "sales_to_ev": "sales / EV", "fcf_margin": "FCF margin",
    "fcf_growth": "FCF growth", "sbc_intensity": "SBC intensity (signed, higher=better)", "mom_12_1": "12-1 month momentum",
    "mom_6_1": "6-1 month momentum", "share_count_change": "share count change (signed, higher=fewer shares)",
    "net_debt_to_ebitda": "net debt / EBITDA (signed)", "interest_coverage": "interest coverage",
    "current_ratio": "current ratio", "low_volatility": "volatility (signed)", "low_drawdown": "1y max drawdown (signed)",
}
PCT = {"rev_growth", "rev_cagr_3y", "op_income_growth", "net_income_growth", "gross_margin", "op_margin", "roe", "roa",
       "roic", "earnings_yield", "fcf_yield", "ebitda_to_ev", "fcf_margin", "fcf_growth", "mom_12_1", "mom_6_1"}


def _fmt(key: str, v: float) -> str:
    return f"{v * 100:.1f}%" if key in PCT else f"{v:.2f}"


def packet(row: Mapping[str, object], extra: str = "") -> str:
    """Plain-text dossier from a screen row (``extra``: filings / estimates lines)."""
    def num(k: str) -> float:
        try:
            v = float(row.get(k))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return math.nan
        return v

    cap, adv = num("market_cap"), num("adv_usd")
    lines = [f"{row.get('symbol')} - {row.get('name')} ({row.get('exchange')}), sector {row.get('sector')}",
             f"market cap ${cap / 1e9:.2f}B, price ${num('price'):.2f}, median daily $ volume ${adv / 1e6:.1f}M",
             f"quant score {num('sigma_score'):.1f}/100 (data coverage {num('coverage') * 100:.0f}%); "
             "category z-scores vs sector peers (0 = typical):"]
    cats = [f"{c} {num('cat_' + c):+.2f}" for c in CATEGORIES if num("cat_" + c) == num("cat_" + c)]
    lines.append("  " + "; ".join(cats))
    lines.append("underlying measures:")
    for fs in CATEGORIES.values():
        parts = [f"{LABELS.get(f, f)} {_fmt(f, num(f))}" for f in fs if num(f) == num(f)]
        if parts:
            lines.append("  " + "; ".join(parts))
    missing = [LABELS.get(f, f) for fs in CATEGORIES.values() for f in fs if num(f) != num(f)]
    if missing:
        lines.append("not available: " + ", ".join(missing[:12]))
    if extra:
        lines.append(extra)
    lines.append("not scored (no free point-in-time data): earnings revisions, catalysts.")
    return "\n".join(lines)


def filings_line(filings: pd.DataFrame, symbol: str, n: int = 6) -> str:
    sub = filings[filings["symbol"] == symbol].sort_values("filed", ascending=False)
    sub = sub[sub["form"].isin(["10-K", "10-Q", "8-K", "20-F", "6-K", "DEF 14A"])].head(n)
    if sub.empty:
        return ""
    return "recent filings: " + "; ".join(f"{r.form} {r.filed}" for r in sub.itertuples())


def estimates_line(est: pd.DataFrame, symbol: str) -> str:
    sub = est[est["symbol"] == symbol]
    if sub.empty:
        return ""
    last = sub[sub["snapshot_date"] == sub["snapshot_date"].max()]
    rows = []
    for r in last.itertuples():
        eps = getattr(r, "eps_estimate_average", None)
        if eps is not None and eps == eps:
            rows.append(f"{getattr(r, 'horizon', '?')} EPS est {eps:.2f}")
    return ("analyst estimates (snapshot " + str(last["snapshot_date"].iloc[0]) + "): " + "; ".join(rows[:4])) if rows else ""
