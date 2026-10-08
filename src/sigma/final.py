"""Final portfolio: weights, confidence, portfolio analysis, and the written report with its audit.

    python -m sigma.final

Reads what the earlier stages wrote in the project folder (committee_audit.json / final_top40.csv,
screen_*.csv, backtest / robustness / holdout / simulation reports, frozen_model.json) and writes
SIGMA_VI_FINAL_REPORT.md and final_top40_weights.csv. Any stage that has not been run is reported as
NOT RUN rather than quietly omitted, and nothing here is described as validated unless its report says so.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MAX_WEIGHT, MIN_WEIGHT, MAX_SECTOR = 0.05, 0.01, 0.30
MIN_VOL = 0.15          # risk floor: a calm quarter must not make a stock look riskless


# ---------------------------------------------------------------------------------------------
# Weights: expected return x confidence x thesis strength / risk, within position limits
# ---------------------------------------------------------------------------------------------
def conviction_raw(df: pd.DataFrame) -> pd.Series:
    """Unconstrained conviction. ``final_score`` (0-100) stands in for expected return - the quant model's
    own ranking, not the LLM's return guess, which is too unreliable to size positions with."""
    risk = df["vol"].fillna(df["vol"].median()).clip(lower=MIN_VOL)
    conf = df["confidence"].clip(lower=10) / 100
    thesis = df["thesis_strength"].fillna(50).clip(lower=10) / 100
    return (df["final_score"].clip(lower=1) / 100) * conf * thesis / risk


def conviction_weights(raw: pd.Series, sector: pd.Series, cap: float = MAX_WEIGHT, floor: float = MIN_WEIGHT,
                       sector_cap: float = MAX_SECTOR) -> pd.Series:
    """Scale to 100%, then iteratively clip to [floor, cap] and to the sector limit, redistributing the
    excess pro rata to the names with room. Raises if the limits cannot all be met."""
    n = len(raw)
    if n * cap < 1 - 1e-9 or n * floor > 1 + 1e-9:
        raise ValueError(f"{n} names cannot satisfy floor {floor:.0%} / cap {cap:.0%}")
    w = raw / raw.sum()
    for _ in range(200):
        over = w > cap + 1e-12
        under = w < floor - 1e-12
        w = w.clip(floor, cap)
        sec = w.groupby(sector).transform("sum")
        sec_over = sec > sector_cap + 1e-12
        if sec_over.any():
            w = w.where(~sec_over, w * (sector_cap / sec))
        gap = 1 - w.sum()
        room = (w < cap - 1e-12) & ~(sec_over) & (w.groupby(sector).transform("sum") < sector_cap - 1e-12)
        if abs(gap) < 1e-10 and not over.any() and not under.any():
            break
        if room.any() and gap > 0:
            w = w + gap * (w.where(room, 0) / w.where(room, 0).sum())
        elif gap < 0:
            w = w + gap * (w / w.sum())
        else:
            break
    return w / w.sum()


def confidence_score(coverage: float, agreement: float, debate_conf: float, n_votes: int, panel_size: int = 6,
                     model_reliability: float = 0.5) -> float:
    """0-100. Quality of data (coverage), agreement among the panel, the debate judge's confidence, and how
    many members actually voted - times a reliability ceiling for the validated model (0.5 until the
    holdout supports more), so no stock can read as certain on the strength of an unproven model."""
    parts = [min(max(coverage, 0), 1), min(max(agreement, 0), 1), min(max(debate_conf, 0), 100) / 100,
             min(n_votes / panel_size, 1)]
    return float(100 * np.mean(parts) * (0.5 + 0.5 * model_reliability))


# ---------------------------------------------------------------------------------------------
# Portfolio analysis
# ---------------------------------------------------------------------------------------------
def exposures(df: pd.DataFrame, weight_col: str) -> dict[str, pd.Series]:
    w = df[weight_col]
    cap = pd.cut(df["market_cap"], [0, 2e9, 10e9, 200e9, float("inf")],
                 labels=["small (<2B)", "mid (2-10B)", "large (10-200B)", "mega (200B+)"])
    return {"sector": w.groupby(df["sector"]).sum().sort_values(ascending=False),
            "market cap": w.groupby(cap, observed=True).sum()}


def risk_stats(returns: pd.DataFrame, weights: pd.Series, bench: pd.Series | None = None) -> dict[str, float]:
    """Historical (not forecast) risk of holding these weights over the supplied daily returns."""
    cols = [c for c in weights.index if c in returns.columns]
    r = returns[cols].dropna(how="all").fillna(0.0)
    w = weights[cols] / weights[cols].sum()
    port = r @ w
    corr = r.corr().to_numpy()
    avg_corr = float((corr.sum() - len(cols)) / (len(cols) * (len(cols) - 1))) if len(cols) > 1 else math.nan
    ev = np.linalg.eigvalsh(np.cov(r.to_numpy().T)) if len(cols) > 1 else np.array([1.0])
    ev = np.clip(ev, 0, None)
    nav = (1 + port).cumprod()
    out = {"names_covered": len(cols), "vol": float(port.std() * math.sqrt(252)),
           "avg_pairwise_corr": avg_corr, "effective_names": float(1 / (w**2).sum()),
           "effective_risk_factors": float(ev.sum() ** 2 / (ev**2).sum()) if ev.sum() > 0 else math.nan,
           "max_drawdown": float((nav / nav.cummax() - 1).min())}
    if bench is not None:
        b = bench.reindex(port.index).fillna(0.0)
        out["beta"] = float(np.cov(port, b)[0, 1] / b.var()) if b.var() > 0 else math.nan
    return out


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------
STAGE_FILES = {
    "Provisional-score backtest (train + validation)": "backtest_report.txt",
    "Robustness (factor tests, ablation, random portfolios, perturbation, regimes)": "robustness_report.txt",
    "Sealed holdout (one look, frozen model)": "holdout_report.txt",
    "Simulation (path generators, drawdown and loss odds)": "simulation_report.txt",
    "Committee (votes, debate, red team, tournament)": "committee_report.txt",
}


def read_stage(home: Path, name: str) -> str | None:
    p = home / name
    return p.read_text(encoding="utf-8").strip() if p.exists() else None


def render(top: pd.DataFrame, watch: pd.DataFrame, expo: Mapping[str, pd.Series], risk: Mapping[str, float] | None,
          home: Path, debates: Mapping[str, Any] | None = None, red: Mapping[str, Any] | None = None) -> str:
    pct = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    debates, red = debates or {}, red or {}
    frozen = (home / "frozen_model.json").exists()
    holdout = read_stage(home, "holdout_report.txt")
    L = [f"# SIGMA VI - final report ({datetime.now():%Y-%m-%d})", "",
         "> **Status: EXPERIMENTAL.** This is a research output, not advice and not a validated strategy. "
         "Read the validation section before relying on any of it.", "",
         "## 1. The 40", "",
         "| # | Symbol | Name | Sector | Weight (conviction) | Weight (equal) | SIGMA score | Confidence |",
         "|--:|---|---|---|--:|--:|--:|--:|"]
    for i, (sym, r) in enumerate(top.iterrows(), 1):
        L.append(f"| {i} | {sym} | {str(r['name'])[:34]} | {r['sector']} | {pct(r['w_conviction'])} | "
                 f"{pct(r['w_equal'])} | {r['final_score']:.1f} | {r['confidence']:.0f} |")
    L += ["", "Conviction weight = quant score x confidence x thesis strength / volatility, then limited to "
          f"{MIN_WEIGHT:.0%}-{MAX_WEIGHT:.0%} per stock and {top.attrs.get('sector_cap', MAX_SECTOR):.0%} per sector"
          f"{' (relaxed from ' + format(MAX_SECTOR, '.0%') + ': too few sectors present)' if top.attrs.get('sector_cap', MAX_SECTOR) > MAX_SECTOR else ''}. Equal weight is shown "
          "alongside because the conviction inputs are unvalidated.", "", "## 2. Why each one (and what could go wrong)", ""]
    for sym, r in top.iterrows():
        d, rd = debates.get(sym, {}), red.get(sym, {})
        bull = next((tx for nm, tx in d.get("transcript", []) if nm == "Bull case"), "")
        L.append(f"**{sym} - {r['name']}** ({r['sector']}). quant {r['sigma_score']:.0f}, committee "
                 f"{r['committee_adj']:+.1f}, debate {r['debate_adj']:+.1f}, tournament {r['tournament_adj']:+.1f}, "
                 f"red team {r['red_penalty']:+.1f}.")
        if bull:
            L.append(f"- Case: {bull[:400].strip()}")
        crit = d.get("surviving_criticisms") or []
        if crit:
            L.append("- Criticisms that survived: " + "; ".join(crit[:3]))
        if rd.get("flags"):
            L.append("- Red team flags: " + "; ".join(rd["flags"][:3]))
        L.append("")
    L += ["## 3. Almost made it (watchlist 41-60)", ""]
    if watch.empty:
        L.append("None yet.")
    else:
        L += ["| Rank | Symbol | Name | Final score |", "|--:|---|---|--:|"]
        L += [f"| {40 + i} | {s} | {str(r['name'])[:34]} | {r['final_score']:.1f} |"
              for i, (s, r) in enumerate(watch.iterrows(), 1)]
    L += ["", "## 4. Portfolio analysis", ""]
    for k, ser in expo.items():
        L.append(f"- By {k} (conviction weights): " + ", ".join(f"{i} {pct(v)}" for i, v in ser.items()))
    if risk:
        L.append(f"- Historical risk of these weights: volatility {pct(risk['vol'])}, average pairwise correlation "
                 f"{risk['avg_pairwise_corr']:.2f}, ~{risk['effective_risk_factors']:.1f} effective independent "
                 f"risk factors, worst drawdown {pct(risk['max_drawdown'])}"
                 + (f", beta to SPY {risk['beta']:.2f}" if 'beta' in risk else "")
                 + f" ({risk['names_covered']} names with price history). Backward-looking.")
    L += ["", "## 5. Validation - what was tested and what it showed", "",
          f"- Methodology frozen before the holdout: **{'yes' if frozen else 'NO'}**.",
          f"- Sealed holdout opened: **{'yes' if holdout else 'NO - not run'}**.", ""]
    for title, fname in STAGE_FILES.items():
        text = read_stage(home, fname)
        L.append(f"### {title}")
        L += ["```", text[:3500], "```"] if text else ["**NOT RUN.** No claim is made for this stage."]
        L.append("")
    L += ["## 6. Model audit and limitations", "",
          "- Point-in-time: filings are usable one day after filing; every revision is kept. Prices are adjusted closes.",
          "- Sectors are approximated from SEC SIC codes; ADR share ratios are parsed from listing names; "
          "foreign (IFRS) filers are mostly not scored.",
          "- Earnings revisions and catalysts are not in the quant score (no free point-in-time history); the "
          "committee sees analyst estimates only for watchlist names, and only from the day archiving began.",
          "- The factor list was chosen after seeing a 2026 screen, so train/validation numbers are optimistic.",
          "- Stocks that stop trading are valued at their last price; bankruptcies are not marked to zero.",
          "- Costs are flat presets, not a full spread-and-impact model.",
          "- The committee is not backtestable (a model has read the past); it only adjusts the quant score within "
          "fixed limits, and it saw only SIGMA's numbers, not filing text or news.",
          "- Simulations reflect the strategy's own short history; they show spread, not edge.",
          "- Adversarial parameter search, a full regime-labelled stress library and a daily-step 10-year engine "
          "described in docs/VALIDATION.md are not built.", ""]
    return "\n".join(L)


def assemble(committee_table: pd.DataFrame, screen: pd.DataFrame, debates: Mapping[str, Any],
             top_n: int = 40, extra: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join committee scores with screen facts, attach confidence and weights; split top / watchlist."""
    scr = screen.set_index("symbol")
    df = committee_table.join(scr[["name", "sector", "market_cap", "coverage", "low_volatility"]], how="left")
    df["vol"] = -df["low_volatility"]
    df["confidence"] = [confidence_score(r.coverage, r.agreement, debates.get(s, {}).get("confidence", 0.0),
                                         int(r.n_votes)) for s, r in df.iterrows()]
    df["thesis_strength"] = [debates.get(s, {}).get("thesis_strength", float("nan")) for s in df.index]
    top, watch = df.head(top_n).copy(), df.iloc[top_n:top_n + extra].copy()
    top["w_equal"] = 1 / len(top)
    sectors = top["sector"].fillna("Unknown")
    cap = max(MAX_SECTOR, 1 / sectors.nunique() + 0.05)       # few sectors present: the limit cannot be met
    top["w_conviction"] = conviction_weights(conviction_raw(top), sectors, sector_cap=cap)
    top.attrs["sector_cap"] = cap
    return top, watch


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - reads the project folder
    from sigma.core.config import DEFAULT_DOTENV
    from sigma.data.paths import data_dir
    from sigma.data.store import ParquetStore

    home = DEFAULT_DOTENV.parent
    audit_p, table_p = home / "committee_audit.json", home / "final_top40.csv"
    if not audit_p.exists():
        import sys
        sys.stdout.write("The committee has not finished yet (no committee_audit.json). Run "
                         "python -m sigma.committee.run first.\n")
        return 1
    audit = json.loads(audit_p.read_text(encoding="utf-8"))
    table = pd.read_csv(table_p, index_col="symbol")
    screen = pd.read_csv(max(home.glob("screen_*.csv"), key=lambda p: p.stat().st_mtime))
    top, watch = assemble(table, screen, {s: d for s, d in audit["debates"].items()})
    risk = None
    store = ParquetStore(data_dir())
    try:
        px = store.read("prices", "all", columns=["symbol", "date", "close"])
        from sigma.factors.market import dedupe_bars
        wide = dedupe_bars(px).pivot(index="date", columns="symbol", values="close").sort_index()
        rets = wide.pct_change().iloc[-756:]
        risk = risk_stats(rets, top["w_conviction"], rets.get("SPY"))
    except Exception:
        risk = None
    text = render(top, watch, exposures(top, "w_conviction"), risk, home, audit["debates"], audit["red"])
    (home / "SIGMA_VI_FINAL_REPORT.md").write_text(text + "\n", encoding="utf-8")
    top.to_csv(home / "final_top40_weights.csv")
    watch.to_csv(home / "final_watchlist_41_60.csv")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
