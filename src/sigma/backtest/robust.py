"""Robustness report for the provisional screen (milestone E2). Holdout stays sealed.

    python -m sigma.backtest.robust

Runs every test in sigma.backtest.robustness on the train and validation periods and writes
robustness_report.txt. It reuses the cached per-date panels (it builds the factor-level ones the first
time, which takes about as long as the first backtest did).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from datetime import date
from typing import Any

import pandas as pd

from sigma.backtest import engine, robustness, splits
from sigma.backtest.run import build_panels, price_matrix, splits_guard
from sigma.core.config import DEFAULT_DOTENV
from sigma.factors.scoring import CATEGORIES, effective_weights

FACTORS = [f for fs in CATEGORIES.values() for f in fs]
CAT_COLS = [f"cat_{c}" for c in CATEGORIES]
PERTURBATIONS = (0.05, 0.10, 0.20)
USED = ("train", "validation")


def analyse(panel: pd.DataFrame, panel12: pd.DataFrame, market: pd.Series, sp: splits.Splits,
            n: int = 40, trials: int = 5000) -> dict[str, Any]:
    """``panel``: next-period returns in ``fwd``; ``panel12``: the same rows with 12-month returns in
    ``fwd`` (rows without one dropped). ``market``: next-period return of a broad index per date."""
    weights = effective_weights()
    w = {k: v for k, v in weights.items() if f"cat_{k}" in panel.columns}
    cols = ["sigma_score", *CAT_COLS, *[f for f in FACTORS if f in panel.columns]]
    res: dict[str, Any] = {"weights": w, "n": n}
    res["ic_1q"] = robustness.ic_by_period(panel, cols, sp.period_of)
    res["ic_12m"] = robustness.ic_by_period(panel12, cols, sp.period_of, step=4)
    res["ablation"] = robustness.ablation(panel, w, sp.period_of, n=n)
    res["random"], res["perturb"], res["regime"] = {}, {}, {}
    for per in USED:
        sub = panel[panel["date"].map(lambda d, per=per: sp.period_of(d) == per)]
        if sub.empty:
            continue
        res["random"][per] = robustness.random_matched(sub, n=n, trials=trials)
        res["perturb"][per] = [robustness.perturbation(sub, w, pct, n=n) for pct in PERTURBATIONS]
        mk = market.reindex(sorted(sub["date"].unique())).dropna()
        res["regime"][per] = robustness.regimes(sub, mk, n=n)
    tests = {}
    for horizon, key in (("1q", "ic_1q"), ("12m", "ic_12m")):
        for per, tab in res[key].items():
            for name, r in tab.iterrows():
                tests[f"{horizon} {per} {name}"] = (float(r["t"]), int(r["n"]))
    res["ledger"] = robustness.correct(tests)
    return res


def _f(x: float, pct: bool = False) -> str:
    return "   n/a" if x != x else (f"{x * 100:6.1f}%" if pct else f"{x:7.3f}")


def render(res: dict[str, Any]) -> str:
    L = ["SIGMA VI robustness of the provisional screen - EXPERIMENTAL (milestone E2)", "=" * 78,
         f"Top {res['n']} by sigma_score. Train and validation only; the holdout was not opened.", ""]
    led = res["ledger"]
    L += [f"Tests run: {len(led)} rank-IC tests. With that many, a few |t| > 2 appear by luck alone.",
          f"  raw p < 0.05: {int((led.p < 0.05).sum())}   after Holm: {int((led.p_holm < 0.05).sum())}"
          f"   after Benjamini-Hochberg: {int((led.p_bh < 0.05).sum())}", ""]
    for horizon, key in (("Next quarter", "ic_1q"), ("12 months ahead (non-overlapping dates)", "ic_12m")):
        L.append(f"Factor rank IC, {horizon}")
        for per in USED:
            tab = res[key].get(per)
            if tab is None:
                continue
            L.append(f"  [{per}]  {'':<22} {'IC':>7} {'t':>6} {'hit':>5} {'n':>3}  p(Holm)")
            tag = "1q" if key == "ic_1q" else "12m"
            for name, r in tab.iterrows():
                ph = led.loc[f"{tag} {per} {name}", "p_holm"]
                mark = "  *" if ph == ph and ph < 0.05 else ""
                L.append(f"          {name.replace('cat_', ''):<26} {_f(r['mean'])} {r['t']:6.2f} "
                         f"{r['hit'] * 100:4.0f}% {int(r['n']):3d}  {_f(ph)}{mark}")
        L.append("")
    L.append("Ablation: drop one category, re-rank. Negative change = that category was helping.")
    ab = res["ablation"]
    for per in USED:
        sub = ab[ab.period == per]
        if sub.empty:
            continue
        L.append(f"  [{per}]  {'':<20} {'IC':>7} {'dIC':>7} {'CAGR':>8} {'dCAGR':>8}")
        for r in sub.itertuples():
            L.append(f"          {r.dropped:<22} {_f(r.ic)} {_f(r.d_ic)} "
                     f"{_f(r.cagr, True)} {_f(r.d_cagr, True)}")
    L.append("")
    L.append("Matched random portfolios (same sector mix each date, gross of costs)")
    for per, r in res["random"].items():
        L.append(f"  [{per}] SIGMA top-N {_f(r['sigma_cagr'], True)} a year; {r['trials']:,} random: mean "
                 f"{_f(r['random_mean'], True)}, 5th-95th pct {_f(r['random_p05'], True)} to "
                 f"{_f(r['random_p95'], True)}; SIGMA beat {r['percentile']:.0f}% of them")
    L.append("")
    L.append("Weight perturbation (every category weight moved at random within +/-x)")
    for per, rows in res["perturb"].items():
        for r in rows:
            L.append(f"  [{per}] +/-{r['pct'] * 100:.0f}%: {r['mean_overlap'] * 100:.0f}% of picks unchanged "
                     f"(worst run {r['min_overlap'] * 100:.0f}%); CAGR {_f(r['cagr_p05'], True)} to "
                     f"{_f(r['cagr_p95'], True)} vs {_f(r['base_cagr'], True)} base")
    L.append("")
    L.append("Regimes (labelled by the market's realised return, so descriptive only)")
    for per, tab in res["regime"].items():
        for r in tab.itertuples():
            L.append(f"  [{per}] {r.regime:<12} {r.quarters:2d} quarters: top-N {_f(r.top_n, True)}, "
                     f"universe {_f(r.universe, True)}, excess {_f(r.excess, True)}, "
                     f"IC {_f(r.ic)} per quarter")
    L += ["", "Reading this honestly",
          "  * Passing needs more than a positive number: edge versus the random portfolios (95th pct+),",
          "    a factor surviving Holm/BH correction, and the same sign in train and validation.",
          "  * Returns are gross of costs here; costs (see backtest_report.txt) only make it worse.",
          "  * Stopped-trading stocks are held at their last price; bankruptcies are not marked to zero.",
          "  * Few quarters means wide error bars. Regime labels use hindsight."]
    return "\n".join(L)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - reads the local data store
    from sigma.backtest.run import BENCHMARKS, _log
    from sigma.data.paths import data_dir
    from sigma.data.store import ParquetStore
    from sigma.screen import _dates

    sp = splits.Splits()
    store = ParquetStore(data_dir())
    _log("Loading adjusted prices...")
    wide = price_matrix(_dates(store.read("prices", "all", columns=["symbol", "date", "close"]), "date"))
    days = [d.date() for d in wide.index]
    dates = engine.rebalance_dates(days, date(2017, 3, 1), sp.validation_end, "Q")
    splits_guard(sp, None).require(dates, None)
    _log(f"{len(dates)} rebalance dates; building or loading factor-level panels (first time is slow)")
    panel = build_panels(store, dates, with_factors=True)
    # 12-month returns stop at the end of validation: a later price would be a holdout price, and the
    # holdout may not inform any design decision. The last four validation dates simply have none.
    all_dates = dates
    f1 = engine.forward_returns(wide, all_dates)
    f12 = engine.forward_returns(wide, all_dates, step=4)

    def long(m: pd.DataFrame, name: str) -> pd.DataFrame:
        out = m.stack().reset_index()
        out.columns = ["date", "symbol", name]
        return out

    data = panel.merge(long(f1, "fwd"), on=["date", "symbol"], how="inner")
    data12 = panel.merge(long(f12, "fwd"), on=["date", "symbol"], how="inner")
    market = f1["SPY"] if "SPY" in f1 else f1[[b for b in BENCHMARKS if b in f1]].iloc[:, 0]
    res = analyse(data, data12, market, sp)
    text = render(res)
    out = DEFAULT_DOTENV.parent / "robustness_report.txt"
    out.write_text(text + "\n", encoding="utf-8")
    sys.stdout.write("\n" + text + "\n")
    _log(f"saved {out.name}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
