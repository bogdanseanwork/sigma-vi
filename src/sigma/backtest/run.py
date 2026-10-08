"""Backtest the provisional screen through time using only what was public at each date.

    python -m sigma.backtest.run                      # train + validation periods (holdout stays sealed)
    python -m sigma.backtest.run --freq M --top 40

Each rebalance date: rebuild the whole screen from that date's data (prices, filings known by then,
delisted companies included), hold the top N equal-weight until the next date, charge costs, and
measure how well the score and each category ranked the next period's returns. Per-date panels are
cached so a re-run only does new work.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from datetime import date, datetime
from typing import Any

import pandas as pd

from sigma.backtest import analysis, engine, splits
from sigma.core.config import DEFAULT_DOTENV
from sigma.factors.scoring import effective_weights

PERIODS = ("train", "validation", "holdout")
BENCHMARKS = ("SPY", "QQQ", "IWM", "RSP", "IWB")


DEFAULT_SPLITS = splits.Splits()


def evaluate(panel: pd.DataFrame, bench: pd.DataFrame, n: int = 40,
             splits: splits.Splits = DEFAULT_SPLITS, unlock: str | None = None,
             periods_per_year: int = 4, ledger: Any = None) -> dict[str, Any]:
    """Statistics per period. ``panel``: date, symbol, sigma_score, cat_*, fwd[, ended];
    ``bench``: forward return per date for each benchmark column."""
    guard = splits_guard(splits, ledger)
    guard.require(panel["date"].unique(), unlock)
    cats = ["sigma_score", *[c for c in panel.columns if c.startswith("cat_")]]
    port = {c: engine.run_portfolio(panel, n=n, cost_bps=bps) for c, bps in engine.COST_BPS.items()}
    uni = engine.universe_benchmark(panel)
    ic_all = analysis.information_coefficients(panel, cats)
    out: dict[str, Any] = {"portfolio": {}, "universe": {}, "benchmarks": {}, "ic": {}, "quintiles": {},
                           "ended": {}, "counts": {}}
    for per in PERIODS:
        dates = [d for d in sorted(panel["date"].unique()) if splits.period_of(d) == per]
        if not dates:
            continue
        sel = pd.Index(dates)
        out["portfolio"][per] = {c: analysis.performance(p.loc[sel.intersection(p.index), "net"],
                                                         periods_per_year) for c, p in port.items()}
        gross = port["base"].loc[sel, "gross"]
        out["portfolio"][per]["gross"] = analysis.performance(gross, periods_per_year)
        out["portfolio"][per]["turnover"] = float(port["base"].loc[sel, "turnover"].mean())
        out["universe"][per] = analysis.performance(uni.reindex(sel), periods_per_year)
        out["benchmarks"][per] = {b: analysis.performance(bench[b].reindex(sel), periods_per_year)
                                  for b in bench.columns}
        sub = panel[panel["date"].isin(dates)]
        out["ic"][per] = pd.DataFrame({c: analysis.ic_summary(ic_all.loc[ic_all.index.isin(dates), c])
                                       for c in cats}).T
        q = analysis.quintile_returns(sub, "sigma_score")
        out["quintiles"][per] = q.mean() if len(q) else q
        picks = sub.groupby("date", group_keys=False).apply(lambda g: g.nlargest(n, "sigma_score"),
                                                            include_groups=False)
        out["ended"][per] = int(picks["ended"].sum()) if "ended" in picks else 0
        out["counts"][per] = {"dates": len(dates), "names_per_date": float(sub.groupby("date").size().mean())}
    return out


def splits_guard(sp: splits.Splits, ledger: Any) -> splits.HoldoutGuard:
    from pathlib import Path
    return splits.HoldoutGuard(sp, Path(ledger) if ledger else DEFAULT_DOTENV.parent / "holdout_ledger.txt")


def _pct(x: float) -> str:
    return "   n/a" if x != x else f"{x * 100:6.1f}%"


def _row(label: str, s: dict[str, float]) -> str:
    if s.get("periods", 0) == 0:
        return f"  {label:<26} no data"
    return (f"  {label:<26} {_pct(s['cagr'])} {_pct(s['volatility'])} {s['sharpe']:6.2f} "
            f"{_pct(s['max_drawdown'])} {_pct(s['hit_rate'])}")


def render(res: dict[str, Any], top_n: int = 40) -> str:
    L = ["SIGMA VI backtest of the provisional screen - EXPERIMENTAL",
         "=" * 78,
         f"Top {top_n} by sigma_score, equal weight, rebalanced each period from that date's data only.",
         "Splits: train / validation shown; the sealed holdout is not opened here.", ""]
    w = effective_weights()
    for per in PERIODS:
        if per not in res["portfolio"]:
            continue
        c = res["counts"][per]
        L += [f"[{per}]  {c['dates']} rebalances, about {c['names_per_date']:,.0f} scored names each",
              f"  {'':<26} {'CAGR':>7} {'vol':>7} {'Sharpe':>6} {'maxDD':>7} {'hit':>7}"]
        p = res["portfolio"][per]
        L += [_row("SIGMA top-N gross", p["gross"]),
              _row("SIGMA top-N optimistic cost", p["optimistic"]),
              _row("SIGMA top-N base cost", p["base"]),
              _row("SIGMA top-N pessimistic cost", p["pessimistic"]),
              _row("universe (equal weight)", res["universe"][per])]
        L += [_row(b, s) for b, s in res["benchmarks"][per].items()]
        L += [f"  average turnover per rebalance: {p['turnover'] * 100:.0f}%; picks that stopped trading "
              f"during the period: {res['ended'][per]}", ""]
        ic = res["ic"][per]
        L += ["  Rank information coefficient vs next-period return (IC > 0: factor ranked correctly)",
              f"  {'':<26} {'mean IC':>8} {'t-stat':>7} {'hit':>6} {'n':>3}"]
        for name, r in ic.iterrows():
            label = name.replace("cat_", "")
            wt = f" ({w.get(label, 0) * 100:.0f}%)" if label in w else ""
            L.append(f"  {label + wt:<26} {r['mean']:8.3f} {r['t']:7.2f} {r['hit'] * 100:5.0f}% "
                     f"{int(r['n']):3d}")
        q = res["quintiles"][per]
        if len(q):
            L += ["", "  Average next-period return by score quintile (1 = lowest score):  "
                  + "  ".join(f"Q{int(k)} {v * 100:5.1f}%" for k, v in q.items())]
        L.append("")
    L += ["Reading this honestly",
          "  * These are the mandate's starting weights, not tuned on this data - but the factor list was",
          "    chosen after seeing the 2026 screen, so treat the train/validation numbers as optimistic.",
          "  * A factor with |t| below about 2 is indistinguishable from noise at this sample size.",
          "  * Stocks that stop trading are valued at their last price; bankruptcies are not marked to zero,",
          "    so results are flattered by an unknown amount. Counted above as 'stopped trading'.",
          "  * Costs are flat presets per side, not the full spread/impact model of VALIDATION.md section 6."]
    return "\n".join(L)


def price_matrix(px: pd.DataFrame) -> pd.DataFrame:
    """Adjusted closes as a dates x symbols matrix (duplicate bars collapsed)."""
    from sigma.factors.market import dedupe_bars

    wide = dedupe_bars(px).pivot(index="date", columns="symbol", values="close")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def _log(msg: str) -> None:
    sys.stdout.write(f"[{datetime.now():%H:%M:%S}] {msg}\n")
    sys.stdout.flush()


def build_panels(store: Any, dates: Sequence[date],  # pragma: no cover
                 log: Callable[[str], None] = _log, with_factors: bool = False) -> pd.DataFrame:
    """Per-date scored universe, cached. ``with_factors`` also keeps the individual factor values
    (needed for factor-level tests) in a separate cache folder."""
    from sigma import screen
    from sigma.factors.scoring import CATEGORIES

    folder = "panels_v2" if with_factors else "panels"
    extra = [f for fs in CATEGORIES.values() for f in fs] if with_factors else []
    frames = []
    for i, d in enumerate(dates, 1):
        cache = store.path("backtests", folder, f"{d}.parquet")
        if not cache.exists():
            log(f"[{i}/{len(dates)}] building the screen as of {d}")
            uni, sectors, fund, mkt, _ = screen.load_inputs(store, d, log=lambda m: None)
            ranked, _ = screen.build_screen(fund, mkt, uni, sectors, screen.Filters())
            keep = ["symbol", "cik", "sector", "market_cap", "adv_usd", "sigma_score", "composite",
                    "coverage", *[c for c in ranked.columns if c.startswith("cat_")], *extra]
            store.write(ranked[keep].assign(date=d), "backtests", folder, f"{d}.parquet")
        frames.append(store.read("backtests", folder, f"{d}.parquet"))
    return pd.concat(frames, ignore_index=True)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover — reads the local data store
    from sigma.data.paths import data_dir
    from sigma.data.store import ParquetStore
    from sigma.screen import _dates

    ap = argparse.ArgumentParser(prog="python -m sigma.backtest.run")
    ap.add_argument("--start", type=date.fromisoformat, default=date(2017, 3, 1))
    ap.add_argument("--freq", choices=["Q", "M"], default="Q")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--unlock-holdout", default=None, help="reason; ONLY for the final frozen-model run")
    args = ap.parse_args(argv)
    sp = splits.Splits()
    store = ParquetStore(data_dir())
    _log("Loading adjusted prices...")
    px = _dates(store.read("prices", "all", columns=["symbol", "date", "close"]), "date")
    wide = price_matrix(px)
    days = [d.date() for d in wide.index]
    end = max(days) if args.unlock_holdout else sp.validation_end
    dates = engine.rebalance_dates(days, args.start, end, args.freq)
    splits_guard(sp, None).require(dates, args.unlock_holdout)
    _log(f"{len(dates)} rebalance dates, {dates[0]} to {dates[-1]}")
    panel = build_panels(store, dates)
    fwd, ended = engine.forward_returns(wide, dates, with_flags=True)
    long = fwd.stack().rename("fwd").reset_index().rename(columns={"level_0": "date", "level_1": "symbol"})
    long.columns = ["date", "symbol", "fwd"]
    flag = ended.stack().rename("ended").reset_index()
    flag.columns = ["date", "symbol", "ended"]
    data = panel.merge(long, on=["date", "symbol"], how="inner")
    data = data.merge(flag, on=["date", "symbol"], how="left")
    bench = fwd[[b for b in BENCHMARKS if b in fwd.columns]]
    res = evaluate(data, bench, n=args.top, splits=sp, unlock=args.unlock_holdout)
    text = render(res, top_n=args.top)
    out = DEFAULT_DOTENV.parent / "backtest_report.txt"
    out.write_text(text + "\n", encoding="utf-8")
    sys.stdout.write("\n" + text + "\n")
    _log(f"saved {out.name}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
