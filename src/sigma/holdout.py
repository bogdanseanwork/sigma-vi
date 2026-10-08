"""The single look at the sealed holdout, with the frozen model.

    python -m sigma.holdout --reason "final frozen-model run"

Refuses to run unless sigma.freeze has frozen the methodology and nothing has changed since. Runs the
unchanged screen on holdout dates only, judges the result against the criteria written down at freeze
time, and writes holdout_report.txt. There is no tuning step after this. If the verdict is bad, the
honest outcomes are to report it as is or to start a new research cycle whose holdout is later data.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any

import pandas as pd

from sigma import freeze
from sigma.backtest import engine, robustness, run, splits
from sigma.core.config import DEFAULT_DOTENV


def assess(data: pd.DataFrame, bench: pd.DataFrame, sp: splits.Splits, n: int = freeze.TOP_N,
           trials: int = 5000, unlock: str = "final", ledger: Any = None) -> dict[str, Any]:
    """Run the frozen method on the holdout rows of ``data`` and compare with the criteria."""
    hold = data[data["date"].map(lambda d: sp.period_of(d) == "holdout")]
    if hold.empty:
        raise ValueError("no holdout dates in the data")
    res = run.evaluate(hold, bench.reindex(sorted(hold["date"].unique())), n=n, splits=sp, unlock=unlock,
                       ledger=ledger)
    per = res["portfolio"]["holdout"]
    ic = res["ic"]["holdout"].loc["sigma_score"]
    rand = robustness.random_matched(hold, n=n, trials=trials)
    net = per["base"]
    result = {"cagr_net": net["cagr"], "cagr_universe": res["universe"]["holdout"]["cagr"],
              "ic": ic["mean"], "ic_t": ic["t"], "random_percentile": rand["percentile"],
              "worst_dd": net["max_drawdown"], "dd_limit": freeze.DD_LIMIT}
    return {"result": result, "verdict": freeze.judge(result), "evaluation": res, "random": rand,
            "dates": sorted(hold["date"].unique())}


def render(a: dict[str, Any], frozen_digest: str, reason: str) -> str:
    pct = lambda x: "n/a" if x != x else f"{x * 100:.1f}%"  # noqa: E731
    r, ev = a["result"], a["evaluation"]
    L = ["SIGMA VI sealed-holdout result - one look, frozen model", "=" * 78,
         f"Opened {datetime.now():%Y-%m-%d %H:%M} - reason: {reason}",
         f"Frozen model digest {frozen_digest[:16]}...   {len(a['dates'])} quarterly rebalances, "
         f"{a['dates'][0]} to {a['dates'][-1]}", "", "Pre-registered criteria"]
    for v in a["verdict"]:
        L.append(f"  [{'PASS' if v['passed'] else 'FAIL'}] {v['criterion']}  ({v['value']})")
    passed = sum(v["passed"] for v in a["verdict"])
    L += ["", f"{passed} of {len(a['verdict'])} criteria passed.", "", "Numbers"]
    p = ev["portfolio"]["holdout"]
    L += [f"  top-{freeze.TOP_N} net (base cost) CAGR {pct(r['cagr_net'])}, gross {pct(p['gross']['cagr'])}, "
          f"pessimistic {pct(p['pessimistic']['cagr'])}",
          f"  equal-weight universe CAGR {pct(r['cagr_universe'])}",
          *[f"  {b}: {pct(s['cagr'])}" for b, s in ev["benchmarks"]["holdout"].items()],
          f"  mean rank IC {r['ic']:.3f} (t {r['ic_t']:.2f}) over {len(a['dates'])} quarters",
          f"  matched random portfolios: SIGMA beat {a['random']['percentile']:.0f}% of "
          f"{a['random']['trials']:,}",
          f"  stocks that stopped trading while held: {ev['ended']['holdout']}", "",
          "Reading this honestly",
          "  * This is one short window. Passing is encouraging, not proof; failing is information, not",
          "    a reason to retune on this data (that would make the holdout training data).",
          "  * Stopped-trading stocks are held at their last price; bankruptcies are not marked to zero."]
    return "\n".join(L)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - reads the local data store
    from sigma.data.paths import data_dir
    from sigma.data.store import ParquetStore
    from sigma.screen import _dates

    ap = argparse.ArgumentParser(prog="python -m sigma.holdout")
    ap.add_argument("--reason", required=True)
    ap.add_argument("--second-look", action="store_true",
                    help="only after a first look; labelled in the ledger")
    args = ap.parse_args(argv)
    try:
        freeze.open_gate(freeze.FROZEN_FILE, freeze.LEDGER_FILE, args.reason, args.second_look)
    except freeze.GateClosed as e:
        sys.stdout.write(f"REFUSED: {e}\n")
        return 1
    sp = splits.Splits()
    store = ParquetStore(data_dir())
    wide = run.price_matrix(_dates(store.read("prices", "all", columns=["symbol", "date", "close"]), "date"))
    days = [d.date() for d in wide.index]
    dates = engine.rebalance_dates(days, sp.validation_end, max(days), "Q")
    panel = run.build_panels(store, dates)
    fwd, ended = engine.forward_returns(wide, dates, with_flags=True)
    long = fwd.stack().reset_index()
    long.columns = ["date", "symbol", "fwd"]
    flag = ended.stack().reset_index()
    flag.columns = ["date", "symbol", "ended"]
    data = panel.merge(long, on=["date", "symbol"]).merge(flag, on=["date", "symbol"], how="left")
    data = data[data["date"] > date.fromisoformat(sp.validation_end.isoformat())]
    bench = fwd[[b for b in run.BENCHMARKS if b in fwd.columns]]
    out = assess(data, bench, sp, unlock=args.reason)
    doc = json.loads(freeze.FROZEN_FILE.read_text(encoding="utf-8"))
    text = render(out, doc["digest"], args.reason)
    (DEFAULT_DOTENV.parent / "holdout_report.txt").write_text(text + "\n", encoding="utf-8")
    sys.stdout.write("\n" + text + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
