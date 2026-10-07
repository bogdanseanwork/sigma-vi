"""Provisional whole-market screen (milestone D) - EXPERIMENTAL, NOT VALIDATED.

    python -m sigma.screen                     # as of the latest price date in the local data
    python -m sigma.screen --as-of 2024-06-28  # any past date: uses only data public by then

Ranks every investable US common stock on the mandate's starting factor weights and marks the
funnel stages (Phase 1 candidates, Phase 2 advancers). Writes screen_<date>.csv and
screen_report.txt to the project folder. This list is the starting point the backtests are designed
to beat or discard; it is not a recommendation.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from sigma.core.config import DEFAULT_DOTENV
from sigma.factors import fundamentals, market, scoring
from sigma.factors.scoring import CATEGORIES, NOT_SCORED, effective_weights

Funnel = list[tuple[str, int]]

_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
        "twenty": 20, "twenty-five": 25, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100}
_FRAC = {"half": 2, "third": 3, "fourth": 4, "quarter": 4, "fifth": 5, "tenth": 10}
_RATIO = re.compile(r"each\s+represent(?:s|ing)\s+(?:an?\s+)?([a-z\-]+|\d+(?:\.\d+)?)(?:\s*\((\d+)\))?", re.I)


def adr_ratio(name: str) -> float | None:
    """Underlying shares per listed share: 1 for ordinary listings, the stated ratio for ADRs.

    SEC share counts are in underlying (ordinary) shares while the ADR trades per depositary share,
    so market cap = price x shares / ratio. ``None`` when an ADR's listing name does not state it.
    """
    if not re.search(r"depositary|\bADS\b|\bADRs?\b", name, re.I):
        return 1.0
    m = _RATIO.search(name)
    if not m:
        return None
    if m.group(2):
        return float(m.group(2))
    word = m.group(1).lower()
    if word.replace(".", "").isdigit():
        return float(word)
    if word.startswith("one-") and word[4:] in _FRAC:
        return 1.0 / _FRAC[word[4:]]
    return float(_NUM[word]) if word in _NUM else None


@dataclass(frozen=True)
class Filters:
    min_market_cap: float = 250e6   # small caps compete, micro caps are excluded (execution risk)
    min_adv_usd: float = 2e6        # median daily dollar volume over ~3 months
    min_price: float = 2.0
    stage1: int = 500               # Phase 1: 300-500 candidates
    stage2: int = 150               # Phase 2: 100-150 advance to deep research


DEFAULT_FILTERS = Filters()


def build_screen(fund: pd.DataFrame, mkt: pd.DataFrame, uni: pd.DataFrame, sectors: pd.DataFrame,
                 f: Filters = DEFAULT_FILTERS) -> tuple[pd.DataFrame, Funnel]:
    funnel: Funnel = []
    df = uni[uni["kind"] == "common"].merge(mkt, left_on="symbol", right_index=True, how="inner")
    funnel.append(("common stocks with a recent price", len(df)))
    df = df[df["cik"].notna()].assign(cik=lambda d: d["cik"].astype(int))
    funnel.append(("linked to an SEC filer", len(df)))
    df = df.sort_values("adv_usd", ascending=False).drop_duplicates("cik", keep="first")
    funnel.append(("one listing per company (most liquid class)", len(df)))
    df = df.merge(fund, left_on="cik", right_index=True, how="inner")
    funnel.append(("has current SEC filings", len(df)))
    sec = sectors.drop_duplicates("cik", keep="last").set_index("cik")[["sic", "sector"]]
    df = df.merge(sec, left_on="cik", right_index=True, how="left")
    df["sector"] = df["sector"].fillna("Unknown")
    df = df[df["sector"] != "Shell"]
    funnel.append(("not a shell / SPAC", len(df)))
    df = df.assign(adr_ratio=df["name"].fillna("").map(adr_ratio))
    df = df[df["adr_ratio"].notna()]
    funnel.append(("ADR share ratio known (or not an ADR)", len(df)))
    # SEC counts underlying shares; express them per listed share so price x shares = market cap
    df = df.assign(shares=pd.to_numeric(df["shares"], errors="coerce") / df["adr_ratio"])
    df = df.assign(market_cap=pd.to_numeric(df["price"]) * df["shares"])
    df = df[df["market_cap"] >= f.min_market_cap]
    funnel.append((f"market cap >= ${f.min_market_cap / 1e6:,.0f}M", len(df)))
    df = df[df["adv_usd"] >= f.min_adv_usd]
    funnel.append(("liquid (median daily $ volume)", len(df)))
    df = df[df["price"] >= f.min_price]
    funnel.append((f"price >= ${f.min_price:g}", len(df)))
    df = df[pd.to_numeric(df["revenue_1y"], errors="coerce").notna() & df["mom_12_1"].notna()]
    funnel.append(("a year of price history and prior-year results", len(df)))
    df = df.reset_index(drop=True)
    scored = scoring.score(df)
    out = pd.concat([df.drop(columns=[c for c in scored.columns if c in df.columns]), scored], axis=1)
    out = out[out["sigma_score"].notna()].sort_values("sigma_score", ascending=False).reset_index(drop=True)
    funnel.append(("scored", len(out)))
    out.insert(0, "rank", range(1, len(out) + 1))
    out["stage"] = "screened"
    out.loc[out["rank"] <= f.stage1, "stage"] = "candidate"
    out.loc[out["rank"] <= f.stage2, "stage"] = "advance"
    return out, funnel


def report(ranked: pd.DataFrame, funnel: Funnel, as_of: date, f: Filters, top: int = 50) -> str:
    w = effective_weights()
    lines = [
        f"SIGMA VI provisional screen - as of {as_of} - EXPERIMENTAL, NOT VALIDATED",
        "=" * 78,
        "Uses only data that was public on the as-of date. Factor weights are the mandate's starting",
        "hypotheses; the backtests (next milestone) decide which factors survive. Not a recommendation.",
        "",
        "Funnel",
        *[f"  {n:>6,}  {name}" for name, n in funnel],
        f"  stage 1 (candidates): top {f.stage1}; stage 2 (advance to research): top {f.stage2}",
        "",
        "Category weights used (mandate weights rescaled over the categories that can be scored)",
        *[f"  {c:<18} {w[c]:6.1%}   {', '.join(CATEGORIES[c])}" for c in w],
        *[f"  {c:<18}  not scored: {why}" for c, why in NOT_SCORED.items()],
        "",
    ]
    adv = ranked[ranked["stage"] == "advance"]
    if len(adv):
        mix = adv["sector"].value_counts()
        lines += [f"Sector mix of the stage-2 list ({len(adv)} companies)",
                  *[f"  {s:<24} {n:>4}" for s, n in mix.items()], ""]
    cats = list(w)
    short = {c: c.split()[0][:6] for c in cats}
    hdr = f"{'rank':>4} {'symbol':<7} {'name':<28} {'sector':<22} {'mcap $B':>8} {'score':>6} " + \
        " ".join(f"{short[c]:>6}" for c in cats)
    title = f"Top {min(top, len(ranked))} (category columns are sector-relative z-scores)"
    lines += [title, hdr, "-" * len(hdr)]
    for _, r in ranked.head(top).iterrows():
        lines.append(
            f"{r['rank']:>4} {r['symbol']:<7} {str(r['name'])[:28]:<28} {str(r['sector'])[:22]:<22} "
            f"{r['market_cap'] / 1e9:>8.1f} {r['sigma_score']:>6.1f} "
            + " ".join(f"{r[f'cat_{c}']:>6.2f}" if pd.notna(r[f"cat_{c}"]) else f"{'-':>6}" for c in cats))
    flags = ranked["shares_rescaled"] if "shares_rescaled" in ranked else pd.Series(dtype=bool)
    rescaled = int(flags.fillna(False).astype(bool).sum())
    lines += [
        "",
        "Known limitations of this provisional screen",
        "  * Sectors are approximated from SEC SIC codes (GICS is not free).",
        "  * Market cap = price x latest diluted share count; multi-class companies are approximate.",
        "  * Foreign filers reporting under IFRS (20-F) are not yet covered by the fundamentals parser.",
        "  * Banks and insurers lack some factors (gross margin, EBITDA) and are not scored on free cash",
        "    flow or accruals (their cash flows mix in loans and policies); they are scored on the rest.",
        "  * ADRs are converted to per-depositary-share counts using the ratio in the listing name; ADRs",
        "    whose name does not state the ratio are set aside until a free ratio source is added.",
        f"  * Share counts corrected for 1000x units errors in the filings: {rescaled}.",
        "  * Revenue uses the largest revenue figure a company tags for a period (some tag only a part).",
        "  * Delisted companies are not yet linked to their SEC filings: fine for today's list, but the",
        "    backtests need that link before their results can be trusted.",
    ]
    return "\n".join(lines)


def _dates(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
    """Parquet readers differ on whether dates come back as datetime64 or date objects: use dates."""
    for c in cols:
        if c in df and pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.date.where(df[c].notna(), None)
    return df


def _log(msg: str) -> None:
    sys.stdout.write(f"[{datetime.now():%H:%M:%S}] {msg}\n")
    sys.stdout.flush()


def load_inputs(store: Any, as_of: date | None, log: Callable[[str], None] | None = None
                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, date]:
    """Everything the screen needs as of one date, using only data public by then."""
    log = log or _log
    uni = store.read("universe.parquet")
    if store.exists("sectors"):
        sectors = store.read("sectors")
    else:
        log("No sector data yet - run:  .\\.venv\\Scripts\\python.exe -m sigma.data.load sectors")
        sectors = pd.DataFrame(columns=["cik", "sic", "sector"])
    if as_of is None:
        spy = store.read("prices", "raw", columns=["symbol", "date"], filters=[("symbol", "=", "SPY")])
        as_of = max(_dates(spy, "date")["date"])
    log(f"Screen as of {as_of}: loading prices...")
    since = as_of - timedelta(days=400)
    pf = [("date", ">=", since), ("date", "<=", as_of)]
    cols = ["symbol", "date", "close", "volume"]
    adj = _dates(store.read("prices", "all", columns=cols, filters=pf), "date")
    raw = _dates(store.read("prices", "raw", columns=cols, filters=pf), "date")
    mkt = market.features(adj, raw, as_of)
    log(f"  price features for {len(mkt):,} symbols; loading fundamentals known by {as_of}...")
    ciks = sorted({int(c) for c in uni.loc[uni["kind"] == "common", "cik"].dropna()})
    window = as_of - timedelta(days=int(4.6 * 365))  # 3-year growth needs ~4 years of periods
    facts = store.read("fundamentals",
                       columns=["cik", "metric", "tag", "tag_rank", "start", "end", "period", "value",
                                "filed", "known_at"],
                       filters=[("known_at", "<=", as_of), ("end", ">=", window), ("cik", "in", ciks)])
    facts = _dates(facts, "start", "end", "filed", "known_at")
    log(f"  {len(facts):,} facts; building trailing-twelve-month figures...")
    fund = fundamentals.snapshot(facts, as_of)
    log(f"  fundamentals for {len(fund):,} companies; scoring...")
    return uni, sectors, fund, mkt, as_of


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover — reads the local data store
    from sigma.data.paths import data_dir
    from sigma.data.store import ParquetStore

    ap = argparse.ArgumentParser(prog="python -m sigma.screen")
    ap.add_argument("--as-of", type=date.fromisoformat, default=None)
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args(argv)
    store = ParquetStore(data_dir())
    uni, sectors, fund, mkt, as_of = load_inputs(store, args.as_of)
    f = Filters()
    ranked, funnel = build_screen(fund, mkt, uni, sectors, f)
    text = report(ranked, funnel, as_of, f, top=args.top)
    folder = DEFAULT_DOTENV.parent
    keep = ["rank", "stage", "symbol", "name", "exchange", "sector", "sic", "market_cap", "price", "adv_usd",
            "sigma_score", "coverage", *[f"cat_{c}" for c in effective_weights()],
            *[x for fs in CATEGORIES.values() for x in fs], "fiscal_end"]
    ranked[keep].to_csv(folder / f"screen_{as_of}.csv", index=False)
    (folder / "screen_report.txt").write_text(text + "\n", encoding="utf-8")
    store.path("screens").mkdir(parents=True, exist_ok=True)
    ranked[keep].to_csv(store.path("screens", f"screen_{as_of}.csv"), index=False)
    sys.stdout.write("\n" + text + "\n")
    _log(f"saved screen_{as_of}.csv and screen_report.txt in {folder}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
