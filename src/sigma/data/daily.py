"""Daily refresh for the watchlist, all on free tiers.

    python -m sigma.data.daily            # prices, filings, analyst estimates, macro
    python -m sigma.data.daily estimates  # one step: prices | filings | estimates | macro

Run it once a day (the roadmap's scheduler calls it). Each step is independent, so one provider being
down never blocks the others, and every step is safe to run twice. Alpha Vantage allows 25 requests a
day, so about 20 watchlist stocks get a fresh estimate snapshot per day, oldest first. Estimates have no
free history, so the daily snapshots *are* the history: each row is stamped with the day we saw it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from sigma.core.config import DEFAULT_DOTENV, export_dotenv
from sigma.data import alpaca
from sigma.data.http import HttpClient
from sigma.data.paths import data_dir
from sigma.data.store import ParquetStore

Log = Callable[[str], None]
Fetch = Callable[[str, Mapping[str, Any]], Any]

AV_DAILY_LIMIT = 25
ESTIMATE_BUDGET = 20           # leaves headroom for retries and the occasional earnings lookup
PRICE_LOOKBACK_DAYS = 10       # re-fetch recent days so a missed run or a late correction self-heals
MACRO_SERIES = {
    "DGS10": "10-year Treasury yield", "DGS2": "2-year Treasury yield", "T10Y2Y": "10y-2y yield curve",
    "FEDFUNDS": "Fed funds rate", "CPIAUCSL": "CPI (all urban)", "UNRATE": "Unemployment rate",
    "VIXCLS": "VIX", "BAMLH0A0HYM2": "High-yield credit spread", "DCOILWTICO": "WTI crude oil",
}
WATCHLIST = Path(__file__).resolve().parents[3] / "config" / "watchlist.toml"


class ProviderLimit(RuntimeError):
    """The provider answered with a quota or error message instead of data."""


def _log(msg: str) -> None:
    sys.stdout.write(f"[{datetime.now():%H:%M:%S}] {msg}\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------------------------
# Watchlist, budget, rotation
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Watchlist:
    stocks: list[str]
    benchmarks: list[str]

    @property
    def all(self) -> list[str]:
        return self.stocks + self.benchmarks


def load_watchlist(path: Path = WATCHLIST) -> Watchlist:
    doc = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return Watchlist([s["symbol"] for s in doc.get("stock", [])],
                     [s["symbol"] for s in doc.get("benchmark", [])])


class Budget:
    """A persisted per-day request counter, so reruns the same day cannot exceed a provider's quota."""

    def __init__(self, path: Path, limit: int = AV_DAILY_LIMIT):
        self.path, self.limit = Path(path), limit

    def _used(self, today: date) -> int:
        try:
            st = json.loads(self.path.read_text(encoding="utf-8"))
            return int(st["used"]) if st["date"] == today.isoformat() else 0
        except (OSError, ValueError, KeyError):
            return 0

    def take(self, wanted: int, today: date) -> int:
        """Reserve up to ``wanted`` requests; returns how many were granted."""
        used = self._used(today)
        granted = max(0, min(wanted, self.limit - used))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state = {"date": today.isoformat(), "used": used + granted}
        self.path.write_text(json.dumps(state), encoding="utf-8")
        return granted


def pick_for_estimates(symbols: Sequence[str], last_snapshot: Mapping[str, date], n: int) -> list[str]:
    """Never-fetched names first, then the stalest; ties keep the watchlist order."""
    order = {s: i for i, s in enumerate(symbols)}
    return sorted(symbols, key=lambda s: (last_snapshot.get(s, date.min), order[s]))[:n]


# ---------------------------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------------------------
def _num(v: Any) -> Any:
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def parse_estimates(doc: Mapping[str, Any], symbol: str, snapshot: date) -> list[dict[str, Any]]:
    """Alpha Vantage EARNINGS_ESTIMATES -> one row per forecast period, stamped with today's date."""
    for key in ("Information", "Note", "Error Message"):
        if doc.get(key):
            raise ProviderLimit(str(doc[key])[:160])
    rows = []
    for e in doc.get("estimates") or []:
        row = {k: _num(v) if k not in ("date", "horizon") else v for k, v in e.items()}
        row.update({"symbol": symbol, "snapshot_date": snapshot})
        rows.append(row)
    return rows


def parse_observations(doc: Mapping[str, Any], series: str, fetched: date) -> pd.DataFrame:
    rows = [{"series": series, "date": date.fromisoformat(o["date"]), "value": float(o["value"]),
             "fetched_at": fetched}
            for o in doc.get("observations", []) if o.get("value") not in (".", None, "")]
    return pd.DataFrame(rows, columns=["series", "date", "value", "fetched_at"])


def merge_observations(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Keep every (series, date, value) once. A revised value is a new row, so old vintages stay visible."""
    both = pd.concat([old, new], ignore_index=True)
    return both.drop_duplicates(["series", "date", "value"], keep="first").reset_index(drop=True)


def parse_filings(doc: Mapping[str, Any], symbol: str, cik: int) -> list[dict[str, Any]]:
    r = (doc.get("filings") or {}).get("recent") or {}
    out = []
    for acc, form, filed, primary in zip(r.get("accessionNumber", []), r.get("form", []),
                                         r.get("filingDate", []), r.get("primaryDocument", []), strict=False):
        d = date.fromisoformat(filed)
        out.append({"symbol": symbol, "cik": cik, "accession": acc, "form": form, "filed": d,
                    "known_at": d + timedelta(days=1), "primary_document": primary})
    return out


# ---------------------------------------------------------------------------------------------
# Steps (network calls are injected, so each step is testable)
# ---------------------------------------------------------------------------------------------
def step_prices(store: ParquetStore, fetch: alpaca.Fetch, symbols: Sequence[str], today: date,
                log: Log = _log) -> str:
    start, end = today - timedelta(days=PRICE_LOOKBACK_DAYS), today - timedelta(days=1)
    total = 0
    for adjustment in ("raw", "all"):
        rows = list(alpaca.iter_bars(fetch, list(symbols), start, end, adjustment))
        total += store.write(pd.DataFrame(rows) if rows else pd.DataFrame(columns=["symbol", "date"]),
                             "prices", adjustment, f"daily-{today.isoformat()}.parquet")
    return f"{total:,} bars for {len(symbols)} symbols, {start} to {end}"


def step_filings(store: ParquetStore, fetch: Fetch, symbols: Sequence[str], log: Log = _log) -> str:
    uni = store.read("universe.parquet", columns=["symbol", "cik"])
    ciks = {r.symbol: int(r.cik) for r in uni.dropna(subset=["cik"]).itertuples()}
    rows: list[dict[str, Any]] = []
    missing = []
    for s in symbols:
        if s not in ciks:
            missing.append(s)  # ETFs (SPY, QQQ) file no 10-Ks; that is expected
            continue
        rows += parse_filings(fetch(f"/submissions/CIK{ciks[s]:010d}.json", {}), s, ciks[s])
    new = pd.DataFrame(rows)
    old = store.read("filings.parquet") if store.exists("filings.parquet") else pd.DataFrame()
    both = pd.concat([old, new], ignore_index=True).drop_duplicates(["accession"], keep="first")
    store.write(both, "filings.parquet")
    skipped = ", ".join(missing) or "none"
    return (f"{len(new):,} recent filings for {len(symbols) - len(missing)} companies "
            f"(no SEC filer: {skipped})")


def step_estimates(store: ParquetStore, fetch: Fetch, symbols: Sequence[str], today: date,
                   budget: Budget, log: Log = _log) -> str:
    last: dict[str, date] = {}
    if store.exists("estimates"):
        snaps = store.read("estimates", columns=["symbol", "snapshot_date"])
        last = snaps.groupby("symbol")["snapshot_date"].max().to_dict() if len(snaps) else {}
    allowed = budget.take(min(ESTIMATE_BUDGET, len(symbols)), today)
    chosen = pick_for_estimates(symbols, last, allowed)
    rows: list[dict[str, Any]] = []
    done = 0
    try:
        for s in chosen:
            doc = fetch("/query", {"function": "EARNINGS_ESTIMATES", "symbol": s})
            rows += parse_estimates(doc, s, today)
            done += 1
    except ProviderLimit as e:
        log(f"  Alpha Vantage stopped after {done} stocks: {e}")
    if rows:
        store.write(pd.DataFrame(rows), "estimates", f"{today.isoformat()}.parquet")
    return (f"{done} of {len(symbols)} stocks snapshotted today ({len(rows)} forecast rows); "
            f"daily allowance {allowed}")


def step_macro(store: ParquetStore, fetch: Fetch, today: date, log: Log = _log) -> str:
    n = 0
    for series in MACRO_SERIES:
        doc = fetch("/fred/series/observations", {"series_id": series, "observation_start": "1990-01-01"})
        new = parse_observations(doc, series, today)
        name = f"{series}.parquet"
        old = store.read("macro", name) if store.exists("macro", name) else new.iloc[0:0]
        merged = merge_observations(old, new)
        n += len(merged) - len(old)
        store.write(merged, "macro", name)
    return f"{len(MACRO_SERIES)} series, {n:,} new or revised observations"


def run_steps(steps: Sequence[tuple[str, Callable[[], str]]], log: Log = _log) -> list[tuple[str, str]]:
    results = []
    for name, fn in steps:
        try:
            msg = fn()
        except Exception as e:
            msg = f"FAILED: {e}"
        log(f"{name}: {msg}")
        results.append((name, msg))
    return results


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - orchestration
    ap = argparse.ArgumentParser(prog="python -m sigma.data.daily")
    steps = ["prices", "filings", "estimates", "macro"]
    ap.add_argument("step", nargs="?", default="all", choices=["all", *steps])
    args = ap.parse_args(argv)
    export_dotenv()
    env = dict(os.environ)
    store, wl, today = ParquetStore(data_dir()), load_watchlist(), date.today()
    alpaca_auth = {"APCA-API-KEY-ID": env["ALPACA_API_KEY_ID"],
                   "APCA-API-SECRET-KEY": env["ALPACA_API_SECRET_KEY"]}
    sec_headers = {"User-Agent": env["SEC_EDGAR_USER_AGENT"], "Accept-Encoding": "identity"}
    data = HttpClient("https://data.alpaca.markets", alpaca_auth, per_minute=180)
    sec = HttpClient("https://data.sec.gov", sec_headers, per_minute=300)
    av = HttpClient("https://www.alphavantage.co", {}, per_minute=5)
    fred = HttpClient("https://api.stlouisfed.org", {}, per_minute=60)

    def with_key(client: HttpClient, key: str, extra: Mapping[str, str]) -> Fetch:
        return lambda path, params: client.get_json(path, {**params, **extra, key[0]: env[key[1]]})

    plan: dict[str, Callable[[], str]] = {
        "prices": lambda: step_prices(store, lambda p: data.get_json("/v2/stocks/bars", p), wl.all, today),
        "filings": lambda: step_filings(store, lambda p, q: sec.get_json(p, q), wl.stocks),
        "estimates": lambda: step_estimates(
            store, with_key(av, ("apikey", "ALPHA_VANTAGE_API_KEY"), {}), wl.stocks, today,
            Budget(store.path("state", "alpha_vantage_budget.json"))),
        "macro": lambda: step_macro(
            store, with_key(fred, ("api_key", "FRED_API_KEY"), {"file_type": "json"}), today),
    }
    names = list(plan) if args.step == "all" else [args.step]
    results = run_steps([(n, plan[n]) for n in names])
    lines = [f"SIGMA VI daily refresh - {datetime.now():%Y-%m-%d %H:%M}",
             f"watchlist: {len(wl.stocks)} stocks + {len(wl.benchmarks)} benchmarks", ""]
    lines += [f"{n:10s} {m}" for n, m in results]
    (DEFAULT_DOTENV.parent / "daily_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if any(m.startswith("FAILED") for _, m in results) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
