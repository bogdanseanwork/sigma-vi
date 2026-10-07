"""Load the whole-market dataset onto this machine.

    python -m sigma.data.load            # everything: universe, fundamentals, prices, report
    python -m sigma.data.load prices     # one step: universe | fundamentals | prices | sectors | report

Resumable: finished pieces are skipped on the next run, so a sleep or crash never costs more than the
piece in progress. Output goes to the data directory (see sigma.data.paths) and a plain-text coverage
report is written to data_report.txt in the project folder. Uses only free sources.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from sigma.core.config import DEFAULT_DOTENV, export_dotenv
from sigma.data import alpaca, sec, universe
from sigma.data.http import HttpClient, HttpError
from sigma.data.paths import data_dir
from sigma.data.store import ParquetStore

PRICE_START = date(2016, 1, 1)
PRICE_CHUNK = 100          # symbols per request batch
FUNDAMENTALS_FLUSH_ROWS = 250_000  # write a Parquet part at this many facts (bounds memory)
Log = Callable[[str], None]


def _log(msg: str) -> None:
    sys.stdout.write(f"[{datetime.now():%H:%M:%S}] {msg}\n")
    sys.stdout.flush()


def _clients(env: Mapping[str, str]) -> dict[str, HttpClient]:  # pragma: no cover — network
    alpaca_auth = {"APCA-API-KEY-ID": env["ALPACA_API_KEY_ID"],
                   "APCA-API-SECRET-KEY": env["ALPACA_API_SECRET_KEY"]}
    sec_headers = {"User-Agent": env["SEC_EDGAR_USER_AGENT"], "Accept-Encoding": "identity"}
    return {
        "alpaca_trading": HttpClient("https://paper-api.alpaca.markets", alpaca_auth, per_minute=150),
        "alpaca_data": HttpClient("https://data.alpaca.markets", alpaca_auth, per_minute=180),
        "sec": HttpClient("https://www.sec.gov", sec_headers, per_minute=300),  # SEC allows 10/s
        "sec_data": HttpClient("https://data.sec.gov", sec_headers, per_minute=300),
    }


# ---------------------------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------------------------
def step_universe(store: ParquetStore, clients: Mapping[str, HttpClient], log: Log = _log) -> int:
    log("Universe: listing every US equity on Alpaca (active and delisted)...")
    assets: list[dict[str, Any]] = []
    for status in ("active", "inactive"):
        params = {"status": status, "asset_class": "us_equity"}
        assets += clients["alpaca_trading"].get_json("/v2/assets", params)
    log(f"  {len(assets):,} listings received; fetching SEC ticker-to-company map...")
    tickers = sec.parse_company_tickers(clients["sec"].get_json("/files/company_tickers_exchange.json"))
    rows = universe.build(assets, tickers)
    n = store.write(rows, "universe.parquet")
    kinds = pd.Series([r["kind"] for r in rows]).value_counts().to_dict()
    log(f"  universe saved: {n:,} listings on major exchanges {kinds}")
    return n


def step_fundamentals(store: ParquetStore, clients: Mapping[str, HttpClient], log: Log = _log) -> int:
    zpath = store.path("raw", "companyfacts.zip")
    fresh = zpath.exists() and (time.time() - zpath.stat().st_mtime) < 20 * 3600
    if fresh:
        log("Fundamentals: SEC companyfacts.zip is less than 20 hours old - reusing it.")
    else:
        log("Fundamentals: downloading SEC companyfacts.zip (about 1-1.5 GB, one file)...")
        last = [0]

        def progress(done: int) -> None:
            if done - last[0] >= 100 << 20:
                last[0] = done
                log(f"  {done / (1 << 30):.2f} GB downloaded")

        clients["sec"].download("/Archives/edgar/daily-index/xbrl/companyfacts.zip", zpath, progress)
    out_dir = store.path("fundamentals")
    done_marker = out_dir / "_built_from.txt"
    stamp = f"{zpath.stat().st_size}:{int(zpath.stat().st_mtime)}"
    if done_marker.exists() and done_marker.read_text().strip() == stamp:
        log("  fundamentals already built from this file - skipping.")
        return 0
    for old in out_dir.glob("*.parquet"):  # rebuilt in full from the new file
        old.unlink()
    log("  parsing every company's filings (keeps every version with its filing date)...")
    batch: list[dict[str, Any]] = []
    part = companies = total = 0
    for rows in sec.iter_companyfacts_zip(zpath):
        if rows:
            companies += 1
            batch.extend(rows)
        if len(batch) >= FUNDAMENTALS_FLUSH_ROWS:
            total += store.write(batch, "fundamentals", f"part-{part:04d}.parquet")
            part, batch = part + 1, []
            log(f"  {companies:,} companies, {total:,} facts")
    if batch:
        total += store.write(batch, "fundamentals", f"part-{part:04d}.parquet")
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker.write_text(stamp)  # written last: only a complete build is ever marked done
    log(f"  fundamentals saved: {companies:,} companies, {total:,} facts")
    return total


def step_sectors(store: ParquetStore, clients: Mapping[str, HttpClient], log: Log = _log) -> int:
    """SIC industry code per company from the SEC's submissions API (one request per company, once)."""
    from sigma.factors.sectors import sector_for_sic

    uni = store.read("universe.parquet")
    ciks = sorted({int(c) for c in uni.loc[uni["kind"] == "common", "cik"].dropna()})
    have = set(store.read("sectors", columns=["cik"])["cik"]) if store.exists("sectors") else set()
    todo = [c for c in ciks if c not in have]
    log(f"Sectors: {len(todo):,} of {len(ciks):,} companies still need an industry code")
    batch: list[dict[str, Any]] = []
    done = 0

    def flush() -> None:
        nonlocal batch
        if batch:
            store.write(batch, "sectors", f"part-{datetime.now():%Y%m%d%H%M%S%f}.parquet")
            batch = []

    for cik in todo:
        try:
            doc = clients["sec_data"].get_json(f"/submissions/CIK{cik:010d}.json")
        except HttpError as e:
            if e.status != 404:
                raise
            doc = {}
        sic = int(doc["sic"]) if str(doc.get("sic") or "").isdigit() else None
        batch.append({"cik": cik, "sic": sic, "sic_description": doc.get("sicDescription") or "",
                      "sector": sector_for_sic(sic)})
        done += 1
        if len(batch) >= 250:
            flush()
            log(f"  {done:,}/{len(todo):,}")
    flush()
    log(f"  sectors saved for {done:,} companies")
    return done


def chunks(items: Sequence[str], size: int) -> list[list[str]]:
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


def step_prices(store: ParquetStore, clients: Mapping[str, HttpClient], log: Log = _log,
                end: date | None = None) -> int:
    uni = store.read("universe.parquet")
    symbols = universe.price_symbols(uni.to_dict("records"))
    end = end or (date.today() - timedelta(days=1))
    batches = chunks(symbols, PRICE_CHUNK)
    log(f"Prices: {len(symbols):,} symbols in {len(batches)} batches, {PRICE_START} to {end}, raw + adjusted")
    total = 0
    for adjustment in ("raw", "all"):
        for i, batch in enumerate(batches):
            name = f"batch-{i:04d}.parquet"
            if store.exists("prices", adjustment, name):
                continue

            def fetch(params: Mapping[str, Any]) -> Mapping[str, Any]:
                return clients["alpaca_data"].get_json("/v2/stocks/bars", params)

            rows = _bars_dropping_rejected(fetch, batch, end, adjustment, log)
            total += store.write(rows if rows else pd.DataFrame(columns=["symbol", "date"]),
                                 "prices", adjustment, name)
            log(f"  {adjustment}: batch {i + 1}/{len(batches)} done ({len(rows):,} bars)")
    log(f"  prices saved: {total:,} new bars")
    return total


def _bars_dropping_rejected(fetch: alpaca.Fetch, batch: list[str], end: date, adjustment: str,
                            log: Log) -> list[dict[str, Any]]:
    """Fetch a batch; if the provider rejects a symbol, drop just that symbol and retry the batch."""
    symbols = list(batch)
    while symbols:
        try:
            return list(alpaca.iter_bars(fetch, symbols, PRICE_START, end, adjustment))
        except HttpError as e:
            m = re.search(r"invalid symbol: ([A-Za-z0-9.\-/]+)", str(e))
            if e.status != 400 or not m or m.group(1) not in symbols:
                raise
            symbols.remove(m.group(1))
            log(f"  skipped {m.group(1)}: the price provider does not recognise this symbol")
    return []


def coverage_report(store: ParquetStore) -> str:
    lines = [f"SIGMA VI data coverage - {datetime.now():%Y-%m-%d %H:%M}", f"data folder: {store.root}", ""]
    if not store.exists("universe.parquet"):
        return "\n".join([*lines, "universe: not loaded"])
    uni = store.read("universe.parquet")
    common = uni[uni["kind"] == "common"]
    lines += [
        f"listings on major exchanges: {len(uni):,}",
        f"  common stocks: {len(common):,} ({int(common['active'].sum()):,} active, "
        f"{int((~common['active']).sum()):,} delisted)",
        f"  by kind: {uni['kind'].value_counts().to_dict()}",
        f"  active common stocks matched to an SEC company: "
        f"{int(common[common['active']]['cik'].notna().sum()):,} of {int(common['active'].sum()):,}",
    ]
    if store.path("fundamentals").exists():
        f = store.read("fundamentals", columns=["cik", "metric", "end", "known_at"])
        if len(f):
            ends = pd.to_datetime(f["end"])
            lines += [
                "",
                f"fundamentals: {len(f):,} facts from {f['cik'].nunique():,} companies",
                f"  periods {ends.min():%Y-%m-%d} .. {ends.max():%Y-%m-%d}",
                f"  active common stocks with fundamentals: "
                f"{int(common[common['active']]['cik'].isin(set(f['cik'])).sum()):,}",
            ]
    if store.path("prices", "raw").exists():
        p = store.read("prices", "raw", columns=["symbol", "date"])
        if len(p):
            have = set(p["symbol"])
            days = pd.to_datetime(p["date"])
            delisted = common[~common["active"]]
            lines += [
                "",
                f"daily prices: {len(p):,} bars for {len(have):,} symbols",
                f"  dates {days.min():%Y-%m-%d} .. {days.max():%Y-%m-%d}",
                f"  delisted common stocks with prices: "
                f"{int(delisted['symbol'].isin(have).sum()):,} of {len(delisted):,}",
            ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover — orchestration
    ap = argparse.ArgumentParser(prog="python -m sigma.data.load")
    ap.add_argument("step", nargs="?", default="all",
                    choices=["all", "universe", "fundamentals", "prices", "sectors", "report"])
    args = ap.parse_args(argv)
    export_dotenv()
    env = dict(os.environ)
    store = ParquetStore(data_dir())
    _log(f"data folder: {store.root}")
    clients = _clients(env) if args.step != "report" else {}
    steps = ["universe", "fundamentals", "prices", "sectors"] if args.step == "all" else [args.step]
    for step in steps:
        if step == "universe":
            step_universe(store, clients)
        elif step == "fundamentals":
            step_fundamentals(store, clients)
        elif step == "prices":
            if not store.exists("universe.parquet"):
                step_universe(store, clients)
            step_prices(store, clients)
        elif step == "sectors":
            step_sectors(store, clients)
    report = coverage_report(store)
    sys.stdout.write("\n" + report + "\n")
    (DEFAULT_DOTENV.parent / "data_report.txt").write_text(report + "\n", encoding="utf-8")
    (store.root / "last_load.json").write_text(json.dumps({"finished": datetime.now().isoformat(),
                                                            "steps": steps}), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
