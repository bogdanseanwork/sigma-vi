"""Investable universe: every US-listed equity on major exchanges, active and delisted.

Rules (SELECTION_SPEC: Investable Universe):
* exchanges NYSE, NASDAQ, NYSE American (AMEX), NYSE Arca, Cboe BZX; OTC excluded
* each listing classified as common / etf / warrant / right / unit / preferred so screens can keep
  operating companies and benchmarks while excluding instruments that are not stocks
* delisted (inactive) symbols are kept so historical tests are not survivorship-biased
The classification is name-based and labelled as such; SEC SIC codes refine it later.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

MAJOR_EXCHANGES = frozenset({"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"})
BENCHMARK_ETFS = frozenset({"SPY", "QQQ", "IWB", "IWM", "RSP", "IWD", "IWF", "MTUM", "QUAL", "VLUE", "USMV"})

_KIND_PATTERNS = [
    ("warrant", re.compile(r"\bwarrants?\b", re.I)),
    ("right", re.compile(r"\brights?\b", re.I)),
    ("unit", re.compile(r"\bunits?\b", re.I)),
    ("preferred", re.compile(r"\bpreferred\b|\bpfd\b|%\s*(series|notes?)\b", re.I)),
    ("etf", re.compile(
        r"\betf\b|\bfund\b|\bishares\b|\bspdr\b|\bproshares\b|\bvaneck\b|\binvesco\b.*\btrust\b", re.I)),
]


def classify(name: str, symbol: str) -> str:
    if "." in symbol and re.search(r"\.(PR|P)[A-Z]?$", symbol):
        return "preferred"
    for kind, pat in _KIND_PATTERNS:
        if pat.search(name):
            return kind
    return "common"


def build(assets: Iterable[Mapping[str, Any]], sec_tickers: Mapping[str, Mapping[str, Any]]
          ) -> list[dict[str, Any]]:
    out = []
    for a in assets:
        if a.get("class") != "us_equity" or a.get("exchange") not in MAJOR_EXCHANGES:
            continue
        symbol = str(a["symbol"]).upper()
        sec = sec_tickers.get(symbol) or sec_tickers.get(symbol.replace(".", "-"))
        out.append({
            "symbol": symbol, "name": a.get("name") or "", "exchange": a["exchange"],
            "active": a.get("status") == "active", "tradable": bool(a.get("tradable")),
            "kind": classify(a.get("name") or "", symbol),
            "cik": int(sec["cik"]) if sec else None,
        })
    return sorted(out, key=lambda r: r["symbol"])


def price_symbols(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Symbols to download prices for: common stocks (active and delisted) plus benchmark ETFs."""
    return sorted(r["symbol"] for r in rows if r["kind"] == "common" or r["symbol"] in BENCHMARK_ETFS)
