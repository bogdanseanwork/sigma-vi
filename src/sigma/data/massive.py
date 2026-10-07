"""Massive (free plan) reference data: historical ticker -> SEC company id for delisted stocks.

The SEC's ticker file only lists companies that trade today, so delisted stocks cannot be joined to
their filings from it. Massive's ticker list keeps inactive tickers with their company id (CIK) and
delisting date. When a ticker was reused by a later company, the most recent delisting wins.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import date
from typing import Any
from urllib.parse import parse_qs, urlparse

Fetch = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]


def parse_results(page: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    for r in page.get("results") or []:
        cik = str(r.get("cik") or "").strip()
        if not cik.isdigit():
            continue
        d = str(r.get("delisted_utc") or "")[:10]
        yield {"symbol": str(r["ticker"]).upper(), "name": r.get("name") or "", "cik": int(cik),
               "delisted": date.fromisoformat(d) if d else None, "type": r.get("type")}


def iter_delisted(fetch: Fetch, limit: int = 1000) -> Iterator[dict[str, Any]]:
    """Every inactive common-stock ticker that has a company id, following the cursor."""
    params: dict[str, Any] = {"market": "stocks", "type": "CS", "active": "false", "limit": limit,
                              "sort": "ticker", "order": "asc"}
    while True:
        page = fetch("/v3/reference/tickers", params)
        yield from parse_results(page)
        nxt = page.get("next_url")
        if not nxt:
            return
        cursor = parse_qs(urlparse(nxt).query).get("cursor")
        if not cursor:
            return
        params = {**params, "cursor": cursor[0]}


def cik_by_symbol(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    best: dict[str, tuple[date, int]] = {}
    for r in rows:
        key = (r["delisted"] or date.max, r["cik"])
        cur = best.get(r["symbol"])
        if cur is None or key[0] > cur[0]:
            best[r["symbol"]] = key
    return {s: v[1] for s, v in best.items()}
