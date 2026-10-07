"""Alpaca market data: daily bars from the free consolidated (SIP) feed, 2016 onward."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import date
from typing import Any

Fetch = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def iter_bars(fetch: Fetch, symbols: Sequence[str], start: date, end: date, adjustment: str
              ) -> Iterator[dict[str, Any]]:
    """Yield one row per symbol-day, following next_page_token until exhausted.

    ``adjustment``: "raw" for actual traded prices (market cap, dollar volume) or "all" for
    split- and dividend-adjusted prices (total-return series).
    """
    token: str | None = None
    while True:
        page = fetch({
            "symbols": ",".join(symbols), "timeframe": "1Day",
            "start": f"{start.isoformat()}T00:00:00Z", "end": f"{end.isoformat()}T23:59:59Z",
            "adjustment": adjustment, "feed": "sip", "limit": 10000, "page_token": token,
        })
        for symbol, bars in (page.get("bars") or {}).items():
            for b in bars:
                yield {
                    "symbol": symbol, "date": date.fromisoformat(b["t"][:10]), "adjustment": adjustment,
                    "open": b.get("o"), "high": b.get("h"), "low": b.get("l"), "close": b["c"],
                    "volume": b.get("v"), "trades": b.get("n"), "vwap": b.get("vw"),
                }
        token = page.get("next_page_token")
        if not token:
            return
