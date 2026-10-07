"""Price-based features as of a date, from bars up to and including that date only.

Returns use split- and dividend-adjusted closes (``adj``); price and dollar volume use actual
traded (raw) prices, since market capitalisation is raw price x shares outstanding.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MONTH = 21


def dedupe_bars(df: pd.DataFrame) -> pd.DataFrame:
    """One bar per symbol-day (the last one loaded wins): repeated loads can leave duplicates."""
    return df.drop_duplicates(["symbol", "date"], keep="last").reset_index(drop=True)


def _ret(c: np.ndarray, back_end: int, back_start: int) -> float:
    if len(c) <= back_start:
        return math.nan
    return float(c[-1 - back_end] / c[-1 - back_start] - 1)


def features(adj: pd.DataFrame, raw: pd.DataFrame, as_of: date, max_stale_days: int = 7) -> pd.DataFrame:
    """Per symbol: last_date, price, adv_usd (63-day median), mom_12_1, mom_6_1, ret_1y, vol_1y, max_dd_1y."""
    adj = dedupe_bars(adj[adj["date"] <= as_of]).sort_values(["symbol", "date"])
    raw = dedupe_bars(raw[raw["date"] <= as_of]).sort_values(["symbol", "date"])
    raw_by = {s: g for s, g in raw.groupby("symbol")}
    rows: list[dict[str, Any]] = []
    for sym, g in adj.groupby("symbol"):
        last = g["date"].iloc[-1]
        if (as_of - last).days > max_stale_days or sym not in raw_by:
            continue
        c = g["close"].to_numpy(dtype=float)
        r = raw_by[sym]
        dollar = (r["close"] * r["volume"]).to_numpy(dtype=float)[-63:]
        year = c[-(TRADING_DAYS + 1):]
        logr = np.diff(np.log(year))
        enough = len(c) > TRADING_DAYS
        rows.append({
            "symbol": sym, "last_date": last, "price": float(r["close"].iloc[-1]),
            "adv_usd": float(np.median(dollar)) if len(dollar) else math.nan,
            "mom_12_1": _ret(c, MONTH, TRADING_DAYS),
            "mom_6_1": _ret(c, MONTH, TRADING_DAYS // 2),
            "ret_1y": _ret(c, 0, TRADING_DAYS),
            "vol_1y": float(np.std(logr, ddof=1) * math.sqrt(TRADING_DAYS)) if enough else math.nan,
            "max_dd_1y": float(1 - np.min(year / np.maximum.accumulate(year))) if enough else math.nan,
        })
    return pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame(
        columns=["last_date", "price", "adv_usd"])
