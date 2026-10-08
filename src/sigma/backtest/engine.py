"""Point-in-time portfolio mechanics: rebalance dates, forward returns, top-N portfolios, costs."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd

# One-way cost in basis points of traded value (spread + slippage + commission + impact), by preset.
# Small caps are costly to trade; the strategy must stay positive under "pessimistic".
MAX_STALE_DAYS = 7  # a price older than this (calendar days) is not a current quote
COST_BPS = {"optimistic": 10.0, "base": 30.0, "pessimistic": 75.0}


def rebalance_dates(trading_days: Sequence[date], start: date, end: date, freq: str = "Q") -> list[date]:
    """Last trading day of each month ("M") or quarter ("Q") in [start, end], complete periods only."""
    days = sorted(d for d in trading_days if start <= d <= end)
    if not days:
        return []
    s = pd.Series(days, index=pd.PeriodIndex(pd.to_datetime(days), freq=freq))
    last = s.groupby(level=0).last()
    out = []
    for period, d in last.items():
        period_end = period.end_time.date()
        # a period is complete if the data runs to within a few days of its end
        if (end >= period_end) or (period_end - days[-1]).days <= 3:
            out.append(d)
    return out


def forward_returns(prices: pd.DataFrame, dates: Sequence[date], with_flags: bool = False, step: int = 1
                    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Return from each rebalance date to the one ``step`` later (default: the next), per symbol
    (rows: date, columns: symbol). ``step=4`` on quarterly dates gives 12-month returns.

    ``prices``: adjusted closes (rows: trading dates, columns: symbols). The start price must be a bar
    from the last few days before the date; the end price is the last price on or before the next date,
    so a stock that stops trading mid-period is valued at its last traded price and flagged ``ended``.
    """
    ts = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    valid = prices.notna().to_numpy()
    stamp = np.where(valid, prices.index.to_numpy()[:, None], np.datetime64("NaT"))
    last_seen = pd.DataFrame(stamp, index=prices.index, columns=prices.columns).ffill().reindex(
        ts, method="ffill")
    last_px = prices.ffill().reindex(ts, method="ffill")
    fresh = (ts.to_numpy()[:, None] - last_seen.to_numpy()) <= np.timedelta64(MAX_STALE_DAYS, "D")
    p0 = last_px.where(fresh)
    rets = pd.DataFrame(last_px.to_numpy()[step:] / p0.to_numpy()[:-step] - 1, index=list(dates[:-step]),
                        columns=prices.columns)
    if not with_flags:
        return rets
    ended = pd.DataFrame(~fresh[step:] & ~np.isnan(p0.to_numpy()[:-step]), index=list(dates[:-step]),
                         columns=prices.columns)
    return rets, ended


def run_portfolio(panel: pd.DataFrame, n: int = 40, cost_bps: float = 30.0, score: str = "sigma_score"
                  ) -> pd.DataFrame:
    """Hold the top ``n`` by ``score`` at each date, equal weight; one row per rebalance date.

    ``panel``: columns date, symbol, <score>, fwd (return to the next rebalance). Names with no
    forward return are skipped (their weight goes to the rest). Turnover is the share of the
    portfolio replaced; the cost is ``cost_bps`` per side on traded value (turnover x 2 sides).
    """
    rows = []
    prev: set[str] = set()
    for d, g in panel.groupby("date", sort=True):
        picks = g.dropna(subset=[score]).nlargest(n, score)
        names = set(picks["symbol"])
        held = picks["fwd"].dropna()
        gross = float(held.mean()) if len(held) else np.nan
        new = len(names - prev) / max(len(names), 1)
        # the first period builds the portfolio from cash (buying only: one side); later periods sell
        # the names dropped and buy their replacements (two sides on the replaced share)
        cost = cost_bps / 1e4 if not prev else new * 2 * cost_bps / 1e4
        rows.append({"date": d, "gross": gross, "turnover": 1.0 if not prev else new, "cost": cost,
                     "net": gross - cost, "n_held": len(held)})
        prev = names
    return pd.DataFrame(rows).set_index("date")


def universe_benchmark(panel: pd.DataFrame) -> pd.Series:
    """Equal-weight return of every scored name each period: the market the screen could choose from."""
    return panel.dropna(subset=["fwd"]).groupby("date")["fwd"].mean()
