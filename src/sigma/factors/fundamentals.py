"""Point-in-time company fundamentals: trailing-twelve-month (TTM) flows and period-end balances.

Everything is computed from SEC facts that were public on the as-of date (``known_at <= as_of``),
using the latest filed version of each period (so restatements count only after they are filed).

TTM for a flow metric at fiscal period end E, in order of preference:
1. the annual (FY) figure ending at E;
2. year-to-date at E + the prior fiscal year - the prior year's same year-to-date
   (how 10-Q cash-flow statements are reported, and exact for income statements too);
3. the sum of four consecutive discrete quarters ending at E.
If none is possible the value is missing; a partial year is never passed off as a full one.
Dates match within ``TOL`` days so 52/53-week fiscal years line up.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import pandas as pd

from sigma.data import sec

TOL = timedelta(days=10)
YEAR = timedelta(days=365)
DURATION_PERIODS = frozenset({"Q", "H1", "9M", "FY"})
MAX_COMPARATIVE_AGE = timedelta(days=5 * 366)  # 10-Ks carry ~3 years of comparatives; older = error

FLOW_METRICS = ("revenue", "cost_of_revenue", "gross_profit", "rnd", "operating_income",
                "interest_expense", "pretax_income", "income_tax", "net_income", "dna", "sbc", "cfo",
                "capex", "acquisitions", "buybacks", "dividends_paid")
INSTANT_METRICS = ("cash", "short_term_investments", "receivables", "inventory", "current_assets",
                   "total_assets", "goodwill", "current_liabilities", "total_liabilities", "debt_current",
                   "debt_noncurrent", "short_term_borrowings", "debt_total", "deferred_revenue", "equity")
# Metrics where some filers tag only a component under the preferred tag (e.g. insurers and REITs
# tag contract revenue under the ASC 606 tag and the total under "Revenues"): take the largest.
MAX_TAG_METRICS = frozenset({"revenue"})
SCALE_GLITCH = (300.0, 3000.0)  # a share count ~1000x its reference is a units error in the filing
ANCHOR_METRICS = ("revenue", "net_income", "operating_income", "cfo")
_YTD_LEN = {"Q": 1, "H1": 2, "9M": 3}


def _near(a: date, b: date) -> bool:
    return abs(a - b) <= TOL


@dataclass(frozen=True)
class Period:
    start: date | None
    end: date
    period: str
    value: float


class Periods:
    """The point-in-time-selected periods of one metric for one company."""

    def __init__(self, items: Iterable[Period]):
        self.items = sorted(items, key=lambda p: (p.end, p.start or date.min))

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, Any]]) -> Periods:
        df = point_in_time(pd.DataFrame(list(rows)), date.max - timedelta(days=1))
        return cls(Period(s if pd.notna(s) else None, e, p, float(v))
                   for s, e, p, v in zip(df["start"], df["end"], df["period"], df["value"], strict=True))

    def durations_ending(self, end: date, kinds: Iterable[str]) -> list[Period]:
        ks = set(kinds)
        return [p for p in self.items if p.period in ks and _near(p.end, end)]

    def latest_duration_end(self) -> date | None:
        ends = [p.end for p in self.items if p.period in DURATION_PERIODS]
        return max(ends) if ends else None

    def instant_near(self, end: date) -> float | None:
        hits = [p for p in self.items if p.period == "I" and _near(p.end, end)]
        return min(hits, key=lambda p: abs(p.end - end)).value if hits else None


def ttm(ps: Periods, end: date) -> float | None:
    fy = ps.durations_ending(end, ["FY"])
    if fy:
        return fy[-1].value
    ytd = ps.durations_ending(end, _YTD_LEN)
    if ytd:
        cur = max(ytd, key=lambda p: (_YTD_LEN[p.period], -(p.start or end).toordinal()))
        assert cur.start is not None
        prev_fy = ps.durations_ending(cur.start - timedelta(days=1), ["FY"])
        prev_ytd = [p for p in ps.durations_ending(cur.end - YEAR, [cur.period])
                    if p.start and _near(p.start, cur.start - YEAR)]
        if prev_fy and prev_ytd:
            return cur.value + prev_fy[-1].value - prev_ytd[-1].value
    total, at = 0.0, end
    for _ in range(4):
        q = ps.durations_ending(at, ["Q"])
        if not q or q[-1].start is None:
            return None
        total += q[-1].value
        at = q[-1].start - timedelta(days=1)
    return total


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop facts whose dates cannot be right (typos such as a period ending in 2215)."""
    end, filed = pd.to_datetime(df["end"]), pd.to_datetime(df["filed"])
    ok = (end <= filed) & (end >= filed - MAX_COMPARATIVE_AGE)
    if "start" in df:
        start = pd.to_datetime(df["start"])
        ok &= start.isna() | (start <= end)
    return df[ok]


def point_in_time(df: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Facts public on ``as_of``; per (company, metric, period) the best tag's latest version."""
    known = df[pd.to_datetime(df["known_at"]) <= pd.Timestamp(as_of)]
    known = known.assign(_f=pd.to_datetime(known["filed"]), _s=pd.to_datetime(known["start"]))
    period = ["cik", "metric", "_s", "end"]
    is_max = known["metric"].isin(MAX_TAG_METRICS)
    ranked = known[~is_max].sort_values(["tag_rank", "_f"], ascending=[True, False])
    ranked = ranked.drop_duplicates(period, keep="first")
    # largest-tag metrics: latest version of each tag, then the largest value per period
    biggest = known[is_max].sort_values("_f", ascending=False).drop_duplicates([*period, "tag"], keep="first")
    biggest = biggest.sort_values(["value", "tag_rank"], ascending=[False, True]).drop_duplicates(period)
    out = pd.concat([ranked, biggest])
    return out.drop(columns=["_f", "_s"]).sort_values(["cik", "metric", "end"]).reset_index(drop=True)


def _latest_shares(ps: Periods) -> float | None:
    end = ps.latest_duration_end()
    if end is None:
        return None
    hits = [p for p in ps.items if p.period in DURATION_PERIODS and p.end == end]
    return min(hits, key=lambda p: p.end - (p.start or p.end)).value


def _share_reference(diluted: Periods | None, cover: Periods | None) -> float | None:
    """A share count at the right scale: the smaller of the cover-page count and recent quarters."""
    refs: list[float] = []
    if cover and cover.items:
        refs.append(cover.items[-1].value)
    if diluted:
        end = diluted.latest_duration_end()
        recent = [p.value for p in diluted.items
                  if p.period in DURATION_PERIODS and end and (end - p.end).days <= 2 * 366 and p.value > 0]
        if recent:
            refs.append(min(recent))
    refs = [r for r in refs if r > 0]
    return min(refs) if refs else None


def _rescale(value: float | None, ref: float | None) -> tuple[float | None, bool]:
    if value is None or not ref or value <= 0:
        return value, False
    if SCALE_GLITCH[0] <= value / ref <= SCALE_GLITCH[1]:
        return value / 1000.0, True
    return value, False


def _debt(row: Mapping[str, Any]) -> float | None:
    cur, non, stb, tot = (row.get(k) for k in ("debt_current", "debt_noncurrent", "short_term_borrowings",
                                                "debt_total"))
    if cur is not None or non is not None:
        return (cur or 0.0) + (non or 0.0) + (stb or 0.0)
    if tot is not None:
        return tot
    return stb


def snapshot(df: pd.DataFrame, as_of: date, max_age_days: int = 200) -> pd.DataFrame:
    """One row per company: TTM flows (now, 1 and 3 years earlier) and balances (now, 1 year earlier).

    ``fiscal_end`` is the latest fiscal period end with income or cash-flow data public on ``as_of``;
    companies whose latest period is older than ``max_age_days`` are treated as not reporting.
    """
    pit = point_in_time(clean(df), as_of)
    out: list[dict[str, Any]] = []
    for cik, g in pit.groupby("cik", sort=True):
        by_metric = {m: Periods(Period(s if pd.notna(s) else None, e, p, float(v))
                                for s, e, p, v in zip(h["start"], h["end"], h["period"], h["value"],
                                                      strict=True))
                     for m, h in g.groupby("metric")}
        anchors = [e for m in ANCHOR_METRICS if m in by_metric
                   if (e := by_metric[m].latest_duration_end()) is not None]
        if not anchors:
            continue
        end = max(anchors)
        if (as_of - end).days > max_age_days:
            continue
        row: dict[str, Any] = {"cik": cik, "fiscal_end": end}
        for m in FLOW_METRICS:
            ps = by_metric.get(m)
            for suffix, at in (("", end), ("_1y", end - YEAR), ("_3y", end - 3 * YEAR)):
                row[m + suffix] = ttm(ps, at) if ps else None
        for m in INSTANT_METRICS:
            ps = by_metric.get(m)
            row[m] = ps.instant_near(end) if ps else None
            row[m + "_1y"] = ps.instant_near(end - YEAR) if ps else None
        row["debt"] = _debt(row)
        sd = by_metric.get("shares_diluted")
        so = by_metric.get("shares_outstanding")
        ref = _share_reference(sd, so)
        latest = (_latest_shares(sd) if sd else None) or (so.items[-1].value if so and so.items else None)
        row["shares"], row["shares_rescaled"] = _rescale(latest, ref)
        row["shares_1y"] = None
        if sd:
            prior = [p for p in sd.items if p.period == "Q" and _near(p.end, end - YEAR)]
            row["shares_1y"] = _rescale(prior[-1].value, ref)[0] if prior else None
        out.append(row)
    return pd.DataFrame(out).set_index("cik") if out else pd.DataFrame(columns=["fiscal_end"])


__all__ = ["Period", "Periods", "clean", "point_in_time", "sec", "snapshot", "ttm"]
