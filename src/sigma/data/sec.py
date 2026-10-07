"""SEC EDGAR XBRL facts → point-in-time fundamentals.

Source: the nightly ``companyfacts.zip`` (every filer, every XBRL fact, each with the date it was
filed). Every *version* of a number is kept: the original 10-Q value and any later re-statement are
separate rows. ``known_at`` is the day after filing, a conservative stand-in for the acceptance time
(the bulk file has dates, not times), so a backtest can never trade on a filing the same day it
appeared.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterable, Iterator, Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any

PARSER_VERSION = 2  # bump when METRICS changes: the loader rebuilds fundamentals from the same file
PERIODIC_FORMS = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A", "10-KT", "10-QT", "20-F", "20-F/A", "40-F"})

# metric -> ordered list of (taxonomy, tag, unit). Earlier tags win when a company reports several.
METRICS: dict[str, list[tuple[str, str, str]]] = {
    "revenue": [("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax", "USD"),
                ("us-gaap", "Revenues", "USD"),
                ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax", "USD"),
                ("us-gaap", "SalesRevenueNet", "USD"),
                ("us-gaap", "SalesRevenueGoodsNet", "USD"),
                ("us-gaap", "SalesRevenueServicesNet", "USD")],
    "cost_of_revenue": [("us-gaap", "CostOfRevenue", "USD"),
                        ("us-gaap", "CostOfGoodsAndServicesSold", "USD"),
                        ("us-gaap", "CostOfGoodsSold", "USD")],
    "gross_profit": [("us-gaap", "GrossProfit", "USD")],
    "rnd": [("us-gaap", "ResearchAndDevelopmentExpense", "USD")],
    "operating_income": [("us-gaap", "OperatingIncomeLoss", "USD")],
    "interest_expense": [("us-gaap", "InterestExpense", "USD"),
                         ("us-gaap", "InterestExpenseNonoperating", "USD")],
    "pretax_income": [
        ("us-gaap", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",  # noqa: E501
         "USD"),
        ("us-gaap",
         "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
         "USD"),
    ],
    "income_tax": [("us-gaap", "IncomeTaxExpenseBenefit", "USD")],
    "net_income": [("us-gaap", "NetIncomeLoss", "USD"),
                   ("us-gaap", "ProfitLoss", "USD")],
    "eps_diluted": [("us-gaap", "EarningsPerShareDiluted", "USD/shares")],
    "shares_diluted": [("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", "shares")],
    "shares_outstanding": [("dei", "EntityCommonStockSharesOutstanding", "shares"),
                           ("us-gaap", "CommonStockSharesOutstanding", "shares")],
    "dna": [("us-gaap", "DepreciationDepletionAndAmortization", "USD"),
            ("us-gaap", "DepreciationAndAmortization", "USD")],
    "sbc": [("us-gaap", "ShareBasedCompensation", "USD"),
            ("us-gaap", "AllocatedShareBasedCompensationExpense", "USD")],
    "cfo": [("us-gaap", "NetCashProvidedByUsedInOperatingActivities", "USD"),
            ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations", "USD")],
    "capex": [("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment", "USD"),
              ("us-gaap", "PaymentsToAcquireProductiveAssets", "USD")],
    "acquisitions": [("us-gaap", "PaymentsToAcquireBusinessesNetOfCashAcquired", "USD")],
    "buybacks": [("us-gaap", "PaymentsForRepurchaseOfCommonStock", "USD")],
    "dividends_paid": [("us-gaap", "PaymentsOfDividends", "USD"),
                       ("us-gaap", "PaymentsOfDividendsCommonStock", "USD")],
    "cash": [("us-gaap", "CashAndCashEquivalentsAtCarryingValue", "USD"),
             ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "USD")],
    "short_term_investments": [("us-gaap", "ShortTermInvestments", "USD"),
                               ("us-gaap", "MarketableSecuritiesCurrent", "USD")],
    "receivables": [("us-gaap", "AccountsReceivableNetCurrent", "USD")],
    "inventory": [("us-gaap", "InventoryNet", "USD")],
    "current_assets": [("us-gaap", "AssetsCurrent", "USD")],
    "total_assets": [("us-gaap", "Assets", "USD")],
    "goodwill": [("us-gaap", "Goodwill", "USD")],
    "current_liabilities": [("us-gaap", "LiabilitiesCurrent", "USD")],
    "total_liabilities": [("us-gaap", "Liabilities", "USD")],
    "debt_current": [("us-gaap", "LongTermDebtCurrent", "USD"),
                     ("us-gaap", "DebtCurrent", "USD"),
                     ("us-gaap", "LongTermDebtAndCapitalLeaseObligationsCurrent", "USD")],
    "debt_noncurrent": [("us-gaap", "LongTermDebtNoncurrent", "USD"),
                        ("us-gaap", "LongTermDebtAndCapitalLeaseObligations", "USD"),
                        ("us-gaap", "LongTermDebt", "USD"),
                        ("us-gaap", "LongTermNotesPayable", "USD"),
                        ("us-gaap", "SeniorLongTermNotes", "USD")],
    "short_term_borrowings": [("us-gaap", "ShortTermBorrowings", "USD"),
                              ("us-gaap", "CommercialPaper", "USD")],
    "debt_total": [("us-gaap", "DebtLongtermAndShorttermCombinedAmount", "USD"),
                   ("us-gaap", "DebtInstrumentCarryingAmount", "USD")],
    "deferred_revenue": [("us-gaap", "ContractWithCustomerLiabilityCurrent", "USD"),
                         ("us-gaap", "DeferredRevenueCurrent", "USD")],
    "equity": [("us-gaap", "StockholdersEquity", "USD"),
               ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "USD")],
}

_LOOKUP: dict[tuple[str, str, str], tuple[str, int]] = {
    (tax, tag, unit): (metric, rank)
    for metric, candidates in METRICS.items()
    for rank, (tax, tag, unit) in enumerate(candidates)
}


def _d(s: str | None) -> date | None:
    return date.fromisoformat(s) if s else None


def classify_period(start: date | None, end: date) -> str:
    if start is None:
        return "I"
    days = (end - start).days + 1
    if 80 <= days <= 100:
        return "Q"
    if 170 <= days <= 190:
        return "H1"
    if 260 <= days <= 285:
        return "9M"
    if 350 <= days <= 380:
        return "FY"
    return "OTHER"


def parse_companyfacts(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten one companyfacts JSON document into rows for the mapped metrics."""
    cik = int(doc["cik"])
    rows: list[dict[str, Any]] = []
    for taxonomy, tags in doc.get("facts", {}).items():
        for tag, body in tags.items():
            for unit, facts in body.get("units", {}).items():
                hit = _LOOKUP.get((taxonomy, tag, unit))
                if hit is None:
                    continue
                metric, rank = hit
                for f in facts:
                    if f.get("form") not in PERIODIC_FORMS or f.get("val") is None:
                        continue
                    start, end, filed = _d(f.get("start")), _d(f["end"]), _d(f["filed"])
                    assert end is not None and filed is not None
                    rows.append({
                        "cik": cik, "metric": metric, "tag": tag, "tag_rank": rank, "unit": unit,
                        "start": start, "end": end, "period": classify_period(start, end),
                        "value": float(f["val"]), "fy": f.get("fy"), "fp": f.get("fp"),
                        "form": f["form"], "filed": filed, "known_at": filed + timedelta(days=1),
                        "accn": f.get("accn"),
                    })
    return rows


def as_known_on(rows: Iterable[Mapping[str, Any]], when: date, metric: str | None = None
                ) -> list[Mapping[str, Any]]:
    """Latest known value per (metric, start, end) as of ``when``; preferred tag wins ties.

    Selection order per period: only rows with known_at <= when; among those, the best-ranked tag;
    for that tag, the most recently filed version (a restatement replaces the original from the day
    it becomes public, never before).
    """
    best: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for r in rows:
        if r["known_at"] > when or (metric and r["metric"] != metric):
            continue
        key = (r["cik"], r["metric"], r["start"], r["end"])
        cur = best.get(key)
        rank = (r["tag_rank"], -r["filed"].toordinal())
        if cur is None or rank < (cur["tag_rank"], -cur["filed"].toordinal()):
            best[key] = r
    return sorted(best.values(), key=lambda r: (r["cik"], r["metric"], r["end"], r["start"] or date.min))


def parse_company_tickers(doc: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """company_tickers_exchange.json → {ticker: {cik, name, exchange}} (current listings only)."""
    idx = {name: i for i, name in enumerate(doc["fields"])}
    out = {}
    for row in doc["data"]:
        out[str(row[idx["ticker"]]).upper()] = {
            "cik": int(row[idx["cik"]]), "name": row[idx["name"]], "exchange": row[idx["exchange"]],
        }
    return out


def iter_companyfacts_zip(path: Path) -> Iterator[list[dict[str, Any]]]:  # pragma: no cover — bulk file
    """Yield parsed rows one company at a time from companyfacts.zip, without extracting it."""
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue
            with zf.open(name) as fh:
                try:
                    doc = json.load(fh)
                except json.JSONDecodeError:
                    continue
            if "cik" in doc:
                yield parse_companyfacts(doc)
