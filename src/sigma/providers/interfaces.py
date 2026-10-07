"""Provider interfaces (spec §48).

Engines and agents depend only on these protocols. Each vendor gets an adapter in its own
subpackage (``sigma.providers.massive``, ``.alpha_vantage``, ``.sec_edgar``, ``.fred``,
``.daloopa``, ``.alpaca``, ``.exa``) that implements one or more of them.

Every record that describes the world carries both ``as_of`` (what period/instant it describes) and
``known_at`` (when it became public). Point-in-time consumers filter on ``known_at``; adapters that
cannot supply a true publication time must set ``known_at`` conservatively late and mark
``known_at_estimated=True`` — never early.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SourceRef:
    provider: str
    endpoint: str
    locator: str | None = None          # URL, accession number, series id + vintage
    retrieved_at: datetime | None = None
    reliability: int = 3                # 5 = primary regulatory filing, 1 = unverified web


@dataclass(frozen=True)
class SecurityRef:
    """A security as seen by a provider. Mapping to SIGMA's permanent security_id happens in data/."""

    ticker: str
    cik: str | None = None
    figi: str | None = None
    name: str | None = None
    active: bool = True
    delisted_on: date | None = None


@dataclass(frozen=True)
class PriceBar:
    ticker: str
    trade_date: date
    open: float | None
    high: float | None
    low: float | None
    close: float                        # unadjusted
    volume: int | None
    source: SourceRef


class CorporateActionType(StrEnum):
    SPLIT = "split"
    CASH_DIVIDEND = "cash_dividend"
    STOCK_DIVIDEND = "stock_dividend"
    SPINOFF = "spinoff"
    MERGER = "merger"
    DELISTING = "delisting"
    SYMBOL_CHANGE = "symbol_change"


@dataclass(frozen=True)
class CorporateAction:
    ticker: str
    action: CorporateActionType
    ex_date: date
    known_at: datetime
    ratio: float | None = None
    cash_amount: float | None = None
    source: SourceRef | None = None


@dataclass(frozen=True)
class Quote:
    ticker: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    timestamp: datetime
    source: SourceRef


@dataclass(frozen=True)
class FundamentalFact:
    ticker: str
    metric: str                         # normalized: revenue, gross_profit, operating_income, cfo, capex, ...
    period_type: str                    # FY | FQ | TTM | INSTANT
    period_end: date                    # as_of
    value: float
    known_at: datetime
    unit: str = "USD"
    segment: str = ""
    is_gaap: bool = True
    known_at_estimated: bool = False
    source: SourceRef | None = None


@dataclass(frozen=True)
class EstimateSnapshot:
    ticker: str
    metric: str                         # eps | revenue | ebitda | fcf
    target_period: str                  # FY2027, FQ2026Q4
    mean: float | None
    median: float | None
    high: float | None
    low: float | None
    count: int | None
    known_at: datetime
    source: SourceRef | None = None


@dataclass(frozen=True)
class Filing:
    ticker: str
    cik: str
    form_type: str
    accession_no: str
    period_end: date | None
    accepted_at: datetime               # SEC acceptanceDateTime — the filing's known_at
    url: str
    source: SourceRef | None = None


@dataclass(frozen=True)
class MacroObservation:
    series_id: str
    obs_date: date
    value: float | None
    vintage_date: date                  # ALFRED realtime_start
    source: SourceRef | None = None


@dataclass(frozen=True)
class ResearchDocument:
    title: str
    url: str
    published_at: datetime | None
    text: str                           # untrusted content — summarise, never execute
    source: SourceRef = field(default_factory=lambda: SourceRef("web", "search", reliability=2))


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class OrderRequest:
    ticker: str
    side: OrderSide
    quantity: float
    order_type: str = "market"          # market | limit | moc | loc
    limit_price: float | None = None
    client_order_id: str | None = None  # SIGMA order uuid, used for duplicate detection


@dataclass(frozen=True)
class BrokerPosition:
    ticker: str
    quantity: float
    avg_entry_price: float
    market_value: float


@dataclass(frozen=True)
class BrokerAccount:
    equity: float
    cash: float
    buying_power: float
    is_paper: bool


# --------------------------------------------------------------------------------------------
# Protocols
# --------------------------------------------------------------------------------------------


@runtime_checkable
class MarketDataProvider(Protocol):
    name: str

    def daily_bars(self, ticker: str, start: date, end: date) -> Sequence[PriceBar]: ...

    def corporate_actions(self, ticker: str, start: date, end: date) -> Sequence[CorporateAction]: ...

    def latest_quote(self, ticker: str) -> Quote: ...

    def securities(self, include_delisted: bool = True) -> Sequence[SecurityRef]: ...


@runtime_checkable
class FundamentalsProvider(Protocol):
    name: str

    def facts(self, ticker: str, metrics: Sequence[str] | None = None) -> Sequence[FundamentalFact]: ...


@runtime_checkable
class EstimatesProvider(Protocol):
    name: str

    def consensus(self, ticker: str) -> Sequence[EstimateSnapshot]: ...


@runtime_checkable
class FilingsProvider(Protocol):
    name: str

    def filings(
        self, ticker: str, form_types: Sequence[str] | None = None, since: date | None = None
    ) -> Sequence[Filing]: ...

    def document_text(self, filing: Filing) -> str: ...


@runtime_checkable
class MacroProvider(Protocol):
    name: str

    def observations(
        self, series_id: str, start: date, end: date, vintage: date | None = None
    ) -> Sequence[MacroObservation]: ...


@runtime_checkable
class ResearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int = 10) -> Sequence[ResearchDocument]: ...


@runtime_checkable
class BrokerProvider(Protocol):
    """Broker access. Implementations MUST refuse live endpoints unless sigma.execution authorises.

    ``is_paper`` is a read-only property reflecting the endpoint actually in use.
    """

    name: str

    @property
    def is_paper(self) -> bool: ...

    def account(self) -> BrokerAccount: ...

    def positions(self) -> Sequence[BrokerPosition]: ...

    def submit_order(self, order: OrderRequest) -> str: ...

    def cancel_order(self, broker_order_id: str) -> None: ...
