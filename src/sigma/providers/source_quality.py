"""Source authority ranking for documents found by web research (Exa).

Replaces the "source-linked values" role Daloopa would have played: numbers are taken from the most
authoritative document available and reconciled against EDGAR XBRL. The level is stored in
``sources.reliability`` (1..5) and the Research Auditor rejects quantitative claims whose best
citation is below FIRST_PARTY unless explicitly marked as third-party estimates.

Search-result highlights are never used as a source of figures — they are truncated and can garble
tables. Figures come from the full fetched document.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import IntEnum
from urllib.parse import urlparse


class Reliability(IntEnum):
    UNVERIFIED = 1     # blogs, forums, unofficial mirrors
    SECONDARY = 2      # reputable media / analysis (reserved; not auto-assigned)
    WIRE_REPRINT = 3   # verbatim copies of company releases on wires and exchanges
    FIRST_PARTY = 4    # the company's own domains (newsroom, IR site)
    REGULATORY = 5     # SEC EDGAR filings


_REGULATORY = ("sec.gov",)
_WIRES = ("businesswire.com", "prnewswire.com", "globenewswire.com", "nasdaq.com", "nyse.com")


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower().rstrip(".")


def _matches(host: str, domain: str) -> bool:
    domain = domain.lower().lstrip(".")
    return host == domain or host.endswith("." + domain)


def reliability_of(url: str, company_domains: Iterable[str] = ()) -> Reliability:
    host = _host(url)
    if any(_matches(host, d) for d in _REGULATORY):
        return Reliability.REGULATORY
    if any(_matches(host, d) for d in company_domains):
        return Reliability.FIRST_PARTY
    if any(_matches(host, d) for d in _WIRES):
        return Reliability.WIRE_REPRINT
    return Reliability.UNVERIFIED


def rank_sources(urls: Sequence[str], company_domains: Iterable[str] = ()) -> list[tuple[str, Reliability]]:
    """Most authoritative first; original order preserved within a level."""
    domains = tuple(company_domains)
    scored = [(u, reliability_of(u, domains)) for u in urls]
    return sorted(scored, key=lambda x: -x[1])
