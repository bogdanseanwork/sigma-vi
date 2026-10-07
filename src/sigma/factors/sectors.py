"""Approximate GICS-style sectors from SEC Standard Industrial Classification (SIC) codes.

SIC is what the SEC publishes for free; GICS is proprietary. The map below follows the usual SIC
ranges and splits the mixed ones (e.g. SIC 28 chemicals vs 2830-2836 pharmaceuticals). It is an
approximation and labelled as such in every report. Blank-check companies (6770) map to "Shell"
so screens can exclude SPACs.
"""

from __future__ import annotations

IT, HC, FIN, RE, EN = "Information Technology", "Health Care", "Financials", "Real Estate", "Energy"
MAT, IND, UT = "Materials", "Industrials", "Utilities"
CD, CS, COM = "Consumer Discretionary", "Consumer Staples", "Communication Services"

# (low, high, sector): the first matching range wins, so specific ranges come before broad ones.
_RANGES: list[tuple[int, int, str]] = [
    (6770, 6770, "Shell"),
    (6798, 6798, RE), (6500, 6553, RE), (6000, 6799, FIN),
    (2830, 2836, HC), (2840, 2844, CS), (2800, 2899, MAT),
    (1000, 1099, MAT), (1200, 1399, EN), (1400, 1499, MAT),
    (1520, 1531, CD), (1500, 1799, IND),
    (100, 999, CS), (2000, 2199, CS), (2200, 2399, CD), (2400, 2499, MAT), (2500, 2599, CD),
    (2600, 2699, MAT), (2700, 2799, COM), (2900, 2999, EN), (3000, 3099, MAT), (3100, 3199, CD),
    (3200, 3399, MAT), (3400, 3499, IND),
    (3570, 3579, IT), (3500, 3599, IND),
    (3630, 3639, CD), (3650, 3652, CD), (3660, 3679, IT), (3600, 3699, IND),
    (3710, 3716, CD), (3750, 3751, CD), (3790, 3799, CD), (3700, 3799, IND),
    (3840, 3851, HC), (3870, 3873, CD), (3800, 3899, IT),
    (3900, 3999, CD),
    (4950, 4959, IND), (4900, 4999, UT), (4800, 4899, COM), (4000, 4799, IND),
    (5120, 5122, HC), (5140, 5149, CS), (5000, 5199, IND),
    (5400, 5499, CS), (5912, 5912, CS), (5200, 5999, CD),
    (7370, 7379, IT), (7310, 7319, COM), (7300, 7399, IND),
    (7800, 7899, COM), (7000, 7999, CD),
    (8731, 8731, HC), (8000, 8099, HC), (8200, 8299, CD), (8100, 8999, IND),
]


def sector_for_sic(sic: int | None) -> str:
    if sic is None:
        return "Unknown"
    for lo, hi, name in _RANGES:
        if lo <= sic <= hi:
            return name
    return "Other"
