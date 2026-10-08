"""The investment committee: short mandates for each of the 20 spec agents, plus the red team.

Stage-A voting uses ``PANEL`` (six deliberately different viewpoints). The debate rounds, red team and
tournament judge have their own prompts in sigma.committee.pipeline.
"""

from __future__ import annotations

PERSONAS: dict[str, tuple[str, str]] = {
    "growth": ("Fundamental Growth Investor", "Durable revenue, EPS, free-cash-flow and unit-economic growth."),
    "quality": ("Quality Compounder Investor", "ROIC, reinvestment runway, recurring revenue, margins, retention, management quality."),
    "value": ("Deep Value Investor", "Extreme expectation mismatches and undervalued assets; margin of safety."),
    "garp": ("GARP Investor", "Strong growth at a reasonable price; PEG-style discipline."),
    "longshort": ("Long/Short Hedge Fund Analyst", "Variant perception versus consensus and the catalysts that close the gap."),
    "forensic": ("Short Seller / Forensic Accountant", "Assume the thesis is wrong: accruals, capitalisation, receivables, inventory, SBC, acquisitions, goodwill, adjusted EBITDA, dilution."),
    "moat": ("Competitive Advantage Specialist", "Network effects, switching costs, scale, brand, IP, regulation, proprietary data, cost advantage."),
    "industry": ("Industry Specialist", "Industry structure, cycle position, pricing power, supply and demand, regulation."),
    "management": ("Management / Capital Allocation Analyst", "Insider alignment, buybacks versus dilution, M&A record, candour, incentives."),
    "macro": ("Macro Strategist", "Rates, inflation, growth, credit, dollar and commodity sensitivity across regimes."),
    "quant": ("Quantitative Factor Analyst", "Factor exposures, crowding, and whether the score is just momentum or value in disguise."),
    "revisions": ("Earnings Revision Analyst", "Direction and breadth of estimate changes; guidance quality."),
    "catalyst": ("Catalyst / Event-Driven Analyst", "Concrete dated events and the mechanism for a re-rating."),
    "disruption": ("Technology & Disruption Analyst", "Who disrupts whom; adoption curves; obsolescence risk."),
    "altdata": ("Alternative Data Analyst", "What non-financial signals would confirm or refute the thesis."),
    "technical": ("Technical / Market Structure Analyst", "Trend, liquidity, short interest, ownership, crowding."),
    "valuation": ("Valuation Specialist", "What the price already assumes; reverse DCF; multiple versus history and peers."),
    "pm": ("Portfolio Manager", "Position sizing, correlation to the rest of the book, opportunity cost."),
    "risk": ("Risk Officer", "Drawdown, leverage, liquidity, tail risk, thesis fragility."),
    "cio": ("Chief Investment Officer", "Final judgement: which is the best use of capital for the next 24 months."),
}

PANEL = ["growth", "quality", "value", "garp", "forensic", "risk"]

ROUNDS = [
    ("Bull case", "bull", "Present the strongest possible argument for owning this stock for the next 24 months."),
    ("Bear case", "bear", "Attempt to invalidate the bull thesis with the strongest evidence you can find."),
    ("Accounting attack", "forensic", "Search for hidden financial weaknesses in the figures provided."),
    ("Competitive attack", "moat", "Explain how competitors could destroy the thesis."),
    ("Valuation attack", "valuation", "Decide whether expectations already discount the upside."),
    ("Macro attack", "macro", "Stress test the company against adverse economic regimes."),
    ("Catalyst attack", "catalyst", "Decide whether the expected re-rating has a real mechanism for occurring."),
    ("Thesis defence", "bull", "Rebut every criticism above using only the evidence provided; concede what cannot be rebutted."),
]

RED_TEAM_CHECKS = [
    "hidden correlations with the rest of the candidate list", "narrative bias", "momentum chasing",
    "valuation blindness", "factor concentration", "accounting manipulation", "adverse macro environments",
    "inflated TAM assumptions", "competitor responses", "promotional management behaviour",
    "crowded trades", "historical analogues that failed",
]
