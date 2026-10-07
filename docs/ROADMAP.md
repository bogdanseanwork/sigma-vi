# SIGMA VI — Roadmap and Status

Last updated 2026-10-07. **Constraint: everything runs for $0** (see ARCHITECTURE §2). Stage on the safety ladder: **Stage 1 — research only.** No strategy is
VALIDATED; nothing trades, paper or live.

## Done

| Phase | Deliverable | Evidence |
|---|---|---|
| 1 Architecture | ARCHITECTURE, VALIDATION, PORTFOLIO_AND_RISK design docs | `docs/` |
| 1 Schema | bitemporal, append-only Postgres schema | applied on Neon `sigma-vi`; `tests/sql/schema_checks.sql` passes locally and the append-only trigger was verified on Neon |
| 1 Security | env-presence config, secret redaction, live trading off by construction | `tests/unit/test_core.py` |
| 1 Providers | bitemporal record types and protocols; source-authority ranking for Exa results | `src/sigma/providers/` |
| 1 Model router | roles from config, budgets, cost ledger, circuit breakers, checkpointed failover | `tests/unit/test_router.py`, mutation-checked |
| 3 Finance engine | returns incl. total returns, risk metrics, ratios, multiples, WACC, DCF, reverse DCF, scenarios, return decomposition | 46 unit tests, hand-derived values; reconciled with Alpha Vantage on real 2025 data to 1e-10 |

## Next, in order

1. **Phase 2 — data adapters** (EDGAR XBRL → `fundamentals`; Alpaca free daily bars for history,
   Massive EOD prices, dividends, splits;
   FRED/ALFRED; Alpha Vantage estimates; Exa document locator), each with recorded-fixture contract
   tests, plus the cache layer and repositories writing to Neon. **Start archiving daily estimate
   snapshots immediately**: point-in-time estimate history cannot be backfilled from the connected
   providers, and every day of archive makes revision signals testable sooner.
2. **Phase 7 — point-in-time backtester** (`PointInTimeView`, leakage tests, cost model presets,
   delisting handling) and nested walk-forward with purging/embargo.
3. **Phase 8 — simulation engine**: the seven generators, mergeable accumulators, path ledger in
   `simulation_chunks`, convergence report, benchmarked against the 30-minute target.
4. **Phase 4/5/6 — agents, tournament, 100-point scoring and investment gate**, running against the
   real router.
5. **Phase 9 — portfolio construction and risk** (shrinkage covariance, Black–Litterman blend,
   hard-limit veto, stress tests, kill-criteria monitor).
6. **Phase 10 — Next.js dashboard**; **Phase 11 — Alpaca paper adapter**; **Phase 12 — live
   failover drills**; **Phase 13 — hardening**.

## Needs from you

| Item | Cost | Blocks |
|---|---|---|
| Run `setup.ps1` on your PC (done once; re-run to update) | $0 | everything local |
| Keys in `.env`: FRED, Alpha Vantage, Massive, Exa, Gemini; Neon connection string; your email for SEC | $0, no card | Phase 2, Phase 4 |
| Alpaca paper account keys | $0 | Phase 11 only |

## Known limits (stated, not hidden)

- Point-in-time analyst-estimate history isn't available from the connected providers. Revision-based
  signals stay EXPERIMENTAL until SIGMA's own daily archive matures (licensing history would cost money).
- Historical index constituents aren't available; backtests use a liquidity-ranked universe built
  from point-in-time data instead.
- Free-tier data limits the universe to a focused watchlist and daily refresh. Alpha Vantage's 25
  requests/day caps estimate archiving at roughly 20 names per day.
- The ≥1M-path simulation tests robustness and tail risk *given* an edge; it cannot create evidence
  of an edge. That comes only from out-of-sample walk-forward results and paper trading.
