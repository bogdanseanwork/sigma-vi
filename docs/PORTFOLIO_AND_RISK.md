# SIGMA VI — Portfolio Construction, Risk, Paper Trading and Execution Controls

Covers spec §17–§20, §39–§41, §50–§51. All limits are defaults in `config/risk.toml`.

---

## 1. From recommendation to position

```
IC decision (BUY/STRONG BUY + confidence + scenarios)
   └─► Investment Gate (12 conditions, spec §17) — any failure → WATCH, with the failed gates listed
         └─► Expected-return model (return engine, §2)
               └─► Capital competition vs holdings, watchlist, cash, benchmark (§3)
                     └─► Optimiser proposes weights (§4)
                           └─► Risk Manager: hard limits + stress (§5) — VETO possible
                                 └─► Order generation (paper) with execution controls (§7)
```

The **Investment Gate** is code (`sigma.research.gate`). Each of the 12 conditions maps to a
measurable check (e.g. "explicit kill criteria" = at least one machine-checkable kill criterion
stored; "Research Auditor review" = auditor status `passed` on the latest research version).

## 2. Return engine

Per holding, per scenario (bear/base/bull with probabilities summing to 1):

```
r ≈ (1+g_rev)(1+Δmargin/margin)(1−Δshares/shares) − 1      # EPS growth
    + shareholder_yield                                       # dividends + net buybacks
    + (exit_multiple / entry_multiple)^(1/T) − 1              # valuation change, annualised
```

`sigma.finance.return_engine` returns both the exact multiplicative form and the additive
decomposition (revenue, margin, share count, multiple, yield) so the dashboard can show where the
expected return comes from. Probability-weighted expected return and downside (bear-case loss and
ES across scenarios) are what the optimiser consumes.

## 3. Capital competition

Every rebalance computes, for every holding and every watchlist name, the same comparable quantity:

```
risk-adjusted expected excess return = (E[r] − r_f) / (marginal contribution to portfolio risk)
```

A holding is flagged **REPLACE** when a candidate exceeds it by a configurable hurdle (default
3 pp of expected return net of round-trip costs and tax drag estimate) for two consecutive reviews,
which prevents churn on noise. "Would we buy it today with cash?" is literally evaluated: the
holding is re-run through the gate as if new.

## 4. Optimiser

Default: **robust mean-variance with shrinkage**, because raw mean-variance amplifies estimation
error.

- Covariance: Ledoit–Wolf shrinkage on 2y daily returns, blended with a factor-model covariance.
- Expected returns: scenario-weighted return engine output × confidence, shrunk toward the
  equilibrium (benchmark-implied) return — a Black–Litterman blend where SIGMA VI's views are the
  "views" and confidence sets their uncertainty.
- Objective: maximise `w·μ − λ·wᵀΣw − κ·turnover_cost(w)` subject to the limits below.
- Alternatives behind the same interface and compared in backtests: equal weight, inverse-vol,
  risk parity, fractional Kelly (≤ ¼ Kelly). Spec §19's
  `size ∝ alpha × confidence × thesis × diversification ÷ risk` is implemented as the
  "heuristic" sizer and is one of the benchmarked alternatives, not assumed best.

## 5. Risk framework

### 5.1 Hard limits (veto)

| Limit | Default |
|---|---|
| Max single position | 8% of NAV (12% for STRONG BUY with confidence ≥ 0.8) |
| Max sector | 30% |
| Max industry | 20% |
| Max single factor exposure (|z|) | 1.0 vs benchmark |
| Portfolio beta | 0.7 – 1.3 |
| Min holdings | 12 |
| Liquidity | position ≤ 10% of 20-day ADV × 5 days to exit |
| Ex-ante tracking error | ≤ 10% |
| ES(97.5%, 1d) | ≤ 3.5% of NAV |
| Correlation cluster | no cluster (ρ > 0.7) above 35% |
| Cash | 0 – 30% |

### 5.2 Soft limits (warn, require IC note)

Drawdown from peak > 15%; any position down > 25% from cost without a thesis review in 10 days;
portfolio turnover > 150%/yr.

### 5.3 Stress tests (spec §41)

Run on every proposed portfolio and nightly on the current one, both per security and aggregate:
recession, yields ±200 bp, credit spreads +300 bp, revenue growth halved, gross margin −500 bp,
competitor pricing −30%, largest customer lost, China revenue eliminated, regulatory shock,
product delay, commodity shock, USD ±10%, liquidity crisis (spreads ×3, ADV −50%),
multiple compression −30%, AI CapEx slowdown. Fundamental shocks flow through each holding's DCF
(deterministic engine); market shocks flow through factor betas. A proposal that breaches the
stress-loss limit (default −25% NAV in the worst scenario) is vetoed.

### 5.4 Kill criteria monitoring

Each thesis stores machine-checkable kill criteria (`metric`, `comparator`, `threshold`,
`source`), e.g. `gross_margin_ttm < 0.55` or `nrr < 1.10`. Data events evaluate them; a hit
creates a mandatory review task and moves the position to REDUCE pending IC.

## 6. Paper trading (Stage 5)

- Alpaca **paper** endpoint only. The broker adapter refuses to construct a live client unless the
  Stage 6/7 conditions in §7 hold.
- Every paper order references the `decision_id` that caused it. Expected outcome (scenario
  distribution, simulated 90% band) is frozen at order time; realised outcome is appended later,
  never edited.
- Tracking: live NAV vs simulated distribution; slippage vs cost model; hit rate by confidence
  bucket; Brier score; drift of factor exposures.
- Minimums before PRODUCTION CANDIDATE: see VALIDATION §8.

## 7. Execution controls (pre-trade checks, all stages)

Applied in `sigma.execution.controls` before any order leaves the system (paper included, so the
controls are proven before they matter):

max position, max trade size (default 2% NAV / 5% ADV), sector/factor/beta limits post-trade,
daily loss limit (−3% NAV halts new buys), portfolio drawdown limit (−20% halts all automation),
liquidity check, stale-price check (quote older than 60 s → reject), duplicate-order detection
(same security/side/size within 10 min), market-hours validation, limit price within X bp of
reference (slippage guard), and a global **kill switch** (DB flag + env flag; either one stops all
order flow). Every check result is logged to `order_events`.

## 8. Safety ladder (spec §50)

| Stage | Gate to enter | Mechanism |
|---|---|---|
| 1 Research | — | default |
| 2 Backtest | strategy registered | `strategy_versions` |
| 3 OOS + walk-forward | stage 2 report stored | evidence row |
| 4 ≥1M simulation | stage 3 passed gates | sim ledger ≥ 1,000,000 & converged |
| 5 Paper | VALIDATED | broker paper only |
| 6 Human-approved live recs | PRODUCTION CANDIDATE + user enables `SIGMA_LIVE_TRADING` | each order needs explicit human approval |
| 7 Constrained automation | explicit written authorisation + extra safeguards | separate flag; not built until requested |

Stages cannot be skipped: each transition checks for the previous stage's evidence rows.
