# SIGMA VI — Validation Framework

Covers spec §21–§40 and §57: point-in-time backtesting, train/validate/test discipline,
walk-forward, the ≥1,000,000-path simulation engine, statistical standards, and strategy
classification. Thresholds below are **defaults in `config/validation.toml`**, versioned, and
themselves subject to the multiple-testing ledger — changing a threshold after seeing results is
recorded as a new hypothesis, not a free edit.

The organising principle: *every layer attempts to falsify the strategy; a strategy is only as
credible as the hardest test it has survived.*

---

## 1. What each layer can and cannot tell you

| Layer | Answers | Cannot answer |
|---|---|---|
| Point-in-time backtest | Would the rules have made money with the information then available? | Whether that was luck or overfitting |
| Walk-forward / OOS | Does the edge persist on data not used to choose the rules? | Whether the future resembles the sample |
| Regime split | Is the edge concentrated in one environment? | Behaviour in regimes absent from history |
| Randomization | Could random signals with the same turnover/exposures do as well? | Economic reason for the edge |
| Factor decomposition | Is the "alpha" just known factor exposure? | Unknown factors |
| Simulation (≥1M paths) | Distribution of outcomes, tail risk and survival *given* modelled return dynamics | **Whether the strategy has an edge** — simulation inherits the edge you feed it |
| Paper trading | Does it work on genuinely unseen data with real execution? | Long-horizon reliability (sample too short) |

The last two rows matter most. A parametric Monte Carlo that draws returns from a distribution with
a positive mean will "show" a positive CAGR no matter how bad the strategy is. So the simulation
engine is fed **out-of-sample walk-forward returns** (never in-sample backtest returns) and its job
is to stress *risk and robustness*, not to prove alpha.

## 2. Point-in-time backtester

### 2.1 Rules

1. Every data access goes through `PointInTimeView(t)`, which only returns rows with
   `known_at <= t`. The engine has no other path to data; this is enforced by type, and a test suite
   feeds deliberately leaky data to prove the view rejects it.
2. Signals computed at the close of day *t* trade no earlier than the open of *t+1*
   (configurable lag; default 1 day for daily data, longer for filings ingested in batch).
3. Fundamentals become visible at SEC `acceptanceDateTime` (plus ingestion lag), not period end.
4. Macro data uses ALFRED vintages — the first-release GDP print, not today's revised number.
5. The universe at *t* is the dated membership table at *t*, including names that later delisted.
   Delisting returns are applied (use the reported delisting return; if missing, assume −30% for
   performance-related delistings and 0% for mergers, flagged as assumptions).
6. Corporate actions are applied on their ex-dates from the corporate-action table, never by using
   back-adjusted price series that embed future splits/dividends into past prices.

### 2.2 Known data gaps (stated, not hidden)

| Requirement | Available source | Gap / mitigation |
|---|---|---|
| PIT fundamentals | EDGAR XBRL (filing timestamps) | Good from ~2009 for most filers; earlier years need a commercial PIT dataset |
| PIT analyst estimates & revisions | none of the connected providers give full PIT history | **Start archiving daily estimate snapshots now**; revision signals stay EXPERIMENTAL until enough archive exists, or license I/B/E/S/FactSet history |
| Delisted securities | Alpha Vantage `LISTING_STATUS` (delisted), Massive ticker history | Delisting *returns* may be missing → assumption above |
| Historical index constituents | not in connected providers | Use a liquidity-ranked universe built from PIT data instead of "S&P 500 members", or license constituent history |
| Bid/ask history | Massive quotes (recent), not decades | Model spread from volatility/size (§6) for older periods |

A backtest report prints which of these assumptions were active. A strategy that depends on a
mitigated gap cannot exceed EXPERIMENTAL.

### 2.3 Engine

Event-driven daily loop over a vectorised core: signals and target weights are computed in batch
per rebalance date; fills, costs, cash, dividends and corporate actions are processed per day.
Outputs: daily NAV, positions, trades, costs, and the metric set in §7. Reproducible given
`(strategy_version, data_snapshot_id, seed, cost_model)`.

## 3. Train / validate / test and walk-forward

```
|------ train ------|-- val --|-gap-|-- test --|
                    |------ train ------|-- val --|-gap-|-- test --|      ← roll forward
```

- **Holdout.** The most recent 20% of history (default) is a sealed test set. It is evaluated once
  per strategy version; each evaluation is logged and counts toward the multiple-testing ledger.
- **Nested walk-forward** inside the remaining 80%: expanding or rolling train window, validation
  window for hyper-parameter selection, then an untouched test window. Default windows: 5y train /
  1y validate / 1y test, step 1y (configurable, and itself swept for robustness).
- **Purging and embargo** (López de Prado): any training observation whose label horizon overlaps
  the test window is removed, and an embargo of `max(label_horizon, 21 trading days)` follows each
  test window.
- The concatenation of test windows forms the **OOS return series** — the only series that may feed
  the simulation engine, randomization tests and the classification gates.

## 4. Statistical standards

| Test | Method | Default pass rule |
|---|---|---|
| OOS performance | Sharpe of concatenated OOS series, with Lo (2002) / Newey-West SE | Sharpe > 0 with one-sided p < 0.05 |
| Multiple testing | Ledger counts every hypothesis and parameter combination ever evaluated per research family; **Deflated Sharpe Ratio** (Bailey & López de Prado 2014) using that count | DSR ≥ 0.95 |
| Overfitting | **Probability of Backtest Overfitting** via CSCV (S=16 partitions) | PBO ≤ 0.20 |
| Signal families | Benjamini–Hochberg FDR across signals tested together | q ≤ 0.10 |
| Randomization | ≥ 1,000 shuffles each of: signal-to-security labels, entry dates (same turnover), random portfolios matched on sector/beta/size | strategy beats 95th pct of each null |
| Factor attribution | Time-series regression on market, size, value, momentum, profitability, investment, low-vol (+ sector dummies); Newey-West SE | residual alpha t-stat ≥ 2.0 *or* documented reason why factor exposure is the intended source |
| Parameter robustness | Sweep each parameter ±50% on a grid; neighbourhood performance | median neighbour Sharpe ≥ 70% of chosen point; no cliff (any neighbour < 0) |
| Regime robustness | Performance per labelled regime (§5.3) | no regime with expected Sharpe < −0.5 *and* > 15% of history; worst-regime max DD within risk limits |
| Transaction costs | optimistic / base / pessimistic cost models | positive OOS net return under **pessimistic** |
| Benchmarks | vs SPY, QQQ, sector ETF, equal-weight universe, random portfolios, simple quality, momentum, value | beats the best simple alternative on risk-adjusted terms, else prefer the simple one (spec §34) |
| Minimum track record | Bailey & López de Prado MinTRL at 95% | OOS length ≥ MinTRL |

## 5. The ≥1,000,000-path simulation engine

### 5.1 What counts as a path

A **qualifying path** is one complete simulated portfolio history over the evaluation horizon
(default 10 years, daily steps) that (a) was generated by a registered generator with a recorded
seed, (b) ran to completion through the strategy's position/cost logic, and (c) contributed all its
metrics to the run's accumulators. Failed, truncated or duplicate-seed paths do not count.
`simulation.ledger` is the single source of the count; the dashboard reads it from the DB, so
"1,000,000+ SIMULATIONS COMPLETED" can only appear when the ledger says so (spec §49).

### 5.2 Allocation (default; total 1,000,000)

| Generator | Paths | Input | What it stresses |
|---|---|---|---|
| Monte Carlo (factor model, Student-t innovations, DCC-style correlation drift) | 250,000 | OOS returns → fitted factor loadings, vol, correlation dynamics | distribution of outcomes incl. fat tails, correlation spikes |
| Stationary block bootstrap (Politis–Romano, block length auto-selected) | 200,000 | OOS daily returns of strategy + benchmark jointly | serial dependence, realistic sequences, benchmark-relative risk |
| Regime-switching (Markov chain over labelled regimes) | 150,000 | per-regime return/vol/correlation fitted on history; transition matrix | regime persistence and adverse sequencing |
| Parameter uncertainty | 150,000 | posterior/bootstrap draws of μ, σ, ρ and valuation inputs | estimation error (mean return halved, vol up, etc.) |
| Tail / stress | 100,000 | historical crisis blocks spliced in + scaled shocks (spec §41 list) | crash survival, drawdown depth, recovery time |
| Execution | 100,000 | spread, slippage, impact, delay, partial-fill distributions | how much edge survives realistic trading |
| Adversarial search | 50,000 | CMA-ES / cross-entropy search over generator parameters to *minimise* strategy utility | the failure surface (spec §37) |

Allocation is a config; any change is versioned. The minimum total is enforced by the ledger.

### 5.3 Regimes

Labelled from PIT data only: market trend (200d), realised-vol tercile, yield-curve slope sign,
Fed policy direction (FRED effective rate 6m change), CPI trend, credit spread tercile, NBER
recession flag (applied with its historical announcement lag in backtests). Named historical
windows (2000–02 tech bust, 2007–09 GFC, 2009–19 expansion, 2020 COVID shock, 2020–21 speculative
growth, 2022 inflation/rate shock) are reported separately.

### 5.4 Compute design

- **Never store paths.** Each worker generates a chunk (default 10,000 paths), runs the strategy
  logic vectorised across the chunk, and folds per-path metrics into mergeable accumulators:
  running moments, P² / t-digest quantile sketches for the reported percentiles, and exact counters
  for threshold probabilities (P[loss], P[DD>20/30/50%], P[underperform]). Per-path summary rows
  (≈20 floats) are kept for 1M paths (~160 MB) so failure modes can be mined.
- **Portfolio-level simulation.** Assets are simulated through a K-factor model plus aggregated
  idiosyncratic risk rather than N full asset paths, cutting memory from O(paths·T·N) to
  O(chunk·T·K). Full asset-level paths are used only for the execution generator on the current
  holdings.
- **Parallelism.** `multiprocessing` across chunks with independent `numpy.random.SeedSequence`
  spawned streams (reproducible, non-overlapping). Numba JIT for the path loop when available, with
  a NumPy fallback that is tested to give identical results.
- **Benchmark target:** 1M × 2,520-step paths in < 30 min on 16 cores. Measured and recorded per run.
- **No LLM calls anywhere inside the engine** (spec Phase 8). An LLM reads the final report only.

### 5.5 Convergence

For every reported statistic: estimate at 10k, 100k, 500k, 1M paths; Monte Carlo standard error
(σ/√n for means; batch-means SE over 100 batches for quantiles and probabilities); and a converged
flag when the 500k→1M change is < 1 SE and the 95% CI half-width is below a per-metric tolerance
(e.g. 10 bp for median CAGR, 0.5 pp for P[DD>30%]). Convergence tables are part of every report.
Remember: convergence proves the *estimate is stable*, not that the *model is right*.

### 5.6 Output

Per spec §35: mean/median/5/25/75/95th pct CAGR, volatility, Sharpe, Sortino, Calmar, max DD,
ES(97.5%), VaR(95/99%), P[loss], P[benchmark underperformance], P[DD>20/30/50%], recovery time,
turnover, holding period, factor exposures; plus **survival rate** (share of paths meeting the
strategy's objective: positive real CAGR, max DD within limit, not underperforming benchmark by
more than X) and the **failure surface** — the generator parameter regions where survival drops
below 50%, summarised as human-readable conditions.

## 6. Transaction-cost model

`cost = commission + ½·spread + slippage + impact` with impact = `k · σ_daily · (Q / ADV)^0.5`
(square-root law). Spread from quotes when available, else from a Corwin–Schultz / size-volatility
model. Three presets:

| Preset | k | spread multiplier | delay | ADV participation cap |
|---|---|---|---|---|
| optimistic | 0.5 | 0.75 | 0 | 10% |
| base | 1.0 | 1.0 | next open | 5% |
| pessimistic | 1.5 | 1.5 | next open + 30 min VWAP drift | 2.5% (rest unfilled, carried) |

## 7. Metric definitions

Implemented in `sigma.finance.metrics` and unit-tested against hand-computed values. Conventions:
log vs simple returns explicit in function names; annualisation factor 252 by default; Sharpe uses
excess returns over the PIT risk-free rate; Sortino uses downside deviation with MAR = 0 by
default; max drawdown on the NAV path including the starting value; Calmar = CAGR / |max DD|.

## 8. Strategy classification

| Level | Requirements | Allowed |
|---|---|---|
| EXPERIMENTAL | anything | research only |
| VALIDATED | PIT backtest with no gap-dependence; nested walk-forward; all §4 gates; ≥1M-path run with converged outputs and survival ≥ configured floor; pessimistic-cost positive | paper trading |
| PRODUCTION CANDIDATE | VALIDATED + ≥ 6 months paper trading (and ≥ 50 independent decisions) with live-vs-expected tracking inside the simulated 90% band; calibration Brier score better than the base-rate forecaster; Risk Manager and Investment Committee sign-off | human-approved live recommendations |

Promotion is a DB state transition (`strategy_versions.status`) that requires the evidence rows to
exist; it cannot be set by hand. Demotion is automatic when live behaviour leaves the expected band
for a configured period. Nothing is ever "proven".

## 9. Self-improvement loop

Changes follow spec §54 literally and are tracked as new `strategy_versions`; each version
re-enters at EXPERIMENTAL. The multiple-testing ledger is per *research family*, so an iteration on
a strategy spends from the same budget as its ancestors.
