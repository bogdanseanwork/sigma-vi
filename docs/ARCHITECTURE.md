# SIGMA VI — System Architecture

Status: v0.1 (Phase 1). Owner: SIGMA VI core. This document is the design of record; code that
disagrees with it is a bug in one or the other, and the fix is recorded here.

Companion documents:

- [`VALIDATION.md`](VALIDATION.md) — backtester, walk-forward, million-path simulation engine,
  statistical standards, strategy classification.
- [`PORTFOLIO_AND_RISK.md`](PORTFOLIO_AND_RISK.md) — portfolio construction, risk framework,
  paper trading, execution controls, safety ladder.
- [`ROADMAP.md`](ROADMAP.md) — phased implementation plan and current status.
- [`../db/migrations/0001_initial.sql`](../db/migrations/0001_initial.sql) — database schema.

---

## 1. Design stance

Three rules shape every decision below.

1. **LLMs reason, code calculates.** Any number that can be computed deterministically is computed
   in `sigma.finance` / `sigma.simulation` and handed to agents as data. An agent never produces a
   CAGR, a DCF value or a covariance; it receives them and argues about the *assumptions*.
2. **Append-only history.** Facts, research, decisions and forecasts are versioned, never
   overwritten. Every row carries `as_of` (when it was true in the world) and `known_at` (when
   SIGMA VI could have known it). That pair is what makes point-in-time backtests and hindsight-free
   decision audits possible.
3. **Every provider is replaceable.** Agents and engines depend on Python protocols in
   `sigma.providers.interfaces`, never on a vendor SDK.

## 2. Integration inventory (as of 2026-10-07)

| Requested | Role | Status | Notes |
|---|---|---|---|
| Massive | prices, options, corporate actions, dividends | connector live | used for the reconciliation fixture; `MASSIVE_API_KEY` for the REST adapter |
| Alpha Vantage | fundamentals, estimates, earnings, transcripts, insider, listing status incl. delisted | connector live | free key: ≤3 analytics metrics per call. **Conventions differ from ours** (see §4.4) |
| SEC EDGAR | filings, XBRL company facts | public API, no connector needed | **primary source for all financial-statement numbers**; requires `SEC_EDGAR_USER_AGENT`, ≤10 req/s |
| Exa | document discovery: IR releases, presentations, shareholder letters, KPIs, guidance | connector live | replaces Daloopa's document role (§4.5) |
| FRED | macro series incl. ALFRED vintages | connector live | backtests use vintages, never revised series |
| Neon / Postgres | durable store | **live**: project `sigma-vi` (`twilight-mountain-23433069`), db `sigma_vi`, aws-us-east-1, Postgres 17 | schema 0001 applied and verified (44 tables, 8 append-only triggers) |
| GitHub | source control | **live**: `bogdanseanwork/sigma-vi` | one commit per milestone |
| Alpaca | account, positions, paper orders | no connector in registry | direct REST adapter, paper endpoint only |
| Daloopa | — | **dropped** (no account available) | role split between EDGAR XBRL (numbers) and Exa (documents) |
| Context7, Superpowers | build-time tooling | in use | Context7 verified the LiteLLM exception mapping |
| Serena | code navigation | not available | not needed at runtime |

Missing credentials are detected at boot by `sigma.core.config` (key *presence* only — values are
never printed or logged). A provider without credentials is marked `unavailable` and skipped by
routing; nothing crashes.

**Build-environment constraint.** The cloud workspace used to write this code blocks outbound calls
except to GitHub and is currently also refusing PyPI. Live-data code paths are therefore exercised by
contract tests against recorded fixtures here, and against the real APIs on the deployment host.

## 3. Component map

```
                ┌──────────────────────── apps/web (Next.js dashboard) ───────────────────────┐
                │ Command Center · Portfolio · Pipeline · Stock · Arena · Backtest · Sim · Risk │
                └───────────────────────────────▲──────────────────────────────────────────────┘
                                                │ REST/JSON (FastAPI, read-mostly)
┌───────────────────────────────────────────────┴───────────────────────────────────────────────┐
│ apps/worker  — event bus consumer, schedules research cycles, runs engines                      │
│                                                                                                 │
│  research.orchestration ──► agents.* (LLM) ──► research.debate (tournament) ──► research.scoring │
│          │                       ▲   │                                                          │
│          │       compressed JSON │   │ requests numbers                                         │
│          ▼                       │   ▼                                                          │
│  ai.router (roles, budgets, fallback, ledger)      finance.* / simulation.* / backtesting.*     │
│          │                                          (deterministic, no LLM calls)               │
│          ▼                                                    │                                 │
│  LiteLLM ─► Anthropic | OpenAI | Gemini                      ▼                                 │
│                                         portfolio.* ──► risk.* (veto) ──► execution.paper       │
└───────────────────────────────┬─────────────────────────────────────────────────────────────────┘
                                │ repositories (SQLAlchemy core, no ORM magic in hot paths)
                ┌───────────────▼────────────────┐       ┌──────────────────────────────┐
                │ Postgres (Neon) — append-only  │◄──────┤ providers.* (cached, PIT-tagged) │
                └────────────────────────────────┘       └──────────────────────────────┘
```

Repository layout (Python package `sigma` under `src/`, mirroring spec §47):

```
src/sigma/
  core/          config (env, key presence), security (redaction), logging (structured, redacting)
  providers/     interfaces.py (protocols) + one subpackage per vendor
  ai/            roles, budgets, router (fallback + checkpoint), ledger (token/cost accounting)
  agents/        one module per specialist; shared AgentOutput schema
  research/      orchestration (event routing), debate (tournament), scoring (100-pt), gate
  finance/       returns, risk metrics, ratios, valuation (DCF, reverse DCF), scenarios, return engine
  portfolio/     construction, optimization, attribution
  risk/          exposures, scenarios, stress, veto rules
  backtesting/   engine, point_in_time, walk_forward, benchmarks, transaction_costs
  simulation/    monte_carlo, bootstrap, regimes, parameter_uncertainty, tail_risk, execution,
                 adversarial, convergence, ledger (path accounting)
  execution/     paper, controls  (live/ exists only as a disabled stub until Stage 7)
  data/          repositories, cache, schemas
apps/web         Next.js dashboard
apps/worker      long-running worker entrypoint
db/migrations    SQL migrations (plain SQL; Alembic wraps them on the deployment host)
tests/           unit, property and contract tests
```

## 4. Data layer

### 4.1 Bitemporal facts

Every fact table has:

- `as_of` — the period or instant the fact describes (e.g. fiscal period end, trade date).
- `known_at` — when the fact became public: SEC `acceptanceDateTime`, earnings release timestamp,
  vendor first-seen time, FRED/ALFRED vintage date.
- `source_id` → `sources` row (provider, URL/accession, retrieval time, content hash).

A point-in-time read is `WHERE known_at <= :t` with the latest `known_at` per key winning. Restated
financials are new rows with a later `known_at`; the original stays. This one rule handles
restatements, estimate revisions and macro data revisions uniformly.

### 4.2 Security master and survivorship

`securities` is keyed by a permanent internal `security_id`, never by ticker. Tickers, names,
exchanges and CIKs live in `security_identifiers` with validity ranges, so ticker reuse
(e.g. a symbol recycled after a delisting) cannot merge two companies. Delisted, bankrupt and
acquired securities are kept with `delisted_at` and `delisting_reason`; delisting returns are stored
in `corporate_actions`. Universe membership (index constituents, liquidity screens) is a dated table,
so a 2012 backtest sees the 2012 universe.

### 4.3 Caching and freshness

`sigma.data.cache` sits in front of every provider:

| Data | Freshness policy | Invalidation trigger |
|---|---|---|
| Filings, transcripts | immutable once retrieved | never (new filing = new key) |
| Fundamentals | until next filing for the issuer | 10-Q/10-K/8-K event |
| Estimates | 24h | estimate-revision event |
| Daily prices | until next close | market close |
| Intraday quotes | 15s (paper trading only) | — |
| Macro series | until next scheduled release | FRED release calendar |
| Agent research | until an input it cited changes | dependency hash change |

Cache keys are `(provider, endpoint, normalized params)`; values store `retrieved_at`, `source_date`
and a content hash. Agent research records the hashes of every input it consumed; when any input
hash changes, only the dependent research is marked stale (spec §11 incremental research).

### 4.4 Vendor conventions (verified 2026-10-07)

The engine was reconciled against Alpha Vantage's analytics on 2025 AAPL/SPY data
(`tests/integration/test_engine_vs_alpha_vantage.py`), matching to 1e-10 once conventions were
aligned. Alpha Vantage analytics:

- compute **total returns** (dividends reinvested). Price-only returns understated SPY's 2025 return
  by 1.4 pp, so the engine provides `total_returns` and backtests always use it;
- report **population** standard deviation (ddof=0); the engine uses sample (ddof=1) everywhere;
- report a `MAX_DRAWDOWN` that is the worst run of **consecutive** down closes, not peak-to-trough.
  It understated AAPL's 2025 drawdown (−23.0% vs the true −30.2%). SIGMA never uses vendor drawdowns
  for risk limits.

### 4.5 Fundamentals without Daloopa

| Need | Source | How |
|---|---|---|
| Statement line items, segments, share counts | EDGAR XBRL company facts | normalised into `fundamentals` with `known_at` = filing acceptance time |
| Company KPIs not in XBRL (ARR, NRR, backlog, users), guidance | Exa → full document fetch | Exa locates the filing exhibit / IR release; the full document (never search highlights, which garble tables) is parsed by an agent; every figure carries a citation |
| Source authority | `sigma.providers.source_quality` | sec.gov = 5, company domains = 4, wires/exchanges = 3, everything else = 1; lookalike domains and unofficial SEC mirrors score 1 |
| Cross-check | Research Auditor | any figure present in both an Exa-located document and XBRL must agree; disagreements are flagged, XBRL wins |

## 5. Agent system

### 5.1 Contract

Every agent is a pure function of its inputs:

```
run(task: AgentTask, ctx: MinimalContext, tools: ToolSubset) -> AgentOutput
```

`AgentOutput` (schema in `sigma.agents.schema`, stored as JSONB) follows spec §9:
`conclusion, score (0–100), confidence (0–1), key_evidence[], risks[], assumptions[],
contradictions[], citations[], follow_up_required`, plus `facts[]`, `estimates[]` and
`inferences[]` kept separate (spec principle 16). Every claim in `key_evidence` must reference a
citation id or a deterministic calculation id; the Research Auditor rejects outputs that do not.

`MinimalContext` is assembled by the orchestrator, not the agent: objective, security, the specific
data slices the role needs, and compressed summaries (≤ ~400 tokens each) of required prior
conclusions. Agents never receive raw conversation history or other agents' full outputs
(spec §7).

### 5.2 Roster and routing

| Agent | Model role | Tools | Triggered by |
|---|---|---|---|
| Fundamental | RESEARCH | fundamentals, filings, finance engine | filing, earnings, initiation |
| Industry / TAM | RESEARCH | research, macro | initiation, product/industry event |
| Competitive Intelligence | RESEARCH | research, fundamentals (peers) | initiation, competitor event |
| Moat | REASONING | research, fundamentals | initiation, quarterly |
| Management | RESEARCH | filings (DEF 14A, Form 4), transcripts | initiation, proxy, insider event |
| Earnings / Expectations | DATA | estimates, earnings | earnings, revision event |
| Valuation | DATA | finance engine (DCF, reverse DCF, multiples) | every refresh of inputs |
| Variant Perception | REASONING | outputs of Fundamental, Expectations, Valuation | after those complete |
| Macro Strategist | REASONING | macro | FOMC, CPI, payrolls, regime change |
| Geopolitical / Regulatory | RESEARCH | research | policy event |
| Technology | RESEARCH | research | product / roadmap event |
| Customer / Supplier | RESEARCH | filings (concentration notes), research | initiation, 10-K |
| Alternative Data | FAST | research | weekly |
| Market | FAST | market data | large move, earnings reaction |
| Quantitative | DATA | backtester, simulation engine | empirical-resolution requests |
| Bear / Short Seller | CRITIC | all research tools | tournament round 3 |
| Risk Manager | DATA + rules | risk engine | before any portfolio change (veto) |
| Portfolio Construction | DATA | portfolio engine | after IC decision |
| Research Auditor | CRITIC | citation store, finance engine | after every agent output |
| Investment Committee Judge | JUDGE | tournament transcript only | tournament round 7 |

Event routing (spec §43) is a static table in `sigma.research.orchestration.routes`; an event
fans out only to the agents listed for it, and each agent first asks the cache "what changed since
my last output?" before spending tokens.

### 5.3 Delegation rule as code

The orchestrator only spawns a separate agent call when the estimated tokens of the isolated call
are lower than adding the work to an existing call, or when independence is required (tournament
round 1). Single-tool lookups are executed directly by the orchestrator with no LLM.

## 6. Adversarial tournament

```
R1 Independent research ──► R2 Cross-examination ──► R3 Bull vs Bear ──► R4 Red team
        (isolated contexts)      (structured critiques)    (steelmanned)      (attacks both)
                                                                                   │
R7 Investment Committee ◄── R6 Empirical resolution ◄── R5 Rebuttal ◄──────────────┘
   (Judge, evidence-weighted)    (backtest / simulation for testable claims)
```

Mechanics:

- **Isolation in R1** is enforced by the orchestrator: an agent's context is built before any peer
  output exists in the store for that cycle.
- **Claims are objects.** Each argument is stored as a `debate_claims` row with
  `claim_type ∈ {factual, estimate, causal, testable}`, its evidence links, and a status
  (`open, supported, weakened, refuted, conceded, withdrawn, untestable`). Critiques reference claim
  ids, so the Arena view can show exactly which arguments survived and why.
- **R6 routing.** A claim tagged `testable` with a machine-readable hypothesis
  (e.g. `{"signal": "eps_revision_3m", "direction": "+", "horizon_days": 126}`) is sent to the
  Quantitative agent, which runs it through the backtester with the multiple-testing ledger. Results
  attach to the claim as evidence. LLMs do not "debate" a testable claim further.
- **Stopping rule (spec §14).** Default path is R1→R2→R5→R7. R3/R4 run for candidates that pass the
  quantitative screen and for any holding whose thesis is challenged. An extra round is allowed only
  if a material disagreement exists *and* a named piece of obtainable evidence could resolve it;
  the orchestrator caps at 2 extra rounds and records unresolved disagreements explicitly.
- **Cross-model diversity (spec §44)** for capital decisions: thesis from one provider family,
  Bear/Red team from another, Judge from a third where available. Configured in `ai.roles`.
- **Judge inputs** are the surviving claims with statuses and evidence quality scores — not raw
  agent prose and not vote counts — so majority opinion cannot dominate.

## 7. Model router

### 7.1 Roles

Application code requests a role, never a model name. `sigma.ai.roles` maps each role to an ordered
fallback chain, configured in `config/models.toml` (overridable by env):

| Role | Default chain (illustrative, configurable) | Used for |
|---|---|---|
| FAST | small Claude → small GPT → Gemini Flash | classification, extraction, routing |
| DATA | mid Claude → mid GPT → Gemini Pro | structured summarisation of data |
| RESEARCH | mid Claude → mid GPT → Gemini Pro | specialist research |
| REASONING | top Claude → top GPT → Gemini Pro | moat, variant perception, macro chains |
| CRITIC | top GPT → top Claude → Gemini Pro | bear, red team, auditor |
| JUDGE | top Claude → top GPT → Gemini Pro | investment committee |
| CODING | top Claude → top GPT | strategy-definition code generation (sandboxed) |
| FALLBACK | whatever is healthy and within budget | last resort |

### 7.2 Failover with checkpoints

A research cycle is a DAG of tasks persisted in `research_tasks`. Each task stores status, inputs
hash, partial output, citations and the model used. On a provider failure the router classifies the
error:

| Error class | Action |
|---|---|
| rate limit / 429 | backoff with jitter up to the task's latency budget, then next model in chain |
| quota / billing / auth | mark provider `unavailable` for the cooldown window; next model |
| outage / 5xx / timeout | circuit breaker opens after N failures; next model |
| context too long | re-assemble a smaller context (more compression), retry same model once, then next |
| cost ceiling would be exceeded | next cheaper model in chain if the role allows, else park the task |

Before switching, the router checkpoints: completed sibling tasks remain completed; the failed
task restarts from its persisted inputs, never from scratch for the cycle. Every fallback is
recorded in `llm_calls.fallback_from`. Implemented and tested in `sigma.ai.router`.

### 7.3 Token budgets and cost ledger

Budgets (spec §46) are enforced per task class before the call is made, using a token estimate of
the assembled context:

| Task class | Max input tok | Max output tok | Typical role |
|---|---|---|---|
| deterministic | 0 | 0 | — |
| retrieval | 0 (or 2k with FAST) | 500 | FAST |
| screen | 4k | 800 | FAST |
| specialist | 24k | 3k | RESEARCH/DATA |
| debate | 16k | 3k | REASONING/CRITIC |
| committee | 40k | 6k | JUDGE |
| capital_decision | 80k | 10k | JUDGE |

Every call writes a `llm_calls` row: agent, task class, provider, model, input/output/cached tokens,
latency, estimated cost, success, fallback. Rolling views give cost per stock, per memo, per
approved investment, per agent, cache savings and delegation savings (spec §45). Agents whose
cost/value ratio is poor are flagged in the AI Operations view (spec §53).

## 8. Security

- Secrets only via environment / secret manager; `sigma.core.security.redact` scrubs anything that
  looks like a key or bearer token from logs, exceptions, prompts and DB writes, and is tested.
- Prompts are built from typed inputs; provider keys are never part of any template.
- Live trading is off by construction: `sigma.execution` refuses any non-paper endpoint unless
  `SIGMA_LIVE_TRADING=true` **and** a signed human approval exists for the specific order
  (Stage 6), and Stage 7 automation needs a separate config flag plus the controls in
  PORTFOLIO_AND_RISK §5.
- Web-research content is untrusted data: it is summarised by agents but never executed, and
  instructions found inside fetched pages are ignored by policy in every agent prompt.

## 9. Decision audit trail

`decisions` is insert-only (a trigger rejects UPDATE/DELETE). A recommendation stores everything in
spec §42, including the IDs of the agent outputs, tournament claims and data snapshot it used.
Revisions create a new row with `supersedes_id`. Realised outcomes are written to
`decision_outcomes` at fixed horizons (1m/3m/6m/12m), which feeds forecast calibration
(Brier scores by confidence bucket) and agent performance evaluation.
