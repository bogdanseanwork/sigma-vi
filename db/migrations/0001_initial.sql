-- SIGMA VI — initial schema (PostgreSQL 15+ / Neon)
--
-- Conventions
--   * Bitemporal facts: `as_of` = when the fact describes the world; `known_at` = when SIGMA VI
--     could first have known it. Point-in-time reads filter on known_at <= t.
--   * Append-only: facts, research, decisions and forecasts are never updated in place. Corrections
--     are new rows. Tables marked [IMMUTABLE] have a trigger rejecting UPDATE/DELETE.
--   * Every externally sourced row references `sources` (lineage, spec principle 17).
--   * Money in numeric; ratios/returns in double precision; timestamps in timestamptz (UTC).

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid()

-- ---------------------------------------------------------------------------------------------
-- Immutability helper
-- ---------------------------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION sigma_reject_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'table % is append-only (% rejected)', TG_TABLE_NAME, TG_OP;
END $$;

-- ---------------------------------------------------------------------------------------------
-- Lineage
-- ---------------------------------------------------------------------------------------------
CREATE TABLE sources (
  source_id       uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  provider        text NOT NULL,                 -- massive | alpha_vantage | daloopa | sec_edgar | fred | exa | alpaca | internal
  endpoint        text NOT NULL,
  locator         text,                          -- URL, SEC accession number, FRED series+vintage, ...
  source_date     timestamptz,                   -- publication time of the underlying document
  retrieved_at    timestamptz NOT NULL DEFAULT now(),
  content_hash    bytea,                         -- sha256 of the raw payload
  reliability     smallint NOT NULL DEFAULT 3 CHECK (reliability BETWEEN 1 AND 5),  -- 5 = primary regulatory
  raw_ref         text                           -- object-storage key of the raw payload
);
CREATE INDEX sources_provider_locator_idx ON sources (provider, locator);

-- Cache index (payloads live in object storage / local disk; this is the freshness ledger)
CREATE TABLE cache_entries (
  cache_key       text PRIMARY KEY,              -- provider|endpoint|normalized-params hash
  source_id       uuid REFERENCES sources,
  retrieved_at    timestamptz NOT NULL,
  expires_at      timestamptz,                   -- NULL = immutable
  content_hash    bytea NOT NULL,
  hit_count       bigint NOT NULL DEFAULT 0
);

-- ---------------------------------------------------------------------------------------------
-- Security master (survivorship-safe)
-- ---------------------------------------------------------------------------------------------
CREATE TABLE securities (
  security_id       bigserial PRIMARY KEY,
  asset_class       text NOT NULL CHECK (asset_class IN ('equity','etf','index','option','fx','crypto','rate')),
  name              text NOT NULL,
  country           text,
  currency          char(3) NOT NULL DEFAULT 'USD',
  listed_at         date,
  delisted_at       date,
  delisting_reason  text CHECK (delisting_reason IN ('merger','bankruptcy','performance','voluntary','other')),
  created_at        timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE security_identifiers (
  security_id   bigint NOT NULL REFERENCES securities,
  id_type       text NOT NULL CHECK (id_type IN ('ticker','cik','figi','cusip','isin','massive','alpha_vantage','daloopa')),
  id_value      text NOT NULL,
  valid_from    date NOT NULL,
  valid_to      date,                            -- NULL = current
  source_id     uuid REFERENCES sources,
  PRIMARY KEY (id_type, id_value, valid_from)
);
CREATE INDEX security_identifiers_sec_idx ON security_identifiers (security_id);

CREATE TABLE security_classifications (
  security_id   bigint NOT NULL REFERENCES securities,
  scheme        text NOT NULL,                   -- gics | sic | naics | internal
  sector        text,
  industry      text,
  sub_industry  text,
  valid_from    date NOT NULL,
  valid_to      date,
  PRIMARY KEY (security_id, scheme, valid_from)
);

CREATE TABLE universe_membership (
  universe      text NOT NULL,                   -- e.g. 'us_liquid_1000', 'sp500'
  security_id   bigint NOT NULL REFERENCES securities,
  valid_from    date NOT NULL,
  valid_to      date,
  known_at      timestamptz NOT NULL,
  source_id     uuid REFERENCES sources,
  PRIMARY KEY (universe, security_id, valid_from)
);

-- ---------------------------------------------------------------------------------------------
-- Market data
-- ---------------------------------------------------------------------------------------------
CREATE TABLE prices_daily (
  security_id   bigint NOT NULL REFERENCES securities,
  trade_date    date NOT NULL,
  open          numeric(20,6),
  high          numeric(20,6),
  low           numeric(20,6),
  close         numeric(20,6) NOT NULL,          -- UNADJUSTED; adjustments come from corporate_actions
  volume        bigint,
  vwap          numeric(20,6),
  source_id     uuid REFERENCES sources,
  PRIMARY KEY (security_id, trade_date)
);

CREATE TABLE corporate_actions (
  action_id     bigserial PRIMARY KEY,
  security_id   bigint NOT NULL REFERENCES securities,
  action_type   text NOT NULL CHECK (action_type IN ('split','cash_dividend','stock_dividend','spinoff','merger','delisting','symbol_change')),
  ex_date       date NOT NULL,
  ratio         double precision,                -- split ratio new/old
  cash_amount   numeric(20,6),
  delisting_return double precision,
  related_security_id bigint REFERENCES securities,
  known_at      timestamptz NOT NULL,
  source_id     uuid REFERENCES sources
);
CREATE INDEX corporate_actions_sec_date_idx ON corporate_actions (security_id, ex_date);

-- ---------------------------------------------------------------------------------------------
-- Fundamentals, estimates, filings, macro (bitemporal)
-- ---------------------------------------------------------------------------------------------
CREATE TABLE filings (
  filing_id       bigserial PRIMARY KEY,
  security_id     bigint NOT NULL REFERENCES securities,
  form_type       text NOT NULL,                 -- 10-K, 10-Q, 8-K, DEF 14A, 4, ...
  accession_no    text UNIQUE NOT NULL,
  period_end      date,
  accepted_at     timestamptz NOT NULL,          -- SEC acceptanceDateTime = known_at
  source_id       uuid REFERENCES sources
);
CREATE INDEX filings_sec_accepted_idx ON filings (security_id, accepted_at);

CREATE TABLE fundamentals (
  security_id     bigint NOT NULL REFERENCES securities,
  metric          text NOT NULL,                 -- normalized name, e.g. 'revenue', 'cfo', 'capex', 'diluted_shares'
  period_type     text NOT NULL CHECK (period_type IN ('FY','FQ','TTM','INSTANT')),
  period_end      date NOT NULL,                 -- as_of
  value           double precision NOT NULL,
  unit            text NOT NULL DEFAULT 'USD',
  segment         text NOT NULL DEFAULT '',      -- '' = consolidated
  is_gaap         boolean NOT NULL DEFAULT true,
  known_at        timestamptz NOT NULL,
  filing_id       bigint REFERENCES filings,
  source_id       uuid REFERENCES sources,
  PRIMARY KEY (security_id, metric, period_type, period_end, segment, known_at)
);
CREATE INDEX fundamentals_pit_idx ON fundamentals (security_id, metric, known_at);

-- Daily snapshots of consensus; this table IS the point-in-time estimates archive going forward.
CREATE TABLE estimates (
  security_id     bigint NOT NULL REFERENCES securities,
  metric          text NOT NULL,                 -- eps, revenue, ebitda, fcf
  target_period   text NOT NULL,                 -- 'FY2027', 'FQ2026Q4'
  statistic       text NOT NULL CHECK (statistic IN ('mean','median','high','low','count','stddev')),
  value           double precision NOT NULL,
  known_at        timestamptz NOT NULL,
  source_id       uuid REFERENCES sources,
  PRIMARY KEY (security_id, metric, target_period, statistic, known_at)
);

CREATE TABLE guidance (
  guidance_id     bigserial PRIMARY KEY,
  security_id     bigint NOT NULL REFERENCES securities,
  metric          text NOT NULL,
  target_period   text NOT NULL,
  low             double precision,
  high            double precision,
  point           double precision,
  known_at        timestamptz NOT NULL,
  source_id       uuid REFERENCES sources
);

CREATE TABLE earnings_events (
  security_id     bigint NOT NULL REFERENCES securities,
  fiscal_period   text NOT NULL,
  reported_at     timestamptz NOT NULL,
  eps_actual      double precision,
  eps_consensus   double precision,              -- consensus as known immediately before reported_at
  revenue_actual  double precision,
  revenue_consensus double precision,
  source_id       uuid REFERENCES sources,
  PRIMARY KEY (security_id, fiscal_period, reported_at)
);

CREATE TABLE macro_observations (
  series_id       text NOT NULL,                 -- FRED id
  obs_date        date NOT NULL,                 -- as_of
  value           double precision,
  vintage_date    date NOT NULL,                 -- ALFRED realtime_start = known_at
  source_id       uuid REFERENCES sources,
  PRIMARY KEY (series_id, obs_date, vintage_date)
);

-- ---------------------------------------------------------------------------------------------
-- Research, agents, tournament
-- ---------------------------------------------------------------------------------------------
CREATE TABLE research_cycles (
  cycle_id        uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  security_id     bigint REFERENCES securities,  -- NULL for macro/portfolio cycles
  trigger_event   text NOT NULL,                 -- initiation | 10-Q | earnings | fomc | price_move | ...
  started_at      timestamptz NOT NULL DEFAULT now(),
  finished_at     timestamptz,
  status          text NOT NULL DEFAULT 'running' CHECK (status IN ('running','completed','parked','failed')),
  data_snapshot_at timestamptz NOT NULL          -- PIT boundary for every agent in the cycle
);

CREATE TABLE research_tasks (                    -- checkpoint unit for model failover
  task_id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  cycle_id        uuid NOT NULL REFERENCES research_cycles,
  agent           text NOT NULL,
  round           smallint NOT NULL DEFAULT 1,   -- tournament round 1..7
  task_class      text NOT NULL,                 -- deterministic | retrieval | screen | specialist | debate | committee | capital_decision
  depends_on      uuid[] NOT NULL DEFAULT '{}',
  inputs_hash     bytea NOT NULL,
  status          text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','running','completed','failed','parked')),
  attempts        smallint NOT NULL DEFAULT 0,
  checkpoint      jsonb,                         -- partial output + citations preserved across failover
  updated_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX research_tasks_cycle_idx ON research_tasks (cycle_id, status);

CREATE TABLE agent_outputs (                     -- [IMMUTABLE]
  output_id       uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  task_id         uuid NOT NULL REFERENCES research_tasks,
  agent           text NOT NULL,
  security_id     bigint REFERENCES securities,
  payload         jsonb NOT NULL,                -- AgentOutput schema (spec §9 + facts/estimates/inferences)
  score           double precision,
  confidence      double precision CHECK (confidence BETWEEN 0 AND 1),
  input_hashes    bytea[] NOT NULL,              -- for incremental staleness detection
  auditor_status  text NOT NULL DEFAULT 'pending' CHECK (auditor_status IN ('pending','passed','failed')),
  created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE citations (
  citation_id     uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  output_id       uuid NOT NULL REFERENCES agent_outputs,
  source_id       uuid REFERENCES sources,
  calc_id         uuid,                          -- or a deterministic calculation id
  locator         text,                          -- page / section / XBRL tag
  claim_text      text NOT NULL
);

CREATE TABLE calculations (                      -- deterministic engine results, reproducible
  calc_id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  function_name   text NOT NULL,                 -- e.g. 'finance.valuation.reverse_dcf'
  code_version    text NOT NULL,                 -- git sha
  inputs          jsonb NOT NULL,
  outputs         jsonb NOT NULL,
  created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE debate_claims (
  claim_id        uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  cycle_id        uuid NOT NULL REFERENCES research_cycles,
  side            text NOT NULL CHECK (side IN ('bull','bear','red_team','specialist')),
  claim_type      text NOT NULL CHECK (claim_type IN ('factual','estimate','causal','testable')),
  text            text NOT NULL,
  hypothesis      jsonb,                         -- machine-readable for testable claims
  origin_output_id uuid REFERENCES agent_outputs,
  created_round   smallint NOT NULL
);

CREATE TABLE claim_events (                      -- status history; claims are never edited
  claim_id        uuid NOT NULL REFERENCES debate_claims,
  round           smallint NOT NULL,
  status          text NOT NULL CHECK (status IN ('open','supported','weakened','refuted','conceded','withdrawn','untestable')),
  by_agent        text NOT NULL,
  rationale       text,
  evidence_ref    uuid,                          -- citation_id, calc_id or backtest_run_id
  created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE scores (
  score_id        uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  cycle_id        uuid NOT NULL REFERENCES research_cycles,
  security_id     bigint NOT NULL REFERENCES securities,
  scheme_version  text NOT NULL,
  total           double precision NOT NULL CHECK (total BETWEEN 0 AND 100),
  categories      jsonb NOT NULL,                -- {category: {score, max, evidence[], confidence, sources[]}}
  created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE theses (
  thesis_id       uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  security_id     bigint NOT NULL REFERENCES securities,
  version         int NOT NULL,
  summary         text NOT NULL,
  variant_perception text NOT NULL,
  scenarios       jsonb NOT NULL,                -- [{name, probability, value_per_share, assumptions}]
  supersedes_id   uuid REFERENCES theses,
  created_at      timestamptz NOT NULL DEFAULT now(),
  UNIQUE (security_id, version)
);

CREATE TABLE catalysts (
  catalyst_id     uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  thesis_id       uuid NOT NULL REFERENCES theses,
  description     text NOT NULL,
  expected_window daterange,
  probability     double precision CHECK (probability BETWEEN 0 AND 1),
  resolved_at     timestamptz,
  outcome         text
);

CREATE TABLE kill_criteria (
  kill_id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  thesis_id       uuid NOT NULL REFERENCES theses,
  description     text NOT NULL,
  metric          text,                          -- machine-checkable form (optional but preferred)
  comparator      text CHECK (comparator IN ('<','<=','>','>=','==')),
  threshold       double precision,
  triggered_at    timestamptz
);

-- ---------------------------------------------------------------------------------------------
-- Decisions and outcomes (audit trail, spec §42)
-- ---------------------------------------------------------------------------------------------
CREATE TABLE decisions (                         -- [IMMUTABLE]
  decision_id     uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  decided_at      timestamptz NOT NULL DEFAULT now(),
  security_id     bigint NOT NULL REFERENCES securities,
  cycle_id        uuid NOT NULL REFERENCES research_cycles,
  thesis_id       uuid REFERENCES theses,
  recommendation  text NOT NULL CHECK (recommendation IN ('STRONG BUY','BUY','WATCH','HOLD','REDUCE','AVOID','EXIT','ADD')),
  confidence      double precision NOT NULL CHECK (confidence BETWEEN 0 AND 1),
  price_at_decision numeric(20,6) NOT NULL,
  score_id        uuid REFERENCES scores,
  expected_return double precision,
  scenario_values jsonb NOT NULL,                -- bull/base/bear values and probabilities
  assumptions     jsonb NOT NULL,
  risks           jsonb NOT NULL,
  gate_results    jsonb NOT NULL,                -- the 12 investment-gate checks
  disagreements   jsonb NOT NULL,
  judge_rationale text NOT NULL,
  models_used     jsonb NOT NULL,                -- [{agent, provider, model, version}]
  llm_cost_usd    numeric(12,4),
  data_snapshot_at timestamptz NOT NULL,
  supersedes_id   uuid REFERENCES decisions
);

CREATE TABLE decision_outcomes (                 -- [IMMUTABLE] appended at fixed horizons
  decision_id     uuid NOT NULL REFERENCES decisions,
  horizon         text NOT NULL CHECK (horizon IN ('1m','3m','6m','12m','24m','exit')),
  measured_at     timestamptz NOT NULL,
  total_return    double precision,
  benchmark_return double precision,
  scenario_realized text,                        -- which scenario the outcome most resembles
  PRIMARY KEY (decision_id, horizon)
);

-- ---------------------------------------------------------------------------------------------
-- Portfolio, orders, trades
-- ---------------------------------------------------------------------------------------------
CREATE TABLE portfolios (
  portfolio_id    bigserial PRIMARY KEY,
  name            text UNIQUE NOT NULL,
  mode            text NOT NULL CHECK (mode IN ('backtest','simulation','paper','live')),
  base_currency   char(3) NOT NULL DEFAULT 'USD',
  created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE holdings_snapshots (
  portfolio_id    bigint NOT NULL REFERENCES portfolios,
  as_of           timestamptz NOT NULL,
  security_id     bigint NOT NULL REFERENCES securities,  -- cash represented by a 'cash' security
  quantity        numeric(24,8) NOT NULL,
  cost_basis      numeric(20,6),
  market_value    numeric(20,4) NOT NULL,
  weight          double precision NOT NULL,
  PRIMARY KEY (portfolio_id, as_of, security_id)
);

CREATE TABLE nav_history (
  portfolio_id    bigint NOT NULL REFERENCES portfolios,
  as_of           date NOT NULL,
  nav             numeric(20,4) NOT NULL,
  cash            numeric(20,4) NOT NULL,
  PRIMARY KEY (portfolio_id, as_of)
);

CREATE TABLE orders (
  order_id        uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  portfolio_id    bigint NOT NULL REFERENCES portfolios,
  decision_id     uuid REFERENCES decisions,
  security_id     bigint NOT NULL REFERENCES securities,
  side            text NOT NULL CHECK (side IN ('buy','sell')),
  quantity        numeric(24,8) NOT NULL CHECK (quantity > 0),
  order_type      text NOT NULL CHECK (order_type IN ('market','limit','moc','loc')),
  limit_price     numeric(20,6),
  broker          text NOT NULL,                 -- 'alpaca_paper' | 'sim'
  broker_order_id text,
  human_approved_by text,                        -- required for any non-paper order (Stage 6)
  created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE order_events (                      -- [IMMUTABLE] pre-trade checks, submissions, fills, rejections
  order_id        uuid NOT NULL REFERENCES orders,
  event_at        timestamptz NOT NULL DEFAULT now(),
  event_type      text NOT NULL,                 -- check_passed | check_failed | submitted | partial_fill | filled | cancelled | rejected | risk_veto
  detail          jsonb NOT NULL
);
CREATE INDEX order_events_order_idx ON order_events (order_id, event_at);

CREATE TABLE trades (
  trade_id        uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  order_id        uuid REFERENCES orders,
  portfolio_id    bigint NOT NULL REFERENCES portfolios,
  security_id     bigint NOT NULL REFERENCES securities,
  executed_at     timestamptz NOT NULL,
  quantity        numeric(24,8) NOT NULL,        -- signed
  price           numeric(20,6) NOT NULL,
  commission      numeric(14,4) NOT NULL DEFAULT 0,
  est_slippage_bp double precision
);

CREATE TABLE risk_snapshots (
  portfolio_id    bigint NOT NULL REFERENCES portfolios,
  as_of           timestamptz NOT NULL,
  metrics         jsonb NOT NULL,                -- beta, vol, ES, VaR, exposures, concentration, liquidity
  stress_results  jsonb,
  breaches        jsonb NOT NULL DEFAULT '[]',
  PRIMARY KEY (portfolio_id, as_of)
);

CREATE TABLE system_flags (                      -- kill switch etc.
  flag            text PRIMARY KEY,
  value           boolean NOT NULL,
  set_by          text NOT NULL,
  set_at          timestamptz NOT NULL DEFAULT now(),
  reason          text
);
INSERT INTO system_flags (flag, value, set_by, reason)
VALUES ('trading_kill_switch', false, 'migration', 'initial'),
       ('live_trading_enabled', false, 'migration', 'Stage 1 — research only');

-- ---------------------------------------------------------------------------------------------
-- Strategies, backtests, simulations, hypothesis ledger
-- ---------------------------------------------------------------------------------------------
CREATE TABLE strategies (
  strategy_id     bigserial PRIMARY KEY,
  name            text UNIQUE NOT NULL,
  research_family text NOT NULL                  -- multiple-testing budget is shared within a family
);

CREATE TABLE strategy_versions (
  strategy_version_id bigserial PRIMARY KEY,
  strategy_id     bigint NOT NULL REFERENCES strategies,
  version         int NOT NULL,
  definition      jsonb NOT NULL,                -- rules, parameters, universe, sizing, cost preset
  code_version    text NOT NULL,
  status          text NOT NULL DEFAULT 'EXPERIMENTAL' CHECK (status IN ('EXPERIMENTAL','VALIDATED','PRODUCTION_CANDIDATE','RETIRED')),
  created_at      timestamptz NOT NULL DEFAULT now(),
  UNIQUE (strategy_id, version)
);

CREATE TABLE hypothesis_ledger (                 -- [IMMUTABLE] every evaluation ever run, spec §30
  ledger_id       bigserial PRIMARY KEY,
  research_family text NOT NULL,
  strategy_version_id bigint REFERENCES strategy_versions,
  hypothesis      jsonb NOT NULL,
  parameters      jsonb NOT NULL,
  dataset_split   text NOT NULL CHECK (dataset_split IN ('train','validation','walk_forward_test','holdout')),
  sharpe          double precision,
  n_obs           int,
  evaluated_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX hypothesis_ledger_family_idx ON hypothesis_ledger (research_family);

CREATE TABLE backtest_runs (
  backtest_run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  strategy_version_id bigint NOT NULL REFERENCES strategy_versions,
  kind            text NOT NULL CHECK (kind IN ('full','walk_forward','holdout','regime','param_sweep','randomization','cost_sensitivity')),
  start_date      date NOT NULL,
  end_date        date NOT NULL,
  cost_preset     text NOT NULL CHECK (cost_preset IN ('optimistic','base','pessimistic')),
  data_snapshot_id text NOT NULL,
  seed            bigint,
  metrics         jsonb NOT NULL,
  assumptions_active jsonb NOT NULL,             -- which data-gap mitigations were in force
  created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE simulation_runs (
  sim_run_id      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  strategy_version_id bigint NOT NULL REFERENCES strategy_versions,
  config          jsonb NOT NULL,                -- generator allocation, horizon, seeds root
  root_seed       bigint NOT NULL,
  status          text NOT NULL DEFAULT 'running' CHECK (status IN ('running','completed','failed')),
  started_at      timestamptz NOT NULL DEFAULT now(),
  finished_at     timestamptz,
  wall_seconds    double precision,
  summary         jsonb,                         -- spec §35 metrics
  convergence     jsonb,                         -- per-metric estimates at 10k/100k/500k/1M with SEs
  failure_surface jsonb
);

-- The only source of truth for "paths completed". Counted per finished chunk.
CREATE TABLE simulation_chunks (
  sim_run_id      uuid NOT NULL REFERENCES simulation_runs,
  generator       text NOT NULL CHECK (generator IN ('monte_carlo','block_bootstrap','regime','parameter_uncertainty','tail_stress','execution','adversarial')),
  chunk_index     int NOT NULL,
  seed_entropy    text NOT NULL,                 -- SeedSequence spawn key
  paths_completed int NOT NULL CHECK (paths_completed > 0),
  accumulators    jsonb NOT NULL,
  finished_at     timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (sim_run_id, generator, chunk_index)
);

CREATE VIEW simulation_path_counts AS
  SELECT sim_run_id, generator, sum(paths_completed)::bigint AS paths
  FROM simulation_chunks GROUP BY sim_run_id, generator;

-- ---------------------------------------------------------------------------------------------
-- AI operations (spec §45)
-- ---------------------------------------------------------------------------------------------
CREATE TABLE llm_calls (
  call_id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  task_id         uuid REFERENCES research_tasks,
  agent           text NOT NULL,
  task_class      text NOT NULL,
  role            text NOT NULL,
  provider        text NOT NULL,
  model           text NOT NULL,
  input_tokens    int NOT NULL DEFAULT 0,
  output_tokens   int NOT NULL DEFAULT 0,
  cached_tokens   int NOT NULL DEFAULT 0,
  latency_ms      int,
  est_cost_usd    numeric(12,6) NOT NULL DEFAULT 0,
  success         boolean NOT NULL,
  error_class     text,
  fallback_from   text,                          -- provider/model that failed before this call
  created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX llm_calls_created_idx ON llm_calls (created_at);
CREATE INDEX llm_calls_agent_idx ON llm_calls (agent, created_at);

CREATE TABLE forecasts (                         -- [IMMUTABLE] calibration, spec §40
  forecast_id     uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  decision_id     uuid REFERENCES decisions,
  agent           text,
  statement       text NOT NULL,
  probability     double precision NOT NULL CHECK (probability BETWEEN 0 AND 1),
  resolves_by     date NOT NULL,
  resolution_rule jsonb NOT NULL,
  created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE forecast_resolutions (              -- [IMMUTABLE]
  forecast_id     uuid PRIMARY KEY REFERENCES forecasts,
  outcome         boolean NOT NULL,
  resolved_at     timestamptz NOT NULL DEFAULT now(),
  evidence        jsonb
);

-- ---------------------------------------------------------------------------------------------
-- Immutability triggers
-- ---------------------------------------------------------------------------------------------
DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['agent_outputs','decisions','decision_outcomes','order_events',
                           'hypothesis_ledger','forecasts','forecast_resolutions','claim_events']
  LOOP
    EXECUTE format('CREATE TRIGGER %I_immutable BEFORE UPDATE OR DELETE ON %I
                    FOR EACH ROW EXECUTE FUNCTION sigma_reject_mutation()', t, t);
  END LOOP;
END $$;

COMMIT;
