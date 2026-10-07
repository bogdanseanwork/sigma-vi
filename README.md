# SIGMA VI

Agentic investment research, portfolio management, backtesting and paper-trading platform.

> Our estimate of future reality − market expectations = potential alpha.
> LLMs reason; deterministic code calculates; every claim is tested before capital is exposed.

**Status:** Stage 1 (research only). See [`docs/ROADMAP.md`](docs/ROADMAP.md).

**Runs for $0.** Every data source and AI model is free (free tiers, government data, or local
models via Ollama). The router refuses to call any model not marked free in `config/models.toml`.
The only money spent is on the stocks themselves.

## What exists

- **Design:** [`ARCHITECTURE`](docs/ARCHITECTURE.md) · [`VALIDATION`](docs/VALIDATION.md) ·
  [`PORTFOLIO_AND_RISK`](docs/PORTFOLIO_AND_RISK.md)
- **Database:** bitemporal, append-only schema in [`db/migrations`](db/migrations), live on Neon
- **`sigma.finance`:** returns (incl. dividend-reinvested total returns), risk metrics, ratios,
  multiples, WACC, DCF, reverse DCF, scenario analysis, return decomposition
- **`sigma.ai`:** model roles, token budgets, cost ledger, multi-provider router with checkpointed
  failover (Gemini free tier and local Ollama models via LiteLLM; free-only by default)
- **`sigma.providers`:** point-in-time data interfaces and source-authority ranking
- **`sigma.core`:** credential discovery by presence only, secret redaction

## Windows setup

From this folder: `powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1`. It installs
everything, runs the tests, creates `.env`, installs Ollama with the local models, and on later runs
updates the code without touching your keys.

## Run the tests

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                       # or: PYTHONPATH=src python -m unittest discover -s tests -t .
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f tests/sql/schema_checks.sql   # against a migrated DB
```

## Configure

Copy `.env.example` to `.env` and fill in keys. `python -m sigma.core.config` reports which
integrations are available (key presence only; values are never printed). Model ids and prices live
in `config/models.toml` — verify them before use.

Live trading is disabled by construction and requires the safety-ladder stages in
`docs/PORTFOLIO_AND_RISK.md` §8.
