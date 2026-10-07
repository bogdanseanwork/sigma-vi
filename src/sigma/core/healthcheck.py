"""Live connection check: one small, free call per service.

Run on the machine with internet access:

    python -m sigma.core.healthcheck

Prints OK / FAIL / SKIPPED per service and writes the same table to ``healthcheck.txt``. Output never
contains key values: errors pass through :func:`sigma.core.security.redact` and are truncated.
Quota cost: Alpha Vantage 1 of its 25 daily requests; Exa about $0.005 of the free monthly credit;
everything else is free and unmetered.
"""

from __future__ import annotations

import json
import os
import sys
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from sigma.core.config import DEFAULT_DOTENV, Integration, export_dotenv, load_settings
from sigma.core.security import redact

Probe = Callable[[], str]
_TIMEOUT = 20
_ROOT = DEFAULT_DOTENV.parent


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str   # OK | FAIL | SKIPPED
    detail: str


def run_checks(probes: Mapping[str, Probe | None], env: Mapping[str, str] | None = None) -> list[CheckResult]:
    results = []
    for name, probe in probes.items():
        if probe is None:
            results.append(CheckResult(name, "SKIPPED", "no key in .env"))
            continue
        try:
            results.append(CheckResult(name, "OK", redact(probe(), env)[:200]))
        except Exception as e:  # report every failure; never crash the whole check
            msg = redact(f"{type(e).__name__}: {e}", env)
            results.append(CheckResult(name, "FAIL", msg[:200]))
    return results


def format_results(results: list[CheckResult]) -> str:
    lines = [f"{'service':<14} {'status':<8} detail", "-" * 72]
    lines += [f"{r.name:<14} {r.status:<8} {r.detail}" for r in results]
    checked = [r for r in results if r.status != "SKIPPED"]
    passed = sum(r.status == "OK" for r in checked)
    lines += ["-" * 72, f"{passed} of {len(checked)} checks passed"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Probes (network; exercised on the user's machine)
# ---------------------------------------------------------------------------------------------
def _get_json(url: str, headers: Mapping[str, str] | None = None, data: bytes | None = None) -> Any:
    req = urllib.request.Request(url, data=data, headers=dict(headers or {}))
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:150]
        raise RuntimeError(f"HTTP {e.code}: {body}") from None


def _probes(env: Mapping[str, str]) -> dict[str, Probe | None]:  # pragma: no cover
    settings = load_settings()
    has = settings.available

    def fred() -> str:
        d = _get_json("https://api.stlouisfed.org/fred/series?series_id=GDP&file_type=json"
                      f"&api_key={env['FRED_API_KEY']}")
        return f"series GDP found: {d['seriess'][0]['title']}"

    def alpha_vantage() -> str:
        d = _get_json("https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=IBM"
                      f"&apikey={env['ALPHA_VANTAGE_API_KEY']}")
        if "Global Quote" not in d:
            raise RuntimeError(str(d)[:150])  # AV reports errors and limits as 200 + a message
        return "IBM quote received (uses 1 of 25 daily requests)"

    def massive() -> str:
        d = _get_json("https://api.massive.com/v3/reference/tickers/AAPL",
                      {"Authorization": f"Bearer {env['MASSIVE_API_KEY']}"})
        return f"ticker AAPL: {d['results']['name']}"

    def sec_edgar() -> str:
        d = _get_json("https://data.sec.gov/submissions/CIK0000320193.json",
                      {"User-Agent": env["SEC_EDGAR_USER_AGENT"]})
        return f"filings index for {d['name']} reachable"

    def exa() -> str:
        body = json.dumps({"query": "Apple investor relations", "numResults": 1}).encode()
        d = _get_json("https://api.exa.ai/search",
                      {"x-api-key": env["EXA_API_KEY"], "Content-Type": "application/json"}, body)
        return f"search returned {len(d.get('results', []))} result"

    def gemini() -> str:
        d = _get_json("https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",
                      {"x-goog-api-key": env["GEMINI_API_KEY"]})
        have = {m["name"].removeprefix("models/") for m in d.get("models", [])}
        wanted = _configured_models("gemini")
        missing = [m for m in wanted if m not in have]
        if missing:
            raise RuntimeError(f"key works, but configured models not offered: {missing}")
        return f"key works; configured models available: {', '.join(wanted)}"

    def ollama() -> str:
        base = env.get("OLLAMA_API_BASE", "http://localhost:11434").rstrip("/")
        try:
            d = _get_json(f"{base}/api/tags")
        except urllib.error.URLError:
            raise RuntimeError("Ollama isn't running - open the Ollama app, then re-run") from None
        have = {m["name"] for m in d.get("models", [])}
        wanted = _configured_models("ollama")
        missing = [m for m in wanted if m not in have and f"{m}:latest" not in have]
        if missing:
            raise RuntimeError(f"Ollama running, models not downloaded yet: {missing}")
        return f"running; models ready: {', '.join(wanted)}"

    def postgres() -> str:
        import psycopg
        with psycopg.connect(env["DATABASE_URL"], connect_timeout=_TIMEOUT) as conn:
            tables = conn.execute(
                "select count(*) from information_schema.tables where table_schema='public'"
            ).fetchone()[0]
            flag = conn.execute(
                "select value from system_flags where flag='live_trading_enabled'"
            ).fetchone()[0]
        return f"connected; {tables} tables; live_trading_enabled={flag}"

    def alpaca() -> str:
        auth = {"APCA-API-KEY-ID": env["ALPACA_API_KEY_ID"],
                "APCA-API-SECRET-KEY": env["ALPACA_API_SECRET_KEY"]}
        acct = _get_json("https://paper-api.alpaca.markets/v2/account", auth)  # paper endpoint, read-only
        bars = _get_json("https://data.alpaca.markets/v2/stocks/bars?symbols=AAPL&timeframe=1Day"
                         "&start=2016-01-04&end=2016-01-08&feed=iex&limit=5", auth)
        n = len(bars.get("bars", {}).get("AAPL", []))
        history = "price history reaches 2016" if n else "NO bars for Jan 2016 - history is shorter"
        return f"paper account {acct.get('status', '?')}; {history}"

    return {
        "fred": fred if has(Integration.FRED) else None,
        "sec_edgar": sec_edgar if has(Integration.SEC_EDGAR) else None,
        "massive": massive if has(Integration.MASSIVE) else None,
        "alpha_vantage": alpha_vantage if has(Integration.ALPHA_VANTAGE) else None,
        "exa": exa if has(Integration.EXA) else None,
        "gemini": gemini if has(Integration.GEMINI) else None,
        "ollama": ollama if has(Integration.OLLAMA) else None,
        "neon_postgres": postgres if has(Integration.POSTGRES) else None,
        "alpaca": alpaca if has(Integration.ALPACA_PAPER) else None,
    }


def _configured_models(provider: str) -> list[str]:
    with open(_ROOT / "config" / "models.toml", "rb") as fh:
        models = tomllib.load(fh)["models"].values()
    return [m["id"].split("/", 1)[1] for m in models if m["provider"] == provider]


def main() -> int:  # pragma: no cover
    export_dotenv()
    env = dict(os.environ)
    text = format_results(run_checks(_probes(env), env))
    sys.stdout.write(text + "\n")
    (_ROOT / "healthcheck.txt").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
