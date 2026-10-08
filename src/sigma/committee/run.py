"""Run the committee on the screen's stage-2 list (the top 150 by quant score).

    python -m sigma.committee.run                 # up to 400 model calls this run, then stop cleanly
    python -m sigma.committee.run --max-calls 800 # run again later: finished calls are never repeated

Free models only (Gemini free tier, then local Ollama). A full pass is roughly 2,000 calls, so it takes a
few sessions; progress is saved after every call. Writes committee_report.txt, committee_audit.json
and final_top40.csv in the project folder (rewritten each run, from whatever is finished).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd

from sigma.committee import evidence, pipeline
from sigma.committee.llm import OutOfBudget, RouterAsker
from sigma.core.config import DEFAULT_DOTENV


def _log(msg: str) -> None:
    sys.stdout.write(f"[{datetime.now():%H:%M:%S}] {msg}\n")
    sys.stdout.flush()


def render(res: dict, cfg: pipeline.Config, calls: int) -> str:
    t: pd.DataFrame = res["table"]
    L = ["SIGMA VI committee result - EXPERIMENTAL, NOT BACKTESTABLE", "=" * 78,
         f"{datetime.now():%Y-%m-%d %H:%M}. Model calls this run: {calls}. Free models only ($0).",
         "A language model cannot be tested point-in-time, so the committee only ADJUSTS the quant score:",
         "  final = quant + panel (+/-5) + debate (+/-5) + tournament (+/-5) - red team (0 to 15).",
         f"Funnel: top 150 -> {cfg.keep_after_votes} (blind votes) -> {cfg.keep_after_debate} (8-round debate) "
         f"-> red team -> tournament -> {cfg.final}.", "",
         f"{'#':>3} {'symbol':<7} {'final':>6} {'quant':>6} {'panel':>6} {'debate':>7} {'tourn':>6} {'red':>6} "
         f"{'agree':>6} {'24m est':>8}"]
    for i, (sym, r) in enumerate(t.head(cfg.final).iterrows(), 1):
        L.append(f"{i:>3} {sym:<7} {r.final_score:6.1f} {r.sigma_score:6.1f} {r.committee_adj:+6.1f} "
                 f"{r.debate_adj:+7.1f} {r.tournament_adj:+6.1f} {r.red_penalty:+6.1f} {r.agreement * 100:5.0f}% "
                 f"{r.exp_return_24m_pct:7.0f}%")
    L += ["", "Watchlist 41-60: " + ", ".join(res["watchlist"]) if res["watchlist"] else "Watchlist: none yet", ""]
    if res["removed_by_red_team"]:
        L.append("Removed by the red team:")
        L += [f"  {s}: {r['worst_flaw']}" for s, r in res["removed_by_red_team"].items()]
        L.append("")
    L += ["Reading this honestly",
          "  * Agents saw only SIGMA's own numbers (ratios, filings index, estimate snapshots) - not 10-K text",
          "    or news - so accounting and competitive 'attacks' work from ratios, not documents.",
          "  * Models are small/free; they can be confidently wrong. Votes are logged in committee_audit.json.",
          "  * The adjustments are judgement-based limits, not fitted weights. Treat movement of more than a",
          "    few ranks driven by the committee as a prompt to read the debate, not as evidence."]
    return "\n".join(L)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - needs local models and data
    from sigma.ai.litellm_client import LiteLLMClient
    from sigma.ai.roles import ModelRegistry
    from sigma.ai.router import ModelRouter
    from sigma.committee.llm import FileCheckpointStore as Store
    from sigma.core.config import export_dotenv
    from sigma.data.paths import data_dir
    from sigma.data.store import ParquetStore

    ap = argparse.ArgumentParser(prog="python -m sigma.committee.run")
    ap.add_argument("--max-calls", type=int, default=400)
    ap.add_argument("--screen", default=None, help="screen CSV (default: the newest screen_*.csv)")
    args = ap.parse_args(argv)
    export_dotenv()
    home = DEFAULT_DOTENV.parent
    path = Path(args.screen) if args.screen else max(home.glob("screen_*.csv"), key=lambda p: p.stat().st_mtime)
    screen = pd.read_csv(path)
    cands = screen[screen["stage"] == "advance"].sort_values("sigma_score", ascending=False)
    _log(f"{len(cands)} companies from {path.name}")

    store = ParquetStore(data_dir())
    filings = store.read("filings.parquet") if store.exists("filings.parquet") else pd.DataFrame(columns=["symbol"])
    est = store.read("estimates") if store.exists("estimates") else pd.DataFrame(columns=["symbol"])
    packets = {}
    for r in cands.to_dict("records"):
        extra = "\n".join(x for x in (evidence.filings_line(filings, r["symbol"]) if len(filings) else "",
                                      evidence.estimates_line(est, r["symbol"]) if len(est) else "") if x)
        packets[r["symbol"]] = evidence.packet(r, extra)

    providers = {"ollama"} | ({"gemini"} if os.environ.get("GEMINI_API_KEY") else set())
    router = ModelRouter(ModelRegistry.from_toml(), LiteLLMClient(), available_providers=providers,
                         checkpoints=Store(store.path("committee", "checkpoints.jsonl")))
    asker = RouterAsker(router, max_calls=args.max_calls)
    cfg = pipeline.DEFAULT
    try:
        res = pipeline.run_pipeline(cands[["symbol", "sigma_score"]], packets, asker, cfg, log=_log)
    except OutOfBudget as e:
        _log(f"Stopped: {e}. Run the same command again to continue; nothing finished is repeated.")
        return 0
    text = render(res, cfg, asker.calls)
    (home / "committee_report.txt").write_text(text + "\n", encoding="utf-8")
    pipeline.dump(res, str(home / "committee_audit.json"))
    res["table"].head(cfg.final).to_csv(home / "final_top40.csv")
    sys.stdout.write("\n" + text + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
