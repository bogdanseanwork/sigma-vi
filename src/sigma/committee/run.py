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
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd

from sigma.ai.router import AllModelsFailed
from sigma.committee import evidence, pipeline
from sigma.committee.personas import ROUNDS
from sigma.committee.llm import OutOfBudget, RouterAsker
from sigma.core.config import DEFAULT_DOTENV


def _log(msg: str) -> None:
    sys.stdout.write(f"[{datetime.now():%H:%M:%S}] {msg}\n")
    sys.stdout.flush()


def describe_failure(exc: AllModelsFailed) -> str:
    """Multi-line, human-readable reason a call failed on every model (one short line per attempt)."""
    lines = [f"Stopped: every model failed for task {exc.task_id}."]
    for a in exc.attempts:
        raw = a.get("detail") or a.get("error_class") or "unknown"
        reason = " ".join(str(raw).split())[:160]
        lines.append(f"  - {a.get('model', '?')}: {reason}")
    lines.append("Nothing finished is lost. Fix the cause (or wait if a provider was busy) and run the "
                 "same command again.")
    return "\n".join(lines)


STAGES = (("vote", "A  blind votes"), ("debate", "B  8-round debates"), ("red", "C  red team"),
          ("duel", "D  tournament"))


def planned_calls(n_companies: int, cfg: pipeline.Config) -> dict[str, int]:
    """Model calls each stage needs when every reply is usable (retries add a few)."""
    n_debate = min(cfg.keep_after_votes, n_companies)
    n_red = min(cfg.keep_after_debate, n_debate)
    return {"vote": n_companies * len(cfg.panel), "debate": n_debate * (len(ROUNDS) + 1), "red": n_red,
            "duel": 2 * len(pipeline._pairs([str(i) for i in range(n_red)], cfg.tournament_opponents, cfg.seed))}


def progress(states: Mapping[str, Mapping], n_companies: int, cfg: pipeline.Config) -> str:
    """How far the committee has got, from its saved checkpoints (no model calls)."""
    done = dict.fromkeys(planned_calls(n_companies, cfg), 0)
    parked = 0
    for task_id, st in states.items():
        if "#retry" in task_id:
            continue
        stage = task_id.split(":", 1)[0]
        if st.get("status") == "completed" and stage in done:
            done[stage] += 1
        elif st.get("status") == "parked":
            parked += 1
    plan = planned_calls(n_companies, cfg)
    total_done, total = sum(min(done[k], plan[k]) for k in plan), sum(plan.values())
    L = [f"SIGMA VI committee progress - {datetime.now():%Y-%m-%d %H:%M}",
         f"{n_companies} companies. Finished model calls are saved; a re-run continues from here.", ""]
    for key, label in STAGES:
        d, p = min(done[key], plan[key]), plan[key]
        L.append(f"  {label:<20} {d:>5,} / {p:<5,}  {100 * d / p if p else 100:5.1f}%")
    L += ["", f"  overall              {total_done:>5,} / {total:<5,}  {100 * total_done / total:5.1f}%"]
    if parked:
        L.append(f"  {parked} call(s) failed on every model last time; they are retried on the next run.")
    return "\n".join(L)


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
    from sigma.screen import pick_screen

    ap = argparse.ArgumentParser(prog="python -m sigma.committee.run")
    ap.add_argument("--max-calls", type=int, default=400)
    ap.add_argument("--top", type=int, default=None,
                    help="only the N best companies by quant score (fewer model calls; e.g. --top 40)")
    ap.add_argument("--status", action="store_true",
                    help="only show how far the committee has got (no model calls)")
    ap.add_argument("--screen", default=None, help="screen CSV (default: the newest full-market screen_*.csv)")
    args = ap.parse_args(argv)
    export_dotenv()
    home = DEFAULT_DOTENV.parent
    path = Path(args.screen) if args.screen else pick_screen(home, log=_log)
    screen = pd.read_csv(path)
    cands = screen[screen["stage"] == "advance"].sort_values("sigma_score", ascending=False)
    if args.top:
        cands = cands.head(args.top)
    _log(f"{len(cands)} companies from {path.name}")

    store = ParquetStore(data_dir())
    filings = store.read("filings.parquet") if store.exists("filings.parquet") else pd.DataFrame(columns=["symbol"])
    est = store.read("estimates") if store.exists("estimates") else pd.DataFrame(columns=["symbol"])
    packets = {}
    for r in cands.to_dict("records"):
        extra = "\n".join(x for x in (evidence.filings_line(filings, r["symbol"]) if len(filings) else "",
                                      evidence.estimates_line(est, r["symbol"]) if len(est) else "") if x)
        packets[r["symbol"]] = evidence.packet(r, extra)

    checkpoints = Store(store.path("committee", "checkpoints.jsonl"))
    cfg = pipeline.DEFAULT

    def save_progress() -> None:
        text = progress(checkpoints.states(), len(cands), cfg)
        (home / "committee_progress.txt").write_text(text + "\n", encoding="utf-8")
        sys.stdout.write("\n" + text + "\n")

    if args.status:
        save_progress()
        return 0

    providers = {"ollama"} | ({"gemini"} if os.environ.get("GEMINI_API_KEY") else set())
    router = ModelRouter(ModelRegistry.from_toml(), LiteLLMClient(), available_providers=providers,
                         checkpoints=checkpoints, notify=_log)
    asker = RouterAsker(router, max_calls=args.max_calls)
    try:
        res = pipeline.run_pipeline(cands[["symbol", "sigma_score"]], packets, asker, cfg, log=_log)
    except OutOfBudget as e:
        _log(f"Stopped: {e}. Run the same command again to continue; nothing finished is repeated.")
        save_progress()
        return 0
    except AllModelsFailed as e:
        _log(describe_failure(e))
        save_progress()
        return 1
    except KeyboardInterrupt:
        _log("Stopped by you (Ctrl+C). Everything finished so far is saved; run the same command to continue.")
        save_progress()
        return 130
    save_progress()
    text = render(res, cfg, asker.calls)
    (home / "committee_report.txt").write_text(text + "\n", encoding="utf-8")
    pipeline.dump(res, str(home / "committee_audit.json"))
    res["table"].head(cfg.final).to_csv(home / "final_top40.csv")
    sys.stdout.write("\n" + text + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
