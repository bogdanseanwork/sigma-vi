"""Committee stages: blind votes -> 8-round debate -> red team -> head-to-head tournament -> final.

EXPERIMENTAL AND NOT BACKTESTABLE. A language model has read the future of every historical date, so
its judgement cannot be tested point-in-time. The committee therefore only *adjusts* the validated-as-
far-as-possible quant score, within fixed limits written down here, and every report says so:

    final = quant score
          + up to +/-5 for the blind-vote panel          (committee_adj)
          + up to +/-5 for the debate verdict            (debate_adj)
          + up to +/-5 for the tournament win rate       (tournament_adj)
          - up to 15 for the red team                    (red_penalty)

A red team verdict of "does not survive" with penalty >= 10 removes the stock outright.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from sigma.ai.budgets import TaskClass
from sigma.ai.roles import Role
from sigma.committee.llm import Asker, parse_json
from sigma.committee.personas import PANEL, PERSONAS, RED_TEAM_CHECKS, ROUNDS

COMMITTEE_MAX, DEBATE_MAX, TOURNAMENT_MAX, RED_MAX = 5.0, 5.0, 5.0, 15.0
REMOVE_PENALTY = 10.0
TRANSCRIPT_CHARS = 900
Log = Callable[[str], None]

GROUND_RULES = (
    "You are one member of an investment committee. Use ONLY the evidence packet; if it is thin, say so "
    "and lower your confidence. Do not invent numbers, news or filings. Horizon: the next 24 months, "
    "judged on risk-adjusted return. Reply with ONLY a JSON object, no other text."
)


@dataclass(frozen=True)
class Config:
    keep_after_votes: int = 100
    keep_after_debate: int = 60
    final: int = 40
    tournament_opponents: int = 5
    panel: tuple[str, ...] = tuple(PANEL)
    seed: int = 7


DEFAULT = Config()


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _num(v: Any, default: float = float("nan")) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if f == f else default


def ask_json(asker: Asker, task_id: str, agent: str, role: Role, tc: TaskClass, system: str, user: str
             ) -> dict[str, Any] | None:
    """One call, one retry with a stricter reminder; None if the model still returns no JSON."""
    out = parse_json(asker.ask(task_id, agent, role, tc, system, user))
    if out is None:
        out = parse_json(asker.ask(task_id + "#retry", agent, role, tc, system,
                                   user + "\n\nYour last reply was not valid JSON. Reply with the JSON object only."))
    return out


# ---------------------------------------------------------------------------------------------
# Stage A: independent blind votes
# ---------------------------------------------------------------------------------------------
def parse_vote(d: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not d or str(d.get("vote", "")).lower() not in ("own", "avoid"):
        return None
    conf = _num(d.get("confidence"))
    if conf != conf:
        return None
    return {"vote": str(d["vote"]).lower(), "confidence": _clip(conf, 0, 100),
            "expected_return_24m_pct": _clip(_num(d.get("expected_return_24m_pct"), 0.0), -100, 300),
            "main_disagreement": str(d.get("main_disagreement", ""))[:300],
            "evidence": [str(e)[:200] for e in (d.get("evidence") or [])][:5]}


def vote(asker: Asker, symbol: str, packet: str, persona: str) -> dict[str, Any] | None:
    title, mandate = PERSONAS[persona]
    system = (f"{GROUND_RULES}\nYou are the {title}. Your lens: {mandate}\nYou vote alone: you have not seen "
              "any other member's view.")
    user = (f"{packet}\n\nWould you own {symbol} for the next 24 months? Reply as JSON: "
            '{"vote":"own"|"avoid","confidence":0-100,"expected_return_24m_pct":number,'
            '"main_disagreement":"what would most likely make you wrong","evidence":["short fact from the packet",...]}')
    return parse_vote(ask_json(asker, f"vote:{symbol}:{persona}", f"{persona}", Role.REASONING,
                               TaskClass.SCREEN, system, user))


def run_votes(asker: Asker, packets: Mapping[str, str], panel: Sequence[str], log: Log | None = None
              ) -> pd.DataFrame:
    rows = []
    for i, (sym, pk) in enumerate(packets.items(), 1):
        for persona in panel:
            v = vote(asker, sym, pk, persona)
            if v:
                rows.append({"symbol": sym, "persona": persona, **v})
        if log and i % 10 == 0:
            log(f"  votes: {i}/{len(packets)} companies")
    return pd.DataFrame(rows, columns=["symbol", "persona", "vote", "confidence", "expected_return_24m_pct",
                                       "main_disagreement", "evidence"])


def aggregate_votes(votes: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    out = []
    for s in symbols:
        v = votes[votes["symbol"] == s] if len(votes) else votes
        if v.empty:
            out.append({"symbol": s, "committee_score": 0.0, "n_votes": 0, "agreement": 0.0,
                        "exp_return_24m_pct": float("nan")})
            continue
        signed = v["confidence"].to_numpy() / 100 * (v["vote"] == "own").map({True: 1, False: -1}).to_numpy()
        own = float((v["vote"] == "own").mean())
        out.append({"symbol": s, "committee_score": float(signed.mean()), "n_votes": len(v),
                    "agreement": abs(own - 0.5) * 2, "exp_return_24m_pct": float(v["expected_return_24m_pct"].mean())})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------
# Stage B: the eight-round debate
# ---------------------------------------------------------------------------------------------
def debate(asker: Asker, symbol: str, packet: str) -> dict[str, Any]:
    transcript: list[tuple[str, str]] = []
    for n, (name, who, instruction) in enumerate(ROUNDS, 1):
        title = PERSONAS.get(who, ("Bull analyst" if who == "bull" else "Bear analyst", ""))[0]
        role = Role.RESEARCH if who == "bull" else Role.CRITIC       # attackers use a different model family
        history = "\n".join(f"[{nm}] {tx[:TRANSCRIPT_CHARS]}" for nm, tx in transcript) or "(nothing yet)"
        system = (f"{GROUND_RULES.replace('Reply with ONLY a JSON object, no other text.', 'Be concrete and brief (under 180 words).')}"
                  f"\nYou are the {title}. Round {n} of 8: {name}.")
        user = f"{packet}\n\nDebate so far:\n{history}\n\nTask: {instruction}"
        text = asker.ask(f"debate:{symbol}:{n}", who, role, TaskClass.DEBATE, system, user)
        transcript.append((name, text.strip()))
    history = "\n".join(f"[{nm}] {tx[:TRANSCRIPT_CHARS]}" for nm, tx in transcript)
    system = (f"{GROUND_RULES}\nYou are the Chief Investment Officer judging a debate. Weigh the strongest "
              "surviving criticisms, not who wrote more.")
    user = (f"{packet}\n\nFull debate:\n{history}\n\nDecide. Reply as JSON: "
            '{"verdict":"own"|"avoid","confidence":0-100,"thesis_strength":0-100,'
            '"surviving_criticisms":["..."],"primary_disagreement":"..."}')
    v = ask_json(asker, f"debate:{symbol}:verdict", "cio", Role.JUDGE, TaskClass.COMMITTEE, system, user) or {}
    verdict = str(v.get("verdict", "")).lower()
    conf = _clip(_num(v.get("confidence"), 0.0), 0, 100)
    return {"symbol": symbol, "verdict": verdict if verdict in ("own", "avoid") else "none",
            "confidence": conf, "thesis_strength": _clip(_num(v.get("thesis_strength"), 0.0), 0, 100),
            "surviving_criticisms": [str(x)[:200] for x in (v.get("surviving_criticisms") or [])][:5],
            "primary_disagreement": str(v.get("primary_disagreement", ""))[:300],
            "transcript": transcript}


def debate_adj(d: Mapping[str, Any]) -> float:
    sign = {"own": 1.0, "avoid": -1.0}.get(d.get("verdict", ""), 0.0)
    return DEBATE_MAX * sign * d.get("confidence", 0.0) / 100


# ---------------------------------------------------------------------------------------------
# Red team and tournament
# ---------------------------------------------------------------------------------------------
def red_team(asker: Asker, symbol: str, packet: str, debate_summary: str) -> dict[str, Any]:
    system = (f"{GROUND_RULES}\nYou are the Red Team. Your only goal is to prove this recommendation wrong. "
              "Check: " + "; ".join(RED_TEAM_CHECKS) + ".")
    user = (f"{packet}\n\nCase made for owning it:\n{debate_summary}\n\nReply as JSON: "
            '{"penalty":0-15,"survives":true|false,"flags":["..."],"worst_flaw":"..."}. '
            "Penalty 0 = nothing found; 10+ = serious enough that it should not be owned.")
    d = ask_json(asker, f"red:{symbol}", "red_team", Role.CRITIC, TaskClass.DEBATE, system, user) or {}
    return {"symbol": symbol, "penalty": _clip(_num(d.get("penalty"), 0.0), 0, RED_MAX),
            "survives": bool(d.get("survives", True)), "flags": [str(f)[:160] for f in (d.get("flags") or [])][:6],
            "worst_flaw": str(d.get("worst_flaw", ""))[:300], "answered": bool(d)}


def _pairs(symbols: Sequence[str], k: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    seen: set[frozenset[str]] = set()
    out: list[tuple[str, str]] = []
    for _ in range(k):
        order = list(symbols)
        rng.shuffle(order)
        for a, b in zip(order[::2], order[1::2], strict=False):
            key = frozenset((a, b))
            if key not in seen:
                seen.add(key)
                out.append((a, b))
    return out


def _duel(asker: Asker, a: str, b: str, packets: Mapping[str, str], first: str) -> str | None:
    """Winner symbol when ``first`` is shown as Option A; None if the reply is unusable."""
    second = b if first == a else a
    system = (f"{GROUND_RULES}\nYou are the Chief Investment Officer. If you could only own one of these two "
              "stocks for the next 24 months, which gives the better probability-weighted return?")
    user = (f"OPTION A\n{packets[first]}\n\nOPTION B\n{packets[second]}\n\nReply as JSON: "
            '{"winner":"A"|"B","confidence":0-100,"reason":"..."}')
    d = ask_json(asker, f"duel:{min(a, b)}:{max(a, b)}:{first}", "cio", Role.JUDGE, TaskClass.DEBATE, system, user)
    w = str((d or {}).get("winner", "")).upper()
    return first if w == "A" else second if w == "B" else None


def tournament(asker: Asker, packets: Mapping[str, str], k: int, seed: int, log: Log | None = None
               ) -> pd.DataFrame:
    """Each stock meets ~k others; every pair is judged twice with the order swapped, and only a winner
    that wins in BOTH orders counts as a win (otherwise it is a draw), which cancels position bias."""
    score = dict.fromkeys(packets, 0.0)
    games = dict.fromkeys(packets, 0)
    pairs = _pairs(list(packets), k, seed)
    for i, (a, b) in enumerate(pairs, 1):
        w1, w2 = _duel(asker, a, b, packets, a), _duel(asker, a, b, packets, b)
        if w1 and w1 == w2:
            score[w1] += 1.0
        elif w1 or w2:
            score[a] += 0.5
            score[b] += 0.5
        else:
            continue
        games[a] += 1
        games[b] += 1
        if log and i % 25 == 0:
            log(f"  tournament: {i}/{len(pairs)} pairs")
    return pd.DataFrame({"symbol": list(packets),
                         "win_rate": [score[s] / games[s] if games[s] else 0.5 for s in packets],
                         "games": [games[s] for s in packets]})


# ---------------------------------------------------------------------------------------------
# Whole pipeline
# ---------------------------------------------------------------------------------------------
def run_pipeline(cands: pd.DataFrame, packets: Mapping[str, str], asker: Asker, cfg: Config = DEFAULT,
                 log: Log | None = None) -> dict[str, Any]:
    """``cands``: symbol, sigma_score (the quant ranking, already the funnel's stage-2 list)."""
    say = log or (lambda m: None)
    base = cands.set_index("symbol")["sigma_score"]
    say(f"Stage A: blind votes on {len(base)} companies by {len(cfg.panel)} members")
    votes = run_votes(asker, {s: packets[s] for s in base.index}, cfg.panel, say)
    agg = aggregate_votes(votes, list(base.index)).set_index("symbol")
    c_adj = agg["committee_score"] * COMMITTEE_MAX
    stage_a = (base + c_adj).sort_values(ascending=False)
    keep_a = list(stage_a.index[:cfg.keep_after_votes])

    say(f"Stage B: 8-round debate on {len(keep_a)} companies")
    debates = {}
    for i, s in enumerate(keep_a, 1):
        debates[s] = debate(asker, s, packets[s])
        if i % 5 == 0:
            say(f"  debates: {i}/{len(keep_a)}")
    d_adj = pd.Series({s: debate_adj(d) for s, d in debates.items()})
    stage_b = (base[keep_a] + c_adj[keep_a] + d_adj).sort_values(ascending=False)
    keep_b = list(stage_b.index[:cfg.keep_after_debate])

    say(f"Stage C: red team on {len(keep_b)} companies")
    reds = {}
    for s in keep_b:
        d = debates[s]
        summary = " ".join(tx[:500] for nm, tx in d["transcript"][:1] + d["transcript"][-1:])
        reds[s] = red_team(asker, s, packets[s], summary)
    removed = [s for s, r in reds.items() if (not r["survives"]) and r["penalty"] >= REMOVE_PENALTY]
    alive = [s for s in keep_b if s not in removed]

    say(f"Stage D: tournament among {len(alive)} survivors")
    tour = tournament(asker, {s: packets[s] for s in alive}, cfg.tournament_opponents, cfg.seed, say) \
        .set_index("symbol")
    t_adj = (tour["win_rate"] * 2 - 1) * TOURNAMENT_MAX
    pen = pd.Series({s: reds[s]["penalty"] for s in alive})
    final = (base[alive] + c_adj[alive] + d_adj[alive] + t_adj - pen).sort_values(ascending=False)

    table = pd.DataFrame({"final_score": final, "sigma_score": base[final.index],
                          "committee_adj": c_adj[final.index], "debate_adj": d_adj[final.index],
                          "tournament_adj": t_adj[final.index], "red_penalty": -pen[final.index],
                          "agreement": agg.loc[final.index, "agreement"],
                          "exp_return_24m_pct": agg.loc[final.index, "exp_return_24m_pct"],
                          "n_votes": agg.loc[final.index, "n_votes"]})
    table.index.name = "symbol"
    return {"table": table, "top": list(final.index[:cfg.final]),
            "watchlist": list(final.index[cfg.final:cfg.final + 20]),
            "removed_by_red_team": {s: reds[s] for s in removed}, "debates": debates, "red": reds,
            "votes": votes, "eliminated_after_votes": list(stage_a.index[cfg.keep_after_votes:]),
            "eliminated_after_debate": list(stage_b.index[cfg.keep_after_debate:])}


def dump(result: Mapping[str, Any], path: str) -> None:
    """Machine-readable audit trail: every debate round, red-team flag and vote."""
    doc = {"top": result["top"], "watchlist": result["watchlist"],
           "removed_by_red_team": result["removed_by_red_team"],
           "debates": {s: {k: v for k, v in d.items()} for s, d in result["debates"].items()},
           "red": result["red"], "votes": result["votes"].to_dict("records")}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
