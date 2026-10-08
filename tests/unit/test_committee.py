"""Committee stages with a scripted fake model: no network, fully deterministic."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from sigma.ai.roles import Role
from sigma.committee import evidence, llm, pipeline
from sigma.committee.personas import PANEL, PERSONAS, ROUNDS


class Fake:
    """Scripted model. ``liked`` symbols get 'own'; ``flawed`` get a red-team kill; everything cached."""

    def __init__(self, liked=(), flawed=(), always_a=False, bad_json_first=False):
        self.liked, self.flawed, self.always_a, self.bad = set(liked), set(flawed), always_a, bad_json_first
        self.calls, self.cache, self.prompts = [], {}, []

    def ask(self, task_id, agent, role, task_class, system, user):
        if task_id in self.cache:
            return self.cache[task_id]
        self.calls.append(task_id)
        self.prompts.append((task_id, system, user))
        kind = task_id.split(":")[0]
        sym = task_id.split(":")[1]
        if self.bad and not task_id.endswith("#retry") and kind == "vote":
            out = "I think this is interesting but cannot format JSON."
        elif kind == "vote":
            like = sym in self.liked
            out = json.dumps({"vote": "own" if like else "avoid", "confidence": 80,
                              "expected_return_24m_pct": 30 if like else -10, "main_disagreement": "x",
                              "evidence": ["a"]})
        elif kind == "debate" and task_id.endswith("verdict"):
            like = sym in self.liked
            out = json.dumps({"verdict": "own" if like else "avoid", "confidence": 70, "thesis_strength": 60,
                              "surviving_criticisms": ["c"], "primary_disagreement": "d"})
        elif kind == "debate":
            out = f"round text for {sym}"
        elif kind == "red":
            bad = sym in self.flawed
            out = json.dumps({"penalty": 12 if bad else 1, "survives": not bad, "flags": ["f"], "worst_flaw": "w"})
        elif kind == "duel":
            a, b = task_id.split(":")[1:3]
            first = task_id.split(":")[3]
            if self.always_a:
                out = json.dumps({"winner": "A", "confidence": 90, "reason": "r"})
            else:                                       # the better stock is the one the fake 'likes'
                better = a if a in self.liked else b if b in self.liked else min(a, b)
                out = json.dumps({"winner": "A" if first == better else "B", "confidence": 60, "reason": "r"})
        else:
            raise AssertionError(task_id)
        self.cache[task_id] = out
        return out


class ParseTests(unittest.TestCase):
    def test_json_is_found_in_fences_and_chatter(self):
        self.assertEqual(llm.parse_json('Sure!\n```json\n{"a": 1}\n```\nthanks'), {"a": 1})
        self.assertEqual(llm.parse_json('blah {"a": {"b": 2}} blah'), {"a": {"b": 2}})
        self.assertIsNone(llm.parse_json("no json here"))
        self.assertIsNone(llm.parse_json(""))
        self.assertIsNone(llm.parse_json("[1, 2]"))

    def test_vote_validation_rejects_garbage_and_clips_numbers(self):
        self.assertIsNone(pipeline.parse_vote({"vote": "maybe", "confidence": 50}))
        self.assertIsNone(pipeline.parse_vote({"vote": "own"}))
        v = pipeline.parse_vote({"vote": "OWN", "confidence": 250, "expected_return_24m_pct": 9999})
        self.assertEqual((v["vote"], v["confidence"], v["expected_return_24m_pct"]), ("own", 100, 300))


class FileCheckpointTests(unittest.TestCase):
    def test_survives_a_restart(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.jsonl"
            llm.FileCheckpointStore(p).save("t1", {"status": "completed", "text": "hi"})
            self.assertEqual(llm.FileCheckpointStore(p).load("t1")["text"], "hi")
            self.assertIsNone(llm.FileCheckpointStore(p).load("nope"))


class PersonaTests(unittest.TestCase):
    def test_all_twenty_agents_and_eight_rounds_exist(self):
        self.assertEqual(len(PERSONAS), 20)
        self.assertEqual(len(ROUNDS), 8)
        self.assertTrue(set(PANEL) <= set(PERSONAS))


class VoteTests(unittest.TestCase):
    def test_votes_are_blind_and_aggregate_to_a_signed_score(self):
        fake = Fake(liked={"AAA"})
        votes = pipeline.run_votes(fake, {"AAA": "pkt A", "BBB": "pkt B"}, PANEL)
        self.assertEqual(len(votes), 12)
        by_symbol = {}
        for task_id, _system, user in fake.prompts:    # every member of the panel gets the identical
            by_symbol.setdefault(task_id.split(":")[1], set()).add(user)   # evidence and nothing else
        self.assertTrue(all(len(v) == 1 for v in by_symbol.values()))
        agg = pipeline.aggregate_votes(votes, ["AAA", "BBB", "CCC"]).set_index("symbol")
        self.assertAlmostEqual(agg.loc["AAA", "committee_score"], 0.8)
        self.assertAlmostEqual(agg.loc["BBB", "committee_score"], -0.8)
        self.assertEqual(agg.loc["CCC", "n_votes"], 0)             # no votes: neutral, flagged by n_votes
        self.assertEqual(agg.loc["AAA", "agreement"], 1.0)

    def test_unparseable_replies_are_retried_once_then_dropped_not_counted_as_votes(self):
        fake = Fake(liked={"AAA"}, bad_json_first=True)
        votes = pipeline.run_votes(fake, {"AAA": "p"}, ["growth"])
        self.assertEqual(len(votes), 1)                           # the retry rescued it
        self.assertTrue(any(c.endswith("#retry") for c in fake.calls))


class DebateTests(unittest.TestCase):
    def test_eight_rounds_plus_a_verdict_and_attackers_use_the_critic_role(self):
        roles = []

        class Spy(Fake):
            def ask(self, task_id, agent, role, tc, system, user):
                roles.append((task_id, role))
                return super().ask(task_id, agent, role, tc, system, user)

        d = pipeline.debate(Spy(liked={"AAA"}), "AAA", "pkt")
        self.assertEqual(len(d["transcript"]), 8)
        self.assertEqual(d["verdict"], "own")
        by = dict(roles)
        self.assertEqual(by["debate:AAA:1"], Role.RESEARCH)       # bull case
        self.assertEqual(by["debate:AAA:2"], Role.CRITIC)         # bear case: a different model family
        self.assertAlmostEqual(pipeline.debate_adj(d), 3.5)

    def test_a_failed_verdict_is_neutral(self):
        class Mute(Fake):
            def ask(self, task_id, *a):
                return "no" if task_id.endswith("verdict") else "text"
        d = pipeline.debate(Mute(), "AAA", "pkt")
        self.assertEqual(d["verdict"], "none")
        self.assertEqual(pipeline.debate_adj(d), 0.0)


class TournamentTests(unittest.TestCase):
    def test_position_bias_cancels_to_draws(self):
        pk = {s: s for s in "ABCDEF"}
        t = pipeline.tournament(Fake(always_a=True), pk, k=3, seed=1).set_index("symbol")
        self.assertTrue((t["win_rate"] == 0.5).all())              # a model that always says 'A' decides nothing

    def test_genuinely_better_stocks_win_more(self):
        pk = {s: s for s in "ABCDEF"}
        t = pipeline.tournament(Fake(liked={"A", "B"}), pk, k=4, seed=1).set_index("symbol")
        self.assertGreater(t.loc[["A", "B"], "win_rate"].mean(), t.loc[["C", "D", "E", "F"], "win_rate"].mean())

    def test_pairs_do_not_repeat(self):
        pairs = pipeline._pairs(list("ABCDEFGH"), 5, 3)
        self.assertEqual(len(pairs), len({frozenset(p) for p in pairs}))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.syms = [f"S{i:02d}" for i in range(12)]
        self.cands = pd.DataFrame({"symbol": self.syms, "sigma_score": [99 - i for i in range(12)]})
        self.packets = {s: f"packet {s}" for s in self.syms}
        self.cfg = pipeline.Config(keep_after_votes=10, keep_after_debate=8, final=5, tournament_opponents=3)

    def test_funnel_sizes_and_adjustment_limits(self):
        fake = Fake(liked=set(self.syms[:6]))
        res = pipeline.run_pipeline(self.cands, self.packets, fake, self.cfg)
        self.assertEqual(len(res["top"]), 5)
        self.assertEqual(len(res["eliminated_after_votes"]), 2)
        self.assertEqual(len(res["eliminated_after_debate"]), 2)
        t = res["table"]
        self.assertTrue((t["committee_adj"].abs() <= 5).all())
        self.assertTrue((t["debate_adj"].abs() <= 5).all())
        self.assertTrue((t["tournament_adj"].abs() <= 5).all())
        self.assertTrue((t["red_penalty"] >= -15).all())
        self.assertTrue(set(res["top"]) <= set(self.syms[:6]) | set(self.syms[:10]))

    def test_the_committee_cannot_overturn_a_large_quant_gap(self):
        cands = self.cands.assign(sigma_score=[100] + [50] * 11)       # S00 far ahead, but disliked
        res = pipeline.run_pipeline(cands, self.packets, Fake(liked=set(self.syms[1:])), self.cfg)
        self.assertIn("S00", res["top"])

    def test_red_team_removes_flagged_stocks_outright(self):
        res = pipeline.run_pipeline(self.cands, self.packets,
                                    Fake(liked=set(self.syms), flawed={"S00"}), self.cfg)
        self.assertIn("S00", res["removed_by_red_team"])
        self.assertNotIn("S00", res["top"])

    def test_resumes_without_repeating_finished_calls(self):
        class Capped(Fake):
            def ask(self, task_id, *a):
                if len(self.calls) >= 40 and task_id not in self.cache:
                    raise llm.OutOfBudget("cap")
                return super().ask(task_id, *a)

        fake = Capped(liked=set(self.syms[:6]))
        with self.assertRaises(llm.OutOfBudget):
            pipeline.run_pipeline(self.cands, self.packets, fake, self.cfg)
        done = len(fake.calls)
        fake.__class__ = Fake                                           # lift the cap; keep the cache
        res = pipeline.run_pipeline(self.cands, self.packets, fake, self.cfg)
        self.assertEqual(len(fake.calls), len(set(fake.calls)))          # nothing was asked twice
        self.assertGreater(len(fake.calls), done)
        self.assertEqual(len(res["top"]), 5)


class ReportTests(unittest.TestCase):
    def test_report_renders_and_states_the_limits(self):
        from sigma.committee import run
        syms = [f"S{i:02d}" for i in range(12)]
        cfg = pipeline.Config(keep_after_votes=10, keep_after_debate=8, final=5, tournament_opponents=3)
        res = pipeline.run_pipeline(pd.DataFrame({"symbol": syms, "sigma_score": range(99, 87, -1)}),
                                    {s: s for s in syms}, Fake(liked=set(syms[:6])), cfg)
        text = run.render(res, cfg, calls=123)
        self.assertIn("NOT BACKTESTABLE", text)
        self.assertIn("Model calls this run: 123", text)
        import re
        self.assertEqual(sum(1 for line in text.splitlines() if re.match(r"\s*\d+ S\d\d ", line)), 5)


class EvidenceTests(unittest.TestCase):
    def test_packet_has_facts_and_lists_gaps_without_inventing(self):
        row = {"symbol": "XYZ", "name": "Xyz Corp", "exchange": "NYSE", "sector": "Energy", "market_cap": 5e9,
               "price": 10.0, "adv_usd": 2e7, "sigma_score": 91.0, "coverage": 0.9, "cat_Growth": 1.2,
               "rev_growth": 0.25, "roic": float("nan")}
        text = evidence.packet(row, extra="recent filings: 10-Q 2026-09-01")
        self.assertIn("revenue growth (log, 1y) 25.0%", text)
        self.assertIn("ROIC", text.split("not available:")[1])
        self.assertNotIn("ROIC nan", text)
        self.assertIn("recent filings", text)
        self.assertIn("not scored", text)


if __name__ == "__main__":
    unittest.main()
