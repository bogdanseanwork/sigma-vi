"""Daily refresh for the watchlist: budget, rotation, parsing, merging. No network, no Parquet."""

import importlib.util
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from sigma.data import daily
from sigma.data.store import ParquetStore

HAS_PARQUET = importlib.util.find_spec("pyarrow") is not None


class BudgetTests(unittest.TestCase):
    def test_budget_is_enforced_and_resets_each_day(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "b.json"
            b = daily.Budget(p, limit=25)
            self.assertEqual(b.take(20, date(2026, 10, 7)), 20)
            self.assertEqual(b.take(20, date(2026, 10, 7)), 5)   # only what is left
            self.assertEqual(b.take(1, date(2026, 10, 7)), 0)
            self.assertEqual(daily.Budget(p, limit=25).take(10, date(2026, 10, 8)), 10)  # new day


class RotationTests(unittest.TestCase):
    def test_never_fetched_first_then_oldest(self):
        last = {"B": date(2026, 10, 1), "C": date(2026, 10, 5), "D": date(2026, 9, 20)}
        self.assertEqual(daily.pick_for_estimates(["A", "B", "C", "D"], last, 3), ["A", "D", "B"])

    def test_everything_fits_when_budget_allows(self):
        self.assertEqual(sorted(daily.pick_for_estimates(["A", "B"], {}, 20)), ["A", "B"])


class EstimateParsingTests(unittest.TestCase):
    def test_rows_are_stamped_with_snapshot_date_and_numbers_coerced(self):
        doc = {"symbol": "MU", "estimates": [
            {"date": "2026-11-30", "horizon": "next fiscal quarter", "eps_estimate_average": "3.21",
             "eps_estimate_analyst_count": "30", "revenue_estimate_average": "15000000000"}]}
        rows = daily.parse_estimates(doc, "MU", date(2026, 10, 7))
        self.assertEqual(rows[0]["symbol"], "MU")
        self.assertEqual(rows[0]["snapshot_date"], date(2026, 10, 7))
        self.assertAlmostEqual(rows[0]["eps_estimate_average"], 3.21)
        self.assertEqual(rows[0]["horizon"], "next fiscal quarter")

    def test_rate_limit_message_raises_instead_of_looking_like_empty_data(self):
        with self.assertRaises(daily.ProviderLimit):
            daily.parse_estimates({"Information": "We have detected your API key ... 25 requests per day"},
                                  "MU", date(2026, 10, 7))

    def test_no_estimates_is_an_empty_list(self):
        self.assertEqual(daily.parse_estimates({"symbol": "X", "estimates": []}, "X", date(2026, 10, 7)), [])


class MacroMergeTests(unittest.TestCase):
    def test_new_observations_added_and_revisions_kept_as_new_rows(self):
        old = pd.DataFrame({"series": "CPI", "date": [date(2026, 8, 1), date(2026, 9, 1)],
                            "value": [300.0, 301.0], "fetched_at": date(2026, 10, 1)})
        new = pd.DataFrame({"series": "CPI", "date": [date(2026, 8, 1), date(2026, 9, 1), date(2026, 10, 1)],
                            "value": [300.0, 301.5, 302.0], "fetched_at": date(2026, 10, 7)})
        out = daily.merge_observations(old, new)
        self.assertEqual(len(out), 4)  # unchanged Aug not duplicated; revised Sep and new Oct added
        sep = out[out.date == date(2026, 9, 1)].sort_values("fetched_at")
        self.assertEqual(list(sep.value), [301.0, 301.5])  # the old vintage is still visible

    def test_parse_observations_skips_missing_markers(self):
        doc = {"observations": [{"date": "2026-10-05", "value": "4.1"}, {"date": "2026-10-06", "value": "."}]}
        df = daily.parse_observations(doc, "DGS10", date(2026, 10, 7))
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["value"], 4.1)


class FilingsTests(unittest.TestCase):
    def test_recent_filings_are_flattened(self):
        doc = {"filings": {"recent": {"accessionNumber": ["0001-26-1", "0001-26-2"], "form": ["10-Q", "8-K"],
                                      "filingDate": ["2026-09-01", "2026-09-15"],
                                      "primaryDocument": ["a.htm", "b.htm"]}}}
        rows = daily.parse_filings(doc, "MU", 723125)
        self.assertEqual([r["form"] for r in rows], ["10-Q", "8-K"])
        self.assertEqual(rows[0]["filed"], date(2026, 9, 1))
        self.assertEqual(rows[0]["known_at"], date(2026, 9, 2))  # same point-in-time rule as fundamentals
        self.assertEqual(rows[0]["cik"], 723125)


class WatchlistTests(unittest.TestCase):
    def test_load_reads_symbols_and_roles(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "w.toml"
            p.write_text('[[stock]]\nsymbol = "MU"\n[[benchmark]]\nsymbol = "SPY"\n', encoding="utf-8")
            w = daily.load_watchlist(p)
            self.assertEqual(w.stocks, ["MU"])
            self.assertEqual(w.benchmarks, ["SPY"])
            self.assertEqual(w.all, ["MU", "SPY"])

    def test_the_shipped_watchlist_parses(self):
        w = daily.load_watchlist(Path(__file__).resolve().parents[2] / "config" / "watchlist.toml")
        self.assertGreaterEqual(len(w.stocks), 10)
        self.assertEqual(sorted(w.benchmarks), ["QQQ", "SPY"])
        self.assertEqual(len(set(w.all)), len(w.all))


class RunIsolationTests(unittest.TestCase):
    def test_one_failing_step_does_not_stop_the_others(self):
        ran = []
        def ok():
            ran.append("ok")
            return "fine"

        def boom():
            raise RuntimeError("down")

        res = daily.run_steps([("a", boom), ("b", ok)])
        self.assertEqual(ran, ["ok"])
        self.assertEqual(res[0][1], "FAILED: down")
        self.assertEqual(res[1][1], "fine")


@unittest.skipUnless(HAS_PARQUET, "needs pyarrow")
class StepTests(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.store = ParquetStore(Path(self._d.name))
        self.today = date(2026, 10, 7)

    def tearDown(self):
        self._d.cleanup()

    def estimates_fetch(self, calls):
        def fetch(path, params):
            calls.append(params["symbol"])
            return {"estimates": [{"date": "2026-12-31", "horizon": "current fiscal year",
                                   "eps_estimate_average": "1.5"}]}
        return fetch

    def test_estimates_rotate_across_days_and_respect_the_daily_budget(self):
        symbols = [f"S{i:02d}" for i in range(30)]
        budget = daily.Budget(self.store.path("b.json"))
        day1, day2 = [], []
        daily.step_estimates(self.store, self.estimates_fetch(day1), symbols, self.today, budget, lambda m: None)
        daily.step_estimates(self.store, self.estimates_fetch(day2), symbols, date(2026, 10, 8), budget,
                             lambda m: None)
        self.assertEqual(len(day1), 20)
        self.assertEqual(len(day2), 20)
        self.assertEqual(set(day1) | set(day2), set(symbols))   # everyone covered within two days
        self.assertTrue(set(symbols[20:]) <= set(day2))         # the ones missed on day 1 go first

    def test_rerunning_the_same_day_cannot_exceed_the_provider_allowance(self):
        symbols = [f"S{i:02d}" for i in range(30)]
        budget = daily.Budget(self.store.path("b.json"))
        calls = []
        for _ in range(3):
            daily.step_estimates(self.store, self.estimates_fetch(calls), symbols, self.today, budget,
                                 lambda m: None)
        self.assertLessEqual(len(calls), 25)

    def test_provider_limit_keeps_what_was_already_fetched(self):
        n = [0]

        def fetch(path, params):
            n[0] += 1
            if n[0] > 2:
                return {"Information": "rate limit"}
            return {"estimates": [{"date": "2026-12-31", "horizon": "x", "eps_estimate_average": "1"}]}
        msg = daily.step_estimates(self.store, fetch, ["A", "B", "C", "D"], self.today,
                                   daily.Budget(self.store.path("b.json")), lambda m: None)
        self.assertIn("2 of 4", msg)
        self.assertIn("rate limit", msg)  # the reason reaches the report, not just the console
        self.assertEqual(len(self.store.read("estimates")), 2)

    def test_macro_twice_adds_nothing_new(self):
        doc = {"observations": [{"date": "2026-10-05", "value": "4.1"}]}
        fetch = lambda path, params: doc  # noqa: E731
        daily.step_macro(self.store, fetch, self.today)
        msg = daily.step_macro(self.store, fetch, date(2026, 10, 8))
        self.assertIn("0 new", msg)
        self.assertEqual(len(self.store.read("macro", "DGS10.parquet")), 1)

    def test_filings_accumulate_without_duplicates_and_skip_etfs(self):
        self.store.write([{"symbol": "MU", "cik": 723125}, {"symbol": "SPY", "cik": None}], "universe.parquet")
        doc = {"filings": {"recent": {"accessionNumber": ["a1"], "form": ["10-Q"],
                                      "filingDate": ["2026-09-01"], "primaryDocument": ["x.htm"]}}}
        fetch = lambda path, params: doc  # noqa: E731
        daily.step_filings(self.store, fetch, ["MU", "SPY"])
        msg = daily.step_filings(self.store, fetch, ["MU", "SPY"])
        self.assertEqual(len(self.store.read("filings.parquet")), 1)
        self.assertIn("SPY", msg)

    def test_prices_write_raw_and_adjusted_daily_files(self):
        def fetch(p):
            return {"bars": {"MU": [{"t": "2026-10-06T04:00:00Z", "o": 1, "h": 1, "l": 1, "c": 2.0, "v": 5}]}}
        daily.step_prices(self.store, fetch, ["MU"], self.today)
        self.assertTrue(self.store.exists("prices", "raw", "daily-2026-10-07.parquet"))
        self.assertTrue(self.store.exists("prices", "all", "daily-2026-10-07.parquet"))


if __name__ == "__main__":
    unittest.main()
