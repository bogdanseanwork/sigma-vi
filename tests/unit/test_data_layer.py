"""Data layer: rate limiting, SEC parsing (point-in-time), Alpaca parsing, universe rules."""

import json
import unittest
from datetime import date
from pathlib import Path

from sigma.data import alpaca, sec, universe
from sigma.data.http import RateLimiter
from sigma.data.paths import data_dir

FIX = Path(__file__).resolve().parents[1] / "fixtures"


def load(rel):
    return json.loads((FIX / rel).read_text())


class RateLimiterTests(unittest.TestCase):
    def test_spaces_calls_to_the_configured_rate(self):
        now = [0.0]
        slept = []

        def sleep(s):
            slept.append(s)
            now[0] += s

        rl = RateLimiter(per_minute=120, clock=lambda: now[0], sleep=sleep)  # one call per 0.5 s
        for _ in range(3):
            rl.wait()
        self.assertEqual(slept, [0.5, 0.5])


class DataDirTests(unittest.TestCase):
    def test_env_override(self):
        self.assertEqual(data_dir({"SIGMA_DATA_DIR": "/tmp/x"}), Path("/tmp/x"))

    def test_windows_default_is_outside_onedrive(self):
        p = data_dir({"LOCALAPPDATA": r"C:\Users\me\AppData\Local"}, platform="win32")
        self.assertIn("AppData", str(p))
        self.assertNotIn("OneDrive", str(p))


class SecParsingTests(unittest.TestCase):
    def setUp(self):
        self.rows = sec.parse_companyfacts(load("sec/CIK0000320193.json"))

    def test_only_mapped_tags_and_periodic_forms(self):
        self.assertTrue(all(r["form"] in sec.PERIODIC_FORMS for r in self.rows))
        self.assertNotIn("SomeUnmappedTag", {r["tag"] for r in self.rows})
        self.assertNotIn("8-K", {r["form"] for r in self.rows})

    def test_every_version_is_kept_with_its_filing_date(self):
        fy18 = [r for r in self.rows if r["metric"] == "revenue" and r["end"] == date(2018, 9, 29)
                and r["period"] == "FY"]
        self.assertEqual(sorted(r["filed"] for r in fy18), [date(2019, 10, 31), date(2020, 10, 30)])

    def test_known_at_is_day_after_filing(self):
        r = next(r for r in self.rows if r["accn"] == "0000320193-19-000010")
        self.assertEqual(r["known_at"], date(2019, 1, 31))

    def test_period_classification(self):
        kinds = {(r["start"], r["end"]): r["period"] for r in self.rows if r["metric"] == "revenue"}
        self.assertEqual(kinds[(date(2017, 12, 31), date(2018, 3, 31))], "Q")
        self.assertEqual(kinds[(date(2017, 10, 1), date(2018, 3, 31))], "H1")
        self.assertEqual(kinds[(date(2017, 10, 1), date(2018, 6, 30))], "9M")
        self.assertEqual(kinds[(date(2017, 10, 1), date(2018, 9, 29))], "FY")

    def test_instant_facts(self):
        a = next(r for r in self.rows if r["metric"] == "total_assets")
        self.assertIsNone(a["start"])
        self.assertEqual(a["period"], "I")
        s = next(r for r in self.rows if r["metric"] == "shares_outstanding")
        self.assertEqual((s["unit"], s["value"]), ("shares", 4443265000.0))

    def test_tag_precedence_recorded(self):
        ranks = {r["tag"]: r["tag_rank"] for r in self.rows if r["metric"] == "revenue"}
        self.assertLess(ranks["RevenueFromContractWithCustomerExcludingAssessedTax"], ranks["Revenues"])

    def test_point_in_time_selection(self):
        # As of 2019-06-01 only the Q1/Q2 10-Q facts were public; the FY2017 comparative arrived later.
        pit = sec.as_known_on(self.rows, date(2019, 6, 1), metric="revenue")
        ends = {(r["start"], r["end"]) for r in pit}
        self.assertIn((date(2017, 12, 31), date(2018, 3, 31)), ends)
        self.assertNotIn((date(2017, 10, 1), date(2018, 9, 29)), ends)
        # For a period reported twice, the preferred tag wins (no double counting).
        q1 = [r for r in pit if r["end"] == date(2017, 12, 30)]
        self.assertEqual(len(q1), 1)
        self.assertEqual(q1[0]["value"], 88293000000.0)

    def test_debt_tags_cover_common_alternatives(self):
        tags = {m: [t for _, t, _ in c] for m, c in sec.METRICS.items()}
        self.assertIn("LongTermDebtAndCapitalLeaseObligations", tags["debt_noncurrent"])
        self.assertIn("ShortTermBorrowings", tags["short_term_borrowings"])
        self.assertIn("DebtLongtermAndShorttermCombinedAmount", tags["debt_total"])
        self.assertGreaterEqual(sec.PARSER_VERSION, 2)

    def test_tickers_file(self):
        t = sec.parse_company_tickers(load("sec/company_tickers_exchange.json"))
        self.assertEqual(t["AAPL"], {"cik": 320193, "name": "Apple Inc.", "exchange": "Nasdaq"})


class AlpacaParsingTests(unittest.TestCase):
    def test_bars_follow_pagination(self):
        pages = iter(load("alpaca/bars_pages.json"))
        seen_tokens = []

        def fetch(params):
            seen_tokens.append(params.get("page_token"))
            return next(pages)

        rows = list(alpaca.iter_bars(fetch, ["AAPL", "MSFT"], date(2016, 1, 1), date(2016, 1, 31), "raw"))
        self.assertEqual(seen_tokens, [None, "tok1"])
        self.assertEqual([(r["symbol"], r["date"]) for r in rows],
                         [("AAPL", date(2016, 1, 4)), ("AAPL", date(2016, 1, 5)), ("MSFT", date(2016, 1, 4))])
        self.assertEqual(rows[0]["close"], 105.35)
        self.assertEqual(rows[0]["adjustment"], "raw")

    def test_bar_request_parameters(self):
        captured = {}

        def fetch(params):
            captured.update(params)
            return {"bars": {}, "next_page_token": None}

        list(alpaca.iter_bars(fetch, ["A", "B"], date(2016, 1, 1), date(2016, 2, 1), "all"))
        self.assertEqual(captured["symbols"], "A,B")
        self.assertEqual(captured["feed"], "sip")
        self.assertEqual(captured["adjustment"], "all")
        self.assertEqual(captured["limit"], 10000)


class UniverseTests(unittest.TestCase):
    def setUp(self):
        tickers = sec.parse_company_tickers(load("sec/company_tickers_exchange.json"))
        self.u = {r["symbol"]: r for r in universe.build(load("alpaca/assets.json"), tickers)}

    def test_kinds(self):
        self.assertEqual(self.u["AAPL"]["kind"], "common")
        self.assertEqual(self.u["SPY"]["kind"], "etf")
        self.assertEqual(self.u["ABCDW"]["kind"], "warrant")
        self.assertEqual(self.u["BAC.PRB"]["kind"], "preferred")

    def test_otc_excluded_delisted_kept(self):
        self.assertNotIn("TINY", self.u)
        self.assertIn("OLDCO", self.u)
        self.assertFalse(self.u["OLDCO"]["active"])

    def test_cik_joined_where_known(self):
        self.assertEqual(self.u["AAPL"]["cik"], 320193)
        self.assertIsNone(self.u["OLDCO"]["cik"])

    def test_non_ticker_symbols_excluded(self):
        rows = universe.build([{"class": "us_equity", "exchange": "NYSE", "symbol": "0029900E0",
                                "name": "Something Common Stock", "status": "inactive"},
                               {"class": "us_equity", "exchange": "NYSE", "symbol": "BRK.B",
                                "name": "Berkshire Hathaway Inc. Class B", "status": "active"}], {})
        self.assertEqual([r["symbol"] for r in rows], ["BRK.B"])

    def test_price_universe(self):
        syms = universe.price_symbols(self.u.values())
        self.assertEqual(syms, ["AAPL", "MSFT", "OLDCO", "SPY"])  # common + benchmark ETFs, sorted


if __name__ == "__main__":
    unittest.main()


class MassiveTests(unittest.TestCase):
    PAGES = [
        {"results": [
            {"ticker": "AABA", "name": "Altaba Inc. Common Stock", "type": "CS", "cik": "0001011006",
             "active": False, "delisted_utc": "2019-10-07T04:00:00Z"},
            {"ticker": "ZZZ", "name": "Old Zzz", "type": "CS", "cik": "0000000111", "active": False,
             "delisted_utc": "2005-01-03T05:00:00Z"}],
         "next_url": "https://api.massive.com/v3/reference/tickers?cursor=abc"},
        {"results": [
            {"ticker": "ZZZ", "name": "New Zzz Corp", "type": "CS", "cik": "0000000222", "active": False,
             "delisted_utc": "2021-06-01T04:00:00Z"},
            {"ticker": "NOCIK", "name": "No Cik", "type": "CS", "active": False}]},
    ]

    def test_pagination_follows_the_cursor(self):
        from sigma.data import massive
        seen = []
        pages = iter(self.PAGES)

        def fetch(path, params):
            seen.append(dict(params))
            return next(pages)

        rows = list(massive.iter_delisted(fetch))
        self.assertEqual(len(rows), 3)  # the record without a company id is dropped
        self.assertNotIn("cursor", seen[0])
        self.assertEqual(seen[1]["cursor"], "abc")
        self.assertEqual(seen[0]["type"], "CS")
        self.assertEqual(seen[0]["active"], "false")

    def test_cik_is_an_int_and_delisting_is_a_date(self):
        from sigma.data import massive
        rows = list(massive.iter_delisted(lambda p, q: self.PAGES[1] | {"next_url": None}))
        self.assertEqual(rows[0]["cik"], 222)
        self.assertEqual(rows[0]["delisted"], date(2021, 6, 1))

    def test_latest_delisting_wins_when_a_symbol_was_reused(self):
        from sigma.data import massive
        rows = [r for p in self.PAGES for r in massive.parse_results(p)]
        self.assertEqual(massive.cik_by_symbol(rows)["ZZZ"], 222)

    def test_fill_missing_ciks_only(self):
        rows = [{"symbol": "AAPL", "cik": 320193}, {"symbol": "ZZZ", "cik": None},
                {"symbol": "ABC", "cik": None}]
        out = universe.fill_ciks(rows, {"AAPL": 999, "ZZZ": 222})
        self.assertEqual([r["cik"] for r in out], [320193, 222, None])
        self.assertEqual([r.get("cik_source") for r in out], [None, "massive", None])
