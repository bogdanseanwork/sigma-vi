"""Loader orchestration with fake clients. Needs pyarrow (installed on the user's machine by setup)."""

import importlib.util
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from sigma.data import load
from sigma.data.store import ParquetStore

FIX = Path(__file__).resolve().parents[1] / "fixtures"
HAS_PARQUET = importlib.util.find_spec("pyarrow") is not None


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get_json(self, path, params=None):
        self.calls.append((path, dict(params or {})))
        r = self.responses[path]
        return r(params) if callable(r) else r


def fake_clients():
    assets = json.loads((FIX / "alpaca/assets.json").read_text())
    active = [a for a in assets if a["status"] == "active"]
    inactive = [a for a in assets if a["status"] == "inactive"]
    return {
        "alpaca_trading": FakeClient(
            {"/v2/assets": lambda p: active if p["status"] == "active" else inactive}),
        "sec": FakeClient({"/files/company_tickers_exchange.json":
                           json.loads((FIX / "sec/company_tickers_exchange.json").read_text())}),
        "alpaca_data": FakeClient({"/v2/stocks/bars": lambda p: {
            "bars": {s: [{"t": "2016-01-04T05:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1.0, "v": 100}]
                     for s in p["symbols"].split(",")},
            "next_page_token": None}}),
    }


class ChunkTests(unittest.TestCase):
    def test_chunks(self):
        self.assertEqual(load.chunks(["a", "b", "c"], 2), [["a", "b"], ["c"]])


@unittest.skipUnless(HAS_PARQUET, "pyarrow not installed")
class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ParquetStore(Path(self.tmp.name))
        self.clients = fake_clients()
        self.log = []

    def tearDown(self):
        self.tmp.cleanup()

    def test_universe_then_prices_then_report(self):
        n = load.step_universe(self.store, self.clients, self.log.append)
        self.assertEqual(n, 6)  # 7 assets minus the OTC listing
        load.step_prices(self.store, self.clients, self.log.append, end=date(2016, 1, 31))
        raw = self.store.read("prices", "raw")
        self.assertEqual(sorted(raw["symbol"].unique()), ["AAPL", "MSFT", "OLDCO", "SPY"])
        report = load.coverage_report(self.store)
        self.assertIn("delisted common stocks with prices: 1 of 1", report)

    def test_prices_drop_symbols_the_provider_rejects(self):
        from sigma.data.http import HttpError
        load.step_universe(self.store, self.clients, self.log.append)
        good = self.clients["alpaca_data"].responses["/v2/stocks/bars"]

        def picky(params):
            if "OLDCO" in params["symbols"].split(","):
                raise HttpError(400, '{"message":"invalid symbol: OLDCO"}')
            return good(params)

        self.clients["alpaca_data"].responses["/v2/stocks/bars"] = picky
        load.step_prices(self.store, self.clients, self.log.append, end=date(2016, 1, 31))
        raw = self.store.read("prices", "raw")
        self.assertEqual(sorted(raw["symbol"].unique()), ["AAPL", "MSFT", "SPY"])
        self.assertTrue(any("OLDCO" in line for line in self.log))

    def test_fundamentals_skip_when_already_built_from_this_file(self):
        import zipfile
        z = self.store.path("raw", "companyfacts.zip")
        z.parent.mkdir(parents=True)
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("CIK0000320193.json", (FIX / "sec/CIK0000320193.json").read_text())
        first = load.step_fundamentals(self.store, {}, self.log.append)
        self.assertGreater(first, 0)
        self.assertEqual(load.step_fundamentals(self.store, {}, self.log.append), 0)
        self.assertTrue(any("already built" in line for line in self.log))
        from sigma.data import sec
        old = sec.PARSER_VERSION
        try:  # a parser change (new tags) rebuilds from the same file
            sec.PARSER_VERSION = old + 1
            self.assertGreater(load.step_fundamentals(self.store, {}, self.log.append), 0)
        finally:
            sec.PARSER_VERSION = old

    def test_links_fill_delisted_company_ids_and_survive_a_universe_rebuild(self):
        load.step_universe(self.store, self.clients, self.log.append)
        massive = FakeClient({"/v3/reference/tickers": {"results": [
            {"ticker": "OLDCO", "name": "Old Co", "type": "CS", "cik": "0000555555", "active": False,
             "delisted_utc": "2019-01-02T05:00:00Z"}], "next_url": None}})
        self.clients["massive"] = massive
        n = load.step_links(self.store, self.clients, self.log.append)
        self.assertEqual(n, 1)
        uni = self.store.read("universe.parquet").set_index("symbol")
        self.assertEqual(int(uni.loc["OLDCO", "cik"]), 555555)
        self.assertEqual(uni.loc["OLDCO", "cik_source"], "massive")
        self.assertEqual(int(uni.loc["AAPL", "cik"]), 320193)           # SEC link untouched
        load.step_universe(self.store, self.clients, self.log.append)   # rebuild keeps the links
        self.assertEqual(int(self.store.read("universe.parquet").set_index("symbol").loc["OLDCO", "cik"]),
                         555555)
        self.assertIn("delisted common stocks linked to an SEC company: 1 of 1",
                      load.coverage_report(self.store))

    def test_sectors_fetch_sic_once_per_company(self):
        load.step_universe(self.store, self.clients, self.log.append)
        subs = FakeClient({"/submissions/CIK0000320193.json": {
            "cik": "320193", "name": "Apple Inc.", "sic": "3571", "sicDescription": "Electronic Computers"},
            "/submissions/CIK0000789019.json": {"cik": "789019", "name": "MICROSOFT CORP", "sic": "7372",
                                                "sicDescription": "Services-Prepackaged Software"}})
        self.clients["sec_data"] = subs
        load.step_sectors(self.store, self.clients, self.log.append)
        s = self.store.read("sectors").set_index("cik")
        self.assertEqual(s.loc[320193, "sector"], "Information Technology")
        self.assertEqual(s.loc[789019, "sic"], 7372)
        calls = len(subs.calls)
        load.step_sectors(self.store, self.clients, self.log.append)  # resumable: nothing re-fetched
        self.assertEqual(len(subs.calls), calls)

    def test_prices_resume_skips_finished_batches(self):
        load.step_universe(self.store, self.clients, self.log.append)
        load.step_prices(self.store, self.clients, self.log.append, end=date(2016, 1, 31))
        calls = len(self.clients["alpaca_data"].calls)
        load.step_prices(self.store, self.clients, self.log.append, end=date(2016, 1, 31))
        self.assertEqual(len(self.clients["alpaca_data"].calls), calls)


if __name__ == "__main__":
    unittest.main()
