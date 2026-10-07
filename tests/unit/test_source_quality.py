import unittest

from sigma.providers.source_quality import Reliability, rank_sources, reliability_of


class ReliabilityTests(unittest.TestCase):
    IR = ("apple.com", "investor.apple.com")

    def test_sec_filing_is_primary_regulatory(self):
        url = "https://www.sec.gov/Archives/edgar/data/320193/000032019325000077/a8-kex991q4202509272025.htm"
        self.assertEqual(reliability_of(url, self.IR), Reliability.REGULATORY)

    def test_company_domains_are_first_party(self):
        url = "https://www.apple.com/newsroom/2025/10/apple-reports-fourth-quarter-results/"
        self.assertEqual(reliability_of(url, self.IR), Reliability.FIRST_PARTY)
        self.assertEqual(reliability_of("https://investor.apple.com/x", self.IR), Reliability.FIRST_PARTY)

    def test_lookalike_domain_is_not_first_party(self):
        self.assertEqual(reliability_of("https://apple.com.evil.example/x", self.IR), Reliability.UNVERIFIED)
        self.assertEqual(reliability_of("https://notapple.com/x", self.IR), Reliability.UNVERIFIED)

    def test_wire_and_exchange_reprints(self):
        self.assertEqual(reliability_of("https://www.nasdaq.com/press-release/x", self.IR),
                         Reliability.WIRE_REPRINT)
        self.assertEqual(reliability_of("https://www.businesswire.com/news/x", self.IR),
                         Reliability.WIRE_REPRINT)

    def test_unofficial_sec_mirror_is_not_regulatory(self):
        self.assertEqual(reliability_of("http://edgar.secdatabase.com/1299/x.htm", self.IR),
                         Reliability.UNVERIFIED)

    def test_rank_orders_by_reliability_and_keeps_ties_stable(self):
        urls = [
            "https://www.nasdaq.com/a",
            "https://blog.example/b",
            "https://www.sec.gov/c",
            "https://www.apple.com/d",
            "https://www.businesswire.com/e",
        ]
        ranked = rank_sources(urls, self.IR)
        self.assertEqual([u for u, _ in ranked], [
            "https://www.sec.gov/c", "https://www.apple.com/d",
            "https://www.nasdaq.com/a", "https://www.businesswire.com/e", "https://blog.example/b",
        ])


if __name__ == "__main__":
    unittest.main()
