"""Point-in-time trailing-twelve-month fundamentals from SEC facts."""

import unittest
from datetime import date, timedelta

import pandas as pd

from sigma.factors import fundamentals as F

D = date.fromisoformat


def fact(metric, start, end, value, filed, cik=1, tag_rank=0):
    s = D(start) if start else None
    e, f = D(end), D(filed)
    return {"cik": cik, "metric": metric, "tag": f"T{tag_rank}", "tag_rank": tag_rank, "unit": "USD",
            "start": s, "end": e, "period": F.sec.classify_period(s, e), "value": float(value),
            "form": "10-Q", "filed": f, "known_at": f + timedelta(days=1)}


# A calendar-year company. FY2023 revenue 400; FY2024 1H = 220 (1H 2023 was 190).
REV = [
    fact("revenue", "2023-01-01", "2023-12-31", 400, "2024-02-20"),
    fact("revenue", "2023-01-01", "2023-06-30", 190, "2023-08-05"),
    fact("revenue", "2024-01-01", "2024-03-31", 105, "2024-05-05"),
    fact("revenue", "2024-04-01", "2024-06-30", 115, "2024-08-05"),   # discrete Q2
    fact("revenue", "2024-01-01", "2024-06-30", 220, "2024-08-05"),   # YTD H1
    fact("revenue", "2023-01-01", "2023-06-30", 190, "2024-08-05"),   # comparative, re-filed
]


def periods(rows):
    return F.Periods.from_rows(rows)


class TtmTests(unittest.TestCase):
    def test_fiscal_year_end_uses_the_annual_figure(self):
        self.assertEqual(F.ttm(periods(REV), D("2023-12-31")), 400)

    def test_mid_year_uses_ytd_plus_prior_year_minus_prior_ytd(self):
        # 220 + 400 - 190
        self.assertEqual(F.ttm(periods(REV), D("2024-06-30")), 430)

    def test_falls_back_to_four_discrete_quarters(self):
        qs = [fact("revenue", s, e, v, "2024-08-05") for s, e, v in [
            ("2023-07-01", "2023-09-30", 1), ("2023-10-01", "2023-12-31", 2),
            ("2024-01-01", "2024-03-31", 3), ("2024-04-01", "2024-06-30", 4)]]
        self.assertEqual(F.ttm(periods(qs), D("2024-06-30")), 10)

    def test_missing_pieces_give_none_rather_than_a_partial_year(self):
        self.assertIsNone(F.ttm(periods(REV[2:5]), D("2024-06-30")))

    def test_52_53_week_years_match_within_tolerance(self):
        rows = [fact("revenue", "2022-09-25", "2023-09-30", 383, "2023-11-03"),
                fact("revenue", "2023-10-01", "2023-12-30", 120, "2024-02-02"),
                fact("revenue", "2022-09-25", "2022-12-31", 117, "2024-02-02")]
        self.assertEqual(F.ttm(periods(rows), D("2023-12-30")), 386)


class CleanTests(unittest.TestCase):
    def test_drops_impossible_dates(self):
        rows = pd.DataFrame([
            fact("revenue", "2023-01-01", "2023-12-31", 1, "2024-02-20"),
            fact("revenue", "2214-10-01", "2215-09-30", 1, "2016-02-20"),   # period ends after filing
            fact("revenue", "1926-02-18", "1927-02-17", 1, "2016-02-20"),   # far older than any comparative
        ])
        out = F.clean(rows)
        self.assertEqual(list(out["end"]), [D("2023-12-31")])


class PointInTimeTests(unittest.TestCase):
    def test_restatement_only_visible_after_it_is_filed(self):
        rows = pd.DataFrame([fact("revenue", "2023-01-01", "2023-12-31", 400, "2024-02-20"),
                             fact("revenue", "2023-01-01", "2023-12-31", 380, "2025-02-20")])
        before = F.point_in_time(rows, D("2024-06-01"))
        after = F.point_in_time(rows, D("2025-06-01"))
        self.assertEqual(before["value"].tolist(), [400])
        self.assertEqual(after["value"].tolist(), [380])

    def test_preferred_tag_wins_for_the_same_period(self):
        rows = pd.DataFrame([fact("revenue", "2023-01-01", "2023-12-31", 1, "2024-02-20", tag_rank=1),
                             fact("revenue", "2023-01-01", "2023-12-31", 400, "2024-02-20", tag_rank=0)])
        self.assertEqual(F.point_in_time(rows, D("2024-06-01"))["value"].tolist(), [400])

    def test_unfiled_facts_invisible(self):
        rows = pd.DataFrame(REV)
        pit = F.point_in_time(rows, D("2024-08-05"))  # known_at is the day after filing
        self.assertNotIn(D("2024-06-30"), set(pit["end"]))


class SnapshotTests(unittest.TestCase):
    def rows(self):
        rows = list(REV)
        rows += [fact("revenue", "2022-01-01", "2022-12-31", 360, "2023-02-20"),
                 fact("revenue", "2022-01-01", "2022-06-30", 170, "2023-08-05"),
                 fact("revenue", "2023-04-01", "2023-06-30", 99, "2023-08-05"),
                 fact("revenue", "2022-07-01", "2022-12-31", 1, "2023-02-20")]  # odd period, ignored
        rows += [fact("net_income", "2024-01-01", "2024-06-30", 22, "2024-08-05"),
                 fact("net_income", "2023-01-01", "2023-12-31", 40, "2024-02-20"),
                 fact("net_income", "2023-01-01", "2023-06-30", 18, "2024-08-05")]
        rows += [fact("total_assets", None, "2024-06-30", 1000, "2024-08-05"),
                 fact("total_assets", None, "2023-06-30", 900, "2023-08-05"),
                 fact("shares_outstanding", None, "2024-07-28", 50, "2024-08-05"),
                 fact("shares_diluted", "2024-04-01", "2024-06-30", 52, "2024-08-05"),
                 fact("shares_diluted", "2024-01-01", "2024-06-30", 51, "2024-08-05")]
        return pd.DataFrame(rows)

    def test_snapshot_aligns_every_metric_to_the_latest_fiscal_period(self):
        snap = F.snapshot(self.rows(), D("2024-09-01"))
        r = snap.loc[1]
        self.assertEqual(r["fiscal_end"], D("2024-06-30"))
        self.assertEqual(r["revenue"], 430)
        self.assertEqual(r["net_income"], 44)          # 22 + 40 - 18
        self.assertEqual(r["total_assets"], 1000)
        self.assertEqual(r["total_assets_1y"], 900)
        self.assertEqual(r["shares"], 52)               # latest-quarter diluted count

    def test_prior_year_ttm_for_growth(self):
        snap = F.snapshot(self.rows(), D("2024-09-01"))
        self.assertEqual(snap.loc[1, "revenue_1y"], 190 + 360 - 170)

    def test_stale_filers_are_dropped(self):
        snap = F.snapshot(self.rows(), D("2025-09-01"), max_age_days=200)
        self.assertNotIn(1, snap.index)

    def test_as_of_respected(self):
        snap = F.snapshot(self.rows(), D("2024-05-01"))
        self.assertEqual(snap.loc[1, "fiscal_end"], D("2023-12-31"))
        self.assertEqual(snap.loc[1, "revenue"], 400)


if __name__ == "__main__":
    unittest.main()


class DataGlitchTests(unittest.TestCase):
    def test_revenue_uses_the_largest_tag_so_partial_tags_never_win(self):
        # Insurers/REITs tag only their contract revenue with the ASC 606 tag; total is under Revenues.
        rows = pd.DataFrame([fact("revenue", "2023-01-01", "2023-12-31", 100, "2024-02-20", tag_rank=0),
                             fact("revenue", "2023-01-01", "2023-12-31", 900, "2024-02-20", tag_rank=1)])
        self.assertEqual(F.point_in_time(rows, D("2024-06-01"))["value"].tolist(), [900])

    def test_other_metrics_keep_tag_precedence(self):
        rows = pd.DataFrame([fact("cfo", "2023-01-01", "2023-12-31", 100, "2024-02-20", tag_rank=0),
                             fact("cfo", "2023-01-01", "2023-12-31", 900, "2024-02-20", tag_rank=1)])
        self.assertEqual(F.point_in_time(rows, D("2024-06-01"))["value"].tolist(), [100])

    def shares_rows(self, latest, cover=None):
        rows = [fact("revenue", "2026-01-01", "2026-06-30", 10, "2026-08-05"),
                fact("shares_diluted", "2026-04-01", "2026-06-30", latest, "2026-08-05"),
                fact("shares_diluted", "2025-10-01", "2025-12-31", 59_763_000, "2026-02-23"),
                fact("shares_diluted", "2025-07-01", "2025-09-30", 59_622_000, "2025-11-04")]
        if cover:
            rows.append(fact("shares_outstanding", None, "2026-08-01", cover, "2026-08-05"))
        return pd.DataFrame(rows)

    def test_share_count_filed_1000x_too_large_is_rescaled(self):
        # Waters' Q2 2026 10-Q tagged 98,204,000,000 diluted shares (true: 98.2 million).
        snap = F.snapshot(self.shares_rows(98_204_000_000), D("2026-09-01"))
        self.assertEqual(snap.loc[1, "shares"], 98_204_000)
        self.assertTrue(snap.loc[1, "shares_rescaled"])

    def test_cover_page_count_is_the_reference_when_available(self):
        # every recent quarter mis-scaled (Taboola restated its comparatives the same way)
        rows = self.shares_rows(291_392_907_000, cover=289_000_000)
        rows.loc[rows["metric"] == "shares_diluted", "value"] *= 1000
        rows.loc[1, "value"] = 291_392_907_000
        snap = F.snapshot(rows, D("2026-09-01"))
        self.assertEqual(snap.loc[1, "shares"], 291_392_907)

    def test_genuine_share_changes_are_left_alone(self):
        snap = F.snapshot(self.shares_rows(120_000_000), D("2026-09-01"))  # doubled: real dilution
        self.assertEqual(snap.loc[1, "shares"], 120_000_000)
        self.assertFalse(snap.loc[1, "shares_rescaled"])

    def test_debt_falls_back_to_the_combined_total(self):
        # DebtLongtermAndShorttermCombinedAmount already includes short-term borrowings
        rows = pd.DataFrame([
            fact("revenue", "2026-01-01", "2026-06-30", 10, "2026-08-05"),
            fact("short_term_borrowings", None, "2026-06-30", 50, "2026-08-05"),
            fact("debt_total", None, "2026-06-30", 700, "2026-08-05")])
        self.assertEqual(F.snapshot(rows, D("2026-09-01")).loc[1, "debt"], 700)

    def test_debt_prefers_the_split_plus_short_term_borrowings(self):
        rows = pd.DataFrame([
            fact("revenue", "2026-01-01", "2026-06-30", 10, "2026-08-05"),
            fact("debt_current", None, "2026-06-30", 100, "2026-08-05"),
            fact("debt_noncurrent", None, "2026-06-30", 400, "2026-08-05"),
            fact("short_term_borrowings", None, "2026-06-30", 50, "2026-08-05"),
            fact("debt_total", None, "2026-06-30", 999, "2026-08-05")])
        self.assertEqual(F.snapshot(rows, D("2026-09-01")).loc[1, "debt"], 550)

    def test_no_debt_tags_means_unknown_not_zero(self):
        rows = pd.DataFrame([fact("revenue", "2026-01-01", "2026-06-30", 10, "2026-08-05")])
        self.assertTrue(pd.isna(F.snapshot(rows, D("2026-09-01")).loc[1, "debt"]))
