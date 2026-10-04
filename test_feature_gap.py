"""
Tests for the Phase-0 gap analysis and the site-occupancy diagnostics.

The question these answer is "what exactly is missing, and how do I get it",
and the second question is why FRP_ANOMALY_VS_SITE came back 99.5% NaN.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import feature_gap as fg
import feature_ledger as fl


def fake_X(n=4000, seed=0):
    """Shaped like the real X: 96.7% single-cluster, so trends are empty."""
    rng = np.random.default_rng(seed)
    single = rng.random(n) < 0.967
    det = np.where(single, 1.0, rng.integers(2, 40, n).astype(float))
    return pd.DataFrame({
        "FRP_MEDIAN": rng.gamma(3.0, 15.0, n),
        "FRP_P90": rng.gamma(3.0, 20.0, n),
        "DETECTION_COUNT_LOG": np.log1p(det),
        "FRP_PERSISTENCE": np.where(single, 1.0, rng.uniform(0.1, 1.0, n)),
        "DURATION_HOURS_LOG": np.log1p(rng.integers(0, 500, n)),
        "AREA_GROWTH_RATE": np.where(single, np.nan, rng.normal(0.1, 0.5, n)),
        "SPREAD_SPEED_MEDIAN": np.where(single, np.nan, rng.gamma(2.0, 3.0, n)),
    })


class TestGapReport(unittest.TestCase):
    def test_every_specified_feature_appears(self):
        report = fg.gap_report(fake_X())
        self.assertEqual(len(report), len(fl.FEATURES_141))
        self.assertEqual(set(report["feature"]), set(fl.FEATURES_141))

    def test_three_buckets_only(self):
        report = fg.gap_report(fake_X())
        self.assertTrue(set(report["status"]) <= {"PRESENT", "SPARSE", "MISSING"})

    def test_populated_columns_are_present(self):
        report = fg.gap_report(fake_X())
        self.assertEqual(
            report[report.feature == "FRP_MEDIAN"]["status"].iloc[0], "PRESENT"
        )

    def test_empty_columns_are_sparse_not_missing(self):
        report = fg.gap_report(fake_X())
        self.assertEqual(
            report[report.feature == "AREA_GROWTH_RATE"]["status"].iloc[0],
            "SPARSE",
        )

    def test_absent_columns_are_missing(self):
        report = fg.gap_report(fake_X())
        missing = report[report["status"] == "MISSING"]
        self.assertGreater(len(missing), 100)
        self.assertIn("DIST_NEAREST_FACILITY_LOG", set(missing["feature"]))

    def test_sparse_is_not_treated_as_missing(self):
        """LightGBM routes NaN natively, so a 96% NaN column is weak, not absent,
        and must not be re-derived from scratch."""
        report = fg.gap_report(fake_X())
        self.assertNotIn(
            "AREA_GROWTH_RATE",
            set(report[report["status"] == "MISSING"]["feature"]),
        )

    def test_nan_fraction_is_recorded(self):
        report = fg.gap_report(fake_X())
        frac = report[report.feature == "AREA_GROWTH_RATE"]["nan_fraction"].iloc[0]
        self.assertGreater(frac, 0.9)


class TestObtainTags(unittest.TestCase):
    def test_every_missing_feature_gets_a_route(self):
        report = fg.gap_report(fake_X())
        missing = report[report["status"] == "MISSING"]
        self.assertTrue(
            set(missing["obtain"]) <= {"DERIVE", "EXTERNAL_RASTER",
                                       "EXTERNAL_TABLE"}
        )
        self.assertNotIn("-", set(missing["obtain"]))

    def test_industrial_features_are_derivable_from_disk(self):
        for name in ("DIST_MINE_LOG", "DIST_NEAREST_FACILITY_LOG"):
            tag, _ = fg.obtain_tag(name)
            self.assertEqual(tag, "DERIVE", name)

    def test_landcover_needs_a_raster_pass(self):
        for name in ("FOREST_FRAC_375M", "NBR_CHANGE"):
            tag, _ = fg.obtain_tag(name)
            self.assertEqual(tag, "EXTERNAL_RASTER", name)

    def test_weather_needs_a_raster_pass(self):
        tag, _ = fg.obtain_tag("TEMPERATURE_ANOMALY_Z")
        self.assertEqual(tag, "EXTERNAL_RASTER")

    def test_sentinel5p_is_tagged_as_a_table_to_re_download(self):
        tag, source = fg.obtain_tag("NO2_ANOMALY_Z")
        self.assertEqual(tag, "EXTERNAL_TABLE")
        self.assertIn("re-download", source)

    def test_lightning_is_a_table(self):
        tag, _ = fg.obtain_tag("LIGHTNING_COUNT_24H")
        self.assertEqual(tag, "EXTERNAL_TABLE")

    def test_derived_marker(self):
        tag, _ = fg.obtain_tag("MISSING_FEATURE_FRACTION")
        self.assertEqual(tag, "DERIVE")


class TestSummaryAndGrouping(unittest.TestCase):
    def test_counts_add_up_to_the_spec(self):
        report = fg.gap_report(fake_X())
        s = fg.summarise(report)
        self.assertEqual(
            s["n_usable"] + s["n_sparse"] + s["n_missing"], len(fl.FEATURES_141)
        )

    def test_grouping_is_by_route_and_group(self):
        report = fg.gap_report(fake_X())
        grouped = fg.missing_by_route(report)
        self.assertTrue(grouped)
        for (route, group), feats in grouped.items():
            self.assertIn(route, {"DERIVE", "EXTERNAL_RASTER",
                                  "EXTERNAL_TABLE"})
            self.assertTrue(feats)


class TestSiteOccupancy(unittest.TestCase):
    """FRP_ANOMALY_VS_SITE came back 99.5% NaN. The grid was the problem."""

    def _coords(self, n=70000, deg=0.1, seed=1):
        rng = np.random.default_rng(seed)
        lon = rng.uniform(68.0, 96.0, n)
        lat = rng.uniform(8.0, 33.0, n)
        return lon, lat

    def test_fine_grid_leaves_most_events_alone(self):
        lon, lat = self._coords()
        occ = fg.site_occupancy(lon, lat, site_deg=0.1)
        self.assertGreater(occ["frac_events_alone"], 0.0)
        self.assertLess(occ["frac_events_in_sites_ge2"], 1.0)

    def test_coarser_grid_gives_more_companies(self):
        lon, lat = self._coords()
        fine = fg.site_occupancy(lon, lat, site_deg=0.1)
        coarse = fg.site_occupancy(lon, lat, site_deg=1.0)
        self.assertGreater(coarse["frac_events_in_sites_ge2"],
                           fine["frac_events_in_sites_ge2"])

    def test_recommendation_picks_a_grid_that_supports_a_baseline(self):
        lon, lat = self._coords()
        table, chosen = fg.recommend_site_deg(lon, lat)
        self.assertIn(chosen, (0.05, 0.1, 0.25, 0.5, 1.0))
        chosen_row = table[table["site_deg"] == chosen].iloc[0]
        # The recommendation must at least beat the 0.1 deg grid.
        fine = table[table["site_deg"] == 0.1].iloc[0]
        self.assertGreaterEqual(
            chosen_row["frac_events_in_sites_ge2"],
            fine["frac_events_in_sites_ge2"],
        )

    def test_occupancy_counts_are_consistent(self):
        lon, lat = self._coords(n=5000)
        occ = fg.site_occupancy(lon, lat, site_deg=0.25)
        self.assertEqual(occ["n_events"], 5000)
        self.assertGreaterEqual(occ["n_sites"], 1)
        self.assertLessEqual(occ["n_sites"], occ["n_events"])
        self.assertAlmostEqual(
            occ["frac_events_alone"] + occ["frac_events_in_sites_ge2"],
            1.0, places=6,
        )


class TestSingleDetectionFlags(unittest.TestCase):
    def test_flags_added_from_the_count_log(self):
        frame = pd.DataFrame({
            "DETECTION_COUNT_LOG": np.log1p([1.0, 5.0]),
            "FRP_PERSISTENCE": [1.0, 0.5],
        })
        out = fg.add_single_detection_flags(frame)
        self.assertEqual(list(out["N_DETECTIONS"]), [1.0, 5.0])
        self.assertEqual(list(out["IS_SINGLE_DETECTION"]), [1.0, 0.0])
        self.assertEqual(list(out["N_DETECTIONS_GE3"]), [0.0, 1.0])

    def test_vacuous_persistence_is_flagged_for_single_detections(self):
        """An event with one detection was 'seen' on its only day, so
        persistence of 1.0 is vacuous rather than evidence of persistence."""
        frame = pd.DataFrame({
            "DETECTION_COUNT_LOG": np.log1p([1.0, 5.0]),
            "FRP_PERSISTENCE": [1.0, 1.0],
        })
        out = fg.add_single_detection_flags(frame)
        self.assertEqual(list(out["PERSISTENCE_IS_VACUOUS"]), [1.0, 0.0])

    def test_missing_count_stays_nan_not_false(self):
        frame = pd.DataFrame({
            "DETECTION_COUNT_LOG": [np.nan],
            "FRP_PERSISTENCE": [np.nan],
        })
        out = fg.add_single_detection_flags(frame)
        self.assertTrue(np.isnan(float(out["IS_SINGLE_DETECTION"].iloc[0])))

    def test_no_imputation_of_the_empty_features(self):
        """The whole point: add a flag, never fill a value."""
        frame = pd.DataFrame({
            "DETECTION_COUNT_LOG": np.log1p([1.0]),
            "AREA_GROWTH_RATE": [np.nan],
            "FRP_PERSISTENCE": [1.0],
        })
        out = fg.add_single_detection_flags(frame)
        self.assertTrue(np.isnan(float(out["AREA_GROWTH_RATE"].iloc[0])))


if __name__ == "__main__":
    unittest.main()
