"""Tests for quality_features.py (P3b, 8 of 141)."""

import unittest

import numpy as np

import feature_ledger as fl
import quality_features as qf


class TestSchemaContract(unittest.TestCase):
    def test_order_matches_ledger(self):
        self.assertEqual(qf.QUALITY_FEATURES, fl.FEATURE_GROUPS[fl.GROUP_QUALITY])
        self.assertEqual(qf.N_QUALITY, 8)

    def test_cloud_fraction_is_the_only_gee_feature(self):
        derived = set(qf.DETECTION_DERIVED)
        all_f = set(qf.QUALITY_FEATURES)
        self.assertEqual(all_f - derived, {qf.GEE_ONLY_FEATURE, qf.POST_ASSEMBLY_FEATURE})
        self.assertEqual(qf.GEE_ONLY_FEATURE, "CLOUD_FRACTION")


class TestConfidence(unittest.TestCase):
    def test_ranks_letters(self):
        self.assertEqual(qf.confidence_rank("l"), 1.0)
        self.assertEqual(qf.confidence_rank("n"), 2.0)
        self.assertEqual(qf.confidence_rank("h"), 3.0)
        self.assertEqual(qf.confidence_rank("H"), 3.0)
        self.assertEqual(qf.confidence_rank(" nominal "), 2.0)

    def test_unknown_label_is_nan_not_zero(self):
        for bad in (None, "", "unknown", "9"):
            self.assertTrue(np.isnan(qf.confidence_rank(bad)), bad)

    def test_numeric_column_passes_through(self):
        self.assertEqual(qf.confidence_rank(3), 3.0)

    def test_event_mean_ignores_unrecognised(self):
        v = qf.event_confidence_mean(["h", "h", "bogus", "n"])
        self.assertAlmostEqual(v, (3.0 + 3.0 + 2.0) / 3.0)

    def test_event_mean_all_missing_is_nan(self):
        self.assertTrue(np.isnan(qf.event_confidence_mean(["x", "y"])))
        self.assertTrue(np.isnan(qf.event_confidence_mean([])))


class TestPixelSize(unittest.TestCase):
    def test_nadir_and_edge(self):
        self.assertAlmostEqual(qf.effective_pixel_size_m(0, "NOAA-20"), 375.0)
        self.assertAlmostEqual(qf.effective_pixel_size_m(65, "NOAA-20"), 750.0)
        self.assertAlmostEqual(qf.effective_pixel_size_m(0, "AQUA"), 1000.0)
        self.assertAlmostEqual(qf.effective_pixel_size_m(45, "AQUA"), 2000.0)

    def test_linear_in_between(self):
        mid = qf.effective_pixel_size_m(32.5, "SNPP")
        self.assertAlmostEqual(mid, 562.5)

    def test_signed_angle_uses_magnitude(self):
        self.assertEqual(
            qf.effective_pixel_size_m(-30, "SNPP"),
            qf.effective_pixel_size_m(30, "SNPP"),
        )

    def test_clamps_beyond_geometric_limit(self):
        self.assertAlmostEqual(qf.effective_pixel_size_m(90, "SNPP"), 750.0)

    def test_unknown_sensor_is_nan(self):
        # A Landsat-derived row must not inherit the VIIRS 375 m profile.
        self.assertTrue(np.isnan(qf.effective_pixel_size_m(0, "LANDSAT-8")))
        self.assertTrue(np.isnan(qf.effective_pixel_size_m(0, None)))

    def test_missing_angle_is_nan(self):
        self.assertTrue(np.isnan(qf.effective_pixel_size_m(None, "SNPP")))
        self.assertTrue(np.isnan(qf.effective_pixel_size_m(float("nan"), "SNPP")))


class TestScanTerms(unittest.TestCase):
    def test_angles_use_unit_vectors(self):
        # +179 and -179 straddle the discontinuity; their mean direction must
        # stay near the -x axis, not collapse toward 0.
        t = qf.event_scan_terms([179.0, -179.0], ["SNPP", "SNPP"])
        self.assertLess(t["SCAN_ANGLE_COS"], -0.99)
        self.assertAlmostEqual(t["SCAN_ANGLE_SIN"], 0.0, places=6)

    def test_view_zenith_is_magnitude_of_scan(self):
        t = qf.event_scan_terms([-40.0], ["SNPP"])
        self.assertGreater(t["VIEW_ZENITH_SIN"], 0.0)
        self.assertGreater(t["VIEW_ZENITH_COS"], 0.0)

    def test_nadir_scan_gives_zero_sine(self):
        t = qf.event_scan_terms([0.0], ["SNPP"])
        self.assertAlmostEqual(t["SCAN_ANGLE_SIN"], 0.0)
        self.assertAlmostEqual(t["SCAN_ANGLE_COS"], 1.0)

    def test_no_scan_angles_is_all_nan(self):
        t = qf.event_scan_terms([], ["SNPP"])
        for key in (
            "PIXEL_SIZE",
            "SCAN_ANGLE_SIN",
            "SCAN_ANGLE_COS",
            "VIEW_ZENITH_SIN",
            "VIEW_ZENITH_COS",
        ):
            self.assertTrue(np.isnan(t[key]), key)

    def test_partially_missing_scan_uses_the_finite_rows(self):
        t = qf.event_scan_terms([0.0, np.nan, 65.0], ["SNPP"] * 3)
        self.assertAlmostEqual(t["PIXEL_SIZE"], (375.0 + 750.0) / 2.0)

    def test_pixel_size_nan_without_satellite(self):
        t = qf.event_scan_terms([10.0], [None])
        self.assertTrue(np.isnan(t["PIXEL_SIZE"]))
        self.assertFalse(np.isnan(t["SCAN_ANGLE_SIN"]))


class TestMissingFeatureFraction(unittest.TestCase):
    def test_counts_present_and_absent_keys(self):
        row = {"A": 1.0, "B": float("nan"), "C": None}
        self.assertAlmostEqual(
            qf.missing_feature_fraction(row, ["A", "B", "C", "D"]), 3 / 4
        )

    def test_non_numeric_counts_as_missing(self):
        self.assertAlmostEqual(
            qf.missing_feature_fraction({"A": "abc"}, ["A"]), 1.0
        )

    def test_empty_row_is_nan_not_one(self):
        self.assertTrue(np.isnan(qf.missing_feature_fraction({})))
        self.assertTrue(np.isnan(qf.missing_feature_fraction(None)))

    def test_complete_row_is_zero(self):
        self.assertAlmostEqual(
            qf.missing_feature_fraction({"A": 0.0, "B": 2.0}, ["A", "B"]), 0.0
        )


class TestComputeQualityFeatures(unittest.TestCase):
    def test_exact_keys_and_order(self):
        out = qf.compute_quality_features(3)
        self.assertEqual(list(out.keys()), qf.QUALITY_FEATURES)
        for k, v in out.items():
            self.assertEqual(len(v), 3, k)

    def test_all_nan_without_input(self):
        out = qf.compute_quality_features(2)
        for k, v in out.items():
            self.assertTrue(np.all(np.isnan(v)), k)

    def test_cloud_fraction_passes_through(self):
        out = qf.compute_quality_features(2, cloud_fraction=[0.3, np.nan])
        self.assertAlmostEqual(out["CLOUD_FRACTION"][0], 0.3)
        self.assertTrue(np.isnan(out["CLOUD_FRACTION"][1]))

    def test_confidence_and_scan_populate(self):
        out = qf.compute_quality_features(
            2,
            confidence_values=[["h", "h"], ["l"]],
            scan_angles_deg=[[0.0, 65.0], [20.0]],
            satellites=[["NOAA-20", "NOAA-20"], ["SNPP"]],
        )
        self.assertAlmostEqual(out["FIRMS_CONFIDENCE"][0], 3.0)
        self.assertAlmostEqual(out["FIRMS_CONFIDENCE"][1], 1.0)
        self.assertAlmostEqual(out["PIXEL_SIZE"][0], (375.0 + 750.0) / 2.0)

    def test_mismatched_input_length_raises(self):
        with self.assertRaises(ValueError):
            qf.compute_quality_features(3, confidence_values=[["h"], ["h"]])

    def test_wrong_shape_raises(self):
        with self.assertRaises(ValueError):
            qf.compute_quality_features(2, cloud_fraction=[0.1, 0.2, 0.3])

    def test_zero_events(self):
        out = qf.compute_quality_features(0)
        for v in out.values():
            self.assertEqual(len(v), 0)


if __name__ == "__main__":
    unittest.main()
