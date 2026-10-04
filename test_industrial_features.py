"""
Tests for the P3a industrial feature core.

Run:  python3 -m unittest test_industrial_features -v
or:   python3 test_industrial_features.py

These run on synthetic geometry only. No Drive access, no network, no
geopandas: the module under test is deliberately I/O-free so the arithmetic can
be pinned down before 7.26M events are pushed through it.
"""

from __future__ import annotations

import unittest

import numpy as np

import feature_ledger as fl
import industrial_features as ind

KM = 1_000.0


def _tree(*xy):
    return np.array(xy, dtype="float64")


class TestSchemaContract(unittest.TestCase):
    def test_matches_ledger_industrial_group_exactly(self):
        """The 18 names and their order must equal the ledger's, always."""
        self.assertEqual(ind.INDUSTRIAL_FEATURES, fl.FEATURE_GROUPS[fl.GROUP_INDUSTRIAL])

    def test_is_the_leakage_partition(self):
        self.assertEqual(ind.INDUSTRIAL_FEATURES, fl.LABEL_DERIVED_FEATURES)

    def test_every_category_has_a_distance_feature(self):
        self.assertEqual(sorted(ind.CATEGORY_FEATURES), sorted(ind.CATEGORIES))
        self.assertEqual(len(ind.CATEGORY_FEATURES), 10)
        for feat in ind.CATEGORY_FEATURES.values():
            self.assertIn(feat, ind.INDUSTRIAL_FEATURES)

    def test_union_feature_is_not_a_category_feature(self):
        self.assertNotIn(ind.UNION_DISTANCE_FEATURE, ind.CATEGORY_FEATURES.values())

    def test_no_duplicate_keys(self):
        self.assertEqual(len(set(ind.INDUSTRIAL_FEATURES)), 18)


class TestProjectionGuard(unittest.TestCase):
    def test_rejects_lon_lat_degrees(self):
        with self.assertRaises(ValueError):
            ind.assert_projected_meters([77.0], [22.0])

    def test_accepts_equal_earth_metres(self):
        ind.assert_projected_meters([7_000_000.0], [2_000_000.0])

    def test_coarse_check_allows_origin_metres(self):
        # (0, 0) is ambiguous between degrees and metres, so the backstop must
        # not fire. This is exactly why crs is validated explicitly as well.
        ind.assert_projected_meters([0.0], [0.0])

    def test_empty_input_is_not_an_error(self):
        ind.assert_projected_meters([], [])

    def test_crs_accepts_authority_string(self):
        ind.assert_projected_crs("EPSG:6933")

    def test_crs_accepts_bare_code(self):
        ind.assert_projected_crs(6933)

    def test_crs_rejects_none(self):
        with self.assertRaises(ValueError):
            ind.assert_projected_crs(None)

    def test_crs_rejects_other_projection(self):
        with self.assertRaises(ValueError):
            ind.assert_projected_crs("EPSG:4326")

    def test_crs_rejects_utm_metre_projection(self):
        # Metres, but not Equal Earth: a different projection rescales
        # distances and must not be silently accepted.
        with self.assertRaises(ValueError):
            ind.assert_projected_crs("EPSG:32643")


class TestLogDistance(unittest.TestCase):
    def test_zero_distance_is_exactly_zero_not_neg_inf(self):
        self.assertEqual(float(ind.log_distance([0.0])[0]), 0.0)

    def test_one_km_is_log10_of_1001(self):
        self.assertAlmostEqual(
            float(ind.log_distance([KM])[0]), np.log10(1.0 + 1000.0), places=12
        )

    def test_monotonic(self):
        d = np.array([0.0, 100.0, 1_000.0, 10_000.0, 1_000_000.0])
        v = ind.log_distance(d)
        self.assertTrue(np.all(np.diff(v) > 0))

    def test_rejects_negative(self):
        with self.assertRaises(ValueError):
            ind.log_distance([-1.0])


class TestNearestAndCounts(unittest.TestCase):
    def setUp(self):
        self.tree = _tree((0.0, 0.0), (500.0, 0.0), (0.0, 3_000.0))

    def test_nearest_distance_is_metric(self):
        pts = _tree((10_000.0, 0.0), (0.0, 0.0))
        d = ind.nearest_distance(pts, self.tree)
        self.assertAlmostEqual(float(d[0]), 9_500.0, places=6)
        self.assertAlmostEqual(float(d[1]), 0.0, places=6)

    def test_empty_tree_is_nan_not_zero(self):
        d = ind.nearest_distance(_tree((1.0, 1.0)), np.empty((0, 2)))
        self.assertTrue(np.isnan(d[0]))

    def test_counts_are_inclusive_of_the_boundary(self):
        # 2 km from the origin: the 500 m and 3 km nodes are all inside 5 km,
        # none are inside 1 km.
        pts = _tree((2_000.0, 0.0))
        self.assertEqual(float(ind.count_within(pts, self.tree, 1_000.0)[0]), 0.0)
        self.assertEqual(float(ind.count_within(pts, self.tree, 5_000.0)[0]), 3.0)

    def test_count_within_counts_each_facility(self):
        pts = _tree((0.0, 0.0))
        self.assertEqual(float(ind.count_within(pts, self.tree, 100.0)[0]), 1.0)
        self.assertEqual(float(ind.count_within(pts, self.tree, 600.0)[0]), 2.0)

    def test_count_within_on_empty_tree_is_zero(self):
        pts = _tree((1.0, 1.0))
        out = ind.count_within(pts, np.empty((0, 2)), 10_000.0)
        self.assertEqual(float(out[0]), 0.0)


class TestDensityContrast(unittest.TestCase):
    def test_isolated_facility_scores_above_zero(self):
        # One facility in an otherwise empty world.
        tree = _tree((0.0, 0.0))
        out = ind.density_contrast(_tree((0.0, 0.0)), tree)
        self.assertGreater(float(out[0]), 0.0)

    def test_uniform_density_scores_near_zero(self):
        # A dense, even grid spanning well past the ring radius, so the local
        # and surrounding means agree and the contrast vanishes.
        grid = np.array(
            [
                (x * 1_000.0, y * 1_000.0)
                for x in range(-60, 61)
                for y in range(-60, 61)
            ],
            dtype="float64",
        )
        out = ind.density_contrast(_tree((0.0, 0.0)), grid)
        self.assertAlmostEqual(float(out[0]), 0.0, delta=0.05)

    def test_empty_tree_is_nan(self):
        out = ind.density_contrast(_tree((0.0, 0.0)), np.empty((0, 2)))
        self.assertTrue(np.isnan(out[0]))


class TestKnownFlare(unittest.TestCase):
    def _args(self, **over):
        base = dict(
            inside_facility=[1.0],
            active_days=[365.0],
            detection_count=[365.0],
            centroid_displacement_m=[50.0],
            frp_cv=[0.05],
            category_is_flare_site=[True],
        )
        base.update(over)
        return base

    def test_permanent_stack_is_a_flare(self):
        self.assertEqual(float(ind.known_flare_flags(**self._args())[0]), 1.0)

    def test_short_burn_at_a_refinery_is_not_a_flare(self):
        out = ind.known_flare_flags(**self._args(active_days=[4.0], detection_count=[6.0]))
        self.assertEqual(float(out[0]), 0.0)

    def test_wind_driven_fire_near_a_plant_is_not_a_flare(self):
        out = ind.known_flare_flags(**self._args(centroid_displacement_m=[45_000.0]))
        self.assertEqual(float(out[0]), 0.0)

    def test_far_from_any_flare_capable_facility_is_not_a_flare(self):
        out = ind.known_flare_flags(**self._args(category_is_flare_site=[False]))
        self.assertEqual(float(out[0]), 0.0)

    def test_missing_evidence_is_nan_not_zero(self):
        out = ind.known_flare_flags(**self._args(frp_cv=[np.nan]))
        self.assertTrue(np.isnan(out[0]))

    def test_flare_site_categories_include_refineries_and_lng(self):
        for cat in ("refinery", "lng"):
            self.assertIn(cat, ind.FLARE_SITE_CATEGORIES)

    def test_wildfire_thresholds_reject_a_single_hot_day(self):
        out = ind.known_flare_flags(
            inside_facility=[0.0],
            active_days=[1.0],
            detection_count=[3.0],
            centroid_displacement_m=[10_000.0],
            frp_cv=[1.4],
            category_is_flare_site=[False],
        )
        self.assertEqual(float(out[0]), 0.0)


class TestComputeIndustrialFeatures(unittest.TestCase):
    def setUp(self):
        self.trees = {
            "refinery": _tree((0.0, 0.0)),
            "powerplant": _tree((50_000.0, 0.0)),
            "mine": _tree((0.0, 120_000.0)),
        }
        self.points = _tree((0.0, 0.0), (30_000.0, 0.0))

    def _call(self, **kw):
        return ind.compute_industrial_features(
            self.points, self.trees, crs=ind.REQUIRED_CRS, **kw
        )

    def test_returns_all_18_keys_in_order(self):
        out = self._call()
        self.assertEqual(list(out.keys()), ind.INDUSTRIAL_FEATURES)

    def test_all_values_have_chunk_length(self):
        out = self._call()
        for key, value in out.items():
            self.assertEqual(len(value), len(self.points), key)

    def test_absent_category_is_nan_distinguishable_from_zero(self):
        out = self._call()
        self.assertTrue(np.isnan(out["DIST_STEEL_LOG"][0]))
        self.assertTrue(np.isnan(out["DIST_CEMENT_LOG"][0]))
        # A category with facilities present must not be NaN.
        self.assertTrue(np.isfinite(out["DIST_REFINERY_LOG"][0]))

    def test_union_distance_sees_a_facility_a_category_tree_missed(self):
        # The 30 km point is 20 km from the powerplant; no single named category
        # is that close, so only the union feature sees it.
        out = self._call()
        self.assertTrue(np.isfinite(out[ind.UNION_DISTANCE_FEATURE][1]))
        self.assertAlmostEqual(
            float(out[ind.UNION_DISTANCE_FEATURE][1]), np.log10(20_000.0 + 1.0), places=6
        )

    def test_inside_facility_is_nan_without_geometry(self):
        out = self._call()
        self.assertTrue(np.isnan(out["INSIDE_FACILITY"]).all())
        self.assertTrue(np.isnan(out["DIST_FACILITY_BOUNDARY_LOG"]).all())

    def test_boundary_distance_is_zero_when_inside(self):
        out = self._call(
            inside_flags=[1.0, 0.0],
            boundary_distance_m=[0.0, 1_200.0],
        )
        self.assertEqual(float(out["DIST_FACILITY_BOUNDARY_LOG"][0]), 0.0)
        self.assertAlmostEqual(
            float(out["DIST_FACILITY_BOUNDARY_LOG"][1]), np.log10(1201.0), places=6
        )

    def test_known_flare_is_nan_without_supporting_evidence(self):
        self.assertTrue(np.isnan(self._call()["KNOWN_FLARE"]).all())

    def test_known_flare_populates_with_full_evidence(self):
        out = self._call(
            inside_flags=[1.0, 0.0],
            active_days=[400.0, 400.0],
            detection_count=[400.0, 400.0],
            centroid_displacement_m=[10.0, 30_000.0],
            frp_cv=[0.02, 0.9],
            category_is_flare_site=[True, False],
        )
        self.assertEqual(float(out["KNOWN_FLARE"][0]), 1.0)
        self.assertEqual(float(out["KNOWN_FLARE"][1]), 0.0)

    def test_empty_gazetteer_yields_nan_distances_and_zero_counts(self):
        out = ind.compute_industrial_features(
            self.points, {}, crs=ind.REQUIRED_CRS
        )
        self.assertTrue(np.isnan(out[ind.UNION_DISTANCE_FEATURE]).all())
        self.assertEqual(float(out["FACILITY_COUNT_10KM"][0]), 0.0)
        self.assertTrue(np.isnan(out["INDUSTRIAL_DENSITY_CONTRAST"]).all())

    def test_rejects_missing_crs(self):
        with self.assertRaises(ValueError):
            ind.compute_industrial_features(self.points, self.trees)

    def test_rejects_geographic_crs(self):
        with self.assertRaises(ValueError):
            ind.compute_industrial_features(self.points, self.trees, crs="EPSG:4326")

    def test_rejects_degree_coordinates(self):
        with self.assertRaises(ValueError):
            ind.assert_projected_meters([77.5], [22.5])

    def test_rejects_malformed_points(self):
        with self.assertRaises(ValueError):
            ind.compute_industrial_features(
                np.zeros((4, 3)), self.trees, crs=ind.REQUIRED_CRS
            )

    def test_single_event_chunk(self):
        out = ind.compute_industrial_features(
            _tree((0.0, 0.0)), self.trees, crs=ind.REQUIRED_CRS
        )
        for value in out.values():
            self.assertEqual(len(value), 1)

    def test_empty_chunk_is_well_formed(self):
        out = ind.compute_industrial_features(
            np.empty((0, 2)), self.trees, crs=ind.REQUIRED_CRS
        )
        self.assertEqual(list(out.keys()), ind.INDUSTRIAL_FEATURES)
        for value in out.values():
            self.assertEqual(len(value), 0)


if __name__ == "__main__":
    unittest.main()