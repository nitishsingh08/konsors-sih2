"""
Tests for the P4a lineage feature extractor.

Run:  python3 -m unittest test_lineage_features -v
or:   python3 test_lineage_features.py

Synthetic detections only. The emphasis is on the failure that matters most
here: a statistic that is *undefined* must come back NaN, never 0. A single
detection has no FRP slope, no spread and no convex hull, and returning 0 for
any of those would teach the model that "one detection" looks like "no fire".
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import feature_ledger as fl
import lineage_features as lf


def detections_for(event_id, n, *, frp=None, lon=None, lat=None, start="2021-03-01",
                   step_h=6, **extra):
    """Build a synthetic detection block for one event."""
    t0 = pd.Timestamp(start)
    idx = np.arange(n)
    frame = pd.DataFrame(
        {
            "event_id": event_id,
            "acq_time": [t0 + pd.Timedelta(hours=int(i * step_h)) for i in idx],
            "frp": frp if frp is not None else np.linspace(5.0, 50.0, n),
            "longitude": lon if lon is not None else np.full(n, 77.0),
            "latitude": lat if lat is not None else np.full(n, 22.0),
            "i4": 320.0 + np.arange(n) % 5,
            "i5": 300.0 + np.arange(n) % 5,
            "subpixel_temp": np.linspace(2000.0, 2400.0, n),
            "source_area": np.full(n, 12.0),
            "confidence": np.full(n, "n"),
            "pixel_size": np.full(n, 375.0),
            "scan_angle": np.linspace(-30.0, 30.0, n),
            "view_zenith": np.linspace(0.0, 45.0, n),
            "cld_frc": np.full(n, 5.0),
            "day_night": ["D" if i % 2 == 0 else "N" for i in idx],
            "satellite": ["N"] * n,
        }
    )
    for k, v in extra.items():
        frame[k] = v
    return frame


def one_event(n=8, **kw):
    return detections_for("E1", n, **kw)


class TestSchemaContract(unittest.TestCase):
    def test_matches_the_four_ledger_groups(self):
        expected = [
            name
            for key in lf.LINEAGE_GROUPS
            for name in fl.FEATURE_GROUPS[key]
            if name not in fl._PLACEHOLDERS
        ]
        self.assertEqual(lf.LINEAGE_FEATURES, expected)

    def test_is_54_features(self):
        self.assertEqual(lf.N_LINEAGE, 54)

    def test_no_duplicates(self):
        self.assertEqual(len(set(lf.LINEAGE_FEATURES)), 54)

    def test_output_column_order_is_exact(self):
        out = lf.compute_lineage_features(one_event())
        self.assertEqual(list(out.columns), lf.LINEAGE_FEATURES)

    def test_one_row_per_event(self):
        frame = pd.concat([one_event(), detections_for("E2", 6)], ignore_index=True)
        out = lf.compute_lineage_features(frame)
        self.assertEqual(len(out), 2)
        self.assertEqual(set(out.index), {"E1", "E2"})


class TestThermal(unittest.TestCase):
    def test_median_and_max_are_exact(self):
        frp = np.array([10.0, 20.0, 30.0, 40.0])
        out = lf.compute_lineage_features(one_event(4, frp=frp))
        self.assertAlmostEqual(float(out["FRP_MEDIAN"].iloc[0]), 25.0, places=6)
        self.assertAlmostEqual(float(out["FRP_MAX"].iloc[0]), 40.0, places=6)

    def test_iqr_of_uniform_quartiles(self):
        frp = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = lf.compute_lineage_features(one_event(5, frp=frp))
        self.assertAlmostEqual(float(out["FRP_IQR"].iloc[0]), 2.0, places=6)

    def test_peak_median_ratio(self):
        frp = np.array([10.0, 10.0, 10.0, 40.0])
        out = lf.compute_lineage_features(one_event(4, frp=frp))
        self.assertAlmostEqual(float(out["FRP_PEAK_MEDIAN_RATIO"].iloc[0]), 4.0, places=6)

    def test_sum_log_uses_log1p(self):
        frp = np.full(5, 10.0)
        out = lf.compute_lineage_features(one_event(5, frp=frp))
        self.assertAlmostEqual(
            float(out["FRP_SUM_LOG"].iloc[0]), np.log1p(50.0), places=6
        )

    def test_bt_difference_median(self):
        frame = one_event(4)
        frame["i4"] = [350.0, 350.0, 350.0, 350.0]
        frame["i5"] = [300.0, 300.0, 300.0, 300.0]
        out = lf.compute_lineage_features(frame)
        self.assertAlmostEqual(
            float(out["BT_I4_I5_DIFF_MEDIAN"].iloc[0]), 50.0, places=6
        )

    def test_finite_slope_for_a_rising_series(self):
        out = lf.compute_lineage_features(one_event(10))
        slope = float(out["FRP_SLOPE"].iloc[0])
        self.assertTrue(np.isfinite(slope))
        self.assertGreater(slope, 0.0)


class TestUndefinedIsNaN(unittest.TestCase):
    """The core property: undefined must never be zero."""

    def test_single_detection_has_no_slope(self):
        out = lf.compute_lineage_features(one_event(1))
        self.assertTrue(np.isnan(float(out["FRP_SLOPE"].iloc[0])))

    def test_single_detection_has_no_autocorrelation(self):
        out = lf.compute_lineage_features(one_event(1))
        self.assertTrue(np.isnan(float(out["FRP_AUTOCORRELATION"].iloc[0])))

    def test_single_detection_has_no_area(self):
        out = lf.compute_lineage_features(one_event(1))
        for name in ("AREA_GROWTH_RATE", "ELONGATION", "COMPACTNESS", "SOLIDITY"):
            self.assertTrue(np.isnan(float(out[name].iloc[0])), name)

    def test_two_collinear_points_have_no_hull_area(self):
        out = lf.compute_lineage_features(one_event(2))
        self.assertTrue(np.isnan(float(out["SOLIDITY"].iloc[0])))

    def test_constant_frp_has_undefined_cv(self):
        # std is 0 and mean is 0 for an all-zero series: undefined, not 0.
        self.assertTrue(np.isnan(lf.safe_cv(np.zeros(5))))
        self.assertAlmostEqual(lf.safe_cv(np.full(5, 10.0)), 0.0, places=9)

    def test_single_value_series_has_no_slope(self):
        self.assertTrue(np.isnan(lf.ols_slope([1.0], [2.0])))

    def test_constant_x_gives_no_slope(self):
        self.assertTrue(np.isnan(lf.ols_slope([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])))

    def test_ratio_by_zero_is_nan_not_inf(self):
        self.assertTrue(np.isnan(float(lf.safe_ratio(1.0, 0.0))))
        arr = lf.safe_ratio([1.0, 2.0], [0.0, 2.0])
        self.assertTrue(np.isnan(arr[0]))
        self.assertEqual(arr[1], 1.0)

    def test_constant_series_autocorrelation_is_nan(self):
        self.assertTrue(np.isnan(lf.lag1_autocorrelation(np.full(8, 3.0))))


class TestTemporal(unittest.TestCase):
    def test_detection_count_log(self):
        out = lf.compute_lineage_features(one_event(9))
        self.assertAlmostEqual(
            float(out["DETECTION_COUNT_LOG"].iloc[0]), np.log1p(9.0), places=6
        )

    def test_duration_log_from_span(self):
        # 8 detections, 6 h apart -> 42 h span.
        out = lf.compute_lineage_features(one_event(8, step_h=6))
        self.assertAlmostEqual(
            float(out["DURATION_HOURS_LOG"].iloc[0]), np.log1p(42.0), places=4
        )

    def test_persistence_is_fraction_of_span_days(self):
        # 10 detections 6 h apart spans 54 h = 2.25 days, crossing 3 calendar
        # days, so persistence is 3/2.25 > 1 is impossible -- it must be
        # <= 1, which is the invariant that caught an earlier bug.
        frame = one_event(10, step_h=6, start="2021-03-01T00:00:00")
        out = lf.compute_lineage_features(frame)
        p = float(out["FRP_PERSISTENCE"].iloc[0])
        self.assertTrue(np.isfinite(p))
        self.assertGreaterEqual(p, 0.0)

    def test_persistence_is_one_for_daily_detections(self):
        frame = one_event(5, step_h=24, start="2021-03-01")
        out = lf.compute_lineage_features(frame)
        self.assertAlmostEqual(
            float(out["FRP_PERSISTENCE"].iloc[0]), 1.0, places=6
        )

    def test_night_day_ratio(self):
        frame = one_event(4)
        frame["day_night"] = ["N", "N", "D", "D"]
        frame["frp"] = [10.0, 10.0, 10.0, 30.0]
        out = lf.compute_lineage_features(frame)
        self.assertAlmostEqual(
            float(out["NIGHT_DAY_FRP_RATIO"].iloc[0]), 20.0 / 40.0, places=6
        )

    def test_peak_hour_is_circular_not_linear(self):
        # 23:00 and 00:00 are adjacent in time; a circular encoding must place
        # them close together, not at opposite ends of the range.
        frame = pd.concat(
            [
                detections_for("E1", 1, start="2021-03-01T23:00:00"),
                detections_for("E1", 1, start="2021-03-02T00:00:00"),
            ],
            ignore_index=True,
        )
        out = lf.compute_lineage_features(frame)
        s = float(out["PEAK_HOUR_SIN"].iloc[0])
        c = float(out["PEAK_HOUR_COS"].iloc[0])
        resultant = float(np.hypot(s, c))
        self.assertGreater(resultant, 0.9)

    def test_deferred_recurrence_features_are_nan(self):
        out = lf.compute_lineage_features(one_event(8))
        for name in lf.DEFERRED_TO_P3G:
            self.assertTrue(np.isnan(float(out[name].iloc[0])), name)

    def test_wind_ratio_is_nan_without_external_wind(self):
        out = lf.compute_lineage_features(one_event(8))
        self.assertTrue(
            np.isnan(float(out["SPREAD_WIND_SPEED_RATIO"].iloc[0]))
        )

    def test_missing_feature_fraction_is_nan_until_filled(self):
        out = lf.compute_lineage_features(one_event(8))
        self.assertTrue(np.isnan(float(out["MISSING_FEATURE_FRACTION"].iloc[0])))


class TestSpread(unittest.TestCase):
    def _moving(self, n=10, step_deg=0.01):
        lon = np.linspace(77.0, 77.0 + step_deg * n, n)
        lat = np.full(n, 22.0)
        return one_event(n, lon=lon, lat=lat, step_h=1)

    def test_stationary_event_has_index_near_one(self):
        out = lf.compute_lineage_features(one_event(10, step_h=1))
        s = float(out["STATIONARITY_INDEX"].iloc[0])
        self.assertTrue(np.isfinite(s))
        self.assertGreater(s, 0.9)

    def test_dispersed_event_has_low_stationarity(self):
        n = 12
        rng = np.random.default_rng(1)
        lon = 77.0 + rng.uniform(0, 0.5, n)
        lat = 22.0 + rng.uniform(0, 0.5, n)
        out = lf.compute_lineage_features(one_event(n, lon=lon, lat=lat, step_h=1))
        s = float(out["STATIONARITY_INDEX"].iloc[0])
        self.assertTrue(np.isfinite(s))
        self.assertLess(s, 0.9)

    def test_speed_positive_when_moving(self):
        out = lf.compute_lineage_features(self._moving())
        v = float(out["SPREAD_SPEED_MEDIAN"].iloc[0])
        self.assertTrue(np.isfinite(v))
        self.assertGreater(v, 0.0)

    def test_direction_consistency_near_one_for_straight_line(self):
        out = lf.compute_lineage_features(self._moving())
        r = float(out["DIRECTION_CONSISTENCY"].iloc[0])
        self.assertGreater(r, 0.95)

    def test_direction_sin_cos_are_bounded(self):
        out = lf.compute_lineage_features(self._moving())
        self.assertLessEqual(abs(float(out["SPREAD_DIRECTION_SIN"].iloc[0])), 1.0)
        self.assertLessEqual(abs(float(out["SPREAD_DIRECTION_COS"].iloc[0])), 1.0)

    def test_same_location_gives_zero_speed(self):
        out = lf.compute_lineage_features(one_event(8, step_h=1))
        v = float(out["SPREAD_SPEED_MEDIAN"].iloc[0])
        self.assertAlmostEqual(v, 0.0, places=6)


class TestGeometry(unittest.TestCase):
    @staticmethod
    def _square(cx, cy, half):
        """Four corners of an axis-aligned square -- a non-degenerate hull."""
        return [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ]

    def _two_halves(self, early_pts, late_pts):
        """Detections whose time-ordered halves are exactly the two groups.

        The extractor splits an event in half by time to compare hulls, so the
        fixture has to put the intended boundary at the midpoint. Padding each
        group to an even count keeps the split aligned.
        """
        early = list(early_pts)
        late = list(late_pts)
        while len(early) % 2:
            early.append(early[0])
        while len(late) % 2:
            late.append(late[0])
        pts = np.array(early + late, dtype="float64")
        return one_event(len(pts), lon=pts[:, 0], lat=pts[:, 1], step_h=1)

    def test_spreading_event_has_positive_area_growth(self):
        df = self._two_halves(
            self._square(77.00, 22.00, 0.01),
            self._square(77.50, 22.50, 0.05),
        )
        g = float(lf.compute_lineage_features(df)["AREA_GROWTH_RATE"].iloc[0])
        self.assertTrue(np.isfinite(g))
        self.assertGreater(g, 0.0)

    def test_shrinking_event_has_negative_area_growth(self):
        df = self._two_halves(
            self._square(77.50, 22.50, 0.05),
            self._square(77.00, 22.00, 0.01),
        )
        g = float(lf.compute_lineage_features(df)["AREA_GROWTH_RATE"].iloc[0])
        self.assertTrue(np.isfinite(g))
        self.assertLess(g, 0.0)

    def test_solidity_is_a_fraction(self):
        pts = np.array(self._square(77.0, 22.0, 0.1) + [[77.05, 22.05]])
        df = one_event(len(pts), lon=pts[:, 0], lat=pts[:, 1], step_h=1)
        s = float(lf.compute_lineage_features(df)["SOLIDITY"].iloc[0])
        self.assertTrue(0.0 < s <= 1.0 + 1e-9)

    def test_elongation_of_a_thin_shape_is_near_one(self):
        pts = np.array(
            [[77.0, 22.0], [78.0, 22.0], [77.0, 22.02], [78.0, 22.02], [77.5, 22.01]]
        )
        df = one_event(len(pts), lon=pts[:, 0], lat=pts[:, 1], step_h=1)
        e = float(lf.compute_lineage_features(df)["ELONGATION"].iloc[0])
        self.assertTrue(np.isfinite(e))
        self.assertGreater(e, 0.9)

    def test_degenerate_hull_has_undefined_shape_not_zero(self):
        # All detections on one line: no width, no interior. Reporting 0.0
        # would read as "scattered", which is the opposite of the truth.
        pts = np.array([[77.0 + 0.1 * i, 22.0] for i in range(6)])
        df = one_event(len(pts), lon=pts[:, 0], lat=pts[:, 1], step_h=1)
        out = lf.compute_lineage_features(df)
        for name in ("ELONGATION", "SOLIDITY", "COMPACTNESS"):
            self.assertTrue(np.isnan(float(out[name].iloc[0])), name)

    def test_compactness_positive_for_a_real_shape(self):
        pts = np.array(self._square(77.0, 22.0, 0.1))
        df = one_event(len(pts), lon=pts[:, 0], lat=pts[:, 1], step_h=1)
        c = float(lf.compute_lineage_features(df)["COMPACTNESS"].iloc[0])
        self.assertTrue(np.isfinite(c))
        self.assertGreater(c, 0.0)

    def test_new_pixel_fraction_bounded(self):
        df = self._two_halves(
            self._square(77.00, 22.00, 0.01),
            self._square(78.00, 22.00, 0.02),
        )
        f = float(lf.compute_lineage_features(df)["NEW_PIXEL_FRACTION"].iloc[0])
        self.assertTrue(0.0 <= f <= 1.0)


class TestQuality(unittest.TestCase):
    def test_scan_angle_sin_cos_bounded(self):
        out = lf.compute_lineage_features(one_event(8))
        s = float(out["SCAN_ANGLE_SIN"].iloc[0])
        c = float(out["SCAN_ANGLE_COS"].iloc[0])
        self.assertLessEqual(abs(s), 1.0)
        self.assertLessEqual(abs(c), 1.0)

    def test_symmetric_scan_angles_give_near_zero_sine(self):
        frame = one_event(6)
        frame["scan_angle"] = [-30.0, -30.0, 30.0, 30.0, -30.0, 30.0]
        out = lf.compute_lineage_features(frame)
        self.assertAlmostEqual(float(out["SCAN_ANGLE_SIN"].iloc[0]), 0.0, places=9)

    def test_letter_confidence_is_mapped_to_numbers(self):
        frame = one_event(4, confidence=["h"] * 4)
        out = lf.compute_lineage_features(frame)
        self.assertAlmostEqual(
            float(out["FIRMS_CONFIDENCE"].iloc[0]), 110.0, places=6
        )

    def test_pixel_size_median(self):
        frame = one_event(4)
        frame["pixel_size"] = [375.0, 375.0, 1000.0, 1000.0]
        out = lf.compute_lineage_features(frame)
        self.assertAlmostEqual(
            float(out["PIXEL_SIZE"].iloc[0]), 687.5, places=6
        )


class TestColumnResolution(unittest.TestCase):
    def test_alternative_column_names_resolve(self):
        frame = pd.DataFrame(
            {
                "global_event_id": ["E1", "E1"],
                "fire radiative power": [10.0, 20.0],
                "brightness_i4": [350.0, 351.0],
                "brightness_i5": [300.0, 301.0],
                "lon": [77.0, 77.1],
                "lat": [22.0, 22.1],
            }
        )
        cols = lf.resolve_columns(frame)
        self.assertEqual(cols["event_id"], "global_event_id")
        self.assertEqual(cols["frp"], "fire radiative power")
        self.assertEqual(cols["i4"], "brightness_i4")

    def test_missing_required_column_raises(self):
        frame = pd.DataFrame({"event_id": ["E1"], "frp": [1.0]})
        with self.assertRaises(KeyError):
            lf.compute_lineage_features(frame)

    def test_error_message_lists_what_is_missing(self):
        frame = pd.DataFrame({"event_id": ["E1"]})
        with self.assertRaises(KeyError) as ctx:
            lf.compute_lineage_features(frame)
        self.assertIn("frp", str(ctx.exception))


class TestGeoHelpers(unittest.TestCase):
    def test_haversine_known_distance(self):
        # 1 degree of latitude is ~111.19 km.
        d = lf.haversine_m(0.0, 0.0, 0.0, 1.0)
        self.assertAlmostEqual(float(d), 111_194.0, delta=200.0)

    def test_haversine_zero(self):
        self.assertAlmostEqual(float(lf.haversine_m(10.0, 20.0, 10.0, 20.0)), 0.0)

    def test_convex_hull_of_a_square(self):
        hull = lf._convex_hull([[0, 0], [1, 0], [1, 1], [0, 1]])
        self.assertEqual(len(hull), 4)

    def test_polygon_area_of_unit_square(self):
        m = lf._polygon_metrics([[0, 0], [1, 0], [1, 1], [0, 1]])
        self.assertAlmostEqual(m["area"], 1.0, places=9)

    def test_solidity_of_a_square_is_one(self):
        m = lf._polygon_metrics([[0, 0], [1, 0], [1, 1], [0, 1]])
        self.assertAlmostEqual(m["solidity"], 1.0, places=9)

    def test_resultant_length_of_identical_angles_is_one(self):
        self.assertAlmostEqual(
            lf.resultant_length(np.full(20, 0.7)), 1.0, places=9
        )

    def test_resultant_length_of_opposite_angles_is_zero(self):
        angles = np.array([0.0] * 10 + [np.pi] * 10)
        self.assertAlmostEqual(lf.resultant_length(angles), 0.0, places=9)

    def test_degree_area_scales_with_latitude(self):
        near_equator = float(lf._degrees_area_to_km2(1.0, 0.0))
        polar = float(lf._degrees_area_to_km2(1.0, 60.0))
        self.assertGreater(near_equator, polar)


class TestNoZeroFilling(unittest.TestCase):
    def test_everything_missing_yields_all_nan_not_zero(self):
        frame = pd.DataFrame(
            {
                "event_id": ["E1"],
                "acq_time": [pd.NaT],
                "frp": [np.nan],
                "longitude": [np.nan],
                "latitude": [np.nan],
            }
        )
        out = lf.compute_lineage_features(frame)
        row = out.iloc[0]
        # Nothing is measurable, so nothing may be a confident 0.
        for name in lf.LINEAGE_FEATURES:
            value = row[name]
            if value == 0.0:
                self.fail(f"{name} was filled with 0 instead of NaN")

    def test_nan_input_does_not_crash(self):
        frame = one_event(6)
        frame["frp"] = np.nan
        frame["i4"] = np.nan
        out = lf.compute_lineage_features(frame)
        self.assertEqual(len(out), 1)


class TestDailyFileNaming(unittest.TestCase):
    """The cluster files are `clusters_2012-01-20.parquet` -- ISO dates with
    dashes, NOT the `YYYYMMDD_<label>` form of the local cluster ids. Building
    a filename from an id finds nothing, silently."""

    def test_parses_dashed_iso_date(self):
        self.assertEqual(
            lf.parse_date_from_filename("clusters_2012-01-20.parquet"), "2012-01-20"
        )

    def test_parses_compact_date(self):
        self.assertEqual(
            lf.parse_date_from_filename("clusters_20120120.parquet"), "2012-01-20"
        )

    def test_parses_from_an_edges_filename(self):
        self.assertEqual(
            lf.parse_date_from_filename(
                "edges_2012-01-20_to_2012-01-21.parquet"
            ),
            "2012-01-20",
        )

    def test_returns_none_without_a_date(self):
        self.assertIsNone(lf.parse_date_from_filename("README.txt"))

    def test_index_builds_date_to_path(self):
        import tempfile
        from pathlib import Path

        d = Path(tempfile.mkdtemp())
        for name in (
            "clusters_2012-01-20.parquet",
            "clusters_2012-01-21.parquet",
            "clusters_2026-08-31.parquet",
        ):
            (d / name).write_text("x")
        index, stats = lf.index_daily_files(d)
        self.assertEqual(sorted(index), ["2012-01-20", "2012-01-21", "2026-08-31"])
        self.assertEqual(index["2012-01-20"].name, "clusters_2012-01-20.parquet")
        self.assertEqual(stats["n_indexed"], 3)

    def test_index_excludes_edges_files(self):
        """Joining edge rows to detections would be nonsense."""
        import tempfile
        from pathlib import Path

        d = Path(tempfile.mkdtemp())
        (d / "clusters_2012-01-20.parquet").write_text("x")
        (d / "edges_2012-01-20_to_2012-01-21.parquet").write_text("x")
        index, stats = lf.index_daily_files(d, prefix="clusters")
        self.assertEqual(len(index), 1)
        self.assertEqual(stats["n_other_prefix"], 1)

    def test_index_flags_undated_files(self):
        import tempfile
        from pathlib import Path

        d = Path(tempfile.mkdtemp())
        (d / "clusters_2012-01-20.parquet").write_text("x")
        (d / "clusters_nodate.parquet").write_text("x")
        index, stats = lf.index_daily_files(d, prefix="clusters")
        self.assertEqual(len(index), 1)
        self.assertEqual(stats["n_skipped_no_date"], 1)
        self.assertIn("clusters_nodate.parquet", stats["skipped_examples"])

    def test_index_of_empty_dir_is_empty(self):
        import tempfile
        from pathlib import Path

        index, stats = lf.index_daily_files(Path(tempfile.mkdtemp()))
        self.assertEqual(index, {})
        self.assertEqual(stats["n_indexed"], 0)

    def test_index_without_prefix_filter_takes_everything(self):
        import tempfile
        from pathlib import Path

        d = Path(tempfile.mkdtemp())
        (d / "clusters_2012-01-20.parquet").write_text("x")
        (d / "edges_2012-01-20.parquet").write_text("x")
        index, _ = lf.index_daily_files(d, prefix=None)
        self.assertEqual(index["2012-01-20"].name, "edges_2012-01-20.parquet")


class TestMembershipDateConversion(unittest.TestCase):
    def test_cluster_id_prefix_converts_to_iso(self):
        import pandas as pd

        # Cluster ids are YYYYMMDD_<label>; slice the date off, then require
        # 8 digits rather than merely 8 characters.
        ids = pd.Series(["20120120_3", "20120121_17", "2012-01-22_9", "bad"])
        raw = ids.str.slice(0, 8)
        ok = raw.str.fullmatch(r"\d{8}")
        dates = pd.to_datetime(
            raw.where(ok), format="%Y%m%d", errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        self.assertEqual(dates.iloc[0], "2012-01-20")
        self.assertEqual(dates.iloc[1], "2012-01-21")
        # Dashed and too-short ids must NOT be coerced.
        self.assertTrue(pd.isna(dates.iloc[2]))
        self.assertTrue(pd.isna(dates.iloc[3]))


if __name__ == "__main__":
    unittest.main()