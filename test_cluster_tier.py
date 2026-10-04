"""
Tests for the P4a tier-1 (cluster-level) lineage extractor.

The finding that motivated this module: `clusters/*.parquet` holds
CLUSTER-level aggregates, not detections. There is no per-detection frp, i4, i5
or confidence there. So 34 of the 54 lineage features are computable from
clusters/ + membership alone, and 20 are not.

Run:  python3 -m unittest test_cluster_tier -v
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import cluster_tier as ct
import lineage_features as lf
import lineage_features as lf


def clusters_for(event_id, n_clusters, *, frp=None, x=None, y=None,
                 start="2021-03-01", step_days=1, radius_km=None, n_det=None):
    """Synthetic cluster rows matching the REAL clusters/ schema."""
    t0 = pd.Timestamp(start)
    idx = np.arange(n_clusters)
    frame = pd.DataFrame(
        {
            "event_id": event_id,
            "local_cluster_id": [f"2021030{i}_{i}" for i in idx],
            "start_time": [t0 + pd.Timedelta(days=int(i * step_days))
                           for i in idx],
            "n_detections": n_det if n_det is not None else np.full(n_clusters, 5),
            "centroid_x": x if x is not None else np.full(n_clusters, 7_500_000.0),
            "centroid_y": y if y is not None else np.full(n_clusters, 1_500_000.0),
            "total_frp": frp if frp is not None else np.full(n_clusters, 50.0),
            "mean_frp": np.full(n_clusters, 10.0),
            "max_frp": frp if frp is not None else np.full(n_clusters, 20.0),
            "duration_hours": np.full(n_clusters, 6.0),
            "n_satellites": np.full(n_clusters, 2),
            "max_radius_km": radius_km if radius_km is not None
            else np.full(n_clusters, 1.0),
        }
    )
    return frame


class TestTierPartition(unittest.TestCase):
    def test_tier1_is_a_subset_of_the_lineage_block(self):
        self.assertEqual(ct.N_CLUSTER_TIER, 34)
        self.assertTrue(set(ct.CLUSTER_TIER_FEATURES) <= set(lf.LINEAGE_FEATURES))

    def test_no_duplicate_tier1_features(self):
        self.assertEqual(len(set(ct.CLUSTER_TIER_FEATURES)), ct.N_CLUSTER_TIER)

    def test_every_lineage_feature_is_accounted_for(self):
        """34 computed + 20 explained == 54, with none unaccounted."""
        self.assertEqual(ct.UNACCOUNTED, [])
        self.assertEqual(
            ct.N_CLUSTER_TIER + len(ct.DEFERRED_REASONS),
            len(lf.LINEAGE_FEATURES),
        )

    def test_tier1_order_follows_the_ledger(self):
        positions = [lf.LINEAGE_FEATURES.index(n) for n in ct.CLUSTER_TIER_FEATURES]
        self.assertEqual(positions, sorted(positions))

    def test_deferred_reasons_are_populated(self):
        for name, reason in ct.DEFERRED_REASONS.items():
            self.assertTrue(reason.strip(), name)

    def test_detection_only_features_are_deferred(self):
        for name in ("BT_I4_MEDIAN", "FRP_MEDIAN"):
            self.assertIn(name, lf.LINEAGE_FEATURES)
        self.assertNotIn("BT_I4_MEDIAN", ct.CLUSTER_TIER_FEATURES)


class TestOutputContract(unittest.TestCase):
    def test_columns_are_exactly_tier1_in_order(self):
        out = ct.compute_cluster_tier_features(clusters_for("E1", 6))
        self.assertEqual(list(out.columns), ct.CLUSTER_TIER_FEATURES)

    def test_one_row_per_event(self):
        df = pd.concat(
            [clusters_for("E1", 6), clusters_for("E2", 4)], ignore_index=True
        )
        out = ct.compute_cluster_tier_features(df)
        self.assertEqual(set(out.index), {"E1", "E2"})

    def test_index_is_named_event_id(self):
        out = ct.compute_cluster_tier_features(clusters_for("E1", 4))
        self.assertEqual(out.index.name, "event_id")

    def test_missing_required_column_raises(self):
        frame = clusters_for("E1", 4).drop(columns=["max_frp"])
        with self.assertRaises(KeyError):
            ct.compute_cluster_tier_features(frame)


class TestAggregates(unittest.TestCase):
    def test_frp_max_and_median(self):
        out = ct.compute_cluster_tier_features(
            clusters_for("E1", 4, frp=np.array([10.0, 20.0, 30.0, 40.0]))
        )
        self.assertAlmostEqual(float(out["FRP_MAX"].iloc[0]), 40.0, places=6)
        self.assertAlmostEqual(float(out["FRP_MEDIAN"].iloc[0]), 25.0, places=6)

    def test_frp_sum_log_uses_total_frp(self):
        out = ct.compute_cluster_tier_features(
            clusters_for("E1", 4, frp=np.array([10.0, 10.0, 10.0, 10.0]))
        )
        self.assertAlmostEqual(
            float(out["FRP_SUM_LOG"].iloc[0]), np.log1p(40.0), places=6
        )

    def test_detection_count_log_uses_n_detections(self):
        out = ct.compute_cluster_tier_features(
            clusters_for("E1", 4, n_det=np.array([1, 2, 3, 4]))
        )
        self.assertAlmostEqual(
            float(out["DETECTION_COUNT_LOG"].iloc[0]), np.log1p(10.0), places=6
        )

    def test_duration_log_from_cluster_time_span(self):
        out = ct.compute_cluster_tier_features(
            clusters_for("E1", 3, step_days=1)
        )
        # 3 clusters 1 day apart -> 48 h span.
        self.assertAlmostEqual(
            float(out["DURATION_HOURS_LOG"].iloc[0]), np.log1p(48.0), places=4
        )

    def test_peak_median_ratio(self):
        out = ct.compute_cluster_tier_features(
            clusters_for("E1", 4, frp=np.array([10.0, 10.0, 10.0, 40.0]))
        )
        self.assertAlmostEqual(
            float(out["FRP_PEAK_MEDIAN_RATIO"].iloc[0]), 4.0, places=6
        )

    def test_density_uses_the_bounding_circle(self):
        out = ct.compute_cluster_tier_features(
            clusters_for("E1", 2, n_det=np.array([10, 10]), radius_km=1.0)
        )
        expected = 20.0 / (np.pi * 1.0 ** 2)
        self.assertAlmostEqual(
            float(out["DETECTION_DENSITY"].iloc[0]), expected, places=3
        )

    def test_max_radius_km_is_converted_to_metres(self):
        # 1 km radius in a _km column must behave like 1000 m in an _m one.
        km = clusters_for("E1", 2, radius_km=1.0)
        m = km.rename(columns={"max_radius_km": "max_radius_m"})
        m["max_radius_m"] = 1000.0
        a = ct.compute_cluster_tier_features(km)
        b = ct.compute_cluster_tier_features(m)
        self.assertAlmostEqual(
            float(a["DETECTION_DENSITY"].iloc[0]),
            float(b["DETECTION_DENSITY"].iloc[0]),
            places=6,
        )


class TestUndefinedIsNaN(unittest.TestCase):
    def test_single_cluster_has_no_trend(self):
        out = ct.compute_cluster_tier_features(clusters_for("E1", 1))
        for name in ("FRP_SLOPE", "FRP_DECAY_RATE", "FRP_AUTOCORRELATION",
                     "SPREAD_SPEED_MEDIAN", "AREA_GROWTH_RATE", "COMPACTNESS",
                     "SOLIDITY", "ELONGATION"):
            self.assertTrue(np.isnan(float(out[name].iloc[0])), name)

    def test_single_cluster_stationarity_is_one(self):
        # Never moved is the most stationary case, not an undefined one.
        out = ct.compute_cluster_tier_features(clusters_for("E1", 1))
        self.assertAlmostEqual(
            float(out["STATIONARITY_INDEX"].iloc[0]), 1.0, places=6
        )

    def test_collinear_clusters_have_undefined_shape(self):
        n = 6
        x = np.linspace(7_500_000.0, 7_500_500.0, n)
        out = ct.compute_cluster_tier_features(
            clusters_for("E1", n, x=x, y=np.full(n, 1_500_000.0))
        )
        for name in ("COMPACTNESS", "SOLIDITY", "NEW_PIXEL_FRACTION"):
            self.assertTrue(np.isnan(float(out[name].iloc[0])), name)

    def test_trend_features_are_never_zero_filled(self):
        """A single cluster has no trend, and a zero would be a lie.

        Note this deliberately excludes FRP_IQR, DURATION_HOURS_LOG and
        TIME_TO_PEAK: for one cluster those are *genuinely* zero (IQR of one
        value is 0, the span is 0 h, the peak is at t=0), not missing data.
        """
        out = ct.compute_cluster_tier_features(clusters_for("E1", 1))
        must_be_nan = [
            "FRP_SLOPE", "FRP_DECAY_RATE", "FRP_AUTOCORRELATION",
            "SPREAD_SPEED_MEDIAN", "SPREAD_SPEED_P90", "SPREAD_SPEED_MAX",
            "CENTROID_DISPLACEMENT_P90", "AREA_GROWTH_RATE",
            "PERIMETER_GROWTH_RATE", "COMPACTNESS", "SOLIDITY",
            "NEW_PIXEL_FRACTION", "ELONGATION",
        ]
        for name in must_be_nan:
            self.assertTrue(np.isnan(float(out[name].iloc[0])), name)

    def test_genuine_zeros_are_allowed(self):
        """IQR of a single cluster really is zero; that is not a fill."""
        out = ct.compute_cluster_tier_features(clusters_for("E1", 1))
        self.assertAlmostEqual(float(out["FRP_IQR"].iloc[0]), 0.0, places=9)
        self.assertAlmostEqual(
            float(out["DURATION_HOURS_LOG"].iloc[0]), 0.0, places=9
        )


class TestSpatial(unittest.TestCase):
    def _track(self, xs, ys):
        n = len(xs)
        return clusters_for(
            "E1", n, x=np.array(xs, dtype="float64"),
            y=np.array(ys, dtype="float64"), step_days=1,
        )

    def test_moving_event_has_positive_speed(self):
        out = ct.compute_cluster_tier_features(
            self._track(np.linspace(7_500_000, 7_504_000, 6),
                        np.full(6, 1_500_000.0))
        )
        self.assertGreater(float(out["SPREAD_SPEED_MEDIAN"].iloc[0]), 0.0)

    def test_stationary_event_speed_is_zero(self):
        out = ct.compute_cluster_tier_features(self._track([7.5e6] * 6, [1.5e6] * 6))
        self.assertAlmostEqual(
            float(out["SPREAD_SPEED_MEDIAN"].iloc[0]), 0.0, places=6
        )

    def test_straight_track_direction_consistency_near_one(self):
        out = ct.compute_cluster_tier_features(
            self._track(np.linspace(7_500_000, 7_504_000, 8),
                        np.full(8, 1_500_000.0))
        )
        self.assertGreater(float(out["DIRECTION_CONSISTENCY"].iloc[0]), 0.95)

    @staticmethod
    def _square(cx, cy, half):
        """Four corners of an axis-aligned square -- a non-degenerate hull."""
        return [
            [cx - half, cy - half], [cx + half, cy - half],
            [cx + half, cy + half], [cx - half, cy + half],
        ]

    def _growing_boxes(self, reverse=False):
        """Early clusters in a tight square, later ones in a wider square.

        Each half must be a genuine square of a *different* size. A strip of
        constant width simply translated has equal hull area in both halves and
        yields exactly 0.0 growth, proving nothing; and a "box" built from two
        repeated points has a degenerate hull, which yields NaN.
        """
        tight = self._square(7_500_000.0, 1_500_000.0, 1.0)
        wide = self._square(7_500_000.0, 1_500_000.0, 5.0)
        first, second = (wide, tight) if reverse else (tight, wide)
        # Pad each half to an even count so the time-ordered midpoint splits
        # exactly on the boundary the fixture intends.
        pts = first * 2 + second * 2
        return [p[0] for p in pts], [p[1] for p in pts]

    def test_expanding_track_has_positive_area_growth(self):
        xs, ys = self._growing_boxes()
        out = ct.compute_cluster_tier_features(self._track(xs, ys))
        g = float(out["AREA_GROWTH_RATE"].iloc[0])
        self.assertTrue(np.isfinite(g), "area growth undefined on an expanding track")
        self.assertGreater(g, 0.0)

    def test_shrinking_track_has_negative_area_growth(self):
        xs, ys = self._growing_boxes(reverse=True)
        out = ct.compute_cluster_tier_features(self._track(xs, ys))
        g = float(out["AREA_GROWTH_RATE"].iloc[0])
        self.assertTrue(np.isfinite(g), "area growth undefined on a shrinking track")
        self.assertLess(g, 0.0)


class TestTemporalRules(unittest.TestCase):
    def test_persistence_is_measured_over_a_fixed_window(self):
        """Regression. Persistence used to be `days / (span_days + 1)`, which is
        1.0 for EVERY event because an event only exists while it is detected.
        Measured over a fixed window it discriminates by duration instead."""
        short = ct.compute_cluster_tier_features(
            clusters_for("E1", 2, step_days=1)
        )
        long_ = ct.compute_cluster_tier_features(
            clusters_for("E2", 20, step_days=1)
        )
        s = float(short["FRP_PERSISTENCE"].iloc[0])
        l = float(long_["FRP_PERSISTENCE"].iloc[0])
        self.assertLess(s, 0.2, f"a 2-day event scored {s}")
        self.assertGreater(l, s, "a longer event must score higher")
        self.assertAlmostEqual(
            l, 20.0 / float(lf.PERSISTENCE_WINDOW_DAYS), places=6
        )

    def test_persistence_is_not_constant_across_durations(self):
        """The specific failure: every event scored exactly 1.0."""
        values = []
        for n in (1, 2, 5, 15, 40):
            out = ct.compute_cluster_tier_features(
                clusters_for(f"E{n}", n, step_days=1)
            )
            values.append(float(out["FRP_PERSISTENCE"].iloc[0]))
        self.assertGreater(len(set(np.round(values, 6))), 1,
                           f"persistence is still constant: {values}")

    def test_persistence_reaches_one_for_a_long_event(self):
        window = float(lf.PERSISTENCE_WINDOW_DAYS)
        out = ct.compute_cluster_tier_features(
            clusters_for("E1", int(window) + 2, step_days=1)
        )
        self.assertAlmostEqual(float(out["FRP_PERSISTENCE"].iloc[0]), 1.0,
                               places=6)

    def test_persistence_window_is_a_module_constant_not_a_column(self):
        """It describes how the feature was computed, not an event, and the
        output is reindexed onto CLUSTER_TIER_FEATURES -- so it must not
        appear as a column."""
        self.assertIsInstance(lf.PERSISTENCE_WINDOW_DAYS, int)
        self.assertGreater(lf.PERSISTENCE_WINDOW_DAYS, 0)
        out = ct.compute_cluster_tier_features(
            clusters_for("E1", 5, step_days=1)
        )
        self.assertNotIn("PERSISTENCE_WINDOW_DAYS", out.columns)
        self.assertEqual(list(out.columns), ct.CLUSTER_TIER_FEATURES)

    def test_persistence_never_exceeds_one(self):
        for n in (2, 3, 5, 10, 45):
            out = ct.compute_cluster_tier_features(
                clusters_for("E1", n, step_days=1)
            )
            p = float(out["FRP_PERSISTENCE"].iloc[0])
            self.assertLessEqual(p, 1.0 + 1e-9, f"n={n} gave {p}")

    def test_peak_hour_is_circular(self):
        frame = pd.concat(
            [
                clusters_for("E1", 1, start="2021-03-01T23:00:00"),
                clusters_for("E1", 1, start="2021-03-02T00:00:00"),
            ],
            ignore_index=True,
        )
        out = ct.compute_cluster_tier_features(frame)
        resultant = float(
            np.hypot(out["PEAK_HOUR_SIN"].iloc[0], out["PEAK_HOUR_COS"].iloc[0])
        )
        self.assertGreater(resultant, 0.9)


class TestDtypesAreWritable(unittest.TestCase):
    """Every tier-1 column must be float64.

    Regression: `safe_ratio` returns a 0-d numpy array for scalar inputs, so a
    dict of those values gave `pd.Series` dtype `object`, and pyarrow refused
    the column with "Can only convert 1-dimensional array values". An all-NaN
    column stays float64 and writes fine, so the failure only appeared at batch
    36 of 90 on the first full run -- long after every unit test had passed.
    """

    def test_all_columns_are_float64(self):
        frame = pd.concat(
            [clusters_for("E1", 8), clusters_for("E2", 3), clusters_for("E3", 1)],
            ignore_index=True,
        )
        out = ct.compute_cluster_tier_features(frame)
        for name in ct.CLUSTER_TIER_FEATURES:
            self.assertTrue(
                pd.api.types.is_float_dtype(out[name]),
                f"{name} is {out[name].dtype}, not float64",
            )

    def test_a_single_real_value_flips_nothing_to_object(self):
        """The exact shape that broke: mostly-NaN with one real value.

        Needs a genuinely 2-D track, because collinear points give a degenerate
        hull and a NaN growth rate, which would not reproduce the bug.
        """
        n = 6
        # Each half of the time-ordered split must be a non-degenerate
        # triangle: a collinear half has no hull and yields NaN growth.
        xs = [7_500_000.0 - 1, 7_500_000.0 + 1, 7_500_000.0 + 1,
              7_500_000.0 - 5, 7_500_000.0 + 5, 7_500_000.0 + 5]
        ys = [1_500_000.0 - 1, 1_500_000.0 - 1, 1_500_000.0 + 1,
              1_500_000.0 - 5, 1_500_000.0 - 5, 1_500_000.0 + 5]
        real = clusters_for("E2", n, x=np.array(xs), y=np.array(ys), step_days=1)
        frame = pd.concat(
            [clusters_for("E1", 1), real], ignore_index=True
        )
        out = ct.compute_cluster_tier_features(frame)
        self.assertGreaterEqual(out["AREA_GROWTH_RATE"].notna().sum(), 1)
        self.assertTrue(pd.api.types.is_float_dtype(out["AREA_GROWTH_RATE"]))
        self.assertTrue(pd.api.types.is_float_dtype(out["SOLIDITY"]))

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("pyarrow"),
        "pyarrow is Colab-only",
    )
    def test_output_is_writable_to_parquet(self):
        """The literal call that failed in production."""
        import tempfile

        frame = pd.concat(
            [clusters_for("E1", 8), clusters_for("E2", 1)], ignore_index=True
        )
        out = ct.compute_cluster_tier_features(frame).reset_index()
        with tempfile.TemporaryDirectory() as tmp:
            out.to_parquet(f"{tmp}/chunk.parquet", index=False)
            back = pd.read_parquet(f"{tmp}/chunk.parquet")
        self.assertEqual(list(back.columns), list(out.columns))

    def test_scalar_coercion_rejects_non_scalars(self):
        with self.assertRaises(ValueError):
            ct._f(np.array([1.0, 2.0]))


class TestIndexRobustness(unittest.TestCase):
    """A merge leaves a non-RangeIndex, and label-based positional indexing
    then reads the wrong rows while still returning plausible numbers."""

    def test_results_are_identical_under_a_shifted_index(self):
        frame = pd.concat(
            [clusters_for("E1", 6), clusters_for("E2", 4)], ignore_index=True
        )
        baseline = ct.compute_cluster_tier_features(frame)
        shifted = frame.copy()
        shifted.index = range(1000, 1000 + len(shifted))
        moved = ct.compute_cluster_tier_features(shifted)
        pd.testing.assert_frame_equal(baseline, moved)

    def test_reversed_index_gives_the_same_answer(self):
        frame = clusters_for("E1", 6)
        baseline = ct.compute_cluster_tier_features(frame)
        shuffled = frame.iloc[::-1]
        out = ct.compute_cluster_tier_features(shuffled)
        for name in ct.CLUSTER_TIER_FEATURES:
            a, b = baseline[name].iloc[0], out[name].iloc[0]
            if np.isnan(a) and np.isnan(b):
                continue
            self.assertAlmostEqual(float(a), float(b), places=6, msg=name)

    def test_overlapping_date_columns_do_not_break_the_join(self):
        # clusters/ and membership/ BOTH have a `date` column. A naive join
        # raises "columns overlap but no suffix specified".
        clusters = clusters_for("E1", 4)
        membership = pd.DataFrame(
            {
                "event_id": ["E1"] * 4,
                "local_cluster_id": clusters["local_cluster_id"].tolist(),
                "date": clusters["date"].tolist()
                if "date" in clusters.columns
                else ["2021-03-01"] * 4,
            }
        ).set_index("local_cluster_id")
        joined = clusters.join(membership[["event_id"]], on="local_cluster_id",
                               how="inner", rsuffix="_mem")
        out = ct.compute_cluster_tier_features(joined, event_col="event_id")
        self.assertEqual(len(out), 1)


class TestReport(unittest.TestCase):
    def test_report_covers_every_tier1_feature(self):
        out = ct.compute_cluster_tier_features(clusters_for("E1", 6))
        rep = ct.cluster_tier_report(out)
        self.assertEqual(rep["n_events"], 1)
        for name in ct.CLUSTER_TIER_FEATURES:
            self.assertIn(name, rep)

    def test_report_of_empty_is_safe(self):
        self.assertEqual(ct.cluster_tier_report(None), {"n_events": 0})


if __name__ == "__main__":
    unittest.main()