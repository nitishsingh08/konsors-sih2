"""
Tests for rule calibration.

A threshold is only meaningful relative to the distribution it is applied to.
`FRP_PERSISTENCE` moved from a constant 1.0 to mostly 0.033, which silently
made every fixed threshold too strict -- INDUSTRIAL_FLARE went 3,571 -> 0 and
MINING 73,060 -> 61. Calibration derives thresholds from percentiles or from
target class counts instead.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import rule_calibration as rc
import weak_labels as wl


def catalogue(n=200_000, seed=0, near_share=0.25):
    """Shaped like the real X: 96.7% single-cluster persistence floor."""
    rng = np.random.default_rng(seed)
    single = rng.random(n) < 0.967
    days = np.where(single, 1.0, rng.integers(1, 60, n).astype(float))
    near = rng.random(n) < near_share
    far = np.log10(1 + rng.uniform(5e4, 2e6, n))
    return pd.DataFrame({
        "KNOWN_FLARE": np.zeros(n), "INSIDE_FACILITY": np.zeros(n),
        "DIST_REFINERY_LOG": np.where(near, np.log10(1 + rng.gamma(2, 400, n)), far),
        "DIST_NEAREST_FACILITY_LOG": far, "FACILITY_COUNT_5KM": np.zeros(n),
        "DIST_POWERPLANT_LOG": far, "DIST_PETROCHEMICAL_LOG": far,
        "DIST_STEEL_LOG": far, "DIST_CEMENT_LOG": far,
        "DIST_MINE_LOG": np.where(near, np.log10(1 + rng.gamma(2, 700, n)), far),
        "FRP_PERSISTENCE": np.minimum(days / 30.0, 1.0),
        "DETECTION_COUNT_LOG": np.log1p(
            np.where(single, 1.0, rng.integers(2, 40, n).astype(float))),
        "TIME_TO_PEAK": rng.uniform(0, 72, n),
        "AREA_GROWTH_RATE": np.where(single, np.nan, rng.normal(0.1, 0.5, n)),
        "SPREAD_SPEED_MEDIAN": np.where(single, np.nan, rng.gamma(2, 3, n)),
        "DIRECTION_CONSISTENCY": np.where(single, np.nan, rng.uniform(0, 1, n)),
    })


class TestDistribution(unittest.TestCase):
    def test_reports_percentiles(self):
        table = rc.distribution(catalogue(20000), ["FRP_PERSISTENCE"])
        self.assertEqual(len(table), 1)
        row = table.iloc[0]
        self.assertEqual(row["n_finite"], 20000)
        self.assertLessEqual(row["p50"], row["p99"])

    def test_skips_absent_columns(self):
        table = rc.distribution(catalogue(5000), ["NOT_A_COLUMN"])
        self.assertEqual(len(table), 0)

    def test_shows_the_single_cluster_floor(self):
        table = rc.distribution(catalogue(50000), ["FRP_PERSISTENCE"])
        row = table.iloc[0]
        self.assertAlmostEqual(row["p50"], 1.0 / 30.0, places=3)


class TestCalibrate(unittest.TestCase):
    def test_thresholds_come_from_the_data(self):
        frame = catalogue(50000)
        thresholds, evidence = rc.calibrate(frame)
        for attr, column, _ in (
            ("flare_persistence", "FRP_PERSISTENCE", None),
            ("mine_persistence", "FRP_PERSISTENCE", None),
        ):
            self.assertTrue(0.0 <= getattr(thresholds, attr) <= 1.0, attr)
        self.assertEqual(len(evidence), 4)

    def test_default_percentiles_sit_above_the_single_cluster_mass(self):
        """Below 0.967 every percentile collapses onto the persistence floor and
        every rule becomes the same rule."""
        for name, p in rc.DEFAULT_PERCENTILES.items():
            self.assertGreater(p, 0.967, f"{name}={p} is inside the floor mass")

    def test_evidence_reports_what_each_threshold_admits(self):
        frame = catalogue(50000)
        _th, evidence = rc.calibrate(frame)
        for item in evidence:
            self.assertGreaterEqual(item.n_above, 0)
            self.assertLessEqual(item.share_above, 1.0)

    def test_missing_column_falls_back_without_raising(self):
        frame = catalogue(20000).drop(columns=["FRP_PERSISTENCE"])
        thresholds, _ = rc.calibrate(frame)
        self.assertIsInstance(thresholds, wl.WeakThresholds)


class TestTargetCounts(unittest.TestCase):
    def test_returns_a_threshold_set(self):
        frame = catalogue(40000)
        thresholds, evidence = rc.calibrate_to_target_counts(
            frame, {wl.INDUSTRIAL_FLARE: 1000, wl.MINING: 3000}
        )
        self.assertIsInstance(thresholds, wl.WeakThresholds)
        self.assertEqual(len(evidence), 4)

    def test_calibration_does_not_reduce_a_collapsed_class(self):
        """The regression this module exists to prevent: a threshold set that
        admits fewer events than the uncalibrated one."""
        frame = catalogue(60000)
        before = wl.assign_weak_labels(frame, wl.WIDE)
        n_before = int((before["label"] == wl.MINING).sum())
        thresholds, _ = rc.calibrate_to_target_counts(
            frame, {wl.MINING: max(n_before, 1)})
        after = wl.assign_weak_labels(frame, thresholds)
        self.assertGreaterEqual(int((after["label"] == wl.MINING).sum()), n_before)

    def test_zero_target_is_ignored(self):
        frame = catalogue(20000)
        thresholds, _ = rc.calibrate_to_target_counts(
            frame, {wl.INDUSTRIAL_FLARE: 0})
        self.assertIsInstance(thresholds, wl.WeakThresholds)

    def test_report_returns_thresholds_table_and_labels(self):
        frame = catalogue(30000)
        thresholds, evidence = rc.calibrate(frame)
        table, report, out = rc.calibration_report(frame, thresholds, evidence)
        self.assertEqual(set(report.columns), set(wl.WEAK_CLASSES) | {"variant"})
        self.assertEqual(len(out["label"]), len(frame))


class TestSerialisation(unittest.TestCase):
    def test_round_trips(self):
        frame = catalogue(20000)
        thresholds, _ = rc.calibrate(frame)
        payload = rc.thresholds_to_dict(thresholds)
        restored = rc.thresholds_from_dict(payload)
        self.assertEqual(
            restored.flare_persistence, thresholds.flare_persistence
        )

    def test_unknown_keys_are_ignored_on_load(self):
        restored = rc.thresholds_from_dict(
            {"flare_persistence": 0.9, "a_field_that_no_longer_exists": 1}
        )
        self.assertAlmostEqual(restored.flare_persistence, 0.9)


if __name__ == "__main__":
    unittest.main()
