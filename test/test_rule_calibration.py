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
        admits fewer events than the uncalibrated one.

        `population_size=len(frame)` is REQUIRED here. Without it the target is
        read as a catalogue-level count and scaled onto this frame -- 40 events
        against a 60,000-row frame becomes a target of 1, so calibration
        deliberately shrinks the class and the assertion below is meaningless.
        That is the same mistake P6 was making, which is why the parameter is
        explicit in the test rather than defaulted away.
        """
        frame = catalogue(60000)
        before = wl.assign_weak_labels(frame, wl.WIDE)
        n_before = int((before["label"] == wl.MINING).sum())
        thresholds, _ = rc.calibrate_to_target_counts(
            frame, {wl.MINING: max(n_before, 1)},
            population_size=len(frame),
        )
        after = wl.assign_weak_labels(frame, thresholds)
        self.assertGreaterEqual(int((after["label"] == wl.MINING).sum()), n_before)


class TestTargetScaling(unittest.TestCase):
    """THE BUG: catalogue-scale targets were applied to a 200,000-row sample.

    `CALIB_TARGETS` are written against 7,265,849 events. P6 PHASE 2 passes a
    200,000-row sample. Two of the three targets exceed the sample size, so
    `admitted(floor) < target` fired immediately for every class and every
    threshold was pinned to the floor of `observed_days / 30`. The persistence
    branch then became vacuous and the whole weak-label hierarchy collapsed
    into two distance terms.
    """

    def test_a_target_larger_than_the_frame_is_scaled_not_collapsed(self):
        scaled, was = rc.scale_targets(
            {"INDUSTRIAL_FIRE": 400_000}, n_rows=200_000
        )
        self.assertTrue(was)
        self.assertLess(scaled["INDUSTRIAL_FIRE"], 200_000)
        self.assertGreaterEqual(scaled["INDUSTRIAL_FIRE"], 1)

    def test_full_scale_frame_is_left_alone(self):
        scaled, was = rc.scale_targets(
            {"INDUSTRIAL_FIRE": 400_000}, n_rows=7_265_849
        )
        self.assertFalse(was)
        self.assertEqual(scaled["INDUSTRIAL_FIRE"], 400_000)

    def test_explicit_population_is_honoured(self):
        scaled, was = rc.scale_targets(
            {"X": 1000}, n_rows=100, population_size=1000
        )
        self.assertTrue(was)
        self.assertEqual(scaled["X"], 100)

    def test_a_scaled_run_moves_the_threshold_off_the_floor(self):
        """The end-to-end symptom, as a before/after on the SAME frame.

        Un-scaled, an over-large target makes `admitted(floor) < target` true
        immediately and the threshold parks on the floor. Scaled, the target is
        reachable and the bisection actually moves.
        """
        frame = catalogue(200_000)
        # MINING, not INDUSTRIAL_FIRE: the synthetic catalogue leaves every
        # `DIST_STEEL/CEMENT/...` column at "far", so the industrial rule's
        # typology conjunct is unsatisfiable there and its count is 0 at ANY
        # persistence threshold. MINING has a satisfiable branch, so it can
        # actually show the collapse and the recovery.
        targets = {wl.MINING: 60_000}

        # population_size == len(frame) => no scaling => target unreachable.
        _floor_th, floor_ev = rc.calibrate_to_target_counts(
            frame, dict(targets), population_size=len(frame))
        self.assertTrue(
            any(e.is_vacuous for e in floor_ev),
            "the unscaled path should still collapse -- that is the bug being "
            "pinned, so if this fails the reproduction itself is stale",
        )

        # Default population => scaled to 1,652 => reachable.
        scaled_th, scaled_ev = rc.calibrate_to_target_counts(frame, dict(targets))
        self.assertGreater(
            scaled_th.mine_persistence,
            1 / 30 + 1e-9,
            "scaling should lift the threshold off the persistence floor",
        )
        self.assertFalse(any(e.is_vacuous for e in scaled_ev),
                         f"scaled run still vacuous: "
                         f"{[e.reason for e in scaled_ev if e.is_vacuous]}")

    def test_vacuity_is_reported_rather_than_silent(self):
        item = rc.Calibration(value=1 / 30, percentile=0.999, n_above=1000,
                              n_total=1000, feature="FRP_PERSISTENCE")
        self.assertTrue(item.is_vacuous)
        self.assertIsNone(item.reason)
        item2 = rc.Calibration(value=0.9, percentile=0.999, n_above=10,
                               n_total=1000, feature="FRP_PERSISTENCE")
        self.assertFalse(item2.is_vacuous)

    def test_report_flags_a_vacuous_threshold(self):
        frame = catalogue(20_000)
        thresholds = wl.WeakThresholds(industrial_persistence=1 / 30)
        evidence = [
            rc.Calibration(value=1 / 30, percentile=0.999,
                           n_above=len(frame), n_total=len(frame),
                           feature="FRP_PERSISTENCE")
        ]
        table, _report, _out = rc.calibration_report(frame, thresholds, evidence)
        self.assertTrue(bool(table["vacuous"].any()))
        self.assertIn("VACUOUS", table.loc[table["vacuous"], "reason"].iloc[0])

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
