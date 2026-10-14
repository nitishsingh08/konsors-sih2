"""
Tests for the P5 label engine.

Run:  python3 -m unittest test_fire_labels -v
or:   python3 test_fire_labels.py

Synthetic frames only. The point of these tests is the two failure modes that
silently corrupt a training set:

  1. a rule firing on *absence* of evidence
  2. a missing column being read as zero

Both are checked explicitly below.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import fire_labels as fl

KM = 1_000.0


def log_m(metres):
    return float(np.log10(1.0 + metres))


def base_row(**over):
    """A row that satisfies no rule: everything missing or far from everything."""
    row = {
        "DIST_NEAREST_FACILITY_LOG": log_m(500 * KM),
        "DIST_REFINERY_LOG": log_m(500 * KM),
        "DIST_PETROCHEMICAL_LOG": log_m(500 * KM),
        "DIST_LNG_LOG": log_m(500 * KM),
        "DIST_MINE_LOG": log_m(500 * KM),
        "active_days": np.nan,
        "detection_count": np.nan,
        "n_satellites": np.nan,
        "centroid_displacement_m": np.nan,
        "FRP_CV": np.nan,
        "FRP_MEDIAN": np.nan,
        "BT_I4_MEDIAN": np.nan,
        "VPD_ANOMALY_Z": np.nan,
        "WIND_SPEED": np.nan,
        "area_km2": np.nan,
        "NBR_CHANGE": np.nan,
        "NBR_RECOVERY": np.nan,
        "BARE_MINING_FRACTION": np.nan,
        "SEASON_FRACTION": np.nan,
        "NIGHT_DAY_FRP_RATIO": np.nan,
        "INSIDE_FACILITY": np.nan,
        "KNOWN_FLARE": np.nan,
    }
    row.update(over)
    return row


def frame(**over):
    return pd.DataFrame([base_row(**over)])


def only_label(df):
    return fl.assign_labels(df)["label"][0]


class TestVocabulary(unittest.TestCase):
    def test_six_classes(self):
        self.assertEqual(len(fl.LABEL_CLASSES), 6)
        self.assertIn(fl.UNKNOWN, fl.LABEL_CLASSES)

    def test_unknown_is_not_a_supervised_target(self):
        self.assertNotIn(fl.UNKNOWN, fl.SUPERVISED_CLASSES)
        self.assertEqual(len(fl.SUPERVISED_CLASSES), 5)

    def test_mining_is_its_own_class(self):
        # The project statement lists mining with the industrial facility
        # types, but it must be separable.
        self.assertIn(fl.MINING, fl.LABEL_CLASSES)

    def test_thresholds_are_frozen(self):
        with self.assertRaises(Exception):
            fl.THRESHOLDS.flare_min_active_days = 1


class TestAbstention(unittest.TestCase):
    """The single most important property: no evidence, no class."""

    def test_all_nan_event_is_unknown(self):
        self.assertEqual(only_label(frame()), fl.UNKNOWN)

    def test_unknown_row_reports_no_rule(self):
        out = fl.assign_labels(frame())
        self.assertEqual(out["rule"][0], "no_rule")
        self.assertEqual(out["confidence"][0], 0.0)

    def test_missing_columns_do_not_become_zero(self):
        # An empty frame with no columns at all must not, for example, see
        # FRP_MEDIAN = 0 and be called agricultural.
        out = fl.assign_labels(pd.DataFrame([{}]))
        self.assertEqual(out["label"][0], fl.UNKNOWN)

    def test_short_facility_distance_alone_is_not_a_label(self):
        # Sitting on a refinery with no thermal evidence proves nothing.
        self.assertEqual(
            only_label(frame(DIST_NEAREST_FACILITY_LOG=log_m(0.0))), fl.UNKNOWN
        )


class TestFlare(unittest.TestCase):
    def _flare(self):
        return frame(
            DIST_REFINERY_LOG=log_m(100.0),
            DIST_NEAREST_FACILITY_LOG=log_m(100.0),
            KNOWN_FLARE=1.0,
            INSIDE_FACILITY=1.0,
            active_days=400.0,
            detection_count=400.0,
            n_satellites=3.0,
            centroid_displacement_m=20.0,
            FRP_CV=0.05,
        )

    def test_persistent_stationary_stack_is_a_flare(self):
        self.assertEqual(only_label(self._flare()), fl.INDUSTRIAL_FLARE)

    def test_rule_name(self):
        self.assertEqual(fl.assign_labels(self._flare())["rule"][0], "flare_persistence")

    def test_single_satellite_is_not_enough(self):
        df = self._flare()
        df.loc[0, "n_satellites"] = 1.0
        df.loc[0, "KNOWN_FLARE"] = np.nan
        self.assertEqual(only_label(df), fl.UNKNOWN)

    def test_wind_driven_event_is_not_a_flare(self):
        df = self._flare()
        df.loc[0, "centroid_displacement_m"] = 40_000.0
        df.loc[0, "KNOWN_FLARE"] = np.nan
        self.assertEqual(only_label(df), fl.UNKNOWN)

    def test_known_flare_zero_overrides_derived_evidence(self):
        # P3a already decided; the engine must not second-guess it upward.
        df = self._flare()
        df.loc[0, "KNOWN_FLARE"] = 0.0
        self.assertNotEqual(only_label(df), fl.INDUSTRIAL_FLARE)

    def test_confidence_increases_with_persistence(self):
        short = fl.assign_labels(self._flare())
        long_frame = self._flare()
        long_frame.loc[0, "active_days"] = 3000.0
        long_out = fl.assign_labels(long_frame)
        self.assertGreater(long_out["confidence"][0], short["confidence"][0])


class TestMining(unittest.TestCase):
    def test_bare_persistent_mine_burn_is_mining(self):
        df = frame(
            DIST_MINE_LOG=log_m(300.0),
            DIST_NEAREST_FACILITY_LOG=log_m(300.0),
            BARE_MINING_FRACTION=0.8,
            active_days=90.0,
        )
        self.assertEqual(only_label(df), fl.MINING)

    def test_vegetated_burn_near_a_mine_is_not_mining(self):
        df = frame(
            DIST_MINE_LOG=log_m(300.0),
            DIST_NEAREST_FACILITY_LOG=log_m(300.0),
            BARE_MINING_FRACTION=0.1,
            active_days=90.0,
        )
        self.assertEqual(only_label(df), fl.UNKNOWN)

    def test_short_mine_fire_is_not_mining(self):
        df = frame(
            DIST_MINE_LOG=log_m(300.0),
            DIST_NEAREST_FACILITY_LOG=log_m(300.0),
            BARE_MINING_FRACTION=0.8,
            active_days=3.0,
        )
        self.assertEqual(only_label(df), fl.UNKNOWN)


class TestIndustrialFire(unittest.TestCase):
    def test_surface_loss_at_a_plant_is_industrial_fire(self):
        df = frame(
            DIST_NEAREST_FACILITY_LOG=log_m(500.0),
            NBR_CHANGE=-0.4,
            NBR_RECOVERY=0.01,
            active_days=10.0,
        )
        self.assertEqual(only_label(df), fl.INDUSTRIAL_FIRE)

    def test_recovered_vegetation_is_not_industrial_fire(self):
        # Regrowth means the surface came back: a seasonal burn, not an
        # industrial site loss.
        df = frame(
            DIST_NEAREST_FACILITY_LOG=log_m(500.0),
            NBR_CHANGE=-0.4,
            NBR_RECOVERY=0.30,
            active_days=10.0,
        )
        self.assertEqual(only_label(df), fl.UNKNOWN)

    def test_surface_loss_far_from_any_facility_is_not_industrial(self):
        df = frame(
            DIST_NEAREST_FACILITY_LOG=log_m(300 * KM),
            NBR_CHANGE=-0.4,
            NBR_RECOVERY=0.0,
            active_days=10.0,
        )
        self.assertEqual(only_label(df), fl.UNKNOWN)


class TestAgricultural(unittest.TestCase):
    def _agri(self, **over):
        row = dict(
            DIST_NEAREST_FACILITY_LOG=log_m(80 * KM),
            area_km2=4.0,
            FRP_MEDIAN=12.0,
            SEASON_FRACTION=0.9,
            NIGHT_DAY_FRP_RATIO=0.2,
            active_days=20.0,
        )
        row.update(over)
        return frame(**row)

    def test_seasonal_low_frp_burn_is_agricultural(self):
        self.assertEqual(only_label(self._agri()), fl.AGRICULTURAL)

    def test_high_frp_is_not_agricultural(self):
        self.assertEqual(only_label(self._agri(FRP_MEDIAN=250.0)), fl.UNKNOWN)

    def test_off_season_burn_is_not_agricultural(self):
        self.assertEqual(only_label(self._agri(SEASON_FRACTION=0.2)), fl.UNKNOWN)

    def test_burning_inside_a_plant_gate_is_not_agricultural(self):
        self.assertEqual(
            only_label(self._agri(DIST_NEAREST_FACILITY_LOG=log_m(300.0))),
            fl.UNKNOWN,
        )


class TestWildfire(unittest.TestCase):
    def _fire(self, **over):
        row = dict(
            DIST_NEAREST_FACILITY_LOG=log_m(200 * KM),
            BT_I4_MEDIAN=380.0,
            VPD_ANOMALY_Z=3.0,
            centroid_displacement_m=20_000.0,
            WIND_SPEED=8.0,
        )
        row.update(over)
        return frame(**row)

    def test_hot_dry_wind_driven_event_is_wildfire(self):
        self.assertEqual(only_label(self._fire()), fl.WILDFIRE)

    def test_stationary_burn_is_not_wildfire(self):
        self.assertEqual(
            only_label(self._fire(centroid_displacement_m=50.0)), fl.UNKNOWN
        )

    def test_wind_driven_burn_at_a_plant_is_not_wildfire(self):
        # Already industrial in character; must not be double-counted.
        self.assertEqual(
            only_label(self._fire(DIST_NEAREST_FACILITY_LOG=log_m(200.0))),
            fl.UNKNOWN,
        )

    def test_cool_event_is_not_wildfire(self):
        self.assertEqual(only_label(self._fire(BT_I4_MEDIAN=310.0)), fl.UNKNOWN)


class TestAbstentionDiagnostics(unittest.TestCase):
    """The first full assembly reported only 'no_rule' for 7,265,849 events,
    which cannot distinguish 'threshold not met' from 'input never computed'.
    Those need completely different fixes, so the absent inputs are named."""

    def _flare_frame(self, **over):
        row = {
            "DIST_NEAREST_FACILITY_LOG": [np.log10(1 + 200.0)],
            "DIST_REFINERY_LOG": [np.log10(1 + 200.0)],
            "DIST_PETROCHEMICAL_LOG": [np.nan],
            "DIST_LNG_LOG": [np.nan],
            "DIST_MINE_LOG": [np.nan],
            "DIST_POWERPLANT_LOG": [np.nan],
            "DIST_STEEL_LOG": [np.nan],
            "DIST_CEMENT_LOG": [np.nan],
            "DIST_LANDFILL_LOG": [np.nan],
            "DIST_PIPELINE_LOG": [np.nan],
            "DIST_WELLPAD_LOG": [np.nan],
            "FRP_CV": [0.05],
            "active_days": [400.0],
            "detection_count": [400.0],
            "centroid_displacement_m": [50.0],
            "n_satellites": [3.0],
            "INSIDE_FACILITY": [1.0],
            "KNOWN_FLARE": [np.nan],
            "BARE_MINING_FRACTION": [np.nan],
            "area_km2": [np.nan],
            "SEASON_FRACTION": [np.nan],
            "NIGHT_DAY_FRP_RATIO": [np.nan],
            "BT_I4_MEDIAN": [np.nan],
            "VPD_ANOMALY_Z": [np.nan],
            "WIND_SPEED": [np.nan],
            "NBR_CHANGE": [np.nan],
            "NBR_RECOVERY": [np.nan],
        }
        row.update(over)
        return pd.DataFrame(row)

    def test_full_evidence_labours_as_a_flare(self):
        out = fl.assign_labels(self._flare_frame())
        self.assertEqual(out["label"][0], fl.INDUSTRIAL_FLARE)

    def test_missing_n_satellites_is_named(self):
        out = fl.assign_labels(self._flare_frame(n_satellites=[np.nan]))
        self.assertEqual(out["label"][0], fl.UNKNOWN)
        self.assertIn("n_satellites", out["missing_inputs"][0])

    def test_missing_frp_cv_is_named(self):
        out = fl.assign_labels(self._flare_frame(FRP_CV=[np.nan]))
        self.assertIn("FRP_CV", out["missing_inputs"][0])

    def test_land_cover_inputs_are_named(self):
        out = fl.assign_labels(self._flare_frame())
        missing = out["missing_inputs"][0]
        for name in ("BARE_MINING_FRACTION", "NBR_CHANGE", "SEASON_FRACTION",
                     "area_km2", "BT_I4_MEDIAN", "VPD_ANOMALY_Z", "WIND_SPEED"):
            self.assertIn(name, missing, name)

    def test_single_cluster_case_names_its_own_gaps(self):
        # The 96.7% case: no trend, no displacement, no persistence.
        out = fl.assign_labels(
            self._flare_frame(FRP_CV=[np.nan],
                              centroid_displacement_m=[np.nan],
                              active_days=[1.0])
        )
        self.assertEqual(out["label"][0], fl.UNKNOWN)
        self.assertIn("FRP_CV", out["missing_inputs"][0])

    def test_missing_is_per_event_not_shared(self):
        """The regression: an earlier merge conflated the RULE index with the
        EVENT index, so a missing input on rule 3 was never reported for
        event 0."""
        frame = self._flare_frame(FRP_CV=[np.nan])
        out = fl.assign_labels(frame)
        self.assertIn("FRP_CV", out["missing_inputs"][0])

    def test_multi_row_frame_reports_each_event_separately(self):
        frame = pd.concat(
            [
                self._flare_frame(),
                self._flare_frame(FRP_CV=[np.nan], n_satellites=[np.nan]),
            ],
            ignore_index=True,
        )
        out = fl.assign_labels(frame)
        self.assertEqual(out["label"][0], fl.INDUSTRIAL_FLARE)
        second = out["missing_inputs"][1]
        self.assertIn("FRP_CV", second)
        self.assertIn("n_satellites", second)
        self.assertNotIn("FRP_CV", out["missing_inputs"][0])

    def test_result_length_matches_the_frame(self):
        out = fl.assign_labels(self._flare_frame())
        for key in ("label", "rule", "confidence", "missing_inputs"):
            self.assertEqual(len(out[key]), len(self._flare_frame()), key)


class TestPrecedence(unittest.TestCase):
    def test_flare_beats_industrial_fire(self):
        df = frame(
            DIST_REFINERY_LOG=log_m(100.0),
            DIST_NEAREST_FACILITY_LOG=log_m(100.0),
            KNOWN_FLARE=1.0,
            active_days=400.0,
            NBR_CHANGE=-0.5,
            NBR_RECOVERY=0.0,
        )
        self.assertEqual(only_label(df), fl.INDUSTRIAL_FLARE)

    def test_mining_beats_generic_industrial_fire(self):
        df = frame(
            DIST_MINE_LOG=log_m(200.0),
            DIST_NEAREST_FACILITY_LOG=log_m(200.0),
            BARE_MINING_FRACTION=0.9,
            active_days=100.0,
            NBR_CHANGE=-0.5,
            NBR_RECOVERY=0.0,
        )
        self.assertEqual(only_label(df), fl.MINING)

    def test_industrial_beats_agricultural(self):
        # Seasonal-looking evidence inside a plant gate stays industrial.
        df = frame(
            DIST_NEAREST_FACILITY_LOG=log_m(300.0),
            NBR_CHANGE=-0.5,
            NBR_RECOVERY=0.0,
            active_days=20.0,
            area_km2=5.0,
            FRP_MEDIAN=10.0,
            SEASON_FRACTION=0.95,
            NIGHT_DAY_FRP_RATIO=0.1,
        )
        self.assertEqual(only_label(df), fl.INDUSTRIAL_FIRE)

    def test_exactly_one_label_per_event(self):
        rng = np.random.default_rng(0)
        n = 500
        df = pd.DataFrame(
            {
                k: rng.uniform(0, 1, n) * 10
                for k in [
                    "DIST_NEAREST_FACILITY_LOG",
                    "active_days",
                    "FRP_MEDIAN",
                    "area_km2",
                ]
            }
        )
        out = fl.assign_labels(df)
        self.assertEqual(len(out["label"]), n)
        self.assertTrue(set(out["label"]).issubset(set(fl.LABEL_CLASSES)))


class TestWeights(unittest.TestCase):
    def test_unknown_is_downweighted_not_dropped(self):
        labels = np.array([fl.UNKNOWN, fl.WILDFIRE], dtype=object)
        w = fl.label_weights(labels, confidence=np.array([0.0, 1.0]))
        self.assertEqual(len(w), 2)
        self.assertLess(w[0], w[1])

    def test_confidence_scales_the_weight(self):
        labels = np.array([fl.INDUSTRIAL_FIRE, fl.INDUSTRIAL_FIRE], dtype=object)
        w = fl.label_weights(labels, confidence=np.array([0.2, 1.0]))
        self.assertLess(w[0], w[1])

    def test_nan_confidence_does_not_zero_the_weight(self):
        labels = np.array([fl.WILDFIRE], dtype=object)
        w = fl.label_weights(labels, confidence=np.array([np.nan]))
        self.assertGreater(w[0], 0.0)

    def test_weights_are_positive(self):
        labels = np.array(fl.LABEL_CLASSES, dtype=object)
        self.assertTrue(np.all(fl.label_weights(labels) > 0))


class TestReport(unittest.TestCase):
    def test_report_counts_add_up(self):
        labels = np.array(
            [fl.UNKNOWN, fl.UNKNOWN, fl.WILDFIRE, fl.MINING], dtype=object
        )
        rep = fl.label_report(labels, rules=np.array(["a", "b", "c", "d"], dtype=object))
        self.assertEqual(rep["n_events"], 4)
        self.assertEqual(sum(rep["counts"].values()), 4)
        self.assertEqual(rep["n_unknown"], 2)
        self.assertEqual(rep["n_supervised"], 2)
        self.assertAlmostEqual(rep["supervised_fraction"], 0.5)

    def test_report_handles_empty(self):
        rep = fl.label_report(np.array([], dtype=object))
        self.assertEqual(rep["n_events"], 0)
        self.assertEqual(rep["supervised_fraction"], 0.0)


class TestUnitConversion(unittest.TestCase):
    def test_log_distance_roundtrip(self):
        for metres in (0.0, 1.0, 1_000.0, 137_000.0):
            self.assertAlmostEqual(
                float(fl._log_to_metres(log_m(metres))), metres, places=4
            )

    def test_nan_distance_stays_nan(self):
        self.assertTrue(np.isnan(float(fl._log_to_metres(np.nan))))

    def test_min_available_ignores_nan(self):
        a = np.array([1.0, np.nan])
        b = np.array([np.nan, 5.0])
        out = fl._min_available(a, b)
        self.assertEqual(out[0], 1.0)
        self.assertEqual(out[1], 5.0)

    def test_min_available_all_nan_is_nan(self):
        out = fl._min_available(np.array([np.nan]), np.array([np.nan]))
        self.assertTrue(np.isnan(out[0]))


if __name__ == "__main__":
    unittest.main()