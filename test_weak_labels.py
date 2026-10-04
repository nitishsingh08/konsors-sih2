"""
Tests for the weak labelling rules and the leakage guard.

Run:  python3 -m unittest test_weak_labels

Two things are pinned here:

  * the UNIT conversion. Every DIST_* feature is log10(1 + d_metres), so a
    radius converts with log10(1+R). The proposal used np.log1p, which makes
    every distance 6-25x tighter than intended and stops industrial events
    being labelled at all.
  * the leakage guard. These labels are manufactured from features the model
    will see, so every rule input must be excluded from the scientific model.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import training_design as td
import weak_labels as wl

FAR = 5_000_000.0   # metres; unambiguously "no facility anywhere near"


def log_m(d):
    return float(np.log10(1.0 + float(d)))


def row(**over):
    """An event far from everything, with no rule satisfied."""
    base = {
        "KNOWN_FLARE": 0.0,
        "INSIDE_FACILITY": 0.0,
        "DIST_REFINERY_LOG": log_m(FAR),
        "DIST_NEAREST_FACILITY_LOG": log_m(FAR),
        "FACILITY_COUNT_5KM": 0.0,
        "DIST_POWERPLANT_LOG": log_m(FAR),
        "DIST_PETROCHEMICAL_LOG": log_m(FAR),
        "DIST_STEEL_LOG": log_m(FAR),
        "DIST_CEMENT_LOG": log_m(FAR),
        "DIST_MINE_LOG": log_m(FAR),
        "FRP_PERSISTENCE": 1.0,
        "DETECTION_COUNT_LOG": float(np.log1p(2)),
        "TIME_TO_PEAK": 6.0,
        "AREA_GROWTH_RATE": np.nan,
        "SPREAD_SPEED_MEDIAN": np.nan,
        "DIRECTION_CONSISTENCY": np.nan,
    }
    base.update(over)
    return pd.DataFrame(base, index=[0])


def only(df, thresholds=None):
    out = wl.assign_weak_labels(df, thresholds or wl.WIDE)
    return out["label"][0]


class TestUnitConversion(unittest.TestCase):
    """The bug that would silently break every distance rule."""

    def test_distance_threshold_is_log10(self):
        self.assertAlmostEqual(wl.distance_threshold(5.0), np.log10(6.0))
        self.assertNotAlmostEqual(wl.distance_threshold(5.0), np.log1p(5.0))

    def test_log1p_would_be_much_too_tight(self):
        """Documents the magnitude of the error, so it cannot come back."""
        for radius in (2.0, 5.0, 10.0):
            wrong = 10 ** np.log1p(radius) - 1
            self.assertGreater(wrong / radius, 2.0,
                               f"radius {radius}: log1p implies {wrong:.0f} m")

    def test_count_threshold_is_log1p(self):
        # DETECTION_COUNT_LOG really is log1p(n), so log1p is correct here.
        self.assertAlmostEqual(wl.count_threshold(3), np.log1p(3.0))


class TestHierarchy(unittest.TestCase):
    def test_a_bare_event_is_unknown(self):
        self.assertEqual(only(row()), wl.UNKNOWN)

    def test_known_flare_flag_short_circuits_everything(self):
        df = row(KNOWN_FLARE=1.0)
        self.assertEqual(only(df), wl.INDUSTRIAL_FLARE)

    def test_inside_refinery_persistent_is_a_flare(self):
        df = row(INSIDE_FACILITY=1.0, DIST_REFINERY_LOG=log_m(500.0),
                 DETECTION_COUNT_LOG=float(np.log1p(3)))
        self.assertEqual(only(df), wl.INDUSTRIAL_FLARE)

    def test_flare_does_not_require_being_inside_the_polygon(self):
        """INSIDE_FACILITY is 0 for all 7,265,849 events, so requiring it made
        the rule unsatisfiable. Distance to a flare-capable facility is the
        robust proxy."""
        df = row(INSIDE_FACILITY=0.0, DIST_REFINERY_LOG=log_m(500.0),
                 DETECTION_COUNT_LOG=float(np.log1p(3)))
        self.assertEqual(only(df), wl.INDUSTRIAL_FLARE)

    def test_beyond_the_refinery_radius_is_not_a_flare(self):
        df = row(DIST_REFINERY_LOG=log_m(50_000.0),
                 DETECTION_COUNT_LOG=float(np.log1p(3)))
        self.assertNotEqual(only(df), wl.INDUSTRIAL_FLARE)

    def test_two_detections_is_not_enough_for_a_flare(self):
        df = row(INSIDE_FACILITY=1.0, DIST_REFINERY_LOG=log_m(500.0),
                 DETECTION_COUNT_LOG=float(np.log1p(2)))
        self.assertNotEqual(only(df), wl.INDUSTRIAL_FLARE)

    def test_near_a_powerplant_with_persistence_is_industrial(self):
        df = row(DIST_NEAREST_FACILITY_LOG=log_m(400.0),
                 DIST_POWERPLANT_LOG=log_m(600.0))
        self.assertEqual(only(df), wl.INDUSTRIAL_FIRE)

    def test_near_but_no_typology_is_not_industrial(self):
        # Close to an unnamed facility: proximity alone is not enough.
        df = row(DIST_NEAREST_FACILITY_LOG=log_m(300.0))
        self.assertEqual(only(df), wl.UNKNOWN)

    def test_facility_count_alone_provides_proximity(self):
        df = row(FACILITY_COUNT_5KM=3.0, DIST_STEEL_LOG=log_m(900.0))
        self.assertEqual(only(df), wl.INDUSTRIAL_FIRE)

    def test_mining_near_a_mine(self):
        df = row(DIST_MINE_LOG=log_m(600.0), FRP_PERSISTENCE=0.9)
        self.assertEqual(only(df), wl.MINING)

    def test_agricultural_far_intermittent_small(self):
        df = row(DIST_NEAREST_FACILITY_LOG=log_m(80_000.0),
                 FRP_PERSISTENCE=0.2, DETECTION_COUNT_LOG=float(np.log1p(2)))
        self.assertEqual(only(df), wl.AGRICULTURAL)

    def test_wildfire_needs_three_independent_signals(self):
        df = row(DIST_NEAREST_FACILITY_LOG=log_m(80_000.0),
                 FRP_PERSISTENCE=0.9, AREA_GROWTH_RATE=0.4,
                 SPREAD_SPEED_MEDIAN=1.2, DIRECTION_CONSISTENCY=0.8)
        self.assertEqual(only(df), wl.WILDFIRE)

    def test_two_wildfire_signals_are_not_enough(self):
        # Only "far from a facility" and "grew in area" are present as wildfire
        # signals. Score 2 < 3, so it cannot be WILDFIRE -- it falls through to
        # AGRICULTURAL, which precedes wildfire and which low persistence plus a
        # small early peak satisfies. The precedence is the hierarchy's, not a
        # bug.
        df = row(DIST_NEAREST_FACILITY_LOG=log_m(80_000.0),
                 FRP_PERSISTENCE=0.2, AREA_GROWTH_RATE=0.4)
        out = wl.assign_weak_labels(df)
        self.assertEqual(float(out["wildfire_score"][0]), 2.0)
        self.assertNotEqual(only(df), wl.WILDFIRE)

    def test_agriculture_precedes_wildfire_in_the_hierarchy(self):
        df = row(DIST_NEAREST_FACILITY_LOG=log_m(80_000.0),
                 FRP_PERSISTENCE=0.2, AREA_GROWTH_RATE=0.4)
        self.assertEqual(only(df), wl.AGRICULTURAL)

    def test_measurable_signals_all_counting_reaches_the_threshold(self):
        # Persistence 0.9 also satisfies agriculture's "< 0.50"? No -- it fails
        # it, so nothing earlier claims the event and wildfire wins.
        df = row(DIST_NEAREST_FACILITY_LOG=log_m(80_000.0),
                 FRP_PERSISTENCE=0.9, AREA_GROWTH_RATE=0.4)
        out = wl.assign_weak_labels(df)
        self.assertEqual(float(out["wildfire_score"][0]), 3.0)
        self.assertEqual(only(df), wl.WILDFIRE)

    def test_precedence_flare_beats_industrial(self):
        df = row(KNOWN_FLARE=1.0, DIST_NEAREST_FACILITY_LOG=log_m(300.0),
                 DIST_POWERPLANT_LOG=log_m(400.0))
        self.assertEqual(only(df), wl.INDUSTRIAL_FLARE)


class TestAbstentionOnMissingEvidence(unittest.TestCase):
    """The MINING rule in the proposal used `pd.isna(persistence) or ...`,
    which makes MINING the fallback for every single-cluster event near a
    mine. Absence of evidence must not become evidence."""

    def test_missing_persistence_does_not_produce_mining(self):
        df = row(DIST_MINE_LOG=log_m(400.0), FRP_PERSISTENCE=np.nan)
        self.assertEqual(only(df), wl.UNKNOWN)

    def test_missing_distance_does_not_produce_a_class(self):
        df = row(DIST_NEAREST_FACILITY_LOG=np.nan,
                 DIST_POWERPLANT_LOG=np.nan,
                 DIST_STEEL_LOG=log_m(300.0))
        self.assertEqual(only(df), wl.UNKNOWN)

    def test_missing_time_to_peak_blocks_agriculture(self):
        df = row(DIST_NEAREST_FACILITY_LOG=log_m(80_000.0),
                 FRP_PERSISTENCE=0.2, DETECTION_COUNT_LOG=float(np.log1p(2)),
                 TIME_TO_PEAK=np.nan)
        self.assertEqual(only(df), wl.UNKNOWN)

    def test_nan_signals_do_not_count_against_the_wildfire_score(self):
        df = row(DIST_NEAREST_FACILITY_LOG=log_m(80_000.0),
                 FRP_PERSISTENCE=0.9, AREA_GROWTH_RATE=np.nan,
                 SPREAD_SPEED_MEDIAN=np.nan, DIRECTION_CONSISTENCY=np.nan)
        out = wl.assign_weak_labels(df)
        self.assertEqual(out["label"][0], wl.UNKNOWN)
        # only 2 of the 5 signals are even measurable
        self.assertEqual(float(out["wildfire_score"][0]), 2.0)

    def test_absent_column_is_not_treated_as_zero(self):
        df = row()
        df = df.drop(columns=["FRP_PERSISTENCE"])
        self.assertEqual(only(df), wl.UNKNOWN)


class TestLabelProvenance(unittest.TestCase):
    def test_every_label_is_marked_weak(self):
        out = wl.assign_weak_labels(row(KNOWN_FLARE=1.0))
        self.assertEqual(out["label_strength"][0], "weak")
        self.assertEqual(out["label_source"][0], "rule_weak_industrial_context")
        self.assertEqual(len(out["label_strength"]), 1)

    def test_unknown_has_no_strength(self):
        out = wl.assign_weak_labels(row())
        self.assertEqual(out["label_strength"][0], "none")

    def test_rule_always_explains_the_label(self):
        for df in (row(), row(KNOWN_FLARE=1.0), row(KNOWN_FLARE=1.0),
                   row(DIST_NEAREST_FACILITY_LOG=log_m(400.0),
                       DIST_POWERPLANT_LOG=log_m(500.0))):
            out = wl.assign_weak_labels(df)
            self.assertNotEqual(out["rule"][0], "")


class TestThresholdVariants(unittest.TestCase):
    def test_tight_radii_label_almost_nothing(self):
        """The 2/5/10 m reading cannot match a real event centroid."""
        df = row(DIST_NEAREST_FACILITY_LOG=log_m(300.0),
                 DIST_POWERPLANT_LOG=log_m(400.0))
        self.assertEqual(only(df, wl.TIGHT), wl.UNKNOWN)
        self.assertEqual(only(df, wl.WIDE), wl.INDUSTRIAL_FIRE)

    def test_report_covers_both_variants(self):
        df = pd.concat(
            [row(), row(DIST_NEAREST_FACILITY_LOG=log_m(300.0),
                        DIST_POWERPLANT_LOG=log_m(400.0))],
            ignore_index=True,
        )
        report = wl.weak_label_report(df)
        self.assertEqual(len(report), 2)
        self.assertIn("variant", report.columns)
        self.assertIn("UNKNOWN", report.columns)


class TestLeakageGuard(unittest.TestCase):
    def test_rule_inputs_are_published(self):
        inputs = wl.label_input_features()
        for name in ("DIST_NEAREST_FACILITY_LOG", "FRP_PERSISTENCE",
                     "DETECTION_COUNT_LOG", "DIST_MINE_LOG",
                     "AREA_GROWTH_RATE", "SPREAD_SPEED_MEDIAN"):
            self.assertIn(name, inputs, name)

    def test_guard_rejects_a_leaking_feature_set(self):
        with self.assertRaises(AssertionError):
            wl.check_no_leakage(["FRP_MEDIAN", "DIST_NEAREST_FACILITY_LOG"])

    def test_guard_accepts_a_clean_feature_set(self):
        self.assertTrue(wl.check_no_leakage(["FRP_MEDIAN", "FRP_MAX"]))

    def test_model_b_excludes_every_rule_input(self):
        import feature_ledger as fl

        _, model_b, report = td.split_feature_sets(list(fl.FEATURES_141))
        for name in wl.LABEL_INPUT_FEATURES:
            if name in fl.FEATURES_141:
                self.assertNotIn(name, model_b, name)

    def test_model_b_also_excludes_the_industrial_eighteen(self):
        import feature_ledger as fl

        _, model_b, _ = td.split_feature_sets(list(fl.FEATURES_141))
        for name in fl.LABEL_DERIVED_FEATURES:
            self.assertNotIn(name, model_b, name)

    def test_report_lists_everything_removed(self):
        import feature_ledger as fl

        model_a, model_b, report = td.split_feature_sets(list(fl.FEATURES_141))
        self.assertEqual(
            len(report["excluded_from_b"]), len(model_a) - len(model_b)
        )


if __name__ == "__main__":
    unittest.main()

class TestConditionPassReport(unittest.TestCase):
    """"Every event is no_rule" is not a diagnosis. Each rule is a conjunction,
    so one unavailable term kills it silently. This names the blocking term."""

    def _far_frame(self, n=1000):
        rng = np.random.default_rng(0)
        far = np.log10(1 + rng.uniform(5e4, 2e6, n))
        return pd.DataFrame({
            "KNOWN_FLARE": np.zeros(n),
            "INSIDE_FACILITY": np.zeros(n),
            "DIST_REFINERY_LOG": far, "DIST_NEAREST_FACILITY_LOG": far,
            "FACILITY_COUNT_5KM": np.zeros(n), "DIST_POWERPLANT_LOG": far,
            "DIST_PETROCHEMICAL_LOG": far, "DIST_STEEL_LOG": far,
            "DIST_CEMENT_LOG": far, "DIST_MINE_LOG": far,
            "FRP_PERSISTENCE": np.ones(n),
            "DETECTION_COUNT_LOG": np.log1p(np.ones(n)),
            "TIME_TO_PEAK": np.full(n, 6.0),
            "AREA_GROWTH_RATE": np.full(n, 0.4),
            "SPREAD_SPEED_MEDIAN": np.full(n, 1.0),
            "DIRECTION_CONSISTENCY": np.full(n, 0.8),
        })

    def test_reports_every_rule(self):
        report = wl.condition_pass_report(self._far_frame())
        for cls in (wl.INDUSTRIAL_FLARE, wl.INDUSTRIAL_FIRE, wl.MINING,
                    wl.AGRICULTURAL, wl.WILDFIRE):
            self.assertIn(cls, set(report["rule"]), cls)

    def test_names_the_blocking_condition(self):
        report = wl.condition_pass_report(self._far_frame())
        weakest = report[report["condition"].str.startswith("weakest:")]
        self.assertEqual(len(weakest), 5)
        for _, row in weakest.iterrows():
            self.assertTrue(row["condition"].startswith("weakest: "))

    def test_conjunction_is_reported(self):
        report = wl.condition_pass_report(self._far_frame())
        self.assertIn("== ALL CONDITIONS ==", set(report["condition"]))
        conj = report[report["condition"] == "== ALL CONDITIONS =="]
        self.assertEqual(len(conj), 5)

    def test_conjunction_never_exceeds_its_weakest_term(self):
        report = wl.condition_pass_report(self._far_frame())
        for cls, group in report.groupby("rule"):
            terms = group[~group["condition"].str.startswith(
                ("==", "weakest:"))]
            conj = int(group[group["condition"] == "== ALL CONDITIONS =="]
                       ["n_pass"].iloc[0])
            self.assertLessEqual(conj, int(terms["n_pass"].min()), cls)

    def test_makes_facility_proximity_the_blocker_when_nothing_is_near(self):
        """With every event 50 km+ from a plant, proximity must be named."""
        report = wl.condition_pass_report(self._far_frame())
        row = report[(report["rule"] == wl.INDUSTRIAL_FIRE)
                     & report["condition"].str.contains("near_any")]
        self.assertEqual(int(row["n_pass"].iloc[0]), 0)

    def test_conditions_are_finite_guarded(self):
        """A NaN input must never count as a pass."""
        frame = self._far_frame(200)
        frame["FRP_PERSISTENCE"] = np.nan
        report = wl.condition_pass_report(frame)
        persist = report[report["condition"].str.contains("FRP_PERSISTENCE>=",
                                                         regex=False)]
        self.assertTrue((persist["n_pass"] == 0).all())
