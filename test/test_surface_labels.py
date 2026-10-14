"""
Tests for the substrate-based surface label.

This label replaces one measured at precision 0.299, so the properties that
matter are about what it refuses to do:

  * it must never call an event SURFACE_FIRE without mapped substrate
  * it must never call an UNMEASURED event a positive
  * it must not collapse to a vacuous branch, which is how the weak rules died
  * it must read evidence that is NOT in X, or model B stays circular

That last one is the whole point and is asserted against `training_design`
rather than in isolation.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import surface_labels as sl


def frame(n=1000, seed=0):
    """A frame shaped like the real join: substrate + thermal evidence."""
    rng = np.random.default_rng(seed)
    cls = rng.choice([10, 20, 30, 40, 50, 60, 70, 80], size=n)
    veg = np.where(np.isin(cls, sorted(sl.VEGETATED_CODES)), 1, 0)
    days = rng.integers(1, 20, size=n)
    return pd.DataFrame({
        "WORLDCOVER_CLASS": cls,
        "LANDCOVER_VEGETATED": veg,
        "FRP_PERSISTENCE": days / 30.0,
        "DETECTION_COUNT_LOG": np.log1p(days),
    })


class SerialisationTests(unittest.TestCase):
    """Metadata must survive a parquet write.

    A `SurfaceThresholds` object in `DataFrame.attrs` is not JSON-serialisable,
    so `to_parquet` warned and refused, and P7i silently fell back to writing a
    7.27M-row CSV. The version and mismatch count are now plain int columns.
    """

    def test_metadata_is_in_columns_not_only_attrs(self):
        out = sl.assign_surface_labels(frame(50))
        self.assertIn("surface_label_version", out.columns)
        self.assertIn("surface_n_substrate_mismatch", out.columns)

    def test_the_columns_are_json_safe(self):
        import json

        out = sl.assign_surface_labels(frame(50))
        payload = json.dumps({
            "version": str(out["surface_label_version"].iloc[0]),
            "mismatch": int(out["surface_n_substrate_mismatch"].iloc[0]),
        })
        self.assertEqual(json.loads(payload)["mismatch"], 0)
        self.assertRegex(json.loads(payload)["version"], r"^\d+\.\d+\.\d+$")

    def test_attrs_still_carries_the_thresholds_in_memory(self):
        out = sl.assign_surface_labels(frame(50))
        self.assertIsInstance(out.attrs.get("thresholds"),
                              sl.SurfaceThresholds)


class InputTests(unittest.TestCase):
    def test_it_refuses_to_build_without_its_inputs(self):
        """Guessing would silently reproduce a proximity label."""
        with self.assertRaises(KeyError) as ctx:
            sl.assign_surface_labels(frame().drop(columns=["LANDCOVER_VEGETATED"]))
        self.assertIn("LANDCOVER_VEGETATED", str(ctx.exception))

    def test_it_names_the_missing_columns(self):
        with self.assertRaises(KeyError) as ctx:
            sl.assign_surface_labels(
                frame().drop(columns=["WORLDCOVER_CLASS", "FRP_PERSISTENCE"]))
        self.assertIn("WORLDCOVER_CLASS", str(ctx.exception))
        self.assertIn("FRP_PERSISTENCE", str(ctx.exception))

    def test_the_primary_discriminator_is_absent_from_x(self):
        """The property that makes model B worth measuring.

        Substrate is the term that BREAKS the circularity, so it must not be
        reachable from X. The thermal terms are a different case: they are
        ordinary lineage features that happen also to be label inputs, already
        excluded from model B by the weak rules. Both facts are asserted
        separately because conflating them hides which is which.
        """
        import feature_ledger as fl

        for f in ("LANDCOVER_VEGETATED", "WORLDCOVER_CLASS"):
            self.assertNotIn(f, fl.FEATURES_141,
                             f"{f} is the primary discriminator and must be "
                             f"absent from the feature ledger")

    def test_the_thermal_terms_are_ledger_features_but_forbidden(self):
        import feature_ledger as fl
        import training_design as td

        forbidden = td.label_input_features(include_surface=True)
        for f in ("FRP_PERSISTENCE", "DETECTION_COUNT_LOG"):
            self.assertIn(f, fl.FEATURES_141,
                          f"{f} is a real lineage feature")
            self.assertIn(f, forbidden,
                          f"{f} must be excluded from model B as a label input")


class PartitionTests(unittest.TestCase):
    def test_surface_inputs_are_excluded_from_model_b(self):
        import training_design as td

        forbidden = td.label_input_features(include_surface=True)
        for f in sl.surface_label_inputs():
            self.assertIn(f, forbidden, f)

    def test_the_old_partition_is_reproducible(self):
        """A before/after comparison needs the old behaviour available."""
        import training_design as td

        old = td.label_input_features(include_surface=False)
        new = td.label_input_features(include_surface=True)
        self.assertTrue(new > old)
        self.assertIn("LANDCOVER_VEGETATED", new - old)

    def test_substrate_is_never_a_predictor(self):
        """Substrate is LABEL evidence. It must not become a model feature.

        `split_feature_sets` iterates FEATURES_141, so a name outside that list
        is in neither model A nor model B. Adding substrate to X would invite
        exactly the mistake already made with DIST_*.
        """
        import feature_ledger as fl
        import training_design as td

        for f in ("LANDCOVER_VEGETATED", "WORLDCOVER_CLASS"):
            self.assertNotIn(f, fl.FEATURES_141)
            self.assertIn(f, td.label_input_features(include_surface=True),
                          "and must stay forbidden if anyone adds it later")


class AssignmentTests(unittest.TestCase):
    def test_non_vegetated_sustained_is_surface_fire(self):
        f = frame()
        f["WORLDCOVER_CLASS"] = 50
        f["LANDCOVER_VEGETATED"] = 0
        f["FRP_PERSISTENCE"] = 0.5
        f["DETECTION_COUNT_LOG"] = float(np.log1p(10))
        out = sl.assign_surface_labels(f)
        self.assertEqual(out["label_surf"].iloc[0], sl.SURFACE_FIRE)

    def test_vegetated_sustained_is_vegetation_fire(self):
        f = frame()
        f["WORLDCOVER_CLASS"] = 40
        f["LANDCOVER_VEGETATED"] = 1
        f["FRP_PERSISTENCE"] = 0.5
        f["DETECTION_COUNT_LOG"] = float(np.log1p(10))
        out = sl.assign_surface_labels(f)
        self.assertEqual(out["label_surf"].iloc[0], sl.VEGETATION_FIRE)

    def test_unmeasured_substrate_is_unknown_never_a_positive(self):
        f = frame(10)
        f["LANDCOVER_VEGETATED"] = -1
        f["WORLDCOVER_CLASS"] = -1
        out = sl.assign_surface_labels(f)
        self.assertTrue((out["label_surf"] == sl.UNKNOWN).all())
        self.assertTrue(out["is_surface_fire"].isna().all())

    def test_missing_persistence_only_affects_the_tier(self):
        """Substrate is the label; thermal evidence is a tier.

        A draft made UNKNOWN depend on persistence, which cost 97% of the
        catalogue. Missing thermal data must therefore NOT unlabel an event
        whose substrate is perfectly well measured.
        """
        f = frame(10)
        f["WORLDCOVER_CLASS"] = 50
        f["LANDCOVER_VEGETATED"] = 0
        f["FRP_PERSISTENCE"] = np.nan
        out = sl.assign_surface_labels(f)
        self.assertTrue((out["label_surf"] == sl.SURFACE_FIRE).all())
        self.assertEqual(int(out["sustained"].sum()), 0)

    def test_nan_never_satisfies_a_threshold(self):
        """'Could not measure' is not evidence."""
        f = frame(5)
        f["FRP_PERSISTENCE"] = np.nan
        self.assertEqual(int(sl.sustained_tier(f).sum()), 0)

    def test_a_single_detection_is_not_in_the_high_tier(self):
        f = frame(5)
        f["FRP_PERSISTENCE"] = 1.0 / 30.0
        f["DETECTION_COUNT_LOG"] = float(np.log1p(1))
        out = sl.assign_surface_labels(f)
        self.assertEqual(int(out["sustained"].sum()), 0)

    def test_the_label_has_no_persistence_threshold_at_all(self):
        """One knob too many was the original bug.

        `FRP_PERSISTENCE = observed_days / 30` and `DETECTION_COUNT_LOG =
        log1p(detections)` are the SAME evidence, so filtering on both left one
        of them near-inert -- the near-vacuous-threshold shape that destroyed the
        weak label's persistence branch. There must be no persistence filter in
        the label itself.
        """
        fields = set(sl.SurfaceThresholds.__dataclass_fields__)
        self.assertNotIn("min_persistence", fields)
        self.assertNotIn("min_detections", fields)
        self.assertIn("tier_min_days", fields,
                      "repeat observation belongs in the reported tier")

    def test_water_and_snow_count_as_non_vegetated(self):
        """Not the intuitive reading -- but they cannot burn vegetation, and
        33,869 events need a home other than a fourth class."""
        self.assertIn(80, sl.NON_VEGETATED_CODES)
        self.assertIn(70, sl.NON_VEGETATED_CODES)
        f = pd.DataFrame({
            "WORLDCOVER_CLASS": [80, 70, 50],
            "LANDCOVER_VEGETATED": [0, 0, 0],
            "FRP_PERSISTENCE": [0.5, 0.5, 0.5],
            "DETECTION_COUNT_LOG": [np.log1p(10)] * 3,
        })
        out = sl.assign_surface_labels(f)
        self.assertTrue((out["label_surf"] == sl.SURFACE_FIRE).all())

    def test_the_two_substrate_columns_are_cross_checked(self):
        """If they disagree, one is wrong, and that must surface."""
        f = frame(10)
        f["WORLDCOVER_CLASS"] = 40          # cropland
        f["LANDCOVER_VEGETATED"] = 0        # claims non-vegetated
        out = sl.assign_surface_labels(f)
        self.assertEqual(int(out["surface_n_substrate_mismatch"].iloc[0]), 10)

    def test_agreeing_columns_report_no_mismatch(self):
        out = sl.assign_surface_labels(frame())
        self.assertEqual(int(out["surface_n_substrate_mismatch"].iloc[0]), 0)

    def test_is_surface_fire_is_one_zero_or_nan(self):
        out = sl.assign_surface_labels(frame())
        vals = set(out["is_surface_fire"].dropna().unique())
        self.assertTrue(vals <= {0.0, 1.0}, vals)


class ThresholdGridTests(unittest.TestCase):
    def test_pool_size_grows_as_more_codes_count_as_non_vegetated(self):
        f = frame(6000)
        sizes = [sl.pool_size(f) for _ in (0,)]
        for extra in ({50, 60, 70, 80}, {50, 60, 70}, {50, 60}, {50}):
            sizes.append(sl.pool_size(f, extra))
        self.assertEqual(sizes, sorted(sizes, reverse=True), sizes)

    def test_the_grid_sweeps_the_substrate_boundary(self):
        g = sl.pool_grid(frame(6000))
        self.assertGreaterEqual(len(g), 3)
        self.assertIn("boundary", g.columns)
        self.assertIn("surface_fire_tiered", g.columns,
                      "the tier cost must be visible, not felt as a surprise")

    def test_the_grid_is_not_all_or_nothing(self):
        """A grid where every row gives the same count has stopped measuring."""
        g = sl.pool_grid(frame(6000))
        self.assertGreater(g["surface_fire"].nunique(), 1)

    def test_the_tier_is_a_strict_subset(self):
        out = sl.assign_surface_labels(frame(6000))
        tiered = int((out["is_surface_fire_tiered"] == 1).sum())
        plain = int((out["label_surf"] == sl.SURFACE_FIRE).sum())
        self.assertLess(tiered, plain)
        self.assertGreater(tiered, 0)

    def test_the_label_leaves_almost_nothing_unknown(self):
        """The reason the filter was removed: 96.7% of events are
        single-cluster, so demanding a second detection left 97% UNKNOWN."""
        out = sl.assign_surface_labels(frame(6000))
        share = float((out["label_surf"] == sl.UNKNOWN).mean())
        self.assertLess(share, 0.05, f"{share:.1%} UNKNOWN")

    def test_thresholds_describe_themselves(self):
        d = sl.SurfaceThresholds().describe()
        self.assertIn("substrate only", d)
        self.assertIn("tier", d)


class ReportTests(unittest.TestCase):
    def test_the_report_covers_every_class(self):
        r = sl.surface_label_report(frame(2000))
        self.assertEqual(set(r["label"]), set(sl.SURFACE_CLASSES))

    def test_the_report_percentages_sum_to_100(self):
        r = sl.surface_label_report(frame(2000))
        self.assertAlmostEqual(float(r["pct"].sum()), 100.0, places=6)

    def test_no_class_is_called_industrial_fire(self):
        """The name is the point. `INDUSTRIAL_FIRE` would repeat the original
        overclaim one level down."""
        for name in (sl.SURFACE_FIRE, sl.VEGETATION_FIRE, sl.UNKNOWN):
            self.assertNotIn("INDUSTRIAL", name)


if __name__ == "__main__":
    unittest.main()