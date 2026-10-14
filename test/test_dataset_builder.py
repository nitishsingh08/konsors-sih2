"""
Tests for P6 dataset assembly.

The properties pinned here are the brief's own requirements, which are all ways
a published number becomes wrong without anyone noticing:

  * raw coordinates must never become training features
  * a cloud-blocked observation is a gap, never a zero
  * anomaly is judged against the site's own baseline
  * low-confidence predictions go to human review, not to the bin
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import cluster_tier as ct
import dataset_builder as db
import fire_labels as lab


def assembled(n=400, seed=0):
    """A frame shaped like the real join: meta + tier1 + industrial."""
    rng = np.random.default_rng(seed)
    lon = rng.uniform(70.0, 90.0, n)
    lat = rng.uniform(10.0, 30.0, n)
    return pd.DataFrame(
        {
            "event_id": [f"E{i:07d}" for i in range(n)],
            "event_centroid_lon": lon,
            "event_centroid_lat": lat,
            "event_centroid_x": lon * 1e5,
            "event_centroid_y": lat * 1e5,
            "start_time": pd.to_datetime("2021-01-01")
            + pd.to_timedelta(rng.integers(0, 400, n), unit="D"),
            "FRP_MEDIAN": rng.gamma(3.0, 15.0, n),
            "DETECTION_COUNT_LOG": np.log1p(rng.integers(1, 40, n)),
            "DURATION_HOURS_LOG": np.log1p(rng.integers(0, 500, n)),
            "FRP_PERSISTENCE": rng.uniform(0.0, 1.0, n),
            "FRP_CV": rng.uniform(0.0, 1.5, n),
            "CENTROID_DISPLACEMENT_MEDIAN": rng.uniform(0.0, 3.0, n),
            "DIST_NEAREST_FACILITY_LOG": np.log10(1.0 + rng.gamma(2.0, 900.0, n)),
            "INSIDE_FACILITY": (rng.uniform(size=n) < 0.2).astype(float),
        }
    )


class TestLocationIsStripped(unittest.TestCase):
    """The brief: 'raw coordinates are never direct training features'."""

    def test_detects_coordinate_columns(self):
        for name in (
            "event_centroid_lon", "event_centroid_lat", "event_centroid_x",
            "event_centroid_y", "grid_x", "grid_y", "latitude", "longitude",
        ):
            self.assertTrue(db.is_location_column(name), name)

    def test_keeps_derived_shape_features(self):
        for name in (
            "ELONGATION", "COMPACTNESS", "SOLIDITY", "FRP_MEDIAN",
            "DIST_NEAREST_FACILITY_LOG", "FRP_ANOMALY_VS_SITE",
            "AREA_GROWTH_RATE", "STATIONARITY_INDEX",
        ):
            self.assertFalse(db.is_location_column(name), name)

    def test_lng_is_liquefied_natural_gas_not_longitude(self):
        """Domain collision: this project has an LNG facility category, so
        `DIST_LNG_LOG` is a gas distance and must not be stripped. Longitude is
        already matched by lon/longitude, so nothing is lost by excluding it."""
        self.assertFalse(db.is_location_column("DIST_LNG_LOG"))
        self.assertFalse(db.is_location_column("lng"))
        self.assertTrue(db.is_location_column("event_centroid_lon"))
        self.assertTrue(db.is_location_column("longitude"))

    def test_site_baseline_features_survive(self):
        """The brief permits location only via a site's history, so the
        site-normalised anomaly IS a legitimate feature."""
        for name in ("SITE_FRP_BASELINE", "SITE_N_EVENTS",
                     "FRP_ANOMALY_VS_SITE"):
            self.assertFalse(db.is_location_column(name), name)
        self.assertTrue(db.is_location_column("site_id"))

    def test_identifiers_are_not_treated_as_features(self):
        for name in ("event_id", "spatial_block", "site_id", "date"):
            self.assertTrue(db.is_location_column(name), name)

    def test_strip_removes_coordinates_and_keeps_the_rest(self):
        frame = assembled(50)
        keep, dropped = db.strip_coordinate_columns(frame)
        assert_all_removed = {
            "event_centroid_lon", "event_centroid_lat",
            "event_centroid_x", "event_centroid_y",
        }
        self.assertTrue(assert_all_removed <= set(dropped))
        self.assertIn("FRP_MEDIAN", keep)
        self.assertTrue(assert_all_removed.isdisjoint(keep))

    def test_assert_fires_on_a_leak(self):
        with self.assertRaises(AssertionError) as ctx:
            db.assert_no_location_features(
                ["FRP_MEDIAN", "event_centroid_lon"]
            )
        self.assertIn("leaked", str(ctx.exception))

    def test_assert_passes_for_a_clean_set(self):
        import industrial_features as ind

        db.assert_no_location_features(
            ct.CLUSTER_TIER_FEATURES + list(ind.INDUSTRIAL_FEATURES)
        )

    def test_real_tier1_block_has_no_location_columns(self):
        self.assertEqual(
            [n for n in ct.CLUSTER_TIER_FEATURES if db.is_location_column(n)],
            [],
        )

    def test_real_industrial_block_has_no_location_columns(self):
        import industrial_features as ind

        self.assertEqual(
            [n for n in ind.INDUSTRIAL_FEATURES if db.is_location_column(n)],
            [],
        )


class TestPartitioning(unittest.TestCase):
    """The assembly crashed at batch 12 of 15 holding >10 GB. Partitioning is
    the fix, so the partitioning itself has to be right."""

    def test_partitions_are_in_range(self):
        ids = [f"EVENT_{i:08d}" for i in range(2000)]
        parts = db.partition_of(ids, 16)
        self.assertEqual(len(parts), len(ids))
        self.assertTrue(((parts >= 0) & (parts < 16)).all())

    def test_partitioning_is_deterministic_within_a_process(self):
        ids = [f"EVENT_{i:08d}" for i in range(500)]
        np.testing.assert_array_equal(
            db.partition_of(ids, 16), db.partition_of(ids, 16)
        )

    def test_partitioning_is_stable_across_processes(self):
        """CRC32, not hash(): Python's string hash is randomised per process by
        PYTHONHASHSEED, so a rerun would place an event in a different shard and
        silently produce a different dataset."""
        import subprocess
        import sys

        import pathlib as _pl

        script = (
            "import sys; sys.path.insert(0, %r);"
            "import dataset_builder as db;"
            "print(int(db.partition_of(['EVENT_00000042'], 16)[0]))" % str(
                _pl.Path(__file__).resolve().parent
            )
        )
        # Two different hash seeds must agree; a randomised string hash would
        # not. This is the property that makes a rerun reproducible.
        expected = int(db.partition_of(["EVENT_00000042"], 16)[0])
        for seed in ("12345", "99999"):
            env = {"PYTHONHASHSEED": seed}
            out = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, env=env, check=True,
            ).stdout.strip()
            self.assertEqual(
                int(out.strip("[] ")), expected,
                f"partition differs under PYTHONHASHSEED={seed}",
            )

    def test_event_lands_in_exactly_one_partition(self):
        frame = pd.DataFrame({"event_id": [f"E{i}" for i in range(500)]})
        parts = db.partition_frame(frame, 16)
        self.assertEqual(len(parts), 16)
        total = sum(len(p) for p in parts)
        self.assertEqual(total, len(frame))
        ids = [i for p in parts for i in p["event_id"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_partitions_are_balanced_enough(self):
        frame = pd.DataFrame({"event_id": [f"EVENT_{i:08d}" for i in range(8000)]})
        parts = db.partition_frame(frame, 16)
        sizes = np.array([len(p) for p in parts])
        self.assertEqual(sizes.sum(), 8000)
        # A hash partition should be near-uniform; a bug that hashed only the
        # prefix would collapse everything into one shard.
        self.assertGreater(sizes.min(), 8000 / 16 / 2)
        self.assertLess(sizes.max(), 8000 / 16 * 2)

    def test_single_id_returns_an_int(self):
        self.assertIsInstance(db.partition_of("EVENT_00000001"), int)

    def test_list_returns_an_array(self):
        self.assertIsInstance(db.partition_of(["EVENT_00000001"]), np.ndarray)

    def test_partition_size_is_safe_for_a_colab_runtime(self):
        # ~454k events per partition at 16 partitions.
        rows = 7_265_849 / db.N_PARTITIONS
        self.assertLess(rows, 600_000)

    def test_whole_catalogue_estimate_exceeds_a_partition(self):
        whole = db.partition_bytes_estimate(7_265_849, 56)
        part = db.partition_bytes_estimate(7_265_849 / db.N_PARTITIONS, 56)
        self.assertGreater(whole, part * 10)

    def test_human_bytes_is_readable(self):
        self.assertEqual(db.human_bytes(512), "512.0 B")
        self.assertTrue(db.human_bytes(5_000_000_000).endswith("GB"))


class TestFeatureBlockValidation(unittest.TestCase):
    """Each block is validated against ITS OWN schema, never the union.

    Checking both blocks against the combined 52-column list made a correct
    tier-1 file report as 'missing' the 18 industrial features that live in the
    other file -- a guaranteed failure on correct data.
    """

    def _cols(self, features):
        return ["event_id"] + list(features)

    def test_a_correct_pair_validates(self):
        import industrial_features as ind

        union, unexpected = db.validate_feature_blocks(
            self._cols(ct.CLUSTER_TIER_FEATURES),
            self._cols(ind.INDUSTRIAL_FEATURES),
        )
        self.assertEqual(len(union), 52)
        self.assertEqual(unexpected, [])

    def test_union_follows_ledger_order(self):
        import industrial_features as ind

        union = db.feature_matrix_columns(
            self._cols(ct.CLUSTER_TIER_FEATURES),
            self._cols(ind.INDUSTRIAL_FEATURES),
        )
        self.assertEqual(
            union, list(ct.CLUSTER_TIER_FEATURES) + list(ind.INDUSTRIAL_FEATURES)
        )

    def test_validating_a_block_against_the_union_is_rejected(self):
        # The exact mistake: handing the tier-1 columns in as the industrial
        # block must fail loudly rather than pass.
        with self.assertRaises(ValueError):
            db.validate_feature_blocks(
                self._cols(ct.CLUSTER_TIER_FEATURES),
                self._cols(ct.CLUSTER_TIER_FEATURES),
            )

    def test_missing_column_is_named(self):
        cols = self._cols(ct.CLUSTER_TIER_FEATURES)
        with self.assertRaises(ValueError) as ctx:
            db.validate_feature_blocks(cols[:-1], cols)
        self.assertIn("FRP_MEDIAN", str(ctx.exception))

    def test_unexpected_column_is_rejected(self):
        import industrial_features as ind

        bad = self._cols(ind.INDUSTRIAL_FEATURES) + ["SOMETHING_ELSE"]
        with self.assertRaises(ValueError) as ctx:
            db.validate_feature_blocks(
                self._cols(ct.CLUSTER_TIER_FEATURES), bad
            )
        self.assertIn("SOMETHING_ELSE", str(ctx.exception))

    def test_duplicate_feature_across_blocks_is_rejected(self):
        import industrial_features as ind

        # Both blocks defining FRP_MEDIAN would give X a duplicate column.
        with self.assertRaises(ValueError):
            db.validate_feature_blocks(
                self._cols(ct.CLUSTER_TIER_FEATURES),
                self._cols(ind.INDUSTRIAL_FEATURES) + ["FRP_MEDIAN"],
            )


class TestFlareInputDerivation(unittest.TestCase):
    """The five inputs KNOWN_FLARE needs, recovered by inverting tier 1."""

    def test_detection_count_inverts_log1p(self):
        frame = assembled(20)
        out = db.derive_flare_inputs(frame)
        expected = np.expm1(frame["DETECTION_COUNT_LOG"].to_numpy())
        np.testing.assert_allclose(
            out["detection_count"].to_numpy(), expected, rtol=1e-9
        )

    def test_displacement_inverts_log10(self):
        frame = assembled(20)
        out = db.derive_flare_inputs(frame)
        expected = np.power(
            10.0, frame["CENTROID_DISPLACEMENT_MEDIAN"].to_numpy()
        ) - 1.0
        np.testing.assert_allclose(
            out["centroid_displacement_m"].to_numpy(), expected, rtol=1e-9
        )

    def test_active_days_from_span_and_persistence(self):
        frame = assembled(20)
        out = db.derive_flare_inputs(frame)
        span_days = np.expm1(frame["DURATION_HOURS_LOG"].to_numpy()) / 24.0
        expected = frame["FRP_PERSISTENCE"].to_numpy() * (span_days + 1.0)
        np.testing.assert_allclose(
            out["active_days"].to_numpy(), expected, rtol=1e-9
        )

    def test_single_cluster_event_cannot_look_like_a_flare(self):
        # DETECTION_COUNT_LOG=log1p(1), DURATION_HOURS_LOG=log1p(0) and
        # FRP_PERSISTENCE=1 (the one day it was seen) are what a
        # single-detection cluster really produces. active_days must land near
        # 1, nowhere near the 200-day flare rule.
        frame = pd.DataFrame(
            {
                "DETECTION_COUNT_LOG": [np.log1p(1.0)],
                "DURATION_HOURS_LOG": [np.log1p(0.0)],
                "FRP_PERSISTENCE": [1.0],
                "FRP_CV": [np.nan],
                "CENTROID_DISPLACEMENT_MEDIAN": [np.nan],
            }
        )
        out = db.derive_flare_inputs(frame)
        self.assertAlmostEqual(float(out["detection_count"].iloc[0]), 1.0)
        self.assertAlmostEqual(float(out["active_days"].iloc[0]), 1.0)
        self.assertLess(float(out["active_days"].iloc[0]), 200)

    def test_nan_source_stays_nan(self):
        frame = pd.DataFrame(
            {
                "DETECTION_COUNT_LOG": [np.nan],
                "DURATION_HOURS_LOG": [np.nan],
                "FRP_PERSISTENCE": [np.nan],
                "FRP_CV": [np.nan],
                "CENTROID_DISPLACEMENT_MEDIAN": [np.nan],
            }
        )
        out = db.derive_flare_inputs(frame)
        self.assertTrue(np.isnan(float(out["detection_count"].iloc[0])))
        self.assertTrue(np.isnan(float(out["frp_cv"].iloc[0])))


class TestSiteBaseline(unittest.TestCase):
    """The brief: 'is this heat unusual for this site?'"""

    def _frame(self, lons, frps):
        return pd.DataFrame(
            {
                "event_centroid_lon": lons,
                "event_centroid_lat": [22.0] * len(lons),
                "FRP_MEDIAN": frps,
            }
        )

    def test_all_events_in_one_cell_share_a_baseline(self):
        frame = self._frame([77.00, 77.01, 77.02], [10.0, 20.0, 30.0])
        out = db.add_site_baseline(frame)
        self.assertEqual(out["site_id"].nunique(), 1)
        self.assertAlmostEqual(
            float(out["SITE_FRP_BASELINE"].iloc[0]), 20.0, places=6
        )
        self.assertEqual(int(out["SITE_N_EVENTS"].iloc[0]), 3)

    def test_anomaly_is_the_ratio_to_the_site_median(self):
        frame = self._frame([77.00, 77.01, 77.02], [10.0, 20.0, 390.0])
        out = db.add_site_baseline(frame)
        # 390 / 20 = 19.5x -- the brief's 34 MW -> 390 MW case.
        self.assertAlmostEqual(
            float(out["FRP_ANOMALY_VS_SITE"].iloc[2]), 19.5, places=6
        )

    def test_a_persistent_flare_site_reads_as_normal(self):
        # Every event at 34 MW: no excursion, and the ratio must be ~1.
        frame = self._frame([77.0, 77.01, 77.02, 77.03], [34.0] * 4)
        out = db.add_site_baseline(frame)
        ratios = out["FRP_ANOMALY_VS_SITE"].to_numpy()
        self.assertTrue(np.allclose(ratios, 1.0, atol=1e-9))

    def test_single_event_site_has_no_anomaly(self):
        # One event cannot be anomalous against itself, and 1.0 would read as
        # confidently normal when the truth is "not enough history".
        frame = self._frame([77.00], [500.0])
        out = db.add_site_baseline(frame)
        self.assertTrue(np.isnan(float(out["FRP_ANOMALY_VS_SITE"].iloc[0])))
        self.assertEqual(int(out["SITE_N_EVENTS"].iloc[0]), 1)

    def test_distant_events_get_separate_baselines(self):
        frame = self._frame([77.00, 85.00], [10.0, 500.0])
        out = db.add_site_baseline(frame)
        self.assertEqual(out["site_id"].nunique(), 2)
        self.assertTrue(out["FRP_ANOMALY_VS_SITE"].isna().all())

    def test_missing_coordinates_land_in_the_unknown_site(self):
        frame = self._frame([np.nan], [10.0])
        out = db.add_site_baseline(frame)
        self.assertEqual(out["site_id"].iloc[0], "unknown")

    def test_log_anomaly_is_nan_for_a_unit_ratio(self):
        self.assertTrue(np.isnan(db.log_anomaly(np.array([1.0]))[0]))

    def test_log_anomaly_of_ten_is_one(self):
        self.assertAlmostEqual(db.log_anomaly(np.array([10.0]))[0], 1.0, places=9)

    def test_zero_baseline_does_not_divide_by_zero(self):
        frame = self._frame([77.0, 77.01, 77.02], [0.0, 0.0, 5.0])
        out = db.add_site_baseline(frame)
        values = out["FRP_ANOMALY_VS_SITE"].to_numpy()
        self.assertFalse(np.isinf(values).any())


class TestMissingnessIsAGap(unittest.TestCase):
    """The brief: 'we show a gap, never a zero'."""

    def test_fraction_reflects_the_141_target(self):
        # 1 of 4 available -> 3/4 missing. 0 of 4 -> 4/4 missing.
        frame = pd.DataFrame({"A": [1.0, np.nan], "B": [np.nan, np.nan]})
        out = db.missing_feature_fraction(frame, ["A", "B"], target=4)
        self.assertAlmostEqual(float(out.iloc[0]), 0.75, places=9)
        self.assertAlmostEqual(float(out.iloc[1]), 1.0, places=9)

    def test_zero_is_never_substituted_for_missing(self):
        frame = pd.DataFrame({"A": [np.nan]})
        out = db.missing_feature_fraction(frame, ["A"])
        self.assertAlmostEqual(float(out.iloc[0]), 1.0, places=9)
        self.assertNotEqual(float(out.iloc[0]), 0.0)

    def test_empty_column_set_yields_nan_not_zero(self):
        frame = pd.DataFrame({"A": [1.0]})
        out = db.missing_feature_fraction(frame, [])
        self.assertTrue(np.isnan(float(out.iloc[0])))


class TestReviewQueue(unittest.TestCase):
    """The brief: below 60% confidence goes to human review, not to the bin."""

    def test_split_at_sixty_percent(self):
        out = db.to_review_queue([0.9, 0.6, 0.59, 0.1])
        self.assertEqual(list(out["accept"]), [True, True, False, False])
        self.assertEqual(list(out["review"]), [False, False, True, True])

    def test_low_confidence_is_kept_not_dropped(self):
        out = db.to_review_queue([0.1, 0.9])
        self.assertEqual(out["accept"].sum() + out["review"].sum(), 2)

    def test_nan_goes_to_review(self):
        # "Cannot decide" belongs in front of a human, not in accept.
        out = db.to_review_queue([np.nan])
        self.assertTrue(bool(out["review"][0]))


class TestLabelWiring(unittest.TestCase):
    def test_labels_run_with_derived_flare_inputs(self):
        frame = assembled(300)
        flare = db.derive_flare_inputs(frame)
        out, report = db.label_frame(
            frame.assign(**flare.to_dict("series"))
        )
        self.assertEqual(len(out["label"]), len(frame))
        self.assertEqual(
            sum(report["counts"].values()), report["n_events"]
        )

    def test_labels_absent_flare_inputs_still_run(self):
        frame = assembled(50)
        out = lab.assign_labels(frame)
        self.assertEqual(len(out["label"]), 50)

    def test_review_threshold_is_the_documented_value(self):
        self.assertEqual(db.REVIEW_CONFIDENCE, 0.60)
        self.assertEqual(db.ALERT_SCORE_THRESHOLD, 3.5)

    def test_target_feature_count_is_141(self):
        self.assertEqual(db.TARGET_N_FEATURES, 141)


if __name__ == "__main__":
    unittest.main()