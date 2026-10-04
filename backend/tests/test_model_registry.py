"""
Tests for the model-serving layer.

The properties under test are the ones that fail *silently*: a booster maps
column positions, so a reordered or misaligned feature frame scores confidently
and wrongly. Every test here is about refusing to do that.

Run:  cd backend && python3 -m pytest tests/test_model_registry.py -v
      (or: python3 -m unittest tests.test_model_registry)
"""

from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _FakeBooster:
    """Stands in for a LightGBM booster.

    Records the frame it was given and returns a deterministic score derived from
    the first feature, so a test can assert *which* values reached the model.
    """

    def __init__(self):
        self.seen = None

    def predict(self, matrix, num_iteration=None):
        self.seen = matrix
        return [float(min(1.0, max(0.0, matrix[0][0] / 100.0)))]


#: Booster instances written by the most recent _write_model call, keyed by
#: model name. Pickle serialises a COPY, so the object a test holds is not the
#: object the registry loaded -- the loaded one has to be read back.
_WRITTEN = {}


def _write_model(directory, name, features, target="INDUSTRIAL_FIRE"):
    booster = _FakeBooster()
    _WRITTEN[name] = booster
    payload = {
        "booster": booster,
        "features": list(features),
        "target": target,
        "label_column": "label_weak",
        "best_iteration": 125,
        "review_confidence": 0.60,
        "split": "year_x_block",
        "n_train": 1_500_000,
        "n_test": 600_000,
    }
    path = Path(directory) / f"model_{name}.pkl"
    with path.open("wb") as fh:
        pickle.dump(payload, fh)
    return booster, path


FEATURES = ["FRP_MEDIAN", "DETECTION_COUNT_LOG", "DIST_REFINERY_LOG"]


class ModelRegistryBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def registry(self):
        from app.services.model_registry import ModelRegistry

        return ModelRegistry(self.dir)


class TestLoading(ModelRegistryBase):
    def test_missing_model_raises_with_guidance(self):
        with self.assertRaises(Exception) as ctx:
            self.registry().load("B")
        self.assertIn("colab_p7_train", str(ctx.exception))

    def test_missing_model_names_what_was_found(self):
        _write_model(self.dir, "A", FEATURES)
        with self.assertRaises(Exception) as ctx:
            self.registry().load("B")
        self.assertIn("Found", str(ctx.exception))

    def test_loads_and_caches(self):
        _write_model(self.dir, "B", FEATURES)
        reg = self.registry()
        first = reg.load("B")
        second = reg.load("B")
        self.assertIs(first, second, "second load should come from cache")
        self.assertEqual(first.features, FEATURES)

    def test_available_lists_artefacts(self):
        _write_model(self.dir, "A", FEATURES)
        _write_model(self.dir, "B", FEATURES)
        self.assertEqual(self.registry().available, ["A", "B"])

    def test_payload_without_features_is_rejected(self):
        path = self.dir / "model_B.pkl"
        with path.open("wb") as fh:
            pickle.dump({"booster": _FakeBooster(), "target": "X"}, fh)
        with self.assertRaises(Exception) as ctx:
            self.registry().load("B")
        self.assertIn("features", str(ctx.exception))


class TestPrediction(ModelRegistryBase):
    def setUp(self):
        super().setUp()
        self.booster, _ = _write_model(self.dir, "B", FEATURES)
        self.reg = self.registry()

    def _seen(self):
        """The frame the LOADED booster actually received."""
        return self.reg.load("B").booster.seen

    def test_full_features_are_passed_in_order(self):
        self.reg.predict("B", {f: float(i + 1) for i, f in enumerate(FEATURES)})
        seen = self._seen()
        self.assertEqual(seen.shape, (1, len(FEATURES)))
        # Column ORDER must follow the stored list, not the caller's dict.
        np.testing.assert_array_equal(
            seen[0], np.array([1.0, 2.0, 3.0], dtype="float32")
        )

    def test_caller_dict_order_is_ignored(self):
        shuffled = {FEATURES[2]: 3.0, FEATURES[0]: 1.0, FEATURES[1]: 2.0}
        self.reg.predict("B", shuffled)
        np.testing.assert_array_equal(
            self._seen()[0], np.array([1.0, 2.0, 3.0], dtype="float32")
        )

    def test_missing_features_become_nan_not_zero(self):
        out = self.reg.predict("B", {"FRP_MEDIAN": 50.0})
        seen = self._seen()[0]
        self.assertEqual(seen[0], 50.0)
        # The two absent features must be NaN, NOT 0.0.
        self.assertTrue(np.isnan(seen[1]))
        self.assertTrue(np.isnan(seen[2]))
        self.assertEqual(out["missing_features"], ["DETECTION_COUNT_LOG",
                                                    "DIST_REFINERY_LOG"])
        self.assertAlmostEqual(out["missing_fraction"], 2 / 3, places=6)

    def test_explicit_none_counts_as_missing(self):
        out = self.reg.predict(
            "B", {"FRP_MEDIAN": 50.0, "DETECTION_COUNT_LOG": None,
                  "DIST_REFINERY_LOG": 1.0}
        )
        self.assertEqual(out["missing_features"], ["DETECTION_COUNT_LOG"])
        self.assertTrue(np.isnan(self._seen()[0][1]))

    def test_nan_input_counts_as_missing(self):
        out = self.reg.predict(
            "B", {"FRP_MEDIAN": float("nan"), "DETECTION_COUNT_LOG": 1.0,
                  "DIST_REFINERY_LOG": 1.0}
        )
        self.assertEqual(out["missing_features"], ["FRP_MEDIAN"])

    def test_unknown_column_is_refused(self):
        with self.assertRaises(Exception) as ctx:
            self.reg.predict("B", {"NOT_A_FEATURE": 1.0})
        self.assertIn("was not trained on", str(ctx.exception))

    def test_non_numeric_value_is_refused(self):
        with self.assertRaises(Exception):
            self.reg.predict("B", {"FRP_MEDIAN": "hot"})

    def test_non_mapping_is_refused(self):
        with self.assertRaises(Exception):
            self.reg.predict("B", [1, 2, 3])

    def test_confidence_band_matches_the_stored_threshold(self):
        high = self.reg.predict("B", {f: 90.0 for f in FEATURES})
        self.assertTrue(high["above_review_threshold"])
        self.assertFalse(high["requires_human_review"])
        self.assertAlmostEqual(high["review_confidence"], 0.60)

        low = self.reg.predict("B", {f: 1.0 for f in FEATURES})
        self.assertTrue(low["requires_human_review"])
        self.assertFalse(low["above_review_threshold"])

    def test_probability_is_bounded(self):
        out = self.reg.predict("B", {f: 1e9 for f in FEATURES})
        self.assertGreaterEqual(out["probability"], 0.0)
        self.assertLessEqual(out["probability"], 1.0)

    def test_all_features_absent_is_reported_not_scored_blind(self):
        out = self.reg.predict("B", {})
        self.assertEqual(out["missing_fraction"], 1.0)
        self.assertEqual(len(out["missing_features"]), len(FEATURES))


class TestModelCard(ModelRegistryBase):
    def test_card_carries_the_contract(self):
        _write_model(self.dir, "B", FEATURES)
        card = self.registry().describe("B")
        self.assertEqual(card["n_features"], len(FEATURES))
        self.assertEqual(card["features"], FEATURES)
        self.assertEqual(card["target"], "INDUSTRIAL_FIRE")
        self.assertEqual(card["trained_on"], "year_x_block")

    def test_model_a_card_states_its_score_is_circular(self):
        _write_model(self.dir, "A", FEATURES)
        caveats = " ".join(self.registry().describe("A")["caveats"])
        self.assertIn("circular", caveats.lower())

    def test_model_b_card_states_it_lost_to_the_baseline(self):
        _write_model(self.dir, "B", FEATURES)
        caveats = " ".join(self.registry().describe("B")["caveats"]).lower()
        self.assertIn("baseline", caveats)
        self.assertIn("overpass", caveats)

    def test_health_does_not_load_anything(self):
        _write_model(self.dir, "B", FEATURES)
        status = self.registry().available
        self.assertEqual(status, ["B"])


class TestRegistryIntegration(unittest.TestCase):
    def test_health_helper_shape(self):
        from app.services.model_registry import ModelRegistry, health, configure

        with tempfile.TemporaryDirectory() as tmp:
            _write_model(tmp, "A", FEATURES)
            _write_model(tmp, "B", FEATURES)
            configure(tmp)
            status = health()
            self.assertEqual(sorted(status["models_found"]), ["A", "B"])
            self.assertEqual(status["serving_default"], "B")

    def test_configure_clears_the_cache(self):
        from app.services.model_registry import ModelRegistry, configure

        with tempfile.TemporaryDirectory() as tmp:
            _write_model(tmp, "B", FEATURES)
            configure(tmp)
            reg = ModelRegistry(tmp)
            self.assertEqual(reg.load("B").name, "B")
            # Repointing elsewhere must not keep serving the old model.
            with tempfile.TemporaryDirectory() as other:
                configure(other)
                self.assertEqual(ModelRegistry(other).available, [])


if __name__ == "__main__":
    unittest.main()