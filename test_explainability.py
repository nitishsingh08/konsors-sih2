"""
Tests for the explainability layer.

The properties that matter:

* an explanation must VARY with the event -- the previous implementation
  returned fixed contributions for every input, which is not an explanation;
* attributions must be built in the MODEL's feature order, since a booster maps
  column positions;
* a missing measurement must be refused, not replaced with zero, because zero
  asserts "measured, and it was zero" and SHAP would attribute the decision to a
  feature nobody had;
* ranking must be by |value| so suppressive features are not hidden.

Run:  python3 -m unittest test_explainability -v
"""

from __future__ import annotations

import unittest

import numpy as np

import explainability as ex


def _real_booster():
    """A genuine LightGBM booster.

    SHAP's TreeExplainer only accepts real tree models -- it rejects a stub
    with InvalidModelError -- so a fake would test nothing. Training a 5-tree
    model on 400 synthetic rows keeps these tests fast while exercising the
    actual code path that production will use.
    """
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3)).astype("float32")
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return lgb.train(
        {"objective": "binary", "verbose": -1, "num_threads": 1,
         "min_data_in_leaf": 5},
        lgb.Dataset(X, label=y),
        num_boost_round=5,
    )


def _row(**over):
    base = {"A_FEATURE": 1.0, "B_FEATURE": 2.0, "C_FEATURE": 0.0}
    base.update(over)
    return base


class TestFamilyAggregation(unittest.TestCase):
    def test_features_map_to_ledger_groups(self):
        self.assertEqual(ex.family_of("FRP_MEDIAN"), "THERMAL")
        self.assertEqual(ex.family_of("DIST_MINE_LOG"), "INDUSTRIAL")
        self.assertEqual(ex.family_of("SITE_FRP_BASELINE"), "other")

    def test_unknown_feature_falls_back_rather_than_crashing(self):
        self.assertEqual(ex.family_of("NOT_A_FEATURE"), "other")

    def test_aggregation_sums_within_family(self):
        values = ex.aggregate_by_family(["FRP_MEDIAN", "FRP_MAX"], [0.3, 0.2])
        self.assertAlmostEqual(values["THERMAL"], 0.5)

    def test_aggregation_keeps_sign_so_cancellation_is_visible(self):
        values = ex.aggregate_by_family(["FRP_MEDIAN", "FRP_MAX"], [0.3, -0.3])
        # Taking absolute values here would report the family as influential
        # when it netted out to nothing.
        self.assertAlmostEqual(values["THERMAL"], 0.0)
        self.assertAlmostEqual(abs(values["THERMAL"]), 0.0)

    def test_empty_input_is_empty_output(self):
        self.assertEqual(ex.aggregate_by_family([], []), {})


class TestTopContributions(unittest.TestCase):
    def test_ranks_by_magnitude_not_by_sign(self):
        """A feature that pushes the score DOWN must still be reported."""
        ranked = ex.top_contributions(
            ["FRP_MEDIAN", "FRP_MAX", "FRP_SUM_LOG"], [0.1, -0.9, 0.05], k=3
        )
        self.assertEqual(ranked[0]["key"], "FRP_MAX")
        self.assertEqual(ranked[0]["direction"], "lowers_score")

    def test_respects_k(self):
        ranked = ex.top_contributions(
            ["FRP_MEDIAN", "FRP_MAX", "FRP_SUM_LOG"], [0.1, 0.9, 0.05], k=2
        )
        self.assertEqual(len(ranked), 2)

    def test_each_entry_carries_its_family(self):
        ranked = ex.top_contributions(["DIST_MINE_LOG"], [0.5])
        self.assertEqual(ranked[0]["family"], "INDUSTRIAL")


class TestShap(unittest.TestCase):
    def setUp(self):
        self.features = ["A_FEATURE", "B_FEATURE", "C_FEATURE"]
        self.booster = _real_booster()

    def test_returns_all_three_features(self):
        exp = ex.shap_explain(self.booster, self.features, _row())
        self.assertEqual(len(exp.values), 3)
        self.assertEqual(exp.method, "shap")

    def test_attributions_follow_model_feature_order(self):
        """A booster maps column positions, so order must come from the model."""
        exp = ex.shap_explain(self.booster, self.features, _row())
        self.assertEqual(exp.features, self.features)

    def test_by_family_is_populated(self):
        exp = ex.shap_explain(self.booster, self.features, _row())
        self.assertTrue(exp.by_family)

    def test_caveats_are_attached(self):
        exp = ex.shap_explain(self.booster, self.features, _row(), model_name="A")
        joined = " ".join(exp.caveats).lower()
        self.assertIn("weak label", joined)

    def test_missing_feature_is_refused_not_zero_filled(self):
        row = _row()
        row.pop("B_FEATURE")
        with self.assertRaises(ex.MissingFeatureError) as ctx:
            ex.shap_explain(self.booster, self.features, row)
        self.assertIn("unmeasured", str(ctx.exception).lower())

    def test_nan_value_counts_as_missing(self):
        row = _row(B_FEATURE=float("nan"))
        with self.assertRaises(ex.MissingFeatureError):
            ex.shap_explain(self.booster, self.features, row)

    def test_explanation_varies_with_the_input(self):
        """The regression that motivated this module: the previous mock
        returned identical contributions for every event."""
        a = ex.shap_explain(self.booster, self.features, _row(A_FEATURE=1.0))
        b = ex.shap_explain(self.booster, self.features, _row(A_FEATURE=9.0))
        self.assertNotEqual(a.values, b.values,
                            "explanation did not change with the input")

    def test_additivity_holds_after_unit_conversion(self):
        """base + sum(phi) is a logit; the prediction is a probability.

        Checking one against the other without converting mixes units and
        produces a residual of 2.3 on a perfectly correct explanation. This
        pins the conversion.
        """
        import math

        for value in (3.0, -2.0):
            exp = ex.shap_explain(
                self.booster, self.features, _row(A_FEATURE=value)
            )
            self.assertIsNotNone(exp.additivity_residual)
            self.assertLess(
                exp.additivity_residual, 1e-9,
                "base + sum(phi), squashed, must reproduce the prediction",
            )
            self.assertAlmostEqual(
                1.0 / (1.0 + math.exp(-exp.margin_prediction)),
                exp.prediction, places=6,
            )

    def test_spaces_are_labelled_in_the_payload(self):
        """A reader must be told which units each number is in."""
        payload = ex.shap_explain(
            self.booster, self.features, _row()
        ).as_dict()
        self.assertEqual(payload["base_value_space"], "log_odds")
        self.assertEqual(payload["prediction_space"], "probability")

    def test_serialises(self):
        payload = ex.shap_explain(self.booster, self.features, _row()).as_dict()
        for key in ("method", "base_value", "prediction", "by_family",
                    "top_features", "caveats"):
            self.assertIn(key, payload)


class TestLime(unittest.TestCase):
    def setUp(self):
        self.features = ["A_FEATURE", "B_FEATURE", "C_FEATURE"]
        self.booster = _real_booster()

    def test_produces_ranked_features(self):
        exp = ex.lime_explain(self.booster, self.features, _row(),
                              n_samples=200, seed=0)
        self.assertEqual(exp.method, "lime")
        self.assertTrue(exp.top_features)

    def test_refuses_a_row_with_missing_features(self):
        """LIME resamples observed data and cannot invent a measurement."""
        row = _row()
        row.pop("C_FEATURE")
        with self.assertRaises(ex.MissingFeatureError):
            ex.lime_explain(self.booster, self.features, row, n_samples=100)

    def test_notes_its_own_stochasticity(self):
        exp = ex.lime_explain(self.booster, self.features, _row(),
                              n_samples=100)
        joined = " ".join(exp.caveats).lower()
        self.assertIn("stochastic", joined)

    def test_kernel_returns_probabilities(self):
        """LIME rejects raw margins; the kernel must sigmoid them."""
        kernel = ex._lime_kernel(self.booster, self.features)
        out = kernel(np.array([[1.0, 2.0, 0.0]]))
        # 2-D, one column per class, each row summing to 1.
        self.assertEqual(out.shape, (1, 2))
        self.assertTrue(((out >= 0.0) & (out <= 1.0)).all())
        np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-9)


class TestMethodComparison(unittest.TestCase):
    def _pair(self, shap_keys, lime_keys):
        s = ex.Explanation(method="shap", model="B", base_value=0.0,
                            prediction=0.5, features=[], values=[],
                            top_features=[{"key": k, "contribution": 0.1}
                                           for k in shap_keys])
        l = ex.Explanation(method="lime", model="B", base_value=0.0,
                            prediction=0.5, features=[], values=[],
                            top_features=[{"key": k, "contribution": 0.1}
                                           for k in lime_keys])
        return s, l

    def test_identical_rankings_agree_strongly(self):
        s, l = self._pair(["A", "B", "C"], ["A", "B", "C"])
        out = ex.compare_methods(s, l)
        self.assertTrue(out["comparable"])
        self.assertEqual(out["agreement"], "strong")
        self.assertAlmostEqual(out["jaccard"], 1.0)

    def test_disagreement_is_reported_not_hidden(self):
        s, l = self._pair(["A", "B", "C"], ["D", "E", "F"])
        out = ex.compare_methods(s, l)
        self.assertEqual(out["agreement"], "weak")
        self.assertEqual(out["only_shap"], ["A", "B", "C"])
        self.assertEqual(out["only_lime"], ["D", "E", "F"])

    def test_absent_lime_is_declared_not_treated_as_agreement(self):
        s, _ = self._pair(["A"], ["A"])
        out = ex.compare_methods(s, None)
        self.assertFalse(out["comparable"])

    def test_empty_comparison_does_not_divide_by_zero(self):
        s, _ = self._pair([], [])
        out = ex.compare_methods(s, None)
        self.assertFalse(out["comparable"])


class TestCaveatRegistry(unittest.TestCase):
    def test_both_models_are_documented(self):
        for name in ("A", "B"):
            self.assertTrue(CAVEATS_LOOKUP := ex.CAVEATS.get(name))
            self.assertTrue(CAVEATS_LOOKUP)

    def test_model_a_caveat_states_the_circularity(self):
        joined = " ".join(ex.CAVEATS["A"]).lower()
        self.assertIn("proximity", joined)
        self.assertIn("not a discovery", joined)

    def test_model_b_caveat_states_it_is_not_causal(self):
        joined = " ".join(ex.CAVEATS["B"]).lower()
        self.assertIn("not evidence of physical cause", joined)


if __name__ == "__main__":
    unittest.main()