"""
Tests for operating-threshold calibration.

The failure this module exists to prevent is concrete: a fixed `0.60` review
threshold applied to a model whose held-out scores sit around 0.13, which flags
0.8% of positives. The tests below pin the behaviours that make the difference
between a calibrated operating point and a misleading one:

  * ties are collapsed, never arbitrarily ordered (HANDOFF §9)
  * an infeasible precision constraint returns `threshold: None`, NOT a number
  * a "recall target" that is only reachable at base-rate precision is reported
    as no discrimination, which is what it is
  * a degenerate point (flagging most of the catalogue) is labelled as such
  * calibrating on test raises
"""

from __future__ import annotations

import unittest

import numpy as np

import threshold_calibration as tc
from training_design import average_precision


def separable(n=4000, seed=0, separation=4.0, prevalence=0.15):
    """A model that genuinely separates: positives sit `separation` SDs high.

    Sigmoid, NOT `clip`. Clipping to [0, 1] piles the negatives onto 0.0 and the
    positives onto 1.0, creating two enormous tie runs -- which is exactly the
    condition `TieTests` exists to check, and it made this generator report a
    useless model. The sigmoid is strictly monotone, so ranking is preserved.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    raw = rng.normal(0.0, 1.0, n) + separation * y
    return y, 1.0 / (1.0 + np.exp(-raw))


def useless(n=4000, seed=0, prevalence=0.15):
    """Positives ranked no higher than negatives -- only prevalence to find."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    return y, rng.random(n)


class FbetaTests(unittest.TestCase):
    def test_matches_the_f2_definition(self):
        # F2 at P=1.0, R=1.0 is 1.0; at P=1.0, R=0 it is 0.
        self.assertAlmostEqual(float(tc.f_beta(1.0, 1.0, 2.0)), 1.0, places=12)
        self.assertAlmostEqual(float(tc.f_beta(1.0, 0.0, 2.0)), 0.0, places=12)

    def test_f2_weights_recall_double(self):
        # P=0.5, R=1.0 -> 5*0.5*1 / (4*0.5 + 1) = 2.5/3
        self.assertAlmostEqual(
            float(tc.f_beta(0.5, 1.0, 2.0)), 2.5 / 3.0, places=12
        )

    def test_zero_over_zero_is_zero_not_nan(self):
        out = float(tc.f_beta(0.0, 0.0, 2.0))
        self.assertEqual(out, 0.0)
        self.assertFalse(np.isnan(out))

    def test_vectorises(self):
        got = tc.f_beta(np.array([1.0, 0.5]), np.array([1.0, 0.0]), 2.0)
        self.assertEqual(got.shape, (2,))
        self.assertAlmostEqual(float(got[0]), 1.0, places=12)


class SeparableModelTests(unittest.TestCase):
    """A model that works must produce a usable, non-degenerate threshold."""

    def test_finds_a_high_precision_operating_point(self):
        y, s = separable()
        out = tc.calibrate_operating_threshold(y, s)
        self.assertTrue(out["feasible"])
        self.assertIsNotNone(out["threshold"])
        self.assertGreaterEqual(out["precision"], 0.90)
        self.assertFalse(out["degenerate"], out["reason"])
        self.assertLess(out["flag_fraction"], tc.MAX_FLAG_FRACTION)

    def test_threshold_reproduces_its_own_metrics(self):
        y, s = separable()
        out = tc.calibrate_operating_threshold(y, s)
        pred = (s >= out["threshold"]).astype(int)
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        self.assertAlmostEqual(out["precision"], tp / (tp + fp), places=12)

    def test_threshold_is_inclusive(self):
        """`>=` must hold the calibrated point, or recall drops by one row."""
        y, s = separable()
        out = tc.calibrate_operating_threshold(y, s)
        rows_at_tau = int(np.sum(np.isclose(s, out["threshold"])))
        self.assertGreater(rows_at_tau, 0)
        self.assertEqual(int(np.sum(s >= out["threshold"])), out["n_predicted"])

    def test_precision_constraint_is_respected(self):
        y, s = separable()
        out = tc.calibrate_operating_threshold(y, s, min_precision=0.95)
        self.assertGreaterEqual(out["precision"], 0.95)

    def test_reports_the_fold_it_saw(self):
        y, s = separable(n=1234, seed=3)
        out = tc.calibrate_operating_threshold(y, s)
        self.assertEqual(out["n_rows"], 1234)
        self.assertEqual(out["n_positives"], int(y.sum()))
        self.assertAlmostEqual(out["prevalence"], y.mean(), places=12)


class UselessModelTests(unittest.TestCase):
    """The important negative cases. These are what Model B actually does."""

    def test_returns_none_when_precision_target_is_unreachable(self):
        y, s = useless()
        out = tc.calibrate_operating_threshold(y, s, min_precision=0.999)
        self.assertIsNone(out["threshold"])
        self.assertFalse(out["feasible"])
        self.assertTrue(out["degenerate"])
        self.assertIn("no threshold reaches precision", out["reason"])

    def test_never_invents_a_threshold(self):
        """A None threshold must never become 0.0 or 1.0 downstream."""
        y, s = useless()
        out = tc.calibrate_operating_threshold(y, s, min_precision=0.999)
        self.assertNotIn(out["threshold"], (0.0, 1.0))

    def test_recall_target_reachable_only_at_base_rate_is_no_discrimination(self):
        y, s = useless()
        out = tc.threshold_for_recall(y, s, min_recall=0.50)
        if out["threshold"] is None:
            self.assertIn("unreachable", out["reason"])
        else:
            # If it IS reachable, precision must be at base rate -- i.e. the
            # model ranks nothing above chance.
            self.assertLess(abs(out["precision"] - y.mean()), 0.05)

    def test_pr_auc_of_a_useless_model_is_the_prevalence(self):
        y, s = useless()
        self.assertAlmostEqual(
            average_precision(y, s), y.mean(), places=2
        )


class TieTests(unittest.TestCase):
    """HANDOFF §9: tied scores must never be treated as ordered."""

    def test_constant_predictor_is_not_a_perfect_classifier(self):
        y = np.array([1] * 30 + [0] * 70)
        s = np.full(100, 0.42)
        out = tc.calibrate_operating_threshold(y, s)
        # Every threshold sees the same ranking, so no precision >= 0.90 exists
        # (precision maxes out at prevalence) and None is the honest answer.
        self.assertIsNone(out["threshold"])
        self.assertAlmostEqual(
            out["best_precision_anywhere"], y.mean(), places=12
        )

    def test_a_run_of_ties_is_measured_at_the_end_of_the_run(self):
        """One positive and one negative sharing a score: precision is 0.5.

        An implementation that stable-sorts the tied pair and happens to put
        the positive first would report precision 1.0 here and find a clean
        high-precision operating point out of pure noise. That is precisely
        the failure HANDOFF §9 forbids, so this is the sharpest test here.
        """
        y = np.array([1, 0])
        s = np.array([0.5, 0.5])
        out = tc.calibrate_operating_threshold(y, s, min_precision=0.90)
        # Tie-aware: precision at the only threshold is 0.5, so 0.90 is
        # unreachable and the honest answer is None.
        self.assertIsNone(out["threshold"])
        self.assertAlmostEqual(out["best_precision_anywhere"], 0.5, places=12)

    def test_ties_are_order_independent(self):
        """Same tie, positive first vs negative first: identical metrics."""
        y = np.array([1, 0])
        s = np.array([0.5, 0.5])
        a = tc.calibrate_operating_threshold(y, s, min_precision=0.1)
        b = tc.calibrate_operating_threshold(y[::-1], s[::-1], min_precision=0.1)
        self.assertAlmostEqual(a["precision"], b["precision"], places=12)
        self.assertAlmostEqual(a["precision"], 0.5, places=12)

    def test_ties_do_not_depend_on_input_order(self):
        y = np.array([1, 0, 1, 0])
        s = np.array([0.9, 0.9, 0.2, 0.2])
        a = tc.calibrate_operating_threshold(y, s, min_precision=0.1)
        b = tc.calibrate_operating_threshold(y[::-1], s[::-1],
                                             min_precision=0.1)
        self.assertAlmostEqual(a["precision"], b["precision"], places=12)
        self.assertAlmostEqual(a["recall"], b["recall"], places=12)


class DegeneracyTests(unittest.TestCase):
    def test_flagging_the_whole_catalogue_is_labelled_degenerate(self):
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        out = tc.calibrate_operating_threshold(y, s, min_precision=0.5)
        self.assertTrue(out["degenerate"])
        self.assertIn("trivial predictor", out["reason"])

    def test_a_tight_point_is_not_degenerate(self):
        y, s = separable()
        out = tc.calibrate_operating_threshold(y, s)
        self.assertFalse(out["degenerate"])


class InputValidationTests(unittest.TestCase):
    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            tc.calibrate_operating_threshold([0, 1, 0], [0.1, 0.9])

    def test_non_binary_labels_raise(self):
        with self.assertRaises(ValueError) as ctx:
            tc.calibrate_operating_threshold([0, 2, 1], [0.1, 0.9, 0.5])
        self.assertIn("0/1", str(ctx.exception))

    def test_multidimensional_input_raises(self):
        with self.assertRaises(ValueError):
            tc.calibrate_operating_threshold([[0, 1]], [[0.1, 0.9]])

    def test_no_positives_returns_none_with_a_reason(self):
        y = np.zeros(50, dtype=int)
        s = np.linspace(0, 1, 50)
        out = tc.calibrate_operating_threshold(y, s)
        self.assertIsNone(out["threshold"])
        self.assertIn("no positives", out["reason"])

    def test_all_positive_returns_none_with_a_reason(self):
        y = np.ones(50, dtype=int)
        s = np.linspace(0, 1, 50)
        out = tc.calibrate_operating_threshold(y, s)
        self.assertIsNone(out["threshold"])
        self.assertIn("entirely positive", out["reason"])

    def test_empty_returns_none_with_a_reason(self):
        out = tc.calibrate_operating_threshold([], [])
        self.assertIsNone(out["threshold"])
        self.assertEqual(out["reason"], "no rows")


class EffectiveThresholdTests(unittest.TestCase):
    """The brief's 0.60 and the calibrated tau are different constraints."""

    def test_calibrated_threshold_wins_when_higher(self):
        self.assertAlmostEqual(tc.effective_threshold(0.83, 0.60), 0.83, 10)

    def test_review_floor_wins_when_higher(self):
        self.assertAlmostEqual(tc.effective_threshold(0.31, 0.60), 0.60, 10)

    def test_missing_calibration_degrades_to_the_brief_floor(self):
        """None must never mean 'no constraint'."""
        self.assertAlmostEqual(tc.effective_threshold(None, 0.60), 0.60, 10)

    def test_accept_requires_both_constraints(self):
        y, s = separable()
        out = tc.calibrate_operating_threshold(y, s)
        bar = tc.effective_threshold(out["threshold"], 0.60)
        self.assertGreaterEqual(bar, 0.60)
        self.assertGreaterEqual(bar, out["threshold"])


class ProvenanceTests(unittest.TestCase):
    def test_calibrating_on_test_raises(self):
        with self.assertRaises(ValueError) as ctx:
            tc.assert_calibrated_on_train("test")
        self.assertIn("TRAIN fold", str(ctx.exception))

    def test_calibrating_on_train_passes(self):
        tc.assert_calibrated_on_train("train")


class DescribeTests(unittest.TestCase):
    def test_describe_states_a_missing_operating_point(self):
        y, s = useless()
        line = tc.describe_calibration(
            tc.calibrate_operating_threshold(y, s, min_precision=0.999), "model B"
        )
        self.assertIn("NO operating point", line)

    def test_describe_states_a_found_threshold(self):
        y, s = separable()
        line = tc.describe_calibration(
            tc.calibrate_operating_threshold(y, s), "model A"
        )
        self.assertIn("model A: tau=", line)
        self.assertIn("precision=", line)


if __name__ == "__main__":
    unittest.main()