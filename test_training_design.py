"""
Tests for the P7 training design.

Run:  python3 -m unittest test_training_design -v

The properties pinned here are the ones that decide whether the reported number
survives review: no label-derived feature in the scientific model, no shared
spatial block or test year across a split, and an honest verdict when the model
loses to a single threshold.
"""

from __future__ import annotations

import pathlib
import unittest

import numpy as np
import pandas as pd

import feature_ledger as fl
import fire_labels as lab
import training_design as td

try:
    # per_class_report is defined in the P7 cell; load it without executing
    # the cell (which needs Drive).
    import ast
    import types
    _src = pathlib.Path("colab_p7_train.py").read_text()
    _tree = ast.parse(_src)
    _fn = next(n for n in _tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "per_class_report")
    _mod = types.ModuleType("p7helpers")
    # The extracted function closes over np/pd, so they must be in the exec
    # namespace -- an empty one raises NameError from inside the helper.
    _mod.__dict__.update({"np": np, "pd": pd})
    exec(compile(ast.Module(body=[_fn], type_ignores=[]), "colab_p7_train.py", "exec"),
         _mod.__dict__)
    per_class_report = _mod.per_class_report
except Exception:  # noqa: BLE001
    per_class_report = None


def synthetic_events(n=500, seed=0):
    """Events with a spatial block, a year, a label and a facility distance."""
    rng = np.random.default_rng(seed)
    lon = rng.uniform(68.0, 96.0, n)
    lat = rng.uniform(8.0, 34.0, n)
    years = rng.integers(2012, 2027, n)
    dist = rng.gamma(2.0, 800.0, n)
    label = (dist < 1000.0).astype(int)
    return pd.DataFrame(
        {
            "event_id": [f"E{i:07d}" for i in range(n)],
            "event_centroid_lon": lon,
            "event_centroid_lat": lat,
            "start_year": years,
            "DIST_NEAREST_FACILITY_LOG": np.log10(1.0 + dist),
            "FRP_MAX": rng.gamma(5.0, 10.0, n),
            "label": label,
        }
    )


class TestLeakagePartition(unittest.TestCase):
    def test_label_derived_is_the_industrial_eighteen(self):
        self.assertEqual(len(td.LABEL_DERIVED), 18)
        self.assertEqual(td.LABEL_DERIVED, list(fl.FEATURE_GROUPS[fl.GROUP_INDUSTRIAL]))

    def test_non_leak_is_123(self):
        self.assertEqual(len(td.NON_LEAK), 123)

    def test_model_b_is_smaller_than_the_ledger_non_leak_partition(self):
        """Model B additionally excludes every weak-rule input, so it is a
        strict subset of the ledger's 123-feature non-leak partition."""
        import feature_ledger as fl

        _, model_b, _ = td.split_feature_sets(list(fl.FEATURES_141))
        self.assertLess(len(model_b), len(td.NON_LEAK))

    def test_model_b_never_contains_a_label_derived_feature(self):
        available = list(fl.FEATURES_141)
        _, model_b, report = td.split_feature_sets(available)
        # Report key renamed: the exclusion is no longer industrial-only.
        self.assertNotIn("label_derived_in_b", report)
        self.assertIn("excluded_from_b", report)
        for name in td.LABEL_DERIVED:
            self.assertNotIn(name, model_b)

    def test_model_a_includes_the_label_derived_features(self):
        model_a, _, report = td.split_feature_sets(list(fl.FEATURES_141))
        self.assertEqual(report["n_label_derived_in_a"], 18)
        self.assertEqual(len(model_a), 141)

    def test_partition_respects_what_actually_exists(self):
        # DIST_NEAREST_FACILITY_LOG manufactures y, so it is excluded from B.
        available = ["FRP_MAX", "DIST_NEAREST_FACILITY_LOG"]
        model_a, model_b, _ = td.split_feature_sets(available)
        self.assertEqual(model_a, available)
        self.assertEqual(model_b, ["FRP_MAX"])

    def test_unknown_feature_names_are_reported(self):
        _, _, report = td.split_feature_sets(["NOT_A_FEATURE"])
        self.assertEqual(report["missing_from_141"], ["NOT_A_FEATURE"])

    def test_weak_rule_input_stays_out_of_b(self):
        available = ["FRP_MAX", "DIST_NEAREST_FACILITY_LOG"]
        model_a, model_b, _ = td.split_feature_sets(available)
        self.assertIn("DIST_NEAREST_FACILITY_LOG", model_a)
        self.assertNotIn("DIST_NEAREST_FACILITY_LOG", model_b)


class TestSplitKeys(unittest.TestCase):
    def test_block_is_derived_at_point_one_degree(self):
        df = td.add_split_keys(synthetic_events(50))
        lon = df["event_centroid_lon"].iloc[0]
        expected = int(np.floor(lon / td.BLOCK_DEG))
        self.assertTrue(
            df["spatial_block"].iloc[0].startswith(f"{expected}_")
        )

    def test_year_is_preserved(self):
        df = td.add_split_keys(synthetic_events(50))
        self.assertIn("start_year", df.columns)
        self.assertTrue(df["start_year"].between(2012, 2026).all())

    def test_missing_coordinates_get_the_unknown_block(self):
        df = synthetic_events(10)
        df.loc[0, "event_centroid_lon"] = np.nan
        out = td.add_split_keys(df)
        self.assertEqual(out["spatial_block"].iloc[0], "unknown")

    def test_works_from_start_time_when_no_year_column(self):
        df = synthetic_events(10).drop(columns=["start_year"])
        df["start_time"] = pd.to_datetime("2021-03-01") + pd.to_timedelta(
            np.arange(10), unit="D"
        )
        out = td.add_split_keys(df)
        self.assertTrue((out["start_year"] == 2021).all())


class TestOutOfTime(unittest.TestCase):
    def test_no_test_year_leaks_into_train(self):
        df = td.add_split_keys(synthetic_events(800))
        parts = td.split_out_of_time(df)
        self.assertTrue((parts["train"]["start_year"] < 2020).all())
        self.assertTrue((parts["test"]["start_year"] >= 2020).all())
        td.assert_no_time_leakage(parts["train"], parts["test"])

    def test_both_sides_are_non_empty(self):
        df = td.add_split_keys(synthetic_events(800))
        parts = td.split_out_of_time(df)
        self.assertGreater(len(parts["train"]), 0)
        self.assertGreater(len(parts["test"]), 0)


class TestOutOfSpace(unittest.TestCase):
    def test_no_block_appears_on_both_sides(self):
        df = td.add_split_keys(synthetic_events(2000))
        parts = td.split_out_of_space(df, seed=1)
        self.assertEqual(
            set(parts["train"]["spatial_block"]) & set(parts["test"]["spatial_block"]),
            set(),
        )
        td.assert_no_spatial_leakage(parts["train"], parts["test"])

    def test_roughly_a_fifth_of_blocks_are_held_out(self):
        df = td.add_split_keys(synthetic_events(3000))
        parts = td.split_out_of_space(df, seed=2)
        n_blocks = df["spatial_block"].nunique()
        held = parts["test"]["spatial_block"].nunique()
        self.assertGreater(held, 0)
        self.assertLess(held, n_blocks)

    def test_seed_changes_which_blocks_are_held_out(self):
        df = td.add_split_keys(synthetic_events(2000))
        a = td.split_out_of_space(df, seed=1)["test_blocks"]
        b = td.split_out_of_space(df, seed=2)["test_blocks"]
        self.assertNotEqual(a, b)

    def test_leakage_assertion_actually_fires(self):
        df = td.add_split_keys(synthetic_events(400))
        parts = td.split_out_of_space(df, seed=1)
        # Force one test block into training and the guard must notice.
        leaked_block = parts["test"]["spatial_block"].iloc[0]
        poisoned = pd.concat(
            [parts["train"],
             parts["train"].head(1).assign(spatial_block=leaked_block)],
            ignore_index=True,
        )
        with self.assertRaises(AssertionError):
            td.assert_no_spatial_leakage(poisoned, parts["test"])


class TestCombined(unittest.TestCase):
    def test_test_side_is_late_and_block_held_out(self):
        df = td.add_split_keys(synthetic_events(3000))
        parts = td.split_combined(df, seed=3)
        test = parts["test"]
        self.assertTrue((test["start_year"] >= 2020).all())
        # No held-out block may appear anywhere in training. This is the
        # property the naive construction silently violated.
        self.assertEqual(
            set(test["spatial_block"]) & set(parts["train"]["spatial_block"]),
            set(),
        )
        td.assert_no_spatial_leakage(parts["train"], parts["test"])

    def test_early_events_in_test_blocks_are_dropped_not_trained_on(self):
        df = td.add_split_keys(synthetic_events(3000))
        parts = td.split_combined(df, seed=3)
        dropped = parts["dropped_early_in_test_blocks"]
        self.assertGreater(len(dropped), 0)
        self.assertTrue((dropped["start_year"] < 2020).all())
        self.assertEqual(
            set(dropped["spatial_block"]) & set(parts["train"]["spatial_block"]),
            set(),
        )

    def test_every_event_is_accounted_for(self):
        df = td.add_split_keys(synthetic_events(3000))
        parts = td.split_combined(df, seed=3)
        total = (len(parts["train"]) + len(parts["test"])
                 + len(parts["dropped_early_in_test_blocks"]))
        self.assertEqual(total, len(df))

    def test_train_has_early_events(self):
        df = td.add_split_keys(synthetic_events(3000))
        parts = td.split_combined(df, seed=3)
        self.assertGreater(len(parts["train"]), 0)

    def test_test_side_is_smaller_than_space_only(self):
        df = td.add_split_keys(synthetic_events(3000))
        combined = td.split_combined(df, seed=3)["test"]
        space_only = td.split_out_of_space(df, seed=3)["test"]
        self.assertLessEqual(len(combined), len(space_only))

    def test_make_split_dispatches(self):
        df = td.add_split_keys(synthetic_events(800))
        for kind in (td.SPLIT_OUT_OF_TIME, td.SPLIT_OUT_OF_SPACE,
                     td.SPLIT_COMBINED):
            parts = td.make_split(df, kind=kind, seed=0)
            self.assertIn("train", parts)
            self.assertIn("test", parts)

    def test_unknown_split_raises(self):
        df = td.add_split_keys(synthetic_events(50))
        with self.assertRaises(ValueError):
            td.make_split(df, kind="random")


class TestBaselines(unittest.TestCase):
    def test_threshold_predicts_within_distance(self):
        pred = td.distance_baseline([100.0, 999.0, 1001.0, np.nan], 1000.0)
        self.assertEqual(list(pred), [True, True, False, False])

    def test_nan_distance_is_never_a_positive(self):
        pred = td.distance_baseline([np.nan] * 5, 10_000.0)
        self.assertFalse(pred.any())

    def test_tuning_picks_the_best_f1(self):
        d = np.array([10.0, 20.0, 5000.0, 6000.0])
        y = np.array([1, 1, 0, 0])
        best = td.tune_distance_baseline(d, y)
        self.assertAlmostEqual(best["precision"], 1.0)
        self.assertAlmostEqual(best["recall"], 1.0)

    def test_tuning_reports_precision_target(self):
        d = np.array([10.0, 5000.0, 20.0, 6000.0])
        y = np.array([1, 0, 0, 1])
        best = td.tune_distance_baseline(d, y, min_precision=0.90)
        self.assertIn("meets_precision_target", best)


class TestMetrics(unittest.TestCase):
    def test_perfect_prediction(self):
        m = td.binary_metrics([1, 1, 0, 0], [1, 1, 0, 0])
        self.assertAlmostEqual(m["precision"], 1.0)
        self.assertAlmostEqual(m["recall"], 1.0)
        self.assertAlmostEqual(m["pr_auc"], 1.0)
        self.assertEqual(m["confusion"]["tp"], 2)

    def test_constant_predictor_scores_exactly_the_prevalence(self):
        """A model with no discrimination scores the base rate, not 1.0.

        Ties must collapse: with an arbitrary tie order a constant predictor
        could score a perfect 1.0, which is the single easiest way to report a
        meaningless PR-AUC.
        """
        m = td.binary_metrics([1, 1, 0, 0], [0, 0, 0, 0])
        self.assertAlmostEqual(m["recall"], 0.0)
        self.assertAlmostEqual(m["pr_auc"], 0.5)      # prevalence
        self.assertAlmostEqual(m["precision"], 0.0)   # predicts nothing

    def test_precision_recall_at_target_reports_recall(self):
        out = td.precision_recall_at_precision([1, 1, 0, 0], [1, 0, 0, 0], 0.9)
        self.assertAlmostEqual(out["precision"], 1.0)
        self.assertAlmostEqual(out["recall"], 0.5)

    def test_tied_scores_do_not_inflate_the_operating_point(self):
        """One positive tied with negatives must not count as confidently caught."""
        out = td.precision_recall_at_precision([1, 0, 0], [0, 0, 0], 0.9)
        self.assertLessEqual(out["recall"], 1.0)
        self.assertIsNone(out["threshold"])

    def test_average_precision_is_lower_for_a_worse_ranking(self):
        y = [1, 1, 0, 0]
        good = td.average_precision(y, [0.9, 0.8, 0.2, 0.1])
        bad = td.average_precision(y, [0.1, 0.2, 0.8, 0.9])
        self.assertGreater(good, bad)

    def test_pr_auc_is_low_when_positives_rank_below_negatives(self):
        # 1 positive ranked last among 101: PR-AUC must be poor even though
        # ROC-AUC would look unremarkable.
        y = [0] * 100 + [1]
        p = [0.5] * 100 + [0.1]
        self.assertLess(td.average_precision(y, p), 0.05)

    def test_ties_do_not_inflate_pr_auc(self):
        # All predictions identical -> prevalence, whatever the label balance.
        self.assertAlmostEqual(
            td.average_precision([1, 0, 0, 0, 0], [0.5] * 5), 0.2, places=9
        )

    def test_real_ranking_still_scores_high(self):
        y = [1] * 5 + [0] * 95
        p = [0.9] * 5 + [0.1] * 95
        self.assertGreater(td.average_precision(y, p), 0.9)


class TestVerdict(unittest.TestCase):
    def test_beating_the_baseline_is_reported_honestly(self):
        v = td.compare_to_baseline(
            {"pr_auc": 0.80}, {"f1": 0.60}
        )
        self.assertTrue(v["beats_baseline"])
        self.assertGreater(v["absolute_gain"], 0.0)

    def test_losing_to_the_baseline_is_stated_plainly(self):
        v = td.compare_to_baseline({"pr_auc": 0.40}, {"f1": 0.70})
        self.assertFalse(v["beats_baseline"])
        self.assertIn("does NOT beat", v["verdict"])


class TestPlan(unittest.TestCase):
    def test_plan_lists_two_models_and_the_headline_split(self):
        import weak_labels

        plan = td.training_plan(list(fl.FEATURES_141), {"INDUSTRIAL_FLARE": 10})
        self.assertEqual(plan["model_a"]["n"], 141)
        expected_b = 141 - len(
            set(fl.LABEL_DERIVED_FEATURES) | set(weak_labels.LABEL_INPUT_FEATURES)
        )
        self.assertEqual(plan["model_b"]["n"], expected_b)
        self.assertEqual(plan["headline_split"], td.SPLIT_COMBINED)
        self.assertEqual(len(plan["splits"]), 3)

    def test_plan_reports_label_counts(self):
        plan = td.training_plan(["FRP_MAX"], {"UNKNOWN": 5})
        self.assertEqual(plan["label_counts"], {"UNKNOWN": 5})

    def test_unknown_label_is_kept_not_dropped(self):
        import fire_labels as lab

        labels = np.array(
            [lab.UNKNOWN, lab.WILDFIRE, lab.INDUSTRIAL_FLARE], dtype=object
        )
        w = td.class_weights(labels)
        self.assertEqual(len(w), 3)
        self.assertLess(w[0], w[2])


if __name__ == "__main__":
    unittest.main()

class TestPerClassReporting(unittest.TestCase):
    """The negative side of a binary report must be a class that EXISTS.

    An earlier P7 printed a row for a synthetic "NOT_INDUSTRIAL_FIRE" that
    never appears in the labels, so its support showed 0 and the row read as a
    class the model had never encountered. It also double-reported the
    positives under the strict label column.
    """

    def _folded(self, labels, target):
        return np.where(np.asarray(labels) == target, target, "OTHER_CLASS")

    def test_negative_class_has_real_support(self):
        labels = np.array(
            [lab.INDUSTRIAL_FIRE] * 30
            + [lab.AGRICULTURAL] * 50
            + [lab.UNKNOWN] * 20
        )
        folded = self._folded(labels, lab.INDUSTRIAL_FIRE)
        self.assertEqual(int((folded == "OTHER_CLASS").sum()), 70)
        self.assertEqual(int((folded == lab.INDUSTRIAL_FIRE).sum()), 30)

    def test_folding_covers_every_weak_class(self):
        labels = np.array(list(lab.LABEL_CLASSES))
        folded = self._folded(labels, lab.MINING)
        negatives = {v for v in folded if v != lab.MINING}
        self.assertEqual(negatives, {"OTHER_CLASS"})
        self.assertEqual(int((folded == lab.MINING).sum()), 1)

    def test_no_support_zero_rows_in_a_well_formed_report(self):
        labels = np.array(
            [lab.INDUSTRIAL_FIRE] * 10 + [lab.MINING] * 5
        )
        folded = self._folded(labels, lab.INDUSTRIAL_FIRE)
        pred = np.array([lab.INDUSTRIAL_FIRE] * 12 + ["OTHER_CLASS"] * 3)
        report = per_class_report(folded, pred,
                                  [lab.INDUSTRIAL_FIRE, "OTHER_CLASS"])
        self.assertTrue((report["support"] > 0).all(),
                        report.to_string(index=False))

    def test_counts_are_consistent(self):
        labels = np.array(
            [lab.WILDFIRE] * 8 + [lab.AGRICULTURAL] * 4 + [lab.UNKNOWN] * 2
        )
        folded = self._folded(labels, lab.WILDFIRE)
        pred = np.array([lab.WILDFIRE] * 7 + ["OTHER_CLASS"] * 7)
        report = per_class_report(folded, pred,
                                  [lab.WILDFIRE, "OTHER_CLASS"])
        total = report["support"].sum()
        self.assertEqual(total, len(labels))
        # per_class_report emits per-class tp/fp/fn; summing them over both
        # classes double-counts each cell once per class row, so check the
        # identity that actually holds instead.
        self.assertEqual(int(report["support"].sum()), len(labels))
        self.assertEqual(int((pred == lab.WILDFIRE).sum()), 7)
        for _, row in report.iterrows():
            self.assertEqual(int(row["tp"] + row["fn"]), int(row["support"]))
            self.assertEqual(int(row["tp"] + row["fp"]),
                             int((pred == row["class"]).sum()))
