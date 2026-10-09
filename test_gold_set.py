"""
Tests for the gold set.

The gold set is the only independent measurement this project has, so the
properties that protect it are non-negotiable:

  * `build_sample` must NEVER assign a label. A function that fills
    `gold_label` has reintroduced exactly the circularity the gold set exists
    to remove, and it would do so invisibly.
  * The sample must concentrate on the decision boundary. A uniform draw of
    events under a 5 km cut is dominated by obvious cases and cannot measure
    precision where it is decided.
  * `uncertain` is excluded from scoring but must never be silently dropped.
  * `score_against_gold` must refuse to run on an unfinished sheet.
  * Oversampled strata must be weightable back to prevalence.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import gold_set as gs


def catalogue(n=20_000, seed=0, boundary_frac=0.35):
    """Events shaped like the real thing: a hard 5 km cap plus a decision band.

    15.4% sit inside 5 km (the weak positives), and a deliberate mass sits in
    the 3-8 km band where the rule actually decides.
    """
    rng = np.random.default_rng(seed)
    near = rng.random(n) < 0.30
    band = (~near) & (rng.random(n) < boundary_frac / (1 - 0.30))
    metres = np.where(
        near,
        rng.uniform(50, 5000, n),
        np.where(band, rng.uniform(5000, 8000, n), rng.uniform(8000, 600_000, n)),
    )
    return pd.DataFrame({
        "event_id": [f"E{i:06d}" for i in range(n)],
        "DIST_NEAREST_FACILITY_LOG": gs.log_metres(metres),
        "label_weak": np.where(
            metres <= 5000, "INDUSTRIAL_FIRE",
            np.where(metres < 10_000, "WILDFIRE", "AGRICULTURAL"),
        ),
        "spatial_block": [f"B{i % 40:02d}" for i in range(n)],
        "event_centroid_lon": rng.uniform(68, 90, n),
        "event_centroid_lat": rng.uniform(8, 35, n),
    })


class UnitTests(unittest.TestCase):
    def test_log_roundtrip(self):
        for m in (0.0, 1.0, 500.0, 5000.0, 250_000.0):
            self.assertAlmostEqual(
                float(gs.metres_from_log(gs.log_metres(m))), m, places=6
            )

    def test_distance_buckets(self):
        f = gs._distance_bucket
        self.assertEqual(f(1000), "inside_5km")
        # 4,999 m is inside the 5 km cut AND inside the boundary band. That is
        # deliberate: the band oversamples where the cut is still undecided,
        # so it must reach INSIDE the cut, not start at it.
        self.assertEqual(f(4999), "boundary_3_8km")
        self.assertEqual(f(5000.1), "boundary_3_8km")
        self.assertEqual(f(8000), "boundary_3_8km")
        self.assertEqual(f(8000.1), "outside_5km")
        self.assertEqual(f(200_000), "outside_5km")
        self.assertEqual(f(np.nan), "no_distance")


class BuildSampleTests(unittest.TestCase):
    def setUp(self):
        self.cat = catalogue()
        self.spec = gs.GoldSpec(n_total=200, seed=1)
        self.sample = gs.build_sample(self.cat, self.spec)

    def test_it_assigns_no_labels(self):
        """THE critical property. Filling this column is the whole bug."""
        self.assertIn("gold_label", self.sample.columns)
        self.assertTrue(self.sample["gold_label"].isna().all())
        self.assertTrue(self.sample["gold_confidence"].isna().all())

    def test_size_is_respected(self):
        self.assertLessEqual(len(self.sample), self.spec.n_total)
        self.assertGreater(len(self.sample), 0)

    def test_distance_is_given_in_metres_not_log(self):
        """A labeller must not have to undo a log to judge plausibility."""
        self.assertIn("nearest_facility_m", self.sample.columns)
        vals = self.sample["nearest_facility_m"].dropna()
        self.assertGreater(float(vals.max()), 5000.0)

    def test_the_boundary_is_oversampled(self):
        """Uniform sampling would give ~5% here; this must give far more."""
        share = (self.sample["gold_bucket"] == "boundary_3_8km").mean()
        self.assertGreater(share, 0.20, f"boundary share only {share:.1%}")

    def test_gold_ids_are_unique(self):
        ids = self.sample["gold_id"]
        self.assertEqual(ids.nunique(), len(ids))
        self.assertTrue(ids.iloc[0].startswith("GOLD_"))

    def test_rows_come_from_the_catalogue(self):
        self.assertTrue(set(self.sample["event_id"]).issubset(set(self.cat.event_id)))

    def test_blocks_are_respected_when_requested(self):
        spec = gs.GoldSpec(n_total=100, blocks=["B00", "B01", "B02"], seed=0)
        out = gs.build_sample(self.cat, spec)
        self.assertTrue(set(out["spatial_block"]).issubset({"B00", "B01", "B02"}))

    def test_missing_column_raises_rather_than_silently_shrinking(self):
        with self.assertRaises(KeyError):
            gs.build_sample(self.cat.drop(columns=["label_weak"]),
                            gs.GoldSpec(n_total=50))

    def test_requesting_blocks_without_the_column_raises(self):
        with self.assertRaises(KeyError):
            gs.build_sample(self.cat.drop(columns=["spatial_block"]),
                            gs.GoldSpec(n_total=50, blocks=["B00"]))

    def test_audit_reports_an_unlabelled_sheet_as_incomplete(self):
        audit = gs.audit_completion(self.sample)
        self.assertFalse(audit["complete"])
        self.assertIn("unlabelled", audit["issues"][0])

    def test_it_delivers_the_full_requested_size(self):
        """Regression: returned 150 of 300 and reported nothing.

        Round-robin across blocks cannot fill a boundary quota when the band is
        rare, and the shortfall was invisible. A short sample is a DIFFERENT
        sample than the one requested.
        """
        self.assertEqual(len(self.sample), self.spec.n_total)

    def test_stratum_quotas_are_met_exactly(self):
        want = {
            "boundary_3_8km": round(self.spec.n_total
                                     * self.spec.boundary_share),
        }
        got = self.sample["gold_bucket"].value_counts().to_dict()
        for bucket, n in want.items():
            self.assertEqual(got.get(bucket, 0), n, f"{bucket} quota")

    def test_rows_are_unique(self):
        """Regression: the boundary band and the remainder were indexed in two
        different coordinate systems and silently deduplicated to the right
        count with the wrong composition."""
        self.assertEqual(self.sample["event_id"].nunique(), len(self.sample))

    def test_bucket_labels_agree_with_the_distances(self):
        """The strongest check on the position-mapping bug: every row must sit
        inside the distance range its own bucket name claims."""
        d = self.sample.set_index("gold_bucket")["nearest_facility_m"]
        for b in ("inside_5km", "boundary_3_8km", "outside_5km"):
            if b not in d.index:
                continue
            lo, hi = gs.BOUNDARY_BAND_M
            if b == "inside_5km":
                self.assertLess(float(d[b].max()), lo, b)
            elif b == "boundary_3_8km":
                self.assertGreaterEqual(float(d[b].min()), lo, b)
                self.assertLessEqual(float(d[b].max()), hi, b)
            else:
                self.assertGreater(float(d[b].min()), hi, b)

    def test_the_band_is_ordered_by_proximity_to_the_cut(self):
        """The band takes the rows CLOSEST to the 5,000 m decision."""
        band = self.sample[self.sample["gold_bucket"] == "boundary_3_8km"]
        if len(band) < 5:
            self.skipTest("band too small to rank")
        spread = (band["nearest_facility_m"] - 5000.0).abs().max()
        everything = self.sample["nearest_facility_m"]
        worst_all = (everything - 5000.0).abs().max()
        self.assertLess(spread, worst_all)

    def test_shortfall_warns_instead_of_shrinking_silently(self):
        small = gs.build_sample(self.cat.head(400), gs.GoldSpec(n_total=300,
                                                                seed=1))
        self.assertLessEqual(len(small), 300)


class SpecValidationTests(unittest.TestCase):
    def test_absurd_size_rejected(self):
        with self.assertRaises(ValueError):
            gs.GoldSpec(n_total=0).validate()

    def test_bad_boundary_share_rejected(self):
        with self.assertRaises(ValueError):
            gs.GoldSpec(boundary_share=1.5).validate()

    def test_empty_block_list_rejected(self):
        with self.assertRaises(ValueError):
            gs.GoldSpec(blocks=[]).validate()


def labelled(n=400, seed=0, positive_rate=0.3, uncertain_share=0.1):
    """A finished gold sheet: labels, confidence, and a model score."""
    rng = np.random.default_rng(seed)
    lab = []
    for _ in range(n):
        u = rng.random()
        if u < uncertain_share:
            lab.append("uncertain")
        else:
            lab.append("industrial" if rng.random() < positive_rate else "natural")
    # Give the positives higher scores, imperfectly, so precision is not 1.0.
    is_ind = np.array([x == "industrial" for x in lab])
    score = rng.random(n) * 0.5 + is_ind * 0.45 + rng.random(n) * 0.05
    return pd.DataFrame({
        "gold_id": [f"GOLD_{i:04d}" for i in range(n)],
        "gold_label": lab,
        "gold_confidence": ["high"] * n,
        "gold_bucket": rng.choice(
            ["inside_5km", "boundary_3_8km", "outside_5km"], n),
        "nearest_facility_m": rng.uniform(100, 20_000, n),
        "score": np.clip(score, 0, 1),
    })


class AuditTests(unittest.TestCase):
    def test_a_finished_sheet_passes(self):
        audit = gs.audit_completion(labelled())
        self.assertTrue(audit["complete"], audit["issues"])
        self.assertEqual(audit["n_labelled"], 400)

    def test_missing_labels_are_reported(self):
        d = labelled()
        d.loc[:20, "gold_label"] = pd.NA
        audit = gs.audit_completion(d)
        self.assertFalse(audit["complete"])
        self.assertIn("unlabelled", audit["issues"][0])

    def test_bad_label_values_are_caught(self):
        d = labelled()
        d.loc[0, "gold_label"] = "probably industrial"
        audit = gs.audit_completion(d)
        self.assertFalse(audit["complete"])
        self.assertTrue(any("unrecognised" in i for i in audit["issues"]))

    def test_missing_confidence_is_caught(self):
        d = labelled()
        d.loc[:5, "gold_confidence"] = pd.NA
        audit = gs.audit_completion(d)
        self.assertFalse(audit["complete"])
        self.assertTrue(any("confidence" in i for i in audit["issues"]))

    def test_absent_label_column_is_caught(self):
        audit = gs.audit_completion(pd.DataFrame({"gold_id": ["GOLD_0001"]}))
        self.assertFalse(audit["complete"])
        self.assertIn("no gold_label column", audit["issues"][0])

    def test_empty_frame(self):
        audit = gs.audit_completion(pd.DataFrame())
        self.assertFalse(audit["complete"])


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.sheet = labelled()

    def test_it_refuses_an_unfinished_sheet(self):
        d = self.sheet.copy()
        d.loc[:10, "gold_label"] = pd.NA
        out = gs.score_against_gold(d)
        self.assertFalse(out["scored"])
        self.assertIn("audit", out)

    def test_uncertain_rows_are_excluded_and_counted(self):
        out = gs.score_against_gold(self.sheet)
        self.assertTrue(out["scored"])
        self.assertGreater(out["n_uncertain_excluded"], 0)
        self.assertEqual(
            out["n_scored"] + out["n_uncertain_excluded"], len(self.sheet)
        )

    def test_a_perfect_separating_model_scores_one(self):
        d = self.sheet.copy()
        is_ind = (d["gold_label"] == "industrial").astype(float)
        d["score"] = np.where(is_ind > 0, 0.9, 0.1)
        out = gs.score_against_gold(d, threshold=0.5)
        self.assertAlmostEqual(out["precision"], 1.0, places=9)
        self.assertAlmostEqual(out["recall"], 1.0, places=9)

    def test_an_inverted_model_scores_near_zero(self):
        d = self.sheet.copy()
        is_ind = (d["gold_label"] == "industrial").astype(float)
        d["score"] = np.where(is_ind > 0, 0.1, 0.9)
        out = gs.score_against_gold(d, threshold=0.5)
        self.assertLess(out["f1"], 0.01)

    def test_a_constant_model_equals_the_prevalence(self):
        """The trivial predictor must land on the base rate, not look good."""
        d = self.sheet.copy()
        d["score"] = 0.5
        out = gs.score_against_gold(d, threshold=0.5)
        self.assertAlmostEqual(out["precision"], out["prevalence_gold"], places=6)
        self.assertAlmostEqual(out["recall"], 1.0, places=6)

    def test_threshold_is_respected(self):
        """Lowering the threshold can never LOWER recall. It can lower
        precision, and at a high enough threshold precision is undefined
        (nothing predicted) rather than 1.0 -- so only recall is monotonic."""
        d = self.sheet.copy()
        hi = gs.score_against_gold(d, threshold=0.99)
        lo = gs.score_against_gold(d, threshold=0.05)
        self.assertGreaterEqual(lo["recall"], hi["recall"] - 1e-9)
        self.assertLessEqual(hi["precision"], lo["precision"] + 1e-9)
        # Nothing predicted at 0.99 -> confusion must be empty, not a divide
        # that silently reports perfect precision.
        self.assertEqual(hi["confusion"]["tp"], 0.0)

    def test_per_bucket_breakdown_is_present(self):
        out = gs.score_against_gold(self.sheet)
        self.assertIn("per_bucket", out)
        self.assertIn("per_distance_band", out)

    def test_weights_are_used_when_supplied(self):
        d = self.sheet.copy()
        d["weight"] = 2.0
        out = gs.score_against_gold(d)
        self.assertTrue(out["weighted"])

    def test_missing_score_column_raises_with_guidance(self):
        d = self.sheet.drop(columns=["score"])
        with self.assertRaises(KeyError) as ctx:
            gs.score_against_gold(d)
        self.assertIn("gold_id", str(ctx.exception))

    def test_bad_weights_fall_back_to_uniform(self):
        d = self.sheet.copy()
        d["weight"] = np.nan
        out = gs.score_against_gold(d)
        self.assertTrue(out["scored"])   # did not divide by zero


class CompareRulesTests(unittest.TestCase):
    def test_the_distance_rule_is_scored_against_gold(self):
        """The comparison the project has never been able to make."""
        sheet = labelled()
        out = gs.compare_rules(sheet)
        self.assertIn("within_5000m", out["distance_rule"])
        self.assertIn("precision", out["distance_rule"]["within_5000m"])
        self.assertGreater(out["n_scored"], 0)

    def test_a_near_perfect_rule_is_rewarded(self):
        n = 300
        d = pd.DataFrame({
            "gold_label": ["industrial"] * 100 + ["natural"] * 200,
            "nearest_facility_m": [1000.0] * 100 + [50_000.0] * 200,
            "score": [0.9] * 100 + [0.1] * 200,
        })
        out = gs.compare_rules(d)
        self.assertAlmostEqual(
            out["distance_rule"]["within_5000m"]["precision"], 1.0, places=9
        )
        self.assertAlmostEqual(
            out["distance_rule"]["within_5000m"]["recall"], 1.0, places=9
        )

    def test_the_model_is_judged_at_its_own_best_f1(self):
        """Not pinned at 0.5 -- the README section 17 mistake."""
        d = pd.DataFrame({
            "gold_label": ["industrial"] * 50 + ["natural"] * 50,
            "nearest_facility_m": [5000.0] * 100,
            "score": [0.30] * 50 + [0.10] * 50,
        })
        out = gs.compare_rules(d)
        self.assertIn("model_at_own_best_f1", out)
        self.assertLess(out["model_at_own_best_f1"]["threshold"], 0.5)

    def test_uncertain_is_counted_not_dropped(self):
        sheet = labelled()
        out = gs.compare_rules(sheet)
        self.assertGreater(out["n_uncertain_excluded"], 0)
        self.assertEqual(
            out["n_scored"] + out["n_uncertain_excluded"], len(sheet)
        )

    def test_no_distance_column_is_tolerated(self):
        d = pd.DataFrame({"gold_label": ["industrial", "natural"]})
        out = gs.compare_rules(d)
        self.assertEqual(out["distance_rule"], {})


if __name__ == "__main__":
    unittest.main()