"""
Structural tests for the 141-feature ledger.

Run:  python3 -m unittest test_feature_ledger -v
or:   python3 test_feature_ledger.py
"""

from __future__ import annotations

import unittest

import feature_ledger as fl


class TestSchemaIntegrity(unittest.TestCase):
    def test_exactly_141_features(self):
        self.assertEqual(fl.N_FEATURES, 141)
        self.assertEqual(len(fl.FEATURES), 141)

    def test_all_names_unique(self):
        self.assertEqual(len(set(fl.FEATURES_141)), 141)

    def test_group_sizes(self):
        for group, expected in fl.EXPECTED_GROUP_SIZES.items():
            actual = sum(1 for f in fl.FEATURES_141 if fl.GROUP_OF[f] == group)
            self.assertEqual(actual, expected, f"group {group}")

    def test_group_sizes_sum_to_141(self):
        self.assertEqual(sum(fl.EXPECTED_GROUP_SIZES.values()), 141)

    def test_no_placeholder_leaked(self):
        for bad in fl._PLACEHOLDERS:
            self.assertNotIn(bad, fl.FEATURES_141)

    def test_leakage_partition(self):
        self.assertEqual(len(fl.LABEL_DERIVED_FEATURES), 18)
        self.assertEqual(len(fl.NON_LEAK_FEATURES), 123)
        self.assertEqual(
            sorted(fl.LABEL_DERIVED_FEATURES + fl.NON_LEAK_FEATURES),
            sorted(fl.FEATURES_141),
        )
        for f in fl.LABEL_DERIVED_FEATURES:
            self.assertEqual(fl.GROUP_OF[f], fl.GROUP_INDUSTRIAL)


class TestAuditRecords(unittest.TestCase):
    def setUp(self):
        self.records = fl.build_audit_records()

    def test_record_count(self):
        self.assertEqual(len(self.records), 141)

    def test_indexes_are_sequential(self):
        self.assertEqual(
            [r["feature_index"] for r in self.records], list(range(1, 142))
        )

    def test_every_status_valid(self):
        for r in self.records:
            self.assertIn(r["status"], fl.VALID_STATUSES, r["feature"])

    def test_no_unset(self):
        for r in self.records:
            self.assertNotIn(r["status"], ("UNSET", "", None), r["feature"])

    def test_order_matches_schema(self):
        self.assertEqual(
            [r["feature"] for r in self.records], fl.FEATURES_141
        )

    def test_metadata_complete(self):
        for r in self.records:
            self.assertTrue(r["group"], r["feature"])
            self.assertTrue(r["dataset"], r["feature"])
            self.assertTrue(r["era_start"], r["feature"])
            self.assertTrue(r["era_end"], r["feature"])

    def test_runtime_verified_defaults_false(self):
        self.assertTrue(all(r["runtime_verified"] is False for r in self.records))

    def test_columns_declared(self):
        for column in fl.LEDGER_COLUMNS:
            self.assertIn(column, self.records[0])

    def test_fwi_seven_are_historical_nan(self):
        fwi = {"FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "FWI_ANOMALY_Z"}
        for r in self.records:
            if r["feature"] in fwi:
                self.assertEqual(r["status"], fl.S_HIST_NAN, r["feature"])
                self.assertEqual(r["phase"], fl.PHASE_P3D, r["feature"])

    def test_four_absent_groups_fully_unimplemented(self):
        # Two groups left this set as the project progressed:
        #   P3a  INDUSTRIAL -> NEW_IMPLEMENTATION (industrial_features.py)
        #   P4a  QUALITY    -> NEW_IMPLEMENTATION (lineage_features.py)
        # Neither is ever a recovered historical implementation.
        absent = []
        for group in (
            fl.GROUP_LANDCOVER,
            fl.GROUP_SMOKE,
            fl.GROUP_SAR,
            fl.GROUP_ATMOSPHERIC,
        ):
            absent.extend(fl.FEATURE_GROUPS[group])

        self.assertEqual(len(absent), 39)
        for r in self.records:
            if r["feature"] in absent:
                self.assertEqual(
                    r["status"], fl.S_MISSING_IMPL, r["feature"]
                )

    def test_lineage_fifty_four_are_new_implementation(self):
        """P4a implemented THERMAL + TEMPORAL + SPREAD + QUALITY from lineage.

        This also pins the supersession: those 54 were previously reported as
        UNVERIFIED_HISTORICAL on group inference alone, which was an overclaim.
        """
        lineage = []
        for group in (
            fl.GROUP_THERMAL,
            fl.GROUP_TEMPORAL,
            fl.GROUP_SPREAD,
            fl.GROUP_QUALITY,
        ):
            lineage.extend(
                f
                for f in fl.FEATURE_GROUPS[group]
                if f not in fl._PLACEHOLDERS
            )

        self.assertEqual(len(lineage), 54)
        for r in self.records:
            if r["feature"] in lineage:
                # The 4 recurrence features are the documented exception.
                if r["feature"] in {
                    "RECURRENCE_RATE_1Y",
                    "RECURRENCE_RATE_3Y",
                    "RECURRENCE_RATE_5Y",
                    "MAX_INTER_EVENT_GAP_LOG",
                }:
                    self.assertEqual(r["status"], fl.S_MISSING_IMPL, r["feature"])
                    continue
                self.assertEqual(r["status"], fl.S_NEW_IMPL, r["feature"])
                self.assertEqual(r["impl_ref"], "p4a_lineage_features", r["feature"])

    def test_no_feature_is_unverified_in_the_lineage_groups(self):
        """The UNVERIFIED_HISTORICAL overclaim must not come back."""
        unverified = {r["feature"] for r in self.records if r["status"] == fl.S_UNVERIFIED}
        for group in (
            fl.GROUP_THERMAL,
            fl.GROUP_TEMPORAL,
            fl.GROUP_SPREAD,
            fl.GROUP_QUALITY,
        ):
            for name in fl.FEATURE_GROUPS[group]:
                self.assertNotIn(name, unverified, name)

    def test_recurrence_four_need_phase_p3g(self):
        recurrence = {
            "RECURRENCE_RATE_1Y",
            "RECURRENCE_RATE_3Y",
            "RECURRENCE_RATE_5Y",
            "MAX_INTER_EVENT_GAP_LOG",
        }
        for r in self.records:
            if r["feature"] in recurrence:
                self.assertEqual(r["phase"], fl.PHASE_P3G, r["feature"])
                # Not implemented: a single-event lineage pass cannot count
                # neighbouring events.
                self.assertEqual(r["status"], fl.S_MISSING_IMPL, r["feature"])

    def test_industrial_eighteen_are_new_implementation(self):
        """P3a wrote the 18 industrial features; they are never historical."""
        industrial = fl.FEATURE_GROUPS[fl.GROUP_INDUSTRIAL]
        self.assertEqual(len(industrial), 18)
        for r in self.records:
            if r["feature"] in industrial:
                self.assertEqual(r["status"], fl.S_NEW_IMPL, r["feature"])
                self.assertEqual(r["phase"], fl.PHASE_P2, r["feature"])


    def test_era_gating(self):
        for r in self.records:
            if r["feature"] in (
                "NO2_ANOMALY_Z",
                "SO2_ANOMALY_Z",
                "CO_ANOMALY_Z",
                "CH4_ANOMALY_Z",
                "HCHO_ANOMALY_Z",
                "AOD_ANOMALY_Z",
                "CO_NO2_RATIO_LOG",
                "SO2_NO2_RATIO_LOG",
                "CH4_CO_RATIO_LOG",
            ):
                self.assertEqual(r["era_start"], "2018-07-02", r["feature"])
            if r["feature"] == "LIGHTNING_COUNT_24H":
                self.assertEqual(r["era_start"], "2017-03-01")
                self.assertEqual(r["era_end"], "2023-11-16")
            if r["feature"] == "VV_CHANGE":
                self.assertEqual(r["era_start"], "2014-05-03")


class TestDriftCheck(unittest.TestCase):
    def test_identical_schema_passes(self):
        result = fl.check_schema_drift(list(fl.FEATURES_141))
        self.assertEqual(result["status"], "OK")
        self.assertTrue(result["same_order"])
        self.assertEqual(result["n_features"], 141)

    def test_none_reports_no_candidate(self):
        self.assertEqual(
            fl.check_schema_drift(None)["status"], "NO_CANDIDATE"
        )

    def test_short_list_raises(self):
        with self.assertRaises(ValueError):
            fl.check_schema_drift(fl.FEATURES_141[:140])

    def test_renamed_feature_raises(self):
        broken = list(fl.FEATURES_141)
        broken[0] = "FRP_MEAN"
        with self.assertRaises(ValueError):
            fl.check_schema_drift(broken)

    def test_extra_feature_raises(self):
        with self.assertRaises(ValueError):
            fl.check_schema_drift(list(fl.FEATURES_141) + ["EXTRA"])

    def test_reorder_raises(self):
        broken = list(fl.FEATURES_141)
        broken[0], broken[1] = broken[1], broken[0]
        with self.assertRaises(ValueError):
            fl.check_schema_drift(broken)

    def test_131_feature_list_raises(self):
        # The retired "131 feature" list must be rejected outright.
        with self.assertRaises(ValueError):
            fl.check_schema_drift(list(fl.FEATURES_141)[:131])


class TestLocalInputHints(unittest.TestCase):
    def test_every_dataset_has_hints_or_is_derived(self):
        derived_strings = {
            fl.DATASET[f] for f in fl.DERIVED_NO_INPUT
        }
        for dataset in set(fl.DATASET.values()):
            self.assertTrue(
                dataset in fl.LOCAL_INPUT_HINTS or dataset in derived_strings,
                f"no local-input hints for dataset: {dataset}",
            )

    def test_every_feature_covered(self):
        for f in fl.FEATURES_141:
            self.assertIsInstance(fl.hints_for(f), list, f)

    def test_derived_features_need_no_input(self):
        for f in fl.DERIVED_NO_INPUT:
            self.assertEqual(fl.hints_for(f), [], f)

    def test_no_gee_required_subset_of_schema(self):
        for f in fl.NO_GEE_REQUIRED:
            self.assertIn(f, fl.FEATURES_141, f)

    def test_atmospheric_features_do_not_match_bare_co(self):
        # Guards against a future edit adding a token that matches
        # "config" / "count" and would falsely flag atmospheric data.
        for token in fl.hints_for("NO2_ANOMALY_Z"):
            self.assertNotIn(token, ("co", "c", "o2"), token)

    def test_sar_tokens_are_not_bare_s1(self):
        for token in fl.hints_for("VV_CHANGE"):
            self.assertNotEqual(token, "s1")


class TestVersionAndKnownSources(unittest.TestCase):
    def test_version_is_declared(self):
        self.assertTrue(fl.LEDGER_VERSION)
        self.assertRegex(fl.LEDGER_VERSION, r"^\d+\.\d+\.\d+$")

    def test_known_external_sources_tokens_resolve(self):
        for folder, tokens in fl.KNOWN_EXTERNAL_SOURCES.items():
            self.assertTrue(tokens, folder)
            for token in tokens:
                self.assertIsInstance(token, str, folder)

    def test_size_floor_keys_are_known_folders(self):
        for folder in fl.SIZE_SANITY_FLOOR_BYTES:
            self.assertIn(
                folder,
                fl.KNOWN_EXTERNAL_SOURCES,
                f"size floor declared for unknown folder {folder}",
            )

    def test_local_input_hints_non_empty_values(self):
        for dataset, tokens in fl.LOCAL_INPUT_HINTS.items():
            self.assertTrue(tokens, dataset)
            self.assertTrue(all(isinstance(t, str) and t for t in tokens))


class TestSelfConsistency(unittest.TestCase):
    def test_dataset_defined_for_every_feature(self):
        for f in fl.FEATURES_141:
            self.assertIn(f, fl.DATASET, f)

    def test_status_map_defined_for_every_feature(self):
        for f in fl.FEATURES_141:
            self.assertIn(f, fl.STATUS_MAP, f)

    def test_validate_schema_is_idempotent(self):
        fl.validate_schema()
        fl.validate_schema()


if __name__ == "__main__":
    unittest.main(verbosity=2)