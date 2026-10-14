"""
Tests for project-root resolution in colab_utils.

The bug this guards against is concrete: the real tree appeared as
`Firm_Clusters (1)`, a macOS-style duplicate, while every cell hardcoded
`/content/drive/MyDrive/Firm_Clusters`. Resolution is by content -- find the
directory that actually contains `full_2012_2026` -- never by exact name.

These tests build a synthetic Drive tree under a temp dir and point MYDRIVE at
it, so no Google Drive is involved.

Run:  python3 -m unittest test_colab_root
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import colab_utils as cu


class ProjectRootBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fakedrive_"))
        self.mydrive = self.tmp / "MyDrive"
        self.mydrive.mkdir(parents=True)
        patcher = mock.patch.object(cu, "MYDRIVE", self.mydrive)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def make_tree(self, dirname, *, external=True, events=True, clusters=True):
        root = self.mydrive / dirname
        (root / "full_2012_2026").mkdir(parents=True)
        if external:
            (root / "full_2012_2026" / "external" / "osm_facilities").mkdir(
                parents=True
            )
        if events:
            (root / "full_2012_2026" / "events").mkdir(parents=True)
        if clusters:
            (root / "full_2012_2026" / "clusters").mkdir(parents=True)
        return root

    def touch(self, root, rel):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
        return path


class TestSuffixRanking(unittest.TestCase):
    def test_exact_name_outranks_any_duplicate(self):
        names = ["Firm_Clusters (1)", "Firm_Clusters"]
        self.assertEqual(
            sorted(names, key=cu._suffix_rank)[0], "Firm_Clusters"
        )

    def test_lowest_duplicate_number_wins(self):
        names = ["Firm_Clusters (7)", "Firm_Clusters (2)"]
        self.assertEqual(
            sorted(names, key=cu._suffix_rank)[0], "Firm_Clusters (2)"
        )

    def test_unrelated_names_are_not_mangled(self):
        self.assertEqual(cu._suffix_rank("Project2"), (0, 0, "Project2"))

    def test_name_without_digits_in_parens_is_exact(self):
        # "Firm_Clusters (backup)" is not a numeric duplicate.
        self.assertEqual(cu._suffix_rank("Firm_Clusters (backup)"),
                         (0, 0, "Firm_Clusters (backup)"))


class TestResolution(ProjectRootBase):
    def test_finds_the_canonical_directory(self):
        expected = self.make_tree("Firm_Clusters")
        self.assertEqual(
            cu.resolve_project_dir(verbose=False), expected
        )

    def test_finds_a_suffixed_duplicate(self):
        """The exact scenario: the tree only exists as 'Firm_Clusters (1)'."""
        expected = self.make_tree("Firm_Clusters (1)")
        self.assertEqual(
            cu.resolve_project_dir(verbose=False), expected
        )

    def test_prefers_the_exact_name_when_both_exist(self):
        self.make_tree("Firm_Clusters (1)")
        exact = self.make_tree("Firm_Clusters")
        self.assertEqual(cu.resolve_project_dir(verbose=False), exact)

    def test_ignores_a_partial_duplicate(self):
        # A duplicate missing events/ must not win over a complete tree.
        self.make_tree("Firm_Clusters (1)", events=False)
        complete = self.make_tree("Firm_Clusters")
        self.assertEqual(cu.resolve_project_dir(verbose=False), complete)

    def test_complete_suffixed_wins_over_incomplete_exact_name(self):
        """The live situation: a hand-made `Firm_Clusters/` holding only
        pipeline/ and qc/, alongside `Firm_Clusters (1)/` holding the real
        full_2012_2026. Name preference must NOT win here -- completeness
        does, or the cells point at an empty folder again.
        """
        stub = self.mydrive / "Firm_Clusters"
        (stub / "pipeline").mkdir(parents=True)
        (stub / "qc").mkdir(parents=True)

        real = self.make_tree("Firm_Clusters (1)")

        self.assertEqual(cu.resolve_project_dir(verbose=False), real)

    def test_two_complete_trees_prefers_exact_name(self):
        # Both are complete, so the canonical name wins. Completeness is the
        # gate; the name is only the tie-break among complete candidates.
        self.make_tree("Firm_Clusters (1)")
        exact = self.make_tree("Firm_Clusters")
        self.assertEqual(cu.resolve_project_dir(verbose=False), exact)

    def test_stub_named_exactly_is_reported_as_incomplete(self):
        stub = self.mydrive / "Firm_Clusters"
        (stub / "pipeline").mkdir(parents=True)
        self.make_tree("Firm_Clusters (2)")
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cu.resolve_project_dir()
        text = buf.getvalue()
        self.assertIn("[no  ]", text)          # the stub scored 0/2
        self.assertIn("Firm_Clusters (2)", text)

    def test_returns_none_when_nothing_matches(self):
        self.assertIsNone(
            cu.resolve_project_dir(verbose=False, allow_refresh=False)
        )

    def test_ignores_unrelated_directories(self):
        (self.mydrive / "Project2").mkdir()
        (self.mydrive / "X_event_feature_inventory").mkdir()
        self.assertIsNone(
            cu.resolve_project_dir(verbose=False, allow_refresh=False)
        )

    def test_finds_the_tree_under_a_totally_different_name(self):
        root = self.make_tree("archive_2024_run")
        self.assertEqual(cu.resolve_project_dir(verbose=False), root)

    def test_single_required_subpath(self):
        root = self.make_tree("Firm_Clusters (2)", events=False, clusters=False)
        found = cu.resolve_project_dir(
            required="full_2012_2026/external", verbose=False
        )
        self.assertEqual(found, root)

    def test_verbose_output_mentions_the_winner(self):
        self.make_tree("Firm_Clusters (1)")
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cu.resolve_project_dir()
        text = buf.getvalue()
        self.assertIn("Firm_Clusters (1)", text)
        self.assertIn("resolved project root", text)

    def test_missing_drive_yields_none(self):
        with mock.patch.object(cu, "path_ok", lambda p: False):
            with mock.patch.object(cu, "is_dir", lambda p: False):
                self.assertIsNone(
                cu.resolve_project_dir(verbose=False, allow_refresh=False)
            )

    def test_dead_mount_does_not_crash_resolution(self):
        import errno as _errno

        def boom(*args, **kwargs):
            raise OSError(_errno.ENOTCONN, "Transport endpoint is not connected")

        with mock.patch.object(cu, "path_ok", lambda p: True):
            with mock.patch.object(Path, "iterdir", boom):
                self.assertIsNone(cu.resolve_project_dir(verbose=False))


class TestArtefactPaths(ProjectRootBase):
    def test_builds_paths_under_the_resolved_root(self):
        root = self.make_tree("Firm_Clusters (1)")
        resolved = cu.resolve_project_dir(verbose=False)
        external = resolved / "full_2012_2026" / "external"
        membership = (
            resolved
            / "full_2012_2026"
            / "events"
            / "event_local_cluster_membership.parquet"
        )
        self.touch(root, "full_2012_2026/events/event_local_cluster_membership.parquet")
        self.assertTrue(cu.path_ok(external))
        self.assertTrue(cu.path_ok(membership))

    def test_canonical_subpath_constant_is_stable(self):
        self.assertEqual(cu.CANONICAL_SUBPATH, "full_2012_2026")
        self.assertEqual(cu.PROJECT_DIRNAME, "Firm_Clusters")


class TestArtefactResolution(ProjectRootBase):
    """The real layout does not nest the way the handoff assumed."""

    def test_finds_clusters_at_the_top_level(self):
        """clusters/ lives beside full_2012_2026/, not inside it."""
        root = self.make_tree("Firm_Clusters")
        (root / "clusters").mkdir(exist_ok=True)
        found = cu.resolve_artefacts(root, wanted=("clusters",), verbose=False)
        self.assertEqual(found["clusters"], root / "clusters")

    def test_finds_clusters_nested_under_the_canonical_dir(self):
        # make_tree already creates clusters/ inside full_2012_2026.
        root = self.make_tree("Firm_Clusters", clusters=False)
        nested = root / "full_2012_2026" / "clusters"
        nested.mkdir()
        found = cu.resolve_artefacts(root, wanted=("clusters",), verbose=False)
        self.assertEqual(found["clusters"], nested)

    def test_ambiguous_leaf_name_is_reported_not_silently_chosen(self):
        """clusters/ exists BOTH at the top level and inside full_2012_2026/.

        Picking one silently is the exact failure this whole resolver exists to
        prevent: either choice yields plausible-looking output from whichever
        directory happens to be emptier or staler. So ambiguity must be raised.
        """
        root = self.make_tree("Firm_Clusters", clusters=False)
        (root / "clusters").mkdir(exist_ok=True)
        (root / "full_2012_2026" / "clusters").mkdir(exist_ok=True)
        matches = cu.find_all_artefact(root, "clusters")
        self.assertEqual(len(matches), 2)
        ambiguous = cu.resolve_artefacts(
            root, wanted=("clusters",), verbose=False
        )
        self.assertEqual(ambiguous["clusters"], root / "clusters")
        self.assertIn("clusters", ambiguous["ambiguous"])

    def test_finds_the_real_layout_fully(self):
        """Reproduces the screenshot: clusters/edges at top level,
        external/ and events/ inside full_2012_2026/."""
        root = self.make_tree("Firm_Clusters")
        for leaf in ("clusters", "edges", "global", "logs"):
            (root / leaf).mkdir(exist_ok=True)
        found = cu.resolve_artefacts(root, verbose=False)
        for leaf in ("clusters", "edges", "external", "events"):
            self.assertIsNotNone(found[leaf], leaf)

    def test_missing_artefact_is_none_not_a_guess(self):
        root = self.make_tree("Firm_Clusters")
        found = cu.resolve_artefacts(root, wanted=("edges",), verbose=False)
        self.assertIsNone(found["edges"])

    def test_find_artefact_reports_depth(self):
        root = self.make_tree("Firm_Clusters")
        nested = root / "full_2012_2026" / "external"
        path, depth = cu.find_artefact(root, "external")
        self.assertEqual(path, nested)
        self.assertEqual(depth, 2)

    def test_find_artefact_returns_none_when_absent(self):
        root = self.make_tree("Firm_Clusters")
        self.assertEqual(cu.find_artefact(root, "nonexistent"), (None, None))

    def test_resolution_is_deterministic(self):
        root = self.make_tree("Firm_Clusters")
        (root / "full_2012_2026" / "a_clusters").mkdir()
        (root / "full_2012_2026" / "b_clusters").mkdir()
        first = cu.find_artefact(root, "a_clusters")[0]
        second = cu.find_artefact(root, "b_clusters")[0]
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first, second)

    def test_missing_root_is_safe(self):
        self.assertIsNone(
            cu.resolve_artefacts(self.mydrive / "nope", verbose=False)["clusters"]
        )

    def test_verbose_lists_missing_with_advice(self):
        import io
        import contextlib

        root = self.make_tree("Firm_Clusters")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cu.resolve_artefacts(root, wanted=("edges",))
        text = buf.getvalue()
        self.assertIn("MISSING", text)
        self.assertIn("survey_tree", text)


class TestPickByFileCount(ProjectRootBase):
    def test_picks_the_copy_with_more_files(self):
        """The live ambiguity: a 3-file top-level clusters/ and the real
        5,337-file one under full_2012_2026/. Depth would pick the wrong one."""
        root = self.make_tree("Firm_Clusters", clusters=False)
        real = root / "full_2012_2026" / "clusters"
        real.mkdir()
        for i in range(40):
            (real / f"2021{i:04d}.parquet").write_text("x")
        stub = root / "clusters"
        stub.mkdir()
        for i in range(3):
            (stub / f"2021{i:04d}.parquet").write_text("x")

        picked, table = cu.pick_by_file_count(root, "clusters")
        self.assertEqual(picked, real)
        self.assertEqual(len(table), 2)
        self.assertEqual(table[0]["n_files"], 40)

    def test_table_is_sorted_by_file_count_descending(self):
        root = self.make_tree("Firm_Clusters", clusters=False)
        (root / "full_2012_2026" / "clusters").mkdir()
        (root / "clusters").mkdir()
        for d, n in (("clusters", 2), ("full_2012_2026/clusters", 9)):
            for i in range(n):
                (root / d / f"f{i}.parquet").write_text("x")
        _, table = cu.pick_by_file_count(root, "clusters")
        counts = [row["n_files"] for row in table]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_single_copy_is_returned(self):
        root = self.make_tree("Firm_Clusters", clusters=False)
        only = root / "full_2012_2026" / "clusters"
        only.mkdir()
        (only / "a.parquet").write_text("x")
        picked, table = cu.pick_by_file_count(root, "clusters")
        self.assertEqual(picked, only)
        self.assertEqual(len(table), 1)

    def test_absent_returns_none_with_empty_table(self):
        root = self.make_tree("Firm_Clusters")
        picked, table = cu.pick_by_file_count(root, "edges")
        self.assertIsNone(picked)
        self.assertEqual(table, [])


class TestStaleListingRecovery(ProjectRootBase):
    """A folder that exists in the Drive browser can be invisible to a stale
    FUSE cache. Resolution must refresh and retry, not give up and not guess."""

    def test_refresh_then_retry_finds_a_late_arriving_tree(self):
        """First listing sees an incomplete Firm_Clusters; after the refresh
        the canonical subpath is there. This is the stale-FUSE-cache case."""
        # Incomplete to begin with; it only becomes complete once the "stale"
        # cache is dropped, which is what the refresh stands in for.
        root = self.make_tree("Firm_Clusters", external=False, events=False)

        calls = {"n": 0}
        real = cu.candidate_project_dirs

        def fake_candidates():
            calls["n"] += 1
            if calls["n"] == 1:
                return [root]           # stale: subpaths not yet visible
            (root / "full_2012_2026" / "external").mkdir(parents=True, exist_ok=True)
            (root / "full_2012_2026" / "events").mkdir(parents=True, exist_ok=True)
            return real()              # fresh: now complete

        with mock.patch.object(cu, "candidate_project_dirs", fake_candidates):
            with mock.patch.object(cu, "force_refresh_drive", lambda: True):
                found = cu.resolve_project_dir(
                    verbose=False, allow_refresh=True, retry_delays=(0, 0, 0)
                )
        self.assertEqual(found, root)
        self.assertGreaterEqual(calls["n"], 2)

    def test_refresh_is_attempted_when_nothing_matches(self):
        self.make_tree("Firm_Clusters", external=False, events=False)
        calls = []
        with mock.patch.object(
            cu, "force_refresh_drive", lambda: calls.append(1) or True
        ):
            cu.resolve_project_dir(
                verbose=False, allow_refresh=True, retry_delays=(0, 0, 0)
            )
        self.assertEqual(len(calls), 1)

    def test_gives_up_cleanly_when_nothing_ever_appears(self):
        self.make_tree("Firm_Clusters", external=False, events=False)
        with mock.patch.object(cu, "force_refresh_drive", lambda: True):
            found = cu.resolve_project_dir(
                verbose=False, allow_refresh=True, retry_delays=(0, 0, 0)
            )
        self.assertIsNone(found)

    def test_allow_refresh_false_skips_the_remount(self):
        self.make_tree("Firm_Clusters", external=False, events=False)
        calls = []
        with mock.patch.object(cu, "force_refresh_drive", lambda: calls.append(1) or True):
            found = cu.resolve_project_dir(verbose=False, allow_refresh=False)
        self.assertIsNone(found)
        self.assertEqual(calls, [])

    def test_off_drive_refresh_is_a_no_op_success(self):
        # Locally /content/drive does not exist, so there is nothing to remount
        # and the refresh counts as done rather than as a failure.
        self.assertTrue(cu.force_refresh_drive())

    def test_a_refresh_failure_is_reported_not_raised(self):
        fake = type("M", (), {})()
        with mock.patch.object(cu, "path_ok", lambda p: True):
            with mock.patch.dict(
                "sys.modules",
                {"google": fake, "google.colab": fake,
                 "google.colab.drive": type("D", (), {"mount": staticmethod(
                     lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))})()},
            ):
                self.assertFalse(cu.force_refresh_drive())

    def test_wait_for_returns_on_first_success(self):
        calls = {"n": 0}

        def pred():
            calls["n"] += 1
            return calls["n"] >= 2

        self.assertTrue(cu.wait_for(pred, delays=(0, 0, 0), label="test"))


if __name__ == "__main__":
    unittest.main()
