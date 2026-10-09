"""
Tests for the cell runner.

The runner exists because this project has paid for stale cell bodies five
times. Its job is narrow and the tests are correspondingly narrow: find the
file on Drive, report the version it declares, and execute it in the CALLER's
namespace so a variable set in the notebook is visible to the cell.

The namespace behaviour is the subtle one. `run()` reads `sys._getframe(1)`
by default; if that were ever wrong, `EARTHENGINE_PROJECT` set in a notebook
would silently vanish and the cell would report "no project found" for reasons
that have nothing to do with the cell.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import colab_runcell as cr

CELL_OK = '''
P7E_CELL_VERSION = "2026-10-04.4"
SEEN = None
def configure(project_id):
    global SEEN
    SEEN = project_id
'''

CELL_RAISES = '''
raise ValueError("deliberate")
'''

CELL_USES_GLOBAL = '''
GOT = globals().get("EARTHENGINE_PROJECT", "<missing>")
'''

CELL_NO_VERSION = '''
print("no marker here")
'''


class VersionParsingTests(unittest.TestCase):
    def test_it_reads_a_cell_version(self):
        self.assertEqual(cr.declared_version(CELL_OK), "2026-10-04.4")

    def test_it_reads_a_dunder_version(self):
        self.assertEqual(cr.declared_version('__version__ = "1.2.0"'),
                         "1.2.0")

    def test_a_cell_with_no_marker_returns_none(self):
        """None, not a guess. A fabricated version would defeat the purpose."""
        self.assertIsNone(cr.declared_version(CELL_NO_VERSION))

    def test_it_handles_single_quotes(self):
        self.assertEqual(
            cr.declared_version('P7E_CELL_VERSION = \'2026-01-01.1\''),
            "2026-01-01.1")


class NamespaceTests(unittest.TestCase):
    """The cell must see variables the notebook set. This is the subtle part."""

    def _run_with(self, src, namespace):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "cell.py").write_text(src)
            with mock.patch.object(cr, "SEARCH_ROOTS", (root,)):
                return cr.run("cell", namespace=namespace)

    def test_the_cell_reads_a_variable_from_the_caller_namespace(self):
        ns = {"EARTHENGINE_PROJECT": "my-project"}
        self._run_with(CELL_USES_GLOBAL, ns)
        self.assertEqual(ns.get("GOT"), "my-project")

    def test_a_missing_variable_is_reported_not_defaulted_silently(self):
        ns = {}
        self._run_with(CELL_USES_GLOBAL, ns)
        self.assertEqual(ns.get("GOT"), "<missing>")

    def test_a_function_the_cell_defines_lands_in_the_caller_namespace(self):
        ns = {}
        rep = self._run_with(CELL_OK, ns)
        self.assertTrue(rep["ok"])
        ns["configure"]("proj-2")
        self.assertEqual(ns.get("SEEN"), "proj-2")

    def test_the_file_name_is_set(self):
        ns = {}
        self._run_with(CELL_NO_VERSION, ns)
        self.assertTrue(str(ns["__file__"]).endswith("cell.py"))

    def test_a_passing_namespace_is_not_modified_in_place_unexpectedly(self):
        """An explicit namespace is used verbatim, not copied -- so a caller
        can pass a dict and inspect what the cell left behind."""
        ns = {}
        rep = self._run_with(CELL_OK, ns)
        self.assertIn("cell", rep)
        self.assertIn("P7E_CELL_VERSION", ns)


class FailureTests(unittest.TestCase):
    """A failing cell must not take the notebook down with it."""

    def _run(self, src):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "cell.py").write_text(src)
            with mock.patch.object(cr, "SEARCH_ROOTS", (root,)):
                return cr.run("cell", namespace={})

    def test_a_raising_cell_is_reported_not_raised(self):
        rep = self._run(CELL_RAISES)
        self.assertFalse(rep["ok"])
        self.assertIn("ValueError", rep["error"])
        self.assertIn("deliberate", rep["error"])

    def test_a_raising_cell_still_returns_its_version(self):
        rep = self._run('P7E_CELL_VERSION = "9.9.9"\nraise RuntimeError("x")')
        self.assertEqual(rep["version"], "9.9.9")

    def test_a_missing_file_names_every_path_tried(self):
        """'File not found' without the list sends people back to pasting."""
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(cr, "SEARCH_ROOTS", (Path(d),)):
                with self.assertRaises(cr.CellNotFound) as ctx:
                    cr.locate("nope")
        msg = str(ctx.exception)
        self.assertIn("Do NOT paste", msg)
        self.assertIn("colab_bootstrap.py", msg)
        # Every attempted path must be named, or the message is unactionable.
        self.assertIn("nope.py", msg)

    def test_the_extension_is_optional(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "thing.py").write_text("X = 1")
            with mock.patch.object(cr, "SEARCH_ROOTS", (root,)):
                self.assertEqual(cr.locate("thing").name, "thing.py")
                self.assertEqual(cr.locate("thing.py").name, "thing.py")


class StatusTests(unittest.TestCase):
    def test_status_reports_version_without_executing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "a.py").write_text(CELL_OK)
            with mock.patch.object(cr, "SEARCH_ROOTS", (root,)):
                out = cr.status("a", "missing_one")
        self.assertTrue(out[0]["found"])
        self.assertEqual(out[0]["version"], "2026-10-04.4")
        self.assertFalse(out[1]["found"])
        self.assertIn("error", out[1])

    def test_status_does_not_run_the_cell(self):
        marker = Path(tempfile.gettempdir()) / "colab_runcell_should_not_exist"
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "w.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('executed')\n"
                'P7E_CELL_VERSION = "1"')
            with mock.patch.object(cr, "SEARCH_ROOTS", (root,)):
                cr.status("w")
        self.assertFalse(marker.exists(),
                         "status() must not execute the cell")


class DuplicateSuffixTests(unittest.TestCase):
    """macOS appends ' (1)' when a file is copied twice. That copy can shadow
    the good one, which is how a stale version gets picked up."""

    def test_it_will_accept_a_suffixed_copy(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "cell (1).py").write_text('V = "1"')
            with mock.patch.object(cr, "SEARCH_ROOTS", (root,)):
                found = cr.locate("cell")
        self.assertEqual(found.name, "cell (1).py")

    def test_an_exact_name_is_preferred_over_a_suffixed_copy(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "cell.py").write_text('V = "exact"')
            (root / "cell (1).py").write_text('V = "copy"')
            with mock.patch.object(cr, "SEARCH_ROOTS", (root,)):
                src, _p = cr.source_of("cell")
        self.assertIn("exact", src)


class BootstrapCellVersionTests(unittest.TestCase):
    """The bootstrap must verify CELL versions, not just module attributes.

    The incident this pins: a 14,229-byte copy of `colab_p7e_gee_probe` was
    uploaded against a 21,060-byte repo file. The bootstrap passed -- modules
    were all correct -- and the cell then failed with
    `Asset 'ESA/WorldCover/v200' is not an Image`, which reads as a dataset
    bug and was actually a file-transfer bug. `hasattr` cannot catch this: a
    cell is a script, so an old one runs perfectly and fails on content.
    """

    def _bootstrap(self):
        return (Path(__file__).resolve().parent
                / "colab_bootstrap.py").read_text()

    def test_it_declares_a_version_floor_for_every_cell(self):
        import re

        src = self._bootstrap()
        floors = re.findall(r'"(colab_p7[a-z0-9_]+)":\s*"([^"]+)"', src)
        self.assertGreaterEqual(len(floors), 6, floors)
        for _cell, _v in floors:
            self.assertTrue(_v, f"{_cell} has no version floor")

    def test_no_cell_is_exempt(self):
        """`None` means uncheckable, which is the original problem."""
        import re

        src = self._bootstrap()
        exempt = re.findall(r'"(colab_p7[a-z0-9_]+)":\s*None', src)
        self.assertEqual(exempt, [],
                         f"cells still exempt from version checking: {exempt}")

    def test_it_reads_versions_back_off_drive(self):
        src = self._bootstrap()
        self.assertIn("CELL VERSIONS", src)
        self.assertIn("PIPELINE / f", src,
                      "must read the file back off Drive, not trust the upload")

    def test_it_aborts_on_a_stale_cell(self):
        self.assertIn("STALE CELLS", self._bootstrap())
        self.assertIn("raise SystemExit(1)", self._bootstrap())

    def test_every_cell_on_disk_declares_a_version(self):
        """And the floors match what the files actually declare."""
        import re

        here = Path(__file__).resolve().parent
        src = self._bootstrap()
        floors = dict(re.findall(r'"(colab_p7[a-z0-9_]+)":\s*"([^"]+)"', src))
        for cell, floor in floors.items():
            path = here / f"{cell}.py"
            self.assertTrue(path.exists(), f"{cell}.py is missing from the repo")
            text = path.read_text()
            declared = cr.declared_version(text)
            self.assertIsNotNone(
                declared,
                f"{cell}.py has no version marker but the bootstrap demands "
                f"{floor}; add one or the cell cannot be verified")
            self.assertGreaterEqual(_tuple(declared), _tuple(floor),
                                    f"{cell}.py declares {declared}, "
                                    f"bootstrap requires >= {floor}")


def _tuple(v):
    return tuple(int("".join(c for c in ch if c.isdigit()) or 0)
                 for ch in str(v).replace("-", ".").split("."))


if __name__ == "__main__":
    unittest.main()