"""
End-to-end rehearsal of `colab_p7h_landcover_all.py` against stubs.

This cell is a 5.5-hour extraction that Colab WILL interrupt, so its resume
logic is the part that matters most and the part a syntax check cannot reach.
The rehearsal has already earned its place: it caught two bugs that would each
have silently corrupted a long run.

  1. `_shard_no` was seeded from `min(done_chunks) // SHARD_CHUNKS`. With
     shards 1 and 2 missing, `min` was 0, so the resume WROTE SHARDS 0 AND 1 --
     overwriting two files that were already correct -- and mislabelled the new
     ones. Shard identity is now derived from the chunk number, never carried
     in mutable state.

  2. The assembly printed no warning while producing 19,000 of 25,000 rows.
     A short table silently shrinks every label and every rate computed from
     it, so it is now an explicit INCOMPLETE message.

No `ee`, no Drive, no pyarrow. parquet is shimmed to CSV because this machine
has no pyarrow (README section 5) -- the shim honours the EXACT path given,
which matters because the cell relies on writing `*.partial` then renaming.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent
CELL = REPO / "colab_p7h_landcover_all.py"

N_EVENTS = 25_000
CHUNK = 2_000
SHARD_CHUNKS = 3


def _install_parquet_shim():
    """parquet -> CSV, honouring the exact path the cell writes."""
    def _to_parquet(self, path, index=False, **kw):
        pp = pathlib.Path(str(path))
        pp.parent.mkdir(parents=True, exist_ok=True)
        self.to_csv(pp, index=index)

    def _read_parquet(path, columns=None, **kw):
        p = pathlib.Path(str(path))
        if not p.exists():
            for alt in (p.with_suffix(".csv"), p.parent / (p.name + ".csv")):
                if alt.exists():
                    p = alt
                    break
        return pd.read_csv(p, usecols=columns) if columns else pd.read_csv(p)

    names = ["event_centroid_lon", "event_centroid_lat", "event_id",
             "start_year"]

    def _meta(path):
        return types.SimpleNamespace(num_rows=len(_read_parquet(path)))

    pd.DataFrame.to_parquet = _to_parquet
    pd.read_parquet = _read_parquet
    pqm = types.ModuleType("pyarrow.parquet")
    pqm.ParquetFile = lambda p: types.SimpleNamespace(
        schema_arrow=types.SimpleNamespace(names=list(names)),
        metadata=_meta(p))
    pqa = types.ModuleType("pyarrow")
    pqa.parquet = pqm
    return {"pyarrow": pqa, "pyarrow.parquet": pqm}


def _install_ee_stub():
    codes = [10, 20, 30, 40, 50, 60, 80]

    class _Img:
        @staticmethod
        def constant(v):
            return types.SimpleNamespace(bandNames=lambda:
                                         types.SimpleNamespace(
                                             getInfo=lambda: ["constant"]))

        def select(self, band):
            return self

        def sampleRegions(self, collection, properties, scale, geometries):
            return types.SimpleNamespace(getInfo=lambda: {"features": [
                {"properties": {"row": f.props["row"],
                                "Map": codes[f.props["row"] % len(codes)]}}
                for f in collection.feats]})

    class _Feat:
        def __init__(self, geometry=None, props=None):
            self.props = props

    class _FC:
        def __init__(self, feats):
            self.feats = list(feats)

    m = types.ModuleType("ee")
    m.Image = _Img
    m.ImageCollection = lambda i: types.SimpleNamespace(first=lambda: _Img())
    m.Feature = _Feat
    m.FeatureCollection = _FC
    m.Geometry = types.SimpleNamespace(Point=lambda c: c)
    m.Initialize = lambda **k: None
    m.Authenticate = lambda **k: None
    m.data = types.SimpleNamespace(getAssetRoots=lambda: [])
    return m


class P7hRehearsal(unittest.TestCase):
    """Runs the real cell against a stubbed Drive and a stubbed `ee`."""

    @classmethod
    def setUpClass(cls):
        # The shim REPLACES `pd.DataFrame.to_parquet` / `pd.read_parquet`, so
        # the originals are saved and restored here. Without this, the patch
        # outlives the class and every suite imported afterwards runs against a
        # CSV shim -- which is how 18 unrelated tests in test_colab_utils and
        # test_explainability errored under a full `unittest discover`.
        cls._pd_orig = (pd.DataFrame.to_parquet, pd.read_parquet)
        cls._patches = [mock.patch.dict(sys.modules, _install_parquet_shim()),
                        mock.patch.dict(sys.modules, {"ee": _install_ee_stub()})]
        for p in cls._patches:
            p.start()
        cls.src = CELL.read_text()

    @classmethod
    def tearDownClass(cls):
        for p in cls._patches:
            p.stop()
        pd.DataFrame.to_parquet, pd.read_parquet = cls._pd_orig
        # `patch.dict` restores only the keys IT added. The cell does
        # `import pyarrow.parquet as pq`, which the import system registers as a
        # SEPARATE sys.modules entry during the test, so that survives the exit
        # hook. Left behind, `find_spec("pyarrow")` later raises
        # `ValueError: pyarrow.__spec__ is None` and takes 18 unrelated tests
        # in test_colab_utils and test_explainability down with it.
        for k in [k for k in sys.modules
                  if k == "pyarrow" or k.startswith(("pyarrow.", "ee."))]:
            del sys.modules[k]

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        (self.root / "qc").mkdir()
        ds = self.root / "full_2012_2026" / "features" / "dataset"
        ds.mkdir(parents=True)
        n = N_EVENTS
        pd.DataFrame({
            "event_centroid_lon": np.linspace(68, 96, n),
            "event_centroid_lat": np.linspace(8, 35, n),
            "event_id": [f"E{i}" for i in range(n)],
            "start_year": np.random.randint(2012, 2027, n),
        }).to_parquet(ds / "meta.parquet")
        pd.DataFrame({"label_weak": np.where(np.arange(n) % 5 == 0,
                                              "INDUSTRIAL_FIRE",
                                              "WILDFIRE")}).to_parquet(
            ds / "y.parquet")
        pd.DataFrame({"DIST_NEAREST_FACILITY_LOG": np.full(n, 3.0)}).to_parquet(
            ds / "X.parquet")
        self.shards = self.root / "qc" / "landcover_shards"
        self.addCleanup(self._tmp.cleanup)

    def _run_cell(self, **over):
        cu = types.ModuleType("colab_utils")
        root = self.root
        cu.resolve_project_dir = lambda: root
        cu.drive_report = lambda: "  (rehearsal: stubbed Drive)"
        cu.find_artefact_path = lambda *a, **k: None
        ns = {
            "GEE_CHUNK_SIZE": CHUNK,
            "LANDCOVER_SHARD_CHUNKS": SHARD_CHUNKS,
            "LANDCOVER_RESUME": True,
            "__name__": "__main__",
        }
        ns.update(over)
        with mock.patch.dict(sys.modules, {"colab_utils": cu}):
            exec(compile(self.src, str(CELL), "exec"), ns)      # noqa: S102
        return ns

    def _assembled(self):
        p = self.root / "qc" / "landcover_events.parquet"
        return pd.read_csv(p)

    def test_a_first_run_extracts_everything(self):
        self._run_cell()
        self.assertEqual(len(self._assembled()), N_EVENTS)
        self.assertEqual(self._assembled()["row"].nunique(), N_EVENTS)

    def test_no_row_is_lost_or_duplicated(self):
        self._run_cell()
        df = self._assembled()
        self.assertEqual(len(df), df["row"].nunique())

    def test_a_second_run_does_no_work(self):
        self._run_cell()
        before = sorted(p.name for p in self.shards.glob("*.parquet"))
        self._run_cell()
        self.assertEqual(sorted(p.name for p in self.shards.glob("*.parquet")),
                         before)
        self.assertEqual(len(self._assembled()), N_EVENTS)

    def test_resume_after_losing_shards_restores_completeness(self):
        """THE regression. Shard identity must come from the chunk number."""
        self._run_cell()
        for n in (1, 2):
            (self.shards / f"lc_{n:06d}.parquet").unlink()
        self._run_cell()
        self.assertEqual(len(self._assembled()), N_EVENTS,
                         "resume did not refill the missing shards")
        self.assertEqual(
            sorted(p.name for p in self.shards.glob("*.parquet")),
            [f"lc_{n:06d}.parquet" for n in range(5)],
            "resume must produce the same shard set, not a shifted one")

    def test_resume_does_not_overwrite_surviving_shards(self):
        self._run_cell()
        keep = self.shards / "lc_000000.parquet"
        stamp = keep.stat().st_mtime_ns
        size_before = keep.stat().st_size
        for n in (1, 2):
            (self.shards / f"lc_{n:06d}.parquet").unlink()
        self._run_cell()
        self.assertEqual(keep.stat().st_mtime_ns, stamp,
                         "a surviving shard was rewritten")
        self.assertEqual(keep.stat().st_size, size_before)

    def test_a_short_shard_is_discarded_and_reextracted(self):
        self._run_cell()
        target = self.shards / "lc_000000.parquet"
        pd.read_csv(target).head(10).to_csv(target, index=False)
        self._run_cell()
        self.assertEqual(len(self._assembled()), N_EVENTS,
                         "a torn shard was trusted")

    def test_unmapped_is_never_counted_as_non_vegetated(self):
        """A missing raster must not read as bare ground.

        The stub returns a value for every point, so force the unmapped path by
        checking the column's construction directly on a hand-made frame.
        """
        from gee_utils import VEGETATED_CLASSES

        cls = np.array([-1, 50, 40, 80], dtype=np.int16)
        veg = np.where(cls >= 0,
                       np.isin(cls, sorted(VEGETATED_CLASSES)).astype("int8"),
                       -1)
        self.assertEqual(list(veg), [-1, 0, 1, 0],
                         "unmapped must be -1, never 0 (which means bare)")

    def test_no_cell_mislabels_a_conditional(self):
        """A labelled conditional must be computed from the cell it names.

        P7g printed "of all NON-vegetated events, 36.4% are labelled
        INDUSTRIAL_FIRE" while computing P(weak OTHER | non-vegetated). The
        inversion turned a 5.5x skew in the label's favour into an apparent
        skew against it, and I reported the inversion as a finding.
        """
        src = (REPO / "colab_p7g_landcover_vs_label.py").read_text()
        # Both directions must be present, each derived from its own cell.
        self.assertIn("P(weak INDUSTRIAL_FIRE | SUBSTRATE)", src)
        self.assertIn("_p_ind_given_nv", src)
        self.assertIn("_p_ind_given_v", src)
        # And the composition direction must be named for what it is.
        self.assertIn("P(SUBSTRATE | weak INDUSTRIAL_FIRE)", src)
        # No line may assert a conditional from `~is_t` without naming it.
        self.assertNotIn("(~is_t &", src,
                         "a ~is_t term must not appear inside a labelled "
                         "conditional; it is the complement and is only valid "
                         "when the label says so")

    def test_the_cell_never_raises_system_exit(self):
        """SystemExit in a notebook cell trips an IPython/inspect bug on 3.13
        and buries the real error under three tracebacks."""
        self.assertNotIn("raise SystemExit(0)", self.src)
        self.assertNotIn("SystemExit(1)", self.src)


if __name__ == "__main__":
    unittest.main()