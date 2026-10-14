"""
End-to-end rehearsal of `colab_p7k_nbr_change.py` against stubs.

This cell is an 11-hour extraction that Colab WILL interrupt, so its resume and
shard-boundary logic is the part that matters and the part a syntax check cannot
reach. Four bugs were found this way, three of them by reading rather than
running:

  1. `gee.monthly_nbr_composite` was CALLED but never existed. The only
     reference in the repo was the call itself, so the cell raised
     AttributeError on the first month. A test that only checks the cell is
     syntactically valid would never have found it -- the name is only
     resolved at run time.

  2. `gee.imagery_gap` returns a DICT keyed by sensor
     (`{"no_landsat": [...], "no_sentinel2": [...]}`). The cell iterated it as a
     flat list of years, yielding the keys and raising
     `ValueError: invalid literal for int(): 'no_landsat'`.

  3. Shards OVERLAPPED. `filled_rows` was `np.arange(0, lo)` plus the current
     chunk, i.e. every row filled so far, so shard k held rows 0..k. Assembly
     sorted without `drop_duplicates`, and `nbr_events.parquet` carried every
     early row once per shard. This one is silent: row counts stay plausible.

  4. Rows were LOST. `continue` on an uncovered chunk skipped the shard-boundary
     write below it, so a shard whose last chunk had no imagery was never
     written at all. Bug 3 and bug 4 pull in opposite directions, which is why
     a run could look complete while being both short and duplicated.

No `ee`, no Drive, no pyarrow. parquet is shimmed to CSV because this machine
has no pyarrow -- the shim honours the EXACT path given, which matters because
the cell relies on writing `*.partial` then renaming it.
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
CELL = REPO / "colab_p7k_nbr_change.py"

N_EVENTS = 6_000
CHUNK = 1_000
SHARD_CHUNKS = 2


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

    names = ["event_id", "start_year", "start_month",
             "event_centroid_lon", "event_centroid_lat"]

    def _meta(path):
        return types.SimpleNamespace(num_rows=len(_read_parquet(path)))

    pd.DataFrame.to_parquet = _to_parquet
    pd.read_parquet = _read_parquet
    pqm = types.ModuleType("pyarrow.parquet")
    pqm.ParquetFile = lambda p: types.SimpleNamespace(
        schema_arrow=types.SimpleNamespace(names=list(names)),
        metadata=_meta(p))
    pqm.read_table = lambda p, columns=None: _read_parquet(p)
    pqa = types.ModuleType("pyarrow")
    pqa.parquet = pqm
    return {"pyarrow": pqa, "pyarrow.parquet": pqm}


#: NBR value the stub returns for a given `(year, month)`, in [-1, 1].
#:
#: Deterministic and DEPENDENT ON BOTH year and month. Dependence on the year is
#: what catches a cell that samples the right month of the wrong year -- an
#: error that would otherwise look fine whenever NBR_PRE and NBR_POST came from
#: the same year.
def _stub_nbr(year: int, month: int) -> float:
    return float((year * 12 + month) % 977) / 977.0 * 2.0 - 1.0


def _month_before(year: int, month: int, lag: int = 1):
    """Mirror of `gee_utils.month_before`, kept local so the stub's arithmetic
    is not what the assertion is testing."""
    total = (year * 12 + (month - 1)) - lag
    return total // 12, total % 12 + 1


def _install_ee_stub():
    """A stub `ee` where each composite returns a value encoding its own month.

    Because the value is a function of `(year, month)`, a test can assert that
    NBR_PRE came from the month PRE_LAG before the event and NBR_POST from the
    event's own month -- rather than merely asserting that DNBR is non-zero. A
    non-zero DNBR is also satisfied by sampling two wrong months, which is
    exactly the defect that made the old draft's dNBR meaningless.
    """
    import gee_utils

    class _Img:
        def __init__(self, year, month):
            self.year = year
            self.month = month

        @staticmethod
        def constant(v):
            return types.SimpleNamespace(bandNames=lambda:
                                         types.SimpleNamespace(
                                             getInfo=lambda: ["constant"]))

        def select(self, band):
            return self

        def sampleRegions(self, collection, properties, scale, geometries):
            val = _stub_nbr(self.year, self.month)
            # Returned OUT of request order on purpose. Matching results back by
            # position is only correct while the server preserves order, which is
            # not documented; matching by the `row` property is.
            feats = [{"properties": {"row": f.props["row"], "NBR": val}}
                     for f in reversed(collection.feats)]
            return types.SimpleNamespace(
                getInfo=lambda: {"features": feats})

    class _Feat:
        def __init__(self, geometry=None, props=None):
            self.props = props

    class _FC:
        def __init__(self, feats):
            self.feats = list(feats)

    m = types.ModuleType("ee")
    m.Image = _Img
    m.ImageCollection = lambda i: types.SimpleNamespace(first=lambda: _Img(0))
    m.Feature = _Feat
    m.FeatureCollection = _FC
    m.Geometry = types.SimpleNamespace(Point=lambda c: c)
    m.Filter = types.SimpleNamespace(lt=lambda a, b: None)
    m.Initialize = lambda **k: None
    m.Authenticate = lambda **k: None
    m.data = types.SimpleNamespace(getAssetRoots=lambda: [])

    # The cell calls `gee.nbr_monthly_composite`, so the stub replaces the
    # helper rather than the collection plumbing underneath it. Keeping the real
    # month arithmetic in the cell is the point: `month_before` is genuine
    # gee_utils code and is what these tests exercise.
    del gee_utils.nbr_monthly_composite
    gee_utils.nbr_monthly_composite = lambda y, mo: _Img(int(y), int(mo))
    return m


class P7kRehearsal(unittest.TestCase):
    """Runs the real cell against a stubbed Drive and a stubbed `ee`."""

    @classmethod
    def setUpClass(cls):
        # The shim REPLACES `pd.DataFrame.to_parquet` / `pd.read_parquet`, so
        # the originals are saved and restored. `test_colab_p7h.py` leaks this
        # monkeypatch into every suite imported after it, which is why
        # `test_colab_utils` and `test_explainability` error under a full
        # `unittest discover`; this file does not repeat that.
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
        # The cell's `import pyarrow.parquet as pq` registers a SEPARATE
        # sys.modules entry that `patch.dict` does not undo, and a stub module
        # has no `__spec__`. Left behind, `find_spec("pyarrow")` raises
        # `ValueError: pyarrow.__spec__ is None` in every later suite.
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
        # All 2015+ so every month is covered: coverage-gap handling is tested
        # separately in gee-utils tests, where it belongs.
        rng = np.random.default_rng(7)
        pd.DataFrame({
            "event_centroid_lon": np.linspace(68, 96, n),
            "event_centroid_lat": np.linspace(8, 35, n),
            "event_id": [f"E{i}" for i in range(n)],
            "start_year": rng.integers(2015, 2024, n),
            "start_month": rng.integers(1, 13, n),
        }).to_parquet(ds / "meta.parquet")
        self.shards = self.root / "qc" / "nbr_shards"
        self.addCleanup(self._tmp.cleanup)

    def _run_cell(self, **over):
        cu = types.ModuleType("colab_utils")
        root = self.root
        cu.resolve_project_dir = lambda: root
        cu.drive_report = lambda: "  (rehearsal: stubbed Drive)"
        cu.MYDRIVE = "/stub"
        # The real `event_year_month` reads start_time; the rehearsal's meta
        # table has no such column, so the arithmetic is stubbed to match what
        # the CSV shim wrote. The month arithmetic itself is exercised for real
        # in test_colab_p7k's MonthArithmeticTests.
        def _ym(path, verbose=False):
            df = pd.read_parquet(path)
            return (df["start_year"].to_numpy().astype("int16"),
                    df["start_month"].to_numpy().astype("int8"))
        cu.event_year_month = _ym
        ns = {
            "GEE_CHUNK_SIZE": CHUNK,
            "NBR_SHARD_CHUNKS": SHARD_CHUNKS,
            "NBR_RESUME": True,
            "__name__": "__main__",
        }
        ns.update(over)
        with mock.patch.dict(sys.modules, {"colab_utils": cu}):
            exec(compile(self.src, str(CELL), "exec"), ns)      # noqa: S102
        return ns

    def _assembled(self):
        return pd.read_csv(self.root / "qc" / "nbr_events.parquet")

    # -- regression 1: the missing helper ------------------------------

    def test_the_monthly_composite_helper_exists(self):
        """The cell called a function that was never written.

        Regression 1. `monthly_nbr_composite` had no definition anywhere in the
        repo, so an 11-hour run died with AttributeError on its first call.
        """
        import gee_utils

        self.assertTrue(hasattr(gee_utils, "nbr_monthly_composite"),
                        "gee.nbr_monthly_composite is missing again")
        src = self.src
        called = [n for n in ("nbr_monthly_composite", "monthly_nbr_composite")
                  if f"gee.{n}(" in src]
        for n in called:
            self.assertTrue(hasattr(gee_utils, n),
                            f"the cell calls gee.{n} but it is not defined")

    # -- regression 2: the dict/list confusion -------------------------

    def test_imagery_gap_is_a_dict_not_a_list(self):
        """Regression 2. Iterating the dict yielded its keys.

        `int("no_landsat")` raised ValueError before any imagery call.
        """
        import gee_utils

        gap = gee_utils.imagery_gap([2012, 2015, 2020])
        self.assertIsInstance(gap, dict,
                              "imagery_gap must stay a dict keyed by sensor")
        for k in gap:
            self.assertIsInstance(k, str)
        self.assertEqual(gap["no_landsat"], [2012])
        # 2014-2016 have no Sentinel-2 L2A but DO have Landsat 8. Only the
        # Landsat gap may gate NBR; unioning the two would drop three usable
        # years from a 14-year catalogue.
        self.assertIn(2015, gap["no_sentinel2"])
        self.assertNotIn(2015, gap["no_landsat"])

    def test_a_2012_event_does_not_crash_the_cell(self):
        """Regression 2, end to end: a pre-2013 year must be skipped, not fatal."""
        pd.DataFrame({
            "event_centroid_lon": np.linspace(68, 96, 200),
            "event_centroid_lat": np.linspace(8, 35, 200),
            "event_id": [f"E{i}" for i in range(200)],
            "start_year": np.full(200, 2012, dtype=np.int16),
            "start_month": np.full(200, 5, dtype=np.int8),
        }).to_parquet(
            self.root / "full_2012_2026" / "features" / "dataset" / "meta.parquet")
        self._run_cell()
        # No crash, and 2012 has no Landsat 8, so nothing is claimed. Crucially
        # NO parquet is written either: an empty table would join downstream as
        # "these events have no NBR", which is a claim rather than an absence.
        self.assertFalse((self.root / "qc" / "nbr_events.parquet").exists(),
                         "an empty nbr_events.parquet was written; a zero-row "
                         "table downstream reads as 'measured, no NBR'")

    # -- regression 3: overlapping shards ------------------------------

    def test_a_first_run_covers_every_event_once(self):
        """Regression 3. Shards must not overlap.

        The old cell accumulated `np.arange(0, lo)` into each shard, so row 0
        appeared in every shard and the assembly had no dedup.
        """
        self._run_cell()
        df = self._assembled()
        self.assertEqual(len(df), N_EVENTS)
        self.assertEqual(df["row"].nunique(), N_EVENTS,
                         "rows are duplicated across shards")

    def test_each_shard_holds_only_its_own_rows(self):
        """A shard's row range is exactly its chunk range -- no more."""
        self._run_cell()
        total_chunks = -(-N_EVENTS // CHUNK)
        for shard_no in range(-(-total_chunks // SHARD_CHUNKS)):
            p = self.shards / f"nbr_{shard_no:06d}.parquet"
            df = pd.read_csv(p)
            lo = shard_no * SHARD_CHUNKS * CHUNK
            hi = min(lo + SHARD_CHUNKS * CHUNK, N_EVENTS)
            self.assertGreaterEqual(int(df["row"].min()), lo)
            self.assertLess(int(df["row"].max()), hi,
                            f"shard {shard_no} holds rows outside its range")

    def test_dnbr_is_not_identically_zero(self):
        """The original defect: NBR_PRE and NBR_POST from the same composite.

        Sampling each event's own month twice makes DNBR exactly 0 everywhere,
        which is a plausible-looking feature column that measures nothing.
        """
        self._run_cell()
        df = self._assembled()
        d = df["DNBR"]
        self.assertGreater(int((d.abs() > 1e-6).sum()), 0,
                           "DNBR is 0 everywhere -- PRE and POST are the "
                           "same composite")

    def test_pre_comes_from_the_month_before_and_post_from_the_event_month(self):
        """Stronger than "DNBR is non-zero": it pins WHICH months were sampled.

        Two wrong months also give a non-zero DNBR, so a non-zero check alone
        would pass a cell that samples the wrong composites. The stub's value is
        a pure function of (year, month), so the correct pair is recoverable.
        """
        self._run_cell()
        df = self._assembled()
        bad_pre = bad_post = 0
        for _, r in df.iterrows():
            y, m = int(r["start_year"]), int(r["start_month"])
            if abs(r["NBR_POST"] - _stub_nbr(y, m)) > 1e-5:
                bad_post += 1
            py, pm = _month_before(y, m, 1)
            if abs(r["NBR_PRE"] - _stub_nbr(py, pm)) > 1e-5:
                bad_pre += 1
        self.assertEqual(bad_post, 0, "NBR_POST is not the event's own month")
        self.assertEqual(bad_pre, 0, "NBR_PRE is not PRE_LAG months earlier")

    # -- regression 4: rows lost at shard boundaries -------------------

    def test_no_row_is_lost_when_a_chunk_has_no_imagery(self):
        """Regression 4. `continue` skipped the shard-boundary write.

        An uncovered chunk must not prevent its shard from being written.
        """
        # Shard 1 is chunks 2-3 -> rows 2000..3999. Force its LAST chunk's
        # month uncovered while the earlier chunk stays covered: under the old
        # control flow the whole shard was never written.
        meta = pd.read_parquet(
            self.root / "full_2012_2026" / "features" / "dataset" / "meta.parquet")
        meta.loc[meta.index[3000:], "start_year"] = 2012     # no Landsat 8
        meta.to_parquet(
            self.root / "full_2012_2026" / "features" / "dataset" / "meta.parquet",
            index=False)
        self._run_cell()
        df = self._assembled()
        # Rows 2000..2999 are covered and MUST survive.
        self.assertIn(2000, set(df["row"]))
        self.assertEqual(len(df[df["row"].between(2000, 2999)]), 1000,
                         "rows were dropped when a later chunk was uncovered")

    def test_resume_after_losing_shards_restores_completeness(self):
        """THE regression. Shard identity must come from the chunk number."""
        self._run_cell()
        for s in (1, 2):
            (self.shards / f"nbr_{s:06d}.parquet").unlink()
        self._run_cell()
        self.assertEqual(len(self._assembled()), N_EVENTS,
                         "resume did not refill the missing shards")
        total_chunks = -(-N_EVENTS // CHUNK)
        want = [f"nbr_{s:06d}.parquet"
                for s in range(-(-total_chunks // SHARD_CHUNKS))]
        self.assertEqual(sorted(p.name for p in self.shards.glob("*.parquet")),
                         want, "resume produced a different shard set")

    def test_resume_does_not_overwrite_surviving_shards(self):
        self._run_cell()
        keep = self.shards / "nbr_000000.parquet"
        stamp = keep.stat().st_mtime_ns
        size_before = keep.stat().st_size
        for s in (1, 2):
            (self.shards / f"nbr_{s:06d}.parquet").unlink()
        self._run_cell()
        self.assertEqual(keep.stat().st_mtime_ns, stamp,
                         "a surviving shard was rewritten")
        self.assertEqual(keep.stat().st_size, size_before)

    def test_a_second_run_does_no_work(self):
        self._run_cell()
        before = sorted(p.name for p in self.shards.glob("*.parquet"))
        mtimes = {p.name: p.stat().st_mtime_ns
                  for p in self.shards.glob("*.parquet")}
        self._run_cell()
        self.assertEqual(sorted(p.name for p in self.shards.glob("*.parquet")),
                         before)
        self.assertEqual({p.name: p.stat().st_mtime_ns
                          for p in self.shards.glob("*.parquet")}, mtimes,
                         "a complete run rewrote shards it should have skipped")

    def test_a_shortened_shard_is_trusted_and_not_rewritten(self):
        """.partial-then-rename means a kill NEVER truncates a shard, so a
        within-band short file can only be a legitimately filled-only shard
        (run 2's shard 0: 220,388 rows of a 250,000-row band, ~12% of rows
        having no imagery in either month). It MUST be trusted: an
        exact-count rule would discard it and redo 66 minutes of work.
        Truncating it externally (as below) is not a state production can
        reach -- the band/duplicate rule is the corruption check that
        matters (the P7h bug put rows 0..250k in shards 0 AND 1)."""
        self._run_cell()
        target = self.shards / "nbr_000000.parquet"
        short = pd.read_csv(target).head(10)
        short.to_csv(target, index=False)
        stat_before = target.stat().st_mtime_ns
        self._run_cell()
        self.assertEqual(target.stat().st_mtime_ns, stat_before,
                         "the shortened shard was rewritten (resume should "
                         "trust it)")
        self.assertEqual(len(pd.read_csv(target)), 10,
                         "the shortened shard's rows were lost")

    def test_an_out_of_band_shard_is_discarded_and_reextracted(self):
        """The corruption the band rule exists for: a shard whose file leaks
        rows from a NEIGHBOUR's band (P7h's early resume wrote rows 0..250k
        into shards 0 and 1). Such a shard is discarded and rebuilt."""
        self._run_cell()
        target = self.shards / "nbr_000000.parquet"
        bad = pd.concat([pd.read_csv(target),
                         pd.read_csv(self.shards / "nbr_000001.parquet")])
        bad.to_csv(target, index=False)
        self._run_cell()
        self.assertEqual(len(self._assembled()), N_EVENTS,
                         "an out-of-band shard was trusted")
        self.assertEqual(len(pd.read_csv(target)), SHARD_CHUNKS * CHUNK,
                         "the out-of-band shard was not rebuilt")

    # -- NaN discipline -------------------------------------------------

    def test_unmeasured_is_nan_never_zero(self):
        """A missing measurement must not read as a measured absence.

        0 for NBR means "no vegetation at all"; NaN means "could not measure".
        Collapsing them would put 0 into the model as a confident reading.
        """
        self._run_cell()
        df = self._assembled()
        self.assertEqual(len(df[df["NBR_PRE"].isna()]), 0)
        self.assertTrue((df["NBR_PRE"].abs() <= 1.0001).all())
        self.assertTrue((df["NBR_POST"].abs() <= 1.0001).all())

    def test_dnbr_is_pre_minus_post(self):
        self._run_cell()
        df = self._assembled()
        expect = df["NBR_PRE"] - df["NBR_POST"]
        self.assertTrue(np.allclose(df["DNBR"].to_numpy(), expect.to_numpy(),
                                    atol=1e-6))

    # -- structure the cell must keep ----------------------------------

    def test_shards_are_written_atomically(self):
        """`.partial` then rename, so a kill leaves no torn parquet."""
        self.assertIn(".partial", self.src)
        self.assertRegex(self.src, r"partial\.replace\(")

    def test_the_cell_never_raises_system_exit_zero(self):
        """SystemExit(0) trips an IPython/inspect bug on Python 3.13."""
        self.assertNotIn("raise SystemExit(0)", self.src)
        self.assertNotIn("SystemExit(1)", self.src)

    def test_an_incomplete_table_says_so_loudly(self):
        """A short table silently shrinks every downstream rate."""
        self._run_cell()
        for s in (1, 2):
            (self.shards / f"nbr_{s:06d}.parquet").unlink()
        self._run_cell()
        self.assertIn("INCOMPLETE", self.src)
        self.assertIn("missing shards", self.src)


class NBRSamplingTests(unittest.TestCase):
    """The NBR triple itself. A swapped band order measures nothing."""

    def test_nir_and_swir_are_not_swapped(self):
        import gee_utils

        swir, nir = gee_utils.NBR_BANDS["landsat"]
        self.assertEqual(swir, "SR_B7")
        self.assertEqual(nir, "SR_B5")


class MonthArithmeticTests(unittest.TestCase):
    """`month_before` crosses a year boundary; naive `month - lag` does not."""

    def test_january_steps_back_into_the_previous_year(self):
        import gee_utils

        self.assertEqual(gee_utils.month_before(2020, 1, 1), (2019, 12))

    def test_lag_spanning_many_months(self):
        import gee_utils

        self.assertEqual(gee_utils.month_before(2015, 3, 14), (2014, 1))

    def test_month_bounds_are_end_exclusive(self):
        """An inclusive end date pulls in the next month's first scenes."""
        import gee_utils

        start, end = gee_utils._month_bounds(2020, 12)
        self.assertEqual(start, "2020-12-01")
        self.assertEqual(end, "2021-01-01")
        start, end = gee_utils._month_bounds(2020, 1)
        self.assertEqual(end, "2020-02-01")


class _NDImage:
    """Just enough of `ee.Image` to pin the call shape."""

    def __init__(self, bands, calls=None):
        self.bands = list(bands)
        # Shared across the whole chain: `select` returns a NEW image, so a
        # per-instance log would lose every call made after it.
        self.calls = calls if calls is not None else []

    def select(self, *names):
        flat = [n for a in names for n in (a if isinstance(a, (list, tuple))
                                           else [a])]
        self.calls.append(f"select{flat}")
        return _NDImage(flat, self.calls)

    def median(self):
        self.calls.append("median")
        return self

    def normalizedDifference(self, band2=None, *extra):
        if extra:
            raise TypeError(
                "Image.normalizedDifference() takes from 1 to 2 positional "
                "arguments but "
                f"{3 + len(extra)} were given")
        first = self.bands[0]
        second = self.bands[1] if band2 is None else band2
        self.calls.append(f"nd({first},{second})")
        return self

    def rename(self, name):
        self.calls.append(f"rename({name})")
        return self


class NormalizedDifferenceTests(unittest.TestCase):
    """`Image.normalizedDifference()` accepts AT MOST ONE band argument.

    The JavaScript overload takes two; the Python one does not. Calling
    `.normalizedDifference(nir, swir)` raised

        TypeError: Image.normalizedDifference() takes from 1 to 2 positional
        arguments but 3 were given

    on the first composite of a real Colab run. The stub below mirrors the REAL
    signature, so the error reproduces locally instead of costing a session.
    """

    def _composite(self):
        """Drive the real helper against a stub whose signature matches `ee`."""
        import gee_utils

        img = _NDImage(["SR_B5", "SR_B7"])
        # The composite no longer carries a scene-level CLOUD_COVER filter
        # (an all-filtered monsoon month is an empty collection and a
        # zero-band image), so filterDate goes straight to .map. The stub
        # offers BOTH paths from filterDate so either composite shape drives.
        after_date = types.SimpleNamespace(
            filter=lambda _f: img,
            map=lambda _m: img)
        stub = types.SimpleNamespace(
            Filter=types.SimpleNamespace(lt=lambda a, b: ("lt", a, b)),
            ImageCollection=lambda _a: types.SimpleNamespace(
                filterDate=lambda a, b: after_date))
        with mock.patch.dict(sys.modules, {"ee": stub}):
            gee_utils.nbr_monthly_composite(
                2020, 5, collection_id="LANDSAT/LC08/C02/T1_L2")
        return img

    def test_the_band_order_carried_by_select_is_nir_then_swir(self):
        """(NIR - SWIR)/(NIR + SWIR). Swapped yields -NBR, still in [-1, 1]."""
        img = self._composite()
        self.assertIn("select['SR_B5', 'SR_B7']", img.calls,
                      "select must put NIR first, then SWIR")
        self.assertIn("nd(SR_B5,SR_B7)", img.calls,
                      "zero-arg normalizedDifference reads the first two "
                      "bands, so the order in select is the whole argument")
        self.assertIn("rename(NBR)", img.calls)

    def test_two_positional_arguments_are_a_type_error(self):
        """The exact shape that broke the real run."""
        img = _NDImage(["SR_B5", "SR_B7"])
        with self.assertRaises(TypeError):
            img.normalizedDifference("SR_B5", "SR_B7")

    def test_an_impossible_month_is_rejected(self):
        import gee_utils

        with self.assertRaises(ValueError):
            gee_utils._month_bounds(2020, 13)
        with self.assertRaises(ValueError):
            gee_utils._month_bounds(2020, 0)


class SARGapTests(unittest.TestCase):
    """S1 starts 2014, LATER than Landsat 8's 2013, so the gap differs."""

    def test_s1_cannot_see_2013_but_landsat_can(self):
        import gee_utils

        s1 = gee_utils.sar_imagery_gap([2013, 2014])
        optical = gee_utils.imagery_gap([2013, 2014])
        self.assertIn(2013, s1["no_sentinel1"])
        self.assertNotIn(2013, optical["no_landsat"],
                         "Landsat 8 does cover 2013")
        self.assertEqual(s1["no_sentinel1"], [2013])

    def test_s1_polarisations_are_the_two_that_exist(self):
        import gee_utils

        self.assertEqual(gee_utils.S1_BANDS, ("VV", "VH"))
        self.assertEqual(gee_utils.S1_GRD, "COPERNICUS/S1_GRD")


if __name__ == "__main__":
    unittest.main()
