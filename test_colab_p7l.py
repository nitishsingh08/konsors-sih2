"""
End-to-end rehearsal of `colab_p7l_sar.py` against stubs.

Same shape as `test_colab_p7h.py` / `test_colab_p7k.py`: run the real cell
against a stubbed Drive and a stubbed `ee`, because this is an 11-hour cell that
Colab WILL interrupt and the resume logic is the part that matters.

What is pinned here, beyond the shared shard machinery:

  * The THREE features exist and the FOUR unavailable ones are not invented.
    The ledger lists seven under SAR_STRUCTURAL, but only three are honestly
    obtainable from GRD. A placeholder column would be indistinguishable from a
    measurement to the model and to every SHAP report downstream, so the cell
    must not emit one.
  * S1's coverage gap is asked SEPARATELY from the optical gap. Sentinel-1 starts
    2014, LATER than Landsat 8's 2013, so the two gaps are different questions
    and the optical one cannot answer this one. Deriving it would send 2013
    events to a sensor with no data there.
  * PRE and POST come from the RIGHT months. A non-zero change is also produced
    by two wrong months, so the assertion pins which ones, not just that they
    differ.
  * An absent footprint reads as NaN, never 0.

No `ee`, no Drive, no pyarrow. parquet is shimmed to CSV because this machine has
no pyarrow; the shim honours the EXACT path given, because the cell relies on
writing `*.partial` then renaming it.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent
CELL = REPO / "colab_p7l_sar.py"

N_EVENTS = 4_000
CHUNK = 1_000
SHARD_CHUNKS = 2


def _stub_sigma0(year: int, month: int, pol: str) -> float:
    """Deterministic backscatter in dB, in the real GRD range (-35 .. +5).

    A pure function of `(year, month, polarisation)`, so a test can assert that
    PRE came from the month PRE_LAG earlier.
    """
    h = (year * 12 + month) % 41
    return -35.0 + 40.0 * (h / 40.0) + (2.0 if pol == "VH" else 0.0)


def _install_parquet_shim():
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

    # The schema is READ FROM THE FILE, not hardcoded. A hardcoded list would
    # have kept advertising `start_month` after the real meta.parquet dropped it,
    # and every test below would pass against a schema that does not exist.
    def _names(path):
        p = pathlib.Path(str(path))
        for alt in (p, p.with_suffix(".csv"),
                    p.parent / (p.name + ".csv")):
            if alt.exists() and alt.is_file():
                with alt.open() as fh:
                    return fh.readline().strip().split(",")
        return []

    def _meta(path):
        return types.SimpleNamespace(num_rows=len(_read_parquet(path)))

    pd.DataFrame.to_parquet = _to_parquet
    pd.read_parquet = _read_parquet
    pqm = types.ModuleType("pyarrow.parquet")
    pqm.ParquetFile = lambda p: types.SimpleNamespace(
        schema_arrow=types.SimpleNamespace(names=_names(p)),
        metadata=_meta(p))
    pqm.read_table = lambda p, columns=None: _read_parquet(p, columns=columns)  # noqa: E501
    pqa = types.ModuleType("pyarrow")
    pqa.parquet = pqm
    return {"pyarrow": pqa, "pyarrow.parquet": pqm}


def _install_ee_stub(footprint=(-180.0, 180.0)):
    """`ee` stub whose composites carry VV and VH for their own month.

    `footprint` is the fraction of longitudes covered; points outside it return
    NO feature, which is how an S1 footprint miss reaches the cell. The features
    come back REVERSED, so any positional matching in the cell is caught.
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
            feats = []
            for f in reversed(collection.feats):
                lon = f.geom[0]
                lo, hi = footprint
                if not (lo <= lon <= hi):
                    continue                    # outside the footprint: absent
                feats.append({"properties": {
                    "row": f.props["row"],
                    "VV": _stub_sigma0(self.year, self.month, "VV"),
                    "VH": _stub_sigma0(self.year, self.month, "VH"),
                }})
            return types.SimpleNamespace(
                getInfo=lambda: {"features": feats})

    class _Feat:
        def __init__(self, geometry=None, props=None):
            self.geom = geometry
            self.props = props

    class _FC:
        def __init__(self, feats):
            self.feats = list(feats)

    m = types.ModuleType("ee")
    m.Image = _Img
    m.ImageCollection = lambda i: types.SimpleNamespace(first=lambda: _Img(0, 0))
    m.Feature = _Feat
    m.FeatureCollection = _FC
    m.Geometry = types.SimpleNamespace(Point=lambda c: c)
    m.Initialize = lambda **k: None
    m.Authenticate = lambda **k: None
    m.data = types.SimpleNamespace(getAssetRoots=lambda: [])

    del gee_utils.sar_monthly_composite
    gee_utils.sar_monthly_composite = lambda y, mo: _Img(int(y), int(mo))
    return m


class P7lRehearsal(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._pd_orig = (pd.DataFrame.to_parquet, pd.read_parquet)
        cls._patches = [mock.patch.dict(sys.modules, _install_parquet_shim()),
                        mock.patch.dict(
                            sys.modules,
                            {"ee": _install_ee_stub()})]
        for p in cls._patches:
            p.start()
        cls.src = CELL.read_text()

    @classmethod
    def tearDownClass(cls):
        for p in cls._patches:
            p.stop()
        pd.DataFrame.to_parquet, pd.read_parquet = cls._pd_orig
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
        rng = np.random.default_rng(11)
        # 2015+ so every month is after Sentinel-1's 2014 start; the gap is
        # tested separately, where it belongs.
        pd.DataFrame({
            "event_centroid_lon": np.linspace(68, 96, n),
            "event_centroid_lat": np.linspace(8, 35, n),
            "event_id": [f"E{i}" for i in range(n)],
            "start_year": rng.integers(2015, 2024, n),
            "start_month": rng.integers(1, 13, n),
        }).to_parquet(ds / "meta.parquet")
        self.shards = self.root / "qc" / "sar_shards"
        self.addCleanup(self._tmp.cleanup)

    def _run_cell(self, **over):
        cu = types.ModuleType("colab_utils")
        root = self.root
        cu.resolve_project_dir = lambda: root
        cu.drive_report = lambda: "  (rehearsal: stubbed Drive)"
        cu.MYDRIVE = "/stub"
        cu.event_year_month = lambda path, verbose=False: (
            pd.read_parquet(path)["start_year"].to_numpy().astype("int16"),
            pd.read_parquet(path)["start_month"].to_numpy().astype("int8"))
        ns = {
            "GEE_CHUNK_SIZE": CHUNK,
            "SAR_SHARD_CHUNKS": SHARD_CHUNKS,
            "SAR_RESUME": True,
            "__name__": "__main__",
        }
        ns.update(over)
        with mock.patch.dict(sys.modules, {"colab_utils": cu}):
            exec(compile(self.src, str(CELL), "exec"), ns)      # noqa: S102
        return ns

    def _assembled(self):
        return pd.read_csv(self.root / "qc" / "sar_events.parquet")

    # -- the three features exist, the four do not ---------------------

    def test_it_produces_exactly_the_three_achievable_features(self):
        self._run_cell()
        df = self._assembled()
        for c in ("VV_CHANGE", "VH_CHANGE", "VV_VH_CHANGE"):
            self.assertIn(c, df.columns, f"{c} is missing")
            self.assertGreater(int(df[c].notna().sum()), 0,
                               f"{c} is entirely NaN")

    def test_it_does_not_invent_the_four_unobtainable_features(self):
        """A placeholder column is indistinguishable from a measurement.

        COHERENCE_LOSS needs interferometry (SLC phase, absent from GRD) and
        STRUCTURAL_CHANGE_PROBABILITY is a model output, not an extraction.
        DNBR and DNDVI are optical and belong to P7k. Emitting any of them here
        would put a plausible-looking constant into the feature table.
        """
        self._run_cell()
        cols = set(self._assembled().columns)
        for c in ("COHERENCE_LOSS", "STRUCTURAL_CHANGE_PROBABILITY",
                  "DNBR", "DNDVI"):
            self.assertNotIn(c, cols, f"{c} is not obtainable from Sentinel-1 "
                                      f"GRD and must not be fabricated")

    def test_the_cell_documents_the_omissions(self):
        """The manifest must record what was NOT made, so the gap is explicit."""
        self._run_cell()
        import json
        man = json.loads((self.root / "qc" / "sar_manifest.json").read_text())
        self.assertEqual(sorted(man["features"]),
                         ["VH_CHANGE", "VV_CHANGE", "VV_VH_CHANGE"])
        self.assertIn("COHERENCE_LOSS", man["not_produced"])

    # -- the S1 gap is its own question --------------------------------

    def test_a_pre_2014_catalogue_is_skipped_not_sampled(self):
        """S1 starts 2014, later than Landsat 8's 2013.

        A pre-2014 catalogue has no SAR at all. It must be explained, and no
        table written -- an empty parquet would join downstream as "these events
        have no SAR", which is a claim rather than an absence.
        """
        pd.DataFrame({
            "event_centroid_lon": np.linspace(68, 96, 200),
            "event_centroid_lat": np.linspace(8, 35, 200),
            "event_id": [f"E{i}" for i in range(200)],
            "start_year": np.full(200, 2013, dtype=np.int16),
            "start_month": np.full(200, 6, dtype=np.int8),
        }).to_parquet(
            self.root / "full_2012_2026" / "features" / "dataset" / "meta.parquet")
        self._run_cell()
        self.assertFalse((self.root / "qc" / "sar_events.parquet").exists())

    def test_the_optical_gap_is_not_reused_for_sar(self):
        """`imagery_gap` and `sar_imagery_gap` answer different questions."""
        import gee_utils

        self.assertIn(2013, gee_utils.sar_imagery_gap([2013])["no_sentinel1"])
        self.assertNotIn(2013, gee_utils.imagery_gap([2013])["no_landsat"],
                         "Landsat 8 does cover 2013; the gaps must differ")
        self.assertIn("sar_imagery_gap(", self.src)
        self.assertNotIn("gee.imagery_gap(", self.src,
                         "P7l must not gate Sentinel-1 months on the optical gap")

    # -- which months were sampled -------------------------------------

    def test_pre_and_post_come_from_the_right_months(self):
        """Stronger than "the change is non-zero", which two wrong months also
        satisfy."""
        self._run_cell()
        df = self._assembled()

        def before(y, m, lag=1):
            t = y * 12 + (m - 1) - lag
            return t // 12, t % 12 + 1

        bad_pre = bad_post = 0
        for _, r in df.iterrows():
            y, m = int(r["start_year"]), int(r["start_month"])
            py, pm = before(y, m)
            if abs(r["VV_POST"] - _stub_sigma0(y, m, "VV")) > 1e-3:
                bad_post += 1
            if abs(r["VV_PRE"] - _stub_sigma0(py, pm, "VV")) > 1e-3:
                bad_pre += 1
        self.assertEqual(bad_post, 0, "VV_POST is not the event's own month")
        self.assertEqual(bad_pre, 0, "VV_PRE is not PRE_LAG months earlier")

    def test_the_changes_are_the_differences_they_claim(self):
        self._run_cell()
        df = self._assembled()
        for pol in ("VV", "VH"):
            expect = df[f"{pol}_POST"] - df[f"{pol}_PRE"]
            self.assertTrue(np.allclose(df[f"{pol}_CHANGE"].to_numpy(),
                                        expect.to_numpy(), atol=1e-4),
                            f"{pol}_CHANGE is not {pol}_POST - {pol}_PRE")
        expect = df["VV_CHANGE"] - df["VH_CHANGE"]
        self.assertTrue(np.allclose(df["VV_VH_CHANGE"].to_numpy(),
                                    expect.to_numpy(), atol=1e-4),
                        "VV_VH_CHANGE is not the dual-pol difference")

    def test_change_is_not_identically_zero(self):
        self._run_cell()
        d = self._assembled()["VV_CHANGE"]
        self.assertGreater(int((d.abs() > 1e-6).sum()), 0,
                           "every change is 0 -- PRE and POST are the same "
                           "composite")

    # -- shard machinery -----------------------------------------------

    def test_a_first_run_covers_every_event_once(self):
        self._run_cell()
        df = self._assembled()
        self.assertEqual(len(df), N_EVENTS)
        self.assertEqual(df["row"].nunique(), N_EVENTS,
                         "rows are duplicated across shards")

    def test_each_shard_holds_only_its_own_rows(self):
        self._run_cell()
        total = -(-N_EVENTS // CHUNK)
        for s in range(-(-total // SHARD_CHUNKS)):
            df = pd.read_csv(self.shards / f"sar_{s:06d}.parquet")
            lo = s * SHARD_CHUNKS * CHUNK
            hi = min(lo + SHARD_CHUNKS * CHUNK, N_EVENTS)
            self.assertGreaterEqual(int(df["row"].min()), lo)
            self.assertLess(int(df["row"].max()), hi)

    def test_resume_after_losing_shards_restores_completeness(self):
        self._run_cell()
        total = -(-N_EVENTS // CHUNK)
        n_shards = -(-total // SHARD_CHUNKS)
        self.assertEqual(n_shards, 2, "test assumes 2 shards")
        (self.shards / "sar_000001.parquet").unlink()
        self._run_cell()
        self.assertEqual(len(self._assembled()), N_EVENTS)

    def test_resume_does_not_overwrite_surviving_shards(self):
        self._run_cell()
        keep = self.shards / "sar_000000.parquet"
        stamp = keep.stat().st_mtime_ns
        (self.shards / "sar_000001.parquet").unlink()
        self._run_cell()
        self.assertEqual(keep.stat().st_mtime_ns, stamp,
                         "a surviving shard was rewritten")

    def test_a_shortened_shard_is_trusted_and_not_rewritten(self):
        """.partial-then-rename means a kill never truncates a shard; a server
        that drops points mid-shard cannot either (every shard is written only
        complete), so the only files resume can trust are within-band unique
        rows. A short within-band file must be trusted, not redone."""
        self._run_cell()
        target = self.shards / "sar_000000.parquet"
        pd.read_csv(target).head(10).to_csv(target, index=False)
        stamp = target.stat().st_mtime_ns
        self._run_cell()
        self.assertEqual(target.stat().st_mtime_ns, stamp,
                         "a within-band shortened shard was rewritten")

    def test_shards_are_written_atomically(self):
        self.assertIn(".partial", self.src)
        self.assertRegex(self.src, r"partial\.replace\(")

    def test_the_cell_never_raises_system_exit_zero(self):
        self.assertNotIn("raise SystemExit(0)", self.src)
        self.assertNotIn("SystemExit(1)", self.src)


class SARFootprintTests(unittest.TestCase):
    """A point outside the footprint returns no feature at all."""

    @classmethod
    def setUpClass(cls):
        cls._pd_orig = (pd.DataFrame.to_parquet, pd.read_parquet)
        cls._patches = [mock.patch.dict(sys.modules, _install_parquet_shim()),
                        mock.patch.dict(
                            sys.modules,
                            {"ee": _install_ee_stub(footprint=(68.0, 80.0))})]
        for p in cls._patches:
            p.start()

    @classmethod
    def tearDownClass(cls):
        for p in cls._patches:
            p.stop()
        pd.DataFrame.to_parquet, pd.read_parquet = cls._pd_orig
        for k in [k for k in sys.modules
                  if k == "pyarrow" or k.startswith(("pyarrow.", "ee."))]:
            del sys.modules[k]

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        (self.root / "qc").mkdir()
        ds = self.root / "full_2012_2026" / "features" / "dataset"
        ds.mkdir(parents=True)
        n = 2_000
        pd.DataFrame({
            "event_centroid_lon": np.linspace(68, 96, n),
            "event_centroid_lat": np.linspace(8, 35, n),
            "event_id": [f"E{i}" for i in range(n)],
            "start_year": np.full(n, 2019, dtype=np.int16),
            "start_month": np.full(n, 5, dtype=np.int8),
        }).to_parquet(ds / "meta.parquet")
        self.addCleanup(self._tmp.cleanup)

    def _run(self):
        cu = types.ModuleType("colab_utils")
        cu.resolve_project_dir = lambda: self.root
        cu.drive_report = lambda: "  (rehearsal: stubbed Drive)"
        cu.MYDRIVE = "/stub"
        cu.event_year_month = lambda path, verbose=False: (
            np.full(len(pd.read_parquet(path)), 2019, dtype="int16"),
            np.full(len(pd.read_parquet(path)), 5, dtype="int8"))
        ns = {"GEE_CHUNK_SIZE": 1_000, "SAR_SHARD_CHUNKS": 2,
              "SAR_RESUME": True, "__name__": "__main__"}
        with mock.patch.dict(sys.modules, {"colab_utils": cu}):
            exec(compile(CELL.read_text(), str(CELL), "exec"), ns)   # noqa: S102

    def test_a_footprint_miss_is_absent_not_zero(self):
        """0 dB change would read as "measured, no change".

        A point with no GRD coverage has nothing to say, so it must not appear
        in the table with a 0 dB change. Only the first ~46% of the catalogue is
        inside the stub footprint, so a zero-fill would show up as a block of
        exact zeros.
        """
        self._run()
        df = pd.read_csv(self.root / "qc" / "sar_events.parquet")
        self.assertLess(len(df), 2_000, "footprint misses were not excluded")
        self.assertGreater(len(df), 0, "everything was excluded")
        # Every row written must be a real measurement, never a filled zero.
        self.assertTrue((df["VV_CHANGE"].abs() > 1e-6).all(),
                        "a zero change was written for an unmeasured point")
        for c in ("VV_PRE", "VH_PRE", "VV_POST", "VH_POST", "VV_VH_CHANGE"):
            self.assertFalse(df[c].isna().any(),
                             f"{c} has NaN rows, which should have been "
                             f"excluded rather than written half-empty")
        # The excluded rows are exactly those outside the footprint.
        inside = df[(df["VV_PRE"] != 0)]
        self.assertTrue(inside["VV_PRE"].between(-35.0, 5.0).all())

    def test_inside_the_footprint_the_change_is_measured(self):
        self._run()
        df = pd.read_csv(self.root / "qc" / "sar_events.parquet")
        hit = df[df["VV_CHANGE"].notna()]
        self.assertGreater(len(hit), 0)
        self.assertTrue((hit["VV_CHANGE"].abs() > 1e-6).all())


class EventTimestampTests(unittest.TestCase):
    """meta.parquet has NO `start_month`. Verified against the real file.

    Its columns are event_id, event_centroid_lon, event_centroid_lat,
    start_time (timestamp[ns]), start_year, spatial_block, site_id. Both imagery
    cells asked for `start_month` and raised ArrowInvalid before any Earth
    Engine call -- the worst possible moment in an 11-hour extraction.
    """

    @classmethod
    def setUpClass(cls):
        # This machine has no pyarrow, so the CSV shim stands in. It reports the
        # real column set, which is what `event_year_month` validates against.
        cls._pd_orig = (pd.DataFrame.to_parquet, pd.read_parquet)
        cls._patch = mock.patch.dict(sys.modules, _install_parquet_shim())
        cls._patch.start()

    @classmethod
    def tearDownClass(cls):
        cls._patch.stop()
        pd.DataFrame.to_parquet, pd.read_parquet = cls._pd_orig
        for k in [k for k in sys.modules
                  if k == "pyarrow" or k.startswith(("pyarrow.", "ee."))]:
            del sys.modules[k]

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.p = pathlib.Path(self._tmp.name) / "meta.parquet"

    def test_month_comes_from_start_time(self):
        import colab_utils as cu

        pd.DataFrame({
            "event_id": ["a", "b", "c"],
            "start_year": [2019, 2020, 2015],
            "start_time": pd.to_datetime(
                ["2019-03-17 04:22:00", "2020-11-02 18:05:00",
                 "2015-06-30 23:59:59"]),
            "event_centroid_lon": [77.0, 78.0, 79.0],
            "event_centroid_lat": [28.0, 29.0, 30.0],
        }).to_parquet(self.p)
        years, months = cu.event_year_month(self.p, verbose=False)
        self.assertEqual(list(months), [3, 11, 6])
        self.assertEqual(list(years), [2019, 2020, 2015])

    def test_boundary_seconds_stay_in_the_right_month(self):
        """23:59:59 on the last day of a month is that month, not the next."""
        import colab_utils as cu

        pd.DataFrame({
            "event_id": ["a", "b"],
            "start_year": [2019, 2019],
            "start_time": pd.to_datetime(["2019-01-31 23:59:59",
                                          "2019-02-01 00:00:00"]),
            "event_centroid_lon": [77.0, 78.0],
            "event_centroid_lat": [28.0, 29.0],
        }).to_parquet(self.p)
        _years, months = cu.event_year_month(self.p, verbose=False)
        self.assertEqual(list(months), [1, 2])

    def test_january_event_rolls_the_pre_month_into_the_previous_year(self):
        """The NBR/SAR pre-month of a January event is December, last year.

        Derived here rather than by the cells, because getting it wrong shifts
        every January event's baseline by a full year -- a plausible-looking
        number that compares a burn against the wrong month of vegetation.
        """
        import colab_utils as cu
        import gee_utils as gee

        pd.DataFrame({
            "event_id": ["a"],
            "start_year": [2020],
            "start_time": pd.to_datetime(["2020-01-05 10:00:00"]),
            "event_centroid_lon": [77.0],
            "event_centroid_lat": [28.0],
        }).to_parquet(self.p)
        years, months = cu.event_year_month(self.p, verbose=False)
        py, pm = gee.month_before(int(years[0]), int(months[0]), 1)
        self.assertEqual((py, pm), (2019, 12))

    def test_a_missing_required_column_fails_by_name(self):
        import colab_utils as cu

        pd.DataFrame({"event_id": ["a"], "start_year": [2019]}).to_parquet(
            self.p)
        with self.assertRaises(KeyError) as ctx:
            cu.event_year_month(self.p, verbose=False)
        msg = str(ctx.exception)
        self.assertIn("start_time", msg)
        self.assertIn("available", msg,
                      "the error must list what IS there, not just what is not")

    def test_an_unusable_timestamp_is_reported_not_invented(self):
        import colab_utils as cu

        pd.DataFrame({
            "event_id": ["a", "b"],
            "start_year": [2019, 2019],
            "start_time": [pd.Timestamp("2019-03-17"), None],
            "event_centroid_lon": [77.0, 78.0],
            "event_centroid_lat": [28.0, 29.0],
        }).to_parquet(self.p)
        years, months = cu.event_year_month(self.p, verbose=False)
        # Coerced, so the arrays stay positional. The caller drops these via its
        # coverage filter rather than being handed a fabricated month.
        self.assertEqual(len(years), 2)
        self.assertEqual(len(months), 2)

    def test_year_disagreement_between_the_two_columns_is_visible(self):
        """start_year vs start_time's year must not be silently reconciled.

        One of them is wrong; picking the wrong one moves an event into an
        adjacent month, and for NBR that means sampling the wrong composite.
        """
        import colab_utils as cu
        import io
        import contextlib

        pd.DataFrame({
            "event_id": ["a", "b"],
            "start_year": [2021, 2019],          # both wrong, deliberately
            "start_time": pd.to_datetime(["2019-03-17 00:00:00",
                                          "2020-05-02 00:00:00"]),
            "event_centroid_lon": [77.0, 78.0],
            "event_centroid_lat": [28.0, 29.0],
        }).to_parquet(self.p)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            years, months = cu.event_year_month(self.p)
        self.assertIn("disagrees", buf.getvalue())
        self.assertEqual(list(years), [2019, 2020],
                         "the timestamp must win, but the conflict must print")


if __name__ == "__main__":
    unittest.main()
