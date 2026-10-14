# ============================================================================
# P7k -- COLAB CELL: NBR (dNBR) for the WHOLE catalogue, resumable
#
# WHY MONTHLY, NOT YEARLY
#
# NBR is `("B5" - "B7") / ("B5" + "B7")` on Landsat 8/9 surface reflectance --
# healthy vegetation reflects strongly in NIR (B5) and weakly in SWIR (B7);
# burned ground does the opposite.  dNBR = NBR_pre - NBR_post, so a POSITIVE
# dNBR means the pixel lost vegetation between the two dates.  A yearly
# composite mixes months and blurs exactly that signal, so this cell builds a
# **monthly** median composite and samples each event TWICE:
#
#     NBR_PRE   = the month BEFORE the event's month
#     NBR_POST  = the event's own month
#
# One month apart is short enough to catch a burn and long enough that a
# cloud-free median exists for most of India in the dry season.
#
# WHY IT IS STILL ~11 HOURS
#
# P7g measured 13.7 s per 5,000-point call, i.e. ~2.7 ms per pixel lookup.
# Two lookups per event x 7.27M events is ~14.5M lookups, so roughly 11 hours
# serially.  Monthly composites cost more to BUILD than yearly ones, but the
# dominant cost is the per-point sample, not the mosaic.  The cell therefore
# groups events by (year, month) and shares one composite across every event
# that needs it, which is where the real saving is.
#
# WHICH EVENTS GET AN ANSWER
#
# Landsat 8 starts 2013-04, Sentinel-2 2017-06.  Events before 2013 have NO
# imagery: their NBR columns stay NaN, never 0.  P7g verified the exact gap.
# The event-month composite must also exist for NBR_POST to be non-NaN.
#
# FOUR BUGS THIS VERSION FIXES  (see test_colab_p7k.py)
#
#   1. `gee.monthly_nbr_composite` never existed. Only the CALL existed.
#   2. `gee.imagery_gap` returns a dict keyed by sensor, not a list of years;
#      iterating it raised ValueError on 'no_landsat'.
#   3. Each shard was written with every row filled SO FAR, so overlapping
#      shards duplicated rows and the assembly sorted them without dedup.
#   4. `continue` on an uncovered chunk skipped the shard-boundary write, so a
#      shard whose last chunk had no imagery was never written at all.
#
# WHY IT MUST BE RESUMABLE  (same reason as P7h)
#
# Colab terminates idle sessions.  Results go to SHARDS of SHARD_CHUNKS
# chunks, each shard written atomically (`.partial` then rename).  On restart
# existing shards are loaded and their chunk ranges skipped.  Re-running is
# always safe and always resumes; nothing already on Drive is recomputed.
#
# OUTPUT
#
# `qc/nbr_events.parquet`, one row per event with:
#     row, event_id, start_year, NBR_PRE, NBR_POST, DNBR
# DNBR is NBR_PRE - NBR_POST, positive where vegetation was lost.  `row` is
# the position in meta.parquet's row order, which is what lets P7i / P7j join
# it back to X without a key.
# ============================================================================

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import colab_utils as cu
import gee_utils as gee

P7K_CELL_VERSION = "2026-10-07.3"

#: Points per `sampleRegions` call.  Same limit as P7h: 10 MiB is ~50k points,
#: so stay well under it.
CHUNK = int(globals().get("GEE_CHUNK_SIZE", 5_000))
#: Chunks per shard.  One shard = 250,000 rows, a few minutes of work, so a
#: kill loses at most that.
SHARD_CHUNKS = int(globals().get("NBR_SHARD_CHUNKS", 50))
#: Resume from a completed run rather than redoing it.
RESUME = bool(globals().get("NBR_RESUME", True))
#: Months to look back for the pre-fire composite.
PRE_LAG = int(globals().get("NBR_PRE_LAG", 1))
#: First year with usable Landsat 8 surface reflectance.
FIRST_IMAGERY_YEAR = 2013

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError(f"no project root under {cu.MYDRIVE}")
DATASET_DIR = PROJECT_ROOT / "full_2012_2026" / "features" / "dataset"
QC_DIR = PROJECT_ROOT / "qc"
SHARD_DIR = QC_DIR / "nbr_shards"
SHARD_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = QC_DIR / "nbr_manifest.json"
#: The user-facing output folder on Drive. The assembled table is written
#: here IN ADDITION to qc/, so the deliverable is in one obvious place;
#: qc/ keeps the pipeline-contract path that P7i/P7j read.
OUTPUT_DIR = PROJECT_ROOT / "nbr_sar_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 74)
print(f"P7k -- FULL NBR (dNBR) EXTRACTION  [v{P7K_CELL_VERSION}]")
print("=" * 74)
print(cu.drive_report())
print(f"  shard dir  : {SHARD_DIR}")
print(f"  chunk {CHUNK:,}  shard {SHARD_CHUNKS * CHUNK:,} rows  "
      f"resume {RESUME}  pre-lag {PRE_LAG}")
EARTHENGINE_PROJECT = "burnished-stone-510609-f5"
_ok = False
_pid = None
try:
    _ok, _pid = gee.authenticate(globals().get("EARTHENGINE_PROJECT"))
    gee.verify_can_read(_pid)
    print(f"  Earth Engine working (project {_pid!r})")
except Exception as exc:                            # noqa: BLE001
    print(f"  Earth Engine NOT usable: {type(exc).__name__}: {str(exc)[:200]}")
    raise SystemExit("P7k needs a verified Earth Engine session")

import ee                                                          # noqa: E402

# ---------------------------------------------------------------------------
# 1. THE COORDINATES
#
# meta.parquet is in the SAME ROW ORDER as X.parquet and y.parquet, which is
# what lets the result be joined back by position.  Only the columns the sampler
# needs are read -- the full meta table is never materialised here.
#
# The lon/lat column names are PROBED, not assumed. `start_month` is NOT a
# column: verified against the real meta.parquet on 2026-10-07, which carries
# `start_time` (timestamp[ns]) and `start_year` but no month. An earlier draft
# asked for `start_month` and raised ArrowInvalid on the first line of real
# work, which for an 11-hour cell is the worst possible moment to find out.
_meta_path = DATASET_DIR / "meta.parquet"
if not _meta_path.exists():
    raise FileNotFoundError(f"{_meta_path} not found -- run P6 first")
_MC = set(pq.ParquetFile(_meta_path).schema_arrow.names)
for _req in ("event_id", "event_centroid_lon", "event_centroid_lat"):
    if _req not in _MC:
        raise KeyError(f"meta.parquet has no column {_req!r}; "
                       f"available: {sorted(_MC)}")
years, months = cu.event_year_month(_meta_path)
_meta = pd.read_parquet(_meta_path, columns=["event_id",
                                             "event_centroid_lon",
                                             "event_centroid_lat"])
ids = _meta["event_id"].to_numpy()
lon = _meta["event_centroid_lon"].to_numpy().astype(np.float32)
lat = _meta["event_centroid_lat"].to_numpy().astype(np.float32)
del _meta
n = len(ids)
_ok = np.isfinite(lon) & np.isfinite(lat)
print(f"  events     : {n:,}   with finite coordinates: {int(_ok.sum()):,} "
      f"({100.0 * _ok.mean():.2f}%)")


# ---------------------------------------------------------------------------
# 2. RESUME STATE
#
# `done_chunks` is rebuilt from the shard files on disk, never from the
# manifest -- the manifest is a convenience readout, the shard files are the
# truth.  An earlier P7h version seeded it from the manifest and re-wrote
# shards that were already correct; the shard files are authoritative instead.
#
# A shard is trusted only if it holds EXACTLY the rows its chunk range covers.
# A torn shard holding half its rows would otherwise be marked done, and its
# missing half would never be re-extracted.
# ---------------------------------------------------------------------------
def shard_path(no: int) -> Path:
    return SHARD_DIR / f"nbr_{no:06d}.parquet"


n_chunks_total = int(np.ceil(n / CHUNK))
shards_total = int(np.ceil(n_chunks_total / SHARD_CHUNKS))

done_chunks: set = set()
_done_shards: set = set()
_done_rows = 0
if RESUME:
    for _p in sorted(SHARD_DIR.glob("nbr_*.parquet")):
        try:
            _sn = int(_p.stem.split("_")[-1])
        except (IndexError, ValueError):
            continue
        _c0 = _sn * SHARD_CHUNKS
        _c1 = min(_c0 + SHARD_CHUNKS, n_chunks_total)
        _row_lo = _c0 * CHUNK
        _row_hi = min(_c1 * CHUNK, n)
        try:
            _t = pq.read_table(_p, columns=["row"])
        except Exception:                            # noqa: BLE001
            print(f"  {_p.name}: UNREADABLE -- re-extracting this shard")
            _p.unlink(missing_ok=True)
            continue
        _rows = _t["row"].to_numpy()
        _rows_n = int(len(_rows))
        # A shard holds only the FILLED rows of its chunk band (rows with no
        # imagery in either month are skipped, so the count is NOT a fixed
        # slot size -- resume run 2 had 220,388 rows in a 250,000-row band
        # and an exact-count check would DISCARD it). Trust = every row is
        # inside the shard's own band and unique; overlap would mean the
        # banding is wrong (the P7h bug, where shards 0 and 1 each carried
        # rows 0..250k).
        #
        # An EMPTY shard is a legitimate outcome (no event in that band had
        # Landsat imagery -- pre-2013 or a month with no clear scenes) and
        # must be TRUSTED as done. The earlier version DISCARDed empty
        # shards too, which is what made the cell report "shards remaining"
        # forever: every re-run re-extracted the same empty bands, never
        # finishing.
        if (_rows_n != 0 and (
                int(_rows.min()) < _row_lo
                or int(_rows.max()) >= _row_hi
                or _rows_n != len(np.unique(_rows)))):
            print(f"  {_p.name}: {_rows_n:,} rows outside its band or not "
                  f"unique -- DISCARD")
            _p.unlink(missing_ok=True)
            continue
        for _c in range(_c0, _c1):
            done_chunks.add(_c)
        _done_shards.add(_sn)
        _done_rows += _rows_n
    if _done_shards:
        print(f"  resuming   : {len(_done_shards)}/{shards_total} shards, "
              f"{_done_rows:,} rows already on Drive")
else:
    for _p in SHARD_DIR.glob("nbr_*.parquet"):
        _p.unlink(missing_ok=True)
    print("  RESUME off: cleared the shard directory")

# Shard identity is a pure function of the chunk number, so it is derived,
# never remembered -- the bug that bit P7h.
print(f"  chunks     : {n_chunks_total:,}   shards: {shards_total}")
_pending_shards = [s for s in range(shards_total) if s not in _done_shards]
print(f"  to run     : {len(_pending_shards):,} shards "
      f"({100.0 * len(_done_shards) / max(shards_total, 1):.1f}% done)")

# ---------------------------------------------------------------------------
# 3. THE MONTHLY COMPOSITES
#
# One composite per (year, month) that any event in the catalogue actually
# needs.  A composite serves two roles: it is NBR_POST for events whose own
# month it is, and NBR_PRE for the events one month later.  So the set of
# months needed is every event month plus PRE_LAG before the earliest one.
#
# `gee.imagery_gap` returns a DICT of gap lists keyed by sensor
# (`{"no_landsat": [...], "no_sentinel2": [...]}`), not a flat list of years.
# An earlier version wrote `{int(y) for y in _gap}` over it, which iterated the
# KEYS and raised `ValueError: invalid literal for int() with base 10:
# 'no_landsat'` -- on the first run, after the hour of auth setup.
# ---------------------------------------------------------------------------
# Every event needs TWO months: its own (NBR_POST) and PRE_LAG before it
# (NBR_PRE). Both are added per event -- an earlier version added only the
# pre-month of the globally EARLIEST event, which left almost every other
# event's own pre-month outside the covered set and silently skipped those
# rows. That is a quiet loss of ~12% of a shard, not a crash.
_needed_months: set = set()
for _y, _m in zip(years.tolist(), months.tolist()):
    _needed_months.add((int(_y), int(_m)))
    _needed_months.add(gee.month_before(int(_y), int(_m), PRE_LAG))

_gap = gee.imagery_gap(sorted({y for y, _ in _needed_months}))
# ONLY the Landsat key decides coverage, because `nbr_monthly_composite` samples
# Landsat and nothing else. Unioning in the Sentinel-2 key would drop 2014-2016,
# which Landsat 8 covers perfectly well, and silently cost the run three years of
# its catalogue -- so the two gaps are NOT the same question.
_gap_years = set(_gap.get("no_landsat", ()))
_covered_months = sorted(
    m for m in _needed_months
    if m[0] not in _gap_years and gee.nbr_month_has_imagery(m[0], m[1]))
print(f"  months needed: {len(_needed_months)}   "
      f"with imagery: {len(_covered_months)}   "
      f"no-imagery years: {sorted(_gap_years) if _gap_years else 'none'}")
_dropped_by_date = sorted(
    m for m in _needed_months
    if m[0] not in _gap_years and not gee.nbr_month_has_imagery(m[0], m[1]))
if _dropped_by_date:
    print(f"  months the YEAR rule passed but the catalog is empty for "
          f"(sensor start is mid-month): {_dropped_by_date}")
    print("  Their events keep NaN -- median() of an empty collection is a")
    print("  zero-band image, which is what crashed run 1 in 2013-01.")

# Building every month's composite up front holds ~170 images at once. Each is
# built on the client as a lazy graph, but the graph is retained per month for
# the whole run, so the composites are built LAZILY per shard instead and
# released with it. A monthly composite is expensive to build on the server;
# the per-point sample still dominates, so this does not change the runtime.
def _composite(cache: dict, y: int, m: int):
    """Memoised composite for `(y, m)`.

    The cache is a PARAMETER, not a module global: the static cell checks
    forbid a function mutating a name it does not itself bind, and a mutable
    module-level dict shared across shards is exactly the hidden state that
    check exists to catch. A fresh dict per shard replaces the old
    `.clear()` call, so nothing leaks between shards.
    """
    key = (int(y), int(m))
    img = cache.get(key)
    if img is None:
        img = gee.nbr_monthly_composite(key[0], key[1])
        cache[key] = img
    return img


# ---------------------------------------------------------------------------
# 4. THE SHARD LOOP
#
# Two lookups per event: its own month's composite (NBR_POST) and the one
# PRE_LAG months before (NBR_PRE).  A month with no composite leaves NaN in
# both -- never 0, because 0 would be indistinguishable from a measured
# "no change", and NaN is how "could not measure" is encoded everywhere else.
#
# SHARD-major, not chunk-major, and this is the fix for two bugs at once:
#
#   1. An earlier version built `filled_rows` as `np.arange(0, lo)` PLUS this
#      chunk's rows, i.e. every row filled SO FAR, and wrote that to each
#      shard. Shard 1 then held rows 0..250k, shard 2 held 0..500k, and the
#      assembly sorted them with no dedup -- so `nbr_events.parquet` carried
#      every early row once per shard. A shard must hold ONLY its own chunks.
#   2. That same version `continue`d on an uncovered chunk, which jumped PAST
#      the `is_shard_end` write further down the loop body. A shard whose LAST
#      chunk had no imagery therefore never wrote at all, silently dropping the
#      filled rows from its earlier chunks. Buffering the whole shard and
#      flushing once removes both failure modes structurally.
#
# ---------------------------------------------------------------------------
def _by_month(rows):
    """`rows` grouped into `(year, month, row_array)` runs, month order kept.

    Grouping is the entire cost saving in this cell: a chunk of 5,000 events
    typically spans a handful of months, so sampling once per month rather than
    once per event is what turns ~1,500 round trips into fewer.
    """
    order: dict = {}
    for r in rows:
        order.setdefault((int(years[r]), int(months[r])), []).append(r)
    return [(y, m, np.asarray(v, dtype=np.int64))
            for (y, m), v in sorted(order.items())]


def _sample(image, rows, band="NBR"):
    """`{row: value}` for one `sampleRegions` call.

    The feature collection carries `row` as a property so the result can be
    matched back by KEY rather than by position -- positional matching is
    correct only while the server returns features in request order, which is
    not a documented guarantee.

    A point with no imagery returns `None` (masked), never 0. The distinction
    between "burnt" and "not measured" is the whole point of the NaN
    convention, and a masked pixel that came back as 0 would read as a
    complete absence of vegetation.
    """
    feats = ee.FeatureCollection(
        [ee.Feature(ee.Geometry.Point([float(lon[r]), float(lat[r])]),
                    {"row": int(r)}) for r in rows])
    res = image.sampleRegions(collection=feats, properties=["row"],
                              scale=100, geometries=False).getInfo()
    got: dict = {}
    for f in res.get("features", []):
        p = f.get("properties", {})
        v = p.get(band)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            v = _first_real(p)
        if v is not None:
            got[p.get("row")] = float(v)
    return got


def _first_real(props, exclude=("row",)):
    """The composite's own value, whatever the server named the band.

    `nbr_monthly_composite` renames its single band to `NBR`, but a composite
    that lost the rename (or an S1 two-band composite) would otherwise raise
    KeyError on a property that IS in the response. Taking the first non-key
    property is what `gee_utils.sample_points_chunked` already does, and the
    lookup stays tolerant rather than assuming a name.
    """
    for k, v in props.items():
        if k in exclude:
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def _flush(shard_no, frames):
    """Write one shard ATOMICALLY, so a kill never leaves a torn parquet.

    `shard_no` is passed in, never tracked as mutable state -- the P7h bug,
    where seeding the shard number from `min(done_chunks)` made a resume
    overwrite correct files and mislabel new ones.

    An empty shard is a LEGITIMATE outcome (no event in that band had
    Landsat imagery -- pre-2013 or a month with no clear scenes). The
    earlier version returned without writing it, so resume never trusted
    it as done and the cell re-extracted the same empty bands forever,
    reporting "shards remaining" on every re-run. Write the empty shard so
    resume can trust it.
    """
    pth = shard_path(shard_no)
    if frames:
        out = pd.concat(frames, ignore_index=True).sort_values("row")
        # A shard's own chunks cannot overlap, so any duplicate here is a bug, and
        # dropping it would hide that bug rather than surface it.
        if len(out) != out["row"].nunique():
            raise RuntimeError(f"shard {shard_no} assembled {len(out):,} rows for "
                               f"{out['row'].nunique():,} distinct rows -- shard "
                               f"boundaries are wrong")
    else:
        out = pd.DataFrame({"row": np.empty(0, dtype=np.int64)})
    pth.parent.mkdir(parents=True, exist_ok=True)
    partial = pth.with_suffix(".partial")
    out.to_parquet(partial, index=False)
    partial.replace(pth)          # atomic on POSIX and on Drive's FUSE


_t0 = time.time()
_written = 0
for _si, _sn in enumerate(_pending_shards, 1):
    _c0 = _sn * SHARD_CHUNKS
    _c1 = min(_c0 + SHARD_CHUNKS, n_chunks_total)
    _frames = []
    _shard_filled = 0
    _composites = {}                  # this shard's composite cache
    for c in range(_c0, _c1):
        if c in done_chunks:
            continue
        lo = c * CHUNK
        hi = min(lo + CHUNK, n)
        rows = np.arange(lo, hi)
        # Only covered rows are sent; uncovered ones stay NaN without a call.
        keep = np.array([(int(years[r]), int(months[r])) in _covered_months
                         for r in rows])
        rows = rows[keep]
        if len(rows) == 0:
            continue
        if not _ok[rows].all():
            rows = rows[_ok[rows]]
            if len(rows) == 0:
                continue

        # Group by the event's OWN month, so one composite is sampled once per
        # distinct month in the chunk rather than once per row.
        for my, mm, grp in _by_month(rows):
            py, pm = gee.month_before(my, mm, PRE_LAG)
            if (py, pm) not in _covered_months:
                continue               # no pre imagery: NBR_PRE stays NaN
            pre = _sample(_composite(_composites, py, pm), grp)
            post = _sample(_composite(_composites, my, mm), grp)
            _f = []
            for r in grp:
                _a = pre.get(int(r))
                _b = post.get(int(r))
                if _a is None and _b is None:
                    continue
                _f.append({
                    "row": int(r),
                    "event_id": ids[r],
                    "start_year": int(years[r]),
                    "start_month": int(mm),
                    "NBR_PRE": np.float32(_a) if _a is not None else np.nan,
                    "NBR_POST": np.float32(_b) if _b is not None else np.nan,
                })
            if _f:
                _fr = pd.DataFrame(_f)
                _fr["DNBR"] = _fr["NBR_PRE"] - _fr["NBR_POST"]
                _frames.append(_fr)
                _shard_filled += len(_fr)

    _flush(_sn, _frames)
    _written += _shard_filled
    done_chunks.update(range(_c0, _c1))
    MANIFEST.write_text(json.dumps({
        "cell_version": P7K_CELL_VERSION,
        "shards_total": shards_total,
        "shards_written_through": _sn,
        "chunk_size": CHUNK,
        "shard_chunks": SHARD_CHUNKS,
        "pre_lag": PRE_LAG,
        "rows_written": _written,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))

    _el = (time.time() - _t0) / 60.0
    _left = max(0, shards_total - len(_done_shards) - _si)
    _eta_h = (_el / _si) * _left / 60.0 if _si else 0.0
    print(f"  shard {_sn:>3}/{shards_total - 1} written "
          f"({_si}/{len(_pending_shards)})  {_shard_filled:,} rows  "
          f"{_el:.1f} min elapsed  ETA {_eta_h:.1f} h")

print(f"  extracted {_written:,} rows in {(time.time() - _t0) / 60:.1f} min")

# ---------------------------------------------------------------------------
# 5. FINALISE
#
# Assemble the shards into `qc/nbr_events.parquet`, NBR columns as float32 (the
# ledger's native type) and the `row` column as the join key back to X.
# ---------------------------------------------------------------------------
def _finalise(shards):
    frames = [pd.read_parquet(p) for p in shards]
    out = pd.concat(frames, ignore_index=True).sort_values("row")
    out = out.drop_duplicates(subset="row", keep="first").reset_index(drop=True)
    del frames

    # COMPLETENESS, LOUDLY. Shards now hold disjoint row ranges, so a short
    # table means shards are genuinely missing. A resume bug once assembled
    # 19,000 of 25,000 rows and reported success; a short NBR table silently
    # shrinks the dNBR feature population, so it must never pass quietly.
    present = sorted(int(p.stem.split("_")[-1]) for p in shards
                     if p.stem.split("_")[-1].isdigit())
    if len(present) < shards_total:
        missing = [s for s in range(shards_total) if s not in present]
        print(f"  INCOMPLETE: {len(present)}/{shards_total} shards present")
        print(f"  missing shards: {missing[:20]}"
              f"{' ...' if len(missing) > 20 else ''}")
        print("  Re-run to fill the gaps -- this is safe and resumes.")
        print("  DO NOT USE THIS TABLE for a feature until every shard is "
              "present.")
    else:
        print(f"  complete: {len(present)}/{shards_total} shards")

    print(f"  rows {len(out):,}  NBR_POST non-NaN "
          f"{int(out['NBR_POST'].notna().sum()):,}")
    print(f"  NBR_PRE  non-NaN {int(out['NBR_PRE'].notna().sum()):,}")
    print(f"  DNBR     non-NaN {int(out['DNBR'].notna().sum()):,}")
    print(f"  DNBR     mean {float(out['DNBR'].mean()):.4f}  "
          f"median {float(out['DNBR'].median()):.4f}  "
          f"> 0 (lost vegetation) {int((out['DNBR'] > 0).sum()):,}")

    # NBR is a signed index in [-1, 1]. A value outside that is a data or
    # geometry error, and letting it through would put a silent outlier into the
    # model -- most often a swapped NBR_BANDS order, which computes -NBR.
    for col in ("NBR_PRE", "NBR_POST"):
        v = out[col].dropna()
        if len(v) and (float(v.min()) < -1.0001 or float(v.max()) > 1.0001):
            print(f"  WARNING: {col} out of range: "
                  f"[{float(v.min()):.4f}, {float(v.max()):.4f}] -- "
                  f"expected [-1, 1]. Check the NBR_BANDS order.")

    qc = out[["row", "event_id", "start_year", "start_month",
              "NBR_PRE", "NBR_POST", "DNBR"]].copy()
    qc.to_parquet(QC_DIR / "nbr_events.parquet", index=False)
    print(f"  wrote {QC_DIR / 'nbr_events.parquet'}  ({len(qc):,} rows)")
    # The user-facing copy in the Drive output folder. Same bytes, second
    # location, so the deliverable is findable without knowing the qc/ layout.
    qc.to_parquet(OUTPUT_DIR / "nbr_events.parquet", index=False)
    print(f"  wrote {OUTPUT_DIR / 'nbr_events.parquet'}  ({len(qc):,} rows)")

    MANIFEST.write_text(json.dumps({
        "cell_version": P7K_CELL_VERSION,
        "shard_count": len(shards),
        "shards_total": shards_total,
        "shards": [p.name for p in shards],
        "rows_written": len(qc),
        "n_events": n,
        "chunk_size": CHUNK,
        "shard_chunks": SHARD_CHUNKS,
        "pre_lag": PRE_LAG,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    print(f"  manifest {MANIFEST}  {len(shards)} shards  {len(qc):,} rows")
    print("  P7k complete")


_shards = sorted(SHARD_DIR.glob("nbr_*.parquet"))
if _shards:
    _finalise(_shards)
else:
    # Guarded, not `raise SystemExit`: in a notebook cell that trips an
    # IPython/inspect bug on Python 3.13 and buries the message under three
    # chained tracebacks. A catalogue with no covered imagery is a REAL outcome
    # (a 2012-only run has none), so it is explained rather than treated as a
    # crash -- and no empty table is written, because a zero-row parquet would
    # join downstream as "no events have NBR", which is a claim, not an absence.
    print("  NOTHING TO FINALISE -- no shard was written.")
    if _needed_months and not _covered_months:
        print(f"  Every month this catalogue needs falls in a no-imagery year: "
              f"{sorted(_gap_years)}.")
        print("  Landsat 8 starts 2013, so a catalogue earlier than that cannot "
              "have NBR at all.")
    elif _pending_shards:
        print(f"  {len(_pending_shards)} shard(s) were pending and none produced "
              f"a row. Check the coordinate filter above.")
    print("  No parquet written. Re-run once imagery is available.")
