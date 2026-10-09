# ============================================================================
# P7l -- COLAB CELL: Sentinel-1 SAR structure change for the WHOLE catalogue
#
# WHAT THIS CELL PRODUCES, AND WHAT IT DELIBERATELY DOES NOT
#
# The ledger lists seven features under SAR_STRUCTURAL:
#
#     VV_CHANGE            <- this cell
#     VH_CHANGE            <- this cell
#     VV_VH_CHANGE         <- this cell
#     COHERENCE_LOSS       <- NOT produced
#     DNBR                 <- P7k (Sentinel-2 / Landsat), not SAR
#     DNDVI                <- P7k, not SAR
#     STRUCTURAL_CHANGE_PROBABILITY   <- NOT produced
#
# Only three of the seven are honestly obtainable from Sentinel-1 GRD. The other
# four are NOT filled with placeholders, because a placeholder in a feature
# column is indistinguishable from a measurement to everything downstream --
# the model would weight it, the SHAP report would attribute to it, and nothing
# downstream could tell it apart from a real observation.
#
#   COHERENCE_LOSS needs INTERFEROMETRY: two SLC acquisitions of the same
#   sub-swath with known baseline. GRD products are single-look complex-free
#   backscatter, so the phase difference that coherence is made of is not in
#   them. It would need a second asset (COPERNICUS/S1_SLC) with different orbit
#   geometry, and a GRD/VV number and an SLC/VV number are not comparable.
#
#   STRUCTURAL_CHANGE_PROBABILITY would be a MODEL OUTPUT, not an extraction.
#   It is a fitted function of the other SAR features, so emitting it here would
#   put a second, unversioned model in the feature table.
#
# WHY RADAR FOR FIRE AT ALL
#
# Optical indices fail on exactly the events worth studying: cloud, smoke and
# the monsoon. Sentinel-1 is active microwave, so it sees through cloud and
# smoke and works at night. A fire's effect on backscatter is real but indirect
# -- it is the change in surface roughness and moisture that a burn leaves, not
# thermal emission. So SAR change is EVIDENCE INDEPENDENT of the optical
# indices rather than a redundant copy of them, which is what makes it worth an
# 11-hour extraction.
#
# WHY THE CHANGE IS TAKEN IN dB DIFFERENCE, NOT dB
#
# GRD backscatter is already calibrated sigma0 in dB, so the median of a month
# IS the observation -- there is no index to compute. The feature is the
# DIFFERENCE between the two months, because sigma0 itself is dominated by
# geometry (incidence angle, terrain) that has nothing to do with the fire:
#
#     VV_CHANGE = VV_POST - VV_PRE
#
# A consistent sign convention matters here more than in the optical case,
# because a fire in forest DECREASES VV (less volume scattering off canopy)
# while one in grass can INCREASE it. The cell therefore reports the sign
# distribution rather than assuming a direction.
#
# WHY MONTHLY, AND WHY PRE_LAG
#
# Identical to P7k: one month before and the event's own month, monthly median.
# Sentinel-1's 12-day repeat orbit means most months hold several looks, so a
# monthly median is genuinely cloud-independent -- there is no atmosphere to
# mask out. `filterBounds`/`filterGeometry` orbit polygon handling is left to the
# collection; GRD scenes that do not cover a point simply do not sample it.
#
# COVERAGE. Sentinel-1 GRD starts 2014-05-03, which is LATER than Landsat 8's
# 2013 and much later than the catalogue's 2012 start. So this cell's gap is
# NOT a prefix of the optical gap, and it must be asked separately
# (`gee.sar_imagery_gap`). Deriving it from `gee.imagery_gap` would send 2013
# events to a sensor that has no data there, and an empty sample would be stored
# as if it were a measurement.
#
# WHY IT MUST BE RESUMABLE  (same reason as P7h and P7k)
#
# ~11 hours. Colab will interrupt it. Shards of SHARD_CHUNKS chunks, written
# atomically (`.partial` then rename), shard identity derived from the chunk
# number, resume state rebuilt from the shard files rather than the manifest.
#
# OUTPUT
#
# `qc/sar_events.parquet`, one row per event with:
#     row, event_id, start_year, start_month,
#     VV_PRE, VH_PRE, VV_POST, VH_POST,
#     VV_CHANGE, VH_CHANGE, VV_VH_CHANGE
# `row` is the position in meta.parquet's row order, so P7i / P7j join it to X
# without a key.
# ============================================================================

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import colab_utils as cu
import gee_utils as gee

P7L_CELL_VERSION = "2026-10-07.3"

#: Points per `sampleRegions` call. One composite carries TWO bands, so the
#: request payload per point is roughly double the optical case; 5,000 stays well
#: inside the 10 MiB limit.
CHUNK = int(globals().get("GEE_CHUNK_SIZE", 5_000))
#: Chunks per shard. One shard is a few minutes of work, so a kill loses at most
#: that.
SHARD_CHUNKS = int(globals().get("SAR_SHARD_CHUNKS", 50))
#: Resume from a completed run rather than redoing it.
RESUME = bool(globals().get("SAR_RESUME", True))
#: Months to look back for the pre-fire composite.
PRE_LAG = int(globals().get("SAR_PRE_LAG", 1))
#: Backscatter is log-scaled; a change larger than this in dB is almost certainly
#: a geometry or layover artefact rather than a burn, and is worth flagging.
MAX_PLAUSIBLE_DB_CHANGE = 15.0

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError(f"no project root under {cu.MYDRIVE}")
DATASET_DIR = PROJECT_ROOT / "full_2012_2026" / "features" / "dataset"
QC_DIR = PROJECT_ROOT / "qc"
SHARD_DIR = QC_DIR / "sar_shards"
SHARD_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = QC_DIR / "sar_manifest.json"
#: The user-facing output folder on Drive. The assembled table is written
#: here IN ADDITION to qc/, so the deliverable is in one obvious place;
#: qc/ keeps the pipeline-contract path that P7i/P7j read.
OUTPUT_DIR = PROJECT_ROOT / "nbr_sar_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 74)
print(f"P7l -- SENTINEL-1 SAR STRUCTURE CHANGE  [v{P7L_CELL_VERSION}]")
print("=" * 74)
print(cu.drive_report())
print(f"  shard dir  : {SHARD_DIR}")
print(f"  chunk {CHUNK:,}  shard {SHARD_CHUNKS * CHUNK:,} rows  "
      f"resume {RESUME}  pre-lag {PRE_LAG}")
print(f"  producing  : VV_CHANGE, VH_CHANGE, VV_VH_CHANGE")
print(f"  NOT making : COHERENCE_LOSS (needs SLC interferometry), "
      f"STRUCTURAL_CHANGE_PROBABILITY (a model output, not an extraction)")

_ok_auth = False
_pid = None
try:
    _ok_auth, _pid = gee.authenticate(globals().get("EARTHENGINE_PROJECT"))
    gee.verify_can_read(_pid)
    print(f"  Earth Engine working (project {_pid!r})")
except Exception as exc:                            # noqa: BLE001
    print(f"  Earth Engine NOT usable: {type(exc).__name__}: {str(exc)[:200]}")
    raise SystemExit("P7l needs a verified Earth Engine session")

import ee                                                          # noqa: E402

# ---------------------------------------------------------------------------
# 1. THE COORDINATES
#
# meta.parquet is in the same row order as X and y, which is what lets the result
# be joined back by position. Column names are PROBED against the schema rather
# than assumed: `start_month` is NOT a column -- verified against the real file
# on 2026-10-07, which carries `start_time` (timestamp[ns]) and `start_year`.
# `cu.event_year_month` derives the month from the timestamp and cross-checks
# the year against it.
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
_ok_coords = np.isfinite(lon) & np.isfinite(lat)
print(f"  events     : {n:,}   with finite coordinates: "
      f"{int(_ok_coords.sum()):,} "
      f"({100.0 * _ok_coords.mean():.2f}%)")


# ---------------------------------------------------------------------------
# 2. RESUME STATE
#
# Rebuilt from the shard files on disk, never from the manifest: the manifest is
# a convenience readout, the shard files are the truth. A shard is trusted only
# if it holds exactly the rows its chunk range covers, so a torn shard is
# re-extracted instead of being marked done.
# ---------------------------------------------------------------------------
def shard_path(no: int) -> Path:
    return SHARD_DIR / f"sar_{no:06d}.parquet"


n_chunks_total = int(np.ceil(n / CHUNK))
shards_total = int(np.ceil(n_chunks_total / SHARD_CHUNKS))

done_chunks: set = set()
_done_shards: set = set()
_done_rows = 0
if RESUME:
    for _p in sorted(SHARD_DIR.glob("sar_*.parquet")):
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
        # slot size -- P7k run 2 had 220,388 rows in a 250,000-row band and
        # an exact-count check would DISCARD it). Trust = every row is inside
        # the shard's own band and unique.
        if (_rows_n == 0
                or int(_rows.min()) < _row_lo
                or int(_rows.max()) >= _row_hi
                or _rows_n != len(np.unique(_rows))):
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
    for _p in SHARD_DIR.glob("sar_*.parquet"):
        _p.unlink(missing_ok=True)
    print("  RESUME off: cleared the shard directory")

# Shard identity is a pure function of the chunk number, so it is derived, never
# remembered -- the P7h bug, where seeding it from `min(done_chunks)` made a
# resume overwrite correct shards and mislabel the new ones.
print(f"  chunks     : {n_chunks_total:,}   shards: {shards_total}")
_pending_shards = [s for s in range(shards_total) if s not in _done_shards]
print(f"  to run     : {len(_pending_shards):,} shards "
      f"({100.0 * len(_done_shards) / max(shards_total, 1):.1f}% done)")

# ---------------------------------------------------------------------------
# 3. THE MONTHS, AND THE SENTINEL-1 COVERAGE GAP
#
# Every event needs two months: its own (POST) and PRE_LAG earlier (PRE). BOTH
# are added per event -- adding only the pre-month of the globally earliest
# event leaves almost every other event's pre-month outside the covered set and
# silently skips those rows.
#
# `sar_imagery_gap` is asked SEPARATELY from `imagery_gap`. Sentinel-1 starts
# 2014, later than Landsat 8's 2013, so the two gaps are different questions
# and the optical one cannot answer this one.
# ---------------------------------------------------------------------------
_needed_months: set = set()
for _y, _m in zip(years.tolist(), months.tolist()):
    _needed_months.add((int(_y), int(_m)))
    _needed_months.add(gee.month_before(int(_y), int(_m), PRE_LAG))

_gap = gee.sar_imagery_gap(sorted({y for y, _ in _needed_months}))
_gap_years = set(_gap.get("no_sentinel1", ()))
# Year-granular coverage is not enough: GEE's S1 GRD archive starts
# 2014-10-03, so 2014-01..09 pass the year rule and are EMPTY in the
# catalog. median() of an empty collection is a zero-band image, which
# crashes sampleRegions with "Invalid numInputs: 0" -- run 1's exact death.
_covered_months = sorted(
    m for m in _needed_months
    if m[0] not in _gap_years and gee.sar_month_has_imagery(m[0], m[1]))
_dropped_by_date = sorted(
    m for m in _needed_months
    if m[0] not in _gap_years and not gee.sar_month_has_imagery(m[0], m[1]))
print(f"  months needed: {len(_needed_months)}   "
      f"with S1 imagery: {len(_covered_months)}   "
      f"no-imagery years: {sorted(_gap_years) if _gap_years else 'none'}")
if _dropped_by_date:
    print(f"  months the YEAR rule passed but the catalog is empty for "
          f"(S1 archive starts 2014-10-03): {_dropped_by_date}")
    print("  Their events keep NaN -- median() of an empty collection is a")
    print("  zero-band image, which is what crashed run 1 with")
    print("  'sampleRegions: Invalid numInputs: 0' in 2014-04.")

# Composites are memoised per shard and released after it. Each is an expensive
# server-side graph, and holding every month for the whole run pins memory for no
# gain -- the per-point sample dominates the runtime either way.
def _composite(cache: dict, y: int, m: int):
    """Memoised composite for `(y, m)`.

    The cache is a PARAMETER, not a module global: the static cell checks
    forbid a function mutating a name it does not itself bind. A fresh dict
    per shard replaces the old `.clear()` call, so nothing leaks between
    shards.
    """
    key = (int(y), int(m))
    img = cache.get(key)
    if img is None:
        img = gee.sar_monthly_composite(key[0], key[1])
        cache[key] = img
    return img


# ---------------------------------------------------------------------------
# 4. THE SHARD LOOP
#
# SHARD-major, matching P7k. Each shard buffers only its OWN chunks and is
# written once, atomically. An earlier chunk-major draft accumulated every row
# filled so far into each shard, so shards overlapped and the assembly
# duplicated rows; and a `continue` on an uncovered chunk skipped the
# shard-boundary write, so a shard whose last chunk had no imagery was never
# written at all. Buffering per shard removes both structurally.
#
# A point outside any GRD footprint returns no feature, so it is simply absent
# from the response and stays NaN -- never 0. Sentinel-1 covers less of the
# globe than Sentinel-2, so a meaningful share of events will be NaN by design.
# ---------------------------------------------------------------------------
def _by_month(rows):
    """`rows` grouped into `(year, month, row_array)`, month order preserved.

    Grouping is the cost saving: one composite is sampled once per distinct
    month in the chunk rather than once per row.
    """
    order: dict = {}
    for r in rows:
        order.setdefault((int(years[r]), int(months[r])), []).append(r)
    return [(y, m, np.asarray(v, dtype=np.int64))
            for (y, m), v in sorted(order.items())]


def _sample(image, rows):
    """`{row: {"VV": v, "VH": v}}` for one `sampleRegions` call.

    Matched back by the `row` PROPERTY, not by position: positional matching is
    correct only while the server returns features in request order, which is
    not a documented guarantee, and a silently shuffled response would attach
    every value to the wrong event.
    """
    feats = ee.FeatureCollection(
        [ee.Feature(ee.Geometry.Point([float(lon[r]), float(lat[r])]),
                    {"row": int(r)}) for r in rows])
    res = image.sampleRegions(collection=feats, properties=["row"],
                              scale=100, geometries=False).getInfo()
    got: dict = {}
    for f in res.get("features", []):
        p = f.get("properties", {})
        vals = {}
        for nm in gee.S1_BANDS:
            v = p.get(nm)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                vals[nm] = float(v)
        if vals:
            got[p.get("row")] = vals
    return got


def _flush(shard_no, frames):
    """Write one shard ATOMICALLY, so a kill never leaves a torn parquet.

    `shard_no` is passed in, never tracked as mutable state.
    """
    if not frames:
        return
    pth = shard_path(shard_no)
    out = pd.concat(frames, ignore_index=True).sort_values("row")
    if len(out) != out["row"].nunique():
        raise RuntimeError(f"shard {shard_no} assembled {len(out):,} rows for "
                           f"{out['row'].nunique():,} distinct rows -- shard "
                           f"boundaries are wrong")
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
        # Uncovered months are skipped without a call; they were NaN and stay
        # NaN. NaN means "could not measure", which 0 cannot express -- 0 dB
        # change would read as "measured, no change".
        keep = np.array([(int(years[r]), int(months[r])) in _covered_months
                         for r in rows])
        rows = rows[keep]
        if len(rows) == 0:
            continue
        if not _ok_coords[rows].all():
            rows = rows[_ok_coords[rows]]
            if len(rows) == 0:
                continue

        for my, mm, grp in _by_month(rows):
            py, pm = gee.month_before(my, mm, PRE_LAG)
            if (py, pm) not in _covered_months:
                continue               # no pre imagery: PRE stays NaN
            pre = _sample(_composite(_composites, py, pm), grp)
            post = _sample(_composite(_composites, my, mm), grp)
            _f = []
            for r in grp:
                a = pre.get(int(r))
                b = post.get(int(r))
                if not a and not b:
                    continue
                a = a or {}
                b = b or {}
                _f.append({
                    "row": int(r),
                    "event_id": ids[r],
                    "start_year": int(years[r]),
                    "start_month": int(mm),
                    "VV_PRE": np.float32(a.get("VV", np.nan)),
                    "VH_PRE": np.float32(a.get("VH", np.nan)),
                    "VV_POST": np.float32(b.get("VV", np.nan)),
                    "VH_POST": np.float32(b.get("VH", np.nan)),
                })
            if _f:
                _fr = pd.DataFrame(_f)
                for _p in ("VV", "VH"):
                    _fr[f"{_p}_CHANGE"] = _fr[f"{_p}_POST"] - _fr[f"{_p}_PRE"]
                # VV/VH ratio is the standard structure proxy: dual-pol
                # separation distinguishes volume scattering (forest) from
                # surface scattering (bare/burnt). Taken as a DIFFERENCE because
                # the ratio itself is dominated by the same incidence-angle
                # geometry the individual channels are.
                _fr["VV_VH_CHANGE"] = (_fr["VV_CHANGE"] - _fr["VH_CHANGE"])
                _frames.append(_fr)
                _shard_filled += len(_fr)

    _flush(_sn, _frames)
    _written += _shard_filled
    done_chunks.update(range(_c0, _c1))
    MANIFEST.write_text(json.dumps({
        "cell_version": P7L_CELL_VERSION,
        "shards_total": shards_total,
        "shards_written_through": _sn,
        "chunk_size": CHUNK,
        "shard_chunks": SHARD_CHUNKS,
        "pre_lag": PRE_LAG,
        "rows_written": _written,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))

    _el = (time.time() - _t0) / 60.0
    _left = max(0, shards_total - len(_done_shards) - _si)
    _eta_h = (_el / _si) * _left / 60.0 if _si else 0.0
    print(f"  shard {_sn:>3}/{shards_total - 1} written "
          f"({_si}/{len(_pending_shards)})  {_shard_filled:,} rows  "
          f"{_el:.1f} min elapsed  ETA {_eta_h:.1f} h")

print(f"  extracted {_written:,} rows in {(time.time() - _t0) / 60.0:.1f} min")


# ---------------------------------------------------------------------------
# 5. FINALISE
# ---------------------------------------------------------------------------
def _finalise(shards):
    frames = [pd.read_parquet(p) for p in shards]
    out = pd.concat(frames, ignore_index=True).sort_values("row")
    out = out.drop_duplicates(subset="row", keep="first").reset_index(drop=True)
    del frames

    # COMPLETENESS, LOUDLY. A short table silently shrinks every downstream rate
    # computed from it, so it must never pass quietly.
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

    cols = ["row", "event_id", "start_year", "start_month",
            "VV_PRE", "VH_PRE", "VV_POST", "VH_POST",
            "VV_CHANGE", "VH_CHANGE", "VV_VH_CHANGE"]
    print(f"  rows {len(out):,}")
    for c in cols[4:]:
        print(f"  {c:<12} non-NaN {int(out[c].notna().sum()):>10,}  "
              f"({100.0 * out[c].notna().mean():.2f}%)")

    # The sign is NOT assumed. A forest fire reduces VV (less volume scattering
    # off canopy) while a grass fire can raise it, so a mean near zero here is a
    # real mixture and not a bug -- but a mean far from zero in ONE channel would
    # suggest a geometry or layover problem rather than fire.
    for c in ("VV_CHANGE", "VH_CHANGE", "VV_VH_CHANGE"):
        v = out[c].dropna()
        if not len(v):
            continue
        print(f"  {c:<12} mean {float(v.mean()):+.3f} dB  "
              f"median {float(v.median()):+.3f}  "
              f">0 {100.0 * float((v > 0).mean()):.1f}%")

    # GRD sigma0 is in dB and bounded in practice well below +/-50. A change
    # beyond that is geometry or layover, not fire, and would otherwise enter the
    # model as an enormous "structural change" outlier.
    for c in ("VV_CHANGE", "VH_CHANGE"):
        v = out[c].dropna()
        if len(v) and float(v.abs().max()) > MAX_PLAUSIBLE_DB_CHANGE:
            _big = int((v.abs() > MAX_PLAUSIBLE_DB_CHANGE).sum())
            print(f"  WARNING: {c}: {_big:,} rows exceed "
                  f"+/-{MAX_PLAUSIBLE_DB_CHANGE:.0f} dB "
                  f"(max {float(v.abs().max()):.1f}). Those are more likely "
                  f"incidence-angle or layover artefacts than fires.")

    qc = out[cols].copy()
    qc.to_parquet(QC_DIR / "sar_events.parquet", index=False)
    print(f"  wrote {QC_DIR / 'sar_events.parquet'}  ({len(qc):,} rows)")
    # The user-facing copy in the Drive output folder. Same bytes, second
    # location, so the deliverable is findable without knowing the qc/ layout.
    qc.to_parquet(OUTPUT_DIR / "sar_events.parquet", index=False)
    print(f"  wrote {OUTPUT_DIR / 'sar_events.parquet'}  ({len(qc):,} rows)")

    MANIFEST.write_text(json.dumps({
        "cell_version": P7L_CELL_VERSION,
        "shard_count": len(shards),
        "shards_total": shards_total,
        "shards": [p.name for p in shards],
        "rows_written": len(qc),
        "n_events": n,
        "chunk_size": CHUNK,
        "shard_chunks": SHARD_CHUNKS,
        "pre_lag": PRE_LAG,
        "features": ["VV_CHANGE", "VH_CHANGE", "VV_VH_CHANGE"],
        "not_produced": ["COHERENCE_LOSS", "STRUCTURAL_CHANGE_PROBABILITY"],
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    print(f"  manifest {MANIFEST}  {len(shards)} shards  {len(qc):,} rows")
    print("  P7l complete")


_shards = sorted(SHARD_DIR.glob("sar_*.parquet"))
if _shards:
    _finalise(_shards)
else:
    # Guarded rather than `raise SystemExit`: in a notebook cell that trips an
    # IPython/inspect bug on Python 3.13 and buries the message. A catalogue
    # with no Sentinel-1 coverage is a REAL outcome, so it is explained rather
    # than treated as a crash -- and no empty table is written, because a
    # zero-row parquet would join downstream as "no events have SAR", which is a
    # claim rather than an absence.
    print("  NOTHING TO FINALISE -- no shard was written.")
    if _needed_months and not _covered_months:
        print(f"  Every month this catalogue needs is before Sentinel-1's "
              f"{gee.S1_FIRST_YEAR} start: {sorted(_gap_years)}.")
        print("  Sentinel-1 GRD begins 2014-05, so these events cannot have "
              "SAR at all.")
    elif _pending_shards:
        print(f"  {len(_pending_shards)} shard(s) were pending and none "
              f"produced a row. Check the coordinate filter above.")
    print("  No parquet written. Re-run once the catalogue reaches 2014.")
