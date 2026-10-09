# ============================================================================
# P7h -- COLAB CELL: land cover for the WHOLE catalogue, resumable
#
# WHY THIS REPLACES THE 10.45 GB DOWNLOAD
#
# P7g answered the question from a 120,000-event sample in 329s. To build a
# usable label you need land cover for all 7,265,849 events, and the choice was:
#
#   direct download   102 ESA WorldCover tiles, ~10.45 GB, hours of transfer
#   Earth Engine      ~1,454 chunked calls, no download, values returned
#
# Measured rate from P7g: 119,999 points in 329s across 24 calls = 13.7s/call.
# So 7.27M points is ~1,454 calls, about **5.5 hours serially**. No bytes move
# across the link except small JSON responses. GEE wins on every axis that
# matters, so this cell is the download's replacement.
#
# WHY IT MUST BE RESUMABLE
#
# Colab terminates idle sessions and caps a session's lifetime. A 5.5 hour
# extraction WILL be interrupted, probably more than once. So:
#
#   * results are written in SHARDS of 50 chunks (250k rows)
#   * a shard is written atomically, so a kill leaves no torn parquet
#   * on restart, existing shards are loaded and their row-index ranges skipped
#   * a manifest records what is done
#
# Re-running is always safe and always resumes. Nothing is ever recomputed that
# already landed on Drive.
#
# WHAT THE OUTPUT BUYS
#
# `LANDCOVER_VEGETATED` is the first label input that is ABSENT FROM X. The
# weak label was built entirely from columns in X, which is why it could be
# reproduced by a model reading those columns. A label with a substrate term
# cannot be, so model B finally has something honest to learn.
#
# Note the class table is in `gee_utils`, verified against the catalog: code 80
# is permanent water and code 70 is snow and ice, and BOTH are excluded from
# VEGETATED. An earlier version had 80 in the vegetated set, which would have
# labelled open water as burnable substrate.
# ============================================================================

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import colab_utils as cu
import gee_utils as gee

P7H_CELL_VERSION = "2026-10-04.2"

#: Points per `sampleRegions` call. Earth Engine's request body limit is 10 MiB,
#: roughly 50k points, so stay well under it.
CHUNK = int(globals().get("GEE_CHUNK_SIZE", 5_000))
#: Chunks per shard. One shard = 250,000 rows, about 3.5 minutes of work, so a
#: kill loses at most that.
SHARD_CHUNKS = int(globals().get("LANDCOVER_SHARD_CHUNKS", 50))
#: Resume from a completed run rather than redoing it.
RESUME = bool(globals().get("LANDCOVER_RESUME", True))

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError(f"no project root under {cu.MYDRIVE}")
DATASET_DIR = PROJECT_ROOT / "full_2012_2026" / "features" / "dataset"
QC_DIR = PROJECT_ROOT / "qc"
SHARD_DIR = QC_DIR / "landcover_shards"
SHARD_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = QC_DIR / "landcover_manifest.json"

print("=" * 74)
print(f"P7h -- FULL LAND-COVER EXTRACTION  [v{P7H_CELL_VERSION}]")
print("=" * 74)
print(cu.drive_report())
print(f"  shard dir  : {SHARD_DIR}")
print(f"  chunk {CHUNK:,}  shard {SHARD_CHUNKS * CHUNK:,} rows  "
      f"resume {RESUME}")

_ok = False
_pid = None
try:
    _ok, _pid = gee.authenticate(globals().get("EARTHENGINE_PROJECT"))
    gee.verify_can_read(_pid)
    print(f"  Earth Engine working (project {_pid!r})")
except Exception as exc:                            # noqa: BLE001
    print(f"  Earth Engine NOT usable: {type(exc).__name__}: {str(exc)[:200]}")
    raise SystemExit("P7h needs a verified Earth Engine session")

import ee                                                          # noqa: E402

# ---------------------------------------------------------------------------
# 1. THE COORDINATES
#
# meta.parquet is in the SAME ROW ORDER as X.parquet and y.parquet, which is
# what lets the result be joined back by position with no spatial matching and
# no chance of pairing an event with a neighbour.
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("1. COORDINATES")
print("=" * 74)
_MC = set(pq.ParquetFile(DATASET_DIR / "meta.parquet").schema_arrow.names)
_keep = [c for c in ("event_centroid_lon", "event_centroid_lat",
                     "event_id", "start_year") if c in _MC]
meta = pd.read_parquet(DATASET_DIR / "meta.parquet", columns=_keep)
N = len(meta)
print(f"  events: {N:,}   columns: {_keep}")

ids = (meta["event_id"].to_numpy() if "event_id" in _keep
       else np.arange(N, dtype=np.int64))
lon = meta["event_centroid_lon"].to_numpy(dtype="float64")
lat = meta["event_centroid_lat"].to_numpy(dtype="float64")
ok_coords = np.isfinite(lon) & np.isfinite(lat)
print(f"  with finite coordinates: {int(ok_coords.sum()):,} "
      f"({100.0 * ok_coords.mean():.2f}%)")
idx_all = np.flatnonzero(ok_coords)

# ---------------------------------------------------------------------------
# 2. WHAT IS ALREADY DONE
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("2. RESUME STATE")
print("=" * 74)


def shard_path(n: int) -> Path:
    return SHARD_DIR / f"lc_{n:06d}.parquet"


n_chunks_total = int(np.ceil(len(idx_all) / CHUNK))
print(f"  total chunks needed: {n_chunks_total:,} "
      f"({len(idx_all):,} points / {CHUNK:,})")

done_chunks = set()
if RESUME:
    for p in sorted(SHARD_DIR.glob("lc_*.parquet")):
        try:
            n = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        # A shard is only trusted if it has the expected row count.
        try:
            rows = pq.ParquetFile(p).metadata.num_rows
        except Exception:                           # noqa: BLE001
            print(f"  {p.name}: UNREADABLE -- re-extracting this shard")
            p.unlink(missing_ok=True)
            continue
        expect = min(SHARD_CHUNKS * CHUNK,
                     max(0, len(idx_all) - n * SHARD_CHUNKS * CHUNK))
        if rows != expect:
            print(f"  {p.name}: {rows:,} rows, expected {expect:,} -- DISCARD")
            p.unlink(missing_ok=True)
            continue
        for c in range(n * SHARD_CHUNKS, (n + 1) * SHARD_CHUNKS):
            if c < n_chunks_total:
                done_chunks.add(c)

print(f"  shards found  : {len(list(SHARD_DIR.glob('lc_*.parquet'))):,}")
print(f"  chunks done   : {len(done_chunks):,} of {n_chunks_total:,} "
      f"({100.0 * len(done_chunks) / max(n_chunks_total, 1):.1f}%)")

if not RESUME:
    for p in SHARD_DIR.glob("lc_*.parquet"):
        p.unlink()
    done_chunks = set()
    print("  RESUME off: cleared the shard directory")

remaining = [c for c in range(n_chunks_total) if c not in done_chunks]
print(f"  chunks to run : {len(remaining):,}")
if remaining:
    _eta = len(remaining) * 13.7 / 3600.0
    print(f"  ETA serial    : {_eta:.1f} hours "
          f"(13.7s/call measured in P7g)")
    print(f"  -- expect Colab to interrupt this. That is fine: re-run and it")
    print(f"     resumes. Nothing already written will be recomputed.")
else:
    print("  nothing to extract -- every chunk is already on Drive")
    print("  (re-run with LANDCOVER_RESUME=False to force a clean rebuild)")

# ---------------------------------------------------------------------------
# 3. EXTRACT
#
# Guarded rather than exited: `raise SystemExit` in a notebook cell trips an
# IPython/inspect bug on Python 3.13 and buries the real message under three
# chained tracebacks.
# ---------------------------------------------------------------------------


def _finish():
    """Assemble the shards into one parquet and print the distribution."""
    print()
    print("=" * 74)
    print("4. ASSEMBLE")
    print("=" * 74)
    _shards = sorted(SHARD_DIR.glob("lc_*.parquet"))
    if not _shards:
        print("  no shards on Drive -- nothing to assemble")
        return
    _parts = [pd.read_parquet(p) for p in _shards]
    df = pd.concat(_parts, ignore_index=True).sort_values("row")
    df = df.drop_duplicates(subset="row", keep="first").reset_index(drop=True)

    # COMPLETENESS, LOUDLY. A resume bug once assembled 19,000 of 25,000 rows
    # and reported success. A short table here silently shrinks every downstream
    # label and every rate computed from it, so it must never pass quietly.
    _expected = int(len(idx_all))
    if len(df) < _expected:
        _missing = _expected - len(df)
        print(f"  INCOMPLETE: {len(df):,} of {_expected:,} rows "
              f"({_missing:,} missing, {100.0 * len(df) / _expected:.1f}%).")
        _done = sorted(int(p.stem.split("_")[1]) for p in _shards
                       if p.stem.split("_")[1].isdigit())
        print(f"  shards present: {_done}")
        print("  Re-run to fill the gaps -- this is safe and resumes.")
        print("  DO NOT USE THIS TABLE for a label until it is complete.")
    else:
        print(f"  complete: {len(df):,} of {_expected:,} rows")

    _out = QC_DIR / "landcover_events.parquet"
    try:
        df.to_parquet(_out, index=False)
        print(f"  Wrote {_out}  ({len(df):,} rows)")
    except Exception as exc:                        # noqa: BLE001
        print(f"  parquet write failed ({type(exc).__name__}); writing CSV")
        _out = QC_DIR / "landcover_events.csv"
        df.to_csv(_out, index=False)
        print(f"  Wrote {_out}  ({len(df):,} rows)")

    _n = len(df)
    print()
    print(f"  {'class':<6} {'substrate':<24} {'n':>12} {'share':>8}")
    for _c, _nm in sorted(gee.WORLDCOVER_CLASSES.items()):
        _k = int((df["WORLDCOVER_CLASS"] == _c).sum())
        print(f"  {_c:<6} {_nm:<24} {_k:>12,} {100.0 * _k / max(_n, 1):>7.2f}%")
    _miss = int((df["WORLDCOVER_CLASS"] < 0).sum())
    print(f"  {'-':<6} {'UNMAPPED':<24} {_miss:>12,} "
          f"{100.0 * _miss / max(_n, 1):>7.2f}%")
    if _miss > 0.01 * _n:
        print()
        print("  >1% unmapped. Check for over-ocean events and coordinates")
        print("  outside the raster before trusting the vegetated fraction.")

    _veg = (df["LANDCOVER_VEGETATED"] == 1)
    _non = (df["LANDCOVER_VEGETATED"] == 0)
    _map = _veg | _non
    print()
    print(f"  mapped      : {int(_map.sum()):,} "
          f"({100.0 * _map.mean():.2f}%)")
    print(f"  vegetated   : {int(_veg.sum()):,} "
          f"({100.0 * _veg[_map].mean():.2f}% of mapped)")
    print(f"  NON-vegetated: {int(_non.sum()):,} "
          f"({100.0 * _non[_map].mean():.2f}% of mapped)")
    print()
    print("=" * 74)
    print("NEXT: build the new label")
    print("=" * 74)
    print("  INDUSTRIAL_SURFACE_FIRE = LANDCOVER_VEGETATED == 0")
    print("                            AND persistence >= threshold  (now that")
    print("                            rule_calibration no longer pins it to the")
    print("                            floor)")
    print("                            AND NBR drop with no recovery  (GEE,")
    print("                            cheap now)")
    print()
    print("  Then recompute the leakage partition: LANDCOVER_VEGETATED becomes")
    print("  a label INPUT, so it must be excluded from model A exactly like")
    print("  DIST_* is. Otherwise the label is reproduced again.")
    print("=" * 74)


if remaining:
    print()
    print("=" * 74)
    print("3. EXTRACTION")
    print("=" * 74)

    _image = gee.worldcover_image()
    _t0 = time.time()
    _written = 0
    _dropped = 0
    _done_shards = set()
    for _c in done_chunks:
        _done_shards.add(_c // SHARD_CHUNKS)

    def _flush(shard_no, frames):
        """Write one shard ATOMICALLY, so a kill never leaves a torn parquet.

        `shard_no` is passed in, never tracked as mutable state. An earlier
        version seeded it from `min(done_chunks) // SHARD_CHUNKS`, which is
        WRONG whenever any shard is missing: with shards 1 and 2 deleted,
        `min` was 0, so the resume wrote shards 0 and 1 -- overwriting two
        files that were already correct and mislabelling the new ones. The run
        then assembled 19,000 of 25,000 rows and said nothing. Shard identity is
        a pure function of the chunk number, so it is derived, never remembered.
        """
        if not frames:
            return
        _path = shard_path(shard_no)
        if _path.exists():
            # Refuse to clobber a shard we already validated.
            _path.unlink()
        _tmp = _path.with_suffix(".partial")
        pd.concat(frames, ignore_index=True).to_parquet(_tmp, index=False)
        _tmp.replace(_path)          # atomic on POSIX and on Drive's FUSE

    # SHARD-major, not chunk-major. Every chunk of a shard goes into one
    # buffer, so a shard is only ever written complete.
    _pending_shards = sorted(set(range(int(np.ceil(n_chunks_total
                                                    / SHARD_CHUNKS))))
                             - _done_shards)
    for _si, _sn in enumerate(_pending_shards, 1):
        _chunks = [c for c in range(_sn * SHARD_CHUNKS,
                                    min((_sn + 1) * SHARD_CHUNKS,
                                        n_chunks_total))
                   if c not in done_chunks]
        _frames = []
        for _c in _chunks:
            _lo = _c * CHUNK
            _hi = min(_lo + CHUNK, len(idx_all))
            _rows = idx_all[_lo:_hi]
            _recs = [{"row": int(r),
                      "lon": float(lon[r]), "lat": float(lat[r])}
                     for r in _rows]
            _msgs = []
            _vals = gee.sample_points_chunked(
                _image, _recs, id_key="row", scale=100,
                band=gee.WORLDCOVER_BAND, printer=_msgs.append,
                progress_every=0)
            for _m in _msgs:
                if "DROPPED" in _m or "FAILED" in _m:
                    print(f"    {_m}")
            _got = 0
            _buf = np.full(len(_rows), -1, dtype=np.int16)
            for _r in _rows:
                _v = _vals.get(int(_r))
                if _v is not None:
                    _buf[_r - _lo] = int(_v)
                    _got += 1
            _dropped += len(_rows) - _got
            _frame = pd.DataFrame({
                "row": _rows.astype(np.int64),
                "event_id": ids[_rows],
                "WORLDCOVER_CLASS": _buf,
            })
            _frame["LANDCOVER_VEGETATED"] = np.where(
                _buf >= 0,
                np.isin(_buf, sorted(gee.VEGETATED_CLASSES)).astype("int8"), -1)
            if "start_year" in _keep:
                _frame["start_year"] = meta["start_year"].to_numpy()[_rows]
            _frames.append(_frame)
            _written += len(_frame)

        _flush(_sn, _frames)
        MANIFEST.write_text(json.dumps({
            "chunks_total": n_chunks_total,
            "shards_total": int(np.ceil(n_chunks_total / SHARD_CHUNKS)),
            "shards_written_through": _sn,
            "chunk_size": CHUNK,
            "shard_chunks": SHARD_CHUNKS,
            "rows": _written,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2))

        _el = time.time() - _t0
        print(f"    shard {_sn:>3} written  ({_si}/{len(_pending_shards)})"
              f"   {_written:,} rows   {_el / 60:.1f} min elapsed")
        for _c in _chunks:
            done_chunks.add(_c)

    print(f"  extracted {_written:,} rows in {(time.time() - _t0) / 60:.1f} min")

    if _dropped:
        print(f"  WARNING: {_dropped:,} points returned no value from Earth")
        print(f"  Engine. They are stored as WORLDCOVER_CLASS = -1 and are NEVER")
        print(f"  treated as 'non-vegetated'. A missing tile must not read as")
        print(f"  bare ground.")


_finish()
