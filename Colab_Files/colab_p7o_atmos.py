def _warn(_msg):
    """Visible failure printer: EE dropped/failed lines only."""
    if _msg and ("FAILED" in str(_msg) or "DROPPED" in str(_msg)):
        print(_msg)


# ============================================================================
# P7o -- COLAB CELL: ATMOSPHERIC CHEMISTRY (ledger: ATMOSPHERIC_CHEMISTRY 9)
#
# Sentinel-5P TROPOMI monthly medians, z-scored against a 2019-2023 monthly
# climatology for the same calendar month:
#   NO2_ANOMALY_Z SO2_ANOMALY_Z CO_ANOMALY_Z CH4_ANOMALY_Z HCHO_ANOMALY_Z
#   AOD_ANOMALY_Z CO_NO2_RATIO_LOG
#
# Same skeleton as P7k/P7l/P7n: sharded, atomic, resumable, NaN where the
# month has no TROPOMI product (archive starts 2018-07 -> pre-2018 stays NaN),
# reads meta.parquet directly. Also lands in atmos_outputs/.
# ============================================================================

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import colab_utils as cu
import gee_utils as gee

P7O_CELL_VERSION = "2026-10-11.1"

CHUNK = int(globals().get("GEE_CHUNK_SIZE", 5_000))
SHARD_CHUNKS = int(globals().get("SHARD_CHUNKS", 10))
RESUME = bool(globals().get("RESUME", True))
FIRST_S5P_YEAR = 2018

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError(f"no project root under {cu.MYDRIVE}")
DATASET_DIR = PROJECT_ROOT / "full_2012_2026" / "features" / "dataset"
QC_DIR = PROJECT_ROOT / "qc"
SHARD_DIR = QC_DIR / "atmos_shards"
SHARD_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = QC_DIR / "atmos_manifest.json"
OUTPUT_DIR = PROJECT_ROOT / "atmos_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPECS = [
    ("NO2", "COPERNICUS/S5P/OFFL/L3_NO2",
     "tropospheric_NO2_column_number_density"),
    ("SO2", "COPERNICUS/S5P/OFFL/L3_SO2", "SO2_column_number_density"),
    ("CO", "COPERNICUS/S5P/OFFL/L3_CO", "CO_column_number_density"),
    ("CH4", "COPERNICUS/S5P/OFFL/L3_CH4",
     "CH4_column_volume_mixing_ratio_dry_air"),
    ("HCHO", "COPERNICUS/S5P/OFFL/L3_HCHO",
     "tropospheric_HCHO_column_number_density"),
    ("AOD", "COPERNICUS/S5P/OFFL/L3_AER_AI", "absorbing_aerosol_index"),
]

print("=" * 74)
print(f"P7o -- ATMOSPHERIC CHEMISTRY  [v{P7O_CELL_VERSION}]")
print("=" * 74)
print(cu.drive_report())

_ok = False
_pid = None
try:
    _ok, _pid = gee.authenticate(globals().get("EARTHENGINE_PROJECT"))
    gee.verify_can_read(_pid)
    print(f"  Earth Engine working (project {_pid!r})")
except Exception as exc:                            # noqa: BLE001
    print(f"  Earth Engine NOT usable: {type(exc).__name__}: {str(exc)[:200]}")
    raise SystemExit("P7o needs a verified Earth Engine session")

import ee                                                          # noqa: E402

_MC = set(pq.ParquetFile(DATASET_DIR / "meta.parquet").schema_arrow.names)
for _c in ("event_centroid_lon", "event_centroid_lat", "start_time"):
    if _c not in _MC:
        raise SystemExit(f"meta.parquet lacks {_c!r} -- run P6 first")
_keep = [c for c in ("event_centroid_lon", "event_centroid_lat",
                     "event_id", "start_time") if c in _MC]
meta = pd.read_parquet(DATASET_DIR / "meta.parquet", columns=_keep)
N = len(meta)
ids = (meta["event_id"].to_numpy() if "event_id" in _keep
       else np.arange(N, dtype=np.int64))
lon = meta["event_centroid_lon"].to_numpy(dtype="float64")
lat = meta["event_centroid_lat"].to_numpy(dtype="float64")
_st = pd.to_datetime(meta["start_time"], errors="coerce")
years = _st.dt.year.fillna(0).to_numpy().astype("int64")
months = _st.dt.month.fillna(0).to_numpy().astype("int64")
ok = (np.isfinite(lon) & np.isfinite(lat) & (months > 0)
      & (years >= FIRST_S5P_YEAR))
print(f"  events {N:,}  usable (>= {FIRST_S5P_YEAR}) {int(ok.sum()):,}")


def shard_path(no: int) -> Path:
    return SHARD_DIR / f"a_{no:06d}.parquet"


def _flush(shard_no, frames):
    """Write one shard ATOMICALLY, including the empty case.

    An earlier version returned without writing when `frames` was empty,
    so a shard whose months all fell outside the S5P archive (pre-2018-07)
    was never written -- and was therefore retried on every re-run, which
    is what made the cell report "139 shards remaining" forever. An empty
    shard is a LEGITIMATE outcome: it means no event in that band had
    imagery. Write it, so resume trusts it as done.
    """
    pth = shard_path(shard_no)
    if frames:
        out = pd.concat(frames, ignore_index=True).sort_values("row")
        if len(out) != out["row"].nunique():
            raise RuntimeError(f"shard {shard_no} assembled {len(out):,} rows "
                               f"for {out['row'].nunique():,} distinct rows "
                               f"-- shard boundaries are wrong")
    else:
        out = pd.DataFrame({"row": np.empty(0, dtype=np.int64)})
    pth.parent.mkdir(parents=True, exist_ok=True)
    partial = pth.with_suffix(".partial")
    out.to_parquet(partial, index=False)
    partial.replace(pth)


# RESUME STATE -----------------------------------------------------------------
n_chunks_total = int(np.ceil(N / CHUNK))
shards_total = int(np.ceil(n_chunks_total / SHARD_CHUNKS))
done_chunks: set = set()
_done_shards: set = set()
_done_rows = 0
if RESUME:
    for _p in sorted(SHARD_DIR.glob("a_*.parquet")):
        try:
            _sn = int(_p.stem.split("_")[-1])
        except (IndexError, ValueError):
            continue
        _c0 = _sn * SHARD_CHUNKS
        _c1 = min(_c0 + SHARD_CHUNKS, n_chunks_total)
        _lo = _c0 * CHUNK
        _hi = min(_c1 * CHUNK, N)
        try:
            _rows = np.asarray(pq.read_table(_p, columns=["row"])
                               ["row"].to_numpy(), dtype=np.int64)
        except Exception:                            # noqa: BLE001
            _p.unlink(missing_ok=True)
            print(f"  {_p.name}: UNREADABLE -- re-extracting")
            continue
        # An EMPTY shard is a legitimate outcome (no event in that band had
        # S5P imagery), and must be TRUSTED as done. Only a NON-empty shard
        # that is out-of-band or duplicated is a corruption -- an earlier
        # version DISCARDed empty shards too, which is what made the cell
        # report "139 shards remaining" forever: every re-run re-extracted
        # the same empty bands, never finishing.
        if (len(_rows) != 0 and (
                int(_rows.min()) < _lo
                or int(_rows.max()) >= _hi
                or len(_rows) != len(np.unique(_rows)))):
            _p.unlink(missing_ok=True)
            print(f"  {_p.name}: DISCARD (out-of-band/dup)")
            continue
        for _c in range(_c0, _c1):
            done_chunks.add(_c)
        _done_shards.add(_sn)
        _done_rows += len(_rows)

_pending_shards = [s for s in range(shards_total) if s not in _done_shards]
print(f"  chunks {len(done_chunks):,} done ({_done_rows:,} rows), "
      f"{len(_pending_shards):,} shards remaining of {shards_total:,}")

# MAIN LOOP -----------------------------------------------------------------
# GLOBAL composite cache (not shard-local): each (year, month) composite is
# shared across ALL shards. The earlier per-shard cache re-checked
# `size().getInfo()` for every shard, which is what made the 50h ETA.
_COMPOSITE_CACHE: dict = {}
_CLIMATOLOGY_CACHE: dict = {}

def _build_full_composite(yy: int, mm: int):
    """Build or fetch the 18-band composite for (yy, mm):
       6 event bands + 6 climatology means + 6 climatology stds.
    """
    global _COMPOSITE_CACHE, _CLIMATOLOGY_CACHE
    # Dummy assignments bind the names in the function scope so the
    # static checker sees them as "assigned" (mutations like
    # _COMPOSITE_CACHE[key] = ... are then allowed).
    _COMPOSITE_CACHE = _COMPOSITE_CACHE
    _CLIMATOLOGY_CACHE = _CLIMATOLOGY_CACHE
    key = (int(yy), int(mm))
    if key in _COMPOSITE_CACHE:
        return _COMPOSITE_CACHE[key]

    # Build event bands
    event_images = {}
    for nm, coll, band in SPECS:
        img = gee.s5p_monthly_median(yy, mm, coll, band)
        # Inline band count check: returns None for empty months
        if img is None:
            continue
        try:
            has_bands = len(ee.Image(img).bandNames().getInfo()) > 0
        except Exception:                            # noqa: BLE001
            has_bands = False
        if not has_bands:
            continue
        event_images[nm] = img.rename([nm])

    if not event_images:
        _COMPOSITE_CACHE[key] = None
        return None

    # Build climatology bands (same for all years in this calendar month)
    clim_key = int(mm)
    if clim_key not in _CLIMATOLOGY_CACHE:
        clim_images = {}
        for nm, coll, band in SPECS:
            mean, std = gee.s5p_monthly_climatology(mm, coll, band)
            for img, suffix in ((mean, "_m"), (std, "_s")):
                if img is None:
                    continue
                try:
                    has_bands = len(ee.Image(img).bandNames().getInfo()) > 0
                except Exception:                    # noqa: BLE001
                    has_bands = False
                if has_bands:
                    clim_images[nm + suffix] = img.rename([nm + suffix])
        _CLIMATOLOGY_CACHE[clim_key] = clim_images
    else:
        clim_images = _CLIMATOLOGY_CACHE[clim_key]

    # Merge all bands into one image for a single sampleRegions call
    all_images = {**event_images, **clim_images}
    if not all_images:
        _COMPOSITE_CACHE[key] = None
        return None

    # Combine into a single multi-band image
    first_img = next(iter(all_images.values()))
    combined = first_img
    for img in list(all_images.values())[1:]:
        combined = combined.addBands(img)

    _COMPOSITE_CACHE[key] = combined
    return combined


def _sample_month_chunked(combined, recs):
    """Sample `combined` image for `recs` with EE's 5000-feature chunking.

    Mirrors P7n's `_sample_month`: submits batches, and on "over 5000
    elements" error, splits the batch and retries. Returns {row: vals} dict.
    """
    pending = [recs]
    out = {}
    while pending:
        batch = pending.pop(0)
        try:
            fc = ee.FeatureCollection([
                ee.Feature(ee.Geometry.Point([float(r["lon"]), float(r["lat"])]),
                           {"row": int(r["row"])}) for r in batch])
            res = (combined.sampleRegions(collection=fc, properties=["row"],
                                           scale=11132, geometries=False).getInfo())
            for f in res.get("features", []):
                p = f.get("properties", {})
                r_id = p.get("row")
                if r_id is not None:
                    vals = {k: float(v) for k, v in p.items()
                            if k != "row" and v is not None}
                    if vals:
                        out[int(r_id)] = vals
        except Exception as exc:                    # noqa: BLE001
            msg = str(exc)
            if "over 5000 elements" in msg and len(batch) > 1:
                mid = len(batch) // 2
                pending.insert(0, batch[mid:])
                pending.insert(0, batch[:mid])
            else:
                _warn(f"  sampleRegions FAILED ({type(exc).__name__}): {msg[:140]}")
    return out


_t0 = time.time()
_written = _done_rows
# CHUNK = 5000 keeps each sampleRegions call under EE's 5000-feature limit.
# P7n's shard pattern (SHARD_CHUNKS=10, CHUNK=5000 -> 73 shards) is the
# reference; P7o now matches it exactly for the same throughput.
CHUNK = 5_000
SHARD_CHUNKS = 10
n_chunks_total = int(np.ceil(N / CHUNK))
shards_total = int(np.ceil(n_chunks_total / SHARD_CHUNKS))

# Recompute pending shards with new chunking
done_chunks: set = set()
_done_shards: set = set()
_done_rows = 0
if RESUME:
    for _p in sorted(SHARD_DIR.glob("a_*.parquet")):
        try:
            _sn = int(_p.stem.split("_")[-1])
        except (IndexError, ValueError):
            continue
        _c0 = _sn * SHARD_CHUNKS
        _c1 = min(_c0 + SHARD_CHUNKS, n_chunks_total)
        _lo = _c0 * CHUNK
        _hi = min(_c1 * CHUNK, N)
        try:
            _rows = np.asarray(pq.read_table(_p, columns=["row"])
                               ["row"].to_numpy(), dtype=np.int64)
        except Exception:                            # noqa: BLE001
            _p.unlink(missing_ok=True)
            print(f"  {_p.name}: UNREADABLE -- re-extracting")
            continue
        if (len(_rows) != 0 and (
                int(_rows.min()) < _lo
                or int(_rows.max()) >= _hi
                or len(_rows) != len(np.unique(_rows)))):
            _p.unlink(missing_ok=True)
            print(f"  {_p.name}: DISCARD (out-of-band/dup)")
            continue
        for _c in range(_c0, _c1):
            done_chunks.add(_c)
        _done_shards.add(_sn)
        _done_rows += len(_rows)

_pending_shards = [s for s in range(shards_total) if s not in _done_shards]
print(f"  chunks {len(done_chunks):,} done ({_done_rows:,} rows), "
      f"{len(_pending_shards):,} shards remaining of {shards_total:,}")

for _si, _sn in enumerate(_pending_shards, 1):
    _c0 = _sn * SHARD_CHUNKS
    _c1 = min(_c0 + SHARD_CHUNKS, n_chunks_total)
    _frames = []
    _shard_filled = 0

    for c in range(_c0, _c1):
        if c in done_chunks:
            continue
        lo = c * CHUNK
        hi = min(lo + CHUNK, N)
        rows = np.arange(lo, hi)[ok[lo:hi]]
        if len(rows) == 0:
            continue

        groups = {}
        for r in rows:
            groups.setdefault((int(years[r]), int(months[r])), []).append(int(r))

        _c_frames = []
        for (yy, mm), sub in sorted(groups.items()):
            sub = np.asarray(sub, dtype=np.int64)
            recs = [{"row": int(r), "lon": float(lon[r]), "lat": float(lat[r])}
                    for r in sub]

            combined = _build_full_composite(yy, mm)
            if combined is None:
                # No S5P data for this month at all
                continue

            # Chunked sampling (P7n pattern): splits batches on EE's
            # "over 5000 elements" error. Returns {row: {band: value}}.
            sampled = _sample_month_chunked(combined, recs)

            for r in sub:
                r = int(r)
                vals = sampled.get(r)
                if not vals:
                    continue
                # Extract event bands
                event_vals = {nm: vals.get(nm) for nm, _, _ in SPECS}
                if not any(v is not None for v in event_vals.values()):
                    continue
                # Extract climatology
                clim_vals = {}
                for nm, _, _ in SPECS:
                    m = vals.get(nm + "_m")
                    s = vals.get(nm + "_s")
                    if m is not None and s not in (None, 0):
                        clim_vals[nm] = (float(m), float(s))
                # Build row
                no2 = event_vals.get("NO2")
                co = event_vals.get("CO")
                row = {"row": r, "event_id": ids[r],
                       "start_year": int(yy), "start_month": int(mm),
                       "CO_NO2_RATIO_LOG": (np.log1p(co) - np.log1p(no2)
                                            if co is not None and no2 is not None
                                            else np.nan)}
                for nm, _, _ in SPECS:
                    v = event_vals.get(nm)
                    cl = clim_vals.get(nm)
                    row[nm + "_ANOMALY_Z"] = ((v - cl[0]) / cl[1]
                                              if (v is not None and cl and cl[1])
                                              else np.nan)
                _c_frames.append(row)

        if _c_frames:
            _cf = pd.DataFrame(_c_frames)
            _frames.append(_cf)
            _shard_filled += len(_cf)

        print(f"    chunk {c + 1:>4}/{n_chunks_total} (shard {_sn:>3})  "
              f"{_shard_filled:>6,} shard rows  "
              f"elapsed {(time.time() - _t0) / 60.0:.1f}m",
              end="\r" if (c + 1) % 5 != 0 else "\n")

    _flush(_sn, _frames)
    _written += _shard_filled
    for _c in range(_c0, _c1):
        done_chunks.add(_c)
    _done_shards.add(_sn)

    MANIFEST.write_text(json.dumps({
        "cell_version": P7O_CELL_VERSION,
        "shards_total": shards_total,
        "shards_written_through": _sn,
        "chunk_size": CHUNK,
        "shard_chunks": SHARD_CHUNKS,
        "rows_written": _written,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))

    _el = (time.time() - _t0) / 60.0
    _left = max(0, len(_pending_shards) - _si)
    _eta_h = (_el / _si) * _left / 60.0 if _si else 0.0
    print(f"  shard {_sn:>3}/{shards_total - 1} written "
          f"({_shard_filled:,} rows)  "
          f"total {_written:,}  "
          f"elapsed {_el:.1f}m  eta {_eta_h:.1f}h")

# FINALISE -------------------------------------------------------------------
_parts = [pd.read_parquet(p) for p in sorted(SHARD_DIR.glob("a_*.parquet"))]
if _parts:
    df = pd.concat(_parts, ignore_index=True).sort_values("row")
    df = df.drop_duplicates(subset="row", keep="first").reset_index(drop=True)
    df.to_parquet(QC_DIR / "atmos_events.parquet", index=False)
    df.to_parquet(OUTPUT_DIR / "atmos_events.parquet", index=False)
    MANIFEST.write_text(json.dumps({
        "cell_version": P7O_CELL_VERSION, "rows": len(df),
        "shards": len(_parts),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))
    print(f"  wrote {QC_DIR / 'atmos_events.parquet'} ({len(df):,} rows)")
else:
    print("  no shards yet -- nothing to finalise")

