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

P7O_CELL_VERSION = "2026-10-07.1"

CHUNK = int(globals().get("GEE_CHUNK_SIZE", 5_000))
SHARD_CHUNKS = int(globals().get("SHARD_CHUNKS", 50))
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
                     "event_id") if c in _MC]
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


def by_band(img, name):
    return img.rename([name])


def probe(img, name, recs):
    return gee.sample_points_chunked(
        by_band(img, name), recs, id_key="row", scale=11132, band=name,
        printer=lambda *a: None, progress_every=0)


def _climatology(mm):
    out = {}
    for nm, coll, band in SPECS:
        mean, std = gee.s5p_monthly_climatology(mm, coll, band)
        out[nm] = (mean, std)
    return out


clim = {m: _climatology(m) for m in range(1, 13)}

# RESUME STATE -----------------------------------------------------------------
n_chunks_total = int(np.ceil(N / CHUNK))
shards_total = int(np.ceil(n_chunks_total / SHARD_CHUNKS))
done_chunks: set = set()
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
        if (len(_rows) == 0 or int(_rows.min()) < _lo
                or int(_rows.max()) >= _hi
                or len(_rows) != len(np.unique(_rows))):
            _p.unlink(missing_ok=True)
            print(f"  {_p.name}: DISCARD (out-of-band/dup)")
            continue
        done_chunks.update(range(_c0, _c1))
remaining = [c for c in range(n_chunks_total) if c not in done_chunks]
print(f"  chunks {len(remaining):,} remaining of {n_chunks_total:,}")
# MAIN LOOP -----------------------------------------------------------------
_t0 = time.time()
_written = 0
for c in remaining:
    lo = c * CHUNK
    hi = min(lo + CHUNK, N)
    rows = np.arange(lo, hi)[ok[lo:hi]]
    if len(rows) == 0:
        continue
    frames = []
    # A 5,000-row chunk can straddle a month/year boundary: group by
    # (year, month) so every event is sampled against ITS OWN month.
    groups = {}
    for r in rows:
        groups.setdefault((int(years[r]), int(months[r])), []).append(int(r))
    for (yy, mm), sub in sorted(groups.items()):
        sub = np.asarray(sub, dtype=np.int64)
        recs = [{"row": int(r), "lon": float(lon[r]), "lat": float(lat[r])}
                for r in sub]
        per = {}
        for nm, coll, band in SPECS:
            img = gee.s5p_monthly_median(yy, mm, coll, band)
            vals = probe(img, nm, recs)
            mean, std = clim[mm][nm]
            cvals = probe(mean, nm + "_m", recs)
            sval = probe(std, nm + "_s", recs)
            for rr in sub:
                rr = int(rr)
                v = vals.get(rr)
                m = cvals.get(rr)
                s = sval.get(rr)
                d = per.setdefault(rr, {})
                if v is not None:
                    d[nm] = float(v)
                if m is not None and s not in (None, 0):
                    d[nm + "_clim"] = (float(m), float(s))
        for r in sub:
            d = per.get(int(r), {})
            if not d:
                continue
            no2 = d.get("NO2")
            co = d.get("CO")
            row = {"row": int(r), "event_id": ids[r],
                   "start_year": int(years[r]), "start_month": int(mm),
                   "CO_NO2_RATIO_LOG": (np.log1p(co) - np.log1p(no2)
                                        if co is not None and no2 is not None
                                        else np.nan)}
            for nm, _c, _b in SPECS:
                v = d.get(nm)
                cl = d.get(nm + "_clim")
                row[nm + "_ANOMALY_Z"] = ((v - cl[0]) / cl[1]
                                          if (v is not None and cl and cl[1])
                                          else np.nan)
            frames.append(row)
    if frames:
        _frame = pd.DataFrame(frames).sort_values("row")
        _sn = (lo // CHUNK) // SHARD_CHUNKS
        _partial = shard_path(_sn).with_suffix(".partial")
        _frame.to_parquet(_partial, index=False)
        _partial.rename(shard_path(_sn))
        _written += len(_frame)
    if c % 20 == 0:
        print(f"  {c + 1:,}/{n_chunks_total:,}  {_written:,} rows  "
              f"{time.time() - _t0:.0f}s")

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

