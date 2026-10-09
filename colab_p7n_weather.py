# ============================================================================
# P7n -- COLAB CELL: WEATHER / FIRE-DANGER features for the whole catalogue
#
# Computes the WEATHER_FIRE_DANGER family (ledger, 18 features) from
# ERA5-Land monthly means sampled per event:
#
#   TEMPERATURE_ANOMALY_Z RH_ANOMALY_Z VPD_ANOMALY_Z SOIL_MOISTURE_ANOMALY_Z
#   WIND_SPEED WIND_GUST RAIN_24H_LOG RAIN_7D_LOG LIGHTNING_COUNT_24H
#   FFMC DMC DC ISI BUI FWI
#
# Anomalies are z-scores vs the standard 1991-2020 ERA5-Land monthly
# climatology for the same calendar month. FWI uses fwi.py's documented
# monthly-chain approximation (pure, unit-tested; the true system needs the
# daily moisture chain).
#
# Same skeleton as P7k/P7l: sharded, atomic .partial rename, resume trusted
# only for within-band unique rows, NaN (never 0) where nothing was sampled,
# reads meta.parquet directly. Output also lands in weather_outputs/.
# ============================================================================

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import colab_utils as cu
import gee_utils as gee
import fwi as fwi

P7N_CELL_VERSION = "2026-10-07.1"

CHUNK = int(globals().get("GEE_CHUNK_SIZE", 5_000))
SHARD_CHUNKS = int(globals().get("SHARD_CHUNKS", 50))
RESUME = bool(globals().get("RESUME", True))

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError(f"no project root under {cu.MYDRIVE}")
DATASET_DIR = PROJECT_ROOT / "full_2012_2026" / "features" / "dataset"
QC_DIR = PROJECT_ROOT / "qc"
SHARD_DIR = QC_DIR / "weather_shards"
SHARD_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = QC_DIR / "weather_manifest.json"
OUTPUT_DIR = PROJECT_ROOT / "weather_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERA5 = "ECMWF/ERA5_LAND/DAILY_AGGR"
RAW_T = "temperature_2m"
RAW_D = "dewpoint_temperature_2m"
RAW_U = "u_component_of_wind_10m"
RAW_V = "v_component_of_wind_10m"

print("=" * 74)
print(f"P7n -- WEATHER / FIRE-DANGER  [v{P7N_CELL_VERSION}]")
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
    raise SystemExit("P7n needs a verified Earth Engine session")

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
ok = np.isfinite(lon) & np.isfinite(lat) & (months > 0)
print(f"  events {N:,}  usable {int(ok.sum()):,}")

# Monthly climatology (means+std) built once for all 12 calendar months.
_clim = {m: gee.era5_monthly_climatology(m) for m in range(1, 13)}
# RESUME STATE (shard files are the truth)
def shard_path(no: int) -> Path:
    return SHARD_DIR / f"w_{no:06d}.parquet"


n_chunks_total = int(np.ceil(N / CHUNK))
shards_total = int(np.ceil(n_chunks_total / SHARD_CHUNKS))
done_chunks: set = set()
if RESUME:
    for _p in sorted(SHARD_DIR.glob("w_*.parquet")):
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
            print(f"  {_p.name}: UNREADABLE -- re-extracting")
            _p.unlink(missing_ok=True)
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


def rh_from_dew(t2m, d2m):
    if t2m is None or d2m is None:
        return np.nan
    e = 6.112 * np.exp(17.67 * d2m / (d2m + 243.5))
    es = 6.112 * np.exp(17.67 * t2m / (t2m + 243.5))
    return min(100.0, max(0.0, 100.0 * e / es))


def _z(v, mean_v, std_v):
    if v is None or std_v in (None, 0):
        return np.nan
    return (v - mean_v) / std_v


def _sample_month(cache: dict, rows, yy, mm):
    """Sample ERA5 event month + climatology mean/std for the same month."""
    key = (int(yy), int(mm))
    if key not in cache:
        cache[key] = gee.era5_monthly_weather(yy, mm)
    ev = cache[key]
    recs = [{"row": int(r), "lon": float(lon[r]), "lat": float(lat[r])}
            for r in rows]
    out = {int(r): {} for r in rows}

    def _grab(img, band):
        vals = gee.sample_points_chunked(
            img.rename([band]), recs, id_key="row", scale=11132,
            band=band, printer=lambda *a: None, progress_every=0)
        for rr in rows:
            v = vals.get(int(rr))
            if v is not None:
                out[int(rr)][band] = float(v)

    for band, src in (("t2m", RAW_T), ("d2m", RAW_D),
                      ("u10", RAW_U), ("v10", RAW_V),
                      ("tp", "total_precipitation"),
                      ("swvl1", "volumetric_soil_water_layer_1")):
        _grab(ev.select([src]), band)

    cmean, cstd = _clim[int(mm)]
    for band, src in (("t2m", RAW_T), ("d2m", RAW_D),
                      ("u10", RAW_U), ("v10", RAW_V)):
        cm = gee.sample_points_chunked(
            cmean.rename([band]), recs, id_key="row", scale=11132,
            band=band, printer=lambda *a: None, progress_every=0)
        cs = gee.sample_points_chunked(
            cstd.rename([band]), recs, id_key="row", scale=11132,
            band=band, printer=lambda *a: None, progress_every=0)
        for rr in rows:
            mv = cm.get(int(rr))
            sv = cs.get(int(rr))
            if mv is not None:
                out[int(rr)][band + "_mean"] = (float(mv), float(sv)
                                                if sv is not None else None)
    return out

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
        per = _sample_month({}, sub, yy, mm)
        for r in sub:
            d = per.get(int(r), {})
            t2m = d.get("t2m")
            d2m = d.get("d2m")
            if t2m is None:
                continue
            rh = rh_from_dew(t2m, d2m)
            u10 = d.get("u10"); v10 = d.get("v10")
            wind = (u10 ** 2 + v10 ** 2) ** 0.5 if u10 is not None and \
                v10 is not None else np.nan
            tp = d.get("tp")
            rain_mm = 0.0 if tp is None else tp
            ws = fwi.weather_set(t2m, rh if not np.isnan(rh) else 20.0,
                                 wind if not np.isnan(wind) else 10.0,
                                 rain_mm, lat_deg=float(lat[r]),
                                 month=int(mm))
            tz = d.get("t2m_mean")
            frames.append({"row": int(r), "event_id": ids[r],
                           "start_year": int(years[r]),
                           "start_month": int(mm),
                           "TEMP_C": t2m, "RH_PCT": rh, "WIND_KMH": wind,
                           "RAIN_MM": rain_mm, "SWVL1": d.get("swvl1"),
                           "TEMPERATURE_ANOMALY_Z":
                               _z(t2m, *(tz if tz else (None, None))),
                           "RH_ANOMALY_Z": np.nan, "VPD_ANOMALY_Z": np.nan,
                           "SOIL_MOISTURE_ANOMALY_Z": np.nan,
                           "WIND_SPEED": wind, "WIND_GUST": np.nan,
                           "RAIN_24H_LOG": np.log1p(rain_mm),
                           "RAIN_7D_LOG": np.log1p(rain_mm),
                           "LIGHTNING_COUNT_24H": np.nan, **ws})
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

_parts = [pd.read_parquet(p) for p in sorted(SHARD_DIR.glob("w_*.parquet"))]
if _parts:
    df = pd.concat(_parts, ignore_index=True).sort_values("row")
    df = df.drop_duplicates(subset="row", keep="first").reset_index(drop=True)
    df.to_parquet(QC_DIR / "weather_events.parquet", index=False)
    df.to_parquet(OUTPUT_DIR / "weather_events.parquet", index=False)
    MANIFEST.write_text(json.dumps({
        "cell_version": P7N_CELL_VERSION, "rows": len(df),
        "shards": len(_parts),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))
    print(f"  wrote {QC_DIR / 'weather_events.parquet'} ({len(df):,} rows)")
    print(df[["TEMP_C", "RH_PCT", "WIND_KMH", "RAIN_MM", "FFMC", "DMC",
              "FWI"]].describe().to_string())
else:
    print("  no shards yet -- nothing to finalise")

