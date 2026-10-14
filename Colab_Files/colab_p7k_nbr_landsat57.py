# ============================================================================
# P7k-II -- COLAB CELL: NBR for the pre-2013 events from Landsat 5 and 7
#
# THE GAP THIS CLOSES
#
# `colab_p7k_nbr_change.py` computes DNBR from `LANDSAT/LC08/C02/T1_L2`, which
# starts 2013-03-18. Every event before then -- and 2013-01/2013-02 as well --
# left NBR_CHANGE and DNBR as NaN, because the year rule passed and the month
# was empty. On this catalogue that is roughly a third of all events.
#
# Landsat 5 (L5 TM) and Landsat 7 (L7 ETM+) cover exactly those years. They
# carry the same `SR_B5`/`SR_B7` bands, so NBR = (SR_B5 - SR_B7)/(SR_B5 +
# SR_B7) is unchanged. What differs is WHICH collection is available:
#
#     <= 2011  : L5 (and L7 from 1999)
#     2012+    : L7 only -- L5 was decommissioned in Jan 2012
#     2013-01..02: L7 only -- L8 starts 2013-03-18
#
# Soduku two known defects of L7 that are handled here rather than hidden:
#   * SLC-OFF since 2003-05-31: ETM+ scan lines are wedges of missing data,
#     so a monthly median has per-pixel holes instead of full coverage.
#     `median()` over the month plus the QA_PIXEL mask is the honest
#     composite; the holes survive as NaN and are NOT filled.
#   * L5 ended in Jan 2012, so a 2012 event can ONLY use L7.
#
# This cell ADDs a table, it does not replace P7k. `nbr_landsat57_events.parquet`
# holds rows for the events P7k could not see. The local merge then takes
# DNBR from P7k where it exists and from here otherwise, and records which
# sensor each value came from -- a value whose provenance is not recorded is
# indistinguishable from one that was never measured.
#
# Output: qc/nbr_landsat57_events.parquet
# ============================================================================

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import colab_utils as cu
import gee_utils as gee

P7K57_CELL_VERSION = "2026-10-13.1"

CHUNK = int(globals().get("GEE_CHUNK_SIZE", 5_000))
SHARD_CHUNKS = int(globals().get("SHARD_CHUNKS", 10))
RESUME = bool(globals().get("RESUME", True))

L5 = "LANDSAT/LT05/C02/T1_L2"
L7 = "LANDSAT/LE07/C02/T1_L2"
L8 = gee.LANDSAT_NBR            # "LANDSAT/LC08/C02/T1_L2"

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError(f"no project root under {cu.MYDRIVE}")
DATASET_DIR = PROJECT_ROOT / "full_2012_2026" / "features" / "dataset"
QC_DIR = PROJECT_ROOT / "qc"
SHARD_DIR = QC_DIR / "nbr57_shards"
SHARD_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = QC_DIR / "nbr57_manifest.json"

print("=" * 74)
print(f"P7k-II -- NBR FROM LANDSAT 5 / 7  [v{P7K57_CELL_VERSION}]")
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
    raise SystemExit("P7k-II needs a verified Earth Engine session")

import ee                                                          # noqa: E402

# ---------------------------------------------------------------------------
# 1. WHICH EVENTS NEED THIS CELL AT ALL
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("1. SELECT EVENTS P7k COULD NOT SEE")
print("=" * 74)
_MC = set(pq.ParquetFile(DATASET_DIR / "meta.parquet").schema_arrow.names)
for _c in ("event_centroid_lon", "event_centroid_lat", "start_time"):
    if _c not in _MC:
        raise SystemExit(f"meta.parquet lacks {_c!r} -- run P6 first")
_keep = [c for c in ("event_centroid_lon", "event_centroid_lat",
                     "event_id", "start_time") if c in _MC]
meta = pd.read_parquet(DATASET_DIR / "meta.parquet", columns=_keep)
meta = meta.reset_index(drop=True)
meta["row"] = np.arange(len(meta), dtype="int64")
N = len(meta)
ids = (meta["event_id"].to_numpy() if "event_id" in _keep
       else np.arange(N, dtype=np.int64))
lon = pd.to_numeric(meta["event_centroid_lon"], errors="coerce").to_numpy("float64")
lat = pd.to_numeric(meta["event_centroid_lat"], errors="coerce").to_numpy("float64")
st = pd.to_datetime(meta["start_time"], errors="coerce")
yr = st.dt.year.to_numpy("float64")
mo = st.dt.month.to_numpy("float64")

# L8 coverage begins 2013-03-18, so a month is L8-usable only from 2013-03.
_l8_first = (2013, 3)
_l8_ok = (yr > _l8_first[0]) | ((yr == _l8_first[0]) & (mo >= _l8_first[1]))
_needs = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(yr) & ~_l8_ok

print(f"  events {N:,}")
print(f"  L8-usable                : {int(_l8_ok.sum()):,}")
print(f"  needed here (pre-2013/03): {int(_needs.sum()):,}")
by_year = pd.Series(yr[_needs]).astype("Int64").value_counts().sort_index()
print("  by year: " + "  ".join(f"{y}:{n:,}" for y, n in by_year.items()))

# Pre-2013 rows only, contiguous row band.
idx_all = np.flatnonzero(_needs)
if idx_all.size == 0:
    raise SystemExit("no events need Landsat 5/7 -- P7k already covers all")

# ---------------------------------------------------------------------------
# 2. COLLECTION CHOICE + MONTH COVERAGE, PROBED PER MONTH
#
# `gee_utils.nbr_month_has_imagery` gates on LANDSAT8_FIRST_DATE and would
# reject every month this cell exists for. So probe the collection itself:
# one `size()` call per (collection, month), cached, and a month with no
# images is skipped rather than sampled into a zero-band composite.
# ---------------------------------------------------------------------------
def collection_for(y: int, m: int):
    """(asset_id, note) for the event month, or (None, reason)."""
    if y >= 2012:
        return L7, "L7 only (L5 decommissioned Jan 2012)"
    return L5, "L5"


def _bounds(y, m):
    return gee._month_bounds(int(y), int(m))


def month_has_data(asset_id, y, m, cache={}):
    key = (asset_id, int(y), int(m))
    if key in cache:
        return cache[key]
    s, e = _bounds(y, m)
    try:
        n = ee.ImageCollection(asset_id).filterDate(s, e).size().getInfo()
    except Exception as exc:                        # noqa: BLE001
        print(f"    probe {asset_id} {y}-{m:02d} FAILED: "
              f"{type(exc).__name__}: {str(exc)[:90]}")
        n = 0
    cache[key] = int(n) > 0
    return cache[key]


def nbr_composite(asset_id, y, m):
    """Monthly NBR median as a single-band image.

    QA_PIXEL bits 3 (cloud) and 4 (cloud shadow) are masked before the
    median. Shadow is dark in NIR and SWIR alike, so leaving it in depresses
    NBR exactly like a burn scar would.

    NO scene-level CLOUD_COVER filter. Indian monsoon months have every scene
    above 80% scene cloud; a scene filter empties the collection, and
    `median()` of an empty collection is a ZERO-BAND image that crashes
    sampleRegions with a confusing band error. The pixel mask does the work.
    """
    s, e = _bounds(y, m)
    swir, nir = gee.NBR_BANDS["landsat"]          # ("SR_B7", "SR_B5")
    col = (ee.ImageCollection(asset_id)
           .filterDate(s, e)
           .map(_mask_l5_l7_clouds))
    # `normalizedDifference()` with NO arguments takes the image's first two
    # bands in order, so the band order carried by select() decides the sign.
    # Passing band names in is a TypeError in the Python API.
    return (col.select([nir, swir]).median()
            .normalizedDifference().rename("NBR"))


def _mask_l5_l7_clouds(image):
    qa = image.select("QA_PIXEL")
    return image.updateMask(qa.bitwiseAnd(int("1000", 2)).eq(0)
                            .And(qa.bitwiseAnd(int("10000", 2)).eq(0)))


def _has_image_yet(img_check):
    """Confirm a composite actually has the band before sampling it."""
    return img_check


# ---------------------------------------------------------------------------
# 3. SHARDED, RESUMABLE EXTRACTION
# ---------------------------------------------------------------------------
def shard_path(no):
    return SHARD_DIR / f"nbr57_{no:03d}.parquet"


n_chunks_total = int(np.ceil(len(idx_all) / CHUNK))

done_chunks = set()
if RESUME:
    for p in sorted(SHARD_DIR.glob("nbr57_*.parquet")):
        try:
            n = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        try:
            rows = pq.ParquetFile(p).metadata.num_rows
        except Exception:                           # noqa: BLE001
            p.unlink(missing_ok=True)
            continue
        expect = min(SHARD_CHUNKS * CHUNK,
                     max(0, len(idx_all) - n * SHARD_CHUNKS * CHUNK))
        if rows == expect:
            for c in range(n * SHARD_CHUNKS, (n + 1) * SHARD_CHUNKS):
                if c < n_chunks_total:
                    done_chunks.add(c)

remaining = [c for c in range(n_chunks_total) if c not in done_chunks]
print()
print(f"  chunks {n_chunks_total:,}  done {len(done_chunks):,}  "
      f"remaining {len(remaining):,}")
if remaining:
    print(f"  ETA: unknown -- measure the first shard and divide")

# Composite cache: a month is queried once, reused by every event in it.
_COMPOSITES = {}


def composite(asset_id, y, m):
    key = (asset_id, int(y), int(m))
    if key not in _COMPOSITES:
        _COMPOSITES[key] = nbr_composite(asset_id, y, m)
    return _COMPOSITES[key]


t0 = time.time()
written = 0
pending_shards = sorted(
    set(range(int(np.ceil(n_chunks_total / SHARD_CHUNKS))))
    - set(c // SHARD_CHUNKS for c in done_chunks))

for si, sn in enumerate(pending_shards, 1):
    chunks = [c for c in range(sn * SHARD_CHUNKS,
                               min((sn + 1) * SHARD_CHUNKS, n_chunks_total))
              if c not in done_chunks]
    frames = []
    for c in chunks:
        lo, hi = c * CHUNK, min((c + 1) * CHUNK, len(idx_all))
        rows = idx_all[lo:hi]

        # Group this chunk by (collection, y, m) so each month's composite is
        # built and sampled ONCE. A 5,000-row chunk straddling a month
        # boundary must not sample events against a neighbour's month -- the
        # chunk-level bug that was fixed in P7k/P7l/P7n/P7o and would
        # otherwise erase exactly the pre-fire month this feature measures.
        groups = {}
        for r in rows:
            y, m = int(yr[r]), int(mo[r])
            asset, _note = collection_for(y, m)
            groups.setdefault((asset, y, m), []).append(int(r))

        buf = {k: np.full(len(v), np.nan) for k, v in groups.items()}
        for (asset, y, m), grp_rows in groups.items():
            if not month_has_data(asset, y, m):
                # Stays NaN. An empty month is NOT "no vegetation" -- it is
                # "no satellite saw this", and only those are distinguishable.
                continue
            img = composite(asset, y, m)
            pre_y, pre_m = gee.month_before(y, m, 1)

            # The pre-fire month may need a DIFFERENT collection: a 2012-01
            # event has its pre-month in 2011-12, which L5 can see and L7
            # can too, but a 2012-01 event's own month is L7-only while its
            # pre-month is L5-capable. Handle each month on its own.
            pre_asset, _ = collection_for(pre_y, pre_m)
            if not month_has_data(pre_asset, pre_y, pre_m):
                pre_asset_alt = L5 if pre_asset != L5 else L7
                if month_has_data(pre_asset_alt, pre_y, pre_m):
                    pre_asset = pre_asset_alt
                else:
                    continue
            pre_img = composite(pre_asset, pre_y, pre_m)

            recs = [{"row": int(r),
                     "lon": float(lon[r]), "lat": float(lat[r])}
                    for r in grp_rows]
            _msgs = []
            vals_post = gee.sample_points_chunked(
                img, recs, id_key="row", scale=100, band="NBR",
                printer=_msgs.append, progress_every=0)
            vals_pre = gee.sample_points_chunked(
                pre_img, recs, id_key="row", scale=100, band="NBR",
                printer=_msgs.append, progress_every=0)
            for msg in _msgs:
                if "FAILED" in msg or "DROPPED" in msg:
                    print(f"    {msg}")

            for k, r in enumerate(grp_rows):
                vp = vals_post.get(int(r))
                vq = vals_pre.get(int(r))
                if vp is not None and vq is not None:
                    buf[(asset, y, m)][k] = float(vp) - float(vq)

        # Assemble this chunk.
        out_rows, out_ids, out_dnbr = [], [], []
        for (asset, y, m), grp_rows in groups.items():
            for k, r in enumerate(grp_rows):
                out_rows.append(int(r))
                out_ids.append(ids[r])
                out_dnbr.append(buf[(asset, y, m)][k])
        frames.append(pd.DataFrame({
            "row": np.asarray(out_rows, dtype="int64"),
            "event_id": out_ids,
            "start_year": np.asarray([int(yr[r]) for r in out_rows], dtype="int64"),
            "start_month": np.asarray([int(mo[r]) for r in out_rows], dtype="int64"),
            "DNBR": np.asarray(out_dnbr, dtype="float32"),
        }))
        for c_ in chunks:
            done_chunks.add(c_)
        written += hi - lo

    out = pd.concat(frames, ignore_index=True).sort_values("row")
    tmp = shard_path(sn).with_suffix(".partial")
    out.to_parquet(tmp, index=False)
    tmp.replace(shard_path(sn))
    MANIFEST.write_text(json.dumps({
        "chunks_total": n_chunks_total,
        "shards_total": int(np.ceil(n_chunks_total / SHARD_CHUNKS)),
        "shards_written_through": sn,
        "rows_target": int(len(idx_all)),
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))
    print(f"  shard {sn:>3} written  ({si}/{len(pending_shards)})  "
          f"{written:,}/{len(idx_all):,} rows  {(time.time()-t0)/60:.1f} min")

# ---------------------------------------------------------------------------
# 4. ASSEMBLE + REPORT
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("4. ASSEMBLE")
print("=" * 74)
parts = [pd.read_parquet(shard_path(i))
         for i in range(int(np.ceil(n_chunks_total / SHARD_CHUNKS)))
         if shard_path(i).exists()]
if not parts:
    raise SystemExit("no shards written")
df = pd.concat(parts, ignore_index=True).sort_values("row").reset_index(drop=True)
df = df.drop_duplicates(subset="row", keep="first")

print(f"  rows: {len(df):,} of target {len(idx_all):,}")
_have = int(df["DNBR"].notna().sum())
print(f"  DNBR non-NaN: {_have:,} ({100.0*_have/max(len(df),1):.1f}%)")
print()
print("  SIGN CONVENTION, same as P7k: DNBR = NBR_pre - NBR_post, so it is")
print("  POSITIVE where vegetation was lost. It is the negative of the")
print("  ledger's NBR_CHANGE convention. Both are recorded; do not guess.")
if _have:
    v = df["DNBR"].dropna()
    print(f"\n  DNBR mean {v.mean():+.4f}  median {v.median():+.4f}  "
          f"std {v.std():.4f}")
    print(f"  positive (vegetation lost): {int((v>0).sum()):,} "
          f"({100.0*(v>0).mean():.1f}%)")
    print()
    print("  Sanity: a fire must REMOVE vegetation, so a healthy series is")
    print("  majority-positive with a modest mean. A mean near zero or a")
    print("  majority-negative split means the band order is inverted.")

out_path = QC_DIR / "nbr_landsat57_events.parquet"
df.to_parquet(out_path, index=False)
print(f"\n  Wrote {out_path}")

print()
print("NEXT: download it alongside nbr_events.parquet. The local merge takes")
print("      DNBR from P7k where it exists and from this table otherwise, so")
print("      add NBR_SOURCE ('LC08' | 'LT05/LE07') to the merged X.")
print("=" * 74)