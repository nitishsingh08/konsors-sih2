# ============================================================================
# P3g -- COLAB CELL: RECURRENCE FEATURES (4) -- vectorised self-join
#
# Ledger: RECURRENCE_RATE_1Y / _3Y / _5Y, MAX_INTER_EVENT_GAP_LOG
#
# These four are the only features that CANNOT be computed one event at a
# time: counting neighbouring events needs a pass over all 7,265,849. Cell 325
# took (event, event_clusters) only, and cells 372/387 needed all_events but
# were signature-incompatible with the dispatcher -- hence P3g stayed open.
#
# HOW THIS SOLVES IT WITHOUT A 7.27M x 7.27M JOIN
#
# A true all-pairs self-join is O(N^2) and unbuildable. The join is restricted
# to the SPATIAL CELL: recurrence at a site only means recurrence in the same
# place. Events are grouped by 1-degree cell (the project's own SITE_DEG = 1.0
# decision, README section 6), and within each cell the time series is sorted,
# so window counts come from `searchsorted` -- O(N log N) overall.
#
# HONEST LIMITS, printed with the result:
#   * A cell is ~110 km. Two distinct facilities in one cell share a
#     recurrence history. This UNDER-counts nothing but OVER-counts nothing
#     either: it measures "how often this neighbourhood burns", which is the
#     intended signal.
#   * An event does not recur with itself. Only strictly PAST events count
#     (dt > 0), so a first-ever event scores 0 rather than 1.
#   * A missing timestamp leaves NaN, never 0. Zero means "this cell really
#     has no earlier event in the window".
#
# Output: qc/recurrence_events.parquet  (row, event_id, 4 features)
# `row` is the 0..N-1 position in meta.parquet, so it joins back by position.
# ============================================================================

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import colab_utils as cu

P3G_CELL_VERSION = "2026-10-13.1"
N_SHARDS = int(globals().get("P3G_NSHARDS", 8))   # row-band shards for resumability

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError(f"no project root under {cu.MYDRIVE}")
QC_DIR = PROJECT_ROOT / "qc"
DATASET_DIR = PROJECT_ROOT / "full_2012_2026" / "features" / "dataset"
SHARD_DIR = QC_DIR / "recurrence_shards"
SHARD_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = QC_DIR / "recurrence_manifest.json"

print("=" * 74)
print(f"P3g -- RECURRENCE FEATURES  [v{P3G_CELL_VERSION}]")
print("=" * 74)
print(cu.drive_report())

# ---------------------------------------------------------------------------
# 1. THE THREE COLUMNS THIS NEEDS -- nothing else, so RAM stays flat
# ---------------------------------------------------------------------------
_MC = set(pq.ParquetFile(DATASET_DIR / "meta.parquet").schema_arrow.names)
for _c in ("event_centroid_lon", "event_centroid_lat", "start_time"):
    if _c not in _MC:
        raise SystemExit(f"meta.parquet lacks {_c!r} -- run P6 first")
_keep = [c for c in ("event_centroid_lon", "event_centroid_lat", "start_time",
                     "event_id") if c in _MC]
meta = pd.read_parquet(DATASET_DIR / "meta.parquet", columns=_keep)
N = len(meta)
ids = (meta["event_id"].to_numpy() if "event_id" in _keep
       else np.arange(N, dtype=np.int64))

lon = pd.to_numeric(meta["event_centroid_lon"], errors="coerce").to_numpy("float64")
lat = pd.to_numeric(meta["event_centroid_lat"], errors="coerce").to_numpy("float64")
t = pd.to_datetime(meta["start_time"], errors="coerce")

ok = np.isfinite(lon) & np.isfinite(lat) & t.notna().to_numpy()
print(f"  events {N:,}   usable (coord+time) {int(ok.sum()):,}   "
      f"unusable {int((~ok).sum()):,} -> NaN, never 0")

# ---------------------------------------------------------------------------
# 2. THE OUTPUT SLOT ARRAY -- unusable rows stay NaN by construction
# ---------------------------------------------------------------------------
REC1 = np.full(N, np.nan, "float32")
REC3 = np.full(N, np.nan, "float32")
REC5 = np.full(N, np.nan, "float32")
GAP = np.full(N, np.nan, "float32")

idx_ok = np.flatnonzero(ok)
day = t.dt.floor("D")
# Days since epoch, so window arithmetic is integer subtraction.
# `.view("int64")` was removed in pandas 3 -- use the underlying numpy array,
# which is nanoseconds since epoch for a datetime64 column.
_day_ns = day.to_numpy("datetime64[ns]").astype("int64")
day_num = (_day_ns // 86_400_000_000_000).astype("int64")
# 1-degree cell ids. lon+180 / lat+90 keeps every cell id unique and positive.
cell_id = ((np.floor(lat).astype("int64") + 90) * 360
           + (np.floor(lon).astype("int64") + 180))

# ---------------------------------------------------------------------------
# 3. RESUME
# ---------------------------------------------------------------------------
def shard_path(n): return SHARD_DIR / f"rec_{n:03d}.parquet"

done = set()
for p in sorted(SHARD_DIR.glob("rec_*.parquet")):
    try:
        n = int(p.stem.split("_")[1])
    except (IndexError, ValueError):
        continue
    try:
        if pq.ParquetFile(p).metadata.num_rows > 0:
            done.add(n)
    except Exception:                                       # noqa: BLE001
        p.unlink(missing_ok=True)

bands = np.array_split(idx_ok, N_SHARDS)
remaining = [(i, b) for i, b in enumerate(bands) if i not in done]
print(f"  shards {len(bands)}   done {len(done)}   remaining {len(remaining)}")
if not remaining:
    print("  every shard already on Drive -- assemble only")

# ---------------------------------------------------------------------------
# 4. THE COMPUTATION, per row band
#
# `band` is the row range this shard owns. Its events are APPENDED to the
# global (cell_id, day) order, then the window counts come from searchsorted
# over the WHOLE cell history -- so shard boundaries never lose a neighbour.
# ---------------------------------------------------------------------------
t0 = time.time()
for shard_no, band in remaining:
    lo, hi = (int(band[0]) if len(band) else 0), (int(band[-1]) + 1 if len(band) else 0)
    part_cell = cell_id[lo:hi]
    part_day = day_num[lo:hi]

    # Full history for every cell this band touches.
    cells_needed = np.unique(part_cell)
    mask = np.isin(cell_id, cells_needed)
    hist_cell = cell_id[mask]
    hist_day = day_num[mask]

    frame = pd.DataFrame({"cell": hist_cell, "day": hist_day})
    frame = frame.sort_values(["cell", "day"], kind="mergesort")
    # Within a cell the times are ascending -> contiguous blocks.
    starts = frame["cell"].searchsorted(cells_needed, side="left")
    ends = frame["cell"].searchsorted(cells_needed, side="right")
    day_sorted = frame["day"].to_numpy()

    for k, cid in enumerate(cells_needed):
        days = day_sorted[starts[k]:ends[k]]
        if days.size == 0:
            continue
        days = days[days > 0]                      # epoch-day 0 is a sentinel
        if days.size == 0:
            continue

        # How many STRICTLY PAST events in each window: [d-W, d).
        # searchsorted(..., 'left') excludes today, so an event never
        # recurs with itself.
        lo_s = np.searchsorted(days, part_day - 365, side="left")
        hi_s = np.searchsorted(days, part_day, side="left")
        r1 = (hi_s - lo_s).astype("float32")
        lo_s = np.searchsorted(days, part_day - 1095, side="left")
        r3 = (hi_s - lo_s).astype("float32")
        lo_s = np.searchsorted(days, part_day - 1825, side="left")
        r5 = (hi_s - lo_s).astype("float32")

        # Max inter-event gap seen SO FAR in this cell's history.
        if days.size >= 2:
            gaps = np.diff(days).astype("float64")
            cum = np.maximum.accumulate(gaps)
            # CLIP TO A VALID INDEX, BOTH ENDS. `np.where(mask, cum[idx], 0.0)`
            # evaluates `cum[idx]` EAGERLY, before the mask selects from it, so
            # an index of `cum.size` raises IndexError even though the mask
            # would have thrown that element away. Clipping only the low end --
            # what the first version did -- leaves the high end out of bounds
            # for any cell with fewer gaps than events, i.e. every cell whose
            # events are mostly simultaneous. `days.size >= 2` guarantees
            # `cum.size >= 1`, so this clip is always in range.
            pos = np.searchsorted(days, part_day, side="left") - 1
            idx = np.clip(pos, 0, cum.size - 1)
            g = np.where((pos >= 0) & (pos < cum.size), cum[idx], 0.0)
            g = np.log1p(g).astype("float32")
        else:
            g = np.full(part_day.size, np.nan, "float32")

        sel = part_cell == cid
        REC1[lo:hi][sel] = r1[sel]
        REC3[lo:hi][sel] = r3[sel]
        REC5[lo:hi][sel] = r5[sel]
        GAP[lo:hi][sel] = g[sel]

    out = pd.DataFrame({
        "row": band.astype("int64"),
        "event_id": ids[band],
        "RECURRENCE_RATE_1Y": REC1[lo:hi],
        "RECURRENCE_RATE_3Y": REC3[lo:hi],
        "RECURRENCE_RATE_5Y": REC5[lo:hi],
        "MAX_INTER_EVENT_GAP_LOG": GAP[lo:hi],
    })
    tmp = shard_path(shard_no).with_suffix(".partial")
    out.to_parquet(tmp, index=False)
    tmp.replace(shard_path(shard_no))          # atomic

    MANIFEST.write_text(json.dumps({
        "shards_total": len(bands),
        "shards_written_through": shard_no,
        "rows": int(bands[0][-1] + 1 if len(bands[0]) else 0),
        "elapsed_s": round(time.time() - t0, 1),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))
    print(f"    shard {shard_no:>3} written  rows {lo:,}..{hi-1:,}  "
          f"{(time.time()-t0)/60:.1f} min")

# ---------------------------------------------------------------------------
# 5. ASSEMBLE + VERIFY COMPLETENESS LOUDLY
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("ASSEMBLE")
print("=" * 74)
parts = [pd.read_parquet(shard_path(i)) for i in range(len(bands))
         if shard_path(i).exists()]
if not parts:
    raise SystemExit("no shards written")
df = pd.concat(parts, ignore_index=True).sort_values("row").reset_index(drop=True)

_expected = int(ok.sum())
_have = int(df["RECURRENCE_RATE_1Y"].notna().sum())
if _have < _expected:
    print(f"  INCOMPLETE: {_have:,} of {_expected:,} computable rows "
          f"({_expected-_have:,} missing). Re-run -- it resumes.")
else:
    print(f"  complete: {_have:,} of {_expected:,} computable rows")

out_path = QC_DIR / "recurrence_events.parquet"
df.to_parquet(out_path, index=False)
print(f"  Wrote {out_path}  ({len(df):,} rows)")

print()
print(f"  {'feature':<28} {'mean':>8} {'median':>8} {'p90':>8} {'p999':>8} "
      f"{'zero%':>7}")
for c in ("RECURRENCE_RATE_1Y", "RECURRENCE_RATE_3Y", "RECURRENCE_RATE_5Y",
          "MAX_INTER_EVENT_GAP_LOG"):
    v = df[c].dropna().to_numpy()
    if v.size == 0:
        continue
    print(f"  {c:<28} {v.mean():>8.2f} {np.median(v):>8.2f} "
          f"{np.percentile(v,90):>8.2f} {np.percentile(v,99.9):>8.2f} "
          f"{100.0*(v==0).mean():>6.1f}%")

print()
print("  What to look for: RECURRENCE_RATE_1Y should be heavily zero --")
print("  most cells burn once. The events that DO have neighbours are the")
print("  persistent industrial sites the label needs to separate.")
print()
print("NEXT: download qc/recurrence_events.parquet, then re-run the local merge")
print("      and retrain (it joins on `row` like the other P7 blocks).")
print("=" * 74)