# ============================================================================
# P9 -- INFERENCE-SAFE SITE BASELINES
#
# PROBLEM (audit finding). `add_site_baseline` in dataset_builder.py
# computes SITE_FRP_BASELINE and SITE_N_EVENTS with
# `grouped.transform("median")` / `("count")` over EVERY event in the
# 1-degree cell -- past AND future. That is not inference-safe: at the
# moment an event occurs, the future events in its cell do not exist
# yet, so the feature cannot be computed at deployment. It also means
# a training event's features encode test-period events.
#
# The recurrence features (colab_p3g_recurrence.py) already do this
# right: they count only STRICTLY PAST events. The site baseline
# should too.
#
# FIX. Recompute the site baseline as a PAST-ONLY, time-aware
# statistic:
#   SITE_N_EVENTS_PAST        count of prior events in the cell
#   SITE_FRP_BASELINE_PAST    median FRP of prior events in the cell
#   FRP_ANOMALY_VS_SITE_PAST  FRP_MEDIAN / SITE_FRP_BASELINE_PAST
# A first-ever event in a cell has no prior history -> NaN (honest
# "no baseline yet"), never a fabricated value.
#
# These are ADDED as new columns; the leaky originals are dropped from
# Model B in local_retrain.py. The recurrence features are untouched
# (already past-only).
#
# Run:  python3 inference_safe_site_baseline.py
# ============================================================================
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, "/Users/aashutosh/konsors_sih2/konsors-sih2")

FEATURES_DIR = Path("/Users/aashutosh/konsors_sih2/konsors-sih2/Features_mapped")
SITE_DEG = 1.0

print("=" * 74)
print("P9 -- INFERENCE-SAFE SITE BASELINES (past-only)")
print("=" * 74)

# ---------------------------------------------------------------------------
# 1. Load the three columns this needs: coordinates, time, FRP
# ---------------------------------------------------------------------------
meta = pd.read_parquet(FEATURES_DIR / "meta.parquet",
                       columns=["event_centroid_lon", "event_centroid_lat",
                                "start_time"])
meta = meta.reset_index().rename(columns={"index": "row"})

X_pf = pq.ParquetFile(FEATURES_DIR / "X_with_p7_full.parquet")
frp = X_pf.read(columns=["row", "FRP_MEDIAN"]).to_pandas()

df = meta.merge(frp, on="row", how="left")
print(f"\n1. LOADED {len(df):,} events")

lon = pd.to_numeric(df["event_centroid_lon"], errors="coerce")
lat = pd.to_numeric(df["event_centroid_lat"], errors="coerce")
t = pd.to_datetime(df["start_time"], errors="coerce")
frp_med = pd.to_numeric(df["FRP_MEDIAN"], errors="coerce")

# 1-degree site id (same rule as dataset_builder.add_site_baseline)
site_x = np.floor(lon / SITE_DEG)
site_y = np.floor(lat / SITE_DEG)
site_id = (site_x.fillna(-999).astype("int64").astype(str)
           + "_" + site_y.fillna(-999).astype("int64").astype(str))
df["site_id"] = site_id
df.loc[lon.isna() | lat.isna(), "site_id"] = "unknown"

# Day number for time ordering (nanoseconds since epoch -> days)
day_ns = t.dt.floor("D").to_numpy("datetime64[ns]").astype("int64")
day_num = (day_ns // 86_400_000_000_000).astype("int64")
df["day_num"] = day_num
df["frp"] = frp_med

usable = np.isfinite(lon.to_numpy()) & np.isfinite(lat.to_numpy()) & t.notna().to_numpy()
print(f"   usable (coord+time): {int(usable.sum()):,}  "
      f"unusable -> NaN: {int((~usable).sum()):,}")

# ---------------------------------------------------------------------------
# 2. Past-only site statistics
#
# Within each site, sort by (day, row). For each event, the prior
# events are those with a strictly smaller (day, row) -- so an event
# never counts itself, and same-day events are ordered by row for
# determinism. The running median/count come from a cumulative
# structure, so this is O(N log N), not O(N^2).
# ---------------------------------------------------------------------------
print("\n2. COMPUTING past-only site statistics...")

work = df[["row", "site_id", "day_num", "frp"]].copy()
# Sort key: site, then time, then row (deterministic tie-break).
work = work.sort_values(["site_id", "day_num", "row"], kind="mergesort")

# Cumulative count of prior events in the site (0 for the first).
work["SITE_N_EVENTS_PAST"] = work.groupby("site_id").cumcount()

# Cumulative median of PRIOR FRP values. `expanding().median()` at
# row i is the median of rows 0..i; shifting by one WITHIN the
# site gives the median of rows 0..i-1, which is exactly the
# prior-event median. The shift must be inside the groupby -- a
# global shift would leak one site's median into the next site's
# first row. This is O(N log N) and vectorised, unlike a per-row
# insertion loop which is O(N^2) on the ~17k-event sites.
work["_exp_med"] = work.groupby("site_id")["frp"].transform(
    lambda s: s.expanding().median())
work["SITE_FRP_BASELINE_PAST"] = work.groupby("site_id")["_exp_med"].shift(1)
work.drop(columns=["_exp_med"], inplace=True)

# ---------------------------------------------------------------------------
# 3. Derive the anomaly ratio and log form
# ---------------------------------------------------------------------------
work["FRP_ANOMALY_VS_SITE_PAST"] = np.where(
    work["SITE_N_EVENTS_PAST"] <= 0,
    np.nan,
    work["frp"] / work["SITE_FRP_BASELINE_PAST"].replace(0.0, np.nan),
)
# log10 ratio; ratio exactly 1.0 (single-event site) is NaN, matching
# the existing log_anomaly convention. Guard the log against zero.
_ratio = work["FRP_ANOMALY_VS_SITE_PAST"]
work["FRP_ANOMALY_LOG_PAST"] = np.where(
    _ratio.notna() & (_ratio > 0),
    np.log10(_ratio.where(_ratio > 0)),
    np.nan,
)

# ---------------------------------------------------------------------------
# 4. Join back to the full feature matrix by row
# ---------------------------------------------------------------------------
print("\n3. JOINING back to feature matrix...")
new_cols = ["SITE_N_EVENTS_PAST", "SITE_FRP_BASELINE_PAST",
            "FRP_ANOMALY_VS_SITE_PAST", "FRP_ANOMALY_LOG_PAST"]
add = work[["row"] + new_cols].sort_values("row")

# Read the full matrix in row groups and append the new columns.
out_path = FEATURES_DIR / "X_with_p7_safe.parquet"
writer = None
total = 0
for rg in range(X_pf.num_row_groups):
    chunk = X_pf.read_row_group(rg).to_pandas()
    chunk = chunk.merge(add, on="row", how="left")
    table = pa.Table.from_pandas(chunk, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(out_path, table.schema)
    writer.write_table(table)
    total += len(chunk)
if writer is not None:
    writer.close()

print(f"   wrote {out_path}  ({total:,} rows)")

# ---------------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------------
print("\n4. PAST-ONLY SITE STATISTICS (vs leaky full-period versions)")
for c in new_cols:
    v = add[c].dropna().to_numpy()
    if v.size == 0:
        print(f"   {c:<28} all NaN")
        continue
    print(f"   {c:<28} mean={v.mean():.3f}  median={np.median(v):.3f}  "
          f"max={v.max():.3f}  coverage={100.0*len(v)/len(add):.1f}%")

# Compare to the leaky originals
for c in ["SITE_N_EVENTS", "SITE_FRP_BASELINE", "FRP_ANOMALY_VS_SITE"]:
    if c in X_pf.schema_arrow.names:
        v = X_pf.read(columns=[c]).to_pandas()[c].dropna().to_numpy()
        print(f"   [leaky] {c:<22} mean={v.mean():.3f}  "
              f"median={np.median(v):.3f}  max={v.max():.3f}  "
              f"coverage={100.0*len(v)/total:.1f}%")

print("\n" + "=" * 74)
print("DONE. Next: drop SITE_N_EVENTS, SITE_FRP_BASELINE,")
print("FRP_ANOMALY_VS_SITE, FRP_ANOMALY_LOG from Model B in")
print("local_retrain.py and use the *_PAST versions instead.")
print("=" * 74)
