# ============================================================================
# P7c -- COLAB CELL: draw the independent gold set
#
# WHY THIS CELL EXISTS
#
# Measured on the full catalogue (P7b, 2026-10-04):
#
#     max distance-to-nearest-facility among 1,119,254 weak positives = 5,000 m
#     positives beyond 5,000 m                                      = 2
#     PR-AUC of DISTANCE ALONE                                      = 0.9811
#     PR-AUC of the honest 29-feature model B                       = 0.3353
#
# So the weak INDUSTRIAL_FIRE label is a 5 km proximity cut with a redundant
# second branch that says the same thing, and every number the pipeline can
# currently produce is uninformative about whether any of it has to do with
# INDUSTRIAL fires rather than with proximity to a gazetteer entry.
#
# This cell draws the sample that fixes that: ~300 events, stratified onto the
# decision boundary, for a human to label from independent evidence. This is
# mandated by HANDOFF section 8 and it is the ONLY route available without a
# download -- India_Firms has no FIRMS `type` column, so there is no on-disk
# independent cause signal.
#
# It LABELS NOTHING. `gold_label` is written empty on purpose. Any code that
# filled that column would be reintroducing the exact circularity the gold set
# exists to remove.
#
# WRITES: qc/gold_set_sample.csv  (the work list)
#         qc/gold_set_sample.geojson (for a map / QGIS)
# ============================================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import colab_utils as cu
import gold_set as gs
import training_design as td

#: See colab_bootstrap.CELL_VERSION_FLOORS. Bump on every change so the
#: bootstrap can tell a current cell from an old one -- an unversioned
#: cell cannot be checked, and an uncheckable cell is what cost five runs.
P7C_CELL_VERSION = "2026-10-04.1"

print("=" * 74)
print(f"P7c -- draw the gold sample [v{P7C_CELL_VERSION}]")
print("=" * 74)
print(cu.drive_report())

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError(f"no project root under {cu.MYDRIVE}")
DATASET_DIR = PROJECT_ROOT / "full_2012_2026" / "features" / "dataset"
QC_DIR = PROJECT_ROOT / "qc"
QC_DIR.mkdir(parents=True, exist_ok=True)

TARGET = globals().get("TARGET", "INDUSTRIAL_FIRE")
N_TOTAL = int(globals().get("N_TOTAL", 300))
BOUNDARY_SHARE = float(globals().get("BOUNDARY_SHARE", 0.50))
SEED = int(globals().get("SEED", 0))
print(f"target {TARGET}   n {N_TOTAL}   boundary share {BOUNDARY_SHARE}   "
      f"seed {SEED}")

# ---------------------------------------------------------------------------
# 1. LOAD only what the sampler needs
#
# Distance, the weak label, the split block and the coordinates. No X read at
# all: `DIST_NEAREST_FACILITY_LOG` lives in X, so exactly one column of that
# table is streamed, in row groups. 7.27M rows, one float column.
# ---------------------------------------------------------------------------

_META_COLS = set(pq.ParquetFile(DATASET_DIR / "meta.parquet").schema_arrow.names)
_X_COLS = list(pq.ParquetFile(DATASET_DIR / "X.parquet").schema_arrow.names)

_NEEDED_META = [
    c for c in ("event_id", "event_centroid_lon", "event_centroid_lat",
                "start_time", "start_year")
    if c in _META_COLS
]
meta = pd.read_parquet(DATASET_DIR / "meta.parquet", columns=_NEEDED_META)

y_cols = set(pq.ParquetFile(DATASET_DIR / "y.parquet").schema_arrow.names)
y = pd.read_parquet(
    DATASET_DIR / "y.parquet",
    columns=[c for c in ("label_weak",) if c in y_cols],
)
labels = y["label_weak"].to_numpy()

if "DIST_NEAREST_FACILITY_LOG" not in _X_COLS:
    raise SystemExit(
        "X.parquet has no DIST_NEAREST_FACILITY_LOG. The gold set is "
        "stratified on distance to the nearest facility, so without that "
        "column it cannot be drawn. Re-run P6."
    )

print()
print(f"meta {meta.shape}   y {y.shape}   streaming 1 X column ...")
dist_log = np.full(len(meta), np.nan, dtype="float64")
_offset = 0
for _batch in pq.ParquetFile(DATASET_DIR / "X.parquet").iter_batches(
    batch_size=500_000, columns=["DIST_NEAREST_FACILITY_LOG"]
):
    _n = _batch.num_rows
    # `Array.to_numpy(zeros_on_null=...)` is not a pyarrow kwarg -- it raised
    # TypeError mid-stream and killed the cell. `to_pandas()` is unambiguous
    # about nulls (they become NaN, which is what "could not measure" means
    # here) and costs nothing at this batch size.
    dist_log[_offset:_offset + _n] = (
        _batch.column(0).to_pandas().to_numpy(dtype="float64", na_value=np.nan)
    )
    _offset += _n
    if _offset % 2_000_000 < _n:
        print(f"    {_offset:,} / {len(meta):,}")
assert _offset == len(meta), f"streamed {_offset} rows, expected {len(meta)}"
_finite = int(np.isfinite(dist_log).sum())
print(f"  done, {_offset:,} rows ({_finite:,} with a finite distance)")
if _finite < len(meta) * 0.5:
    raise SystemExit(
        f"only {_finite:,} of {len(meta):,} rows have a distance. The gold set "
        "is stratified on it, so this cannot proceed. Check that "
        "DIST_NEAREST_FACILITY_LOG is populated before running P7c."
    )

# ---------------------------------------------------------------------------
# 2. Restrict to the HELD-OUT blocks
#
# The gold set is only useful if it scores the model on regions the model was
# not trained on. Sampling across the whole catalogue would let a labeller's
# verdict be explained by training data.
# ---------------------------------------------------------------------------
meta = td.add_split_keys(meta)
_parts = td.make_split(meta, kind=td.SPLIT_COMBINED, seed=0)
_held_blocks = sorted(_parts["test"]["spatial_block"].unique())
print()
print(f"held-out blocks: {len(_held_blocks)}  "
      f"(e.g. {_held_blocks[:6]})")

frame = meta.copy()
frame["label_weak"] = labels
frame["DIST_NEAREST_FACILITY_LOG"] = dist_log
frame = frame[frame["spatial_block"].isin(_held_blocks)].copy()
print(f"rows in held-out blocks: {len(frame):,} of {len(meta):,}")
if frame.empty:
    raise SystemExit("no rows in the held-out blocks")

# ---------------------------------------------------------------------------
# 3. Draw the sample
# ---------------------------------------------------------------------------
spec = gs.GoldSpec(
    n_total=N_TOTAL,
    boundary_share=BOUNDARY_SHARE,
    blocks=_held_blocks,
    seed=SEED,
)
sample = gs.build_sample(
    frame, spec, label_col="label_weak", target=TARGET,
)

print()
print("=" * 74)
print("SAMPLE COMPOSITION")
print("=" * 74)
print(f"  rows drawn: {len(sample)} of {len(frame):,} "
      f"({100.0 * len(sample) / len(frame):.3f}%)")
print()
print(sample["gold_bucket"].value_counts().to_string())
print()
print(f"  weak-positive rate in the sample: "
      f"{100.0 * sample['weak_positive'].mean():.1f}%")
print(f"  weak-positive rate in the population: "
      f"{100.0 * (frame['label_weak'] == TARGET).mean():.1f}%")
print("  (the sample deliberately overstates it; see the weight column below)")
_d = sample["nearest_facility_m"].dropna()
print()
print(f"  distance to nearest facility, metres:")
for _q in (0, 10, 25, 50, 75, 90, 100):
    print(f"    p{_q:<3} {_d.quantile(_q / 100.0):>10,.0f} m")

# ---------------------------------------------------------------------------
# 4. Sampling weights -- so the score can be corrected back to prevalence
#
# The sample oversamples the boundary and the weak positives. Scoring it
# unweighted and quoting the result as a field precision is exactly the quiet
# overstatement this project has produced before. These weights undo the
# stratification, per bucket, against the held-out population.
# ---------------------------------------------------------------------------
_pop = frame.copy()
_pop["_bucket"] = gs._distance_bucket(
    gs.metres_from_log(_pop["DIST_NEAREST_FACILITY_LOG"]))
_pop_share = _pop["_bucket"].value_counts(normalize=True)
_smp_share = sample["gold_bucket"].value_counts(normalize=True)
sample["weight"] = (
    sample["gold_bucket"].map(_pop_share).fillna(0.0)
    / sample["gold_bucket"].map(_smp_share).fillna(1.0)
)
print()
print("  sampling weight by bucket (population share / sample share):")
for _b in sorted(sample["gold_bucket"].unique()):
    print(f"    {_b:<18} {_smp_share.get(_b, 0.0):>7.3%} in sample -> "
          f"weight {float(sample.loc[sample.gold_bucket == _b, 'weight'].iloc[0]):.4f}")
print(f"  weights sum to {sample['weight'].sum():.1f} over "
      f"{len(sample)} rows (mean weight "
      f"{sample['weight'].mean():.4f})")

# ---------------------------------------------------------------------------
# 5. WRITE
# ---------------------------------------------------------------------------
_keep = [
    "gold_id", "gold_label", "gold_confidence", "gold_notes",
    "gold_bucket", "weight", "nearest_facility_m", "DIST_NEAREST_FACILITY_LOG",
    "weak_label", "weak_positive",
    "event_centroid_lon", "event_centroid_lat", "start_time", "start_year",
    "spatial_block", "event_id",
]
_keep = [c for c in _keep if c in sample.columns]
out = sample[_keep].copy()

# The log column is DROPPED from the work list on purpose. It is
# `weak_label`'s own input, and handing a labeller the exact threshold the
# weak rule used invites them to label toward it. `nearest_facility_m` is
# enough for them to judge plausibility, and `weak_label` stays visible only
# because it is needed to report the rule's score afterwards.
out = out.drop(columns=["DIST_NEAREST_FACILITY_LOG"], errors="ignore")

_csv = QC_DIR / "gold_set_sample.csv"
out.to_csv(_csv, index=False)
print()
print(f"Wrote {_csv}  ({len(out)} rows)")

_features = []
for _, _row in out.iterrows():
    _lon, _lat = _row.get("event_centroid_lon"), _row.get("event_centroid_lat")
    if pd.isna(_lon) or pd.isna(_lat):
        continue
    # Explicit loop, not a comprehension over `r` while the dict comprehension
    # binds `row`: the two names drifted apart and raised NameError after the
    # CSV had already been written, so the run looked half-finished.
    _features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [float(_lon), float(_lat)],
        },
        "properties": {
            k: (None if pd.isna(v) else v)
            for k, v in _row.items()
            if k not in ("event_centroid_lon", "event_centroid_lat")
        },
    })
_geo = {"type": "FeatureCollection", "features": _features}
_geo_path = QC_DIR / "gold_set_sample.geojson"
_geo_path.write_text(json.dumps(_geo, indent=1, default=str))
print(f"Wrote {_geo_path}  ({len(_features)} features)")
if len(_features) != len(out):
    print(f"  NOTE: {len(out) - len(_features)} rows had no coordinates and were")
    print("  omitted from the GeoJSON. They are still in the CSV.")

print()
print("=" * 74)
print("LABELLING PROTOCOL")
print("=" * 74)
print(gs.LABELLING_PROTOCOL)
print("Fill `gold_label` in the CSV with industrial / natural / uncertain,")
print("then come back and score it. Nothing in this pipeline can do that")
print("for you -- that is the whole point of the exercise.")
print("=" * 74)