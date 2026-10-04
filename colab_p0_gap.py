# ============================================================================
# COLAB CELL -- Phase 0 gap analysis + rule diagnostics
#
# Run this INSTEAD of P7. It answers, with numbers:
#   1. which of the 141 specified features are actually present
#   2. which are present but mostly NaN (kept -- LightGBM routes NaN natively)
#   3. which are missing entirely, and how each one can be obtained
#   4. why every event came out `no_rule` -- the blocking condition, named
#   5. why FRP_ANOMALY_VS_SITE was 99.5% NaN, and what grid fixes it
#
# Reads X.parquet / y.parquet / meta.parquet. Writes two CSVs to qc/.
# ============================================================================

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

import colab_utils as cu
import feature_gap as fg
import feature_ledger as fl
import training_design as td
import weak_labels as wl

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)

print("=" * 78)
print("PHASE 0 -- GAP ANALYSIS AND RULE DIAGNOSTICS")
print("=" * 78)
print(f"ledger {fl.LEDGER_VERSION}   spec size {len(fl.FEATURES_141)}")
print()
print("MODULE STAMPS -- if these differ from your repo, you are running a")
print("stale pasted cell, and its output is missing whatever changed.")
print("-" * 78)
for module in (fl, fg, wl, td, cu):
    try:
        size = Path(inspect.getfile(module)).stat().st_size
        print(f"  {module.__name__:<20s} {size:>9,} bytes")
    except Exception as exc:                       # noqa: BLE001
        print(f"  {module.__name__:<20s} !! {exc}")
print("-" * 78)

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError(f"no project root under {cu.MYDRIVE}")
DATASET_DIR = PROJECT_ROOT / "full_2012_2026" / "features" / "dataset"
QC_DIR = PROJECT_ROOT / "qc"
QC_DIR.mkdir(parents=True, exist_ok=True)

for name in ("X.parquet", "y.parquet", "meta.parquet"):
    if not cu.path_ok(DATASET_DIR / name):
        raise FileNotFoundError(
            f"{name} missing. Run colab_p6_dataset.py first."
        )

print()
print("loading X, y, meta ...")
X = pd.read_parquet(DATASET_DIR / "X.parquet")
y = pd.read_parquet(DATASET_DIR / "y.parquet")
meta = pd.read_parquet(DATASET_DIR / "meta.parquet")
print(f"  X {X.shape}   y {y.shape}   meta {meta.shape}")

# ===========================================================================
# 1. THE GAP
# ===========================================================================

print()
print("=" * 78)
print("1. FEATURE GAP vs THE 141-FEATURE SPEC")
print("=" * 78)

report = fg.gap_report(X)
summary = fg.summarise(report)

print(f"  PRESENT (usable) : {summary['n_usable']:>4}")
print(f"  SPARSE (>90% NaN): {summary['n_sparse']:>4}   kept -- LightGBM "
      "routes NaN natively")
print(f"  MISSING          : {summary['n_missing']:>4}")
print(f"  coverage         : {summary['coverage_frac']:>6.1%} of the spec")
print()
print("  SPARSE columns (weak signal, not broken -- do NOT impute):")
sparse = report[report["status"] == "SPARSE"].sort_values("nan_fraction")
for _, r in sparse.iterrows():
    print(f"    {r['feature']:32s} {r['nan_fraction']:>7.1%} NaN")

print()
print("  MISSING features by route to obtain them:")
for route, count in sorted(summary["obtain_for_missing"].items(),
                           key=lambda kv: -kv[1]):
    print(f"    {route:18s} {count:>4}")

# `missing_by_route` keys are (route, group) pairs.
grouped = fg.missing_by_route(report)
for route in ("DERIVE", "EXTERNAL_RASTER", "EXTERNAL_TABLE"):
    keys = sorted(k for k in grouped if k[0] == route)
    if not keys:
        continue
    print()
    print(f"  --- {route} ---")
    for route_key, group in keys:
        feats = grouped[(route_key, group)]
        print(f"    {group:24s} {len(feats):>3d}  e.g. {', '.join(feats[:3])}")

# The DERIVE list is short enough to print in full, and it is the work that
# needs no download at all -- so it is the first thing worth doing.
derive_feats = sorted(
    f for f in report[report["obtain"] == "DERIVE"]["feature"]
)
if derive_feats:
    print()
    print(f"  DERIVE in full ({len(derive_feats)} -- no downloads needed):")
    for i in range(0, len(derive_feats), 3):
        print("    " + "  ".join(f"{name:30s}"
                                for name in derive_feats[i:i + 3]))

# ===========================================================================
# 2. WHY IS EVERYTHING no_rule?
# ===========================================================================

print()
print("=" * 78)
print("2. WHY EVERY EVENT CAME OUT no_rule -- per-condition pass counts")
print("=" * 78)

if "label_weak" in y.columns:
    lab_col = "label_weak"
    print(f"  using weak labels from column '{lab_col}'")
else:
    lab_col = "label"
    print(f"  column 'label_weak' absent -- using '{lab_col}'.")
    print("  That means colab_p6_dataset.py ran an older pasted cell; the weak")
    print("  hierarchy and the missing-input diagnostics were not applied.")

labelling = X.copy()
for column in ("DIST_REFINERY_LOG", "DIST_NEAREST_FACILITY_LOG",
               "FACILITY_COUNT_5KM", "DIST_POWERPLANT_LOG",
               "DIST_PETROCHEMICAL_LOG", "DIST_STEEL_LOG",
               "DIST_CEMENT_LOG", "DIST_MINE_LOG", "AREA_GROWTH_RATE",
               "SPREAD_SPEED_MEDIAN", "DIRECTION_CONSISTENCY",
               "TIME_TO_PEAK", "FRP_PERSISTENCE", "DETECTION_COUNT_LOG",
               "INSIDE_FACILITY", "KNOWN_FLARE"):
    if column in X.columns:
        labelling[column] = X[column]

conditions = wl.condition_pass_report(labelling, wl.WIDE)
for cls in ("INDUSTRIAL_FLARE", "INDUSTRIAL_FIRE", "MINING", "AGRICULTURAL",
            "WILDFIRE"):
    block = conditions[conditions["rule"] == cls]
    print()
    print(f"  {cls}")
    print(block[["condition", "n_pass", "pass_frac"]].to_string(index=False))

print()
print("  Label distribution actually produced:")
counts = y[lab_col].value_counts()
for cls, n in counts.items():
    print(f"    {cls:22s} {n:>10,}  {100.0 * n / max(len(y), 1):>6.2f}%")

# ===========================================================================
# 3. WHY FRP_ANOMALY_VS_SITE WAS 99.5% NaN
# ===========================================================================

print()
print("=" * 78)
print("3. SITE GRID -- why the anomaly column is nearly empty")
print("=" * 78)

lon = meta["event_centroid_lon"].to_numpy()
lat = meta["event_centroid_lat"].to_numpy()
table, chosen = fg.recommend_site_deg(lon, lat)
print(table.to_string(index=False))
print()
import dataset_builder as db
print(f"  currently used: SITE_DEG = {db.SITE_DEG} deg")
print(f"  recommended   : {chosen} deg")
if chosen != db.SITE_DEG:
    print(f"  -> change SITE_DEG in dataset_builder.py to {chosen} and re-run P6")
    print("     to populate FRP_ANOMALY_VS_SITE")
print()
if "FRP_ANOMALY_VS_SITE" in X.columns:
    print(f"  FRP_ANOMALY_VS_SITE populated: "
          f"{X['FRP_ANOMALY_VS_SITE'].notna().mean():.2%}")
print("  A per-site baseline needs company: a cell holding exactly one event")
print("  has nothing to compare against, so the ratio is undefined by design.")
print("  Coarsen the grid, or define sites by proximity to a facility.")

# ===========================================================================
# 4. WHAT MODEL B CAN STILL USE
# ===========================================================================

print()
print("=" * 78)
print("4. LEAKAGE PARTITION -- what the scientific model may use")
print("=" * 78)
model_a, model_b, leakage = td.split_feature_sets(list(X.columns))
print(f"  model A (operational) : {len(model_a):>3} features")
print(f"  model B (scientific) : {len(model_b):>3} features")
print(f"  excluded as label inputs: "
      f"{leakage['n_label_derived_in_a']} industrial + "
      f"{leakage['n_weak_rule_inputs_in_a']} weak-rule")
print()
print("  model B keeps:")
for name in model_b:
    print(f"    {name}")
if len(model_b) <= 12:
    print()
    print("  !! Model B is nearly empty. A result from it would be thin evidence,")
    print("     and the honest headline stays model A plus an independent")
    print("     hand-labelled sample.")

# ===========================================================================
# 5. WRITE
# ===========================================================================

gap_path = QC_DIR / "p0_feature_gap.csv"
cond_path = QC_DIR / "p0_rule_conditions.csv"
site_path = QC_DIR / "p0_site_occupancy.csv"
report.to_csv(gap_path, index=False)
conditions.to_csv(cond_path, index=False)
table.to_csv(site_path, index=False)

print()
print(f"Wrote {gap_path}")
print(f"Wrote {cond_path}")
print(f"Wrote {site_path}")
print()
print("NEXT, in this order")
print("  1. Implement the DERIVE rows first -- no downloads, and they are the")
print("     features the rule engine is waiting on.")
print("  2. Join a real flare catalogue (VIIRS Nightfire / GGFR). That is what")
print("     makes KNOWN_FLARE non-null and gives INDUSTRIAL_FLARE a seed.")
print("  3. One Earth Engine pass for land cover: it unblocks the AGRICULTURAL")
print("     and WILDFIRE rules, which currently have no discriminating input.")
print("  4. Re-run P6, then P7.")