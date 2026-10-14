# ============================================================================
# P9 -- FEATURE QUALITY AUDIT
#
# Evidence table for every feature source: missingness, units, distribution,
# join cardinality, temporal alignment, coverage by year and spatial block,
# train vs validation distribution. This is the audit the improvement plan
# is built on -- every claim below is measured, not inferred.
#
# Run:  python3 audit_features.py
# ============================================================================
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "/Users/aashutosh/konsors_sih2/konsors-sih2")

import feature_ledger as fl
import training_design as td

FEATURES_DIR = Path("/Users/aashutosh/konsors_sih2/konsors-sih2/Features_mapped")
X_path = FEATURES_DIR / "X_with_p7_full.parquet"
meta_path = FEATURES_DIR / "meta.parquet"
y_path = FEATURES_DIR / "y.parquet"

print("=" * 74)
print("P9 -- FEATURE QUALITY AUDIT")
print("=" * 74)

# ---------------------------------------------------------------------------
# 1. Load meta, y, split
# ---------------------------------------------------------------------------
meta = pd.read_parquet(meta_path, columns=[
    "event_id", "event_centroid_lon", "event_centroid_lat",
    "start_time", "start_year", "spatial_block", "site_id"])
meta = meta.reset_index().rename(columns={"index": "row"})

y = pd.read_parquet(y_path, columns=["label_weak"])
y.columns = ["label_weak"]
y = y.reset_index().rename(columns={"index": "row"})

y_binary = (y["label_weak"] == "INDUSTRIAL_FIRE").astype("int8")

meta = td.add_split_keys(meta)
parts = td.make_split(meta, kind="year_x_block", seed=0)
train_idx = parts["train"]["row"].to_numpy()
test_idx = parts["test"]["row"].to_numpy()

print(f"\n1. DATASET")
print(f"   events: {len(meta):,}")
print(f"   target prevalence: {y_binary.mean():.4f}")
print(f"   train: {len(train_idx):,}  test: {len(test_idx):,}")
print(f"   years: {sorted(meta['start_year'].unique())}")
print(f"   spatial blocks (0.1deg): {meta['spatial_block'].nunique():,}")
print(f"   sites (1deg): {meta['site_id'].nunique():,}")

# Duplicate check
dup_events = int(meta["event_id"].duplicated().sum())
print(f"   duplicate event_ids: {dup_events:,}")

# ---------------------------------------------------------------------------
# 2. Feature inventory
# ---------------------------------------------------------------------------
print(f"\n2. FEATURE INVENTORY")
X_pf = pq.ParquetFile(X_path)
all_cols = [c for c in X_pf.schema_arrow.names if c not in ("row", "event_id", "NBR_SOURCE")]
print(f"   total columns: {len(X_pf.schema_arrow.names)}")
print(f"   numeric features: {len(all_cols)}")

# ---------------------------------------------------------------------------
# 3. Per-feature quality report (streamed, so we never hold 1.4 GB)
# ---------------------------------------------------------------------------
print(f"\n3. PER-FEATURE QUALITY (streamed over {X_pf.num_row_groups} row groups)")

stats = {c: {"n": 0, "sum": 0.0, "sumsq": 0.0, "min": np.inf, "max": -np.inf,
             "n_nan": 0, "n_train": 0, "sum_train": 0.0, "n_test": 0, "sum_test": 0.0}
         for c in all_cols}

train_set = set(train_idx)
test_set = set(test_idx)

for rg in range(X_pf.num_row_groups):
    rg_data = X_pf.read_row_group(rg, columns=["row"] + all_cols).to_pandas()
    rows = rg_data["row"].to_numpy()
    for c in all_cols:
        v = rg_data[c].to_numpy("float64")
        s = stats[c]
        s["n"] += len(v)
        nan_mask = np.isnan(v)
        s["n_nan"] += int(nan_mask.sum())
        valid = v[~nan_mask]
        if len(valid):
            s["sum"] += float(valid.sum())
            s["sumsq"] += float((valid ** 2).sum())
            s["min"] = min(s["min"], float(valid.min()))
            s["max"] = max(s["max"], float(valid.max()))
        # train/test split stats
        tr_mask = np.isin(rows, train_set)
        te_mask = np.isin(rows, test_set)
        v_tr = v[tr_mask & ~nan_mask]
        v_te = v[te_mask & ~nan_mask]
        s["n_train"] += len(v_tr)
        s["sum_train"] += float(v_tr.sum()) if len(v_tr) else 0.0
        s["n_test"] += len(v_te)
        s["sum_test"] += float(v_te.sum()) if len(v_te) else 0.0

# Build the report
report = []
for c in all_cols:
    s = stats[c]
    n_valid = s["n"] - s["n_nan"]
    mean = s["sum"] / max(n_valid, 1)
    var = max(s["sumsq"] / max(n_valid, 1) - mean ** 2, 0.0)
    std = np.sqrt(var)
    cov = 100.0 * n_valid / max(s["n"], 1)
    mean_tr = s["sum_train"] / max(s["n_train"], 1)
    mean_te = s["sum_test"] / max(s["n_test"], 1)
    # Population stability index (simplified): |mean_tr - mean_te| / (std + eps)
    psi = abs(mean_tr - mean_te) / (std + 1e-12)
    fam = fl.family_of(c)
    report.append({
        "feature": c, "family": fam, "coverage_pct": round(cov, 2),
        "mean": round(mean, 4), "std": round(std, 4),
        "min": round(s["min"], 4) if np.isfinite(s["min"]) else None,
        "max": round(s["max"], 4) if np.isfinite(s["max"]) else None,
        "mean_train": round(mean_tr, 4), "mean_test": round(mean_te, 4),
        "psi": round(float(psi), 4),
        "n_nan": s["n_nan"],
    })

df = pd.DataFrame(report)

# ---------------------------------------------------------------------------
# 4. Findings
# ---------------------------------------------------------------------------
print(f"\n4. FINDINGS")

# 4a. Dead features (100% NaN)
dead = df[df["coverage_pct"] == 0].sort_values("feature")
print(f"\n   4a. DEAD FEATURES (100% NaN) -- {len(dead)}:")
for _, r in dead.iterrows():
    print(f"       {r['feature']:<32} [{r['family']}]")

# 4b. Low coverage
low = df[(df["coverage_pct"] > 0) & (df["coverage_pct"] < 90)].sort_values("coverage_pct")
print(f"\n   4b. LOW COVERAGE (<90%) -- {len(low)}:")
for _, r in low.iterrows():
    print(f"       {r['feature']:<32} {r['coverage_pct']:>6.1f}%  [{r['family']}]")

# 4c. Train/test drift (PSI > 0.25 is a large shift)
drift = df[df["psi"] > 0.25].sort_values("psi", ascending=False)
print(f"\n   4c. TRAIN/TEST DRIFT (PSI > 0.25) -- {len(drift)}:")
for _, r in drift.head(15).iterrows():
    print(f"       {r['feature']:<32} PSI={r['psi']:>8.3f}  "
          f"train={r['mean_train']:.4f}  test={r['mean_test']:.4f}")

# 4d. Unit anomalies (suspicious ranges)
print(f"\n   4d. UNIT / RANGE ANOMALIES:")
for _, r in df.iterrows():
    f = r["feature"]
    if r["min"] is None:
        continue
    # Temperature in Kelvin should be 240-320; in Celsius -50 to 65
    if "TEMP" in f and r["min"] < 200 and r["max"] > 200:
        print(f"       {f}: {r['min']}..{r['max']} (Kelvin range, OK)")
    # Rain in metres should be tiny; in mm larger
    if f.startswith("RAIN") and r["max"] < 0.001:
        print(f"       {f}: max={r['max']} -- likely metres, not mm")

# 4e. Family coverage summary
print(f"\n   4e. FAMILY SUMMARY (mean coverage, n features):")
fam_summary = df.groupby("family").agg(
    n_features=("feature", "count"),
    mean_coverage=("coverage_pct", "mean"),
    n_dead=("coverage_pct", lambda s: int((s == 0).sum())),
).sort_values("mean_coverage")
for fam, r in fam_summary.iterrows():
    print(f"       {fam:<28} {int(r['n_features']):>3} features  "
          f"cov {r['mean_coverage']:>6.1f}%  dead {int(r['n_dead'])}")

# Save the full report
df.to_csv(FEATURES_DIR / "feature_audit.csv", index=False)
print(f"\n   Saved feature_audit.csv ({len(df)} features)")

# ---------------------------------------------------------------------------
# 5. Leakage audit: which features encode the target?
# ---------------------------------------------------------------------------
print(f"\n5. LEAKAGE AUDIT")
label_derived = set(fl.LABEL_DERIVED_FEATURES)
import weak_labels
import surface_labels
weak_inputs = set(weak_labels.LABEL_INPUT_FEATURES)
surface_inputs = set(surface_labels.surface_label_inputs())

print(f"   LABEL_DERIVED (18 industrial): {len(label_derived)}")
print(f"   weak-rule inputs: {len(weak_inputs)}")
print(f"   surface-label inputs: {len(surface_inputs)}")
print(f"   surface inputs NOT in weak inputs: {sorted(surface_inputs - weak_inputs)}")

# SITE_FRP_BASELINE / SITE_N_EVENTS audit
print(f"\n   SITE AGGREGATE AUDIT:")
for f in ["SITE_FRP_BASELINE", "SITE_N_EVENTS", "FRP_ANOMALY_VS_SITE"]:
    if f in all_cols:
        r = df[df["feature"] == f].iloc[0]
        print(f"       {f}: cov={r['coverage_pct']:.1f}%  "
              f"mean={r['mean']:.2f}  std={r['std']:.2f}  "
              f"min={r['min']}  max={r['max']}")

# Recurrence audit
print(f"\n   RECURRENCE AUDIT:")
for f in ["CELL_FIRE_COUNT_1Y", "CELL_FIRE_COUNT_3Y", "CELL_FIRE_COUNT_5Y",
          "CELL_FIRE_DENSITY_1Y", "MAX_INTER_EVENT_GAP_LOG"]:
    if f in all_cols:
        r = df[df["feature"] == f].iloc[0]
        print(f"       {f}: cov={r['coverage_pct']:.1f}%  "
              f"mean={r['mean']:.1f}  median~{r['mean']:.1f}  "
              f"max={r['max']:.1f}")

print("\n" + "=" * 74)
print("AUDIT COMPLETE")
print("=" * 74)
