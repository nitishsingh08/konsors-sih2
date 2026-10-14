# ============================================================================
# CELL 3: SHAP Analysis on retrained models
# ============================================================================
import shap
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pickle
import json
from pathlib import Path

ROOT = Path("/content/drive/MyDrive/Firm_Clusters")
DATASET_DIR = ROOT / "full_2012_2026" / "features" / "dataset"
QC_DIR = ROOT / "qc"
MODEL_DIR = QC_DIR / "models"

print("=" * 70)
print("P7 SHAP ANALYSIS")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1. Load models
# -----------------------------------------------------------------------------
print("\n1. Loading models...")
with open(MODEL_DIR / "model_A.pkl", "rb") as f:
    payload_A = pickle.load(f)
with open(MODEL_DIR / "model_B.pkl", "rb") as f:
    payload_B = pickle.load(f)

model_A = payload_A["booster"]
model_B = payload_B["booster"]
features_A = payload_A["features"]
features_B = payload_B["features"]

print(f"  Model A: {len(features_A)} features, {model_A.best_iteration} trees")
print(f"  Model B: {len(features_B)} features, {model_B.best_iteration} trees")

# -----------------------------------------------------------------------------
# 2. Load test data (same split as training)
# -----------------------------------------------------------------------------
print("\n2. Loading test data...")
import training_design as td
meta = pd.read_parquet(DATASET_DIR / "meta.parquet", columns=["event_id", "start_year", "spatial_block", "site_id"])
meta = meta.reset_index().rename(columns={"index": "row"})
meta = td.add_split_keys(meta)
parts = td.make_split(meta, kind="year_x_block", seed=0)
test_idx = parts["test"]["row"].to_numpy()

X_path = DATASET_DIR / "X_with_p7.parquet"
X_pf = pq.ParquetFile(X_path)

def load_test_rows(rows, features):
    chunks = []
    for rg in range(X_pf.num_row_groups):
        rg_data = X_pf.read_row_group(rg, columns=["row"] + features).to_pandas()
        mask = rg_data["row"].isin(rows)
        if mask.any():
            chunks.append(rg_data[mask].drop(columns=["row"]))
    if chunks:
        return pd.concat(chunks, ignore_index=True)
    return pd.DataFrame(columns=features)

X_test_A = load_test_rows(test_idx, features_A)
X_test_B = load_test_rows(test_idx, features_B)

for df in [X_test_A, X_test_B]:
    for c in df.columns:
        df[c] = df[c].astype("float32")

print(f"  Test A: {X_test_A.shape}  Test B: {X_test_B.shape}")

# -----------------------------------------------------------------------------
# 3. Compute SHAP (sample for speed - SHAP on 888k rows is slow)
# -----------------------------------------------------------------------------
print("\n3. Computing SHAP...")
# Stratified sample: 50k test rows
SAMPLE_N = 50000
np.random.seed(42)
sample_idx = np.random.choice(len(X_test_A), min(SAMPLE_N, len(X_test_A)), replace=False)

X_sample_A = X_test_A.iloc[sample_idx].reset_index(drop=True)
X_sample_B = X_test_B.iloc[sample_idx].reset_index(drop=True)

print(f"  Sample size: {len(X_sample_A)}")

print("  Model A SHAP...")
explainer_A = shap.TreeExplainer(model_A)
shap_A = explainer_A.shap_values(X_sample_A[features_A])
if isinstance(shap_A, list):
    shap_A = shap_A[1]  # positive class

print("  Model B SHAP...")
explainer_B = shap.TreeExplainer(model_B)
shap_B = explainer_B.shap_values(X_sample_B[features_B])
if isinstance(shap_B, list):
    shap_B = shap_B[0]

# -----------------------------------------------------------------------------
# 4. Aggregate by family & save
# -----------------------------------------------------------------------------
print("\n4. Aggregating by family...")
import feature_ledger as fl
family_map = {name: group for group, names in fl.FEATURE_GROUPS.items() for name in names}

def aggregate_shap(shap_vals, features):
    mean_abs = np.abs(shap_vals).mean(axis=0)
    by_feature = dict(zip(features, mean_abs))
    by_family = {}
    for feat, val in by_feature.items():
        fam = family_map.get(feat, "other")
        by_family[fam] = by_family.get(fam, 0) + val
    return by_feature, by_family

feat_A, fam_A = aggregate_shap(shap_A, features_A)
feat_B, fam_B = aggregate_shap(shap_B, features_B)

# Top 20 features
top_A = dict(sorted(feat_A.items(), key=lambda x: x[1], reverse=True)[:20])
top_B = dict(sorted(feat_B.items(), key=lambda x: x[1], reverse=True)[:20])

print("\n  Model A top 10:")
for f, v in list(top_A.items())[:10]:
    print(f"    {f}: {v:.6f}")
print("\n  Model B top 10:")
for f, v in list(top_B.items())[:10]:
    print(f"    {f}: {v:.6f}")

print("\n  Model A by family:")
for f, v in sorted(fam_A.items(), key=lambda x: x[1], reverse=True):
    print(f"    {f}: {v:.6f}")
print("\n  Model B by family:")
for f, v in sorted(fam_B.items(), key=lambda x: x[1], reverse=True):
    print(f"    {f}: {v:.6f}")

# -----------------------------------------------------------------------------
# 5. Save results
# -----------------------------------------------------------------------------
print("\n5. Saving...")
np.save(QC_DIR / "shap_values_A.npy", shap_A)
np.save(QC_DIR / "shap_values_B.npy", shap_B)

results = {
    "model_A": {
        "mean_abs_shap": feat_A,
        "top_features": top_A,
        "by_family": fam_A,
        "n_samples": len(X_sample_A),
    },
    "model_B": {
        "mean_abs_shap": feat_B,
        "top_features": top_B,
        "by_family": fam_B,
        "n_samples": len(X_sample_B),
    },
}
with open(QC_DIR / "shap_analysis.json", "w") as f:
    json.dump(results, f, indent=2)

# -----------------------------------------------------------------------------
# 6. Per-event SHAP example (using explainability.py)
# -----------------------------------------------------------------------------
print("\n6. Per-event SHAP example...")
from explainability import shap_explain, compare_methods

# Pick one test event (first test row)
test_row_idx = test_idx[0]
event_row = X_test_A.iloc[0].to_dict()
exp_A = shap_explain(model_A, features_A, event_row, model_name="A")
exp_B = shap_explain(model_B, features_B, event_row, model_name="B")

# Get event_id from meta (meta has 'row' column after reset_index)
event_id = meta.loc[meta["row"] == test_row_idx, "event_id"].values[0]

print(f"  Event: {event_id}")
print(f"  Model A prediction: {exp_A.prediction:.4f}")
print(f"  Model B prediction: {exp_B.prediction:.4f}")
print(f"  Model A top 5: {exp_A.top_features[:5]}")
print(f"  Model B top 5: {exp_B.top_features[:5]}")

# Compare
comp = compare_methods(exp_A, exp_B)
print(f"  Agreement: {comp['agreement']} (Jaccard={comp['jaccard']:.3f})")

print("\n" + "=" * 70)
print("SHAP COMPLETE")
print("=" * 70)