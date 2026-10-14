# ============================================================================
# colab_p7m_shap_analysis.py -- SHAP analysis for model interpretability
# ============================================================================

import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from pathlib import Path

import colab_utils as cu

P7M_CELL_VERSION = "2026-10-05.1"

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError(f"no project root under {cu.MYDRIVE}")
QC_DIR = PROJECT_ROOT / "qc"
QC_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 74)
print(f"P7m -- SHAP ANALYSIS  [v{P7M_CELL_VERSION}]")
print("=" * 74)

try:
    import shap
    print(f"SHAP version: {shap.__version__}")
except ImportError:
    print("SHAP not installed. Run: pip install shap")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# 1. LOAD MODELS AND DATA
# ---------------------------------------------------------------------------
print("=" * 74)
print("1. LOAD MODELS AND DATA")
print("=" * 74)

MODEL_A_PATH = PROJECT_ROOT / "qc" / "models" / "model_A.pkl"
MODEL_B_PATH = PROJECT_ROOT / "qc" / "models" / "model_B.pkl"
PRED_A_PATH = PROJECT_ROOT / "qc" / "heldout_pred_A.csv"
PRED_B_PATH = PROJECT_ROOT / "qc" / "heldout_pred_B.csv"

if not Path(MODEL_A_PATH).exists() or not Path(MODEL_B_PATH).exists():
    raise SystemExit("Model pickles not found. Run colab_p7_train.py first.")

if not Path(PRED_A_PATH).exists() or not Path(PRED_B_PATH).exists():
    raise SystemExit("Prediction CSVs not found. Run colab_p7_train.py first.")

# Load models
import pickle
with open(MODEL_A_PATH, "rb") as f:
    payload_A = pickle.load(f)
with open(MODEL_B_PATH, "rb") as f:
    payload_B = pickle.load(f)

# `colab_p7_train.py` persists a DICT, not a bare booster:
#     {"booster": ..., "features": [...], "target": ..., ...}
# Treating the dict as a booster raises `AttributeError: 'dict' object has
# no attribute 'feature_name'`. Unpack it here.
if not isinstance(payload_A, dict) or "booster" not in payload_A:
    raise ValueError(
        f"Model A at {MODEL_A_PATH} is not the dict payload written by "
        f"colab_p7_train.py (got {type(payload_A).__name__}). Re-run P7 "
        f"training and re-pickle.")
if not isinstance(payload_B, dict) or "booster" not in payload_B:
    raise ValueError(
        f"Model B at {MODEL_B_PATH} is not the dict payload written by "
        f"colab_p7_train.py (got {type(payload_B).__name__}). Re-run P7 "
        f"training and re-pickle.")

model_A = payload_A["booster"]
model_B = payload_B["booster"]
features_A = list(payload_A["features"])
features_B = list(payload_B["features"])

# A booster maps COLUMN POSITIONS, so the stored feature list is the only
# safe way to align a frame with it. `model.feature_name()` returns the
# names in the booster's own order, which matches the stored list when
# P7 wrote them from the same source -- but the stored list is the
# contract, so use it.
if features_A != model_A.feature_name():
    print(f"  WARNING: stored feature list for model A differs from the "
          f"booster's own order. Using the stored list.")
if features_B != model_B.feature_name():
    print(f"  WARNING: stored feature list for model B differs from the "
          f"booster's own order. Using the stored list.")

print(f"Model A: {len(features_A)} features, "
      f"{getattr(model_A, 'best_iteration', '?')} trees")
print(f"Model B: {len(features_B)} features, "
      f"{getattr(model_B, 'best_iteration', '?')} trees")

# Load predictions to get test indices
pred_A = pd.read_csv(PRED_A_PATH)
pred_B = pd.read_csv(PRED_B_PATH)

# Load test data (same as training). Paths go through the resolver, not a
# literal -- a literal broke twice when the tree moved to a duplicated
# folder, and once when a stale mount hid it.
import pyarrow.parquet as pq
import pyarrow as pa

DATASET_DIR = PROJECT_ROOT / "full_2012_2026" / "features" / "dataset"
META_PATH = DATASET_DIR / "meta.parquet"
X_PATH = DATASET_DIR / "X.parquet"

# Load test indices (same as training)
meta = pd.read_parquet(META_PATH)
import training_design as td
meta = td.add_split_keys(meta)
parts = td.make_split(meta, kind="year_x_block", seed=0)
test_ids = set(parts["test"]["event_id"])

# Load test event_ids from predictions
test_event_ids_A = set(pred_A["event_id"].tolist())
test_event_ids_B = set(pred_B["event_id"].tolist())
test_event_ids = test_event_ids_A | test_event_ids_B

# Load X for test events only
X_cols = list(pq.ParquetFile(X_PATH).schema_arrow.names)
X_test = pd.read_parquet(X_PATH, columns=["row"] + [c for c in X_cols if c != "row"])

# Filter to test events
meta_test = pd.read_parquet(META_PATH, columns=["row", "event_id"])
test_rows = meta_test[meta_test["event_id"].isin(test_event_ids)]["row"].tolist()
X_test = X_test.iloc[test_rows].reset_index(drop=True)

print(f"Test set size: {len(X_test):,}")

# ---------------------------------------------------------------------------
# 2. SHAP ANALYSIS
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("2. SHAP ANALYSIS")
print("=" * 74)

# `TreeExplainer.shap_values`, NOT `model.predict(..., pred_contrib=True)`.
# LightGBM's booster has no `pred_contrib` argument; that call raises
# TypeError and was the second blocker in this cell.
print("Computing SHAP for Model A...")
explainer_A = shap.TreeExplainer(model_A)
shap_values_A = explainer_A.shap_values(X_test[features_A])

# Model B SHAP
print("Computing SHAP for Model B...")
explainer_B = shap.TreeExplainer(model_B)
shap_values_B = explainer_B.shap_values(X_test[features_B])

# `shap_values` may be a list (one array per class) for multiclass models.
# These are binary classifiers, so take the single array if wrapped.
if isinstance(shap_values_A, list):
    shap_values_A = shap_values_A[0]
if isinstance(shap_values_B, list):
    shap_values_B = shap_values_B[0]

# Save SHAP values
np.save("qc/shap_values_A.npy", shap_values_A)
np.save("qc/shap_values_B.npy", shap_values_B)

# ---------------------------------------------------------------------------
# 3. SHAP PLOTS
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("3. SHAP PLOTS")
print("=" * 74)

# Summary plots
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values_A, X_test, feature_names=model_A.feature_name(), show=False)
plt.title("Model A SHAP Summary")
plt.tight_layout()
plt.savefig("qc/shap_summary_A.png", dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values_B, X_test, feature_names=model_B.feature_name(), show=False)
plt.title("Model B SHAP Summary")
plt.tight_layout()
plt.savefig("qc/shap_summary_B.png", dpi=150, bbox_inches='tight')
plt.close()

# Bar plots
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values_A, X_test, feature_names=model_A.feature_name(), plot_type="bar", show=False)
plt.title("Model A SHAP Feature Importance")
plt.tight_layout()
plt.savefig("qc/shap_bar_A.png", dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values_B, X_test, feature_names=model_B.feature_name(), plot_type="bar", show=False)
plt.title("Model B SHAP Feature Importance")
plt.tight_layout()
plt.savefig("qc/shap_bar_B.png", dpi=150, bbox_inches='tight')
plt.close()

# Dependence plots for top features
top_features = ['PEAK_HOUR_COS', 'PEAK_HOUR_SIN', 'FRP_CV', 'FRP_MEDIAN', 'DETECTION_DENSITY', 'OVERPASS_DETECTION_RATIO', 'FRP_P90', 'FRP_MAX']

for feat in ['PEAK_HOUR_COS', 'PEAK_HOUR_SIN', 'FRP_CV', 'FRP_MEDIAN', 'DETECTION_DENSITY', 'OVERPASS_DETECTION_RATIO', 'FRP_P90', 'FRP_MAX']:
    if feat in model_A.feature_name():
        shap.dependence_plot(feat, shap_values_A, X_test, feature_names=model_A.feature_name(), show=False)
        plt.title(f"Model A SHAP Dependence: {feat}")
        plt.savefig(f"qc/shap_dep_A_{feat}.png", dpi=150, bbox_inches='tight')
        plt.close()

    if feat in model_B.feature_name():
        shap.dependence_plot(feat, shap_values_B, X_test, feature_names=model_B.feature_name(), show=False)
        plt.title(f"Model B SHAP Dependence: {feat}")
        plt.savefig(f"qc/shap_dep_B_{feat}.png", dpi=150, bbox_inches='tight')
        plt.close()

# ---------------------------------------------------------------------------
# 4. SAVE RESULTS
# ---------------------------------------------------------------------------
results = {
    "model_A": {
        "mean_abs_shap": {f: float(np.abs(shap_values_A[:, i]).mean()) 
                          for i, f in enumerate(model_A.feature_name())},
        "top_features": dict(sorted(
            {f: float(np.abs(shap_values_A[:, i]).mean()) 
             for i, f in enumerate(model_A.feature_name())}.items(), 
            key=lambda x: x[1], reverse=True)[:20])
    },
    "model_B": {
        "mean_abs_shap": {f: float(np.abs(shap_values_B[:, i]).mean()) 
                          for i, f in enumerate(model_B.feature_name())},
        "top_features": dict(sorted(
            {f: float(np.abs(shap_values_B[:, i]).mean()) 
             for i, f in enumerate(model_B.feature_name())}.items(), 
            key=lambda x: x[1], reverse=True)[:20])
    }
}

with open("qc/shap_analysis.json", "w") as f:
    json.dump(results, f, indent=2)

print()
print("SHAP analysis complete!")
print("Saved: qc/shap_summary_A.png, qc/shap_summary_B.png")
print("Saved: qc/shap_bar_A.png, qc/shap_bar_B.png")
print("Saved: qc/shap_dep_*.png")
print("Saved: qc/shap_values_A.npy, qc/shap_values_B.npy")
print("Saved: qc/shap_analysis.json")
print("=" * 74)
print("END")