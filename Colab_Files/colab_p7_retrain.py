# ============================================================================
# CELL 2: Retrain Model A & B with merged features
# ============================================================================
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pickle
from pathlib import Path
import training_design as td
import threshold_calibration as tc
from explainability import shap_explain, compare_methods

ROOT = Path("/content/drive/MyDrive/Firm_Clusters")
DATASET_DIR = ROOT / "full_2012_2026" / "features" / "dataset"
QC_DIR = ROOT / "qc"
MODEL_DIR = QC_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("P7 RETRAIN WITH P7 FEATURES")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1. Load data with streaming (memory efficient)
# -----------------------------------------------------------------------------
print("\n1. Loading data...")
X_path = DATASET_DIR / "X_with_p7.parquet"
meta_path = DATASET_DIR / "meta.parquet"
y_path = DATASET_DIR / "y.parquet"

# meta.parquet has implicit row index (0..N-1), not explicit 'row' column
meta = pd.read_parquet(meta_path, columns=["event_id", "start_year", "spatial_block", "site_id"])
meta = meta.reset_index().rename(columns={"index": "row"})  # add explicit row index

y = pd.read_parquet(y_path, columns=["label_weak"])
y.columns = ["label_weak"]
y = y.reset_index().rename(columns={"index": "row"})  # align row index

# Target: INDUSTRIAL_FIRE vs rest (binary)
target_name = "INDUSTRIAL_FIRE"
y_binary = (y["label_weak"] == target_name).astype("int8")

print(f"  events: {len(meta):,}")
print(f"  target prevalence: {y_binary.mean():.4f}")

# -----------------------------------------------------------------------------
# 2. Split (same as before: year_x_block)
# -----------------------------------------------------------------------------
print("\n2. Splitting...")
meta = td.add_split_keys(meta)
parts = td.make_split(meta, kind="year_x_block", seed=0)

train_idx = parts["train"]["row"].to_numpy()
test_idx = parts["test"]["row"].to_numpy()

print(f"  train: {len(train_idx):,}  test: {len(test_idx):,}")

# -----------------------------------------------------------------------------
# 3. Load X in chunks, select model features
# -----------------------------------------------------------------------------
print("\n3. Loading features...")
X_pf = pq.ParquetFile(X_path)
all_cols = X_pf.schema_arrow.names

# Get feature lists from ledger (excluding label-derived)
import feature_ledger as fl
label_derived = set(fl.LABEL_DERIVED_FEATURES)
weak_rule_inputs = {"FRP_PERSISTENCE", "DETECTION_COUNT_LOG", "TIME_TO_PEAK",
                    "AREA_GROWTH_RATE", "SPREAD_SPEED_MEDIAN", "DIRECTION_CONSISTENCY"}

model_A_features = [c for c in all_cols if c != "row"]
model_B_features = [c for c in all_cols if c != "row" and c not in label_derived and c not in weak_rule_inputs]

print(f"  Model A features: {len(model_A_features)}")
print(f"  Model B features: {len(model_B_features)}")

# Stream train/test matrices
def load_split(rows, features):
    """Stream row groups, filter to requested rows & features."""
    chunks = []
    row_set = set(rows)
    for rg in range(X_pf.num_row_groups):
        rg_data = X_pf.read_row_group(rg, columns=["row"] + features).to_pandas()
        mask = rg_data["row"].isin(row_set)
        if mask.any():
            chunks.append(rg_data[mask].drop(columns=["row"]))
    if chunks:
        return pd.concat(chunks, ignore_index=True)
    return pd.DataFrame(columns=features)

print("  Loading train...")
X_train_A = load_split(train_idx, model_A_features)
X_train_B = load_split(train_idx, model_B_features)
y_train = y_binary.iloc[train_idx].reset_index(drop=True)

print("  Loading test...")
X_test_A = load_split(test_idx, model_A_features)
X_test_B = load_split(test_idx, model_B_features)
y_test = y_binary.iloc[test_idx].reset_index(drop=True)

# Convert to float32, NaN stays NaN
for df in [X_train_A, X_train_B, X_test_A, X_test_B]:
    for c in df.columns:
        df[c] = df[c].astype("float32")

print(f"  Train A: {X_train_A.shape}  Test A: {X_test_A.shape}")
print(f"  Train B: {X_train_B.shape}  Test B: {X_test_B.shape}")

# -----------------------------------------------------------------------------
# 4. Train Model A (all features)
# -----------------------------------------------------------------------------
print("\n4. Training Model A...")
params_A = {
    "objective": "binary",
    "metric": "aucpr",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "random_state": 0,
}
train_set_A = lgb.Dataset(X_train_A, label=y_train, feature_name=model_A_features, free_raw_data=False)
valid_set_A = lgb.Dataset(X_test_A, label=y_test, feature_name=model_A_features, free_raw_data=False, reference=train_set_A)

model_A = lgb.train(params_A, train_set_A, num_boost_round=2000,
                    valid_sets=[valid_set_A], valid_names=["valid"],
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])

# -----------------------------------------------------------------------------
# 5. Train Model B (non-leak features)
# -----------------------------------------------------------------------------
print("\n5. Training Model B...")
params_B = params_A.copy()
train_set_B = lgb.Dataset(X_train_B, label=y_train, feature_name=model_B_features, free_raw_data=False)
valid_set_B = lgb.Dataset(X_test_B, label=y_test, feature_name=model_B_features, free_raw_data=False, reference=train_set_B)

model_B = lgb.train(params_B, train_set_B, num_boost_round=2000,
                    valid_sets=[valid_set_B], valid_names=["valid"],
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])

# -----------------------------------------------------------------------------
# 6. Predict on test
# -----------------------------------------------------------------------------
print("\n6. Predicting...")
pred_A = model_A.predict(X_test_A, num_iteration=model_A.best_iteration)
pred_B = model_B.predict(X_test_B, num_iteration=model_B.best_iteration)

# -----------------------------------------------------------------------------
# 7. Evaluate
# -----------------------------------------------------------------------------
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score

def eval_model(y_true, y_pred, name):
    pr_auc = average_precision_score(y_true, y_pred)
    prec, rec, thr = precision_recall_curve(y_true, y_pred)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_f1_idx = np.argmax(f1s)
    best_thr = thr[best_f1_idx] if best_f1_idx < len(thr) else 0.5
    best_f1 = f1s[best_f1_idx]
    
    # Precision @ 0.90
    p90_idx = np.where(prec >= 0.90)[0]
    if len(p90_idx):
        p90_rec = rec[p90_idx[0]]
        p90_thr = thr[p90_idx[0]]
    else:
        p90_rec = 0.0
        p90_thr = 1.0
    
    print(f"  {name}: PR-AUC={pr_auc:.4f}, best_F1={best_f1:.4f}@thr={best_thr:.3f}, "
          f"recall@prec>=0.90={p90_rec:.4f}@thr={p90_thr:.3f}")
    return {"pr_auc": pr_auc, "best_f1": best_f1, "best_thr": best_thr,
            "p90_recall": p90_rec, "p90_thr": p90_thr}

eval_A = eval_model(y_test, pred_A, "Model A")
eval_B = eval_model(y_test, pred_B, "Model B")

# -----------------------------------------------------------------------------
# 8. Threshold calibration (F2, precision >= 0.90 on TRAIN)
# -----------------------------------------------------------------------------
print("\n7. Threshold calibration...")
pred_train_A = model_A.predict(X_train_A, num_iteration=model_A.best_iteration)
pred_train_B = model_B.predict(X_train_B, num_iteration=model_B.best_iteration)

tau_A = tc.calibrate_threshold(y_train, pred_train_A, target_precision=0.90, beta=2.0)
tau_B = tc.calibrate_threshold(y_train, pred_train_B, target_precision=0.90, beta=2.0)

print(f"  Model A tau: {tau_A}")
print(f"  Model B tau: {tau_B}")

# -----------------------------------------------------------------------------
# 8. Save models + predictions
# -----------------------------------------------------------------------------
print("\n8. Saving artefacts...")
payload_A = {
    "booster": model_A,
    "features": model_A_features,
    "target": target_name,
    "split": "year_x_block",
    "tau": tau_A,
    "eval": eval_A,
}
payload_B = {
    "booster": model_B,
    "features": model_B_features,
    "target": target_name,
    "split": "year_x_block",
    "tau": tau_B,
    "eval": eval_B,
}

with open(MODEL_DIR / "model_A.pkl", "wb") as f:
    pickle.dump(payload_A, f)
with open(MODEL_DIR / "model_B.pkl", "wb") as f:
    pickle.dump(payload_B, f)

# Save predictions
meta_test = meta.iloc[test_idx].reset_index(drop=True)
pd.DataFrame({"event_id": meta_test["event_id"], "score": pred_A, "y_true": y_test}).to_csv(QC_DIR / "heldout_pred_A.csv", index=False)
pd.DataFrame({"event_id": meta_test["event_id"], "score": pred_B, "y_true": y_test}).to_csv(QC_DIR / "heldout_pred_B.csv", index=False)

print("  Saved model_A.pkl, model_B.pkl")
print("  Saved heldout_pred_A.csv, heldout_pred_B.csv")

print("\n" + "=" * 70)
print("RETRAIN COMPLETE")
print("=" * 70)