# ============================================================================
# LOCAL RETRAIN — Model A & B with merged P7 features
# ============================================================================
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pickle
import json
from pathlib import Path
import sys

sys.path.insert(0, "/Users/aashutosh/konsors_sih2/konsors-sih2")

import feature_ledger as fl
import training_design as td
import threshold_calibration as tc
from explainability import shap_explain, compare_methods

FEATURES_DIR = Path("/Users/aashutosh/konsors_sih2/konsors-sih2/Features_mapped")
MODEL_DIR = FEATURES_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

X_path = FEATURES_DIR / "X_with_p7_full.parquet"
meta_path = FEATURES_DIR / "meta.parquet"
y_path = FEATURES_DIR / "y.parquet"

print("=" * 70)
print("LOCAL RETRAIN WITH P7 FEATURES")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1. Load meta & y
# -----------------------------------------------------------------------------
print("\n1. Loading meta & labels...")
meta = pd.read_parquet(meta_path, columns=["event_id", "event_centroid_lon", "event_centroid_lat", "start_time", "start_year", "spatial_block", "site_id"])
meta = meta.reset_index().rename(columns={"index": "row"})

y = pd.read_parquet(y_path, columns=["label_weak"])
y.columns = ["label_weak"]
y = y.reset_index().rename(columns={"index": "row"})

# Target: INDUSTRIAL_FIRE vs rest
target_name = "INDUSTRIAL_FIRE"
y_binary = (y["label_weak"] == target_name).astype("int8")

print(f"  events: {len(meta):,}")
print(f"  target prevalence: {y_binary.mean():.4f}")

# -----------------------------------------------------------------------------
# 2. Split (year_x_block), then carve a CALIBRATION fold out of train
# -----------------------------------------------------------------------------
#
# WHY A SECOND SPLIT. The operating threshold must not be frozen on scores the
# model has already memorised. The previous version calibrated on TRAIN
# PREDICTIONS from a model fit on that same train fold, and the overconfidence
# that buys is large: a tau chosen that way reports precision 0.9000 in-sample
# and 0.8243 on the held-out fold. ~8 points of precision overstated, in the
# direction of auto-alerting.
#
# So train is split once more, with the project's own out-of-time AND
# out-of-space rule, and the inner test fold becomes the calibration set. The
# guarantee that matters is the spatial one: zero of the 4,239 calibration
# blocks appear in the fit fold, so no site is half-remembered.
#
#     fit   -> fit the booster
#     calib -> freeze tau          (never touched by training)
#     test  -> report PR-AUC       (never touched at all)
#
# NO EARLY STOPPING, deliberately. `calib` is already spent freezing tau; using
# it a second time to pick best_iteration would re-open exactly the optimistic
# bias this split exists to remove. num_boost_round stays fixed, and the test
# fold is no longer passed to LightGBM as a validation set at all.
print("\n2. Splitting (year_x_block + nested calibration fold)...")
meta = td.add_split_keys(meta)
parts = td.make_split(meta, kind="year_x_block", seed=0)

train_idx = parts["train"]["row"].to_numpy()
test_idx = parts["test"]["row"].to_numpy()

inner = td.make_split(parts["train"].reset_index(drop=True), kind="year_x_block", seed=1)
fit_idx = inner["train"]["row"].to_numpy()
calib_idx = inner["test"]["row"].to_numpy()

_shared = set(inner["test"]["spatial_block"]) & set(inner["train"]["spatial_block"])
assert not _shared, f"calibration blocks leaked into fit: {len(_shared)} blocks"

print(f"  fit:   {len(fit_idx):,}   (trained on)")
print(f"  calib: {len(calib_idx):,}   (threshold frozen here; 0 shared blocks)")
print(f"  test:  {len(test_idx):,}   (reported, never used for any decision)")

# -----------------------------------------------------------------------------
# 3. Feature lists (leakage partition)
# -----------------------------------------------------------------------------
print("\n3. Building feature lists...")
X_pf = pq.ParquetFile(X_path)
all_cols = X_pf.schema_arrow.names

label_derived = set(fl.LABEL_DERIVED_FEATURES)
weak_rule_inputs = {"FRP_PERSISTENCE", "DETECTION_COUNT_LOG", "TIME_TO_PEAK",
                    "AREA_GROWTH_RATE", "SPREAD_SPEED_MEDIAN", "DIRECTION_CONSISTENCY"}
# P3d daily FWI columns are corrupt (lat/lon swap -> Arctic temps, join
# collapsed: 7.2M rows share one Arctic value). The honest monthly versions
# from P7n live under the _MONTHLY / TEMP_K / RAIN_M / RAIN_7D_LOG names, so
# the plain-named daily copies are excluded until Colab is re-run.
corrupt_daily = {"FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "FWI_ANOMALY_Z",
                 "TEMP_C", "RH_PCT", "WIND_KMH", "RAIN_MM", "RAIN_7D_MM",
                 "SWVL1", "VPD_KPA"}
exclude_features = ({"row", "NBR_SOURCE", "event_id"}  # string/provenance columns
                    | corrupt_daily)

model_A_features = [c for c in all_cols if c not in exclude_features]
model_B_features = [c for c in all_cols if c not in exclude_features and c not in label_derived and c not in weak_rule_inputs]

print(f"  Model A features: {len(model_A_features)}")
print(f"  Model B features: {len(model_B_features)}")

# -----------------------------------------------------------------------------
# 4. Stream load train/test
# -----------------------------------------------------------------------------
print("\n4. Loading train/test matrices...")

def load_split(rows, features):
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

print("  Loading fit...")
X_fit_A = load_split(fit_idx, model_A_features)
X_fit_B = load_split(fit_idx, model_B_features)
y_fit = y_binary.iloc[fit_idx].reset_index(drop=True)

print("  Loading calib...")
X_cal_A = load_split(calib_idx, model_A_features)
X_cal_B = load_split(calib_idx, model_B_features)
y_cal = y_binary.iloc[calib_idx].reset_index(drop=True)

print("  Loading test...")
X_test_A = load_split(test_idx, model_A_features)
X_test_B = load_split(test_idx, model_B_features)
y_test = y_binary.iloc[test_idx].reset_index(drop=True)

# float32
for df in [X_fit_A, X_fit_B, X_cal_A, X_cal_B, X_test_A, X_test_B]:
    for c in df.columns:
        df[c] = df[c].astype("float32")

print(f"  fit:   {X_fit_A.shape}  calib: {X_cal_A.shape}  test: {X_test_A.shape}")

# -----------------------------------------------------------------------------
# 5. Train Model A
# -----------------------------------------------------------------------------
print("\n5. Training Model A...")
# Fixed on purpose -- see the split note. No early stopping, so the booster's
# best_iteration always equals this and `predict` is not truncating on a fold
# that also chose tau.
N_BOOST_ROUNDS = 500
params_A = {
    "objective": "binary",
    # `average_precision`, not "aucpr": LightGBM 4.7 silently emits NO
    # evaluation result for the "aucpr" alias, so log_evaluation(100) below
    # printed nothing and any early_stopping() on this would raise. The metric
    # does not affect the fitted model (the objective is `binary` and PR-AUC is
    # computed by sklearn in `eval_model`); it only affects monitoring.
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "random_state": 0,
    "n_jobs": -1,
}
train_set_A = lgb.Dataset(X_fit_A, label=y_fit, feature_name=model_A_features, free_raw_data=False)
# Calib is logged for monitoring only. It is NOT used to pick best_iteration --
# see the note at the split. Test is deliberately not passed here at all.
valid_set_A = lgb.Dataset(X_cal_A, label=y_cal, feature_name=model_A_features, free_raw_data=False, reference=train_set_A)

model_A = lgb.train(params_A, train_set_A, num_boost_round=N_BOOST_ROUNDS,
                    valid_sets=[valid_set_A], valid_names=["calib"],
                    callbacks=[lgb.log_evaluation(100)])

# -----------------------------------------------------------------------------
# 6. Train Model B
# -----------------------------------------------------------------------------
print("\n6. Training Model B...")
params_B = params_A.copy()
train_set_B = lgb.Dataset(X_fit_B, label=y_fit, feature_name=model_B_features, free_raw_data=False)
valid_set_B = lgb.Dataset(X_cal_B, label=y_cal, feature_name=model_B_features, free_raw_data=False, reference=train_set_B)

model_B = lgb.train(params_B, train_set_B, num_boost_round=N_BOOST_ROUNDS,
                    valid_sets=[valid_set_B], valid_names=["calib"],
                    callbacks=[lgb.log_evaluation(100)])

# -----------------------------------------------------------------------------
# 7. Predict & evaluate
# -----------------------------------------------------------------------------
print("\n7. Predicting & evaluating...")
from sklearn.metrics import average_precision_score, precision_recall_curve

def eval_model(y_true, y_pred, name):
    pr_auc = average_precision_score(y_true, y_pred)
    prec, rec, thr = precision_recall_curve(y_true, y_pred)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_f1_idx = np.argmax(f1s)
    best_thr = thr[best_f1_idx] if best_f1_idx < len(thr) else 0.5
    best_f1 = f1s[best_f1_idx]
    
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

pred_A = model_A.predict(X_test_A, num_iteration=model_A.best_iteration)
pred_B = model_B.predict(X_test_B, num_iteration=model_B.best_iteration)

eval_A = eval_model(y_test, pred_A, "Model A")
eval_B = eval_model(y_test, pred_B, "Model B")

# -----------------------------------------------------------------------------
# 8. Threshold calibration (F2, precision >= 0.90) on the CALIBRATION fold
# -----------------------------------------------------------------------------
#
# Scores here come from a fold the booster never trained on. That is the whole
# point: it is what makes the quoted precision survive contact with the test
# fold. `tc.assert_calibrated_on_train("calib")` is deliberately NOT called --
# that guard exists to catch test-fold calibration, and the calibration fold is
# a third thing: it is held out of training precisely so it can play that role.
print("\n8. Threshold calibration (on held-out calib fold)...")
pred_cal_A = model_A.predict(X_cal_A, num_iteration=model_A.best_iteration)
pred_cal_B = model_B.predict(X_cal_B, num_iteration=model_B.best_iteration)

tau_A = tc.calibrate_operating_threshold(y_cal, pred_cal_A, min_precision=0.90, beta=2.0)
tau_B = tc.calibrate_operating_threshold(y_cal, pred_cal_B, min_precision=0.90, beta=2.0)

print(f"  Model A: {tc.describe_calibration(tau_A, 'A')}")
print(f"  Model B: {tc.describe_calibration(tau_B, 'B')}")

# Honesty check: report what the frozen tau ACTUALLY delivers on test next to
# what it promised on calib. A gap here is the early-warning that the
# calibration fold stopped looking like the test fold.
for tag, cal, pred in (("A", tau_A, pred_A), ("B", tau_B, pred_B)):
    t = cal.get("threshold")
    if t is None:
        print(f"  Model {tag}: no operating point freezable -> everything reviewed")
        continue
    flagged = pred >= t
    got_p = float(y_test[flagged].mean()) if flagged.any() else 0.0
    got_r = float(y_test[flagged].sum() / y_test.sum())
    print(f"  Model {tag} on TEST at tau={t:.4f}: precision {got_p:.4f} "
          f"(promised {cal['precision']:.4f}, gap {got_p - cal['precision']:+.4f})  "
          f"recall {got_r:.4f}")

# -----------------------------------------------------------------------------
# 9. Save models + predictions
# -----------------------------------------------------------------------------
#
# ARTIFACT CONTRACT. These keys are what `backend/app/services/model_registry.py`
# reads, and they are not interchangeable with other spellings:
#
#     operating_threshold     float | None   the frozen tau
#     auto_accept_threshold   float          max(tau, review_confidence)
#     review_confidence       float          the brief's 0.60 human-review floor
#
# The previous version wrote the whole calibration DICT under a key named `tau`,
# which the registry never looks for. `operating_threshold` came back None, and
# the backend silently fell back to the bare 0.60 literal -- so the calibrated
# operating point existed only in this log output and never reached a single
# alert. Same shape as `colab_p7_train.py`; keep the two in step.
import dataset_builder as db

print("\n9. Saving artefacts...")


def build_payload(booster, features, cal, eval_result):
    tau = cal.get("threshold")
    return {
        "booster": booster,
        "features": features,
        "target": target_name,
        "label_column": "weak",
        "best_iteration": int(booster.best_iteration),
        "split": "year_x_block",
        "review_confidence": float(db.REVIEW_CONFIDENCE),
        "operating_threshold": tau,
        "operating_point": cal,
        "auto_accept_threshold": tc.effective_threshold(tau, db.REVIEW_CONFIDENCE),
        "n_fit": int(len(fit_idx)),
        "n_calib": int(len(calib_idx)),
        "n_test": int(len(test_idx)),
        "eval": eval_result,
    }


payload_A = build_payload(model_A, model_A_features, tau_A, eval_A)
payload_B = build_payload(model_B, model_B_features, tau_B, eval_B)

# Fail loudly here rather than let the registry quietly degrade to 0.60 again.
for tag, pl in (("A", payload_A), ("B", payload_B)):
    assert isinstance(pl["auto_accept_threshold"], float), f"{tag}: auto_accept must be a float"
    assert pl["operating_threshold"] is None or isinstance(pl["operating_threshold"], float), \
        f"{tag}: operating_threshold must be a float or None, not {type(pl['operating_threshold']).__name__}"

with open(MODEL_DIR / "model_A.pkl", "wb") as f:
    pickle.dump(payload_A, f)
with open(MODEL_DIR / "model_B.pkl", "wb") as f:
    pickle.dump(payload_B, f)

# Save predictions
meta_test = meta.iloc[test_idx].reset_index(drop=True)
pd.DataFrame({"event_id": meta_test["event_id"], "score": pred_A, "y_true": y_test}).to_csv(FEATURES_DIR / "heldout_pred_A.csv", index=False)
pd.DataFrame({"event_id": meta_test["event_id"], "score": pred_B, "y_true": y_test}).to_csv(FEATURES_DIR / "heldout_pred_B.csv", index=False)

print("  Saved model_A.pkl, model_B.pkl")
print("  Saved heldout_pred_A.csv, heldout_pred_B.csv")

# -----------------------------------------------------------------------------
# 10. SHAP analysis (50k sample)
# -----------------------------------------------------------------------------
print("\n10. SHAP analysis (50k sample)...")
import shap

SAMPLE_N = 50000
np.random.seed(42)
sample_idx = np.random.choice(len(X_test_A), min(SAMPLE_N, len(X_test_A)), replace=False)

X_sample_A = X_test_A.iloc[sample_idx].reset_index(drop=True)
X_sample_B = X_test_B.iloc[sample_idx].reset_index(drop=True)

print(f"  Sample size: {len(X_sample_A)}")

print("  Model A SHAP...")
explainer_A = shap.TreeExplainer(model_A)
shap_A = explainer_A.shap_values(X_sample_A[model_A_features])
if isinstance(shap_A, list):
    shap_A = shap_A[1]

print("  Model B SHAP...")
explainer_B = shap.TreeExplainer(model_B)
shap_B = explainer_B.shap_values(X_sample_B[model_B_features])
if isinstance(shap_B, list):
    shap_B = shap_B[0]

# `fl.family_of` resolves the 141-spec AND the P7-merge extras. The previous
# inline map came from FEATURE_GROUPS alone, so the 26 extra columns fell
# into a bucket named "other" that reported 1.36 of Model B's mean|SHAP| --
# larger than any real family, and 7 of the top 12 features sat in it.
def aggregate_shap(shap_vals, features):
    mean_abs = np.abs(shap_vals).mean(axis=0)
    by_feature = dict(zip(features, mean_abs))
    by_family = {}
    for feat, val in by_feature.items():
        fam = fl.family_of(feat)
        by_family[fam] = by_family.get(fam, 0) + val
    return by_feature, by_family

feat_A, fam_A = aggregate_shap(shap_A, model_A_features)
feat_B, fam_B = aggregate_shap(shap_B, model_B_features)

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

# Save SHAP
np.save(FEATURES_DIR / "shap_values_A.npy", shap_A)
np.save(FEATURES_DIR / "shap_values_B.npy", shap_B)

results = {
    "model_A": {"mean_abs_shap": feat_A, "top_features": top_A, "by_family": fam_A, "n_samples": len(X_sample_A)},
    "model_B": {"mean_abs_shap": feat_B, "top_features": top_B, "by_family": fam_B, "n_samples": len(X_sample_B)},
}
with open(FEATURES_DIR / "shap_analysis.json", "w") as f:
    json.dump(results, f, indent=2)

# Per-event example
#
# The per-event explanation no longer needs a COMPLETE row: shap_explain
# routes NaN down the tree's own missing branch (shap >= 0.41), which is
# the same path booster.predict takes. The previous version raised
# MissingFeatureError on any NaN, and no complete row exists on this
# catalogue -- 26 of Model B's features are unmeasured on most rows
# (SPREAD 3.3% covered, THERMAL 67%, ATMOSPHERIC 59%) -- so the
# explanation was unreachable. The first test row is now explainable,
# and the Explanation carries `missing_features` so the reader can see
# which attributions came from the missing branch.
print("\n11. Per-event SHAP example...")
test_row_idx = test_idx[0]
event_row = X_test_A.iloc[0].to_dict()
exp_A = shap_explain(model_A, model_A_features, event_row, model_name="A")
exp_B = shap_explain(model_B, model_B_features, event_row, model_name="B")

event_id = meta.loc[meta["row"] == test_row_idx, "event_id"].values[0]
print(f"  Event: {event_id}")
print(f"  Model A prediction: {exp_A.prediction:.4f}  "
      f"(margin {exp_A.margin_prediction:.4f}, "
      f"additivity residual {exp_A.additivity_residual:.2e})")
print(f"  Model B prediction: {exp_B.prediction:.4f}  "
      f"(margin {exp_B.margin_prediction:.4f}, "
      f"additivity residual {exp_B.additivity_residual:.2e})")
print(f"  Model A top 5: {exp_A.top_features[:5]}")
print(f"  Model B top 5: {exp_B.top_features[:5]}")
print(f"  Model A unmeasured: {len(exp_A.missing_features)}/{exp_A.n_features_expected}")
print(f"  Model B unmeasured: {len(exp_B.missing_features)}/{exp_B.n_features_expected}")

comp = compare_methods(exp_A, exp_B)
print(f"  Agreement: {comp['agreement']} (Jaccard={comp['jaccard']:.3f})")

print("\n" + "=" * 70)
print("RETRAIN + SHAP COMPLETE")
print("=" * 70)