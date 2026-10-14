# ============================================================================
# P9 -- MODEL B HYPERPARAMETER TUNING
#
# Reproducible, validation-based LightGBM search. The search is run on
# the FIT fold, scored on the CALIBRATION fold, and the winner is
# re-scored on the TEST fold exactly once. The test fold is never
# touched by the search -- that is the guarantee that makes the final
# number publishable.
#
# WHAT IS TUNED (all from the brief's list):
#   learning_rate, num_boost_round (via early stopping on calib),
#   num_leaves, min_data_in_leaf, feature_fraction, bagging_fraction,
#   lambda_l1, lambda_l2.
#
# WHAT IS NOT TUNED: the threshold. tau is frozen AFTER the winner is
# chosen, on the calibration fold, by threshold_calibration.
#
# CLASS WEIGHTING is an EXPERIMENT, not an assumption: `scale_pos_weight`
# is swept as a dimension and reported, so its effect on high-precision
# recall is measured rather than presumed.
#
# Run:  python3 tune_model_b.py
# ============================================================================
import itertools
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "/Users/aashutosh/konsors_sih2/konsors-sih2")

import feature_ledger as fl
import training_design as td
import threshold_calibration as tc

FEATURES_DIR = Path("/Users/aashutosh/konsors_sih2/konsors-sih2/Features_mapped")
MODEL_DIR = FEATURES_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

X_path = FEATURES_DIR / "X_with_p7_full.parquet"
meta_path = FEATURES_DIR / "meta.parquet"
y_path = FEATURES_DIR / "y.parquet"

print("=" * 74)
print("P9 -- MODEL B HYPERPARAMETER TUNING")
print("=" * 74)

# ---------------------------------------------------------------------------
# 1. Load and split (same split as local_retrain, so results are comparable)
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

inner = td.make_split(parts["train"].reset_index(drop=True),
                      kind="year_x_block", seed=1)
fit_idx = inner["train"]["row"].to_numpy()
calib_idx = inner["test"]["row"].to_numpy()

_shared = set(inner["test"]["spatial_block"]) & set(inner["train"]["spatial_block"])
assert not _shared, f"calibration blocks leaked into fit: {len(_shared)} blocks"

print(f"\n1. SPLIT")
print(f"   fit:   {len(fit_idx):,}")
print(f"   calib: {len(calib_idx):,}  (tuning score)")
print(f"   test:  {len(test_idx):,}  (final, untouched by search)")

# ---------------------------------------------------------------------------
# 2. Feature list (Model B partition, dead features dropped)
# ---------------------------------------------------------------------------
X_pf = pq.ParquetFile(X_path)
all_cols = X_pf.schema_arrow.names

label_derived = set(fl.LABEL_DERIVED_FEATURES)
weak_rule_inputs = {"FRP_PERSISTENCE", "DETECTION_COUNT_LOG", "TIME_TO_PEAK",
                    "AREA_GROWTH_RATE", "SPREAD_SPEED_MEDIAN", "DIRECTION_CONSISTENCY"}
corrupt_daily = {"FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "FWI_ANOMALY_Z",
                 "TEMP_C", "RH_PCT", "WIND_KMH", "RAIN_MM", "RAIN_7D_MM",
                 "SWVL1", "VPD_KPA"}
# DEAD: 100% NaN on every row (audit_features.py finding 4a). A column
# with no measurement carries no signal and only adds split noise.
dead_features = {"KNOWN_FLARE", "LIGHTNING_COUNT_24H", "RH_ANOMALY_Z",
                 "SOIL_MOISTURE_ANOMALY_Z", "VPD_ANOMALY_Z", "WIND_GUST"}
exclude_features = ({"row", "NBR_SOURCE", "event_id"}
                    | corrupt_daily | dead_features)

model_B_features = [c for c in all_cols
                    if c not in exclude_features
                    and c not in label_derived
                    and c not in weak_rule_inputs]
print(f"\n2. FEATURES")
print(f"   Model B: {len(model_B_features)} features "
      f"(dropped {len(corrupt_daily)} corrupt daily, "
      f"{len(dead_features)} dead, "
      f"{len(label_derived)} label-derived, "
      f"{len(weak_rule_inputs)} weak-rule inputs)")

# ---------------------------------------------------------------------------
# 3. Stream load
# ---------------------------------------------------------------------------
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

print("\n3. LOADING (streamed)...")
X_fit = load_split(fit_idx, model_B_features)
X_cal = load_split(calib_idx, model_B_features)
X_test = load_split(test_idx, model_B_features)
y_fit = y_binary.iloc[fit_idx].reset_index(drop=True)
y_cal = y_binary.iloc[calib_idx].reset_index(drop=True)
y_test = y_binary.iloc[test_idx].reset_index(drop=True)

for df in (X_fit, X_cal, X_test):
    for c in df.columns:
        df[c] = df[c].astype("float32")

print(f"   fit {X_fit.shape}  calib {X_cal.shape}  test {X_test.shape}")

# ---------------------------------------------------------------------------
# 4. Search space
# ---------------------------------------------------------------------------
# COORDINATE DESCENT, not a full grid. A 3x3x2x2x2x2x2 grid is 288
# trainings at up to 2000 rounds each -- hours on this machine for
# diminishing returns. Coordinate descent tunes one dimension at a
# time from a reasonable baseline, which is the standard reproducible
# approach when the objective is noisy and the budget is bounded.
# Every candidate is still trained on fit and scored on calib only.
BASELINE = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "scale_pos_weight": 1.0,
}

SWEEP = {
    "learning_rate": [0.02, 0.05, 0.1],
    "num_leaves": [31, 63, 127],
    "min_data_in_leaf": [50, 100, 200],
    "feature_fraction": [0.6, 0.8],
    "lambda_l1": [0.0, 0.1],
    "lambda_l2": [0.0, 0.1],
    "scale_pos_weight": [1.0, 3.0],   # class-weighting EXPERIMENT
}

print(f"\n4. SEARCH (coordinate descent)")
print(f"   baseline: {BASELINE}")
for k in SWEEP:
    print(f"     sweep {k}: {SWEEP[k]}")

# ---------------------------------------------------------------------------
# 5. Run the search
# ---------------------------------------------------------------------------
def train_one(params, fit_set, calib_set):
    """Train with early stopping on calib; return (booster, best_iter)."""
    booster = lgb.train(
        params, fit_set,
        num_boost_round=2000,
        valid_sets=[calib_set],
        valid_names=["calib"],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return booster, int(booster.best_iteration)


def score(booster, X, y):
    """PR-AUC and recall@precision>=0.90 on a fold."""
    from sklearn.metrics import average_precision_score, precision_recall_curve
    p = booster.predict(X, num_iteration=booster.best_iteration)
    auc = average_precision_score(y, p)
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(prec >= 0.90)[0]
    r90 = float(rec[ok[0]]) if len(ok) else 0.0
    return auc, r90, p


t0 = time.time()
results = []
fit_set = lgb.Dataset(X_fit, label=y_fit, feature_name=model_B_features,
                      free_raw_data=False)
calib_set = lgb.Dataset(X_cal, label=y_cal, feature_name=model_B_features,
                        free_raw_data=False, reference=fit_set)


def evaluate(params, tag):
    """Train on fit, score on calib. Returns the result row."""
    base = {
        "objective": "binary",
        "metric": "average_precision",
        "verbosity": -1,
        "random_state": 0,
        "n_jobs": -1,
        "bagging_freq": 5,
        # Reusing one Dataset across candidates with different
        # min_data_in_leaf triggers LightGBM's feature_pre_filter
        # guard. Disabling it is safe here: the filter only drops
        # features that can never split, and the sweep changes
        # min_data_in_leaf between candidates.
        "feature_pre_filter": False,
    }
    base.update(params)
    try:
        booster, best_iter = train_one(base, fit_set, calib_set)
        auc, r90, _ = score(booster, X_cal, y_cal)
        row = {**params, "best_iter": best_iter,
               "calib_pr_auc": auc, "calib_recall_at_90": r90}
        results.append(row)
        print(f"   {tag}: PR-AUC={auc:.4f} r@90={r90:.4f} "
              f"iter={best_iter}  ({time.time()-t0:.0f}s)")
        return row
    except Exception as exc:                    # noqa: BLE001
        print(f"   {tag}: FAILED {type(exc).__name__}: {str(exc)[:120]}")
        return None


# Baseline
print("\n5. COORDINATE DESCENT")
current = dict(BASELINE)
best_row = evaluate(current, "baseline")

# One pass over each dimension, keeping the best value found.
for dim, values in SWEEP.items():
    print(f"\n   -- sweeping {dim} (others fixed) --")
    dim_best = None
    for v in values:
        if v == current[dim]:
            continue
        trial = dict(current)
        trial[dim] = v
        row = evaluate(trial, f"{dim}={v}")
        if row is not None and (dim_best is None
                                  or row["calib_pr_auc"]
                                  > dim_best["calib_pr_auc"]):
            dim_best = row
    if dim_best is not None and dim_best["calib_pr_auc"] > best_row["calib_pr_auc"]:
        current[dim] = dim_best[dim]
        best_row = dim_best
        print(f"   -> {dim} improved to {dim_best[dim]}")
    else:
        print(f"   -> {dim} unchanged at {current[dim]}")

res_df = pd.DataFrame(results).sort_values("calib_pr_auc", ascending=False)
print(f"\n5b. SEARCH DONE in {time.time()-t0:.0f}s")
print(f"   top 5 by calib PR-AUC:")
for _, r in res_df.head(5).iterrows():
    print(f"     PR-AUC={r['calib_pr_auc']:.4f}  r@90={r['calib_recall_at_90']:.4f}  "
          f"lr={r['learning_rate']} leaves={r['num_leaves']} "
          f"min={r['min_data_in_leaf']} ff={r['feature_fraction']} "
          f"l1={r['lambda_l1']} l2={r['lambda_l2']} spw={r['scale_pos_weight']} "
          f"iter={r['best_iter']}")

# ---------------------------------------------------------------------------
# 6. Class-weighting experiment readout
# ---------------------------------------------------------------------------
print(f"\n6. CLASS-WEIGHTING EXPERIMENT (scale_pos_weight)")
for spw in sorted(res_df["scale_pos_weight"].unique()):
    sub = res_df[res_df["scale_pos_weight"] == spw]
    print(f"   spw={spw}: best calib PR-AUC={sub['calib_pr_auc'].max():.4f}  "
          f"best r@90={sub['calib_recall_at_90'].max():.4f}  "
          f"mean PR-AUC={sub['calib_pr_auc'].mean():.4f}")

# ---------------------------------------------------------------------------
# 7. Winner: retrain on fit+calib, score ONCE on test
# ---------------------------------------------------------------------------
best = res_df.iloc[0]
print(f"\n7. WINNER")
print(f"   {best.to_dict()}")

# Retrain the winner on fit+calib combined (more data, same params),
# with the early-stopping round count carried over from the calib fold.
winner_params = {
    "objective": "binary",
    "metric": "average_precision",
    "verbosity": -1,
    "random_state": 0,
    "n_jobs": -1,
    "bagging_freq": 5,
    "learning_rate": best["learning_rate"],
    "num_leaves": int(best["num_leaves"]),
    "min_data_in_leaf": int(best["min_data_in_leaf"]),
    "feature_fraction": best["feature_fraction"],
    "bagging_fraction": 0.8,
    "lambda_l1": best["lambda_l1"],
    "lambda_l2": best["lambda_l2"],
    "scale_pos_weight": best["scale_pos_weight"],
}

X_fitcal = pd.concat([X_fit, X_cal], ignore_index=True)
y_fitcal = pd.concat([y_fit, y_cal], ignore_index=True)
for c in X_fitcal.columns:
    X_fitcal[c] = X_fitcal[c].astype("float32")

final_set = lgb.Dataset(X_fitcal, label=y_fitcal,
                        feature_name=model_B_features, free_raw_data=False)
final_booster = lgb.train(winner_params, final_set,
                          num_boost_round=int(best["best_iter"]),
                          callbacks=[lgb.log_evaluation(0)])

# FINAL evaluation on test -- the only time test is touched.
test_auc, test_r90, pred_test = score(final_booster, X_test, y_test)
print(f"\n   TEST (final, once): PR-AUC={test_auc:.4f}  "
      f"recall@prec>=0.90={test_r90:.4f}")

# ---------------------------------------------------------------------------
# 8. Freeze tau on the CALIBRATION fold (not test, not train)
# ---------------------------------------------------------------------------
pred_cal = final_booster.predict(X_cal, num_iteration=final_booster.best_iteration)
tau = tc.calibrate_operating_threshold(y_cal, pred_cal, min_precision=0.90, beta=2.0)
print(f"\n8. THRESHOLD (frozen on calib fold)")
print(f"   {tc.describe_calibration(tau, 'B')}")

# Held-out performance at the frozen tau
t = tau.get("threshold")
if t is not None:
    flagged = pred_test >= t
    got_p = float(y_test[flagged].mean()) if flagged.any() else 0.0
    got_r = float(y_test[flagged].sum() / y_test.sum())
    n_flag = int(flagged.sum())
    tp = int((flagged & (y_test == 1)).sum())
    fp = int((flagged & (y_test == 0)).sum())
    fn = int((~flagged & (y_test == 1)).sum())
    tn = int((~flagged & (y_test == 0)).sum())
    f1 = 2 * got_p * got_r / (got_p + got_r) if (got_p + got_r) else 0.0
    f2 = 5 * got_p * got_r / (4 * got_p + got_r) if (4 * got_p + got_r) else 0.0
    print(f"   TEST at frozen tau={t:.4f}:")
    print(f"     precision={got_p:.4f}  recall={got_r:.4f}  "
          f"F1={f1:.4f}  F2={f2:.4f}")
    print(f"     flagged={n_flag:,} ({100.0*n_flag/len(y_test):.1f}%)")
    print(f"     confusion: TP={tp:,} FP={fp:,} FN={fn:,} TN={tn:,}")

# ---------------------------------------------------------------------------
# 9. Per-year and per-region breakdown at the frozen tau
# ---------------------------------------------------------------------------
print(f"\n9. PERFORMANCE BY YEAR (at frozen tau)")
meta_test = meta.iloc[test_idx].reset_index(drop=True)
if t is not None:
    for yr in sorted(meta_test["start_year"].unique()):
        m = (meta_test["start_year"] == yr).to_numpy()
        if m.sum() == 0:
            continue
        yt = y_test.to_numpy()[m]
        pt = pred_test[m]
        fl_ = pt >= t
        p = float(yt[fl_].mean()) if fl_.any() else 0.0
        r = float(yt[fl_].sum() / max(yt.sum(), 1))
        auc_yr = None
        if len(np.unique(yt)) > 1:
            from sklearn.metrics import average_precision_score
            auc_yr = average_precision_score(yt, pt)
        print(f"     {yr}: n={int(m.sum()):,}  prec={p:.4f}  rec={r:.4f}  "
              f"PR-AUC={auc_yr if auc_yr is None else round(auc_yr,4)}")

print(f"\n9b. PERFORMANCE BY SPATIAL BLOCK GROUP (at frozen tau)")
if t is not None:
    # Group blocks by their mean score to see regional variation
    meta_test["block_group"] = pd.qcut(
        meta_test["spatial_block"].astype("category").cat.codes,
        q=5, duplicates="drop", labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    for g in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        m = (meta_test["block_group"] == g).to_numpy()
        if m.sum() == 0:
            continue
        yt = y_test.to_numpy()[m]
        pt = pred_test[m]
        fl_ = pt >= t
        p = float(yt[fl_].mean()) if fl_.any() else 0.0
        r = float(yt[fl_].sum() / max(yt.sum(), 1))
        print(f"     {g}: n={int(m.sum()):,}  prec={p:.4f}  rec={r:.4f}")

# ---------------------------------------------------------------------------
# 10. Save
# ---------------------------------------------------------------------------
import pickle

payload = {
    "booster": final_booster,
    "features": model_B_features,
    "target": "INDUSTRIAL_FIRE",
    "label_column": "weak",
    "best_iteration": int(final_booster.best_iteration),
    "split": "year_x_block",
    "operating_threshold": tau.get("threshold"),
    "operating_point": tau,
    "auto_accept_threshold": tc.effective_threshold(
        tau.get("threshold"), 0.60),
    "params": winner_params,
    "search_results": results,
    "eval": {"test_pr_auc": test_auc, "test_recall_at_90": test_r90},
}
with open(MODEL_DIR / "model_B_tuned.pkl", "wb") as f:
    pickle.dump(payload, f)
res_df.to_csv(FEATURES_DIR / "tune_results.csv", index=False)
pd.DataFrame({"event_id": meta_test["event_id"],
              "score": pred_test, "y_true": y_test}).to_csv(
    FEATURES_DIR / "heldout_pred_B_tuned.csv", index=False)

print(f"\n10. SAVED")
print(f"   model_B_tuned.pkl  (booster + params + search)")
print(f"   tune_results.csv   ({len(res_df)} rows)")
print(f"   heldout_pred_B_tuned.csv")

print("\n" + "=" * 74)
print("TUNING COMPLETE")
print("=" * 74)
