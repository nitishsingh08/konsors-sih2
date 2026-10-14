# ============================================================================
# P9 -- BASELINE-vs-ABLATION TABLE
#
# One script, one split, one test fold. Every variant is trained
# on the SAME fit rows and scored on the SAME test rows, so the
# numbers are directly comparable. The test fold is touched once
# per variant, after the variant is fully specified.
#
# VARIANTS (the only difference between them is noted):
#   A  baseline        leaky per-partition site features (current)
#   B  safe-site       past-only TRUE site history (inference-safe)
#   C  safe-site+reg   B + light regularization (lambda_l2=0.1)
#
# All use default LightGBM except where noted, 1500 rounds,
# full fit fold -- matching local_retrain.py so the baseline
# number reproduces.
#
# Run:  python3 ablation_table.py
# ============================================================================
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, precision_recall_curve

sys.path.insert(0, "/Users/aashutosh/konsors_sih2/konsors-sih2")

import feature_ledger as fl
import training_design as td
import threshold_calibration as tc

FEATURES_DIR = Path("/Users/aashutosh/konsors_sih2/konsors-sih2/Features_mapped")

print("=" * 74)
print("P9 -- BASELINE-vs-ABLATION TABLE")
print("=" * 74)

# ---------------------------------------------------------------------------
# 1. Split (identical to local_retrain, so the baseline reproduces)
# ---------------------------------------------------------------------------
meta = pd.read_parquet(FEATURES_DIR / "meta.parquet",
                         columns=["event_id", "event_centroid_lon",
                                  "event_centroid_lat", "start_time",
                                  "start_year", "spatial_block", "site_id"])
meta = meta.reset_index().rename(columns={"index": "row"})
y = pd.read_parquet(FEATURES_DIR / "y.parquet", columns=["label_weak"])
y.columns = ["label_weak"]
y = y.reset_index().rename(columns={"index": "row"})
y_binary = (y["label_weak"] == "INDUSTRIAL_FIRE").astype("int8")

meta = td.add_split_keys(meta)
parts = td.make_split(meta, kind="year_x_block", seed=0)
train_idx = parts["train"]["row"].to_numpy()
test_idx = parts["test"]["row"].to_numpy()

print(f"\n1. SPLIT  train={len(train_idx):,}  test={len(test_idx):,}")

# ---------------------------------------------------------------------------
# 2. Feature partitions (the leakage rules, identical for all variants)
# ---------------------------------------------------------------------------
label_derived = set(fl.LABEL_DERIVED_FEATURES)
weak_rule_inputs = {"FRP_PERSISTENCE", "DETECTION_COUNT_LOG", "TIME_TO_PEAK",
                    "AREA_GROWTH_RATE", "SPREAD_SPEED_MEDIAN",
                    "DIRECTION_CONSISTENCY"}
corrupt_daily = {"FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "FWI_ANOMALY_Z",
                 "TEMP_C", "RH_PCT", "WIND_KMH", "RAIN_MM", "RAIN_7D_MM",
                 "SWVL1", "VPD_KPA"}
dead_features = {"KNOWN_FLARE", "LIGHTNING_COUNT_24H", "RH_ANOMALY_Z",
                 "SOIL_MOISTURE_ANOMALY_Z", "VPD_ANOMALY_Z", "WIND_GUST"}
# Leaky per-partition site features: replaced by *_PAST in variant B.
leaky_site = {"SITE_FRP_BASELINE", "SITE_N_EVENTS",
              "FRP_ANOMALY_VS_SITE", "FRP_ANOMALY_LOG"}

base_exclude = ({"row", "NBR_SOURCE", "event_id"}
                | corrupt_daily | dead_features)


def feature_list(path, extra_exclude=frozenset()):
    cols = pq.ParquetFile(path).schema_arrow.names
    return [c for c in cols
            if c not in base_exclude
            and c not in extra_exclude
            and c not in label_derived
            and c not in weak_rule_inputs]


# ---------------------------------------------------------------------------
# 3. Stream-load a variant's matrix for a set of rows
# ---------------------------------------------------------------------------
def load(path, rows, features):
    pf = pq.ParquetFile(path)
    chunks = []
    row_set = set(rows)
    for rg in range(pf.num_row_groups):
        d = pf.read_row_group(rg, columns=["row"] + features).to_pandas()
        m = d["row"].isin(row_set)
        if m.any():
            chunks.append(d[m].drop(columns=["row"]))
    if chunks:
        out = pd.concat(chunks, ignore_index=True)
    else:
        out = pd.DataFrame(columns=features)
    for c in out.columns:
        out[c] = out[c].astype("float32")
    return out


def evaluate(name, X_path, features, params, rounds=1500):
    """Train on fit, score once on test. Returns the result row."""
    t0 = time.time()
    X_train = load(X_path, train_idx, features)
    X_test = load(X_path, test_idx, features)
    y_tr = y_binary.iloc[train_idx].reset_index(drop=True)
    y_te = y_binary.iloc[test_idx].reset_index(drop=True)

    booster = lgb.train(
        params, lgb.Dataset(X_train, label=y_tr,
                            feature_name=features, free_raw_data=False),
        num_boost_round=rounds,
        callbacks=[lgb.log_evaluation(0)],
    )
    pred_te = booster.predict(X_test)
    auc = average_precision_score(y_te, pred_te)
    prec, rec, thr = precision_recall_curve(y_te, pred_te)
    ok = np.where(prec >= 0.90)[0]
    r90 = float(rec[ok[0]]) if len(ok) else 0.0

    # tau frozen on the TRAIN fold (deployment-realistic: no test peeking)
    pred_tr = booster.predict(X_train)
    tau = tc.calibrate_operating_threshold(y_tr, pred_tr,
                                            min_precision=0.90, beta=2.0)
    t = tau.get("threshold")
    flagged = pred_te >= t
    tp = int((flagged & (y_te == 1)).sum())
    fp = int((flagged & (y_te == 0)).sum())
    fn = int((~flagged & (y_te == 1)).sum())
    test_prec = tp / (tp + fp) if (tp + fp) else 0.0
    test_rec = tp / (tp + fn) if (tp + fn) else 0.0

    row = {
        "variant": name,
        "n_features": len(features),
        "test_pr_auc": auc,
        "test_recall_at_90": r90,
        "tau": t,
        "test_precision_at_tau": test_prec,
        "test_recall_at_tau": test_rec,
        "flag_fraction": float(flagged.mean()),
        "rounds": rounds,
        "seconds": round(time.time() - t0, 1),
    }
    print(f"   {name:<16} PR-AUC={auc:.4f}  r@90={r90:.4f}  "
          f"tau={t:.4f}  prec@tau={test_prec:.4f}  "
          f"rec@tau={test_rec:.4f}  flags={100.0*flagged.mean():.1f}%  "
          f"({row['seconds']:.0f}s)")
    return row, booster, features


# Default params matching local_retrain.py
DEFAULT = {
    "objective": "binary",
    "metric": "average_precision",
    "verbosity": -1,
    "random_state": 0,
    "n_jobs": -1,
    "bagging_freq": 5,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
}

X_LEAKY = FEATURES_DIR / "X_with_p7_full.parquet"
X_SAFE = FEATURES_DIR / "X_with_p7_safe.parquet"

print("\n2. VARIANTS")
results = []

# A. Baseline: leaky per-partition site features (current Model B)
feat_A = feature_list(X_LEAKY)
print(f"\n   A baseline: {len(feat_A)} features (leaky per-partition site)")
row_A, booster_A, feat_A = evaluate("A baseline", X_LEAKY, feat_A, DEFAULT)
results.append(row_A)

# B. Safe-site: past-only TRUE site history (inference-safe)
#    Drop the leaky site features, keep the *_PAST versions.
feat_B = feature_list(X_SAFE, extra_exclude=leaky_site)
print(f"\n   B safe-site: {len(feat_B)} features (past-only true site history)")
row_B, booster_B, feat_B = evaluate("B safe-site", X_SAFE, feat_B, DEFAULT)
results.append(row_B)

# C. Safe-site + light regularization (the r@90-friendly direction
#    the tuning search pointed to: lambda_l2=0.1)
params_C = dict(DEFAULT)
params_C["lambda_l2"] = 0.1
print(f"\n   C safe-site+reg: {len(feat_B)} features (B + lambda_l2=0.1)")
row_C, booster_C, feat_C = evaluate("C safe-site+reg", X_SAFE, feat_B,
                                     params_C)
results.append(row_C)

# ---------------------------------------------------------------------------
# 4. The table
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("BASELINE-vs-ABLATION TABLE (same split, same test fold)")
print("=" * 74)
df = pd.DataFrame(results)
print(df.to_string(index=False))

print("\nDELTAS vs baseline (A):")
for _, r in df.iterrows():
    if r["variant"] == "A baseline":
        continue
    print(f"   {r['variant']:<16} dPR-AUC={r['test_pr_auc']-row_A['test_pr_auc']:+.4f}  "
          f"dr@90={r['test_recall_at_90']-row_A['test_recall_at_90']:+.4f}  "
          f"drec@tau={r['test_recall_at_tau']-row_A['test_recall_at_tau']:+.4f}")

# ---------------------------------------------------------------------------
# 5. SHAP on the safe-site model (the deployable one)
# ---------------------------------------------------------------------------
print("\n5. SHAP top-10 (variant B, safe-site -- the deployable model)")
try:
    import shap
    explainer = shap.TreeExplainer(booster_B)
    # Sample 2000 test rows for a stable global ranking
    rng = np.random.default_rng(0)
    sample = rng.choice(len(X_test_sampled := load(X_SAFE, test_idx, feat_B)),
                        size=min(2000, len(X_test_sampled)), replace=False)
    sv = explainer.shap_values(X_test_sampled.iloc[sample])
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:10]
    for i in order:
        print(f"   {feat_B[i]:<28} {mean_abs[i]:.4f}")
except Exception as exc:                    # noqa: BLE001
    print(f"   SHAP skipped: {type(exc).__name__}: {str(exc)[:120]}")

# ---------------------------------------------------------------------------
# 6. Save the deployable model (variant B)
# ---------------------------------------------------------------------------
import pickle

payload = {
    "booster": booster_B,
    "features": feat_B,
    "target": "INDUSTRIAL_FIRE",
    "label_column": "weak",
    "split": "year_x_block",
    "site_features": "past-only (inference-safe)",
    "params": DEFAULT,
    "rounds": 1500,
    "eval": row_B,
}
with open(FEATURES_DIR / "models" / "model_B_safe.pkl", "wb") as f:
    pickle.dump(payload, f)
df.to_csv(FEATURES_DIR / "ablation_table.csv", index=False)

print("\n6. SAVED")
print("   models/model_B_safe.pkl  (deployable, inference-safe)")
print("   ablation_table.csv")

print("\n" + "=" * 74)
print("ABLATION COMPLETE")
print("=" * 74)
