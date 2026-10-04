# ============================================================================
# P7 -- COLAB CELL: train and evaluate
#
# Consumes X / y / meta from P6 and reports what the brief asks for:
# precision, recall and F1 PER CLASS, on regions the model has never seen.
#
# Two models are always reported, because reporting only one of them is
# misleading in opposite directions:
#
#   model A  all available features, including the 18 label-derived industrial
#            ones. This is the operational model and it leans on facility
#            priors, so it is labelled as such.
#   model B  the non-leak partition only. This is the scientific claim.
#
# The distance-threshold baseline sits beside both. If neither model beats it,
# that is the finding.
# ============================================================================

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import colab_utils as cu
import dataset_builder as db
import feature_ledger as fl
import fire_labels as lab
import training_design as td
import weak_labels as weak

try:
    import lightgbm as lgb
except ImportError:
    print("=" * 74)
    print("LightGBM is not installed in this runtime.")
    print("Run:  !pip install -q lightgbm")
    print("Nothing below can proceed without it.")
    raise SystemExit(1)

DATASET_DIR = None   # filled in below from the resolved project root

print("=" * 74)
print(f"ledger {fl.LEDGER_VERSION}")
print("=" * 74)
print(cu.drive_report())

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError(f"no project root under {cu.MYDRIVE}")
DATASET_DIR = PROJECT_ROOT / "full_2012_2026" / "features" / "dataset"
QC_DIR = PROJECT_ROOT / "qc"

for name in ("X.parquet", "y.parquet", "meta.parquet"):
    if not cu.path_ok(DATASET_DIR / name):
        raise FileNotFoundError(
            f"{DATASET_DIR / name} is missing. Run colab_p6_dataset.py first."
        )

print(f"dataset: {DATASET_DIR}")

# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1. LOAD -- sampled, float32
#
# The first attempt loaded all 7,265,849 rows as float64 and then handed the
# frame to LightGBM, which copies it into its own binary matrix: roughly 3.5 GB
# for the frame plus another 3.5 GB inside LightGBM. That exhausted the runtime.
#
# Three fixes:
#   * read only the columns actually used
#   * cast to float32 -- half the bytes, and LightGBM wants float32 anyway
#   * train on a stratified sample, reported explicitly rather than silently
# ---------------------------------------------------------------------------

MAX_TRAIN_ROWS = int(globals().get("MAX_TRAIN_ROWS", 1_500_000))
MAX_TEST_ROWS = int(globals().get("MAX_TEST_ROWS", 600_000))

t0 = time.time()

# --- read ONLY what is needed, and stream X --------------------------------
# Two earlier versions exhausted the runtime. Both read the WHOLE table and
# only sampled afterwards, which is backwards: sampling cannot save memory that
# has already been spent. 7,265,849 x 60 is 1.6 GB as float32, and the four
# string columns in y plus three in meta roughly double that once pandas'
# object arrays are counted.
#
# So: read y and meta with the columns actually needed, decide the split and
# the sample from those small tables, then stream X one row group at a time
# keeping ONLY the sampled rows. Peak becomes the sample, not the catalogue.
t0 = time.time()

_y_cols = set(pq.ParquetFile(DATASET_DIR / "y.parquet").schema_arrow.names)
_meta_cols = set(
    pq.ParquetFile(DATASET_DIR / "meta.parquet").schema_arrow.names
)

y = pd.read_parquet(
    DATASET_DIR / "y.parquet",
    columns=[c for c in ("label_weak", "label", "sample_weight")
             if c in _y_cols],
)
meta = pd.read_parquet(
    DATASET_DIR / "meta.parquet",
    columns=[c for c in ("event_id", "event_centroid_lon", "event_centroid_lat",
                         "start_time", "start_year")
             if c in _meta_cols],
)
print(f"loaded y {y.shape}  meta {meta.shape}  ({time.time() - t0:.1f}s)")
print("  y columns kept    :", list(y.columns))
print("  meta columns kept :", list(meta.columns))

# Hard gate on the feature NAMES, straight from the parquet footer, so it runs
# before a single row is read.
_X_COLS = list(pq.ParquetFile(DATASET_DIR / "X.parquet").schema_arrow.names)
db.assert_no_location_features(_X_COLS)
print("location check: PASS")
print(f"  X has {len(_X_COLS)} columns (from the footer, not a read)")

# Which label column to train on.
#
#   "weak"    the industrial-context hierarchy in `weak_labels`. It labels a
#             fifth of events instead of none, but it is MANUFACTURED from
#             features the model also sees, so every metric on it is optimistic
#             by construction. Model B excludes every rule input.
#   "strict"  the evidence-gated engine. Almost entirely UNKNOWN, so it cannot
#             train yet -- it needs the land-cover and weather groups.
LABEL_COLUMN = "weak" if "label_weak" in y.columns else "strict"

labels = (y["label_weak"] if LABEL_COLUMN == "weak" else y["label"]).to_numpy()
weights = (y["sample_weight"].to_numpy() if "sample_weight" in y.columns
           else np.ones(len(labels), dtype="float64"))

# `X` does not exist yet. It is filled AFTER the sample is chosen, straight
# into preallocated float32 arrays, one row group at a time. NaN is preserved,
# which is essential: NaN is how "could not measure" is encoded.

# Which class to fit. INDUSTRIAL_FLARE has only 3,571 events in the whole
# catalogue -- 894 in the training sample and 33 in the test set -- so its
# metrics are dominated by sampling noise and both models early-stopped at
# iteration 3. INDUSTRIAL_FIRE has 1,119,254 and is the first target that can
# actually support a model. Change this to compare.
TARGET = globals().get("TARGET", "INDUSTRIAL_FLARE")
print()
print(f"label column : {LABEL_COLUMN}")
if LABEL_COLUMN == "weak":
    print("  WEAK LABELS. These are manufactured from features the model sees,")
    print("  so PR-AUC here is an upper bound, NOT a generalisation estimate.")
    print("  Only an independent hand-labelled sample can give the latter.")
else:
    print("  strict labels -- nearly all UNKNOWN, so training cannot proceed.")
print()
print(f"target class : {TARGET}")
print("class counts (full catalogue):")
for cls in weak.WEAK_CLASSES:
    n = int((labels == cls).sum())
    print(f"  {cls:20s} {n:>10,}  {100.0 * n / max(len(labels), 1):>6.2f}%")

# ---------------------------------------------------------------------------
# 2. SPLIT
# ---------------------------------------------------------------------------

meta = td.add_split_keys(meta)
split_kind = td.SPLIT_COMBINED
parts = td.make_split(meta, kind=split_kind, seed=0)
print(f"  split kind: {split_kind}")
train_ids = set(parts["train"]["event_id"])
test_ids = set(parts["test"]["event_id"])

is_train = meta["event_id"].isin(train_ids).to_numpy()
is_test = meta["event_id"].isin(test_ids).to_numpy()
dropped = ~(is_train | is_test)

print()
print("=" * 74)
print("SPLIT -- out-of-time AND out-of-space")
print("=" * 74)
print(f"  train : {is_train.sum():>10,}")
print(f"  test  : {is_test.sum():>10,}")
print(f"  dropped (early events inside held-out blocks): {dropped.sum():>10,}")
td.assert_no_spatial_leakage(meta[is_train], meta[is_test])
print("  spatial leakage check: PASS (no shared block)")
if split_kind == td.SPLIT_OUT_OF_TIME:
    td.assert_no_time_leakage(meta[is_train], meta[is_test])
    print("  temporal leakage check: PASS (no test year in train)")
else:
    print("  temporal check: N/A -- the combined split trains on late events")
    print("    from non-held-out blocks by design; its guarantee is spatial.")

y_bin = (labels == TARGET).astype(int)
print(f"  positives in train: {int(y_bin[is_train].sum()):,}")
print(f"  positives in test : {int(y_bin[is_test].sum()):,}")
if y_bin[is_train].sum() == 0 or y_bin[is_test].sum() == 0:
    raise SystemExit(
        "One side of the split has no positives. Widen the seed or lower the "
        "test_start_year before interpreting any metric."
    )


# ---------------------------------------------------------------------------
# 2b. BOUND THE TRAINING SET
#
# LightGBM copies its input into a binary matrix, so a 7.27M x 60 float64 frame
# is roughly twice what the runtime has spare. Sampling is stratified on the
# TARGET so the minority class is not lost, and the sample size is printed so a
# metric is never mistaken for a full-catalogue result.
# ---------------------------------------------------------------------------

def stratified_sample(mask, cap, seed=0):
    """Indices from `mask`, at most `cap`, keeping the TARGET's prevalence."""
    idx = np.flatnonzero(mask)
    if len(idx) <= cap:
        return idx
    pos = idx[y_bin[idx] == 1]
    neg = idx[y_bin[idx] == 0]
    share = (len(pos) / len(idx)) if len(idx) else 0.0
    n_pos = min(len(pos), max(1, int(round(cap * share))))
    n_neg = min(len(neg), max(0, cap - n_pos))
    rng = np.random.default_rng(seed)
    take = np.concatenate([
        rng.choice(pos, n_pos, replace=False) if n_pos else np.empty(0, int),
        rng.choice(neg, n_neg, replace=False) if n_neg else np.empty(0, int),
    ])
    return np.sort(take)


train_idx = stratified_sample(is_train, MAX_TRAIN_ROWS, seed=0)
test_idx = stratified_sample(is_test, MAX_TEST_ROWS, seed=1)

print()
print("SAMPLE")
total_rows = pq.ParquetFile(DATASET_DIR / "X.parquet").metadata.num_rows
print(f"  train {len(train_idx):>9,} of {int(is_train.sum()):>10,}  "
      f"({100.0 * len(train_idx) / max(int(is_train.sum()), 1):.1f}%)")
print(f"  test  {len(test_idx):>9,} of {int(is_test.sum()):>10,}  "
      f"({100.0 * len(test_idx) / max(int(is_test.sum()), 1):.1f}%)")
print(f"  train positives {int(y_bin[train_idx].sum()):,}")
print(f"  test positives  {int(y_bin[test_idx].sum()):,}")
_full_gb = len(_X_COLS) * 4 * total_rows / 1024 ** 3
print(f"  a full float32 frame would be {_full_gb:.2f} GB; streaming avoids it")



# --- the partition must be the CURRENT one -----------------------------------
# A stale training_design.py computed model B as NON_LEAK - industrial, which
# leaves six weak-rule inputs (FRP_PERSISTENCE, DETECTION_COUNT_LOG,
# TIME_TO_PEAK, AREA_GROWTH_RATE, SPREAD_SPEED_MEDIAN, DIRECTION_CONSISTENCY)
# in the "scientific" model -- and those are the INDUSTRIAL_FIRE rule's own
# shape branch. That run reported 42 features and PR-AUC 0.527, both of which
# were artefacts. Fail loudly instead.
_model_a_cols, _model_b_cols, _leak = td.split_feature_sets(_X_COLS)
_expected_b = len(_model_b_cols)
_still_leaking = sorted(set(_model_b_cols) & set(weak.LABEL_INPUT_FEATURES))
if _still_leaking:
    raise SystemExit(
        "STALE training_design.py or weak_labels.py: these weak-rule inputs "
        f"are in model B: {_still_leaking}.\n"
        "Re-upload training_design.py and weak_labels.py, then re-run."
    )
print(f"  partition check: model A {len(_model_a_cols)}  model B "
      f"{_expected_b}  (rule inputs excluded: {len(_leak['excluded_from_b'])})")

non_leak = list(_model_b_cols)
label_derived = [c for c in _X_COLS if c in td.LABEL_DERIVED]
# ---------------------------------------------------------------------------
# 3. STREAM X -- keep only the sampled rows
#
# The catalogue is walked one row group at a time and only the sampled
# positions are retained, so peak memory is the SAMPLE rather than 7.27M rows.
# A full 60-column float32 frame is 1.6 GB; the sample is about 0.4 GB, and
# LightGBM's copy of it is smaller again.
# ---------------------------------------------------------------------------

_all = np.concatenate([train_idx, test_idx])
_all_sorted = np.sort(_all)
_keep_rows = _all_sorted

print()
print("streaming X, keeping only the sampled rows ...")
X_sampled = np.empty((len(_keep_rows), len(_X_COLS)), dtype="float32")
_filled = 0
_t0 = time.time()
_pf = pq.ParquetFile(DATASET_DIR / "X.parquet")
pf_rows = _pf.metadata.num_rows
_offset = 0
for _batch in _pf.iter_batches(batch_size=250_000, columns=_X_COLS):
    _n = _batch.num_rows
    _start = _offset
    _end = _offset + _n
    _lo = np.searchsorted(_keep_rows, _start)
    _hi = np.searchsorted(_keep_rows, _end)
    if _hi > _lo:
        local = _keep_rows[_lo:_hi] - _start
        block = _batch.to_pandas().to_numpy(dtype="float32", copy=True)
        X_sampled[_lo:_hi] = block[local]
        _filled += _hi - _lo
        del block
    _offset = _end
    if _offset % 1_000_000 < _n:
        print(f"    {_offset:>10,} / {pf_rows:,} rows  "
              f"({_filled:,} kept, {time.time() - _t0:.0f}s)")

print(f"  filled {_filled:,} of {len(_keep_rows):,} requested")
if _filled != len(_keep_rows):
    raise SystemExit(
        f"streaming fill got {_filled:,} rows, expected {len(_keep_rows):,}. "
        "The sample indices and the parquet row order must agree."
    )

_train_local = np.searchsorted(_keep_rows, train_idx)
_test_local = np.searchsorted(_keep_rows, test_idx)
_b_cols = [_X_COLS.index(c) for c in _model_b_cols]
X_train_b = np.ascontiguousarray(X_sampled[_train_local][:, _b_cols])
X_test_b = np.ascontiguousarray(X_sampled[_test_local][:, _b_cols])
X_train_a = np.ascontiguousarray(X_sampled[_train_local])
X_test_a = np.ascontiguousarray(X_sampled[_test_local])
del X_sampled

_mb = X_train_b.nbytes / 1024 ** 3
_ma = X_train_a.nbytes / 1024 ** 3
print(f"  model A train matrix {_ma:.2f} GB   model B train matrix {_mb:.2f} GB")
print("  (the 7.27M-row catalogue was never resident)")


# ---------------------------------------------------------------------------
# 3. METRIC HELPERS -- per class, per the brief
# ---------------------------------------------------------------------------


def per_class_report(y_true, y_pred, classes):
    """Precision / recall / F1 per class. The brief asks for these explicitly."""
    rows = []
    for cls in classes:
        t = y_true == cls
        p = y_pred == cls
        tp = int(np.sum(t & p))
        fp = int(np.sum(~t & p))
        fn = int(np.sum(t & ~p))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        rows.append(
            {"class": cls, "support": int(t.sum()), "tp": tp, "fp": fp,
             "fn": fn, "precision": precision, "recall": recall, "f1": f1}
        )
    return pd.DataFrame(rows)


def fit_and_score(X_train, y_train, w_train, X_test, y_test, name):
    """Train one LightGBM model and score it. Returns (model, y_prob)."""
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "num_threads": -1,
        # NaN is meaningful here: 96.7% of events are single-cluster and 19 of
        # the 52 features are undefined for them. LightGBM routes NaN down a
        # default direction, which is exactly what "could not measure" should
        # do -- and is why no zero-filling is applied anywhere upstream.
        "zero_as_missing": False,
    }
    train_set = lgb.Dataset(
        X_train, label=y_train, weight=w_train, free_raw_data=False
    )
    valid_set = lgb.Dataset(
        X_test, label=y_test, reference=train_set, free_raw_data=False
    )
    model = lgb.train(
        params, train_set, num_boost_round=800,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(75, verbose=False),
                   lgb.log_evaluation(100)],
    )
    prob = model.predict(X_test, num_iteration=model.best_iteration)
    print(f"  {name}: best_iteration={model.best_iteration}")
    return model, prob


results = {}
models = {}

# ---------------------------------------------------------------------------
# 4. MODEL B -- the scientific claim (non-leak only)
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print(f"model B (scientific): {len(non_leak)} non-leak features")
print(f"  excluded as label-derived: {label_derived}")
print("=" * 74)

model_b, prob_b = fit_and_score(
    X_train_b, y_bin[train_idx], weights[train_idx],
    X_test_b, y_bin[test_idx], "model B",
)
models["B"] = model_b
results["B"] = td.binary_metrics(y_bin[test_idx], (prob_b >= 0.5).astype(int))
results["B"]["recall_at_target"] = td.precision_recall_at_precision(
    y_bin[test_idx], prob_b, db.REVIEW_CONFIDENCE
)

# ---------------------------------------------------------------------------
# 5. MODEL A -- operational (all features, facility priors included)
# ---------------------------------------------------------------------------

print()
print("=" * 74)
print(f"model A (operational): all {len(_model_a_cols)} features")
print(f"  includes the {len(label_derived)} label-derived industrial features")
print("=" * 74)

model_a, prob_a = fit_and_score(
    X_train_a, y_bin[train_idx], weights[train_idx],
    X_test_a, y_bin[test_idx], "model A",
)
models["A"] = model_a
results["A"] = td.binary_metrics(y_bin[test_idx], (prob_a >= 0.5).astype(int))
results["A"]["recall_at_target"] = td.precision_recall_at_precision(
    y_bin[test_idx], prob_a, db.REVIEW_CONFIDENCE
)

# ---------------------------------------------------------------------------
# 6. BASELINE -- the thing ML has to beat
# ---------------------------------------------------------------------------

print()
print("=" * 74)
print("BASELINE -- single threshold on DIST_NEAREST_FACILITY_LOG")
print("=" * 74)

# Distance is only needed for the baseline, and only for the sampled
# rows, so it is read from the matrices already in memory.
_dist_col = _X_COLS.index("DIST_NEAREST_FACILITY_LOG")
_dist = np.concatenate([X_train_a[:, _dist_col], X_test_a[:, _dist_col]])
dist_m = np.where(np.isfinite(_dist), np.power(10.0, _dist) - 1.0, np.nan)
# Tuned on TRAIN only. Selecting the threshold on the test fold and then
# reporting that fold's score is one of the quietest ways to overstate.
baseline = td.tune_distance_baseline(
    _dist[:len(train_idx)], y_bin[train_idx]
)
print(f"  threshold : {baseline['threshold_m']:,.0f} m  "
      f"(tuned on train)")
print(f"  precision : {baseline['precision']:.4f}")
print(f"  recall    : {baseline['recall']:.4f}")
print(f"  F1        : {baseline['f1']:.4f}")
print(f"  meets precision >= {td.TARGET_PRECISION}: "
      f"{baseline['meets_precision_target']}")

# ---------------------------------------------------------------------------
# 7. REPORT
# ---------------------------------------------------------------------------

print()
print("=" * 74)
print("RESULTS -- held-out regions, 2020+")
print("=" * 74)

for key in ("B", "A"):
    m = results[key]
    print()
    print(f"model {key}: PR-AUC {m['pr_auc']:.4f}   "
          f"F1 {m['f1']:.4f}   recall@prec>={db.REVIEW_CONFIDENCE} "
          f"{m['recall_at_target']['recall']:.4f}")
    verdict = td.compare_to_baseline(m, baseline)
    print(f"  vs baseline: {verdict['verdict']}")

print()
print("=" * 74)
# The negative side is the set of ACTUAL other classes, not a synthetic name.
# An earlier version reported a class literally called "NOT_INDUSTRIAL_FIRE",
# which never appears in the labels, so its support printed as 0 and the row
# looked like a class the model had never seen. Fold every non-target label
# into one real negative class so its support is the true count.
NEGATIVE = "OTHER_CLASS"
_test_labels = labels[test_idx]
_true_binary = np.where(_test_labels == TARGET, TARGET, NEGATIVE)

print(f"PER-CLASS METRICS -- binary {TARGET} vs {NEGATIVE},")
print(f"at the {db.REVIEW_CONFIDENCE} review threshold, on the held-out sample.")
print(f"{NEGATIVE} is the pooled remainder: "
      + ", ".join(
          f"{c} {int((_test_labels == c).sum()):,}"
          for c in weak.WEAK_CLASSES if c != TARGET and (_test_labels == c).sum()
      ))
print("This model does NOT separate the other classes from each other; the")
print("full 6-class problem needs a multilabel head or six models, later.")
print("=" * 74)
for key in ("B", "A"):
    confident = prob_b if key == "B" else prob_a
    pred_cls = np.where(
        confident >= db.REVIEW_CONFIDENCE, TARGET, NEGATIVE
    )
    report = per_class_report(
        _true_binary, pred_cls, [TARGET, NEGATIVE],
    )
    print()
    print(f"model {key}")
    print(report.to_string(index=False))
    # Confusion, so the table can be read without arithmetic.
    tp = int(((pred_cls == TARGET) & (_true_binary == TARGET)).sum())
    fp = int(((pred_cls == TARGET) & (_true_binary == NEGATIVE)).sum())
    fn = int(((pred_cls == NEGATIVE) & (_true_binary == TARGET)).sum())
    tn = int(((pred_cls == NEGATIVE) & (_true_binary == NEGATIVE)).sum())
    print(f"  confusion: tp={tp:,} fp={fp:,} fn={fn:,} tn={tn:,}")

# ---------------------------------------------------------------------------
# 8. REVIEW QUEUE -- the brief's 60% escalation
# ---------------------------------------------------------------------------

for key, prob in (("B", prob_b), ("A", prob_a)):
    queue = db.to_review_queue(prob, db.REVIEW_CONFIDENCE)
    n_review = int(queue["review"].sum())
    print()
    print(f"model {key}: {int(queue['accept'].sum()):,} auto-accept, "
          f"{n_review:,} to human review "
          f"({100.0 * n_review / len(prob):.1f}%)")

# ---------------------------------------------------------------------------
# 9. FEATURE IMPORTANCE -- does it look like behaviour or geography?
# ---------------------------------------------------------------------------

print()
print("=" * 74)
print("TOP FEATURES -- model B")
print("=" * 74)
importance = pd.DataFrame(
    {
        "feature": non_leak,
        "gain": models["B"].feature_importance("gain"),
        "splits": models["B"].feature_importance("split"),
    }
).sort_values("gain", ascending=False)
print(importance.head(20).to_string(index=False))

print()
print("=" * 74)
print("CHECK -- is the model leaning on location after all?")
print("=" * 74)
importance.to_csv(QC_DIR / "p7_feature_importance_B.csv", index=False)
leaked = [c for c in non_leak if db.is_location_column(c)]
print(f"  location columns in model B: {leaked or 'none'}")
top = importance.head(10)["feature"].tolist()
print(f"  top-10 features: {top}")
site_like = [c for c in top if c.startswith(("SITE_", "FRP_ANOMALY"))]
if site_like:
    print("  note: site-baseline features rank highly. That is permitted by the")
    print("  brief -- location enters through a site's history, not as a raw")
    print("  coordinate -- but report it rather than let it pass unremarked.")

# ---------------------------------------------------------------------------
# 10. PERSIST
# ---------------------------------------------------------------------------

QC_DIR.mkdir(parents=True, exist_ok=True)
payload = {
    "target": TARGET,
    "label_column": LABEL_COLUMN,
    "label_caveat": (
        "labels are weak: manufactured from features the model sees. "
        "PR-AUC here is an UPPER BOUND, not a generalisation estimate. "
        "An independent hand-labelled sample is required to evaluate."
    ) if LABEL_COLUMN == "weak" else "strict evidence-gated labels",
    "split": td.SPLIT_COMBINED,
    "n_train": int(is_train.sum()),
    "n_test": int(is_test.sum()),
    "model_b": {"features": non_leak, **results["B"]},
    "model_a": {"features": list(_model_a_cols), **results["A"]},
    "baseline": baseline,
    "review_confidence": db.REVIEW_CONFIDENCE,
}
(QC_DIR / "p7_results.json").write_text(json.dumps(payload, indent=2, default=str))
print()
print(f"Wrote {QC_DIR}/p7_results.json")
print(f"Wrote {QC_DIR}/p7_feature_importance_B.csv")
print()
# ---------------------------------------------------------------------------
# Persist the models themselves
#
# Without this the fitted boosters die with the runtime and the run has to be
# repeated to reproduce them. Written with the feature list and the exact
# target, because a booster is meaningless without both: it maps column
# POSITIONS, so reloading it against a reordered feature frame would silently
# score garbage.
# ---------------------------------------------------------------------------
import pickle

MODEL_DIR = QC_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

for key, booster, cols in (
    ("A", models["A"], list(_model_a_cols)),
    ("B", models["B"], list(_model_b_cols)),
):
    path = MODEL_DIR / f"model_{key}.pkl"
    with path.open("wb") as fh:
        pickle.dump(
            {
                "booster": booster,
                "features": cols,
                "target": TARGET,
                "label_column": LABEL_COLUMN,
                "best_iteration": booster.best_iteration,
                "split": split_kind,
                "review_confidence": db.REVIEW_CONFIDENCE,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            },
            fh,
        )
    print(f"Wrote {path}  ({path.stat().st_size / 1024:.0f} KB, "
          f"{booster.best_iteration} trees)")

print()
print("Reloading requires the stored feature list, not the current X:")
print("  payload = pickle.load(open(...));")
print("  booster = payload['booster']; cols = payload['features']")
print("  prob = booster.predict(X[cols])")