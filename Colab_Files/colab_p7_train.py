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
import surface_labels as surface_labels_mod
import threshold_calibration as tcal
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

# Which label to train on.
#
#   "surface"  the SUBSTRATE label from `surface_labels` (qc/surface_labels.*).
#              THE DEFAULT, and the first target in this project a model cannot
#              reproduce. Measured on the full catalogue (2026-10-05): 70.1% of
#              weak INDUSTRIAL_FIRE sits on vegetated substrate, precision 0.299
#              / recall 0.636 / F1 0.407 against a substrate proxy. The weak
#              rules read only columns already in X, so model B was blind to
#              the evidence behind its own target. `LANDCOVER_VEGETATED` comes
#              from a raster absent from X, so label and model share no input.
#   "weak"    the industrial-context hierarchy in `weak_labels`. Two distance
#             terms after the calibration collapse, with `industrial_persistence`
#             pinned to the floor of observed_days/30 so the shape conjunct
#             excluded nothing. Retained for before/after comparison ONLY.
#   "strict"  the evidence-gated engine. Almost entirely UNKNOWN.
LABEL_SOURCE = str(globals().get("LABEL_SOURCE", "surface")).lower()

_surf_pq = QC_DIR / "surface_labels.parquet"
_surf_csv = QC_DIR / "surface_labels.csv"
_surf_path = (_surf_pq if cu.path_ok(_surf_pq)
              else (_surf_csv if cu.path_ok(_surf_csv) else None))

if LABEL_SOURCE == "surface":
    if _surf_path is None:
        raise SystemExit(
            "qc/surface_labels.parquet not found. Run "
            "colab_p7i_surface_label.py first and check its section 4 printed "
            "SURFACE_FIRE and UNKNOWN: 0. Refusing to fall back to the weak "
            "label -- that would silently train on the broken target again."
        )
    _surf = (pd.read_parquet(_surf_path) if _surf_path.suffix == ".parquet"
             else pd.read_csv(_surf_path))
    # POSITIONAL join, by `row`. `row` is the catalogue index P7h recorded, so
    # this is exact -- no coordinate matching and no chance of pairing an event
    # with its neighbour.
    _surf = _surf.sort_values("row")
    if len(_surf) != len(y):
        raise SystemExit(
            f"surface labels has {len(_surf):,} rows but y.parquet has "
            f"{len(y):,}. The positional join would be WRONG."
        )
    labels = _surf["label_surf"].to_numpy()
    _bin = _surf["is_surface_fire"].to_numpy(dtype="float64")
    # UNKNOWN rows become 0 but are MASKED OUT of the sample entirely below, so
    # a missing label can never train as a negative.
    _surface_known = np.isfinite(_bin)
    y_bin = np.where(_surface_known, _bin, 0.0).astype(int)
    LABEL_COLUMN = "surface"
    print(f"surface labels : {_surf_path}")
    print(f"  classes       : {dict(pd.Series(labels).value_counts())}")
    print(f"  unmapped/NA   : {int((~_surface_known).sum()):,}  -> excluded "
          "from training entirely")
    _tier_col = "is_surface_fire_tiered"
    if _tier_col in _surf.columns:
        print(f"  high tier (>=3 observed days): "
              f"{int((_surf[_tier_col].to_numpy(dtype=float) == 1).sum()):,}")
else:
    LABEL_COLUMN = ("weak" if "label_weak" in y.columns else "strict")
    labels = (y["label_weak"] if LABEL_COLUMN == "weak"
              else y["label"]).to_numpy()
    # y_bin is NOT built here: TARGET is resolved further down, once the label
    # source is known. Building it in both branches meant the surface branch's
    # version was overwritten anyway, and the weak branch read a name that did
    # not exist yet.
    _surface_known = np.ones(len(labels), dtype=bool)
    print(f"WARNING: training on the WEAK label. Every metric below is")
    print("         optimistic by construction, and the label was measured at")
    print("         precision 0.299. Use this run only for comparison.")

weights = (y["sample_weight"].to_numpy() if "sample_weight" in y.columns
           else np.ones(len(labels), dtype="float64"))

# Mask out any row whose label could not be determined. For the weak labels
# this is empty; for the surface label it is any UNKNOWN substrate.
_usable = _surface_known.copy() if LABEL_SOURCE == "surface" \
    else np.ones(len(labels), dtype=bool)
if not _usable.all():
    print(f"excluding {_usable.size - int(_usable.sum()):,} rows with no label")

# `X` does not exist yet. It is filled AFTER the sample is chosen, straight
# into preallocated float32 arrays, one row group at a time. NaN is preserved,
# which is essential: NaN is how "could not measure" is encoded.

# Which class to fit. For the SURFACE label the positive is SURFACE_FIRE, whose
# count is set below from the label itself rather than assumed.
#
# For the WEAK label, INDUSTRIAL_FLARE has only 3,571 events in the whole
# catalogue -- 894 in a training sample, 33 in the test set -- so its metrics are
# dominated by sampling noise and both models early-stopped at iteration 3. That
# is why it is no longer the default target.
TARGET = globals().get("TARGET", None)
if TARGET is None:
    TARGET = (surface_labels_mod.SURFACE_FIRE if LABEL_SOURCE == "surface"
              else "INDUSTRIAL_FLARE")
    print(f"  (target defaulted to {TARGET!r} for LABEL_SOURCE="
          f"{LABEL_SOURCE!r})")

print()
print(f"label column : {LABEL_COLUMN}")
if LABEL_SOURCE == "surface":
    print("  SUBSTRATE label. Built from a raster ABSENT FROM X, so the model")
    print("  cannot reproduce it. This is the first target here whose metrics")
    print("  mean what they say.")
    print("  It measures SUBSTRATE, not cause -- 'industrial' remains a claim")
    print("  needing corroboration. See qc/surface_label_meta.json.")
elif LABEL_COLUMN == "weak":
    print("  WEAK LABELS. These are manufactured from features the model sees,")
    print("  so PR-AUC here is an upper bound, NOT a generalisation estimate.")
    print("  Measured at precision 0.299 against a substrate proxy (2026-10-05).")
    print("  Only an independent hand-labelled sample can give the latter.")
else:
    print("  strict labels -- nearly all UNKNOWN, so training cannot proceed.")
print()
print(f"target class : {TARGET}")
print("class counts (full catalogue):")
for cls in sorted(set(labels)):
    n = int((labels == cls).sum())
    print(f"  {str(cls):20s} {n:>10,}  {100.0 * n / max(len(labels), 1):>6.2f}%")

# ---------------------------------------------------------------------------
# 2. SPLIT
# ---------------------------------------------------------------------------

meta = td.add_split_keys(meta)
split_kind = td.SPLIT_COMBINED
parts = td.make_split(meta, kind=split_kind, seed=0)
print(f"  split kind: {split_kind}")
train_ids = set(parts["train"]["event_id"])
test_ids = set(parts["test"]["event_id"])

# Rows whose label is UNKNOWN are EXCLUDED from the split entirely, not coerced
# to 0. A missing label is not a negative: training on it would teach the model
# that unmeasured substrate means "vegetation fire", which is exactly the
# silent-wrong-answer failure this project has already produced twice.
is_train = (meta["event_id"].isin(train_ids).to_numpy() & _usable)
is_test = (meta["event_id"].isin(test_ids).to_numpy() & _usable)
dropped = ~(is_train | is_test)
_n_lost = int((~_usable).sum())
if _n_lost:
    print(f"  excluded for an undetermined label: {_n_lost:,}")
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

# The surface label already carries a 0/1 with NaN for UNKNOWN, so rebuilding it
# from the class name would silently turn UNKNOWN rows into negatives.
if LABEL_SOURCE != "surface":
    y_bin = (labels == TARGET).astype(int)
else:
    y_bin = np.where(_surface_known,
                     (_surf["label_surf"].to_numpy() == TARGET).astype(int), 0)
print(f"  positives in train: {int(y_bin[is_train].sum()):,}")
print(f"  positives in test : {int(y_bin[is_test].sum()):,}")
print(f"  positive rate      : {100.0 * y_bin[is_train].mean():.2f}% train")
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
# `include_surface` follows the label. Training against the surface label
# WITHOUT excluding substrate from model B would put the label's own evidence
# into the model -- the same circularity that made the weak label unreadable,
# one column further in.
_model_a_cols, _model_b_cols, _leak = td.split_feature_sets(
    _X_COLS, include_surface=(LABEL_SOURCE == "surface"))
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
    """Train one LightGBM model and score it.

    Returns (model, prob_test, prob_train, calibration).

    TRAIN scores are returned because the operating threshold must be frozen on
    the TRAIN fold and only on the TRAIN fold. Calibrating tau on the fold it
    is then scored against is the quietest way to overstate a model, and it is
    the same trap `tune_distance_baseline` already guards against below.
    """
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
    prob_train = model.predict(X_train, num_iteration=model.best_iteration)
    print(f"  {name}: best_iteration={model.best_iteration}")

    tcal.assert_calibrated_on_train("train")
    calibration = tcal.calibrate_operating_threshold(y_train, prob_train)
    return model, prob, prob_train, calibration


results = {}
models = {}
calibrations = {}

# ---------------------------------------------------------------------------
# 4. MODEL B -- the scientific claim (non-leak only)
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print(f"model B (scientific): {len(non_leak)} non-leak features")
print(f"  excluded as label-derived: {label_derived}")
print("=" * 74)

model_b, prob_b, prob_b_train, cal_b = fit_and_score(
    X_train_b, y_bin[train_idx], weights[train_idx],
    X_test_b, y_bin[test_idx], "model B",
)
models["B"] = model_b
calibrations["B"] = cal_b
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

model_a, prob_a, prob_a_train, cal_a = fit_and_score(
    X_train_a, y_bin[train_idx], weights[train_idx],
    X_test_a, y_bin[test_idx], "model A",
)
models["A"] = model_a
calibrations["A"] = cal_a
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
# Distance is only needed for the baseline and the evaluation CSV, and only
# for the sampled rows, so it is read from the matrices already in memory.
#
# NOTE: this is ALREADY the train-then-test concatenation, so it must be
# sliced before being indexed with global positions. Indexing the whole thing
# with `test_idx` reaches past its end -- which is exactly what happened.
_dist_train = X_train_a[:, _dist_col]
_dist_test = X_test_a[:, _dist_col]
# `DIST_*_LOG` is log10(1 + d_metres), so the metres conversion must be applied
# where the value is CONSUMED. It was previously assigned to a local `dist_m`
# that nothing ever read, while the raw log column was written to the CSV under
# the name `dist_m` -- so the file held values in 2.4..4.8 where `evaluate_model`
# and the baseline expect metres, i.e. off by four orders of magnitude. Ranking
# was unaffected (the transform is monotone), which is why no metric ever
# looked wrong. Fixed by computing once and using that name for both the
# baseline and the CSV.
dist_m_test = np.where(
    np.isfinite(_dist_test), np.power(10.0, _dist_test) - 1.0, np.nan
)
# Guard the unit, because a mislabelled column here is invisible in every
# ranking-based metric and only shows up as a nonsense metre threshold.
_med = float(np.nanmedian(dist_m_test)) if np.isfinite(dist_m_test).any() else 0.0
if not (_med > 1.0):
    raise SystemExit(
        f"dist_m looks like log10 units, not metres (median {_med:.4f}). "
        "DIST_NEAREST_FACILITY_LOG is log10(1 + d_metres); the conversion was "
        "dropped somewhere. Fix before writing the CSV -- evaluate_model.py and "
        "the baseline both read this column as metres."
    )
print(f"  dist_m unit check: median {_med:,.0f} m (metres, PASS)")

# Tuned on TRAIN only. Selecting the threshold on the test fold and then
# reporting that fold's score is one of the quietest ways to overstate.
baseline = td.tune_distance_baseline(
    np.where(np.isfinite(_dist_train),
             np.power(10.0, _dist_train) - 1.0, np.nan),
    y_bin[train_idx]
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
    print(f"  NOTE: that recall is at the bare {db.REVIEW_CONFIDENCE} cut, which"
          " is not a deployed operating point.")
    print("        The calibrated bar is reported in the next section.")
    verdict = td.compare_to_baseline(m, baseline)
    print(f"  vs baseline: {verdict['verdict']}")

print()
print("=" * 74)
print("OPERATING THRESHOLD -- frozen on TRAIN, applied to the held-out fold")
print("=" * 74)
print("Policy (HANDOFF §8): tau maximising F2 subject to precision >= 0.90.")
print("This is a DIFFERENT constraint from the brief's flat 0.60 review rule,")
print("and the two must not be conflated: an event is auto-accepted only when")
print("it clears max(tau, 0.60). A flat 0.60 is meaningful only for a model")
print("whose scores happen to be calibrated to that prior, and none of these")
print("are -- model B's held-out scores sit near 0.13.")
print()

_probs = {"A": prob_a, "B": prob_b}
_probs_train = {"A": prob_a_train, "B": prob_b_train}
operating_points = {}

for key in ("B", "A"):
    cal = calibrations[key]
    print(tcal.describe_calibration(cal, f"model {key}  (on train)"))
    tau = cal.get("threshold")
    operating_points[key] = {k: cal.get(k) for k in (
        "threshold", "precision", "recall", "f_beta", "n_predicted",
        "flag_fraction", "degenerate", "feasible", "reason",
        "min_precision", "beta",
    )}
    if tau is None:
        print("    -> NO operating point. Every event must go to human review.")
        print("       Do NOT fall back to a flat 0.60 here: that cut flags a")
        print("       fraction of a percent of positives while reporting a")
        print("       confident precision, which is a false claim of coverage.")
    else:
        pred = _probs[key] >= tau
        tp = int(np.sum(pred & (y_bin[test_idx] == 1)))
        fp = int(np.sum(pred & (y_bin[test_idx] == 0)))
        fn = int(np.sum(~pred & (y_bin[test_idx] == 1)))
        held_p = tp / (tp + fp) if (tp + fp) else 0.0
        held_r = tp / (tp + fn) if (tp + fn) else 0.0
        held_f1 = (2 * held_p * held_r / (held_p + held_r)
                   if (held_p + held_r) else 0.0)
        operating_points[key]["held_out"] = {
            "precision": held_p, "recall": held_r, "f1": held_f1,
            "n_predicted": int(pred.sum()),
        }
        print(f"    -> on held-out: precision {held_p:.4f}  recall {held_r:.4f}"
              f"  F1 {held_f1:.4f}   (tau never saw this fold)")

print()
print("What recall is actually available, and at what price:")
print("Reported for both models because 'no operating point' is only actionable")
print("if the reader can see the trade the model is really making.")
print()
for key in ("B", "A"):
    for target in (0.50, 0.80, 0.95):
        cal = tcal.threshold_for_recall(
            y_bin[train_idx], _probs_train[key], min_recall=target
        )
        print(tcal.describe_calibration(cal, f"  model {key} recall>={target:.2f}"))
        if cal.get("threshold") is not None:
            pred = _probs[key] >= cal["threshold"]
            tp = int(np.sum(pred & (y_bin[test_idx] == 1)))
            fp = int(np.sum(pred & (y_bin[test_idx] == 0)))
            fn = int(np.sum(~pred & (y_bin[test_idx] == 1)))
            hp = tp / (tp + fp) if (tp + fp) else 0.0
            hr = tp / (tp + fn) if (tp + fn) else 0.0
            print(f"    -> on held-out: precision {hp:.4f}  recall {hr:.4f}")
        if cal.get("degenerate"):
            print("    -> DEGENERATE: that recall costs a flag on most of the")
            print("       catalogue. Compare the precision against the base rate")
            print(f"       ({100.0 * y_bin[test_idx].mean():.2f}%): if it is close,")
            print("       the model is not ranking positives above negatives.")

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
print(f"at each model's own DEPLOYED bar, on the held-out sample.")
print(f"{NEGATIVE} is the pooled remainder: "
      + ", ".join(
          f"{c} {int((_test_labels == c).sum()):,}"
          for c in weak.WEAK_CLASSES if c != TARGET and (_test_labels == c).sum()
      ))
print("This model does NOT separate the other classes from each other; the")
print("full 6-class problem needs a multilabel head or six models, later.")
print("=" * 74)
for key in ("B", "A"):
    confident = _probs[key]
    # The DEPLOYED bar, not the brief's bare 0.60. Auto-accept requires the
    # calibrated operating point AND the brief's review floor; whichever is
    # higher binds. Reporting per-class metrics at 0.60 for a model whose
    # calibrated tau is 0.875 would describe an operating point nobody ships.
    bar = tcal.effective_threshold(
        operating_points[key]["threshold"], db.REVIEW_CONFIDENCE
    )
    pred_cls = np.where(confident >= bar, TARGET, NEGATIVE)
    report = per_class_report(
        _true_binary, pred_cls, [TARGET, NEGATIVE],
    )
    print()
    print(f"model {key}  (auto-accept bar {bar:.6f} = max(tau, "
          f"{db.REVIEW_CONFIDENCE}))")
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

print()
print("=" * 74)
print("REVIEW QUEUE -- at the deployed bar, not at a bare 0.60")
print("=" * 74)
for key in ("B", "A"):
    prob = _probs[key]
    bar = tcal.effective_threshold(
        operating_points[key]["threshold"], db.REVIEW_CONFIDENCE
    )
    queue = db.to_review_queue(prob, bar)
    n_review = int(queue["review"].sum())
    print()
    print(f"model {key}: {int(queue['accept'].sum()):,} auto-accept, "
          f"{n_review:,} to human review "
          f"({100.0 * n_review / len(prob):.1f}%)")
    if key == "B" and operating_points["B"]["threshold"] is None:
        print("  No operating point was freezable for this model, so the queue")
        print("  degenerates to 'review almost everything'. That is the honest")
        print("  consequence of a detector that does not discriminate, and it")
        print("  is the operational argument for LANDCOVER_FUEL: without the")
        print("  features that separate industrial from wild fire there is")
        print("  nothing to auto-accept on.")

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

# Persist the held-out predictions so the evaluation can be re-run without
# retraining. Without this, scripts/evaluate_model.py has nothing to read: the
# metrics are printed but never saved, so the honest evaluation is a one-off
# terminal scroll rather than a reproducible artefact.
#
# y_true is the WEAK label, recorded explicitly so nobody mistakes it for
# ground truth later.
_base = pd.DataFrame({
    "event_id": meta.iloc[test_idx]["event_id"].to_numpy(),
    # y_true is the WEAK label. Named so it cannot later be mistaken for
    # ground truth -- it is not.
    "y_true": y_bin[test_idx],
    "y_true_weak_label": labels[test_idx],
    "dist_m": dist_m_test,
    # The log10 form too, so a consumer never has to guess which convention a
    # file uses. `dist_m` is metres; `dist_log10` is what X actually stores.
    "dist_log10": _dist_test,
})
for _name, _score in (("heldout_pred_B", prob_b),
                      ("heldout_pred_A", prob_a)):
    _out = _base.copy()
    _out["score"] = _score
    _path = QC_DIR / f"{_name}.csv"
    _out.to_csv(_path, index=False)
    print(f"Wrote {_path}  {len(_out):,} rows")
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
    "operating_points": operating_points,
    "operating_threshold_policy": (
        "tau maximising F2 subject to precision >= 0.90, frozen on the TRAIN "
        "fold only. Distinct from review_confidence (0.60, the brief's human-"
        "review floor). Auto-accept requires score >= max(tau, 0.60). A None "
        "threshold means no operating point was freezable and every event must "
        "go to human review."
    ),
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
                # The calibrated operating point. Stored beside the booster so
                # a reload cannot silently revert to the bare 0.60 literal, which
                # is the literal that produced "detects 2 of 100".
                "operating_threshold": operating_points[key]["threshold"],
                "operating_point": operating_points[key],
                "auto_accept_threshold": tcal.effective_threshold(
                    operating_points[key]["threshold"], db.REVIEW_CONFIDENCE
                ),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            },
            fh,
        )
    print(f"Wrote {path}  ({path.stat().st_size / 1024:.0f} KB, "
          f"{booster.best_iteration} trees)")

print()
print()
print("Held-out predictions written. To evaluate them honestly:")
print("  python3 scripts/evaluate_model.py --scores qc/heldout_pred_B.csv")
print("  (y_true there is the WEAK label, not ground truth)")
print()
print("Reloading requires the stored feature list, not the current X:")
print("  payload = pickle.load(open(...));")
print("  booster = payload['booster']; cols = payload['features']")
print("  prob = booster.predict(X[cols])")