"""
P7 -- TRAINING DESIGN.

Pure logic: splits, baselines, metrics, leakage enforcement. No LightGBM import
here so this module is testable without the dependency, and so the evaluation
design can be verified before anyone fits a model.

The four rules this encodes, all of which are ways a fire-classification paper
ends up reporting a number that does not survive review:

1. **Two models, always.** Model A uses every available feature including the
   18 `LABEL_DERIVED` industrial ones, which encode facility proximity -- the
   very thing being predicted. It is the operational model and must be labelled
   as such. Model B uses only the non-leak partition and is the scientific
   claim. Reporting A alone is leakage; reporting B alone hides that the
   operational system uses facility priors.

2. **No random splits.** Random splits leak both ways: neighbouring events in
   space land on both sides, and adjacent days in time do too. Both inflate
   scores. Three splits are provided, and the combined `year x block` split is
   the one that gets reported as the headline.

3. **Baselines must be beaten or the ML is pointless.** A single threshold on
   distance-to-nearest-facility is the thing any reviewer will ask about. If
   the model cannot beat it, say so plainly.

4. **`UNKNOWN` is kept, not dropped.** It carries a low sample weight so the
   class prior stays honest. Dropping it silently inflates every metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import feature_ledger as fl

# ---------------------------------------------------------------------------
# LEAKAGE PARTITION
# ---------------------------------------------------------------------------

#: The 18 industrial features, whose inputs build the label. Model A only.
LABEL_DERIVED: "list[str]" = list(fl.LABEL_DERIVED_FEATURES)

#: The scientific partition as the ledger defines it (123 features).
NON_LEAK: "list[str]" = list(fl.NON_LEAK_FEATURES)


def label_input_features(include_surface: bool = True):
    """Features that manufacture y and must never enter the scientific model.

    Three sources:
      * the 18 `LABEL_DERIVED` industrial features
      * every input of the weak rule hierarchy in `weak_labels`
      * every input of the SURFACE label in `surface_labels`

    The second set is not optional. The weak rules reduce largely to "was this
    event near a facility", so `DIST_*` decides y; leaving `DIST_*` in X would
    let the model learn the labelling rule instead of the class.

    THE THIRD SET IS WHAT MAKES MODEL B WORTH MEASURING. Measured on the full
    catalogue (2026-10-05): 70.1% of weak INDUSTRIAL_FIRE sits on vegetated
    substrate, precision 0.299 against a substrate proxy. More importantly the
    weak rules read only columns that are already IN X, so excluding them
    leaves model B blind to the evidence behind its own target -- it cannot win
    because the target carried no information it was allowed to use.

    `surface_labels` reads `LANDCOVER_VEGETATED`, which comes from a raster
    absent from X. With that in the label, and excluded here, the two no longer
    share an input and model B's score means something.

    `include_surface=False` reproduces the OLD partition exactly, which is what
    a before/after comparison needs.
    """
    import weak_labels

    forbidden = set(fl.LABEL_DERIVED_FEATURES) | set(
        weak_labels.LABEL_INPUT_FEATURES)
    if include_surface:
        import surface_labels
        forbidden |= set(surface_labels.surface_label_inputs())
    return forbidden


def split_feature_sets(available, include_surface: bool = True):
    """Partition the *available* features into A (all) and B (non-leak).

    Parameters
    ----------
    available : list of feature names actually computed. Passing only what
        exists matters: the partition is computed over reality, so a missing
        feature cannot be silently counted as available.

    Returns
    -------
    (model_a, model_b) as lists, plus a report dict naming what was dropped.
    """
    import weak_labels

    have = set(available)
    forbidden = label_input_features(include_surface)
    model_a = [f for f in fl.FEATURES_141 if f in have]
    model_b = [f for f in fl.FEATURES_141
               if f in have and f not in forbidden]
    report = {
        "n_available": len(model_a),
        "n_label_derived_in_a": len([f for f in model_a
                                      if f in fl.LABEL_DERIVED_FEATURES]),
        "n_weak_rule_inputs_in_a": len([f for f in model_a
                                        if f in weak_labels.LABEL_INPUT_FEATURES]),
        "n_model_b": len(model_b),
        "missing_from_141": sorted(have - set(fl.FEATURES_141)),
        "excluded_from_b": sorted(f for f in model_a
                                  if f not in model_b),
    }
    leaked = sorted(set(model_b) & forbidden)
    if leaked:
        raise ValueError(
            f"target leakage: these features manufacture y but are in model B: "
            f"{leaked}"
        )
    return model_a, model_b, report


# ---------------------------------------------------------------------------
# SPLITS -- deliberately non-random
# ---------------------------------------------------------------------------

#: Held-out period for the out-of-time split.
TEST_START_YEAR = 2020

#: Grid resolution for the out-of-space split, in degrees.
BLOCK_DEG = 0.1

SPLIT_OUT_OF_TIME = "out_of_time"
SPLIT_OUT_OF_SPACE = "out_of_space"
SPLIT_COMBINED = "year_x_block"


def add_split_keys(frame, *, year_col="start_year", lon_col="event_centroid_lon",
                   lat_col="event_centroid_lat"):
    """Attach the year and spatial-block keys the splits need.

    Adds `start_year` and `spatial_block` (a `lon_lat` string at BLOCK_DEG
    resolution). Both are cheap, deterministic, and derived only from
    coordinates and time -- never from the label.
    """
    df = frame
    if year_col in df.columns:
        year = pd.to_numeric(df[year_col], errors="coerce")
    elif "start_time" in df.columns:
        year = pd.to_datetime(df["start_time"], errors="coerce").dt.year
    else:
        raise KeyError(f"need {year_col} or start_time to derive the year")

    lon = pd.to_numeric(df[lon_col], errors="coerce")
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    block_x = np.floor(lon / BLOCK_DEG)
    block_y = np.floor(lat / BLOCK_DEG)

    df = df.copy()
    df["start_year"] = year.astype("int64")
    df["spatial_block"] = (
        block_x.fillna(-999).astype("int64").astype(str)
        + "_"
        + block_y.fillna(-999).astype("int64").astype(str)
    )
    df.loc[lon.isna() | lat.isna(), "spatial_block"] = "unknown"
    return df


def split_out_of_time(frame, test_start_year=TEST_START_YEAR):
    """Train <= 2019, test >= 2020. Temporal generalisation only."""
    year = pd.to_numeric(frame["start_year"], errors="coerce")
    train = frame[year < test_start_year]
    test = frame[year >= test_start_year]
    return {"train": train, "test": test}


def split_out_of_space(frame, seed=0):
    """Hold out whole 0.1 degree blocks. Spatial generalisation only.

    Block-level, not row-level: two events 200 m apart in the same block must
    never straddle the split, which is exactly what a random split does.
    """
    blocks = frame["spatial_block"].unique()
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(blocks))
    n_test = max(1, int(round(0.2 * len(shuffled))))
    test_blocks = set(shuffled[:n_test])
    is_test = frame["spatial_block"].isin(test_blocks)
    return {"train": frame[~is_test], "test": frame[is_test],
            "test_blocks": test_blocks}


def split_combined(frame, seed=0, test_start_year=TEST_START_YEAR):
    """Out-of-time AND out-of-space simultaneously.

        test  = year >= test_start_year  AND  block held out
        train = every event whose block is NOT held out
        drop  = early events that happen to sit in a held-out block

    The dropped group is what makes this a real spatial holdout. A naive
    "test = late AND held-out, train = everything else" leaks, because the
    held-out block's *early* events land in training and the test block is then
    not unseen at all. Dropping those events costs some training data and buys
    the guarantee; the alternative quietly inflates the headline number.

    Training does include late events from non-held-out blocks. That is
    deliberate: the temporal claim rests on the test events being late and
    unseen, not on training being frozen in the past.
    """
    space = split_out_of_space(frame, seed=seed)
    held = frame["spatial_block"].isin(space["test_blocks"]).to_numpy()
    year = pd.to_numeric(frame["start_year"], errors="coerce").to_numpy()
    late = np.nan_to_num(year, nan=0) >= test_start_year

    is_test = held & late
    is_train = ~held
    return {
        "train": frame[is_train],
        "test": frame[is_test],
        "dropped_early_in_test_blocks": frame[held & ~late],
        "test_blocks": space["test_blocks"],
    }


def make_split(frame, kind=SPLIT_COMBINED, seed=0):
    if kind == SPLIT_OUT_OF_TIME:
        return split_out_of_time(frame)
    if kind == SPLIT_OUT_OF_SPACE:
        return split_out_of_space(frame, seed=seed)
    if kind == SPLIT_COMBINED:
        return split_combined(frame, seed=seed)
    raise ValueError(f"unknown split {kind!r}")


def assert_no_spatial_leakage(train, test, block_col="spatial_block"):
    """Fail if any spatial block appears on both sides.

    A split that shares a block is not an out-of-space split, whatever it was
    named. Cheap to check and it catches the most common evaluation error.
    """
    shared = set(train[block_col].unique()) & set(test[block_col].unique())
    if shared:
        raise AssertionError(
            f"{len(shared)} spatial block(s) appear in both train and test, "
            f"e.g. {sorted(shared)[:5]}"
        )


def assert_no_time_leakage(train, test, year_col="start_year",
                           test_start_year=TEST_START_YEAR):
    """Fail if any test-year event is in training.

    ONLY valid for the out-of-time split. The combined split deliberately
    trains on late events from NON-held-out blocks, so its temporal guarantee is
    different: the test events are late AND in blocks never seen in training.
    Applying this assertion there rejects a correct split -- which it did, with
    "training set contains 3,417,935 event(s) from 2020 or later" on a run whose
    spatial guarantee had already passed.
    """
    years = pd.to_numeric(train[year_col], errors="coerce")
    if (years >= test_start_year).any():
        raise AssertionError(
            f"out-of-time split violated: training set contains "
            f"{int((years >= test_start_year).sum())} event(s) from "
            f"{test_start_year} or later. If you are using SPLIT_COMBINED this "
            "assertion does not apply -- use assert_no_spatial_leakage only."
        )


# ---------------------------------------------------------------------------
# BASELINES -- the thing the model must beat
# ---------------------------------------------------------------------------

#: Distance thresholds (metres) for the single-feature baseline.
BASELINE_THRESHOLDS_M = (250.0, 500.0, 1000.0, 2000.0, 5000.0)


def distance_baseline(distances_m, threshold_m):
    """Predict positive where the nearest facility is within `threshold_m`.

    NaN distance -> negative, because "we could not measure it" must not be
    read as "close". Returns a boolean array.
    """
    d = np.asarray(distances_m, dtype="float64")
    return np.isfinite(d) & (d <= float(threshold_m))


def tune_distance_baseline(distances_m, y, thresholds=BASELINE_THRESHOLDS_M,
                           min_precision=0.90):
    """Pick the best threshold on precision, then report its full curve.

    Tuned on the TRAINING fold only. Selecting a threshold on the test fold and
    then reporting that fold's score is one of the quietest ways to overstate a
    result.
    """
    d = np.asarray(distances_m, dtype="float64")
    y = np.asarray(y)
    best = None
    for t in thresholds:
        pred = distance_baseline(d, t)
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        row = {"threshold_m": float(t), "precision": precision,
               "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        if best is None or f1 > best["f1"]:
            best = row
    best["meets_precision_target"] = bool(best["precision"] >= min_precision)
    return best


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

#: Operating-point target from the project statement.
TARGET_PRECISION = 0.90


def _tie_aware_curve(y, score):
    """Precision/recall at each DISTINCT score, ties collapsed.

    Returns (scores_desc, precision, recall, predicted) where precision is
    measured at the END of each run of equal scores.

    Why this matters: a model that outputs many ties -- and LightGBM always
    does -- would otherwise have its tied positives sorted ahead of tied
    negatives by an arbitrary stable sort. That inflates both PR-AUC and the
    operating-point recall without any real discrimination. Collapsing ties is
    the standard correction and the reason a constant predictor scores exactly
    the class prevalence instead of a flattering 1.0.
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype="float64")
    if score.size == 0:
        z = np.empty(0)
        return z, z, z, z

    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    s_sorted = score[order]

    tp = np.cumsum(y_sorted)
    predicted = np.arange(1, len(y_sorted) + 1)

    # Last index of each run of equal scores.
    change = np.empty(len(s_sorted), dtype=bool)
    change[-1] = True
    change[:-1] = s_sorted[1:] != s_sorted[:-1]
    ends = np.where(change)[0]

    precision = tp[ends] / predicted[ends]
    recall = tp[ends] / max(int(y.sum()), 1)
    return s_sorted[ends], precision, recall, predicted[ends]


def precision_recall_at_precision(y_true, y_pred, min_precision=TARGET_PRECISION):
    """Recall at the highest-recall threshold holding precision >= target.

    Sweeps from the most conservative threshold to the most permissive, which
    is what "operating point" means operationally: deploy the model only where
    it stays trustworthy, then measure how much fire you actually catch.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred)
    if y.size == 0:
        return {"threshold": None, "precision": 0.0, "recall": 0.0,
                "n_predicted": 0}
    scores, precision, recall, predicted = _tie_aware_curve(y, p)
    if scores.size == 0:
        return {"threshold": None, "precision": 0.0, "recall": 0.0,
                "n_predicted": 0}
    ok = np.where(precision >= min_precision)[0]
    if len(ok) == 0:
        return {"threshold": None, "precision": float(precision[0]),
                "recall": 0.0, "n_predicted": 0}
    i = int(ok[np.argmax(recall[ok])])
    return {"threshold": float(scores[i]), "precision": float(precision[i]),
            "recall": float(recall[i]), "n_predicted": int(predicted[i])}


def binary_metrics(y_true, y_pred):
    """Precision, recall, F1, F2, PR-AUC (average precision) and confusion."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(int)
    tp = int(np.sum((p == 1) & (y == 1)))
    fp = int(np.sum((p == 1) & (y == 0)))
    fn = int(np.sum((p == 0) & (y == 1)))
    tn = int(np.sum((p == 0) & (y == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    f2 = (5 * precision * recall / (4 * precision + recall)
          if (4 * precision + recall) else 0.0)
    return {
        "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "f2": float(f2),
        "pr_auc": float(average_precision(y, p)),
        "recall_at_precision_target": precision_recall_at_precision(y, p),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def average_precision(y_true, y_pred):
    """Area under the precision-recall curve, step interpolation, ties collapsed.

    PR-AUC rather than ROC-AUC: under heavy class imbalance ROC-AUC is
    dominated by true negatives and looks impressive for a useless model.

    With every score tied -- a constant predictor -- this returns the class
    prevalence, which is the honest score for "no discrimination at all".
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred, dtype="float64")
    n_pos = int(y.sum())
    if n_pos == 0 or p.size == 0:
        return 0.0
    _, precision, recall, _ = _tie_aware_curve(y, p)
    ap = 0.0
    prev_r = 0.0
    for r, pr in zip(recall, precision):
        ap += (r - prev_r) * pr
        prev_r = r
    return float(ap)


def compare_to_baseline(model_metrics, baseline_metrics):
    """Did the model beat the distance threshold?

    Returns an explicit verdict. A model that does not beat a one-feature
    threshold has added nothing and should be reported as such rather than
    buried under a PR-AUC figure.
    """
    gain = model_metrics["pr_auc"] - baseline_metrics["f1"]
    return {
        "model_pr_auc": model_metrics["pr_auc"],
        "baseline_f1": baseline_metrics["f1"],
        "absolute_gain": float(gain),
        "beats_baseline": bool(gain > 0.0),
        "verdict": (
            "ML beats the single-threshold baseline."
            if gain > 0.0
            else "ML does NOT beat the single-threshold baseline; report that "
                 "plainly rather than quoting PR-AUC on its own."
        ),
    }


# ---------------------------------------------------------------------------
# SAMPLE WEIGHTS
# ---------------------------------------------------------------------------


def class_weights(labels, base=None):
    """Per-event sample weights from the label engine.

    Delegates to fire_labels so the weighting rule lives in one place. Labels
    are passed through unchanged -- UNKNOWN events keep a low weight rather
    than being removed, so the class prior stays honest.
    """
    import fire_labels as lab

    return lab.label_weights(labels, base=base)


def training_plan(available_features, label_counts):
    """The plan the P7 cell prints before fitting anything."""
    model_a, model_b, report = split_feature_sets(available_features)
    return {
        "model_a": {
            "features": model_a,
            "n": len(model_a),
            "role": "operational; uses facility priors, must be labelled as such",
        },
        "model_b": {
            "features": model_b,
            "n": len(model_b),
            "role": "scientific; this is the number the paper reports",
        },
        "leakage_report": report,
        "label_counts": dict(label_counts),
        "splits": [SPLIT_COMBINED, SPLIT_OUT_OF_TIME, SPLIT_OUT_OF_SPACE],
        "headline_split": SPLIT_COMBINED,
        "baselines": ["distance threshold on DIST_NEAREST_FACILITY_LOG"],
        "metric": "PR-AUC plus recall at precision >= 0.90",
    }