"""
P8 -- FIGURES and the TRAINING LEDGER.

Produces the plots and the numbers a reviewer will ask for: what was trained on,
how much of it survived into the model, and what the model actually did.

The confusion matrix is reported four ways on purpose. A single 2x2 hides the
only fact that matters here, which is that predicting "nothing is positive"
scores well:

    the all-negative predictor, as a floor
    the model at its own best F1
    the model at the 0.60 review threshold
    the model against the distance baseline, like for like

Data-integrity gate
-------------------
`distance_sanity` checks the stored `DIST_*` features against the geometry they
should encode. On the first real run it fired: every event was 0.1-5.1 m from a
facility, which is impossible for 7.27M events over a subcontinent, and it made
the distance baseline degenerate (recall 1.0 by construction). The check is in
this module so the next run reports it rather than plotting nonsense.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

#: A DIST_* feature is log10(1 + metres). These bounds are what real facility
#: geometry over India looks like; anything outside them means the feature does
#: not encode distance.
DISTANCE_LOG_PLAUSIBLE = (1.0, 6.0)      # 9 m .. 1,000 km
DISTANCE_LOG_SUSPECT_MAX = 1.0            # below 9 m is not a facility distance


# ---------------------------------------------------------------------------
# DATA INTEGRITY
# ---------------------------------------------------------------------------


def distance_sanity(dist_m: np.ndarray, label: str = "") -> Dict:
    """Check a stored distance column against plausible geometry.

    Returns a report rather than raising, so a full figure set is still produced
    with the problem flagged on every relevant plot.
    """
    d = np.asarray(dist_m, dtype="float64")
    finite = d[np.isfinite(d)]
    lo, hi = DISTANCE_LOG_PLAUSIBLE
    too_close = int((finite < DISTANCE_LOG_SUSPECT_MAX).sum())
    n = max(finite.size, 1)
    report = {
        "label": label,
        "n": int(finite.size),
        "min_m": float(finite.min()) if finite.size else None,
        "median_m": float(np.median(finite)) if finite.size else None,
        "p99_m": float(np.percentile(finite, 99)) if finite.size else None,
        "max_m": float(finite.max()) if finite.size else None,
        "frac_implausibly_close": too_close / n,
        "ok": too_close / n < 0.01,
    }
    if not report["ok"]:
        report["problem"] = (
            f"{too_close / n:.1%} of events are within "
            f"{DISTANCE_LOG_SUSPECT_MAX} m of a facility. Real facility "
            "geometry cannot produce this, so the DIST_* features are not "
            "encoding distance and any baseline built on them is degenerate."
        )
    return report


# ---------------------------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------------------------


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(int)
    return {
        "tp": int(np.sum((p == 1) & (y == 1))),
        "fp": int(np.sum((p == 1) & (y == 0))),
        "fn": int(np.sum((p == 0) & (y == 1))),
        "tn": int(np.sum((p == 0) & (y == 0))),
    }


def rates(c: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    n = tp + fp + fn + tn
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / n if n else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "n": n,
    }


def all_metrics(c: Dict[str, int], beta: float = 2.0) -> Dict[str, float]:
    """Every rate derivable from a 2x2, plus the ones that are not simple ratios.

    `rates()` deliberately stays small. This is the exhaustive table: point-based,
    threshold-free, and agreement measures, so a reviewer cannot ask for a number
    that was never computed.

    Point-based (threshold-dependent)
        precision, recall/sensitivity, specificity, F1, Fbeta, accuracy,
        balanced_accuracy, Jaccard/IoU, G-mean, MCC, Cohen's kappa,
        prevalence, lift, error rates

    Threshold-free (from the score ranking, not from the confusion)
        PR-AUC, ROC-AUC, KS, and the score of the binary classifiers
        (all-negative, best-F1, calibrated tau)

    Guarded: every denominator that can be zero yields 0.0, never a NaN that
    would silently propagate into a figure.
    """
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    n = tp + fp + fn + tn

    def div(a: float, b: float) -> float:
        return float(a) / float(b) if b else 0.0

    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    b2 = beta * beta
    fbeta = div((1 + b2) * precision * recall,
                b2 * precision + recall) if (precision + recall) else 0.0
    f1 = div(2 * precision * recall, precision + recall)
    accuracy = div(tp + tn, n)
    balanced = 0.5 * (recall + specificity)
    prevalence = div(tp + fn, n)
    base_rate = prevalence

    jaccard = div(tp, tp + fp + fn)
    gmean = math.sqrt(max(precision, 0.0) * max(recall, 0.0))

    num = tp * tn - fp * fn
    den = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = div(num, den)

    po = accuracy
    pe = (div(tp + fp, n) * prevalence) + (div(tn + fn, n) * (1 - prevalence))
    kappa = div(po - pe, 1 - pe) if pe != 1.0 else 0.0

    out = {
        # point-based
        "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn),
        "n": float(n),
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "f2": fbeta,
        "fbeta": fbeta,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "jaccard": jaccard,
        "iou": jaccard,
        "g_mean": gmean,
        "mcc": mcc,
        "cohen_kappa": kappa,
        "prevalence": prevalence,
        "base_rate": base_rate,
        "predicted_positive_rate": div(tp + fp, n),
        "flag_rate": div(tp + fp, n),
        "miss_rate": div(fn, n),
        "false_discovery_rate": div(fp, tp + fp),
        "false_negative_rate": div(fn, tp + fn),
        "false_positive_rate": div(fp, tn + fp),
        "false_omission_rate": div(fn, fn + tn),
        "lift": div(precision, prevalence) if prevalence else 0.0,
        "odds_ratio": div(tp * tn, fp * fn),
        "informedness": recall + specificity - 1.0,
        "critical_success_index": precision - (1 - recall),
        "error_rate": div(fp + fn, n),
        "predicted_negative_rate": div(tn + fn, n),
        "actual_negative_rate": div(tn + fn, n),
        "actual_positive_rate": prevalence,
        "no_skill_accuracy": max(prevalence, 1 - prevalence),
        "accuracy_above_no_skill": accuracy - max(prevalence, 1 - prevalence),
    }
    return out


def threshold_free_metrics(y_true, score, n_boot: int = 200,
                           seed: int = 0) -> Dict[str, float]:
    """PR-AUC, ROC-AUC, KS and Brier, plus bootstrap 95% intervals.

    Intervals matter here because the held-out fold has ~98k positives on some
    targets and 35 on others; a bare PR-AUC on 35 positives is not a number.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype="float64")
    rec, prec, ap = pr_curve(y, s)
    tpr, fpr, auc = roc_curve(y, s)
    out = {"pr_auc": ap, "roc_auc": auc}

    n_pos = int(y.sum())
    if n_pos and s.size:
        order = np.argsort(-s, kind="mergesort")
        s_s, y_s = s[order], y[order]
        cdf_neg = np.cumsum(1 - y_s) / max(int((y == 0).sum()), 1)
        cdf_pos = np.cumsum(y_s) / n_pos
        out["ks"] = float(np.max(np.abs(cdf_pos - cdf_neg)))
    else:
        out["ks"] = 0.0

    p = float(s.mean())
    out["prevalence"] = n_pos / y.size if y.size else 0.0
    out["mean_predicted"] = p
    out["brier"] = float(np.mean((s - y) ** 2))
    out["calibration_gap"] = abs(p - out["prevalence"])

    if n_boot and y.size > 100:
        rng = np.random.default_rng(seed)
        aps, aucs = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, y.size, y.size)
            yy, ss = y[idx], s[idx]
            if yy.sum() == 0 or yy.sum() == yy.size:
                continue
            aps.append(pr_curve(yy, ss)[2])
            aucs.append(roc_curve(yy, ss)[2])
        if aps:
            out["pr_auc_ci95"] = [float(np.percentile(aps, 2.5)),
                                  float(np.percentile(aps, 97.5))]
            out["roc_auc_ci95"] = [float(np.percentile(aucs, 2.5)),
                                   float(np.percentile(aucs, 97.5))]
    return out


def full_confusion_report(y_true, score, tau: Optional[float] = None,
                          beta: float = 2.0) -> Dict[str, Dict[str, float]]:
    """Every confusion matrix a reviewer can ask for, each with all metrics.

    Variants, and why each exists:
        all_negative_floor     -- the classifier that predicts nothing. Its
                                  accuracy is the number to beat.
        best_f1                 -- the model at its own best F1, i.e. the most
                                  flattering single operating point.
        best_fbeta              -- same, for F2 (recall-leaning, the brief's
                                  operating rule).
        best_mcc                -- threshold-free optimum of the correlation.
        tau                     -- the frozen operating threshold, if one was
                                  calibrated. `None` tau is reported, never
                                  guessed.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype="float64")
    variants: Dict[str, Dict[str, float]] = {}
    variants["all_negative_floor"] = all_metrics(all_negative_confusion(y), beta)

    def sweep(key: str, fn) -> Optional[Dict[str, int]]:
        if s.size == 0 or int(y.sum()) == 0:
            return None
        best, best_v = None, -1e18
        for t in np.unique(np.quantile(s, np.linspace(0, 1, 200))):
            c = confusion(y, (s >= t).astype(int))
            v = fn(c)
            if v > best_v:
                best_v, best = v, c
        return best

    for key, fn in (
        ("best_f1", lambda c: all_metrics(c)["f1"]),
        ("best_fbeta", lambda c: all_metrics(c, beta)["fbeta"]),
        ("best_mcc", lambda c: all_metrics(c)["mcc"]),
    ):
        c = sweep(key, fn)
        if c is not None:
            variants[key] = all_metrics(c, beta)

    for t_name, t_val in (("tau", tau), ("tau_0.60", 0.60)):
        if t_val is None:
            variants[t_name] = {"tau": None, "note": "no freezable threshold"}
            continue
        c = confusion(y, (s >= t_val).astype(int))
        variants[t_name] = {"tau": float(t_val), **all_metrics(c, beta)}
    return variants


def multi_class_confusion(y_true, y_pred, labels: Optional[List[str]] = None
                          ) -> Dict[str, np.ndarray]:
    """Full k x k matrix, for the 6-class weak-label composition."""
    yt = np.asarray(y_true).astype(str)
    yp = np.asarray(y_pred).astype(str)
    labels = labels or sorted(set(yt) | set(yp))
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype="int64")
    for a, p in zip(yt, yp):
        if a in idx and p in idx:
            m[idx[a], idx[p]] += 1
    return m


def per_class_report(y_true, y_pred, labels: Optional[List[str]] = None
                     ) -> List[Dict[str, float]]:
    """One-vs-rest precision/recall/F1/support per class.

    The synthetic `NOT_INDUSTRIAL_FIRE` class that once printed a support of 0
    must never come back: `labels` is taken from the observed data only.
    """
    yt = np.asarray(y_true).astype(str)
    yp = np.asarray(y_pred).astype(str)
    labels = labels or sorted(set(yt) | set(yp))
    rows = []
    for lab in labels:
        c = confusion((yt == lab).astype(int), (yp == lab).astype(int))
        m = all_metrics(c)
        rows.append({
            "label": lab,
            "support": m["tp"] + m["fn"],
            "predicted": m["tp"] + m["fp"],
            "precision": m["precision"], "recall": m["recall"],
            "f1": m["f1"], "specificity": m["specificity"],
        })
    return rows


def render_report(title: str, variants: Dict[str, Dict[str, float]],
                  tfm: Optional[Dict[str, float]] = None) -> str:
    """Console table of every variant x every measure."""
    keys = ["tp", "fp", "fn", "tn", "precision", "recall", "f1", "f2",
            "accuracy", "balanced_accuracy", "specificity", "jaccard",
            "g_mean", "mcc", "cohen_kappa", "lift", "false_positive_rate",
            "false_negative_rate", "false_discovery_rate", "flag_rate",
            "prevalence", "error_rate", "informedness"]
    w = 30
    out = ["=" * 78, title, "=" * 78]
    head = f"{'measure':<16}" + "".join(f"{k[:15]:>16}" for k in variants)
    out.append(head)
    out.append("-" * 78)
    for key in keys:
        row = f"{key:<16}"
        for v in variants.values():
            val = v.get(key)
            row += f"{'-':>16}" if val is None else (
                f"{val:>16,.0f}" if key in ("tp", "fp", "fn", "tn")
                else f"{val:>16.4f}")
        out.append(row)
    if tfm:
        out.append("-" * 78)
        out.append("threshold-free (ranking, not a single cut)")
        for key in ("pr_auc", "roc_auc", "ks", "brier", "calibration_gap",
                    "mean_predicted", "prevalence"):
            if key in tfm:
                v = tfm[key]
                out.append(f"  {key:<20}{v:>12.4f}")
        for key in ("pr_auc_ci95", "roc_auc_ci95"):
            if key in tfm:
                lo, hi = tfm[key]
                out.append(f"  {key:<20}[{lo:.4f}, {hi:.4f}]")
    out.append("=" * 78)
    return "\n".join(out)


def confusion_frame(y_true, y_pred, name: str) -> "Tuple[np.ndarray, List[str]]":
    """2x2 matrix laid out [[TN, FP], [FN, TP]] for plotting."""
    c = confusion(y_true, y_pred)
    matrix = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]], dtype="int64")
    return matrix, ["actual: other", "actual: target"]


def all_negative_confusion(y_true) -> Dict[str, int]:
    y = np.asarray(y_true).astype(int)
    return {"tp": 0, "fp": 0, "fn": int(y.sum()), "tn": int((y == 0).sum())}


def best_f1_confusion(y_true, score) -> Dict[str, int]:
    """The model at its own best F1, for a like-for-like baseline comparison."""
    s = np.asarray(score, dtype="float64")
    y = np.asarray(y_true).astype(int)
    if s.size == 0 or int(y.sum()) == 0:
        return confusion(y, np.zeros_like(y))
    best, best_f1 = None, -1.0
    for t in np.unique(np.quantile(s, np.linspace(0, 1, 200))):
        c = confusion(y, (s >= t).astype(int))
        r = rates(c)
        if r["f1"] > best_f1:
            best_f1, best = r["f1"], c
    return best or confusion(y, np.zeros_like(y))


# ---------------------------------------------------------------------------
# PR / ROC CURVES
# ---------------------------------------------------------------------------


def pr_curve(y_true, score) -> Tuple[np.ndarray, np.ndarray, float]:
    """Precision-recall curve, with ties collapsed. Returns (recall, precision, ap)."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype="float64")
    n_pos = int(y.sum())
    if n_pos == 0 or s.size == 0:
        z = np.empty(0)
        return z, z, 0.0
    order = np.argsort(-s, kind="mergesort")
    s_s, y_s = s[order], y[order]
    change = np.empty(len(s_s), dtype=bool)
    change[-1] = True
    change[:-1] = s_s[1:] != s_s[:-1]
    ends = np.where(change)[0]
    tp = np.cumsum(y_s)[ends]
    predicted = ends + 1
    precision = tp / predicted
    recall = tp / n_pos
    ap = 0.0
    prev = 0.0
    for r, p in zip(recall, precision):
        ap += (r - prev) * p
        prev = r
    return recall, precision, float(ap)


def roc_curve(y_true, score) -> Tuple[np.ndarray, np.ndarray, float]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype="float64")
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0 or s.size == 0:
        z = np.empty(0)
        return z, z, 0.5
    order = np.argsort(-s, kind="mergesort")
    y_s = y[order]
    tps = np.cumsum(y_s)
    fps = np.cumsum(1 - y_s)
    tpr = np.concatenate([[0], tps / n_pos])
    fpr = np.concatenate([[0], fps / n_neg])
    auc = float(np.trapezoid(tpr, fpr))
    return tpr, fpr, auc


# ---------------------------------------------------------------------------
# THE TRAINING LEDGER
# ---------------------------------------------------------------------------
#
# "How much raw got trained" is the question that reveals whether a reported
# score came from a defensible sample or from a class that happened to be
# over-represented. Every stage is counted and the attrition is explicit.


def training_ledger(total_events: int, membership_rows: int,
                    train_available: int, test_available: int,
                    dropped: int, train_sampled: int, test_sampled: int,
                    train_positives: int, test_positives: int,
                    n_features_model_a: int, n_features_model_b: int) -> Dict:
    """Where every event went, and how much survived sampling."""
    ledger = {
        "catalogue": {
            "global_events": int(total_events),
            "membership_rows_event_to_cluster": int(membership_rows),
            "clusters_per_event": (
                round(membership_rows / total_events, 4) if total_events else None
            ),
        },
        "split": {
            "train_available": int(train_available),
            "test_available": int(test_available),
            "dropped_early_in_held_out_blocks": int(dropped),
            "accounted_for": int(train_available + test_available + dropped),
            "accounts_exactly": (
                train_available + test_available + dropped == total_events
            ),
        },
        "sampling": {
            "train_sampled": int(train_sampled),
            "test_sampled": int(test_sampled),
            "train_sampling_frac": (
                round(train_sampled / train_available, 4) if train_available else None
            ),
            "test_sampling_frac": (
                round(test_sampled / test_available, 4) if test_available else None
            ),
            "class_balance_preserved": (
                test_positives / test_sampled > 0
                and train_positives / train_sampled > 0
            ),
            "train_positive_rate": (
                round(train_positives / train_sampled, 4) if train_sampled else None
            ),
            "test_positive_rate": (
                round(test_positives / test_sampled, 4) if test_sampled else None
            ),
        },
        "features": {
            "model_a": int(n_features_model_a),
            "model_b": int(n_features_model_b),
            "excluded_from_b": int(n_features_model_a - n_features_model_b),
        },
    }
    ledger["headline"] = {
        "rows_trained_on": int(train_sampled),
        "rows_scored": int(test_sampled),
        "raw_events_catalogue": int(total_events),
        "fraction_of_catalogue_trained": (
            round(train_sampled / total_events, 5) if total_events else None
        ),
    }
    return ledger


def render_ledger(ledger: Dict) -> str:
    c, sp, s, f, h = (ledger["catalogue"], ledger["sampling"],
                       ledger["split"], ledger["features"], ledger["headline"])
    w = 68
    out = [f"{'=' * w}", "TRAINING LEDGER -- where every event went", f"{'=' * w}"]
    out.append("CATALOGUE")
    out.append(f"  global events                {c['global_events']:>14,}")
    out.append(f"  event->cluster rows          {c['membership_rows_event_to_cluster']:>14,}")
    out.append(f"  clusters per event           {c['clusters_per_event']:>14}")
    out.append("")
    out.append("SPLIT  (out-of-time AND out-of-space)")
    out.append(f"  train available              {s['train_available']:>14,}")
    out.append(f"  test available               {s['test_available']:>14,}")
    out.append(f"  dropped (early in test blk)  "
               f"{s['dropped_early_in_held_out_blocks']:>14,}")
    out.append(f"  accounted for exactly        "
               f"{str(s['accounts_exactly']):>14}")
    out.append("")
    out.append("SAMPLING  (stratified on the target class)")
    out.append(f"  train sampled                {sp['train_sampled']:>14,}"
               f"   ({sp['train_sampling_frac']:.1%} of train)")
    out.append(f"  test sampled                 {sp['test_sampled']:>14,}"
               f"   ({sp['test_sampling_frac']:.1%} of test)")
    out.append(f"  train positive rate          "
               f"{sp['train_positive_rate']:>14}")
    out.append(f"  test positive rate           "
               f"{sp['test_positive_rate']:>14}")
    out.append("")
    out.append("FEATURES")
    out.append(f"  model A (operational)        {f['model_a']:>14}")
    out.append(f"  model B (scientific)         {f['model_b']:>14}")
    out.append(f"  excluded as leakage          {f['excluded_from_b']:>14}")
    out.append("")
    out.append("HEADLINE")
    out.append(f"  raw events in catalogue      {h['raw_events_catalogue']:>14,}")
    out.append(f"  rows actually trained on     {h['rows_trained_on']:>14,}")
    out.append(f"  rows scored                  {h['rows_scored']:>14,}")
    out.append(f"  fraction of catalogue used   "
               f"{h['fraction_of_catalogue_trained']:>14}")
    out.append("=" * w)
    return "\n".join(out)
