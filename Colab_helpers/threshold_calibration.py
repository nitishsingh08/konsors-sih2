# ============================================================================
# Operating-threshold calibration
#
# WHY THIS MODULE EXISTS
#
# The project statement fixes a review rule -- "any prediction below 60%
# confidence goes to human review" -- and HANDOFF §8 fixes a separate, stricter
# policy: freeze an operating threshold tau that maximises F2 subject to
# precision >= 0.90. The second one was written down and never implemented.
#
# `dataset_builder.REVIEW_CONFIDENCE = 0.60` and `model_registry.REVIEW_CONFIDENCE
# = 0.60` are bare literals. Applying a fixed 0.60 to an arbitrary model is only
# meaningful if the model happens to be calibrated to a 0.60 prior, and none of
# these are. Model B's scores sit around 0.13 on the held-out sample, so a 0.60
# cut flags 0.8% of the positives -- which is the "detects 2 of 100" symptom.
#
# The two numbers are NOT the same thing and must not be conflated:
#
#   review_confidence   0.60, mandated by the brief. Below it, a human looks.
#   operating_threshold calibrated here. Below it, the model has nothing to say.
#
# An event is auto-accepted only when it clears BOTH. See
# `effective_threshold`.
#
# WHAT IS DELIBERATELY NOT HERE
#
# No threshold is ever chosen on the test fold and then scored on that same fold.
# Every caller must pass TRAIN-fold scores; `colab_p7_train.py` does, and
# `assert_calibrated_on_train` exists to make the provenance explicit.
#
# Ties are collapsed via `training_design._tie_aware_curve` rather than sorted,
# because LightGBM emits large runs of equal scores and an arbitrary stable
# order inside a run inflates both precision and recall. That is the failure
# HANDOFF §9 records as "never let tied model scores be treated as ordered".
# ============================================================================

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from training_design import TARGET_PRECISION, _tie_aware_curve

#: F2 weights recall double, matching `training_design.binary_metrics`. The
#: brief's operating point is about catching fire, not about file size.
DEFAULT_BETA = 2.0

#: An "operating point" that flags more than this share of events is not an
#: operating point -- it is the trivial predictor wearing a threshold's clothes.
#: The value is reported rather than enforced, because on a genuinely weak model
#: the honest answer IS "there is no usable operating point", and hiding that
#: behind a rejection would be worse than stating it.
MAX_FLAG_FRACTION = 0.20


def f_beta(precision, recall, beta: float = DEFAULT_BETA):
    """F_beta. Undefined combinations return 0.0 rather than NaN."""
    p = np.asarray(precision, dtype="float64")
    r = np.asarray(recall, dtype="float64")
    b2 = float(beta) ** 2
    denom = b2 * p + r
    out = np.zeros(np.broadcast(p, r).shape, dtype="float64")
    ok = denom > 0
    out[ok] = (1.0 + b2) * p[ok] * r[ok] / denom[ok]
    return out if out.ndim else float(out)


def _empty(reason: str, **extra) -> Dict[str, object]:
    """A calibration that could not be performed, with the reason attached.

    `threshold` is None in every case here. Callers must treat None as "send
    everything to review", never as "no constraint".
    """
    payload = {
        "threshold": None,
        "precision": 0.0,
        "recall": 0.0,
        "f_beta": 0.0,
        "n_predicted": 0,
        "flag_fraction": 0.0,
        "degenerate": True,
        "feasible": False,
        "reason": reason,
    }
    payload.update(extra)
    return payload


def _check_inputs(y_true, score) -> None:
    y = np.asarray(y_true)
    s = np.asarray(score, dtype="float64")
    if y.ndim != 1 or s.ndim != 1:
        raise ValueError("y_true and score must both be 1-D")
    if y.size != s.size:
        raise ValueError(
            f"y_true has {y.size} rows but score has {s.size}; they must "
            "describe the same rows in the same order"
        )
    if not np.isin(np.unique(y), (0, 1)).all():
        raise ValueError(
            f"y_true must be 0/1, got values {np.unique(y)[:8].tolist()}"
        )


def calibrate_operating_threshold(
    y_true,
    score,
    *,
    min_precision: float = TARGET_PRECISION,
    beta: float = DEFAULT_BETA,
) -> Dict[str, object]:
    """Freeze tau: highest-F_beta threshold holding precision >= min_precision.

    Returns a dict that always carries `threshold`, and always carries enough
    for a reader to see WHY. On the current model B this returns
    `threshold: None` with an explanatory `reason`, which is the correct and
    publishable result: at any precision >= 0.90 that model has no recall to
    trade, so there is no operating point to freeze.

    `threshold` is a score value; predict with `score >= threshold`. The
    comparison must be `>=`, not `>`, so the calibrated point is included in the
    flagged set.
    """
    _check_inputs(y_true, score)
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype="float64")
    n = int(y.size)

    if n == 0:
        return _empty("no rows")
    n_pos = int(y.sum())
    if n_pos == 0:
        return _empty(
            "no positives in the calibration fold",
            n_rows=n, n_positives=0,
        )
    if n_pos == n:
        return _empty(
            "calibration fold is entirely positive; there is no negative class "
            "to hold precision against",
            n_rows=n, n_positives=n_pos,
        )

    scores, precision, recall, predicted = _tie_aware_curve(y, s)
    if scores.size == 0:
        return _empty("no distinct scores", n_rows=n, n_positives=n_pos)

    value = f_beta(precision, recall, beta)
    feasible = precision >= min_precision

    base = {
        "n_rows": n,
        "n_positives": n_pos,
        "prevalence": n_pos / n,
        "min_precision": float(min_precision),
        "beta": float(beta),
        "best_precision_anywhere": float(precision.max()),
    }

    if not feasible.any():
        best = int(np.argmax(value))
        return _empty(
            f"no threshold reaches precision {min_precision:.2f}; the best "
            f"precision anywhere on the curve is "
            f"{float(precision.max()):.4f}, at recall "
            f"{float(recall[best]):.4f}. There is no operating point to freeze.",
            **base,
        )

    candidates = np.flatnonzero(feasible)
    # argmax over feasible only. F_beta already rewards recall 2:1, so ties in
    # value are broken by the curve order, which is descending score.
    i = int(candidates[int(np.argmax(value[candidates]))])

    flagged = int(predicted[i])
    fraction = flagged / n
    return {
        "threshold": float(scores[i]),
        "precision": float(precision[i]),
        "recall": float(recall[i]),
        "f_beta": float(value[i]),
        "n_predicted": flagged,
        "flag_fraction": float(fraction),
        "degenerate": bool(fraction > MAX_FLAG_FRACTION),
        "feasible": True,
        "reason": (
            f"flags {fraction:.1%} of rows; an operating point above "
            f"{MAX_FLAG_FRACTION:.0%} is the trivial predictor, not a model"
            if fraction > MAX_FLAG_FRACTION
            else f"flags {fraction:.1%} of rows"
        ),
        **base,
    }


def threshold_for_recall(
    y_true,
    score,
    *,
    min_recall: float,
) -> Dict[str, object]:
    """Highest-precision threshold that still achieves `min_recall`.

    This is the diagnostic counterpart to the operating point. When the brief's
    precision >= 0.90 is unreachable, the useful question is not "what is the
    best threshold" but "what recall is actually available, and at what
    precision". That is a recall-constrained sweep, and it is how you tell a
    model that cannot discriminate (recall only reachable at base-rate
    precision) from one that merely has not been tuned.
    """
    _check_inputs(y_true, score)
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype="float64")
    n = int(y.size)
    if n == 0:
        return _empty("no rows")
    n_pos = int(y.sum())
    if n_pos == 0:
        return _empty("no positives in the calibration fold",
                      n_rows=n, n_positives=0)

    scores, precision, recall, predicted = _tie_aware_curve(y, s)
    if scores.size == 0:
        return _empty("no distinct scores", n_rows=n, n_positives=n_pos)

    reachable = recall >= min_recall
    base = {
        "n_rows": n,
        "n_positives": n_pos,
        "prevalence": n_pos / n,
        "min_recall": float(min_recall),
        "max_recall_reachable": float(recall.max()),
    }
    if not reachable.any():
        return _empty(
            f"recall {min_recall:.2f} is unreachable; the maximum on this fold "
            f"is {float(recall.max()):.4f} at threshold "
            f"{float(scores[-1]):.6g} (precision "
            f"{float(precision[-1]):.4f}, which is the class prevalence -- "
            "i.e. the model does not rank positives above negatives).",
            **base,
        )

    candidates = np.flatnonzero(reachable)
    # Among thresholds that reach the recall target, take the most precise.
    i = int(candidates[int(np.argmax(precision[candidates]))])
    fraction = int(predicted[i]) / n
    return {
        "threshold": float(scores[i]),
        "precision": float(precision[i]),
        "recall": float(recall[i]),
        "f_beta": float(f_beta(precision[i], recall[i])),
        "n_predicted": int(predicted[i]),
        "flag_fraction": float(fraction),
        "degenerate": bool(fraction > MAX_FLAG_FRACTION),
        "feasible": True,
        "reason": f"flags {fraction:.1%} of rows",
        **base,
    }


def effective_threshold(
    operating_threshold: Optional[float],
    review_confidence: float,
) -> float:
    """The bar an event must clear to be auto-accepted: max(tau, 0.60).

    Both constraints are real and they are not the same constraint.
    `review_confidence` is the brief's human-review floor. `operating_threshold`
    is where the model stops being confident. Auto-accept requires both, so the
    binding one is the larger. When no operating point could be frozen
    (threshold None) the brief's floor stands alone -- which correctly degrades
    to "almost everything is reviewed", never to "almost nothing is".
    """
    floor = float(review_confidence)
    if operating_threshold is None:
        return floor
    return max(floor, float(operating_threshold))


def assert_calibrated_on_train(fold: str) -> None:
    """Refuse to proceed if a threshold was calibrated on anything but train.

    Selecting tau on the test fold and then reporting that fold's recall is the
    quietest way to overstate a model, and `colab_p7_train.py` already carries
    that warning for the distance baseline. This makes it a type error instead
    of a habit.
    """
    if fold != "train":
        raise ValueError(
            f"thresholds must be calibrated on the TRAIN fold, not {fold!r}. "
            "Choosing tau on test and then reporting that fold's score inflates "
            "the result."
        )


def describe_calibration(calibration: Dict[str, object], name: str) -> str:
    """One line a human can read in a Colab log without any interpretation."""
    tau = calibration.get("threshold")
    if tau is None:
        return f"{name}: NO operating point -- {calibration.get('reason', '')}"
    return (
        f"{name}: tau={float(tau):.6f}  "
        f"precision={float(calibration['precision']):.4f}  "
        f"recall={float(calibration['recall']):.4f}  "
        f"F2={float(calibration['f_beta']):.4f}  "
        f"flags {100.0 * float(calibration['flag_fraction']):.1f}% of rows"
        + ("  [DEGENERATE]" if calibration.get("degenerate") else "")
    )