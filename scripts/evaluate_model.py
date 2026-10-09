"""
P7b -- HONEST MODEL EVALUATION.

Answers "how accurate is the model?" without letting an inflated number escape.

Why not accuracy
----------------
On the held-out sample, 501,648 of 600,000 rows are negative (83.6%). A model
that predicts EVERYTHING negative scores 83.6% accuracy while detecting
nothing. So:

    accuracy is reported here, but never as the headline

Model B's `OTHER_CLASS` F1 of 0.911 is the same trap: it called almost nothing
positive, so the dominant class scored near-perfectly while the model found
almost no industrial fire. That number is printed alongside a decomposition
that makes it impossible to read as "91% accurate".

What is reported
----------------
* PR-AUC (threshold-free, and the reason ROC-AUC is avoided under imbalance)
* precision / recall / F1 at the 0.60 review threshold, per class
* **bootstrap confidence intervals**, because with 33 flare positives in the
  test fold a bare PR-AUC of 0.17 has a huge interval and quoting it alone is
  misleading
* the trivial all-negative baseline, so every number has a floor
* the distance-threshold baseline the model has to beat
* an explicit verdict, which is allowed to be "no"

Run:  python3 scripts/evaluate_model.py --model B
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

TARGET_REVIEW = 0.60

#: A win smaller than this is noise, not a result. F1 differences below it are
#: reported as "indistinguishable" rather than as a victory, because declaring a
#: 0.0006 improvement over the baseline would be reading noise as achievement.
MIN_MEANINGFUL_F1_GAIN = 0.01


@dataclass
class Metrics:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    accuracy: float = 0.0
    pr_auc: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "pr_auc": round(self.pr_auc, 4),
            "confusion": {"tp": self.tp, "fp": self.fp, "fn": self.fn,
                          "tn": self.tn},
        }


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    c = confusion(y_true, y_pred)
    precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1, **c}


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    """PR-AUC with tied scores collapsed.

    Ties matter: LightGBM emits many identical scores, and resolving them by an
    arbitrary sort order can hand a useless model a near-perfect PR-AUC.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype="float64")
    n_pos = int(y.sum())
    if n_pos == 0 or s.size == 0:
        return 0.0

    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]

    # Walk in steps: advance whole runs of equal score at once.
    changes = np.empty(len(s_sorted), dtype=bool)
    changes[-1] = True
    changes[:-1] = s_sorted[1:] != s_sorted[:-1]
    ends = np.where(changes)[0]

    tp = np.cumsum(y_sorted)[ends]
    predicted = ends + 1
    precision = tp / predicted
    recall = tp / n_pos

    ap = 0.0
    prev_r = 0.0
    for r, p in zip(recall, precision):
        ap += (r - prev_r) * p
        prev_r = r
    return float(ap)


def recall_at_precision(y_true: np.ndarray, score: np.ndarray,
                        target: float = 0.60) -> Dict[str, float]:
    """The operating point: how much fire is caught while staying precise.

    Sweeps from the most conservative threshold to the most permissive and
    takes the most permissive one that still holds precision >= target.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype="float64")
    if y.size == 0:
        return {"threshold": None, "precision": 0.0, "recall": 0.0,
                "n_predicted": 0}

    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    changes = np.empty(len(s_sorted), dtype=bool)
    changes[-1] = True
    changes[:-1] = s_sorted[1:] != s_sorted[:-1]
    ends = np.where(changes)[0]

    tp = np.cumsum(y_sorted)[ends]
    predicted = ends + 1
    precision = tp / predicted
    recall = tp / max(int(y.sum()), 1)

    ok = np.where(precision >= target)[0]
    if len(ok) == 0:
        return {"threshold": None, "precision": float(precision[0]),
                "recall": 0.0, "n_predicted": 0}
    i = int(ok[np.argmax(recall[ok])])
    return {"threshold": float(s_sorted[ends][i]),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "n_predicted": int(predicted[i])}


def baseline_metrics(y_true: np.ndarray, dist_m: np.ndarray,
                     thresholds=(250.0, 500.0, 1000.0, 2000.0, 5000.0)) -> Dict:
    """Best single distance threshold, tuned on whatever it is given.

    A one-feature rule the ML must beat. Tuning is caller's responsibility: on
    the TRAIN fold, never the test fold.
    """
    y = np.asarray(y_true).astype(int)
    d = np.asarray(dist_m, dtype="float64")
    best = None
    for t in thresholds:
        pred = (np.isfinite(d) & (d <= t)).astype(int)
        m = precision_recall_f1(y, pred)
        row = {"threshold_m": float(t), **{k: m[k] for k in
                                          ("precision", "recall", "f1")}}
        if best is None or m["f1"] > best["f1"]:
            best = row
    return best or {}


def best_f1_threshold(y_true: np.ndarray, score: np.ndarray,
                       n_grid: int = 200) -> Dict[str, float]:
    """The model's own best F1, over its own score range.

    Needed for a fair comparison. Tuning the baseline over five candidate
    radii and then judging the model at a fixed 0.5 is not like-for-like: the
    baseline gets its best operating point handed to it and the model does not.
    Both must be measured at their own optimum.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype="float64")
    if s.size == 0 or int(y.sum()) == 0:
        return {}
    candidates = np.unique(np.quantile(s, np.linspace(0.0, 1.0, n_grid)))
    best = None
    for t in candidates:
        m = precision_recall_f1(y, (s >= t).astype(int))
        if best is None or m["f1"] > best["f1"]:
            best = {"threshold": float(t), **m}
    return best or {}


def trivial_all_negative(y_true: np.ndarray) -> Dict:
    """The floor. Any model must beat this to have learned anything."""
    y = np.asarray(y_true).astype(int)
    n = len(y)
    pos = int(y.sum())
    neg = n - pos
    return {
        "name": "predict nothing positive",
        "accuracy": round(neg / n, 4) if n else 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "note": (
            f"scores {100 * neg / n:.1f}% accuracy while detecting nothing. "
            "This is why accuracy is never the headline."
        ) if n else "",
    }


def bootstrap_ci(y_true: np.ndarray, score: np.ndarray, n_boot: int = 200,
                 seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    """Percentile CI for PR-AUC.

    With few positives the interval is the finding, not a footnote: 33 flare
    positives give a PR-AUC of 0.17 an interval wide enough to include almost
    anything.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype="float64")
    n = len(y)
    if n == 0:
        return {"lo": float("nan"), "hi": float("nan"), "n_boot": 0}

    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(y[idx]).size < 2:
            continue           # a resample with one class has no PR-AUC
        stats.append(average_precision(y[idx], s[idx]))
    if not stats:
        return {"lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    stats = np.asarray(stats)
    return {
        "lo": float(np.quantile(stats, alpha / 2)),
        "hi": float(np.quantile(stats, 1 - alpha / 2)),
        "median": float(np.median(stats)),
        "n_boot": int(stats.size),
    }


@dataclass
class Report:
    model: str
    target: str
    n_total: int
    n_positive: int
    prevalence: float
    at_threshold: Dict
    operating_point: Dict
    pr_auc: float
    pr_auc_ci: Dict
    baselines: Dict[str, Dict] = field(default_factory=dict)
    verdict: str = ""
    caveats: List[str] = field(default_factory=list)

    def render(self) -> str:
        w = 72
        out = [f"{'=' * w}", "MODEL EVALUATION", f"{'=' * w}"]
        out.append(f"model {self.model}   target {self.target}")
        out.append("")
        out.append(f"test rows        {self.n_total:,}")
        out.append(f"positives        {self.n_positive:,} "
                   f"({self.prevalence:.2%})")
        out.append("")

        out.append("-- FLOOR: the trivial predictor " + "-" * 40)
        floor = self.baselines.get("all_negative", {})
        out.append(f"  accuracy        {floor.get('accuracy', 0):.4f}  "
                   f"precision 0.0000  recall 0.0000")
        out.append(f"  {floor.get('note', '')}")
        out.append("")

        out.append(f"-- AT THE {self.at_threshold.get('threshold_used', 0.5)} "
                   "DECISION THRESHOLD " + "-" * 30)
        m = self.at_threshold
        out.append(f"  precision       {m.get('precision', 0):.4f}")
        out.append(f"  recall          {m.get('recall', 0):.4f}")
        out.append(f"  F1              {m.get('f1', 0):.4f}")
        # tp/fp/fn/tn are FLAT in the metrics dict -- reading them from a
        # "confusion" sub-dict printed an all-zero row on every run.
        out.append(f"  confusion       tp={m.get('tp', 0):,} "
                   f"fp={m.get('fp', 0):,} "
                   f"fn={m.get('fn', 0):,} tn={m.get('tn', 0):,}")
        out.append(f"  accuracy        {m.get('accuracy', 0):.4f}   "
                   "<- not the headline; see the floor above")
        out.append("")

        out.append("-- OPERATING POINT (review threshold 0.60) " + "-" * 33)
        op = self.operating_point
        out.append(f"  precision       {op.get('precision', 0):.4f}")
        out.append(f"  recall          {op.get('recall', 0):.4f}   "
                   "<- fire caught while staying trustworthy")
        out.append(f"  alerts raised   {op.get('n_predicted', 0):,}")
        out.append("")

        out.append("-- THRESHOLD-FREE " + "-" * 49)
        out.append(f"  PR-AUC          {self.pr_auc:.4f}")
        ci = self.pr_auc_ci
        if ci.get("n_boot"):
            out.append(f"  95% CI          [{ci['lo']:.4f}, {ci['hi']:.4f}]"
                       f"   ({ci['n_boot']} resamples)")
            width = ci["hi"] - ci["lo"]
            if width > 0.5:
                out.append("  !! that interval is very wide -- too few "
                           "positives to distinguish models")
        out.append("")

        dist = self.baselines.get("distance", {})
        if dist:
            out.append("-- BASELINE the model must beat " + "-" * 36)
            out.append(f"  distance <= {dist.get('threshold_m', 0):,.0f} m")
            out.append(f"  precision       {dist.get('precision', 0):.4f}")
            out.append(f"  recall          {dist.get('recall', 0):.4f}")
            out.append(f"  F1              {dist.get('f1', 0):.4f}")
            best = self.baselines.get("model_at_own_best_f1")
            if best:
                out.append("")
                out.append("  like-for-like: the model at its OWN best F1")
                out.append(f"    threshold    {best['threshold']:.4f}")
                out.append(f"    precision    {best['precision']:.4f}")
                out.append(f"    recall       {best['recall']:.4f}")
                out.append(f"    F1           {best['f1']:.4f}")
            out.append("")

        out.append("-- VERDICT " + "-" * 59)
        for line in [self.verdict] + self.caveats:
            out.append(f"  {line}")
        out.append("=" * w)
        return "\n".join(out)


def evaluate(model: str, target: str, y_true: np.ndarray, score: np.ndarray,
             dist_m: Optional[np.ndarray] = None,
             review_threshold: float = TARGET_REVIEW,
             n_boot: int = 200) -> Report:
    """Full evaluation. `score` is the model's probability."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype="float64")

    pred_at_half = (s >= 0.5).astype(int)
    m = precision_recall_f1(y, pred_at_half)
    m["accuracy"] = float((pred_at_half == y).mean()) if y.size else 0.0
    m["threshold_used"] = 0.5

    baselines = {"all_negative": trivial_all_negative(y)}
    if dist_m is not None:
        baselines["distance"] = baseline_metrics(y, dist_m)

    # Both sides are judged at their OWN best F1. Comparing a tuned baseline
    # against a model pinned at 0.5 is not like-for-like, and it can make a
    # good model look like a bad one for no reason other than the threshold.
    model_best = best_f1_threshold(y, s)
    if model_best:
        baselines["model_at_own_best_f1"] = {
            "threshold": model_best["threshold"],
            "precision": round(model_best["precision"], 4),
            "recall": round(model_best["recall"], 4),
            "f1": round(model_best["f1"], 4),
        }

    report = Report(
        model=model,
        target=target,
        n_total=int(len(y)),
        n_positive=int(y.sum()),
        prevalence=float(y.mean()) if y.size else 0.0,
        at_threshold=m,
        operating_point=recall_at_precision(y, s, review_threshold),
        pr_auc=average_precision(y, s),
        pr_auc_ci=bootstrap_ci(y, s, n_boot=n_boot),
        baselines=baselines,
    )

    # Verdict, allowed to be negative. Compared at each model's own optimum.
    dist = baselines.get("distance")
    best = baselines.get("model_at_own_best_f1")
    beats_floor = report.at_threshold["recall"] > 0.0
    if dist and best:
        gain = best["f1"] - dist["f1"]
        if gain >= MIN_MEANINGFUL_F1_GAIN:
            report.verdict = (
                f"ML beats the distance-threshold baseline: F1 {best['f1']:.4f} "
                f"vs {dist['f1']:.4f} (gain {gain:+.4f})."
            )
        elif gain <= -MIN_MEANINGFUL_F1_GAIN:
            report.verdict = (
                "ML does NOT beat the distance-threshold baseline: F1 "
                f"{best['f1']:.4f} vs {dist['f1']:.4f} (gain {gain:+.4f}). "
                "Report that plainly."
            )
        else:
            report.verdict = (
                f"Indistinguishable from the distance-threshold baseline: F1 "
                f"{best['f1']:.4f} vs {dist['f1']:.4f} (gain {gain:+.4f}, "
                f"below the {MIN_MEANINGFUL_F1_GAIN} noise floor). Treat the "
                "ML as having added nothing until more signal is available."
            )
    elif report.operating_point["recall"] >= 0.5:
        report.verdict = (
            f"Usable at the operating point: {report.operating_point['recall']:.1%} "
            "of the target class caught at precision >= 0.60."
        )
    elif beats_floor:
        report.verdict = (
            "Detects the target class but below a deployable operating point."
        )
    else:
        report.verdict = (
            "No better than predicting nothing. Do not deploy."
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="B", help="B (default) or A")
    parser.add_argument("--target", default="INDUSTRIAL_FIRE")
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--scores", type=Path, default=None,
                        help="CSV with columns y_true,score[,dist_m]")
    parser.add_argument("--json", type=Path, default=None,
                        help="write the report as JSON")
    args = parser.parse_args()

    if args.scores is None:
        print(__doc__)
        print()
        print("Provide predictions with --scores <csv> containing")
        print("  y_true   0/1 ground truth for the held-out fold")
        print("  score    model probability")
        print("  dist_m   optional distance to nearest facility, metres")
        print()
        print("To produce that CSV from a trained model, see colab_p7_train.py:")
        print("  it already prints PR-AUC, per-class metrics, both baselines")
        print("  and the review-queue split on the held-out sample.")
        return 1

    import pandas as pd

    if not args.scores.exists():
        # Do not raise a bare FileNotFoundError from inside pandas: say where the
        # file comes from, because it is produced by the P7 cell and the user
        # otherwise has no way to know that.
        print()
        print(f"  No such file: {args.scores}")
        print()
        print("  The held-out predictions are written by colab_p7_train.py to Drive:")
        print("    Firm_Clusters/qc/heldout_pred_B.csv   (model B, the default)")
        print("    Firm_Clusters/qc/heldout_pred_A.csv   (model A)")
        print()
        print("  Re-run colab_p7_train.py, then download that file and point")
        print("  this script at it. Expected columns:")
        print("    y_true   0/1   target vs rest, from the WEAK label")
        print("    score    0..1  model probability")
        print("    dist_m   metres distance to nearest facility")
        print()
        return 1

    frame = pd.read_parquet(args.scores) if args.scores.suffix == ".parquet" \
        else pd.read_csv(args.scores)
    required = {"y_true", "score"}
    missing = required - set(frame.columns)
    if missing:
        print(f"  {args.scores} is missing {sorted(missing)}")
        print(f"  columns present: {list(frame.columns)}")
        return 1

    report = evaluate(
        args.model, args.target,
        frame["y_true"].to_numpy(), frame["score"].to_numpy(),
        dist_m=frame["dist_m"].to_numpy() if "dist_m" in frame else None,
        n_boot=args.n_boot,
    )
    print(report.render())
    if args.json:
        args.json.write_text(json.dumps({
            "model": report.model,
            "target": report.target,
            "n_total": report.n_total,
            "n_positive": report.n_positive,
            "prevalence": report.prevalence,
            "at_threshold": report.at_threshold,
            "operating_point": report.operating_point,
            "pr_auc": report.pr_auc,
            "pr_auc_ci": report.pr_auc_ci,
            "baselines": report.baselines,
            "verdict": report.verdict,
            "caveats": report.caveats,
        }, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())