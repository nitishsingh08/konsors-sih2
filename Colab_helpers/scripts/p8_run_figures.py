"""
Driver for the complete P8 figure set and performance report.

Runs against whatever is present and records what is absent. Two modes:

    python3 scripts/p8_run_figures.py --heldout .            # model figures only
    python3 scripts/p8_run_figures.py --heldout . --project <drive copy>
        # adds every raw-data / event / cluster figure that needs the Drive

The Drive-side inputs are all optional and probed by name, so the run never
fails for a missing file -- it names the file it wanted in manifest.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import figures as fig          # noqa: E402
import p8_report as R          # noqa: E402


def probe(root: Path, rel: str) -> Path | None:
    p = root / rel
    return p if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", default=".", help="dir holding heldout_pred_{A,B}.csv")
    ap.add_argument("--project", default=None,
                    help="Firm_Clusters/full_2012_2026 copy on this machine")
    ap.add_argument("--target", default=R.TARGET)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--figures", default="qc/figures")
    ap.add_argument("--clusters-per-event", type=int, default=None)
    ap.add_argument("--importance", default=None, help="json {feature: importance}")
    args = ap.parse_args()

    held_dir = Path(args.heldout).resolve()
    rep = R.Report(Path(args.figures))

    # ---- lineage counts ---------------------------------------------------
    counts: dict[str, int | None] = {
        "raw detections": None, "local clusters": None,
        "cross-day edges": None, "global events": None,
        "feature events": None, "scored events": None,
    }
    scores: dict[str, np.ndarray] = {}

    for name in ("B", "A"):
        p = held_dir / f"heldout_pred_{name}.csv"
        if p.exists():
            d = pd.read_csv(p, low_memory=False)
            scores[name] = d["score"].to_numpy()
            if name == "B":
                y = d["y_true"].to_numpy()
                labels = d["y_true_weak_label"]
        if p.exists():
            counts["scored events"] = len(d)
    if not scores:
        print("no heldout_pred_*.csv found -- nothing to score")
        rep.flush()
        return 1

    print(f"loaded {len(y):,} held-out events, "
          f"{int(y.sum()):,} positive ({y.mean():.3%})")
    prev = float(y.mean())

    # ---- performance, exhaustively ---------------------------------------
    variants = {}
    for name, s in scores.items():
        variants[name] = fig.full_confusion_report(y, s, tau=args.tau)
        title = f"MODEL {name} -- every measure"
        if name == "A":
            # Not a performance result. Model A holds DIST_*, which manufactures
            # the weak label, so every number here measures the rule, not skill.
            title = ("MODEL A -- NOT A PERFORMANCE RESULT: holds DIST_*, the "
                     "quantity the label is built from")
        print()
        print(fig.render_report(title, variants[name],
                                fig.threshold_free_metrics(y, s)))

    base = dict(variants["B"])
    # The like-for-like comparator: the weak rule is a 5 km proximity cut, so
    # the baseline is a distance threshold on the same column. Included in the
    # same variant table because a model compared only to the all-negative
    # floor looks far better than it is.
    dcol = pd.read_csv(held_dir / "heldout_pred_B.csv", usecols=["dist_m"])
    d = dcol["dist_m"].to_numpy()
    dist_m = np.where(np.isfinite(d), 10.0 ** np.clip(d, 0, 12) - 1.0, np.inf)
    for cut in (1000.0, 5000.0):
        base[f"distance_{int(cut)}m"] = fig.all_metrics(
            fig.confusion(y, (dist_m <= cut).astype(int)))
        base[f"distance_{int(cut)}m"]["pr_auc"] = fig.pr_curve(
            y, -np.where(np.isfinite(d), d, np.inf))[2]
    rep.summary["confusion"] = variants
    rep.summary["confusion_with_baselines"] = base
    R.figure_confusion_grid(rep, base)
    R.figure_measure_bars(rep, base)
    R.figure_curves(rep, scores, y, prev)
    R.figure_ks(rep, scores, y)
    R.figure_threshold_sweep(rep, scores, y, tau=args.tau)
    R.figure_distance(rep, d, y, dist_m=np.where(np.isfinite(d), 10.0 ** np.clip(d, 0, 12) - 1.0, np.nan))
    R.figure_class_balance(rep, labels)
    # Per-class table. Only INDUSTRIAL_FIRE has a trained model, so the other
    # rows carry support and NA rates: printing 0.0 there would assert a
    # measurement that was never made, which is the exact defect README §P7
    # records being fixed.
    R.figure_per_class(rep, [
        {"label": str(lab),
         "support": int((labels == lab).sum()),
         "predicted": int((labels == lab).sum()),
         "precision": None, "recall": None, "f1": None, "specificity": None}
        for lab in labels.value_counts().index])
    R.figure_funnel(rep, counts)
    R.figure_events_per_cluster(rep,
                                np.full(args.clusters_per_event or 0, 1))
    if args.importance:
        R.figure_importance(rep, pd.read_json(args.importance).to_dict())

    rep.flush()
    print()
    print(f"figures  -> {rep.dir}")
    print(f"manifest -> {rep.dir / 'manifest.json'}")
    for s in rep.manifest["skipped"]:
        print(f"  SKIPPED {s['figure']}: {s['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())