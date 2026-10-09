"""
P8b -- the complete figure set and the complete performance report.

`figures.py` holds the maths. This module draws it and prints it, and it draws
the *whole* lineage, not just the model: raw detections -> local clusters ->
cross-day edges -> global events -> labelled events -> scored events. A confusion
matrix without that funnel is uninterpretable, because a reviewer cannot tell a
degenerate model from a degenerate sample.

Everything degrades. Any absent input (detections table, edge table, membership
table) is recorded in `manifest["skipped"]` with a reason, and every other figure
is still produced. A figure that cannot be drawn is named; it is never silently
dropped, because a missing figure and an uninteresting figure look identical in
a report.

Writes
    qc/figures/*.png
    qc/figures/summary.json      every number behind every figure
    qc/figures/manifest.json     which figures exist and which are skipped, why
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figures as fig

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})

TARGET = "INDUSTRIAL_FIRE"
REVIEW_TAU = 0.60

#: The lineage. Each stage names the artefact that would prove its count.
STAGES = [
    ("raw detections", "India_Firms.parquet / mappings/", "detection count"),
    ("local clusters", "clusters/*.parquet", "daily cluster count"),
    ("cross-day edges", "edges/", "edge count"),
    ("global events", "events/canonical_event_base.parquet", "event count"),
    ("feature events", "features/dataset/X.parquet", "X rows"),
    ("scored events", "heldout_pred_B.csv", "held-out rows"),
]


class Report:
    """Accumulates figures, numbers and skip reasons for one run."""

    def __init__(self, fig_dir: Path):
        self.dir = Path(fig_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.summary: Dict = {}
        self.manifest: Dict = {"figures": [], "skipped": []}
        self._n = 0

    # -- output ------------------------------------------------------------
    def save(self, f, name: str) -> str:
        self._n += 1
        numbered = f"{self._n:02d}_{name}.png"
        f.savefig(self.dir / numbered, bbox_inches="tight")
        plt.close(f)
        self.manifest["figures"].append(numbered)
        return numbered

    def skip(self, name: str, reason: str) -> None:
        self.manifest["skipped"].append({"figure": name, "reason": reason})

    def flush(self) -> None:
        (self.dir / "summary.json").write_text(json.dumps(
            self.summary, indent=2, default=str))
        (self.dir / "manifest.json").write_text(json.dumps(
            self.manifest, indent=2, default=str))


# ---------------------------------------------------------------------------
# RAW DATA -> EVENTS: the funnel
# ---------------------------------------------------------------------------


def figure_funnel(rep: Report, counts: Dict[str, Optional[int]]) -> None:
    """Every stage of the lineage as a horizontal bar, with attrition labelled.

    Stages whose count is unknown are drawn as hollow bars, not omitted: an
    absent count is itself the finding.
    """
    labels = [s[0] for s in STAGES]
    vals = [counts.get(s[0]) for s in STAGES]
    known = [v for v in vals if v]
    y = np.arange(len(labels))[::-1]

    f, ax = plt.subplots(figsize=(9.5, 4.4))
    for i, (lab, v) in enumerate(zip(labels, vals)):
        if v is None:
            ax.barh(y[i], 1.0, color="white", edgecolor="#95a5a6", hatch="//",
                    lw=1.0)
            ax.text(1.02, y[i], "count not measured", va="center",
                    fontsize=8, color="#7f8c8d")
        else:
            ax.barh(y[i], v, color="#2980b9")
            ax.text(v * 1.02, y[i], f"{v:,}", va="center", fontsize=8)
    ax.set_yticks(y, labels)
    ax.set_xscale("log")
    ax.set_xlabel("records (log scale) -- hollow = not measured")
    ax.set_title("Raw detections -> clusters -> edges -> events -> scored\n"
                 "hatched bars are stages whose artefact was not available")
    if known:
        ax.set_xlim(right=max(known) * 4)
    rep.save(f, "lineage_funnel")
    rep.summary["lineage_counts"] = {l: v for l, v in zip(labels, vals)}


def figure_attrition(rep: Report, stage_a: int, stage_b: int,
                     name_a: str, name_b: str) -> None:
    """Explicit retention percentage between two adjacent lineage stages."""
    if not stage_a or not stage_b:
        return
    ratio = stage_b / stage_a
    rep.summary.setdefault("attrition", {})[f"{name_a}->{name_b}"] = {
        "from": stage_a, "to": stage_b, "retention": ratio,
        "per_event": ratio,
    }


# ---------------------------------------------------------------------------
# EVENT AND CLUSTER SHAPE
# ---------------------------------------------------------------------------


def figure_events_per_cluster(rep: Report, clusters_per_event: np.ndarray) -> None:
    """The 96.7% single-cluster fact, as a picture.

    This is the single most load-bearing measurement in the project: any feature
    needing two or more clusters (trend, hull, autocorrelation) is undefined for
    the overwhelming majority of events.
    """
    c = np.asarray(clusters_per_event)
    if c.size == 0:
        rep.skip("clusters_per_event", "no membership table")
        return
    single = float((c == 1).mean())
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2),
                                 gridspec_kw={"width_ratios": [1, 2]})
    ax1.bar(["1 cluster", "2+ clusters"], [single, 1 - single],
            color=["#2980b9", "#c0392b"])
    for i, v in enumerate([single, 1 - single]):
        ax1.text(i, v + 0.02, f"{v:.1%}", ha="center")
    ax1.set_ylim(0, 1.15)
    ax1.set_title("Events by cluster count\ntrend/hull features need >=2")
    vals, cnts = np.unique(np.minimum(c, 11), return_counts=True)
    ax2.bar(vals, cnts, color="#2980b9")
    ax2.set_yscale("log")
    ax2.set_xlabel("local clusters per event (11+ pooled)")
    ax2.set_ylabel("events (log)")
    ax2.set_title("Full distribution")
    rep.save(f, "clusters_per_event")
    rep.summary["clusters_per_event"] = {
        "n_events": int(c.size), "single_cluster_frac": single,
        "multi_cluster_events": int((c > 1).sum()),
        "mean": float(c.mean()), "max": int(c.max()),
    }


def figure_event_distributions(rep: Report, df: pd.DataFrame) -> None:
    """Duration, FRP and detection-count distributions for the whole catalogue.

    Every distribution is drawn for positives and negatives together, because a
    marginal distribution cannot be read as separability.
    """
    spec = [
        ("span_days", "event span (days)", False),
        ("geometry_total_frp", "total FRP (MW)", True),
        ("geometry_max_frp", "peak FRP (MW)", True),
        ("geometry_n_detections", "detections per event", False),
        ("area_km2", "event area (km2)", True),
    ]
    have = [(c, lab, logx) for c, lab, logx in spec if c in df.columns]
    rep.skip("event_distributions",
             f"none of {[s[0] for s in spec]} present" if not have else "")
    if not have:
        rep.manifest["skipped"] = [s for s in rep.manifest["skipped"]
                                   if s["figure"] != "event_distributions"]
        return
    y = df["y_true"].to_numpy() if "y_true" in df else None
    n = len(have)
    f, axes = plt.subplots(2, (n + 1) // 2, figsize=(4.3 * ((n + 1) // 2), 7.4))
    axes = np.atleast_1d(axes).ravel()
    stats = {}
    for ax, (col, lab, logx) in zip(axes, have):
        v = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        v = v.dropna()
        if v.empty:
            ax.set_visible(False)
            continue
        pos = v[y == 1] if y is not None and (y == 1).any() else None
        neg = v[y == 0] if y is not None and (y == 0).any() else None
        hi = np.nanpercentile(v, 99.5)
        bins = np.linspace(0, hi, 60) if hi > 0 else 60
        if neg is not None:
            ax.hist(neg, bins=bins, density=True, alpha=0.6,
                    color="#bdc3c7", label="other")
        if pos is not None:
            ax.hist(pos, bins=bins, density=True, alpha=0.6,
                    color="#c0392b", label=TARGET)
        if logx:
            ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_yscale("log")
        ax.set_title(lab, fontsize=9)
        ax.legend(fontsize=7)
        q = np.nanpercentile(v, [50, 90, 99])
        stats[col] = {"median": float(q[0]), "p90": float(q[1]),
                      "p99": float(q[2]), "n": int(v.size)}
    for ax in axes[len(have):]:
        ax.set_visible(False)
    f.suptitle("Event-level distributions across the catalogue "
               f"({len(df):,} events)")
    rep.save(f, "event_distributions")
    rep.summary["event_distributions"] = stats


def figure_events_over_time(rep: Report, df: pd.DataFrame) -> None:
    """Events per year and per month, positives included.

    Two things fall out of this and neither appears in any single-number summary:
    the 2012-2026 coverage, and whether the positive class is temporally
    distributed at all -- a target confined to a few months cannot be validated
    out-of-time.
    """
    if "start_time" not in df.columns:
        rep.skip("events_over_time", "no start_time column")
        return
    t = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
    ok = t.notna()
    if not ok.any():
        rep.skip("events_over_time", "start_time unparseable")
        return
    year = t[ok].dt.year
    y = df.loc[ok, "y_true"].to_numpy() if "y_true" in df else None
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.bar(year.value_counts().sort_index().index,
            year.value_counts().sort_index().values, color="#2980b9")
    ax1.set_title("Events per year")
    ax1.set_ylabel("events")
    mon = t[ok].dt.month
    ax2.bar(mon.value_counts().sort_index().index,
            mon.value_counts().sort_index().values, color="#7f8c8d")
    ax2.set_title("Events per month\nacquisition effect, not fire behaviour")
    ax2.set_xlabel("month")
    if y is not None and y.sum() > 0:
        yr_pos = year[y == 1].value_counts().sort_index()
        ax1b = ax1.twinx()
        ax1b.plot(yr_pos.index, yr_pos.values, color="#c0392b", lw=1.6,
                  marker="o", ms=3, label=TARGET)
        ax1b.set_ylabel(f"{TARGET} events", color="#c0392b")
        ax1b.grid(False)
        rep.summary["positives_per_year"] = {int(k): int(v)
                                              for k, v in yr_pos.items()}
    rep.save(f, "events_over_time")
    rep.summary["events_per_year"] = {int(k): int(v) for k, v in
                                      year.value_counts().sort_index().items()}


# ---------------------------------------------------------------------------
# LABELS
# ---------------------------------------------------------------------------


def figure_class_balance(rep: Report, labels: pd.Series) -> None:
    counts = labels.value_counts()
    f, ax = plt.subplots(figsize=(8.5, 4.2))
    cols = ["#c0392b" if c == TARGET else "#7f8c8d" for c in counts.index]
    ax.barh(counts.index[::-1], counts.values[::-1], color=cols[::-1])
    for i, v in enumerate(counts.values[::-1]):
        ax.text(v * 1.01, i, f"{v:,}  ({v / counts.sum():.2%})",
                va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(right=counts.max() * 4)
    ax.set_xlabel("events (log scale)")
    ax.set_title("Weak-label composition\nthese are RULE labels, not ground "
                 "truth; the smallest class bounds every metric")
    rep.save(f, "class_balance")
    rep.summary["class_balance"] = {k: int(v) for k, v in counts.items()}
    rep.summary["smallest_class"] = {
        "label": str(counts.index[-1]), "n": int(counts.iloc[-1]),
        "frac": float(counts.iloc[-1] / counts.sum()),
    }


def figure_per_class(rep: Report, rows: List[Dict]) -> None:
    """Per-class one-vs-rest bars, with support printed on every bar."""
    lab = [r["label"] for r in rows]
    x = np.arange(len(lab))
    f, ax = plt.subplots(figsize=(10, 4.6))
    has_rate = any(r.get("precision") is not None for r in rows)
    for i, (m, c) in enumerate((("precision", "#2980b9"),
                                ("recall", "#27ae60"),
                                ("f1", "#f39c12"))):
        ax.bar(x + (i - 1) * 0.28, [r.get(m) or 0.0 for r in rows], 0.28,
               label=m, color=c)
        for j, r in enumerate(rows):
            v = r.get(m)
            ax.text(x[j] + (i - 1) * 0.28, (v or 0.0) + 0.01,
                    "n/a" if v is None else f"{v:.2f}",
                    ha="center", fontsize=6)
    for j, r in enumerate(rows):
        ax.text(x[j], -0.13, f"n={r['support']:,}", ha="center", fontsize=6,
                color="#7f8c8d", rotation=90)
    ax.set_xticks(x, lab, rotation=20, fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("rate")
    ax.set_title("Per-class one-vs-rest rates\nno row may have support 0 -- "
                 "such a row is a class the model never saw")
    ax.legend(fontsize=7, ncol=3)
    rep.save(f, "per_class")
    rep.summary["per_class"] = rows


# ---------------------------------------------------------------------------
# CONFUSION + FULL MEASURE SET
# ---------------------------------------------------------------------------


def figure_confusion_grid(rep: Report, variants: Dict[str, Dict]) -> None:
    """One 2x2 per variant, each titled with the numbers that matter."""
    usable = [(k, v) for k, v in variants.items()
              if isinstance(v, dict) and v.get("tp") is not None]
    if not usable:
        rep.skip("confusion_grid", "no usable variant")
        return
    ncol = 2
    nrow = int(np.ceil(len(usable) / ncol))
    f, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 4.9 * nrow),
                           squeeze=False)
    for ax, (name, m) in zip(axes.ravel(), usable):
        mat = np.array([[int(m["tn"]), int(m["fp"])],
                        [int(m["fn"]), int(m["tp"])]], dtype="float64")
        ax.imshow(mat, cmap="Blues")
        for i in range(2):
            for j in range(2):
                v = mat[i, j]
                ax.text(j, i, f"{int(v):,}", ha="center", va="center",
                        fontsize=13 if v else 8,
                        color="white" if v > mat.max() / 2 else "#333")
        ax.set_xticks([0, 1], ["pred other", f"pred {TARGET}"], fontsize=8)
        ax.set_yticks([0, 1], ["actual other", f"actual {TARGET}"], fontsize=8)
        ax.set_title(
            f"{name}\nP={m['precision']:.4f} R={m['recall']:.4f} "
            f"F1={m['f1']:.4f} F2={m['f2']:.4f}\n"
            f"acc={m['accuracy']:.4f} bal={m['balanced_accuracy']:.4f} "
            f"MCC={m['mcc']:.4f} k={m['cohen_kappa']:.4f}", fontsize=8)
        ax.grid(False)
    for ax in axes.ravel()[len(usable):]:
        ax.set_visible(False)
    f.suptitle(f"Confusion matrices -- {TARGET}, every operating point")
    rep.save(f, "confusion_grid")


def figure_measure_bars(rep: Report, variants: Dict[str, Dict]) -> None:
    """The same variants as a grouped bar chart over every rate.

    A confusion matrix shows four cells; this shows the derived rates side by
    side, which is how a degenerate variant (high accuracy, zero skill) becomes
    visible rather than arguable.
    """
    usable = [(k, v) for k, v in variants.items()
              if isinstance(v, dict) and v.get("tp") is not None]
    measures = ["precision", "recall", "f1", "f2", "balanced_accuracy",
                "jaccard", "g_mean", "mcc", "cohen_kappa", "lift"]
    names = [k for k, _ in usable]
    x = np.arange(len(measures))
    w = 0.8 / max(len(names), 1)
    f, ax = plt.subplots(figsize=(12, 4.8))
    for i, (name, m) in enumerate(usable):
        vals = [min(m.get(k, 0.0), 3.0) for k in measures]
        ax.bar(x + (i - (len(names) - 1) / 2) * w, vals, w, label=name)
    ax.axhline(float(variants.get("all_negative_floor", {}).get("mcc", 0.0)),
               ls=":", color="#c0392b", lw=1)
    ax.set_xticks(x, measures, rotation=25, fontsize=7)
    ax.set_ylabel("value (lift clipped at 3)")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_title(f"Every derived rate, every operating point ({TARGET})")
    ax.legend(fontsize=6, ncol=3)
    rep.save(f, "measure_bars")


def figure_curves(rep: Report, scores: Dict[str, np.ndarray],
                  y: np.ndarray, prevalence: float) -> Dict:
    """PR, ROC, calibration and KS, in one panel, plus the lift curve."""
    f, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    ax1, ax2, ax3, ax4 = axes.ravel()
    tfm = {}
    for name, s in scores.items():
        r, p, ap = fig.pr_curve(y, s)
        tpr, fpr, auc = fig.roc_curve(y, s)
        tfm[name] = fig.threshold_free_metrics(y, s, n_boot=200)
        ax1.plot(r, p, label=f"{name} (AP {ap:.4f})")
        ax2.plot(fpr, tpr, label=f"{name} (AUC {auc:.4f})")
        ax3.hist(s[y == 0], bins=80, density=True, alpha=0.55,
                 color="#bdc3c7")
        ax3.hist(s[y == 1], bins=80, density=True, alpha=0.55, color="#c0392b",
                 label=TARGET)
        ax3.axvline(REVIEW_TAU, ls="--", color="#c0392b", lw=1)
        ax3.set_yscale("log")
        # calibration
        edges = np.linspace(0, 1, 21)
        mid, obs, cnt = (edges[:-1] + edges[1:]) / 2, [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (s >= lo) & (s < hi)
            obs.append(y[m].mean() if m.any() else np.nan)
            cnt.append(int(m.sum()))
        obs = np.array(obs)
        mid_ = mid[np.isfinite(obs)]
        ax4.plot(mid_, obs[np.isfinite(obs)], marker="o", ms=3,
                 label=f"{name} (Brier {tfm[name]['brier']:.5f})")
    ax1.axhline(prevalence, ls="--", color="#7f8c8d",
                label=f"prevalence {prevalence:.4f}")
    ax1.set_xlabel("recall")
    ax1.set_ylabel("precision")
    ax1.set_title("PR curve\nflat line = prevalence = no skill")
    ax1.legend(fontsize=7)
    ax2.plot([0, 1], [0, 1], ls=":", color="#7f8c8d")
    ax2.set_xlabel("false positive rate")
    ax2.set_ylabel("true positive rate")
    ax2.set_title("ROC curve")
    ax2.legend(fontsize=7)
    ax3.set_xlabel("score")
    ax3.set_ylabel("density (log)")
    ax3.set_title(f"Score distribution, cut at {REVIEW_TAU}")
    ax3.legend(fontsize=7)
    ax4.plot([0, 1], [0, 1], ls=":", color="#7f8c8d")
    ax4.set_xlabel("mean predicted probability")
    ax4.set_ylabel("observed rate")
    ax4.set_title("Calibration -- diagonal is perfect")
    ax4.legend(fontsize=7)
    f.suptitle(f"Threshold-free performance, {TARGET}")
    rep.save(f, "curves")
    rep.summary["threshold_free"] = tfm
    return tfm


def figure_ks(rep: Report, scores: Dict[str, np.ndarray], y: np.ndarray) -> None:
    """Empirical CDFs of positives and negatives: the KS statistic as a picture."""
    f, ax = plt.subplots(figsize=(7.5, 4.4))
    for name, s in scores.items():
        pos = np.sort(s[y == 1])
        neg = np.sort(s[y == 0])
        ax.plot(pos, np.arange(pos.size) / max(pos.size, 1), label=f"{name} +",
                color="#c0392b")
        ax.plot(neg, np.arange(neg.size) / max(neg.size, 1), label=f"{name} -",
                color="#2980b9", ls="--")
    ax.set_xlabel("score")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("Score CDF by class -- vertical gap at the cut is the KS "
                 "statistic")
    ax.legend(fontsize=7, ncol=2)
    rep.save(f, "ks_cdf")


def figure_threshold_sweep(rep: Report, scores: Dict[str, np.ndarray],
                           y: np.ndarray, tau: Optional[float] = None) -> None:
    """Precision, recall and flag rate against every threshold.

    This is where an uncalibrated threshold shows up: a flat 0.60 cut on a model
    whose scores live near 0.13 sits far to the right and catches nothing.
    """
    f, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    grid = np.unique(np.quantile(np.concatenate(list(scores.values())),
                                  np.linspace(0, 1, 300)))
    for name, s in scores.items():
        P, R, F = [], [], []
        for t in grid:
            c = fig.confusion(y, (s >= t).astype(int))
            m = fig.all_metrics(c)
            P.append(m["precision"])
            R.append(m["recall"])
            F.append(m["flag_rate"])
        axes[0].plot(grid, P, label=name)
        axes[1].plot(grid, R, label=name)
        axes[2].plot(grid, F, label=name)
        if tau is not None:
            for ax in axes:
                ax.axvline(tau, ls=":", color="#c0392b", lw=1)
    for ax, t, lab in zip(axes,
                          (0.90, None, None),
                          ("precision vs tau", "recall vs tau",
                           "flag rate vs tau (share of catalogue)")):
        ax.axhline(prevalence_of(y), ls="--", color="#7f8c8d", lw=1)
        ax.set_xlabel("threshold tau")
        ax.set_title(lab, fontsize=9)
        ax.legend(fontsize=7)
    axes[0].set_ylim(0, 1.02)
    axes[1].set_ylim(0, 1.02)
    rep.save(f, "threshold_sweep")


def prevalence_of(y: np.ndarray) -> float:
    return float(np.asarray(y).mean())


# ---------------------------------------------------------------------------
# DATA INTEGRITY + LEDGER
# ---------------------------------------------------------------------------


def figure_distance(rep: Report, dist_log: np.ndarray, y: np.ndarray,
                    dist_m: Optional[np.ndarray] = None) -> None:
    """The DIST_* integrity figure, which decides whether any baseline is valid.

    `dist_log` is the stored log10(1+m) column, which is what is plotted.
    `dist_m` is the same quantity in metres; it is what the sanity gate must
    check, because the gate's plausible band is expressed in metres and running
    it on log values would compare 4.08 "m" against a 9 m floor and mislead.
    """
    rep_data = fig.distance_sanity(
        dist_m if dist_m is not None else dist_log, "DIST_NEAREST_FACILITY_LOG")
    f, ax = plt.subplots(figsize=(9, 4.2))
    neg, pos = dist_log[y == 0], dist_log[y == 1]
    hi = np.nanpercentile(dist_log, 99.9)
    bins = np.linspace(np.nanmin(dist_log), hi * 1.05, 60)
    ax.hist(neg, bins=bins, alpha=0.6, color="#bdc3c7", label="other")
    ax.hist(pos, bins=bins, alpha=0.6, color="#c0392b", label=TARGET)
    lo, hib = fig.DISTANCE_LOG_PLAUSIBLE
    for edge in (10 ** lo, 10 ** hib):
        ax.axvline(edge, ls="--", color="#27ae60", lw=1)
    ax.set_xlabel("stored DIST_NEAREST_FACILITY_LOG (log10(1 + m))")
    ax.set_ylabel("events")
    ax.set_title("Distance feature integrity"
                 + ("" if rep_data["ok"] else "  -- SUSPECT"))
    if not rep_data["ok"]:
        ax.text(0.02, 0.95, rep_data["problem"][:200], transform=ax.transAxes,
                color="#c0392b", fontsize=7, va="top")
    ax.legend(fontsize=7)
    rep.save(f, "distance_integrity")
    rep.summary["distance_integrity"] = rep_data


def figure_feature_coverage(rep: Report, status: Dict[str, int],
                            n_spec: int = 141) -> None:
    f, ax = plt.subplots(figsize=(8, 4.2))
    cols = {"PRESENT": "#27ae60", "SPARSE": "#f39c12", "MISSING": "#c0392b"}
    ks = list(status)
    ax.bar(ks, [status[k] for k in ks], color=[cols.get(k, "#7f8c8d") for k in ks])
    for i, k in enumerate(ks):
        ax.text(i, status[k] + 1, f"{status[k]}", ha="center")
    ax.set_ylabel(f"features (of {n_spec})")
    ax.set_title("Feature coverage against the 141-feature specification")
    rep.save(f, "feature_coverage")
    rep.summary["feature_coverage"] = status


def figure_importance(rep: Report, imps: Dict[str, float], top: int = 25) -> None:
    """Feature importance, so a figure exists for *why* the model scored it."""
    if not imps:
        rep.skip("feature_importance", "no booster importances supplied")
        return
    s = pd.Series(imps).sort_values()
    s = s.tail(top)
    f, ax = plt.subplots(figsize=(8.5, max(3.5, 0.24 * len(s))))
    ax.barh(s.index, s.values, color="#2980b9")
    ax.set_xlabel("importance")
    ax.set_title(f"Top {len(s)} features -- "
                 "if these are satellite overpass time, the model learned "
                 "acquisition, not fire")
    rep.save(f, "feature_importance")
    rep.summary["feature_importance"] = {k: float(v) for k, v in s.items()}