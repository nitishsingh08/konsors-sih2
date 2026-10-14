# ============================================================================
# COLAB CELL: figures -- events, clusters, confusion matrices, training ledger
#
# Reads the held-out predictions and the dataset tables, and writes PNGs plus a
# JSON summary to Drive. Run after colab_p7_train.py.
#
#   python3 scripts/fetch_models.py     # if you want the SHAP figures too
#   then, in Colab:  this cell
#
# Every figure that uses a DIST_* feature carries a data-integrity banner if the
# stored distance is implausible. On the first real run it fired, which is why
# the banner exists.
# ============================================================================

P8_CELL_VERSION = "2026-10-04.1"

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import colab_utils as cu
import feature_gap as fgap
import figures as fig
import p8_report as rep_mod

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False,
})
FIG_DIR = None
SUMMARY = {}


def _save(fig_obj, name):
    path = FIG_DIR / name
    fig_obj.savefig(path, bbox_inches="tight")
    plt.close(fig_obj)
    print(f"  wrote {path.name}")
    return path


print("=" * 78)
print(f"P8 -- FIGURES  [v{P8_CELL_VERSION}]")
print("=" * 78)

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError("no project root")
CANON = PROJECT_ROOT / "full_2012_2026"
DATASET = CANON / "features" / "dataset"
QC_DIR = PROJECT_ROOT / "qc"
FIG_DIR = QC_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. DATA INTEGRITY -- before anything is plotted
# ---------------------------------------------------------------------------

print()
print("1. DATA INTEGRITY")
print("-" * 78)

held = {}
for name in ("B", "A"):
    path = DATASET / f"heldout_pred_{name}.csv"
    if not cu.path_ok(path):
        print(f"  MISSING {path.name} -- re-run colab_p7_train.py")
        continue
    held[name] = pd.read_csv(path)
    print(f"  loaded {path.name}: {len(held[name]):,} rows")

if not held:
    raise SystemExit("no held-out predictions to plot")

integrity = {}
for name, frame in held.items():
    report = fig.distance_sanity(frame["dist_m"].to_numpy(),
                                f"model {name} dist_m")
    integrity[name] = report
    status = "OK" if report["ok"] else "SUSPECT"
    print(f"  model {name}: {status}   "
           f"median {report['median_m']:.2f} m, "
           f"max {report['max_m']:.2f} m, "
           f"{report['frac_implausibly_close']:.1%} under 1 m")
    if not report["ok"]:
        print()
        print("  " + "\n  ".join(report["problem"].split(". ")))
SUMMARY["data_integrity"] = integrity

degraded = any(not r["ok"] for r in integrity.values())
BANNER = (
    "DISTANCE FEATURES IMPLAUSIBLE -- not encoding metres"
    if degraded else ""
)

# ---------------------------------------------------------------------------
# 2. CLASS BALANCE AND SPLIT
# ---------------------------------------------------------------------------

print()
print("2. CLASS BALANCE")
print("-" * 78)

b = held.get("B")
if b is not None:
    counts = b["y_true_weak_label"].value_counts()
    fig_obj, ax = plt.subplots(figsize=(8, 4.2))
    colors = ["#c0392b" if c == "INDUSTRIAL_FIRE" else "#7f8c8d"
              for c in counts.index]
    ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1])
    for i, v in enumerate(counts.values[::-1]):
        ax.text(v * 1.01, i, f"{v:,}  ({v / len(b):.2%})",
                va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("events (log scale)")
    ax.set_title("Weak-label composition of the held-out sample\n"
                 "target class in red -- these are RULE labels, not ground truth")
    ax.set_xlim(right=counts.max() * 3)
    _save(fig_obj, "01_class_balance.png")
    SUMMARY["class_balance"] = {k: int(v) for k, v in counts.items()}

# ---------------------------------------------------------------------------
# 3. CONFUSION MATRICES -- four of them, because one hides everything
# ---------------------------------------------------------------------------

if b is not None:
    print()
    print("3. CONFUSION MATRICES")
    print("-" * 78)

    y = b["y_true"].to_numpy()
    variants = [
        ("all-negative (floor)", fig.all_negative_confusion(y), "#95a5a6"),
        ("model B @ best F1", fig.best_f1_confusion(y, b["score"].to_numpy()),
         "#2980b9"),
        ("model B @ review 0.60",
         fig.confusion(y, (b["score"].to_numpy() >= 0.60).astype(int)),
         "#27ae60"),
        ("model A @ review 0.60",
         fig.confusion(y, (held["A"]["score"].to_numpy() >= 0.60).astype(int))
         if "A" in held else None, "#8e44ad"),
    ]
    # Full measure set for the console and for summary.json; the grid itself is
    # drawn by p8_report so the four-cell layout exists in exactly one place.
    full = fig.full_confusion_report(y, b["score"].to_numpy())
    print(fig.render_report(f"MODEL B -- every measure (n={len(b):,}, "
                            f"positives={int(y.sum()):,})", full,
                            fig.threshold_free_metrics(y, b["score"].to_numpy())))
    SUMMARY["confusion_full"] = full

    usable = [(n, c, col) for n, c, col in variants if c is not None]

    fig_obj, axes = plt.subplots(1, len(usable), figsize=(4.1 * len(usable), 4.3))
    axes = np.atleast_1d(axes)
    for ax, (name, c, colour) in zip(axes, usable):
        mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(mat, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]:,}", ha="center", va="center",
                        fontsize=12 if mat[i, j] > 0 else 8,
                        color="white" if mat[i, j] > mat.max() / 2 else "#333")
        ax.set_xticks([0, 1], ["pred other", "pred target"], fontsize=8)
        ax.set_yticks([0, 1], ["actual other", "actual target"], fontsize=8)
        r = fig.rates(c)
        ax.set_title(
            f"{name}\nP={r['precision']:.3f}  R={r['recall']:.3f}  "
            f"F1={r['f1']:.3f}", fontsize=9, color=colour)
        ax.grid(False)
    if BANNER:
        fig_obj.suptitle(BANNER, color="#c0392b", fontsize=10)
    fig_obj.suptitle(
        f"Confusion on {len(b):,} held-out events "
        f"({int(y.sum()):,} positive, {y.mean():.1%})")
    _save(fig_obj, "02_confusion_matrices.png")

    SUMMARY["confusion"] = {
        name: {**c, **fig.rates(c)} for name, c, _ in usable
    }

    # The floor, stated numerically. 83.6% accuracy means nothing.
    floor = fig.all_negative_confusion(y)
    fr = fig.rates(floor)
    print(f"  floor (all-negative): accuracy {fr['accuracy']:.4f}  "
          f"recall {fr['recall']:.4f}   <- not a model")

# ---------------------------------------------------------------------------
# 4. PR CURVE and SCORE DISTRIBUTION
# ---------------------------------------------------------------------------

if b is not None:
    print()
    print("4. PR CURVES AND SCORE DISTRIBUTION")
    print("-" * 78)

    fig_obj, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, colour in (("B", "#2980b9"), ("A", "#8e44ad")):
        if name not in held:
            continue
        r, p, ap = fig.pr_curve(b["y_true"].to_numpy(), held[name]["score"].to_numpy())
        ax1.plot(r, p, label=f"model {name} (AP {ap:.3f})", color=colour)
    prevalence = float(b["y_true"].mean())
    ax1.axhline(prevalence, ls="--", color="#7f8c8d",
                label=f"prevalence {prevalence:.3f}")
    ax1.set_xlabel("recall")
    ax1.set_ylabel("precision")
    ax1.set_title("Precision-recall\nflat line = prevalence = no skill")
    ax1.legend(fontsize=8)

    for name, colour in (("B", "#2980b9"), ("A", "#8e44ad")):
        if name not in held:
            continue
        pos = held[name].loc[held[name]["y_true"] == 1, "score"]
        neg = held[name].loc[held[name]["y_true"] == 0, "score"]
        ax2.hist(neg, bins=80, alpha=0.6, density=True, color="#bdc3c7",
                  label=f"other (model {name})")
        ax2.hist(pos, bins=80, alpha=0.6, density=True, color=colour,
                  label=f"target (model {name})")
    ax2.axvline(0.60, ls="--", color="#c0392b", label="review threshold 0.60")
    ax2.set_yscale("log")
    ax2.set_xlabel("predicted probability")
    ax2.set_title("Score distribution\nseparation, or its absence, is visible here")
    ax2.legend(fontsize=7)
    _save(fig_obj, "03_pr_and_scores.png")

    for name in ("B", "A"):
        if name in held:
            _, _, ap = fig.pr_curve(b["y_true"].to_numpy(),
                                    held[name]["score"].to_numpy())
            _, _, auc = fig.roc_curve(b["y_true"].to_numpy(),
                                       held[name]["score"].to_numpy())
            SUMMARY.setdefault("curves", {})[name] = {
                "pr_auc": round(ap, 4), "roc_auc": round(auc, 4)}

# ---------------------------------------------------------------------------
# 5. THE DISTANCE FEATURES -- shown raw, with the plausibility band
# ---------------------------------------------------------------------------

if b is not None:
    print()
    print("5. DISTANCE TO NEAREST FACILITY")
    print("-" * 78)

    fig_obj, ax = plt.subplots(figsize=(9, 4.2))
    pos = b.loc[b["y_true"] == 1, "dist_m"]
    neg = b.loc[b["y_true"] == 0, "dist_m"]
    bins = np.logspace(np.log10(max(min(b["dist_m"].min(), 1e-3)),
                       np.log10(b["dist_m"].max()) * 1.05, 60)
    ax.hist(neg, bins=bins, alpha=0.6, color="#bdc3c7", label="other")
    ax.hist(pos, bins=bins, alpha=0.6, color="#c0392b", label="target")
    lo, hi = fig.DISTANCE_LOG_PLAUSIBLE
    for edge in (10 ** lo, 10 ** hi):
        ax.axvline(edge, ls="--", color="#27ae60", lw=1)
    ax.axvspan(1e-3, 10 ** lo, color="#c0392b", alpha=0.08)
    ax.set_xscale("log")
    ax.set_xlabel("stored DIST_NEAREST_FACILITY_LOG, inverted to metres")
    ax.set_title("Distance to nearest facility"
                 + ("\nDISTANCE FEATURES IMPLAUSIBLE" if degraded else ""))
    if degraded:
        ax.text(0.02, 0.95,
                f"{integrity['B']['frac_implausibly_close']:.0%} of events "
                "under 1 m -- these are not metres",
                transform=ax.transAxes, color="#c0392b", fontsize=9, va="top")
    ax.legend(fontsize=8)
    _save(fig_obj, "04_distance_distribution.png")

# ---------------------------------------------------------------------------
# 6. CLUSTER STRUCTURE
# ---------------------------------------------------------------------------

print()
print("6. CLUSTER STRUCTURE")
print("-" * 78)

members = CANON / "events" / "event_local_cluster_membership.parquet"
if cu.path_ok(members):
    import pyarrow.parquet as pq

    n_clusters = pq.ParquetFile(members).metadata.num_rows
    n_events = 7_265_849
    ledger = fig.training_ledger(
        total_events=n_events, membership_rows=n_clusters,
        train_available=5_772_935, test_available=888_156, dropped=604_758,
        train_sampled=1_500_000, test_sampled=600_000,
        train_positives=228_063, test_positives=98_352,
        n_features_model_a=53, n_features_model_b=29,
    )
    print(fig.render_ledger(ledger))
    SUMMARY["training_ledger"] = ledger
    (QC_DIR / "training_ledger.json").write_text(json.dumps(ledger, indent=2))
    print(f"\n  wrote training_ledger.json")
else:
    print(f"  membership table not found at {members}")

# ---------------------------------------------------------------------------
# 7. FEATURE AVAILABILITY
# ---------------------------------------------------------------------------

print()
print("7. FEATURE AVAILABILITY")
print("-" * 78)

if b is not None:
    X = pq.ParquetFile(DATASET / "X.parquet")
    cols = list(X.schema_arrow.names)
    sparsity = figmod = None
    try:
        import pyarrow.parquet as _pq
        header = _pq.ParquetFile(DATASET / "X.parquet")
        rows = header.metadata.num_rows
    except Exception:
        rows = n_events

    avail = figmod = None
    from feature_gap import gap_report
    import pandas as _pd
    rep = gap_report(_pd.DataFrame(columns=cols))
    fig_obj, ax = plt.subplots(figsize=(10, 5.2))
    counts = rep["status"].value_counts()
    colours = {"PRESENT": "#27ae60", "SPARSE": "#f39c12", "MISSING": "#c0392b"}
    ax.bar(counts.index, counts.values,
           color=[colours.get(i, "#7f8c8d") for i in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, f"{v}", ha="center")
    ax.set_ylim(0, max(counts.values) * 1.15)
    ax.set_ylabel("features (of 141)")
    ax.set_title("Feature coverage against the 141-feature specification")
    _save(fig_obj, "05_feature_availability.png")
    SUMMARY["feature_status"] = {k: int(v) for k, v in counts.items()}

# ---------------------------------------------------------------------------
# 8. WRITE SUMMARY
# ---------------------------------------------------------------------------

(FIG_DIR / "summary.json").write_text(json.dumps(SUMMARY, indent=2, default=str))
print()
print(f"figures  -> {FIG_DIR}")
print(f"summary  -> {FIG_DIR / 'summary.json'}")
print()
if degraded:
    print("=" * 78)
    print("DATA INTEGRITY PROBLEM -- read before quoting any result")
    print("=" * 78)
    for name, report in integrity.items():
        print(f"  model {name}: {report['problem']}")
    print()
    print("  Consequence: the distance-threshold baseline predicts EVERY event")
    print("  positive, so its recall of 1.0 is structural, and its precision is")
    print("  just the class prevalence. Every comparison against it is void until")
    print("  the DIST_* features are recomputed from real geometry.")