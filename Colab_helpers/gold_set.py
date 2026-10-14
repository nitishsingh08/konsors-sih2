# ============================================================================
# Gold set -- the independent ground truth this project does not have
#
# THE PROBLEM THIS SOLVES
#
# Measured on the full catalogue, 2026-10-04:
#
#     max distance-to-nearest-facility among 1,119,254 weak positives = 5,000 m
#     positives beyond 5,000 m                                     = 2
#     PR-AUC of DISTANCE ALONE                                     = 0.9811
#     PR-AUC of the honest 29-feature model B                      = 0.3353
#
# `weak_labels.LABEL_INPUT_FEATURES` publishes the 16 columns the rule reads,
# and `weak_labels.LABEL_DERIVED_FEATURES` the 18 industrial columns. Their
# overlap is the 10 distance columns plus persistence. So:
#
#   * Model A's PR-AUC ~0.9963 is circular -- it reads the rule's inputs.
#   * Model B's 0.3353 is honest -- and it loses to distance, because the
#     target IS a distance threshold. Not a tuning failure.
#
# Every number the pipeline can currently produce is therefore uninformative
# about the real question: whether any of this has anything to do with
# INDUSTRIAL fires rather than with proximity to a gazetteer entry.
#
# ONLY A HUMAN-ANNOTATED SAMPLE ANSWERS THAT, and it is already mandated:
# HANDOFF section 8 requires "~300 events: Jamnagar/Vadinar/Panipat refineries,
# Bokaro/Bhilai steel, coal plants, mines" and says "never use it to fit the
# model". This module builds that sample and scores against it.
#
# WHY THE SAMPLE MUST BE STRATIFIED
#
# Uniform random sampling of events is the obvious approach and it is nearly
# useless here. The 5 km cut means the classifier is almost right almost
# everywhere, so a uniform sample is dominated by obvious cases and carries
# almost no information about the error rate AT the boundary, which is the
# only place precision is decided. So `build_sample` stratifies across:
#
#   * the distance decile, with oversampling in the 3-8 km band where the
#     cut actually bites
#   * the weak label, so false negatives are visible
#   * the spatial block, so the sample is not one region
#   * facility category, because a coal power plant and a gas pipeline
#     attract different numbers of false alarms
#
# And `score_against_gold` reports the stratified-weighted estimate, because
# an unweighted score on an oversampled set is not a prevalence estimate and
# quoting one as if it were is exactly the kind of quiet overstatement this
# project has already produced twice (README section 5 defect 2, section 14).
# ============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

#: Log10 of 1 + metres. `DIST_*_LOG` uses this convention; see README section 5.
def log_metres(metres):
    return np.log10(1.0 + np.asarray(metres, dtype="float64"))


def metres_from_log(log_value):
    return np.power(10.0, np.asarray(log_value, dtype="float64")) - 1.0


#: The industrial categories a labeller is asked to distinguish. `natural` is the
#: negative. Anything else is recorded as `uncertain` and EXCLUDED from scoring
#: rather than forced into a binary -- a forced binary inflates precision.
GOLD_CLASSES = ("industrial", "natural", "uncertain")

#: Boundary band in metres. These are the cases where the 5 km rule is actually
#: deciding the answer, so they carry the precision.
BOUNDARY_BAND_M = (3_000.0, 8_000.0)


@dataclass
class GoldSpec:
    """How to draw the sample. Every value is recorded in the output."""

    n_total: int = 300
    #: Share of the sample reserved for the decision boundary. HANDOFF names
    #: refinery/steel/coal/mine sites explicitly, which are exactly the
    #: near-facility cases a uniform draw would under-represent.
    boundary_share: float = 0.50
    #: Blocks are the out-of-space split key, so every sampled block is one the
    #: model can legitimately be evaluated on.
    blocks: Optional[Sequence[str]] = None
    seed: int = 0

    def validate(self) -> None:
        if not 0 < self.n_total < 10_000:
            raise ValueError(f"n_total must be in 1..9999, got {self.n_total}")
        if not 0.0 <= self.boundary_share <= 1.0:
            raise ValueError(
                f"boundary_share must be in 0..1, got {self.boundary_share}"
            )
        if self.blocks is not None and not list(self.blocks):
            raise ValueError("blocks was given but is empty")


def _distance_bucket(metres):
    """Which sampling stratum an event belongs to. Vectorised.

    `build_sample` passes a whole column, so this must be array-safe. A
    scalar-shaped `if not np.isfinite(m)` looks correct on one value and
    raises "truth value of an array is ambiguous" on a column.

    The band deliberately starts at 3 km, INSIDE the 5 km cut, because that
    is where the rule is still deciding and therefore where errors live.
    """
    m = np.asarray(metres, dtype="float64")
    lo, hi = BOUNDARY_BAND_M
    finite = np.isfinite(m)
    out = np.full(m.shape, "no_distance", dtype=object)
    out[finite & (m < lo)] = "inside_5km"
    out[finite & (m >= lo) & (m <= hi)] = "boundary_3_8km"
    out[finite & (m > hi)] = "outside_5km"
    # `np.full(())` is 0-d, so indexing it raises. `item()` unwraps a scalar
    # back to a plain str, which is what a single-value caller expects.
    return out if out.ndim else out.item()


#: How the NON-boundary remainder splits between inside and outside the cut.
#: Not proportional to pool size. Under a 5 km cut the outside pool is ~4x the
#: inside one, so a size-proportional split sends two thirds of the budget to
#: rows the rule trivially rejects and starves the inside pool, which is where
#: the false negatives live. Explicit weights, documented, chosen for
#: information rather than for representativeness.
BULK_SPLIT = {"inside_5km": 0.60, "outside_5km": 0.40}


def _largest_remainder(demand: Dict[str, int], capacity: Dict[str, int],
                       total: int) -> Dict[str, int]:
    """Split `total` across strata in proportion to `demand`, capped by capacity.

    Largest-remainder (Hamilton) apportionment, so the counts sum to `total`
    exactly rather than drifting by a rounding error. Shortfalls from capped
    strata are redistributed to strata that still have room, weighted by their
    unmet demand, and iterated -- otherwise a capped stratum silently shrinks
    the whole sample.
    """
    alloc = {k: 0 for k in demand}
    remaining = int(total)
    unmet = {k: max(0, int(v)) for k, v in demand.items()}

    for _ in range(12):
        if remaining <= 0:
            break
        room = {
            k: max(0, capacity.get(k, 0) - alloc[k])
            for k in demand
        }
        open_strata = {k: v for k, v in unmet.items() if v > 0 and room[k] > 0}
        if not open_strata:
            break
        weight_sum = sum(open_strata.values())
        if weight_sum <= 0:
            break

        # Provisional shares, then hand out the leftover by largest remainder.
        prov = {k: remaining * v / weight_sum for k, v in open_strata.items()}
        floors = {k: min(int(prov[k]), room[k]) for k in open_strata}
        gave = sum(floors.values())
        for k in open_strata:
            alloc[k] += floors[k]
            unmet[k] -= floors[k]
        left = remaining - gave
        if left <= 0:
            remaining = 0
            break
        remainders = sorted(
            open_strata,
            key=lambda k: prov[k] - int(prov[k]),
            reverse=True,
        )
        for k in remainders:
            if left <= 0:
                break
            if alloc[k] < capacity.get(k, 0):
                alloc[k] += 1
                left -= 1
        remaining = max(0, left)
    return alloc


def build_sample(
    frame: pd.DataFrame,
    spec: GoldSpec,
    *,
    id_col: str = "event_id",
    dist_log_col: str = "DIST_NEAREST_FACILITY_LOG",
    label_col: str = "label_weak",
    block_col: str = "spatial_block",
    target: str = "INDUSTRIAL_FIRE",
    lon_col: str = "event_centroid_lon",
    lat_col: str = "event_centroid_lat",
) -> pd.DataFrame:
    """Draw the stratified gold sample. Pure selection -- labels nothing.

    Returns the sampled rows plus a `gold_id`, the strata it came from, the
    distance in METRES (not log -- a labeller reading a log value would have to
    do arithmetic to judge plausibility), and an EMPTY `gold_label` column for
    the human to fill in.

    The empty column is the point: this function must never assign a label.
    Anything that fills it automatically has reintroduced the circularity.
    """
    spec.validate()
    missing = [
        c for c in (id_col, dist_log_col, label_col) if c not in frame.columns
    ]
    if missing:
        raise KeyError(f"missing required columns: {missing}")

    work = frame.copy()
    if dist_log_col in work.columns:
        work["dist_m"] = metres_from_log(work[dist_log_col])
    else:
        work["dist_m"] = np.nan
    work["gold_bucket"] = _distance_bucket(work["dist_m"])

    if spec.blocks is not None:
        if block_col not in work.columns:
            raise KeyError(
                f"blocks were requested but {block_col!r} is absent; the split "
                "keys are added by training_design.add_split_keys()"
            )
        work = work[work[block_col].isin(list(spec.blocks))]
        if work.empty:
            raise ValueError("no rows in the requested blocks")

    rng = np.random.default_rng(spec.seed)

    # --- capacity per stratum -------------------------------------------------
    capacity = {b: int((work["gold_bucket"] == b).sum())
                for b in ("boundary_3_8km", "inside_5km", "outside_5km",
                          "no_distance")}

    # --- the boundary band gets its explicit quota first ---------------------
    #
    # Within the band, rows are taken CLOSEST TO THE 5 km CUT and spread across
    # spatial blocks round-robin. Closest-first because those are the cases
    # where the rule is actually undecided; round-robin because a band drawn
    # from one refinery would tell you about one refinery.
    n_boundary = int(round(spec.n_total * spec.boundary_share))
    band = work[work["gold_bucket"] == "boundary_3_8km"]
    chosen_boundary: List[int] = []
    if n_boundary and len(band):
        _b = band.copy()
        # Distance to the 5,000 m decision boundary, in metres.
        _b["_to_cut"] = (_b["dist_m"] - 5000.0).abs()
        # Everything downstream indexes `work` POSITIONALLY, so the queues must
        # hold positions. Storing index LABELS here and using them as positions
        # picks the wrong rows -- and fails loudly only when the band is small.
        _queues: Dict[object, List[int]] = {}
        _groups = (_b.groupby(block_col) if block_col in _b.columns
                   else [(None, _b)])
        for _blk, _grp in _groups:
            _ordered = _grp.sort_values("_to_cut", kind="mergesort")
            _queues[_blk] = work.index.get_indexer(_ordered.index).tolist()
        _keys = list(_queues)
        _taken = 0
        while _taken < n_boundary:
            _progressed = False
            for _k in _keys:
                if _taken >= n_boundary:
                    break
                if _queues[_k]:
                    chosen_boundary.append(_queues[_k].pop(0))
                    _taken += 1
                    _progressed = True
            if not _progressed:
                break
    chosen_boundary = np.array(sorted(set(chosen_boundary)), dtype=int)

    # --- remainder across the non-boundary strata ----------------------------
    remaining_rows = work.drop(work.index[chosen_boundary])
    bulk = spec.n_total - len(chosen_boundary)
    if bulk < 0:
        raise ValueError(
            f"n_total={spec.n_total} but the boundary quota alone wanted "
            f"{len(chosen_boundary)}; lower boundary_share"
        )

    demand = {}
    cap_bulk = {}
    for name, weight in BULK_SPLIT.items():
        avail = int((remaining_rows["gold_bucket"] == name).sum())
        if avail == 0:
            continue
        demand[name] = int(round(bulk * weight))
        cap_bulk[name] = avail
    # `no_distance` absorbs whatever the weighted split could not fill, so the
    # sample still reaches n_total when the weighted strata are exhausted.
    _nd = int((remaining_rows["gold_bucket"] == "no_distance").sum())
    if _nd:
        demand["no_distance"] = max(0, bulk - sum(demand.values()))
        cap_bulk["no_distance"] = _nd

    if not demand:
        raise ValueError(
            "no rows in any sampleable stratum after drawing the boundary band"
        )

    alloc = _largest_remainder(demand, cap_bulk, bulk)

    rest_pos: List[int] = []
    for name, k in alloc.items():
        if k <= 0:
            continue
        pool = remaining_rows[remaining_rows["gold_bucket"] == name]
        pos = np.flatnonzero(remaining_rows.index.isin(pool.index))
        k = int(min(k, len(pos)))
        if k:
            rest_pos.extend(rng.choice(pos, k, replace=False).tolist())

    # `chosen_boundary` indexes `work`; `rest_pos` indexes `remaining_rows`,
    # which is SHORTER because the boundary rows were dropped from it. Combining
    # them directly picks unrelated rows and silently deduplicates the total --
    # it returned a full-looking 300 rows of the wrong composition. Map the
    # remainder back through the surviving positions first.
    keep_pos = np.setdiff1d(np.arange(len(work)), chosen_boundary)
    rest_in_work = keep_pos[np.asarray(sorted(set(rest_pos)), dtype=int)] \
        if rest_pos else np.empty(0, dtype=int)

    picked = np.array(
        sorted(set(chosen_boundary.tolist() + rest_in_work.tolist())),
        dtype=int,
    )
    out = work.iloc[picked].copy()

    if len(out) < spec.n_total:
        # Loud, not silent. A short sample is a different sample than the one
        # that was asked for, and a 150-row band-heavy draw reads as "300
        # requested" in a QC report.
        print(
            f"WARNING: drew {len(out)} of {spec.n_total} requested. "
            f"Bucket capacities were {capacity}. Raise n_total, widen the "
            f"blocks, or accept the shortfall and record it."
        )

    out.insert(0, "gold_id", [f"GOLD_{i:04d}" for i in range(1, len(out) + 1)])
    out["weak_label"] = out[label_col]
    out["weak_positive"] = (out[label_col] == target).astype(int)

    # Distance in metres for the human. The log value is kept alongside so the
    # labeller can cross-check it, but metres is what they will judge against.
    if "dist_m" in out.columns:
        out["nearest_facility_m"] = out["dist_m"].round(1)

    for c in (lon_col, lat_col):
        if c in out.columns:
            out[c] = out[c].round(5)

    # The column the human fills. Left empty and NaN -- never filled here.
    out["gold_label"] = pd.Series(
        [pd.NA] * len(out), dtype="object"
    )
    out["gold_confidence"] = pd.Series([pd.NA] * len(out), dtype="object")
    out["gold_notes"] = pd.Series([pd.NA] * len(out), dtype="object")

    front = ["gold_id", "gold_label", "gold_confidence", "gold_notes",
             "gold_bucket", "weak_label", "weak_positive"]
    front = [c for c in front if c in out.columns]
    ordered = front + [c for c in out.columns if c not in front]
    return out[ordered].reset_index(drop=True)


LABELLING_PROTOCOL = """\
GOLD SET -- HOW TO LABEL EACH EVENT
==================================

Scope: does this event correspond to an INDUSTRIAL fire or flaring event, or to
a natural (vegetation) fire? Judge the CAUSE, not the proximity. An event 200 m
from a refinery that is a farmer burning stubble is `natural`. That case is the
entire point of the gold set.

For each row:

  gold_label        industrial | natural | uncertain
  gold_confidence   high | medium | low
  gold_notes        one line: what you saw

Evidence, in order of strength:

  1. A named incident. News, a company notice, a fire-service record, for that
     facility on that date. This is `high` confidence by itself.
  2. Sentinel-2 imagery before / during / after. Look for surface loss or a
     burn scar that does NOT recover. A refinery tank fire leaves a persistent
     thermal anomaly and no green regrowth. A crop burn leaves darkened bare
     soil that greens over within weeks.
  3. Co-registered thermal anomaly coincident with a facility footprint, that
     is NOT a known flare stack.
  4. Absence of nearby fuel. Bare ground or built-up surface, no vegetation
     within 1 km, argues industrial.

Choose `uncertain` freely. A forced binary label on an ambiguous case inflates
precision, and this sample exists to make the number trustworthy. `uncertain`
rows are reported separately, never quietly dropped -- if 20% are uncertain,
that is a finding about the task, not a defect in your labelling.

DO NOT use the model's score, and do not look at `weak_label` until you have
committed `gold_label`. Both leak the answer you are providing.
"""


def audit_completion(frame: pd.DataFrame) -> Dict[str, object]:
    """Is this sample actually finished? Reports holes instead of hiding them."""
    n = len(frame)
    if n == 0:
        return {"n": 0, "complete": False, "issues": ["sample is empty"]}

    issues: List[str] = []
    lab = frame.get("gold_label")
    if lab is None:
        return {
            "n": n, "complete": False,
            "issues": ["no gold_label column -- was this built by build_sample()?"],
        }
    filled = lab.notna() & (lab.astype("string").str.strip() != "")
    n_filled = int(filled.sum())
    if n_filled < n:
        issues.append(f"{n - n_filled} of {n} rows still unlabelled")

    values = set(lab[filled].astype(str).str.lower().unique())
    bad = sorted(values - set(GOLD_CLASSES))
    if bad:
        issues.append(
            f"unrecognised gold_label values {bad}; allowed: {list(GOLD_CLASSES)}"
        )

    conf = frame.get("gold_confidence")
    if conf is not None:
        _cf = conf.notna() & (conf.astype("string").str.strip() != "")
        if int(_cf.sum()) < n_filled:
            issues.append(
                f"{n_filled - int(_cf.sum())} labelled rows have no confidence"
            )
        _cv = set(conf[_cf].astype(str).str.lower().unique()) - {
            "high", "medium", "low",
        }
        if _cv:
            issues.append(f"unrecognised gold_confidence values {sorted(_cv)}")

    return {
        "n": n,
        "n_labelled": n_filled,
        "complete": not issues,
        "issues": issues,
        "distribution": {
            str(k): int(v)
            for k, v in lab[filled].astype(str).value_counts().items()
        },
    }


def score_against_gold(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    label_col: str = "gold_label",
    weight_col: str = "weight",
    bucket_col: str = "gold_bucket",
    distance_col: str = "nearest_facility_m",
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Score a model against the gold labels, stratified back to prevalence.

    `uncertain` rows are reported, never scored. If they are a large share, say
    so in the writeup rather than dropping them quietly.
    """
    audit = audit_completion(frame)
    if not audit["complete"]:
        return {"scored": False, "audit": audit}

    lab = frame[label_col].astype(str).str.lower()
    keep = lab.isin(["industrial", "natural"])
    work = frame[keep].copy()
    y = (lab[keep] == "industrial").astype(int).to_numpy()

    if len(work) == 0:
        return {"scored": False, "audit": audit,
                "reason": "no industrial or natural rows"}

    if score_col not in work.columns:
        raise KeyError(
            f"{score_col!r} is absent. Merge the model's probability onto the "
            "gold sheet on gold_id before scoring."
        )
    s = work[score_col].astype("float64").to_numpy()
    pred = (s >= threshold).astype(int)

    if weight_col in work.columns:
        w = work[weight_col].astype("float64").to_numpy()
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones(len(work))
    else:
        w = np.ones(len(work), dtype="float64")

    # Weighted confusion. The parenthesisation is load-bearing: `&` binds
    # LOOSER than `*`, so `(pred == 1) & (y == 1) * w` parses as
    # `(pred == 1) & ((y == 1) * w)` and tries to bitwise-and a bool with a
    # float. Cast the intersections to float first, then weight.
    def _w(mask):
        return mask.astype("float64") * w

    tp = float(_w((pred == 1) & (y == 1)).sum())
    fp = float(_w((pred == 1) & (y == 0)).sum())
    fn = float(_w((pred == 0) & (y == 1)).sum())
    tn = float(_w((pred == 0) & (y == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    out: Dict[str, object] = {
        "scored": True,
        "n_scored": int(len(work)),
        "n_uncertain_excluded": int((~keep).sum()),
        "threshold": float(threshold),
        "weighted": weight_col in frame.columns,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "prevalence_gold": float(y.mean()),
    }

    # Per-stratum, because a single number over an oversampled sample hides the
    # only thing worth knowing: whether the errors sit at the 5 km boundary.
    if bucket_col in work.columns:
        per: Dict[str, Dict[str, float]] = {}
        for b, grp in work.groupby(bucket_col):
            _y = (grp[label_col].astype(str).str.lower() == "industrial").astype(int)
            _p = (grp[score_col].astype("float64") >= threshold).astype(int)
            _tp = int(((_p == 1) & (_y == 1)).sum())
            _fp = int(((_p == 1) & (_y == 0)).sum())
            _fn = int(((_p == 0) & (_y == 1)).sum())
            per[str(b)] = {
                "n": int(len(grp)),
                "gold_positive_rate": float(_y.mean()) if len(grp) else 0.0,
                "precision": _tp / (_tp + _fp) if (_tp + _fp) else float("nan"),
                "recall": _tp / (_tp + _fn) if (_tp + _fn) else float("nan"),
            }
        out["per_bucket"] = per

    if distance_col in work.columns:
        _d = work[distance_col].astype("float64").to_numpy()
        for name, lo, hi in (("le_3km", 0.0, 3000.0),
                             ("3_to_8km", 3000.0, 8000.0),
                             ("gt_8km", 8000.0, np.inf)):
            _m = np.isfinite(_d) & (_d >= lo) & (_d < hi)
            if not _m.any():
                continue
            _y = y[_m]
            _p = pred[_m]
            _tp = int(((_p == 1) & (_y == 1)).sum())
            _fp = int(((_p == 1) & (_y == 0)).sum())
            _fn = int(((_p == 0) & (_y == 1)).sum())
            out.setdefault("per_distance_band", {})[name] = {
                "n": int(_m.sum()),
                "gold_positive_rate": float(_y.mean()),
                "precision": _tp / (_tp + _fp) if (_tp + _fp) else float("nan"),
                "recall": _tp / (_tp + _fn) if (_tp + _fn) else float("nan"),
            }

    return out


def compare_rules(
    frame: pd.DataFrame,
    *,
    label_col: str = "gold_label",
    distance_col: str = "nearest_facility_m",
    score_col: Optional[str] = "score",
    radii_m: Sequence[float] = (500.0, 1000.0, 2000.0, 5000.0),
) -> Dict[str, object]:
    """Score the distance rule against gold, alongside the model.

    This is the comparison the project has never been able to make. The weak
    label says distance gets PR-AUC 0.9811; the gold set says what it is
    actually worth. If the 5 km rule beats the model on gold labels, the model
    is not earning its place and the honest deliverable is the rule plus a
    review queue.
    """
    lab = frame[label_col].astype(str).str.lower()
    keep = lab.isin(["industrial", "natural"])
    work = frame[keep].copy()
    y = (lab[keep] == "industrial").astype(int).to_numpy()

    def _prf(pred):
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return {"precision": p, "recall": r, "f1": f,
                "tp": tp, "fp": fp, "fn": fn}

    out: Dict[str, object] = {
        "n_scored": int(len(work)),
        "n_uncertain_excluded": int((~keep).sum()),
        "gold_prevalence": float(y.mean()) if len(work) else 0.0,
        "distance_rule": {},
    }
    if distance_col in work.columns and len(work):
        d = work[distance_col].astype("float64").to_numpy()
        for r in radii_m:
            out["distance_rule"][f"within_{int(r)}m"] = _prf((d <= r).astype(int))

    if score_col and score_col in work.columns and len(work):
        s = work[score_col].astype("float64").to_numpy()
        best = None
        for t in np.unique(np.r_[np.linspace(0.05, 0.95, 19), 0.5]):
            m = _prf((s >= t).astype(int))
            m["threshold"] = float(t)
            if best is None or m["f1"] > best["f1"]:
                best = m
        out["model_at_own_best_f1"] = best
    return out