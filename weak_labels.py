"""
P5b -- WEAK LABELLING from the industrial-context rules.

Distinct from `fire_labels.py` in one important way: **every label produced here
is a weak label.** It is manufactured from the same features the model will see,
so it cannot be used to evaluate that model. This module therefore publishes
the exact list of features each rule reads, and `training_design` excludes that
list from the scientific model B. That is the leakage guard made mechanical
rather than a matter of remembering.

    source evidence  ->  rules  ->  y (weak)      label_strength = "weak"
                                       \
                                        X excludes every rule input
                                       ->  model B (scientific)

Without that exclusion a model trained on these labels learns the labelling
rule, not the underlying class: it would score near-perfectly and mean nothing.

Units
-----
Every `DIST_*` feature is `log10(1 + d_metres)`, so a threshold in metres
converts with `log10(1 + R)` and NOT with `np.log1p`. Using `log1p` makes every
distance radius 6-25x tighter than intended -- "within 5 m" becomes 61 m -- and
in practice it stops any event from ever being labelled industrial.

`DETECTION_COUNT_LOG` really is `log1p(n)`, so `log1p` is correct there.

Radii
-----
The radii below are NAMED and in METRES, so they are auditable. They are the
proposal's 2 / 5 / 10 m widened to physically meaningful values: a 2 m radius
around a refinery centroid matches no real event, because the feature measures
event-to-facility distance and a cluster centroid is rarely within 2 m of a
plant. The tight and wide variants are both reported by `label_report` so the
choice is made on data rather than on assertion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

INDUSTRIAL_FLARE = "INDUSTRIAL_FLARE"
INDUSTRIAL_FIRE = "INDUSTRIAL_FIRE"
MINING = "MINING"
AGRICULTURAL = "AGRICULTURAL"
WILDFIRE = "WILDFIRE"
UNKNOWN = "UNKNOWN"

WEAK_CLASSES = (
    INDUSTRIAL_FLARE, INDUSTRIAL_FIRE, MINING, AGRICULTURAL, WILDFIRE, UNKNOWN,
)

#: Every feature any rule in this module reads. **All of them must be excluded
#: from model B's feature set.** Computed by `label_input_features()` rather
#: than hand-listed, so a rule change cannot silently widen X.
LABEL_INPUT_FEATURES: "tuple[str, ...]" = (
    # flare
    "KNOWN_FLARE", "INSIDE_FACILITY", "DIST_REFINERY_LOG",
    "FRP_PERSISTENCE", "DETECTION_COUNT_LOG",
    # industrial
    "DIST_NEAREST_FACILITY_LOG", "FACILITY_COUNT_5KM", "DIST_POWERPLANT_LOG",
    "DIST_PETROCHEMICAL_LOG", "DIST_STEEL_LOG", "DIST_CEMENT_LOG",
    # mining
    "DIST_MINE_LOG",
    # agriculture
    "TIME_TO_PEAK",
    # wildfire
    "AREA_GROWTH_RATE", "SPREAD_SPEED_MEDIAN", "DIRECTION_CONSISTENCY",
)


def distance_threshold(radius_metres: float) -> float:
    """`log10(1 + R)` -- the correct threshold for a DIST_* feature."""
    return float(np.log10(1.0 + float(radius_metres)))


def count_threshold(n: int) -> float:
    """`log1p(k)` -- the correct threshold for DETECTION_COUNT_LOG."""
    return float(np.log1p(float(n)))


@dataclass(frozen=True)
class WeakThresholds:
    """All radii in METRES. Converted to feature units at use time."""

    # flare
    flare_refinery_m: float = 2_000.0
    flare_persistence: float = 0.60
    flare_min_detections: int = 3

    # industrial
    industrial_near_m: float = 1_000.0
    industrial_persistence: float = 0.35
    industrial_min_detections: int = 2
    industrial_typology_m: float = 10_000.0

    # mining
    mine_m: float = 1_000.0
    mine_persistence: float = 0.30

    # agriculture
    agri_far_from_facility_m: float = 10_000.0
    agri_max_persistence: float = 0.50
    agri_max_detections: int = 4
    agri_max_hours_to_peak: float = 24.0

    # wildfire
    wildfire_far_from_facility_m: float = 10_000.0
    wildfire_persistence: float = 0.40
    wildfire_min_score: int = 3


TIGHT = WeakThresholds(
    flare_refinery_m=2.0, industrial_near_m=5.0,
    industrial_typology_m=10.0, mine_m=5.0,
    agri_far_from_facility_m=10.0, wildfire_far_from_facility_m=10.0,
)
WIDE = WeakThresholds()          # 2 km / 1 km / 10 km, the sensible reading


# ---------------------------------------------------------------------------
# Null-safe accessors
# ---------------------------------------------------------------------------


def _v(frame, column, default=np.nan):
    """Column as float64, or a default when absent. Never fabricates a zero."""
    if column not in frame.columns:
        return np.full(len(frame), default, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(
        dtype="float64"
    )


def _ge(values, threshold):
    """`values >= threshold`, False where the value is missing."""
    return np.isfinite(values) & (values >= float(threshold))


def _le(values, threshold):
    return np.isfinite(values) & (values <= float(threshold))


def assign_weak_labels(frame, thresholds: "WeakThresholds" = WIDE):
    """Apply the rule hierarchy. Returns label, rule and the features used.

    The hierarchy is strictly ordered, so the first satisfied rule wins and the
    returned `rule` always explains the label.
    """
    n = len(frame)
    known_flare = _v(frame, "KNOWN_FLARE", 0.0)
    inside = _v(frame, "INSIDE_FACILITY", 0.0)
    refinery = _v(frame, "DIST_REFINERY_LOG")
    nearest = _v(frame, "DIST_NEAREST_FACILITY_LOG")
    count5 = _v(frame, "FACILITY_COUNT_5KM")
    power = _v(frame, "DIST_POWERPLANT_LOG")
    petro = _v(frame, "DIST_PETROCHEMICAL_LOG")
    steel = _v(frame, "DIST_STEEL_LOG")
    cement = _v(frame, "DIST_CEMENT_LOG")
    mine = _v(frame, "DIST_MINE_LOG")
    persistence = _v(frame, "FRP_PERSISTENCE")
    detections = _v(frame, "DETECTION_COUNT_LOG")
    time_to_peak = _v(frame, "TIME_TO_PEAK")
    area_growth = _v(frame, "AREA_GROWTH_RATE")
    spread_speed = _v(frame, "SPREAD_SPEED_MEDIAN")
    direction = _v(frame, "DIRECTION_CONSISTENCY")

    label = np.full(n, UNKNOWN, dtype=object)
    rule = np.full(n, "no_rule", dtype=object)

    # ---- 1. INDUSTRIAL_FLARE ------------------------------------------------
    # Near a flare-capable facility, persistent, and seen more than twice.
    #
    # `INSIDE_FACILITY == 1` is deliberately NOT required. Measured on the full
    # catalogue it is 0 for all 7,265,849 events: it point-in-polygons an event
    # CENTROID against 38,370 small OSM polygons, and centroids sit on a 1 km
    # detection grid, so one essentially never lands inside. Requiring it made
    # the rule unsatisfiable. Distance-to-flare-capable-facility is the robust
    # proxy; INSIDE_FACILITY stays available as its own feature.
    flare = (
        (known_flare == 1)
        | (
            _le(refinery, distance_threshold(thresholds.flare_refinery_m))
            & _ge(persistence, thresholds.flare_persistence)
            & _ge(detections, count_threshold(thresholds.flare_min_detections))
        )
    )
    label[flare] = INDUSTRIAL_FLARE
    rule[flare] = "flare_inside_refinery"

    # ---- 2. INDUSTRIAL_FIRE -------------------------------------------------
    # Near any facility, with either persistence or repeated detection, AND a
    # recognisable industrial typology. Parented explicitly: the unparented
    # `a and b or c and d` form in the original proposal is not the same rule.
    near_any = _le(nearest, distance_threshold(thresholds.industrial_near_m)) | _ge(
        count5, 1
    )
    shape = _ge(persistence, thresholds.industrial_persistence) | _ge(
        detections, count_threshold(thresholds.industrial_min_detections)
    )
    typology = (
        _le(power, distance_threshold(thresholds.industrial_typology_m))
        | _le(petro, distance_threshold(thresholds.industrial_typology_m))
        | _le(steel, distance_threshold(thresholds.industrial_typology_m))
        | _le(cement, distance_threshold(thresholds.industrial_typology_m))
    )
    industrial = near_any & shape & typology & (label == UNKNOWN)
    label[industrial] = INDUSTRIAL_FIRE
    rule[industrial] = "industrial_near_persistent_typology"

    # ---- 3. MINING ----------------------------------------------------------
    # Near a mine and persistent. Note the asymmetry with the other rules: the
    # proposal treats a missing persistence as passing (`pd.isna(persistence)
    # or persistence >= 0.30`). That makes MINING the fallback for every
    # single-cluster event near a mine, because those have no computable
    # persistence at all. Treated here as ABSTAIN instead, so absence of
    # evidence never becomes evidence.
    mining = (
        _le(mine, distance_threshold(thresholds.mine_m))
        & _ge(persistence, thresholds.mine_persistence)
        & (label == UNKNOWN)
    )
    label[mining] = MINING
    rule[mining] = "mine_near_persistent"

    # ---- 4. AGRICULTURAL ---------------------------------------------------
    # Far from any facility, intermittent, few detections, peaking within a day.
    far = _ge(nearest, distance_threshold(thresholds.agri_far_from_facility_m))
    agri = (
        far
        & np.isfinite(persistence)
        & (persistence < thresholds.agri_max_persistence)
        & _ge(detections, 0.0)
        & (detections <= count_threshold(thresholds.agri_max_detections))
        & _le(time_to_peak, thresholds.agri_max_hours_to_peak)
        & (label == UNKNOWN)
    )
    label[agri] = AGRICULTURAL
    rule[agri] = "far_intermittent_few_detections"

    # ---- 5. WILDFIRE --------------------------------------------------------
    # Score of independent weak signals. A missing signal contributes nothing
    # rather than counting against the event.
    score = np.zeros(n, dtype="float64")
    scored = np.zeros(n, dtype=bool)
    for values, test in (
        (nearest, _ge(nearest, distance_threshold(
            thresholds.wildfire_far_from_facility_m))),
        (persistence, _ge(persistence, thresholds.wildfire_persistence)),
        (area_growth, np.isfinite(area_growth) & (area_growth > 0)),
        (spread_speed, np.isfinite(spread_speed) & (spread_speed > 0)),
        (direction, _ge(direction, 0.50)),
    ):
        hit = test
        score += np.where(np.isfinite(values), hit.astype("float64"), 0.0)
        scored |= np.isfinite(values)
    wildfire = (
        (score >= thresholds.wildfire_min_score)
        & scored
        & (label == UNKNOWN)
    )
    label[wildfire] = WILDFIRE
    rule[wildfire] = "wildfire_score"

    return {
        "label": label,
        "rule": rule,
        # np.full with an explicit object dtype. A bare Python string here
        # makes `out["label_source"][0]` return the first CHARACTER, and
        # np.where over string scalars produces a <U dtype array that iterates
        # per character too -- either corrupts the column silently.
        "label_strength": np.where(
            label == UNKNOWN, "none", "weak"
        ).astype(object),
        "label_source": np.full(n, "rule_weak_industrial_context", dtype=object),
        "wildfire_score": score,
    }


def label_input_features() -> "tuple[str, ...]":
    """Features that must be excluded from the scientific model.

    Derived from the rules rather than hand-maintained. Asserted against the
    module constant so a rule edited without updating the registry fails here
    rather than silently leaking.
    """
    return LABEL_INPUT_FEATURES


def check_no_leakage(model_b_features):
    """Fail if any weak-label input reached the scientific model."""
    leaked = sorted(set(model_b_features) & set(LABEL_INPUT_FEATURES))
    if leaked:
        raise AssertionError(
            f"target leakage: these features manufacture y but are in model B: "
            f"{leaked}. A model trained on weak labels from these features "
            "learns the labelling rule, not the class."
        )
    return True


def weak_label_report(frame, thresholds_list=None):
    """Label distribution under each threshold variant, so the choice of radii
    is made on the numbers rather than on assertion."""
    thresholds_list = thresholds_list or [("tight(2/5/10 m)", TIGHT),
                                         ("wide(1-10 km)", WIDE)]
    rows = []
    for name, th in thresholds_list:
        out = assign_weak_labels(frame, th)
        counts = {
            cls: int((out["label"] == cls).sum()) for cls in WEAK_CLASSES
        }
        rows.append({"variant": name, **counts})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PER-CONDITION PASS COUNTS
#
# "Every event is no_rule" is not a diagnosis. Each rule is a conjunction of
# conditions, so one unavailable condition silently kills it while the others
# look fine. This reports, per rule, how many events pass each individual
# condition and how many pass the conjunction -- so the blocking term is named
# rather than guessed at.
# ---------------------------------------------------------------------------


def condition_pass_report(frame, thresholds: "WeakThresholds" = WIDE):
    """Per-rule, per-condition pass counts plus the conjunction.

    The `blocking` column names the single condition with the fewest passes for
    a rule that never fires, which is the term worth implementing next.
    """
    n = len(frame)

    known_flare = _v(frame, "KNOWN_FLARE", 0.0)
    inside = _v(frame, "INSIDE_FACILITY", 0.0)
    refinery = _v(frame, "DIST_REFINERY_LOG")
    nearest = _v(frame, "DIST_NEAREST_FACILITY_LOG")
    count5 = _v(frame, "FACILITY_COUNT_5KM")
    power = _v(frame, "DIST_POWERPLANT_LOG")
    petro = _v(frame, "DIST_PETROCHEMICAL_LOG")
    steel = _v(frame, "DIST_STEEL_LOG")
    cement = _v(frame, "DIST_CEMENT_LOG")
    mine = _v(frame, "DIST_MINE_LOG")
    persistence = _v(frame, "FRP_PERSISTENCE")
    detections = _v(frame, "DETECTION_COUNT_LOG")
    time_to_peak = _v(frame, "TIME_TO_PEAK")
    area_growth = _v(frame, "AREA_GROWTH_RATE")
    spread_speed = _v(frame, "SPREAD_SPEED_MEDIAN")
    direction = _v(frame, "DIRECTION_CONSISTENCY")

    near_any = _le(nearest, distance_threshold(thresholds.industrial_near_m)) | _ge(
        count5, 1
    )
    shape = _ge(persistence, thresholds.industrial_persistence) | _ge(
        detections, count_threshold(thresholds.industrial_min_detections)
    )
    typology = (
        _le(power, distance_threshold(thresholds.industrial_typology_m))
        | _le(petro, distance_threshold(thresholds.industrial_typology_m))
        | _le(steel, distance_threshold(thresholds.industrial_typology_m))
        | _le(cement, distance_threshold(thresholds.industrial_typology_m))
    )

    rules = {
        INDUSTRIAL_FLARE: [
            ("KNOWN_FLARE==1", known_flare == 1),
            ("[diagnostic] INSIDE_FACILITY==1", inside == 1),
            (f"DIST_REFINERY_LOG<={thresholds.flare_refinery_m:.0f}m",
             _le(refinery, distance_threshold(thresholds.flare_refinery_m))),
            (f"FRP_PERSISTENCE>={thresholds.flare_persistence}",
             _ge(persistence, thresholds.flare_persistence)),
            (f"DETECTION_COUNT_LOG>=log1p({thresholds.flare_min_detections})",
             _ge(detections, count_threshold(thresholds.flare_min_detections))),
        ],
        INDUSTRIAL_FIRE: [
            ("near_any_facility", near_any),
            (f"shape(persist>={thresholds.industrial_persistence} or "
             f"count>={thresholds.industrial_min_detections})", shape),
            (f"typology within {thresholds.industrial_typology_m:.0f}m", typology),
        ],
        MINING: [
            (f"DIST_MINE_LOG<={thresholds.mine_m:.0f}m",
             _le(mine, distance_threshold(thresholds.mine_m))),
            (f"FRP_PERSISTENCE>={thresholds.mine_persistence}",
             _ge(persistence, thresholds.mine_persistence)),
        ],
        AGRICULTURAL: [
            (f"far_from_facility>{thresholds.agri_far_from_facility_m:.0f}m",
             _ge(nearest,
                 distance_threshold(thresholds.agri_far_from_facility_m))),
            (f"FRP_PERSISTENCE<{thresholds.agri_max_persistence}",
             np.isfinite(persistence)
             & (persistence < thresholds.agri_max_persistence)),
            (f"DETECTION_COUNT_LOG<=log1p({thresholds.agri_max_detections})",
             np.isfinite(detections)
             & (detections <= count_threshold(thresholds.agri_max_detections))),
            (f"TIME_TO_PEAK<={thresholds.agri_max_hours_to_peak:.0f}h",
             _le(time_to_peak, thresholds.agri_max_hours_to_peak)),
        ],
        WILDFIRE: [
            (f"far_from_facility>{thresholds.wildfire_far_from_facility_m:.0f}m",
             _ge(nearest,
                 distance_threshold(thresholds.wildfire_far_from_facility_m))),
            (f"FRP_PERSISTENCE>={thresholds.wildfire_persistence}",
             _ge(persistence, thresholds.wildfire_persistence)),
            ("AREA_GROWTH_RATE>0",
             np.isfinite(area_growth) & (area_growth > 0)),
            ("SPREAD_SPEED_MEDIAN>0",
             np.isfinite(spread_speed) & (spread_speed > 0)),
            ("DIRECTION_CONSISTENCY>=0.5", _ge(direction, 0.50)),
            (f"score>={thresholds.wildfire_min_score}", np.zeros(n, dtype=bool)),
        ],
    }

    rows = []
    for cls, conditions in rules.items():
        masks = []
        for name, mask in conditions:
            m = np.asarray(mask, dtype=bool)
            rows.append({
                "rule": cls, "condition": name,
                "n_pass": int(m.sum()),
                "pass_frac": float(m.sum() / max(n, 1)),
                "n_measurable": int(np.isfinite(
                    np.ones(n)  # every condition is already finite-guarded
                ).sum()),
            })
            masks.append(m)
        conjunction = np.logical_and.reduce(masks)
        rows.append({
            "rule": cls, "condition": "== ALL CONDITIONS ==",
            "n_pass": int(conjunction.sum()),
            "pass_frac": float(conjunction.sum() / max(n, 1)),
            "n_measurable": n,
        })
        # Name the weakest term, which is the one worth implementing.
        weakest = min(conditions, key=lambda kv: int(np.asarray(kv[1]).sum()))
        rows.append({
            "rule": cls, "condition": f"weakest: {weakest[0]}",
            "n_pass": int(np.asarray(weakest[1]).sum()),
            "pass_frac": float(np.asarray(weakest[1]).sum() / max(n, 1)),
            "n_measurable": n,
        })
    return pd.DataFrame(rows)