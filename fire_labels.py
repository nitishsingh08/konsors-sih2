"""
P5 -- LABEL ENGINE (rule cascade, per HANDOFF section 8).

Assigns one of six classes to each global fire event from evidence that is NOT
used as a model feature. Labels come from the INDUSTRIAL partition plus a
handful of persistence/thermal signals, which is exactly why those 18 features
are declared `LABEL_DERIVED` and excluded from the scientific model B.

    INDUSTRIAL_FLARE   stack/flare: persistent, stationary, constant output
    INDUSTRIAL_FIRE    facility-adjacent burn with surface loss
    MINING             bare-mining land cover + persistent thermal
    AGRICULTURAL       seasonal, large, low FRP, diurnal
    WILDFIRE           hot, dry, wind-driven displacement
    UNKNOWN            insufficient evidence -- KEPT, down-weighted

Two properties this module guarantees, because both are ways a labelling engine
silently corrupts a dataset:

1. **Absence of evidence never produces a class.** A rule can only fire on
   positive evidence for that class. If the evidence is missing the event falls
   through to `UNKNOWN`, which is retained with a low `sample_weight` rather than
   dropped or force-assigned.

2. **Missing is NaN, never 0.** `MISSING_FEATURE_FRACTION` is computed across
   the other 140 features precisely so that "we could not measure this" stays
   distinguishable from "we measured zero". Any NaN input makes the rules that
   depend on it abstain rather than guess.

Every threshold here is a `NEW_IMPLEMENTATION` decision declared in this file.
None of it is inherited from the historical notebooks, which contain no label
code at all. Thresholds are collected in one dataclass so they can be audited,
and the module is unit-tested against synthetic events.

Precedence matters and is fixed: the cascade is evaluated in the order below and
the first rule that fires wins. Flare is checked first because a flare is a
subset of "near a facility" and the more specific claim should win.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# CLASS VOCABULARY
# ---------------------------------------------------------------------------

INDUSTRIAL_FLARE = "INDUSTRIAL_FLARE"
INDUSTRIAL_FIRE = "INDUSTRIAL_FIRE"
MINING = "MINING"
AGRICULTURAL = "AGRICULTURAL"
WILDFIRE = "WILDFIRE"
UNKNOWN = "UNKNOWN"

LABEL_CLASSES: "tuple[str, ...]" = (
    INDUSTRIAL_FLARE,
    INDUSTRIAL_FIRE,
    MINING,
    AGRICULTURAL,
    WILDFIRE,
    UNKNOWN,
)

#: The five classes a model is actually trained to predict. UNKNOWN is not a
#: prediction target; it marks events we decline to label.
SUPERVISED_CLASSES: "tuple[str, ...]" = LABEL_CLASSES[:-1]

#: Relative weight per class during fitting. UNKNOWN events stay in the dataset
#: so the class prior stays honest, but contribute little to the gradient.
DEFAULT_WEIGHTS: "dict[str, float]" = {
    INDUSTRIAL_FLARE: 3.0,
    INDUSTRIAL_FIRE: 2.0,
    MINING: 2.0,
    AGRICULTURAL: 1.0,
    WILDFIRE: 1.0,
    UNKNOWN: 0.2,
}

# ---------------------------------------------------------------------------
# THRESHOLDS -- every label decision in one auditable place
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelThresholds:
    """All label thresholds. Frozen so a run cannot mutate them mid-flight."""

    # -- INDUSTRIAL_FLARE ----------------------------------------------------
    # A flare is a continuous stationary source, so every condition is about
    # persistence and lack of displacement rather than magnitude.
    flare_min_active_days: int = 200
    flare_min_detection_day_ratio: float = 0.25
    flare_max_displacement_m: float = 2_000.0
    flare_max_frp_cv: float = 0.60
    flare_min_satellites: int = 2
    #: A flare-capable facility within this distance counts as "at" the plant.
    flare_max_facility_m: float = 1_000.0

    # -- INDUSTRIAL_FIRE -----------------------------------------------------
    fire_max_facility_m: float = 2_000.0
    #: Surface loss that does not recover: NBR drop at least this negative,
    #: and no regrowth by the later composite.
    fire_min_nbr_drop: float = -0.10
    fire_max_nbr_recovery: float = 0.05
    fire_min_active_days: int = 3

    # -- MINING --------------------------------------------------------------
    mine_max_facility_m: float = 1_000.0
    mine_min_bare_fraction: float = 0.50
    mine_min_active_days: int = 30

    # -- AGRICULTURAL -------------------------------------------------------
    agri_min_area_km2: float = 1.0
    agri_max_frp_median_mw: float = 30.0
    #: Fraction of detections that must fall in the local crop season.
    agri_min_season_fraction: float = 0.70
    #: Burning crops at night is rare; a low night/day FRP ratio indicates
    #: scheduled residue burning rather than a wild fire.
    agri_max_night_day_frp_ratio: float = 0.60
    agri_min_active_days: int = 5

    # -- WILDFIRE ------------------------------------------------------------
    wildfire_min_bt_k: float = 350.0
    wildfire_min_vpd_kpa: float = 2.0
    wildfire_min_displacement_m: float = 5_000.0
    wildfire_min_wind_speed: float = 4.0

    #: An event must be at least this close to *some* facility to be eligible
    #: for an industrial class at all.
    industrial_max_facility_m: float = 2_000.0

    extra: "dict[str, float]" = field(default_factory=dict)


THRESHOLDS = LabelThresholds()


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def _col(frame, name):
    """Fetch a column as float64, or an all-NaN array if absent.

    An absent column must not silently become zero, so this returns NaN and
    every rule depending on it abstains.
    """
    if name not in frame:
        return np.full(len(frame), np.nan, dtype="float64")
    values = np.asarray(frame[name], dtype="float64")
    if values.shape[0] != len(frame):
        raise ValueError(
            f"column {name!r} has {values.shape[0]} rows, expected {len(frame)}"
        )
    return values


def _log_to_metres(log_value):
    """Invert log10(1 + d_m) -> d_m. NaN-preserving."""
    v = np.asarray(log_value, dtype="float64")
    return np.where(np.isfinite(v), np.power(10.0, v) - 1.0, np.nan)


def facility_distances_m(frame):
    """Metre distances to the nearest facility of each kind.

    Reads the P3a features in their log10(1 + m) form and converts back, so the
    label engine never has to agree with the extractor about units twice.
    """
    return {
        "nearest": _log_to_metres(_col(frame, "DIST_NEAREST_FACILITY_LOG")),
        "refinery": _log_to_metres(_col(frame, "DIST_REFINERY_LOG")),
        "powerplant": _log_to_metres(_col(frame, "DIST_POWERPLANT_LOG")),
        "petrochemical": _log_to_metres(_col(frame, "DIST_PETROCHEMICAL_LOG")),
        "steel": _log_to_metres(_col(frame, "DIST_STEEL_LOG")),
        "cement": _log_to_metres(_col(frame, "DIST_CEMENT_LOG")),
        "lng": _log_to_metres(_col(frame, "DIST_LNG_LOG")),
        "mine": _log_to_metres(_col(frame, "DIST_MINE_LOG")),
        "landfill": _log_to_metres(_col(frame, "DIST_LANDFILL_LOG")),
        "pipeline": _log_to_metres(_col(frame, "DIST_PIPELINE_LOG")),
        "wellpad": _log_to_metres(_col(frame, "DIST_WELLPAD_LOG")),
    }


def _min_available(*arrays):
    """Element-wise minimum, ignoring NaN. NaN only where all inputs are NaN."""
    stacked = np.vstack(arrays)
    with np.errstate(invalid="ignore"):
        masked = np.where(np.isfinite(stacked), stacked, np.inf)
    out = masked.min(axis=0)
    return np.where(np.isfinite(out), out, np.nan)


# ---------------------------------------------------------------------------
# RULE ENGINE
# ---------------------------------------------------------------------------


def assign_labels(frame, thresholds: "LabelThresholds" = THRESHOLDS):
    """Assign a label to every row of `frame`.

    Parameters
    ----------
    frame : pandas DataFrame or mapping of column -> array. Required columns
        are the P3a industrial features; every other column is optional and its
        absence causes the dependent rules to abstain (never to guess).

    Returns
    -------
    dict with:
        label        (n,) object array of LABELS_CLASSES values
        rule         (n,) object array of the rule that fired, "no_rule" for
                     UNKNOWN, "insufficient_evidence" when a rule was close but
                     its evidence was missing
        confidence   (n,) float64 in [0, 1]: how strongly the rule fired,
                     derived from how far past each threshold the event sits.
                     Used for sample weights and for manual review ordering.
    """
    n = len(frame)
    dist = facility_distances_m(frame)

    active_days = _col(frame, "active_days")
    detection_count = _col(frame, "detection_count")
    n_satellites = _col(frame, "n_satellites")
    displacement = _col(frame, "centroid_displacement_m")
    frp_cv = _col(frame, "FRP_CV")
    frp_median = _col(frame, "FRP_MEDIAN")
    bt_median = _col(frame, "BT_I4_MEDIAN")
    vpd = _col(frame, "VPD_ANOMALY_Z")
    wind_speed = _col(frame, "WIND_SPEED")
    area_km2 = _col(frame, "area_km2")
    nbr_change = _col(frame, "NBR_CHANGE")
    nbr_recovery = _col(frame, "NBR_RECOVERY")
    bare_fraction = _col(frame, "BARE_MINING_FRACTION")
    season_fraction = _col(frame, "SEASON_FRACTION")
    night_day_ratio = _col(frame, "NIGHT_DAY_FRP_RATIO")

    inside = _col(frame, "INSIDE_FACILITY")
    known_flare = _col(frame, "KNOWN_FLARE")

    label = np.full(n, UNKNOWN, dtype=object)
    rule = np.full(n, "insufficient_evidence", dtype=object)
    confidence = np.zeros(n, dtype="float64")

    # ---- why did each rule not fire? --------------------------------------
    # The first full assembly labelled 100% UNKNOWN and reported only
    # "no_rule", which gave no way to tell "threshold not met" from "the input
    # was never computed". Those demand completely different responses, so the
    # missing inputs are named per event and aggregated in the QC.
    # Collected as (name, missing_mask) pairs rather than pre-joined strings:
    # a per-event Python join over 7.27M rows x 6 rules is minutes of work, and
    # the obvious "stacked arrays merged by index" version silently conflated
    # the rule index with the event index.
    missing_pairs: list = []

    def note(*names):
        for name, values in names:
            values = np.asarray(values, dtype="float64")
            if values.shape[0] != n:
                continue
            missing_pairs.append((name, ~np.isfinite(values)))

    # Facility types that host flares.
    flare_capable = _min_available(
        dist["refinery"], dist["petrochemical"], dist["lng"]
    )

    # ---- Rule 1: INDUSTRIAL_FLARE ------------------------------------------
    days_ratio = np.divide(
        detection_count,
        np.maximum(active_days, 1.0),
        out=np.full(n, np.nan),
        where=np.isfinite(active_days) & np.isfinite(detection_count),
    )

    # KNOWN_FLARE (P3a) is accepted as-is when present: it already encodes the
    # same persistence rule. It is re-derived only when it is missing.
    # The flare rule's own inputs, recorded before it is evaluated. Without
    # this note the rule is silent: an event that abstains because
    # `n_satellites` was never computed looks identical to one that simply
    # failed the threshold, and the two need completely different fixes.
    note(("active_days", active_days),
         ("detection_count", detection_count),
         ("n_satellites", n_satellites),
         ("FRP_CV", frp_cv),
         ("centroid_displacement_m", displacement),
         ("flare_capable_distance", flare_capable))

    derived_flare = (
        np.isfinite(flare_capable)
        & (flare_capable <= thresholds.flare_max_facility_m)
        & (active_days >= thresholds.flare_min_active_days)
        & (days_ratio >= thresholds.flare_min_detection_day_ratio)
        & (displacement <= thresholds.flare_max_displacement_m)
        & (frp_cv <= thresholds.flare_max_frp_cv)
        & (n_satellites >= thresholds.flare_min_satellites)
    )

    flare = np.where(
        np.isfinite(known_flare), known_flare > 0.5, derived_flare
    )
    flare = np.nan_to_num(flare, nan=0.0).astype(bool)
    if flare.any():
        label[flare] = INDUSTRIAL_FLARE
        rule[flare] = "flare_persistence"
        # Confidence rises with how far past the persistence bar the event is.
        with np.errstate(invalid="ignore"):
            margin = (active_days[flare] - thresholds.flare_min_active_days) / max(
                thresholds.flare_min_active_days, 1
            )
        confidence[flare] = np.clip(0.7 + 0.3 * np.tanh(margin), 0.0, 1.0)

    # ---- Rule 2: MINING ----------------------------------------------------
    # Checked before the generic industrial fire rule because an open-pit mine
    # fire is industrial *and* mining; the project statement lists mining
    # separately, so it gets its own class.
    note(("BARE_MINING_FRACTION", bare_fraction),
         ("active_days", active_days))
    mining = (
        (np.isfinite(dist["mine"]) & (dist["mine"] <= thresholds.mine_max_facility_m))
        & (bare_fraction >= thresholds.mine_min_bare_fraction)
        & (active_days >= thresholds.mine_min_active_days)
    )
    mining &= label == UNKNOWN
    if mining.any():
        label[mining] = MINING
        rule[mining] = "mine_bare_persistent"
        confidence[mining] = 0.8

    # ---- Rule 3: INDUSTRIAL_FIRE -------------------------------------------
    surface_loss = (
        np.isfinite(nbr_change) & (nbr_change <= thresholds.fire_min_nbr_drop)
    )
    # "does not recover": the later composite must not have regrown.
    no_recovery = (
        np.isfinite(nbr_recovery) & (nbr_recovery <= thresholds.fire_max_nbr_recovery)
    )
    industrial = np.isfinite(dist["nearest"]) & (
        dist["nearest"] <= thresholds.fire_max_facility_m
    )
    # "Surface loss that does not recover" is a conjunction, not a disjunction.
    # Using OR here mislabels every seasonal burn at a plant gate: cropland
    # inside an industrial boundary drops in NBR and then regrows, and an OR
    # would call all of that INDUSTRIAL_FIRE. Both measurements must be present,
    # so a missing recovery composite abstains rather than assuming.
    note(("NBR_CHANGE", nbr_change), ("NBR_RECOVERY", nbr_recovery),
         ("active_days", active_days))
    industrial_fire = industrial & surface_loss & no_recovery
    industrial_fire &= active_days >= thresholds.fire_min_active_days
    industrial_fire &= label == UNKNOWN
    if industrial_fire.any():
        label[industrial_fire] = INDUSTRIAL_FIRE
        rule[industrial_fire] = "facility_surface_loss"
        confidence[industrial_fire] = 0.7

    # ---- Rule 4: AGRICULTURAL ----------------------------------------------
    agricultural = (
        (area_km2 >= thresholds.agri_min_area_km2)
        & (frp_median <= thresholds.agri_max_frp_median_mw)
        & (season_fraction >= thresholds.agri_min_season_fraction)
        & (night_day_ratio <= thresholds.agri_max_night_day_frp_ratio)
        & (active_days >= thresholds.agri_min_active_days)
    )
    # Burning inside a plant gate is not agriculture. Requiring the distance to
    # be *known* avoids labelling unknown-proximity events as cropland.
    agricultural &= np.isfinite(dist["nearest"])
    agricultural &= ~(
        np.isfinite(dist["nearest"])
        & (dist["nearest"] <= thresholds.industrial_max_facility_m)
    )
    agricultural &= label == UNKNOWN
    note(("SEASON_FRACTION", season_fraction),
         ("NIGHT_DAY_FRP_RATIO", night_day_ratio),
         ("area_km2", area_km2))
    if agricultural.any():
        label[agricultural] = AGRICULTURAL
        rule[agricultural] = "seasonal_low_frp"
        confidence[agricultural] = 0.6

    # ---- Rule 5: WILDFIRE --------------------------------------------------
    wildfire = (
        (bt_median >= thresholds.wildfire_min_bt_k)
        & (vpd >= thresholds.wildfire_min_vpd_kpa)
        & (displacement >= thresholds.wildfire_min_displacement_m)
        & (wind_speed >= thresholds.wildfire_min_wind_speed)
    )
    # A fire that is wind-driven *and* inside a plant gate is already
    # industrial; do not double-count it.
    wildfire &= ~(
        np.isfinite(dist["nearest"])
        & (dist["nearest"] <= thresholds.industrial_max_facility_m)
    )
    wildfire &= label == UNKNOWN
    note(("BT_I4_MEDIAN", bt_median), ("VPD_ANOMALY_Z", vpd),
         ("WIND_SPEED", wind_speed))
    if wildfire.any():
        label[wildfire] = WILDFIRE
        rule[wildfire] = "hot_dry_wind_driven"
        confidence[wildfire] = 0.6

    # ---- Everything left is UNKNOWN ----------------------------------------
    still = label == UNKNOWN
    rule[still & (rule == "insufficient_evidence")] = "no_rule"

    # Merge the per-rule notes into one string per event: the distinct inputs
    # that were unavailable for the rules that did not fire.
    # One vectorised pass per (name, mask). 15 passes over 7.27M rows is
    # seconds; a Python-level join would be minutes.
    merged = np.array([""] * n, dtype=object)
    for name, mask in missing_pairs:
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            continue
        merged = np.where(
            mask,
            np.char.add(
                np.char.add(np.where(merged == "", "", np.char.add(merged, ",")),
                            name),
                "",
            ),
            merged,
        )

    return {"label": label, "rule": rule, "confidence": confidence,
            "missing_inputs": merged}


def label_weights(
    labels, confidence=None, base: "dict[str, float] | None" = None
):
    """Per-event `sample_weight` for LightGBM.

    Combines the class weight with the rule's confidence, so a borderline
    industrial call counts for less than an unambiguous one and UNKNOWN events
    barely influence the fit without being discarded.
    """
    table = dict(base or DEFAULT_WEIGHTS)
    labels = np.asarray(labels, dtype=object)
    weights = np.array([table.get(v, 1.0) for v in labels], dtype="float64")
    if confidence is not None:
        conf = np.asarray(confidence, dtype="float64")
        conf = np.where(np.isfinite(conf), conf, 1.0)
        weights = weights * np.clip(conf, 0.05, 1.0)
    return weights


def label_report(labels, rules=None):
    """Counts per class and per firing rule. The P5 acceptance check."""
    labels = np.asarray(labels, dtype=object)
    total = len(labels)
    out = {
        "n_events": total,
        "counts": {c: int((labels == c).sum()) for c in LABEL_CLASSES},
        "fractions": {
            c: (float((labels == c).sum()) / total if total else 0.0)
            for c in LABEL_CLASSES
        },
    }
    if rules is not None:
        rules = np.asarray(rules, dtype=object)
        out["rules"] = {
            str(r): int((rules == r).sum()) for r in sorted(set(rules.tolist()))
        }
    out["n_supervised"] = int(sum(out["counts"][c] for c in SUPERVISED_CLASSES))
    out["n_unknown"] = out["counts"][UNKNOWN]
    out["supervised_fraction"] = (
        out["n_supervised"] / total if total else 0.0
    )
    return out
