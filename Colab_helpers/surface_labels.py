# ============================================================================
# surface_labels.py -- a label built from evidence the model CANNOT see
#
# WHY THIS EXISTS
#
# The weak `INDUSTRIAL_FIRE` label was measured, not merely suspected, to be
# broken. Over the full catalogue (P7h, 2026-10-05):
#
#     substrate:  cropland 48.35%  tree 32.69%  grassland 8.89%
#                shrubland 2.79%  built-up 3.77%  bare 3.01%  water 0.47%
#
#     70.1% of weak INDUSTRIAL_FIRE sits on VEGETATED substrate.
#     Scored against non-vegetated substrate as a proxy:
#         precision 0.299   recall 0.636   F1 0.407
#
# Precision is the damning number: of everything the label calls an industrial
# fire, roughly 70% is farmland that can burn. The mechanism is structural --
# Indian industrial sites are ringed by cropland, so a 5 km proximity cut
# selects cropland almost by construction.
#
# WHY THE OLD LABEL WAS CIRCULAR, NOT JUST WRONG
#
# `weak_labels.LABEL_INPUT_FEATURES` publishes the 16 columns its rules read,
# and `training_design` excludes them from the scientific model. So model B
# was blind to exactly the evidence that made its own target. Fixing the label
# without changing WHICH EVIDENCE it uses leaves that intact. This module
# changes the evidence: the primary term is `LANDCOVER_VEGETATED`, which comes
# from a raster absent from X, so no model can reproduce the label from X.
#
# THE NAME IS DELIBERATE, AND IT IS NOT `INDUSTRIAL_FIRE`
#
# What the evidence supports is "a fire on non-vegetated substrate". That is
# measurably true and independently verifiable. "Industrial" is a claim about
# CAUSE, which needs either an industrial site corroboration or NBR burn-scar
# evidence. Calling this class INDUSTRIAL_FIRE would repeat the original sin
# one level down: naming a thing for what we hope it means.
#
# So:
#   SURFACE_FIRE        fire on built-up / bare / water substrate
#   VEGETATION_FIRE     fire on crop / grass / tree / shrub / wetland
#
# and the gazetteer becomes CORROBORATION for the first, reported alongside it,
# rather than the definition of it.
#
# HONEST LIMITS, STATED UP FRONT
#
#   * Non-vegetated substrate is not industrial. A landfill fire, a stubble
#     burn on a concrete pad, and a rural structure fire all qualify.
#   * Vegetated substrate is not agricultural. A forest fire inside a mining
#     lease does not qualify, and arguably should.
#   * Built-up substrate has LOW NBR to begin with, so an NBR-drop test is
#     much weaker here than on vegetation. That is why NBR is NOT in the base
#     rule: it would add cost and noise for little discrimination on exactly
#     the substrate this class is defined over.
#   * WorldCover is a 2021 static map applied to events from 2012-2026.
#
# The class is therefore a defensible, non-circular, substrate-based target --
# not a finished answer to "which fires are industrial". It is the first target
# in this project that a model cannot trivially reproduce, and that is what
# makes every metric computed against it worth reading.
# ============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

#: The two classes. Deliberately named for substrate, not for cause.
SURFACE_FIRE = "SURFACE_FIRE"
VEGETATION_FIRE = "VEGETATION_FIRE"
UNKNOWN = "UNKNOWN"

SURFACE_CLASSES = (SURFACE_FIRE, VEGETATION_FIRE, UNKNOWN)

#: Bumped when the label definition changes. Written into the output so a
#: stored label states which definition produced it.
_VERSION = "1.0.0"

#: WorldCover class codes. Codes 70 (snow and ice) and 80 (permanent water) are
#: in here deliberately and are WRONG for the intuitive reading: a thermal
#: anomaly over water or a glacier is not an industrial fire. They are included
#: because "cannot burn vegetation" is the physical property being tested, and
#: because excluding them would need a separate negative class for 33,869 events.
NON_VEGETATED_CODES = frozenset({50, 60, 70, 80})
VEGETATED_CODES = frozenset({10, 20, 30, 40, 90, 95, 100})


#: Codes that are non-vegetated. A separate constant rather than a threshold
#: in the label, because this boundary IS the label and there is nothing to tune
#: about it. `NON_VEGETATED_CODES` below; this is the default code set.
DEFAULT_NEGATED_CODES = frozenset({50, 60, 70, 80})


@dataclass(frozen=True)
class SurfaceThresholds:
    """Every number the surface rule uses, in one auditable place.

    There is exactly ONE knob, and it is not in the label itself.

    A first draft carried `min_persistence` AND `min_detections` as filters.
    That was wrong twice over. `FRP_PERSISTENCE = observed_days / 30` and
    `DETECTION_COUNT_LOG = log1p(detections)` measure the SAME evidence --
    repeat observation -- so one of the two thresholds was near-inert: the pool
    grid moved by 1,859 events across a six-fold change in `min_persistence`
    while `min_detections` moved it by 26,000. That is precisely the
    near-vacuous-threshold shape that destroyed the weak label's persistence
    branch, reintroduced by accident.

    And filtering on repeat observation at all cost 97% of the catalogue:
    96.7% of events are single-cluster, so requiring a second detection left
    7,026,705 of 7,265,849 rows UNKNOWN.

    So the label is substrate only, and repeat observation is reported as a
    separate CONFIDENCE TIER by `sustained_tier`. Two thresholds on one kind
    of evidence is one threshold too many.
    """

    #: Codes treated as non-vegetated. This IS the class boundary.
    negated_codes: frozenset = DEFAULT_NEGATED_CODES
    #: Repeat observation required for the HIGH tier, in observed days.
    #: Reported, not filtered on. 3.3% of events have more than one detection.
    tier_min_days: int = 3

    def describe(self) -> str:
        return (f"label = substrate only (non-vegetated codes "
                f"{sorted(self.negated_codes)}); "
                f"high-confidence tier = >={self.tier_min_days} observed days")


def _ge(series, value) -> pd.Series:
    """`series >= value`, where a missing value is False.

    Explicit rather than relying on NaN comparison semantics, because "could not
    measure" must never satisfy an evidence threshold.
    """
    return series.notna() & (series >= value)


def _le(series, value) -> pd.Series:
    return series.notna() & (series <= value)


def surface_label_inputs() -> list:
    """The columns `assign_surface_labels` reads.

    `LANDCOVER_VEGETATED` is the important one: it is derived from a raster and
    appears nowhere in X, so it cannot be reproduced from the feature table.
    """
    return ["LANDCOVER_VEGETATED", "WORLDCOVER_CLASS", "FRP_PERSISTENCE",
            "DETECTION_COUNT_LOG"]


def assign_surface_labels(frame: pd.DataFrame,
                          thresholds: Optional[SurfaceThresholds] = None,
                          ) -> pd.DataFrame:
    """Classify each event by substrate and by whether the heat was sustained.

    Order matters and is one-directional: an unmapped or unmeasured event is
    UNKNOWN, and UNKNOWN is never promoted. A rule that can only ever add
    evidence cannot manufacture a positive.

    Returns the input frame plus:
        label_surf      one of SURFACE_CLASSES
        is_surface_fire 1/0, and NaN where unknown
        substrate_known 1/0
        sustained       1/0 -- the thermal evidence, independent of substrate
    """
    th = thresholds or SurfaceThresholds()

    missing = [c for c in surface_label_inputs() if c not in frame.columns]
    if missing:
        raise KeyError(
            f"missing label inputs: {missing}. The surface label reads "
            f"{surface_label_inputs()}; without them it cannot be built, and "
            f"guessing would silently produce a proximity label again."
        )

    out = frame.copy()
    veg = out["LANDCOVER_VEGETATED"]

    substrate_known = veg.notna() & (out["WORLDCOVER_CLASS"] >= 0)
    # Cross-check the two substrate columns against each other. If they
    # disagree, one of them is wrong and that must surface rather than resolve
    # silently in favour of whichever was written first.
    derived = np.where(out["WORLDCOVER_CLASS"] >= 0,
                       np.isin(out["WORLDCOVER_CLASS"],
                               sorted(VEGETATED_CODES)).astype(float), np.nan)
    mismatch = substrate_known & (derived != veg)
    n_mismatch = int(mismatch.sum())

    label = np.full(len(out), UNKNOWN, dtype=object)
    # Substrate is the WHOLE rule. One-directional: UNKNOWN unless the
    # substrate is measured, and no threshold anywhere can promote an unmeasured
    # row. A draft filtered on repeat observation too, which cost 97% of the
    # catalogue (96.7% of events are single-cluster) and bought nothing, since
    # persistence and detection count are the same evidence.
    _known = substrate_known.to_numpy()
    _nonveg = out["WORLDCOVER_CLASS"].isin(sorted(th.negated_codes)).to_numpy()
    label[_known & _nonveg] = SURFACE_FIRE
    label[_known & ~_nonveg] = VEGETATION_FIRE

    out["label_surf"] = label
    out["substrate_known"] = substrate_known.astype("int8")
    out["sustained"] = sustained_tier(out, th).astype("int8")
    out["is_surface_fire"] = np.where(
        label == SURFACE_FIRE, 1,
        np.where(label == VEGETATION_FIRE, 0, np.nan))
    #: SURFACE_FIRE restricted to the high-confidence tier. A stricter target
    #: for anyone who wants it, reported alongside rather than instead of.
    out["is_surface_fire_tiered"] = np.where(
        label == SURFACE_FIRE, out["sustained"], np.nan)

    # Metadata goes in COLUMNS, not `.attrs`. A `SurfaceThresholds` in
    # `.attrs` is not JSON-serialisable, so `to_parquet` warns and refuses,
    # and P7i silently fell back to a 7.27M-row CSV. Two ordinary int columns
    # survive every writer.
    out["surface_label_version"] = _VERSION
    out["surface_n_substrate_mismatch"] = int(n_mismatch)
    out.attrs["thresholds"] = th
    return out


def sustained_tier(frame: pd.DataFrame,
                   thresholds: Optional[SurfaceThresholds] = None) -> pd.Series:
    """Repeat observation, as a CONFIDENCE TIER rather than a label filter.

    Reported, never subtracted. Requiring it inside the label discarded 97% of
    the catalogue for no gain, because `FRP_PERSISTENCE = observed_days / 30`
    and `DETECTION_COUNT_LOG = log1p(detections)` are the same evidence.
    """
    th = thresholds or SurfaceThresholds()
    have = [c for c in ("FRP_PERSISTENCE", "DETECTION_COUNT_LOG")
            if c in frame.columns]
    if not have:
        return pd.Series(False, index=frame.index)
    t = _ge(frame["FRP_PERSISTENCE"], th.tier_min_days / 30.0)
    if "DETECTION_COUNT_LOG" in frame.columns:
        t = t & _ge(frame["DETECTION_COUNT_LOG"],
                    float(np.log1p(th.tier_min_days)))
    return t


def surface_label_report(frame: pd.DataFrame,
                         thresholds: Optional[SurfaceThresholds] = None,
                         ) -> pd.DataFrame:
    """Class counts and the evidence breakdown, for the cell's report."""
    th = thresholds or SurfaceThresholds()
    out = assign_surface_labels(frame, th)
    rows = []
    n = len(out)
    for cls in SURFACE_CLASSES:
        m = out["label_surf"] == cls
        rows.append({
            "label": cls,
            "n": int(m.sum()),
            "pct": 100.0 * int(m.sum()) / max(n, 1),
            "pct_sustained": (100.0 * float(out.loc[m, "sustained"].mean())
                              if m.any() else float("nan")),
            "median_persistence": (float(out.loc[m, "FRP_PERSISTENCE"].median())
                                   if m.any() else float("nan")),
        })
    return pd.DataFrame(rows)


def pool_size(frame: pd.DataFrame, extra_negated_codes=()) -> int:
    """SURFACE_FIRE count for a substrate boundary.

    The only thing worth sweeping is WHICH CODES count as non-vegetated, because
    that is the class boundary. Sweeping a repeat-observation filter was wrong:
    it is the same evidence twice, and it discarded 97% of the catalogue.
    """
    codes = frozenset(DEFAULT_NEGATED_CODES | set(extra_negated_codes))
    th = SurfaceThresholds(negated_codes=codes)
    out = assign_surface_labels(frame, th)
    return int((out["label_surf"] == SURFACE_FIRE).sum())


def pool_grid(frame: pd.DataFrame) -> pd.DataFrame:
    """Pool size across the substrate-boundary choices worth considering.

    Includes the tier sizes, so the cost of demanding repeat observation is
    visible as a number rather than felt as a surprise.
    """
    rows = []
    variants = {
        "catalogued {50,60,70,80}": DEFAULT_NEGATED_CODES,
        "built+bare only {50,60}": frozenset({50, 60}),
        "built+bare+water {50,60,80}": frozenset({50, 60, 80}),
        "strictly built {50}": frozenset({50}),
    }
    for name, codes in variants.items():
        th = SurfaceThresholds(negated_codes=codes)
        out = assign_surface_labels(frame, th)
        rows.append({
            "boundary": name,
            "surface_fire": int((out["label_surf"] == SURFACE_FIRE).sum()),
            "vegetation_fire": int(
                (out["label_surf"] == VEGETATION_FIRE).sum()),
            "unknown": int((out["label_surf"] == UNKNOWN).sum()),
            "surface_fire_tiered": int(
                (out["is_surface_fire_tiered"] == 1).sum()),
        })
    return pd.DataFrame(rows)