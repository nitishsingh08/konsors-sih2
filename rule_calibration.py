"""
Rule calibration from the observed feature distribution.

A rule threshold is only meaningful relative to the distribution it is applied
to. `FRP_PERSISTENCE` was fixed to `observed_days / 30`, which moved it from a
constant 1.0 to mostly 0.03-0.3 -- so the thresholds written against the old
scale silently became far too strict, and INDUSTRIAL_FLARE fell from 3,571
events to 0 while MINING fell to 61.

Recalibrating by hand after every distribution change is how thresholds drift
away from the data without anyone noticing. This derives each threshold from
percentiles of the real catalogue instead, and reports how many events each
candidate admits, so the choice is visible rather than asserted.

The calibrated thresholds are a function of the data, so they are written out
alongside the labels and recorded in the manifest. Re-running the labeller on the
same data reproduces the same labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

import weak_labels as wl

#: Percentiles used to propose thresholds.
#:
#: These must sit ABOVE the share of events that carry the minimum persistence.
#: 96.7% of events are single-cluster and score exactly 1/30 = 0.0333, so any
#: percentile at or below 0.967 resolves to that same floor and every rule
#: collapses onto it. A target of ~0.05% of 7.27M events (about 3,600) needs a
#: percentile near 0.9995.
DEFAULT_PERCENTILES = {
    "flare_persistence": 0.9995,
    "industrial_persistence": 0.99,
    "mine_persistence": 0.999,
    "wildfire_persistence": 0.98,
}


@dataclass(frozen=True)
class Calibration:
    """A threshold plus the evidence behind it."""

    value: float
    percentile: float
    n_above: int
    n_total: int
    feature: str

    @property
    def share_above(self) -> float:
        return self.n_above / max(self.n_total, 1)


def _finite(values) -> np.ndarray:
    arr = np.asarray(values, dtype="float64")
    return arr[np.isfinite(arr)]


def quantile_for(frame, column, percentile, default=1.0):
    """Threshold admitting `1 - percentile` of the finite values.

    Tolerates an absent column: the whole column is treated as missing and
    `default` is returned, rather than raising. A rule whose input is not in X
    yet must still be calibratable, so the report can say which inputs are
    missing rather than dying on them.
    """
    if column not in frame.columns:
        return float(default), 0
    values = _finite(frame[column])
    if values.size == 0:
        return float(default), 0
    value = float(np.quantile(values, percentile))
    return value, int((values >= value).sum())


def calibrate(frame, percentiles=None, base=None):
    """Derive every scale-sensitive threshold from the data.

    Returns (thresholds, evidence) where `thresholds` is a `WeakThresholds`
    instance ready to pass to `assign_weak_labels`, and `evidence` lists what
    each threshold became and why.
    """
    percentiles = {**DEFAULT_PERCENTILES, **(percentiles or {})}
    base = base or wl.WIDE
    updates = {}
    evidence = []

    mapping = (
        ("flare_persistence", "FRP_PERSISTENCE", "flare_persistence"),
        ("industrial_persistence", "FRP_PERSISTENCE", "industrial_persistence"),
        ("mine_persistence", "FRP_PERSISTENCE", "mine_persistence"),
        ("wildfire_persistence", "FRP_PERSISTENCE", "wildfire_persistence"),
    )
    for attr, column, key in mapping:
        p = percentiles[key]
        value, n_above = quantile_for(frame, column, p)
        updates[attr] = value
        finite = _finite(frame[column]) if column in frame \
            else np.empty(0)
        evidence.append(Calibration(
            value=value, percentile=p, n_above=n_above,
            n_total=int(finite.size), feature=column,
        ))

    thresholds = replace(base, **updates)
    return thresholds, evidence


def calibrate_to_target_counts(frame, targets, percentiles=None, base=None):
    """Pick thresholds so each class lands near a target event count.

    Bisects the percentile until the rule admits roughly `targets[class]`
    events. More directly interpretable than a percentile, because the number the
    label engine will actually produce is the thing being chosen.
    """
    percentiles = {**DEFAULT_PERCENTILES, **(percentiles or {})}
    thresholds, evidence = calibrate(frame, percentiles, base)

    finite = _finite(frame["FRP_PERSISTENCE"]) if "FRP_PERSISTENCE" in frame \
        else np.empty(0)
    if finite.size:
        ceiling = float(np.quantile(finite, 1.0 - 1.0 / max(finite.size, 2)))
    else:
        ceiling = 1.0
    floor = float(finite.min()) if finite.size else 0.0

    for cls, target in targets.items():
        if target <= 0:
            continue
        attr = {
            wl.INDUSTRIAL_FLARE: "flare_persistence",
            wl.INDUSTRIAL_FIRE: "industrial_persistence",
            wl.MINING: "mine_persistence",
        }.get(cls)
        if attr is None:
            continue

        def admitted(value):
            trial = replace(thresholds, **{attr: float(value)})
            out = wl.assign_weak_labels(frame, trial)
            return int((out["label"] == cls).sum())

        # Admitted count is monotone non-increasing in the threshold, so a plain
        # bisection on the VALUE converges. Bisecting the percentile got the
        # direction wrong and stalled on the first candidate.
        lo, hi = floor, max(ceiling, floor + 1e-9)
        best, best_err = getattr(thresholds, attr), None
        if admitted(lo) < target:
            # Even the floor admits too few: no threshold can reach the target.
            best = lo
        else:
            for _ in range(40):
                mid = (lo + hi) / 2.0
                n = admitted(mid)
                err = abs(n - target)
                if best_err is None or err < best_err:
                    best_err, best = err, mid
                if n > target:
                    lo = mid          # raise the bar to admit fewer
                else:
                    hi = mid
                if err <= max(50, target * 0.01):
                    break
        thresholds = replace(thresholds, **{attr: best})

    # Recompute evidence against the final thresholds.
    final = []
    for attr, key in (
        ("flare_persistence", "flare_persistence"),
        ("industrial_persistence", "industrial_persistence"),
        ("mine_persistence", "mine_persistence"),
        ("wildfire_persistence", "wildfire_persistence"),
    ):
        p = percentiles[key]
        value = getattr(thresholds, attr)
        finite = _finite(frame["FRP_PERSISTENCE"]) \
            if "FRP_PERSISTENCE" in frame else np.empty(0)
        final.append(Calibration(
            value=value, percentile=p,
            n_above=int((finite >= value).sum()), n_total=len(finite),
            feature="FRP_PERSISTENCE",
        ))
    return thresholds, final


def distribution(frame, columns=None, quantiles=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)):
    """Percentile table for the features the rules actually read."""
    columns = columns or [
        "FRP_PERSISTENCE", "DETECTION_COUNT_LOG", "TIME_TO_PEAK",
        "AREA_GROWTH_RATE", "SPREAD_SPEED_MEDIAN", "DIRECTION_CONSISTENCY",
        "DIST_NEAREST_FACILITY_LOG", "DIST_REFINERY_LOG", "DIST_MINE_LOG",
    ]
    rows = []
    for column in columns:
        if column not in frame.columns:
            continue
        values = _finite(frame[column])
        if values.size == 0:
            continue
        row = {"feature": column, "n_finite": int(values.size),
               "frac_finite": values.size / max(len(frame), 1)}
        for q in quantiles:
            row[f"p{int(q * 100)}"] = float(np.quantile(values, q))
        row["max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_report(frame, thresholds, evidence):
    """What each threshold became, and what the resulting labels look like."""
    out = wl.assign_weak_labels(frame, thresholds)
    report = wl.weak_label_report(frame, [("calibrated", thresholds)])
    rows = []
    for item in evidence:
        rows.append({
            "threshold": item.feature,
            "percentile": item.percentile,
            "value": round(item.value, 6),
            "n_above": item.n_above,
            "share_above": round(item.share_above, 4),
        })
    return pd.DataFrame(rows), report, out


def thresholds_to_dict(thresholds):
    return asdict(thresholds)


def thresholds_from_dict(payload):
    """Rebuild a WeakThresholds, ignoring keys the dataclass no longer has."""
    known = {f for f in wl.WeakThresholds.__dataclass_fields__}
    return wl.WeakThresholds(**{k: v for k, v in payload.items() if k in known})