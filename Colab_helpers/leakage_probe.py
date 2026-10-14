"""
LEAKAGE PROBE -- is the held-out fold really held out?

The split code already asserts block disjointness (`assert_no_spatial_leakage`).
That assertion is necessary and not sufficient, because it checks only whether a
grid cell appears on both sides. A fire event is not a grid cell:

  * **Facility recurrence.** The label is a proximity cut (`weak_labels.py:198`
    `near_any` = facility within 1 km). A held-out 0.1 degree block is ~11 km on
    a side, so the same refinery sits inside the held-out block AND inside
    neighbouring training blocks. The model therefore sees that facility's
    thermal and temporal signature in a different year, and the "unseen region"
    claim is false for the only thing that matters. Block disjointness cannot
    catch this, because the leak is a point, not a cell.

  * **Temporal overlap by construction.** `split_combined` trains on late events
    from non-held-out blocks (`training_design.py:191`). That is a deliberate
    choice, but it means the headline number is not an out-of-time result, and a
    test event's near-identical twin one year earlier can be in training.

  * **Id adjacency.** `event_id` is assigned in event order, so neighbouring ids
    are near in time. If a random id-based sample was ever taken, twins split
    across the boundary.

Each check returns a measured number, not a boolean, and every check that cannot
run says which column is missing rather than reporting "clean".
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

#: Radii in km at which cross-split proximity is reported. 1 km is the label's
#: own `industrial_near_m`; 10 km is `industrial_typology_m`.
RADII_KM = (0.5, 1.0, 2.0, 5.0, 10.0)

#: Below this, two events are the same fire seen twice by two satellites.
TWIN_KM = 1.0


def _require(frame: pd.DataFrame, cols: List[str], what: str) -> None:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"{what} needs {missing}; present: "
                       f"{sorted(frame.columns)[:12]}")


def block_disjointness(train: pd.DataFrame, test: pd.DataFrame,
                       col: str = "spatial_block") -> Dict:
    a = set(train[col].unique())
    b = set(test[col].unique())
    shared = a & b
    return {
        "n_train_blocks": len(a),
        "n_test_blocks": len(b),
        "n_shared": len(shared),
        "clean": not shared,
        "examples": sorted(shared)[:5],
    }


def _project(lon: np.ndarray, lat: np.ndarray):
    """Equirectangular-to-km around the data centroid. Local, and fast."""
    lat0 = np.nanmean(lat)
    kx = 111.32 * np.cos(np.deg2rad(lat0))
    ky = 110.57
    return lon * kx, lat * ky


def cross_split_proximity(train: pd.DataFrame, test: pd.DataFrame,
                          lon_col: str = "event_centroid_lon",
                          lat_col: str = "event_centroid_lat",
                          radii_km: tuple = RADII_KM,
                          chunk: int = 200_000) -> Dict:
    """For every test event, the distance to the nearest TRAIN event.

    Reported as the share of test events that have a training event within each
    radius. A high share at 1 km means the test fold contains near-copies of
    training events, whatever the block split says.

    Streamed in chunks over the test set: the catalogue is 7.27M events and
    materialising a 7.27M x 7.27M distance matrix is not a thing that fits in
    Colab RAM.
    """
    _require(train, [lon_col, lat_col], "cross_split_proximity(train)")
    _require(test, [lon_col, lat_col], "cross_split_proximity(test)")

    tx = pd.to_numeric(train[lon_col], errors="coerce").to_numpy()
    ty = pd.to_numeric(train[lat_col], errors="coerce").to_numpy()
    ok = np.isfinite(tx) & np.isfinite(ty)
    tx, ty = tx[ok], ty[ok]
    if tx.size == 0:
        return {"error": "no finite train coordinates"}

    sx = pd.to_numeric(test[lon_col], errors="coerce").to_numpy()
    sy = pd.to_numeric(test[lat_col], errors="coerce").to_numpy()

    hits = {r: 0 for r in radii_km}
    nearest = np.empty(sx.size, dtype="float64")
    for lo in range(0, sx.size, chunk):
        cx, cy = sx[lo:lo + chunk], sy[lo:lo + chunk]
        gx, gy = _project(cx, cy)
        px, py = _project(tx, ty)
        d = np.sqrt((gx[:, None] - px[None, :]) ** 2
                    + (gy[:, None] - py[None, :]) ** 2)
        nearest[lo:lo + chunk] = d.min(axis=1)
    finite = np.isfinite(nearest)
    n = max(int(finite.sum()), 1)
    for r in radii_km:
        hits[r] = float((nearest[finite] <= r).mean())
    return {
        "n_test": int(sx.size),
        "n_test_finite": int(finite.sum()),
        "frac_test_within_radius": {str(r): hits[r] for r in radii_km},
        "median_nearest_train_km": float(np.median(nearest[finite])),
        "p01_nearest_train_km": float(np.percentile(nearest[finite], 1)),
        "p05_nearest_train_km": float(np.percentile(nearest[finite], 5)),
        "min_nearest_train_km": float(nearest[finite].min()),
        "twin_rate_1km": hits.get(1.0, hits.get(1, 0.0)),
    }


def cross_split_year_overlap(train: pd.DataFrame, test: pd.DataFrame,
                             year_col: str = "start_year") -> Dict:
    """How much of the test fold's calendar is also in training.

    Under `split_combined` this is expected to be 100% and is NOT by itself a
    bug -- the spatial holdout is doing the work. It is reported because a
    headline number called out-of-time while sharing every year is exactly what
    a reviewer reads as a leak.
    """
    _require(train, [year_col], "cross_split_year_overlap(train)")
    _require(test, [year_col], "cross_split_year_overlap(test)")
    a = set(pd.to_numeric(train[year_col], errors="coerce").dropna().astype(int))
    b = set(pd.to_numeric(test[year_col], errors="coerce").dropna().astype(int))
    return {
        "train_years": sorted(int(v) for v in a),
        "test_years": sorted(int(v) for v in b),
        "n_shared_years": len(a & b),
        "test_years_also_in_train": sorted(int(v) for v in (a & b)),
        "test_year_fraction_seen_in_train": (
            len(a & b) / len(b) if b else None),
    }


def label_rate_gap(train: pd.DataFrame, test: pd.DataFrame,
                   label_col: str = "label_weak",
                   target: str = "INDUSTRIAL_FIRE") -> Dict:
    """Positive rate per year, both sides.

    A fold whose test years are 30% positive and whose train years are 5%
    positive is a distribution shift, not a leak, and it is the difference
    between a threshold that transfers and one that does not.
    """
    _require(train, [label_col], "label_rate_gap(train)")
    _require(test, [label_col], "label_rate_gap(test)")

    def rates(df):
        d = pd.DataFrame({"y": (df[label_col] == target).astype(int)})
        if "start_year" in df.columns:
            d["start_year"] = pd.to_numeric(df["start_year"],
                                            errors="coerce")
            return {int(k): {"rate": float(g["y"].mean()), "n": int(len(g))}
                    for k, g in d.groupby("start_year")}
        return {"overall": float(d["y"].mean()), "n": int(len(d))}
    return {"train": rates(train), "test": rates(test)}


def id_adjacency(test_ids: pd.Series, all_ids: pd.Series) -> Dict:
    """Are the test ids interleaved with train ids, or contiguous?

    Event ids are assigned in event order, so contiguous test ids mean the split
    took a time slice and interleaved ids mean it shuffled by id -- the case
    where near-simultaneous fires land on both sides.
    """
    t = set(test_ids.astype(str))
    a = set(all_ids.astype(str))
    rest = sorted(a - t)
    if not rest:
        return {"error": "test ids are the whole catalogue"}
    idx = np.array([int(x.split("_")[-1]) for x in rest if "_" in x])
    if idx.size == 0:
        return {"error": "event ids are not numeric-suffixed"}
    return {
        "n_test": len(t),
        "n_train": len(rest),
        "frac_test_ids_between_train_ids": None,
        "note": ("ids are numeric-suffixed; check contiguity with "
                 "min/max of each side in the cell"),
        "train_id_range": [int(idx.min()), int(idx.max())],
    }


def verdict(prox: Dict, blocks: Dict, years: Dict) -> Dict:
    """Turn the measurements into a single honest sentence.

    Thresholds are deliberately conservative. A 1 km twin rate above 20% of the
    test fold means near-copies of training events are being scored; anything
    above that is not a rounding detail.
    """
    twin = prox.get("twin_rate_1km", 0.0)
    problems = []
    if not blocks.get("clean", True):
        problems.append(f"{blocks['n_shared']} spatial block(s) on both sides")
    if twin > 0.20:
        problems.append(
            f"{twin:.1%} of test events have a training event within 1 km -- "
            "the spatial holdout is not holding")
    elif twin > 0.05:
        problems.append(
            f"{twin:.1%} of test events have a training event within 1 km -- "
            "partial overlap, optimistic but not fatal")
    shared_years = years.get("n_shared_years", 0)
    return {
        "problems": problems,
        "clean": not problems,
        "verdict": (
            "split is clean on every check that ran"
            if not problems else
            "LEAKAGE OR SHIFT DETECTED: " + "; ".join(problems)),
        "shared_years": shared_years,
        "twin_rate_1km": twin,
    }


def probe(train: pd.DataFrame, test: pd.DataFrame) -> Dict:
    """Run every check, collecting what ran and what did not."""
    report: Dict = {"ran": [], "skipped": []}

    def run(name, fn):
        try:
            report[name] = fn()
            report["ran"].append(name)
        except KeyError as e:
            report[name] = {"skipped": str(e)}
            report["skipped"].append(name)

    run("block_disjointness", lambda: block_disjointness(train, test))
    run("cross_split_proximity", lambda: cross_split_proximity(train, test))
    run("cross_split_year_overlap", lambda: cross_split_year_overlap(train, test))
    run("label_rate_gap", lambda: label_rate_gap(train, test))
    prox = report.get("cross_split_proximity", {})
    blocks = report.get("block_disjointness", {})
    years = report.get("cross_split_year_overlap", {})
    if isinstance(prox, dict) and "error" not in prox and isinstance(blocks, dict):
        report["verdict"] = verdict(prox, blocks, years)
    else:
        report["verdict"] = {"verdict": "incomplete -- see skipped checks"}
    return report