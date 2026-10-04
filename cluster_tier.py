"""
P4a (tier 1) -- CLUSTER-TIER lineage features.

The recon established what `clusters/*.parquet` actually holds. It is
CLUSTER-level, not detection-level:

    date  local_cluster_id  start_time  end_time  n_detections
    centroid_x  centroid_y  total_frp  mean_frp  max_frp
    n_observation_times  duration_hours  satellites  n_satellites
    max_radius_m  max_radius_km  centroid_lon  centroid_lat

There is no per-detection `frp`, `i4`, `i5` or `confidence` there. Those live
in `India_Firms.parquet` and require the `mappings/` bridge
(`detection_index -> local_cluster_id`) to reach an event.

That splits the 54 lineage features in two:

    TIER 1 (this file, clusters/ + membership only)
        36 features. No join to the 15.4M-row detection table, so this is a
        minutes-long pass rather than an hours-long one.

    TIER 2 (lineage_features.py, needs India_Firms + mappings/)
        12 features: BT_I4/I5 medians and differences, subpixel temperature,
        source area, night/day FRP ratio, confidence, scan and view angles.

Doing tier 1 first is what makes the whole thing tractable: it produces two
thirds of the lineage block from 532 MB of cluster aggregates.

Statistical rules are identical to tier 2 and deliberately strict:

* A statistic a single cluster cannot define -- FRP slope, autocorrelation,
  spread speed, hull geometry -- is **NaN**, not 0. One cluster legitimately has
  no trend.
* A missing measurement is NaN, never 0.
* The output is exactly the tier-1 feature list, in ledger order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lineage_features import (
    LINEAGE_FEATURES,
    MIN_FOR_GEOMETRY,
    MIN_FOR_SPREAD,
    MIN_FOR_TREND,
    PERSISTENCE_WINDOW_DAYS,
    _convex_hull,
    _count_outside,
    _polygon_metrics,
    haversine_m,
    lag1_autocorrelation,
    ols_slope,
    resultant_length,
    safe_cv,
    safe_log1p,
    safe_ratio,
)

#: Features this tier can produce. Order follows the ledger.
CLUSTER_TIER_FEATURES: "list[str]" = [
    "FRP_MEDIAN", "FRP_P90", "FRP_MAX", "FRP_SUM_LOG", "FRP_IQR", "FRP_CV",
    "FRP_PEAK_MEDIAN_RATIO", "FRP_SLOPE", "FRP_DECAY_RATE",
    "DURATION_HOURS_LOG", "DETECTION_COUNT_LOG", "OVERPASS_DETECTION_RATIO",
    "DETECTION_DENSITY", "FRP_AUTOCORRELATION", "TIME_TO_PEAK",
    "FRP_PERSISTENCE", "PEAK_HOUR_SIN", "PEAK_HOUR_COS",
    "CENTROID_DISPLACEMENT_MEDIAN", "CENTROID_DISPLACEMENT_P90",
    "SPREAD_SPEED_MEDIAN", "SPREAD_SPEED_P90", "SPREAD_SPEED_MAX",
    "CENTROID_JITTER", "STATIONARITY_INDEX", "SPREAD_DIRECTION_SIN",
    "SPREAD_DIRECTION_COS", "DIRECTION_CONSISTENCY",
    "AREA_GROWTH_RATE", "PERIMETER_GROWTH_RATE", "ELONGATION",
    "COMPACTNESS", "SOLIDITY", "NEW_PIXEL_FRACTION",
]

N_CLUSTER_TIER = len(CLUSTER_TIER_FEATURES)

# Features deliberately NOT in tier 1, with the reason each one is excluded.
# Computed, not hand-listed, so a new lineage feature cannot silently fall
# through the cracks between the two tiers.
DEFERRED_REASONS: "dict[str, str]" = {
    # --- detection tier: reachable once India_Firms is joined in ---
    # (verified column names: brightness, daynight, confidence, scan)
    "BT_I4_MEDIAN": "detection tier: needs India_Firms `brightness`",
    "NIGHT_DAY_FRP_RATIO": "detection tier: needs India_Firms `daynight`",
    "FIRMS_CONFIDENCE": "detection tier: needs India_Firms `confidence`",
    "SCAN_ANGLE_SIN": "detection tier: needs India_Firms `scan`",
    "SCAN_ANGLE_COS": "detection tier: needs India_Firms `scan`",
    "VIEW_ZENITH_SIN": "detection tier: needs India_Firms `scan` (the VZA)",
    "VIEW_ZENITH_COS": "detection tier: needs India_Firms `scan` (the VZA)",
    "SPREAD_WIND_SPEED_RATIO": "needs an external wind field (era5, SUSPECT)",
    "CLOUD_FRACTION": "needs MODIS QA bits, not a FIRMS column",
    "MISSING_FEATURE_FRACTION": "needs every other group present first",
    "RECURRENCE_RATE_1Y": "needs a cross-event pass over all 7.27M events",
    "RECURRENCE_RATE_3Y": "needs a cross-event pass over all 7.27M events",
    "RECURRENCE_RATE_5Y": "needs a cross-event pass over all 7.27M events",
    "MAX_INTER_EVENT_GAP_LOG": "needs a cross-event pass over all 7.27M events",
}

# Features India_Firms physically cannot supply, imported rather than restated
# so the two registries cannot drift apart.
from lineage_features import NOT_AVAILABLE_FEATURES  # noqa: E402

DEFERRED_REASONS.update(NOT_AVAILABLE_FEATURES)

#: The lineage features that are neither tier 1 nor in DEFERRED_REASONS.
UNACCOUNTED: "list[str]" = [
    n
    for n in LINEAGE_FEATURES
    if n not in CLUSTER_TIER_FEATURES and n not in DEFERRED_REASONS
]

# Every one of the 54 must be either computed here or explained.
assert not UNACCOUNTED, f"lineage features with no tier: {UNACCOUNTED}"
assert set(CLUSTER_TIER_FEATURES) <= set(LINEAGE_FEATURES)
assert len(set(CLUSTER_TIER_FEATURES)) == N_CLUSTER_TIER
assert N_CLUSTER_TIER + len(DEFERRED_REASONS) == len(LINEAGE_FEATURES), (
    N_CLUSTER_TIER + len(DEFERRED_REASONS), len(LINEAGE_FEATURES)
)

REQUIRED_COLUMNS = (
    "event_id", "n_detections", "total_frp", "mean_frp", "max_frp",
)


def _series(df, name):
    return pd.to_numeric(df[name], errors="coerce")


def _f(value):
    """Coerce a per-event statistic to a plain float or NaN.

    `safe_ratio` and friends return numpy arrays -- for scalar inputs, 0-d
    arrays. Putting those straight into the per-event result dicts makes
    `pd.Series(dict)` infer dtype `object`, and pyarrow then refuses the column
    on write: "Can only convert 1-dimensional array values". An all-NaN column
    stays float64 and writes fine, so the failure appears only once a real value
    finally lands in it -- several batches into a long run.
    """
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"expected a scalar statistic, got shape {arr.shape}")
    out = float(arr.reshape(-1)[0])
    return out if np.isfinite(out) else np.nan


def _keep_index(values, like):
    """Re-wrap a numpy result as a Series carrying `like`'s index.

    `safe_log1p` and `safe_ratio` return ndarrays by design, which silently
    drops the groupby index. Assigning that back to a per-event frame then
    raises a length mismatch -- or worse, would align positionally and put
    every value in the wrong event if the lengths happened to match.
    """
    if isinstance(values, pd.Series):
        return values
    return pd.Series(np.asarray(values, dtype="float64"), index=like.index)


def compute_cluster_tier_features(clusters, event_col="event_id"):
    """Compute the 36 cluster-tier features for every event.

    Parameters
    ----------
    clusters : DataFrame, one row per (event, local cluster). Must carry the
        per-cluster aggregates from `clusters/` plus the `event_id` column
        joined in from membership on `local_cluster_id`.
    event_col : the event identifier column.

    Returns
    -------
    DataFrame indexed by event_id with exactly CLUSTER_TIER_FEATURES columns in
    ledger order.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in clusters.columns]
    if missing:
        raise KeyError(
            f"cluster tier requires {missing}; got {list(clusters.columns)}"
        )

    df = clusters
    key = df[event_col]
    out = pd.DataFrame(index=pd.Index(pd.unique(key), name=event_col))

    n_det = _series(df, "n_detections")
    total_frp = _series(df, "total_frp")
    max_frp = _series(df, "max_frp")

    # ---- FRP distribution over the event's clusters -----------------------
    grp_max = max_frp.groupby(key)
    out["FRP_MAX"] = grp_max.max()
    out["FRP_MEDIAN"] = grp_max.median()
    out["FRP_P90"] = grp_max.quantile(0.90)
    out["FRP_IQR"] = grp_max.quantile(0.75) - grp_max.quantile(0.25)
    out["FRP_SUM_LOG"] = _keep_index(
        safe_log1p(total_frp.groupby(key).sum(min_count=1)),
        total_frp.groupby(key).sum(min_count=1),
    )
    out["FRP_CV"] = grp_max.apply(
        lambda s: safe_cv(s.to_numpy()), include_groups=False
    )
    out["FRP_PEAK_MEDIAN_RATIO"] = _keep_index(
        safe_ratio(grp_max.max(), grp_max.median()), grp_max.max()
    )

    # ---- counts, duration, persistence ------------------------------------
    det_total = n_det.groupby(key).sum(min_count=1)
    out["DETECTION_COUNT_LOG"] = _keep_index(safe_log1p(det_total), det_total)

    time_col = next(
        (c for c in ("start_time", "acq_time", "date") if c in df.columns), None
    )
    if time_col is not None:
        times = pd.to_datetime(df[time_col], errors="coerce")
        span_h = (
            times.groupby(key).max() - times.groupby(key).min()
        ).dt.total_seconds() / 3600.0
        out["DURATION_HOURS_LOG"] = _keep_index(safe_log1p(span_h), span_h)

        days = times.dt.floor("D").groupby(key).nunique()
        out["OVERPASS_DETECTION_RATIO"] = _keep_index(
            safe_ratio(det_total, days), days
        )
        # PERSISTENCE over a FIXED trailing window, not over the event's own span.
        #
        # The original definition was `days / (span_days + 1)`, which is a
        # tautology: an event only exists while it is being detected, so it is
        # observed on every day of its span and the ratio is exactly 1.0 for all
        # 7,265,849 events. A constant column carries no information, and it
        # silently zeroed the AGRICULTURAL rule, whose condition
        # `FRP_PERSISTENCE < 0.5` could never be satisfied.
        #
        # Against a fixed window it discriminates: a two-day flare scores
        # 2/30, a month-long burn 30/30.
        span_days = (span_h / 24.0).reindex(out.index).to_numpy()
        observed_days = np.maximum(span_days + 1.0, 1.0)
        window_days = np.maximum(
            np.full(len(out), float(PERSISTENCE_WINDOW_DAYS)), observed_days
        )
        out["FRP_PERSISTENCE"] = _keep_index(
            safe_ratio(observed_days, window_days), days
        )

        hour = times.dt.hour + times.dt.minute / 60.0
        usable = hour.notna().to_numpy()
        theta = np.radians(2.0 * np.pi * hour.to_numpy() / 24.0)
        sin = pd.Series(np.sin(theta), index=df.index).where(usable)
        cos = pd.Series(np.cos(theta), index=df.index).where(usable)
        out["PEAK_HOUR_SIN"] = sin.groupby(key).mean()
        out["PEAK_HOUR_COS"] = cos.groupby(key).mean()
    else:
        out["DURATION_HOURS_LOG"] = np.nan
        out["OVERPASS_DETECTION_RATIO"] = np.nan
        out["FRP_PERSISTENCE"] = np.nan
        out["PEAK_HOUR_SIN"] = np.nan
        out["PEAK_HOUR_COS"] = np.nan

    # ---- geometry inputs ---------------------------------------------------
    cx = next((c for c in ("centroid_x", "centroid_lon") if c in df.columns), None)
    cy = next((c for c in ("centroid_y", "centroid_lat") if c in df.columns), None)
    radius_col = next(
        (c for c in ("max_radius_m", "max_radius_km") if c in df.columns), None
    )
    radius_m = None
    if radius_col is not None:
        radius_m = _series(df, radius_col)
        if radius_col.endswith("_km"):
            radius_m = radius_m * 1000.0

    if radius_m is not None:
        # Bounding-circle area, taken per event from its LARGEST cluster radius.
        # It must be grouped before use: a per-row area would broadcast against
        # a per-event detection count and produce one value per cluster.
        radius_per_event = radius_m.groupby(key).max()
        area_km2 = np.pi * (radius_per_event ** 2) / 1e6
        out["DETECTION_DENSITY"] = _keep_index(
            safe_ratio(det_total.reindex(radius_per_event.index), area_km2),
            radius_per_event,
        )
        rgrp = radius_m.groupby(key)
        # A single radius gives no elongation: a circle has none by construction.
        out["ELONGATION"] = _keep_index(
            np.where(
                rgrp.count() >= MIN_FOR_GEOMETRY,
                1.0 - safe_ratio(rgrp.min(), rgrp.max()),
                np.nan,
            ),
            radius_per_event,
        )
    else:
        out["DETECTION_DENSITY"] = np.nan
        out["ELONGATION"] = np.nan

    # ---- spatial track, per event -----------------------------------------
    spatial = _spatial_track(df, key, cx, cy, time_col)
    for name, values in spatial.items():
        out[name] = values

    # ---- per-event trends --------------------------------------------------
    slope, decay, autocorr, ttp = {}, {}, {}, {}
    hours = None
    if time_col is not None:
        hours = (
            pd.to_datetime(df[time_col], errors="coerce") - pd.Timestamp("2012-01-01")
        ).dt.total_seconds() / 3600.0

    for event_id, sub in df.groupby(event_col, sort=False):
        f = _series(sub, "max_frp").to_numpy()
        if hours is None or len(sub) < MIN_FOR_TREND:
            slope[event_id] = np.nan
            decay[event_id] = np.nan
        else:
            # Read the group's own time column: label-based .loc breaks on a
            # non-RangeIndex, which a merge produces.
            h = (
                pd.to_datetime(sub[time_col], errors="coerce")
                - pd.Timestamp("2012-01-01")
            ).dt.total_seconds().to_numpy() / 3600.0
            ok = np.isfinite(h) & np.isfinite(f)
            slope[event_id] = ols_slope(h[ok], f[ok])
            with np.errstate(divide="ignore", invalid="ignore"):
                logf = np.where(f > 0, np.log(f), np.nan)
            ok2 = np.isfinite(h) & np.isfinite(logf)
            decay[event_id] = ols_slope(h[ok2], logf[ok2])
        autocorr[event_id] = lag1_autocorrelation(f)
        if hours is not None and np.isfinite(f).any():
            h = (
                pd.to_datetime(sub[time_col], errors="coerce")
                - pd.Timestamp("2012-01-01")
            ).dt.total_seconds().to_numpy() / 3600.0
            ok = np.isfinite(h) & np.isfinite(f)
            if ok.any():
                peak = np.nanmax(f[ok])
                at_peak = h[ok][f[ok] >= peak]
                ttp[event_id] = float(np.nanmin(at_peak)) - float(np.nanmin(h[ok]))
            else:
                ttp[event_id] = np.nan
        else:
            ttp[event_id] = np.nan

    out["FRP_SLOPE"] = pd.Series({k: _f(v) for k, v in slope.items()})
    out["FRP_DECAY_RATE"] = pd.Series({k: _f(v) for k, v in decay.items()})
    out["FRP_AUTOCORRELATION"] = pd.Series(
        {k: _f(v) for k, v in autocorr.items()}
    )
    out["TIME_TO_PEAK"] = pd.Series({k: _f(v) for k, v in ttp.items()})

    for name in CLUSTER_TIER_FEATURES:
        if name not in out.columns:
            out[name] = np.nan

    return out.reindex(columns=CLUSTER_TIER_FEATURES)


def _spatial_track(df, key, cx, cy, time_col):
    """Per-event spatial statistics from the cluster-centroid track."""
    names = (
        "CENTROID_DISPLACEMENT_MEDIAN", "CENTROID_DISPLACEMENT_P90",
        "SPREAD_SPEED_MEDIAN", "SPREAD_SPEED_P90", "SPREAD_SPEED_MAX",
        "CENTROID_JITTER", "STATIONARITY_INDEX", "SPREAD_DIRECTION_SIN",
        "SPREAD_DIRECTION_COS", "DIRECTION_CONSISTENCY",
        "AREA_GROWTH_RATE", "PERIMETER_GROWTH_RATE", "COMPACTNESS",
        "SOLIDITY", "NEW_PIXEL_FRACTION",
    )
    tables = {n: {} for n in names}

    if cx is None or cy is None or time_col is None:
        index = pd.Index(pd.unique(key), name=key.name)
        return {n: pd.Series(np.nan, index=index) for n in names}

    for event_id, sub in df.groupby(key.name, sort=False):
        # Slice the group's own columns rather than indexing a pre-built array
        # by index label. Label-based positional indexing silently reads the
        # wrong rows whenever the frame's index is not a clean RangeIndex --
        # which is exactly what happens after a merge.
        p = np.column_stack(
            [_series(sub, cx).to_numpy(), _series(sub, cy).to_numpy()]
        )
        h = (
            pd.to_datetime(sub[time_col], errors="coerce")
            - pd.Timestamp("2012-01-01")
        ).dt.total_seconds().to_numpy() / 3600.0
        good = np.isfinite(p).all(axis=1) & np.isfinite(h)
        p, h = p[good], h[good]

        if p.shape[0] >= MIN_FOR_SPREAD:
            order = np.argsort(h, kind="stable")
            p, h = p[order], h[order]
            keep = np.concatenate([[0], np.where(np.diff(h) > 0)[0] + 1])
            p, h = p[keep], h[keep]

            steps = haversine_m(
                p[:-1, 0], p[:-1, 1], p[1:, 0], p[1:, 1]
            )
            dt = np.diff(h)
            ok = np.isfinite(steps) & np.isfinite(dt) & (dt > 0)
            speeds = steps[ok] / (dt[ok] / 3600.0)

            tables["SPREAD_SPEED_MEDIAN"][event_id] = _f(
                float(np.median(speeds)) if speeds.size else np.nan
            )
            tables["SPREAD_SPEED_P90"][event_id] = _f(
                float(np.percentile(speeds, 90)) if speeds.size else np.nan
            )
            tables["SPREAD_SPEED_MAX"][event_id] = _f(
                float(np.max(speeds)) if speeds.size else np.nan
            )

            net = haversine_m(p[0, 0], p[0, 1], p[-1, 0], p[-1, 1])
            path_len = float(np.nansum(steps))
            tables["CENTROID_DISPLACEMENT_MEDIAN"][event_id] = (
                net if np.isfinite(net) else np.nan
            )
            tables["CENTROID_DISPLACEMENT_P90"][event_id] = (
                float(np.percentile(steps, 90)) if steps.size else np.nan
            )
            tables["CENTROID_JITTER"][event_id] = (
                float(np.mean(steps)) if steps.size else np.nan
            )
            # Zero path length is the most stationary case, not an undefined one.
            tables["STATIONARITY_INDEX"][event_id] = (
                1.0 - (net / path_len) if path_len > 0 else 1.0
            )

            angles = np.arctan2(np.diff(p[:, 1]), np.diff(p[:, 0]))
            if angles.size:
                tables["SPREAD_DIRECTION_SIN"][event_id] = float(
                    np.sin(angles).mean()
                )
                tables["SPREAD_DIRECTION_COS"][event_id] = float(
                    np.cos(angles).mean()
                )
                tables["DIRECTION_CONSISTENCY"][event_id] = resultant_length(angles)
            else:
                for n in ("SPREAD_DIRECTION_SIN", "SPREAD_DIRECTION_COS",
                          "DIRECTION_CONSISTENCY"):
                    tables[n][event_id] = np.nan
        else:
            for n in ("SPREAD_SPEED_MEDIAN", "SPREAD_SPEED_P90",
                      "SPREAD_SPEED_MAX", "CENTROID_DISPLACEMENT_MEDIAN",
                      "CENTROID_DISPLACEMENT_P90", "CENTROID_JITTER",
                      "STATIONARITY_INDEX", "SPREAD_DIRECTION_SIN",
                      "SPREAD_DIRECTION_COS", "DIRECTION_CONSISTENCY"):
                tables[n][event_id] = np.nan
            if p.shape[0] == 1:
                # One point: the event did not move at all. That is the most
                # stationary case, not an undefined one. Speed stays NaN
                # because dividing by a zero interval is genuinely undefined.
                tables["STATIONARITY_INDEX"][event_id] = 1.0

        # Hull geometry needs at least MIN_FOR_GEOMETRY distinct points, and
        # a degenerate hull yields NaN rather than 0 for every shape metric.
        mid = max(2, p.shape[0] // 2)
        first_m = (
            _polygon_metrics(p[:mid]) if p.shape[0] >= MIN_FOR_GEOMETRY else None
        )
        last_m = (
            _polygon_metrics(p[mid:]) if p.shape[0] - mid >= 2 else None
        )
        usable = (
            first_m is not None and last_m is not None
            and not first_m.get("degenerate") and not last_m.get("degenerate")
        )
        if usable:
            tables["AREA_GROWTH_RATE"][event_id] = safe_ratio(
                last_m["area"] - first_m["area"], first_m["area"]
            )
            tables["PERIMETER_GROWTH_RATE"][event_id] = safe_ratio(
                last_m["perimeter"] - first_m["perimeter"],
                first_m["perimeter"],
            )
            tables["COMPACTNESS"][event_id] = last_m["compactness"]
            tables["SOLIDITY"][event_id] = last_m["solidity"]
            tables["NEW_PIXEL_FRACTION"][event_id] = safe_ratio(
                _count_outside(first_m, p[mid:]), max(1, p.shape[0] - mid)
            )
        else:
            for n in ("AREA_GROWTH_RATE", "PERIMETER_GROWTH_RATE",
                      "COMPACTNESS", "SOLIDITY", "NEW_PIXEL_FRACTION"):
                tables[n][event_id] = np.nan

    index = pd.Index(pd.unique(key), name=key.name)
    # Coerce every value to a plain float HERE, at Series construction.
    # `safe_ratio` returns a 0-d array for scalar inputs, and a dict of those
    # yields dtype `object`, which pyarrow refuses on write. An all-NaN column
    # stays float64 and writes fine, so the failure only appears once a real
    # value lands in it -- batch 36 of 90 on the first full run.
    return {
        n: pd.Series({k: _f(v) for k, v in table.items()}).reindex(index)
        for n, table in tables.items()
    }


def cluster_tier_report(features):
    """Coverage summary for a tier-1 result frame."""
    if features is None or len(features) == 0:
        return {"n_events": 0}
    out = {"n_events": int(len(features))}
    for name in CLUSTER_TIER_FEATURES:
        col = features[name]
        out[name] = float(1.0 - col.isna().mean())
    return out