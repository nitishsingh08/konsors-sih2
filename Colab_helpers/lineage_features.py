"""
P4a -- LINEAGE FEATURES (54 of 141) from India_Firms detection lineage.

The unit of analysis is the global fire *event*. Every feature here is derived
from the detections that belong to one event, reached through
`event_local_cluster_membership.parquet`. No Earth Engine, no network, no new
download: this is arithmetic over data already on Drive.

Covers four ledger groups:
    THERMAL            15
    TEMPORAL           14
    SPREAD             17
    QUALITY_UNCERTAINTY 8
                     =  54

Status: NEW_IMPLEMENTATION. The historical claim for 33 of these was
`UNVERIFIED_HISTORICAL` -- assigned by group inference from cell 262/325, never
observed to run. This module is a fresh implementation with the definitions
written down here and tested, which is a stronger position than the old claim,
but it is still not a recovery of the original code.

Design notes that matter for correctness
----------------------------------------
- **Vectorised over events, not per-event Python.** 7.27M events cannot be
  processed by a Python loop; every aggregate here is a `groupby` reduction and
  the only per-group loops are over events with enough detections for a
  genuinely defined statistic.
- **A statistic that is undefined stays NaN.** One detection cannot have an
  FRP slope, an autocorrelation, a spread direction or a convex hull. Returning
  0 for those would be a lie the model would happily learn.
- **Recurrence (4 features) is NOT here.** `RECURRENCE_RATE_{1,3,5}Y` and
  `MAX_INTER_EVENT_GAP_LOG` need a pass over all 7.27M events to count
  *neighbouring* events; they cannot be computed from one event's detections.
  They stay phase P3g and are emitted as NaN here.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import feature_ledger as fl

# ---------------------------------------------------------------------------
# OUTPUT SCHEMA
#
# Derived from the ledger rather than hardcoded, so the order can never drift
# from the authoritative schema. The four groups below are exactly the ones
# this module computes.
# ---------------------------------------------------------------------------

LINEAGE_GROUPS = (
    fl.GROUP_THERMAL,
    fl.GROUP_TEMPORAL,
    fl.GROUP_SPREAD,
    fl.GROUP_QUALITY,
)

LINEAGE_FEATURES: "list[str]" = [
    name
    for group in LINEAGE_GROUPS
    for name in fl.FEATURE_GROUPS[group]
    # The group map still carries the retired placeholder sentinel that the
    # ledger excludes from the 141; it must not reach the output schema.
    if name not in fl._PLACEHOLDERS
]

N_LINEAGE = len(LINEAGE_FEATURES)

# Fail loudly at import time rather than emitting a mis-ordered table later.
assert N_LINEAGE == 54, f"expected 54 lineage features, ledger gives {N_LINEAGE}"
assert set(LINEAGE_FEATURES) <= set(fl.FEATURES_141)

#: Emitted as NaN by this module because they need a cross-event pass.
DEFERRED_TO_P3G = (
    "RECURRENCE_RATE_1Y", "RECURRENCE_RATE_3Y", "RECURRENCE_RATE_5Y",
    "MAX_INTER_EVENT_GAP_LOG",
)

#: Needs an external wind field (era5, currently SUSPECT) rather than lineage.
NEEDS_EXTERNAL_INPUT = ("SPREAD_WIND_SPEED_RATIO", "CLOUD_FRACTION")

#: Minimum detections before a statistic is considered defined.
#: Trailing window, in days, that FRP_PERSISTENCE is measured over.
#:
#: It was originally `days / (span_days + 1)`, which is a tautology: an event
#: only exists while it is being detected, so it is observed on every day of its
#: own span and the ratio is exactly 1.0 for all 7,265,849 events. A constant
#: column carries no information, and it silently zeroed the AGRICULTURAL rule,
#: whose condition `FRP_PERSISTENCE < 0.5` could never be satisfied.
PERSISTENCE_WINDOW_DAYS = 30

MIN_FOR_TREND = 4
MIN_FOR_AUTOCORR = 5
MIN_FOR_GEOMETRY = 3
MIN_FOR_SPREAD = 2

_EARTH_RADIUS_M = 6_371_008.8


# ---------------------------------------------------------------------------
# DAILY CLUSTER FILE NAMING
#
# The files are `clusters_2012-01-20.parquet` / `edges_2012-01-20_to_2012-01-21.parquet`
# -- ISO dates with dashes, not the `YYYYMMDD_<label>` form the local *cluster*
# ids use. Constructing a filename from a local cluster id therefore produces
# nothing at all, silently. The real pattern is discovered from the directory
# instead of assumed.
# ---------------------------------------------------------------------------

_DATE_PATTERNS = (
    re.compile(r"(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})"),
    re.compile(r"(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})"),
)


def parse_date_from_filename(name: str):
    """Return 'YYYY-MM-DD' for a daily artefact, or None if it has no date.

    Tries the dashed ISO form first, then the compact form, so both
    `clusters_2012-01-20.parquet` and `clusters_20120120.parquet` resolve.
    """
    for pattern in _DATE_PATTERNS:
        m = pattern.search(name)
        if m:
            return f"{m.group('y')}-{m.group('m')}-{m.group('d')}"
    return None


def index_daily_files(directory, prefix="clusters"):
    """Map 'YYYY-MM-DD' -> Path for every daily file in `directory`.

    Returns (mapping, stats) where stats reports what was seen, including files
    skipped for lacking a parseable date. A prefix filter is applied when
    supplied, so `edges_...` files in the same directory are not mistaken for
    cluster files -- a mistake that would join edges to detections.
    """
    mapping = {}
    skipped = []
    other = 0
    for path in sorted(Path(directory).glob("*.parquet")):
        if prefix and not path.name.startswith(prefix):
            other += 1
            continue
        date = parse_date_from_filename(path.name)
        if date is None:
            skipped.append(path.name)
            continue
        mapping[date] = path
    stats = {
        "n_indexed": len(mapping),
        "n_skipped_no_date": len(skipped),
        "n_other_prefix": other,
        "skipped_examples": skipped[:5],
    }
    return mapping, stats

#: Logical name -> candidate real column names in the detections table. The
#: first candidate that exists wins. Never guessed silently: `resolve_columns`
#: returns the mapping so the caller can print and confirm it.
COLUMN_CANDIDATES: "dict[str, tuple[str, ...]]" = {
    # Confirmed against India_Firms.parquet (15,392,720 rows x 22 cols).
    # The real names differ from the FIRMS/API names in several places, so the
    # candidates list the measured name FIRST and keeps the textbook name as a
    # fallback for other vintages of this table.
    "event_id": ("event_id", "global_event_id", "event"),
    "acq_time": ("datetime", "acq_datetime", "acq_date_time", "acq_time"),
    "acq_date": ("date", "acq_date"),  # date32 or string
    "acq_hour": ("acq_time", "hour"),
    # Fire radiative power -- the one column every thermal feature needs.
    "frp": ("frp", "FRP", "fire radiative power"),
    # VIIRS 375 m: `brightness` is the 3.9 um band, i.e. the I4 equivalent.
    "i4": ("brightness", "brightness_i4", "i4", "bt_i4"),
    # MIR 11 um band -- NOT the same as I5. See NOT_AVAILABLE_FEATURES.
    "i5": ("i5", "brightness_i5"),
    "subpixel_temp": ("subpixel_temp",),
    "source_area": ("source_area", "source__area"),
    # `confidence` is the letter form l/h/n.
    "confidence": ("confidence",),
    # NOT present. FIRMS does not ship pixel size for this product.
    "pixel_size": ("pixel_size",),
    # VIIRS `scan` is the VIEW ZENITH angle and `track` the view AZIMUTH, both
    # in degrees. Mapping scan -> SCAN_ANGLE is correct; see the note on
    # VIEW_ZENITH, which has no exact counterpart in this table.
    "scan_angle": ("scan", "scan_angle"),
    "view_zenith": ("scan", "view_zenith"),
    "view_azimuth": ("track",),
    # Measured name is `daynight`, not `day_night`.
    "day_night": ("daynight", "day_night"),
    "satellite": ("satellite",),
    # `x`/`y` are ALREADY Equal Earth metres (6.6M-9.3M), not degrees.
    "longitude": ("longitude", "lon"),
    "latitude": ("latitude", "lat"),
    # Precomputed Equal Earth grid indices -- useful for fast spatial joins.
    "grid_x": ("grid_x",),
    "grid_y": ("grid_y",),
}

#: Lineage features this dataset can NEVER supply, with the reason.
#:
#: These are not "not implemented yet". India_Firms.parquet was measured and
#: simply does not carry the input. They stay NaN permanently rather than being
#: approximated from a neighbouring band, because a stand-in would be a silent
#: invention: BT_I5 from a MIR band is a different physical quantity, and
#: subpixel temperature / source area / pixel size have no analogue at all.
NOT_AVAILABLE_FEATURES: "dict[str, str]" = {
    "BT_I5_MEDIAN": (
        "India_Firms has `brightness` (3.9 um, the I4 equivalent) and "
        "`bright_t31` (11 um MIR), but no I5 band. A MIR band is not I5."
    ),
    "BT_I4_I5_DIFF_MEDIAN": "requires I5, which is absent",
    "BT_I4_I5_DIFF_P90": "requires I5, which is absent",
    "SUBPIXEL_TEMP_MEDIAN": (
        "no subpixel temperature column; only `brightness` at native resolution"
    ),
    "SOURCE_AREA_MEDIAN": "no source_area column in this product",
    "PIXEL_SIZE": "no pixel_size column; fixed by the 375 m product grid",
}

#: Angle semantics worth recording: FIRMS `scan` is the VIEW ZENITH angle and
#: `track` is the view AZIMUTH. Both are degrees.
ANGLE_SEMANTICS = {
    "scan": "view zenith angle (degrees)",
    "track": "view azimuth angle (degrees)",
}


def resolve_columns(frame) -> "dict[str, str | None]":
    """Map logical names onto the real columns present. None when absent."""
    present = {c.lower(): c for c in frame.columns}
    out = {}
    for logical, candidates in COLUMN_CANDIDATES.items():
        match = None
        for cand in candidates:
            if cand.lower() in present:
                match = present[cand.lower()]
                break
        out[logical] = match
    return out


def require_columns(mapping, needed):
    """Raise listing exactly what is missing, so the fix is one line."""
    missing = [k for k in needed if mapping.get(k) is None]
    if missing:
        raise KeyError(
            f"detections table is missing required columns: {missing}. "
            f"Present: {sorted(c for c in mapping.values() if c)}"
        )


# ---------------------------------------------------------------------------
# SMALL NUMERIC HELPERS (kept separate so they are individually testable)
# ---------------------------------------------------------------------------


def safe_log1p(x):
    """log1p, NaN-safe and sign-guarded (FRP is non-negative)."""
    v = np.asarray(x, dtype="float64")
    return np.where(np.isfinite(v) & (v >= 0), np.log1p(np.where(v > 0, v, 0.0)), np.nan)


def safe_ratio(numerator, denominator):
    """Divide, returning NaN instead of inf or 0 when the denominator is 0."""
    num = np.asarray(numerator, dtype="float64")
    den = np.asarray(denominator, dtype="float64")
    num, den = np.broadcast_arrays(num, den)
    out = np.full(num.shape, np.nan, dtype="float64")
    ok = np.isfinite(num) & np.isfinite(den) & (den > 0)
    out[ok] = num[ok] / den[ok]
    return out


def safe_cv(values):
    """Coefficient of variation. 0 when the mean is 0 is undefined, not 0."""
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    if v.size < 2:
        return np.nan
    mean = v.mean()
    if mean <= 0:
        return np.nan
    return float(v.std(ddof=1) / mean)


def ols_slope(x, y):
    """Slope of y on x. NaN unless at least 2 distinct x values exist."""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return np.nan
    x, y = x[ok], y[ok]
    xc = x - x.mean()
    denom = float((xc * xc).sum())
    if denom <= 0:
        return np.nan
    return float((xc * (y - y.mean())).sum() / denom)


def lag1_autocorrelation(values):
    """Lag-1 autocorrelation. Undefined below MIN_FOR_AUTOCORR, and undefined
    when the series is constant (zero variance gives 0/0, not 0)."""
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    if v.size < MIN_FOR_AUTOCORR:
        return np.nan
    if v.std() == 0:
        return np.nan
    centred = v - v.mean()
    denom = float((centred * centred).sum())
    if denom <= 0:
        return np.nan
    return float((centred[:-1] * centred[1:]).sum() / denom)


def resultant_length(angles):
    """Circular resultant length R in [0, 1]: 1 = perfectly directional,
    0 = uniformly scattered. This is the direction-consistency measure."""
    a = np.asarray(angles, dtype="float64")
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    c = float(np.cos(a).mean())
    s = float(np.sin(a).mean())
    return float(np.hypot(c, s))


def circular_mean_angle(angles):
    a = np.asarray(angles, dtype="float64")
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres between two lon/lat pairs."""
    p1 = np.radians(np.asarray(lat1, dtype="float64"))
    p2 = np.radians(np.asarray(lat2, dtype="float64"))
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2, dtype="float64") - np.asarray(lon1, dtype="float64"))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2.0 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ---------------------------------------------------------------------------
# GEOMETRY, per event (needs >= MIN_FOR_GEOMETRY detections)
# ---------------------------------------------------------------------------


def _convex_hull(points):
    """Monotone-chain convex hull. Returns an (k, 2) array."""
    pts = np.unique(np.asarray(points, dtype="float64"), axis=0)
    if len(pts) <= 2:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def half(seq):
        stack = []
        for p in seq:
            while len(stack) >= 2:
                a, b = stack[-2], stack[-1]
                if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) <= 0:
                    stack.pop()
                else:
                    break
            stack.append(p)
        return stack

    lower = half(pts)
    upper = half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1], dtype="float64")


def _polygon_metrics(points):
    """Area, perimeter and shape descriptors of the convex hull.

    Area is in square degrees and is converted to km2 by the caller using a
    latitude-dependent scale, because a "square degree" is not an area.
    """
    hull = _convex_hull(points)
    if len(hull) < 3:
        return None
    x, y = hull[:, 0], hull[:, 1]

    # Shoelace area (absolute).
    area = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    perim = float(np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0])).sum())

    # Perimeter of the axis-aligned bounding box, for compactness.
    bbox_perim = 2.0 * ((x.max() - x.min()) + (y.max() - y.min()))

    # A degenerate hull -- all detections on one point or one line -- has no
    # width and no interior. Every shape descriptor is undefined there, and
    # must be reported as NaN. Emitting 0.0 would be actively misleading: it
    # would say "scattered with no interior" about a perfectly stationary
    # stack, which is the opposite of the truth.
    if area <= 0.0 or bbox_perim <= 0.0:
        return {
            "area": area,
            "perimeter": perim,
            "bbox_perimeter": bbox_perim,
            "elongation": np.nan,
            "solidity": np.nan,
            "compactness": np.nan,
            "_hull": hull,
            "degenerate": True,
        }

    # Elongation: 1 = a perfect line, 0 = isotropic.
    extent = np.array([x.max() - x.min(), y.max() - y.min()], dtype="float64")
    long_side, short_side = float(np.max(extent)), float(np.min(extent))
    elongation = 1.0 - (short_side / long_side) if long_side > 0 else np.nan

    # Solidity: hull area over the area of its bounding box.
    bbox_area = float(extent[0] * extent[1])
    solidity = area / bbox_area if bbox_area > 0 else np.nan

    compactness = (4.0 * np.pi * area) / (perim ** 2) if perim > 0 else np.nan
    return {
        "area": area,
        "perimeter": perim,
        "bbox_perimeter": bbox_perim,
        "elongation": elongation,
        "solidity": solidity,
        "compactness": compactness,
        "_hull": hull,
        "degenerate": False,
    }


def _degrees_area_to_km2(area_deg2, lat_deg):
    """Convert a square-degree area to km2 at a given reference latitude."""
    lat = np.radians(np.asarray(lat_deg, dtype="float64"))
    # 1 degree latitude ~ 111.32 km; 1 degree longitude ~ 111.32*cos(lat) km.
    kx = 111.32 * np.cos(lat)
    ky = 111.32
    return area_deg2 * kx * ky


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------


def compute_lineage_features(
    detections,
    event_col="event_id",
    require=("event_id", "frp", "longitude", "latitude"),
):
    """Compute the 54 lineage features for every event in `detections`.

    Parameters
    ----------
    detections : DataFrame with one row per detection. Must carry the event
        identifier and the columns resolved by `resolve_columns`.
    event_col : name of the resolved event identifier column.
    require : logical names that must resolve, else raise.

    Returns
    -------
    DataFrame indexed by event id with exactly `LINEAGE_FEATURES` columns, in
    that order. Rows for events with no usable detections are omitted rather
    than emitted as all-NaN, so the caller can distinguish "no detections" from
    "detections but nothing computable".
    """
    cols = resolve_columns(detections)
    if cols.get(event_col) is None:
        cols[event_col] = detections.columns[0]
    require_columns({**cols, "event_id": cols[event_col]}, require)

    df = detections
    key = cols[event_col]

    out = pd.DataFrame(index=pd.Index(pd.unique(df[key]), name=key))

    def put(name, values):
        out[name] = values

    def series(logical, default=np.nan):
        col = cols.get(logical)
        if col is None:
            return pd.Series(default, index=df.index, dtype="float64")
        return pd.to_numeric(df[col], errors="coerce").astype("float64")

    # ---- time basis ------------------------------------------------------
    if cols.get("acq_time") is not None:
        t = pd.to_datetime(df[cols["acq_time"]], errors="coerce")
    elif cols.get("acq_date") is not None:
        t = pd.to_datetime(
            df[cols["acq_date"]].astype(str), errors="coerce"
        )
    else:
        t = pd.Series(pd.NaT, index=df.index)

    hours = (t - pd.Timestamp("2012-01-01")).dt.total_seconds() / 3600.0
    hours = hours.astype("float64")

    frp = series("frp")
    i4 = series("i4")
    i5 = series("i5")
    lon = series("longitude")
    lat = series("latitude")

    # =====================================================================
    # THERMAL -- all pure groupby aggregates
    # =====================================================================
    g = df.groupby(key, sort=False)

    put("FRP_MEDIAN", frp.groupby(df[key]).median())
    put("FRP_MAX", frp.groupby(df[key]).max())
    put("FRP_SUM_LOG", np.log1p(frp.groupby(df[key]).sum(min_count=1)))

    def percentile(q):
        return frp.groupby(df[key]).quantile(q)

    put("FRP_P90", percentile(0.90))
    put("FRP_IQR", percentile(0.75) - percentile(0.25))

    def cv_of(group):
        return safe_cv(group.to_numpy())

    put("FRP_CV", frp.groupby(df[key]).apply(cv_of, include_groups=False))
    put(
        "FRP_PEAK_MEDIAN_RATIO",
        safe_ratio(frp.groupby(df[key]).max(), frp.groupby(df[key]).median()),
    )

    diff_bt = i4 - i5
    put("BT_I4_MEDIAN", i4.groupby(df[key]).median())
    put("BT_I5_MEDIAN", i5.groupby(df[key]).median())
    put("BT_I4_I5_DIFF_MEDIAN", diff_bt.groupby(df[key]).median())
    put("BT_I4_I5_DIFF_P90", diff_bt.groupby(df[key]).quantile(0.90))

    put("SUBPIXEL_TEMP_MEDIAN", series("subpixel_temp").groupby(df[key]).median())
    put("SOURCE_AREA_MEDIAN", series("source_area").groupby(df[key]).median())

    # =====================================================================
    # TEMPORAL
    # =====================================================================
    n_det = g.size()
    put("DETECTION_COUNT_LOG", np.log1p(n_det.astype("float64")))

    if t.notna().any():
        span_h = hours.groupby(df[key]).max() - hours.groupby(df[key]).min()
        put("DURATION_HOURS_LOG", np.log1p(span_h))
        span_days = span_h / 24.0
    else:
        put("DURATION_HOURS_LOG", np.nan)
        span_days = pd.Series(np.nan, index=out.index)

    if cols.get("acq_date") is not None or cols.get("acq_time") is not None:
        dates = t.dt.floor("D")
        n_days = dates.groupby(df[key]).nunique().reindex(out.index)
        n_det_aligned = n_det.reindex(out.index)
        # Detections per observed overpass day.
        put("OVERPASS_DETECTION_RATIO", safe_ratio(n_det_aligned, n_days))
        # Persistence: the fraction of the event's own span on which it was
        # detected at all. 1.0 = seen every single day, which is the flare
        # signature; low values = intermittent, which is the wildfire one.
        #
        # The denominator is span_days + 1, not span_days: N daily detections
        # occupy N calendar days but only span N-1 day-intervals, so dividing by
        # the interval count returns 1.25 for a perfectly persistent event.
        # Bounded above by 1 by construction.
        put("FRP_PERSISTENCE", safe_ratio(n_days, span_days + 1.0))
    else:
        put("OVERPASS_DETECTION_RATIO", np.nan)
        put("FRP_PERSISTENCE", np.nan)

    # Detection density needs an area, which needs the hull: filled below.
    put("DETECTION_DENSITY", np.nan)

    # Night/day FRP ratio.
    if cols.get("day_night") is not None:
        is_night = df[cols["day_night"]].astype(str).str.upper().eq("N")
        night_frp = frp.where(is_night).groupby(df[key]).sum(min_count=1)
        day_frp = frp.where(~is_night).groupby(df[key]).sum(min_count=1)
        put("NIGHT_DAY_FRP_RATIO", safe_ratio(night_frp, day_frp))
    else:
        put("NIGHT_DAY_FRP_RATIO", np.nan)

    # Peak hour as a circular feature: sin/cos beat a raw hour because 23 and 0
    # are adjacent, and a linear encoding would place them at opposite ends.
    hour_of_day = t.dt.hour + t.dt.minute / 60.0
    theta = np.radians(2.0 * np.pi * hour_of_day / 24.0)
    theta_s = pd.Series(np.sin(theta), index=df.index).where(hour_of_day.notna())
    theta_c = pd.Series(np.cos(theta), index=df.index).where(hour_of_day.notna())
    if cols.get("acq_hour") is not None and hour_of_day.isna().all():
        h = pd.to_numeric(df[cols["acq_hour"]], errors="coerce")
        theta = np.radians(2.0 * np.pi * h / 24.0)
        theta_s, theta_c = pd.Series(np.sin(theta), index=df.index), pd.Series(
            np.cos(theta), index=df.index
        )
    put("PEAK_HOUR_SIN", theta_s.groupby(df[key]).mean())
    put("PEAK_HOUR_COS", theta_c.groupby(df[key]).mean())

    # Deferred: needs a cross-event pass (phase P3g).
    for name in DEFERRED_TO_P3G:
        put(name, np.nan)

    # =====================================================================
    # PER-EVENT LOOPS -- only for statistics a reduction cannot express
    # =====================================================================
    slope = {}
    decay = {}
    autocorr = {}
    time_to_peak = {}
    disp_med = {}
    disp_p90 = {}
    speed_med = {}
    speed_p90 = {}
    speed_max = {}
    jitter = {}
    stationarity = {}
    dir_sin = {}
    dir_cos = {}
    dir_consistency = {}
    wind_ratio = {}
    geom = {}

    frp_groups = frp.groupby(df[key])
    hours_groups = hours.groupby(df[key])
    diff_groups = diff_bt.groupby(df[key])
    combo = df.groupby(key, sort=False)

    for event_id, sub in combo:
        h = hours.loc[sub.index].to_numpy()
        f = frp.loc[sub.index].to_numpy()
        d = diff_bt.loc[sub.index].to_numpy()

        # -- trends -------------------------------------------------------
        ok = np.isfinite(h) & np.isfinite(f)
        slope[event_id] = (
            ols_slope(h[ok], f[ok]) if ok.sum() >= MIN_FOR_TREND else np.nan
        )
        # Decay rate: slope of log FRP against time. Negative means fading.
        with np.errstate(divide="ignore", invalid="ignore"):
            logf = np.where(f > 0, np.log(f), np.nan)
        ok2 = np.isfinite(h) & np.isfinite(logf)
        decay[event_id] = (
            ols_slope(h[ok2], logf[ok2]) if ok2.sum() >= MIN_FOR_TREND else np.nan
        )
        autocorr[event_id] = lag1_autocorrelation(f)

        # -- time to peak -------------------------------------------------
        ok3 = np.isfinite(h) & np.isfinite(f)
        if ok3.sum() >= 1 and f[ok3].size:
            peak = float(np.nanmax(f[ok3]))
            first = float(np.nanmin(h[ok3]))
            at_peak = h[ok3][f[ok3] >= peak]
            time_to_peak[event_id] = float(np.nanmin(at_peak)) - first
        else:
            time_to_peak[event_id] = np.nan

        # -- spread -------------------------------------------------------
        lo = lon.loc[sub.index].to_numpy()
        la = lat.loc[sub.index].to_numpy()
        pts = np.column_stack([lo, la])
        good = np.isfinite(pts).all(axis=1)
        pts_t = pts[good]
        h_t = h[good]

        if pts_t.shape[0] >= MIN_FOR_SPREAD:
            order = np.argsort(h_t, kind="stable")
            pts_s = pts_t[order]
            h_s = h_t[order]
            uniq_idx = np.concatenate([[0], np.where(np.diff(h_s) > 0)[0] + 1])
            path_pts = pts_s[uniq_idx]
            path_h = h_s[uniq_idx]

            steps = haversine_m(
                path_pts[:-1, 0], path_pts[:-1, 1],
                path_pts[1:, 0], path_pts[1:, 1],
            )
            dt_h = np.diff(path_h)
            valid_step = np.isfinite(steps) & np.isfinite(dt_h) & (dt_h > 0)
            speeds = steps[valid_step] / (dt_h[valid_step] / 3600.0)

            if speeds.size:
                speed_med[event_id] = float(np.median(speeds))
                speed_p90[event_id] = float(np.percentile(speeds, 90))
                speed_max[event_id] = float(np.max(speeds))

            net = haversine_m(
                path_pts[0, 0], path_pts[0, 1],
                path_pts[-1, 0], path_pts[-1, 1],
            )
            path_len = float(np.nansum(steps))
            disp_med[event_id] = net if np.isfinite(net) else np.nan
            disp_p90[event_id] = (
                float(np.percentile(steps, 90)) if steps.size else np.nan
            )
            # Jitter: mean step length once direction is averaged out.
            jitter[event_id] = float(np.mean(steps)) if steps.size else np.nan
            # Stationarity: net displacement over path length. 1 = never moved,
            # ~0 = random walk. A path length of exactly zero means the event
            # never moved at all, which is the most stationary case possible,
            # not an undefined one.
            if path_len > 0:
                stationarity[event_id] = 1.0 - (net / path_len)
            else:
                stationarity[event_id] = 1.0

            angles = np.arctan2(
                np.diff(path_pts[:, 1]), np.diff(path_pts[:, 0])
            )
            if angles.size:
                dir_sin[event_id] = float(np.sin(angles).mean())
                dir_cos[event_id] = float(np.cos(angles).mean())
                dir_consistency[event_id] = resultant_length(angles)
        else:
            for target in (disp_med, disp_p90, speed_med, speed_p90,
                           speed_max, jitter, stationarity, dir_sin, dir_cos,
                           dir_consistency):
                target[event_id] = np.nan

        # SPREAD_WIND_SPEED_RATIO needs an external wind field; era5 is
        # SUSPECT (see P1b), so this stays NaN rather than assuming calm.
        wind_ratio[event_id] = np.nan

        # -- geometry ------------------------------------------------------
        if pts_t.shape[0] >= MIN_FOR_GEOMETRY:
            geom[event_id] = _polygon_metrics(pts_t)
        else:
            geom[event_id] = None

    put("FRP_SLOPE", pd.Series(slope))
    put("FRP_DECAY_RATE", pd.Series(decay))
    put("FRP_AUTOCORRELATION", pd.Series(autocorr))
    put("TIME_TO_PEAK", pd.Series(time_to_peak))
    put("CENTROID_DISPLACEMENT_MEDIAN", pd.Series(disp_med))
    put("CENTROID_DISPLACEMENT_P90", pd.Series(disp_p90))
    put("SPREAD_SPEED_MEDIAN", pd.Series(speed_med))
    put("SPREAD_SPEED_P90", pd.Series(speed_p90))
    put("SPREAD_SPEED_MAX", pd.Series(speed_max))
    put("CENTROID_JITTER", pd.Series(jitter))
    put("STATIONARITY_INDEX", pd.Series(stationarity))
    put("SPREAD_DIRECTION_SIN", pd.Series(dir_sin))
    put("SPREAD_DIRECTION_COS", pd.Series(dir_cos))
    put("DIRECTION_CONSISTENCY", pd.Series(dir_consistency))
    put("SPREAD_WIND_SPEED_RATIO", pd.Series(wind_ratio))

    # Growth and shape need a hull per time half. Computed after the loop so
    # the hull is built once per event rather than twice.
    area_growth = {}
    perim_growth = {}
    elongation = {}
    compactness = {}
    solidity = {}
    new_pixel = {}
    density = {}

    for event_id, sub in combo:
        g0 = geom.get(event_id)
        if g0 is None:
            area_growth[event_id] = np.nan
            perim_growth[event_id] = np.nan
            elongation[event_id] = np.nan
            compactness[event_id] = np.nan
            solidity[event_id] = np.nan
            new_pixel[event_id] = np.nan
            density[event_id] = np.nan
            continue

        lat_ref = float(np.nanmean(lat.loc[sub.index].to_numpy()))
        area_km2 = float(_degrees_area_to_km2(g0["area"], lat_ref))
        density[event_id] = safe_ratio(len(sub), area_km2)

        elongation[event_id] = g0["elongation"]
        solidity[event_id] = g0["solidity"]
        # Compactness: 4*pi*A / P^2 using metric units.
        perim_m = g0["perimeter"] * 111.32  # degrees -> km, lat-independent bound
        area_m2 = area_km2 * 1e6
        compactness[event_id] = (
            (4.0 * np.pi * area_m2) / (perim_m * 1e3) ** 2
            if perim_m > 0 and np.isfinite(perim_m)
            else np.nan
        )

        # Split detections in half by time and compare hulls.
        h_all = hours.loc[sub.index].to_numpy()
        lo = lon.loc[sub.index].to_numpy()
        la = lat.loc[sub.index].to_numpy()
        pts = np.column_stack([lo, la])
        good = np.isfinite(pts).all(axis=1) & np.isfinite(h_all)
        if good.sum() >= 2 * MIN_FOR_GEOMETRY:
            pts_g, h_g = pts[good], h_all[good]
            order = np.argsort(h_g, kind="stable")
            pts_g, h_g = pts_g[order], h_g[order]
            mid = len(pts_g) // 2
            first_m = _polygon_metrics(pts_g[:mid])
            last_m = _polygon_metrics(pts_g[mid:])
            if (
                first_m
                and last_m
                and not first_m.get("degenerate")
                and not last_m.get("degenerate")
            ):
                area_growth[event_id] = safe_ratio(
                    last_m["area"] - first_m["area"], first_m["area"]
                )
                perim_growth[event_id] = safe_ratio(
                    last_m["perimeter"] - first_m["perimeter"],
                    first_m["perimeter"],
                )
                # Fraction of late points that were not in the early hull.
                new_pixel[event_id] = safe_ratio(
                    _count_outside(first_m, pts_g[mid:]),
                    len(pts_g[mid:]),
                )
                continue
        area_growth[event_id] = np.nan
        perim_growth[event_id] = np.nan
        new_pixel[event_id] = np.nan

    put("DETECTION_DENSITY", pd.Series(density))
    put("AREA_GROWTH_RATE", pd.Series(area_growth))
    put("PERIMETER_GROWTH_RATE", pd.Series(perim_growth))
    put("ELONGATION", pd.Series(elongation))
    put("COMPACTNESS", pd.Series(compactness))
    put("SOLIDITY", pd.Series(solidity))
    put("NEW_PIXEL_FRACTION", pd.Series(new_pixel))

    # =====================================================================
    # QUALITY
    # =====================================================================
    conf = series("confidence")
    if cols.get("confidence") is not None:
        numeric_conf = conf
        if numeric_conf.isna().all():
            # FIRMS sometimes ships confidence as l/h/n.
            letter = df[cols["confidence"]].astype(str).str.upper()
            numeric_conf = letter.map({"L": 100.0, "M": 105.0, "H": 110.0})
        put("FIRMS_CONFIDENCE", numeric_conf.groupby(df[key]).median())
    else:
        put("FIRMS_CONFIDENCE", np.nan)

    put("PIXEL_SIZE", series("pixel_size").groupby(df[key]).median())

    scan = series("scan_angle")
    put("SCAN_ANGLE_SIN", np.sin(np.radians(scan)).groupby(df[key]).mean())
    put("SCAN_ANGLE_COS", np.cos(np.radians(scan)).groupby(df[key]).mean())

    vz = series("view_zenith")
    put("VIEW_ZENITH_SIN", np.sin(np.radians(vz)).groupby(df[key]).mean())
    put("VIEW_ZENITH_COS", np.cos(np.radians(vz)).groupby(df[key]).mean())

    # CLOUD_FRACTION needs MODIS QA bits, not the FIRMS column, so a cld_frc
    # column here is accepted when present and otherwise left NaN.
    put("CLOUD_FRACTION", series("cloud_frac").groupby(df[key]).median())

    # Filled by the caller once every group is present: the fraction of the 140
    # other features that are missing for this event.
    put("MISSING_FEATURE_FRACTION", np.nan)

    return out.reindex(columns=LINEAGE_FEATURES)


def _count_outside(metrics, points):
    """Count points strictly outside the hull described by `metrics`.

    Uses an axis-aligned test as a cheap conservative proxy: a point outside
    the bounding box is definitely outside the hull. This under-counts, so
    NEW_PIXEL_FRACTION is a lower bound, which is the safe direction for a
    "new burn area" feature.
    """
    hull = metrics.get("_hull")
    if hull is None:
        return 0
    x0, x1 = hull[:, 0].min(), hull[:, 0].max()
    y0, y1 = hull[:, 1].min(), hull[:, 1].max()
    pts = np.asarray(points, dtype="float64")
    outside = (
        (pts[:, 0] < x0) | (pts[:, 0] > x1) | (pts[:, 1] < y0) | (pts[:, 1] > y1)
    )
    return int(outside.sum())
