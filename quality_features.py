"""
P3b -- QUALITY_UNCERTAINTY FEATURES (8 of 141).

Pure computation core for the QUALITY_UNCERTAINTY group, mirroring
`industrial_features.py`: no Drive paths and no I/O, so the Colab driver
`colab_p3b_quality.py` supplies the per-event detection records.

Status: NEW_IMPLEMENTATION. `extract_quality_features` exists in no historical
notebook, so nothing here is a recovery and nothing here may be reported as
`COMPUTED_HISTORICAL`.

What is and is not derivable from India_Firms
---------------------------------------------
The FIRMS detection records carry `confidence`, `scan`, `track`, `brightness`,
`bright_ti4`, `frp`, `daynight`, `satellite` and `instrument`. Seven of the
eight ledger features are computable from those alone. `CLOUD_FRACTION` is the
exception: FIRMS delivers no per-detection cloud field, so it needs the MOD14
QA bands over the event footprint and stays `MISSING_IMPLEMENTATION` here --
NaN, never 0.

One stated approximation
------------------------
FIRMS exposes the *scan* angle but not the *view zenith*. For a nadir-looking
VIIRS/ MODIS instrument they coincide in magnitude, so
VIEW_ZENITH_* is derived from the scan angle as view_zenith = |scan| (radians).
That is an approximation, not a measurement, and it is wrong for a genuinely
oblique viewing geometry. It is stated here, in the ledger note, and in the
driver QC output so no later reader mistakes it for a direct measurement.

Schema contract
---------------
`compute_quality_features` returns exactly the 8 keys, in ledger order, for
every call. Missing input becomes NaN, never 0 and never a dropped key:
"the detection record did not carry this field" is a different claim from
"the field was zero".
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Canonical output order -- must equal feature_ledger.FEATURE_GROUPS[QUALITY]
# ---------------------------------------------------------------------------

QUALITY_FEATURES: "list[str]" = [
    "FIRMS_CONFIDENCE",
    "PIXEL_SIZE",
    "SCAN_ANGLE_SIN",
    "SCAN_ANGLE_COS",
    "VIEW_ZENITH_SIN",
    "VIEW_ZENITH_COS",
    "CLOUD_FRACTION",
    "MISSING_FEATURE_FRACTION",
]

N_QUALITY = len(QUALITY_FEATURES)

#: Features the driver can fill from India_Firms alone.
DETECTION_DERIVED: "tuple[str, ...]" = (
    "FIRMS_CONFIDENCE",
    "PIXEL_SIZE",
    "SCAN_ANGLE_SIN",
    "SCAN_ANGLE_COS",
    "VIEW_ZENITH_SIN",
    "VIEW_ZENITH_COS",
)

#: The one feature that cannot come from FIRMS; needs MOD14 QA over the footprint.
GEE_ONLY_FEATURE = "CLOUD_FRACTION"

#: Needs the assembled 141-vector for the event, so it can only be filled last.
POST_ASSEMBLY_FEATURE = "MISSING_FEATURE_FRACTION"

#: FIRMS confidence labels, ranked. `np` is the FIRMS abbreviation for nominal.
CONFIDENCE_RANK: "dict[str, float]" = {
    "l": 1.0,
    "low": 1.0,
    "n": 2.0,
    "np": 2.0,
    "nominal": 2.0,
    "h": 3.0,
    "high": 3.0,
}

# ---------------------------------------------------------------------------
# VIIRS effective pixel footprint (NEW implementation -- stated model)
# ---------------------------------------------------------------------------
# VNP14 swath product: 375 m pixels at nadir growing towards the scan edge.
# This is the linear model FIRMS itself uses for its `pixel_size` field. It is
# only applied to VIIRS sensors; a MODIS record is scaled from its 1 km nadir
# footprint instead, and an unknown sensor yields NaN rather than a guess.
VIIRS_NADIR_PIXEL_M = 375.0
VIIRS_EDGE_PIXEL_M = 750.0
VIIRS_MAX_SCAN_ANGLE_DEG = 65.0

MODIS_NADIR_PIXEL_M = 1000.0
MODIS_EDGE_PIXEL_M = 2000.0
MODIS_MAX_SCAN_ANGLE_DEG = 45.0

DEG2RAD = np.pi / 180.0


def confidence_rank(value) -> "float":
    """Map one FIRMS confidence label onto 1..3, or NaN if unrecognised.

    Case-insensitive. Anything not in `CONFIDENCE_RANK` -- including a numeric
    code the source has not documented -- becomes NaN, because a wrong rank
    silently biases FIRMS_CONFIDENCE for every event that carries it.
    """
    if value is None:
        return float("nan")
    text = str(value).strip().lower()
    if text in CONFIDENCE_RANK:
        return CONFIDENCE_RANK[text]
    # Tolerate a column that was already expanded to 1/2/3.
    if text in ("1.0", "2.0", "3.0", "1", "2", "3"):
        return float(text)
    return float("nan")


def effective_pixel_size_m(scan_angle_deg, satellite=None) -> "float":
    """Effective ground pixel size in metres, from the scan angle.

    Linear in |scan angle|, 0 at nadir to the full edge footprint at the sensor's
    maximum scan angle. `satellite` selects the sensor profile:

        VIIRS (SNPP, NOAA-20, NOAA-21)   375 m -> 750 m over 65 deg
        MODIS (AQUA, TERRA)           1000 m -> 2000 m over 45 deg

    NaN when the scan angle is missing or unparseable, or when the sensor is
    neither VIIRS nor MODIS. An unknown sensor must not be silently treated as
    VIIRS: it would shrink a MODIS footprint by a factor of ~2.7.
    """
    if scan_angle_deg is None or (isinstance(scan_angle_deg, float) and np.isnan(scan_angle_deg)):
        return float("nan")
    try:
        angle = abs(float(scan_angle_deg))
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(angle):
        return float("nan")

    sat = "" if satellite is None else str(satellite).strip().upper()
    if "VIIRS" in sat or sat in ("SNPP", "NOAA-20", "NOAA-21", "NPP", "JPSS-1", "JPSS-2"):
        nadir, edge, max_deg = VIIRS_NADIR_PIXEL_M, VIIRS_EDGE_PIXEL_M, VIIRS_MAX_SCAN_ANGLE_DEG
    elif "MODIS" in sat or sat in ("AQUA", "TERRA"):
        nadir, edge, max_deg = MODIS_NADIR_PIXEL_M, MODIS_EDGE_PIXEL_M, MODIS_MAX_SCAN_ANGLE_DEG
    else:
        return float("nan")

    # Clamp rather than extrapolate: a reported angle beyond the sensor's
    # geometric limit is a data error, and extrapolating would invent a size.
    frac = min(angle, max_deg) / max_deg
    return float(nadir + frac * (edge - nadir))


def _as_float_array(values, n) -> "np.ndarray":
    if values is None:
        return np.full(n, np.nan, dtype="float64")
    arr = np.asarray(values, dtype="float64")
    if arr.shape != (n,):
        raise ValueError(f"expected shape ({n},); got {arr.shape}")
    return arr


def _mean_finite(values) -> "float":
    """Mean over the finite entries, or NaN when nothing is finite.

    Detection records are ragged: a field can be present on some rows of an
    event and missing on others. Averaging only the rows that carry the field is
    what the feature means; counting a missing row as 0 would drag every partial
    event toward "low confidence".
    """
    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def event_scan_terms(
    scan_angles_deg,
    satellites=None,
) -> "dict[str, float]":
    """Per-event SCAN_ANGLE_* and VIEW_ZENITH_* terms, plus PIXEL_SIZE.

    All four are averaged over the event's detections. Angles are handled as
    unit vectors (mean of sin, mean of cos) rather than as raw degrees, because
    a scan angle is signed and crosses the discontinuity at +/-180 deg, where a
    naive arithmetic mean of an event spanning the seam collapses toward zero.

    VIEW_ZENITH_* uses the stated approximation view_zenith = |scan angle|. It
    is NaN whenever the scan angle is, never a default of 0.
    """
    angles = _as_float_array(scan_angles_deg, len(scan_angles_deg or []))
    n = len(angles)
    nan = float("nan")
    if n == 0:
        return {
            "PIXEL_SIZE": nan,
            "SCAN_ANGLE_SIN": nan,
            "SCAN_ANGLE_COS": nan,
            "VIEW_ZENITH_SIN": nan,
            "VIEW_ZENITH_COS": nan,
        }

    finite = np.isfinite(angles)
    if not finite.any():
        return {
            "PIXEL_SIZE": nan,
            "SCAN_ANGLE_SIN": nan,
            "SCAN_ANGLE_COS": nan,
            "VIEW_ZENITH_SIN": nan,
            "VIEW_ZENITH_COS": nan,
        }

    good = angles[finite]

    if satellites is None:
        sizes = np.full(good.shape, np.nan, dtype="float64")
    else:
        sats = np.asarray(satellites, dtype=object)[finite]
        sizes = np.array(
            [effective_pixel_size_m(a, s) for a, s in zip(good, sats)],
            dtype="float64",
        )

    sin_mean = float(np.mean(np.sin(good * DEG2RAD)))
    cos_mean = float(np.mean(np.cos(good * DEG2RAD)))
    zenith = np.abs(good)

    return {
        "PIXEL_SIZE": _mean_finite(sizes),
        "SCAN_ANGLE_SIN": sin_mean,
        "SCAN_ANGLE_COS": cos_mean,
        "VIEW_ZENITH_SIN": float(np.mean(np.sin(zenith * DEG2RAD))),
        "VIEW_ZENITH_COS": float(np.mean(np.cos(zenith * DEG2RAD))),
    }


def event_confidence_mean(confidence_values) -> "float":
    """Mean FIRMS confidence rank (1 low / 2 nominal / 3 high) over detections.

    Accepts either the raw label column or an already-ranked numeric column.
    NaN when the event has no recognisable confidence value.
    """
    values = list(confidence_values or [])
    if not values:
        return float("nan")
    return _mean_finite([confidence_rank(v) for v in values])


def missing_feature_fraction(feature_row, feature_names=None) -> "float":
    """Fraction of the 141 ledger features that are NaN for one event.

    This is the event's total data-completeness score, so it can only be
    computed after every group has been assembled -- the driver calls it last.

    A key that is entirely absent is counted as missing, not skipped, otherwise
    a group that never ran would look complete. An empty feature row yields NaN
    rather than 1.0, because "nothing was computed" is not "everything failed".
    """
    if feature_row is None:
        return float("nan")
    if hasattr(feature_row, "to_dict") and not isinstance(feature_row, dict):
        feature_row = feature_row.to_dict()
    if not isinstance(feature_row, dict) or not feature_row:
        return float("nan")

    names = list(feature_names) if feature_names is not None else list(feature_row)
    if not names:
        return float("nan")

    missing = 0
    for name in names:
        if name not in feature_row:
            missing += 1
            continue
        value = feature_row[name]
        try:
            if value is None or not np.isfinite(float(value)):
                missing += 1
        except (TypeError, ValueError):
            missing += 1
    return missing / float(len(names))


def compute_quality_features(
    n_events,
    confidence_values=None,
    scan_angles_deg=None,
    satellites=None,
    cloud_fraction=None,
    missing_fraction=None,
) -> "dict[str, np.ndarray]":
    """Compute all 8 QUALITY_UNCERTAINTY features for one chunk of events.

    Parameters
    ----------
    n_events : int
        Number of events in this chunk. All returned arrays have this length.
    confidence_values : list of lists, or None. Per event, the `confidence`
        column of that event's detection records (raw labels or ranks).
    scan_angles_deg : list of lists, or None. Per event, the `scan` column in
        degrees. Used for the SCAN_ANGLE_*, VIEW_ZENITH_* and PIXEL_SIZE terms.
    satellites : list of lists, or None. Per event, the `satellite` column, used
        only to pick the pixel-size profile. When None, PIXEL_SIZE is NaN even
        where the scan angle is known -- the sensor is what makes 375 m valid.
    cloud_fraction : (n,) array, or None. MOD14 QA derived; None keeps
        CLOUD_FRACTION NaN, which is its status in the ledger.
    missing_fraction : (n,) array, or None. Output of `missing_feature_fraction`
        per event, available only after assembly.

    Returns
    -------
    dict with exactly `QUALITY_FEATURES` keys, in that order, each a float64
    array of length n_events.
    """
    n = int(n_events)
    if n < 0:
        raise ValueError("n_events must be non-negative")
    nan = np.full(n, np.nan, dtype="float64")
    out: "dict[str, np.ndarray]" = {}

    conf_lists = confidence_values or [None] * n
    scan_lists = scan_angles_deg or [None] * n
    sat_lists = satellites or [None] * n
    if not (len(conf_lists) == len(scan_lists) == len(sat_lists) == n):
        raise ValueError(
            "confidence_values, scan_angles_deg and satellites must each have "
            f"length n_events={n}"
        )

    conf = np.array(
        [event_confidence_mean(c) for c in conf_lists], dtype="float64"
    )
    out["FIRMS_CONFIDENCE"] = conf

    scan_terms = [event_scan_terms(s, sat) for s, sat in zip(scan_lists, sat_lists)]
    for key in (
        "PIXEL_SIZE",
        "SCAN_ANGLE_SIN",
        "SCAN_ANGLE_COS",
        "VIEW_ZENITH_SIN",
        "VIEW_ZENITH_COS",
    ):
        out[key] = np.array([t[key] for t in scan_terms], dtype="float64")

    cf = _as_float_array(cloud_fraction, n) if cloud_fraction is not None else nan.copy()
    out[GEE_ONLY_FEATURE] = cf

    mf = (
        _as_float_array(missing_fraction, n)
        if missing_fraction is not None
        else nan.copy()
    )
    out[POST_ASSEMBLY_FEATURE] = mf

    _assert_shape_and_order(out, n)
    return out


def _assert_shape_and_order(out, n) -> None:
    if list(out.keys()) != QUALITY_FEATURES:
        missing = [k for k in QUALITY_FEATURES if k not in out]
        extra = [k for k in out if k not in QUALITY_FEATURES]
        raise ValueError(
            f"quality feature set/order mismatch. missing={missing} extra={extra}"
        )
    for k, v in out.items():
        if len(v) != n:
            raise ValueError(f"{k} has length {len(v)}, expected {n}")
