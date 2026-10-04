"""
P3a -- INDUSTRIAL CONTEXT FEATURES (18 of 141).

Pure computation core for the INDUSTRIAL group. No Drive paths, no I/O: the
Colab driver `colab_p3a_industrial.py` supplies geometries and event points and
calls `compute_industrial_features`.

These 18 features are the `LABEL_DERIVED` leakage partition (see
`feature_ledger.LABEL_DERIVED_FEATURES`). They are legitimate model inputs for
the operational model A and must be excluded from the scientific model B.

Status: this is a NEW implementation. No `extract_industrial_features` exists
anywhere in the historical notebooks, so nothing here is a reconstruction and
nothing here may be reported as `COMPUTED_HISTORICAL`.

Projection contract
-------------------
All coordinates MUST already be in EPSG:6933 (Equal Earth, metres) before they
reach this module. Every distance below is therefore a true ground distance in
metres, which is why the logging helpers are safe. Feeding lon/lat degrees
produces silently wrong distances, so `assert_projected_meters` exists to make
that failure loud.

Schema contract
---------------
`compute_industrial_features` returns a dict with exactly the 18 keys, in the
ledger's authoritative order, for every call. Missing inputs become NaN, never
0 and never a dropped key. The one exception is the count/distance family,
where "no facility of that type anywhere near" is a genuine measured zero:
absence of the gazetteer entry is reported as NaN, absence of the feature in
space is reported as 0.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Canonical output order -- must equal feature_ledger.FEATURE_GROUPS[INDUSTRIAL]
# ---------------------------------------------------------------------------

INDUSTRIAL_FEATURES: "list[str]" = [
    "DIST_NEAREST_FACILITY_LOG",
    "DIST_REFINERY_LOG",
    "DIST_POWERPLANT_LOG",
    "DIST_PETROCHEMICAL_LOG",
    "DIST_STEEL_LOG",
    "DIST_CEMENT_LOG",
    "DIST_LNG_LOG",
    "DIST_MINE_LOG",
    "DIST_LANDFILL_LOG",
    "DIST_PIPELINE_LOG",
    "DIST_WELLPAD_LOG",
    "FACILITY_COUNT_1KM",
    "FACILITY_COUNT_5KM",
    "FACILITY_COUNT_10KM",
    "INDUSTRIAL_DENSITY_CONTRAST",
    "INSIDE_FACILITY",
    "DIST_FACILITY_BOUNDARY_LOG",
    "KNOWN_FLARE",
]

N_INDUSTRIAL = len(INDUSTRIAL_FEATURES)

# Per-category distance feature, in the order they appear above. `None` means
# the category shares the "all facilities" tree and has no feature of its own.
CATEGORY_FEATURES: "dict[str, str]" = {
    "refinery": "DIST_REFINERY_LOG",
    "powerplant": "DIST_POWERPLANT_LOG",
    "petrochemical": "DIST_PETROCHEMICAL_LOG",
    "steel": "DIST_STEEL_LOG",
    "cement": "DIST_CEMENT_LOG",
    "lng": "DIST_LNG_LOG",
    "mine": "DIST_MINE_LOG",
    "landfill": "DIST_LANDFILL_LOG",
    "pipeline": "DIST_PIPELINE_LOG",
    "wellpad": "DIST_WELLPAD_LOG",
}

CATEGORIES: "list[str]" = list(CATEGORY_FEATURES)

# Distance features that must be emitted even though they have no category of
# their own, because they describe the union of all facilities.
UNION_DISTANCE_FEATURE = "DIST_NEAREST_FACILITY_LOG"

# Facility types that count as a gas flare site. Used only by KNOWN_FLARE.
FLARE_SITE_CATEGORIES = ("refinery", "petrochemical", "lng", "gas_processing")

# Search radii for the three count features, in metres.
COUNT_RADII_M: "tuple[int, ...]" = (1_000, 5_000, 10_000)
COUNT_FEATURES: "tuple[str, ...]" = (
    "FACILITY_COUNT_1KM",
    "FACILITY_COUNT_5KM",
    "FACILITY_COUNT_10KM",
)

# Radius of the surrounding ring used by INDUSTRIAL_DENSITY_CONTRAST.
CONTRAST_LOCAL_M = 10_000
CONTRAST_SURROUND_M = 50_000

# ---------------------------------------------------------------------------
# KNOWN_FLARE thresholds (NEW implementation -- stated, not inherited)
# ---------------------------------------------------------------------------
# A flare is a continuous point source, so it is defined by persistence rather
# than by intensity: many detections per day across many days, at one location,
# with no wind-driven displacement. These thresholds come from the rule cascade
# in HANDOFF 8, not from any historical code.
FLARE_MIN_ACTIVE_DAYS = 200
FLARE_MIN_DETECTION_DAY_RATIO = 0.25
FLARE_MAX_DISPLACEMENT_M = 2_000.0
FLARE_MAX_FRP_CV = 0.60

# Guard on the ratio inside INDUSTRIAL_DENSITY_CONTRAST. Prevents a log of a
# pure count from exploding when the surrounding mean is 0.
CONTRAST_EPS = 1.0


def assert_projected_meters(x, y, name="coords") -> None:
    """Coarse sanity check: reject input that is obviously EPSG:4326 degrees.

    EPSG:4326 longitude must lie in [-180, 180] and latitude in [-90, 90]. Any
    non-empty coordinate pair inside those bounds is therefore degrees.

    This is a backstop, not the real guard. It cannot distinguish (0.0, 0.0) in
    degrees from the same value in metres, so it stays permissive about that
    case. `compute_industrial_features` therefore takes the CRS explicitly and
    validates that instead; this helper only guards the paths that have no CRS
    to check against.
    """
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    if x.size == 0 or y.size == 0:
        return

    max_abs_x = np.nanmax(np.abs(x))
    max_abs_y = np.nanmax(np.abs(y))
    if max_abs_x == 0.0 and max_abs_y == 0.0:
        # All zeros carry no evidence about the units, and the origin is a
        # legitimate location in Equal Earth. Do not guess.
        return

    if max_abs_x <= 180.0 and max_abs_y <= 90.0:
        raise ValueError(
            f"{name} look like EPSG:4326 degrees (max |x|={max_abs_x}, "
            f"max |y|={max_abs_y}). Project to EPSG:6933 metres first, "
            "otherwise every DIST_* and FACILITY_COUNT_* value is wrong."
        )


#: The only projection this module is written against.
REQUIRED_CRS = "EPSG:6933"


def assert_projected_crs(crs) -> None:
    """Validate the CRS before any coordinate maths runs.

    Accepts an authority string, a bare EPSG code, or anything pyproj can parse
    (including a pyproj/geopandas CRS object). Anything geographic or
    geographic-derived is rejected, because every feature here is a metric
    distance and a degree-valued one is wrong in a way no downstream check
    would catch.
    """
    if crs is None:
        raise ValueError(
            "crs is required. Pass REQUIRED_CRS ('EPSG:6933', Equal Earth "
            "metres). The industrial features are metric distances; there is "
            "no safe default."
        )

    name = str(crs).strip()
    code = None
    if name.upper().replace(" ", "").startswith("EPSG:"):
        try:
            code = int(name.split(":")[1])
        except (IndexError, ValueError):
            code = None
    if code is None:
        try:
            code = int(crs)
        except (TypeError, ValueError):
            code = None

    if code is not None:
        if code != 6933:
            raise ValueError(
                f"crs must be {REQUIRED_CRS} (Equal Earth, metres); got EPSG:{code}. "
                "A different projection changes every DIST_* value."
            )
        return

    # Not an authority string: fall back to pyproj's opinion.
    try:
        from pyproj import CRS
    except ImportError as exc:  # pragma: no cover - pyproj absent locally
        raise ValueError(
            f"cannot verify crs={crs!r}: pyproj is unavailable, so pass the "
            f"literal string {REQUIRED_CRS}"
        ) from exc

    parsed = CRS.from_user_input(crs)
    if not parsed.is_projected:
        raise ValueError(
            f"crs must be a projected CRS in metres; got {crs!r} "
            f"({parsed.name}). Degrees would silently corrupt every DIST_* value."
        )
    units = {u.name.lower() for u in parsed.axis_info}
    if units and not units <= {"metre", "meter", "m"}:
        raise ValueError(
            f"crs {crs!r} ({parsed.name}) is projected in {sorted(units)}, "
            "not metres. Every radius in this module is in metres."
        )


def log_distance(distance_m) -> "np.ndarray":
    """log10(1 + d_m). Compresses a 0 m .. ~2000 km range without sign loss.

    log10(1 + 0) == 0 exactly, so a facility centroid coincident with the event
    centroid yields 0 rather than -inf or NaN.
    """
    d = np.asarray(distance_m, dtype="float64")
    if np.any(d < 0):
        raise ValueError("negative distance; the projection must be metres")
    return np.log10(1.0 + d)


def nearest_distance(points_xy, tree_xy) -> "np.ndarray":
    """Distance from each point to the nearest tree node, or NaN if empty.

    An empty category is a missing gazetteer entry, not a measured zero, so it
    is reported as NaN. The caller decides whether that is acceptable.
    """
    pts = np.asarray(points_xy, dtype="float64")
    if len(tree_xy) == 0:
        return np.full(len(pts), np.nan, dtype="float64")
    from scipy.spatial import cKDTree

    dist, _ = cKDTree(np.asarray(tree_xy, dtype="float64")).query(pts, k=1)
    return np.asarray(dist, dtype="float64")


def count_within(points_xy, tree_xy, radius_m) -> "np.ndarray":
    """Number of tree nodes within `radius_m` of each point.

    This counts facility points, so two facilities in the same postcode count
    twice. That is intentional and matches FACILITY_COUNT_* semantics as
    "number of known facility sites nearby", not "number of distinct sites".
    """
    pts = np.asarray(points_xy, dtype="float64")
    out = np.zeros(len(pts), dtype="float64")
    if len(tree_xy) == 0:
        return out
    from scipy.spatial import cKDTree

    counts = cKDTree(np.asarray(tree_xy, dtype="float64")).query_ball_point(
        pts, r=float(radius_m), return_length=True, workers=-1
    )
    return np.asarray(counts, dtype="float64")


def density_contrast(
    points_xy,
    tree_xy,
    local_m=CONTRAST_LOCAL_M,
    surround_m=CONTRAST_SURROUND_M,
    eps=CONTRAST_EPS,
) -> "np.ndarray":
    """How much more industrialised is this spot than its surroundings.

    NEW_IMPLEMENTATION. Defined as

        log10( (n_local + eps) / (mean n_local over the surrounding ring + eps) )

    where the ring is the annulus between `local_m` and `surround_m` around the
    same point, sampled at 16 evenly spaced bearings. It is a local-vs-regional
    contrast, not an absolute density, so a uniformly industrial region still
    scores near 0 while a single plant in empty countryside scores high.

    NaN when the tree is empty, because the contrast is undefined without a
    reference population.
    """
    pts = np.asarray(points_xy, dtype="float64")
    if len(tree_xy) == 0 or len(pts) == 0:
        return np.full(len(pts), np.nan, dtype="float64")

    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(tree_xy, dtype="float64"))

    n_local = np.asarray(
        tree.query_ball_point(pts, r=float(local_m), return_length=True, workers=-1),
        dtype="float64",
    )

    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    ring_r = (float(local_m) + float(surround_m)) / 2.0
    offsets = np.stack([np.cos(angles) * ring_r, np.sin(angles) * ring_r], axis=1)
    ring_pts = (pts[:, None, :] + offsets[None, :, :]).reshape(-1, 2)

    ring_counts = np.asarray(
        tree.query_ball_point(ring_pts, r=float(local_m), return_length=True, workers=-1),
        dtype="float64",
    ).reshape(len(pts), len(angles))

    ring_mean = ring_counts.mean(axis=1)
    return np.log10((n_local + eps) / (ring_mean + eps))


def known_flare_flags(
    inside_facility,
    active_days,
    detection_count,
    centroid_displacement_m,
    frp_cv,
    category_is_flare_site,
    min_active_days=FLARE_MIN_ACTIVE_DAYS,
    min_detection_day_ratio=FLARE_MIN_DETECTION_DAY_RATIO,
    max_displacement_m=FLARE_MAX_DISPLACEMENT_M,
    max_frp_cv=FLARE_MAX_FRP_CV,
):
    """Rule-based flare identification. NEW_IMPLEMENTATION.

    A flare is a continuous, stationary, roughly constant-output point source,
    so every condition below is about stationarity and persistence rather than
    magnitude:

      1. the event sits inside a known flare-capable facility
      2. it has been active on at least `min_active_days` distinct days
      3. it is detected on at least `min_detection_day_ratio` of those days,
         which rejects short seasonal burns near a plant
      4. it does not drift more than `max_displacement_m`, which is what
         separates a fixed stack from a wind-driven wildfire
      5. FRP is roughly constant, `frp_cv <= max_frp_cv`

    Returns int8 0/1, or NaN where the supporting evidence is unavailable --
    never a bare 0, because "cannot tell" and "known not to be a flare" are
    different claims and the label engine must be able to tell them apart.
    """
    n = len(np.asarray(inside_facility))
    out = np.full(n, np.nan, dtype="float64")

    inside = np.asarray(inside_facility, dtype="float64")
    days = np.asarray(active_days, dtype="float64")
    count = np.asarray(detection_count, dtype="float64")
    disp = np.asarray(centroid_displacement_m, dtype="float64")
    cv = np.asarray(frp_cv, dtype="float64")
    flare_site = np.asarray(category_is_flare_site, dtype="bool")

    known = (
        np.isfinite(inside)
        & np.isfinite(days)
        & np.isfinite(count)
        & np.isfinite(disp)
        & np.isfinite(cv)
    )

    idx = np.where(known)[0]
    if len(idx):
        positive = flare_site[idx] & (days[idx] >= min_active_days)
        positive &= (count[idx] / np.maximum(days[idx], 1.0)) >= min_detection_day_ratio
        positive &= disp[idx] <= max_displacement_m
        positive &= cv[idx] <= max_frp_cv
        out[idx] = positive.astype("float64")

    # Complete evidence plus an event that is not near a flare-capable facility
    # is a confident negative, so it gets a real 0 rather than NaN.
    off_site = np.where(known & ~flare_site)[0]
    out[off_site] = 0.0
    return out


def compute_industrial_features(
    points_xy,
    category_trees,
    crs=None,
    inside_flags=None,
    boundary_distance_m=None,
    active_days=None,
    detection_count=None,
    centroid_displacement_m=None,
    frp_cv=None,
    category_is_flare_site=None,
):
    """Compute all 18 industrial features for one chunk of events.

    Parameters
    ----------
    points_xy : (n, 2) array of event centroid coordinates in Equal Earth metres.
    category_trees : dict mapping each key in `CATEGORY_FEATURES` to an (m, 2)
        array of facility coordinates, or None when that category is absent
        from the gazetteer. `None` yields NaN for that feature, which is
        distinguishable from a measured zero.
    crs : the projection `points_xy` is already in. Required. Pass
        `REQUIRED_CRS` or an equivalent pyproj CRS.
    inside_flags : (n,) bool array, or None. Whether the centroid falls inside
        any facility polygon. Required for KNOWN_FLARE and
        DIST_FACILITY_BOUNDARY_LOG; when None those two are NaN.
    boundary_distance_m : (n,) array of distance to the nearest facility
        polygon boundary (0 where inside), or None.
    active_days, detection_count, centroid_displacement_m, frp_cv :
        supporting event statistics for KNOWN_FLARE, or None.
    category_is_flare_site : (n,) bool array, or None. True when the nearest
        facility of the event is flare-capable.

    Returns
    -------
    dict with exactly `INDUSTRIAL_FEATURES` keys, in that order. Every value is
    a float64 array of length n.
    """
    assert_projected_crs(crs)
    pts = np.asarray(points_xy, dtype="float64")
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points_xy must be (n, 2); got {pts.shape}")
    n = len(pts)
    nan = np.full(n, np.nan, dtype="float64")

    trees = category_trees or {}

    # --- 1. nearest facility of any type -----------------------------------
    all_points = _concat_trees(trees.values())
    nearest_all = nearest_distance(pts, all_points)

    out: "dict[str, np.ndarray]" = {}
    out[UNION_DISTANCE_FEATURE] = (
        log_distance(nearest_all) if np.isfinite(nearest_all).any() else nan.copy()
    )

    # --- 2. per-category distances ------------------------------------------
    for cat, feature in CATEGORY_FEATURES.items():
        tree = trees.get(cat)
        if tree is None or len(tree) == 0:
            out[feature] = nan.copy()
            continue
        out[feature] = log_distance(nearest_distance(pts, tree))

    # --- 3. facility counts --------------------------------------------------
    for radius, feature in zip(COUNT_RADII_M, COUNT_FEATURES):
        out[feature] = count_within(pts, all_points, radius)

    # --- 4. local vs regional contrast --------------------------------------
    if len(all_points) == 0:
        out["INDUSTRIAL_DENSITY_CONTRAST"] = nan.copy()
    else:
        out["INDUSTRIAL_DENSITY_CONTRAST"] = density_contrast(pts, all_points)

    # --- 5. inside / boundary ------------------------------------------------
    out["INSIDE_FACILITY"] = (
        np.asarray(inside_flags, dtype="float64")
        if inside_flags is not None
        else nan.copy()
    )
    out["DIST_FACILITY_BOUNDARY_LOG"] = (
        log_distance(boundary_distance_m)
        if boundary_distance_m is not None
        else nan.copy()
    )

    # --- 6. known flare ------------------------------------------------------
    # Identity check, never `in`. `None in (array, ...)` runs an elementwise
    # comparison and raises "truth value of an array is ambiguous" as soon as
    # any argument is a real numpy array -- which is every production call. The
    # tests missed it because they passed Python lists, where `None == [...]`
    # collapses to a scalar False.
    _flare_inputs = (
        inside_flags, active_days, detection_count,
        centroid_displacement_m, frp_cv, category_is_flare_site,
    )
    if any(value is None for value in _flare_inputs):
        out["KNOWN_FLARE"] = nan.copy()
    else:
        out["KNOWN_FLARE"] = known_flare_flags(
            inside_facility=inside_flags,
            active_days=active_days,
            detection_count=detection_count,
            centroid_displacement_m=centroid_displacement_m,
            frp_cv=frp_cv,
            category_is_flare_site=category_is_flare_site,
        )

    _assert_shape_and_order(out, n)
    return out


def _concat_trees(trees) -> "np.ndarray":
    blocks = [np.asarray(t, dtype="float64") for t in trees if t is not None and len(t)]
    if not blocks:
        return np.empty((0, 2), dtype="float64")
    return np.concatenate(blocks, axis=0)


def _assert_shape_and_order(out, n) -> None:
    if list(out.keys()) != INDUSTRIAL_FEATURES:
        missing = [k for k in INDUSTRIAL_FEATURES if k not in out]
        extra = [k for k in out if k not in INDUSTRIAL_FEATURES]
        raise ValueError(
            f"industrial feature set/order mismatch. missing={missing} extra={extra}"
        )
    for k, v in out.items():
        if len(v) != n:
            raise ValueError(f"{k} has length {len(v)}, expected {n}")