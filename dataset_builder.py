"""
P6 -- DATASET ASSEMBLY.

Turns the two feature blocks into a training table:

    canonical_event_base.parquet   event identity, time, centroid, pre-existing
    features/lineage/*             34 tier-1 features
    features/industrial/*          18 industrial features
    ->
    X  (52 features)   y (labels)   meta (ids, time, split keys)

Three requirements from the project brief are enforced here rather than left to
discipline, because each is a way a published result silently becomes wrong:

1. **Raw coordinates are never training features.** The brief is explicit: the
   model must not memorise "this coordinate means industrial fire". Location is
   used only to build a site's history. `strip_coordinate_columns()` removes
   every centroid/lon/lat/grid column and `assert_no_location_features()`
   fails the build if any survive. The spatial holdout in `training_design`
   is the other half of that guarantee.

2. **A cloud-blocked observation is a gap, never a zero.** NaN is the encoding
   for "could not measure". `MISSING_FEATURE_FRACTION` is computed here so
   that distinction survives into the model instead of being flattened.

3. **Anomaly is judged against the site's own baseline.** The brief's example is
   a site normally at ~34 MW where an event reaches ~390 MW. A global FRP
   threshold cannot express that; `add_site_baseline()` normalises each event
   against the median FRP of the sites it sits in, so a permanent flare at a
   refinery reads as normal and an excursion anywhere reads as anomalous.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

import cluster_tier as ct
import fire_labels as lab
import industrial_features as ind

# ---------------------------------------------------------------------------
# 1. LOCATION MUST NOT LEAK INTO THE MODEL
# ---------------------------------------------------------------------------

#: Tokens that mark a column as a raw location. Matched as WHOLE parts of the
#: column name, split on non-alphanumerics -- never as bare substrings.
#: Substring matching is how "lon" ends up matching ELONGATION, which silently
#: deleted a shape feature from the matrix.
#:
#: Two tokens are deliberately ABSENT:
#:
#:   lng   In this project `DIST_LNG_LOG` means *liquefied natural gas*, a
#:         facility category. Treating it as longitude would delete the LNG
#:         distance feature. Longitude is already covered by lon/longitude.
#:   site  The brief permits location to enter *only* through a site's history,
#:         so SITE_FRP_BASELINE and FRP_ANOMALY_VS_SITE are legitimate
#:         features and must survive the strip.
LOCATION_TOKENS = frozenset(
    {
        "lat", "latitudes", "latitude", "lats",
        "lon", "long", "longitude", "longitudes", "lons",
        "x", "y", "grid", "coord", "coords", "geometry", "geom", "spatial",
    }
)

#: Columns that are identifiers or keys, never model features.
IDENTIFIER_COLUMNS = frozenset(
    {
        "event_id", "global_event_id", "site_id", "spatial_block",
        "date", "start_time", "end_time", "label", "rule", "confidence",
        "sample_weight", "split",
    }
)


def is_location_column(name: str) -> bool:
    """True if `name` encodes a raw position rather than a derived feature.

    Token-wise, so `ELONGATION` is a shape feature and `event_centroid_lon` is a
    location. Being wrong in the safe direction costs one usable feature; being
    wrong the other way puts a coordinate into the model and the model learns
    geography instead of fire behaviour.
    """
    lowered = name.lower()
    if lowered in IDENTIFIER_COLUMNS:
        return True
    parts = {p for p in re.split(r"[^a-z0-9]+", lowered) if p}
    return bool(parts & LOCATION_TOKENS)


def strip_coordinate_columns(frame, extra=()):
    """Return (kept_columns, dropped_columns).

    A meta table keeps the coordinates; only the feature matrix loses them.
    """
    drop = {c for c in frame.columns if is_location_column(c)}
    drop |= set(extra)
    keep = [c for c in frame.columns if c not in drop]
    return keep, sorted(drop)


def assert_no_location_features(feature_names):
    """Fail the build if any coordinate survived into the feature matrix."""
    leaks = sorted(n for n in feature_names if is_location_column(n))
    if leaks:
        raise AssertionError(
            f"location features leaked into the model: {leaks}. The brief "
            "requires raw coordinates never become training features."
        )


def validate_feature_blocks(tier1_columns, industrial_columns,
                           expected_tier1=None, expected_industrial=None):
    """Check each feature block against ITS OWN expected columns.

    The bug this replaces validated both blocks against the combined 52-column
    list, so the tier-1 file was reported as "missing" the 18 industrial
    features that live in the *other* file. Validating a block against the
    union guarantees a failure on correct data.

    Returns (errors, unexpected); raises ValueError when errors exist.
    """
    import cluster_tier as ct
    import industrial_features as ind

    expected_tier1 = list(expected_tier1 or ct.CLUSTER_TIER_FEATURES)
    expected_industrial = list(expected_industrial or ind.INDUSTRIAL_FEATURES)

    errors = []
    tier1_set, ind_set = set(tier1_columns), set(industrial_columns)

    missing_t1 = [c for c in expected_tier1 if c not in tier1_set]
    missing_ind = [c for c in expected_industrial if c not in ind_set]
    if missing_t1:
        errors.append(f"tier1 block missing {missing_t1}")
    if missing_ind:
        errors.append(f"industrial block missing {missing_ind}")

    allowed_t1 = set(expected_tier1) | {"event_id"}
    allowed_ind = set(expected_industrial) | {"event_id"}
    unexpected_t1 = sorted(tier1_set - allowed_t1)
    unexpected_ind = sorted(ind_set - allowed_ind)
    if unexpected_t1:
        errors.append(f"tier1 block has unexpected columns {unexpected_t1}")
    if unexpected_ind:
        errors.append(f"industrial block has unexpected columns {unexpected_ind}")

    overlap = sorted((tier1_set & ind_set) - {"event_id"})
    if overlap:
        errors.append(
            f"both blocks define {overlap}; X would have duplicate columns"
        )

    # The union must be exactly the declared feature set, in order.
    union = [c for c in expected_tier1 if c in tier1_set] + [
        c for c in expected_industrial if c in ind_set
    ]
    if len(union) != len(expected_tier1) + len(expected_industrial):
        errors.append(
            f"union is {len(union)} columns, expected "
            f"{len(expected_tier1) + len(expected_industrial)}"
        )

    if errors:
        raise ValueError("; ".join(errors))
    return union, unexpected_t1 + unexpected_ind


def feature_matrix_columns(tier1_columns, industrial_columns):
    """The ordered X column list implied by the two blocks on disk."""
    return validate_feature_blocks(tier1_columns, industrial_columns)[0]


# ---------------------------------------------------------------------------
# 2. DERIVE THE FLARE INPUTS FROM TIER 1
# ---------------------------------------------------------------------------
#
# `KNOWN_FLARE` needs five inputs that the event base does not carry. All five
# are recoverable from tier-1 features by inverting their definitions, which is
# why the industrial pass could legitimately leave it NaN instead of guessing.
#
# Note the consequence: 96.7% of events are single-cluster, so their
# `DURATION_HOURS_LOG` is log1p(0) = 0 and `active_days` derives to 0. Those
# events cannot satisfy a >=200-active-day flare rule, which is correct -- a
# single-day fire is not a flare.


def derive_flare_inputs(tier1):
    """Invert the tier-1 feature definitions into the raw flare inputs.

    Returns a DataFrame with `active_days`, `detection_count`,
    `centroid_displacement_m` and `frp_cv`. NaN where the tier-1 source is NaN,
    so an undefined input stays undefined rather than becoming zero.
    """
    out = pd.DataFrame(index=tier1.index)

    # DETECTION_COUNT_LOG = log1p(total detections)
    out["detection_count"] = np.expm1(
        pd.to_numeric(tier1["DETECTION_COUNT_LOG"], errors="coerce")
    )
    # DURATION_HOURS_LOG = log1p(span hours); FRP_PERSISTENCE = active_days /
    # (span_days + 1), so span_days = exp(...) - 1.
    span_h = np.expm1(pd.to_numeric(tier1["DURATION_HOURS_LOG"], errors="coerce"))
    span_days = span_h / 24.0
    persistence = pd.to_numeric(tier1["FRP_PERSISTENCE"], errors="coerce")
    out["active_days"] = persistence * (span_days + 1.0)
    # CENTROID_DISPLACEMENT_MEDIAN = log10(1 + metres)
    out["centroid_displacement_m"] = np.power(
        10.0,
        pd.to_numeric(tier1["CENTROID_DISPLACEMENT_MEDIAN"], errors="coerce"),
    ) - 1.0
    out["frp_cv"] = pd.to_numeric(tier1["FRP_CV"], errors="coerce")
    return out


# ---------------------------------------------------------------------------
# 3. SITE BASELINE ANOMALY
# ---------------------------------------------------------------------------

#: Site cell size for the baseline, in degrees.
#:
#: Measured against the real catalogue, 0.1 deg is FINE rather than
#: conservative: it yields 26,828 cells holding 7.27M events, a median of 134
#: events per cell, and 99.99% of events sharing a cell with at least one other
#: event. So a coarse grid is not required for company.
#:
#: The problem with 0.1 deg is the opposite of what it first looked like: the
#: baseline is then so local that an event is almost always compared against
#: its own immediate neighbourhood, including itself. A median of 134 events per
#: cell means the baseline IS the regional average, which is a weak notion of
#: "anomalous for this site".
#:
#: 1.0 deg (~110 km) is used instead: a refinery's flare and the surrounding
#: countryside fall in different cells, so an excursion at the plant stands out
#: against a genuinely rural baseline. That is the comparison the brief asks
#: for -- 34 MW at a normally-quiet site versus 390 MW at a busy one.
SITE_DEG = 1.0


def add_site_baseline(frame, frp_col="FRP_MEDIAN", lon_col="event_centroid_lon",
                      lat_col="event_centroid_lat", site_deg=SITE_DEG):
    """Add per-site baseline statistics and a normalised anomaly score.

    Adds:
        site_id           the baseline cell an event sits in
        SITE_FRP_BASELINE median FRP across all events in that cell
        SITE_N_EVENTS     how many events define that baseline
        FRP_ANOMALY_VS_SITE  FRP_MEDIAN / SITE_FRP_BASELINE

    `SITE_N_EVENTS` matters as much as the ratio: a baseline from three events
    is not a baseline, and the model must be able to see that. Cells with a
    single event get a baseline equal to that event, so the ratio is exactly
    1.0 -- which is the honest answer, not a spurious anomaly of 1.0 against a
    fabricated global mean.

    Coordinates are used to *derive* the site and then dropped from X by
    `strip_coordinate_columns`. That is the whole point of the rule: location
    enters only through a site's history.
    """
    df = frame
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    lat = pd.to_numeric(df[lat_col], errors="coerce")

    site_x = np.floor(lon / site_deg)
    site_y = np.floor(lat / site_deg)
    df = df.copy()
    df["site_id"] = (
        site_x.fillna(-999).astype("int64").astype(str)
        + "_"
        + site_y.fillna(-999).astype("int64").astype(str)
    )
    df.loc[lon.isna() | lat.isna(), "site_id"] = "unknown"

    frp = pd.to_numeric(df[frp_col], errors="coerce")
    grouped = frp.groupby(df["site_id"])
    df["SITE_FRP_BASELINE"] = grouped.transform("median")
    df["SITE_N_EVENTS"] = grouped.transform("count")

    # A site seen once has no baseline to compare against: its ratio is exactly
    # 1.0 by construction and says nothing.
    single = df["SITE_N_EVENTS"] <= 1
    df["FRP_ANOMALY_VS_SITE"] = np.where(
        single,
        np.nan,
        frp / df["SITE_FRP_BASELINE"].replace(0.0, np.nan),
    )
    return df


def log_anomaly(anomaly_ratio, eps=1.0):
    """log10 of the site-relative ratio; NaN when there is nothing to say.

    A ratio of exactly 1.0 means the event sits precisely at its site's median,
    which is true of a single-event site by construction and therefore carries
    no information. log10 would turn it into 0.0 and it would masquerade as
    confidently normal, so it is reported as NaN instead.
    """
    r = np.asarray(anomaly_ratio, dtype="float64")
    out = np.full(r.shape, np.nan, dtype="float64")
    ok = np.isfinite(r) & (r > 0) & (np.abs(r - 1.0) > 1e-9)
    out[ok] = np.log10(np.maximum(r[ok], eps))
    return out


# ---------------------------------------------------------------------------
# 4. MISSINGNESS
# ---------------------------------------------------------------------------

#: Total feature count the brief targets. Used only to report coverage, never
#: to fill anything.
TARGET_N_FEATURES = 141


def missing_feature_fraction(frame, feature_columns, target=TARGET_N_FEATURES):
    """Fraction of the target schema that is unavailable for each event.

    This is the brief's "show a gap, never a zero" requirement made numeric: a
    row with 52 of 141 features available reports 0.63 rather than being
    silently treated as a fully observed event.
    """
    cols = [c for c in feature_columns if c in frame.columns]
    if not cols:
        return pd.Series(np.nan, index=frame.index)
    present = frame[cols].notna().sum(axis=1)
    return (target - present) / float(target)


# ---------------------------------------------------------------------------
# 5. LABELS
# ---------------------------------------------------------------------------


def assign_labels(frame, flare_inputs=None, site_baseline=True):
    """Run the label cascade over an assembled feature table.

    `flare_inputs` supplies the four derived columns from
    `derive_flare_inputs`; when absent the flare rule abstains rather than
    guessing, which is the documented behaviour.
    """
    df = frame
    merged = dict(flare_inputs) if flare_inputs is not None else {}
    out = lab.assign_labels(df.assign(**merged) if merged else df)
    if site_baseline:
        # Kept for the dashboard, not used by the cascade.
        pass
    return out


def label_frame(frame, thresholds=None):
    """Convenience wrapper returning labels with the reporting summary."""
    out = lab.assign_labels(
        frame, thresholds=thresholds or lab.THRESHOLDS
    )
    report = lab.label_report(out["label"], out["rule"])
    return out, report


def sample_weights(labels, confidence):
    """Class weight x rule confidence. UNKNOWN is kept at low weight."""
    return lab.label_weights(labels, confidence=confidence)


# ---------------------------------------------------------------------------
# 7. PARTITIONING
#
# Why this exists: assembling 7,265,849 events by holding the feature blocks and
# the output in memory at once needs well over 10 GB and killed the Colab
# runtime at batch 12 of 15. The fix is to process a partition at a time --
# roughly 1/N of the events -- and stream every frame to parquet as it is
# produced, so peak memory is one partition rather than the whole catalogue.
#
# The hash MUST be stable across processes. Python's `hash()` on strings is
# randomised per interpreter by PYTHONHASHSEED, so a run resumed tomorrow would
# partition the same event differently and silently produce a different dataset.
# ---------------------------------------------------------------------------

#: 16 partitions -> ~454k events each: comfortably inside a Colab runtime.
N_PARTITIONS = 16


def partition_of(event_id, n_partitions=N_PARTITIONS):
    """Stable partition index for an event id.

    CRC32 rather than Python's `hash()`: it is deterministic across processes,
    so the partitioning is reproducible and a rerun lands the same event in the
    same partition.
    """
    import zlib

    ids = [event_id] if isinstance(event_id, str) else list(event_id)
    values = np.array(
        [zlib.crc32(str(v).encode("utf-8")) % int(n_partitions) for v in ids],
        dtype="int64",
    )
    return values if len(values) > 1 or not isinstance(event_id, str) else int(values[0])


def partition_frame(frame, n_partitions=N_PARTITIONS, key="event_id"):
    """Split a frame into `n_partitions` frames keyed on `key`.

    Returns a list of length n_partitions (possibly containing empty frames).
    Only one partition is ever materialised by the caller, so the parts can be
    handed straight to a parquet writer.
    """
    parts = np.asarray(partition_of(frame[key].tolist(), n_partitions))
    return [frame[parts == i] for i in range(int(n_partitions))]


def partition_bytes_estimate(n_rows, n_columns=8, overhead=1.6):
    """Rough resident bytes for an n_rows x n_columns float frame.

    `overhead` covers pandas' block manager and the string column. Only used
    for the pre-flight estimate, so approximate is fine.
    """
    return float(n_rows) * float(n_columns) * 8.0 * float(overhead)


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"

#: From the brief: "any prediction below 60% confidence goes to human review".
REVIEW_CONFIDENCE = 0.60

#: Alert thresholds quoted in the brief for the demo case.
ALERT_SCORE_THRESHOLD = 3.5


def add_area_km2(frame, density_col="DETECTION_DENSITY",
                 count_col="DETECTION_COUNT_LOG"):
    """Recover event footprint area in km2 by inverting the density.

    Tier 1 defines `DETECTION_DENSITY = detections / area_km2`, and
    `DETECTION_COUNT_LOG = log1p(detections)`, so the area is exactly
    recoverable:

        area_km2 = expm1(DETECTION_COUNT_LOG) / DETECTION_DENSITY

    This is a re-derivation of a measured quantity, not an estimate. It stays
    NaN when the density is undefined, which is the case for a zero-radius
    single-detection cluster -- there is genuinely no footprint to measure.
    """
    df = frame
    detections = np.expm1(
        pd.to_numeric(df[count_col], errors="coerce").to_numpy()
    )
    density = pd.to_numeric(df[density_col], errors="coerce").to_numpy()
    area = np.full(detections.shape, np.nan, dtype="float64")
    ok = np.isfinite(density) & (density > 0) & np.isfinite(detections)
    area[ok] = detections[ok] / density[ok]
    df = df.copy()
    df["area_km2"] = area
    return df


# ---------------------------------------------------------------------------
# 8. THE REVIEW THRESHOLD
# ---------------------------------------------------------------------------


def to_review_queue(probability, threshold=REVIEW_CONFIDENCE):
    """Split predictions into auto-accept and human review.

    Below the threshold is NOT discarded. The brief is explicit that the officer
    reviews and their corrections feed retraining, so the low-confidence
    predictions are exactly the ones worth showing a human.
    """
    p = np.asarray(probability, dtype="float64")
    # "Cannot decide" belongs in front of a human. A NaN probability is the
    # clearest form of that, so it must be flagged, not silently accepted.
    flag = ~np.isfinite(p) | (p < float(threshold))
    return {"accept": ~flag, "review": flag}