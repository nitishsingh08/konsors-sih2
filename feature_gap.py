"""
PHASE 0 -- feature gap analysis.

Compares the 141-feature specification against what is actually in `X.parquet`
and sorts every feature into exactly one actionable bucket:

    PRESENT        in X, populated enough to use
    SPARSE         in X, but more than `sparse_threshold` NaN. Kept as-is:
                   LightGBM routes NaN natively, so a 96% NaN column is a weak
                   signal rather than a broken one. It is NOT imputed.
    MISSING        not in X at all -- the real gap

and every MISSING feature is tagged with how to obtain it:

    DERIVE            computable from data already on Drive. No download.
    EXTERNAL_RASTER   needs an Earth Engine / raster pass.
    EXTERNAL_TABLE    needs a registry or catalogue that is not on Drive yet.

The tag is derived from the ledger's own `DATASET` declaration rather than a
hand-written list, so it cannot drift from the specification.

Also reports the occupancy of every derived helper (how many events share each
site cell), because `FRP_ANOMALY_VS_SITE` came back 99.5% NaN and the cause was
the grid being far too fine: 7.27M events over ~2.75M land cells means the
median cell holds ONE event, so there is no baseline to compare against.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import feature_ledger as fl

#: Above this NaN share a column is reported as sparse.
SPARSE_THRESHOLD = 0.90

# ---------------------------------------------------------------------------
# HOW TO OBTAIN A MISSING FEATURE
# ---------------------------------------------------------------------------

#: Sources that are computable from what is already on Drive.
DERIVE_MARKERS = ("India_Firms", "DERIVED", "Global Energy Monitor",
                  "OpenStreetMap")

#: Sources that need a raster or cube pass.
RASTER_MARKERS = ("GEE", "ERA5", "CHIRPS", "SoilGrids", "SRTM", "NASADEM",
                  "Copernicus", "WorldCover", "MCD12Q1", "MOD13Q1",
                  "MCD64A1", "MOD14A1", "VNP14A1")

#: Sources that are registries or catalogues, not rasters.
TABLE_SOURCES = {
    "VIIRS Nightfire": "VIIRS Nightfire (EOG/CSM) + World Bank GGFR",
    "World Bank GGFR": "World Bank Global Gas Flaring Tracker",
    "GEM": "Global Energy Monitor sector trackers",
    "ISS-LIS": "ISS-LIS lightning climatology (NASA Earthdata)",
    "Sentinel-5P": "Sentinel-5P OFFL_L3 (must re-download: sentinel5p is 444 B)",
    "Sentinel-1": "Sentinel-1 GRD (GEE)",
    "Sentinel-2": "Sentinel-2 L2A (GEE)",
}


def obtain_tag(feature):
    """How to obtain `feature`: derive / external_raster / external_table.

    Derived from the ledger's DATASET string so the tag always matches the
    specification's own declaration of where a feature comes from.
    """
    dataset = (fl.DATASET.get(feature) or "").strip()
    lowered = dataset.lower()

    if lowered.startswith("derived"):
        return "DERIVE", dataset

    for marker in TABLE_SOURCES:
        if marker.lower() in lowered:
            return "EXTERNAL_TABLE", TABLE_SOURCES[marker]

    # Order matters: India_Firms means derivable, even though some of those
    # features are also GEE-derived in principle. The on-Drive table wins.
    if any(m in dataset for m in DERIVE_MARKERS):
        return "DERIVE", dataset
    if any(m in dataset for m in RASTER_MARKERS):
        return "EXTERNAL_RASTER", dataset
    return "EXTERNAL_TABLE", dataset or "(undeclared)"


# ---------------------------------------------------------------------------
# THE GAP REPORT
# ---------------------------------------------------------------------------


def gap_report(X, sparse_threshold=SPARSE_THRESHOLD):
    """Compare `X` against the 141-feature spec.

    Returns a DataFrame with one row per feature, `status` and `obtain`.
    """
    columns = set(X.columns)
    rows = []
    nan_fraction = {}

    for feature in fl.FEATURES_141:
        if feature in columns:
            series = X[feature]
            if pd.api.types.is_numeric_dtype(series):
                frac = float(series.isna().mean())
            else:
                frac = float(series.isna().mean())
            nan_fraction[feature] = frac
            status = "PRESENT" if frac <= sparse_threshold else "SPARSE"
            obtain, source = ("-", "-")
        else:
            status = "MISSING"
            obtain, source = obtain_tag(feature)
        rows.append(
            {
                "feature": feature,
                "group": fl.GROUP_OF.get(feature, "?"),
                "status": status,
                "nan_fraction": nan_fraction.get(feature, float("nan")),
                "obtain": obtain,
                "source": source,
                "phase": fl.STATUS_MAP.get(feature, ("", "", "", ""))[2]
                if isinstance(fl.STATUS_MAP.get(feature), tuple)
                else "",
            }
        )
    return pd.DataFrame(rows)


def summarise(report):
    """Counts per status and per obtain-route, for the headline."""
    out = {
        "n_spec": int(len(report)),
        "status": dict(Counter(report["status"])),
        "obtain_for_missing": dict(
            Counter(report.loc[report["status"] == "MISSING", "obtain"])
        ),
        "obtain_for_sparse": dict(
            Counter(report.loc[report["status"] == "SPARSE", "obtain"])
        ),
    }
    usable = report["status"] == "PRESENT"
    out["n_usable"] = int(usable.sum())
    out["n_sparse"] = int((report["status"] == "SPARSE").sum())
    out["n_missing"] = int((report["status"] == "MISSING").sum())
    out["coverage_frac"] = out["n_usable"] / max(out["n_spec"], 1)
    return out


def missing_by_route(report):
    """Missing features grouped by how to obtain them, for the work order."""
    missing = report[report["status"] == "MISSING"]
    grouped = defaultdict(list)
    for _, row in missing.iterrows():
        grouped[(row["obtain"], row["group"])].append(row["feature"])
    return grouped


# ---------------------------------------------------------------------------
# SITE OCCUPANCY -- why FRP_ANOMALY_VS_SITE was 99.5% NaN
# ---------------------------------------------------------------------------


def site_occupancy(lon, lat, site_deg=0.1):
    """Events per site cell, and how many sites would support a baseline.

    A per-site anomaly needs at least two events in the cell. With 7.27M events
    spread over roughly 2.75M land cells at 0.1 deg, the MEDIAN cell holds one
    event, so a 0.1 deg grid cannot support site baselines at all. This
    quantifies that instead of leaving it as a mystery NaN rate.
    """
    lon = pd.to_numeric(pd.Series(lon), errors="coerce")
    lat = pd.to_numeric(pd.Series(lat), errors="coerce")
    key = (
        np.floor(lon / site_deg).fillna(-999).astype("int64").astype(str)
        + "_"
        + np.floor(lat / site_deg).fillna(-999).astype("int64").astype(str)
    )
    counts = key.value_counts()
    out = {
        "site_deg": site_deg,
        "n_events": int(len(key)),
        "n_sites": int(len(counts)),
        "events_per_site_mean": float(counts.mean()),
        "events_per_site_median": float(counts.median()),
        "frac_events_alone": float(counts[counts == 1].sum() / max(len(key), 1)),
        "frac_events_in_sites_ge2": float(
            counts[counts >= 2].sum() / max(len(key), 1)
        ),
    }
    return out


def recommend_site_deg(lon, lat, candidates=(0.05, 0.1, 0.25, 0.5, 1.0)):
    """Pick the coarsest grid that still leaves a meaningful baseline.

    Recommends the smallest grid at which the majority of events share a cell
    with at least one other event, because a site baseline needs company.
    """
    rows = []
    for deg in candidates:
        occ = site_occupancy(lon, lat, deg)
        rows.append({"site_deg": deg, **occ})
    table = pd.DataFrame(rows)
    ok = table[table["frac_events_in_sites_ge2"] >= 0.5]
    chosen = float(ok["site_deg"].iloc[-1]) if len(ok) else float(
        table["site_deg"].iloc[-1]
    )
    return table, chosen


# ---------------------------------------------------------------------------
# SINGLE-DETECTION FALLBACKS
# ---------------------------------------------------------------------------
#
# 96.7% of events are single-cluster, so every trend and geometry feature is
# legitimately undefined for them. The fix is NOT to impute: it is to add an
# explicit flag so the model can condition on "this was one detection" instead
# of inferring it from a pattern of NaNs.


def add_single_detection_flags(frame, count_col="DETECTION_COUNT_LOG",
                               persistence_col="FRP_PERSISTENCE"):
    """Add explicit flags for the single-detection regime.

    `N_DETECTIONS_GE3` and `IS_SINGLE_DETECTION` let LightGBM split on the
    regime directly. Imputing the empty features instead would invent
    measurements, and a model cannot tell a fabricated value from a real one.
    """
    df = frame
    detections = np.expm1(
        pd.to_numeric(df[count_col], errors="coerce").to_numpy()
    )
    out = df.copy()
    out["N_DETECTIONS"] = detections
    out["IS_SINGLE_DETECTION"] = np.where(
        np.isfinite(detections), (detections <= 1).astype("float64"), np.nan
    )
    out["N_DETECTIONS_GE3"] = np.where(
        np.isfinite(detections), (detections >= 3).astype("float64"), np.nan
    )
    if persistence_col in df.columns:
        p = pd.to_numeric(df[persistence_col], errors="coerce").to_numpy()
        # Persistence of exactly 1.0 with one detection is the degenerate case:
        # an event exists only on the day it was seen, so "seen every day" is
        # vacuously true. Recorded so the model can discount it.
        out["PERSISTENCE_IS_VACUOUS"] = np.where(
            np.isfinite(p) & np.isfinite(detections) & (detections <= 1),
            1.0, 0.0,
        )
    return out