"""
FEATURE LEDGER -- authoritative 141-feature schema + implementation audit.

This module is the single source of truth for the fire-event feature set.
Every notebook (Colab production, local analysis) must import FEATURES_141
from here. Do not re-declare the list anywhere else.

Authoritative order was fixed from the recovered notebook cells 233 and 246
(identical, 141 names, 141 unique). The earlier "131 feature" figure in
handover documents was an arithmetic error and is retired.

Status vocabulary (7 values, mutually exclusive):

  COMPUTED_HISTORICAL      Historical code located, function complete.
  UNVERIFIED_HISTORICAL    Assigned to a historical cell by group inference.
                           Code believed present but NOT yet confirmed at
                           runtime. Cleared by P2 (10-event test).
  HISTORICAL_NAN           Historical code explicitly hardcodes NaN.
  MISSING_IMPLEMENTATION   No executable implementation exists anywhere.
  MISSING_INPUT            Implementation exists/planned but the required
                           dataset does not cover part of the study period.
  EXTRACTION_ERROR         Code exists but failed at runtime (filled at P2/P4).
  RETIRED                  Name kept for schema stability, deliberately dropped.

`runtime_verified` is an orthogonal flag (default False). It is only set True
by the P2/P4 verification passes, so the ledger never overclaims.
"""

from __future__ import annotations

# Bump on every structural change. The P1 Colab cell compares this against
# the version it expects and refuses to proceed on a stale Drive copy --
# a stale module fails deep inside later cells with confusing errors.
#
# 1.1.0 -> 1.2.0 at P3a: added S_NEW_IMPL ("NEW_IMPLEMENTATION") and moved the
# 18 INDUSTRIAL features onto it. This bump matters: the version guard is the
# only thing that catches a stale /content copy, and the vocabulary change was
# invisible to it while the version string stayed 1.1.0.
# 1.2.0 -> 1.3.0 at P4a: the 54 THERMAL/TEMPORAL/SPREAD/QUALITY features moved
# onto NEW_IMPLEMENTATION. Implemented fresh in lineage_features.py from the
# India_Firms detection lineage; the previous UNVERIFIED_HISTORICAL claim was
# group inference from cell 262/325, never observed to run.
LEDGER_VERSION = "1.3.0"

# ---------------------------------------------------------------------------
# 1. THE 141 FEATURES, IN AUTHORITATIVE ORDER
# ---------------------------------------------------------------------------

GROUP_THERMAL = "THERMAL"
GROUP_TEMPORAL = "TEMPORAL"
GROUP_SPREAD = "SPREAD"
GROUP_TERRAIN = "TERRAIN"
GROUP_LANDCOVER = "LANDCOVER_FUEL"
GROUP_INDUSTRIAL = "INDUSTRIAL"
GROUP_WEATHER = "WEATHER_FIRE_DANGER"
GROUP_SMOKE = "OPTICAL_SMOKE"
GROUP_SAR = "SAR_STRUCTURAL"
GROUP_ATMOSPHERIC = "ATMOSPHERIC_CHEMISTRY"
GROUP_QUALITY = "QUALITY_UNCERTAINTY"

# (group, [features in authoritative order within the group])
FEATURE_GROUPS: "dict[str, list[str]]" = {
    # 1. THERMAL -- 15
    GROUP_THERMAL: [
        "FRP_MEDIAN",
        "FRP_P90",
        "FRP_MAX",
        "FRP_SUM_LOG",
        "FRP_IQR",
        "FRP_CV",
        "FRP_PEAK_MEDIAN_RATIO",
        "FRP_SLOPE",
        "FRP_DECAY_RATE",
        "BT_I4_MEDIAN",
        "BT_I5_MEDIAN",
        "BT_I4_I5_DIFF_MEDIAN",
        "BT_I4_I5_DIFF_P90",
        "SUBPIXEL_TEMP_MEDIAN",
        "SOURCE_AREA_MEDIAN",
    ],
    # 2. TEMPORAL -- 14
    GROUP_TEMPORAL: [
        "DURATION_HOURS_LOG",
        "DETECTION_COUNT_LOG",
        "OVERPASS_DETECTION_RATIO",
        "DETECTION_DENSITY",
        "RECURRENCE_RATE_1Y",
        "RECURRENCE_RATE_3Y",
        "RECURRENCE_RATE_3Y_UNUSED_PLACEHOLDER",  # REMOVED BELOW
        "RECURRENCE_RATE_5Y",
        "MAX_INTER_EVENT_GAP_LOG",
        "FRP_AUTOCORRELATION",
        "TIME_TO_PEAK",
        "FRP_PERSISTENCE",
        "NIGHT_DAY_FRP_RATIO",
        "PEAK_HOUR_SIN",
        "PEAK_HOUR_COS",
    ],
    # 3. SPREAD / DYNAMICS -- 17
    GROUP_SPREAD: [
        "CENTROID_DISPLACEMENT_MEDIAN",
        "CENTROID_DISPLACEMENT_P90",
        "SPREAD_SPEED_MEDIAN",
        "SPREAD_SPEED_P90",
        "SPREAD_SPEED_MAX",
        "CENTROID_JITTER",
        "STATIONARITY_INDEX",
        "SPREAD_DIRECTION_SIN",
        "SPREAD_DIRECTION_COS",
        "DIRECTION_CONSISTENCY",
        "AREA_GROWTH_RATE",
        "PERIMETER_GROWTH_RATE",
        "ELONGATION",
        "COMPACTNESS",
        "SOLIDITY",
        "NEW_PIXEL_FRACTION",
        "SPREAD_WIND_SPEED_RATIO",
    ],
    # 4. TERRAIN -- 12
    GROUP_TERRAIN: [
        "ELEVATION_MEDIAN",
        "ELEVATION_IQR",
        "ELEVATION_RANGE",
        "RELATIVE_ELEVATION",
        "SLOPE_MEDIAN",
        "SLOPE_P90",
        "ASPECT_SIN",
        "ASPECT_COS",
        "TPI",
        "TRI",
        "CURVATURE",
        "RIDGE_VALLEY_POSITION",
    ],
    # 5. LAND COVER / FUEL -- 16
    GROUP_LANDCOVER: [
        "FOREST_FRAC_375M",
        "FOREST_FRAC_1KM",
        "FOREST_FRAC_10KM",
        "FOREST_LOCAL_REGIONAL_CONTRAST",
        "CROPLAND_FRAC_1KM",
        "CROPLAND_FRAC_10KM",
        "CROPLAND_LOCAL_REGIONAL_CONTRAST",
        "BUILTUP_FRAC_1KM",
        "BUILTUP_FRAC_10KM",
        "BARE_MINING_FRAC_1KM",
        "WATER_FRAC_1KM",
        "LANDCOVER_ENTROPY",
        "NDVI_MEDIAN",
        "NDVI_ANOMALY_Z",
        "NBR_CHANGE",
        "NDMI_ANOMALY_Z",
    ],
    # 6. INDUSTRIAL CONTEXT -- 18
    GROUP_INDUSTRIAL: [
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
    ],
    # 7. WEATHER / FIRE DANGER -- 18
    GROUP_WEATHER: [
        "TEMPERATURE_ANOMALY_Z",
        "RH_ANOMALY_Z",
        "VPD_ANOMALY_Z",
        "WIND_SPEED",
        "WIND_GUST",
        "WIND_DIRECTION_SIN",
        "WIND_DIRECTION_COS",
        "RAIN_24H_LOG",
        "RAIN_7D_LOG",
        "SOIL_MOISTURE_ANOMALY_Z",
        "FFMC",
        "DMC",
        "DC",
        "ISI",
        "BUI",
        "FWI",
        "FWI_ANOMALY_Z",
        "LIGHTNING_COUNT_24H",
    ],
    # 8. OPTICAL / SMOKE -- 7
    GROUP_SMOKE: [
        "SMOKE_PROBABILITY",
        "SMOKE_LENGTH_LOG",
        "SMOKE_DIRECTION_SIN",
        "SMOKE_DIRECTION_COS",
        "DARK_SMOKE_PROBABILITY",
        "LIGHT_SMOKE_PROBABILITY",
        "BURN_SCAR_GROWTH_RATE",
    ],
    # 9. SAR / STRUCTURAL -- 7
    GROUP_SAR: [
        "VV_CHANGE",
        "VH_CHANGE",
        "VV_VH_CHANGE",
        "COHERENCE_LOSS",
        "DNBR",
        "DNDVI",
        "STRUCTURAL_CHANGE_PROBABILITY",
    ],
    # 10. ATMOSPHERIC CHEMISTRY -- 9
    GROUP_ATMOSPHERIC: [
        "NO2_ANOMALY_Z",
        "SO2_ANOMALY_Z",
        "CO_ANOMALY_Z",
        "CH4_ANOMALY_Z",
        "HCHO_ANOMALY_Z",
        "AOD_ANOMALY_Z",
        "CO_NO2_RATIO_LOG",
        "SO2_NO2_RATIO_LOG",
        "CH4_CO_RATIO_LOG",
    ],
    # 11. QUALITY / UNCERTAINTY -- 8
    GROUP_QUALITY: [
        "FIRMS_CONFIDENCE",
        "PIXEL_SIZE",
        "SCAN_ANGLE_SIN",
        "SCAN_ANGLE_COS",
        "VIEW_ZENITH_SIN",
        "VIEW_ZENITH_COS",
        "CLOUD_FRACTION",
        "MISSING_FEATURE_FRACTION",
    ],
}

# Sentinel names that must never reach the final schema.
_PLACEHOLDERS = {"RECURRENCE_RATE_3Y_UNUSED_PLACEHOLDER"}

# ---------------------------------------------------------------------------
# 2. FLATTENED, AUTHORITATIVE ORDER
# ---------------------------------------------------------------------------

FEATURES_141: "list[str]" = []
GROUP_OF: "dict[str, str]" = {}
GROUP_INDEX: "dict[str, str]" = {}

for _group, _names in FEATURE_GROUPS.items():
    for _name in _names:
        if _name in _PLACEHOLDERS:
            continue
        if _name in GROUP_OF:
            raise ValueError(f"Duplicate feature name in schema: {_name}")
        FEATURES_141.append(_name)
        GROUP_OF[_name] = _group
        GROUP_INDEX[_name] = _group

FEATURES: "list[str]" = list(FEATURES_141)
N_FEATURES = len(FEATURES_141)

# Expected count per group (used by the self-check in build_audit).
EXPECTED_GROUP_SIZES: "dict[str, int]" = {
    GROUP_THERMAL: 15,
    GROUP_TEMPORAL: 14,
    GROUP_SPREAD: 17,
    GROUP_TERRAIN: 12,
    GROUP_LANDCOVER: 16,
    GROUP_INDUSTRIAL: 18,
    GROUP_WEATHER: 18,
    GROUP_SMOKE: 7,
    GROUP_SAR: 7,
    GROUP_ATMOSPHERIC: 9,
    GROUP_QUALITY: 8,
}

# Features that are label-derived and MUST be excluded from the
# scientifically clean model partition (see P6 leakage policy).
LABEL_DERIVED_FEATURES: "list[str]" = list(FEATURE_GROUPS[GROUP_INDUSTRIAL])

NON_LEAK_FEATURES: "list[str]" = [
    f for f in FEATURES_141 if f not in set(LABEL_DERIVED_FEATURES)
]

# ---------------------------------------------------------------------------
# 3. STATUS VOCABULARY
# ---------------------------------------------------------------------------

S_COMPUTED = "COMPUTED_HISTORICAL"
S_UNVERIFIED = "UNVERIFIED_HISTORICAL"
S_HIST_NAN = "HISTORICAL_NAN"
S_MISSING_IMPL = "MISSING_IMPLEMENTATION"
S_MISSING_INPUT = "MISSING_INPUT"
S_ERROR = "EXTRACTION_ERROR"
S_RETIRED = "RETIRED"

# An implementation exists in this repo but has no historical counterpart, so
# it must never be reported as COMPUTED_HISTORICAL. Added at P3a for the 18
# INDUSTRIAL features, which had no extractor anywhere in the notebooks.
S_NEW_IMPL = "NEW_IMPLEMENTATION"

VALID_STATUSES = (
    S_COMPUTED,
    S_UNVERIFIED,
    S_HIST_NAN,
    S_MISSING_IMPL,
    S_MISSING_INPUT,
    S_ERROR,
    S_RETIRED,
    S_NEW_IMPL,
)

# Statuses that mean "no code exists for this feature yet". A new implementation
# is deliberately NOT one of these, even though it still needs a runtime pass.
STILL_UNIMPLEMENTED = (S_MISSING_IMPL, S_MISSING_INPUT, S_ERROR)

# ---------------------------------------------------------------------------
# 4. IMPLEMENTATION PHASES (planning metadata, not status)
# ---------------------------------------------------------------------------

PHASE_P0 = "P0_recover_cell262"
PHASE_P2 = "P2"
PHASE_P3A = "P3a_industrial"
PHASE_P3B = "P3b_quality"
PHASE_P3C = "P3c_landcover"
PHASE_P3D = "P3d_fwi"
PHASE_P3E = "P3e_sar"
PHASE_P3F = "P3f_smoke_atmospheric"
PHASE_P3G = "P3g_recurrence"
PHASE_NONE = ""

CELL_262 = "cell_262_extract_fire_features"
CELL_263 = "cell_263_extract_terrain_features"
CELL_264 = "cell_264_extract_weather_features"
CELL_325 = "cell_325_extract_event_derived_features"
CELL_314 = "cell_314_extract_event_features_dispatcher"
CELL_260 = "cell_260_event_geometry"

# ---------------------------------------------------------------------------
# 5. ERA WINDOWS -- earliest/latest date the underlying dataset can exist
# ---------------------------------------------------------------------------

ERA_FULL = ("2012-01-20", "2026-08-31")

ERA = {
    "firsms_native": ERA_FULL,          # from India_Firms.parquet
    "modis": ("2000-02-24", "2026-08-31"),
    "viirs_snpp": ("2012-01-20", "2026-08-31"),
    "sentinel1": ("2014-05-03", "2026-08-31"),
    "sentinel2": ("2015-06-27", "2026-08-31"),
    "sentinel5p": ("2018-07-02", "2026-08-31"),
    "iss_lis": ("2017-03-01", "2023-11-16"),
    "smap": ("2015-04-03", "2026-08-31"),
    "era5_land": ("1950-01-01", "2026-08-31"),
    "gem_osm": ("static_current_version", "static_current_version"),
}

# ---------------------------------------------------------------------------
# 6. DATASET SOURCE PER FEATURE
# ---------------------------------------------------------------------------

DATASET_THERMAL = "MODIS/061/MOD14A1 + NASA/VIIRS/002/VNP14A1 + India_Firms.parquet"
DATASET_DETECTIONS = "India_Firms.parquet (detection-level, via lineage)"
DATASET_TERRAIN = "GEE SRTM 30m / NASADEM / Copernicus DEM GLO-30"
DATASET_LANDCOVER = "GEE ESA_WorldCover_10m + MODIS/061/MCD12Q1 + MODIS/061/MOD13Q1"
DATASET_INDUSTRIAL = "Global Energy Monitor + OpenStreetMap Overpass"
DATASET_WEATHER = "ERA5-Land (GEE) + CHIRPS + SoilGrids"
DATASET_LIGHTNING = "ISS-LIS (NASA Earthdata)"
DATASET_SMOKE = "Sentinel-5P OFFL_L3 AERONOMI + Sentinel-2 L2A"
DATASET_SAR = "Sentinel-1 GRD (+ Sentinel-2 for DNBR/DNDVI)"
DATASET_ATMOSPHERIC = "Sentinel-5P OFFL_L3 (NO2/SO2/CO/CH4/HCHO/AERONOMI)"

DATASET = {}
for _f in FEATURE_GROUPS[GROUP_THERMAL]:
    DATASET[_f] = DATASET_THERMAL
for _f in FEATURE_GROUPS[GROUP_TEMPORAL]:
    DATASET[_f] = DATASET_DETECTIONS
for _f in FEATURE_GROUPS[GROUP_SPREAD]:
    DATASET[_f] = DATASET_DETECTIONS
for _f in FEATURE_GROUPS[GROUP_TERRAIN]:
    DATASET[_f] = DATASET_TERRAIN
for _f in FEATURE_GROUPS[GROUP_LANDCOVER]:
    DATASET[_f] = DATASET_LANDCOVER
for _f in FEATURE_GROUPS[GROUP_INDUSTRIAL]:
    DATASET[_f] = DATASET_INDUSTRIAL
for _f in FEATURE_GROUPS[GROUP_WEATHER]:
    DATASET[_f] = DATASET_WEATHER
DATASET["LIGHTNING_COUNT_24H"] = DATASET_LIGHTNING
for _f in FEATURE_GROUPS[GROUP_SMOKE]:
    DATASET[_f] = DATASET_SMOKE
for _f in FEATURE_GROUPS[GROUP_SAR]:
    DATASET[_f] = DATASET_SAR
for _f in FEATURE_GROUPS[GROUP_ATMOSPHERIC]:
    DATASET[_f] = DATASET_ATMOSPHERIC
for _f in FEATURE_GROUPS[GROUP_QUALITY]:
    DATASET[_f] = DATASET_DETECTIONS
DATASET["CLOUD_FRACTION"] = "MODIS/061/MOD14A1 QA bits + MODIS/061/MCD12Q1"
DATASET["MISSING_FEATURE_FRACTION"] = "DERIVED (computed last from the other 140)"

# ---------------------------------------------------------------------------
# 6b. LOCAL INPUT HINTS
#
# Substrings searched for, case-insensitively, against the file and directory
# names of a local `external/` tree. Used by the P1b inventory cell to answer
# "which of the 141 features have their input data already on disk?".
#
# Tokens are deliberately specific. A bare "co" or "s1" would match
# "config"/"s1_helpers", so multi-gazetteer tokens are used instead.
# ---------------------------------------------------------------------------

LOCAL_INPUT_HINTS: "dict[str, list[str]]" = {
    DATASET_DETECTIONS: [
        "india_firms",
        "firms",
        "detection",
    ],
    DATASET_THERMAL: [
        "india_firms",
        "firms",
        "vnp14",
        "viirs",
        "mod14",
        "fire",
    ],
    DATASET_TERRAIN: [
        "srtm",
        "dem",
        "terrain",
        "elevation",
        "nasadem",
        "copdem",
        "slope",
    ],
    DATASET_LANDCOVER: [
        "worldcover",
        "mcd12q1",
        "modis",
        "landcover",
        "land_cover",
        "ndvi",
        "vegetation",
        "esa_",
    ],
    DATASET_INDUSTRIAL: [
        "gem",
        "global_energy_monitor",
        "global_energymonitor",
        "osm",
        "overpass",
        "industrial",
        "facility",
        "facilities",
        "refiner",
        "petrochem",
        "power_plant",
        "powerplant",
        "mine",
        "steel",
        "cement",
        "lng",
        "pipeline",
        "wellpad",
        "landfill",
        "flare",
        "infra",
    ],
    DATASET_WEATHER: [
        "era5",
        "era5_land",
        "era5land",
        "chirps",
        "soilgrids",
        "soil_grid",
        "imerg",
        "gpm",
        "precip",
        "weather",
        "climate",
        "meteorolog",
    ],
    DATASET_LIGHTNING: [
        "lis",
        "lightning",
        "wwlln",
        "glm",
        "fdsn",
    ],
    DATASET_SMOKE: [
        "s5p",
        "sentinel5",
        "sentinel_5",
        "sentinel-5",
        "aero",
        "aod",
        "smoke",
        "burn",
        "sentinel2",
        "sentinel_2",
        "sentinel-2",
        "s2",
        "l2a",
    ],
    DATASET_SAR: [
        "sentinel1",
        "sentinel_1",
        "sentinel-1",
        "grd",
        "slc",
        "sar",
        "vv_",
        "vh_",
    ],
    DATASET_ATMOSPHERIC: [
        "s5p",
        "sentinel5",
        "sentinel_5",
        "sentinel-5",
        "no2",
        "so2",
        "ch4",
        "hcho",
        "aod",
        "atmos",
        "offl_l3",
    ],
    "MODIS/061/MOD14A1 QA bits + MODIS/061/MCD12Q1": [
        "mod14",
        "mcd12q1",
        "modis",
        "qa",
        "cloud",
    ],
}

# Features whose value is derived from the ledger itself and therefore never
# need an external file.
DERIVED_NO_INPUT = {
    "MISSING_FEATURE_FRACTION",
    "INSIDE_FACILITY",
    "DIST_FACILITY_BOUNDARY_LOG",
    "INDUSTRIAL_DENSITY_CONTRAST",
}


def hints_for(feature: str) -> "list[str]":
    """Local-input search tokens for one feature."""
    if feature in DERIVED_NO_INPUT:
        return []
    return LOCAL_INPUT_HINTS.get(DATASET[feature], [])


# Features that can be computed with NO remote-sensing data at all, because
# they only need the detection table or the ledger itself.
NO_GEE_REQUIRED = [
    f
    for f in FEATURES_141
    if DATASET[f] == DATASET_DETECTIONS or f in DERIVED_NO_INPUT
]

# ---------------------------------------------------------------------------
# 6c. KNOWN EXTERNAL SOURCES (verified against the P1b inventory)
#
# Confirmed present under
# Firm_Clusters/full_2012_2026/external on 2026-10-02:
#
#   VIIRS_FOOTPRINTS/        12 files, 1.08 GB   yearly .geoparquet 2012-2026
#   era5_land/              175 files, 0.02 GB   monthly .nc
#   sentinel5p/             506 files, 0.00 GB   monthly .tif per variable
#   osm_facilities/          37 files, 0.28 GB   per-category .geoparquet
#   india_mopitt_gfas_output/177 files, 0.02 GB monthly parquet (GFAS wind)
#   DL_FIRE_M-C61_814501/     3 files, 0.09 GB   FIRMS archive + NRT + readme
#   gee_features/             1 file             batch_manifest.csv only
#
# folder name -> tokens that prove the source exists locally
# ---------------------------------------------------------------------------

KNOWN_EXTERNAL_SOURCES = {
    "VIIRS_FOOTPRINTS": ["viirs_footprints", "footprints"],
    "era5_land": ["era5_land", "era5land"],
    "sentinel5p": ["sentinel5p", "s5p"],
    "osm_facilities": ["osm_facilities", "facility_categories"],
    "india_mopitt_gfas_output": ["mopitt", "gfas"],
    "DL_FIRE_M-C61": ["dl_fire", "fire_archive", "fire_nrt"],
    "gee_features": ["gee_features"],
}

# Folder -> expected minimum plausible total bytes. Below this the data is
# probably empty / all-NaN / clipped to nothing and must not be trusted.
SIZE_SANITY_FLOOR_BYTES = {
    "era5_land": 200 * 1024 * 1024,       # 175 monthly India crops
    "sentinel5p": 200 * 1024 * 1024,      # 506 monthly rasters
    "india_mopitt_gfas_output": 50 * 1024 * 1024,
    "VIIRS_FOOTPRINTS": 500 * 1024 * 1024,
    "osm_facilities": 10 * 1024 * 1024,
}

# ---------------------------------------------------------------------------
# 7. STATUS / REFERENCE MAP (the forensic findings, encoded)
# ---------------------------------------------------------------------------

STATUS_MAP: "dict[str, tuple[str, str, str, str]]" = {
    # feature -> (status, impl_ref, phase, notes)
}


def _set(names, status, impl_ref, phase, notes=""):
    for n in names:
        STATUS_MAP[n] = (status, impl_ref, phase, notes)


# -- Fire FRP trio: cell 262 only partially recovered, currently returns None.
_set(
    ["FRP_MEDIAN", "FRP_P90", "FRP_MAX"],
    S_MISSING_IMPL,
    CELL_262,
    PHASE_P0,
    "Cell 262 partially recovered (MODIS branch only); function returns None. "
    "P0 is open.",
)

# -- Remaining thermal: expected in cell 262/302/314 inline FRP block, not
#    confirmed at runtime.
_set(
    [
        "FRP_SUM_LOG",
        "FRP_IQR",
        "FRP_CV",
        "FRP_PEAK_MEDIAN_RATIO",
        "FRP_SLOPE",
        "FRP_DECAY_RATE",
        "BT_I4_MEDIAN",
        "BT_I5_MEDIAN",
        "BT_I4_I5_DIFF_MEDIAN",
        "BT_I4_I5_DIFF_P90",
        "SUBPIXEL_TEMP_MEDIAN",
        "SOURCE_AREA_MEDIAN",
    ],
    S_UNVERIFIED,
    CELL_262,
    PHASE_P0,
    "Assigned by group inference from cell 262/302/314 inline FRP block. "
    "Confirm at P2.",
)

# -- Event-derived temporal/spread: cell 325 is the confirmed dispatcher
#    helper for cell 314.
_set(
    [
        "DURATION_HOURS_LOG",
        "DETECTION_COUNT_LOG",
        "OVERPASS_DETECTION_RATIO",
        "DETECTION_DENSITY",
        "FRP_AUTOCORRELATION",
        "TIME_TO_PEAK",
        "FRP_PERSISTENCE",
        "NIGHT_DAY_FRP_RATIO",
        "PEAK_HOUR_SIN",
        "PEAK_HOUR_COS",
        "CENTROID_DISPLACEMENT_MEDIAN",
        "CENTROID_DISPLACEMENT_P90",
        "SPREAD_SPEED_MEDIAN",
        "SPREAD_SPEED_P90",
        "SPREAD_SPEED_MAX",
        "CENTROID_JITTER",
        "STATIONARITY_INDEX",
        "SPREAD_DIRECTION_SIN",
        "SPREAD_DIRECTION_COS",
        "DIRECTION_CONSISTENCY",
        "SPREAD_WIND_SPEED_RATIO",
    ],
    S_UNVERIFIED,
    CELL_325,
    PHASE_P2,
    "Cell 325 is the compatible helper for cell 314 (2-arg signature). "
    "Confirm each key is actually emitted at P2.",
)

# -- Geometry-derived shape metrics: forensic search found schema entries only.
_set(
    [
        "AREA_GROWTH_RATE",
        "PERIMETER_GROWTH_RATE",
        "ELONGATION",
        "COMPACTNESS",
        "SOLIDITY",
        "NEW_PIXEL_FRACTION",
    ],
    S_MISSING_IMPL,
    PHASE_NONE,
    PHASE_P3A,
    "Schema-only. No computation code found. Geometry metrics need the "
    "detection-level geometry per event (convex hull / buffered footprint).",
)

# -- Recurrence: cell 325 has no all_events argument, so these fall to NaN.
_set(
    [
        "RECURRENCE_RATE_1Y",
        "RECURRENCE_RATE_3Y",
        "RECURRENCE_RATE_5Y",
        "MAX_INTER_EVENT_GAP_LOG",
    ],
    S_MISSING_IMPL,
    CELL_325,
    PHASE_P3G,
    "Cell 325 takes (event, event_clusters) only. Cell 372/387 need "
    "all_events and are signature-incompatible with cell 314. Needs a "
    "chunked self-join over all 7,265,849 events.",
)

# -- Terrain: cell 263 is the only historical implementation and is complete.
_set(
    FEATURE_GROUPS[GROUP_TERRAIN],
    S_COMPUTED,
    CELL_263,
    PHASE_NONE,
    "Only historical implementation; recovered intact.",
)

# -- Lineage block, implemented fresh at P4a in lineage_features.py from the
#    India_Firms detection lineage. This REPLACES an UNVERIFIED_HISTORICAL claim
#    that was only ever group inference from cell 262/325: those 33 were never
#    observed to run, so the definitions here are written down and unit-tested
#    rather than inherited. Still not runtime_verified.
_set(
    [
        f
        for f in (
            FEATURE_GROUPS[GROUP_THERMAL]
            + FEATURE_GROUPS[GROUP_TEMPORAL]
            + FEATURE_GROUPS[GROUP_SPREAD]
            + FEATURE_GROUPS[GROUP_QUALITY]
        )
        if f not in _PLACEHOLDERS
    ],
    S_NEW_IMPL,
    "p4a_lineage_features",
    PHASE_P2,
    "Implemented at P4a in lineage_features.py, 54 features over the 4 groups, "
    "vectorised via groupby so 7.27M events never enter a Python loop. No GEE, "
    "no new download: arithmetic over event_local_cluster_membership + the "
    "daily cluster files. Supersedes the UNVERIFIED_HISTORICAL claim, which was "
    "inference not observation. Undefined statistics return NaN, never 0.",
)

# -- Recurrence must be re-asserted AFTER the lineage block above. Those four
#    are NOT implemented: counting neighbouring events needs a pass over all
#    7,265,849 events, which no single-event lineage pass can do. The P4a block
#    emits them as NaN, and this restores the honest status and phase so the
#    blanket assignment above cannot quietly claim them.
_set(
    [
        "RECURRENCE_RATE_1Y",
        "RECURRENCE_RATE_3Y",
        "RECURRENCE_RATE_5Y",
        "MAX_INTER_EVENT_GAP_LOG",
    ],
    S_MISSING_IMPL,
    "p4a_lineage_features",
    PHASE_P3G,
    "Emitted as NaN by lineage_features.py. Needs a chunked self-join over all "
    "events to count neighbours; cannot be done one event at a time.",
)

# -- Weather: cell 264 verified, except the seven FWI keys which it hardcodes.
_set(
    [f for f in FEATURE_GROUPS[GROUP_WEATHER] if f != "LIGHTNING_COUNT_24H"],
    S_UNVERIFIED,
    CELL_264,
    PHASE_P2,
    "Cell 264 verified implementation (use it; cell 310 is a stub and must "
    "NOT be used).",
)
_set(
    ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "FWI_ANOMALY_Z"],
    S_HIST_NAN,
    CELL_264,
    PHASE_P3D,
    "Cell 264 hardcodes all seven to np.nan in both branches. Implement "
    "standard Canadian FWI sequential equations (Van Wagner 1987).",
)
_set(
    ["LIGHTNING_COUNT_24H"],
    S_MISSING_IMPL,
    PHASE_NONE,
    PHASE_P3F,
    "ISS-LIS covers 2017-03-01..2023-11-16 only. Needs a lightning source; "
    "era-gated.",
)

# -- Industrial: NEW implementation written at P3a. extract_industrial_features
#    never existed in any notebook, so these cannot be COMPUTED_HISTORICAL.
#    Implemented in industrial_features.py, driven by
#    colab_p3a_industrial.py against external/osm_facilities. runtime_verified
#    stays False until P2/P4 actually execute it over real events.
_set(
    FEATURE_GROUPS[GROUP_INDUSTRIAL],
    S_NEW_IMPL,
    "p3a_industrial_features",
    PHASE_P2,
    "extract_industrial_features never existed in any notebook; this is a NEW "
    "implementation, not a recovery. cKDTree + STRtree over the 37 verified "
    "osm_facilities geoparquets in EPSG:6933, no GEE cost. These 18 are the "
    "LABEL_DERIVED leakage partition.",
)
_set(
    FEATURE_GROUPS[GROUP_LANDCOVER],
    S_MISSING_IMPL,
    PHASE_NONE,
    PHASE_P3C,
    "extract_landcover_features absent.",
)
_set(
    FEATURE_GROUPS[GROUP_SAR],
    S_MISSING_IMPL,
    PHASE_NONE,
    PHASE_P3E,
    "extract_sar_features absent. Sentinel-1 starts 2014-05; DNBR/DNDVI "
    "need Sentinel-2 (2015-06).",
)
_set(
    FEATURE_GROUPS[GROUP_ATMOSPHERIC],
    S_MISSING_IMPL,
    PHASE_NONE,
    PHASE_P3F,
    "extract_atmospheric_features absent. Sentinel-5P starts 2018-07-02.",
)
_set(
    FEATURE_GROUPS[GROUP_SMOKE],
    S_MISSING_IMPL,
    PHASE_NONE,
    PHASE_P3F,
    "extract_smoke_features absent. BURN_SCAR_GROWTH_RATE is schema-only in "
    "every prior forensic search; must never be inferred from FRP.",
)
# NOTE: GROUP_QUALITY is deliberately NOT set here. It is covered by the P4a
# lineage block above, which implements all 8 features from India_Firms. Setting
# it again here would silently overwrite that with MISSING_IMPLEMENTATION --
# exactly the ordering hazard this file has already sprung once.

# Fail loudly if any feature escaped assignment.
_unassigned = [f for f in FEATURES_141 if f not in STATUS_MAP]
if _unassigned:
    raise ValueError(f"Features with no status assigned: {_unassigned}")


# ---------------------------------------------------------------------------
# 8. BUILDER
# ---------------------------------------------------------------------------

LEDGER_COLUMNS = [
    "feature_index",
    "feature",
    "group",
    "status",
    "runtime_verified",
    "impl_ref",
    "phase",
    "dataset",
    "era_start",
    "era_end",
    "label_derived",
    "notes",
]


def _era_for(feature: str) -> "tuple[str, str]":
    group = GROUP_OF[feature]
    if group in (GROUP_THERMAL, GROUP_TEMPORAL, GROUP_SPREAD, GROUP_QUALITY):
        return ERA["firsms_native"]
    if group == GROUP_TERRAIN:
        return ERA_FULL
    if group in (GROUP_LANDCOVER,):
        return ERA["sentinel2"] if feature in ("NBR_CHANGE", "NDVI_ANOMALY_Z") else ERA_FULL
    if group == GROUP_INDUSTRIAL:
        return ERA["gem_osm"]
    if group == GROUP_WEATHER:
        if feature == "LIGHTNING_COUNT_24H":
            return ERA["iss_lis"]
        return ERA_FULL
    if group == GROUP_SMOKE:
        return ERA["sentinel5p"]
    if group == GROUP_SAR:
        return ERA["sentinel1"]
    if group == GROUP_ATMOSPHERIC:
        return ERA["sentinel5p"]
    return ERA_FULL


def build_audit_records() -> "list[dict]":
    """Return the audit ledger as a list of dicts (no pandas dependency)."""
    label_derived = set(LABEL_DERIVED_FEATURES)
    records = []
    for i, feature in enumerate(FEATURES_141, start=1):
        status, impl_ref, phase, notes = STATUS_MAP[feature]
        era_start, era_end = _era_for(feature)
        records.append(
            {
                "feature_index": i,
                "feature": feature,
                "group": GROUP_OF[feature],
                "status": status,
                "runtime_verified": False,
                "impl_ref": impl_ref or "",
                "phase": phase or "",
                "dataset": DATASET[feature],
                "era_start": era_start,
                "era_end": era_end,
                "label_derived": feature in label_derived,
                "notes": notes,
            }
        )
    return records


def validate_schema() -> None:
    """Structural self-check. Raises on any violation."""
    if N_FEATURES != 141:
        raise ValueError(f"Schema must have exactly 141 features, got {N_FEATURES}")
    if len(set(FEATURES_141)) != 141:
        raise ValueError("Duplicate feature names in schema")
    for group, expected in EXPECTED_GROUP_SIZES.items():
        actual = sum(1 for f in FEATURES_141 if GROUP_OF[f] == group)
        if actual != expected:
            raise ValueError(
                f"Group {group}: expected {expected} features, found {actual}"
            )
    for feature in FEATURES_141:
        if STATUS_MAP[feature][0] not in VALID_STATUSES:
            raise ValueError(f"Invalid status for {feature}")
    if len(LABEL_DERIVED_FEATURES) != 18:
        raise ValueError(
            "LABEL_DERIVED_FEATURES must be exactly the 18 industrial features, "
            f"got {len(LABEL_DERIVED_FEATURES)}"
        )
    if len(NON_LEAK_FEATURES) != 123:
        raise ValueError(
            f"NON_LEAK_FEATURES must be 123, got {len(NON_LEAK_FEATURES)}"
        )


# ---------------------------------------------------------------------------
# 9. DRIFT CHECK
# ---------------------------------------------------------------------------

def check_schema_drift(candidate) -> "dict":
    """Compare an in-notebook feature list against the frozen schema.

    Raises ValueError when the two disagree in names, size or order, so a
    drifting notebook can never silently diverge from the ledger.
    """
    if candidate is None:
        return {"status": "NO_CANDIDATE"}

    candidate = list(candidate)

    only_candidate = [f for f in candidate if f not in set(FEATURES_141)]
    only_frozen = [f for f in FEATURES_141 if f not in set(candidate)]
    same_order = candidate == FEATURES_141

    if (
        only_candidate
        or only_frozen
        or len(candidate) != N_FEATURES
        or not same_order
    ):
        raise ValueError(
            "SCHEMA DRIFT between notebook and ledger.\n"
            f"  candidate length : {len(candidate)}\n"
            f"  ledger length    : {N_FEATURES}\n"
            f"  candidate only   : {only_candidate}\n"
            f"  ledger only      : {only_frozen}\n"
            f"  identical order  : {same_order}"
        )

    return {
        "status": "OK",
        "same_order": same_order,
        "n_features": len(candidate),
    }


validate_schema()