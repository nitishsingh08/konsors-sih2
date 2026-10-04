# ============================================================================
# P3a -- COLAB CELL: industrial context features for all global fire events
#
# Run this AFTER the P1 cell, and after P1b has confirmed that
#   external/osm_facilities/ is populated (37 files, 0.28 GB -- verified OK).
#
# What it does
#   1. Reads every geoparquet under external/osm_facilities/, normalises the
#      facility taxonomy onto the 10 categories the ledger needs, and prints
#      the resolved mapping so you can confirm it before any compute happens.
#   2. Projects facility coordinates and event centroids to EPSG:6933 metres.
#   3. Streams canonical_event_base.parquet in batches -- never loading all
#      7,265,849 events into memory -- and emits the 18 INDUSTRIAL features
#      per event using cKDTree + STRtree.
#   4. Writes chunked parquet + a QC report.
#
# Cost: no Earth Engine, no API quota. Pure CPU geometry.
#
# THE CELL IS SAFE BY DEFAULT. It stops after step 1 and prints what it found.
# Nothing is written until you set INSPECT_ONLY = False and confirm the printed
# CATEGORY_MAP is right, because a silently wrong category map would poison
# both the model features and the labels.
# ============================================================================

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import feature_ledger as fl
import industrial_features as ind
import colab_utils as cu

print("Ledger version   :", fl.LEDGER_VERSION)
print("Industrial module: OK")
print(cu.drive_report())

# ---------------------------------------------------------------------------
# 0. CONFIGURATION
# ---------------------------------------------------------------------------

INSPECT_ONLY = False         # flip to False once CATEGORY_MAP below is verified

# Printed after the config block, not before: `MODE` reads INSPECT_ONLY, so
# putting it above the assignment was a NameError waiting to happen.
MODE = "INSPECT ONLY (writes nothing)" if INSPECT_ONLY else "FULL PASS (writes)"
print(f"MODE: {MODE}")
CHUNK_ROWS = 200_000         # events per batch; raise if RAM allows
MAX_EVENTS = None            # set an int to smoke-test on a subset, e.g. 50_000
TARGET_CRS = ind.REQUIRED_CRS  # EPSG:6933
BOUNDARY_DENSIFY_M = 50.0    # boundary sampling step for the boundary distance

# ---- Resolve Drive locations ------------------------------------------------
# Everything derived from PROJECT_ROOT is computed HERE, after resolution.
# Deriving paths in the config block above ran before resolution and raised
# `TypeError: unsupported operand type(s) for /: 'NoneType' and 'str'`.
cu.require_drive("the facility gazetteer")

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError(
        f"No directory under {cu.MYDRIVE} contains {cu.CANONICAL_SUBPATH}.\n"
        "Run colab_survey_tree.py and paste the output."
    )

_CANON = PROJECT_ROOT / "full_2012_2026"
EXTERNAL_ROOT, _ext_table = cu.pick_by_file_count(PROJECT_ROOT, "external")
if EXTERNAL_ROOT is None:
    EXTERNAL_ROOT = _CANON / "external"
EVENTS_DIR, _ev_table = cu.pick_by_file_count(PROJECT_ROOT, "events")
if EVENTS_DIR is None:
    EVENTS_DIR = _CANON / "events"

FACILITY_DIR = EXTERNAL_ROOT / "osm_facilities"
OUT_DIR = _CANON / "features" / "industrial"
QC_DIR = PROJECT_ROOT / "qc"

print()
print("RESOLVED LOCATIONS")
print("-" * 74)
for label, path in (
    ("project root", PROJECT_ROOT),
    ("external", EXTERNAL_ROOT),
    ("facilities", FACILITY_DIR),
    ("events", EVENTS_DIR),
    ("output", OUT_DIR),
    ("qc", QC_DIR),
):
    flag = "ok " if cu.path_ok(path) else "MISSING"
    print(f"  [{flag}] {label:14s} {path}")
print("-" * 74)

# ---------------------------------------------------------------------------
# 1. GAZETTEER TAXONOMY
#
# The 37 files are named after OSM/GEM categories, not after the 10 ledger
# categories, and several files overlap (e.g. `industrial.geoparquet` almost
# certainly contains the members of `cement`, `steel`, `lng`, ... as well).
# Mapping them by hand here is the single highest-leverage thing in this cell:
# a wrong mapping silently corrupts both the features and the labels.
#
# Each entry is (regex over the filename, target category or None to skip).
# Order matters -- first match wins.
# ---------------------------------------------------------------------------

CATEGORY_MAP: "list[tuple[str, str | None]]" = [
    (r"refiner", "refinery"),
    (r"petrochem|petro_chem", "petrochemical"),
    (r"power[_ ]?plant|powerstation|thermal_power", "powerplant"),
    (r"steel|metal|blast_furnace|coking", "steel"),
    (r"cement|lime|clinker", "cement"),
    (r"(^|_)lng(_|$)|liquef|gas_terminal|gas_process|gas_plant", "lng"),
    (r"mine|mining|quarry|aggregate|open_pit|coal_", "mine"),
    (r"landfill|waste|incinerat|slag", "landfill"),
    (r"pipeline|pipe_?line|petroduct", "pipeline"),
    (r"well[_ ]?pad|wellhead|drill_?site|oil_?pad|onshore", "wellpad"),
    # Supersets: keep only as a fallback when no specific file exists.
    (r"all_industrial|^industrial\.", None),   # skipped: duplicates the above
    (r"facilit|facility", None),               # skipped: ditto
]

# Event centroid columns in canonical_event_base.parquet. Verified names are
# centroid_lon / centroid_lat (see HANDOFF 4, cell 260 event_geometry); the
# cell prints the real schema so this can be corrected in place if they differ.
# Discovered automatically in STEP 2 from the real event-table schema; these
# are only fallbacks if discovery fails.
EVENT_LON_COL = "event_centroid_lon"
EVENT_LAT_COL = "event_centroid_lat"
X_COL = None
Y_COL = None
COORD_KIND = None

# Optional event columns feeding KNOWN_FLARE. Missing ones degrade KNOWN_FLARE
# to NaN rather than to 0, so the absence is visible in the QC report.
FLARE_SUPPORT_COLUMNS = {
    "active_days": "active_days",
    "detection_count": "detection_count",
    "centroid_displacement_m": "centroid_displacement_m",
    "frp_cv": "frp_cv",
}

# ---------------------------------------------------------------------------
# 2. INSPECT THE GAZETTEER  (always runs)
# ---------------------------------------------------------------------------


def resolve_category(stem: str):
    """Map a geoparquet stem onto one of the 10 ledger categories, or None."""
    name = stem.lower()
    for pattern, category in CATEGORY_MAP:
        if re.search(pattern, name):
            return category
    return None


def _dir_penalty(path):
    """Ranking key for a directory name: unsuffixed wins over " (n)" copies.

    The gazetteer is duplicated across directories (`facility_categories/` and
    a macOS-style `facility_categories (1)/`), so the same filename appears
    twice with different row counts. Loading both double-counts every facility
    in the duplicated folder, which biases the FACILITY_COUNT_* features and
    the DIST_* trees.
    """
    penalty = 0
    for part in path.relative_to(FACILITY_DIR).parts[:-1]:
        m = re.search(r"\s*\((\d+)\)$", part)
        if m:
            penalty += 1_000 + int(m.group(1))
    return penalty


def list_facility_files():
    if not cu.path_ok(FACILITY_DIR):
        raise FileNotFoundError(
            f"{FACILITY_DIR} not found. Run colab_p1b_external_inventory.py "
            "first -- osm_facilities must be confirmed present before P3a."
        )
    files = sorted(
        [p for p in FACILITY_DIR.rglob("*.geoparquet")]
        + [p for p in FACILITY_DIR.rglob("*.parquet") if "geoparquet" not in p.suffix]
    )
    if not files:
        raise FileNotFoundError(f"No geoparquet files under {FACILITY_DIR}")

    # Collapse same-named duplicates: keep the one in the least-suffixed
    # directory, and report every drop so it is visible rather than silent.
    best = {}
    for path in files:
        key = path.name
        rank = (_dir_penalty(path), str(path))
        if key not in best or rank < best[key][0]:
            best[key] = (rank, path)

    kept = sorted((v[1] for v in best.values()), key=lambda p: p.name)
    dropped = [p for p in files if p not in set(kept)]

    print()
    print("gazetteer duplicate handling")
    print("-" * 74)
    print(f"  files found   : {len(files)}")
    print(f"  unique names  : {len(kept)}")
    print(f"  dropped as dup: {len(dropped)}")
    if dropped:
        print("  dropped copies (a directory was duplicated on Drive):")
        for path in sorted(dropped, key=lambda p: str(p))[:12]:
            kept_twin = next(
                (q for q in kept if q.name == path.name), None
            )
            rel = path.relative_to(FACILITY_DIR)
            note = ""
            if kept_twin is not None:
                note = f"  kept {kept_twin.relative_to(FACILITY_DIR)}"
            print(f"    {rel}{note}")
    print("-" * 74)
    return kept


def read_gazetteer():
    """Read every facility file, keeping points, polygons and category labels.

    Returns (records, per-file report). `records` entries are
    {"category", "x", "y", "boundary": [np.ndarray | None]}.
    """
    files = list_facility_files()
    print("=" * 78)
    print("P3a STEP 1 -- GAZETTEER INSPECTION")
    print("=" * 78)
    print(f"files: {len(files)}")

    import shapely
    from shapely import wkb as shapely_wkb

    records = []
    report = []

    for path in files:
        category = resolve_category(path.stem)
        try:
            table = pq.read_table(path)
        except Exception as exc:                       # noqa: BLE001
            report.append({"file": path.name, "category": category, "error": str(exc)})
            print(f"  SKIP {path.name}: {exc}")
            continue

        cols = {c.lower(): c for c in table.column_names}
        geom_col = next(
            (cols[k] for k in ("geometry", "geom", "wkb_geometry") if k in cols), None
        )
        if geom_col is None:
            report.append(
                {"file": path.name, "category": category, "error": "no geometry column"}
            )
            print(f"  SKIP {path.name}: columns {table.column_names}")
            continue

        geoms = shapely.from_wkb(np.asarray(table.column(geom_col).to_numpy(zero_copy_only=False)))
        geoms = np.asarray(geoms, dtype="object")

        # Category label column, used to refine the filename mapping when the
        # file is a superset such as `industrial`.
        label_col = next(
            (cols[k] for k in ("category", "facility_type", "type", "class", "tags")
             if k in cols),
            None,
        )
        labels = (
            table.column(label_col).to_numpy(zero_copy_only=False)
            if label_col
            else None
        )

        point_count = 0
        polygon_count = 0
        unresolved: dict = {}

        for i, geom in enumerate(geoms):
            if geom is None or getattr(geom, "is_empty", True):
                continue
            cat = category
            if cat is None and labels is not None:
                cat = refine_category(labels[i])
            if cat is None:
                # Unmapped and unlabelled: keep the coordinates out of every
                # category tree rather than guessing. Counted for the report.
                key = "UNMAPPED"
                unresolved[key] = unresolved.get(key, 0) + 1
                continue

            if geom.geom_type == "Point":
                records.append(
                    {
                        "category": cat,
                        "x": geom.x,
                        "y": geom.y,
                        "boundary": None,
                        "geom": None,
                        # Representative point for the distance trees. For a
                        # Point this is the point itself.
                        "rep_x": geom.x,
                        "rep_y": geom.y,
                        "is_point": True,
                    }
                )
                point_count += 1
            else:
                rings = densify_boundary(geom)
                if rings is not None:
                    # A polygon still has to appear in the DIST_* trees, or it
                    # is invisible to the model. Its centroid is the
                    # representative point. Without this, `mine` contributed
                    # 425 points while 10,886 polygons were dropped, and
                    # `petrochemical` had no points at all -- making
                    # DIST_PETROCHEMICAL_LOG 100% NaN and DIST_MINE wrong for
                    # every event near a polygonal mine.
                    try:
                        centroid = geom.centroid
                        rep_x, rep_y = float(centroid.x), float(centroid.y)
                        if not np.isfinite(rep_x) or not np.isfinite(rep_y):
                            rep_x = rep_y = None
                    except Exception:              # noqa: BLE001
                        rep_x = rep_y = None
                    records.append(
                        {
                            "category": cat,
                            "x": None,
                            "y": None,
                            "boundary": rings,
                            # The original geometry, kept because a point-in-
                            # polygon test needs real rings. Rebuilding a polygon
                            # from the densified open path would be invalid.
                            "geom": geom,
                            "rep_x": rep_x,
                            "rep_y": rep_y,
                            "is_point": False,
                        }
                    )
                    polygon_count += 1

        report.append(
            {
                "file": path.name,
                "category": category,
                "rows": len(geoms),
                "points": point_count,
                "polygons": polygon_count,
                "unmapped": unresolved,
                "columns": list(table.column_names),
            }
        )
        flag = "skip" if category is None else f"-> {category}"
        print(
            f"  {path.name:52s} rows={len(geoms):>7d} "
            f"pts={point_count:>7d} poly={polygon_count:>6d}  {flag}"
        )

    return records, report


def refine_category(raw_label) -> "str | None":
    """Map an in-file category label onto the 10 ledger categories."""
    if raw_label is None:
        return None
    text = raw_label.decode() if isinstance(raw_label, bytes) else str(raw_label)
    return resolve_category(text)


def densify_boundary(geom, step_m=BOUNDARY_DENSIFY_M):
    """Sample a polygon's rings at `step_m` so a boundary distance can be
    approximated with a cKDTree.

    Exact for points and for polygons far from any vertex; inside the polygon
    the distance is 0 regardless, so the sampling only affects the outside
    case, where the error is bounded by step_m / 2. Documented approximation,
    not an exact boundary-to-boundary distance.
    """
    import shapely
    from shapely.geometry import LineString, MultiLineString, MultiPolygon

    parts = [geom] if geom.geom_type != "MultiPolygon" else list(geom.geoms)
    lines = []
    for part in parts:
        if not hasattr(part, "exterior"):
            continue
        rings = [part.exterior] + list(getattr(part, "interiors", []))
        for ring in rings:
            ring_coords = np.asarray(ring.coords, dtype="float64")[:-1]
            if len(ring_coords) >= 2:
                lines.append(LineString(ring_coords))
    if not lines:
        return None
    line = lines[0] if len(lines) == 1 else MultiLineString(lines)
    try:
        sampled = shapely.segmentize(line, max_segment_length=float(step_m))
    except AttributeError:                             # older shapely
        sampled = line
    coords = np.asarray(shapely.get_coordinates(sampled), dtype="float64")
    return coords if len(coords) else None


records, gazetteer_report = read_gazetteer()

print()
print("-" * 78)
print("RESOLVED CATEGORY MAP -- CONFIRM THIS BEFORE COMPUTING")
print("-" * 78)
per_category_points: dict = {}
per_category_polys: dict = {}
for rec in records:
    bucket_points = per_category_points.setdefault(rec["category"], 0)
    bucket_polys = per_category_polys.setdefault(rec["category"], 0)
    if rec["boundary"] is None:
        per_category_points[rec["category"]] = bucket_points + 1
    else:
        per_category_polys[rec["category"]] = bucket_polys + 1

for cat in ind.CATEGORIES:
    np_pts = per_category_points.get(cat, 0)
    np_poly = per_category_polys.get(cat, 0)
    status = "ok" if (np_pts or np_poly) else "MISSING -> feature will be NaN"
    print(
        f"  {cat:14s} points={np_pts:>8d} polygons={np_poly:>6d}   {status}"
    )
stray = per_category_points.get("UNMAPPED", 0) + per_category_polys.get("UNMAPPED", 0)
print(f"  unmapped records dropped: {stray}")
print()
print("Categories that end up MISSING are not an error: the corresponding")
print("DIST_* feature is reported as NaN, which is distinguishable from a")
print("measured zero. Add a mapping above if a category should have been found.")

# ---------------------------------------------------------------------------
# 3. INSPECT THE EVENT TABLE
# ---------------------------------------------------------------------------

print()
print("=" * 78)
print("P3a STEP 2 -- EVENT TABLE SCHEMA")
print("=" * 78)

EVENT_BASE = next(
    (
        p
        for p in (
            EVENTS_DIR / "canonical_event_base.parquet",
            EVENTS_DIR / "events.parquet",
            EVENTS_DIR / "global_event_base.parquet",
        )
        if cu.path_ok(p)
    ),
    None,
)
if EVENT_BASE is None:
    print()
    print("parquet files under", EVENTS_DIR)
    for p_ in sorted(EVENTS_DIR.glob("*.parquet"))[:20]:
        print("   ", p_.name)
    raise FileNotFoundError(
        "No canonical event base found under "
        f"{EVENTS_DIR}. It is a frozen artefact: restore it from Drive, never "
        "rebuild it."
    )

event_pf = pq.ParquetFile(EVENT_BASE)
event_columns = event_pf.schema_arrow.names
print(f"rows      : {event_pf.metadata.num_rows:,}")
print(f"row groups: {event_pf.metadata.num_row_groups:,}")
print(f"columns   : {event_columns}")

# --- automatic coordinate discovery ---------------------------------------
# The handoff names `centroid_lon`/`centroid_lat`, but the real table uses
# `event_centroid_lon`/`event_centroid_lat` AND carries pre-projected
# `event_centroid_x`/`event_centroid_y`. Preferring the projected pair removes
# a projection step entirely and eliminates a class of unit error, so it wins
# when both are present -- but only after the magnitudes confirm metres.
_printable = {c.lower(): c for c in event_columns}
_pairs = (
    ("event_centroid_x", "event_centroid_y"),
    ("centroid_x", "centroid_y"),
    ("event_centroid_lon", "event_centroid_lat"),
    ("centroid_lon", "centroid_lat"),
    ("lon", "lat"),
    ("longitude", "latitude"),
)


def _looks_like_metres(df, col):
    """True when `col` holds projected metres rather than degrees."""
    try:
        head = pd.to_numeric(
            df[col].head(2000), errors="coerce"
        ).abs().max()
    except Exception:                                # noqa: BLE001
        return False
    if not np.isfinite(head):
        return False
    # Degrees never exceed 180; Equal Earth metres are in the millions.
    return head > 1000.0


_probe_cols = [c for pair in _pairs for c in pair if c in event_columns]
_probe = (
    pd.read_parquet(EVENT_BASE, columns=_probe_cols).head(2000)
    if _probe_cols
    else pd.DataFrame()
)
COORD_SOURCE = None
for cx, cy in _pairs:
    if cx in event_columns and cy in event_columns:
        if _looks_like_metres(_probe, cx) and _looks_like_metres(_probe, cy):
            X_COL, Y_COL, COORD_KIND = cx, cy, "projected metres"
            COORD_SOURCE = f"{cx}/{cy} (already in {TARGET_CRS})"
            break
        EVENT_LON_COL, EVENT_LAT_COL = cx, cy
        COORD_KIND = "lon/lat degrees -> must project"
        COORD_SOURCE = f"{cx}/{cy}"
        break

if COORD_SOURCE is None:
    print()
    print("!! No coordinate pair found in the event table.")
    print(f"   columns: {event_columns}")
    print("   Set X_COL / Y_COL by hand and re-run. Do not guess: a wrong")
    print("   centroid silently shifts every industrial feature.")
    raise KeyError("no coordinate columns")

print()
print(f"coordinate columns : {COORD_SOURCE}")
print(f"coordinate kind    : {COORD_KIND}")
if COORD_KIND != "projected metres":
    print("  -> these are degrees; the cell will project to "
          f"{TARGET_CRS} before any distance maths")

# --- pre-existing feature columns -----------------------------------------
# The event base already carries cell-325-style aggregates. Note them so they
# can be compared against the P4a lineage output rather than recomputed blind.
PREEXISTING = [
    c
    for c in event_columns
    if c.startswith("geometry_")
    or c in ("n_detections", "total_frp", "max_frp", "duration_hours",
             "n_local_clusters")
]
if PREEXISTING:
    print()
    print(f"PRE-EXISTING aggregate columns ({len(PREEXISTING)}): "
          f"{PREEXISTING}")
    print("  These look like cell-325 outputs. The P4a lineage pass computes")
    print("  an independent version of several of them; the two should be")
    print("  compared, not assumed equal.")

# Columns that appear twice with a _1 suffix indicate a bad merge at some
# point. Harmless here (only the un-suffixed names are read) but worth naming.
_suffixed = [c for c in event_columns if re.fullmatch(r".+_1", c)]
if _suffixed:
    print()
    print(f"WARNING: {len(_suffixed)} column(s) look like merge duplicates: "
          f"{_suffixed[:8]}")
    print("  This cell reads only the un-suffixed originals, so no action is")
    print("  needed here -- but a previous merge left debris in the table.")

present_flare = {k: v for k, v in FLARE_SUPPORT_COLUMNS.items() if v in event_columns}
missing_flare = {k: v for k, v in FLARE_SUPPORT_COLUMNS.items() if v not in event_columns}
print()
print("KNOWN_FLARE support columns present:", present_flare or "none")
print("KNOWN_FLARE support columns missing:", missing_flare or "none")
if missing_flare:
    print(
        "  KNOWN_FLARE will be NaN for every event. That is the honest result: "
        "it needs an all-events pass (P3g) or the cell-325 outputs."
    )

# ---------------------------------------------------------------------------
# 4. STOP HERE UNLESS CONFIRMED
# ---------------------------------------------------------------------------

if INSPECT_ONLY:
    print()
    print("=" * 78)
    print("INSPECTION COMPLETE -- NOTHING WRITTEN")
    print("=" * 78)
    print("1. Check the CATEGORY MAP above. Any MISSING category that should")
    print("   exist? Add a regex to CATEGORY_MAP and re-run this cell.")
    print("2. If the centroid columns were wrong, fix EVENT_LON_COL/ _LAT_COL.")
    print("------------------------------------------------------------")
    print("STOPPED IN INSPECT MODE. Nothing was written.")
    print("To actually run: edit INSPECT_ONLY to False, then re-run.")
    print("------------------------------------------------------------")
    _STOP = True
else:
    _STOP = False
    print()
    print("QC_DIR.mkdir(parents=True, exist_ok=True)")
    with open(QC_DIR / "p3a_gazetteer_report.json", "w") as fh:
        json.dump(
            {
                "files": gazetteer_report,
                "points_per_category": per_category_points,
                "polygons_per_category": per_category_polys,
                "event_columns": event_columns,
                "flare_support_present": list(present_flare),
                "flare_support_missing": list(missing_flare),
            },
            fh,
            indent=2,
        )
    print("Wrote", QC_DIR / "p3a_gazetteer_report.json")

# ===========================================================================
def run_full_pass():
    """Run the 18 industrial features over every event."""
    # 5. COMPUTE  (only reached once INSPECT_ONLY = False)
    # ===========================================================================

    import shapely
    from shapely import STRtree
    from shapely.geometry import LineString, MultiLineString

    print()
    print("=" * 78)
    print("P3a STEP 3 -- PROJECTION AND TREES")
    print("=" * 78)

    from pyproj import Transformer

    to_target = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)


    def project_points(lon, lat):
        """Vectorised lon/lat -> Equal Earth metres."""
        x, y = to_target.transform(np.asarray(lon, dtype="float64"),
                                   np.asarray(lat, dtype="float64"))
        x = np.asarray(x, dtype="float64")
        y = np.asarray(y, dtype="float64")
        # Infinite transform results mean the point fell outside the projection.
        bad = ~np.isfinite(x) | ~np.isfinite(y)
        if bad.any():
            print(f"  !! {int(bad.sum())} event centroids failed to project -> NaN")
            x[bad] = np.nan
            y[bad] = np.nan
        return x, y


    # --- per-category point trees ----------------------------------------------
    category_trees: dict = {}
    for cat in ind.CATEGORIES:
        # Every facility with a usable representative point, polygons included.
        # Filtering on `boundary is None` here would silently drop every polygon
        # from the distance features.
        pts = [
            (r["rep_x"], r["rep_y"])
            for r in records
            if r["category"] == cat
            and r.get("rep_x") is not None
            and r.get("rep_y") is not None
        ]
        if pts:
            arr = np.array(pts, dtype="float64")
            catx, caty = project_points(arr[:, 0], arr[:, 1])
            good = np.isfinite(catx) & np.isfinite(caty)
            category_trees[cat] = np.column_stack([catx[good], caty[good]])
        else:
            category_trees[cat] = None
        n = 0 if category_trees[cat] is None else len(category_trees[cat])
        print(f"  {cat:14s} point tree: {n:>8,d}")

    # --- boundary point cloud for DIST_FACILITY_BOUNDARY_LOG ---------------------
    boundary_chunks = [r["boundary"] for r in records if r["boundary"] is not None]
    boundary_xy = None
    if boundary_chunks:
        raw = np.concatenate(boundary_chunks, axis=0)
        bx, by = project_points(raw[:, 0], raw[:, 1])
        good = np.isfinite(bx) & np.isfinite(by)
        boundary_xy = np.column_stack([bx[good], by[good]])
        print(f"  boundary samples: {len(boundary_xy):,d} "
              f"(step {BOUNDARY_DENSIFY_M:.0f} m, error <= {BOUNDARY_DENSIFY_M / 2:.0f} m outside)")
    else:
        print("  no polygon boundaries -> DIST_FACILITY_BOUNDARY_LOG will be NaN")

    # --- polygons for INSIDE_FACILITY -------------------------------------------
    # The original geometries are reused here, not the densified ring samples: the
    # point-in-polygon answer must come from valid closed rings.
    polys = [rec["geom"] for rec in records if rec["geom"] is not None]
    polys = [p for p in polys if p.is_valid and not p.is_empty]
    inside_tree = STRtree(polys) if polys else None
    print(f"  STRtree polygons: {len(polys)}")
    if not polys:
        print("  no usable polygons -> INSIDE_FACILITY will be NaN")

    print()
    print("=" * 78)
    print("P3a STEP 4 -- STREAMING THE 7,265,849 EVENTS")
    print("=" * 78)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    read_cols = [X_COL, Y_COL]
    read_cols += list(present_flare.values())
    read_cols += [c for c in ("event_id",) if c in event_columns]
    read_cols = list(dict.fromkeys(read_cols))
    print("reading columns:", read_cols)

    need_cols = set(read_cols)
    null_counts = {name: 0 for name in ind.INDUSTRIAL_FEATURES}
    rows_written = 0
    chunk_files = []
    events_seen = 0

    batch_iter = event_pf.iter_batches(batch_size=CHUNK_ROWS, columns=read_cols)
    for batch_no, batch in enumerate(batch_iter):
        frame = batch.to_pandas()
        if frame.empty:
            continue
        if MAX_EVENTS is not None and events_seen >= MAX_EVENTS:
            break

        col_x = pd.to_numeric(frame[X_COL], errors="coerce").to_numpy("float64")
        col_y = pd.to_numeric(frame[Y_COL], errors="coerce").to_numpy("float64")
        if COORD_KIND == "projected metres":
            # Already Equal Earth metres. Projecting again would rescale every
            # distance, so this branch deliberately does nothing.
            x, y = col_x, col_y
        else:
            x, y = project_points(col_x, col_y)
        pts = np.column_stack([x, y])
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.any():
            continue
        if not finite.all():
            # Unprojectable centroids still need their 18 keys emitted as NaN.
            pts = np.where(finite[:, None], pts, np.nan)

        inside = None
        if inside_tree is not None:
            geoms = shapely.points(x, y)
            hits = inside_tree.query(geoms, predicate="intersects")
            inside = np.zeros(len(pts), dtype="float64")
            inside[np.unique(hits[0])] = 1.0
            del geoms

        boundary = None
        if boundary_xy is not None:
            boundary = ind.nearest_distance(pts[np.isfinite(x) & np.isfinite(y)], boundary_xy)
            full = np.full(len(pts), np.nan)
            full[finite] = boundary
            boundary = full

        flare_site = None
        if inside_tree is not None:
            # A centroid inside any polygon is treated as flare-capable only when
            # the polygon belongs to a flare-capable category; the STRtree here is
            # over all polygons, so fall back to the union distance plus the
            # category trees for the per-category test.
            flare_site = np.zeros(len(pts), dtype=bool)
            flare_pts = []
            for cat in ind.FLARE_SITE_CATEGORIES:
                tree = category_trees.get(cat)
                if tree is None or len(tree) == 0:
                    continue
                d = ind.nearest_distance(pts[finite], tree)
                sub = np.zeros(int(finite.sum()), dtype=bool)
                sub[np.nan_to_num(d, nan=np.inf) <= 10_000.0] = True
                flare_pts.append(sub)
            if flare_pts:
                flare_site[finite] = np.any(np.vstack(flare_pts), axis=0)
            else:
                flare_site = None

        support = {}
        for key, col in present_flare.items():
            support[key] = pd.to_numeric(frame[col], errors="coerce").to_numpy("float64")

        features = ind.compute_industrial_features(
            pts,
            category_trees,
            crs=TARGET_CRS,
            inside_flags=inside,
            boundary_distance_m=boundary,
            category_is_flare_site=flare_site,
            **support,
        )

        out = pd.DataFrame(features, columns=ind.INDUSTRIAL_FEATURES)
        if "event_id" in frame.columns:
            out.insert(0, "event_id", frame["event_id"].to_numpy())

        for name in ind.INDUSTRIAL_FEATURES:
            null_counts[name] += int(out[name].isna().sum())

        path = OUT_DIR / f"industrial_chunk_{batch_no:04d}.parquet"
        out.to_parquet(path, index=False, compression="snappy")
        chunk_files.append({"file": path.name, "rows": len(out)})
        rows_written += len(out)
        events_seen += len(out)

        if batch_no % 5 == 0:
            print(f"  chunk {batch_no:>4d}  rows={len(out):>7,d}  "
                  f"total={rows_written:>10,d}")

    print()
    print(f"rows written: {rows_written:,}")

    # ---------------------------------------------------------------------------
    # 6. QC
    # ---------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("P3a QC")
    print("=" * 78)
    print(f"{'feature':30s} {'NaN %':>8s} {'min':>12s} {'max':>12s}")
    summary = {}
    for name in ind.INDUSTRIAL_FEATURES:
        nulls = null_counts[name]
        frac = 100.0 * nulls / rows_written if rows_written else float("nan")
        summary[name] = {"nan_fraction": frac, "nan_count": nulls}
        print(f"{name:30s} {frac:>7.2f}% {'-':>12s} {'-':>12s}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "event_base": str(EVENT_BASE),
        "events_total_in_table": event_pf.metadata.num_rows,
        "rows_written": rows_written,
        "target_crs": TARGET_CRS,
        "chunk_rows": CHUNK_ROWS,
        "max_events": MAX_EVENTS,
        "boundary_densify_m": BOUNDARY_DENSIFY_M,
        "category_points": {k: 0 if v is None else len(v) for k, v in category_trees.items()},
        "boundary_samples": 0 if boundary_xy is None else len(boundary_xy),
        "feature_order": ind.INDUSTRIAL_FEATURES,
        "null_fraction": summary,
        "chunks": chunk_files,
    }
    with open(OUT_DIR / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("Wrote", OUT_DIR / "manifest.json")

    print()
    print("Status to record in feature_ledger.py once you have inspected the")
    print("output and a 10-event spot check passes: all 18 INDUSTRIAL features")
    print("move from MISSING_IMPLEMENTATION to NEW_IMPLEMENTATION. They are NOT")
    print("COMPUTED_HISTORICAL, because no historical implementation exists, and")
    print("runtime_verified stays False until P2/P4 actually execute them.")


if not _STOP:
    run_full_pass()
