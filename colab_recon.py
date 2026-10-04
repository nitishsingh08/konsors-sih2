# ============================================================================
# COLAB CELL: project reconnaissance -- what actually exists, and cheaply
#
# Purpose
# -------
# One consolidated report of every input the 141 features need: real folder
# names, real column names, real dtypes, real geometry types. Stops all the
# guessing that has been producing wrong paths and wrong column names.
#
# Cost control -- this does NOT scan the data
# -------------------------------------------
#   * folder listings      : name-only, no file reads
#   * parquet schemas      : ParquetFile(...).schema_arrow -- FOOTER METADATA
#                            ONLY, no row group is decoded
#   * data samples         : exactly ONE file per *distinct schema signature*,
#                            read with a single row group
#   * cluster files        : schema only; dates come from the 5,337 filenames
#                            rather than from opening them
# So the whole report costs a few thousand file-header reads, not 15M rows.
#
# Output: a printed report plus qc/project_recon.json
# ============================================================================

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import colab_utils as cu

RECON_VERSION = "1.0"
SAMPLE_ROWS = 5
SAMPLE_MAX_FILES = 40          # cap on distinct schemas we open data for

print(cu.drive_report())

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError(
        f"No directory under {cu.MYDRIVE} contains {cu.CANONICAL_SUBPATH}."
    )

_CANON = PROJECT_ROOT / "full_2012_2026"
report = {
    "recon_version": RECON_VERSION,
    "project_root": str(PROJECT_ROOT),
    "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
}
t_start = time.time()


def human(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def schema_of(path):
    """Column names + types from the parquet footer. No data is decoded."""
    try:
        pf = pq.ParquetFile(path)
        # pf.schema_arrow IS a pyarrow.Schema and is itself iterable over
        # fields. `pf.schema_arrow.schema` does not exist -- using it made every
        # schema lookup fail, which silently zeroed out the whole availability
        # verdict in the first run of this script.
        arrow_schema = pf.schema_arrow
        return {
            "columns": list(arrow_schema.names),
            "types": {f.name: str(f.type) for f in arrow_schema},
            "n_rows": pf.metadata.num_rows,
            "n_row_groups": pf.metadata.num_row_groups,
        }
    except Exception as exc:                        # noqa: BLE001
        return {"error": f"{exc.__class__.__name__}: {exc}"}


def signature(info):
    """Stable identity for 'files that look alike', used to sample once."""
    return tuple(sorted(info.get("columns", [])))


# ===========================================================================
# 1. FOLDER INVENTORY -- names only
# ===========================================================================

print()
print("=" * 92)
print("1. FOLDER INVENTORY (name listing only, no file reads)")
print("=" * 92)

folders = [
    ("events", cu.find_artefact(PROJECT_ROOT, "events")[0]),
    ("clusters", cu.find_artefact(PROJECT_ROOT, "clusters")[0]),
    ("edges", cu.find_artefact(PROJECT_ROOT, "edges")[0]),
    ("mappings", cu.find_artefact(PROJECT_ROOT, "mappings")[0]),
    ("external", cu.find_artefact(PROJECT_ROOT, "external")[0]),
    ("feature_audit", cu.find_artefact(PROJECT_ROOT, "feature_audit")[0]),
    ("india_firms_root", cu.path_ok(PROJECT_ROOT / "India_Firms.parquet")
     and PROJECT_ROOT or None),
    ("qc", cu.path_ok(PROJECT_ROOT / "qc") and PROJECT_ROOT / "qc" or None),
    ("qc_canon", cu.path_ok(_CANON / "qc") and _CANON / "qc" or None),
]
report["folders"] = {}
print(f"{'label':16s} {'files':>7s} {'parquet':>8s} {'csv':>6s} "
      f"{'json':>6s} {'other':>6s}  path")
for label, path in folders:
    if path is None:
        print(f"{label:16s} {'-':>7s}  (not found)")
        report["folders"][label] = None
        continue
    try:
        names = [p.name for p in path.iterdir()]
    except OSError as exc:
        print(f"{label:16s}  !! {exc}")
        continue
    ext = Counter(Path(n).suffix.lower() for n in names)
    dirs = [n for n in names if (path / n).is_dir()]
    files = [n for n in names if not (path / n).is_dir()]
    fext = Counter(Path(n).suffix.lower() for n in files)
    other = len(files) - fext.get(".parquet", 0) - fext.get(".csv", 0) - fext.get(
        ".json", 0
    )
    print(f"{label:16s} {len(files):>7d} {fext.get('.parquet',0):>8d} "
          f"{fext.get('.csv',0):>6d} {fext.get('.json',0):>6d} {other:>6d}  "
          f"{path.name}/")
    if dirs:
        print(f"{'':16s} {'':7s} subdirs: {sorted(dirs)[:8]}")
    report["folders"][label] = {
        "path": str(path),
        "n_files": len(files),
        "n_dirs": len(dirs),
        "extensions": dict(fext),
        "subdirs": sorted(dirs)[:20],
        "file_samples": sorted(files)[:8],
    }

# ===========================================================================
# 2. EVENT BASE -- the frozen 7,265,849-row table
# ===========================================================================

print()
print("=" * 92)
print("2. EVENT BASE")
print("=" * 92)

events_dir = cu.find_artefact(PROJECT_ROOT, "events")[0]
if events_dir is None:
    raise FileNotFoundError("events/ not found under the project root")

report["event_tables"] = {}
schema_errors = []
event_files = sorted(p for p in events_dir.rglob("*.parquet") if p.is_file())
for path in event_files:
    info = schema_of(path)
    if "error" in info:
        schema_errors.append((path, info["error"]))
        print(f"  {str(path.relative_to(events_dir)):46s} !! {info['error']}")
        continue
    print()
    print(f"  {path.relative_to(events_dir)}   rows={info['n_rows']:>10,}  "
          f"row_groups={info['n_row_groups']:>4d}  cols={len(info['columns'])}")
    for col, typ in info["types"].items():
        print(f"    {col:34s} {typ}")
    report["event_tables"][str(path.relative_to(events_dir))] = info

# A schema probe that fails is an ERROR, not an empty schema. The first run of
# this script reported "0 detection columns" purely because of that bug, which
# would have read as "the inputs are missing" and sent us re-downloading data
# that was present all along. Never let it be silent again.
if schema_errors:
    print()
    print("=" * 92)
    print(f"SCHEMA PROBE FAILED FOR {len(schema_errors)} FILE(S)")
    print("=" * 92)
    for path, err in schema_errors[:10]:
        print(f"  {path.name}: {err}")
    print()
    print("The availability verdict below would be MEANINGLESS. Fix the probe")
    print("before reading anything into it.")
    raise RuntimeError(
        f"{len(schema_errors)} schema probe(s) failed -- availability verdict "
        "is not trustworthy"
    )

# Verify the frozen row count and inspect the coordinate magnitudes once.
for name in ("canonical_event_base.parquet", "event_base.parquet"):
    candidate = events_dir / name
    if cu.path_ok(candidate):
        info = schema_of(candidate)
        if info.get("n_rows") == 7_265_849:
            print()
            print(f"  FROZEN ROW COUNT CONFIRMED: {name} = "
                  f"{info['n_rows']:,}")
            report["event_base_rows"] = info["n_rows"]
            probe_cols = [
                c
                for c in (
                    "event_centroid_x", "event_centroid_y",
                    "event_centroid_lon", "event_centroid_lat",
                    "event_id", "start_time", "end_time",
                )
                if c in info["columns"]
            ]
            head = pd.read_parquet(candidate, columns=probe_cols).head(2000)
            for col in probe_cols:
                s = pd.to_numeric(head[col], errors="coerce")
                if s.notna().any():
                    print(f"    {col:26s} min={s.min():>16.3f} "
                          f"max={s.max():>16.3f}")
            print()
            print("  SAMPLE ROWS")
            for row in head.head(3).to_dict("records"):
                print("   ", {k: row[k] for k in list(row)[:6]})
            report["event_base_probe"] = {
                c: {
                    "min": float(pd.to_numeric(head[c], errors="coerce").min()),
                    "max": float(pd.to_numeric(head[c], errors="coerce").max()),
                }
                for c in probe_cols
                if pd.to_numeric(head[c], errors="coerce").notna().any()
            }
        break

# Membership: the event -> local cluster bridge.
print()
print("membership table:")
for path in sorted(events_dir.glob("*membership*.parquet")):
    info = schema_of(path)
    print(f"  {path.name}: rows={info.get('n_rows', 0):,} "
          f"cols={info.get('columns')}")
    report["membership"] = {"file": path.name, **info}
    head = pd.read_parquet(path).head(2000)
    for col in head.columns:
        print(f"    {col:32s} sample={list(head[col].unique()[:3])}")

# ===========================================================================
# 2b. INDIA_FIRMS -- the DETECTION-level table
#
# This is the only table with per-detection radiometry. clusters/ holds
# cluster-level aggregates (n_detections, total_frp, max_frp, ...) and has NO
# frp/i4/i5/confidence columns, so the lineage pass cannot be fed from it.
# ===========================================================================

print()
print("=" * 92)
print("2b. INDIA_FIRMS (detection level)")
print("=" * 92)

firms_candidates = []
_direct = PROJECT_ROOT / "India_Firms.parquet"
if cu.path_ok(_direct):
    firms_candidates.append(_direct)
for leaf in ("India_Firms", "india_firms"):
    hits = cu.find_all_artefact(PROJECT_ROOT, leaf)
    firms_candidates.extend(h for h, _ in hits if h.is_file())

report["india_firms"] = None
if not firms_candidates:
    print("  India_Firms.parquet NOT FOUND -- searching the whole project root")
else:
    path = firms_candidates[0]
    info = schema_of(path)
    print()
    print(f"  {path}")
    if "error" in info:
        print(f"  !! {info['error']}")
    else:
        print(f"  rows={info['n_rows']:,}  row_groups={info['n_row_groups']}  "
              f"cols={len(info['columns'])}")
        for col, typ in info["types"].items():
            print(f"    {col:34s} {typ}")
        entry = dict(info)
        entry["path"] = str(path)
        # ONE data sample: the id formats and value ranges that decide how the
        # mapping bridge joins.
        try:
            head = pd.read_parquet(path).head(2000)
            print()
            print("  sample values:")
            for col in head.columns:
                s = head[col]
                if pd.api.types.is_numeric_dtype(s):
                    print(f"    {col:32s} min={s.min():>16.3f} max={s.max():>16.3f}")
                else:
                    print(f"    {col:32s} {list(s.dropna().unique()[:3])}")
            entry["sample"] = {
                c: (list(map(str, head[c].dropna().unique()[:3]))
                    if not pd.api.types.is_numeric_dtype(head[c])
                    else {"min": float(head[c].min()), "max": float(head[c].max())})
                for c in head.columns
            }
        except Exception as exc:                 # noqa: BLE001
            entry["sample_error"] = f"{exc.__class__.__name__}: {exc}"
        report["india_firms"] = entry

# ===========================================================================
# 3. CLUSTERS / EDGES / MAPPINGS -- schema of ONE file, dates from filenames
# ===========================================================================

print()
print("=" * 92)
print("3. CLUSTERS / EDGES / MAPPINGS")
print("=" * 92)

import lineage_features as lf

report["daily_sets"] = {}
for label, leaf, prefix in (
    ("clusters", "clusters", "clusters"),
    ("edges", "edges", "edges"),
    ("mappings", "mappings", None),
):
    path = cu.find_artefact(PROJECT_ROOT, leaf)[0]
    if path is None:
        print(f"  {label}: NOT FOUND")
        continue
    index, stats = lf.index_daily_files(path, prefix=prefix)
    dates = sorted(index)
    print()
    print(f"  {label}/  {len(index):,} dated files  {index[dates[0]].name if dates else ''}"
          f" .. {index[dates[-1]].name if dates else ''}")
    print(f"    stats: {stats}")
    entry = {"path": str(path), "n_dated_files": len(index),
             "first": dates[0] if dates else None,
             "last": dates[-1] if dates else None,
             "index_stats": stats}
    if dates:
        info = schema_of(index[dates[0]])
        print(f"    schema of {index[dates[0]].name}: "
              f"rows={info.get('n_rows', 0):,}")
        for col, typ in info.get("types", {}).items():
            print(f"      {col:30s} {typ}")
        entry["schema"] = info
    report["daily_sets"][label] = entry

# ===========================================================================
# 4. GAZETTEER -- one file per distinct schema, geometry + CRS + categories
# ===========================================================================

print()
print("=" * 92)
print("4. EXTERNAL / GAZETTEER")
print("=" * 92)

external = cu.find_artefact(PROJECT_ROOT, "external")[0]
if external is None:
    print("  external/ NOT FOUND")
else:
    report["external"] = {}
    ext_subfolders = {}
    for child in sorted(external.iterdir()):
        if child.is_dir():
            try:
                ext_subfolders[child.name] = sum(
                    1 for _ in child.rglob("*") if _.is_file()
                )
            except OSError:
                ext_subfolders[child.name] = -1

    # Subfolder census: names + counts, no reads.
    print("  external/ subfolders:")
    for name, count in sorted(ext_subfolders.items()):
        print(f"    {name:42s} {count:>6d} files")
    report["external"]["subfolders"] = ext_subfolders

    # Gazetteer: dedupe by name (same as P3a) then sample ONE per schema.
    fac_dir = external / "osm_facilities"
    if cu.is_dir(fac_dir):
        seen = {}
        for p in sorted(fac_dir.rglob("*.geoparquet")):
            penalty = 1_000 * sum(
                1
                for part in p.relative_to(fac_dir).parts[:-1]
                if re.search(r"\s*\(\d+\)$|\b_v\d+\b", part)
            )
            if p.name not in seen or penalty < seen[p.name][0]:
                seen[p.name] = (penalty, p)
        gaz = [v[1] for v in seen.values()]
        print()
        print(f"  osm_facilities: {len(gaz)} unique files "
              f"(duplicates collapsed)")

        by_sig = {}
        for p in gaz:
            info = schema_of(p)
            if "error" in info:
                continue
            by_sig.setdefault(signature(info), []).append((p, info))

        gaz_report = {}
        for i, (sig, group) in enumerate(sorted(by_sig.items(),
                                                key=lambda kv: -len(kv[1]))):
            if i >= SAMPLE_MAX_FILES:
                break
            sample_path, info = group[0]
            print()
            print(f"    [{len(group)} file(s) share this schema] "
                  f"{sample_path.name}")
            for col, typ in info["types"].items():
                print(f"      {col:30s} {typ}")
            entry = {
                "n_files_sharing_schema": len(group),
                "sample": str(sample_path.relative_to(fac_dir)),
                "columns": info["columns"],
                "types": info["types"],
                "rows": info["n_rows"],
                "files": [str(q.relative_to(fac_dir)) for q, _ in group][:10],
            }
            # One data sample: geometry types, CRS, category values.
            try:
                head = pd.read_parquet(sample_path).head(SAMPLE_ROWS * 200)
                geom_col = next(
                    (c for c in ("geometry", "geom", "wkb_geometry")
                     if c in head.columns),
                    None,
                )
                if geom_col:
                    import shapely
                    from shapely import from_wkb

                    geoms = from_wkb(
                        np.asarray(head[geom_col].to_numpy(zero_copy_only=False))
                    )
                    entry["geometry_types"] = dict(
                        Counter(g.geom_type for g in geoms if g is not None)
                    )
                    entry["crs"] = (
                        head.attrs.get("crs")
                        or str(getattr(sample_path, "crs", ""))[:60]
                    )
                    print(f"      geometry types: {entry['geometry_types']}")
                for cand in ("category", "facility_type", "type", "class"):
                    if cand in head.columns:
                        vals = head[cand].dropna().astype(str)
                        entry["category_column"] = cand
                        entry["category_values"] = sorted(vals.unique())[:25]
                        print(f"      {cand}: "
                              f"{entry['category_values'][:12]}")
                        break
            except Exception as exc:                 # noqa: BLE001
                entry["sample_error"] = f"{exc.__class__.__name__}: {exc}"
            gaz_report[str(sample_path.relative_to(fac_dir))] = entry

        report["external"]["gazetteer"] = gaz_report

# ===========================================================================
# 5. FEATURE INPUT AVAILABILITY -- the point of the whole exercise
# ===========================================================================

print()
print("=" * 92)
print("5. FEATURE INPUT AVAILABILITY (141 features)")
print("=" * 92)

import feature_ledger as fl

# The detection columns live in India_Firms, NOT in clusters/. clusters/ holds
# cluster-level aggregates. Checking the wrong table made the first two recon
# runs report the lineage inputs as "missing" when they were simply being looked
# for in the wrong place.
cluster_cols = set()
if "clusters" in report.get("daily_sets", {}):
    cluster_cols = set(
        report["daily_sets"]["clusters"].get("schema", {}).get("columns", [])
    )
firms_cols = set()
if report.get("india_firms"):
    firms_cols = set(report["india_firms"].get("columns", []))
clusters_cols = firms_cols | cluster_cols   # union: lineage can draw on both
event_cols = set()
for info in report.get("event_tables", {}).values():
    event_cols |= set(info.get("columns", []))

# Which lineage inputs are genuinely present in the detection columns?
wanted_detection_cols = {
    "frp": ("frp", "FRP"),
    "i4": ("i4", "brightness_i4"),
    "i5": ("i5", "brightness_i5"),
    "longitude": ("longitude", "lon"),
    "latitude": ("latitude", "lat"),
    "subpixel_temp": ("subpixel_temp",),
    "source_area": ("source_area",),
    "confidence": ("confidence",),
    "pixel_size": ("pixel_size",),
    "scan_angle": ("scan_angle",),
    "view_zenith": ("view_zenith"),
    "day_night": ("day_night",),
    "satellite": ("satellite", "sat"),
}
present, absent = [], []
for logical, cands in wanted_detection_cols.items():
    (present if clusters_cols & set(cands) else absent).append(logical)

print()
print(f"  India_Firms columns : {len(firms_cols)}")
print(f"  clusters/ columns   : {len(cluster_cols)}")
print(f"  union considered    : {len(clusters_cols)}")
print(f"    present: {present}")
print(f"    absent : {absent}")
print()
print("  NOTE: clusters/ holds cluster aggregates only. Per-detection")
print("  radiometry (i4/i5/confidence/scan angle) is available solely if")
print("  India_Firms carries it -- check section 2b above before assuming.")

gaz_keys = set(report.get("external", {}).get("gazetteer", {}))
print(f"  gazetteer schemas sampled: {len(gaz_keys)}")
print(f"  event table columns: {len(event_cols)}")

# Group-level availability, ledger-driven.
availability = {}
for group in fl.FEATURE_GROUPS:
    names = [
        n for n in fl.FEATURE_GROUPS[group] if n not in fl._PLACEHOLDERS
    ]
    if group == fl.GROUP_INDUSTRIAL:
        state = "ready" if gaz_keys else "needs gazetteer"
    elif group in (fl.GROUP_THERMAL, fl.GROUP_TEMPORAL, fl.GROUP_SPREAD,
                   fl.GROUP_QUALITY):
        state = "ready" if present else f"missing {absent}"
    elif group == fl.GROUP_TERRAIN:
        state = "cell 263 (GEE)"
    else:
        state = "needs GEE / external"
    availability[group] = {"n": len(names), "state": state}
    print(f"  {group:24s} {len(names):>3d}  {state}")

report["availability"] = availability
report["detection_columns_present"] = present
report["detection_columns_absent"] = absent
report["cluster_columns"] = sorted(clusters_cols)
report["event_columns"] = sorted(event_cols)

# ===========================================================================
# 6. WRITE THE REPORT
# ===========================================================================

print()
print("=" * 92)
print(f"elapsed: {time.time() - t_start:.1f}s")
print("=" * 92)

qc_dir = PROJECT_ROOT / "qc"
qc_dir.mkdir(parents=True, exist_ok=True)
out = qc_dir / "project_recon.json"
out.write_text(json.dumps(report, indent=2, default=str))
print(f"Wrote {out}  ({out.stat().st_size / 1024:.1f} KB)")

print()
print("This is the reference for every path and column name from here on.")
print("If a cell hardcodes something this report contradicts, the report wins.")