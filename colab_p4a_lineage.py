# ============================================================================
# P4a -- COLAB CELL: 54 lineage features for every global fire event
#
# Run after the P3a industrial cell. Both are pure local arithmetic: no Earth
# Engine, no API quota, no new download.
#
# Input
#   event_local_cluster_membership.parquet   event_id -> local cluster ids
#   clusters/YYYYMMDD.parquet                5,337 daily detection files
#   India_Firms.parquet                      15,392,720 detections
#
# Strategy
#   The membership table is small (7.27M events). The detections are not. So
#   detections are read in date-batches, joined to events via the membership
#   map, and fed to lineage_features.compute_lineage_features() in event
#   chunks. Nothing larger than one batch is ever resident.
#
#   INSPECT_ONLY = True by default: prints the real column names of all three
#   inputs and the resolved mapping, and writes nothing. The extraction keys
#   off column names, so a wrong guess here would silently produce wrong
#   features rather than an error.
# ============================================================================

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import feature_ledger as fl
import lineage_features as lf
import colab_utils as cu

print("Ledger version :", fl.LEDGER_VERSION)
print("Lineage module :", lf.N_LINEAGE, "features")
print(cu.drive_report())
cu.require_drive("the cluster and membership tables")

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError(
        f"No directory under {cu.MYDRIVE} contains {cu.CANONICAL_SUBPATH}.\n"
        "Run colab_p3a_diagnose_external.py to locate the tree."
    )
# `clusters/` exists in TWO places on this Drive (Firm_Clusters/clusters and
# Firm_Clusters/full_2012_2026/clusters). Choose by file count, not by depth --
# the nesting in the handoff was never correct.
CLUSTERS_DIR, _clusters_table = cu.pick_by_file_count(PROJECT_ROOT, "clusters")
EVENTS_DIR, _events_table = cu.pick_by_file_count(PROJECT_ROOT, "events")
OUT_DIR = PROJECT_ROOT / "full_2012_2026" / "features" / "lineage"

print()
print("ARTEFACT CANDIDATES")
print("-" * 74)
for label, table in (("clusters", _clusters_table), ("events", _events_table)):
    print(f"  {label}/:")
    for row in table:
        print(f"    {row['n_files']:>7d} files  depth={row['depth']}  {row['path']}")
if CLUSTERS_DIR is None or EVENTS_DIR is None:
    raise FileNotFoundError(
        "Could not locate clusters/ or events/ under the project root. "
        "Run colab_survey_tree.py and paste the output."
    )
print("-" * 74)
print(f"  using clusters : {CLUSTERS_DIR}")
print(f"  using events   : {EVENTS_DIR}")

# The membership table name is confirmed by the inspection pass below.
MEMBERSHIP = None  # set after inspection

INSPECT_ONLY = True          # flip to False once the mapping below is confirmed
DATE_BATCH = 60              # days of clusters per read batch
EVENT_CHUNK = 250_000        # events per compute chunk
# Resolved by CONTENT, not by name -- see colab_p3a_industrial.py.
PROJECT_ROOT = None
CLUSTERS_DIR = None
MEMBERSHIP = None
OUT_DIR = None
QC_DIR = (PROJECT_ROOT / "qc") if PROJECT_ROOT is not None else Path("/content/qc")

# Event id / cluster id column names, confirmed by the inspection printout.
EVENT_ID_COL = "event_id"
CLUSTER_ID_COL = "local_cluster_id"


def require(path: Path, what: str) -> Path:
    if not cu.path_ok(path):
        raise FileNotFoundError(
            f"{what} not found at {path}. It is a frozen artefact -- restore "
            "it from Drive. Do NOT re-run build_full_global_events()."
        )
    return path


if not cu.is_dir(CLUSTERS_DIR):
    raise FileNotFoundError(f"clusters dir not usable: {CLUSTERS_DIR}")
if not cu.is_dir(EVENTS_DIR):
    raise FileNotFoundError(f"events dir not usable: {EVENTS_DIR}")

# ---------------------------------------------------------------------------
# 1. INSPECT -- real schemas, resolved mapping
# ---------------------------------------------------------------------------

print()
print("=" * 78)
print("P4a STEP 1 -- INPUT INSPECTION")
print("=" * 78)

# Membership file: try the known name in the resolved events dir, then any
# parquet there, so a rename does not need a code edit.
_MEMBERSHIP_NAMES = (
    "event_local_cluster_membership.parquet",
    "local_cluster_membership.parquet",
)
MEMBERSHIP = next(
    (EVENTS_DIR / n for n in _MEMBERSHIP_NAMES if cu.path_ok(EVENTS_DIR / n)),
    None,
)
if MEMBERSHIP is None:
    _any = sorted(EVENTS_DIR.glob("*.parquet"))
    print()
    print("No membership file under", EVENTS_DIR)
    print("parquet files present:")
    for p_ in _any[:20]:
        print("   ", p_.name)
    raise FileNotFoundError(
        "Set MEMBERSHIP by hand once you know which file holds the "
        "event -> local cluster mapping, then re-run."
    )
print("membership file :", MEMBERSHIP.name)

membership = pq.read_table(MEMBERSHIP)
print(f"membership rows : {membership.num_rows:,}")
print(f"membership cols : {membership.column_names}")

mem_cols = {c.lower(): c for c in membership.column_names}
for candidate in (EVENT_ID_COL, "global_event_id", "event"):
    if candidate in mem_cols:
        EVENT_ID_COL = mem_cols[candidate]
        break
for candidate in (CLUSTER_ID_COL, "cluster_id", "local_cluster"):
    if candidate in mem_cols:
        CLUSTER_ID_COL = mem_cols[candidate]
        break
print()
print(f"  event id column   : {EVENT_ID_COL}")
print(f"  local cluster col : {CLUSTER_ID_COL}")

cu.require_drive()

# Filenames are `clusters_2012-01-20.parquet` -- ISO dates with dashes, NOT the
# `YYYYMMDD_<label>` form the local cluster ids use. Building a path from an id
# would silently find nothing, so the index is discovered from the directory.
CLUSTER_INDEX, cluster_stats = lf.index_daily_files(CLUSTERS_DIR, prefix="clusters")
print()
print(f"daily cluster files indexed: {len(CLUSTER_INDEX):,}")
print(f"  {cluster_stats}")
if not CLUSTER_INDEX:
    raise FileNotFoundError(
        f"No dated cluster files under {CLUSTERS_DIR}. Sample names:\n"
        + "\n".join(
            f"   {p.name}" for p in sorted(CLUSTERS_DIR.glob("*.parquet"))[:5]
        )
    )
ordered_dates = sorted(CLUSTER_INDEX)
print(f"  first: {ordered_dates[0]}  ->  {CLUSTER_INDEX[ordered_dates[0]].name}")
print(f"  last : {ordered_dates[-1]}  ->  {CLUSTER_INDEX[ordered_dates[-1]].name}")

sample_path = CLUSTER_INDEX[ordered_dates[0]]
sample = pq.ParquetFile(sample_path)
detection_columns = sample.schema_arrow.names
print()
print(f"detection columns ({len(detection_columns)}):")
for name in detection_columns:
    print(f"  {name}")

probe = pd.read_parquet(sample_path)
resolved = lf.resolve_columns(probe)
print()
print("RESOLVED LINEAGE MAPPING")
print("-" * 78)
missing_required = []
for logical, actual in resolved.items():
    mark = "ok " if actual else "ABSENT"
    print(f"  {logical:16s} {mark:6s} {actual or ''}")
    if actual is None and logical in ("event_id", "frp", "longitude", "latitude"):
        missing_required.append(logical)

# The cluster files must expose a coordinate pair. Local cluster ids encode the
# date, so the centroid normally lives in the file; if not, it must be joined
# from India_Firms and this cell must not guess.
if missing_required:
    print()
    print("=" * 78)
    print("CANNOT PROCEED -- required columns unresolved")
    print("=" * 78)
    print(f"missing: {missing_required}")
    print()
    print("Set the *_COL constants at the top of this cell to the real names")
    print("printed above, then re-run. Do not guess: a wrong column name here")
    print("produces wrong features, not an error.")

if INSPECT_ONLY:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    with open(QC_DIR / "p4a_input_schema.json", "w") as fh:
        json.dump(
            {
                "membership_columns": membership.column_names,
                "event_id_col": EVENT_ID_COL,
                "cluster_id_col": CLUSTER_ID_COL,
                "n_cluster_files": len(CLUSTER_INDEX),
                "cluster_file_index_stats": cluster_stats,
                "cluster_first": ordered_dates[0],
                "cluster_last": ordered_dates[-1],
                "detection_columns": detection_columns,
                "resolved_mapping": {k: v for k, v in resolved.items()},
                "missing_required": missing_required,
            },
            fh,
            indent=2,
        )
    print()
    print("Wrote", QC_DIR / "p4a_input_schema.json")
    print("INSPECTION COMPLETE -- nothing written. Set INSPECT_ONLY = False to run.")
    # No SystemExit: IPython renders it as a traceback reading like a
    # crash. Returning normally is enough; nothing follows in this cell.
    pass

# ===========================================================================
# 2. COMPUTE
# ===========================================================================

print()
print("=" * 78)
print("P4a STEP 2 -- EVENT -> LOCAL CLUSTER MAP")
print("=" * 78)

t0 = time.time()
event_ids = membership.column(EVENT_ID_COL).to_numpy()
cluster_ids = membership.column(CLUSTER_ID_COL).to_numpy()
n_events = len(np.unique(event_ids))
print(f"membership pairs : {len(event_ids):,}")
print(f"distinct events  : {n_events:,}")
print(f"load time        : {time.time() - t0:.1f}s")

# The membership table carries an explicit `date` column (confirmed by recon:
# columns event_id, local_cluster_id, date). Prefer it over parsing `YYYYMMDD`
# out of the cluster id -- a dedicated column cannot drift from the id format.
DATE_SOURCE = None
for _cand in ("date", "acq_date", "cluster_date", "event_date"):
    if _cand in membership.column_names:
        _idx = membership.column_names.index(_cand)
        _raw = membership.column(_idx).to_pandas()
        dates = pd.to_datetime(_raw, errors="coerce").dt.strftime("%Y-%m-%d")
        DATE_SOURCE = _cand
        print(f"  membership date column : {DATE_SOURCE}")
        break

if DATE_SOURCE is None:
    # Fall back to the id prefix. Must be 8 DIGITS, not merely 8 characters: a
    # malformed id would otherwise become a NaT that silently drops a day.
    print("  no membership date column; deriving from the cluster id prefix")
    cluster_series = pd.Series(cluster_ids.astype(str))
    raw_dates = cluster_series.str.slice(0, 8)
    _well_formed = raw_dates.str.fullmatch(r"\d{8}")
    dates = pd.to_datetime(
        raw_dates.where(_well_formed), format="%Y%m%d", errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    _malformed = int((~_well_formed).sum())
    if _malformed:
        print(f"  {int(_well_formed.sum()):,} well-formed ids; "
              f"{_malformed:,} unparseable (first: "
              f"{list(cluster_series[~_well_formed].unique()[:3])})")
unique_dates = sorted(d for d in dates.dropna().unique())
print(f"distinct dates   : {len(unique_dates):,}")

# Only days that actually have a file on disk.
_all_dates = set(dates.dropna())
unique_dates = sorted(d for d in _all_dates if d in CLUSTER_INDEX)
missing_days = sorted(_all_dates - set(CLUSTER_INDEX))
if missing_days:
    print(f"  {len(missing_days)} membership date(s) have no cluster file, "
          f"e.g. {missing_days[:3]} -- those detections will be skipped")
print(f"dates to read    : {len(unique_dates):,}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

feature_frames = []
null_counts = {name: 0 for name in lf.LINEAGE_FEATURES}
rows_total = 0

for batch_no, start in enumerate(range(0, len(unique_dates), DATE_BATCH)):
    window = unique_dates[start:start + DATE_BATCH]
    wanted = set(window)

    # Which local clusters fall inside this date window?
    mask = dates.isin(wanted).to_numpy()
    if not mask.any():
        continue

    batch_events = pd.DataFrame(
        {
            EVENT_ID_COL: event_ids[mask],
            CLUSTER_ID_COL: cluster_ids[mask].astype(str),
        }
    )

    detections = []
    for day in window:
        path = CLUSTER_INDEX[day]
        if not cu.path_ok(path):
            print(f"  !! unreadable {path.name}")
            continue
        detections.append(pq.read_table(path))
    if not detections:
        continue
    n_day_files = len(detections)

    import pyarrow as pa

    det = pa.concat_tables(detections).to_pandas()
    del detections

    # Join detections to their event through the local cluster id.
    #
    # The two sides may name that column differently (membership uses
    # `local_cluster_id`, the cluster files may use `cluster_id`), so it is
    # detected on the detections side explicitly and renamed to match.
    det_cols = {c.lower(): c for c in det.columns}
    det_key = next(
        (
            det_cols[c]
            for c in (
                CLUSTER_ID_COL.lower(),
                "local_cluster_id",
                "cluster_id",
                "local_cluster",
                "cluster",
            )
            if c in det_cols
        ),
        None,
    )
    if det_key is None:
        raise KeyError(
            f"No local-cluster column in the detection files. Columns present: "
            f"{sorted(det.columns)[:20]}. Set DET_KEY below to the right name."
        )
    if det_key != CLUSTER_ID_COL:
        det = det.rename(columns={det_key: CLUSTER_ID_COL})

    merged = batch_events.merge(
        det, on=CLUSTER_ID_COL, how="inner", suffixes=("", "_det")
    )
    del det

    # A zero-row merge is the signature of a mismatched key: either the id
    # formats differ or the membership table references days we did not load.
    # Failing loudly here matters, because an empty chunk appended to the
    # feature frames looks exactly like a successful run with no output.
    if merged.empty:
        print(
            f"  !! batch {batch_no}: join produced 0 rows. "
            f"{len(batch_events):,} membership ids matched nothing across "
            f"{n_day_files} day file(s). Check that the membership "
            f"cluster ids match {CLUSTER_ID_COL} in the cluster files."
        )
        continue

    computed = lf.compute_lineage_features(merged, event_col=EVENT_ID_COL)
    feature_frames.append(computed.reset_index())
    rows_total += len(computed)

    for name in lf.LINEAGE_FEATURES:
        null_counts[name] += int(computed[name].isna().sum())

    if batch_no % 5 == 0:
        print(
            f"  batch {batch_no:>3d}  dates={len(window):>3d}  "
            f"events={rows_total:>10,}  {time.time() - t0:>7.1f}s"
        )

print()
print(f"events processed: {rows_total:,}")
print(f"wall time       : {time.time() - t0:.1f}s")

# ---------------------------------------------------------------------------
# 3. WRITE + QC
# ---------------------------------------------------------------------------

features = pd.concat(feature_frames, ignore_index=True)
del feature_frames
features = features.drop_duplicates(subset=[EVENT_ID_COL])

# Contract: exactly the 54 lineage columns, in ledger order.
assert list(features.columns) == [EVENT_ID_COL] + lf.LINEAGE_FEATURES, (
    "output schema drifted from the ledger"
)
print()
print(f"output rows     : {len(features):,}")
print(f"output columns  : {len(features.columns)} (event id + 54 features)")

path = OUT_DIR / "lineage_features.parquet"
features.to_parquet(path, index=False, compression="snappy")
print("Wrote", path)

print()
print("=" * 78)
print("P4a QC -- NaN fraction per feature")
print("=" * 78)
summary = {}
for name in lf.LINEAGE_FEATURES:
    frac = null_counts[name] / rows_total if rows_total else float("nan")
    summary[name] = frac
    flag = ""
    if frac > 0.5:
        flag = "  <-- mostly unavailable, check the column mapping"
    print(f"  {name:32s} {frac:>7.2%}{flag}")

manifest = {
    "event_id_col": EVENT_ID_COL,
    "cluster_id_col": CLUSTER_ID_COL,
    "events_in_membership": int(n_events),
    "events_processed": int(rows_total),
    "date_batch": DATE_BATCH,
    "event_chunk": EVENT_CHUNK,
    "feature_order": lf.LINEAGE_FEATURES,
    "nan_fraction": summary,
    "deferred_to_p3g": list(lf.DEFERRED_TO_P3G),
    "needs_external_input": list(lf.NEEDS_EXTERNAL_INPUT),
}
with open(OUT_DIR / "manifest.json", "w") as fh:
    json.dump(manifest, fh, indent=2)
print("Wrote", OUT_DIR / "manifest.json")

print()
print("NEXT")
print("  1. Read the NaN table above. Anything above ~50% means a column name")
print("     resolved to the wrong thing -- fix the mapping and re-run.")
print("  2. MISSING_FEATURE_FRACTION is still NaN: it is the fraction of the")
print("     other 140 features missing per event, so it can only be computed")
print("     after P3a + weather are joined. Do not fill it with 0.")
print("  3. Join P3a + lineage -> 72 features per event. Weather/terrain next.")
print("  4. runtime_verified stays False until the P2 10-event test runs.")