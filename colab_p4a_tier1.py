# ============================================================================
# P4a TIER 1 -- 34 lineage features for every event, from clusters/ + membership
#
# This is the cheapest real feature pass in the project: it reads 532 MB of
# cluster-level aggregates and never touches the 15.4M-row detection table.
#
# Inputs
#   clusters/clusters_YYYY-MM-DD.parquet          5,337 files, 532 MB
#       date local_cluster_id start_time end_time n_detections centroid_x/y
#       total_frp mean_frp max_frp n_observation_times duration_hours
#       satellites n_satellites max_radius_m max_radius_km centroid_lon/lat
#   events/event_local_cluster_membership.parquet  7,589,305 rows
#       event_id  local_cluster_id  date
#
# Output
#   features/lineage/lineage_tier1_chunk_NNNN.parquet
#       event_id + the 34 CLUSTER_TIER_FEATURES, ledger order
#
# INSPECT_ONLY = True by default: prints the real schemas, verifies the join on
# one small window, writes nothing. A join that silently matches nothing looks
# exactly like a successful run, so that has to be checked before the full pass.
# ============================================================================

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import cluster_tier as ct
import colab_utils as cu
import feature_ledger as fl
import lineage_features as lf

INSPECT_ONLY = False
DATE_BATCH = 60           # days of clusters per read
EVENT_CHUNK = 400_000     # events per written chunk
VERIFY_DAYS = 3           # days used for the pre-flight join check

MODE = "INSPECT ONLY (writes nothing)" if INSPECT_ONLY else "FULL PASS (writes)"
print("=" * 74)
print(f"ledger {fl.LEDGER_VERSION}   lineage {lf.N_LINEAGE}   "
      f"tier1 {ct.N_CLUSTER_TIER}")
print(f"MODE: {MODE}")
print("=" * 74)
print(cu.drive_report())

# ---------------------------------------------------------------------------
# 1. RESOLVE INPUTS
# ---------------------------------------------------------------------------

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError(
        f"No directory under {cu.MYDRIVE} contains {cu.CANONICAL_SUBPATH}."
    )
_CANON = PROJECT_ROOT / "full_2012_2026"

CLUSTERS_DIR, _ = cu.pick_by_file_count(PROJECT_ROOT, "clusters")
EVENTS_DIR, _ = cu.pick_by_file_count(PROJECT_ROOT, "events")
OUT_DIR = _CANON / "features" / "lineage"
QC_DIR = PROJECT_ROOT / "qc"

if CLUSTERS_DIR is None or EVENTS_DIR is None:
    raise FileNotFoundError("clusters/ or events/ not found under the project root")

MEMBERSHIP = next(
    (
        EVENTS_DIR / n
        for n in (
            "event_local_cluster_membership.parquet",
            "local_cluster_membership.parquet",
        )
        if cu.path_ok(EVENTS_DIR / n)
    ),
    None,
)
if MEMBERSHIP is None:
    print("parquet files under", EVENTS_DIR)
    for p in sorted(EVENTS_DIR.glob("*.parquet"))[:20]:
        print("   ", p.name)
    raise FileNotFoundError(
        "Set MEMBERSHIP by hand to the event<->cluster mapping, then re-run."
    )

print()
print("RESOLVED LOCATIONS")
print("-" * 74)
for label, path in (
    ("clusters", CLUSTERS_DIR),
    ("events", EVENTS_DIR),
    ("membership", MEMBERSHIP),
    ("output", OUT_DIR),
    ("qc", QC_DIR),
):
    flag = "ok " if cu.path_ok(path) else "new"
    print(f"  [{flag}] {label:12s} {path}")
print("-" * 74)

# ---------------------------------------------------------------------------
# 2. INDEX THE DAILY FILES
# ---------------------------------------------------------------------------

CLUSTER_INDEX, cluster_stats = lf.index_daily_files(CLUSTERS_DIR, prefix="clusters")
if not CLUSTER_INDEX:
    raise FileNotFoundError(f"No dated cluster files under {CLUSTERS_DIR}")
ordered_dates = sorted(CLUSTER_INDEX)
print()
print(f"daily cluster files : {len(CLUSTER_INDEX):,}  "
      f"{ordered_dates[0]} .. {ordered_dates[-1]}")
print(f"  {cluster_stats}")

# ---------------------------------------------------------------------------
# 3. MEMBERSHIP
# ---------------------------------------------------------------------------

t0 = time.time()
membership = pq.read_table(MEMBERSHIP).to_pandas()
print()
print(f"membership rows : {len(membership):,}  "
      f"cols={list(membership.columns)}  ({time.time() - t0:.1f}s)")

MEM_EVENT_COL = next(
    (c for c in ("event_id", "global_event_id") if c in membership.columns), None
)
MEM_CLUSTER_COL = next(
    (c for c in ("local_cluster_id", "cluster_id") if c in membership.columns), None
)
if MEM_EVENT_COL is None or MEM_CLUSTER_COL is None:
    raise KeyError(
        f"membership needs an event id and a cluster id; has "
        f"{list(membership.columns)}"
    )
MEM_DATE_COL = next(
    (c for c in ("date", "acq_date", "cluster_date") if c in membership.columns), None
)
print(f"  event col   : {MEM_EVENT_COL}")
print(f"  cluster col : {MEM_CLUSTER_COL}")
print(f"  date col    : {MEM_DATE_COL or '(none -- will derive from cluster id)'}")

n_events_total = membership[MEM_EVENT_COL].nunique()
print(f"  distinct events : {n_events_total:,}")

membership[MEM_CLUSTER_COL] = membership[MEM_CLUSTER_COL].astype(str)
member_index = membership.set_index(MEM_CLUSTER_COL)

# ---------------------------------------------------------------------------
# 4. SCHEMA + PRE-FLIGHT JOIN CHECK
# ---------------------------------------------------------------------------

sample_day = ordered_dates[0]
sample = pd.read_parquet(CLUSTER_INDEX[sample_day])
print()
print("=" * 74)
print(f"SAMPLE CLUSTER FILE  {CLUSTER_INDEX[sample_day].name}  "
      f"rows={len(sample):,}")
print("=" * 74)
for col, typ in pq.ParquetFile(CLUSTER_INDEX[sample_day]).schema_arrow.types and [
    (f.name, str(f.type)) for f in
    pq.ParquetFile(CLUSTER_INDEX[sample_day]).schema_arrow
]:
    print(f"  {col:26s} {typ}")

sample_cluster_col = next(
    (
        c
        for c in (
            MEM_CLUSTER_COL, "local_cluster_id", "cluster_id", "local_cluster"
        )
        if c in sample.columns
    ),
    None,
)
if sample_cluster_col is None:
    raise KeyError(
        f"No cluster id column in {sample_day}. Columns: {list(sample.columns)}"
    )
print()
print(f"  cluster id column : {sample_cluster_col}")

# THE check that matters: does the membership table actually cover these
# clusters? If it does not, every join below produces nothing while looking
# like a clean run.
sample_ids = sample[sample_cluster_col].astype(str)
hits = sample_ids.isin(member_index.index)
print()
print("=" * 74)
print("PRE-FLIGHT JOIN CHECK")
print("=" * 74)
print(f"  clusters in this file        : {len(sample_ids):,}")
print(f"  found in membership         : {int(hits.sum()):,}")
print(f"  unmatched                   : {int((~hits).sum()):,}")
if not hits.any():
    raise RuntimeError(
        f"ZERO of {len(sample_ids):,} cluster ids in {sample_day} appear in "
        f"{MEMBERSHIP.name}. The two tables are not joinable as-is -- check "
        "whether the id formats differ (e.g. '20120120_0' vs '2012-01-20_0')."
    )
if (~hits).any():
    print(f"  unmatched examples          : "
          f"{list(sample_ids[~hits])[:5]}")
    print(f"  membership examples         : "
          f"{list(member_index.index[:3])}")
    print("  -> a partial match is expected if this day is the first one;")
    print("     if the unmatched share is large, STOP and compare id formats.")

# ---------------------------------------------------------------------------
# 5. INSPECT-ONLY EXIT
# ---------------------------------------------------------------------------

if INSPECT_ONLY:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "clusters_dir": str(CLUSTERS_DIR),
        "n_daily_files": len(CLUSTER_INDEX),
        "first": ordered_dates[0],
        "last": ordered_dates[-1],
        "cluster_index_stats": cluster_stats,
        "membership": str(MEMBERSHIP),
        "membership_rows": int(len(membership)),
        "membership_event_col": MEM_EVENT_COL,
        "membership_cluster_col": MEM_CLUSTER_COL,
        "membership_date_col": MEM_DATE_COL,
        "distinct_events": int(n_events_total),
        "sample_file": str(CLUSTER_INDEX[sample_day].name),
        "sample_cluster_col": sample_cluster_col,
        "preflight_matched": int(hits.sum()),
        "preflight_unmatched": int((~hits).sum()),
        "output_features": ct.CLUSTER_TIER_FEATURES,
    }
    (QC_DIR / "p4a_tier1_schema.json").write_text(
        json.dumps(report, indent=2)
    )
    print()
    print(f"matched {int(hits.sum()):,} / {len(sample_ids):,}  "
          "-- join is viable")
    print(f"Wrote {QC_DIR / 'p4a_tier1_schema.json'}")
    print()
    print("INSPECTION COMPLETE -- nothing written to features/.")
    print("------------------------------------------------------------")
    print("STOPPED IN INSPECT MODE. Nothing was written.")
    print("To actually run: edit INSPECT_ONLY to False, then re-run.")
    print("------------------------------------------------------------")
    _STOP = True
else:
    _STOP = False

# ===========================================================================
# 6. FULL PASS
# ===========================================================================

def run_full_pass():
    """Execute the whole pass. Called only when INSPECT_ONLY is False."""
    print()
    print("=" * 74)
    print("P4a TIER 1 -- FULL PASS  (writing to features/lineage/)")
    print("=" * 74)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    batch_size = max(1, int(DATE_BATCH))

    null_counts = {name: 0 for name in ct.CLUSTER_TIER_FEATURES}
    rows_out = 0
    chunk_files = []
    events_seen = set()
    unmatched_clusters = 0
    clusters_read = 0

    for batch_no, start in enumerate(range(0, len(ordered_dates), batch_size)):
        window = ordered_dates[start:start + batch_size]

        tables = []
        for day in window:
            path = CLUSTER_INDEX[day]
            if not cu.path_ok(path):
                print(f"  !! unreadable {path.name}")
                continue
            tables.append(pq.read_table(path))
        if not tables:
            continue
        det = pa.concat_tables(tables).to_pandas()
        del tables

        clusters_read += len(det)
        det[sample_cluster_col] = det[sample_cluster_col].astype(str)

        # Keep only clusters the membership table knows about, and say how many
        # were dropped: an unexpected number here means the join keys disagree.
        known = det[sample_cluster_col].isin(member_index.index)
        n_unknown = int((~known).sum())
        if n_unknown:
            unmatched_clusters += n_unknown
            if batch_no == 0:
                print(f"  batch 0: {n_unknown:,} of {len(det):,} clusters are not "
                      f"in membership (expected only if this is the first day)")
        det = det[known]
        if det.empty:
            continue

        # The membership table carries its own `date`, which collides with the
        # cluster file's `date` and makes the join raise outright. Bring across
        # only what is actually needed.
        bring = [MEM_EVENT_COL]
        if MEM_DATE_COL and MEM_DATE_COL not in det.columns:
            bring.append(MEM_DATE_COL)
        elif MEM_DATE_COL:
            # Same name on both sides: keep the cluster file's own value and skip
            # the duplicate rather than failing on an overlapping column.
            print("  note: membership date column shares a name with the cluster "
                  "file; using the cluster file's value")

        joined = det.join(
            member_index[bring], on=sample_cluster_col, how="inner", rsuffix="_mem"
        )
        if joined.empty:
            print(f"  !! batch {batch_no}: join produced 0 rows -- STOP")
            break

        computed = ct.compute_cluster_tier_features(joined, event_col=MEM_EVENT_COL)
        # One row per event: an event spans many daily batches, so later batches
        # would otherwise re-emit events already written.
        fresh = computed[~computed.index.isin(events_seen)]
        if fresh.empty:
            continue
        events_seen.update(fresh.index.tolist())

        for name in ct.CLUSTER_TIER_FEATURES:
            null_counts[name] += int(fresh[name].isna().sum())

        out = fresh.reset_index()
        path_out = OUT_DIR / f"lineage_tier1_chunk_{batch_no:04d}.parquet"
        out.to_parquet(path_out, index=False, compression="snappy")
        chunk_files.append({"file": path_out.name, "rows": len(out)})
        rows_out += len(out)

        if batch_no % 5 == 0:
            print(f"  batch {batch_no:>4d}  days={len(window):>3d}  "
                  f"events={rows_out:>10,}  {time.time() - t0:>7.1f}s")

    # ---------------------------------------------------------------------------
    # 7. QC
    # ---------------------------------------------------------------------------

    print()
    print("=" * 74)
    print("P4a TIER 1 -- QC")
    print("=" * 74)
    print(f"cluster rows read      : {clusters_read:,}")
    print(f"clusters not in member : {unmatched_clusters:,}")
    print(f"events written         : {rows_out:,}  of {n_events_total:,} in "
          f"membership ({100.0 * rows_out / max(n_events_total, 1):.1f}%)")

    assert list(features_cols := [c for c in ct.CLUSTER_TIER_FEATURES]) == \
        ct.CLUSTER_TIER_FEATURES
    print(f"feature columns        : {len(features_cols)}")

    missing = n_events_total - rows_out
    if missing > 0:
        print()
        print(f"  NOTE: {missing:,} membership events produced no features. That is")
        print("  expected only if those events have no clusters on disk. Events with")
        print("  a single cluster still produce FRP/count/duration values; only the")
        print("  trend and shape features are NaN for them.")

    print()
    print(f"{'feature':32s} {'NaN %':>8s}")
    summary = {}
    for name in ct.CLUSTER_TIER_FEATURES:
        frac = null_counts[name] / rows_out if rows_out else float("nan")
        summary[name] = frac
        flag = "  <-- check" if frac > 0.6 and name not in ct.DEFERRED_REASONS else ""
        print(f"  {name:30s} {frac:>7.2%}{flag}")

    manifest = {
        "clusters_dir": str(CLUSTERS_DIR),
        "membership": str(MEMBERSHIP),
        "membership_rows": int(len(membership)),
        "events_in_membership": int(n_events_total),
        "events_written": int(rows_out),
        "cluster_rows_read": int(clusters_read),
        "clusters_not_in_membership": int(unmatched_clusters),
        "date_batch": batch_size,
        "tier": 1,
        "feature_order": ct.CLUSTER_TIER_FEATURES,
        "nan_fraction": summary,
        "deferred_features": list(ct.DEFERRED_REASONS),
        "chunks": chunk_files,
    }
    (OUT_DIR / "manifest_tier1.json").write_text(json.dumps(manifest, indent=2))
    print()
    print("Wrote", OUT_DIR / "manifest_tier1.json")

    print()
    print("NEXT")
    print("  1. Read the NaN table. A high NaN share on a trend feature usually")
    print("     means single-cluster events, which is correct. A high NaN share on")
    print("     FRP_MEDIAN or DETECTION_COUNT_LOG means a broken join.")
    print("  2. Run colab_p3a_industrial.py for the 18 industrial features.")
    print("  3. Join tier1 + industrial -> 52 features, then label and train.")
    print("  4. The 20 deferred features are NOT here by accident -- see")
    print("     cluster_tier.DEFERRED_REASONS for the reason each one is absent.")


if not _STOP:
    run_full_pass()
