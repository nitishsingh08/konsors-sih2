# ============================================================================
# P6 -- COLAB CELL: assemble the training table
#
# Joins the two completed feature blocks onto the event base and builds
# X (52 features), y (labels) and meta (ids, time, split keys).
#
# Enforces the brief's requirements rather than leaving them to discipline:
#   * raw coordinates never enter X
#   * a cloud-blocked observation stays a gap (NaN), never a zero
#   * anomaly is judged against the site's own baseline
#
# Runs in chunks: 7.27M events never fit in memory alongside three frames.
# ============================================================================

import gc
import json
import time
from dataclasses import asdict
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import cluster_tier as ct
import colab_utils as cu
import dataset_builder as db
import feature_gap as fg
import feature_ledger as fl
import fire_labels as lab
import weak_labels as weak
import industrial_features as ind
import rule_calibration as rc
import training_design as td

INSPECT_ONLY = True
SITE_DEG = db.SITE_DEG
# 16 partitions -> ~454k events per step. Peak memory is one partition plus the
# current source chunk, not the whole catalogue. The first attempt held both
# feature blocks AND every accumulated output chunk (~10 GB) and the runtime
# died at batch 12 of 15.
N_PARTITIONS = db.N_PARTITIONS

# Which radius reading of the weak hierarchy to apply. Reported both ways in the
# QC so the choice is made on the counts, not on assertion.
#: Target class sizes used to calibrate the persistence thresholds. Chosen so a
#: class stays rare but learnable: ~0.05% of 7.27M is about 3,600 events.
CALIB_TARGETS = {
    weak.INDUSTRIAL_FLARE: 3_600,
    weak.INDUSTRIAL_FIRE: 400_000,
    weak.MINING: 60_000,
}

#: Rows sampled per partition for calibration. 16 x 12,500 = 200,000, ample
#: for percentiles in the 99.5th to 99.95th range.
CALIB_SAMPLE_PER_PARTITION = 12_500

#: Columns the weak rules read, plus the derived inputs they need.
CALIB_INPUT_COLUMNS = (
    "DIST_REFINERY_LOG", "DIST_NEAREST_FACILITY_LOG", "FACILITY_COUNT_5KM",
    "DIST_POWERPLANT_LOG", "DIST_PETROCHEMICAL_LOG", "DIST_STEEL_LOG",
    "DIST_CEMENT_LOG", "DIST_MINE_LOG", "FRP_PERSISTENCE",
    "DETECTION_COUNT_LOG", "TIME_TO_PEAK", "AREA_GROWTH_RATE",
    "SPREAD_SPEED_MEDIAN", "DIRECTION_CONSISTENCY", "INSIDE_FACILITY",
    "KNOWN_FLARE",
)

print("=" * 74)
print(f"ledger {fl.LEDGER_VERSION}   tier1 {ct.N_CLUSTER_TIER}   "
      f"industrial {ind.N_INDUSTRIAL}   "
      f"labels {len(lab.LABEL_CLASSES)}")
print("=" * 74)
print(cu.drive_report())

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError(f"no project root under {cu.MYDRIVE}")
_CANON = PROJECT_ROOT / "full_2012_2026"
FEATURES_DIR = _CANON / "features"
EVENTS_DIR = cu.pick_by_file_count(PROJECT_ROOT, "events")[0]
QC_DIR = PROJECT_ROOT / "qc"
OUT_DIR = FEATURES_DIR / "dataset"

TIER1_FILES = sorted((FEATURES_DIR / "lineage").glob("lineage_tier1_chunk_*.parquet"))
IND_FILES = sorted((FEATURES_DIR / "industrial").glob("industrial_chunk_*.parquet"))

print()
print("RESOLVED INPUTS")
print("-" * 74)
print(f"  tier1 chunks     : {len(TIER1_FILES)}")
print(f"  industrial chunks: {len(IND_FILES)}")
print(f"  events dir       : {EVENTS_DIR}")
print(f"  output           : {OUT_DIR}")
print("-" * 74)

if not TIER1_FILES or not IND_FILES:
    raise FileNotFoundError(
        "Both feature blocks must exist before P6.\n"
        f"  tier1 chunks found     : {len(TIER1_FILES)}\n"
        f"  industrial chunks found: {len(IND_FILES)}\n"
        "Run colab_p4a_tier1.py and colab_p3a_industrial.py first."
    )

# ---------------------------------------------------------------------------
# The feature list is fixed here, from the ledger, not discovered from the
# files -- so a chunk that silently gained or lost a column cannot change X.
# ---------------------------------------------------------------------------

TIER1_EXPECTED = list(ct.CLUSTER_TIER_FEATURES)
INDUSTRIAL_EXPECTED = list(ind.INDUSTRIAL_FEATURES)
FEATURE_COLUMNS = TIER1_EXPECTED + INDUSTRIAL_EXPECTED

# FRP_ANOMALY_VS_SITE and the site baseline are added by this cell.
DERIVED_FEATURE_COLUMNS = [
    "SITE_FRP_BASELINE", "SITE_N_EVENTS", "FRP_ANOMALY_VS_SITE",
    "MISSING_FEATURE_FRACTION",
    # Regime flags. 19 tier-1 columns are >90% NaN because 96.7% of events
    # are single-cluster and have no trend or geometry to measure. Filling them
    # would invent measurements, but the model still needs to know WHICH REGIME
    # an event is in. These flags give it that explicitly instead of leaving it
    # to infer regime from a pattern of NaNs.
    "N_DETECTIONS", "IS_SINGLE_DETECTION", "N_DETECTIONS_GE3",
    "PERSISTENCE_IS_VACUOUS",
]

ALL_FEATURES = FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS

# Guard before anything is written: no coordinate may reach the model.
db.assert_no_location_features(ALL_FEATURES)
print()
print(f"feature matrix: {len(ALL_FEATURES)} columns "
      f"({len(FEATURE_COLUMNS)} computed + {len(DERIVED_FEATURE_COLUMNS)} derived)")
print("location check: PASS -- no coordinate column in X")

# Pre-flight memory estimate, so a crash is explained rather than mysterious.
_est_total = int(
    pq.ParquetFile(TIER1_FILES[0]).metadata.num_rows
) * len(TIER1_FILES)
_partition_rows = _est_total / float(N_PARTITIONS)
print()
print("MEMORY PLAN")
print("-" * 74)
print(f"  whole-catalogue X would need "
      f"{db.human_bytes(db.partition_bytes_estimate(_est_total, len(ALL_FEATURES)))}"
      "  -> not attempted")
print(f"  {N_PARTITIONS} partitions, ~{_partition_rows:,.0f} events each")
print(f"  per-partition working set ~"
      f"{db.human_bytes(db.partition_bytes_estimate(_partition_rows, len(ALL_FEATURES) * 2))}")
print("-" * 74)

if INSPECT_ONLY:
    print()
    print("--- tier1 chunk schema (first file) ---")
    print(pq.ParquetFile(TIER1_FILES[0]).schema_arrow)
    print("--- industrial chunk schema (first file) ---")
    print(pq.ParquetFile(IND_FILES[0]).schema_arrow)
    print()
    print("INSPECTION COMPLETE -- nothing written.")
    print("Set INSPECT_ONLY = False to assemble the dataset.")
    _STOP = True
else:
    _STOP = False


# ===========================================================================
def run_assembly():
    """Assemble X / y / meta by streaming, one partition at a time.

    Three phases, none of which holds more than one chunk plus one partition:

      1. repartition the event base, tier 1 and industrial into N parquet
         shards keyed on a stable CRC32 of event_id
      2. for each partition: read its three shards, derive, label, and append
         straight to the output parquet writers
      3. delete the shards

    Nothing accumulates in a Python list. The previous version held both full
    feature blocks and every output chunk, needing >10 GB and dying at batch 12.
    """
    import shutil

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_DIR = OUT_DIR / "_shards"
    if cu.path_ok(SHARD_DIR):
        shutil.rmtree(SHARD_DIR)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    base_path = next(
        (EVENTS_DIR / n for n in ("canonical_event_base.parquet",
                                   "event_base.parquet")
         if cu.path_ok(EVENTS_DIR / n)),
        None,
    )
    if base_path is None:
        raise FileNotFoundError(f"no event base parquet under {EVENTS_DIR}")
    print()
    print(f"event base: {base_path.name}")

    # geometry_n_satellites is required by the flare rule and lives ONLY in the
    # event base -- tier 1 does not carry it. Omitting it made every event
    # abstain, which is why the first assembly labelled 100% UNKNOWN.
    base_cols = [c for c in ("event_id", "event_centroid_lon",
                             "event_centroid_lat", "start_time",
                             "geometry_n_satellites")
                 if c in pq.ParquetFile(base_path).schema_arrow.names]
    if "geometry_n_satellites" not in base_cols:
        print("  !! geometry_n_satellites absent; the flare rule will abstain")

    # ---- phase 1: repartition every input into shards --------------------
    print()
    print("PHASE 1 -- repartition into shards")
    print("-" * 74)

    writers: dict = {}
    counts = Counter()

    def shard_writer(kind, part):
        key = (kind, part)
        if key not in writers:
            # LazyParquetWriter, not pq.ParquetWriter: the latter needs a schema
            # at construction, which is impossible for a shard that may never
            # receive a row.
            writers[key] = cu.LazyParquetWriter(
                SHARD_DIR / f"{kind}_p{part:02d}.parquet"
            )
        return writers[key]

    def write_shard(kind, frame):
        if frame.empty:
            return
        parts = db.partition_of(frame["event_id"].tolist(), N_PARTITIONS)
        for part in range(N_PARTITIONS):
            piece = frame[parts == part]
            if piece.empty:
                continue
            shard_writer(kind, part).write_frame(piece)
            counts[f"{kind}_p{part:02d}"] += len(piece)

    reader = pq.ParquetFile(base_path)
    n_base = 0
    for batch in reader.iter_batches(batch_size=500_000, columns=base_cols):
        frame = batch.to_pandas()
        n_base += len(frame)
        write_shard("base", frame)
        del frame
    print(f"  event base   : {n_base:,} rows -> {N_PARTITIONS} shards")

    for label, files in (("tier1", TIER1_FILES), ("industrial", IND_FILES)):
        n_rows = 0
        for path in files:
            frame = pd.read_parquet(path)
            n_rows += len(frame)
            write_shard(label, frame)
            del frame                      # released before the next file
        print(f"  {label:12s}: {n_rows:,} rows -> {N_PARTITIONS} shards")
        time.sleep(0)                      # yield, keeps the GC prompt

    for writer in writers.values():
        writer.close()
    # Drop shard writers whose partition never received a row: they would leave
    # empty files behind that the read phase would then try to open.
    # `wrote_anything`, NOT `opened`: close() above cleared the live writer, so
    # `opened` reads False for every shard and this deleted all sixteen. That
    # produced an empty PHASE 2 and a silent 0-event "success".
    empty = [
        f"{k[0]}_p{k[1]:02d}.parquet"
        for k, w in writers.items()
        if not w.wrote_anything
    ]
    if empty:
        print(f"  removed {len(empty)} empty shard(s)")
    for name in empty:
        (SHARD_DIR / name).unlink(missing_ok=True)
    writers.clear()
    shard_files = sorted(SHARD_DIR.glob("*.parquet"))
    shard_rows = sum(
        pq.ParquetFile(p).metadata.num_rows for p in shard_files
    )
    print(f"  shard dir: {SHARD_DIR}")
    print(f"  shards on disk: {len(shard_files)}  rows: {shard_rows:,}")
    if shard_rows < n_base:
        raise RuntimeError(
            f"PHASE 1 wrote {shard_rows:,} of {n_base:,} event-base rows. "
            "Refusing to continue: PHASE 2 would join against incomplete "
            "shards and report a smaller dataset as if it were complete."
        )

    # ---- phase 2: one partition at a time --------------------------------
    print()
    print("PHASE 2 -- build X / y / meta, one partition at a time")
    print("-" * 74)

    x_writers, y_writers, m_writers = {}, {}, {}
    calib_parts = []
    null_counts = {c: 0 for c in ALL_FEATURES}
    label_counts = Counter()
    # Distinct events blocked per missing input. A dict-of-sets, NOT a Counter:
    # summing per-string counts double-counted an input noted by two rules and
    # reported 14,531,698 blocked for a 7,265,849-row table.
    missing_pairs: dict = {}
    rule_counts = Counter()
    weak_counts = Counter()
    written = 0

    for part in range(N_PARTITIONS):
        base_p = SHARD_DIR / f"base_p{part:02d}.parquet"
        t1_p = SHARD_DIR / f"tier1_p{part:02d}.parquet"
        in_p = SHARD_DIR / f"industrial_p{part:02d}.parquet"
        if not (cu.path_ok(base_p) and cu.path_ok(t1_p) and cu.path_ok(in_p)):
            continue

        base_f = pd.read_parquet(base_p)
        t1_f = pd.read_parquet(t1_p)
        in_f = pd.read_parquet(in_p)

        joined = base_f.merge(t1_f, on="event_id", how="inner").merge(
            in_f, on="event_id", how="inner"
        )
        del base_f, t1_f, in_f
        if joined.empty:
            continue

        # derived
        flare_inputs = db.derive_flare_inputs(joined)
        # n_satellites comes straight from the event base.
        if "geometry_n_satellites" in joined.columns:
            flare_inputs["n_satellites"] = pd.to_numeric(
                joined["geometry_n_satellites"], errors="coerce"
            )
        # area_km2 is exactly invertible from the density: density = detections
        # per km2, and both inputs are present.
        joined = db.add_area_km2(joined)
        if "area_km2" in joined.columns:
            flare_inputs["area_km2"] = joined["area_km2"]
        joined = db.add_site_baseline(joined, site_deg=SITE_DEG)
        joined["FRP_ANOMALY_LOG"] = db.log_anomaly(joined["FRP_ANOMALY_VS_SITE"])
        joined["MISSING_FEATURE_FRACTION"] = db.missing_feature_fraction(
            joined, FEATURE_COLUMNS
        )
        joined = fg.add_single_detection_flags(joined)

        # labels
        label_out = lab.assign_labels(
            joined.assign(**flare_inputs.to_dict("series"))
        )
        # ---- calibration sample -------------------------------------------
        # Thresholds must be calibrated on the WHOLE catalogue, but the pass
        # runs one partition at a time, so the calibration inputs are sampled
        # here and consumed in PHASE 3. Systematic sampling (every Nth row) so
        # the sample is not biased toward any partition's date range.
        _need = [c for c in CALIB_INPUT_COLUMNS if c in joined.columns]
        if _need:
            _step = max(1, len(joined) // CALIB_SAMPLE_PER_PARTITION)
            calib_parts.append(joined.iloc[::_step][_need + ["event_id"]])

        # Two labellers, kept as separate columns.
        #   label_evidence : the strict, evidence-gated engine -- almost all
        #                    UNKNOWN, and defensible where it fires.
        #   label_weak     : the industrial-context rule hierarchy, which
        #                    labels far more but is manufactured from features
        #                    the model sees, so it is WEAK and carries its own
        #                    provenance. Never used to score model B.
        joined["label"] = label_out["label"]
        joined["label_rule"] = label_out["rule"]
        # Count DISTINCT events blocked by each input. Summing the per-string
        # Counter double-counts any input noted by two rules, which reported
        # 14,531,698 "events blocked" for a 7,265,849-event table.
        for event_id, entry in zip(joined["event_id"],
                                   label_out["missing_inputs"]):
            if entry:
                for nm in entry.split(","):
                    missing_pairs.setdefault(nm, set()).add(event_id)
        joined["label_confidence"] = label_out["confidence"]
        joined["sample_weight"] = lab.label_weights(
            label_out["label"], label_out["confidence"]
        )
        joined["missing_label_inputs"] = label_out["missing_inputs"]
        joined = td.add_split_keys(joined)

        for cls, n in Counter(label_out["label"]).items():
            label_counts[cls] += n
        for rule_name, n in Counter(label_out["rule"]).items():
            rule_counts[rule_name] += n
        for col in ALL_FEATURES:
            null_counts[col] += int(joined[col].isna().sum())

        # strip anything not in the declared feature list, then assert
        keep = [c for c in joined.columns if c in ALL_FEATURES]
        x_frame = joined[ALL_FEATURES]
        db.assert_no_location_features(list(x_frame.columns))

        # `y` is written in PHASE 3, after calibration.
        y_frame = joined[["event_id", "label", "label_rule",
                          "label_confidence", "sample_weight",
                          "missing_label_inputs"]]
        m_frame = joined[["event_id", "event_centroid_lon",
                          "event_centroid_lat", "start_time", "start_year",
                          "spatial_block", "site_id"]]
        del joined, keep

        for kind, frame, store in (
            ("X", x_frame, x_writers),
            ("meta", m_frame, m_writers),
        ):
            if kind not in store:
                store[kind] = cu.LazyParquetWriter(OUT_DIR / f"{kind}.parquet")
            store[kind].write_frame(frame)
            del frame

        written += len(x_frame)
        del x_frame, y_frame, m_frame, label_out, flare_inputs
        gc.collect()

        print(f"  partition {part:>2d}  rows={written:>9,}  "
              f"{time.time() - t0:>7.1f}s")

    for store in (x_writers, y_writers, m_writers):
        for writer in store.values():
            writer.close()
    # `y` is deliberately absent here: it is written in PHASE 3, after the
    # thresholds are calibrated. Reporting it as 0 at this point raised a false
    # "a write was lost" alarm on a completely successful run.
    assembled = {
        "X": x_writers["X"].rows_written if "X" in x_writers else 0,
        "meta": m_writers["meta"].rows_written if "meta" in m_writers else 0,
    }
    print()
    print(f"  written: X {assembled['X']:,}  meta {assembled['meta']:,}  "
          f"(y deferred to PHASE 3)")

    # A 0-event assembly must abort loudly. Reporting "NEXT: colab_p7_train.py"
    # after writing empty parquet files wastes a whole cycle and reads as
    # success.
    if assembled["X"] == 0:
        raise RuntimeError(
            "PHASE 2 assembled 0 events. The shards exist but the "
            "base/tier1/industrial join matched nothing. Check that the event "
            "id values are identical across all three inputs -- this ran to "
            "completion and printed NEXT, which is exactly the failure mode "
            "that must not pass silently."
        )
    for kind, n in assembled.items():
        if n != assembled["X"]:
            print(f"  !! {kind}.parquet has {n:,} rows, expected "
                  f"{assembled['X']:,} -- a write was lost")
    if assembled["X"] != written:
        print(f"  !! {written:,} events were processed but only "
              f"{assembled['X']:,} reached X.parquet")

    # =================================================================
    # PHASE 3 -- calibrate the weak thresholds, then label
    #
    # Calibration needs the whole distribution, but the pass is partitioned, so
    # it happens between PHASE 2 and the label pass. Without this the
    # persistence thresholds stayed tied to the pre-fix scale: INDUSTRIAL_FLARE
    # fell 3,571 -> 0 and MINING 73,060 -> 61 in one re-run.
    # =================================================================
    print()
    print("=" * 74)
    print("PHASE 3 -- CALIBRATE THRESHOLDS, THEN LABEL")
    print("=" * 74)

    calib_sample = (
        pd.concat(calib_parts, ignore_index=True)
        if calib_parts
        else pd.DataFrame()
    )
    print(f"  calibration sample: {len(calib_sample):,} rows")

    print()
    print("  FRP_PERSISTENCE distribution (the input that moved scale):")
    print(rc.distribution(
        calib_sample, ["FRP_PERSISTENCE"]).to_string(index=False))

    thresholds, evidence = rc.calibrate_to_target_counts(
        calib_sample, CALIB_TARGETS, base=weak.WIDE)
    calib_table, calib_report, _ = rc.calibration_report(
        calib_sample, thresholds, evidence)
    print()
    print("  calibrated thresholds:")
    print(calib_table.to_string(index=False))
    print()
    print("  weak labels on the SAMPLE, before vs after calibration:")
    print(calib_report.to_string(index=False))

    (QC_DIR / "p6_rule_calibration.json").write_text(json.dumps(
        {"targets": CALIB_TARGETS,
         "thresholds": rc.thresholds_to_dict(thresholds),
         "evidence": [asdict(e) for e in evidence]},
        indent=2, default=str))
    print(f"  Wrote {QC_DIR}/p6_rule_calibration.json")

    # Label every partition with the calibrated thresholds.
    print()
    print("  labelling all partitions with the calibrated thresholds ...")
    y_writer = None
    labelled = 0
    for part in range(N_PARTITIONS):
        base_p = SHARD_DIR / f"base_p{part:02d}.parquet"
        t1_p = SHARD_DIR / f"tier1_p{part:02d}.parquet"
        in_p = SHARD_DIR / f"industrial_p{part:02d}.parquet"
        if not (cu.path_ok(base_p) and cu.path_ok(t1_p) and cu.path_ok(in_p)):
            continue
        j = (
            pd.read_parquet(base_p)
            .merge(pd.read_parquet(t1_p), on="event_id", how="inner")
            .merge(pd.read_parquet(in_p), on="event_id", how="inner")
        )
        if j.empty:
            continue
        lo = lab.assign_labels(j)
        wo = weak.assign_weak_labels(j, thresholds)
        for cls, n in Counter(wo["label"]).items():
            weak_counts[cls] += n
        yf = pd.DataFrame({
            "event_id": j["event_id"].to_numpy(),
            "label": lo["label"],
            "label_rule": lo["rule"],
            "label_confidence": lo["confidence"],
            "sample_weight": lab.label_weights(lo["label"], lo["confidence"]),
            "label_weak": wo["label"],
            "label_weak_rule": wo["rule"],
            "label_weak_source": wo["label_source"],
            "missing_label_inputs": lo["missing_inputs"],
        })
        if y_writer is None:
            y_writer = cu.LazyParquetWriter(OUT_DIR / "y.parquet")
        y_writer.write_frame(yf)
        labelled += len(yf)
        if part % 5 == 0:
            print(f"    partition {part:>2d}  labelled={labelled:>9,}")

    if y_writer is not None:
        y_writer.close()
    print(f"  y.parquet rows: {labelled:,}")
    if labelled != assembled["X"]:
        print(f"  !! y has {labelled:,} rows, expected "
              f"{assembled['X']:,} -- the label pass lost rows")

    # ---- phase 4: drop the shards ---------------------------------------
    shutil.rmtree(SHARD_DIR, ignore_errors=True)
    print()
    print(f"  shards removed; assembled {written:,} events")

    # ---- QC --------------------------------------------------------------
    print()
    print("=" * 74)
    print("P6 QC")
    print("=" * 74)
    total = max(written, 1)
    print(f"  events assembled : {written:,}  (event base had {n_base:,})")
    if written != n_base:
        print(f"  !! {n_base - written:,} events lost in the join")
    print()
    print("  labels")
    for cls in lab.LABEL_CLASSES:
        n = label_counts.get(cls, 0)
        print(f"    {cls:20s} {n:>10,}  {100.0 * n / total:>6.2f}%")
    _unknown = label_counts.get(lab.UNKNOWN, 0)
    print(f"    {'supervised':20s} {total - _unknown:>10,}  "
          f"{100.0 * (total - _unknown) / total:>6.2f}%")
    print()
    print("  firing rules")
    for rule_name, n in sorted(rule_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {rule_name:28s} {n:>10,}")
    print()
    print()
    print("  WEAK LABELS, CALIBRATED thresholds (industrial-context")
    print("  hierarchy; manufactured from features the model sees, so")
    print("  never used to score model B)")
    for cls in weak.WEAK_CLASSES:
        n = weak_counts.get(cls, 0)
        print(f"    {cls:20s} {n:>10,}  {100.0 * n / total:>6.2f}%")
    print()
    print("  WHY LABELS ABSTAIN -- rule inputs that were never available")
    print("  (this is the list to implement; each line is one fix)")
    if missing_pairs:
        for nm in sorted(missing_pairs, key=lambda k: -len(missing_pairs[k])):
            n = len(missing_pairs[nm])
            print(f"    {nm:32s} {n:>10,} of {total:,} events blocked "
                  f"({100.0 * n / total:.1f}%)")
    else:
        print("    (none -- every rule input was available)")
    print()
    print("  feature NaN%")
    for col in ALL_FEATURES:
        print(f"    {col:32s} {100.0 * null_counts[col] / total:>6.2f}%")

    dense = sum(1 for c in ALL_FEATURES
                if null_counts[c] / total < 0.5)
    print()
    print(f"  features >50% populated : {dense} of {len(ALL_FEATURES)}")

    manifest = {
        "n_events": int(written),
        "n_event_base": int(n_base),
        "n_partitions": N_PARTITIONS,
        "features": list(ALL_FEATURES),
        "label_counts": {c: int(label_counts.get(c, 0))
                         for c in lab.LABEL_CLASSES},
        "label_rules": {k: int(v) for k, v in rule_counts.items()},
        "label_missing_inputs": {k: len(v) for k, v in missing_pairs.items()},
        "weak_label_counts": {c: int(weak_counts.get(c, 0))
                              for c in weak.WEAK_CLASSES},
        "weak_label_inputs": list(weak.LABEL_INPUT_FEATURES),
        "weak_thresholds": rc.thresholds_to_dict(thresholds),
        "weak_calibration_targets": CALIB_TARGETS,
        "nan_fraction": {c: null_counts[c] / total for c in ALL_FEATURES},
        "review_confidence": db.REVIEW_CONFIDENCE,
        "alert_score_threshold": db.ALERT_SCORE_THRESHOLD,
        "coordinate_features": [],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print()
    print(f"Wrote {OUT_DIR}/X.parquet, y.parquet, meta.parquet, manifest.json")
    print("NEXT: colab_p7_train.py")


if not _STOP:
    run_assembly()