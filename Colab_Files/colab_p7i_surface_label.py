# ============================================================================
# P7i -- COLAB CELL: build the SUBSTRATE label, replacing the broken one
#
# WHAT THIS REPLACES, AND WHY
#
# Measured on the full catalogue (P7h, 2026-10-05):
#
#     substrate  cropland 48.35%   tree 32.69%   grassland 8.89%
#                shrubland 2.79%  built-up 3.77%  bare 3.01%  water 0.47%
#
#     70.1% of weak INDUSTRIAL_FIRE sits on VEGETATED substrate
#     vs a substrate proxy:  precision 0.299  recall 0.636  F1 0.407
#     P(weak INDUSTRIAL | non-vegetated) = 0.636
#     P(weak INDUSTRIAL | vegetated)     = 0.116     -> 5.5x
#
# Precision is the damning number. And `qc/p6_rule_calibration.json` shows why
# it could not have been otherwise: `industrial_persistence = 1/30`, the FLOOR
# of `observed_days / 30`, so the rule's `shape` conjunct excluded nothing and
# the label reduced to two distance terms.
#
# WHY CIRCULARITY MATTERS AS MUCH AS ACCURACY
#
# `weak_labels.LABEL_INPUT_FEATURES` publishes the 16 columns its rules read,
# and `training_design` excludes them from model B. So model B was blind to the
# evidence behind its own target. Substrate changes that: it comes from a raster
# absent from X, so after `label_input_features(include_surface=True)` the
# label and the model share NO input and model B's score means something.
#
# THE NAME IS DELIBERATE. NOT `INDUSTRIAL_FIRE`.
#
# What the evidence supports is "a fire on non-vegetated substrate". "Industrial"
# is a claim about CAUSE, which needs site corroboration or NBR burn-scar
# evidence. Calling this INDUSTRIAL_FIRE would repeat the original overclaim one
# level down. The gazetteer becomes CORROBORATION reported alongside the class,
# not the definition of it.
#
# NBR IS DELIBERATELY ABSENT FROM THE BASE RULE
#
# The obvious addition is an NBR drop test. It is left out on purpose: built-up
# substrate already has low NBR, so a drop is weak and noisy exactly on the
# substrate this class is defined over -- while being excellent on vegetation,
# which is the negative class. It would add ~6 hours of extraction and little
# discrimination. Section 5 reports what an NBR term could add, so the decision
# is made with the trade visible rather than by omission.
#
# THE THRESHOLD IS CHOSEN FROM A GRID, NOT ASSUMED
#
# `min_persistence` is the number that decides class size, and the weak label's
# failure was a threshold nobody chose. Section 4 prints the pool size across the
# whole grid, the cell records which setting was used, and that record is written
# beside the label.
# ============================================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import colab_utils as cu
import surface_labels as sl
import training_design as td

P7I_CELL_VERSION = "2026-10-05.1"

#: The one number that decides class size. `observed_days = value * 30`.
#: The tier only. There is no persistence FILTER on the label: repeat
#: observation is reported as a confidence tier, because
#: `FRP_PERSISTENCE = observed_days/30` and `DETECTION_COUNT_LOG` are the same
#: evidence, and filtering on it discarded 97% of the catalogue for nothing.
TIER_MIN_DAYS = int(globals().get("SURFACE_TIER_MIN_DAYS", 3))

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError(f"no project root under {cu.MYDRIVE}")
DATASET_DIR = PROJECT_ROOT / "full_2012_2026" / "features" / "dataset"
QC_DIR = PROJECT_ROOT / "qc"

print("=" * 74)
print(f"P7i -- SUBSTRATE LABEL  [v{P7I_CELL_VERSION}]")
print("=" * 74)
print(cu.drive_report())
print(f"  label   : substrate only, no persistence filter")
print(f"  tier    : >={TIER_MIN_DAYS} observed days, reported not filtered")
print("  NOTE: NOT called INDUSTRIAL_FIRE. See the header for why.")

# ---------------------------------------------------------------------------
# 1. JOIN
#
# `landcover_events.parquet` was written by P7h with an explicit `row` column
# equal to the position in meta/X/y, so the join is positional and exact. No
# coordinate matching, no risk of pairing an event with its neighbour.
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("1. JOIN")
print("=" * 74)

_lc_path = QC_DIR / "landcover_events.parquet"
if not cu.path_ok(_lc_path):
    _alt = QC_DIR / "landcover_events.csv"
    if cu.path_ok(_alt):
        _lc_path = _alt
    else:
        raise SystemExit(
            "qc/landcover_events.parquet not found. Run "
            "colab_p7h_landcover_all.py to completion first -- section 4 must "
            "have printed 'complete: 7,265,849 of 7,265,849 rows'."
        )
print(f"  land cover : {_lc_path}")

lc = pd.read_parquet(_lc_path) if _lc_path.suffix == ".parquet" \
    else pd.read_csv(_lc_path)
print(f"    rows {len(lc):,}   columns {list(lc.columns)}")

_XC = set(pq.ParquetFile(DATASET_DIR / "X.parquet").schema_arrow.names)
_thermal = [c for c in ("FRP_PERSISTENCE", "DETECTION_COUNT_LOG")
            if c in _XC]
if len(_thermal) < 2:
    raise SystemExit(
        f"X.parquet is missing thermal columns; found {_thermal}. The label "
        f"cannot be built without them."
    )

N = len(lc)
print(f"  streaming {len(_thermal)} thermal columns for {N:,} rows "
      f"(ONE pass) ...")
_therm = pd.DataFrame({"row": np.arange(N, dtype=np.int64)})
for _c in _thermal:
    _therm[_c] = np.full(N, np.nan)

# ONE pass over X.parquet, filling every requested column. An earlier draft
# re-read the whole 7.27M-row file once per column, with a stray `break` in the
# middle -- four full passes where one suffices, on a table that takes minutes
# to read.
_off = 0
for _b in pq.ParquetFile(DATASET_DIR / "X.parquet").iter_batches(
    batch_size=500_000, columns=list(_thermal)
):
    _blk = _b.to_pandas()
    _n = _blk.shape[0]
    for _c in _thermal:
        _therm[_c].to_numpy()[_off:_off + _n] = _blk[_c].to_numpy(
            dtype="float64", na_value=np.nan)
    _off += _n
print(f"  thermal rows {len(_therm):,}  read {_off:,} from X")

if _off != N:
    raise SystemExit(
        f"X.parquet yielded {_off:,} rows but the land cover table has {N:,}. "
        f"The positional join would be WRONG -- stop and check which P6/P7h "
        f"wrote these."
    )

frame = lc.merge(_therm[["row"] + _thermal], on="row", how="left")
print(f"  merged     {len(frame):,}")
_miss = int(frame[_thermal].isna().all(axis=1).sum())
if _miss:
    print(f"  WARNING: {_miss:,} rows have no thermal data and will be UNKNOWN")

# ---------------------------------------------------------------------------
# 2. SUBSTRATE CROSS-CHECK
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("2. SUBSTRATE INTEGRITY")
print("=" * 74)
_map = int((frame["WORLDCOVER_CLASS"] >= 0).sum())
print(f"  mapped      : {_map:,} ({100.0 * _map / len(frame):.2f}%)")
print(f"  unmapped    : {len(frame) - _map:,}  -> these become UNKNOWN, "
      "never a positive")
for _c, _nm in ((50, "built-up"), (60, "bare / sparse"), (70, "snow and ice"),
                (80, "permanent water")):
    _k = int((frame["WORLDCOVER_CLASS"] == _c).sum())
    print(f"  {_nm:<20} {_k:>10,}")
_nv = int((frame["LANDCOVER_VEGETATED"] == 0).sum())
print(f"  NON-vegetated total: {_nv:,} ({100.0 * _nv / len(frame):.2f}%)")

# ---------------------------------------------------------------------------
# 3. THRESHOLD GRID
#
# Printed BEFORE the choice, so the class size is not a surprise and the
# setting is made with the trade-off visible.
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("3. POOL SIZE ACROSS THE THRESHOLD GRID")
print("=" * 74)
_grid = sl.pool_grid(frame)
print(_grid.to_string(index=False))
print()
print("  The weak label's `industrial_persistence` was 1/30 -- the floor, so")
print("  the persistence branch passed everything. The first row here is that")
print("  same setting, and it is included so the comparison is explicit.")

# ---------------------------------------------------------------------------
# 4. BUILD
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("4. BUILD THE LABEL")
print("=" * 74)
th = sl.SurfaceThresholds(tier_min_days=TIER_MIN_DAYS)
print(f"  using: {th.describe()}")
labelled = sl.assign_surface_labels(frame, th)
_mismatch = labelled.attrs.get("n_substrate_mismatch", 0)
if _mismatch:
    print(f"  WARNING: {_mismatch:,} rows where LANDCOVER_VEGETATED and")
    print("  WORLDCOVER_CLASS disagree. Reported, not silently resolved.")
_rep = sl.surface_label_report(frame, th)
print()
print(_rep.to_string(index=False))

_sf = int((labelled["label_surf"] == sl.SURFACE_FIRE).sum())
_vf = int((labelled["label_surf"] == sl.VEGETATION_FIRE).sum())
print()
print(f"  SURFACE_FIRE     : {_sf:,} ({100.0 * _sf / len(labelled):.2f}%)")
print(f"  VEGETATION_FIRE  : {_vf:,} ({100.0 * _vf / len(labelled):.2f}%)")
print(f"  UNKNOWN          : {len(labelled) - _sf - _vf:,}")
print()
print("  If SURFACE_FIRE is very small, the persistence floor is too high for")
print("  training. Say so in the writeup rather than lowering it until the")
print("  number looks better.")

# ---------------------------------------------------------------------------
# 5. CORROBORATION, and what NBR could add
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("5. CORROBORATION -- the gazetteer as evidence, not as definition")
print("=" * 74)
_dcol = "DIST_NEAREST_FACILITY_LOG"
if _dcol in _XC:
    _col = np.full(len(frame), np.nan)
    _o = 0
    for _bb in pq.ParquetFile(DATASET_DIR / "X.parquet").iter_batches(
        batch_size=500_000, columns=[_dcol]
    ):
        _col[_o:_o + _bb.num_rows] = (_bb.column(0).to_pandas()
                                      .to_numpy(dtype="float64",
                                                na_value=np.nan))
        _o += _bb.num_rows
    frame[_dcol] = _col
    _m = np.power(10.0, frame[_dcol].to_numpy()) - 1.0
    _is_sf = (labelled["label_surf"] == sl.SURFACE_FIRE).to_numpy()
    _fin = np.isfinite(_m)
    print("  median distance to nearest facility (m):")
    for _name, _sel in (("SURFACE_FIRE", _is_sf),
                        ("VEGETATION_FIRE",
                         (labelled["label_surf"] == sl.VEGETATION_FIRE).to_numpy())):
        _s = _sel & _fin
        if _s.sum():
            print(f"    {_name:<18} n={int(_s.sum()):>8,}  "
                  f"median {np.median(_m[_s]):>10,.0f} m   "
                  f"within 5 km {100.0 * np.mean(_m[_s] <= 5000):>5.1f}%")
    print()
    print("  Read this as CORROBORATION, not as the class definition:")
    print("  - SURFACE_FIRE nearer facilities means substrate and gazetteer")
    print("    agree, which supports the 'industrial' reading.")
    print("  - VEGETATION_FIRE ALSO near facilities is the mislabelled")
    print("    farmland P7g measured, now visible as a separate class rather")
    print("    than hidden inside INDUSTRIAL_FIRE.")
else:
    print(f"  {_dcol} absent from X; skipping corroboration")

print()
print("  WHAT AN NBR TERM COULD ADD, AND WHY IT IS NOT IN THE BASE RULE")
print("  - Built-up substrate already has low NBR, so an NBR *drop* is weak")
print("    and noisy exactly on the substrate SURFACE_FIRE is defined over.")
print("  - The same test is strong on vegetation -- which is the NEGATIVE")
print("    class, so it would sharpen a distinction the substrate term")
print("    already makes.")
print("  - Cost is ~6 hours of Earth Engine extraction for a second stage.")
print("  - It also measures DAMAGE, which 'non-vegetated substrate' does not,")
print("    and damage is the stronger claim. Worth doing if time allows,")
print("    as SURFACE_FIRE_BURNED = SURFACE_FIRE AND dNBR < threshold.")

# ---------------------------------------------------------------------------
# 6. THE LEAKAGE PARTITION, RECOMPUTED
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("6. LEAKAGE PARTITION")
print("=" * 74)
import feature_ledger as fl                                  # noqa: E402

_old = td.label_input_features(include_surface=False)
_new = td.label_input_features(include_surface=True)
print(f"  forbidden set: {len(_old)} -> {len(_new)}")
print(f"  added: {sorted(_new - _old)}")
_a, _b, _rep2 = td.split_feature_sets(list(fl.FEATURES_141),
                                      include_surface=True)
print(f"  model A features: {len(_a)}")
print(f"  model B features: {len(_b)}")
for _f in ("LANDCOVER_VEGETATED", "WORLDCOVER_CLASS"):
    print(f"    {_f} in model B: {_f in _b}  <- MUST be False")
print()
print("  Substrate is LABEL evidence and is deliberately NOT a predictor.")
print("  Adding it to X would invite exactly the mistake already made with")
print("  DIST_*.")

# ---------------------------------------------------------------------------
# 7. WRITE
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("7. WRITE")
print("=" * 74)
_out = labelled[["row"] + [c for c in ("event_id", "WORLDCOVER_CLASS",
                                      "LANDCOVER_VEGETATED",
                                      "FRP_PERSISTENCE",
                                      "DETECTION_COUNT_LOG",
                                      "sustained", "substrate_known",
                                      "label_surf", "is_surface_fire",
                                      "is_surface_fire_tiered",
                                      "surface_label_version")
                       if c in labelled.columns]].copy()
if "start_year" in labelled.columns:
    _out["start_year"] = labelled["start_year"]

try:
    _p = QC_DIR / "surface_labels.parquet"
    _out.to_parquet(_p, index=False)
    print(f"  Wrote {_p}  ({len(_out):,} rows)")
except Exception as exc:                                    # noqa: BLE001
    print(f"  parquet unavailable ({type(exc).__name__}); writing CSV")
    _p = QC_DIR / "surface_labels.csv"
    _out.to_csv(_p, index=False)
    print(f"  Wrote {_p}  ({len(_out):,} rows)")

_meta = {
    "cell_version": P7I_CELL_VERSION,
    "label_name": sl.SURFACE_FIRE,
    "why_not_industrial_fire": (
        "'Industrial' is a claim about cause. The evidence here supports "
        "'a fire on non-vegetated substrate'. Naming the class for what we "
        "hope it means is the mistake that produced the old label."
    ),
    "label_rule": "substrate only; no persistence filter",
    "tier_min_days": TIER_MIN_DAYS,
    "why_no_persistence_filter": (
        "FRP_PERSISTENCE and DETECTION_COUNT_LOG are the same evidence "
        "(repeat observation). Filtering on both left one near-inert and "
        "discarded 97% of the catalogue, since 96.7% of events are "
        "single-cluster. Reported as a tier instead."
    ),
    "n_surface_fire_tiered": int(
        (labelled["is_surface_fire_tiered"] == 1).sum()),
    "label_inputs": sl.surface_label_inputs(),
    "n_surface_fire": _sf,
    "n_vegetation_fire": _vf,
    "n_unknown": int(len(labelled) - _sf - _vf),
    "n_substrate_mismatch": _mismatch,
    "weak_label_proxy_scores": {
        "note": ("P7g, against non-vegetated substrate as a proxy. NOT "
                 "accuracy -- substrate is not ground truth."),
        "precision": 0.299, "recall": 0.636, "f1": 0.407,
    },
    "nbr_in_base_rule": False,
    "nbr_rationale": (
        "Built-up substrate already has low NBR, so a drop is weak on exactly "
        "the substrate SURFACE_FIRE is defined over, while being strong on "
        "vegetation which is the negative class. Deferred to a second stage."
    ),
    "leakage_partition": {
        "forbidden_before": len(_old),
        "forbidden_now": len(_new),
        "model_b_features": len(_b),
    },
}
(QC_DIR / "surface_label_meta.json").write_text(json.dumps(_meta, indent=2))
print(f"  Wrote {QC_DIR / 'surface_label_meta.json'}")
print()
print("  The thresholds used are recorded in that JSON, beside the label.")
print("  A label whose thresholds live only in a shell history is a label")
print("  nobody can reproduce.")

print()
print("=" * 74)
print("END -- paste sections 3, 4 and 5.")
print("=" * 74)
print()
print("NEXT: re-run P7 against this label. Model B will then be measured")
print("against a target it cannot reproduce, which is the first time any")
print("number in this project will mean what it says.")