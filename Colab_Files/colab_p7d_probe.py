# ============================================================================
# P7d -- COLAB CELL: is there an independent cause label on Drive?
#
# TWO OPEN QUESTIONS, AND ONE MAY SETTLE THE WHOLE PROJECT
#
# QUESTION 1 -- `external/DL_FIRE_M-C61_814501/` holds 3 CSV files, 89 MB.
#
#   A FIRMS distribution archive in CSV form ships a `type` column:
#       0 = AGRICULTURAL   1 = ARCHIVE (urban)   2 = GRASSLAND
#       3 = SHORELINE     4 = AVIATION
#   That is a CAUSE classification produced from the satellite itself. It has
#   nothing to do with the OSM/GEM facility gazetteer that manufactured the
#   weak label, so it is exactly the independent signal this project lacks.
#
#   `India_Firms.parquet` does NOT have it (22 columns, no `type`). But this CSV
#   is a different download and may. P1b inventoried the folder by size and
#   extension only -- it never opened a CSV header.
#
#   If `type` is present AND the date range covers the catalogue, then the
#   no-download impasse is over and the weak label can be REPLACED rather than
#   merely criticised.
#
# QUESTION 2 -- what are the ACTUAL calibrated thresholds?
#
# P7b measured the rule's conjuncts at hardcoded thresholds, then warned that
# the real ones come from `rule_calibration`. That warning was load-bearing:
#
#     shape: persistence >= 0.0   pos 100.00%   neg 100.00%   separates NOTHING
#     shape: detections >= 2      pos  38.54%   neg  36.07%   separates NOTHING
#
# `shape = persistence >= t OR detections >= t`. If the calibrated `t` is 0.0
# then that branch is vacuously true for every event and the OR never excludes
# anything -- leaving the label as `near_any AND typology`, i.e. two distance
# terms and no fire-behaviour term at all. That would make the persistence
# machinery decorative. This cell reads the real values and recomputes.
#
# NOT THIRD QUESTION -- VIIRS_FOOTPRINTS (0.95 GB). Worth checking its schema,
# but footprints describe WHERE a detection was seen, not WHY. `type` is worth
# far more, so it is probed first.
#
# NOTHING IS WRITTEN. Paste the output back.
# ============================================================================

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import colab_utils as cu
import weak_labels as weak

#: See colab_bootstrap.CELL_VERSION_FLOORS. Bump on every change so the
#: bootstrap can tell a current cell from an old one -- an unversioned
#: cell cannot be checked, and an uncheckable cell is what cost five runs.
P7D_CELL_VERSION = "2026-10-04.1"

print("=" * 74)
print(f"P7d -- independent cause label, and the real calibrated thresholds [v{P7D_CELL_VERSION}]")
print("=" * 74)
print(cu.drive_report())

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError(f"no project root under {cu.MYDRIVE}")
DATASET_DIR = PROJECT_ROOT / "full_2012_2026" / "features" / "dataset"
QC_DIR = PROJECT_ROOT / "qc"
EXTERNAL_ROOT = next(
    (p for p in (PROJECT_ROOT / "full_2012_2026" / "external",
                 PROJECT_ROOT / "external")
     if cu.path_ok(p)),
    PROJECT_ROOT / "full_2012_2026" / "external",
)
print(f"dataset  : {DATASET_DIR}")
print(f"external : {EXTERNAL_ROOT}")

# ---------------------------------------------------------------------------
# 1. THE CALIBRATED THRESHOLDS
#
# Read before probing the CSV, because this is fast and it decides how much
# the P7b conjunct table can be trusted.
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("1. CALIBRATED THRESHOLDS -- what the rule ACTUALLY uses")
print("=" * 74)

_cal_path = QC_DIR / "p6_rule_calibration.json"
if not cu.path_ok(_cal_path):
    print(f"  {_cal_path} is absent.")
    print("  P6 phase 3 writes it. If it does not exist, P6 was run before the")
    print("  calibration phase existed -- re-run P6. Every threshold below is")
    print("  then whatever weak_labels defaults to, which is a guess.")
else:
    _cal = json.loads(_cal_path.read_text())
    print(json.dumps(_cal, indent=2, default=str))

# The dataclass the rule actually reads at runtime. Printing it settles
# whether the calibrated values above were ever plumbed through.
print()
print("weak_labels.WeakThresholds in this session (defaults unless P6 overrode):")
print(weak.WeakThresholds())

# --- is the persistence branch vacuous at the REAL threshold? --------------
print()
print("=" * 74)
print("2. IS THE `shape` CONJUNCT VACUOUS?")
print("=" * 74)
print("The rule is `shape = persistence >= t  OR  detections >= t`.")
print("A branch is vacuous if EVERY row passes it. Then the OR excludes")
print("nothing and the label is `near_any AND typology` -- two distance terms")
print("and no fire-behaviour term at all.")
print()

_T = weak.WeakThresholds()
# The label vector, needed by section 2b. Read before the streaming so a
# missing file fails immediately rather than after a 7.27M-row pass.
_y_cols = set(pq.ParquetFile(DATASET_DIR / "y.parquet").schema_arrow.names)
_labels_full = pd.read_parquet(
    DATASET_DIR / "y.parquet",
    columns=["label_weak"] if "label_weak" in _y_cols else ["label"],
)["label_weak" if "label_weak" in _y_cols else "label"].to_numpy()
print()
print(f"catalogue rows: {len(_labels_full):,}")
# Count thresholds in weak_labels are log1p(detections); distance ones are
# log10(1 + m). Convert for display so the numbers are checkable by eye.
_p_t = float(getattr(_T, "industrial_persistence", np.nan))
_p_d_log = float(getattr(_T, "industrial_min_detections", np.nan))
_near_log = float(getattr(_T, "industrial_near_m", np.nan))
_typ_log = float(getattr(_T, "industrial_typology_m", np.nan))
print(f"  industrial_persistence      t = {_p_t:.6g}")
print(f"  industrial_min_detections  t = {_p_d_log:.6g} "
      f"(= log1p({np.expm1(_p_d_log):.0f}) detections)")
print(f"  industrial_near_m           = {np.power(10.0, _near_log) - 1.0:,.0f} m")
print(f"  industrial_typology_m       = {np.power(10.0, _typ_log) - 1.0:,.0f} m")
print()
if _p_t <= 0.0:
    print("  => industrial_persistence <= 0, and FRP_PERSISTENCE is")
    print("     observed_days/30 which is >= 0 by construction. So that branch")
    print("     is TRUE FOR EVERY EVENT and the OR is vacuous.")
    print("     The label is `near_any AND typology` -- pure proximity.")
else:
    print(f"  industrial_persistence = {_p_t:.6g} > 0, so the branch is not")
    print("  vacuous BY CONSTRUCTION. Whether it is vacuous IN PRACTICE -- i.e.")
    print("  whether nearly every event clears it -- is measured below.")

# ---------------------------------------------------------------------------
# 2b. THE FULL CONJUNCT DECOMPOSITION, AT THE REAL THRESHOLDS
#
# This is the measurement that characterises the label exactly. Stream six
# columns over the catalogue and evaluate each conjunct, each pair, and the
# full rule, plus the ablation: what each conjunct contributes to the
# prevalence and to the distance-only ranking.
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("2b. FULL DECOMPOSITION AT THE CALIBRATED THRESHOLDS")
print("=" * 74)

_NEEDED = [
    "DIST_NEAREST_FACILITY_LOG", "FRP_PERSISTENCE", "DETECTION_COUNT_LOG",
    "DIST_POWERPLANT_LOG", "DIST_PETROCHEMICAL_LOG",
    "DIST_STEEL_LOG", "DIST_CEMENT_LOG",
]
_XC = list(pq.ParquetFile(DATASET_DIR / "X.parquet").schema_arrow.names)
_HAVE = [c for c in _NEEDED if c in _XC]
print(f"  streaming {len(_HAVE)}/{len(_NEEDED)} columns: {_HAVE}")
if len(_HAVE) < len(_NEEDED):
    print("  missing:", [c for c in _NEEDED if c not in _XC])
else:
    import time as _time
    _t0 = _time.time()
    _cols = {c: np.empty(len(_labels_full), dtype="float64") for c in _HAVE}
    _off = 0
    for _b in pq.ParquetFile(DATASET_DIR / "X.parquet").iter_batches(
        batch_size=500_000, columns=_HAVE
    ):
        _n = _b.num_rows
        _blk = _b.to_pandas()
        for _c in _HAVE:
            _cols[_c][_off:_off + _n] = _blk[_c].to_numpy(
                dtype="float64", na_value=np.nan)
        _off += _n
    print(f"  done, {_off:,} rows in {_time.time() - _t0:.0f}s")

    def _dl(m):
        return np.log10(1.0 + m)

    _t_pers = float(getattr(_T, "industrial_persistence", 0.35))
    _t_det = float(getattr(_T, "industrial_min_detections", 2))
    _t_near_log = float(getattr(_T, "industrial_near_m", 1000.0))
    _t_typ_log = float(getattr(_T, "industrial_typology_m", 10000.0))

    _pers = _cols["FRP_PERSISTENCE"]
    _det = _cols["DETECTION_COUNT_LOG"]
    _near = _cols["DIST_NEAREST_FACILITY_LOG"]
    _typ = np.min(
        np.column_stack([_cols[c] for c in
                         ("DIST_POWERPLANT_LOG", "DIST_PETROCHEMICAL_LOG",
                          "DIST_STEEL_LOG", "DIST_CEMENT_LOG")]),
        axis=1,
    )

    _C = {
        "near_any":   _near <= _t_near_log,
        "shape":      (_pers >= _t_pers) | (_det >= _t_det),
        "typology":   _typ <= _t_typ_log,
    }
    # NaN must NOT satisfy a threshold. NaN comparisons are False, which is
    # correct here: "could not measure" is not evidence of persistence.
    _shape_branch_pers = _pers >= _t_pers
    _shape_branch_det = _det >= _t_det

    print()
    print(f"  thresholds: persistence >= {_t_pers:g}   "
          f"detections >= {_t_det:g}   "
          f"near <= {10 ** _t_near_log - 1:,.0f} m   "
          f"typology <= {10 ** _t_typ_log - 1:,.0f} m")
    print()
    print(f"  {'conjunct':<26} {'share of catalogue':>20}")
    for _k, _v in _C.items():
        print(f"  {_k:<26} {100.0 * float(np.mean(_v)):>19.2f}%")
    print(f"  {'  shape/ persistence':<26} {100.0 * float(np.mean(_shape_branch_pers)):>19.2f}%"
          f"   (branch alone)")
    print(f"  {'  shape/ detections':<26} {100.0 * float(np.mean(_shape_branch_det)):>19.2f}%"
          f"   (branch alone)")
    print(f"  {'  shape/ EITHER':<26} {100.0 * float(np.mean(_C['shape'])):>19.2f}%")

    if float(np.mean(_shape_branch_pers)) > 0.98:
        print()
        print("  => the persistence branch is VACUOUS IN PRACTICE: it passes")
        print("     almost every event, so the OR excludes nothing and `shape`")
        print("     is doing no work at this threshold.")
    elif float(np.mean(_C["shape"])) > 0.95:
        print()
        print("  => `shape` is nearly vacuous at this threshold.")

    _full = _C["near_any"] & _C["shape"] & _C["typology"]
    print()
    print(f"  ALL THREE conjuncts        {100.0 * float(np.mean(_full)):>19.2f}%"
          f"   <- observed prevalence should match")
    print()
    print("  ABLATION -- prevalence with each conjunct REMOVED:")
    print(f"    near_any only            {100.0 * float(np.mean(_C['near_any'])):>19.2f}%")
    print(f"    shape only               {100.0 * float(np.mean(_C['shape'])):>19.2f}%")
    print(f"    typology only            {100.0 * float(np.mean(_C['typology'])):>19.2f}%")
    print(f"    near AND typology        "
          f"{100.0 * float(np.mean(_C['near_any'] & _C['typology'])):>19.2f}%"
          f"   <- no fire-behaviour term at all")
    print(f"    near AND shape           "
          f"{100.0 * float(np.mean(_C['near_any'] & _C['shape'])):>19.2f}%")
    print(f"    shape AND typology       "
          f"{100.0 * float(np.mean(_C['shape'] & _C['typology'])):>19.2f}%")
    print()
    print("  If `near AND typology` reproduces the prevalence closely, then")
    print("  dropping `shape` costs almost nothing and the label carries NO")
    print("  fire-behaviour information -- it is two distance terms.")

    # Distance-only ranking, and what the shape term adds on top of it.
    try:
        from sklearn.metrics import average_precision_score
        _fin = np.isfinite(_near)
        _y = _full[_fin].astype(int)
        _d = _near[_fin]
        print()
        print(f"  PR-AUC of distance ALONE against the full rule: "
              f"{average_precision_score(_y, -_d):.4f}")
        print("  That is the bar an honest model must clear.")
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# 3. THE FIRMS `type` COLUMN
#
# The single highest-value probe in this project. Open headers only -- these
# files may be tens of millions of rows and reading one is enough.
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("3. FIRMS `type` -- AN INDEPENDENT CAUSE LABEL?")
print("=" * 74)

_looking_for = ("DL_FIRE", "firm", "csv")
_candidates = [
    p for p in EXTERNAL_ROOT.rglob("*")
    if p.is_file() and p.suffix.lower() in (".csv", ".txt", ".gz", ".zip")
]
print(f"  CSV/TXT/ZIP files under external/: {len(_candidates)}")
for p in sorted(_candidates)[:40]:
    try:
        size = p.stat().st_size
    except OSError:
        continue
    print(f"    {size / 1024 ** 2:>9.1f} MB  {p.relative_to(EXTERNAL_ROOT)}")

_TYPE_MAP = {
    "0": "AGRICULTURAL", "1": "ARCHIVE_URBAN", "2": "GRASSLAND",
    "3": "SHORELINE", "4": "AVIATION",
}

print()
print("-" * 74)
print("Reading the first line of each candidate as a header ...")
print("-" * 74)
_found_type_file = None
for p in sorted(_candidates):
    try:
        with p.open("r", errors="replace") as fh:
            head = fh.readline()
    except OSError as exc:
        print(f"  [skip] {p.name}: {exc}")
        continue
    cols = [c.strip() for c in re.split(r"[,\t;]", head)]
    has_type = any(c.lower() == "type" for c in cols)
    print()
    print(f"  {p.relative_to(EXTERNAL_ROOT)}")
    print(f"    columns ({len(cols)}): {cols}")
    print(f"    has FIRMS `type`: {has_type}")
    if has_type and _found_type_file is None:
        _found_type_file = p
        print("    *** CANDIDATE INDEPENDENT LABEL ***")

if _found_type_file is None:
    print()
    print("  => NO FIRMS `type` column in any CSV under external/.")
    print("     Combined with India_Firms having no `type`, there is NO")
    print("     independent cause label on Drive. The gold set is the only")
    print("     route that needs no download.")
else:
    p = _found_type_file
    print()
    print("=" * 74)
    print(f"MEASURING `{p.name}` -- is it usable as a label?")
    print("=" * 74)
    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception as exc:                       # noqa: BLE001
        print(f"  full read failed ({type(exc).__name__}); trying header only")
        df = pd.read_csv(p, nrows=0)
        raise SystemExit("read failed -- report this, do not guess")

    print(f"  rows: {len(df):,}   columns: {len(df.columns)}")
    low = {str(c).lower(): c for c in df.columns}
    _tc = low.get("type")
    if _tc is None:
        raise SystemExit("type column vanished between header and full read")
    _ac = low.get("acq_date") or low.get("acquisition_date")
    _dc = low.get("latitude") or low.get("lat")
    _lc = low.get("longitude") or low.get("lon")

    print()
    print("  type distribution:")
    _vc = df[_tc].value_counts(dropna=False).sort_index()
    for k, v in _vc.items():
        _name = _TYPE_MAP.get(str(k), "")
        print(f"    {k}  {_name:<14} {int(v):>12,}  {100.0 * v / len(df):>6.2f}%")

    if _ac is not None:
        print()
        print(f"  acq_date range: {df[_ac].min()} -> {df[_ac].max()}")
        print("  catalogue range: 2012-01-20 -> 2026-08-31")
        print("  -> coverage must OVERLAP the catalogue or the label cannot be")
        print("     joined. Report any mismatch rather than assuming.")

    if _dc is not None and _lc is not None:
        _in_india = int((
            (df[_dc] >= 6) & (df[_dc] <= 36)
            & (df[_lc] >= 68) & (df[_lc] <= 98)
        ).sum())
        print()
        print(f"  lat range: {df[_dc].min():.3f} .. {df[_dc].max():.3f}")
        print(f"  lon range: {df[_lc].min():.3f} .. {df[_lc].max():.3f}")
        print(f"  rows inside India (lat 6-36, lon 68-98): {_in_india:,}")

    print()
    print("=" * 74)
    print("DECISION")
    print("=" * 74)
    print("`type` is present. Before anything is built on it, answer these:")
    print("  1. Does its date range cover 2012-2026?")
    print("  2. Is it DETECTION-level (one row per VIIRS hit) or EVENT-level?")
    print("     Detection-level means an aggregation is needed, and the")
    print("     majority type over an event is the natural label.")
    print("  3. AGRICULTURAL vs (everything else) is the contrast that is")
    print("     defensible WITHOUT a facility gazetteer. ARCHIVE/GRASSLAND/")
    print("     SHORELINE/AVIATION are not 'industrial' either -- do NOT")
    print("     relabel them as such. The honest framing is:")
    print("        'separation of VEGETATION fires from non-vegetation ones'")
    print("     which is a real, publishable, independent question -- and one")
    print("     the industrial framing can then be layered on top of.")
    print()
    print("Do NOT write anything yet. Report the three answers above and the")
    print("type distribution, then the replacement label gets designed.")

print()
print("=" * 74)
print("END -- paste the whole output above.")
print("=" * 74)
print("No files were written.")