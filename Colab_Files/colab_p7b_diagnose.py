# ============================================================================
# P7b -- COLAB CELL: diagnose what the target actually IS, before training
#
# WHY THIS CELL EXISTS
#
# Measured on the FULL catalogue (run 2026-10-04), over 1,119,254 positives:
#
#     max distance-to-nearest-facility among ALL positives = 5,000 m
#     positives beyond 5,000 m                          = 2  (0.0002%)
#     PR-AUC of DISTANCE ALONE                          = 0.9811
#     PR-AUC of the honest 29-feature model B           = 0.3353
#
# `weak_labels.assign_weak_labels` defines INDUSTRIAL_FIRE as
#
#     (DIST_NEAREST_FACILITY <= 5000 m  or  FACILITY_COUNT_5KM >= 1)
#     and (FRP_PERSISTENCE >= t  or  DETECTION_COUNT_LOG >= t)
#     and min(DIST_POWERPLANT, DIST_PETROCHEMICAL, DIST_STEEL, DIST_CEMENT) <= t
#
# `FACILITY_COUNT_5KM >= 1` and `DIST_NEAREST_FACILITY <= 5000 m` are the SAME
# statement, so near_any's two branches are redundant. The entire spatial
# content of the target is one 5 km cut, which is also why distance alone
# reaches PR-AUC 0.9811.
#
# CONSEQUENCE: no hyperparameter, no feature count, no extra raster fixes this.
# LANDCOVER_FUEL (16) and WEATHER (18) would help a target that consulted land
# cover and weather. This one does not.
#
# WHAT THIS CELL ESTABLISHES
#
#   1. The binding constraint, on the full catalogue rather than one fold.
#   2. Which conjunct actually DISCRIMINATES -- measured on negatives too,
#      because a conjunct satisfied by 100% of positives may still be
#      satisfied by 100% of negatives, in which case it does no work.
#   3. Whether India_Firms carries an INDEPENDENT per-detection cause label.
#   4. What external/ actually contains -- at BOTH candidate paths, because
#      the duplicate-folder hazard has already bitten this project twice.
#   5. Whether Earth Engine can be authenticated.
#
# NOTHING IS WRITTEN. Read the output and paste it back.
# ============================================================================

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import colab_utils as cu
import dataset_builder as db
import weak_labels as weak

#: See colab_bootstrap.CELL_VERSION_FLOORS. Bump on every change so the
#: bootstrap can tell a current cell from an old one -- an unversioned
#: cell cannot be checked, and an uncheckable cell is what cost five runs.
P7B_CELL_VERSION = "2026-10-04.1"

print("=" * 74)
print(f"P7b DIAGNOSTIC -- what is the target, really? [v{P7B_CELL_VERSION}]")
print("=" * 74)
print(cu.drive_report())

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError(f"no project root under {cu.MYDRIVE}")
DATASET_DIR = PROJECT_ROOT / "full_2012_2026" / "features" / "dataset"
QC_DIR = PROJECT_ROOT / "qc"

# external/ exists at TWO candidate paths and guessing is not safe. P1b
# reported 911 files / 1.50 GB under `full_2012_2026/external`, while a first
# run of this cell probed `<root>/external`, found nothing, and would have
# reported "no rasters anywhere" -- a false conclusion from a wrong path.
# That duplicate-folder hazard has already produced one wrong "the Drive is
# missing" diagnosis in this project. Probe both and report which is real.
EXTERNAL_CANDIDATES = [
    PROJECT_ROOT / "external",
    PROJECT_ROOT / "full_2012_2026" / "external",
]
EXTERNAL_ROOT = next(
    (p for p in EXTERNAL_CANDIDATES if cu.path_ok(p)), EXTERNAL_CANDIDATES[0]
)

print(f"dataset  : {DATASET_DIR}")
print(f"external : probing all candidates")
for _c in EXTERNAL_CANDIDATES:
    _state = "PRESENT" if cu.path_ok(_c) else "absent"
    _n = sum(1 for _ in _c.glob("**/*") if _.is_file()) if _state == "PRESENT" else 0
    print(f"            [{_state:>7}] {_c}  ({_n:,} files)")
print(f"  using   : {EXTERNAL_ROOT}")


# ---------------------------------------------------------------------------
# 1. THE BINDING CONSTRAINT, ON THE FULL CATALOGUE
#
# Read ONLY the columns the rule reads. Twelve columns over 7.27M rows, read
# in row groups, is a few seconds -- and it is the difference between asserting
# the 5 km cap on one fold and measuring it on everything.
# ---------------------------------------------------------------------------

RULE_COLS = [
    "DIST_NEAREST_FACILITY_LOG", "FACILITY_COUNT_5KM",
    "DIST_POWERPLANT_LOG", "DIST_PETROCHEMICAL_LOG",
    "DIST_STEEL_LOG", "DIST_CEMENT_LOG",
    "FRP_PERSISTENCE", "DETECTION_COUNT_LOG",
]
_X_COLS = list(pq.ParquetFile(DATASET_DIR / "X.parquet").schema_arrow.names)
_have = [c for c in RULE_COLS if c in _X_COLS]
_missing = [c for c in RULE_COLS if c not in _X_COLS]
print()
print("=" * 74)
print("1. RULE INPUTS -- present in X?")
print("=" * 74)
print(f"  present : {len(_have)}/{len(RULE_COLS)}  {_have}")
print(f"  missing : {_missing or 'none'}")
if _missing:
    print("  A missing rule input means P6 wrote labels from a column that is")
    print("  not in X. Report that -- it is a defect, not a detail.")

# --- labels, small columns only ---------------------------------------------
y = pd.read_parquet(
    DATASET_DIR / "y.parquet",
    columns=[c for c in ("label_weak",) if c in
             pq.ParquetFile(DATASET_DIR / "y.parquet").schema_arrow.names],
)
labels = y["label_weak"].to_numpy()
is_fire = labels == "INDUSTRIAL_FIRE"
print()
print(f"catalogue rows        : {len(labels):,}")
print(f"INDUSTRIAL_FIRE (weak): {int(is_fire.sum()):,} "
      f"({100.0 * is_fire.mean():.2f}%)")

# --- stream the rule inputs, keeping only positives + a matched negative set -
#
# Keeping every negative would be 6.1M rows for no extra information: the
# question is the DISTRIBUTION OF POSITIVES. So positives are kept in full and
# negatives are sampled, which is enough to state the base rate.
t0 = time.time()
pos_rows: list[np.ndarray] = []
neg_rows: list[np.ndarray] = []
NEG_SAMPLE_CAP = 400_000

_pf = pq.ParquetFile(DATASET_DIR / "X.parquet")
_offset = 0
_neg_seen = 0
_rng = np.random.default_rng(0)
for _batch in _pf.iter_batches(batch_size=250_000, columns=_have):
    _blk = _batch.to_pandas()
    _n = _blk.shape[0]
    _lo, _hi = _offset, _offset + _n

    _mask = is_fire[_lo:_hi]
    if _mask.any():
        pos_rows.append(_blk.loc[_mask, _have].to_numpy(dtype="float32"))
    _nm = ~_mask
    if _nm.any() and _neg_seen < NEG_SAMPLE_CAP:
        _take = min(int(_nm.sum()), NEG_SAMPLE_CAP - _neg_seen)
        _idx = np.flatnonzero(_nm)
        if _take < len(_idx):
            _idx = _rng.choice(_idx, _take, replace=False)
        neg_rows.append(
            _blk.iloc[_idx][_have].to_numpy(dtype="float32"))
        _neg_seen += _take
    _offset = _hi

_pos = np.vstack(pos_rows) if pos_rows else np.empty((0, len(_have)), "float32")
_neg = np.vstack(neg_rows) if neg_rows else np.empty((0, len(_have)), "float32")
_col = {c: i for i, c in enumerate(_have)}
print()
print(f"streamed {len(_pos):,} positives and {len(_neg):,} sampled negatives "
      f"in {time.time() - t0:.0f}s")

print()
print("=" * 74)
print("THE MEASUREMENT -- is the label just a distance rule?")
print("=" * 74)
if "DIST_NEAREST_FACILITY_LOG" in _col and len(_pos):
    _i = _col["DIST_NEAREST_FACILITY_LOG"]
    _plog = _pos[:, _i].astype("float64")
    _finite = np.isfinite(_plog)
    _plog_f = _plog[_finite]
    # DIST_*_LOG is log10(1 + d_metres); see README section 5 defect 1.
    _pm = 10.0 ** _plog_f - 1.0
    print(f"positives with a finite distance : {int(_finite.sum()):,} "
          f"of {len(_pos):,}")
    if len(_pm):
        print(f"  min {np.nanmin(_pm):,.0f} m   median {np.nanmedian(_pm):,.0f} m"
              f"   p99 {np.nanpercentile(_pm, 99):,.0f} m")
        print(f"  MAX {np.nanmax(_pm):,.0f} m   <-- the whole story is this number")
        for _r in (500, 1000, 2000, 5000, 10_000, 50_000):
            _beyond = int(np.sum(_pm > _r))
            print(f"  positives beyond {_r:>6,} m : {_beyond:>8,} "
                  f"({100.0 * _beyond / len(_pm):.4f}%)")
        # A verdict on whether the cap is exact. It is NOT zero at 5,000 m:
        # float equality on 10**x - 1 leaves a couple of rows a single ulp
        # over, so the test needs a tolerance. An earlier version used `== 0`
        # and printed "NOT a hard cut" against 2 rows in 1.1M -- a rounding
        # artefact read as evidence.
        _cap = float(np.nanmax(_pm))
        _over = int(np.sum(_pm > 5000.0 * (1 + 1e-9)))
        _share = _over / len(_pm)
        print()
        if _share < 1e-4:
            print("  => CONFIRMED: a hard 5,000 m proximity cut.")
            print(f"     {_over} of {len(_pm):,} positives ({_share:.6%}) exceed")
            print("     it, which is rounding on the log10 conversion, not slack.")
            print("     No model beats that cut without the distance feature,")
            print("     and any model WITH the distance scores its own label.")
        else:
            print(f"  => NOT a clean cut: {_share:.4%} of positives exceed 5 km.")
            print("     Re-examine before concluding anything.")
    # And the cheap baseline it implies.
    try:
        from sklearn.metrics import average_precision_score
        if len(_neg) and "DIST_NEAREST_FACILITY_LOG" in _col:
            _nl = _neg[:, _col["DIST_NEAREST_FACILITY_LOG"]].astype("float64")
            _nl = _nl[np.isfinite(_nl)]
            _all_y = np.r_[np.ones(len(_plog_f)), np.zeros(len(_nl))]
            _all_s = np.r_[_plog_f, _nl]
            print()
            print(f"  PR-AUC of DISTANCE ALONE: "
                  f"{average_precision_score(_all_y, -_all_s):.4f}")
            print("  That is the bar a legitimate model must clear.")
            print("  Model B scores 0.3353 against it.")
    except ImportError:
        pass

# --- which conjunct actually binds? -----------------------------------------
print()
print("WHICH CONJUNCT DOES THE WORK?")
print("Share of POSITIVES and NEGATIVES satisfying each conjunct independently.")
print("A conjunct at ~100%/~100% does NO work -- it separates nothing.")
print("A conjunct at 100%/~0% IS the label.")
print()
print(f"  {'conjunct':<44} {'pos':>9} {'neg':>9} {'separates':>10}")

_checks = []
if "DIST_NEAREST_FACILITY_LOG" in _col:
    _checks.append(("near_any: distance <= 5,000 m",
                    _col, lambda v: v[:, _col["DIST_NEAREST_FACILITY_LOG"]]
                    <= np.log10(1 + 5000)))
if "FACILITY_COUNT_5KM" in _col:
    _checks.append(("near_any: count5 >= 1 (same statement)",
                    _col, lambda v: v[:, _col["FACILITY_COUNT_5KM"]] >= 1))
if "FRP_PERSISTENCE" in _col:
    _checks.append(("shape: persistence >= 0.0 (any)",
                    _col, lambda v: v[:, _col["FRP_PERSISTENCE"]] >= 0.0))
if "DETECTION_COUNT_LOG" in _col:
    _checks.append(("shape: detections >= 2",
                    _col, lambda v: v[:, _col["DETECTION_COUNT_LOG"]] >= np.log1p(2)))
if all(c in _col for c in ("DIST_POWERPLANT_LOG", "DIST_PETROCHEMICAL_LOG",
                           "DIST_STEEL_LOG", "DIST_CEMENT_LOG")):
    _tcol = ("DIST_POWERPLANT_LOG", "DIST_PETROCHEMICAL_LOG",
             "DIST_STEEL_LOG", "DIST_CEMENT_LOG")

    def _typ(v, _r=5000.0):
        _t = np.min(np.column_stack(
            [v[:, _col[c]] for c in _tcol]), axis=1)
        return _t <= np.log10(1 + _r)

    _checks.append(("typology: any of the 4 within 5 km", _col, _typ))

    def _typ2(v):
        _t = np.min(np.column_stack(
            [v[:, _col[c]] for c in _tcol]), axis=1)
        return _t <= np.log10(1 + 10_000.0)

    _checks.append(("typology: any of the 4 within 10 km", _col, _typ2))

for _name, _c, _fn in _checks:
    _pm_ = np.asarray(_fn(_pos), dtype=bool)
    _nm_ = np.asarray(_fn(_neg), dtype=bool) if len(_neg) else np.array([])
    if len(_pm_):
        _pp = 100.0 * _pm_.mean()
        _nn = 100.0 * _nm_.mean() if len(_nm_) else float("nan")
        _sep = "*** ALL" if _pp > 99.9 and _nn < 0.1 else (
            "yes" if abs(_pp - _nn) > 20 else "no")
        print(f"  {_name:<44} {_pp:>8.2f}% {_nn:>8.2f}% {_sep:>10}")

print()
print("Read the '*** ALL' rows. If `distance <= 5,000 m` is one of them, then")
print("the weak INDUSTRIAL_FIRE label IS the 5 km proximity rule, the rule")
print("hierarchy adds nothing, and PR-AUC 0.9811 from distance alone is the")
print("ceiling any honest model is being asked to beat.")
print()
print("NOTE: these are the UNCALIBRATED thresholds written as literals above.")
print("P6 derives the real ones via rule_calibration into")
print("qc/p6_rule_calibration.json. A conjunct that looks non-discriminating")
print("here may still discriminate at its calibrated threshold -- so read")
print("qc/p6_rule_calibration.json before concluding. The MAX-distance result")
print("above is threshold-independent and does stand regardless.")


# ---------------------------------------------------------------------------
# 2. IS THERE AN INDEPENDENT LABEL ON DISK?
#
# India_Firms is the raw FIRMS archive. FIRMS ships a per-detection `type`
# (0=AGRICULTURAL, 1=ARCHIVE, 2=GRASS, 3=SHORELINE, 4=AVIATION) and a letter
# `confidence` (l/h/n). Those are a CAUSE signal derived from the satellite
# itself -- completely independent of a facility gazetteer. If present, a
# non-proximity label is possible TODAY with no download.
# ---------------------------------------------------------------------------

print()
print("=" * 74)
print("2. INDEPENDENT LABEL SOURCES")
print("=" * 74)
_ind_candidates = []
for _pat in ("India_Firms*.parquet", "*/India_Firms*.parquet"):
    _ind_candidates.extend(cu.MYDRIVE.glob(_pat))
_ind = sorted(set(_ind_candidates))
print(f"India_Firms candidates: {len(_ind)}")
for _p in _ind[:5]:
    print(f"  {_p}")

if _ind:
    _schema = pq.ParquetFile(_ind[0]).schema_arrow
    print()
    print(f"  columns ({len(_schema.names)}): {list(_schema.names)}")
    _low = {str(n).lower(): str(n) for n in _schema.names}
    _interesting = [k for k in ("type", "confidence", "frp", "brightness",
                                "daynight", "satellite", "acq_time",
                                "acq_date", "latitude", "longitude")
                    if k in _low]
    print(f"  of interest present: {_interesting}")
    _cause = _low.get("type")
    print()
    if _cause is None:
        print("  NO `type` COLUMN. The independent-cause route via India_Firms")
        print("  is not available, and a non-proximity label needs external")
        print("  data. Report this; do not substitute proximity for cause.")
    else:
        print(f"  FOUND `{_cause}`. This is a per-detection CAUSE code that has")
        print("  nothing to do with a facility gazetteer.")
        print("  -> sample its distribution next:")
        print(f"       df = pd.read_parquet({str(_ind[0])!r}, columns=[{_cause!r}])")
        print(f"       print(df[{_cause!r}].value_counts(dropna=False))")
        print("  NOTE: `type` labels the BURN TYPE, not 'industrial'. It")
        print("  separates agriculture from everything else, which is real")
        print("  independent signal but is NOT the brief's industrial class.")
        print("  It is however enough to test whether an honest model has ANY")
        print("  signal at all, which model B could not demonstrate.")

_conf = _low.get("confidence")
if _conf:
    print(f"  `{_conf}` also present: letter-grade detection confidence, usable")
    print("  as a sample_weight and as a QA feature. NOT a cause label.")


# ---------------------------------------------------------------------------
# 3. WHAT EXTERNAL DATA ACTUALLY EXISTS
#
# A non-proximity target needs evidence that a fire caused SURFACE DAMAGE and
# did not recover, or that the fuel was not vegetation. That means land cover
# and/or NBR. Check whether any raster is already on Drive before planning a
# GEE run over 7.27M points.
# ---------------------------------------------------------------------------

print()
print("=" * 74)
print("3. EXTERNAL INVENTORY -- is there land cover / NBR / SAR yet?")
print("=" * 74)
if not cu.path_ok(EXTERNAL_ROOT):
    print(f"  {EXTERNAL_ROOT} is not visible.")
else:
    _folders = []
    for _p in sorted(EXTERNAL_ROOT.rglob("*")):
        if _p.is_dir():
            continue
        try:
            _size = _p.stat().st_size
        except OSError:
            continue
        _folders.append((str(_p.parent.relative_to(EXTERNAL_ROOT)), _p.suffix,
                         _size))
    _agg: dict = {}
    for _d, _ext, _size in _folders:
        _a = _agg.setdefault(_d, {"n": 0, "bytes": 0, "ext": set()})
        _a["n"] += 1
        _a["bytes"] += _size
        _a["ext"].add(_ext)
    print(f"  files: {len(_folders):,}   total "
          f"{sum(_s for _, _, _s in _folders) / 1024 ** 3:.2f} GB")
    print()
    print(f"  {'folder':<44} {'files':>6} {'GB':>8}  ext")
    for _d in sorted(_agg, key=lambda k: -_agg[k]["bytes"])[:25]:
        _a = _agg[_d]
        print(f"  {_d[:44]:<44} {_a['n']:>6} {_a['bytes'] / 1024 ** 3:>8.3f}  "
              f"{''.join(sorted(e for e in _a['ext'] if e))[:18]}")

    _RASTERISH = {".tif", ".tiff", ".nc", ".img", ".vrt"}
    _have_raster = sorted({_d for _d, _e, _sz in _folders
                           if _e.lower() in _RASTERISH})
    _lc = [_d for _d in _have_raster
           if any(t in _d.lower() for t in
                  ("landcover", "land_cover", "worldcover", "modis", "nbr",
                   "esa", "sentinel2", "s2_", "sar", "sentinel1", "s1_"))]
    print()
    print(f"  raster-bearing folders : {len(_have_raster)}")
    print(f"  land-cover / NBR / SAR : {_lc or 'NONE'}")
    if not _lc:
        print()
        print("  -> No land cover, no NBR, no SAR on Drive. A non-proximity")
        print("     label therefore has NO evidence source yet, and the honest")
        print("     options are (a) Earth Engine extraction, or (b) an")
        print("     independent external event catalogue. Both are downloads.")
        print("     Do NOT proceed to retrain against the current weak label")
        print("     expecting a better number; the ceiling is the distance rule.")


# ---------------------------------------------------------------------------
# 4. IS EARTH ENGINE AVAILABLE?
#
# Decides whether route (a) is possible. Cheap to check, and checking after
# planning a 7.27M-point extraction is how that plan becomes wasted work.
# ---------------------------------------------------------------------------

print()
print("=" * 74)
print("4. EARTH ENGINE -- usable for a land-cover extraction?")
print("=" * 74)
try:
    import ee
    # Colab cannot complete the OAuth flow unattended: it needs a click on a
    # Google consent screen. So ATTEMPT auth, and if it fails say exactly what
    # the user must run themselves rather than reporting "unavailable" -- the
    # two are very different situations and only one of them blocks the work.
    _authed = False
    try:
        ee.Initialize()
        _authed = True
        print("  ee.Initialize(): OK (already authenticated this session)")
    except Exception as _exc:                      # noqa: BLE001
        print(f"  ee.Initialize() failed: {type(_exc).__name__}")
        print()
        print("  Attempting ee.Authenticate() -- WATCH FOR A CONSENT SCREEN.")
        print("  If a link opens, accept it; the cell continues either way.")
        try:
            ee.Authenticate()
            ee.Initialize()
            _authed = True
            print("  ee.Authenticate(): OK")
        except Exception as _exc2:                 # noqa: BLE001
            print(f"  ee.Authenticate() failed: {type(_exc2).__name__}")
    if _authed:
        try:
            _first = ee.FeatureCollection("FAO/GAUL/2015/level0").first()
            print(f"  sanity read GAUL level0 first name: "
                  f"{_first.get('name').getInfo()}")
            _wc = ee.Image("ESA/WorldCover/v200").select("cover")
            print(f"  ESA/WorldCover/v200 band 'cover': "
                  f"{_wc.bandNames().getInfo()}")
            print("  -> GEE route VIABLE.")
            print("     Land cover + NBR (COPERNICUS/S2_SR_HARMONIZED) is the")
            print("     only evidence that distinguishes a real industrial")
            print("     surface loss from a proximity artefact. Sample it per")
            print("     event in tiles -- do NOT use Export.image, which will")
            print("     blow the quota over 7.27M points.")
        except Exception as _exc3:                 # noqa: BLE001
            print(f"  authenticated but a sanity read failed: "
                  f"{type(_exc3).__name__}: {_exc3}")
    else:
        print()
        print("  NOT AUTHENTICATED. To fix, run this in a cell yourself:")
        print("      import ee")
        print("      ee.Authenticate()          # click the consent screen")
        print("      ee.Initialize()")
        print("  or, on the command line, run:  earthengine authenticate")
        print("  NOTE: a Colab runtime that cannot reach the OAuth page will")
        print("  never authenticate. If the consent screen does not open, this")
        print("  route needs a local runtime instead.")
except ImportError:
    print("  earthengine-api NOT INSTALLED -> !pip install earthengine-api")

print()
print("=" * 74)
print("END -- paste the whole output above.")
print("=" * 74)
print("No files were written. Nothing was retrained.")