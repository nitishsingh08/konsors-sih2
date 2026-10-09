# ============================================================================
# BOOTSTRAP -- install every repo module, then verify them against each other.
#
# Self-healing by design: it installs EVERY .py it finds in the upload AND
# every .py already in Drive/pipeline. It has no MODULES list, so a module
# added to the repo later cannot be silently skipped by a stale pasted cell --
# which is exactly the failure that produced three rounds of
# `ModuleNotFoundError` in this project.
#
# Upload whatever .py files you have, then run this. It will tell you precisely
# which module is missing and where to get it, rather than dying on the first
# import.
# ============================================================================

import re
import sys
from pathlib import Path

from google.colab import files

PIPELINE = Path("/content/drive/MyDrive/Firm_Clusters/pipeline")
LOCAL = Path("/content")
DRIVE_OK = PIPELINE.exists()

#: Modules the cells import, with the minimum acceptable ledger version.
#: Every module the pipeline imports. A module missing from this list is not
#: reported until some later cell dies with a bare ImportError, which is how
#: cluster_tier / weak_labels / dataset_builder went missing twice.
REQUIRED = {
    "feature_ledger": None,      # version checked separately
    "industrial_features": None,
    "lineage_features": None,
    "cluster_tier": None,
    "fire_labels": None,
    "weak_labels": None,
    "dataset_builder": None,
    "training_design": None,
    "feature_gap": None,
    "rule_calibration": None,
    "colab_utils": None,
    # Added 2026-10-04. These were missing, which is why a PARTIAL upload went
    # unnoticed: the cell reported success while `threshold_calibration`,
    # `gold_set` and `gee_utils` were never installed, and the failure only
    # appeared much later as an unrelated-looking error. The list must cover
    # every module the later cells import -- adding a module without adding it
    # here reintroduces exactly this hole.
    "threshold_calibration": None,
    "gold_set": None,
    "gee_utils": None,
    "quality_features": None,
    "explainability": None,
    "colab_runcell": None,
    # Added 2026-10-04 with P8/P8b: the figure set and the leakage probe. Same
    # hole, same rule -- if these are absent the cell runs and then fails on an
    # ImportError that looks like a code bug rather than a missing upload.
    "figures": None,
    "p8_report": None,
    "leakage_probe": None,
}

#: Minimum acceptable version per CELL, checked against the uploaded bytes.
#:
#: Modules are verified by `hasattr`, which catches a stale copy. Cells have no
#: such handle: a cell is a script, so an OLD cell still runs, still imports
#: fine, and still fails in a way that looks like a live bug. That is exactly
#: what happened on 2026-10-04 -- a 14,229-byte copy of `colab_p7e_gee_probe`
#: was uploaded against a 21,060-byte repo file, and the failure
#: (`Asset 'ESA/WorldCover/v200' is not an Image`) was reported as a dataset
#: problem when it was a file-transfer problem.
#:
#: Bump these when the corresponding cell changes. A cell with NO version
#: marker at all is always rejected: an unversioned cell cannot be checked, and
#: an uncheckable cell is exactly what caused this.
CELL_VERSION_FLOORS = {
    # No cell is exempt. An unversioned cell cannot be verified, which is
    # precisely what made the 14,229-byte upload undetectable.
    "colab_p7b_diagnose": "2026-10-04.1",
    "colab_p7c_goldset": "2026-10-04.1",
    "colab_p7d_probe": "2026-10-04.1",
    "colab_p7e0_reauth": "2026-10-04.1",
    "colab_p7e_gee_probe": "2026-10-04.4",
    "colab_p7f_landcover_direct": "2026-10-04.1",
    "colab_p7g_landcover_vs_label": "2026-10-04.3",
    "colab_p8_figures": "2026-10-04.1",
    "colab_p8b_leakage": "2026-10-04.1",
}

#: Printed on every successful bootstrap, so a stale copy in Colab announces
#: itself on the first line instead of being inferred from a later failure.
BOOTSTRAP_VERSION = "2026-10-05.2"

print("=" * 70)
print(f"BOOTSTRAP  [v{BOOTSTRAP_VERSION}]")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. Collect candidates: the upload first, then whatever is already on Drive.
# ---------------------------------------------------------------------------

candidates = {}   # module name -> (source, bytes)

uploaded = files.upload()
print(f"upload dialog returned {len(uploaded)} file(s): {sorted(uploaded)}")

for key, payload in uploaded.items():
    if not key.endswith(".py"):
        continue
    # macOS/Windows append " (2)" to a duplicated filename, so strip any suffix
    # before matching rather than doing an exact key lookup.
    stem = re.sub(r"\s*\(\d+\)$", "", Path(key).stem)
    candidates[stem] = ("upload", payload)

if DRIVE_OK:
    for path in sorted(PIPELINE.glob("*.py")):
        stem = path.stem
        if stem in candidates:
            continue          # the upload is newer, keep it
        try:
            candidates[stem] = ("drive", path.read_bytes())
        except OSError as exc:
            print(f"  !! could not read {path}: {exc}")
else:
    print()
    print("  Drive pipeline dir not visible; relying on the upload alone.")

# ---------------------------------------------------------------------------
# 2. Install, reporting anything still missing.
# ---------------------------------------------------------------------------

PIPELINE.mkdir(parents=True, exist_ok=True)
installed, missing = [], []

for mod in REQUIRED:
    entry = candidates.get(mod)
    if entry is None:
        missing.append(mod)
        continue
    source, payload = entry
    (LOCAL / f"{mod}.py").write_bytes(payload)
    (PIPELINE / f"{mod}.py").write_bytes(payload)
    sys.modules.pop(mod, None)
    installed.append(mod)
    print(f"  {mod:22s} <- {source:6s} ({len(payload):,} bytes)")

# Anything else that showed up is installed too, so a future module works
# without editing this cell.
for mod, (source, payload) in sorted(candidates.items()):
    if mod in installed:
        continue
    (LOCAL / f"{mod}.py").write_bytes(payload)
    (PIPELINE / f"{mod}.py").write_bytes(payload)
    sys.modules.pop(mod, None)
    print(f"  {mod:22s} <- {source:6s} ({len(payload):,} bytes) [extra]")

print()
if missing:
    print("=" * 70)
    print(f"MISSING: {missing}")
    print("=" * 70)
    print("Re-run the file picker and select ALL of these together, from the")
    print("repo folder konsors-sih2:")
    for mod in missing:
        print(f"  - {mod}.py")
    print()
    print("The picker returns only what you highlight, not the whole folder.")
    raise SystemExit(1)

print(f"installed: {installed}")

# ---------------------------------------------------------------------------
# 3. Verify. Import failures here name the module instead of crashing a
#    feature cell three steps later.
# ---------------------------------------------------------------------------

print()
missing_import = []
for mod in REQUIRED:
    try:
        __import__(mod)
    except Exception as exc:                      # noqa: BLE001
        missing_import.append(f"{mod} ({exc.__class__.__name__}: {exc})")

if missing_import:
    print("=" * 70)
    print("IMPORT FAILED")
    print("=" * 70)
    for line in missing_import:
        print(f"  {line}")
    print()
    print("A file installed but will not import is usually a syntax error or a")
    print("missing dependency. Re-upload that module from the repo.")
    raise SystemExit(1)

import colab_utils as cu
import cluster_tier as ct
import dataset_builder as db
import feature_gap as fgap
import rule_calibration as rcal
import feature_ledger as fl
import fire_labels as lab
import industrial_features as ind
import lineage_features as lf
import training_design as td
import weak_labels as weak
import threshold_calibration as tcal
import gold_set as gs
import gee_utils as gee
import quality_features as qf
import explainability as expl
import leakage_probe as leak
import figures as figs


def version_parts(v):
    out = []
    for chunk in str(v).split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def too_old(found, expected):
    if found is None:
        return True
    f, e = version_parts(found), version_parts(expected)
    w = max(len(f), len(e))
    f, e = f + (0,) * (w - len(f)), e + (0,) * (w - len(e))
    return f < e


print()
print("=" * 70)
print("CROSS-MODULE CHECKS")
print("=" * 70)

problems = []

if too_old(fl.LEDGER_VERSION, "1.3.0"):
    problems.append(
        f"stale feature_ledger: {fl.LEDGER_VERSION} < 1.3.0 -- re-upload it"
    )
if not hasattr(fl, "LOCAL_INPUT_HINTS"):
    problems.append("stale feature_ledger: no LOCAL_INPUT_HINTS")
if not hasattr(fl, "S_NEW_IMPL"):
    problems.append("stale feature_ledger: no NEW_IMPLEMENTATION status")
if ind.INDUSTRIAL_FEATURES != fl.FEATURE_GROUPS[fl.GROUP_INDUSTRIAL]:
    problems.append("industrial feature order drifted from the ledger")
if ind.REQUIRED_CRS != "EPSG:6933":
    problems.append(f"unexpected CRS {ind.REQUIRED_CRS}")
if lf.N_LINEAGE != 54:
    problems.append(f"lineage block is {lf.N_LINEAGE}, expected 54")
if not set(lf.LINEAGE_FEATURES) <= set(fl.FEATURES_141):
    problems.append("lineage features are not a subset of the ledger")
if len(lab.LABEL_CLASSES) != 6:
    problems.append(f"{len(lab.LABEL_CLASSES)} label classes, expected 6")
if 107 not in cu.DEAD_MOUNT_ERRNOS:
    problems.append("colab_utils is missing the errno-107 dead-mount entry")
if ind.INDUSTRIAL_FEATURES != fl.FEATURE_GROUPS[fl.GROUP_INDUSTRIAL]:
    problems.append("industrial block does not match the ledger")
if ct.N_CLUSTER_TIER + len(weak.LABEL_INPUT_FEATURES) < 1:
    problems.append("cluster_tier / weak_labels look empty")
if len(fl.FEATURES_141) != 141:
    problems.append(f"ledger has {len(fl.FEATURES_141)} features, expected 141")
if not hasattr(db, "SITE_DEG") or not hasattr(fgap, "gap_report"):
    problems.append("dataset_builder / feature_gap are stale")
if not hasattr(rcal, "calibrate_to_target_counts"):
    problems.append("rule_calibration is stale")

# --- the 2026-10-04 modules -------------------------------------------------
# Each of these was added after the original REQUIRED list was written, and
# their absence is exactly what let a partial upload pass unnoticed. Check the
# SPECIFIC attribute a later cell depends on, not merely that the import
# worked: an old copy imports fine and then fails somewhere else entirely.
if not hasattr(tcal, "calibrate_operating_threshold"):
    problems.append("threshold_calibration is stale -- no calibrate_operating_threshold")
if not hasattr(tcal, "effective_threshold"):
    problems.append("threshold_calibration is stale -- no effective_threshold")
if not hasattr(rcal, "scale_targets"):
    problems.append("rule_calibration is stale -- no scale_targets "
                    "(catalogue targets will not be scaled onto the "
                    "calibration frame and every threshold pins to the floor)")
if not hasattr(rcal, "DEFAULT_POPULATION_SIZE"):
    problems.append("rule_calibration is stale -- no DEFAULT_POPULATION_SIZE")
if not hasattr(gs, "build_sample") or not hasattr(gs, "score_against_gold"):
    problems.append("gold_set is stale -- no build_sample / score_against_gold")
if not hasattr(gs, "GOLD_CLASSES"):
    problems.append("gold_set is stale -- no GOLD_CLASSES")
if not hasattr(gee, "authenticate") or not hasattr(gee, "verify_can_read"):
    problems.append("gee_utils is stale -- no authenticate / verify_can_read")
if not hasattr(gee, "EEPermissionError"):
    problems.append("gee_utils is stale -- no EEPermissionError "
                    "(a 403 will not be reported clearly)")
if not hasattr(gee, "NBR_BANDS") or not hasattr(gee, "dataset_kind"):
    problems.append("gee_utils is stale -- no NBR_BANDS / dataset_kind")
if not hasattr(gee, "project_ids_from_dotenv"):
    problems.append("gee_utils is stale -- no project_ids_from_dotenv")
if len(getattr(qf, "QUALITY_FEATURES", []) or []) == 0:
    problems.append("quality_features looks empty (no QUALITY_FEATURES)")
if not hasattr(expl, "shap_explain") or not hasattr(expl, "top_contributions"):
    problems.append("explainability is stale -- no shap_explain / top_contributions")
if not hasattr(leak, "cross_split_proximity") or not hasattr(leak, "probe"):
    problems.append("leakage_probe is stale -- no cross_split_proximity / probe")
if not hasattr(leak, "RADII_KM"):
    problems.append("leakage_probe is stale -- no RADII_KM")
if not hasattr(figs, "all_metrics") or not hasattr(figs, "full_confusion_report"):
    problems.append("figures is stale -- no all_metrics / full_confusion_report "
                    "(the P8 confusion report would silently lack most measures)")
if not hasattr(__import__("colab_runcell"), "run"):
    problems.append("colab_runcell is stale -- no run()")

if problems:
    print("FAILED:")
    for line in problems:
        print(f"  - {line}")
    raise SystemExit(1)

print(f"  ledger      : {fl.LEDGER_VERSION}")
print(f"  industrial  : {ind.N_INDUSTRIAL} features")
print(f"  lineage     : {lf.N_LINEAGE} features")
print(f"  labels      : {len(lab.LABEL_CLASSES)} classes")
print(f"  weak labels : {len(weak.LABEL_INPUT_FEATURES)} rule inputs published")
print(f"  site grid   : {db.SITE_DEG} deg")
print(f"  calibration: {len(rcal.DEFAULT_PERCENTILES)} thresholds derived")
print(f"  thresholds  : {hasattr(tcal, 'calibrate_operating_threshold')} "
      f"(target-scaling fix: {hasattr(rcal, 'scale_targets')})")
print(f"  gold set    : {len(gs.GOLD_CLASSES)} classes, "
      f"boundary band {gs.BOUNDARY_BAND_M}")
print(f"  gee_utils   : {gee.__version__}  "
      f"(permission-aware: {hasattr(gee, 'EEPermissionError')})")


# ---------------------------------------------------------------------------
# CELL VERSIONS -- the check that would have caught the 14,229-byte upload
#
# Modules are verified by hasattr. A CELL is a script, so an old copy runs
# happily and fails in a way that reads as a live bug. So each uploaded cell is
# read back off Drive and its declared version compared against the floor.
# ---------------------------------------------------------------------------
import re as _re

_VERSION_RE = _re.compile(
    r"([A-Z][A-Z0-9_]*_VERSION|__version__)\s*=\s*[\"']([^\"']+)[\"']")


def _declared_version(text):
    for _n, v in _VERSION_RE.findall(text):
        return v
    return None


def _floor_ok(found, floor):
    """Version-tuple ordering, so 1.10.0 > 1.2.0."""
    def parts(v):
        return tuple(int("".join(c for c in ch if c.isdigit()) or 0)
                     for ch in str(v).replace("-", ".").split("."))
    return parts(found) >= parts(floor)


print()
print("=" * 70)
print("CELL VERSIONS (read back off Drive)")
print("=" * 70)
cell_problems = []
_floors = CELL_VERSION_FLOORS
for _stem, _floor in sorted(_floors.items()):
    _path = PIPELINE / f"{_stem}.py"
    if not _path.exists():
        cell_problems.append(f"{_stem}: not uploaded at all")
        print(f"  {_stem:<28} NOT UPLOADED")
        continue
    _v = _declared_version(_path.read_text(errors="replace"))
    if _floor is None:
        _note = "no floor (exempt)"
        print(f"  {_stem:<28} {_v or 'no marker':<20} {_note}")
        continue
    if _v is None:
        cell_problems.append(
            f"{_stem}: NO VERSION MARKER -- this is an old copy that predates "
            f"versioning, or a different file. Cannot be verified.")
        print(f"  {_stem:<28} NO MARKER  <-- STALE")
        continue
    if not _floor_ok(_v, _floor):
        cell_problems.append(f"{_stem}: version {_v} < required {_floor}")
        print(f"  {_stem:<28} {_v:<20} STALE (need >= {_floor})")
    else:
        print(f"  {_stem:<28} {_v:<20} ok")

if cell_problems:
    print()
    print("STALE CELLS -- these will run OLD code:")
    for _line in cell_problems:
        print(f"  - {_line}")
    print()
    print("A cell is pasted notebook TEXT, so editing the repo file changes")
    print("nothing about a copy you already uploaded or pasted. Re-upload the")
    print(".py files themselves, and run cells via colab_runcell rather than")
    print("pasting their bodies:")
    print()
    print("    import colab_runcell")
    print('    colab_runcell.run("colab_p7e_gee_probe")')
    raise SystemExit(1)
print("  all modules agree")
print()
print(cu.drive_report())
print()
print("BOOTSTRAP OK -- next: colab_p3a_industrial.py, colab_p4a_lineage.py")
print("=" * 70)