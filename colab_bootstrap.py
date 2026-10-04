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
}

print("=" * 70)
print("BOOTSTRAP")
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
print("  all modules agree")
print()
print(cu.drive_report())
print()
print("BOOTSTRAP OK -- next: colab_p3a_industrial.py, colab_p4a_lineage.py")
print("=" * 70)