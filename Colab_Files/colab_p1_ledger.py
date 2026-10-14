# ============================================================================
# P1 -- COLAB CELL
# Paste this into the Colab notebook that owns the 141-feature extractor.
#
# Purpose
#   1. Load the frozen ledger module.
#   2. DRIFT CHECK: compare the module schema against whatever FEATURES_141 the
#      notebook already has in memory, and against cells 233/246 if present.
#   3. Emit the audit ledger + work list into the Drive QC directory.
#   4. Replace any ad-hoc FEATURES_141 definition with the frozen one.
#
# Before running: the module file must exist in the session. Either
#   !cp /content/drive/MyDrive/Firm_Clusters/pipeline/feature_ledger.py /content/
# or paste the file contents above this cell.
# ============================================================================

# ----------------------------------------------------------------------------
# 1. LOAD THE LEDGER MODULE
#
# Colab runs on a different machine from your laptop. feature_ledger.py must
# exist IN THE SESSION. Four fallbacks are tried in order, so this cell cannot
# fail just because the file is not on Drive yet:
#
#   A. already at /content/feature_ledger.py
#   B. already importable
#   C. found on Drive (several known locations)
#   D. upload from your laptop via the Colab file picker
# ----------------------------------------------------------------------------

from pathlib import Path
import importlib
import json
import shutil
import sys

import pandas as pd

MODULE_NAME = "feature_ledger"
MODULE_FILE = Path("/content/feature_ledger.py")

# Candidate Drive locations, most likely first.
DRIVE_CANDIDATES = [
    Path("/content/drive/MyDrive/Firm_Clusters/pipeline/feature_ledger.py"),
    Path("/content/drive/MyDrive/Firm_Clusters/feature_ledger.py"),
    Path("/content/drive/MyDrive/konsors-sih2/feature_ledger.py"),
    Path("/content/drive/MyDrive/pipeline/feature_ledger.py"),
    Path("/content/drive/MyDrive/feature_ledger.py"),
]

MODULE_SOURCE = None

# The version this cell needs. Declared before loading so tier A can reject a
# stale /content copy instead of silently preferring it.
EXPECTED_LEDGER_VERSION = "1.2.0"


def _ledger_version_in(path):
    """Read LEDGER_VERSION out of a .py file without importing it."""
    import re as _re

    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return None
    m = _re.search(r'^LEDGER_VERSION\s*=\s*["\']([^"\']+)["\']', text, _re.M)
    return m.group(1) if m else None


def _version_parts(v):
    out = []
    for chunk in str(v).split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def _is_usable(found):
    """True when a file's declared version is at least what this cell needs.

    A newer file is usable. Only an older one is stale, so gaining a status in
    the ledger never forces every cell to be re-pasted.
    """
    if found is None:
        return False
    f = _version_parts(found)
    e = _version_parts(EXPECTED_LEDGER_VERSION)
    width = max(len(f), len(e))
    f = f + (0,) * (width - len(f))
    e = e + (0,) * (width - len(e))
    return f >= e


# ---- A. already on disk in the session
# Existence is not enough: an older /content copy survives a runtime restart and
# would be preferred forever, so its declared version must be checked too.
if MODULE_FILE.exists():
    found = _ledger_version_in(MODULE_FILE)
    if _is_usable(found):
        MODULE_SOURCE = "already present in /content"
    else:
        print("=" * 70)
        print(f"STALE copy in /content (has {found}, needs "
              f"{EXPECTED_LEDGER_VERSION} or newer) -- ignoring it")
        print("=" * 70)
        MODULE_FILE.unlink()
        if MODULE_NAME in sys.modules:
            del sys.modules[MODULE_NAME]

# ---- B. already importable
if MODULE_SOURCE is None and MODULE_NAME in sys.modules:
    MODULE_SOURCE = "already imported in this kernel"

# ---- C. on Drive at a known location
# Verified, not trusted: an out-of-date copy on Drive is the single most likely
# cause of a confusing AttributeError several cells later.
if MODULE_SOURCE is None:
    for candidate in DRIVE_CANDIDATES:
        if not candidate.exists():
            continue
        found = _ledger_version_in(candidate)
        if not _is_usable(found):
            print(f"  skipping stale Drive copy: {candidate} (has {found})")
            continue
        shutil.copy(candidate, MODULE_FILE)
        MODULE_SOURCE = f"copied from Drive: {candidate}"
        break

# ---- D. nowhere usable -> ask the user to upload it
if MODULE_SOURCE is None:
    print("=" * 70)
    print(f"feature_ledger.py v{EXPECTED_LEDGER_VERSION} NOT FOUND in the session")
    print("=" * 70)
    print("Upload it now: a file picker will open. Pick feature_ledger.py")
    print("from your laptop (the repo folder konsors-sih2).")
    print()
    from google.colab import files

    uploaded = files.upload()

    # macOS/Windows append " (2)" to a duplicated filename, so an exact key
    # lookup fails on the second upload of the same name.
    import re as _re

    key = next(
        (k for k in uploaded if _re.fullmatch(rf"{MODULE_NAME}.*\.py", k)), None
    )
    if key is None:
        raise FileNotFoundError(
            f"Expected {MODULE_NAME}.py in the upload, got {list(uploaded)}"
        )

    data = uploaded[key]
    MODULE_FILE.write_bytes(data)
    MODULE_SOURCE = f"uploaded from laptop ({key!r})"

    found = _ledger_version_in(MODULE_FILE)
    if not _is_usable(found):
        raise RuntimeError(
            f"uploaded {MODULE_NAME}.py declares version {found}, but this "
            f"cell needs {EXPECTED_LEDGER_VERSION} or newer. Upload the "
            f"current file from the repo, not an old copy."
        )

print("=" * 70)
print("Ledger module source:", MODULE_SOURCE)
print("=" * 70)

# ---- Version guard --------------------------------------------------------
# EXPECTED_LEDGER_VERSION is declared above, before loading, so the loader
# itself could reject a stale copy. This is the final check on whatever was
# actually imported.
actual_version = getattr(fl, "LEDGER_VERSION", "<missing>")

print(f"Ledger version: {actual_version}  (this cell expects "
      f"{EXPECTED_LEDGER_VERSION})")


def _too_old(found, expected):
    """True when `found` is strictly older than `expected`.

    Delegates to the same version comparison the loader uses, so the final
    guard and the file checks can never disagree. Compared as a version tuple
    rather than for string equality: a newer module than the cell expects is
    fine -- the cell asks for a minimum, not an exact match -- so a ledger that
    gains a status does not lock the pipeline until every cell is re-pasted.
    The failure this replaces was a cell expecting 1.1.0 raising "STALE ...
    found 1.2.0" at a module that was in fact current and correct.
    """
    if found is None:
        return True
    f, e = _version_parts(found), _version_parts(expected)
    width = max(len(f), len(e))
    f = f + (0,) * (width - len(f))
    e = e + (0,) * (width - len(e))
    return f < e


if _too_old(actual_version, EXPECTED_LEDGER_VERSION):
    raise RuntimeError(
        "STALE feature_ledger.py IN THE SESSION.\n"
        f"  expected : {EXPECTED_LEDGER_VERSION} or newer\n"
        f"  found    : {actual_version}\n"
        "\n"
        "Your copy on Drive is out of date. Fix it with ONE of:\n"
        "\n"
        "  A) re-upload from your laptop (overwrite on Drive):\n"
        "       from google.colab import files\n"
        "       from pathlib import Path\n"
        "       up = files.upload()\n"
        "       data = up['feature_ledger.py']\n"
        "       Path('/content/drive/MyDrive/Firm_Clusters/pipeline/"
        "feature_ledger.py').write_bytes(data)\n"
        "       Path('/content/feature_ledger.py').write_bytes(data)\n"
        "\n"
        "  B) or just upload it straight to the session when this cell\n"
        "     prompts for the file picker.\n"
        "\n"
        "Then restart the runtime (Runtime > Restart session) and re-run\n"
        "this cell from the top."
    )

# Optional attribute guard: report anything the cell depends on that the
# loaded module does not provide.
_missing_attrs = [
    name for name in
    ("LOCAL_INPUT_HINTS", "hints_for", "NO_GEE_REQUIRED",
     "KNOWN_EXTERNAL_SOURCES", "check_schema_drift")
    if not hasattr(fl, name)
]
if _missing_attrs:
    raise RuntimeError(
        f"STALE feature_ledger.py: missing {_missing_attrs}. "
        f"Loaded version {actual_version}."
    )

print("Version check passed.")
print()

# Fresh import every time, so an edited file is actually picked up.
for stale in [m for m in sys.modules if m == MODULE_NAME]:
    del sys.modules[stale]

sys.path.insert(0, "/content")
import feature_ledger as fl

importlib.reload(fl)

print("Ledger module loaded from:", MODULE_FILE)
print("features:", fl.N_FEATURES)
print("groups  :", len(fl.EXPECTED_GROUP_SIZES))
print("non-leak:", len(fl.NON_LEAK_FEATURES))
print("label-derived:", len(fl.LABEL_DERIVED_FEATURES))

# ----------------------------------------------------------------------------
# 2. DRIFT CHECK AGAINST THE NOTEBOOK'S OWN SCHEMA
# ----------------------------------------------------------------------------

FROZEN = fl.FEATURES_141

notebook_schema = None

for name in ("FEATURES_141", "FEATURES", "X_EVENT", "X_event"):
    if name in globals():
        candidate = list(globals()[name])
        if candidate and isinstance(candidate[0], str):
            notebook_schema = candidate
            print(f"Found in-session schema under '{name}' "
                  f"({len(candidate)} names)")
            break

if notebook_schema is None:
    print("No in-session schema found -- nothing to diff. "
          "This is fine on a fresh kernel.")
    drift = {"status": "NO_CANDIDATE"}
else:
    drift = fl.check_schema_drift(notebook_schema)

    print("=" * 70)
    print("SCHEMA DRIFT CHECK")
    print("=" * 70)
    print("result    :", drift["status"])
    print("same order:", drift.get("same_order"))
    print("features  :", drift.get("n_features", len(notebook_schema)))

    if not drift.get("same_order", True):
        print("Same names, different order -> adopting frozen order.")

# ----------------------------------------------------------------------------
# 3. RETIRE THE OLD '131 FEATURE' CLAIM
# ----------------------------------------------------------------------------

assert len(FROZEN) == 141
assert len(set(FROZEN)) == 141
print()
print("Confirmed: 141 features, 141 unique.")
print("The '131 feature' figure in older handover notes is an arithmetic")
print("error and must not be used again.")

# ----------------------------------------------------------------------------
# 4. REPLACE THE IN-SESSION SCHEMA WITH THE FROZEN ONE
# ----------------------------------------------------------------------------

FEATURES_141 = list(FROZEN)
FEATURES = list(FROZEN)

print()
print("FEATURES_141 rebound to the frozen ledger order.")

# ----------------------------------------------------------------------------
# 5. PERSIST THE MODULE TO DRIVE, THEN WRITE THE LEDGER
# ----------------------------------------------------------------------------

# Saving the module back to Drive means every future session finds it at
# fallback C and never needs the upload step again.
PIPELINE_DIR = Path("/content/drive/MyDrive/Firm_Clusters/pipeline")
PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

MODULE_DRIVE_COPY = PIPELINE_DIR / "feature_ledger.py"
MODULE_DRIVE_COPY.write_bytes(MODULE_FILE.read_bytes())
print("Module persisted to:", MODULE_DRIVE_COPY)

QC_DIR = Path("/content/drive/MyDrive/Firm_Clusters/qc")
QC_DIR.mkdir(parents=True, exist_ok=True)

records = fl.build_audit_records()
audit = pd.DataFrame(records, columns=fl.LEDGER_COLUMNS)

audit_path = QC_DIR / "feature_audit_141.parquet"
audit.to_parquet(audit_path, index=False)

audit_csv = QC_DIR / "feature_audit_141.csv"
audit.to_csv(audit_csv, index=False)

schema_path = QC_DIR / "FEATURES_141.json"
schema_path.write_text(
    json.dumps(
        {
            "features": FROZEN,
            "group_of": fl.GROUP_OF,
            "dataset_of": fl.DATASET,
            "label_derived": fl.LABEL_DERIVED_FEATURES,
            "non_leak": fl.NON_LEAK_FEATURES,
        },
        indent=2,
    )
)

# ----------------------------------------------------------------------------
# 6. REPORT
# ----------------------------------------------------------------------------

print("=" * 70)
print("GATE P1 -- FEATURE AUDIT LEDGER")
print("=" * 70)

print("rows:", len(audit))
print("counts by status:")
for status, count in audit["status"].value_counts().items():
    print(f"  {status:<24} {count:>4}")

print()
print("counts by group:")
for group in fl.EXPECTED_GROUP_SIZES:
    sub = audit[audit["group"] == group]
    detail = ", ".join(
        f"{k}={v}" for k, v in sub["status"].value_counts().items()
    )
    print(f"  {group:<24} {len(sub):>4}  ({detail})")

print()
print("open work by phase:")
for phase, sub in audit[audit["phase"] != ""].groupby("phase"):
    print(f"  {phase:<24} {len(sub):>4}")

print()
print("label-derived (leakage partition) :",
      audit["label_derived"].sum(), "features")
print("non-leak (scientific partition)  :",
      (~audit["label_derived"]).sum(), "features")

# ----------------------------------------------------------------------------
# 7. GATE ASSERTIONS
# ----------------------------------------------------------------------------

assert len(audit) == 141, "ledger must have 141 rows"
assert audit["feature"].nunique() == 141, "duplicate features"
assert list(audit["feature"]) == FROZEN, "order drift"
assert not audit["status"].isin(["UNSET", "", None]).any()
assert audit["status"].isin(fl.VALID_STATUSES).all()
assert audit["dataset"].notna().all()
assert audit["era_start"].notna().all()
assert not audit["runtime_verified"].any(), (
    "runtime_verified must stay False until P2/P4 verification passes"
)

print()
print("Wrote:")
for p in (audit_path, audit_csv, schema_path):
    print("  ", p)

print()
print("GATE P1 PASSED")
print()
print("Next: P0 -- recover cell 262 (extract_fire_features),")
print("then P2 -- run the 10-event forensic test and clear")
print("runtime_verified for every feature it exercises.")