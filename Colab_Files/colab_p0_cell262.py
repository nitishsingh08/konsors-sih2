# ============================================================================
# P0 -- COLAB CELL: search the Drive QC artifacts for the complete
#      extract_fire_features() (historical cell 262)
#
# Run this AFTER the P1 cell. It searches every known QC JSON for the missing
# function body and prints the largest contiguous match, so you can paste it
# back verbatim instead of inventing VIIRS band names.
# ============================================================================

from pathlib import Path
import json
import re

QC_DIR = Path("/content/drive/MyDrive/Firm_Clusters/qc")

TARGET_FILES = [
    "historical_feature_code_recovery_2026-09-27.json",
    "feature_formula_implementation_audit.json",
    "feature_groups_implementation_discovery.json",
    "feature_implementation_function_trace.json",
    "141_feature_exact_implementation_trace.json",
    "141_feature_authoritative_definition_extract.json",
    "141_feature_missing_gap_recovery.json",
]

# Search every .json under qc/ too, in case new artifacts were added.
all_json = sorted(QC_DIR.rglob("*.json")) if QC_DIR.exists() else []
targets = [QC_DIR / name for name in TARGET_FILES]
targets += [p for p in all_json if p not in targets]

if not targets:
    raise FileNotFoundError(
        f"No QC artifacts found under {QC_DIR}. "
        "Check that the drive is mounted and the path is correct."
    )

TERMS = [
    "def extract_fire_features",
    "extract_fire_features",
    "VIIRS_FIRE",
    "VNP14A1",
    "MaxFRP",
    "FRP_MEDIAN",
    "FRP_P90",
    "FRP_MAX",
    "I4",
    "I5",
]

print("=" * 70)
print("P0 -- RECOVERING extract_fire_features (historical cell 262)")
print("=" * 70)
print(f"scanning {len(targets)} JSON artifacts under {QC_DIR}")
print()

hits = []

for path in targets:
    if not path.exists():
        continue
    try:
        text = path.read_text(errors="ignore")
    except OSError as exc:
        print(f"  skip {path.name}: {exc}")
        continue

    found = [t for t in TERMS if t in text]
    if not found:
        continue

    print("-" * 70)
    print(f"{path.name}   ({len(text):,} chars)")
    print(f"  matched: {found}")

    # Extract the largest contiguous block containing the function def.
    idx = text.find("def extract_fire_features")
    if idx != -1:
        # Walk braces to find the end of the function body.
        start = idx
        depth = 0
        seen_def = False
        end = None
        for i in range(idx, min(len(text), idx + 200000)):
            ch = text[i]
            if ch == "{":
                depth += 1
                seen_def = True
            elif ch == "}":
                depth -= 1
                if seen_def and depth == 0:
                    end = i + 1
                    break
        block = text[start:end] if end else text[start:start + 20000]
        hits.append((path.name, len(block), block))
        print(f"  >>> function body found, {len(block):,} chars")
    else:
        hits.append((path.name, 0, ""))
    print()

# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------

print("=" * 70)
print("RESULT")
print("=" * 70)

bodies = [(n, L, b) for n, L, b in hits if L > 0]

if not bodies:
    print()
    print("No complete function body found in the QC artifacts.")
    print()
    print("DECISION: leave extract_fire_features as a 3-key NaN stub.")
    print("The ledger already records FRP_MEDIAN / FRP_P90 / FRP_MAX as")
    print("MISSING_IMPLEMENTATION under phase P0, so nothing breaks.")
    print("Move on to P3a (industrial, 18 features, no GEE cost).")
    print()
    print("Do NOT invent VIIRS band names. If you want FRP from imagery")
    print("instead, inspect the collection yourself first:")
    print()
    print("    print(VIIRS_FIRE.first().bandNames().getInfo())")
    print("    print(MODIS_FIRE.first().bandNames().getInfo())")
    print()
    print("and then write extract_fire_features as a NEW implementation,")
    print("updating the ledger status to NEW_IMPLEMENTATION.")

else:
    bodies.sort(key=lambda t: t[1], reverse=True)
    best_name, best_len, best_block = bodies[0]

    print(f"best artifact : {best_name}")
    print(f"body length   : {best_len:,} chars")
    print()
    print("=" * 70)
    print("EXTRACTED FUNCTION")
    print("=" * 70)
    print(best_block)
    print("=" * 70)
    print()
    print("NEXT STEPS")
    print("  1. Paste the function above into a NEW cell.")
    print("  2. Replace only the date construction:")
    print("         start = ee.Date(str(event['start_time']))")
    print("         end   = ee.Date(str(event['end_time'])).advance(1, 'day')")
    print("     with:")
    print("         start = ee.Date(ee_timestamp(event['start_time']))")
    print("         end   = ee.Date(ee_timestamp(event['end_time'])).advance(1, 'day')")
    print("  3. Verify the collection IDs match the ones already restored:")
    print("         MODIS_FIRE = ee.ImageCollection('MODIS/061/MOD14A1')")
    print("         VIIRS_FIRE = ee.ImageCollection('NASA/VIIRS/002/VNP14A1')")
    print("  4. Single-event test:")
    print("         fire = extract_fire_features(test_event, test_geometry)")
    print("         assert isinstance(fire, dict)")
    print("         assert {'FRP_MEDIAN','FRP_P90','FRP_MAX'} <= set(fire)")
    print("  5. Update the ledger status for those 3 features to")
    print("     COMPUTED_HISTORICAL in feature_ledger.py, re-upload, re-run P1.")