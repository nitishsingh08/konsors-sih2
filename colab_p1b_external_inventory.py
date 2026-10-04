# ============================================================================
# P1b -- COLAB CELL: inventory the local `external/` tree
#
# Run this right after the P1 cell.
#
# Purpose
#   1. Find the external data you already downloaded/extracted under
#      Firm_Clusters/full_2012_2026/external (and any other likely location).
#   2. Report what is actually there: folders, counts, sizes, extensions.
#   3. Match it against the 141 features to answer the question that matters:
#      "how many of the 129 open features have their input already on disk?"
#   4. Write qc/external_data_inventory.json + qc/external_coverage_matrix.csv
#   5. Print the revised work order, with local-only features first.
#
# This changes the plan: any feature whose input is local can be implemented
# in plain pandas/scipy with no Earth Engine call and no per-event API cost.
# ============================================================================

from pathlib import Path
import json
import re
from collections import defaultdict

import pandas as pd

import feature_ledger as fl
import colab_utils as cu

print("Ledger version:", getattr(fl, "LEDGER_VERSION", "<missing>"))
if not hasattr(fl, "LOCAL_INPUT_HINTS"):
    print()
    print("=" * 70)
    print("STALE MODULE: feature_ledger.py has no LOCAL_INPUT_HINTS")
    print("=" * 70)
    print("You are running an old copy. Steps 1-4 (the folder listing) are")
    print("still valid, but the feature matching in step 5 cannot run.")
    print()
    print("Fix: re-upload the current feature_ledger.py, restart the runtime,")
    print("re-run the P1 cell, then re-run this one.")
    print()
    raise RuntimeError("stale feature_ledger.py")

# ---------------------------------------------------------------------------
# 1. LOCATE THE EXTERNAL TREE
# ---------------------------------------------------------------------------

# Resolved by content, like every other cell. The hardcoded SEARCH_ROOTS list
# broke twice: once when the tree moved to a duplicated folder, and once when a
# stale mount made a folder that plainly exists invisible.
_PROJECT = cu.resolve_project_dir()
SEARCH_ROOTS = []
external_root = None
searched = []

if _PROJECT is not None:
    hits = cu.find_all_artefact(_PROJECT, "external")
    SEARCH_ROOTS = [path for path, _ in hits]
    searched = list(SEARCH_ROOTS)
    if SEARCH_ROOTS:
        external_root = SEARCH_ROOTS[0]
    else:
        # Present but empty listings can be a stale mount; refresh once.
        cu.force_refresh_drive()
        hits = cu.find_all_artefact(_PROJECT, "external")
        SEARCH_ROOTS = [path for path, _ in hits]
        searched = list(SEARCH_ROOTS)
        external_root = SEARCH_ROOTS[0] if SEARCH_ROOTS else None
else:
    print("!! could not resolve the project root; run colab_survey_tree.py")
if external_root is None:
    SEARCH_ROOTS = searched

print("=" * 70)
print("P1b -- EXTERNAL DATA INVENTORY")
print("=" * 70)
print("searched:")
for p in searched:
    print(f"   {'FOUND ' if cu.is_dir(p) else 'absent'}  {p}")

if external_root is None:
    print()
    print("No `external/` directory found at any known location.")
    print("Set EXTERNAL_ROOT below to the real path and re-run this cell.")
    EXTERNAL_ROOT = None
else:
    EXTERNAL_ROOT = external_root
    print()
    print("Using:", EXTERNAL_ROOT)

# ---------------------------------------------------------------------------
# 2. WALK THE TREE
# ---------------------------------------------------------------------------

SKIP_DIRS = {".ipynb_checkpoints", "__pycache__", ".cache", ".ipynb"}

if EXTERNAL_ROOT is None:
    files = []
    subdirs = []
    per_dir = defaultdict(lambda: {"count": 0, "bytes": 0})
    ext_counts = defaultdict(int)
    ext_bytes = defaultdict(int)
else:
    files = []
    subdirs = []
    per_dir = defaultdict(lambda: {"count": 0, "bytes": 0})
    ext_counts = defaultdict(int)
    ext_bytes = defaultdict(int)

    for path in sorted(EXTERNAL_ROOT.rglob("*")):
        rel_parts = path.relative_to(EXTERNAL_ROOT).parts
        if any(part in SKIP_DIRS for part in rel_parts):
            continue

        if path.is_dir():
            subdirs.append(path)
            continue

        try:
            size = path.stat().st_size
        except OSError:
            size = 0

        files.append(path)

        top = rel_parts[0] if len(rel_parts) > 1 else "(root)"
        per_dir[top]["count"] += 1
        per_dir[top]["bytes"] += size

        ext = (path.suffix or "(none)").lower()
        ext_counts[ext] += 1
        ext_bytes[ext] += size

total_bytes = sum(p.stat().st_size for p in files) if files else 0

print()
print(f"subdirectories : {len(subdirs)}")
print(f"files          : {len(files):,}")
print(f"total size     : {total_bytes / 1024**3:.2f} GB"
      f"  ({total_bytes:,} bytes)")

# ---------------------------------------------------------------------------
# 3. PER-SUBFOLDER BREAKDOWN
# ---------------------------------------------------------------------------

print()
print("-" * 70)
print("BY TOP-LEVEL FOLDER")
print("-" * 70)

for name in sorted(per_dir):
    info = per_dir[name]
    print(f"  {name:<28} {info['count']:>7,} files  "
          f"{info['bytes'] / 1024**3:>8.2f} GB")

print()
print("-" * 70)
print("BY FILE EXTENSION")
print("-" * 70)

for ext in sorted(ext_counts, key=lambda e: -ext_bytes[e]):
    print(f"  {ext:<28} {ext_counts[ext]:>7,} files  "
          f"{ext_bytes[ext] / 1024**3:>8.2f} GB")

# ---------------------------------------------------------------------------
# 4. SAMPLE FILENAMES PER FOLDER (first 6 each)
# ---------------------------------------------------------------------------

print()
print("-" * 70)
print("SAMPLE FILENAMES")
print("-" * 70)

by_folder = defaultdict(list)
for path in files:
    rel = path.relative_to(EXTERNAL_ROOT)
    top = rel.parts[0] if len(rel.parts) > 1 else "(root)"
    by_folder[top].append(str(rel))

for folder in sorted(by_folder):
    samples = sorted(by_folder[folder])[:6]
    print(f"  [{folder}]")
    for sample in samples:
        print(f"      {sample}")
    if len(by_folder[folder]) > 6:
        print(f"      ... +{len(by_folder[folder]) - 6:,} more")

# ---------------------------------------------------------------------------
# 5. MATCH EXTERNAL CONTENT AGAINST THE 141 FEATURES
# ---------------------------------------------------------------------------

if files:
    haystack = "\n".join(
        str(p.relative_to(EXTERNAL_ROOT)) for p in files
    ).lower() + "\n" + "\n".join(
        str(p.relative_to(EXTERNAL_ROOT)) for p in subdirs
    ).lower()
else:
    haystack = ""

TOKEN_MATCHES = defaultdict(list)
for token in sorted({
    token
    for tokens in fl.LOCAL_INPUT_HINTS.values()
    for token in tokens
}):
    if token in haystack:
        TOKEN_MATCHES[token] = True

matched_tokens = set(TOKEN_MATCHES)

# ---------------------------------------------------------------------------
# 4b. SIZE SANITY CHECK
#
# A folder full of tiny rasters usually means the download produced empty or
# all-NaN files (wrong AOI, wrong variable name, clipped to nothing). Treat
# those as NOT available until proven otherwise -- otherwise the whole plan
# is built on empty files.
# ---------------------------------------------------------------------------

TRUSTED = set()
suspect = set()

for folder, floor_bytes in fl.SIZE_SANITY_FLOOR_BYTES.items():
    info = per_dir.get(folder)
    if info is None:
        continue
    verdict = "OK" if info["bytes"] >= floor_bytes else "SUSPECT: too small"
    print(f"  {folder:<26} {info['bytes'] / 1024**2:>9.1f} MB "
          f"(floor {floor_bytes / 1024**2:.0f} MB)  -> {verdict}")
    if info["bytes"] >= floor_bytes:
        TRUSTED.add(folder)
    else:
        suspect.add(folder)

if suspect:
    print()
    print("  Suspect folders are NOT counted as available input.")
    print("  Open one file from each and check it, e.g.:")
    for folder in sorted(suspect):
        sample = sorted(by_folder[folder])[:1]
        if sample:
            print(f"      xarray.open_dataset('{EXTERNAL_ROOT}/{sample[0]}')")
            print(f"      rasterio.open('{EXTERNAL_ROOT}/{sample[0]}').read()")
            print()

# Known-source tokens are only trusted if their folder passed the size check.
for folder, tokens in fl.KNOWN_EXTERNAL_SOURCES.items():
    if folder in suspect:
        for token in tokens:
            matched_tokens.discard(token)

rows = []
for feature in fl.FEATURES_141:
    status, impl_ref, phase, notes = fl.STATUS_MAP[feature]
    tokens = fl.hints_for(feature)
    hit_tokens = [t for t in tokens if t in matched_tokens]
    rows.append(
        {
            "feature_index": fl.FEATURES_141.index(feature) + 1,
            "feature": feature,
            "group": fl.GROUP_OF[feature],
            "status": status,
            "phase": phase,
            "dataset": fl.DATASET[feature],
            "needs_external_input": bool(tokens),
            "local_input_available": bool(hit_tokens),
            "matched_tokens": ",".join(hit_tokens),
            "needs_gee": bool(tokens) and not hit_tokens,
            "no_input_required": feature in fl.NO_GEE_REQUIRED,
        }
    )

coverage = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 6. REPORT
# ---------------------------------------------------------------------------

open_mask = coverage["status"] != fl.S_COMPUTED

print("=" * 70)
print("COVERAGE OF THE 129 OPEN FEATURES")
print("=" * 70)

open_total = int(open_mask.sum())
local_ready = int((open_mask & coverage["local_input_available"]).sum())
gee_needed = int((open_mask & coverage["needs_gee"]).sum())
no_input = int((open_mask & coverage["no_input_required"]).sum())

print(f"open features                     : {open_total}")
print(f"  input already on disk (no GEE)  : {local_ready}")
print(f"  no input required at all        : {no_input}")
print(f"  must call Earth Engine          : {gee_needed}")

if local_ready:
    print()
    print("-" * 70)
    print("IMPLEMENTABLE LOCALLY -- do these FIRST (no GEE, no API cost)")
    print("-" * 70)
    print(f"{'#':>4}  {'group':<22} {'feature':<34} tokens")
    sub = coverage[open_mask & coverage["local_input_available"]]
    for _, r in sub.iterrows():
        print(f"{r.feature_index:>4}  {r.group:<22} {r.feature:<34} "
              f"{r.matched_tokens}")
else:
    print()
    print("No local match yet. Check the token list and the real folder")
    print("names printed above, then adjust fl.LOCAL_INPUT_HINTS if the")
    print("data is there under unexpected names.")

if gee_needed:
    print()
    print("-" * 70)
    print("STILL REQUIRE EARTH ENGINE")
    print("-" * 70)
    sub = coverage[open_mask & coverage["needs_gee"]]
    by_group = sub.groupby("group").size().sort_values(ascending=False)
    for group, count in by_group.items():
        print(f"  {group:<24} {count:>4}")

print()
print("-" * 70)
print("MATCHED TOKENS (recognised in your folder names, after size QC)")
print("-" * 70)
print("  " + (", ".join(sorted(matched_tokens)) if matched_tokens else "(none)"))
print()
print("trusted folders :", sorted(TRUSTED))
print("suspect folders :", sorted(suspect))
print()
print("NOTE: 0.00 GB of 506 .tif files means those rasters are almost")
print("certainly empty or all-NaN. Verify before building any plan on them.")

# ---------------------------------------------------------------------------
# 7. WRITE ARTIFACTS
# ---------------------------------------------------------------------------

QC_DIR = (_PROJECT / "qc") if _PROJECT is not None else Path("/content/qc")
QC_DIR.mkdir(parents=True, exist_ok=True)

inventory = {
    "external_root": str(EXTERNAL_ROOT) if EXTERNAL_ROOT else None,
    "searched_paths": [str(p) for p in searched],
    "n_subdirectories": len(subdirs),
    "n_files": len(files),
    "total_bytes": total_bytes,
    "total_gb": round(total_bytes / 1024**3, 4),
    "by_folder": {
        k: {"files": v["count"], "bytes": v["bytes"]}
        for k, v in sorted(per_dir.items())
    },
    "by_extension": {
        k: {"files": ext_counts[k], "bytes": ext_bytes[k]}
        for k in sorted(ext_counts)
    },
    "samples": {k: sorted(v)[:20] for k, v in sorted(by_folder.items())},
    "matched_tokens": sorted(matched_tokens),
    "trusted_folders": sorted(TRUSTED),
    "suspect_folders": sorted(suspect),
    "size_sanity_floor_bytes": fl.SIZE_SANITY_FLOOR_BYTES,
    "coverage_summary": {
        "open_features": open_total,
        "local_input_available": local_ready,
        "no_input_required": no_input,
        "needs_gee": gee_needed,
    },
}

inv_path = QC_DIR / "external_data_inventory.json"
inv_path.write_text(json.dumps(inventory, indent=2))

cov_path = QC_DIR / "external_coverage_matrix.csv"
coverage.to_csv(cov_path, index=False)

print()
print("Wrote:")
print("  ", inv_path)
print("  ", cov_path)
print()
print("=" * 70)
print("WHAT TO DO NEXT")
print("=" * 70)
print(" 1. Paste the printed output back to me so the ledger can be")
print("    updated with real paths, sizes and per-feature availability.")
print(" 2. If local_ready > 0, those features are reclassified from")
print("    MISSING_IMPLEMENTATION to LOCAL_INPUT_AVAILABLE and move to")
print("    the front of the work order.")
print(" 3. Then run P0 (colab_p0_cell262.py) to settle extract_fire_features.")
print(" 4. Do NOT start any Earth Engine loop yet.")