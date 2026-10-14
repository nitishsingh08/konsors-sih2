# ============================================================================
# COLAB CELL: survey the real Drive layout and pin the artefact paths
#
# Run this BEFORE colab_p3a_industrial.py or colab_p4a_lineage.py.
#
# Why it exists
# -------------
# The handoff documents paths that do not match the Drive:
#
#   handoff says                          drive actually has
#   ---------------------------           ---------------------------------
#   Firm_Clusters/full_2012_2026/clusters  Firm_Clusters/clusters        AND
#                                         Firm_Clusters/full_2012_2026/clusters
#   MyDrive/India_Firms/India_Firms.parquet Firm_Clusters/India_Firms.parquet
#
# `clusters/` and `edges/` exist in TWO places. Picking one silently would read
# whichever copy is stale or empty and produce plausible-looking nonsense, so
# this cell prints file counts, sizes and samples for both and lets the numbers
# decide.
#
# It writes nothing.
# ============================================================================

import json
from pathlib import Path

import colab_utils as cu

MAX_DEPTH = 3
SAMPLE_FILES = 3

print(cu.drive_report())
print()

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError(
        f"No directory under {cu.MYDRIVE} contains {cu.CANONICAL_SUBPATH}."
    )
print()


def survey(path, depth=0, max_depth=MAX_DEPTH):
    """Yield (relative_path, dir, n_files, bytes, samples) breadth-first."""
    root = Path(path)
    queue = [(root, depth)]
    while queue:
        current, d = queue.pop(0)
        files, dirs, total = [], 0, 0
        try:
            entries = sorted(current.iterdir())
        except OSError as exc:
            print(f"  !! cannot list {current}: {exc}")
            continue
        for entry in entries:
            try:
                if entry.is_dir():
                    dirs += 1
                else:
                    files.append(entry)
                    total += entry.stat().st_size
            except OSError:
                continue
        yield current, d, files, dirs, total
        if d < max_depth:
            for entry in entries:
                try:
                    if entry.is_dir():
                        queue.append((entry, d + 1))
                except OSError:
                    continue


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} TB"


# ---------------------------------------------------------------------------
# 1. Tree survey
# ---------------------------------------------------------------------------

print("=" * 96)
print(f"TREE SURVEY -- {PROJECT_ROOT}  (depth <= {MAX_DEPTH})")
print("=" * 96)
print(f"{'path':62s} {'files':>7s} {'dirs':>6s} {'size':>11s}")

inventory = {}
for current, depth, files, dirs, total in survey(PROJECT_ROOT):
    rel = current.relative_to(PROJECT_ROOT)
    label = str(rel) if str(rel) != "." else "."
    inventory[str(rel)] = {
        "n_files": len(files),
        "n_dirs": dirs,
        "bytes": total,
        "samples": [f.name for f in files[:SAMPLE_FILES]],
    }
    print(f"{label:62s} {len(files):>7d} {dirs:>6d} {human(total):>11s}")

# ---------------------------------------------------------------------------
# 2. Ambiguity report -- the part that actually matters
# ---------------------------------------------------------------------------

print()
print("=" * 96)
print("AMBIGUOUS FOLDER NAMES (same name at more than one depth)")
print("=" * 96)

by_name = {}
for rel, info in inventory.items():
    if rel == ".":
        continue
    leaf = Path(rel).name
    by_name.setdefault(leaf, []).append((rel, info))

ambiguous = {k: v for k, v in by_name.items() if len(v) > 1}
if not ambiguous:
    print("  none")
for leaf, entries in sorted(ambiguous.items()):
    print()
    print(f"  {leaf}/  appears {len(entries)} times:")
    for rel, info in sorted(entries, key=lambda t: t[1]["n_files"], reverse=True):
        verdict = ""
        if info["n_files"] == 0 and info["bytes"] == 0:
            verdict = "   <-- EMPTY, ignore this copy"
        print(
            f"    {rel:56s} files={info['n_files']:>6d} "
            f"size={human(info['bytes']):>10s}{verdict}"
        )
        if info["samples"]:
            print(f"      samples: {info['samples']}")

# ---------------------------------------------------------------------------
# 3. The specific artefacts the cells need
# ---------------------------------------------------------------------------

print()
print("=" * 96)
print("ARTEFACTS THE CELLS NEED")
print("=" * 96)

wanted = ("clusters", "edges", "events", "external")
resolved = cu.resolve_artefacts(PROJECT_ROOT, wanted=wanted, verbose=False)

for leaf in wanted:
    matches = cu.find_all_artefact(PROJECT_ROOT, leaf)
    print()
    print(f"  {leaf}/  ({cu.ARTEFACTS.get(leaf, '')})")
    if not matches:
        print("    NOT FOUND anywhere under the project root")
        continue
    for path, depth in matches:
        try:
            files = list(path.iterdir())
            n = len(files)
            size = sum(f.stat().st_size for f in files if f.is_file())
        except OSError:
            n, size = -1, 0
        print(f"    depth={depth}  {path}")
        print(f"      entries={n}  size={human(size)}")
        try:
            names = sorted(f.name for f in files)[:4]
            print(f"      sample names: {names}")
        except OSError:
            pass

# Specific files the cells open by name.
print()
print("  named files:")
for rel in (
    "full_2012_2026/events/canonical_event_base.parquet",
    "full_2012_2026/events/event_local_cluster_membership.parquet",
    "India_Firms.parquet",
    "full_2012_2026/external/osm_facilities",
):
    target = PROJECT_ROOT / rel
    state = "present" if cu.path_ok(target) else "ABSENT at this path"
    extra = ""
    if not cu.path_ok(target):
        # Search by leaf name, since the layout may differ.
        leaf = Path(rel).name
        hits = cu.find_all_artefact(PROJECT_ROOT, leaf)
        if hits:
            extra = "  found instead at: " + ", ".join(str(h[0]) for h in hits[:3])
    print(f"    {rel:62s} {state}{extra}")

# ---------------------------------------------------------------------------
# 4. Verdict
# ---------------------------------------------------------------------------

print()
print("=" * 96)
print("VERDICT")
print("=" * 96)

real_clusters = None
matches = cu.find_all_artefact(PROJECT_ROOT, "clusters")
if matches:
    scored = []
    for path, depth in matches:
        try:
            files = [f for f in path.iterdir() if f.is_file()]
        except OSError:
            files = []
        scored.append((len(files), depth, path))
    scored.sort(reverse=True)
    best = scored[0]
    real_clusters = best[2]
    print(f"  clusters/ : {len(scored)} copies found")
    for count, depth, path in scored:
        print(f"    {count:>6d} files  depth={depth}  {path}")
    print()
    print(f"  CANDIDATE for P4a: {real_clusters}")
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        print("  !! The top two copies have the SAME file count. Do not guess --")
        print("     open both and compare names/dates before committing.")
else:
    print("  clusters/ NOT FOUND")

print()
print("NEXT")
print("  Paste this output back. The P3a/P4a cells will be pinned to the paths")
print("  confirmed here, so they stop guessing and stop failing on FileNotFound.")
print("=" * 96)
