# ============================================================================
# P3a -- COLAB CELL: diagnose a missing external/ tree
#
# Run this when colab_p3a_industrial.py raises
#   FileNotFoundError: .../external/osm_facilities not found
#
# It distinguishes the cases that look identical from the error message but
# need completely different fixes:
#
#   A) Drive not mounted          -> remount, nothing is lost
#   B) Drive mounted, empty       -> wrong Google account
#   C) Data moved/renamed         -> fix the path constant
#   D) Data genuinely deleted     -> re-download, and it is ~40 GB
#
# Nothing is written. Read the verdict at the bottom.
# ============================================================================

from pathlib import Path

import colab_utils as cu

EXPECTED_EXTERNAL = Path(
    "/content/drive/MyDrive/Firm_Clusters/full_2012_2026/external"
)
MYDRIVE = Path("/content/drive/MyDrive")
DRIVE_ROOT = Path("/content/drive")

print(cu.drive_report())
print("=" * 74)
print("P3a -- EXTERNAL TREE DIAGNOSTIC")
print("=" * 74)

# ---------------------------------------------------------------------------
# 1. Is Drive mounted at all?
# ---------------------------------------------------------------------------

print()
print("1. MOUNT STATE")
print("-" * 74)
print(f"  /content/drive exists        : {cu.path_ok(DRIVE_ROOT)}")
print(f"  /content/drive/MyDrive exists : {cu.path_ok(MYDRIVE)}")

if not cu.path_ok(DRIVE_ROOT):
    print()
    print("  VERDICT: A) Drive is NOT mounted.")
    print("  Fix: Runtime > Change runtime type, or simply re-authenticate:")
    print("       from google.colab import drive")
    print("       drive.mount('/content/drive', force_remount=True)")
    print("  Nothing is lost -- Colab restarts detach the mount, not the data.")
    # No SystemExit: IPython renders it as a traceback that reads like a
    # crash. Returning normally is enough; this cell has nothing after it.
    pass

if not cu.path_ok(MYDRIVE):
    print()
    print("  VERDICT: A') /content/drive exists but MyDrive does not.")
    print("  Fix: from google.colab import drive")
    print("       drive.mount('/content/drive', force_remount=True)")
    # No SystemExit: IPython renders it as a traceback that reads like a
    # crash. Returning normally is enough; this cell has nothing after it.
    pass

# ---------------------------------------------------------------------------
# 2. What is actually in MyDrive?
# ---------------------------------------------------------------------------

print()
print("2. TOP LEVEL OF MyDrive")
print("-" * 74)

entries = [] if not cu.path_ok(MYDRIVE) else sorted(p for p in MYDRIVE.iterdir())
if not entries:
    print("  (empty)")
    print()
    print("  VERDICT: B) Drive is mounted but MyDrive is EMPTY.")
    print("  You are almost certainly signed into a different Google account")
    print("  than the one holding the 1.50 GB external/ tree. Check which")
    print("  account Colab is using before assuming anything was deleted.")
    # No SystemExit: IPython renders it as a traceback that reads like a
    # crash. Returning normally is enough; this cell has nothing after it.
    pass

for p in entries:
    kind = "dir " if p.is_dir() else "file"
    print(f"  [{kind}] {p.name}")

# A Drive holding nothing but what THIS session just wrote is, for our
# purposes, empty. Treat it as case B: the account is wrong, and every byte of
# the real tree is safe in the other account.
SESSION_WRITTEN = {"Firm_Clusters", "Colab Notebooks"}
contents = {p.name for p in entries}
only_session_written = contents <= SESSION_WRITTEN

firm = MYDRIVE / "Firm_Clusters"
print()
print(f"3. Firm_Clusters present: {cu.path_ok(firm)}")
if cu.path_ok(firm):
    for p in sorted(firm.iterdir()):
        kind = "dir " if p.is_dir() else "file"
        size = ""
        if p.is_file():
            size = f"  {p.stat().st_size / 1024:.0f} KB"
        print(f"  [{kind}] {p.name}{size}")

print()
print("3b. IS THIS DRIVE ACTUALLY YOURS?")
print("-" * 74)
if only_session_written:
    print(f"  MyDrive contains only: {sorted(contents)}")
    print("  Every one of those was written by THIS session a few minutes ago")
    print("  (the bootstrap cell writes Firm_Clusters/qc and")
    print("  Firm_Clusters/pipeline). A Drive with 1.50 GB of external data")
    print("  and 5,337 cluster files would show far more than this.")
    print()
    print("  VERDICT: B) WRONG GOOGLE ACCOUNT (not deletion).")
    print("  Colab is mounted to a different Drive than the one holding the")
    print("  project. Nothing has been lost -- the real tree is intact in the")
    print("  other account.")
else:
    print(f"  MyDrive contains {len(contents)} top-level entries, including")
    print(f"  {sorted(contents)[:10]}")
    print("  This looks like a real Drive, so account mix-up is unlikely.")

# ---------------------------------------------------------------------------
# 3. Hunt for the external tree wherever it moved to
# ---------------------------------------------------------------------------

print()
print("4. SEARCH FOR THE TREE")
print("-" * 74)

if cu.path_ok(EXPECTED_EXTERNAL):
    print(f"  FOUND at the expected path: {EXPECTED_EXTERNAL}")
    verdict = "ok"
else:
    print(f"  absent: {EXPECTED_EXTERNAL}")
    verdict = None

# Search by name, bounded depth, so a large Drive does not crawl forever.
candidates = []
if cu.path_ok(MYDRIVE):
    for depth in (1, 2, 3):
        pattern = "/".join(["*"] * depth)
        for hit in MYDRIVE.glob(f"{pattern}/external"):
            candidates.append(hit)
        for hit in MYDRIVE.glob(pattern):
            if hit.name in ("full_2012_2026", "external"):
                candidates.append(hit)

# De-duplicate while preserving order.
seen = set()
unique = []
for c in candidates:
    if c not in seen:
        seen.add(c)
        unique.append(c)

if unique:
    print()
    print("  possible locations:")
    for c in unique[:40]:
        print(f"    {c}")
    if verdict is None:
        verdict = "moved"
        print()
        print("  VERDICT: C) The data exists but NOT at the hard-coded path.")
        print("  Fix: set FACILITY_DIR / EXTERNAL_ROOT in "
              "colab_p3a_industrial.py to one of the paths above.")
else:
    print()
    print("  no directory named 'external' or 'full_2012_2026' within depth 3")

# ---------------------------------------------------------------------------
# 4. Look for the specific artefacts P3a needs
# ---------------------------------------------------------------------------

print()
print("5. P3a REQUIRED ARTEFACTS")
print("-" * 74)

need = {
    "osm_facilities (37 geoparquet, 0.28 GB)": "osm_facilities",
    "canonical_event_base.parquet (7,265,849 rows)": "canonical_event_base.parquet",
}

for label, needle in need.items():
    if verdict == "ok":
        base = EXPECTED_EXTERNAL
        if needle == "canonical_event_base.parquet":
            base = EXPECTED_EXTERNAL.parent / "events"
    else:
        base = None

    if base is not None:
        target = base / needle
        state = "present" if cu.path_ok(target) else "MISSING"
        print(f"  {label:52s} {state}  ({target})")
    else:
        print(f"  {label:52s} unknown (tree not located)")

# A moved or deleted folder is very often still recoverable from Trash.
trash = MYDRIVE / ".Trash"
print()
print("6. TRASH")
print("-" * 74)
if not cu.path_ok(trash):
    print("  no .Trash directory on this Drive")
elif only_session_written:
    print("  .Trash exists but this Drive is the wrong account, so its Trash")
    print("  is irrelevant. Check the Trash of the account that holds the data.")
else:
    hits = []
    try:
        for p in trash.rglob("*"):
            if p.name in ("external", "full_2012_2026"):
                hits.append(p)
    except PermissionError:
        print("  .Trash is not readable without the full Drive permission")
    if hits:
        for h in hits[:20]:
            print(f"  FOUND IN TRASH: {h}")
        print("  Restore it from Drive > Trash, then re-run this cell.")
    else:
        print("  no external/ or full_2012_2026 in Trash")

# ---------------------------------------------------------------------------
# 5. Verdict
# ---------------------------------------------------------------------------

print()
print("=" * 74)
if verdict == "ok":
    print("VERDICT: OK -- the tree is where the cell expects it.")
    print("Re-run colab_p3a_industrial.py as-is.")
elif verdict == "moved":
    print("VERDICT: C) Data present, path wrong.")
    print("Update EXTERNAL_ROOT / FACILITY_DIR / EVENT_BASE in")
    print("colab_p3a_industrial.py to the real location, then re-run.")
elif only_session_written:
    print("VERDICT: B) WRONG GOOGLE ACCOUNT -- nothing has been lost.")
    print()
    print("MyDrive holds only what this session wrote minutes ago. The 1.50 GB")
    print("external/ tree, the 5,337 daily cluster files and the 7,265,849-row")
    print("canonical_event_base.parquet are in a DIFFERENT Google account.")
    print()
    print("How to confirm and fix, in this exact order:")
    print()
    print("  1. Open a PRIVATE/incognito browser window and go to")
    print("     drive.google.com. Note which account shows a")
    print("     Firm_Clusters/full_2012_2026 folder. That is the account")
    print("     that matters. Do this BEFORE changing anything in Colab.")
    print()
    print("  2. In Colab, click the folder icon in the toolbar and sign out of")
    print("     the current Google account, then sign in as the account from")
    print("     step 1.")
    print()
    print("  3. Runtime > Restart session. This is required: the Drive")
    print("     credential is cached per runtime, so a restart is what")
    print("     actually swaps the account.")
    print()
    print("  4. Run colab_p3a_remount_drive.py. It probes a fresh mountpoint,")
    print("     which avoids the 'Mountpoint must not already contain")
    print("     files' error that plain force_remount=True hits here.")
    print()
    print("  NOTE: do not run drive.mount('/content/drive', force_remount=True)")
    print("  right now -- /content/drive holds the stub and is non-empty, so")
    print("  that command raises ValueError. And do not delete")
    print("  /content/drive/* until you know which account you are on: those")
    print("  paths may be the only copy of the event base.")
    print()
    print("IF YOU CANNOT GET INTO THE OTHER ACCOUNT, STOP AND SAY SO")
    print("before touching anything. The recovery is not a re-run of the")
    print("pipeline -- it is recovering the Drive itself.")
    print()
    print("DO NOT re-run build_full_global_events(). The clustering is intact")
    print("in the other account; re-running it would be slow, expensive and")
    print("would break the frozen regression anchors in HANDOFF section 2.")
    print()
    print("Also: delete the stub Firm_Clusters/ this session just created in")
    print("the wrong account, so it cannot be mistaken for the real one later.")
else:
    print("VERDICT: D) The external/ tree is NOT in this Drive at all.")
    print()
    print("P1b DID find this tree at the expected path earlier, so if the")
    print("account is confirmed correct then the files were moved or removed.")
    print("Check Trash (section 6) and shared drives before re-downloading.")
    print()
    print("If it is genuinely gone, the recovery cost is the ~40 GB in README")
    print("section 3:")
    print("    VIIRS_FOOTPRINTS 1.08 GB   osm_facilities 0.28 GB")
    print("    india_mopitt_gfas 0.02 GB  era5_land + sentinel5p (suspect)")
print("=" * 74)