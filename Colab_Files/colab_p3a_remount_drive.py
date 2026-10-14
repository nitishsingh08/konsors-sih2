# ============================================================================
# P3a -- COLAB CELL: get back to the Drive that actually holds the data
#
# Situation this solves
# ---------------------
#   drive.mount('/content/drive', force_remount=True)
#   -> ValueError: Mountpoint must not already contain files
#
# That is NOT a corruption. Colab refuses to remount a non-empty mountpoint,
# and /content/drive is non-empty because the bootstrap cell wrote
# Firm_Clusters/{qc,pipeline} into it a few minutes ago.
#
# ⚠️ DO NOT "FIX" IT BY DELETING /content/drive/*
#
# Those two directories may be the stub in the wrong account, or they may be
# the only copy of canonical_event_base.parquet (7,265,849 rows) and the 5,337
# daily cluster files. Nothing is deleted automatically here, and nothing
# should be deleted manually until you know which account you are on.
#
# What this cell does
# -------------------
#   1. Mounts to a FRESH probe directory, so nothing existing is touched.
#   2. Reports which account that probe is attached to and whether the
#      project's data is visible there.
#   3. Only if the probe shows a healthy tree does it offer the consolidation
#      step, and even then it prints the command for you to run deliberately.
# ============================================================================

import os
import time
from pathlib import Path

from google.colab import drive

import colab_utils as cu

CURRENT = Path("/content/drive")
# A fresh directory per run. Never rmtree a mountpoint: it is a live FUSE
# mount and the kernel cancels the operation (OSError errno 125), which is what
# broke the previous run.
PROBE = Path(f"/content/drive_probe_{int(time.time())}")
EXPECTED = "Firm_Clusters/full_2012_2026/external"
EVENTS = "Firm_Clusters/full_2012_2026/events/canonical_event_base.parquet"

print(cu.drive_report())
print("=" * 74)
print("P3a -- DRIVE REMOUNT RECOVERY")
print("=" * 74)

# ---------------------------------------------------------------------------
# 1. What is already sitting in the current mountpoint?
# ---------------------------------------------------------------------------

print()
print("1. CURRENT MOUNTPOINT /content/drive")
print("-" * 74)

if not cu.path_ok(CURRENT):
    print("  does not exist -- nothing to worry about, mount directly")
    current_entries = set()
else:
    current_entries = {p.name for p in CURRENT.iterdir()}
    print(f"  /content/drive/MyDrive exists : {cu.path_ok(CURRENT / 'MyDrive')}")
    if cu.path_ok(CURRENT / "MyDrive"):
        for p in sorted((CURRENT / "MyDrive").iterdir()):
            print(f"    [{'dir ' if p.is_dir() else 'file'}] {p.name}")
    print()
    print(f"  top level of /content/drive: {sorted(current_entries)}")
    stub_only = current_entries <= {"MyDrive", ".shortcut-targets-"}
    print()
    if stub_only:
        print("  This contains only what this session wrote. It is a stub in the")
        print("  wrong account and safe to discard -- but only AFTER the probe")
        print("  below confirms the real data is reachable elsewhere.")

# ---------------------------------------------------------------------------
# 2. Mount a fresh probe directory. Non-destructive by construction.
# ---------------------------------------------------------------------------

status = cu.drive_status()
if status["dead_mount"]:
    print()
    print("!! /content/drive is a DEAD MOUNT. Unmount it before mounting")
    print("   anything, or the new mount will fail the same way:")
    print()
    print("      drive.unmount('/content/drive')")
    print("      drive.mount('/content/drive', force_remount=True)")
    print()
    print("   If drive.unmount() itself raises, restart the runtime first:")
    print("   Runtime > Restart session, then re-run this cell.")
    print("   Note the probe below works regardless, because it mounts a")
    print("   different mountpoint that is not affected.")

print()
print("2. MOUNTING A PROBE (does not touch /content/drive)")
print("-" * 74)

if cu.path_ok(PROBE) and any(PROBE.iterdir()):
    print(f"  probe path already in use: {PROBE}")
    print("  generating a different one")
    PROBE = Path(f"/content/drive_probe_{int(time.time())}_x")
PROBE.mkdir(parents=True, exist_ok=True)

drive.mount(str(PROBE), force_remount=True)
probe_mydrive = PROBE / "MyDrive"

print()
print(f"  probe mounted at {PROBE}")
print()
print("  IMPORTANT: Colab reuses the credential already attached to this")
print("  runtime. If the probe shows the same stub, the runtime is still")
print("  signed into the wrong account and step 3 applies.")

if not cu.path_ok(probe_mydrive):
    print()
    print("  probe has no MyDrive -- the mount did not attach as expected.")
    print("  Restart the runtime and re-run this cell.")
    # No SystemExit: IPython renders it as a traceback that reads like a
    # crash. Returning normally is enough; this cell has nothing after it.
    pass

probe_entries = sorted(p.name for p in probe_mydrive.iterdir())
print()
print("3. PROBE CONTENTS -- MyDrive top level")
print("-" * 74)
for name in probe_entries:
    print(f"  [dir ] {name}" if (probe_mydrive / name).is_dir() else f"  [file] {name}")

# ---------------------------------------------------------------------------
# 3. Verdict on the probe
# ---------------------------------------------------------------------------

has_external = cu.path_ok(probe_mydrive / EXPECTED)
has_events = cu.path_ok(probe_mydrive / EVENTS)

print()
print("=" * 74)
print("PROBE VERDICT")
print("=" * 74)
print(f"  {EXPECTED:52s} {'present' if has_external else 'ABSENT'}")
print(f"  {EVENTS:52s} {'present' if has_events else 'ABSENT'}")

if has_external and has_events:
    print()
    print("  GOOD -- the probe is attached to the account holding the project.")
    print()
    print("  Consolidate by clearing the stub and mounting the real thing:")
    print()
    print("      import shutil")
    print("      # only the stub: safe to remove once the probe confirmed it")
    print("      shutil.rmtree('/content/drive', ignore_errors=True)")
    print("      drive.mount('/content/drive', force_remount=True)")
    print()
    print("  If rmtree raises OSError errno 125 (Operation canceled), the")
    print("  mountpoint is busy. Unmount instead, which is cleaner anyway:")
    print()
    print("      drive.unmount('/content/drive')")
    print("      drive.mount('/content/drive', force_remount=True)")
    print()
    print("  /content/feature_ledger.py and /content/industrial_features.py")
    print("  are OUTSIDE that path, so your installed modules survive. The")
    print("  bootstrap cell's copy in Drive/pipeline does not -- that is fine,")
    print("  it is regenerated on the next bootstrap run.")
    print()
    print("  Then re-run colab_p3a_diagnose_external.py -> expect VERDICT: OK.")

elif not probe_entries or set(probe_entries) <= {"Firm_Clusters"}:
    print()
    print("  BAD -- the probe shows the same near-empty Drive.")
    print()
    print("  The current mount cannot see the data. Either the account is")
    print("  wrong, or the existing mount is stale.")
    print()
    print("  Stale mounts are the more common cause: an earlier session may")
    print("  have mounted Drive before the correct account was authorised, and")
    print("  the mountpoint keeps serving that old view. Try, in order:")
    print()
    print("    1. drive.unmount('/content/drive')")
    print("       drive.mount('/content/drive', force_remount=True)")
    print("       (unmount first -- remounting a non-empty mountpoint fails)")
    print("    2. If that still shows nothing, Runtime > Restart session, then")
    print("       run this cell again.")
    print("    3. Only then suspect the account: folder icon > Sign out, sign")
    print("       into the account owning Firm_Clusters/full_2012_2026, and")
    print("       restart the runtime.")
    print()
    print("  Nothing is lost: the data is in the other mount or account,")
    print("  untouched.")
else:
    print()
    print("  PARTIAL -- some data but not all. The account may be right and the")
    print("  tree incomplete, or shared drives are not attached.")
    print("  Check the Colab folder icon for 'Shared with me' / shared drives,")
    print("  then re-run colab_p3a_diagnose_external.py for a deeper search.")

print("=" * 74)