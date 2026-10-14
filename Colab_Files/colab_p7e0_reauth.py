# ============================================================================
# P7e0 -- COLAB CELL: switch Earth Engine / Drive to a different Google account
#
# WHY THIS CELL
#
# You changed Google account. Two INDEPENDENT things are now bound to that
# account and they fail separately:
#
#   1. The Earth Engine CREDENTIAL. Colab caches the token at
#      /root/.config/earthengine/credentials. `ee.Authenticate()` against an
#      existing cache does not re-prompt, so you keep getting the OLD account
#      while believing you re-authenticated. The cache must be deleted first.
#
#   2. The GCP PROJECT. A project belongs to the account that created it. If
#      you switch accounts, the previous project now belongs to someone else
#      and every call returns:
#
#        403 PERMISSION_DENIED -- Caller does not have required permission
#        to use project <ID>. Grant the caller the role
#        roles/serviceusage.serviceUsageConsumer
#
#      That 403 is NOT an authentication failure. The credentials are fine.
#      `ee.Initialize(project=...)` still SUCCEEDS, which is the trap: it
#      accepts credentials without checking that the caller may USE the
#      project. Only an actual data read reveals the problem.
#
#   3. THE DRIVE MOUNT, which is the one people forget. `drive.mount()` is
#      per-account. If you switched accounts and remount, you are now writing
#      into a DIFFERENT Drive. This cell reports which account Drive is
#      actually serving, because the rest of the project resolves its project
#      root from that Drive.
#
# NOTHING IS WRITTEN except a small diagnostic record under qc/.
# ============================================================================

import json
import os
import shutil
import subprocess
from pathlib import Path

import colab_utils as cu

PROJECT_ROOT = cu.resolve_project_dir()
QC_DIR = (PROJECT_ROOT / "qc") if PROJECT_ROOT is not None else Path("/content")
QC_DIR.mkdir(parents=True, exist_ok=True)

#: Where the Earth Engine token lives in a Colab runtime. Checked rather than
#: assumed, so a moved path shows up as "not found" instead of a silent
#: failure to actually clear anything.
EE_CRED_DIRS = [
    Path("/root/.config/earthengine"),
    Path.home() / ".config" / "earthengine",
    Path("/content/.config/earthengine"),
]

#: See colab_bootstrap.CELL_VERSION_FLOORS. Bump on every change so the
#: bootstrap can tell a current cell from an old one -- an unversioned
#: cell cannot be checked, and an uncheckable cell is what cost five runs.
P7E0_CELL_VERSION = "2026-10-04.1"

print("=" * 74)
print(f"P7e0 -- switch Google account for Earth Engine and Drive [v{P7E0_CELL_VERSION}]")
print("=" * 74)
print(cu.drive_report())

# ---------------------------------------------------------------------------
# 1. WHICH ACCOUNT IS DRIVE SERVING?
#
# Read BEFORE clearing anything, so you know the starting state.
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("1. WHICH GOOGLE ACCOUNT IS ACTIVE?")
print("=" * 74)

_email = None
try:
    from google.colab import auth as _colab_auth  # type: ignore

    _info = _colab_auth.google.auth._default()          # type: ignore[attr-defined]
    _creds = _info[1] if isinstance(_info, tuple) and len(_info) > 1 else None
    if _creds is not None:
        _email = getattr(_creds, "id_token", None)
        try:
            _email = _creds.id_token.get("email")       # type: ignore[union-attr]
        except Exception:                              # noqa: BLE001
            pass
except Exception as exc:                                # noqa: BLE001
    print(f"  (could not read the credential directly: "
          f"{type(exc).__name__})")

if _email:
    print(f"  Colab / Drive account : {_email}")
else:
    print("  Colab / Drive account : not directly readable in this runtime")
    print("  Check it by hand: the profile icon top-right of Colab, and")
    print("  Google Drive -> the account shown on the Drive menu.")

try:
    from google.colab import drive as _drive        # type: ignore

    _mounts = _drive.mounted.get("/content/drive")    # type: ignore[attr-defined]
    print(f"  Drive mount state    : {_mounts}")
except Exception:                                      # noqa: BLE001
    pass

print()
print("  IMPORTANT: Drive is per-account. If you switched accounts and")
print("  remounted, the whole project now resolves against a different")
print("  Drive. Confirm the project root above is the one holding")
print("  Firm_Clusters/ before writing anything.")

# ---------------------------------------------------------------------------
# 2. CLEAR THE EARTH ENGINE CACHE
#
# `ee.Authenticate()` alone will NOT re-prompt while a valid token is cached,
# so the old account persists. Deleting the cache is the actual fix.
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("2. EARTH ENGINE CREDENTIAL CACHE")
print("=" * 74)

_found = []
for _d in EE_CRED_DIRS:
    if _d.exists():
        _files = sorted(p.name for p in _d.iterdir())
        print(f"  PRESENT {_d}")
        print(f"          contains: {_files}")
        _found.append(_d)
    else:
        print(f"  absent  {_d}")

if not _found:
    print()
    print("  No cached credential directory found in any standard location.")
    print("  So the token is not cached here -- which usually means ee has")
    print("  never authenticated in this runtime, or it is stored elsewhere.")
    print("  Just run ee.Authenticate() in the cell below and accept the")
    print("  consent screen, choosing the account you want.")

_CLEAR = bool(globals().get("EE_RESET", True))
print()
if _CLEAR:
    for _d in _found:
        try:
            shutil.rmtree(_d)
            print(f"  removed {_d}")
        except Exception as exc:                       # noqa: BLE001
            print(f"  FAILED to remove {_d}: {type(exc).__name__}: {exc}")
            print("  Run this by hand if it persists:")
            print(f"      !rm -rf {_d}")
    if _found:
        print()
        print("  Cache cleared. The next `ee.Authenticate()` MUST prompt, and")
        print("  you must pick the NEW account on the consent screen. If the")
        print("  screen opens without asking, the token is cached elsewhere --")
        print("  restart the Colab runtime and repeat.")
else:
    print("  EE_RESET is False -- cache left in place.")

# ---------------------------------------------------------------------------
# 3. WHAT TO RUN NEXT
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("3. DO THIS NOW, IN ITS OWN CELL")
print("=" * 74)
print("""
    import ee
    ee.Authenticate()          # MUST prompt. Choose the NEW account.
    ee.Initialize(project='<PROJECT-BELONGING-TO-THE-NEW-ACOUNT>')

Then, in another cell, check which projects the new account can actually use:

    import ee
    ee.Initialize(project='<ID>')
    print(ee.data.getAssetRoots())
""")
print("  `getAssetRoots()` returns `projects/<ID>/assets` entries. Those are")
print("  the projects THIS account may use. If your intended project is not in")
print("  that list, it belongs to the other account.")
print()
print("=" * 74)
print("THE 403 YOU HIT, EXPLAINED")
print("=" * 74)
print("  'Caller does not have required permission to use project")
print("   festive-music-501617-n8'")
print()
print("  means the project is not usable by the account now active. It is not")
print("  a credential problem -- `Initialize` accepted the credentials fine.")
print("  Two fixes, pick one:")
print()
print("  (a) RECOMMENDED -- create a project under the NEW account:")
print("        https://console.cloud.google.com/")
print("        pick YOUR account in the project selector")
print("        Create project")
print("        APIs & Services -> Library -> 'Earth Engine API' -> Enable")
print("        https://code.earthengine.google.com/ -> pick it in the dropdown")
print("        (registering there also registers the project with Earth Engine)")
print("        then EARTHENGINE_PROJECT = '<new-id>'")
print()
print("  (b) Keep the old project and grant the new account the role:")
print("        https://console.cloud.google.com/iam-admin/iam?project=<old-id>")
print("        add the new account's email with role")
print("        roles/serviceusage.serviceUsageConsumer")
print("      Only do this if you still have access to the old project.")

_record = {
    "colab_account_readable": bool(_email),
    "credential_cache_found": [str(d) for d in _found],
    "credential_cache_cleared": _CLEAR and bool(_found),
}
(QC_DIR / "p7e0_account_state.json").write_text(
    json.dumps(_record, indent=2, default=str))
print()
print(f"Wrote {QC_DIR / 'p7e0_account_state.json'}")
print("=" * 74)
print("Then re-run colab_p7e_gee_probe.py. It now performs a real data read")
print("before reporting success, so a 403 surfaces as one clear message")
print("instead of four confusing per-dataset load failures.")
print("=" * 74)