#!/usr/bin/env python3
"""
Fetch the trained model artefacts from Google Drive into ./models.

Written by colab_p7_train.py to
  My Drive/Firm_Clusters/qc/models/model_{A,B}.pkl
and never committed: a .pkl is a pickle, so it executes code on load and a
3.6 MB binary per revision bloats history for something reproducible.

Usage
-----
    python3 scripts/fetch_models.py
    python3 scripts/fetch_models.py --dest /some/other/dir
    python3 scripts/fetch_models.py --verify-only

Options
-------
--dest DIR       where to write the artefacts (default: <repo>/models)
--verify-only    check what is present locally without downloading

Why verification matters
-----------------------
The artefacts are pickled inside Colab, so they only unpickle under a
compatible LightGBM version. `--verify-only` therefore LOADS each model and
checks its contract, rather than merely confirming the bytes exist: a 3.6 MB
file that raises on unpickle is not a usable model.
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "models"
DRIVE_RELATIVE = Path("Firm_Clusters/qc/models")
MODELS = ("model_B.pkl", "model_A.pkl")

#: What a loaded artefact must expose. A pickle that lacks these was written by
#: an older cell and cannot be served safely.
REQUIRED_KEYS = ("booster", "features", "target")


class ArtifactError(RuntimeError):
    pass


def verify(path: Path) -> dict:
    """Load one artefact and check its contract. Returns a summary."""
    try:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
    except Exception as exc:  # noqa: BLE001
        raise ArtifactError(
            f"{path.name}: could not unpickle ({exc.__class__.__name__}: {exc}).\n"
            "  The file is pickled inside Colab, so it needs a compatible "
            "LightGBM locally:\n"
            "    python3 -m pip install -U lightgbm\n"
            "  If that fails, re-export from Colab."
        ) from exc

    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise ArtifactError(
            f"{path.name}: missing {missing}. Written by an older "
            "colab_p7_train.py -- re-run it."
        )
    features = list(payload["features"])
    if not features:
        raise ArtifactError(f"{path.name}: feature list is empty")
    return {
        "file": path.name,
        "size_kb": round(path.stat().st_size / 1024),
        "target": payload["target"],
        "n_features": len(features),
        "best_iteration": payload.get("best_iteration"),
        "label_column": payload.get("label_column"),
        "split": payload.get("split"),
        "caveat": "CIRCULAR: trained on weak labels" if path.stem == "model_A"
                  else "scientific partition: no proximity or rule inputs",
    }


def find_drive_root() -> Path | None:
    """Locate a mounted Google Drive, if one exists on this machine."""
    candidates = [
        Path("~/Google Drive"),
        Path("~/Library/CloudStorage/GoogleDrive-Shared"),
        Path("~/Library/CloudStorage/GoogleDrive"),
        Path("~/Dropbox"),           # some setups symlink Drive here
    ]
    for base in candidates:
        expanded = base.expanduser()
        if expanded.exists() and (expanded / DRIVE_RELATIVE).exists():
            return expanded / DRIVE_RELATIVE
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help="where to write the artefacts")
    parser.add_argument("--verify-only", action="store_true",
                        help="check local artefacts without downloading")
    args = parser.parse_args()
    dest = args.dest.expanduser()

    print("=" * 70)
    print("MODEL ARTEFACTS")
    print("=" * 70)

    if args.verify_only:
        if not dest.exists():
            print(f"  no directory at {dest}")
            return 1
        failures = 0
        for name in MODELS:
            path = dest / name
            if not path.exists():
                print(f"  MISSING  {name}")
                failures += 1
                continue
            try:
                info = verify(path)
            except ArtifactError as exc:
                print(f"  INVALID  {exc}")
                failures += 1
                continue
            print(f"  OK       {name}  {info['size_kb']} KB  "
                  f"{info['n_features']} features  target={info['target']}")
            print(f"           {info['caveat']}")
        print()
        print("export MODEL_DIR=" + str(dest))
        return 1 if failures else 0

    source = find_drive_root()
    if source is None:
        print()
        print("  Could not find a mounted Google Drive holding:")
        print(f"    {DRIVE_RELATIVE}")
        print()
        print("  Options:")
        print("   1. Install Google Drive for Desktop, then re-run.")
        print("   2. Download manually from drive.google.com:")
        print("      My Drive > Firm_Clusters > qc > models")
        print("      Save both model_A.pkl and model_B.pkl here:")
        print(f"        {dest}")
        print("   3. Fetch from Colab instead (see README section 16).")
        print()
        return 1

    print(f"  source : {source}")
    print(f"  dest   : {dest}")
    print()
    dest.mkdir(parents=True, exist_ok=True)

    failures = 0
    for name in MODELS:
        src = source / name
        if not src.exists():
            print(f"  MISSING {src}")
            failures += 1
            continue
        target = dest / name
        shutil.copy2(src, target)
        try:
            info = verify(target)
        except ArtifactError as exc:
            print(f"  COPIED but INVALID  {exc}")
            failures += 1
            continue
        print(f"  OK  {name}  {info['size_kb']} KB  "
              f"{info['n_features']} features  target={info['target']}")
        print(f"      {info['caveat']}")

    print()
    if failures:
        print(f"  {failures} artefact(s) missing or invalid")
        return 1
    print("  all artefacts present and loadable")
    print()
    print("Next:")
    print(f"  export MODEL_DIR={dest}")
    print("  cd backend && uvicorn app.main:app --reload")
    return 0


if __name__ == "__main__":
    sys.exit(main())