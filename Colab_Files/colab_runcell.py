# ============================================================================
# colab_runcell.py -- run a cell from Drive, so a pasted notebook cell cannot go stale
#
# THE PROBLEM, WHICH HAS NOW COST FIVE RUNS
#
# There are two different kinds of code in this project and only one of them
# updates automatically:
#
#   MODULES  colab_utils.py, gee_utils.py, gold_set.py, ...
#            Installed by colab_bootstrap.py from an upload, into
#            Drive/pipeline. Editing the repo file and re-uploading changes
#            what runs.
#
#   CELLS    colab_p7e_gee_probe.py, colab_p6_dataset.py, ...
#            Pasted INTO a notebook cell as text. Editing the repo file
#            changes NOTHING about the text already sitting in the notebook.
#
# The last state is the dangerous one, because the symptom is a module report-
# ing a new version next to a cell report-ing an old one:
#
#     P7e VERIFY GEE  [cell version 2026-10-04.3]
#       gee_utils: 1.2.0                     <- module is current
#     ...
#     WorldCover sampleRegions FAILED: Asset 'ESA/WorldCover/v200' is not an Image
#
# Section 2 used the new `verify_datasets` and worked. Section 3 was the OLD
# cell text, still calling `ee.Image(...)` on a collection. Same notebook, same
# session, two different generations of the same file.
#
# Prior incidents: README section 5 defects 3 and 5, the partially-uploaded
# picker in section 6, the byte-identical P7e output, and this. Every one had
# the same shape, and every one cost a full Colab run to diagnose.
#
# THE FIX
#
# Stop pasting cell bodies. Paste this instead:
#
#     import colab_runcell
#     colab_runcell.run("colab_p7e_gee_probe")
#
# `run` fetches the file from Drive/pipeline -- the same place the bootstrap
# installed it -- and executes it in the CALLER's namespace. So the notebook
# cell becomes a two-line stub that cannot go stale, and re-uploading the cell
# file is all it takes to change what runs.
#
# The cell's own version marker is printed BEFORE it runs, so a mismatch is
# visible even if the cell then fails.
# ============================================================================

from __future__ import annotations

import re
import sys
import traceback
from pathlib import Path
from typing import List, Optional

#: Where the bootstrap installs uploaded modules and cells.
SEARCH_ROOTS = (
    Path("/content/drive/MyDrive/Firm_Clusters/pipeline"),
    Path("/content"),
)

#: Matches the version banner a cell prints, if it has one.
VERSION_RE = re.compile(r"([A-Z][A-Z0-9_]*_VERSION|__version__)\s*=\s*"
                        r"[\"']([^\"']+)[\"']")


class CellNotFound(RuntimeError):
    """The cell file was not uploaded, so there is nothing to run."""


def _candidate_paths(cell_name: str) -> List[Path]:
    name = cell_name if cell_name.endswith(".py") else f"{cell_name}.py"
    out: List[Path] = []
    for root in SEARCH_ROOTS:
        out.append(root / name)
        # Colab Notebooks / numbered duplicates, e.g. "foo (1).py".
        try:
            if root.is_dir():
                out.extend(sorted(p for p in root.glob(f"{Path(name).stem}*.py")
                                  if p not in out))
        except OSError:
            continue
    return out


def locate(cell_name: str) -> Path:
    """Find the cell file, or raise with instructions.

    Reports every path tried. "File not found" without the list is the failure
    mode that sends people re-pasting the body instead of re-uploading.
    """
    tried = _candidate_paths(cell_name)
    for p in tried:
        try:
            if p.is_file():
                return p
        except OSError:
            continue
    raise CellNotFound(
        f"could not find {cell_name} in any of:\n"
        + "\n".join(f"  {p}" for p in tried)
        + "\n\nFix: add the cell .py to the upload in colab_bootstrap.py and "
          "re-run it. Do NOT paste the cell body into the notebook -- that is "
          "how the notebook went stale in the first place."
    )


def declared_version(source: str) -> Optional[str]:
    """The version a cell declares, so staleness is visible before it runs."""
    for _name, value in VERSION_RE.findall(source):
        return value
    return None


def source_of(cell_name: str) -> tuple[str, Path]:
    path = locate(cell_name)
    try:
        src = path.read_text()
    except OSError as exc:
        raise CellNotFound(f"could not read {path}: {exc}") from None
    return src, path


def run(cell_name: str, *, namespace: Optional[dict] = None) -> dict:
    """Execute a cell script in the CALLER's globals.

    `namespace` defaults to the caller's module globals, which is what makes
    `EARTHENGINE_PROJECT = '...'` set in a notebook cell visible to the cell
    it runs. Passing a private dict instead would silently hide it, which is a
    confusing way to lose a variable.

    Returns a small report dict. Never raises on a cell failure -- a traceback
    from the cell is the cell's own output and the caller may want to continue.
    """
    src, path = source_of(cell_name)
    version = declared_version(src)

    print("=" * 74)
    print(f"RUN {path.name}   [{version or 'no version marker'}]")
    print(f"  from {path}")
    print(f"  bytes {len(src):,}")
    print("=" * 74)

    glb = namespace if namespace is not None else sys._getframe(1).f_globals
    glb["__file__"] = str(path)
    glb["__name__"] = "__main__"

    report = {"cell": path.name, "path": str(path), "version": version,
              "ok": True, "error": None}
    try:
        code = compile(src, str(path), "exec")
        exec(code, glb)                          # noqa: S102 -- by design
    except BaseException as exc:                  # noqa: BLE001
        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        print()
        print("!" * 74)
        print(f"CELL RAISED {type(exc).__name__}: {exc}")
        print("!" * 74)
        traceback.print_exc()
    return report


def status(*cell_names: str) -> List[dict]:
    """What version is on Drive for each named cell. No execution."""
    out = []
    for name in cell_names:
        try:
            src, path = source_of(name)
            out.append({"cell": name, "found": True,
                        "path": str(path),
                        "version": declared_version(src),
                        "bytes": len(src)})
        except CellNotFound as exc:
            out.append({"cell": name, "found": False, "path": None,
                        "version": None, "bytes": 0, "error": str(exc)})
    return out