"""
Small shared helpers for the Colab cells.

Kept in its own module so the errno-107 handling is written once instead of
being re-implemented -- slightly differently -- in every cell.

Why this exists
---------------
When Drive is unmounted without a clean remount, `/content/drive` is left as a
dead FUSE mountpoint. On that state:

    Path("/content/drive").exists()   -> OSError [Errno 107]
                                         Transport endpoint is not connected

It does NOT return False. So a plain `.exists()` guard does not protect a cell;
it crashes it, and the traceback points at the cell's own logic rather than at
the real problem. `path_ok()` swallows exactly that and reports the mount as
absent, which is what it functionally is.
"""

from __future__ import annotations

import errno
import os
import re
from pathlib import Path

#: errno values that mean "the mount is dead", not "the path is missing".
#:
#: Written as literal numbers rather than `errno.ENOTCONN` etc. because those
#: constants are platform-dependent: ENOTCONN is 107 on Linux (where Colab
#: runs) but 57 on macOS. A set built from the names silently omits the value
#: that matters on whichever host the code is running on -- which is exactly
#: the bug that made this module's own test fail on a Mac.
#:
#:   107 ENOTCONN  Linux   "Transport endpoint is not connected"
#:    57 ENOTCONN  macOS   same condition, different number
#:   116 ESTALE    Linux   "Stale file handle"
#:    70 ESTALE    macOS
#:     5 EIO       both    "Input/output error"
#:    19 ENODEV    both    "No such device"
DEAD_MOUNT_ERRNOS = {5, 19, 57, 70, 107, 116}


def path_ok(path) -> bool:
    """True only if `path` is reachable. Never raises, whatever the mount state.

    Use this in every `if not X.exists()` guard in the Colab cells.
    """
    try:
        return Path(path).exists()
    except OSError as exc:
        if exc.errno in DEAD_MOUNT_ERRNOS:
            return False
        raise


def is_dir(path) -> bool:
    try:
        return Path(path).is_dir()
    except OSError as exc:
        if exc.errno in DEAD_MOUNT_ERRNOS:
            return False
        raise


def _safe_listdir(path):
    """(entries, dead) for `path`. Never raises."""
    try:
        return sorted(p.name for p in Path(path).iterdir()), False
    except OSError as exc:
        if exc.errno in DEAD_MOUNT_ERRNOS:
            return [], True
        if exc.errno == errno.ENOENT:
            return [], False
        raise


def drive_status() -> dict:
    """Describe the Drive mount state. Safe to call at any time."""
    root = Path("/content/drive")
    mydrive = root / "MyDrive"
    status = {
        "drive_root": path_ok(root),
        "mydrive": path_ok(mydrive),
        "dead_mount": False,
        "entries": [],
    }
    # Listing is the only reliable probe: a dead mountpoint can still stat
    # successfully on its root while every child fails.
    if status["mydrive"]:
        entries, dead = _safe_listdir(mydrive)
        status["entries"] = entries
        status["dead_mount"] = dead
    elif status["drive_root"]:
        entries, dead = _safe_listdir(root)
        status["entries"] = entries
        status["dead_mount"] = dead
    return status


def drive_report() -> str:
    """Human-readable Drive state, safe to print at the top of any cell."""
    s = drive_status()
    lines = ["=" * 70, "DRIVE STATE", "=" * 70]
    lines.append(f"  /content/drive reachable : {s['drive_root']}")
    lines.append(f"  MyDrive reachable       : {s['mydrive']}")
    if s["dead_mount"]:
        lines.append("")
        lines.append("  !! DEAD MOUNT (errno 107). The unmount left the FUSE")
        lines.append("     mountpoint attached with no server behind it.")
        lines.append("     Fix, in order:")
        lines.append("       1. Runtime > Restart session   <- do this first")
        lines.append("       2. from google.colab import drive")
        lines.append("          drive.mount('/content/drive', force_remount=True)")
        lines.append("     Do not try to shutil.rmtree it: that raises errno 125.")
    lines.append(f"  MyDrive entries         : {s['entries'] or '(none)'}")
    lines.append("=" * 70)
    return "\n".join(lines)


def require_drive(what: str = "project data") -> None:
    """Abort with actionable advice if Drive is not usable."""
    s = drive_status()
    if s["dead_mount"]:
        raise RuntimeError(
            f"Drive is a dead mount (errno 107); cannot read {what}.\n"
            + drive_report()
            + "\n  Runtime > Restart session, then re-mount."
        )
    if not s["mydrive"]:
        raise RuntimeError(
            f"Drive is not mounted; cannot read {what}.\n"
            + drive_report()
            + "\n  from google.colab import drive\n"
            "  drive.mount('/content/drive', force_remount=True)"
        )


# ---------------------------------------------------------------------------
# PROJECT ROOT RESOLUTION
#
# Why this exists: the real tree turned up as `Firm_Clusters (1)`, a
# macOS-style duplicate of `Firm_Clusters`. Every cell used to hardcode
# /content/drive/MyDrive/Firm_Clusters/..., so the moment the folder got
# duplicated or renamed the cells failed with FileNotFoundError while the data
# sat right there under a suffixed name.
#
# Resolution is by CONTENT, not by name: find the directory that actually
# contains full_2012_2026, preferring the exact name and otherwise taking the
# shortest matching name. That way " (1)", " (2)" and renames all resolve.
# ---------------------------------------------------------------------------

MYDRIVE = Path("/content/drive/MyDrive")
PROJECT_DIRNAME = "Firm_Clusters"
CANONICAL_SUBPATH = "full_2012_2026"

#: Subpaths the feature cells need, in preference order.
REQUIRED_ARTEFACTS = (
    f"{CANONICAL_SUBPATH}/external",
    f"{CANONICAL_SUBPATH}/events",
)

# ---------------------------------------------------------------------------
# STALE-LISTING RECOVERY
#
# Colab mounts Drive over FUSE, which caches directory listings. Adding or
# deleting folders in the Drive *browser* while a Colab mount is live does not
# always invalidate that cache, so a folder that plainly exists in the browser
# can be invisible to the runtime. The symptom seen here was
# `MyDrive entries: ['Firm_Clusters']` with Firm_Clusters reporting 0 required
# subpaths, while the browser showed full_2012_2026 sitting right inside it.
#
# The fix is to force a fresh mount and then retry, rather than to trust the
# first listing or to start guessing at paths.
# ---------------------------------------------------------------------------

#: Seconds to wait between retries while a fresh listing propagates.
RETRY_DELAYS = (2, 5, 10, 20)


def force_refresh_drive():
    """Remount Drive to drop the FUSE directory cache.

    Returns True if a refresh was attempted (or is unnecessary because Drive is
    not mounted here, e.g. in local unit tests), False only if the remount
    genuinely failed. Never raises: the caller wants to fall through to a retry
    or a clear message, not an exception.
    """
    if not path_ok(Path("/content/drive")):
        # Not a Colab Drive mount at all (local test run).
        return True
    try:
        from google.colab import drive

        drive.mount("/content/drive", force_remount=True)
        return True
    except Exception as exc:                        # noqa: BLE001
        print(f"  force_refresh_drive failed: {exc.__class__.__name__}: {exc}")
        return False


def wait_for(predicate, delays=None, label="condition"):
    """Retry `predicate()` until it returns truthy or the delays run out.

    Returns whatever predicate returned last. Prints each attempt so a stall is
    visible rather than silent.

    `delays` is read at call time, not bound as a default argument, so
    RETRY_DELAYS stays overridable (the previous default-argument form captured
    the tuple at def time and silently ignored any override).
    """
    import time

    delays = RETRY_DELAYS if delays is None else delays
    last = None
    for i, delay in enumerate((0.0,) + tuple(delays)):
        if delay:
            print(f"  retry {i}/{len(delays)} in {delay:.0f}s ...")
            time.sleep(delay)
        last = predicate()
        if last:
            print(f"  {label} satisfied on attempt {i}")
            return last
    print(f"  {label} still not satisfied after {len(delays)} retries")
    return last


def _suffix_rank(name: str) -> tuple:
    """Sort key preferring the exact name, then the shortest duplicate suffix."""
    m = re.match(r"^(?P<base>.*?)\s*\((?P<n>\d+)\)$", name)
    if not m:
        return (0, 0, name)          # exact name wins outright
    return (1, int(m.group("n")), m.group("base"))


def candidate_project_dirs():
    """Every plausible project root, best first."""
    if not path_ok(MYDRIVE):
        return []
    try:
        names = sorted((p.name for p in MYDRIVE.iterdir()), key=_suffix_rank)
    except OSError:
        return []

    out = []
    for name in names:
        base = MYDRIVE / name
        if not is_dir(base):
            continue
        out.append(base)
    return out


def resolve_project_dir(
    required=None, verbose=True, allow_refresh=True, retry_delays=None
):
    """Locate the project root by looking for the canonical subpath.

    Returns the Path, or None when nothing contains it. `required` may be a
    single relative subpath or an iterable of them; all must be present for a
    candidate to win, which prevents picking a partial duplicate.

    When nothing matches and `allow_refresh` is set, the Drive mount is
    refreshed and the search retried, because a folder that exists in the
    browser can be invisible to a stale FUSE cache.
    """
    wanted = required or REQUIRED_ARTEFACTS
    if isinstance(wanted, str):
        wanted = (wanted,)

    def attempt():
        scored = []
        for base in candidate_project_dirs():
            present = [cu_path_ok(base / w) for w in wanted]
            hits = sum(present)
            scored.append((hits, len(wanted), base, all(present)))
        complete = [s for s in scored if s[3]]
        return (complete[0][2] if complete else None), scored

    found, scored = attempt()

    if found is None and allow_refresh:
        if verbose:
            print()
            print("NOTHING MATCHED. The most likely cause is a stale Drive FUSE")
            print("listing: folders added or deleted in the Drive browser while")
            print("this runtime holds a mount are not always visible here.")
            print()
            print("what the runtime can currently see under MyDrive:")
            entries, dead = _safe_listdir(MYDRIVE)
            if dead:
                print("  (dead mount)")
            for name in entries:
                print(f"  - {name}")
            print()
            print("refreshing the mount and retrying ...")
            print()
        if force_refresh_drive():
            found = wait_for(
                lambda: attempt()[0],
                delays=retry_delays,
                label="project root visible after remount",
            )
            scored = attempt()[1]

    if verbose and scored:
        print("PROJECT ROOT CANDIDATES")
        print("-" * 74)
        for hits, total, base, complete in scored:
            mark = "OK  " if complete else ("part" if hits else "no  ")
            print(f"  [{mark}] {base}  ({hits}/{total} required subpaths)")
        print("-" * 74)
        if found is None:
            print("  no candidate contains the required subpaths")
            print()
            print("  If the folders plainly exist in drive.google.com, the mount")
            print("  is stale. Runtime > Restart session, then re-run.")
        else:
            print(f"resolved project root: {found}")
            print("-" * 74)

    return found


def cu_path_ok(path) -> bool:
    """Indirection so resolve_project_dir can call path_ok defined above."""
    return path_ok(path)


# ---------------------------------------------------------------------------
# ARTEFACT RESOLUTION
#
# The real Drive layout does NOT nest the way the handoff assumed:
#
#   Firm_Clusters/
#       clusters/            <- top level, not under full_2012_2026/
#       edges/               <- top level
#       global/  logs/
#       full_2012_2026/
#           external/  events/  qc/
#       India_Firms.parquet  <- top level, not MyDrive/India_Firms/
#
# So a leaf name is resolved by SEARCHING for it, not by joining a guessed
# path. A wrong guess raises FileNotFoundError; a search either finds the
# directory or reports every place it looked.
# ---------------------------------------------------------------------------

#: Leaf names the feature cells need, with a human description.
ARTEFACTS = {
    "clusters": "daily cluster parquet files",
    "edges": "cross-day cluster edges",
    "external": "external data tree (gazetteers, footprints)",
    "events": "event base + membership tables",
}

MAX_SEARCH_DEPTH = 4


def find_all_artefact(project_root, leaf_name, max_depth=MAX_SEARCH_DEPTH):
    """Every directory named `leaf_name` under `project_root`, shallowest first.

    The Drive currently holds `clusters/` and `edges/` in TWO places --
    `Firm_Clusters/clusters/` and `Firm_Clusters/full_2012_2026/clusters/`.
    Picking one silently is the failure this resolver exists to prevent: either
    choice silently reads whichever copy is staler or emptier. So all matches
    are returned and the ambiguity is surfaced.
    """
    root = Path(project_root)
    if not path_ok(root):
        return []
    matches = []
    queue = [(root, 0)]
    seen = {root}
    while queue:
        current, depth = queue.pop(0)
        if current.name == leaf_name:
            matches.append((current, depth))
            continue          # do not descend into a match
        if depth >= max_depth:
            continue
        try:
            children = sorted(p for p in current.iterdir() if p.is_dir())
        except OSError:
            continue
        for child in children:
            if child in seen:
                continue
            seen.add(child)
            queue.append((child, depth + 1))
    matches.sort(key=lambda t: (t[1], str(t[0])))
    return matches


def find_artefact(project_root, leaf_name, max_depth=MAX_SEARCH_DEPTH):
    """Locate `leaf_name`, preferring the shallowest match. (Path, depth|None)."""
    matches = find_all_artefact(project_root, leaf_name, max_depth)
    if not matches:
        return None, None
    return matches[0][0], matches[0][1]


def resolve_artefacts(project_root, wanted=None, verbose=True):
    """Resolve every artefact the cells need, by name search.

    Returns dict leaf -> Path|None. Prints a table so the mapping is auditable:
    a silently wrong artefact path is the failure mode that produces plausible
    but incorrect features.
    """
    wanted = tuple(wanted or ARTEFACTS)
    found = {}
    ambiguous = {}
    if verbose:
        print("ARTEFACT RESOLUTION (by name search under the project root)")
        print("-" * 78)
        print(f"  root: {project_root}")
        print("-" * 78)

    for leaf in wanted:
        matches = find_all_artefact(project_root, leaf)
        if len(matches) > 1:
            ambiguous[leaf] = [m[0] for m in matches]
        path, depth = matches[0] if matches else (None, None)
        found[leaf] = path
        if verbose:
            if path is None:
                print(f"  [MISSING] {leaf:10s} {ARTEFACTS.get(leaf, '')}")
            elif len(matches) > 1:
                print(f"  [AMBIG  ] {leaf:10s} {len(matches)} matches:")
                for candidate, d in matches:
                    print(f"               depth={d}  {candidate}")
            else:
                rel = path.relative_to(Path(project_root))
                print(f"  [ok     ] {leaf:10s} depth={depth}  {rel}")

    if verbose:
        print("-" * 78)
        if ambiguous:
            print(f"  AMBIGUOUS: {sorted(ambiguous)}")
            print("  The same folder name exists at more than one depth. Run")
            print("  colab_survey_tree.py and compare file counts before")
            print("  trusting either copy.")
        missing = [k for k, v in found.items() if v is None]
        if missing:
            print(f"  MISSING: {missing}")
            print("  Do not proceed. Re-run colab_survey_tree.py and paste the")
            print("  output -- these names come from the handoff, not from the")
            print("  actual Drive layout, which differs.")
        if not ambiguous and not missing:
            print("  all required artefacts located, unambiguously")

    found["ambiguous"] = ambiguous
    return found


def pick_by_file_count(project_root, leaf_name, max_depth=MAX_SEARCH_DEPTH):
    """Choose between duplicate copies by how many files each actually holds.

    `clusters/` exists in two places on this Drive. Depth is the wrong tie-break
    because the nesting in the handoff was never right; the number of files is
    the evidence. Returns (path, table) where table lists every candidate so
    the choice stays auditable, or (None, table) when nothing matched.
    """
    matches = find_all_artefact(project_root, leaf_name, max_depth)
    table = []
    for path, depth in matches:
        try:
            n_files = sum(1 for p in path.iterdir() if p.is_file())
        except OSError:
            n_files = -1
        table.append({"path": path, "depth": depth, "n_files": n_files})
    if not table:
        return None, table
    table.sort(key=lambda t: t["n_files"], reverse=True)
    return table[0]["path"], table



class LazyParquetWriter:
    """A parquet writer whose schema comes from the first table written.

    `pyarrow.parquet.ParquetWriter` requires the schema at construction, which
    is incompatible with opening a writer only when the first partition is
    ready -- the lazy pattern the assembly cell uses for 16 shards and three
    outputs. This defers construction to the first `write_table` and reuses the
    writer afterwards.

    `close()` is safe on a writer that never received a table, so callers do not
    need to track whether anything was written.
    """

    def __init__(self, path, compression="snappy"):
        self.path = Path(path)
        self.compression = compression
        self._writer = None
        self.rows_written = 0
        #: Sticky: records that anything was ever written. `opened` cannot be
        #: used for this after close(), because close() clears the writer and
        #: `opened` then reads False for a file that holds a million rows.
        self.wrote_anything = False

    def write_table(self, table):
        import pyarrow.parquet as pq

        if table is None or getattr(table, "num_rows", 0) == 0:
            return
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(
                self.path, table.schema, compression=self.compression
            )
        self._writer.write_table(table)
        self.rows_written += table.num_rows
        self.wrote_anything = True

    def write_frame(self, frame):
        # Emptiness is checked before importing pyarrow: an empty partition must
        # be a no-op even where pyarrow is absent, and importing first turned a
        # silent skip into a ModuleNotFoundError.
        if frame is None or len(frame) == 0:
            return
        import pyarrow as pa

        self.write_table(pa.Table.from_pandas(frame, preserve_index=False))

    @property
    def opened(self):
        """True while a writer is live. NOT a record of whether rows were written."""
        return self._writer is not None

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
