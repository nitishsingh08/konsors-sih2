# ============================================================================
# P7f -- COLAB CELL: land cover WITHOUT Earth Engine
#
# HONEST SCOPE FIRST
#
# This cell exists because `ee.Initialize()` fails with "no project found"
# until a Google Cloud project exists and is registered. Your CREDENTIALS are
# fine -- `ee.Authenticate()` succeeds. The blocker is a Cloud console project,
# about three minutes of work.
#
# So read the size before choosing this over GEE:
#
#     ESA WorldCover v200 2021, 10 m, 3x3-degree tiles
#     102 tiles cover India (lat 5-38, lon 66-99)
#     ~110 MB each  ->  ~10.4 GB total
#
# That was measured against the live bucket, not guessed. Two consequences:
#
#   * GEE is CHEAPER IN BYTES. `reduceRegions` returns 7.27M sampled values,
#     not 10 GB of raster. If you can spend three minutes in the Cloud console,
#     do that instead.
#   * This cell is the fallback for when you cannot. It is resumable, it caches
#     to Drive, and it refuses to start a large fetch without explicit consent.
#
# Only the 10 m product is published in the free bucket. There is no 100 m COG,
# despite the product documentation mentioning one -- verified: every
# `_100m_..._Map.tif` under `v200/2021/` returns HTTP 404.
#
# WHAT LAND COVER BUYS
#
# WorldCover has 11 substrate classes, and "was there burnable vegetation here?"
# is answerable from them:
#
#     vegetation   tree, shrub, grassland, cropland, herbaceous, wetland, moss
#     non-vegetated  built-up, bare/sparse, water
#
# That question has NOTHING to do with a facility gazetteer, which is the
# problem with the current label. A label of "burnt a non-vegetated substrate"
# is independent of the 5 km proximity rule and unreadable from X.
#
# It is NOT the same claim as "industrial". A stubble burn 300 m from a
# refinery is cropland, and this correctly calls it a vegetation fire. That is
# the point: it stops being a proximity proxy.
#
# WHAT IT DOES NOT GIVE YOU
#
# Surface DAMAGE -- NBR drop and non-recovery -- needs a Sentinel-2 time series,
# which means GEE or an equivalent download. Land cover gives you the substrate,
# not the burn scar. Add NBR as a second stage if you get GEE working.
#
# RESOLUTION: 300 m
#
# Each 10 m tile is 36000x36000. Read at full resolution that is 1.3 GB
# decompressed per tile, which is neither necessary nor affordable. The cell
# reads each tile downsampled to ~300 m with nearest-neighbour resampling. For
# a substrate CLASS at an event centroid, 300 m is ample -- and it is stated
# rather than buried, because a resolution claim that is not stated is a
# resolution claim that is wrong.
# ============================================================================

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import colab_utils as cu

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError(f"no project root under {cu.MYDRIVE}")
QC_DIR = PROJECT_ROOT / "qc"
CACHE = QC_DIR / "worldcover_tiles"
QC_DIR.mkdir(parents=True, exist_ok=True)

HOST = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
#: Verified against the live bucket: 10 m only, under /map/.
TEMPLATE = f"{HOST}/v200/2021/map/ESA_WorldCover_10m_2021_v200_{{tile}}_Map.tif"
TILE_DEG = 3
#: Downsample factor from the native 10 m.
DOWNSAMPLE = 30                      # -> ~300 m

N_EVENTS = int(globals().get("N_LANDCOVER_EVENTS", 600_000))
MAX_GB = float(globals().get("WORLDCOVER_MAX_GB", 12.0))
ALLOW_DOWNLOAD = bool(globals().get("WORLDCOVER_DOWNLOAD", False))

#: See colab_bootstrap.CELL_VERSION_FLOORS. Bump on every change so the
#: bootstrap can tell a current cell from an old one.
P7F_CELL_VERSION = "2026-10-04.1"

print("=" * 74)
print(f"P7f -- land cover without Earth Engine  [v{P7F_CELL_VERSION}]")
print("=" * 74)
print(cu.drive_report())
print(f"template : {TEMPLATE}")
print(f"tile     : {TILE_DEG}x{TILE_DEG} deg, 10 m native, read at "
      f"~{10 * DOWNSAMPLE} m")
print(f"cache    : {CACHE}")
print(f"scope    : {N_EVENTS:,} events, cap {MAX_GB:.1f} GB, "
      f"download {'ENABLED' if ALLOW_DOWNLOAD else 'DISABLED'}")

try:
    import rasterio
    from rasterio.enums import Resampling
    print(f"rasterio : {rasterio.__version__}")
except ImportError:
    print("rasterio NOT INSTALLED -> !pip install rasterio")


# ---------------------------------------------------------------------------
# 1. EVENTS
# ---------------------------------------------------------------------------
# find_artefact_path returns a Path, not the (Path, depth) tuple that
# find_artefact returns. Unpacking by hand in a cell is exactly how P7e ended
# up handing a TUPLE to read_parquet.
_src = cu.find_artefact_path(PROJECT_ROOT, "canonical_event_base.parquet")
if _src is None:
    _src = cu.find_artefact_path(PROJECT_ROOT, "meta.parquet")
    if _src is not None:
        print()
        print("NOTE: using meta.parquet, not the event base.")
if _src is None:
    print("could not find canonical_event_base.parquet or meta.parquet")
    print(cu.describe_missing_artefact(PROJECT_ROOT,
                                       "canonical_event_base.parquet"))
    print(cu.describe_missing_artefact(PROJECT_ROOT, "meta.parquet"))
    raise SystemExit("no event table found")
print(f"  event source: {_src}")

_avail = set(pq.ParquetFile(_src).schema_arrow.names)
_keep = [c for c in ("event_centroid_lon", "event_centroid_lat") if c in _avail]
if len(_keep) < 2:
    raise SystemExit(f"{_src} has no lon/lat; found {_keep}")

meta = pd.read_parquet(_src, columns=_keep).dropna().reset_index(drop=True)
print()
print(f"events with coordinates: {len(meta):,}  from {_src.name}")

if len(meta) > N_EVENTS:
    # Stratified on a 1-degree cell. Plain `sample(frac=)` over-weights dense
    # clusters, which is exactly where a facility gazetteer already dominates.
    _frac = N_EVENTS / len(meta)
    _cell = (np.floor(meta["event_centroid_lat"]).astype(int).astype(str)
             + "_"
             + np.floor(meta["event_centroid_lon"]).astype(int).astype(str))
    _n0 = len(meta)
    meta = (meta.assign(_cell=_cell.to_numpy())
                .groupby("_cell", group_keys=False, sort=False)
                .sample(frac=_frac, random_state=0)
                .drop(columns=["_cell"])
                .reset_index(drop=True))
    print(f"stratified subsample   : {len(meta):,} of {_n0:,} "
          f"({100.0 * len(meta) / _n0:.1f}%, {_cell.nunique():,} cells)")

lat = meta["event_centroid_lat"].to_numpy()
lon = meta["event_centroid_lon"].to_numpy()
print(f"lat {lat.min():.2f}..{lat.max():.2f}   lon {lon.min():.2f}..{lon.max():.2f}")


# ---------------------------------------------------------------------------
# 2. TILE NAMES
#
# WorldCover tiles are 3x3 degrees named N{lat}E{lon} with the sign folded
# into a zero-padded number: 75E is E075, 5W is W005. Getting this wrong
# produces 404s that read like a network fault.
# ---------------------------------------------------------------------------
def tile_name(lat_v: int, lon_v: int) -> str:
    ns = "N" if lat_v >= 0 else "S"
    ew = "E" if lon_v >= 0 else "W"
    return f"{ns}{abs(lat_v):02d}{ew}{abs(lon_v):03d}"


tile_lat = np.floor(lat / TILE_DEG).astype(int) * TILE_DEG
tile_lon = np.floor(lon / TILE_DEG).astype(int) * TILE_DEG
tile_keys = sorted({(int(a), int(b)) for a, b in zip(tile_lat, tile_lon)})
tiles = [tile_name(a, b) for a, b in tile_keys]

print()
print(f"tiles touched by these events: {len(tiles)}")
print(f"  {', '.join(tiles[:10])}{' ...' if len(tiles) > 10 else ''}")


# ---------------------------------------------------------------------------
# 3. SIZE EVERY TILE BEFORE DOWNLOADING ANYTHING
# ---------------------------------------------------------------------------
try:
    import requests
except ImportError:
    print("requests NOT INSTALLED -> !pip install requests")
    raise SystemExit("P7f needs requests to size the tiles first")

print()
print("HEAD-ing each tile ...")
found, total, missing = [], 0, []
for t in tiles:
    url = TEMPLATE.format(tile=t)
    try:
        r = requests.head(url, timeout=30, allow_redirects=True)
        if r.status_code == 200:
            n = int(r.headers.get("Content-Length", 0))
            found.append((t, url, n))
            total += n
        else:
            missing.append((t, r.status_code))
    except Exception as exc:                        # noqa: BLE001
        missing.append((t, f"{type(exc).__name__}"))

_cached_bytes = 0
_cached = 0
if CACHE.exists():
    for _t, _u, _n in found:
        _p = CACHE / f"wc_{_t}.tif"
        if _p.exists() and _p.stat().st_size == _n:
            _cached += 1
            _cached_bytes += _n
_to_go_bytes = max(0, total - _cached_bytes)

print()
print(f"  tiles reachable : {len(found)}/{len(tiles)}")
if missing:
    # 404 on a 3x3-degree grid cell is NORMAL, not a fault: WorldCover only
    # publishes tiles containing land. Over India roughly 19 of 121 grid slots
    # are open ocean (e.g. N06E066 in the Arabian Sea, N09E066 west of Gujarat)
    # and will always 404. Events landing in one are off-coast and are reported
    # as UNMAPPED in the sampling section.
    print(f"  no such tile    : {len(missing)}  (expected for ocean cells)")
    print(f"                     e.g. "
          + ", ".join(f"{t}" for t, _s in missing[:6]))
print(f"  already cached  : {_cached}  ({_cached_bytes / 1024 ** 3:.2f} GB)")
print(f"  to download     : {len(found) - _cached}")
print(f"  transfer left   : {_to_go_bytes / 1024 ** 3:.2f} GB "
      f"(of {total / 1024 ** 3:.2f} GB total)")

# ---------------------------------------------------------------------------
# 4. CONSENT
# ---------------------------------------------------------------------------
_to_go = len(found) - _cached
_go_bytes = _to_go_bytes
print()
print("=" * 74)
print("SCOPE")
print("=" * 74)
print(f"  events        : {len(meta):,} (stratified on 1-degree cells)")
print(f"  tile reads    : {len(found)}")
print(f"  transfer left : {_go_bytes / 1024 ** 3:.2f} GB")
print()
if _go_bytes / 1024 ** 3 > MAX_GB:
    print(f"  BLOCKED: {_go_bytes / 1024 ** 3:.2f} GB exceeds "
          f"WORLDCOVER_MAX_GB={MAX_GB:.1f}.")
    print("  Two honest options:")
    print("    (a) Spend three minutes creating a Cloud project and use GEE,")
    print("        which transfers sampled values instead of 10 GB of raster.")
    print("    (b) Raise WORLDCOVER_MAX_GB deliberately and let it run; the")
    print("        cache makes it resumable across sessions.")
    ALLOW_DOWNLOAD = False
elif not ALLOW_DOWNLOAD:
    print("  NOT DOWNLOADING. Set WORLDCOVER_DOWNLOAD = True in a cell above")
    print("  and re-run, once you have read the number above. A multi-GB")
    print("  fetch should never begin without a human deciding to start it.")

if not ALLOW_DOWNLOAD:
    print()
    print("=" * 74)
    print("P7f STOPPED before downloading. Nothing was written.")
    print("=" * 74)
else:
    # -----------------------------------------------------------------------
    # 5. DOWNLOAD (resumable)
    # -----------------------------------------------------------------------
    CACHE.mkdir(parents=True, exist_ok=True)
    print()
    print(f"downloading {_to_go} tile(s), {_go_bytes / 1024 ** 3:.2f} GB, "
          f"to {CACHE} ...")
    t0 = time.time()
    for i, (t, url, n) in enumerate(found, 1):
        dest = CACHE / f"wc_{t}.tif"
        if dest.exists() and dest.stat().st_size == n:
            continue
        try:
            tmp = dest.with_suffix(".part")
            with requests.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            tmp.replace(dest)          # atomic, so a kill leaves no half-tile
        except Exception as exc:                    # noqa: BLE001
            print(f"    [{i}/{len(found)}] FAIL {t}: "
                  f"{type(exc).__name__}: {str(exc)[:110]}")
    print(f"  download finished in {time.time() - t0:.0f}s")

    # -----------------------------------------------------------------------
    # 6. SAMPLE
    # -----------------------------------------------------------------------
    print()
    print("=" * 74)
    print("SAMPLING -- substrate class at each event centroid")
    print("=" * 74)
    classes = np.full(len(meta), -1, dtype=np.int16)
    t0 = time.time()
    for (la, lo) in tile_keys:
        t = tile_name(la, lo)
        path = CACHE / f"wc_{t}.tif"
        if not path.exists():
            continue
        idx = np.flatnonzero((tile_lat == la) & (tile_lon == lo))
        if idx.size == 0:
            continue
        try:
            with rasterio.open(path) as ds:
                oh = max(1, ds.height // DOWNSAMPLE)
                ow = max(1, ds.width // DOWNSAMPLE)
                # Nearest, not average: averaging 10 m classes to 300 m would
                # produce fractional class codes that match no class at all.
                arr = ds.read(1, out_shape=(oh, ow),
                              resampling=Resampling.nearest)
                xs = ((lon[idx] - ds.bounds.left) / ds.res[0]
                      / DOWNSAMPLE).astype(int)
                ys = ((lat[idx] - ds.bounds.top) / ds.res[1]
                      / DOWNSAMPLE).astype(int)
                ok = ((xs >= 0) & (xs < ow) & (ys >= 0) & (ys < oh))
                if ok.any():
                    classes[idx[ok]] = arr[ys[ok], xs[ok]]
        except Exception as exc:                    # noqa: BLE001
            print(f"  {t}: {type(exc).__name__}: {str(exc)[:110]}")
        if (la, lo) == tile_keys[min(len(tile_keys) - 1, len(tile_keys) // 2)]:
            print(f"    ... {time.time() - t0:.0f}s elapsed")

    _n_ok = int((classes >= 0).sum())
    print(f"  sampled {_n_ok:,} of {len(meta):,} in {time.time() - t0:.0f}s")

    # Transcribed from the ESA WorldCover class table, NOT from memory. The
    # previous table named 80 "herbaceous" when it is PERMANENT WATER BODIES,
    # and put it in the vegetated set -- which would have labelled open water as
    # burnable substrate and corrupted the very label this work exists to fix.
    _NAMES = {10: "tree cover", 20: "shrubland", 30: "grassland",
              40: "cropland", 50: "built-up", 60: "bare / sparse vegetation",
              70: "snow and ice", 80: "permanent water bodies",
              90: "herbaceous wetland", 95: "mangroves",
              100: "moss and lichen"}
    print()
    print(f"  {'class':<7} {'name':<14} {'n':>12} {'share':>8}")
    for c, nm in sorted(_NAMES.items()):
        k = int((classes == c).sum())
        print(f"  {c:<7} {nm:<14} {k:>12,} {100.0 * k / len(meta):>7.2f}%")
    _miss = int((classes < 0).sum())
    print(f"  {'-':<7} {'UNMAPPED':<14} {_miss:>12,} "
          f"{100.0 * _miss / len(meta):>7.2f}%")
    if _miss > 0.02 * len(meta):
        print()
        print("  >2% unmapped is too many to trust. Check the tile naming and")
        print("  the bounds maths. A near-zero share means the URL template is")
        print("  wrong, not that the data is absent.")

    # Burnable substrate only. Built-up, bare, snow and PERMANENT WATER are
    # excluded: 80 and 70 were previously in this set.
    _VEG = sorted({10, 20, 30, 40, 90, 95, 100})
    _NON_VEG = sorted({50, 60, 70, 80})
    _mapped = classes >= 0
    _veg = np.isin(classes, _VEG)
    out = pd.DataFrame({
        "event_centroid_lon": lon,
        "event_centroid_lat": lat,
        "WORLDCOVER_CLASS": classes,
        "LANDCOVER_NAME": [_NAMES.get(int(c), "unmapped") for c in classes],
        # -1 for unmapped, so a missing tile can never read as "bare".
        "LANDCOVER_VEGETATED": np.where(_mapped, _veg.astype("int8"), -1),
    })
    _path = QC_DIR / "landcover_worldcover.csv"
    out.to_csv(_path, index=False)
    print()
    print(f"Wrote {_path}  ({len(out):,} rows, "
          f"{10 * DOWNSAMPLE} m resolution)")
    if _mapped.any():
        print(f"vegetation substrate among mapped: "
              f"{100.0 * _veg[_mapped].mean():.1f}%")
    print()
    print("NEXT STEP, and the one that actually matters:")
    print("  WORLDCOVER_CLASS is NOT a valid model feature while the weak")
    print("  label ignores it -- that would be leakage of the same kind as")
    print("  DIST_*. It must first be used to BUILD a new label:")
    print("    substrate_non_vegetated = LANDCOVER_VEGETATED == 0")
    print("  then re-run P7 with the leakage partition recomputed so")
    print("  LANDCOVER_VEGETATED is excluded from model A.")