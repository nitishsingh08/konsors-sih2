# ============================================================================
# P3d -- COLAB CELL: DAILY FWI CHAIN + daily weather, from ERA5-Land
#
# WHAT THIS FIXES
#
# The seven FWI features (FFMC DMC DC ISI BUI FWI FWI_ANOMALY_Z) are the
# ledger's HISTORICAL_NAN block: cell 264 hardcoded all seven in both
# branches. `fwi.py` computed them, but on a ONE-MONTH step and a 12-value
# day-length table. `fwi_daily.py` replaces both approximations with a real
# daily chain. It also fixes three genuine bugs in `fwi.py`:
#
#   1. The moisture<->code inverse was wrong, so RAIN DRIED THE FUEL.
#   2. The no-rain branch re-derived moisture from RH, so FFMC had NO
#      memory -- it sat at its start value whatever the weather.
#   3. DMC's day-length factor was in hours, so one hot day drove DMC from
#      6 to 420.
#
# HOW IT IS CHEAP ENOUGH TO RUN
#
# The chain must be daily and per-place, so the work is done once per
# 1-degree CELL, not per event: ~900 cells x ~5,340 days. Events then read
# their own cell and date out of the result. That is the difference between
# 7.26M chains and 900.
#
# ONE CALL PER DAY, SHARDED BY MONTH, RESUMABLE. Each call samples every cell
# centroid in one `sampleRegions` -- ~900 points, well under the 10 MiB
# request limit. A month is ~30 calls and one shard, so a crash loses 30 days.
#
# Output: qc/fwi_daily_events.parquet   (row, event_id, 7 FWI + daily weather)
# ============================================================================

import json
import time
from pathlib import Path

import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import colab_utils as cu
import gee_utils as gee
import fwi_daily as fd

P3D_CELL_VERSION = "2026-10-13.1"

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError(f"no project root under {cu.MYDRIVE}")
DATASET_DIR = PROJECT_ROOT / "full_2012_2026" / "features" / "dataset"
QC_DIR = PROJECT_ROOT / "qc"
SHARD_DIR = QC_DIR / "fwi_daily_shards"
SHARD_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = QC_DIR / "fwi_daily_manifest.json"

#: ERA5-Land daily aggregates. DAILY_AGGR, NOT HOURLY -- but DO NOT TRUST THE
#: ASSET ID: this cell first ran against `DAILY_AGGR` and got a month of
#: all-NaN shards, because most days in that collection had no imagery for the
#: requested window. So the id is PROBED against a known-complete month below
#: and the first candidate that actually carries data wins. A silent fallback
#: to an empty collection is what already cost P7o 100% of its points.
ERA5_CANDIDATES = [
    "ECMWF/ERA5_LAND/DAILY_AGGR",
    "ECMWF/ERA5_LAND/HOURLY",
]
#: Probe over a month that is certainly inside every archive.
PROBE_RANGE = ("2020-01-01", "2020-02-01")
#: All 12 calendar months, because the chain must run continuously through
#: winter -- overwintering is not modelled (see fwi_daily's docstring).
CHAIN_START = (2012, 1, 1)
CHAIN_END = (2026, 8, 31)

print("=" * 74)
print(f"P3d -- DAILY FWI CHAIN  [v{P3D_CELL_VERSION}]")
print("=" * 74)
print(cu.drive_report())

_ok = False
_pid = None
try:
    _ok, _pid = gee.authenticate(globals().get("EARTHENGINE_PROJECT"))
    gee.verify_can_read(_pid)
    print(f"  Earth Engine working (project {_pid!r})")
except Exception as exc:                            # noqa: BLE001
    print(f"  Earth Engine NOT usable: {type(exc).__name__}: {str(exc)[:200]}")
    raise SystemExit("P3d needs a verified Earth Engine session")

import ee                                                          # noqa: E402

# ---------------------------------------------------------------------------
# 1a. WHICH COLLECTION ACTUALLY HAS DATA
#
# Probed, not assumed. The first run of this cell picked `DAILY_AGGR` on
# faith, wrote a month of shards that were 100% NaN, and only the per-month
# coverage print made it visible. An asset id that exists but is empty is
# indistinguishable from an asset id that does not exist until you count.
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("1a. COLLECTION PROBE")
print("=" * 74)
ERA5 = None
for _cand in ERA5_CANDIDATES:
    try:
        _n = ee.ImageCollection(_cand).filterDate(*PROBE_RANGE).size().getInfo()
        _days = ee.ImageCollection(_cand).filterDate(*PROBE_RANGE).distinct(
            "system:time_start").size().getInfo()
    except Exception as exc:                        # noqa: BLE001
        print(f"  {_cand:<32} ERROR {type(exc).__name__}: {str(exc)[:70]}")
        continue
    _has_bands = False
    try:
        _im = ee.ImageCollection(_cand).filterDate(*PROBE_RANGE).first()
        _has_bands = bool(_im.bandNames().getInfo())
    except Exception:                               # noqa: BLE001
        pass
    print(f"  {_cand:<32} images {int(_n):>5,}  days {int(_days):>3,}  "
          f"bands {'yes' if _has_bands else 'NO'}")
    if ERA5 is None and int(_days) >= 28 and _has_bands:
        ERA5 = _cand
if ERA5 is None:
    raise SystemExit(
        "no ERA5-Land collection carries data for the probe window. The chain "
        "cannot run -- do NOT let it run on an empty archive.")
print(f"  --> using {ERA5}")

# ---------------------------------------------------------------------------
# 1b. BAND NAMES, PROBED NOT ASSUMED
#
# This project has been bitten three times by a band name that was almost
# right (`CLOUD_COVER` vs `CLOUDY_PIXEL_PERCENTAGE`, `Map` vs `cover`,
# `temperature_2m` vs `temperature_2m_stdDev`). So resolve them once, print
# them, and refuse a silent partial match.
# ---------------------------------------------------------------------------
WANTED = {
    "temperature_2m": "TEMP_C",
    "dewpoint_temperature_2m": "DEWPOINT_C",
    "u_component_of_wind_10m": "UWIND_MS",
    "v_component_of_wind_10m": "VWIND_MS",
    "total_precipitation_sum": "RAIN_MM",
    "volumetric_soil_water_layer_1": "SWVL1",
}
#: How each band must be reduced to a day. RAIN is an ACCUMULATION, so the
#: daily total is a SUM; averaging 24 hourly rainfall values under-reads the
#: day by roughly 24x. Everything else is a state variable, so the mean is
#: right -- and that is what the FWI equations expect.
REDUCE_AS = {
    "TEMP_C": "mean",
    "DEWPOINT_C": "mean",
    "UWIND_MS": "mean",
    "VWIND_MS": "mean",
    "RAIN_MM": "sum",
    "SWVL1": "mean",
}
#: UNITS. Earth Engine hands back SI, so both must be converted before any
#: equation that expects field units sees them. The probe confirmed
#: DAILY_AGGR carries `total_precipitation_sum` (metres, already a daily
#: total) and `temperature_2m` (KELVIN).
#:
#: The existing P7n output is MISLABELED on exactly these two: its `TEMP_C`
#: column holds kelvin (median 299.4) and its `RAIN_MM` holds metres (median
#: 0.0003). A model does not care -- it is invariant to a monotone rescale --
#: so Model B is unaffected. FWI is NOT invariant: feeding 299 K into the
#: moisture equations makes the equilibrium content wildly negative and every
#: index meaningless. So convert here, and validate what came out.
UNIT_SCALE = {
    "TEMP_C": (lambda v: v - 273.15, "K -> deg C"),
    "DEWPOINT_C": (lambda v: v - 273.15, "K -> deg C"),
    "UWIND_MS": (lambda v: v, "m/s"),
    "VWIND_MS": (lambda v: v, "m/s"),
    "RAIN_MM": (lambda v: v * 1000.0, "m -> mm"),
    "SWVL1": (lambda v: v, "m3/m3"),
}
#: Physical plausibility, used to refuse a day that is quietly wrong.
#: These are wide enough for any Indian climate and narrow enough to catch a
#: unit error: kelvin-as-Celsius gives TEMP_C of 250-320.
#:
#: `RAIN_MM` is checked SEPARATELY, on its maximum rather than its bounds,
#: because a metres-for-mm error is invisible to a range check: 0.0001..0.1 m
#: is perfectly "within 0..1000". What identifies it is that no day is wet --
#: a real Indian daily series has days of tens of mm, so a maximum under 1 mm
#: means the column is a thousand times too small.
PLAUSIBLE = {
    "TEMP_C": (-50.0, 65.0),
    "DEWPOINT_C": (-60.0, 65.0),
    "UWIND_MS": (-60.0, 60.0),
    "VWIND_MS": (-60.0, 60.0),
    "SWVL1": (0.0, 1.0),
}
#: Bands whose UNIT error a range check cannot see. (min plausible maximum,
#: hard absolute maximum). Daily rainfall must reach at least 1 mm somewhere
#: and cannot reach 1 m anywhere.
UNIT_SENSITIVE_MAX = {
    "RAIN_MM": (1.0, 1000.0),
}
#: ERA5-Land's daily accumulation carries floating-point negatives of around
#: -1e-5 m, i.e. -0.01 mm. Rain cannot be negative, so clip -- do NOT reject
#: the day over it: rejecting cost 345 cells per occurrence and caught a
#: rounding artefact, not a data error.
CLIP_NON_NEGATIVE = ("RAIN_MM",)

print()
print("=" * 74)
print("1b. BAND PROBE")
print("=" * 74)
_first = ee.ImageCollection(ERA5).filterDate(*PROBE_RANGE).first()
_avail = set(_first.bandNames().getInfo())
print(f"  bands available: {sorted(_avail)}")

# `BANDS` maps SHORT NAME -> actual Earth Engine band name. Each band is then
# reduced and RENAMED to its short name in `daily_composite`, so the property
# key read back from `sampleRegions` is the short name and the mapping is
# written exactly once. The first version stored the mapping the other way
# round (EE name -> short) and read `p.get(BANDS["temperature_2m"])`, which
# is "TEMP_C" -- a key that never exists -- so EVERY value became NaN. The
# failure was invisible: a perfectly-shaped parquet of nothing.
BANDS = {}
for ee_name, short in WANTED.items():
    if ee_name in _avail:
        BANDS[short] = ee_name
    else:
        # A `_sum` suffix vs plain, and vice versa: precipitation is the one
        # that genuinely differs between the daily and hourly collections.
        alts = [b for b in _avail if b.startswith(ee_name.rsplit("_", 1)[0])]
        if len(alts) == 1:
            BANDS[short] = alts[0]
            print(f"  NOTE: {ee_name!r} not present; using {alts[0]!r}")
        else:
            raise SystemExit(
                f"ERA5-Land daily has no band for {ee_name!r} (wanted as "
                f"{short}); candidates {sorted(alts)}. Check the collection "
                f"id -- DAILY_AGGR and HOURLY do not carry the same names.")

# Every requested band must have resolved, or the chain runs on a partial
# input and reports FWI that is quietly meaningless.
_missing = [s for s in WANTED.values() if s not in BANDS]
if _missing:
    raise SystemExit(f"band probe did not resolve {_missing} -- refusing to "
                     "run a chain on a partial input")
print(f"  resolved: {BANDS}")


def _num(value):
    """A property as a float, or NaN for a masked / missing pixel.

    `sampleRegions` returns None for a masked pixel and omits the property
    entirely when the band was not in the image. `float(None)` raises
    TypeError, which is what the first version did -- the day aborted and the
    parquet silently had fewer rows than days.
    """
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")

# ---------------------------------------------------------------------------
# 2. THE GRID CELLS -- one chain per cell, events join back by cell+date
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("2. CELL GRID")
print("=" * 74)
_MC = set(pq.ParquetFile(DATASET_DIR / "meta.parquet").schema_arrow.names)
for _c in ("event_centroid_lon", "event_centroid_lat", "start_time"):
    if _c not in _MC:
        raise SystemExit(f"meta.parquet lacks {_c!r} -- run P6 first")
_keep = [c for c in ("event_centroid_lon", "event_centroid_lat",
                     "event_id", "start_time") if c in _MC]
meta = pd.read_parquet(DATASET_DIR / "meta.parquet", columns=_keep)
N = len(meta)

lat_num = pd.to_numeric(meta["event_centroid_lat"], errors="coerce").to_numpy("float64")
lon_num = pd.to_numeric(meta["event_centroid_lon"], errors="coerce").to_numpy("float64")

# Cell id from BOTH coordinates, in a fixed order. Integer arithmetic on
# floor() of each: cell = (lat_int + 90) * 360 + (lon_int + 180), so lat is
# the major axis and lon the minor -- the same encoding `colab_p3g` uses.
# NEVER derive the id from latitude alone, or lat/lon cells alias across
# the map, and never from a swapped pair: the first version of this cell
# assigned latitude and longitude the wrong way round, so its day-length
# used longitudes and its join keys never matched.
lat_f = np.floor(lat_num).astype("int64")
lon_f = np.floor(lon_num).astype("int64")
cell_id = (lat_f + 90) * 360 + (lon_f + 180)

# ORDERED CELL TABLE: (cell_id, lat_centre, lon_centre), sorted by id so the
# index IS the cell number.
cell_tab = (pd.DataFrame({"cell": cell_id,
                          "lat_c": lat_num, "lon_c": lon_num})
            .groupby("cell", as_index=False)
            .agg(lat_c=("lat_c", "mean"), lon_c=("lon_c", "mean")))
cell_tab = cell_tab.sort_values("cell").reset_index(drop=True)
CELL_LAT = cell_tab["lat_c"].to_numpy("float64")
CELL_LON = cell_tab["lon_c"].to_numpy("float64")
N_CELLS = len(cell_tab)
print(f"  events {N:,}   distinct 1-degree cells {N_CELLS:,}")
print(f"  cell id range {int(cell_tab['cell'].min()):,} .. "
      f"{int(cell_tab['cell'].max()):,}")
print(f"  lat centres {CELL_LAT.min():.1f} .. {CELL_LAT.max():.1f}   "
      f"lon centres {CELL_LON.min():.1f} .. {CELL_LON.max():.1f}")

# ONE cell FeatureCollection per sampling chunk. `sampleRegions` on a 355-
# point collection returned only 40 of them -- silently, with no error -- so
# the cell collection is CHUNKED and every dropped cell is tracked.
SAMPLE_CELL_CHUNK = 40
#: POSITION in the ordered cell table -> how many days it was dropped. The
#: count must be exact for the geography/flakiness split below to mean
#: anything, so it is filled by comparing global id to global id (see the
#: sampling loop). A cell dropped on EVERY day is over water or masked; a cell
#: dropped on a few days is API flakiness.
DROPPED_CELLS: dict = {}


def cell_fc_range(lo, hi):
    """FeatureCollection for cells [lo, hi) of the ordered cell table.

    The `cell` property carries the GLOBAL cell id (the same `cell_id` the
    events join on), NOT the positional index. The first version wrote the
    index here, so the chain's `cell` column held 0..354 while the events'
    `cell_id` held 34,833..45,257 -- the merge matched nothing and the whole
    feature collapsed onto one Arctic row.
    """
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point(float(CELL_LON[i]), float(CELL_LAT[i])),
                   {"cell": int(cell_tab["cell"].iloc[i])})
        for i in range(lo, hi)
    ])

# ---------------------------------------------------------------------------
# 3. THE DAY INDEX, AND WHICH MONTHS ARE ALREADY DONE
# ---------------------------------------------------------------------------
def _all_days():
    out = []
    cur = datetime.date(*CHAIN_START)
    end = datetime.date(*CHAIN_END)
    while cur <= end:
        out.append(cur)
        cur += datetime.timedelta(days=1)
    return out


DAYS = _all_days()
print(f"  chain span: {DAYS[0]} .. {DAYS[-1]}  ({len(DAYS):,} days)")

# Month shards, so a crash loses ~30 days of sampling.
def _month_key(dt):
    return f"{dt.year:04d}{dt.month:02d}"


months = sorted({_month_key(d) for d in DAYS})
shard_of = {d: _month_key(d) for d in DAYS}

done_months = {p.stem.split("_")[-1] for p in SHARD_DIR.glob("fwid_*.parquet")}
to_do = [mk for mk in months if mk not in done_months]
print(f"  months total {len(months)}   done {len(done_months)}   "
      f"remaining {len(to_do)}")
if not to_do:
    print("  every month already sampled -- chain and join only")

# ---------------------------------------------------------------------------
# 4. SAMPLE ERA5-LAND, ONE CALL PER DAY
# ---------------------------------------------------------------------------
def daily_composite(d0, d1):
    """One day, one image, bands named by their SHORT name.

    Per-band reduction, because the two families need different ones:
    rainfall is an ACCUMULATION so a day is the SUM of its hours, while
    temperature/RH/wind/soil-water are STATE variables whose daily mean is
    what the FWI equations want. A single `.mean()` over the whole collection
    -- what the first version did -- under-reads rain by roughly 24x on an
    hourly archive and makes FWI respond to the wrong wetting.

    Bands are RENAMED to the short names here, so the property key read back
    from `sampleRegions` is the short name. Doing the mapping in one place
    removes the whole class of bug that produced a shard of nothing.
    """
    imgs = []
    for short, ee_name in BANDS.items():
        col = ee.ImageCollection(ERA5).filterDate(d0, d1).select(ee_name)
        red = col.sum() if REDUCE_AS.get(short, "mean") == "sum" else col.mean()
        imgs.append(red.rename(short))
    # `ee.Image([img, ...])` concatenates bands in list order.
    return ee.Image(imgs)


def sample_day(img, day):
    """Every cell for one day, chunked, reporting any cell it cannot get.

    WHAT THE SHORT READ ACTUALLY IS
    ------------------------------
    `sampleRegions` returned 36 of 40 points, and 40 of 355 on the first run.
    The missing ones are not a transport limit and not random: ERA5-Land is a
    LAND product, so a cell whose centroid falls over water or an inland sea
    returns no feature at all. The first run silently produced a chain built on
    40 of 355 cells and reported it as complete.

    So a dropped cell is RECORDED, not fatal. Aborting here would cost the
    whole 50-minute run over a handful of offshore cells that no event
    depends on. What must not happen is the drop going unnoticed, so:
      * every dropped cell is filled with NaN for that day,
      * the dropped set is tracked across days,
      * if the same cells drop every day that is a GEOGRAPHY problem and they
        are printed; if the set changes day to day it is API flakiness.

    The 50% coverage gate at the end is what actually decides whether the
    run is usable.
    """
    rows = []
    for lo in range(0, N_CELLS, SAMPLE_CELL_CHUNK):
        hi = min(lo + SAMPLE_CELL_CHUNK, N_CELLS)
        want = hi - lo
        res = (img.sampleRegions(collection=cell_fc_range(lo, hi),
                                 properties=["cell"], scale=11132,
                                 geometries=False)
               .getInfo())
        feats = res.get("features", [])
        got = {int(f["properties"]["cell"]) for f in feats}
        if len(got) != want:
            # Compare global ids to global ids. `range(lo, hi)` yields
            # POSITIONS into the ordered cell table (0..N_CELLS-1), while
            # `got` holds the GLOBAL ids the `cell` property now carries
            # (35,528..44,911). The two sets never intersect, so testing
            # positions against `got` flagged EVERY cell in the chunk as
            # dropped whenever any one was missing -- which collapsed the
            # geography/flakiness split this dictionary exists to make.
            #
            # Keys stay positional on purpose: the report further down reads
            # `CELL_LAT[c]`, which is a positional table.
            chunk_ids = [int(cell_tab["cell"].iloc[i]) for i in range(lo, hi)]
            missing = [i for i, cid in zip(range(lo, hi), chunk_ids) if cid not in got]
            for i in missing:
                DROPPED_CELLS[i] = DROPPED_CELLS.get(i, 0) + 1
            if not got:
                raise RuntimeError(
                    f"{day}: chunk {lo}..{hi} returned NO features at all. "
                    "Empty-everything means the composite is wrong (band or "
                    "reduction), not that these cells are over water.")
        for f in feats:
            p = f["properties"]
            rec = {"cell": int(p["cell"]),
                   "year": day.year, "month": day.month, "day": day.day,
                   "doy": day.timetuple().tm_yday}
            for short in BANDS:
                rec[short] = _num(p.get(short))
            rows.append(rec)
    return pd.DataFrame(rows)


t0 = time.time()
_days_empty = 0
_days_sampled = 0
_days_all_nan = 0
_days_implausible = 0
_days_short = 0
for k, mk in enumerate(to_do, 1):
    days_in = [d for d in DAYS if _month_key(d) == mk]
    frames = []
    for d in days_in:
        d0, d1 = (d.strftime("%Y-%m-%d"),
                  (d + datetime.timedelta(days=1)).strftime("%Y-%m-%d"))
        # A day with no images must not silently become zeros: an empty
        # collection's mean() has the right bands but no data, so check size
        # first -- the same trap that cost P7o 100% of its points. Count it,
        # so an archive that stops early is visible instead of reshaping the
        # chain into something that looks like weather.
        n_img = ee.ImageCollection(ERA5).filterDate(d0, d1).size().getInfo()
        if not n_img:
            _days_empty += 1
            continue
        _n_before = len(DROPPED_CELLS)
        df = sample_day(daily_composite(d0, d1), d)
        if len(DROPPED_CELLS) > _n_before:
            _days_short += 1

        # UNIT AND PLAUSIBILITY GATE, per day, before it can be written.
        # A day that fails is DROPPED, not written, because a silently wrong
        # value is worse than a missing one -- the chain carries the previous
        # state through a gap, which is modelled and reported, whereas a wrong
        # value propagates into every downstream index as if measured.
        _bad = []
        for short in BANDS:
            fn, _note = UNIT_SCALE[short]
            df[short] = fn(df[short].to_numpy("float64"))
            v = df[short]
            if short in CLIP_NON_NEGATIVE:
                # A small negative is an accumulation artefact, not weather.
                df[short] = np.maximum(0.0, df[short].to_numpy("float64"))
                v = df[short]
            if short in PLAUSIBLE and v.notna().any():
                lo_ok, hi_ok = PLAUSIBLE[short]
                if not (v.min() >= lo_ok and v.max() <= hi_ok):
                    _bad.append(f"{short} {v.min():.4g}..{v.max():.4g} "
                                f"outside [{lo_ok}, {hi_ok}] ({_note})")
            if short in UNIT_SENSITIVE_MAX and v.notna().any():
                lo_ok, hi_ok = UNIT_SENSITIVE_MAX[short]
                _mx = float(v.max())
                if not (lo_ok <= _mx <= hi_ok):
                    _bad.append(f"{short} max {_mx:.4g} outside "
                                f"[{lo_ok}, {hi_ok}] {_note} -- "
                                "if it is small the column is a "
                                "thousandfold too small")
        if _bad:
            _days_implausible += 1
            print(f"    !! {d} rejected: {'; '.join(_bad)}")
            continue

        if df[["TEMP_C", "DEWPOINT_C", "RAIN_MM"]].isna().all().all():
            _days_all_nan += 1
        frames.append(df)
        _days_sampled += 1
    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame(columns=["cell", "year", "month", "day", "doy",
                                   "TEMP_C", "DEWPOINT_C", "UWIND_MS",
                                   "VWIND_MS", "RAIN_MM", "SWVL1"])
    tmp = SHARD_DIR / f"fwid_{mk}.parquet.partial"
    df.to_parquet(tmp, index=False)
    tmp.replace(SHARD_DIR / f"fwid_{mk}.parquet")
    MANIFEST.write_text(json.dumps({
        "months_total": len(months), "months_done": len(done_months) + k,
        "cells": int(N_CELLS), "chain_days": int(len(DAYS)),
        "days_sampled": _days_sampled, "days_empty": _days_empty,
        "days_all_nan": _days_all_nan,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))
    _cov = (100.0 * df["TEMP_C"].notna().mean()) if len(df) else 0.0
    print(f"    month {mk}  {len(df):>7,} rows  cov {_cov:5.1f}%  "
          f"empty-days {_days_empty}  all-NaN days {_days_all_nan}  "
          f"({k}/{len(to_do)})  {(time.time()-t0)/60:.1f} min")

# The chain needs at most a handful of empty days -- the wetting-end of the
# archive, or a month where the satellite genuinely saw nothing. A large share
# means the archive is shorter than the catalogue, which silently distorts the
# whole FWI feature rather than erroring.
print()
print("  " + "-" * 70)
print(f"  days sampled {_days_sampled:,}   empty {_days_empty:,}   "
      f"all-NaN {_days_all_nan:,}   days with a dropped cell {_days_short:,}")
if DROPPED_CELLS:
    _stable = [(c, n) for c, n in DROPPED_CELLS.items() if n == _days_sampled]
    _flaky = [(c, n) for c, n in DROPPED_CELLS.items() if n != _days_sampled]
    print(f"  cells `sampleRegions` would not return: {len(DROPPED_CELLS)} "
          f"of {N_CELLS}")
    # Geography or flakiness -- the two need opposite responses.
    if _stable:
        print(f"    dropped EVERY day ({len(_stable)}) -- these are over "
              "water or masked, and ERA5-Land is a land-only product:")
        for c, _ in sorted(_stable)[:15]:
            print(f"      cell {c:>4}  lat {CELL_LAT[c]:>6.2f}  "
                  f"lon {CELL_LON[c]:>6.2f}")
    if _flaky:
        print(f"    dropped on SOME days only ({len(_flaky)}) -- API "
              f"flakiness, worth a re-run if it grows: "
              f"{sorted(c for c, _ in _flaky)[:15]}")
if _days_all_nan:
    print("  FAILURE: days returned rows but no VALUES. This is a band or a")
    print("  resolution problem, not weather. Fix it before the chain runs --")
    print("  otherwise FWI is computed from a default start code.")
elif _days_empty > 0.10 * len(DAYS):
    print(f"  WARNING: {100.0*_days_empty/len(DAYS):.1f}% of days had no "
          "imagery. The archive is probably shorter than the catalogue.")
    print("  The chain carries the previous state through those days, which")
    print("  is the right behaviour, but it means FWI is stale in that span.")
else:
    print("  Sampling looks healthy.")
print("  " + "-" * 70)

# ---------------------------------------------------------------------------
# 5. RUN THE CHAIN FOR EVERY CELL
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("5. THE DAILY CHAIN")
print("=" * 74)
parts = [pd.read_parquet(SHARD_DIR / f"fwid_{mk}.parquet")
         for mk in sorted(done_months | set(to_do))
         if (SHARD_DIR / f"fwid_{mk}.parquet").exists()]
daily = pd.concat(parts, ignore_index=True)
print(f"  sampled rows: {len(daily):,}  cells: {daily['cell'].nunique():,}")

# Derived daily weather. Wind is the magnitude of the 10 m components, in
# km/h -- the unit `fwi_daily` and the published FWI equations both expect.
daily = daily.sort_values(["cell", "year", "month", "day"]).reset_index(drop=True)
daily["WIND_KMH"] = np.hypot(daily["UWIND_MS"], daily["VWIND_MS"]) * 3.6
daily["RH_PCT"] = fd.humidity_from_dewpoint(daily["TEMP_C"].to_numpy(),
                                            daily["DEWPOINT_C"].to_numpy())
daily["VPD_KPA"] = fd.vpd_kpa(daily["TEMP_C"].to_numpy(), daily["RH_PCT"].to_numpy())

# 7-day rolling rain, per cell, on the sorted series. A plain rolling() would
# bleed across a cell boundary, hence the groupby.
daily["RAIN_7D_MM"] = (daily.groupby("cell")["RAIN_MM"]
                       .transform(lambda s: s.rolling(7, min_periods=1).sum()))

# The chain. Arrays are (days, cells) shaped per cell, so run cell by cell --
# a few thousand short loops, still seconds.
print("  running the chain (vectorised over days, looped over cells)...")
# The sampled `cell` is the GLOBAL cell id (34,833..45,257); the ordered
# latitude table is indexed by POSITION (0..354). Without this map the first
# version indexed CELL_LAT by the global id and raised IndexError, and an
# earlier version silently ran the chain at lat 0 for every cell.
_pos_of_cell = {int(cell_tab["cell"].iloc[i]): i
                for i in range(N_CELLS)}
out_rows = []
for cid, grp in daily.groupby("cell", sort=True):
    grp = grp.sort_values(["year", "month", "day"])
    if len(grp) < 2:
        continue
    pos = _pos_of_cell.get(int(cid))
    if pos is None:
        continue
    lat_cell = CELL_LAT[pos]
    ch = fd.chain_for_cells(
        np.asarray(grp["TEMP_C"])[:, None],
        np.asarray(grp["RH_PCT"])[:, None],
        np.asarray(grp["WIND_KMH"])[:, None],
        np.asarray(grp["RAIN_MM"])[:, None],
        np.array([lat_cell], dtype="float64"),
        grp["doy"].to_numpy(),
    )
    for j, key in enumerate(["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]):
        grp[key] = ch[key][:, 0]
    out_rows.append(grp[["cell", "year", "month", "day",
                         "FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]])
chain = pd.concat(out_rows, ignore_index=True)
print(f"  chain rows: {len(chain):,}")
for k in ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]:
    v = chain[k].to_numpy("float64")
    print(f"    {k:5s} {np.nanmin(v):>8.2f} .. {np.nanmax(v):>8.2f}   "
          f"mean {np.nanmean(v):>7.2f}")

# ---------------------------------------------------------------------------
# 6. FWI ANOMALY Z -- vs the cell's own calendar-month distribution
#
# Relative to the SAME cell and month, not a national mean: a Gangetic cell
# and a Rajasthan cell have different baselines, and a single national
# climatology would call Bihar anomalously dangerous by construction.
# ---------------------------------------------------------------------------
chain["month_i"] = chain["month"]
stats = (chain.groupby(["cell", "month_i"])["FWI"]
         .agg(mu="mean", sd="std").reset_index())
stats["sd"] = stats["sd"].replace(0.0, np.nan)
chain = chain.merge(stats, on=["cell", "month_i"], how="left")
chain["FWI_ANOMALY_Z"] = (chain["FWI"] - chain["mu"]) / chain["sd"]

# ---------------------------------------------------------------------------
# 7. JOIN BACK TO EVENTS BY (cell, date)
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("6. JOIN EVENTS")
print("=" * 74)
st_ev = pd.to_datetime(meta["start_time"], errors="coerce")

# THE JOIN KEY IS THE CELL ID, NOT A CENTROID PAIR.
#
# The first version keyed the join on (floor(lat)+0.5, floor(lon)+0.5) while
# the chain table carried the cell's own mean of the event centroids -- two
# different values for the same cell, so the merge silently matched NOTHING
# and reported 0.0% coverage on all fourteen features. `cell_id` is computed
# once above from floor(lat), floor(lon); the chain carries it as `cell`; here
# it is recomputed from the same inputs. One definition, used three times.
#
# The second version of this cell wrote the chain's `cell` as the POSITIONAL
# index (0..354) while events carried the global id (34,833..45,257), so the
# merge matched nothing either and the parquet came back full of a single
# Arctic row. The cell property now carries the global id (see cell_fc_range).
#
# DTYPE. `st.dt.year` returns a float when NaT is present, and the chain's
# year/month/day are ints. Merging on mismatched dtypes raises in modern
# pandas, so cast both sides to int64 before the merge.
ev = pd.DataFrame({
    "row": np.arange(N, dtype="int64"),
    "event_id": (meta["event_id"].to_numpy() if "event_id" in _keep
                 else np.arange(N, dtype="int64")),
    "cell": cell_id,
    "year": st_ev.dt.year.to_numpy("int64"),
    "month": st_ev.dt.month.to_numpy("int64"),
    "day": st_ev.dt.day.to_numpy("int64"),
})

keep_chain = ["cell", "year", "month", "day",
              "FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "FWI_ANOMALY_Z"]
out = ev.merge(chain[keep_chain],
              on=["cell", "year", "month", "day"], how="left")

# Also carry the DAILY weather at the event, so P7n's monthly means can be
# superseded rather than duplicated.
keep_w = ["cell", "year", "month", "day",
          "TEMP_C", "RH_PCT", "WIND_KMH", "RAIN_MM", "RAIN_7D_MM",
          "SWVL1", "VPD_KPA"]
dw = (daily.groupby(["cell", "year", "month", "day"], as_index=False)
      .agg(**{c: (c, "first") for c in ["TEMP_C", "RH_PCT", "WIND_KMH",
                                        "RAIN_MM", "RAIN_7D_MM", "SWVL1",
                                        "VPD_KPA"]}))
out = out.merge(dw[keep_w], on=["cell", "year", "month", "day"], how="left")

final_cols = ["row", "event_id", "FFMC", "DMC", "DC", "ISI", "BUI", "FWI",
              "FWI_ANOMALY_Z", "TEMP_C", "RH_PCT", "WIND_KMH", "RAIN_MM",
              "RAIN_7D_MM", "SWVL1", "VPD_KPA"]
final = out[final_cols].copy()

# COVERAGE GATE. The last run wrote a parquet of 7,265,849 rows in which
# fourteen columns were 0.0% covered, and the only clue was a printed table.
# A feature that is 0% covered and then merged into the training frame is
# indistinguishable from a feature that was never measured, so it must not be
# written. A genuine gap (pre-chain events, pre-2018 archive) shows as partial
# coverage, which this still allows.
min_cov = 0.50
print()
print(f"  {'feature':<20} {'non-NaN':>12} {'coverage':>10} {'mean':>9} "
      f"{'min':>9} {'max':>9}")
for c in final_cols[2:]:
    v = final[c]
    print(f"  {c:<20} {int(v.notna().sum()):>12,} "
          f"{100.0*v.notna().mean():>9.1f}% {v.mean():>9.2f} "
          f"{v.min():>9.2f} {v.max():>9.2f}")

worst = min(100.0 * final[c].notna().mean() for c in final_cols[2:])
print()
if worst < min_cov * 100.0:
    raise SystemExit(
        f"COVERAGE TOO LOW: worst feature {worst:.1f}% < {min_cov*100:.0f}%. "
        "Refusing to write a table that is mostly empty. The usual causes, "
        "in order: (1) a stale /content copy of this cell, (2) the band map "
        "inverted so every property read is NaN, (3) the join key differing "
        "between the chain and the events. Fix that, then re-run.")

out_path = QC_DIR / "fwi_daily_events.parquet"
final.to_parquet(out_path, index=False)
print(f"  Wrote {out_path}  ({len(final):,} rows, {len(final_cols)} cols)")

print()
print("  NOTE: zero-coverage years mean the chain's start date is after the")
print("  event's date, not that the feature failed. Events before 2012-01-01")
print("  plus the ~2-week spin-up are NaN by construction.")
print()
print("NEXT: download qc/fwi_daily_events.parquet. It joins on `row` like the")
print("      other P7 blocks, so add it to the local merge and re-train.")
print("=" * 74)