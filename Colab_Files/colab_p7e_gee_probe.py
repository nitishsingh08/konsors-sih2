# ============================================================================
# P7e -- COLAB CELL: VERIFY the Earth Engine route before building on it
#
# WHY VERIFY FIRST
#
# Two things are established beyond doubt:
#
#   1. The weak `INDUSTRIAL_FIRE` label is two DISTANCE terms and nothing else.
#      `qc/p6_rule_calibration.json` shows `industrial_persistence = 1/30` with
#      `n_above == n_total` -- the threshold is the FLOOR of
#      `observed_days / 30`, so the `shape` conjunct excludes nothing. Cause:
#      catalogue-scale targets (industrial 400,000) applied to P6's 200,000-row
#      calibration sample. Fixed in `rule_calibration.scale_targets`.
#
#   2. `FRP_PERSISTENCE` and `DETECTION_COUNT_LOG` are label-rule inputs, so
#      `training_design` excludes them from model B. Even a NON-VACUOUS
#      persistence threshold cannot help model B: the evidence would still come
#      from columns the scientific model is forbidden to see. Fixing the
#      calibration makes the label BETTER; it cannot make it INDEPENDENT.
#
# For model B to have signal, the label must be built from evidence absent from
# X entirely -- surface damage and fuel type:
#
#     LABEL  <- NBR drop + no recovery + non-vegetated + persistent thermal
#     MODEL  <- VIIRS thermal and temporal features already in X
#
# The two share no input. That is what makes model B's number mean something.
#
# WHAT THIS CELL DOES -- verification only, ~3 minutes
#
#   1. Authenticate, including resolving the CLOUD PROJECT id that
#      `ee.Initialize()` now demands and cannot infer.
#   2. Confirm each dataset loads and has the bands named.
#   3. Sample REAL values for REAL gold-set events, so we know the pixels carry
#      information rather than zeros.
#   4. Time one 5,000-point tile to extrapolate the cost of a full extraction.
#
# Nothing is written except one small JSON probe record under qc/.
#
# NO SystemExit ANYWHERE IN THIS CELL. Raising it inside a notebook cell trips
# an IPython/`inspect` bug on Python 3.13 that buries the real error under three
# chained tracebacks, the last of which is a `TypeError` in IPython's own
# pretty-printer. Every exit path here prints and guards instead.
# ============================================================================

import json
import time

import numpy as np
import pandas as pd

import colab_utils as cu
import gee_utils as gee

#: Bumped whenever this cell's logic changes. Printed on the FIRST line so a
#: stale copy in Colab is obvious at a glance instead of being inferred from a
#: downstream failure. This project has been bitten by stale cells three times
#: (README section 5 defects 3 and 5, and the "partially uploaded picker"
#: incident in section 6), always with the same shape: an old file on Drive
#: keeps working while silently producing old behaviour.
P7E_CELL_VERSION = "2026-10-04.4"

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError(f"no project root under {cu.MYDRIVE}")
QC_DIR = PROJECT_ROOT / "qc"
QC_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 74)
print(f"P7e VERIFY GEE  [cell version {P7E_CELL_VERSION}]")
print("=" * 74)
print(cu.drive_report())
print(f"  gee_utils: {getattr(gee, '__version__', 'unknown')}")
print("  If the version above does not match the repo, this cell is STALE --")
print("  re-upload the modules through colab_bootstrap.py and restart.")

# ---------------------------------------------------------------------------
# 1. AUTHENTICATION
#
# The blocker is NOT credentials. `ee.Initialize()` reports
# "no project found" when it cannot infer which Cloud project to bill, which is
# a different problem and a different fix.
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("1. AUTHENTICATION")
print("=" * 74)

print("  project ids visible to this runtime:", gee.candidate_project_ids() or
      "NONE -- you must supply one")

_ok = False
_pid = None
_error = None
_perm = None
try:
    _ok, _pid = gee.authenticate(
        # A notebook variable wins, so a cell above is enough; otherwise
        # gee_utils falls back to a local git-ignored `.env`.
        globals().get("EARTHENGINE_PROJECT")
    )
except gee.EEAuthError as exc:
    _error = exc.cause

# `Initialize` succeeding is NOT a working session. It accepts credentials
# without checking the caller may USE the project, so a project owned by a
# different account -- exactly what happens after an account switch -- sails
# through here and then 403s on every data call. Prove a real read works.
if _ok:
    try:
        gee.verify_can_read(_pid)
        print(f"  Earth Engine working. project = {_pid!r} (data reads verified)")
    except gee.EEPermissionError as exc:
        _perm = exc
        print(f"  Earth Engine INITIALISED BUT NOT USABLE with project {_pid!r}.")
        print()
        print(exc.REMEDIATION)
        print()
        _vis = gee.projects_visible_to_current_account()
        print(f"  projects this account can currently use: {_vis or 'NONE FOUND'}")
        if _vis:
            print(f"  -> try one of these instead of {_pid!r}")
        else:
            print("  -> the account in use has no Earth Engine project yet.")
            print("     Create one, then re-run. See (a) above.")
    except Exception as exc:                       # noqa: BLE001
        print(f"  data-read probe failed unexpectedly: "
              f"{type(exc).__name__}: {str(exc)[:200]}")

if not _ok or _perm is not None:
    print()
    if _perm is not None:
        print("  STOPPED: the project initialised but REFUSED a data read.")
        print("  See the permission remediation printed above. Nothing was written.")
    else:
        print("  NOT INITIALISED.")
        print()
        print(gee.EEAuthError.REMEDIATION)
        print()
        print("TO SUPPLY THE PROJECT ID (no code edit needed)")
        print("-" * 74)
        print("  Option A -- a cell above P7e:")
        print("      EARTHENGINE_PROJECT = 'your-project-id'")
        print("  Option B -- a local `.env` in the project root. It is already in")
        print("  `.gitignore`, so the id is never committed:")
        print('      echo "EARTHENGINE_PROJECT=your-project-id" >> .env')
        print("      # and upload it alongside the modules")
        print()
        if _error:
            print("  what was tried:")
            print(f"    {_error}")
        print()
        print("  The single most likely cause, in order:")
        print("    a) no Cloud project was ever created for this account")
        print("    b) the project exists but the Earth Engine API is not enabled")
        print("    c) the project is not registered with Earth Engine")
        print()
        print("  If ee.Authenticate() cannot open a consent screen inside Colab,")
        print("  authenticate from a local terminal instead:")
        print("      pip install earthengine-api")
        print("      earthengine authenticate   # prints a URL; open it locally")
        print("  then run P7e with EARTHENGINE_PROJECT set.")
    print()
    print("=" * 74)
    print("P7e STOPPED before verifying anything. Nothing was written.")
    print("=" * 74)
else:
    # -----------------------------------------------------------------------
    # 2. DATASETS AND BANDS
    # -----------------------------------------------------------------------
    print()
    print("=" * 74)
    print("2. DATASETS AND BANDS")
    print("=" * 74)
    print("  purpose of each:")
    print("    worldcover     fuel type at the centroid -- the single most")
    print("                   useful new signal for separating industrial")
    print("                   surface loss from stubble burning")
    print("    s2_sr          NBR needs SWIR (B12) and NIR (B8A) to detect")
    print("                   surface loss and whether it recovered")
    print("    landsat_nbr    30 m NBR for the days Sentinel-2 is clouded")
    print("    modis          independent land cover, to cross-check WorldCover")
    print()
    _ds = gee.verify_datasets()
    for _name, _info in _ds.items():
        print(f"  {_name:<16} {_info['image_id']}  ({_info['kind']})")
        if _info.get("error"):
            print(f"    LOAD FAILED: {_info['error']}")
            if "is not an Image" in _info["error"]:
                print("    -> this asset is an ImageCollection and must be built")
                print("       with ee.ImageCollection(...).first()")
            continue
        # `_info['wanted']` is present on EVERY branch now, but `.get()` keeps
        # the reporting code from being the thing that crashes.
        _w = _info.get("wanted", [])
        if _info.get("ok"):
            print(f"    ok -- {len(_w)} wanted band(s) present "
                  f"({', '.join(_w)}), {_info.get('n_bands', 0)} total")
        else:
            print(f"    MISSING BANDS {_info.get('missing')} -- available "
                  f"{_info.get('n_bands', 0)}")
            print("    -> substitute or drop. Do not assume band names hold.")
        if _name == "worldcover":
            print(f"    note: WorldCover's land cover band is "
                  f"{gee.WORLDCOVER_BAND!r}, not 'cover'.")

    # -----------------------------------------------------------------------
    # 3. REAL VALUES FOR REAL EVENTS
    #
    # The check that matters. A dataset can list its bands and still return
    # zeros over the AOI -- that is how `sentinel5p` (444 B) and `era5_land`
    # (120 KB/month) got onto Drive unnoticed.
    # -----------------------------------------------------------------------
    print()
    print("=" * 74)
    print("3. REAL VALUES FOR REAL EVENTS")
    print("=" * 74)
    import ee

    _gold_csv = QC_DIR / "gold_set_sample.csv"
    if not cu.path_ok(_gold_csv):
        print(f"  {_gold_csv} not found. Run colab_p7c_goldset.py first.")
        print("  P7e needs real coordinates to sample.")
    else:
        _gold = pd.read_csv(_gold_csv)
        print(f"  gold sample: {len(_gold)} rows")
        _probe = _gold[_gold["event_centroid_lon"].notna()
                       & _gold["event_centroid_lat"].notna()].head(8)
        if _probe.empty:
            print("  no coordinates in the gold sample")
        else:
            print(f"  probing {len(_probe)} locations:")
            for _, _r in _probe.iterrows():
                _st = str(_r.get("start_time", ""))[:10]
                print(f"    {_r['gold_id']}  {_r['event_centroid_lat']:.4f},"
                      f"{_r['event_centroid_lon']:.4f}  {_st}  "
                      f"{_r['nearest_facility_m']:>10,.0f} m to facility")

            _pts = ee.FeatureCollection([
                ee.Feature(ee.Geometry.Point(
                    [float(_r["event_centroid_lon"]),
                     float(_r["event_centroid_lat"])]),
                    {"gold_id": str(_r["gold_id"])})
                for _, _r in _probe.iterrows()
            ])

            _wc_vals = {}
            try:
                # MUST be ImageCollection(...).first(). The catalog snippet is
                # exactly that: ee.Image("ESA/WorldCover/v200") raises
                # `Asset ... is not an Image`.
                _wc = gee.worldcover_image()
                _res = _wc.sampleRegions(collection=_pts,
                                         properties=["gold_id"], scale=100,
                                         geometries=True).getInfo()
                for _f in _res.get("features", []):
                    _wc_vals[_f["properties"]["gold_id"]] = \
                        _f["properties"].get(gee.WORLDCOVER_BAND)
                print()
                print(f"  ESA/WorldCover class ({gee.WORLDCOVER_BAND!r}) "
                      "at each point")
                for _c, _nm in sorted(gee.WORLDCOVER_CLASSES.items()):
                    print(f"    {_c:>3}  {_nm}")
                print()
                for _gid in sorted(_wc_vals):
                    _v = _wc_vals[_gid]
                    _nm = gee.WORLDCOVER_CLASSES.get(
                        int(_v), "UNMAPPED") if _v is not None else "null"
                    _veg = (int(_v) in gee.VEGETATED_CLASSES
                            if _v is not None else None)
                    print(f"    {_gid}  class={_v}  {_nm}"
                          + (f"  [{'vegetated' if _veg else 'NON-vegetated'}]"
                             if _veg is not None else ""))
                _nn = sum(1 for _v in _wc_vals.values() if _v is not None)
                print()
                print(f"  non-null {_nn}/{len(_wc_vals)}"
                      + ("  -- OK" if _nn == len(_wc_vals)
                         else "  -- INVESTIGATE before trusting this raster"))
                # The whole point of the column, stated as a number.
                _mapped = [_v for _v in _wc_vals.values() if _v is not None]
                if _mapped:
                    _nv = sum(1 for _v in _mapped
                              if int(_v) in gee.VEGETATED_CLASSES)
                    print(f"  vegetation substrate: "
                          f"{100.0 * _nv / len(_mapped):.0f}% of "
                          f"{len(_mapped)} mapped points")
            except Exception as exc:                # noqa: BLE001
                print(f"  WorldCover sampleRegions FAILED: "
                      f"{type(exc).__name__}: {str(exc)[:300]}")

            # NBR around the first event -- the expensive probe, so one only.
            _r0 = _probe.iloc[0]
            _start = str(_r0.get("start_time", ""))[:10]
            try:
                _date = ee.Date(_start)
                _swir, _nir = gee.NBR_BANDS["sentinel2"]
                # MUST be an ImageCollection. `ee.Image("COPERNICUS/S2_HARMONIZED")`
                # does not raise -- it builds a composite Image with no
                # filterDate, so the failure surfaces as
                # `AttributeError: 'Image' object has no attribute 'filterDate'`
                # a long way from the actual mistake.
                _win = (ee.ImageCollection(gee.S2_SR)
                        .filterDate(_date.advance(-60, "day"),
                                    _date.advance(60, "day")))
                _n = _win.size().getInfo()
                print()
                print(f"  S2 images within +/-60 d of {_start}: {_n}")
                if _n == 0:
                    print("    -> era window or AOI problem. Report it.")
                else:
                    _nbr = _win.normalizedDifference([_swir, _nir])
                    _geom = _pts.filter(
                        ee.Filter.eq("gold_id", str(_r0["gold_id"]))).geometry()
                    _stats = _nbr.reduceRegion(
                        reducer=ee.Reducer.mean(), geometry=_geom,
                        scale=100, bestEffort=True).getInfo()
                    print(f"    mean NBR (100 m window) at "
                          f"{str(_r0['gold_id'])}: {_stats.get('NBR')}")
                    print(f"    NBR = ({_swir} - {_nir}) / ({_swir} + {_nir})"
                          "  <- order matters, see gee_utils.NBR_BANDS")
                    print("    a burn scar reads strongly negative; ~0 is bare or")
                    print("    built; >0.2 is vegetated. All-zero or all-NaN means")
                    print("    the band is not returning data over this AOI.")
            except Exception as exc:                # noqa: BLE001
                print(f"  S2 NBR probe FAILED: {type(exc).__name__}: "
                      f"{str(exc)[:300]}")

        # -------------------------------------------------------------------
        # 3b. IMAGERY COVERAGE OF THE SAMPLE
        #
        # Sentinel-2 L2A starts 2017 and Landsat 8 starts 2013, so with only
        # those configured the early catalogue years have NO imagery. The gold
        # set contains 2012 events, which is not hypothetical.
        # -------------------------------------------------------------------
        if not _probe.empty and "start_time" in _probe.columns:
            _years = pd.to_datetime(_probe["start_time"],
                                    errors="coerce").dt.year.dropna()
            _gap = gee.imagery_gap(_years)
            print()
            print("=" * 74)
            print("3b. IMAGERY COVERAGE")
            print("=" * 74)
            print(f"  sample years           : "
                  f"{sorted(int(y) for y in _years)}")
            print(f"  Sentinel-2 L2A from    : {gee.S2_SR_FIRST_YEAR}")
            print(f"  Landsat 8 from         : {gee.LANDSAT8_FIRST_YEAR}")
            print(f"  Landsat 5 / 7 from     : 1999  (not yet checked here)")
            if _gap["no_landsat"]:
                print(f"  NO Landsat coverage    : years {_gap['no_landsat']}")
                print("     Landsat 5/7 are the only option for these. They must")
                print("     be added or those rows cannot be image-labelled.")
            if _gap["no_sentinel2"]:
                print(f"  no Sentinel-2 coverage : years {_gap['no_sentinel2']}")
                print("     expected; Landsat is the backstop.")
            if not _gap["no_landsat"]:
                print("  every sample year has Landsat 8 coverage")

        # -------------------------------------------------------------------
        # 4. FEASIBILITY
        #
        # One giant `sampleRegions` over 7.27M points will be refused or will
        # time out. Tiled `reduceRegions` is the reliable pattern; here we time
        # ONE tile and extrapolate, so the scope is justified rather than
        # guessed.
        # -------------------------------------------------------------------
        print()
        print("=" * 74)
        print("4. FEASIBILITY -- how many Earth Engine calls?")
        print("=" * 74)
        TILE = 5_000
        _src = cu.find_artefact_path(PROJECT_ROOT,
                                     "canonical_event_base.parquet")
        if _src is None:
            print("  canonical_event_base.parquet not found; cannot time a tile.")
            print(cu.describe_missing_artefact(
                PROJECT_ROOT, "canonical_event_base.parquet"))
            print("  Pass the location explicitly and re-run to get the number.")
        else:
            try:
                _meta = pd.read_parquet(_src, columns=["event_centroid_lon",
                                                       "event_centroid_lat"])
                print(f"  event base rows: {len(_meta):,}  from {_src}")
                _sub = _meta.sample(TILE, random_state=0)
                _fc = ee.FeatureCollection([
                    ee.Feature(ee.Geometry.Point([float(_a), float(_b)]))
                    for _a, _b in zip(_sub["event_centroid_lon"],
                                      _sub["event_centroid_lat"])])
                _img = (ee.Image(gee.WORLDCOVER).select("cover")
                        .addBands(ee.Image(gee.S2_SR).select("B8A")
                                  .mean().multiply(0.0001)))
                _t1 = time.time()
                _img.reduceRegions(collection=_fc,
                                   reducer=ee.Reducer.mean(),
                                   scale=100).getInfo()
                _dt = time.time() - _t1
                _tiles = int(np.ceil(len(_meta) / TILE))
                _serial_h = _dt * _tiles / 3600.0
                print()
                print(f"  ONE tile of {TILE:,} points: {_dt:.1f}s")
                print(f"  -> {len(_meta):,} points needs ~{_tiles:,} tiles")
                print(f"  -> SERIAL:        {_serial_h:.1f} hours")
                print(f"  -> 8 workers:     {_serial_h / 8:.1f} hours")
                print()
                print("  VERDICT: "
                      + ("feasible within a Colab session."
                         if _serial_h / 8 < 12 else
                         "TOO EXPENSIVE for all 7.27M events."))
                print()
                print("  If it is expensive, the right scope is a stratified")
                print("  SUBSAMPLE, not the full catalogue. A 400k-event")
                print("  stratified sample costs 1/18 of the full pass and is")
                print("  ample for training AND for validating a rule.")
                print("  Land cover is STATIC, so one sample per location")
                print("  suffices -- there is no per-date cost to amortise.")
            except Exception as exc:                # noqa: BLE001
                print(f"  tile timing FAILED: {type(exc).__name__}: "
                      f"{str(exc)[:300]}")
                print("  Without this number the extraction scope is a guess.")

    _record = {
        "authenticated": bool(_ok),
        "project_id": _pid,
        "datasets": _ds if _ok else {},
    }
    (QC_DIR / "p7e_gee_probe.json").write_text(
        json.dumps(_record, indent=2, default=str))
    print()
    print(f"Wrote {QC_DIR / 'p7e_gee_probe.json'}")
    print("=" * 74)
    print("END -- paste the whole output above.")
    print("=" * 74)