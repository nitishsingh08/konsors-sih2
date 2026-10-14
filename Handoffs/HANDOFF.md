# HANDOFF — Fire-Event Industrial Classification Pipeline

Generated 2026-10-02. Read this first in any new session.

`README.md` is the standing project-status and change log; keep it updated
whenever stage or status changes.

---

## 0. THE TASK

Build an AI-enabled geospatial system that classifies **industrial fires and
persistent thermal sources** from NASA FIRMS VIIRS data, per the project
statement. Two deliverables:

- **i.** Segregate industrial fires (refineries, petrochemical, power plants,
  steel, cement, LNG, mining) from forest/agricultural/natural fires
- **ii.** GIS solution — storage plus visualization as a map overlay

Domain context: thermal data + land cover + industrial databases + satellite
imagery, integrated into one classifier.

---

## 1. RESEARCH OBJECTIVE AND CURRENT STAGE

The unit of analysis is the **global fire event**, not the raw detection.

```
15,392,720 VIIRS detections (SNPP / NOAA-20 / NOAA-21)
        ↓  daily cKDTree 1 km + 12 h, connected components
7,589,305 local clusters
        ↓  cross-day cluster edges (323,804)
7,265,849 global fire events
        ↓  141 event-level features        ← WE ARE HERE
X_event
        ↓  independent labelling
y_event (INDUSTRIAL_FLARE / INDUSTRIAL_FIRE / MINING /
         AGRICULTURAL / WILDFIRE / UNKNOWN)
        ↓
LightGBM → GIS layer
```

**Backend:** FastAPI model serving added (`model_registry.py`, `POST /api/predict`,
`GET /api/models[/{name}]`, 22 tests). Default model B, since A's 0.9963 is circular.
Startup no longer depends on the database: a bad `DATABASE_URL` degrades instead of
killing inference. Local dev uses the SQLite DB in `backend/`.

**Static cell checks now cover 6 driver cells including P6 and P7.** They catch undeclared accumulators, undefined free names, and use-before-assignment -- the last found a latent `NameError` in P3a that would have crashed the next run.

**P7 first run failed twice, both my fault:** exhausted RAM by loading 7.27M x 60 as float64 into LightGBM (now float32 + stratified 1.5M sample), and `assert_no_time_leakage` rejected the *combined* split, whose guarantee is spatial by design. Both fixed.

**Stage:** `P1 CLOSED`, `P1b CLOSED`, `P3a DATA WRITTEN (18)`,
`P4a DATA WRITTEN (34 tier-1)`, `P5/P5b LABELS COMPLETE`,
`P6 ASSEMBLY COMPLETE (7,265,849 x 56)`, `P0 GAP ANALYSIS COMPLETE`,
`P7 RETARGETED AT label_weak`. Ledger 1.3.0, Gate P1 passes, 497 local tests (13 of them rule calibration).

**Weak labels now populate:** INDUSTRIAL_FIRE 1,119,254 (15.40%) · WILDFIRE
178,581 · MINING 73,060 · INDUSTRIAL_FLARE 3,571 · UNKNOWN 81.08%. AGRICULTURAL
is still 0 because it needs `FRP_PERSISTENCE < 0.5` and **P4a must be re-run**
to rewrite the tier-1 chunks -- P6 only re-reads them.

**Label rebalance after the P4a re-run:** INDUSTRIAL_FLARE 3,571 -> 0 and MINING
73,060 -> 61, AGRICULTURAL 0 -> 3,939,494. `FRP_PERSISTENCE` changed SCALE
(constant 1.0 -> observed_days/30), so the old thresholds became too strict.
**Resolved:** P6 PHASE 3 now derives every persistence threshold from the
catalogue via `rule_calibration.py`, targeting flare 3,600 / industrial 400k /
mining 60k. If the distribution moves, re-run P6 -- never hand-edit thresholds.

**Two standing caveats.** `label_weak` is manufactured from features the model
also sees, so PR-AUC on it is an upper bound, not generalisation; and the strict
`label` column remains ~100% UNKNOWN pending land cover and weather.

**The operating threshold is now implemented (2026-10-04).** §8's policy —
"freeze tau maximising F2 subject to precision >= 0.90" — had never been written.
`REVIEW_CONFIDENCE = 0.60` was a bare literal applied to models that are not
calibrated to a 0.60 prior, and model B's held-out scores sit near 0.13, so the
flat cut flagged **0.8% of positives**. That was the "detects 2 of 100"
symptom. `threshold_calibration.py` now freezes tau on the TRAIN fold only and
returns `threshold: None` with a reason when the constraint is unreachable.
Measured: model B's only precision-0.90 point is tau=0.875 at 0.04% recall;
model A's tau=0.033 flags 16.5% of rows against a 16.39% prevalence, which is
the circularity of §15 shown numerically. **This does not make either model
better.** It makes the deployed bar honest. The performance gap is a feature
gap — see below.

**Measured coverage: 33 usable / 20 sparse / 88 missing = 23.4% of the 141
spec.** The 88 split 18 DERIVE / 46 EXTERNAL_RASTER / 24 EXTERNAL_TABLE; 6 of
the 18 DERIVE features are permanently unavailable, leaving **12** realisable
with no download. Model B has **29** features (18 industrial + 16 weak-rule
inputs excluded).

**Three defects the Phase-0 run exposed, all now fixed:**
1. `FRP_PERSISTENCE` was exactly **1.0 for all 7,265,849 events** -- the
   definition was a tautology. It silently zeroed the AGRICULTURAL rule.
   Redefined over a fixed 30-day window.
2. `INSIDE_FACILITY == 1` for **zero** events (centroid vs small OSM polygons).
   It was a required conjunct of the flare rule, making it unsatisfiable.
   Dropped; distance-to-refinery is the robust proxy.
3. `SITE_DEG = 0.1` was misread as too fine; it is too *local*. Now **1.0 deg**.

**The 100% UNKNOWN was not unsatisfiable rules:** measured on the real data,
`INDUSTRIAL_FIRE` passes 1,122,726 (15.5%) and `MINING` 147,962 (2.0%). Both
were ready; P6 had run a stale pasted cell, so `label_weak` never existed. After
the fix P6 produced 1,119,254 and 73,060 respectively -- the small shortfall is
rule precedence, not error.
**84 of 141 features are implemented** (12 computed + 72 new). 40 remain: 23
need an Earth Engine run, 16 need `sentinel5p` verified, 1 lightning. The Drive
turned out to be a **stale mount**, not a wrong account — the probe found
`external/` and the event base immediately. Full training path in README §5.

---

## 2. VALIDATED BASELINE — DO NOT CHANGE

```
Source                /content/drive/MyDrive/India_Firms/India_Firms.parquet
Rows                  15,392,720  ×  22 columns
Date range            2012-01-20 → 2026-08-31
Observed dates        5,337  (5,338 calendar; 2012-03-25 missing)
Spatial threshold     1,000 m   (EPSG:6933, Equal Earth)
Temporal threshold    12 hours
Algorithm             cKDTree → temporal filter → connected_components
Local cluster ID      YYYYMMDD_<component_label>
```

Frozen regression anchors — if any of these change, stop:

| Experiment | local clusters | cross-day edges | global events |
|---|---|---|---|
2021-03-31 | 12,641 | — | — |
2021-03-31→04-01 | 25,542 | 2,066 | 23,476 |
7-day | 77,244 | 8,738 | 68,514 |
March 2021 | 230,239 | 21,379 | 208,884 |
2-month | 366,067 | 33,490 | 332,607 |
Full 2021 | 700,146 | 40,916 | 659,280 |

March 31 detail: 37,801 detections / 89,322 spatial pairs / 75,626 valid
pairs / 12,641 clusters. Cross-day: 15,887 detection links → 2,066 cluster
links.

Canonical artifacts (built, verified — **do not rebuild**):

```
/content/drive/MyDrive/Firm_Clusters/full_2012_2026/
    clusters/                          5,337 daily parquet files
    edges/                             cross-day edge files
    events/canonical_event_base.parquet      7,265,849 rows, ~0.376 GB
    events/event_local_cluster_membership.parquet
    qc/10_event_cluster_forensic_subset.parquet   10 real events
    external/                          1.50 GB, 911 files (see §5)
```

`build_full_global_events()` must never be re-run.

---

## 3. THE 141-FEATURE LEDGER (P1 — COMPLETE)

### Files in repo

| File | Purpose |
|---|---|
`feature_ledger.py` | **Single source of truth.** 141 features in authoritative order, group map, status map, dataset map, local-input hints, era windows, drift check. `LEDGER_VERSION = "1.2.0"` |
`build_feature_audit.py` | Emits ledger CSV/JSON + P3 work list locally; asserts Gate P1 |
`test_feature_ledger.py` | **39 tests, all passing** |
`colab_p1_ledger.py` | Colab cell: 4-tier module loader, version guard, drift check, writes ledger to Drive |
`colab_p1b_external_inventory.py` | Colab cell: walks `external/`, size-sanity QC, matches to features |
`colab_p0_cell262.py` | Colab cell: recovers historical `extract_fire_features` from QC JSONs |
`industrial_features.py` | **New at P3a.** Pure computation core for the 18 INDUSTRIAL features — no I/O, no Drive paths |
`colab_p3a_industrial.py` | Colab cell: maps the 37 `osm_facilities` files onto the 10 ledger categories, projects to EPSG:6933, streams all 7.26M events |
`test_industrial_features.py` | **48 tests** on synthetic geometry |
`qc/` | `feature_audit_141.csv`, `feature_status_summary.json`, `FEATURES_141.json`, `missing_features_by_phase.md` |

### The 141 features, 11 groups

| # | Group | n | Source dataset |
|---|---|---|---|
1 | THERMAL | 15 | VNP14A1 + MOD14A1 + India_Firms |
2 | TEMPORAL | 14 | India_Firms (via detection lineage) |
3 | SPREAD | 17 | India_Firms |
4 | TERRAIN | 12 | SRTM / NASADEM / Copernicus DEM |
5 | LANDCOVER_FUEL | 16 | WorldCover + MCD12Q1 + MOD13Q1 |
6 | INDUSTRIAL | 18 | GEM + OSM ← **leakage partition** |
7 | WEATHER_FIRE_DANGER | 18 | ERA5-Land + CHIRPS + SoilGrids |
8 | OPTICAL_SMOKE | 7 | S5P AERONOMI + S2 |
9 | SAR_STRUCTURAL | 7 | Sentinel-1 GRD |
10 | ATMOSPHERIC_CHEMISTRY | 9 | S5P L3 |
11 | QUALITY_UNCERTAINTY | 8 | India_Firms + MOD14 QA |

**The list is exactly 141** (land cover is 16, not 15). The "131 feature"
figure in older handover docs was an arithmetic error — retired, and
`check_schema_drift` now rejects a 131-length list.

### Status vocabulary (8 values)

```
COMPUTED_HISTORICAL       12   TERRAIN, cell 263, recovered intact
NEW_IMPLEMENTATION        72   P3a INDUSTRIAL 18 + P4a lineage 54
UNVERIFIED_HISTORICAL     10   cell 264 weather (verified code, unrun)
HISTORICAL_NAN            7   FFMC DMC DC ISI BUI FWI FWI_ANOMALY_Z
MISSING_IMPLEMENTATION    40   23 need a GEE run, 16 need sentinel5p,
                                 4 recurrence (cross-event pass), 1 lightning
MISSING_INPUT / EXTRACTION_ERROR / RETIRED   0   (filled at P2/P4)
```

**The 33 former `UNVERIFIED_HISTORICAL` lineage claims were superseded at P4a.**
They were reported as historical on group inference from cell 262/325 and had
never been observed to run. They are now written-down, unit-tested definitions
in `lineage_features.py`. A test asserts none of the four lineage groups can
drift back to `UNVERIFIED_HISTORICAL`.

**`UNVERIFIED_HISTORICAL` was added deliberately.** Those 43 were previously
reported as "computed" on group inference alone — an overclaim. `runtime_verified`
is an orthogonal boolean, default `False`, asserted false by the gate. Only
P2 clears it.

**`NEW_IMPLEMENTATION` was added at P3a, for the same reason.**
`extract_industrial_features` never existed in any notebook, so the 18
INDUSTRIAL features have code but no history. They are deliberately *not*
`COMPUTED_HISTORICAL`. `STILL_UNIMPLEMENTED` in the ledger is the tuple of
statuses that still mean "no code exists"; a new implementation is not in it,
even though it still needs a runtime pass.

### Gate P1 output (passed, re-run after P3a)

```
rows 141 · unique 141 · order clean · UNSET 0
THERMAL 15 (UNVERIFIED=12, MISSING=3)
TEMPORAL 14 (UNVERIFIED=10, MISSING=4)
SPREAD 17 (UNVERIFIED=11, MISSING=6)
TERRAIN 12 (COMPUTED=12)
LANDCOVER_FUEL 16 (MISSING=16)
INDUSTRIAL 18 (NEW_IMPLEMENTATION=18)
WEATHER 18 (UNVERIFIED=10, HISTORICAL_NAN=7, MISSING=1)
OPTICAL_SMOKE 7 · SAR 7 · ATMOSPHERIC 9 · QUALITY 8   (all MISSING)
LABEL_DERIVED 18 · NON_LEAK 123
111 of 141 still unimplemented · 0 runtime_verified
```

Written to Drive: `qc/feature_audit_141.parquet`, `.csv`, `FEATURES_141.json`

### Open work by phase

```
P0  recover cell 262                       15
P2  verify via 10-event test               49
P3a event shape/geometry only                6
P3b quality (from India_Firms)              8
P3c land cover 16
P3d Canadian FWI 7
P3e SAR 7
P3f smoke + atmospheric + lightning       17
P3g recurrence / inter-event gap           4
```

---

## 4. HISTORICAL EXTRACTOR FORENSICS (essential reading)

The 141-feature extractor exists **only in a Colab session and in Drive QC
JSONs** — it is in none of `Stage1–Stage6.ipynb` in this repo. Stage5 is the
production-run notebook. Do not search the repo for it.

Seven historical definitions of `extract_event_features` were found. **Cell 314
is the desired dispatcher** — the only one calling all ten feature groups.

Compatible dependency set:

| Function | Cell | Signature | Notes |
|---|---|---|---|
`event_geometry` | 260 | `(event_clusters)` | uses `centroid_lon`/`centroid_lat` — works with the cluster subset |
`extract_event_derived_features` | **325** | `(event, event_clusters)` | 23 features, local `safe_cv()`. **Must be 325, not 372/387** (3-arg → `TypeError` under cell 314) |
`extract_fire_features` | 262 | `(event, geometry)` | **INCOMPLETE — returns `None`. P0 open** |
`extract_terrain_features` | 263 | | only implementation, complete |
`extract_weather_features` | 264 | | verified. **Never use cell 310** (stub) |

### Six absent functions

```
extract_landcover_features      extract_sar_features
extract_industrial_features     extract_atmospheric_features
extract_smoke_features          extract_quality_features
```

Covering 65 features. Until implemented → NaN stubs + `MISSING_IMPLEMENTATION`.
`extract_industrial_features` is no longer in this list: it was written at P3a
(see below). 52 features remain unimplemented.

### P3a — industrial context (code complete, NOT yet run)

`extract_industrial_features` never existed, so this is new code. Input is the
one external folder P1b cleared: 37 geoparquets, 0.28 GB, under
`external/osm_facilities/`.

```
industrial_features.py      pure core, no I/O, 48 tests on synthetic geometry
colab_p3a_industrial.py     Drive driver
```

Computation: 10 per-category `DIST_*_LOG` (cKDTree, `log10(1+d_m)`) + 1 union +
3 `FACILITY_COUNT_{1,5,10}KM` + `INDUSTRIAL_DENSITY_CONTRAST` (10 km local vs a
30 km ring at 16 bearings) + `INSIDE_FACILITY` (STRtree) +
`DIST_FACILITY_BOUNDARY_LOG` (rings sampled every 50 m, error ≤ 25 m outside) +
`KNOWN_FLARE`.

Decisions that must not drift:

- **`crs` is a required argument**, validated to be EPSG:6933. A degree-valued
  distance is wrong in a way nothing downstream would catch.
- **Missing category → NaN, never 0.** "No mines downloaded" ≠ "no mine within
  10 km". The 3 count features do return 0 for empty space.
- **`KNOWN_FLARE`** needs all five of: inside a flare-capable facility
  (refinery/petrochemical/lng/gas_processing), ≥200 active days, detected on
  ≥25% of them, ≤2 km centroid drift, FRP CV ≤0.60. Missing evidence → NaN.
- **`INSPECT_ONLY = True` by default.** The 37 files are named after OSM/GEM
  categories, and `industrial.geoparquet` likely *contains* cement/steel/lng
  members, so the mapping must be eyeballed once before computing. Nothing is
  written in inspection mode.
- Streams `canonical_event_base.parquet` with `iter_batches(200_000)`;
  `MAX_EVENTS` exists for a smoke test. Never loads 7.26M rows.

`runtime_verified` stays `False` for all 18 until P2/P4 executes them.

### FWI forensic finding

`FFMC, DMC, DC, ISI, BUI, FWI, FWI_ANOMALY_Z` — cell 264 hardcodes all seven to
`np.nan` in **both** branches. No downstream code overwrites them. Legitimate
schema members, NaN placeholders. Implement standard Canadian FWI sequential
equations (Van Wagner 1987) and label as `NEW_IMPLEMENTATION`.

### BURN_SCAR_GROWTH_RATE forensic finding

Schema-only. No burn-scar mask, no pre/post NBR, no area/perimeter growth code
exists anywhere. **Never infer it from FRP.** Same applies to
`AREA_GROWTH_RATE, PERIMETER_GROWTH_RATE, ELONGATION, COMPACTNESS, SOLIDITY,
NEW_PIXEL_FRACTION, NBR_CHANGE, DNBR, DNDVI`.

### Restored GEE dependencies (verified)

```python
MODIS_FIRE = ee.ImageCollection("MODIS/061/MOD14A1")
VIIRS_FIRE = ee.ImageCollection("NASA/VIIRS/002/VNP14A1")

def ee_timestamp(v):
    if pd.isna(v): raise ValueError("Missing datetime value")
    return pd.Timestamp(v).strftime("%Y-%m-%dT%H:%M:%S")
```
The original `ee.Date(str(event["start_time"]))` throws
`EEException: Date: Bad date/time`. Use `ee_timestamp()`.

`EVENT_00000002` verified: 1 VIIRS image, 1 MODIS image, geometry valid.

### Recurrence features

`RECURRENCE_RATE_1Y/3Y/5Y, MAX_INTER_EVENT_GAP_LOG` need an event-index pass
over all 7.26M events. Cell 325 has no `all_events`; cells 372/387 do but are
signature-incompatible with cell 314. → phase P3g.

---

## 5. EXTERNAL DATA — VERIFIED INVENTORY

`/content/drive/MyDrive/Firm_Clusters/full_2012_2026/external`
**911 files, 213 subdirs, 1.50 GB**

| Folder | Files | Size | Notes |
|---|---|---|---|
`VIIRS_FOOTPRINTS/` | 12 | **1.08 GB** | yearly `.geoparquet` 2012–2026 + combined v3 |
`osm_facilities/` | 37 | **0.28 GB** | `facility_categories/*.geoparquet` — cement, lng, mines, pipelines + 30 more |
`DL_FIRE_M-C61_814501/` | 3 | 0.09 GB | FIRMS archive + NRT + readme |
`era5_land/` | 175 | 0.02 GB | monthly `.nc` — **~120 KB each, suspect** |
`india_mopitt_gfas_output/` | 177 | 0.02 GB | monthly parquet, GFAS wind |
`sentinel5p/` | 506 | **0.00 GB** | monthly `.tif` — **~5 KB each, almost certainly empty** |
`gee_features/` | 1 | — | `logs/batch_manifest.csv` only |

Extensions: 34 `.geoparquet` (1.18 GB), 1 `.geojson`, 2 `.pbf`, 4 `.csv`,
176 `.parquet`, 175 `.nc`, 505 `.tif`, 8 `.json`, 5 `.txt`, 1 `.jsonl`.

### Two red flags

`sentinel5p` — 506 rasters at 0.00 GB is ~5 KB each. Real S5P monthly
composites over India are tens of MB. Almost certainly empty/all-NaN/wrong AOI.
**The 9 atmospheric features must not be planned around until verified.**
Layout is per-gas subfolders (`sentinel5p/ch4/S5P_CH4_2018_04.tif`).

`era5_land` — 120 KB/monthly-file is plausible only if clipped to a tiny AOI.

`SIZE_SANITY_FLOOR_BYTES` in the ledger encodes floors (era5 200 MB, s5p
200 MB, GFAS 50 MB, VIIRS 500 MB, OSM 10 MB). Suspect folders are marked and
**their tokens are discarded** before feature matching.

Verify one file before building anything on it:

```python
import xarray as xr
ds = xr.open_dataset(f"{EXTERNAL_ROOT}/era5_land/era5_land_2012_01.nc")
print(ds.data_vars); print(float(ds[list(ds.data_vars)[0]].max()))
```

### Era windows (enforced by tests)

```
VIIRS S-NPP 2012-01-20 · Sentinel-1 2014-05-03 · Sentinel-2 2015-06-27
Sentinel-5P 2018-07-02 · SMAP 2015-04-03 · ISS-LIS 2017-03-01..2023-11-16
ERA5-Land full period · GEM/OSM static current version
```
Absence of data must never be encoded as zero anomaly.

---

## 6. RESOLVED — was the macOS FILENAME SUFFIX blocker

**P1b is closed; this is fixed.** Recorded for history only.

`files.upload()` returned the key `feature_ledger (2).py` because macOS
appended `(2)` to the duplicated filename, so `up["feature_ledger.py"]`
raised `KeyError` — before writing anything, which left Colab on the stale
module and made P1b's last cell fail with
`AttributeError: module 'feature_ledger' has no attribute 'LOCAL_INPUT_HINTS'`.
The tolerant snippet below resolved it; keep using it for future uploads.

```python
from google.colab import files
from pathlib import Path
import re, shutil

up = files.upload()

# macOS/Windows duplicate-file suffixes: " (2)", " (3)", ...
key = next((k for k in up if re.fullmatch(r"feature_ledger.*\.py", k)), None)
if key is None:
    raise RuntimeError(f"No feature_ledger*.py in upload: {list(up)}")
print("using upload key:", repr(key))

data = up[key]
Path("/content/feature_ledger.py").write_bytes(data)
Path("/content/drive/MyDrive/Firm_Clusters/pipeline/feature_ledger.py").write_bytes(data)

sys.modules.pop("feature_ledger", None)
import feature_ledger as fl
assert fl.LEDGER_VERSION == "1.2.0", f"still stale: {fl.LEDGER_VERSION}"
assert hasattr(fl, "LOCAL_INPUT_HINTS"), "stale module: no LOCAL_INPUT_HINTS"
print("module updated, version", fl.LEDGER_VERSION)
```

Then **Runtime → Restart session**, re-run P1 cell (expect
`Ledger version: 1.2.0` + `Version check passed.`), then re-run P1b.

### The multi-module upload cell — use this from now on

**Why the bootstrap cell exists (three separate traps, all now closed):**

0. **The Colab file picker returns only the files you highlight**, not the
   folder contents. Selecting one of two modules yields
   `got ['feature_ledger.py']`. Always highlight every module in one
   selection. The bootstrap cell now prints what the dialog returned, installs
   whatever arrived, and re-prompts for only the missing ones rather than
   aborting after a partial upload.
1. The P1 loader's tier A took "the file exists in `/content`" with no version
   check, and because `LEDGER_VERSION` stayed at `1.1.0` through the P3a
   vocabulary change, the guard compared 1.1.0 to 1.1.0 and validated an old
   file. Gate P1 then re-ran against the pre-P3a ledger. Fixed: version is now
   **1.2.0**, tier A parses the declared version and discards a stale copy,
   tier C skips stale Drive candidates, tier D matches the `(2)` suffix and
   verifies what was uploaded.
2. **The notebook cell was itself stale.** A cell pasted before the fix kept
   saying `already present in /content` and `expects 1.1.0`, and rejected a
   correct `1.2.0` module. All comparisons are now minimum-not-exact version
   tuples (`1.10.0 > 1.2.0`), so gaining a status never locks the pipeline
   again. **Use the bootstrap cell in README §6 instead of re-pasting
   `colab_p1_ledger.py`** — it installs both modules, cross-checks them, and
   runs Gate P1 through the ledger's own `build_audit_records()`.

`industrial_features.py` is a second module, so `colab_p3a_industrial.py` also
needs it in `/content`. Any new module must be uploaded before a cell imports
it. This cell handles every module in one shot:

```python
from google.colab import files
from pathlib import Path
import sys, re

up = files.upload()

wanted = ("feature_ledger", "industrial_features")
for mod in wanted:
    key = next((k for k in up if re.fullmatch(rf"{mod}.*\.py", k)), None)
    if key is None:
        raise RuntimeError(f"{mod}.py missing from upload; got {list(up)}")
    data = up[key]
    Path(f"/content/{mod}.py").write_bytes(data)
    dst = Path("/content/drive/MyDrive/Firm_Clusters/pipeline") / f"{mod}.py"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    sys.modules.pop(mod, None)
    print(f"{mod:22s} <- {key!r}  ({len(data):,} bytes)")

import feature_ledger as fl
import industrial_features as ind

assert fl.LEDGER_VERSION == "1.2.0", f"stale ledger: {fl.LEDGER_VERSION}"
assert hasattr(fl, "LOCAL_INPUT_HINTS"), "stale ledger: no LOCAL_INPUT_HINTS"
assert hasattr(ind, "S_NEW_IMPL"), "stale ledger: no NEW_IMPLEMENTATION status"
assert ind.INDUSTRIAL_FEATURES == fl.FEATURE_GROUPS[fl.GROUP_INDUSTRIAL]
assert ind.REQUIRED_CRS == "EPSG:6933"
print("modules OK")
```

**Use `colab_bootstrap.py` — paste that file's contents, not a copy of any
earlier cell.** It has no hardcoded module list: it installs every `.py` from the
upload *and* every `.py` in `Drive/pipeline`, so a newly added module cannot be
silently skipped. A hardcoded list caused three rounds of `ModuleNotFoundError`
here (`industrial_features`, `lineage_features`, `colab_utils`). Modules that
must be present: `feature_ledger`, `industrial_features`, `lineage_features`,
`fire_labels`, `colab_utils`.

**Drive errno 107.** `drive.unmount()` without a clean remount leaves
`/content/drive` as a dead FUSE mountpoint, where `Path.exists()` *raises*
`OSError [Errno 107]` instead of returning False. Every Drive check goes through
`colab_utils.path_ok()` / `require_drive()` for this reason; raw `.exists()`
guards are not safe there. Recovery is `drive.unmount` + `drive.mount`, or a
runtime restart if the unmount itself raises. Never `rmtree` the mountpoint
(errno 125). Note `errno.ENOTCONN` is 107 on Linux but 57 on macOS, so
`DEAD_MOUNT_ERRNOS` is written as literal numbers.

Cell order once modules are in place:
`bootstrap -> colab_p3a_remount_drive.py -> colab_p3a_diagnose_external.py
-> colab_p3a_industrial.py -> colab_p4a_lineage.py`.

---

## 7. EXACT NEXT STEPS

1. **Delete the stale `colab_p1_ledger.py` notebook cell** and use the bootstrap
   cell in README §6. Upload both `feature_ledger.py` and
   `industrial_features.py`. Expect `INDUSTRIAL: {'NEW_IMPLEMENTATION': 18}`
   and `GATE P1 PASSED`.
2. **Switch to the correct Google account, then re-verify.** Root
   cause: the existing `/content/drive` mount served a stale,
   near-empty view — exactly what the
   bootstrap cell wrote minutes earlier — so Colab was attached to a Drive with
   none of the project data. The 1.50 GB `external/` tree, the 5,337 daily
   cluster files and `canonical_event_base.parquet` are intact in the *other*
   account.

   ⚠️ `drive.mount('/content/drive', force_remount=True)` will fail with
   `ValueError: Mountpoint must not already contain files`, because the stub
   makes the mountpoint non-empty. **Do not delete `/content/drive/*` to
   "clear" it** — those paths may be the only copy of the event base. Run
   `colab_p3a_remount_drive.py`, which mounts a *fresh probe directory* and is
   non-destructive by construction. On a BAD verdict, no file operation helps:
   the credential must change. Verified on the local machine — there is **no**
   fallback copy (no `India_Firms.parquet`, `canonical_event_base.parquet`,
   `full_2012_2026/` or `Firm_Clusters/` under `~`, and no `rclone`/`gcloud`),
   so the other account is the only place the data exists.

   1. Private/incognito window → `drive.google.com`; note the account showing
      `Firm_Clusters/full_2012_2026`. Do this before touching Colab.
   2. Colab folder icon → Sign out → sign in as that account.
   3. **Runtime → Restart session.** Mandatory: the Drive credential is cached
      per runtime, so the restart is what actually swaps the account.
   4. `colab_p3a_remount_drive.py`, then `colab_p3a_diagnose_external.py`.

   ⚠️ Do not run `drive.mount('/content/drive', force_remount=True)` while the
   stub is present — non-empty mountpoint → `ValueError`. Use the probe cell.
   ⚠️ If the other account cannot be reached, **stop** — the fix is recovering
   the Drive, not re-running the pipeline.

   ⚠️ **Never re-run `build_full_global_events()`** to "regenerate" the missing
   tree: it is intact elsewhere, and regenerating would break the frozen
   regression anchors in §2.
   ⚠️ Ignore the coverage figures from any P1b run against the wrong Drive —
   with zero files on disk every token matcher fails, so "125 must call Earth
   Engine" is meaningless.
3. **Run `colab_p3a_industrial.py` with `INSPECT_ONLY = True`** once the tree is
   located. Read the printed CATEGORY MAP and the event-table schema. Confirm the
   centroid column names and that every category you expect is present. Then
   re-run with `INSPECT_ONLY = False` (`MAX_EVENTS = 50_000` first as a smoke
   test).
4. **Probe the two SUSPECT folders** — one `era5_land` `.nc` and one
   `sentinel5p` `.tif`. If both empty, decide now on the ~40 GB re-download.
   The 17 weather + 7 smoke + 9 atmospheric features are blocked until then.
5. **Run `colab_p0_cell262.py`** to settle `extract_fire_features`:
   - body found → paste verbatim, change **only** the two `ee.Date(str(...))`
     lines to `ee.Date(ee_timestamp(...))`, single-event test, then flip the
     3 FRP statuses to `COMPUTED_HISTORICAL`
   - no body → leave as NaN stub, proceed. **Do not invent VIIRS band names.**
6. **P3b quality** — the India_Firms-derived `QUALITY_UNCERTAINTY` features.
7. **P2** — 10-event forensic test, asserts 141 keys in exact order, then
   clears `runtime_verified` (this is what verifies P3a's 18).
8. **P4** — chunked production loop (Loop 1 cheap/vector for all events,
   Loop 2 remote-sense on a stratified subset).
8. **P5 labelling → P6 dataset → P7 LightGBM → P8 PostGIS → P9 GIS overlay.**
   `fire_labels.py` (6-class rule cascade, 42 tests) is already written. It
   abstains to `UNKNOWN` rather than guessing, so until the lineage extractor
   and the GEE features land, only the `INDUSTRIAL_FLARE` rule is reachable.
   **Do not lower label thresholds to manufacture labels** — raise coverage by
   implementing extractors instead. The full ordered path to training, including
   which features are actually trainable, is README §5.

---

## 8. LABEL DESIGN (per PS)

Labels must derive from evidence **not** used as model features.

```
INDUSTRIAL_FLARE    facility inside/adjacent + thermal persistence
                    (constant FRP, 200+ days, no wind displacement, n_sat≥2)
INDUSTRIAL_FIRE     facility adjacent + surface loss that does not recover
MINING              bare-mining land cover + persistent thermal
AGRICULTURAL        seasonal + large area + low FRP + diurnal pattern
WILDFIRE            BT>350 K + high VPD + wind-driven displacement
UNKNOWN             insufficient evidence — KEEP these, use sample_weight
```

Leakage policy declared before any training:

| Partition | n | Use |
|---|---|---|
`LABEL_DERIVED` (all 18 industrial) | 18 | screening + interpretation only |
`NON_LEAK` | 123 | the scientific model — this is what the paper reports |

Report **two** models: A = all 141 (operational, honest about using facility
priors), B = 123 non-leak (scientific).

Build a **hand-labelled gold set** (~300 events: Jamnagar/Vadinar/Panipat
refineries, Bokaro/Bhilai steel, coal plants, mines) to audit the rule engine.
Never use it to fit the model.

Baselines to beat: single threshold on `DIST_NEAREST_FACILITY_LOG`, and
LightGBM on thermal+temporal only. If ML cannot beat the distance threshold,
the ML adds nothing — say so.

Validation splits (not random — both spatial and temporal leakage inflate):
1. out-of-time: train ≤2019, test 2020–2026 → PR-AUC, recall@precision≥0.90
2. out-of-space: GroupKFold on 0.1° grid
3. combined `year × block` for the honest final number

Freeze an operating threshold τ maximising F2 subject to precision ≥ 0.90.
**Now implemented** — `threshold_calibration.py`, calibrated on the TRAIN fold
only, persisted in the pickle. Note that τ and the 0.60 review floor are
*different constraints*: auto-accept requires `score >= max(τ, 0.60)`. A `τ` of
`None` means no operating point was freezable and every event goes to a human;
it must never be read as "no constraint".

---

## 9. HARD RULES

- Do not re-run `build_full_global_events()`
- Do not load all 7.26M events into pandas
- Do not start the 7.26M feature loop before P2 passes
- Do not start any Earth Engine loop before the external-data inventory is verified
- Do not replace cell 325 with cell 372/387
- Do not use the cell 310 weather stub
- Do not invent historical feature implementations — label anything new
  `NEW_IMPLEMENTATION`
- Never silently convert missing implementation into "computed"
- `runtime_verified` stays `False` until P2/P4 actually execute the feature
- Every event output must contain exactly 141 keys in the authoritative order
- Never infer `BURN_SCAR_GROWTH_RATE` from FRP
- Classify industrial vs natural **and** separate mining (per PS, mining is
  listed with the industrial facility types)
- P3a: never pass coordinates to `compute_industrial_features` without an
  explicit `crs`, and never report a missing gazetteer category as 0
- Never approximate `BT_I5_MEDIAN` from `bright_t31`: MIR is not I5. Six
  features are permanently unavailable and stay NaN
- Never let tied model scores be treated as ordered -- LightGBM emits ties
  constantly, and an arbitrary tie order turns a useless model into PR-AUC 1.0
- Report model A and model B. Reporting A alone is facility-prior leakage;
  reporting B alone hides that the operational system uses those priors
- Never apply a bare `REVIEW_CONFIDENCE = 0.60` as a classifier threshold. It is
  the brief's human-review floor, not an operating point, and it is meaningless
  on a model that is not calibrated to that prior. Use
  `effective_threshold(tau, 0.60)`
- Never substitute a threshold when calibration returns `None`. Report that no
  operating point exists; a fallback number becomes a false coverage claim
- `dist_m` is metres and `DIST_*_LOG` is `log10(1 + d_metres)`. They were once
  interchanged in the evaluation CSVs, which moved the column by four orders of
  magnitude while leaving every ranking-based metric untouched. Unit labels are
  now guarded at runtime in P7 and statically in `test_colab_cells`
- P3a: verify the printed CATEGORY MAP by eye before computing — a wrong
  mapping corrupts both the features and the labels, and neither shows up as
  an error
- Commit nothing to git without an explicit request

---

## 10. FILES IN REPO

```
feature_ledger.py                  schema + status map + drift check (v1.3.0)
weak_labels.py                     P5b weak industrial-context hierarchy + leakage registry
feature_gap.py                     Phase 0 gap / obtain-routes / site occupancy
test_feature_gap.py                24 tests
rule_calibration.py                percentile / target-count threshold calibration
test_rule_calibration.py           13 tests
test_weak_labels.py                39 tests
industrial_features.py             P3a industrial computation core (18 features)
colab_utils.py                    dead-mount-safe checks + project-root resolution
lineage_features.py                P4a tier 2: detection-level lineage
cluster_tier.py                    P4a tier 1: 34 features from clusters/ only
training_design.py                 P7: leakage partition, splits, baselines, metrics
threshold_calibration.py           P7: operating threshold (F2, precision >= 0.90), TRAIN only
test_threshold_calibration.py      33 tests: ties, infeasible precision, degeneracy, None handling
colab_recon.py                     Colab: whole-project recon -> qc/project_recon.json
fire_labels.py                     P5 label rule cascade (6 classes)
build_feature_audit.py             local ledger builder, asserts Gate P1
test_feature_ledger.py             39 tests, all passing
test_industrial_features.py        48 tests, all passing
test_fire_labels.py                42 tests, all passing
test_lineage_features.py           59 tests, all passing
test_colab_utils.py                22 tests
test_colab_root.py                 41 tests
test_dataset_builder.py            50 tests
test_cluster_tier.py                32 tests
test_training_design.py             41 tests
colab_p1_ledger.py                 Colab: load module, version guard, drift, write ledger
colab_p1b_external_inventory.py    Colab: inventory external/, size QC, match features
colab_p0_cell262.py                Colab: recover extract_fire_features
colab_p3a_industrial.py            Colab: map gazetteer, project, stream 7.26M events
colab_p3a_diagnose_external.py     Colab: read-only A/B/C/D verdict, missing external tree
colab_p3a_remount_drive.py         Colab: non-destructive probe mount to locate the real Drive
colab_p4a_lineage.py                Colab: stream daily clusters -> 54 features per event
colab_bootstrap.py                  THE bootstrap cell (self-healing, no MODULES list)
colab_p7k_nbr_change.py             Colab: monthly Landsat NBR per event, sharded+resumable
colab_p7l_sar.py                    Colab: monthly Sentinel-1 VV/VH per event, sharded+resumable
colab_p7k_nbr.py                    RETIRED draft -- stub that refuses to run
test_colab_p7k.py                   26 tests: P7k rehearsal against stubbed Drive/ee
test_colab_p7l.py                   23 tests: P7l rehearsal against stubbed Drive/ee
qc/                                generated ledger + work list
Stage1..Stage6.ipynb               clustering + production (no feature extractor)
backend/ frontend/ react-frontend/ AGNI-NETRA + SPARC design docs
README.md                          standing project status + change log
SYSTEM_ARCHITECTURE.md
```

Repo has **no** AGENTS.md lint/typecheck commands — this is a data-science
repo. Verification =
`python3 -m unittest test_feature_ledger test_industrial_features test_fire_labels test_lineage_features test_colab_utils test_colab_root test_cluster_tier test_training_design test_colab_cells test_dataset_builder test_weak_labels test_feature_gap test_threshold_calibration test_explainability test_rule_calibration test_gee_utils test_colab_p7k test_colab_p7l` (718 tests) and
`python3 build_feature_audit.py --outdir qc`.
Backend: `cd backend && python3 -m unittest discover -s tests` (30 tests).

Note: local machine has no `pyarrow`, so `build_feature_audit.py` writes CSV
only. Colab has pandas+pyarrow and writes parquet.

---

## 16. P7k (NBR) and P7l (SAR) -- built 2026-10-07, not yet run

Two new multi-hour Earth Engine cells complete the burn-evidence chain.
Both are sharded and resumable in exactly the P7h pattern (shard identity
derived from the chunk number, atomic `.partial` rename, resume validated
against expected row counts), both sample `meta.parquet` DIRECTLY (no P7h
dependency), and both were rehearsed end-to-end against stubbed Drive/ee
(`test_colab_p7k.py` 26 tests, `test_colab_p7l.py` 23 tests).

### What each cell produces

| Cell | Output | Columns | Sensor / composite |
|---|---|---|---|
| `colab_p7k_nbr_change.py` | `qc/nbr_events.parquet` | row, event_id, start_year, start_month, NBR_PRE, NBR_POST, DNBR | Landsat C2 L2 monthly median, band `NBR` |
| `colab_p7l_sar.py` | `qc/sar_events.parquet` | row, event_id, start_year, start_month, VV_PRE, VH_PRE, VV_POST, VH_POST, VV_CHANGE, VH_CHANGE, VV_VH_CHANGE | Sentinel-1 GRD monthly median, bands `VV`,`VH` (dB) |

DNBR = NBR_PRE − NBR_POST, **positive where vegetation was lost** — this
matches the ledger's DNBR ("burn severity", mean 0.05) and the schema
comment. Note it is the NEGATIVE of fire_labels' `NBR_CHANGE` convention
(negative = burnt); both signs are recorded here so no downstream code has
to guess. VV/VH are dB medians; changes are post − pre.

### New `gee_utils` helpers (v bumped to match)

`month_before(y, m, lag)`, `nbr_monthly_composite(y, m, collection_id=None)`
(single band `NBR`, standard (NIR−SWIR) convention, QA_PIXEL bits 3+4 masked,
**CLOUD_COVER < 80 filter**), `_month_bounds(y, m)`, `sar_imagery_gap(years)`,
`sar_monthly_composite(y, m)` (VV/VH, both orbit passes mixed — documented),
`colab_utils.event_year_month(meta_path)`.

### The bug this session caught before it cost a run

`nbr_monthly_composite` originally filtered on
`CLOUDY_PIXEL_PERCENTAGE < 80` — a **Sentinel-2** property. Landsat C2 L2
carries `CLOUD_COVER`; filtering on a property the asset does not have
silently excludes EVERY image, the composite goes empty, and the 11-hour run
returns pure NaN with no error. Fixed to `CLOUD_COVER`. This is the same
class of failure as the P7e zero-value AOI check: band/property presence
must be verified against the catalog, not assumed.

### Two static-check blind spots fixed in `test_colab_cells.py`

1. `test_no_augmented_name_is_undeclared` did not count function PARAMETERS
   as bindings (`ast.arg` is not a Store-context Name), so any function that
   wrote into its own parameter was flagged. Parameters are now bound.
2. Both P7k/P7l cells consequently take the composite cache as an explicit
   parameter (`_composite(cache, y, m)`) instead of mutating a module global
   — the hidden-state pattern the check exists to catch.

### Retired: `colab_p7k_nbr.py`

The first P7k draft (per-event EE calls, own-month composites so
NBR_CHANGE ≡ 0, crash on missing coordinate columns) is now a stub that
prints its defects and exits. A stale copy sits in Drive/pipeline from
earlier uploads; uploading the stub replaces it.

### How to run (Colab)

```python
import colab_runcell
colab_runcell.run("colab_p7k_nbr_change")   # ~11 h, resumable
colab_runcell.run("colab_p7l_sar")          # ~22 h (4 samples/event), resumable
```

Coverage: NBR from 2013 (Landsat 8), SAR from 2014 (Sentinel-1). Events
before those years, with unparseable `start_time`, or with non-finite
coordinates keep NaN — never 0. DNBR/DNDVI in the ledger's SAR group come
from Sentinel-2 (separate question, not yet built); COHERENCE_LOSS has no
GRD route and stays NaN rather than being faked.

### Run 1 (2026-10-07, Colab): two crashes, same root cause, both fixed (v.2)

Both cells died on the FIRST sample call. The month-coverage check was
YEAR-granular, but the catalogs start mid-month:

- **P7k** — `imagery_gap` says Landsat 8 = 2013, but L8 L2 data starts
  **2013-03-18**. Months 2013-01/2013-02 passed the year rule, their
  collection was EMPTY, and `median()` of an empty ImageCollection is a
  ZERO-BAND image → `Image.normalizedDifference: Input image must have at
  least 2 bands`.
- **P7l** — GEE's S1 GRD archive starts **2014-10-03** (the ledger's
  2014-05-03 era is first acquisition, which GEE does not carry). Months
  2014-01..09 were "covered" → empty → `Image.sampleRegions: Invalid
  numInputs: 0`.

Fixes (cell versions 2026-10-07.2): gee_utils gains
`LANDSAT8_FIRST_DATE` / `S1_GRD_FIRST_DATE` and the month-granular
predicates `nbr_month_has_imagery` / `sar_month_has_imagery`; both cells
intersect the year rule with the month rule and print the dropped months.
Also removed the scene-level `CLOUD_COVER < 80` filter from
`nbr_monthly_composite`: Indian monsoon months can have every scene above
80% scene cloud, which is the same empty-collection crash one run later;
the QA_PIXEL pixel mask does the real work and a median over clear pixels
is the composite. Rehearsal stubs updated to the new chain, 4 new
`MonthCoverageTests` in test_gee_utils (722 total).

### Output folder (user request, v.2)

Both cells now also write the assembled table to
**`Firm_Clusters/nbr_sar_outputs/`** (`nbr_events.parquet`,
`sar_events.parquet`) in addition to the pipeline-contract `qc/` paths
that P7i/P7j read. Same bytes, two locations.

### Run 2 outcome and two more fixes (2026-10-07)

- **P7k COMPLETED.** 30/30 shards, 6,763,812 rows written (220,388-row
  shard 0 was correctly trusted and skipped by the new resumed rule; total
  ~5.1 h). NBR_POST non-NaN 6,447,430; DNBR mean +0.077 / median +0.055,
  5,118,131 events positive (vegetation lost) — sanely burn-like.
- **P7l shard-trust bug + short-shard bug** fixed in the same edit as P7k:
  a completed shard holds only FILLED rows (not a fixed slot count), so the
  resume check now trusts any within-band unique row set instead of an
  exact-count match — an exact-count check would have DISCARDED and redone
  every partially-filled shard on resume. New tests: shortened shard is
  trusted / out-of-band shard is discarded (P7h-style corruption).
- **P7l run 2 crash (EW swath) fixed.** `sar_monthly_composite` mixed
  Sentinel-1 **EW** scenes (bands HH/HV) into the median, so
  `select(["VV","VH"])` died with "Band pattern 'VV' did not match any
  bands. Available bands: [HH, HV, angle]". Now filtered to
  `instrumentMode == "IW"`. Regression test added. `gee_utils` v +1.
- Suite: 724 tests, OK.

### P7n (weather/FWI) and P7o (atmospheric chemistry) -- new, 2026-10-07

New extraction cells complete the WEATHER_FIRE_DANGER and ATMOSPHERIC_CHEMISTRY
ledger groups:

- `fwi.py` (+ `test_fwi_weather.py`): pure, dependency-free Canadian FWI
  system (FFMC/DMC/DC/ISI/BUI/FWI). DOCUMENTED approximation: moisture codes
  carried on a monthly step, not the daily chain -- recorded in the module
  docstring. Unit-tested for bounds and monotonicity.
- `gee_utils` gains `era5_monthly_weather`, `era5_monthly_climatology`
  (1991-2020), `s5p_monthly_median`, `s5p_monthly_climatology` (2019-2023),
  and `sample_bands_chunked` (multi-band point sampling).
- `colab_p7n_weather.py` (v2026-10-07.1) -> `qc/weather_events.parquet`
  (+ weather_outputs/): T/RH/VPD/SOIL anomaly z-scores, WIND_SPEED, RAIN logs,
  FWI family. NaN for unparseable dates; reads meta.parquet; sharded+resumable.
- `colab_p7o_atmos.py` (v2026-10-07.1) -> `qc/atmos_events.parquet`
  (+ atmos_outputs/): NO2/SO2/CO/CH4/HCHO/AOD anomaly z vs 2019-2023 monthly
  climatology + CO_NO2_RATIO_LOG. Pre-2018 events stay NaN (TROPOMI archive).
- **A chunk-level month-boundary bug was caught and fixed in all three new
  cells**: a 5,000-row chunk can straddle two months; grouping now happens per
  (year, month) within the chunk, so no event is ever sampled against a
  neighbour's month.
- Suite: 732 tests, OK (13 new). NOTE: the local `python3` on this machine
  resolves to Homebrew 3.14 (no numpy); tests must run with
  `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`.