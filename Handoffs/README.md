# SPARC — Industrial Fire & Thermal-Anomaly Classification Pipeline

AI-enabled geospatial system that classifies **industrial fires and persistent
thermal sources** from NASA FIRMS VIIRS data, and publishes them as a GIS
overlay.

Deep forensic detail lives in [`HANDOFF.md`](HANDOFF.md) — read it first in any
new session. This file is the **living status + change log**.

---

## 1. Pipeline

```
15,392,720 VIIRS detections (SNPP / NOAA-20 / NOAA-21)
        ↓  daily cKDTree 1 km + 12 h, connected components
7,589,305 local clusters
        ↓  cross-day cluster edges (323,804)
7,265,849 global fire events          ← unit of analysis
        ↓  141 event-level features    ← CURRENT STAGE
X_event
        ↓  independent rule-based labelling
y_event (INDUSTRIAL_FLARE / INDUSTRIAL_FIRE / MINING /
         AGRICULTURAL / WILDFIRE / UNKNOWN)
        ↓  LightGBM → PostGIS → GIS overlay
```

### Status

| Phase | State | Notes |
|---|---|---|
| P0 | **open** | recover `extract_fire_features` from cell 262 (15 features) |
| P1 | **closed** | 141-feature ledger frozen, `LEDGER_VERSION 1.3.0`, gate passes |
| P1b | **closed** | external inventory verified: 911 files / 213 subdirs / 1.50 GB. `external_data_inventory.json` + `external_coverage_matrix.csv` written to `qc/` |
| P3a | **code done, not yet run** | 18 INDUSTRIAL features implemented (`industrial_features.py` + `colab_p3a_industrial.py`), 48 tests. Needs the Drive, then `INSPECT_ONLY = True` → `False` |
| P4a | **code done, not yet run** | 54 lineage features implemented (`lineage_features.py` + `colab_p4a_lineage.py`), 59 tests. Same inspection-first pattern |
| P5 | **code done** | `fire_labels.py`, 6-class cascade, 42 tests. Abtains when evidence is missing |
| P2 | pending | 10-event runtime test, clears `runtime_verified` |
| P3b–g | pending | quality 8, land cover 16, FWI 7, SAR 7, smoke/atmos 17, recurrence 4 |
| P4 | pending | chunked production feature loop |
| P5–P9 | pending | labelling → dataset → LightGBM → PostGIS → GIS overlay |

**Feature readiness: 141 declared · 12 `COMPUTED_HISTORICAL` · 72 `NEW_IMPLEMENTATION` · 10 `UNVERIFIED_HISTORICAL` · 7 `HISTORICAL_NAN` · 40 unimplemented · 0 `runtime_verified`.**

---

## 2. Confirmed Drive layout (via `colab_survey_tree.py`)

Measured, not assumed. Sizes and counts are from the survey run.

```
My Drive/Firm_Clusters/                       3 files  4 dirs   622.7 MB
  India_Firms.parquet                                      622.7 MB  15,392,720 rows
  inspect_fire_features.py                                  8 KB   <- possible cell-262 body
  pipeline/    6 files                     120.3 KB   <- bootstrap writes modules here
  qc/          5 files                     100.7 KB
  qc(old)/    75 files  6 dirs             256.6 MB   incl. corrupt_files/ 3.5 MB
  full_2012_2026/                            13 dirs
    clusters/          5,337 files          532.0 MB   clusters_YYYY-MM-DD.parquet
    edges/             5,335 files           12.8 MB   edges_D1_to_D2.parquet
    mappings/          5,337 files          150.6 MB
    events/              8 files  3 dirs      1.7 GB
      event_geometry/    2 files              314.5 MB
      event_features/    empty
      event_qc/          empty
    external/            3 files 10 dirs     210.4 MB   <- see below, CHANGED
    feature_audit/      19 files  1 dir        1.7 MB
      dataset_previews/ 179 files             43.4 KB
    analysis/            0 files  8 dirs        0 B     (temporal/ 14 files)
    feature_report/      5 files              638.1 KB
    features/            1 file               3.0 KB    feature_terrain.py
    features_missing/    0 files  1 dir         0 B
    logs/                1 file               769.8 KB
    qc/                  4 files               55.0 KB   10_event forensics
    shubham_thermal_feature_analysis/ 1 file  2.8 MB
```

### Three findings that would have broken the run

| Finding | Consequence |
|---|---|
| Cluster files are `clusters_2012-01-20.parquet` — ISO dates with **dashes**, not `YYYYMMDD` | building `f"{day}.parquet"` from a cluster id finds **zero files, silently**. Now indexed by pattern discovery |
| `edges/` sits beside `clusters/`, 5,335 files | a naive `*.parquet` glob would pull edge rows into the detection join. Filtered by prefix |
| Membership ids need `.str.slice(0,8)` **and** an 8-**digit** check | a dashed or short id coerces to `NaT`, silently dropping that day |

### `external/` — recon counts match P1b exactly

A survey run reported `external/` as 210.4 MB, which suggested the tree had
changed. It had not: that was **another stale FUSE listing**. The recon (run
after a forced refresh) reproduces P1b's counts:

| Folder | Recon | P1b |
|---|---|---|
| `VIIRS_FOOTPRINTS/` | 12 | 12 |
| `era5_land/` | 175 | 175 |
| `sentinel5p/` | 506 | 506 |
| `india_mopitt_gfas_output/` | 177 | 177 |
| `osm_facilities/` | 37 | 37 |
| `DL_FIRE_M-C61_814501/` | 3 | 3 |
| `gee_features/` | 1 | 1 |

**P1b's numbers stand.** Lesson: a size or count that contradicts a prior
verified measurement is evidence of a stale mount, not of changed data.



---

### Event base schema (measured, 7,265,849 rows / 60 row groups)

Two findings here matter more than the path fixes.

**1. The centroid columns are `event_centroid_lon`/`_lat`, and
`event_centroid_x`/`_y` are already projected.** P3a now discovers the
coordinate pair automatically and checks the magnitudes before trusting it:
values beyond 180 are metres, values inside are degrees. Because the `x`/`y`
pair is already Equal Earth, P3a skips projection entirely — which removes a
whole class of unit error rather than merely fixing a column name.

**2. The event base already contains cell-325-style aggregates:**

```
n_local_clusters  n_detections  total_frp  max_frp  duration_hours
geometry_total_frp  geometry_max_frp  geometry_mean_frp
geometry_n_detections  geometry_duration_hours
geometry_max_radius_m  geometry_max_radius_km
geometry_n_satellites  geometry_cluster_count
```

These are almost certainly the cell-325 output that `HANDOFF` §4 recorded as
never having been located. That is a **P0/P2 opportunity**: the lineage block
can be cross-checked against a pre-existing independent computation instead of
being taken on trust. Do not assume they agree — compare them.

Also note `*_1` duplicates (`start_time_1`, `n_detections_1`, …), debris from a
merge at some point. Only the un-suffixed columns are read, so nothing breaks,
but the table is not clean.

### The gazetteer is duplicated

`osm_facilities/` holds a directory duplicated on Drive, so the same filenames
appear twice with different row counts:

```
facility_categories/       cement 157 · steel 214 · mines 199 · lng 23 ...
facility_categories (1)/   cement  50 · steel  58 · mines 283 · lng  0 ...
india_osm_facilities.geoparquet   147,606 rows and 165,869 rows (two copies)
all_industrial_facilities.geoparquet  3,551 rows
all_industrial_facilities_v2.geoparquet  74,021 rows
```

Loading both copies would double-count facilities and bias every
`FACILITY_COUNT_*`. P3a now collapses same-named duplicates, keeps the
unsuffixed directory, and **prints every dropped file**.

Category coverage after label-based fallback: refinery 4pt/70poly,
powerplant 37,728/29,388, petrochemical 0/56, steel 216/50, cement 76/40,
lng 3/2, mine 828/18,030, landfill 1,312/3,949, pipeline 37/60, wellpad 24/3.
`landfill` and `petrochemical` have **no** file of their own — they come from
`india_osm_facilities.geoparquet`'s in-file category column, which is correct
but worth confirming.

## 3. The lineage data path — corrected by recon

**`clusters/` holds cluster-level aggregates, not detections.** This overturned
the P4a design, which had assumed detections lived there.

```sql
-- clusters/clusters_YYYY-MM-DD.parquet, 5,337 files, 532 MB
date  local_cluster_id  start_time  end_time  n_detections
centroid_x  centroid_y  total_frp  mean_frp  max_frp
n_observation_times  duration_hours  satellites  n_satellites
max_radius_m  max_radius_km  centroid_lon  centroid_lat
```

There is no per-detection `frp`, `i4`, `i5`, `confidence` or scan angle. The
real detection table is `Firm_Clusters/India_Firms.parquet` (622.7 MB, 15,392,720
rows × 22 cols), reachable via `mappings/detection_cluster_mapping_*.parquet`
(`detection_index → local_cluster_id`).

### Two tiers, and why the split matters

| Tier | Input | n | Cost |
|---|---|---|---|
| **Tier 1** `cluster_tier.py` | `clusters/` + membership | **34** | minutes — no join to 15.4M detections |
| Tier 2 `lineage_features.py` | `India_Firms` + `mappings/` | 12 | hours |

Tier 1 covers all 9 FRP features, 9 temporal, 10 spatial, and 6 geometry
metrics — two thirds of the lineage block from 532 MB of aggregates. The 20
deferred features each carry a recorded reason in `cluster_tier.DEFERRED_REASONS`
(12 need per-detection radiometry, 4 need a cross-event pass, 1 needs era5 wind,
1 needs MODIS QA, 1 needs every group, 1 more). A module-level assertion fails
if any of the 54 is unaccounted for.

### Bridges available

```
mappings/detection_cluster_mapping_YYYY-MM-DD.parquet
    detection_index (int64)  local_cluster_id (string)
events/event_local_cluster_membership.parquet          7,589,305 rows
    event_id  local_cluster_id  date                  <-- explicit date column
edges/edges_D1_to_D2.parquet                          5,335 files
    cluster_a  cluster_b
```

`membership/` (7,589,305 rows = the 7,589,305 local clusters) is the
event↔cluster bridge; `mappings/` is the detection↔cluster bridge. Tier 1 needs
only the first.

---

## 4. Frozen baseline — do not change

Source `Firm_Clusters/India_Firms.parquet`, 15,392,720 rows × 22 cols,
2012-01-20 → 2026-08-31, 5,337 observed dates. Thresholds 1,000 m
(EPSG:6933) and 12 h. Algorithm: cKDTree → temporal filter →
connected_components. Local cluster ID `YYYYMMDD_<component_label>`.

Frozen regression anchors — if any of these change, **stop**:

| Experiment | local clusters | cross-day edges | global events |
|---|---|---|---|
| 2021-03-31 | 12,641 | — | — |
| 2021-03-31→04-01 | 25,542 | 2,066 | 23,476 |
| 7-day | 77,244 | 8,738 | 68,514 |
| March 2021 | 230,239 | 21,379 | 208,884 |
| 2-month | 366,067 | 33,490 | 332,607 |
| Full 2021 | 700,146 | 40,916 | 659,280 |

March 31 detail: 37,801 detections / 89,322 spatial pairs / 75,626 valid
pairs / 12,641 clusters. Cross-day: 15,887 detection links → 2,066 cluster
links.

`build_full_global_events()` must **never** be re-run.

---

## 5. Two label columns, and why

| Column | Engine | Character | Use |
|---|---|---|---|
| `label` | `fire_labels.py`, evidence-gated | almost all UNKNOWN; defensible where it fires | the honest target |
| `label_weak` | `weak_labels.py`, industrial-context hierarchy | labels far more | **never** to score model B |

**Every `label_weak` value is manufactured from features the model also sees.**
`weak_labels.LABEL_INPUT_FEATURES` publishes every input the rules read, and
`training_design` excludes all of them from model B. On the full schema that is
**24 of 141 features** removed: the 18 industrial plus 6 weak-rule inputs. With
the real 56-feature block it is worse — **17 of 24**, leaving model B with 7.

That cost is the honest price of labelling from your own features, and it is
exactly why external ground truth (facility registries, event catalogues) is the
only route to a label set that can validate a model.

### Two defects found in the proposed hierarchy

1. **Unit error.** `DIST_*` is `log10(1 + d_metres)`; the proposal compared it
   to `np.log1p`. Every distance radius became 6–25× tighter than intended —
   "within 5 m" silently meant 61 m — and with 2/5/10 m radii essentially
   nothing could be labelled. Distances now use `log10(1+R)`; `DETECTION_COUNT_LOG`
   genuinely is `log1p(n)`, so `log1p` is right there.
2. **`FRP_PERSISTENCE` is ≈1.0 for almost every event**, because an event exists
   precisely on the days it was detected. So `persistence >= 0.35` is true for
   ~96.7% of rows and is **not a discriminator**. The `industrial_shape` test
   therefore reduces to "near a facility", which is the leakage above, made
   concrete.

The MINING rule's `pd.isna(persistence) or persistence >= 0.30` was also changed
to abstain on missing: it made MINING the fallback for every single-cluster event
near a mine, turning absence of evidence into evidence.

---

## 6. Measured feature coverage (Phase 0, 2026-10-03)

`colab_p0_gap.py` against the real `X.parquet` (7,265,849 × 56):

```
PRESENT (usable) :  33
SPARSE (>90% NaN):  20     kept, never imputed
MISSING          :  88
coverage         : 23.4% of the 141-feature spec
```

The 20 sparse columns are the trend and geometry features, undefined for the
96.7% of events that are single-cluster. That is structural, not a defect, and
LightGBM routes the NaNs natively.

### The 88 missing, by route to obtain them

| Route | n | Groups |
|---|---|---|
| **DERIVE** | **18** | THERMAL 6 · QUALITY 7 · TEMPORAL 5 · SPREAD 1 |
| EXTERNAL_RASTER | 46 | WEATHER 17 · LANDCOVER 16 · TERRAIN 12 · QUALITY 1 |
| EXTERNAL_TABLE | 24 | ATMOSPHERIC 9 · SMOKE 7 · SAR 7 · WEATHER 1 |

**Only 18 features can be obtained with no download at all**, and they are
exactly the detection-tier features still needing `India_Firms` joined through
`mappings/`: `BT_I4_MEDIAN`, `BT_I5_*`, `SUBPIXEL_TEMP_MEDIAN`,
`SOURCE_AREA_MEDIAN`, `FIRMS_CONFIDENCE`, `SCAN_ANGLE_*`, `VIEW_ZENITH_*`,
`PIXEL_SIZE`, `NIGHT_DAY_FRP_RATIO`. Six of those are **permanently
unavailable** (no I5 band, no subpixel temperature, no source area, no pixel
size), so the realisable count is **12**.

That reframes the work: the cheap win is 12 features, not 18, and the two
single highest-value external blocks remain **land cover (16)** and
**weather (17)**.

### Leakage partition, measured

```
model A (operational) : 53 features
model B (scientific) : 29 features
excluded as label inputs: 18 industrial + 16 weak-rule
```

Model B keeps the 15 thermal features, the duration/overpass/density terms, the
spatial track, and the geometry metrics. **This corrects an earlier estimate of
7**, which was computed from a hand-picked 17-column list rather than the real
53-column `X`.

### Why every event was `no_rule` — the measured blocking terms

| Rule | conjunction | Blocking condition |
|---|---|---|
| INDUSTRIAL_FLARE | 0 | `KNOWN_FLARE==1` (no catalogue) **and** `INSIDE_FACILITY==1` |
| INDUSTRIAL_FIRE | **1,122,726 (15.5%)** | none — this rule was satisfiable all along |
| MINING | **147,962 (2.0%)** | none |
| AGRICULTURAL | 0 | `FRP_PERSISTENCE < 0.5`, impossible |
| WILDFIRE | 0 | needs land cover; no discriminating input exists |

The 100% UNKNOWN was **not** the rules being unsatisfiable. `INDUSTRIAL_FIRE` and
`MINING` were ready the whole time — P6 ran a stale cell, so `label_weak` was
never produced.

### Three defects this run exposed

1. **`FRP_PERSISTENCE` was exactly 1.0 for all 7,265,849 events.** The
   definition `days / (span_days + 1)` is a tautology: an event exists only
   while detected, so it is seen on every day of its own span. A constant
   column carries no information, and it silently zeroed AGRICULTURAL, whose
   condition `FRP_PERSISTENCE < 0.5` could never hold. Now measured over a fixed
   30-day window: a 2-day event scores 0.067, a 60-day event 1.000.
2. **`INSIDE_FACILITY == 1` for zero events.** It point-in-polygons an event
   centroid against 38,370 small OSM polygons; centroids sit on a 1 km detection
   grid and almost never land inside. The rule wrongly required it. Dropped as a
   conjunct — `DIST_REFINERY_LOG <= 2000 m` passes 17,727 events and is the
   robust proxy. The feature stays in X as its own signal.
3. **The site grid was misread.** 0.1° is *not* too fine: it gives 26,828 cells
   with a median of 134 events each, so 99.99% of events have company. The
   problem is the opposite — a baseline that local is effectively a regional
   average, so almost nothing looks anomalous. `SITE_DEG` is now **1.0°**
   (~110 km), which puts a refinery and its surrounding countryside in different
   cells. Verified: the anomaly column is 100% populated and a 390 MW spike
   against a 34 MW baseline scores **11.5×** — the brief's own scenario.

## 7. P1b result — external data coverage

P1b is **closed**. The upload blocker described in HANDOFF §6 is resolved
(`files.upload()` had returned `feature_ledger (2).py`; the tolerant snippet
fixed it) and the module now imports at `LEDGER_VERSION 1.2.0` with
`LOCAL_INPUT_HINTS` present.

Written to Drive:
`qc/external_data_inventory.json`, `qc/external_coverage_matrix.csv`.

Inventory: 213 subdirs · 911 files · 1.50 GB (1,606,010,787 bytes).

| Folder | Files | Size | Verdict |
|---|---|---|---|
| `VIIRS_FOOTPRINTS/` | 12 | 1.08 GB | **OK** — yearly `.geoparquet` 2012–2026 + combined v3 |
| `osm_facilities/` | 37 | 0.28 GB | **OK** — `facility_categories/*.geoparquet` (cement, lng, mines, pipelines + 31 more); P3a and label source |
| `DL_FIRE_M-C61_814501/` | 3 | 0.09 GB | OK — FIRMS archive + NRT + readme |
| `india_mopitt_gfas_output/` | 177 | 0.02 GB | OK — monthly parquet under `final/year=YYYY/month=MM/` |
| `era5_land/` | 175 | 0.02 GB | **SUSPECT** — ~120 KB per monthly `.nc`; plausible only if clipped to a tiny AOI |
| `sentinel5p/` | 506 | 0.00 GB | **SUSPECT** — `ch4/S5P_CH4_2014_…tif` layout, 505 `.tif` totalling ~0 GB |
| `gee_features/` | 1 | 0.00 GB | `logs/batch_manifest.csv` only — no past GEE run ever wrote output |

Extensions: 34 `.geoparquet` (1.18 GB), 1 `.geojson` (0.13 GB), 2 `.pbf`,
4 `.csv`, 176 `.parquet`, 175 `.nc`, 505 `.tif`, 8 `.json`, 5 `.txt`, 1 `.jsonl`.

Matched tokens: `cement, ch4, era5, era5_land, facilities, facility, fire,
firms, hcho, industrial, lng, mine, no2, osm, pipeline, powerplant, refiner,
s5p, sentinel5, so2, steel, viirs, wellpad`.

### Coverage of the 129 open features

```
input already on disk (no GEE)  : 100
no input required at all        :  41
must call Earth Engine          :  25   (LANDCOVER 16, SAR 7, QUALITY 1, WEATHER 1)
```

### ⚠️ The "100 local" figure is inflated by 33

The local-ready table matched the tokens `era5`/`era5_land` and `s5p`/`sentinel5`,
which is exactly what the size-sanity filter was supposed to suppress. Those two
folders are SUSPECT, so **33 of those 100 rest on data that may be empty**:

```
WEATHER_FIRE_DANGER  #93–109        17   era5_land only
OPTICAL_SMOKE        #111–117        7   sentinel5p only
ATMOSPHERIC_CHEMISTRY#125–133        9   sentinel5p only
```

Real local-ready count is **67**: 47 India_Firms-derived (#1–46 plus #134–139),
18 industrial from `osm_facilities`, and 41 needing no input at all. Treat the
33 as `UNVERIFIED_INPUT` until probed — do not start work on them and do not
encode their absence as a zero anomaly.

Probe before committing to any re-download (~40 GB decision):

```python
import xarray as xr
ds = xr.open_dataset(f"{EXTERNAL_ROOT}/era5_land/era5_land_2012_01.nc")
print(ds.data_vars); print(float(ds[list(ds.data_vars)[0]].max()))
```

### Immediate work order

1. **P3a run** — `colab_p3a_industrial.py` in Colab, `INSPECT_ONLY = True` first.
   Confirm the printed CATEGORY MAP, then flip to `False`. See §4.
2. Probe `era5_land` + `sentinel5p` (one file each).
3. **P0** — run `colab_p0_cell262.py` to settle `extract_fire_features`. Body
   found → paste verbatim, swap only the two `ee.Date(str(...))` calls for
   `ee.Date(ee_timestamp(...))`, single-event test, then flip the 3 FRP statuses
   to `COMPUTED_HISTORICAL`. Not found → leave the NaN stub, move on. Never
   invent VIIRS band names.
4. **P3b** — the 7 India_Firms quality features (#134–139).
5. **P2** — 10-event forensic test: 141 keys in exact order, then clear
   `runtime_verified`.
6. No Earth Engine loop before the external inventory questions above are settled.

---

## 8. P3a — industrial context (18 features)

> **Blocker — SOLVED DIAGNOSIS: wrong Google account.**
>
> `colab_p3a_industrial.py` raised `FileNotFoundError` on
> `external/osm_facilities`; re-running P1b found no `external/` at any of its
> five roots. The diagnostic shows why:
>
> ```
> MyDrive/  ->  Firm_Clusters/  ->  pipeline, qc
> ```
>
> Those are **exactly the two directories the bootstrap cell wrote minutes
> earlier**. A Drive holding 1.50 GB of `external/` plus 5,337 daily cluster
> files would not look like this. Drive is mounted; Colab is simply attached to
> a *different* Google account.
>
> **Nothing has been lost.** The real tree, the 5,337 cluster parquet files and
> `canonical_event_base.parquet` (7,265,849 rows) are intact in the other
> account.
>
> `ValueError: Mountpoint must not already contain files` on
> `force_remount=True` is **not** corruption — Colab simply refuses to remount a
> non-empty mountpoint, and `/content/drive` is non-empty because the bootstrap
> cell wrote `Firm_Clusters/{qc,pipeline}` into it.
>
> ⚠️ **Do not "fix" this by deleting `/content/drive/*`.** Those directories may
> be the stub in the wrong account, or they may be the only copy of
> `canonical_event_base.parquet` and the 5,337 cluster files.
>
> Run **`colab_p3a_remount_drive.py`**. It mounts a *fresh probe directory*, so
> nothing existing is touched, and reports:
>
> | Probe verdict | Meaning | Action |
> |---|---|---|
> | **GOOD** | probe sees `external/` + the event base | clear the stub, mount `/content/drive`, re-verify |
> | **BAD** | probe shows the same near-empty Drive | the runtime is on the wrong account — see below |
> | **PARTIAL** | some data, not all | check shared drives, then re-run the diagnostic |
>
> If the verdict is BAD, no file operation can help; the credential must change.
> **Verified locally:** there is no copy of this data on this machine (no
> `India_Firms.parquet`, `canonical_event_base.parquet`, `full_2012_2026/` or
> `Firm_Clusters/` anywhere under `~`, and no `rclone`/`gcloud`), so the other
> account is the only place it exists.
>
> 1. Open a **private/incognito** window → `drive.google.com`. Note which
>    account shows `Firm_Clusters/full_2012_2026`. Do this *before* changing
>    anything in Colab.
> 2. Colab toolbar → folder icon → **Sign out** → sign in as that account.
> 3. **Runtime → Restart session.** Required — the Drive credential is cached
>    per runtime, so the restart is what actually swaps the account.
> 4. Re-run `colab_p3a_remount_drive.py`, then
>    `colab_p3a_diagnose_external.py` for `VERDICT: OK`.
>
> ⚠️ Do **not** run `drive.mount('/content/drive', force_remount=True)` at this
> point — the mountpoint holds the stub and is non-empty, so it raises
> `ValueError`. And do not delete `/content/drive/*` until you know which
> account you are on.
>
> ⚠️ **If you cannot get into the other account, stop and say so** rather than
> re-running anything. The recovery would be recovering the Drive, not the
> pipeline.
>
> Deleting the stub afterwards leaves `/content/feature_ledger.py` and
> `/content/industrial_features.py` untouched — they live outside `/content/drive`.
>
> ⚠️ **Do NOT re-run `build_full_global_events()`.** The clustering is intact in
> the other account; regenerating it would be slow, expensive, and would break
> the frozen regression anchors in §2.
>
> **Do not trust P1b coverage numbers from a failed run.** The re-run that
> reported "125 must call Earth Engine" and "0 local match" was operating on an
> empty Drive, so every token matcher failed. Those figures are meaningless.
>
> Cell order: `bootstrap → colab_p3a_remount_drive.py → colab_p3a_diagnose_external.py → colab_p3a_industrial.py`



`extract_industrial_features` **never existed** in any of `Stage1–Stage6.ipynb`,
so this is a new implementation, not a recovery. It is therefore recorded as
`NEW_IMPLEMENTATION` (an 8th status added to the vocabulary), never as
`COMPUTED_HISTORICAL`.

| File | Role |
|---|---|
| `industrial_features.py` | Pure computation core. No Drive paths, no I/O, fully unit-tested |
| `test_industrial_features.py` | **48 tests** on synthetic geometry |
| `colab_p3a_industrial.py` | Drive driver: gazetteer mapping, projection, streaming loop |
| `colab_p3a_diagnose_external.py` | Read-only: tells A/B/C/D apart when the external tree is missing |

### How the 18 are computed

```
10 × DIST_<CATEGORY>_LOG      cKDTree per category, log10(1 + d_metres)
 1 × DIST_NEAREST_FACILITY_LOG cKDTree over the union of all categories
 3 × FACILITY_COUNT_{1,5,10}KM  count of facility sites within radius
 1 × INDUSTRIAL_DENSITY_CONTRAST log10(local / surrounding-ring), 16 bearings
 1 × INSIDE_FACILITY           STRtree point-in-polygon
 1 × DIST_FACILITY_BOUNDARY_LOG  nearest sampled boundary ring
 1 × KNOWN_FLARE               persistence + stationarity rule (below)
```

Decisions encoded in the code, all documented in the docstrings:

- **Projection is mandatory.** `compute_industrial_features(..., crs=...)`
  refuses to run without an explicit CRS and rejects anything but EPSG:6933.
  A degree-valued distance is wrong in a way no downstream check would catch.
- **Missing category → NaN, not 0.** An absent gazetteer entry ("no mines were
  downloaded") is a different claim from "no mine within 10 km", and only the
  second one is a measured zero. The 3 count features keep 0 for empty space.
- **`log10(1 + d)`** so a 0 m distance is exactly 0, not `-inf`.
- **`KNOWN_FLARE`** requires all five of: inside a flare-capable facility, ≥200
  active days, detected on ≥25% of those days, ≤2 km centroid drift, FRP
  CV ≤ 0.60. Missing evidence yields NaN, never a bare 0.
- **`INDUSTRIAL_DENSITY_CONTRAST`** is local-vs-regional (10 km local ring vs a
  30 km surrounding ring sampled at 16 bearings), so a uniformly industrial
  region scores ~0 while a lone plant in empty country scores high.
- **`DIST_FACILITY_BOUNDARY_LOG`** samples polygon rings every 50 m and uses a
  cKDTree, bounding the error at 25 m outside the polygon. Inside, the distance
  is 0 from `INSIDE_FACILITY` regardless, so the approximation is harmless.

### The cell is safe by default

`INSPECT_ONLY = True` stops after reading the gazetteer and prints the resolved
category mapping plus the real event-table schema. Nothing is written. This
matters because the 37 files are named after OSM/GEM categories, not the 10
ledger categories, and `industrial.geoparquet` very likely *contains* the
members of `cement`, `steel`, `lng` and so on. A wrong mapping silently
corrupts both the features and the labels, so it must be eyeballed once.

Events are streamed with `ParquetFile.iter_batches(CHUNK_ROWS=200_000)` — the
7,265,849 rows are never materialised in memory. Set `MAX_EVENTS = 50_000` for a
smoke test before committing to the full run.

---

## 9. Feature pass results (both complete)

```
P4a tier 1   7,265,849 events / 7,265,849 (100.0%)   34 features
             7,589,305 cluster rows read, 0 unmatched
P3a industrial   7,265,849 rows written               18 features
```

### Finding 1 — 96.7% of events are single-cluster

`7,589,305` clusters across `7,265,849` events is **1.044 clusters per
event**, and the NaN table confirms it: every feature needing ≥2 clusters is
96.7% NaN, and every one needing ≥4 is 99.8% NaN.

| Tier-1 feature | NaN | Why |
|---|---|---|
| FRP_MEDIAN / P90 / MAX / SUM_LOG / IQR / PEAK_MEDIAN_RATIO | **0.0%** | one cluster suffices |
| DURATION_HOURS_LOG, DETECTION_COUNT_LOG, OVERPASS_DETECTION_RATIO | **0.0%** | one cluster suffices |
| TIME_TO_PEAK, FRP_PERSISTENCE, PEAK_HOUR_SIN/COS | **0.0%** | defined from one observation |
| STATIONARITY_INDEX | **0.0%** | a lone point is maximally stationary |
| FRP_CV, SPREAD_*, CENTROID_DISPLACEMENT_* | 96.7% | need ≥2 clusters |
| FRP_SLOPE, FRP_DECAY_RATE, FRP_AUTOCORRELATION | 99.8% | need ≥4 clusters |
| AREA_GROWTH_RATE, COMPACTNESS, SOLIDITY, NEW_PIXEL_FRACTION | 99.97% | need ≥3 non-collinear points |

Only **~239,000 events (3.3%) span two or more days.** This is not a bug — it
is the real shape of the data, and it is the single most important fact for
modelling. Consequences:

* **15 of the 34 tier-1 features carry all the signal.** They are the
  workhorses: FRP distribution, counts, duration, persistence, peak hour.
* 19 features are near-useless for 97% of rows. They are not wrong, but a
  model can only exploit them on the ~239k multi-day events — which are, not
  coincidentally, precisely the persistent industrial ones worth detecting.
* LightGBM handles NaN natively, so this is not a blocker. It does mean the
  headline result rests on the 15 dense features.
* `DETECTION_DENSITY` at 63.8% NaN is legitimate: a single-detection cluster
  has zero radius, so the density is undefined rather than infinite.

### Finding 2 — polygons were invisible to the distance features

`DIST_PETROCHEMICAL_LOG` came back **100% NaN**, and the cause is a real bug:
the per-category trees were built only from *point* records, so every polygon
was excluded.

| Category | points | polygons | effect of the bug |
|---|---|---|---|
| petrochemical | 0 | 28 | 100% NaN — feature dead |
| mine | 425 | 10,886 | `DIST_MINE` wrong for 96% of mines |
| refinery | 2 | 35 | `DIST_REFINERY` used 2 of 37 |
| landfill | 658 | 1,976 | `DIST_LANDFILL` under-counted |
| pipeline | 15 | 25 | `DIST_PIPELINE` used 15 of 40 |

Polygons now contribute their **centroid** as a representative point in the
`DIST_*` and `FACILITY_COUNT_*` trees, while the densified rings still drive
`DIST_FACILITY_BOUNDARY_LOG` and the original geometry still drives
`INSIDE_FACILITY`. The first P3a run must be discarded and re-run.

`KNOWN_FLARE` is 100% NaN, which is correct and expected: the event base
carries none of `active_days`, `detection_count`, `centroid_displacement_m` or
`frp_cv`. Those all come from tier 1, so `KNOWN_FLARE` is computed downstream
rather than here.

---

## 10. Feature coverage against the brief — the honest picture

**56 of 141 features are computed.** 85 are not, and the gap is not spread
evenly across the claims the project makes.

| Computed | n | Status |
|---|---|---|
| TERRAIN | 12 | cell 263, needs an Earth Engine run |
| INDUSTRIAL | 18 | done; label-derived, so excluded from model B |
| Lineage tier 1 | 34 | done |
| Derived | 4 | `SITE_FRP_BASELINE`, `SITE_N_EVENTS`, `FRP_ANOMALY_VS_SITE`, `MISSING_FEATURE_FRACTION` |

### Each claim against the evidence

| Claim from the brief | Verdict | Blocked by |
|---|---|---|
| detect a thermal anomaly | **supported** | — |
| site-relative scoring (34 MW → 390 MW, score 5.9 vs 3.5) | **supported** | — |
| cloud gaps shown as gaps, never zeros | **supported** | — |
| per-class precision / recall / F1 | **partial** | only `INDUSTRIAL_FLARE` is labelable today |
| "likely cause" | **not yet** | land cover 16 + weather: `BARE_MINING_FRAC`, `NBR_CHANGE`, `VPD_ANOMALY_Z` |
| "likely spreading direction" | **not yet** | `SPREAD_WIND_SPEED_RATIO` unimplemented, era5 unverified |
| "the polygon" | **not yet** | hull metrics exist (`ELONGATION`, `COMPACTNESS`, `SOLIDITY`, `max_radius_km`) but no stored geometry |

`SPREAD_DIRECTION_SIN/COS` records where the fire **has been**, not where it
**will go**. Predicting the latter needs wind, and the wind field is exactly one
of the 85 missing features.

### The 85 missing features, in value order

| Block | n | Cost | Unblocks |
|---|---|---|---|
| **LANDCOVER_FUEL** | 16 | one Earth Engine run | cause attribution **and** the MINING + INDUSTRIAL_FIRE labels |
| Weather (FWI 7 + era5 17) | 18 | verify era5, then code | spread direction, `VPD_ANOMALY_Z`, WILDFIRE labels |
| Recurrence | 4 | cross-event pass, no external data | cheap; 4 features |
| SAR | 7 | one Earth Engine run | structural change |
| Smoke + atmospheric | 16 | re-download `sentinel5p` (444 B) | plume direction, AOD |

**LANDCOVER_FUEL is the highest-value single move.** One Earth Engine run for 16
features that simultaneously enable the "likely cause" claim and two of the six
label classes. Until then a 56-feature model is a defensible anomaly detector,
not the full product.

---
## 11. Where the pipeline stands (2026-10-03, after P6)

`7,265,849` events × 56 features assembled. Weak labels now populate:

| Class | Count | Share |
|---|---|---|
| INDUSTRIAL_FIRE | 1,119,254 | 15.40% |
| WILDFIRE | 178,581 | 2.46% |
| MINING | 73,060 | 1.01% |
| INDUSTRIAL_FLARE | 3,571 | 0.05% |
| AGRICULTURAL | **0** | 0.00% |
| UNKNOWN | 5,891,383 | 81.08% |

`INDUSTRIAL_FIRE` at 1,119,254 matches the 1,122,726 predicted from the
per-condition report, which validates the whole diagnostic chain. Three fixes are
confirmed live: `INSIDE_FACILITY` no longer gates the flare rule (3,571 flares),
persistence is no longer constant (178,581 wildfires reach score ≥ 3), and
`SITE_DEG = 1.0` populates `FRP_ANOMALY_VS_SITE` fully.

### AGRICULTURAL is still 0 — an ordering constraint

It requires `FRP_PERSISTENCE < 0.5`. **P6 never re-executes `cluster_tier.py`** —
it re-reads the tier-1 chunks already on Drive, which were written *before* the
persistence fix and still hold the constant 1.0.

**P4a must be re-run** to rewrite those chunks. Re-running P6 cannot substitute.

### Two caveats that attach to every number above

1. **`label_weak` is manufactured from features the model also sees.** PR-AUC on
   it is an upper bound, not generalisation. Excluding rule inputs from model B
   protects the scientific partition; it does not make the weak labels
   themselves trustworthy.
2. **The strict `label` column is still 100% UNKNOWN**, which is correct — it
   needs land cover and weather. The gap table names the blockers:
   `BARE_MINING_FRACTION`, `NBR_CHANGE`, `NBR_RECOVERY`, `SEASON_FRACTION`,
   `BT_I4_MEDIAN`, `VPD_ANOMALY_Z`, `WIND_SPEED`, `NIGHT_DAY_FRP_RATIO`,
   `area_km2`.

---

## 12. Rule calibration is now automatic

`FRP_PERSISTENCE` moved from a constant 1.0 to `observed_days / 30`, which is a
change of **scale**, not just variance. The thresholds written against the old
scale became far too strict in one re-run:

| Class | before the fix | after | cause |
|---|---|---|---|
| INDUSTRIAL_FLARE | 3,571 | **0** | `persistence >= 0.60` now needs an 18-day event |
| MINING | 73,060 | **61** | `persistence >= 0.30` now needs a 9-day event |
| AGRICULTURAL | 0 | 3,939,494 | `persistence < 0.50` became reachable |
| INDUSTRIAL_FIRE | 1,119,254 | 434,805 | `count >= 2` is now the binding branch |

**P6 is now three phases.** The thresholds are derived from the catalogue rather
than hardcoded:

```
PHASE 1  shard all three inputs into 16 partitions
PHASE 2  write X and meta; sample 200,000 rows of rule inputs
PHASE 3  calibrate thresholds from the sample, THEN label every partition
PHASE 4  delete the shards
```

`rule_calibration.py` bisects each persistence threshold until its class lands
near `CALIB_TARGETS` (flare 3,600 · industrial 400k · mining 60k), and prints
both the before and after distributions so the change is visible. Calibrated
values are written to `qc/p6_rule_calibration.json` and into the manifest, so
the labels are reproducible and the thresholds are auditable.

Two defaults matter: the percentiles must sit **above** the 96.7%
single-cluster mass, or every rule collapses onto the persistence floor. A test
asserts that.

**If the feature distribution moves, re-run P6. Do not hand-edit thresholds.**

## 13. Why 35 of 56 columns are populated — and what actually improves it

The 21 columns that are >50% NaN are **not missing data**. They are undefined
by construction:

| Block | dense | sparse | why the sparse ones are empty |
|---|---|---|---|
| tier 1 | 15 | **19** | every one is a trend or geometry statistic needing ≥2–4 clusters; **96.7% of events have exactly 1** |
| industrial | 17 | 1 | `KNOWN_FLARE` — no flare catalogue has been joined |
| derived | 4 | 0 | — |

A one-cluster event has no FRP slope, no autocorrelation, no convex hull and no
elongation. **No honest computation can fill them**, and imputing values would
mean the model could not tell an invented measurement from a real one.

### The three improvements that are real

**1. Regime flags (done — now in X, 4 extra columns).** The model still needs to
know which regime an event is in. Instead of leaving it to infer that from a
pattern of NaNs, X now carries:

```
N_DETECTIONS            exact count
IS_SINGLE_DETECTION     1 if one cluster
N_DETECTIONS_GE3        1 if three or more
PERSISTENCE_IS_VACUOUS  1 when a single-detection event scores persistence 1.0
```

The last one matters most: a one-detection event is "seen every day of its
span" *vacuously*, so `FRP_PERSISTENCE = 1.0` carries no evidence of
persistence at all, and the rules should discount it.

These are model-only derived columns, deliberately **not** added to the frozen
141-feature ledger.

**2. The 12 derivable features** (no download, ~2 h). `India_Firms` joined
through `mappings/`: recurrence ×4, `NIGHT_DAY_FRP_RATIO`, `FIRMS_CONFIDENCE`,
`SCAN_ANGLE_*`, `VIEW_ZENITH_*`, `BT_I4_MEDIAN`. The recurrence block in
particular needs a cross-event pass and needs no external data at all.

**3. External blocks, in value order.**

| Source | n | Unlocks |
|---|---|---|
| **Land cover** (WorldCover/MODIS) | 16 | `BARE_MINING_FRAC`, `NBR_CHANGE` → cause attribution + 2 label classes |
| **Weather** (ERA5-Land/CHIRPS) | 17 | `VPD_ANOMALY_Z` → forward spread direction |
| **VIIRS Nightfire / GGFR** | ~2 | `KNOWN_FLARE`, currently 100% NaN |
| DEM / nightlights / population | ~18 | context |
| Sentinel-5P | 16 | must re-download: **444 B** on Drive |
| SAR | 7 | structural change |

### What will not help

Trying to make the 20 sparse columns denser. They are undefined, not missing.
The only way to change them is to change the unit of analysis — per-detection
geometry rather than per-cluster — and even then a single-detection event still
has one point.

## 14. P7 memory and the combined split

**The RAM crash was a P7 design fault.** The first version loaded all
7,265,849 × 60 as float64 and handed the frame to LightGBM, which copies it
into its own binary matrix — roughly 3.5 GB each, ~7 GB total, plus the
pandas intermediates. Three fixes:

| Change | Effect |
|---|---|
| read only the columns used | fewer intermediates |
| `astype("float32")` | halves bytes; LightGBM's native type anyway |
| stratified sample: 1.5M train / 600k test | bounds the working set |

Sampling is **stratified on the target class** so the 0.05% minority is not
swamped, and the sample size is printed so a metric is never mistaken for a
full-catalogue result. `NaN` survives the float32 cast, which matters: NaN is how
"could not measure" is encoded.

**The `assert_no_time_leakage` failure was a wrong assertion, not a wrong
split.** The combined split deliberately trains on late events from
*non-held-out* blocks — its guarantee is spatial, not temporal — so the temporal
assertion rejected a correct split. It now applies only to the out-of-time split,
and the cell prints which checks apply:

```
spatial leakage check: PASS (no shared block)
temporal check: N/A -- the combined split trains on late events
  from non-held-out blocks by design; its guarantee is spatial.
```

Measured on the real catalogue: train 5,772,935 · test 888,156 · 604,758
dropped (early events inside held-out blocks).

### Model B's 0.527 was an artefact of a stale module
That run reported **42** model-B features. The arithmetic identifies the cause
exactly: `60 available − 18 industrial = 42`, i.e. the partition was
`NON_LEAK − industrial` from a **pre-P5b `training_design.py`**. The current
version excludes the union of the industrial 18 and the 16 weak-rule inputs, so
model B is **29** features.
The six that leaked in are exactly the INDUSTRIAL_FIRE rule's own `shape` branch:
```
FRP_PERSISTENCE  DETECTION_COUNT_LOG  TIME_TO_PEAK
AREA_GROWTH_RATE  SPREAD_SPEED_MEDIAN  DIRECTION_CONSISTENCY
```
So B was partly learning the rule, and its 0.527 means nothing. P7 now asserts
the partition and **refuses to run**, naming the leaking columns:
```
STALE training_design.py or weak_labels.py: these weak-rule inputs are in
model B: [...]. Re-upload training_design.py and weak_labels.py, then re-run.
```
### P7 memory: sample during the read, not after
Both earlier OOMs shared one cause: the full 7,265,849 × 60 table was
materialised **before** sampling. Sampling cannot reclaim memory already spent.
| | Before | After |
|---|---|---|
| X | full table, 1.62 GB | streamed row group by row group |
| y | all 9 columns incl. 4 strings | `label_weak`, `label`, `sample_weight` only |
| meta | all 7 columns incl. 3 strings | split keys only |
| peak resident | ~4.3 GB raw, ~9 GB with object arrays | **0.47 GB** |
| peak row group | — | 17 MB |
Verified the streaming index arithmetic against synthetic row groups: 2,100,000
requested rows filled exactly, with train/test mapping back correctly.
### Session restarts wipe `/content`
After a crash, `/content` is empty, so every module must be re-uploaded and
re-imported before the next cell — otherwise a cell silently falls back to a
Drive copy that may be stale. That is how the 42-feature run happened after a
restart: the uploaded `training_design.py` predated the P5b change.
## 15. Correct first result, and what it means

```
partition check : model A 53   model B 29   (rule inputs excluded: 24)
streaming peak  : 0.34 GB  (the 7.27M-row catalogue was never resident)

model A (operational)  PR-AUC 0.9963   F1 0.9978   recall@0.60 1.000
model B (scientific)  PR-AUC 0.1721   F1 0.0341   recall@0.60 0.022
baseline (250 m)      precision 0.152   recall 1.000  F1 0.264
```

**Model A's 0.9963 is circular, not excellent.** It has `DIST_REFINERY_LOG` and
friends, which *are* the INDUSTRIAL_FIRE rule's conditions. It learned the rule
almost perfectly — which is exactly what it should do, and exactly why it is
not evidence.

**Model B's 0.1721 is the honest scientific number, and it loses to the
baseline's F1 of 0.264.** With all 24 proximity and rule-input columns removed,
the remaining thermal and temporal features cannot identify facility-proximity
fires in held-out regions.

Model B's own feature importance explains why: `PEAK_HOUR_COS` and
`PEAK_HOUR_SIN` dominate, then `DETECTION_DENSITY` and `FRP_CV`. That is
**satellite overpass timing**, not fire behaviour. VIIRS overpasses cluster at
particular local times, and anything correlated with acquisition hour will look
predictive regardless of cause.

So the first trained model says: *with the features currently computed, and
with the leakage removed, the industrial-fire class is not predictable in
unseen regions.* That is a real, publishable negative result — and it tells you
exactly which blocks to build next.

### Models are now persisted

`qc/models/model_{A,B}.pkl`, each storing the booster, **its feature list**, the
target, the split kind and the review threshold. The feature list is not
optional: a booster maps column *positions*, so reloading one against a
reordered feature frame would silently score garbage.

```
payload = pickle.load(open("qc/models/model_B.pkl", "rb"))
prob    = payload["booster"].predict(X[payload["features"]])
```

## 16. Backend: real model serving

The FastAPI backend existed but **every endpoint returned hardcoded sample
values**. Added a genuine inference layer that loads the P7 artefacts.

| File | Role |
|---|---|
| `backend/app/services/model_registry.py` | loads `qc/models/model_{A,B}.pkl`, caches, validates, serves |
| `backend/app/schemas/schemas.py` | `PredictRequest` / `PredictResponse` / `ModelCard` / `ModelsStatus` |
| `backend/app/api/routes.py` | `POST /api/predict`, `GET /api/models`, `GET /api/models/{name}` |
| `backend/app/core/config.py` | `MODEL_DIR`, `SERVING_MODEL_DEFAULT` |
| `backend/tests/test_model_registry.py` | 22 tests |

### Three properties that fail silently if skipped

1. **The feature list is part of the model.** A booster maps column
   *positions*, so a reordered frame scores confidently and wrongly. The stored
   list is applied, never inferred from the caller's dict, and the caller's dict
   order is irrelevant. Pinned by a test.
2. **Absent features become NaN, never 0.** NaN is how "could not measure" is
   encoded throughout the pipeline. A zero would assert "measured, and it was
   zero".
3. **Unknown columns are refused (422).** A feature the model never saw cannot
   be passed in silently.

A missing model returns **503 with the reason**, not a fabricated score.

### Defaults

`SERVING_MODEL_DEFAULT = "B"`. Model A is reachable by request but is **not** the
default, because its 0.9963 PR-AUC is circular. `/api/models/{name}` carries a
`caveats` field naming that for both models, so a caller cannot surface A's
number without also surfacing why it is meaningless.

```
health   : {'status': 'healthy'}
models   : 200 {'models_found': ['A','B'], 'serving_default': 'B'}
card B   : 200 n_feat=29
high     : 200 p=0.9  review=False
low      : 200 p=0.05 review=True  missing=['DETECTION_COUNT_LOG','DIST_REFINERY_LOG']
bad col  : 422 model B was not trained on 1 of these columns
no model : 404 no artefact at ...
```

### Startup no longer depends on the database

An unreachable `DATABASE_URL` previously raised out of `create_all()` and killed
the process, so a dead Postgres took **inference** down even though inference
needs no database. `lifespan` now loads the model registry first, then attempts
the database and continues in degraded mode if it fails. The FIRMS poller is
skipped without one, since it writes to the DB.

```
health : {'status': 'healthy', 'database': False, 'models': ['B'],
         'notes': ['database unavailable: DB-backed endpoints will error,
                   model serving is unaffected']}
predict: 200
```

Verified with a deliberately broken `DATABASE_URL`: the server boots and serves.
`/health` reports `database` separately from `status`, because a single
healthy/unhealthy flag hides which half is broken.

Local development uses `sqlite:///./agninetra_local.db` (a 90 KB database is
already in the repo). Point `DATABASE_URL` back at Postgres/PostGIS once a server
is actually running.

### Model artefacts live in VS Code at `models/`

Written by `colab_p7_train.py` to Drive and **not committed** (a `.pkl` executes
code on load). Fetch them with:

```bash
python3 scripts/fetch_models.py              # from a mounted Drive
python3 scripts/fetch_models.py --verify-only
```

`--verify-only` *loads* each artefact and checks its contract rather than just
confirming the bytes exist, because a pickle made in Colab that will not
unpickle locally is not a usable model.

**macOS needs OpenMP before LightGBM will import at all:**

```
OSError: dlopen(...lib_lightgbm.dylib): Library not loaded: @rpath/libomp.dylib
```
```bash
brew install libomp && python3 -m pip install -U lightgbm
```

This is the most common reason a Colab-trained model will not load locally.

### Still mock

`/events`, `/features`, `/shap`, `/plume`, `/forecast`, `/spread`,
`/exposure` and `/similar` still return hardcoded sample values. `/shap` in
particular returns *fixed* contributions regardless of input. Wiring those to
real features and a real SHAP computation is the next piece of work — not
something to fake now.


## 17. How to check model accuracy

**Do not use accuracy.** On the held-out sample, 501,648 of 600,000 rows are
negative, so a model that predicts *everything negative* scores **83.6% accuracy
while detecting nothing**:

```
test rows        600,000
positives         98,352 (16.39%)

-- FLOOR: the trivial predictor --
  accuracy 0.8361  precision 0.0000  recall 0.0000
```

Model B's reported `OTHER_CLASS` F1 of 0.911 is the same trap: it called almost
nothing positive, so the dominant class looked near-perfect while the model found
essentially no industrial fire. **That is not "91% accurate".**

The predictions must exist first. `colab_p7_train.py` writes them to Drive:

```
Firm_Clusters/qc/heldout_pred_B.csv   model B (default)
Firm_Clusters/qc/heldout_pred_A.csv   model A
```

```bash
python3 scripts/evaluate_model.py --scores heldout_pred_B.csv
#   y_true   0/1   target vs rest, from the WEAK label
#   score    0..1  model probability
#   dist_m   metres distance to nearest facility
```

`y_true` in that file is the weak label, not ground truth. It is named
`y_true_weak_label` alongside it so it cannot later be mistaken for one.

| Block | Why it exists |
|---|---|
| trivial floor | the score a model that learned nothing gets |
| operating point | fire caught at precision >= 0.60 — the deployable number |
| PR-AUC + **95% bootstrap CI** | a bare PR-AUC on 33 flare positives has a huge interval |
| distance baseline | what the model must beat |
| **like-for-like** | the model at its *own* best F1, not pinned at 0.5 |
| verdict | allowed to be negative |

### The comparison must be like-for-like

An earlier version tuned the baseline over five radii but judged the model at a
fixed 0.5 — handing the baseline its best operating point while denying the model
one. It reported "does NOT beat baseline" for a model that genuinely lost *and*
for a strong synthetic model that beat it. Both are now measured at their own
optimum, so the verdict is earned.

### Your actual numbers

```
model B : PR-AUC 0.1721   F1 0.0341   recall@prec>=0.6 0.0223
baseline: F1 0.2640   ->  ML does NOT beat the baseline
```

That is the honest state, and it is why **LANDCOVER_FUEL is next**: model B is
missing the only information that distinguishes industrial from wild fire.


## 18. Explainability: SHAP and LIME

The backend's `/events/{id}/shap` called `precompute_shap_explanations`, which
returns **the same fixed contributions whatever the event**. An explanation that
does not vary with its input is not an explanation, and serving one next to a
real prediction is worse than serving nothing.

| File | Role |
|---|---|
| `explainability.py` | SHAP `TreeExplainer`, LIME, family aggregation, method comparison |
| `backend/app/api/routes.py` | `POST /api/events/{event_id}/explain` |
| `test_explainability.py` | 29 tests |

```bash
curl -X POST localhost:8000/api/events/E123/explain \
  -H 'content-type: application/json' \
  -d '{"features":{"FRP_MEDIAN":40.2,"DETECTION_COUNT_LOG":1.6},
       "model":"B","method":"both"}'
```

`method` is `shap` (exact, deterministic), `lime` (local, stochastic) or
`both`. Both is the default, because disagreement is the finding.

### Why both

SHAP is exact and deterministic for trees, so it is the one to trust per event.
LIME is a **control**: if the two disagree on which features matter, the ranking
is not stable and should not be published. The response carries
`stability.jaccard` and an `agreement` verdict.

### Four things that would have produced wrong numbers

1. **NaN is not zero.** Both methods refuse a row with unmeasured features
   rather than substituting 0, which would let an explanation credit a feature
   nobody measured. LIME additionally cannot work at all — it resamples observed
   data.
2. **Attribution is in log-odds, prediction is a probability.** `base + sum(phi)`
   is a margin; comparing it to `booster.predict` gives a residual of **2.3** on a
   perfectly correct explanation. Both are returned with their space named, and
   `additivity_residual` is the check that belongs — it comes out at **1e-16**.
3. **Ranking is by |value|**, so features that push the score *down* are still
   reported. A signed ranking hides every suppressive feature.
4. **Family aggregation keeps signs**, so positives cancelling inside a family
   net to zero instead of appearing influential.

### The caveat that matters most

SHAP explains the **model**, not the fire. Model B's attributions describe how
the thermal features reproduce the weak label; model A's largely restate "near a
facility", because that is what its label was. Both responses carry this in
`caveats` so it cannot be separated from the numbers.

Verified on a real booster: explanations differ between events, additivity is
exact, and SHAP/LIME agree on the features that actually drive the model.

---

## 19. Path to training

```
feature_ledger.py                 schema, status map, drift check (v1.2.0) — single source of truth
industrial_features.py            P3a industrial computation core (18 features)
fire_labels.py                    P5 label rule cascade (6 classes), thresholds in one dataclass
weak_labels.py                    P5b industrial-context hierarchy, explicitly WEAK, + leakage registry
feature_gap.py                    Phase 0: spec-vs-X gap, obtain-routes, site occupancy
dataset_builder.py                P6: location stripping, site baseline, missingness, review queue
lineage_features.py               Tier 2: detection-level lineage (from India_Firms)
training_design.py                P7: leakage partition, non-random splits, baselines, metrics
cluster_tier.py                   Tier 1: cluster-level lineage, 34 features, no detections needed
test_cluster_tier.py              39 tests (1 skipped: pyarrow is Colab-only)
test_training_design.py           42 tests
test_weak_labels.py                39 tests
rule_calibration.py                percentile/target-count threshold calibration
test_rule_calibration.py           13 tests
test_feature_gap.py               24 tests
test_colab_cells.py               12 tests: static checks on the driver cells
test_dataset_builder.py           50 tests
colab_utils.py                    dead-mount-safe Drive checks + project-root resolution
test_colab_root.py                41 tests: root resolution, artefacts, stale-cache recovery
colab_survey_tree.py              Colab: print the real Drive tree and pin artefact paths
colab_recon.py                    Colab: whole-project recon -> qc/project_recon.json
colab_p4a_tier1.py                 Colab: 34 features from clusters/ + membership (no detection join)
colab_bootstrap.py                THE bootstrap cell: installs every .py, no MODULES list
colab_p4a_lineage.py              Colab: stream 5,337 daily cluster files -> 54 features per event
build_feature_audit.py            local ledger builder, asserts Gate P1
test_feature_ledger.py            39 tests
test_industrial_features.py       48 tests
test_fire_labels.py               50 tests
test_lineage_features.py          69 tests
test_cluster_tier.py              39 tests (1 skipped: pyarrow is Colab-only)
test_training_design.py           42 tests
test_weak_labels.py                39 tests
rule_calibration.py                percentile/target-count threshold calibration
test_rule_calibration.py           13 tests
test_feature_gap.py               24 tests
test_colab_cells.py               12 tests: static checks on the driver cells
test_dataset_builder.py           50 tests
test_colab_utils.py               13 tests
colab_p1_ledger.py                Colab: module loader, version guard, drift check
colab_p1b_external_inventory.py   Colab: inventory external/, size QC, match to features
colab_p0_cell262.py               Colab: recover extract_fire_features
colab_p3a_industrial.py           Colab: gazetteer mapping + streaming 18 industrial features
colab_p3a_diagnose_external.py    Colab: read-only A/B/C/D verdict on a missing external tree
colab_p3a_remount_drive.py       Colab: non-destructive probe mount to find the real Drive
qc/                               feature_audit_141.csv, feature_status_summary.json,
                                  FEATURES_141.json, missing_features_by_phase.md
Stage1..Stage6.ipynb              clustering + production run (no feature extractor)
backend/ frontend/ react-frontend/ AGNI-NETRA + SPARC design docs
SYSTEM_ARCHITECTURE.md
```

### Verification

```bash
python3 -m unittest test_feature_ledger test_industrial_features \
       test_fire_labels test_lineage_features test_colab_utils \
       test_colab_root test_cluster_tier test_training_design \
       test_colab_cells test_dataset_builder \
       test_weak_labels test_training_design test_feature_gap   # 477
python3 build_feature_audit.py --outdir qc   # CSV locally; parquet in Colab
```

Local machine has no `pyarrow`, so the audit writes CSV only.

---



## 20. Change log

| Date | Phase | Change |
|---|---|---|
| 2026-10-02 | P0/P1 | Extractor forensics complete; cell 314 chosen as dispatcher, cell 325 as event-derived provider, cell 263 terrain, cell 264 weather. Six functions absent (65 features). FWI 7 confirmed hardcoded `NaN` in cell 264. `BURN_SCAR_GROWTH_RATE` confirmed schema-only. |
| 2026-10-02 | P1 | `feature_ledger.py` v1.2.0 frozen: 141 features, 11 groups, 7-value status vocabulary, `UNVERIFIED_HISTORICAL` added (43) to retract the earlier "computed" overclaim; `check_schema_drift` now rejects the erroneous 131-length list. `build_feature_audit.py` + 38 tests added; Gate P1 passed. |
| 2026-10-02 | P1 | GEE dependencies restored: `MODIS/061/MOD14A1`, `NASA/VIIRS/002/VNP14A1`, and `ee_timestamp()` replacing `ee.Date(str(...))` which threw `EEException: Date: Bad date/time`. `EVENT_00000002` verified: 1 VIIRS + 1 MODIS image, valid geometry. |
| 2026-10-02 | P1b | External inventory built and verified: 911 files, 213 subdirs, 1.50 GB. Two red flags recorded (`sentinel5p` 0.00 GB, `era5_land` 120 KB/file); `SIZE_SANITY_FLOOR_BYTES` added and suspect tokens discarded before feature matching. |
| 2026-10-02 | P1b | **Closed.** Upload blocker cleared with the HANDOFF §6 snippet. Inventory: 213 subdirs / 911 files / 1.50 GB. Coverage of the 129 open features: 100 on-disk, 41 no-input, 25 GEE-only. Wrote `qc/external_data_inventory.json` + `qc/external_coverage_matrix.csv`. |
| 2026-10-02 | P1b | **Correction.** The 100 "local-ready" figure is inflated by 33: the matcher accepted `era5`/`s5p` tokens from the two SUSPECT folders the size-sanity filter was meant to exclude. Honest local-ready = **67** (47 India_Firms, 18 industrial, + 41 no-input). The 33 are held at `UNVERIFIED_INPUT` pending an xarray/raster probe. |
| 2026-10-02 | P1b | **Blocker:** `files.upload()` returned `feature_ledger (2).py` → `KeyError`, so the new module never landed and Colab kept the stale one. Tolerant upload snippet (with a `hasattr(fl, "LOCAL_INPUT_HINTS")` guard) written into HANDOFF §6. Awaiting: re-upload → restart → re-run P1 → re-run P1b. |
| 2026-10-02 | P3a | `industrial_features.py` written: all 18 INDUSTRIAL features from the verified `osm_facilities` gazetteer. Projection made mandatory (`crs` argument, EPSG:6933 only); missing category → NaN rather than 0; `KNOWN_FLARE` given an explicit five-condition persistence/stationarity rule; `INDUSTRIAL_DENSITY_CONTRAST` defined as local-vs-regional. |
| 2026-10-02 | P3a | `test_industrial_features.py` added — 48 tests on synthetic geometry. Two real bugs caught and fixed: a wrong kwarg into `known_flare_flags`, and a projection guard that rejected legitimate `(0,0)` Equal Earth metres (now skipped as evidence-free, with `crs` validation carrying the real guarantee). |
| 2026-10-02 | P3a | `colab_p3a_industrial.py` written. `INSPECT_ONLY = True` by default: prints the resolved category map and the real event schema, writes nothing. Streams `canonical_event_base.parquet` via `iter_batches(200_000)` so the 7.26M rows are never held in memory; `MAX_EVENTS` for a smoke test. Fixed a real flaw during review: `INSIDE_FACILITY` now uses the original geometries, because rebuilding polygons from the densified ring samples yields invalid geometry. |
| 2026-10-02 | P3a | Ledger: added `NEW_IMPLEMENTATION` as an 8th status and flipped the 18 INDUSTRIAL features from `MISSING_IMPLEMENTATION` to it. They are explicitly **not** `COMPUTED_HISTORICAL` — no historical implementation exists. `runtime_verified` stays `False`. Gate P1 re-passed: 12 COMPUTED / 43 UNVERIFIED / 7 HIST_NAN / 18 NEW / 61 MISSING. |
| 2026-10-02 | P3a | Bootstrap cell hardened against **partial uploads**. The picker returned only `feature_ledger.py`, so the cell aborted *after* successfully installing the ledger. It now prints what the dialog returned, installs everything present, re-prompts for only the missing modules (up to twice), and tolerates the `(2)` suffix on the retry. Verified against four cases: partial-then-full, both-at-once, suffix-on-retry, and never-supplied. |
| 2026-10-02 | P3a | Bootstrap succeeded: `GATE P1 PASSED`, `INDUSTRIAL: {'NEW_IMPLEMENTATION': 18}`, ledger 1.2.0, both modules installed. |
| 2026-10-02 | P3a | Diagnostic re-run confirmed **case B** (wrong Google account). Checked the local machine for a fallback copy: none — no `India_Firms.parquet`, `canonical_event_base.parquet`, `full_2012_2026/` or `Firm_Clusters/` anywhere under `~`, and no `rclone`/`gcloud` installed. The other account is the sole location of the data. Corrected the diagnostic's own advice, which had recommended `force_remount=True` on a non-empty mountpoint (the same `ValueError`). |
| 2026-10-02 | P3a | `force_remount=True` failed with `ValueError: Mountpoint must not already contain files` — Colab refuses to remount a non-empty mountpoint, and the stub `Firm_Clusters/` made it non-empty. **Explicitly did not** solve this by deleting `/content/drive/*`, since those paths may hold the only copy of the 7.26M-row event base. Added `colab_p3a_remount_drive.py`, which mounts a *fresh probe directory* (non-destructive by construction) and reports GOOD / BAD / PARTIAL. All four branches tested. |
| 2026-10-02 | P3a | **Correction — my "wrong account" diagnosis was WRONG.** The probe returned `VERDICT: GOOD`: `external/` and `canonical_event_base.parquet` are both present, alongside `Anamoly.ipynb`, `Project2` and `X_event_feature_inventory` — clearly the real Drive. The account was correct all along; `/content/drive` was serving a **stale, near-empty mount view**. Lesson recorded: a mountpoint listing only this session's own writes indicates a stale mount, and the first move is `drive.unmount(...)` + remount, not an account switch. The `GOOD` branch worked as designed and was non-destructive. |
| 2026-10-02 | recon | Added `colab_recon.py`: a whole-project reconnaissance report covering every folder, column name, dtype, gazetteer geometry type/CRS, membership id format, and a ledger-driven availability verdict for all 141 features. Sampled by design — parquet **footer metadata** only for schemas, and exactly **one data read per distinct schema signature**, so 16,000 files cost ~5 data reads. Cluster/edge dates come from filenames rather than opening 5,337 files. Output: `qc/project_recon.json`, which supersedes every hardcoded path and column name in the handoff. |
| 2026-10-02 | recon | **Bug in `colab_recon.py` itself:** `schema_of()` used `pf.schema_arrow.schema`, which does not exist — `schema_arrow` *is* the Schema. Every schema lookup failed, so sections 2–5 reported "0 detection columns" and a bogus availability verdict implying the inputs were missing. They were present all along. Fixed, and the script now **raises** if any schema probe fails, because a failed probe reading as "no columns" could trigger a pointless re-download of data that already exists. |
| 2026-10-02 | recon | **The membership table has an explicit `date` column** (`event_id`, `local_cluster_id`, `20120120_0`, `2012-01-20`). P4a now prefers it over parsing `YYYYMMDD` out of the cluster id, since a dedicated column cannot drift from the id format; the id-parsing fallback is retained. |
| 2026-10-02 | recon | The earlier claim that `external/` shrank from 1.50 GB to 210.4 MB was **wrong** — that reading came from the same stale FUSE cache. The recon, run after a forced refresh, reproduces P1b's per-folder counts exactly (12/175/506/177/37/3/1). P1b's numbers stand; recorded here so the false alarm is not re-investigated. |
| 2026-10-02 | P3a | **Coordinate columns discovered, not guessed.** The handoff's `centroid_lon`/`centroid_lat` do not exist; the real names are `event_centroid_lon`/`_lat`, and `event_centroid_x`/`_y` are **already projected to Equal Earth**. P3a now auto-detects the pair and verifies by magnitude (>180 = metres) whether to skip projection. Projecting pre-projected coordinates would have rescaled every industrial distance. |
| 2026-10-02 | P3a | **The event base already holds cell-325-style aggregates** (`geometry_total_frp`, `geometry_max_frp`, `geometry_n_satellites`, `geometry_n_detections`, …) — the cell-325 output `HANDOFF` §4 recorded as never located. This is a P0/P2 opportunity: the P4a lineage block can be cross-checked against an independent pre-existing computation instead of trusted blindly. |
| 2026-10-02 | P3a | **Gazetteer duplicated.** `facility_categories/` and `facility_categories (1)/` hold the same filenames with different row counts, plus two copies each of `india_osm_facilities.geoparquet` (147,606 and 165,869 rows). Loading both double-counts facilities and biases `FACILITY_COUNT_*`. P3a now collapses same-named duplicates, prefers the unsuffixed directory, and prints every dropped file. |
| 2026-10-04 | P7 | **The FastAPI backend served real predictions for the first time.** Every existing endpoint returned hardcoded sample values. Added `model_registry.py` plus `POST /api/predict`, `GET /api/models`, `GET /api/models/{name}`, with 22 tests. Three silent-failure modes are enforced: the stored feature list is applied rather than the caller's dict order (a booster maps column *positions*), absent features become NaN and never 0, and unknown columns are refused with 422. A missing model returns 503 with the reason instead of a fabricated score. Default serving model is **B**, since A's 0.9963 is circular, and the model card carries the caveat so A's number cannot be shown without it. |
| 2026-10-04 | P7 | **Fixed: an unreachable `DATABASE_URL` killed startup, taking model serving with it.** `create_all()` raised inside `lifespan`, so a dead Postgres prevented a booster and a parquet file from being served. The registry now loads first, the database is attempted and a failure degrades instead of raising, and the FIRMS poller is skipped since it needs the DB. Verified with a deliberately broken URL: server boots, `predict` returns 200. `/health` now reports `database` separately from `status`. Local dev switched to the SQLite database already in the repo. |
| 2026-10-04 | P8 | **SHAP and LIME added, and the mock SHAP removed.** `precompute_shap_explanations` returned the SAME fixed contributions for every event; it is replaced by `POST /api/events/{id}/explain` backed by `explainability.py` (29 tests). Both methods are offered by default because disagreement is the finding, and the response carries a `stability.jaccard`. |
| 2026-10-04 | P8 | **Four silent-wrong-number risks closed.** NaN is never replaced with 0 (both methods refuse an incomplete row; LIME cannot work at all since it resamples observed data). SHAP attributions are log-odds while `predict` is a probability, so comparing them gave a residual of 2.3 on a correct explanation -- both are now returned with their space named and `additivity_residual` comes out at 1e-16. Ranking is by magnitude so suppressive features are not hidden. Family aggregation keeps signs so cancellations net out. |


| 2026-10-04 | P7 | **`IndexError` writing the held-out CSV.** The sampled matrices are train-then-test concatenated, and the new CSV block indexed that concatenation with a GLOBAL catalogue position, reaching past its end (`index 2154661 out of bounds for size 2100000`). Fixed by slicing per side (`_dist_train` / `_dist_test`), and pinned by a static guard that rejects any per-side lookup against a concatenated array -- verified by reintroducing both the original bug and the missing slice. |
| 2026-10-04 | P7 | **`scripts/evaluate_model.py` existed but had no input to read** — I gave a filename in an example that was never produced. `colab_p7_train.py` now writes `heldout_pred_{B,A}.csv` to Drive, and the script explains where they come from when the file is absent. Also fixed two real defects it exposed: the confusion line read a nested `confusion` key that does not exist, so it printed all zeros on every run; and a 0.0006 F1 gap was being declared a victory, now gated by a 1% noise floor that reports *indistinguishable* instead. |

| 2026-10-04 | P7 | **Fixed a reporting defect before publication.** The per-class table reported a synthetic class `NOT_INDUSTRIAL_FIRE`, which never appears in the labels, so its support printed as **0** and the row read as a class the model had never seen. Every non-target label is now folded into a real `OTHER_CLASS`, so its support is the true count, and the composition of that class is printed explicitly alongside a confusion line. Four tests pin it, including that no support-zero row can reappear. |
| 2026-10-04 | P7 | **Correct first result.** Model A PR-AUC 0.9963 is circular (it holds the rule's own `DIST_*` columns). Model B PR-AUC 0.1721 loses to the baseline's F1 0.264 -- with all 24 rule inputs removed, the remaining thermal/temporal features cannot identify facility-proximity fires in unseen regions. Model B's importance is dominated by `PEAK_HOUR_COS`/`PEAK_HOUR_SIN`, i.e. **satellite overpass timing**, not fire behaviour. This is a genuine negative result and it identifies the blocks to build next. |
| 2026-10-04 | P7 | **Models are now persisted** to `qc/models/model_{A,B}.pkl` with the booster, its feature list, the target, the split and the review threshold. Previously the fitted boosters died with the runtime. The feature list is stored because a booster maps column *positions* -- reloading against a reordered frame would silently score garbage. |
| 2026-10-04 | P7 | **Model B's PR-AUC 0.527 was an artefact of a STALE `training_design.py`.** It reported 42 features; `60 - 18 = 42` identifies the old `NON_LEAK - industrial` partition, which leaves six weak-rule inputs (`FRP_PERSISTENCE`, `DETECTION_COUNT_LOG`, `TIME_TO_PEAK`, `AREA_GROWTH_RATE`, `SPREAD_SPEED_MEDIAN`, `DIRECTION_CONSISTENCY`) inside the "scientific" model -- precisely the INDUSTRIAL_FIRE rule's own shape branch. Correct model B is **29** features. P7 now asserts the partition and refuses to run, naming the leaking columns. |
| 2026-10-04 | P7 | **Both OOMs had one cause: the full table was materialised before sampling.** Sampling cannot reclaim memory already spent. P7 now reads `y` and `meta` with only the needed columns and streams `X` one row group at a time, keeping only sampled rows: peak falls from ~4.3 GB raw (~9 GB with pandas object arrays) to **0.47 GB**, with a 17 MB peak row group. Verified the streaming index arithmetic against synthetic row groups. |
| 2026-10-04 | infra | **A crashed session wipes `/content`.** Every module must be re-uploaded after a restart, and a cell will otherwise silently import a stale Drive copy -- which is exactly how the 42-feature run happened after a restart. The partition assertion now turns that silent failure into a loud one. |
| 2026-10-04 | P7 | **First model trained. Results are real but neither is evidence of skill, for structural reasons.** Model A PR-AUC 0.4925 / model B 0.0001 / baseline F1 0.020. Model B is asked to predict a *proximity rule* using features from which all 18 proximity columns were deliberately removed, so there is nothing to learn; model A has `DIST_REFINERY_LOG`, which **is** the rule's first condition, so its score is circular. This is the deep consequence of labelling from your own features. |
| 2026-10-04 | P7 | **The target was too small.** `INDUSTRIAL_FLARE` has 3,571 events total: 894 train positives and **33 test positives**, so PR-AUC is not estimable, and both models early-stopped at iteration 3 of 75. `INDUSTRIAL_FIRE` (1,119,254) is the first target with enough data; `TARGET` is now a one-line switch. |
| 2026-10-04 | P7 | **Fixed a reporting defect the run exposed:** the per-class table claimed six classes for a binary model, reporting every non-target row as `UNKNOWN` — a class it never predicted — so five rows of zeros looked like failures. Now reports `TARGET` vs `NOT_<TARGET>` and states the limitation. |
| 2026-10-04 | QA | **`NameError: train_idx is not defined` was my edit, not the data.** An over-eager replace swapped the split masks for sample indices in the leakage assertion, 46 lines above where those indices are assigned. Reverted to `is_train`/`is_test` -- which is also more correct: sampling cannot affect leakage. |
| 2026-10-04 | QA | **New static guard: use-before-assignment**, because both the undefined-name check and the mutation check passed that bug cleanly -- the name exists, just too late. The module scan pre-seeds only what Python hoists (imports, def/class); function bodies still see all globals, since they run after the module finishes. Verified by reintroducing the bug. Adding `colab_p6_dataset.py` and `colab_p7_train.py` to the guarded list immediately found a **second latent `NameError`**: `MODE` read `INSPECT_ONLY` eight lines above its assignment in `colab_p3a_industrial.py`, which would have crashed the next P3a run. |
| 2026-10-04 | P7 | **P7 exhausted the runtime.** It loaded all 7,265,849 x 60 as float64, then LightGBM copied it into its own binary matrix: ~3.5 GB each. Now reads only the columns used, casts to float32, and trains on a stratified sample (1.5M train / 600k test) with the sample size printed so a metric is never read as a full-catalogue result. NaN survives the float32 cast, which matters because NaN is how "could not measure" is encoded. |
| 2026-10-04 | P7 | **`assert_no_time_leakage` was rejecting a correct split.** The combined out-of-time+out-of-space split deliberately trains on late events from *non-held-out* blocks -- its guarantee is spatial, not temporal -- so asserting "no test year in train" failed on a valid design with 3,417,935 false positives. The assertion is now scoped to the out-of-time split, and the cell states which checks apply and why. |
| 2026-10-04 | QA | **The static check missed a real bug because it only looked for `+=`.** `missing_pairs` was mutated via `.setdefault()`, which an `AugAssign` check cannot see, and it survived a full P6 run as a `NameError`. The check now catches every mutation form: `NAME +=`, `NAME[k] +=`, `NAME[k] =`, and `NAME.setdefault/append/add/update/extend/insert()`. Proven against a minimal reproduction. Lesson: a lint that only knows one syntax of a mistake is not a guard. |
| 2026-10-04 | P5b | **Label rebalance after the persistence fix: INDUSTRIAL_FLARE 3,571 -> 0 and MINING 73,060 -> 61, while AGRICULTURAL went 0 -> 3,939,494.** Not a new bug -- `FRP_PERSISTENCE` changed *scale*, and thresholds set against a constant 1.0 became far too strict. This is the cost of fixing a tautological feature, and it has to be paid by recalibrating. |
| 2026-10-04 | P5b | **`rule_calibration.py` added (13 tests) and wired into P6 as PHASE 3.** Thresholds are derived from the catalogue: a 200,000-row sample is taken during PHASE 2, then each persistence threshold is bisected until its class lands near a target count. Both the before and after distributions are printed, the calibrated values go to `qc/p6_rule_calibration.json` and the manifest, and re-calibrating on the same data reproduces identical thresholds. Default percentiles must sit above the 96.7% single-cluster mass or every rule collapses onto the floor -- asserted by a test. |
| 2026-10-04 | P6 | P6 restructured into four phases: shard, write X/meta + sample, calibrate then label, delete shards. `y` is written in PHASE 3 because calibration needs the whole distribution while the pass is partitioned. A row-count check fails loudly if the label pass and X disagree. |
| 2026-10-03 | P6 | **P6 completed: 7,265,849 events, weak labels now populate** — INDUSTRIAL_FIRE 1,119,254 (15.4%), WILDFIRE 178,581, MINING 73,060, INDUSTRIAL_FLARE 3,571, UNKNOWN 81.1%. The 1,119,254 matches the 1,122,726 predicted from the per-condition report, validating the diagnostic chain. Three fixes confirmed live. |
| 2026-10-03 | P6 | **AGRICULTURAL still 0, and the cause is an ordering constraint:** P6 re-reads the tier-1 chunks already on Drive, which were written *before* the persistence fix, so it still sees the constant 1.0. `cluster_tier.py` is only executed by P4a. **P4a must be re-run before AGRICULTURAL becomes reachable** — re-running P6 cannot substitute. |
| 2026-10-03 | P6 | Fixed a reporting bug: "events blocked" summed the per-string Counter, so an input noted by two rules was double-counted and reported 14,531,698 blocked for a 7,265,849-row table. Now counts distinct events per input and prints the percentage. |
| 2026-10-03 | P7 | P7 retargeted at `label_weak` (the strict column cannot train: it is ~100% UNKNOWN by design). It prints an explicit caveat that PR-AUC on weak labels is an **upper bound, not generalisation**, and records that caveat in `p7_results.json` so it cannot be separated from the numbers later. |
| 2026-10-03 | P6 | **P6 crashed with `NameError: weak_counts`** — the accumulator was incremented but never declared, and it only surfaced at partition 0 of the full 7.27M-event pass. Declared, and added a static check that every `NAME +=` inside a cell function is assigned in that same function, so this class of error is caught before a long run. |
| 2026-10-03 | P6 | Ran the whole assembly path on a synthetic tree to verify all four recent fixes together: weak labels produce **every class** (AGRICULTURAL 1371 now fires, INDUSTRIAL_FLARE 8 now fires), `FRP_ANOMALY_VS_SITE` is 99.9% populated, and `missing_inputs` names the five rule inputs that remain blocked (`BARE_MINING_FRACTION`, `NBR_CHANGE`, `NBR_RECOVERY`, `SEASON_FRACTION`, `NIGHT_DAY_FRP_RATIO`). |
| 2026-10-03 | infra | `colab_bootstrap.py` REQUIRED list expanded from 5 to 10 modules. `cluster_tier`, `weak_labels`, `dataset_builder`, `training_design` and `feature_gap` were absent, so a missing one surfaced as a bare `ImportError` inside a later cell rather than a clear "missing module" report at bootstrap. |
| 2026-10-03 | P0 | **Phase-0 run: `FRP_PERSISTENCE` was exactly 1.0 for all 7,265,849 events.** The definition `days/(span_days+1)` is a tautology — an event exists only while detected, so it is seen on every day of its own span. A constant column carries no information and it silently zeroed AGRICULTURAL, whose condition `FRP_PERSISTENCE < 0.5` could never hold. Now measured over a fixed 30-day window: 2-day event 0.067, 60-day event 1.000. |
| 2026-10-03 | P5b | **`INSIDE_FACILITY == 1` for zero of 7.27M events.** It point-in-polygons an event centroid against 38,370 small OSM polygons; centroids sit on a 1 km grid and almost never land inside. The flare rule wrongly required it, making the rule unsatisfiable. Dropped as a conjunct; `DIST_REFINERY_LOG <= 2000 m` (17,727 events) is the robust proxy. The feature stays in X as its own signal. |
| 2026-10-03 | P0 | **The 100% UNKNOWN was not unsatisfiable rules.** Measured: `INDUSTRIAL_FIRE` conjunction passes **1,122,726 (15.5%)** and `MINING` **147,962 (2.0%)**. Both were ready; P6 ran a stale cell so `label_weak` was never produced. |
| 2026-10-03 | P4a | **Site grid misread and corrected.** 0.1 deg is not too fine — 26,828 cells with a median of 134 events each means 99.99% of events already have company. The defect is the opposite: a baseline that local is effectively a regional average, so nothing looks anomalous. `SITE_DEG` is now **1.0 deg** (~110 km), which separates a refinery from its surroundings. Verified: anomaly column 100% populated, a 390 MW spike against a 34 MW baseline scores 11.5x. |
| 2026-10-03 | P0 | Corrected an earlier claim: model B has **29** features, not 7. The 7 came from a hand-picked 17-column list; the real 53-column `X` leaves 29 after excluding 18 industrial and 16 weak-rule inputs. |
| 2026-10-03 | P0 | Phase-0 gap run against the real X: **33 usable / 20 sparse / 88 missing, 23.4% coverage.** The 88 split **18 DERIVE / 46 EXTERNAL_RASTER / 24 EXTERNAL_TABLE**, but 6 of the 18 DERIVE features are permanently unavailable (no I5 band, no subpixel temperature, no source area, no pixel size), so the realisable count is **12**. Corrected a bug where `missing_by_route` keys were unpacked as nested pairs, which aborted the cell before it printed any of them. |
| 2026-10-03 | P0 | `feature_gap.py` + `colab_p0_gap.py`: the Phase-0 gap analysis. Every one of the 141 spec features is bucketed **PRESENT / SPARSE (>90% NaN, kept, never imputed) / MISSING**, and each missing feature is tagged DERIVE / EXTERNAL_RASTER / EXTERNAL_TABLE from the ledger's own `DATASET` declaration, so the tag cannot drift from the spec. 24 tests. |
| 2026-10-03 | P0 | **Explained `FRP_ANOMALY_VS_SITE` being 99.5% NaN.** 7.27M events over roughly 2.75M land cells at 0.1 deg means the *median* cell holds one event, so there is no company to establish a baseline against — it is arithmetic, not a bug. `site_occupancy()` and `recommend_site_deg()` now measure occupancy and recommend a grid, instead of leaving it as a mystery. |
| 2026-10-03 | P5b | `condition_pass_report()` — "every event is no_rule" is not a diagnosis. Each rule is a conjunction, so one absent term kills it silently while the rest look healthy. The report gives, per rule, how many events pass each condition, the conjunction, and the **weakest term named**, which is the feature to implement next. |
| 2026-10-03 | infra | `colab_p0_gap.py` prints module sizes at the top. A stale pasted cell is the most expensive failure mode in this project — it runs to completion and prints a plausible report while silently omitting whatever changed. Comparing the stamps against the repo makes that visible in one glance. |
| 2026-10-03 | P4a | `add_single_detection_flags()` adds `IS_SINGLE_DETECTION`, `N_DETECTIONS_GE3` and `PERSISTENCE_IS_VACUOUS` so the model can condition on the single-detection regime explicitly. It adds flags and never imputes the empty features: a fabricated value is indistinguishable from a measurement to a model. |
| 2026-10-03 | P5b | `weak_labels.py` implements the industrial-context rule hierarchy as an explicitly **weak** labeller. Two defects in the proposed rules were found and fixed: a **unit error** (`log10(1+d_metres)` compared against `np.log1p`, making every distance radius 6-25x too tight — 5 m silently meant 61 m and nothing could be labelled), and a rule that treated missing persistence as passing. |
| 2026-10-03 | P5b | **`FRP_PERSISTENCE` is ≈1.0 for nearly every event** — an event exists on exactly the days it was detected — so `persistence >= 0.35` is true for ~96.7% of rows and discriminates nothing. The `industrial_shape` condition consequently reduces to *"was this event near a facility"*. This is target leakage made concrete, and it is why the weak-rule inputs are now excluded from model B. |
| 2026-10-03 | P7 | **Leakage guard made mechanical.** `weak_labels.LABEL_INPUT_FEATURES` publishes every feature the rules read; `training_design.split_feature_sets()` excludes the 18 industrial features *plus* those inputs from model B and raises if any slips through. Model B drops from 141 to 117 on the full schema; on the real 56-feature block it drops to 7. A test asserts model B is a strict subset of the ledger's 123-feature non-leak partition. |
| 2026-10-03 | P5b | Bug in that module: `label_source` was a bare Python string, so `out["label_source"][0]` returned the character `'r'`, and `np.where` over string scalars yields a `<U` array that iterates per character. Both now use explicit object dtype. |
| 2026-10-02 | P6 | **First assembly labelled 100% UNKNOWN under `no_rule`,** which could not distinguish "threshold not met" from "input never computed". The engine now returns `missing_inputs` per event, naming the absent rule inputs, and P6 aggregates them into a *why labels abstain* table — the list of features to implement. Two fixes fell out: `n_satellites` was never read from the event base (it lives there as `geometry_n_satellites`), and `area_km2` is exactly invertible from `DETECTION_DENSITY`. |
| 2026-10-02 | P5 | Diagnostic bug worth recording: the first `missing_inputs` merge conflated the **rule index with the event index**, so a gap on rule 3 was never reported for event 0 — and on a single-row frame it silently produced nothing at all. Rewritten as one vectorised pass per (name, mask); a Python-level join over 7.27M rows x 6 rules would have taken minutes. |
| 2026-10-02 | P6 | **PHASE 2 assembled 0 events while reporting success.** The empty-shard cleanup tested `w.opened` *after* calling `close()`, and `close()` clears the live writer — so every one of the 16 shards looked empty and was deleted. `wrote_anything` is now a sticky flag that survives close, pinned by a test. Two further hardening steps: PHASE 1 now verifies the shards actually contain the event-base row count and refuses to continue if not, and a 0-event assembly raises instead of printing "NEXT: colab_p7_train.py". |
| 2026-10-02 | P6 | `pq.ParquetWriter` needs the schema at construction, which is incompatible with the lazy open-if-needed pattern the assembly uses for 16 shards and 3 outputs — it died at PHASE 1. Added `colab_utils.LazyParquetWriter`, which derives the schema from the first table written. Empty shards are now deleted rather than left as files the read phase would fail to open, and the cell prints the row count of each output so a lost write is visible. |
| 2026-10-02 | P6 | Ordering bug in that helper: `write_frame` imported pyarrow before checking for an empty frame, turning a silent skip into `ModuleNotFoundError` where pyarrow is absent. The guard now comes first. |
| 2026-10-02 | P6 | **Colab RAM crash at batch 12 of 15 — the assembly held both full feature blocks (7.27M x 35 and 7.27M x 19, ~3 GB) *and* every accumulated output chunk in Python lists. A single whole-catalogue X frame is ~4.9 GB, so peak demand exceeded 10 GB. Rewritten to stream: inputs are repartitioned into 16 parquet shards keyed on a stable CRC32 of `event_id`, each partition (~454k events) is processed and written straight to its output writer, and the shards are then deleted. Nothing accumulates in a list. |
| 2026-10-02 | P6 | Partitioning uses **CRC32, not Python's `hash()`** — the string hash is randomised per process by `PYTHONHASHSEED`, so a rerun would place an event in a different shard and silently produce a different dataset. A test runs the partitioner in two subprocesses under different hash seeds and requires agreement. |
| 2026-10-02 | P6 | **P6 bug: each feature block was validated against the combined 52-column list**, so a correct tier-1 file reported as "missing" the 18 industrial features that live in the other file — a guaranteed failure on correct data. `validate_feature_blocks()` now checks each block against its own schema, rejects duplicates across blocks, and verifies the union matches ledger order. Six tests pin it, including the exact mistaken call. |
| 2026-10-02 | P6 | `dataset_builder.py` + `colab_p6_dataset.py` + `colab_p7_train.py` written, closing the path to a first trained model. The builder encodes three brief requirements as code rather than as discipline: raw coordinates stripped and asserted out of X, `MISSING_FEATURE_FRACTION` keeping "could not measure" a gap rather than a zero, and `add_site_baseline()` judging FRP against each site's own median (the 34 MW → 390 MW case) with `SITE_N_EVENTS` so a single-event site cannot masquerade as normal. |
| 2026-10-02 | P6 | Two domain traps in the location rule, both caught by tests. Substring matching made `"lon"` match **ELONGATION**, silently deleting a shape feature from the matrix; matching is now token-wise. And **`DIST_LNG_LOG` is liquefied natural gas, not longitude** — treating `lng` as a coordinate would have deleted the LNG distance feature. |
| 2026-10-02 | P6 | `KNOWN_FLARE`'s four inputs are recoverable by inverting the tier-1 definitions (`derive_flare_inputs`), which is why the industrial pass was right to leave it NaN rather than guessing. 96.7% of events are single-cluster, so their `active_days` is ~1 and no single-day fire can satisfy a 200-day flare rule — correct, and a useful sanity check. |
| 2026-10-02 | P7 | `colab_p7_train.py` reports per-class precision/recall/F1 as the brief asks, on the combined out-of-time+out-of-space split, for both model A (operational, with facility priors) and model B (scientific, non-leak only), with the distance-threshold baseline tuned on train only and the 60% human-review queue counted rather than discarded. |
| 2026-10-02 | P4a | **Ordering bug in P3a:** `FACILITY_DIR = EXTERNAL_ROOT / "osm_facilities"` sat in the config block *above* the Drive-resolution block, so it ran while `EXTERNAL_ROOT` was still `None` → `TypeError: unsupported operand type(s) for /: 'NoneType' and 'str'`. All derived paths now build after resolution, with a `RESOLVED LOCATIONS` table, and a static check confirms no cell derives a path before `PROJECT_ROOT` is assigned. |
| 2026-10-02 | P4a | Hardened the membership↔detection join. The cluster id column is detected on the detections side across five candidate names (the two sides may spell it differently), and **a zero-row merge now prints a loud diagnostic** instead of `continue`-ing: an empty chunk appended to the feature frames is indistinguishable from a successful run that produced nothing. |
| 2026-10-02 | P4a | **Survey caught a silent-zero bug.** Cluster files are `clusters_2012-01-20.parquet` (ISO dates with dashes); P4a was building `f"{day}.parquet"` from `YYYYMMDD` cluster ids, which would have matched **no files and reported success**. Added `index_daily_files()` to discover the real pattern (accepts both dashed and compact), plus `parse_date_from_filename()`. `edges/` sits beside `clusters/`, so a naive `*.parquet` glob would have joined edge rows to detections — now filtered by prefix. Membership ids require `.str.slice(0,8)` **and** an 8-digit check; a dashed id coerces to `NaT` and silently drops a day. Verified by simulating the real directory layout. |
| 2026-10-02 | P4a | `external/` has been restructured and shrank from 1.50 GB / 911 files to **210.4 MB**. `sentinel5p` is still effectively empty (444 B), so the 9 ATMOSPHERIC features stay blocked; `osm_facilities` is now nested in 3 subdirs; `firms/` and `firms (1)/` are empty. **P1b's coverage numbers are stale** and must be re-run before any availability claim. |
| 2026-10-02 | infra | **Stale Drive FUSE cache.** `colab_survey_tree.py` reported `MyDrive entries: ['Firm_Clusters']` with Firm_Clusters scoring 0/2 required subpaths, while the browser plainly showed `full_2012_2026` inside it — the folder was invisible because it had been added/edited in the browser while the Colab mount was live. `resolve_project_dir()` now detects the no-match case, prints what the runtime *can* see, calls `force_refresh_drive()` (a `force_remount` to drop the FUSE cache) and retries with backoff, rather than reporting failure and letting the caller guess at paths. |
| 2026-10-02 | infra | Bug in that fix: `wait_for(..., delays=RETRY_DELAYS)` bound the delays as a **default argument**, so the tuple was captured at def time and any override was silently ignored — which also made the test suite take 111 s of real sleeping. Now read at call time, with `retry_delays` plumbed through `resolve_project_dir()`. Suite is back to 0.5 s. |
| 2026-10-02 | infra | **The Drive layout does not match the handoff.** Confirmed from the actual folders: `clusters/` and `edges/` exist in BOTH `Firm_Clusters/` and `Firm_Clusters/full_2012_2026/`; `India_Firms.parquet` is at `Firm_Clusters/` not `MyDrive/India_Firms/`; `full_2012_2026/` also holds `analysis`, `feature_audit`, `feature_report`, `features`, `features_missing`, `mappings`, `shubham_thermal_feature_analysis`. Depth is the wrong tie-break for the duplicate `clusters/`, so `pick_by_file_count()` chooses by how many files each copy actually holds and prints the full candidate table. Added `colab_survey_tree.py` to print the tree with counts/sizes and pin every path empirically. The event base and membership file are now discovered by name instead of hardcoded. |
| 2026-10-02 | infra | A hand-made `Firm_Clusters/` containing only `pipeline/` and `qc/` was added to Drive, creating a second candidate that is *incomplete*. Verified the resolver still picks `Firm_Clusters (1)`: completeness is the gate and the folder name is only a tie-break among complete candidates, so an empty stub cannot shadow the real tree. Pinned by three new tests. |
| 2026-10-02 | infra | **Root cause of the P3a/P4a `FileNotFoundError`s: the tree is now `Firm_Clusters (1)`**, a macOS-style duplicate of `Firm_Clusters`. Every cell hardcoded `/content/drive/MyDrive/Firm_Clusters/...`, so the data was present and reachable while the cells still failed. Added `cu.resolve_project_dir()`, which resolves the project root **by content** — it finds the directory that actually contains `full_2012_2026` with all required subpaths, preferring the exact name and otherwise the lowest numeric suffix. Verified against a synthetic Drive containing `Project2`, a notebook file and only `Firm_Clusters (1)`. Partial duplicates are rejected so a truncated copy cannot win. |
| 2026-10-02 | infra | Replaced the bootstrap cell with **`colab_bootstrap.py`**, which has no hardcoded `MODULES` list: it installs every `.py` from the upload *and* every `.py` already in `Drive/pipeline`, then cross-checks the modules and prints the Drive state. The hardcoded list had already caused three rounds of `ModuleNotFoundError` (`industrial_features`, then `lineage_features`, then `colab_utils`) because each new module needed a docs edit that was easy to miss. Verified against three upload shapes, including a `colab_utils (2).py` macOS-suffixed key. |
| 2026-10-02 | infra | `colab_utils.py` added after `drive.unmount` left `/content/drive` as a dead FUSE mount. `Path.exists()` **raises** errno 107 on that state rather than returning False, so the `if not X.exists()` guards were not protecting the cells — they were crashing them, with tracebacks pointing at innocent cell logic. All Drive checks in P1b/P3a/P3d now go through `path_ok()`/`require_drive()`, which report the mount as absent and print recovery steps. |
| 2026-10-02 | infra | Bug in the fix itself: `DEAD_MOUNT_ERRNOS` was built from `errno.ENOTCONN`, which is **107 on Linux but 57 on macOS**, so the set silently omitted the value that matters on the host running the tests. Caught by `test_colab_utils`, which failed on this Mac. Now literal numbers `{5, 19, 57, 70, 107, 116}`. |
| 2026-10-02 | P4a | `lineage_features.py` written: all 54 THERMAL/TEMPORAL/SPREAD/QUALITY features from the India_Firms detection lineage, vectorised via `groupby` so 7.27M events never enter a Python loop. The output list is **derived from the ledger**, not hardcoded, so schema order cannot drift. `colab_p4a_lineage.py` streams the 5,337 daily cluster files in date batches. |
| 2026-10-02 | P4a | **Four real bugs caught by the 59 tests:** (1) output order put recurrence features in the wrong place vs the ledger; (2) `STATIONARITY_INDEX` returned NaN for a perfectly static event — a zero path length is the *most* stationary case, now 1.0; (3) `FRP_PERSISTENCE` returned 1.25 for a fully persistent event, since N daily detections span only N−1 day-intervals — denominator is now `span_days + 1`, bounded by 1; (4) degenerate hulls (all points collinear) reported `SOLIDITY`/`COMPACTNESS`/`ELONGATION` as **0.0**, which reads as "scattered" about a perfectly stationary stack — now NaN. |
| 2026-10-02 | P4a | Ledger ordering hazard: a blanket `_set` over the four lineage groups silently overwrote `QUALITY_UNCERTAINTY` back to `MISSING_IMPLEMENTATION`, and also swallowed the P3G phase for the 4 recurrence features. Both fixed — the quality group is no longer re-set, and recurrence is explicitly re-asserted as `MISSING_IMPLEMENTATION`/`P3G` after the block. `LEDGER_VERSION` → **1.3.0**. |
| 2026-10-02 | P4a | Gate P1 re-passed: 12 COMPUTED / 72 NEW_IMPL / 10 UNVERIFIED / 7 HIST_NAN / 40 MISSING. The 33 former `UNVERIFIED_HISTORICAL` lineage claims are now written-down, tested definitions — a stronger position than the old inference-only claim, and a test asserts none can drift back. |
| 2026-10-02 | P4a | Bootstrap cell updated: `MODULES` now lists all four modules, so `lineage_features` and `fire_labels` are actually installed instead of failing with `ModuleNotFoundError`. Cross-module assertions extended to `lf.N_LINEAGE == 54`, lineage-subset-of-ledger, and 6 label classes. Verified end-to-end locally: ledger 1.3.0, industrial 18, lineage 54, labels 6, `GATE P1 PASSED`. |
| 2026-10-02 | P5 | `fire_labels.py` written: the 6-class rule cascade with every threshold in one frozen `LabelThresholds` dataclass, so label decisions are auditable. Two guarantees are enforced and tested — a rule can only fire on **positive** evidence (absence always yields `UNKNOWN`, kept with weight 0.2 rather than dropped), and a missing column is NaN so dependent rules abstain rather than read it as 0. Fixed precedence: FLARE → MINING → INDUSTRIAL_FIRE → AGRICULTURAL → WILDFIRE → UNKNOWN. |
| 2026-10-02 | P5 | **Bug caught by test:** `INDUSTRIAL_FIRE` used `surface_loss OR no_recovery`, which mislabelled every seasonal burn inside a plant gate (cropland drops in NBR then regrows). The spec is surface loss that does **not** recover — a conjunction. Corrected to `AND`; a missing recovery composite now abstains. |
| 2026-10-02 | docs | README §5 added: the path to training, with an explicit statement that the full 141-feature model is **not** trainable yet (~64 features available), the ordered steps P3a → lineage extractor → P5 → P6 → P7, the two-model A/B leakage split, the baselines to beat, and a warning not to lower label thresholds to manufacture labels. |
| 2026-10-02 | P3a | **Blocker:** `colab_p3a_industrial.py` → `FileNotFoundError` on `external/osm_facilities`, and P1b found no `external/` at any of its 5 roots. Since bootstrap had just written `qc/*.parquet` to `/content/drive`, Drive *is* mounted — so this is a path or account problem, not a mount problem. Added `colab_p3a_diagnose_external.py`, a read-only cell that reports A (not mounted) / B (wrong account) / C (moved) / D (absent) and searches to depth 3 for a relocated tree. |
| 2026-10-02 | P3a | **Version guard made a minimum, not an exact match.** A cell expecting `1.1.0` raised `STALE ... found 1.2.0` at a module that was correct — the real problem was that the *notebook cell* was stale text, not the repo file. All comparisons (`_is_usable`, `_too_old`) now use version-tuple ordering, so `1.10.0 > 1.2.0` and a newer module is always accepted. Also replaced the old exact-match upload key. |
| 2026-10-02 | P3a | Added a single **bootstrap cell** (README §6) that installs both modules, cross-checks them against each other, and runs Gate P1 via the ledger's own `build_audit_records()`. Replaces re-pasting `colab_p1_ledger.py`, which is what let a stale cell persist. Verified locally: all 6 gate assertions pass and INDUSTRIAL reports `NEW_IMPLEMENTATION: 18`. |
| 2026-10-02 | P3a | **Root cause of the stale module found.** The P1 loader's tier A accepted any `/content/feature_ledger.py` that merely *existed*, and since `LEDGER_VERSION` was left at 1.1.0 while the vocabulary changed, the version guard compared 1.1.0 to 1.1.0 and passed an old file. Gate P1 therefore re-ran against the pre-P3a ledger (`INDUSTRIAL 18 MISSING_IMPLEMENTATION`, no `NEW_IMPLEMENTATION`). Fixed: `LEDGER_VERSION` → **1.2.0**, tier A now reads the declared version out of the file and discards a stale copy, tier C skips stale Drive candidates, and tier D matches the macOS `(2)` suffix and verifies the uploaded version. |
| 2026-10-02 | P3a | Colab run blocked by `ModuleNotFoundError: industrial_features` — the module is new and was never uploaded to `/content`. Added a general multi-module upload cell (§6) that tolerates the macOS `(2)` suffix for both modules and asserts the ledger version, `LOCAL_INPUT_HINTS`, `S_NEW_IMPL`, feature-order equality and the CRS before any compute. |
| 2026-10-02 | docs | `HANDOFF.md` written (10 sections) and this `README.md` created as the standing status/change log. |
| 2026-10-04 | P7 | **Operating threshold implemented.** `HANDOFF` §8 mandated "freeze an operating threshold tau maximising F2 subject to precision >= 0.90" and it had never been written. `REVIEW_CONFIDENCE = 0.60` was a bare literal applied to models that are not calibrated to a 0.60 prior: model B's held-out scores sit near 0.13, so the flat cut flagged **0.8% of positives** — the "detects 2 of 100" symptom. New `threshold_calibration.py` freezes tau on the **TRAIN fold only**, tie-aware via `training_design._tie_aware_curve`, and returns `threshold: None` with a stated reason when the precision constraint is unreachable rather than inventing a number. |
| 2026-10-04 | P7 | **Measured on the existing held-out predictions** (`threshold_calibration.py` against `heldout_pred_{A,B}.csv`). Model B: the only threshold reaching precision >= 0.90 is **tau = 0.875**, catching **0.04%** of positives. Reaching 50% recall costs a flag on **25.5% of the catalogue** at precision 0.32 against a 0.164 base rate, flagged DEGENERATE. Model A: tau = **0.033**, flagging **16.5%** of rows against a **16.39%** prevalence — numeric proof that model A reproduces the labelling rule rather than classifying, which §15 had asserted from inspection. |
| 2026-10-04 | P7 | Per-class metrics and the review queue now gate on `effective_threshold = max(tau, 0.60)`, i.e. the DEPLOYED bar, and the cell states plainly when no operating point was freezable instead of silently falling back to 0.60. Calibrated thresholds persist in `qc/models/model_{A,B}.pkl` and in `p7_results.json`. |
| 2026-10-04 | bugfix | **`dist_m` in the evaluation CSVs held log10 units, not metres.** The conversion `10**x - 1` was assigned to a local that nothing read, and the raw `DIST_NEAREST_FACILITY_LOG` column was written under the name `dist_m`. The files contain 2.47–4.76 where `evaluate_model.py` and the baseline expect metres — four orders of magnitude. No metric ever looked wrong because `10**x - 1` is monotone, so every ranking-based number was unaffected; that is why it survived. Fixed, with a runtime unit guard in P7, a `dist_log10` column so the convention is never ambiguous, and 4 static tests. |
| 2026-10-04 | backend | `model_registry` serves the calibrated `operating_threshold` and `auto_accept_threshold` from the pickle, reports `has_operating_point`, and warns on load when an artefact predates calibration. The brief's 0.60 is retained as `review_confidence` and kept a **separate** field: they are different constraints and `requires_human_review` now gates on the effective bar. Pre-calibration pickles still load and degrade to "review everything". 8 new tests; schemas updated. |
| 2026-10-04 | infra | `threshold_calibration.py` + 33 tests; 9 new static checks in `test_colab_cells` (`TestP7DistanceUnits`, `TestP7OperatingThreshold`), both verified to fail when the original defects are reintroduced. Local suite **579 tests OK** (5 skipped, pyarrow is Colab-only) + **30 backend tests OK**. |
| 2026-10-04 | P7b | **The weak `INDUSTRIAL_FIRE` label is a 5 km proximity cut.** Measured on the full catalogue (P7b): max distance-to-nearest-facility among all **1,119,254** positives = **5,000 m**, with **2** rows beyond it (0.0002%, float rounding on `10**x - 1`). **PR-AUC of distance alone = 0.9811** vs model B's 0.3353. `FACILITY_COUNT_5KM >= 1` is satisfied by 100% of positives and is the *same statement* as `DIST_NEAREST <= 5000 m`, so `near_any`'s two branches are redundant. Consequence: **no feature work can improve model B**, because the target contains nothing beyond distance and persistence. Land cover and weather are irrelevant to this label -- they would help a target that consulted them, and this one does not. |
| 2026-10-04 | P7b | Two defects in the P7b cell itself, both fixed and both worth recording. (1) `external/` was probed at `<root>/external` while P1b found the tree at `<root>/full_2012_2026/external`, so the cell reported "no rasters anywhere" from a wrong path -- the duplicate-folder hazard that already produced one false "the Drive is missing" diagnosis. Both candidates are now probed and reported. (2) The hard-cut verdict used `== 0` and printed "NOT a hard cut" against 2 rows in 1.1M. Now a tolerance. A rounding artefact read as evidence is the exact failure mode this project keeps hitting. |
| 2026-10-04 | P7c | **`gold_set.py` + `colab_p7c_goldset.py` + 42 tests.** `INDUSTRIAL_FIRMS.parquet` has **no FIRMS `type` column**, so there is no on-disk independent cause signal and no download-free automated label. The gold set is therefore the only route to a defensible number, as HANDOFF §8 already mandates. The sampler is stratified onto the **3-8 km decision boundary** (deliberately starting inside the 5 km cut, where the rule is still undecided), round-robins across spatial blocks so it is not one refinery, and emits **sampling weights** so an oversampled sheet can be scored back to prevalence. `build_sample` never writes `gold_label`; a test asserts it. `uncertain` rows are reported, never dropped. |
| 2026-10-04 | P7c | Three real bugs caught by the gold-set tests: `(pred == 1) & (y == 1) * w` parses as `bool & float` because `&` binds looser than `*`; `_distance_bucket` used scalar `if not np.isfinite(m)` on a column; and **the sampler indexed two different coordinate systems** -- the boundary band by position in `work`, the remainder by position in the shorter `remaining_rows` -- which returned a full-looking 300 rows of entirely the wrong composition, with strata at 157/18/125 instead of 150/90/60. All three are now regression-tested, including one that asserts every row's distance range agrees with its own bucket label. |
| 2026-10-04 | P7c | P7c crashed mid-stream: `pyarrow.Array.to_numpy(zeros_on_null=...)` is not a pyarrow kwarg (it is a pandas/`ArrowDtype` concept), so the cell died on the first row group. Replaced with `to_pandas().to_numpy(na_value=nan)`, which is unambiguous about nulls. Added a finite-distance guard so the cell refuses to draw a distance-stratified sample from a NaN column instead of silently bucketing everything into `no_distance`. |
| 2026-10-04 | P7d | **`external/` located and inventoried** at `full_2012_2026/external` (914 files, 1.70 GB) -- the earlier "not visible" was my own path bug, now fixed by probing both candidates. Contents: `VIIRS_FOOTPRINTS/` 1.08 GB, `osm_facilities/` 0.25 GB, `DL_FIRE_M-C61_814501/` **3 CSV files, 89 MB**, `era5_land/` 175 `.nc` at ~120 KB (still suspect), `sentinel5p/` 303 `.tif` at ~1-4 KB (still empty), `india_mopitt_gfas_output/` 22 tiny parquet. **No land cover, no NBR, no SAR.** Earth Engine cannot authenticate from Colab (`ee.Authenticate()` raises), so the GEE route needs a local runtime. |
| 2026-10-04 | P7d | **Highest-value open lead:** `DL_FIRE_M-C61_814501/` is a FIRMS distribution archive, and a FIRMS CSV ships a `type` column (0 AGRICULTURAL / 1 ARCHIVE / 2 GRASS / 3 SHORELINE / 4 AVIATION) -- a cause classification derived from the satellite, independent of the gazetteer that manufactured the weak label. P1b inventoried that folder by size and extension and never opened a header. P7d probes it. If usable, the no-download impasse on labelling is over. |
| 2026-10-04 | P7d | **Retracting an over-claim.** P7b measured `shape: persistence >= 0.0` at 100% positives / 100% negatives and I wrote that the label was "pure proximity with no fire-behaviour term". That measurement used threshold **0.0**, not the calibrated one -- `WeakThresholds.industrial_persistence` defaults to **0.35**, and `FRP_PERSISTENCE = observed_days/30`, so ~0.35 needs ~11 observed days and is satisfied by far fewer than every event. The `shape` conjunct may discriminate after all. P7d measures the full decomposition at the calibrated thresholds and reports the ablation (`near AND typology` vs the full rule) so the claim rests on the number rather than on an inspection. What survives regardless of `shape`: the max-positive distance is 5,000 m, and distance alone reaches PR-AUC 0.9811. |
| 2026-10-04 | P5b | **ROOT CAUSE FOUND — the weak-label calibration never ran.** `qc/p6_rule_calibration.json` shows `industrial_persistence = 0.03333333333333333` with `n_above == n_total == 201840`. `0.0333` is `1/30`, the FLOOR of `FRP_PERSISTENCE = observed_days/30`, so `persistence >= 1/30` is true for **every event in the catalogue**. The rule's `shape` conjunct was vacuous. It reported a threshold and a target, so it looked calibrated; the threshold was simply the floor. |
| 2026-10-04 | P5b | **Why:** `calibrate_to_target_counts` bisects until the rule admits ~`target` events, but P6 PHASE 2 passes a **200,000-row sample** while `CALIB_TARGETS` are written against the **7,265,849-row catalogue** (flare 3,600 · industrial **400,000** · mining 60,000). Two of three exceed the sample size, so `admitted(floor) < target` fired on the first evaluation for every class and every threshold was pinned to the floor. Fixed by `rule_calibration.scale_targets()` (catalogue-scale targets rescaled onto the frame, with `population_size` explicit and the rescale printed). Same collapse hit `flare_persistence` and `mine_persistence`, which is why README §12 recorded MINING falling to 61. |
| 2026-10-04 | P5b | Added a **vacuity detector** so this cannot recur silently: `Calibration.is_vacuous` is computed from `n_above/n_total >= 0.9999` (data-driven, not from an annotation string), `vacuity_reason()` derives the explanation, and `calibration_report` gains `vacuous` + `reason` columns and prints a WARNING naming the offending values. 7 new tests including a before/after on the same frame proving the unscaled path collapses and the scaled path does not. |
| 2026-10-04 | P5b | **Confirmed the label reduces to two distance terms:** `(facility within 5 km) AND (power/petrochemical/steel/cement within 10 km)`. `FACILITY_COUNT_5KM >= 1` is the same statement as `DIST_NEAREST <= 1000 m`, and the measured max positive distance is exactly 5,000 m. `shape` contributes nothing. Matches the measured 15.11% of negatives satisfying the typology term. |
| 2026-10-04 | P5b | **Why fixing the calibration is necessary but NOT sufficient.** `FRP_PERSISTENCE` and `DETECTION_COUNT_LOG` are label-rule inputs, so `training_design` excludes them from model B. A non-vacuous persistence threshold makes the label BETTER; it cannot make it INDEPENDENT, because the evidence would still come from columns the scientific model is forbidden to see. Model B needs a label built from inputs that are **absent from X entirely** — NBR drop, no recovery, non-vegetated fuel. Hence P7e. |
| 2026-10-04 | P7e | `colab_p7e_gee_probe.py`: verify-before-build cell for Earth Engine. Confirms the datasets and wanted bands exist, samples **real values for real gold-set events** (the check that catches an all-zero raster, which is how `sentinel5p` at 444 B and `era5_land` at 120 KB/month survived until now), and **times one 5,000-point tile to extrapolate the cost of the full extraction**. Land cover is a static raster, so it is one sample per location, not per date. |
| 2026-10-04 | P7e | **The Earth Engine blocker is a missing Cloud PROJECT, not credentials.** `ee.Initialize()` bare raises `no project found. Call with project=` -- a message that reads like an auth failure when credentials may well be fine. Earth Engine cannot infer which Cloud project to bill. Two independent things are needed: credentials (`ee.Authenticate()`) and a registered project id (`ee.Initialize(project=...)`). `gee_utils.py` resolves the project from an explicit argument, then `EARTHENGINE_PROJECT`/`GOOGLE_CLOUD_PROJECT`/`GCP_PROJECT`, then Colab secrets, then `gcloud`, then Earth Engine's own asset roots -- and `EEAuthError` carries the full remediation instead of a bare message. 20 tests, no `ee` import required. |
| 2026-10-04 | P7e | **`raise SystemExit` inside a notebook cell is unusable here.** On Python 3.13 it trips an IPython/`inspect` bug and buries the real error under three chained tracebacks, the last being a `TypeError: object of type 'NoneType' has no len()` in IPython's own pretty-printer. P7e now has zero `SystemExit` calls; every exit path prints diagnostics and guards instead. |
| 2026-10-04 | P7e | P7e re-written to verify rather than build: resolve the project, confirm datasets and bands, **sample real values for real gold-set events** (catches an all-zero raster, which is how `sentinel5p` at 444 B and `era5_land` at 120 KB/month got onto Drive unnoticed), and **time one 5,000-point tile** to extrapolate the full-extraction cost. Land cover is static, so one sample per location suffices and a stratified subsample is a legitimate scope rather than a compromise. |
| 2026-10-04 | P7f | **Measured, not guessed: direct WorldCover download is ~10.4 GB.** ESA WorldCover v200 lives at `.../v200/2021/map/ESA_WorldCover_10m_2021_v200_<N27E072>_Map.tif`, 3x3-degree tiles, ~110 MB each, `Accept-Ranges: bytes`, DEFLATE COG. **102 tiles cover India** (lat 5-38, lon 66-99) = **10.45 GB**. The other 19 of 121 grid slots are open ocean and legitimately 404 -- WorldCover publishes only tiles containing land. **Consequence: GEE is cheaper in bytes**, because `reduceRegions` returns 7.27M sampled values rather than 10 GB of raster. P7f is therefore the FALLBACK for when the Cloud console is unavailable, not the preferred route. |
| 2026-10-04 | P7f | **My first URL pattern was wrong and 404'd.** Verified against the live bucket before shipping: there is no `_100m_` COG under `v200/2021/` despite the product documentation mentioning one, and the path is `/map/` not the bucket root. Caught by HEAD-ing real tiles rather than trusting the documented filename -- the same class of error as the `dist_m` log/metre mixup and the two wrong `external/` paths. |
| 2026-10-04 | P7f | `LANDCOVER_VEGETATED` distinguishes burnable substrate (tree, shrub, grassland, cropland, herbaceous, wetland, moss) from non-vegetated (built-up, bare, water). "Was there burnable vegetation here?" is answerable from it and is **completely independent of the facility gazetteer**, which is what the current label is made of. Explicitly NOT the same claim as "industrial": a stubble burn 300 m from a refinery is cropland and will be called a vegetation fire. Sampled at **~300 m** by nearest-neighbour downsample from the native 10 m (averaging would yield fractional class codes matching no class); the resolution is stated in the output rather than buried. Column is `-1` when unmapped so a missing tile can never read as "bare". |
| 2026-10-04 | P7e | **Earth Engine auth: `ee.Authenticate()` succeeds; only the Cloud PROJECT is missing.** Confirmed from the attempt log -- `Authenticate()` never appears as a failure, while `Initialize(project=None)` fails three times with `no project found`. Auth needs two independent things: credentials (one browser consent) and a registered GCP project id. Added `.env` support to `gee_utils.project_ids_from_dotenv()` so the project id can be supplied every session without ever being committed: **`.env` is already in `.gitignore`**, and a test asserts it stays there, because a literal written into a tracked `.py` would publish the billing-account name with the repo. A GCP project id is an identifier, not a credential -- it carries no access on its own -- but it is kept out of git regardless. Handles quotes, `export` prefixes, inline comments and CRLF (a Windows-edited `.env` silently breaks a naive parser). |
| 2026-10-04 | P7e0 | **Account switch: three INDEPENDENT things break, and the 403 is not an auth failure.** (1) Colab caches the Earth Engine token at `/root/.config/earthengine/credentials`, and `ee.Authenticate()` does **not** re-prompt while a valid cache exists -- so you keep getting the old account while believing you re-authenticated. `colab_p7e0_reauth.py` reports which account Drive is serving, clears the cache (checking each candidate path rather than assuming one), and states the re-auth steps. (2) **A GCP project belongs to the account that created it.** After switching, the previous project 403s every call. (3) **Drive is per-account** -- remounting after a switch points the whole project at a different Drive, which is the failure nobody remembers. |
| 2026-10-04 | P7e | **`ee.Initialize` succeeding is NOT a working session.** It accepts credentials without checking the caller may USE the project, so `Initialize(project=...)` returned success and every subsequent data call 403'd with `roles/serviceusage.serviceUsageConsumer`. Added `EEPermissionError` (deliberately NOT a subclass of `EEAuthError` -- they need different fixes, and conflating them sends you to re-auth when credentials were already fine), `verify_can_read()` which performs one real band-name read before reporting success, and `projects_visible_to_current_account()` which reads `getAssetRoots()` to answer "which project can THIS account use" instead of guessing. P7e now surfaces a 403 as one clear message rather than four per-dataset load failures. |
| 2026-10-04 | P7e | **Three real bugs in P7e itself, all caught after the run.** (1) `cu.find_artefact` returns a `(Path, depth)` **tuple**, but P7e treated it as a Path -- so `if _src is None` never fired and pandas received a tuple, failing with `TypeError: cannot construct a FileSource from (None, None)`, which names neither the cause nor the file. Added `cu.find_artefact_path()` so the mistake is unrepresentable. (2) `ee.Image("COPERNICUS/S2_HARMONIZED")` does not raise -- it builds a composite Image with no `filterDate`, surfacing much later as `AttributeError: 'Image' object has no attribute 'filterDate'`. Added `COLLECTION_IDS` / `dataset_kind()` / `ee_dataset()` so the distinction is data, not per-call-site care. (3) P7e0 shipped with `files` instead of `_files`, a plain NameError, caught by the existing `test_no_use_before_assignment` the moment P7e0 was registered -- which is exactly what that check is for. |
| 2026-10-04 | P7e | **Stale cell detected from its own output.** P7e was re-run and the output was byte-identical to the previous run -- including the string `Earth Engine initialised`, which the fixed version no longer prints (it prints `Earth Engine working ... (data reads verified)`). The three fixed bugs (tuple passed to `read_parquet`, `ee.Image` on an ImageCollection, weak auth check) were all still present, so the fixes never reached Colab: the file picker had uploaded only some modules. **This is the project's fourth stale-cell incident** (section 5 defects 3 and 5, the partially-uploaded picker in section 6, and now this), always with the same shape. Added `P7E_CELL_VERSION = "2026-10-04.3"` and `gee_utils.__version__`, both printed on the FIRST line, plus a `STALE` warning. A stale copy now announces itself in the output instead of being diagnosed from a downstream failure. |
| 2026-10-04 | infra | **The bootstrap's `REQUIRED` set was the hole that let a partial upload pass.** It listed 11 modules and did not include `threshold_calibration`, `gold_set`, `gee_utils`, `quality_features` or `explainability` -- all added later. So the cell reported success while the modules a downstream cell imported were never installed, and the failure surfaced far away as an unrelated-looking error. All five are now in `REQUIRED`, and each gets an attribute-level staleness check for the SPECIFIC symbol the later cells call (`calibrate_operating_threshold`, `scale_targets`, `build_sample`, `verify_can_read`, `EEPermissionError`, `dataset_kind`, `project_ids_from_dotenv`, `QUALITY_FEATURES`, `shap_explain`). Verified: simulating an old upload now produces 6 named failures and exits 1, where before it printed success. **The rule: adding a module means adding it here too**, or the same hole reopens. |
| 2026-10-04 | infra | `BOOTSTRAP_VERSION` printed on the first line, alongside `P7E_CELL_VERSION` and `gee_utils.__version__`. Staleness in this project has now cost four runs (section 5 defects 3 and 5, the partially-uploaded picker in section 6, and the byte-identical P7e output). All three version markers print unconditionally so an old file announces itself rather than being inferred from a downstream failure. |
| 2026-10-04 | P7e | **Earth Engine catalog verified; four dataset definitions were wrong.** Checked against the catalog rather than assumed. (1) **ALL FOUR datasets are ImageCollections, including WorldCover** -- `ee.Image("ESA/WorldCover/v200")` raises `Asset ... is not an Image`. `IMAGE_IDS` is now deliberately `frozenset()`. (2) **WorldCover's land cover band is `Map`, not `cover`** -- the band check would have reported the single most useful band in the exercise as missing. (3) **`COPERNICUS/S2_HARMONIZED` is Level-1C TOA; the correct asset is `COPERNICUS/S2_SR_HARMONIZED` (Level-2A surface reflectance)**. NBR from TOA carries an atmospheric path radiance that varies between dates -- exactly the signal NBR exists to isolate. They are two distinct collections, not a rename. (4) **COVERAGE GAP: S2 L2A starts 2017, Landsat 8 starts 2013, the catalogue starts 2012.** `GOLD_0001` is 2012-11-08 and neither satellite can see it. Landsat 5/7 added, plus `gee.imagery_gap()` and a P7e section 3b that reports it. |
| 2026-10-04 | P7e | **The WorldCover class table was wrong for every code at 70 and above.** It read 80 as "herbaceous", but 80 is **permanent water bodies**, and 80 was in the VEGETATED set. That would have labelled open water as burnable substrate -- corrupting the very label this work exists to fix. Corrected to the catalog table: 70 snow and ice, 80 permanent water bodies, 90 herbaceous wetland, 95 mangroves, 100 moss and lichen. `VEGETATED_CLASSES = {10,20,30,40,90,95,100}`; water and snow are excluded. |
| 2026-10-04 | P7e | **Two code paths disagreed about what water is.** `colab_p7f_landcover_direct.py` carried its own copy of the class table and had the same error. Two independent routes to one column cannot each hold their own copy -- one will drift. `gee_utils` is now the single source and a test asserts both routes agree on every class name and on the vegetated set. |
| 2026-10-04 | P7e | **`verify_can_read` probed with an asset, which conflated a healthy session with a bad asset id.** It reported `Earth Engine session works but the probe failed (Asset ... is not an Image)` -- reassuring-sounding and wrong. A permission check must not depend on any dataset being correct, so the probe now uses `ee.Image.constant(1)` and touches no asset. |
| 2026-10-04 | P7e | **`KeyError: 'wanted'`** -- `verify_datasets` returned `wanted` on the success branch only, so a caller reading it raised. One dataset problem produced two bugs: the failure, plus a crash in the reporting code. All branches now return the same key set, with `wanted` and `error` always present, and P7e uses `.get()`. |
| 2026-10-04 | P7e | **EARTH ENGINE IS WORKING.** `burnished-stone-510609-f5` verified with a real data read; all four datasets load and every wanted band is present (WorldCover `Map`, MODIS `LC_Type1`, S2 `B12`+`B8A`, Landsat `SR_B7`+`SR_B5`). The GCP project hurdle that blocked four previous runs is cleared. |
| 2026-10-04 | P7e | **Stale-cell diagnosis, fifth incident, now mechanised.** The run reported `P7e VERIFY GEE [cell version 2026-10-04.3]` alongside `gee_utils: 1.2.0`. That combination is the tell: **modules travel via the bootstrap upload, cells travel as pasted notebook text.** Editing the repo never changes text already sitting in a notebook. Section 2 used the new `verify_datasets` and worked; section 3 was still the old body calling `ee.Image(ESA/WorldCover/v200)`. Same notebook, same session, two generations of one file. |
| 2026-10-04 | infra | **`colab_runcell.py` -- stop pasting cell bodies.** `run("colab_p7e_gee_probe")` fetches the cell from `Drive/pipeline` (where the bootstrap installs it) and `exec`s it in the CALLER's namespace, so a notebook cell becomes a two-line stub that cannot go stale. The cell's version marker prints BEFORE it runs, so a mismatch is visible even if the cell then fails. `status()` reports versions without executing. The caller-namespace default matters: a private dict would hide `EARTHENGINE_PROJECT` set in a notebook, and the cell would report "no project found" for reasons unrelated to itself. 17 tests. |
| 2026-10-04 | infra | **`cu.describe_missing_artefact()`** replaces "could not find canonical_event_base.parquet" -- which is unactionable, and which in this project has already produced one false "the data is missing" diagnosis. It now distinguishes absent from present-but-too-deep from present-under-a-similar-name, and lists what it found. `MAX_SEARCH_DEPTH` is 4, so the depth case is a live possibility. |
| 2026-10-04 | P7e | **Real finding from the probe: `S2 images within +/-60 d of 2012-11-08: 0`.** Not a failure -- Sentinel-2 did not exist in 2012. Confirms the coverage gap: S2 L2A starts 2017, Landsat 8 starts 2013, catalogue starts 2012. `GOLD_0001` has no Sentinel-2 imagery. Landsat 5/7 are now configured and P7e section 3b reports which years are uncovered. |
| 2026-10-04 | P7e | **The uploaded file was a different, older copy.** `colab_runcell` reported `[no version marker]` and 14,229 bytes against a 21,060-byte repo file. The marker made this unambiguous where an old *number* would only have been suspicious. Consequence: all four dataset fixes were absent from Drive, so section 3 still called `ee.Image(ESA/WorldCover/v200)` and section 4 still handed a tuple to `read_parquet`. Note the runner did its job correctly -- the failure was in *what got uploaded*, not in the mechanism. |
| 2026-10-04 | infra | **The bootstrap now verifies CELL versions, which `hasattr` cannot do.** A module is verified by attribute; a cell is a SCRIPT, so an old copy runs perfectly and fails on its content, looking like a live bug. `CELL_VERSION_FLOORS` declares a minimum for every `colab_p7*.py`; the bootstrap reads each cell back off `Drive/pipeline`, compares declared versions with version-tuple ordering, and **exits 1** on any that are missing, unversioned, or too old. The 14,229-byte upload would now abort at bootstrap instead of failing 3 sections later as a dataset error. |
| 2026-10-04 | infra | **No cell is exempt from version checking.** `P7B`/`P7C`/`P7D`/`P7E0` had no version constant and were set to `None` (unverifiable) -- which is the original problem in miniature, so markers were added to all six cells and every floor tightened. `P7E0_CELL_VERSION` had to be renamed from `P7E_CELL_VERSION` to avoid colliding with the P7e probe: both files are "7E", and the version regex takes the first match. 5 tests assert every floor matches what the file on disk actually declares. |
| 2026-10-04 | infra | The `test_no_use_before_assignment` check caught a placeholder (`{stem}`) left unsubstituted in all four newly-versioned cells -- a `NameError` at first line of output. The marker insertion was templated, the title substitution was not, and the static checker found it before Colab did. |
| 2026-10-04 | P7e | **P7e PASSES.** Earth Engine verified with a real data read; all four datasets load; every wanted band present (WorldCover `Map`, MODIS `LC_Type1`, S2 `B12`+`B8A`, Landsat `SR_B7`+`SR_B5`). Land cover sampled 8/8 with no nulls. The GCP project hurdle that blocked five runs is cleared. |
| 2026-10-04 | P7e | **THE FIRST EMPIRICAL EVIDENCE AGAINST THE WEAK LABEL.** Sampled substrate under real gold-set events: `GOLD_0001` built-up (2,597 m to facility), `GOLD_0002` built-up (1,731 m), but `GOLD_0007` **cropland** (5,007 m) and `GOLD_0008` **cropland** (7,374 m). The weak rule calls the last two INDUSTRIAL_FIRE -- they sit inside its 5 km proximity cut and the `shape` conjunct is vacuous -- while the substrate under them is farmland. 75% of the sampled points are vegetated. Eight points is not evidence; it is a reason to go and get evidence. |
| 2026-10-04 | P7e | **Imagery coverage gap confirmed by measurement, not inference.** Section 3b: sample years 2012-2023; Sentinel-2 L2A from 2017, Landsat 8 from 2013. **`GOLD_0001` (2012-11-08) has NO Landsat and NO Sentinel-2 coverage**, and 2013 and 2015 have no Sentinel-2. Landsat 5/7 are the only option for 2012 rows. |
| 2026-10-04 | infra | **BUG: `cu.find_all_artefact` only ever examined DIRECTORIES** (`if p.is_dir()`), so a file could never match at any depth. `find_artefact(root, "canonical_event_base.parquet")` returned None and the `max_depth=4` limit looked like the cause; the depth was irrelevant. Found by `describe_missing_artefact`, which used unbounded `rglob` and reported the file at `full_2012_2026/events/` while the bounded search missed it -- which is how the two functions disagreed. Now matches files and directories, still surfaces duplicate copies. |
| 2026-10-04 | P7g | **`colab_p7g_landcover_vs_label.py` -- the decisive experiment.** Samples land cover across ~120k events stratified on the weak label and cross-tabulates substrate against `label_weak`. Turns "the label is a distance rule" from an argument about source code into a measured disagreement rate. Reports the headline (what share of weak INDUSTRIAL_FIRE sits on vegetated substrate), the inverse (what share of built-up substrate the label catches), and the 2-5 km band where the 5 km cut actually decides. Also reports the **WorldCover-is-2021-static** caveat against event years, because applying a 2021 substrate map to a 2012 event assumes nine years of no change -- stated rather than assumed. |
| 2026-10-04 | P7g | **`sampleRegions` has a HARD 10 MiB request-payload limit.** 120,000 points failed with `Request payload size exceeds the limit: 10485760 bytes` -- rejected on the *request*, before any compute is billed, so it reads as a data failure when it is a transport limit. Added `gee_utils.sample_points_chunked()`: chunks of 5,000, and any chunk that still trips the limit is genuinely **sub-divided**, not merely retried. |
| 2026-10-04 | P7g | **The first chunker was broken in a way only a test could catch.** It pre-sliced the chunk and "retried" by halving a `size` variable that was never used to re-slice -- so the retry re-sent the identical oversized feature collection and failed identically forever. Replaced with a work queue that really divides. A second real bug: `import ee` ran *before* the empty-input early return, so an empty sample raised `ModuleNotFoundError` for a call that had nothing to do with Earth Engine. |
| 2026-10-04 | P7g | **Dropped points are now LOUD.** A chunk that fails outright used to vanish, and downstream it would be indistinguishable from "no land cover there" -- the silent-wrong-answer failure this project keeps paying for. `sample_points_chunked` warns with the count and percentage dropped. |
| 2026-10-04 | P7g | **A test I wrote asserted the wrong thing twice.** `all(n <= max_ok for n in sizes)` is false for a correct implementation, because the FIRST call is supposed to exceed the limit -- that is how the limit is discovered -- and because splitting 1,000 queues two 500s that are both attempted before either halves. Both assertions hid the behaviour under test rather than checking it. Replaced with the properties that actually matter: it strictly reduces, it terminates, and **no point is lost**. |
| 2026-10-04 | P7g | **THE RESULT. The weak `INDUSTRIAL_FIRE` label is empirically wrong, and by how much.** Over 119,999 events (100% mapped, 329s): **70.1% of weak INDUSTRIAL_FIRE sits on VEGETATED substrate** (cropland, grassland, tree, shrubland), only 29.9% on built-up/bare/water. Scored against non-vegetated substrate as a proxy: **precision 0.299, recall 0.636, F1 0.407**. Extrapolated: **~784,800 of 1,119,254 weak positives sit on vegetation**. **Precision is the damning number** -- of everything the label calls an industrial fire, ~70% is farmland that can burn. Recall is comparatively good only because non-vegetated substrate is rare. |
| 2026-10-04 | P7g | **RETRACTED AND CORRECTED -- the asymmetry below was inverted.** P7g printed `of all NON-vegetated events, 36.4% are labelled INDUSTRIAL_FIRE` while computing `P(weak OTHER | non-vegetated)`. Recomputed from the same cross-tab: **`P(weak INDUSTRIAL | non-vegetated) = 0.636`** and **`P(weak INDUSTRIAL | vegetated) = 0.116`**, a **5.5x skew in the label's FAVOUR**, not against it. Corrected proxy scores are **precision 0.299, recall 0.636, F1 0.407** (I previously reported recall 0.364 / F1 0.328 -- the recall was the complement). The cell is fixed to print both directions from their own cells, and `test_colab_p7h.py` now fails if a `~is_t` term reappears inside a labelled conditional. |
| 2026-10-04 | P7g | **The 5 km cut, confirmed empirically on 95,017 events rather than inferred from source.** Beyond 5 km the weak label fires on exactly 0.0% of events, at 96.8-98.4% vegetated. Inside 2-5 km -- where the cut decides -- it fires on 71.4% of events that are 89.7% vegetated: ~12,540 farmland events of which ~10,000 are labelled INDUSTRIAL_FIRE. That band is where essentially all the error lives, and it is precisely the band `colab_p7c_goldset.py` oversamples. |
| 2026-10-04 | P7g | WorldCover v200 is a **2021 static map**; 15,759 of the sampled events predate 2015, where the nine-years-unchanged assumption is weakest. Reported alongside the cross-tab rather than buried. Wetland, scrub and post-failure regeneration are the classes most likely to have changed. |
| 2026-10-04 | P7h | **`colab_p7h_landcover_all.py` replaces the 10.45 GB download.** All 7,265,849 events = 1,454 chunked `sampleRegions` calls, no download -- GEE returns values, not raster. At P7g's measured 13.7s/call that is **~5.5 hours serially**, in 30 shards of 250,000 rows. **Resumable by necessity**: Colab kills idle sessions, so results are written as shards, each flushed ATOMICALLY (`.partial` then rename) so a kill never leaves a torn parquet, and a manifest records progress. Re-running resumes and never recomputes a completed chunk. A shard whose row count does not match its expected span is DISCARDED rather than trusted. |
| 2026-10-04 | P7h | Progress arithmetic stated up front: 1,454 chunks, 250,000 rows/shard, 11.4 min/shard, 5.5 h total, resumable. A 5.5-hour extraction WILL be interrupted; the cell is written for that rather than pretending it will not be. Unmapped points are stored as `WORLDCOVER_CLASS = -1` and `LANDCOVER_VEGETATED = -1`, never as non-vegetated -- a missing tile must not read as bare ground, which is how a coverage gap would silently become a label. |
| 2026-10-04 | P7h | **P7h shipped with a stray `_finish()` call placed before its definition**, caught by `colab_runcell`'s traceback, and fixed. The reorder that removed `raise SystemExit(0)` had put the call in the wrong place. Verified with an AST check that the definition precedes every call, and by executing the cell. |
| 2026-10-04 | P7h | **`test_colab_p7h.py` -- the cell is now rehearsed end-to-end against stubbed Drive and `ee`, and it caught two bugs that would each have corrupted a 5.5-hour run.** (1) `_shard_no` was seeded from `min(done_chunks) // SHARD_CHUNKS`; with shards 1-2 missing, `min` was 0, so the resume **wrote shards 0 and 1, overwriting two files that were already correct**, and mislabelled the new ones. Shard identity is now a pure function of the chunk number and extraction is shard-major, so a shard is only ever written complete. (2) The assembly printed **no warning while producing 19,000 of 25,000 rows**. A short table silently shrinks every label and every rate derived from it, so assembly now prints `INCOMPLETE` with the gap and refuses to bless the table. |
| 2026-10-04 | P7h | Rehearsal covers: first run extracts everything; no row lost or duplicated; a second run does zero work; resume after losing shards restores completeness **and reproduces the same shard set**; a surviving shard is **not** rewritten (mtime asserted); a short/torn shard is discarded and re-extracted; `WORLDCOVER_CLASS = -1` yields `LANDCOVER_VEGETATED = -1`, never 0, so unmapped cannot read as bare ground. |
| 2026-10-05 | P7h | **FULL EXTRACTION COMPLETE: 7,265,849 of 7,265,849 rows, 100% mapped, 0 unmapped**, in 361 minutes across 30 shards, without a single resume (Colab stayed alive). Substrate distribution: cropland 48.35%, tree cover 32.69%, grassland 8.89%, shrubland 2.79%, built-up 3.77%, bare/sparse 3.01%, permanent water 0.47%, wetland 0.03%. **Non-vegetated substrate: 526,385 events (7.24%)** -- this is the candidate pool for a non-proximity label. `qc/landcover_events.parquet` is the first feature table in this project derived from evidence absent from X, and therefore the first that a model cannot trivially reproduce. |
| 2026-10-05 | P7h | The substrate mix explains the whole result: **cropland plus tree cover is 81% of the catalogue**, so a 5 km proximity cut around Indian industrial sites -- which are ringed by farmland -- selects cropland almost by construction. Built-up is only 3.77%. Any label built on proximity will therefore be dominated by vegetation fires. |
| 2026-10-05 | P7i | **`surface_labels.py` -- a target built from evidence the model cannot see.** Measured on the full catalogue: 70.1% of weak INDUSTRIAL_FIRE sits on vegetated substrate, precision 0.299 / recall 0.636 / F1 0.407 against a substrate proxy, `P(weak INDUSTRIAL \| non-vegetated)=0.636` vs `P(... \| vegetated)=0.116` (5.5x). Crucially the old rules read only columns already IN X, so model B was blind to the evidence behind its own target. `LANDCOVER_VEGETATED` comes from a raster absent from X, so after `label_input_features(include_surface=True)` the label and model share NO input. |
| 2026-10-05 | P7i | **The class is named `SURFACE_FIRE`, not `INDUSTRIAL_FIRE`, on purpose.** What the evidence supports is "a fire on non-vegetated substrate" -- measurably true. "Industrial" is a claim about CAUSE needing site corroboration or NBR burn-scar evidence. Naming it for what we hope it means repeats the original overclaim one level down. The gazetteer becomes CORROBORATION reported alongside, not the definition. The brief's agricultural/forest separation is answered more directly this way: cropland 48.35% vs built-up 3.77% of the catalogue. |
| 2026-10-05 | P7i | **Two thresholds on one kind of evidence, then removed.** A first draft filtered on BOTH `min_persistence` and `min_detections`. But `FRP_PERSISTENCE = observed_days/30` and `DETECTION_COUNT_LOG = log1p(detections)` measure the SAME thing -- repeat observation -- so one was near-inert: the pool grid moved 1,859 events across a six-fold change in `min_persistence` while `min_detections` moved it 26,000. That is exactly the near-vacuous-threshold shape that destroyed the weak label's persistence branch, reintroduced by accident. Worse, requiring repeat observation discarded **97% of the catalogue** (7,026,705 of 7,265,849 UNKNOWN) because 96.7% of events are single-cluster. The label is now substrate only; repeat observation is a reported CONFIDENCE TIER (`is_surface_fire_tiered`). 27 tests. |
| 2026-10-05 | P7i | **`training_design.label_input_features(include_surface=True)`** now adds `LANDCOVER_VEGETATED` and `WORLDCOVER_CLASS` to the forbidden set (24 -> 26), and the partition excludes substrate from model B **if anyone ever adds it to the ledger**. Substrate is deliberately NOT a predictor: `split_feature_sets` iterates FEATURES_141, and adding it to X would invite precisely the mistake already made with `DIST_*`. `include_surface=False` reproduces the old partition exactly for before/after comparison. |
| 2026-10-05 | P7i | **NBR is deliberately absent from the base rule**, with the reason printed by the cell rather than left as an omission: built-up substrate already has low NBR, so a *drop* is weak and noisy exactly on the substrate SURFACE_FIRE is defined over, while being strong on vegetation -- which is the negative class the substrate term already handles. Deferred as `SURFACE_FIRE_BURNED = SURFACE_FIRE AND dNBR < threshold` (~6h more extraction), because damage is a stronger claim than substrate. |
| 2026-10-05 | P7i | **P7i results. `SURFACE_FIRE` 526,385 (7.24%), `VEGETATION_FIRE` 6,739,464 (92.76%), UNKNOWN 0.** Substrate split: built-up 273,891, bare 218,625, water 33,865, snow 4. High-confidence tier (>=3 observed days): 3,378. |
| 2026-10-05 | P7i | **THE CORROBORATION IS STRONG, AND IT IS INDEPENDENT.** Median distance to nearest facility: **SURFACE_FIRE 1,302 m (79.5% within 5 km)** vs **VEGETATION_FIRE 13,194 m (16.3% within 5 km)** -- a 10x median separation and a 4.9x within-5km rate, between a raster that never saw the gazetteer and the gazetteer itself. That is real support for reading SURFACE_FIRE as industrial, which is why the class is named for what it measures and the gazetteer is reported as corroboration rather than folded into the definition. The weak label claimed 1,119,254 industrial fires (15.40%); SURFACE_FIRE is 7.24%. **The old label was roughly 2x too large AND mostly on farmland.** |
| 2026-10-05 | P7 | **P7 now trains on the surface label by default.** `LABEL_SOURCE` selects `surface` (new default) / `weak` / `strict`. It refuses to fall back to the weak label if `qc/surface_labels.*` is missing, because a silent fallback would train on the broken target again. The partition follows the label: `split_feature_sets(..., include_surface=(LABEL_SOURCE == "surface"))`, so substrate is excluded from model B when substrate is what the label is built from. TARGET defaults to `SURFACE_FIRE`. |
| 2026-10-05 | P7 | **UNKNOWN rows are EXCLUDED from the split, not coerced to 0.** A missing label is not a negative: training on it teaches the model that unmeasured substrate means "vegetation fire" -- the silent-wrong-answer failure this project has produced twice. The surface label carries 0/1/NaN, so `y_bin` is rebuilt from `label_surf == TARGET` rather than from a class-name comparison that would fold UNKNOWN into the negative class. |
| 2026-10-05 | P7 | Fixed: `DataFrame.attrs` carrying a `SurfaceThresholds` is not JSON-serialisable, so `to_parquet` refused and P7i **silently fell back to a 7.27M-row CSV**. Metadata now travels in two plain int/string columns (`surface_label_version`, `surface_n_substrate_mismatch`), which every writer accepts. 30 tests for the label. |
| 2026-10-05 | P7 | **Note on the run above: it used the WEAK label with INDUSTRIAL_FLARE (3,571 events) as target, so model B's PR-AUC 0.0001 and model A's 0.4925 describe that, not the new target.** FLARE has 33 positives in a 600,000-row sample; its metrics are sampling noise. It is no longer the default target. |
| 2026-10-07 | P7k/P7l | **NBR and SAR extraction cells built and rehearsed, not yet run.** `colab_p7k_nbr_change.py` (Landsat monthly NBR, `qc/nbr_events.parquet`, ~11 h) and `colab_p7l_sar.py` (Sentinel-1 GRD monthly VV/VH, `qc/sar_events.parquet`, ~22 h), both P7h-pattern sharded/resumable, both reading `meta.parquet` directly (no P7h dependency). Rehearsed against stubbed Drive/ee: `test_colab_p7k.py` 26 tests, `test_colab_p7l.py` 23 tests. Caught pre-run: the composite filtered on `CLOUDY_PIXEL_PERCENTAGE`, a **Sentinel-2** property absent from Landsat C2 L2 — every image would have been silently excluded and the 11-hour run returned pure NaN. Fixed to `CLOUD_COVER`. Also fixed a `test_colab_cells` blind spot (function parameters are bindings, `ast.arg` is not a Store Name), which forced the composite cache out of a module global and into an explicit parameter. The retired P7k draft (`colab_p7k_nbr.py`) is now a stub that refuses to run. See HANDOFF §16. |
| 2026-10-07 | infra | gee_utils gains `month_before`, `nbr_monthly_composite` (standard (NIR−SWIR) NBR, QA_PIXEL cloud+shadow mask, single band `NBR`), `_month_bounds`, `sar_imagery_gap`, `sar_monthly_composite` (VV/VH dB, both orbit passes); `colab_utils` gains `event_year_month`. Full local suite now 718 tests, OK. |
| 2026-10-07 | P7k/P7l | **Run 1 crashed on BOTH cells before the first sample; fixed in v2026-10-07.2.** Root cause: month coverage was YEAR-granular but the catalogs start mid-month — Landsat 8 L2 starts 2013-03-18 (2013-01/02 "covered" but empty → `normalizedDifference: must have at least 2 bands`) and GEE's S1 GRD archive starts 2014-10-03 (2014-01..09 empty → `sampleRegions: Invalid numInputs: 0`). `median()` of an empty ImageCollection is a zero-band image, so the crash is silent-until-sample. gee_utils now carries `LANDSAT8_FIRST_DATE` / `S1_GRD_FIRST_DATE` and month-granular predicates `nbr_month_has_imagery` / `sar_month_has_imagery`; the cells intersect year and month rules and print the dropped months. Also removed the `CLOUD_COVER < 80` scene filter (an all-clouded monsoon month is the same empty-collection crash; QA_PIXEL masks per pixel instead). Outputs now also land in `Firm_Clusters/nbr_sar_outputs/`. Suite: 722 tests, OK. |
