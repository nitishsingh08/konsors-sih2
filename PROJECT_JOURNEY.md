# SPARC — Fire-Event Industrial Classification: The Project Journey

Chronological technical record of the system that classifies industrial fires
and persistent thermal sources from NASA FIRMS VIIRS data. Companion to
HANDOFF.md (status) and README.md (change log). Numbers are measured.

---

## 1. Data sources and merging

15,392,720 VIIRS detections from three 375 m instruments merged into one
table (`India_Firms.parquet`, 622.7 MB):

| Source | Instrument | Era |
|---|---|---|
| NASA FIRMS VIIRS S-NPP | VIIRS 375m | 2012-01-20 -> |
| NASA FIRMS VIIRS NOAA-20 | VIIRS 375m | 2018+ (evening passes) |
| NASA FIRMS VIIRS NOAA-21 | VIIRS 375m | newest instrument |

Every detection keeps sensor, FRP, brightness temperature (I4/I5), scan
angle, confidence and UTC time so event-level aggregates can be recomputed
from raw data.

## 2. Clustering — how events form

1. **Daily groups**: detections cut by calendar date (VIIRS ~12 h revisit).
2. **Spatial link**: daily cKDTree joins detections < 1 km apart.
3. **Temporal link**: detections within 12 h join the same group.
4. **Connected components**: -> 7,589,305 local clusters.
5. **Cross-day edges**: cluster on day D links to D+1 if same fire
   (323,804 edges).
6. **Global events**: connected components over the edge graph =
   **7,265,849 global fire events** — the unit of analysis.

## 3. Feature engineering

- **Temporal / global / event-level (P4a)**: tier-1 `cluster_tier.py`
  (34 features streamed from daily clusters); tier-2 `lineage_features.py`
  (54: persistence, stationarity, autocorrelation, spread dynamics, centroid
  drift).
- **Industrial (P3a, `industrial_features.py`, 18)**: GEM+OSM gazetteer —
  log-distance to nearest facility, counts in 1/5/10 km, inside-facility,
  known-flare, density contrast.
- **External rasters** (Earth Engine + external/): LANDCOVER 16 (P7h DONE),
  NBR burn evidence (P7k DONE), SAR S1 IW VV/VH (P7l RUN), WEATHER/FWI 18
  (ERA5), SMOKE 7 + ATMOSPHERIC 9 (TROPOMI), TERRAIN 12 (SRTM), QUALITY 8.

## 4. Feature counts

Frozen schema (`feature_ledger.py`, **LEDGER_VERSION 1.3.0**) declares
**141 features**: THERMAL 15, TEMPORAL 14, SPREAD 17, TERRAIN 12,
LANDCOVER 16, INDUSTRIAL 18, WEATHER 18, SMOKE 7, SAR 7, ATMOSPHERIC 9,
QUALITY 8. ~60 implemented and computed. Dataset:
`full_2012_2026/features/dataset`, 7,265,849 rows. **Model A gets 53
features; Model B gets 29** (24 are label-rule inputs excluded by the
leakage partition). NBR/SAR add evidence absent from the thermal side,
which is what Model B most needs.

## 5. Model journey: Model A vs Model B

The weak label is MANUFACTURED from features the model also sees, so a model
given those inputs can reproduce the rule — its score is an upper bound.

| | Model A (operational) | Model B (scientific) |
|---|---|---|
| Features | all available incl. rule inputs | 29, rule inputs excluded |
| First correct run | PR-AUC 0.9963 F1 0.9978 | PR-AUC 0.1721 F1 0.0341 |
| Verdict | circular — never quote | the honest number |

**Lessons:** (1) A's 0.9963 is circular. (2) The weak label was empirically
wrong — 70.1% of weak INDUSTRIAL_FIRE is on vegetated substrate (proxy
precision 0.299, recall 0.636); the 5 km distance cut carried the error.
(3) Rule calibration replaced hand thresholds (persistence floor 1/30
excluded nothing). (4) SURFACE_FIRE (substrate label, land cover + NBR) is
the first label input ABSENT from X — a model cannot reproduce it. (5) P7
streams row groups (peak 0.47 GB) and samples 1.5M train / 600k test.
(6) Model B on SURFACE_FIRE after P7h: PR-AUC 0.105 vs 0.49 distance
baseline — that is the label ceiling, not a fitting failure.

## 6. Component glossary

### Data & labels
- **gold set** (`gold_set.py`, `colab_p7c_goldset.py`) — hand-curated firmly-
  evidenced events for validation.
- **fire_labels** (6-class cascade) / **weak_labels** (industrial-context
  hierarchy + leakage registry) / **surface_labels** (SURFACE_FIRE, substrate
  + NBR — deliberately not named INDUSTRIAL_FIRE).
- **cluster_tier / dataset_builder** — tier-1 features and the 7.27M x 60
  X/meta/y streaming write.
- **feature_ledger** (v1.0 -> 1.3.0) — the authoritative 141-feature schema;
  every notebook imports it and drift-checks.

### Training & honesty
- **training_design.py** — A/B leakage partition, stratified streaming
  sampling, combined/out-of-time/out-of-space splits, baselines.
- **rule_calibration.py** — percentile-derived weak-label thresholds.
- **threshold_calibration.py** — operating threshold (F2, precision >= 0.90),
  TRAIN only.
- **explainability.py / leakage_probe.py / confidence.py** — SHAP plumbing,
  label-reproduction detection, deployment uncertainty.
- **feature_gap.py / build_feature_audit.py / quality_features.py** —
  missing-implementation audit, qc/feature_audit_141.csv, FIRMS quality family.

### Infra
- **colab_runcell.py / colab_bootstrap.py** — bootstrap installs repo modules
  from an upload; runcell executes a cell file from Drive with a version
  banner, so pasted notebook cells cannot go stale (they did, five times —
  README 2026-10-04 incidents).
- **colab_utils.py** — Drive-mount-safe checks, project-root resolution.
- **gee_utils.py** — EE auth (project resolution + remediation), dataset-kind
  verification, pinned NBR band order, monthly composites, imagery gap,
  S1 IW + VV/VH polarization filters.
- **figures.py / p8_report.py** — P8 charts and report.

### test_*.py — what each validates
- industrial (48), lineage (59), cluster_tier (32): feature math.
- fire_labels (42), weak_labels (39), surface_labels: label rules, provenance,
  sign conventions.
- feature_ledger (39), feature_gap (24): 141 names, group sizes, gaps.
- training_design (41), threshold_calibration (33), rule_calibration (13):
  partitions, splits, thresholds.
- dataset_builder (50): streaming assembly.
- colab_utils / colab_root: Drive resolution.
- colab_cells (30): STATIC checks — undefined names, use-before-assign,
  undeclared accumulators, module-constant shadowing.
- colab_p7h / colab_p7k (26) / colab_p7l: END-TO-END rehearsal of multi-hour
  cells against stubbed Drive+ee (resume, torn/out-of-band shards, band rules).
- gee_utils (80): EE auth fallbacks, NBR band order, month coverage, S1
  IW/VV/VH filters.
- backend tests (30): model_registry payload/contract, integration.

## 7. Backend & frontend

- **FastAPI** (`backend/app/main.py`): model registry boots FIRST, database
  second — inference never depends on Postgres; DB gone = degraded, not dead.
- **model_registry.py**: loads `{booster, features, target}`; predict() rebuilds
  rows in the exact trained column order. Default serving model = B (A is
  circular).
- **Endpoints**: `POST /api/predict`, `POST /events/{id}/explain` (SHAP),
  `POST /events/{id}/reviews` (ground truth), GET model metadata. PostGIS
  schema in `backend/db/schema.sql`.
- **Frontends**: `frontend/` (Leaflet) and `react-frontend/` (Vite+TS). Both
  currently show demo features; integration swaps those for live `/api/events`
  + `/api/predict`.

## 8. Flow diagram

```
FIRMS VIIRS S-NPP / NOAA-20 / NOAA-21  15,392,720 detections
        |  daily cKDTree 1 km + 12 h
        v
7,589,305 local clusters ──> cross-day edges (323,804)
        |  connected components
        v
7,265,849 global fire events
        |
        +--> P1b external inventory (911 files) --> OSM/GEM gazetteer
        v
P4a tier-1(34) + lineage(54)  P3a industrial(18)   [~60 of 141 computed]
        |
        +--> P7h Land cover (EE WorldCover) 7.27M   [DONE]
        |       +--> P7k NBR (Landsat monthly) 6.76M [DONE]
        |       +--> P7l SAR (S1 IW VV/VH)            [RUN]
        v
P6 dataset: X(60) y(weak|surf) meta   7,265,849 x 60
        |
        v
P7 LightGBM streaming --> Model A (53, operational) + Model B (29, honest)
        |                    |                              |
        |   leakage_probe / P8b verifies A is circular      v
        |                                        SHAP(P7m) + threshold_calibration
        v
FastAPI (model_registry: booster + feature-order + target)
        |  PostGIS + pgvector; /api/predict /explain /reviews /events
        v
Leaflet / React frontend  -->  live map overlay (no more demo points)
```