# AGNI-NETRA: Backend & ML System Architecture

> **Project:** AGNI-NETRA (SIH / Advanced Thermal Source Monitoring & Early Warning System)  
> **Target Dashboard:** v2 Frontend (Leaflet + ECharts + 141-Feature Catalog)  
> **Document Purpose:** Complete specification of **why** this architecture is needed and **how** it is implemented end-to-end.

---

## 1. Why This System Architecture Is Needed

### 1.1 The Operational Problem in India
Satellite-based thermal detection (e.g., NASA FIRMS, MODIS, VIIRS) frequently flags thermal anomalies across India. However, raw thermal detections suffer from severe operational flaws:
1. **High False-Alarm Rates:** Hot bare soil, reflective industrial sheds, solar farms, and brick kilns are frequently misclassified as active wildfires.
2. **Ambiguity in Source Class:** A thermal anomaly at 375m pixel resolution looks identical whether it is:
   - An open forest wildfire in Uttarakhand,
   - Stubble/paddy burning in Punjab/Haryana,
   - A routine petrochemical flare in Gujarat,
   - An underground coal fire in Jharia, Jharkhand.
3. **No Explainability:** Forest rangers and disaster management authorities (NDRF/SDMA) cannot deploy resources based on a "black-box" confidence percentage without knowing *why* the model made that determination.
4. **Cloud Occlusion & Data Sparsity:** Optical satellites cannot see through monsoon clouds; radar (SAR) revisits every 6–12 days; satellite overpasses are intermittent.

### 1.2 Why 141 Features Are Essential
To reliably classify an event and predict its spread, single-source thermal data is inadequate. The 141 features span **11 distinct physical families**:
- **Heat (15 features):** Peak/Median Fire Radiative Power (FRP), 4µm vs 11µm brightness temperature differences, subpixel temperature estimates.
- **Timing (14 features):** Burn duration, autocorrelation across passes, historical recurrence (1-year, 3-year, 5-year cycles).
- **Movement & Spread (17 features):** Centroid displacement, spread speed, elongation, compactness, stationarity index (distinguishing stationary industrial flares from moving wildfires).
- **Terrain (12 features):** Elevation, slope, aspect, Topographic Position Index (TPI), Terrain Ruggedness Index (TRI) via DEM.
- **Fuel & Land Cover (16 features):** Proximity to forests vs croplands vs built-up areas (ESA WorldCover 10m), NDVI anomalies, leaf moisture (NDMI).
- **Industry Nearby (18 features):** Exact distances and boundary checks to refineries, power plants, steel mills, mines, and pipelines (GEM / OSM).
- **Weather & Fire Danger (18 features):** Wind speed, wind gusts, relative humidity, rain, and Canadian Fire Weather Indices (FFMC, DMC, DC, ISI, BUI, FWI).
- **Smoke & Burn Signs (7 features):** Optical smoke plumes, plume length, dark smoke probability, burn scar growth.
- **Radar & Structure (7 features):** Sentinel-1 SAR backscatter changes (VV, VH), coherence loss (penetrating smoke/clouds).
- **Air Chemistry (9 features):** TROPOMI Sentinel-5P gas anomalies ($NO_2$, $SO_2$, $CO$, $CH_4$, $HCHO$) to identify chemical signatures of industrial emissions vs biomass burning.
- **Data Quality (8 features):** Sensor scan angle, view zenith, satellite confidence, cloud cover, and missing feature fractions.

### 1.3 Why Traditional Backends Fail Here
- A standard CRUD database (like simple MongoDB or MySQL) cannot calculate geodesic distances to pipelines, cannot compute intersections with smoke plume polygons, and cannot execute vector similarity searches.
- Running 141 raw geospatial feature extractions synchronously per request would take 15–30 seconds, causing frontend timeouts.
- **Solution:** A **Tiered Feature Store** combining PostgreSQL/PostGIS, Redis in-memory grids, and pre-indexed spatial layers brings end-to-end inference latency under **60 milliseconds**.

---

## 2. System Architecture Overview

```
                                      [ SATELLITE & DATA FEEDS ]
                     NASA FIRMS (Hotspots) | ERA5/IMD (Weather) | Sentinel-1/2/5P | DEM / OSM
                                                  │
                                                  ▼
                                      [ DATA INGESTION ENGINE ]
                           (FastAPI In-Process APScheduler with DB Advisory Lock)
                                                  │
                                                  ▼
                                     [ TIERED FEATURE STORE ]
        ┌─────────────────────────────────────────┼────────────────────────────────────────┐
        ▼                                         ▼                                        ▼
   [ TIER 1: STATIC ]                   [ TIER 2: SLOW DYNAMIC ]                 [ TIER 3: REAL-TIME ]
PostGIS Vector & Raster                 In-Process LRU / Redis                  Fast In-Memory Math
- DEM Terrain (12 feats)                - Hourly Wind Grids (u, v)              - Thermal stats (FRP)
- Industry Infrastructure (18 feats)    - Weather & FWI Indices (18 feats)      - Multi-pass clustering
- ESA WorldCover (16 feats)             - Active event rolling cache            - Ellipse spread dynamics
        └─────────────────────────────────────────┬────────────────────────────────────────┘
                                                  │ (141-Feature Vector assembled in <50ms)
                                                  ▼
                                  [ ML INFERENCE & XAI PIPELINE ]
                                   LightGBM via ONNX Runtime (<2ms)
                                                  │
                                                  ├───────────────► Precomputed TreeSHAP (<5ms lookup)
                                                  ├───────────────► pgvector Exact Cosine Search (<2ms)
                                                  ▼
                                       [ FASTAPI ASYNC REST API ]
                                  Matching docs/api-contract.md
                                                  │
                                                  ▼
                                     [ AGNI-NETRA V2 FRONTEND ]
                                  (Leaflet Map + ECharts Dashboard)
```

---

## 3. How It Will Be Implemented

### 3.1 Technology Stack Selection
1. **API Framework:** `FastAPI` (Python 3.11+)
   - Native async support, high concurrency, auto-generated OpenAPI/Swagger docs.
   - Pydantic models enforcing exact contract shapes defined in `docs/api-contract.md`.
   - Ingests satellite feeds via an in-process `APScheduler` guarded by a PostgreSQL advisory lock (or pinned to a single worker).
2. **Database:** `PostgreSQL 16` + `PostGIS` + `pgvector`
   - **PostGIS:** Spatial indexing (`GIST`) for polygon queries, distance to facilities, and exposure calculations.
   - **pgvector:** 141-dimensional vector embeddings for instant past event similarity (`GET /api/events/{id}/similar`) using exact cosine distance.
3. **Caching Layer:** In-process LRU cache (`cachetools` / async memoization) for wind grids and hourly forecasts (Redis optional for multi-node scale).
4. **Machine Learning Core:**
   - **Model:** `LightGBM` exported to `ONNX` format.
     - *Why:* Native handling of missing values/NaNs (essential when cloud cover blocks optical/SAR features).
   - **Inference Runtime:** `onnxruntime` (C++ backend, execution in ~1.2ms).
   - **Explainability (XAI):** `FastTreeSHAP` precomputed at ingestion time (stored in `event_features.shap_json` for instantaneous $O(1)$ lookups).
5. **Spatial Engine:** `Shapely` + `GeoPandas` + `PyProj`
   - Used for calculating plume envelopes, fire-spread ellipses, and road/school exposure metrics.

---

### 3.2 Database Schema Implementation

```sql
-- 1. Enable Spatial and Vector Extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Places / Monitored Sites
CREATE TABLE places (
    id VARCHAR(32) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    state VARCHAR(100) NOT NULL,
    category VARCHAR(50), -- 'industrial', 'forest', 'farmland', 'mine'
    geom GEOMETRY(Point, 4326) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX idx_places_geom ON places USING GIST(geom);

-- 3. Thermal Events (Satellite Hotspot Clusters)
CREATE TABLE events (
    id VARCHAR(32) PRIMARY KEY,
    place_id VARCHAR(32) REFERENCES places(id),
    detected_at TIMESTAMP WITH TIME ZONE NOT NULL,
    geom GEOMETRY(Point, 4326) NOT NULL,
    frp_median DOUBLE PRECISION,
    predicted_class VARCHAR(50) NOT NULL, -- 'wildfire' | 'agricultural_burning' | 'gas_flare' | 'industrial' | 'mining' | 'unknown'
    confidence DOUBLE PRECISION,          -- 0.0 to 1.0 (calibrated against heuristic labeling rules)
    baseline_status VARCHAR(20) NOT NULL DEFAULT 'routine', -- 'routine' | 'abnormal' (site-level baseline deviation)
    status VARCHAR(30) NOT NULL DEFAULT 'unreviewed',       -- 'unreviewed' | 'confirmed' | 'false_alarm' (analyst review workflow)
    missing_fraction DOUBLE PRECISION DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX idx_events_geom ON events USING GIST(geom);
CREATE INDEX idx_events_detected_at ON events(detected_at DESC);
CREATE INDEX idx_events_baseline_status ON events(baseline_status);
CREATE INDEX idx_events_status ON events(status);

-- 4. 141 Features, Precomputed SHAP & Versioned Embeddings
CREATE TABLE event_features (
    event_id VARCHAR(32) PRIMARY KEY REFERENCES events(id) ON DELETE CASCADE,
    feature_schema_version INT DEFAULT 1, -- Protects pgvector embeddings from silent drift
    features_json JSONB NOT NULL,         -- 141 features (core populated, unbuilt marked missing: true)
    shap_json JSONB,                      -- Precomputed SHAP values (O(1) instant lookup)
    embedding VECTOR(141)                 -- Standardized vector for similarity search
);
CREATE INDEX idx_event_schema_version ON event_features(feature_schema_version);

-- INDEXING NOTE (Conditional / Deferred):
-- 1. GIN on features_json: DEFERRED. Since the API fetches features_json as a whole
--    document by primary key (event_id), a GIN index adds write overhead without
--    accelerating lookups. Add only once individual JSON keys are filtered in WHERE clauses.
--    -- CREATE INDEX idx_event_features_json ON event_features USING GIN(features_json);
-- 2. IVFFlat on embedding: DEFERRED. For demo/initial scale (<50,000 events), brute-force
--    exact cosine search (`ORDER BY embedding <=> query LIMIT k`) runs in < 2ms with 100% recall.
--    IVFFlat requires thousands of rows to train cluster centroids properly and hurts recall on small datasets.
--    -- CREATE INDEX idx_event_embedding ON event_features USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- 5. Geospatial Overlays (Plume & Spread Polygons)
CREATE TABLE event_geometries (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR(32) REFERENCES events(id) ON DELETE CASCADE,
    layer_type VARCHAR(50) NOT NULL,    -- 'plume_core', 'plume_outer', 'spread_outlook', 'footprint'
    horizon_hours INT,
    polygon GEOMETRY(Polygon, 4326) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX idx_event_geometries_polygon ON event_geometries USING GIST(polygon);

-- 6. Analyst Reviews & Ground Truth Feedback
CREATE TABLE analyst_reviews (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR(32) REFERENCES events(id) ON DELETE CASCADE,
    verdict VARCHAR(20) NOT NULL,       -- 'confirm', 'change', 'false', 'field'
    analyst_class VARCHAR(50),          -- Matches 6-class taxonomy if reclassified
    note TEXT,
    reviewer VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX idx_analyst_reviews_event ON analyst_reviews(event_id);
```

---

### 3.3 Pragmatic "Before You Begin" Engineering Decisions

Since this backend is starting from scratch (currently running on frontend sample data), these 7 battle-tested decisions avoid premature complexity:

1. **Lightweight In-Process Scheduler with Concurrency Guard (Drop Celery):**  
   Replace Celery/Redis queue with **APScheduler** running inside the FastAPI process. To prevent duplicate polling of NASA FIRMS when multiple workers run:
   - **Deployment constraint:** Explicitly pin the backend deployment to a single worker process (`uvicorn app.main:app --workers 1`). This is recommended for demo and hackathon scale.
   - **Multi-worker lock (optional):** If running with multiple workers, guard the scheduled job using a PostgreSQL session advisory lock (`SELECT pg_try_advisory_lock(849201)`). If the lock is not acquired, another worker is already polling, preventing duplicate requests and write collisions.
2. **Defensible Heuristic Rule-Based Labeling:**  
   Clean ground-truth datasets for "industrial vs stubble vs wildfire" across India do not exist off-the-shelf. Instead of fabricating arbitrary labels or assuming complex probabilistic label models, use **deterministic if/else heuristic rules** based on spatial proximity, season, and land cover:
   - `Inside GEM industrial boundary / < 1km to refinery/plant` + `high stationarity` $\rightarrow$ **`gas_flare`** or **`industrial`**
   - `Inside ESA Cropland` + `burning season (Oct–Nov / Apr–May)` + `fast spread` $\rightarrow$ **`agricultural_burning`**
   - `Inside ESA Forest` + `high FWI / slope` $\rightarrow$ **`wildfire`**
   - `Inside active mine polygon / coalfield` $\rightarrow$ **`mining`**
   - Otherwise $\rightarrow$ **`unknown`**  
   This deterministic heuristic pattern is completely transparent and defensible when judges ask how training labels were derived.
3. **Core Feature Trimming (~50–60 Active, Rest Schema-Ready):**  
   Building real extraction pipelines for all 141 features—especially Sentinel-1 SAR and TROPOMI chemistry—is extremely labor-intensive.  
   - **Actively implement:** Thermal (15), Terrain (12), Fuel/LandCover (16), Industry proximity (18), and Weather/FWI (18).
   - **Keep in schema:** Mark SAR and Chem features as `null` with `"missing": true`. The frontend's 141-feature explorer still renders honestly without blocking backend progress.
4. **Precomputed SHAP at Event Creation:**  
   Do **not** compute SHAP live during `GET /api/events/{id}/shap`. Calculate SHAP once when the event is ingested or updated, and store the result in `event_features.shap_json`. Serving SHAP becomes an instantaneous $O(1)$ database lookup with zero demo latency spikes.
5. **Schema Versioning & Deferred Indexes for Vector Embeddings:**  
   - The `feature_schema_version` column in `event_features` ensures that if feature definitions or normalizations change, outdated embeddings are not silently compared against new embeddings in `pgvector`.
   - `ivfflat` and `GIN` indexes are deferred: for demo-scale tables ($<50,000$ events), exact brute-force `ORDER BY embedding <=> query LIMIT k` runs in $<2\text{ ms}$ with $100\%$ recall, avoiding the accuracy loss and centroid training requirements of IVFFlat.
6. **Lean In-Process Caching (Redis Optional for MVP):**  
   Use an in-process LRU cache (`cachetools` or Python `functools.lru_cache`) with TTL for wind grids and weather forecast lookups. Redis is added only if multiple backend worker replicas are deployed later.
7. **Calibrated Confidence Scores:**  
   Confidence is calibrated (via isotonic regression or Platt scaling) against our heuristic weak-supervision labels to reduce raw softmax overconfidence — it reflects consistency with our labeling rules, not verified ground-truth accuracy.

---

### 3.4 Endpoints & Pydantic Schema Strategy

#### Key Entity Models
```python
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime

PredictedClass = Literal[
    "wildfire",
    "agricultural_burning",
    "gas_flare",
    "industrial",
    "mining",
    "unknown"
]

BaselineStatus = Literal["routine", "abnormal"]
AnalystStatus = Literal["unreviewed", "confirmed", "false_alarm"]

class EventSummary(BaseModel):
    id: str
    place_id: Optional[str] = None
    detected_at: datetime
    lat: float
    lon: float
    frp_median: float
    predicted_class: PredictedClass
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated consistency with labeling rules")
    baseline_status: BaselineStatus = Field(
        default="routine", 
        description="'routine' for normal site operations (e.g. permitted flare), 'abnormal' for unexpected excursions"
    )
    status: AnalystStatus = Field(
        default="unreviewed", 
        description="Analyst review workflow state"
    )
    missing_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
```

#### API Endpoints Overview
Each API endpoint maps directly to the shapes required by the frontend ([docs/api-contract.md](file:///c:/Users/NITISH%20SINGH/Downloads/agni-netra-v2/agni-netra-v2/docs/api-contract.md)):

| Endpoint | Method | Source & Logic | Latency Budget |
| :--- | :--- | :--- | :--- |
| `/api/events` | GET | List events with `predicted_class` (6-class taxonomy), `baseline_status`, and `status`. | < 15ms |
| `/api/events/{id}/features` | GET | Reads `event_features.features_json` (core real values + missing flags). | < 10ms |
| `/api/events/{id}/shap` | GET | Instant lookup from precomputed `event_features.shap_json`. | < 5ms |
| `/api/events/{id}/forecast`| GET | Fetches 73-hour weather projection (cached in-process). | < 5ms |
| `/api/wind` | GET | Returns gridded $u$ and $v$ wind vectors from in-memory cache. | < 10ms |
| `/api/events/{id}/plume` | GET | Computes Gaussian trajectory & dispersion polygon from wind bearing & speed. | < 15ms |
| `/api/events/{id}/spread` | GET | Produces 1h, 3h, 6h Huygens elliptical fire-front outlook for `wildfire` and `agricultural_burning`. | < 20ms |
| `/api/events/{id}/exposure`| GET | PostGIS `ST_Intersects` query counting settlements, schools, clinics inside plume. | < 25ms |
| `/api/events/{id}/similar` | GET | Exact `pgvector` cosine similarity (`<=>`) filtered by `feature_schema_version`. | < 10ms |
| `/api/events/{id}/reviews` | POST | Inserts analyst classification verdict (`confirm`, `change`, `false`, `field`) into DB. | < 10ms |

---

## 4. Phased Implementation Roadmap (Lean & Demo-Focused)

### Phase 1: Database Setup & Mock Contract Bridge (Week 1)
- [x] Frontend deployed on Vercel (`konsors-sih2.vercel.app`).
- [x] Architecture doc updated with pragmatic lean constraints.
- [ ] Initialize Supabase / Neon PostgreSQL instance with PostGIS and pgvector.
- [ ] Create FastAPI app with CORS middleware and Pydantic schemas from `api-contract.md` (pinned to 1 worker).
- [ ] Bridge frontend: Replace `sample.js` in frontend with live `fetch()` calls to the FastAPI backend.

### Phase 2: Core Static GIS & Heuristic Rule-Based Labeling (Week 2)
- [ ] Ingest SRTM 30m DEM for India and Global Energy Monitor (GEM) facilities.
- [ ] Implement the core ~50–60 feature pipelines (Terrain, Industry distance, LandCover, Weather).
- [ ] Build the deterministic heuristic rule-based labeling pipeline for 6 classes (`wildfire`, `agricultural_burning`, `gas_flare`, `industrial`, `mining`, `unknown`).
- [ ] Train LightGBM classifier on heuristic labels and calibrate confidence via Isotonic Regression.

### Phase 3: Precomputed SHAP, Embeddings & Reviews (Week 3)
- [ ] On event ingestion, precompute TreeSHAP contributions and store in `event_features.shap_json`.
- [ ] Generate 141-D embeddings (with versioning `feature_schema_version = 1`) and query with exact brute-force cosine distance.
- [ ] Wire up `/api/events/{id}/reviews` so analyst confirm/reject inputs persist to database.

### Phase 4: NRT Satellite Polling & Live Polish (Week 4)
- [ ] Add `APScheduler` cron in FastAPI pulling NASA FIRMS VIIRS 375m active fires every 15 minutes (with DB advisory lock or single worker).
- [ ] Fast DBSCAN clustering of raw thermal pixels into discrete `events` with baseline status tagging (`routine` vs `abnormal`).
- [ ] Generate indicative Huygens ellipses (for `wildfire` and `agricultural_burning`) and smoke plume cones.
- [ ] Rehearse live judge demo showing explainability, exposure analysis, and analyst review loop.

---

## 5. Summary: Why This Architecture Wins

1. **Scientifically Defensible:** Replaces black-box guesses with calibrated probabilities and transparent heuristic rules that judges will respect.
2. **Rock-Solid Demo Stability:** Pinned single-worker scheduling (or DB advisory lock) and deferred indexing prevent deadlocks, double-polling, and memory spikes during presentations.
3. **Sub-15ms Dashboard Latency:** Precomputing SHAP and embeddings at ingestion turns expensive AI operations into instant $O(1)$ lookups.
4. **Clean Operational Separation:** `baseline_status` distinguishes routine permitted activity from anomalies, while `status` handles analyst triage.
5. **Honest Multimodal Handling:** Real pipelines for the 50–60 highest-impact features, with clean schema-level support for future SAR and atmospheric chemistry expansions.

