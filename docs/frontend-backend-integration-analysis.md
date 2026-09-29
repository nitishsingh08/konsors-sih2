# AGNI-NETRA frontend/backend integration analysis

## Executive summary

The `frontend` is a static HTML/JavaScript application. It currently makes **no backend API
requests**: there is no `fetch`, Axios, XMLHttpRequest, WebSocket, GraphQL client, API base URL,
or environment configuration in the source tree. The application is populated by deterministic
sample generators in:

- `frontend/js/data/sample.js` — places, events, detections, histories, baselines and context.
- `frontend/js/data/features.js` — the 141-feature vector and sample SHAP values.
- `frontend/js/data/weather.js` — forecast, wind grids, plume, spread, footprints, exposure and
  similar-event data.
- `frontend/js/analysis/tabs.js` — browser-local analyst reviews.

The intended live integration is described partly in `docs/api-contract.md`. That document names
10 v2 endpoint groups. The README also refers to unchanged v1 endpoints for places, events, site
history, land cover, imagery and weather summary, but their exact paths and JSON contracts are not
present in this repository. Those v1 endpoints must be defined before implementation.

The backend tree currently contains only package initializers and is not runnable: there is no
FastAPI entrypoint, router, schema, database setup, migration, configuration, or implemented
endpoint.

## What the frontend actually needs

### A. Core data required by the main application

These are inferred from the fields consumed throughout `ui.js`, `pages.js`, `leaflet-map.js` and
`sample.js`. The exact URL paths are recommendations because the v1 contract is missing.

| Recommended API | Purpose | Minimum data required |
|---|---|---|
| `GET /api/places` | Populate the command map, search, regions and watchlist | `id`, `kind`, `code`, `name`, `state`, `district`, `lat`, `lon`, `class`, `type`, `median_frp_mw`, `mad`, `coverage`, `first_seen_at` |
| `GET /api/events` | Populate the time-window map, KPIs, priority queue and alerts | `id`, `place_id`, `observed_at`, `class`, `confidence`, `alternative_class`, `alternative_probability`, `status`, `z_score`, `frp_mw`, `sensor`, `detection_count`, `area_km2`, `needs_review`, local satellite-pass hour |
| `GET /api/places/{place_id}/events` | Place report history and event selection | Same event shape as above, sorted by observation time; pagination is recommended |
| `GET /api/events/{event_id}/detections` | Raw detection layer and density hexagons | Point coordinates as `[lon, lat]`; preferably sensor, FRP, timestamp and confidence per point |
| `GET /api/places/{place_id}/history` | 12-month timeline, baseline and comparison charts | `observations[]` with `observed_at`, `frp_mw`, `detection_count`, `sensor`, `z_score`; `gaps[]` with start/end; baseline summary |
| `GET /api/places/{place_id}/context` | Place report land-cover, facility, terrain and image cards | Land-cover percentages, nearest-facility distances, NDVI/NBR/NDBI, clear-pixel fraction, image age, VPD, recent rain, elevation, slope, coverage |
| `GET /api/places/{place_id}/nearby` | “Other heat nearby” map/card | Nearby source name, class, distance km, bearing/radian or coordinates, status |
| `GET /api/metadata/model` | Model-health page | Per-class precision/recall/F1/example counts, confusion matrix, model version, feature-set version, source freshness |
| `GET /api/metadata/sources` | Freshness and quality displays | Source name, observed/updated timestamp, age, target/status and note |

The frontend currently derives some of these responses from local data rather than making separate
calls. A backend may combine them into a single event/place-detail response for fewer round trips,
but the adapter must still expose the internal shapes used by the UI.

### B. Explicitly documented v2 endpoints

These paths and wire formats are already specified in `docs/api-contract.md`.

| Endpoint | Consumer | Important response requirements |
|---|---|---|
| `GET /api/events/{id}/features` | Feature groups, feature explorer and CSV export | `event_id`, `features` keyed by all 141 feature names, each with `value`, `percentile`, `missing`; `missing_fraction` |
| `GET /api/events/{id}/shap` | “Why” card, group contributions and feature explorer | `event_id`, `by_family`, `top_features[]` with feature key and signed contribution |
| `GET /api/events/{id}/forecast` | 72-hour timeline, weather card, plume/spread calculations | 73 hourly values for wind speed/gust, wind-from direction, temperature, RH, rain and FFMC; DMC/DC and previous-24-hour rain |
| `GET /api/wind?bbox={west,south,east,north}&hour={hour}` | National map wind layer | Regular row-major grid: `lo0`, `la1`, `dx`, `dy`, `nx`, `ny`, `u[]`, `v[]`; u/east and v/north in m/s |
| `GET /api/events/{id}/wind-grid?hour={hour}` | Advanced analysis wind layer | Same grid shape, centered on the selected event |
| `GET /api/events/{id}/plume?hours={hours}` | Smoke layer and smoke summary | `reach_km`, `bearing_deg`, `trajectory`, `core`, `outer`; coordinates are `[lon, lat]` |
| `GET /api/events/{id}/spread?horizons=1,3,6` | Indicative fire-spread layer/table | `null` or omitted for non-vegetation classes; otherwise horizons with hours, reach, rate range, heading, area and rings plus `uncertainty_ring` |
| `GET /api/events/{id}/footprints` | Last eight observed footprint passes | `frames[]` with timestamp, area and closed polygon ring |
| `GET /api/events/{id}/exposure` | Settlements, schools, clinics, roads and water inside zones | `zones[]` with name, area, settlements, schools, clinics, road km and water bodies |
| `GET /api/events/{id}/similar` | Similar-events tab | Ranked `matches[]` with event id, similarity and outcome (`confirmed`, `false_alarm`, `unreviewed`) |
| `POST /api/events/{id}/reviews` | Persist analyst review | Body includes `event_id`, `verdict`, optional `analyst_class`, `note`, timestamp and authenticated reviewer |
| `GET /api/reviews` | Shared review history/export | Review records, preferably filterable by event, date, verdict, reviewer and pagination |

The v2 document calls the above “exact JSON shapes”, but the current UI does not consume those
wire shapes directly. An integration adapter is still required. Examples:

- Forecast wire fields such as `wind_speed_ms` and `wind_from_deg` must become the internal
  `W.spd` and `W.dir` arrays. The UI also needs `vpd[]`, `n`, and `t0`; VPD can be returned by the
  API or derived from temperature and RH.
- SHAP `by_family`/`top_features` must become the internal `{ fam, all, feats }` shape used by
  `shapOf()` and the feature explorer.
- Plume `trajectory` and `bearing_deg` must become `traj` and `bearing`; the UI also expects
  `head` and `H`.
- Spread `hours`, `reach_km`, `rate_kmh_lo`, `rate_kmh_hi`, `heading_deg` and `area_km2` must
  become the internal `t`, `D`, `rateLo`, `rateHi`, `dirTo` and `area` fields. The advanced
  analysis also reads `meanSlope`, so either return it or make that UI copy conditional.
- Feature records need the static metadata from `FDEF` joined with the API values so the UI can
  render labels, units, groups and “feeds labels” flags.

## Features that do not currently call the backend

The following are UI-only or currently mocked and should not be treated as existing API
integrations:

- CSV and GeoJSON exports are generated in the browser from the current in-memory event data.
- CAP alert XML is only previewed/copied. The “Mock send to reviewer” action explicitly transmits
  nothing and must not be connected to a send operation without an authorization workflow.
- Watchlist state is stored in `localStorage` under `agni-watch`.
- Analyst reviews are stored in `localStorage` under `agni-reviews` until the review endpoints are
  wired.
- Theme, map position, selected filters and some UI state are local/browser state.
- Terrain profiles, nearby sources and the local advanced-analysis map are generated locally.
- The “model health”, source freshness and most regional aggregates are hard-coded sample values.

The generated `frontend/dist/agni-netra-single.html` is a bundled copy of the same static app, not
a separate API client. It must be rebuilt after source integration changes.

## Connection requirements

### Frontend changes

1. Add a runtime API base URL, for example `window.AGNI_API_BASE` or a small configuration file
   loaded before the application scripts. Do not hard-code a development host into every data
   function. The deployed static frontend needs separate dev/staging/production values.
2. Add one shared HTTP client with:
   - URL joining and query encoding;
   - `AbortController` timeouts/cancellation;
   - JSON content negotiation;
   - consistent handling for 401/403, 404, 409, 422 and 5xx responses;
   - request IDs and useful console/reporting diagnostics;
   - retry rules only for safe idempotent GET requests;
   - response validation before data reaches the render functions.
3. Replace or wrap the sample functions rather than mixing sample and live values silently. The
   header badge and per-card “Sample” labels should be driven by a live-data flag and response
   freshness metadata.
4. Add loading, empty, stale and error states. Current rendering assumes every value exists and
   immediately calls methods such as `.toFixed()`, `.map()` and `.find()`.
5. Decide whether data is loaded eagerly or on demand. A practical first pass is:
   - load places and the current event window on startup;
   - load place history/context on opening a report;
   - load forecast/analysis endpoints when advanced analysis opens;
   - load feature, SHAP, exposure, similar and reviews per analysis tab.
6. Add an event/place cache keyed by ID and query parameters. Forecasts, footprints and feature
   vectors should not be refetched on every tab repaint or timeline tick.
7. Replace local review persistence with optimistic or confirmed `POST` handling, and retain a
   local offline queue only if offline review entry is a product requirement.
8. Rebuild `dist/agni-netra-single.html` after integrating the source files.

### Backend/API requirements

- Serve HTTPS JSON endpoints and enable CORS for the deployed frontend origin.
- Choose one stable API prefix/version, preferably `/api/v1`; either update the documented paths
  or provide a reverse-proxy alias from `/api/...`.
- Use ISO-8601 timestamps with timezone offsets on the wire. The UI displays IST but should not
  receive display-formatted strings such as `20 Sep 2026, 14:30 IST` for machine fields.
- Preserve units: wind m/s, distances km, area km², FRP MW, temperature °C, RH percent,
  coordinates `[lon, lat]`, wind directions in meteorological “from” degrees.
- Return `null`/`missing: true` for unavailable observations; do not convert missing data into
  zero. This matters for cloud, radar revisit, facility coverage and satellite chemistry.
- Validate class/status enums. Current class IDs are `wildfire`, `agricultural_burning`,
  `gas_flare`, `mining`, `industrial`, `unknown`; statuses are `abnormal`, `routine`,
  `baseline_building`, `not_applicable`.
- Support pagination and server-side filtering for events and reviews. The current sample has a
  one-year window but the live data will grow.
- Version model and feature-set metadata. The place report shows model and feature versions and
  the analyst review flow depends on knowing which model produced an event.
- Add authorization for analyst reviews and any future alert action. CAP draft generation can be
  client-side, but publication/sending must be a separate authenticated, audited backend action.
- Add health/readiness endpoints for API, database, object storage, model artifacts and ingestion
  freshness.

## Backend data/services needed behind the APIs

The empty backend scaffold indicates the intended service boundaries. To produce the responses,
the implementation needs at least:

- FIRMS/VIIRS/MODIS ingestion and event clustering;
- persistent place/site registry and event history;
- model inference, calibrated confidence, class/status and SHAP explanations;
- the 141-feature builder with missingness and percentile/reference distributions;
- forecast provider plus Canadian FWI calculations or equivalent server-side derived values;
- wind-grid generation and forecast-version tracking;
- plume/dispersion calculation and indicative spread calculation;
- DEM/terrain, WorldCover/HLS, GEM/OSM facility data;
- OSM/population spatial joins for exposure counts;
- similarity search over historical feature vectors and reviewed outcomes;
- authenticated analyst review storage and audit history;
- source freshness/quality metrics and model metadata.

The intended directory plan in `docs/backend-directory-structure.md` already names modules for
events, features, predictions, ingestion, reviews, models and jobs. Those files have not yet been
implemented.

## Recommended integration order

1. Define the missing v1 contracts for places, events, history, context, detections, metadata and
   nearby sources. Freeze IDs, enums, timestamps, units and coordinate order.
2. Implement `/health` plus read-only `places` and `events` endpoints and connect the command map.
3. Connect place history/context and replace the static place report.
4. Connect forecast and wind-grid endpoints; then replace local plume/spread calculations with
   response adapters.
5. Connect features and SHAP, preserving the static `FDEF` catalogue for labels and units.
6. Connect footprints, exposure and similar events.
7. Connect reviews with authentication, reviewer identity and audit logging.
8. Replace model-health/source-freshness sample constants and add contract tests against fixture
   responses.
9. Add failure/stale-data UI states, then rebuild and test both `index.html` and the bundled dist
   file.

## Current blockers and decisions to resolve

- No live API client or API base URL exists in the frontend.
- The v1 endpoint paths and response schema are referenced but missing.
- The backend is a directory skeleton only.
- The existing v2 wire contracts do not exactly match the internal objects consumed by the UI;
  adapters or UI refactoring are required.
- Authentication/authorization is not defined. Reviews can be shared only after user identity,
  permissions and audit requirements are agreed.
- The production map tile provider is not configured. The frontend currently uses the public OSM
  tile URL, which is suitable for the demo but should be replaced or fronted by an approved tile
  provider for production traffic.

## Source evidence reviewed

- `frontend/index.html` — script load order and static asset loading.
- `frontend/js/data/sample.js` — sample place/event/history/context model.
- `frontend/js/data/features.js` — 141 feature definitions and sample feature/SHAP generators.
- `frontend/js/data/weather.js` — sample forecast, grid, plume, spread, footprint, exposure and
  similar-event generators.
- `frontend/js/ui.js`, `frontend/js/pages.js` — main map, reports, regions, watchlist, model and
  alert consumers.
- `frontend/js/analysis/analysis.js`, `frontend/js/analysis/tabs.js` — advanced analysis consumers
  and browser-local reviews.
- `docs/api-contract.md` — explicit v2 endpoint shapes.
- `docs/backend-directory-structure.md` and `backend/` — intended but unimplemented backend.

