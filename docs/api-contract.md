# API contract, version 2

This lists the JSON shapes the frontend needs. Everything here is currently produced by
`js/data/sample.js`, `js/data/features.js` and `js/data/weather.js`. Replace those functions
with `fetch` calls returning these shapes, and the UI needs no other changes.

Version 1's contract (places, events, site history, SHAP, land cover, imagery, weather
summary) still applies — see the version 1 README if you kept it. This file covers what
version 2 adds: the 6-class taxonomy with baseline/workflow statuses, the 141-feature vector,
the wind grid, the smoke plume, the fire-spread outlook, exposure counts, similar events, and analyst reviews.

## 0. Event Entity & Taxonomy — `GET /api/events` and `GET /api/events/{id}`

Each event object carries classification, baseline deviation, and analyst workflow status:

```json
{
  "id": "E0218",
  "place_id": "P0042",
  "detected_at": "2026-09-20T14:15:00+05:30",
  "lat": 15.1245,
  "lon": 76.6214,
  "frp_median": 99.4,
  "predicted_class": "wildfire",
  "confidence": 0.88,
  "baseline_status": "abnormal",
  "status": "unreviewed",
  "missing_fraction": 0.08
}
```

- **`predicted_class`**: One of the six standardized classes:
  - `"wildfire"` (forest / natural vegetation fire)
  - `"agricultural_burning"` (crop residue / stubble burning)
  - `"gas_flare"` (petrochemical / refinery flare stack)
  - `"industrial"` (steel mills, cement kilns, factory thermal sources)
  - `"mining"` (coal fires, active mine workings)
  - `"unknown"` (insufficient or conflicting signature)
- **`confidence`**: `0.0` to `1.0`. Calibrated (via isotonic regression / Platt scaling) against our heuristic labeling rules to reduce raw softmax overconfidence — it reflects consistency with labeling rules, not verified ground-truth accuracy.
- **`baseline_status`**: `"routine"` vs `"abnormal"`. Identifies site-level operational deviation (e.g. a routine permitted flare at a known refinery is `"routine"`, whereas an unexpected fire or flare excursion is `"abnormal"`).
- **`status`**: Analyst triage workflow state (`"unreviewed"`, `"confirmed"`, `"false_alarm"`). Separate from `baseline_status`.

## 1. Feature vector — `GET /api/events/{id}/features`

One entry per feature, keyed by the names in `js/data/features.js` (`FDEF`). 141 keys total,
grouped as `thermal`, `temporal`, `spread`, `terrain`, `fuel`, `industry`, `weather`, `smoke`,
`sar`, `chem`, `quality`.

```json
{
  "event_id": "E0218",
  "features": {
    "FRP_MEDIAN": { "value": 99.4, "percentile": 91.2, "missing": false },
    "BT_I4_I5_DIFF_MEDIAN": { "value": 38.1, "percentile": 64.0, "missing": false },
    "SMOKE_PROBABILITY": { "value": null, "percentile": null, "missing": true }
  },
  "missing_fraction": 0.08
}
```

- `value` is in the real unit shown to the person (MW, km, %, K, and so on) — the frontend does
  the log-display conversion; send the real-world number, not the log-transformed model input.
- `percentile` (0–100) is this event's value against similar past events. Send `null` when you
  don't have a reference distribution yet (booleans and angle features never need one).
- `missing: true` for anything cloud, radar revisit, or a missing facility record blocked. The
  frontend never fabricates a value for a missing feature.

## 2. Model explanation — `GET /api/events/{id}/shap`

```json
{
  "event_id": "E0218",
  "by_family": { "thermal": 0.34, "temporal": 0.28, "industry": 0.41, "fuel": -0.12, "...": 0 },
  "top_features": [
    { "key": "DIST_NEAREST_FACILITY_LOG", "contribution": 0.30 },
    { "key": "RECURRENCE_RATE_3Y", "contribution": 0.24 }
  ]
}
```

`contribution` is signed (positive pushes toward the predicted class). Feature labels, units and
which features "also feed labels" come from the static catalogue in `features.js`, so the
backend does not need to send display text — just key and number.

## 3. Weather forecast — `GET /api/events/{id}/forecast`

Hourly, 0 to 72 hours from the event time. This is the one response every wind, plume and
spread calculation is built from.

```json
{
  "event_id": "E0218",
  "issued_at": "2026-09-19T16:30:00+05:30",
  "hours": 73,
  "wind_speed_ms": [3.1, 3.3, "...73 values"],
  "wind_gust_ms": [4.6, 4.9, "..."],
  "wind_from_deg": [128, 130, "..."],
  "temperature_c": [30.2, 29.8, "..."],
  "relative_humidity_pct": [58, 61, "..."],
  "rain_mm": [0, 0, "..."],
  "ffmc": [82.1, 81.9, "..."],
  "dmc": 40.2,
  "dc": 310.5,
  "rain_past_24h_mm": 0
}
```

`wind_from_deg` is the compass bearing the wind is blowing FROM (meteorological convention),
0 to 360. `ffmc` is hourly; `dmc`/`dc` update once a day, repeat the latest value across hours.
If you don't run the fire-weather codes yourself, send temperature/humidity/wind/rain only and
compute FFMC/DMC/DC/ISI/BUI/FWI client-side with the formulas in `js/data/weather.js`
(`calcISI`, `calcBUI`, `calcFWI` — standard Canadian FWI system).

## 4. Wind grid (map layer) — `GET /api/wind?bbox=...&hour=0` and `GET /api/events/{id}/wind-grid?hour=0`

A regular lat/lon grid of u (east) and v (north) wind components in m/s.

```json
{
  "lo0": 76.5, "la1": 15.2, "dx": 0.08, "dy": 0.08, "nx": 45, "ny": 45,
  "u": [1.2, 1.3, "... nx*ny values, row-major from la1 downward"],
  "v": [0.4, 0.5, "..."]
}
```

This is exactly the shape `WindLayer.setGrid()` and `gridSample()` in `js/map/wind-layer.js`
and `js/data/weather.js` expect. The national map calls `/api/wind`, the analysis workspace
calls the per-event version centered on the source. Typical grid: 30–50 cells per side is
enough for the visual; the frontend interpolates between cells.

## 5. Smoke plume — `GET /api/events/{id}/plume?hours=6`

```json
{
  "event_id": "E0218", "hours": 6, "reach_km": 43.2, "bearing_deg": 312,
  "trajectory": [[76.62, 15.12], [76.60, 15.15], "... one point per 30 min"],
  "core": [[76.63, 15.10], "... closed ring, core concentration"],
  "outer": [[76.64, 15.09], "... closed ring, wind-uncertainty envelope"]
}
```

Coordinates are `[lon, lat]`. `core` and `outer` are both closed rings (first point repeated or
implicitly closed — the frontend closes it either way). If your dispersion model already
outputs a proper Gaussian-plume or HYSPLIT concentration field, `core` can be a real isopleth
instead of the simple widening wedge the sample uses — the frontend just draws whatever ring
you send.

## 6. Fire spread outlook — `GET /api/events/{id}/spread?horizons=1,3,6`

Wildfire and agricultural-burning events only. Omit or return `null` for every other class —
the frontend already shows "Not applicable" in that case and never asks for this endpoint's
data if the event's class is not one of those two.

```json
{
  "event_id": "E0218", "class": "High",
  "horizons": [
    { "hours": 1, "reach_km": 0.7, "rate_kmh_lo": 0.4, "rate_kmh_hi": 1.3, "heading_deg": 315, "area_km2": 0.1, "ring": [["lon","lat"], "..."] },
    { "hours": 3, "...": "..." },
    { "hours": 6, "...": "..." }
  ],
  "uncertainty_ring": [["lon","lat"], "..."]
}
```

`class` is one of `Low`, `Moderate`, `High`, `Very high` (from ISI, see `spreadClass()`).
`ring` is the outlook boundary at that horizon; `uncertainty_ring` is the wider "could go
further" boundary drawn dashed. Always label this indicative in your own UI copy if you add
any — the frontend's copy already says "not a validated fire-front forecast" and that framing
should not be softened when real modelling replaces the sample ellipse.

## 7. Observed footprint history — `GET /api/events/{id}/footprints`

```json
{ "event_id": "E0218", "frames": [
  { "t": "2026-09-16T09:00:00+05:30", "area_km2": 0.4, "ring": [["lon","lat"], "..."] },
  "... up to the last 8 overpasses"
]}
```

## 8. Exposure — `GET /api/events/{id}/exposure`

Counts inside the smoke-plume core (6 h) and, for vegetation fires, the 6-hour spread outlook.

```json
{ "zones": [
  { "name": "Smoke plume, core, next 6 hours", "area_km2": 40.6, "settlements": 3, "schools": 1, "clinics": 0, "road_km": 12, "water_bodies": 1 }
]}
```

Source these from OpenStreetMap features inside each polygon plus a gridded population
product. OSM is uneven in rural India — say so in your own release notes, the way this build's
sample banner does.

## 9. Similar events — `GET /api/events/{id}/similar`

```json
{ "matches": [
  { "event_id": "E0091", "similarity": 0.88, "outcome": "confirmed" }
]}
```

`outcome` is one of `confirmed`, `false_alarm`, `unreviewed`. Compute similarity however you
like (nearest neighbours on the feature vector is the obvious start); the frontend just ranks
and displays whatever you send, highest similarity first.

## 10. Analyst reviews — `POST /api/events/{id}/reviews`, `GET /api/reviews`

The frontend currently stores reviews in the browser's own storage (`localStorage`), so a
review made on one device is not seen by anyone else. That is the one piece you should wire up
early, since these reviews are your path to real training labels. Body shape, matching what
`js/analysis/tabs.js` (`REVIEWS`, `reviewsCsv`) already writes:

```json
{ "event_id": "E0218", "verdict": "change", "analyst_class": "mining", "note": "text", "at": "20 Sep 2026, 14:30 IST", "reviewer": "optional, once you add auth" }
```

`verdict` is one of `confirm`, `change`, `false`, `field`. Once this is a real endpoint, swap
`js/analysis/tabs.js`'s `saveReviews()`/`reviewsOf()` for `fetch` calls — the rest of the tab's
rendering code does not need to change.

## Things to keep in mind

- **Load order matters in the current build.** `js/core.js` must load before
  `js/map/wind-layer.js`, `js/map/leaflet-map.js` and `js/analysis/analysis.js`, and
  `js/analysis/analysis.js` must load before `js/analysis/tabs.js` (`tabs.js`'s `ATAB` object
  needs `tabWind`, defined in `analysis.js`). If you introduce a bundler, this ordering
  constraint goes away naturally; until then, do not reorder `index.html`'s script tags.
- **OSM tiles.** `js/map/leaflet-map.js` has one constant, `OSM_URL`, pointing at the public
  OpenStreetMap tile server. That is fine for this demo and not fine for production traffic —
  point it at your own tile provider before real use, per OSM's tile usage policy.
- **Units.** Wind speed is always metres per second on the wire; the frontend converts to
  km/h for display. Distances are kilometres. Keep that boundary so the unit conversions in
  `js/data/weather.js` don't end up double-applied.
