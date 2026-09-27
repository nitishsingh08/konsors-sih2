# AGNI-NETRA dashboard — version 2

Everything from version 1 (the command map, place reports, regional analytics, watchlist,
model health, methods and alert hand-off pages) plus:

- **A real map.** Leaflet with OpenStreetMap tiles, with an offline plain-outline fallback if
  tiles can't be reached (tested by literally blocking the tile requests — it falls back
  cleanly and tells the person why).
- **Wind on the map.** An animated particle layer (or a still arrow field) built on a small
  Canvas-based Leaflet layer (`js/map/wind-layer.js`), fed by a sample wind grid for now.
- **Advanced analysis**, opened from any place report: its own map with the smoke plume, an
  indicative fire-spread outlook for wildfire and crop burning, the last 8 observed footprints,
  distance rings, a terrain line with an elevation chart, a measuring tool, and a 72-hour
  forecast timeline you can scrub or play.
- **All 141 model features**, organised into the same 11 groups your model spec uses, each
  with plain-language reads, a searchable and exportable feature explorer, and an explicit
  note on which features also feed the training labels (so their contribution should be read
  with more care).
- **Exposure, similar events, and an analyst review tool** that saves confirm/correct/false-alarm
  calls on this device and exports them as CSV — the seed of a real labelled dataset.

**Everything is still generated sample data.** The "Sample data" badge stays in the header, and
the wind, plume and spread panels carry their own "Sample outlook" / "Indicative" labels so
nothing here is mistaken for a real forecast.

## Run it

Same as before — no build step:

1. Double-click `index.html`. It works fully offline (map tiles need internet; everything else
   doesn't).
2. Or serve the folder: `python3 -m http.server 8000`, then open http://localhost:8000.
3. `dist/agni-netra-single.html` is the whole thing, Leaflet and all, in one file you
   can email or open directly with no folder.

## What's new in the folder

    agni-netra-v2/
      index.html
      css/
        base.css                the version 1 styling (unchanged)
        analysis.css             the Leaflet map, wind layer, and advanced analysis workspace
      js/
        core.js                  NEW - small helpers moved out of ui.js (see "load order" below)
        data/
          geo.js                 India outline and state borders (unchanged from v1)
          sample.js               places, events, time series (this was v1's data.js)
          features.js             NEW - the 141-feature catalogue and per-event sample generator
          weather.js               NEW - wind, fire-weather indices, plume, spread, terrain
        map/
          wind-layer.js            NEW - the animated wind layer for Leaflet
          leaflet-map.js           NEW - the command map, replacing v1's drawn-SVG map
        analysis/
          analysis.js              NEW - the advanced analysis workspace shell, map and timeline
          tabs.js                  NEW - the 13 sections inside it (heat, timing, ... , review)
        ui.js                      trimmed: map code removed, review chip and glossary added
        pages.js                   place-report additions (Advanced analysis button, outlook card)
      vendor/
        echarts.min.js
        leaflet/                   Leaflet 1.9.4, its CSS, marker images, and its licence
      assets/geo/                  source boundary files (unchanged)
      docs/
        api-contract.md            NEW - exact JSON shapes for wind, plume, spread, the 141
                                    features, exposure, similar events, and analyst reviews
      tools/                       same three scripts as v1, now aware of the extra files

## Load order — please don't reorder the script tags

`index.html` loads scripts in an order that matters, because most of these files are plain
`<script>` tags (not modules) sharing one global scope:

1. `js/core.js` must load before `wind-layer.js`, `leaflet-map.js`, and `analysis.js` — they
   read `REDUCED`/`store`/`cv` as soon as they're parsed, not just inside functions.
2. `js/analysis/analysis.js` must load before `js/analysis/tabs.js` — `tabs.js` builds its
   `ATAB` lookup table from `tabWind`, which is defined in `analysis.js`.

Get the order wrong and the browser console will show `ReferenceError`s or
`Cannot access '...' before initialization` — that second one is the telltale sign of exactly
this kind of ordering mistake (a `const` whose initialiser threw, leaving it permanently
inaccessible for the rest of the page's life). If you introduce a bundler (Vite, esbuild) this
whole constraint disappears, since bundlers resolve dependencies rather than relying on
`<script>` order.

## Honesty rules carried through from version 1

- A class is never shown without its confidence.
- Wildfire and crop-burning spread is always labelled "Indicative" and capped at a rough
  ellipse from wind, fuel dryness and slope — never presented as a validated fire-front
  forecast.
- Every other source class shows "Not applicable" for spread, with an explanation, since fire
  spread modelling doesn't apply to a source that stays inside its site.
- The smoke plume says where smoke can travel, and is explicit that it says nothing about how
  thick the smoke is.
- Response/SOP guidance and the CAP alert draft are still marked "Proposed" and still don't
  send anything anywhere.
- Any feature that also feeds the training labels (facility distance, "known flare", land
  cover) is flagged in the explorer and in the group reads, so its contribution to a call is
  read with appropriate caution rather than taken at face value.

## Connecting the real backend

See `docs/api-contract.md` for the full set of endpoints (feature vectors, SHAP, wind grid,
plume, spread outlook, footprints, exposure, similar events, analyst reviews) with exact JSON
shapes matching what the frontend already expects. Version 1's README (if you kept a copy)
still covers the places/events/site-history endpoints, which are unchanged in version 2.

## Boundaries and credits

Same as version 1 — see the boundaries note there. Leaflet is BSD-2-Clause licensed; its
licence file is in `vendor/leaflet/`. Map tiles are (c) OpenStreetMap contributors, shown with
the required attribution in the map's bottom-right corner; swap `OSM_URL` in
`js/map/leaflet-map.js` for your own tile provider before any real deployment, per OSM's tile
usage policy.
