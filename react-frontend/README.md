# AGNI-NETRA — React + TypeScript

A React and TypeScript rewrite of the vanilla HTML/JS dashboard in [`../frontend`](../frontend).
The styling is unchanged: `src/styles/base.css` and `src/styles/analysis.css` are byte-for-byte
copies of `frontend/css/*.css`, and `#app`, `.rail`, `.main`, `#view` and every other class the
original markup used still drives the same rules.

The sample data layer is a faithful port, not a re-imagining. Places, events, the 141 model
features, weather, plumes, spread outlooks, explanations, CSV/GeoJSON/CAP exports and the shape
outlines all produce **exactly** the same numbers as the vanilla build (verified — see
[Verification](#verification)).

## Run it

```bash
npm install
npm run dev        # http://localhost:8000
npm run build      # type-check + production build into dist/
npm run preview    # serve the production build
```

## What changed, and what did not

| | |
| --- | --- |
| **CSS** | Copied verbatim. Nothing was added, removed or renamed. |
| **HTML** | Became JSX. The element tree, class names, `aria-*` attributes and DOM order of the command map overlays are preserved. |
| **JS** | Became TypeScript modules plus React components. Same algorithms, same seeded random, same numbers. |
| **`data-act` / `data-tip` attributes** | Gone. Actions are now real event handlers, and tooltips are a single shared element driven from `useTip()`. |
| **Load order** | Was a documented constraint in the vanilla build; with a bundler the import graph replaces it. |
| **`css` inline styles** | Only where a chart library needs them. Layout still comes from the stylesheet. |

One inherited quirk worth knowing about: on a wide screen the analysis toolbar (seven layer chips)
and the Flow/Arrows + Measure group both sit at the top of the analysis map, because the stylesheet
positions them and the media query only separates them below 1180 px. The vanilla build has exactly
the same overlap. Fixing it properly means changing a CSS rule, so it was left as it is.

Two deliberate departures from the original, both in markup or state handling, never in CSS:

- **The wind control is rendered after the legend** on the command map. The stylesheet stacks
  `.ov` panels in DOM order, and in the vanilla build the legend painted over the wind slider, so
  the slider could not be dragged. Reordering the JSX fixes it without touching a style rule.
- **The open analysis section is part of the URL** (`#/map/S001?w=90&a=2026-09-20&x=thermal`), which
  is what the original's `x=` hash parameter did. Everything else in that hash is unchanged too.

## The logo

`src/assets/` holds the mark, both as editable SVG sources and as ready-to-use PNGs.

| file | what it is |
| --- | --- |
| `logo.svg` | the mark, for light surfaces |
| `logo-on-dark.svg` | same mark with a lightened ring, for dark surfaces |
| `logo-mono.svg` | single colour, inherits `currentColor`, for any UI |
| `icon.svg` | optical small-size variant (heavier ring, narrower gap) for the favicon and app icon |
| `logo-{512,256,128,64,32}.png` | raster exports, exact pixel size, transparent background |
| `logo-on-dark-{512,256,128,64,32}.png` | same, dark colourway |
| `icon-{512,192,32}.png` | small-size variant rasters |

`public/favicon.svg`, `public/favicon-32.png` and `public/icon-192.png` are the same artwork
wired into `index.html`, and the rail in `components/Shell.tsx` imports the two colourways and
swaps them with the theme.

**The idea.** The name is *agni* (fire) and *netra* (eye), so the mark is both: an aperture ring
broken at the base — the eye that watches, and a lens aperture and a satellite's field of view —
holding a two-tongued flame. The flame's two tongues are what stop it reading as a water droplet
or a map pin, and the broken ring is what stops it reading as a plain circle. Colours come from the
app's own palette: the flame runs the signal red `--danger` `#E5484D` through amber `--warn`
`#E8A93B`, the ring is the cool blue-slate the dashboard already uses for water and coast lines.

**Why those proportions.** The ring is r=25 with a 44° gap and a 3.8 stroke, which keeps both
elements legible down to 16 px; the small-size variant thickens the stroke to 4.8 and narrows the
gap to 40° so the ring survives being rendered as a favicon. Both variants use the identical flame
path, so the identity never changes, only the optical weight.

### The model mark

The application mark above is an instrument watching for heat. The **model** is not an instrument, it
is the thing that reads the heat, so it needed a different idea: not a flame that was decorative,
but a form that says *this is a classifier*. It is a separate mark in `src/assets/model/`, and it
appears on the Model and data health page.

| file | what it is |
| --- | --- |
| `model-mark.svg` | the model mark, for light surfaces |
| `model-mark-on-dark.svg` | same, lifted, for dark surfaces |
| `model-mark-mono.svg` | single fixed ink, for one-colour print |
| `model-mark-icon.svg` | optical small-size variant, 5 bars |
| `model-mark-{512,256,128,64,32}.png` | raster exports, exact size, transparent |
| `model-mark-on-dark-{512,256,128,64,32}.png` | same, dark colourway |
| `model-mark-icon-{512,192,32}.png` | small-size variant rasters |

**The idea.** The flame is not drawn and then decorated — it is **drawn by its own bars**. Bar
lengths trace a flame profile, so the outer boundary of the bar field *is* the flame, and the top
bars step sideways to make the flick. The same nine bars are a histogram: long at the base,
decaying to nothing at the tip, which is what a distribution of detections looks like. One form,
two readings, no clip path.

**Why the bars are discrete, not a gradient.** Each bar is one flat colour off a nine-step
ember-to-gold ramp. A continuous gradient inside a 1.3px bar just averages to mud at 24px, which
is the size the mark actually runs at in the page head.

**Why the flick has to be there.** Without the sideways step at the top the mark collapses into a
striped funnel — it is a cone, not a fire. And an earlier version clipped horizontal bars to a
flame *outline*; that always lost the flick, because a bar grid can only trace a smooth edge.
Drawing the bars directly, rather than subtracting them from a shape, is what fixed it.

**Why the base is offset by one bar.** The second bar from the bottom is the widest, not the
first, so the base has a shoulder instead of reading as a slab that was cut off.

**Why there is no baseline axis or outline.** Both were tried. With centred bars a y-axis is
meaningless, and a stroke around the outside turned the mark into a sticker.

Source of truth for the geometry is `mark.js`, which writes all four SVGs from one set of
constants — change `PROFILE` or `FLICK` and re-run rather than editing the SVGs by hand.

## Where things went

```
src/
  main.tsx                     entry, stylesheets
  App.tsx                      providers + hash router
  lib/
    types.ts                   domain types (Place, Event, FeatureVector, Weather, …)
    core.ts                    clamp, colours, storage, seeded random, median/quant
    time.ts                    IST-aware date helpers, TODAY/START anchors
    files.ts                   clipboard and file saving
  data/
    classes.ts                 CLS, CLSMAP, STATUS, FAMILIES, isVeg
    geo.ts                     India outline and state borders (copied verbatim)
    places.ts                  sites, transient fires, events, series, stats, context
    features.ts                the 141 features, SHAP-style explanation, footprints
    weather.ts                 wind, fire weather indices, plume, spread, terrain
    analysis.ts                exposure, similar events, freshness, CAP draft, exports
  state/
    ThemeProvider.tsx          light/dark, persisted
    ReviewsProvider.tsx        analyst reviews and the watchlist, persisted
    ToastProvider.tsx          transient messages
    useFilters.ts              the shareable command-page filters, in the URL
    selectors.ts               passes / winEvents / groups / pickEvent
  components/
    Shell.tsx                  rail, header, search, routes
    ui/                        Icon, Glyph, Tooltip, Primitives, Chart
  pages/
    CommandPage.tsx            filters, KPIs, map, scrubber, side panel
    Filters.tsx Kpis.tsx Scrubber.tsx MapOverlays.tsx
    RegionsPage.tsx WatchlistPage.tsx ModelPage.tsx MethodsPage.tsx AlertsPage.tsx
  report/
    Queue.tsx PlaceReport.tsx PlaceCards.tsx AreaReport.tsx
  analysis/
    AdvancedAnalysis.tsx       the workspace shell, map and forecast timeline
    parts.tsx                  feature tiles, sections, reads, compass
    tabs/                      14 sections: wind, 8 feature groups, air, explorer,
                               exposure, similar events, analyst review
  assets/                     the app logo: SVG sources and PNG exports
  assets/model/               the model mark: SVG sources and PNG exports
  map/
    CommandMap.tsx AnalysisMap.tsx
    leafletBase.ts             tiles, offline base, hexagon binning, div icons
    WindLayer.ts               the canvas particle wind layer
```

Original → new:

| vanilla | React |
| --- | --- |
| `index.html` | `index.html` + `main.tsx` |
| `css/base.css`, `css/analysis.css` | `src/styles/*.css` (unchanged) |
| `js/core.js` | `src/lib/core.ts` |
| `js/data/geo.js` | `src/data/geo.ts` |
| `js/data/sample.js` | `src/data/classes.ts`, `src/data/places.ts` |
| `js/data/features.js` | `src/data/features.ts` |
| `js/data/weather.js` | `src/data/weather.ts` |
| `js/ui.js` | `src/state/*`, `src/components/Shell.tsx`, `src/components/ui/*` |
| `js/pages.js` | `src/pages/*`, `src/report/*` |
| `js/map/leaflet-map.js` | `src/map/CommandMap.tsx`, `src/map/leafletBase.ts` |
| `js/map/wind-layer.js` | `src/map/WindLayer.ts` |
| `js/analysis/analysis.js` | `src/analysis/AdvancedAnalysis.tsx`, `src/map/AnalysisMap.tsx` |
| `js/analysis/tabs.js` | `src/analysis/tabs/*` |
| `vendor/echarts.min.js`, `vendor/leaflet/*` | `echarts` and `leaflet` npm packages, tree-shaken |

`tools/`, `dist/agni-netra-single.html` and `assets/geo/*.json` have no equivalent: the first two
are build artefacts of the old no-build setup, and the JSON boundary files are the *source* of the
outline that is inlined in `data/geo.ts`.

## How the state is arranged

- **In the URL** — the page, the selected place, the date window, the class toggles, the status,
  sensor and confidence filters, and the open analysis section. These are the parameters the
  original wrote into the hash, so a link still carries the same view.
- **In context** — theme, analyst reviews, watchlist and toasts. All three persist to
  `localStorage` under the same keys as the original.
- **Local to the page** — layers, wind hour, drawing mode, the drawn area, report expand, pinned
  event, which card is expanded, the scrubber play state. None of it is shareable, so none of it
  lives in a global store.
- **Derived, never stored** — filtered events, the queue, the selected event for a report, the
  wind grid, the plume, the spread outlook. These are `useMemo` over the sample generators.

Leaflet and ECharts are imperative libraries, so they are mounted into a `ref`'d element and driven
from effects; both are torn down on unmount, and the map also watches its own container with a
`ResizeObserver` so expanding the report or resizing the window reflows it correctly.

## Verification

The data layer was diffed against the vanilla build by loading `frontend/js/*.js` in a sandbox and
comparing every generated value:

- 333 places, 727 events, 141 feature definitions
- every event field, per-place time series, cloud gaps, baselines, stats and hour histograms
- all 141 feature values, percentiles, angles, missing-value flags, per-group and per-feature
  contribution weights
- weather arrays, wind grids, plumes, spread horizons and rings, terrain profiles, footprints
- CAP XML, CSV, GeoJSON, the outlook sentence, similar-event lists, exposure counts, freshness rows

**Result: identical, to the last decimal place.** `capXml`, `eventsCsv` and `eventsGeo` match the
original output byte for byte.

The UI was driven in headless Chrome: every page, all 14 analysis sections, the filters, the
scrubber and its play mode, the map layers (density, raw detections, facilities, wind flow and
wind arrows), area drawing, report expansion, the before/after slider, the search box, the
keyboard shortcuts (`/`, `j`, `k`, `l`, `Escape`), the theme toggle, review saving and the URL
round-trip. No console errors or warnings.

## Honesty rules, carried over unchanged

- A class is never shown without its confidence.
- Wildfire and crop-burning spread stays labelled "Indicative"; every other class shows
  "Not applicable", with the reason.
- The smoke plume says where smoke can travel, not how thick it is.
- Response/SOP guidance and the CAP draft stay marked "Proposed" and send nothing.
- Features that also feed the training labels are flagged wherever they appear.
- Everything is still generated sample data, and the header says so.

## Connecting the backend

See [`../docs/api-contract.md`](../docs/api-contract.md) for the endpoint shapes. The seam is small
and deliberate: replace the bodies of `PLACES`/`EVENTS` in `data/places.ts`, `featuresOf()` in
`data/features.ts`, `weatherOf()` in `data/weather.ts` and the `similarOf`/`exposureOf` helpers in
`data/analysis.ts` with fetches, and nothing above the data layer changes.
