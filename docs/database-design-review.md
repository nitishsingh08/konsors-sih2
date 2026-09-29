# Database design review

**Subject:** the submitted 5-table ER diagram (`Place`, `Event`, `EventFeatures`, `EventGeometry`,
`AnalystReview`).

**Question asked:** should the `Place` table exist?

**Short answer:** `Place` is the wrong fix for a real problem, and removing it alone is not
sufficient. The diagram's actual defect is that it models location as a *single point on a single
row* when the domain is a *cluster of points over time*. `Place` is one symptom of that. Fifteen
further issues sit behind it, two of which will corrupt data silently rather than fail loudly.

**Proposed replacement:** nine tables in four schemas, plus four views — down from the 22 the first
draft of this review proposed. §15 covers the design and §15.8 is explicit about what was cut, what
it costs, and what would bring each table back. No finding was dropped to get there.

**Status:** backend not implemented. `backend/` is an empty scaffold — no ORM, no migrations, no
config. There is no deployed schema to migrate, so everything below is still a design decision
rather than a change to live data. The frontend runs on generated sample data
(`react-frontend-test/src/data/`, `frontend/js/data/`) and calls no API at all.

---

## 1. Method

The diagram was compared against three sources, in decreasing order of authority:

1. **What the code actually reads.** `react-frontend-test/src/lib/types.ts` and
   `react-frontend-test/src/data/` are the canonical domain model; the vanilla build in `frontend/js/`
   is a port of it and agrees.
2. **Rules the project has already written down.** `docs/architecture-flow-diagram.md:105-111`
   ("Rules it encodes") and the "Honesty rules" in `frontend/README.md:84-98` and
   `react-frontend-test/README.md:224-232`.
3. **The prior DDL in `docs/Project_Flow.pdf`.** A 48-section design document containing real
   PostgreSQL/PostGIS DDL. It predates the shipped 6-class/141-feature model, so its *column set*
   is stale — but its *structural* decisions are sound and the diagram contradicts several of them.

Where the diagram contradicts a rule the project already committed to, that is called out as such
rather than presented as a fresh opinion.

One structural note about the PDF that is easy to miss: it contains **zero** `CREATE TABLE`
statements for places, reviews, analysts or users. It was never a place-or-review schema. So there
is no prior art in this repo to defer to on the two questions at the centre of this review.

---

## 2. F1 — `Place` conflates two incompatible entities

`PlaceKind` is a union of `'site' | 'transient'` (`types.ts:52`). Both variants are built into the
same `PLACES` array, but they are different kinds of thing:

| | `site` | `transient` |
|---|---|---|
| Built at | `places.ts:186-217` | `places.ts:220-256` |
| Count | 53 (from the `SITES` table, `places.ts:22-76`) | 280 (`transient()` called with 130 and 150, `places.ts:254-255`) |
| Has a stable external code | yes — `'GJ-01'`, `'OD-04'` (`places.ts:22-76`, assigned at `:193`) | no — the constant `'T'` (`places.ts:233`) |
| Has a computed baseline | yes — `med`/`mad` from its own history | no — hardcoded `med: 10, mad: 0.3` (`places.ts:244-245`) |
| Has a z-score | yes (`places.ts:152-155`) | never — hardcoded `null` |
| Has a status | `abnormal` / `routine` / `baseline_building` | always `not_applicable` (`places.ts:249`) |
| Direction of derivation | place → events | **event → place** |

That last row is the problem. A transient `Place` is *manufactured from an event*. Every field
duplicates the event that produced it:

```
id:        'T' + sequence                    places.ts:232   synthetic
code:      'T'                               places.ts:233   constant
name:      'Forest fire near ' + zone        places.ts:234   derived from event class
state:     zn[1]                             places.ts:235   copied off the zone tuple
district:  zn[0]                             places.ts:236   copied off the zone tuple
lat:       zn[2] + jitter(±0.18)             places.ts:228   derived
lon:       zn[3] + jitter(±0.18)             places.ts:229   derived
cls:       <the event's class>               places.ts:240   copied from event
type:      cls === 'wildfire' ? ... : ...    places.ts:241   a ternary of cls
firstSeen: t                                 places.ts:242   the event's own timestamp
med:       10                                places.ts:245   hardcoded
mad:       0.3                               places.ts:244   hardcoded
```

And `mkEvent` then explicitly declines to compute a z-score for them (`places.ts:150`,
`places.ts:155-157`) and hardcodes `status: 'not_applicable'` (`places.ts:249`).

**84% of `Place` rows (280 of 333) carry no information that is not already on the event.** In a
frontend this is harmless — the array is in memory and nobody normalises. In a database it is 280
rows that must be inserted, joined, kept consistent with the event, and given a foreign key, in
exchange for nothing.

Worse, the duplication is not even stable. A transient place's `lat`/`lon` is
`zone centre + jitter`, while its single event's own position is the place's position
(`ptsOf` jitters *from* `p.lon`/`p.lat`, `places.ts:262-270`) — so place location and event location
drift apart by construction.

**In a database, that grouping is a view, not a table.** The reason `Place` appears to be a parent of
`Event` in the frontend is a UI concern: the command map and place report list by place
(`EVP` indexed by place id, `places.ts:110`; watchlist is a set of place ids,
`ReviewsProvider.tsx:12`). Grouping a list is a query. It is not a reason to materialise a row per
group, per event, forever.

## 3. F2 — but the `site` half cannot be deleted

The tempting conclusion — "delete `Place`" — is wrong, and it would break the product.

The z-score is the mechanism that decides whether an event is worth a human's attention:

```
sd = 1.4826 * p.mad                                    places.ts:152
frp = exp(log(p.med) + zz * sd) * 1.1                  places.ts:154
z = (log(frp) - log(p.med)) / sd                      places.ts:155
```

`p.med` and `p.mad` are properties of a *location's own observation history*. Therefore `status` is
defined **relative to that location**, not absolutely: a gas flare at a known refinery is `routine`
at 40 MW and `abnormal` at 300 MW. The same FRP is a different verdict in a different place. A design
that only stores coordinates cannot make that judgement, because coordinates carry no history.

`baseline_building` makes the dependency explicit — it is assigned when
`tt - p.firstSeen < 20 * DAY` (`places.ts:213`, `places.ts:215`), i.e. *"this site has not been
observed long enough to have a baseline."* That sentence is unimplementable without a persistent
site row and its observation series.

The site row is also load-bearing for identity and retrieval:

- the watchlist is a set of place ids (`ReviewsProvider.tsx:12`), persisted under
  `localStorage['agni-watch']` (`ReviewsProvider.tsx:18-19`)
- search matches `name + district + state + code` and deliberately ranks sites above transient
  entries (`selectors.ts:84-85`)

**Conclusion: split the table. The curated half becomes `core.site`. The synthetic half is
deleted.** Section 14 covers how the frontend contract survives this.

## 4. F3 — triple location storage, with no rule about which wins

The diagram carries all three of these at once:

```
Place.lat / Place.lon
Event.lat / Event.lon
Event.placeId   (nullable)
```

Nothing states which is authoritative, and the design is self-contradictory: `Event.placeId` is
nullable *and* the event has its own coordinates, so a row may have a place, different coordinates
from that place, or no place at all. Every consumer must guess.

The code already resolved this deliberately. `Event` (`types.ts:134-154`) has **no `lat`/`lon`
fields at all**, and every geometry calculation anchors on the place:

```
localGrid(p)   →  { lo0: p.lon - 1.8, ... }     weather.ts:165-171
plumeOf(p)     →  const traj = [[p.lon, p.lat, 0, 0]]   weather.ts:176-180
footprint ring →  offsetLL(p.lon, p.lat, ...)  features.ts:516
GeoJSON export →  coordinates: [p.lon, p.lat]   data/analysis.ts:117
CAP alert      →  <circle>${p.lat},${p.lon}    data/analysis.ts:166
```

The diagram reintroduces exactly the ambiguity the frontend went out of its way to eliminate.
Whichever way it is resolved, two of the three must go.

## 5. F4 — the `Place` column set is wrong in both directions

Missing, though the code reads all of them: `kind`, `code`, `district`, `class`/`type`, `med`, `mad`,
`cover`, `firstSeen` (`types.ts:113-132`, in particular `:115-119`, `:120-121`, `:122-125`,
`:127`).

Present, with no counterpart anywhere in the domain: `category`. There is no such concept in
`types.ts`, in the `SITES` table, or in the API contract.

## 6. F5 — no idempotency key (the one true blocker)

Every primary key in the diagram is an opaque `String`. Satellite ingestion is at-least-once: FIRMS,
VIIRS and MODIS all re-publish overlapping windows, and any re-run of the pipeline re-delivers data.

Without a deterministic natural key, each run inserts a fresh set of events. The failure is silent —
nothing errors, the API returns 200, the map looks right. But because `status` and `z` are derived
from the accumulated observation series (§3), the corruption compounds: baselines are computed over
a history that now contains each detection twice or three times, medians and MAD drift, and every
`abnormal` verdict becomes unreliable. There is no point at which you can tell this happened without
re-deriving from raw.

The project's own prior DDL already had this right (`Project_Flow.pdf` §3):

```sql
CREATE TABLE raw.thermal_events (
  event_id BIGSERIAL PRIMARY KEY,
  source_event_id TEXT,
  source TEXT NOT NULL,
  acquisition_time TIMESTAMPTZ NOT NULL,
  location GEOGRAPHY(Point, 4326) NOT NULL,
  ...
);
CREATE INDEX idx_thermal_events_location ON raw.thermal_events USING GIST(location);
```

A surrogate key for joins, a natural key for dedupe, and a spatial column for queries. The diagram
has none of the three.

**Fix:** a materialised, unique `core.event.ingest_key` — a deterministic hash of
`(source, satellite, acquisition window, quantised centroid)` — with an `INSERT ... ON CONFLICT
(ingest_key) DO UPDATE`. Quantise the centroid to ~4 decimal places (≈11 m, well below the ~375 m
VIIRS pixel) so a float wobble between deliveries does not create a false distinct key.

## 7. F6 — no observation or detection history

The diagram stores `frpMedian` as a scalar. That number is an aggregate, and an aggregate is only
meaningful if the values it aggregates are kept. Nothing keeps them.

Downstream, this breaks:

- `GET /api/places/{id}/history`, which the contract defines as `observations[]` with
  `observed_at, frp_mw, detection_count, sensor, z_score`, plus `gaps[]`
  (`frontend-backend-integration-analysis.md:40`)
- `PlaceStats` (`types.ts:104-111`) — active days in 30/90, months active, consecutive-day streak,
  days since last
- **14 of the 141 features**, the entire `temporal` family
- the baseline itself, which cannot be recomputed or audited

The same applies to detections. The contract asks for "sensor, FRP, timestamp and confidence per
point" (`frontend-backend-integration-analysis.md:36`) and the frontend models them as
`Event._pts` (`types.ts:150`). With no table, `areaKm2` (`types.ts:147`), the plume anchor and the
density-hexagon layer are all unreproducible.

This is the second most serious issue after F5, and it is the one that makes the domain work
impossible rather than merely unreliable.

## 8. F7 — predictions violate a rule the project already wrote down

The diagram puts `predictedClass` and `confidence` directly on `Event`. Against
`docs/architecture-flow-diagram.md:106`:

> Every prediction stores the model version and the feature-set version, so it is auditable.

The diagram stores neither. Three consequences:

1. **No auditability.** A prediction cannot be tied to the model that made it, so you cannot answer
   "why was this classified as mining?" after a retrain, and you cannot reproduce a historical
   verdict.
2. **Only one prediction can ever exist.** Retraining cannot be compared against the old model,
   because there is nowhere to put the new result.
3. **The alternative class has no home.** `Event.cls2` / `Event.p2` (`types.ts:140-141`) — the
   runner-up class and its probability — are consumed by the UI and have no column.

`status` is also model output, not a property of the event. It derives from the z-score *and* the
prediction, so it belongs on the prediction, versioned with it.

Predictions must be a separate table: `(event_id, model_id, feature_version, predicted_at, class,
confidence, alt_class, alt_probability, class_probabilities, shap, threshold)`, N per event, with
`model_registry` to give the model an identity (`Project_Flow.pdf` §15, §16).

## 9. F8 — JSON blobs where the project specified typed columns

**`EventFeatures.featuresJson Json`.** The prior design specifies otherwise
(`Project_Flow.pdf` §11 defines `ml.feature_vectors` with ~50 typed `DOUBLE PRECISION` columns; §13
defines `ml.feature_schema` keyed `(feature_version, feature_index)`), and §34 is emphatic:

> **34. Very important: feature ordering** — "This is one of the easiest ways to accidentally break
> an ML system… maintain a fixed `feature_schema`… `FEATURE_VERSION = "v1"`"

A JSON blob defeats this: no column types, no per-feature nullability, no SQL access for drift
analysis ("which events are missing `ndvi`?", "has `frp_p90` drifted since the last retrain?"), and
no way to enforce that the vector was built in the declared order.

Its **1:1-with-event** shape is also wrong. A schema bump should *add a row*, not overwrite the
history. `ml.event_features` should be keyed `(event_id, feature_version)`.

**`EventGeometry.geometryJson Json`.** This contradicts the stated reason for using PostGIS
(`docs/architecture-flow-diagram.md:89`):

> typed rows into PostGIS as geometry, which is what makes `ST_DWithin` and `ST_Distance` ordinary
> queries

With a JSON blob, every proximity query is a sequential scan plus a JSON parse. This is the table
that most needs to be spatial. Use `geometry(Geometry,4326)` with a GIST index, and make `layerType`
an enum — a bare `String` is not queryable without a scan or a hard-coded literal list.

**`EventFeatures.embedding String?`** — a vector stored as text, for a shipped feature
("similar events"). Use `pgvector`, with the dimensionality keyed off an `embedding_model` row so
different embedding sizes can coexist.

**`featureSchemaVersion: Int`** conflicts with the PDF's `TEXT` semantic version. An integer
invites silent incompatibility: `3` means nothing, `3.2.0` is self-describing.

## 10. F9 — `reviewer: String?` with no users table

You cannot have a reviewer without a principal. This is not a new observation — it is the blocker
already logged in `docs/frontend-backend-integration-analysis.md:200-201`:

> Authentication/authorization is not defined. Reviews can be shared only after user identity,
> permissions and audit requirements are agreed.

`docs/api-contract.md:184` anticipated the field ("`reviewer`: optional, once you add auth"), but
`Review` (`types.ts:337-342`) has no identity field, `addReview` takes no identity
(`ReviewsProvider.tsx:37-43`), and the review CSV header has no reviewer column
(`ReviewsProvider.tsx:76`). Reviews currently live only in `localStorage['agni-reviews']`
(`ReviewsProvider.tsx:18`).

Compounding problems in the diagram's `AnalystReview`:

- `at: String?` — a timestamp as a nullable free-text string. The contract explicitly forbids
  display-formatted strings on the wire in favour of ISO-8601 with an offset
  (`frontend-backend-integration-analysis.md:136-137`), and the frontend's own current value is
  human-readable (`'20 Sep 2026, 14:30 IST'`). Use `timestamptz NOT NULL DEFAULT now()`.
- **No snapshot of what the model said.** The review is the label. If it is not bound to a specific
  prediction, the label silently rots when the model is retrained, and the training loop this whole
  feature exists to serve becomes unsound. Anchor to `prediction_id` and snapshot
  `model_class` / `model_confidence`.
- No uniqueness constraint, so a retried `POST` double-inserts. Add a client-supplied
  idempotency key.
- No `superseded_at`, so an edit destroys the earlier verdict instead of versioning it.
- No audit fields.
- `id: Int` while every other PK in the diagram is `String`.

## 11. F10 — nothing is enforced

Every column in the diagram is `String?`. Specifically missing:

- CHECK constraints for the 6 classes (`types.ts:10-16`), 4 statuses (`types.ts:18`), 4 verdicts
  (`types.ts:335`)
- `0 <= confidence <= 1` on `Event.confidence` and `Event.missingFraction` (currently plain `Float`,
  unconstrained)
- `NOT NULL` on anything, including the columns that must be present for a row to mean anything
- `created_at` / `updated_at` on every table — the diagram has no audit columns at all
- soft delete, so a curated site cannot be retired without losing its history
- a row version, for the optimistic review write the frontend is specified to perform
  (`frontend-backend-integration-analysis.md:128`)

Note also that `missingFraction: Float` (non-null) is lossy on its own: the contract requires
per-feature `missing: true` flags, and one scalar cannot express *which* features were missing.

And `Event.status` / `Event.baselineStatus` are near-synonymous free-text `String`s, where the
domain has exactly one field (`types.ts:18`) with four values. Splitting it in two loses
`not_applicable` — the exact status transient and vegetation-fire events need
(`places.ts:249`) — and stores as a string the one thing that is a computed, per-location quantity:
the z-score, which has no column anywhere in the diagram.

## 12. F11 — no reference or metadata tables

| Needed | For | Backing table today |
|---|---|---|
| something queryable for state/district | `Place.state` and `Place.district` are free text, so "all events in Karnataka" cannot be answered reliably | none |
| land cover | 16 `fuel` features | none |
| facility distances | 18 `industry` features | none |
| terrain | 12 `terrain` features | none |
| weather observations / forecasts | 18 `weather` features, `/api/events/{id}/forecast` | none |
| a model identity | F7 | none |
| a job/run record | `architecture-flow-diagram.md:110` | none |
| an ingestion-freshness record | `GET /api/metadata/sources` | none |

`Project_Flow.pdf` already defines `reference.industrial_facilities`, `reference.land_cover`,
`reference.terrain`, `raw.weather_observations`, `ml.model_registry` and
`monitoring.processing_jobs`. The diagram drops all of them.

**How the fix is scaled down.** The finding is that the diagram has *no way* to answer these
questions, not that every row in the PDF deserves a table. The proposed design answers six of these
eight with columns and one table instead: state/district become indexed `TEXT` (§15.7), the 46
land-cover/terrain/industry features are persisted in `ml.event_feature` and read back for the
context cards, the model identity is a `models.json` manifest (§15.1), and both the job record and
the freshness record are `ops.job` (§15.6). Only the *source* reference data is deliberately not
stored — see §15.8 for the cost and the trigger to re-add it.

## 13. F12–F14 — remaining

**No event lifecycle.** Fires are clustered, and re-clustered as new detections arrive — the frontend
grows `nDet` (`types.ts:146`) and re-derives area, plume and footprint. There is no representation of
cluster membership, supersession or merging, so "the current set of events" is undefined after the
second ingest pass. Needs either a membership table or a `supersedes`/`merged_into` chain.

**No partitioning or retention.** Satellite ingest is append-only and time-series-shaped. The largest
table (`detection`) needs monthly `PARTITION BY RANGE` plus BRIN on the time column, and a stated
retention policy. The diagram shows no awareness of table growth at all.

**Bidirectional back-references are not schema.** The diagram shows `Place.events`, `Event.features`,
`Event.geometries`, `Event.reviews`, `AnalystReview.event` and `Event.place` — six arrows for three
relationships. In the database only the FK direction exists. Worth stating explicitly so nobody
implements both directions and creates a circular dependency.

**Coordinate order is unstated.** The codebase is strict about this — `LonLat = [lon, lat]`
(`types.ts:7`), GeoJSON emits `[lon, lat]` (`data/analysis.ts:117`), CAP emits `lat,lon`
(`data/analysis.ts:166`), and `Project_Flow.pdf` warns `POINT(longitude latitude)`. A schema that
does not state the order is a schema that will eventually get it backwards.

## 14. Why the frontend still works after `Place` is removed

The concern with deleting `Place` is the API contract. It is not a real problem, because the
frontend's `Place` object is a *read model*, not an entity:

- Every field the UI reads is either a site attribute (from `core.site`) or an event aggregate
- `lat`/`lon` come from the event centroid
- `med`, `mad` map to `site.median_log_frp` / `site.mad_log_frp`; `cover` to `site.coverage`
- `state`, `district` are columns on the event
- `name` is `site.name` where a site exists, else a derived label
  (`'Forest fire near ' + <zone>` — the same string the frontend already builds at
  `places.ts:234`, but computed at query time instead of stored)

So `api.v_place` is a `UNION ALL` of sites and derived event anchors, reproducing the existing
`Place` payload exactly. The contract in `frontend-backend-integration-analysis.md:32-42` is
satisfied, `GET /api/places` still returns the same shape, and the frontend requires no change. The
integration doc already anticipates this (§"What the frontend actually needs": "a backend may combine
them into a single event/place-detail response for fewer round trips, but the adapter must still
expose the internal shapes used by the UI").

The win: 280 synthetic rows per snapshot become zero rows, and they can no longer drift out of sync
with their events.

## 15. Proposed design

**Nine tables, four schemas, four views.** An earlier draft of this review proposed 22 tables. That was
over-specified for a backend with no application code, and most of it is recoverable later. §15.8
records what was cut, what it costs, and what would bring each table back.

```
core    site · event · detection · event_geometry
ml      event_feature (141 cols) · prediction
app     analyst · analyst_review
ops     job

api     v_place · v_event · v_place_history · v_source_freshness        (views, no storage)
```

The layering rule from `architecture-flow-diagram.md:105` — *"Raw, derived and model-output data live
in different tables and different layers"* — is preserved: `core` holds ingested and clustered
observations, `ml` holds model output, `app` holds identity, `ops` holds execution state.

### 15.1 The three consolidations that do the most work

**`ingest.detection` + `core.event_detection` → one `core.detection`.** This is not only a saving, it
is a better model. A detection is one satellite fire pixel. Whether it has been clustered yet is a
*state*, not a different kind of thing:

```
event_id  BIGINT NULL → core.event     NULL until the clusterer runs
```

Clustering becomes an `UPDATE` instead of a join-table insert. Re-clustering — which the domain
genuinely needs, because fires grow and merge as detections arrive (F12) — is: mark the old event
`merged_into_event_id`, insert the new event, then `UPDATE detection SET event_id = …`. The
membership is still fully expressed as data; it just lives in a nullable column instead of a
junction table. One fewer table, one fewer schema, one fewer join on the map render.

**`feature_schema` table → a code constant.** The feature catalogue is *already* a hardcoded array
in the frontend (`FDEF` at `data/features.ts:186-188`, and `features.js:7-184` in the vanilla
build`). Putting a second copy in the database creates two sources of truth for the same artefact.
The discipline `Project_Flow.pdf` §34 is asking for — a frozen feature order — is better served by
one ordered tuple in code, `ml/feature_schema.py`:

```python
FEATURE_VERSION = "1.0.0"
FEATURES: tuple[FeatureDef, ...] = ( ... )   # ordered; index IS the model column index
```

with a contract test asserting `tuple(f.key for f in FEATURES)` equals the model artefact's
`feature_name` list, and that it equals the frontend's `FDEF` keys. The database stores only the
`feature_version` string. The mapping the old table was going to hold moves into the tuple:

| Frontend flag | Meaning | Code field |
|---|---|---|
| `x` | stored in log space, displayed real — `features.ts:341` does `Math.exp(raw) - 1` | `transform="log1p"` |
| `k` | also feeds the training labels, must be disclosed (README honesty rules) | `label_feeding=True` |
| `b` | yes/no, and no percentile — `features.ts:311`, `features.ts:348` | `binary=True` |
| `_SIN`/`_COS` | angle pair | `angle_pair=True` |

**`model_registry` → a manifest file beside the artefact.** The model is a file in
`backend/artifacts/`. Its metadata belongs next to it, not split into SQL. `models.json` carries
`model_version`, `feature_version`, `artifact_uri`, `sha256`, `trained_at`, `metrics`,
`decision_threshold`. `ml.prediction.model_version` is the reference. Commit the manifest to the repo
(so the audit trail is versioned) while keeping the `.bin` ignored.

### 15.2 `core.site` — the only former `Place` row that earns storage

```
id, code UNIQUE, name, kind, expected_class
geom         geometry(Point,4326) NOT NULL          GIST
state, district                                   TEXT — see §15.7
median_log_frp, mad_log_frp, baseline_sample_count,
baseline_window_days, baseline_computed_at         the baseline, folded in
coverage, is_active, first_seen_at, last_seen_at, created_at, updated_at
```

`median_log_frp` and `mad_log_frp` replace the old `site_baseline` table. One baseline window per
site, which is all the domain needs — the frontend's 7/30/90/365 filters are *display* windows
(`state/useFilters.ts:8`), not baseline windows. `baseline_sample_count` is what makes
`status = 'baseline_building'` **derivable** rather than a stored mystery string.

`sd = 1.4826 · mad_log_frp` is computed, not stored — it is a constant times another column.

### 15.3 `core.event` — the anchor

```
ingest_key    TEXT UNIQUE          the idempotency key (F5)
site_id       BIGINT NULL → core.site    nullable: most events have no site
centroid      geometry(Point,4326) NOT NULL     (F3: one authority, never two)
points        geometry(MultiPoint,4326)         render layer, denormalised, GIST
footprint     geometry(MultiPolygon,4326)       convex hull, for area and containment
state, district            TEXT        (see §15.7 — no admin_unit table)
observed_at, first_seen_at, last_seen_at, closed_at    TIMESTAMPTZ
frp_median_mw, frp_p90_mw, frp_max_mw, frp_sum_mw        NUMERIC
z_score       NUMERIC NULL        NULL when the baseline is insufficient — never 0
area_km2, detection_count, sensor, local_pass_hour
status        event_status        ENUM, 4 values
needs_review  BOOLEAN
merged_into_event_id  BIGINT NULL → core.event           (F12)
created_at, updated_at   TIMESTAMPTZ NOT NULL
```

`z_score` deserves a note. Per `architecture-flow-diagram.md:107` — *"Missing data returns
`missing: true` / `null`, never zero"* — and per `places.ts:155`, `z` is only computed when
`status !== 'baseline_building'`. A `z` of `0` is a real, meaningful value: it means the event's FRP
is exactly the site's median. Using `0` to mean "no baseline" would conflate "perfectly normal" with
"unknown" — the most dangerous possible collision in this schema, because both render as "nothing to
see here". So the column is nullable, and `NULL` is the only representation of "not computable".

### 15.4 Location: three shapes, because one event has many lat/lon

This is the part the submitted diagram gets most wrong, and it is worth stating separately. An event
is a *cluster* of detections (`places.ts:262-270` generates one point per detection, jittered around
the anchor), and different consumers need different shapes:

| Where | Type | Job |
|---|---|---|
| `event.centroid` | `geometry(Point,4326)` NOT NULL | map viewport + `ST_DWithin` filtering, single hop, GIST |
| `event.points` | `geometry(MultiPoint,4326)` | render layer; denormalised cache, GIST |
| `event.footprint` | `geometry(MultiPolygon,4326)` | convex hull, for area and containment |
| `detection.geom` | `geometry(Point,4326)` NOT NULL | the authoritative points, each with its own satellite, sensor, FRP, brightness temperature, acquisition time and confidence |

`detection.geom` is the answer to F6 — the per-point provenance the contract asks for
(`frontend-backend-integration-analysis.md:36`) — and the raw time series behind every baseline.

`event.points` is a deliberate denormalisation: the map render is the hottest query in the product,
and a join to `detection` per frame is not worth the normalisation. `detection` stays the source of
truth.

### 15.5 `ml.event_feature` — 141 nullable float columns

Keyed `(event_id, feature_version)`. All 141 columns nullable, and `NULL` means "not measured",
never `0`. The 11 families and their counts, verified against `data/features.ts:33-184`:

| # | Family | Count | Keys |
|---|---|---|---|
| 1 | thermal | 15 | `FRP_MEDIAN` … `SOURCE_AREA_MEDIAN` |
| 2 | temporal | 14 | `DURATION_HOURS_LOG` … `PEAK_HOUR_COS` |
| 3 | spread | 17 | `CENTROID_DISPLACEMENT_MEDIAN` … `SPREAD_WIND_SPEED_RATIO` |
| 4 | terrain | 12 | `ELEVATION_MEDIAN` … `RIDGE_VALLEY_POSITION` |
| 5 | fuel | 16 | `FOREST_FRAC_375M` … `NDMI_ANOMALY_Z` |
| 6 | industry | 18 | `DIST_NEAREST_FACILITY_LOG` … `KNOWN_FLARE` |
| 7 | weather | 18 | `TEMPERATURE_ANOMALY_Z` … `LIGHTNING_COUNT_24H` |
| 8 | smoke | 7 | `SMOKE_PROBABILITY` … `BURN_SCAR_GROWTH_RATE` |
| 9 | sar | 7 | `VV_CHANGE` … `STRUCTURAL_CHANGE_PROBABILITY` |
| 10 | chem | 9 | `NO2_ANOMALY_Z` … `CH4_CO_RATIO_LOG` |
| 11 | quality | 8 | `FIRMS_CONFIDENCE` … `MISSING_FEATURE_FRACTION` |
| | **total** | **141** | |

Plus `missing_fraction NUMERIC CHECK (0 <= missing_fraction AND missing_fraction <= 1)`.

Field naming note: the column names are the **exact API feature keys** (`FRP_MEDIAN`, not
`frpMedian`), because those keys *are* the wire contract — `frontend-backend-integration-analysis.md`
specifies features are "keyed by all 141 feature names". One spelling, one source of truth, no
translation layer to drift.

`MISSING_FEATURE_FRACTION` appears both as a `quality` feature and as a `missing_fraction` column.
That is not redundancy: the feature is a *model input*, the column is the *API response field*.

### 15.6 `ml.prediction`, `app.*`, `ops.job`

`ml.prediction` — N per event, never columns on `event` (F7):

```
id, event_id → core.event
model_version TEXT NOT NULL, feature_version TEXT NOT NULL    no registry FK; see §15.1
predicted_at, predicted_class, confidence, alt_class, alt_probability
class_probabilities JSONB, shap_values JSONB, decision_threshold
is_current BOOLEAN
```

`app.analyst` and `app.analyst_review` answer F9 — a reviewer cannot exist without a principal.
`analyst_review` anchors to `prediction_id` and snapshots `model_class` / `model_confidence`, so the
training label cannot rot when the model is retrained.

`ops.job` absorbs the old `ingestion_run` and `processing_job`. One row per
`(source, window, stage)`:

```
id, source, window_start, window_end, stage, status,
event_id NULL, detection_count, event_count, attempt, error_message,
started_at, completed_at, created_at
```

`stage = 'INGEST'` is the run row; other stages are per-event work. One table, both jobs. This is
what `GET /api/metadata/sources` and `v_source_freshness` read.

### 15.7 Why there is no `admin_unit` table

State and district are two indexed `TEXT` columns on `event` and `site`. An `admin_unit` table with
geometry was specified so "all events in Karnataka" would be reliable, but:

- the filtering need is served by a `TEXT` column and a B-tree index
- the polygon layer is **already vendored in the frontend** (`data/geo.ts:9`, `StateOutline[]` at
  `types.ts:172-175`) and is only used for map drawing
- India has a fixed ~36 states/UTs and ~780 districts, so a FK buys little integrity for a lookup
  that never changes

The cost is that district names are not referentially checked. That is the right trade at this size;
normalise if districts are ever edited or aliased.

### 15.8 What was cut, what it costs, and what brings it back

| Dropped | What it costs | Trigger to re-add |
|---|---|---|
| `land_cover`, `terrain`, `facility` | Cannot *draw* the polygons — no "show me why this is cropland" map. Classification stays fully reproducible, because the 46 fuel/terrain/industry features are persisted. | A UI that visualises the underlying layers |
| `admin_unit` | District names not referentially checked | Districts being edited or aliased |
| `observation` | Site history and the display windows become a `GROUP BY` over `detection` rather than a read of a purpose-built table. 53 sites × 1 year is cheap. | Latency on `/history` as event volume grows |
| `site_baseline` | One baseline window per site, not one per window. Adequate — display windows ≠ baseline windows. | Needing a 30-day baseline beside a 365-day one |
| `feature_schema` | Drift risk between `FEATURES` in Python and `FDEF` in TypeScript | Mitigate with a contract test; do **not** re-add the table |
| `model_registry` | No metric history in SQL; model comparison needs the manifests diffed | A model-comparison dashboard |
| `event_embedding`, `embedding_model` | Similar-events search must scan `event_feature` rows. Fine to ~10⁵ events. | The first slow `/similar` query — then add pgvector |
| `ingestion_run` | No per-run counts as separate columns (they sit on the job row) | Per-run throughput reporting |

The reference-geometry cut is the one most likely to be regretted, because "Other heat nearby" and
the place report's context cards are shipped UI. Checked: they still work. `PlaceContext`
(`types.ts:60-77`) is land-cover fractions, facility distances, NDVI/NBR/NDBI, clear-pixel fraction,
VPD, rain, elevation and slope — every one of which is already a stored column in `event_feature`.
The cards read the site's most recent feature row. "Other heat nearby" is `ST_DWithin` over `event`.

## 16. Reference Prisma schema

**Not yet written to a file.** A Prisma schema for the earlier 22-table draft exists only as a chat
artefact and is now superseded by §15. It should be written fresh as `backend/db/schema.prisma`
alongside `backend/db/schema.sql`, covering the reduced design:

```
core    Site · Event · Detection (partitioned) · EventGeometry
ml      EventFeature (141 cols) · Prediction
app     Analyst · AnalystReview
ops     Job
```

Five enums: `EventClass` (6 values, `types.ts:10-16`) · `EventStatus` (4, `types.ts:18`) ·
`ReviewVerdict` (4, `types.ts:335`) · `SiteKind` · `JobStatus`.

`site_kind` is a simplification of the earlier 11-value enum: with the `facility` table dropped, the
distinction that survives is the one the 53 curated sites are actually sorted by in the UI
(`selectors.ts:85` ranks sites above everything else) plus the class each is expected to produce.
`refinery · power_plant · petrochemical · steel · cement · mine · landfill · gas_flare · other` is
enough; the eight `industry` distance features are computed from the external GEM/OSM extract at
feature-build time, not from a local `facility_kind` column.

Two deliberate choices worth recording:

- PostGIS columns are `Unsupported("geometry(Point,4326)")` — Prisma has no native geometry type and
  cannot query these through the client. All spatial access is raw SQL.
- Feature fields are named in `UPPER_SNAKE_CASE` to match the API keys. Prisma permits uppercase
  field names; if a toolchain rejects them, the mechanical fix is camelCase plus
  `@map("UPPER_SNAKE_CASE")` on each of the 141.

### What Prisma cannot express

A Prisma-only schema is not a production schema. These all need raw SQL:

| Need | Why Prisma cannot do it |
|---|---|
| GIST indexes on the 6 geometry columns | `@@index` cannot specify `USING GIST` |
| CHECK constraints (0..1, 0..23) | no syntax |
| `PARTITION BY RANGE` on `core.detection` | no syntax |
| BRIN indexes on append-only time columns | no syntax |
| Expression/partial unique indexes | no syntax — which is why `ingest_key` is a materialised column rather than an index trick |
| `updated_at` and retention triggers | `@updatedAt` is client-side only; the database does not enforce it |
| `api.*` views | needs the `views` preview feature |

The expression-index limitation is the one that actually improves the design: because Prisma cannot
express a unique index on `(source, satellite, acquisition_time, round(lon,3), round(lat,3))`,
`ingest_key` has to be a real, materialised, unique column. That is better anyway — it is readable
in `psql`, and you can query "which event does this raw detection belong to?" without recomputing a
hash.

## 17. Invariants to enforce

Each traces back to a rule the project already committed to.

| Invariant | Source |
|---|---|
| `detection.ingest_key` unique; ingestion upserts, never inserts | F5 |
| `event.ingest_key` unique; clustering upserts, never inserts | F5 |
| `event.centroid NOT NULL`; `event.site_id` nullable; one location authority | F3 |
| `detection.geom NOT NULL`, `POINT(longitude latitude)` | `Project_Flow.pdf` |
| Every enum field is an ENUM, not a `String` | F10 |
| `0 <= confidence <= 1`, `0 <= missing_fraction <= 1`, `0 <= local_pass_hour <= 23` | F10, `:107` |
| `z_score IS NULL` means "no baseline" — never `0` | `:107` |
| `status = 'baseline_building'` ⟺ `baseline_sample_count` below threshold — derived, not stored | F2 |
| Every prediction carries `model_version` + `feature_version` | `:106` |
| Every review anchors to a `prediction_id` and snapshots the model output | F9 |
| All geometry is PostGIS with a GIST index | `:89` |
| All timestamps are `TIMESTAMPTZ` | `:136-137` |
| Every table has `created_at` / `updated_at` | F10 |
| `FEATURES` order in code is frozen; a contract test pins it to the model artefact and to `FDEF` | `Project_Flow.pdf` §34 |
| No leakage: features use only data strictly before event time | `Project_Flow.pdf` §26 |

## 18. Open decisions

These are not assumptions to be made silently.

1. **Auth mechanism.** `app.analyst` is modelled; the identity provider is not. JWT, OIDC, or an
   agency SSO — this needs deciding before `POST /api/events/{id}/reviews` exists, per
   `frontend-backend-integration-analysis.md:200-201`.
2. **Multi-tenancy.** The schema assumes a single authority and has no tenant column. Wrong if
   several state agencies share this deployment, and it is a much later migration to add.
3. **Raw detection retention.** `core.detection` is the largest table. How many years, and what
   happens on partition detach?
4. **How far to lean on the wide `event_feature` table.** 141 columns is near PostgreSQL's 1600
   limit, so it is comfortable, but every feature recompute rewrites a wide row. At sustained
   high event rates, a narrow `(event_id, feature_version, feature_index, value)` sibling becomes
   worth its write amplification. Do not build it speculatively — build it when write volume hurts.
5. **Where baselines live for site-less events.** `core.site` is right for curated assets, but
   transient events still need *some* comparison basis. Options: nearest site within a radius, a
   fixed spatial grid cell, or `not_applicable` (what the frontend does today, `places.ts:249`).
   The frontend's current answer is the last one, and it is defensible.
6. **Whether the deferred tables in §15.8 are actually deferred.** The reference-geometry cut is the
   one to watch. It is safe for every currently-specified endpoint, and it becomes wrong the moment
   someone wants to draw a land-cover or facility layer.
   amplification across 141 rows per event. Only build it if a query actually needs it.
5. **Where baselines live for site-less events.** `reference.site` is right for curated assets, but
   transient events still need *some* comparison basis. Options: nearest site within a radius, a
   fixed spatial grid cell, or `not_applicable` (what the frontend does today,
   `places.ts:249`). The frontend's current answer is the last one, and it is defensible.

---

## Appendix — findings index

| # | Finding | Evidence |
|---|---|---|
| F1 | `Place` conflates curated sites with 280 synthetic event-derived rows | `types.ts:52`; `places.ts:22-76`, `:186-217`, `:220-256`, `:231-246`, `:254-255`, `:244-245`, `:249` |
| F2 | The site half is irreducible: status and z-score are location-relative | `places.ts:152-155`, `:213`, `:215`; `ReviewsProvider.tsx:12`; `selectors.ts:84-85` |
| F3 | Triple location storage with no authority rule; code has exactly one | `types.ts:120-121`, `:134-154`; `weather.ts:165-171`, `:176-180`; `features.ts:516`; `data/analysis.ts:117`, `:166` |
| F4 | `Place` columns wrong in both directions (`category` invented, 8 fields missing) | `types.ts:113-132` |
| F5 | No idempotency key; re-ingest silently corrupts every baseline | diagram PKs; `Project_Flow.pdf` §3 |
| F6 | No observation or detection table; history and 14 temporal features unreproducible | `frontend-backend-integration-analysis.md:36`, `:40`; `types.ts:104-111`, `:147`, `:150` |
| F7 | Predictions lack model/feature version; only one can ever exist | `architecture-flow-diagram.md:106`; `types.ts:140-141`; `Project_Flow.pdf` §15, §16 |
| F8 | JSON blobs for features and geometry; embedding as text; `Int` schema version | `Project_Flow.pdf` §11, §13, §34; `architecture-flow-diagram.md:89`; `features.ts:311`, `:341`, `:348` |
| F9 | `reviewer` with no users table; `at` as a nullable String; no prediction anchor | `frontend-backend-integration-analysis.md:200-201`; `api-contract.md:184`; `types.ts:337-342`; `ReviewsProvider.tsx:18-19`, `:37-43`, `:76` |
| F10 | No CHECKs, no NOT NULLs, no audit columns, no soft delete, no row version | `types.ts:10-16`, `:18`, `:335`; `architecture-flow-diagram.md:107` |
| F11 | No reference or metadata tables at all (fix is scaled down — see §12) | `Project_Flow.pdf` §5, §7, §9, §10, §16, §34 |
| F12 | No event lifecycle or re-clustering representation | `types.ts:146`; `places.ts:262-270` |
| F13 | No partitioning or retention | — |
| F14 | Bidirectional back-references are ORM artifacts, not schema | diagram |
| F15 | Coordinate order unstated | `types.ts:7`; `data/analysis.ts:117`, `:166`; `Project_Flow.pdf` |
| F16 | Over-specified: 22 tables proposed for a backend with no application code | §15, §15.8 |

F1–F10 are defects in the submitted diagram. F11–F15 are defects it inherits by omission. F16 is a
defect in this review's own first draft, caught on review.

---

## Appendix 2 — what changed between drafts

| | Earlier draft | This version |
|---|---|---|
| Tables | 22 | **9** |
| Schemas | 6 | **4** |
| Reference geometry stored | yes | no (§15.8) |
| Feature catalogue | a table | a code constant (§15.1) |
| Model metadata | a table | a JSON manifest (§15.1) |
| Cluster membership | a junction table | a nullable FK (§15.1) |
| Site history | a table | derived from `detection` (§15.8) |
| Embeddings | two tables | none (§15.8) |

No finding was dropped to achieve this. F1–F16 all remain open against the submitted diagram, and
each fix still holds — `detection` is still a real table (F6), `prediction` is still separate from
`event` (F7), `analyst` still exists (F9), geometry is still PostGIS (F8), and `ingest_key` is still
unique (F5). What changed is how much of the *surrounding* apparatus each fix needs.