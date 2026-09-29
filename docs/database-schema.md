# Database schema

The production data model for AGNI-NETRA. Nine tables, four schemas, five enums, four views.

```sql
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f backend/db/schema.sql
```

| Artefact | What it is |
|---|---|
| `backend/db/schema.sql` | The DDL. 981 lines. Applies clean to PostgreSQL 16 + PostGIS 3.4. |
| `backend/db/generate_schema.py` | Emits the 141 feature columns from the frontend catalogue, so they cannot drift. |
| `docs/database-design-review.md` | Findings against the original 5-table diagram, and why each was rejected. |

Everything below has been executed against a live database, not just reasoned about. The
verification log is at the end.

---

## 1. The shape of it

```
                       app.analyst
                            │ 1
                            │
                            ▼ n
   core.site ──────────► core.event ◄────────── ml.prediction ──► (models.json)
   (curated site)  1    n │      ▲  1                    ▲
                      n  │      │  n                    │ n
             core.detection   └────── ml.event_feature    │
             (one satellite     ml.event_geometry        │
              fire pixel)      (plume / spread /          │
                  │             footprint layers)         │
                  └── ops.job ◄─┘                          │
                      (ingest run + processing stage) ─────┘
                                                    app.analyst_review
```

| Schema | Holds | Why separate |
|---|---|---|
| `core` | Ingested and clustered observations | Ingested, not derived |
| `ml` | Model output | Recomputable from `core`; safe to truncate and rebuild |
| `app` | Identity | Reviewed by people, not by machines |
| `ops` | Execution state | Never joined to domain queries |
| `api` | Views only, no tables | The read models the endpoints serve |

That separation is a rule the project already committed to
(`docs/architecture-flow-diagram.md:105`): *"Raw, derived and model-output data live in different
tables and different layers."* Dropping `ml` and rebuilding it from `core` should never be a
dangerous operation, and this layout is what makes that true.

---

## 2. The domain, in one paragraph

Satellites report **detections** — individual fire pixels with a timestamp, a position, a satellite
and a fire-power reading. Detections near each other in space and time are a single **event** — a
burning fire. Some events happen at a known, curated **site** — a refinery, a mine, a gas flare — and
those sites accumulate history, which is what lets the system say "unusual *for here*". Each event is
described by **141 features**, scored into a **prediction**, and can be **reviewed** by an analyst,
and that review is the training label for the next model.

Four ideas carry the whole design:

1. **An event owns its location.** There is no separate place table to disagree with it.
2. **A detection belongs to an event, or does not yet.** Clustering is a state, not a different kind
   of thing.
3. **A prediction is versioned and separate from the event.** Models get replaced; their output has
   to survive that.
4. **A review is a label against a specific prediction.** Otherwise it quietly rots.

---

## 3. `core.site` — the curated asset

One row per monitored heat source with identity and history. 53 of these exist. This is the only
survivor from the original `Place` table.

| Column | Type | Why |
|---|---|---|
| `code` | `text UNIQUE` | `'GJ-01'` — stable against renames, and the frontend search matches on it (`selectors.ts:84`) |
| `kind` | `site_kind` | `refinery · power_plant · petrochemical · steel · cement · mine · landfill · gas_flare · other` |
| `expected_class` | `event_class` | What a flare *should* look like. A refinery scoring `wildfire` is a finding. |
| `location` | `geometry(Point,4326)` NOT NULL, GIST | `ST_DWithin` for "events near this site" |
| `state`, `district` | `text` | Indexed, not a FK — see §9 |
| `median_log_frp` | `numeric(10,6)` | The baseline |
| `mad_log_frp` | `numeric(10,6)` | Its spread |
| `baseline_sample_count` | `integer` | Makes `baseline_building` **derivable** |
| `baseline_window_days` | `integer` DEFAULT 365 | Which window the above describes |
| `is_active` | `boolean` | Retires a site without losing its history |

**Why the baseline lives here.** The z-score decides whether a human should look:

```
sd = 1.4826 * mad_log_frp
z  = (ln(frp) - ln(median_log_frp)) / sd
```

A gas flare at 40 MW is `routine`; at 300 MW it is `abnormal`. That judgement is impossible without
a persistent site row and its history, which is why `Place` could not simply be deleted. `sd` is not
stored — it is a constant times another column.

**Why `baseline_sample_count` is here.** `status = 'baseline_building'` used to be a stored string
meaning "not enough history yet". Now it is computed: fewer than N samples means the site is still
building its baseline. The string stops being a fact the database has to be told.

---

## 4. `core.event` — the anchor

One cluster of detections. **It owns the location.**

| Column | Type | Why |
|---|---|---|
| `ingest_key` | `text UNIQUE` | The idempotency key. See §5. |
| `site_id` | `bigint NULL` → `core.site` | Nullable: most events have no site |
| `merged_into_event_id` | `bigint NULL` → `core.event` | Re-clustering. "Current" = `IS NULL` |
| `centroid` | `geometry(Point,4326)` NOT NULL, GIST | Map viewport and `ST_DWithin`, single hop |
| `points` | `geometry(MultiPoint,4326)`, GIST | Render layer. Denormalised cache |
| `footprint` | `geometry(MultiPolygon,4326)`, GIST | Convex hull, for area and containment |
| `z_score` | `numeric(7,4) NULL` | See below |
| `frp_median_mw`, `frp_p90_mw`, `frp_max_mw`, `frp_sum_mw` | `numeric` | Aggregates of the member detections |
| `status` | `event_status` | 4 values: `abnormal · routine · baseline_building · not_applicable` |
| `local_pass_hour` | `smallint` CHECK 0–23 | Overpass time, in local terms |

### Why three geometry shapes

Different queries need different shapes, and picking one shape for all of them is the mistake:

- **`centroid`** is the selective one. A map viewport query filters thousands of events down to the
  ones on screen; a point is the cheapest thing to test. Verified: `Index Scan using event_centroid_gix`.
- **`points`** is what the browser actually draws. Keeping it on the event means the map render is
  one row read, not a join to `detection` per frame.
- **`footprint`** is the hull, for area and containment tests.

All three are caches or conveniences. `core.detection.geom` is the truth.

### Why `z_score` is nullable, and why that matters

`z = 0` is a real value: the event's fire power is exactly the site median — perfectly normal. If `0`
also meant "we have no baseline", the two most different situations in the system would render
identically, and both would render as *"nothing to see here"*. So the column is nullable and `NULL`
is the only representation of "not computable". This is the same rule as
`docs/architecture-flow-diagram.md:107` — *"Missing data returns `missing: true` / `null`, never
zero."*

The `CHECK` constraints on this table reject a negative FRP, a negative area, a pass hour of 25, and
an event that is its own merge target. All four verified.

---

## 5. `core.detection` — one satellite fire pixel

The raw input, and the answer to "where does the history live?"

| Column | Type | Why |
|---|---|---|
| `ingest_key` + `acquisition_time` | `UNIQUE` pair | The idempotency key |
| `event_id` | `bigint NULL` → `core.event` | **`NULL` until clustered** |
| `source`, `satellite`, `sensor` | `text NOT NULL` | Provenance per point |
| `location` | `geometry(Point,4326)` NOT NULL, GIST | The authoritative point |
| `frp_mw`, `brightness_temp`, `confidence`, `day_night`, `cloud_fraction` | `numeric` | The measurement |

**Why `event_id` is nullable.** Whether a detection has been clustered yet is a *state*, not a
different kind of thing. So clustering is an `UPDATE`:

```sql
UPDATE core.detection SET event_id = :new_event WHERE ingest_key = :k;
```

and re-clustering is fully expressible as data:

```sql
UPDATE core.event SET merged_into_event_id = :new_event WHERE id = :old_event;
INSERT INTO core.event (...) VALUES (...);          -- the replacement
UPDATE core.detection SET event_id = :new_event
 WHERE event_id = :old_event;                        -- re-point the members
```

The membership lives in a column rather than a junction table. One fewer table, one fewer join on
the hottest query, and no information lost.

**Why the key is `(ingest_key, acquisition_time)`.** `core.detection` is partitioned by month, and
PostgreSQL requires a partitioned table's unique constraints to include the partition key. This is
not a weakening: `acquisition_time` is a *component* of the hash that produces `ingest_key`, so the
same detection always carries the same timestamp, and the composite is exactly as strong as a global
unique on `ingest_key` alone.

**Why it matters.** Satellite feeds are at-least-once. Without this key, re-running ingest inserts a
fresh copy of everything, and because baselines are computed from the accumulated series, the
resulting corruption is **silent** — no error, plausible-looking numbers, every `abnormal` verdict
quietly untrustworthy, and nothing recoverable without re-deriving from raw.

### Partitioning

Monthly, with a `DEFAULT` partition so the table works before any partition exists.

```sql
SELECT create_detections_partition('2026-10');    -- 'YYYY-MM' or a date
```

Verified: a November query touches only `detections_2025_11`. Two operational notes that will bite
if ignored:

- Insert detections in **chronological order**. BRIN summarises per block range; if timestamps are
  scattered, every block spans the full range and the index excludes nothing. This was measured, not
  assumed.
- Once a month has rows in `detection_default`, creating that month as a real partition requires a
  full scan of the default. Detach the month first.
- Retention means detaching and archiving whole partitions, not `DELETE` row by row.

At 300k rows / 113 MB the planner still preferred a parallel sequential scan over BRIN. That is the
expected crossover — partitioning does the heavy lifting for time ranges, and BRIN is a secondary
optimisation inside each partition.

---

## 6. `ml.event_feature` and `ml.prediction`

### `ml.event_feature` — 141 columns, keyed `(event_id, feature_version)`

| Column | Type |
|---|---|
| `event_id`, `feature_version` | Primary key |
| `missing_fraction` | `numeric(5,4)` CHECK 0–1 — the API's response field |
| `FRP_MEDIAN` … `MISSING_FEATURE_FRACTION` | 141 × `double precision`, all nullable |
| `computed_at` | `timestamptz` |

All 141 are nullable and `NULL` means "not measured", never `0` — the same rule as `z_score`. A
column that is genuinely `0` and a column that was never measured must never look the same.

**Why keyed by version, 1:1 would be wrong.** A feature-schema bump *adds a row*. It must never
overwrite the vector a past prediction was made from, or that prediction becomes unexplainable.

**Why the columns are named `FRP_MEDIAN` and not `frpMedian`.** Those names are the wire contract —
`docs/frontend-backend-integration-analysis.md` specifies features are "keyed by all 141 feature
names". One spelling, one source of truth, no translation layer to drift.

**Where the order lives.** Not in the physical column order. It is frozen in code as
`ml/feature_schema.py::FEATURES`, an ordered tuple, and pinned by a contract test asserting it equals
both the model artefact's feature-name list and the frontend's `FDEF` catalogue. Columns may then be
reordered or added in SQL without touching the model. This is the discipline
`Project_Flow.pdf` §34 asks for, and the reason there is no `feature_schema` table: the catalogue is
already a code constant, and a second copy in the database would be two sources of truth.

Each of the 141 columns carries a `COMMENT` recording its family, unit, reference mean and spread,
bounds, and the frontend's flag semantics — `x` stored log1p, `k` **also feeds the training labels
and must be disclosed in the UI**, `b` binary with no percentile.

### `ml.prediction` — N per event, never columns on `event`

| Column | Why |
|---|---|
| `model_version`, `feature_version` | The audit rule (`architecture-flow-diagram.md:106`) |
| `predicted_class`, `confidence` | The output |
| `alt_class`, `alt_probability` | The runner-up. `CHECK` forces them to be set together |
| `class_probabilities`, `shap_values` | `jsonb`, GIN-indexed, for the "why" panel |
| `is_current` | Partial unique index — at most one current prediction per event |

The original diagram put `predictedClass` and `confidence` straight onto `Event`. That breaks the
project's own audit rule, and it can only ever hold one prediction, so retraining can never be
compared against the old model. Verified: a second `is_current` prediction is rejected, and
superseding works.

The model manifest is `artifacts/models.json` — `model_version`, `feature_version`, `artifact_uri`,
`sha256`, `trained_at`, `metrics`, `decision_threshold`. The model is a file; its metadata belongs
beside it, not split into SQL. Commit the manifest, ignore the `.bin`.

---

## 7. `app.analyst` and `app.analyst_review`

`reviewer text` with no user table was the original defect. You cannot have a reviewer without a
principal — a blocker the project had already logged at
`docs/frontend-backend-integration-analysis.md:200-201`.

| `analyst_review` column | Why |
|---|---|
| `prediction_id` NOT NULL | Anchors the label to a model state |
| `model_class_snapshot`, `model_confidence_snapshot` | Agreement is computable later without joining a since-superseded prediction |
| `request_id` + partial unique | A retried `POST` must not double-insert. Verified: `ON CONFLICT … DO NOTHING` leaves exactly one row |
| `superseded_at` | An edit supersedes, it does not overwrite. The review history survives |
| `verdict` + `analyst_class` | `CHECK` forces `analyst_class` to be present exactly when the verdict is `change` |

That last one is small and valuable: the most common review bug is a "changed to mining" verdict
submitted without the new class, which silently produces an unusable label.

`review_training_idx` — a partial index on live, label-bearing reviews — is the query that builds the
next training set.

---

## 8. `core.event_geometry` and `ops.job`

**`core.event_geometry`** — plume, spread, and footprint layers as PostGIS, not JSON, so
`ST_DWithin`, `ST_Distance` and `ST_Intersects` are ordinary indexed queries. `layer` is an enum-by-
`CHECK` (`plume_core · plume_outer · plume_trajectory · spread_ring · spread_uncertainty · footprint
· detection_hull · terrain_profile`) rather than a native enum, because these values come from the
geometry builders and a `CHECK` keeps `ALTER TYPE` out of feature work. Uniqueness is
`(event_id, layer, horizon_hours, schema_version)`.

**`ops.job`** — one row per `(source, window, stage)`, absorbing both the ingestion run and the
per-event processing job. `stage = 'ingest'` is the run row. This is what
`GET /api/metadata/sources` reads, and what keeps a long job off an API request path
(`architecture-flow-diagram.md:110`).

---

## 9. The four views

No storage. These are the read models, and `api.v_place` is the answer to "where did `Place` go?"

| View | Serves |
|---|---|
| `api.v_event` | Event + its current prediction, `merged_into_event_id IS NULL`, coordinates as `[lon, lat]` |
| `api.v_place` | The old `Place` payload, exactly |
| `api.v_place_history` | `observations[]` for `GET /api/places/{id}/history`, with `z_score` recomputed live |
| `api.v_source_freshness` | Source age and ingestion health, recomputed on read |

### `api.v_place` — why the frontend needs no change

```sql
SELECT ... FROM core.site WHERE is_active          -- curated sites
UNION ALL
SELECT ... FROM core.event WHERE site_id IS NULL   -- derived anchors, labelled at query time
```

The second branch is the old 280 synthetic `transient` Place rows, now **zero rows**. The label is
still produced — `COALESCE(nearest_site_label || ' near ' || district, district)` — and the nearest
site is found with a `ST_DWithin` radius of 0.15° (~16 km) so a Kerala fire is not labelled "near
Jamnagar refinery". Verified: an event 330 km from the nearest site correctly falls back to the
district name.

`GET /api/places` returns the same shape as before, so `docs/api-contract.md` and the frontend are
untouched. **280 rows that could drift out of sync with their events become zero.**

### `api.v_place_history` — why there is no observation table

Site history is a `GROUP BY` over `core.detection` joined to `core.event`, with `z_score` computed
from the site's baseline at read time:

```sql
(ln(sum(frp_mw)) - median_log_frp) / (1.4826 * mad_log_frp)
```

53 sites over a year is cheap, and the payoff is that **the baseline is always recomputable from raw
input**. A stored observation table would be a second copy of data already present, free to disagree
with the detections it was derived from. Verified end to end: an FRP of 35 MW against a site median
of 30 MW gives `z = 0.5199`, and `NULL` when the baseline is absent.

### Why there is no `admin_unit` table

`state` and `district` are indexed `text` on `event` and `site`. An `admin_unit` table with geometry
would buy referential integrity on a fixed list of ~36 states and ~780 districts, while the polygon
layer is already vendored in the frontend (`data/geo.ts:9`) and only used for drawing. Normalise if
districts are ever edited or aliased.

---

## 10. Constraints worth knowing about

30 `CHECK` constraints, 18 unique indexes, 11 foreign keys, 6 GIST indexes. The ones that carry
design decisions rather than catching typos:

| Constraint | Protects against |
|---|---|
| `event.centroid NOT NULL` | An event that cannot be placed on a map |
| `event.z_score` nullable, no default | Conflating "normal" with "unknown" |
| `event_no_self_merge` | An event that supersedes itself |
| `detection_ingest_uniq` | Silent double-counting on re-ingest |
| `detection.event_id` nullable | Forcing a fake cluster |
| `prediction_alt_paired` | A runner-up class with no probability |
| `prediction_current_uniq` (partial) | Two "current" predictions |
| `review_class_paired` | A `change` verdict with no new class — an unusable label |
| `review_request_uniq` (partial) | Double-inserted reviews on a retried request |
| `event_feature_missing_range` | A missing-fraction over 100% |
| `event_hour_range` | A satellite pass at hour 25 |

`ops.touch_updated_at()` is a `BEFORE UPDATE` trigger applied to every mutable table, so no row can
silently lose its audit trail. Prisma's `@updatedAt` is client-side only and would not do this.

---

## 11. What was deliberately left out

| Not stored | Why | Bring it back when |
|---|---|---|
| Land cover, terrain, facility polygons | The 46 features derived from them are persisted, so classification is reproducible. Only *drawing* them is lost. | A "show me why this is cropland" map |
| Feature catalogue as a table | It is already a code constant. A second copy is two sources of truth. | Never — use the contract test |
| Model registry table | The model is a file; its metadata belongs beside it. | A model-comparison dashboard |
| Embeddings | Similarity can scan `event_feature`; fine to ~10⁵ events. | The first slow `/similar` query |
| Event embeddings as a separate dimension key | One embedding size is enough. | A second embedding model |
| An `observation` table | Derived from `detection`, and would be a second copy free to disagree. | `/history` latency under load |
| Multi-tenancy | Single authority assumed. | More than one agency on one deployment |

The reference-geometry row is the one most likely to be regretted. It is safe for every endpoint
currently specified, because the place report's context cards read the stored features directly.

---

## 12. Open decisions

1. **Auth mechanism.** `app.analyst` is modelled; the identity provider is not. JWT, OIDC, or agency
   SSO — decide before `POST /api/events/{id}/reviews` exists.
2. **Multi-tenancy.** No tenant column. Wrong if several state agencies share this deployment, and a
   much later migration.
3. **Detection retention.** How many years, and what happens on partition detach.
4. **Feature storage shape.** 141 columns is comfortable against PostgreSQL's 1600 limit, but every
   recompute rewrites a wide row. A narrow `(event_id, feature_version, feature_index, value)`
   sibling becomes worth its write amplification at sustained volume — build it when writes hurt,
   not before.
5. **Baselines for site-less events.** Options: nearest site within a radius, a fixed spatial grid
   cell, or `not_applicable` — which is what the frontend does today (`places.ts:249`) and is
   defensible.

---

## 13. Verification log

Executed against `postgis/postgis:16-3.4` (PostgreSQL 16.4) on a throwaway container.

**Schema applies clean** — `psql -v ON_ERROR_STOP=1`, exit 0, from an empty database.

```
tables (excl. partition) = 9        views              = 4
enums                   = 5        feature columns    = 141
check constraints       = 30       unique indexes     = 18
fk constraints          = 11       declared GIST      = 6
column comments         = 148
```

**Constraints rejected as intended**

| Test | Result |
|---|---|
| Duplicate `detection.ingest_key` | rejected — `detection_default_ingest_key_acquisition_time_key` |
| Duplicate `event.ingest_key` | rejected — `event_ingest_key_key` |
| `event.centroid = NULL` | rejected — not-null violation |
| `detection.confidence = 1.4` | rejected — `detection_conf_range` |
| `event.local_pass_hour = 25` | rejected — `event_hour_range` |
| `event.merged_into_event_id = id` | rejected — `event_no_self_merge` |
| Two `is_current` predictions | rejected — `prediction_current_uniq` |
| `alt_class` without `alt_probability` | rejected — `prediction_alt_paired` |
| `verdict='change'` without `analyst_class` | rejected — `review_class_paired` |
| Duplicate review `request_id` | rejected; `ON CONFLICT … DO NOTHING` leaves exactly 1 row |

**Behaviour confirmed**

- `z_score` accepts both `0.0` and `NULL`, and they stay distinct.
- Detection inserts with `event_id IS NULL`, then clusters by `UPDATE` → 0 unclustered, 1 clustered.
- Superseding a prediction works; the partial unique index then allows the replacement.
- `api.v_event` returns the event joined to its current prediction, `[lon, lat]`.
- `api.v_place` returns both branches; a site-less event 330 km from the nearest site correctly falls
  back to the district label.
- `api.v_place_history` computes `z = 0.5199` for FRP 35 against a median of 30, and `NULL` with no
  baseline.
- `api.v_source_freshness` reports `ok` with a 9-minute age.

**Indexes used as intended**

```
ST_DWithin  →  Index Scan using event_centroid_gix
KNN (<->)   →  Index Scan using site_location_gix
Pruning     →  Seq Scan on detections_2025_11        (Oct and Dec partitions not scanned)
BRIN        →  Bitmap Index Scan on detection_default_acquisition_time_idx
              (usable, but a parallel seq scan still won at 300k rows / 113 MB)
```

**Bugs this process caught in the DDL itself** — each found by running it, not by reading it:

1. A comment placed after a column's trailing comma swallowed the comma, silently turning 141
   columns into a syntax error. Family labels now lead each group.
2. `api.v_place` contained a nonsense `ST_ClipByBox2D` call left over from an earlier draft.
3. `create_detections_partition('2026-10')` — the documented usage — failed, because `'2026-10'` is
   not a valid `date`. Now accepts `'YYYY-MM'`, a full date string, or a `date`.
4. The `api` schema was never created, so the views failed on a fresh database.
