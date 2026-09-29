# Complete production directory structure

This is the target repository structure for the AGNI-NETRA application. The frontend
tree records the existing files and must not be moved, renamed, or deleted. The backend
tree is the production-ready structure to implement under `backend/`.

```text
konsors-sih2/
├── .gitignore
├── LICENSE
├── README.md
├── docs/
│   ├── backend-directory-structure.md
│   └── api-contract.md
│
├── frontend/                                      # Deployable static frontend
│   ├── index.html
│   ├── vercel.json
│   ├── README.md
│   ├── LICENSE
│   │
│   ├── css/
│   │   ├── base.css
│   │   └── analysis.css
│   │
│   ├── js/
│   │   ├── core.js
│   │   ├── ui.js
│   │   ├── pages.js
│   │   ├── analysis/
│   │   │   ├── analysis.js
│   │   │   └── tabs.js
│   │   ├── data/
│   │   │   ├── features.js
│   │   │   ├── geo.js
│   │   │   ├── sample.js
│   │   │   └── weather.js
│   │   └── map/
│   │       ├── leaflet-map.js
│   │       └── wind-layer.js
│   │
│   ├── assets/
│   │   └── geo/
│   │       ├── country.json
│   │       └── states.json
│   │
│   ├── vendor/
│   │   ├── echarts.min.js
│   │   ├── ECHARTS-LICENSE.txt
│   │   ├── ECHARTS-NOTICE.txt
│   │   └── leaflet/
│   │       ├── leaflet.js
│   │       ├── leaflet.css
│   │       ├── LEAFLET-LICENSE.txt
│   │       └── images/
│   │           ├── layers.png
│   │           ├── layers-2x.png
│   │           ├── marker-icon.png
│   │           ├── marker-icon-2x.png
│   │           └── marker-shadow.png
│   │
│   ├── tools/
│   │   ├── build_geo.py
│   │   ├── build_single_file.py
│   │   └── prepare_boundaries.py
│   └── dist/
│       └── agni-netra-single.html
│
└── backend/                                      # Python production services
    ├── README.md
    ├── Makefile
    ├── pyproject.toml
    ├── alembic.ini
    ├── .env.example
    ├── .dockerignore
    │
    ├── app/
    │   ├── __init__.py
    │   ├── main.py                               # FastAPI/ASGI entrypoint
    │   │
    │   ├── api/
    │   │   ├── __init__.py
    │   │   ├── router.py                          # Mounts all API versions
    │   │   └── v1/
    │   │       ├── __init__.py
    │   │       ├── router.py
    │   │       ├── dependencies.py                # Auth, DB, pagination dependencies
    │   │       └── endpoints/
    │   │           ├── __init__.py
    │   │           ├── health.py                  # Liveness and readiness checks
    │   │           ├── events.py                  # Event listing and details
    │   │           ├── predictions.py             # Prediction and SHAP responses
    │   │           ├── features.py                # Feature vectors
    │   │           ├── forecast.py                # Weather and wind grid
    │   │           ├── map.py                     # Plume, spread, footprints
    │   │           ├── exposure.py                 # Settlements, schools, roads
    │   │           ├── similar_events.py           # Similar-event search
    │   │           ├── reviews.py                  # Analyst review CRUD
    │   │           └── metadata.py                # Classes, regions, model metadata
    │   │
    │   ├── core/
    │   │   ├── __init__.py
    │   │   ├── config.py                          # Typed environment configuration
    │   │   ├── constants.py
    │   │   ├── logging.py                         # JSON logging configuration
    │   │   ├── security.py                        # JWT/API key and CORS settings
    │   │   ├── errors.py                          # Domain-to-HTTP error mapping
    │   │   ├── middleware.py                      # Request IDs and timing
    │   │   └── lifecycle.py                       # Startup and shutdown hooks
    │   │
    │   ├── shared/
    │   │   ├── __init__.py
    │   │   ├── db/
    │   │   │   ├── __init__.py
    │   │   │   ├── base.py                        # SQLAlchemy declarative base
    │   │   │   ├── session.py                     # Engine and session factory
    │   │   │   ├── transaction.py                 # Transaction helpers
    │   │   │   └── health.py                      # Database readiness check
    │   │   ├── observability/
    │   │   │   ├── __init__.py
    │   │   │   ├── request_context.py
    │   │   │   ├── metrics.py                     # Prometheus metrics
    │   │   │   ├── tracing.py                     # OpenTelemetry tracing
    │   │   │   └── health.py
    │   │   ├── storage/
    │   │   │   ├── __init__.py
    │   │   │   ├── object_store.py
    │   │   │   └── paths.py
    │   │   ├── pagination.py
    │   │   ├── time.py
    │   │   ├── exceptions.py
    │   │   └── types.py
    │   │
    │   ├── modules/                             # Business modules by capability
    │   │   ├── events/
    │   │   │   ├── __init__.py
    │   │   │   ├── domain/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── entities.py
    │   │   │   │   ├── value_objects.py
    │   │   │   │   ├── enums.py
    │   │   │   │   └── services.py
    │   │   │   ├── application/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── commands.py
    │   │   │   │   ├── queries.py
    │   │   │   │   └── ports.py
    │   │   │   ├── infrastructure/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── models.py
    │   │   │   │   ├── repository.py
    │   │   │   │   └── queries.py
    │   │   │   └── presentation/
    │   │   │       ├── __init__.py
    │   │   │       └── schemas.py
    │   │   │
    │   │   ├── features/
    │   │   │   ├── __init__.py
    │   │   │   ├── domain/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── feature_schema.py
    │   │   │   │   ├── feature_groups.py
    │   │   │   │   └── transformations.py
    │   │   │   ├── application/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── build_features.py
    │   │   │   │   └── ports.py
    │   │   │   ├── infrastructure/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── source_queries.py
    │   │   │   │   └── feature_repository.py
    │   │   │   └── presentation/
    │   │   │       ├── __init__.py
    │   │   │       └── schemas.py
    │   │   │
    │   │   ├── predictions/
    │   │   │   ├── __init__.py
    │   │   │   ├── domain/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── entities.py
    │   │   │   │   ├── value_objects.py
    │   │   │   │   └── services.py
    │   │   │   ├── application/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── predict_event.py
    │   │   │   │   ├── explain_prediction.py
    │   │   │   │   └── ports.py
    │   │   │   ├── infrastructure/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── models.py
    │   │   │   │   └── repository.py
    │   │   │   └── presentation/
    │   │   │       ├── __init__.py
    │   │   │       └── schemas.py
    │   │   │
    │   │   ├── models/
    │   │   │   ├── __init__.py
    │   │   │   ├── domain/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── entities.py
    │   │   │   │   └── value_objects.py
    │   │   │   ├── application/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── load_model.py
    │   │   │   │   ├── validate_model.py
    │   │   │   │   └── ports.py
    │   │   │   └── infrastructure/
    │   │   │       ├── __init__.py
    │   │   │       ├── lightgbm_predictor.py
    │   │   │       ├── artifact_store.py
    │   │   │       └── repository.py
    │   │   │
    │   │   ├── ingestion/
    │   │   │   ├── __init__.py
    │   │   │   ├── domain/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── source_records.py
    │   │   │   │   ├── policies.py
    │   │   │   │   └── enums.py
    │   │   │   ├── application/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── ingest_firms.py
    │   │   │   │   ├── ingest_hls.py
    │   │   │   │   ├── ingest_era5.py
    │   │   │   │   ├── ingest_land_cover.py
    │   │   │   │   ├── ingest_dem.py
    │   │   │   │   └── ports.py
    │   │   │   ├── infrastructure/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── firms_client.py
    │   │   │   │   ├── hls_client.py
    │   │   │   │   ├── era5_client.py
    │   │   │   │   ├── land_cover_client.py
    │   │   │   │   ├── dem_client.py
    │   │   │   │   ├── object_storage.py
    │   │   │   │   └── repositories.py
    │   │   │   └── presentation/
    │   │   │       ├── __init__.py
    │   │   │       └── schemas.py
    │   │   │
    │   │   ├── reviews/
    │   │   │   ├── __init__.py
    │   │   │   ├── domain/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── entities.py
    │   │   │   │   └── enums.py
    │   │   │   ├── application/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── create_review.py
    │   │   │   │   └── list_reviews.py
    │   │   │   ├── infrastructure/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── models.py
    │   │   │   │   └── repository.py
    │   │   │   └── presentation/
    │   │   │       ├── __init__.py
    │   │   │       └── schemas.py
    │   │   │
    │   │   └── jobs/
    │   │       ├── __init__.py
    │   │       ├── domain/
    │   │       │   ├── __init__.py
    │   │       │   ├── entities.py
    │   │       │   └── states.py
    │   │       ├── application/
    │   │       │   ├── __init__.py
    │   │       │   ├── claim_job.py
    │   │       │   ├── complete_job.py
    │   │       │   └── fail_job.py
    │   │       └── infrastructure/
    │   │           ├── __init__.py
    │   │           ├── models.py
    │   │           └── repository.py
    │   │
    │   ├── workers/
    │   │   ├── __init__.py
    │   │   ├── ingestion_worker.py
    │   │   ├── feature_worker.py
    │   │   ├── prediction_worker.py
    │   │   └── scheduler.py
    │   │
    │   └── ml/
    │       ├── __init__.py
    │       ├── training/
    │       │   ├── __init__.py
    │       │   ├── datasets.py
    │       │   ├── features.py
    │       │   ├── train.py
    │       │   ├── evaluate.py
    │       │   └── package_model.py
    │       └── evaluation/
    │           ├── __init__.py
    │           ├── metrics.py
    │           └── reports.py
    │
    ├── migrations/
    │   ├── env.py
    │   ├── script.py.mako
    │   └── versions/
    │       ├── 0001_create_schemas.py
    │       ├── 0002_create_raw_tables.py
    │       ├── 0003_create_reference_tables.py
    │       ├── 0004_create_event_tables.py
    │       ├── 0005_create_feature_tables.py
    │       ├── 0006_create_prediction_tables.py
    │       ├── 0007_create_review_tables.py
    │       └── 0008_create_monitoring_tables.py
    │
    ├── tests/
    │   ├── conftest.py
    │   ├── unit/
    │   │   ├── core/
    │   │   │   ├── test_config.py
    │   │   │   └── test_security.py
    │   │   ├── events/
    │   │   │   ├── test_entities.py
    │   │   │   └── test_services.py
    │   │   ├── features/
    │   │   │   ├── test_transformations.py
    │   │   │   └── test_feature_builder.py
    │   │   ├── predictions/
    │   │   │   ├── test_predict_event.py
    │   │   │   └── test_explanations.py
    │   │   └── ingestion/
    │   │       └── test_ingestion_policies.py
    │   ├── integration/
    │   │   ├── test_database.py
    │   │   ├── test_prediction_flow.py
    │   │   ├── test_ingestion_flow.py
    │   │   └── test_review_flow.py
    │   ├── contract/
    │   │   ├── test_api_contract.py
    │   │   └── fixtures/
    │   │       ├── event.json
    │   │       ├── prediction.json
    │   │       └── wind-grid.json
    │   └── fixtures/
    │       ├── factories.py
    │       └── sample_data.py
    │
    ├── deploy/
    │   ├── docker/
    │   │   ├── Dockerfile.api
    │   │   ├── Dockerfile.worker
    │   │   └── .dockerignore
    │   ├── docker-compose.yml
    │   └── kubernetes/
    │       ├── namespace.yaml
    │       ├── configmap.yaml
    │       ├── secret.example.yaml
    │       ├── api-deployment.yaml
    │       ├── worker-deployment.yaml
    │       ├── service.yaml
    │       ├── ingress.yaml
    │       ├── migration-job.yaml
    │       ├── horizontal-pod-autoscaler.yaml
    │       └── pod-disruption-budget.yaml
    │
    ├── scripts/
    │   ├── migrate.sh
    │   ├── seed_reference_data.sh
    │   ├── train_model.sh
    │   ├── verify_model.sh
    │   └── healthcheck.sh
    │
    └── data/                                      # Runtime only; never commit contents
        ├── raw/
        ├── processed/
        ├── cache/
        └── artifacts/
```

## Responsibility boundaries

- `api/` contains HTTP routing, dependency injection, validation, and response mapping.
  It must not contain database queries, vendor API calls, or ML implementation.
- `domain/` contains business rules and remains independent of FastAPI, SQLAlchemy,
  cloud SDKs, and external services.
- `application/` contains use cases and ports. `infrastructure/` implements database,
  storage, model, and external-provider adapters.
- `workers/` handles long-running ingestion, feature generation, predictions, and
  scheduled jobs. These operations must not block API requests.
- `migrations/` contains forward-only schema changes. Runtime data, secrets, caches,
  logs, virtual environments, dependencies, and model artifacts are not committed.
- The frontend remains a static deployment. Keep its current script load order and use
  `docs/api-contract.md` as the integration contract when replacing sample data.
