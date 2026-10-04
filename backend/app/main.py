import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import Base, engine
from app.api.routes import router as api_router
from app.services.scheduler import start_scheduler, stop_scheduler
from sqlalchemy.exc import SQLAlchemyError

from app.services.model_registry import configure as configure_models
from app.services.model_registry import health as model_health

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: model registry first, then the database, then the poller.
    #
    # Model serving is independent of the database -- a booster and a parquet
    # file are all /api/predict needs. Previously a bad DATABASE_URL raised out
    # of create_all() and killed the process, so an unreachable Postgres took
    # inference down with it even though nothing about inference needed it.
    # The DB-backed endpoints report the failure instead.
    logger.info("Starting SPARC backend service...")

    configure_models(settings.MODEL_DIR)
    available = model_health()["models_found"]
    if available:
        logger.info("Models available: %s (default %s)",
                    available, settings.SERVING_MODEL_DEFAULT)
    else:
        logger.warning(
            "No model artefacts under %s -- /api/predict will return 503. "
            "Run colab_p7_train.py, and set MODEL_DIR if they live elsewhere.",
            settings.MODEL_DIR,
        )

    db_ok = _init_database()
    app.state.db_available = db_ok

    if db_ok:
        start_scheduler()
    else:
        # The FIRMS poller writes to the DB, so it cannot run without one.
        logger.warning("Scheduler not started: database unavailable")
    yield
    logger.info("Shutting down SPARC backend service...")
    if db_ok:
        stop_scheduler()


def _init_database() -> bool:
    """Create tables if possible. Returns False instead of raising.

    A development machine without Postgres should still serve predictions and
    return the sample-data endpoints; a missing database is a degraded mode, not
    a reason to refuse to boot.
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database ready")
        return True
    except SQLAlchemyError as exc:
        logger.error("Database unavailable, continuing in degraded mode: %s", exc)
        return False

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="SPARC Geospatial Earth Observation & Thermal Source Monitoring API",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routes
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {
        "service": "SPARC API",
        "version": settings.VERSION,
        "docs_url": "/docs",
        "status": "operational"
    }

@app.get("/health")
def health_check():
    """Liveness plus what is actually usable.

    "healthy" with `database: false` is honest and useful: inference is up, the
    DB-backed endpoints are not. A single "healthy"/"unhealthy" would hide which
    half is broken.
    """
    db_ok = getattr(app.state, "db_available", None)
    return {
        "status": "healthy",
        "database": db_ok,
        "models": model_health()["models_found"],
        "notes": (
            []
            if db_ok
            else ["database unavailable: DB-backed endpoints will error, "
                  "model serving is unaffected"]
        ),
    }

if __name__ == "__main__":
    import uvicorn
    # Pinned to a single worker to prevent double-polling of NASA FIRMS
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, workers=1)
