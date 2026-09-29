import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import Base, engine
from app.api.routes import router as api_router
from app.services.scheduler import start_scheduler, stop_scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize tables and background scheduler
    logger.info("Starting AGNI-NETRA backend service...")
    Base.metadata.create_all(bind=engine)
    start_scheduler()
    yield
    # Shutdown: Clean up background scheduler
    logger.info("Shutting down AGNI-NETRA backend service...")
    stop_scheduler()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AGNI-NETRA v2 Geospatial Earth Observation & Thermal Source Monitoring API",
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
        "service": "AGNI-NETRA API",
        "version": settings.VERSION,
        "docs_url": "/docs",
        "status": "operational"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    # Pinned to a single worker to prevent double-polling of NASA FIRMS
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, workers=1)
