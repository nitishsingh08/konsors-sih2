import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "SPARC API"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api"
    DEBUG: bool = True
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://konsors-sih2.vercel.app",
        "*"
    ]
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./agninetra_local.db")
    
    # NASA FIRMS NRT
    FIRMS_MAP_KEY: str = os.getenv("FIRMS_MAP_KEY", "")
    POLL_INTERVAL_MINUTES: int = 15
    ADVISORY_LOCK_ID: int = 849201
    
    # Feature catalog versioning
    FEATURE_SCHEMA_VERSION: int = 1

    # Trained model artefacts written by colab_p7_train.py (qc/models/).
    # Empty disables model serving and the endpoints report the reason rather
    # than falling back to a fabricated score.
    MODEL_DIR: str = os.getenv("MODEL_DIR", "qc/models")
    SERVING_MODEL_DEFAULT: str = os.getenv("SERVING_MODEL_DEFAULT", "B")

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
