from datetime import datetime
from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field

# 6-Class Taxonomy
PredictedClass = Literal[
    "wildfire",
    "agricultural_burning",
    "gas_flare",
    "industrial",
    "mining",
    "unknown"
]

BaselineStatus = Literal["routine", "abnormal"]
AnalystStatus = Literal["unreviewed", "confirmed", "false_alarm"]
ReviewVerdict = Literal["confirm", "change", "false", "field"]

# --- Events ---
class EventBase(BaseModel):
    id: str
    place_id: Optional[str] = None
    detected_at: datetime
    lat: float
    lon: float
    frp_median: float = 0.0
    predicted_class: PredictedClass
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated against heuristic labeling rules")
    baseline_status: BaselineStatus = "routine"
    status: AnalystStatus = "unreviewed"
    missing_fraction: float = Field(default=0.0, ge=0.0, le=1.0)

class EventResponse(EventBase):
    class Config:
        from_attributes = True

# --- 141 Features (api-contract Section 1) ---
class FeatureValue(BaseModel):
    value: Optional[float] = None
    percentile: Optional[float] = None
    missing: bool = False

class EventFeaturesResponse(BaseModel):
    event_id: str
    features: Dict[str, FeatureValue]
    missing_fraction: float = 0.0

# --- SHAP Explanation (api-contract Section 2) ---
class TopFeatureContribution(BaseModel):
    key: str
    contribution: float

class ShapResponse(BaseModel):
    event_id: str
    by_family: Dict[str, float]
    top_features: List[TopFeatureContribution]

# --- Weather Forecast (api-contract Section 3) ---
class ForecastResponse(BaseModel):
    event_id: str
    issued_at: str
    hours: int = 73
    wind_speed_ms: List[float]
    wind_gust_ms: List[float]
    wind_from_deg: List[float]
    temperature_c: List[float]
    relative_humidity_pct: List[float]
    rain_mm: List[float]
    ffmc: List[float]
    dmc: float
    dc: float
    rain_past_24h_mm: float = 0.0

# --- Wind Grid (api-contract Section 4) ---
class WindGridResponse(BaseModel):
    lo0: float
    la1: float
    dx: float
    dy: float
    nx: int
    ny: int
    u: List[float]
    v: List[float]

# --- Smoke Plume (api-contract Section 5) ---
class PlumeResponse(BaseModel):
    event_id: str
    hours: int = 6
    reach_km: float
    bearing_deg: float
    trajectory: List[List[float]] # [[lon, lat], ...]
    core: List[List[float]]
    outer: List[List[float]]

# --- Spread Outlook (api-contract Section 6) ---
class SpreadHorizon(BaseModel):
    hours: int
    reach_km: float
    rate_kmh_lo: float
    rate_kmh_hi: float
    heading_deg: float
    area_km2: float
    ring: List[List[float]]

class SpreadResponse(BaseModel):
    event_id: str
    class_: str = Field(..., alias="class")
    horizons: List[SpreadHorizon]
    uncertainty_ring: List[List[float]]

# --- Exposure (api-contract Section 8) ---
class ExposureZone(BaseModel):
    name: str
    area_km2: float
    settlements: int = 0
    schools: int = 0
    clinics: int = 0
    road_km: float = 0.0
    water_bodies: int = 0

class ExposureResponse(BaseModel):
    zones: List[ExposureZone]

# --- Similar Events (api-contract Section 9) ---
class SimilarEventMatch(BaseModel):
    event_id: str
    similarity: float
    outcome: Literal["confirmed", "false_alarm", "unreviewed"]

class SimilarEventsResponse(BaseModel):
    matches: List[SimilarEventMatch]

# --- Analyst Reviews (api-contract Section 10) ---
class ReviewCreate(BaseModel):
    event_id: str
    verdict: ReviewVerdict
    analyst_class: Optional[str] = None
    note: Optional[str] = None
    reviewer: Optional[str] = None
    at: Optional[str] = None

class ReviewResponse(ReviewCreate):
    id: int
    created_at: datetime
    class Config:
        from_attributes = True
