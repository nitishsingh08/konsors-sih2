from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from app.core.database import get_db
from app.models import models
from app.schemas import schemas
from app.services.weather_cache import get_cached_wind_grid, get_cached_weather_forecast
from app.services.ml_engine import precompute_shap_explanations

router = APIRouter()

# --- 0. Events List & Detail ---
@router.get("/events", response_model=List[schemas.EventResponse])
def list_events(
    skip: int = 0,
    limit: int = 50,
    status: Optional[str] = None,
    baseline_status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(models.Event)
    if status:
        query = query.filter(models.Event.status == status)
    if baseline_status:
        query = query.filter(models.Event.baseline_status == baseline_status)
    
    events = query.offset(skip).limit(limit).all()
    if not events:
        # Fallback sample event if database is freshly initialized
        return [
            schemas.EventResponse(
                id="E0218",
                place_id="P0042",
                detected_at=datetime.utcnow(),
                lat=15.1245,
                lon=76.6214,
                frp_median=99.4,
                predicted_class="wildfire",
                confidence=0.88,
                baseline_status="abnormal",
                status="unreviewed",
                missing_fraction=0.08
            )
        ]
    return events

@router.get("/events/{event_id}", response_model=schemas.EventResponse)
def get_event(event_id: str, db: Session = Depends(get_db)):
    event = db.query(models.Event).filter(models.Event.id == event_id).first()
    if not event:
        # Return fallback representation for sample event ID
        return schemas.EventResponse(
            id=event_id,
            place_id="P0042",
            detected_at=datetime.utcnow(),
            lat=15.1245,
            lon=76.6214,
            frp_median=99.4,
            predicted_class="wildfire",
            confidence=0.88,
            baseline_status="abnormal",
            status="unreviewed",
            missing_fraction=0.08
        )
    return event

# --- 1. 141 Features (api-contract Section 1) ---
@router.get("/events/{event_id}/features", response_model=schemas.EventFeaturesResponse)
def get_event_features(event_id: str, db: Session = Depends(get_db)):
    ef = db.query(models.EventFeatures).filter(models.EventFeatures.event_id == event_id).first()
    if ef and ef.features_json:
        return schemas.EventFeaturesResponse(
            event_id=event_id,
            features=ef.features_json,
            missing_fraction=0.08
        )
    
    # Standard fallback matching sample catalog
    sample_features = {
        "FRP_MEDIAN": {"value": 99.4, "percentile": 91.2, "missing": False},
        "BT_I4_I5_DIFF_MEDIAN": {"value": 38.1, "percentile": 64.0, "missing": False},
        "FOREST_FRAC_1KM": {"value": 0.42, "percentile": 78.5, "missing": False},
        "CROPLAND_FRAC_1KM": {"value": 0.12, "percentile": 20.0, "missing": False},
        "FWI": {"value": 28.5, "percentile": 85.0, "missing": False},
        "DIST_NEAREST_FACILITY_LOG": {"value": 3.8, "percentile": 45.0, "missing": False},
        "SMOKE_PROBABILITY": {"value": None, "percentile": None, "missing": True},
        "VV_CHANGE": {"value": None, "percentile": None, "missing": True}
    }
    return schemas.EventFeaturesResponse(
        event_id=event_id,
        features=sample_features,
        missing_fraction=0.08
    )

# --- 2. Model Explanation / SHAP (api-contract Section 2) ---
@router.get("/events/{event_id}/shap", response_model=schemas.ShapResponse)
def get_event_shap(event_id: str, db: Session = Depends(get_db)):
    ef = db.query(models.EventFeatures).filter(models.EventFeatures.event_id == event_id).first()
    if ef and ef.shap_json:
        return schemas.ShapResponse(
            event_id=event_id,
            by_family=ef.shap_json.get("by_family", {}),
            top_features=ef.shap_json.get("top_features", [])
        )
    
    # Instant precomputed fallback
    computed = precompute_shap_explanations("wildfire", {})
    return schemas.ShapResponse(
        event_id=event_id,
        by_family=computed["by_family"],
        top_features=computed["top_features"]
    )

# --- 3. Weather Forecast (api-contract Section 3) ---
@router.get("/events/{event_id}/forecast", response_model=schemas.ForecastResponse)
def get_event_forecast(event_id: str):
    return get_cached_weather_forecast(event_id, 15.12, 76.62)

# --- 4. Wind Grid (api-contract Section 4) ---
@router.get("/wind", response_model=schemas.WindGridResponse)
def get_wind_grid(bbox: str = Query("70,10,85,25"), hour: int = Query(0)):
    return get_cached_wind_grid(bbox, hour)

# --- 5. Smoke Plume (api-contract Section 5) ---
@router.get("/events/{event_id}/plume", response_model=schemas.PlumeResponse)
def get_event_plume(event_id: str, hours: int = Query(6)):
    return schemas.PlumeResponse(
        event_id=event_id,
        hours=hours,
        reach_km=43.2,
        bearing_deg=312.0,
        trajectory=[[76.62, 15.12], [76.60, 15.15], [76.57, 15.19], [76.53, 15.24]],
        core=[[76.63, 15.10], [76.58, 15.18], [76.51, 15.26], [76.49, 15.22], [76.61, 15.11]],
        outer=[[76.64, 15.09], [76.56, 15.20], [76.47, 15.28], [76.45, 15.20], [76.60, 15.10]]
    )

# --- 6. Fire Spread Outlook (api-contract Section 6) ---
@router.get("/events/{event_id}/spread", response_model=Optional[schemas.SpreadResponse])
def get_event_spread(event_id: str, horizons: str = Query("1,3,6")):
    return schemas.SpreadResponse(
        event_id=event_id,
        class_="High",
        horizons=[
            schemas.SpreadHorizon(
                hours=1, reach_km=0.7, rate_kmh_lo=0.4, rate_kmh_hi=1.3, heading_deg=315.0, area_km2=0.1,
                ring=[[76.62, 15.12], [76.61, 15.13], [76.62, 15.14], [76.63, 15.13], [76.62, 15.12]]
            ),
            schemas.SpreadHorizon(
                hours=3, reach_km=2.1, rate_kmh_lo=0.5, rate_kmh_hi=1.8, heading_deg=315.0, area_km2=0.6,
                ring=[[76.62, 15.12], [76.60, 15.14], [76.61, 15.16], [76.64, 15.14], [76.62, 15.12]]
            ),
            schemas.SpreadHorizon(
                hours=6, reach_km=4.8, rate_kmh_lo=0.6, rate_kmh_hi=2.1, heading_deg=315.0, area_km2=1.8,
                ring=[[76.62, 15.12], [76.58, 15.16], [76.60, 15.19], [76.66, 15.15], [76.62, 15.12]]
            )
        ],
        uncertainty_ring=[[76.62, 15.12], [76.56, 15.18], [76.60, 15.22], [76.68, 15.16], [76.62, 15.12]]
    )

# --- 8. Exposure (api-contract Section 8) ---
@router.get("/events/{event_id}/exposure", response_model=schemas.ExposureResponse)
def get_event_exposure(event_id: str):
    return schemas.ExposureResponse(
        zones=[
            schemas.ExposureZone(
                name="Smoke plume, core, next 6 hours",
                area_km2=40.6,
                settlements=3,
                schools=1,
                clinics=0,
                road_km=12.4,
                water_bodies=1
            )
        ]
    )

# --- 9. Similar Events (api-contract Section 9) ---
@router.get("/events/{event_id}/similar", response_model=schemas.SimilarEventsResponse)
def get_similar_events(event_id: str):
    return schemas.SimilarEventsResponse(
        matches=[
            schemas.SimilarEventMatch(event_id="E0091", similarity=0.88, outcome="confirmed"),
            schemas.SimilarEventMatch(event_id="E0142", similarity=0.81, outcome="unreviewed")
        ]
    )

# --- 10. Analyst Reviews (api-contract Section 10) ---
@router.get("/reviews", response_model=List[schemas.ReviewResponse])
def list_reviews(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return db.query(models.AnalystReview).offset(skip).limit(limit).all()

@router.post("/events/{event_id}/reviews", response_model=schemas.ReviewResponse)
def submit_review(event_id: str, review: schemas.ReviewCreate, db: Session = Depends(get_db)):
    db_review = models.AnalystReview(
        event_id=event_id,
        verdict=review.verdict,
        analyst_class=review.analyst_class,
        note=review.note,
        reviewer=review.reviewer or "Analyst",
        at=review.at or datetime.utcnow().strftime("%d %b %Y, %H:%M IST")
    )
    db.add(db_review)
    
    # Update event review status if event exists
    event = db.query(models.Event).filter(models.Event.id == event_id).first()
    if event:
        if review.verdict == "confirm":
            event.status = "confirmed"
        elif review.verdict == "false":
            event.status = "false_alarm"
        elif review.verdict == "change" and review.analyst_class:
            event.predicted_class = review.analyst_class
            event.status = "confirmed"
            
    db.commit()
    db.refresh(db_review)
    return db_review
