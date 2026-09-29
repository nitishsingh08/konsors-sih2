import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base

class Place(Base):
    __tablename__ = "places"

    id = Column(String(32), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    state = Column(String(100), nullable=False)
    category = Column(String(50), nullable=True) # industrial, forest, farmland, mine
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    events = relationship("Event", back_populates="place")

class Event(Base):
    __tablename__ = "events"

    id = Column(String(32), primary_key=True, index=True)
    place_id = Column(String(32), ForeignKey("places.id"), nullable=True)
    detected_at = Column(DateTime, nullable=False, index=True)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    frp_median = Column(Float, default=0.0)
    
    # 6-class taxonomy: wildfire | agricultural_burning | gas_flare | industrial | mining | unknown
    predicted_class = Column(String(50), nullable=False, default="unknown", index=True)
    confidence = Column(Float, default=0.5) # Calibrated against heuristic rules
    
    # Operational distinction:
    baseline_status = Column(String(20), nullable=False, default="routine", index=True) # routine | abnormal
    status = Column(String(30), nullable=False, default="unreviewed", index=True)       # unreviewed | confirmed | false_alarm
    
    missing_fraction = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    place = relationship("Place", back_populates="events")
    features = relationship("EventFeatures", back_populates="event", uselist=False, cascade="all, delete-orphan")
    geometries = relationship("EventGeometry", back_populates="event", cascade="all, delete-orphan")
    reviews = relationship("AnalystReview", back_populates="event", cascade="all, delete-orphan")

class EventFeatures(Base):
    __tablename__ = "event_features"

    event_id = Column(String(32), ForeignKey("events.id"), primary_key=True)
    feature_schema_version = Column(Integer, default=1, index=True)
    features_json = Column(JSON, nullable=False) # 141 features dict matching api-contract
    shap_json = Column(JSON, nullable=True)       # Precomputed SHAP contributions (O(1) lookup)
    embedding = Column(JSON, nullable=True)       # Stored as JSON array or VECTOR(141) in Postgres

    event = relationship("Event", back_populates="features")

class EventGeometry(Base):
    __tablename__ = "event_geometries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(32), ForeignKey("events.id"), nullable=False, index=True)
    layer_type = Column(String(50), nullable=False) # plume_core | plume_outer | spread_outlook | footprint
    horizon_hours = Column(Integer, nullable=True)
    geometry_json = Column(JSON, nullable=False)    # Coordinates array [[lon, lat], ...]
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    event = relationship("Event", back_populates="geometries")

class AnalystReview(Base):
    __tablename__ = "analyst_reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(32), ForeignKey("events.id"), nullable=False, index=True)
    verdict = Column(String(20), nullable=False)    # confirm | change | false | field
    analyst_class = Column(String(50), nullable=True) # 6-class taxonomy if reclassified
    note = Column(Text, nullable=True)
    reviewer = Column(String(100), nullable=True)
    at = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    event = relationship("Event", back_populates="reviews")
