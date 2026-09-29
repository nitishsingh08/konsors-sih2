import datetime
import random
from app.core.database import engine, SessionLocal, Base
from app.models.models import Place, Event, EventFeatures, EventGeometry
from app.services.ml_engine import precompute_shap_explanations, generate_141_embedding
from app.services.heuristic_rules import evaluate_heuristic_rules

# Top persistent sites across India
SAMPLE_PLACES = [
    ("GJ-01", "Refinery cluster", "Gujarat", "industrial", 22.35, 69.85),
    ("GJ-02", "Petrochemical complex", "Gujarat", "industrial", 21.72, 72.62),
    ("GJ-03", "Oil field flares", "Gujarat", "gas_flare", 23.55, 72.40),
    ("JH-01", "Coal seam fire zone", "Jharkhand", "mining", 23.75, 86.42),
    ("JH-02", "Steel works & coking plant", "Jharkhand", "industrial", 22.80, 86.20),
    ("OD-01", "Smelter & captive power", "Odisha", "industrial", 20.85, 85.15),
    ("OD-02", "Open-cast coal basin", "Odisha", "mining", 20.95, 85.22),
    ("KA-01", "Steel plant & slag dump", "Karnataka", "industrial", 15.18, 76.65),
    ("PB-01", "Paddy stubble belt", "Punjab", "farmland", 30.90, 75.85),
    ("UK-01", "Pine forest ridge", "Uttarakhand", "forest", 30.15, 78.75)
]

def seed_database():
    print("Connecting to database and creating tables...")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        # Check if already seeded
        if db.query(Place).count() > 0:
            print(f"Database already contains {db.query(Place).count()} places and {db.query(Event).count()} events.")
            return

        print("Seeding initial places...")
        for pid, name, state, cat, lat, lon in SAMPLE_PLACES:
            place = Place(id=pid, name=name, state=state, category=cat, lat=lat, lon=lon)
            db.add(place)
        db.commit()

        print("Seeding active thermal events with 141 features and SHAP...")
        now = datetime.datetime.now(datetime.timezone.utc)

        event_configs = [
            ("E0218", "KA-01", 15.18, 76.65, 125.4, 0.45, 0.10, 0.92, 0.85, 3.8, 12.0),
            ("E0219", "GJ-01", 22.35, 69.85, 95.0, 0.05, 0.02, 0.05, 0.95, 38.0, 8.0),
            ("E0220", "PB-01", 30.90, 75.85, 68.2, 0.10, 0.65, 5.2, 0.20, 2.5, 18.0),
            ("E0221", "UK-01", 30.15, 78.75, 180.5, 0.75, 0.05, 15.0, 0.10, 1.2, 32.0),
            ("E0222", "JH-01", 23.75, 86.42, 110.0, 0.15, 0.05, 0.8, 0.88, 28.0, 10.0)
        ]

        for eid, pid, lat, lon, frp, forest, crop, dist_fac, stat, bt_diff, fwi in event_configs:
            features = {
                "FRP_MEDIAN": {"value": frp, "percentile": 85.0, "missing": False},
                "BT_I4_I5_DIFF_MEDIAN": {"value": bt_diff, "percentile": 72.0, "missing": False},
                "FOREST_FRAC_1KM": {"value": forest, "percentile": 60.0, "missing": False},
                "CROPLAND_FRAC_1KM": {"value": crop, "percentile": 50.0, "missing": False},
                "DIST_NEAREST_FACILITY_LOG": {"value": dist_fac, "percentile": 40.0, "missing": False},
                "STATIONARITY_INDEX": {"value": stat, "percentile": 80.0, "missing": False},
                "FWI": {"value": fwi, "percentile": 75.0, "missing": False},
                "SPREAD_SPEED_MEDIAN": {"value": 0.25 if stat < 0.5 else 0.02, "percentile": 50.0, "missing": False},
                "RECURRENCE_RATE_3Y": {"value": 0.85 if stat > 0.7 else 0.15, "percentile": 80.0, "missing": False},
                "KNOWN_FLARE": {"value": 1 if pid in ["GJ-01", "GJ-03"] else 0, "percentile": 50.0, "missing": False},
                "INSIDE_FACILITY": {"value": 1 if stat > 0.8 else 0, "percentile": 50.0, "missing": False},
                "SMOKE_PROBABILITY": {"value": None, "percentile": None, "missing": True},
                "VV_CHANGE": {"value": None, "percentile": None, "missing": True}
            }

            pred_class, conf, baseline = evaluate_heuristic_rules(lat, lon, now, features)
            shap_data = precompute_shap_explanations(pred_class, features)
            embedding = generate_141_embedding(features)

            event = Event(
                id=eid,
                place_id=pid,
                detected_at=now,
                lat=lat,
                lon=lon,
                frp_median=frp,
                predicted_class=pred_class,
                confidence=conf,
                baseline_status=baseline,
                status="unreviewed",
                missing_fraction=0.08
            )
            db.add(event)

            feat_row = EventFeatures(
                event_id=eid,
                feature_schema_version=1,
                features_json=features,
                shap_json=shap_data,
                embedding=embedding
            )
            db.add(feat_row)

        db.commit()
        print("Database seeded successfully with initial places and events!")
    except Exception as e:
        db.rollback()
        print(f"Error seeding database: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
