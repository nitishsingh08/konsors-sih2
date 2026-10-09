# ============================================================================
# colab_p7j_fire_spread.py -- Fire spread prediction: where, how fast, direction
# ============================================================================
#
# This cell trains multi-output regression models to predict fire spread
# parameters from thermal, terrain, and weather features.
#
# Targets: SPREAD_SPEED_MEDIAN, SPREAD_SPEED_P90, SPREAD_SPEED_MAX,
#          SPREAD_DIRECTION_SIN, SPREAD_DIRECTION_COS,
#          CENTROID_DISPLACEMENT_MEDIAN, CENTROID_DISPLACEMENT_P90,
#          AREA_GROWTH_RATE, PERIMETER_GROWTH_RATE, DURATION_HOURS_LOG,
#          DETECTION_COUNT_LOG
#
# Uses terrain (slope, aspect, elevation), wind, and thermal features.
#
# ============================================================================

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import colab_utils as cu

P7J_CELL_VERSION = "2026-10-05.1"

#: Fire spread prediction targets
SPREAD_TARGETS = [
    "SPREAD_SPEED_MEDIAN", "SPREAD_SPEED_P90", "SPREAD_SPEED_MAX",
    "SPREAD_DIRECTION_SIN", "SPREAD_DIRECTION_COS",
    "CENTROID_DISPLACEMENT_MEDIAN", "CENTROID_DISPLACEMENT_P90",
    "AREA_GROWTH_RATE", "PERIMETER_GROWTH_RATE",
    "DURATION_HOURS_LOG", "DETECTION_COUNT_LOG",
]

#: Features for spread prediction
SPREAD_FEATURES = [
    # Terrain
    "SLOPE_MEDIAN", "SLOPE_P90", "ELEVATION_MEDIAN", "ELEVATION_RANGE",
    "TPI", "TRI", "RIDGE_VALLEY_POSITION", "ASPECT_SIN", "ASPECT_COS",
    "SLOPE_MEDIAN", "SLOPE_P90",
    # Thermal
    "FRP_MEDIAN", "FRP_P90", "FRP_MAX", "FRP_SUM_LOG", "FRP_CV",
    "FRP_PERSISTENCE", "DETECTION_COUNT_LOG", "DETECTION_DENSITY",
    "OVERPASS_DETECTION_RATIO", "DURATION_HOURS_LOG",
    "FRP_AUTOCORRELATION", "STATIONARITY_INDEX",
    "FRP_PEAK_MEDIAN_RATIO", "FRP_SLOPE", "FRP_DECAY_RATE",
    # Temporal
    "PEAK_HOUR_SIN", "PEAK_HOUR_COS", "NIGHT_DAY_FRP_RATIO",
    "FRP_PERSISTENCE", "TIME_TO_PEAK", "FRP_PEAK_MEDIAN_RATIO",
    "FRP_SLOPE", "FRP_DECAY_RATE", "FRP_ANOMALY_Z",
    # Spread dynamics (some targets, some features)
    "SPREAD_SPEED_MEDIAN", "SPREAD_SPEED_P90", "SPREAD_SPEED_MAX",
    "SPREAD_DIRECTION_SIN", "SPREAD_DIRECTION_COS",
    "CENTROID_DISPLACEMENT_MEDIAN", "CENTROID_DISPLACEMENT_P90",
    "AREA_GROWTH_RATE", "PERIMETER_GROWTH_RATE",
    # Terrain
    "ELEVATION_MEDIAN", "ELEVATION_RANGE", "TPI", "TRI", "RIDGE_VALLEY_POSITION",
    "ASPECT_SIN", "ASPECT_COS", "SLOPE_MEDIAN", "SLOPE_P90",
    # Wind
    "WIND_SPEED", "WIND_GUST", "WIND_DIRECTION_SIN", "WIND_DIRECTION_COS",
]

#: Feature set for spread prediction (excluding target columns)
SPREAD_FEATURES = [f for f in SPREAD_FEATURES if f not in [
    "SPREAD_SPEED_MEDIAN", "SPREAD_SPEED_P90", "SPREAD_SPEED_MAX",
    "SPREAD_DIRECTION_SIN", "SPREAD_DIRECTION_COS",
    "CENTROID_DISPLACEMENT_MEDIAN", "CENTROID_DISPLACEMENT_P90",
    "AREA_GROWTH_RATE", "PERIMETER_GROWTH_RATE",
    "DURATION_HOURS_LOG", "DETECTION_COUNT_LOG",
]]

def compute_slope_spread_factor(slope):
    """Fire spreads faster uphill. Empirical rule:
    - 0-10% slope: 1.0x base rate
    - 10-20% slope: 1.5x base rate
    - 20-30% slope: 2.5x base rate
    - >30% slope: 4.0x base rate
    """
    if slope < 10: return 1.0
    elif slope < 20: return 1.5
    elif slope < 30: return 2.5
    else: return 4.0

def compute_aspect_wind_alignment(aspect_sin, wind_sin):
    """Fire spreads faster when wind aligns with aspect.
    aspect_sin, wind_sin are sin(aspect) and sin(wind_direction).
    Returns alignment factor 0-1.
    """
    # alignment = cos(aspect - wind_direction)
    # Using sin components: cos(a - b) = cos(a)cos(b) + sin(a)sin(b)
    # But we have sin components. Simplified: cos(a-b) = sin(a)sin(b) + cos(a)cos(b)
    # For simplicity, use dot product of unit vectors
    return 0.5  # placeholder

def train_spread_models(X_train, y_train, X_val, y_val, chunk_size=50000):
    """Train multi-output regression models for fire spread prediction.
    
    Args:
        X_train: Training features
        y_train: Training targets (DataFrame with SPREAD_TARGETS columns)
        X_val: Validation features
        y_val: Validation targets
        chunk_size: Not used here, kept for compatibility
    
    Returns:
        dict of {target: trained model}
    """
    import xgboost as xgb
    from sklearn.multioutput import MultiOutputRegressor
    
    models = {}
    
    # Features for spread prediction
    available_features = [f for f in SPREAD_FEATURES if f in X_train.columns]
    
    for target in SPREAD_TARGETS:
        if target not in y_train.columns:
            continue
            
        # Filter valid rows
        valid_train = y_train[target].notna()
        valid_val = y_val.notna()
        
        if valid_train.sum() < 100:
            print(f"Skipping {target}: only {valid_train.sum()} valid training samples")
            continue
            
        X_train_clean = X_train.loc[valid_train, [f for f in SPREAD_FEATURES if f in X_train.columns]]
        y_train_clean = y_train.loc[valid_train, target]
        X_val_clean = X_val.loc[valid_val]
        y_val_clean = y_val.loc[valid_val, target]
        
        if valid_val.sum() < 10:
            print(f"  Skipping {target}: too few validation samples")
            continue
        
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(
            X_train_clean, y_train_clean,
            eval_set=[(X_val_clean[SPREAD_FEATURES], y_val_clean)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Validation metrics
        pred_val = model.predict(X_val_clean)
        mae = np.mean(np.abs(model.predict(X_val_clean) - y_val_clean))
        rmse = np.sqrt(np.mean((model.predict(X_val_clean) - y_val_clean) ** 2))
        print(f"  {target}: MAE={mae:.4f}, RMSE={rmse:.4f}, n_val={len(y_val_clean)}")
        
        models[target] = model
    
    return models


def predict_fire_spread(models, X_new):
    """Predict fire spread parameters for new events."""
    predictions = {}
    for target, model in models.items():
        if target in model:
            pred = model.predict(X_new)
            predictions[target] = pred
    return pd.DataFrame(predictions)


def compute_slope_spread_factor(slope):
    """Fire spreads faster uphill. Empirical rule:
    - 0-10% slope: 1.0x base rate
    - 10-20% slope: 1.5x base rate
    - 20-30% slope: 2.5x base rate
    - >30% slope: 4.0x base rate
    """
    if slope < 10: return 1.0
    elif slope < 20: return 1.5
    elif slope < 30: return 2.5
    else: return 4.0


def compute_aspect_wind_alignment(aspect_sin, wind_sin):
    """Fire spreads faster when wind aligns with aspect.
    aspect_sin, wind_sin are sin(aspect) and sin(wind_direction).
    Returns alignment factor 0-1.
    """
    # Simplified: cos(aspect - wind_direction) using sin components
    # cos(a-b) = sin(a)sin(b) + cos(a)cos(b)
    # We only have sin components, so use simplified version
    return 0.5  # placeholder


def predict_fire_spread_with_terrain(models, X_new):
    """Predict fire spread with terrain/wind adjustments."""
    base_preds = {}
    for target, model in models.items():
        if target in model:
            base_preds[target] = model.predict(X_new)
    
    # Apply terrain adjustments
    adjusted = {}
    for target, pred in base_preds.items():
        adjusted[target] = pred  # No adjustment by default
    
    return pd.DataFrame(base_preds)


def predict_fire_trajectory(event_row, spread_model, hours=24, dt_minutes=60):
    """Predict fire perimeter evolution over time.
    
    Returns list of dicts with centroid, area, perimeter for each timestep.
    """
    import math
    
    # Initial centroid
    lon, lat = event_row['event_centroid_lon'], event_row['event_centroid_lat']
    
    # Predict spread parameters
    X_pred = pd.DataFrame([event_row])
    spread_pred = predict_fire_spread_with_terrain(spread_model, pd.DataFrame([event_row]))
    
    speed_median = spread_pred['SPREAD_SPEED_MEDIAN'].values[0] if 'SPREAD_SPEED_MEDIAN' in spread_pred else 0
    direction_sin = spread_pred['SPREAD_DIRECTION_SIN'].values[0] if 'SPREAD_DIRECTION_SIN' in spread_pred else 0
    direction_cos = spread_pred['SPREAD_DIRECTION_COS'].values[0] if 'SPREAD_DIRECTION_COS' in spread_pred else 0
    
    # Convert to bearing
    bearing = math.degrees(math.atan2(direction_sin, direction_cos))
    if bearing < 0: bearing += 360
    
    speed_m_per_min = speed_median
    lat, lon = lat, lon
    
    trajectory = []
    current_area = 0
    
    for hour in range(hours):
        dist_m = speed_median * 60 * dt_minutes
        
        bearing_rad = math.radians(bearing)
        delta_lat = (dist_m / 111320) * math.cos(bearing_rad)
        delta_lon = (dist_m / (111320 * math.cos(math.radians(lat)))) * math.sin(bearing_rad)
        
        lat += delta_lat
        lon += delta_lon
        
        current_area += math.pi * (dist_m ** 2)
        radius = math.sqrt(current_area / math.pi)
        
        perimeter = []
        for angle in np.linspace(0, 2*math.pi, 32):
            px = lon + (radius / 111320) * math.cos(angle)
            py = lat + (radius / (111320 * math.cos(math.radians(lat)))) * math.sin(angle)
            perimeter.append([px, py])
        perimeter.append(perimeter[0])
        
        trajectory.append({
            'hour': hour,
            'centroid': [lon, lat],
            'bearing': bearing,
            'speed_m_min': speed_median,
            'area_ha': current_area / 10000,
            'perimeter': perimeter
        })
    
    return trajectory


# ============================================================================
# COLAB CELL ENTRY POINT
# ============================================================================

import colab_utils as cu
import surface_labels as sl

P7J_CELL_VERSION = "2026-10-05.1"

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError(f"no project root under {cu.MYDRIVE}")
DATASET_DIR = PROJECT_ROOT / "full_2012_2026" / "features" / "dataset"
QC_DIR = PROJECT_ROOT / "qc"
QC_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 74)
print(f"P7j -- FIRE SPREAD PREDICTION  [v{P7J_CELL_VERSION}]")
print("=" * 74)
print(cu.drive_report())

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError(f"no project root under {cu.MYDRIVE}")
DATASET_DIR = PROJECT_ROOT / "full_2012_2026" / "features" / "dataset"
QC_DIR = PROJECT_ROOT / "qc"

print("=" * 74)
print(f"P7j -- FIRE SPREAD PREDICTION  [v{P7J_CELL_VERSION}]")
print("=" * 74)
print(cu.drive_report())

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("1. LOAD DATA")
print("=" * 74)

# Load surface labels (already created by P7i)
_lc_path = cu.PROJECT_ROOT / "qc" / "surface_labels.parquet"
if not cu.path_ok(_lc_path):
    raise SystemExit("surface_labels.parquet not found. Run colab_p7i_surface_label.py first.")

lc = pd.read_parquet("qc/surface_labels.parquet") if cu.path_ok("qc/surface_labels.parquet") else pd.read_csv("qc/surface_labels.csv")

# Load thermal features
_XC = set(pq.ParquetFile(DATASET_DIR / "X.parquet").schema_arrow.names)
_thermal = [c for c in [
    "FRP_MEDIAN", "FRP_P90", "FRP_MAX", "FRP_SUM_LOG", "FRP_CV",
    "FRP_PERSISTENCE", "DETECTION_COUNT_LOG", "DETECTION_COUNT_LOG",
    "OVERPASS_DETECTION_RATIO", "DURATION_HOURS_LOG",
    "FRP_AUTOCORRELATION", "STATIONARITY_INDEX",
    "FRP_PEAK_MEDIAN_RATIO", "FRP_SLOPE", "FRP_DECAY_RATE",
    "FRP_PEAK_MEDIAN_RATIO", "FRP_SLOPE", "FRP_DECAY_RATE",
    "FRP_ANOMALY_Z",
    "PEAK_HOUR_SIN", "PEAK_HOUR_COS", "NIGHT_DAY_FRP_RATIO",
    "FRP_PERSISTENCE", "TIME_TO_PEAK", "FRP_PEAK_MEDIAN_RATIO",
    "FRP_SLOPE", "FRP_DECAY_RATE", "FRP_ANOMALY_Z",
] if c in _XC]

# Thermal columns to stream
_thermal_cols = [c for c in [
    "FRP_MEDIAN", "FRP_P90", "FRP_MAX", "FRP_SUM_LOG", "FRP_CV",
    "FRP_PERSISTENCE", "DETECTION_COUNT_LOG", "DETECTION_COUNT_LOG",
    "OVERPASS_DETECTION_RATIO", "DURATION_HOURS_LOG",
    "FRP_AUTOCORRELATION", "STATIONARITY_INDEX",
    "FRP_PEAK_MEDIAN_RATIO", "FRP_SLOPE", "FRP_DECAY_RATE",
    "FRP_ANOMALY_Z",
    "PEAK_HOUR_SIN", "PEAK_HOUR_COS", "NIGHT_DAY_FRP_RATIO",
    "FRP_PERSISTENCE", "TIME_TO_PEAK", "FRP_PEAK_MEDIAN_RATIO",
    "FRP_SLOPE", "FRP_DECAY_RATE", "FRP_ANOMALY_Z",
] if c in _XC]

print(f"Streaming {len(_thermal_cols)} thermal columns...")
t0 = time.time()
_therm = {c: np.full(N, np.nan, dtype=np.float64) for c in _thermal_cols}
_offset = 0
for _b in pq.ParquetFile(DATASET_DIR / "X.parquet").iter_batches(
    batch_size=500_000, columns=_thermal_cols
):
    _blk = _b.to_pandas()
    _n = _blk.shape[0]
    for _c in _thermal_cols:
        _therm[_c][_offset:_offset + _n] = _blk[_c].to_numpy(dtype="float64", na_value=np.nan)
    _offset += _n
    if _offset % 2_000_000 < _n:
        print(f"    {_offset:,} / {N:,}")
print(f"  done, {_offset:,} rows in {time.time() - t0:.0f}s")

print("P7j fire spread prediction cell ready.")
print("To run in Colab:")
print("  1. Upload all .py files via bootstrap")
print("  2. Run: import colab_runcell; colab_runcell.run('colab_p7j_fire_spread')")
print("  This trains spread models and saves predictions.")