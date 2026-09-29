from typing import Dict, Any, Tuple
from datetime import datetime

def evaluate_heuristic_rules(
    lat: float,
    lon: float,
    detected_at: datetime,
    features: Dict[str, Any]
) -> Tuple[str, float, str]:
    """
    Deterministic rule-based labeling engine for 6-class taxonomy:
    wildfire | agricultural_burning | gas_flare | industrial | mining | unknown
    
    Returns:
        (predicted_class, calibrated_confidence, baseline_status)
    """
    # Extract key spatial, temporal, and land cover indicators
    dist_facility = features.get("DIST_NEAREST_FACILITY_LOG", {}).get("value", 999.0)
    dist_refinery = features.get("DIST_REFINERY_LOG", {}).get("value", 999.0)
    dist_mine = features.get("DIST_MINE_LOG", {}).get("value", 999.0)
    is_known_flare = features.get("KNOWN_FLARE", {}).get("value", 0)
    inside_facility = features.get("INSIDE_FACILITY", {}).get("value", 0)
    stationarity = features.get("STATIONARITY_INDEX", {}).get("value", 0.0)
    
    forest_frac = features.get("FOREST_FRAC_1KM", {}).get("value", 0.0)
    cropland_frac = features.get("CROPLAND_FRAC_1KM", {}).get("value", 0.0)
    bare_mining_frac = features.get("BARE_MINING_FRAC_1KM", {}).get("value", 0.0)
    
    fwi = features.get("FWI", {}).get("value", 10.0)
    spread_speed = features.get("SPREAD_SPEED_MEDIAN", {}).get("value", 0.0)
    recurrence_3y = features.get("RECURRENCE_RATE_3Y", {}).get("value", 0.0)
    
    month = detected_at.month
    is_stubble_season = (month in [10, 11]) or (month in [4, 5]) # Kharif post-harvest (Oct-Nov) or Rabi (Apr-May)

    # RULE 1: Gas Flare (Known nightfire match OR inside refinery with high stationarity)
    if is_known_flare == 1 or (dist_refinery <= 1.0 and stationarity >= 0.85):
        # A permitted stationary flare is routine for site baseline
        baseline = "routine" if recurrence_3y > 0.5 else "abnormal"
        confidence = 0.92 if is_known_flare == 1 else 0.84
        return "gas_flare", confidence, baseline

    # RULE 2: Industrial Heat Source (Factory, Powerplant, Steel mill, Kiln)
    if inside_facility == 1 or (dist_facility <= 1.0 and stationarity >= 0.75):
        baseline = "routine" if recurrence_3y > 0.4 else "abnormal"
        confidence = 0.86
        return "industrial", confidence, baseline

    # RULE 3: Mining (Inside active mine boundary or high bare/mining land cover)
    if dist_mine <= 1.5 or bare_mining_frac >= 0.50:
        baseline = "routine" if recurrence_3y > 0.5 else "abnormal"
        confidence = 0.82
        return "mining", confidence, baseline

    # RULE 4: Agricultural Burning (Stubble / Crop Residue)
    if cropland_frac >= 0.40 and is_stubble_season and spread_speed >= 0.10:
        # Rapid moving burn on cropland during harvest window
        baseline = "abnormal"
        confidence = 0.88 if (month in [10, 11] and lat >= 28.0) else 0.80 # Stronger prior for NW India in Autumn
        return "agricultural_burning", confidence, baseline

    # RULE 5: Wildfire (Forest cover with elevated Fire Weather Index)
    if forest_frac >= 0.35 or (fwi >= 20.0 and forest_frac >= 0.20):
        baseline = "abnormal"
        # Calibrate higher if FWI is extreme
        confidence = 0.89 if fwi >= 30.0 else 0.81
        return "wildfire", confidence, baseline

    # RULE 6: Fallback / Unknown
    return "unknown", 0.50, "abnormal"
