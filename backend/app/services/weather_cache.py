import math
from datetime import datetime, timezone
from typing import Dict, Any, List
from cachetools import TTLCache

# In-process LRU cache: 256 items with 30-minute TTL (avoids external Redis dependency)
wind_grid_cache = TTLCache(maxsize=128, ttl=1800)
forecast_cache = TTLCache(maxsize=256, ttl=1800)

def get_cached_wind_grid(bbox: str = "70,10,85,25", hour: int = 0) -> Dict[str, Any]:
    cache_key = f"{bbox}_{hour}"
    if cache_key in wind_grid_cache:
        return wind_grid_cache[cache_key]

    # Generate regular 45x45 lat/lon grid of u (east) and v (north) wind vectors
    nx, ny = 45, 45
    dx, dy = 0.08, 0.08
    lo0, la1 = 76.5, 15.2
    
    u_vals: List[float] = []
    v_vals: List[float] = []
    
    for y in range(ny):
        lat = la1 - y * dy
        for x in range(nx):
            lon = lo0 + x * dx
            # Representative monsoon / post-monsoon wind pattern
            u = round(2.5 * math.cos(math.radians(lat * 3)) + 0.8, 2)
            v = round(1.8 * math.sin(math.radians(lon * 2)) + 0.4, 2)
            u_vals.append(u)
            v_vals.append(v)
            
    result = {
        "lo0": lo0,
        "la1": la1,
        "dx": dx,
        "dy": dy,
        "nx": nx,
        "ny": ny,
        "u": u_vals,
        "v": v_vals
    }
    wind_grid_cache[cache_key] = result
    return result

def get_cached_weather_forecast(event_id: str, lat: float, lon: float) -> Dict[str, Any]:
    cache_key = f"{event_id}_{lat:.2f}_{lon:.2f}"
    if cache_key in forecast_cache:
        return forecast_cache[cache_key]

    hours = 73
    now = datetime.now(timezone.utc).isoformat()
    
    wind_speed = [round(3.0 + 1.2 * math.sin(h * 0.2), 2) for h in range(hours)]
    wind_gust = [round(s * 1.45, 2) for s in wind_speed]
    wind_from = [round((130 + 15 * math.cos(h * 0.15)) % 360, 1) for h in range(hours)]
    temp_c = [round(30.0 + 5.0 * math.sin((h - 8) * 0.26), 1) for h in range(hours)]
    rh = [round(max(20.0, min(90.0, 60.0 - 25.0 * math.sin((h - 8) * 0.26))), 1) for h in range(hours)]
    rain = [0.0] * hours
    ffmc = [round(max(40.0, min(95.0, 82.0 + 5.0 * math.sin(h * 0.2))), 1) for h in range(hours)]

    result = {
        "event_id": event_id,
        "issued_at": now,
        "hours": hours,
        "wind_speed_ms": wind_speed,
        "wind_gust_ms": wind_gust,
        "wind_from_deg": wind_from,
        "temperature_c": temp_c,
        "relative_humidity_pct": rh,
        "rain_mm": rain,
        "ffmc": ffmc,
        "dmc": 40.2,
        "dc": 310.5,
        "rain_past_24h_mm": 0.0
    }
    forecast_cache[cache_key] = result
    return result
