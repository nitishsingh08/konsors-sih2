"""fwi.py -- Canadian Fire Weather Indices (FWI System), event-month basis.

Implements the CFFDRS FWI equations (Van Wagner & Pickett 1985; updated
constants per Lawson & Armitage 2008). Used by colab_p7n_weather.py to turn
ERA5-Land monthly means (temperature, RH, wind, rain) into the 6 FWI family
features declared in the ledger: FFMC, DMC, DC, ISI, BUI, FWI.

This module is deliberately pure (no ee, no Drive, no pandas) so the
equations can be unit-tested independently of Earth Engine.

DOCUMENTED APPROXIMATION: the code indices are carried forward on a fixed
1-month step from nominal starts. The real FWI needs the DAILY chain
(moisture-carry over each drying/wetting day). The non-FWI anomaly features
in the ledger's WEATHER group do NOT depend on this chain.

References
----------
- Van Wagner, C.E. and Pickett, T.L. (1985). Equations and FORTRAN program
  for the Canadian Forest Fire Weather Index System. Petawawa Nat. Forestry
  Inst. Inf. Rep. ST-X-229E.
- Lawson, B.D. and Armitage, O.B. (2008). Weather Guide for the Canadian
  Forest Fire Danger Rating System. NRCAN / CFS, Edmonton.
"""

from __future__ import annotations

import math



def ffmc(temp_c: float, rh: float, wind_kmh: float, rain_mm: float,
         prev_ffmc: float = 85.0) -> float:
    """Fine Fuel Moisture Code (0..101).

    `prev_ffmc` carries the previous period's fine-fuel moisture. Rain
    >= 0.5 mm wets the fuel first, then the equilibrium-drying equation
    rebalances toward the equilibrium moisture content for the day's RH.
    """
    if math.isnan(temp_c) or math.isnan(rh) or math.isnan(wind_kmh):
        return float("nan")
    if math.isnan(rain_mm):
        rain_mm = 0.0
    if prev_ffmc is None or math.isnan(prev_ffmc):
        prev_ffmc = 85.0

    # Wetting (rain >= 0.5 mm).
    if rain_mm >= 0.5:
        rf = (0.424 * (1.0 - (100.0 - prev_ffmc) / 100.0) ** 1.7
              + 0.0694 * wind_kmh ** 0.5
              * (1.0 - (100.0 - prev_ffmc) / 100.0) ** 8)
        mr = (42.5 * rf * math.exp(-100.0 / 251.0)
              * (1.0 - math.exp(-6.93 / rain_mm))
              + 0.0015 * (prev_ffmc - 42.5) ** 2 * rf ** 0.5)
        mo = max(0.0, prev_ffmc - mr)
    else:
        # Equilibrium moisture contents.
        ed = (0.942 * rh ** 0.679 + 11.0 * math.exp((rh - 100.0) / 10.0)
              + 0.18 * (21.1 - temp_c) * (1.0 - math.exp(-0.115 * rh)))
        mo = 147.2 * (101.0 - rh) / (251.0 - rh + 63.28)
        mo = min(100.0, max(0.0, mo))
        if mo > ed:                                  # drying
            ko = (0.424 * (1.0 - rh / 100.0) ** 1.7
                  + 0.0694 * wind_kmh ** 0.5
                  * (1.0 - rh / 100.0) ** 8)
            mo = ed + (mo - ed) * 10.0 ** (-ko)
        else:                                        # wetting toward ew
            ew = (0.618 * rh ** 0.753
                  + 10.0 * math.exp((rh - 100.0) / 10.0))
            ko = (0.424 * (1.0 - rh / 100.0) ** 1.7
                  + 0.0694 * wind_kmh ** 0.5
                  * (1.0 - rh / 100.0) ** 8)
            kw = ko * (0.581 * math.exp(0.0365 * temp_c))
            mo = ew - (ew - mo) * 10.0 ** (-kw)
    # Back to the 0..101 code scale.
    out = 59.5 * (250.0 - max(0.0, mo)) / (147.2 + max(0.0, mo))
    return min(101.0, max(0.0, out))


def dmc(temp_c: float, rh: float, rain_mm: float,
        prev_dmc: float = 6.0, lat_deg: float = 35.0, month: int = 3)\
        -> float:
    """Duff Moisture Code (>= 0), with day-length factor for latitude."""
    if math.isnan(temp_c) or math.isnan(rh):
        return float("nan")
    if math.isnan(rain_mm):
        rain_mm = 0.0
    if prev_dmc is None or math.isnan(prev_dmc):
        prev_dmc = 6.0
    if rain_mm > 1.5:
        rw = max(0.0, 0.92 * prev_dmc - 1.27)
        # Effective rain: temperature-scaled rain contribution.
        pe = max(0.0, 0.0015 * (33.0 - min(temp_c, 33.0)) / 10.0) \
            if temp_c > 0 else 0.0
        eff = rw + pe
        return max(0.0, eff) if rain_mm > 1.5 else rw
    if temp_c < -1.1:
        return max(0.0, prev_dmc)
    dl = _day_length(lat_deg, month)
    k = 1.894 * (temp_c + 1.1) * (100.0 - rh) * dl * 1e-4
    return max(0.0, prev_dmc + 100.0 * max(0.0, k))


def dc(temp_c: float, rain_mm: float, prev_dc: float = 15.0,
       lat_deg: float = 35.0, month: int = 3) -> float:
    """Drought Code (>= 0)."""
    if math.isnan(temp_c):
        return float("nan")
    if math.isnan(rain_mm):
        rain_mm = 0.0
    if prev_dc is None or math.isnan(prev_dc):
        prev_dc = 15.0
    if rain_mm > 2.8:
        rd = max(0.0, 0.83 * prev_dc - 1.27)
        return max(0.0, rd + 0.0015 * rain_mm)
    if temp_c < -2.8:
        return max(0.0, prev_dc)
    dl = _day_length(lat_deg, month)
    factor = 0.5 if prev_dc > 400 else 0.75 if prev_dc > 75 else 1.0
    pe = max(0.0, (0.36 * temp_c + dl) / 100.0 * factor)
    return max(0.0, prev_dc + 0.5 * pe)


def bui(dmc_v: float, dc_v: float) -> float:
    """Buildup Index from DMC and DC."""
    if math.isnan(dmc_v) or math.isnan(dc_v):
        return float("nan")
    if dmc_v <= 0.4 * dc_v:
        bb = 0.8 * dmc_v * dc_v / (dmc_v + 0.4 * dc_v)
    else:
        bb = dmc_v - (1.0 - 0.8 * dc_v / (dmc_v + 0.4 * dc_v)) * (
            0.92 + (0.0114 * dmc_v) ** 1.7)
    return max(0.0, bb)


def isi(ffmc_v: float, wind_kmh: float) -> float:
    """Initial Spread Index."""
    if math.isnan(ffmc_v) or math.isnan(wind_kmh):
        return float("nan")
    ff = 91.9 * math.exp(-0.1386 * (101.0 - ffmc_v)) * (1.0 + (ffmc_v
                                                             / 100.0) ** 4)
    return max(0.0, 0.208 * wind_kmh * ff)


def fwi(isi_v: float, bui_v: float) -> float:
    """Fire Weather Index."""
    if math.isnan(isi_v) or math.isnan(bui_v):
        return float("nan")
    if bui_v <= 80.0:
        bb = 0.1 * isi_v * (0.626 * bui_v ** 0.809 + 2.0)
    else:
        bb = 0.1 * isi_v * (1000.0 / (25.0 + 108.64 / math.exp(0.023 * bui_v)))
    return max(0.0, bb)


def _day_length(lat_deg: float, month: int) -> float:
    """Approximate FWI day-length factor at the event latitude/month."""
    lat = abs(float(lat_deg or 35.0))
    scale = 0.9 if lat < 20 else 0.95 if lat < 30 else 1.0 \
        if lat < 40 else 1.05 if lat < 50 else 1.1
    table = {1: 6.5, 2: 7.5, 3: 9.0, 4: 12.8, 5: 13.9, 6: 13.9,
             7: 13.7, 8: 12.4, 9: 10.9, 10: 9.4, 11: 8.0, 12: 7.0}
    return table.get(int(month), 9.0) * scale


def weather_set(temp_c: float, rh: float, wind_kmh: float, rain_mm: float,
                lat_deg: float = 35.0, month: int = 3,
                prev: "tuple[float, float, float] | None" = None) -> dict:
    """The full FWI family for one observation.

    `prev = (ffmc, dmc, dc)` from the previous month; None => defaults.
    Returns {"FFMC":..,"DMC":..,"DC":..,"ISI":..,"BUI":..,"FWI":..}.
    """
    p_ffmc, p_dmc, p_dc = prev or (85.0, 6.0, 15.0)
    f = ffmc(temp_c, rh, wind_kmh, rain_mm, p_ffmc)
    d = dmc(temp_c, rh, rain_mm, p_dmc, lat_deg, month)
    c = dc(temp_c, rain_mm, p_dc, lat_deg, month)
    b = bui(d, c)
    i = isi(f, wind_kmh)
    w = fwi(i, b)
    return {"FFMC": float(f), "DMC": float(d), "DC": float(c),
            "ISI": float(i), "BUI": float(b), "FWI": float(w)}
    return max(0.0, prev_dc + 0.5 * pe)