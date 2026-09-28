/* =====================================================================
   The 141 event-level features (X_event), grouped as in the model spec.
   Everything here is SAMPLE data. Replace featuresOf() with a call that
   returns the real feature vector for an event (see docs/api-contract.md).
   Port of frontend/js/data/features.js.
   ===================================================================== */
import { clamp, R } from '../lib/core';
import { HOUR } from '../lib/time';
import { FAMILIES, isVeg } from './classes';
import { PMAP, contextOf } from './places';
import { calcBUI, calcFWI, calcISI, offsetLL, rad, weatherOf } from './weather';
import type {
  ClassId,
  Event,
  FamilyId,
  FeatureDef,
  FeatureVector,
  FootprintFrame,
  LonLat,
  Place,
  ShapResult,
} from '../lib/types';

/** key, group, label, unit, source, mean, spread, min, max, flags */
type FeatureRow = [
  key: string, g: FamilyId, label: string, unit: string, src: string,
  mu: number, sd: number, lo: number, hi: number, fl?: string,
];

/* flags: x = model sees log, shown in real units | k = also used to build labels | b = yes/no */
const ROWS: FeatureRow[] = [
  /* 1. Heat (15) */
  ['FRP_MEDIAN', 'thermal', 'Median fire power', 'MW', 'FIRMS', 12, 8, 0.5, 400],
  ['FRP_P90', 'thermal', 'Fire power, top 10% of passes', 'MW', 'FIRMS', 22, 14, 1, 600],
  ['FRP_MAX', 'thermal', 'Peak fire power', 'MW', 'FIRMS', 30, 20, 1, 900],
  ['FRP_SUM_LOG', 'thermal', 'Total heat released (log)', 'log MW', 'FIRMS', 4.6, 1.1, 0.5, 9],
  ['FRP_IQR', 'thermal', 'Fire power spread (IQR)', 'MW', 'FIRMS', 9, 7, 0, 300],
  ['FRP_CV', 'thermal', 'Fire power variability', 'ratio', 'FIRMS', 0.5, 0.3, 0, 3],
  ['FRP_PEAK_MEDIAN_RATIO', 'thermal', 'Peak against median', 'ratio', 'FIRMS', 2.1, 0.9, 1, 12],
  ['FRP_SLOPE', 'thermal', 'Fire power trend', 'MW/h', 'Derived', 0, 1.2, -20, 20],
  ['FRP_DECAY_RATE', 'thermal', 'Fire power decay rate', '/h', 'Derived', 0.06, 0.05, 0, 1],
  ['BT_I4_MEDIAN', 'thermal', 'Brightness, 4 µm band', 'K', 'VIIRS I4', 342, 14, 300, 368],
  ['BT_I5_MEDIAN', 'thermal', 'Brightness, 11 µm band', 'K', 'VIIRS I5', 304, 8, 280, 340],
  ['BT_I4_I5_DIFF_MEDIAN', 'thermal', 'Band gap (4 µm minus 11 µm)', 'K', 'VIIRS I4/I5', 38, 14, 2, 90],
  ['BT_I4_I5_DIFF_P90', 'thermal', 'Band gap, top 10%', 'K', 'VIIRS I4/I5', 52, 16, 5, 110],
  ['SUBPIXEL_TEMP_MEDIAN', 'thermal', 'Estimated temperature of the hot part', 'K', 'Derived', 900, 320, 500, 2300],
  ['SOURCE_AREA_MEDIAN', 'thermal', 'Estimated hot area inside a pixel', 'm²', 'Derived', 2500, 2600, 10, 60000],
  /* 2. Timing (14) */
  ['DURATION_HOURS_LOG', 'temporal', 'How long it has burned', 'h', 'Derived', 3.4, 1.2, 0.7, 8, 'x'],
  ['DETECTION_COUNT_LOG', 'temporal', 'Detections in this event', 'count', 'Derived', 3.3, 1, 0.7, 7, 'x'],
  ['OVERPASS_DETECTION_RATIO', 'temporal', 'Share of passes that saw it', 'frac', 'Derived', 0.6, 0.25, 0.02, 1],
  ['DETECTION_DENSITY', 'temporal', 'Detections per km²', '/km²', 'Derived', 6, 4, 0.2, 60],
  ['RECURRENCE_RATE_1Y', 'temporal', 'Seen again within 1 year', 'frac', 'Derived', 0.3, 0.3, 0, 1, 'k'],
  ['RECURRENCE_RATE_3Y', 'temporal', 'Seen again within 3 years', 'frac', 'Derived', 0.3, 0.3, 0, 1, 'k'],
  ['RECURRENCE_RATE_5Y', 'temporal', 'Seen again within 5 years', 'frac', 'Derived', 0.3, 0.3, 0, 1, 'k'],
  ['MAX_INTER_EVENT_GAP_LOG', 'temporal', 'Longest quiet gap', 'days', 'Derived', 3.2, 1.3, 0, 7, 'x'],
  ['FRP_AUTOCORRELATION', 'temporal', 'Pass-to-pass steadiness', 'r', 'Derived', 0.5, 0.3, -0.6, 1],
  ['TIME_TO_PEAK', 'temporal', 'Time to reach peak', 'h', 'Derived', 8, 8, 0, 72],
  ['FRP_PERSISTENCE', 'temporal', 'Fire power persistence', 'frac', 'Derived', 0.5, 0.25, 0, 1],
  ['NIGHT_DAY_FRP_RATIO', 'temporal', 'Night against day fire power', 'ratio', 'Derived', 1, 0.5, 0.1, 5],
  ['PEAK_HOUR_SIN', 'temporal', 'Peak hour (sine)', '', 'Derived', 0, 1, -1, 1],
  ['PEAK_HOUR_COS', 'temporal', 'Peak hour (cosine)', '', 'Derived', 0, 1, -1, 1],
  /* 3. Movement and growth (17) */
  ['CENTROID_DISPLACEMENT_MEDIAN', 'spread', 'Typical move between passes', 'km', 'Derived', 0.3, 0.3, 0, 12],
  ['CENTROID_DISPLACEMENT_P90', 'spread', 'Biggest moves between passes', 'km', 'Derived', 0.8, 0.7, 0, 25],
  ['SPREAD_SPEED_MEDIAN', 'spread', 'Typical spread speed', 'km/h', 'Derived', 0.15, 0.15, 0, 6],
  ['SPREAD_SPEED_P90', 'spread', 'Fast spread speed', 'km/h', 'Derived', 0.4, 0.4, 0, 8],
  ['SPREAD_SPEED_MAX', 'spread', 'Fastest spread speed', 'km/h', 'Derived', 0.7, 0.6, 0, 10],
  ['CENTROID_JITTER', 'spread', 'Centre wobble', 'km', 'Derived', 0.15, 0.1, 0, 3],
  ['STATIONARITY_INDEX', 'spread', 'Stays in one place', 'frac', 'Derived', 0.6, 0.3, 0, 1],
  ['SPREAD_DIRECTION_SIN', 'spread', 'Spread direction (sine)', '', 'Derived', 0, 1, -1, 1],
  ['SPREAD_DIRECTION_COS', 'spread', 'Spread direction (cosine)', '', 'Derived', 0, 1, -1, 1],
  ['DIRECTION_CONSISTENCY', 'spread', 'Keeps one direction', 'frac', 'Derived', 0.5, 0.3, 0, 1],
  ['AREA_GROWTH_RATE', 'spread', 'Area growth', 'km²/h', 'Derived', 0.1, 0.2, -1, 12],
  ['PERIMETER_GROWTH_RATE', 'spread', 'Edge growth', 'km/h', 'Derived', 0.2, 0.3, -1, 15],
  ['ELONGATION', 'spread', 'Elongation', 'ratio', 'Derived', 1.6, 0.6, 1, 8],
  ['COMPACTNESS', 'spread', 'Compactness', 'frac', 'Derived', 0.6, 0.2, 0.1, 1],
  ['SOLIDITY', 'spread', 'Solidity', 'frac', 'Derived', 0.85, 0.1, 0.3, 1],
  ['NEW_PIXEL_FRACTION', 'spread', 'New pixels each pass', 'frac', 'Derived', 0.2, 0.2, 0, 1],
  ['SPREAD_WIND_SPEED_RATIO', 'spread', 'Spread speed against wind speed', 'ratio', 'Derived', 0.03, 0.03, 0, 0.4],
  /* 4. Terrain (12) */
  ['ELEVATION_MEDIAN', 'terrain', 'Elevation', 'm', 'DEM', 300, 250, 0, 4500],
  ['ELEVATION_IQR', 'terrain', 'Elevation spread (IQR)', 'm', 'DEM', 20, 25, 0, 400],
  ['ELEVATION_RANGE', 'terrain', 'Elevation range', 'm', 'DEM', 60, 70, 0, 1200],
  ['RELATIVE_ELEVATION', 'terrain', 'Height above surroundings', 'm', 'DEM', 0, 25, -150, 150],
  ['SLOPE_MEDIAN', 'terrain', 'Slope', '°', 'DEM', 3, 3, 0, 40],
  ['SLOPE_P90', 'terrain', 'Steep side slope', '°', 'DEM', 7, 5, 0, 55],
  ['ASPECT_SIN', 'terrain', 'Slope faces (sine)', '', 'DEM', 0, 1, -1, 1],
  ['ASPECT_COS', 'terrain', 'Slope faces (cosine)', '', 'DEM', 0, 1, -1, 1],
  ['TPI', 'terrain', 'Topographic position', '', 'DEM', 0, 8, -40, 40],
  ['TRI', 'terrain', 'Ruggedness', 'm', 'DEM', 8, 8, 0, 150],
  ['CURVATURE', 'terrain', 'Curvature', '', 'DEM', 0, 0.5, -3, 3],
  ['RIDGE_VALLEY_POSITION', 'terrain', 'Ridge or valley (−1 valley, +1 ridge)', '', 'DEM', 0, 0.5, -1, 1],
  /* 5. Fuel and land cover (16) */
  ['FOREST_FRAC_375M', 'fuel', 'Forest within 375 m', 'frac', 'WorldCover', 0.15, 0.25, 0, 1, 'k'],
  ['FOREST_FRAC_1KM', 'fuel', 'Forest within 1 km', 'frac', 'WorldCover', 0.15, 0.25, 0, 1, 'k'],
  ['FOREST_FRAC_10KM', 'fuel', 'Forest within 10 km', 'frac', 'WorldCover', 0.15, 0.2, 0, 1, 'k'],
  ['FOREST_LOCAL_REGIONAL_CONTRAST', 'fuel', 'Forest here against the region', '', 'WorldCover', 0, 0.3, -1, 1, 'k'],
  ['CROPLAND_FRAC_1KM', 'fuel', 'Cropland within 1 km', 'frac', 'WorldCover', 0.25, 0.3, 0, 1, 'k'],
  ['CROPLAND_FRAC_10KM', 'fuel', 'Cropland within 10 km', 'frac', 'WorldCover', 0.25, 0.25, 0, 1, 'k'],
  ['CROPLAND_LOCAL_REGIONAL_CONTRAST', 'fuel', 'Cropland here against the region', '', 'WorldCover', 0, 0.3, -1, 1, 'k'],
  ['BUILTUP_FRAC_1KM', 'fuel', 'Built-up land within 1 km', 'frac', 'WorldCover', 0.1, 0.2, 0, 1, 'k'],
  ['BUILTUP_FRAC_10KM', 'fuel', 'Built-up land within 10 km', 'frac', 'WorldCover', 0.1, 0.15, 0, 1, 'k'],
  ['BARE_MINING_FRAC_1KM', 'fuel', 'Bare or mined ground within 1 km', 'frac', 'WorldCover', 0.1, 0.2, 0, 1, 'k'],
  ['WATER_FRAC_1KM', 'fuel', 'Water within 1 km', 'frac', 'WorldCover', 0.04, 0.08, 0, 1],
  ['LANDCOVER_ENTROPY', 'fuel', 'Land cover mix', 'bits', 'WorldCover', 1.1, 0.5, 0, 2.6],
  ['NDVI_MEDIAN', 'fuel', 'Greenness (NDVI)', '', 'HLS', 0.4, 0.2, -0.1, 0.9],
  ['NDVI_ANOMALY_Z', 'fuel', 'Greenness against normal', 'z', 'HLS', 0, 1, -4, 4],
  ['NBR_CHANGE', 'fuel', 'Burn ratio change', '', 'HLS', -0.05, 0.15, -0.8, 0.4],
  ['NDMI_ANOMALY_Z', 'fuel', 'Leaf moisture against normal', 'z', 'HLS', 0, 1, -4, 4],
  /* 6. Industry nearby (18) */
  ['DIST_NEAREST_FACILITY_LOG', 'industry', 'Nearest facility', 'km', 'GEM / OSM', 2.2, 1.4, 0, 6.3, 'xk'],
  ['DIST_REFINERY_LOG', 'industry', 'Nearest refinery', 'km', 'GEM / OSM', 3.4, 1.3, 0, 6.3, 'xk'],
  ['DIST_POWERPLANT_LOG', 'industry', 'Nearest power plant', 'km', 'GEM / OSM', 3.2, 1.3, 0, 6.3, 'xk'],
  ['DIST_PETROCHEMICAL_LOG', 'industry', 'Nearest petrochemical plant', 'km', 'GEM / OSM', 3.5, 1.3, 0, 6.3, 'xk'],
  ['DIST_STEEL_LOG', 'industry', 'Nearest steel plant', 'km', 'GEM / OSM', 3.4, 1.3, 0, 6.3, 'xk'],
  ['DIST_CEMENT_LOG', 'industry', 'Nearest cement plant', 'km', 'GEM / OSM', 3.3, 1.3, 0, 6.3, 'xk'],
  ['DIST_LNG_LOG', 'industry', 'Nearest LNG terminal', 'km', 'GEM / OSM', 4.2, 1.2, 0, 6.3, 'xk'],
  ['DIST_MINE_LOG', 'industry', 'Nearest mine', 'km', 'GEM / OSM', 3.1, 1.3, 0, 6.3, 'xk'],
  ['DIST_LANDFILL_LOG', 'industry', 'Nearest landfill', 'km', 'OSM', 3.3, 1.2, 0, 6.3, 'xk'],
  ['DIST_PIPELINE_LOG', 'industry', 'Nearest pipeline', 'km', 'GEM / OSM', 3, 1.4, 0, 6.3, 'xk'],
  ['DIST_WELLPAD_LOG', 'industry', 'Nearest well pad', 'km', 'GEM / OSM', 3.6, 1.3, 0, 6.3, 'xk'],
  ['FACILITY_COUNT_1KM', 'industry', 'Facilities within 1 km', 'count', 'GEM / OSM', 0.6, 1.1, 0, 25, 'k'],
  ['FACILITY_COUNT_5KM', 'industry', 'Facilities within 5 km', 'count', 'GEM / OSM', 2, 3, 0, 60, 'k'],
  ['FACILITY_COUNT_10KM', 'industry', 'Facilities within 10 km', 'count', 'GEM / OSM', 4, 6, 0, 120, 'k'],
  ['INDUSTRIAL_DENSITY_CONTRAST', 'industry', 'Industry here against the region', '', 'GEM / OSM', 0, 1, -3, 5, 'k'],
  ['INSIDE_FACILITY', 'industry', 'Inside a facility boundary', 'yes/no', 'GEM / OSM', 0.15, 0, 0, 1, 'bk'],
  ['DIST_FACILITY_BOUNDARY_LOG', 'industry', 'Distance to a facility boundary', 'km', 'GEM / OSM', 2, 1.4, 0, 6.3, 'xk'],
  ['KNOWN_FLARE', 'industry', 'Matches a known flare', 'yes/no', 'VIIRS Nightfire', 0.05, 0, 0, 1, 'bk'],
  /* 7. Weather and fire danger (18) */
  ['TEMPERATURE_ANOMALY_Z', 'weather', 'Temperature against normal', 'z', 'ERA5 / forecast', 0, 1, -4, 4],
  ['RH_ANOMALY_Z', 'weather', 'Humidity against normal', 'z', 'ERA5 / forecast', 0, 1, -4, 4],
  ['VPD_ANOMALY_Z', 'weather', 'Air dryness against normal', 'z', 'ERA5 / forecast', 0, 1, -4, 4],
  ['WIND_SPEED', 'weather', 'Wind speed', 'm/s', 'ERA5 / forecast', 3.5, 1.8, 0, 22],
  ['WIND_GUST', 'weather', 'Wind gust', 'm/s', 'ERA5 / forecast', 6, 3, 0, 35],
  ['WIND_DIRECTION_SIN', 'weather', 'Wind direction (sine)', '', 'ERA5 / forecast', 0, 1, -1, 1],
  ['WIND_DIRECTION_COS', 'weather', 'Wind direction (cosine)', '', 'ERA5 / forecast', 0, 1, -1, 1],
  ['RAIN_24H_LOG', 'weather', 'Rain, last 24 hours', 'mm', 'ERA5 / forecast', 0.3, 0.6, 0, 5, 'x'],
  ['RAIN_7D_LOG', 'weather', 'Rain, last 7 days', 'mm', 'ERA5 / forecast', 1.4, 1.2, 0, 6, 'x'],
  ['SOIL_MOISTURE_ANOMALY_Z', 'weather', 'Soil moisture against normal', 'z', 'ERA5 / forecast', 0, 1, -4, 4],
  ['FFMC', 'weather', 'Fine fuel moisture code (FFMC)', '', 'Derived (FWI)', 80, 10, 30, 99],
  ['DMC', 'weather', 'Duff moisture code (DMC)', '', 'Derived (FWI)', 40, 30, 0, 250],
  ['DC', 'weather', 'Drought code (DC)', '', 'Derived (FWI)', 300, 200, 0, 800],
  ['ISI', 'weather', 'Initial spread index (ISI)', '', 'Derived (FWI)', 5, 4, 0, 40],
  ['BUI', 'weather', 'Build-up index (BUI)', '', 'Derived (FWI)', 60, 35, 0, 250],
  ['FWI', 'weather', 'Fire weather index (FWI)', '', 'Derived (FWI)', 10, 8, 0, 80],
  ['FWI_ANOMALY_Z', 'weather', 'Fire weather against normal', 'z', 'Derived (FWI)', 0, 1, -4, 4],
  ['LIGHTNING_COUNT_24H', 'weather', 'Lightning strikes, last 24 hours', 'count', 'Lightning network', 1, 3, 0, 200],
  /* 8. Smoke and burn signs (7) */
  ['SMOKE_PROBABILITY', 'smoke', 'Chance smoke is visible', 'prob', 'Optical (HLS)', 0.3, 0.3, 0, 1],
  ['SMOKE_LENGTH_LOG', 'smoke', 'Smoke plume length', 'km', 'Optical (HLS)', 1.5, 1, 0, 5, 'x'],
  ['SMOKE_DIRECTION_SIN', 'smoke', 'Smoke direction (sine)', '', 'Optical (HLS)', 0, 1, -1, 1],
  ['SMOKE_DIRECTION_COS', 'smoke', 'Smoke direction (cosine)', '', 'Optical (HLS)', 0, 1, -1, 1],
  ['DARK_SMOKE_PROBABILITY', 'smoke', 'Chance the smoke is dark', 'prob', 'Optical (HLS)', 0.25, 0.25, 0, 1],
  ['LIGHT_SMOKE_PROBABILITY', 'smoke', 'Chance the smoke is light', 'prob', 'Optical (HLS)', 0.3, 0.25, 0, 1],
  ['BURN_SCAR_GROWTH_RATE', 'smoke', 'Burn scar growth', 'km²/day', 'Optical (HLS)', 0.05, 0.1, 0, 3],
  /* 9. Radar and structure (7) */
  ['VV_CHANGE', 'sar', 'VV backscatter change', 'dB', 'Sentinel-1', 0, 1.5, -8, 8],
  ['VH_CHANGE', 'sar', 'VH backscatter change', 'dB', 'Sentinel-1', 0, 1.6, -8, 8],
  ['VV_VH_CHANGE', 'sar', 'VV to VH ratio change', 'dB', 'Sentinel-1', 0, 1.2, -6, 6],
  ['COHERENCE_LOSS', 'sar', 'Radar coherence loss', 'frac', 'Sentinel-1', 0.2, 0.2, 0, 1],
  ['DNBR', 'sar', 'Burn severity (dNBR)', '', 'Sentinel-2 change', 0.05, 0.2, -0.5, 1.2],
  ['DNDVI', 'sar', 'Greenness loss (dNDVI)', '', 'Sentinel-2 change', 0.04, 0.15, -0.4, 1],
  ['STRUCTURAL_CHANGE_PROBABILITY', 'sar', 'Chance of structural change', 'prob', 'Sentinel-1', 0.2, 0.2, 0, 1],
  /* 10. Air chemistry (9) */
  ['NO2_ANOMALY_Z', 'chem', 'Nitrogen dioxide against normal', 'z', 'TROPOMI', 0, 1, -4, 6],
  ['SO2_ANOMALY_Z', 'chem', 'Sulphur dioxide against normal', 'z', 'TROPOMI', 0, 1, -4, 6],
  ['CO_ANOMALY_Z', 'chem', 'Carbon monoxide against normal', 'z', 'TROPOMI', 0, 1, -4, 6],
  ['CH4_ANOMALY_Z', 'chem', 'Methane against normal', 'z', 'TROPOMI', 0, 1, -4, 6],
  ['HCHO_ANOMALY_Z', 'chem', 'Formaldehyde against normal', 'z', 'TROPOMI', 0, 1, -4, 6],
  ['AOD_ANOMALY_Z', 'chem', 'Haze (aerosol) against normal', 'z', 'Aerosol product', 0, 1, -4, 6],
  ['CO_NO2_RATIO_LOG', 'chem', 'CO to NO₂ ratio (log)', 'log ratio', 'TROPOMI', 0, 0.8, -3, 3],
  ['SO2_NO2_RATIO_LOG', 'chem', 'SO₂ to NO₂ ratio (log)', 'log ratio', 'TROPOMI', 0, 0.8, -3, 3],
  ['CH4_CO_RATIO_LOG', 'chem', 'CH₄ to CO ratio (log)', 'log ratio', 'TROPOMI', 0, 0.8, -3, 3],
  /* 11. Data quality (8) */
  ['FIRMS_CONFIDENCE', 'quality', 'FIRMS confidence', '%', 'FIRMS', 72, 15, 10, 100],
  ['PIXEL_SIZE', 'quality', 'Pixel size', 'km²', 'FIRMS', 0.2, 0.1, 0.1, 4.5],
  ['SCAN_ANGLE_SIN', 'quality', 'Scan angle (sine)', '', 'FIRMS', 0, 1, -1, 1],
  ['SCAN_ANGLE_COS', 'quality', 'Scan angle (cosine)', '', 'FIRMS', 0, 1, -1, 1],
  ['VIEW_ZENITH_SIN', 'quality', 'Viewing angle (sine)', '', 'FIRMS', 0, 1, -1, 1],
  ['VIEW_ZENITH_COS', 'quality', 'Viewing angle (cosine)', '', 'FIRMS', 0, 1, -1, 1],
  ['CLOUD_FRACTION', 'quality', 'Cloud cover', 'frac', 'FIRMS / HLS', 0.3, 0.3, 0, 1],
  ['MISSING_FEATURE_FRACTION', 'quality', 'Features missing', 'frac', 'Derived', 0.08, 0.08, 0, 1],
];

export const FDEF: FeatureDef[] = ROWS.map(([key, g, label, unit, src, mu, sd, lo, hi, fl = '']) => ({
  key, g, label, unit, src, mu, sd, lo, hi, fl,
}));

export const FBY: Record<string, FeatureDef> = Object.fromEntries(FDEF.map((d) => [d.key, d]));

export const FGROUP_COUNT: Partial<Record<FamilyId, number>> = FAMILIES.reduce(
  (o, f) => {
    o[f.id] = FDEF.filter((d) => d.g === f.id).length;
    return o;
  },
  {} as Partial<Record<FamilyId, number>>,
);

/** typical values per class, in the same units as the catalogue [mean, spread] */
const FCV: Record<ClassId, Record<string, [number, number]>> = {
  gas_flare: {
    FRP_MEDIAN: [14, 5], BT_I4_MEDIAN: [357, 4], BT_I4_I5_DIFF_MEDIAN: [56, 9], BT_I4_I5_DIFF_P90: [66, 9],
    SUBPIXEL_TEMP_MEDIAN: [1650, 260], SOURCE_AREA_MEDIAN: [40, 30], DURATION_HOURS_LOG: [6.3, 0.5],
    OVERPASS_DETECTION_RATIO: [0.92, 0.07], RECURRENCE_RATE_1Y: [0.93, 0.06], RECURRENCE_RATE_3Y: [0.93, 0.06],
    RECURRENCE_RATE_5Y: [0.92, 0.07], NIGHT_DAY_FRP_RATIO: [1.25, 0.25], STATIONARITY_INDEX: [0.97, 0.02],
    CENTROID_DISPLACEMENT_MEDIAN: [0.03, 0.02], SPREAD_SPEED_MEDIAN: [0.01, 0.01], AREA_GROWTH_RATE: [0, 0.03],
    FRP_AUTOCORRELATION: [0.82, 0.1], INSIDE_FACILITY: [0.9, 0], KNOWN_FLARE: [0.85, 0], DIST_WELLPAD_LOG: [0.8, 0.9],
    DIST_NEAREST_FACILITY_LOG: [0.6, 0.6], FOREST_FRAC_1KM: [0.03, 0.04], CROPLAND_FRAC_1KM: [0.12, 0.12],
    NDVI_MEDIAN: [0.2, 0.08], SMOKE_PROBABILITY: [0.12, 0.1], CH4_ANOMALY_Z: [1.4, 1],
  },
  industrial: {
    FRP_MEDIAN: [24, 12], BT_I4_MEDIAN: [349, 8], BT_I4_I5_DIFF_MEDIAN: [44, 10], SUBPIXEL_TEMP_MEDIAN: [1000, 250],
    DURATION_HOURS_LOG: [6, 0.7], OVERPASS_DETECTION_RATIO: [0.85, 0.12], RECURRENCE_RATE_1Y: [0.88, 0.1],
    RECURRENCE_RATE_3Y: [0.88, 0.1], RECURRENCE_RATE_5Y: [0.87, 0.1], STATIONARITY_INDEX: [0.94, 0.04],
    CENTROID_DISPLACEMENT_MEDIAN: [0.05, 0.04], SPREAD_SPEED_MEDIAN: [0.02, 0.02], INSIDE_FACILITY: [0.92, 0],
    KNOWN_FLARE: [0.08, 0], DIST_NEAREST_FACILITY_LOG: [0.5, 0.5], FACILITY_COUNT_1KM: [3.5, 2],
    FACILITY_COUNT_5KM: [8, 4], BUILTUP_FRAC_1KM: [0.3, 0.15], FOREST_FRAC_1KM: [0.04, 0.05],
    NDVI_MEDIAN: [0.22, 0.08], SO2_ANOMALY_Z: [0.8, 1],
  },
  mining: {
    FRP_MEDIAN: [11, 5], BT_I4_MEDIAN: [338, 9], DURATION_HOURS_LOG: [5.8, 0.8],
    OVERPASS_DETECTION_RATIO: [0.7, 0.2], RECURRENCE_RATE_1Y: [0.82, 0.12], RECURRENCE_RATE_3Y: [0.85, 0.1],
    RECURRENCE_RATE_5Y: [0.85, 0.1], STATIONARITY_INDEX: [0.8, 0.12], INSIDE_FACILITY: [0.55, 0],
    KNOWN_FLARE: [0.02, 0], DIST_MINE_LOG: [0.3, 0.4], DIST_NEAREST_FACILITY_LOG: [1.2, 0.8],
    BARE_MINING_FRAC_1KM: [0.55, 0.15], SLOPE_P90: [12, 5], TRI: [16, 8], COHERENCE_LOSS: [0.35, 0.2],
    SO2_ANOMALY_Z: [1, 1], CO_ANOMALY_Z: [0.9, 1],
  },
  wildfire: {
    FRP_MEDIAN: [12, 8], BT_I4_MEDIAN: [335, 10], DURATION_HOURS_LOG: [3.2, 0.9],
    OVERPASS_DETECTION_RATIO: [0.4, 0.2], RECURRENCE_RATE_1Y: [0.1, 0.1], RECURRENCE_RATE_3Y: [0.15, 0.12],
    RECURRENCE_RATE_5Y: [0.18, 0.12], STATIONARITY_INDEX: [0.25, 0.18], CENTROID_DISPLACEMENT_MEDIAN: [0.6, 0.4],
    SPREAD_SPEED_MEDIAN: [0.3, 0.2], SPREAD_SPEED_P90: [0.8, 0.5], AREA_GROWTH_RATE: [0.5, 0.5],
    ELONGATION: [2.2, 0.8], INSIDE_FACILITY: [0.02, 0], KNOWN_FLARE: [0.005, 0], DIST_NEAREST_FACILITY_LOG: [3.4, 0.7],
    FOREST_FRAC_375M: [0.7, 0.2], FOREST_FRAC_1KM: [0.7, 0.2], FOREST_FRAC_10KM: [0.6, 0.2],
    NDVI_MEDIAN: [0.62, 0.1], NDVI_ANOMALY_Z: [-0.6, 1], SLOPE_MEDIAN: [7, 4], SLOPE_P90: [15, 6],
    FWI: [18, 9], ISI: [9, 5], VPD_ANOMALY_Z: [0.8, 1], SMOKE_PROBABILITY: [0.55, 0.25],
    DARK_SMOKE_PROBABILITY: [0.4, 0.25], BURN_SCAR_GROWTH_RATE: [0.3, 0.3], DNBR: [0.3, 0.2],
  },
  agricultural_burning: {
    FRP_MEDIAN: [7, 4], BT_I4_MEDIAN: [332, 8], DURATION_HOURS_LOG: [2.6, 0.7],
    OVERPASS_DETECTION_RATIO: [0.35, 0.2], RECURRENCE_RATE_1Y: [0.35, 0.2], RECURRENCE_RATE_3Y: [0.5, 0.2],
    RECURRENCE_RATE_5Y: [0.55, 0.2], STATIONARITY_INDEX: [0.4, 0.2], CENTROID_DISPLACEMENT_MEDIAN: [0.25, 0.2],
    SPREAD_SPEED_MEDIAN: [0.12, 0.08], INSIDE_FACILITY: [0.01, 0], KNOWN_FLARE: [0.003, 0],
    DIST_NEAREST_FACILITY_LOG: [3, 0.7], CROPLAND_FRAC_375M: [0.85, 0.1], CROPLAND_FRAC_1KM: [0.82, 0.1],
    CROPLAND_FRAC_10KM: [0.75, 0.12], FOREST_FRAC_1KM: [0.03, 0.04], NDVI_MEDIAN: [0.28, 0.1],
    NBR_CHANGE: [-0.12, 0.1], SLOPE_MEDIAN: [1, 0.8], AOD_ANOMALY_Z: [1.6, 1], CO_ANOMALY_Z: [1.1, 1],
    CO_NO2_RATIO_LOG: [0.7, 0.6], SMOKE_PROBABILITY: [0.5, 0.25],
  },
  unknown: {
    FRP_MEDIAN: [4, 2.5], BT_I4_MEDIAN: [326, 7], DURATION_HOURS_LOG: [5, 1],
    OVERPASS_DETECTION_RATIO: [0.5, 0.25], RECURRENCE_RATE_1Y: [0.5, 0.3], STATIONARITY_INDEX: [0.85, 0.12],
    INSIDE_FACILITY: [0.3, 0], KNOWN_FLARE: [0.02, 0], DIST_LANDFILL_LOG: [0.5, 0.6],
    FIRMS_CONFIDENCE: [50, 12], SMOKE_PROBABILITY: [0.15, 0.15],
  },
};

/** which inputs push each class up (sample contribution weights) */
const FSHAP: Record<ClassId, Record<string, number>> = {
  gas_flare: {
    KNOWN_FLARE: 0.3, RECURRENCE_RATE_1Y: 0.28, INSIDE_FACILITY: 0.22, BT_I4_I5_DIFF_MEDIAN: 0.22,
    SUBPIXEL_TEMP_MEDIAN: 0.2, OVERPASS_DETECTION_RATIO: 0.18, STATIONARITY_INDEX: 0.16,
    DIST_NEAREST_FACILITY_LOG: 0.14, DIST_WELLPAD_LOG: 0.12, CH4_ANOMALY_Z: 0.07,
    FOREST_FRAC_1KM: -0.05, NIGHT_DAY_FRP_RATIO: 0.08,
  },
  industrial: {
    INSIDE_FACILITY: 0.3, DIST_NEAREST_FACILITY_LOG: 0.24, RECURRENCE_RATE_3Y: 0.2, BUILTUP_FRAC_1KM: 0.15,
    FACILITY_COUNT_1KM: 0.14, STATIONARITY_INDEX: 0.12, FRP_PERSISTENCE: 0.1, SO2_ANOMALY_Z: 0.06,
    NDVI_MEDIAN: 0.05, KNOWN_FLARE: -0.06,
  },
  mining: {
    DIST_MINE_LOG: 0.32, BARE_MINING_FRAC_1KM: 0.26, RECURRENCE_RATE_3Y: 0.2, COHERENCE_LOSS: 0.09,
    SLOPE_P90: 0.08, TRI: 0.07, SO2_ANOMALY_Z: 0.06, CO_ANOMALY_Z: 0.05, INSIDE_FACILITY: -0.05,
    FOREST_FRAC_1KM: -0.06,
  },
  wildfire: {
    FOREST_FRAC_1KM: 0.3, NDVI_MEDIAN: 0.18, SPREAD_SPEED_MEDIAN: 0.16, FWI: 0.16, AREA_GROWTH_RATE: 0.14,
    DURATION_HOURS_LOG: 0.12, SMOKE_PROBABILITY: 0.12, VPD_ANOMALY_Z: 0.1, RECURRENCE_RATE_1Y: 0.1,
    DIST_NEAREST_FACILITY_LOG: 0.1, DNBR: 0.08, INSIDE_FACILITY: -0.05,
  },
  agricultural_burning: {
    CROPLAND_FRAC_1KM: 0.34, DURATION_HOURS_LOG: 0.14, NBR_CHANGE: 0.12, RECURRENCE_RATE_3Y: 0.12,
    DIST_NEAREST_FACILITY_LOG: 0.1, AOD_ANOMALY_Z: 0.08, CO_NO2_RATIO_LOG: 0.06, STATIONARITY_INDEX: 0.06,
    FOREST_FRAC_1KM: -0.07, INSIDE_FACILITY: -0.05,
  },
  unknown: {
    DIST_LANDFILL_LOG: 0.18, STATIONARITY_INDEX: 0.1, INSIDE_FACILITY: 0.06, FIRMS_CONFIDENCE: -0.12,
    BT_I4_MEDIAN: -0.14, FRP_MEDIAN: -0.1, CLOUD_FRACTION: -0.08, MISSING_FEATURE_FRACTION: -0.06,
  },
};

/* ---- formatting ---- */
const normCdf = (z: number): number => {
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = 0.3989423 * Math.exp((-z * z) / 2);
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return z > 0 ? 1 - p : p;
};

export function fmtNum(v: number): string {
  const a = Math.abs(v);
  if (a >= 1000) return Math.round(v).toLocaleString('en-IN');
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(3).replace(/0+$/, '').replace(/\.$/, '') || '0';
}

export function fmtFeat(d: FeatureDef, v: number | null): string {
  if (v == null || isNaN(v)) return '—';
  if (d.fl.includes('b')) return v ? 'Yes' : 'No';
  if (d.unit === 'frac' || d.unit === 'prob') return Math.round(v * 100) + '%';
  if (d.unit === 'z') return (v >= 0 ? '+' : '') + v.toFixed(1) + ' σ';
  if (d.unit === 'count') return String(Math.round(v));
  if (d.unit === '%') return Math.round(v) + '%';
  return fmtNum(v) + (d.unit ? ' ' + d.unit : '');
}

export const isAngle = (k: string): boolean => /_(SIN|COS)$/.test(k);
export const angleName = (k: string): string => k.replace(/_(SIN|COS)$/, '');

/* ---- the full feature vector for one event (sample) ---- */
export function featuresOf(p: Place, ev: Event): FeatureVector {
  if (ev._f) return ev._f;
  const r = R('feat:' + ev.id);
  const ctx = contextOf(p);
  const W = weatherOf(p, ev);
  const c = ev.cls;
  const ov = FCV[c] || {};
  const out: FeatureVector['by'] = {};
  const ang: Record<string, number> = {};

  ang.WIND_DIRECTION = W.dir[0];
  ang.SPREAD_DIRECTION = (W.dir[0] + 180 + r.range(-30, 30) + 360) % 360;
  ang.SMOKE_DIRECTION = (W.dir[0] + 180 + r.range(-18, 18) + 360) % 360;
  ang.PEAK_HOUR = (ev.hr / 24) * 360;
  ang.SCAN_ANGLE = r.range(-52, 52);
  ang.VIEW_ZENITH = Math.abs(ang.SCAN_ANGLE) * 1.12 + r.range(0, 4);

  const put = (d: FeatureDef, raw: number) => {
    const shown = d.fl.includes('x') ? Math.exp(raw) - 1 : raw;
    const o = ov[d.key];
    const z = d.sd ? (raw - (o ? o[0] : d.mu)) / (o ? o[1] : d.sd || 1) : 0;
    out[d.key] = {
      d,
      raw,
      v: shown,
      pct: d.fl.includes('b') || isAngle(d.key) ? null : normCdf(z) * 100,
      missing: false,
    };
  };

  FDEF.forEach((d) => {
    if (isAngle(d.key)) {
      const n = angleName(d.key);
      const a = (ang[n] != null ? ang[n] : (ang[n] = r.range(0, 360))) * (Math.PI / 180);
      const v = /_SIN$/.test(d.key) ? Math.sin(a) : Math.cos(a);
      out[d.key] = { d, raw: v, v, pct: null, missing: false };
      return;
    }
    const o = ov[d.key];
    const mu = o ? o[0] : d.mu;
    const sd = o ? o[1] : d.sd;
    put(d, d.fl.includes('b') ? (r() < mu ? 1 : 0) : clamp(mu + sd * r.norm(), d.lo, d.hi));
  });

  const set = (k: string, shown: number) => {
    const d = FBY[k];
    put(d, d.fl.includes('x') ? Math.log(1 + Math.max(0, shown)) : shown);
  };

  /* tie the vector to what the rest of the app already shows for this event */
  const pt = ctx.parts;
  const pc = (n: string) => (pt.find((x) => x[0] === n) || [0, 0])[1] / 100;
  const jit = (v: number, s: number) => clamp(v + r.norm() * s, 0, 1);

  set('FRP_MAX', ev.frp * 1.15);
  set('FRP_MEDIAN', ev.frp * r.range(0.5, 0.75));
  set('FRP_P90', ev.frp * r.range(0.85, 1));
  set('FRP_SUM_LOG', Math.log(ev.frp * ev.nDet * 1.2 + 1));
  set('DETECTION_COUNT_LOG', ev.nDet);

  set('FOREST_FRAC_1KM', pc('Forest'));
  set('FOREST_FRAC_375M', jit(pc('Forest'), 0.06));
  set('FOREST_FRAC_10KM', clamp(pc('Forest') * 0.7 + 0.1 + r.norm() * 0.05, 0, 1));
  set('FOREST_LOCAL_REGIONAL_CONTRAST', pc('Forest') - 0.18);
  set('CROPLAND_FRAC_1KM', pc('Cropland'));
  set('CROPLAND_FRAC_10KM', clamp(pc('Cropland') * 0.8 + 0.05 + r.norm() * 0.05, 0, 1));
  set('CROPLAND_LOCAL_REGIONAL_CONTRAST', pc('Cropland') - 0.3);
  set('BUILTUP_FRAC_1KM', pc('Industrial or built-up'));
  set('BUILTUP_FRAC_10KM', clamp(pc('Industrial or built-up') * 0.5 + 0.04, 0, 1));
  set('BARE_MINING_FRAC_1KM', pc('Bare ground'));
  set('WATER_FRAC_1KM', pc('Water and other') * 0.5);

  const ent = -pt.reduce((a, x) => a + (x[1] > 0 ? (x[1] / 100) * Math.log2(x[1] / 100) : 0), 0);
  set('LANDCOVER_ENTROPY', ent);

  set('NDVI_MEDIAN', ctx.ndvi);
  set('NBR_CHANGE', c === 'wildfire' ? -(ctx.nbr * 0.5) : -0.04 * r.range(0, 2));
  set('ELEVATION_MEDIAN', ctx.elev);
  set('SLOPE_MEDIAN', ctx.slope);
  set('SLOPE_P90', ctx.slope * 2.1 + 1);

  const dk = ctx.d;
  set('DIST_REFINERY_LOG', dk.refinery);
  set('DIST_POWERPLANT_LOG', dk.power);
  set('DIST_STEEL_LOG', dk.steel);
  set('DIST_CEMENT_LOG', dk.steel * r.range(0.8, 1.6));
  set('DIST_MINE_LOG', dk.mine);
  set('DIST_LANDFILL_LOG', dk.landfill);
  set('DIST_PETROCHEMICAL_LOG', /Petro/.test(p.type) ? r.range(0, 0.8) : dk.refinery * r.range(1, 2));
  set('DIST_LNG_LOG', /LNG/.test(p.type) ? r.range(0, 0.6) : r.range(30, 300));
  const near = Math.min(
    dk.refinery, dk.power, dk.steel, dk.mine, dk.landfill,
    p.kind === 'site' && p.cls !== 'unknown' ? r.range(0, 1.2) : 99,
  );
  set('DIST_NEAREST_FACILITY_LOG', near > 90 ? r.range(6, 60) : near);

  /* weather: take the first forecast hour so the vector agrees with the wind map */
  set('WIND_SPEED', W.spd[0]);
  set('WIND_GUST', W.gust[0]);
  set('FFMC', W.ffmc[0]);
  set('DMC', W.dmc);
  set('DC', W.dc);
  const isiV = calcISI(W.ffmc[0], W.spd[0] * 3.6);
  const buiV = calcBUI(W.dmc, W.dc);
  const fwiV = calcFWI(isiV, buiV);
  set('ISI', isiV);
  set('BUI', buiV);
  set('FWI', fwiV);
  set('FWI_ANOMALY_Z', clamp((fwiV - 10) / 8, -3, 4));
  set('VPD_ANOMALY_Z', clamp((W.vpd[0] - 2.2) / 1.1, -3, 4));
  set('RH_ANOMALY_Z', clamp(-(W.vpd[0] - 2.2) / 1.6 + r.norm() * 0.3, -3, 3));
  set('RAIN_24H_LOG', W.rainPast24);
  set('RAIN_7D_LOG', W.rainPast24 * 2.4 + r.range(0, 6));
  set('SPREAD_WIND_SPEED_RATIO', (out.SPREAD_SPEED_MEDIAN.v! * 1000) / 3600 / Math.max(0.5, W.spd[0]));
  set('CLOUD_FRACTION', clamp(1 - ctx.valid + r.norm() * 0.05, 0, 1));

  const sc = Math.abs(ang.SCAN_ANGLE);
  const mod = ev.sensor.startsWith('MODIS');
  set('PIXEL_SIZE', (mod ? 1 : 0.14) * (1 + Math.pow(sc / 55, 2) * 2.2));
  set('FIRMS_CONFIDENCE', mod ? clamp(62 + r.norm() * 14, 10, 100) : clamp((ov.FIRMS_CONFIDENCE ? ov.FIRMS_CONFIDENCE[0] : 78) + r.norm() * 10, 10, 100));
  set('DETECTION_DENSITY', ev.nDet / Math.max(ev.area, 0.05));

  /* missing data: cloud hides optical inputs, radar and gas satellites pass rarely */
  const cloudy = ctx.valid < 0.38;
  FDEF.forEach((d) => {
    const pm =
      d.g === 'sar' ? 0.28 : d.g === 'chem' ? 0.22 : d.g === 'smoke' ? (cloudy ? 0.55 : 0.12)
      : d.src === 'HLS' ? (cloudy ? 0.7 : 0.04) : 0.015;
    if (out[d.key] && d.key !== 'MISSING_FEATURE_FRACTION' && r() < pm) {
      out[d.key].missing = true;
      out[d.key].v = null;
      out[d.key].pct = null;
    }
  });
  const miss = FDEF.filter((d) => out[d.key].missing).length;
  set('MISSING_FEATURE_FRACTION', miss / FDEF.length);

  return (ev._f = { by: out, missing: miss, W, ang });
}

/* ---- sample explanation: which features pushed the call, and by how much ---- */
export function shapOf(ev: Event): ShapResult {
  if (ev._shap) return ev._shap;
  const p = PMAP[ev.pid];
  const f = featuresOf(p, ev);
  const r = R('shap:' + ev.id);
  const Wt = FSHAP[ev.cls] || {};
  const all: Record<string, number> = {};
  const fam = {} as ShapResult['fam'];
  FAMILIES.forEach((g) => (fam[g.id] = 0));
  FDEF.forEach((d) => {
    const e = f.by[d.key];
    let w = 0;
    if (!e.missing) {
      w = Wt[d.key] != null ? Wt[d.key] * r.range(0.8, 1.25) * (0.6 + ev.conf * 0.5) : r.norm() * 0.012;
    }
    all[d.key] = w;
    fam[d.g] += w;
  });
  const feats = FDEF.filter((d) => Wt[d.key] != null && !f.by[d.key].missing)
    .map((d) => ({ k: d.key, l: d.label, f: d.g, w: all[d.key], v: fmtFeat(d, f.by[d.key].v) }))
    .sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
  return (ev._shap = { feats: feats.slice(0, 8), fam, all });
}

/* the last eight observed footprints for an event */
export function footprintsFor(p: Place, ev: Event): FootprintFrame[] {
  const f = featuresOf(p, ev);
  return footprintFrames(p, ev, f.ang.SPREAD_DIRECTION, f.by.ELONGATION.v ?? 1.6);
}

function footprintFrames(p: Place, ev: Event, dirTo: number, elong: number): FootprintFrame[] {
  const r = R('fp8:' + ev.id);
  const grow = isVeg(ev.cls) || ev.status === 'abnormal';
  const el = grow ? clamp(elong, 1.2, 3) : 1.15;
  const frames: FootprintFrame[] = [];
  const n = 8;
  const ph = r.range(0, 6);
  const ph2 = r.range(0, 6);
  const v = [Math.sin(rad(dirTo)), Math.cos(rad(dirTo))];
  const q = [Math.cos(rad(dirTo)), -Math.sin(rad(dirTo))];
  for (let i = 0; i < n; i++) {
    const k = grow ? 0.22 + (0.78 * i) / (n - 1) : 0.92 + 0.08 * r();
    const area = Math.max(0.02, ev.area * k);
    const b = Math.sqrt(area / (Math.PI * el));
    const a = b * el;
    const shift = grow ? (i - (n - 1)) * Math.sqrt(area) * 0.18 : 0;
    const ring: LonLat[] = [];
    for (let s = 0; s < 40; s++) {
      const th = (s / 40) * 2 * Math.PI;
      const m = 1 + 0.16 * Math.sin(3 * th + ph) + 0.09 * Math.sin(5 * th + ph2);
      const X = shift + a * m * Math.cos(th);
      const Y = b * m * Math.sin(th);
      ring.push(offsetLL(p.lon, p.lat, (X * v[0] + Y * q[0]) * 1000, (X * v[1] + Y * q[1]) * 1000));
    }
    frames.push({ t: ev.t - (n - 1 - i) * 12 * HOUR, area, ring });
  }
  return frames;
}

export const FSRC_TIP =
  'These inputs come from the same sources used to make the training labels, so the model can partly read its own answer back. Treat their contribution with care.';
