'use strict';
/* =====================================================================
   The 141 event-level features (X_event), grouped as in the model spec.
   Everything here is SAMPLE data. Replace featuresOf() with a call that
   returns the real feature vector for an event (see docs/api-contract.md).
   ===================================================================== */
const FDEF = [];
/* flags: x = model sees log, shown in real units | k = also used to build labels | b = yes/no | (sin/cos pairs are detected by name) */
function F(key, g, label, unit, src, mu, sd, lo, hi, fl) { FDEF.push({ key, g, label, unit, src, mu, sd, lo, hi, fl: fl || '' }); }

/* 1. Heat (15) */
F('FRP_MEDIAN', 'thermal', 'Median fire power', 'MW', 'FIRMS', 12, 8, .5, 400);
F('FRP_P90', 'thermal', 'Fire power, top 10% of passes', 'MW', 'FIRMS', 22, 14, 1, 600);
F('FRP_MAX', 'thermal', 'Peak fire power', 'MW', 'FIRMS', 30, 20, 1, 900);
F('FRP_SUM_LOG', 'thermal', 'Total heat released (log)', 'log MW', 'FIRMS', 4.6, 1.1, .5, 9);
F('FRP_IQR', 'thermal', 'Fire power spread (IQR)', 'MW', 'FIRMS', 9, 7, 0, 300);
F('FRP_CV', 'thermal', 'Fire power variability', 'ratio', 'FIRMS', .5, .3, 0, 3);
F('FRP_PEAK_MEDIAN_RATIO', 'thermal', 'Peak against median', 'ratio', 'FIRMS', 2.1, .9, 1, 12);
F('FRP_SLOPE', 'thermal', 'Fire power trend', 'MW/h', 'Derived', 0, 1.2, -20, 20);
F('FRP_DECAY_RATE', 'thermal', 'Fire power decay rate', '/h', 'Derived', .06, .05, 0, 1);
F('BT_I4_MEDIAN', 'thermal', 'Brightness, 4 µm band', 'K', 'VIIRS I4', 342, 14, 300, 368);
F('BT_I5_MEDIAN', 'thermal', 'Brightness, 11 µm band', 'K', 'VIIRS I5', 304, 8, 280, 340);
F('BT_I4_I5_DIFF_MEDIAN', 'thermal', 'Band gap (4 µm minus 11 µm)', 'K', 'VIIRS I4/I5', 38, 14, 2, 90);
F('BT_I4_I5_DIFF_P90', 'thermal', 'Band gap, top 10%', 'K', 'VIIRS I4/I5', 52, 16, 5, 110);
F('SUBPIXEL_TEMP_MEDIAN', 'thermal', 'Estimated temperature of the hot part', 'K', 'Derived', 900, 320, 500, 2300);
F('SOURCE_AREA_MEDIAN', 'thermal', 'Estimated hot area inside a pixel', 'm²', 'Derived', 2500, 2600, 10, 60000);
/* 2. Timing (14) */
F('DURATION_HOURS_LOG', 'temporal', 'How long it has burned', 'h', 'Derived', 3.4, 1.2, .7, 8, 'x');
F('DETECTION_COUNT_LOG', 'temporal', 'Detections in this event', 'count', 'Derived', 3.3, 1, .7, 7, 'x');
F('OVERPASS_DETECTION_RATIO', 'temporal', 'Share of passes that saw it', 'frac', 'Derived', .6, .25, .02, 1);
F('DETECTION_DENSITY', 'temporal', 'Detections per km²', '/km²', 'Derived', 6, 4, .2, 60);
F('RECURRENCE_RATE_1Y', 'temporal', 'Seen again within 1 year', 'frac', 'Derived', .3, .3, 0, 1, 'k');
F('RECURRENCE_RATE_3Y', 'temporal', 'Seen again within 3 years', 'frac', 'Derived', .3, .3, 0, 1, 'k');
F('RECURRENCE_RATE_5Y', 'temporal', 'Seen again within 5 years', 'frac', 'Derived', .3, .3, 0, 1, 'k');
F('MAX_INTER_EVENT_GAP_LOG', 'temporal', 'Longest quiet gap', 'days', 'Derived', 3.2, 1.3, 0, 7, 'x');
F('FRP_AUTOCORRELATION', 'temporal', 'Pass-to-pass steadiness', 'r', 'Derived', .5, .3, -.6, 1);
F('TIME_TO_PEAK', 'temporal', 'Time to reach peak', 'h', 'Derived', 8, 8, 0, 72);
F('FRP_PERSISTENCE', 'temporal', 'Fire power persistence', 'frac', 'Derived', .5, .25, 0, 1);
F('NIGHT_DAY_FRP_RATIO', 'temporal', 'Night against day fire power', 'ratio', 'Derived', 1, .5, .1, 5);
F('PEAK_HOUR_SIN', 'temporal', 'Peak hour (sine)', '', 'Derived', 0, 1, -1, 1);
F('PEAK_HOUR_COS', 'temporal', 'Peak hour (cosine)', '', 'Derived', 0, 1, -1, 1);
/* 3. Movement and growth (17) */
F('CENTROID_DISPLACEMENT_MEDIAN', 'spread', 'Typical move between passes', 'km', 'Derived', .3, .3, 0, 12);
F('CENTROID_DISPLACEMENT_P90', 'spread', 'Biggest moves between passes', 'km', 'Derived', .8, .7, 0, 25);
F('SPREAD_SPEED_MEDIAN', 'spread', 'Typical spread speed', 'km/h', 'Derived', .15, .15, 0, 6);
F('SPREAD_SPEED_P90', 'spread', 'Fast spread speed', 'km/h', 'Derived', .4, .4, 0, 8);
F('SPREAD_SPEED_MAX', 'spread', 'Fastest spread speed', 'km/h', 'Derived', .7, .6, 0, 10);
F('CENTROID_JITTER', 'spread', 'Centre wobble', 'km', 'Derived', .15, .1, 0, 3);
F('STATIONARITY_INDEX', 'spread', 'Stays in one place', 'frac', 'Derived', .6, .3, 0, 1);
F('SPREAD_DIRECTION_SIN', 'spread', 'Spread direction (sine)', '', 'Derived', 0, 1, -1, 1);
F('SPREAD_DIRECTION_COS', 'spread', 'Spread direction (cosine)', '', 'Derived', 0, 1, -1, 1);
F('DIRECTION_CONSISTENCY', 'spread', 'Keeps one direction', 'frac', 'Derived', .5, .3, 0, 1);
F('AREA_GROWTH_RATE', 'spread', 'Area growth', 'km²/h', 'Derived', .1, .2, -1, 12);
F('PERIMETER_GROWTH_RATE', 'spread', 'Edge growth', 'km/h', 'Derived', .2, .3, -1, 15);
F('ELONGATION', 'spread', 'Elongation', 'ratio', 'Derived', 1.6, .6, 1, 8);
F('COMPACTNESS', 'spread', 'Compactness', 'frac', 'Derived', .6, .2, .1, 1);
F('SOLIDITY', 'spread', 'Solidity', 'frac', 'Derived', .85, .1, .3, 1);
F('NEW_PIXEL_FRACTION', 'spread', 'New pixels each pass', 'frac', 'Derived', .2, .2, 0, 1);
F('SPREAD_WIND_SPEED_RATIO', 'spread', 'Spread speed against wind speed', 'ratio', 'Derived', .03, .03, 0, .4);
/* 4. Terrain (12) */
F('ELEVATION_MEDIAN', 'terrain', 'Elevation', 'm', 'DEM', 300, 250, 0, 4500);
F('ELEVATION_IQR', 'terrain', 'Elevation spread (IQR)', 'm', 'DEM', 20, 25, 0, 400);
F('ELEVATION_RANGE', 'terrain', 'Elevation range', 'm', 'DEM', 60, 70, 0, 1200);
F('RELATIVE_ELEVATION', 'terrain', 'Height above surroundings', 'm', 'DEM', 0, 25, -150, 150);
F('SLOPE_MEDIAN', 'terrain', 'Slope', '°', 'DEM', 3, 3, 0, 40);
F('SLOPE_P90', 'terrain', 'Steep side slope', '°', 'DEM', 7, 5, 0, 55);
F('ASPECT_SIN', 'terrain', 'Slope faces (sine)', '', 'DEM', 0, 1, -1, 1);
F('ASPECT_COS', 'terrain', 'Slope faces (cosine)', '', 'DEM', 0, 1, -1, 1);
F('TPI', 'terrain', 'Topographic position', '', 'DEM', 0, 8, -40, 40);
F('TRI', 'terrain', 'Ruggedness', 'm', 'DEM', 8, 8, 0, 150);
F('CURVATURE', 'terrain', 'Curvature', '', 'DEM', 0, .5, -3, 3);
F('RIDGE_VALLEY_POSITION', 'terrain', 'Ridge or valley (−1 valley, +1 ridge)', '', 'DEM', 0, .5, -1, 1);
/* 5. Fuel and land cover (16) */
F('FOREST_FRAC_375M', 'fuel', 'Forest within 375 m', 'frac', 'WorldCover', .15, .25, 0, 1, 'k');
F('FOREST_FRAC_1KM', 'fuel', 'Forest within 1 km', 'frac', 'WorldCover', .15, .25, 0, 1, 'k');
F('FOREST_FRAC_10KM', 'fuel', 'Forest within 10 km', 'frac', 'WorldCover', .15, .2, 0, 1, 'k');
F('FOREST_LOCAL_REGIONAL_CONTRAST', 'fuel', 'Forest here against the region', '', 'WorldCover', 0, .3, -1, 1, 'k');
F('CROPLAND_FRAC_1KM', 'fuel', 'Cropland within 1 km', 'frac', 'WorldCover', .25, .3, 0, 1, 'k');
F('CROPLAND_FRAC_10KM', 'fuel', 'Cropland within 10 km', 'frac', 'WorldCover', .25, .25, 0, 1, 'k');
F('CROPLAND_LOCAL_REGIONAL_CONTRAST', 'fuel', 'Cropland here against the region', '', 'WorldCover', 0, .3, -1, 1, 'k');
F('BUILTUP_FRAC_1KM', 'fuel', 'Built-up land within 1 km', 'frac', 'WorldCover', .1, .2, 0, 1, 'k');
F('BUILTUP_FRAC_10KM', 'fuel', 'Built-up land within 10 km', 'frac', 'WorldCover', .1, .15, 0, 1, 'k');
F('BARE_MINING_FRAC_1KM', 'fuel', 'Bare or mined ground within 1 km', 'frac', 'WorldCover', .1, .2, 0, 1, 'k');
F('WATER_FRAC_1KM', 'fuel', 'Water within 1 km', 'frac', 'WorldCover', .04, .08, 0, 1);
F('LANDCOVER_ENTROPY', 'fuel', 'Land cover mix', 'bits', 'WorldCover', 1.1, .5, 0, 2.6);
F('NDVI_MEDIAN', 'fuel', 'Greenness (NDVI)', '', 'HLS', .4, .2, -.1, .9);
F('NDVI_ANOMALY_Z', 'fuel', 'Greenness against normal', 'z', 'HLS', 0, 1, -4, 4);
F('NBR_CHANGE', 'fuel', 'Burn ratio change', '', 'HLS', -.05, .15, -.8, .4);
F('NDMI_ANOMALY_Z', 'fuel', 'Leaf moisture against normal', 'z', 'HLS', 0, 1, -4, 4);
/* 6. Industry nearby (18) */
F('DIST_NEAREST_FACILITY_LOG', 'industry', 'Nearest facility', 'km', 'GEM / OSM', 2.2, 1.4, 0, 6.3, 'xk');
F('DIST_REFINERY_LOG', 'industry', 'Nearest refinery', 'km', 'GEM / OSM', 3.4, 1.3, 0, 6.3, 'xk');
F('DIST_POWERPLANT_LOG', 'industry', 'Nearest power plant', 'km', 'GEM / OSM', 3.2, 1.3, 0, 6.3, 'xk');
F('DIST_PETROCHEMICAL_LOG', 'industry', 'Nearest petrochemical plant', 'km', 'GEM / OSM', 3.5, 1.3, 0, 6.3, 'xk');
F('DIST_STEEL_LOG', 'industry', 'Nearest steel plant', 'km', 'GEM / OSM', 3.4, 1.3, 0, 6.3, 'xk');
F('DIST_CEMENT_LOG', 'industry', 'Nearest cement plant', 'km', 'GEM / OSM', 3.3, 1.3, 0, 6.3, 'xk');
F('DIST_LNG_LOG', 'industry', 'Nearest LNG terminal', 'km', 'GEM / OSM', 4.2, 1.2, 0, 6.3, 'xk');
F('DIST_MINE_LOG', 'industry', 'Nearest mine', 'km', 'GEM / OSM', 3.1, 1.3, 0, 6.3, 'xk');
F('DIST_LANDFILL_LOG', 'industry', 'Nearest landfill', 'km', 'OSM', 3.3, 1.2, 0, 6.3, 'xk');
F('DIST_PIPELINE_LOG', 'industry', 'Nearest pipeline', 'km', 'GEM / OSM', 3, 1.4, 0, 6.3, 'xk');
F('DIST_WELLPAD_LOG', 'industry', 'Nearest well pad', 'km', 'GEM / OSM', 3.6, 1.3, 0, 6.3, 'xk');
F('FACILITY_COUNT_1KM', 'industry', 'Facilities within 1 km', 'count', 'GEM / OSM', .6, 1.1, 0, 25, 'k');
F('FACILITY_COUNT_5KM', 'industry', 'Facilities within 5 km', 'count', 'GEM / OSM', 2, 3, 0, 60, 'k');
F('FACILITY_COUNT_10KM', 'industry', 'Facilities within 10 km', 'count', 'GEM / OSM', 4, 6, 0, 120, 'k');
F('INDUSTRIAL_DENSITY_CONTRAST', 'industry', 'Industry here against the region', '', 'GEM / OSM', 0, 1, -3, 5, 'k');
F('INSIDE_FACILITY', 'industry', 'Inside a facility boundary', 'yes/no', 'GEM / OSM', .15, 0, 0, 1, 'bk');
F('DIST_FACILITY_BOUNDARY_LOG', 'industry', 'Distance to a facility boundary', 'km', 'GEM / OSM', 2, 1.4, 0, 6.3, 'xk');
F('KNOWN_FLARE', 'industry', 'Matches a known flare', 'yes/no', 'VIIRS Nightfire', .05, 0, 0, 1, 'bk');
/* 7. Weather and fire danger (18) */
F('TEMPERATURE_ANOMALY_Z', 'weather', 'Temperature against normal', 'z', 'ERA5 / forecast', 0, 1, -4, 4);
F('RH_ANOMALY_Z', 'weather', 'Humidity against normal', 'z', 'ERA5 / forecast', 0, 1, -4, 4);
F('VPD_ANOMALY_Z', 'weather', 'Air dryness against normal', 'z', 'ERA5 / forecast', 0, 1, -4, 4);
F('WIND_SPEED', 'weather', 'Wind speed', 'm/s', 'ERA5 / forecast', 3.5, 1.8, 0, 22);
F('WIND_GUST', 'weather', 'Wind gust', 'm/s', 'ERA5 / forecast', 6, 3, 0, 35);
F('WIND_DIRECTION_SIN', 'weather', 'Wind direction (sine)', '', 'ERA5 / forecast', 0, 1, -1, 1);
F('WIND_DIRECTION_COS', 'weather', 'Wind direction (cosine)', '', 'ERA5 / forecast', 0, 1, -1, 1);
F('RAIN_24H_LOG', 'weather', 'Rain, last 24 hours', 'mm', 'ERA5 / forecast', .3, .6, 0, 5, 'x');
F('RAIN_7D_LOG', 'weather', 'Rain, last 7 days', 'mm', 'ERA5 / forecast', 1.4, 1.2, 0, 6, 'x');
F('SOIL_MOISTURE_ANOMALY_Z', 'weather', 'Soil moisture against normal', 'z', 'ERA5 / forecast', 0, 1, -4, 4);
F('FFMC', 'weather', 'Fine fuel moisture code (FFMC)', '', 'Derived (FWI)', 80, 10, 30, 99);
F('DMC', 'weather', 'Duff moisture code (DMC)', '', 'Derived (FWI)', 40, 30, 0, 250);
F('DC', 'weather', 'Drought code (DC)', '', 'Derived (FWI)', 300, 200, 0, 800);
F('ISI', 'weather', 'Initial spread index (ISI)', '', 'Derived (FWI)', 5, 4, 0, 40);
F('BUI', 'weather', 'Build-up index (BUI)', '', 'Derived (FWI)', 60, 35, 0, 250);
F('FWI', 'weather', 'Fire weather index (FWI)', '', 'Derived (FWI)', 10, 8, 0, 80);
F('FWI_ANOMALY_Z', 'weather', 'Fire weather against normal', 'z', 'Derived (FWI)', 0, 1, -4, 4);
F('LIGHTNING_COUNT_24H', 'weather', 'Lightning strikes, last 24 hours', 'count', 'Lightning network', 1, 3, 0, 200);
/* 8. Smoke and burn signs (7) */
F('SMOKE_PROBABILITY', 'smoke', 'Chance smoke is visible', 'prob', 'Optical (HLS)', .3, .3, 0, 1);
F('SMOKE_LENGTH_LOG', 'smoke', 'Smoke plume length', 'km', 'Optical (HLS)', 1.5, 1, 0, 5, 'x');
F('SMOKE_DIRECTION_SIN', 'smoke', 'Smoke direction (sine)', '', 'Optical (HLS)', 0, 1, -1, 1);
F('SMOKE_DIRECTION_COS', 'smoke', 'Smoke direction (cosine)', '', 'Optical (HLS)', 0, 1, -1, 1);
F('DARK_SMOKE_PROBABILITY', 'smoke', 'Chance the smoke is dark', 'prob', 'Optical (HLS)', .25, .25, 0, 1);
F('LIGHT_SMOKE_PROBABILITY', 'smoke', 'Chance the smoke is light', 'prob', 'Optical (HLS)', .3, .25, 0, 1);
F('BURN_SCAR_GROWTH_RATE', 'smoke', 'Burn scar growth', 'km²/day', 'Optical (HLS)', .05, .1, 0, 3);
/* 9. Radar and structure (7) */
F('VV_CHANGE', 'sar', 'VV backscatter change', 'dB', 'Sentinel-1', 0, 1.5, -8, 8);
F('VH_CHANGE', 'sar', 'VH backscatter change', 'dB', 'Sentinel-1', 0, 1.6, -8, 8);
F('VV_VH_CHANGE', 'sar', 'VV to VH ratio change', 'dB', 'Sentinel-1', 0, 1.2, -6, 6);
F('COHERENCE_LOSS', 'sar', 'Radar coherence loss', 'frac', 'Sentinel-1', .2, .2, 0, 1);
F('DNBR', 'sar', 'Burn severity (dNBR)', '', 'Sentinel-2 change', .05, .2, -.5, 1.2);
F('DNDVI', 'sar', 'Greenness loss (dNDVI)', '', 'Sentinel-2 change', .04, .15, -.4, 1);
F('STRUCTURAL_CHANGE_PROBABILITY', 'sar', 'Chance of structural change', 'prob', 'Sentinel-1', .2, .2, 0, 1);
/* 10. Air chemistry (9) */
F('NO2_ANOMALY_Z', 'chem', 'Nitrogen dioxide against normal', 'z', 'TROPOMI', 0, 1, -4, 6);
F('SO2_ANOMALY_Z', 'chem', 'Sulphur dioxide against normal', 'z', 'TROPOMI', 0, 1, -4, 6);
F('CO_ANOMALY_Z', 'chem', 'Carbon monoxide against normal', 'z', 'TROPOMI', 0, 1, -4, 6);
F('CH4_ANOMALY_Z', 'chem', 'Methane against normal', 'z', 'TROPOMI', 0, 1, -4, 6);
F('HCHO_ANOMALY_Z', 'chem', 'Formaldehyde against normal', 'z', 'TROPOMI', 0, 1, -4, 6);
F('AOD_ANOMALY_Z', 'chem', 'Haze (aerosol) against normal', 'z', 'Aerosol product', 0, 1, -4, 6);
F('CO_NO2_RATIO_LOG', 'chem', 'CO to NO₂ ratio (log)', 'log ratio', 'TROPOMI', 0, .8, -3, 3);
F('SO2_NO2_RATIO_LOG', 'chem', 'SO₂ to NO₂ ratio (log)', 'log ratio', 'TROPOMI', 0, .8, -3, 3);
F('CH4_CO_RATIO_LOG', 'chem', 'CH₄ to CO ratio (log)', 'log ratio', 'TROPOMI', 0, .8, -3, 3);
/* 11. Data quality (8) */
F('FIRMS_CONFIDENCE', 'quality', 'FIRMS confidence', '%', 'FIRMS', 72, 15, 10, 100);
F('PIXEL_SIZE', 'quality', 'Pixel size', 'km²', 'FIRMS', .2, .1, .1, 4.5);
F('SCAN_ANGLE_SIN', 'quality', 'Scan angle (sine)', '', 'FIRMS', 0, 1, -1, 1);
F('SCAN_ANGLE_COS', 'quality', 'Scan angle (cosine)', '', 'FIRMS', 0, 1, -1, 1);
F('VIEW_ZENITH_SIN', 'quality', 'Viewing angle (sine)', '', 'FIRMS', 0, 1, -1, 1);
F('VIEW_ZENITH_COS', 'quality', 'Viewing angle (cosine)', '', 'FIRMS', 0, 1, -1, 1);
F('CLOUD_FRACTION', 'quality', 'Cloud cover', 'frac', 'FIRMS / HLS', .3, .3, 0, 1);
F('MISSING_FEATURE_FRACTION', 'quality', 'Features missing', 'frac', 'Derived', .08, .08, 0, 1);

const FBY = Object.fromEntries(FDEF.map(d => [d.key, d]));
const FGROUP_COUNT = FAMILIES.reduce((o, f) => (o[f.id] = FDEF.filter(d => d.g === f.id).length, o), {});

/* typical values per class, in the same units as the catalogue [mean, spread]; booleans use [chance, 0] */
const FCV = {
  gas_flare: { FRP_MEDIAN: [14, 5], BT_I4_MEDIAN: [357, 4], BT_I4_I5_DIFF_MEDIAN: [56, 9], BT_I4_I5_DIFF_P90: [66, 9], SUBPIXEL_TEMP_MEDIAN: [1650, 260], SOURCE_AREA_MEDIAN: [40, 30], DURATION_HOURS_LOG: [6.3, .5], OVERPASS_DETECTION_RATIO: [.92, .07], RECURRENCE_RATE_1Y: [.93, .06], RECURRENCE_RATE_3Y: [.93, .06], RECURRENCE_RATE_5Y: [.92, .07], NIGHT_DAY_FRP_RATIO: [1.25, .25], STATIONARITY_INDEX: [.97, .02], CENTROID_DISPLACEMENT_MEDIAN: [.03, .02], SPREAD_SPEED_MEDIAN: [.01, .01], AREA_GROWTH_RATE: [0, .03], FRP_AUTOCORRELATION: [.82, .1], INSIDE_FACILITY: [.9, 0], KNOWN_FLARE: [.85, 0], DIST_WELLPAD_LOG: [.8, .9], DIST_NEAREST_FACILITY_LOG: [.6, .6], FOREST_FRAC_1KM: [.03, .04], CROPLAND_FRAC_1KM: [.12, .12], NDVI_MEDIAN: [.2, .08], SMOKE_PROBABILITY: [.12, .1], CH4_ANOMALY_Z: [1.4, 1] },
  industrial: { FRP_MEDIAN: [24, 12], BT_I4_MEDIAN: [349, 8], BT_I4_I5_DIFF_MEDIAN: [44, 10], SUBPIXEL_TEMP_MEDIAN: [1000, 250], DURATION_HOURS_LOG: [6, .7], OVERPASS_DETECTION_RATIO: [.85, .12], RECURRENCE_RATE_1Y: [.88, .1], RECURRENCE_RATE_3Y: [.88, .1], RECURRENCE_RATE_5Y: [.87, .1], STATIONARITY_INDEX: [.94, .04], CENTROID_DISPLACEMENT_MEDIAN: [.05, .04], SPREAD_SPEED_MEDIAN: [.02, .02], INSIDE_FACILITY: [.92, 0], KNOWN_FLARE: [.08, 0], DIST_NEAREST_FACILITY_LOG: [.5, .5], FACILITY_COUNT_1KM: [3.5, 2], FACILITY_COUNT_5KM: [8, 4], BUILTUP_FRAC_1KM: [.3, .15], FOREST_FRAC_1KM: [.04, .05], NDVI_MEDIAN: [.22, .08], SO2_ANOMALY_Z: [.8, 1] },
  mining: { FRP_MEDIAN: [11, 5], BT_I4_MEDIAN: [338, 9], DURATION_HOURS_LOG: [5.8, .8], OVERPASS_DETECTION_RATIO: [.7, .2], RECURRENCE_RATE_1Y: [.82, .12], RECURRENCE_RATE_3Y: [.85, .1], RECURRENCE_RATE_5Y: [.85, .1], STATIONARITY_INDEX: [.8, .12], INSIDE_FACILITY: [.55, 0], KNOWN_FLARE: [.02, 0], DIST_MINE_LOG: [.3, .4], DIST_NEAREST_FACILITY_LOG: [1.2, .8], BARE_MINING_FRAC_1KM: [.55, .15], SLOPE_P90: [12, 5], TRI: [16, 8], COHERENCE_LOSS: [.35, .2], SO2_ANOMALY_Z: [1, 1], CO_ANOMALY_Z: [.9, 1] },
  wildfire: { FRP_MEDIAN: [12, 8], BT_I4_MEDIAN: [335, 10], DURATION_HOURS_LOG: [3.2, .9], OVERPASS_DETECTION_RATIO: [.4, .2], RECURRENCE_RATE_1Y: [.1, .1], RECURRENCE_RATE_3Y: [.15, .12], RECURRENCE_RATE_5Y: [.18, .12], STATIONARITY_INDEX: [.25, .18], CENTROID_DISPLACEMENT_MEDIAN: [.6, .4], SPREAD_SPEED_MEDIAN: [.3, .2], SPREAD_SPEED_P90: [.8, .5], AREA_GROWTH_RATE: [.5, .5], ELONGATION: [2.2, .8], INSIDE_FACILITY: [.02, 0], KNOWN_FLARE: [.005, 0], DIST_NEAREST_FACILITY_LOG: [3.4, .7], FOREST_FRAC_375M: [.7, .2], FOREST_FRAC_1KM: [.7, .2], FOREST_FRAC_10KM: [.6, .2], NDVI_MEDIAN: [.62, .1], NDVI_ANOMALY_Z: [-.6, 1], SLOPE_MEDIAN: [7, 4], SLOPE_P90: [15, 6], FWI: [18, 9], ISI: [9, 5], VPD_ANOMALY_Z: [.8, 1], SMOKE_PROBABILITY: [.55, .25], DARK_SMOKE_PROBABILITY: [.4, .25], BURN_SCAR_GROWTH_RATE: [.3, .3], DNBR: [.3, .2] },
  agricultural_burning: { FRP_MEDIAN: [7, 4], BT_I4_MEDIAN: [332, 8], DURATION_HOURS_LOG: [2.6, .7], OVERPASS_DETECTION_RATIO: [.35, .2], RECURRENCE_RATE_1Y: [.35, .2], RECURRENCE_RATE_3Y: [.5, .2], RECURRENCE_RATE_5Y: [.55, .2], STATIONARITY_INDEX: [.4, .2], CENTROID_DISPLACEMENT_MEDIAN: [.25, .2], SPREAD_SPEED_MEDIAN: [.12, .08], INSIDE_FACILITY: [.01, 0], KNOWN_FLARE: [.003, 0], DIST_NEAREST_FACILITY_LOG: [3, .7], CROPLAND_FRAC_375M: [.85, .1], CROPLAND_FRAC_1KM: [.82, .1], CROPLAND_FRAC_10KM: [.75, .12], FOREST_FRAC_1KM: [.03, .04], NDVI_MEDIAN: [.28, .1], NBR_CHANGE: [-.12, .1], SLOPE_MEDIAN: [1, .8], AOD_ANOMALY_Z: [1.6, 1], CO_ANOMALY_Z: [1.1, 1], CO_NO2_RATIO_LOG: [.7, .6], SMOKE_PROBABILITY: [.5, .25] },
  unknown: { FRP_MEDIAN: [4, 2.5], BT_I4_MEDIAN: [326, 7], DURATION_HOURS_LOG: [5, 1], OVERPASS_DETECTION_RATIO: [.5, .25], RECURRENCE_RATE_1Y: [.5, .3], STATIONARITY_INDEX: [.85, .12], INSIDE_FACILITY: [.3, 0], KNOWN_FLARE: [.02, 0], DIST_LANDFILL_LOG: [.5, .6], FIRMS_CONFIDENCE: [50, 12], SMOKE_PROBABILITY: [.15, .15] }
};
/* which inputs push each class up (sample contribution weights) */
const FSHAP = {
  gas_flare: { KNOWN_FLARE: .3, RECURRENCE_RATE_1Y: .28, INSIDE_FACILITY: .22, BT_I4_I5_DIFF_MEDIAN: .22, SUBPIXEL_TEMP_MEDIAN: .2, OVERPASS_DETECTION_RATIO: .18, STATIONARITY_INDEX: .16, DIST_NEAREST_FACILITY_LOG: .14, DIST_WELLPAD_LOG: .12, CH4_ANOMALY_Z: .07, FOREST_FRAC_1KM: -.05, NIGHT_DAY_FRP_RATIO: .08 },
  industrial: { INSIDE_FACILITY: .3, DIST_NEAREST_FACILITY_LOG: .24, RECURRENCE_RATE_3Y: .2, BUILTUP_FRAC_1KM: .15, FACILITY_COUNT_1KM: .14, STATIONARITY_INDEX: .12, FRP_PERSISTENCE: .1, SO2_ANOMALY_Z: .06, NDVI_MEDIAN: .05, KNOWN_FLARE: -.06 },
  mining: { DIST_MINE_LOG: .32, BARE_MINING_FRAC_1KM: .26, RECURRENCE_RATE_3Y: .2, COHERENCE_LOSS: .09, SLOPE_P90: .08, TRI: .07, SO2_ANOMALY_Z: .06, CO_ANOMALY_Z: .05, INSIDE_FACILITY: -.05, FOREST_FRAC_1KM: -.06 },
  wildfire: { FOREST_FRAC_1KM: .3, NDVI_MEDIAN: .18, SPREAD_SPEED_MEDIAN: .16, FWI: .16, AREA_GROWTH_RATE: .14, DURATION_HOURS_LOG: .12, SMOKE_PROBABILITY: .12, VPD_ANOMALY_Z: .1, RECURRENCE_RATE_1Y: .1, DIST_NEAREST_FACILITY_LOG: .1, DNBR: .08, INSIDE_FACILITY: -.05 },
  agricultural_burning: { CROPLAND_FRAC_1KM: .34, DURATION_HOURS_LOG: .14, NBR_CHANGE: .12, RECURRENCE_RATE_3Y: .12, DIST_NEAREST_FACILITY_LOG: .1, AOD_ANOMALY_Z: .08, CO_NO2_RATIO_LOG: .06, STATIONARITY_INDEX: .06, FOREST_FRAC_1KM: -.07, INSIDE_FACILITY: -.05 },
  unknown: { DIST_LANDFILL_LOG: .18, STATIONARITY_INDEX: .1, INSIDE_FACILITY: .06, FIRMS_CONFIDENCE: -.12, BT_I4_MEDIAN: -.14, FRP_MEDIAN: -.1, CLOUD_FRACTION: -.08, MISSING_FEATURE_FRACTION: -.06 }
};

const normCdf = z => { const t = 1 / (1 + .2316419 * Math.abs(z)), d = .3989423 * Math.exp(-z * z / 2), p = d * t * (.3193815 + t * (-.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274)))); return z > 0 ? 1 - p : p; };
function fmtNum(v) { const a = Math.abs(v); return a >= 1000 ? Math.round(v).toLocaleString('en-IN') : a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : a >= 1 ? v.toFixed(2) : v.toFixed(3).replace(/0+$/, '').replace(/\.$/, '') || '0'; }
function fmtFeat(d, v) {
  if (v == null || isNaN(v)) return '—';
  if (d.fl.includes('b')) return v ? 'Yes' : 'No';
  if (d.unit === 'frac' || d.unit === 'prob') return Math.round(v * 100) + '%';
  if (d.unit === 'z') return (v >= 0 ? '+' : '') + v.toFixed(1) + ' σ';
  if (d.unit === 'count') return String(Math.round(v));
  if (d.unit === '%') return Math.round(v) + '%';
  return fmtNum(v) + (d.unit ? ' ' + d.unit : '');
}
const isAngle = k => /_(SIN|COS)$/.test(k);
const angleName = k => k.replace(/_(SIN|COS)$/, '');

/* the full feature vector for one event (sample) */
function featuresOf(p, ev) {
  if (ev._f) return ev._f;
  const r = R('feat:' + ev.id), ctx = contextOf(p), W = weatherOf(p, ev), c = ev.cls, ov = FCV[c] || {}, out = {}, ang = {};
  ang.WIND_DIRECTION = W.dir[0]; ang.SPREAD_DIRECTION = (W.dir[0] + 180 + r.range(-30, 30) + 360) % 360; ang.SMOKE_DIRECTION = (W.dir[0] + 180 + r.range(-18, 18) + 360) % 360;
  ang.PEAK_HOUR = ev.hr / 24 * 360; ang.SCAN_ANGLE = r.range(-52, 52); ang.VIEW_ZENITH = Math.abs(ang.SCAN_ANGLE) * 1.12 + r.range(0, 4);
  const put = (d, raw) => { const shown = d.fl.includes('x') ? Math.exp(raw) - 1 : raw; const z = d.sd ? (raw - (ov[d.key] ? ov[d.key][0] : d.mu)) / ((ov[d.key] ? ov[d.key][1] : d.sd) || 1) : 0; out[d.key] = { d, raw, v: shown, pct: (d.fl.includes('b') || isAngle(d.key)) ? null : normCdf(z) * 100, missing: false }; };
  FDEF.forEach(d => {
    if (isAngle(d.key)) { const a = (ang[angleName(d.key)] != null ? ang[angleName(d.key)] : (ang[angleName(d.key)] = r.range(0, 360))) * Math.PI / 180; const v = /_SIN$/.test(d.key) ? Math.sin(a) : Math.cos(a); out[d.key] = { d, raw: v, v, pct: null, missing: false }; return; }
    const o = ov[d.key], mu = o ? o[0] : d.mu, sd = o ? o[1] : d.sd;
    put(d, d.fl.includes('b') ? (r() < mu ? 1 : 0) : clamp(mu + sd * r.norm(), d.lo, d.hi));
  });
  const set = (k, shown) => { const d = FBY[k]; put(d, d.fl.includes('x') ? Math.log(1 + Math.max(0, shown)) : shown); };
  /* tie the vector to what the rest of the app already shows for this event */
  const pt = ctx.parts, pc = n => (pt.find(x => x[0] === n) || [0, 0])[1] / 100, jit = (v, s) => clamp(v + r.norm() * s, 0, 1);
  set('FRP_MAX', ev.frp * 1.15); set('FRP_MEDIAN', ev.frp * r.range(.5, .75)); set('FRP_P90', ev.frp * r.range(.85, 1)); set('FRP_SUM_LOG', Math.log(ev.frp * ev.nDet * 1.2 + 1));
  set('DETECTION_COUNT_LOG', ev.nDet);
  set('FOREST_FRAC_1KM', pc('Forest')); set('FOREST_FRAC_375M', jit(pc('Forest'), .06)); set('FOREST_FRAC_10KM', clamp(pc('Forest') * .7 + .1 + r.norm() * .05, 0, 1)); set('FOREST_LOCAL_REGIONAL_CONTRAST', pc('Forest') - .18);
  set('CROPLAND_FRAC_1KM', pc('Cropland')); set('CROPLAND_FRAC_10KM', clamp(pc('Cropland') * .8 + .05 + r.norm() * .05, 0, 1)); set('CROPLAND_LOCAL_REGIONAL_CONTRAST', pc('Cropland') - .3);
  set('BUILTUP_FRAC_1KM', pc('Industrial or built-up')); set('BUILTUP_FRAC_10KM', clamp(pc('Industrial or built-up') * .5 + .04, 0, 1)); set('BARE_MINING_FRAC_1KM', pc('Bare ground')); set('WATER_FRAC_1KM', pc('Water and other') * .5);
  const ent = -pt.reduce((a, x) => a + (x[1] > 0 ? x[1] / 100 * Math.log2(x[1] / 100) : 0), 0); set('LANDCOVER_ENTROPY', ent);
  set('NDVI_MEDIAN', ctx.ndvi); set('NBR_CHANGE', c === 'wildfire' ? -(ctx.nbr * .5) : -.04 * r.range(0, 2));
  set('ELEVATION_MEDIAN', ctx.elev); set('SLOPE_MEDIAN', ctx.slope); set('SLOPE_P90', ctx.slope * 2.1 + 1);
  const dk = ctx.d, ln = k => Math.log(1 + k);
  set('DIST_REFINERY_LOG', dk.refinery); set('DIST_POWERPLANT_LOG', dk.power); set('DIST_STEEL_LOG', dk.steel); set('DIST_CEMENT_LOG', dk.steel * r.range(.8, 1.6)); set('DIST_MINE_LOG', dk.mine); set('DIST_LANDFILL_LOG', dk.landfill);
  set('DIST_PETROCHEMICAL_LOG', /Petro/.test(p.type) ? r.range(0, .8) : dk.refinery * r.range(1, 2));
  set('DIST_LNG_LOG', /LNG/.test(p.type) ? r.range(0, .6) : r.range(30, 300));
  const near = Math.min(dk.refinery, dk.power, dk.steel, dk.mine, dk.landfill, p.kind === 'site' && p.cls !== 'unknown' ? r.range(0, 1.2) : 99);
  set('DIST_NEAREST_FACILITY_LOG', near > 90 ? r.range(6, 60) : near);
  /* weather: take the first forecast hour so the vector agrees with the wind map */
  set('WIND_SPEED', W.spd[0]); set('WIND_GUST', W.gust[0]);
  set('FFMC', W.ffmc[0]); set('DMC', W.dmc); set('DC', W.dc);
  const isiV = calcISI(W.ffmc[0], W.spd[0] * 3.6), buiV = calcBUI(W.dmc, W.dc), fwiV = calcFWI(isiV, buiV);
  set('ISI', isiV); set('BUI', buiV); set('FWI', fwiV); set('FWI_ANOMALY_Z', clamp((fwiV - 10) / 8, -3, 4));
  set('VPD_ANOMALY_Z', clamp((W.vpd[0] - 2.2) / 1.1, -3, 4)); set('RH_ANOMALY_Z', clamp(-(W.vpd[0] - 2.2) / 1.6 + r.norm() * .3, -3, 3));
  set('RAIN_24H_LOG', W.rainPast24); set('RAIN_7D_LOG', W.rainPast24 * 2.4 + r.range(0, 6));
  set('SPREAD_WIND_SPEED_RATIO', out.SPREAD_SPEED_MEDIAN.v * 1000 / 3600 / Math.max(.5, W.spd[0]));
  set('CLOUD_FRACTION', clamp(1 - ctx.valid + r.norm() * .05, 0, 1));
  const sc = Math.abs(ang.SCAN_ANGLE), mod = ev.sensor.startsWith('MODIS');
  set('PIXEL_SIZE', (mod ? 1 : .14) * (1 + Math.pow(sc / 55, 2) * 2.2));
  set('FIRMS_CONFIDENCE', mod ? clamp(62 + r.norm() * 14, 10, 100) : clamp((ov.FIRMS_CONFIDENCE ? ov.FIRMS_CONFIDENCE[0] : 78) + r.norm() * 10, 10, 100));
  set('DETECTION_DENSITY', ev.nDet / Math.max(ev.area, .05));
  /* missing data: cloud hides optical inputs, radar and gas satellites pass only every few days */
  const cloudy = ctx.valid < .38;
  FDEF.forEach(d => { const pm = d.g === 'sar' ? .28 : d.g === 'chem' ? .22 : d.g === 'smoke' ? (cloudy ? .55 : .12) : (d.src === 'HLS' ? (cloudy ? .7 : .04) : .015); if (out[d.key] && d.key !== 'MISSING_FEATURE_FRACTION' && r() < pm) { out[d.key].missing = true; out[d.key].v = null; out[d.key].pct = null; } });
  const miss = FDEF.filter(d => out[d.key].missing).length; set('MISSING_FEATURE_FRACTION', miss / FDEF.length);
  return ev._f = { by: out, missing: miss, W, ang };
}

/* sample explanation: which features pushed the call, and by how much */
function shapOf(ev) {
  if (ev._shap) return ev._shap;
  const p = PMAP[ev.pid], f = featuresOf(p, ev), r = R('shap:' + ev.id), Wt = FSHAP[ev.cls] || {}, all = {}, fam = {};
  FAMILIES.forEach(g => fam[g.id] = 0);
  FDEF.forEach(d => { const e = f.by[d.key]; let w = 0; if (!e.missing) { w = Wt[d.key] != null ? Wt[d.key] * r.range(.8, 1.25) * (.6 + ev.conf * .5) : r.norm() * .012; } all[d.key] = w; fam[d.g] += w; });
  const feats = FDEF.filter(d => Wt[d.key] != null && !f.by[d.key].missing).map(d => ({ k: d.key, l: d.label, f: d.g, w: all[d.key], v: fmtFeat(d, f.by[d.key].v) })).sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
  return ev._shap = { feats: feats.slice(0, 8), fam, all };
}

/* the feature groups, used by the explorer and the group tabs */
const FSRC_TIP = 'These inputs come from the same sources used to make the training labels, so the model can partly read its own answer back. Treat their contribution with care.';
