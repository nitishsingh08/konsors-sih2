'use strict';
/* =====================================================================
   The sections of the Advanced analysis panel.
   Each tab returns { html, after } and reads the event's 141 features.
   ===================================================================== */
const FE = () => featuresOf(A.p, A.ev);
const fv = k => FE().by[k];
const fnum = k => { const e = fv(k); return e && !e.missing ? e.v : null; };
const fs = k => { const e = fv(k); return e.missing ? 'no data' : fmtFeat(e.d, e.v); };
const pctWord = e => e.pct == null || e.missing ? '' : (e.pct >= 50 ? `higher than ${Math.round(e.pct)}% of similar events` : `lower than ${Math.round(100 - e.pct)}% of similar events`);
function ftile(k) {
  const e = fv(k), d = e.d, val = e.missing ? '<span class="dim">No data</span>' : esc(fmtFeat(d, e.v));
  return `<div class="stat"><div class="l">${esc(d.label)}${d.fl.includes('k') ? ` <span class="lk" data-tip="${esc(FSRC_TIP)}">also feeds labels</span>` : ''}</div><div class="v">${val}</div>${e.pct != null && !e.missing ? `<div class="mini"><i style="width:${e.pct.toFixed(0)}%"></i></div><div class="u sm2">${pctWord(e)}</div>` : ''}</div>`;
}
const ftiles = keys => `<div class="stats">${keys.map(ftile).join('')}</div>`;
const reads = arr => { const a = arr.filter(Boolean); return a.length ? `<section class="card" style="margin-top:12px"><div class="ch"><h3>What this says</h3></div><div class="cbody"><ul class="reads">${a.map(s => `<li>${s}</li>`).join('')}</ul></div></section>` : ''; };
const sec = (title, body, right, tip) => `<section class="card" style="margin-top:12px"><div class="ch"><h3>${title}</h3>${tip ? info(tip) : ''}<span class="r">${right || ''}</span></div><div class="cbody">${body}</div></section>`;
function groupTable(gid) {
  const list = FDEF.filter(d => d.g === gid), E = FE().by;
  return `<details class="card" style="margin-top:12px"><summary class="cbody" style="cursor:pointer;font-weight:600">All ${list.length} features in this group</summary><div class="cbody" style="padding-top:0"><table class="tbl small"><tbody>${list.map(d => { const e = E[d.key]; return `<tr><td>${esc(d.label)}<div class="dim fk">${d.key}</div></td><td class="r num">${e.missing ? '<span class="dim">no data</span>' : esc(fmtFeat(d, e.v))}</td></tr>`; }).join('')}</tbody></table></div></details>`;
}
const compassSvg = arrows => { const S_ = 116, c = S_ / 2; return `<svg width="${S_}" height="${S_}" viewBox="0 0 ${S_} ${S_}" role="img" aria-label="Compass"><circle cx="${c}" cy="${c}" r="46" fill="none" stroke="var(--line2)"/><circle cx="${c}" cy="${c}" r="2" fill="var(--dim)"/>${['N', 'E', 'S', 'W'].map((t, i) => `<text x="${c + Math.sin(i * Math.PI / 2) * 55}" y="${c - Math.cos(i * Math.PI / 2) * 55 + 4}" text-anchor="middle" style="fill:var(--dim);font-size:10px">${t}</text>`).join('')}${arrows.map(a => `<g transform="rotate(${a.deg} ${c} ${c})"><line x1="${c}" y1="${c + 10}" x2="${c}" y2="${c - 40}" stroke="${a.color}" stroke-width="2.4" stroke-linecap="round"/><path d="M${c} ${c - 44} l-5 9 l10 0 z" fill="${a.color}"/></g>`).join('')}</svg>`; };
function meter(label, value, lo, hi, mu, sd, fmt) {
  const pos = clamp((value - lo) / (hi - lo), 0, 1) * 100, a = clamp((mu - sd - lo) / (hi - lo), 0, 1) * 100, b = clamp((mu + sd - lo) / (hi - lo), 0, 1) * 100;
  return `<div class="meter"><div class="row between small"><span>${label}</span><b class="num">${fmt ? fmt(value) : value.toFixed(0)}</b></div><div class="track"><i class="band" style="left:${a}%;width:${Math.max(2, b - a)}%"></i><i class="mark" style="left:${pos}%"></i></div></div>`;
}
const zrow = k => { const e = fv(k), z = e.missing ? 0 : clamp(e.v, -4, 6), w = Math.abs(z) / 6 * 50, c = ccol(A.ev.cls); return `<div class="shr" style="grid-template-columns:150px 1fr 56px"><span class="tx2">${esc(e.d.label.replace(' against normal', ''))}</span><span class="ax"><i style="${e.missing ? 'display:none' : z >= 0 ? `left:50%;width:${w}%;background:${c}` : `right:50%;width:${w}%;background:var(--dim)`}"></i></span><span class="vv">${e.missing ? 'no data' : (z >= 0 ? '+' : '') + z.toFixed(1)}</span></div>`; };

/* ---------- the group tabs ---------- */
function readsFor(gid) {
  const E = FE().by, n = fnum, cls = A.ev.cls, p = A.p; const r = [];
  const P = k => fv(k).pct == null ? null : Math.round(fv(k).pct);
  if (gid === 'thermal') {
    r.push(`Median fire power is ${fs('FRP_MEDIAN')} and the peak is ${fs('FRP_MAX')}. ${P('FRP_MEDIAN') != null ? 'That is ' + pctWord(fv('FRP_MEDIAN')) + '.' : ''}`);
    if (n('BT_I4_I5_DIFF_MEDIAN') != null) r.push(n('BT_I4_I5_DIFF_MEDIAN') > 48 ? 'The gap between the 4 µm and 11 µm bands is wide. That usually means a small, very hot source such as a flare or furnace.' : n('BT_I4_I5_DIFF_MEDIAN') < 30 ? 'The two infrared bands sit close together, which is more like cooler, spread-out burning such as a vegetation fire.' : 'The two infrared bands are in the middle range, so this does not point clearly either way.');
    if (n('SUBPIXEL_TEMP_MEDIAN') != null && n('SUBPIXEL_TEMP_MEDIAN') > 1300) r.push(`The hot part is estimated at ${fs('SUBPIXEL_TEMP_MEDIAN')}, hotter than most vegetation fires.`);
    if (n('FRP_SLOPE') != null) r.push(n('FRP_SLOPE') > .8 ? 'Fire power is climbing between passes.' : n('FRP_SLOPE') < -.8 ? 'Fire power is fading between passes.' : 'Fire power is roughly steady between passes.');
  }
  if (gid === 'temporal') {
    r.push(n('RECURRENCE_RATE_1Y') != null ? `It has shown up again in ${fs('RECURRENCE_RATE_1Y')} of the past year's windows and ${fs('RECURRENCE_RATE_3Y')} over three years. ${n('RECURRENCE_RATE_1Y') > .75 ? 'Coming back this regularly is what a fixed source does.' : n('RECURRENCE_RATE_1Y') < .2 ? 'It rarely returns here, which fits a one-off fire.' : ''}` : '');
    r.push(`It has burned for about ${fs('DURATION_HOURS_LOG')} and was seen on ${fs('OVERPASS_DETECTION_RATIO')} of passes.`);
    if (n('NIGHT_DAY_FRP_RATIO') != null) r.push(n('NIGHT_DAY_FRP_RATIO') > 1.15 ? 'It burns at least as hot at night as by day, which is typical of industrial sources.' : n('NIGHT_DAY_FRP_RATIO') < .6 ? 'It is much weaker at night, which is typical of fires that follow the daily heating cycle.' : '');
  }
  if (gid === 'spread') {
    r.push(n('STATIONARITY_INDEX') != null ? (n('STATIONARITY_INDEX') > .85 ? 'It stays put. The centre barely moves between passes.' : n('STATIONARITY_INDEX') < .45 ? 'It is moving. The centre shifts noticeably between passes.' : 'It moves a little between passes.') : '');
    r.push(`Typical spread speed is ${fs('SPREAD_SPEED_MEDIAN')} and the fastest is ${fs('SPREAD_SPEED_MAX')}.`);
    if (n('DIRECTION_CONSISTENCY') != null && n('STATIONARITY_INDEX') < .6) r.push(n('DIRECTION_CONSISTENCY') > .6 ? `It keeps heading one way, toward the ${compass(FE().ang.SPREAD_DIRECTION)}.` : 'It wanders, with no steady direction.');
  }
  if (gid === 'terrain') {
    r.push(`The ground is about ${fs('ELEVATION_MEDIAN')} above sea level with a typical slope of ${fs('SLOPE_MEDIAN')}. The steeper sides reach ${fs('SLOPE_P90')}.`);
    if (n('SLOPE_P90') > 15) r.push('Slopes this steep make uphill fire runs much faster, so the wind direction against the slope matters here.');
    if (n('RIDGE_VALLEY_POSITION') != null) r.push(n('RIDGE_VALLEY_POSITION') > .3 ? 'It sits on or near a ridge, where wind is stronger and fire spreads more freely.' : n('RIDGE_VALLEY_POSITION') < -.3 ? 'It sits in a valley, where smoke can pool and wind can funnel along the floor.' : '');
  }
  if (gid === 'fuel') {
    r.push(`Within 1 km the land is ${fs('FOREST_FRAC_1KM')} forest, ${fs('CROPLAND_FRAC_1KM')} cropland and ${fs('BUILTUP_FRAC_1KM')} built-up.`);
    if (n('NDVI_ANOMALY_Z') != null) r.push(n('NDVI_ANOMALY_Z') < -1 ? 'The vegetation is browner than normal for this time of year, so it is drier fuel.' : n('NDVI_ANOMALY_Z') > 1 ? 'The vegetation is greener than normal, which slows burning.' : 'The vegetation is close to normal for the season.');
    else r.push('Imagery is missing here (likely cloud), so vegetation readings are not available.');
    if (n('NBR_CHANGE') != null && n('NBR_CHANGE') < -.15) r.push('The burn ratio has dropped, which is a sign of burnt ground.');
  }
  if (gid === 'industry') {
    r.push(`The nearest facility is ${fs('DIST_NEAREST_FACILITY_LOG')} away and ${fv('INSIDE_FACILITY').v ? 'this point is inside a facility boundary' : 'this point is outside any known boundary'}.`);
    if (fv('KNOWN_FLARE').v) r.push('It matches a known flare in the VIIRS Nightfire catalogue.');
    r.push(`${fs('FACILITY_COUNT_1KM')} facilities within 1 km, ${fs('FACILITY_COUNT_10KM')} within 10 km.`);
    r.push(`<span class="dim">Facility and flare matches also help make the training labels, so the model can partly read its own answer back. Check the call against the other groups before trusting it.</span>`);
  }
  if (gid === 'weather') {
    r.push(`Fire weather index is ${fs('FWI')}, ${n('FWI_ANOMALY_Z') != null ? (n('FWI_ANOMALY_Z') > 1 ? 'well above what is normal for now' : n('FWI_ANOMALY_Z') < -1 ? 'below normal for now' : 'close to normal for now') : ''}.`);
    r.push(`Wind is ${fs('WIND_SPEED')} with gusts to ${fs('WIND_GUST')}. Rain in the last 24 hours: ${fs('RAIN_24H_LOG')}.`);
    if (n('VPD_ANOMALY_Z') > 1.2) r.push('The air is much drier than normal, which dries fuel quickly.');
  }
  if (gid === 'quality') {
    r.push(`FIRMS rates this detection at ${fs('FIRMS_CONFIDENCE')} confidence. One pixel covers ${fs('PIXEL_SIZE')}.`);
    if (n('CLOUD_FRACTION') > .5) r.push('Cloud covers more than half the area, so imagery-based inputs are weak or missing.');
    r.push(`${Math.round(FE().missing)} of ${FDEF.length} features are missing for this event.`);
  }
  return r;
}
const GROUP_TILES = {
  thermal: ['FRP_MEDIAN', 'FRP_MAX', 'FRP_P90', 'BT_I4_MEDIAN', 'BT_I4_I5_DIFF_MEDIAN', 'SUBPIXEL_TEMP_MEDIAN', 'SOURCE_AREA_MEDIAN', 'FRP_SLOPE', 'FRP_DECAY_RATE'],
  temporal: ['DURATION_HOURS_LOG', 'DETECTION_COUNT_LOG', 'OVERPASS_DETECTION_RATIO', 'RECURRENCE_RATE_1Y', 'MAX_INTER_EVENT_GAP_LOG', 'TIME_TO_PEAK'],
  spread: ['SPREAD_SPEED_MEDIAN', 'SPREAD_SPEED_P90', 'STATIONARITY_INDEX', 'AREA_GROWTH_RATE', 'ELONGATION', 'NEW_PIXEL_FRACTION'],
  terrain: ['ELEVATION_MEDIAN', 'SLOPE_MEDIAN', 'SLOPE_P90', 'TRI', 'TPI', 'RIDGE_VALLEY_POSITION'],
  fuel: ['NDVI_MEDIAN', 'NDVI_ANOMALY_Z', 'NDMI_ANOMALY_Z', 'NBR_CHANGE', 'LANDCOVER_ENTROPY', 'WATER_FRAC_1KM'],
  industry: ['DIST_NEAREST_FACILITY_LOG', 'INSIDE_FACILITY', 'KNOWN_FLARE', 'FACILITY_COUNT_1KM', 'FACILITY_COUNT_5KM', 'FACILITY_COUNT_10KM', 'INDUSTRIAL_DENSITY_CONTRAST', 'DIST_FACILITY_BOUNDARY_LOG'],
  weather: ['TEMPERATURE_ANOMALY_Z', 'RH_ANOMALY_Z', 'VPD_ANOMALY_Z', 'SOIL_MOISTURE_ANOMALY_Z', 'WIND_SPEED', 'WIND_GUST', 'RAIN_24H_LOG', 'RAIN_7D_LOG', 'LIGHTNING_COUNT_24H'],
  quality: ['FIRMS_CONFIDENCE', 'PIXEL_SIZE', 'CLOUD_FRACTION', 'MISSING_FEATURE_FRACTION']
};
function groupChart(gid) {
  const T = tk(), E = FE().by, col = ccol(A.ev.cls);
  if (gid === 'thermal') {
    const peak = fnum('FRP_MAX') || 10, tp = fnum('TIME_TO_PEAK') || 6, dec = fnum('FRP_DECAY_RATE') || .06, med = fnum('FRP_MEDIAN') || peak * .5, d = [];
    for (let h = 0; h <= 48; h++) d.push([h, +(h <= tp ? med * .3 + (peak - med * .3) * Math.pow(h / Math.max(tp, .5), 1.4) : peak * Math.exp(-dec * (h - tp))).toFixed(1)]);
    return { html: sec('Heat over time', '<div id="g-c1"></div><p class="dim small" style="margin:4px 0 0">A sketch drawn from the peak, time to peak and decay rate. Not the raw passes.</p>', 'Sketch'), after: () => achart('#g-c1', { grid: { left: 44, right: 12, top: 12, bottom: 24 }, tooltip: Object.assign(tt(), { trigger: 'axis' }), xAxis: Object.assign({ type: 'value', name: 'hours', nameLocation: 'end', nameTextStyle: { color: T.dim, fontSize: 10 } }, axb(), { splitLine: { show: false } }), yAxis: Object.assign({ type: 'value', name: 'MW', nameTextStyle: { color: T.dim, fontSize: 10, align: 'left' } }, axb()), series: [{ type: 'line', data: d, showSymbol: false, smooth: .3, lineStyle: { color: col, width: 2 }, areaStyle: { color: alpha(col, .16) }, markLine: { silent: true, symbol: 'none', label: { formatter: 'peak', color: T.dim, fontSize: 10 }, lineStyle: { color: T.dim, type: 'dashed' }, data: [{ xAxis: tp }] } }] }, 170) };
  }
  if (gid === 'temporal') {
    const v = ['RECURRENCE_RATE_1Y', 'RECURRENCE_RATE_3Y', 'RECURRENCE_RATE_5Y'].map(fnum);
    return { html: sec('How often it comes back', '<div id="g-c1"></div>', 'Share of windows'), after: () => achart('#g-c1', { grid: { left: 40, right: 12, top: 8, bottom: 24 }, tooltip: Object.assign(tt(), { trigger: 'axis', formatter: q => `${q[0].name}: ${q[0].data == null ? 'no data' : Math.round(q[0].data * 100) + '%'}` }), xAxis: Object.assign({ type: 'category', data: ['1 year', '3 years', '5 years'] }, axb(), { splitLine: { show: false } }), yAxis: Object.assign({ type: 'value', min: 0, max: 1, axisLabel: { color: T.dim, fontSize: 10, formatter: x => Math.round(x * 100) + '%' } }, axb()), series: [{ type: 'bar', barWidth: '42%', data: v, itemStyle: { color: col, borderRadius: [3, 3, 0, 0] } }] }, 150) };
  }
  if (gid === 'spread') {
    const fr = footprintsOf(A.p, A.ev), lat0 = A.p.lat, cen = fr.map(f => { const lo = f.ring.reduce((a, q) => a + q[0], 0) / f.ring.length, la = f.ring.reduce((a, q) => a + q[1], 0) / f.ring.length; return [+((lo - A.p.lon) * mLon(lat0) / 1000).toFixed(2), +((la - A.p.lat) * M_LAT / 1000).toFixed(2)]; });
    return { html: sec('Where the centre has moved', '<div id="g-c1"></div><p class="dim small" style="margin:4px 0 0">Centre of the footprint on each of the last 8 passes, in km from the current position.</p>', 'Last 8 passes'), after: () => achart('#g-c1', { grid: { left: 44, right: 16, top: 12, bottom: 28 }, tooltip: Object.assign(tt(), { formatter: q => `Pass ${q.dataIndex + 1}<br>${q.data[0]} km east, ${q.data[1]} km north` }), xAxis: Object.assign({ type: 'value', name: 'km east', nameLocation: 'middle', nameGap: 20, nameTextStyle: { color: T.dim, fontSize: 10 }, scale: true }, axb()), yAxis: Object.assign({ type: 'value', name: 'km north', nameTextStyle: { color: T.dim, fontSize: 10, align: 'left' }, scale: true }, axb()), series: [{ type: 'line', data: cen, symbolSize: 7, lineStyle: { color: col, width: 2 }, itemStyle: { color: col }, endLabel: { show: true, formatter: 'now', color: T.dim, fontSize: 10 } }] }, 200) };
  }
  if (gid === 'terrain') {
    const asp = FE().ang.ASPECT, wind = (FE().ang.WIND_DIRECTION + 180) % 360;
    return { html: sec('Which way the ground faces', `<div class="row" style="gap:18px;align-items:center">${compassSvg([{ deg: asp, color: col }, { deg: wind, color: 'var(--tx2)' }])}<div class="small tx2" style="display:grid;gap:6px"><span class="row"><i class="sw" style="background:${col}"></i>Slope faces ${compass(asp)}</span><span class="row"><i class="sw" style="background:var(--tx2)"></i>Wind blows toward ${compass(wind)}</span><span class="dim">A fire spreads fastest when the wind pushes it uphill.</span></div></div>`, 'Slope and wind'), after: null };
  }
  if (gid === 'fuel') {
    const f = (a, b) => [a, b].map(x => x == null ? 0 : x), rows = [['375 m', fnum('FOREST_FRAC_375M') || 0, 0, 0], ['1 km', fnum('FOREST_FRAC_1KM') || 0, fnum('CROPLAND_FRAC_1KM') || 0, fnum('BUILTUP_FRAC_1KM') || 0], ['10 km', fnum('FOREST_FRAC_10KM') || 0, fnum('CROPLAND_FRAC_10KM') || 0, fnum('BUILTUP_FRAC_10KM') || 0]];
    return { html: sec('Land cover at three scales', '<div id="g-c1"></div><div class="legendrow"><span><i class="sw" style="background:#4F8F6B"></i>Forest</span><span><i class="sw" style="background:#B9A45B"></i>Cropland</span><span><i class="sw" style="background:#6F86A6"></i>Built-up</span><span><i class="sw" style="background:var(--line2)"></i>Other</span></div><p class="dim small" style="margin:6px 0 0">At 375 m only forest is measured.</p>', 'Share of area'), after: () => achart('#g-c1', { grid: { left: 52, right: 12, top: 6, bottom: 22 }, tooltip: Object.assign(tt(), { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: v => Math.round(v * 100) + '%' }), xAxis: Object.assign({ type: 'value', max: 1, axisLabel: { color: T.dim, fontSize: 10, formatter: x => Math.round(x * 100) + '%' } }, axb()), yAxis: Object.assign({ type: 'category', data: rows.map(r => r[0]), inverse: true }, axb(), { splitLine: { show: false } }), series: [['Forest', 1, '#4F8F6B'], ['Cropland', 2, '#B9A45B'], ['Built-up', 3, '#6F86A6']].map(([n, i, c]) => ({ name: n, type: 'bar', stack: 's', barWidth: 20, itemStyle: { color: c }, data: rows.map(r => +r[i].toFixed(3)) })).concat([{ name: 'Other', type: 'bar', stack: 's', barWidth: 20, itemStyle: { color: T.line2 }, data: rows.map(r => +Math.max(0, 1 - r[1] - r[2] - r[3]).toFixed(3)) }]) }, 140) };
  }
  if (gid === 'industry') {
    const keys = ['DIST_REFINERY_LOG', 'DIST_POWERPLANT_LOG', 'DIST_PETROCHEMICAL_LOG', 'DIST_STEEL_LOG', 'DIST_CEMENT_LOG', 'DIST_LNG_LOG', 'DIST_MINE_LOG', 'DIST_LANDFILL_LOG', 'DIST_PIPELINE_LOG', 'DIST_WELLPAD_LOG'].map(k => ({ k, l: FBY[k].label.replace('Nearest ', ''), v: fnum(k) })).filter(x => x.v != null).sort((a, b) => b.v - a.v);
    return { html: sec('Distance to each kind of facility', '<div id="g-c1"></div><p class="dim small" style="margin:4px 0 0">Log scale, so the near ones stay readable.</p>', 'km'), after: () => achart('#g-c1', { grid: { left: 118, right: 22, top: 6, bottom: 22 }, tooltip: Object.assign(tt(), { formatter: q => `${q.name}: ${q.value < 10 ? q.value.toFixed(1) : Math.round(q.value)} km` }), xAxis: Object.assign({ type: 'log', min: .1 }, axb()), yAxis: Object.assign({ type: 'category', data: keys.map(x => x.l) }, axb(), { splitLine: { show: false }, axisLabel: { color: T.tx2, fontSize: 11 } }), series: [{ type: 'bar', barWidth: 11, data: keys.map((x, i) => ({ value: Math.max(.1, x.v), itemStyle: { color: i === keys.length - 1 ? col : T.dim, borderRadius: [0, 3, 3, 0] } })) }] }, 60 + keys.length * 24) };
  }
  if (gid === 'weather') {
    const W = FE().W, i = 0, isi = fnum('ISI'), rows = [['Fuel moisture (FFMC)', fnum('FFMC'), 30, 99, FBY.FFMC.mu, FBY.FFMC.sd], ['Duff moisture (DMC)', fnum('DMC'), 0, 200, FBY.DMC.mu, FBY.DMC.sd], ['Drought (DC)', fnum('DC'), 0, 700, FBY.DC.mu, FBY.DC.sd], ['Spread index (ISI)', isi, 0, 30, FBY.ISI.mu, FBY.ISI.sd], ['Build-up (BUI)', fnum('BUI'), 0, 200, FBY.BUI.mu, FBY.BUI.sd], ['Fire weather (FWI)', fnum('FWI'), 0, 60, FBY.FWI.mu, FBY.FWI.sd]];
    return { html: sec('The fire weather system', `<div class="meters">${rows.map(r => r[1] == null ? '' : meter(r[0], r[1], r[2], r[3], r[4], r[5], v => v.toFixed(v < 10 ? 1 : 0))).join('')}</div><p class="dim small" style="margin:6px 0 0">The pale band is the usual range. The marker is this event.</p>`, 'Canadian FWI system', 'fwi'), after: null };
  }
  if (gid === 'quality') {
    const rows = FAMILIES.map(g => { const l = FDEF.filter(d => d.g === g.id), ok = l.filter(d => !FE().by[d.key].missing).length; return { g, ok, n: l.length }; });
    return { html: sec('How complete the inputs are', `<div style="display:grid;gap:8px">${rows.map(r => `<div class="pbar" style="grid-template-columns:170px 1fr 56px;margin:0"><span class="tx2">${r.g.label}</span><span class="tr"><i style="width:${r.ok / r.n * 100}%;background:${r.ok === r.n ? 'var(--tx)' : r.ok / r.n > .6 ? 'var(--tx2)' : 'var(--warn)'}"></i></span><span class="num tx2" style="text-align:right">${r.ok} of ${r.n}</span></div>`).join('')}</div>`, 'Features available'), after: null };
  }
  return { html: '', after: null };
}
function tabGroup(gid) {
  const ch = groupChart(gid), tiles = GROUP_TILES[gid];
  let html = ftiles(tiles) + ch.html + reads(readsFor(gid));
  if (gid === 'quality') {
    const fr = freshnessOf(A.ev);
    html += sec('How fresh each source is', `<table class="tbl small"><tbody>${fr.map(f => `<tr><td>${f.name}<div class="dim fk">${f.note}</div></td><td class="r">${f.label}</td></tr>`).join('')}</tbody></table>`, 'Sample ages') + sec('Viewing geometry', `<div class="stats">${[['Scan angle', Math.abs(FE().ang.SCAN_ANGLE).toFixed(0) + '°', 'from straight down'], ['Viewing angle', FE().ang.VIEW_ZENITH.toFixed(0) + '°', FE().ang.VIEW_ZENITH > 50 ? 'far from the centre of the pass' : 'close to the centre of the pass'], ['Sensor', A.ev.sensor.replace('VIIRS ', 'V ').replace('MODIS ', 'M '), '']].map(([l, v, s]) => tile(l, v, s)).join('')}</div>`);
  }
  html += groupTable(gid);
  return { html, after: ch.after };
}
function tabAir() {
  const fr = freshnessOf(A.ev), age = n => fr.find(f => f.name.startsWith(n)).label, E = FE().by, ang = FE().ang;
  const smoke = ftiles(['SMOKE_PROBABILITY', 'DARK_SMOKE_PROBABILITY', 'LIGHT_SMOKE_PROBABILITY', 'SMOKE_LENGTH_LOG', 'BURN_SCAR_GROWTH_RATE']) + (E.SMOKE_DIRECTION_SIN.missing ? '' : `<div class="row" style="gap:14px;margin-top:10px;align-items:center">${compassSvg([{ deg: ang.SMOKE_DIRECTION, color: ccol(A.ev.cls) }])}<span class="small tx2">Smoke is drifting toward the ${compass(ang.SMOKE_DIRECTION)}.</span></div>`);
  const sar = ftiles(['VV_CHANGE', 'VH_CHANGE', 'VV_VH_CHANGE', 'COHERENCE_LOSS', 'DNBR', 'DNDVI', 'STRUCTURAL_CHANGE_PROBABILITY']);
  const zn = n => fnum(n), hints = [];
  if (zn('CO_NO2_RATIO_LOG') != null && zn('CO_NO2_RATIO_LOG') > .6) hints.push('Carbon monoxide is high next to nitrogen dioxide. That often goes with smouldering, low-efficiency burning such as crop residue or vegetation.');
  if (zn('SO2_ANOMALY_Z') != null && zn('SO2_ANOMALY_Z') > 1.2) hints.push('Sulphur dioxide is well above normal. That is more common near coal, smelting and some industrial burning.');
  if (zn('CH4_ANOMALY_Z') != null && zn('CH4_ANOMALY_Z') > 1.2) hints.push('Methane is raised, which can go with gas flaring or venting.');
  const air = `<div style="display:grid;gap:6px">${['NO2_ANOMALY_Z', 'SO2_ANOMALY_Z', 'CO_ANOMALY_Z', 'CH4_ANOMALY_Z', 'HCHO_ANOMALY_Z', 'AOD_ANOMALY_Z'].map(zrow).join('')}</div>${ftiles(['CO_NO2_RATIO_LOG', 'SO2_NO2_RATIO_LOG', 'CH4_CO_RATIO_LOG']).replace('class="stats"', 'class="stats" style="margin-top:10px"')}${hints.length ? `<ul class="reads" style="margin-top:10px">${hints.map(h => `<li>${h}</li>`).join('')}</ul><p class="dim small" style="margin:6px 0 0">These are hints, not proof. Other sources can produce the same readings.</p>` : ''}`;
  return { html: sec('Smoke seen from above', smoke, age('Optical'), 'smokeopt') + sec('Ground change, from radar', sar, age('Radar'), 'sar') + sec('What is in the air', air, age('Air'), 'chem') + groupTable('smoke') + groupTable('sar') + groupTable('chem'), after: null };
}

/* ---------- all 141 features ---------- */
function exRows() {
  const E = FE().by, sh = shapOf(A.ev).all, q = A.ex.q.trim().toLowerCase(); let rows = FDEF.filter(d => (A.ex.g === 'all' || d.g === A.ex.g) && (!q || (d.label + ' ' + d.key + ' ' + d.src).toLowerCase().includes(q)) && (A.ex.only === 'all' || (A.ex.only === 'missing' ? E[d.key].missing : A.ex.only === 'linked' ? d.fl.includes('k') : Math.abs(sh[d.key]) >= .05)));
  const keyf = { pct: d => E[d.key].pct == null || E[d.key].missing ? -1 : Math.abs(E[d.key].pct - 50), contrib: d => Math.abs(sh[d.key]), name: d => d.label, group: d => FAMILIES.findIndex(g => g.id === d.g) }[A.ex.sort];
  rows = rows.slice().sort((a, b) => A.ex.sort === 'name' ? (keyf(a) > keyf(b) ? 1 : -1) : A.ex.sort === 'group' ? keyf(a) - keyf(b) : keyf(b) - keyf(a));
  return rows.map(d => { const e = E[d.key], w = sh[d.key], gl = FAMILIES.find(g => g.id === d.g).label;
    return `<tr class="${e.missing ? 'miss' : ''}"><td><div>${esc(d.label)}</div><div class="dim fk">${d.key} <span class="gtag">${gl}</span></div></td><td class="r num">${e.missing ? '<span class="dim">no data</span>' : esc(fmtFeat(d, e.v))}</td><td style="min-width:110px">${e.pct != null && !e.missing ? `<div class="mini w"><i style="width:${e.pct.toFixed(0)}%"></i></div><div class="u sm2">${Math.round(e.pct)}th pct</div>` : '<span class="dim">n/a</span>'}</td><td class="small">${esc(d.src)}</td><td>${e.missing ? '<span class="chip rev">Missing</span>' : ''}${d.fl.includes('k') ? `<span class="chip prop" data-tip="${esc(FSRC_TIP)}">Feeds labels</span>` : ''}</td><td class="r num" style="color:${w >= .05 ? ccol(A.ev.cls) : w <= -.05 ? 'var(--tx2)' : 'var(--dim)'}">${e.missing ? '' : (w >= 0 ? '+' : '') + w.toFixed(2)}</td></tr>`; }).join('');
}
function tabExplorer() {
  const opt = (v, l, cur) => `<option value="${v}" ${cur === v ? 'selected' : ''}>${l}</option>`;
  const html = `<div class="row wrap" style="margin-bottom:10px"><div class="search" style="flex:1;min-width:150px"><span class="ic">${icon('search', 16)}</span><input id="exq" data-act="exq" placeholder="Search features" value="${esc(A.ex.q)}" aria-label="Search features"></div>
    <select class="sel" data-act="exg" aria-label="Group">${opt('all', 'All groups', A.ex.g)}${FAMILIES.map(g => opt(g.id, g.label, A.ex.g)).join('')}</select>
    <select class="sel" data-act="exo" aria-label="Show">${opt('all', 'Show all', A.ex.only)}${opt('missing', 'Only missing', A.ex.only)}${opt('linked', 'Only those that feed labels', A.ex.only)}${opt('strong', 'Only strong contributors', A.ex.only)}</select>
    <select class="sel" data-act="exs" aria-label="Sort">${opt('pct', 'Most unusual first', A.ex.sort)}${opt('contrib', 'Biggest contribution first', A.ex.sort)}${opt('name', 'Name', A.ex.sort)}${opt('group', 'Group', A.ex.sort)}</select>
    <button class="btn sm" data-act="exexport">${icon('download', 14)}Export</button></div>
    <p class="dim small" style="margin:0 0 8px">Each feature is compared with similar events. The last column is the sample contribution to the call.</p>
    <div class="exwrap"><table class="tbl small ex"><thead><tr><th>Feature</th><th class="r">Value</th><th>Against similar events</th><th>Source</th><th>Notes</th><th class="r">Pushes the call</th></tr></thead><tbody id="ex-body">${exRows()}</tbody></table></div>`;
  return { html, after: null };
}
const exRefresh = () => { const b = $('#ex-body'); if (b) b.innerHTML = exRows(); };
function exCsv() {
  const E = FE().by, sh = shapOf(A.ev).all;
  return ['key,group,label,value,percentile,source,missing,feeds_labels,contribution'].concat(FDEF.map(d => { const e = E[d.key]; return [d.key, d.g, d.label, e.missing ? '' : (e.v == null ? '' : (typeof e.v === 'number' ? +e.v.toFixed(4) : e.v)), e.pct == null ? '' : e.pct.toFixed(1), d.src, e.missing, d.fl.includes('k'), e.missing ? '' : sh[d.key].toFixed(3)].map(csvCell).join(','); })).join('\n');
}

/* ---------- who is in the way ---------- */
function tabExposure() {
  const zones = exposureOf(A.p, A.ev, A.pl, A.sp);
  const row = (l, k, z) => `<tr><td>${l}</td><td class="r num">${z[k]}</td></tr>`;
  const html = `<div class="banner plain small" style="margin-bottom:12px"><span><b>Sample counts.</b> In the live build these come from OpenStreetMap features inside each zone, and population from a gridded population dataset. OSM is patchy in rural India, so treat low numbers as a floor.</span></div>
  ${zones.map(z => sec(esc(z.name), `<div class="stats">${tile('Area', z.area.toFixed(1) + ' <span class="u">km²</span>')}${tile('Villages and towns', z.settlements)}${tile('Schools', z.schools)}${tile('Clinics and hospitals', z.clinics)}${tile('Main roads', z.roadKm + ' <span class="u">km</span>')}${tile('Water bodies', z.water)}</div>`)).join('')}
  <p class="dim small" style="margin-top:12px">The smoke zone is where smoke can travel, so it says who might smell it, not who is at risk. It says nothing about how thick the smoke is.</p>`;
  return { html, after: null };
}
/* ---------- similar events ---------- */
function tabSimilar() {
  const list = similarOf(A.ev);
  const html = `<div class="banner plain small" style="margin-bottom:12px"><span><b>Sample list.</b> The live build finds these by comparing the full feature vector against past events with a known outcome.</span></div>
  ${list.length ? `<div class="queue">${list.map(x => `<button class="qi" data-act="advgo" data-id="${x.p.id}" style="grid-template-columns:auto minmax(0,1fr) auto"><span>${ring(x.sim, 40, 4, ccol(x.e.cls))}</span><span style="min-width:0"><span class="t" style="display:block">${esc(x.p.name)}</span><span class="s" style="display:block">${fmtD(x.e.t)}, ${esc(x.p.district)}, ${esc(x.p.state)}</span><span class="chips">${cbadge(x.e.cls)}<span class="chip ${x.outcome.startsWith('Marked') ? 'rev' : ''}">${x.outcome}</span></span></span><span class="m"><b class="num">${x.e.frp.toFixed(0)} MW</b><span>${Math.round(x.sim * 100)}% alike</span></span></button>`).join('')}</div>` : '<div class="empty"><b>Nothing earlier to compare with</b>This is the first event of its kind in the sample.</div>'}`;
  return { html, after: null };
}
/* ---------- analyst review ---------- */
let REVIEWS = {}; try { REVIEWS = JSON.parse(store.get('agni-reviews') || '{}') || {}; } catch (e) { REVIEWS = {}; }
const reviewsOf = id => REVIEWS[id] || [];
const reviewedChip = e => reviewsOf(e.id).length ? '<span class="chip impl" data-tip="An analyst has reviewed this event.">Reviewed</span>' : '';
function saveReviews() { store.set('agni-reviews', JSON.stringify(REVIEWS)); }
function reviewsCsv() {
  return ['event_id,place,model_class,model_confidence,verdict,analyst_class,note,saved_at'].concat(Object.entries(REVIEWS).flatMap(([id, l]) => { const ev = EVENTS.find(e => e.id === id); if (!ev) return []; return l.map(r => [id, PMAP[ev.pid].name, ev.cls, ev.conf.toFixed(3), r.verdict, r.cls || '', r.note || '', r.at].map(csvCell).join(',')); })).join('\n');
}
const RV = [['confirm', 'Confirm the model’s class'], ['change', 'Change the class'], ['false', 'This is a false alarm'], ['field', 'Needs a field check']];
function tabReview() {
  const ev = A.ev, list = reviewsOf(ev.id), total = Object.values(REVIEWS).reduce((a, l) => a + l.length, 0);
  const html = `<p class="sum" style="margin:0 0 12px">The model said <b>${esc(CLSMAP[ev.cls].label)}</b>, ${Math.round(ev.conf * 100)}% sure${ev.review ? ', and flagged it for a person to check' : ''}. Your call is saved on this device. Confirmed and corrected calls can be exported and used as real training labels later.</p>
  <section class="card"><div class="cbody"><fieldset class="rv"><legend class="sr">Your verdict</legend>${RV.map(([v, l], i) => `<label class="rvopt"><input type="radio" name="rv" value="${v}" ${i === 0 ? 'checked' : ''} data-act="rvpick"><span>${l}</span></label>`).join('')}</fieldset>
    <div id="rv-cls" style="display:none;margin:8px 0"><label class="dim small" for="rv-sel">Should be</label><br><select id="rv-sel" class="sel">${CLS.map(c => `<option value="${c.id}" ${c.id === ev.cls2 ? 'selected' : ''}>${c.label}</option>`).join('')}</select></div>
    <label class="dim small" for="rv-note" style="display:block;margin-top:10px">Note</label><textarea id="rv-note" rows="4" placeholder="What did you see? For example: the satellite view shows a new pad next to the stack."></textarea>
    <div class="row wrap" style="margin-top:10px"><button class="btn pri" data-act="rvsave">Save my review</button><span class="dim small" id="rv-msg"></span></div></div></section>
  ${list.length ? sec('Earlier reviews of this event', `<div class="dist small">${list.slice().reverse().map(r => `<div><span><b>${esc(RV.find(x => x[0] === r.verdict)[1])}</b>${r.cls ? ', as ' + esc(CLSMAP[r.cls].label) : ''}${r.note ? `<br><span class="dim">${esc(r.note)}</span>` : ''}</span><span class="dim">${esc(r.at)}</span></div>`).join('')}</div>`) : ''}
  <div class="row between" style="margin-top:14px"><span class="dim small">${total} review${total === 1 ? '' : 's'} saved on this device</span><button class="btn sm" data-act="rvexport" ${total ? '' : 'disabled'}>${icon('download', 14)}Export all as CSV</button></div>`;
  return { html, after: null };
}

const ATAB = { wind: tabWind, thermal: () => tabGroup('thermal'), temporal: () => tabGroup('temporal'), spread: () => tabGroup('spread'), terrain: () => tabGroup('terrain'), fuel: () => tabGroup('fuel'), industry: () => tabGroup('industry'), weather: () => tabGroup('weather'), air: tabAir, quality: () => tabGroup('quality'), explorer: tabExplorer, exposure: tabExposure, similar: tabSimilar, review: tabReview };
