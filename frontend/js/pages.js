'use strict';
/* ===== Chart helpers ===== */
const CH = [], AF = [];
function chart(target, opt, h) {
  const el = typeof target === 'string' ? $(target) : target; if (!el) return null;
  if (!window.echarts) { el.innerHTML = '<div class="empty">Charts could not load in this view.</div>'; return null; }
  if (h) el.style.height = h + 'px';
  const c = echarts.init(el, null, { renderer: 'svg' });
  c.setOption(Object.assign({ textStyle: { fontFamily: cv('--font') }, animationDuration: 350 }, opt)); CH.push(c); return c;
}
function disposeCharts() { CH.splice(0).forEach(c => { try { c.dispose(); } catch (e) {} }); }
const tt = () => { const T = tk(); return { backgroundColor: T.panel, borderColor: T.line2, textStyle: { color: T.tx, fontSize: 12 }, extraCssText: 'box-shadow:0 8px 24px rgba(0,0,0,.3);border-radius:8px' }; };
const axb = () => { const T = tk(); return { axisLine: { lineStyle: { color: T.line2 } }, axisTick: { show: false }, axisLabel: { color: T.dim, fontSize: 11 }, splitLine: { lineStyle: { color: T.line } } }; };
const card = (title, body, o = {}) => `<section class="card ${o.cls || ''}"><div class="ch"><h3>${title}</h3>${o.info ? info(o.info) : ''}<span class="r">${o.right || ''}</span></div><div class="cbody">${body}</div></section>`;
const pl = (n, w) => n === 1 ? w : w + 's';
const layout2 = (L, R) => S.expanded ? `<div class="rep-cols"><div>${L.join('')}</div><div>${R.join('')}</div></div>` : L.concat(R).join('');
const km = v => v > 10 ? 'over 10 km' : v.toFixed(1) + ' km';

/* ===== CSV / GeoJSON / CAP ===== */
const csvCell = v => { v = v == null ? '' : String(v); return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; };
function eventsCsv(evs) {
  const head = ['event_id', 'place', 'district', 'state', 'lat', 'lon', 'time_ist', 'class', 'confidence', 'second_class', 'status', 'needs_review', 'peak_frp_mw', 'robust_z', 'sensor', 'detections', 'footprint_km2'];
  return [head.join(',')].concat(evs.map(e => { const p = PMAP[e.pid]; return [e.id, p.name, p.district, p.state, p.lat.toFixed(4), p.lon.toFixed(4), fmtDT(e.t), e.cls, e.conf.toFixed(3), e.cls2, e.status, e.review, e.frp.toFixed(1), e.z == null ? '' : e.z.toFixed(2), e.sensor, e.nDet, e.area].map(csvCell).join(','); })).join('\n');
}
function eventsGeo(evs) { return JSON.stringify({ type: 'FeatureCollection', features: evs.map(e => { const p = PMAP[e.pid]; return { type: 'Feature', geometry: { type: 'Point', coordinates: [+p.lon.toFixed(4), +p.lat.toFixed(4)] }, properties: { id: e.id, place: p.name, class: e.cls, confidence: +e.conf.toFixed(3), status: e.status, needs_review: e.review, frp_mw: +e.frp.toFixed(1), time: fmtDT(e.t), sample_data: true } }; }) }, null, 1); }
const xe = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' }[c]));
function capXml(ev) {
  const p = PMAP[ev.pid], d = dparts(TODAY), sent = `${d.y}-${pad(d.m + 1)}-${pad(d.d)}T${pad(d.h)}:${pad(d.mi)}:00+05:30`;
  return `<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>AGNI-${ev.id}-DRAFT</identifier>
  <sender>sample@agni-netra.invalid</sender>
  <sent>${sent}</sent>
  <status>Draft</status>
  <msgType>Alert</msgType>
  <scope>Restricted</scope>
  <restriction>For authorized alerting agencies only</restriction>
  <info>
    <language>en-IN</language>
    <category>Fire</category>
    <event>${xe(STATUS[ev.status].label + ' thermal source: ' + CLSMAP[ev.cls].label.toLowerCase())}</event>
    <urgency>Unknown</urgency>
    <severity>Unknown</severity>
    <certainty>Possible</certainty>
    <headline>${xe('Possible ' + CLSMAP[ev.cls].label.toLowerCase() + ' event near ' + p.district + ', ' + p.state)}</headline>
    <description>${xe('Satellite thermal detection classified as ' + CLSMAP[ev.cls].label.toLowerCase() + ' with ' + Math.round(ev.conf * 100) + '% calibrated confidence. Peak fire power ' + ev.frp.toFixed(0) + ' MW. Automated classification, to be verified.')}</description>
    <instruction>For review by the authorized alerting authority. Not for public release.</instruction>
    <area>
      <areaDesc>${xe(p.district + ', ' + p.state)}</areaDesc>
      <circle>${p.lat.toFixed(4)},${p.lon.toFixed(4)} 2.0</circle>
    </area>
  </info>
</alert>`;
}

/* ===== Place report ===== */
function headlineOf(p, ev) {
  if (ev.status === 'abnormal') return 'Burning hotter than usual, and worth a look.';
  if (ev.review) return 'Not sure about this one. A person should take a look.';
  if (ev.status === 'baseline_building') return 'New here, so it is too early to say what normal looks like.';
  return { wildfire: 'A short-lived vegetation fire.', agricultural_burning: 'Looks like crop burning.', gas_flare: 'A steady flare, behaving the way it usually does.', industrial: 'Ordinary industrial heat. Nothing unusual today.', mining: 'Heat from the mine, in line with its history.', unknown: 'Hard to say what this is.' }[ev.cls];
}
function summaryText(p, ev, st, ser) {
  if (p.kind === 'transient') return `This burned for a short spell: ${ev.nDet} detections over ${ev.area} km². Fires like this have no normal level to compare with, so the call rests on what is around it and how it behaves.`;
  if (ev.status === 'baseline_building') return `First spotted ${fmtD(p.firstSeen)}. There are only ${ser.obs.filter(o => o.t <= ev.t).length} clear looks so far, so nothing is claimed yet about whether this is normal.`;
  if (ev.status === 'abnormal') return `Its fire power is ${ev.z.toFixed(1)} steps above this site's usual level. That is unusual for this place, so a person should look at it.`;
  return `A long-running ${lc(CLSMAP[ev.cls].label)} source. It has been active on ${st.d90} of the last 90 days and sits within its usual range${ev.z != null ? ` (${ev.z >= 0 ? '+' : ''}${ev.z.toFixed(1)})` : ''}.`;
}
function headerCard(p, ev, st) {
  const w = S.watch.has(p.id), last = (EVP[p.id] || []).filter(e => e.t <= S.asOf).pop() || ev;
  return `<section class="card"><div class="cbody">
  <div class="row between" style="align-items:flex-start;gap:12px"><div style="min-width:0"><h2 style="margin:0;font-size:18px;font-weight:600">${esc(p.name)}</h2><div class="dim" style="margin-top:2px">${esc(p.district)}, ${esc(p.state)}</div></div>
  <button class="star" data-act="watch" data-id="${p.id}" aria-pressed="${w}" aria-label="${w ? 'Remove from' : 'Add to'} watchlist">${icon('star', 20)}</button></div>
  <div class="row wrap" style="margin:10px 0 12px"><span class="chip">${esc(p.type)}</span>${schip(ev.status)}${ev.review ? rchip() : ''}${reviewedChip(ev)}</div>
  <dl class="kv small"><dt>Coordinates</dt><dd class="num">${p.lat.toFixed(3)}°N, ${p.lon.toFixed(3)}°E</dd><dt>${p.kind === 'site' ? 'Site code' : 'Kind'}</dt><dd>${p.kind === 'site' ? esc(p.code) : 'Transient event, no fixed site'}</dd><dt>First seen</dt><dd>${fmtD(p.firstSeen)}</dd><dt>Last event</dt><dd>${fmtDT(last.t)}</dd><dt>Facility record</dt><dd>${p.kind === 'site' && p.cls !== 'unknown' ? 'Matched to a facility (sample)' : p.type === 'Landfill' ? 'Matched to a landfill (sample)' : 'No facility match'}</dd></dl>
  <div class="row wrap" style="margin-top:12px"><button class="btn pri sm" data-act="adv" data-id="${p.id}">${icon('layers', 14)}Advanced analysis</button><button class="btn sm" data-act="copylink">${icon('copy', 14)}Copy link</button><button class="btn sm" data-act="exportcsv" data-id="${p.id}">${icon('download', 14)}Export events</button></div></div></section>`;
}
function verdictCard(p, ev, st, ser) {
  const c = ccol(ev.cls), c2 = ccol(ev.cls2), rest = Math.max(0, 1 - ev.conf - ev.p2);
  const bar = (l, v, col) => `<div class="pbar"><span>${l}</span><span class="tr"><i style="width:${(v * 100).toFixed(1)}%;background:${col}"></i></span><span class="num tx2" style="text-align:right">${(v * 100).toFixed(0)}%</span></div>`;
  return card('What we think this is', `<p class="headline">${esc(headlineOf(p, ev))}</p><div class="verdict"><div style="color:var(--tx)">${ring(ev.conf, 96, 8, ev.review ? cv('--warn') : c)}</div><div><div style="font-size:19px;font-weight:600;line-height:1.2">${cbadge(ev.cls).replace('cb', 'cb" style="font-size:19px')}</div><div class="dim small" style="margin-top:3px">Calibrated confidence ${info('conf')}</div><div class="dim small">${esc(CLSMAP[ev.cls].blurb)}</div></div></div>
  <div style="margin-top:12px">${bar(esc(CLSMAP[ev.cls].label), ev.conf, c)}${bar(esc(CLSMAP[ev.cls2].label), ev.p2, c2)}${bar('Everything else', rest, 'var(--dim)')}</div>
  ${ev.review ? `<div class="banner">${icon('search', 16)}<span><b>Needs human review.</b> Confidence is below 60%, so a person should check this before anyone acts on it.</span></div>` : ''}
  ${ev.status === 'abnormal' ? `<div class="banner abn"><span><b>Abnormal for this site.</b> Robust z-score ${ev.z.toFixed(1)}, above the 3.5 threshold.</span></div>` : ''}
  <p class="sum">${summaryText(p, ev, st, ser)}</p>`, { right: 'Event ' + ev.id });
}
function whyCard(ev, sh) {
  const mx = Math.max(...FAMILIES.map(f => Math.abs(sh.fam[f.id])), .01), c = ccol(ev.cls);
  const rows = FAMILIES.map(f => ({ f, v: sh.fam[f.id] })).sort((a, b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, 6).map(({ f, v }) => {
    const w = Math.abs(v) / mx * 50;
    return `<div class="shr"><span class="tx2">${f.label}</span><span class="ax"><i style="${v >= 0 ? `left:50%;width:${w}%;background:${c}` : `right:50%;width:${w}%;background:var(--dim)`}"></i></span><span class="vv">${v >= 0 ? '+' : ''}${v.toFixed(2)}</span></div>`;
  }).join('');
  const list = S.shapAll ? `<div class="flist">${sh.feats.map(f => `<div class="frow"><span>${esc(f.l)}</span><span class="fv">${esc(f.v)}</span><span class="tnum" style="color:${f.w >= 0 ? c : 'var(--tx2)'}">${f.w >= 0 ? '+' : ''}${f.w.toFixed(2)}</span></div>`).join('')}</div>` : '';
  return card('What drove that call', `<div class="shap">${rows}</div><button class="btn sm" data-act="shapall" style="margin-top:12px">${S.shapAll ? 'Hide top features' : 'Show top 8 features'}</button>${list}`, { info: 'shap', right: 'Sample explanation' });
}
function timelineCard(p, ev, ser) {
  AF.push(() => chTimeline(p, ev, ser));
  const t = p.kind === 'site' ? `<div class="legendrow"><span><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="${ccol(p.cls)}"/></svg>VIIRS pass</span><span><svg width="10" height="10"><path d="M5 0L10 5L5 10L0 5Z" fill="${ccol(p.cls)}"/></svg>MODIS pass</span><span><i class="sw" style="background:${alpha(ccol(p.cls), .3)}"></i>Normal range</span><span><i class="sw" style="background:var(--danger);border-radius:50%"></i>Unusual</span><span><i class="sw" style="background:var(--raised);border:1px dashed var(--dim)"></i>No clear view ${info('gap')}</span></div>` : `<div class="legendrow"><span>Transient events show only the days the fire was seen.</span></div>`;
  return card('Heat over the past year', `<div id="tl" class="chart" style="height:270px"></div>${t}`, { info: 'frp', right: 'Fire radiative power, log scale' });
}
function chTimeline(p, ev, ser) {
  const T = tk(), col = ccol(p.cls), obs = ser.obs, el = $('#tl'); if (!el) return;
  if (!obs.length) { el.innerHTML = '<div class="empty">No detections yet.</div>'; return; }
  const site = p.kind === 'site', b = ser.base, useB = site && b && b.n >= 20, abn = o => o.z != null && o.z > 3.5;
  const data = obs.map(o => ({ value: [o.t, +o.frp.toFixed(2)], symbol: o.sensor.startsWith('MODIS') ? 'diamond' : 'circle', symbolSize: abn(o) ? 11 : 6.5, itemStyle: { color: abn(o) ? T.danger : col, borderColor: T.panel, borderWidth: 1 }, o }));
  const mkA = [], mkL = [];
  if (useB) {
    mkA.push([{ yAxis: b.lo, itemStyle: { color: alpha(col, .11) } }, { yAxis: b.hi }]);
    mkL.push({ yAxis: b.med, lineStyle: { color: col, type: 'dashed', width: 1 }, label: { formatter: 'Median', color: T.dim, position: 'insideStartTop', fontSize: 10 } }, { yAxis: b.p95, lineStyle: { color: T.dim, type: 'dotted', width: 1 }, label: { formatter: 'P95', color: T.dim, position: 'insideStartTop', fontSize: 10 } });
  }
  (ser.gaps || []).forEach(g => mkA.push([{ xAxis: g[0], itemStyle: { color: alpha(T.dim, .12), borderColor: alpha(T.dim, .55), borderType: 'dashed', borderWidth: 1 } }, { xAxis: g[1] }]));
  mkL.push({ xAxis: ev.t, lineStyle: { color: T.tx, width: 1 }, label: { formatter: 'Selected event', color: T.tx, position: 'insideEndTop', fontSize: 10 } });
  const fr = obs.map(o => o.frp), ymin = Math.pow(10, Math.floor(Math.log10(Math.max(.4, Math.min(...fr, useB ? b.lo : Infinity) * .85)))), ymax = Math.pow(10, Math.ceil(Math.log10(Math.max(...fr, useB ? b.hi : 0) * 1.15)));
  const z0 = Math.min(TODAY - 150 * DAY, ev.t - 60 * DAY);
  const opt = {
    grid: { left: 50, right: 16, top: 22, bottom: site ? 50 : 28 },
    tooltip: Object.assign(tt(), { trigger: 'item', formatter: q => { const o = q.data.o; return `<b>${fmtD(o.t)}</b><br>${o.frp.toFixed(1)} MW, ${o.sensor}<br>${o.cnt} detection${o.cnt > 1 ? 's' : ''}${o.z != null ? `<br>Deviation ${o.z >= 0 ? '+' : ''}${o.z.toFixed(1)}` : ''}`; } }),
    xAxis: Object.assign({ type: 'time' }, axb(), site ? {} : { min: obs[0].t - 5 * DAY, max: obs[obs.length - 1].t + 5 * DAY }),
    yAxis: Object.assign({ type: 'log', min: ymin, max: ymax, name: 'MW', nameTextStyle: { color: T.dim, fontSize: 11, align: 'left' } }, axb()),
    series: [{ type: 'scatter', data, markArea: { silent: true, data: mkA }, markLine: { silent: true, symbol: 'none', data: mkL } }]
  };
  if (site) opt.dataZoom = [{ type: 'inside', startValue: z0, endValue: TODAY, filterMode: 'none' }, { type: 'slider', height: 16, bottom: 6, startValue: z0, endValue: TODAY, filterMode: 'none', borderColor: T.line, backgroundColor: 'transparent', fillerColor: alpha(T.tx, .08), handleSize: 12, textStyle: { color: T.dim, fontSize: 10 }, dataBackground: { lineStyle: { color: T.line2 }, areaStyle: { color: T.line } } }];
  chart(el, opt);
}
function fingerprintCard(p, st, ser) {
  AF.push(() => {
    const T = tk(), col = ccol(p.cls), m = new Map(); ser.obs.forEach(o => { const k = iso(o.t); m.set(k, (m.get(k) || 0) + o.cnt); });
    chart('#cal', { tooltip: Object.assign(tt(), { formatter: q => `${q.data[0]}<br>${q.data[1]} detection${q.data[1] > 1 ? 's' : ''}` }), visualMap: { show: false, min: 0, max: 5, inRange: { color: [alpha(col, .3), col] } }, calendar: { top: 20, left: 26, right: 6, bottom: 2, cellSize: ['auto', 11], range: [iso(START), iso(TODAY)], itemStyle: { borderWidth: 2, borderColor: T.panel, color: T.raised }, splitLine: { show: false }, yearLabel: { show: false }, monthLabel: { color: T.dim, fontSize: 10, nameMap: MONTHS }, dayLabel: { firstDay: 1, color: T.dim, fontSize: 9, margin: 6 } }, series: [{ type: 'heatmap', coordinateSystem: 'calendar', data: [...m.entries()] }] }, 132);
    const H = hoursOf(p);
    chart('#hrs', { grid: { left: 34, right: 8, top: 8, bottom: 22 }, tooltip: Object.assign(tt(), { trigger: 'axis', formatter: q => `${q[0].axisValue}:00 to ${q[0].axisValue}:59<br>${q[0].data.value} detections` }), xAxis: Object.assign({ type: 'category', data: H.map((_, i) => i) }, axb(), { axisLabel: { color: T.dim, fontSize: 10, interval: 3 } }), yAxis: Object.assign({ type: 'value' }, axb(), { minInterval: 1 }), series: [{ type: 'bar', barWidth: '70%', data: H.map((v, i) => ({ value: v, itemStyle: { color: (i < 6 || i >= 18) ? T.dim : col, borderRadius: [2, 2, 0, 0] } })) }] }, 110);
  });
  const tile = (l, v, u) => `<div class="stat"><div class="l">${l}</div><div class="v">${v}${u ? ` <span class="u">${u}</span>` : ''}</div></div>`;
  return card('How it behaves', `<div id="cal" class="chart"></div>
  <div class="stats" style="margin:10px 0">${tile('Active, last 30 d', st.d30, pl(st.d30, 'day'))}${tile('Active, last 90 d', st.d90, pl(st.d90, 'day'))}${tile('Current streak', st.streak, pl(st.streak, 'day'))}${tile('Months active', st.months, 'of 12')}${tile('Since previous', st.sinceLast == null ? 'n/a' : st.sinceLast, st.sinceLast == null ? '' : pl(st.sinceLast, 'day'))}${tile('Detections', ser.obs.reduce((a, o) => a + o.cnt, 0), '12 mo')}</div>
  <div class="dim small" style="margin-bottom:2px">When in the day it is seen (local hour of satellite pass) ${info('gap')}</div><div id="hrs" class="chart"></div>
  <div class="legendrow"><span><i class="sw" style="background:var(--dim)"></i>Night passes</span><span><i class="sw" style="background:${ccol(p.cls)}"></i>Day passes</span></div>`, { right: 'Last 12 months' });
}
function baselineCard(p, ev, ser) {
  if (p.kind === 'transient') return card('What is normal here', `<p class="dim" style="margin:0">Not applicable. Wildfires and crop burning are short events with no normal level, so they are judged on footprint, land cover and season instead.</p>`, { info: 'baseline', right: schip('not_applicable') });
  const b = ser.base, nAt = ser.obs.filter(o => o.t <= ev.t).length, ok = nAt >= 20, last = ser.obs[ser.obs.length - 1];
  const tile = (l, v, u, tip) => `<div class="stat"><div class="l">${l}${tip ? ' ' + info(tip) : ''}</div><div class="v">${v}${u ? ` <span class="u">${u}</span>` : ''}</div></div>`;
  return card('What is normal here', `<div class="row between small"><span>${ok ? 'Baseline trusted' : 'Baseline building'}</span><span class="num tx2">${Math.min(nAt, 20)} of 20 clear observations needed</span></div><div class="prog" style="margin:6px 0 12px"><i style="width:${Math.min(1, nAt / 20) * 100}%"></i></div>
  ${ok ? `<div class="stats">${tile('Median', b.med.toFixed(1), 'MW')}${tile('Top of normal', b.hi.toFixed(1), 'MW')}${tile('95th percentile', b.p95.toFixed(1), 'MW')}${tile('Normal variation', b.mad.toFixed(2), 'log MAD', 'mad')}${tile('Observations', b.n, '')}${tile('Deviation now', last && last.z != null ? (last.z >= 0 ? '+' : '') + last.z.toFixed(1) : 'n/a', '', 'z')}</div>` : `<p class="dim small" style="margin:0">Until 20 clear observations are in, this place is shown as baseline building and no claim is made about normal or abnormal.</p>`}`, { info: 'baseline', right: schip(ev.status) });
}
function footprintCard(p, ev) {
  AF.push(() => {
    const T = tk(), r = R('fp:' + ev.id), grow = p.cls === 'wildfire' || ev.status === 'abnormal', n = 8;
    const d = Array.from({ length: n }, (_, i) => [ev.t - (n - 1 - i) * 12 * HOUR, +(ev.area * (grow ? .3 + .7 * i / (n - 1) : .85 + r() * .3)).toFixed(2)]), col = ccol(p.cls);
    chart('#fp', { grid: { left: 42, right: 12, top: 10, bottom: 24 }, tooltip: Object.assign(tt(), { trigger: 'axis', formatter: q => `${fmtDT(q[0].data[0])}<br>${q[0].data[1]} km²` }), xAxis: Object.assign({ type: 'time' }, axb(), { axisLabel: { color: T.dim, fontSize: 10, formatter: v => fmtS(v) } }), yAxis: Object.assign({ type: 'value', name: 'km²', nameTextStyle: { color: T.dim, fontSize: 10, align: 'left' } }, axb()), series: [{ type: 'line', data: d, smooth: .25, symbolSize: 6, lineStyle: { color: col, width: 2 }, itemStyle: { color: col }, areaStyle: { color: alpha(col, .14) } }] }, 130);
  });
  return card('How big it has been', `<div id="fp" class="chart"></div><div class="dim small" style="margin-top:4px">Area covered by neighbouring detections in the last 8 passes. Observed, not predicted.</div>`, { info: 'footprint', right: 'Observed' });
}
const LC_COL = { 'Cropland': '#B9A45B', 'Forest': '#4F8F6B', 'Bare ground': '#9C8A78', 'Industrial or built-up': '#6F86A6', 'Grass and scrub': '#8FAF6B', 'Water and other': '#5B8DB3' };
function surroundCard(p, ctx) {
  AF.push(() => { const T = tk(); chart('#lc', { tooltip: Object.assign(tt(), { formatter: q => `${q.name}: ${q.value}%` }), series: [{ type: 'pie', radius: ['56%', '82%'], avoidLabelOverlap: true, label: { show: false }, itemStyle: { borderColor: T.panel, borderWidth: 2 }, data: ctx.parts.map(([n, v]) => ({ name: n, value: v, itemStyle: { color: LC_COL[n] } })) }] }, 150); });
  const warn = p.cover !== 'good' ? `<div class="banner plain" style="margin-top:12px"><span><b>${p.cover === 'weak' ? 'Facility records are thin here.' : 'Facility records are partly complete here.'}</b> A missing nearby facility may simply not be mapped. ${info('cover')}</span></div>` : '';
  const d = ctx.d;
  return card('What is around it', `<div class="row" style="gap:16px;align-items:center"><div id="lc" style="width:150px;flex:none"></div><div style="flex:1;display:grid;gap:4px" class="small">${ctx.parts.map(([n, v]) => `<div class="row between"><span class="row"><i class="sw" style="background:${LC_COL[n]}"></i>${n}</span><span class="num tx2">${v}%</span></div>`).join('')}</div></div>
  <div class="dim small" style="margin:12px 0 4px">Nearest facilities</div><div class="dist small">${[['Refinery or petrochemical', d.refinery], ['Power plant', d.power], ['Steel or cement plant', d.steel], ['Mine', d.mine], ['Landfill', d.landfill]].map(([l, v]) => `<div><span>${l}</span><span class="num">${km(v)}</span></div>`).join('')}</div>${warn}`, { right: 'ESA WorldCover, GEM, OSM' });
}
function drawScene(cvs, kind, seed, after) {
  const w = cvs.width = 360, h = cvs.height = 190, g = cvs.getContext('2d'), r = R(seed);
  if (!g) return;
  const pal = { forest: ['#1f3d2b', '#3a6b47'], cropland: ['#a9a35f', '#7f9a55'], industrial: ['#59626d', '#7b848e'], bare: ['#a08a6c', '#c4ac88'], mining: ['#7d5f47', '#a98866'] }[kind];
  const off = document.createElement('canvas'); off.width = 36; off.height = 19; const og = off.getContext('2d');
  for (let y = 0; y < 19; y++) for (let x = 0; x < 36; x++) { og.fillStyle = hexMix(pal[0], pal[1], r()); og.fillRect(x, y, 1, 1); }
  g.imageSmoothingEnabled = true; g.drawImage(off, 0, 0, w, h);
  if (kind === 'cropland') { for (let i = 0; i < 26; i++) { g.fillStyle = hexMix(r() < .5 ? '#c9c07a' : '#6f9250', '#4a5a3a', r() * .35); g.globalAlpha = .55; g.fillRect(r() * w, r() * h, r.range(30, 80), r.range(18, 46)); } g.globalAlpha = 1; }
  if (kind === 'industrial') { g.fillStyle = '#c4cad1'; for (let i = 0; i < 14; i++) g.fillRect(r() * w * .9 + 12, r() * h * .8 + 14, r.range(14, 42), r.range(9, 24)); g.fillStyle = '#e4e8ec'; for (let i = 0; i < 5; i++) { g.beginPath(); g.arc(r() * w * .8 + 30, r() * h * .7 + 24, r.range(6, 12), 0, 7); g.fill(); } g.strokeStyle = '#3e454d'; g.lineWidth = 3; g.beginPath(); g.moveTo(0, h * .7); g.lineTo(w, h * .55); g.stroke(); }
  if (kind === 'mining') { g.strokeStyle = 'rgba(40,25,15,.4)'; g.lineWidth = 2; for (let i = 1; i < 8; i++) { g.beginPath(); g.ellipse(w * .5, h * .5, i * 22, i * 12, .2, 0, 7); g.stroke(); } }
  if (kind === 'forest') { g.fillStyle = 'rgba(10,30,18,.4)'; for (let i = 0; i < 40; i++) { g.beginPath(); g.arc(r() * w, r() * h, r.range(6, 18), 0, 7); g.fill(); } }
  if (after) { const gr = g.createRadialGradient(w * .5, h * .5, 4, w * .5, h * .5, 70); gr.addColorStop(0, 'rgba(18,12,10,.85)'); gr.addColorStop(1, 'rgba(18,12,10,0)'); g.fillStyle = gr; g.fillRect(0, 0, w, h); const gl = g.createRadialGradient(w * .5, h * .5, 0, w * .5, h * .5, 16); gl.addColorStop(0, 'rgba(255,255,255,.95)'); gl.addColorStop(1, 'rgba(255,255,255,0)'); g.fillStyle = gl; g.fillRect(w * .5 - 20, h * .5 - 20, 40, 40); }
}
function imageryCard(p, ev, ctx) {
  const kind = { wildfire: 'forest', agricultural_burning: 'cropland', gas_flare: 'industrial', industrial: 'industrial', mining: 'mining', unknown: 'bare' }[p.cls];
  AF.push(() => {
    const a = $('#im-a'), b = $('#im-b'), inp = $('#im-r'), box = $('#im-box'); if (!a || !b) return;
    drawScene(a, kind, 'im:' + ev.id, false); drawScene(b, kind, 'im:' + ev.id, true);
    const set = v => { b.style.clipPath = `inset(0 0 0 ${v}%)`; $('#im-h').style.left = v + '%'; }; set(50);
    inp.addEventListener('input', () => set(+inp.value));
  });
  const tile = (l, v) => `<div class="stat"><div class="l">${l}</div><div class="v">${v}</div></div>`;
  return card('Before and after', `<div class="compare" id="im-box" role="img" aria-label="Before and after image slider, illustrative"><canvas id="im-a"></canvas><canvas id="im-b"></canvas><div class="hd" id="im-h"></div><span class="lb" style="left:8px">Before ${fmtS(ev.t - ctx.sceneDays * DAY)}</span><span class="lb" style="right:8px">After ${fmtS(ev.t)}</span><input id="im-r" type="range" min="2" max="98" value="50" aria-label="Drag to compare before and after"></div>
  <div class="dim small" style="margin:6px 0 10px">Illustrative placeholder, not real imagery. The live build shows the clearest Harmonized Landsat Sentinel-2 scene before the event.</div>
  <div class="stats">${tile('Vegetation (NDVI)', ctx.ndvi.toFixed(2))}${tile('Burn ratio (NBR)', ctx.nbr.toFixed(2))}${tile('Built-up (NDBI)', ctx.ndbi.toFixed(2))}${tile('Clear pixels', Math.round(ctx.valid * 100) + '%')}${tile('Scene age', ctx.sceneDays + ' d')}${tile('Built-up share', Math.round(ctx.built * 100) + '%')}</div>`, { right: 'Pre-event scene' });
}
function conditionsCard(ctx) {
  const tile = (l, v, u, tip) => `<div class="stat"><div class="l">${l}${tip ? ' ' + info(tip) : ''}</div><div class="v">${v} <span class="u">${u}</span></div></div>`;
  return card('Weather and ground', `<div class="stats">${tile('Air dryness', ctx.vpd.toFixed(1), 'kPa', 'vpd')}${tile('Rain, last 72 h', ctx.rain.toFixed(0), 'mm')}${tile('Elevation', ctx.elev, 'm')}${tile('Slope', ctx.slope.toFixed(1), '°')}</div>`, { right: 'Weather and terrain' });
}
function nearbyCard(p) {
  const nb = nearbyOf(p), W = 250, C = W / 2, sc = 100 / 10;
  const rings = [2, 5, 10].map(k => `<circle cx="${C}" cy="${C}" r="${k * sc}" fill="none" stroke="var(--line2)" stroke-dasharray="3 4"/><text x="${C + 3}" y="${C - k * sc - 3}" style="fill:var(--dim);font-size:10px">${k} km</text>`).join('');
  const dots = nb.map(n => { const x = C + Math.cos(n.ang) * n.km * sc, y = C + Math.sin(n.ang) * n.km * sc; return `<g transform="translate(${x.toFixed(1)} ${y.toFixed(1)})">${n.st === 'abnormal' ? `<circle r="9" fill="none" stroke="var(--danger)" stroke-width="1.6"/>` : ''}<path d="${glyphPath(n.cls, 5)}" ${n.cls === 'unknown' ? `fill="none" stroke="${ccol(n.cls)}" stroke-width="1.5"` : `fill="${ccol(n.cls)}"`}/></g>`; }).join('');
  return card('Other heat nearby', `<div class="row" style="gap:14px;align-items:flex-start;flex-wrap:wrap"><svg width="${W}" height="${W}" viewBox="0 0 ${W} ${W}" style="max-width:100%;flex:none" role="img" aria-label="Sources within 10 kilometres">${rings}<circle cx="${C}" cy="${C}" r="7" fill="var(--tx)"/>${dots}</svg>
  <div style="flex:1;min-width:150px" class="small">${nb.length ? nb.map(n => `<div class="row between" style="padding:5px 0;border-bottom:1px solid var(--line)"><span class="row">${glyph(n.cls, 12)}${esc(n.name)}</span><span class="num tx2">${n.km} km${n.st === 'abnormal' ? ' , abnormal' : ''}</span></div>`).join('') : '<div class="dim">No other thermal sources within 10 km.</div>'}</div></div>`, { right: 'Within 10 km (sample)' });
}
function historyCard(p, ev) {
  const evs = (EVP[p.id] || []).slice().sort((a, b) => b.t - a.t).slice(0, 12);
  return card('Past events here', `<table class="tbl small"><thead><tr><th>Date</th><th>Class</th><th>Status</th><th class="r">Conf.</th><th class="r">Peak</th></tr></thead><tbody>${evs.map(e => `<tr class="click" data-act="pickev" data-id="${e.id}" ${e.id === ev.id ? 'style="outline:1px solid var(--tx);outline-offset:-1px"' : ''}><td>${fmtS(e.t)} ${dparts(e.t).y}</td><td>${cbadge(e.cls)}</td><td>${schip(e.status)}</td><td class="r">${Math.round(e.conf * 100)}%</td><td class="r">${e.frp.toFixed(0)} MW</td></tr>`).join('')}</tbody></table><div class="dim small" style="margin-top:6px">Select a row to load that event above.</div>`, { right: (EVP[p.id] || []).length + ' events, 12 months' });
}
function reliabilityCard(p, ev, ser, ctx) {
  const nAt = p.kind === 'site' ? ser.obs.filter(o => o.t <= ev.t).length : 3;
  const f = [['Thermal signal', clamp(.5 + ev.conf * .45, 0, 1)], ['History depth', p.kind === 'site' ? clamp(nAt / 60, 0, 1) : .2], ['Facility and land data', { good: .9, partial: .6, weak: .35 }[p.cover]], ['Imagery quality', ctx.valid]];
  const all = f[0][1] * .3 + f[1][1] * .25 + f[2][1] * .25 + f[3][1] * .2, lab = all >= .75 ? 'High' : all >= .55 ? 'Moderate' : 'Low';
  const src = { gas_flare: 'Nightfire flare catalogue match and facility match', industrial: 'Facility match and industrial land use', mining: 'Coal and mine facility match with land use', wildfire: 'Forest land cover, season and short duration', agricultural_burning: 'Cropland cover, season and short duration', unknown: 'No confident match' }[ev.cls];
  return card('How far to trust this', `<div class="row between"><b style="font-size:16px">${lab} reliability</b><span class="dim small">Overall ${Math.round(all * 100)} of 100</span></div>
  <div style="display:grid;gap:8px;margin:10px 0 14px">${f.map(([l, v]) => `<div class="pbar" style="grid-template-columns:150px 1fr 40px;margin:0"><span class="tx2">${l}</span><span class="tr"><i style="width:${v * 100}%;background:var(--tx)"></i></span><span class="num tx2" style="text-align:right">${Math.round(v * 100)}</span></div>`).join('')}</div>
  <dl class="kv small"><dt>Sensor</dt><dd>${esc(ev.sensor)}</dd><dt>Detections</dt><dd>${ev.nDet} over ${ev.area} km²</dd><dt>Label source</dt><dd>${src} (weak label, not ground truth)</dd><dt>Model</dt><dd>agni-xgb 0.3.1, sample</dd><dt>Features</dt><dd>fs-2026.09, sample</dd><dt>Last ingest</dt><dd>2 h 41 min ago, sample</dd></dl>`, { right: 'Sample values' });
}
function ergFor(p, ev) {
  if (ev.cls === 'gas_flare' || /LNG/.test(p.type)) return ['ERG 2024 Guide 115', 'Gases, flammable (including refrigerated liquids)'];
  if (/Refinery|Petro/.test(p.type)) return ['ERG 2024 Guide 128', 'Flammable liquids (non-polar, water-immiscible)'];
  return null;
}
function responseCard(p, ev) {
  const g = ergFor(p, ev);
  const ref = g ? `<dl class="kv small"><dt>Reference</dt><dd>${g[0]}</dd><dt>Topic</dt><dd>${g[1]}</dd></dl>` : `<p class="small tx2" style="margin:0">No chemical guide applies without knowing what is burning. Use the facility's own emergency plan or the relevant state fire or mine emergency protocol.</p>`;
  return card('Response reference', `${ref}<div class="banner plain small">Generic reference only. Not a substitute for an on-scene hazmat assessment or the facility's site-specific plan. Guidance text is looked up, never written by the model.</div>
  <div class="row wrap" style="margin-top:12px"><button class="btn sm" data-act="capxml" aria-pressed="${S.capXml}">${S.capXml ? 'Hide' : 'Preview'} alert draft</button><button class="btn sm" data-act="capcopy" data-id="${ev.id}">${icon('copy', 14)}Copy draft</button></div>
  ${S.capXml ? `<pre class="xml" style="margin-top:10px">${esc(capXml(ev))}</pre>` : ''}
  <div class="dim small" style="margin-top:8px">A draft in the standard Common Alerting Protocol format for an authorized agency to review. It is not sent anywhere.</div>`, { right: '<span class="chip prop">Proposed</span>' });
}
function exportCard(p) {
  return card('Export', `<div class="row wrap"><button class="btn sm" data-act="exportcsv" data-id="${p.id}">${icon('download', 14)}Events as CSV</button><button class="btn sm" data-act="exportgeo" data-id="${p.id}">${icon('download', 14)}Events as GeoJSON</button><button class="btn sm" disabled>PDF incident report <span class="chip prop">Proposed</span></button></div>`);
}
function outlookCard(p, ev) {
  const W = weatherOf(p, ev);
  return card('Wind and smoke outlook', `<p class="sum" style="margin:0">${esc(outlookLine(p, ev, W))}</p><div class="row wrap" style="margin-top:12px"><button class="btn pri sm" data-act="adv" data-id="${p.id}">${icon('layers', 14)}Open advanced analysis</button><span class="dim small">Wind, smoke drift, fire spread and all 141 features.</span></div>`, { right: '<span class="chip prop">Sample outlook</span>' });
}
function placeReport(p) {
  AF.length = 0;
  const ev = pickEvent(p), ser = seriesOf(p), st = statsOf(p), ctx = contextOf(p), sh = shapOf(ev);
  const c = { header: headerCard(p, ev, st), verdict: verdictCard(p, ev, st, ser), why: whyCard(ev, sh), time: timelineCard(p, ev, ser), fp: fingerprintCard(p, st, ser), base: baselineCard(p, ev, ser), foot: footprintCard(p, ev), sur: surroundCard(p, ctx), img: imageryCard(p, ev, ctx), cond: conditionsCard(ctx), near: nearbyCard(p), hist: historyCard(p, ev), rel: reliabilityCard(p, ev, ser, ctx), resp: responseCard(p, ev), exp: exportCard(p), out: outlookCard(p, ev) };
  const html = S.expanded ? layout2([c.header, c.verdict, c.out, c.why, c.base, c.sur, c.rel, c.resp], [c.time, c.fp, c.foot, c.img, c.cond, c.near, c.hist, c.exp]) : [c.header, c.verdict, c.out, c.why, c.time, c.fp, c.base, c.foot, c.sur, c.img, c.cond, c.near, c.hist, c.rel, c.resp, c.exp].join('');
  return { html, after: () => AF.forEach(f => { try { f(); } catch (e) { console.error(e); } }) };
}

/* ===== Area report ===== */
function areaReport() {
  AF.length = 0; const a = S.area;
  const evs = CUR.evs.filter(e => { const p = PMAP[e.pid]; return p.lon >= a.lo0 && p.lon <= a.lo1 && p.lat >= a.la0 && p.lat <= a.la1; }), gs = groups(evs).sort((x, y) => (sev(y.top) - sev(x.top)) || (y.top.t - x.top.t));
  const by = {}; CLS.forEach(c => by[c.id] = 0); evs.forEach(e => by[e.cls]++);
  const mx = Math.max(1, ...Object.values(by)), abn = gs.filter(g => sev(g.top) === 3).length, rev = evs.filter(e => e.review).length;
  AF.push(() => {
    const T = tk(), nb = 12, bw = S.win * DAY / nb, lo = S.asOf - S.win * DAY, cnt = Array(nb).fill(0); evs.forEach(e => cnt[clamp(Math.floor((e.t - lo) / bw), 0, nb - 1)]++);
    chart('#ar-t', { grid: { left: 30, right: 8, top: 8, bottom: 22 }, tooltip: Object.assign(tt(), { trigger: 'axis' }), xAxis: Object.assign({ type: 'category', data: cnt.map((_, i) => fmtS(lo + i * bw)) }, axb(), { axisLabel: { color: T.dim, fontSize: 10, interval: 2 } }), yAxis: Object.assign({ type: 'value', minInterval: 1 }, axb()), series: [{ type: 'bar', data: cnt, itemStyle: { color: T.dim, borderRadius: [2, 2, 0, 0] } }] }, 120);
  });
  const head = card('Selected area', `<div class="stats" style="grid-template-columns:repeat(2,minmax(0,1fr))">${[['Events', evs.length], ['Places', gs.length], ['Abnormal', abn], ['Needs review', rev]].map(([l, v]) => `<div class="stat"><div class="l">${l}</div><div class="v">${v}</div></div>`).join('')}</div>
  <div class="dim small" style="margin-top:10px">${a.la0.toFixed(1)}° to ${a.la1.toFixed(1)}°N, ${a.lo0.toFixed(1)}° to ${a.lo1.toFixed(1)}°E, ${fmtS(S.asOf - S.win * DAY + DAY)} to ${fmtD(S.asOf)}. Filters on the map apply here too.</div>
  <div class="row wrap" style="margin-top:12px"><button class="btn sm" data-act="exportarea">${icon('download', 14)}Export events</button><button class="btn sm" data-act="close">Clear area</button></div>`);
  const mix = card('Class mix', `<div class="shap">${CLS.map(c => `<div class="pbar" style="grid-template-columns:150px 1fr 30px;margin:0"><span class="cb">${glyph(c.id, 12)}${c.label}</span><span class="tr"><i style="width:${by[c.id] / mx * 100}%;background:${ccol(c.id)}"></i></span><span class="num tx2" style="text-align:right">${by[c.id]}</span></div>`).join('')}</div>`);
  const trend = card('Events over time', `<div id="ar-t" class="chart"></div>`);
  const list = card('Places inside', gs.length ? `<div class="queue">${gs.slice(0, 12).map(qItem).join('')}</div>${gs.length > 12 ? `<div class="dim small" style="margin-top:8px">and ${gs.length - 12} more</div>` : ''}` : `<div class="empty"><b>No events in this box</b>Draw a larger area or widen the date window.</div>`);
  return { html: layout2([head, mix, trend], [list]), after: () => AF.forEach(f => f()) };
}

/* ===== Pages ===== */
const NAV = [['map', 'Command', 'map'], ['regions', 'Regions', 'bars'], ['watch', 'Watchlist', 'star'], ['model', 'Model', 'gauge'], ['methods', 'Methods', 'book'], ['alerts', 'Hand-off', 'send']];
const pageHead = (t, lead) => `<div><h1>${t}</h1><p class="lead">${lead}</p></div>`;
const sampleNote = txt => `<div class="banner"><span><b>Sample data.</b> ${txt}</span></div>`;

function pageRegions() {
  const evs = EVENTS.filter(passes), T = tk();
  AF.push(() => {
    const T = tk(), mk = t => { const p = dparts(t); return p.y * 12 + p.m; }, m0 = mk(START), m1 = mk(TODAY), labels = [], keys = [];
    for (let m = m0; m <= m1; m++) { keys.push(m); labels.push(MONTHS[m % 12] + ' ' + String(Math.floor(m / 12)).slice(2)); }
    chart('#rg-mix', { grid: { left: 36, right: 10, top: 10, bottom: 26 }, tooltip: Object.assign(tt(), { trigger: 'axis', axisPointer: { type: 'shadow' } }), xAxis: Object.assign({ type: 'category', data: labels }, axb()), yAxis: Object.assign({ type: 'value' }, axb()), series: CLS.map(c => ({ name: c.label, type: 'bar', stack: 'a', barWidth: '62%', itemStyle: { color: ccol(c.id) }, data: keys.map(k => evs.filter(e => e.cls === c.id && mk(e.t) === k).length) })) }, 250);
    const wk = 52, ty = Array(wk).fill(0); evs.forEach(e => { ty[Math.min(wk - 1, Math.floor((e.t - START) / (7 * DAY)))] += e.nDet; });
    const r = R('prior'), pr = Array.from({ length: 5 }, () => ty.map(v => v * r.range(.55, 1.6))), mn = ty.map((_, i) => Math.min(...pr.map(y => y[i]))), mxv = ty.map((_, i) => Math.max(...pr.map(y => y[i]))), md = ty.map((_, i) => median(pr.map(y => y[i])));
    const wl = ty.map((_, i) => fmtS(START + i * 7 * DAY));
    chart('#rg-wk', { grid: { left: 46, right: 10, top: 10, bottom: 26 }, tooltip: Object.assign(tt(), { trigger: 'axis' }), xAxis: Object.assign({ type: 'category', data: wl, boundaryGap: false }, axb(), { axisLabel: { color: T.dim, fontSize: 10, interval: 7 } }), yAxis: Object.assign({ type: 'value' }, axb()), series: [{ name: 'Low', type: 'line', stack: 'b', data: mn.map(Math.round), lineStyle: { opacity: 0 }, symbol: 'none', tooltip: { show: false } }, { name: 'Earlier years, range', type: 'line', stack: 'b', data: mxv.map((v, i) => Math.round(v - mn[i])), lineStyle: { opacity: 0 }, symbol: 'none', areaStyle: { color: alpha(T.dim, .22) } }, { name: 'Earlier years, median', type: 'line', data: md.map(Math.round), symbol: 'none', lineStyle: { color: T.dim, type: 'dashed', width: 1.2 } }, { name: 'This year', type: 'line', data: ty, symbol: 'none', lineStyle: { color: T.tx, width: 2 } }] }, 250);
    const dm = new Map(); evs.forEach(e => { const p = PMAP[e.pid], k = p.district + ', ' + p.state; dm.set(k, (dm.get(k) || 0) + 1); });
    const top = [...dm.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10).reverse();
    chart('#rg-dist', { grid: { left: 190, right: 30, top: 6, bottom: 20 }, tooltip: Object.assign(tt(), { trigger: 'item' }), xAxis: Object.assign({ type: 'value' }, axb()), yAxis: Object.assign({ type: 'category', data: top.map(t => t[0]) }, axb(), { axisLabel: { color: T.tx2, fontSize: 11, width: 178, overflow: 'truncate' }, splitLine: { show: false } }), series: [{ type: 'bar', data: top.map(t => t[1]), barWidth: '62%', itemStyle: { color: T.dim, borderRadius: [0, 3, 3, 0] } }] }, 300);
    const H = Array(24).fill(0); evs.forEach(e => { H[Math.floor(e.hr)] += 1; });
    chart('#rg-hr', { grid: { left: 36, right: 8, top: 8, bottom: 22 }, tooltip: Object.assign(tt(), { trigger: 'axis' }), xAxis: Object.assign({ type: 'category', data: H.map((_, i) => i) }, axb(), { axisLabel: { color: T.dim, fontSize: 10, interval: 2 } }), yAxis: Object.assign({ type: 'value' }, axb()), series: [{ type: 'bar', barWidth: '70%', data: H.map((v, i) => ({ value: v, itemStyle: { color: (i < 6 || i >= 18) ? T.dim : T.tx2, borderRadius: [2, 2, 0, 0] } })) }] }, 300);
    cmpChart();
  });
  const sites = PLACES.filter(p => p.kind === 'site').map(p => { const l = (EVP[p.id] || []), abn = l.filter(e => e.status === 'abnormal').length; return { p, abn, n: l.length, last: l[l.length - 1] }; }).sort((a, b) => b.abn - a.abn || b.n - a.n).slice(0, 10);
  const rank = `<div class="queue">${sites.map(({ p, abn, last }) => { const ser = seriesOf(p).obs.filter(o => o.t > TODAY - 90 * DAY).map(o => o.frp); return `<button class="qi" data-act="sel" data-id="${p.id}" style="grid-template-columns:minmax(0,1fr) auto"><span style="min-width:0"><span class="t" style="display:block">${esc(p.name)}</span><span class="chips" style="margin-top:3px">${cbadge(p.cls)}${abn ? `<span class="chip abn">${abn} abnormal</span>` : ''}</span></span><span style="color:var(--dim)">${spark(ser.slice(-40), 96, 26)}</span></button>`; }).join('')}</div>`;
  const opts = PLACES.filter(p => p.kind === 'site').map(p => `<option value="${p.id}" %S%>${esc(p.name)}</option>`).join('');
  const selA = `<select class="sel" data-act="cmpA" aria-label="First place" style="max-width:100%">${opts.replace(/%S%/g, '').replace(`value="${S.cmpA}"`, `value="${S.cmpA}" selected`)}</select>`, selB = `<select class="sel" data-act="cmpB" aria-label="Second place" style="max-width:100%">${opts.replace(/%S%/g, '').replace(`value="${S.cmpB}"`, `value="${S.cmpB}" selected`)}</select>`;
  return `${pageHead('Regional analytics', 'How thermal events are spread over classes, places and seasons across India in the past 12 months. Map filters apply here too.')}
  ${sampleNote('Everything on this page is generated sample data. Earlier-year values are modelled for the demo.')}
  <div class="g2">${card('Events by class and month', '<div id="rg-mix" class="chart"></div>' + `<div class="legendrow">${CLS.map(c => `<span>${glyph(c.id, 11)}${c.label}</span>`).join('')}</div>`, { right: 'Events per month' })}${card('This year against earlier years', '<div id="rg-wk" class="chart"></div><div class="legendrow"><span><i class="sw" style="background:var(--tx)"></i>This year</span><span><i class="sw" style="background:var(--dim);opacity:.5"></i>Range of five earlier years (modelled)</span></div>', { right: 'Detections per week' })}</div>
  <div class="g2">${card('Busiest districts', '<div id="rg-dist" class="chart"></div>', { right: 'Events, 12 months' })}${card('When detections happen', '<div id="rg-hr" class="chart"></div><div class="legendrow"><span><i class="sw" style="background:var(--dim)"></i>Night passes</span><span><i class="sw" style="background:var(--tx2)"></i>Day passes</span></div>', { right: 'Local hour of pass' })}</div>
  <div class="g2">${card('Persistent sites to watch', rank, { right: 'Most abnormal events first' })}${card('Compare two places', `<div class="row wrap" style="margin-bottom:10px">${selA}${selB}</div><div id="rg-cmp" class="chart"></div><div id="rg-cmpt"></div>`, { right: 'Last 120 days' })}</div>`;
}
function cmpChart() {
  const T = tk(), a = PMAP[S.cmpA], b = PMAP[S.cmpB]; if (!a || !b || !$('#rg-cmp')) return;
  const mkS = (p, col) => ({ name: p.name, type: 'line', showSymbol: true, symbolSize: 4, connectNulls: true, lineStyle: { color: col, width: 1.6 }, itemStyle: { color: col }, data: seriesOf(p).obs.filter(o => o.t > TODAY - 120 * DAY).map(o => [o.t, +o.frp.toFixed(2)]) });
  const cA = ccol(a.cls), cB = a.cls === b.cls ? cv('--tx') : ccol(b.cls);
  chart('#rg-cmp', { grid: { left: 46, right: 10, top: 10, bottom: 26 }, tooltip: Object.assign(tt(), { trigger: 'axis', formatter: q => q.map(x => `${x.seriesName}: ${x.data[1]} MW`).join('<br>') }), xAxis: Object.assign({ type: 'time' }, axb()), yAxis: Object.assign({ type: 'log', min: .5 }, axb()), series: [mkS(a, cA), mkS(b, cB)] }, 220);
  const row = p => { const ser = seriesOf(p), st = statsOf(p), bs = ser.base, abn = (EVP[p.id] || []).filter(e => e.status === 'abnormal').length, last = ser.obs[ser.obs.length - 1], nt = hoursOf(p).slice(0, 6).concat(hoursOf(p).slice(18)).reduce((x, y) => x + y, 0) / hoursOf(p).reduce((x, y) => x + y, 1);
    return [cbadge(p.cls), bs.med.toFixed(1) + ' MW', bs.p95.toFixed(1) + ' MW', st.d90 + ' of 90', Math.round(nt * 100) + '%', abn, last && last.z != null ? (last.z >= 0 ? '+' : '') + last.z.toFixed(1) : 'n/a']; };
  const ra = row(a), rb = row(b), labs = ['Class', 'Median fire power', '95th percentile', 'Active days', 'Night detections', 'Abnormal events, 12 mo', 'Deviation now'];
  $('#rg-cmpt').innerHTML = `<table class="tbl small"><thead><tr><th></th><th>${esc(a.district)}</th><th>${esc(b.district)}</th></tr></thead><tbody>${labs.map((l, i) => `<tr><td class="dim">${l}</td><td>${ra[i]}</td><td>${rb[i]}</td></tr>`).join('')}</tbody></table>`;
}

function watchRows() {
  const q = S.wq.trim().toLowerCase(), rows = PLACES.filter(p => p.kind === 'site').map(p => { const l = (EVP[p.id] || []).filter(e => e.t <= S.asOf), e = l[l.length - 1]; return { p, e }; }).filter(r => r.e && (S.wst === 'all' || (S.wst === 'review' ? r.e.review : S.wst === 'watched' ? S.watch.has(r.p.id) : r.e.status === S.wst)) && (!q || (r.p.name + ' ' + r.p.state + ' ' + r.p.code).toLowerCase().includes(q)));
  const k = S.sort.k, d = S.sort.dir, val = r => ({ name: r.p.name, cls: r.e.cls, status: STATUS[r.e.status].rank, conf: r.e.conf, t: r.e.t, med: r.p.med }[k]);
  return rows.sort((a, b) => { const x = val(a), y = val(b); return (x > y ? 1 : x < y ? -1 : 0) * d; });
}
function watchTable() {
  const rows = watchRows(), th = (k, l, cls = '') => `<th class="${cls}"><button data-act="sort" data-k="${k}" aria-label="Sort by ${l}">${l}${S.sort.k === k ? (S.sort.dir > 0 ? ' ▲' : ' ▼') : ''}</button></th>`;
  return `<table class="tbl"><thead><tr><th style="width:32px"></th>${th('name', 'Place')}${th('cls', 'Class')}${th('status', 'Status')}${th('conf', 'Confidence', 'r')}${th('t', 'Last event')}${th('med', 'Median power', 'r')}<th>90-day trend</th></tr></thead><tbody>${rows.map(({ p, e }) => `<tr class="click" data-act="sel" data-id="${p.id}"><td><button class="star" data-act="watch" data-id="${p.id}" aria-pressed="${S.watch.has(p.id)}" aria-label="Watch ${esc(p.name)}">${icon('star', 16)}</button></td><td><b>${esc(p.name)}</b><div class="dim small">${esc(p.state)}, ${esc(p.code)}</div></td><td>${cbadge(e.cls)}</td><td>${schip(e.status)} ${e.review ? rchip() : ''}</td><td class="r">${Math.round(e.conf * 100)}%</td><td>${fmtS(e.t)} ${dparts(e.t).y}</td><td class="r">${p.med} MW</td><td style="color:var(--dim)">${spark(seriesOf(p).obs.filter(o => o.t > TODAY - 90 * DAY).map(o => o.frp).slice(-40), 96, 24)}</td></tr>`).join('')}</tbody></table>${rows.length ? '' : '<div class="empty" style="margin:14px"><b>No places match</b>Clear the search or pick another status.</div>'}<div class="dim small" style="padding:10px">${rows.length} places</div>`;
}
function pageWatch() {
  return `${pageHead('Watchlist and site registry', 'Every persistent site the system tracks, with its latest call. Star places to follow them.')}
  <div class="row wrap"><div class="search" style="max-width:320px"><span class="ic">${icon('search', 16)}</span><input id="wq" data-act="wq" placeholder="Filter by name, state or code" value="${esc(S.wq)}" aria-label="Filter places"></div>
  <select class="sel" data-act="wst" aria-label="Status filter">${[['all', 'All statuses'], ['abnormal', 'Abnormal'], ['routine', 'Routine'], ['baseline_building', 'Baseline building'], ['review', 'Needs review'], ['watched', 'Starred only']].map(([v, l]) => `<option value="${v}" ${S.wst === v ? 'selected' : ''}>${l}</option>`).join('')}</select>
  <span style="flex:1"></span><button class="btn sm" data-act="exportreg">${icon('download', 14)}Export registry</button></div>
  <div class="card" id="wtable" style="overflow:auto">${watchTable()}</div>`;
}

const FRESH = [['FIRMS VIIRS (NOAA-20, NOAA-21, Suomi-NPP)', '2 h 41 min old', 'Within the 3 h target', 'ok'], ['FIRMS MODIS (Terra, Aqua)', '3 h 05 min old', 'A little over target', 'warn'], ['Facility records (Global Energy Monitor)', 'Updated 18 Aug 2026', 'Refreshed monthly', 'ok'], ['OpenStreetMap industrial extract', 'Updated 14 Sep 2026', 'Refreshed weekly', 'ok'], ['Land cover (ESA WorldCover)', '2021 map', 'Static, 10 m', 'ok'], ['Satellite imagery (HLS)', 'Newest clear scene 3 d old', 'Limited by cloud', 'warn'], ['Weather (ERA5 for training)', 'About 5 d behind', 'Live runs use a forecast source', 'ok']];
function confMatrix() {
  const ids = METRICS.map(m => m.id), W = { gas_flare: { industrial: .7, unknown: .3 }, industrial: { gas_flare: .35, mining: .35, unknown: .3 }, mining: { industrial: .6, unknown: .4 }, wildfire: { agricultural_burning: .6, unknown: .4 }, agricultural_burning: { wildfire: .6, unknown: .4 }, unknown: { industrial: .3, wildfire: .25, agricultural_burning: .25, gas_flare: .2 } };
  return METRICS.map(m => { const row = Object.fromEntries(ids.map(i => [i, 0])), ok = Math.round(m.n * m.r); row[m.id] = ok; const rest = m.n - ok; Object.entries(W[m.id]).forEach(([k, w]) => row[k] += Math.round(rest * w)); return ids.map(i => row[i]); });
}
function pageModel() {
  const macro = METRICS.reduce((a, m) => a + m.f1, 0) / METRICS.length;
  AF.push(() => {
    const T = tk(), cm = confMatrix(), names = METRICS.map(m => CLSMAP[m.id].label.replace('Agricultural burning', 'Ag. burning').replace(' source', '')), data = []; let mx = 0;
    cm.forEach((row, i) => row.forEach((v, j) => { data.push([j, i, v]); mx = Math.max(mx, v); }));
    chart('#cm', { grid: { left: 96, right: 10, top: 10, bottom: 54 }, tooltip: Object.assign(tt(), { formatter: q => `True ${names[q.data[1]]}<br>Predicted ${names[q.data[0]]}: ${q.data[2]}` }), xAxis: Object.assign({ type: 'category', data: names, name: 'Predicted', nameLocation: 'middle', nameGap: 36, nameTextStyle: { color: T.dim } }, axb(), { axisLabel: { color: T.dim, fontSize: 10, interval: 0, rotate: 20 }, splitLine: { show: false } }), yAxis: Object.assign({ type: 'category', data: names, inverse: true, name: 'True', nameTextStyle: { color: T.dim } }, axb(), { axisLabel: { color: T.dim, fontSize: 10 }, splitLine: { show: false } }), visualMap: { show: false, min: 0, max: mx, inRange: { color: [T.raised, cv('--c-industrial')] } }, series: [{ type: 'heatmap', data, label: { show: true, color: T.tx, fontSize: 10 }, itemStyle: { borderColor: T.panel, borderWidth: 2 } }] }, 280);
    const rr = R('rel'), xs = [.05, .15, .25, .35, .45, .55, .65, .75, .85, .95];
    chart('#rel', { grid: { left: 40, right: 12, top: 10, bottom: 30 }, tooltip: Object.assign(tt(), { trigger: 'axis' }), xAxis: Object.assign({ type: 'value', min: 0, max: 1, name: 'Predicted confidence', nameLocation: 'middle', nameGap: 22, nameTextStyle: { color: T.dim, fontSize: 10 } }, axb()), yAxis: Object.assign({ type: 'value', min: 0, max: 1, name: 'Observed', nameTextStyle: { color: T.dim, fontSize: 10, align: 'left' } }, axb()), series: [{ type: 'line', data: [[0, 0], [1, 1]], symbol: 'none', lineStyle: { color: T.dim, type: 'dashed', width: 1 } }, { type: 'line', data: xs.map(x => [x, +clamp(x + rr.range(-.06, .04), 0, 1).toFixed(2)]), symbolSize: 6, lineStyle: { color: T.tx, width: 2 }, itemStyle: { color: T.tx } }] }, 240);
  });
  const runs = Array.from({ length: 8 }, (_, i) => { const t = TODAY - (i * 3 + .7) * HOUR, r = R('run' + i); return [fmtDT(t), r.int(4, 9) + ' min', nf(r.int(1800, 5200)), nf(r.int(30, 140)), i === 3 ? 'Retried once' : 'Completed']; });
  return `${pageHead('Model and data health', 'What the system is running on, how fresh each source is, and how well it is doing. This is the page that shows whether to trust the rest.')}
  ${sampleNote('The numbers below are placeholders for the layout. Replace them with results from a spatially blocked hold-out set with independently checked labels.')}
  <div class="g2">${card('Data freshness', `<table class="tbl small"><tbody>${FRESH.map(([n, a, s, k]) => `<tr><td>${n}</td><td class="tx2">${a}</td><td><span class="chip ${k === 'warn' ? 'rev' : ''}">${s}</span></td></tr>`).join('')}</tbody></table>`, { right: 'Sample values' })}
  ${card('Pipeline runs', `<table class="tbl small"><thead><tr><th>Started</th><th>Time taken</th><th class="r">Detections</th><th class="r">Events</th><th>Result</th></tr></thead><tbody>${runs.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td><td class="r">${r[2]}</td><td class="r">${r[3]}</td><td>${r[4]}</td></tr>`).join('')}</tbody></table>`, { right: 'Every 3 hours' })}</div>
  <div class="g21">${card('Per-class results', `<table class="tbl small"><thead><tr><th>Class</th><th class="r">Precision</th><th class="r">Recall</th><th class="r">F1</th><th class="r">Examples</th></tr></thead><tbody>${METRICS.map(m => `<tr><td>${cbadge(m.id)}</td><td class="r">${m.p.toFixed(2)}</td><td class="r">${m.r.toFixed(2)}</td><td class="r">${m.f1.toFixed(2)}</td><td class="r">${m.n}</td></tr>`).join('')}<tr><td><b>Macro average</b></td><td></td><td></td><td class="r"><b>${macro.toFixed(2)}</b></td><td class="r">${METRICS.reduce((a, m) => a + m.n, 0)}</td></tr></tbody></table><p class="small tx2" style="margin:10px 0 0">If macro F1 on the real hold-out set comes in under about 0.6, lead with the explanations and the abnormal-versus-routine judgement, not the accuracy figure.</p>`, { right: 'Hold-out set' })}
  ${card('Calibration', '<div id="rel" class="chart"></div><div class="row between small" style="margin-top:6px"><span class="tx2">Brier score 0.11</span><span class="tx2">Dashed line is perfect</span></div>', { info: 'conf', right: 'Reliability diagram' })}</div>
  <div class="g2">${card('Confusion matrix', '<div id="cm" class="chart"></div>', { right: 'Counts' })}
  ${card('How it is validated', `<ul class="list"><li>Blocks of land are kept apart between training and testing, with a buffer, so neighbouring pixels of one site never sit on both sides.</li><li>A later period is held out in time, so the model is tested on days it has not seen.</li><li>Labels come from several weak sources: the Nightfire flare catalogue, facility trackers, land cover with season, and Sentinel-2 burn scars for confirmation. A few hundred are checked by hand and kept aside.</li><li>Probabilities are calibrated after training, and anything under 60% is sent for human review.</li></ul>`)}</div>`;
}

function pageMethods() {
  const imp = ['Command map with filters, layers, hex density, time scrubber and area drawing', 'Priority queue, KPI strip and place reports with explanations, timelines and baselines', 'Regional analytics, watchlist, and model health pages', 'Exports of events as CSV and GeoJSON'], prop = ['Live connection to the FIRMS pipeline, PostGIS and the classifier', 'Real satellite imagery, and a satellite basemap with vector tiles', 'Looked-up ERG and NDMA guidance for the inferred category', 'Alert drafts in the standard Common Alerting Protocol format for an authorized agency to review and send', 'Smoke plume affected-zone estimate from live wind', 'One-page PDF incident report'];
  return `${pageHead('Methods and limits', 'What each label means, how status is decided, what this build does, and what it cannot do.')}
  <div class="g2">${card('Source classes', `<div style="display:grid;gap:10px">${CLS.map(c => `<div><div class="cb" style="font-weight:600">${glyph(c.id, 13)}${c.label}</div><div class="tx2 small">${c.blurb}</div></div>`).join('')}</div>`)}
  ${card('How status is decided', `<p class="tx2" style="margin:0 0 10px">The class says what the source is. A separate rule, based on the place's own history, says whether it is behaving normally.</p><ol class="list"><li>Attach the event to a site and take that site's earlier fire power readings, on a log scale.</li><li>Take the median and the median absolute deviation, so a few extreme days do not skew what counts as normal.</li><li>Score the event as its distance above the median in units of that spread.</li><li>Call it abnormal if the score is above about 3.5 and it persists or grows, if a new source appears inside a facility, or if a quiet site switches on.</li><li>With fewer than 20 clear observations, show baseline building and claim nothing.</li></ol>`)}</div>
  <div class="g2">${card('What this build does', `<ul class="list">${imp.map(x => `<li>${x} <span class="chip impl">Built, on sample data</span></li>`).join('')}</ul>`)}${card('What is still proposed', `<ul class="list">${prop.map(x => `<li>${x} <span class="chip prop">Proposed</span></li>`).join('')}</ul>`)}</div>
  ${card('Limits to keep in mind', `<ul class="list"><li>Satellites cannot see through cloud. A gap in the timeline is missing data, never zero fire.</li><li>Fires smaller or cooler than the sensor threshold are missed, and one pixel is 375 m (VIIRS) or 1 km (MODIS) across, so it can hold several sources.</li><li>Facility records are incomplete in parts of India. Where coverage is weak, the report says so.</li><li>The data is near real time, meaning hours, not seconds. This is a triage and monitoring tool, not a dispatch system.</li><li>The class is the most likely source inferred from context. It is not a confirmed cause, and low-confidence calls are flagged for a person to check.</li><li>Every number on these screens is generated sample data until the live pipeline is connected.</li></ul>`)}`;
}

function pageAlerts() {
  const abn = winEvents().filter(e => e.status === 'abnormal').sort((a, b) => b.t - a.t).slice(0, 8);
  if (!S.capId || !abn.find(e => e.id === S.capId)) S.capId = abn[0] ? abn[0].id : null;
  const ev = abn.find(e => e.id === S.capId);
  return `${pageHead('Alert hand-off', 'Drafts for an authorized alerting agency to review. This system does not send public alerts.')}
  <div class="banner plain"><span><span class="chip prop">Proposed</span> Only authorized government agencies can issue public alerts in India. The system prepares the content in the standard Common Alerting Protocol format for one of them to review. Sending here is a mock and goes nowhere.</span></div>
  ${ev ? `<div class="g21"><div style="min-width:0">${card('Abnormal events in the current window', `<div class="queue">${abn.map(e => { const p = PMAP[e.pid]; return `<button class="qi" data-act="capsel" data-id="${e.id}" aria-current="${e.id === S.capId}" style="grid-template-columns:auto minmax(0,1fr) auto"><span>${ring(e.conf, 36, 4, ccol(e.cls))}</span><span style="min-width:0"><span class="t" style="display:block">${esc(p.name)}</span><span class="s" style="display:block">${esc(p.district)}, ${esc(p.state)}</span></span><span class="m"><b class="num">${e.frp.toFixed(0)} MW</b><span>${ago(e.t)}</span></span></button>`; }).join('')}</div>`)}</div>
  <div style="display:grid;gap:16px;align-content:start;min-width:0;grid-template-columns:minmax(0,1fr)">${card('Alert draft', `<pre class="xml">${esc(capXml(ev))}</pre><div class="row wrap" style="margin-top:12px"><button class="btn sm" data-act="capcopy" data-id="${ev.id}">${icon('copy', 14)}Copy draft</button><button class="btn sm pri" data-act="capsend" data-id="${ev.id}">Mock send to reviewer</button></div>`, { right: '<span class="chip prop">Draft</span>' })}
  ${card('Mock delivery log', S.capLog.length ? `<div class="dist small">${S.capLog.map(l => `<div><span>${esc(l.t)} ${esc(l.n)}</span><span class="dim">Mock only, not sent</span></div>`).join('')}</div>` : '<div class="dim small">Nothing sent yet. Mock sends appear here.</div>')}</div></div>` : `<div class="empty"><b>No abnormal events in this window</b>Widen the date window on the Command page to draft one.</div>`}`;
}

/* ===== Shell and routing ===== */
let hashLock = false, PENDING_ADV = null;
function writeHash() {
  const q = [`w=${S.win}`, `a=${iso(S.asOf)}`]; if (S.status !== 'all') q.push('s=' + S.status); if (S.cls.size < CLS.length) q.push('c=' + [...S.cls].join(',')); if (S.minConf) q.push('m=' + Math.round(S.minConf * 100)); if (S.sensor !== 'all') q.push('n=' + S.sensor); if (A.open) q.push('x=' + A.tab);
  const h = '#/' + S.page + (S.sel ? '/' + S.sel : '') + '?' + q.join('&'); hashLock = true;
  try { history.replaceState(null, '', h); } catch (e) { try { location.hash = h; } catch (e2) {} } setTimeout(() => { hashLock = false; }, 0);
}
function readHash() {
  const h = (location.hash || '').replace(/^#\/?/, ''); if (!h) return; const [path, qs] = h.split('?'), [pg, id] = path.split('/');
  if (NAV.some(n => n[0] === pg)) S.page = pg; if (id && PMAP[id]) S.sel = id;
  (qs || '').split('&').forEach(kv => { const [k, v] = kv.split('='); if (!v) return; if (k === 'w' && [7, 30, 90, 365].includes(+v)) S.win = +v; if (k === 'a') { const t = Date.parse(v + 'T09:00:00Z'); if (!isNaN(t)) S.asOf = clamp(t, START, TODAY); } if (k === 's') S.status = v; if (k === 'm') S.minConf = clamp(+v / 100, 0, .9); if (k === 'n') S.sensor = v; if (k === 'x') PENDING_ADV = v; if (k === 'c') { const c = v.split(',').filter(x => CLSMAP[x]); if (c.length) S.cls = new Set(c); } });
}
function shell() {
  $('#app').innerHTML = `<nav class="rail" aria-label="Main"><div class="logo" aria-hidden="true">${LOGO}</div>${NAV.map(n => `<button class="nv" data-act="nav" data-p="${n[0]}" aria-current="${S.page === n[0] ? 'page' : 'false'}">${icon(n[2], 21)}<span>${n[1]}</span></button>`).join('')}<div class="sp"></div></nav>
  <div class="main"><header class="top"><div class="brand"><b>SPARC</b><span>Thermal source monitoring</span></div>
  <div class="search"><span class="ic">${icon('search', 16)}</span><input id="q" type="search" autocomplete="off" placeholder="Search a place, district or site code" aria-label="Search places" role="combobox" aria-expanded="false" aria-controls="sugg"><kbd>/</kbd><div class="sugg" id="sugg" hidden></div></div>
  <span class="sp"></span><span class="pill" data-tip="Sample value. The live pipeline shows the age of the newest FIRMS data here."><i></i>FIRMS data 2 h 41 min old</span><span class="sample" data-tip="Every place, event and number in this build is generated sample data.">Sample data</span>
  <button class="btn ic" data-act="theme" aria-label="Switch theme">${icon(S.theme === 'dark' ? 'sun' : 'moon', 17)}</button></header><main id="view"></main></div>`;
}
function mapPageHtml() {
  return `<div class="cmd" id="cmd"><div class="mapcol"><div class="filters" id="filters"></div><div class="kpis" id="kpis"></div>
  <div class="mapbox" id="mapbox"><div id="lmap" role="application" aria-label="Map of thermal events over India"></div>
    <div class="ov tl"><button class="btn sm" data-act="layers" aria-pressed="${S.showLayers}">${icon('layers', 15)}Layers</button></div>
    <div class="ov layers" id="layersp" style="${S.showLayers ? '' : 'display:none'}"><div class="dim small">Base map</div><label><input type="radio" name="base" value="street" data-act="base" ${LM.base !== 'plain' ? 'checked' : ''}>Street map (OpenStreetMap)</label><label><input type="radio" name="base" value="plain" data-act="base" ${LM.base === 'plain' ? 'checked' : ''}>Plain outline (works offline)</label><div class="dim small" style="margin-top:4px">On the map</div>${[['events', 'Classified events'], ['raw', 'Raw satellite detections'], ['facilities', 'Facility footprints'], ['density', 'Density (hexagons)'], ['wind', 'Wind forecast']].map(([k, l]) => `<label><input type="checkbox" data-act="layer" data-k="${k}" ${S.layers[k] ? 'checked' : ''}>${l}</label>`).join('')}</div>
    <div class="ov tr"><button class="btn sm" data-act="draw" aria-pressed="${S.drawing}">${icon('draw', 15)}Draw area</button><button class="btn sm ic" data-act="zreset" aria-label="Reset view">${icon('reset', 15)}</button></div>
    <div class="ov hint" id="drawhint" style="display:none">Drag on the map to draw a box</div>
    <div class="ov windctl" id="windctl" style="display:none"><div class="row between small"><b id="windlbl">Now</b><span class="seg" role="group" aria-label="Wind style"><button data-act="windstyle" data-v="flow" aria-pressed="${S.windStyle === 'flow'}">Flow</button><button data-act="windstyle" data-v="arrows" aria-pressed="${S.windStyle === 'arrows'}">Arrows</button></span></div><input type="range" min="0" max="72" step="3" value="${S.windH}" data-act="windh" aria-label="Wind forecast hour"><div class="row small"><span class="dim">Slow</span><span id="windbar" class="windbar"></span><span class="dim">Fast</span></div><div class="dim small">Sample wind. Your weather feed replaces it.</div></div>
    <div class="ov legend" id="legend"></div>
    <div class="ov zoom"><button data-act="zin" aria-label="Zoom in">${icon('plus', 16)}</button><button data-act="zout" aria-label="Zoom out">${icon('minus', 16)}</button></div>
  </div><div class="scrub" id="scrub"></div></div>
  <aside class="side" aria-label="Incidents"><div class="side-head" id="side-head"></div><div class="side-body" id="side-body"></div></aside></div>`;
}
function renderLegend() {
  const el = $('#legend'); if (!el) return;
  el.innerHTML = CLS.map(c => `<button data-act="tcls" data-id="${c.id}" aria-pressed="${S.cls.has(c.id)}">${glyph(c.id, 13)}${c.label}</button>`).join('') + `<div class="sep"></div><div class="k"><svg width="14" height="14"><circle cx="7" cy="7" r="5.5" fill="none" stroke="var(--danger)" stroke-width="2"/></svg>Abnormal for its site</div><div class="k"><svg width="14" height="14"><circle cx="7" cy="7" r="3" fill="var(--warn)"/></svg>Needs review</div><div class="k"><svg width="14" height="14"><circle cx="7" cy="7" r="5.5" fill="none" stroke="var(--tx2)" stroke-dasharray="3 3"/></svg>Baseline building</div>`;
}
function navHighlight() { $$('.nv').forEach(b => b.setAttribute('aria-current', b.dataset.p === S.page ? 'page' : 'false')); }
function renderView() {
  closeAdvanced(true); disposeMap(); disposeCharts(); AF.length = 0; const v = $('#view'); if (!v) return; stopPlay();
  if (S.page === 'map') { v.innerHTML = mapPageHtml(); bindMap(); renderLegend(); refreshMapPage(); syncCmd(); syncDrawBtn(); if (PENDING_ADV && S.sel) { const t = PENDING_ADV; PENDING_ADV = null; openAdvanced(S.sel, t); } return; }
  const html = { regions: pageRegions, watch: pageWatch, model: pageModel, methods: pageMethods, alerts: pageAlerts }[S.page]();
  v.innerHTML = `<div class="page"><div class="page-in">${html}</div></div>`; AF.forEach(f => { try { f(); } catch (e) { console.error(e); } }); writeHash();
}
const _refresh = refreshMapPage;
refreshMapPage = function () { renderLegend(); _refresh(); writeHash(); };

/* ===== Play, search, actions ===== */
let PT = null;
function stopPlay() { if (PT) clearInterval(PT); PT = null; S.play = false; }
function updScrub() {
  const ai = (S.asOf - START) / DAY, W = 730, bw = W / 365, r = $('#scrub rect'), dl = $('#scrub .dl b'), inp = $('#scrub input');
  if (r) { const x0 = Math.max(0, (ai - S.win) * bw), x1 = ai * bw; r.setAttribute('x', x0); r.setAttribute('width', Math.max(2, x1 - x0)); }
  if (dl) dl.textContent = fmtS(S.asOf - S.win * DAY + DAY) + ' to ' + fmtD(S.asOf); if (inp && document.activeElement !== inp) inp.value = Math.round(ai);
}
function startPlay() {
  if (S.asOf >= TODAY - DAY) S.asOf = Math.min(TODAY, START + S.win * DAY);
  S.play = true; renderScrub();
  PT = setInterval(() => {
    S.asOf = Math.min(TODAY, S.asOf + (S.win > 60 ? 3 : 1) * DAY); const inp = $('#scrub input'); if (inp) inp.value = Math.round((S.asOf - START) / DAY);
    updScrub(); renderKpis(); renderMap(); if (!S.sel && !S.area) renderSide(); if (S.asOf >= TODAY) { stopPlay(); renderScrub(); writeHash(); }
  }, 240);
}
let sugg = [];
function onSearch() {
  const q = $('#q').value.trim().toLowerCase(), box = $('#sugg'); S.q = q; S.si = 0;
  if (!q) { box.hidden = true; $('#q').setAttribute('aria-expanded', 'false'); return; }
  sugg = PLACES.filter(p => (p.name + ' ' + p.district + ' ' + p.state + ' ' + p.code).toLowerCase().includes(q)).sort((a, b) => (b.kind === 'site') - (a.kind === 'site')).slice(0, 8);
  box.hidden = false; $('#q').setAttribute('aria-expanded', 'true');
  box.innerHTML = sugg.length ? sugg.map((p, i) => `<button data-act="sel" data-id="${p.id}" class="${i === 0 ? 'on' : ''}" role="option">${glyph(p.cls, 12)}<span><b>${esc(p.name)}</b><br><small>${esc(p.district)}, ${esc(p.state)}</small></span></button>`).join('') : `<div style="padding:12px" class="dim">No place matches "${esc(q)}".</div>`;
}
function closeSearch() { const b = $('#sugg'); if (b) b.hidden = true; const q = $('#q'); if (q) { q.value = ''; q.setAttribute('aria-expanded', 'false'); } }
function goSel(id) {
  closeSearch(); const p = PMAP[id]; if (!p) return; closeAdvanced(true);
  if (S.page !== 'map') { S.page = 'map'; navHighlight(); S.sel = id; S.area = null; S.evId = null; S.expanded = false; LM.saved = { lat: p.lat, lon: p.lon, zoom: 9 }; renderView(); return; }
  select(id);
}
function act(n, el, e) {
  switch (n) {
    case 'nav': S.page = el.dataset.p; navHighlight(); renderView(); break;
    case 'theme': S.theme = S.theme === 'dark' ? 'light' : 'dark'; document.documentElement.setAttribute('data-theme', S.theme); store.set('agni-theme', S.theme); { const re = A.open ? { id: A.p.id, tab: A.tab } : null; shell(); renderView(); if (re && S.page === 'map') openAdvanced(re.id, re.tab); } break;
    case 'tcls': { const id = el.dataset.id; S.cls.has(id) ? S.cls.delete(id) : S.cls.add(id); if (!S.cls.size) CLS.forEach(c => S.cls.add(c.id)); if (S.page === 'map') refreshMapPage(); break; }
    case 'status': S.status = el.value; refreshMapPage(); break;
    case 'sensor': S.sensor = el.value; refreshMapPage(); break;
    case 'conf': S.minConf = +el.value / 100; { const l = $('#fcov'); if (l) l.textContent = el.value + '%'; } renderKpis(); renderMap(); if (!S.sel && !S.area) renderSide(); renderScrub(); writeHash(); break;
    case 'win': S.win = +el.dataset.w; refreshMapPage(); break;
    case 'asof': S.asOf = START + (+el.value) * DAY; updScrub(); renderKpis(); renderMap(); if (!S.sel && !S.area) renderSide(); writeHash(); break;
    case 'play': if (S.play) { stopPlay(); renderScrub(); } else startPlay(); break;
    case 'layers': S.showLayers = !S.showLayers; $('#layersp').style.display = S.showLayers ? '' : 'none'; el.setAttribute('aria-pressed', S.showLayers); break;
    case 'layer': S.layers[el.dataset.k] = el.checked; if (el.dataset.k === 'wind') windSync(); else renderMap(); break;
    case 'draw': S.drawing = !S.drawing; syncDrawBtn(); break;
    case 'zin': case 'zout': mapZoom(n === 'zin' ? 1 : -1); break;
    case 'zreset': mapReset(); break;
    case 'sel': goSel(el.dataset.id); break;
    case 'back': if (S.expanded) { S.expanded = false; syncCmd(); renderSide(); mapInvalidate(); } else clearSel(); break;
    case 'close': clearSel(); break;
    case 'expand': S.expanded = !S.expanded; syncCmd(); renderSide(); mapInvalidate(); break;
    case 'moreq': S.qshow += 30; renderSide(true); break;
    case 'shapall': S.shapAll = !S.shapAll; renderSide(true); break;
    case 'pickev': S.evId = el.dataset.id; renderSide(true); break;
    case 'capxml': S.capXml = !S.capXml; renderSide(true); break;
    case 'watch': { const id = el.dataset.id; S.watch.has(id) ? S.watch.delete(id) : S.watch.add(id); store.set('agni-watch', [...S.watch].join(',')); el.setAttribute('aria-pressed', S.watch.has(id)); toast(S.watch.has(id) ? 'Added to watchlist' : 'Removed from watchlist'); if (S.page === 'watch' && S.wst === 'watched') $('#wtable').innerHTML = watchTable(); break; }
    case 'copylink': copyText(location.href.split('#')[0] + '#/' + S.page + (S.sel ? '/' + S.sel : ''), 'Link copied'); break;
    case 'exportcsv': { const id = el.dataset.id, l = EVP[id] || []; saveFile(`agni-netra-${PMAP[id].code === 'T' ? id : PMAP[id].code}-events.csv`, eventsCsv(l), 'CSV'); break; }
    case 'exportgeo': { const id = el.dataset.id; saveFile(`agni-netra-${PMAP[id].code === 'T' ? id : PMAP[id].code}-events.json`, eventsGeo(EVP[id] || []), 'GeoJSON'); break; }
    case 'exportarea': { const a = S.area, l = CUR.evs.filter(x => { const p = PMAP[x.pid]; return p.lon >= a.lo0 && p.lon <= a.lo1 && p.lat >= a.la0 && p.lat <= a.la1; }); saveFile('agni-netra-area-events.csv', eventsCsv(l), 'CSV'); break; }
    case 'exportreg': saveFile('agni-netra-site-registry.csv', eventsCsv(watchRows().map(r => r.e)), 'CSV'); break;
    case 'sort': { const k = el.dataset.k; S.sort = S.sort.k === k ? { k, dir: -S.sort.dir } : { k, dir: k === 'name' || k === 'cls' ? 1 : -1 }; $('#wtable').innerHTML = watchTable(); break; }
    case 'wq': S.wq = el.value; { const t = $('#wtable'); if (t) t.innerHTML = watchTable(); } break;
    case 'wst': S.wst = el.value; $('#wtable').innerHTML = watchTable(); break;
    case 'cmpA': S.cmpA = el.value; renderView(); break;
    case 'cmpB': S.cmpB = el.value; renderView(); break;
    case 'capsel': S.capId = el.dataset.id; renderView(); break;
    case 'base': setBase(el.value); renderMap(); break;
    case 'windh': S.windH = +el.value; windSync(); break;
    case 'windstyle': S.windStyle = el.dataset.v; if (LM.wind) LM.wind.setStyle(S.windStyle); $$('[data-act="windstyle"]').forEach(b => b.setAttribute('aria-pressed', b.dataset.v === S.windStyle)); break;
    case 'adv': openAdvanced(el.dataset.id); break;
    case 'advclose': { const pid = A.p ? A.p.id : null; closeAdvanced(); if (pid && S.sel === pid) renderSide(true); break; }
    case 'advtab': A.tab = el.dataset.t; advPanel(); writeHash(); break;
    case 'advlayer': A.show[el.dataset.k] = !A.show[el.dataset.k]; el.setAttribute('aria-pressed', A.show[el.dataset.k]); advDraw(); break;
    case 'advstyle': A.style = el.dataset.v; if (A.wind) A.wind.setStyle(A.style); $$('[data-act="advstyle"]').forEach(b => b.setAttribute('aria-pressed', b.dataset.v === A.style)); break;
    case 'advmeasure': A.measure = !A.measure; el.setAttribute('aria-pressed', A.measure); if (!A.measure) { A.mpts = []; advMeasureDraw(); } if (A.map) A.map.getContainer().style.cursor = A.measure ? 'crosshair' : ''; break;
    case 'advop': A.op = +el.value / 100; advDraw(); break;
    case 'advh': advSetHour(+el.value); break;
    case 'advplay': advPlay(); break;
    case 'advfit': advFit(); break;
    case 'advreplay': advReplay(); break;
    case 'advjson': saveFile(`agni-netra-${A.ev.id}-outlook.json`, JSON.stringify({ event: A.ev.id, place: A.p.name, as_of: fmtDT(A.ev.t), sample_data: true, wind: { hourly_speed_ms: A.W.spd.map(v => +v.toFixed(2)), hourly_from_deg: A.W.dir.map(v => +v.toFixed(0)) }, plume_6h_reach_km: +plumeOf(A.p, A.W, 6).reachKm.toFixed(2), spread: A.sp ? A.sp.horizons.map(z => ({ hours: z.t, reach_km: +z.D.toFixed(2), rate_kmh: [+z.rateLo.toFixed(2), +z.rateHi.toFixed(2)], class: z.cls })) : null }, null, 1), 'JSON'); break;
    case 'advgo': openAdvanced(el.dataset.id, A.tab); break;
    case 'exq': A.ex.q = el.value; exRefresh(); break;
    case 'exg': A.ex.g = el.value; exRefresh(); break;
    case 'exo': A.ex.only = el.value; exRefresh(); break;
    case 'exs': A.ex.sort = el.value; exRefresh(); break;
    case 'exexport': saveFile(`agni-netra-${A.ev.id}-features.csv`, exCsv(), 'CSV'); break;
    case 'rvpick': { const c = $('#rv-cls'); if (c) c.style.display = el.value === 'change' ? '' : 'none'; break; }
    case 'rvsave': { const v = ($('input[name="rv"]:checked') || {}).value || 'confirm', note = ($('#rv-note') || {}).value || '', cls = v === 'change' ? $('#rv-sel').value : null; (REVIEWS[A.ev.id] = REVIEWS[A.ev.id] || []).push({ verdict: v, cls, note: note.trim(), at: fmtDT(TODAY) }); saveReviews(); toast('Review saved on this device'); advPanel(); break; }
    case 'rvexport': saveFile('agni-netra-analyst-reviews.csv', reviewsCsv(), 'CSV'); break;
    case 'capsend': { const ev = EVENTS.find(x => x.id === el.dataset.id); S.capLog.unshift({ t: fmtDT(TODAY), n: PMAP[ev.pid].name }); toast('Mock send logged. Nothing was transmitted.'); renderView(); break; }
    case 'capcopy': { const ev = EVENTS.find(x => x.id === el.dataset.id); copyText(capXml(ev), 'Alert draft copied'); break; }
  }
}
const _renderSide = renderSide;
renderSide = function (keep) { const b = $('#side-body'), st = b ? b.scrollTop : 0; _renderSide(); if (keep && b) b.scrollTop = st; };

function init() {
  const saved = store.get('agni-theme'), pref = typeof matchMedia !== 'undefined' && matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  S.theme = saved || 'dark'; document.documentElement.setAttribute('data-theme', S.theme);
  (store.get('agni-watch') || '').split(',').filter(Boolean).forEach(x => S.watch.add(x));
  S.cmpA = (PLACES.find(p => p.code === 'JH-01') || PLACES[0]).id; S.cmpB = (PLACES.find(p => p.code === 'JH-02') || PLACES[1]).id;
  readHash(); shell(); if (S.sel && S.page === 'map') { const p = PMAP[S.sel]; LM.saved = { lat: p.lat, lon: p.lon, zoom: 9 }; }
  renderView();
}
document.addEventListener('click', e => {
  const a = e.target.closest('[data-act]'); const n = a && a.dataset.act;
  if (!a || ['status', 'sensor', 'conf', 'asof', 'layer', 'cmpA', 'cmpB', 'wq', 'wst', 'advh', 'advop', 'exq', 'exg', 'exo', 'exs', 'windh', 'base', 'rvpick'].includes(n)) { if (!e.target.closest('.search')) { const b = $('#sugg'); if (b && !b.hidden) b.hidden = true; } return; }
  if (e.target.closest('.star') && a.classList.contains('click')) return;
  act(n, a, e); e.stopPropagation();
});
document.addEventListener('input', e => { if (e.target.id === 'q') return onSearch(); const a = e.target.closest('[data-act]'); if (a && ['conf', 'asof', 'wq', 'advh', 'advop', 'exq', 'windh'].includes(a.dataset.act)) act(a.dataset.act, a, e); });
document.addEventListener('change', e => { const a = e.target.closest('[data-act]'); if (a && ['status', 'sensor', 'layer', 'cmpA', 'cmpB', 'wst', 'exg', 'exo', 'exs', 'base', 'rvpick'].includes(a.dataset.act)) act(a.dataset.act, a, e); });
document.addEventListener('pointerover', e => { const t = e.target.closest('[data-tip]'); if (t) { const r = t.getBoundingClientRect(); showTip(t.dataset.tip, r.left, r.bottom); } });
document.addEventListener('pointerout', e => { if (e.target.closest('[data-tip]')) hideTip(); });
document.addEventListener('keydown', e => {
  const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName);
  if (e.key === '/' && !typing) { e.preventDefault(); const q = $('#q'); if (q) q.focus(); return; }
  if (e.target.id === 'q') { const items = $$('#sugg button'); if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); S.si = clamp(S.si + (e.key === 'ArrowDown' ? 1 : -1), 0, Math.max(0, items.length - 1)); items.forEach((b, i) => b.classList.toggle('on', i === S.si)); } else if (e.key === 'Enter' && sugg[S.si]) goSel(sugg[S.si].id); else if (e.key === 'Escape') { closeSearch(); e.target.blur(); } return; }
  if (typing) return;
  if (e.key === 'Escape') { if (A.open) { closeAdvanced(); return; } if (S.drawing) { S.drawing = false; syncDrawBtn(); } else if (S.sel || S.area) clearSel(); }
  if (S.page === 'map' && (e.key === 'j' || e.key === 'k')) { const l = queueList(); if (!l.length) return; let i = l.findIndex(g => g.p.id === S.sel); i = clamp(i + (e.key === 'j' ? 1 : -1), 0, l.length - 1); select(l[i].p.id); }
  if (S.page === 'map' && e.key === 'l') { const b = $('[data-act="layers"]'); if (b) b.click(); }
});
window.addEventListener('resize', () => { CH.forEach(c => { try { c.resize(); } catch (e) {} }); if (S.page === 'map') mapInvalidate(); if (A.map) A.map.invalidateSize(); A.ach.forEach(c => { try { c.resize(); } catch (e) {} }); if (A.tch) { try { A.tch.resize(); } catch (e) {} } });
window.addEventListener('hashchange', () => { if (hashLock) return; readHash(); renderView(); });
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
