'use strict';
/* =====================================================================
   Advanced analysis workspace.
   Opens from any place report. Left: sections. Middle: a Leaflet map with
   wind, smoke plume and fire-spread layers plus a forecast timeline.
   Right: the section content (see tabs.js).
   ===================================================================== */
const A = { open: false, p: null, ev: null, W: null, tab: 'wind', h: 6, map: null, g: {}, wind: null, style: REDUCED ? 'arrows' : 'flow', show: { wind: true, plume: true, spread: true, rings: true, foot: true, terr: true, near: false }, op: .8, ach: [], playT: null, fpT: null, measure: false, mpts: [], pl: null, sp: null, tch: null, fell: false, ex: { q: '', g: 'all', only: 'all', sort: 'pct' } };
const ANAV = [
  ['Outlook', [['wind', 'Wind and spread']]],
  ['What we measured', [['thermal', 'Heat'], ['temporal', 'Timing'], ['spread', 'Movement and growth'], ['terrain', 'Terrain'], ['fuel', 'Fuel and land cover'], ['industry', 'Industry nearby'], ['weather', 'Weather and fire danger'], ['air', 'Smoke, radar and air'], ['quality', 'Data quality']]],
  ['Tools', [['explorer', 'All 141 features'], ['exposure', 'Who is in the way'], ['similar', 'Similar events'], ['review', 'Analyst review']]]
];
const ATITLES = Object.fromEntries(ANAV.flatMap(g => g[1]));

function achart(target, opt, h) {
  const el = typeof target === 'string' ? $(target) : target; if (!el) return null;
  if (!window.echarts) { el.innerHTML = '<div class="empty">Charts could not load in this view.</div>'; return null; }
  if (h) el.style.height = h + 'px';
  const c = echarts.init(el, null, { renderer: 'svg' }); c.setOption(Object.assign({ textStyle: { fontFamily: cv('--font') }, animationDuration: 300 }, opt)); A.ach.push(c); return c;
}
const achDispose = () => { A.ach.splice(0).forEach(c => { try { c.dispose(); } catch (e) {} }); A._tm = null; };

/* ---------- open and close ---------- */
function openAdvanced(pid, tab) {
  const p = PMAP[pid]; if (!p || typeof L === 'undefined') return;
  closeAdvanced(true);
  const ev = pickEvent(p); A.p = p; A.ev = ev; A.W = weatherOf(p, ev); A.tab = tab || A.tab || 'wind'; A.h = 6; A.open = true; A.fell = false; A.measure = false; A.mpts = [];
  const host = $('#view'); if (!host) return; const el = document.createElement('div'); el.id = 'adv'; el.className = 'adv'; el.setAttribute('role', 'dialog'); el.setAttribute('aria-label', 'Advanced analysis'); el.innerHTML = advShell(); host.appendChild(el);
  advMapInit(); advDraw(); advPanel(); advTimeChart(); writeHash();
}
function closeAdvanced(silent) {
  clearInterval(A.playT); clearInterval(A.fpT); A.playT = A.fpT = null; achDispose(); if (A.tch) { try { A.tch.dispose(); } catch (e) {} A.tch = null; }
  if (A.map) { try { A.map.remove(); } catch (e) {} A.map = null; } A.wind = null; A.g = {};
  const el = $('#adv'); if (el) el.remove(); const was = A.open; A.open = false; if (!silent && was) writeHash();
}
function advShell() {
  const p = A.p, ev = A.ev;
  return `<header class="adv-top"><button class="btn sm" data-act="advclose">${icon('back', 14)}Back to the report</button>
    <div style="min-width:0"><h2>${esc(p.name)}</h2><div class="dim small">${esc(p.district)}, ${esc(p.state)}. Conditions as of ${fmtDT(ev.t)}.</div></div>
    <span class="row wrap" style="margin-left:6px">${cbadge(ev.cls)}${schip(ev.status)}${ev.review ? rchip() : ''}</span><span style="flex:1"></span>
    <span class="chip prop" data-tip="Wind, smoke and spread here are sample values shaped like real model output. Your backend will supply the real ones.">Sample outlook</span></header>
  <div class="adv-body">
    <nav class="adv-nav" aria-label="Analysis sections">${ANAV.map(([g, items]) => `<div class="adv-navg"><div class="adv-navh">${g}</div>${items.map(([id, l]) => `<button data-act="advtab" data-t="${id}" aria-current="${A.tab === id}">${l}${FGROUP_COUNT[id] ? `<span class="n">${FGROUP_COUNT[id]}</span>` : ''}</button>`).join('')}</div>`).join('')}</nav>
    <section class="adv-main">
      <div class="adv-mapwrap"><div id="advmap"></div>
        <div class="ov tl advtools" id="advtools">${advToolsHtml()}</div>
        <div class="ov tr advtools2"><div class="seg" role="group" aria-label="Wind style"><button data-act="advstyle" data-v="flow" aria-pressed="${A.style === 'flow'}">Flow</button><button data-act="advstyle" data-v="arrows" aria-pressed="${A.style === 'arrows'}">Arrows</button></div><button class="btn sm" data-act="advmeasure" aria-pressed="${A.measure}">${icon('draw', 14)}Measure</button></div>
        <div class="ov advkey" id="advkey"></div>
        <div class="ov advop"><label class="row small" for="advop">Overlay strength <input id="advop" type="range" min="25" max="100" value="${Math.round(A.op * 100)}" data-act="advop"></label></div>
      </div>
      <div class="adv-time"><div class="row between"><div class="row"><button class="btn ic sm" data-act="advplay" aria-label="Play forecast">${icon('play', 14)}</button><b id="advhl" class="num"></b></div><span class="dim small">Wind, gusts and humidity, next 72 hours</span></div>
        <div id="advtc" style="height:118px"></div><div class="advslide"><input type="range" min="0" max="72" step="1" value="${A.h}" data-act="advh" aria-label="Forecast hour"></div></div>
    </section>
    <aside class="adv-panel" id="advpanel" aria-live="polite"></aside>
  </div>`;
}
function advToolsHtml() {
  const t = [['wind', 'Wind'], ['plume', 'Smoke plume'], ['spread', 'Fire spread'], ['foot', 'Observed footprint'], ['rings', 'Distance rings'], ['terr', 'Terrain line'], ['near', 'Other heat nearby']];
  return t.map(([k, l]) => `<button class="fchip" data-act="advlayer" data-k="${k}" aria-pressed="${A.show[k]}">${l}</button>`).join('');
}

/* ---------- the analysis map ---------- */
function advMapInit() {
  const el = $('#advmap'), p = A.p, map = A.map = L.map(el, { zoomControl: true, attributionControl: true, preferCanvas: true, minZoom: 5, maxZoom: 16, zoomSnap: .25, wheelPxPerZoomLevel: 90 });
  map.attributionControl.setPrefix(false); map.zoomControl.setPosition('bottomright');
  map.createPane('base').style.zIndex = 250; map.createPane('wind').style.zIndex = 350;
  map.setView([p.lat, p.lon], 9);
  if (LM.base === 'plain' || LM.warned) plainBase(map).addTo(map);
  else {
    const t = L.tileLayer(OSM_URL, { maxZoom: 19, attribution: OSM_ATTR, crossOrigin: true }); let ok = 0, err = 0;
    t.on('tileload', () => ok++); t.on('tileerror', () => { err++; if (err >= 6 && !ok && !A.fell) { A.fell = true; map.removeLayer(t); plainBase(map).addTo(map); } }); t.addTo(map);
  }
  A.g = { ter: L.layerGroup().addTo(map), near: L.layerGroup().addTo(map), rings: L.layerGroup().addTo(map), foot: L.layerGroup().addTo(map), plume: L.layerGroup().addTo(map), spread: L.layerGroup().addTo(map), src: L.layerGroup().addTo(map), note: L.layerGroup().addTo(map), meas: L.layerGroup().addTo(map) };
  A.wind = new WindLayer({ pane: 'wind', count: 1100, style: A.style, opacity: A.op });
  map.on('click', e => { if (!A.measure) return; A.mpts.push(e.latlng); advMeasureDraw(); });
}
const glyphIcon = (cls, r, ring) => L.divIcon({ className: 'mkicon', iconSize: [40, 40], iconAnchor: [20, 20], html: `<svg width="40" height="40" viewBox="0 0 40 40" style="overflow:visible"><g transform="translate(20 20)">${ring ? `<circle r="${r * 1.9}" fill="${alpha(cv('--tx'), .1)}" stroke="var(--tx)" stroke-width="1.5"/>` : ''}<path d="${glyphPath(cls, r)}" style="${cls === 'unknown' ? `fill:${alpha(ccol(cls), .2)};stroke:${ccol(cls)};stroke-width:2px` : `fill:${ccol(cls)};stroke:var(--sea);stroke-width:1.5px`}"/></g></svg>` });
const noteIcon = (txt, cls) => L.divIcon({ className: 'notewrap', iconSize: [0, 0], html: `<div class="note ${cls || ''}">${txt}</div>` });
function advDraw() {
  const map = A.map; if (!map) return; const p = A.p, ev = A.ev, W = A.W, h = A.h, col = ccol(ev.cls), T = tk(), op = A.op;
  Object.values(A.g).forEach(g => g.clearLayers()); A._tm = null;
  if (A.show.wind) { if (!map.hasLayer(A.wind)) A.wind.addTo(map); A.wind.setGrid(localGrid(p, W, h)); A.wind.setOpacity(A.op); } else if (map.hasLayer(A.wind)) map.removeLayer(A.wind);
  const pl = A.pl = plumeOf(p, W, Math.max(1, h)), sp = A.sp = spreadOf(p, ev, W), ln = ring => ring.map(q => [q[1], q[0]]);
  if (A.show.plume) {
    L.polygon(ln(pl.outer), { interactive: false, color: col, weight: 1, dashArray: '4 4', opacity: .55 * op, fillColor: col, fillOpacity: .07 * op }).addTo(A.g.plume);
    L.polygon(ln(pl.core), { interactive: false, color: col, weight: 1.4, opacity: .9 * op, fillColor: col, fillOpacity: .26 * op }).addTo(A.g.plume);
    L.polyline(pl.traj.map(q => [q[1], q[0]]), { interactive: false, color: col, weight: 1.4, dashArray: '2 5', opacity: .9 * op }).addTo(A.g.plume);
    [3, 6, 12, 24, 36, 48, 72].filter(t => t <= pl.H).forEach(t => { const q = pl.traj[t * 2]; if (q) L.marker([q[1], q[0]], { interactive: false, icon: L.divIcon({ className: 'tick', html: `<span>+${t} h</span>`, iconSize: [0, 0] }) }).addTo(A.g.plume); });
    L.marker([pl.head[1], pl.head[0]], { interactive: false, icon: noteIcon(`Smoke could reach ${pl.reachKm.toFixed(0)} km by ${hourLabel(W.t0, h || 1)}`, 'tilt') }).addTo(A.g.note);
  }
  if (A.show.spread && sp) {
    L.polygon(ln(sp.unc), { interactive: false, color: col, weight: 1, dashArray: '3 5', opacity: .6 * op, fillOpacity: 0 }).addTo(A.g.spread);
    sp.horizons.slice().reverse().forEach((z, i) => { L.polygon(ln(z.ring), { interactive: false, color: T.tx, weight: 1.4, opacity: op, fillColor: col, fillOpacity: (.12 + .05 * (2 - i)) * op }).addTo(A.g.spread).bindTooltip(`+${z.t} h: about ${z.D.toFixed(1)} km`, { sticky: true, className: 'lt', opacity: 1 }); });
    const z6 = sp.horizons[2], head = offsetLL(p.lon, p.lat, Math.sin(rad(z6.dirTo)) * z6.D * 1000, Math.cos(rad(z6.dirTo)) * z6.D * 1000);
    L.marker([head[1], head[0]], { interactive: false, icon: noteIcon(`Fire could advance about ${z6.D.toFixed(1)} km in 6 h. Indicative only.`, 'warn') }).addTo(A.g.note);
  }
  if (A.show.foot) { const fr = footprintsOf(p, ev), idx = A.fpT ? A.fpIdx : fr.length - 1; fr.slice(0, idx + 1).forEach((f, i) => L.polygon(ln(f.ring), { interactive: false, color: col, weight: i === idx ? 1.8 : .8, opacity: i === idx ? 1 : .5, fillColor: col, fillOpacity: i === idx ? .38 : .06 }).addTo(A.g.foot)); }
  if (A.show.rings) [2, 5, 10].forEach(k => { L.circle([p.lat, p.lon], { radius: k * 1000, interactive: false, color: T.dim, weight: 1, dashArray: '3 5', fill: false, opacity: .8 }).addTo(A.g.rings); const q = offsetLL(p.lon, p.lat, 0, k * 1000); L.marker([q[1], q[0]], { interactive: false, icon: L.divIcon({ className: 'tick', html: `<span>${k} km</span>`, iconSize: [0, 0] }) }).addTo(A.g.rings); });
  if (A.show.terr) {
    const dir = A.sp ? A.sp.horizons[1].dirTo : pl.bearing, tr = terrainAlong(p, dir, 10);
    L.polyline(tr.pts.map(q => [q.lat, q.lon]), { interactive: false, color: T.tx, weight: 1.4, dashArray: '8 6', opacity: .75 * op }).addTo(A.g.ter); A.tr = tr;
  }
  if (A.show.near) nearbyOf(p).forEach(n => { const q = offsetLL(p.lon, p.lat, Math.cos(n.ang) * n.km * 1000, Math.sin(n.ang) * n.km * 1000); L.marker([q[1], q[0]], { icon: glyphIcon(n.cls, 5), keyboard: false }).bindTooltip(`${esc(n.name)}, ${n.km} km`, { className: 'lt', direction: 'top', opacity: 1 }).addTo(A.g.near); });
  L.marker([p.lat, p.lon], { icon: glyphIcon(ev.cls, 8, true), keyboard: false, zIndexOffset: 800 }).bindTooltip(`<b>${esc(p.name)}</b><br>${esc(CLSMAP[ev.cls].label)}`, { className: 'lt', direction: 'top', offset: [0, -12], opacity: 1 }).addTo(A.g.src);
  const sh = windShift(W); if (sh) { const away = (FE().ang.WIND_DIRECTION + 180) % 360, back = (away + 180) % 360, q = offsetLL(p.lon, p.lat, Math.sin(rad(back)) * 3400, Math.cos(rad(back)) * 3400); L.marker([q[1], q[0]], { interactive: false, icon: noteIcon(`Wind turns toward the ${compass(sh.to)} around ${hourLabel(W.t0, sh.h)}`, 'tilt2') }).addTo(A.g.note); }
  advMeasureDraw(true); advKey();
}
function advKey() {
  const el = $('#advkey'); if (!el) return;
  el.innerHTML = `<div class="row small"><span>Wind speed</span><span id="advbar" class="windbar" style="background:${windGradientCss()}"></span><span class="dim">0 to 12+ m/s</span></div>
  <div class="row small" style="gap:12px;flex-wrap:wrap"><span class="row"><i class="sw" style="background:${alpha(ccol(A.ev.cls), .5)}"></i>Smoke, core</span><span class="row"><i class="sw" style="background:${alpha(ccol(A.ev.cls), .12)};border:1px dashed ${ccol(A.ev.cls)}"></i>Where it might go instead</span>${A.sp ? `<span class="row"><i class="sw" style="border:1.5px solid var(--tx)"></i>Fire spread, +1, +3, +6 h</span>` : ''}</div>`;
}
function advMeasureDraw(keep) {
  if (!A.map) return; A.g.meas.clearLayers(); if (!A.mpts.length) return;
  L.polyline(A.mpts, { color: cv('--tx'), weight: 2, interactive: false }).addTo(A.g.meas); let d = 0;
  A.mpts.forEach((q, i) => { L.circleMarker(q, { radius: 3.5, color: cv('--tx'), weight: 1.5, fillColor: cv('--panel'), fillOpacity: 1, interactive: false }).addTo(A.g.meas); if (i) d += A.map.distance(A.mpts[i - 1], q); });
  const last = A.mpts[A.mpts.length - 1]; if (A.mpts.length > 1) L.marker(last, { interactive: false, icon: noteIcon(`${(d / 1000).toFixed(2)} km`, '') }).addTo(A.g.meas);
}

/* ---------- forecast timeline ---------- */
function advTimeChart() {
  const W = A.W, T = tk(), hrs = W.spd.map((_, i) => i), sh = windShift(W);
  A.tch = achart('#advtc', {
    grid: { left: 42, right: 42, top: 14, bottom: 20 }, tooltip: Object.assign(tt(), { trigger: 'axis', formatter: q => { const i = q[0].dataIndex; return `<b>${hourLabel(W.t0, i)}</b><br>Wind ${(W.spd[i] * 3.6).toFixed(0)} km/h from the ${compassShort(W.dir[i])}, gusts ${(W.gust[i] * 3.6).toFixed(0)}<br>Humidity ${W.rh[i].toFixed(0)}%, ${W.temp[i].toFixed(0)}°C${W.rain[i] ? `<br>Rain ${W.rain[i].toFixed(1)} mm` : ''}`; } }),
    xAxis: Object.assign({ type: 'category', data: hrs, boundaryGap: false }, axb(), { splitLine: { show: false }, axisLabel: { color: T.dim, fontSize: 10, interval: 0, formatter: v => (+v % 12 === 0 ? hourLabel(W.t0, +v) : '') } }),
    yAxis: [Object.assign({ type: 'value', name: 'km/h', nameTextStyle: { color: T.dim, fontSize: 10, align: 'left' }, min: 0 }, axb()), Object.assign({ type: 'value', min: 0, max: 100, name: '%', nameTextStyle: { color: T.dim, fontSize: 10, align: 'right' } }, axb(), { splitLine: { show: false } })],
    series: [
      { name: 'Wind', type: 'line', showSymbol: false, data: W.spd.map(v => +(v * 3.6).toFixed(1)), lineStyle: { color: T.tx, width: 2 }, markLine: { silent: true, symbol: 'none', label: { show: false }, lineStyle: { color: T.tx, width: 1.2 }, data: [{ xAxis: A.h }] }, markArea: sh ? { silent: true, itemStyle: { color: alpha(T.warn, .13) }, data: [[{ xAxis: Math.max(0, sh.h - 2) }, { xAxis: sh.h + 3 }]] } : undefined },
      { name: 'Gusts', type: 'line', showSymbol: false, data: W.gust.map(v => +(v * 3.6).toFixed(1)), lineStyle: { color: T.dim, width: 1, type: 'dashed' } },
      { name: 'Humidity', type: 'line', yAxisIndex: 1, showSymbol: false, data: W.rh.map(v => +v.toFixed(0)), lineStyle: { color: ccol('industrial'), width: 1.4 } },
      { name: 'Direction', type: 'scatter', symbol: 'path://M0,-7 L4.5,5 L0,2.5 L-4.5,5 Z', symbolSize: 11, symbolRotate: 0, itemStyle: { color: T.tx2 }, data: hrs.filter(i => i % 6 === 0).map(i => ({ value: [i, +(W.spd[i] * 3.6).toFixed(1) + 4], symbolRotate: -((W.dir[i] + 180) % 360) })), tooltip: { show: false } }
    ]
  });
  A.ach = A.ach.filter(c => c !== A.tch); advHourLabel();
}
function advHourLabel() { const b = $('#advhl'); if (b) b.textContent = A.h === 0 ? 'Now' : `In ${A.h} h, ${hourLabel(A.W.t0, A.h)}`; }
function advSetHour(h) {
  A.h = h; advHourLabel(); advDraw();
  if (A.tch) A.tch.setOption({ series: [{ markLine: { data: [{ xAxis: h }] } }] });
  if (A.tab === 'wind') advLive();
}
function advPlay() {
  if (A.playT) { clearInterval(A.playT); A.playT = null; const b = $('[data-act="advplay"]'); if (b) b.innerHTML = icon('play', 14); return; }
  const b = $('[data-act="advplay"]'); if (b) b.innerHTML = icon('pause', 14); if (A.h >= 72) A.h = 0;
  A.playT = setInterval(() => { A.h = A.h + 1; if (A.h > 72) { clearInterval(A.playT); A.playT = null; A.h = 72; const bb = $('[data-act="advplay"]'); if (bb) bb.innerHTML = icon('play', 14); } const s = $('.advslide input'); if (s) s.value = A.h; advSetHour(A.h); }, 260);
}
function advReplay() {
  if (A.fpT) { clearInterval(A.fpT); A.fpT = null; }
  const fr = footprintsOf(A.p, A.ev); A.fpIdx = 0; A.show.foot = true; advDraw(); const t = $('[data-k="foot"]'); if (t) t.setAttribute('aria-pressed', 'true');
  A.fpT = setInterval(() => { A.fpIdx++; if (A.fpIdx >= fr.length) { clearInterval(A.fpT); A.fpT = null; advDraw(); return; } advDraw(); const c = $('#advreplay'); if (c) c.textContent = `Pass ${A.fpIdx + 1} of ${fr.length}, ${fmtDT(fr[A.fpIdx].t)}`; }, 650);
  if (A.map) A.map.flyTo([A.p.lat, A.p.lon], Math.max(A.map.getZoom(), 10.5), { duration: .6 });
}

/* ---------- panel: tab switching ---------- */
function advPanel() {
  achDispose(); const box = $('#advpanel'); if (!box) return;
  advTimeChartKeep();
  const fn = ATAB[A.tab] || ATAB.wind, r = fn(); box.innerHTML = `<div class="adv-ph"><h3>${esc(ATITLES[A.tab] || '')}</h3></div>${r.html}`; box.scrollTop = 0;
  $$('.adv-nav button').forEach(b => b.setAttribute('aria-current', b.dataset.t === A.tab)); if (r.after) r.after();
}
function advTimeChartKeep() { /* the timeline lives outside the panel, so it is left alone when tabs change */ }
function advFit() {
  if (!A.map || !A.pl) return; let b = L.latLngBounds([[A.p.lat, A.p.lon]]); (A.show.plume ? A.pl.outer : []).forEach(q => b.extend([q[1], q[0]])); if (A.sp && A.show.spread) A.sp.unc.forEach(q => b.extend([q[1], q[0]]));
  A.map.fitBounds(b, { paddingTopLeft: [50, 50], paddingBottomRight: [50, 130], maxZoom: 11, animate: !REDUCED });
}

/* ---------- Wind and spread tab ---------- */
const tile = (l, v, sub) => `<div class="stat"><div class="l">${l}</div><div class="v">${v}</div>${sub ? `<div class="u sm2">${sub}</div>` : ''}</div>`;
function windLive() {
  const p = A.p, ev = A.ev, W = A.W, h = A.h, w = windAt(W, h), i = Math.round(h), rain6 = W.rain.slice(i, i + 6).reduce((a, b) => a + b, 0), pl = A.pl, sp = A.sp;
  const isi = calcISI(W.ffmc[i], w.spd * 3.6), bui = calcBUI(W.dmc, W.dc), fwi = calcFWI(isi, bui);
  let s = `<div class="stats">${tile('Wind', Math.round(w.spd * 3.6) + ' <span class="u">km/h</span>', 'from the ' + compass(w.dir))}${tile('Gusts', Math.round(W.gust[i] * 3.6) + ' <span class="u">km/h</span>')}${tile('Humidity', Math.round(W.rh[i]) + '<span class="u">%</span>')}${tile('Air dryness', W.vpd[i].toFixed(1) + ' <span class="u">kPa</span>', W.vpd[i] > 3 ? 'very dry air' : W.vpd[i] > 1.6 ? 'dry air' : 'damp air')}${tile('Temperature', Math.round(W.temp[i]) + '<span class="u">°C</span>')}${tile('Rain, next 6 h', rain6.toFixed(1) + ' <span class="u">mm</span>')}</div>`;
  s += `<section class="card" style="margin-top:12px"><div class="ch"><h3>Smoke</h3>${info('plume')}<span class="r">Where it can travel</span></div><div class="cbody"><p class="sum" style="margin:0">Smoke released now could drift about <b>${pl.reachKm.toFixed(1)} km</b> toward the ${compass(pl.bearing)} by ${hourLabel(W.t0, h || 1)}. The paler outline shows where it might go if the forecast wind is a little off.</p><p class="dim small" style="margin:8px 0 0">This shows where smoke can travel. It does not say how thick it will be.</p></div></section>`;
  if (sp) {
    const z = sp.horizons[2], why = [`fuel is ${W.ffmc[i] > 85 ? 'very dry' : W.ffmc[i] > 70 ? 'fairly dry' : 'damp'} (FFMC ${W.ffmc[i].toFixed(0)})`, `wind ${Math.round(w.spd * 3.6)} km/h`, z.meanSlope > 1 ? `ground rises ${z.meanSlope.toFixed(0)}° along the wind line, which speeds it up` : z.meanSlope < -1 ? `ground falls ${Math.abs(z.meanSlope).toFixed(0)}° along the wind line, which slows it` : 'flat ground along the wind line'];
    s += `<section class="card" style="margin-top:12px"><div class="ch"><h3>Fire spread outlook</h3>${info('spreadind')}<span class="r"><span class="chip prop">Indicative</span></span></div><div class="cbody"><p class="sum" style="margin:0 0 10px">${sp.cls} spread rate: ${why.join(', ')}.</p>
      <table class="tbl small"><thead><tr><th>In</th><th class="r">Reach</th><th class="r">Speed</th><th>Heading</th><th class="r">Area</th></tr></thead><tbody>${sp.horizons.map(z => `<tr><td>${z.t} h</td><td class="r">${z.D.toFixed(1)} km</td><td class="r">${z.rateLo.toFixed(1)} to ${z.rateHi.toFixed(1)} km/h</td><td>${compass(z.dirTo)}</td><td class="r">${z.area.toFixed(1)} km²</td></tr>`).join('')}</tbody></table>
      <p class="dim small" style="margin:8px 0 0">A rough guide from wind, fuel dryness and slope. It is not a validated fire-front forecast.</p></div></section>`;
  } else s += `<section class="card" style="margin-top:12px"><div class="ch"><h3>Fire spread outlook</h3><span class="r"><span class="chip">Not applicable</span></span></div><div class="cbody"><p class="sum" style="margin:0">This looks like ${lc(CLSMAP[ev.cls].label)}, and it stays inside its site. Spread is only estimated for wildfire and crop burning. For this place, the smoke plume is the part that matters.</p></div></section>`;
  s += `<section class="card" style="margin-top:12px"><div class="ch"><h3>Fire weather at this hour</h3>${info('fwi')}</div><div class="cbody"><div class="stats">${tile('Fuel moisture (FFMC)', W.ffmc[i].toFixed(0))}${tile('Spread index (ISI)', isi.toFixed(1), spreadClass(isi))}${tile('Fire weather (FWI)', fwi.toFixed(1))}</div></div></section>`;
  return s;
}
function advLive() {
  const el = $('#adv-live'); if (el) el.innerHTML = windLive(); advElev();
  const hl = $('#advheadline'); if (hl) hl.textContent = outlookLine(A.p, A.ev, A.W);
}
function advElev() {
  if (!A.tr || !$('#adv-elev')) return; const T = tk(), tr = A.tr, col = ccol(A.ev.cls);
  let c = A.ach.find(x => x.getDom && x.getDom().id === 'adv-elev');
  const opt = { grid: { left: 46, right: 12, top: 12, bottom: 24 }, tooltip: Object.assign(tt(), { trigger: 'axis', formatter: q => `${q[0].axisValue} km along the wind line<br>${Math.round(q[0].data)} m` }),
    xAxis: Object.assign({ type: 'category', data: tr.pts.map(q => q.km.toFixed(1)), boundaryGap: false }, axb(), { axisLabel: { color: T.dim, fontSize: 10, interval: 7 }, splitLine: { show: false } }),
    yAxis: Object.assign({ type: 'value', scale: true, name: 'm', nameTextStyle: { color: T.dim, fontSize: 10, align: 'left' } }, axb()),
    series: [{ type: 'line', data: tr.pts.map(q => +q.elev.toFixed(1)), showSymbol: false, smooth: .3, lineStyle: { color: col, width: 2 }, areaStyle: { color: alpha(col, .16) } }] };
  if (!c) { c = achart('#adv-elev', opt, 150); if (c) c.getZr().on('mousemove', e => { const pt = c.convertFromPixel({ seriesIndex: 0 }, [e.offsetX, e.offsetY]); if (!pt || !A.map || !A.tr) return; const i = clamp(Math.round(pt[0]), 0, A.tr.pts.length - 1), q = A.tr.pts[i]; if (!A._tm) A._tm = L.circleMarker([q.lat, q.lon], { radius: 6, color: cv('--tx'), weight: 2, fillColor: cv('--panel'), fillOpacity: 1, interactive: false }).addTo(A.g.ter); else A._tm.setLatLng([q.lat, q.lon]); }); }
  else c.setOption(opt);
}
function tabWind() {
  const html = `<p class="headline" id="advheadline">${esc(outlookLine(A.p, A.ev, A.W))}</p>
    <div class="row wrap" style="margin:0 0 12px"><button class="btn sm" data-act="advfit">${icon('reset', 14)}Fit the map to the outlook</button><button class="btn sm" data-act="advreplay">${icon('play', 14)}Replay how it grew</button><button class="btn sm" data-act="advjson">${icon('download', 14)}Save the outlook as JSON</button></div>
    <div class="dim small" id="advreplay" style="min-height:18px;margin:-4px 0 8px"></div>
    <div id="adv-live">${windLive()}</div>
    <section class="card" style="margin-top:12px"><div class="ch"><h3>Ground along the wind line</h3>${info('elev')}<span class="r">Hover to see it on the map</span></div><div class="cbody"><div id="adv-elev"></div><p class="dim small" style="margin:6px 0 0">The dashed line on the map. Uphill runs speed a fire up, downhill runs slow it.</p></div></section>
    <details class="card" style="margin-top:12px"><summary class="cbody" style="cursor:pointer;font-weight:600">How this is estimated</summary><div class="cbody" style="padding-top:0"><ul class="list small"><li><b>Wind, humidity and rain</b> come from a weather forecast. Here they are sample values shaped like a real forecast.</li><li><b>Smoke plume:</b> a parcel of smoke drifts with the forecast wind, hour by hour. The outline widens with distance, and the paler edge grows with forecast time to allow for wind error.</li><li><b>Fire spread:</b> the fire weather indices (FFMC, ISI, BUI, FWI) come from the standard Canadian system. Spread speed grows with ISI and with uphill slope, and the shape gets longer as the wind picks up. It is a rough guide, not a validated forecast.</li><li>Nothing here has been checked against real fires yet.</li></ul></div></details>`;
  return { html, after: () => { advElev(); advFit(); } };
}
