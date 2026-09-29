'use strict';
/* ===== Helpers (see js/core.js for $, $$, esc, clamp, cv, ccol, alpha, hexMix, tk, REDUCED, store, lc) ===== */

/* ===== Icons ===== */
const IC = {
  map: '<path d="M3 6.5l6-2.5 6 2.5 6-2.5v13l-6 2.5-6-2.5-6 2.5z"/><path d="M9 4v13.5M15 6.5V20"/>',
  bars: '<path d="M4 20v-7M10 20V5M16 20v-10M22 20H3"/>',
  star: '<path d="M12 3.5l2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.9 6.8 19.6l1-5.8L3.5 9.7l5.9-.9z"/>',
  gauge: '<path d="M4 17a8 8 0 1 1 16 0"/><path d="M12 17l4-5"/>',
  book: '<path d="M5 5a2 2 0 0 1 2-2h11v16H7a2 2 0 0 0-2 2z"/><path d="M5 19V5"/>',
  send: '<path d="M21 3L3 10.5l6.5 2.5L12 20z"/><path d="M9.5 13L21 3"/>',
  search: '<circle cx="11" cy="11" r="6.5"/><path d="M16 16l4.5 4.5"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M5 5l1.8 1.8M17.2 17.2L19 19M5 19l1.8-1.8M17.2 6.8L19 5"/>',
  moon: '<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>',
  layers: '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>',
  draw: '<rect x="4" y="4" width="16" height="16" rx="2" stroke-dasharray="3 3"/>',
  plus: '<path d="M12 5v14M5 12h14"/>', minus: '<path d="M5 12h14"/>',
  play: '<path d="M7 4.5v15l12-7.5z" fill="currentColor"/>', pause: '<path d="M8 5v14M16 5v14"/>',
  x: '<path d="M6 6l12 12M18 6L6 18"/>',
  expand: '<path d="M4 10V4h6M20 14v6h-6M4 4l6 6M20 20l-6-6"/>',
  shrink: '<path d="M10 4v6H4M14 20v-6h6M10 10L4 4M14 14l6 6"/>',
  download: '<path d="M12 4v11M7 11l5 5 5-5M5 20h14"/>',
  copy: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/>',
  back: '<path d="M15 5l-7 7 7 7"/>',
  reset: '<circle cx="12" cy="12" r="6.5"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/>',
  check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>'
};
const icon = (n, s = 18) => `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${IC[n]}</svg>`;
const LOGO = `<svg width="34" height="34" viewBox="0 0 34 34" aria-hidden="true"><path d="M3 17c4-7 9-10.5 14-10.5S27 10 31 17c-4 7-9 10.5-14 10.5S7 24 3 17z" fill="none" stroke="currentColor" stroke-width="1.6"/><circle cx="17" cy="17" r="6.2" fill="none" stroke="currentColor" stroke-width="1.6"/><circle cx="17" cy="17" r="2.4" fill="currentColor"/></svg>`;

/* class glyphs: shape carries the meaning as well as colour */
function glyphPath(cls, r, cx = 0, cy = 0) {
  switch (cls) {
    case 'wildfire': return `M${cx} ${cy - r * 1.1}L${cx + r * 1.05} ${cy + r * .8}L${cx - r * 1.05} ${cy + r * .8}Z`;
    case 'agricultural_burning': return `M${cx - r * .85} ${cy - r * .85}h${r * 1.7}v${r * 1.7}h${-r * 1.7}Z`;
    case 'mining': { let d = ''; for (let i = 0; i < 6; i++) { const a = Math.PI / 3 * i + Math.PI / 6; d += (i ? 'L' : 'M') + (cx + r * 1.05 * Math.cos(a)).toFixed(2) + ' ' + (cy + r * 1.05 * Math.sin(a)).toFixed(2); } return d + 'Z'; }
    case 'industrial': return `M${cx} ${cy - r * 1.2}L${cx + r * 1.2} ${cy}L${cx} ${cy + r * 1.2}L${cx - r * 1.2} ${cy}Z`;
    default: return `M${cx - r} ${cy}a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0Z`;
  }
}
const glyph = (cls, size = 12) => { const r = size / 2 - 1.2, c = ccol(cls); return `<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" aria-hidden="true"><path d="${glyphPath(cls, r, size / 2, size / 2)}" ${cls === 'unknown' ? `fill="none" stroke="${c}" stroke-width="1.6"` : `fill="${c}"`}/></svg>`; };
const cbadge = cls => `<span class="cb">${glyph(cls, 12)}${esc(CLSMAP[cls].label)}</span>`;
const schip = st => `<span class="chip ${STATUS[st].k}" data-tip="${esc(STATUS[st].tip)}">${STATUS[st].label}</span>`;
const rchip = () => `<span class="chip rev" data-tip="Confidence is below 60%. A person should check this before anyone acts on it.">Needs review</span>`;
function ring(v, size = 38, sw = 4, col) {
  const r = (size - sw) / 2, c = 2 * Math.PI * r, big = size > 60;
  return `<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" role="img" aria-label="Confidence ${Math.round(v * 100)} percent"><circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="var(--line2)" stroke-width="${sw}"/><circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="${col}" stroke-width="${sw}" stroke-linecap="round" stroke-dasharray="${(c * v).toFixed(1)} ${c.toFixed(1)}" transform="rotate(-90 ${size / 2} ${size / 2})"/><text x="50%" y="50%" text-anchor="middle" dominant-baseline="central" fill="currentColor" font-size="${big ? 21 : 11.5}" font-weight="600" style="font-variant-numeric:tabular-nums">${Math.round(v * 100)}${big ? '%' : ''}</text></svg>`;
}
function spark(vals, w = 96, h = 26, col = 'currentColor') {
  if (!vals.length) return '';
  const mx = Math.max(...vals, 1), n = vals.length;
  const pts = vals.map((v, i) => `${(i / Math.max(n - 1, 1) * w).toFixed(1)},${(h - 2 - v / mx * (h - 4)).toFixed(1)}`).join(' ');
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-hidden="true"><polyline points="${pts}" fill="none" stroke="${col}" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/></svg>`;
}
const GL = {
  frp: 'Fire radiative power (FRP) is how much heat a fire gives off, in megawatts. Bigger is hotter or larger.',
  mad: 'MAD is the median absolute deviation. It measures how much a site normally varies, and it is not thrown off by a few extreme days the way an average is.',
  z: 'The robust z-score says how far today\'s fire power sits above this site\'s own normal level, in units of normal variation. Above about 3.5 is unusual.',
  conf: 'Calibrated confidence: when the system says 80%, it should be right about 8 times in 10 on held-out data.',
  baseline: 'A baseline is a site\'s own normal range. It needs about 20 clear observations before it is trusted.',
  shap: 'Each bar shows how much a group of inputs pushed the system toward this class. Bars to the right support the call, bars to the left argue against it.',
  gap: 'Cloud or no satellite pass. A gap is missing data, not zero fire.',
  footprint: 'Footprint is the area covered by neighbouring detections grouped into one event. Observed, not predicted.',
  vpd: 'Vapour pressure deficit measures how dry the air is. Dry air helps vegetation fires spread.',
  plume: 'The smoke plume drifts with the forecast wind, hour by hour. It shows where smoke can travel, not how thick it is. The paler outline allows for the wind forecast being a little off.',
  spreadind: 'A rough guide to how fast a vegetation fire could advance, from wind, how dry the fuel is, and slope. It has not been checked against real fires, so treat it as indicative.',
  fwi: 'The Canadian Fire Weather Index system turns temperature, humidity, wind and rain into indices. FFMC is how dry the finest fuel is, ISI how fast a fire would spread, and FWI overall fire intensity.',
  elev: 'Ground height along the dashed line on the map, downwind of the source. Uphill stretches speed a fire up.',
  smokeopt: 'Smoke and burn signs read from optical satellite images. Cloud can hide them, and the scene may be days old.',
  sar: 'Radar sees through cloud and measures how the ground surface changed. Passes are a few days apart.',
  chem: 'Gas readings from a satellite that passes once a day, compared with what is normal for the place and season. They point to burning conditions but cannot prove a source.',
  cover: 'How complete the facility records (Global Energy Monitor and OpenStreetMap) are around this place. Weak coverage means a nearby facility may simply not be mapped.'
};
const info = k => `<button class="info" type="button" data-tip="${esc(GL[k])}" aria-label="What is this?">i</button>`;

/* tooltip + toast */
let TIP = null;
function showTip(html, x, y) {
  if (!TIP) { TIP = document.createElement('div'); TIP.className = 'tip'; TIP.setAttribute('role', 'tooltip'); document.body.appendChild(TIP); }
  TIP.innerHTML = html; TIP.style.display = 'block';
  const w = TIP.offsetWidth, h = TIP.offsetHeight;
  TIP.style.left = clamp(x + 14, 8, innerWidth - w - 8) + 'px';
  TIP.style.top = clamp(y + 14, 8, innerHeight - h - 8) + 'px';
}
const hideTip = () => { if (TIP) TIP.style.display = 'none'; };
let toastT = null;
function toast(msg) {
  let t = $('.toast'); if (!t) { t = document.createElement('div'); t.className = 'toast'; t.setAttribute('role', 'status'); document.body.appendChild(t); }
  t.textContent = msg; t.style.display = 'block'; clearTimeout(toastT); toastT = setTimeout(() => { t.style.display = 'none'; }, 2600);
}
async function copyText(txt, okMsg) {
  try { await navigator.clipboard.writeText(txt); toast(okMsg || 'Copied'); return; } catch (e) {}
  try { const ta = document.createElement('textarea'); ta.value = txt; ta.style.position = 'fixed'; ta.style.opacity = '0'; document.body.appendChild(ta); ta.select(); document.execCommand('copy'); ta.remove(); toast(okMsg || 'Copied'); }
  catch (e) { toast('Could not copy in this view'); }
}
let DL = null;
(async () => { try { if (window.claude && window.claude.use) DL = await window.claude.use('downloads'); } catch (e) { DL = null; } })();
async function saveFile(filename, data, label) {
  if (DL) { try { await DL.save({ filename, data }); toast('Saved ' + filename); } catch (e) { if (!e || e.code !== 'declined') toast('Could not save the file'); } }
  else if (!window.claude) {
    try { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([data], { type: 'text/plain;charset=utf-8' })); a.download = filename; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 1500); toast('Saved ' + filename); }
    catch (e) { await copyText(data, (label || 'File') + ' copied to clipboard'); }
  }
  else await copyText(data, (label || 'File') + ' copied to clipboard (saving files is not available in this view)');
}

/* ===== State ===== */
const S = {
  page: 'map', asOf: TODAY, win: 90, cls: new Set(CLS.map(c => c.id)), status: 'all', sensor: 'all', minConf: 0,
  layers: { raw: true, events: true, facilities: true, density: false, wind: false }, windH: 0, windStyle: REDUCED ? 'arrows' : 'flow', showLayers: false,
  sel: null, evId: null, area: null, expanded: false, drawing: false, play: false, q: '', si: 0,
  theme: 'dark', watch: new Set(), sort: { k: 't', dir: -1 }, wq: '', wst: 'all', cmpA: 'S037', cmpB: 'S039', qshow: 30, shapAll: false, capId: null, capLog: [], capXml: false
};
const sev = e => e.status === 'abnormal' ? 3 : e.review ? 2 : 1;
function passes(e) {
  if (!S.cls.has(e.cls)) return false;
  if (S.status === 'review') { if (!e.review) return false; } else if (S.status !== 'all' && e.status !== S.status) return false;
  if (S.sensor !== 'all' && !e.sensor.startsWith(S.sensor)) return false;
  return e.conf >= S.minConf;
}
const winEvents = (off = 0) => { const hi = S.asOf - off * S.win * DAY, lo = hi - S.win * DAY; return EVENTS.filter(e => e.t > lo && e.t <= hi && passes(e)); };
function groups(evs) {
  const m = new Map();
  evs.forEach(e => { let g = m.get(e.pid); if (!g) { g = { p: PMAP[e.pid], evs: [] }; m.set(e.pid, g); } g.evs.push(e); });
  return [...m.values()].map(g => { g.top = g.evs.slice().sort((a, b) => (sev(b) - sev(a)) || (b.t - a.t))[0]; return g; });
}
function pickEvent(p) {
  const evs = EVP[p.id] || [];
  if (S.evId) { const e = evs.find(x => x.id === S.evId); if (e) return e; }
  const inW = evs.filter(e => e.t <= S.asOf && e.t > S.asOf - S.win * DAY);
  const pool = inW.length ? inW : evs.filter(e => e.t <= S.asOf);
  return (pool.length ? pool : evs).slice().sort((a, b) => (sev(b) - sev(a)) || (b.t - a.t))[0];
}

/* ===== Filters, KPI strip, scrubber, queue ===== */
function renderFilters() {
  const el = $('#filters'); if (!el) return;
  el.innerHTML = CLS.map(c => `<button class="fchip" data-act="tcls" data-id="${c.id}" aria-pressed="${S.cls.has(c.id)}" data-tip="${esc(c.blurb)}">${glyph(c.id, 12)}${esc(c.label)}</button>`).join('') +
    `<span class="row gap4"><label class="dim small" for="fst">Status</label><select id="fst" class="sel" data-act="status" aria-label="Status filter">${[['all', 'All'], ['abnormal', 'Abnormal'], ['routine', 'Routine'], ['baseline_building', 'Baseline building'], ['not_applicable', 'Transient fire'], ['review', 'Needs review']].map(([v, l]) => `<option value="${v}" ${S.status === v ? 'selected' : ''}>${l}</option>`).join('')}</select></span>` +
    `<span class="row gap4"><label class="dim small" for="fse">Sensor</label><select id="fse" class="sel" data-act="sensor" aria-label="Sensor filter">${[['all', 'All'], ['VIIRS', 'VIIRS 375 m'], ['MODIS', 'MODIS 1 km']].map(([v, l]) => `<option value="${v}" ${S.sensor === v ? 'selected' : ''}>${l}</option>`).join('')}</select></span>` +
    `<label class="conf" for="fco">Minimum confidence <input id="fco" type="range" min="0" max="90" step="5" value="${Math.round(S.minConf * 100)}" data-act="conf"><b class="num" id="fcov" style="min-width:34px">${Math.round(S.minConf * 100)}%</b></label>`;
}
function series(evs, fn, buckets) {
  const lo = S.asOf - S.win * DAY, bw = S.win * DAY / buckets, out = Array.from({ length: buckets }, () => []);
  evs.forEach(e => { const i = clamp(Math.floor((e.t - lo) / bw), 0, buckets - 1); out[i].push(e); });
  return out.map(fn);
}
function renderKpis() {
  const el = $('#kpis'); if (!el) return; const cur = winEvents(0), prev = winEvents(1), nb = Math.min(24, Math.max(7, S.win > 30 ? 18 : S.win));
  const agg = evs => ({ det: evs.reduce((a, e) => a + e.nDet, 0), ev: evs.length, abn: new Set(evs.filter(e => e.status === 'abnormal').map(e => e.pid)).size, sites: new Set(evs.filter(e => PMAP[e.pid].kind === 'site').map(e => e.pid)).size, rev: evs.filter(e => e.review).length });
  const a = agg(cur), b = agg(prev);
  const defs = [
    ['Detections', a.det, b.det, series(cur, x => x.reduce((s, e) => s + e.nDet, 0), nb), 'Raw satellite hot pixels grouped into events in this window.', false],
    ['Events', a.ev, b.ev, series(cur, x => x.length, nb), 'Groups of nearby detections treated as one occurrence.', false],
    ['Abnormal places', a.abn, b.abn, series(cur, x => x.filter(e => e.status === 'abnormal').length, nb), 'Places behaving unusually compared with their own history.', true],
    ['Active sites', a.sites, b.sites, series(cur, x => new Set(x.filter(e => PMAP[e.pid].kind === 'site').map(e => e.pid)).size, nb), 'Known industrial or mining sites with at least one event in this window.', false],
    ['Needs review', a.rev, b.rev, series(cur, x => x.filter(e => e.review).length, nb), 'Events where confidence is below 60%, so a person should check.', false]
  ];
  el.innerHTML = defs.map(([l, v, pv, sr, tip, alarm]) => {
    const d = pv ? Math.round((v - pv) / pv * 100) : null, dt = d == null ? (v ? 'new' : 'no change') : (d > 0 ? '+' : '') + d + '%';
    return `<div class="kpi ${alarm && v ? 'alarm' : ''}"><div class="l">${l}<button class="info" type="button" data-tip="${esc(tip)}" aria-label="About ${l}">i</button></div><div class="v num">${nf(v)}</div><div class="d"><b class="num">${dt}</b><span>vs previous ${S.win} d</span></div><div style="color:var(--dim);margin-top:4px">${spark(sr, 120, 24, alarm && v ? 'var(--danger)' : 'currentColor')}</div></div>`;
  }).join('');
}
function renderScrub() {
  const el = $('#scrub'); if (!el) return;
  const cnt = new Array(365).fill(0); EVENTS.forEach(e => { if (passes(e)) { const i = Math.floor((e.t - START) / DAY); if (i >= 0 && i < 365) cnt[i]++; } });
  const mx = Math.max(...cnt, 1), W = 730, H = 42, bw = W / 365, ai = (S.asOf - START) / DAY, x1 = ai * bw, x0 = Math.max(0, (ai - S.win) * bw);
  let bars = ''; cnt.forEach((c, i) => { if (c) { const h = Math.max(2, c / mx * (H - 12)); bars += `<rect x="${(i * bw).toFixed(2)}" y="${H - 12 - h}" width="${Math.max(1, bw - .5).toFixed(2)}" height="${h}" style="fill:var(--dim)"/>`; } });
  let ticks = ''; for (let i = 0; i < 365; i++) { const t = START + i * DAY, p = dparts(t); if (p.d === 1) ticks += `<text x="${(i * bw).toFixed(1)}" y="${H - 1}" style="fill:var(--dim);font-size:9.5px">${MONTHS[p.m]}</text>`; }
  el.innerHTML = `<button class="btn ic" data-act="play" aria-label="${S.play ? 'Pause' : 'Play'} history">${icon(S.play ? 'pause' : 'play', 15)}</button>
  <div class="hist"><svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true"><rect x="${x0}" y="0" width="${Math.max(2, x1 - x0)}" height="${H - 12}" style="fill:var(--tx);fill-opacity:.12;stroke:var(--tx);stroke-opacity:.55;stroke-width:1px;vector-effect:non-scaling-stroke"/>${bars}${ticks}</svg><input type="range" min="0" max="364" step="1" value="${Math.round(ai)}" data-act="asof" aria-label="Show events up to this date"></div>
  <div class="dl"><b class="num">${fmtS(S.asOf - S.win * DAY + DAY)} to ${fmtD(S.asOf)}</b><span class="seg" style="margin-top:4px">${[7, 30, 90, 365].map(w => `<button data-act="win" data-w="${w}" aria-pressed="${S.win === w}">${w === 365 ? '1 y' : w + ' d'}</button>`).join('')}</span></div>`;
}
const qItem = g => { const e = g.top, p = g.p; return `<button class="qi" data-act="sel" data-id="${p.id}" aria-current="${S.sel === p.id}"><span>${ring(e.conf, 40, 4, e.review ? cv('--warn') : ccol(e.cls))}</span><span style="min-width:0"><span class="t" style="display:block">${esc(p.name)}</span><span class="s" style="display:block">${esc(p.district)}, ${esc(p.state)}</span><span class="chips">${cbadge(e.cls)}${schip(e.status)}${e.review ? rchip() : ''}${reviewedChip(e)}</span></span><span class="m"><b class="num">${e.frp.toFixed(0)} MW</b><span>${ago(e.t)}</span>${g.evs.length > 1 ? `<span>${g.evs.length} events</span>` : ''}</span></button>`; };
function queueList() { return CUR.groups.slice().sort((a, b) => (sev(b.top) - sev(a.top)) || (b.top.t - a.top.t)); }
function renderSide() {
  const head = $('#side-head'), body = $('#side-body'); if (!head || !body) return;
  disposeCharts();
  if (S.area) {
    head.innerHTML = `<button class="btn sm" data-act="back">${icon('back', 14)}All incidents</button><h2>Area report</h2><span class="sp" style="flex:1"></span><button class="btn ic sm" data-act="expand" aria-label="${S.expanded ? 'Shrink' : 'Expand'} report">${icon(S.expanded ? 'shrink' : 'expand', 15)}</button>`;
    const r = areaReport(); body.innerHTML = `<div class="${S.expanded ? 'rep-wide' : 'rep-stack'}">${r.html}</div>`; r.after(); body.scrollTop = 0; return;
  }
  if (S.sel) {
    const p = PMAP[S.sel];
    head.innerHTML = `<button class="btn sm" data-act="back">${icon('back', 14)}${S.expanded ? 'Back to map' : 'All incidents'}</button><h2 style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis">Place report</h2><span style="flex:1"></span><button class="btn ic sm" data-act="expand" aria-label="${S.expanded ? 'Shrink' : 'Expand'} report">${icon(S.expanded ? 'shrink' : 'expand', 15)}</button><button class="btn ic sm" data-act="close" aria-label="Close report">${icon('x', 15)}</button>`;
    const r = placeReport(p); body.innerHTML = `<div class="${S.expanded ? 'rep-wide' : 'rep-stack'}">${r.html}</div>`; r.after(); body.scrollTop = 0; return;
  }
  const list = queueList(), abn = list.filter(g => sev(g.top) === 3).length;
  head.innerHTML = `<h2>Priority queue</h2><span class="dim small">${list.length} places${abn ? `, ${abn} abnormal` : ''}</span>`;
  if (!list.length) { body.innerHTML = `<div class="empty"><b>Nothing matches these filters</b>Widen the date window, turn more classes on, or lower the minimum confidence.</div>`; return; }
  body.innerHTML = `<div class="dim small" style="padding:0 2px">Abnormal first, then needs review, then newest.</div><div class="queue">${list.slice(0, S.qshow).map(qItem).join('')}</div>${list.length > S.qshow ? `<button class="btn" data-act="moreq" style="justify-self:center">Show ${Math.min(30, list.length - S.qshow)} more</button>` : ''}`;
}
function syncCmd() { const c = $('#cmd'); if (c) c.classList.toggle('expanded', S.expanded && (S.sel || S.area)); }
function refreshMapPage() { renderFilters(); renderKpis(); renderScrub(); renderMap(); renderSide(); }
function select(id) {
  S.sel = id; S.area = null; S.evId = null; S.shapAll = false; S.capXml = false; S.expanded = false; syncCmd(); writeHash();
  renderMap(); renderSide(); const p = PMAP[id]; if (p) flyTo(p.lon, p.lat, Math.max(LM.map ? LM.map.getZoom() : 4, 8.5));
}
function clearSel() { S.sel = null; S.area = null; S.evId = null; S.expanded = false; syncCmd(); writeHash(); renderMap(); renderSide(); }
