'use strict';
/* =====================================================================
   The command map. Leaflet + OpenStreetMap tiles.
   The tile address is one setting below. OSM's public tile server is fine for
   a demo, not for heavy or production traffic: point OSM_URL at your own
   tile provider before you go live. If tiles cannot load (offline), the map
   drops back to a plain India outline that needs no network.
   ===================================================================== */
const OSM_URL = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';
const OSM_ATTR = '© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors';
const IN_BOUNDS = [[6, 68], [37.5, 98]];
const LM = { map: null, g: {}, tile: null, plain: null, base: store.get('agni-base') || 'street', ok: 0, err: 0, warned: false, saved: null, wind: null };
let CUR = { groups: [], evs: [] };

/* hexagon binning helpers (equal-area-ish grid over India) */
const hx = lon => (lon - 67.5) * .927, hy = lat => 38.5 - lat, hlon = x => x / .927 + 67.5, hlat = y => 38.5 - y;
function hexRound(q, r) { let x = q, z = r, y = -x - z, rx = Math.round(x), ry = Math.round(y), rz = Math.round(z); const dx = Math.abs(rx - x), dy = Math.abs(ry - y), dz = Math.abs(rz - z); if (dx > dy && dx > dz) rx = -ry - rz; else if (dy > dz) ry = -rx - rz; else rz = -rx - ry; return [rx, rz]; }
const RAMPS = { dark: ['#1a3550', '#22607f', '#3f95aa', '#90d3d8', '#eaf8f6'], light: ['#dbe7f3', '#a3c6e2', '#5f9dca', '#2f71a8', '#143f6e'] };
function rampAt(t) { const R_ = RAMPS[S.theme], f = clamp(t, 0, 1) * (R_.length - 1), i = Math.min(R_.length - 2, Math.floor(f)); return hexMix(R_[i], R_[i + 1], f - i); }
const ll = ring => ring.map(p => [p[1], p[0]]);

/* plain, offline base: country outline and state lines */
function plainBase(map) {
  const g = L.layerGroup(), rd = L.svg({ pane: 'base', padding: .5 }), land = cv('--land'), st = cv('--state'), co = cv('--coast');
  COUNTRY.forEach(r => L.polygon(ll(r), { pane: 'base', renderer: rd, interactive: false, stroke: false, fillColor: land, fillOpacity: 1 }).addTo(g));
  STATES.forEach(s => s.r.forEach(r => L.polyline(ll(r), { pane: 'base', renderer: rd, interactive: false, color: st, weight: .8 }).addTo(g)));
  COUNTRY.forEach(r => L.polyline(ll(r), { pane: 'base', renderer: rd, interactive: false, color: co, weight: 1.3, lineJoin: 'round' }).addTo(g));
  return g;
}
function setBase(mode, auto) {
  const map = LM.map; if (!map) return; LM.base = mode; if (!auto) store.set('agni-base', mode);
  if (LM.tile && map.hasLayer(LM.tile)) map.removeLayer(LM.tile);
  if (LM.plain && map.hasLayer(LM.plain)) map.removeLayer(LM.plain);
  if (mode === 'street') {
    if (!LM.tile) {
      LM.tile = L.tileLayer(OSM_URL, { maxZoom: 19, attribution: OSM_ATTR, crossOrigin: true });
      LM.tile.on('tileload', () => { LM.ok++; }); LM.tile.on('tileerror', () => { LM.err++; if (LM.err >= 6 && !LM.ok && !LM.warned) { LM.warned = true; setBase('plain', true); toast('Could not reach the map tiles. Showing the offline outline instead.'); } });
    }
    LM.tile.addTo(map);
  } else { LM.plain = plainBase(map); LM.plain.addTo(map); }
  $$('input[name="base"]').forEach(i => { i.checked = i.value === LM.base; });
}

function bindMap() {
  const el = $('#lmap'); if (!el) return;
  if (typeof L === 'undefined') { el.innerHTML = '<div class="empty" style="margin:20px">The map library did not load.</div>'; return; }
  disposeMap();
  const map = LM.map = L.map(el, { zoomControl: false, attributionControl: true, preferCanvas: true, minZoom: 4, maxZoom: 16, zoomSnap: .25, zoomDelta: .5, wheelPxPerZoomLevel: 90 });
  map.attributionControl.setPrefix(false);
  map.createPane('base').style.zIndex = 250; map.createPane('wind').style.zIndex = 350;
  if (LM.saved) map.setView([LM.saved.lat, LM.saved.lon], LM.saved.zoom, { animate: false }); else map.fitBounds(IN_BOUNDS, { padding: [8, 8] });
  LM.g = { hex: L.layerGroup().addTo(map), fac: L.layerGroup().addTo(map), raw: L.layerGroup().addTo(map), mk: L.layerGroup().addTo(map), area: L.layerGroup().addTo(map) };
  LM.warned = false; LM.err = 0; LM.ok = 0;
  setBase(LM.base === 'plain' ? 'plain' : 'street', true);
  map.on('moveend', () => { const c = map.getCenter(); LM.saved = { lat: c.lat, lon: c.lng, zoom: map.getZoom() }; });
  bindDraw(map); windSync(); renderMap();
}
function disposeMap() {
  if (!LM.map) return; const c = LM.map.getCenter(); LM.saved = { lat: c.lat, lon: c.lng, zoom: LM.map.getZoom() };
  try { LM.map.remove(); } catch (e) {} LM.map = null; LM.g = {}; LM.tile = null; LM.plain = null; LM.wind = null;
}

function markerIcon(g) {
  const p = g.p, e = g.top, r = p.kind === 'site' ? 6.5 : 4.8, c = ccol(e.cls), sel = S.sel === p.id, Z = 48, m = Z / 2;
  let s = `<svg width="${Z}" height="${Z}" viewBox="0 0 ${Z} ${Z}" aria-hidden="true" style="overflow:visible"><g transform="translate(${m} ${m})">`;
  if (e.status === 'abnormal') s += `<circle class="pulse" r="${r * 1.7}" fill="none" stroke="var(--danger)" stroke-width="2"/><circle r="${r * 1.7}" fill="none" stroke="var(--danger)" stroke-width="2"/>`;
  if (e.status === 'baseline_building') s += `<circle r="${r * 1.75}" fill="none" stroke="var(--tx2)" stroke-width="1.2" stroke-dasharray="3 3"/>`;
  s += `<path d="${glyphPath(e.cls, r)}" style="${e.cls === 'unknown' ? `fill:${alpha(c, .18)};stroke:${c};stroke-width:2px` : `fill:${c};stroke:var(--sea);stroke-width:1.2px`}"/>`;
  if (e.review) s += `<circle cx="${r * .95}" cy="${-r * .95}" r="2.6" fill="var(--warn)" stroke="var(--sea)" stroke-width="1"/>`;
  if (sel) s += `<circle r="${r * 2.5}" fill="none" stroke="var(--tx)" stroke-width="1.6"/>`;
  return L.divIcon({ className: 'mkicon', html: s + '</g></svg>', iconSize: [Z, Z], iconAnchor: [m, m] });
}

function renderMap() {
  const map = LM.map; if (!map) return;
  const evs = winEvents(), gs = groups(evs); CUR = { groups: gs, evs };
  Object.values(LM.g).forEach(g => g.clearLayers());
  const cv_ = cv('--tx2');
  if (S.layers.density) {
    const pts = []; evs.forEach(e => ptsOf(e).forEach(q => pts.push(q)));
    if (pts.length) {
      const size = .5, bins = new Map();
      pts.forEach(([lo, la]) => { const x = hx(lo), y = hy(la), [q, r] = hexRound((Math.sqrt(3) / 3 * x - y / 3) / size, 2 / 3 * y / size), k = q + ',' + r; bins.set(k, (bins.get(k) || 0) + 1); });
      const mx = Math.max(...bins.values());
      bins.forEach((n, k) => { const [q, r] = k.split(',').map(Number), cx = size * Math.sqrt(3) * (q + r / 2), cy = size * 1.5 * r, ring = []; for (let i = 0; i < 6; i++) { const a = Math.PI / 180 * (60 * i - 30); ring.push([hlat(cy + size * Math.sin(a)), hlon(cx + size * Math.cos(a))]); } L.polygon(ring, { interactive: false, stroke: false, fillColor: rampAt(Math.sqrt(n / mx)), fillOpacity: .62 }).addTo(LM.g.hex); });
    }
  }
  if (S.layers.facilities) gs.forEach(g => { const p = g.p; if (p.kind !== 'site' || p.cls === 'unknown') return; const c = ccol(p.cls), a = ((hstr(p.id) % 60) - 30) * Math.PI / 180, hw = .016, hh = .011, ring = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]].map(([x, y]) => [p.lat + x * Math.sin(a) + y * Math.cos(a), p.lon + x * Math.cos(a) - y * Math.sin(a)]); L.polygon(ring, { interactive: false, color: c, weight: 1, fillColor: c, fillOpacity: .18 }).addTo(LM.g.fac); });
  if (S.layers.raw) { const all = []; evs.forEach(e => ptsOf(e).forEach(q => all.push(q))); const step = Math.max(1, Math.ceil(all.length / 2200)); all.forEach((q, i) => { if (i % step) return; L.circleMarker([q[1], q[0]], { radius: 1.7, stroke: false, fillColor: cv_, fillOpacity: .5, interactive: false }).addTo(LM.g.raw); }); }
  if (S.layers.events) gs.forEach(g => {
    const p = g.p, e = g.top, m = L.marker([p.lat, p.lon], { icon: markerIcon(g), keyboard: true, title: p.name + ', ' + CLSMAP[e.cls].label, zIndexOffset: sev(e) * 100 + (S.sel === p.id ? 500 : 0) + (p.kind === 'site' ? 50 : 0) });
    m.bindTooltip(`<b>${esc(p.name)}</b><br>${esc(p.district)}, ${esc(p.state)}<br>${esc(CLSMAP[e.cls].label)}, ${Math.round(e.conf * 100)}% sure<br>${STATUS[e.status].label}${e.review ? ', needs review' : ''}<br>${e.frp.toFixed(0)} MW, ${ago(e.t)}`, { direction: 'top', offset: [0, -12], className: 'lt', opacity: 1 });
    m.on('click', () => select(p.id)); m.addTo(LM.g.mk);
  });
  if (S.area) { const a = S.area; L.rectangle([[a.la0, a.lo0], [a.la1, a.lo1]], { interactive: false, color: cv('--tx'), weight: 1.5, dashArray: '6 4', fillColor: cv('--tx'), fillOpacity: .06 }).addTo(LM.g.area); }
}

/* drawing a box to get an area report */
function bindDraw(map) {
  const c = map.getContainer(); let st = null, rect = null;
  c.addEventListener('pointerdown', e => {
    if (!S.drawing || e.button > 0 || e.target.closest('.ov')) return;
    st = map.mouseEventToLatLng(e); rect = L.rectangle([st, st], { color: cv('--tx'), weight: 1.5, dashArray: '6 4', fillColor: cv('--tx'), fillOpacity: .08, interactive: false }).addTo(map);
    try { c.setPointerCapture(e.pointerId); } catch (x) {} e.preventDefault(); e.stopPropagation();
  }, true);
  c.addEventListener('pointermove', e => { if (st && rect) rect.setBounds(L.latLngBounds(st, map.mouseEventToLatLng(e))); });
  c.addEventListener('pointerup', e => {
    if (!st) return; const b = rect.getBounds(); map.removeLayer(rect); st = null; rect = null;
    if (Math.abs(b.getEast() - b.getWest()) < .05) return;
    S.drawing = false; mapDrawMode(false); S.area = { lo0: b.getWest(), lo1: b.getEast(), la0: b.getSouth(), la1: b.getNorth() }; S.sel = null; S.evId = null; S.expanded = false;
    syncCmd(); renderMap(); renderSide(); syncDrawBtn();
  });
}
function mapDrawMode(on) { if (!LM.map) return; LM.map.dragging[on ? 'disable' : 'enable'](); LM.map.getContainer().style.cursor = on ? 'crosshair' : ''; }

/* view helpers used by the rest of the app */
function flyTo(lon, lat, zoom) {
  if (!LM.map) { LM.saved = { lat, lon, zoom }; return; }
  if (REDUCED) LM.map.setView([lat, lon], zoom, { animate: false }); else LM.map.flyTo([lat, lon], zoom, { duration: .9 });
}
const mapZoom = d => { if (LM.map) LM.map[d > 0 ? 'zoomIn' : 'zoomOut'](1); };
const mapReset = () => { if (LM.map) LM.map.fitBounds(IN_BOUNDS, { padding: [8, 8] }); };
const mapInvalidate = () => { if (LM.map) setTimeout(() => LM.map && LM.map.invalidateSize(), 60); };
function syncDrawBtn() { const b = $('[data-act="draw"]'); if (b) b.setAttribute('aria-pressed', S.drawing); mapDrawMode(S.drawing); const h = $('#drawhint'); if (h) h.style.display = S.drawing ? 'block' : 'none'; }

/* national wind on the command map */
function windSync() {
  const map = LM.map; if (!map) return;
  if (S.layers.wind) {
    if (!LM.wind) LM.wind = new WindLayer({ pane: 'wind', count: 1500, style: S.windStyle });
    if (LM.wind.options.style !== S.windStyle) LM.wind.options.style = S.windStyle; LM.wind.setGrid(nationalGrid(S.windH)); if (!map.hasLayer(LM.wind)) LM.wind.addTo(map);
  } else if (LM.wind && map.hasLayer(LM.wind)) map.removeLayer(LM.wind);
  const ctl = $('#windctl'); if (ctl) { ctl.style.display = S.layers.wind ? '' : 'none'; const lb = $('#windlbl'); if (lb) lb.textContent = S.windH === 0 ? 'Now' : `In ${S.windH} h, ${hourLabel(TODAY, S.windH)}`; const bar = $('#windbar'); if (bar) bar.style.background = windGradientCss(); }
}
