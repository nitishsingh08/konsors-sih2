/* =====================================================================
   Shared Leaflet pieces: the tile source, the offline plain base map,
   hexagon binning for the density layer, and the map view that survives
   a remount. Port of the map helpers in frontend/js/map/leaflet-map.js.
   ===================================================================== */
import L from 'leaflet';
import { clamp, ccol, cv, hexMix, hstr } from '../lib/core';
import { COUNTRY, STATES } from '../data/geo';
import type { ClassId, LonLat } from '../lib/types';

export const OSM_URL = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';
export const OSM_ATTR = '© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors';
export const IN_BOUNDS: L.LatLngBoundsExpression = [
  [6, 68],
  [37.5, 98],
];

export type BaseMode = 'street' | 'plain';

/** [lon, lat] pairs to Leaflet [lat, lng] pairs */
export const ll = (ring: LonLat[]): [number, number][] => ring.map((p) => [p[1], p[0]]);

/** plain, offline base: country outline and state lines */
export function plainBase(map?: L.Map): L.LayerGroup {
  const g = L.layerGroup();
  const rd = L.svg({ pane: 'base', padding: 0.5 });
  const land = cv('--land');
  const st = cv('--state');
  const co = cv('--coast');
  COUNTRY.forEach((r) =>
    L.polygon(ll(r), { pane: 'base', renderer: rd, interactive: false, stroke: false, fillColor: land, fillOpacity: 1 }).addTo(g),
  );
  STATES.forEach((s) =>
    s.r.forEach((r) => L.polyline(ll(r), { pane: 'base', renderer: rd, interactive: false, color: st, weight: 0.8 }).addTo(g)),
  );
  COUNTRY.forEach((r) =>
    L.polyline(ll(r), { pane: 'base', renderer: rd, interactive: false, color: co, weight: 1.3, lineJoin: 'round' }).addTo(g),
  );
  if (map) g.addTo(map);
  return g;
}

export function makePanes(map: L.Map): void {
  map.createPane('base').style.zIndex = '250';
  map.createPane('wind').style.zIndex = '350';
}

export interface TileProbe {
  /** called once when the tile server clearly cannot be reached */
  onFail: () => void;
  layer: L.TileLayer;
}

/** OSM tiles, with the offline fallback trigger from the original build */
export function addTiles(map: L.Map, onFail: () => void): TileProbe {
  const t = L.tileLayer(OSM_URL, { maxZoom: 19, attribution: OSM_ATTR, crossOrigin: true });
  let ok = 0;
  let err = 0;
  let fell = false;
  t.on('tileload', () => ok++);
  t.on('tileerror', () => {
    err++;
    if (err >= 6 && !ok && !fell) {
      fell = true;
      map.removeLayer(t);
      onFail();
    }
  });
  t.addTo(map);
  return { onFail, layer: t };
}

/* ---- hexagon binning (an equal-area-ish grid over India) ---- */
export const hx = (lon: number): number => (lon - 67.5) * 0.927;
export const hy = (lat: number): number => 38.5 - lat;
export const hlon = (x: number): number => x / 0.927 + 67.5;
export const hlat = (y: number): number => 38.5 - y;

export function hexRound(q: number, r: number): [number, number] {
  const x = q;
  const z = r;
  const y = -x - z;
  let rx = Math.round(x);
  let ry = Math.round(y);
  let rz = Math.round(z);
  const dx = Math.abs(rx - x);
  const dy = Math.abs(ry - y);
  const dz = Math.abs(rz - z);
  if (dx > dy && dx > dz) rx = -ry - rz;
  else if (dy > dz) ry = -rx - rz;
  else rz = -rx - ry;
  return [rx, rz];
}

const RAMPS: Record<'dark' | 'light', string[]> = {
  dark: ['#1a3550', '#22607f', '#3f95aa', '#90d3d8', '#eaf8f6'],
  light: ['#dbe7f3', '#a3c6e2', '#5f9dca', '#2f71a8', '#143f6e'],
};

export const rampAt = (t: number, theme: 'dark' | 'light'): string => {
  const r = RAMPS[theme];
  const f = clamp(t, 0, 1) * (r.length - 1);
  const i = Math.min(r.length - 2, Math.floor(f));
  return hexMix(r[i], r[i + 1], f - i);
};

/** the little rotated rectangle that stands for a facility footprint */
export function facilityRing(lat: number, lon: number, id: string): [number, number][] {
  const a = ((hstr(id) % 60) - 30) * (Math.PI / 180);
  const hw = 0.016;
  const hh = 0.011;
  return ([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]] as [number, number][]).map(
    ([x, y]) => [lat + x * Math.sin(a) + y * Math.cos(a), lon + x * Math.cos(a) - y * Math.sin(a)],
  );
}

/** a square marker icon: the class glyph, plus rings for status */
export function markerIcon(opts: {
  cls: ClassId;
  status: 'abnormal' | 'baseline_building' | 'routine' | 'not_applicable';
  review: boolean;
  selected: boolean;
  transient: boolean;
}): L.DivIcon {
  const r = opts.transient ? 4.8 : 6.5;
  const c = ccol(opts.cls);
  const Z = 48;
  const m = Z / 2;
  let s = `<g transform="translate(${m} ${m})">`;
  if (opts.status === 'abnormal') {
    s += `<circle class="pulse" r="${r * 1.7}" fill="none" stroke="var(--danger)" stroke-width="2"/><circle r="${r * 1.7}" fill="none" stroke="var(--danger)" stroke-width="2"/>`;
  }
  if (opts.status === 'baseline_building') {
    s += `<circle r="${r * 1.75}" fill="none" stroke="var(--tx2)" stroke-width="1.2" stroke-dasharray="3 3"/>`;
  }
  s += `<path d="${glyphD(opts.cls, r)}" style="${
    opts.cls === 'unknown'
      ? `fill:${alphaCss(c, 0.18)};stroke:${c};stroke-width:2px`
      : `fill:${c};stroke:var(--sea);stroke-width:1.2px`
  }"/>`;
  if (opts.review) s += `<circle cx="${r * 0.95}" cy="${-r * 0.95}" r="2.6" fill="var(--warn)" stroke="var(--sea)" stroke-width="1"/>`;
  if (opts.selected) s += `<circle r="${r * 2.5}" fill="none" stroke="var(--tx)" stroke-width="1.6"/>`;
  return L.divIcon({
    className: 'mkicon',
    html: `<svg width="${Z}" height="${Z}" viewBox="0 0 ${Z} ${Z}" aria-hidden="true" style="overflow:visible">${s}</g></svg>`,
    iconSize: [Z, Z],
    iconAnchor: [m, m],
  });
}

export function glyphIcon(cls: ClassId, r: number, ring?: boolean): L.DivIcon {
  const c = ccol(cls);
  return L.divIcon({
    className: 'mkicon',
    iconSize: [40, 40],
    iconAnchor: [20, 20],
    html: `<svg width="40" height="40" viewBox="0 0 40 40" style="overflow:visible"><g transform="translate(20 20)">${
      ring ? `<circle r="${r * 1.9}" fill="${alphaCss(cv('--tx'), 0.1)}" stroke="var(--tx)" stroke-width="1.5"/>` : ''
    }<path d="${glyphD(cls, r)}" style="${
      cls === 'unknown' ? `fill:${alphaCss(c, 0.2)};stroke:${c};stroke-width:2px` : `fill:${c};stroke:var(--sea);stroke-width:1.5px`
    }"/></g></svg>`,
  });
}

export const noteIcon = (txt: string, cls?: string): L.DivIcon =>
  L.divIcon({ className: 'notewrap', iconSize: [0, 0], html: `<div class="note ${cls ?? ''}">${txt}</div>` });

export const tickIcon = (txt: string): L.DivIcon =>
  L.divIcon({ className: 'tick', html: `<span>${txt}</span>`, iconSize: [0, 0] });

/* --- small helpers, kept local so this module has no UI imports --- */
function alphaCss(hex: string, a: number): string {
  const h = (hex || '').trim();
  if (h[0] !== '#' || h.length !== 7) return hex;
  const n = parseInt(h.slice(1), 16);
  return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`;
}

function glyphD(cls: ClassId, r: number): string {
  switch (cls) {
    case 'wildfire':
      return `M0 ${-r * 1.1}L${r * 1.05} ${r * 0.8}L${-r * 1.05} ${r * 0.8}Z`;
    case 'agricultural_burning':
      return `M${-r * 0.85} ${-r * 0.85}h${r * 1.7}v${r * 1.7}h${-r * 1.7}Z`;
    case 'mining': {
      let d = '';
      for (let i = 0; i < 6; i++) {
        const a = (Math.PI / 3) * i + Math.PI / 6;
        d += (i ? 'L' : 'M') + (r * 1.05 * Math.cos(a)).toFixed(2) + ' ' + (r * 1.05 * Math.sin(a)).toFixed(2);
      }
      return d + 'Z';
    }
    case 'industrial':
      return `M0 ${-r * 1.2}L${r * 1.2} 0L0 ${r * 1.2}L${-r * 1.2} 0Z`;
    default:
      return `M${-r} 0a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0Z`;
  }
}

/* ---- the last map view, so switching pages and back keeps the framing ---- */
export interface SavedView {
  lat: number;
  lon: number;
  zoom: number;
}

let saved: SavedView | null = null;

export const getSavedView = (): SavedView | null => saved;
export const setSavedView = (v: SavedView | null): void => {
  saved = v;
};
