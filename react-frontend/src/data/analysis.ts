/* =====================================================================
   Derived analysis values that sit between the sample data and the UI:
   exposure counts, similar events, source freshness, the one-line wind
   outlook, and the CAP alert draft. Ported from the tail of
   frontend/js/data/weather.js and frontend/js/pages.js.
   ===================================================================== */
import { R, clamp } from '../lib/core';
import { TODAY, dparts, fmtDT, hourLabel, pad } from '../lib/time';
import { CLSMAP, STATUS } from './classes';
import { EVENTS, PMAP } from './places';
import { areaKm2, compass, plumeOf, windAt, windShift } from './weather';
import type {
  Event,
  LonLat,
  ExposureZone,
  FreshnessRow,
  Plume,
  Place,
  SimilarRow,
  SpreadOutlook,
  Weather,
} from '../lib/types';

/** who and what sits inside the zones (sample counts; real ones come from OSM) */
export function exposureOf(ev: Event, plume: Plume, spread: SpreadOutlook | null): ExposureZone[] {
  const r = R('expo:' + ev.id);
  const dens = { wildfire: 0.22, agricultural_burning: 0.9, industrial: 0.6, gas_flare: 0.35, mining: 0.5, unknown: 1.1 }[ev.cls] || 0.5;
  const zone = (name: string, ring: LonLat[]): ExposureZone => {
    const a = areaKm2(ring);
    return {
      name,
      area: a,
      settlements: Math.round(dens * a * r.range(0.04, 0.09)),
      schools: Math.round(dens * a * r.range(0.008, 0.02)),
      clinics: Math.round(dens * a * r.range(0.002, 0.007)),
      roadKm: Math.round(dens * a * r.range(0.25, 0.6)),
      water: Math.round(a * r.range(0.01, 0.05)),
    };
  };
  const z: ExposureZone[] = [zone('Smoke plume, core, next 6 hours', plume.core)];
  if (spread) z.push(zone('Fire spread outlook, 6 hours', spread.horizons[2].ring));
  return z;
}

export function similarOf(ev: Event): SimilarRow[] {
  const r = R('sim:' + ev.id);
  return EVENTS.filter((e) => e.cls === ev.cls && e.pid !== ev.pid && e.t < ev.t)
    .map((e) => ({ e, d: Math.abs(Math.log(e.frp / ev.frp)) + Math.abs(e.conf - ev.conf) }))
    .sort((a, b) => a.d - b.d)
    .slice(0, 5)
    .map((x) => ({
      e: x.e,
      p: PMAP[x.e.pid],
      sim: clamp(0.96 - x.d * 0.25 - r() * 0.03, 0.6, 0.96),
      outcome: r() < 0.72 ? 'Confirmed by an analyst' : r() < 0.6 ? 'Marked as a false alarm' : 'Not reviewed',
    }));
}

export function freshnessOf(ev: Event): FreshnessRow[] {
  const r = R('fr:' + ev.id);
  return [
    { name: 'Thermal detections (FIRMS)', label: '2 h 41 min old', hours: 2.7, note: 'VIIRS and MODIS, near real time' },
    { name: 'Weather forecast', label: 'issued at event time', hours: 0, note: 'Wind, humidity, temperature, rain' },
    { name: 'Optical imagery (HLS)', label: r.int(2, 8) + ' days old', hours: 0, note: 'Limited by cloud' },
    { name: 'Radar (Sentinel-1)', label: r.int(3, 10) + ' days old', hours: 0, note: 'Passes every few days' },
    { name: 'Air chemistry (TROPOMI)', label: r.int(6, 40) + ' hours old', hours: 0, note: 'One pass a day, then processing' },
    { name: 'Facility records (GEM, OSM)', label: 'updated ' + r.int(10, 60) + ' days ago', hours: 0, note: 'Coverage is uneven in India' },
  ];
}

export function outlookLine(p: Place, _ev: Event, W: Weather): string {
  const w0 = windAt(W, 0);
  const pl = plumeOf(p, W, 6);
  const sh = windShift(W);
  const lvl = w0.spd < 3 ? 'light' : w0.spd < 6 ? 'moderate' : 'strong';
  let s = `Wind is ${lvl} from the ${compass(w0.dir)} at ${Math.round(w0.spd * 3.6)} km/h`;
  if (sh) s += ` and swings round toward the ${compass(sh.to)} around ${hourLabel(W.t0, sh.h)}`;
  return s + `. Smoke could drift about ${pl.reachKm.toFixed(0)} km ${compass(pl.bearing)} within 6 hours.`;
}

/* ---- exports ---- */
export const csvCell = (v: unknown): string => {
  const s = v == null ? '' : String(v);
  return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
};

export const CSV_HEAD = [
  'event_id', 'place', 'district', 'state', 'lat', 'lon', 'time_ist', 'class', 'confidence',
  'second_class', 'status', 'needs_review', 'peak_frp_mw', 'robust_z', 'sensor', 'detections', 'footprint_km2',
];

export function eventsCsv(evs: Event[]): string {
  return [CSV_HEAD.join(',')]
    .concat(
      evs.map((e) => {
        const p = PMAP[e.pid];
        return [
          e.id, p.name, p.district, p.state, p.lat.toFixed(4), p.lon.toFixed(4), fmtDT(e.t), e.cls,
          e.conf.toFixed(3), e.cls2, e.status, e.review, e.frp.toFixed(1), e.z == null ? '' : e.z.toFixed(2),
          e.sensor, e.nDet, e.area,
        ]
          .map(csvCell)
          .join(',');
      }),
    )
    .join('\n');
}

export function eventsGeo(evs: Event[]): string {
  return JSON.stringify(
    {
      type: 'FeatureCollection',
      features: evs.map((e) => {
        const p = PMAP[e.pid];
        return {
          type: 'Feature',
          geometry: { type: 'Point', coordinates: [+p.lon.toFixed(4), +p.lat.toFixed(4)] },
          properties: {
            id: e.id, place: p.name, class: e.cls, confidence: +e.conf.toFixed(3), status: e.status,
            needs_review: e.review, frp_mw: +e.frp.toFixed(1), time: fmtDT(e.t), sample_data: true,
          },
        };
      }),
    },
    null,
    1,
  );
}

const xe = (s: string): string =>
  String(s).replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' })[c] as string,
  );

/** A draft CAP 1.2 alert for an authorized agency to review. It is not sent. */
export function capXml(ev: Event): string {
  const p = PMAP[ev.pid];
  const d = dparts(TODAY);
  const sent = `${d.y}-${pad(d.m + 1)}-${pad(d.d)}T${pad(d.h)}:${pad(d.mi)}:00+05:30`;
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
    <description>${xe(
      'Satellite thermal detection classified as ' + CLSMAP[ev.cls].label.toLowerCase() + ' with ' +
      Math.round(ev.conf * 100) + '% calibrated confidence. Peak fire power ' + ev.frp.toFixed(0) +
      ' MW. Automated classification, to be verified.',
    )}</description>
    <instruction>For review by the authorized alerting authority. Not for public release.</instruction>
    <area>
      <areaDesc>${xe(p.district + ', ' + p.state)}</areaDesc>
      <circle>${p.lat.toFixed(4)},${p.lon.toFixed(4)} 2.0</circle>
    </area>
  </info>
</alert>`;
}

