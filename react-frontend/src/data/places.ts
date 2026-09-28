/* =====================================================================
   The sample data set: persistent sites, transient fires, their events,
   and the per-place time series. Port of frontend/js/data/sample.js.
   EVERYTHING HERE IS GENERATED SAMPLE DATA.
   ===================================================================== */
import { R, median, quant } from '../lib/core';
import { DAY, HOUR, TODAY, START, dparts } from '../lib/time';
import type {
  Baseline,
  ClassId,
  Event,
  LonLat,
  Metric,
  NearbySource,
  Place,
  PlaceContext,
  PlaceSeries,
  PlaceStats,
} from '../lib/types';

/* code, name, state, district, lat, lon, class, type, median MW, facility coverage */
const SITES: [string, string, string, string, number, number, ClassId, string, number, Place['cover']][] = [
  ['GJ-01', 'Refinery cluster', 'Gujarat', 'Jamnagar', 22.35, 69.85, 'industrial', 'Refinery', 48, 'good'],
  ['GJ-02', 'Petrochemical complex', 'Gujarat', 'Bharuch', 21.72, 72.62, 'industrial', 'Petrochemical', 35, 'good'],
  ['GJ-03', 'Oil field flares', 'Gujarat', 'Mehsana', 23.55, 72.4, 'gas_flare', 'Oil and gas field', 14, 'good'],
  ['GJ-04', 'Gas gathering station', 'Gujarat', 'Anand', 22.35, 72.75, 'gas_flare', 'Gas processing', 11, 'good'],
  ['GJ-05', 'LNG terminal', 'Gujarat', 'Bharuch', 21.65, 72.52, 'industrial', 'LNG terminal', 22, 'good'],
  ['GJ-06', 'Industrial estate', 'Gujarat', 'Surat', 21.15, 72.72, 'industrial', 'Industrial estate', 18, 'good'],
  ['RJ-01', 'Desert oil field flares', 'Rajasthan', 'Barmer', 25.75, 71.35, 'gas_flare', 'Oil and gas field', 19, 'partial'],
  ['RJ-02', 'Cement works', 'Rajasthan', 'Chittorgarh', 24.87, 74.62, 'industrial', 'Cement plant', 16, 'good'],
  ['RJ-03', 'Solar park', 'Rajasthan', 'Jodhpur', 27.53, 71.92, 'unknown', 'Solar park', 4, 'good'],
  ['PB-01', 'Refinery complex', 'Punjab', 'Bathinda', 30.18, 74.98, 'industrial', 'Refinery', 33, 'good'],
  ['HR-01', 'Refinery complex', 'Haryana', 'Panipat', 29.38, 76.95, 'industrial', 'Refinery', 31, 'good'],
  ['UP-01', 'Refinery complex', 'Uttar Pradesh', 'Mathura', 27.55, 77.7, 'industrial', 'Refinery', 29, 'good'],
  ['UP-02', 'Power and cement belt', 'Uttar Pradesh', 'Sonbhadra', 24.15, 83.03, 'industrial', 'Power and cement', 26, 'partial'],
  ['UP-03', 'Coal power cluster', 'Uttar Pradesh', 'Sonbhadra', 24.2, 82.72, 'industrial', 'Coal power', 37, 'partial'],
  ['DL-01', 'Landfill site', 'Delhi', 'East Delhi', 28.63, 77.33, 'unknown', 'Landfill', 7, 'good'],
  ['MH-01', 'LNG terminal', 'Maharashtra', 'Ratnagiri', 17.0, 73.25, 'industrial', 'LNG terminal', 20, 'good'],
  ['MH-02', 'Onshore gas terminal', 'Maharashtra', 'Raigad', 18.9, 72.93, 'gas_flare', 'Gas terminal', 17, 'good'],
  ['MH-03', 'Coal power station', 'Maharashtra', 'Chandrapur', 19.95, 79.3, 'industrial', 'Coal power', 34, 'good'],
  ['MH-04', 'Landfill site', 'Maharashtra', 'Mumbai', 19.06, 72.92, 'unknown', 'Landfill', 6, 'good'],
  ['KA-01', 'Coastal refinery', 'Karnataka', 'Dakshina Kannada', 12.95, 74.83, 'industrial', 'Refinery', 27, 'good'],
  ['KA-02', 'Integrated steel plant', 'Karnataka', 'Ballari', 15.15, 76.65, 'industrial', 'Steel plant', 52, 'good'],
  ['KA-03', 'Landfill site', 'Karnataka', 'Bengaluru Rural', 12.92, 77.47, 'unknown', 'Landfill', 5, 'good'],
  ['KL-01', 'Refinery complex', 'Kerala', 'Ernakulam', 9.97, 76.36, 'industrial', 'Refinery', 25, 'good'],
  ['TN-01', 'Refinery cluster', 'Tamil Nadu', 'Chennai', 13.17, 80.26, 'industrial', 'Refinery', 28, 'good'],
  ['TN-02', 'Lignite mine and power', 'Tamil Nadu', 'Cuddalore', 11.6, 79.48, 'mining', 'Lignite mine', 21, 'good'],
  ['TN-03', 'Steel works', 'Tamil Nadu', 'Salem', 11.7, 78.1, 'industrial', 'Steel plant', 30, 'good'],
  ['TN-04', 'Cement works', 'Tamil Nadu', 'Ariyalur', 11.15, 79.08, 'industrial', 'Cement plant', 15, 'good'],
  ['AP-01', 'Gas field flares', 'Andhra Pradesh', 'East Godavari', 16.85, 82.12, 'gas_flare', 'Oil and gas field', 16, 'partial'],
  ['AP-02', 'Gas processing plant', 'Andhra Pradesh', 'East Godavari', 17.02, 81.8, 'gas_flare', 'Gas processing', 13, 'partial'],
  ['AP-03', 'Refinery and steel', 'Andhra Pradesh', 'Visakhapatnam', 17.7, 83.22, 'industrial', 'Refinery and steel', 44, 'good'],
  ['TS-01', 'Coalfield', 'Telangana', 'Peddapalli', 18.75, 79.5, 'mining', 'Coal mine', 19, 'partial'],
  ['TS-02', 'Cement belt', 'Telangana', 'Nalgonda', 16.9, 79.6, 'industrial', 'Cement plant', 15, 'partial'],
  ['OD-01', 'Refinery and port', 'Odisha', 'Jagatsinghpur', 20.3, 86.62, 'industrial', 'Refinery', 36, 'good'],
  ['OD-02', 'Coalfield', 'Odisha', 'Angul', 20.95, 85.22, 'mining', 'Coal mine', 24, 'partial'],
  ['OD-03', 'Coalfield', 'Odisha', 'Jharsuguda', 21.8, 83.95, 'mining', 'Coal mine', 22, 'partial'],
  ['OD-04', 'Steel works', 'Odisha', 'Sundargarh', 22.25, 84.85, 'industrial', 'Steel plant', 48, 'good'],
  ['OD-05', 'Iron ore mines', 'Odisha', 'Keonjhar', 21.9, 85.6, 'mining', 'Iron ore mine', 9, 'partial'],
  ['OD-06', 'Power and aluminium complex', 'Odisha', 'Angul', 20.85, 85.1, 'industrial', 'Power and aluminium', 31, 'partial'],
  ['JH-01', 'Coalfield with surface fires', 'Jharkhand', 'Dhanbad', 23.75, 86.42, 'mining', 'Coal mine fires', 26, 'partial'],
  ['JH-02', 'Steel works', 'Jharkhand', 'Purbi Singhbhum', 22.8, 86.2, 'industrial', 'Steel plant', 50, 'good'],
  ['JH-03', 'Steel works', 'Jharkhand', 'Bokaro', 23.67, 86.15, 'industrial', 'Steel plant', 42, 'good'],
  ['JH-04', 'Coalfield', 'Jharkhand', 'Ramgarh', 23.65, 85.4, 'mining', 'Coal mine', 18, 'partial'],
  ['WB-01', 'Coalfield', 'West Bengal', 'Paschim Bardhaman', 23.62, 87.13, 'mining', 'Coal mine', 20, 'partial'],
  ['WB-02', 'Petrochemical port', 'West Bengal', 'Purba Medinipur', 22.03, 88.1, 'industrial', 'Petrochemical', 33, 'good'],
  ['CG-01', 'Coal power cluster', 'Chhattisgarh', 'Korba', 22.35, 82.7, 'industrial', 'Coal power', 39, 'partial'],
  ['CG-02', 'Coalfield', 'Chhattisgarh', 'Korba', 22.42, 82.62, 'mining', 'Coal mine', 23, 'partial'],
  ['CG-03', 'Integrated steel plant', 'Chhattisgarh', 'Durg', 21.2, 81.38, 'industrial', 'Steel plant', 46, 'good'],
  ['CG-04', 'Iron ore mines', 'Chhattisgarh', 'Dantewada', 18.7, 81.2, 'mining', 'Iron ore mine', 8, 'weak'],
  ['AS-01', 'Oil field flares', 'Assam', 'Tinsukia', 27.45, 95.35, 'gas_flare', 'Oil and gas field', 18, 'weak'],
  ['AS-02', 'Oil field flares', 'Assam', 'Sivasagar', 26.98, 94.62, 'gas_flare', 'Oil and gas field', 15, 'weak'],
  ['AS-03', 'Refinery', 'Assam', 'Golaghat', 26.5, 93.96, 'industrial', 'Refinery', 24, 'weak'],
  ['MP-01', 'Coal mines', 'Madhya Pradesh', 'Singrauli', 24.1, 82.65, 'mining', 'Coal mine', 25, 'partial'],
  ['MP-02', 'Cement works', 'Madhya Pradesh', 'Satna', 24.6, 80.83, 'industrial', 'Cement plant', 14, 'good'],
];

const NEW_SITES = new Set(['MP-02', 'AS-03', 'OD-06', 'TS-02', 'KL-01', 'GJ-06', 'CG-04', 'UP-02', 'JH-04']);
const FORCE_ABN: Record<string, number> = { 'JH-01': 2, 'GJ-03': 3, 'TN-01': 1, 'OD-04': 5, 'WB-02': 4, 'MH-02': 6 };

/** district, state, lat, lon - the zones transient fires are placed in */
const WILD_ZONES: [string, string, number, number][] = [
  ['Balaghat', 'Madhya Pradesh', 21.8, 80.2], ['Mandla', 'Madhya Pradesh', 22.6, 80.4],
  ['Bastar', 'Chhattisgarh', 19.1, 81.9], ['Mayurbhanj', 'Odisha', 21.8, 86.3],
  ['Kandhamal', 'Odisha', 20.4, 84.1], ['Almora', 'Uttarakhand', 29.6, 79.7],
  ['Pauri Garhwal', 'Uttarakhand', 30.1, 78.8], ['Kangra', 'Himachal Pradesh', 32.1, 76.3],
  ['Nagarkurnool', 'Telangana', 16.4, 78.6], ['Chamarajanagar', 'Karnataka', 11.9, 76.9],
  ['Gadchiroli', 'Maharashtra', 20.2, 80.0], ['Amravati', 'Maharashtra', 21.4, 77.2],
  ['West Kameng', 'Arunachal Pradesh', 27.3, 92.4], ['Dima Hasao', 'Assam', 25.5, 93.0],
  ['Kalahandi', 'Odisha', 19.9, 83.2], ['Idukki', 'Kerala', 9.8, 77.1],
];

const AGRI_ZONES: [string, string, number, number][] = [
  ['Sangrur', 'Punjab', 30.25, 75.85], ['Ludhiana', 'Punjab', 30.9, 75.85],
  ['Patiala', 'Punjab', 30.34, 76.4], ['Amritsar', 'Punjab', 31.6, 74.9],
  ['Karnal', 'Haryana', 29.7, 76.98], ['Kurukshetra', 'Haryana', 29.97, 76.85],
  ['Fatehabad', 'Haryana', 29.5, 75.45], ['Meerut', 'Uttar Pradesh', 29.0, 77.7],
  ['Muzaffarnagar', 'Uttar Pradesh', 29.47, 77.7], ['Shahjahanpur', 'Uttar Pradesh', 27.88, 79.9],
  ['Sri Ganganagar', 'Rajasthan', 29.9, 73.9],
];

/** how likely each month of the year is to see each kind of transient fire */
const WILD_W = [0.6, 1, 2.2, 3, 2.5, 0.7, 0.1, 0.05, 0.1, 0.4, 0.6, 0.7];
const AGRI_W = [0.1, 0.1, 0.3, 1.4, 1, 0.1, 0, 0, 0.2, 3, 3.5, 0.4];

/* ---- build places and events ---- */
export const PLACES: Place[] = [];
export const EVENTS: Event[] = [];
export const PMAP: Record<string, Place> = {};
export const EVP: Record<string, Event[]> = {};

const CONF: Record<ClassId, [number, number]> = {
  wildfire: [0.7, 0.95],
  agricultural_burning: [0.72, 0.94],
  gas_flare: [0.78, 0.96],
  mining: [0.65, 0.9],
  industrial: [0.7, 0.93],
  unknown: [0.42, 0.62],
};

const ALT: Record<ClassId, ClassId[]> = {
  gas_flare: ['industrial'],
  industrial: ['gas_flare', 'mining'],
  mining: ['industrial'],
  wildfire: ['agricultural_burning'],
  agricultural_burning: ['wildfire'],
  unknown: ['industrial', 'wildfire', 'gas_flare'],
};

const SENS = ['VIIRS NOAA-20', 'VIIRS NOAA-21', 'VIIRS Suomi-NPP', 'MODIS Aqua', 'MODIS Terra'];
const SW = [0.28, 0.26, 0.24, 0.12, 0.1];
const HRS = [1.5, 10.5, 13.5, 22.5];

function pickSensor(r: () => number): string {
  const x = r();
  let a = 0;
  for (let i = 0; i < SENS.length; i++) {
    a += SW[i];
    if (x < a) return SENS[i];
  }
  return SENS[0];
}

function mkEvent(p: Place, t: number, r: ReturnType<typeof R>, status: Event['status'], z0: number | null): Event {
  const lowc = r() < 0.08 && p.cls !== 'unknown';
  const conf = lowc ? r.range(0.48, 0.6) : r.range(...CONF[p.cls]);
  const alt = r.pick(ALT[p.cls]);
  const p2 = (1 - conf) * r.range(0.55, 0.85);
  let frp: number;
  let z: number | null = null;
  if (p.kind === 'site') {
    const sd = 1.4826 * p.mad;
    const zz = z0 != null ? z0 : Math.max(-2.2, Math.min(2.2, r.norm()));
    frp = Math.exp(Math.log(p.med) + zz * sd) * 1.1;
    if (status !== 'baseline_building') z = (Math.log(frp) - Math.log(p.med)) / sd;
  } else {
    frp = Math.exp(Math.log(p.cls === 'wildfire' ? 14 : 9) + r.norm() * (p.cls === 'wildfire' ? 0.7 : 0.5));
  }
  const nDet = p.kind === 'site' ? Math.max(4, Math.round(r.range(8, 30) + p.med * 0.6)) : p.cls === 'wildfire' ? r.int(4, 140) : r.int(2, 18);
  const ev: Event = {
    id: 'E' + String(en++).padStart(4, '0'),
    pid: p.id,
    t,
    cls: p.cls,
    conf,
    cls2: alt,
    p2,
    status,
    z,
    frp,
    sensor: pickSensor(r),
    nDet,
    area: +(nDet * 0.14 * r.range(0.8, 1.3)).toFixed(2),
    review: conf < 0.6,
    hr: (r.pick(HRS) + r.range(-1.4, 1.4) + 24) % 24,
  };
  EVENTS.push(ev);
  (EVP[p.id] = EVP[p.id] || []).push(ev);
  return ev;
}

let pn = 1;
let en = 1;

(function build() {
  SITES.forEach((s) => {
    const [code, name, state, district, lat, lon, cls, type, med, cover] = s;
    const r = R('site:' + code);
    const isNew = NEW_SITES.has(code);
    const p: Place = {
      id: 'S' + String(pn++).padStart(3, '0'),
      kind: 'site',
      code,
      name: `${name}, ${district}`,
      state,
      district,
      lat,
      lon,
      cls,
      type,
      med,
      mad: r.range(0.14, 0.32),
      cover,
      firstSeen: isNew ? TODAY - r.int(20, 55) * DAY : TODAY - r.int(400, 1500) * DAY,
    };
    PLACES.push(p);
    PMAP[p.id] = p;
    const from = Math.max(p.firstSeen, START);
    const n = Math.max(2, Math.round(r.int(6, 13) * Math.min(1, (TODAY - from) / (320 * DAY))));
    for (let i = 0; i < n; i++) {
      const t = from + r() * (TODAY - from);
      const tt = Math.min(TODAY - 2 * HOUR, Math.round(t / DAY) * DAY + Math.round(9 * HOUR));
      const young = tt - p.firstSeen < 20 * DAY;
      const abn = !young && r() < 0.05;
      mkEvent(p, tt, r, young ? 'baseline_building' : abn ? 'abnormal' : 'routine', abn ? r.range(3.9, 6.2) : null);
    }
    if (FORCE_ABN[code] && !isNew) mkEvent(p, TODAY - FORCE_ABN[code] * DAY + 2 * HOUR, r, 'abnormal', r.range(4, 6.4));
  });

  function transient(cls: ClassId, zones: [string, string, number, number][], W: number[], count: number) {
    const r = R('trans:' + cls);
    let made = 0;
    while (made < count) {
      const d = r.int(0, 364);
      const t = START + d * DAY;
      const m = dparts(t).m;
      if (r() * Math.max(...W) > W[m]) continue;
      const zn = r.pick(zones);
      const lat = zn[2] + r.range(-0.18, 0.18);
      const lon = zn[3] + r.range(-0.18, 0.18);
      const p: Place = {
        id: 'T' + String(pn++).padStart(4, '0'),
        kind: 'transient',
        code: 'T',
        name: (cls === 'wildfire' ? 'Forest fire near ' : 'Crop residue burning near ') + zn[0],
        state: zn[1],
        district: zn[0],
        lat,
        lon,
        cls,
        type: cls === 'wildfire' ? 'Vegetation fire' : 'Crop residue fire',
        firstSeen: t,
        cover: 'partial',
        mad: 0.3,
        med: 10,
      };
      PLACES.push(p);
      PMAP[p.id] = p;
      mkEvent(p, t + Math.round(r.range(1, 14)) * HOUR, r, 'not_applicable', null);
      made++;
    }
  }

  transient('wildfire', WILD_ZONES, WILD_W, 130);
  transient('agricultural_burning', AGRI_ZONES, AGRI_W, 150);

  EVENTS.sort((a, b) => a.t - b.t);
  Object.values(EVP).forEach((a) => a.sort((x, y) => x.t - y.t));
})();

/* ---- raw detection points around an event (deterministic) ---- */
export function ptsOf(ev: Event): LonLat[] {
  if (ev._pts) return ev._pts;
  const p = PMAP[ev.pid];
  const r = R('pts:' + ev.id);
  const n = Math.min(ev.nDet, 36);
  const sg = p.kind === 'site' ? 0.012 + Math.min(0.05, ev.area * 0.004) : 0.02 + Math.min(0.12, Math.sqrt(ev.area) * 0.03);
  ev._pts = Array.from({ length: n }, () => [p.lon + r.norm() * sg, p.lat + r.norm() * sg * 0.9]);
  return ev._pts;
}

/* ---- time series for a place ---- */
export function baselineOf(obs: { frp: number }[]): Baseline {
  const ln = obs.map((o) => Math.log(o.frp));
  const m = median(ln);
  const mad = median(ln.map((x) => Math.abs(x - m))) || 0.1;
  const sd = 1.4826 * mad;
  return {
    n: obs.length,
    mLn: m,
    med: Math.exp(m),
    mad,
    sd,
    p95: quant(
      obs.map((o) => o.frp),
      0.95,
    ),
    lo: Math.exp(m - 3.5 * sd),
    hi: Math.exp(m + 3.5 * sd),
  };
}

export function seriesOf(p: Place): PlaceSeries {
  if (p._ser) return p._ser;
  const r = R('ser:' + p.id);
  const evs = EVP[p.id] || [];
  const obs: PlaceSeries['obs'] = [];
  const gaps: [number, number][] = [];
  const sd = 1.4826 * p.mad;

  if (p.kind === 'transient') {
    const ev = evs[0];
    const nd = r.int(1, 4);
    for (let i = 0; i < nd; i++) {
      obs.push({
        t: ev.t + i * DAY * r.range(0.5, 1.3),
        frp: Math.max(1, ev.frp * Math.exp(r.norm() * 0.35) * (i === 0 ? 1 : 0.8)),
        cnt: r.int(1, 5),
        sensor: r() < 0.7 ? ev.sensor : 'VIIRS Suomi-NPP',
        z: null,
      });
    }
    obs.sort((a, b) => a.t - b.t);
    return (p._ser = { obs, gaps: [] });
  }

  const off = { until: -1 };
  let cloudRun = 0;
  let cloudStart = 0;
  const abn = evs.filter((e) => e.status === 'abnormal');
  for (let d = 0; d <= 364; d++) {
    const t = START + d * DAY + 9 * HOUR;
    if (t < p.firstSeen) continue;
    const m = dparts(t).m;
    const pobs = m >= 5 && m <= 8 ? 0.38 : m === 9 || m === 10 ? 0.7 : 0.82;
    if (d > off.until && r() < 0.012) off.until = d + r.int(5, 12);
    const seen = r() < pobs;
    if (!seen) {
      if (!cloudRun) cloudStart = t;
      cloudRun++;
      continue;
    }
    if (cloudRun >= 5) gaps.push([cloudStart, t - DAY]);
    cloudRun = 0;
    if (d <= off.until) continue;
    let ln = Math.log(p.med) + r.norm() * sd * 0.75;
    abn.forEach((e) => {
      const dd = Math.abs(t - e.t) / DAY;
      if (dd < 3.2) ln = Math.log(p.med) + (e.z! * (1 - dd * 0.22)) * sd + r.norm() * sd * 0.2;
    });
    obs.push({
      t,
      frp: Math.exp(ln),
      cnt: r.int(1, 3),
      sensor:
        r() < 0.78
          ? r.pick(['VIIRS NOAA-20', 'VIIRS NOAA-21', 'VIIRS Suomi-NPP'])
          : r.pick(['MODIS Aqua', 'MODIS Terra']),
      z: null,
    });
  }
  const base = baselineOf(obs);
  obs.forEach((o) => {
    o.z = base.n >= 20 ? (Math.log(o.frp) - base.mLn) / base.sd : null;
  });
  return (p._ser = { obs, gaps, base });
}

export function statsOf(p: Place): PlaceStats {
  const obs = seriesOf(p).obs;
  const days = (n: number) => new Set(obs.filter((o) => o.t > TODAY - n * DAY).map((o) => Math.floor(o.t / DAY))).size;
  const set = new Set(obs.map((o) => Math.floor(o.t / DAY)));
  let streak = 0;
  for (let d = Math.floor(TODAY / DAY); set.has(d); d--) streak++;
  const last = obs.length ? obs[obs.length - 1].t : null;
  return {
    d30: days(30),
    d90: days(90),
    months: new Set(obs.map((o) => dparts(o.t).m)).size,
    streak,
    last,
    sinceLast: last ? Math.max(0, Math.round((TODAY - last) / DAY)) : null,
  };
}

/** how many observations land in each local hour of the day, 0 to 23 */
export function hoursOf(p: Place): number[] {
  const r = R('hr:' + p.id);
  const night: Record<ClassId, number> = {
    gas_flare: 0.8,
    industrial: 0.6,
    mining: 0.55,
    wildfire: 0.25,
    agricultural_burning: 0.1,
    unknown: 0.4,
  };
  const H = new Array<number>(24).fill(0);
  const n = 90 + r.int(0, 40);
  for (let i = 0; i < n; i++) {
    const nt = r() < night[p.cls];
    H[r.pick(nt ? [1, 1, 2, 22, 23] : [10, 11, 13, 13, 14])]++;
  }
  return H;
}

/* ---- typical value ranges per class, used for the sample context ---- */
const PROF: Record<string, Record<string, [number, number]>> = {
  base: {
    frp: [6, 40], bt: [335, 360], area: [0.3, 2], persist: [40, 90], night: [45, 75], dOil: [15, 80],
    dInd: [0.2, 8], dMine: [10, 80], dLand: [10, 70], crop: [2, 30], forest: [0, 20], bare: [5, 35],
    indus: [5, 40], ndvi: [0.15, 0.5], ndbi: [0.05, 0.3], vpd: [1.2, 3.8], rain: [0, 4],
  },
  gas_flare: {
    frp: [8, 30], bt: [352, 367], persist: [74, 90], night: [70, 95], dOil: [0.05, 1.2], dInd: [0.05, 1.5],
    indus: [25, 60], bare: [15, 45], forest: [0, 8], crop: [0, 15],
  },
  industrial: {
    frp: [12, 60], bt: [345, 364], area: [0.4, 2], persist: [60, 90], night: [50, 75], dInd: [0.05, 1.2],
    dOil: [3, 40], indus: [30, 70], ndbi: [0.2, 0.4], forest: [0, 10],
  },
  mining: {
    frp: [6, 40], dMine: [0, 1], bare: [45, 75], persist: [35, 90], night: [55, 70], dInd: [0.3, 6],
    ndbi: [0.05, 0.2], forest: [0, 12],
  },
  wildfire: {
    frp: [6, 70], bt: [325, 355], area: [0.5, 9], persist: [0, 5], night: [15, 35], dInd: [8, 45],
    forest: [55, 90], crop: [0, 15], ndvi: [0.5, 0.78], vpd: [2.5, 5], rain: [0, 3], bare: [0, 8], indus: [0, 2],
  },
  agricultural_burning: {
    frp: [4, 22], bt: [322, 342], area: [0.15, 1.2], persist: [0, 4], night: [5, 20], dInd: [6, 40],
    crop: [70, 95], ndvi: [0.15, 0.4], vpd: [1.8, 3.5], rain: [0, 3], bare: [2, 15], indus: [0, 3], forest: [0, 8],
  },
  unknown: {
    frp: [2, 9], bt: [320, 338], area: [0.1, 0.8], persist: [10, 60], night: [20, 60], dInd: [0.3, 5],
    dLand: [0.1, 3], ndbi: [0.1, 0.35], indus: [5, 25], forest: [0, 10],
  },
};

function profOf(cls: string, key: string, r: ReturnType<typeof R>): number {
  const a = (PROF[cls] && PROF[cls][key]) || PROF.base[key];
  if (!a) return 0;
  return r.range(a[0], a[1]);
}

/** land cover, distances, imagery and weather values around a place (sample) */
export function contextOf(p: Place): PlaceContext {
  if (p._ctx) return p._ctx;
  const r = R('ctx:' + p.id);
  const c = p.cls;
  const crop = profOf(c, 'crop', r);
  const forest = profOf(c, 'forest', r);
  const bare = profOf(c, 'bare', r);
  const ind = profOf(c, 'indus', r);
  const grass = r.range(2, 12);
  const built = ind;
  const sum = crop + forest + bare + built + grass;
  const water = Math.max(0, 100 - sum);
  const parts: [string, number][] = [
    ['Cropland', crop],
    ['Forest', forest],
    ['Bare ground', bare],
    ['Industrial or built-up', built],
    ['Grass and scrub', grass],
    ['Water and other', water],
  ];
  const tot = parts.reduce((a, x) => a + x[1], 0);
  parts.forEach((x) => (x[1] = +((x[1] / tot) * 100).toFixed(1)));

  const d: PlaceContext['d'] = {
    refinery: p.type.includes('Refinery') || p.type.includes('Petro') ? r.range(0, 0.8) : r.range(40, 400),
    power: /power|Power/.test(p.type) ? r.range(0, 0.9) : r.range(15, 200),
    steel: /Steel|Cement/.test(p.type) ? r.range(0, 0.9) : r.range(15, 250),
    mine: c === 'mining' ? r.range(0, 0.7) : r.range(10, 150),
    landfill: p.type === 'Landfill' ? r.range(0, 0.3) : r.range(8, 90),
  };
  if (p.kind === 'transient') Object.keys(d).forEach((k) => (d[k as keyof typeof d] = r.range(6, 90)));

  return (p._ctx = {
    parts,
    d,
    fac: c === 'unknown' && p.type !== 'Landfill' ? null : true,
    ndvi: profOf(c, 'ndvi', r),
    nbr: r.range(0.05, 0.55),
    ndbi: profOf(c, 'ndbi', r),
    built: ind / 100,
    valid: r.range(0.25, 0.96),
    sceneDays: r.int(2, 11),
    vpd: profOf(c, 'vpd', r),
    rain: profOf(c, 'rain', r),
    elev: r.int(5, 900),
    slope: r.range(0, 9),
  });
}

/** other micro-sources near a place (sample) */
export function nearbyOf(p: Place): NearbySource[] {
  const r = R('nb:' + p.id);
  const n = p.kind === 'site' ? r.int(3, 6) : r.int(0, 2);
  const out: NearbySource[] = [];
  const nm: Record<ClassId, string[]> = {
    industrial: ['Furnace stack', 'Boiler stack', 'Storage yard', 'Kiln'],
    gas_flare: ['Flare stack B', 'Flare stack C', 'Compressor vent'],
    mining: ['Waste heap', 'Coal stockpile', 'Surface fire patch'],
    unknown: ['Heap', 'Reflective roof'],
    wildfire: ['Fire front'],
    agricultural_burning: ['Burning field'],
  };
  const names = nm[p.cls] || ['Source'];
  for (let i = 0; i < n; i++) {
    const c = r() < 0.75 ? p.cls : r.pick(['industrial', 'gas_flare', 'unknown'] as const);
    out.push({
      name: r.pick(names) + ' ' + String.fromCharCode(65 + i),
      cls: c,
      km: +r.range(0.4, 9.5).toFixed(1),
      ang: r.range(0, 6.283),
      st: r() < 0.12 && p.kind === 'site' ? 'abnormal' : 'routine',
    });
  }
  return out;
}

/* ---- sample model-health numbers ---- */
export const METRICS: Metric[] = (
  [
    ['wildfire', 0.81, 0.84, 412],
    ['agricultural_burning', 0.86, 0.83, 530],
    ['gas_flare', 0.79, 0.74, 188],
    ['mining', 0.68, 0.61, 142],
    ['industrial', 0.76, 0.82, 615],
    ['unknown', 0.52, 0.49, 160],
  ] as [ClassId, number, number, number][]
).map(([id, p, r, n]) => ({ id, p, r, n, f1: (2 * p * r) / (p + r) }));
