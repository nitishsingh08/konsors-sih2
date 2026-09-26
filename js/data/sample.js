'use strict';
/* ===== Time helpers (all sample data is anchored to 20 Sep 2026) ===== */
const DAY = 864e5, HOUR = 36e5, IST = 5.5 * HOUR;
const TODAY = Date.UTC(2026, 8, 20, 9, 0, 0), START = TODAY - 364 * DAY;
const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const pad = n => String(n).padStart(2, '0');
const dparts = t => { const d = new Date(t + IST); return { y: d.getUTCFullYear(), m: d.getUTCMonth(), d: d.getUTCDate(), h: d.getUTCHours(), mi: d.getUTCMinutes() }; };
const fmtD = t => { const p = dparts(t); return `${p.d} ${MONTHS[p.m]} ${p.y}`; };
const fmtDT = t => { const p = dparts(t); return `${p.d} ${MONTHS[p.m]} ${p.y}, ${pad(p.h)}:${pad(p.mi)} IST`; };
const fmtS = t => { const p = dparts(t); return `${p.d} ${MONTHS[p.m]}`; };
const iso = t => { const p = dparts(t); return `${p.y}-${pad(p.m + 1)}-${pad(p.d)}`; };
const ago = t => { const h = (TODAY - t) / HOUR; if (h < 1) return 'under 1 h ago'; if (h < 24) return Math.round(h) + ' h ago'; return Math.round(h / 24) + ' d ago'; };
const nf = n => Math.round(n).toLocaleString('en-IN');

/* ===== Seeded random ===== */
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function hstr(s) { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }
function R(seed) {
  const r = mulberry32(typeof seed === 'string' ? hstr(seed) : seed);
  r.range = (a, b) => a + (b - a) * r();
  r.int = (a, b) => Math.floor(a + (b - a + 1) * r());
  r.pick = a => a[Math.floor(r() * a.length)];
  r.norm = () => { let u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  return r;
}
const median = a => { if (!a.length) return NaN; const s = [...a].sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const quant = (a, q) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; };

/* ===== Classes and statuses ===== */
const CLS = [
  { id: 'wildfire', label: 'Wildfire', blurb: 'Vegetation fire in forest, grass or scrub. Short-lived, no normal level.' },
  { id: 'agricultural_burning', label: 'Agricultural burning', blurb: 'Crop residue burning on farmland, strongly seasonal.' },
  { id: 'gas_flare', label: 'Gas flare', blurb: 'Burn-off at oil and gas sites. Persistent, so judged against its own history.' },
  { id: 'mining', label: 'Mining heat', blurb: 'Coal seam and mine fires, or heat from mining works.' },
  { id: 'industrial', label: 'Industrial source', blurb: 'Furnaces, stacks and process heat, or a fire inside a facility.' },
  { id: 'unknown', label: 'Unknown', blurb: 'Low confidence, or not a fire at all (reflective surfaces, waste heaps).' }
];
const CLSMAP = Object.fromEntries(CLS.map(c => [c.id, c]));
const STATUS = {
  abnormal: { label: 'Abnormal', k: 'abn', rank: 4, tip: 'Unusual for this site compared with its own history.' },
  routine: { label: 'Routine', k: 'rt', rank: 1, tip: 'Within the normal range for this site.' },
  baseline_building: { label: 'Baseline building', k: 'bb', rank: 2, tip: 'Not enough history yet to judge. No claim is made about normal or abnormal.' },
  not_applicable: { label: 'Transient fire', k: 'na', rank: 0, tip: 'Wildfires and crop burning are short events, so there is no normal level to compare with.' }
};
const FAMILIES = [
  { id: 'thermal', label: 'Heat' }, { id: 'temporal', label: 'Timing' }, { id: 'spread', label: 'Movement and growth' },
  { id: 'terrain', label: 'Terrain' }, { id: 'fuel', label: 'Fuel and land cover' }, { id: 'industry', label: 'Industry nearby' },
  { id: 'weather', label: 'Weather and fire danger' }, { id: 'smoke', label: 'Smoke and burn signs' }, { id: 'sar', label: 'Radar and structure' },
  { id: 'chem', label: 'Air chemistry' }, { id: 'quality', label: 'Data quality' }
];

/* ===== Persistent sites (sample). Names are generic; incidents are made up. ===== */
const SITES = [
  ['GJ-01','Refinery cluster','Gujarat','Jamnagar',22.35,69.85,'industrial','Refinery',48,'good'],
  ['GJ-02','Petrochemical complex','Gujarat','Bharuch',21.72,72.62,'industrial','Petrochemical',35,'good'],
  ['GJ-03','Oil field flares','Gujarat','Mehsana',23.55,72.40,'gas_flare','Oil and gas field',14,'good'],
  ['GJ-04','Gas gathering station','Gujarat','Anand',22.35,72.75,'gas_flare','Gas processing',11,'good'],
  ['GJ-05','LNG terminal','Gujarat','Bharuch',21.65,72.52,'industrial','LNG terminal',22,'good'],
  ['GJ-06','Industrial estate','Gujarat','Surat',21.15,72.72,'industrial','Industrial estate',18,'good'],
  ['RJ-01','Desert oil field flares','Rajasthan','Barmer',25.75,71.35,'gas_flare','Oil and gas field',19,'partial'],
  ['RJ-02','Cement works','Rajasthan','Chittorgarh',24.87,74.62,'industrial','Cement plant',16,'good'],
  ['RJ-03','Solar park','Rajasthan','Jodhpur',27.53,71.92,'unknown','Solar park',4,'good'],
  ['PB-01','Refinery complex','Punjab','Bathinda',30.18,74.98,'industrial','Refinery',33,'good'],
  ['HR-01','Refinery complex','Haryana','Panipat',29.38,76.95,'industrial','Refinery',31,'good'],
  ['UP-01','Refinery complex','Uttar Pradesh','Mathura',27.55,77.70,'industrial','Refinery',29,'good'],
  ['UP-02','Power and cement belt','Uttar Pradesh','Sonbhadra',24.15,83.03,'industrial','Power and cement',26,'partial'],
  ['UP-03','Coal power cluster','Uttar Pradesh','Sonbhadra',24.20,82.72,'industrial','Coal power',37,'partial'],
  ['DL-01','Landfill site','Delhi','East Delhi',28.63,77.33,'unknown','Landfill',7,'good'],
  ['MH-01','LNG terminal','Maharashtra','Ratnagiri',17.00,73.25,'industrial','LNG terminal',20,'good'],
  ['MH-02','Onshore gas terminal','Maharashtra','Raigad',18.90,72.93,'gas_flare','Gas terminal',17,'good'],
  ['MH-03','Coal power station','Maharashtra','Chandrapur',19.95,79.30,'industrial','Coal power',34,'good'],
  ['MH-04','Landfill site','Maharashtra','Mumbai',19.06,72.92,'unknown','Landfill',6,'good'],
  ['KA-01','Coastal refinery','Karnataka','Dakshina Kannada',12.95,74.83,'industrial','Refinery',27,'good'],
  ['KA-02','Integrated steel plant','Karnataka','Ballari',15.15,76.65,'industrial','Steel plant',52,'good'],
  ['KA-03','Landfill site','Karnataka','Bengaluru Rural',12.92,77.47,'unknown','Landfill',5,'good'],
  ['KL-01','Refinery complex','Kerala','Ernakulam',9.97,76.36,'industrial','Refinery',25,'good'],
  ['TN-01','Refinery cluster','Tamil Nadu','Chennai',13.17,80.26,'industrial','Refinery',28,'good'],
  ['TN-02','Lignite mine and power','Tamil Nadu','Cuddalore',11.60,79.48,'mining','Lignite mine',21,'good'],
  ['TN-03','Steel works','Tamil Nadu','Salem',11.70,78.10,'industrial','Steel plant',30,'good'],
  ['TN-04','Cement works','Tamil Nadu','Ariyalur',11.15,79.08,'industrial','Cement plant',15,'good'],
  ['AP-01','Gas field flares','Andhra Pradesh','East Godavari',16.85,82.12,'gas_flare','Oil and gas field',16,'partial'],
  ['AP-02','Gas processing plant','Andhra Pradesh','East Godavari',17.02,81.80,'gas_flare','Gas processing',13,'partial'],
  ['AP-03','Refinery and steel','Andhra Pradesh','Visakhapatnam',17.70,83.22,'industrial','Refinery and steel',44,'good'],
  ['TS-01','Coalfield','Telangana','Peddapalli',18.75,79.50,'mining','Coal mine',19,'partial'],
  ['TS-02','Cement belt','Telangana','Nalgonda',16.90,79.60,'industrial','Cement plant',15,'partial'],
  ['OD-01','Refinery and port','Odisha','Jagatsinghpur',20.30,86.62,'industrial','Refinery',36,'good'],
  ['OD-02','Coalfield','Odisha','Angul',20.95,85.22,'mining','Coal mine',24,'partial'],
  ['OD-03','Coalfield','Odisha','Jharsuguda',21.80,83.95,'mining','Coal mine',22,'partial'],
  ['OD-04','Steel works','Odisha','Sundargarh',22.25,84.85,'industrial','Steel plant',48,'good'],
  ['OD-05','Iron ore mines','Odisha','Keonjhar',21.90,85.60,'mining','Iron ore mine',9,'partial'],
  ['OD-06','Power and aluminium complex','Odisha','Angul',20.85,85.10,'industrial','Power and aluminium',31,'partial'],
  ['JH-01','Coalfield with surface fires','Jharkhand','Dhanbad',23.75,86.42,'mining','Coal mine fires',26,'partial'],
  ['JH-02','Steel works','Jharkhand','Purbi Singhbhum',22.80,86.20,'industrial','Steel plant',50,'good'],
  ['JH-03','Steel works','Jharkhand','Bokaro',23.67,86.15,'industrial','Steel plant',42,'good'],
  ['JH-04','Coalfield','Jharkhand','Ramgarh',23.65,85.40,'mining','Coal mine',18,'partial'],
  ['WB-01','Coalfield','West Bengal','Paschim Bardhaman',23.62,87.13,'mining','Coal mine',20,'partial'],
  ['WB-02','Petrochemical port','West Bengal','Purba Medinipur',22.03,88.10,'industrial','Petrochemical',33,'good'],
  ['CG-01','Coal power cluster','Chhattisgarh','Korba',22.35,82.70,'industrial','Coal power',39,'partial'],
  ['CG-02','Coalfield','Chhattisgarh','Korba',22.42,82.62,'mining','Coal mine',23,'partial'],
  ['CG-03','Integrated steel plant','Chhattisgarh','Durg',21.20,81.38,'industrial','Steel plant',46,'good'],
  ['CG-04','Iron ore mines','Chhattisgarh','Dantewada',18.70,81.20,'mining','Iron ore mine',8,'weak'],
  ['AS-01','Oil field flares','Assam','Tinsukia',27.45,95.35,'gas_flare','Oil and gas field',18,'weak'],
  ['AS-02','Oil field flares','Assam','Sivasagar',26.98,94.62,'gas_flare','Oil and gas field',15,'weak'],
  ['AS-03','Refinery','Assam','Golaghat',26.50,93.96,'industrial','Refinery',24,'weak'],
  ['MP-01','Coal mines','Madhya Pradesh','Singrauli',24.10,82.65,'mining','Coal mine',25,'partial'],
  ['MP-02','Cement works','Madhya Pradesh','Satna',24.60,80.83,'industrial','Cement plant',14,'good']
];
const NEW_SITES = new Set(['MP-02', 'AS-03', 'OD-06', 'TS-02', 'KL-01', 'GJ-06', 'CG-04', 'UP-02', 'JH-04']);
const FORCE_ABN = { 'JH-01': 2, 'GJ-03': 3, 'TN-01': 1, 'OD-04': 5, 'WB-02': 4, 'MH-02': 6 };
const WILD_ZONES = [['Balaghat','Madhya Pradesh',21.8,80.2],['Mandla','Madhya Pradesh',22.6,80.4],['Bastar','Chhattisgarh',19.1,81.9],['Mayurbhanj','Odisha',21.8,86.3],['Kandhamal','Odisha',20.4,84.1],['Almora','Uttarakhand',29.6,79.7],['Pauri Garhwal','Uttarakhand',30.1,78.8],['Kangra','Himachal Pradesh',32.1,76.3],['Nagarkurnool','Telangana',16.4,78.6],['Chamarajanagar','Karnataka',11.9,76.9],['Gadchiroli','Maharashtra',20.2,80.0],['Amravati','Maharashtra',21.4,77.2],['West Kameng','Arunachal Pradesh',27.3,92.4],['Dima Hasao','Assam',25.5,93.0],['Kalahandi','Odisha',19.9,83.2],['Idukki','Kerala',9.8,77.1]];
const AGRI_ZONES = [['Sangrur','Punjab',30.25,75.85],['Ludhiana','Punjab',30.9,75.85],['Patiala','Punjab',30.34,76.4],['Amritsar','Punjab',31.6,74.9],['Karnal','Haryana',29.7,76.98],['Kurukshetra','Haryana',29.97,76.85],['Fatehabad','Haryana',29.5,75.45],['Meerut','Uttar Pradesh',29.0,77.7],['Muzaffarnagar','Uttar Pradesh',29.47,77.7],['Shahjahanpur','Uttar Pradesh',27.88,79.9],['Sri Ganganagar','Rajasthan',29.9,73.9]];
const WILD_W = [.6, 1, 2.2, 3, 2.5, .7, .1, .05, .1, .4, .6, .7];
const AGRI_W = [.1, .1, .3, 1.4, 1, .1, 0, 0, .2, 3, 3.5, .4];

/* ===== Build places and events ===== */
const PLACES = [], EVENTS = [], PMAP = {}, EVP = {};
(function build() {
  const CONF = { wildfire: [.7, .95], agricultural_burning: [.72, .94], gas_flare: [.78, .96], mining: [.65, .9], industrial: [.7, .93], unknown: [.42, .62] };
  const ALT = { gas_flare: ['industrial'], industrial: ['gas_flare', 'mining'], mining: ['industrial'], wildfire: ['agricultural_burning'], agricultural_burning: ['wildfire'], unknown: ['industrial', 'wildfire', 'gas_flare'] };
  const SENS = ['VIIRS NOAA-20', 'VIIRS NOAA-21', 'VIIRS Suomi-NPP', 'MODIS Aqua', 'MODIS Terra'], SW = [.28, .26, .24, .12, .10];
  const HRS = [1.5, 10.5, 13.5, 22.5];
  let pn = 1, en = 1;
  const pickSensor = r => { const x = r(); let a = 0; for (let i = 0; i < SENS.length; i++) { a += SW[i]; if (x < a) return SENS[i]; } return SENS[0]; };
  function mk(p, t, r, status, z0) {
    const lowc = r() < .08 && p.cls !== 'unknown';
    const conf = lowc ? r.range(.48, .6) : r.range(...CONF[p.cls]);
    const alt = r.pick(ALT[p.cls]);
    const p2 = (1 - conf) * r.range(.55, .85);
    let frp, z = null;
    if (p.kind === 'site') {
      const sd = 1.4826 * p.mad;
      const zz = z0 != null ? z0 : Math.max(-2.2, Math.min(2.2, r.norm()));
      frp = Math.exp(Math.log(p.med) + zz * sd) * 1.1;
      if (status !== 'baseline_building') z = (Math.log(frp) - Math.log(p.med)) / sd;
    } else frp = Math.exp(Math.log(p.cls === 'wildfire' ? 14 : 9) + r.norm() * (p.cls === 'wildfire' ? .7 : .5));
    const nDet = p.kind === 'site' ? Math.max(4, Math.round(r.range(8, 30) + p.med * .6)) : (p.cls === 'wildfire' ? r.int(4, 140) : r.int(2, 18));
    const ev = { id: 'E' + String(en++).padStart(4, '0'), pid: p.id, t, cls: p.cls, conf, cls2: alt, p2, status, z, frp, sensor: pickSensor(r), nDet,
      area: +(nDet * .14 * r.range(.8, 1.3)).toFixed(2), review: conf < .6, hr: (r.pick(HRS) + r.range(-1.4, 1.4) + 24) % 24 };
    EVENTS.push(ev); (EVP[p.id] = EVP[p.id] || []).push(ev); return ev;
  }
  SITES.forEach(s => {
    const [code, name, state, district, lat, lon, cls, type, med, cover] = s, r = R('site:' + code);
    const isNew = NEW_SITES.has(code);
    const p = { id: 'S' + String(pn++).padStart(3, '0'), kind: 'site', code, name: `${name}, ${district}`, state, district, lat, lon, cls, type, med, mad: r.range(.14, .32), cover,
      firstSeen: isNew ? TODAY - r.int(20, 55) * DAY : TODAY - r.int(400, 1500) * DAY };
    PLACES.push(p); PMAP[p.id] = p;
    const from = Math.max(p.firstSeen, START), n = Math.max(2, Math.round(r.int(6, 13) * Math.min(1, (TODAY - from) / (320 * DAY))));
    for (let i = 0; i < n; i++) {
      const t = from + r() * (TODAY - from) - 0, tt = Math.min(TODAY - 2 * HOUR, Math.round(t / DAY) * DAY + Math.round(9 * HOUR) - 0);
      const young = tt - p.firstSeen < 20 * DAY;
      const abn = !young && r() < .05;
      mk(p, tt, r, young ? 'baseline_building' : abn ? 'abnormal' : 'routine', abn ? r.range(3.9, 6.2) : null);
    }
    if (FORCE_ABN[code] && !isNew) mk(p, TODAY - FORCE_ABN[code] * DAY + 2 * HOUR, r, 'abnormal', r.range(4, 6.4));
  });
  function transient(cls, zones, W, count) {
    const r = R('trans:' + cls); let made = 0;
    while (made < count) {
      const d = r.int(0, 364), t = START + d * DAY, m = dparts(t).m;
      if (r() * Math.max(...W) > W[m]) continue;
      const zn = r.pick(zones), lat = zn[2] + r.range(-.18, .18), lon = zn[3] + r.range(-.18, .18);
      const p = { id: 'T' + String(pn++).padStart(4, '0'), kind: 'transient', code: 'T', name: (cls === 'wildfire' ? 'Forest fire near ' : 'Crop residue burning near ') + zn[0], state: zn[1], district: zn[0], lat, lon, cls, type: cls === 'wildfire' ? 'Vegetation fire' : 'Crop residue fire', firstSeen: t, cover: 'partial', mad: .3, med: 10 };
      PLACES.push(p); PMAP[p.id] = p;
      mk(p, t + Math.round(r.range(1, 14)) * HOUR, r, 'not_applicable', null); made++;
    }
  }
  transient('wildfire', WILD_ZONES, WILD_W, 130);
  transient('agricultural_burning', AGRI_ZONES, AGRI_W, 150);
  EVENTS.sort((a, b) => a.t - b.t);
  Object.values(EVP).forEach(a => a.sort((x, y) => x.t - y.t));
})();

/* raw detection points around an event (deterministic) */
function ptsOf(ev) {
  if (ev._pts) return ev._pts;
  const p = PMAP[ev.pid], r = R('pts:' + ev.id), n = Math.min(ev.nDet, 36);
  const sg = p.kind === 'site' ? .012 + Math.min(.05, ev.area * .004) : .02 + Math.min(.12, Math.sqrt(ev.area) * .03);
  ev._pts = Array.from({ length: n }, () => [p.lon + r.norm() * sg, p.lat + r.norm() * sg * .9]);
  return ev._pts;
}

/* ===== Time series for a place ===== */
function seriesOf(p) {
  if (p._ser) return p._ser;
  const r = R('ser:' + p.id), evs = EVP[p.id] || [], obs = [], gaps = [];
  const sd = 1.4826 * p.mad;
  if (p.kind === 'transient') {
    const ev = evs[0], nd = r.int(1, 4);
    for (let i = 0; i < nd; i++) obs.push({ t: ev.t + i * DAY * r.range(.5, 1.3), frp: Math.max(1, ev.frp * Math.exp(r.norm() * .35) * (i === 0 ? 1 : .8)), cnt: r.int(1, 5), sensor: r() < .7 ? ev.sensor : 'VIIRS Suomi-NPP', z: null });
    obs.sort((a, b) => a.t - b.t);
    return p._ser = { obs, gaps: [] };
  }
  const off = { until: -1 }; let cloudRun = 0, cloudStart = 0;
  const abn = evs.filter(e => e.status === 'abnormal');
  for (let d = 0; d <= 364; d++) {
    const t = START + d * DAY + 9 * HOUR; if (t < p.firstSeen) continue;
    const m = dparts(t).m, pobs = (m >= 5 && m <= 8) ? .38 : (m === 9 || m === 10) ? .7 : .82;
    if (d > off.until && r() < .012) off.until = d + r.int(5, 12);
    const seen = r() < pobs;
    if (!seen) { if (!cloudRun) cloudStart = t; cloudRun++; continue; }
    if (cloudRun >= 5) gaps.push([cloudStart, t - DAY]); cloudRun = 0;
    if (d <= off.until) continue;
    let ln = Math.log(p.med) + r.norm() * sd * .75, z = null;
    abn.forEach(e => { const dd = Math.abs(t - e.t) / DAY; if (dd < 3.2) ln = Math.log(p.med) + (e.z * (1 - dd * .22)) * sd + r.norm() * sd * .2; });
    obs.push({ t, frp: Math.exp(ln), cnt: r.int(1, 3), sensor: r() < .78 ? r.pick(['VIIRS NOAA-20', 'VIIRS NOAA-21', 'VIIRS Suomi-NPP']) : r.pick(['MODIS Aqua', 'MODIS Terra']), z });
  }
  const base = baselineOf(obs, p);
  obs.forEach(o => { o.z = base.n >= 20 ? (Math.log(o.frp) - base.mLn) / base.sd : null; });
  return p._ser = { obs, gaps, base };
}
function baselineOf(obs, p) {
  const ln = obs.map(o => Math.log(o.frp)), m = median(ln), mad = median(ln.map(x => Math.abs(x - m))) || .1, sd = 1.4826 * mad;
  return { n: obs.length, mLn: m, med: Math.exp(m), mad, sd, p95: quant(obs.map(o => o.frp), .95), lo: Math.exp(m - 3.5 * sd), hi: Math.exp(m + 3.5 * sd) };
}
function statsOf(p) {
  const ser = seriesOf(p), obs = ser.obs; const out = {};
  const days = n => new Set(obs.filter(o => o.t > TODAY - n * DAY).map(o => Math.floor(o.t / DAY))).size;
  out.d30 = days(30); out.d90 = days(90);
  out.months = new Set(obs.map(o => dparts(o.t).m)).size;
  const set = new Set(obs.map(o => Math.floor(o.t / DAY)));
  let streak = 0; for (let d = Math.floor(TODAY / DAY); set.has(d); d--) streak++;
  out.streak = streak;
  const last = obs.length ? obs[obs.length - 1].t : null; out.last = last;
  out.sinceLast = last ? Math.max(0, Math.round((TODAY - last) / DAY)) : null;
  return out;
}
function hoursOf(p) {
  const r = R('hr:' + p.id), night = { gas_flare: .8, industrial: .6, mining: .55, wildfire: .25, agricultural_burning: .1, unknown: .4 }[p.cls];
  const H = Array(24).fill(0), base = [1, 2, 10, 11, 13, 14, 22, 23];
  const n = 90 + r.int(0, 40);
  for (let i = 0; i < n; i++) {
    const nt = r() < night, pool = nt ? [1, 1, 2, 22, 23] : [10, 11, 13, 13, 14];
    H[r.pick(pool)]++;
  }
  return H;
}

/* ===== Explanations now live in features.js (141 features) ===== */
const PROF = {
  base:{frp:[6,40],bt:[335,360],area:[.3,2],persist:[40,90],night:[45,75],dOil:[15,80],dInd:[.2,8],dMine:[10,80],dLand:[10,70],crop:[2,30],forest:[0,20],bare:[5,35],indus:[5,40],ndvi:[.15,.5],ndbi:[.05,.3],vpd:[1.2,3.8],rain:[0,4]},
  gas_flare:{frp:[8,30],bt:[352,367],persist:[74,90],night:[70,95],dOil:[.05,1.2],dInd:[.05,1.5],indus:[25,60],bare:[15,45],forest:[0,8],crop:[0,15]},
  industrial:{frp:[12,60],bt:[345,364],area:[.4,2],persist:[60,90],night:[50,75],dInd:[.05,1.2],dOil:[3,40],indus:[30,70],ndbi:[.2,.4],forest:[0,10]},
  mining:{frp:[6,40],dMine:[0,1],bare:[45,75],persist:[35,90],night:[55,70],dInd:[.3,6],ndbi:[.05,.2],forest:[0,12]},
  wildfire:{frp:[6,70],bt:[325,355],area:[.5,9],persist:[0,5],night:[15,35],dInd:[8,45],forest:[55,90],crop:[0,15],ndvi:[.5,.78],vpd:[2.5,5],rain:[0,3],bare:[0,8],indus:[0,2]},
  agricultural_burning:{frp:[4,22],bt:[322,342],area:[.15,1.2],persist:[0,4],night:[5,20],dInd:[6,40],crop:[70,95],ndvi:[.15,.4],vpd:[1.8,3.5],rain:[0,3],bare:[2,15],indus:[0,3],forest:[0,8]},
  unknown:{frp:[2,9],bt:[320,338],area:[.1,.8],persist:[10,60],night:[20,60],dInd:[.3,5],dLand:[.1,3],ndbi:[.1,.35],indus:[5,25],forest:[0,10]}
};
function profOf(cls, key, r) { const a = (PROF[cls] && PROF[cls][key]) || PROF.base[key]; if (!a) return 0; return r.range(a[0], a[1]); }
/* values for land cover, distances, imagery, weather (per place, sample) */
function contextOf(p) {
  if (p._ctx) return p._ctx; const r = R('ctx:' + p.id), c = p.cls;
  let crop = profOf(c, 'crop', r), forest = profOf(c, 'forest', r), bare = profOf(c, 'bare', r), ind = profOf(c, 'indus', r), grass = r.range(2, 12);
  const built = ind; const sum = crop + forest + bare + built + grass, water = Math.max(0, 100 - sum);
  const parts = [['Cropland', crop], ['Forest', forest], ['Bare ground', bare], ['Industrial or built-up', built], ['Grass and scrub', grass], ['Water and other', water]];
  const tot = parts.reduce((a, x) => a + x[1], 0); parts.forEach(x => x[1] = +(x[1] / tot * 100).toFixed(1));
  const fac = c === 'unknown' && p.type !== 'Landfill' ? null : true;
  const d = { refinery: p.type.includes('Refinery') || p.type.includes('Petro') ? r.range(0, .8) : r.range(40, 400), power: /power|Power/.test(p.type) ? r.range(0, .9) : r.range(15, 200), steel: /Steel|Cement/.test(p.type) ? r.range(0, .9) : r.range(15, 250), mine: c === 'mining' ? r.range(0, .7) : r.range(10, 150), landfill: p.type === 'Landfill' ? r.range(0, .3) : r.range(8, 90) };
  if (p.kind === 'transient') Object.keys(d).forEach(k => d[k] = r.range(6, 90));
  p._ctx = { parts, d, fac, ndvi: profOf(c, 'ndvi', r), nbr: r.range(.05, .55), ndbi: profOf(c, 'ndbi', r), built: ind / 100, valid: r.range(.25, .96), sceneDays: r.int(2, 11), vpd: profOf(c, 'vpd', r), rain: profOf(c, 'rain', r), elev: r.int(5, 900), slope: r.range(0, 9) };
  return p._ctx;
}
/* nearby micro-sources (sample) */
function nearbyOf(p) {
  const r = R('nb:' + p.id), n = p.kind === 'site' ? r.int(3, 6) : r.int(0, 2), out = [];
  const nm = { industrial: ['Furnace stack', 'Boiler stack', 'Storage yard', 'Kiln'], gas_flare: ['Flare stack B', 'Flare stack C', 'Compressor vent'], mining: ['Waste heap', 'Coal stockpile', 'Surface fire patch'], unknown: ['Heap', 'Reflective roof'], wildfire: ['Fire front'], agricultural_burning: ['Burning field'] }[p.cls] || ['Source'];
  for (let i = 0; i < n; i++) { const c = r() < .75 ? p.cls : r.pick(['industrial', 'gas_flare', 'unknown']); out.push({ name: r.pick(nm) + ' ' + String.fromCharCode(65 + i), cls: c, km: +r.range(.4, 9.5).toFixed(1), ang: r.range(0, 6.283), st: r() < .12 && p.kind === 'site' ? 'abnormal' : 'routine' }); }
  return out;
}

/* ===== Sample model health / region numbers ===== */
const METRICS = [
  ['wildfire', .81, .84, 412], ['agricultural_burning', .86, .83, 530], ['gas_flare', .79, .74, 188], ['mining', .68, .61, 142], ['industrial', .76, .82, 615], ['unknown', .52, .49, 160]
].map(([id, p, r, n]) => ({ id, p, r, n, f1: 2 * p * r / (p + r) }));
