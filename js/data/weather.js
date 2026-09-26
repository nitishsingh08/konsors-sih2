'use strict';
/* =====================================================================
   Sample wind, weather, smoke plume and spread outlook.
   The fire-weather maths (FFMC, ISI, BUI, FWI) is the standard Canadian system.
   The plume and spread shapes are simple and INDICATIVE. They are stand-ins for
   what your backend will return (see docs/api-contract.md).
   ===================================================================== */
const M_LAT = 110540, mLon = lat => 111320 * Math.cos(lat * Math.PI / 180), rad = d => d * Math.PI / 180;
const COMPASS8 = ['north', 'north-east', 'east', 'south-east', 'south', 'south-west', 'west', 'north-west'];
const compass = deg => COMPASS8[Math.round((((deg % 360) + 360) % 360) / 45) % 8];
const compassShort = deg => ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'][Math.round((((deg % 360) + 360) % 360) / 45) % 8];
const offsetLL = (lon, lat, dx, dy) => [lon + dx / mLon(lat), lat + dy / M_LAT];
const DOW = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
function hourLabel(t0, h) { const d = new Date(t0 + h * HOUR + IST), hh = d.getUTCHours(); return `${DOW[d.getUTCDay()]} ${hh % 12 || 12} ${hh >= 12 ? 'pm' : 'am'}`; }
const vpdOf = (T, RH) => .6108 * Math.exp(17.27 * T / (T + 237.3)) * (1 - RH / 100);

/* --- Canadian Fire Weather Index pieces --- */
function calcISI(ffmc, windKmh) { const m = 147.2 * (101 - ffmc) / (59.5 + ffmc), fF = 91.9 * Math.exp(-.1386 * m) * (1 + Math.pow(m, 5.31) / 4.93e7); return .208 * Math.exp(.05039 * windKmh) * fF; }
function calcBUI(dmc, dc) { if (dmc <= 0 && dc <= 0) return 0; return dmc <= .4 * dc ? .8 * dmc * dc / (dmc + .4 * dc) : dmc - (1 - .8 * dc / (dmc + .4 * dc)) * (.92 + Math.pow(.0114 * dmc, 1.7)); }
function calcFWI(isi, bui) { const fD = bui <= 80 ? .626 * Math.pow(bui, .809) + 2 : 1000 / (25 + 108.64 * Math.exp(-.023 * bui)), B = .1 * isi * fD; return B > 1 ? Math.exp(2.72 * Math.pow(.434 * Math.log(B), .647)) : B; }
const spreadClass = isi => isi < 3 ? 'Low' : isi < 8 ? 'Moderate' : isi < 15 ? 'High' : 'Very high';

/* --- hourly forecast for a place, 0 to 72 hours from the event time --- */
function weatherOf(p, ev) {
  if (ev._wx) return ev._wx;
  const r = R('wx:' + ev.id), ctx = contextOf(p), m = dparts(ev.t).m, dry = [1, 2, 3, 4, 9, 10].includes(m), mon = m >= 5 && m <= 8;
  const T0 = [21, 24, 29, 34, 37, 34, 31, 30, 30, 30, 26, 22][m] + r.norm() * 1.5, rh0 = clamp((mon ? 82 : dry ? 38 : 55) + r.norm() * 8, 18, 96);
  const spd0 = r.range(1.6, 6.2) + (ctx.slope > 5 ? .8 : 0), dir0 = r.range(0, 360), shiftAt = r.int(5, 30), shiftBy = (r() < .5 ? -1 : 1) * r.range(55, 115), h0 = dparts(ev.t).h;
  const ffBase = mon ? 66 + r.norm() * 5 : dry ? 88 + r.norm() * 3 : 78 + r.norm() * 4, rainAt = r() < (mon ? .55 : .2) ? r.int(6, 40) : -1;
  const N = 73, W = { t0: ev.t, n: N, spd: [], gust: [], dir: [], temp: [], rh: [], vpd: [], rain: [], ffmc: [] };
  let cum = 0;
  for (let h = 0; h < N; h++) {
    const di = Math.sin(((h0 + h) % 24 - 9) / 24 * 2 * Math.PI), s = clamp(spd0 * (.78 + .3 * di) + r.norm() * .25, .4, 24), sh = 1 / (1 + Math.exp(-(h - shiftAt) / 1.6));
    W.dir.push((((dir0 + shiftBy * sh + r.norm() * 3) % 360) + 360) % 360); W.spd.push(s); W.gust.push(s * r.range(1.35, 1.75));
    const T = T0 + 4.2 * di - 1.5, rn = rainAt >= 0 && h >= rainAt && h < rainAt + 4 ? r.range(.5, 3.5) : 0; cum += rn;
    const RH = clamp(rh0 - 15 * di + r.norm() * 2 + rn * 4, 10, 100);
    W.temp.push(T); W.rain.push(rn); W.rh.push(RH); W.vpd.push(vpdOf(T, RH)); W.ffmc.push(clamp(ffBase + (55 - RH) * .12 - cum * 4, 30, 98));
  }
  W.dmc = clamp((mon ? 18 : dry ? 70 : 40) + r.norm() * 8, 2, 200); W.dc = clamp((mon ? 120 : dry ? 380 : 250) + r.norm() * 50, 10, 750); W.rainPast24 = r() < .3 ? r.range(0, mon ? 14 : 3) : 0;
  return ev._wx = W;
}
function windAt(W, h) {
  const i = clamp(Math.floor(h), 0, W.n - 2), f = clamp(h - i, 0, 1), s = W.spd[i] * (1 - f) + W.spd[i + 1] * f, dd = ((W.dir[i + 1] - W.dir[i] + 540) % 360) - 180, d = (W.dir[i] + dd * f + 360) % 360, a = rad(d);
  return { spd: s, dir: d, u: -s * Math.sin(a), v: -s * Math.cos(a) };
}
function windShift(W) {
  let best = null; for (let h = 0; h < 36; h++) { const d = Math.abs(((W.dir[h + 4] - W.dir[h] + 540) % 360) - 180); if (!best || d > best.d) best = { d, h: h + 2 }; }
  return best && best.d >= 35 ? { h: best.h, deg: best.d, to: W.dir[best.h + 2] } : null;
}

/* --- wind fields for the map (u east, v north, metres per second) --- */
function baseWind(lat, lon, h) {
  const t = h / 24 * 2 * Math.PI, a = (lat - 8) / 30;
  return [3.6 * (1 - a) + 1.6 * Math.sin(lon * .19 + lat * .11 + t * .55 + .6) + 1.1 * Math.sin(lat * .35 - lon * .09 + t * .8) - .4, 1.9 * (1 - a) + 1.5 * Math.cos(lon * .15 - lat * .17 + t * .5) + .9 * Math.cos(lat * .31 + lon * .12 - t * .7)];
}
function makeGrid(b, nx, ny, fn) {
  const u = new Float32Array(nx * ny), v = new Float32Array(nx * ny), dx = (b.lo1 - b.lo0) / (nx - 1), dy = (b.la1 - b.la0) / (ny - 1);
  for (let j = 0; j < ny; j++) { const lat = b.la1 - j * dy; for (let i = 0; i < nx; i++) { const w = fn(lat, b.lo0 + i * dx); u[j * nx + i] = w[0]; v[j * nx + i] = w[1]; } }
  return { lo0: b.lo0, la1: b.la1, dx, dy, nx, ny, u, v };
}
function gridSample(g, lon, lat) {
  const fx = (lon - g.lo0) / g.dx, fy = (g.la1 - lat) / g.dy; if (fx < 0 || fy < 0 || fx > g.nx - 1 || fy > g.ny - 1) return null;
  const i = Math.floor(fx), j = Math.floor(fy), i1 = Math.min(i + 1, g.nx - 1), j1 = Math.min(j + 1, g.ny - 1), tx = fx - i, ty = fy - j;
  const bl = a => a[j * g.nx + i] * (1 - tx) * (1 - ty) + a[j * g.nx + i1] * tx * (1 - ty) + a[j1 * g.nx + i] * (1 - tx) * ty + a[j1 * g.nx + i1] * tx * ty;
  return [bl(g.u), bl(g.v)];
}
const nationalGrid = h => makeGrid({ lo0: 66, lo1: 99, la0: 5, la1: 39 }, 34, 35, (la, lo) => baseWind(la, lo, h));
function localGrid(p, W, h) {
  const w = windAt(W, h), b0 = baseWind(p.lat, p.lon, h);
  return makeGrid({ lo0: p.lon - 1.8, lo1: p.lon + 1.8, la0: p.lat - 1.8, la1: p.lat + 1.8 }, 45, 45, (la, lo) => { const b = baseWind(la, lo, h); return [w.u + (b[0] - b0[0]) * .6, w.v + (b[1] - b0[1]) * .6]; });
}

/* --- smoke plume: drift with the forecast wind, widening downwind (rural, neutral conditions) --- */
function plumeOf(p, W, H) {
  H = clamp(H, 1, 72); const steps = Math.round(H * 2), traj = [[p.lon, p.lat, 0, 0]]; let lon = p.lon, lat = p.lat, x = 0;
  for (let s = 1; s <= steps; s++) { const w = windAt(W, (s - .5) / 2), dx = w.u * 1800, dy = w.v * 1800; [lon, lat] = offsetLL(lon, lat, dx, dy); x += Math.hypot(dx, dy); traj.push([lon, lat, x, s / 2]); }
  const cL = [], cR = [], oL = [], oR = [];
  for (let i = 0; i < traj.length; i++) {
    const a = traj[Math.max(0, i - 1)], b = traj[Math.min(traj.length - 1, i + 1)], dxm = (b[0] - a[0]) * mLon(p.lat), dym = (b[1] - a[1]) * M_LAT, L = Math.hypot(dxm, dym) || 1, nx = -dym / L, ny = dxm / L;
    const xx = traj[i][2], hc = Math.max(250, 2 * .08 * xx / Math.sqrt(1 + 1e-4 * xx)) + 150, ho = hc + Math.min(xx * Math.tan(rad(8 + .5 * traj[i][3])), xx * .9), P = traj[i];
    cL.push(offsetLL(P[0], P[1], nx * hc, ny * hc)); cR.push(offsetLL(P[0], P[1], -nx * hc, -ny * hc)); oL.push(offsetLL(P[0], P[1], nx * ho, ny * ho)); oR.push(offsetLL(P[0], P[1], -nx * ho, -ny * ho));
  }
  const head = traj[traj.length - 1], bearing = (Math.atan2((head[0] - p.lon) * mLon(p.lat), (head[1] - p.lat) * M_LAT) * 180 / Math.PI + 360) % 360;
  return { traj, core: cL.concat(cR.reverse()), outer: oL.concat(oR.reverse()), reachKm: x / 1000, head: [head[0], head[1]], bearing, H };
}

/* --- terrain along a line from the source --- */
function terrainAlong(p, dirTo, kmMax, n) {
  n = n || 41; kmMax = kmMax || 10;
  const r = R('terr:' + p.id + ':' + Math.round(dirTo / 15)), ctx = contextOf(p), relief = { wildfire: 160, mining: 70, agricultural_burning: 15, gas_flare: 20, industrial: 25, unknown: 20 }[p.cls] || 30;
  const f1 = r.range(0, 6.28), f2 = r.range(0, 6.28), tr = r.range(-1, 1), v = [Math.sin(rad(dirTo)), Math.cos(rad(dirTo))], pts = []; let sum = 0, up = 0;
  for (let i = 0; i < n; i++) {
    const km = kmMax * i / (n - 1), el = ctx.elev + relief * (.6 * Math.sin(km * .55 + f1) + .3 * Math.sin(km * 1.35 + f2) + .5 * tr * km / kmMax), ll = offsetLL(p.lon, p.lat, v[0] * km * 1000, v[1] * km * 1000);
    pts.push({ km, elev: el, lon: ll[0], lat: ll[1] });
    if (i) { const sl = Math.atan((el - pts[i - 1].elev) / ((km - pts[i - 1].km) * 1000)) * 180 / Math.PI; sum += sl; up = Math.max(up, sl); }
  }
  return { pts, meanSlope: sum / (n - 1), maxUp: up };
}

/* --- fire spread outlook (INDICATIVE): vegetation fires only --- */
const isVeg = c => c === 'wildfire' || c === 'agricultural_burning';
function ellipseRing(p, dirTo, D, LB) {
  const a = D * .55, b = a / LB, c = D - a, v = [Math.sin(rad(dirTo)), Math.cos(rad(dirTo))], q = [Math.cos(rad(dirTo)), -Math.sin(rad(dirTo))], ring = [];
  for (let k = 0; k < 48; k++) { const th = k / 48 * 2 * Math.PI, X = c + a * Math.cos(th), Y = b * Math.sin(th); ring.push(offsetLL(p.lon, p.lat, (X * v[0] + Y * q[0]) * 1000, (X * v[1] + Y * q[1]) * 1000)); }
  return ring;
}
function areaKm2(ring) {
  if (!ring || ring.length < 3) return 0; const lat0 = ring[0][1], lon0 = ring[0][0]; let s = 0;
  for (let i = 0; i < ring.length; i++) { const a = ring[i], b = ring[(i + 1) % ring.length]; s += ((a[0] - lon0) * mLon(lat0) / 1000) * ((b[1] - lat0) * M_LAT / 1000) - ((b[0] - lon0) * mLon(lat0) / 1000) * ((a[1] - lat0) * M_LAT / 1000); }
  return Math.abs(s) / 2;
}
function spreadOf(p, ev, W) {
  if (!isVeg(ev.cls)) return null;
  const hz = [1, 3, 6], out = [];
  hz.forEach(t => {
    let su = 0, sv = 0, ff = 0, n = 0; for (let s = 0; s <= t * 2; s++) { const w = windAt(W, s / 2); su += w.u; sv += w.v; ff += W.ffmc[Math.min(72, Math.round(s / 2))]; n++; }
    su /= n; sv /= n; ff /= n; const spd = Math.hypot(su, sv), from = (Math.atan2(-su, -sv) * 180 / Math.PI + 360) % 360, to = (from + 180) % 360, kmh = spd * 3.6;
    const ter = terrainAlong(p, to, 8), slopeF = ter.meanSlope > 0 ? 1 + clamp(.06 * ter.meanSlope, 0, 1) : clamp(1 + .03 * ter.meanSlope, .6, 1);
    const isi = calcISI(ff, kmh), rate = clamp(.055 * isi * slopeF, .05, 6), LB = clamp(.936 * Math.exp(.2566 * kmh) + .461 * Math.exp(-.1548 * kmh) - .397, 1, 6), D = rate * t, ring = ellipseRing(p, to, D, LB);
    out.push({ t, isi, rate, rateLo: rate * .5, rateHi: rate * 1.8, D, dirTo: to, LB, slopeF, meanSlope: ter.meanSlope, area: areaKm2(ring), ring, cls: spreadClass(isi) });
  });
  const last = out[2], unc = ellipseRing(p, last.dirTo, last.D * 1.6, Math.max(1, last.LB * .85));
  return { horizons: out, unc, cls: out[2].cls };
}

/* --- the footprints seen on the last eight passes --- */
function footprintsOf(p, ev) {
  const f = featuresOf(p, ev), r = R('fp8:' + ev.id), grow = isVeg(ev.cls) || ev.status === 'abnormal', dirTo = f.ang.SPREAD_DIRECTION, el = grow ? clamp(f.by.ELONGATION.v, 1.2, 3) : 1.15, frames = [], n = 8, ph = r.range(0, 6), ph2 = r.range(0, 6);
  const v = [Math.sin(rad(dirTo)), Math.cos(rad(dirTo))], q = [Math.cos(rad(dirTo)), -Math.sin(rad(dirTo))];
  for (let i = 0; i < n; i++) {
    const k = grow ? .22 + .78 * i / (n - 1) : .92 + .08 * r(), area = Math.max(.02, ev.area * k), b = Math.sqrt(area / (Math.PI * el)), a = b * el, shift = grow ? (i - (n - 1)) * Math.sqrt(area) * .18 : 0, ring = [];
    for (let s = 0; s < 40; s++) { const th = s / 40 * 2 * Math.PI, m = 1 + .16 * Math.sin(3 * th + ph) + .09 * Math.sin(5 * th + ph2), X = shift + a * m * Math.cos(th), Y = b * m * Math.sin(th); ring.push(offsetLL(p.lon, p.lat, (X * v[0] + Y * q[0]) * 1000, (X * v[1] + Y * q[1]) * 1000)); }
    frames.push({ t: ev.t - (n - 1 - i) * 12 * HOUR, area, ring });
  }
  return frames;
}

/* --- who and what sits inside the zones (sample counts; real ones come from OSM) --- */
function exposureOf(p, ev, plume, spread) {
  const r = R('expo:' + ev.id), dens = { wildfire: .22, agricultural_burning: .9, industrial: .6, gas_flare: .35, mining: .5, unknown: 1.1 }[ev.cls] || .5;
  const zone = (name, ring) => { const a = areaKm2(ring); return { name, area: a, settlements: Math.round(dens * a * r.range(.04, .09)), schools: Math.round(dens * a * r.range(.008, .02)), clinics: Math.round(dens * a * r.range(.002, .007)), roadKm: Math.round(dens * a * r.range(.25, .6)), water: Math.round(a * r.range(.01, .05)) }; };
  const z = [zone('Smoke plume, core, next 6 hours', plumeOf(p, ev._wx || weatherOf(p, ev), 6).core)];
  if (spread) z.push(zone('Fire spread outlook, 6 hours', spread.horizons[2].ring));
  return z;
}
function similarOf(ev) {
  const r = R('sim:' + ev.id);
  return EVENTS.filter(e => e.cls === ev.cls && e.pid !== ev.pid && e.t < ev.t).map(e => ({ e, d: Math.abs(Math.log(e.frp / ev.frp)) + Math.abs(e.conf - ev.conf) })).sort((a, b) => a.d - b.d).slice(0, 5)
    .map(x => ({ e: x.e, p: PMAP[x.e.pid], sim: clamp(.96 - x.d * .25 - r() * .03, .6, .96), outcome: r() < .72 ? 'Confirmed by an analyst' : r() < .6 ? 'Marked as a false alarm' : 'Not reviewed' }));
}
function freshnessOf(ev) {
  const r = R('fr:' + ev.id);
  return [
    { name: 'Thermal detections (FIRMS)', label: '2 h 41 min old', hours: 2.7, note: 'VIIRS and MODIS, near real time' },
    { name: 'Weather forecast', label: 'issued at event time', hours: 0, note: 'Wind, humidity, temperature, rain' },
    { name: 'Optical imagery (HLS)', label: r.int(2, 8) + ' days old', hours: 0, note: 'Limited by cloud' },
    { name: 'Radar (Sentinel-1)', label: r.int(3, 10) + ' days old', hours: 0, note: 'Passes every few days' },
    { name: 'Air chemistry (TROPOMI)', label: r.int(6, 40) + ' hours old', hours: 0, note: 'One pass a day, then processing' },
    { name: 'Facility records (GEM, OSM)', label: 'updated ' + r.int(10, 60) + ' days ago', hours: 0, note: 'Coverage is uneven in India' }
  ];
}
function outlookLine(p, ev, W) {
  const w0 = windAt(W, 0), pl = plumeOf(p, W, 6), sh = windShift(W), lvl = w0.spd < 3 ? 'light' : w0.spd < 6 ? 'moderate' : 'strong';
  let s = `Wind is ${lvl} from the ${compass(w0.dir)} at ${Math.round(w0.spd * 3.6)} km/h`;
  if (sh) s += ` and swings round toward the ${compass(sh.to)} around ${hourLabel(W.t0, sh.h)}`;
  return s + `. Smoke could drift about ${pl.reachKm.toFixed(0)} km ${compass(pl.bearing)} within 6 hours.`;
}
