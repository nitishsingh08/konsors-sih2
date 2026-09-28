/* =====================================================================
   Sample wind, weather, smoke plume and spread outlook.
   The fire-weather maths (FFMC, ISI, BUI, FWI) is the standard Canadian system.
   The plume and spread shapes are simple and INDICATIVE. They are stand-ins for
   what the backend will return (see docs/api-contract.md).
   Port of frontend/js/data/weather.js.
   ===================================================================== */
import { clamp, R } from '../lib/core';
import { dparts } from '../lib/time';
import { isVeg } from './classes';
import { contextOf } from './places';
import type { Event, LonLat, Plume, Place, SpreadHorizon, SpreadOutlook, Terrain, Weather, Wind, WindGrid } from '../lib/types';

export const M_LAT = 110540;
export const mLon = (lat: number): number => 111320 * Math.cos((lat * Math.PI) / 180);
export const rad = (d: number): number => (d * Math.PI) / 180;

const COMPASS8 = ['north', 'north-east', 'east', 'south-east', 'south', 'south-west', 'west', 'north-west'];
const COMPASS8_SHORT = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];

export const compass = (deg: number): string => COMPASS8[Math.round((((deg % 360) + 360) % 360) / 45) % 8];
export const compassShort = (deg: number): string => COMPASS8_SHORT[Math.round((((deg % 360) + 360) % 360) / 45) % 8];

/** move dx metres east and dy metres north */
export const offsetLL = (lon: number, lat: number, dx: number, dy: number): LonLat => [lon + dx / mLon(lat), lat + dy / M_LAT];

const vpdOf = (T: number, RH: number): number => 0.6108 * Math.exp((17.27 * T) / (T + 237.3)) * (1 - RH / 100);

/* --- Canadian Fire Weather Index pieces --- */
export function calcISI(ffmc: number, windKmh: number): number {
  const m = (147.2 * (101 - ffmc)) / (59.5 + ffmc);
  const fF = 91.9 * Math.exp(-0.1386 * m) * (1 + Math.pow(m, 5.31) / 4.93e7);
  return 0.208 * Math.exp(0.05039 * windKmh) * fF;
}

export function calcBUI(dmc: number, dc: number): number {
  if (dmc <= 0 && dc <= 0) return 0;
  return dmc <= 0.4 * dc
    ? (0.8 * dmc * dc) / (dmc + 0.4 * dc)
    : dmc - (1 - (0.8 * dc) / (dmc + 0.4 * dc)) * (0.92 + Math.pow(0.0114 * dmc, 1.7));
}

export function calcFWI(isi: number, bui: number): number {
  const fD = bui <= 80 ? 0.626 * Math.pow(bui, 0.809) + 2 : 1000 / (25 + 108.64 * Math.exp(-0.023 * bui));
  const B = 0.1 * isi * fD;
  return B > 1 ? Math.exp(2.72 * Math.pow(0.434 * Math.log(B), 0.647)) : B;
}

export const spreadClass = (isi: number): string => (isi < 3 ? 'Low' : isi < 8 ? 'Moderate' : isi < 15 ? 'High' : 'Very high');

/* --- hourly forecast for a place, 0 to 72 hours from the event time --- */
export function weatherOf(p: Place, ev: Event): Weather {
  if (ev._wx) return ev._wx;
  const r = R('wx:' + ev.id);
  const ctx = contextOf(p);
  const m = dparts(ev.t).m;
  const dry = [1, 2, 3, 4, 9, 10].includes(m);
  const mon = m >= 5 && m <= 8;
  const T0 = [21, 24, 29, 34, 37, 34, 31, 30, 30, 30, 26, 22][m] + r.norm() * 1.5;
  const rh0 = clamp((mon ? 82 : dry ? 38 : 55) + r.norm() * 8, 18, 96);
  const spd0 = r.range(1.6, 6.2) + (ctx.slope > 5 ? 0.8 : 0);
  const dir0 = r.range(0, 360);
  const shiftAt = r.int(5, 30);
  const shiftBy = (r() < 0.5 ? -1 : 1) * r.range(55, 115);
  const h0 = dparts(ev.t).h;
  const ffBase = mon ? 66 + r.norm() * 5 : dry ? 88 + r.norm() * 3 : 78 + r.norm() * 4;
  const rainAt = r() < (mon ? 0.55 : 0.2) ? r.int(6, 40) : -1;

  const N = 73;
  const W: Weather = {
    t0: ev.t, n: N, spd: [], gust: [], dir: [], temp: [], rh: [], vpd: [], rain: [], ffmc: [],
    dmc: 0, dc: 0, rainPast24: 0,
  };
  let cum = 0;
  for (let h = 0; h < N; h++) {
    const di = Math.sin((((h0 + h) % 24 - 9) / 24) * 2 * Math.PI);
    const s = clamp(spd0 * (0.78 + 0.3 * di) + r.norm() * 0.25, 0.4, 24);
    const sh = 1 / (1 + Math.exp(-(h - shiftAt) / 1.6));
    W.dir.push(((((dir0 + shiftBy * sh + r.norm() * 3) % 360) + 360) % 360));
    W.spd.push(s);
    W.gust.push(s * r.range(1.35, 1.75));
    const T = T0 + 4.2 * di - 1.5;
    const rn = rainAt >= 0 && h >= rainAt && h < rainAt + 4 ? r.range(0.5, 3.5) : 0;
    cum += rn;
    const RH = clamp(rh0 - 15 * di + r.norm() * 2 + rn * 4, 10, 100);
    W.temp.push(T);
    W.rain.push(rn);
    W.rh.push(RH);
    W.vpd.push(vpdOf(T, RH));
    W.ffmc.push(clamp(ffBase + (55 - RH) * 0.12 - cum * 4, 30, 98));
  }
  W.dmc = clamp((mon ? 18 : dry ? 70 : 40) + r.norm() * 8, 2, 200);
  W.dc = clamp((mon ? 120 : dry ? 380 : 250) + r.norm() * 50, 10, 750);
  W.rainPast24 = r() < 0.3 ? r.range(0, mon ? 14 : 3) : 0;
  return (ev._wx = W);
}

export function windAt(W: Weather, h: number): Wind {
  const i = clamp(Math.floor(h), 0, W.n - 2);
  const f = clamp(h - i, 0, 1);
  const s = W.spd[i] * (1 - f) + W.spd[i + 1] * f;
  const dd = ((W.dir[i + 1] - W.dir[i] + 540) % 360) - 180;
  const d = (W.dir[i] + dd * f + 360) % 360;
  const a = rad(d);
  return { spd: s, dir: d, u: -s * Math.sin(a), v: -s * Math.cos(a) };
}

/** the biggest direction swing in the next 36 hours, if it is worth showing */
export function windShift(W: Weather): { h: number; deg: number; to: number } | null {
  let best: { d: number; h: number } | null = null;
  for (let h = 0; h < 36; h++) {
    const d = Math.abs(((W.dir[h + 4] - W.dir[h] + 540) % 360) - 180);
    if (!best || d > best.d) best = { d, h: h + 2 };
  }
  return best && best.d >= 35 ? { h: best.h, deg: best.d, to: W.dir[best.h + 2] } : null;
}

/* --- wind fields for the map (u east, v north, metres per second) --- */
function baseWind(lat: number, lon: number, h: number): [number, number] {
  const t = (h / 24) * 2 * Math.PI;
  const a = (lat - 8) / 30;
  return [
    3.6 * (1 - a) + 1.6 * Math.sin(lon * 0.19 + lat * 0.11 + t * 0.55 + 0.6) + 1.1 * Math.sin(lat * 0.35 - lon * 0.09 + t * 0.8) - 0.4,
    1.9 * (1 - a) + 1.5 * Math.cos(lon * 0.15 - lat * 0.17 + t * 0.5) + 0.9 * Math.cos(lat * 0.31 + lon * 0.12 - t * 0.7),
  ];
}

function makeGrid(b: { lo0: number; lo1: number; la0: number; la1: number }, nx: number, ny: number, fn: (lat: number, lon: number) => [number, number]): WindGrid {
  const u = new Float32Array(nx * ny);
  const v = new Float32Array(nx * ny);
  const dx = (b.lo1 - b.lo0) / (nx - 1);
  const dy = (b.la1 - b.la0) / (ny - 1);
  for (let j = 0; j < ny; j++) {
    const lat = b.la1 - j * dy;
    for (let i = 0; i < nx; i++) {
      const w = fn(lat, b.lo0 + i * dx);
      u[j * nx + i] = w[0];
      v[j * nx + i] = w[1];
    }
  }
  return { lo0: b.lo0, la1: b.la1, dx, dy, nx, ny, u, v };
}

export function gridSample(g: WindGrid, lon: number, lat: number): [number, number] | null {
  const fx = (lon - g.lo0) / g.dx;
  const fy = (g.la1 - lat) / g.dy;
  if (fx < 0 || fy < 0 || fx > g.nx - 1 || fy > g.ny - 1) return null;
  const i = Math.floor(fx);
  const j = Math.floor(fy);
  const i1 = Math.min(i + 1, g.nx - 1);
  const j1 = Math.min(j + 1, g.ny - 1);
  const tx = fx - i;
  const ty = fy - j;
  const bl = (a: Float32Array) =>
    a[j * g.nx + i] * (1 - tx) * (1 - ty) +
    a[j * g.nx + i1] * tx * (1 - ty) +
    a[j1 * g.nx + i] * (1 - tx) * ty +
    a[j1 * g.nx + i1] * tx * ty;
  return [bl(g.u), bl(g.v)];
}

export const nationalGrid = (h: number): WindGrid =>
  makeGrid({ lo0: 66, lo1: 99, la0: 5, la1: 39 }, 34, 35, (la, lo) => baseWind(la, lo, h));

export const localGrid = (p: Place, W: Weather, h: number): WindGrid => {
  const w = windAt(W, h);
  const b0 = baseWind(p.lat, p.lon, h);
  return makeGrid({ lo0: p.lon - 1.8, lo1: p.lon + 1.8, la0: p.lat - 1.8, la1: p.lat + 1.8 }, 45, 45, (la, lo) => {
    const b = baseWind(la, lo, h);
    return [w.u + (b[0] - b0[0]) * 0.6, w.v + (b[1] - b0[1]) * 0.6];
  });
};

/* --- smoke plume: drifts with the forecast wind, widening downwind --- */
export function plumeOf(p: Place, W: Weather, H: number): Plume {
  H = clamp(H, 1, 72);
  const steps = Math.round(H * 2);
  const traj: Plume['traj'] = [[p.lon, p.lat, 0, 0]];
  let lon = p.lon;
  let lat = p.lat;
  let x = 0;
  for (let s = 1; s <= steps; s++) {
    const w = windAt(W, (s - 0.5) / 2);
    const dx = w.u * 1800;
    const dy = w.v * 1800;
    [lon, lat] = offsetLL(lon, lat, dx, dy);
    x += Math.hypot(dx, dy);
    traj.push([lon, lat, x, s / 2]);
  }
  const cL: LonLat[] = [];
  const cR: LonLat[] = [];
  const oL: LonLat[] = [];
  const oR: LonLat[] = [];
  for (let i = 0; i < traj.length; i++) {
    const a = traj[Math.max(0, i - 1)];
    const b = traj[Math.min(traj.length - 1, i + 1)];
    const dxm = (b[0] - a[0]) * mLon(p.lat);
    const dym = (b[1] - a[1]) * M_LAT;
    const L = Math.hypot(dxm, dym) || 1;
    const nx = -dym / L;
    const ny = dxm / L;
    const xx = traj[i][2];
    const hc = Math.max(250, (2 * 0.08 * xx) / Math.sqrt(1 + 1e-4 * xx)) + 150;
    const ho = hc + Math.min(xx * Math.tan(rad(8 + 0.5 * traj[i][3])), xx * 0.9);
    const P = traj[i];
    cL.push(offsetLL(P[0], P[1], nx * hc, ny * hc));
    cR.push(offsetLL(P[0], P[1], -nx * hc, -ny * hc));
    oL.push(offsetLL(P[0], P[1], nx * ho, ny * ho));
    oR.push(offsetLL(P[0], P[1], -nx * ho, -ny * ho));
  }
  const head = traj[traj.length - 1];
  const bearing = (Math.atan2((head[0] - p.lon) * mLon(p.lat), (head[1] - p.lat) * M_LAT) * 180) / Math.PI;
  return {
    traj,
    core: cL.concat(cR.reverse()),
    outer: oL.concat(oR.reverse()),
    reachKm: x / 1000,
    head: [head[0], head[1]],
    bearing: (bearing + 360) % 360,
    H,
  };
}

/* --- terrain along a line from the source --- */
export function terrainAlong(p: Place, dirTo: number, kmMax = 10, n = 41): Terrain {
  const r = R('terr:' + p.id + ':' + Math.round(dirTo / 15));
  const ctx = contextOf(p);
  const relief = { wildfire: 160, mining: 70, agricultural_burning: 15, gas_flare: 20, industrial: 25, unknown: 20 }[p.cls] || 30;
  const f1 = r.range(0, 6.28);
  const f2 = r.range(0, 6.28);
  const tr = r.range(-1, 1);
  const v = [Math.sin(rad(dirTo)), Math.cos(rad(dirTo))];
  const pts: Terrain['pts'] = [];
  let sum = 0;
  let up = 0;
  for (let i = 0; i < n; i++) {
    const km = (kmMax * i) / (n - 1);
    const el = ctx.elev + relief * (0.6 * Math.sin(km * 0.55 + f1) + 0.3 * Math.sin(km * 1.35 + f2) + (0.5 * tr * km) / kmMax);
    const ll = offsetLL(p.lon, p.lat, v[0] * km * 1000, v[1] * km * 1000);
    pts.push({ km, elev: el, lon: ll[0], lat: ll[1] });
    if (i) {
      const sl = (Math.atan((el - pts[i - 1].elev) / ((km - pts[i - 1].km) * 1000)) * 180) / Math.PI;
      sum += sl;
      up = Math.max(up, sl);
    }
  }
  return { pts, meanSlope: sum / (n - 1), maxUp: up };
}

/* --- fire spread outlook (INDICATIVE): vegetation fires only --- */
function ellipseRing(p: Place, dirTo: number, D: number, LB: number): LonLat[] {
  const a = D * 0.55;
  const b = a / LB;
  const c = D - a;
  const v = [Math.sin(rad(dirTo)), Math.cos(rad(dirTo))];
  const q = [Math.cos(rad(dirTo)), -Math.sin(rad(dirTo))];
  const ring: LonLat[] = [];
  for (let k = 0; k < 48; k++) {
    const th = (k / 48) * 2 * Math.PI;
    const X = c + a * Math.cos(th);
    const Y = b * Math.sin(th);
    ring.push(offsetLL(p.lon, p.lat, (X * v[0] + Y * q[0]) * 1000, (X * v[1] + Y * q[1]) * 1000));
  }
  return ring;
}

export function areaKm2(ring: LonLat[] | undefined): number {
  if (!ring || ring.length < 3) return 0;
  const lat0 = ring[0][1];
  const lon0 = ring[0][0];
  let s = 0;
  for (let i = 0; i < ring.length; i++) {
    const a = ring[i];
    const b = ring[(i + 1) % ring.length];
    s +=
      ((a[0] - lon0) * mLon(lat0) / 1000) * ((b[1] - lat0) * M_LAT / 1000) -
      ((b[0] - lon0) * mLon(lat0) / 1000) * ((a[1] - lat0) * M_LAT / 1000);
  }
  return Math.abs(s) / 2;
}

export function spreadOf(p: Place, ev: Event, W: Weather): SpreadOutlook | null {
  if (!isVeg(ev.cls)) return null;
  const hz = [1, 3, 6];
  const out: SpreadHorizon[] = [];
  hz.forEach((t) => {
    let su = 0;
    let sv = 0;
    let ff = 0;
    let n = 0;
    for (let s = 0; s <= t * 2; s++) {
      const w = windAt(W, s / 2);
      su += w.u;
      sv += w.v;
      ff += W.ffmc[Math.min(72, Math.round(s / 2))];
      n++;
    }
    su /= n;
    sv /= n;
    ff /= n;
    const spd = Math.hypot(su, sv);
    const from = (Math.atan2(-su, -sv) * 180) / Math.PI;
    const to = (from + 180 + 360) % 360;
    const kmh = spd * 3.6;
    const ter = terrainAlong(p, to, 8);
    const slopeF = ter.meanSlope > 0 ? 1 + clamp(0.06 * ter.meanSlope, 0, 1) : clamp(1 + 0.03 * ter.meanSlope, 0.6, 1);
    const isi = calcISI(ff, kmh);
    const rate = clamp(0.055 * isi * slopeF, 0.05, 6);
    const LB = clamp(0.936 * Math.exp(0.2566 * kmh) + 0.461 * Math.exp(-0.1548 * kmh) - 0.397, 1, 6);
    const D = rate * t;
    const ring = ellipseRing(p, to, D, LB);
    out.push({ t, isi, rate, rateLo: rate * 0.5, rateHi: rate * 1.8, D, dirTo: to, LB, slopeF, meanSlope: ter.meanSlope, area: areaKm2(ring), ring, cls: spreadClass(isi) });
  });
  const last = out[2];
  const unc = ellipseRing(p, last.dirTo, last.D * 1.6, Math.max(1, last.LB * 0.85));
  return { horizons: out, unc, cls: out[2].cls };
}

