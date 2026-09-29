/* =====================================================================
   Small helpers shared by every module: maths, colours, storage and the
   seeded random generator. A port of frontend/js/core.js plus the random
   helpers that lived in frontend/js/data/sample.js.
   ===================================================================== */
import type { ClassId } from './types';

export const clamp = (x: number, a: number, b: number): number => Math.max(a, Math.min(b, x));

/** Reads a custom property off :root, so colours follow the active theme. */
export const cv = (name: string): string =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** The theme colour of one source class, e.g. ccol('wildfire') -> #3FB68B */
export const ccol = (id: ClassId | string): string => cv('--c-' + id);

export interface ThemeKeys {
  tx: string;
  tx2: string;
  dim: string;
  line: string;
  line2: string;
  panel: string;
  raised: string;
  danger: string;
  warn: string;
  inv: string;
}

export const tk = (): ThemeKeys => ({
  tx: cv('--tx'),
  tx2: cv('--tx2'),
  dim: cv('--dim'),
  line: cv('--line'),
  line2: cv('--line2'),
  panel: cv('--panel'),
  raised: cv('--raised'),
  danger: cv('--danger'),
  warn: cv('--warn'),
  inv: cv('--inv'),
});

export const store = {
  get(k: string): string | null {
    try {
      return localStorage.getItem(k);
    } catch {
      return null;
    }
  },
  set(k: string, v: string): void {
    try {
      localStorage.setItem(k, v);
    } catch {
      /* private mode, quota, or disabled storage: the app still works */
    }
  },
};

/** #rrggbb + alpha -> rgba(). Anything else is passed through unchanged. */
export const alpha = (hex: string, a: number): string => {
  const h = (hex || '').trim();
  if (h[0] !== '#' || h.length !== 7) return hex;
  const n = parseInt(h.slice(1), 16);
  return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`;
};

export const hexMix = (a: string, b: string, t: number): string => {
  const A = parseInt(a.slice(1), 16);
  const B = parseInt(b.slice(1), 16);
  const c = (i: number) => Math.round(((A >> i) & 255) * (1 - t) + ((B >> i) & 255) * t);
  return `rgb(${c(16)},${c(8)},${c(0)})`;
};

/** "gas flare" -> "Gas flare" (lowercase first letter only) */
export const lc = (s: string): string => s.charAt(0).toLowerCase() + s.slice(1);

/* ---- seeded random: every number in the sample data is reproducible ---- */
export function mulberry32(a: number): () => number {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function hstr(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export interface Rand {
  (): number;
  range(a: number, b: number): number;
  int(a: number, b: number): number;
  pick<T>(a: readonly T[]): T;
  norm(): number;
}

export function R(seed: string | number): Rand {
  const r = mulberry32(typeof seed === 'string' ? hstr(seed) : seed) as Rand;
  r.range = (a: number, b: number) => a + (b - a) * r();
  r.int = (a: number, b: number) => Math.floor(a + (b - a + 1) * r());
  r.pick = <T,>(a: readonly T[]) => a[Math.floor(r() * a.length)];
  r.norm = () => {
    let u = 0;
    let v = 0;
    while (!u) u = r();
    while (!v) v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  return r;
}

export const median = (a: readonly number[]): number => {
  if (!a.length) return NaN;
  const s = [...a].sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

export const quant = (a: readonly number[], q: number): number => {
  const s = [...a].sort((x, y) => x - y);
  return s[Math.min(s.length - 1, Math.floor(q * s.length))];
};
