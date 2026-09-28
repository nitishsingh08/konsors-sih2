/* =====================================================================
   Time helpers. All sample data is anchored to 20 Sep 2026, and every
   displayed time is in IST regardless of the viewer's own timezone, which
   is what the original build did by shifting a fixed offset.
   ===================================================================== */

export const DAY = 864e5;
export const HOUR = 36e5;
export const IST = 5.5 * HOUR;

export const TODAY = Date.UTC(2026, 8, 20, 9, 0, 0);
export const START = TODAY - 364 * DAY;

export const MONTHS = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
];

export const DOW = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

export interface DateParts {
  y: number;
  m: number;
  d: number;
  h: number;
  mi: number;
}

export const pad = (n: number): string => String(n).padStart(2, '0');

export const dparts = (t: number): DateParts => {
  const d = new Date(t + IST);
  return { y: d.getUTCFullYear(), m: d.getUTCMonth(), d: d.getUTCDate(), h: d.getUTCHours(), mi: d.getUTCMinutes() };
};

export const fmtD = (t: number): string => {
  const p = dparts(t);
  return `${p.d} ${MONTHS[p.m]} ${p.y}`;
};

export const fmtDT = (t: number): string => {
  const p = dparts(t);
  return `${p.d} ${MONTHS[p.m]} ${p.y}, ${pad(p.h)}:${pad(p.mi)} IST`;
};

export const fmtS = (t: number): string => {
  const p = dparts(t);
  return `${p.d} ${MONTHS[p.m]}`;
};

export const iso = (t: number): string => {
  const p = dparts(t);
  return `${p.y}-${pad(p.m + 1)}-${pad(p.d)}`;
};

export const ago = (t: number): string => {
  const h = (TODAY - t) / HOUR;
  if (h < 1) return 'under 1 h ago';
  if (h < 24) return `${Math.round(h)} h ago`;
  return `${Math.round(h / 24)} d ago`;
};

export const nf = (n: number): string => Math.round(n).toLocaleString('en-IN');

/** "Tue 6 pm", h hours after t0 */
export function hourLabel(t0: number, h: number): string {
  const d = new Date(t0 + h * HOUR + IST);
  const hh = d.getUTCHours();
  return `${DOW[d.getUTCDay()]} ${hh % 12 || 12} ${hh >= 12 ? 'pm' : 'am'}`;
}
