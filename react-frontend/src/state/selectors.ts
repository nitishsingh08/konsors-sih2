/* =====================================================================
   Pure selectors over the sample data, plus the hooks that bind them to
   the current filters. These replace the global `S` object and the
   `passes` / `winEvents` / `groups` / `pickEvent` helpers of ui.js.
   ===================================================================== */
import { useMemo } from 'react';
import { DAY } from '../lib/time';
import { EVENTS, EVP, PLACES, PMAP } from '../data/places';
import type { Event, Place } from '../lib/types';
import { useFilters, type Filters } from './useFilters';

export interface Group {
  p: Place;
  evs: Event[];
  top: Event;
}

/** abnormal first, then needs review, then everything else */
export const sev = (e: Event): number => (e.status === 'abnormal' ? 3 : e.review ? 2 : 1);

export function passes(f: Filters, e: Event): boolean {
  if (!f.cls.has(e.cls)) return false;
  if (f.status === 'review') {
    if (!e.review) return false;
  } else if (f.status !== 'all' && e.status !== f.status) return false;
  if (f.sensor !== 'all' && !e.sensor.startsWith(f.sensor)) return false;
  return e.conf >= f.minConf;
}

export function winEvents(f: Filters, off = 0): Event[] {
  const hi = f.asOf - off * f.win * DAY;
  const lo = hi - f.win * DAY;
  return EVENTS.filter((e) => e.t > lo && e.t <= hi && passes(f, e));
}

export function groups(evs: Event[]): Group[] {
  const m = new Map<string, { p: Place; evs: Event[]; top?: Event }>();
  evs.forEach((e) => {
    let g = m.get(e.pid);
    if (!g) {
      g = { p: PMAP[e.pid], evs: [] };
      m.set(e.pid, g);
    }
    g.evs.push(e);
  });
  return [...m.values()].map((g) => {
    g.top = g.evs.slice().sort((a, b) => sev(b) - sev(a) || b.t - a.t)[0];
    return g as Group;
  });
}

const sortTop = (a: Group, b: Group) => sev(b.top) - sev(a.top) || b.top.t - a.top.t;

export interface Current {
  evs: Event[];
  groups: Group[];
  queue: Group[];
}

export function useCurrent(): Current {
  const [f] = useFilters();
  return useMemo(() => {
    const evs = winEvents(f);
    const gs = groups(evs);
    return { evs, groups: gs, queue: gs.slice().sort(sortTop) };
  }, [f]);
}

/** the event a place report should show: the pinned one, else the most recent in window */
export function pickEvent(p: Place, f: Filters, pinnedId: string | null): Event {
  const evs = EVP[p.id] || [];
  if (pinnedId) {
    const e = evs.find((x) => x.id === pinnedId);
    if (e) return e;
  }
  const inW = evs.filter((e) => e.t <= f.asOf && e.t > f.asOf - f.win * DAY);
  const pool = inW.length ? inW : evs.filter((e) => e.t <= f.asOf);
  return (pool.length ? pool : evs).slice().sort((a, b) => sev(b) - sev(a) || b.t - a.t)[0];
}

export function searchPlaces(q: string): Place[] {
  const s = q.trim().toLowerCase();
  if (!s) return [];
  return PLACES.filter((p) => (p.name + ' ' + p.district + ' ' + p.state + ' ' + p.code).toLowerCase().includes(s))
    .sort((a, b) => Number(b.kind === 'site') - Number(a.kind === 'site'))
    .slice(0, 8);
}
