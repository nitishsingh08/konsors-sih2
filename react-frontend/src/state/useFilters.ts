import { useCallback, useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';
import { clamp } from '../lib/core';
import { START, TODAY, iso } from '../lib/time';
import { CLS, CLSMAP } from '../data/classes';
import type { ClassId, StatusId } from '../lib/types';

export const WINDOWS = [7, 30, 90, 365] as const;
export type Window = (typeof WINDOWS)[number];

export type StatusFilter = StatusId | 'all' | 'review';
export type SensorFilter = 'all' | 'VIIRS' | 'MODIS';

export interface Filters {
  asOf: number;
  win: Window;
  status: StatusFilter;
  cls: Set<ClassId>;
  minConf: number;
  sensor: SensorFilter;
}

const ALL_CLS = new Set<ClassId>(CLS.map((c) => c.id));

/**
 * The shareable half of the command-page state. It lives in the URL query so a
 * link carries the date window, filters and confidence floor, exactly as the
 * original build did with the hash.
 */
export function useFilters(): [Filters, (patch: Partial<Filters>) => void] {
  const [params, setParams] = useSearchParams();

  const filters = useMemo<Filters>(() => {
    const winRaw = Number(params.get('w'));
    const win = (WINDOWS as readonly number[]).includes(winRaw) ? (winRaw as Window) : 90;
    const aRaw = params.get('a');
    const aT = aRaw ? Date.parse(aRaw + 'T09:00:00Z') : NaN;
    const asOf = isNaN(aT) ? TODAY : clamp(aT, START, TODAY);
    const status = (params.get('s') as StatusFilter) || 'all';
    const c = params.get('c');
    const listed = c ? c.split(',').filter((x) => CLSMAP[x as ClassId]) : [];
    const cls = listed.length ? new Set<ClassId>(listed as ClassId[]) : new Set(ALL_CLS);
    const m = Number(params.get('m'));
    const sensor = (params.get('n') as SensorFilter) || 'all';
    return {
      asOf,
      win,
      status: (['all', 'review', 'abnormal', 'routine', 'baseline_building', 'not_applicable'] as string[]).includes(status) ? status : 'all',
      cls,
      minConf: isNaN(m) ? 0 : clamp(m / 100, 0, 0.9),
      sensor: (['all', 'VIIRS', 'MODIS'] as string[]).includes(sensor) ? sensor : 'all',
    };
  }, [params]);

  const set = useCallback(
    (patch: Partial<Filters>) => {
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          const w = patch.win ?? filters.win;
          next.set('w', String(w));
          next.set('a', iso(patch.asOf ?? filters.asOf));
          const status = patch.status ?? filters.status;
          if (status !== 'all') next.set('s', status);
          else next.delete('s');
          const cls = patch.cls ?? filters.cls;
          if (cls.size < ALL_CLS.size) next.set('c', [...cls].join(','));
          else next.delete('c');
          const m = patch.minConf ?? filters.minConf;
          if (m) next.set('m', String(Math.round(m * 100)));
          else next.delete('m');
          const sensor = patch.sensor ?? filters.sensor;
          if (sensor !== 'all') next.set('n', sensor);
          else next.delete('n');
          return next;
        },
        { replace: true },
      );
    },
    [filters, setParams],
  );

  return [filters, set];
}
