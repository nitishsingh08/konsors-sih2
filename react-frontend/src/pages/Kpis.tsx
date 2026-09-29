import { useState } from 'react';
import { DAY, nf } from '../lib/time';
import { clamp } from '../lib/core';
import { PMAP } from '../data/places';
import type { Event } from '../lib/types';
import { winEvents } from '../state/selectors';
import { useFilters } from '../state/useFilters';
import { Spark, TipButton } from '../components/ui/Primitives';
import { Icon } from '../components/ui/Icon';

interface KpiDef {
  label: string;
  value: number;
  prev: number;
  series: number[];
  tip: string;
  alarm?: boolean;
}

export function Kpis({
  collapsed,
  onToggle,
}: {
  collapsed?: boolean;
  onToggle?: () => void;
} = {}) {
  const [internalCollapsed, setInternalCollapsed] = useState(false);
  const isCollapsed = collapsed !== undefined ? collapsed : internalCollapsed;
  const toggle = onToggle || (() => setInternalCollapsed((c) => !c));

  const [f] = useFilters();
  const cur = winEvents(f, 0);
  const prev = winEvents(f, 1);
  const nb = Math.min(24, Math.max(7, f.win > 30 ? 18 : f.win));

  const agg = (evs: typeof cur) => ({
    det: evs.reduce((a, e) => a + e.nDet, 0),
    ev: evs.length,
    abn: new Set(evs.filter((e) => e.status === 'abnormal').map((e) => e.pid)).size,
    sites: new Set(evs.filter((e) => PMAP[e.pid].kind === 'site').map((e) => e.pid)).size,
    rev: evs.filter((e) => e.review).length,
  });
  const lo = f.asOf - f.win * DAY;
  const bw = (f.win * DAY) / nb;
  const bucket = (evs: Event[]) => {
    const out: Event[][] = Array.from({ length: nb }, () => []);
    evs.forEach((e) => out[clamp(Math.floor((e.t - lo) / bw), 0, nb - 1)].push(e));
    return out;
  };
  const series = (fn: (bucket: Event[]) => number) => bucket(cur).map(fn);
  const count = (pred: (e: Event) => boolean) => series((b) => b.filter(pred).length);

  const a = agg(cur);
  const b = agg(prev);
  const defs: KpiDef[] = [
    {
      label: 'Detections',
      value: a.det,
      prev: b.det,
      series: series((x) => x.reduce((s, e) => s + e.nDet, 0)),
      tip: 'Raw satellite hot pixels grouped into events in this window.',
    },
    {
      label: 'Events',
      value: a.ev,
      prev: b.ev,
      series: count(() => true),
      tip: 'Groups of nearby detections treated as one occurrence.',
    },
    {
      label: 'Abnormal places',
      value: a.abn,
      prev: b.abn,
      series: count((e) => e.status === 'abnormal'),
      tip: 'Places behaving unusually compared with their own history.',
      alarm: true,
    },
    {
      label: 'Active sites',
      value: a.sites,
      prev: b.sites,
      series: series((x) => new Set(x.filter((e) => PMAP[e.pid].kind === 'site').map((e) => e.pid)).size),
      tip: 'Known industrial or mining sites with at least one event in this window.',
    },
    {
      label: 'Needs review',
      value: a.rev,
      prev: b.rev,
      series: count((e) => e.review),
      tip: 'Events where confidence is below 60%, so a person should check.',
    },
  ];

  return (
    <div className={'kpis-box' + (isCollapsed ? ' collapsed' : '')}>
      <div className="kpis-header">
        <div className="kpis-header-left">
          <span className="kpis-tag">Summary metrics</span>
          {isCollapsed && (
            <div className="kpis-compact-stats">
              <span className="kpis-stat-pill">
                <b>{nf(a.det)}</b> detections
              </span>
              <span className="dot">·</span>
              <span className="kpis-stat-pill">
                <b>{nf(a.ev)}</b> events
              </span>
              <span className="dot">·</span>
              <span className={'kpis-stat-pill' + (a.abn > 0 ? ' alarm' : '')}>
                <b>{a.abn}</b> abnormal
              </span>
              <span className="dot">·</span>
              <span className="kpis-stat-pill">
                <b>{a.sites}</b> active sites
              </span>
              <span className="dot">·</span>
              <span className="kpis-stat-pill">
                <b>{a.rev}</b> needs review
              </span>
            </div>
          )}
        </div>
        <button
          type="button"
          className="btn sm kpis-toggle-btn"
          onClick={toggle}
          aria-label={isCollapsed ? 'Expand metrics window' : 'Collapse metrics window'}
          title={isCollapsed ? 'Expand metrics' : 'Collapse metrics'}
        >
          <Icon name={isCollapsed ? 'chevron-down' : 'chevron-up'} size={13} />
          <span>{isCollapsed ? 'Expand' : 'Collapse'}</span>
        </button>
      </div>

      {!isCollapsed && (
        <div className="kpis">
          {defs.map((d) => {
            const diff = d.prev ? Math.round(((d.value - d.prev) / d.prev) * 100) : null;
            const dt = diff == null ? (d.value ? 'new' : 'no change') : (diff > 0 ? '+' : '') + diff + '%';
            const alarm = d.alarm && d.value > 0;
            return (
              <div className={'kpi' + (alarm ? ' alarm' : '')} key={d.label}>
                <div className="l">
                  {d.label}
                  <TipButton text={d.tip} label={'About ' + d.label} />
                </div>
                <div className="v num">{nf(d.value)}</div>
                <div className="d">
                  <b className="num">{dt}</b>
                  <span>vs previous {f.win} d</span>
                </div>
                <div style={{ color: 'var(--dim)', marginTop: 4 }}>
                  <Spark values={d.series} w={120} h={24} color={alarm ? 'var(--danger)' : 'currentColor'} />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
