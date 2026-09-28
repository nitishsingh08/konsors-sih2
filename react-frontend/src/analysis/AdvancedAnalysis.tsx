/* =====================================================================
   The advanced analysis workspace: opens over the place report with its
   own map, a scrubable 72-hour forecast timeline, and the section panel.
   Port of frontend/js/analysis/analysis.js and the ANAV section list.
   ===================================================================== */
import { useEffect, useMemo, useRef, useState } from 'react';
import L from 'leaflet';
import { alpha, ccol, tk } from '../lib/core';
import { fmtDT, hourLabel } from '../lib/time';
import { PMAP } from '../data/places';
import { featuresOf, footprintsFor } from '../data/features';
import { outlookLine } from '../data/analysis';
import { compassShort, plumeOf, spreadOf, terrainAlong, weatherOf, windShift } from '../data/weather';
import { pickEvent } from '../state/selectors';
import { useFilters } from '../state/useFilters';
import { usePrefersReducedMotion } from '../hooks/usePrefersReducedMotion';
import { useFiles } from '../lib/files';
import { AnalysisMap, type AnalysisMapHandle, type ShowFlags } from '../map/AnalysisMap';
import type { BaseMode } from '../map/leafletBase';
import { windGradientCss, type WindStyle } from '../map/WindLayer';
import { Chart, axisBox, tooltipBox } from '../components/ui/Chart';
import { Icon } from '../components/ui/Icon';
import { EventChips, Seg } from '../components/ui/Primitives';
import { useTip } from '../components/ui/Tooltip';
import { WindTab } from './tabs/WindTab';
import { GroupTab } from './tabs/GroupTab';
import { AirTab } from './tabs/AirTab';
import { ExplorerTab, initialExplorer, type ExplorerState } from './tabs/ExplorerTab';
import { ExposureTab } from './tabs/ExposureTab';
import { ReviewTab, SimilarTab } from './tabs/SimilarTab';
import { FGROUP_COUNT } from '../data/features';
import type { Event, FamilyId, FeatureVector, Plume, Place, SpreadOutlook, Terrain, Weather } from '../lib/types';

export const TABS = [
  'wind', 'thermal', 'temporal', 'spread', 'terrain', 'fuel', 'industry', 'weather', 'air', 'quality', 'explorer', 'exposure', 'similar', 'review',
] as const;
export type TabId = (typeof TABS)[number];

export const ANAV: { group: string; items: { id: TabId; label: string }[] }[] = [
  { group: 'Outlook', items: [{ id: 'wind', label: 'Wind and spread' }] },
  {
    group: 'What we measured',
    items: [
      { id: 'thermal', label: 'Heat' },
      { id: 'temporal', label: 'Timing' },
      { id: 'spread', label: 'Movement and growth' },
      { id: 'terrain', label: 'Terrain' },
      { id: 'fuel', label: 'Fuel and land cover' },
      { id: 'industry', label: 'Industry nearby' },
      { id: 'weather', label: 'Weather and fire danger' },
      { id: 'air', label: 'Smoke, radar and air' },
      { id: 'quality', label: 'Data quality' },
    ],
  },
  {
    group: 'Tools',
    items: [
      { id: 'explorer', label: 'All 141 features' },
      { id: 'exposure', label: 'Who is in the way' },
      { id: 'similar', label: 'Similar events' },
      { id: 'review', label: 'Analyst review' },
    ],
  },
];

export const ATITLES: Record<TabId, string> = Object.fromEntries(ANAV.flatMap((g) => g.items.map((i) => [i.id, i.label]))) as Record<TabId, string>;

const TOOLS: [keyof ShowFlags, string][] = [
  ['wind', 'Wind'],
  ['plume', 'Smoke plume'],
  ['spread', 'Fire spread'],
  ['foot', 'Observed footprint'],
  ['rings', 'Distance rings'],
  ['terr', 'Terrain line'],
  ['near', 'Other heat nearby'],
];

export function AdvancedAnalysis({
  placeId,
  tab,
  onTab,
  onClose,
  base,
  onBaseFallback,
}: {
  placeId: string;
  tab: string;
  onTab: (t: string) => void;
  onClose: () => void;
  base: BaseMode;
  onBaseFallback: () => void;
}) {
  const [filters] = useFilters();
  const reduced = usePrefersReducedMotion();
  const { saveFile } = useFiles();
  const mapRef = useRef<AnalysisMapHandle>(null);

  const [pid, setPid] = useState(placeId);
  const [h, setH] = useState(6);
  const [style, setStyle] = useState<WindStyle>(() => (reduced ? 'arrows' : 'flow'));
  const [show, setShow] = useState<ShowFlags>({ wind: true, plume: true, spread: true, rings: true, foot: true, terr: true, near: false });
  const [op, setOp] = useState(0.8);
  const [measure, setMeasure] = useState(false);
  const [mpts, setMpts] = useState<L.LatLng[]>([]);
  const [fpIdx, setFpIdx] = useState(-1);
  const [playing, setPlaying] = useState(false);
  const [replayLabel, setReplayLabel] = useState('');
  const [ex, setEx] = useState<ExplorerState>(initialExplorer);

  useEffect(() => setPid(placeId), [placeId]);
  useEffect(() => setStyle(reduced ? 'arrows' : 'flow'), [reduced]);

  const p = PMAP[pid];
  const ev = pickEvent(p, filters, null);
  const W = useMemo(() => weatherOf(p, ev), [p, ev]);
  const F: FeatureVector = useMemo(() => featuresOf(p, ev), [p, ev]);
  const plume = useMemo(() => plumeOf(p, W, Math.max(1, h)), [p, W, h]);
  const spread = useMemo(() => spreadOf(p, ev, W), [p, ev, W]);
  const footprints = useMemo(() => footprintsFor(p, ev), [p, ev]);
  const terrain = useMemo(
    () => (show.terr ? terrainAlong(p, spread ? spread.horizons[1].dirTo : plume.bearing, 10) : null),
    [show.terr, p, spread, plume.bearing],
  );

  const active = (TABS as readonly string[]).includes(tab) ? (tab as TabId) : 'wind';

  /* forecast playback */
  const hRef = useRef(h);
  hRef.current = h;
  useEffect(() => {
    if (!playing) return;
    const t = window.setInterval(() => setH((x) => (x >= 72 ? 72 : x + 1)), 260);
    return () => window.clearInterval(t);
  }, [playing]);
  useEffect(() => {
    if (playing && h >= 72) setPlaying(false);
  }, [h, playing]);

  /* footprint replay */
  const fpTimer = useRef<number | undefined>(undefined);
  const replay = () => {
    window.clearInterval(fpTimer.current);
    setShow((s) => ({ ...s, foot: true }));
    setFpIdx(0);
    const t = window.setInterval(() => {
      setFpIdx((i) => {
        if (i + 1 >= footprints.length) {
          window.clearInterval(fpTimer.current);
          setFpIdx(-1);
          return -1;
        }
        setReplayLabel(`Pass ${i + 2} of ${footprints.length}, ${fmtDT(footprints[i + 1].t)}`);
        return i + 1;
      });
    }, 650);
    fpTimer.current = t;
  };
  useEffect(() => () => window.clearInterval(fpTimer.current), []);

  const saveJson = () => {
    saveFile(
      `agni-netra-${ev.id}-outlook.json`,
      JSON.stringify(
        {
          event: ev.id,
          place: p.name,
          as_of: fmtDT(ev.t),
          sample_data: true,
          wind: { hourly_speed_ms: W.spd.map((v) => +v.toFixed(2)), hourly_from_deg: W.dir.map((v) => +v.toFixed(0)) },
          plume_6h_reach_km: +plumeOf(p, W, 6).reachKm.toFixed(2),
          spread: spread
            ? spread.horizons.map((z) => ({ hours: z.t, reach_km: +z.D.toFixed(2), rate_kmh: [+z.rateLo.toFixed(2), +z.rateHi.toFixed(2)], class: z.cls }))
            : null,
        },
        null,
        1,
      ),
      'JSON',
    );
  };

  const timeline = useMemo(() => {
    const T = tk();
    const hrs = W.spd.map((_, i) => i);
    const sh = windShift(W);
    return {
      grid: { left: 42, right: 42, top: 14, bottom: 20 },
      tooltip: {
        ...tooltipBox(),
        trigger: 'axis',
        formatter: (q: { dataIndex: number }) => {
          const i = q.dataIndex;
          return `<b>${hourLabel(W.t0, i)}</b><br>Wind ${(W.spd[i] * 3.6).toFixed(0)} km/h from the ${compassShort(W.dir[i])}, gusts ${(
            W.gust[i] * 3.6
          ).toFixed(0)}<br>Humidity ${W.rh[i].toFixed(0)}%, ${W.temp[i].toFixed(0)}°C${W.rain[i] ? `<br>Rain ${W.rain[i].toFixed(1)} mm` : ''}`;
        },
      },
      xAxis: {
        type: 'category', data: hrs, boundaryGap: false, ...axisBox(), splitLine: { show: false },
        axisLabel: { color: T.dim, fontSize: 10, interval: 0, formatter: (v: number) => (+v % 12 === 0 ? hourLabel(W.t0, +v) : '') },
      },
      yAxis: [
        { type: 'value', name: 'km/h', nameTextStyle: { color: T.dim, fontSize: 10, align: 'left' }, min: 0, ...axisBox() },
        { type: 'value', min: 0, max: 100, name: '%', nameTextStyle: { color: T.dim, fontSize: 10, align: 'right' }, ...axisBox(), splitLine: { show: false } },
      ],
      series: [
        {
          name: 'Wind', type: 'line', showSymbol: false, data: W.spd.map((v) => +(v * 3.6).toFixed(1)),
          lineStyle: { color: T.tx, width: 2 },
          markLine: { silent: true, symbol: 'none', label: { show: false }, lineStyle: { color: T.tx, width: 1.2 }, data: [{ xAxis: h }] },
          markArea: sh ? { silent: true, itemStyle: { color: alpha(T.warn, 0.13) }, data: [[{ xAxis: Math.max(0, sh.h - 2) }, { xAxis: sh.h + 3 }]] } : undefined,
        },
        { name: 'Gusts', type: 'line', showSymbol: false, data: W.gust.map((v) => +(v * 3.6).toFixed(1)), lineStyle: { color: T.dim, width: 1, type: 'dashed' } },
        { name: 'Humidity', type: 'line', yAxisIndex: 1, showSymbol: false, data: W.rh.map((v) => +v.toFixed(0)), lineStyle: { color: ccol('industrial'), width: 1.4 } },
        {
          name: 'Direction', type: 'scatter', symbol: 'path://M0,-7 L4.5,5 L0,2.5 L-4.5,5 Z', symbolSize: 11, symbolRotate: 0,
          itemStyle: { color: T.tx2 },
          data: hrs.filter((i) => i % 6 === 0).map((i) => ({ value: [i, +(W.spd[i] * 3.6).toFixed(1) + 4], symbolRotate: -((W.dir[i] + 180) % 360) })),
          tooltip: { show: false },
        },
      ],
    };
  }, [W, h]);

  const col = ccol(ev.cls);

  return (
    <div className="adv" role="dialog" aria-label="Advanced analysis">
      <header className="adv-top">
        <button className="btn sm" onClick={onClose}>
          <Icon name="back" size={14} />
          Back to the report
        </button>
        <div style={{ minWidth: 0 }}>
          <h2>{p.name}</h2>
          <div className="dim small">
            {p.district}, {p.state}. Conditions as of {fmtDT(ev.t)}.
          </div>
        </div>
        <span className="row wrap" style={{ marginLeft: 6 }}>
          <EventChips ev={ev} />
        </span>
        <span style={{ flex: 1 }} />
        <span className="chip prop" {...useTip('Wind, smoke and spread here are sample values shaped like real model output. Your backend will supply the real ones.')}>
          Sample outlook
        </span>
      </header>

      <div className="adv-body">
        <nav className="adv-nav" aria-label="Analysis sections">
          {ANAV.map((g) => (
            <div className="adv-navg" key={g.group}>
              <div className="adv-navh">{g.group}</div>
              {g.items.map((it) => (
                <button key={it.id} aria-current={active === it.id} onClick={() => onTab(it.id)}>
                  {it.label}
                  {FGROUP_COUNT[it.id as FamilyId] ? <span className="n">{FGROUP_COUNT[it.id as FamilyId]}</span> : null}
                </button>
              ))}
            </div>
          ))}
        </nav>

        <section className="adv-main">
          <div className="adv-mapwrap">
            <AnalysisMap
              ref={mapRef}
              p={p}
              ev={ev}
              W={W}
              h={h}
              plume={plume}
              spread={spread}
              terrain={terrain}
              footprints={footprints}
              fpIdx={fpIdx}
              show={show}
              op={op}
              style={style}
              measure={measure}
              mpts={mpts}
              base={base}
              onMeasurePoint={(at) => setMpts((m) => [...m, at])}
              onTilesFail={onBaseFallback}
            />
            <div className="ov tl advtools">
              {TOOLS.map(([k, l]) => (
                <button key={k} className="fchip" aria-pressed={show[k]} onClick={() => setShow((s) => ({ ...s, [k]: !s[k] }))}>
                  {l}
                </button>
              ))}
            </div>
            <div className="ov tr advtools2">
              <Seg
                value={style}
                onChange={setStyle}
                label="Wind style"
                options={[
                  { value: 'flow' as WindStyle, label: 'Flow' },
                  { value: 'arrows' as WindStyle, label: 'Arrows' },
                ]}
              />
              <button
                className="btn sm"
                aria-pressed={measure}
                onClick={() => {
                  setMeasure((m) => {
                    if (m) setMpts([]);
                    return !m;
                  });
                }}
              >
                <Icon name="draw" size={14} />
                Measure
              </button>
            </div>
            <div className="ov advkey">
              <div className="row small">
                <span>Wind speed</span>
                <span className="windbar" style={{ background: windGradientCss() }} />
                <span className="dim">0 to 12+ m/s</span>
              </div>
              <div className="row small" style={{ gap: 12, flexWrap: 'wrap' }}>
                <span className="row">
                  <i className="sw" style={{ background: alpha(col, 0.5) }} />
                  Smoke, core
                </span>
                <span className="row">
                  <i className="sw" style={{ background: alpha(col, 0.12), border: `1px dashed ${col}` }} />
                  Where it might go instead
                </span>
                {spread && (
                  <span className="row">
                    <i className="sw" style={{ border: '1.5px solid var(--tx)' }} />
                    Fire spread, +1, +3, +6 h
                  </span>
                )}
              </div>
            </div>
            <div className="ov advop">
              <label className="row small" htmlFor="advop">
                Overlay strength
                <input id="advop" type="range" min={25} max={100} value={Math.round(op * 100)} onChange={(e) => setOp(Number(e.target.value) / 100)} />
              </label>
            </div>
          </div>

          <div className="adv-time">
            <div className="row between">
              <div className="row">
                <button className="btn ic sm" aria-label="Play forecast" onClick={() => setPlaying((v) => !v)}>
                  <Icon name={playing ? 'pause' : 'play'} size={14} />
                </button>
                <b className="num">{h === 0 ? 'Now' : `In ${h} h, ${hourLabel(W.t0, h)}`}</b>
              </div>
              <span className="dim small">Wind, gusts and humidity, next 72 hours</span>
            </div>
            <Chart option={timeline} height={118} />
            <div className="advslide">
              <input type="range" min={0} max={72} step={1} value={h} aria-label="Forecast hour" onChange={(e) => setH(Number(e.target.value))} />
            </div>
          </div>
        </section>

        <aside className="adv-panel" aria-live="polite">
          <div className="adv-ph">
            <h3>{ATITLES[active]}</h3>
          </div>
          <Panel
            tab={active}
            p={p}
            ev={ev}
            F={F}
            W={W}
            h={h}
            plume={plume}
            spread={spread}
            terrain={terrain}
            headline={outlookLine(p, ev, W)}
            replayLabel={replayLabel}
            onFit={() => mapRef.current?.fit()}
            onReplay={replay}
            onJson={saveJson}
            onHoverElev={(at) => mapRef.current?.setElevMarker(at)}
            ex={ex}
            setEx={setEx}
            onGo={setPid}
          />
        </aside>
      </div>
    </div>
  );
}

function Panel({
  tab,
  p,
  ev,
  F,
  W,
  h,
  plume,
  spread,
  terrain,
  headline,
  replayLabel,
  onFit,
  onReplay,
  onJson,
  onHoverElev,
  ex,
  setEx,
  onGo,
}: {
  tab: TabId;
  p: Place;
  ev: Event;
  F: FeatureVector;
  W: Weather;
  h: number;
  plume: Plume;
  spread: SpreadOutlook | null;
  terrain: Terrain | null;
  headline: string;
  replayLabel: string;
  onFit: () => void;
  onReplay: () => void;
  onJson: () => void;
  onHoverElev: (at: [number, number] | null) => void;
  ex: ExplorerState;
  setEx: (s: ExplorerState) => void;
  onGo: (placeId: string) => void;
}) {
  if (tab === 'wind') {
    return <WindTab ev={ev} W={W} h={h} headline={headline} plume={plume} spread={spread} terrain={terrain} replayLabel={replayLabel} onFit={onFit} onReplay={onReplay} onJson={onJson} onHoverElev={onHoverElev} />;
  }
  if (tab === 'air') return <AirTab ev={ev} F={F} />;
  if (tab === 'explorer') return <ExplorerTab ev={ev} F={F} state={ex} onChange={setEx} />;
  if (tab === 'exposure') return <ExposureTab ev={ev} plume={plume} spread={spread} />;
  if (tab === 'similar') return <SimilarTab ev={ev} onGo={onGo} />;
  if (tab === 'review') return <ReviewTab ev={ev} />;
  return <GroupTab gid={tab as FamilyId} p={p} ev={ev} F={F} />;
}
