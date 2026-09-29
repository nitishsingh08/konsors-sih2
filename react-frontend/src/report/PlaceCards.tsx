/* =====================================================================
   The chart-heavy cards of the place report: the year of heat, the
   behaviour fingerprint, footprint growth, surroundings, a before/after
   image slider, ground conditions, nearby heat and the event history.
   Ported from frontend/js/pages.js.
   ===================================================================== */
import { useEffect, useMemo, useRef, useState } from 'react';
import { alpha, ccol, hexMix, R, tk } from '../lib/core';
import { DAY, MONTHS, START, TODAY, dparts, fmtD, fmtDT, fmtS, iso } from '../lib/time';
import { Chart, axisBox, tooltipBox } from '../components/ui/Chart';
import { Card, Info, LegendRow, ReviewChip, Stat, StatusChip, Swatch } from '../components/ui/Primitives';
import { ClassBadge, Glyph, glyphPath } from '../components/ui/Glyph';
import { outlookLine } from '../data/analysis';
import { EVP, hoursOf, nearbyOf } from '../data/places';
import { weatherOf } from '../data/weather';
import type { Event, Observation, Place, PlaceContext, PlaceSeries, PlaceStats } from '../lib/types';

const pl = (n: number, w: string) => (n === 1 ? w : w + 's');
const km = (v: number) => (v > 10 ? 'over 10 km' : v.toFixed(1) + ' km');

/* ---------- year of heat ---------- */
export function TimelineCard({ p, ev, ser }: { p: Place; ev: Event; ser: PlaceSeries }) {
  const T = tk();
  const col = ccol(p.cls);
  const obs = ser.obs;

  const option = useMemo(() => {
    if (!obs.length) return null;
    const site = p.kind === 'site';
    const b = ser.base;
    const useB = site && !!b && b.n >= 20;
    const abn = (o: Observation) => o.z != null && o.z > 3.5;
    const data = obs.map((o) => ({
      value: [o.t, +o.frp.toFixed(2)],
      symbol: o.sensor.startsWith('MODIS') ? 'diamond' : 'circle',
      symbolSize: abn(o) ? 11 : 6.5,
      itemStyle: { color: abn(o) ? T.danger : col, borderColor: T.panel, borderWidth: 1 },
      o,
    }));
    const mkA: unknown[] = [];
    const mkL: unknown[] = [];
    if (useB && b) {
      mkA.push([{ yAxis: b.lo, itemStyle: { color: alpha(col, 0.11) } }, { yAxis: b.hi }]);
      mkL.push(
        { yAxis: b.med, lineStyle: { color: col, type: 'dashed', width: 1 }, label: { formatter: 'Median', color: T.dim, position: 'insideStartTop', fontSize: 10 } },
        { yAxis: b.p95, lineStyle: { color: T.dim, type: 'dotted', width: 1 }, label: { formatter: 'P95', color: T.dim, position: 'insideStartTop', fontSize: 10 } },
      );
    }
    ser.gaps.forEach((g) =>
      mkA.push([
        { xAxis: g[0], itemStyle: { color: alpha(T.dim, 0.12), borderColor: alpha(T.dim, 0.55), borderType: 'dashed', borderWidth: 1 } },
        { xAxis: g[1] },
      ]),
    );
    mkL.push({ xAxis: ev.t, lineStyle: { color: T.tx, width: 1 }, label: { formatter: 'Selected event', color: T.tx, position: 'insideEndTop', fontSize: 10 } });
    const fr = obs.map((o) => o.frp);
    const ymin = Math.pow(10, Math.floor(Math.log10(Math.max(0.4, Math.min(...fr, useB ? b!.lo : Infinity) * 0.85))));
    const ymax = Math.pow(10, Math.ceil(Math.log10(Math.max(...fr, useB ? b!.hi : 0) * 1.15)));
    const z0 = Math.min(TODAY - 150 * DAY, ev.t - 60 * DAY);
    return {
      grid: { left: 50, right: 16, top: 22, bottom: site ? 50 : 28 },
      tooltip: {
        ...tooltipBox(),
        trigger: 'item',
        formatter: (q: { data: { o: Observation } }) => {
          const o = q.data.o;
          return `<b>${fmtD(o.t)}</b><br>${o.frp.toFixed(1)} MW, ${o.sensor}<br>${o.cnt} detection${o.cnt > 1 ? 's' : ''}${
            o.z != null ? `<br>Deviation ${o.z >= 0 ? '+' : ''}${o.z.toFixed(1)}` : ''
          }`;
        },
      },
      xAxis: { type: 'time', ...axisBox(), ...(site ? {} : { min: obs[0].t - 5 * DAY, max: obs[obs.length - 1].t + 5 * DAY }) },
      yAxis: { type: 'log', min: ymin, max: ymax, name: 'MW', nameTextStyle: { color: T.dim, fontSize: 11, align: 'left' }, ...axisBox() },
      series: [{ type: 'scatter', data, markArea: { silent: true, data: mkA }, markLine: { silent: true, symbol: 'none', data: mkL } }],
      ...(site
        ? {
            dataZoom: [
              { type: 'inside', startValue: z0, endValue: TODAY, filterMode: 'none' },
              {
                type: 'slider', height: 16, bottom: 6, startValue: z0, endValue: TODAY, filterMode: 'none', borderColor: T.line,
                backgroundColor: 'transparent', fillerColor: alpha(T.tx, 0.08), handleSize: 12, textStyle: { color: T.dim, fontSize: 10 },
                dataBackground: { lineStyle: { color: T.line2 }, areaStyle: { color: T.line } },
              },
            ],
          }
        : {}),
    };
  }, [p, ev, ser, T, col]);

  const legend =
    p.kind === 'site' ? (
      <LegendRow>
        <span>
          <svg width="10" height="10" aria-hidden="true">
            <circle cx="5" cy="5" r="4" fill={col} />
          </svg>
          VIIRS pass
        </span>
        <span>
          <svg width="10" height="10" aria-hidden="true">
            <path d="M5 0L10 5L5 10L0 5Z" fill={col} />
          </svg>
          MODIS pass
        </span>
        <span>
          <Swatch color={alpha(col, 0.3)} />
          Normal range
        </span>
        <span>
          <Swatch color="var(--danger)" style={{ borderRadius: '50%' }} />
          Unusual
        </span>
        <span>
          <Swatch color="var(--raised)" border="1px dashed var(--dim)" />
          No clear view <Info k="gap" />
        </span>
      </LegendRow>
    ) : (
      <LegendRow>
        <span>Transient events show only the days the fire was seen.</span>
      </LegendRow>
    );

  return (
    <Card title="Heat over the past year" info="frp" right="Fire radiative power, log scale">
      {option ? <Chart option={option} height={270} className="chart" /> : <div className="empty">No detections yet.</div>}
      {legend}
    </Card>
  );
}

/* ---------- behaviour fingerprint ---------- */
export function FingerprintCard({ p, st, ser }: { p: Place; st: PlaceStats; ser: PlaceSeries }) {
  const col = ccol(p.cls);
  const calendar = useMemo(() => {
    const T = tk();
    const m = new Map<string, number>();
    ser.obs.forEach((o) => {
      const k = iso(o.t);
      m.set(k, (m.get(k) || 0) + o.cnt);
    });
    return {
      tooltip: {
        ...tooltipBox(),
        formatter: (q: { data: [string, number] }) => `${q.data[0]}<br>${q.data[1]} detection${q.data[1] > 1 ? 's' : ''}`,
      },
      visualMap: { show: false, min: 0, max: 5, inRange: { color: [alpha(col, 0.3), col] } },
      calendar: {
        top: 20, left: 26, right: 6, bottom: 2, cellSize: ['auto', 11], range: [iso(START), iso(TODAY)],
        itemStyle: { borderWidth: 2, borderColor: T.panel, color: T.raised },
        splitLine: { show: false },
        yearLabel: { show: false },
        monthLabel: { color: T.dim, fontSize: 10, nameMap: MONTHS },
        dayLabel: { firstDay: 1, color: T.dim, fontSize: 9, margin: 6 },
      },
      series: [{ type: 'heatmap', coordinateSystem: 'calendar', data: [...m.entries()] }],
    };
  }, [ser, col]);

  const hours = useMemo(() => {
    const T = tk();
    const H = hoursOf(p);
    return {
      grid: { left: 34, right: 8, top: 8, bottom: 22 },
      tooltip: {
        ...tooltipBox(),
        trigger: 'axis',
        formatter: (q: { axisValue: number; data: { value: number } }) => `${q.axisValue}:00 to ${q.axisValue}:59<br>${q.data.value} detections`,
      },
      xAxis: { type: 'category', data: H.map((_, i) => i), ...axisBox(), axisLabel: { color: T.dim, fontSize: 10, interval: 3 } },
      yAxis: { type: 'value', ...axisBox(), minInterval: 1 },
      series: [
        {
          type: 'bar',
          barWidth: '70%',
          data: H.map((v, i) => ({ value: v, itemStyle: { color: i < 6 || i >= 18 ? T.dim : col, borderRadius: [2, 2, 0, 0] } })),
        },
      ],
    };
  }, [p, col]);

  const total = ser.obs.reduce((a, o) => a + o.cnt, 0);
  return (
    <Card title="How it behaves" right="Last 12 months">
      <Chart option={calendar} height={132} className="chart" />
      <div className="stats" style={{ margin: '10px 0' }}>
        <Stat label="Active, last 30 d" value={st.d30} unit={pl(st.d30, 'day')} />
        <Stat label="Active, last 90 d" value={st.d90} unit={pl(st.d90, 'day')} />
        <Stat label="Current streak" value={st.streak} unit={pl(st.streak, 'day')} />
        <Stat label="Months active" value={st.months} unit="of 12" />
        <Stat label="Since previous" value={st.sinceLast == null ? 'n/a' : st.sinceLast} unit={st.sinceLast == null ? undefined : pl(st.sinceLast, 'day')} />
        <Stat label="Detections" value={total} unit="12 mo" />
      </div>
      <div className="dim small" style={{ marginBottom: 2 }}>
        When in the day it is seen (local hour of satellite pass) <Info k="gap" />
      </div>
      <Chart option={hours} height={110} className="chart" />
      <LegendRow>
        <span>
          <Swatch color="var(--dim)" />
          Night passes
        </span>
        <span>
          <Swatch color={col} />
          Day passes
        </span>
      </LegendRow>
    </Card>
  );
}

/* ---------- footprint growth ---------- */
export function FootprintCard({ p, ev }: { p: Place; ev: Event }) {
  const option = useMemo(() => {
    const T = tk();
    const r = R('fp:' + ev.id);
    const grow = p.cls === 'wildfire' || ev.status === 'abnormal';
    const n = 8;
    const d = Array.from({ length: n }, (_, i) => [
      ev.t - (n - 1 - i) * 12 * 36e5,
      +(ev.area * (grow ? 0.3 + (0.7 * i) / (n - 1) : 0.85 + r() * 0.3)).toFixed(2),
    ]);
    const col = ccol(p.cls);
    return {
      grid: { left: 42, right: 12, top: 10, bottom: 24 },
      tooltip: { ...tooltipBox(), trigger: 'axis', formatter: (q: { data: [number, number] }) => `${fmtDT(q.data[0])}<br>${q.data[1]} km²` },
      xAxis: { type: 'time', ...axisBox(), axisLabel: { color: T.dim, fontSize: 10, formatter: (v: number) => fmtS(v) } },
      yAxis: { type: 'value', name: 'km²', nameTextStyle: { color: T.dim, fontSize: 10, align: 'left' }, ...axisBox() },
      series: [{ type: 'line', data: d, smooth: 0.25, symbolSize: 6, lineStyle: { color: col, width: 2 }, itemStyle: { color: col }, areaStyle: { color: alpha(col, 0.14) } }],
    };
  }, [p, ev]);
  return (
    <Card title="How big it has been" info="footprint" right="Observed">
      <Chart option={option} height={130} className="chart" />
      <div className="dim small" style={{ marginTop: 4 }}>
        Area covered by neighbouring detections in the last 8 passes. Observed, not predicted.
      </div>
    </Card>
  );
}

/* ---------- surroundings ---------- */
const LC_COL: Record<string, string> = {
  Cropland: '#B9A45B',
  Forest: '#4F8F6B',
  'Bare ground': '#9C8A78',
  'Industrial or built-up': '#6F86A6',
  'Grass and scrub': '#8FAF6B',
  'Water and other': '#5B8DB3',
};

export function SurroundCard({ p, ctx }: { p: Place; ctx: PlaceContext }) {
  const T = tk();
  const option = useMemo(
    () => ({
      tooltip: { ...tooltipBox(), formatter: (q: { name: string; value: number }) => `${q.name}: ${q.value}%` },
      series: [
        {
          type: 'pie',
          radius: ['56%', '82%'],
          avoidLabelOverlap: true,
          label: { show: false },
          itemStyle: { borderColor: T.panel, borderWidth: 2 },
          data: ctx.parts.map(([n, v]) => ({ name: n, value: v, itemStyle: { color: LC_COL[n] } })),
        },
      ],
    }),
    [ctx, T],
  );
  const d = ctx.d;
  return (
    <Card title="What is around it" right="ESA WorldCover, GEM, OSM">
      <div className="row" style={{ gap: 16, alignItems: 'center' }}>
        <Chart option={option} height={150} className="chart" style={{ width: 150, flex: 'none' }} />
        <div style={{ flex: 1, display: 'grid', gap: 4 }} className="small">
          {ctx.parts.map(([n, v]) => (
            <div className="row between" key={n}>
              <span className="row">
                <Swatch color={LC_COL[n]} />
                {n}
              </span>
              <span className="num tx2">{v}%</span>
            </div>
          ))}
        </div>
      </div>
      <div className="dim small" style={{ margin: '12px 0 4px' }}>
        Nearest facilities
      </div>
      <div className="dist small">
        {(
          [
            ['Refinery or petrochemical', d.refinery],
            ['Power plant', d.power],
            ['Steel or cement plant', d.steel],
            ['Mine', d.mine],
            ['Landfill', d.landfill],
          ] as [string, number][]
        ).map(([l, v]) => (
          <div key={l}>
            <span>{l}</span>
            <span className="num">{km(v)}</span>
          </div>
        ))}
      </div>
      {p.cover !== 'good' && (
        <div className="banner plain" style={{ marginTop: 12 }}>
          <span>
            <b>{p.cover === 'weak' ? 'Facility records are thin here.' : 'Facility records are partly complete here.'}</b> A missing nearby
            facility may simply not be mapped. <Info k="cover" />
          </span>
        </div>
      )}
    </Card>
  );
}

/* ---------- before and after ---------- */
const SCENE_PAL: Record<string, [string, string]> = {
  forest: ['#1f3d2b', '#3a6b47'],
  cropland: ['#a9a35f', '#7f9a55'],
  industrial: ['#59626d', '#7b848e'],
  bare: ['#a08a6c', '#c4ac88'],
  mining: ['#7d5f47', '#a98866'],
};

function drawScene(cvs: HTMLCanvasElement, kind: string, seed: string, after: boolean) {
  const w = (cvs.width = 360);
  const h = (cvs.height = 190);
  const g = cvs.getContext('2d');
  if (!g) return;
  const r = R(seed);
  const pal = SCENE_PAL[kind] ?? SCENE_PAL.bare;

  const off = document.createElement('canvas');
  off.width = 36;
  off.height = 19;
  const og = off.getContext('2d')!;
  for (let y = 0; y < 19; y++) {
    for (let x = 0; x < 36; x++) {
      og.fillStyle = hexMix(pal[0], pal[1], r());
      og.fillRect(x, y, 1, 1);
    }
  }
  g.imageSmoothingEnabled = true;
  g.drawImage(off, 0, 0, w, h);

  if (kind === 'cropland') {
    for (let i = 0; i < 26; i++) {
      g.fillStyle = hexMix(r() < 0.5 ? '#c9c07a' : '#6f9250', '#4a5a3a', r() * 0.35);
      g.globalAlpha = 0.55;
      g.fillRect(r() * w, r() * h, r.range(30, 80), r.range(18, 46));
    }
    g.globalAlpha = 1;
  }
  if (kind === 'industrial') {
    g.fillStyle = '#c4cad1';
    for (let i = 0; i < 14; i++) g.fillRect(r() * w * 0.9 + 12, r() * h * 0.8 + 14, r.range(14, 42), r.range(9, 24));
    g.fillStyle = '#e4e8ec';
    for (let i = 0; i < 5; i++) {
      g.beginPath();
      g.arc(r() * w * 0.8 + 30, r() * h * 0.7 + 24, r.range(6, 12), 0, 7);
      g.fill();
    }
    g.strokeStyle = '#3e454d';
    g.lineWidth = 3;
    g.beginPath();
    g.moveTo(0, h * 0.7);
    g.lineTo(w, h * 0.55);
    g.stroke();
  }
  if (kind === 'mining') {
    g.strokeStyle = 'rgba(40,25,15,.4)';
    g.lineWidth = 2;
    for (let i = 1; i < 8; i++) {
      g.beginPath();
      g.ellipse(w * 0.5, h * 0.5, i * 22, i * 12, 0.2, 0, 7);
      g.stroke();
    }
  }
  if (kind === 'forest') {
    g.fillStyle = 'rgba(10,30,18,.4)';
    for (let i = 0; i < 40; i++) {
      g.beginPath();
      g.arc(r() * w, r() * h, r.range(6, 18), 0, 7);
      g.fill();
    }
  }
  if (after) {
    const gr = g.createRadialGradient(w * 0.5, h * 0.5, 4, w * 0.5, h * 0.5, 70);
    gr.addColorStop(0, 'rgba(18,12,10,.85)');
    gr.addColorStop(1, 'rgba(18,12,10,0)');
    g.fillStyle = gr;
    g.fillRect(0, 0, w, h);
    const gl = g.createRadialGradient(w * 0.5, h * 0.5, 0, w * 0.5, h * 0.5, 16);
    gl.addColorStop(0, 'rgba(255,255,255,.95)');
    gl.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = gl;
    g.fillRect(w * 0.5 - 20, h * 0.5 - 20, 40, 40);
  }
}

export function ImageryCard({ p, ev, ctx }: { p: Place; ev: Event; ctx: PlaceContext }) {
  const kind = { wildfire: 'forest', agricultural_burning: 'cropland', gas_flare: 'industrial', industrial: 'industrial', mining: 'mining', unknown: 'bare' }[p.cls];
  const a = useRef<HTMLCanvasElement>(null);
  const b = useRef<HTMLCanvasElement>(null);
  const [split, setSplit] = useState(50);

  useEffect(() => {
    if (a.current) drawScene(a.current, kind, 'im:' + ev.id, false);
    if (b.current) drawScene(b.current, kind, 'im:' + ev.id, true);
  }, [kind, ev.id]);

  return (
    <Card title="Before and after" right="Pre-event scene">
      <div className="compare" role="img" aria-label="Before and after image slider, illustrative">
        <canvas ref={a} />
        <canvas ref={b} style={{ clipPath: `inset(0 0 0 ${split}%)` }} />
        <div className="hd" style={{ left: split + '%' }} />
        <span className="lb" style={{ left: 8 }}>
          Before {fmtS(ev.t - ctx.sceneDays * DAY)}
        </span>
        <span className="lb" style={{ right: 8 }}>
          After {fmtS(ev.t)}
        </span>
        <input
          type="range"
          min={2}
          max={98}
          value={split}
          aria-label="Drag to compare before and after"
          onChange={(e) => setSplit(Number(e.target.value))}
        />
      </div>
      <div className="dim small" style={{ margin: '6px 0 10px' }}>
        Illustrative placeholder, not real imagery. The live build shows the clearest Harmonized Landsat Sentinel-2 scene before the event.
      </div>
      <div className="stats">
        <Stat label="Vegetation (NDVI)" value={ctx.ndvi.toFixed(2)} />
        <Stat label="Burn ratio (NBR)" value={ctx.nbr.toFixed(2)} />
        <Stat label="Built-up (NDBI)" value={ctx.ndbi.toFixed(2)} />
        <Stat label="Clear pixels" value={Math.round(ctx.valid * 100) + '%'} />
        <Stat label="Scene age" value={ctx.sceneDays + ' d'} />
        <Stat label="Built-up share" value={Math.round(ctx.built * 100) + '%'} />
      </div>
    </Card>
  );
}

/* ---------- weather and ground ---------- */
export function ConditionsCard({ ctx }: { ctx: PlaceContext }) {
  return (
    <Card title="Weather and ground" right="Weather and terrain">
      <div className="stats">
        <Stat label={<span>Air dryness <Info k="vpd" /></span>} value={ctx.vpd.toFixed(1)} unit="kPa" />
        <Stat label="Rain, last 72 h" value={ctx.rain.toFixed(0)} unit="mm" />
        <Stat label="Elevation" value={ctx.elev} unit="m" />
        <Stat label="Slope" value={ctx.slope.toFixed(1)} unit="°" />
      </div>
    </Card>
  );
}

/* ---------- other heat nearby ---------- */
export function NearbyCard({ p }: { p: Place }) {
  const nb = nearbyOf(p);
  const W = 250;
  const C = W / 2;
  const sc = 10;
  return (
    <Card title="Other heat nearby" right="Within 10 km (sample)">
      <div className="row" style={{ gap: 14, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <svg width={W} height={W} viewBox={`0 0 ${W} ${W}`} style={{ maxWidth: '100%', flex: 'none' }} role="img" aria-label="Sources within 10 kilometres">
          {[2, 5, 10].map((k) => (
            <g key={k}>
              <circle cx={C} cy={C} r={k * sc} fill="none" stroke="var(--line2)" strokeDasharray="3 4" />
              <text x={C + 3} y={C - k * sc - 3} style={{ fill: 'var(--dim)', fontSize: 10 }}>
                {k} km
              </text>
            </g>
          ))}
          <circle cx={C} cy={C} r={7} fill="var(--tx)" />
          {nb.map((n) => {
            const x = C + Math.cos(n.ang) * n.km * sc;
            const y = C + Math.sin(n.ang) * n.km * sc;
            return (
              <g key={n.name} transform={`translate(${x.toFixed(1)} ${y.toFixed(1)})`}>
                {n.st === 'abnormal' && <circle r={9} fill="none" stroke="var(--danger)" strokeWidth="1.6" />}
                <path
                  d={glyphPath(n.cls, 5)}
                  {...(n.cls === 'unknown' ? { fill: 'none', stroke: ccol(n.cls), strokeWidth: 1.5 } : { fill: ccol(n.cls) })}
                />
              </g>
            );
          })}
        </svg>
        <div style={{ flex: 1, minWidth: 150 }} className="small">
          {nb.length ? (
            nb.map((n) => (
              <div className="row between" key={n.name} style={{ padding: '5px 0', borderBottom: '1px solid var(--line)' }}>
                <span className="row">
                  <Glyph cls={n.cls} size={12} />
                  {n.name}
                </span>
                <span className="num tx2">
                  {n.km} km{n.st === 'abnormal' ? ' , abnormal' : ''}
                </span>
              </div>
            ))
          ) : (
            <div className="dim">No other thermal sources within 10 km.</div>
          )}
        </div>
      </div>
    </Card>
  );
}

/* ---------- past events here ---------- */
export function HistoryCard({ p, ev, onPick }: { p: Place; ev: Event; onPick: (id: string) => void }) {
  const evs = (EVP[p.id] || []).slice().sort((a, b) => b.t - a.t).slice(0, 12);
  return (
    <Card title="Past events here" right={(EVP[p.id] || []).length + ' events, 12 months'}>
      <table className="tbl small">
        <thead>
          <tr>
            <th>Date</th>
            <th>Class</th>
            <th>Status</th>
            <th className="r">Conf.</th>
            <th className="r">Peak</th>
          </tr>
        </thead>
        <tbody>
          {evs.map((e) => (
            <tr
              key={e.id}
              className="click"
              onClick={() => onPick(e.id)}
              style={e.id === ev.id ? { outline: '1px solid var(--tx)', outlineOffset: '-1px' } : undefined}
            >
              <td>
                {fmtS(e.t)} {dparts(e.t).y}
              </td>
              <td>
                <ClassBadge cls={e.cls} />
              </td>
              <td>
                <StatusChip status={e.status} />
                {e.review && <ReviewChip />}
              </td>
              <td className="r">{Math.round(e.conf * 100)}%</td>
              <td className="r">{e.frp.toFixed(0)} MW</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="dim small" style={{ marginTop: 6 }}>
        Select a row to load that event above.
      </div>
    </Card>
  );
}

/* ---------- wind and smoke outlook ---------- */
export function OutlookCard({ p, ev, onOpen }: { p: Place; ev: Event; onOpen: () => void }) {
  const line = outlookLine(p, ev, weatherOf(p, ev));
  return (
    <Card title="Wind and smoke outlook" right={<span className="chip prop">Sample outlook</span>}>
      <p className="sum" style={{ margin: 0 }}>
        {line}
      </p>
      <div className="row wrap" style={{ marginTop: 12 }}>
        <button className="btn pri sm" onClick={onOpen}>
          Open advanced analysis
        </button>
        <span className="dim small">Wind, smoke drift, fire spread and all 141 features.</span>
      </div>
    </Card>
  );
}
