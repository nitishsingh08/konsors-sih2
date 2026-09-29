/* =====================================================================
   The area report: what was drawn on the map.
   Port of areaReport() in frontend/js/pages.js.
   ===================================================================== */
import { useMemo } from 'react';
import { ccol, clamp, tk } from '../lib/core';
import { DAY, fmtD, fmtS } from '../lib/time';
import { CLS } from '../data/classes';
import { PMAP } from '../data/places';
import { Chart, axisBox, tooltipBox } from '../components/ui/Chart';
import { Bar, Card, Empty, Stat } from '../components/ui/Primitives';
import { ClassBadge } from '../components/ui/Glyph';
import { Icon } from '../components/ui/Icon';
import { groups, sev, type Group } from '../state/selectors';
import type { Event } from '../lib/types';

export interface AreaBox {
  lo0: number;
  lo1: number;
  la0: number;
  la1: number;
}

export const inArea = (a: AreaBox, e: Event) => {
  const p = PMAP[e.pid];
  return p.lon >= a.lo0 && p.lon <= a.lo1 && p.lat >= a.la0 && p.lat <= a.la1;
};

export function AreaReport({
  area,
  evs,
  asOf,
  win,
  sel,
  onSelect,
  onExport,
  onClear,
  expanded,
}: {
  area: AreaBox;
  evs: Event[];
  asOf: number;
  win: number;
  sel: string | null;
  onSelect: (id: string) => void;
  onExport: () => void;
  onClear: () => void;
  expanded: boolean;
}) {
  const inside = evs.filter((e) => inArea(area, e));
  const gs = groups(inside).sort((x, y) => sev(y.top) - sev(x.top) || y.top.t - x.top.t);
  const by = Object.fromEntries(CLS.map((c) => [c.id, 0])) as Record<string, number>;
  inside.forEach((e) => by[e.cls]++);
  const mx = Math.max(1, ...Object.values(by));
  const abn = gs.filter((g) => sev(g.top) === 3).length;
  const rev = inside.filter((e) => e.review).length;

  const T = tk();
  const trend = useMemo(() => {
    const nb = 12;
    const bw = (win * DAY) / nb;
    const lo = asOf - win * DAY;
    const cnt = new Array<number>(nb).fill(0);
    inside.forEach((e) => cnt[clamp(Math.floor((e.t - lo) / bw), 0, nb - 1)]++);
    return {
      grid: { left: 30, right: 8, top: 8, bottom: 22 },
      tooltip: { ...tooltipBox(), trigger: 'axis' },
      xAxis: { type: 'category', data: cnt.map((_, i) => fmtS(lo + i * bw)), ...axisBox(), axisLabel: { color: T.dim, fontSize: 10, interval: 2 } },
      yAxis: { type: 'value', minInterval: 1, ...axisBox() },
      series: [{ type: 'bar', data: cnt, itemStyle: { color: T.dim, borderRadius: [2, 2, 0, 0] } }],
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inside, win, asOf, T]);

  const head = (
    <Card title="Selected area">
      <div className="stats" style={{ gridTemplateColumns: 'repeat(2,minmax(0,1fr))' }}>
        <Stat label="Events" value={inside.length} />
        <Stat label="Places" value={gs.length} />
        <Stat label="Abnormal" value={abn} />
        <Stat label="Needs review" value={rev} />
      </div>
      <div className="dim small" style={{ marginTop: 10 }}>
        {area.la0.toFixed(1)}° to {area.la1.toFixed(1)}°N, {area.lo0.toFixed(1)}° to {area.lo1.toFixed(1)}°E, {fmtS(asOf - win * DAY + DAY)} to{' '}
        {fmtD(asOf)}. Filters on the map apply here too.
      </div>
      <div className="row wrap" style={{ marginTop: 12 }}>
        <button className="btn sm" onClick={onExport}>
          <Icon name="download" size={14} />
          Export events
        </button>
        <button className="btn sm" onClick={onClear}>
          Clear area
        </button>
      </div>
    </Card>
  );

  const mix = (
    <Card title="Class mix">
      <div className="shap">
        {CLS.map((c) => (
          <Bar key={c.id} label={<ClassBadge cls={c.id} />} value={by[c.id] / mx} color={ccol(c.id)} columns="150px 1fr 30px" />
        ))}
      </div>
    </Card>
  );
  const trendCard = (
    <Card title="Events over time">
      <Chart option={trend} height={120} className="chart" />
    </Card>
  );
  const list = (
    <Card title="Places inside">
      {gs.length ? (
        <>
          <div className="queue">
            {gs.slice(0, 12).map((g) => (
              <AreaRow key={g.p.id} g={g} sel={sel} onSelect={onSelect} />
            ))}
          </div>
          {gs.length > 12 && <div className="dim small" style={{ marginTop: 8 }}>and {gs.length - 12} more</div>}
        </>
      ) : (
        <Empty title="No events in this box">Draw a larger area or widen the date window.</Empty>
      )}
    </Card>
  );

  if (expanded) {
    return (
      <div className="rep-cols">
        <div>{[head, mix, trendCard]}</div>
        <div>{[list]}</div>
      </div>
    );
  }
  return <div className="rep-stack">{[head, mix, trendCard, list]}</div>;
}

function AreaRow({ g, sel, onSelect }: { g: Group; sel: string | null; onSelect: (id: string) => void }) {
  const e = g.top;
  const p = g.p;
  return (
    <button className="qi" aria-current={sel === p.id} onClick={() => onSelect(p.id)}>
      <span>
        <ClassBadge cls={e.cls} />
      </span>
      <span style={{ minWidth: 0 }}>
        <span className="t" style={{ display: 'block' }}>
          {p.name}
        </span>
        <span className="s" style={{ display: 'block' }}>
          {p.district}, {p.state}
        </span>
      </span>
      <span className="m">
        <b className="num">{e.frp.toFixed(0)} MW</b>
      </span>
    </button>
  );
}
