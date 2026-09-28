/* =====================================================================
   Regional analytics: how thermal events spread over classes, places and
   seasons. Port of pageRegions() in frontend/js/pages.js.
   ===================================================================== */
import { useMemo, useState } from 'react';
import { alpha, ccol, median, R, tk } from '../lib/core';
import { DAY, HOUR, MONTHS, START, TODAY, fmtS } from '../lib/time';
import { CLS } from '../data/classes';
import { EVENTS, EVP, PLACES, PMAP, seriesOf, statsOf, hoursOf } from '../data/places';
import { passes } from '../state/selectors';
import { useFilters } from '../state/useFilters';
import { useNav } from '../components/Shell';
import { Chart, axisBox, tooltipBox } from '../components/ui/Chart';
import { Banner, Card, Spark } from '../components/ui/Primitives';
import { ClassBadge, Glyph } from '../components/ui/Glyph';
import { PageHead } from './PageHead';
import { eventsCsv } from '../data/analysis';
import { useFiles } from '../lib/files';
import type { Place } from '../lib/types';

export function RegionsPage() {
  const [f] = useFilters();
  const { goPlace } = useNav();
  const [cmpA, setCmpA] = useState(() => (PLACES.find((p) => p.code === 'JH-01') ?? PLACES[0]).id);
  const [cmpB, setCmpB] = useState(() => (PLACES.find((p) => p.code === 'JH-02') ?? PLACES[1]).id);
  const { saveFile } = useFiles();
  const evs = useMemo(() => EVENTS.filter((e) => passes(f, e)), [f]);

  const mk = (t: number) => {
    const d = new Date(t + 5.5 * HOUR);
    return d.getUTCFullYear() * 12 + d.getUTCMonth();
  };
  const T = tk();

  const mix = useMemo(() => {
    const m0 = mk(START);
    const m1 = mk(TODAY);
    const labels: string[] = [];
    const keys: number[] = [];
    for (let m = m0; m <= m1; m++) {
      keys.push(m);
      labels.push(MONTHS[m % 12] + ' ' + String(Math.floor(m / 12)).slice(2));
    }
    return {
      grid: { left: 36, right: 10, top: 10, bottom: 26 },
      tooltip: { ...tooltipBox(), trigger: 'axis', axisPointer: { type: 'shadow' } },
      xAxis: { type: 'category', data: labels, ...axisBox() },
      yAxis: { type: 'value', ...axisBox() },
      series: CLS.map((c) => ({
        name: c.label,
        type: 'bar',
        stack: 'a',
        barWidth: '62%',
        itemStyle: { color: ccol(c.id) },
        data: keys.map((k) => evs.filter((e) => e.cls === c.id && mk(e.t) === k).length),
      })),
    };
  }, [evs]);

  const weekly = useMemo(() => {
    const wk = 52;
    const ty = new Array<number>(wk).fill(0);
    evs.forEach((e) => {
      ty[Math.min(wk - 1, Math.floor((e.t - START) / (7 * DAY)))] += e.nDet;
    });
    const r = R('prior');
    const pr = Array.from({ length: 5 }, () => ty.map((v) => v * r.range(0.55, 1.6)));
    const mn = ty.map((_, i) => Math.min(...pr.map((y) => y[i])));
    const mx = ty.map((_, i) => Math.max(...pr.map((y) => y[i])));
    const md = ty.map((_, i) => median(pr.map((y) => y[i])));
    const wl = ty.map((_, i) => fmtS(START + i * 7 * DAY));
    return {
      grid: { left: 46, right: 10, top: 10, bottom: 26 },
      tooltip: { ...tooltipBox(), trigger: 'axis' },
      xAxis: { type: 'category', data: wl, boundaryGap: false, ...axisBox(), axisLabel: { color: T.dim, fontSize: 10, interval: 7 } },
      yAxis: { type: 'value', ...axisBox() },
      series: [
        { name: 'Low', type: 'line', stack: 'b', data: mn.map(Math.round), lineStyle: { opacity: 0 }, symbol: 'none', tooltip: { show: false } },
        {
          name: 'Earlier years, range', type: 'line', stack: 'b', data: mx.map((v, i) => Math.round(v - mn[i])),
          lineStyle: { opacity: 0 }, symbol: 'none', areaStyle: { color: alpha(T.dim, 0.22) },
        },
        { name: 'Earlier years, median', type: 'line', data: md.map(Math.round), symbol: 'none', lineStyle: { color: T.dim, type: 'dashed', width: 1.2 } },
        { name: 'This year', type: 'line', data: ty, symbol: 'none', lineStyle: { color: T.tx, width: 2 } },
      ],
    };
  }, [evs, T]);

  const districts = useMemo(() => {
    const dm = new Map<string, number>();
    evs.forEach((e) => {
      const p = PMAP[e.pid];
      const k = p.district + ', ' + p.state;
      dm.set(k, (dm.get(k) || 0) + 1);
    });
    const top = [...dm.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10).reverse();
    return {
      grid: { left: 190, right: 30, top: 6, bottom: 20 },
      tooltip: { ...tooltipBox(), trigger: 'item' },
      xAxis: { type: 'value', ...axisBox() },
      yAxis: { type: 'category', data: top.map((t) => t[0]), ...axisBox(), axisLabel: { color: T.tx2, fontSize: 11, width: 178, overflow: 'truncate' }, splitLine: { show: false } },
      series: [{ type: 'bar', data: top.map((t) => t[1]), barWidth: '62%', itemStyle: { color: T.dim, borderRadius: [0, 3, 3, 0] } }],
    };
  }, [evs, T]);

  const hours = useMemo(() => {
    const H = new Array<number>(24).fill(0);
    evs.forEach((e) => {
      H[Math.floor(e.hr)] += 1;
    });
    return {
      grid: { left: 36, right: 8, top: 8, bottom: 22 },
      tooltip: { ...tooltipBox(), trigger: 'axis' },
      xAxis: { type: 'category', data: H.map((_, i) => i), ...axisBox(), axisLabel: { color: T.dim, fontSize: 10, interval: 2 } },
      yAxis: { type: 'value', ...axisBox() },
      series: [
        {
          type: 'bar', barWidth: '70%',
          data: H.map((v, i) => ({ value: v, itemStyle: { color: i < 6 || i >= 18 ? T.dim : T.tx2, borderRadius: [2, 2, 0, 0] } })),
        },
      ],
    };
  }, [evs, T]);

  const sites = useMemo(
    () =>
      PLACES.filter((p) => p.kind === 'site')
        .map((p) => {
          const l = EVP[p.id] || [];
          return { p, abn: l.filter((e) => e.status === 'abnormal').length, n: l.length };
        })
        .sort((a, b) => b.abn - a.abn || b.n - a.n)
        .slice(0, 10),
    [],
  );

  const a = PMAP[cmpA] ?? PLACES[0];
  const b = PMAP[cmpB] ?? PLACES[1];

  const compare = useMemo(() => {
    const mkS = (p: Place, col: string) => ({
      name: p.name,
      type: 'line',
      showSymbol: true,
      symbolSize: 4,
      connectNulls: true,
      lineStyle: { color: col, width: 1.6 },
      itemStyle: { color: col },
      data: seriesOf(p).obs.filter((o) => o.t > TODAY - 120 * DAY).map((o) => [o.t, +o.frp.toFixed(2)]),
    });
    const cA = ccol(a.cls);
    const cB = a.cls === b.cls ? tk().tx : ccol(b.cls);
    return {
      grid: { left: 46, right: 10, top: 10, bottom: 26 },
      tooltip: { ...tooltipBox(), trigger: 'axis', formatter: (q: { seriesName: string; data: [number, number] }[]) => q.map((x) => `${x.seriesName}: ${x.data[1]} MW`).join('<br>') },
      xAxis: { type: 'time', ...axisBox() },
      yAxis: { type: 'log', min: 0.5, ...axisBox() },
      series: [mkS(a, cA), mkS(b, cB)],
    };
  }, [a, b]);

  const cmpTable = useMemo(() => {
    const row = (p: Place) => {
      const ser = seriesOf(p);
      const st = statsOf(p);
      const bs = ser.base!;
      const abn = (EVP[p.id] || []).filter((e) => e.status === 'abnormal').length;
      const last = ser.obs[ser.obs.length - 1];
      const H = hoursOf(p);
      const nt = [...H.slice(0, 6), ...H.slice(18)].reduce((x, y) => x + y, 0) / H.reduce((x, y) => x + y, 1);
      return [
        <ClassBadge key="c" cls={p.cls} />,
        bs.med.toFixed(1) + ' MW',
        bs.p95.toFixed(1) + ' MW',
        st.d90 + ' of 90',
        Math.round(nt * 100) + '%',
        abn,
        last && last.z != null ? (last.z >= 0 ? '+' : '') + last.z.toFixed(1) : 'n/a',
      ];
    };
    const ra = row(a);
    const rb = row(b);
    const labs = ['Class', 'Median fire power', '95th percentile', 'Active days', 'Night detections', 'Abnormal events, 12 mo', 'Deviation now'];
    return (
      <table className="tbl small">
        <thead>
          <tr>
            <th />
            <th>{a.district}</th>
            <th>{b.district}</th>
          </tr>
        </thead>
        <tbody>
          {labs.map((l, i) => (
            <tr key={l}>
              <td className="dim">{l}</td>
              <td>{ra[i]}</td>
              <td>{rb[i]}</td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  }, [a, b]);

  return (
    <div className="page">
      <div className="page-in">
        <PageHead title="Regional analytics" lead="How thermal events are spread over classes, places and seasons across India in the past 12 months. Map filters apply here too." />
        <Banner>
          <span>
            <b>Sample data.</b> Everything on this page is generated sample data. Earlier-year values are modelled for the demo.
          </span>
        </Banner>

        <div className="g2">
          <Card title="Events by class and month" right="Events per month">
            <Chart option={mix} height={250} className="chart" />
            <div className="legendrow">
              {CLS.map((c) => (
                <span key={c.id}>
                  <Glyph cls={c.id} size={11} />
                  {c.label}
                </span>
              ))}
            </div>
          </Card>
          <Card title="This year against earlier years" right="Detections per week">
            <Chart option={weekly} height={250} className="chart" />
            <div className="legendrow">
              <span>
                <i className="sw" style={{ background: 'var(--tx)' }} />
                This year
              </span>
              <span>
                <i className="sw" style={{ background: 'var(--dim)', opacity: 0.5 }} />
                Range of five earlier years (modelled)
              </span>
            </div>
          </Card>
        </div>

        <div className="g2">
          <Card title="Busiest districts" right="Events, 12 months">
            <Chart option={districts} height={300} className="chart" />
          </Card>
          <Card title="When detections happen" right="Local hour of pass">
            <Chart option={hours} height={300} className="chart" />
            <div className="legendrow">
              <span>
                <i className="sw" style={{ background: 'var(--dim)' }} />
                Night passes
              </span>
              <span>
                <i className="sw" style={{ background: 'var(--tx2)' }} />
                Day passes
              </span>
            </div>
          </Card>
        </div>

        <div className="g2">
          <Card title="Persistent sites to watch" right="Most abnormal events first">
            <div className="queue">
              {sites.map(({ p, abn }) => (
                <button key={p.id} className="qi" style={{ gridTemplateColumns: 'minmax(0,1fr) auto' }} onClick={() => goPlace(p.id)}>
                  <span style={{ minWidth: 0 }}>
                    <span className="t" style={{ display: 'block' }}>
                      {p.name}
                    </span>
                    <span className="chips" style={{ marginTop: 3 }}>
                      <ClassBadge cls={p.cls} />
                      {abn ? <span className="chip abn">{abn} abnormal</span> : null}
                    </span>
                  </span>
                  <span style={{ color: 'var(--dim)' }}>
                    <Spark values={seriesOf(p).obs.filter((o) => o.t > TODAY - 90 * DAY).map((o) => o.frp).slice(-40)} w={96} h={26} />
                  </span>
                </button>
              ))}
            </div>
          </Card>
          <Card title="Compare two places" right="Last 120 days">
            <div className="row wrap" style={{ marginBottom: 10 }}>
              <select className="sel" aria-label="First place" style={{ maxWidth: '100%' }} value={a.id} onChange={(e) => setCmpA(e.target.value)}>
                {PLACES.filter((p) => p.kind === 'site').map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
              <select className="sel" aria-label="Second place" style={{ maxWidth: '100%' }} value={b.id} onChange={(e) => setCmpB(e.target.value)}>
                {PLACES.filter((p) => p.kind === 'site').map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
              <button className="btn sm" onClick={() => saveFile('agni-netra-region-events.csv', eventsCsv(evs), 'CSV')}>
                Export this view
              </button>
            </div>
            <Chart option={compare} height={220} className="chart" />
            {cmpTable}
          </Card>
        </div>
      </div>
    </div>
  );
}
