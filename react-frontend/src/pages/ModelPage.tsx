/* =====================================================================
   Model and data health: freshness, pipeline runs, per-class results,
   calibration and the confusion matrix.
   Port of pageModel() in frontend/js/pages.js.
   ===================================================================== */
import { useMemo } from 'react';
import { clamp, ccol, R, tk } from '../lib/core';
import { nf } from '../lib/time';
import { fmtDT, HOUR, TODAY } from '../lib/time';
import { CLSMAP } from '../data/classes';
import { METRICS } from '../data/places';
import { Chart, axisBox, tooltipBox } from '../components/ui/Chart';
import modelMarkLight from '../assets/model/model-mark.svg';
import modelMarkDark from '../assets/model/model-mark-on-dark.svg';
import { Banner, Card } from '../components/ui/Primitives';
import { ClassBadge } from '../components/ui/Glyph';
import { PageHead } from './PageHead';
import { useTheme } from '../state/ThemeProvider';
import type { ClassId } from '../lib/types';

const FRESH: [string, string, string, 'ok' | 'warn'][] = [
  ['FIRMS VIIRS (NOAA-20, NOAA-21, Suomi-NPP)', '2 h 41 min old', 'Within the 3 h target', 'ok'],
  ['FIRMS MODIS (Terra, Aqua)', '3 h 05 min old', 'A little over target', 'warn'],
  ['Facility records (Global Energy Monitor)', 'Updated 18 Aug 2026', 'Refreshed monthly', 'ok'],
  ['OpenStreetMap industrial extract', 'Updated 14 Sep 2026', 'Refreshed weekly', 'ok'],
  ['Land cover (ESA WorldCover)', '2021 map', 'Static, 10 m', 'ok'],
  ['Satellite imagery (HLS)', 'Newest clear scene 3 d old', 'Limited by cloud', 'warn'],
  ['Weather (ERA5 for training)', 'About 5 d behind', 'Live runs use a forecast source', 'ok'],
];

export function ModelPage() {
  const { theme } = useTheme();
  const macro = METRICS.reduce((a, m) => a + m.f1, 0) / METRICS.length;
  const T = tk();

  const confusion = useMemo(() => {
    const ids = METRICS.map((m) => m.id);
    const W: Record<ClassId, Partial<Record<ClassId, number>>> = {
      gas_flare: { industrial: 0.7, unknown: 0.3 },
      industrial: { gas_flare: 0.35, mining: 0.35, unknown: 0.3 },
      mining: { industrial: 0.6, unknown: 0.4 },
      wildfire: { agricultural_burning: 0.6, unknown: 0.4 },
      agricultural_burning: { wildfire: 0.6, unknown: 0.4 },
      unknown: { industrial: 0.3, wildfire: 0.25, agricultural_burning: 0.25, gas_flare: 0.2 },
    };
    const names = METRICS.map((m) => CLSMAP[m.id].label.replace('Agricultural burning', 'Ag. burning').replace(' source', ''));
    const cm = METRICS.map((m) => {
      const row = Object.fromEntries(ids.map((i) => [i, 0])) as Record<ClassId, number>;
      const ok = Math.round(m.n * m.r);
      row[m.id] = ok;
      const rest = m.n - ok;
      Object.entries(W[m.id]).forEach(([k, w]) => {
        row[k as ClassId] += Math.round(rest * (w ?? 0));
      });
      return ids.map((i) => row[i]);
    });
    const data: [number, number, number][] = [];
    let mx = 0;
    cm.forEach((row, i) => row.forEach((v, j) => {
      data.push([j, i, v]);
      mx = Math.max(mx, v);
    }));
    return {
      grid: { left: 96, right: 10, top: 10, bottom: 54 },
      tooltip: { ...tooltipBox(), formatter: (q: { data: [number, number, number] }) => `True ${names[q.data[1]]}<br>Predicted ${names[q.data[0]]}: ${q.data[2]}` },
      xAxis: {
        type: 'category', data: names, name: 'Predicted', nameLocation: 'middle', nameGap: 36, nameTextStyle: { color: T.dim }, ...axisBox(),
        axisLabel: { color: T.dim, fontSize: 10, interval: 0, rotate: 20 }, splitLine: { show: false },
      },
      yAxis: {
        type: 'category', data: names, inverse: true, name: 'True', nameTextStyle: { color: T.dim }, ...axisBox(),
        axisLabel: { color: T.dim, fontSize: 10 }, splitLine: { show: false },
      },
      visualMap: { show: false, min: 0, max: mx, inRange: { color: [T.raised, ccol('industrial')] } },
      series: [{ type: 'heatmap', data, label: { show: true, color: T.tx, fontSize: 10 }, itemStyle: { borderColor: T.panel, borderWidth: 2 } }],
    };
  }, [T]);

  const reliability = useMemo(() => {
    const r = R('rel');
    const xs = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95];
    return {
      grid: { left: 40, right: 12, top: 10, bottom: 30 },
      tooltip: { ...tooltipBox(), trigger: 'axis' },
      xAxis: { type: 'value', min: 0, max: 1, name: 'Predicted confidence', nameLocation: 'middle', nameGap: 22, nameTextStyle: { color: T.dim, fontSize: 10 }, ...axisBox() },
      yAxis: { type: 'value', min: 0, max: 1, name: 'Observed', nameTextStyle: { color: T.dim, fontSize: 10, align: 'left' }, ...axisBox() },
      series: [
        { type: 'line', data: [[0, 0], [1, 1]], symbol: 'none', lineStyle: { color: T.dim, type: 'dashed', width: 1 } },
        {
          type: 'line', data: xs.map((x) => [x, +clamp(x + r.range(-0.06, 0.04), 0, 1).toFixed(2)]), symbolSize: 6,
          lineStyle: { color: T.tx, width: 2 }, itemStyle: { color: T.tx },
        },
      ],
    };
  }, [T]);

  const runs = Array.from({ length: 8 }, (_, i) => {
    const t = TODAY - (i * 3 + 0.7) * HOUR;
    const r = R('run' + i);
    return [fmtDT(t), r.int(4, 9) + ' min', nf(r.int(1800, 5200)), nf(r.int(30, 140)), i === 3 ? 'Retried once' : 'Completed'];
  });

  return (
    <div className="page">
      <div className="page-in">
        <PageHead
          title="Model and data health"
          lead="What the system is running on, how fresh each source is, and how well it is doing. This is the page that shows whether to trust the rest."
          mark={theme === 'dark' ? modelMarkDark : modelMarkLight}
        />
        <Banner>
          <span>
            <b>Sample data.</b> The numbers below are placeholders for the layout. Replace them with results from a spatially blocked hold-out set
            with independently checked labels.
          </span>
        </Banner>

        <div className="g2">
          <Card title="Data freshness" right="Sample values">
            <table className="tbl small">
              <tbody>
                {FRESH.map(([n, a, s, k]) => (
                  <tr key={n}>
                    <td>{n}</td>
                    <td className="tx2">{a}</td>
                    <td>
                      <span className={'chip ' + (k === 'warn' ? 'rev' : '')}>{s}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
          <Card title="Pipeline runs" right="Every 3 hours">
            <table className="tbl small">
              <thead>
                <tr>
                  <th>Started</th>
                  <th>Time taken</th>
                  <th className="r">Detections</th>
                  <th className="r">Events</th>
                  <th>Result</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r, i) => (
                  <tr key={i}>
                    <td>{r[0]}</td>
                    <td>{r[1]}</td>
                    <td className="r">{r[2]}</td>
                    <td className="r">{r[3]}</td>
                    <td>{r[4]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>

        <div className="g21">
          <Card title="Per-class results" right="Hold-out set">
            <table className="tbl small">
              <thead>
                <tr>
                  <th>Class</th>
                  <th className="r">Precision</th>
                  <th className="r">Recall</th>
                  <th className="r">F1</th>
                  <th className="r">Examples</th>
                </tr>
              </thead>
              <tbody>
                {METRICS.map((m) => (
                  <tr key={m.id}>
                    <td>
                      <ClassBadge cls={m.id} />
                    </td>
                    <td className="r">{m.p.toFixed(2)}</td>
                    <td className="r">{m.r.toFixed(2)}</td>
                    <td className="r">{m.f1.toFixed(2)}</td>
                    <td className="r">{m.n}</td>
                  </tr>
                ))}
                <tr>
                  <td>
                    <b>Macro average</b>
                  </td>
                  <td />
                  <td />
                  <td className="r">
                    <b>{macro.toFixed(2)}</b>
                  </td>
                  <td className="r">{METRICS.reduce((a, m) => a + m.n, 0)}</td>
                </tr>
              </tbody>
            </table>
            <p className="small tx2" style={{ margin: '10px 0 0' }}>
              If macro F1 on the real hold-out set comes in under about 0.6, lead with the explanations and the abnormal-versus-routine
              judgement, not the accuracy figure.
            </p>
          </Card>
          <Card title="Calibration" info="conf" right="Reliability diagram">
            <Chart option={reliability} height={240} className="chart" />
            <div className="row between small" style={{ marginTop: 6 }}>
              <span className="tx2">Brier score 0.11</span>
              <span className="tx2">Dashed line is perfect</span>
            </div>
          </Card>
        </div>

        <div className="g2">
          <Card title="Confusion matrix" right="Counts">
            <Chart option={confusion} height={280} className="chart" />
          </Card>
          <Card title="How it is validated">
            <ul className="list">
              <li>Blocks of land are kept apart between training and testing, with a buffer, so neighbouring pixels of one site never sit on both sides.</li>
              <li>A later period is held out in time, so the model is tested on days it has not seen.</li>
              <li>
                Labels come from several weak sources: the Nightfire flare catalogue, facility trackers, land cover with season, and Sentinel-2
                burn scars for confirmation. A few hundred are checked by hand and kept aside.
              </li>
              <li>Probabilities are calibrated after training, and anything under 60% is sent for human review.</li>
            </ul>
          </Card>
        </div>
      </div>
    </div>
  );
}
