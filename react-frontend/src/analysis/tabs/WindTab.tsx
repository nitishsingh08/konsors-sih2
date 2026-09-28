/* =====================================================================
   The wind and spread tab: the live numbers for the selected forecast
   hour, the smoke plume, the fire spread outlook, the fire weather
   system, and the ground along the wind line.
   Port of tabWind()/windLive() in frontend/js/analysis/analysis.js.
   ===================================================================== */
import { useMemo, type ReactNode } from 'react';
import { alpha, ccol, clamp, lc, tk } from '../../lib/core';
import { hourLabel } from '../../lib/time';
import { CLSMAP } from '../../data/classes';
import { calcBUI, calcFWI, calcISI, compass, spreadClass, windAt } from '../../data/weather';
import { Chart, axisBox, tooltipBox, type ECharts } from '../../components/ui/Chart';
import { Card } from '../../components/ui/Primitives';
import { Icon } from '../../components/ui/Icon';
import type { Event, Plume, SpreadOutlook, Terrain, Weather } from '../../lib/types';

export const tile = (l: string, v: ReactNode, sub?: string) => (
  <div className="stat">
    <div className="l">{l}</div>
    <div className="v">{v}</div>
    {sub && <div className="u sm2">{sub}</div>}
  </div>
);

export function WindLive({ ev, W, h, plume, spread }: { ev: Event; W: Weather; h: number; plume: Plume; spread: SpreadOutlook | null }) {
  const w = windAt(W, h);
  const i = Math.round(h);
  const rain6 = W.rain.slice(i, i + 6).reduce((a, b) => a + b, 0);
  const isi = calcISI(W.ffmc[i], w.spd * 3.6);
  const bui = calcBUI(W.dmc, W.dc);
  const fwi = calcFWI(isi, bui);

  return (
    <>
      <div className="stats">
        {tile('Wind', <>{Math.round(w.spd * 3.6)} <span className="u">km/h</span></>, 'from the ' + compass(w.dir))}
        {tile('Gusts', <>{Math.round(W.gust[i] * 3.6)} <span className="u">km/h</span></>)}
        {tile('Humidity', <>{Math.round(W.rh[i])}<span className="u">%</span></>)}
        {tile(
          'Air dryness',
          <>{W.vpd[i].toFixed(1)} <span className="u">kPa</span></>,
          W.vpd[i] > 3 ? 'very dry air' : W.vpd[i] > 1.6 ? 'dry air' : 'damp air',
        )}
        {tile('Temperature', <>{Math.round(W.temp[i])}<span className="u">°C</span></>)}
        {tile('Rain, next 6 h', <>{rain6.toFixed(1)} <span className="u">mm</span></>)}
      </div>

      <Card title="Smoke" info="plume" right="Where it can travel" style={{ marginTop: 12 }}>
        <p className="sum" style={{ margin: 0 }}>
          Smoke released now could drift about <b>{plume.reachKm.toFixed(1)} km</b> toward the {compass(plume.bearing)} by{' '}
          {hourLabel(W.t0, h || 1)}. The paler outline shows where it might go if the forecast wind is a little off.
        </p>
        <p className="dim small" style={{ margin: '8px 0 0' }}>
          This shows where smoke can travel. It does not say how thick it will be.
        </p>
      </Card>

      {spread ? (
        <Card title="Fire spread outlook" info="spreadind" right={<span className="chip prop">Indicative</span>} style={{ marginTop: 12 }}>
          <p className="sum" style={{ margin: '0 0 10px' }}>
            {spread.cls} spread rate:{' '}
            {[
              `fuel is ${W.ffmc[i] > 85 ? 'very dry' : W.ffmc[i] > 70 ? 'fairly dry' : 'damp'} (FFMC ${W.ffmc[i].toFixed(0)})`,
              `wind ${Math.round(w.spd * 3.6)} km/h`,
              spread.horizons[2].meanSlope > 1
                ? `ground rises ${spread.horizons[2].meanSlope.toFixed(0)}° along the wind line, which speeds it up`
                : spread.horizons[2].meanSlope < -1
                  ? `ground falls ${Math.abs(spread.horizons[2].meanSlope).toFixed(0)}° along the wind line, which slows it`
                  : 'flat ground along the wind line',
            ].join(', ')}
            .
          </p>
          <table className="tbl small">
            <thead>
              <tr>
                <th>In</th>
                <th className="r">Reach</th>
                <th className="r">Speed</th>
                <th>Heading</th>
                <th className="r">Area</th>
              </tr>
            </thead>
            <tbody>
              {spread.horizons.map((z) => (
                <tr key={z.t}>
                  <td>{z.t} h</td>
                  <td className="r">{z.D.toFixed(1)} km</td>
                  <td className="r">
                    {z.rateLo.toFixed(1)} to {z.rateHi.toFixed(1)} km/h
                  </td>
                  <td>{compass(z.dirTo)}</td>
                  <td className="r">{z.area.toFixed(1)} km²</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="dim small" style={{ margin: '8px 0 0' }}>
            A rough guide from wind, fuel dryness and slope. It is not a validated fire-front forecast.
          </p>
        </Card>
      ) : (
        <Card title="Fire spread outlook" right={<span className="chip">Not applicable</span>} style={{ marginTop: 12 }}>
          <p className="sum" style={{ margin: 0 }}>
            This looks like {lc(CLSMAP[ev.cls].label)}, and it stays inside its site. Spread is only estimated for wildfire and crop burning. For
            this place, the smoke plume is the part that matters.
          </p>
        </Card>
      )}

      <Card title="Fire weather at this hour" info="fwi" style={{ marginTop: 12 }}>
        <div className="stats">
          {tile('Fuel moisture (FFMC)', W.ffmc[i].toFixed(0))}
          {tile('Spread index (ISI)', isi.toFixed(1), spreadClass(isi))}
          {tile('Fire weather (FWI)', fwi.toFixed(1))}
        </div>
      </Card>
    </>
  );
}

/** the elevation profile along the wind line, with hover tracking on the map */
export function ElevChart({ terrain, cls, onHover }: { terrain: Terrain; cls: Event['cls']; onHover: (at: [number, number] | null) => void }) {
  const T = tk();
  const col = ccol(cls);
  const option = useMemo(
    () => ({
      grid: { left: 46, right: 12, top: 12, bottom: 24 },
      tooltip: {
        ...tooltipBox(),
        trigger: 'axis',
        formatter: (q: { axisValue: string; data: number }) => `${q.axisValue} km along the wind line<br>${Math.round(q.data)} m`,
      },
      xAxis: { type: 'category', data: terrain.pts.map((q) => q.km.toFixed(1)), boundaryGap: false, ...axisBox(), axisLabel: { color: T.dim, fontSize: 10, interval: 7 }, splitLine: { show: false } },
      yAxis: { type: 'value', scale: true, name: 'm', nameTextStyle: { color: T.dim, fontSize: 10, align: 'left' }, ...axisBox() },
      series: [{ type: 'line', data: terrain.pts.map((q) => +q.elev.toFixed(1)), showSymbol: false, smooth: 0.3, lineStyle: { color: col, width: 2 }, areaStyle: { color: alpha(col, 0.16) } }],
    }),
    [terrain, col, T],
  );

  const onInit = (c: ECharts) => {
    c.getZr().on('mousemove', (e: { offsetX: number; offsetY: number }) => {
      const pt = c.convertFromPixel({ seriesIndex: 0 }, [e.offsetX, e.offsetY]) as number[] | null;
      if (!pt) return;
      const i = clamp(Math.round(pt[0]), 0, terrain.pts.length - 1);
      const q = terrain.pts[i];
      onHover([q.lat, q.lon]);
    });
    c.getZr().on('globalout', () => onHover(null));
  };

  return <Chart option={option} height={150} onInit={onInit} />;
}

export function WindTab({
  ev,
  W,
  h,
  headline,
  plume,
  spread,
  terrain,
  replayLabel,
  onFit,
  onReplay,
  onJson,
  onHoverElev,
}: {
  ev: Event;
  W: Weather;
  h: number;
  headline: string;
  plume: Plume;
  spread: SpreadOutlook | null;
  terrain: Terrain | null;
  replayLabel: string;
  onFit: () => void;
  onReplay: () => void;
  onJson: () => void;
  onHoverElev: (at: [number, number] | null) => void;
}) {
  return (
    <>
      <p className="headline">{headline}</p>
      <div className="row wrap" style={{ margin: '0 0 12px' }}>
        <button className="btn sm" onClick={onFit}>
          <Icon name="reset" size={14} />
          Fit the map to the outlook
        </button>
        <button className="btn sm" onClick={onReplay}>
          <Icon name="play" size={14} />
          Replay how it grew
        </button>
        <button className="btn sm" onClick={onJson}>
          <Icon name="download" size={14} />
          Save the outlook as JSON
        </button>
      </div>
      <div className="dim small" style={{ minHeight: 18, margin: '-4px 0 8px' }}>
        {replayLabel}
      </div>

      <WindLive ev={ev} W={W} h={h} plume={plume} spread={spread} />

      {terrain && (
        <Card title="Ground along the wind line" info="elev" right="Hover to see it on the map" style={{ marginTop: 12 }}>
          <ElevChart terrain={terrain} cls={ev.cls} onHover={onHoverElev} />
          <p className="dim small" style={{ margin: '6px 0 0' }}>
            The dashed line on the map. Uphill runs speed a fire up, downhill runs slow it.
          </p>
        </Card>
      )}

      <details className="card" style={{ marginTop: 12 }}>
        <summary className="cbody" style={{ cursor: 'pointer', fontWeight: 600 }}>
          How this is estimated
        </summary>
        <div className="cbody" style={{ paddingTop: 0 }}>
          <ul className="list small">
            <li>
              <b>Wind, humidity and rain</b> come from a weather forecast. Here they are sample values shaped like a real forecast.
            </li>
            <li>
              <b>Smoke plume:</b> a parcel of smoke drifts with the forecast wind, hour by hour. The outline widens with distance, and the
              paler edge grows with forecast time to allow for wind error.
            </li>
            <li>
              <b>Fire spread:</b> the fire weather indices (FFMC, ISI, BUI, FWI) come from the standard Canadian system. Spread speed grows
              with ISI and with uphill slope, and the shape gets longer as the wind picks up. It is a rough guide, not a validated forecast.
            </li>
            <li>Nothing here has been checked against real fires yet.</li>
          </ul>
        </div>
      </details>
    </>
  );
}
