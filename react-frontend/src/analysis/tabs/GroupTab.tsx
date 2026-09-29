/* =====================================================================
   The nine feature-group sections: tiles for the headline values, one
   chart, the plain-language reads and the full table.
   Port of tabGroup()/readsFor()/groupChart() in frontend/js/analysis/tabs.js.
   ===================================================================== */
import type { ReactNode } from 'react';
import { alpha, ccol, tk } from '../../lib/core';
import { compass, M_LAT, mLon } from '../../data/weather';
import { FBY, footprintsFor } from '../../data/features';
import { FAMILIES } from '../../data/classes';
import { freshnessOf } from '../../data/analysis';
import { Chart, axisBox, tooltipBox } from '../../components/ui/Chart';
import { Meter } from '../../components/ui/Primitives';
import { Compass, FTiles, GroupTable, Reads, Sec, fv, fs, pctWord } from '../parts';
import { tile } from './WindTab';
import type { Event, FamilyId, FeatureVector, Place } from '../../lib/types';

const GROUP_TILES: Record<string, string[]> = {
  thermal: ['FRP_MEDIAN', 'FRP_MAX', 'FRP_P90', 'BT_I4_MEDIAN', 'BT_I4_I5_DIFF_MEDIAN', 'SUBPIXEL_TEMP_MEDIAN', 'SOURCE_AREA_MEDIAN', 'FRP_SLOPE', 'FRP_DECAY_RATE'],
  temporal: ['DURATION_HOURS_LOG', 'DETECTION_COUNT_LOG', 'OVERPASS_DETECTION_RATIO', 'RECURRENCE_RATE_1Y', 'MAX_INTER_EVENT_GAP_LOG', 'TIME_TO_PEAK'],
  spread: ['SPREAD_SPEED_MEDIAN', 'SPREAD_SPEED_P90', 'STATIONARITY_INDEX', 'AREA_GROWTH_RATE', 'ELONGATION', 'NEW_PIXEL_FRACTION'],
  terrain: ['ELEVATION_MEDIAN', 'SLOPE_MEDIAN', 'SLOPE_P90', 'TRI', 'TPI', 'RIDGE_VALLEY_POSITION'],
  fuel: ['NDVI_MEDIAN', 'NDVI_ANOMALY_Z', 'NDMI_ANOMALY_Z', 'NBR_CHANGE', 'LANDCOVER_ENTROPY', 'WATER_FRAC_1KM'],
  industry: ['DIST_NEAREST_FACILITY_LOG', 'INSIDE_FACILITY', 'KNOWN_FLARE', 'FACILITY_COUNT_1KM', 'FACILITY_COUNT_5KM', 'FACILITY_COUNT_10KM', 'INDUSTRIAL_DENSITY_CONTRAST', 'DIST_FACILITY_BOUNDARY_LOG'],
  weather: ['TEMPERATURE_ANOMALY_Z', 'RH_ANOMALY_Z', 'VPD_ANOMALY_Z', 'SOIL_MOISTURE_ANOMALY_Z', 'WIND_SPEED', 'WIND_GUST', 'RAIN_24H_LOG', 'RAIN_7D_LOG', 'LIGHTNING_COUNT_24H'],
  quality: ['FIRMS_CONFIDENCE', 'PIXEL_SIZE', 'CLOUD_FRACTION', 'MISSING_FEATURE_FRACTION'],
};

/* ---- the chart that goes with each group ---- */
function groupChart(gid: FamilyId, p: Place, ev: Event, F: FeatureVector) {
  const T = tk();
  const col = ccol(ev.cls);

  if (gid === 'thermal') {
    const peak = fv(F, 'FRP_MAX') || 10;
    const tp = fv(F, 'TIME_TO_PEAK') || 6;
    const dec = fv(F, 'FRP_DECAY_RATE') || 0.06;
    const med = fv(F, 'FRP_MEDIAN') || peak * 0.5;
    const d: [number, number][] = [];
    for (let h = 0; h <= 48; h++) {
      d.push([h, +(h <= tp ? med * 0.3 + (peak - med * 0.3) * Math.pow(h / Math.max(tp, 0.5), 1.4) : peak * Math.exp(-dec * (h - tp))).toFixed(1)]);
    }
    return (
      <Sec title="Heat over time" right="Sketch">
        <Chart
          option={{
            grid: { left: 44, right: 12, top: 12, bottom: 24 },
            tooltip: { ...tooltipBox(), trigger: 'axis' },
            xAxis: { type: 'value', name: 'hours', nameLocation: 'end', nameTextStyle: { color: T.dim, fontSize: 10 }, ...axisBox(), splitLine: { show: false } },
            yAxis: { type: 'value', name: 'MW', nameTextStyle: { color: T.dim, fontSize: 10, align: 'left' }, ...axisBox() },
            series: [
              {
                type: 'line', data: d, showSymbol: false, smooth: 0.3, lineStyle: { color: col, width: 2 }, areaStyle: { color: alpha(col, 0.16) },
                markLine: { silent: true, symbol: 'none', label: { formatter: 'peak', color: T.dim, fontSize: 10 }, lineStyle: { color: T.dim, type: 'dashed' }, data: [{ xAxis: tp }] },
              },
            ],
          }}
          height={170}
        />
        <p className="dim small" style={{ margin: '4px 0 0' }}>
          A sketch drawn from the peak, time to peak and decay rate. Not the raw passes.
        </p>
      </Sec>
    );
  }

  if (gid === 'temporal') {
    const v = ['RECURRENCE_RATE_1Y', 'RECURRENCE_RATE_3Y', 'RECURRENCE_RATE_5Y'].map((k) => fv(F, k));
    return (
      <Sec title="How often it comes back" right="Share of windows">
        <Chart
          option={{
            grid: { left: 40, right: 12, top: 8, bottom: 24 },
            tooltip: { ...tooltipBox(), trigger: 'axis', formatter: (q: { name: string; data: number | null }) => `${q.name}: ${q.data == null ? 'no data' : Math.round(q.data * 100) + '%'}` },
            xAxis: { type: 'category', data: ['1 year', '3 years', '5 years'], ...axisBox(), splitLine: { show: false } },
            yAxis: { type: 'value', min: 0, max: 1, axisLabel: { color: T.dim, fontSize: 10, formatter: (x: number) => Math.round(x * 100) + '%' }, ...axisBox() },
            series: [{ type: 'bar', barWidth: '42%', data: v, itemStyle: { color: col, borderRadius: [3, 3, 0, 0] } }],
          }}
          height={150}
        />
      </Sec>
    );
  }

  if (gid === 'spread') {
    const fr = footprintsFor(p, ev);
    const lat0 = p.lat;
    const cen = fr.map((f) => {
      const lo = f.ring.reduce((a, q) => a + q[0], 0) / f.ring.length;
      const la = f.ring.reduce((a, q) => a + q[1], 0) / f.ring.length;
      return [+(((lo - p.lon) * mLon(lat0)) / 1000).toFixed(2), +(((la - p.lat) * M_LAT) / 1000).toFixed(2)];
    });
    const option = {
        grid: { left: 44, right: 16, top: 12, bottom: 28 },
        tooltip: { ...tooltipBox(), formatter: (q: { dataIndex: number; data: number[] }) => `Pass ${q.dataIndex + 1}<br>${q.data[0]} km east, ${q.data[1]} km north` },
        xAxis: { type: 'value', name: 'km east', nameLocation: 'middle', nameGap: 20, nameTextStyle: { color: T.dim, fontSize: 10 }, scale: true, ...axisBox() },
        yAxis: { type: 'value', name: 'km north', nameTextStyle: { color: T.dim, fontSize: 10, align: 'left' }, scale: true, ...axisBox() },
        series: [{ type: 'line', data: cen, symbolSize: 7, lineStyle: { color: col, width: 2 }, itemStyle: { color: col }, endLabel: { show: true, formatter: 'now', color: T.dim, fontSize: 10 } }],
    };
    return (
      <Sec title="Where the centre has moved" right="Last 8 passes">
        <Chart option={option} height={200} />
        <p className="dim small" style={{ margin: '4px 0 0' }}>
          Centre of the footprint on each of the last 8 passes, in km from the current position.
        </p>
      </Sec>
    );
  }

  if (gid === 'terrain') {
    const asp = F.ang.ASPECT ?? 0;
    const wind = ((F.ang.WIND_DIRECTION ?? 0) + 180) % 360;
    return (
      <Sec title="Which way the ground faces" right="Slope and wind">
        <div className="row" style={{ gap: 18, alignItems: 'center' }}>
          <Compass arrows={[{ deg: asp, color: col }, { deg: wind, color: 'var(--tx2)' }]} />
          <div className="small tx2" style={{ display: 'grid', gap: 6 }}>
            <span className="row">
              <i className="sw" style={{ background: col }} />
              Slope faces {compass(asp)}
            </span>
            <span className="row">
              <i className="sw" style={{ background: 'var(--tx2)' }} />
              Wind blows toward {compass(wind)}
            </span>
            <span className="dim">A fire spreads fastest when the wind pushes it uphill.</span>
          </div>
        </div>
      </Sec>
    );
  }

  if (gid === 'fuel') {
    const rows: [string, number, number, number][] = [
      ['375 m', fv(F, 'FOREST_FRAC_375M') || 0, 0, 0],
      ['1 km', fv(F, 'FOREST_FRAC_1KM') || 0, fv(F, 'CROPLAND_FRAC_1KM') || 0, fv(F, 'BUILTUP_FRAC_1KM') || 0],
      ['10 km', fv(F, 'FOREST_FRAC_10KM') || 0, fv(F, 'CROPLAND_FRAC_10KM') || 0, fv(F, 'BUILTUP_FRAC_10KM') || 0],
    ];
    return (
      <Sec title="Land cover at three scales" right="Share of area">
        <Chart
          option={{
            grid: { left: 52, right: 12, top: 6, bottom: 22 },
            tooltip: { ...tooltipBox(), trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: (v: number) => Math.round(v * 100) + '%' },
            xAxis: { type: 'value', max: 1, axisLabel: { color: T.dim, fontSize: 10, formatter: (x: number) => Math.round(x * 100) + '%' }, ...axisBox() },
            yAxis: { type: 'category', data: rows.map((r) => r[0]), inverse: true, ...axisBox(), splitLine: { show: false } },
            series: [
              ...(
                [
                  ['Forest', 1, '#4F8F6B'],
                  ['Cropland', 2, '#B9A45B'],
                  ['Built-up', 3, '#6F86A6'],
                ] as [string, number, string][]
              ).map(([n, i, c]) => ({
                name: n, type: 'bar', stack: 's', barWidth: 20, itemStyle: { color: c }, data: rows.map((r) => +Number(r[i]).toFixed(3)),
              })),
              { name: 'Other', type: 'bar', stack: 's', barWidth: 20, itemStyle: { color: T.line2 }, data: rows.map((r) => +Math.max(0, 1 - r[1] - r[2] - r[3]).toFixed(3)) },
            ],
          }}
          height={140}
        />
        <div className="legendrow">
          <span>
            <i className="sw" style={{ background: '#4F8F6B' }} />
            Forest
          </span>
          <span>
            <i className="sw" style={{ background: '#B9A45B' }} />
            Cropland
          </span>
          <span>
            <i className="sw" style={{ background: '#6F86A6' }} />
            Built-up
          </span>
          <span>
            <i className="sw" style={{ background: 'var(--line2)' }} />
            Other
          </span>
        </div>
        <p className="dim small" style={{ margin: '6px 0 0' }}>
          At 375 m only forest is measured.
        </p>
      </Sec>
    );
  }

  if (gid === 'industry') {
    const keys = (
      [
        'DIST_REFINERY_LOG', 'DIST_POWERPLANT_LOG', 'DIST_PETROCHEMICAL_LOG', 'DIST_STEEL_LOG', 'DIST_CEMENT_LOG',
        'DIST_LNG_LOG', 'DIST_MINE_LOG', 'DIST_LANDFILL_LOG', 'DIST_PIPELINE_LOG', 'DIST_WELLPAD_LOG',
      ] as const
    )
      .map((k) => ({ k, l: FBY[k].label.replace('Nearest ', ''), v: fv(F, k) }))
      .filter((x) => x.v != null)
      .map((x) => ({ l: x.l, v: x.v as number }))
      .sort((a, b) => b.v - a.v);
    return (
      <Sec title="Distance to each kind of facility" right="km">
        <Chart
          option={{
            grid: { left: 118, right: 22, top: 6, bottom: 22 },
            tooltip: { ...tooltipBox(), formatter: (q: { name: string; value: number }) => `${q.name}: ${q.value < 10 ? q.value.toFixed(1) : Math.round(q.value)} km` },
            xAxis: { type: 'log', min: 0.1, ...axisBox() },
            yAxis: { type: 'category', data: keys.map((x) => x.l), ...axisBox(), splitLine: { show: false }, axisLabel: { color: T.tx2, fontSize: 11 } },
            series: [
              {
                type: 'bar', barWidth: 11,
                data: keys.map((x, i) => ({ value: Math.max(0.1, x.v), itemStyle: { color: i === keys.length - 1 ? col : T.dim, borderRadius: [0, 3, 3, 0] } })),
              },
            ],
          }}
          height={60 + keys.length * 24}
        />
        <p className="dim small" style={{ margin: '4px 0 0' }}>
          Log scale, so the near ones stay readable.
        </p>
      </Sec>
    );
  }

  if (gid === 'weather') {
    const isi = fv(F, 'ISI');
    const rows: [string, number | null, number, number, number, number][] = [
      ['Fuel moisture (FFMC)', fv(F, 'FFMC'), 30, 99, FBY.FFMC.mu, FBY.FFMC.sd],
      ['Duff moisture (DMC)', fv(F, 'DMC'), 0, 200, FBY.DMC.mu, FBY.DMC.sd],
      ['Drought (DC)', fv(F, 'DC'), 0, 700, FBY.DC.mu, FBY.DC.sd],
      ['Spread index (ISI)', isi, 0, 30, FBY.ISI.mu, FBY.ISI.sd],
      ['Build-up (BUI)', fv(F, 'BUI'), 0, 200, FBY.BUI.mu, FBY.BUI.sd],
      ['Fire weather (FWI)', fv(F, 'FWI'), 0, 60, FBY.FWI.mu, FBY.FWI.sd],
    ];
    return (
      <Sec title="The fire weather system" right="Canadian FWI system" tip="fwi">
        <div className="meters">
          {rows.map((r) => (r[1] == null ? null : <Meter key={r[0]} label={r[0]} value={r[1]} lo={r[2]} hi={r[3]} mu={r[4]} sd={r[5]} fmt={(v) => v.toFixed(v < 10 ? 1 : 0)} />))}
        </div>
        <p className="dim small" style={{ margin: '6px 0 0' }}>
          The pale band is the usual range. The marker is this event.
        </p>
      </Sec>
    );
  }

  if (gid === 'quality') {
    const rows = FAMILIES.map((g) => {
      const l = Object.values(FBY).filter((d) => d.g === g.id);
      const ok = l.filter((d) => !F.by[d.key]?.missing).length;
      return { g, ok, n: l.length };
    });
    return (
      <Sec title="How complete the inputs are" right="Features available">
        <div style={{ display: 'grid', gap: 8 }}>
          {rows.map((r) => (
            <div key={r.g.id} style={{ display: 'grid', gridTemplateColumns: '170px 1fr 56px', gap: 8, alignItems: 'center' }}>
              <span className="tx2">{r.g.label}</span>
              <span className="tr" style={{ height: 8, background: 'var(--raised)', borderRadius: 99, overflow: 'hidden' }}>
                <i style={{ display: 'block', height: '100%', width: `${(r.ok / r.n) * 100}%`, background: r.ok === r.n ? 'var(--tx)' : r.ok / r.n > 0.6 ? 'var(--tx2)' : 'var(--warn)' }} />
              </span>
              <span className="num tx2" style={{ textAlign: 'right' }}>
                {r.ok} of {r.n}
              </span>
            </div>
          ))}
        </div>
      </Sec>
    );
  }

  return null;
}

/* ---- the plain-language reads for each group ---- */
function readsFor(gid: FamilyId, F: FeatureVector): ReactNode[] {
  const n = (k: string) => fv(F, k);
  const r: ReactNode[] = [];
  const pctOf = (k: string) => (F.by[k]?.pct == null ? null : pctWord(F.by[k]));

  if (gid === 'thermal') {
    r.push(
      <>
        Median fire power is {fs(F, 'FRP_MEDIAN')} and the peak is {fs(F, 'FRP_MAX')}.{' '}
        {pctOf('FRP_MEDIAN') != null && `That is ${pctOf('FRP_MEDIAN')}.`}
      </>,
    );
    if (n('BT_I4_I5_DIFF_MEDIAN') != null) {
      r.push(
        n('BT_I4_I5_DIFF_MEDIAN')! > 48
          ? 'The gap between the 4 µm and 11 µm bands is wide. That usually means a small, very hot source such as a flare or furnace.'
          : n('BT_I4_I5_DIFF_MEDIAN')! < 30
            ? 'The two infrared bands sit close together, which is more like cooler, spread-out burning such as a vegetation fire.'
            : 'The two infrared bands are in the middle range, so this does not point clearly either way.',
      );
    }
    if (n('SUBPIXEL_TEMP_MEDIAN') != null && n('SUBPIXEL_TEMP_MEDIAN')! > 1300) {
      r.push(`The hot part is estimated at ${fs(F, 'SUBPIXEL_TEMP_MEDIAN')}, hotter than most vegetation fires.`);
    }
    if (n('FRP_SLOPE') != null) {
      r.push(n('FRP_SLOPE')! > 0.8 ? 'Fire power is climbing between passes.' : n('FRP_SLOPE')! < -0.8 ? 'Fire power is fading between passes.' : 'Fire power is roughly steady between passes.');
    }
  }

  if (gid === 'temporal') {
    r.push(
      n('RECURRENCE_RATE_1Y') != null ? (
        <>
          It has shown up again in {fs(F, 'RECURRENCE_RATE_1Y')} of the past year's windows and {fs(F, 'RECURRENCE_RATE_3Y')} over three years.{' '}
          {n('RECURRENCE_RATE_1Y')! > 0.75 ? 'Coming back this regularly is what a fixed source does.' : n('RECURRENCE_RATE_1Y')! < 0.2 ? 'It rarely returns here, which fits a one-off fire.' : ''}
        </>
      ) : null,
    );
    r.push(`It has burned for about ${fs(F, 'DURATION_HOURS_LOG')} and was seen on ${fs(F, 'OVERPASS_DETECTION_RATIO')} of passes.`);
    if (n('NIGHT_DAY_FRP_RATIO') != null) {
      r.push(
        n('NIGHT_DAY_FRP_RATIO')! > 1.15
          ? 'It burns at least as hot at night as by day, which is typical of industrial sources.'
          : n('NIGHT_DAY_FRP_RATIO')! < 0.6
            ? 'It is much weaker at night, which is typical of fires that follow the daily heating cycle.'
            : '',
      );
    }
  }

  if (gid === 'spread') {
    r.push(
      n('STATIONARITY_INDEX') != null
        ? n('STATIONARITY_INDEX')! > 0.85
          ? 'It stays put. The centre barely moves between passes.'
          : n('STATIONARITY_INDEX')! < 0.45
            ? 'It is moving. The centre shifts noticeably between passes.'
            : 'It moves a little between passes.'
        : '',
    );
    r.push(`Typical spread speed is ${fs(F, 'SPREAD_SPEED_MEDIAN')} and the fastest is ${fs(F, 'SPREAD_SPEED_MAX')}.`);
    if (n('DIRECTION_CONSISTENCY') != null && n('STATIONARITY_INDEX')! < 0.6) {
      r.push(n('DIRECTION_CONSISTENCY')! > 0.6 ? `It keeps heading one way, toward the ${compass(F.ang.SPREAD_DIRECTION ?? 0)}.` : 'It wanders, with no steady direction.');
    }
  }

  if (gid === 'terrain') {
    r.push(`The ground is about ${fs(F, 'ELEVATION_MEDIAN')} above sea level with a typical slope of ${fs(F, 'SLOPE_MEDIAN')}. The steeper sides reach ${fs(F, 'SLOPE_P90')}.`);
    if (n('SLOPE_P90')! > 15) r.push('Slopes this steep make uphill fire runs much faster, so the wind direction against the slope matters here.');
    if (n('RIDGE_VALLEY_POSITION') != null) {
      r.push(
        n('RIDGE_VALLEY_POSITION')! > 0.3
          ? 'It sits on or near a ridge, where wind is stronger and fire spreads more freely.'
          : n('RIDGE_VALLEY_POSITION')! < -0.3
            ? 'It sits in a valley, where smoke can pool and wind can funnel along the floor.'
            : '',
      );
    }
  }

  if (gid === 'fuel') {
    r.push(`Within 1 km the land is ${fs(F, 'FOREST_FRAC_1KM')} forest, ${fs(F, 'CROPLAND_FRAC_1KM')} cropland and ${fs(F, 'BUILTUP_FRAC_1KM')} built-up.`);
    if (n('NDVI_ANOMALY_Z') != null) {
      r.push(
        n('NDVI_ANOMALY_Z')! < -1
          ? 'The vegetation is browner than normal for this time of year, so it is drier fuel.'
          : n('NDVI_ANOMALY_Z')! > 1
            ? 'The vegetation is greener than normal, which slows burning.'
            : 'The vegetation is close to normal for the season.',
      );
    } else {
      r.push('Imagery is missing here (likely cloud), so vegetation readings are not available.');
    }
    if (n('NBR_CHANGE') != null && n('NBR_CHANGE')! < -0.15) r.push('The burn ratio has dropped, which is a sign of burnt ground.');
  }

  if (gid === 'industry') {
    r.push(
      `The nearest facility is ${fs(F, 'DIST_NEAREST_FACILITY_LOG')} away and ${F.by.INSIDE_FACILITY?.v ? 'this point is inside a facility boundary' : 'this point is outside any known boundary'}.`,
    );
    if (F.by.KNOWN_FLARE?.v) r.push('It matches a known flare in the VIIRS Nightfire catalogue.');
    r.push(`${fs(F, 'FACILITY_COUNT_1KM')} facilities within 1 km, ${fs(F, 'FACILITY_COUNT_10KM')} within 10 km.`);
    r.push(
      <span className="dim">
        Facility and flare matches also help make the training labels, so the model can partly read its own answer back. Check the call against
        the other groups before trusting it.
      </span>,
    );
  }

  if (gid === 'weather') {
    const fwz = n('FWI_ANOMALY_Z');
    r.push(
      `Fire weather index is ${fs(F, 'FWI')}, ${
        fwz != null ? (fwz > 1 ? 'well above what is normal for now' : fwz < -1 ? 'below normal for now' : 'close to normal for now') : ''
      }.`,
    );
    r.push(`Wind is ${fs(F, 'WIND_SPEED')} with gusts to ${fs(F, 'WIND_GUST')}. Rain in the last 24 hours: ${fs(F, 'RAIN_24H_LOG')}.`);
    if (n('VPD_ANOMALY_Z')! > 1.2) r.push('The air is much drier than normal, which dries fuel quickly.');
  }

  if (gid === 'quality') {
    r.push(`FIRMS rates this detection at ${fs(F, 'FIRMS_CONFIDENCE')} confidence. One pixel covers ${fs(F, 'PIXEL_SIZE')}.`);
    if (n('CLOUD_FRACTION')! > 0.5) r.push('Cloud covers more than half the area, so imagery-based inputs are weak or missing.');
    r.push(`${Math.round(F.missing)} of ${Object.keys(FBY).length} features are missing for this event.`);
  }

  return r;
}

export function GroupTab({ gid, p, ev, F }: { gid: FamilyId; p: Place; ev: Event; F: FeatureVector }) {
  const keys = GROUP_TILES[gid] ?? [];
  return (
    <>
      {keys.length > 0 && <FTiles keys={keys} F={F} />}
      {groupChart(gid, p, ev, F)}
      <Reads items={readsFor(gid, F)} />
      {gid === 'quality' && <QualityExtras ev={ev} F={F} />}
      <GroupTable gid={gid} F={F} />
    </>
  );
}

function QualityExtras({ ev, F }: { ev: Event; F: FeatureVector }) {
  const fr = freshnessOf(ev);
  return (
    <>
      <Sec title="How fresh each source is" right="Sample ages">
        <table className="tbl small">
          <tbody>
            {fr.map((f) => (
              <tr key={f.name}>
                <td>
                  {f.name}
                  <div className="dim fk">{f.note}</div>
                </td>
                <td className="r">{f.label}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Sec>
      <Sec title="Viewing geometry">
        <div className="stats">
          {tile('Scan angle', Math.abs(F.ang.SCAN_ANGLE ?? 0).toFixed(0) + '°', 'from straight down')}
          {tile('Viewing angle', (F.ang.VIEW_ZENITH ?? 0).toFixed(0) + '°', (F.ang.VIEW_ZENITH ?? 0) > 50 ? 'far from the centre of the pass' : 'close to the centre of the pass')}
          {tile('Sensor', ev.sensor.replace('VIIRS ', 'V ').replace('MODIS ', 'M '))}
        </div>
      </Sec>
    </>
  );
}
