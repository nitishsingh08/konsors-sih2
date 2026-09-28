/* =====================================================================
   Smoke, radar and air chemistry in one section.
   Port of tabAir() in frontend/js/analysis/tabs.js.
   ===================================================================== */
import { ccol } from '../../lib/core';
import { freshnessOf } from '../../data/analysis';
import { compass } from '../../data/weather';
import { Compass, FTiles, GroupTable, Sec, ZRow, fv } from '../parts';
import type { Event, FeatureVector } from '../../lib/types';

export function AirTab({ ev, F }: { ev: Event; F: FeatureVector }) {
  const fr = freshnessOf(ev);
  const age = (n: string) => fr.find((f) => f.name.startsWith(n))?.label ?? '';
  const ang = F.ang;

  const smoke = (
    <>
      <FTiles keys={['SMOKE_PROBABILITY', 'DARK_SMOKE_PROBABILITY', 'LIGHT_SMOKE_PROBABILITY', 'SMOKE_LENGTH_LOG', 'BURN_SCAR_GROWTH_RATE']} F={F} />
      {!F.by.SMOKE_DIRECTION_SIN?.missing && (
        <div className="row" style={{ gap: 14, marginTop: 10, alignItems: 'center' }}>
          <Compass arrows={[{ deg: ang.SMOKE_DIRECTION ?? 0, color: ccol(ev.cls) }]} />
          <span className="small tx2">Smoke is drifting toward the {compass(ang.SMOKE_DIRECTION ?? 0)}.</span>
        </div>
      )}
    </>
  );

  const sar = <FTiles keys={['VV_CHANGE', 'VH_CHANGE', 'VV_VH_CHANGE', 'COHERENCE_LOSS', 'DNBR', 'DNDVI', 'STRUCTURAL_CHANGE_PROBABILITY']} F={F} />;

  const hints: string[] = [];
  const co = fv(F, 'CO_NO2_RATIO_LOG');
  if (co != null && co > 0.6) {
    hints.push('Carbon monoxide is high next to nitrogen dioxide. That often goes with smouldering, low-efficiency burning such as crop residue or vegetation.');
  }
  const so2 = fv(F, 'SO2_ANOMALY_Z');
  if (so2 != null && so2 > 1.2) hints.push('Sulphur dioxide is well above normal. That is more common near coal, smelting and some industrial burning.');
  const ch4 = fv(F, 'CH4_ANOMALY_Z');
  if (ch4 != null && ch4 > 1.2) hints.push('Methane is raised, which can go with gas flaring or venting.');

  const air = (
    <>
      <div style={{ display: 'grid', gap: 6 }}>
        {['NO2_ANOMALY_Z', 'SO2_ANOMALY_Z', 'CO_ANOMALY_Z', 'CH4_ANOMALY_Z', 'HCHO_ANOMALY_Z', 'AOD_ANOMALY_Z'].map((k) => (
          <ZRow key={k} F={F} k={k} cls={ev.cls} />
        ))}
      </div>
      <div style={{ marginTop: 10 }}>
        <FTiles keys={['CO_NO2_RATIO_LOG', 'SO2_NO2_RATIO_LOG', 'CH4_CO_RATIO_LOG']} F={F} />
      </div>
      {hints.length > 0 && (
        <>
          <ul className="reads" style={{ marginTop: 10 }}>
            {hints.map((h) => (
              <li key={h}>{h}</li>
            ))}
          </ul>
          <p className="dim small" style={{ margin: '6px 0 0' }}>
            These are hints, not proof. Other sources can produce the same readings.
          </p>
        </>
      )}
    </>
  );

  return (
    <>
      <Sec title="Smoke seen from above" right={age('Optical')} tip="smokeopt">
        {smoke}
      </Sec>
      <Sec title="Ground change, from radar" right={age('Radar')} tip="sar">
        {sar}
      </Sec>
      <Sec title="What is in the air" right={age('Air')} tip="chem">
        {air}
      </Sec>
      <GroupTable gid="smoke" F={F} />
      <GroupTable gid="sar" F={F} />
      <GroupTable gid="chem" F={F} />
    </>
  );
}
