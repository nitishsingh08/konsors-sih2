/* =====================================================================
   Who is in the way: what sits inside the smoke and spread zones.
   Port of tabExposure() in frontend/js/analysis/tabs.js.
   ===================================================================== */
import { exposureOf } from '../../data/analysis';
import { Stat } from '../../components/ui/Primitives';
import { Sec } from '../parts';
import type { Event, Plume, SpreadOutlook } from '../../lib/types';

export function ExposureTab({ ev, plume, spread }: { ev: Event; plume: Plume; spread: SpreadOutlook | null }) {
  const zones = exposureOf(ev, plume, spread);
  return (
    <>
      <div className="banner plain small" style={{ marginBottom: 12 }}>
        <span>
          <b>Sample counts.</b> In the live build these come from OpenStreetMap features inside each zone, and population from a gridded population
          dataset. OSM is patchy in rural India, so treat low numbers as a floor.
        </span>
      </div>
      {zones.map((z) => (
        <Sec key={z.name} title={z.name}>
          <div className="stats">
            <Stat label="Area" value={z.area.toFixed(1)} unit="km²" />
            <Stat label="Villages and towns" value={z.settlements} />
            <Stat label="Schools" value={z.schools} />
            <Stat label="Clinics and hospitals" value={z.clinics} />
            <Stat label="Main roads" value={z.roadKm} unit="km" />
            <Stat label="Water bodies" value={z.water} />
          </div>
        </Sec>
      ))}
      <p className="dim small" style={{ marginTop: 12 }}>
        The smoke zone is where smoke can travel, so it says who might smell it, not who is at risk. It says nothing about how thick the smoke is.
      </p>
    </>
  );
}
