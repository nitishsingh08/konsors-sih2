import { CLS, STATUS_OPTIONS } from '../data/classes';
import { Glyph } from '../components/ui/Glyph';
import { useFilters } from '../state/useFilters';
import { useTip } from '../components/ui/Tooltip';

export function Filters() {
  const [f, set] = useFilters();
  return (
    <div className="filters">
      {CLS.map((c) => {
        const on = f.cls.has(c.id);
        return (
          <button
            key={c.id}
            className="fchip"
            aria-pressed={on}
            onClick={() => {
              const next = new Set(f.cls);
              if (next.has(c.id)) next.delete(c.id);
              else next.add(c.id);
              if (!next.size) CLS.forEach((x) => next.add(x.id));
              set({ cls: next });
            }}
            {...useTip(c.blurb)}
          >
            <Glyph cls={c.id} size={12} />
            {c.label}
          </button>
        );
      })}
      <span className="row gap4">
        <label className="dim small" htmlFor="fst">
          Status
        </label>
        <select id="fst" className="sel" aria-label="Status filter" value={f.status} onChange={(e) => set({ status: e.target.value as never })}>
          {STATUS_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </span>
      <span className="row gap4">
        <label className="dim small" htmlFor="fse">
          Sensor
        </label>
        <select id="fse" className="sel" aria-label="Sensor filter" value={f.sensor} onChange={(e) => set({ sensor: e.target.value as never })}>
          {option('all', 'All')}
          {option('VIIRS', 'VIIRS 375 m')}
          {option('MODIS', 'MODIS 1 km')}
        </select>
      </span>
      <label className="conf" htmlFor="fco">
        Minimum confidence
        <input
          id="fco"
          type="range"
          min={0}
          max={90}
          step={5}
          value={Math.round(f.minConf * 100)}
          onChange={(e) => set({ minConf: Number(e.target.value) / 100 })}
        />
        <b className="num" style={{ minWidth: 34 }}>
          {Math.round(f.minConf * 100)}%
        </b>
      </label>
    </div>
  );
}

function option(v: string, l: string) {
  return (
    <option key={v} value={v}>
      {l}
    </option>
  );
}
