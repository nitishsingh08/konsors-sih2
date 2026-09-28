import { useEffect, useRef } from 'react';
import { DAY, MONTHS, START, dparts, fmtD, fmtS } from '../lib/time';
import { EVENTS } from '../data/places';
import { passes } from '../state/selectors';
import { useFilters, WINDOWS, type Window } from '../state/useFilters';
import { Icon } from '../components/ui/Icon';

const W = 730;
const H = 42;
const DAYS = 365;

/** the 12-month histogram with a draggable "up to this date" handle */
export function Scrubber({
  play,
  onTogglePlay,
}: {
  play: boolean;
  onTogglePlay: () => void;
}) {
  const [f, set] = useFilters();
  const ref = useRef<HTMLInputElement>(null);

  const cnt = new Array(DAYS).fill(0);
  EVENTS.forEach((e) => {
    if (passes(f, e)) {
      const i = Math.floor((e.t - START) / DAY);
      if (i >= 0 && i < DAYS) cnt[i]++;
    }
  });
  const mx = Math.max(...cnt, 1);
  const bw = W / DAYS;
  const ai = (f.asOf - START) / DAY;
  const x1 = ai * bw;
  const x0 = Math.max(0, (ai - f.win) * bw);

  const bars = cnt.map((c, i) =>
    c ? (
      <rect
        key={i}
        x={(i * bw).toFixed(2)}
        y={H - 12 - Math.max(2, (c / mx) * (H - 12))}
        width={Math.max(1, bw - 0.5).toFixed(2)}
        height={Math.max(2, (c / mx) * (H - 12))}
        style={{ fill: 'var(--dim)' }}
      />
    ) : null,
  );
  const ticks = [];
  for (let i = 0; i < DAYS; i++) {
    const p = dparts(START + i * DAY);
    if (p.d === 1) {
      ticks.push(
        <text key={i} x={(i * bw).toFixed(1)} y={H - 1} style={{ fill: 'var(--dim)', fontSize: 9.5 }}>
          {MONTHS[p.m]}
        </text>,
      );
    }
  }

  /* keep the handle where the filters say it is, even while playing */
  useEffect(() => {
    if (ref.current && document.activeElement !== ref.current) ref.current.value = String(Math.round(ai));
  }, [ai]);

  return (
    <div className="scrub">
      <button className="btn ic" aria-label={play ? 'Pause' : 'Play'} onClick={onTogglePlay}>
        <Icon name={play ? 'pause' : 'play'} size={15} />
      </button>
      <div className="hist">
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
          <rect
            x={x0}
            y={0}
            width={Math.max(2, x1 - x0)}
            height={H - 12}
            style={{ fill: 'var(--tx)', fillOpacity: 0.12, stroke: 'var(--tx)', strokeOpacity: 0.55, strokeWidth: 1, vectorEffect: 'non-scaling-stroke' }}
          />
          {bars}
          {ticks}
        </svg>
        <input
          ref={ref}
          type="range"
          min={0}
          max={DAYS - 1}
          step={1}
          defaultValue={Math.round(ai)}
          aria-label="Show events up to this date"
          onChange={(e) => set({ asOf: START + Number(e.target.value) * DAY })}
        />
      </div>
      <div className="dl">
        <b className="num">
          {fmtS(f.asOf - f.win * DAY + DAY)} to {fmtD(f.asOf)}
        </b>
        <span className="seg" style={{ marginTop: 4 }}>
          {WINDOWS.map((w) => (
            <button key={w} aria-pressed={f.win === w} onClick={() => set({ win: w as Window })}>
              {w === 365 ? '1 y' : w + ' d'}
            </button>
          ))}
        </span>
      </div>
    </div>
  );
}
