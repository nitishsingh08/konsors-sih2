/* =====================================================================
   The priority queue: one row per place, most urgent first.
   Port of qItem()/renderSide() in frontend/js/ui.js.
   ===================================================================== */
import { ago } from '../lib/time';
import { ccol, cv } from '../lib/core';
import { Empty, EventChips, Ring } from '../components/ui/Primitives';
import { sev, type Group } from '../state/selectors';

export function QueueItem({ g, current, onSelect }: { g: Group; current: boolean; onSelect: (id: string) => void }) {
  const e = g.top;
  const p = g.p;
  return (
    <button className="qi" aria-current={current} onClick={() => onSelect(p.id)}>
      <span>
        <Ring v={e.conf} size={40} sw={4} color={e.review ? cv('--warn') : ccol(e.cls)} />
      </span>
      <span style={{ minWidth: 0 }}>
        <span className="t" style={{ display: 'block' }}>
          {p.name}
        </span>
        <span className="s" style={{ display: 'block' }}>
          {p.district}, {p.state}
        </span>
        <span className="chips">
          <EventChips ev={e} />
        </span>
      </span>
      <span className="m">
        <b className="num">{e.frp.toFixed(0)} MW</b>
        <span>{ago(e.t)}</span>
        {g.evs.length > 1 && <span>{g.evs.length} events</span>}
      </span>
    </button>
  );
}

export function QueueList({
  groups,
  sel,
  onSelect,
  limit,
}: {
  groups: Group[];
  sel: string | null;
  onSelect: (id: string) => void;
  limit?: number;
}) {
  const sorted = groups.slice().sort((a, b) => sev(b.top) - sev(a.top) || b.top.t - a.top.t);
  if (!sorted.length) {
    return (
      <Empty title="Nothing matches these filters">
        Widen the date window, turn more classes on, or lower the minimum confidence.
      </Empty>
    );
  }
  const shown = limit ? sorted.slice(0, limit) : sorted;
  return (
    <div className="queue">
      {shown.map((g) => (
        <QueueItem key={g.p.id} g={g} current={sel === g.p.id} onSelect={onSelect} />
      ))}
    </div>
  );
}
