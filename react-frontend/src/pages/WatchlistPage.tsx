/* =====================================================================
   Watchlist and site registry: every persistent site with its latest
   call, star-able and exportable.
   Port of pageWatch() and watchTable() in frontend/js/pages.js.
   ===================================================================== */
import { useMemo, useState } from 'react';
import { STATUS } from '../data/classes';
import { dparts, fmtS, TODAY, DAY } from '../lib/time';
import { EVP, PLACES, seriesOf } from '../data/places';
import { eventsCsv } from '../data/analysis';
import { useNav } from '../components/Shell';
import { Icon } from '../components/ui/Icon';
import { EventChips, Spark, Star } from '../components/ui/Primitives';
import { PageHead } from './PageHead';
import { useFiles } from '../lib/files';
import { useReviews } from '../state/ReviewsProvider';
import { useToast } from '../state/ToastProvider';
import type { Event, Place } from '../lib/types';

type SortKey = 'name' | 'cls' | 'status' | 'conf' | 't' | 'med';

const WST: { value: string; label: string }[] = [
  { value: 'all', label: 'All statuses' },
  { value: 'abnormal', label: 'Abnormal' },
  { value: 'routine', label: 'Routine' },
  { value: 'baseline_building', label: 'Baseline building' },
  { value: 'review', label: 'Needs review' },
  { value: 'watched', label: 'Starred only' },
];

export function WatchlistPage() {
  const { goPlace } = useNav();
  const { watch, toggleWatch } = useReviews();
  const { saveFile } = useFiles();
  const toast = useToast();
  const [q, setQ] = useState('');
  const [wst, setWst] = useState('all');
  const [sort, setSort] = useState<{ k: SortKey; dir: number }>({ k: 't', dir: -1 });

  const rows = useMemo(() => {
    const s = q.trim().toLowerCase();
    const list = PLACES.filter((p) => p.kind === 'site')
      .map((p) => {
        const l = (EVP[p.id] || []).filter((e) => e.t <= TODAY);
        return { p, e: l[l.length - 1] };
      })
      .filter((r): r is { p: Place; e: Event } => !!r.e)
      .filter(
        (r) =>
          (wst === 'all' ||
            (wst === 'review' ? r.e.review : wst === 'watched' ? watch.has(r.p.id) : r.e.status === wst)) &&
          (!s || (r.p.name + ' ' + r.p.state + ' ' + r.p.code).toLowerCase().includes(s)),
      );
    const val = (r: { p: Place; e: Event }): string | number => {
      switch (sort.k) {
        case 'name':
          return r.p.name;
        case 'cls':
          return r.e.cls;
        case 'status':
          return STATUS[r.e.status].rank;
        case 'conf':
          return r.e.conf;
        case 'med':
          return r.p.med;
        default:
          return r.e.t;
      }
    };
    return list.sort((a, b) => {
      const x = val(a);
      const y = val(b);
      return (x > y ? 1 : x < y ? -1 : 0) * sort.dir;
    });
  }, [q, wst, sort, watch]);

  const th = (k: SortKey, l: string, cls = '') => (
    <th className={cls}>
      <button
        onClick={() => setSort((s) => (s.k === k ? { k, dir: -s.dir } : { k, dir: k === 'name' || k === 'cls' ? 1 : -1 }))}
        aria-label={'Sort by ' + l}
      >
        {l}
        {sort.k === k ? (sort.dir > 0 ? ' ▲' : ' ▼') : ''}
      </button>
    </th>
  );

  return (
    <div className="page">
      <div className="page-in">
        <PageHead title="Watchlist and site registry" lead="Every persistent site the system tracks, with its latest call. Star places to follow them." />
        <div className="row wrap">
          <div className="search" style={{ maxWidth: 320 }}>
            <span className="ic">
              <Icon name="search" size={16} />
            </span>
            <input placeholder="Filter by name, state or code" aria-label="Filter places" value={q} onChange={(e) => setQ(e.target.value)} />
          </div>
          <select className="sel" aria-label="Status filter" value={wst} onChange={(e) => setWst(e.target.value)}>
            {WST.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
          <span style={{ flex: 1 }} />
          <button className="btn sm" onClick={() => saveFile('agni-netra-site-registry.csv', eventsCsv(rows.map((r) => r.e)), 'CSV')}>
            <Icon name="download" size={14} />
            Export registry
          </button>
        </div>

        <div className="card" style={{ overflow: 'auto' }}>
          <table className="tbl">
            <thead>
              <tr>
                <th style={{ width: 32 }} />
                {th('name', 'Place')}
                {th('cls', 'Class')}
                {th('status', 'Status')}
                {th('conf', 'Confidence', 'r')}
                {th('t', 'Last event')}
                {th('med', 'Median power', 'r')}
                <th>90-day trend</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(({ p, e }) => (
                <tr key={p.id} className="click" onClick={() => goPlace(p.id)}>
                  <td onClick={(ev) => ev.stopPropagation()}>
                    <Star
                      on={watch.has(p.id)}
                      size={16}
                      label={'Watch ' + p.name}
                      onClick={() => toast(toggleWatch(p.id) ? 'Added to watchlist' : 'Removed from watchlist')}
                    />
                  </td>
                  <td>
                    <b>{p.name}</b>
                    <div className="dim small">
                      {p.state}, {p.code}
                    </div>
                  </td>
                  <td>
                    <EventChips ev={e} />
                  </td>
                  <td className="r">{Math.round(e.conf * 100)}%</td>
                  <td>
                    {fmtS(e.t)} {dparts(e.t).y}
                  </td>
                  <td className="r">{p.med} MW</td>
                  <td style={{ color: 'var(--dim)' }}>
                    <Spark values={seriesOf(p).obs.filter((o) => o.t > TODAY - 90 * DAY).map((o) => o.frp).slice(-40)} w={96} h={24} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {rows.length === 0 && (
            <div className="empty" style={{ margin: 14 }}>
              <b>No places match</b>
              Clear the search or pick another status.
            </div>
          )}
          <div className="dim small" style={{ padding: 10 }}>
            {rows.length} places
          </div>
        </div>
      </div>
    </div>
  );
}
