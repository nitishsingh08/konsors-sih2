/* =====================================================================
   The 141-feature explorer: search, filter, sort and export every input
   for the selected event.
   Port of tabExplorer()/exRows()/exCsv() in frontend/js/analysis/tabs.js.
   ===================================================================== */
import { useMemo } from 'react';
import { ccol } from '../../lib/core';
import { csvCell } from '../../data/analysis';
import { FDEF, FSRC_TIP, fmtFeat, shapOf } from '../../data/features';
import { FAMILIES } from '../../data/classes';
import { Icon } from '../../components/ui/Icon';
import { MiniBar } from '../../components/ui/Primitives';
import { useTip } from '../../components/ui/Tooltip';
import { useFiles } from '../../lib/files';
import type { Event, FeatureDef, FeatureVector } from '../../lib/types';

export type ExplorerSort = 'pct' | 'contrib' | 'name' | 'group';
export type ExplorerOnly = 'all' | 'missing' | 'linked' | 'strong';

export interface ExplorerState {
  q: string;
  g: string;
  only: ExplorerOnly;
  sort: ExplorerSort;
}

export const initialExplorer: ExplorerState = { q: '', g: 'all', only: 'all', sort: 'pct' };

export function ExplorerTab({ ev, F, state, onChange }: {
  ev: Event;
  F: FeatureVector;
  state: ExplorerState;
  onChange: (s: ExplorerState) => void;
}) {
  const { saveFile } = useFiles();
  const sh = shapOf(ev);
  const col = ccol(ev.cls);

  const rows = useMemo(() => {
    const q = state.q.trim().toLowerCase();
    const list = FDEF.filter(
      (d) =>
        (state.g === 'all' || d.g === state.g) &&
        (!q || (d.label + ' ' + d.key + ' ' + d.src).toLowerCase().includes(q)) &&
        (state.only === 'all' ||
          (state.only === 'missing' ? !!F.by[d.key]?.missing : state.only === 'linked' ? d.fl.includes('k') : Math.abs(sh.all[d.key]) >= 0.05)),
    );
    const keyf = (d: FeatureDef): number => {
      switch (state.sort) {
        case 'contrib':
          return Math.abs(sh.all[d.key]);
        case 'group':
          return FAMILIES.findIndex((f) => f.id === d.g);
        default:
          return F.by[d.key]?.pct == null || F.by[d.key]?.missing ? -1 : Math.abs(F.by[d.key]!.pct! - 50);
      }
    };
    const sorted = [...list];
    if (state.sort === 'name') sorted.sort((a, b) => a.label.localeCompare(b.label));
    else sorted.sort((a, b) => keyf(b) - keyf(a));
    return sorted;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state, F, sh]);

  const exportCsv = () => {
    const csv = ['key,group,label,value,percentile,source,missing,feeds_labels,contribution']
      .concat(
        FDEF.map((d) => {
          const e = F.by[d.key];
          const v = e.missing || e.v == null ? '' : typeof e.v === 'number' ? +e.v.toFixed(4) : e.v;
          return [d.key, d.g, d.label, v, e.pct == null ? '' : e.pct.toFixed(1), d.src, e.missing, d.fl.includes('k'), e.missing ? '' : sh.all[d.key].toFixed(3)]
            .map(csvCell)
            .join(',');
        }),
      )
      .join('\n');
    saveFile(`agni-netra-${ev.id}-features.csv`, csv, 'CSV');
  };

  return (
    <>
      <div className="row wrap" style={{ marginBottom: 10 }}>
        <div className="search" style={{ flex: 1, minWidth: 150 }}>
          <span className="ic">
            <Icon name="search" size={16} />
          </span>
          <input
            id="exq"
            placeholder="Search features"
            aria-label="Search features"
            value={state.q}
            onChange={(e) => onChange({ ...state, q: e.target.value })}
          />
        </div>
        <select className="sel" aria-label="Group" value={state.g} onChange={(e) => onChange({ ...state, g: e.target.value })}>
          <option value="all">All groups</option>
          {FAMILIES.map((g) => (
            <option key={g.id} value={g.id}>
              {g.label}
            </option>
          ))}
        </select>
        <select className="sel" aria-label="Show" value={state.only} onChange={(e) => onChange({ ...state, only: e.target.value as ExplorerOnly })}>
          <option value="all">Show all</option>
          <option value="missing">Only missing</option>
          <option value="linked">Only those that feed labels</option>
          <option value="strong">Only strong contributors</option>
        </select>
        <select className="sel" aria-label="Sort" value={state.sort} onChange={(e) => onChange({ ...state, sort: e.target.value as ExplorerSort })}>
          <option value="pct">Most unusual first</option>
          <option value="contrib">Biggest contribution first</option>
          <option value="name">Name</option>
          <option value="group">Group</option>
        </select>
        <button className="btn sm" onClick={exportCsv}>
          <Icon name="download" size={14} />
          Export
        </button>
      </div>
      <p className="dim small" style={{ margin: '0 0 8px' }}>
        Each feature is compared with similar events. The last column is the sample contribution to the call.
      </p>
      <div className="exwrap">
        <table className="tbl small ex">
          <thead>
            <tr>
              <th>Feature</th>
              <th className="r">Value</th>
              <th>Against similar events</th>
              <th>Source</th>
              <th>Notes</th>
              <th className="r">Pushes the call</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((d) => {
              const e = F.by[d.key];
              const w = sh.all[d.key];
              const gl = FAMILIES.find((f) => f.id === d.g)?.label ?? '';
              return (
                <tr key={d.key} className={e?.missing ? 'miss' : ''}>
                  <td>
                    <div>{d.label}</div>
                    <div className="dim fk">
                      {d.key} <span className="gtag">{gl}</span>
                    </div>
                  </td>
                  <td className="r num">{e?.missing ? <span className="dim">no data</span> : fmtFeat(d, e?.v ?? null)}</td>
                  <td style={{ minWidth: 110 }}>
                    {e?.pct != null && !e.missing ? (
                      <>
                        <MiniBar value={e.pct / 100} />
                        <div className="u sm2">{Math.round(e.pct)}th pct</div>
                      </>
                    ) : (
                      <span className="dim">n/a</span>
                    )}
                  </td>
                  <td className="small">{d.src}</td>
                  <td>
                    {e?.missing && <span className="chip rev">Missing</span>}
                    {d.fl.includes('k') && (
                      <span className="chip prop" {...useTip(FSRC_TIP)}>
                        Feeds labels
                      </span>
                    )}
                  </td>
                  <td className="r num" style={{ color: w >= 0.05 ? col : w <= -0.05 ? 'var(--tx2)' : 'var(--dim)' }}>
                    {e?.missing ? '' : `${w >= 0 ? '+' : ''}${w.toFixed(2)}`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}
