/* =====================================================================
   Building blocks for the analysis panel: a feature tile, a titled
   section, the plain-language reads, the full group table and the
   compass rose.
   Ported from the helpers at the top of frontend/js/analysis/tabs.js.
   ===================================================================== */
import type { ReactNode } from 'react';
import { clamp, ccol } from '../lib/core';
import { Card, MiniBar } from '../components/ui/Primitives';
import { useTip } from '../components/ui/Tooltip';
import { FBY, FSRC_TIP, fmtFeat } from '../data/features';
import type { ClassId, FamilyId, FeatureDef, FeatureVector } from '../lib/types';

export const pctWord = (e: { pct: number | null; missing: boolean }): string =>
  e.pct == null || e.missing ? '' : e.pct >= 50 ? `higher than ${Math.round(e.pct)}% of similar events` : `lower than ${Math.round(100 - e.pct)}% of similar events`;

export function FTile({ d, e }: { d: FeatureDef; e: FeatureVector['by'][string] | undefined }) {
  if (!e) return null;
  return (
    <div className="stat">
      <div className="l">
        {d.label}
        {d.fl.includes('k') && (
          <span className="lk" {...useTip(FSRC_TIP)}>
            also feeds labels
          </span>
        )}
      </div>
      <div className="v">{e.missing ? <span className="dim">No data</span> : fmtFeat(d, e.v)}</div>
      {e.pct != null && !e.missing && (
        <>
          <MiniBar value={e.pct / 100} />
          <div className="u sm2">{pctWord(e)}</div>
        </>
      )}
    </div>
  );
}

export function FTiles({ keys, F }: { keys: string[]; F: FeatureVector }) {
  return (
    <div className="stats">
      {keys.map((k) => (
        <FTile key={k} d={FBY[k]} e={F.by[k]} />
      ))}
    </div>
  );
}

export function Sec({ title, children, right, tip, mt = 12 }: { title: string; children: ReactNode; right?: ReactNode; tip?: string; mt?: number }) {
  return (
    <Card title={title} info={tip} right={right} style={{ marginTop: mt }}>
      {children}
    </Card>
  );
}

export function Reads({ items, title = 'What this says' }: { items: ReactNode[]; title?: string }) {
  const a = items.filter(Boolean);
  if (!a.length) return null;
  return (
    <Sec title={title}>
      <ul className="reads">
        {a.map((s, i) => (
          <li key={i}>{s}</li>
        ))}
      </ul>
    </Sec>
  );
}

export function GroupTable({ gid, F }: { gid: FamilyId; F: FeatureVector }) {
  const list = Object.values(FBY).filter((d) => d.g === gid);
  return (
    <details className="card" style={{ marginTop: 12 }}>
      <summary className="cbody" style={{ cursor: 'pointer', fontWeight: 600 }}>
        All {list.length} features in this group
      </summary>
      <div className="cbody" style={{ paddingTop: 0 }}>
        <table className="tbl small">
          <tbody>
            {list.map((d) => {
              const e = F.by[d.key];
              return (
                <tr key={d.key}>
                  <td>
                    {d.label}
                    <div className="dim fk">{d.key}</div>
                  </td>
                  <td className="r num">{e?.missing ? <span className="dim">no data</span> : fmtFeat(d, e?.v ?? null)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </details>
  );
}

export function Compass({ arrows }: { arrows: { deg: number; color: string }[] }) {
  const s = 116;
  const c = s / 2;
  return (
    <svg width={s} height={s} viewBox={`0 0 ${s} ${s}`} role="img" aria-label="Compass">
      <circle cx={c} cy={c} r={46} fill="none" stroke="var(--line2)" />
      <circle cx={c} cy={c} r={2} fill="var(--dim)" />
      {['N', 'E', 'S', 'W'].map((t, i) => (
        <text
          key={t}
          x={c + Math.sin((i * Math.PI) / 2) * 55}
          y={c - Math.cos((i * Math.PI) / 2) * 55 + 4}
          textAnchor="middle"
          style={{ fill: 'var(--dim)', fontSize: 10 }}
        >
          {t}
        </text>
      ))}
      {arrows.map((a, i) => (
        <g key={i} transform={`rotate(${a.deg} ${c} ${c})`}>
          <line x1={c} y1={c + 10} x2={c} y2={c - 40} stroke={a.color} strokeWidth="2.4" strokeLinecap="round" />
          <path d={`M${c} ${c - 44} l-5 9 l10 0 z`} fill={a.color} />
        </g>
      ))}
    </svg>
  );
}

/** a feature value as a number, or null when it is missing */
export const fv = (F: FeatureVector, k: string): number | null => {
  const e = F.by[k];
  return e && !e.missing ? e.v : null;
};

export const fs = (F: FeatureVector, k: string): string => {
  const e = F.by[k];
  return e?.missing ? 'no data' : fmtFeat(e.d, e.v);
};

export function ZRow({ F, k, cls }: { F: FeatureVector; k: string; cls: ClassId }) {
  const e = F.by[k];
  const z = e?.missing ? 0 : clamp(e!.v ?? 0, -4, 6);
  const w = (Math.abs(z) / 6) * 50;
  const c = ccol(cls);
  return (
    <div className="shr" style={{ gridTemplateColumns: '150px 1fr 56px' }}>
      <span className="tx2">{e?.d.label.replace(' against normal', '')}</span>
      <span className="ax">
        <i
          style={
            e?.missing
              ? { display: 'none' }
              : z >= 0
                ? { left: '50%', width: `${w}%`, background: c }
                : { right: '50%', width: `${w}%`, background: 'var(--dim)' }
          }
        />
      </span>
      <span className="vv">{e?.missing ? 'no data' : `${z >= 0 ? '+' : ''}${z.toFixed(1)}`}</span>
    </div>
  );
}
