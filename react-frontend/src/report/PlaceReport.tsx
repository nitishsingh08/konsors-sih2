/* =====================================================================
   The place report: what the system thinks, why, how it behaves, and
   what an analyst can do about it.
   Ported from the placeReport() section of frontend/js/pages.js.
   ===================================================================== */
import type { ReactNode } from 'react';
import { ccol, clamp, cv, lc } from '../lib/core';
import { fmtD, fmtDT } from '../lib/time';
import { CLSMAP, FAMILIES } from '../data/classes';
import { EVP, contextOf, seriesOf, statsOf } from '../data/places';
import { featuresOf, shapOf } from '../data/features';
import { capXml } from '../data/analysis';
import { useFiles } from '../lib/files';
import { useReviews } from '../state/ReviewsProvider';
import { ClassBadge } from '../components/ui/Glyph';
import { Icon } from '../components/ui/Icon';
import { Banner, Bar, Card, Info, KV, ReviewChip, ReviewedChip, Ring, Shr, Star, StatusChip } from '../components/ui/Primitives';
import {
  ConditionsCard,
  FingerprintCard,
  FootprintCard,
  HistoryCard,
  ImageryCard,
  NearbyCard,
  OutlookCard,
  SurroundCard,
  TimelineCard,
} from './PlaceCards';
import type { Event, Place, PlaceSeries, PlaceStats } from '../lib/types';

/* ---------- the read in plain words ---------- */
function headlineOf(ev: Event): string {
  if (ev.status === 'abnormal') return 'Burning hotter than usual, and worth a look.';
  if (ev.review) return 'Not sure about this one. A person should take a look.';
  if (ev.status === 'baseline_building') return 'New here, so it is too early to say what normal looks like.';
  return {
    wildfire: 'A short-lived vegetation fire.',
    agricultural_burning: 'Looks like crop burning.',
    gas_flare: 'A steady flare, behaving the way it usually does.',
    industrial: 'Ordinary industrial heat. Nothing unusual today.',
    mining: 'Heat from the mine, in line with its history.',
    unknown: 'Hard to say what this is.',
  }[ev.cls];
}

function summaryText(p: Place, ev: Event, st: PlaceStats, ser: PlaceSeries): string {
  if (p.kind === 'transient') {
    return `This burned for a short spell: ${ev.nDet} detections over ${ev.area} km². Fires like this have no normal level to compare with, so the call rests on what is around it and how it behaves.`;
  }
  if (ev.status === 'baseline_building') {
    return `First spotted ${fmtD(p.firstSeen)}. There are only ${ser.obs.filter((o) => o.t <= ev.t).length} clear looks so far, so nothing is claimed yet about whether this is normal.`;
  }
  if (ev.status === 'abnormal') {
    return `Its fire power is ${ev.z!.toFixed(1)} steps above this site's usual level. That is unusual for this place, so a person should look at it.`;
  }
  return `A long-running ${lc(CLSMAP[ev.cls].label)} source. It has been active on ${st.d90} of the last 90 days and sits within its usual range${
    ev.z != null ? ` (${ev.z >= 0 ? '+' : ''}${ev.z.toFixed(1)})` : ''
  }.`;
}

/* ---------- the cards ---------- */
function HeaderCard({ p, ev, asOf, watched, onWatch, onOpenAdvanced, onCopyLink, onExportCsv }: {
  p: Place;
  ev: Event;
  asOf: number;
  watched: boolean;
  onWatch: () => void;
  onOpenAdvanced: () => void;
  onCopyLink: () => void;
  onExportCsv: () => void;
}) {
  const last = (EVP[p.id] || []).filter((e) => e.t <= asOf).pop() || ev;
  return (
    <section className="card">
      <div className="cbody">
        <div className="row between" style={{ alignItems: 'flex-start', gap: 12 }}>
          <div style={{ minWidth: 0 }}>
            <h2 style={{ margin: 0, fontSize: 18, fontWeight: 600 }}>{p.name}</h2>
            <div className="dim" style={{ marginTop: 2 }}>
              {p.district}, {p.state}
            </div>
          </div>
          <Star on={watched} onClick={onWatch} label={watched ? 'Remove from watchlist' : 'Add to watchlist'} />
        </div>
        <div className="row wrap" style={{ margin: '10px 0 12px' }}>
          <span className="chip">{p.type}</span>
          <StatusChip status={ev.status} />
          {ev.review && <ReviewChip />}
          <ReviewedChip eventId={ev.id} />
        </div>
        <KV
          items={[
            ['Coordinates', <span className="num">{p.lat.toFixed(3)}°N, {p.lon.toFixed(3)}°E</span>],
            [p.kind === 'site' ? 'Site code' : 'Kind', p.kind === 'site' ? p.code : 'Transient event, no fixed site'],
            ['First seen', fmtD(p.firstSeen)],
            ['Last event', fmtDT(last.t)],
            [
              'Facility record',
              p.kind === 'site' && p.cls !== 'unknown'
                ? 'Matched to a facility (sample)'
                : p.type === 'Landfill'
                  ? 'Matched to a landfill (sample)'
                  : 'No facility match',
            ],
          ]}
        />
        <div className="row wrap" style={{ marginTop: 12 }}>
          <button className="btn pri sm" onClick={onOpenAdvanced}>
            <Icon name="layers" size={14} />
            Advanced analysis
          </button>
          <button className="btn sm" onClick={onCopyLink}>
            <Icon name="copy" size={14} />
            Copy link
          </button>
          <button className="btn sm" onClick={onExportCsv}>
            <Icon name="download" size={14} />
            Export events
          </button>
        </div>
      </div>
    </section>
  );
}

function VerdictCard({ p, ev, st, ser }: { p: Place; ev: Event; st: PlaceStats; ser: PlaceSeries }) {
  const c = ccol(ev.cls);
  const c2 = ccol(ev.cls2);
  const rest = Math.max(0, 1 - ev.conf - ev.p2);
  return (
    <Card title="What we think this is" right={'Event ' + ev.id}>
      <p className="headline">{headlineOf(ev)}</p>
      <div className="verdict">
        <div style={{ color: 'var(--tx)' }}>
          <Ring v={ev.conf} size={96} sw={8} color={ev.review ? cv('--warn') : c} />
        </div>
        <div>
          <div style={{ fontSize: 19, fontWeight: 600, lineHeight: 1.2 }}>
            <ClassBadge cls={ev.cls} />
          </div>
          <div className="dim small" style={{ marginTop: 3 }}>
            Calibrated confidence <Info k="conf" />
          </div>
          <div className="dim small">{CLSMAP[ev.cls].blurb}</div>
        </div>
      </div>
      <div style={{ marginTop: 12 }}>
        <Bar label={CLSMAP[ev.cls].label} value={ev.conf} color={c} />
        <Bar label={CLSMAP[ev.cls2].label} value={ev.p2} color={c2} />
        <Bar label="Everything else" value={rest} color="var(--dim)" />
      </div>
      {ev.review && (
        <Banner icon={<Icon name="search" size={16} />}>
          <b>Needs human review.</b> Confidence is below 60%, so a person should check this before anyone acts on it.
        </Banner>
      )}
      {ev.status === 'abnormal' && (
        <Banner kind="abn">
          <span>
            <b>Abnormal for this site.</b> Robust z-score {ev.z!.toFixed(1)}, above the 3.5 threshold.
          </span>
        </Banner>
      )}
      <p className="sum">{summaryText(p, ev, st, ser)}</p>
    </Card>
  );
}

function WhyCard({ ev, shapAll, onToggleShap }: { ev: Event; shapAll: boolean; onToggleShap: () => void }) {
  const sh = shapOf(ev);
  const mx = Math.max(...FAMILIES.map((f) => Math.abs(sh.fam[f.id])), 0.01);
  const c = ccol(ev.cls);
  const rows = FAMILIES.map((f) => ({ f, v: sh.fam[f.id] }))
    .sort((a, b) => Math.abs(b.v) - Math.abs(a.v))
    .slice(0, 6)
    .map(({ f, v }) => <Shr key={f.id} label={f.label} v={v} mx={mx} color={c} />);
  return (
    <Card title="What drove that call" info="shap" right="Sample explanation">
      <div className="shap">{rows}</div>
      <button className="btn sm" style={{ marginTop: 12 }} onClick={onToggleShap}>
        {shapAll ? 'Hide top features' : 'Show top 8 features'}
      </button>
      {shapAll && (
        <div className="flist">
          {sh.feats.map((f) => (
            <div className="frow" key={f.k}>
              <span>{f.l}</span>
              <span className="fv">{f.v}</span>
              <span className="tnum" style={{ color: f.w >= 0 ? c : 'var(--tx2)' }}>
                {f.w >= 0 ? '+' : ''}
                {f.w.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}

function BaselineCard({ p, ev, ser }: { p: Place; ev: Event; ser: PlaceSeries }) {
  if (p.kind === 'transient') {
    return (
      <Card title="What is normal here" info="baseline" right={<StatusChip status="not_applicable" />}>
        <p className="dim" style={{ margin: 0 }}>
          Not applicable. Wildfires and crop burning are short events with no normal level, so they are judged on footprint, land cover and season
          instead.
        </p>
      </Card>
    );
  }
  const b = ser.base!;
  const nAt = ser.obs.filter((o) => o.t <= ev.t).length;
  const ok = nAt >= 20;
  const last = ser.obs[ser.obs.length - 1];
  const tile = (l: string, v: ReactNode, u?: string, tip?: string) => (
    <div className="stat">
      <div className="l">
        {l}
        {tip ? ' ' : ''}
        {tip && <Info k={tip} />}
      </div>
      <div className="v">
        {v}
        {u ? <span className="u"> {u}</span> : null}
      </div>
    </div>
  );
  return (
    <Card title="What is normal here" info="baseline" right={<StatusChip status={ev.status} />}>
      <div className="row between small">
        <span>{ok ? 'Baseline trusted' : 'Baseline building'}</span>
        <span className="num tx2">{Math.min(nAt, 20)} of 20 clear observations needed</span>
      </div>
      <div className="prog" style={{ margin: '6px 0 12px' }}>
        <i style={{ width: `${Math.min(1, nAt / 20) * 100}%` }} />
      </div>
      {ok ? (
        <div className="stats">
          {tile('Median', b.med.toFixed(1), 'MW')}
          {tile('Top of normal', b.hi.toFixed(1), 'MW')}
          {tile('95th percentile', b.p95.toFixed(1), 'MW')}
          {tile('Normal variation', b.mad.toFixed(2), 'log MAD', 'mad')}
          {tile('Observations', b.n)}
          {tile('Deviation now', last && last.z != null ? (last.z >= 0 ? '+' : '') + last.z.toFixed(1) : 'n/a', undefined, 'z')}
        </div>
      ) : (
        <p className="dim small" style={{ margin: 0 }}>
          Until 20 clear observations are in, this place is shown as baseline building and no claim is made about normal or abnormal.
        </p>
      )}
    </Card>
  );
}

function ReliabilityCard({ p, ev, ser, ctx }: { p: Place; ev: Event; ser: PlaceSeries; ctx: ReturnType<typeof contextOf> }) {
  const nAt = p.kind === 'site' ? ser.obs.filter((o) => o.t <= ev.t).length : 3;
  const f: [string, number][] = [
    ['Thermal signal', clamp(0.5 + ev.conf * 0.45, 0, 1)],
    ['History depth', p.kind === 'site' ? clamp(nAt / 60, 0, 1) : 0.2],
    ['Facility and land data', { good: 0.9, partial: 0.6, weak: 0.35 }[p.cover]],
    ['Imagery quality', ctx.valid],
  ];
  const all = f[0][1] * 0.3 + f[1][1] * 0.25 + f[2][1] * 0.25 + f[3][1] * 0.2;
  const lab = all >= 0.75 ? 'High' : all >= 0.55 ? 'Moderate' : 'Low';
  const src = {
    gas_flare: 'Nightfire flare catalogue match and facility match',
    industrial: 'Facility match and industrial land use',
    mining: 'Coal and mine facility match with land use',
    wildfire: 'Forest land cover, season and short duration',
    agricultural_burning: 'Cropland cover, season and short duration',
    unknown: 'No confident match',
  }[ev.cls];
  return (
    <Card title="How far to trust this" right="Sample values">
      <div className="row between">
        <b style={{ fontSize: 16 }}>{lab} reliability</b>
        <span className="dim small">Overall {Math.round(all * 100)} of 100</span>
      </div>
      <div style={{ display: 'grid', gap: 8, margin: '10px 0 14px' }}>
        {f.map(([l, v]) => (
          <Bar key={l} label={l} value={v} color="var(--tx)" columns="150px 1fr 40px" />
        ))}
      </div>
      <KV
        items={[
          ['Sensor', ev.sensor],
          ['Detections', `${ev.nDet} over ${ev.area} km²`],
          ['Label source', `${src} (weak label, not ground truth)`],
          ['Model', 'agni-xgb 0.3.1, sample'],
          ['Features', 'fs-2026.09, sample'],
          ['Last ingest', '2 h 41 min ago, sample'],
        ]}
      />
    </Card>
  );
}

function ergFor(p: Place, ev: Event): [string, string] | null {
  if (ev.cls === 'gas_flare' || /LNG/.test(p.type)) return ['ERG 2024 Guide 115', 'Gases, flammable (including refrigerated liquids)'];
  if (/Refinery|Petro/.test(p.type)) return ['ERG 2024 Guide 128', 'Flammable liquids (non-polar, water-immiscible)'];
  return null;
}

function ResponseCard({ p, ev, capXmlShown, onToggleCap, onCopyCap }: {
  p: Place;
  ev: Event;
  capXmlShown: boolean;
  onToggleCap: () => void;
  onCopyCap: () => void;
}) {
  const g = ergFor(p, ev);
  return (
    <Card title="Response reference" right={<span className="chip prop">Proposed</span>}>
      {g ? (
        <KV items={[['Reference', g[0]], ['Topic', g[1]]]} />
      ) : (
        <p className="small tx2" style={{ margin: 0 }}>
          No chemical guide applies without knowing what is burning. Use the facility's own emergency plan or the relevant state fire or mine
          emergency protocol.
        </p>
      )}
      <Banner kind="plain" className="small">
        Generic reference only. Not a substitute for an on-scene hazmat assessment or the facility's site-specific plan. Guidance text is looked
        up, never written by the model.
      </Banner>
      <div className="row wrap" style={{ marginTop: 12 }}>
        <button className="btn sm" aria-pressed={capXmlShown} onClick={onToggleCap}>
          {capXmlShown ? 'Hide' : 'Preview'} alert draft
        </button>
        <button className="btn sm" onClick={onCopyCap}>
          <Icon name="copy" size={14} />
          Copy draft
        </button>
      </div>
      {capXmlShown && <pre className="xml" style={{ marginTop: 10 }}>{capXml(ev)}</pre>}
      <div className="dim small" style={{ marginTop: 8 }}>
        A draft in the standard Common Alerting Protocol format for an authorized agency to review. It is not sent anywhere.
      </div>
    </Card>
  );
}

function ExportCard({ p, onCsv, onGeo }: { p: Place; onCsv: () => void; onGeo: () => void }) {
  return (
    <Card title="Export">
      <div className="row wrap">
        <button className="btn sm" onClick={onCsv}>
          <Icon name="download" size={14} />
          Events as CSV
        </button>
        <button className="btn sm" onClick={onGeo}>
          <Icon name="download" size={14} />
          Events as GeoJSON
        </button>
        <button className="btn sm" disabled>
          PDF incident report <span className="chip prop">Proposed</span>
        </button>
      </div>
      <span className="sr">{p.name}</span>
    </Card>
  );
}

/* ---------- the report ---------- */
export function PlaceReport({
  p,
  ev,
  asOf,
  expanded,
  shapAll,
  capXmlShown,
  onToggleShap,
  onToggleCap,
  onPickEvent,
  onOpenAdvanced,
  onWatch,
  onCopyLink,
  onCsv,
  onGeo,
}: {
  p: Place;
  ev: Event;
  asOf: number;
  expanded: boolean;
  shapAll: boolean;
  capXmlShown: boolean;
  onToggleShap: () => void;
  onToggleCap: () => void;
  onPickEvent: (id: string) => void;
  onOpenAdvanced: () => void;
  onWatch: () => void;
  onCopyLink: () => void;
  onCsv: () => void;
  onGeo: () => void;
}) {
  const { watch } = useReviews();
  const { copyText } = useFiles();

  const ser = seriesOf(p);
  const st = statsOf(p);
  const ctx = contextOf(p);
  featuresOf(p, ev); // warm the memoised vector so the first read is instant

  const left: ReactNode[] = [
    <HeaderCard key="h" p={p} ev={ev} asOf={asOf} watched={watch.has(p.id)} onWatch={onWatch} onOpenAdvanced={onOpenAdvanced} onCopyLink={onCopyLink} onExportCsv={onCsv} />,
    <VerdictCard key="v" p={p} ev={ev} st={st} ser={ser} />,
    <OutlookCard key="o" p={p} ev={ev} onOpen={onOpenAdvanced} />,
    <WhyCard key="w" ev={ev} shapAll={shapAll} onToggleShap={onToggleShap} />,
    <BaselineCard key="b" p={p} ev={ev} ser={ser} />,
    <SurroundCard key="s" p={p} ctx={ctx} />,
    <ReliabilityCard key="r" p={p} ev={ev} ser={ser} ctx={ctx} />,
    <ResponseCard key="p" p={p} ev={ev} capXmlShown={capXmlShown} onToggleCap={onToggleCap} onCopyCap={() => copyText(capXml(ev), 'Alert draft copied')} />,
  ];
  const right: ReactNode[] = [
    <TimelineCard key="t" p={p} ev={ev} ser={ser} />,
    <FingerprintCard key="f" p={p} st={st} ser={ser} />,
    <FootprintCard key="fp" p={p} ev={ev} />,
    <ImageryCard key="i" p={p} ev={ev} ctx={ctx} />,
    <ConditionsCard key="c" ctx={ctx} />,
    <NearbyCard key="n" p={p} />,
    <HistoryCard key="h2" p={p} ev={ev} onPick={onPickEvent} />,
    <ExportCard key="x" p={p} onCsv={onCsv} onGeo={onGeo} />,
  ];

  if (expanded) {
    return (
      <div className="rep-cols">
        <div>{left}</div>
        <div>{right}</div>
      </div>
    );
  }
  return <div className="rep-stack">{[...left, ...right]}</div>;
}
