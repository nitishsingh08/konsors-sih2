/* =====================================================================
   Small presentational pieces used across every screen: cards, chips,
   bars, rings and the plain-language glossary behind the "i" buttons.
   ===================================================================== */
import type { CSSProperties, ReactNode } from 'react';
import { clamp } from '../../lib/core';
import { STATUS } from '../../data/classes';
import { useReviews } from '../../state/ReviewsProvider';
import type { Event, StatusId } from '../../lib/types';
import { ClassBadge } from './Glyph';
import { useTip } from './Tooltip';

export function Card({
  title,
  info,
  right,
  children,
  className = '',
  style,
  as: As = 'section',
}: {
  title?: ReactNode;
  info?: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
  as?: 'section' | 'div' | 'details';
}) {
  return (
    <As className={'card ' + className} style={style}>
      {title != null && (
        <div className="ch">
          <h3>{title}</h3>
          {info && <Info k={info} />}
          <span className="r">{right}</span>
        </div>
      )}
      <div className="cbody">{children}</div>
    </As>
  );
}

export function Banner({
  kind = 'warn',
  children,
  icon,
  style,
  className = '',
}: {
  kind?: 'warn' | 'abn' | 'plain';
  children: ReactNode;
  icon?: ReactNode;
  style?: CSSProperties;
  className?: string;
}) {
  return (
    <div className={'banner ' + (kind === 'warn' ? '' : kind) + ' ' + className} style={style}>
      {icon}
      <span>{children}</span>
    </div>
  );
}

export function StatusChip({ status }: { status: StatusId }) {
  return (
    <span className={'chip ' + STATUS[status].k} {...useTip(STATUS[status].tip)}>
      {STATUS[status].label}
    </span>
  );
}

export const ReviewChip = () => (
  <span className="chip rev" {...useTip('Confidence is below 60%. A person should check this before anyone acts on it.')}>
    Needs review
  </span>
);

export function ReviewedChip({ eventId }: { eventId: string }) {
  const { reviews } = useReviews();
  if (!reviews[eventId]?.length) return null;
  return (
    <span className="chip impl" {...useTip('An analyst has reviewed this event.')}>
      Reviewed
    </span>
  );
}

export function EventChips({ ev }: { ev: Event }) {
  return (
    <>
      <ClassBadge cls={ev.cls} />
      <StatusChip status={ev.status} />
      {ev.review && <ReviewChip />}
      <ReviewedChip eventId={ev.id} />
    </>
  );
}

export function Info({ k, label = 'What is this?' }: { k: string; label?: string }) {
  return (
    <button className="info" type="button" aria-label={label} {...useTip(GLOSSARY[k])}>
      i
    </button>
  );
}

/** the same little "i" button, but with its explanation passed directly */
export function TipButton({ text, label = 'What is this?' }: { text: string; label?: string }) {
  return (
    <button className="info" type="button" aria-label={label} {...useTip(text)}>
      i
    </button>
  );
}

/** plain-language explanations, the same text the original showed on hover */
export const GLOSSARY: Record<string, string> = {
  frp: 'Fire radiative power (FRP) is how much heat a fire gives off, in megawatts. Bigger is hotter or larger.',
  mad: 'MAD is the median absolute deviation. It measures how much a site normally varies, and it is not thrown off by a few extreme days the way an average is.',
  z: "The robust z-score says how far today's fire power sits above this site's own normal level, in units of normal variation. Above about 3.5 is unusual.",
  conf: 'Calibrated confidence: when the system says 80%, it should be right about 8 times in 10 on held-out data.',
  baseline: "A baseline is a site's own normal range. It needs about 20 clear observations before it is trusted.",
  shap: 'Each bar shows how much a group of inputs pushed the system toward this class. Bars to the right support the call, bars to the left argue against it.',
  gap: 'Cloud or no satellite pass. A gap is missing data, not zero fire.',
  footprint: 'Footprint is the area covered by neighbouring detections grouped into one event. Observed, not predicted.',
  vpd: 'Vapour pressure deficit measures how dry the air is. Dry air helps vegetation fires spread.',
  plume: 'The smoke plume drifts with the forecast wind, hour by hour. It shows where smoke can travel, not how thick it is. The paler outline allows for the wind forecast being a little off.',
  spreadind: 'A rough guide to how fast a vegetation fire could advance, from wind, how dry the fuel is, and slope. It has not been checked against real fires, so treat it as indicative.',
  fwi: 'The Canadian Fire Weather Index system turns temperature, humidity, wind and rain into indices. FFMC is how dry the finest fuel is, ISI how fast a fire would spread, and FWI overall fire intensity.',
  elev: 'Ground height along the dashed line on the map, downwind of the source. Uphill stretches speed a fire up.',
  smokeopt: 'Smoke and burn signs read from optical satellite images. Cloud can hide them, and the scene may be days old.',
  sar: 'Radar sees through cloud and measures how the ground surface changed. Passes are a few days apart.',
  chem: 'Gas readings from a satellite that passes once a day, compared with what is normal for the place and season. They point to burning conditions but cannot prove a source.',
  cover: 'How complete the facility records (Global Energy Monitor and OpenStreetMap) are around this place. Weak coverage means a nearby facility may simply not be mapped.',
};

/** confidence ring with the number in the middle */
export function Ring({
  v,
  size = 38,
  sw = 4,
  color,
}: {
  v: number;
  size?: number;
  sw?: number;
  color?: string;
}) {
  const r = (size - sw) / 2;
  const c = 2 * Math.PI * r;
  const big = size > 60;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`Confidence ${Math.round(v * 100)} percent`}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--line2)" strokeWidth={sw} />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth={sw}
        strokeLinecap="round"
        strokeDasharray={`${(c * v).toFixed(1)} ${c.toFixed(1)}`}
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
      />
      <text
        x="50%"
        y="50%"
        textAnchor="middle"
        dominantBaseline="central"
        fill="currentColor"
        fontSize={big ? 21 : 11.5}
        fontWeight={600}
        style={{ fontVariantNumeric: 'tabular-nums' }}
      >
        {Math.round(v * 100)}
        {big ? '%' : ''}
      </text>
    </svg>
  );
}

/** tiny trend line */
export function Spark({ values, w = 96, h = 26, color = 'currentColor' }: { values: number[]; w?: number; h?: number; color?: string }) {
  if (!values.length) return null;
  const mx = Math.max(...values, 1);
  const n = values.length;
  const pts = values.map((v, i) => `${((i / Math.max(n - 1, 1)) * w).toFixed(1)},${(h - 2 - (v / mx) * (h - 4)).toFixed(1)}`).join(' ');
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden="true" focusable="false">
      <polyline points={pts} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

/** a label, a proportional bar and the value */
export function Bar({ label, value, color, columns }: { label: ReactNode; value: number; color?: string; columns?: string }) {
  return (
    <div className="pbar" style={columns ? { gridTemplateColumns: columns, margin: 0 } : undefined}>
      <span>{label}</span>
      <span className="tr">
        <i style={{ width: `${(clamp(value, 0, 1) * 100).toFixed(1)}%`, background: color }} />
      </span>
      <span className="num tx2" style={{ textAlign: 'right' }}>
        {Math.round(value * 100)}%
      </span>
    </div>
  );
}

export function Stat({ label, value, unit, children }: { label: ReactNode; value: ReactNode; unit?: string; children?: ReactNode }) {
  return (
    <div className="stat">
      <div className="l">{label}</div>
      <div className="v">
        {value}
        {unit && <span className="u"> {unit}</span>}
      </div>
      {children}
    </div>
  );
}

export function MiniBar({ value }: { value: number }) {
  return (
    <div className="mini w">
      <i style={{ width: `${clamp(value, 0, 1) * 100}%` }} />
    </div>
  );
}

export function Swatch({ color, border, style }: { color?: string; border?: string; style?: CSSProperties }) {
  return <i className="sw" style={{ background: color, border, ...style }} />;
}

export function LegendRow({ children }: { children: ReactNode }) {
  return <div className="legendrow">{children}</div>;
}

export function Empty({ title, children }: { title?: ReactNode; children: ReactNode }) {
  return (
    <div className="empty">
      {title != null && <b>{title}</b>}
      {children}
    </div>
  );
}

export function Seg<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: readonly { value: T; label: string }[];
  onChange: (v: T) => void;
  label?: string;
}) {
  return (
    <span className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} type="button" aria-pressed={value === o.value} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </span>
  );
}

/** progress bar that can be hovered for a longer explanation */
export function Shr({ label, v, mx, color, columns }: { label: ReactNode; v: number; mx: number; color: string; columns?: string }) {
  const w = (Math.abs(v) / mx) * 50;
  return (
    <div className="shr" style={columns ? { gridTemplateColumns: columns } : undefined}>
      <span className="tx2">{label}</span>
      <span className="ax">
        <i
          style={
            v >= 0
              ? { left: '50%', width: `${w}%`, background: color }
              : { right: '50%', width: `${w}%`, background: 'var(--dim)' }
          }
        />
      </span>
      <span className="vv">
        {v >= 0 ? '+' : ''}
        {v.toFixed(2)}
      </span>
    </div>
  );
}

/** label, band showing the usual range, and a marker for this value */
export function Meter({
  label,
  value,
  lo,
  hi,
  mu,
  sd,
  fmt,
}: {
  label: string;
  value: number;
  lo: number;
  hi: number;
  mu: number;
  sd: number;
  fmt?: (v: number) => string;
}) {
  const pos = clamp((value - lo) / (hi - lo), 0, 1) * 100;
  const a = clamp((mu - sd - lo) / (hi - lo), 0, 1) * 100;
  const b = clamp((mu + sd - lo) / (hi - lo), 0, 1) * 100;
  return (
    <div className="meter">
      <div className="row between small">
        <span>{label}</span>
        <b className="num">{fmt ? fmt(value) : value.toFixed(0)}</b>
      </div>
      <div className="track">
        <i className="band" style={{ left: `${a}%`, width: `${Math.max(2, b - a)}%` }} />
        <i className="mark" style={{ left: `${pos}%` }} />
      </div>
    </div>
  );
}

export function KV({ items }: { items: [ReactNode, ReactNode][] }) {
  return (
    <dl className="kv small">
      {items.map(([k, v], i) => (
        <div key={i} style={{ display: 'contents' }}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

/** small interactive star, used for the watchlist (CSS fills it when pressed) */
export function Star({ on, onClick, label, size = 20 }: { on: boolean; onClick: () => void; label: string; size?: number }) {
  return (
    <button className="star" type="button" aria-pressed={on} aria-label={label} onClick={onClick}>
      <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.7} strokeLinejoin="round" aria-hidden="true" focusable="false">
        <path d="M12 3.5l2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.9 6.8 19.6l1-5.8L3.5 9.7l5.9-.9z" />
      </svg>
    </button>
  );
}
