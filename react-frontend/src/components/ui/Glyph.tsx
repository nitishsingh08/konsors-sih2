import { ccol } from '../../lib/core';
import { CLSMAP } from '../../data/classes';
import type { ClassId } from '../../lib/types';

/* Class glyphs: the shape carries the meaning as well as the colour, so the
   map still reads without colour. Also used to build Leaflet div icons. */
export function glyphPath(cls: ClassId, r: number, cx = 0, cy = 0): string {
  switch (cls) {
    case 'wildfire':
      return `M${cx} ${cy - r * 1.1}L${cx + r * 1.05} ${cy + r * 0.8}L${cx - r * 1.05} ${cy + r * 0.8}Z`;
    case 'agricultural_burning':
      return `M${cx - r * 0.85} ${cy - r * 0.85}h${r * 1.7}v${r * 1.7}h${-r * 1.7}Z`;
    case 'mining': {
      let d = '';
      for (let i = 0; i < 6; i++) {
        const a = (Math.PI / 3) * i + Math.PI / 6;
        d += (i ? 'L' : 'M') + (cx + r * 1.05 * Math.cos(a)).toFixed(2) + ' ' + (cy + r * 1.05 * Math.sin(a)).toFixed(2);
      }
      return d + 'Z';
    }
    case 'industrial':
      return `M${cx} ${cy - r * 1.2}L${cx + r * 1.2} ${cy}L${cx} ${cy + r * 1.2}L${cx - r * 1.2} ${cy}Z`;
    default:
      return `M${cx - r} ${cy}a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0Z`;
  }
}

export function Glyph({ cls, size = 12 }: { cls: ClassId; size?: number }) {
  const c = ccol(cls);
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true" focusable="false">
      <path
        d={glyphPath(cls, size / 2 - 1.2, size / 2, size / 2)}
        {...(cls === 'unknown'
          ? { fill: 'none', stroke: c, strokeWidth: 1.6 }
          : { fill: c })}
      />
    </svg>
  );
}

export function ClassBadge({ cls, style, className = '' }: { cls: ClassId; style?: React.CSSProperties; className?: string }) {
  return (
    <span className={'cb ' + className} style={style}>
      <Glyph cls={cls} size={12} />
      {CLSMAP[cls].label}
    </span>
  );
}
