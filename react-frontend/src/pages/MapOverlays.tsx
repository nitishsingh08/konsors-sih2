/* =====================================================================
   The DOM overlays that sit on top of the command map: legend, layer
   panel, draw button, wind control and zoom buttons.
   Port of the overlay markup in mapPageHtml() and renderLegend().
   ===================================================================== */
import { CLS } from '../data/classes';
import { Glyph } from '../components/ui/Glyph';
import { Icon } from '../components/ui/Icon';
import { Seg } from '../components/ui/Primitives';
import { TODAY, hourLabel } from '../lib/time';
import { windGradientCss, type WindStyle } from '../map/WindLayer';
import type { LayerFlags } from '../map/CommandMap';
import type { BaseMode } from '../map/leafletBase';

const LAYER_LABELS: [keyof LayerFlags, string][] = [
  ['events', 'Classified events'],
  ['raw', 'Raw satellite detections'],
  ['facilities', 'Facility footprints'],
  ['density', 'Density (hexagons)'],
  ['wind', 'Wind forecast'],
];

export function MapOverlays({
  layers,
  onLayer,
  showLayers,
  onToggleLayers,
  drawing,
  onToggleDraw,
  base,
  onBase,
  windH,
  onWindH,
  windStyle,
  onWindStyle,
  cls,
  onToggleClass,
  onZoomIn,
  onZoomOut,
  onReset,
}: {
  layers: LayerFlags;
  onLayer: (k: keyof LayerFlags) => void;
  showLayers: boolean;
  onToggleLayers: () => void;
  drawing: boolean;
  onToggleDraw: () => void;
  base: BaseMode;
  onBase: (b: BaseMode) => void;
  windH: number;
  onWindH: (h: number) => void;
  windStyle: WindStyle;
  onWindStyle: (s: WindStyle) => void;
  cls: Set<string>;
  onToggleClass: (id: string) => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onReset: () => void;
}) {
  return (
    <>
      <div className="ov tl">
        <button className="btn sm" aria-pressed={showLayers} onClick={onToggleLayers}>
          <Icon name="layers" size={15} />
          Layers
        </button>
      </div>

      <div className="ov layers" style={showLayers ? undefined : { display: 'none' }}>
        <div className="dim small">Base map</div>
        <label>
          <input type="radio" name="base" value="street" checked={base !== 'plain'} onChange={() => onBase('street')} />
          Street map (OpenStreetMap)
        </label>
        <label>
          <input type="radio" name="base" value="plain" checked={base === 'plain'} onChange={() => onBase('plain')} />
          Plain outline (works offline)
        </label>
        <div className="dim small" style={{ marginTop: 4 }}>
          On the map
        </div>
        {LAYER_LABELS.map(([k, l]) => (
          <label key={k}>
            <input type="checkbox" checked={layers[k]} onChange={() => onLayer(k)} />
            {l}
          </label>
        ))}
      </div>

      <div className="ov tr">
        <button className="btn sm" aria-pressed={drawing} onClick={onToggleDraw}>
          <Icon name="draw" size={15} />
          Draw area
        </button>
        <button className="btn sm ic" aria-label="Reset view" onClick={onReset}>
          <Icon name="reset" size={15} />
        </button>
      </div>

      {drawing && <div className="ov hint">Drag on the map to draw a box</div>}

      <div className="ov legend">
        {CLS.map((c) => (
          <button key={c.id} aria-pressed={cls.has(c.id)} onClick={() => onToggleClass(c.id)}>
            <Glyph cls={c.id} size={13} />
            {c.label}
          </button>
        ))}
        <div className="sep" />
        <div className="k">
          <svg width="14" height="14" aria-hidden="true">
            <circle cx="7" cy="7" r="5.5" fill="none" stroke="var(--danger)" strokeWidth="2" />
          </svg>
          Abnormal for its site
        </div>
        <div className="k">
          <svg width="14" height="14" aria-hidden="true">
            <circle cx="7" cy="7" r="3" fill="var(--warn)" />
          </svg>
          Needs review
        </div>
        <div className="k">
          <svg width="14" height="14" aria-hidden="true">
            <circle cx="7" cy="7" r="5.5" fill="none" stroke="var(--tx2)" strokeDasharray="3 3" />
          </svg>
          Baseline building
        </div>
      </div>

      <div className="ov windctl" style={layers.wind ? undefined : { display: 'none' }}>
        <div className="row between small">
          <b>{windH === 0 ? 'Now' : `In ${windH} h, ${hourLabel(TODAY, windH)}`}</b>
          <Seg
            value={windStyle}
            onChange={onWindStyle}
            label="Wind style"
            options={[
              { value: 'flow' as WindStyle, label: 'Flow' },
              { value: 'arrows' as WindStyle, label: 'Arrows' },
            ]}
          />
        </div>
        <input type="range" min={0} max={72} step={3} value={windH} aria-label="Wind forecast hour" onChange={(e) => onWindH(Number(e.target.value))} />
        <div className="row small">
          <span className="dim">Slow</span>
          <span className="windbar" style={{ background: windGradientCss() }} />
          <span className="dim">Fast</span>
        </div>
        <div className="dim small">Sample wind. Your weather feed replaces it.</div>
      </div>

      <div className="ov zoom">
        <button aria-label="Zoom in" onClick={onZoomIn}>
          <Icon name="plus" size={16} />
        </button>
        <button aria-label="Zoom out" onClick={onZoomOut}>
          <Icon name="minus" size={16} />
        </button>
      </div>
    </>
  );
}
