/* =====================================================================
   The command map. Leaflet + OpenStreetMap tiles, with the offline plain
   outline fallback from the original build.
   Port of frontend/js/map/leaflet-map.js.
   ===================================================================== */
import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import L from 'leaflet';
import { ago } from '../lib/time';
import { ccol, cv } from '../lib/core';
import { CLSMAP, STATUS } from '../data/classes';
import { PMAP, ptsOf } from '../data/places';
import { nationalGrid } from '../data/weather';
import { sev, type Group } from '../state/selectors';
import { useTheme } from '../state/ThemeProvider';
import { usePrefersReducedMotion } from '../hooks/usePrefersReducedMotion';
import {
  addTiles,
  facilityRing,
  getSavedView,
  hexRound,
  hlat,
  hlon,
  hx,
  hy,
  IN_BOUNDS,
  makePanes,
  markerIcon,
  plainBase,
  rampAt,
  setSavedView,
  type BaseMode,
} from './leafletBase';
import { WindLayer, type WindStyle } from './WindLayer';

export interface LayerFlags {
  raw: boolean;
  events: boolean;
  facilities: boolean;
  density: boolean;
  wind: boolean;
}

export interface AreaBox {
  lo0: number;
  lo1: number;
  la0: number;
  la1: number;
}

export interface CommandMapProps {
  groups: Group[];
  sel: string | null;
  layers: LayerFlags;
  windH: number;
  windStyle: WindStyle;
  drawing: boolean;
  area: AreaBox | null;
  base: BaseMode;
  focus: { id: string; seq: number } | null;
  onSelect: (id: string) => void;
  onArea: (a: AreaBox) => void;
  onBaseFallback: () => void;
}

export interface CommandMapHandle {
  zoom: (d: number) => void;
  reset: () => void;
  invalidateSize: () => void;
}

/** Leaflet is imperative, so the map is driven from effects and exposed
    through a small handle for the zoom buttons that live outside it. */
export const CommandMap = forwardRef<CommandMapHandle, CommandMapProps>(function CommandMap(props, ref) {
  const { groups, sel, layers, windH, windStyle, drawing, area, base, focus, onSelect, onArea, onBaseFallback } = props;
  const { theme } = useTheme();
  const reduced = usePrefersReducedMotion();
  const el = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const g = useRef<Record<string, L.LayerGroup>>({});
  const wind = useRef<WindLayer | null>(null);
  const tileRef = useRef<L.TileLayer | null>(null);
  const plainRef = useRef<L.LayerGroup | null>(null);
  const selRef = useRef(sel);
  selRef.current = sel;

  /* create the map once */
  useEffect(() => {
    const node = el.current;
    if (!node) return;
    const map = L.map(node, {
      zoomControl: false,
      attributionControl: true,
      preferCanvas: true,
      minZoom: 4,
      maxZoom: 16,
      zoomSnap: 0.25,
      zoomDelta: 0.5,
      wheelPxPerZoomLevel: 90,
    });
    map.attributionControl.setPrefix(false);
    makePanes(map);
    const saved = getSavedView();
    if (saved) map.setView([saved.lat, saved.lon], saved.zoom, { animate: false });
    else map.fitBounds(IN_BOUNDS, { padding: [8, 8] });
    g.current = {
      hex: L.layerGroup().addTo(map),
      fac: L.layerGroup().addTo(map),
      raw: L.layerGroup().addTo(map),
      mk: L.layerGroup().addTo(map),
      area: L.layerGroup().addTo(map),
    };
    mapRef.current = map;

    const ro = new ResizeObserver(() => map.invalidateSize());
    ro.observe(node);

    return () => {
      ro.disconnect();
      const c = map.getCenter();
      setSavedView({ lat: c.lat, lon: c.lng, zoom: map.getZoom() });
      wind.current = null;
      map.remove();
      mapRef.current = null;
      g.current = {};
    };
  }, []);

  /* base map: street tiles or the offline outline */
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (tileRef.current && map.hasLayer(tileRef.current)) map.removeLayer(tileRef.current);
    if (plainRef.current && map.hasLayer(plainRef.current)) map.removeLayer(plainRef.current);
    tileRef.current = null;
    plainRef.current = null;
    if (base === 'street') {
      addTiles(map, onBaseFallback);
    } else {
      plainRef.current = plainBase(map);
    }
  }, [base, theme, onBaseFallback]);

  /* everything drawn on top of the base */
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    Object.values(g.current).forEach((x) => x.clearLayers());
    const tx2 = cv('--tx2');

    if (layers.density) {
      const pts: [number, number][] = [];
      groups.forEach((gr) => gr.evs.forEach((e) => ptsOf(e).forEach((q) => pts.push(q))));
      if (pts.length) {
        const size = 0.5;
        const bins = new Map<string, number>();
        pts.forEach(([lo, la]) => {
          const x = hx(lo);
          const y = hy(la);
          const [q, r] = hexRound((Math.sqrt(3) / 3 * x - y / 3) / size, (2 / 3) * y / size);
          const k = q + ',' + r;
          bins.set(k, (bins.get(k) || 0) + 1);
        });
        const mx = Math.max(...bins.values());
        bins.forEach((n, k) => {
          const [q, r] = k.split(',').map(Number);
          const cx = size * Math.sqrt(3) * (q + r / 2);
          const cy = size * 1.5 * r;
          const ring: [number, number][] = [];
          for (let i = 0; i < 6; i++) {
            const a = (Math.PI / 180) * (60 * i - 30);
            ring.push([hlat(cy + size * Math.sin(a)), hlon(cx + size * Math.cos(a))]);
          }
          L.polygon(ring, { interactive: false, stroke: false, fillColor: rampAt(Math.sqrt(n / mx), theme), fillOpacity: 0.62 }).addTo(
            g.current.hex,
          );
        });
      }
    }

    if (layers.facilities) {
      groups.forEach((gr) => {
        const p = gr.p;
        if (p.kind !== 'site' || p.cls === 'unknown') return;
        const c = ccol(p.cls);
        L.polygon(facilityRing(p.lat, p.lon, p.id), {
          interactive: false,
          color: c,
          weight: 1,
          fillColor: c,
          fillOpacity: 0.18,
        }).addTo(g.current.fac);
      });
    }

    if (layers.raw) {
      const all: [number, number][] = [];
      groups.forEach((gr) => gr.evs.forEach((e) => ptsOf(e).forEach((q) => all.push(q))));
      const step = Math.max(1, Math.ceil(all.length / 2200));
      all.forEach((q, i) => {
        if (i % step) return;
        L.circleMarker([q[1], q[0]], { radius: 1.7, stroke: false, fillColor: tx2, fillOpacity: 0.5, interactive: false }).addTo(
          g.current.raw,
        );
      });
    }

    if (layers.events) {
      groups.forEach((gr) => {
        const p = gr.p;
        const e = gr.top;
        const m = L.marker([p.lat, p.lon], {
          icon: markerIcon({
            cls: e.cls,
            status: e.status,
            review: e.review,
            selected: sel === p.id,
            transient: p.kind !== 'site',
          }),
          keyboard: true,
          title: p.name + ', ' + CLSMAP[e.cls].label,
          zIndexOffset: sev(e) * 100 + (sel === p.id ? 500 : 0) + (p.kind === 'site' ? 50 : 0),
        });
        m.bindTooltip(
          `<b>${p.name}</b><br>${p.district}, ${p.state}<br>${CLSMAP[e.cls].label}, ${Math.round(e.conf * 100)}% sure<br>${STATUS[e.status].label}${
            e.review ? ', needs review' : ''
          }<br>${e.frp.toFixed(0)} MW, ${ago(e.t)}`,
          { direction: 'top', offset: [0, -12], className: 'lt', opacity: 1 },
        );
        m.on('click', () => onSelect(p.id));
        m.addTo(g.current.mk);
      });
    }

    if (area) {
      L.rectangle(
        [
          [area.la0, area.lo0],
          [area.la1, area.lo1],
        ],
        { interactive: false, color: cv('--tx'), weight: 1.5, dashArray: '6 4', fillColor: cv('--tx'), fillOpacity: 0.06 },
      ).addTo(g.current.area);
    }
  }, [groups, layers, sel, area, theme, onSelect]);

  /* national wind forecast */
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (layers.wind) {
      if (!wind.current) wind.current = new WindLayer({ pane: 'wind', count: 1500, style: windStyle });
      if (wind.current.options.style !== windStyle) wind.current.setStyle(windStyle);
      wind.current.setGrid(nationalGrid(windH));
      if (!map.hasLayer(wind.current)) wind.current.addTo(map);
    } else if (wind.current && map.hasLayer(wind.current)) {
      map.removeLayer(wind.current);
    }
  }, [layers.wind, windH, windStyle, theme]);

  /* drag a box on the map to get an area report */
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const c = map.getContainer();
    map.dragging[drawing ? 'disable' : 'enable']();
    c.style.cursor = drawing ? 'crosshair' : '';
    if (!drawing) return;

    let start: L.LatLng | null = null;
    let rect: L.Rectangle | null = null;

    const down = (ev: PointerEvent) => {
      if (ev.button > 0 || (ev.target as HTMLElement).closest('.ov')) return;
      start = map.mouseEventToLatLng(ev);
      rect = L.rectangle([[start.lat, start.lng], [start.lat, start.lng]], {
        color: cv('--tx'),
        weight: 1.5,
        dashArray: '6 4',
        fillColor: cv('--tx'),
        fillOpacity: 0.08,
        interactive: false,
      }).addTo(map);
      try {
        c.setPointerCapture(ev.pointerId);
      } catch {
        /* not all browsers support pointer capture here */
      }
      ev.preventDefault();
      ev.stopPropagation();
    };
    const move = (ev: PointerEvent) => {
      if (start && rect) rect.setBounds(L.latLngBounds(start, map.mouseEventToLatLng(ev)));
    };
    const up = () => {
      if (!start || !rect) return;
      const b = rect.getBounds();
      map.removeLayer(rect);
      start = null;
      rect = null;
      if (Math.abs(b.getEast() - b.getWest()) < 0.05) return;
      onArea({ lo0: b.getWest(), lo1: b.getEast(), la0: b.getSouth(), la1: b.getNorth() });
    };

    c.addEventListener('pointerdown', down, true);
    c.addEventListener('pointermove', move);
    c.addEventListener('pointerup', up);
    return () => {
      c.removeEventListener('pointerdown', down, true);
      c.removeEventListener('pointermove', move);
      c.removeEventListener('pointerup', up);
    };
  }, [drawing, onArea]);

  /* fly to a place when one is selected from the queue or search */
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !focus) return;
    const p = PMAP[focus.id];
    if (!p) return;
    const z = Math.max(map.getZoom(), 8.5);
    if (reduced) map.setView([p.lat, p.lon], z, { animate: false });
    else map.flyTo([p.lat, p.lon], z, { duration: 0.9 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focus?.seq]);

  useImperativeHandle(
    ref,
    () => ({
      zoom: (d: number) => mapRef.current?.[d > 0 ? 'zoomIn' : 'zoomOut'](1),
      reset: () => mapRef.current?.fitBounds(IN_BOUNDS, { padding: [8, 8] }),
      invalidateSize: () => setTimeout(() => mapRef.current?.invalidateSize(), 60),
    }),
    [],
  );

  return <div id="lmap" ref={el} role="application" aria-label="Map of thermal events over India" />;
});
