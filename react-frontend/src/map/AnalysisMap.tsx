/* =====================================================================
   The advanced analysis map: local wind, smoke plume, fire spread,
   observed footprints, distance rings, the terrain line, nearby sources
   and the measuring tool.
   Port of the map half of frontend/js/analysis/analysis.js.
   ===================================================================== */
import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import L from 'leaflet';
import { ccol, cv, tk } from '../lib/core';
import { CLSMAP } from '../data/classes';
import { hourLabel } from '../lib/time';
import { compass, localGrid, offsetLL, rad, windShift } from '../data/weather';
import { nearbyOf } from '../data/places';
import { useTheme } from '../state/ThemeProvider';
import { addTiles, glyphIcon, ll, makePanes, noteIcon, plainBase, tickIcon, type BaseMode } from './leafletBase';
import { WindLayer, type WindStyle } from './WindLayer';
import type { Event, FootprintFrame, Plume, Place, SpreadOutlook, Terrain, Weather } from '../lib/types';

export interface ShowFlags {
  wind: boolean;
  plume: boolean;
  spread: boolean;
  foot: boolean;
  rings: boolean;
  terr: boolean;
  near: boolean;
}

export interface AnalysisMapHandle {
  /** move the marker that follows the elevation chart hover */
  setElevMarker: (at: [number, number] | null) => void;
  fit: () => void;
}

export interface AnalysisMapProps {
  p: Place;
  ev: Event;
  W: Weather;
  h: number;
  plume: Plume;
  spread: SpreadOutlook | null;
  terrain: Terrain | null;
  footprints: FootprintFrame[];
  /** index of the last footprint frame to draw, or -1 for all of them */
  fpIdx: number;
  show: ShowFlags;
  op: number;
  style: WindStyle;
  measure: boolean;
  mpts: L.LatLng[];
  base: BaseMode;
  onMeasurePoint: (at: L.LatLng) => void;
  onTilesFail: () => void;
}

export const AnalysisMap = forwardRef<AnalysisMapHandle, AnalysisMapProps>(function AnalysisMap(props, ref) {
  const { p, ev, W, h, plume, spread, terrain, footprints, fpIdx, show, op, style, measure, mpts, base, onMeasurePoint, onTilesFail } =
    props;
  const { theme } = useTheme();
  const el = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const wind = useRef<WindLayer | null>(null);
  const g = useRef<Record<string, L.LayerGroup>>({});
  const elevMarker = useRef<L.CircleMarker | null>(null);

  /* create the map once for this place */
  useEffect(() => {
    const node = el.current;
    if (!node) return;
    const map = L.map(node, {
      zoomControl: true,
      attributionControl: true,
      preferCanvas: true,
      minZoom: 5,
      maxZoom: 16,
      zoomSnap: 0.25,
      wheelPxPerZoomLevel: 90,
    });
    map.attributionControl.setPrefix(false);
    map.zoomControl.setPosition('bottomright');
    makePanes(map);
    map.setView([p.lat, p.lon], 9);
    g.current = {
      ter: L.layerGroup().addTo(map),
      near: L.layerGroup().addTo(map),
      rings: L.layerGroup().addTo(map),
      foot: L.layerGroup().addTo(map),
      plume: L.layerGroup().addTo(map),
      spread: L.layerGroup().addTo(map),
      src: L.layerGroup().addTo(map),
      note: L.layerGroup().addTo(map),
      meas: L.layerGroup().addTo(map),
    };
    wind.current = new WindLayer({ pane: 'wind', count: 1100, style, opacity: op });
    mapRef.current = map;
    return () => {
      wind.current = null;
      elevMarker.current = null;
      map.remove();
      mapRef.current = null;
      g.current = {};
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* base map */
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (base === 'plain') plainBase(map).addTo(map);
    else addTiles(map, onTilesFail);
  }, [base, theme, onTilesFail]);

  const drawMeasure = () => {
    const map = mapRef.current;
    if (!map) return;
    g.current.meas.clearLayers();
    if (!mpts.length) return;
    L.polyline(mpts, { color: cv('--tx'), weight: 2, interactive: false }).addTo(g.current.meas);
    let d = 0;
    mpts.forEach((q, i) => {
      L.circleMarker(q, { radius: 3.5, color: cv('--tx'), weight: 1.5, fillColor: cv('--panel'), fillOpacity: 1, interactive: false }).addTo(
        g.current.meas,
      );
      if (i) d += map.distance(mpts[i - 1], q);
    });
    if (mpts.length > 1) {
      L.marker(mpts[mpts.length - 1], { interactive: false, icon: noteIcon(`${(d / 1000).toFixed(2)} km`) }).addTo(g.current.meas);
    }
  };

  /* every overlay, redrawn whenever the forecast hour or a layer changes */
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    Object.values(g.current).forEach((x) => x.clearLayers());
    elevMarker.current = null;
    const col = ccol(ev.cls);
    const T = tk();

    if (show.wind && wind.current) {
      if (!map.hasLayer(wind.current)) wind.current.addTo(map);
      wind.current.setGrid(localGrid(p, W, h));
      wind.current.setOpacity(op);
    } else if (wind.current && map.hasLayer(wind.current)) {
      map.removeLayer(wind.current);
    }

    if (show.plume) {
      L.polygon(ll(plume.outer), {
        interactive: false, color: col, weight: 1, dashArray: '4 4', opacity: 0.55 * op,
        fillColor: col, fillOpacity: 0.07 * op,
      }).addTo(g.current.plume);
      L.polygon(ll(plume.core), {
        interactive: false, color: col, weight: 1.4, opacity: 0.9 * op,
        fillColor: col, fillOpacity: 0.26 * op,
      }).addTo(g.current.plume);
      L.polyline(plume.traj.map((q) => [q[1], q[0]] as [number, number]), {
        interactive: false, color: col, weight: 1.4, dashArray: '2 5', opacity: 0.9 * op,
      }).addTo(g.current.plume);
      [3, 6, 12, 24, 36, 48, 72]
        .filter((t) => t <= plume.H)
        .forEach((t) => {
          const q = plume.traj[t * 2];
          if (q) L.marker([q[1], q[0]], { interactive: false, icon: tickIcon(`+${t} h`) }).addTo(g.current.plume);
        });
      L.marker([plume.head[1], plume.head[0]], {
        interactive: false,
        icon: noteIcon(`Smoke could reach ${plume.reachKm.toFixed(0)} km by ${hourLabel(W.t0, h || 1)}`, 'tilt'),
      }).addTo(g.current.note);
    }

    if (show.spread && spread) {
      L.polygon(ll(spread.unc), { interactive: false, color: col, weight: 1, dashArray: '3 5', opacity: 0.6 * op, fillOpacity: 0 }).addTo(
        g.current.spread,
      );
      spread.horizons
        .slice()
        .reverse()
        .forEach((z, i) => {
          L.polygon(ll(z.ring), {
            interactive: false, color: T.tx, weight: 1.4, opacity: op, fillColor: col, fillOpacity: (0.12 + 0.05 * (2 - i)) * op,
          })
            .addTo(g.current.spread)
            .bindTooltip(`+${z.t} h: about ${z.D.toFixed(1)} km`, { sticky: true, className: 'lt', opacity: 1 });
        });
      const z6 = spread.horizons[2];
      const head = offsetLL(p.lon, p.lat, Math.sin(rad(z6.dirTo)) * z6.D * 1000, Math.cos(rad(z6.dirTo)) * z6.D * 1000);
      L.marker([head[1], head[0]], {
        interactive: false,
        icon: noteIcon(`Fire could advance about ${z6.D.toFixed(1)} km in 6 h. Indicative only.`, 'warn'),
      }).addTo(g.current.note);
    }

    if (show.foot && footprints.length) {
      const idx = fpIdx < 0 ? footprints.length - 1 : Math.min(fpIdx, footprints.length - 1);
      footprints.slice(0, idx + 1).forEach((f, i) =>
        L.polygon(ll(f.ring), {
          interactive: false, color: col, weight: i === idx ? 1.8 : 0.8, opacity: i === idx ? 1 : 0.5,
          fillColor: col, fillOpacity: i === idx ? 0.38 : 0.06,
        }).addTo(g.current.foot),
      );
    }

    if (show.rings) {
      [2, 5, 10].forEach((k) => {
        L.circle([p.lat, p.lon], { radius: k * 1000, interactive: false, color: T.dim, weight: 1, dashArray: '3 5', fill: false, opacity: 0.8 }).addTo(
          g.current.rings,
        );
        const q = offsetLL(p.lon, p.lat, 0, k * 1000);
        L.marker([q[1], q[0]], { interactive: false, icon: tickIcon(`${k} km`) }).addTo(g.current.rings);
      });
    }

    if (show.terr && terrain) {
      L.polyline(terrain.pts.map((q) => [q.lat, q.lon] as [number, number]), {
        interactive: false, color: T.tx, weight: 1.4, dashArray: '8 6', opacity: 0.75 * op,
      }).addTo(g.current.ter);
    }

    if (show.near) {
      nearbyOf(p).forEach((n) => {
        const q = offsetLL(p.lon, p.lat, Math.cos(n.ang) * n.km * 1000, Math.sin(n.ang) * n.km * 1000);
        L.marker([q[1], q[0]], { icon: glyphIcon(n.cls, 5), keyboard: false })
          .bindTooltip(`${n.name}, ${n.km} km`, { className: 'lt', direction: 'top', opacity: 1 })
          .addTo(g.current.near);
      });
    }

    L.marker([p.lat, p.lon], { icon: glyphIcon(ev.cls, 8, true), keyboard: false, zIndexOffset: 800 })
      .bindTooltip(`<b>${p.name}</b><br>${CLSMAP[ev.cls].label}`, { className: 'lt', direction: 'top', offset: [0, -12], opacity: 1 })
      .addTo(g.current.src);

    const sh = windShift(W);
    if (sh) {
      const away = (W.dir[0] + 180) % 360;
      const back = (away + 180) % 360;
      const q = offsetLL(p.lon, p.lat, Math.sin(rad(back)) * 3400, Math.cos(rad(back)) * 3400);
      L.marker([q[1], q[0]], {
        interactive: false,
        icon: noteIcon(`Wind turns toward the ${compass(sh.to)} around ${hourLabel(W.t0, sh.h)}`, 'tilt2'),
      }).addTo(g.current.note);
    }

    drawMeasure();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [h, show, op, plume, spread, terrain, footprints, fpIdx, W, p, ev, base, style, theme]);

  /* measuring points changed */
  useEffect(() => {
    drawMeasure();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mpts]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const onClick = (e: L.LeafletMouseEvent) => {
      if (!measure) return;
      onMeasurePoint(e.latlng);
    };
    map.on('click', onClick);
    return () => {
      map.off('click', onClick);
    };
  }, [measure, onMeasurePoint]);

  useImperativeHandle(
    ref,
    () => ({
      setElevMarker: (at) => {
        const map = mapRef.current;
        if (!map) return;
        if (!at) {
          if (elevMarker.current) {
            map.removeLayer(elevMarker.current);
            elevMarker.current = null;
          }
          return;
        }
        if (!elevMarker.current) {
          elevMarker.current = L.circleMarker(at, {
            radius: 6, color: cv('--tx'), weight: 2, fillColor: cv('--panel'), fillOpacity: 1, interactive: false,
          }).addTo(g.current.ter);
        } else {
          elevMarker.current.setLatLng(at);
        }
      },
      fit: () => {
        const map = mapRef.current;
        if (!map) return;
        let b = L.latLngBounds([[p.lat, p.lon]]);
        if (show.plume) plume.outer.forEach((q) => b.extend([q[1], q[0]]));
        if (spread && show.spread) spread.unc.forEach((q) => b.extend([q[1], q[0]]));
        map.fitBounds(b, { paddingTopLeft: [50, 50], paddingBottomRight: [50, 130], maxZoom: 11, animate: true });
      },
    }),
    [plume, spread, show, p],
  );

  useEffect(() => {
    const node = el.current;
    if (!node) return;
    const ro = new ResizeObserver(() => mapRef.current?.invalidateSize());
    ro.observe(node);
    return () => ro.disconnect();
  }, []);

  return <div id="advmap" ref={el} />;
});
