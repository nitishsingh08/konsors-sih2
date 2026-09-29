/* =====================================================================
   The command page: filters, KPI strip, the map, the time scrubber and
   the side panel that shows either the queue, a place report or an area
   report. Port of the command page in frontend/js/ui.js + pages.js.
   ===================================================================== */
import { useCallback, useEffect, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { store } from '../lib/core';
import { DAY, START, TODAY } from '../lib/time';
import { PMAP, EVP } from '../data/places';
import { eventsCsv, eventsGeo } from '../data/analysis';
import { useFiles } from '../lib/files';
import { useFilters } from '../state/useFilters';
import { pickEvent, sev, useCurrent, type Group } from '../state/selectors';
import { useReviews } from '../state/ReviewsProvider';
import { useToast } from '../state/ToastProvider';
import { useNav } from '../components/Shell';
import { Icon } from '../components/ui/Icon';
import { Empty } from '../components/ui/Primitives';
import { CommandMap, type CommandMapHandle, type LayerFlags } from '../map/CommandMap';
import type { BaseMode } from '../map/leafletBase';
import type { WindStyle } from '../map/WindLayer';
import { usePrefersReducedMotion } from '../hooks/usePrefersReducedMotion';
import { Filters } from './Filters';
import { Kpis } from './Kpis';
import { Scrubber } from './Scrubber';
import { MapOverlays } from './MapOverlays';
import { QueueList } from '../report/Queue';
import { PlaceReport } from '../report/PlaceReport';
import { AreaReport, inArea, type AreaBox } from '../report/AreaReport';
import { AdvancedAnalysis } from '../analysis/AdvancedAnalysis';
import { CLS } from '../data/classes';

const initialBase = (): BaseMode => {
  const b = store.get('agni-base');
  return b === 'plain' ? 'plain' : 'street';
};

export function CommandPage({ sel }: { sel: string | null }) {
  const [filters, setFilters] = useFilters();
  const cur = useCurrent();
  const { go, goPlace } = useNav();
  const { toggleWatch } = useReviews();
  const { copyText, saveFile } = useFiles();
  const toast = useToast();
  const reduced = usePrefersReducedMotion();

  /* view-local state: the parts of the command page that are not shareable */
  const [layers, setLayers] = useState<LayerFlags>({ raw: true, events: true, facilities: true, density: false, wind: false });
  const [windH, setWindH] = useState(0);
  const [windStyle, setWindStyle] = useState<WindStyle>(() => (reduced ? 'arrows' : 'flow'));
  const [showLayers, setShowLayers] = useState(false);
  const [drawing, setDrawing] = useState(false);
  const [base, setBase] = useState<BaseMode>(initialBase);
  const [area, setArea] = useState<AreaBox | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [evId, setEvId] = useState<string | null>(null);
  const [shapAll, setShapAll] = useState(false);
  const [capXmlShown, setCapXmlShown] = useState(false);
  const [qshow, setQshow] = useState(30);
  const [play, setPlay] = useState(false);
  const [focus, setFocus] = useState<{ id: string; seq: number } | null>(null);
  const mapRef = useRef<CommandMapHandle>(null);

  /* the open analysis section is part of the link, as in the original build */
  const [params, setParams] = useSearchParams();
  const advTab = params.get('x');
  const setAdvTab = useCallback(
    (t: string | null) =>
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          if (t) next.set('x', t);
          else next.delete('x');
          return next;
        },
        { replace: true },
      ),
    [setParams],
  );

  /* keep the shareable filters in the URL even on a bare #/map link */
  useEffect(() => {
    if (!params.get('w') || !params.get('a')) setFilters({});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* selecting a place resets the report's own view state, as the original did */
  useEffect(() => {
    setArea(null);
    setEvId(null);
    setShapAll(false);
    setCapXmlShown(false);
    setExpanded(false);
    if (sel) setFocus((f) => ({ id: sel, seq: (f?.seq ?? 0) + 1 }));
  }, [sel]);

  /* keep the default wind style in step with the motion preference */
  useEffect(() => {
    setWindStyle(reduced ? 'arrows' : 'flow');
  }, [reduced]);

  /* history playback */
  const filtersRef = useRef(filters);
  filtersRef.current = filters;
  const setRef = useRef(setFilters);
  setRef.current = setFilters;
  useEffect(() => {
    if (!play) return;
    const t = window.setInterval(() => {
      const f = filtersRef.current;
      const next = Math.min(TODAY, f.asOf + (f.win > 60 ? 3 : 1) * DAY);
      setRef.current({ asOf: next });
      if (next >= TODAY) setPlay(false);
    }, 240);
    return () => window.clearInterval(t);
  }, [play]);

  const togglePlay = useCallback(() => {
    if (play) {
      setPlay(false);
      return;
    }
    if (filters.asOf >= TODAY - DAY) setFilters({ asOf: Math.min(TODAY, START + filters.win * DAY) });
    setPlay(true);
  }, [play, filters.asOf, setFilters]);

  const setBaseMode = useCallback(
    (b: BaseMode, persist = true) => {
      setBase(b);
      if (persist) store.set('agni-base', b);
    },
    [],
  );

  const onBaseFallback = useCallback(() => {
    setBaseMode('plain', false);
    toast('Could not reach the map tiles. Showing the offline outline instead.');
  }, [setBaseMode, toast]);

  const toggleClass = useCallback(
    (id: string) => {
      const next = new Set(filters.cls);
      if (next.has(id as never)) next.delete(id as never);
      else next.add(id as never);
      if (!next.size) CLS.forEach((c) => next.add(c.id));
      setFilters({ cls: next });
    },
    [filters.cls, setFilters],
  );

  const onSelect = useCallback(
    (id: string) => {
      setEvId(null);
      goPlace(id);
    },
    [goPlace],
  );

  const onArea = useCallback(
    (a: AreaBox) => {
      setDrawing(false);
      setArea(a);
      setEvId(null);
      setExpanded(false);
      if (sel) go('map');
    },
    [sel, go],
  );

  const clearSel = useCallback(() => {
    setArea(null);
    setEvId(null);
    setExpanded(false);
    if (sel) go('map');
  }, [sel, go]);

  const [kpisCollapsed, setKpisCollapsed] = useState(false);
  const [sideCollapsed, setSideCollapsed] = useState(false);

  const toggleKpis = useCallback(() => {
    setKpisCollapsed((c) => !c);
    setTimeout(() => mapRef.current?.invalidateSize(), 60);
  }, []);

  const toggleSide = useCallback(() => {
    setSideCollapsed((c) => !c);
    setTimeout(() => mapRef.current?.invalidateSize(), 60);
  }, []);

  /* keyboard: j/k walk the queue, l toggles layers, q toggles side panel, escape backs out */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = document.activeElement;
      if (el && /INPUT|TEXTAREA|SELECT/.test(el.tagName)) return;
      if (e.key === 'Escape') {
        if (advTab) {
          setAdvTab(null);
          return;
        }
        if (drawing) setDrawing(false);
        else if (sel || area) clearSel();
        return;
      }
      if (advTab) return;
      if (e.key === 'q') {
        toggleSide();
        return;
      }
      if (e.key === 'j' || e.key === 'k') {
        const l = cur.queue;
        if (!l.length) return;
        const i = Math.max(0, l.findIndex((g: Group) => g.p.id === sel));
        const j = Math.min(l.length - 1, Math.max(0, i + (e.key === 'j' ? 1 : -1)));
        onSelect(l[j].p.id);
      }
      if (e.key === 'l') setShowLayers((s) => !s);
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [cur.queue, sel, area, drawing, advTab, clearSel, onSelect, toggleSide]);

  const place = sel ? PMAP[sel] : null;
  const ev = place ? pickEvent(place, filters, evId) : null;
  const abn = cur.queue.filter((g) => sev(g.top) === 3).length;
  const csvName = place ? `sparc-${place.code === 'T' ? place.id : place.code}-events.csv` : '';

  return (
    <div className={'cmd' + (expanded && (sel || area) ? ' expanded' : '') + (sideCollapsed ? ' side-collapsed' : '')}>
      <div className="mapcol">
        <Filters />
        <Kpis collapsed={kpisCollapsed} onToggle={toggleKpis} />
        <div className="mapbox" id="mapbox">
          <CommandMap
            ref={mapRef}
            groups={cur.groups}
            sel={sel}
            layers={layers}
            windH={windH}
            windStyle={windStyle}
            drawing={drawing}
            area={area}
            base={base}
            focus={focus}
            onSelect={onSelect}
            onArea={onArea}
            onBaseFallback={onBaseFallback}
          />
          {sideCollapsed && (
            <button
              className="side-dock-toggle"
              aria-label="Expand Priority Queue and incidents panel"
              title="Expand Priority Queue (or press Q)"
              onClick={toggleSide}
            >
              <Icon name="chevron-left" size={15} />
              <span>Priority Queue</span>
              <span className="badge">{cur.queue.length}</span>
            </button>
          )}
          <MapOverlays
            layers={layers}
            onLayer={(k) => setLayers((l) => ({ ...l, [k]: !l[k] }))}
            showLayers={showLayers}
            onToggleLayers={() => setShowLayers((s) => !s)}
            drawing={drawing}
            onToggleDraw={() => setDrawing((d) => !d)}
            base={base}
            onBase={(b) => setBaseMode(b)}
            windH={windH}
            onWindH={setWindH}
            windStyle={windStyle}
            onWindStyle={setWindStyle}
            cls={filters.cls as Set<string>}
            onToggleClass={toggleClass}
            onZoomIn={() => mapRef.current?.zoom(1)}
            onZoomOut={() => mapRef.current?.zoom(-1)}
            onReset={() => mapRef.current?.reset()}
          />
        </div>
        <Scrubber play={play} onTogglePlay={togglePlay} />
      </div>

      <aside className="side" aria-label="Incidents">
        <div className="side-head">
          {area ? (
            <>
              <button className="btn sm" onClick={() => setArea(null)}>
                <Icon name="back" size={14} />
                All incidents
              </button>
              <h2>Area report</h2>
              <span style={{ flex: 1 }} />
              <button
                className="btn ic sm"
                aria-label={expanded ? 'Shrink' : 'Expand'}
                onClick={() => {
                  setExpanded((x) => !x);
                  mapRef.current?.invalidateSize();
                }}
              >
                <Icon name={expanded ? 'shrink' : 'expand'} size={15} />
              </button>
              <button
                className="btn ic sm"
                title="Collapse panel"
                aria-label="Collapse panel"
                onClick={toggleSide}
              >
                <Icon name="chevron-right" size={15} />
              </button>
            </>
          ) : place && ev ? (
            <>
              <button
                className="btn sm"
                onClick={() => {
                  if (expanded) {
                    setExpanded(false);
                    mapRef.current?.invalidateSize();
                  } else {
                    clearSel();
                  }
                }}
              >
                <Icon name="back" size={14} />
                {expanded ? 'Back to map' : 'All incidents'}
              </button>
              <h2 style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>Place report</h2>
              <span style={{ flex: 1 }} />
              <button
                className="btn ic sm"
                aria-label={expanded ? 'Shrink' : 'Expand'}
                onClick={() => {
                  setExpanded((x) => !x);
                  mapRef.current?.invalidateSize();
                }}
              >
                <Icon name={expanded ? 'shrink' : 'expand'} size={15} />
              </button>
              <button className="btn ic sm" aria-label="Close report" onClick={clearSel}>
                <Icon name="x" size={15} />
              </button>
              <button
                className="btn ic sm"
                title="Collapse panel"
                aria-label="Collapse panel"
                onClick={toggleSide}
              >
                <Icon name="chevron-right" size={15} />
              </button>
            </>
          ) : (
            <>
              <h2>Priority queue</h2>
              <span className="dim small">
                {cur.queue.length} places{abn ? `, ${abn} abnormal` : ''}
              </span>
              <span style={{ flex: 1 }} />
              <button
                className="btn ic sm"
                title="Collapse Priority Queue"
                aria-label="Collapse Priority Queue"
                onClick={toggleSide}
              >
                <Icon name="chevron-right" size={15} />
              </button>
            </>
          )}
        </div>
        <div className="side-body">
          {area ? (
            <AreaReport
              area={area}
              evs={cur.evs}
              asOf={filters.asOf}
              win={filters.win}
              sel={sel}
              onSelect={onSelect}
              onExport={() => saveFile('agni-netra-area-events.csv', eventsCsv(cur.evs.filter((e) => inArea(area, e))), 'CSV')}
              onClear={() => setArea(null)}
              expanded={expanded}
            />
          ) : place && ev ? (
            <PlaceReport
              p={place}
              ev={ev}
              asOf={filters.asOf}
              expanded={expanded}
              shapAll={shapAll}
              capXmlShown={capXmlShown}
              onToggleShap={() => setShapAll((s) => !s)}
              onToggleCap={() => setCapXmlShown((s) => !s)}
              onPickEvent={setEvId}
              onOpenAdvanced={() => setAdvTab(advTab ?? 'wind')}
              onWatch={() => toast(toggleWatch(place.id) ? 'Added to watchlist' : 'Removed from watchlist')}
              onCopyLink={() => copyText(location.href, 'Link copied')}
              onCsv={() => saveFile(csvName, eventsCsv(EVP[place.id] || []), 'CSV')}
              onGeo={() => saveFile(csvName.replace('.csv', '.json'), eventsGeo(EVP[place.id] || []), 'GeoJSON')}
            />
          ) : cur.queue.length ? (
            <>
              <div className="dim small" style={{ padding: '0 2px' }}>
                Abnormal first, then needs review, then newest.
              </div>
              <QueueList groups={cur.queue} sel={sel} onSelect={onSelect} limit={qshow} />
              {cur.queue.length > qshow && (
                <button className="btn" style={{ justifySelf: 'center' }} onClick={() => setQshow((n) => n + 30)}>
                  Show {Math.min(30, cur.queue.length - qshow)} more
                </button>
              )}
            </>
          ) : (
            <Empty title="Nothing matches these filters">
              Widen the date window, turn more classes on, or lower the minimum confidence.
            </Empty>
          )}
        </div>
      </aside>

      {place && advTab && (
        <AdvancedAnalysis
          placeId={place.id}
          tab={advTab}
          onTab={(t) => setAdvTab(t)}
          onClose={() => {
            const pid = place.id;
            setAdvTab(null);
            if (pid === sel) mapRef.current?.invalidateSize();
          }}
          base={base}
          onBaseFallback={onBaseFallback}
        />
      )}
    </div>
  );
}

