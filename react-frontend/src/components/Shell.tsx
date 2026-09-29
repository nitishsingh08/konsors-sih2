/* =====================================================================
   Page ids, the navigation rail and the header. The shareable filter
   state rides in the URL query, so nav links keep it.
   ===================================================================== */
import { useCallback, useEffect, useRef, useState } from 'react';
import { Navigate, Outlet, Route, Routes, useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { Icon, type IconName } from './ui/Icon';
import { CommandPage } from '../pages/CommandPage';
import { RegionsPage } from '../pages/RegionsPage';
import { WatchlistPage } from '../pages/WatchlistPage';
import { ModelPage } from '../pages/ModelPage';
import { MethodsPage } from '../pages/MethodsPage';
import { AlertsPage } from '../pages/AlertsPage';
import { PMAP } from '../data/places';
import { setSavedView } from '../map/leafletBase';
import { searchPlaces } from '../state/selectors';
import { useTheme } from '../state/ThemeProvider';
import { Glyph } from './ui/Glyph';
import logoForLight from '../assets/logo.svg';
import logoForDark from '../assets/logo-on-dark.svg';
import { useTip } from './ui/Tooltip';
import { clamp } from '../lib/core';
import type { Place } from '../lib/types';

export type PageId = 'map' | 'regions' | 'watch' | 'model' | 'methods' | 'alerts';

export const NAV: readonly { id: PageId; label: string; icon: IconName }[] = [
  { id: 'map', label: 'Command', icon: 'map' },
  { id: 'regions', label: 'Regions', icon: 'bars' },
  { id: 'watch', label: 'Watchlist', icon: 'star' },
  { id: 'model', label: 'Model', icon: 'gauge' },
  { id: 'methods', label: 'Methods', icon: 'book' },
  { id: 'alerts', label: 'Hand-off', icon: 'send' },
];

const PAGES = new Set<string>(NAV.map((n) => n.id));

/** navigate without dropping the current filter query */
export function useNav() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const { page } = useParams();

  const go = useCallback(
    (to: PageId, sel?: string | null) => {
      const q = params.toString();
      navigate({ pathname: '/' + to + (sel ? '/' + sel : ''), search: q ? '?' + q : '' });
    },
    [navigate, params],
  );

  /** select a place: from another page it also frames the map on it */
  const goPlace = useCallback(
    (id: string) => {
      const p = PMAP[id];
      if (!p) return;
      if (page !== 'map') setSavedView({ lat: p.lat, lon: p.lon, zoom: 9 });
      go('map', id);
    },
    [go, page],
  );

  return { go, goPlace, page };
}

/**
 * The rail, the header and the routed view. This is a layout route, so the
 * page and the selected place are read from the URL with useParams().
 */
export function Shell() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Navigate to="/map" replace />} />
        <Route path="/:page" element={<PageSwitch />} />
        <Route path="/:page/:sel" element={<PageSwitch />} />
        <Route path="*" element={<Navigate to="/map" replace />} />
      </Route>
    </Routes>
  );
}

/** the page id in the URL, or the command page if it is not one of ours */
function usePageId(): PageId {
  const { page } = useParams();
  return PAGES.has(page as PageId) ? (page as PageId) : 'map';
}

function Layout() {
  const active = usePageId();
  const { theme } = useTheme();
  return (
    <>
      <nav className="rail" aria-label="Main">
        <div className="logo" aria-hidden="true">
          {/* the dark colourway is the one that reads on the dark rail panel */}
          <img src={theme === 'dark' ? logoForDark : logoForLight} alt="" width={38} height={38} />
        </div>
        <RailButtons active={active} />
        <div className="sp" />
      </nav>
      <div className="main">
        <TopBar />
        <main id="view" aria-live="polite">
          <Outlet />
        </main>
      </div>
    </>
  );
}

function PageSwitch() {
  const page = usePageId();
  const { sel } = useParams();
  const valid = sel && PMAP[sel] ? sel : null;
  switch (page) {
    case 'regions':
      return <RegionsPage />;
    case 'watch':
      return <WatchlistPage />;
    case 'model':
      return <ModelPage />;
    case 'methods':
      return <MethodsPage />;
    case 'alerts':
      return <AlertsPage />;
    default:
      return <CommandPage sel={valid} />;
  }
}

function RailButtons({ active }: { active: PageId }) {
  const { go } = useNav();
  return (
    <>
      {NAV.map((n) => (
        <button key={n.id} className="nv" aria-current={active === n.id ? 'page' : 'false'} onClick={() => go(n.id)}>
          <Icon name={n.icon} size={21} />
          <span>{n.label}</span>
        </button>
      ))}
    </>
  );
}

function TopBar() {
  const { theme, toggle } = useTheme();
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = document.activeElement;
      const typing = !!el && /INPUT|TEXTAREA|SELECT/.test(el.tagName);
      if (e.key === '/' && !typing) {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, []);

  return (
    <header className="top">
      <div className="brand">
        <b>SPARC</b>
        <span>Thermal source monitoring</span>
      </div>
      <SearchBox inputRef={inputRef} />
      <span className="sp" />
      <span className="pill" {...useTip('Sample value. The live pipeline shows the age of the newest FIRMS data here.')}>
        <i />
        FIRMS data 2 h 41 min old
      </span>
      <span className="sample" {...useTip('Every place, event and number in this build is generated sample data.')}>
        Sample data
      </span>
      <button className="btn ic" aria-label="Switch theme" onClick={toggle}>
        <Icon name={theme === 'dark' ? 'sun' : 'moon'} size={17} />
      </button>
    </header>
  );
}

function SearchBox({ inputRef }: { inputRef: React.RefObject<HTMLInputElement | null> }) {
  const { goPlace } = useNav();
  const [q, setQ] = useState('');
  const [open, setOpen] = useState(false);
  const [si, setSi] = useState(0);
  const [results, setResults] = useState<Place[]>([]);
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const list = searchPlaces(q);
    setResults(list);
    setSi(0);
    setOpen(q.trim().length > 0);
  }, [q]);

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, []);

  const pick = (p: Place) => {
    setOpen(false);
    setQ('');
    inputRef.current?.blur();
    goPlace(p.id);
  };

  return (
    <div className="search" ref={boxRef}>
      <span className="ic">
        <Icon name="search" size={16} />
      </span>
      <input
        id="q"
        ref={inputRef}
        type="search"
        autoComplete="off"
        placeholder="Search a place, district or site code"
        aria-label="Search places"
        role="combobox"
        aria-expanded={open}
        aria-controls="sugg"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
            e.preventDefault();
            setSi((i) => clamp(i + (e.key === 'ArrowDown' ? 1 : -1), 0, Math.max(0, results.length - 1)));
          } else if (e.key === 'Enter' && results[si]) {
            e.preventDefault();
            pick(results[si]);
          } else if (e.key === 'Escape') {
            setQ('');
            setOpen(false);
            e.currentTarget.blur();
          }
        }}
      />
      <kbd>/</kbd>
      {open && (
        <div className="sugg" id="sugg" role="listbox">
          {results.length ? (
            results.map((p, i) => (
              <button key={p.id} role="option" aria-selected={i === si} className={i === si ? 'on' : ''} onClick={() => pick(p)}>
                <Glyph cls={p.cls} size={12} />
                <span>
                  <b>{p.name}</b>
                  <br />
                  <small>
                    {p.district}, {p.state}
                  </small>
                </span>
              </button>
            ))
          ) : (
            <div style={{ padding: 12 }} className="dim">
              No place matches "{q}".
            </div>
          )}
        </div>
      )}
    </div>
  );
}
