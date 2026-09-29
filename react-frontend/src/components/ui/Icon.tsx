/* =====================================================================
   Inline icons. The same stroke set as frontend/js/ui.js, as React nodes
   so nothing relies on injecting markup.
   ===================================================================== */
import type { JSX, ReactNode } from 'react';

export type IconName =
  | 'map' | 'bars' | 'star' | 'gauge' | 'book' | 'send' | 'search' | 'sun' | 'moon'
  | 'layers' | 'draw' | 'plus' | 'minus' | 'play' | 'pause' | 'x' | 'expand' | 'shrink'
  | 'download' | 'copy' | 'back' | 'reset' | 'check'
  | 'chevron-up' | 'chevron-down' | 'chevron-left' | 'chevron-right';

const PATHS: Record<IconName, JSX.Element> = {
  map: (
    <>
      <path d="M3 6.5l6-2.5 6 2.5 6-2.5v13l-6 2.5-6-2.5-6 2.5z" />
      <path d="M9 4v13.5M15 6.5V20" />
    </>
  ),
  bars: (
    <>
      <path d="M4 20v-7M10 20V5M16 20v-10M22 20H3" />
    </>
  ),
  star: <path d="M12 3.5l2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.9 6.8 19.6l1-5.8L3.5 9.7l5.9-.9z" />,
  gauge: (
    <>
      <path d="M4 17a8 8 0 1 1 16 0" />
      <path d="M12 17l4-5" />
    </>
  ),
  book: (
    <>
      <path d="M5 5a2 2 0 0 1 2-2h11v16H7a2 2 0 0 0-2 2z" />
      <path d="M5 19V5" />
    </>
  ),
  send: (
    <>
      <path d="M21 3L3 10.5l6.5 2.5L12 20z" />
      <path d="M9.5 13L21 3" />
    </>
  ),
  search: (
    <>
      <circle cx="11" cy="11" r="6.5" />
      <path d="M16 16l4.5 4.5" />
    </>
  ),
  sun: (
    <>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M5 5l1.8 1.8M17.2 17.2L19 19M5 19l1.8-1.8M17.2 6.8L19 5" />
    </>
  ),
  moon: <path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z" />,
  layers: (
    <>
      <path d="M12 3l9 5-9 5-9-5z" />
      <path d="M3 13l9 5 9-5" />
    </>
  ),
  draw: <rect x="4" y="4" width="16" height="16" rx="2" strokeDasharray="3 3" />,
  plus: <path d="M12 5v14M5 12h14" />,
  minus: <path d="M5 12h14" />,
  play: <path d="M7 4.5v15l12-7.5z" fill="currentColor" />,
  pause: <path d="M8 5v14M16 5v14" />,
  x: <path d="M6 6l12 12M18 6L6 18" />,
  expand: <path d="M4 10V4h6M20 14v6h-6M4 4l6 6M20 20l-6-6" />,
  shrink: <path d="M10 4v6H4M14 20v-6h6M10 10L4 4M14 14l6 6" />,
  download: <path d="M12 4v11M7 11l5 5 5-5M5 20h14" />,
  copy: (
    <>
      <rect x="8" y="8" width="12" height="12" rx="2" />
      <path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" />
    </>
  ),
  back: <path d="M15 5l-7 7 7 7" />,
  reset: (
    <>
      <circle cx="12" cy="12" r="6.5" />
      <path d="M12 2v4M12 18v4M2 12h4M18 12h4" />
    </>
  ),
  check: <path d="M5 12.5l4.5 4.5L19 7.5" />,
  'chevron-up': <path d="M18 15l-6-6-6 6" />,
  'chevron-down': <path d="M6 9l6 6 6-6" />,
  'chevron-left': <path d="M15 19l-6-6 6-6" />,
  'chevron-right': <path d="M9 5l6 6-6 6" />,
};

export function Icon({
  name,
  size = 18,
  children,
  ...rest
}: {
  name: IconName;
  size?: number;
  children?: ReactNode;
} & Omit<JSX.IntrinsicElements['svg'], 'name'>) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.7}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {children ?? PATHS[name]}
    </svg>
  );
}
