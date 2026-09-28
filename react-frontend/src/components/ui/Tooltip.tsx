/* =====================================================================
   One tooltip element for the whole app, driven from React event props
   instead of the original global pointerover delegation.
   ===================================================================== */
import { createContext, useCallback, useContext, useMemo, useState, type PointerEvent, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { clamp } from '../../lib/core';

interface TipState {
  text: string;
  x: number;
  y: number;
}

const Ctx = createContext<{ show: (text: string, x: number, y: number) => void; hide: () => void } | null>(null);

export function TooltipProvider({ children }: { children: ReactNode }) {
  const [tip, setTip] = useState<TipState | null>(null);

  const show = useCallback((text: string, x: number, y: number) => setTip({ text, x, y }), []);
  const hide = useCallback(() => setTip(null), []);
  const value = useMemo(() => ({ show, hide }), [show, hide]);

  return (
    <Ctx.Provider value={value}>
      {children}
      {tip && <TipBody tip={tip} />}
    </Ctx.Provider>
  );
}

function TipBody({ tip }: { tip: TipState }) {
  return createPortal(
    <div
      className="tip"
      role="tooltip"
      ref={(el) => {
        if (!el) return;
        el.style.left = clamp(tip.x + 14, 8, window.innerWidth - el.offsetWidth - 8) + 'px';
        el.style.top = clamp(tip.y + 14, 8, window.innerHeight - el.offsetHeight - 8) + 'px';
      }}
    >
      {tip.text}
    </div>,
    document.body,
  );
}

export interface TipProps {
  onPointerEnter: (e: PointerEvent<HTMLElement>) => void;
  onPointerLeave: () => void;
}

/** Spread onto any element that should explain itself: <button {...useTip('...')}> */
export function useTip(text?: string): Partial<TipProps> {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error('useTip must be used inside TooltipProvider');
  if (!text) return {};
  return {
    onPointerEnter: (e: PointerEvent<HTMLElement>) => {
      const r = e.currentTarget.getBoundingClientRect();
      ctx.show(text, r.left, r.bottom);
    },
    onPointerLeave: ctx.hide,
  };
}
