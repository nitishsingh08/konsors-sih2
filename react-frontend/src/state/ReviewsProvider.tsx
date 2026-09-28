import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react';
import { store } from '../lib/core';
import { fmtDT, TODAY } from '../lib/time';
import { PMAP, EVENTS } from '../data/places';
import { csvCell } from '../data/analysis';
import type { ClassId, Review, ReviewMap, Verdict } from '../lib/types';

interface ReviewsCtx {
  reviews: ReviewMap;
  reviewsOf: (eventId: string) => Review[];
  addReview: (eventId: string, verdict: Verdict, cls: ClassId | null, note: string) => void;
  toggleWatch: (placeId: string) => boolean;
  watch: Set<string>;
  total: number;
}

const Ctx = createContext<ReviewsCtx | null>(null);
const KEY_REVIEWS = 'agni-reviews';
const KEY_WATCH = 'agni-watch';

const load = (key: string): ReviewMap => {
  try {
    return JSON.parse(store.get(key) || '{}') || {};
  } catch {
    return {};
  }
};

export function ReviewsProvider({ children }: { children: ReactNode }) {
  const [reviews, setReviews] = useState<ReviewMap>(() => load(KEY_REVIEWS));
  const [watch, setWatch] = useState<Set<string>>(
    () => new Set((store.get(KEY_WATCH) || '').split(',').filter(Boolean)),
  );

  const reviewsOf = useCallback((eventId: string) => reviews[eventId] || [], [reviews]);

  const addReview = useCallback((eventId: string, verdict: Verdict, cls: ClassId | null, note: string) => {
    setReviews((prev) => {
      const next: ReviewMap = { ...prev, [eventId]: [...(prev[eventId] || []), { verdict, cls, note, at: fmtDT(TODAY) }] };
      store.set(KEY_REVIEWS, JSON.stringify(next));
      return next;
    });
  }, []);

  const toggleWatch = useCallback(
    (placeId: string) => {
      const had = watch.has(placeId);
      const next = new Set(watch);
      if (had) next.delete(placeId);
      else next.add(placeId);
      store.set(KEY_WATCH, [...next].join(','));
      setWatch(next);
      return !had;
    },
    [watch],
  );

  const total = useMemo(() => Object.values(reviews).reduce((a, l) => a + l.length, 0), [reviews]);

  const value = useMemo(
    () => ({ reviews, reviewsOf, addReview, toggleWatch, watch, total }),
    [reviews, reviewsOf, addReview, toggleWatch, watch, total],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useReviews(): ReviewsCtx {
  const v = useContext(Ctx);
  if (!v) throw new Error('useReviews must be used inside ReviewsProvider');
  return v;
}

/** CSV of every review saved on this device, for building a labelled dataset. */
export function reviewsCsv(reviews: ReviewMap): string {
  return ['event_id,place,model_class,model_confidence,verdict,analyst_class,note,saved_at']
    .concat(
      Object.entries(reviews).flatMap(([id, l]) => {
        const ev = EVENTS.find((e) => e.id === id);
        if (!ev) return [];
        return l.map((r) =>
          [id, PMAP[ev.pid].name, ev.cls, ev.conf.toFixed(3), r.verdict, r.cls || '', r.note || '', r.at]
            .map(csvCell)
            .join(','),
        );
      }),
    )
    .join('\n');
}
