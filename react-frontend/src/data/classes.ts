/* =====================================================================
   Source classes, statuses and feature families. Port of the class and
   status tables in frontend/js/data/sample.js.
   ===================================================================== */
import type { ClassDef, ClassId, FamilyDef, StatusDef, StatusId } from '../lib/types';

export const CLS: readonly ClassDef[] = [
  { id: 'wildfire', label: 'Wildfire', blurb: 'Vegetation fire in forest, grass or scrub. Short-lived, no normal level.' },
  { id: 'agricultural_burning', label: 'Agricultural burning', blurb: 'Crop residue burning on farmland, strongly seasonal.' },
  { id: 'gas_flare', label: 'Gas flare', blurb: 'Burn-off at oil and gas sites. Persistent, so judged against its own history.' },
  { id: 'mining', label: 'Mining heat', blurb: 'Coal seam and mine fires, or heat from mining works.' },
  { id: 'industrial', label: 'Industrial source', blurb: 'Furnaces, stacks and process heat, or a fire inside a facility.' },
  { id: 'unknown', label: 'Unknown', blurb: 'Low confidence, or not a fire at all (reflective surfaces, waste heaps).' },
];

export const CLSMAP: Record<ClassId, ClassDef> = Object.fromEntries(
  CLS.map((c) => [c.id, c]),
) as Record<ClassId, ClassDef>;

export const STATUS: Record<StatusId, StatusDef> = {
  abnormal: { label: 'Abnormal', k: 'abn', rank: 4, tip: 'Unusual for this site compared with its own history.' },
  routine: { label: 'Routine', k: 'rt', rank: 1, tip: 'Within the normal range for this site.' },
  baseline_building: { label: 'Baseline building', k: 'bb', rank: 2, tip: 'Not enough history yet to judge. No claim is made about normal or abnormal.' },
  not_applicable: { label: 'Transient fire', k: 'na', rank: 0, tip: 'Wildfires and crop burning are short events, so there is no normal level to compare with.' },
};

export const STATUS_OPTIONS: readonly { value: StatusId | 'all' | 'review'; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'abnormal', label: 'Abnormal' },
  { value: 'routine', label: 'Routine' },
  { value: 'baseline_building', label: 'Baseline building' },
  { value: 'not_applicable', label: 'Transient fire' },
  { value: 'review', label: 'Needs review' },
];

export const FAMILIES: readonly FamilyDef[] = [
  { id: 'thermal', label: 'Heat' },
  { id: 'temporal', label: 'Timing' },
  { id: 'spread', label: 'Movement and growth' },
  { id: 'terrain', label: 'Terrain' },
  { id: 'fuel', label: 'Fuel and land cover' },
  { id: 'industry', label: 'Industry nearby' },
  { id: 'weather', label: 'Weather and fire danger' },
  { id: 'smoke', label: 'Smoke and burn signs' },
  { id: 'sar', label: 'Radar and structure' },
  { id: 'chem', label: 'Air chemistry' },
  { id: 'quality', label: 'Data quality' },
];

/** Fire spread outlook is only estimated for vegetation fires. */
export const isVeg = (c: ClassId): boolean => c === 'wildfire' || c === 'agricultural_burning';
