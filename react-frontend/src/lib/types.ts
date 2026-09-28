/* =====================================================================
   Domain types for AGNI-NETRA. These mirror the shapes the vanilla build
   produced with plain objects, so the sample-data generators keep the same
   numbers and the same reads.
   ===================================================================== */

export type LonLat = [number, number];

/* ---- source classes and statuses ---- */
export type ClassId =
  | 'wildfire'
  | 'agricultural_burning'
  | 'gas_flare'
  | 'mining'
  | 'industrial'
  | 'unknown';

export type StatusId = 'abnormal' | 'routine' | 'baseline_building' | 'not_applicable';

export type FamilyId =
  | 'thermal'
  | 'temporal'
  | 'spread'
  | 'terrain'
  | 'fuel'
  | 'industry'
  | 'weather'
  | 'smoke'
  | 'sar'
  | 'chem'
  | 'quality';

export interface ClassDef {
  id: ClassId;
  label: string;
  blurb: string;
}

export interface StatusDef {
  label: string;
  k: 'abn' | 'rt' | 'bb' | 'na';
  rank: number;
  tip: string;
}

export interface FamilyDef {
  id: FamilyId;
  label: string;
}

/* ---- places and events ---- */
export type PlaceKind = 'site' | 'transient';
export type CoverLevel = 'good' | 'partial' | 'weak';

export interface FacilityDistances {
  refinery: number;
  power: number;
  steel: number;
  mine: number;
  landfill: number;
}

export interface PlaceContext {
  parts: [string, number][];
  d: FacilityDistances;
  fac: boolean | null;
  ndvi: number;
  nbr: number;
  ndbi: number;
  built: number;
  valid: number;
  sceneDays: number;
  vpd: number;
  rain: number;
  elev: number;
  slope: number;
}

export interface Observation {
  t: number;
  frp: number;
  cnt: number;
  sensor: string;
  z: number | null;
}

export interface Baseline {
  n: number;
  mLn: number;
  med: number;
  mad: number;
  sd: number;
  p95: number;
  lo: number;
  hi: number;
}

export interface PlaceSeries {
  obs: Observation[];
  gaps: [number, number][];
  base?: Baseline;
}

export interface PlaceStats {
  d30: number;
  d90: number;
  months: number;
  streak: number;
  last: number | null;
  sinceLast: number | null;
}

export interface Place {
  id: string;
  kind: PlaceKind;
  code: string;
  name: string;
  state: string;
  district: string;
  lat: number;
  lon: number;
  cls: ClassId;
  type: string;
  med: number;
  mad: number;
  cover: CoverLevel;
  firstSeen: number;
  /** memoised series, see data/places.ts */
  _ser?: PlaceSeries;
  /** memoised surroundings, see data/places.ts */
  _ctx?: PlaceContext;
}

export interface Event {
  id: string;
  pid: string;
  t: number;
  cls: ClassId;
  conf: number;
  cls2: ClassId;
  p2: number;
  status: StatusId;
  z: number | null;
  frp: number;
  sensor: string;
  nDet: number;
  area: number;
  review: boolean;
  hr: number;
  _pts?: LonLat[];
  _f?: FeatureVector;
  _shap?: ShapResult;
  _wx?: Weather;
}

export interface NearbySource {
  name: string;
  cls: ClassId;
  km: number;
  ang: number;
  st: 'abnormal' | 'routine';
}

export interface Metric {
  id: ClassId;
  p: number;
  r: number;
  n: number;
  f1: number;
}

export interface StateOutline {
  n: string;
  r: LonLat[][];
}

/* ---- weather, plume, spread ---- */
export interface Weather {
  t0: number;
  n: number;
  spd: number[];
  gust: number[];
  dir: number[];
  temp: number[];
  rh: number[];
  vpd: number[];
  rain: number[];
  ffmc: number[];
  dmc: number;
  dc: number;
  rainPast24: number;
}

export interface Wind {
  spd: number;
  dir: number;
  u: number;
  v: number;
}

/** u east, v north, metres per second on a regular lat/lon grid */
export interface WindGrid {
  lo0: number;
  la1: number;
  dx: number;
  dy: number;
  nx: number;
  ny: number;
  u: Float32Array;
  v: Float32Array;
}

export type PlumeTrajPoint = [lon: number, lat: number, metres: number, hours: number];

export interface Plume {
  traj: PlumeTrajPoint[];
  core: LonLat[];
  outer: LonLat[];
  reachKm: number;
  head: LonLat;
  bearing: number;
  H: number;
}

export interface TerrainPoint {
  km: number;
  elev: number;
  lon: number;
  lat: number;
}

export interface Terrain {
  pts: TerrainPoint[];
  meanSlope: number;
  maxUp: number;
}

export interface SpreadHorizon {
  t: number;
  isi: number;
  rate: number;
  rateLo: number;
  rateHi: number;
  D: number;
  dirTo: number;
  LB: number;
  slopeF: number;
  meanSlope: number;
  area: number;
  ring: LonLat[];
  cls: string;
}

export interface SpreadOutlook {
  horizons: SpreadHorizon[];
  unc: LonLat[];
  cls: string;
}

export interface FootprintFrame {
  t: number;
  area: number;
  ring: LonLat[];
}

export interface ExposureZone {
  name: string;
  area: number;
  settlements: number;
  schools: number;
  clinics: number;
  roadKm: number;
  water: number;
}

export interface FreshnessRow {
  name: string;
  label: string;
  hours: number;
  note: string;
}

export interface SimilarRow {
  e: Event;
  p: Place;
  sim: number;
  outcome: string;
}

/* ---- features and explanations ---- */
export interface FeatureDef {
  key: string;
  g: FamilyId;
  label: string;
  unit: string;
  src: string;
  mu: number;
  sd: number;
  lo: number;
  hi: number;
  /** x = model sees log, shown real units | k = also builds labels | b = yes/no */
  fl: string;
}

export interface FeatureEntry {
  d: FeatureDef;
  raw: number;
  v: number | null;
  pct: number | null;
  missing: boolean;
}

export interface FeatureVector {
  by: Record<string, FeatureEntry>;
  missing: number;
  W: Weather;
  ang: Record<string, number>;
}

export interface ShapFeature {
  k: string;
  l: string;
  f: FamilyId;
  w: number;
  v: string;
}

export interface ShapResult {
  feats: ShapFeature[];
  fam: Record<FamilyId, number>;
  all: Record<string, number>;
}

/* ---- analyst reviews ---- */
export type Verdict = 'confirm' | 'change' | 'false' | 'field';

export interface Review {
  verdict: Verdict;
  cls: ClassId | null;
  note: string;
  at: string;
}

export type ReviewMap = Record<string, Review[]>;

export interface CapLogEntry {
  t: string;
  n: string;
}
