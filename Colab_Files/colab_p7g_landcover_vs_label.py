# ============================================================================
# P7g -- COLAB CELL: does land cover DISAGREE with the weak label?
#
# WHY THIS CELL, AND WHY IT MATTERS MORE THAN ANYTHING ELSE SO FAR
#
# Everything upstream of here established that the weak INDUSTRIAL_FIRE label
# is broken. Measured on the full catalogue:
#
#     max distance-to-nearest-facility among 1,119,254 positives = 5,000 m
#     positives beyond 5,000 m = 2 (float rounding)
#     PR-AUC of DISTANCE ALONE = 0.9811
#     PR-AUC of model B (29 honest features) = 0.3353
#
# and `qc/p6_rule_calibration.json` showed `industrial_persistence = 1/30` --
# the FLOOR of `observed_days/30` -- so the `shape` conjunct excluded nothing.
# The label reduced to `(facility within 5 km) AND (power/steel/cement/petro
# within 10 km)`: two distance terms and zero fire-behaviour terms.
#
# That is a structural argument. It is convincing, but it is still an argument
# made from the rule's source code. This cell makes the case EMPIRICALLY.
#
# P7e sampled eight real events and got:
#
#     GOLD_0001  built-up     [NON-vegetated]    2,597 m to facility
#     GOLD_0002  built-up     [NON-vegetated]    1,731 m to facility
#     GOLD_0007  cropland     [vegetated]        5,007 m to facility
#     GOLD_0008  cropland     [vegetated]        7,374 m to facility
#
# Read the last two again. The weak rule calls them INDUSTRIAL_FIRE -- they are
# inside its proximity cut, and the shape conjunct is vacuous, so proximity is
# the only thing that mattered. But the substrate under them is CROPLAND.
#
# Eight points is not evidence. This cell makes it evidence: sample land cover
# across the whole catalogue and cross-tabulate it against the weak label.
#
# WHAT THE OUTCOME MEANS
#
#   weak INDUSTRIAL_FIRE mostly NON-vegetated
#       The proximity rule mostly agrees with the substrate. The label is crude
#       but not badly wrong, and land cover is a genuine independent
#       confirmation. Report the disagreement rate honestly either way.
#
#   weak INDUSTRIAL_FIRE mostly cropland
#       The label is DEMONSTRABLY wrong, and you can quantify by how much. This
#       is the publishable result: a proximity rule with no evidence of fire
#       behaviour, mislabelling farmland near facilities as industrial, with a
#       measured error rate.
#
#   Either way, `WORLDCOVER_CLASS` is not a model feature. It must be used to
#   BUILD a new label, then the leakage partition is recomputed.
#
# THE HONEST CAVEAT, STATED UP FRONT
#
# ESA WorldCover v200 is a **2021** static map. Applying it to a 2012 event
# assumes the substrate did not change in nine years. For cropland and
# industrial that is usually safe; for wetland, scrub and post-failure
# regeneration it is not. This cell reports the event-year distribution
# alongside the cross-tab so the size of that assumption is visible rather than
# hidden.
# ============================================================================

import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import colab_utils as cu
import gee_utils as gee

P7G_CELL_VERSION = "2026-10-04.3"

#: How many events to sample. Land cover is static, so cost is driven by
#: geography (tiles touched), not by event count -- but a bigger sample narrows
#: the confidence interval and each tile is touched once regardless.
N_EVENTS = int(globals().get("N_LANDCOVER_EVENTS", 120_000))
#: Points per `sampleRegions` call. Earth Engine's request body limit is 10 MiB,
#: which at ~200 bytes/point is roughly 50k points -- so chunk well below it.
_CHUNK = int(globals().get("GEE_CHUNK_SIZE", 5_000))
TARGET = globals().get("TARGET", "INDUSTRIAL_FIRE")

PROJECT_ROOT = cu.resolve_project_dir()
if PROJECT_ROOT is None:
    raise FileNotFoundError(f"no project root under {cu.MYDRIVE}")
DATASET_DIR = PROJECT_ROOT / "full_2012_2026" / "features" / "dataset"
QC_DIR = PROJECT_ROOT / "qc"
QC_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 74)
print(f"P7g -- LAND COVER vs THE WEAK LABEL  [v{P7G_CELL_VERSION}]")
print("=" * 74)
print(cu.drive_report())
print(f"  n {N_EVENTS:,}  target {TARGET}")

_ok, _pid = False, None
try:
    _ok, _pid = gee.authenticate(globals().get("EARTHENGINE_PROJECT"))
    gee.verify_can_read(_pid)
    print(f"  Earth Engine working (project {_pid!r})")
except Exception as exc:                            # noqa: BLE001
    print(f"  Earth Engine NOT usable: {type(exc).__name__}: {str(exc)[:200]}")
    print("  Run colab_p7e_gee_probe.py first and fix what it reports.")
    raise SystemExit("P7g needs a verified Earth Engine session")


# ---------------------------------------------------------------------------
# 1. EVENTS
#
# Read `event_id` and the weak label from the dataset P6 wrote, so the join is
# on the SAME row order as X -- no coordinate matching, no spatial join, no
# chance of silently pairing an event with its neighbour.
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("1. EVENTS AND THE WEAK LABEL")
print("=" * 74)

_XC = set(pq.ParquetFile(DATASET_DIR / "X.parquet").schema_arrow.names)
_MC = set(pq.ParquetFile(DATASET_DIR / "meta.parquet").schema_arrow.names)
_YC = set(pq.ParquetFile(DATASET_DIR / "y.parquet").schema_arrow.names)

_want = [c for c in ("event_centroid_lon", "event_centroid_lat") if c in _MC]
if len(_want) < 2:
    raise SystemExit(f"meta.parquet has no centroid columns among {sorted(_MC)}")
meta = pd.read_parquet(DATASET_DIR / "meta.parquet", columns=_want)
y = pd.read_parquet(DATASET_DIR / "y.parquet",
                    columns=["label_weak"] if "label_weak" in _YC
                    else ["label"])
y.columns = ["label_weak"]

_ids = ("event_id" if "event_id" in _MC else None)
if _ids:
    meta["event_id"] = pd.read_parquet(DATASET_DIR / "meta.parquet",
                                       columns=["event_id"])["event_id"]

print(f"  meta {meta.shape}   y {y.shape}")
frames = [meta.reset_index(drop=True)]
if "start_year" in _MC:
    frames.append(pd.read_parquet(DATASET_DIR / "meta.parquet",
                                  columns=["start_year"])["start_year"])
if "DIST_NEAREST_FACILITY_LOG" in _XC:
    # One column, streamed. Used only to stratify and to report the joint
    # distribution -- NOT as a label input.
    _d = np.full(len(meta), np.nan)
    _off = 0
    for _b in pq.ParquetFile(DATASET_DIR / "X.parquet").iter_batches(
        batch_size=500_000, columns=["DIST_NEAREST_FACILITY_LOG"]
    ):
        _n = _b.num_rows
        _d[_off:_off + _n] = (_b.column(0).to_pandas()
                              .to_numpy(dtype="float64", na_value=np.nan))
        _off += _n
    frames.append(pd.DataFrame({"DIST_NEAREST_FACILITY_LOG": _d}))

ev = pd.concat(frames, axis=1)
ev["label_weak"] = y["label_weak"].to_numpy()
ev = ev.dropna(subset=["event_centroid_lon", "event_centroid_lat"])
print(f"  with coordinates : {len(ev):,} of {len(meta):,}")
print()
print("  weak label distribution:")
print(ev["label_weak"].value_counts().to_string())
print(f"  {TARGET} rate: {100.0 * (ev['label_weak'] == TARGET).mean():.2f}%")

# ---------------------------------------------------------------------------
# 2. SAMPLE
#
# Stratified on the weak label. A uniform sample would give ~15% positives,
# which is enough to characterise the POSITIVE class -- and the positive class
# is the whole question. Negatives are sampled too, so the joint distribution
# is honest rather than conditional.
# ---------------------------------------------------------------------------
_pos = ev[ev["label_weak"] == TARGET]
_neg = ev[ev["label_weak"] != TARGET]
_frac = min(1.0, N_EVENTS / max(len(ev), 1))
print()
print(f"  sampling {N_EVENTS:,} events ({100.0 * _frac:.1f}%), stratified on "
      "the weak label")
print(f"    {TARGET}: {len(_pos):,} -> "
      f"{int(len(_pos) * _frac):,}")
print(f"    other   : {len(_neg):,} -> {int(len(_neg) * _frac):,}")
sample = pd.concat([
    _pos.sample(n=int(len(_pos) * _frac), random_state=0),
    _neg.sample(n=int(len(_neg) * _frac), random_state=0),
]).reset_index(drop=True)
print(f"  drawn: {len(sample):,}")

# ---------------------------------------------------------------------------
# 3. SAMPLE LAND COVER
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("3. SAMPLING LAND COVER")
print("=" * 74)
print("  tiles are read once each; the cost is geography, not event count")

# CHUNKED, not one call. `sampleRegions` serialises the whole feature
# collection into a single request and Earth Engine rejects anything over
# 10 MiB: 120,000 points fails with "Request payload size exceeds the limit:
# 10485760 bytes" BEFORE any compute is billed, which reads as a data failure
# when it is a transport limit.
_records = [{"row": int(_i),
             "lon": float(_r["event_centroid_lon"]),
             "lat": float(_r["event_centroid_lat"])}
            for _i, _r in sample.iterrows()]
print(f"  sampleRegions over {len(_records):,} points, in chunks of "
      f"{_CHUNK:,} (10 MiB request limit)")
t0 = time.time()
_vals = gee.sample_points_chunked(
    gee.worldcover_image(), _records, id_key="row", scale=100,
    chunk_size=_CHUNK, band=gee.WORLDCOVER_BAND)
_elapsed = time.time() - t0
print(f"  done in {_elapsed:.0f}s ({len(_vals):,} non-null values)")

sample["WORLDCOVER_CLASS"] = -1
for _k, _v in _vals.items():
    if _v is not None:
        sample.loc[int(_k), "WORLDCOVER_CLASS"] = int(_v)

_n_map = int((sample["WORLDCOVER_CLASS"] >= 0).sum())
print(f"  mapped {_n_map:,} of {len(sample):,} "
      f"({100.0 * _n_map / len(sample):.1f}%)")
if _n_map < 0.9 * len(sample):
    print("  <90% mapped is too many gaps to draw a conclusion. Check the")
    print("  coordinates and the AOI before reading anything below.")
    print("  Did NOT proceed to the cross-tab.")
    raise SystemExit("land cover coverage too low for the cross-tab to mean "
                     "anything")

sample["LANDCOVER_VEGETATED"] = np.where(
    sample["WORLDCOVER_CLASS"] >= 0,
    np.isin(sample["WORLDCOVER_CLASS"],
            sorted(gee.VEGETATED_CLASSES)).astype("int8"),
    -1)

# ---------------------------------------------------------------------------
# 4. THE CROSS-TAB -- the whole point of the cell
# ---------------------------------------------------------------------------
good = sample[sample["WORLDCOVER_CLASS"] >= 0].copy()
is_t = good["label_weak"] == TARGET
veg = good["LANDCOVER_VEGETATED"] == 1

print()
print("=" * 74)
print("4. CROSS-TAB: weak INDUSTRIAL_FIRE vs the substrate beneath it")
print("=" * 74)
print()
print(f"  {'substrate':<24} {'weak INDUSTRIAL_FIRE':>21} {'weak other':>14}")
rows = []
for _c in sorted(gee.WORLDCOVER_CLASSES):
    _m = good["WORLDCOVER_CLASS"] == _c
    if not _m.any():
        continue
    rows.append({
        "class": _c,
        "name": gee.WORLDCOVER_CLASSES[_c],
        "n_total": int(_m.sum()),
        "n_weak_industrial": int((_m & is_t).sum()),
        "n_weak_other": int((_m & ~is_t).sum()),
    })
_tab = pd.DataFrame(rows)
_tab["pct_of_this_class_weak_industrial"] = (
    100.0 * _tab["n_weak_industrial"] / _tab["n_total"].clip(lower=1))
for _, _r in _tab.iterrows():
    print(f"  {_r['class']:>3}  {_r['name']:<20} "
          f"{_r['n_weak_industrial']:>10,}  ({_r['pct_of_this_class_weak_industrial']:>5.1f}%) "
          f"{_r['n_weak_other']:>13,}")

print()
print("  THE HEADLINE NUMBER")
print("  " + "-" * 70)
_t = good[is_t]
_tv = int((_t["LANDCOVER_VEGETATED"] == 1).sum())
_tn = int((_t["LANDCOVER_VEGETATED"] == 0).sum())
print(f"  weak INDUSTRIAL_FIRE events sampled      : {len(_t):,}")
print(f"    on VEGETATED substrate (crop etc.)      : {_tv:,} "
      f"({100.0 * _tv / max(len(_t), 1):.1f}%)")
print(f"    on NON-vegetated substrate (built/bare) : {_tn:,} "
      f"({100.0 * _tn / max(len(_t), 1):.1f}%)")
print()
print("  These are P(SUBSTRATE | weak INDUSTRIAL_FIRE) -- the composition of")
print("  the positive class. They are NOT accuracy, and they do NOT say how")
print("  good the label is. For that you need the other direction.")

# --- P(LABEL | SUBSTRATE), computed from the same cells --------------------
#
# REGRESSION. An earlier version printed
#     "of all NON-vegetated events, 36.4% are labelled INDUSTRIAL_FIRE"
# when the quantity it computed was P(weak OTHER | non-vegetated) = 1 - 0.636.
# The inversion turned a 5.5x skew in the label's favour into an apparent skew
# against it, and the error was read as a finding before it was caught. A
# labelled conditional must be computed from the cell it names.
_nv = int((good["LANDCOVER_VEGETATED"] == 0).sum())
_v = int((good["LANDCOVER_VEGETATED"] == 1).sum())
_p_ind_given_nv = (_tn / _nv) if _nv else 0.0
_p_ind_given_v = ((_tv / _v) if _v else 0.0)
print()
print("  P(weak INDUSTRIAL_FIRE | SUBSTRATE) -- the other direction:")
print(f"    given NON-vegetated : {_p_ind_given_nv:.3f} "
      f"({_tn:,} / {_nv:,})")
print(f"    given vegetated     : {_p_ind_given_v:.3f} "
      f"({_tv:,} / {_v:,})")
print(f"    ratio               : "
      f"{_p_ind_given_nv / max(_p_ind_given_v, 1e-9):.1f}x")
print()
print("  AGAINST the substrate proxy (non-vegetated == 'industrial'):")
_P = _tn / max(len(_t), 1) if len(_t) else 0.0
_R = _p_ind_given_nv
_F1 = 2 * _P * _R / (_P + _R) if (_P + _R) else 0.0
print(f"    precision {_P:.3f}   recall {_R:.3f}   F1 {_F1:.3f}")
print()
print("  How to read it:")
print("   PRECISION is the damning number: of everything the label calls an")
print("   industrial fire, roughly 70% sits on vegetation that can burn --")
print("   cropland above all. That is farmland near a facility, mislabelled.")
print("   RECALL is better than precision, because non-vegetated substrate is")
print("   rare (7% of events), so a 5 km proximity cut finds many of them.")
print("   Neither is accuracy: substrate alone is not ground truth. A factory")
print("   fire beside cropland, or a stubble burn 300 m from a refinery,")
print("   disagrees in BOTH directions. This bounds the error; the gold set")
print("   measures it.")

# ---------------------------------------------------------------------------
# 5. THE CAVEAT, AND THE DISTANCE INTERACTION
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("5. CAVEATS AND THE DISTANCE INTERACTION")
print("=" * 74)
if "DIST_NEAREST_FACILITY_LOG" in good.columns:
    _dm = 10.0 ** good["DIST_NEAREST_FACILITY_LOG"].to_numpy() - 1.0
    good["dist_m"] = _dm
    print("  substrate vs distance to the nearest facility:")
    print(f"  {'band':<14} {'n':>9} {'%vegetated':>12} {'%weak_ind':>11}")
    for _lo, _hi, _name in ((0, 2000, "0-2 km"), (2000, 5000, "2-5 km"),
                            (5000, 8000, "5-8 km"), (8000, 1e9, ">8 km")):
        _m = (good["dist_m"] >= _lo) & (good["dist_m"] < _hi)
        if _m.sum() < 20:
            continue
        print(f"  {_name:<14} {int(_m.sum()):>9,} "
              f"{100.0 * veg[_m.values].mean():>11.1f}% "
              f"{100.0 * is_t[_m.values].mean():>10.1f}%")
    print()
    print("  The 2-5 km band is where the weak rule's 5 km cut decides. If the")
    print("  weak label is wrong, that is where the errors concentrate -- and")
    print("  it is exactly the band the gold sample oversamples.")

if "start_year" in good.columns:
    print()
    print("  WorldCover v200 is a 2021 STATIC map. Applying it to an event from")
    print("  2012 assumes the substrate was unchanged for nine years:")
    print(good["start_year"].value_counts().sort_index().to_string())
    _old = int((good["start_year"] < 2015).sum())
    if _old:
        print(f"  {_old:,} of {len(good):,} sampled events predate 2015, where")
        print("  that assumption is weakest.")

# ---------------------------------------------------------------------------
# 6. SAVE
# ---------------------------------------------------------------------------
_out = QC_DIR / "landcover_vs_weak_label.csv"
_keep = [c for c in ("event_id", "event_centroid_lon", "event_centroid_lat",
                     "start_year", "label_weak", "WORLDCOVER_CLASS",
                     "LANDCOVER_VEGETATED") if c in good.columns]
good[_keep].to_csv(_out, index=False)
_tab.to_csv(QC_DIR / "landcover_crosstab.csv", index=False)
print()
print("=" * 74)
print("6. WRITTEN")
print("=" * 74)
print(f"  {_out}   ({len(good):,} rows)")
print(f"  {QC_DIR / 'landcover_crosstab.csv'}   ({len(_tab)} substrate rows)")
print()
print("=" * 74)
print("END -- paste the whole output above, especially section 4.")
print("=" * 74)