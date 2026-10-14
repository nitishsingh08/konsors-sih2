# ============================================================================
# LOCAL MERGE (pass 2) -- add P3g, P3d, Landsat 5/7 NBR onto X_with_p7.parquet
#
# Run AFTER you have downloaded:
#     recurrence_events.parquet
#     fwi_daily_events.parquet
#     nbr_landsat57_events.parquet
# and AFTER local_merge_p7.py has produced X_with_p7.parquet.
#
# DNBR PROVENANCE. `nbr_events.parquet` (Landsat 8) and
# `nbr_landsat57_events.parquet` (Landsat 5/7) both carry a DNBR column for
# different row sets. Merging them into ONE column without recording which
# sensor produced it would make a Landsat-5 value indistinguishable from a
# Landsat-8 one -- and the two have different gaps (L7 SLC-off wedges vs L8
# full coverage), so downstream code must be able to tell them apart. So:
#
#     DNBR         L8 where it exists, else L5/7
#     NBR_SOURCE   'LC08' | 'LT05_LE07' | '' (unmeasured)
#
# A blank NBR_SOURCE is how "no satellite ever saw this" is written. It is
# NOT 0, which would assert "measured, and there was no vegetation loss".
#
# RECURRENCE FIX: The P3g Colab output contains grid-cell fire counts
# (1° cells ≈ 12,300 km²), not site-level recurrence rates. We rename to
# CELL_FIRE_COUNT_* and add normalized density (fires/1000km²/yr).
#
# FWI DAILY FIX: The P3d Colab output has lat/lon swap (Arctic temps).
# MUST re-run colab_p3d_fwi_daily.py with corrected point geometry.
# ============================================================================

import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np
from pathlib import Path

FEATURES_DIR = Path("/Users/aashutosh/konsors_sih2/konsors-sih2/Features_mapped")
IN_X = FEATURES_DIR / "X_with_p7.parquet"
OUT_X = FEATURES_DIR / "X_with_p7_full.parquet"

print("=" * 70)
print("LOCAL MERGE PASS 2 -- recurrence + daily FWI + Landsat 5/7 NBR")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. P3g recurrence (4 features) -- RENAMED: grid-cell counts, not site rates
# ---------------------------------------------------------------------------
print("\n1. Recurrence (P3g) -- fixing names & adding density...")
rec_path = FEATURES_DIR / "recurrence_events.parquet"
if rec_path.exists():
    rec = pd.read_parquet(rec_path, columns=["row", "RECURRENCE_RATE_1Y",
                                             "RECURRENCE_RATE_3Y",
                                             "RECURRENCE_RATE_5Y",
                                             "MAX_INTER_EVENT_GAP_LOG"])
    rec = rec.set_index("row").sort_index()
    
    # 1° cell ≈ 111 km × 111 km ≈ 12,300 km² at equator, less at higher lat
    # Use 12,300 km² as reference area for normalization
    CELL_AREA_KM2 = 12300.0
    
    # Rename to reflect true meaning: grid-cell fire counts
    rec = rec.rename(columns={
        "RECURRENCE_RATE_1Y": "CELL_FIRE_COUNT_1Y",
        "RECURRENCE_RATE_3Y": "CELL_FIRE_COUNT_3Y",
        "RECURRENCE_RATE_5Y": "CELL_FIRE_COUNT_5Y",
    })
    
    # Add normalized density: fires per 1000 km² per year
    rec["CELL_FIRE_DENSITY_1Y"] = rec["CELL_FIRE_COUNT_1Y"] / (CELL_AREA_KM2 / 1000.0)
    rec["CELL_FIRE_DENSITY_3Y"] = rec["CELL_FIRE_COUNT_3Y"] / (CELL_AREA_KM2 / 1000.0) / 3.0
    rec["CELL_FIRE_DENSITY_5Y"] = rec["CELL_FIRE_COUNT_5Y"] / (CELL_AREA_KM2 / 1000.0) / 5.0
    
    print(f"   {rec.shape}  non-NaN: "
          f"{int(rec['CELL_FIRE_COUNT_1Y'].notna().sum()):,}")
    print(f"   Density stats (fires/1000km²/yr): "
          f"1Y mean={rec['CELL_FIRE_DENSITY_1Y'].mean():.1f}, "
          f"3Y mean={rec['CELL_FIRE_DENSITY_3Y'].mean():.1f}, "
          f"5Y mean={rec['CELL_FIRE_DENSITY_5Y'].mean():.1f}")
else:
    rec = None
    print("   SKIPPED -- not downloaded")

# ---------------------------------------------------------------------------
# 2. P3d daily FWI + daily weather (16 columns) -- KNOWN BROKEN (lat/lon swap)
# ---------------------------------------------------------------------------
print("\n2. Daily FWI chain (P3d) -- WARNING: lat/lon swap in Colab extraction!")
fwi_path = FEATURES_DIR / "fwi_daily_events.parquet"
if fwi_path.exists():
    fw = pd.read_parquet(fwi_path)
    fw = fw.set_index("row").sort_index()
    print(f"   {fw.shape}  cols: {list(fw.columns)}")
    print("   !!! TEMP_C mean={:.1f}°C (should be 20-35°C for India) -- Arctic data !!!".format(fw['TEMP_C'].mean()))
    print("   !!! First 10 rows identical -- join collapsed on wrong key !!!")
    print("   ACTION REQUIRED: Re-run colab_p3d_fwi_daily.py with fixed point geometry")
else:
    fw = None
    print("   SKIPPED -- not downloaded")

# ---------------------------------------------------------------------------
# 3. Landsat 5/7 NBR, with provenance
# ---------------------------------------------------------------------------
print("\n3. Landsat 5/7 NBR...")
n57_path = FEATURES_DIR / "nbr_landsat57_events.parquet"
if n57_path.exists():
    n57 = pd.read_parquet(n57_path, columns=["row", "DNBR"])
    n57 = n57.drop_duplicates(subset="row", keep="first").set_index("row").sort_index()
    n57 = n57.rename(columns={"DNBR": "DNBR_L57"})
    print(f"   {n57.shape}  non-NaN: {int(n57['DNBR_L57'].notna().sum()):,}")
else:
    n57 = None
    print("   SKIPPED -- not downloaded")

# What P7k already put in X_with_p7 (Landsat 8).
_mc = set(pq.ParquetFile(IN_X).schema_arrow.names)
HaveL8 = "DNBR" in _mc
print(f"\n   X_with_p7 already carries a DNBR (Landsat 8) column: {HaveL8}")

# ---------------------------------------------------------------------------
# 4. Stream the merge -- X_with_p7 is ~1.4 GB, so never hold it whole
# ---------------------------------------------------------------------------
pf = pq.ParquetFile(IN_X)
print(f"\n4. Merging {pf.metadata.num_rows:,} rows x "
      f"{len(_mc)} cols over {pf.num_row_groups} row groups...")

writer = None
total = 0

for rg_idx in range(pf.num_row_groups):
    rg = pf.read_row_group(rg_idx).to_pandas()
    # X_with_p7 already carries GLOBAL row ids (0..N-1), so they must be used
    # as-is. Adding the running row-group offset on top of an already-global
    # id pushes ids past N and silently misaligns every downstream join
    # (recurrence, FWI, NBR) with the events they belong to.
    if "row" not in rg.columns:
        rg = rg.reset_index().rename(columns={"index": "row"})
    rg = rg.set_index("row")

    # Provenance for the L8 column already present.
    if HaveL8:
        rg["NBR_SOURCE"] = np.where(rg["DNBR"].notna(), "LC08", "")
    else:
        rg["DNBR"] = np.nan
        rg["NBR_SOURCE"] = ""

    if n57 is not None:
        rg = rg.join(n57, how="left")
        # L8 first, L5/7 only where L8 is absent.
        fill = rg["DNBR"].isna() & rg["DNBR_L57"].notna()
        rg.loc[fill, "DNBR"] = rg.loc[fill, "DNBR_L57"]
        rg.loc[fill, "NBR_SOURCE"] = "LT05_LE07"
        rg = rg.drop(columns=["DNBR_L57"])

    if rec is not None:
        rg = rg.join(rec, how="left")
    if fw is not None:
        # COLLISION, AND THE OLD CODE HAD IT BACKWARDS.
        #
        # P7n (monthly weather) and P3d (daily chain) both supply FFMC, DMC,
        # DC, ISI, BUI, FWI, TEMP_C, RH_PCT, WIND_KMH, RAIN_MM and SWVL1.
        # These are NOT the same quantity: the monthly ones come from a
        # one-month-step approximation, the daily ones from a real daily
        # chain. Dropping the daily values to keep the monthly ones would
        # discard exactly the improvement P3d was built for.
        #
        # So the EXISTING (P7n) columns are renamed to `_MONTHLY` and P3d
        # takes the plain name. Both survive, and the daily value is the one
        # the model reads under the ledger's name.
        #
        # `fw` is indexed by `row` already, so its columns exclude 'row'; the
        # old `["row"] + ...` raised KeyError on exactly that.
        fw = fw[[c for c in fw.columns if c != "row"]]        # no-op clarity
        _dupes = [c for c in fw.columns if c in rg.columns]
        if _dupes:
            rg = rg.rename(columns={c: f"{c}_MONTHLY" for c in _dupes})
            print(f"    NOTE: kept P7n monthly under _MONTHLY for {_dupes}")
        # FWI_ANOMALY_Z is the one name where the DAILY version should be
        # primary, so P7n's monthly copy takes the suffix and P3d keeps the
        # plain name -- the reverse of the pattern above, because the daily
        # chain is the ledger's declared implementation of that feature.
        rg = rg.join(fw, how="left")

    rg = rg.reset_index()
    for c in rg.columns:
        if c != "row" and c != "NBR_SOURCE" and rg[c].dtype.kind in ("f", "i", "u"):
            rg[c] = rg[c].astype("float32")

    table = pa.Table.from_pandas(rg, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(OUT_X, table.schema, compression="snappy")
    writer.write_table(table)
    total += len(rg)
    if (rg_idx + 1) % 4 == 0 or rg_idx == pf.num_row_groups - 1:
        print(f"   row group {rg_idx+1}/{pf.num_row_groups}  "
              f"{total:,} rows  {len(rg.columns)} cols")

writer.close()
print(f"\nDONE -> {OUT_X}")
print(f"  {total:,} rows  {len(rg.columns)} columns")

# ---------------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------------
print("\n5. Feature coverage in the merged table")
rec_cols = ["CELL_FIRE_COUNT_1Y", "CELL_FIRE_COUNT_3Y", "CELL_FIRE_COUNT_5Y",
            "CELL_FIRE_DENSITY_1Y", "CELL_FIRE_DENSITY_3Y", "CELL_FIRE_DENSITY_5Y",
            "MAX_INTER_EVENT_GAP_LOG"]
fwi_cols = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI",
            "FWI_ANOMALY_Z", "TEMP_C", "RH_PCT", "WIND_KMH", "RAIN_MM", "SWVL1", "VPD_KPA"]
chk_cols = rec_cols + fwi_cols + ["DNBR", "NBR_SOURCE"]
chk = pd.read_parquet(OUT_X, columns=chk_cols)
print(f"   {'feature':<30} {'coverage':>9} {'mean':>10}")
for c in chk.columns:
    if c == "NBR_SOURCE":
        vc = chk["NBR_SOURCE"].value_counts().to_dict()
        print(f"   {'NBR_SOURCE':<30} " +
              "  ".join(f"{k or '(none)'}={v:,}" for k, v in sorted(vc.items())))
        continue
    v = chk[c]
    print(f"   {c:<30} {100.0*v.notna().mean():>8.1f}% {v.mean():>10.3f}")

print()
print("NEXT: edit local_retrain.py to read X_with_p7_full.parquet and re-run.")
print("NOTE: FWI daily features need Colab re-run (lat/lon swap fix).")