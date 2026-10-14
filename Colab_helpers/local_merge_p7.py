# ============================================================================
# LOCAL MERGE — run on your Mac (needs base files from Drive too)
# ============================================================================
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np
from pathlib import Path

# CONFIG: Update these paths to your local copies
FEATURES_DIR = Path("/Users/aashutosh/konsors_sih2/konsors-sih2/Features_mapped")
BASE_DIR = Path("/Users/aashutosh/konsors_sih2/konsors-sih2/Features_mapped")
OUTPUT_DIR = BASE_DIR

print("=" * 70)
print("LOCAL MERGE P7 FEATURES")
print("=" * 70)

# Check base files exist
for f in ["X.parquet", "meta.parquet", "y.parquet"]:
    p = BASE_DIR / f
    if not p.exists():
        print(f"MISSING: {p}")
        print(f"\nDownload from Drive:")
        print(f"  Firm_Clusters/full_2012_2026/features/dataset/{f}")
        print(f"  Firm_Clusters/full_2012_2026/events/canonical_event_base.parquet")
        exit(1)

print("\nLoading feature blocks (local)...")
lc = pd.read_parquet(FEATURES_DIR / "landcover_events.parquet", columns=["row", "WORLDCOVER_CLASS", "LANDCOVER_VEGETATED"])
lc = lc.set_index("row")
print(f"  landcover: {lc.shape}")

nbr = pd.read_parquet(FEATURES_DIR / "nbr_events.parquet", columns=["row", "DNBR"])
nbr = nbr.set_index("row")
print(f"  NBR: {nbr.shape}")

sar = pd.read_parquet(FEATURES_DIR / "sar_events.parquet", columns=["row", "VV_CHANGE", "VH_CHANGE", "VV_VH_CHANGE"])
sar = sar.set_index("row")
print(f"  SAR: {sar.shape}")

weather_cols = [c for c in pd.read_parquet(FEATURES_DIR / "weather_events.parquet").columns if c != "row"]
weather = pd.read_parquet(FEATURES_DIR / "weather_events.parquet", columns=["row"] + weather_cols)
weather = weather.set_index("row")
weather = weather.drop(columns=["event_id", "start_year", "start_month"], errors="ignore")
print(f"  weather: {weather.shape}  cols: {list(weather.columns)}")

# RENAME THE TWO MISLABELED P7n COLUMNS, AT THE SOURCE.
#
# Measured on the downloaded file: `TEMP_C` holds KELVIN (median 299.44) and
# `RAIN_MM` holds METRES (median 0.0003). Earth Engine returns SI and the P7n
# cell never converted, while the ledger names field units. A tree model does
# not care -- it is invariant to a monotone rescale -- so Model B's ranking is
# unaffected and nothing needs retraining to "fix" it. But the NAMES are now
# lies, and P3d is about to add a `TEMP_C` that is genuinely Celsius. Two
# different quantities in one column name is how a K value gets fed to an
# equation expecting degrees.
#
# So the P7n raw columns keep their values and are given honest names, and the
# P3d daily columns take the field-unit names.
UNIT_RENAMES = {"TEMP_C": "TEMP_K", "RAIN_MM": "RAIN_M"}
_known_bad = [c for c in UNIT_RENAMES if c in weather.columns]
if len(_known_bad) == len(UNIT_RENAMES):
    _t = weather["TEMP_C"].dropna()
    _r = weather["RAIN_MM"].dropna()
    _t_is_kelvin = bool(_t.median() > 200)
    _r_is_metres = bool(_r.median() < 0.5)
    print(f"  P7n TEMP_C median {_t.median():.2f} -> "
          f"{'KELVIN, renamed TEMP_K' if _t_is_kelvin else 'looks like C, left alone'}")
    print(f"  P7n RAIN_MM median {_r.median():.5f} -> "
          f"{'METRES, renamed RAIN_M' if _r_is_metres else 'looks like mm, left alone'}")
    if _t_is_kelvin:
        weather = weather.rename(columns={"TEMP_C": "TEMP_K"})
    if _r_is_metres:
        weather = weather.rename(columns={"RAIN_MM": "RAIN_M"})
else:
    print(f"  P7n unit rename skipped (columns present: {_known_bad})")
print(f"  after rename: {list(weather.columns)}")

atmos_cols = [c for c in pd.read_parquet(FEATURES_DIR / "atmos_events.parquet").columns if c != "row"]
atmos = pd.read_parquet(FEATURES_DIR / "atmos_events.parquet", columns=["row"] + atmos_cols)
atmos = atmos.set_index("row")
atmos = atmos.drop(columns=["event_id", "start_year", "start_month"], errors="ignore")
print(f"  atmos: {atmos.shape}  cols: {list(atmos.columns)}")

# Stream merge X.parquet
X_path = BASE_DIR / "X.parquet"
X_new_path = OUTPUT_DIR / "X_with_p7.parquet"

X_pf = pq.ParquetFile(X_path)
print(f"\nX.parquet: {X_pf.metadata.num_rows:,} rows, {X_pf.num_row_groups} row groups")

writer = None
total_rows = 0
row_offset = 0

for rg_idx in range(X_pf.num_row_groups):
    print(f"\n  Row group {rg_idx+1}/{X_pf.num_row_groups}...")
    rg = X_pf.read_row_group(rg_idx).to_pandas()
    
    if "row" not in rg.columns:
        rg = rg.reset_index().rename(columns={"index": "row"})
    
    # Convert local row index to global by adding offset
    rg["row"] = rg["row"] + row_offset
    row_offset += len(rg)
    
    rg = rg.set_index("row")
    
    merged = rg.join(lc, how="left").join(nbr, how="left").join(sar, how="left").join(weather, how="left").join(atmos, how="left")
    merged = merged.reset_index()
    
    for c in merged.columns:
        if c != "row" and merged[c].dtype.kind in ("f", "i"):
            merged[c] = merged[c].astype("float32")
    
    table = pa.Table.from_pandas(merged, preserve_index=False)
    
    if writer is None:
        writer = pq.ParquetWriter(X_new_path, table.schema, compression="snappy")
    writer.write_table(table)
    
    total_rows += len(merged)
    print(f"    wrote {len(merged):,} rows (total {total_rows:,})")

writer.close()
print(f"\nDONE: {X_new_path} — {total_rows:,} rows, {len(merged.columns)} columns")