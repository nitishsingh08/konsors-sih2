# ============================================================================
# CELL 1 (FIXED): Merge all feature blocks into X_with_p7.parquet — STREAMING
# ============================================================================
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/content/drive/MyDrive/Firm_Clusters")
DATASET_DIR = ROOT / "full_2012_2026" / "features" / "dataset"
QC_DIR = ROOT / "qc"

print("=" * 70)
print("MERGE P7 FEATURES (streaming, low memory)")
print("=" * 70)

# -----------------------------------------------------------------------------
# Load all feature blocks into memory (they're small: landcover 75MB, nbr 188MB, etc)
# -----------------------------------------------------------------------------
print("\nLoading feature blocks...")
lc = pd.read_parquet(QC_DIR / "landcover_events.parquet", columns=["row", "WORLDCOVER_CLASS", "LANDCOVER_VEGETATED"])
lc = lc.set_index("row")
print(f"  landcover: {lc.shape}")

nbr = pd.read_parquet(QC_DIR / "nbr_events.parquet", columns=["row", "DNBR"])
nbr = nbr.set_index("row")
print(f"  NBR: {nbr.shape}")

sar = pd.read_parquet(QC_DIR / "sar_events.parquet", columns=["row", "VV_CHANGE", "VH_CHANGE", "VV_VH_CHANGE"])
sar = sar.set_index("row")
print(f"  SAR: {sar.shape}")

weather_cols = [c for c in pd.read_parquet(QC_DIR / "weather_events.parquet", nthreads=1).columns if c != "row"]
weather = pd.read_parquet(QC_DIR / "weather_events.parquet", columns=weather_cols)
weather = weather.set_index("row")
print(f"  weather: {weather.shape}  cols: {list(weather.columns)}")

atmos_cols = [c for c in pd.read_parquet(QC_DIR / "atmos_events.parquet", nthreads=1).columns if c != "row"]
atmos = pd.read_parquet(QC_DIR / "atmos_events.parquet", columns=atmos_cols)
atmos = atmos.set_index("row")
print(f"  atmos: {atmos.shape}  cols: {list(atmos.columns)}")

# -----------------------------------------------------------------------------
# Stream X.parquet row groups, merge, write output incrementally
# -----------------------------------------------------------------------------
X_path = DATASET_DIR / "X.parquet"
X_new_path = DATASET_DIR / "X_with_p7.parquet"

X_pf = pq.ParquetFile(X_path)
print(f"\nX.parquet: {X_pf.metadata.num_rows:,} rows, {X_pf.num_row_groups} row groups")

# Build output schema
first_rg = X_pf.read_row_group(0).to_pandas()
if "row" not in first_rg.columns:
    first_rg = first_rg.reset_index().rename(columns={"index": "row"})
first_rg = first_rg.set_index("row")

# Merge first row group to get output columns
merged_sample = first_rg.join(lc, how="left").join(nbr, how="left").join(sar, how="left").join(weather, how="left").join(atmos, how="left")
output_cols = merged_sample.columns.tolist()
print(f"Output columns: {len(output_cols)}")

# Write incrementally
writer = None
total_rows = 0

for rg_idx in range(X_pf.num_row_groups):
    print(f"\n  Row group {rg_idx+1}/{X_pf.num_row_groups}...")
    rg = X_pf.read_row_group(rg_idx).to_pandas()
    
    if "row" not in rg.columns:
        rg = rg.reset_index().rename(columns={"index": "row"})
    rg = rg.set_index("row")
    
    # Merge all feature blocks
    merged = rg.join(lc, how="left").join(nbr, how="left").join(sar, how="left").join(weather, how="left").join(atmos, how="left")
    merged = merged.reset_index()
    
    # Convert to float32 for feature columns (keep row as int64)
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
print(f"\nDONE: {X_new_path} — {total_rows:,} rows, {len(output_cols)} columns")

# Verify
print("\nVerifying...")
v = pd.read_parquet(X_new_path, nthreads=1)
print(f"  shape: {v.shape}")
print(f"  columns: {list(v.columns)}")
print(f"  dtypes: {v.dtypes.value_counts().to_dict()}")