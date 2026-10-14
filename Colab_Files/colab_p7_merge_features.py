# ============================================================================
# P7 RETRAIN + SHAP — complete pipeline
# Run each cell sequentially in Colab
# ============================================================================

# -----------------------------------------------------------------------------
# CELL 1: Merge all feature blocks into X_new.parquet
# -----------------------------------------------------------------------------
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/content/drive/MyDrive/Firm_Clusters")
DATASET_DIR = ROOT / "full_2012_2026" / "features" / "dataset"
QC_DIR = ROOT / "qc"

print("Loading base X...")
X = pd.read_parquet(DATASET_DIR / "X.parquet")
# Ensure 'row' column exists (0-based index)
if "row" not in X.columns:
    X = X.reset_index().rename(columns={"index": "row"})
print(f"  base X: {X.shape}")

print("Loading meta...")
meta = pd.read_parquet(DATASET_DIR / "meta.parquet", columns=["event_id", "start_year"])
# Ensure 'row' column exists
if "row" not in meta.columns:
    meta = meta.reset_index().rename(columns={"index": "row"})
print(f"  meta: {meta.shape}")

# -----------------------------------------------------------------------------
# Merge landcover
# -----------------------------------------------------------------------------
print("\nMerging landcover...")
lc = pd.read_parquet(QC_DIR / "landcover_events.parquet", columns=["row", "WORLDCOVER_CLASS", "LANDCOVER_VEGETATED"])
print(f"  landcover: {lc.shape}")
X = X.merge(lc, on="row", how="left", validate="1:1")
print(f"  after merge: {X.shape}")

# -----------------------------------------------------------------------------
# Merge NBR (DNBR = NBR_PRE - NBR_POST, positive = vegetation loss)
# -----------------------------------------------------------------------------
print("\nMerging NBR...")
nbr = pd.read_parquet(QC_DIR / "nbr_events.parquet", columns=["row", "DNBR"])
print(f"  NBR: {nbr.shape}")
X = X.merge(nbr, on="row", how="left", validate="1:1")
print(f"  after merge: {X.shape}")

# -----------------------------------------------------------------------------
# Merge SAR (VV_CHANGE, VH_CHANGE, VV_VH_CHANGE)
# -----------------------------------------------------------------------------
print("\nMerging SAR...")
sar = pd.read_parquet(QC_DIR / "sar_events.parquet", columns=["row", "VV_CHANGE", "VH_CHANGE", "VV_VH_CHANGE"])
print(f"  SAR: {sar.shape}")
X = X.merge(sar, on="row", how="left", validate="1:1")
print(f"  after merge: {X.shape}")

# -----------------------------------------------------------------------------
# Merge Weather (18 features)
# -----------------------------------------------------------------------------
print("\nMerging weather...")
weather_cols = ["row", "TEMPERATURE_ANOMALY_Z", "RH_ANOMALY_Z", "VPD_ANOMALY_Z",
                "WIND_SPEED", "WIND_GUST", "WIND_DIRECTION_SIN", "WIND_DIRECTION_COS",
                "RAIN_24H_LOG", "RAIN_7D_LOG", "SOIL_MOISTURE_ANOMALY_Z",
                "FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "FWI_ANOMALY_Z"]
weather = pd.read_parquet(QC_DIR / "weather_events.parquet", columns=[c for c in weather_cols if c != "row"])
# Check what's actually available
avail = [c for c in weather_cols if c in pd.read_parquet(QC_DIR / "weather_events.parquet", nthreads=1).columns]
print(f"  available weather cols: {avail}")
weather = pd.read_parquet(QC_DIR / "weather_events.parquet", columns=avail)
X = X.merge(weather, on="row", how="left", validate="1:1")
print(f"  after merge: {X.shape}")

# -----------------------------------------------------------------------------
# Merge Atmospheric (9 features)
# -----------------------------------------------------------------------------
print("\nMerging atmospheric...")
atmos_cols = ["row", "NO2_ANOMALY_Z", "SO2_ANOMALY_Z", "CO_ANOMALY_Z",
              "CH4_ANOMALY_Z", "HCHO_ANOMALY_Z", "AOD_ANOMALY_Z",
              "CO_NO2_RATIO_LOG", "SO2_NO2_RATIO_LOG", "CH4_CO_RATIO_LOG"]
avail = [c for c in atmos_cols if c in pd.read_parquet(QC_DIR / "atmos_events.parquet", nthreads=1).columns]
print(f"  available atmos cols: {avail}")
atmos = pd.read_parquet(QC_DIR / "atmos_events.parquet", columns=avail)
X = X.merge(atmos, on="row", how="left", validate="1:1")
print(f"  after merge: {X.shape}")

# -----------------------------------------------------------------------------
# Save new X
# -----------------------------------------------------------------------------
X_new_path = DATASET_DIR / "X_with_p7.parquet"
print(f"\nSaving to {X_new_path}...")
X.to_parquet(X_new_path, index=False)
print(f"  DONE: {X.shape[1]} features, {X.shape[0]:,} rows")

# Verify feature list
print("\nFeature groups in new X:")
import feature_ledger as fl
for grp in fl.FEATURE_GROUPS:
    present = [f for f in fl.FEATURE_GROUPS[grp] if f in X.columns]
    print(f"  {grp}: {len(present)}/{len(fl.FEATURE_GROUPS[grp])}  {present[:5]}{'...' if len(present)>5 else ''}")