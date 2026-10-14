# ============================================================================
# COLAB DIAGNOSTIC — run this first, paste output back
# ============================================================================

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import os
from pathlib import Path

ROOT = Path("/content/drive/MyDrive/Firm_Clusters")
print("=" * 60)
print("DRIVE ROOT EXISTS:", ROOT.exists())
print("=" * 60)

# 1. Full tree survey
print("\n1. TOP-LEVEL:")
for p in sorted(ROOT.iterdir()):
    size = p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    print(f"  {p.name:30s}  {size/1e6:>8.1f} MB  {'DIR' if p.is_dir() else 'FILE'}")

# 2. Check key artefacts
print("\n2. KEY ARTEFACTS:")
artefacts = [
    "full_2012_2026/events/canonical_event_base.parquet",
    "full_2012_2026/events/event_local_cluster_membership.parquet",
    "full_2012_2026/events/event_geometry/",
    "full_2012_2026/features/dataset/X.parquet",
    "full_2012_2026/features/dataset/y.parquet",
    "full_2012_2026/features/dataset/meta.parquet",
    "full_2012_2026/external/",
    "qc/landcover_events.parquet",
    "qc/landcover_shards/",
    "qc/nbr_events.parquet",
    "qc/sar_events.parquet",
    "qc/weather_events.parquet",
    "qc/atmos_events.parquet",
    "qc/models/model_A.pkl",
    "qc/models/model_B.pkl",
    "qc/heldout_pred_A.csv",
    "qc/heldout_pred_B.csv",
]
for a in artefacts:
    p = ROOT / a
    exists = p.exists()
    if exists:
        if p.is_dir():
            files = list(p.glob("*"))
            print(f"  ✓ {a}  ({len(files)} items)")
        else:
            print(f"  ✓ {a}  ({p.stat().st_size/1e6:.1f} MB)")
    else:
        print(f"  ✗ {a}")

# 3. Check landcover specifically
print("\n3. LANDCOVER DETAIL:")
lc_shards = ROOT / "qc/landcover_shards"
if lc_shards.exists():
    shards = sorted(lc_shards.glob("lc_*.parquet"))
    print(f"  shards: {len(shards)}")
    if shards:
        import pyarrow.parquet as pq
        total_rows = sum(pq.ParquetFile(s).metadata.num_rows for s in shards)
        print(f"  total rows: {total_rows:,}")
        # sample first shard
        df = pd.read_parquet(shards[0])
        print(f"  columns: {list(df.columns)}")
        print(f"  sample:\n{df.head()}")
else:
    print("  no landcover_shards dir")

# 4. Check feature ledger version
print("\n4. PIPELINE MODULES IN /content:")
for mod in ["feature_ledger", "industrial_features", "lineage_features", 
            "cluster_tier", "training_design", "gee_utils", "explainability"]:
    p = Path(f"/content/{mod}.py")
    if p.exists():
        print(f"  ✓ {mod}.py  ({p.stat().st_size/1e3:.1f} KB)")
    else:
        print(f"  ✗ {mod}.py")

# 5. Quick meta.parquet schema
print("\n5. META.PARQUET SCHEMA:")
try:
    import pyarrow.parquet as pq
    meta_path = ROOT / "full_2012_2026/features/dataset/meta.parquet"
    if meta_path.exists():
        pf = pq.ParquetFile(meta_path)
        print(f"  columns: {pf.schema_arrow.names}")
        print(f"  rows: {pf.metadata.num_rows:,}")
    else:
        print("  NOT FOUND")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "=" * 60)
print("PASTE THIS OUTPUT BACK")
print("=" * 60)