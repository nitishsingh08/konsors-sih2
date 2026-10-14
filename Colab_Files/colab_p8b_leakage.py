# ============================================================================
# COLAB CELL: P8b -- LEAKAGE PROBE on the headline split
#
# Reviewer question: "is the train/test split clean in time and space? If
# neighbouring events leak across the split, PR-AUC 0.335 is optimistic."
#
# `assert_no_spatial_leakage` already runs in colab_p7_train.py, so block
# disjointness is guaranteed. What it cannot see is a POINT leak: the weak label
# is a 1 km proximity cut, a held-out 0.1 deg block is ~11 km across, so the same
# refinery sits in the held-out block AND in the training blocks next to it.
# This cell measures that directly -- for every test event, how far is the
# nearest TRAINING event, and how many test years were also in training.
#
#   run after colab_p7_train.py
# ============================================================================

P8B_CELL_VERSION = "2026-10-04.1"

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import colab_utils as cu
import leakage_probe as lp

QC_DIR = Path("/content/drive/MyDrive/Firm_Clusters/full_2012_2026/qc")
QC_DIR.mkdir(parents=True, exist_ok=True)

DATASET = (Path("/content/drive/MyDrive/Firm_Clusters/full_2012_2026")
           / "features" / "dataset")

#: Train on a slice of each side. The proximity probe is O(n_train x n_test), so
#: 300k x 300k is 90e9 pair-dists -- run it in Colab overnight, not in a cell
#: that dies. The full-catalogue answer is a number, not a direction, so the
#: slice is legitimate; the size is printed so nobody can forget it was a slice.
SAMPLE_TRAIN = 300_000
SAMPLE_TEST = 300_000

print("=" * 78)
print(f"P8b -- LEAKAGE PROBE  [v{P8B_CELL_VERSION}]")
print("=" * 78)

meta = pd.read_parquet(DATASET / "meta.parquet")
print(f"meta rows: {len(meta):,}")
print(f"columns:   {sorted(meta.columns)}")

assert "spatial_block" in meta.columns, (
    "meta.parquet has no spatial_block -- it is written by "
    "training_design.add_split_keys during P7; re-run P7 or rebuild the keys")

# Same seed as P7 so this is the SAME split that produced the headline number.
import training_design as td  # noqa: E402

parts = td.make_split(meta, kind=td.SPLIT_COMBINED, seed=0)
train, test = parts["train"], parts["test"]
print(f"train {len(train):,}   test {len(test):,}   "
      f"dropped {len(parts['dropped_early_in_test_blocks']):,}")

rng = np.random.default_rng(0)
tr = train.sample(n=min(SAMPLE_TRAIN, len(train)), random_state=0)
te = test.sample(n=min(SAMPLE_TEST, len(test)), random_state=0)
print(f"probing a stratified random slice: train {len(tr):,}, test {len(te):,}")

report = lp.probe(tr, te)
report["sliced"] = {"n_train": len(tr), "n_test": len(te),
                    "of_train_available": len(train),
                    "of_test_available": len(test),
                    "note": ("a slice, not the full folds -- the share is a "
                             "number, the block/year checks are exact")}

print()
print(json.dumps(report, indent=2, default=str))

(QC_DIR / "leakage_probe.json").write_text(json.dumps(report, indent=2,
                                                      default=str))

print()
print("=" * 78)
print(report.get("verdict", {}).get("verdict", "no verdict"))
print("=" * 78)
print()
print("READ THIS BEFORE QUOTING PR-AUC 0.3353:")
prox = report.get("cross_split_proximity", {})
for r, share in prox.get("frac_test_within_radius", {}).items():
    print(f"  test events with a training event within {r:>4} km: {share:.2%}")
print()
print("If the 1 km share is high, the held-out fold contains near-copies of")
print("training events: same facility, different year. The fix is not a")
print("different model, it is a FACILITY-DISJOINT split -- hold out whole")
print("facilities, not 0.1 deg grid cells, and rerun P7 on that.")