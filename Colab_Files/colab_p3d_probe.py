# ============================================================================
# QUICK PROBE -- run BEFORE colab_p3d_fwi_daily.py, paste output here.
#
# Answers in seconds, not hours:
#   1. Which ERA5-Land collection actually carries data for a known month.
#   2. What its bands are really called.
#   3. How many days of the chain window it can reach.
#
# This exists because the P3d cell's first run wrote a month of 100%-NaN shards
# and it only showed up in a coverage print. An asset id that exists but is
# empty looks exactly like weather until you count.
# ============================================================================

import ee

CANDIDATES = [
    "ECMWF/ERA5_LAND/DAILY_AGGR",
    "ECMWF/ERA5_LAND/HOURLY",
]

# A month that is certainly inside every archive.
PROBE = ("2020-01-01", "2020-02-01")
# And the most recent month, to see how far the archive actually reaches.
TAIL = ("2025-12-25", "2026-01-10")

print("=" * 74)
print("ERA5-LAND COLLECTION PROBE")
print("=" * 74)

for cid in CANDIDATES:
    print()
    print("-" * 74)
    print(cid)
    print("-" * 74)
    try:
        col = ee.ImageCollection(cid)
        for label, win in (("probe  2020-01", PROBE), ("tail   2025-12", TAIL)):
            n = col.filterDate(*win).size().getInfo()
            days = col.filterDate(*win).distinct("system:time_start").size().getInfo()
            print(f"  {label}   images {int(n):>7,}   distinct days {int(days):>4,}")
    except Exception as exc:                                    # noqa: BLE001
        print(f"  ERROR {type(exc).__name__}: {str(exc)[:120]}")
        continue
    try:
        first = ee.ImageCollection(cid).filterDate(*PROBE).first()
        names = first.bandNames().getInfo()
        print(f"  bands ({len(names)}): {sorted(names)}")
    except Exception as exc:                                    # noqa: BLE001
        print(f"  bandNames FAILED: {type(exc).__name__}: {str(exc)[:100]}")

# ---------------------------------------------------------------------------
# Full chain window: 2012-01-01 .. 2026-08-31
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("CHAIN WINDOW REACH (2012-01-01 .. 2026-08-31)")
print("=" * 74)
for cid in CANDIDATES:
    try:
        col = ee.ImageCollection(cid).filterDate("2012-01-01", "2026-08-31")
        n = col.size().getInfo()
        days = col.distinct("system:time_start").size().getInfo()
        want_days = 5360
        print(f"  {cid:<32} images {int(n):>9,}  days {int(days):>5,} "
              f"of {want_days:,}  ({100.0*int(days)/want_days:.0f}%)")
    except Exception as exc:                                    # noqa: BLE001
        print(f"  {cid:<32} ERROR {type(exc).__name__}")

print()
print("=" * 74)
print("README")
print("=" * 74)
print("""
Pick the collection that has:
  * bands named temperature_2m / dewpoint_temperature_2m / etc, AND
  * a high fraction of the 5,360 chain days.

If DAILY_AGGR has the bands but only a fraction of the days, the chain will
still run -- it carries the previous state through the empty days -- but FWI
goes stale in that span, which the cell now counts and reports.

Paste all of this back and I will confirm the cell picks the right one.
""")