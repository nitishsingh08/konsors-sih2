# ============================================================================
# DEPRECATED 2026-10-07 -- colab_p7k_nbr.py is RETIRED.
# Run colab_p7k_nbr_change.py instead.
#
# WHY THIS FILE STILL EXISTS
#
# This was the first P7k draft, and it failed in Colab before it ever
# reached Earth Engine (SystemExit: Missing columns
# ['event_centroid_lon', 'event_centroid_lat']). Its defects, all fixed in
# colab_p7k_nbr_change.py:
#
#   1. It loaded qc/landcover_events.parquet and demanded coordinate columns
#      that P7h never writes (P7h stores only row/event_id/class), so it
#      crashed after the landcover load. P7k change reads meta.parquet
#      directly and needs no P7h output at all.
#   2. It sampled BOTH NBR_PRE and NBR_POST from the event's OWN month
#      composite, so NBR_CHANGE was identically 0 for every event -- a
#      feature that carries no information, produced after an 11-hour run.
#   3. It issued one Earth Engine request PER EVENT (~7.27M round trips),
#      with no chunking, no sharding, no resume and no manifest. A Colab
#      kill would have lost everything.
#
# Keeping the file as a LOUD guard rather than deleting it: a stale copy of
# the draft still sits in Drive/pipeline from earlier uploads, and deleting
# the repo file would not remove that copy. This stub, once uploaded,
# replaces it and refuses to run.
# ============================================================================

P7K_NBR_DEPRECATED_VERSION = "2026-10-07.1"

print("=" * 74)
print(f"colab_p7k_nbr.py DEPRECATED  [v{P7K_NBR_DEPRECATED_VERSION}]")
print("=" * 74)
print("  This is the retired P7k draft. Its per-event EE calls, own-month")
print("  composites and missing-column crash are all fixed in:")
print()
print("      colab_p7k_nbr_change.py    <-- run this instead")
print()
print("  Re-upload the repo files and run:")
print("      import colab_runcell")
print("      colab_runcell.run('colab_p7k_nbr_change')")
raise SystemExit("colab_p7k_nbr.py is retired -- run colab_p7k_nbr_change")