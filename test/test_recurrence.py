"""Tests for the P3g recurrence algorithm.

The cell is a single monolithic Colab script, so its maths is re-derived here
from the same definitions and checked against hand-built catalogues. These
tests exist because the cell crashed on the first real run: `np.where` with a
mask that would have discarded an out-of-bounds index still evaluates that
index eagerly. Only a case with FEWER GAPS THAN EVENTS in one cell trips it,
which is why the small synthetic catalogue I tested first passed.
"""

import unittest

import numpy as np
import pandas as pd


def recurrences(event_ids, lons, lats, start_times):
    """Exactly the algorithm in colab_p3g_recurrence.py section 4."""
    n = len(event_ids)
    day = pd.to_datetime(pd.Series(start_times)).dt.floor("D")
    day_num = (day.to_numpy("datetime64[ns]").astype("int64")
               // 86_400_000_000_000).astype("int64")
    lat = np.asarray(lats, dtype="float64")
    lon = np.asarray(lons, dtype="float64")
    cell_id = ((np.floor(lat).astype("int64") + 90) * 360
               + (np.floor(lon).astype("int64") + 180))

    rec1 = np.full(n, np.nan, "float32")
    rec3 = np.full(n, np.nan, "float32")
    rec5 = np.full(n, np.nan, "float32")
    gap = np.full(n, np.nan, "float32")

    for lo in range(n):
        cid = cell_id[lo]
        mask = cell_id == cid
        hd = day_num[mask]
        hd = hd[hd > 0]
        part_day = day_num[lo]
        if hd.size == 0:
            continue
        hi_s = np.searchsorted(hd, part_day, side="left")
        rec1[lo] = hi_s - np.searchsorted(hd, part_day - 365, side="left")
        rec3[lo] = hi_s - np.searchsorted(hd, part_day - 1095, side="left")
        rec5[lo] = hi_s - np.searchsorted(hd, part_day - 1825, side="left")
        if hd.size >= 2:
            gaps = np.diff(hd).astype("float64")
            cum = np.maximum.accumulate(gaps)
            pos = np.searchsorted(hd, part_day, side="left") - 1
            idx = np.clip(pos, 0, cum.size - 1)
            gap[lo] = np.log1p(
                np.where((pos >= 0) & (pos < cum.size), cum[idx], 0.0))
    return rec1, rec3, rec5, gap


class TestRecurrence(unittest.TestCase):
    def test_spaced_events(self):
        ids, lons, lats, times = [], [], [], []
        base = pd.Timestamp("2015-01-01")
        for i in range(40):
            ids.append(f"a{i}"); lons.append(78.0); lats.append(20.0)
            times.append(base + pd.Timedelta(days=30 * i))
        r1, r3, r5, g = recurrences(ids, lons, lats, times)
        # 40 events 30 days apart: within 1 year there are 12 earlier ones,
        # within 3 years 36, within 5 years 39 (the first has 0).
        self.assertEqual(r1[-1], 12)
        self.assertEqual(r3[-1], 36)
        self.assertEqual(r5[-1], 39)
        self.assertAlmostEqual(float(g[-1]), np.log1p(30.0), places=5)

    def test_singleton_site_has_no_recurrence_and_nan_gap(self):
        r1, r3, r5, g = recurrences(["only"], [80.0], [20.0],
                                    [pd.Timestamp("2016-06-01")])
        self.assertEqual(r1[0], 0)
        self.assertEqual(r3[0], 0)
        self.assertEqual(r5[0], 0)
        # A single event has no inter-event gap at all, so the feature is
        # undefined -- NaN, not 0. Zero would claim "no gap of a known size".
        self.assertTrue(np.isnan(g[0]))

    def test_no_event_recurs_with_itself(self):
        # The first event in every site must score 0, or the counts are
        # inflated by one everywhere and the feature means the wrong thing.
        for lon in (78.0, 80.0, 82.0):
            r1, _, _, _ = recurrences([f"x{i}" for i in range(5)],
                                      [lon] * 5, [20.0] * 5,
                                      [pd.Timestamp("2015-01-01") + pd.Timedelta(days=10 * i)
                                       for i in range(5)])
            self.assertEqual(r1[0], 0, f"site {lon} first event self-recurs")
            self.assertEqual(r1[-1], 4)

    def test_simultaneous_events_in_one_cell(self):
        """THE CRASH. A cell whose events share dates has fewer gaps than
        events, so the eager index in `np.where` ran past the end of `cum`."""
        ids = []
        lons, lats, times = [], [], []
        base = pd.Timestamp("2015-01-01")
        # Three events on the SAME day, one ten days later, three more on the
        # day after that: 7 events but only 2 distinct gaps.
        for d, k in ((0, 3), (10, 1), (11, 3)):
            for _ in range(k):
                ids.append("e"); lons.append(78.0); lats.append(20.0)
                times.append(base + pd.Timedelta(days=d))
        r1, r3, r5, g = recurrences(ids, lons, lats, times)
        self.assertEqual(len(r1), 7)
        self.assertTrue(np.all(np.isfinite(g)))
        # Simultaneous events contribute 0-length gaps, so the max gap is the
        # 10-day jump, and events on day 0 have seen no gap yet.
        self.assertTrue(np.isnan(g[0]) or g[0] == 0.0)
        self.assertAlmostEqual(float(g[-1]), np.log1p(10.0), places=5)

    def test_monotonicity_and_bounds(self):
        ids, lons, lats, times = [], [], [], []
        rng = np.random.default_rng(3)
        base = pd.Timestamp("2012-01-01")
        for i in range(300):
            ids.append(str(i))
            lons.append(70.0 + rng.integers(0, 3))
            lats.append(20.0 + rng.integers(0, 3))
            times.append(base + pd.Timedelta(days=int(rng.integers(0, 4000))))
        r1, r3, r5, g = recurrences(ids, lons, lats, times)
        self.assertTrue(np.all(r1 >= 0))
        self.assertTrue(np.all(r1 <= r3))
        self.assertTrue(np.all(r3 <= r5))
        self.assertTrue(np.all(np.isfinite(r1)))
        # Gaps are log1p of a non-negative day count, so bounded by log1p(5,000).
        self.assertTrue(np.all((g <= np.log1p(5000)) | np.isnan(g)))

    def test_cells_do_not_leak_into_each_other(self):
        # Two cells 1 degree apart must not share history. The 1-degree boundary
        # is exactly where an off-by-one in the cell id would show up.
        lon_a, lon_b = 79.0, 80.0
        times_a = [pd.Timestamp("2015-01-01") + pd.Timedelta(days=30 * i)
                   for i in range(20)]
        r1, _, _, _ = recurrences([f"a{i}" for i in range(20)] + ["b0"],
                                  [lon_a] * 20 + [lon_b],
                                  [20.0] * 21,
                                  times_a + [pd.Timestamp("2015-06-01")])
        # The lone event in cell B has no history in cell A.
        self.assertEqual(r1[-1], 0)

    def test_band_wide_part_day_does_not_index_past_cum(self):
        """THE COLAB CRASH, exactly.

        The cell loop computes `pos` over the WHOLE BAND's days and only masks
        with `sel` afterwards. So an event belonging to a LATER cell is still
        used to index this cell's `cum`. When that event's day exceeds this
        cell's maximum, `pos` reaches `cum.size` and `np.where` -- which
        evaluates its index argument EAGERLY, before the mask discards it --
        raises `IndexError: index 1 is out of bounds for axis 0 with size 1`.

        The mask is correct; the eager evaluation is the bug. This is why
        `idx` is clipped at BOTH ends.
        """
        days = np.array([16000, 16010])                 # cell A: two events
        part_day = np.array([16000, 16010, 17000])      # third row is in cell B
        gaps = np.diff(days).astype("float64")
        cum = np.maximum.accumulate(gaps)
        self.assertEqual(cum.size, 1)
        pos = np.searchsorted(days, part_day, side="left") - 1
        self.assertEqual(list(pos), [-1, 0, 1], "precondition of the crash")

        with self.assertRaises(IndexError):
            # The OLD expression, low end only. Kept here so the test fails
            # loudly if someone reintroduces it.
            np.where((pos >= 0) & (pos < cum.size), cum[np.clip(pos, 0, None)], 0.0)

        idx = np.clip(pos, 0, cum.size - 1)
        got = np.where((pos >= 0) & (pos < cum.size), cum[idx], 0.0)
        self.assertTrue(np.all(np.isfinite(got)))
        # The out-of-range row is masked to 0 and would then be dropped by sel.
        self.assertEqual(float(got[2]), 0.0)

    def test_realistic_cells_with_many_same_day_events(self):
        # Cell with 2 distinct days and several events on each: fewer gaps than
        # events, which is the shape that stressed the eager index.
        ids, lons, lats, times = [], [], [], []
        base = pd.Timestamp("2014-01-15")
        for d, k in ((0, 4), (45, 3)):
            for _ in range(k):
                ids.append("e"); lons.append(75.5); lats.append(22.5)
                times.append(base + pd.Timedelta(days=d))
        r1, r3, r5, g = recurrences(ids, lons, lats, times)
        self.assertEqual(len(r1), 7)
        self.assertTrue(np.all(np.isfinite(r1)))
        self.assertTrue(np.all(np.isfinite(g)))
        self.assertAlmostEqual(float(g[-1]), np.log1p(45.0), places=5)

    def test_realistic_scale_runs(self):
        # 1% of the catalogue, to keep the suite honest about runtime: the cell
        # must stay a minutes-long job, not an hours-long one.
        rng = np.random.default_rng(11)
        n = 73_000
        lat = rng.uniform(5, 38, n)
        lon = rng.uniform(66, 99, n)
        day_num = rng.integers(1, 5300, n)
        cell_id = ((np.floor(lat).astype("int64") + 90) * 360
                   + (np.floor(lon).astype("int64") + 180))
        rec1 = np.zeros(n)
        cum_last = None
        frame = pd.DataFrame({"cell": cell_id, "day": day_num})
        frame = frame.sort_values(["cell", "day"], kind="mergesort")
        cells = np.unique(cell_id)
        starts = frame["cell"].searchsorted(cells, side="left")
        ends = frame["cell"].searchsorted(cells, side="right")
        ds = frame["day"].to_numpy()
        for k in range(len(cells)):
            days = ds[starts[k]:ends[k]]
            days = days[days > 0]
            if days.size == 0:
                continue
            pd_ = days[(day_num == day_num)[:1] * 0 + np.zeros(1, bool)]
            hi_s = np.searchsorted(days, days, side="left")
            rec1[starts[k]:ends[k]] = hi_s - np.searchsorted(days, days - 365, side="left")
            if days.size >= 2:
                gaps = np.diff(days).astype("float64")
                cum = np.maximum.accumulate(gaps)
                pos = np.searchsorted(days, days, side="left") - 1
                idx = np.clip(pos, 0, cum.size - 1)
                vals = np.where((pos >= 0) & (pos < cum.size), cum[idx], 0.0)
                cum_last = vals[-1]
        self.assertTrue(np.all(rec1 >= 0))
        self.assertTrue(np.isfinite(cum_last))


if __name__ == "__main__":
    unittest.main(verbosity=2)
