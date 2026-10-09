import unittest

import numpy as np
import pandas as pd

import leakage_probe as lp


def frame(n, lons, lats, years, blocks=None, labels=None, prefix="E"):
    return pd.DataFrame({
        "event_id": [f"{prefix}_{i:08d}" for i in range(n)],
        "event_centroid_lon": lons,
        "event_centroid_lat": lats,
        "start_year": years,
        "start_time": pd.to_datetime(years, format="%Y") + pd.to_timedelta(
            np.arange(n), unit="D"),
        "spatial_block": blocks if blocks is not None else ["b0"] * n,
        "label_weak": labels if labels is not None else ["UNKNOWN"] * n,
    })


class TestBlockDisjointness(unittest.TestCase):
    def test_clean(self):
        a = frame(2, [0.0, 0.0], [0.0, 0.0], [2018, 2019], blocks=["a", "a"])
        b = frame(2, [5.0, 5.0], [5.0, 5.0], [2021, 2022], blocks=["b", "b"])
        r = lp.block_disjointness(a, b)
        self.assertTrue(r["clean"])
        self.assertEqual(r["n_shared"], 0)

    def test_shared_block_detected(self):
        a = frame(2, [0.0, 0.0], [0.0, 0.0], [2018, 2019], blocks=["a", "z"])
        b = frame(2, [5.0, 5.0], [5.0, 5.0], [2021, 2022], blocks=["b", "z"])
        r = lp.block_disjointness(a, b)
        self.assertFalse(r["clean"])
        self.assertEqual(r["n_shared"], 1)
        self.assertEqual(r["examples"], ["z"])


class TestProximity(unittest.TestCase):
    def test_twin_detected(self):
        # test event 50 m from a train event, on the other side of a block edge
        a = frame(1, [75.0], [26.0], [2018], blocks=["a"])
        b = frame(1, [75.00045], [26.00045], [2021], blocks=["b"])
        r = lp.cross_split_proximity(a, b)
        self.assertGreater(r["twin_rate_1km"], 0.99)
        self.assertLess(r["min_nearest_train_km"], 0.1)

    def test_far_apart_is_clean(self):
        a = frame(1, [75.0], [26.0], [2018], blocks=["a"])
        b = frame(1, [80.0], [30.0], [2021], blocks=["b"])
        r = lp.cross_split_proximity(a, b)
        self.assertEqual(r["twin_rate_1km"], 0.0)
        self.assertGreater(r["median_nearest_train_km"], 400.0)

    def test_radius_shares_are_monotone(self):
        a = frame(1, [75.0], [26.0], [2018])
        b = frame(1, [75.06], [26.0], [2021])          # ~6.6 km away
        r = lp.cross_split_proximity(a, b)
        sh = r["frac_test_within_radius"]
        self.assertEqual(sh["1.0"], 0.0)
        self.assertEqual(sh["5.0"], 0.0)
        self.assertEqual(sh["10.0"], 1.0)

    def test_missing_column_is_named(self):
        a = frame(1, [75.0], [26.0], [2018])
        b = a.drop(columns=["event_centroid_lon"])
        with self.assertRaises(KeyError) as ctx:
            lp.cross_split_proximity(a, b)
        self.assertIn("event_centroid_lon", str(ctx.exception))

    def test_nonfinite_coordinates_dropped(self):
        a = frame(1, [75.0], [26.0], [2018])
        b = frame(2, [np.nan, 75.0001], [26.0, 26.0001], [2021, 2021])
        r = lp.cross_split_proximity(a, b)
        self.assertEqual(r["n_test_finite"], 1)


class TestYears(unittest.TestCase):
    def test_overlap_counted(self):
        a = frame(1, [75.0], [26.0], [2021])
        b = frame(1, [80.0], [30.0], [2021])
        r = lp.cross_split_year_overlap(a, b)
        self.assertEqual(r["n_shared_years"], 1)
        self.assertEqual(r["test_year_fraction_seen_in_train"], 1.0)

    def test_true_out_of_time(self):
        a = frame(1, [75.0], [26.0], [2015])
        b = frame(1, [80.0], [30.0], [2021])
        r = lp.cross_split_year_overlap(a, b)
        self.assertEqual(r["n_shared_years"], 0)
        self.assertEqual(r["test_year_fraction_seen_in_train"], 0.0)


class TestVerdict(unittest.TestCase):
    def test_high_twin_rate_flags(self):
        v = lp.verdict({"twin_rate_1km": 0.31},
                       {"clean": True, "n_shared": 0},
                       {"n_shared_years": 7})
        self.assertFalse(v["clean"])
        self.assertIn("LEAKAGE", v["verdict"])

    def test_moderate_twin_rate_flags(self):
        v = lp.verdict({"twin_rate_1km": 0.09}, {"clean": True, "n_shared": 0},
                       {})
        self.assertFalse(v["clean"])

    def test_shared_block_flags_even_with_no_twins(self):
        v = lp.verdict({"twin_rate_1km": 0.0},
                       {"clean": False, "n_shared": 4}, {})
        self.assertFalse(v["clean"])
        self.assertIn("both sides", v["verdict"])

    def test_clean_split(self):
        v = lp.verdict({"twin_rate_1km": 0.001},
                       {"clean": True, "n_shared": 0}, {})
        self.assertTrue(v["clean"])


class TestLabelRateGap(unittest.TestCase):
    def test_yearly_rates(self):
        a = frame(4, [75.0] * 4, [26.0] * 4, [2018] * 4,
                  labels=["INDUSTRIAL_FIRE"] * 2 + ["UNKNOWN"] * 2)
        b = frame(4, [80.0] * 4, [30.0] * 4, [2021] * 4,
                  labels=["INDUSTRIAL_FIRE", "UNKNOWN", "UNKNOWN", "UNKNOWN"])
        r = lp.label_rate_gap(a, b)
        self.assertAlmostEqual(r["train"][2018]["rate"], 0.5)
        self.assertAlmostEqual(r["test"][2021]["rate"], 0.25)


class TestProbe(unittest.TestCase):
    def test_probe_collects_skips(self):
        a = frame(2, [75.0, 75.1], [26.0, 26.1], [2018, 2019],
                  blocks=["a", "a"])
        b = frame(2, [80.0, 80.1], [30.0, 30.1], [2021, 2022],
                  blocks=["b", "b"]).drop(columns=["label_weak"])
        r = lp.probe(a, b)
        self.assertIn("cross_split_proximity", r["ran"])
        self.assertIn("label_rate_gap", r["skipped"])
        self.assertIn("verdict", r)

    def test_probe_verdict_on_clean(self):
        a = frame(2, [75.0, 75.1], [26.0, 26.1], [2018, 2019],
                  blocks=["a", "a"])
        b = frame(2, [80.0, 80.1], [30.0, 30.1], [2021, 2022],
                  blocks=["b", "b"])
        r = lp.probe(a, b)
        self.assertTrue(r["verdict"]["clean"])


if __name__ == "__main__":
    unittest.main()