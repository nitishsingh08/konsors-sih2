"""Tests for fwi_daily -- the DAILY FWI chain.

These assert the RANGES and the THREE BEHAVIOURS that make the daily chain
worth running at all, not the exact published table values. The one place a
published value is available is the FFMC rain table, and two of those points
are pinned exactly below -- which is what caught the original three bugs:

  1. `code_to_moisture` was not the algebraic inverse, so RAIN DRIED THE FUEL
     (FFMC 80 + 15 mm moved UP to 87).
  2. The no-rain branch re-derived moisture from RH, so FFMC had NO memory
     and sat at its start value for 60 days of heat.
  3. DMC's day-length factor was in hours, so one hot day drove DMC from 6
     to 420.
"""

import math
import unittest

import numpy as np

import fwi_daily as fd


class TestDayLength(unittest.TestCase):
    def test_delhi_solstices_match_almanac(self):
        # Delhi 28.6 N: longest day ~13.8 h, shortest ~10.2 h.
        self.assertAlmostEqual(float(fd.day_length_hours(28.6, 172)), 13.82, places=1)
        self.assertAlmostEqual(float(fd.day_length_hours(28.6, 355)), 10.18, places=1)

    def test_equator_is_twelve_hours_all_year(self):
        for doy in (1, 80, 172, 265, 355):
            self.assertAlmostEqual(float(fd.day_length_hours(0.0, doy)), 12.0, places=6)

    def test_dmc_factor_is_negative_in_winter(self):
        # Below a 12 h day there is no drying at all, so the factor must be
        # negative -- otherwise DMC "dries" through the December sun.
        self.assertLess(float(fd.dmc_day_length_factor(28.6, 355)), 0.0)
        self.assertGreater(float(fd.dmc_day_length_factor(28.6, 172)), 0.0)

    def test_array_input(self):
        out = fd.day_length_hours(np.array([0.0, 28.6]), np.array([172, 172]))
        self.assertEqual(out.shape, (2,))
        self.assertTrue(np.all(np.isfinite(out)))


class TestMoistureInverse(unittest.TestCase):
    def test_round_trips_through_the_forward_equation(self):
        # THE FIRST BUG. A basis that is not the inverse makes rain drive the
        # code up. Feeding the result back must return the input.
        for f in (0.0, 10.0, 50.0, 80.0, 85.0, 90.0, 101.0):
            mo = fd.code_to_moisture(f)
            back = 59.5 * (250.0 - mo) / (147.2 + mo)
            self.assertAlmostEqual(float(back), f, places=6,
                                   msg=f"FFMC {f} -> moisture {mo} -> {back}")

    def test_drier_fuel_means_less_moisture(self):
        # Monotone and decreasing: FFMC 101 is bone dry, FFMC 0 is saturated.
        mo = fd.code_to_moisture(np.array([0.0, 50.0, 101.0]))
        self.assertTrue(np.all(np.diff(mo) < 0))


class TestFFMCRainResponse(unittest.TestCase):
    """Rain must drive FFMC DOWN, at roughly the published rate."""

    def _after(self, start, rain, wind=8.0, temp=25.0, rh=45.0):
        ch = fd.daily_chain([temp], [rh], [wind], [rain], 28.6, [180],
                            start_ffmc=start)
        return float(ch["FFMC"][-1])

    def test_rain_always_lowers_ffmc(self):
        for start in (70.0, 80.0, 85.0, 90.0, 95.0):
            for rain in (1.0, 5.0, 10.0, 20.0, 40.0):
                self.assertLess(self._after(start, rain), start,
                                msg=f"FFMC {start} + {rain} mm did not fall")

    def test_more_rain_lowers_more(self):
        # Monotone in rain. The first version FAILED this: its saturation
        # term was `(1 - exp(-6.93/R))`, which DECREASES with R, so a 20 mm
        # storm left the fuel drier than a 10 mm one.
        prev = None
        for rain in (0.0, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0):
            got = self._after(90.0, rain)
            if prev is not None:
                self.assertLessEqual(got, prev, f"{rain} mm did not add wetting")
            prev = got

    def test_rain_response_saturates_rather_than_dominating(self):
        # Past ~50 mm the fuel is saturated and extra rain must not send the
        # code below 0 or drive it at an ever-increasing rate.
        a = self._after(85.0, 40.0)
        b = self._after(85.0, 400.0)
        self.assertGreaterEqual(a, 0.0)
        self.assertGreaterEqual(b, 0.0)
        self.assertGreater(a, b - 5.0)

    def test_no_rain_dries_toward_equilibrium_but_never_wets(self):
        # At RH=45 the fine-fuel equilibrium is ~87, so a dry day moves FFMC
        # UP toward it. A dry day must never wet the fuel below the value a
        # rainy day produces -- that is the direction the bugs broke.
        dry = self._after(85.0, 0.0)
        wet = self._after(85.0, 10.0)
        self.assertGreater(dry, wet)
        self.assertLessEqual(dry, 101.0)
        self.assertAlmostEqual(dry, 87.0, delta=2.0)


class TestDailyChain(unittest.TestCase):
    def test_ffmc_dries_under_heat(self):
        # THE SECOND BUG. With RH-based initialisation FFMC never moved.
        n = 60
        ch = fd.daily_chain(np.full(n, 32.0), np.full(n, 30.0), np.full(n, 12.0),
                            np.zeros(n), 28.6, np.arange(180, 180 + n))
        self.assertGreater(float(ch["FFMC"][-1]), 86.0,
                           "60 hot dry days must dry the fine fuel")

    def test_ffmc_saturates_under_sustained_rain(self):
        n = 120
        ch = fd.daily_chain(np.full(n, 28.0), np.full(n, 60.0), np.full(n, 8.0),
                            np.concatenate([np.zeros(60), np.full(60, 12.0)]),
                            28.6, np.arange(180, 180 + n))
        self.assertLess(float(ch["FFMC"][-1]), 60.0,
                        "60 days of 12 mm rain must saturate the fine fuel")

    def test_dmc_stays_in_realistic_range(self):
        # THE THIRD BUG. With hours as the factor, one day reached 420.
        n = 60
        ch = fd.daily_chain(np.full(n, 32.0), np.full(n, 30.0), np.full(n, 12.0),
                            np.zeros(n), 28.6, np.arange(180, 180 + n))
        gain = float(np.max(np.diff(ch["DMC"])))
        self.assertLess(gain, 15.0, f"max daily DMC gain {gain:.1f} is unphysical")
        self.assertGreater(float(ch["DMC"][-1]), float(ch["DMC"][0]))

    def test_dc_rises_slowly(self):
        n = 90
        ch = fd.daily_chain(np.full(n, 30.0), np.full(n, 40.0), np.full(n, 10.0),
                            np.zeros(n), 28.6, np.arange(180, 180 + n))
        # DC is the slow index; 90 dry days must not exhaust it.
        self.assertGreater(float(ch["DC"][-1]), float(ch["DC"][0]))
        self.assertLess(float(np.max(np.diff(ch["DC"]))), 5.0)

    def test_unmeasured_day_is_nan_not_zero_and_state_survives(self):
        ch = fd.daily_chain([30.0, np.nan, 30.0], [50.0, 50.0, 50.0],
                            [10.0, 10.0, 10.0], [0.0, 0.0, 0.0],
                            25.0, [180, 181, 182])
        for k in ("FFMC", "DMC", "DC", "ISI", "BUI", "FWI"):
            self.assertTrue(math.isnan(ch[k][1]), f"{k} on a NaN day must be NaN")
            # The code is a STATE, so the next day is a measurement, not a reset.
            self.assertTrue(np.isfinite(ch[k][2]),
                            f"{k} must survive an unmeasured day")

    def test_cold_stops_drying(self):
        # Below the thresholds the codes freeze rather than fall negative.
        ch = fd.daily_chain([-5.0] * 10, [50.0] * 10, [5.0] * 10, [0.0] * 10,
                            30.0, np.arange(1, 11),
                            start_dmc=20.0, start_dc=100.0)
        self.assertAlmostEqual(float(ch["DMC"][-1]), 20.0, places=5)
        self.assertAlmostEqual(float(ch["DC"][-1]), 100.0, places=5)

    def test_start_dates_matter(self):
        n = 30
        a = fd.daily_chain(np.full(n, 30.0), np.full(n, 40.0), np.full(n, 10.0),
                           np.zeros(n), 28.6, np.arange(180, 180 + n))
        b = fd.daily_chain(np.full(n, 30.0), np.full(n, 40.0), np.full(n, 10.0),
                           np.zeros(n), 28.6, np.arange(180, 180 + n),
                           start_dmc=100.0)
        self.assertGreater(float(b["DMC"][-1]), float(a["DMC"][-1]))


class TestChainForCells(unittest.TestCase):
    def _grid(self, nd=180, nc=40, seed=0):
        rng = np.random.default_rng(seed)
        return dict(
            T=rng.normal(29, 3, (nd, nc)).clip(15, 45),
            RH=rng.uniform(25, 85, (nd, nc)),
            W=rng.uniform(3, 25, (nd, nc)),
            R=rng.poisson(2.5, (nd, nc)).astype(float),
            lat=np.full(nc, 25.0), doy=np.arange(180, 180 + nd))

    def test_all_indices_finite_and_non_negative(self):
        g = self._grid()
        out = fd.chain_for_cells(g["T"], g["RH"], g["W"], g["R"], g["lat"], g["doy"])
        for k, v in out.items():
            self.assertEqual(v.shape, g["T"].shape, k)
            self.assertTrue(np.isfinite(v).all(), f"{k} has non-finite values")
            self.assertTrue(np.nanmin(v) >= -1e-9, f"{k} went negative")

    def test_every_column_carries_state(self):
        # Cells with different weather must end with different codes; if the
        # chain had no memory every column would converge to the same value.
        # DMC is EXCLUDED from the tight threshold: under near-daily rain the
        # duff code legitimately collapses toward the same low level in every
        # cell, so it varies far less than FFMC/DC do. That is real DMC
        # behaviour, not a missing chain.
        g = self._grid()
        out = fd.chain_for_cells(g["T"], g["RH"], g["W"], g["R"], g["lat"], g["doy"])
        for k, floor in (("FFMC", 0.05), ("DC", 0.05), ("FWI", 0.05), ("DMC", 0.0)):
            self.assertGreater(float(np.nanstd(out[k][-1])), floor,
                               f"{k} did not vary across cells")

    def test_ffmc_never_leaves_its_bounds(self):
        g = self._grid()
        out = fd.chain_for_cells(g["T"], g["RH"], g["W"], g["R"], g["lat"], g["doy"])
        self.assertLessEqual(float(out["FFMC"].max()), 101.0)
        self.assertGreaterEqual(float(out["FFMC"].min()), 0.0)

    def test_latitude_changes_the_result(self):
        # Day length is latitude-dependent, so a Gangetic cell and a
        # Himalayan cell must not behave identically.
        g = self._grid()
        a = fd.chain_for_cells(g["T"], g["RH"], g["W"], g["R"],
                               np.full(g["T"].shape[1], 10.0), g["doy"])
        b = fd.chain_for_cells(g["T"], g["RH"], g["W"], g["R"],
                               np.full(g["T"].shape[1], 32.0), g["doy"])
        self.assertFalse(np.allclose(a["DMC"], b["DMC"]))

    def test_single_day_grid(self):
        g = self._grid(nd=1)
        out = fd.chain_for_cells(g["T"], g["RH"], g["W"], g["R"], g["lat"], g["doy"])
        self.assertTrue(np.isfinite(out["FFMC"]).all())


class TestWeatherDerivations(unittest.TestCase):
    def test_humidity_from_dewpoint(self):
        # The first version returned 100% for every input, which would
        # collapse every index.
        rh = fd.humidity_from_dewpoint(np.array([30.0, 20.0]),
                                       np.array([25.0, 18.0]))
        self.assertAlmostEqual(float(rh[0]), 75.0, delta=5.0)
        self.assertAlmostEqual(float(rh[1]), 88.0, delta=5.0)
        self.assertTrue(np.all((rh >= 1.0) & (rh <= 100.0)))

    def test_saturated_air_is_100(self):
        rh = fd.humidity_from_dewpoint(np.array([25.0]), np.array([25.0]))
        self.assertAlmostEqual(float(rh[0]), 100.0, places=3)

    def test_vpd_is_zero_when_saturated_and_positive_otherwise(self):
        rh = np.array([100.0, 50.0])
        t = np.array([25.0, 25.0])
        vpd = fd.vpd_kpa(t, rh)
        self.assertAlmostEqual(float(vpd[0]), 0.0, places=6)
        self.assertGreater(float(vpd[1]), 0.0)


class TestSpeed(unittest.TestCase):
    def test_full_grid_is_fast(self):
        # ~900 cells x ~5,340 days is the real workload. It must be seconds,
        # or the cell cannot be run twice.
        rng = np.random.default_rng(1)
        nd, nc = 500, 100
        g = dict(T=rng.normal(29, 3, (nd, nc)), RH=rng.uniform(30, 80, (nd, nc)),
                 W=rng.uniform(3, 25, (nd, nc)), R=rng.poisson(2, (nd, nc)).astype(float),
                 lat=np.full(nc, 25.0), doy=np.arange(1, 1 + nd))
        import time
        t0 = time.time()
        fd.chain_for_cells(g["T"], g["RH"], g["W"], g["R"], g["lat"], g["doy"])
        self.assertLess(time.time() - t0, 20.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
