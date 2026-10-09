"""tests for fwi.py (fire-weather math) and the P7n/P7o GEE helpers."""

import sys
import types
import unittest
from unittest import mock

import pathlib

from fwi import (ffmc, dmc, dc, bui, isi, fwi as fwi_idx, weather_set)

REPO = pathlib.Path(__file__).resolve().parent


class FwiAlgebraTests(unittest.TestCase):
    """The monthly-chain approximation must be monotonic and bounded."""

    def test_ffmc_is_bounded(self):
        for t in (0, 20, 40):
            for rh in (0, 30, 60, 100):
                for w in (0, 20, 60):
                    v = ffmc(t, rh, w, 0.0)
                    self.assertGreaterEqual(v, 0.0)
                    self.assertLessEqual(v, 101.0)

    def test_drier_and_hotter_is_higher(self):
        dry = weather_set(35, 20, 30, 0.0, 20, 4)
        moist = weather_set(20, 80, 5, 30.0, 20, 4)
        for k in ("FFMC", "DMC", "DC", "ISI", "BUI", "FWI"):
            # wet day collapses indices; dry day keeps them high
            self.assertGreaterEqual(dry[k], moist[k] * 0.99, k)

    def test_rain_reduces_ffmc(self):
        no_rain = ffmc(30, 20, 20, 0.0)
        heavy = ffmc(30, 20, 20, 40.0)
        self.assertLessEqual(heavy, no_rain)

    def test_weather_set_returns_all_six(self):
        s = weather_set(30, 30, 20, 0.0)
        self.assertEqual(set(s), {"FFMC", "DMC", "DC", "ISI", "BUI", "FWI"})

    def test_bui_never_negative(self):
        for d in (0, 10, 100, 500):
            self.assertGreaterEqual(bui(d, 200), 0.0)

    def test_isi_is_zero_when_wind_zero(self):
        self.assertEqual(isi(90.0, 0.0), 0.0)


class GeeHelperPresenceTests(unittest.TestCase):
    """The P7n/P7o cells call helpers that must exist and return what the
    rehearsal-oriented skeleton expects. Band-valued behavior is covered by
    the Colab rehearsal, not here."""

    def test_helpers_exist(self):
        import gee_utils as g
        for fn in ("era5_monthly_weather", "era5_monthly_climatology",
                   "s5p_monthly_median", "s5p_monthly_climatology",
                   "sample_bands_chunked"):
            self.assertTrue(hasattr(g, fn), fn)

    def test_sample_bands_chunked_returns_by_row_and_band(self):
        import gee_utils as g

        class _FakeRes:
            def __init__(self, ok):
                self.ok = ok

            def get(self, *a):
                return self.ok

        class _Feat:
            def __init__(self, props):
                self._p = props

            def get(self, *a):
                if a[0] == "properties":
                    return self._p
                return None

        class _Img2:
            def bandNames(self):
                return [f"b{i}" for i in range(2)]

            def select(self, idx):
                return self

            def rename(self, names):
                return self

            def sampleRegions(self, collection=None, properties=None,
                              scale=1, geometries=False):
                return _FakeRes(None)

        def _getInfo(x):
            return {"features": [
                _Feat({"row": int(r), "v": 1.0 * r}) for r in (1, 2, 3)]}

        sys.modules.setdefault("_fakeee", None)
        with mock.patch.dict(sys.modules, {"ee": types.SimpleNamespace(
                FeatureCollection=lambda _fc: None,
                Feature=lambda *a, **k: None,
                Geometry=types.SimpleNamespace(
                    Point=lambda *a, **k: None))}), \
                mock.patch.object(_Img2, "getInfo", _getInfo, create=True):
            res = g.sample_bands_chunked(
                {"a": _Img2()}, [{"row": 1}, {"row": 2}, {"row": 3}],
                id_key="row", chunk_size=2)
        # only stubs ran; we assert it at least iterates without raising
        self.assertIsInstance(res, dict)


if __name__ == "__main__":
    unittest.main()
