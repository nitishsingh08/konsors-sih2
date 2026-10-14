"""
Tests for Earth Engine helpers.

`ee.Initialize()` currently fails with "no project found" unless a Cloud
project id is supplied, and that message reads like a credentials problem when
it is not. These tests pin the part that is testable without Earth Engine: how
a project id gets resolved, and that the failure path produces remediation text
rather than a bare exception.

Nothing here imports `ee`. The dataset check is exercised through a stub.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gee_utils as gee


class ProjectResolutionTests(unittest.TestCase):
    def setUp(self):
        for var in gee.PROJECT_ENV_VARS:
            os.environ.pop(var, None)
        self.addCleanup(
            lambda: [os.environ.pop(v, None) for v in gee.PROJECT_ENV_VARS]
        )

    def test_no_env_vars_means_no_candidates(self):
        # `gcloud` may genuinely be absent; either way the contract is a list.
        self.assertIsInstance(gee.candidate_project_ids(), list)

    def test_first_env_var_wins_and_order_is_preserved(self):
        os.environ["EARTHENGINE_PROJECT"] = "aaa"
        os.environ["GOOGLE_CLOUD_PROJECT"] = "bbb"
        self.assertEqual(gee.candidate_project_ids()[0], "aaa")
        self.assertIn("bbb", gee.candidate_project_ids())

    def test_earlier_var_outranks_later(self):
        os.environ["EARTHENGINE_PROJECT"] = "first"
        os.environ["GCP_PROJECT"] = "last"
        found = gee.candidate_project_ids()
        self.assertEqual(found.index("first"), 0)

    def test_blank_and_duplicate_values_are_dropped(self):
        os.environ["EARTHENGINE_PROJECT"] = "   "
        os.environ["GOOGLE_CLOUD_PROJECT"] = "dup"
        os.environ["GCP_PROJECT"] = "dup"
        found = gee.candidate_project_ids()
        self.assertNotIn("", found)
        self.assertNotIn("   ", found)
        self.assertEqual(found.count("dup"), 1)

    def test_explicit_argument_is_tried_first(self):
        """The project the user typed must beat what is in the environment."""
        os.environ["EARTHENGINE_PROJECT"] = "from-env"
        os.environ["GOOGLE_CLOUD_PROJECT"] = "from-env"
        attempted = []

        class _FakeEE:
            @staticmethod
            def Initialize(project=None):
                attempted.append(project)
                if project != "explicit":
                    raise RuntimeError("no project found")

        ok, pid = _call_with_fake(_FakeEE, "explicit")
        self.assertTrue(ok)
        self.assertEqual(pid, "explicit")
        self.assertEqual(attempted[0], "explicit")


def _call_with_fake(fake_ee, project_id):
    """Drive `gee.authenticate` against a stub `ee` module."""
    import sys

    with mock.patch.dict(sys.modules, {"ee": fake_ee}, create=True):
        return gee.authenticate(project_id, allow_interactive=False)


class AuthFlowTests(unittest.TestCase):
    def _ee_that_needs(self, *projects):
        """A stub `ee` whose Initialize only succeeds for the given projects."""
        state = {"attempts": [], "authenticated": False}

        class _FakeEE:
            class _Exc(Exception):
                pass

            @staticmethod
            def Initialize(project=None):
                state["attempts"].append(project)
                if project in projects:
                    return None
                raise _FakeEE._Exc("no project found")

            @staticmethod
            def Authenticate():
                state["authenticated"] = True

        return _FakeEE, state

    def _run(self, fake_ee, project_id=None, env=None):
        import sys

        if env:
            for k, v in env.items():
                os.environ[k] = v
                self.addCleanup(os.environ.pop, k, None)
        with mock.patch.dict(sys.modules, {"ee": fake_ee}, create=True):
            return gee.authenticate(project_id, allow_interactive=False)

    def test_explicit_project_succeeds_immediately(self):
        fake, state = self._ee_that_needs("my-proj")
        ok, pid = self._run(fake, "my-proj")
        self.assertTrue(ok)
        self.assertEqual(pid, "my-proj")
        self.assertEqual(state["attempts"], ["my-proj"])

    def test_project_is_discovered_from_the_environment(self):
        fake, state = self._ee_that_needs("env-proj")
        ok, pid = self._run(fake, env={"EARTHENGINE_PROJECT": "env-proj"})
        self.assertTrue(ok)
        self.assertEqual(pid, "env-proj")

    def test_no_project_available_raises_with_remediation(self):
        fake, _state = self._ee_that_needs()
        with self.assertRaises(gee.EEAuthError) as ctx:
            self._run(fake)
        text = str(ctx.exception)
        self.assertIn("ee.Authenticate()", text)
        self.assertIn("Earth Engine API", text)
        self.assertIn("code.earthengine.google.com", text)

    def test_the_error_lists_what_was_actually_tried(self):
        """A failure that does not say what it tried sends people in circles."""
        fake, _state = self._ee_that_needs()
        with self.assertRaises(gee.EEAuthError) as ctx:
            self._run(fake, "attempted-proj")
        self.assertIn("attempted-proj", str(ctx.exception))

    def test_it_does_not_ask_for_consent_when_not_permitted(self):
        """`allow_interactive=False` must not pop a browser window."""
        fake, state = self._ee_that_needs()
        with self.assertRaises(gee.EEAuthError):
            self._run(fake)
        self.assertFalse(state["authenticated"])


class DatasetKindTests(unittest.TestCase):
    """`ee.Image` on an ImageCollection does not raise -- it fails much later.

    `ee.Image("COPERNICUS/S2_HARMONIZED")` builds a composite Image with no
    `filterDate`, so the mistake surfaces as
    `AttributeError: 'Image' object has no attribute 'filterDate'` far from the
    actual mistake. Hence this is pinned as data.
    """

    def test_every_dataset_this_project_uses_is_a_collection(self):
        """VERIFIED against the EE catalog 2026-10-04. WorldCover and MODIS
        were previously classified as images and both raised
        `Asset ... is not an Image`."""
        for did in (gee.WORLDCOVER, gee.MODIS_LANDCOVER, gee.S2_SR,
                    gee.LANDSAT_NBR, gee.LANDSAT5_NBR, gee.LANDSAT7_NBR):
            self.assertEqual(gee.dataset_kind(did), "collection", did)

    def test_no_dataset_is_classified_as_a_bare_image(self):
        """Empty on purpose. Adding to it requires checking the catalog snippet
        first -- that is how WorldCover and MODIS got in wrongly."""
        self.assertEqual(gee.IMAGE_IDS, frozenset())

    def test_the_two_sets_are_disjoint(self):
        self.assertEqual(gee.COLLECTION_IDS & gee.IMAGE_IDS, frozenset())

    def test_unknown_ids_default_to_collection(self):
        """The safer default: a collection supports filterDate, an image does
        not, so guessing 'image' breaks every dated query."""
        self.assertEqual(gee.dataset_kind("SOME/UNKNOWN_DATASET"), "collection")

    def test_ee_dataset_always_builds_a_collection(self):
        import sys

        built = []

        class _Coll:
            def __init__(self, i):
                built.append(("collection", i))

        class _Img:
            def __init__(self, i):
                built.append(("image", i))

        class _EE:
            Image = _Img
            ImageCollection = _Coll

        with mock.patch.dict(sys.modules, {"ee": _EE}, create=True):
            gee.ee_dataset(gee.S2_SR)
            gee.ee_dataset(gee.WORLDCOVER)
        self.assertEqual(built, [("collection", gee.S2_SR),
                                 ("collection", gee.WORLDCOVER)])

    def test_worldcover_image_helper_uses_collection_first(self):
        """`ee.Image("ESA/WorldCover/v200")` raises `is not an Image`. The helper
        must go through ImageCollection(...).first().select('Map')."""
        import sys

        calls = []

        class _Sel:
            def __init__(self, b):
                calls.append(("select", b))

        class _Img:
            def __init__(self, asset_id):
                calls.append(("first", asset_id))

            def select(self, b):
                return _Sel(b)

        class _Coll:
            def __init__(self, asset_id):
                self._id = asset_id
                calls.append(("collection", asset_id))

            def first(self):
                return _Img(self._id)

        class _EE:
            ImageCollection = _Coll

        with mock.patch.dict(sys.modules, {"ee": _EE}, create=True):
            gee.worldcover_image()
        self.assertEqual(calls, [("collection", gee.WORLDCOVER),
                                 ("first", gee.WORLDCOVER),
                                 ("select", "Map")])

    def test_verify_datasets_reports_kind(self):
        class _Names:
            def getInfo(self):
                return ["a", "b", "c"]

        class _Obj:
            def bandNames(self):
                return _Names()

            def first(self):
                return self

        class _Img(_Obj):
            def __init__(self, i):
                pass

        class _Coll(_Obj):
            def __init__(self, i):
                pass

        class _EE:
            Image = _Img
            ImageCollection = _Coll

        import sys

        with mock.patch.dict(sys.modules, {"ee": _EE}, create=True):
            out = gee.verify_datasets({
                "s2": (gee.S2_SR, ["a"], "collection"),
                "wc": (gee.WORLDCOVER, ["a"], "collection"),
            })
        self.assertEqual(out["s2"]["kind"], "collection")
        self.assertEqual(out["wc"]["kind"], "collection")
        self.assertTrue(out["s2"]["ok"])


class PermissionTests(unittest.TestCase):
    """`Initialize` succeeding is NOT a working session.

    It accepts credentials without checking the caller may USE the project, so
    a project owned by a different account -- exactly what an account switch
    produces -- reports success and then 403s on every data call. That gap let
    four confusing per-dataset failures through before anyone saw the cause.
    """

    def test_a_permission_denial_becomes_its_own_error_type(self):
        import sys

        class _Img:
            def __init__(self, *a):
                pass

            @staticmethod
            def constant(_v):
                class _X:
                    @staticmethod
                    def bandNames():
                        class _N:
                            def getInfo(self):
                                raise RuntimeError(
                                    "Caller does not have required permission "
                                    "to use project some-id. Grant the caller "
                                    "the role "
                                    "roles/serviceusage.serviceUsageConsumer")
                        return _N()
                return _X()

        class _EE:
            Image = _Img

        with mock.patch.dict(sys.modules, {"ee": _EE}, create=True):
            with self.assertRaises(gee.EEPermissionError) as ctx:
                gee.verify_can_read("some-id")
        self.assertEqual(ctx.exception.project_id, "some-id")
        msg = str(ctx.exception)
        self.assertIn("DIFFERENT GOOGLE ACCOUNT", msg)
        self.assertIn("serviceUsageConsumer", msg)

    def test_the_probe_uses_no_dataset_at_all(self):
        """Regression: the probe used ee.Image(WORLDCOVER), so a misconfigured
        asset id produced `Earth Engine session works but the probe failed`
        -- a healthy session reported as broken. A permission check must not
        depend on any dataset being correct."""
        import sys

        touched = []

        class _Img:
            def __init__(self, i=None):
                touched.append(i)

            @staticmethod
            def constant(_v):
                class _X:
                    @staticmethod
                    def bandNames():
                        class _N:
                            def getInfo(self):
                                return ["constant"]
                        return _N()
                return _X()

        class _EE:
            Image = _Img

        with mock.patch.dict(sys.modules, {"ee": _EE}, create=True):
            gee.verify_can_read("pid")
        self.assertNotIn(gee.WORLDCOVER, touched)

    def test_the_remediation_names_both_causes(self):
        self.assertIn("(a)", gee.EEPermissionError.REMEDIATION)
        self.assertIn("(b)", gee.EEPermissionError.REMEDIATION)

    def test_a_permission_error_is_not_an_auth_error(self):
        """They need different fixes; conflating them sends people to re-auth
        when the credentials were already fine."""
        self.assertFalse(issubclass(gee.EEPermissionError, gee.EEAuthError))
        self.assertFalse(issubclass(gee.EEAuthError, gee.EEPermissionError))

    def test_projects_visible_reads_the_asset_roots(self):
        import sys

        class _EE:
            class data:
                @staticmethod
                def getAssetRoots():
                    return [
                        {"id": "projects/alpha-project/assets"},
                        {"id": "projects/beta-project/assets"},
                        {"id": "nonsense"},
                    ]

        with mock.patch.dict(sys.modules, {"ee": _EE}, create=True):
            got = gee.projects_visible_to_current_account()
        self.assertEqual(got, ["alpha-project", "beta-project"])

    def test_projects_visible_survives_a_broken_api(self):
        import sys

        class _EE:
            class data:
                @staticmethod
                def getAssetRoots():
                    raise RuntimeError("method removed in this version")

        with mock.patch.dict(sys.modules, {"ee": _EE}, create=True):
            self.assertEqual(gee.projects_visible_to_current_account(), [])


class VersionTests(unittest.TestCase):
    """Staleness must be visible on the first line of output, not inferred.

    This project has been bitten by a stale cell three times. `gee_utils` and
    `colab_p7e_gee_probe.py` therefore both carry a version that is printed
    unconditionally, so an old copy on Drive announces itself instead of
    silently producing old behaviour.
    """

    def test_the_module_exposes_a_version(self):
        self.assertRegex(gee.__version__, r"^\d+\.\d+\.\d+$")

    def test_p7e_prints_both_versions(self):
        src = (Path(__file__).resolve().parent
               / "colab_p7e_gee_probe.py").read_text()
        self.assertIn("P7E_CELL_VERSION", src)
        self.assertIn("cell version", src)
        self.assertIn("gee.__version__", src.replace("getattr(gee, '__version__'",
                                                     "gee.__version__"))

    def test_the_stale_warning_is_actually_present(self):
        src = (Path(__file__).resolve().parent
               / "colab_p7e_gee_probe.py").read_text()
        self.assertIn("STALE", src)


class DotenvTests(unittest.TestCase):
    """The project id must be usable every session without ever being committed."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _write(self, text):
        (self.root / ".env").write_text(text, newline="")

    def test_it_reads_the_project_key(self):
        self._write("EARTHENGINE_PROJECT=my-proj-123\n")
        self.assertEqual(gee.project_ids_from_dotenv(self.root), ["my-proj-123"])

    def test_it_is_gitignored_so_the_id_is_never_committed(self):
        """The whole point of using `.env`. If this stops being true the
        mechanism leaks a billing-account name into the repository."""
        gitignore = (Path(__file__).resolve().parent / ".gitignore")
        text = gitignore.read_text()
        self.assertTrue(any(line.strip() == ".env" for line in text.splitlines()),
                        ".env is no longer git-ignored; do not put a project "
                        "id in it")

    def test_quotes_and_whitespace_are_stripped(self):
        self._write('EARTHENGINE_PROJECT = "  spaced-proj  " \n')
        self.assertEqual(gee.project_ids_from_dotenv(self.root), ["spaced-proj"])

    def test_single_quotes_are_stripped(self):
        self._write("EARTHENGINE_PROJECT='sq-proj'\n")
        self.assertEqual(gee.project_ids_from_dotenv(self.root), ["sq-proj"])

    def test_export_prefix_is_handled(self):
        self._write("export EARTHENGINE_PROJECT=exp-proj\n")
        self.assertEqual(gee.project_ids_from_dotenv(self.root), ["exp-proj"])

    def test_comments_and_blank_lines_are_skipped(self):
        self._write("# a comment\n\nEARTHENGINE_PROJECT=ok-proj\n# trailing\n")
        self.assertEqual(gee.project_ids_from_dotenv(self.root), ["ok-proj"])

    def test_crlf_line_endings_are_handled(self):
        """A file edited on Windows arrives with \\r and silently breaks a
        naive parser."""
        self._write("EARTHENGINE_PROJECT=crlf-proj\r\n")
        self.assertEqual(gee.project_ids_from_dotenv(self.root), ["crlf-proj"])

    def test_inline_trailing_comment_is_dropped(self):
        self._write("EARTHENGINE_PROJECT=real-proj # my project\n")
        self.assertEqual(gee.project_ids_from_dotenv(self.root), ["real-proj"])

    def test_unrelated_keys_are_ignored(self):
        self._write("DATABASE_URL=postgres://x\nEARTHENGINE_PROJECT=keep\n")
        self.assertEqual(gee.project_ids_from_dotenv(self.root), ["keep"])

    def test_a_malformed_line_does_not_raise(self):
        self._write("this line has no equals sign\nEARTHENGINE_PROJECT=fine\n")
        self.assertEqual(gee.project_ids_from_dotenv(self.root), ["fine"])

    def test_missing_file_is_not_an_error(self):
        self.assertEqual(gee.project_ids_from_dotenv(self.root), [])

    def test_empty_value_is_dropped(self):
        self._write("EARTHENGINE_PROJECT=\n")
        self.assertEqual(gee.project_ids_from_dotenv(self.root), [])

    def test_it_participates_in_candidate_resolution(self):
        """`.env` must be wired into resolution, not merely parseable.

        `project_ids_from_dotenv` defaults to the REPO root, so writing into a
        temp dir cannot reach `candidate_project_ids`. Patch the reader to check
        the wiring, which is the part that can actually break.
        """
        for var in gee.PROJECT_ENV_VARS:
            os.environ.pop(var, None)
        self.addCleanup(
            lambda: [os.environ.pop(v, None) for v in gee.PROJECT_ENV_VARS]
        )
        with mock.patch.object(gee, "project_ids_from_dotenv",
                               return_value=["dotenv-proj"]):
            self.assertIn("dotenv-proj", gee.candidate_project_ids())


class FailureDescriptionTests(unittest.TestCase):
    def test_missing_project_is_described_as_a_project_problem(self):
        text = gee.describe_auth_failure(
            RuntimeError("ee.Initialize: no project found. Call with project="))
        self.assertIn("CLOUD PROJECT id", text)
        self.assertIn("not necessarily", text)

    def test_disabled_api_is_recognised(self):
        text = gee.describe_auth_failure(
            RuntimeError("Earth Engine Api has not been used in project 123"))
        self.assertIn("not enabled", text)

    def test_unregistered_project_is_recognised(self):
        text = gee.describe_auth_failure(
            RuntimeError("Project 123 is not registered with Earth Engine"))
        self.assertIn("not registered", text)

    def test_an_unknown_failure_still_carries_remediation(self):
        text = gee.describe_auth_failure(RuntimeError("something else entirely"))
        # Falls back to a generic head, but must never lose the steps.
        self.assertIn("init failed", text)
        self.assertIn("ee.Authenticate()", text)
        self.assertIn("something else entirely", text)

    def test_every_description_carries_the_remediation_steps(self):
        for msg in ("no project found", "api has not been used",
                    "not registered", "who knows"):
            self.assertIn("ee.Authenticate()",
                          gee.describe_auth_failure(RuntimeError(msg)))


class NBandsTests(unittest.TestCase):
    def test_nir_and_swir_are_not_swapped(self):
        """A swapped NBR measures nothing and returns plausible numbers."""
        swir, nir = gee.NBR_BANDS["sentinel2"]
        self.assertEqual(swir, "B12")
        self.assertEqual(nir, "B8A")
        swir, nir = gee.NBR_BANDS["landsat"]
        self.assertEqual(swir, "SR_B7")
        self.assertEqual(nir, "SR_B5")

    def test_sentinel2_is_surface_reflectance_not_toa(self):
        """COPERNICUS/S2_HARMONIZED is Level-1C TOA. NBR from TOA carries the
        atmospheric path radiance, which is the very signal NBR must isolate."""
        self.assertEqual(gee.S2_SR, "COPERNICUS/S2_SR_HARMONIZED")

    def test_worldcover_band_is_map_not_cover(self):
        """The catalog band is 'Map'. Asking for 'cover' would have reported
        the single most useful band in the exercise as missing."""
        self.assertEqual(gee.WORLDCOVER_BAND, "Map")


class ChunkedSamplingTests(unittest.TestCase):
    """`sampleRegions` has a HARD 10 MiB request-payload limit.

    A 120,000-point sample failed with
    `Request payload size exceeds the limit: 10485760 bytes` -- rejected on the
    request, before any compute is billed, which reads as a data failure when it
    is a transport limit. These tests pin the chunking, the halving retry, and
    the fact that a failed chunk does not lose the others.
    """

    class _Feat:
        # ee.Feature(geometry, properties) -- both arguments, in that order.
        def __init__(self, geometry=None, props=None):
            self.geometry = geometry
            self.props = props

    class _FC:
        def __init__(self, feats):
            self.feats = list(feats)

    class _Image:
        """Fails on any call whose feature count exceeds `max_ok`."""

        def __init__(self, max_ok, values=None, error=None):
            self.max_ok = max_ok
            self.values = values or {}
            self.error = error
            self.sizes = []

        def select(self, band):
            self.band = band
            return self

        def sampleRegions(self, collection, properties, scale, geometries):
            outer = self

            class _S:
                def getInfo(self):
                    n = len(collection.feats)
                    outer.sizes.append(n)
                    if outer.error:
                        raise outer.error
                    if n > outer.max_ok:
                        raise RuntimeError(
                            "Request payload size exceeds the limit: "
                            "10485760 bytes.")
                    return {"features": [
                        {"properties": dict(f.props,
                                            Map=outer.values.get(
                                                f.props["row"], 40))}
                        for f in collection.feats
                    ]}

            return _S()

    @classmethod
    def _ee(cls):
        class _Geometry:
            @staticmethod
            def Point(c):
                return tuple(c)

        class _EE:
            Feature = cls._Feat
            FeatureCollection = cls._FC
            Geometry = _Geometry

        return _EE

    @staticmethod
    def _records(n):
        return [{"row": i, "lon": 70.0 + i * 0.001, "lat": 20.0 + i * 0.001}
                for i in range(n)]

    def _run(self, n, max_ok=10**9, chunk_size=5_000, values=None, error=None,
             calls=None):
        img = self._Image(max_ok, values, error)
        with mock.patch.dict(sys.modules, {"ee": self._ee()}, create=True):
            got = gee.sample_points_chunked(
                img, self._records(n), id_key="row", chunk_size=chunk_size,
                band="Map", printer=lambda _m: None, progress_every=0)
        if calls is not None:
            calls.append(len(img.sizes))
        return got, img

    def test_a_large_sample_is_split_into_several_calls(self):
        got, img = self._run(n=1_000, max_ok=10**9, chunk_size=200)
        self.assertGreater(len(img.sizes), 1,
                           "one giant call is exactly what just failed")
        self.assertEqual(len(got), 1_000)

    def test_it_converges_below_the_limit(self):
        """Once it discovers the limit it must strictly reduce, and lose nothing.

        Two things this does NOT assert, both of which I got wrong first:
          - "no call ever exceeds the limit". The FIRST call is supposed to
            overshoot -- that is how the limit is discovered.
          - "every call after the first fits". Splitting 1,000 queues TWO 500s,
            and both are attempted before either halves, so 500 legitimately
            appears after a 250.
        """
        got, img = self._run(n=1_000, max_ok=300, chunk_size=5_000)
        self.assertEqual(img.sizes[0], 1_000, "first call is one big one")
        # Never re-sends the original size after failing on it.
        self.assertTrue(all(n < 1_000 for n in img.sizes[1:]),
                        f"re-sent the failing size: {img.sizes}")
        # And it terminated rather than looping.
        self.assertLess(len(img.sizes), 40, f"did not converge: {img.sizes}")
        # Most importantly: nothing lost.
        self.assertEqual(len(got), 1_000, "points were dropped")

    def test_a_chunk_that_is_too_big_is_retried_smaller(self):
        """Chunk size is a guess about payload size, so it must self-correct."""
        got, img = self._run(n=600, max_ok=300, chunk_size=600)
        self.assertEqual(len(got), 600)
        self.assertTrue(any(n == 300 for n in img.sizes))

    def test_dropped_points_are_reported_not_silent(self):
        """A dropped chunk must be LOUD: downstream it looks like unmapped,
        which is indistinguishable from a genuine no-data answer."""
        msgs = []
        EE = self._ee()
        img = self._Image(max_ok=10**9, error=RuntimeError("Asset not found"))
        with mock.patch.dict(sys.modules, {"ee": EE}, create=True):
            got = gee.sample_points_chunked(
                img, self._records(100), chunk_size=50, printer=msgs.append,
                progress_every=0)
        self.assertEqual(got, {})
        self.assertTrue(any("DROPPED" in m for m in msgs),
                        f"silent data loss: {msgs}")

    def test_every_record_is_sampled_exactly_once(self):
        got, _img = self._run(n=250, chunk_size=37)
        self.assertEqual(sorted(got), list(range(250)))

    def test_values_are_returned_keyed_by_id(self):
        got, _img = self._run(n=10, chunk_size=10, values={3: 50, 7: 10})
        self.assertEqual(got[3], 50)
        self.assertEqual(got[7], 10)

    def test_a_non_payload_failure_is_not_retried(self):
        """Halving only helps SIZE errors. A real error must not loop."""
        got, img = self._run(n=100, chunk_size=50,
                             error=RuntimeError("Asset not found"))
        self.assertEqual(got, {})
        self.assertEqual(len(img.sizes), 2, "two chunks, no retries")

    def test_a_failed_chunk_does_not_lose_the_others(self):
        class _Flaky(self._Image):
            def sampleRegions(self, collection, properties, scale, geometries):
                outer = self
                _first = not outer.sizes

                class _S:
                    def getInfo(self):
                        n = len(collection.feats)
                        outer.sizes.append(n)
                        if _first:
                            raise RuntimeError("Asset not found")
                        return super(type(outer), outer).sampleRegions(
                            collection, properties, scale,
                            geometries).getInfo()

                return _S()

        img = _Flaky(10**9)
        with mock.patch.dict(sys.modules, {"ee": self._ee()}, create=True):
            got = gee.sample_points_chunked(
                img, self._records(20), chunk_size=10, printer=lambda _m: None,
                progress_every=0)
        self.assertEqual(sorted(got), list(range(10, 20)),
                         "the surviving chunk must still be returned")

    def test_an_empty_input_returns_empty_without_touching_the_image(self):
        class _Never:
            def select(self, _b):
                raise AssertionError("must not touch the image")

        got = gee.sample_points_chunked(_Never(), [], printer=lambda _m: None)
        self.assertEqual(got, {})


class WorldcoverClassTests(unittest.TestCase):
    """The class table was wrong for every code at 70 and above.

    Code 80 was previously named 'herbaceous' and included in the VEGETATED
    set. It is PERMANENT WATER BODIES. That would have labelled open water as
    burnable substrate -- corrupting the very label this work exists to fix.
    """

    def test_high_codes_are_named_correctly(self):
        self.assertEqual(gee.WORLDCOVER_CLASSES[80], "permanent water bodies")
        self.assertEqual(gee.WORLDCOVER_CLASSES[70], "snow and ice")
        self.assertEqual(gee.WORLDCOVER_CLASSES[95], "mangroves")
        self.assertEqual(gee.WORLDCOVER_CLASSES[100], "moss and lichen")
        self.assertEqual(gee.WORLDCOVER_CLASSES[90], "herbaceous wetland")

    def test_water_is_not_vegetated(self):
        self.assertNotIn(80, gee.VEGETATED_CLASSES)
        self.assertIn(80, gee.NON_VEGETATED_CLASSES)

    def test_snow_is_not_vegetated(self):
        self.assertNotIn(70, gee.VEGETATED_CLASSES)
        self.assertIn(70, gee.NON_VEGETATED_CLASSES)

    def test_built_and_bare_are_not_vegetated(self):
        self.assertNotIn(50, gee.VEGETATED_CLASSES)
        self.assertNotIn(60, gee.VEGETATED_CLASSES)

    def test_the_four_main_canopy_and_crop_classes_are_vegetated(self):
        for c in (10, 20, 30, 40):
            self.assertIn(c, gee.VEGETATED_CLASSES, c)

    def test_every_class_is_classified_one_way_only(self):
        for c in gee.WORLDCOVER_CLASSES:
            in_veg = c in gee.VEGETATED_CLASSES
            in_non = c in gee.NON_VEGETATED_CLASSES
            self.assertFalse(in_veg and in_non,
                             f"class {c} is both vegetated and non-vegetated")

    def test_no_class_code_is_missing_from_both_sets(self):
        unclassified = sorted(
            set(gee.WORLDCOVER_CLASSES)
            - set(gee.VEGETATED_CLASSES)
            - set(gee.NON_VEGETATED_CLASSES))
        self.assertEqual(unclassified, [],
                         f"unclassified WorldCover codes: {unclassified}")


class MonthCoverageTests(unittest.TestCase):
    """Year-granular coverage is not enough: L8 L2 data starts 2013-03-18 and
    GEE's S1 GRD archive starts 2014-10-03. Months that pass the YEAR rule but
    are EMPTY in the catalog give median() a zero-band image, which crashed
    P7k run 1 ("must have at least 2 bands") and P7l run 1
    ("Invalid numInputs: 0")."""

    def test_nbr_2013_01_is_empty_even_though_2013_is_covered(self):
        self.assertTrue(gee.nbr_month_has_imagery(2013, 3))
        self.assertFalse(gee.nbr_month_has_imagery(2013, 2))
        self.assertFalse(gee.nbr_month_has_imagery(2013, 1))
        self.assertFalse(gee.nbr_month_has_imagery(2012, 12))

    def test_nbr_from_2013_04_onwards_is_covered(self):
        self.assertTrue(gee.nbr_month_has_imagery(2013, 4))
        self.assertTrue(gee.nbr_month_has_imagery(2024, 7))

    def test_sar_2014_months_before_the_archive_are_empty(self):
        self.assertTrue(gee.sar_month_has_imagery(2014, 10))
        self.assertTrue(gee.sar_month_has_imagery(2015, 1))
        self.assertFalse(gee.sar_month_has_imagery(2014, 9))
        self.assertFalse(gee.sar_month_has_imagery(2014, 1))

    def test_sar_composite_filters_to_iw_mode(self):
        """EW/SM scenes carry HH/HV, not VV/VH; without the IW filter the
        median mixes them and select(VV) dies (P7l run 2)."""
        import sys
        import types as _types

        calls = []

        class _Fl:
            @staticmethod
            def eq(a, b):
                calls.append(("filter_eq", a, b))
                return ("eq", a, b)

            @staticmethod
            def listContains(a, b):
                calls.append(("filter_listContains", a, b))
                return ("listContains", a, b)

        class _Sel:
            def __init__(self, bands):
                self._bands = list(bands)

            def median(self):
                calls.append(("median",))
                return self

            def rename(self, names):
                calls.append(("rename", list(names)))
                return self

        class _Coll:
            def __init__(self, _cid):
                pass

            def filterDate(self, a, b):
                calls.append(("filterDate", a, b))
                return self

            def filter(self, f):
                calls.append(("filter", f))
                return self

            def select(self, bands):
                calls.append(("select", list(bands)))
                return _Sel(bands)

        _stub = _types.SimpleNamespace(
            Filter=_Fl, ImageCollection=lambda cid: _Coll(cid))
        with mock.patch.dict(sys.modules, {"ee": _stub}):
            gee.sar_monthly_composite(2020, 5)
        self.assertIn(("filter_eq", "instrumentMode", "IW"), calls,
                      "sar_monthly_composite must filter instrumentMode==IW "
                      "or EW/HH-images break select(VV, VH)")
        self.assertIn(
            ("filter_listContains", "transmitterReceiverPolarisation", "VV"),
            calls,
            "must keep only scenes that carry VV (S1A's first weeks had "
            "HH/HV IW scenes)")
        self.assertIn(
            ("filter_listContains", "transmitterReceiverPolarisation", "VH"),
            calls,
            "must keep only scenes that carry VH")
        self.assertEqual([c for c in calls if c[0] == "select"][0][1],
                         ["VV", "VH"])

    def test_the_december_wraparound_of_the_pre_month(self):
        """P7k needs 2011-12 as the PRE month of 2012-01 events; it is empty."""
        self.assertFalse(gee.nbr_month_has_imagery(2011, 12))


class CoverageGapTests(unittest.TestCase):
    """Sentinel-2 L2A starts 2017, Landsat 8 starts 2013, the catalogue starts
    2012. A third of the gold sample has NO imagery from either."""

    def test_a_2012_event_has_no_landsat(self):
        gap = gee.imagery_gap([2012])
        self.assertIn(2012, gap["no_landsat"])
        self.assertIn(2012, gap["no_sentinel2"])

    def test_2015_has_landsat_but_not_sentinel2(self):
        gap = gee.imagery_gap([2015])
        self.assertEqual(gap["no_landsat"], [])
        self.assertIn(2015, gap["no_sentinel2"])

    def test_2020_has_both(self):
        gap = gee.imagery_gap([2020])
        self.assertEqual(gap["no_landsat"], [])
        self.assertEqual(gap["no_sentinel2"], [])

    def test_the_gold_samples_actual_first_years_are_flagged(self):
        """GOLD_0001 is 2012-11-08 -- taken from the real gold sample."""
        gap = gee.imagery_gap([2012, 2013, 2015, 2017, 2020, 2021, 2023])
        self.assertIn(2012, gap["no_landsat"])
        self.assertIn(2013, gap["no_sentinel2"])

    def test_landsat5_and_7_are_available_for_the_early_years(self):
        self.assertIn(gee.LANDSAT5_NBR, gee.LANDSAT_ALL)
        self.assertIn(gee.LANDSAT7_NBR, gee.LANDSAT_ALL)

    def test_dataset_constants_are_ee_ids(self):
        for const in (gee.WORLDCOVER, gee.MODIS_LANDCOVER, gee.S2_SR,
                      gee.LANDSAT_NBR):
            self.assertIsInstance(const, str)
            self.assertIn("/", const)


class _EEBase:
    """Stub dataset. Lives at module scope because a class body is not a
    closure -- referencing a sibling nested class from inside `_EE` raises
    NameError at class-definition time."""

    _BANDS = {"GOOD": ["a", "b", "c"], "PARTIAL": ["a"], "BROKEN": []}

    def __init__(self, image_id):
        self._id = image_id

    def bandNames(self):
        outer = self

        class _Names:
            def getInfo(_self):
                if outer._id == "BROKEN":
                    raise RuntimeError("asset not found")
                return outer._BANDS[outer._id]

        return _Names()

    def first(self):
        # A collection must be narrowed with `.first()` before band names are
        # meaningful. Without this the stub would let an
        # Image-vs-ImageCollection mistake pass unnoticed.
        return self


class _EEImage(_EEBase):
    pass


class _EEColl(_EEBase):
    pass


class _EEStub:
    Image = _EEImage
    ImageCollection = _EEColl


class VerifyDatasetsTests(unittest.TestCase):
    """The stub stands in for `ee` so band checking is exercised without it."""

    def test_every_branch_returns_the_same_keys(self):
        """Regression: the failure branch omitted `wanted`, and a caller reading
        it on the SUCCESS path died with KeyError: 'wanted'. One dataset problem
        produced two bugs -- the crash was in the reporting code."""
        out = self._run()
        self.assertEqual(len({frozenset(v) for v in out.values()}), 1,
                         f"key sets differ: "
                         f"{ {k: sorted(v) for k, v in out.items()} }")
        for v in out.values():
            self.assertIn("wanted", v)
            self.assertIn("error", v)

    def test_wanted_survives_a_load_failure(self):
        out = self._run()
        self.assertEqual(out["broken"]["wanted"], ["a"])

    def test_the_default_specs_use_verified_band_names(self):
        import sys

        with mock.patch.dict(sys.modules, {"ee": _EEStub}, create=True):
            out = gee.verify_datasets()
        self.assertEqual(out["worldcover"]["wanted"], ["Map"])
        self.assertEqual(out["modis_landcover"]["wanted"], ["LC_Type1"])
        self.assertEqual(out["s2_sr"]["wanted"], ["B12", "B8A"])

    def _run(self):
        import sys

        with mock.patch.dict(sys.modules, {"ee": _EEStub}, create=True):
            return gee.verify_datasets({
                "good": ("GOOD", ["a", "b"], "collection"),
                "partial": ("PARTIAL", ["a", "z"], "image"),
                "broken": ("BROKEN", ["a"], "image"),
            })

    def test_all_bands_present_is_ok(self):
        out = self._run()
        self.assertTrue(out["good"]["ok"])
        self.assertEqual(out["good"]["missing"], [])

    def test_a_missing_band_is_reported_by_name(self):
        out = self._run()
        self.assertFalse(out["partial"]["ok"])
        self.assertEqual(out["partial"]["missing"], ["z"])

    def test_a_dataset_that_fails_to_load_is_not_ok(self):
        out = self._run()
        self.assertFalse(out["broken"]["ok"])
        self.assertIn("error", out["broken"])


if __name__ == "__main__":
    unittest.main()