# ============================================================================
# gee_utils.py -- Earth Engine authentication that survives the current API
#
# WHY THIS MODULE EXISTS
#
# `ee.Initialize()` with no arguments now raises:
#
#     ee.Initialize: no project found. Call with project= or see
#     http://goo.gle/ee-auth.
#
# That message is misleading in two ways, and both cost a Colab run:
#
#   1. It reads like "you are not authenticated" when credentials may well be
#      present. Earth Engine simply cannot infer WHICH Cloud project to bill,
#      and reports that as a missing project.
#   2. The real requirement is a registered Google Cloud project with the
#      Earth Engine API enabled. `ee.Authenticate()` alone does not satisfy it.
#
# So authentication is two independent things:
#
#     credentials   -- `ee.Authenticate()`, one browser consent
#     project       -- a Cloud project id, passed as `project=`
#
# and you need both. This module resolves the second from several sources so
# the user is not asked to hardcode it in six different cells.
#
# THE PROJECT MUST BE REGISTERED WITH EARTH ENGINE
#
# A fresh Cloud project is not automatically visible to EE. Registering is
# either automatic on first `Initialize(project=...)` or done by hand at
# https://code.earthengine.google.com/ (the project appears in the dropdown).
# Until that happens, `Initialize` fails with a permissions error even though
# the credentials are perfect.
# ============================================================================

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: Environment variables checked, in order, before asking the user. Colab
#: secrets land here, which is how a notebook avoids a hardcoded project id.
PROJECT_ENV_VARS = ("EARTHENGINE_PROJECT", "GOOGLE_CLOUD_PROJECT",
                    "GOOGLE_CLOUD_PROJECT_ID", "GCP_PROJECT")

#: Fallback Cloud project, tried before anything else.
#:
#: Set this ONCE in your notebook, at the very top, and every cell finds it:
#:
#:     import gee_utils as gee
#:     gee.DEFAULT_PROJECT = "my-project-id"
#:
#: Why this exists: `authenticate()` resolves the project from the environment, a
#: local `.env`, a Colab secret, or `gcloud`. If none of those is configured --
#: which is the normal state of a fresh Colab runtime, since all four are
#: per-session or per-machine -- then it calls `Initialize()` with no project
#: and gets
#:
#:     ee.Initialize: no project found. Call with project=
#:
#: three times, and reports failure for a session that is in fact perfectly
#: usable. `ee.Initialize(project=...)` by hand always works, which is what makes
#: the message so confusing: the account is fine and the quota is fine; only the
#: billing project was never supplied.
#:
#: Deliberately None rather than a literal id. A project id names a BILLING
#: ACCOUNT, which is the same category as a credential: `.env` is git-ignored
#: for exactly this reason, and a literal here would ship with the repo.
DEFAULT_PROJECT: Optional[str] = None

#: Local, git-ignored. A GCP project id is an IDENTIFIER, not a credential --
#: it carries no access on its own -- but it still names a billing account and
#: belongs in the same category as `.env`. `.env` is already in `.gitignore`, so
#: reading it here means no new file needs ignoring and no value is ever
#: committed. A literal written into a tracked `.py` would be published with
#: the repo.
#: Bumped when the module's behaviour changes. P7e prints it so a stale copy
#: in Colab is visible on the first line of output rather than inferred from a
#: confusing downstream failure.
__version__ = "1.4.0"

DOTENV_NAMES = (".env",)
PROJECT_DOTENV_KEYS = ("EARTHENGINE_PROJECT", "GOOGLE_CLOUD_PROJECT",
                       "GCP_PROJECT")

#: Land cover. Static, so one sample per location suffices -- no date window.
#:
#: VERIFIED against the Earth Engine Data Catalog on 2026-10-04. ALL of these
#: are ImageCollections, INCLUDING WorldCover: `ee.Image("ESA/WorldCover/v200")`
#: raises `Asset ... is not an Image`. The catalog snippet is
#: `ee.ImageCollection("ESA/WorldCover/v200").first()`.
WORLDCOVER = "ESA/WorldCover/v200"
MODIS_LANDCOVER = "MODIS/061/MCD12Q1"

#: Level-2A SURFACE REFLECTANCE, not the Level-1C TOA collection.
#:
#:   COPERNICUS/S2_HARMONIZED     = Level-1C TOA, from 2015-06-27
#:   COPERNICUS/S2_SR_HARMONIZED  = Level-2A SR, from 2017-03-28
#:
#: Two DISTINCT collections, not a rename. NBR computed from TOA reflectance is
#: a methodological error: TOA carries an atmospheric path radiance that varies
#: between dates, which is exactly the signal NBR exists to measure.
S2_SR = "COPERNICUS/S2_SR_HARMONIZED"

#: COVERAGE GAP. Landsat 8 starts 2013-03-18 and Sentinel-2 starts 2015 (or
#: 2017 for L2A). The catalogue spans 2012-2026 and the gold set contains
#: events from 2012 and 2013 -- which NEITHER of those can see. Landsat 5 and 7
#: are required for the early years or a third of the sample has no imagery.
LANDSAT_NBR = "LANDSAT/LC08/C02/T1_L2"      # 2013-03-18 ->
LANDSAT5_NBR = "LANDSAT/LT05/C02/T1_L2"     # 1999-05-28 -> 2013
LANDSAT7_NBR = "LANDSAT/LE07/C02/T1_L2"     # 1999-05-28 ->
LANDSAT_ALL = (LANDSAT5_NBR, LANDSAT7_NBR, LANDSAT_NBR)
#: First year Sentinel-2 L2A surface reflectance exists.
S2_SR_FIRST_YEAR = 2017
#: First year Landsat 8 exists.
LANDSAT8_FIRST_YEAR = 2013

#: THE NBR TRIPLE, PINNED. `normalizedDifference` computes (first - second) /
#: (first + second), so the ORDER decides whether you measure vegetation or its
#: negative. Swapping them yields a plausible number in [-1, 1] that means
#: nothing -- worse than a crash.
#:
#: Verified band names: Sentinel-2 B12 = SWIR2, B8A = Red Edge (the 20 m band;
#: wide NIR is B8). Landsat SR_B7 = SWIR2, SR_B5 = NIR.
NBR_BANDS = {"sentinel2": ("B12", "B8A"), "landsat": ("SR_B7", "SR_B5")}

#: WorldCover's land cover band is called "Map". NOT "cover" -- which is what an
#: earlier version asked for, and which would have reported the single most
#: useful band in the whole exercise as missing.
WORLDCOVER_BAND = "Map"

#: WorldCover class codes, transcribed from the catalog class table.
#:
#: The five codes at 70 and above were previously mapped to entirely the wrong
#: names -- 80 is PERMANENT WATER BODIES, not herbaceous. Including it in the
#: "vegetated" set would have labelled open water as burnable substrate, which
#: would have corrupted the new label rather than fixing the old one.
WORLDCOVER_CLASSES = {
    10: "tree cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built-up",
    60: "bare / sparse vegetation",
    70: "snow and ice",
    80: "permanent water bodies",
    90: "herbaceous wetland",
    95: "mangroves",
    100: "moss and lichen",
}

#: Substrate that can actually carry a fire. Everything else is built, bare,
#: snow or water, and burning there is not a vegetation fire by definition.
VEGETATED_CLASSES = frozenset({10, 20, 30, 40, 90, 95, 100})
NON_VEGETATED_CLASSES = frozenset({50, 60, 70, 80})

#: Which datasets are ImageCollections rather than single Images.
#:
#: `ee.Image("COPERNICUS/S2_HARMONIZED")` does not raise. It silently builds a
#: composite Image that has no `filterDate`, so the failure surfaces much later
#: as `AttributeError: 'Image' object has no attribute 'filterDate'` -- a long
#: way from the mistake. Hence this is data, not a convention.
#:
#: VERIFIED: every dataset this project uses is a collection. `IMAGE_IDS` is
#: deliberately EMPTY. Do not add to it without checking the catalog snippet.
COLLECTION_IDS = frozenset({WORLDCOVER, MODIS_LANDCOVER, S2_SR, LANDSAT_NBR,
                            LANDSAT5_NBR, LANDSAT7_NBR})
IMAGE_IDS = frozenset()


class EEPermissionError(RuntimeError):
    """Initialised fine, but the project refuses data reads.

    SEPARATE FROM `EEAuthError` ON PURPOSE. `ee.Initialize(project=...)`
    succeeds the moment the credentials are accepted -- it does not check that
    the caller may USE the project. So a project owned by a different account,
    or one where the caller lacks `roles/serviceusage.serviceUsageConsumer`,
    produces a cheerful "initialised" followed by a 403 on every single data
    call. Treating that as success is what let four confusing load failures
    through before anyone looked at the real problem.
    """

    REMEDIATION = (
        "Earth Engine initialised but the project refuses data reads:\n"
        "    Caller does not have required permission to use project <ID>\n"
        "\n"
        "Two distinct causes, and they need different fixes:\n"
        "\n"
        "  (a) THE PROJECT BELONGS TO A DIFFERENT GOOGLE ACCOUNT.\n"
        "      Most likely after switching accounts. Fix: create a project\n"
        "      under the account you are now using --\n"
        "      https://console.cloud.google.com/ -> select YOUR account ->\n"
        "      Create project -> enable 'Earth Engine API' -> register at\n"
        "      https://code.earthengine.google.com/ -- then use THAT id.\n"
        "\n"
        "  (b) The project is on the right account but the role is missing.\n"
        "      Grant the caller roles/serviceusage.serviceUsageConsumer at\n"
        "      https://console.cloud.google.com/iam-admin/iam\n"
        "\n"
        "Quick check of which account is active and which projects it can see:\n"
        "    import ee\n"
        "    ee.Initialize()\n"
        "    print(ee.data.getAssetRoots())\n"
        "The 'projects/<ID>/assets' entries are the projects this account can\n"
        "actually use. If your intended project is absent from that list, it\n"
        "is not yours to use."
    )

    def __init__(self, project_id: Optional[str], cause: str):
        super().__init__(f"project {project_id!r} refused data reads\n\n"
                         f"{self.REMEDIATION}\n\noriginal: {cause[:400]}")
        self.project_id = project_id
        self.cause = cause


def verify_can_read(project_id: Optional[str] = None) -> None:
    """Prove the session can actually READ DATA, not merely initialise.

    `ee.Initialize` succeeding is not a working session. This makes one real
    round trip before reporting success.

    THE PROBE USES A CONSTANT, NOT AN ASSET. An earlier version probed with
    `ee.Image(WORLDCOVER)`, which produced
    `Earth Engine session works but the probe failed (Asset ... is not an
    Image)` -- conflating a perfectly healthy session with a misconfigured
    asset id, and burying the real fix under a reassuring-sounding message.
    A permission check must not depend on any dataset being correct.

    Raises `EEPermissionError` for the 403 family, so a permission problem is
    never reported as a session problem.
    """
    import ee

    try:
        # No asset, no dataset, no configuration. Pure session check.
        bands = ee.Image.constant(1).bandNames().getInfo()
        if not bands:
            raise EEPermissionError(project_id,
                                     "a constant image returned no bands")
        return
    except EEPermissionError:
        raise
    except Exception as exc:                        # noqa: BLE001
        text = f"{type(exc).__name__}: {exc}"
        if ("PERMISSION_DENIED" in str(exc)
                or "does not have required permission" in str(exc)
                or "403" in str(exc)):
            raise EEPermissionError(project_id, text) from None
        raise RuntimeError(
            "The Earth Engine session failed a basic round trip that uses NO "
            f"dataset ({text[:200]}). That points at the session itself, not "
            "at any asset id."
        ) from None


def imagery_gap(years) -> Dict[str, List[int]]:
    """Which requested years no satellite can actually see.

    Sentinel-2 L2A starts 2017 and Landsat 8 starts 2013, so with only those two
    configured, catalogue years before 2013 have NO imagery whatsoever. The
    gold set contains 2012 events, so this is not hypothetical: those rows
    cannot be labelled from imagery and must either be dropped or labelled from
    other evidence. Reported rather than discovered later as empty pixels.
    """
    out: Dict[str, List[int]] = {"no_landsat": [], "no_sentinel2": []}
    for y in sorted({int(v) for v in years if v is not None and
                     str(v).strip()}):
        if y < LANDSAT8_FIRST_YEAR:
            out["no_landsat"].append(y)
        if y < S2_SR_FIRST_YEAR:
            out["no_sentinel2"].append(y)
    return out


def projects_visible_to_current_account() -> List[str]:
    """Project ids the CURRENT credentials can actually use.

    This is the question to ask after an account change: the project you were
    using before may not belong to the account you are on now, and the asset
    roots are the authoritative answer rather than a guess.
    """
    import ee

    found: List[str] = []
    try:
        roots = ee.data.getAssetRoots() or []
        for r in roots:
            _id = r.get("id") if isinstance(r, dict) else str(r)
            if _id and _id.startswith("projects/"):
                parts = _id.split("/")
                if len(parts) > 1 and parts[1] not in found:
                    found.append(parts[1])
    except Exception:                               # noqa: BLE001
        pass
    return found


def sample_points_chunked(image, records, id_key: str = "row",
                          scale: int = 100, chunk_size: int = 5_000,
                          band: Optional[str] = None,
                          progress_every: int = 10,
                          printer=print):
    """`sampleRegions` in chunks, because a single call has a hard size limit.

    `Image.sampleRegions(collection=fc).getInfo()` serialises the ENTIRE feature
    collection into one request body, and Earth Engine rejects anything over
    **10 MiB** with

        EEException: Request payload size exceeds the limit: 10485760 bytes.

    At ~200 bytes per point that is roughly 50k points, so a 120k-point sample
    fails -- and it fails on the *request*, before any compute is billed, which
    makes it look like a data problem when it is a transport one.

    The default `chunk_size` of 5,000 leaves a very large margin, and any chunk
    that still trips the limit is retried at half size before giving up. Cost is
    driven by chunk COUNT, so too small wastes round trips and too large gets
    rejected; 5,000 keeps a 7.27M-event extraction under ~1,500 calls.

    Returns `{id: value}`.
    """
    records = list(records)
    total = len(records)
    out: Dict[object, object] = {}
    if total == 0:
        # Checked BEFORE `import ee`. An empty sample must not require the
        # module -- the caller may be probing before Earth Engine is installed,
        # and importing here raised ModuleNotFoundError for a call that had
        # nothing to do with Earth Engine.
        return out

    import ee

    if band:
        image = image.select(band)

    def _one(chunk):
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r["lon"]), float(r["lat"])]),
                       {id_key: r[id_key]})
            for r in chunk
        ])
        res = image.sampleRegions(collection=fc, properties=[id_key],
                                  scale=scale, geometries=False).getInfo()
        got = {}
        for f in res.get("features", []):
            p = f.get("properties", {})
            v = p.get(band) if band else _first_value_excluding(p, id_key)
            got[p.get(id_key)] = v
        return got

    # A QUEUE, not a fixed slice. An earlier version pre-sliced the chunk and
    # merely halved a `size` number on retry -- so the retry re-sent exactly the
    # same oversized feature collection and failed identically, forever. The
    # payload limit is about how many features are in the request, so the only
    # cure is to actually send fewer.
    pending = [records[i:i + chunk_size]
               for i in range(0, total, chunk_size)]
    done = 0
    n_calls = 0
    dropped = 0
    while pending:
        chunk = pending.pop(0)
        try:
            out.update(_one(chunk))
            done += len(chunk)
            n_calls += 1
            if progress_every and n_calls % progress_every == 0 and pending:
                printer(f"    {done:,} / {total:,} points "
                        f"({n_calls} calls)")
        except Exception as exc:                    # noqa: BLE001
            text = str(exc)
            size_problem = ("payload size" in text
                            or "exceeds the limit" in text
                            or "request too large" in text.lower()
                            or "entity too large" in text.lower())
            if size_problem and len(chunk) > MIN_CHUNK:
                _half = max(1, len(chunk) // 2)
                pending.insert(0, chunk[_half:])
                pending.insert(0, chunk[:_half])
                printer(f"    payload limit at {len(chunk):,} points; "
                        f"splitting")
                continue
            printer(f"    chunk of {len(chunk):,} FAILED: "
                    f"{type(exc).__name__}: {text[:160]}")
            done += len(chunk)
            dropped += len(chunk)

    if dropped:
        # NEVER silent. A dropped chunk means those points simply do not appear
        # downstream, where they would be indistinguishable from "no land cover
        # there" -- the exact silent-wrong-answer failure this project keeps
        # hitting.
        printer(f"  WARNING: {dropped:,} of {total:,} points were DROPPED "
                f"({100.0 * dropped / total:.2f}%). They will look like "
                f"unmapped, not like missing.")

    return out


#: Never split below this. Below a few hundred points per request the round
#: trips cost more than the payload they save.
MIN_CHUNK = 250


def _first_value_excluding(props, exclude_key):
    """The single non-key property value from a sampleRegions result."""
    for k, v in props.items():
        if k != exclude_key:
            return v
    return None


def dataset_kind(image_id: str) -> str:
    """`"collection"` or `"image"`, so callers build the right ee object.

    Unknown ids default to `"collection"`, which is the safer default: a
    collection supports `.filterDate()` and an image does not, so guessing
    "image" fails on every dated query.
    """
    return "image" if image_id in IMAGE_IDS else "collection"


def ee_dataset(image_id: str):
    """Build the correct `ee` object for a dataset id.

    The single place that decides. Every cell should call this rather than
    `ee.Image(...)` directly, because the distinction is invisible in the
    constructor and only shows up as a missing method later.
    """
    import ee

    if dataset_kind(image_id) == "image":
        return ee.Image(image_id)
    return ee.ImageCollection(image_id)


def worldcover_image():
    """The WorldCover land cover band as a single Image.

    `ee.Image("ESA/WorldCover/v200")` raises
    `Asset 'ESA/WorldCover/v200' is not an Image`. The catalog snippet is
    `ee.ImageCollection("ESA/WorldCover/v200").first()`, and the band is
    called "Map".
    """
    import ee

    return ee.ImageCollection(WORLDCOVER).first().select(WORLDCOVER_BAND)


def landsat_for_year(year: int):
    """`(ImageCollection, asset_id)` covering `year`.

    Sentinel-2 L2A starts 2017 and Landsat 8 starts 2013, so neither covers a
    2012 event. Landsat 5 and 7 are selected for those years rather than
    silently returning nothing.
    """
    import ee

    chosen = LANDSAT5_NBR if int(year) < LANDSAT8_FIRST_YEAR else LANDSAT_NBR
    return ee.ImageCollection(chosen), chosen


def _month_bounds(year: int, month: int) -> Tuple[str, str]:
    """`(start, end)` as ISO dates bounding one calendar month, end exclusive.

    End is the FIRST day of the next month, not the last day of this one: an
    inclusive end date silently pulls the next month's first scenes into the
    composite, so a "pre-fire" month built that way contains post-fire imagery
    and the change it is supposed to measure is partly erased.
    """
    year = int(year)
    month = int(month)
    if not 1 <= month <= 12:
        raise ValueError(f"month must be 1..12, got {month!r}")
    ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
    return (f"{year:04d}-{month:02d}-01", f"{ny:04d}-{nm:02d}-01")


def month_before(year: int, month: int, lag: int = 1) -> Tuple[int, int]:
    """`(year, month)` `lag` months before `(year, month)`, year-aware.

    P7k walks backwards from an event's own month to find its pre-fire month.
    Doing that with `month - lag` alone crosses into month 0 for January, and
    then `_month_bounds` either raises or -- worse -- silently clamps, putting
    the pre-fire composite in the wrong year.
    """
    year = int(year)
    month = int(month)
    if lag < 0:
        raise ValueError(f"lag must be >= 0, got {lag!r}")
    total = (year * 12 + (month - 1)) - int(lag)
    return total // 12, total % 12 + 1


def nbr_monthly_composite(year: int, month: int, collection_id: Optional[str] = None):
    """A cloud-robust monthly NBR composite, as a single-band Image.

    Returns an image with exactly one band named `NBR`, so a caller can
    `sampleRegions` it and read one value per point. NBR is
    `(NIR - SWIR) / (NIR + SWIR)`; `NBR_BANDS` holds `(swir, nir)` in that order,
    so `normalizedDifference(nir, swir)` is the correct call -- passing the tuple
    straight through computes `-NBR`, which is still in [-1, 1] and still looks
    like a reflectance index.

    The composite is a MEDIAN, not a mean: a mean over a month is dragged by
    cloud and by residual cloud shadow, and a burn scar's NBR drop is the
    quantity of interest. Median needs an `ee.Reducer`, which cannot be
    serialised out of this module, so the reducer is built here rather than in
    the caller.

    `collection_id` overrides the sensor chosen by `landsat_for_year`, which is
    what lets the rehearsal drive this with a stub. Prefer the default.
    """
    import ee

    start, end = _month_bounds(year, month)
    asset = collection_id or landsat_for_year(year)[1]
    swir, nir = NBR_BANDS["landsat"]

    # Cloud mask BEFORE the median. `qa_pixel` bit 3 is cloud, bit 4 is cloud
    # shadow; masking only bit 3 leaves the shadow, which is dark in both NIR
    # and SWIR and so biases NBR downward exactly like a burn scar.
    #
    # NO scene-level CLOUD_COVER filter. Run 1 crashed mid-catalogue for a
    # different reason (empty months), but a scene filter carries the same
    # failure mode in Indian monsoon: July/August tiles routinely exceed 80%
    # scene cloud, an all-filtered month is an empty collection, and median()
    # of empty is a zero-band image that crashes sampleRegions. The QA_PIXEL
    # mask below does the real work per pixel, and a median over the remaining
    # clear pixels is exactly what a monthly composite should be.
    col = (ee.ImageCollection(asset)
           .filterDate(start, end)
           .map(_mask_landsat_clouds))
    # Zero-argument `normalizedDifference()` computes (band1 - band2) / (band1 +
    # band2) over the image's FIRST TWO bands, in that order. So the ORDER is
    # carried entirely by `select([nir, swir])` above, and passing both band
    # names here is a TypeError -- the Python API accepts at most one band
    # argument, unlike the JavaScript overload. Getting this wrong in either
    # direction is silent: a swapped pair yields -NBR, still in [-1, 1], still
    # looking like a reflectance index.
    return (col.select([nir, swir])
            .median()
            .normalizedDifference()
            .rename("NBR"))


def _mask_landsat_clouds(image):
    """Drop cloud and cloud-shadow pixels; unmasked pixels become NaN downstream.

    LANDSAT/C02/T1_L2 bit-3 = cloud, bit-4 = cloud shadow. Both must go: shadow
    is dark in NIR and SWIR alike, so leaving it in depresses NBR and is
    indistinguishable from vegetation loss.
    """
    qa = image.select("QA_PIXEL")
    return image.updateMask(qa.bitwiseAnd(int("1000", 2)).eq(0)
                            .And(qa.bitwiseAnd(int("10000", 2)).eq(0)))


#: Sentinel-1 GRD. Collection starts 2014-05-03 (Sentinel-1A) with S1B adding
#: coverage from 2016-07; S1A alone failed from 2015 to 2016, and the 12-day
#: revisit means most months have several usable looks.
S1_GRD = "COPERNICUS/S1_GRD"
S1_FIRST_YEAR = 2014
#: The two polarisations every GRD product carries. VV and VH are the only
#: bands available before Sentinel-1C's added HHV, so HHV is deliberately not
#: claimed here.
S1_BANDS = ("VV", "VH")


def sar_imagery_gap(years) -> Dict[str, List[int]]:
    """Which requested years Sentinel-1 GRD cannot see.

    Separate from `imagery_gap`, which answers the Landsat/Sentinel-2 question.
    S1 starts 2014, LATER than Landsat 8's 2013, so the gap is not a prefix of
    the optical gap and cannot be derived from it -- a cell that skipped years
    using the optical gap would send 2013 events to Sentinel-1, which has no
    data there, and store the empty sample as a real measurement.
    """
    out: Dict[str, List[int]] = {"no_sentinel1": []}
    for y in sorted({int(v) for v in years if v is not None and str(v).strip()}):
        if y < S1_FIRST_YEAR:
            out["no_sentinel1"].append(y)
    return out


#: First month with REAL catalog coverage. LANDSAT8_FIRST_YEAR = 2013 is not
#: enough: L8 L2 data starts 2013-03-18, so 2013-01 and 2013-02 pass the year
#: rule and are EMPTY in the catalog -- median() of an empty ImageCollection
#: is a ZERO-BAND image, and sampling it crashes with
#: "Image.normalizedDifference: Input image must have at least 2 bands".
#: Month granularity is therefore not an optimisation; it is the difference
#: between a run and a crash.
LANDSAT8_FIRST_DATE = "2013-03-18"
#: GEE's COPERNICUS/S1_GRD archive starts 2014-10-03. The ledger's era start
#: (2014-05-03) is first acquisition, which GEE does not carry; the same
#: empty-composite crash applies to 2014-01..09 if only the year is checked.
S1_GRD_FIRST_DATE = "2014-10-03"


def _month_at_least(year: int, month: int, first_date: str) -> bool:
    """`(y, m)` is at the month given by an ISO `first_date` or later."""
    first = (int(first_date[:4]), int(first_date[5:7]))
    return (int(year), int(month)) >= first


def nbr_month_has_imagery(year: int, month: int) -> bool:
    """True if a Landsat monthly NBR composite can actually be non-empty."""
    return _month_at_least(year, month, LANDSAT8_FIRST_DATE)


def sar_month_has_imagery(year: int, month: int) -> bool:
    """True if a Sentinel-1 GRD monthly composite can actually be non-empty."""
    return _month_at_least(year, month, S1_GRD_FIRST_DATE)


def sar_monthly_composite(year: int, month: int, collection_id: str = S1_GRD):
    """A monthly VV/VH composite as a two-band Image (`VV`, `VH`).

    Backscatter is already a calibrated physical quantity in dB, so unlike NBR
    there is no index to compute -- a median over the month IS the feature.

    Only IW (Interferometric Wide swath) images are used, AND only ones that
    actually carry VV/VH. Two distinct failure modes taught this:

    - **EW/SM swath scenes** carry HH/HV, not VV/VH (`S1A_EW_GRDH...` bands
      `[HH, HV, angle]`). Filtering `instrumentMode == "IW"` removes those.
    - **Sentinel-1A's first weeks in orbit (Oct-Dec 2014)** acquired some IW
      scenes in HH/HV too (`S1A_IW_GRDH_1SDH_20141006...` bands
      `[HH, HV, angle]`), so mode alone is not enough. The catalog's
      `transmitterReceiverPolarisation` metadata is the reliable test, hence
      the two `listContains(..., "VV") / "VH"` filters. Without them the
      median mixes HH-only scenes and `select(["VV","VH"])` dies with
      "Band pattern 'VV' did not match any bands. Available bands: [HH, HV,
      angle]" -- P7l run 3's exact crash.

    Only GRD is used. SLC would allow interferometry (and hence coherence), but
    it is a different asset with different orbit geometry and far narrower
    coverage; a GRD/VV number and an SLC/VV number are not comparable, so the
    two must never be mixed within one feature column.
    """
    import ee

    start, end = _month_bounds(year, month)
    col = (ee.ImageCollection(collection_id)
           .filterDate(start, end)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains(
               "transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains(
               "transmitterReceiverPolarisation", "VH")))
    return col.select(list(S1_BANDS)).median().rename(list(S1_BANDS))


class EEAuthError(RuntimeError):
    """Raised when Earth Engine cannot be initialised.

    Carries the remediation text, because the bare `ee.Initialize` message does
    not tell the user what to actually do.
    """

    REMEDIATION = (
        "Earth Engine needs BOTH credentials and a Cloud project.\n"
        "  1. In a cell of its own (not inside another cell):\n"
        "         import ee; ee.Authenticate()\n"
        "     Accept the consent screen that opens.\n"
        "  2. You need a Google Cloud project with the Earth Engine API\n"
        "     enabled. Create one at https://console.cloud.google.com/\n"
        "     -> create project -> APIs & Services -> Library ->\n"
        "     'Earth Engine API' -> Enable.\n"
        "  3. Register the project with Earth Engine, either implicitly by\n"
        "     calling ee.Initialize(project='<ID>') or at\n"
        "     https://code.earthengine.google.com/ (pick it in the\n"
        "     project dropdown).\n"
        "  4. Then pass it in:  ee.Initialize(project='<ID>')\n"
        "     or export EARTHENGINE_PROJECT=<ID> and let gee_utils find it."
    )

    def __init__(self, cause: str):
        super().__init__(f"{cause}\n\n{self.REMEDIATION}")
        self.cause = cause


def project_ids_from_dotenv(root: Optional[Path] = None) -> List[str]:
    """Read a project id out of a local `.env`.

    Deliberately a tiny parser rather than a `dotenv` dependency: the file only
    ever needs `KEY=VALUE`, and a project this repo does not otherwise use is
    not worth a new dependency. Handles `export KEY=VALUE`, quotes, `#`
    comments, and CRLF line endings -- the last of which appears whenever the
    file has been edited on Windows.
    """
    root = Path(root or Path(__file__).resolve().parent)
    found: List[str] = []

    def _add(value: Optional[str]) -> None:
        # Order matters: strip whitespace, then the quotes, then whitespace
        # again. Unquoting first leaves the inner padding behind, because after
        # the quotes go the ends are ordinary characters and strip() stops.
        v = (value or "").strip().strip("'\"").strip()
        if v and v not in found:
            found.append(v)

    for name in DOTENV_NAMES:
        path = root / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.split(" #")[0] if " #" in value else value
            if key in PROJECT_DOTENV_KEYS:
                _add(value)
    return found


def candidate_project_ids() -> List[str]:
    """Every project id we can see, most trustworthy first."""
    found: List[str] = []

    def _add(value: Optional[str]) -> None:
        v = (value or "").strip()
        if v and v not in found:
            found.append(v)

    for var in PROJECT_ENV_VARS:
        _add(os.environ.get(var))

    # The in-notebook override. First, because it is the only one of these that
    # survives neither a Drive copy nor a stale `.env` confusing the search.
    _add(DEFAULT_PROJECT)

    # A local `.env`, which is already git-ignored. This is the mechanism that
    # lets the project id be used every session without ever being committed.
    for value in project_ids_from_dotenv():
        _add(value)

    # Colab secrets, if the notebook stored one under a conventional name.
    try:                                            # pragma: no cover
        from google.colab import userdata            # type: ignore

        for name in ("EARTHENGINE_PROJECT", "GOOGLE_CLOUD_PROJECT",
                     "gcp_project"):
            try:
                _add(userdata.get(name))
            except Exception:                       # noqa: BLE001
                continue
    except ImportError:
        pass

    # gcloud, when it happens to be installed and configured.
    if shutil.which("gcloud"):                       # pragma: no cover
        try:
            out = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True, text=True, timeout=15,
            )
            _add(out.stdout.strip())
        except Exception:                           # noqa: BLE001
            pass

    return found


def projects_from_ee(credentials_only: bool = True) -> List[str]:
    """Project ids that Earth Engine can see for the current credentials.

    The asset root is `projects/<ID>/assets`, so the id is recoverable from the
    authenticated session itself. This is the fallback when nothing was
    configured, and it is how a user who has registered exactly one project
    avoids being asked for the id.
    """
    import ee

    ids: List[str] = []
    try:
        for root in ee.data.getAssetRoots() or []:
            _ = root
    except Exception:                               # noqa: BLE001
        # Older API shape; fall through to the filesystem view.
        pass
    try:
        import ee.data

        if hasattr(ee.data, "_get_projects"):
            for name in ee.data._get_projects() or []:    # type: ignore[attr-defined]
                if name and name not in ids:
                    ids.append(str(name))
    except Exception:                               # noqa: BLE001
        pass
    return ids


def authenticate(project_id: Optional[str] = None, *,
                 allow_interactive: bool = True,
                 ) -> Tuple[bool, Optional[str]]:
    """Get Earth Engine into a usable state. Returns `(ok, project_id)`.

    Tries, in order:
      1. `Initialize(project=<explicit>)`
      2. `Initialize()` with a project found from the environment
      3. `Initialize()` bare
      4. `Authenticate()` then retry the above

    Raises `EEAuthError` with remediation text if none work. Never calls
    `SystemExit`, because that triggers an IPython/`inspect` crash on Python
    3.13 that buries the real message under three tracebacks.
    """
    import ee

    attempts: List[str] = []
    order: List[Optional[str]] = [project_id]
    order.extend(candidate_project_ids())
    # An explicit None first would make every call "try bare, then try
    # configured", and bare always fails the same way -- so the project-bearing
    # ids come first when any exist.
    order = [p for p in order if p] + [None]

    def _try_init(pid: Optional[str]) -> None:
        if pid:
            ee.Initialize(project=pid)
        else:
            ee.Initialize()

    # First, credentials may already be cached.
    for pid in order:
        try:
            _try_init(pid)
            return True, pid
        except Exception as exc:                    # noqa: BLE001
            attempts.append(f"Initialize(project={pid!r}): "
                            f"{type(exc).__name__}: {str(exc)[:160]}")

    if not allow_interactive:
        raise EEAuthError("no project id could be resolved. Tried:\n  "
                          + "\n  ".join(attempts))

    # Now consent, in the caller's own session.
    try:
        ee.Authenticate()
    except Exception as exc:                        # noqa: BLE001
        attempts.append(f"Authenticate(): {type(exc).__name__}: {str(exc)[:160]}")
        raise EEAuthError(
            "ee.Authenticate() itself failed.\n  " + "\n  ".join(attempts)
        ) from None

    for pid in order + [None]:
        try:
            _try_init(pid)
            return True, pid
        except Exception as exc:                    # noqa: BLE001
            attempts.append(f"after Authenticate, Initialize(project={pid!r}): "
                            f"{type(exc).__name__}: {str(exc)[:160]}")

    # Last resort: ask Earth Engine which projects these credentials can see.
    for pid in projects_from_ee():
        try:
            _try_init(pid)
            return True, pid
        except Exception as exc:                    # noqa: BLE001
            attempts.append(f"Initialize(project={pid!r}) [from EE]: "
                            f"{type(exc).__name__}: {str(exc)[:160]}")

    raise EEAuthError(
        "Earth Engine could not be initialised. Attempts:\n  "
        + "\n  ".join(attempts)
        + "\n\nNone of the project-id sources yielded a project. Set ONE of "
          "these, at the top of the notebook, before any cell that calls "
          "authenticate():\n"
          "    import gee_utils as gee; gee.DEFAULT_PROJECT = '<id>'\n"
          "    %env EARTHENGINE_PROJECT=<id>\n"
          "Find the id at https://code.earthengine.google.com/ (the project "
          "dropdown). A bare `ee.Initialize(project='<id>')` always works, so "
          "if that succeeds for you then credentials and quota are fine and "
          "only the billing project was missing.")


def describe_auth_failure(exc: BaseException) -> str:
    """Turn an `ee` exception into something actionable."""
    text = str(exc)
    if "no project found" in text:
        head = ("Earth Engine is missing a CLOUD PROJECT id, not necessarily "
                "credentials. `ee.Initialize()` cannot infer which project to "
                "bill and says so confusingly.")
        tail = ("\n\nIf you have a project, tell gee_utils about it by adding "
                "one line to a local `.env` (already git-ignored):\n"
                "    EARTHENGINE_PROJECT=<your-project-id>\n"
                "or set it in the notebook:\n"
                "    EARTHENGINE_PROJECT = '<your-project-id>'")
    elif "Earth Engine Api has not been used" in text or "API has not been" in text:
        head = ("The Earth Engine API is not enabled on that Cloud project. "
                "Enable it in APIs & Services -> Library.")
        tail = ""
    elif "not registered" in text.lower() or "Earth Engine account" in text:
        head = ("That Cloud project exists but is not registered with Earth "
                "Engine. Register it at https://code.earthengine.google.com/.")
        tail = ""
    else:
        head = f"Earth Engine init failed ({type(exc).__name__})."
        tail = ""
    return f"{head}\n\n{EEAuthError.REMEDIATION}{tail}\n\noriginal: {text[:400]}"


def verify_datasets(specs: Optional[dict] = None) -> dict:
    """Load each dataset and report whether its wanted bands exist.

    Band presence is necessary but NOT sufficient: a dataset can list its bands
    and still return zeros over the AOI. That is how `sentinel5p` (444 B) and
    `era5_land` (120 KB/month) survived on Drive. Values must be sampled
    separately -- see `sample_points`.

    Accepts either `{name: (id, bands)}` or `{name: (id, bands, kind)}`.
    """
    import ee

    if specs is None:
        specs = {
            # Band names verified against the catalog: WorldCover's land cover
            # band is "Map", not "cover".
            "worldcover": (WORLDCOVER, [WORLDCOVER_BAND]),
            "modis_landcover": (MODIS_LANDCOVER, ["LC_Type1"]),
            "s2_sr": (S2_SR, list(NBR_BANDS["sentinel2"])),
            "landsat_nbr": (LANDSAT_NBR, list(NBR_BANDS["landsat"])),
        }

    out: dict = {}
    for name, spec in specs.items():
        image_id = spec[0]
        wanted = list(spec[1])
        kind = spec[2] if len(spec) > 2 else dataset_kind(image_id)
        try:
            # Build via the shared helper so the collection/image distinction
            # cannot be re-introduced per call site.
            obj = ee.Image(image_id) if kind == "image" \
                else ee.ImageCollection(image_id)
            bands = obj.first().bandNames().getInfo() if kind == "collection" \
                else obj.bandNames().getInfo()
            missing = [b for b in wanted if b not in bands]
            # Every branch returns the SAME key set. The failure branch used to
            # omit "wanted", and a caller reading it on the success path died
            # with `KeyError: 'wanted'` -- a crash in the reporting code
            # triggered by a dataset problem, i.e. two bugs for the price of one.
            out[name] = {"image_id": image_id, "kind": kind, "wanted": wanted,
                         "ok": not missing, "missing": missing,
                         "n_bands": len(bands), "error": None}
        except Exception as exc:                    # noqa: BLE001
            out[name] = {"image_id": image_id, "kind": kind, "wanted": wanted,
                         "ok": False, "missing": wanted, "n_bands": 0,
                         "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    return out


#: Back-compat alias. `ee_dataset` returns the correctly-typed object.
def dataset(image_id: str):
    """Deprecated alias for `ee_dataset`."""
    return ee_dataset(image_id)
def era5_monthly_weather(year: int, month: int,
                         collection_id: str = "ECMWF/ERA5_LAND/DAILY_AGGR"):
    """ERA5-Land monthly MEAN of the weather/fuel fields P7n needs.

    ECMWF/ERA5_LAND/DAILY_AGGR carries daily aggregates; a monthly MEAN is
    the feature. Bands kept: temperature_2m, dewpoint_temperature_2m,
    u/v_component_of_wind_10m, total_precipitation_sum (NOTE the _sum:
    ERA5-Land DAILY_AGGR names the daily aggregate total_precipitation_SUM,
    not total_precipitation -- a wrong name empties the whole composite
    silently),
    volumetric_soil_water_layer_1. RH / VPD / wind speed / FWI are derived
    LOCALLY in the cell, never on the server.
    """
    import ee

    start, end = _month_bounds(year, month)
    bands = ["temperature_2m", "dewpoint_temperature_2m",
             "u_component_of_wind_10m", "v_component_of_wind_10m",
             "total_precipitation_sum", "volumetric_soil_water_layer_1"]
    return (ee.ImageCollection(collection_id)
            .filterDate(start, end).select(bands).mean())


def s5p_monthly_median(year: int, month: int, collection_id: str,
                        band: str):
    """Sentinel-5P monthly MEDIAN of one TROPOMI trace-gas band.

    Returns `None` when the collection is empty for that month, NOT a 0-band
    image. Sampling a 0-band image raises "Band pattern 'X' was applied to an
    Image with no bands", and because `sample_points_chunked` catches the
    exception per-chunk the points are silently dropped -- which is how P7o
    run 1 lost 100% of its points across the whole run while the first 45
    chunks reported success. Returning `None` lets callers short-circuit
    without an extra `bandNames().getInfo()` round trip per call.

    The size check is one round trip per (month, band), not per chunk: the
    result is cached by the caller's composite cache, so a month shared by
    1,000 chunks costs one `getInfo`, not 1,000.
    """
    import ee

    start, end = _month_bounds(year, month)
    col = (ee.ImageCollection(collection_id)
           .filterDate(start, end).select([band]))
    if col.size().getInfo() == 0:
        return None
    return col.median()


def s5p_monthly_climatology(month: int, collection_id: str, band: str,
                            first_year: int = 2019,
                            last_year: int = 2023) -> tuple:
    """`(mean_image, std_image)` of one S5P band for a calendar month over a
    fixed window, to z-score the event month.

    Returns `(None, None)` when the window has no data at all, so callers
    using `probe` short-circuit the same way they do for an empty event
    month. Months with no data in the window contribute NOTHING -- they are
    filtered out rather than included as 0-band images, which would
    propagate into the mean and std and produce a 0-band result.
    """
    import ee

    imgs = []
    for y in range(int(first_year), int(last_year) + 1):
        img = s5p_monthly_median(y, month, collection_id, band)
        if img is not None:
            imgs.append(img)
    if not imgs:
        return None, None
    stack = ee.ImageCollection(imgs)
    return stack.mean(), stack.reduce(ee.Reducer.stdDev()).rename([band])


def era5_monthly_climatology(month: int,
                             collection_id: str =
                             "ECMWF/ERA5_LAND/DAILY_AGGR",
                             first_year: int = 1991,
                             last_year: int = 2020) -> tuple:
    """`(mean_image, std_image)` of ERA5-Land monthly means for a calendar
    month over the standard 1991-2020 window."""
    import ee

    bands = ["temperature_2m", "dewpoint_temperature_2m",
             "u_component_of_wind_10m", "v_component_of_wind_10m"]
    imgs = [ee.ImageCollection(collection_id)
            .filterDate(*_month_bounds(y, month)).select(bands).mean()
            for y in range(int(first_year), int(last_year) + 1)]
    stack = ee.ImageCollection(imgs)
    # stdDev reducer appends '_stdDev' to band names; rename back so both mean
    # and std images share identical band names.
    return stack.mean(), stack.reduce(ee.Reducer.stdDev()).rename(bands)


def sample_bands_chunked(images: dict, records, id_key: str = "row",
                         scale: int = 11132, chunk_size: int = 5_000,
                         progress_every: int = 0, printer=print) -> dict:
    """`sampleRegions` over SEVERAL single-band images, chunked.

    `images` maps a band name to an `ee.Image`. Every point is sampled in
    EVERY image; the result is `{row: {band: value, ...}}`. The default
    ~11 km scale matches ERA5-Land's grid (weather is not a 30 m quantity).
    """
    import ee

    def _one(chunk):
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r["lon"]), float(r["lat"])]),
                       {id_key: r[id_key]}) for r in chunk])
        got = {}
        for band, img in images.items():
            res = (img.select(img.bandNames().get(0)).rename(["v"])
                   .sampleRegions(collection=fc, properties=[id_key],
                                  scale=scale, geometries=False).getInfo())
            for f in res.get("features", []):
                p = f.get("properties", {})
                v = p.get("v")
                if v is not None:
                    got.setdefault(p.get(id_key), {})[band] = float(v)
        return got

    total = len(records)
    pending = [records[i:i + chunk_size]
               for i in range(0, total, chunk_size)]
    out = {}
    for i, chunk in enumerate(pending):
        try:
            out.update(_one(chunk))
        except Exception as exc:                    # noqa: BLE001
            printer(f"  chunk {i} FAILED ({type(exc).__name__}): "
                    f"{str(exc)[:140]}")
    return out