# Model artefacts

Trained boosters live on Google Drive, not in git:

```
My Drive/Firm_Clusters/qc/models/model_A.pkl    operational, weak labels (circular)
My Drive/Firm_Clusters/qc/models/model_B.pkl    scientific, no leakage (default)
```

Written by `colab_p7_train.py`.

## Get them

```bash
python3 scripts/fetch_models.py          # copies from a mounted Drive
python3 scripts/fetch_models.py --verify-only
```

If Google Drive is not mounted, download from drive.google.com:
`My Drive > Firm_Clusters > qc > models`, and save both files here.

Then:

```bash
export MODEL_DIR=$(pwd)/models
cd backend && uvicorn app.main:app --reload
```

## macOS: install libomp FIRST

LightGBM will not import on a clean macOS without OpenMP:

```
OSError: dlopen(...lib_lightgbm.dylib): Library not loaded:
@rpath/libomp.dylib
```

```bash
brew install libomp
python3 -m pip install -U lightgbm
python3 -c "import lightgbm; print(lightgbm.__version__)"
```

This is the single most common reason a Colab-trained model will not load
locally. Because the artefact is a pickle produced in Colab, a local LightGBM
incompatible with that build fails on `pickle.load` -- which is why
`--verify-only` *loads* each file and checks its contract rather than just
confirming the bytes exist.

## Not in git, deliberately

A `.pkl` executes code on load, and 3.6 MB per revision bloats history for
something reproducible from `colab_p7_train.py`. Regenerate rather than commit.

## Sanity check

`GET /api/models/B` must report `n_features: 29` and `target: INDUSTRIAL_FIRE`.
Anything else means the wrong artefact or a stale cell.
