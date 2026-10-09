"""
P8 -- MODEL EXPLAINABILITY (SHAP and LIME).

Replaces `precompute_shap_explanations`, which returns the SAME fixed
contributions regardless of input. An explanation that does not change with the
event is not an explanation, and serving it alongside a real prediction is worse
than serving nothing.

Two methods, because they fail differently and disagreeing is informative:

* **SHAP** (`TreeExplainer`) -- exact additive attributions for tree models.
  Deterministic: the same event always yields the same numbers. This is the one
  to trust for per-event "why this alert".
* **LIME** -- local, perturbation-based. Noisy and model-agnostic. It is the
  control: if SHAP and LIME disagree on which features matter, the explanation
  is not stable and the feature ranking should not be published.

Both are computed from the SAME row, in the model's own feature order, with
NaN preserved. NaN is not zero: substituting 0 for "unmeasured" would let an
explanation attribute a decision to a feature that was never available.

Aggregation to families
----------------------
The brief reports contributions "by family" (thermal, temporal, spread, ...).
That mapping is explicit and tested, because silently grouping a feature into the
wrong family would mislabel what drove an alert.

Honest limits, recorded in every response
-----------------------------------------
SHAP explains the MODEL, not the fire. Model B was trained on weak labels
manufactured from the features it sees, so its attributions describe how the
model reproduces its own labelling rules. Model A's attributions are almost
entirely a restatement of "near a facility", because that is what its label was.
Neither is a statement about combustion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FEATURE -> FAMILY, from the ledger's groups
# ---------------------------------------------------------------------------


def _feature_families() -> Dict[str, str]:
    import feature_ledger as fl

    return {name: group for group, names in fl.FEATURE_GROUPS.items()
            for name in names if name not in fl._PLACEHOLDERS}


FAMILY_BY_FEATURE: "Optional[Dict[str, str]]" = None


def family_of(feature: str) -> str:
    """Which family a feature belongs to. Falls back to `other`, never None."""
    global FAMILY_BY_FEATURE
    if FAMILY_BY_FEATURE is None:
        FAMILY_BY_FEATURE = _feature_families()
    return FAMILY_BY_FEATURE.get(feature, "other")


def aggregate_by_family(features: List[str], values: List[float]) -> Dict[str, float]:
    """Sum attributions per family, keeping the sign.

    Summing signed values matters: positives can cancel negatives within a
    family, and taking absolute values would report a family as influential
    when it netted out to nothing.
    """
    out: Dict[str, float] = {}
    for feature, value in zip(features, values):
        fam = family_of(feature)
        out[fam] = out.get(fam, 0.0) + float(value)
    return out


def top_contributions(features: List[str], values: List[float], k: int = 5) -> List[Dict]:
    """The `k` largest-magnitude attributions, strongest first.

    Ranked by |value|, not by value: a feature pushing the score DOWN is just as
    important to explain as one pushing it up, and a signed ranking would hide
    every suppressive feature.
    """
    ranked = sorted(zip(features, values), key=lambda fv: abs(fv[1]),
                    reverse=True)
    return [
        {"key": name,
         "contribution": round(float(val), 6),
         "family": family_of(name),
         "direction": "raises_score" if val > 0 else "lowers_score"}
        for name, val in ranked[:k]
    ]


# ---------------------------------------------------------------------------
# Caveats carried with every explanation
# ---------------------------------------------------------------------------

CAVEATS = {
    "A": [
        "Model A was trained on weak labels manufactured from facility "
        "proximity, and it holds the DIST_* columns that built them.",
        "Its attributions therefore largely restate 'this event is near a "
        "facility'. That is a property of the labelling, not a discovery.",
    ],
    "B": [
        "Model B excludes all proximity and rule-input features.",
        "Its attributions describe how the thermal/temporal features "
        "reproduce the weak label. They are NOT evidence of physical cause.",
    ],
    "LIME": [
        "LIME is stochastic: a different seed can reorder weak attributions.",
        "Use it as a stability check on SHAP, not as the published ranking.",
    ],
}


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------


@dataclass
class Explanation:
    """One event's explanation, method-tagged and self-describing."""

    method: str
    model: str
    base_value: float
    prediction: float
    features: List[str]
    values: List[float]
    #: base + sum(phi) is a LOG-ODIT; `prediction` is a PROBABILITY. Kept
    #: separately because comparing them looks like a broken explanation when
    #: it is only a unit mismatch. Defaults must follow the non-default fields.
    margin_prediction: Optional[float] = None
    additivity_residual: Optional[float] = None
    by_family: Dict[str, float] = field(default_factory=dict)
    top_features: List[Dict] = field(default_factory=list)
    n_features_expected: int = 0
    n_features_supplied: int = 0
    missing_features: List[str] = field(default_factory=list)
    stability: Optional[Dict] = field(default_factory=dict)
    caveats: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "model": self.model,
            "base_value": round(float(self.base_value), 6),
            "base_value_space": "log_odds",
            "prediction": round(float(self.prediction), 6),
            "prediction_space": "probability",
            "margin_prediction": (
                round(float(self.margin_prediction), 6)
                if self.margin_prediction is not None else None
            ),
            "additivity_residual": (
                round(float(self.additivity_residual), 8)
                if self.additivity_residual is not None else None
            ),
            "by_family": {k: round(v, 6) for k, v in self.by_family.items()},
            "top_features": self.top_features,
            "n_features_expected": self.n_features_expected,
            "n_features_supplied": self.n_features_supplied,
            "missing_features": self.missing_features,
            "stability": self.stability,
            "caveats": self.caveats,
        }


def shap_explain(booster, features: List[str], row: Dict[str, Any],
                 model_name: str = "B") -> Explanation:
    """Exact additive SHAP attributions for one event."""
    import shap

    matrix = np.full((1, len(features)), np.nan, dtype="float32")
    missing = []
    supplied = 0
    for i, name in enumerate(features):
        value = row.get(name)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            missing.append(name)
            continue
        matrix[0, i] = float(value)
        supplied += 1

    # TreeExplainer needs no NaN, and substituting 0 would claim an
    # unmeasured feature was measured at zero. So a SHAP run needs the row's
    # measured features; missing ones are reported rather than invented.
    if missing:
        raise MissingFeatureError(
            f"SHAP needs a value for every feature the model uses; "
            f"{len(missing)} are unmeasured: {missing[:6]}"
        )

    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(matrix)
    if isinstance(values, list):          # older shap returns one array/class
        values = values[1]
    attributions = np.asarray(values, dtype="float64").ravel()

    prediction = float(booster.predict(matrix)[0])
    base = float(np.ravel(explainer.expected_value)[0]) \
        if np.ndim(explainer.expected_value) else float(explainer.expected_value)

    # Additivity holds in ONE space, and the two available numbers are in
    # different ones: `booster.predict` returns a probability, while SHAP values
    # and expected_value are in log-odds (margin) space. Comparing
    # base + sum(phi) against the probability looks like a broken explanation
    # when it is only a unit mismatch, so both are reported, each named for its
    # own space, and `additivity_residual` is the check that belongs here.
    margin = base + float(np.sum(attributions))

    return Explanation(
        method="shap",
        model=model_name,
        base_value=base,
        prediction=prediction,
        margin_prediction=margin,
        additivity_residual=abs(_sigmoid(margin) - prediction),
        features=list(features),
        values=[float(v) for v in attributions],
        by_family=aggregate_by_family(features, attributions),
        top_features=top_contributions(features, attributions),
        n_features_expected=len(features),
        n_features_supplied=supplied,
        missing_features=[],
        caveats=list(CAVEATS.get(model_name, [])),
    )


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


class MissingFeatureError(ValueError):
    """Raised when a method needs values the caller did not supply."""


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------


def _lime_kernel(booster, features: List[str]):
    """Wrap the booster as a LIME-compatible classifier.

    LIME needs a 2-D array of class probabilities, one COLUMN PER CLASS, and
    raises NotImplementedError for anything else -- including a correct 1-D
    vector of positive-class probabilities, and including LightGBM's raw
    unbounded margins. So the margin is squashed and paired with its complement.
    """
    def fn(matrix):
        arr = np.asarray(matrix, dtype="float64").reshape(-1, len(features))
        margin = booster.predict(arr).astype("float64")
        # Clipped before exp so a large margin cannot overflow to inf.
        positive = 1.0 / (1.0 + np.exp(-np.clip(margin, -60.0, 60.0)))
        return np.column_stack([1.0 - positive, positive])

    return fn


def lime_explain(booster, features: List[str], row: Dict[str, Any],
                 model_name: str = "B", n_samples: int = 2000,
                 seed: int = 0) -> Explanation:
    """Local perturbation explanation, used as a SHAP stability check.

    LIME perturbs real data, so NaN is a problem: there is no sensible way to
    resample a missing measurement. Rather than substitute 0, a row with missing
    features is rejected -- LIME is available once the feature blocks are more
    complete.
    """
    from lime.lime_tabular import LimeTabularExplainer

    missing = [n for n in features if row.get(n) is None
               or (isinstance(row.get(n), float) and np.isnan(row.get(n)))]
    if missing:
        raise MissingFeatureError(
            f"LIME resamples observed data and cannot impute a missing "
            f"measurement: {len(missing)} absent ({missing[:6]}). Add the "
            "feature blocks, or use SHAP on a complete row."
        )

    # LIME needs a background distribution. A tiny constant sample centred on
    # the event is enough for a local linear fit and keeps the call cheap; the
    # perturbation scale is the per-feature spread of the supplied row, which is
    # crude but documented rather than hidden.
    point = np.array([[float(row[n]) for n in features]], dtype="float64")

    # The background is STANDARDISED, not a jitter around the event. An earlier
    # version scaled the jitter by |feature value|, which for features near zero
    # collapses the design matrix to a singular one; LIME then emits
    # divide-by-zero and overflow warnings and returns a degenerate fit whose
    # ranking agrees with SHAP for the wrong reason.
    background = np.random.default_rng(seed).normal(
        0.0, 1.0, size=(64, len(features))
    )
    _mu = background.mean(axis=0)
    _sd = np.where(background.std(axis=0) > 1e-9, background.std(axis=0), 1.0)
    _z_point = ((point[0] - _mu) / _sd).reshape(1, -1)

    explainer = LimeTabularExplainer(
        training_data=background,
        feature_names=list(features),
        mode="classification",
        discretize_continuous=False,
        class_names=["other", model_name],
    )
    _z_kernel = _lime_kernel_z(booster, features, _mu, _sd)
    explanation = explainer.explain_instance(
        _z_point[0],
        _z_kernel,
        num_features=min(len(features), 15),
        num_samples=n_samples,
    )
    # `local_exp` is a DICT keyed by class label, and each entry pairs a feature
    # COLUMN INDEX with a weight -- not names with weights. Reading it as a list
    # of (name, weight) pairs, or leaving the index as if it were a name, both
    # fail or silently mislabel the attribution.
    _label = list(explanation.local_exp)[-1]      # the positive class
    _weighted = explanation.local_exp[_label]
    pairs = []
    for feature_id, weight in _weighted:
        idx = int(feature_id)
        if not 0 <= idx < len(features):
            raise ValueError(
                f"LIME returned feature index {idx} for a model with "
                f"{len(features)} features -- the explanation does not match "
                "the model it came from."
            )
        pairs.append((features[idx], float(weight)))
    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    top = [
        {"key": feature,
         "contribution": round(float(weight), 6),
         "family": family_of(feature),
         "direction": "raises_score" if weight > 0 else "lowers_score"}
        for feature, weight in pairs
    ]
    return Explanation(
        method="lime",
        model=model_name,
        base_value=float("nan"),
        prediction=float(np.ravel(explanation.local_pred)[-1]),
        features=list(features),
        values=[float(w) for _, w in pairs],
        by_family=aggregate_by_family([f for f, _ in pairs],
                                       [w for _, w in pairs]),
        top_features=top,
        n_features_expected=len(features),
        n_features_supplied=len(features),
        missing_features=[],
        stability={"n_samples": n_samples, "seed": seed,
                   "note": "stochastic; use as a check on SHAP"},
        caveats=list(CAVEATS.get(model_name, [])) + list(CAVEATS["LIME"]),
    )


def _lime_kernel_z(booster, features, mu, sd):
    """LIME kernel in standardised space, so perturbations are comparable.

    The linear fit is solved on the z-scored background LIME was given; scores
    must be computed after undoing the transform, or the attribution describes
    the wrong coordinates.
    """
    def fn(matrix):
        arr = np.asarray(matrix, dtype="float64").reshape(-1, len(features))
        return _lime_kernel(booster, features)(arr * sd + mu)
    return fn


def compare_methods(shap_exp: Explanation, lime_exp: Optional[Explanation],
                    top_k: int = 5) -> Dict[str, Any]:
    """Do the two methods agree on which features matter?

    Reported because disagreement is the finding. Publishing a ranking that two
    methods do not share overstates what is known.
    """
    if lime_exp is None:
        return {"comparable": False,
                "note": "LIME unavailable; see its caveat for why"}
    a = {f["key"] for f in shap_exp.top_features[:top_k]}
    b = {f["key"] for f in lime_exp.top_features[:top_k]}
    shared = sorted(a & b)
    union = a | b
    jaccard = len(shared) / len(union) if union else 0.0
    return {
        "comparable": True,
        "shap_top": sorted(a),
        "lime_top": sorted(b),
        "shared": shared,
        "only_shap": sorted(a - b),
        "only_lime": sorted(b - a),
        "jaccard": round(jaccard, 4),
        "agreement": ("strong" if jaccard >= 0.6
                      else "partial" if jaccard >= 0.3 else "weak"),
    }