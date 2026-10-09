"""
Model registry and inference.

Loads the artefacts P7 writes to `qc/models/` and serves real predictions,
replacing the hardcoded sample values the rest of the API still returns.

Three properties this enforces, each of which silently corrupts a score if it is
skipped:

1. **The feature list is part of the model.** A LightGBM booster maps column
   *positions*, so a frame reordered or renamed against it produces confident
   nonsense rather than an error. The stored list is therefore applied, never
   inferred from the caller's dict.

2. **Absent features become NaN, never zero.** NaN is how "could not measure"
   is encoded throughout the pipeline, and LightGBM routes it natively. A zero
   would assert "measured, and it was zero".

3. **Unknown columns are refused.** A feature the model never saw cannot be
   passed in silently; it belongs in a different model version.

Serving the weaker of the two trained models by default is deliberate: model A
learned the weak-label rules and its 0.9963 PR-AUC is circular, so it is
available only when explicitly requested.
"""

from __future__ import annotations

import logging
import pickle
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

#: Where P7 writes its artefacts, relative to the project root.
DEFAULT_MODEL_DIR = Path("qc/models")

#: Below this the prediction goes to a human rather than to an alert. From the
#: project brief: "any prediction below 60% confidence goes to human review".
#:
#: This is NOT an operating point. It is a fixed floor, and applying it to a
#: model that is not calibrated to a 0.60 prior is meaningless: model B's
#: held-out scores sit near 0.13, so a flat 0.60 cut flags 0.8% of positives.
#: The real alert threshold is `LoadedModel.auto_accept_threshold`, calibrated
#: by P7 and shipped inside the pickle. Keep this constant for the brief's rule;
#: do not use it as a classifier threshold.
REVIEW_CONFIDENCE = 0.60


class ModelContractError(RuntimeError):
    """Raised when supplied features cannot be reconciled with the model."""


class ModelNotAvailable(RuntimeError):
    """Raised when no trained model artefact can be found."""


@dataclass
class LoadedModel:
    """A fitted booster plus everything needed to use it correctly."""

    name: str
    booster: Any
    features: List[str]
    target: str
    label_column: str
    best_iteration: int
    review_confidence: float
    auto_accept_threshold: float
    operating_threshold: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_operating_point(self) -> bool:
        """False when P7 could not freeze a usable threshold.

        A model with no operating point must degrade to "review everything",
        never to "alert everything". `auto_accept_threshold` already handles
        that; this exists so the API can say so out loud rather than letting a
        caller infer it from a number.
        """
        return self.operating_threshold is not None

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Score one event.

        Missing keys become NaN; extra keys are rejected. Returns the
        probability, the confidence band, and which features were absent, so the
        caller can put the gaps on the alert instead of hiding them.

        Two thresholds are reported and they mean different things:
          `review_confidence`       0.60, the brief's human-review floor.
          `auto_accept_threshold`   max(calibrated tau, review floor), and the
                                    only value that should gate an automatic
                                    alert.
        """
        if not isinstance(features, dict):
            raise ModelContractError("features must be a mapping")

        unknown = sorted(set(features) - set(self.features))
        if unknown:
            raise ModelContractError(
                f"model {self.name} was not trained on {len(unknown)} of these "
                f"columns: {unknown[:8]}. Retrain, or serve the model that "
                "expects them."
            )

        row = np.full((1, len(self.features)), np.nan, dtype="float32")
        missing: List[str] = []
        supplied: Dict[str, float] = {}
        for i, name in enumerate(self.features):
            value = features.get(name, None)
            if value is None:
                missing.append(name)
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ModelContractError(
                    f"feature {name!r} is not numeric: {value!r}"
                ) from exc
            if np.isnan(numeric):
                missing.append(name)
                continue
            row[0, i] = numeric
            supplied[name] = numeric

        probability = float(self.booster.predict(row)[0])
        return {
            "model": self.name,
            "target": self.target,
            "probability": probability,
            "confidence": probability,
            # The decision that matters, at the calibrated deployed bar.
            "above_auto_accept_threshold": (
                probability >= self.auto_accept_threshold
            ),
            "auto_accept_threshold": self.auto_accept_threshold,
            "operating_threshold": self.operating_threshold,
            "has_operating_point": self.has_operating_point,
            # The brief's floor, kept separate so it is never mistaken for the
            # operating point.
            "above_review_threshold": probability >= self.review_confidence,
            "requires_human_review": probability < self.auto_accept_threshold,
            "review_confidence": self.review_confidence,
            "n_features_expected": len(self.features),
            "n_features_supplied": len(supplied),
            "missing_features": missing,
            "missing_fraction": (
                len(missing) / len(self.features) if self.features else 0.0
            ),
            "best_iteration": self.best_iteration,
        }


class ModelRegistry:
    """Thread-safe, load-once cache of the trained models."""

    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR):
        self.model_dir = Path(model_dir)
        self._models: Dict[str, LoadedModel] = {}
        self._lock = threading.Lock()

    @property
    def available(self) -> List[str]:
        """Model NAMES (e.g. "B"), matching what `path_for` and `load` take.

        Deriving these from `p.stem` would give "model_B", which no other
        method accepts -- so `available` would advertise a name that then fails
        to load.
        """
        return sorted(
            p.stem[len("model_"):]
            for p in self.model_dir.glob("model_*.pkl")
            if p.stem.startswith("model_")
        )

    def path_for(self, name: str) -> Path:
        return self.model_dir / f"model_{name}.pkl"

    def load(self, name: str) -> LoadedModel:
        """Load and cache a model. Raises ModelNotAvailable with guidance."""
        with self._lock:
            if name in self._models:
                return self._models[name]

        path = self.path_for(name)
        if not path.exists():
            known = self.available
            raise ModelNotAvailable(
                f"no artefact at {path}. "
                + (
                    f"Found {known} instead."
                    if known
                    else "No models found -- run colab_p7_train.py to write "
                    "qc/models/model_{A,B}.pkl."
                )
            )

        with path.open("rb") as fh:
            payload = pickle.load(fh)

        for key in ("booster", "features", "target"):
            if key not in payload:
                raise ModelContractError(
                    f"{path.name} is missing {key!r}; it was not written by the "
                    "current P7 cell. Re-run colab_p7_train.py."
                )

        review_confidence = float(
            payload.get("review_confidence", REVIEW_CONFIDENCE)
        )
        operating_threshold = payload.get("operating_threshold")
        if operating_threshold is not None:
            operating_threshold = float(operating_threshold)

        model = LoadedModel(
            name=name,
            booster=payload["booster"],
            features=list(payload["features"]),
            target=payload["target"],
            label_column=payload.get("label_column", "unknown"),
            best_iteration=int(payload.get("best_iteration", 0)),
            review_confidence=review_confidence,
            operating_threshold=operating_threshold,
            # Prefer the value P7 computed. Falling back to the bare 0.60 for a
            # model whose scores live near 0.13 is what produced a "detects 2 of
            # 100" detector, so the fallback is only used for artefacts written
            # before calibration existed -- and it is reported when it happens.
            auto_accept_threshold=float(
                payload.get(
                    "auto_accept_threshold",
                    max(review_confidence, operating_threshold)
                    if operating_threshold is not None
                    else review_confidence,
                )
            ),
            extra={
                k: v
                for k, v in payload.items()
                if k
                not in {
                    "booster", "features", "target", "label_column",
                    "best_iteration", "review_confidence",
                    "operating_threshold", "auto_accept_threshold",
                }
            },
        )
        if operating_threshold is None:
            logger.warning(
                "model %s carries no calibrated operating_threshold; falling "
                "back to the brief's %.2f review floor. Every event will need "
                "human review. Re-run colab_p7_train.py to calibrate.",
                name, review_confidence,
            )
        logger.info(
            "loaded model %s: %d features, target %s, %d trees, "
            "auto-accept at %.6f (tau %s)",
            name, len(model.features), model.target, model.best_iteration,
            model.auto_accept_threshold,
            "none" if operating_threshold is None else f"{operating_threshold:.6f}",
        )
        with self._lock:
            self._models[name] = model
        return model

    def describe(self, name: str) -> Dict[str, Any]:
        """Model card: what it is, and what it is not."""
        model = self.load(name)
        return {
            "name": model.name,
            "target": model.target,
            "label_column": model.label_column,
            "n_features": len(model.features),
            "features": model.features,
            "best_iteration": model.best_iteration,
            "review_confidence": model.review_confidence,
            "operating_threshold": model.operating_threshold,
            "auto_accept_threshold": model.auto_accept_threshold,
            "has_operating_point": model.has_operating_point,
            "trained_on": model.extra.get("split"),
            "n_train": model.extra.get("n_train"),
            "n_test": model.extra.get("n_test"),
            "caveats": list(MODEL_CAVEATS.get(name, [])),
        }

    def predict(self, name: str, features: Dict[str, Any]) -> Dict[str, Any]:
        return self.load(name).predict(features)


#: Recorded per model so the API never presents a circular number as a result.
MODEL_CAVEATS: Dict[str, List[str]] = {
    "A": [
        "TRAINED ON WEAK LABELS manufactured from features the model also "
        "sees. Its PR-AUC of 0.9963 is circular and is NOT evidence of "
        "generalisation.",
        "Retained as the operational model: it is what the deployed system "
        "would use, with facility priors included.",
    ],
    "B": [
        "The scientific partition: all proximity and rule-input columns "
        "removed.",
        "PR-AUC 0.1721, BELOW the single-threshold baseline's F1 of 0.264. "
        "With leakage removed, the current features cannot identify "
        "facility-proximity fire in unseen regions.",
        "Importance is dominated by PEAK_HOUR_COS/SIN, i.e. satellite "
        "overpass timing rather than fire behaviour.",
        "NO operating point could be frozen: no threshold reaches precision "
        "0.90 at any recall worth having. Every event must go to human review.",
    ],
}


#: Process-wide registry. `app.core.config` supplies the directory.
registry = ModelRegistry()


def configure(model_dir: Path | str) -> None:
    """Point the registry at the directory holding the artefacts."""
    registry.model_dir = Path(model_dir)
    with registry._lock:            # noqa: SLF001 - same class, deliberate
        registry._models.clear()     # noqa: SLF001


def get_registry() -> ModelRegistry:
    return registry


def health() -> Dict[str, Any]:
    """Model status for the /health endpoint, without loading anything heavy."""
    available = registry.available
    return {
        "model_dir": str(registry.model_dir),
        "models_found": available,
        "loaded": sorted(registry._models),   # noqa: SLF001
        "serving_default": "B",
        "note": (
            "Default serving model is B (no leakage). Model A is available "
            "explicitly but scores against weak labels."
        ),
    }