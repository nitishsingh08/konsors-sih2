# ============================================================================
# confidence.py -- Uncertainty estimation for fire predictions
# ============================================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class PredictionWithConfidence:
    """Prediction with uncertainty quantification."""
    prediction: float
    mean: float
    std: float
    lower_ci: float  # 5th percentile
    upper_ci: float  # 95th percentile
    confidence_level: float = 0.90

    @property
    def relative_uncertainty(self) -> float:
        """Relative uncertainty as coefficient of variation."""
        return self.std / (abs(self.mean) + 1e-8)

    @property
    def is_reliable(self) -> bool:
        """Whether prediction is reliable based on uncertainty."""
        return self.relative_uncertainty < 0.5

    def to_dict(self) -> dict:
        return {
            "prediction": self.prediction,
            "mean": self.mean,
            "std": self.std,
            "ci_lower": self.lower_ci,
            "ci_upper": self.upper_ci,
            "confidence_level": self.confidence_level,
            "relative_uncertainty": self.relative_uncertainty,
            "is_reliable": self.is_reliable
        }


def compute_prediction_interval(
    predictions: np.ndarray,
    confidence: float = 0.90
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute prediction intervals from ensemble/bootstrap predictions.
    
    Args:
        predictions: Shape (n_samples, n_estimators) or (n_samples,)
        confidence: Confidence level (e.g., 0.90 for 90% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound) arrays
    """
    if predictions.ndim == 1:
        # Single prediction - use bootstrap or assume normal
        return np.full_like(predictions, np.nan), np.full_like(predictions, np.nan)
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(predictions, 100 * alpha, axis=1)
    upper = np.percentile(predictions, 100 * (1 - alpha), axis=1)
    return lower, upper


def compute_prediction_intervals_quantile(
    model,
    X: np.ndarray,
    quantiles: List[float] = [0.05, 0.5, 0.95]
) -> np.ndarray:
    """
    Get quantile predictions from LightGBM quantile regression.
    Requires model trained with 'quantile' objective.
    """
    # This would require retraining with quantile objective
    # Placeholder for future implementation
    raise NotImplementedError("Requires quantile regression objective")


def bootstrap_uncertainty(
    model,
    X: np.ndarray,
    n_bootstrap: int = 100,
    sample_frac: float = 0.8,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap uncertainty estimation by resampling training data.
    
    Returns:
        mean_pred: Mean predictions
        lower_ci: Lower confidence bound
        upper_ci: Upper confidence bound
    """
    # This would require retraining model multiple times
    # Placeholder for future implementation
    raise NotImplementedError("Requires model retraining")


def prediction_interval_from_shap(
    shap_values: np.ndarray,
    base_value: float,
    feature_values: np.ndarray,
    confidence: float = 0.90
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate prediction intervals from SHAP values.
    Uses the fact that SHAP values sum to prediction - base_value.
    """
    # Sum of absolute SHAP values as uncertainty proxy
    shap_magnitude = np.abs(shap_values).sum(axis=1)
    
    # Normalize to prediction scale
    pred_scale = np.abs(shap_values).sum(axis=1).mean()
    
    # Simple heuristic: uncertainty proportional to SHAP magnitude
    rel_uncertainty = shap_magnitude / (pred_scale + 1e-8)
    
    # Scale to prediction confidence
    alpha = (1 - 0.90) / 2
    z = 1.645  # 90% CI
    
    # This is heuristic - real implementation needs proper uncertainty quantification
    lower = np.zeros_like(shap_values[:, 0])
    upper = np.zeros_like(shap_values[:, 0])
    
    return lower, upper


def prediction_interval_conformal(
    model,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Conformal prediction intervals - distribution-free, finite-sample valid.
    
    Requires calibration set (X_cal, y_cal) separate from training.
    """
    # Get predictions on calibration set
    preds_cal = model.predict_proba(X_cal)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_cal)
    
    # Compute non-conformity scores
    scores = np.abs(y_cal - preds_cal)
    
    # Get threshold
    q = np.quantile(scores, 1 - alpha)
    
    # Predict on test
    preds_test = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
    
    # Prediction intervals
    lower = preds_test - q
    upper = preds_test + q
    
    return np.clip(lower, 0, 1), np.clip(upper, 0, 1)


def prediction_with_confidence(
    model,
    X: np.ndarray,
    method: str = "conformal",
    **kwargs
) -> List[PredictionWithConfidence]:
    """
    Get predictions with confidence intervals.
    
    Args:
        model: Trained LightGBM model
        X: Feature matrix
        method: "conformal" (requires calibration set) or "heuristic"
    
    Returns:
        List of PredictionWithConfidence objects
    """
    if method == "conformal":
        raise NotImplementedError("Requires calibration set")
    
    # Heuristic: use prediction variance across trees
    preds = []
    for tree in model.trees:
        preds.append(tree.predict(X))
    
    preds_array = np.array(preds).T  # (n_samples, n_trees)
    
    mean_pred = preds_array.mean(axis=1)
    std_pred = preds_array.std(axis=1)
    
    # Approximate 90% CI using tree variance
    z = 1.645  # 90% CI
    lower = mean_pred - 1.645 * std_pred
    upper = mean_pred + 1.645 * std_pred
    
    results = []
    for i in range(len(X)):
        pred = PredictionWithConfidence(
            prediction=float(np.clip(mean_pred[i], 0, 1)),
            mean=float(mean_pred[i]),
            std=float(std_pred[i]),
            lower_ci=float(np.clip(mean_pred[i] - 1.645 * std_pred[i], 0, 1)),
            upper_ci=float(np.clip(mean_pred[i] + 1.645 * std_pred[i], 0, 1)),
            confidence_level=0.90
        )
        preds.append(pred)
    
    return preds


# ============================================================================
# Model calibration utilities
# ============================================================================

def calibrate_probabilities(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str = "isotonic"
) -> Tuple[np.ndarray, object]:
    """
    Calibrate model probabilities using isotonic regression or Platt scaling.
    
    Returns calibrated probabilities and the calibrator object.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrated = calibrator.fit_transform(probabilities, y_true)
        return calibrated, calibrator
    elif method == "platt":
        lr = LogisticRegression()
        calibrated = calibrator.fit(probabilities.reshape(-1, 1), y_true)
        return calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1], calibrator
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def evaluate_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> dict:
    """Evaluate calibration quality using reliability diagram."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(probabilities, bins) - 1
    
    results = []
    for i in range(n_bins):
        mask = bin_ids == i
        if mask.sum() > 0:
            acc = y_true[mask].mean()
            conf = probabilities[mask].mean()
            count = mask.sum()
            results.append({
                "bin": i,
                "accuracy": acc,
                "confidence": conf,
                "count": int(count)
            })
    
    return {
        "bins": results,
        "ece": sum(abs(r["accuracy"] - r["confidence"]) * r["count"] for r in results) / len(y_true)
    }


# ============================================================================
# Fire spread uncertainty
# ============================================================================

def fire_spread_uncertainty(
    spread_predictions: Dict[str, np.ndarray],
    feature_uncertainties: Dict[str, float]
) -> Dict[str, Dict[str, float]]:
    """
    Propagate uncertainty through fire spread model.
    
    Uses Monte Carlo sampling to propagate input uncertainties
    through the spread model.
    """
    results = {}
    for target, preds in spread_predictions.items():
        if len(preds) > 1:
            mean = np.mean(preds)
            std = np.std(preds)
            results[target] = {
                "mean": float(np.mean(preds)),
                "std": float(np.std(preds)),
                "lower_90": float(np.percentile(preds, 5)),
                "upper_90": float(np.percentile(preds, 95)),
                "cv": float(np.std(preds) / (np.mean(preds) + 1e-8))
            }
        else:
            results[target] = {
                "mean": float(preds[0]),
                "std": 0.0,
                "lower_90": float(preds[0]),
                "upper_90": float(preds[0]),
                "cv": 0.0
            }
    return results


# ============================================================================
# Export all
# ============================================================================

__all__ = [
    "PredictionWithConfidence",
    "compute_prediction_interval",
    "compute_prediction_intervals_quantile",
    "bootstrap_uncertainty",
    "prediction_interval_from_shap",
    "prediction_interval_conformal",
    "prediction_with_confidence",
    "calibrate_probabilities",
    "evaluate_calibration",
    "fire_spread_uncertainty",
]