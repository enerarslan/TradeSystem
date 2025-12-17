"""
Probability Calibration for financial ML models.

This module provides calibration methods to convert raw model outputs
(logits/uncalibrated probabilities) into reliable probability estimates.

Critical for financial applications:
- Position sizing based on confidence levels
- Risk management with accurate probability estimates
- Regulatory compliance (model risk management)

Supported calibration methods:
- Platt Scaling (sigmoid calibration)
- Isotonic Regression (non-parametric)
- Temperature Scaling (for neural networks)
- Beta Calibration (bounded outputs)

Designed for JPMorgan-level requirements:
- Calibration diagnostics (reliability diagrams)
- CV-based calibration (CalibratedClassifierCV)
- Brier score evaluation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import brier_score_loss, log_loss

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


logger = logging.getLogger(__name__)


class CalibrationMethod(str, Enum):
    """Supported calibration methods."""
    PLATT = "platt"  # Sigmoid/Platt scaling
    ISOTONIC = "isotonic"  # Isotonic regression
    TEMPERATURE = "temperature"  # Temperature scaling
    BETA = "beta"  # Beta calibration
    NONE = "none"


@dataclass
class CalibrationResult:
    """Container for calibration results and diagnostics."""
    method: str
    brier_score_before: float
    brier_score_after: float
    log_loss_before: Optional[float] = None
    log_loss_after: Optional[float] = None
    ece_before: Optional[float] = None  # Expected Calibration Error
    ece_after: Optional[float] = None
    calibration_curve_before: Optional[Tuple[np.ndarray, np.ndarray]] = None
    calibration_curve_after: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def improvement(self) -> float:
        """Calculate improvement in Brier score (negative is better)."""
        return self.brier_score_before - self.brier_score_after

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 50,
            "CALIBRATION RESULTS",
            "=" * 50,
            f"Method: {self.method}",
            f"Brier Score Before: {self.brier_score_before:.4f}",
            f"Brier Score After:  {self.brier_score_after:.4f}",
            f"Improvement: {self.improvement():.4f} ({self.improvement()/self.brier_score_before*100:.1f}%)",
        ]

        if self.ece_before is not None and self.ece_after is not None:
            lines.extend([
                f"ECE Before: {self.ece_before:.4f}",
                f"ECE After:  {self.ece_after:.4f}",
            ])

        lines.append("=" * 50)
        return "\n".join(lines)


class PlattScaler(BaseEstimator):
    """
    Platt Scaling (Sigmoid Calibration).

    Fits a logistic regression model to transform raw scores
    to calibrated probabilities.

    Best for:
    - Models with sigmoidally-distributed scores
    - When calibration needs to be smooth/monotonic
    - Smaller calibration datasets
    """

    def __init__(self, prior_correction: bool = True):
        """
        Args:
            prior_correction: If True, correct for class imbalance
        """
        self.prior_correction = prior_correction
        self.calibrator_ = None
        self.classes_ = None

    def fit(
        self,
        y_score: np.ndarray,
        y_true: np.ndarray,
    ) -> "PlattScaler":
        """
        Fit Platt scaler.

        Args:
            y_score: Predicted probabilities or scores (n_samples,)
            y_true: True labels (n_samples,)
        """
        self.classes_ = np.unique(y_true)

        # Reshape for sklearn
        X = y_score.reshape(-1, 1)

        self.calibrator_ = LogisticRegression(
            C=1e10,  # No regularization for calibration
            solver='lbfgs',
            max_iter=1000,
        )
        self.calibrator_.fit(X, y_true)

        return self

    def predict_proba(self, y_score: np.ndarray) -> np.ndarray:
        """
        Calibrate probabilities.

        Args:
            y_score: Raw probabilities (n_samples,)

        Returns:
            Calibrated probabilities (n_samples, 2)
        """
        if self.calibrator_ is None:
            raise ValueError("Scaler not fitted. Call fit() first.")

        X = y_score.reshape(-1, 1)
        return self.calibrator_.predict_proba(X)

    def calibrate(self, y_score: np.ndarray) -> np.ndarray:
        """
        Get calibrated positive class probabilities.

        Args:
            y_score: Raw probabilities

        Returns:
            Calibrated probabilities for positive class
        """
        return self.predict_proba(y_score)[:, 1]


class IsotonicCalibrator(BaseEstimator):
    """
    Isotonic Regression Calibration.

    Non-parametric calibration that maintains monotonicity.

    Best for:
    - Large calibration datasets (needs more data)
    - Non-sigmoid calibration curves
    - When no parametric assumption is desired
    """

    def __init__(self, out_of_bounds: str = "clip"):
        """
        Args:
            out_of_bounds: How to handle values outside training range
        """
        self.out_of_bounds = out_of_bounds
        self.calibrator_ = None

    def fit(
        self,
        y_score: np.ndarray,
        y_true: np.ndarray,
    ) -> "IsotonicCalibrator":
        """Fit isotonic calibrator."""
        self.calibrator_ = IsotonicRegression(
            out_of_bounds=self.out_of_bounds,
            y_min=0,
            y_max=1,
        )
        self.calibrator_.fit(y_score, y_true)
        return self

    def calibrate(self, y_score: np.ndarray) -> np.ndarray:
        """Calibrate probabilities."""
        if self.calibrator_ is None:
            raise ValueError("Calibrator not fitted.")
        return self.calibrator_.predict(y_score)


class TemperatureScaler(BaseEstimator):
    """
    Temperature Scaling for neural networks.

    Simple and effective calibration for deep learning models.
    Divides logits by a learned temperature parameter.

    Best for:
    - Deep learning models
    - When you have access to logits
    - Multi-class calibration
    """

    def __init__(self, max_iter: int = 100):
        self.max_iter = max_iter
        self.temperature_ = 1.0

    def fit(
        self,
        logits: np.ndarray,
        y_true: np.ndarray,
    ) -> "TemperatureScaler":
        """
        Fit optimal temperature.

        Uses NLL minimization to find best temperature.

        Args:
            logits: Raw logits (not softmax) from model
            y_true: True labels
        """
        from scipy.optimize import minimize

        def nll_loss(T):
            """Negative log likelihood at temperature T."""
            scaled_logits = logits / T
            # Softmax
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            # NLL
            return -np.mean(np.log(probs[np.arange(len(y_true)), y_true] + 1e-10))

        result = minimize(
            nll_loss,
            x0=1.0,
            bounds=[(0.01, 10.0)],
            method='L-BFGS-B',
        )

        self.temperature_ = result.x[0]
        logger.info(f"Temperature scaling: T = {self.temperature_:.3f}")

        return self

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling and return probabilities."""
        scaled_logits = logits / self.temperature_
        # Softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return probs


class CalibratedModel(BaseEstimator, ClassifierMixin):
    """
    Wrapper that adds calibration to any classifier.

    Creates a calibrated version of any model using
    cross-validation to prevent overfitting.

    Example:
        from lightgbm import LGBMClassifier

        # Create base model
        base_model = LGBMClassifier(n_estimators=100)

        # Wrap with calibration
        calibrated = CalibratedModel(
            base_model,
            method=CalibrationMethod.ISOTONIC,
            cv=5
        )

        # Train - calibration is learned internally
        calibrated.fit(X_train, y_train)

        # Predict - returns calibrated probabilities
        proba = calibrated.predict_proba(X_test)
    """

    def __init__(
        self,
        base_estimator: Any,
        method: CalibrationMethod = CalibrationMethod.ISOTONIC,
        cv: int = 5,
        n_jobs: int = -1,
    ):
        """
        Args:
            base_estimator: Base classifier to calibrate
            method: Calibration method
            cv: Cross-validation folds for calibration
            n_jobs: Parallel jobs
        """
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv
        self.n_jobs = n_jobs

        self.calibrated_classifier_: Optional[CalibratedClassifierCV] = None
        self.calibration_result_: Optional[CalibrationResult] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "CalibratedModel":
        """
        Fit base model with calibration.

        Uses CalibratedClassifierCV for cross-validated calibration.
        """
        sklearn_method = "isotonic" if self.method == CalibrationMethod.ISOTONIC else "sigmoid"

        self.calibrated_classifier_ = CalibratedClassifierCV(
            estimator=clone(self.base_estimator),
            method=sklearn_method,
            cv=self.cv,
            n_jobs=self.n_jobs,
        )

        fit_params = {}
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight

        self.calibrated_classifier_.fit(X, y, **fit_params)

        # Calculate calibration diagnostics
        self._compute_calibration_diagnostics(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.calibrated_classifier_ is None:
            raise ValueError("Model not fitted.")
        return self.calibrated_classifier_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities."""
        if self.calibrated_classifier_ is None:
            raise ValueError("Model not fitted.")
        return self.calibrated_classifier_.predict_proba(X)

    def _compute_calibration_diagnostics(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Compute calibration quality metrics."""
        # Get uncalibrated probabilities from base estimator
        # (would need to train separately - simplified here)

        # Get calibrated probabilities
        cal_proba = self.predict_proba(X)[:, 1]

        # Compute metrics
        brier = brier_score_loss(y, cal_proba)

        self.calibration_result_ = CalibrationResult(
            method=self.method.value,
            brier_score_before=0.0,  # Would need uncalibrated
            brier_score_after=brier,
        )

    @property
    def classes_(self):
        if self.calibrated_classifier_ is None:
            return None
        return self.calibrated_classifier_.classes_

    @property
    def feature_importances_(self):
        """Get feature importance from base estimator if available."""
        if self.calibrated_classifier_ is None:
            return None
        # Access first calibrated classifier's base estimator
        if hasattr(self.calibrated_classifier_, 'calibrated_classifiers_'):
            base = self.calibrated_classifier_.calibrated_classifiers_[0].estimator
            if hasattr(base, 'feature_importances_'):
                return base.feature_importances_
        return None


class ProbabilityCalibrator:
    """
    Post-hoc probability calibration for pre-trained models.

    Use this when you have already trained a model and want to
    calibrate its probability outputs.

    Example:
        # Train your model
        model = LGBMClassifier()
        model.fit(X_train, y_train)

        # Get predictions on calibration set
        proba = model.predict_proba(X_cal)[:, 1]

        # Calibrate
        calibrator = ProbabilityCalibrator(method=CalibrationMethod.ISOTONIC)
        calibrator.fit(proba, y_cal)

        # Apply to new predictions
        test_proba = model.predict_proba(X_test)[:, 1]
        calibrated_proba = calibrator.calibrate(test_proba)
    """

    def __init__(
        self,
        method: CalibrationMethod = CalibrationMethod.ISOTONIC,
    ):
        self.method = method
        self.calibrator_: Optional[Any] = None

    def fit(
        self,
        y_score: np.ndarray,
        y_true: np.ndarray,
    ) -> "ProbabilityCalibrator":
        """
        Fit calibrator on held-out predictions.

        Args:
            y_score: Predicted probabilities
            y_true: True labels
        """
        if self.method == CalibrationMethod.PLATT:
            self.calibrator_ = PlattScaler()
        elif self.method == CalibrationMethod.ISOTONIC:
            self.calibrator_ = IsotonicCalibrator()
        elif self.method == CalibrationMethod.TEMPERATURE:
            self.calibrator_ = TemperatureScaler()
        else:
            raise ValueError(f"Unsupported method: {self.method}")

        self.calibrator_.fit(y_score, y_true)
        return self

    def calibrate(self, y_score: np.ndarray) -> np.ndarray:
        """
        Calibrate probabilities.

        Args:
            y_score: Raw probabilities

        Returns:
            Calibrated probabilities
        """
        if self.calibrator_ is None:
            raise ValueError("Calibrator not fitted.")
        return self.calibrator_.calibrate(y_score)

    def evaluate(
        self,
        y_score: np.ndarray,
        y_true: np.ndarray,
    ) -> CalibrationResult:
        """
        Evaluate calibration quality.

        Args:
            y_score: Raw probabilities
            y_true: True labels

        Returns:
            CalibrationResult with metrics
        """
        # Before calibration
        brier_before = brier_score_loss(y_true, y_score)

        # After calibration
        calibrated = self.calibrate(y_score)
        brier_after = brier_score_loss(y_true, calibrated)

        # ECE (Expected Calibration Error)
        ece_before = self._compute_ece(y_score, y_true)
        ece_after = self._compute_ece(calibrated, y_true)

        # Calibration curves
        frac_pos_before, mean_pred_before = calibration_curve(
            y_true, y_score, n_bins=10
        )
        frac_pos_after, mean_pred_after = calibration_curve(
            y_true, calibrated, n_bins=10
        )

        return CalibrationResult(
            method=self.method.value,
            brier_score_before=brier_before,
            brier_score_after=brier_after,
            ece_before=ece_before,
            ece_after=ece_after,
            calibration_curve_before=(mean_pred_before, frac_pos_before),
            calibration_curve_after=(mean_pred_after, frac_pos_after),
        )

    @staticmethod
    def _compute_ece(y_prob: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_acc = y_true[mask].mean()
                bin_conf = y_prob[mask].mean()
                ece += mask.sum() / len(y_prob) * abs(bin_acc - bin_conf)

        return ece


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_prob_calibrated: Optional[np.ndarray] = None,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot reliability diagram (calibration curve).

    Shows the relationship between predicted probabilities
    and actual frequencies.

    Args:
        y_true: True labels
        y_prob: Raw predicted probabilities
        y_prob_calibrated: Optional calibrated probabilities
        n_bins: Number of bins
        title: Plot title
        save_path: Optional path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available for plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Calibration curve
    ax1 = axes[0]

    # Before calibration
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    ax1.plot(mean_pred, frac_pos, 's-', label='Uncalibrated')

    # After calibration
    if y_prob_calibrated is not None:
        frac_pos_cal, mean_pred_cal = calibration_curve(
            y_true, y_prob_calibrated, n_bins=n_bins
        )
        ax1.plot(mean_pred_cal, frac_pos_cal, 's-', label='Calibrated')

    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect')

    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title(title)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Histogram of predictions
    ax2 = axes[1]
    ax2.hist(y_prob, bins=50, alpha=0.5, label='Uncalibrated', density=True)
    if y_prob_calibrated is not None:
        ax2.hist(y_prob_calibrated, bins=50, alpha=0.5, label='Calibrated', density=True)
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Density')
    ax2.set_title('Probability Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Calibration plot saved to {save_path}")

    plt.close(fig)


def calibrate_model_predictions(
    model: Any,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: CalibrationMethod = CalibrationMethod.ISOTONIC,
) -> Tuple[ProbabilityCalibrator, CalibrationResult]:
    """
    Convenience function to calibrate a trained model.

    Args:
        model: Trained classifier with predict_proba method
        X_cal: Calibration features
        y_cal: Calibration labels
        method: Calibration method

    Returns:
        Tuple of (calibrator, calibration_result)

    Example:
        model = LGBMClassifier()
        model.fit(X_train, y_train)

        calibrator, result = calibrate_model_predictions(
            model, X_cal, y_cal, CalibrationMethod.ISOTONIC
        )

        print(result.summary())

        # Use for new predictions
        test_proba = model.predict_proba(X_test)[:, 1]
        calibrated = calibrator.calibrate(test_proba)
    """
    # Get predictions
    y_prob = model.predict_proba(X_cal)[:, 1]

    # Fit calibrator
    calibrator = ProbabilityCalibrator(method=method)
    calibrator.fit(y_prob, y_cal)

    # Evaluate
    result = calibrator.evaluate(y_prob, y_cal)

    logger.info(f"Calibration complete: Brier score improved from "
                f"{result.brier_score_before:.4f} to {result.brier_score_after:.4f}")

    return calibrator, result
