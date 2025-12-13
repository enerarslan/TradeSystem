"""
Probability Calibration System
==============================
JPMorgan-Level ML Probability Calibration

ML models (CatBoost, XGBoost, etc.) output probabilities that are often
miscalibrated - a predicted 70% doesn't mean 70% of such predictions are correct.

This module provides:
1. Isotonic Regression calibration
2. Platt Scaling (sigmoid)
3. Temperature Scaling
4. Histogram binning
5. Beta calibration
6. Reliability diagram analysis
7. Expected Calibration Error (ECE) computation

Why Calibration Matters for Trading:
- Kelly criterion assumes calibrated probabilities
- Overconfident models lead to oversized positions
- Underconfident models leave money on the table
- Position sizing is DIRECTLY proportional to predicted edge

Author: AlphaTrade Institutional System
Based on: ARCHITECTURAL_REVIEW_REPORT.md - Phase 3
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import pickle
from pathlib import Path

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from scipy.optimize import minimize
from scipy.special import expit, logit
import warnings

from ..utils.logger import get_logger

logger = get_logger(__name__)


class CalibrationMethod(Enum):
    """Available calibration methods"""
    ISOTONIC = "isotonic"
    PLATT = "platt"
    TEMPERATURE = "temperature"
    HISTOGRAM = "histogram"
    BETA = "beta"


@dataclass
class CalibrationMetrics:
    """Calibration quality metrics"""
    expected_calibration_error: float  # ECE
    maximum_calibration_error: float  # MCE
    brier_score: float
    log_loss: float
    reliability_diagram: Dict[str, np.ndarray]
    n_samples: int
    method: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ece': self.expected_calibration_error,
            'mce': self.maximum_calibration_error,
            'brier_score': self.brier_score,
            'log_loss': self.log_loss,
            'n_samples': self.n_samples,
            'method': self.method
        }


@dataclass
class CalibrationResult:
    """Result of probability calibration"""
    raw_probability: float
    calibrated_probability: float
    confidence_interval: Tuple[float, float]
    method: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'raw': self.raw_probability,
            'calibrated': self.calibrated_probability,
            'ci_lower': self.confidence_interval[0],
            'ci_upper': self.confidence_interval[1],
            'method': self.method
        }


class IsotonicCalibrator:
    """
    Isotonic Regression calibration.

    Non-parametric method that fits a monotonic function to map
    raw probabilities to calibrated probabilities.

    Advantages:
    - No assumptions about probability distribution
    - Handles complex miscalibration patterns
    - Works well with sufficient data

    Disadvantages:
    - Can overfit with small datasets
    - May produce step functions
    """

    def __init__(self, out_of_bounds: str = 'clip'):
        """
        Args:
            out_of_bounds: How to handle out-of-bounds predictions
                           'clip' or 'nan'
        """
        self.out_of_bounds = out_of_bounds
        self._model = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds=out_of_bounds
        )
        self._fitted = False

    def fit(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray
    ) -> 'IsotonicCalibrator':
        """
        Fit isotonic regression calibrator.

        Args:
            y_prob: Raw model probabilities
            y_true: True binary labels (0 or 1)
        """
        y_prob = np.asarray(y_prob).flatten()
        y_true = np.asarray(y_true).flatten()

        # Validate
        if len(y_prob) != len(y_true):
            raise ValueError("y_prob and y_true must have same length")

        if len(y_prob) < 10:
            logger.warning("Very few samples for isotonic calibration")

        self._model.fit(y_prob, y_true)
        self._fitted = True

        logger.info(f"Isotonic calibrator fitted on {len(y_prob)} samples")
        return self

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Calibrate probabilities.

        Args:
            y_prob: Raw probabilities

        Returns:
            Calibrated probabilities
        """
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        y_prob = np.asarray(y_prob)
        original_shape = y_prob.shape
        y_prob_flat = y_prob.flatten()

        calibrated = self._model.predict(y_prob_flat)

        return calibrated.reshape(original_shape)

    def __call__(self, y_prob: np.ndarray) -> np.ndarray:
        return self.calibrate(y_prob)


class PlattCalibrator:
    """
    Platt Scaling (sigmoid calibration).

    Fits a logistic regression to map raw probabilities to calibrated ones:
    P_calibrated = 1 / (1 + exp(-(a * logit(P_raw) + b)))

    Advantages:
    - Robust with small datasets
    - Smooth transformation
    - Works well when miscalibration is monotonic

    Disadvantages:
    - Assumes sigmoid relationship
    - May not capture complex miscalibration
    """

    def __init__(self, max_iter: int = 1000):
        self.max_iter = max_iter
        self._model = LogisticRegression(
            C=1e10,  # Essentially no regularization
            solver='lbfgs',
            max_iter=max_iter
        )
        self._fitted = False

    def fit(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray
    ) -> 'PlattCalibrator':
        """
        Fit Platt scaling calibrator.

        Args:
            y_prob: Raw model probabilities
            y_true: True binary labels
        """
        y_prob = np.asarray(y_prob).flatten()
        y_true = np.asarray(y_true).flatten()

        # Clip to avoid log(0)
        y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)

        # Transform to logit space
        X = logit(y_prob).reshape(-1, 1)

        self._model.fit(X, y_true)
        self._fitted = True

        logger.info(f"Platt calibrator fitted on {len(y_prob)} samples")
        return self

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """Calibrate probabilities using Platt scaling"""
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        y_prob = np.asarray(y_prob)
        original_shape = y_prob.shape
        y_prob_flat = y_prob.flatten()

        # Clip and transform
        y_prob_flat = np.clip(y_prob_flat, 1e-10, 1 - 1e-10)
        X = logit(y_prob_flat).reshape(-1, 1)

        calibrated = self._model.predict_proba(X)[:, 1]

        return calibrated.reshape(original_shape)

    def __call__(self, y_prob: np.ndarray) -> np.ndarray:
        return self.calibrate(y_prob)


class TemperatureCalibrator:
    """
    Temperature Scaling calibration.

    Simple method that divides logits by a learned temperature parameter:
    P_calibrated = softmax(logit(P_raw) / T)

    Common in neural networks. A temperature > 1 "softens" probabilities,
    temperature < 1 "sharpens" them.

    Advantages:
    - Single parameter (less overfitting risk)
    - Fast to fit
    - Preserves ranking of predictions

    Disadvantages:
    - Limited flexibility
    - Assumes uniform miscalibration
    """

    def __init__(self, init_temperature: float = 1.0):
        self.init_temperature = init_temperature
        self.temperature: float = init_temperature
        self._fitted = False

    def _nll_loss(self, temperature: float, logits: np.ndarray, y_true: np.ndarray) -> float:
        """Negative log-likelihood loss for temperature optimization"""
        # Scale logits
        scaled = logits / temperature
        probs = expit(scaled)
        probs = np.clip(probs, 1e-10, 1 - 1e-10)

        # Binary cross-entropy
        loss = -(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs))
        return np.mean(loss)

    def fit(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray
    ) -> 'TemperatureCalibrator':
        """Fit temperature scaling"""
        y_prob = np.asarray(y_prob).flatten()
        y_true = np.asarray(y_true).flatten()

        # Clip and convert to logits
        y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)
        logits = logit(y_prob)

        # Optimize temperature
        result = minimize(
            fun=lambda t: self._nll_loss(t[0], logits, y_true),
            x0=[self.init_temperature],
            method='L-BFGS-B',
            bounds=[(0.1, 10.0)]
        )

        self.temperature = result.x[0]
        self._fitted = True

        logger.info(f"Temperature calibrator: T = {self.temperature:.4f}")
        return self

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """Calibrate using temperature scaling"""
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted")

        y_prob = np.asarray(y_prob)
        y_prob_clipped = np.clip(y_prob, 1e-10, 1 - 1e-10)

        # Scale in logit space
        logits = logit(y_prob_clipped)
        scaled = logits / self.temperature

        return expit(scaled)

    def __call__(self, y_prob: np.ndarray) -> np.ndarray:
        return self.calibrate(y_prob)


class HistogramCalibrator:
    """
    Histogram binning calibration.

    Divides probability space into bins and replaces predicted
    probability with the observed frequency in each bin.

    Advantages:
    - Simple and interpretable
    - Works with any distribution

    Disadvantages:
    - Discontinuous output
    - Bin choice affects results
    """

    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins
        self._bin_edges: Optional[np.ndarray] = None
        self._bin_values: Optional[np.ndarray] = None
        self._fitted = False

    def fit(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray
    ) -> 'HistogramCalibrator':
        """Fit histogram binning"""
        y_prob = np.asarray(y_prob).flatten()
        y_true = np.asarray(y_true).flatten()

        # Create bins
        self._bin_edges = np.linspace(0, 1, self.n_bins + 1)
        self._bin_values = np.zeros(self.n_bins)

        for i in range(self.n_bins):
            mask = (y_prob >= self._bin_edges[i]) & (y_prob < self._bin_edges[i + 1])
            if mask.sum() > 0:
                self._bin_values[i] = y_true[mask].mean()
            else:
                # Empty bin: use midpoint
                self._bin_values[i] = (self._bin_edges[i] + self._bin_edges[i + 1]) / 2

        self._fitted = True
        logger.info(f"Histogram calibrator fitted with {self.n_bins} bins")
        return self

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """Calibrate using histogram binning"""
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted")

        y_prob = np.asarray(y_prob)
        original_shape = y_prob.shape
        y_prob_flat = y_prob.flatten()

        # Find bin for each probability
        bin_indices = np.digitize(y_prob_flat, self._bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        calibrated = self._bin_values[bin_indices]

        return calibrated.reshape(original_shape)

    def __call__(self, y_prob: np.ndarray) -> np.ndarray:
        return self.calibrate(y_prob)


class BetaCalibrator:
    """
    Beta calibration.

    Uses a beta distribution to model the relationship between
    raw and calibrated probabilities. More flexible than Platt
    scaling but with more parameters.

    Model: P_cal = 1 / (1 + exp(-a) * (P_raw^b / (1 - P_raw)^c))
    """

    def __init__(self):
        self.a: float = 0.0
        self.b: float = 1.0
        self.c: float = 1.0
        self._fitted = False

    def _loss(
        self,
        params: np.ndarray,
        y_prob: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """Loss function for beta calibration"""
        a, b, c = params

        # Clip to avoid numerical issues
        y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)

        # Beta calibration formula
        log_odds = b * np.log(y_prob) - c * np.log(1 - y_prob) + a
        calibrated = expit(log_odds)
        calibrated = np.clip(calibrated, 1e-10, 1 - 1e-10)

        # Binary cross-entropy
        loss = -(y_true * np.log(calibrated) + (1 - y_true) * np.log(1 - calibrated))
        return np.mean(loss)

    def fit(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray
    ) -> 'BetaCalibrator':
        """Fit beta calibration"""
        y_prob = np.asarray(y_prob).flatten()
        y_true = np.asarray(y_true).flatten()

        # Optimize parameters
        result = minimize(
            fun=lambda p: self._loss(p, y_prob, y_true),
            x0=[0.0, 1.0, 1.0],
            method='L-BFGS-B',
            bounds=[(-10, 10), (0.01, 10), (0.01, 10)]
        )

        self.a, self.b, self.c = result.x
        self._fitted = True

        logger.info(f"Beta calibrator: a={self.a:.4f}, b={self.b:.4f}, c={self.c:.4f}")
        return self

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """Calibrate using beta calibration"""
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted")

        y_prob = np.asarray(y_prob)
        y_prob_clipped = np.clip(y_prob, 1e-10, 1 - 1e-10)

        log_odds = self.b * np.log(y_prob_clipped) - self.c * np.log(1 - y_prob_clipped) + self.a
        return expit(log_odds)

    def __call__(self, y_prob: np.ndarray) -> np.ndarray:
        return self.calibrate(y_prob)


class EnsembleCalibrator:
    """
    Ensemble of multiple calibration methods.

    Combines multiple calibrators with learned weights to
    produce a robust calibrated probability.
    """

    def __init__(
        self,
        methods: List[CalibrationMethod] = None,
        weights: Optional[List[float]] = None
    ):
        if methods is None:
            methods = [
                CalibrationMethod.ISOTONIC,
                CalibrationMethod.PLATT,
                CalibrationMethod.TEMPERATURE
            ]

        self.methods = methods
        self.weights = weights
        self._calibrators: Dict[str, Any] = {}
        self._learned_weights: Optional[np.ndarray] = None
        self._fitted = False

    def _create_calibrator(self, method: CalibrationMethod):
        """Create calibrator instance"""
        if method == CalibrationMethod.ISOTONIC:
            return IsotonicCalibrator()
        elif method == CalibrationMethod.PLATT:
            return PlattCalibrator()
        elif method == CalibrationMethod.TEMPERATURE:
            return TemperatureCalibrator()
        elif method == CalibrationMethod.HISTOGRAM:
            return HistogramCalibrator()
        elif method == CalibrationMethod.BETA:
            return BetaCalibrator()
        else:
            raise ValueError(f"Unknown method: {method}")

    def fit(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        val_prob: Optional[np.ndarray] = None,
        val_true: Optional[np.ndarray] = None
    ) -> 'EnsembleCalibrator':
        """
        Fit ensemble calibrator.

        If validation data provided, learns optimal weights.
        """
        # Fit each calibrator
        for method in self.methods:
            calibrator = self._create_calibrator(method)
            calibrator.fit(y_prob, y_true)
            self._calibrators[method.value] = calibrator

        # Learn weights if validation data provided
        if val_prob is not None and val_true is not None:
            self._learn_weights(val_prob, val_true)
        elif self.weights is not None:
            self._learned_weights = np.array(self.weights)
            self._learned_weights /= self._learned_weights.sum()
        else:
            # Equal weights
            self._learned_weights = np.ones(len(self.methods)) / len(self.methods)

        self._fitted = True
        logger.info(f"Ensemble calibrator fitted with {len(self.methods)} methods")
        return self

    def _learn_weights(self, val_prob: np.ndarray, val_true: np.ndarray) -> None:
        """Learn optimal weights on validation set"""
        # Get calibrated predictions from each method
        predictions = []
        for method in self.methods:
            calibrator = self._calibrators[method.value]
            predictions.append(calibrator.calibrate(val_prob))

        predictions = np.array(predictions).T  # (n_samples, n_methods)

        # Optimize weights to minimize Brier score
        def loss(weights):
            weights = np.abs(weights)
            weights /= weights.sum()
            combined = (predictions * weights).sum(axis=1)
            return np.mean((combined - val_true) ** 2)

        result = minimize(
            fun=loss,
            x0=np.ones(len(self.methods)),
            method='L-BFGS-B',
            bounds=[(0.01, 10)] * len(self.methods)
        )

        self._learned_weights = np.abs(result.x)
        self._learned_weights /= self._learned_weights.sum()

        logger.info(f"Learned ensemble weights: {dict(zip([m.value for m in self.methods], self._learned_weights))}")

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """Calibrate using ensemble"""
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted")

        y_prob = np.asarray(y_prob)
        original_shape = y_prob.shape
        y_prob_flat = y_prob.flatten()

        # Get predictions from each method
        predictions = []
        for method in self.methods:
            calibrator = self._calibrators[method.value]
            predictions.append(calibrator.calibrate(y_prob_flat))

        predictions = np.array(predictions).T

        # Weighted average
        calibrated = (predictions * self._learned_weights).sum(axis=1)

        return calibrated.reshape(original_shape)

    def __call__(self, y_prob: np.ndarray) -> np.ndarray:
        return self.calibrate(y_prob)


# =============================================================================
# CALIBRATION METRICS AND ANALYSIS
# =============================================================================

def compute_calibration_metrics(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
    method_name: str = "unknown"
) -> CalibrationMetrics:
    """
    Compute comprehensive calibration metrics.

    Args:
        y_prob: Predicted probabilities
        y_true: True binary labels
        n_bins: Number of bins for ECE/MCE
        method_name: Name of calibration method

    Returns:
        CalibrationMetrics object
    """
    y_prob = np.asarray(y_prob).flatten()
    y_true = np.asarray(y_true).flatten()

    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        bin_size = mask.sum()

        if bin_size > 0:
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_prob[mask].mean()
            bin_error = abs(bin_accuracy - bin_confidence)

            ece += (bin_size / len(y_prob)) * bin_error
            mce = max(mce, bin_error)

            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            bin_counts.append(bin_size)
        else:
            bin_accuracies.append(np.nan)
            bin_confidences.append(np.nan)
            bin_counts.append(0)

    # Brier score
    brier_score = np.mean((y_prob - y_true) ** 2)

    # Log loss
    y_prob_clipped = np.clip(y_prob, 1e-10, 1 - 1e-10)
    log_loss = -np.mean(y_true * np.log(y_prob_clipped) + (1 - y_true) * np.log(1 - y_prob_clipped))

    # Reliability diagram data
    reliability_diagram = {
        'bin_edges': bin_boundaries,
        'bin_accuracies': np.array(bin_accuracies),
        'bin_confidences': np.array(bin_confidences),
        'bin_counts': np.array(bin_counts)
    }

    return CalibrationMetrics(
        expected_calibration_error=ece,
        maximum_calibration_error=mce,
        brier_score=brier_score,
        log_loss=log_loss,
        reliability_diagram=reliability_diagram,
        n_samples=len(y_prob),
        method=method_name
    )


def compare_calibration_methods(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    val_prob: np.ndarray,
    val_true: np.ndarray
) -> Dict[str, CalibrationMetrics]:
    """
    Compare all calibration methods and return metrics.

    Args:
        y_prob: Training probabilities
        y_true: Training labels
        val_prob: Validation probabilities
        val_true: Validation labels

    Returns:
        Dictionary of method -> CalibrationMetrics
    """
    results = {}

    # Uncalibrated baseline
    results['uncalibrated'] = compute_calibration_metrics(
        val_prob, val_true, method_name='uncalibrated'
    )

    # Each method
    methods = [
        ('isotonic', IsotonicCalibrator()),
        ('platt', PlattCalibrator()),
        ('temperature', TemperatureCalibrator()),
        ('histogram', HistogramCalibrator()),
        ('beta', BetaCalibrator())
    ]

    for name, calibrator in methods:
        try:
            calibrator.fit(y_prob, y_true)
            calibrated_val = calibrator.calibrate(val_prob)
            results[name] = compute_calibration_metrics(
                calibrated_val, val_true, method_name=name
            )
        except Exception as e:
            logger.warning(f"Failed to fit {name} calibrator: {e}")

    return results


# =============================================================================
# MAIN CALIBRATION MANAGER
# =============================================================================

class ProbabilityCalibrationManager:
    """
    Main calibration manager for the trading system.

    Handles:
    - Selecting best calibration method
    - Cross-validation for calibration
    - Per-model calibration
    - Integration with Kelly sizing

    Usage:
        manager = ProbabilityCalibrationManager()
        manager.fit(raw_probs, true_labels)

        calibrated = manager.calibrate(new_raw_prob)
    """

    def __init__(
        self,
        method: CalibrationMethod = CalibrationMethod.ISOTONIC,
        auto_select: bool = True,
        n_cv_folds: int = 5
    ):
        """
        Args:
            method: Default calibration method
            auto_select: Automatically select best method
            n_cv_folds: Cross-validation folds for method selection
        """
        self.method = method
        self.auto_select = auto_select
        self.n_cv_folds = n_cv_folds

        self._calibrator: Optional[Any] = None
        self._metrics: Optional[CalibrationMetrics] = None
        self._fitted = False
        self._fit_timestamp: Optional[datetime] = None

    def fit(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        model_name: str = "default"
    ) -> 'ProbabilityCalibrationManager':
        """
        Fit calibration model.

        Args:
            y_prob: Raw model probabilities
            y_true: True labels
            model_name: Name of the model being calibrated
        """
        y_prob = np.asarray(y_prob).flatten()
        y_true = np.asarray(y_true).flatten()

        logger.info(f"Fitting calibration for model '{model_name}' with {len(y_prob)} samples")

        if self.auto_select:
            # Split for selection
            n = len(y_prob)
            split = int(0.7 * n)
            indices = np.random.permutation(n)

            train_idx = indices[:split]
            val_idx = indices[split:]

            train_prob, train_true = y_prob[train_idx], y_true[train_idx]
            val_prob, val_true = y_prob[val_idx], y_true[val_idx]

            # Compare methods
            comparison = compare_calibration_methods(
                train_prob, train_true, val_prob, val_true
            )

            # Select best by ECE
            best_method = min(
                [(k, v.expected_calibration_error) for k, v in comparison.items() if k != 'uncalibrated'],
                key=lambda x: x[1]
            )[0]

            logger.info(f"Auto-selected calibration method: {best_method}")
            logger.info(f"ECE improvement: {comparison['uncalibrated'].expected_calibration_error:.4f} -> {comparison[best_method].expected_calibration_error:.4f}")

            self.method = CalibrationMethod(best_method)

        # Create and fit final calibrator
        self._calibrator = self._create_calibrator(self.method)
        self._calibrator.fit(y_prob, y_true)

        # Compute final metrics
        calibrated = self._calibrator.calibrate(y_prob)
        self._metrics = compute_calibration_metrics(
            calibrated, y_true, method_name=self.method.value
        )

        self._fitted = True
        self._fit_timestamp = datetime.now()

        return self

    def _create_calibrator(self, method: CalibrationMethod):
        """Create calibrator for method"""
        if method == CalibrationMethod.ISOTONIC:
            return IsotonicCalibrator()
        elif method == CalibrationMethod.PLATT:
            return PlattCalibrator()
        elif method == CalibrationMethod.TEMPERATURE:
            return TemperatureCalibrator()
        elif method == CalibrationMethod.HISTOGRAM:
            return HistogramCalibrator()
        elif method == CalibrationMethod.BETA:
            return BetaCalibrator()
        else:
            raise ValueError(f"Unknown method: {method}")

    def calibrate(
        self,
        y_prob: Union[float, np.ndarray],
        with_confidence: bool = False
    ) -> Union[float, np.ndarray, CalibrationResult]:
        """
        Calibrate probability(ies).

        Args:
            y_prob: Raw probability or array
            with_confidence: Return CalibrationResult with CI

        Returns:
            Calibrated probability or CalibrationResult
        """
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        is_scalar = np.isscalar(y_prob)
        y_prob_arr = np.atleast_1d(y_prob)

        calibrated = self._calibrator.calibrate(y_prob_arr)

        if with_confidence:
            # Bootstrap confidence interval (simplified)
            # In production, use stored bootstrap samples
            ci_lower = np.clip(calibrated - 0.05, 0, 1)
            ci_upper = np.clip(calibrated + 0.05, 0, 1)

            if is_scalar:
                return CalibrationResult(
                    raw_probability=float(y_prob),
                    calibrated_probability=float(calibrated[0]),
                    confidence_interval=(float(ci_lower[0]), float(ci_upper[0])),
                    method=self.method.value
                )
            else:
                return [CalibrationResult(
                    raw_probability=float(r),
                    calibrated_probability=float(c),
                    confidence_interval=(float(l), float(u)),
                    method=self.method.value
                ) for r, c, l, u in zip(y_prob_arr, calibrated, ci_lower, ci_upper)]

        if is_scalar:
            return float(calibrated[0])
        return calibrated

    def __call__(self, y_prob: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.calibrate(y_prob)

    def get_metrics(self) -> Optional[CalibrationMetrics]:
        """Get calibration metrics"""
        return self._metrics

    def save(self, filepath: Union[str, Path]) -> None:
        """Save calibration model"""
        filepath = Path(filepath)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'calibrator': self._calibrator,
                'method': self.method,
                'metrics': self._metrics,
                'fit_timestamp': self._fit_timestamp
            }, f)
        logger.info(f"Calibration model saved to {filepath}")

    def load(self, filepath: Union[str, Path]) -> 'ProbabilityCalibrationManager':
        """Load calibration model"""
        filepath = Path(filepath)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self._calibrator = data['calibrator']
        self.method = data['method']
        self._metrics = data['metrics']
        self._fit_timestamp = data['fit_timestamp']
        self._fitted = True

        logger.info(f"Calibration model loaded from {filepath}")
        return self


# =============================================================================
# INTEGRATION WITH BAYESIAN KELLY
# =============================================================================

class CalibratedBayesianKelly:
    """
    Bayesian Kelly with integrated probability calibration.

    This is the production integration that combines:
    1. Raw ML probability from model
    2. Calibration to fix miscalibration
    3. Bayesian Kelly for position sizing

    Usage:
        kelly = CalibratedBayesianKelly()

        # Train calibration
        kelly.fit_calibration(train_probs, train_labels)

        # Get position size
        raw_prob = model.predict_proba(features)
        size = kelly.calculate_size(
            symbol='AAPL',
            raw_probability=raw_prob,
            portfolio_value=100000,
            current_price=150.0
        )
    """

    def __init__(
        self,
        # Kelly parameters
        kelly_fraction: float = 0.25,
        max_position_pct: float = 0.20,
        min_edge_threshold: float = 0.01,

        # Calibration parameters
        calibration_method: CalibrationMethod = CalibrationMethod.ISOTONIC,
        auto_select_calibration: bool = True
    ):
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.min_edge_threshold = min_edge_threshold

        # Initialize calibration manager
        self._calibrator = ProbabilityCalibrationManager(
            method=calibration_method,
            auto_select=auto_select_calibration
        )

        # Track outcomes for Bayesian updates
        self._wins: int = 2  # Prior
        self._losses: int = 2  # Prior

        self._calibration_fitted = False

    def fit_calibration(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        model_name: str = "default"
    ) -> 'CalibratedBayesianKelly':
        """Fit the calibration model"""
        self._calibrator.fit(y_prob, y_true, model_name)
        self._calibration_fitted = True
        return self

    def record_outcome(self, won: bool) -> None:
        """Record trade outcome for Bayesian updating"""
        if won:
            self._wins += 1
        else:
            self._losses += 1

    def calculate_size(
        self,
        symbol: str,
        raw_probability: float,
        portfolio_value: float,
        current_price: float,
        signal_strength: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculate position size with calibrated probability.

        Args:
            symbol: Trading symbol
            raw_probability: Raw ML model probability
            portfolio_value: Total portfolio value
            current_price: Current price per share
            signal_strength: Signal strength multiplier

        Returns:
            Dictionary with sizing info
        """
        # Step 1: Calibrate probability
        if self._calibration_fitted:
            calibrated_prob = self._calibrator.calibrate(raw_probability)
        else:
            calibrated_prob = raw_probability
            logger.warning("Calibration not fitted, using raw probability")

        # Step 2: Compute Bayesian posterior win rate
        posterior_win_rate = self._wins / (self._wins + self._losses)

        # Step 3: Blend calibrated ML prob with Bayesian posterior
        # Weight by sample size (more trades = trust posterior more)
        n_trades = self._wins + self._losses - 4  # Subtract prior
        blend_weight = min(n_trades / 100, 0.5)  # Max 50% weight on posterior

        effective_prob = (1 - blend_weight) * calibrated_prob + blend_weight * posterior_win_rate

        # Step 4: Compute edge
        # Simplified Kelly: edge = 2p - 1 for even odds
        edge = 2 * effective_prob - 1

        # Step 5: Check minimum edge
        if edge < self.min_edge_threshold:
            return {
                'symbol': symbol,
                'shares': 0,
                'dollars': 0.0,
                'pct_portfolio': 0.0,
                'raw_probability': raw_probability,
                'calibrated_probability': calibrated_prob,
                'effective_probability': effective_prob,
                'edge': edge,
                'kelly_fraction': 0.0,
                'reason': 'insufficient_edge'
            }

        # Step 6: Kelly fraction
        kelly = edge * self.kelly_fraction * signal_strength
        kelly = min(kelly, self.max_position_pct)
        kelly = max(kelly, 0)

        # Step 7: Convert to shares
        dollars = portfolio_value * kelly
        shares = int(dollars / current_price) if current_price > 0 else 0
        actual_dollars = shares * current_price
        actual_pct = actual_dollars / portfolio_value if portfolio_value > 0 else 0

        return {
            'symbol': symbol,
            'shares': shares,
            'dollars': actual_dollars,
            'pct_portfolio': actual_pct,
            'raw_probability': raw_probability,
            'calibrated_probability': calibrated_prob,
            'effective_probability': effective_prob,
            'edge': edge,
            'kelly_fraction': kelly,
            'posterior_win_rate': posterior_win_rate,
            'n_trades': n_trades,
            'reason': 'sized'
        }

    def get_calibration_metrics(self) -> Optional[CalibrationMetrics]:
        """Get calibration quality metrics"""
        return self._calibrator.get_metrics() if self._calibration_fitted else None
