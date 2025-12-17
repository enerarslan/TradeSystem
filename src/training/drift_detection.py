"""
Model Drift Detection Module for AlphaTrade System.

This module provides institutional-grade drift detection to identify when
model performance degrades and retraining is needed.

Implements:
- Population Stability Index (PSI) for feature drift
- Kolmogorov-Smirnov test for distribution comparison
- Jensen-Shannon divergence
- Performance metric tracking (Sharpe ratio degradation, accuracy drop)

Reference:
    "Machine Learning for Asset Managers" by de Prado (2020)
    "Advances in Financial Machine Learning" by de Prado (2018)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

logger = logging.getLogger(__name__)


class DriftSeverity(str, Enum):
    """Severity levels for detected drift."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftType(str, Enum):
    """Types of drift that can be detected."""
    FEATURE = "feature"
    PREDICTION = "prediction"
    PERFORMANCE = "performance"
    CONCEPT = "concept"


class DriftRecommendation(str, Enum):
    """Recommended actions based on drift detection."""
    MONITOR = "monitor"
    INVESTIGATE = "investigate"
    RETRAIN = "retrain"
    ALERT = "alert"


@dataclass
class DriftResult:
    """Container for drift detection results."""
    is_drift_detected: bool
    drift_score: float
    drift_type: DriftType
    affected_features: List[str] = field(default_factory=list)
    severity: DriftSeverity = DriftSeverity.LOW
    recommendation: DriftRecommendation = DriftRecommendation.MONITOR
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "is_drift_detected": self.is_drift_detected,
            "drift_score": self.drift_score,
            "drift_type": self.drift_type.value,
            "affected_features": self.affected_features,
            "severity": self.severity.value,
            "recommendation": self.recommendation.value,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class DriftThresholds:
    """Configurable thresholds for drift detection."""
    # PSI thresholds
    psi_warning: float = 0.10
    psi_critical: float = 0.25

    # KS test thresholds
    ks_warning: float = 0.05
    ks_critical: float = 0.01

    # Jensen-Shannon divergence thresholds
    js_warning: float = 0.10
    js_critical: float = 0.20

    # Performance degradation thresholds
    sharpe_degradation_warning: float = 0.20  # 20% drop
    sharpe_degradation_critical: float = 0.40  # 40% drop
    accuracy_degradation_warning: float = 0.05  # 5% drop
    accuracy_degradation_critical: float = 0.10  # 10% drop


class DriftDetector:
    """
    Comprehensive drift detection for ML models in production.

    Detects:
    1. Feature drift - changes in input feature distributions
    2. Prediction drift - changes in model output distributions
    3. Performance drift - degradation in model metrics over time
    4. Concept drift - changes in the relationship between features and target

    Usage:
        # Initialize with reference data (training data)
        detector = DriftDetector(X_train, y_train, thresholds)

        # Check for drift in production data
        feature_result = detector.detect_feature_drift(X_production)
        prediction_result = detector.detect_prediction_drift(predictions)
        performance_result = detector.detect_performance_drift(current_metrics)

        # Get comprehensive report
        report = detector.get_drift_report()
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        reference_target: Optional[pd.Series] = None,
        thresholds: Optional[DriftThresholds] = None,
        feature_names: Optional[List[str]] = None,
        n_bins: int = 10,
    ):
        """
        Initialize drift detector with reference (training) data.

        Args:
            reference_data: Training/reference feature data
            reference_target: Training/reference target values
            thresholds: Custom thresholds for drift detection
            feature_names: Names of features to monitor
            n_bins: Number of bins for PSI calculation
        """
        self.reference_data = reference_data
        self.reference_target = reference_target
        self.thresholds = thresholds or DriftThresholds()
        self.n_bins = n_bins

        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(reference_data, pd.DataFrame):
            self.feature_names = reference_data.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(reference_data.shape[1])]

        # Pre-compute reference statistics
        self._reference_stats = self._compute_statistics(reference_data)
        self._reference_bins = self._compute_bins(reference_data)

        # Store drift history for tracking
        self._drift_history: List[DriftResult] = []
        self._performance_history: List[Dict[str, float]] = []

        logger.info(
            f"DriftDetector initialized with {len(self.feature_names)} features, "
            f"{len(reference_data)} reference samples"
        )

    def _compute_statistics(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics for each feature."""
        stats_dict = {}
        df = pd.DataFrame(data)

        for i, col in enumerate(df.columns):
            col_data = df.iloc[:, i].dropna()
            stats_dict[self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"] = {
                "mean": col_data.mean(),
                "std": col_data.std(),
                "min": col_data.min(),
                "max": col_data.max(),
                "median": col_data.median(),
                "skew": col_data.skew(),
                "kurtosis": col_data.kurtosis(),
            }

        return stats_dict

    def _compute_bins(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Compute histogram bins for each feature."""
        bins_dict = {}
        df = pd.DataFrame(data)

        for i, col in enumerate(df.columns):
            col_data = df.iloc[:, i].dropna()
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"

            # Create bins based on percentiles for robustness
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            bins_dict[feature_name] = np.percentile(col_data, percentiles)

        return bins_dict

    def calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI measures the shift in distribution between two datasets.
        PSI < 0.10: No significant change
        0.10 <= PSI < 0.25: Some change, worth monitoring
        PSI >= 0.25: Significant change, action required

        Args:
            reference: Reference distribution values
            current: Current distribution values
            bins: Bin edges for histogram

        Returns:
            PSI value
        """
        # Handle NaN values
        reference = np.array(reference)
        current = np.array(current)
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]

        if len(reference) == 0 or len(current) == 0:
            return 0.0

        # Create bins if not provided
        if bins is None:
            # Use percentile-based bins from combined data
            combined = np.concatenate([reference, current])
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            bins = np.percentile(combined, percentiles)

        # Calculate histograms
        ref_counts, _ = np.histogram(reference, bins=bins)
        cur_counts, _ = np.histogram(current, bins=bins)

        # Convert to proportions (add small constant to avoid division by zero)
        epsilon = 1e-10
        ref_pct = (ref_counts + epsilon) / (ref_counts.sum() + epsilon * len(ref_counts))
        cur_pct = (cur_counts + epsilon) / (cur_counts.sum() + epsilon * len(cur_counts))

        # Calculate PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

        return float(psi)

    def calculate_ks_statistic(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Calculate Kolmogorov-Smirnov statistic and p-value.

        Tests if two samples come from the same distribution.

        Args:
            reference: Reference distribution values
            current: Current distribution values

        Returns:
            Tuple of (KS statistic, p-value)
        """
        reference = np.array(reference)
        current = np.array(current)
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]

        if len(reference) < 10 or len(current) < 10:
            return 0.0, 1.0

        statistic, p_value = stats.ks_2samp(reference, current)
        return float(statistic), float(p_value)

    def calculate_js_divergence(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculate Jensen-Shannon divergence.

        Symmetric measure of distribution difference (0 to 1).

        Args:
            reference: Reference distribution values
            current: Current distribution values
            bins: Bin edges for histogram

        Returns:
            JS divergence value
        """
        reference = np.array(reference)
        current = np.array(current)
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]

        if len(reference) == 0 or len(current) == 0:
            return 0.0

        # Create bins if not provided
        if bins is None:
            combined = np.concatenate([reference, current])
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            bins = np.percentile(combined, percentiles)

        # Calculate histograms
        ref_counts, _ = np.histogram(reference, bins=bins)
        cur_counts, _ = np.histogram(current, bins=bins)

        # Normalize to probability distributions
        epsilon = 1e-10
        ref_prob = (ref_counts + epsilon) / (ref_counts.sum() + epsilon * len(ref_counts))
        cur_prob = (cur_counts + epsilon) / (cur_counts.sum() + epsilon * len(cur_counts))

        # Calculate JS divergence
        js_div = jensenshannon(ref_prob, cur_prob)

        return float(js_div)

    def detect_feature_drift(
        self,
        current_data: pd.DataFrame,
        features_to_check: Optional[List[str]] = None,
    ) -> DriftResult:
        """
        Detect drift in feature distributions.

        Args:
            current_data: Current production feature data
            features_to_check: Specific features to check (None = all)

        Returns:
            DriftResult with feature drift analysis
        """
        current_df = pd.DataFrame(current_data)

        if features_to_check is None:
            features_to_check = self.feature_names

        psi_scores = {}
        ks_results = {}
        js_scores = {}
        drifted_features = []

        for i, feature in enumerate(features_to_check):
            if i >= len(self.reference_data.columns) or i >= len(current_df.columns):
                continue

            ref_values = self.reference_data.iloc[:, i].values
            cur_values = current_df.iloc[:, i].values

            # Get bins for this feature
            bins = self._reference_bins.get(feature)

            # Calculate drift metrics
            psi = self.calculate_psi(ref_values, cur_values, bins)
            ks_stat, ks_pval = self.calculate_ks_statistic(ref_values, cur_values)
            js_div = self.calculate_js_divergence(ref_values, cur_values, bins)

            psi_scores[feature] = psi
            ks_results[feature] = {"statistic": ks_stat, "p_value": ks_pval}
            js_scores[feature] = js_div

            # Check if feature has drifted
            if (psi > self.thresholds.psi_warning or
                ks_pval < self.thresholds.ks_warning or
                js_div > self.thresholds.js_warning):
                drifted_features.append(feature)

        # Calculate overall drift score (average PSI)
        avg_psi = np.mean(list(psi_scores.values())) if psi_scores else 0.0
        max_psi = max(psi_scores.values()) if psi_scores else 0.0

        # Determine severity
        if max_psi >= self.thresholds.psi_critical:
            severity = DriftSeverity.CRITICAL
            recommendation = DriftRecommendation.RETRAIN
        elif max_psi >= self.thresholds.psi_warning:
            severity = DriftSeverity.HIGH
            recommendation = DriftRecommendation.INVESTIGATE
        elif len(drifted_features) > len(features_to_check) * 0.3:
            severity = DriftSeverity.MEDIUM
            recommendation = DriftRecommendation.INVESTIGATE
        else:
            severity = DriftSeverity.LOW
            recommendation = DriftRecommendation.MONITOR

        result = DriftResult(
            is_drift_detected=len(drifted_features) > 0,
            drift_score=avg_psi,
            drift_type=DriftType.FEATURE,
            affected_features=drifted_features,
            severity=severity,
            recommendation=recommendation,
            details={
                "psi_scores": psi_scores,
                "ks_results": ks_results,
                "js_scores": js_scores,
                "max_psi": max_psi,
                "n_drifted_features": len(drifted_features),
                "n_total_features": len(features_to_check),
            },
        )

        self._drift_history.append(result)
        return result

    def detect_prediction_drift(
        self,
        current_predictions: np.ndarray,
        reference_predictions: Optional[np.ndarray] = None,
    ) -> DriftResult:
        """
        Detect drift in model predictions.

        Args:
            current_predictions: Current model predictions
            reference_predictions: Reference predictions (from validation set)

        Returns:
            DriftResult with prediction drift analysis
        """
        if reference_predictions is None:
            # Use stored reference if available
            if hasattr(self, '_reference_predictions'):
                reference_predictions = self._reference_predictions
            else:
                logger.warning("No reference predictions available for drift detection")
                return DriftResult(
                    is_drift_detected=False,
                    drift_score=0.0,
                    drift_type=DriftType.PREDICTION,
                    severity=DriftSeverity.LOW,
                    recommendation=DriftRecommendation.MONITOR,
                )

        # Calculate drift metrics
        psi = self.calculate_psi(reference_predictions, current_predictions)
        ks_stat, ks_pval = self.calculate_ks_statistic(reference_predictions, current_predictions)
        js_div = self.calculate_js_divergence(reference_predictions, current_predictions)

        # Determine if drift detected
        is_drift = (
            psi > self.thresholds.psi_warning or
            ks_pval < self.thresholds.ks_warning or
            js_div > self.thresholds.js_warning
        )

        # Determine severity
        if psi >= self.thresholds.psi_critical:
            severity = DriftSeverity.CRITICAL
            recommendation = DriftRecommendation.RETRAIN
        elif psi >= self.thresholds.psi_warning:
            severity = DriftSeverity.HIGH
            recommendation = DriftRecommendation.INVESTIGATE
        else:
            severity = DriftSeverity.LOW
            recommendation = DriftRecommendation.MONITOR

        result = DriftResult(
            is_drift_detected=is_drift,
            drift_score=psi,
            drift_type=DriftType.PREDICTION,
            severity=severity,
            recommendation=recommendation,
            details={
                "psi": psi,
                "ks_statistic": ks_stat,
                "ks_p_value": ks_pval,
                "js_divergence": js_div,
            },
        )

        self._drift_history.append(result)
        return result

    def detect_performance_drift(
        self,
        current_metrics: Dict[str, float],
        reference_metrics: Optional[Dict[str, float]] = None,
    ) -> DriftResult:
        """
        Detect degradation in model performance metrics.

        Args:
            current_metrics: Current performance metrics
            reference_metrics: Reference (baseline) metrics

        Returns:
            DriftResult with performance drift analysis
        """
        if reference_metrics is None:
            if hasattr(self, '_reference_metrics'):
                reference_metrics = self._reference_metrics
            else:
                logger.warning("No reference metrics available for drift detection")
                return DriftResult(
                    is_drift_detected=False,
                    drift_score=0.0,
                    drift_type=DriftType.PERFORMANCE,
                    severity=DriftSeverity.LOW,
                    recommendation=DriftRecommendation.MONITOR,
                )

        degradations = {}
        significant_degradations = []

        # Check Sharpe ratio
        if "sharpe_ratio" in current_metrics and "sharpe_ratio" in reference_metrics:
            ref_sharpe = reference_metrics["sharpe_ratio"]
            cur_sharpe = current_metrics["sharpe_ratio"]

            if ref_sharpe != 0:
                sharpe_degradation = (ref_sharpe - cur_sharpe) / abs(ref_sharpe)
                degradations["sharpe_ratio"] = sharpe_degradation

                if sharpe_degradation > self.thresholds.sharpe_degradation_warning:
                    significant_degradations.append("sharpe_ratio")

        # Check accuracy/IC
        for metric in ["accuracy", "ic", "auc"]:
            if metric in current_metrics and metric in reference_metrics:
                ref_val = reference_metrics[metric]
                cur_val = current_metrics[metric]

                if ref_val != 0:
                    degradation = (ref_val - cur_val) / abs(ref_val)
                    degradations[metric] = degradation

                    if degradation > self.thresholds.accuracy_degradation_warning:
                        significant_degradations.append(metric)

        # Calculate overall drift score
        drift_score = np.mean(list(degradations.values())) if degradations else 0.0
        max_degradation = max(degradations.values()) if degradations else 0.0

        # Determine severity
        if max_degradation >= self.thresholds.sharpe_degradation_critical:
            severity = DriftSeverity.CRITICAL
            recommendation = DriftRecommendation.RETRAIN
        elif max_degradation >= self.thresholds.sharpe_degradation_warning:
            severity = DriftSeverity.HIGH
            recommendation = DriftRecommendation.INVESTIGATE
        elif len(significant_degradations) > 0:
            severity = DriftSeverity.MEDIUM
            recommendation = DriftRecommendation.INVESTIGATE
        else:
            severity = DriftSeverity.LOW
            recommendation = DriftRecommendation.MONITOR

        result = DriftResult(
            is_drift_detected=len(significant_degradations) > 0,
            drift_score=drift_score,
            drift_type=DriftType.PERFORMANCE,
            affected_features=significant_degradations,
            severity=severity,
            recommendation=recommendation,
            details={
                "degradations": degradations,
                "current_metrics": current_metrics,
                "reference_metrics": reference_metrics,
                "max_degradation": max_degradation,
            },
        )

        self._drift_history.append(result)
        self._performance_history.append(current_metrics)
        return result

    def set_reference_predictions(self, predictions: np.ndarray) -> None:
        """Store reference predictions for comparison."""
        self._reference_predictions = np.array(predictions)

    def set_reference_metrics(self, metrics: Dict[str, float]) -> None:
        """Store reference metrics for comparison."""
        self._reference_metrics = metrics

    def get_drift_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive drift report.

        Returns:
            Dictionary with complete drift analysis
        """
        if not self._drift_history:
            return {"status": "No drift checks performed"}

        # Aggregate results
        feature_results = [r for r in self._drift_history if r.drift_type == DriftType.FEATURE]
        prediction_results = [r for r in self._drift_history if r.drift_type == DriftType.PREDICTION]
        performance_results = [r for r in self._drift_history if r.drift_type == DriftType.PERFORMANCE]

        # Summary statistics
        report = {
            "summary": {
                "total_checks": len(self._drift_history),
                "drift_detected_count": sum(1 for r in self._drift_history if r.is_drift_detected),
                "critical_count": sum(1 for r in self._drift_history if r.severity == DriftSeverity.CRITICAL),
                "high_count": sum(1 for r in self._drift_history if r.severity == DriftSeverity.HIGH),
            },
            "feature_drift": {
                "checks": len(feature_results),
                "avg_score": np.mean([r.drift_score for r in feature_results]) if feature_results else 0,
                "most_drifted_features": self._get_most_drifted_features(feature_results),
            },
            "prediction_drift": {
                "checks": len(prediction_results),
                "avg_score": np.mean([r.drift_score for r in prediction_results]) if prediction_results else 0,
            },
            "performance_drift": {
                "checks": len(performance_results),
                "avg_score": np.mean([r.drift_score for r in performance_results]) if performance_results else 0,
            },
            "recommendations": self._generate_recommendations(),
            "history": [r.to_dict() for r in self._drift_history[-10:]],  # Last 10 checks
        }

        return report

    def _get_most_drifted_features(
        self,
        feature_results: List[DriftResult],
        top_n: int = 5,
    ) -> List[str]:
        """Get features with highest drift scores."""
        feature_scores = {}

        for result in feature_results:
            if "psi_scores" in result.details:
                for feature, score in result.details["psi_scores"].items():
                    if feature not in feature_scores:
                        feature_scores[feature] = []
                    feature_scores[feature].append(score)

        # Calculate average score per feature
        avg_scores = {f: np.mean(scores) for f, scores in feature_scores.items()}

        # Sort by score and return top N
        sorted_features = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        return [f for f, _ in sorted_features[:top_n]]

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on drift analysis."""
        recommendations = []

        if not self._drift_history:
            return ["Start monitoring by running drift detection periodically"]

        recent_results = self._drift_history[-5:]  # Last 5 checks

        # Check for critical drift
        critical_count = sum(1 for r in recent_results if r.severity == DriftSeverity.CRITICAL)
        if critical_count > 0:
            recommendations.append("URGENT: Critical drift detected - consider immediate model retraining")

        # Check for consistent drift
        drift_count = sum(1 for r in recent_results if r.is_drift_detected)
        if drift_count >= 3:
            recommendations.append("Consistent drift pattern detected - schedule model retraining")

        # Check for feature-specific issues
        all_drifted = []
        for r in recent_results:
            all_drifted.extend(r.affected_features)

        if all_drifted:
            from collections import Counter
            feature_counts = Counter(all_drifted)
            common_drifted = [f for f, c in feature_counts.items() if c >= 2]

            if common_drifted:
                recommendations.append(
                    f"Investigate data pipeline for features: {', '.join(common_drifted[:5])}"
                )

        # Performance-specific recommendations
        perf_results = [r for r in recent_results if r.drift_type == DriftType.PERFORMANCE]
        if perf_results:
            avg_perf_drift = np.mean([r.drift_score for r in perf_results])
            if avg_perf_drift > 0.1:
                recommendations.append("Performance degradation trend detected - review model strategy")

        if not recommendations:
            recommendations.append("System stable - continue regular monitoring")

        return recommendations


def calculate_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    Standalone PSI calculation function.

    Args:
        reference: Reference distribution
        current: Current distribution
        n_bins: Number of bins

    Returns:
        PSI value
    """
    detector = DriftDetector.__new__(DriftDetector)
    detector.n_bins = n_bins
    return detector.calculate_psi(reference, current)
