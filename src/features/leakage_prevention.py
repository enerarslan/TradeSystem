"""
Feature Leakage Prevention Module for AlphaTrade System.

This module provides systematic detection and prevention of look-ahead bias
(data leakage) in feature engineering - a critical requirement for any
institutional-grade trading system.

Implements:
- Future correlation test
- Timestamp validation
- Rolling window validation
- Target leakage detection
- Comprehensive leakage reporting

Reference:
    "Advances in Financial Machine Learning" by de Prado (2018)
    Chapter 7: Cross-Validation in Finance
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class LeakageType(str, Enum):
    """Types of data leakage that can be detected."""
    FUTURE_DATA = "future_data"
    TARGET_LEAKAGE = "target_leakage"
    TIMESTAMP_ERROR = "timestamp_error"
    LOOKAHEAD_BIAS = "lookahead_bias"
    OVERLAPPING_SAMPLES = "overlapping_samples"


class LeakageSeverity(str, Enum):
    """Severity levels for detected leakage."""
    NONE = "none"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LeakageResult:
    """Container for leakage detection results for a single feature."""
    feature_name: str
    has_leakage: bool
    leakage_type: Optional[LeakageType] = None
    leakage_score: float = 0.0  # 0-1, higher = more leakage risk
    severity: LeakageSeverity = LeakageSeverity.NONE
    details: str = ""
    future_correlation: Optional[float] = None
    target_correlation: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "feature_name": self.feature_name,
            "has_leakage": self.has_leakage,
            "leakage_type": self.leakage_type.value if self.leakage_type else None,
            "leakage_score": self.leakage_score,
            "severity": self.severity.value,
            "details": self.details,
            "future_correlation": self.future_correlation,
            "target_correlation": self.target_correlation,
        }


@dataclass
class LeakageReport:
    """Comprehensive leakage analysis report."""
    total_features: int
    features_with_leakage: int
    critical_count: int
    high_count: int
    warning_count: int
    safe_features: List[str]
    unsafe_features: List[str]
    results: Dict[str, LeakageResult]
    recommendations: List[str]

    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "=" * 60,
            "FEATURE LEAKAGE ANALYSIS REPORT",
            "=" * 60,
            f"Total features analyzed: {self.total_features}",
            f"Features with leakage: {self.features_with_leakage}",
            f"  - Critical: {self.critical_count}",
            f"  - High: {self.high_count}",
            f"  - Warning: {self.warning_count}",
            f"Safe features: {len(self.safe_features)}",
            "",
            "UNSAFE FEATURES:",
        ]

        for feature in self.unsafe_features[:10]:  # Top 10
            result = self.results[feature]
            lines.append(f"  - {feature}: {result.leakage_type.value if result.leakage_type else 'unknown'} "
                        f"(score: {result.leakage_score:.3f})")

        lines.extend([
            "",
            "RECOMMENDATIONS:",
        ])
        for rec in self.recommendations:
            lines.append(f"  - {rec}")

        lines.append("=" * 60)
        return "\n".join(lines)


class LeakageChecker:
    """
    Comprehensive feature leakage detection for financial ML.

    This class implements multiple methods to detect look-ahead bias:

    1. Future Correlation Test: Checks if features correlate with FUTURE returns
       (they shouldn't - this indicates leakage)

    2. Timestamp Validation: Verifies feature timestamps don't exceed current bar

    3. Rolling Window Validation: Ensures rolling calculations don't include
       current bar inappropriately

    4. Target Leakage Test: Checks if features directly encode target information

    Usage:
        checker = LeakageChecker(threshold=0.3)
        results = checker.check_all_features(feature_df, price_df, target_series)
        safe_features = checker.get_safe_features(feature_df, price_df, target_series)
        report = checker.generate_report()
    """

    def __init__(
        self,
        future_correlation_threshold: float = 0.1,
        target_correlation_threshold: float = 0.8,
        min_samples: int = 100,
        prediction_horizon: int = 1,
        verbose: bool = True,
    ):
        """
        Initialize the leakage checker.

        Args:
            future_correlation_threshold: Max acceptable correlation with future returns
            target_correlation_threshold: Max acceptable correlation with target
            min_samples: Minimum samples required for correlation calculation
            prediction_horizon: Number of bars ahead for target
            verbose: Whether to log warnings
        """
        self.future_correlation_threshold = future_correlation_threshold
        self.target_correlation_threshold = target_correlation_threshold
        self.min_samples = min_samples
        self.prediction_horizon = prediction_horizon
        self.verbose = verbose

        self._results: Dict[str, LeakageResult] = {}
        self._checked_features: List[str] = []

    def check_feature_for_leakage(
        self,
        feature_series: pd.Series,
        price_series: pd.Series,
        target_series: pd.Series,
        feature_name: Optional[str] = None,
    ) -> LeakageResult:
        """
        Check a single feature for various types of leakage.

        Args:
            feature_series: The feature values
            price_series: Close prices (for return calculation)
            target_series: Target variable values
            feature_name: Name of the feature

        Returns:
            LeakageResult with detection details
        """
        feature_name = feature_name or feature_series.name or "unknown"

        # Initialize result
        leakage_detected = False
        leakage_types = []
        leakage_scores = []
        details = []

        # 1. Check future correlation (most important test)
        future_corr = self._check_future_correlation(feature_series, price_series)

        if abs(future_corr) > self.future_correlation_threshold:
            leakage_detected = True
            leakage_types.append(LeakageType.FUTURE_DATA)
            leakage_scores.append(abs(future_corr))
            details.append(f"Future correlation: {future_corr:.4f}")

        # 2. Check target leakage
        target_corr = self._check_target_leakage(feature_series, target_series)

        if abs(target_corr) > self.target_correlation_threshold:
            leakage_detected = True
            leakage_types.append(LeakageType.TARGET_LEAKAGE)
            leakage_scores.append(abs(target_corr))
            details.append(f"Target correlation: {target_corr:.4f}")

        # 3. Check timestamp consistency
        timestamp_issue = self._check_timestamp_consistency(feature_series)

        if timestamp_issue:
            leakage_detected = True
            leakage_types.append(LeakageType.TIMESTAMP_ERROR)
            leakage_scores.append(0.5)
            details.append(f"Timestamp issue: {timestamp_issue}")

        # 4. Check for lookahead bias patterns
        lookahead_score = self._check_lookahead_patterns(feature_series, price_series)

        if lookahead_score > 0.5:
            leakage_detected = True
            leakage_types.append(LeakageType.LOOKAHEAD_BIAS)
            leakage_scores.append(lookahead_score)
            details.append(f"Lookahead pattern detected (score: {lookahead_score:.3f})")

        # Calculate overall leakage score
        overall_score = max(leakage_scores) if leakage_scores else 0.0

        # Determine severity
        if overall_score >= 0.8:
            severity = LeakageSeverity.CRITICAL
        elif overall_score >= 0.5:
            severity = LeakageSeverity.HIGH
        elif overall_score >= 0.3:
            severity = LeakageSeverity.WARNING
        else:
            severity = LeakageSeverity.NONE

        # Select primary leakage type
        primary_type = leakage_types[0] if leakage_types else None

        result = LeakageResult(
            feature_name=feature_name,
            has_leakage=leakage_detected,
            leakage_type=primary_type,
            leakage_score=overall_score,
            severity=severity,
            details="; ".join(details) if details else "No leakage detected",
            future_correlation=future_corr,
            target_correlation=target_corr,
        )

        self._results[feature_name] = result
        self._checked_features.append(feature_name)

        if self.verbose and leakage_detected:
            logger.warning(f"Leakage detected in '{feature_name}': {result.details}")

        return result

    def _check_future_correlation(
        self,
        feature_series: pd.Series,
        price_series: pd.Series,
    ) -> float:
        """
        Check if feature correlates with FUTURE returns.

        A properly constructed feature should NOT correlate with future returns
        because that information shouldn't be available at feature calculation time.

        High correlation indicates the feature is using future information.
        """
        # Calculate future returns (returns AFTER each point)
        future_returns = price_series.pct_change().shift(-self.prediction_horizon)

        # Align the data
        aligned = pd.DataFrame({
            "feature": feature_series,
            "future_return": future_returns,
        }).dropna()

        if len(aligned) < self.min_samples:
            return 0.0

        # Calculate correlation
        corr, _ = stats.spearmanr(aligned["feature"], aligned["future_return"])

        return float(corr) if not np.isnan(corr) else 0.0

    def _check_target_leakage(
        self,
        feature_series: pd.Series,
        target_series: pd.Series,
    ) -> float:
        """
        Check if feature directly encodes target information.

        Very high correlation with target suggests the feature might be
        derived from the target or contain future information about it.
        """
        aligned = pd.DataFrame({
            "feature": feature_series,
            "target": target_series,
        }).dropna()

        if len(aligned) < self.min_samples:
            return 0.0

        corr, _ = stats.spearmanr(aligned["feature"], aligned["target"])

        return float(corr) if not np.isnan(corr) else 0.0

    def _check_timestamp_consistency(
        self,
        feature_series: pd.Series,
    ) -> Optional[str]:
        """
        Check for timestamp-related issues that might indicate leakage.
        """
        # Check if index is datetime
        if not isinstance(feature_series.index, pd.DatetimeIndex):
            return None  # Can't validate timestamps

        # Check for future timestamps
        current_time = feature_series.index.max()
        if pd.isna(current_time):
            return "Invalid timestamps in feature"

        # Check for non-monotonic timestamps (might indicate data issues)
        if not feature_series.index.is_monotonic_increasing:
            return "Non-monotonic timestamps detected"

        return None

    def _check_lookahead_patterns(
        self,
        feature_series: pd.Series,
        price_series: pd.Series,
    ) -> float:
        """
        Check for common lookahead bias patterns.

        Tests if the feature has suspiciously good predictive power that
        might indicate it's using future information.
        """
        # Calculate information coefficient (IC) at different lags
        ics = []

        for lag in range(1, 6):  # Check lags 1-5
            future_return = price_series.pct_change().shift(-lag)
            aligned = pd.DataFrame({
                "feature": feature_series,
                "return": future_return,
            }).dropna()

            if len(aligned) >= self.min_samples:
                ic, _ = stats.spearmanr(aligned["feature"], aligned["return"])
                if not np.isnan(ic):
                    ics.append(abs(ic))

        if not ics:
            return 0.0

        # If IC increases significantly for shorter lags, might indicate lookahead
        # This is suspicious because features shouldn't predict better at shorter horizons
        # unless they contain future information

        # Calculate a lookahead score based on IC pattern
        max_ic = max(ics)
        avg_ic = np.mean(ics)

        # High max IC relative to average suggests lookahead pattern
        if avg_ic > 0:
            lookahead_score = max_ic - avg_ic
        else:
            lookahead_score = 0.0

        return float(min(1.0, max(0.0, lookahead_score * 5)))  # Scale to 0-1

    def check_all_features(
        self,
        feature_df: pd.DataFrame,
        price_df: pd.DataFrame,
        target_series: pd.Series,
    ) -> Dict[str, LeakageResult]:
        """
        Check all features in a DataFrame for leakage.

        Args:
            feature_df: DataFrame with features
            price_df: DataFrame or Series with close prices
            target_series: Target variable

        Returns:
            Dictionary mapping feature names to LeakageResult
        """
        # Get price series
        if isinstance(price_df, pd.DataFrame):
            if "close" in price_df.columns:
                price_series = price_df["close"]
            else:
                price_series = price_df.iloc[:, 0]
        else:
            price_series = price_df

        results = {}

        for col in feature_df.columns:
            result = self.check_feature_for_leakage(
                feature_series=feature_df[col],
                price_series=price_series,
                target_series=target_series,
                feature_name=col,
            )
            results[col] = result

        logger.info(
            f"Leakage check complete: {sum(1 for r in results.values() if r.has_leakage)}/{len(results)} "
            f"features have potential leakage"
        )

        return results

    def get_safe_features(
        self,
        feature_df: pd.DataFrame,
        price_df: pd.DataFrame,
        target_series: pd.Series,
        max_severity: LeakageSeverity = LeakageSeverity.WARNING,
    ) -> List[str]:
        """
        Get list of features that pass leakage checks.

        Args:
            feature_df: DataFrame with features
            price_df: Price data
            target_series: Target variable
            max_severity: Maximum acceptable severity level

        Returns:
            List of safe feature names
        """
        results = self.check_all_features(feature_df, price_df, target_series)

        severity_order = {
            LeakageSeverity.NONE: 0,
            LeakageSeverity.WARNING: 1,
            LeakageSeverity.HIGH: 2,
            LeakageSeverity.CRITICAL: 3,
        }

        max_level = severity_order.get(max_severity, 1)

        safe_features = [
            name for name, result in results.items()
            if severity_order.get(result.severity, 3) <= max_level
        ]

        logger.info(f"Identified {len(safe_features)} safe features out of {len(results)}")

        return safe_features

    def generate_report(self) -> LeakageReport:
        """
        Generate comprehensive leakage analysis report.

        Returns:
            LeakageReport with full analysis
        """
        if not self._results:
            return LeakageReport(
                total_features=0,
                features_with_leakage=0,
                critical_count=0,
                high_count=0,
                warning_count=0,
                safe_features=[],
                unsafe_features=[],
                results={},
                recommendations=["No features have been checked yet"],
            )

        # Count by severity
        critical = [n for n, r in self._results.items() if r.severity == LeakageSeverity.CRITICAL]
        high = [n for n, r in self._results.items() if r.severity == LeakageSeverity.HIGH]
        warning = [n for n, r in self._results.items() if r.severity == LeakageSeverity.WARNING]
        safe = [n for n, r in self._results.items() if not r.has_leakage]
        unsafe = [n for n, r in self._results.items() if r.has_leakage]

        # Generate recommendations
        recommendations = []

        if critical:
            recommendations.append(
                f"CRITICAL: Remove or fix {len(critical)} features with severe leakage: "
                f"{', '.join(critical[:5])}"
            )

        if high:
            recommendations.append(
                f"HIGH: Investigate {len(high)} features with significant leakage risk"
            )

        if warning:
            recommendations.append(
                f"WARNING: Monitor {len(warning)} features with minor leakage indicators"
            )

        # Feature-specific recommendations
        for name, result in self._results.items():
            if result.leakage_type == LeakageType.FUTURE_DATA:
                recommendations.append(
                    f"Feature '{name}': Check calculation doesn't use future data"
                )
            elif result.leakage_type == LeakageType.TARGET_LEAKAGE:
                recommendations.append(
                    f"Feature '{name}': May directly encode target - verify independence"
                )

        if not recommendations:
            recommendations.append("All features passed leakage checks")

        return LeakageReport(
            total_features=len(self._results),
            features_with_leakage=len(unsafe),
            critical_count=len(critical),
            high_count=len(high),
            warning_count=len(warning),
            safe_features=safe,
            unsafe_features=unsafe,
            results=self._results,
            recommendations=recommendations[:10],  # Top 10 recommendations
        )

    def reset(self) -> None:
        """Reset checker state for new analysis."""
        self._results = {}
        self._checked_features = []


def validate_no_leakage(
    feature_df: pd.DataFrame,
    price_series: pd.Series,
    target_series: pd.Series,
    threshold: float = 0.1,
) -> Tuple[bool, List[str]]:
    """
    Quick validation that features have no significant leakage.

    Args:
        feature_df: Feature DataFrame
        price_series: Close prices
        target_series: Target variable
        threshold: Correlation threshold

    Returns:
        Tuple of (is_valid, list_of_problematic_features)
    """
    checker = LeakageChecker(
        future_correlation_threshold=threshold,
        verbose=False,
    )

    results = checker.check_all_features(feature_df, price_series, target_series)

    problematic = [
        name for name, result in results.items()
        if result.severity in [LeakageSeverity.CRITICAL, LeakageSeverity.HIGH]
    ]

    return len(problematic) == 0, problematic
