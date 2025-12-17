"""
Leakage Detector for validating ML pipelines.

This module provides comprehensive checks for various types of data leakage
that can invalidate backtest results:
1. Feature leakage (using future data in features)
2. Target leakage (information from target in features)
3. Train-test leakage (test data in training)
4. Temporal leakage (incorrect time ordering)

Designed for JPMorgan-level institutional requirements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LeakageType(Enum):
    """Types of data leakage."""

    FEATURE_LOOKAHEAD = "feature_lookahead"      # Features use future data
    TARGET_LEAKAGE = "target_leakage"            # Target info leaks to features
    TRAIN_TEST_OVERLAP = "train_test_overlap"    # Overlap between train/test
    TEMPORAL_SHUFFLE = "temporal_shuffle"         # Data is shuffled in time
    SCALER_LEAKAGE = "scaler_leakage"            # Scaler fitted on all data
    CV_LEAKAGE = "cv_leakage"                    # CV fold leakage
    SURVIVORSHIP = "survivorship_bias"           # Only survivors in data


@dataclass
class LeakageWarning:
    """Represents a detected leakage issue."""

    leakage_type: LeakageType
    severity: str  # "critical", "warning", "info"
    description: str
    location: str  # Where in the code/data
    recommendation: str
    detected_at: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        return (
            f"[{self.severity.upper()}] {self.leakage_type.value}: "
            f"{self.description} ({self.location})"
        )


@dataclass
class LeakageReport:
    """Complete leakage detection report."""

    warnings: List[LeakageWarning]
    passed: bool
    summary: str
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def critical_count(self) -> int:
        return sum(1 for w in self.warnings if w.severity == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for w in self.warnings if w.severity == "warning")

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "warnings": [
                {
                    "type": w.leakage_type.value,
                    "severity": w.severity,
                    "description": w.description,
                    "location": w.location,
                    "recommendation": w.recommendation,
                }
                for w in self.warnings
            ],
            "summary": self.summary,
            "timestamp": self.timestamp.isoformat(),
        }


class LeakageDetector:
    """
    Comprehensive data leakage detector for ML pipelines.

    Checks for common sources of look-ahead bias and data leakage
    that can invalidate backtest results.

    Example usage:
        detector = LeakageDetector(strict=True)

        # Check for leakage in features
        report = detector.check_all(
            features=X,
            target=y,
            train_idx=train_indices,
            test_idx=test_indices,
            timestamps=df.index,
        )

        if not report.passed:
            raise ValueError(f"Data leakage detected: {report.summary}")
    """

    def __init__(
        self,
        strict: bool = True,
        max_correlation_threshold: float = 0.95,
        temporal_tolerance: int = 0,
    ) -> None:
        """
        Initialize leakage detector.

        Args:
            strict: If True, any leakage fails validation
            max_correlation_threshold: Max allowed feature-target correlation
            temporal_tolerance: Allowed temporal overlap in indices
        """
        self.strict = strict
        self.max_correlation_threshold = max_correlation_threshold
        self.temporal_tolerance = temporal_tolerance

        self._warnings: List[LeakageWarning] = []

    def check_all(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        timestamps: Optional[pd.Index] = None,
        purge_gap: Optional[int] = None,
    ) -> LeakageReport:
        """
        Run all leakage checks.

        Args:
            features: Feature DataFrame
            target: Target Series
            train_idx: Training indices
            test_idx: Test indices
            timestamps: Datetime index
            purge_gap: Expected purge gap

        Returns:
            LeakageReport with all findings
        """
        self._warnings = []

        # Run all checks
        self._check_train_test_overlap(train_idx, test_idx)
        self._check_temporal_ordering(timestamps, train_idx, test_idx)
        self._check_feature_target_correlation(features, target)
        self._check_future_information(features, timestamps)

        if purge_gap:
            self._check_purge_gap(train_idx, test_idx, purge_gap)

        # Generate report
        passed = len([w for w in self._warnings if w.severity == "critical"]) == 0
        if self.strict:
            passed = len(self._warnings) == 0

        summary = self._generate_summary()

        return LeakageReport(
            warnings=self._warnings,
            passed=passed,
            summary=summary,
        )

    def check_feature_pipeline(
        self,
        pipeline,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
    ) -> LeakageReport:
        """
        Check feature pipeline for leakage.

        Args:
            pipeline: Feature pipeline object
            train_data: Training data
            test_data: Test data

        Returns:
            LeakageReport
        """
        self._warnings = []

        # Check if pipeline was fitted
        if hasattr(pipeline, "_is_fitted") and not pipeline._is_fitted:
            self._add_warning(
                LeakageType.SCALER_LEAKAGE,
                "critical",
                "Pipeline not fitted before transform",
                "FeaturePipeline",
                "Call fit() on training data before transform()",
            )

        # Check scaler statistics
        if hasattr(pipeline, "processor") and hasattr(pipeline.processor, "_scaler"):
            scaler = pipeline.processor._scaler
            if scaler is not None and hasattr(scaler, "mean_"):
                # Scaler is fitted - verify it's from training data
                train_features = pipeline.generate_features(train_data)
                expected_mean = train_features.mean()

                # Compare scaler mean to training mean
                scaler_mean = pd.Series(scaler.mean_, index=train_features.columns)
                if not np.allclose(expected_mean.dropna(), scaler_mean.dropna(), rtol=0.1):
                    self._add_warning(
                        LeakageType.SCALER_LEAKAGE,
                        "warning",
                        "Scaler statistics may not match training data",
                        "FeatureProcessor",
                        "Ensure scaler is fitted only on training data",
                    )

        passed = len([w for w in self._warnings if w.severity == "critical"]) == 0
        summary = self._generate_summary()

        return LeakageReport(
            warnings=self._warnings,
            passed=passed,
            summary=summary,
        )

    def check_cv_leakage(
        self,
        cv_splitter,
        n_samples: int,
        purge_gap: int,
        prediction_horizon: int,
        max_lookback: int,
    ) -> LeakageReport:
        """
        Check cross-validation for leakage.

        Args:
            cv_splitter: Cross-validation splitter
            n_samples: Number of samples
            purge_gap: Configured purge gap
            prediction_horizon: Prediction horizon
            max_lookback: Maximum feature lookback

        Returns:
            LeakageReport
        """
        self._warnings = []

        # Check if purge gap is sufficient
        required_purge = prediction_horizon + max_lookback
        if purge_gap < required_purge:
            self._add_warning(
                LeakageType.CV_LEAKAGE,
                "critical",
                f"Purge gap ({purge_gap}) is less than required "
                f"({required_purge} = horizon + lookback)",
                "PurgedKFoldCV",
                f"Set purge_gap >= {required_purge}",
            )

        # Check each split
        X_dummy = np.zeros((n_samples, 1))
        for i, (train_idx, test_idx) in enumerate(cv_splitter.split(X_dummy)):
            overlap = set(train_idx) & set(test_idx)
            if overlap:
                self._add_warning(
                    LeakageType.TRAIN_TEST_OVERLAP,
                    "critical",
                    f"Fold {i}: Train-test overlap at indices {list(overlap)[:5]}...",
                    f"CV Fold {i}",
                    "Check CV implementation",
                )

            # Check temporal gap
            if len(train_idx) > 0 and len(test_idx) > 0:
                train_before = train_idx[train_idx < test_idx.min()]
                if len(train_before) > 0:
                    gap = test_idx.min() - train_before.max()
                    if gap < purge_gap:
                        self._add_warning(
                            LeakageType.CV_LEAKAGE,
                            "critical",
                            f"Fold {i}: Gap ({gap}) less than purge_gap ({purge_gap})",
                            f"CV Fold {i}",
                            "Increase purge_gap parameter",
                        )

        passed = len([w for w in self._warnings if w.severity == "critical"]) == 0
        summary = self._generate_summary()

        return LeakageReport(
            warnings=self._warnings,
            passed=passed,
            summary=summary,
        )

    def _check_train_test_overlap(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> None:
        """Check for overlap between train and test sets."""
        overlap = set(train_idx) & set(test_idx)

        if overlap:
            self._add_warning(
                LeakageType.TRAIN_TEST_OVERLAP,
                "critical",
                f"Train-test overlap detected at {len(overlap)} indices",
                "Data Split",
                "Ensure train and test sets are disjoint",
            )

    def _check_temporal_ordering(
        self,
        timestamps: Optional[pd.Index],
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> None:
        """Check for temporal ordering issues."""
        if timestamps is None:
            return

        # Check if any training data is after test data
        if len(train_idx) > 0 and len(test_idx) > 0:
            train_times = timestamps[train_idx]
            test_times = timestamps[test_idx]

            if train_times.max() >= test_times.min():
                self._add_warning(
                    LeakageType.TEMPORAL_SHUFFLE,
                    "warning",
                    f"Training data extends to {train_times.max()}, "
                    f"but test data starts at {test_times.min()}",
                    "Temporal Ordering",
                    "Ensure training data precedes test data",
                )

    def _check_feature_target_correlation(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> None:
        """Check for suspicious feature-target correlations."""
        aligned = features.align(target, join="inner")[0]
        aligned_target = target.loc[aligned.index]

        for col in aligned.columns:
            corr = aligned[col].corr(aligned_target)

            if abs(corr) > self.max_correlation_threshold:
                self._add_warning(
                    LeakageType.TARGET_LEAKAGE,
                    "warning",
                    f"Feature '{col}' has {corr:.3f} correlation with target",
                    f"Feature: {col}",
                    "Investigate if feature contains target information",
                )

    def _check_future_information(
        self,
        features: pd.DataFrame,
        timestamps: Optional[pd.Index],
    ) -> None:
        """Check for features that might contain future information."""
        if timestamps is None:
            return

        # Look for suspicious column names
        suspicious_patterns = ["forward", "future", "next", "predict", "target"]

        for col in features.columns:
            col_lower = col.lower()
            for pattern in suspicious_patterns:
                if pattern in col_lower:
                    self._add_warning(
                        LeakageType.FEATURE_LOOKAHEAD,
                        "warning",
                        f"Feature '{col}' name suggests future information",
                        f"Feature: {col}",
                        "Verify feature does not use future data",
                    )
                    break

    def _check_purge_gap(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        expected_purge: int,
    ) -> None:
        """Check if purge gap is properly enforced."""
        if len(train_idx) == 0 or len(test_idx) == 0:
            return

        # Check gap before test
        train_before = train_idx[train_idx < test_idx.min()]
        if len(train_before) > 0:
            actual_gap = test_idx.min() - train_before.max()
            if actual_gap < expected_purge:
                self._add_warning(
                    LeakageType.CV_LEAKAGE,
                    "critical",
                    f"Purge gap ({actual_gap}) less than required ({expected_purge})",
                    "Purge Gap",
                    f"Increase purge gap to at least {expected_purge}",
                )

    def _add_warning(
        self,
        leakage_type: LeakageType,
        severity: str,
        description: str,
        location: str,
        recommendation: str,
    ) -> None:
        """Add a warning to the list."""
        warning = LeakageWarning(
            leakage_type=leakage_type,
            severity=severity,
            description=description,
            location=location,
            recommendation=recommendation,
        )
        self._warnings.append(warning)
        logger.warning(str(warning))

    def _generate_summary(self) -> str:
        """Generate summary of findings."""
        if not self._warnings:
            return "No leakage detected"

        critical = self.critical_count
        warnings = self.warning_count

        return (
            f"Found {len(self._warnings)} issues: "
            f"{critical} critical, {warnings} warnings"
        )

    @property
    def critical_count(self) -> int:
        return sum(1 for w in self._warnings if w.severity == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for w in self._warnings if w.severity == "warning")


def validate_no_leakage(
    features: pd.DataFrame,
    target: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    raise_on_failure: bool = True,
) -> bool:
    """
    Convenience function to validate no leakage.

    Args:
        features: Features
        target: Target
        train_idx: Training indices
        test_idx: Test indices
        raise_on_failure: Raise exception on leakage

    Returns:
        True if no leakage detected

    Raises:
        ValueError: If leakage detected and raise_on_failure=True
    """
    detector = LeakageDetector(strict=True)
    report = detector.check_all(features, target, train_idx, test_idx)

    if not report.passed and raise_on_failure:
        raise ValueError(f"Data leakage detected: {report.summary}")

    return report.passed
