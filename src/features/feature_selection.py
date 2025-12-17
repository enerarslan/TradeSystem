"""
Feature Selection Module for AlphaTrade System.

This module provides automated feature selection to reduce dimensionality
and improve model performance while preventing overfitting.

Implements:
- Variance threshold selection
- Correlation filter
- Mutual Information selection
- Model-based importance (LightGBM)
- Recursive Feature Elimination (RFE)
- Stability Selection

Reference:
    "Advances in Financial Machine Learning" by de Prado (2018)
    Chapter 8: Feature Importance
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    mutual_info_classif,
    mutual_info_regression,
    RFE,
)
from scipy import stats

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class SelectionMethod(str, Enum):
    """Available feature selection methods."""
    VARIANCE = "variance"
    CORRELATION = "correlation"
    MUTUAL_INFO = "mutual_info"
    IMPORTANCE = "importance"
    RFE = "rfe"
    STABILITY = "stability"
    COMBINED = "combined"


@dataclass
class SelectionResult:
    """Result from feature selection."""
    method: SelectionMethod
    n_features_original: int
    n_features_selected: int
    selected_features: List[str]
    feature_scores: Dict[str, float]
    dropped_features: List[str]
    selection_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method.value,
            "n_features_original": self.n_features_original,
            "n_features_selected": self.n_features_selected,
            "selected_features": self.selected_features,
            "feature_scores": self.feature_scores,
            "dropped_features": self.dropped_features,
        }

    def summary(self) -> str:
        """Generate text summary."""
        return (
            f"Feature Selection ({self.method.value}): "
            f"{self.n_features_original} -> {self.n_features_selected} features "
            f"({self.n_features_selected/self.n_features_original*100:.1f}% retained)"
        )


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Automated feature selection for financial ML.

    Implements multiple selection strategies that can be used individually
    or combined for robust feature selection.

    Usage:
        selector = FeatureSelector(method="importance", n_features=50)
        selector.fit(X_train, y_train)
        X_selected = selector.transform(X_test)

        # Or combined selection
        selector = FeatureSelector(method="combined", n_features=50)
        selected_features = selector.get_selected_features()
    """

    def __init__(
        self,
        method: Union[SelectionMethod, str] = SelectionMethod.IMPORTANCE,
        n_features: Optional[int] = None,
        threshold: Optional[float] = None,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        importance_threshold: float = 0.001,
        task_type: str = "regression",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Initialize the feature selector.

        Args:
            method: Selection method to use
            n_features: Number of features to select (None = auto)
            threshold: Score threshold for selection
            variance_threshold: Minimum variance for variance filter
            correlation_threshold: Maximum correlation for correlation filter
            importance_threshold: Minimum importance score
            task_type: "regression" or "classification"
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs
        """
        self.method = SelectionMethod(method) if isinstance(method, str) else method
        self.n_features = n_features
        self.threshold = threshold
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.importance_threshold = importance_threshold
        self.task_type = task_type
        self.random_state = random_state
        self.n_jobs = n_jobs

        # State
        self._feature_names: List[str] = []
        self._selected_features: List[str] = []
        self._feature_scores: Dict[str, float] = {}
        self._is_fitted: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "FeatureSelector":
        """
        Fit the selector on training data.

        Args:
            X: Feature DataFrame
            y: Target series

        Returns:
            self
        """
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()
        else:
            self._feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Convert to arrays
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y

        # Apply selection method
        if self.method == SelectionMethod.VARIANCE:
            self._fit_variance(X_arr)
        elif self.method == SelectionMethod.CORRELATION:
            self._fit_correlation(X_arr)
        elif self.method == SelectionMethod.MUTUAL_INFO:
            self._fit_mutual_info(X_arr, y_arr)
        elif self.method == SelectionMethod.IMPORTANCE:
            self._fit_importance(X_arr, y_arr)
        elif self.method == SelectionMethod.RFE:
            self._fit_rfe(X_arr, y_arr)
        elif self.method == SelectionMethod.STABILITY:
            self._fit_stability(X_arr, y_arr)
        elif self.method == SelectionMethod.COMBINED:
            self._fit_combined(X_arr, y_arr)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self._is_fitted = True
        logger.info(
            f"Feature selection complete: {len(self._feature_names)} -> "
            f"{len(self._selected_features)} features"
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data to selected features.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with selected features only
        """
        if not self._is_fitted:
            raise ValueError("Selector not fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            return X[self._selected_features]
        else:
            # Assume array - return selected columns
            indices = [self._feature_names.index(f) for f in self._selected_features]
            return X[:, indices]

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)

    def get_selected_features(self) -> List[str]:
        """Get list of selected feature names."""
        return self._selected_features

    def get_feature_scores(self) -> Dict[str, float]:
        """Get scores for all features."""
        return self._feature_scores

    def get_selection_result(self) -> SelectionResult:
        """Get complete selection result."""
        dropped = [f for f in self._feature_names if f not in self._selected_features]

        return SelectionResult(
            method=self.method,
            n_features_original=len(self._feature_names),
            n_features_selected=len(self._selected_features),
            selected_features=self._selected_features,
            feature_scores=self._feature_scores,
            dropped_features=dropped,
        )

    def _fit_variance(self, X: np.ndarray) -> None:
        """Select features with variance above threshold."""
        # Calculate variance for each feature
        variances = np.var(X, axis=0)

        for i, var in enumerate(variances):
            self._feature_scores[self._feature_names[i]] = float(var)

        # Select features above threshold
        mask = variances >= self.variance_threshold
        self._selected_features = [
            self._feature_names[i] for i in range(len(mask)) if mask[i]
        ]

        # Also limit to n_features if specified
        if self.n_features and len(self._selected_features) > self.n_features:
            sorted_features = sorted(
                self._selected_features,
                key=lambda f: self._feature_scores[f],
                reverse=True,
            )
            self._selected_features = sorted_features[:self.n_features]

    def _fit_correlation(self, X: np.ndarray) -> None:
        """Remove highly correlated features."""
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X, rowvar=False)

        # Handle NaN values
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Track which features to drop
        n_features = X.shape[1]
        to_drop = set()

        for i in range(n_features):
            if i in to_drop:
                continue

            for j in range(i + 1, n_features):
                if j in to_drop:
                    continue

                if abs(corr_matrix[i, j]) > self.correlation_threshold:
                    # Drop the feature with higher mean correlation
                    mean_corr_i = np.mean(np.abs(corr_matrix[i, :]))
                    mean_corr_j = np.mean(np.abs(corr_matrix[j, :]))

                    if mean_corr_i > mean_corr_j:
                        to_drop.add(i)
                    else:
                        to_drop.add(j)

        # Calculate scores (inverse of mean correlation)
        for i in range(n_features):
            mean_corr = np.mean(np.abs(corr_matrix[i, :]))
            self._feature_scores[self._feature_names[i]] = 1.0 - mean_corr

        # Select features not in drop list
        self._selected_features = [
            self._feature_names[i] for i in range(n_features) if i not in to_drop
        ]

    def _fit_mutual_info(self, X: np.ndarray, y: np.ndarray) -> None:
        """Select features with highest mutual information with target."""
        # Choose MI function based on task type
        if self.task_type == "classification":
            mi_func = mutual_info_classif
        else:
            mi_func = mutual_info_regression

        # Calculate mutual information
        mi_scores = mi_func(
            X, y,
            random_state=self.random_state,
            n_neighbors=3,
        )

        # Store scores
        for i, score in enumerate(mi_scores):
            self._feature_scores[self._feature_names[i]] = float(score)

        # Sort by MI score
        sorted_features = sorted(
            self._feature_names,
            key=lambda f: self._feature_scores[f],
            reverse=True,
        )

        # Select top features
        n_select = self.n_features or int(len(sorted_features) * 0.5)
        self._selected_features = sorted_features[:n_select]

    def _fit_importance(self, X: np.ndarray, y: np.ndarray) -> None:
        """Select features based on model importance scores."""
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available, falling back to RandomForest")
            self._fit_importance_rf(X, y)
            return

        # Train LightGBM model
        if self.task_type == "classification":
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1,
            )
        else:
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1,
            )

        # Handle NaN values
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y_clean = np.nan_to_num(y, nan=0.0)

        model.fit(X_clean, y_clean)

        # Get importance scores
        importances = model.feature_importances_

        # Normalize to sum to 1
        importances = importances / importances.sum()

        # Store scores
        for i, score in enumerate(importances):
            self._feature_scores[self._feature_names[i]] = float(score)

        # Sort by importance
        sorted_features = sorted(
            self._feature_names,
            key=lambda f: self._feature_scores[f],
            reverse=True,
        )

        # Select features above threshold or top N
        if self.n_features:
            self._selected_features = sorted_features[:self.n_features]
        else:
            self._selected_features = [
                f for f in sorted_features
                if self._feature_scores[f] >= self.importance_threshold
            ]

    def _fit_importance_rf(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fallback: Use RandomForest for importance."""
        if self.task_type == "classification":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

        X_clean = np.nan_to_num(X, nan=0.0)
        y_clean = np.nan_to_num(y, nan=0.0)

        model.fit(X_clean, y_clean)

        importances = model.feature_importances_

        for i, score in enumerate(importances):
            self._feature_scores[self._feature_names[i]] = float(score)

        sorted_features = sorted(
            self._feature_names,
            key=lambda f: self._feature_scores[f],
            reverse=True,
        )

        n_select = self.n_features or int(len(sorted_features) * 0.5)
        self._selected_features = sorted_features[:n_select]

    def _fit_rfe(self, X: np.ndarray, y: np.ndarray) -> None:
        """Recursive Feature Elimination."""
        n_select = self.n_features or int(X.shape[1] * 0.5)

        # Use RandomForest as base estimator
        if self.task_type == "classification":
            estimator = RandomForestClassifier(
                n_estimators=50,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators=50,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

        X_clean = np.nan_to_num(X, nan=0.0)
        y_clean = np.nan_to_num(y, nan=0.0)

        # Run RFE
        rfe = RFE(
            estimator=estimator,
            n_features_to_select=n_select,
            step=0.1,
        )
        rfe.fit(X_clean, y_clean)

        # Store rankings (lower = better)
        for i, rank in enumerate(rfe.ranking_):
            self._feature_scores[self._feature_names[i]] = 1.0 / rank

        # Get selected features
        self._selected_features = [
            self._feature_names[i]
            for i in range(len(rfe.support_))
            if rfe.support_[i]
        ]

    def _fit_stability(self, X: np.ndarray, y: np.ndarray, n_bootstrap: int = 50) -> None:
        """
        Stability Selection: Run selection multiple times with bootstrap.

        Features consistently selected across runs are more robust.
        """
        selection_counts = {f: 0 for f in self._feature_names}

        for i in range(n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(len(X), len(X), replace=True)
            X_boot = X[idx]
            y_boot = y[idx]

            # Run importance-based selection
            temp_selector = FeatureSelector(
                method=SelectionMethod.IMPORTANCE,
                n_features=self.n_features or int(X.shape[1] * 0.5),
                task_type=self.task_type,
                random_state=self.random_state + i,
                n_jobs=self.n_jobs,
            )
            temp_selector._feature_names = self._feature_names
            temp_selector._fit_importance(X_boot, y_boot)

            # Count selections
            for f in temp_selector._selected_features:
                selection_counts[f] += 1

        # Calculate stability scores (fraction of times selected)
        for f in self._feature_names:
            self._feature_scores[f] = selection_counts[f] / n_bootstrap

        # Select features that appear in >50% of runs
        stability_threshold = 0.5
        self._selected_features = [
            f for f in self._feature_names
            if self._feature_scores[f] >= stability_threshold
        ]

        # If too few, add more based on score
        if len(self._selected_features) < 10:
            sorted_features = sorted(
                self._feature_names,
                key=lambda f: self._feature_scores[f],
                reverse=True,
            )
            self._selected_features = sorted_features[:max(10, self.n_features or 10)]

    def _fit_combined(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Combined selection: Use multiple methods and take intersection/union.
        """
        # Run multiple methods
        methods_to_run = [
            SelectionMethod.VARIANCE,
            SelectionMethod.CORRELATION,
            SelectionMethod.IMPORTANCE,
        ]

        method_selections = {}
        for method in methods_to_run:
            temp_selector = FeatureSelector(
                method=method,
                n_features=self.n_features,
                variance_threshold=self.variance_threshold,
                correlation_threshold=self.correlation_threshold,
                importance_threshold=self.importance_threshold,
                task_type=self.task_type,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            temp_selector._feature_names = self._feature_names
            temp_selector.fit(
                pd.DataFrame(X, columns=self._feature_names),
                pd.Series(y),
            )
            method_selections[method] = set(temp_selector._selected_features)

        # Combine scores
        for f in self._feature_names:
            # Count how many methods selected this feature
            count = sum(1 for m in method_selections.values() if f in m)
            self._feature_scores[f] = count / len(methods_to_run)

        # Take features selected by at least 2 methods
        self._selected_features = [
            f for f in self._feature_names
            if self._feature_scores[f] >= 0.66  # At least 2 out of 3
        ]

        # Ensure minimum features
        if len(self._selected_features) < 10:
            sorted_features = sorted(
                self._feature_names,
                key=lambda f: self._feature_scores[f],
                reverse=True,
            )
            n_select = max(10, self.n_features or 10)
            self._selected_features = sorted_features[:n_select]


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "importance",
    n_features: Optional[int] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    """
    Convenience function for feature selection.

    Args:
        X: Feature DataFrame
        y: Target series
        method: Selection method
        n_features: Number of features to select
        **kwargs: Additional arguments for FeatureSelector

    Returns:
        Tuple of (selected_X, selected_feature_names, feature_scores)
    """
    selector = FeatureSelector(
        method=method,
        n_features=n_features,
        **kwargs,
    )

    X_selected = selector.fit_transform(X, y)
    selected_features = selector.get_selected_features()
    scores = selector.get_feature_scores()

    return X_selected, selected_features, scores
