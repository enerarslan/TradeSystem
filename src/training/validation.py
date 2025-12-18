"""
Time-series cross-validation strategies with purging and embargo.

This module provides sophisticated cross-validation methods designed for
financial time-series that prevent look-ahead bias and information leakage.

Based on the methodologies from:
- "Advances in Financial Machine Learning" by Marcos Lopez de Prado
- "The Sharpe Ratio Efficient Frontier" by Bailey & López de Prado

Designed for JPMorgan-level requirements:
- Statistically rigorous validation
- Information leakage prevention
- Multiple validation strategies
- Visualization and diagnostics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Generator, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class SplitInfo:
    """Information about a single train/test split."""
    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    purge_start: Optional[int] = None
    purge_end: Optional[int] = None
    embargo_start: Optional[int] = None
    embargo_end: Optional[int] = None


class PurgedKFoldCV:
    """
    Purged K-Fold Cross-Validation for time-series data.

    This implementation prevents information leakage by:
    1. Purging: Removing samples from training that overlap with test labels
    2. Embargo: Adding a gap after test samples before using them for training

    The purge gap should be set to:
        purge_gap = prediction_horizon + max_feature_lookback

    Example:
        cv = PurgedKFoldCV(
            n_splits=5,
            purge_gap=20,  # prediction_horizon + max lookback
            embargo_pct=0.01
        )

        for train_idx, test_idx in cv.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            predictions = model.predict(X[test_idx])

    Reference:
        Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
        Chapter 7: Cross-Validation in Finance.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: Optional[int] = None,
        embargo_pct: float = 0.01,
        prediction_horizon: int = 1,
        max_feature_lookback: int = 20,
    ):
        """
        Initialize Purged K-Fold CV.

        Args:
            n_splits: Number of folds
            purge_gap: Number of samples to purge between train/test.
                      If None, calculated as prediction_horizon + max_feature_lookback
            embargo_pct: Percentage of test set to embargo after test period
            prediction_horizon: How many bars ahead the prediction targets
            max_feature_lookback: Maximum lookback used in feature calculation
        """
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")

        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.prediction_horizon = prediction_horizon
        self.max_feature_lookback = max_feature_lookback

        # Calculate purge gap if not provided
        if purge_gap is None:
            self.purge_gap = prediction_horizon + max_feature_lookback
        else:
            self.purge_gap = purge_gap

    def get_purge_gap(self) -> int:
        """Get the purge gap (samples between train end and test start)."""
        return self.purge_gap

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return number of splits."""
        return self.n_splits

    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices with purging and embargo.

        Args:
            X: Feature data
            y: Target data (optional)
            groups: Group labels (optional, not used)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate fold sizes
        fold_size = n_samples // self.n_splits

        # Calculate embargo period (applied after each test set)
        embargo = int(self.embargo_pct * fold_size)

        # Log diagnostic info
        logger.info(f"PurgedKFoldCV split: n_samples={n_samples}, n_splits={self.n_splits}, "
                   f"fold_size={fold_size}, purge_gap={self.purge_gap}, embargo={embargo}")

        # Adjust purge_gap if it's too large for the data
        effective_purge_gap = self.purge_gap
        max_reasonable_purge = fold_size // 2  # Purge shouldn't exceed half of fold size
        if effective_purge_gap > max_reasonable_purge and max_reasonable_purge > 0:
            logger.warning(f"Purge gap ({self.purge_gap}) is too large for data. "
                          f"Reducing to {max_reasonable_purge}")
            effective_purge_gap = max_reasonable_purge

        for i in range(self.n_splits):
            # Determine test indices for this fold
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            test_indices = indices[test_start:test_end]

            # Build train indices with purging
            train_indices = []

            # Before test set (with purge gap)
            if test_start > 0:
                train_before_end = test_start - effective_purge_gap
                if train_before_end > 0:
                    train_indices.extend(indices[:train_before_end])

            # After test set (with embargo)
            if test_end < n_samples:
                train_after_start = test_end + embargo
                if train_after_start < n_samples:
                    train_indices.extend(indices[train_after_start:])

            train_indices = np.array(train_indices)

            logger.debug(f"Fold {i}: test=[{test_start}:{test_end}], "
                        f"train_before_end={test_start - effective_purge_gap if test_start > 0 else 'N/A'}, "
                        f"train_after_start={test_end + embargo if test_end < n_samples else 'N/A'}, "
                        f"train_samples={len(train_indices)}")

            if len(train_indices) == 0:
                logger.warning(f"Fold {i}: No training samples after purging "
                              f"(test_start={test_start}, test_end={test_end}, "
                              f"purge_gap={effective_purge_gap}, embargo={embargo})")
                continue

            yield train_indices, test_indices

    def get_split_info(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> List[SplitInfo]:
        """
        Get detailed information about all splits.

        Useful for visualization and debugging.
        """
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        embargo = int(self.embargo_pct * fold_size)

        splits = []

        for i, (train_idx, test_idx) in enumerate(self.split(X)):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples

            # Determine purge and embargo ranges
            purge_start = max(0, test_start - self.purge_gap)
            purge_end = test_start

            embargo_start = test_end
            embargo_end = min(n_samples, test_end + embargo)

            splits.append(SplitInfo(
                fold=i,
                train_indices=train_idx,
                test_indices=test_idx,
                train_start=int(train_idx.min()) if len(train_idx) > 0 else 0,
                train_end=int(train_idx.max()) if len(train_idx) > 0 else 0,
                test_start=int(test_start),
                test_end=int(test_end),
                purge_start=int(purge_start),
                purge_end=int(purge_end),
                embargo_start=int(embargo_start),
                embargo_end=int(embargo_end),
            ))

        return splits

    def visualize(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        figsize: Tuple[int, int] = (14, 8),
    ) -> Optional["plt.Figure"]:
        """
        Visualize the cross-validation splits.

        Creates a chart showing train/test/purge/embargo regions for each fold.
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available for visualization")
            return None

        splits = self.get_split_info(X)
        n_samples = len(X)

        fig, ax = plt.subplots(figsize=figsize)

        colors = {
            'train': '#2ecc71',
            'test': '#e74c3c',
            'purge': '#f39c12',
            'embargo': '#9b59b6',
        }

        for split in splits:
            y_pos = split.fold

            # Draw training regions (before and after test)
            train_idx = split.train_indices
            if len(train_idx) > 0:
                # Find contiguous regions
                diffs = np.diff(train_idx)
                breaks = np.where(diffs > 1)[0]

                if len(breaks) == 0:
                    ax.barh(y_pos, train_idx[-1] - train_idx[0],
                           left=train_idx[0], height=0.6,
                           color=colors['train'], label='Train' if split.fold == 0 else '')
                else:
                    # Before test
                    end1 = train_idx[breaks[0]]
                    ax.barh(y_pos, end1 - train_idx[0],
                           left=train_idx[0], height=0.6,
                           color=colors['train'], label='Train' if split.fold == 0 else '')

                    # After test
                    start2 = train_idx[breaks[0] + 1]
                    ax.barh(y_pos, train_idx[-1] - start2,
                           left=start2, height=0.6,
                           color=colors['train'])

            # Test region
            ax.barh(y_pos, split.test_end - split.test_start,
                   left=split.test_start, height=0.6,
                   color=colors['test'], label='Test' if split.fold == 0 else '')

            # Purge region
            ax.barh(y_pos, split.purge_end - split.purge_start,
                   left=split.purge_start, height=0.6,
                   color=colors['purge'], alpha=0.5,
                   label='Purge' if split.fold == 0 else '')

            # Embargo region
            ax.barh(y_pos, split.embargo_end - split.embargo_start,
                   left=split.embargo_start, height=0.6,
                   color=colors['embargo'], alpha=0.5,
                   label='Embargo' if split.fold == 0 else '')

        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Fold')
        ax.set_yticks(range(self.n_splits))
        ax.set_title(f'Purged {self.n_splits}-Fold Cross-Validation\n'
                    f'(purge_gap={self.purge_gap}, embargo_pct={self.embargo_pct:.1%})')
        ax.legend(loc='upper right')
        ax.set_xlim(0, n_samples)

        plt.tight_layout()
        return fig


class CombinatorialPurgedKFoldCV:
    """
    Combinatorial Purged Cross-Validation.

    Tests on multiple non-overlapping paths through time, providing
    a more robust estimate of out-of-sample performance.

    This method generates all valid combinations of test folds,
    ensuring temporal order is respected.

    Example:
        cv = CombinatorialPurgedKFoldCV(
            n_splits=5,
            n_test_splits=2,
            purge_gap=20
        )

        for train_idx, test_idx in cv.split(X):
            # test_idx contains 2 non-contiguous folds
            model.fit(X[train_idx], y[train_idx])
            predictions = model.predict(X[test_idx])

    Reference:
        Lopez de Prado, M. (2018). Chapter 12: Backtesting through Cross-Validation.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        purge_gap: int = 10,
        embargo_pct: float = 0.01,
    ):
        """
        Initialize Combinatorial Purged K-Fold CV.

        Args:
            n_splits: Total number of folds
            n_test_splits: Number of folds to use as test set per path
            purge_gap: Samples to purge between train and test
            embargo_pct: Percentage of test to embargo
        """
        if n_test_splits >= n_splits:
            raise ValueError("n_test_splits must be < n_splits")

        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

        # Calculate number of paths
        self.n_paths = self._n_combinations()

    def _n_combinations(self) -> int:
        """Calculate number of test fold combinations."""
        from math import comb
        return comb(self.n_splits, self.n_test_splits)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return number of paths."""
        return self.n_paths

    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for all combinatorial paths.

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        fold_size = n_samples // self.n_splits
        embargo = int(self.embargo_pct * fold_size)

        # Get all fold boundaries
        fold_bounds = []
        for i in range(self.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            fold_bounds.append((start, end))

        # Generate all combinations of test folds
        for test_fold_indices in combinations(range(self.n_splits), self.n_test_splits):
            # Collect test indices
            test_indices = []
            for fold_idx in test_fold_indices:
                start, end = fold_bounds[fold_idx]
                test_indices.extend(indices[start:end])
            test_indices = np.array(test_indices)

            # Build train indices with purging
            train_mask = np.ones(n_samples, dtype=bool)

            for fold_idx in test_fold_indices:
                start, end = fold_bounds[fold_idx]

                # Mark test region
                train_mask[start:end] = False

                # Mark purge region (before test)
                purge_start = max(0, start - self.purge_gap)
                train_mask[purge_start:start] = False

                # Mark embargo region (after test)
                embargo_end = min(n_samples, end + embargo)
                train_mask[end:embargo_end] = False

            train_indices = indices[train_mask]

            if len(train_indices) == 0:
                continue

            yield train_indices, test_indices


class WalkForwardValidator:
    """
    Walk-Forward Validation for time-series.

    Simulates realistic model deployment by training on historical data
    and testing on forward periods, then rolling the window forward.

    Supports both:
    - Sliding window: Fixed-size training window
    - Expanding window: Growing training window

    Example:
        wf = WalkForwardValidator(
            train_period=252*26,  # 1 year of 15-min bars
            test_period=21*26,    # 1 month
            step_size=21*26,      # Step 1 month
            expanding=False       # Sliding window
        )

        for train_idx, test_idx in wf.split(X):
            model.fit(X[train_idx], y[train_idx])
            predictions = model.predict(X[test_idx])
    """

    def __init__(
        self,
        train_period: int,
        test_period: int,
        step_size: Optional[int] = None,
        expanding: bool = False,
        purge_gap: int = 10,
        embargo_bars: int = 5,
        min_train_samples: int = 100,
    ):
        """
        Initialize Walk-Forward Validator.

        Args:
            train_period: Number of bars in training window
            test_period: Number of bars in test window
            step_size: Bars to move forward each iteration (default: test_period)
            expanding: If True, use expanding window; if False, sliding window
            purge_gap: Bars to purge between train end and test start
            embargo_bars: Bars to embargo after test
            min_train_samples: Minimum samples required for training
        """
        self.train_period = train_period
        self.test_period = test_period
        self.step_size = step_size or test_period
        self.expanding = expanding
        self.purge_gap = purge_gap
        self.embargo_bars = embargo_bars
        self.min_train_samples = min_train_samples

    def get_n_splits(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y=None,
        groups=None,
    ) -> int:
        """Calculate number of walk-forward splits."""
        n_samples = len(X)
        n_splits = 0
        current_test_start = self.train_period + self.purge_gap

        while current_test_start + self.test_period <= n_samples:
            n_splits += 1
            current_test_start += self.step_size

        return n_splits

    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate walk-forward train/test splits.

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        current_test_start = self.train_period + self.purge_gap

        while current_test_start + self.test_period <= n_samples:
            # Test indices
            test_end = current_test_start + self.test_period
            test_indices = indices[current_test_start:test_end]

            # Train indices
            if self.expanding:
                # Expanding window: train from beginning
                train_start = 0
            else:
                # Sliding window: fixed-size training
                train_start = max(0, current_test_start - self.purge_gap - self.train_period)

            train_end = current_test_start - self.purge_gap
            train_indices = indices[train_start:train_end]

            # Check minimum samples
            if len(train_indices) < self.min_train_samples:
                current_test_start += self.step_size
                continue

            yield train_indices, test_indices

            # Move forward
            current_test_start += self.step_size

    def get_split_info(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> List[dict]:
        """Get detailed information about each split."""
        splits = []
        for i, (train_idx, test_idx) in enumerate(self.split(X)):
            splits.append({
                'fold': i,
                'train_start': int(train_idx.min()),
                'train_end': int(train_idx.max()),
                'train_size': len(train_idx),
                'test_start': int(test_idx.min()),
                'test_end': int(test_idx.max()),
                'test_size': len(test_idx),
            })
        return splits

    def visualize(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        figsize: Tuple[int, int] = (14, 8),
    ) -> Optional["plt.Figure"]:
        """Visualize walk-forward splits."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        splits = self.get_split_info(X)
        n_samples = len(X)

        fig, ax = plt.subplots(figsize=figsize)

        for split in splits:
            y_pos = split['fold']

            # Train region
            ax.barh(y_pos, split['train_end'] - split['train_start'],
                   left=split['train_start'], height=0.6,
                   color='#2ecc71', label='Train' if split['fold'] == 0 else '')

            # Test region
            ax.barh(y_pos, split['test_end'] - split['test_start'],
                   left=split['test_start'], height=0.6,
                   color='#e74c3c', label='Test' if split['fold'] == 0 else '')

        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Fold')
        ax.set_title(f'Walk-Forward Validation ({len(splits)} folds)\n'
                    f'Train: {self.train_period}, Test: {self.test_period}, '
                    f'{"Expanding" if self.expanding else "Sliding"} window')
        ax.legend(loc='upper right')
        ax.set_xlim(0, n_samples)

        plt.tight_layout()
        return fig


class TimeSeriesSplit:
    """
    Standard time-series split with optional purging.

    Simple forward-chaining cross-validation where each successive
    training set is a superset of the previous one.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 0,
        test_size: Optional[int] = None,
        gap: int = 0,
    ):
        """
        Initialize TimeSeriesSplit.

        Args:
            n_splits: Number of splits
            purge_gap: Bars to purge between train and test
            test_size: Fixed test size (if None, calculated automatically)
            gap: Additional gap between train and test (in addition to purge)
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.test_size = test_size
        self.gap = gap

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices for time-series splits."""
        n_samples = len(X)
        indices = np.arange(n_samples)

        if self.test_size is not None:
            test_size = self.test_size
        else:
            test_size = n_samples // (self.n_splits + 1)

        total_gap = self.purge_gap + self.gap

        test_starts = []
        for i in range(self.n_splits):
            test_start = n_samples - (self.n_splits - i) * test_size
            test_starts.append(test_start)

        for test_start in test_starts:
            test_end = test_start + test_size

            train_end = test_start - total_gap
            if train_end <= 0:
                continue

            train_indices = indices[:train_end]
            test_indices = indices[test_start:test_end]

            yield train_indices, test_indices


# Utility functions
def calculate_purge_gap(
    prediction_horizon: int,
    max_lookback: int,
    buffer: int = 0,
) -> int:
    """
    Calculate appropriate purge gap for cross-validation.

    Args:
        prediction_horizon: How many bars ahead the model predicts
        max_lookback: Maximum lookback used in feature calculation
        buffer: Additional buffer for safety

    Returns:
        Recommended purge gap
    """
    return prediction_horizon + max_lookback + buffer


def validate_cv_splits(
    cv,
    X: Union[pd.DataFrame, np.ndarray],
    min_train_samples: int = 100,
    min_test_samples: int = 20,
) -> bool:
    """
    Validate that CV splits meet minimum requirements.

    Args:
        cv: Cross-validation splitter
        X: Data to split
        min_train_samples: Minimum training samples per fold
        min_test_samples: Minimum test samples per fold

    Returns:
        True if all splits are valid
    """
    for i, (train_idx, test_idx) in enumerate(cv.split(X)):
        if len(train_idx) < min_train_samples:
            logger.warning(
                f"Fold {i}: Only {len(train_idx)} train samples "
                f"(min: {min_train_samples})"
            )
            return False

        if len(test_idx) < min_test_samples:
            logger.warning(
                f"Fold {i}: Only {len(test_idx)} test samples "
                f"(min: {min_test_samples})"
            )
            return False

    return True


def check_cv_leakage(
    cv,
    X: Union[pd.DataFrame, np.ndarray],
    purge_gap: Optional[int] = None,
) -> dict:
    """
    Check cross-validation splits for potential data leakage.

    Validates that:
    1. No overlap between train and test indices
    2. Proper temporal ordering is maintained
    3. Purge gap is respected

    Args:
        cv: Cross-validation splitter
        X: Data to split
        purge_gap: Expected minimum gap between train max and test min

    Returns:
        Dictionary with leakage check results
    """
    results = {
        'has_leakage': False,
        'issues': [],
        'warnings': [],
        'splits_checked': 0,
        'all_gaps': [],
    }

    # Get purge_gap from cv if available
    if purge_gap is None and hasattr(cv, 'purge_gap'):
        purge_gap = cv.purge_gap

    for i, (train_idx, test_idx) in enumerate(cv.split(X)):
        results['splits_checked'] += 1

        # Check 1: No overlap
        train_set = set(train_idx)
        test_set = set(test_idx)
        overlap = train_set.intersection(test_set)

        if len(overlap) > 0:
            results['has_leakage'] = True
            results['issues'].append(
                f"Fold {i}: Found {len(overlap)} overlapping indices"
            )

        # Check 2: Temporal ordering
        if len(train_idx) > 0 and len(test_idx) > 0:
            train_max = np.max(train_idx)
            test_min = np.min(test_idx)
            gap = test_min - train_max

            results['all_gaps'].append(gap)

            # Gap must be positive
            if gap <= 0:
                results['has_leakage'] = True
                results['issues'].append(
                    f"Fold {i}: Test indices ({test_min}) not after train indices ({train_max})"
                )

            # Check purge gap
            if purge_gap is not None and gap < purge_gap:
                results['warnings'].append(
                    f"Fold {i}: Gap ({gap}) is less than purge_gap ({purge_gap}). "
                    f"Risk of look-ahead bias."
                )

    # Summary statistics
    if results['all_gaps']:
        results['min_gap'] = min(results['all_gaps'])
        results['max_gap'] = max(results['all_gaps'])
        results['mean_gap'] = np.mean(results['all_gaps'])

    return results


def validate_cv_for_finance(
    cv,
    X: Union[pd.DataFrame, np.ndarray],
    prediction_horizon: int = 5,
    max_feature_lookback: int = 200,
    buffer: int = 10,
) -> dict:
    """
    Comprehensive validation for financial time-series cross-validation.

    Checks that the CV configuration is appropriate for preventing
    data leakage in financial ML.

    Args:
        cv: Cross-validation splitter
        X: Data to split
        prediction_horizon: How many bars ahead predictions target
        max_feature_lookback: Maximum lookback in feature engineering
        buffer: Additional safety buffer

    Returns:
        Dictionary with validation results and recommendations
    """
    # Calculate recommended purge gap
    recommended_purge = prediction_horizon + max_feature_lookback + buffer

    # Check current purge gap
    current_purge = getattr(cv, 'purge_gap', None)

    results = {
        'valid': True,
        'issues': [],
        'recommendations': [],
        'recommended_purge_gap': recommended_purge,
        'current_purge_gap': current_purge,
    }

    # Check if purge gap is sufficient
    if current_purge is not None and current_purge < recommended_purge:
        results['valid'] = False
        results['issues'].append(
            f"Purge gap ({current_purge}) is less than recommended "
            f"({recommended_purge} = {prediction_horizon} + {max_feature_lookback} + {buffer}). "
            f"This may cause look-ahead bias."
        )
        results['recommendations'].append(
            f"Set purge_gap >= {recommended_purge}"
        )

    # Run leakage check
    leakage_results = check_cv_leakage(cv, X, purge_gap=recommended_purge)

    if leakage_results['has_leakage']:
        results['valid'] = False
        results['issues'].extend(leakage_results['issues'])

    if leakage_results['warnings']:
        results['recommendations'].extend(
            [f"Address warning: {w}" for w in leakage_results['warnings']]
        )

    # Check split sizes
    split_valid = validate_cv_splits(cv, X)
    if not split_valid:
        results['valid'] = False
        results['issues'].append("Some CV splits have insufficient samples")

    results['leakage_check'] = leakage_results

    return results
