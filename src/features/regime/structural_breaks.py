"""
Structural Break Detection.

This module implements tests for detecting structural breaks (regime changes)
in time series data using:
1. CUSUM tests (cumulative sum)
2. Bai-Perron tests for multiple breaks
3. Chow test for known break points

Reference:
    Bai, J. and Perron, P. (2003) - "Computation and analysis of multiple
    structural change models"

Designed for JPMorgan-level institutional requirements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class BreakPoint:
    """Represents a detected structural break."""

    index: int                    # Index of break point
    date: pd.Timestamp           # Date of break
    statistic: float             # Test statistic value
    p_value: float               # P-value (if available)
    confidence: float            # Confidence level
    direction: str               # "increase" or "decrease"
    magnitude: float             # Size of the break


@dataclass
class CUSUMResult:
    """Container for CUSUM test results."""

    cusum: pd.Series             # CUSUM values over time
    upper_bound: pd.Series       # Upper significance bound
    lower_bound: pd.Series       # Lower significance bound
    breaks: List[BreakPoint]     # Detected break points
    is_stable: bool              # Whether series is stable


@dataclass
class StructuralBreakResult:
    """Container for structural break analysis."""

    breaks: List[BreakPoint]     # All detected breaks
    n_breaks: int                # Number of breaks found
    break_dates: List[pd.Timestamp]  # Dates of breaks
    segments: List[Tuple[pd.Timestamp, pd.Timestamp]]  # Stable segments
    bic: float                   # Model selection criterion


class CUSUMTest:
    """
    CUSUM (Cumulative Sum) test for structural change.

    Tests for parameter instability by tracking cumulative deviations
    from expected values. A break is detected when CUSUM exceeds
    significance bounds.

    Example usage:
        test = CUSUMTest(significance=0.05)
        result = test.test(returns)

        if not result.is_stable:
            print(f"Breaks detected at: {result.breaks}")
    """

    def __init__(
        self,
        significance: float = 0.05,
        method: str = "ols",
        min_segment: int = 20,
    ) -> None:
        """
        Initialize CUSUM test.

        Args:
            significance: Significance level for bounds
            method: "ols" for OLS-based CUSUM, "recursive" for recursive residuals
            min_segment: Minimum segment size
        """
        self.significance = significance
        self.method = method
        self.min_segment = min_segment

    def test(self, series: pd.Series) -> CUSUMResult:
        """
        Perform CUSUM test.

        Args:
            series: Time series to test

        Returns:
            CUSUMResult with test statistics and detected breaks
        """
        n = len(series)
        values = series.values

        if n < 2 * self.min_segment:
            logger.warning(f"Series too short for CUSUM test: {n}")
            return CUSUMResult(
                cusum=pd.Series(dtype=float),
                upper_bound=pd.Series(dtype=float),
                lower_bound=pd.Series(dtype=float),
                breaks=[],
                is_stable=True,
            )

        # Calculate CUSUM
        if self.method == "ols":
            cusum = self._ols_cusum(values)
        else:
            cusum = self._recursive_cusum(values)

        # Calculate significance bounds
        upper, lower = self._calculate_bounds(n)

        # Detect breaks
        breaks = self._detect_breaks(cusum, upper, lower, series.index)

        # Determine stability
        is_stable = len(breaks) == 0

        return CUSUMResult(
            cusum=pd.Series(cusum, index=series.index),
            upper_bound=pd.Series(upper, index=series.index),
            lower_bound=pd.Series(lower, index=series.index),
            breaks=breaks,
            is_stable=is_stable,
        )

    def _ols_cusum(self, values: np.ndarray) -> np.ndarray:
        """Calculate OLS-based CUSUM."""
        n = len(values)
        mean = values.mean()
        std = values.std()

        if std == 0:
            return np.zeros(n)

        # Standardized deviations
        deviations = (values - mean) / std

        # Cumulative sum
        cusum = np.cumsum(deviations) / np.sqrt(n)

        return cusum

    def _recursive_cusum(self, values: np.ndarray) -> np.ndarray:
        """Calculate recursive residual CUSUM."""
        n = len(values)
        k = self.min_segment  # Initial estimation window

        recursive_residuals = []

        for t in range(k, n):
            # Estimate mean from past data
            past_mean = values[:t].mean()
            past_std = values[:t].std()

            if past_std > 0:
                residual = (values[t] - past_mean) / past_std
            else:
                residual = 0

            recursive_residuals.append(residual)

        # Pad with zeros for initial period
        cusum_values = [0] * k + list(np.cumsum(recursive_residuals) / np.sqrt(n - k))

        return np.array(cusum_values)

    def _calculate_bounds(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate significance bounds."""
        # Critical value based on significance level
        if self.significance == 0.01:
            a = 1.63
        elif self.significance == 0.05:
            a = 1.36
        elif self.significance == 0.10:
            a = 1.22
        else:
            # Use normal approximation
            a = stats.norm.ppf(1 - self.significance / 2)

        # Time-varying bounds
        t = np.arange(1, n + 1)
        bound = a * np.sqrt(t / n)

        return bound, -bound

    def _detect_breaks(
        self,
        cusum: np.ndarray,
        upper: np.ndarray,
        lower: np.ndarray,
        index: pd.Index,
    ) -> List[BreakPoint]:
        """Detect break points where CUSUM crosses bounds."""
        breaks = []

        # Find crossings
        crossed_upper = cusum > upper
        crossed_lower = cusum < lower

        # Find first crossing points
        if crossed_upper.any():
            first_upper = np.where(crossed_upper)[0][0]
            breaks.append(BreakPoint(
                index=first_upper,
                date=index[first_upper],
                statistic=cusum[first_upper],
                p_value=np.nan,
                confidence=1 - self.significance,
                direction="increase",
                magnitude=cusum[first_upper] - upper[first_upper],
            ))

        if crossed_lower.any():
            first_lower = np.where(crossed_lower)[0][0]
            breaks.append(BreakPoint(
                index=first_lower,
                date=index[first_lower],
                statistic=cusum[first_lower],
                p_value=np.nan,
                confidence=1 - self.significance,
                direction="decrease",
                magnitude=lower[first_lower] - cusum[first_lower],
            ))

        return sorted(breaks, key=lambda x: x.index)


class StructuralBreakDetector:
    """
    Comprehensive structural break detector.

    Combines multiple methods:
    1. CUSUM test for gradual changes
    2. Supremum Wald test for abrupt changes
    3. Information criteria for optimal number of breaks

    Example usage:
        detector = StructuralBreakDetector(max_breaks=5)
        result = detector.detect(returns)

        for bp in result.breaks:
            print(f"Break at {bp.date}: {bp.direction}")
    """

    def __init__(
        self,
        max_breaks: int = 5,
        min_segment: int = 20,
        significance: float = 0.05,
        trim_pct: float = 0.15,
    ) -> None:
        """
        Initialize detector.

        Args:
            max_breaks: Maximum number of breaks to detect
            min_segment: Minimum segment size
            significance: Significance level
            trim_pct: Percentage to trim from ends
        """
        self.max_breaks = max_breaks
        self.min_segment = min_segment
        self.significance = significance
        self.trim_pct = trim_pct

        self._cusum_test = CUSUMTest(significance=significance, min_segment=min_segment)

    def detect(
        self,
        series: pd.Series,
        method: str = "bic",
    ) -> StructuralBreakResult:
        """
        Detect structural breaks.

        Args:
            series: Time series to analyze
            method: Break selection method ("bic", "aic", "sequential")

        Returns:
            StructuralBreakResult
        """
        n = len(series)
        values = series.values

        if n < 2 * self.min_segment:
            return StructuralBreakResult(
                breaks=[],
                n_breaks=0,
                break_dates=[],
                segments=[(series.index[0], series.index[-1])],
                bic=np.nan,
            )

        # Calculate Wald statistics for all potential break points
        trim = int(n * self.trim_pct)
        potential_breaks = range(trim, n - trim)

        wald_stats = []
        for k in potential_breaks:
            stat = self._wald_statistic(values, k)
            wald_stats.append((k, stat))

        # Find breaks using specified method
        if method == "sequential":
            breaks = self._sequential_detection(wald_stats, series)
        else:
            breaks = self._information_criterion_detection(wald_stats, series, method)

        # Create segments
        segments = self._create_segments(breaks, series)

        # Calculate BIC for selected model
        bic = self._calculate_bic(values, [b.index for b in breaks])

        return StructuralBreakResult(
            breaks=breaks,
            n_breaks=len(breaks),
            break_dates=[b.date for b in breaks],
            segments=segments,
            bic=bic,
        )

    def _wald_statistic(self, values: np.ndarray, k: int) -> float:
        """Calculate Wald statistic for break at index k."""
        n = len(values)

        # Split data
        before = values[:k]
        after = values[k:]

        # Calculate means
        mean_before = before.mean()
        mean_after = after.mean()

        # Calculate pooled variance
        var_before = before.var()
        var_after = after.var()
        pooled_var = ((k - 1) * var_before + (n - k - 1) * var_after) / (n - 2)

        if pooled_var == 0:
            return 0

        # Wald statistic
        wald = ((mean_after - mean_before) ** 2) / (pooled_var * (1/k + 1/(n-k)))

        return wald

    def _sequential_detection(
        self,
        wald_stats: List[Tuple[int, float]],
        series: pd.Series,
    ) -> List[BreakPoint]:
        """Sequential break detection."""
        breaks = []
        n = len(series)

        # Critical value for Wald test
        crit_value = stats.chi2.ppf(1 - self.significance, df=1)

        # Sort by Wald statistic
        sorted_stats = sorted(wald_stats, key=lambda x: x[1], reverse=True)

        for idx, stat in sorted_stats:
            if len(breaks) >= self.max_breaks:
                break

            if stat < crit_value:
                break

            # Check minimum distance from existing breaks
            min_dist = self.min_segment
            valid = True
            for existing in breaks:
                if abs(idx - existing.index) < min_dist:
                    valid = False
                    break

            if valid:
                # Determine direction
                before_mean = series.iloc[:idx].mean()
                after_mean = series.iloc[idx:].mean()
                direction = "increase" if after_mean > before_mean else "decrease"

                breaks.append(BreakPoint(
                    index=idx,
                    date=series.index[idx],
                    statistic=stat,
                    p_value=1 - stats.chi2.cdf(stat, df=1),
                    confidence=1 - self.significance,
                    direction=direction,
                    magnitude=abs(after_mean - before_mean),
                ))

        return sorted(breaks, key=lambda x: x.index)

    def _information_criterion_detection(
        self,
        wald_stats: List[Tuple[int, float]],
        series: pd.Series,
        criterion: str,
    ) -> List[BreakPoint]:
        """Select breaks using information criterion."""
        values = series.values
        n = len(values)

        best_breaks = []
        best_ic = self._calculate_ic(values, [], criterion)

        # Try different numbers of breaks
        for m in range(1, self.max_breaks + 1):
            # Get top m candidates
            sorted_stats = sorted(wald_stats, key=lambda x: x[1], reverse=True)
            candidate_indices = [s[0] for s in sorted_stats[:m * 2]]

            # Find best combination
            from itertools import combinations
            best_combo = None
            best_combo_ic = float('inf')

            for combo in combinations(candidate_indices, m):
                # Check minimum spacing
                combo_sorted = sorted(combo)
                valid = True
                for i in range(len(combo_sorted) - 1):
                    if combo_sorted[i + 1] - combo_sorted[i] < self.min_segment:
                        valid = False
                        break

                if valid:
                    ic = self._calculate_ic(values, list(combo), criterion)
                    if ic < best_combo_ic:
                        best_combo_ic = ic
                        best_combo = combo_sorted

            if best_combo and best_combo_ic < best_ic:
                best_ic = best_combo_ic
                best_breaks = []
                for idx in best_combo:
                    stat = dict(wald_stats).get(idx, 0)
                    before_mean = series.iloc[:idx].mean()
                    after_mean = series.iloc[idx:].mean()
                    direction = "increase" if after_mean > before_mean else "decrease"

                    best_breaks.append(BreakPoint(
                        index=idx,
                        date=series.index[idx],
                        statistic=stat,
                        p_value=1 - stats.chi2.cdf(stat, df=1) if stat > 0 else 1.0,
                        confidence=1 - self.significance,
                        direction=direction,
                        magnitude=abs(after_mean - before_mean),
                    ))

        return best_breaks

    def _calculate_ic(
        self,
        values: np.ndarray,
        break_indices: List[int],
        criterion: str,
    ) -> float:
        """Calculate information criterion for model with given breaks."""
        n = len(values)

        # Calculate residual sum of squares
        rss = 0
        segments = [0] + sorted(break_indices) + [n]

        for i in range(len(segments) - 1):
            start, end = segments[i], segments[i + 1]
            segment = values[start:end]
            segment_mean = segment.mean()
            rss += np.sum((segment - segment_mean) ** 2)

        # Number of parameters (one mean per segment)
        k = len(break_indices) + 1

        # Log-likelihood (assuming normal errors)
        sigma2 = rss / n
        if sigma2 == 0:
            sigma2 = 1e-10
        ll = -n / 2 * (np.log(2 * np.pi) + np.log(sigma2) + 1)

        # Information criterion
        if criterion == "bic":
            return -2 * ll + k * np.log(n)
        elif criterion == "aic":
            return -2 * ll + 2 * k
        else:
            return -2 * ll + k * np.log(n)

    def _calculate_bic(self, values: np.ndarray, break_indices: List[int]) -> float:
        """Calculate BIC for final model."""
        return self._calculate_ic(values, break_indices, "bic")

    def _create_segments(
        self,
        breaks: List[BreakPoint],
        series: pd.Series,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Create list of stable segments."""
        if not breaks:
            return [(series.index[0], series.index[-1])]

        segments = []
        break_indices = [0] + [b.index for b in breaks] + [len(series) - 1]

        for i in range(len(break_indices) - 1):
            start_idx = break_indices[i]
            end_idx = break_indices[i + 1]
            segments.append((series.index[start_idx], series.index[end_idx]))

        return segments


def detect_structural_breaks(
    series: pd.Series,
    max_breaks: int = 5,
) -> StructuralBreakResult:
    """
    Convenience function for structural break detection.

    Args:
        series: Time series to analyze
        max_breaks: Maximum number of breaks

    Returns:
        StructuralBreakResult
    """
    detector = StructuralBreakDetector(max_breaks=max_breaks)
    return detector.detect(series)


def cusum_test(series: pd.Series, significance: float = 0.05) -> CUSUMResult:
    """
    Convenience function for CUSUM test.

    Args:
        series: Time series to test
        significance: Significance level

    Returns:
        CUSUMResult
    """
    test = CUSUMTest(significance=significance)
    return test.test(series)
