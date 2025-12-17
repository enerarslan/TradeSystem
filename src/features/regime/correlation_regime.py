"""
Correlation Regime Detection.

This module identifies changes in market correlation structure,
which is critical for:
1. Portfolio diversification (correlations spike in crises)
2. Factor exposure management
3. Risk parity strategies

Reference:
    "Risk Parity Fundamentals" by Qian (2016)

Designed for JPMorgan-level institutional requirements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CorrelationRegime(Enum):
    """Correlation regime states."""

    LOW_CORR = "low_correlation"         # Diversification works
    NORMAL_CORR = "normal_correlation"   # Typical market
    HIGH_CORR = "high_correlation"       # Stress building
    CRISIS_CORR = "crisis_correlation"   # All correlations spike


@dataclass
class CorrelationRegimeResult:
    """Container for correlation regime results."""

    regimes: pd.Series                   # Regime at each timestamp
    average_correlation: pd.Series       # Rolling average pairwise correlation
    correlation_dispersion: pd.Series    # Std dev of pairwise correlations
    eigenvalue_ratio: pd.Series          # First eigenvalue / sum (market factor)
    rolling_correlation: pd.DataFrame    # Rolling correlation matrix at each time


class CorrelationRegimeDetector:
    """
    Detects correlation regime changes in asset returns.

    Monitors:
    1. Average pairwise correlation
    2. First principal component dominance
    3. Correlation dispersion

    Crisis periods are characterized by correlation spike where
    all assets move together, destroying diversification benefits.

    Example usage:
        detector = CorrelationRegimeDetector(lookback=60)
        result = detector.detect(returns_panel)

        if result.regimes.iloc[-1] == CorrelationRegime.CRISIS_CORR:
            reduce_risk()
    """

    def __init__(
        self,
        lookback: int = 60,
        min_periods: int = 30,
        correlation_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize detector.

        Args:
            lookback: Rolling window for correlation calculation
            min_periods: Minimum periods for correlation calculation
            correlation_thresholds: Custom thresholds for regime classification
        """
        self.lookback = lookback
        self.min_periods = min_periods

        # Default thresholds for average correlation
        self.correlation_thresholds = correlation_thresholds or {
            "low": 0.2,
            "normal": 0.4,
            "high": 0.6,
        }

    def detect(
        self,
        returns: pd.DataFrame,
        method: str = "pearson",
    ) -> CorrelationRegimeResult:
        """
        Detect correlation regimes.

        Args:
            returns: Returns panel (timestamps x assets)
            method: Correlation method ("pearson", "spearman")

        Returns:
            CorrelationRegimeResult
        """
        # Calculate rolling average correlation
        avg_corr = self._rolling_average_correlation(returns, method)

        # Calculate correlation dispersion
        corr_disp = self._rolling_correlation_dispersion(returns, method)

        # Calculate eigenvalue ratio (first PC dominance)
        eigen_ratio = self._rolling_eigenvalue_ratio(returns)

        # Classify regimes
        regimes = self._classify_regimes(avg_corr, corr_disp)

        # Get final rolling correlation matrix
        rolling_corr = returns.rolling(
            window=self.lookback,
            min_periods=self.min_periods,
        ).corr()

        return CorrelationRegimeResult(
            regimes=regimes,
            average_correlation=avg_corr,
            correlation_dispersion=corr_disp,
            eigenvalue_ratio=eigen_ratio,
            rolling_correlation=rolling_corr,
        )

    def _rolling_average_correlation(
        self,
        returns: pd.DataFrame,
        method: str,
    ) -> pd.Series:
        """Calculate rolling average pairwise correlation."""

        def avg_corr(window):
            if len(window.dropna()) < self.min_periods:
                return np.nan
            corr_matrix = window.corr(method=method)
            # Get upper triangle (excluding diagonal)
            upper = corr_matrix.values[np.triu_indices(len(corr_matrix), k=1)]
            return np.nanmean(upper)

        return returns.rolling(
            window=self.lookback,
            min_periods=self.min_periods,
        ).apply(lambda x: avg_corr(returns.loc[x.index]), raw=False).mean(axis=1)

    def _rolling_correlation_dispersion(
        self,
        returns: pd.DataFrame,
        method: str,
    ) -> pd.Series:
        """Calculate rolling dispersion of pairwise correlations."""

        def corr_std(window):
            if len(window.dropna()) < self.min_periods:
                return np.nan
            corr_matrix = window.corr(method=method)
            upper = corr_matrix.values[np.triu_indices(len(corr_matrix), k=1)]
            return np.nanstd(upper)

        return returns.rolling(
            window=self.lookback,
            min_periods=self.min_periods,
        ).apply(lambda x: corr_std(returns.loc[x.index]), raw=False).mean(axis=1)

    def _rolling_eigenvalue_ratio(
        self,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """
        Calculate rolling first eigenvalue ratio.

        High ratio = market factor dominates = high correlation
        """
        ratios = []
        dates = []

        for i in range(self.lookback, len(returns)):
            window = returns.iloc[i - self.lookback:i]

            if window.dropna().shape[0] < self.min_periods:
                ratios.append(np.nan)
                dates.append(returns.index[i])
                continue

            # Calculate covariance and eigenvalues
            cov_matrix = window.cov()
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Descending

            # First eigenvalue ratio
            total = eigenvalues.sum()
            if total > 0:
                ratio = eigenvalues[0] / total
            else:
                ratio = np.nan

            ratios.append(ratio)
            dates.append(returns.index[i])

        return pd.Series(ratios, index=dates, name="eigenvalue_ratio")

    def _classify_regimes(
        self,
        avg_corr: pd.Series,
        corr_disp: pd.Series,
    ) -> pd.Series:
        """Classify into correlation regimes."""
        thresholds = self.correlation_thresholds

        def classify(corr):
            if pd.isna(corr):
                return CorrelationRegime.NORMAL_CORR
            if corr < thresholds["low"]:
                return CorrelationRegime.LOW_CORR
            elif corr < thresholds["normal"]:
                return CorrelationRegime.NORMAL_CORR
            elif corr < thresholds["high"]:
                return CorrelationRegime.HIGH_CORR
            else:
                return CorrelationRegime.CRISIS_CORR

        return avg_corr.apply(classify)

    def get_correlation_breakdown(
        self,
        returns: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Get detailed correlation breakdown at a specific time.

        Args:
            returns: Returns panel
            as_of: Timestamp for analysis

        Returns:
            Correlation matrix as of that date
        """
        idx = returns.index.get_loc(as_of, method="ffill")
        start_idx = max(0, idx - self.lookback)
        window = returns.iloc[start_idx:idx + 1]
        return window.corr()

    def detect_correlation_breakdowns(
        self,
        returns: pd.DataFrame,
        threshold_change: float = 0.2,
    ) -> pd.DataFrame:
        """
        Detect sudden correlation regime changes.

        Args:
            returns: Returns panel
            threshold_change: Minimum correlation change to flag

        Returns:
            DataFrame with breakdown events
        """
        result = self.detect(returns)
        avg_corr = result.average_correlation

        # Calculate correlation changes
        corr_change = avg_corr.diff()

        # Find large changes
        breakdowns = corr_change[corr_change.abs() > threshold_change]

        events = []
        for date, change in breakdowns.items():
            events.append({
                "date": date,
                "correlation_change": change,
                "new_correlation": avg_corr.loc[date],
                "new_regime": result.regimes.loc[date].value,
                "direction": "spike" if change > 0 else "collapse",
            })

        return pd.DataFrame(events)


def detect_correlation_regime(returns: pd.DataFrame) -> CorrelationRegimeResult:
    """
    Convenience function for correlation regime detection.

    Args:
        returns: Returns panel (timestamps x assets)

    Returns:
        CorrelationRegimeResult
    """
    detector = CorrelationRegimeDetector()
    return detector.detect(returns)
