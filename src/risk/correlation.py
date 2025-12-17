"""
Correlation analysis for AlphaTrade system.

This module provides:
- Correlation matrix calculation
- Correlation filtering
- Diversification analysis
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger


class CorrelationAnalyzer:
    """
    Portfolio correlation analysis.

    Provides correlation analysis for:
    - Position diversification
    - Risk monitoring
    - Concentration limits
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        lookback: int = 60,
    ) -> None:
        """
        Initialize the analyzer.

        Args:
            returns: Asset returns DataFrame
            lookback: Lookback period for correlation
        """
        self.returns = returns
        self.lookback = lookback
        self._corr_matrix: pd.DataFrame | None = None

    @property
    def correlation_matrix(self) -> pd.DataFrame:
        """Get correlation matrix."""
        if self._corr_matrix is None:
            self._corr_matrix = self.returns.tail(self.lookback).corr()
        return self._corr_matrix

    def get_highly_correlated_pairs(
        self,
        threshold: float = 0.7,
    ) -> list[tuple[str, str, float]]:
        """
        Find highly correlated asset pairs.

        Args:
            threshold: Correlation threshold

        Returns:
            List of (asset1, asset2, correlation) tuples
        """
        corr = self.correlation_matrix

        pairs = []
        for i, col1 in enumerate(corr.columns):
            for col2 in corr.columns[i + 1:]:
                correlation = corr.loc[col1, col2]
                if abs(correlation) >= threshold:
                    pairs.append((col1, col2, correlation))

        # Sort by correlation
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        return pairs

    def check_correlation_limits(
        self,
        positions: pd.Series,
        max_correlation: float = 0.7,
    ) -> dict:
        """
        Check if positions violate correlation limits.

        Args:
            positions: Current position weights
            max_correlation: Maximum allowed correlation

        Returns:
            Dictionary with violation information
        """
        violations = []

        # Get non-zero positions
        active = positions[positions != 0]

        for i, (sym1, _) in enumerate(active.items()):
            for sym2, _ in list(active.items())[i + 1:]:
                if sym1 in self.correlation_matrix.index and sym2 in self.correlation_matrix.index:
                    corr = self.correlation_matrix.loc[sym1, sym2]
                    if abs(corr) > max_correlation:
                        violations.append({
                            "asset1": sym1,
                            "asset2": sym2,
                            "correlation": corr,
                        })

        return {
            "has_violations": len(violations) > 0,
            "num_violations": len(violations),
            "violations": violations,
        }

    def apply_correlation_filter(
        self,
        positions: pd.Series,
        max_correlation: float = 0.7,
    ) -> pd.Series:
        """
        Reduce positions that are highly correlated.

        Args:
            positions: Position weights
            max_correlation: Maximum allowed correlation

        Returns:
            Filtered positions
        """
        filtered = positions.copy()
        active = positions[positions != 0]

        # Find correlated pairs
        for sym1 in active.index:
            for sym2 in active.index:
                if sym1 >= sym2:
                    continue

                if sym1 not in self.correlation_matrix.index or sym2 not in self.correlation_matrix.index:
                    continue

                corr = abs(self.correlation_matrix.loc[sym1, sym2])

                if corr > max_correlation:
                    # Reduce smaller position
                    if abs(filtered[sym1]) < abs(filtered[sym2]):
                        reduction = (corr - max_correlation) / (1 - max_correlation)
                        filtered[sym1] *= (1 - reduction)
                    else:
                        reduction = (corr - max_correlation) / (1 - max_correlation)
                        filtered[sym2] *= (1 - reduction)

        return filtered

    def get_average_correlation(self) -> float:
        """
        Get average pairwise correlation.

        Returns:
            Average correlation
        """
        corr = self.correlation_matrix
        n = len(corr)

        # Upper triangle excluding diagonal
        upper_tri = np.triu(corr.values, k=1)
        avg_corr = upper_tri.sum() / (n * (n - 1) / 2)

        return avg_corr

    def get_diversification_ratio(
        self,
        weights: pd.Series,
    ) -> float:
        """
        Calculate diversification ratio.

        DR = weighted average volatility / portfolio volatility
        Higher is more diversified.

        Args:
            weights: Portfolio weights

        Returns:
            Diversification ratio
        """
        # Asset volatilities
        vols = self.returns.tail(self.lookback).std()

        # Weighted average volatility
        weighted_vol = (weights.abs() * vols).sum()

        # Portfolio volatility
        cov = self.returns.tail(self.lookback).cov()
        port_vol = np.sqrt(weights.values @ cov.values @ weights.values)

        if port_vol == 0:
            return 1.0

        return weighted_vol / port_vol

    def get_effective_number_of_bets(
        self,
        weights: pd.Series,
    ) -> float:
        """
        Calculate effective number of independent bets.

        ENB = 1 / sum(weight_i^2)
        This is the inverse of the Herfindahl index.

        Args:
            weights: Portfolio weights

        Returns:
            Effective number of bets
        """
        # Normalize weights
        w = weights.abs() / weights.abs().sum()

        # Herfindahl index
        hhi = (w ** 2).sum()

        if hhi == 0:
            return 0.0

        return 1 / hhi

    def rolling_correlation(
        self,
        asset1: str,
        asset2: str,
        window: int | None = None,
    ) -> pd.Series:
        """
        Calculate rolling correlation between two assets.

        Args:
            asset1: First asset
            asset2: Second asset
            window: Rolling window

        Returns:
            Rolling correlation series
        """
        window = window or self.lookback

        return self.returns[asset1].rolling(window=window).corr(
            self.returns[asset2]
        )

    def get_correlation_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of correlations.

        Returns:
            Summary DataFrame
        """
        corr = self.correlation_matrix

        # Get upper triangle values
        upper_tri = []
        for i, col1 in enumerate(corr.columns):
            for col2 in corr.columns[i + 1:]:
                upper_tri.append(corr.loc[col1, col2])

        return pd.DataFrame({
            "statistic": ["mean", "median", "std", "min", "max", "pct_above_0.7", "pct_above_0.5"],
            "value": [
                np.mean(upper_tri),
                np.median(upper_tri),
                np.std(upper_tri),
                np.min(upper_tri),
                np.max(upper_tri),
                (np.array(upper_tri) > 0.7).mean() * 100,
                (np.array(upper_tri) > 0.5).mean() * 100,
            ],
        })


def calculate_correlation_matrix(
    returns: pd.DataFrame,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    lookback: int | None = None,
) -> pd.DataFrame:
    """
    Calculate correlation matrix.

    Args:
        returns: Asset returns DataFrame
        method: Correlation method
        lookback: Lookback period

    Returns:
        Correlation matrix
    """
    if lookback:
        returns = returns.tail(lookback)

    return returns.corr(method=method)


def get_sector_correlations(
    returns: pd.DataFrame,
    sectors: dict[str, list[str]],
    lookback: int = 60,
) -> pd.DataFrame:
    """
    Calculate sector-level correlations.

    Args:
        returns: Asset returns DataFrame
        sectors: Dictionary mapping sector names to symbols
        lookback: Lookback period

    Returns:
        Sector correlation matrix
    """
    sector_returns = {}

    for sector, symbols in sectors.items():
        valid_symbols = [s for s in symbols if s in returns.columns]
        if valid_symbols:
            sector_returns[sector] = returns[valid_symbols].mean(axis=1)

    sector_df = pd.DataFrame(sector_returns)

    return sector_df.tail(lookback).corr()
