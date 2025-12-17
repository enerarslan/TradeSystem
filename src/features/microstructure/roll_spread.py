"""
Roll Spread and Amihud Illiquidity Measures.

This module implements classic microstructure measures:
1. Roll Spread - Effective bid-ask spread from price reversals
2. Amihud Illiquidity - Price impact per dollar volume

Reference:
    Roll, R. (1984) - "A Simple Implicit Measure of the Effective Bid-Ask Spread"
    Amihud, Y. (2002) - "Illiquidity and Stock Returns"

Designed for JPMorgan-level institutional requirements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RollSpreadResult:
    """Container for Roll spread results."""

    spread: pd.Series              # Roll spread estimate
    spread_pct: pd.Series          # Spread as percentage of price
    valid_ratio: float             # Ratio of valid (non-NaN) estimates


@dataclass
class AmihudResult:
    """Container for Amihud illiquidity results."""

    illiquidity: pd.Series         # Amihud illiquidity
    illiquidity_log: pd.Series     # Log Amihud (more normally distributed)
    relative_illiquidity: pd.Series  # Relative to rolling average


class RollSpread:
    """
    Roll Spread estimator.

    Estimates the effective bid-ask spread using the covariance of
    consecutive price changes. Based on the insight that:
    - Buy orders push price to ask
    - Sell orders push price to bid
    - This creates negative serial correlation

    Roll spread = 2 * sqrt(-cov(ΔP_t, ΔP_{t-1}))

    Example usage:
        estimator = RollSpread(lookback=20)
        result = estimator.estimate(close)

        # Use as transaction cost estimate
        estimated_cost = result.spread_pct.iloc[-1]
    """

    def __init__(
        self,
        lookback: int = 20,
        min_periods: int = 10,
    ) -> None:
        """
        Initialize Roll spread estimator.

        Args:
            lookback: Rolling window for estimation
            min_periods: Minimum periods for valid estimate
        """
        self.lookback = lookback
        self.min_periods = min_periods

    def estimate(self, close: pd.Series) -> RollSpreadResult:
        """
        Estimate Roll spread.

        Args:
            close: Close price series

        Returns:
            RollSpreadResult with spread estimates
        """
        # Calculate price changes
        price_change = close.diff()

        # Calculate rolling covariance of consecutive changes
        rolling_cov = self._rolling_covariance(price_change, price_change.shift(1))

        # Roll spread = 2 * sqrt(-cov) when cov < 0
        # When cov >= 0, set spread to 0 (no bid-ask bounce detected)
        spread = pd.Series(index=close.index, dtype=float)

        negative_cov_mask = rolling_cov < 0
        spread[negative_cov_mask] = 2 * np.sqrt(-rolling_cov[negative_cov_mask])
        spread[~negative_cov_mask] = 0

        # Spread as percentage of price
        spread_pct = spread / close

        # Calculate valid ratio
        valid_ratio = spread.notna().sum() / len(spread)

        return RollSpreadResult(
            spread=spread,
            spread_pct=spread_pct,
            valid_ratio=valid_ratio,
        )

    def _rolling_covariance(
        self,
        x: pd.Series,
        y: pd.Series,
    ) -> pd.Series:
        """Calculate rolling covariance."""
        # Combine into DataFrame for rolling calculation
        df = pd.DataFrame({"x": x, "y": y})

        def cov_func(window):
            return window["x"].cov(window["y"])

        return df.rolling(
            window=self.lookback,
            min_periods=self.min_periods,
        ).apply(lambda w: w.iloc[:, 0].cov(w.iloc[:, 1]), raw=False).iloc[:, 0]


class AmihudIlliquidity:
    """
    Amihud Illiquidity Ratio estimator.

    Measures the daily price impact per dollar volume:
        Illiquidity = |return| / dollar_volume

    Higher values indicate less liquid markets (more price impact).

    Example usage:
        estimator = AmihudIlliquidity(lookback=20)
        result = estimator.estimate(returns, volume, close)

        # Use to adjust position sizing
        if result.relative_illiquidity.iloc[-1] > 2.0:
            reduce_order_size()
    """

    def __init__(
        self,
        lookback: int = 20,
        scale: float = 1e6,
    ) -> None:
        """
        Initialize Amihud illiquidity estimator.

        Args:
            lookback: Rolling window for averaging
            scale: Scaling factor (typically 1e6 for million)
        """
        self.lookback = lookback
        self.scale = scale

    def estimate(
        self,
        returns: pd.Series,
        volume: pd.Series,
        close: pd.Series,
    ) -> AmihudResult:
        """
        Estimate Amihud illiquidity.

        Args:
            returns: Return series
            volume: Volume series
            close: Close prices

        Returns:
            AmihudResult with illiquidity measures
        """
        # Dollar volume
        dollar_volume = volume * close

        # Daily Amihud ratio
        # Scale by 1e6 to get reasonable numbers
        daily_amihud = (returns.abs() / dollar_volume) * self.scale

        # Rolling average (more stable)
        illiquidity = daily_amihud.rolling(
            window=self.lookback,
            min_periods=5,
        ).mean()

        # Log transformation (more normally distributed)
        illiquidity_log = np.log(illiquidity.replace(0, np.nan))

        # Relative illiquidity (vs rolling mean)
        long_term_avg = illiquidity.rolling(
            window=self.lookback * 5,
            min_periods=self.lookback,
        ).mean()
        relative = illiquidity / long_term_avg

        return AmihudResult(
            illiquidity=illiquidity,
            illiquidity_log=illiquidity_log,
            relative_illiquidity=relative,
        )


def calculate_roll_spread(close: pd.Series, lookback: int = 20) -> pd.Series:
    """
    Convenience function to calculate Roll spread.

    Args:
        close: Close prices
        lookback: Rolling window

    Returns:
        Roll spread series (as percentage of price)
    """
    estimator = RollSpread(lookback=lookback)
    result = estimator.estimate(close)
    return result.spread_pct


def calculate_amihud(
    returns: pd.Series,
    volume: pd.Series,
    close: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """
    Convenience function to calculate Amihud illiquidity.

    Args:
        returns: Return series
        volume: Volume series
        close: Close prices
        lookback: Rolling window

    Returns:
        Amihud illiquidity series
    """
    estimator = AmihudIlliquidity(lookback=lookback)
    result = estimator.estimate(returns, volume, close)
    return result.illiquidity


def calculate_liquidity_score(
    close: pd.Series,
    volume: pd.Series,
    returns: pd.Series,
) -> pd.Series:
    """
    Calculate composite liquidity score.

    Combines Roll spread and Amihud into single score.
    Lower score = more liquid.

    Args:
        close: Close prices
        volume: Volume
        returns: Returns

    Returns:
        Composite liquidity score (normalized)
    """
    # Roll spread
    roll = RollSpread(lookback=20)
    roll_result = roll.estimate(close)
    roll_z = (roll_result.spread_pct - roll_result.spread_pct.rolling(60).mean()) / \
             roll_result.spread_pct.rolling(60).std()

    # Amihud
    amihud = AmihudIlliquidity(lookback=20)
    amihud_result = amihud.estimate(returns, volume, close)
    amihud_z = (amihud_result.illiquidity_log - amihud_result.illiquidity_log.rolling(60).mean()) / \
               amihud_result.illiquidity_log.rolling(60).std()

    # Composite (average of z-scores)
    liquidity_score = (roll_z.fillna(0) + amihud_z.fillna(0)) / 2

    return liquidity_score
