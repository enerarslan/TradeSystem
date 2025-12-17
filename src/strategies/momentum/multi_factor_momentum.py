"""
Multi-Factor Momentum Strategy for AlphaTrade system.

This strategy combines:
- Price momentum
- Volume momentum
- Volatility-adjusted momentum
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.strategies.base import BaseStrategy


class MultiFactorMomentumStrategy(BaseStrategy):
    """
    Multi-factor momentum strategy.

    Combines multiple momentum factors:
    - Price momentum across multiple lookback periods
    - Volume momentum (buying pressure)
    - Volatility-adjusted momentum (risk-adjusted)

    Ranks stocks and goes long top performers.
    """

    DEFAULT_PARAMS = {
        "lookback_periods": [5, 10, 20, 60],
        "price_momentum_weight": 0.4,
        "volume_momentum_weight": 0.3,
        "vol_adj_momentum_weight": 0.3,
        "top_n_long": 5,
        "bottom_n_short": 0,
        "min_momentum_threshold": 0.0,
        "rebalance_frequency": 1,  # Every N bars
    }

    def __init__(
        self,
        name: str = "MultiFactorMomentum",
        params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the strategy.

        Args:
            name: Strategy name
            params: Strategy parameters
        """
        merged_params = self.DEFAULT_PARAMS.copy()
        if params:
            merged_params.update(params)

        super().__init__(name, merged_params)

    def _calculate_price_momentum(
        self,
        prices: pd.DataFrame,
        lookbacks: list[int],
    ) -> pd.DataFrame:
        """
        Calculate price momentum across multiple lookback periods.

        Args:
            prices: Price panel (timestamps x symbols)
            lookbacks: Lookback periods

        Returns:
            Momentum scores
        """
        momentum_scores = []

        for lookback in lookbacks:
            # Calculate returns
            returns = prices.pct_change(lookback)

            # Rank cross-sectionally
            ranked = returns.rank(axis=1, pct=True) - 0.5

            momentum_scores.append(ranked)

        # Average across lookbacks
        combined = sum(momentum_scores) / len(momentum_scores)
        return combined

    def _calculate_volume_momentum(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        lookback: int = 20,
    ) -> pd.DataFrame:
        """
        Calculate volume momentum (buying pressure).

        Args:
            prices: Price panel
            volumes: Volume panel
            lookback: Lookback period

        Returns:
            Volume momentum scores
        """
        # Price direction
        price_change = prices.diff()

        # Volume on up days vs down days
        up_volume = volumes.where(price_change > 0, 0).rolling(lookback).sum()
        down_volume = volumes.where(price_change < 0, 0).rolling(lookback).sum()

        # Volume momentum ratio
        vol_momentum = (up_volume - down_volume) / (up_volume + down_volume + 1e-10)

        # Rank cross-sectionally
        ranked = vol_momentum.rank(axis=1, pct=True) - 0.5

        return ranked

    def _calculate_vol_adjusted_momentum(
        self,
        prices: pd.DataFrame,
        lookback: int = 20,
    ) -> pd.DataFrame:
        """
        Calculate volatility-adjusted momentum (Sharpe-like).

        Args:
            prices: Price panel
            lookback: Lookback period

        Returns:
            Risk-adjusted momentum scores
        """
        # Daily returns
        returns = prices.pct_change()

        # Rolling mean and std
        rolling_mean = returns.rolling(lookback).mean()
        rolling_std = returns.rolling(lookback).std()

        # Sharpe-like ratio
        sharpe = rolling_mean / (rolling_std + 1e-10)

        # Rank cross-sectionally
        ranked = sharpe.rank(axis=1, pct=True) - 0.5

        return ranked

    def generate_signals(
        self,
        data: pd.DataFrame | dict[str, pd.DataFrame],
        features: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """
        Generate momentum signals.

        Args:
            data: OHLCV data (dict of DataFrames by symbol)
            features: Pre-computed features (optional)

        Returns:
            Signal DataFrame (timestamps x symbols)
        """
        # Convert to price panel
        if isinstance(data, dict):
            prices = pd.DataFrame({sym: df["close"] for sym, df in data.items()})
            volumes = pd.DataFrame({sym: df["volume"] for sym, df in data.items()})
        else:
            # Assume multiindex DataFrame
            prices = data["close"].unstack()
            volumes = data["volume"].unstack()

        lookbacks = self.params["lookback_periods"]

        # Calculate momentum factors
        price_mom = self._calculate_price_momentum(prices, lookbacks)
        vol_mom = self._calculate_volume_momentum(prices, volumes)
        vol_adj_mom = self._calculate_vol_adjusted_momentum(prices)

        # Combine factors
        composite = (
            self.params["price_momentum_weight"] * price_mom
            + self.params["volume_momentum_weight"] * vol_mom
            + self.params["vol_adj_momentum_weight"] * vol_adj_mom
        )

        # Generate signals based on ranking
        signals = self._rank_to_signals(composite)

        logger.debug(f"Generated momentum signals: shape={signals.shape}")
        return signals

    def _rank_to_signals(self, scores: pd.DataFrame) -> pd.DataFrame:
        """
        Convert composite scores to trading signals.

        Args:
            scores: Composite momentum scores

        Returns:
            Signal DataFrame (-1, 0, 1)
        """
        signals = pd.DataFrame(0, index=scores.index, columns=scores.columns)

        top_n = self.params["top_n_long"]
        bottom_n = self.params["bottom_n_short"]
        threshold = self.params["min_momentum_threshold"]

        for idx in scores.index:
            row = scores.loc[idx].dropna()

            if len(row) == 0:
                continue

            # Apply threshold
            if threshold > 0:
                row = row[row.abs() > threshold]

            # Get top/bottom ranked stocks
            if len(row) > 0:
                sorted_symbols = row.sort_values(ascending=False)

                # Long top N
                for symbol in sorted_symbols.head(top_n).index:
                    signals.loc[idx, symbol] = 1

                # Short bottom N
                if bottom_n > 0:
                    for symbol in sorted_symbols.tail(bottom_n).index:
                        signals.loc[idx, symbol] = -1

        return signals

    def calculate_positions(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        capital: float,
    ) -> pd.DataFrame:
        """
        Convert signals to position weights.

        Args:
            signals: Signal DataFrame
            prices: Current prices
            capital: Available capital

        Returns:
            Position weights (equal weight among signals)
        """
        positions = signals.copy().astype(float)

        for idx in positions.index:
            row = positions.loc[idx]

            # Equal weight among positions
            n_long = (row > 0).sum()
            n_short = (row < 0).sum()

            if n_long > 0:
                positions.loc[idx, row > 0] = 1.0 / max(n_long, 1)

            if n_short > 0:
                positions.loc[idx, row < 0] = -1.0 / max(n_short, 1)

        return positions
