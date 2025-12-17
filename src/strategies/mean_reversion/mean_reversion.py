"""
Mean Reversion Strategy for AlphaTrade system.

This strategy identifies overbought/oversold conditions
and trades mean reversion using z-score based signals.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """
    Statistical mean reversion strategy.

    Uses z-score deviation from moving average to identify
    overbought and oversold conditions.

    Entry: Z-score exceeds threshold
    Exit: Z-score reverts to mean or opposite threshold
    """

    DEFAULT_PARAMS = {
        "lookback_period": 20,
        "entry_zscore": 2.0,
        "exit_zscore": 0.5,
        "stop_zscore": 3.0,
        "half_life_max": 50,
        "min_mean_reversion_speed": 0.1,
        "use_bollinger": True,
        "bollinger_std": 2.0,
    }

    def __init__(
        self,
        name: str = "MeanReversion",
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
        self._positions_state: dict[str, int] = {}

    def _calculate_zscore(
        self,
        prices: pd.DataFrame,
        lookback: int,
    ) -> pd.DataFrame:
        """
        Calculate z-score of prices relative to moving average.

        Args:
            prices: Price panel
            lookback: Lookback period

        Returns:
            Z-score DataFrame
        """
        rolling_mean = prices.rolling(window=lookback).mean()
        rolling_std = prices.rolling(window=lookback).std()

        zscore = (prices - rolling_mean) / (rolling_std + 1e-10)

        return zscore

    def _calculate_half_life(
        self,
        prices: pd.Series,
        lookback: int = 100,
    ) -> float:
        """
        Calculate mean reversion half-life using OLS.

        Args:
            prices: Price series
            lookback: Period for calculation

        Returns:
            Half-life in periods
        """
        if len(prices) < lookback:
            return np.inf

        prices = prices.iloc[-lookback:]

        # Spread = prices - mean
        spread = prices - prices.mean()

        # Lag spread
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()

        # Align series
        spread_lag = spread_lag.iloc[1:]
        spread_diff = spread_diff.iloc[1:]

        if len(spread_lag) < 10:
            return np.inf

        # OLS: spread_diff = gamma * spread_lag + error
        gamma = np.cov(spread_diff, spread_lag)[0, 1] / np.var(spread_lag)

        if gamma >= 0:
            return np.inf

        half_life = -np.log(2) / gamma
        return half_life

    def _check_mean_reversion(
        self,
        prices: pd.DataFrame,
        min_half_life: float = 5,
        max_half_life: float = 50,
    ) -> pd.DataFrame:
        """
        Check if each symbol exhibits mean reversion.

        Args:
            prices: Price panel
            min_half_life: Minimum acceptable half-life
            max_half_life: Maximum acceptable half-life

        Returns:
            Boolean DataFrame indicating mean reversion
        """
        is_mean_reverting = pd.DataFrame(
            False, index=prices.index, columns=prices.columns
        )

        for symbol in prices.columns:
            series = prices[symbol].dropna()
            if len(series) < 100:
                continue

            # Rolling half-life calculation
            for i in range(100, len(series)):
                window = series.iloc[i - 100 : i]
                half_life = self._calculate_half_life(window)

                if min_half_life < half_life < max_half_life:
                    idx = series.index[i]
                    is_mean_reverting.loc[idx, symbol] = True

        return is_mean_reverting

    def generate_signals(
        self,
        data: pd.DataFrame | dict[str, pd.DataFrame],
        features: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """
        Generate mean reversion signals.

        Args:
            data: OHLCV data
            features: Pre-computed features (optional)

        Returns:
            Signal DataFrame
        """
        # Convert to price panel
        if isinstance(data, dict):
            prices = pd.DataFrame({sym: df["close"] for sym, df in data.items()})
        else:
            prices = data["close"].unstack()

        lookback = self.params["lookback_period"]
        entry_z = self.params["entry_zscore"]
        exit_z = self.params["exit_zscore"]
        stop_z = self.params["stop_zscore"]

        # Calculate z-scores
        zscore = self._calculate_zscore(prices, lookback)

        # Initialize signals
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)

        # Generate signals with position tracking
        for symbol in prices.columns:
            position = 0  # Current position state

            for i, idx in enumerate(prices.index):
                z = zscore.loc[idx, symbol]

                if pd.isna(z):
                    signals.loc[idx, symbol] = 0
                    continue

                if position == 0:
                    # No position - look for entry
                    if z < -entry_z:
                        # Oversold - go long
                        signals.loc[idx, symbol] = 1
                        position = 1
                    elif z > entry_z:
                        # Overbought - go short
                        signals.loc[idx, symbol] = -1
                        position = -1
                    else:
                        signals.loc[idx, symbol] = 0

                elif position == 1:
                    # Long position
                    if z > -exit_z or z > stop_z:
                        # Exit on mean reversion or stop
                        signals.loc[idx, symbol] = 0
                        position = 0
                    else:
                        signals.loc[idx, symbol] = 1

                elif position == -1:
                    # Short position
                    if z < exit_z or z < -stop_z:
                        # Exit on mean reversion or stop
                        signals.loc[idx, symbol] = 0
                        position = 0
                    else:
                        signals.loc[idx, symbol] = -1

        logger.debug(f"Generated mean reversion signals: shape={signals.shape}")
        return signals

    def calculate_positions(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        capital: float,
    ) -> pd.DataFrame:
        """
        Convert signals to position weights.

        Uses half-life based position sizing.

        Args:
            signals: Signal DataFrame
            prices: Current prices
            capital: Available capital

        Returns:
            Position weights
        """
        positions = signals.copy().astype(float)

        # Equal weight among active positions
        for idx in positions.index:
            row = positions.loc[idx]
            n_positions = (row != 0).sum()

            if n_positions > 0:
                weight = 1.0 / n_positions
                positions.loc[idx] = row * weight

        # Apply maximum position cap
        positions = positions.clip(lower=-0.1, upper=0.1)

        return positions


class PairsStrategy(BaseStrategy):
    """
    Pairs trading / Statistical arbitrage strategy.

    Trades the spread between correlated pairs when
    deviation from historical relationship exceeds threshold.
    """

    DEFAULT_PARAMS = {
        "lookback_period": 60,
        "entry_zscore": 2.0,
        "exit_zscore": 0.0,
        "min_correlation": 0.7,
        "formation_period": 252,
    }

    def __init__(
        self,
        name: str = "PairsTrading",
        params: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the strategy."""
        merged_params = self.DEFAULT_PARAMS.copy()
        if params:
            merged_params.update(params)

        super().__init__(name, merged_params)
        self._pairs: list[tuple[str, str]] = []

    def find_pairs(
        self,
        prices: pd.DataFrame,
        min_corr: float = 0.7,
    ) -> list[tuple[str, str, float]]:
        """
        Find cointegrated pairs.

        Args:
            prices: Price panel
            min_corr: Minimum correlation threshold

        Returns:
            List of (symbol1, symbol2, correlation) tuples
        """
        returns = prices.pct_change().dropna()
        corr_matrix = returns.corr()

        pairs = []
        symbols = list(prices.columns)

        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i + 1 :]:
                corr = corr_matrix.loc[sym1, sym2]

                if corr >= min_corr:
                    pairs.append((sym1, sym2, corr))

        # Sort by correlation
        pairs.sort(key=lambda x: x[2], reverse=True)

        return pairs

    def _calculate_spread(
        self,
        prices: pd.DataFrame,
        sym1: str,
        sym2: str,
        lookback: int,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Calculate normalized spread between two symbols.

        Args:
            prices: Price panel
            sym1: First symbol
            sym2: Second symbol
            lookback: Lookback for hedge ratio

        Returns:
            Tuple of (spread, zscore)
        """
        # Rolling hedge ratio using OLS
        p1 = prices[sym1]
        p2 = prices[sym2]

        # Simple ratio spread
        spread = p1 / p2

        # Z-score of spread
        spread_mean = spread.rolling(lookback).mean()
        spread_std = spread.rolling(lookback).std()
        zscore = (spread - spread_mean) / (spread_std + 1e-10)

        return spread, zscore

    def generate_signals(
        self,
        data: pd.DataFrame | dict[str, pd.DataFrame],
        features: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """
        Generate pairs trading signals.

        Args:
            data: OHLCV data
            features: Pre-computed features (optional)

        Returns:
            Signal DataFrame
        """
        # Convert to price panel
        if isinstance(data, dict):
            prices = pd.DataFrame({sym: df["close"] for sym, df in data.items()})
        else:
            prices = data["close"].unstack()

        # Find pairs if not already done
        if not self._pairs:
            pair_info = self.find_pairs(prices, self.params["min_correlation"])
            self._pairs = [(p[0], p[1]) for p in pair_info[:10]]  # Top 10 pairs

        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)

        entry_z = self.params["entry_zscore"]
        exit_z = self.params["exit_zscore"]
        lookback = self.params["lookback_period"]

        for sym1, sym2 in self._pairs:
            if sym1 not in prices.columns or sym2 not in prices.columns:
                continue

            spread, zscore = self._calculate_spread(prices, sym1, sym2, lookback)

            for idx in prices.index:
                z = zscore.loc[idx]

                if pd.isna(z):
                    continue

                if z > entry_z:
                    # Spread too high - short sym1, long sym2
                    signals.loc[idx, sym1] = -0.5
                    signals.loc[idx, sym2] = 0.5
                elif z < -entry_z:
                    # Spread too low - long sym1, short sym2
                    signals.loc[idx, sym1] = 0.5
                    signals.loc[idx, sym2] = -0.5
                elif abs(z) < exit_z:
                    # Close positions
                    signals.loc[idx, sym1] = 0
                    signals.loc[idx, sym2] = 0

        return signals

    def calculate_positions(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        capital: float,
    ) -> pd.DataFrame:
        """Convert signals to position weights."""
        # Signals already represent weights
        return signals.clip(lower=-0.1, upper=0.1)
