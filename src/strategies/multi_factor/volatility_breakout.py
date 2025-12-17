"""
Volatility Breakout Strategy for AlphaTrade system.

This strategy identifies breakouts from volatility-based
price channels and trades with the momentum.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.strategies.base import BaseStrategy


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    ATR-based volatility breakout strategy.

    Enters positions when price breaks out of ATR-based
    channels with volume confirmation.

    Features:
    - ATR-based entry threshold
    - Volume confirmation
    - Dynamic stop-loss
    - Trailing stops
    """

    DEFAULT_PARAMS = {
        "atr_period": 14,
        "atr_multiplier": 2.0,
        "volume_confirm_multiplier": 1.5,
        "stop_loss_atr": 1.5,
        "take_profit_atr": 3.0,
        "trailing_stop_atr": 1.0,
        "min_atr_threshold": 0.001,
        "lookback_high_low": 20,
    }

    def __init__(
        self,
        name: str = "VolatilityBreakout",
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
        self._position_state: dict[str, dict] = {}

    def _calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int,
    ) -> pd.Series:
        """
        Calculate Average True Range.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period

        Returns:
            ATR series
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()

        return atr

    def _check_volume_confirmation(
        self,
        volume: pd.Series,
        multiplier: float,
        period: int = 20,
    ) -> pd.Series:
        """
        Check if volume confirms the breakout.

        Args:
            volume: Volume series
            multiplier: Volume multiplier threshold
            period: Moving average period

        Returns:
            Boolean series for volume confirmation
        """
        volume_ma = volume.rolling(window=period).mean()
        return volume > (volume_ma * multiplier)

    def _generate_single_stock_signals(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> pd.Series:
        """
        Generate signals for a single stock.

        Args:
            df: OHLCV DataFrame
            symbol: Stock symbol

        Returns:
            Signal series
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df.get("volume", pd.Series(1, index=df.index))

        atr_period = self.params["atr_period"]
        atr_mult = self.params["atr_multiplier"]
        vol_mult = self.params["volume_confirm_multiplier"]
        lookback = self.params["lookback_high_low"]

        # Calculate ATR
        atr = self._calculate_atr(high, low, close, atr_period)

        # Calculate breakout levels
        highest_high = high.rolling(window=lookback).max()
        lowest_low = low.rolling(window=lookback).min()

        # ATR-based channels
        upper_band = close.shift(1) + (atr * atr_mult)
        lower_band = close.shift(1) - (atr * atr_mult)

        # Volume confirmation
        vol_confirm = self._check_volume_confirmation(volume, vol_mult)

        # Initialize signals
        signals = pd.Series(0, index=df.index)

        # Initialize position tracking
        if symbol not in self._position_state:
            self._position_state[symbol] = {
                "position": 0,
                "entry_price": 0,
                "entry_atr": 0,
                "highest_since_entry": 0,
                "lowest_since_entry": np.inf,
            }

        state = self._position_state[symbol]

        for i, idx in enumerate(df.index):
            if i < lookback:
                continue

            current_close = close.loc[idx]
            current_atr = atr.loc[idx]
            current_high = high.loc[idx]
            current_low = low.loc[idx]
            vol_confirmed = vol_confirm.loc[idx]

            if pd.isna(current_atr) or current_atr < self.params["min_atr_threshold"]:
                signals.loc[idx] = state["position"]
                continue

            if state["position"] == 0:
                # Look for breakout entry
                # Upward breakout
                if current_close > upper_band.loc[idx] and vol_confirmed:
                    signals.loc[idx] = 1
                    state["position"] = 1
                    state["entry_price"] = current_close
                    state["entry_atr"] = current_atr
                    state["highest_since_entry"] = current_high

                # Downward breakout
                elif current_close < lower_band.loc[idx] and vol_confirmed:
                    signals.loc[idx] = -1
                    state["position"] = -1
                    state["entry_price"] = current_close
                    state["entry_atr"] = current_atr
                    state["lowest_since_entry"] = current_low

                else:
                    signals.loc[idx] = 0

            elif state["position"] == 1:
                # Long position management
                state["highest_since_entry"] = max(
                    state["highest_since_entry"], current_high
                )

                # Check stop loss
                stop_level = state["entry_price"] - (
                    self.params["stop_loss_atr"] * state["entry_atr"]
                )

                # Check trailing stop
                trailing_stop = state["highest_since_entry"] - (
                    self.params["trailing_stop_atr"] * state["entry_atr"]
                )

                # Check take profit
                take_profit = state["entry_price"] + (
                    self.params["take_profit_atr"] * state["entry_atr"]
                )

                if current_close < stop_level or current_close < trailing_stop:
                    # Exit on stop
                    signals.loc[idx] = 0
                    state["position"] = 0
                elif current_close > take_profit:
                    # Exit on take profit
                    signals.loc[idx] = 0
                    state["position"] = 0
                else:
                    signals.loc[idx] = 1

            elif state["position"] == -1:
                # Short position management
                state["lowest_since_entry"] = min(
                    state["lowest_since_entry"], current_low
                )

                # Check stop loss
                stop_level = state["entry_price"] + (
                    self.params["stop_loss_atr"] * state["entry_atr"]
                )

                # Check trailing stop
                trailing_stop = state["lowest_since_entry"] + (
                    self.params["trailing_stop_atr"] * state["entry_atr"]
                )

                # Check take profit
                take_profit = state["entry_price"] - (
                    self.params["take_profit_atr"] * state["entry_atr"]
                )

                if current_close > stop_level or current_close > trailing_stop:
                    # Exit on stop
                    signals.loc[idx] = 0
                    state["position"] = 0
                elif current_close < take_profit:
                    # Exit on take profit
                    signals.loc[idx] = 0
                    state["position"] = 0
                else:
                    signals.loc[idx] = -1

        return signals

    def generate_signals(
        self,
        data: pd.DataFrame | dict[str, pd.DataFrame],
        features: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """
        Generate volatility breakout signals.

        Args:
            data: OHLCV data
            features: Pre-computed features (not used)

        Returns:
            Signal DataFrame
        """
        if isinstance(data, dict):
            # Multi-stock data
            signals_dict = {}
            for symbol, df in data.items():
                signals_dict[symbol] = self._generate_single_stock_signals(df, symbol)

            signals = pd.DataFrame(signals_dict)
        else:
            # Single stock or MultiIndex
            if isinstance(data.index, pd.MultiIndex):
                signals_dict = {}
                for symbol in data.index.get_level_values(1).unique():
                    df = data.xs(symbol, level=1)
                    signals_dict[symbol] = self._generate_single_stock_signals(
                        df, symbol
                    )
                signals = pd.DataFrame(signals_dict)
            else:
                signals = self._generate_single_stock_signals(data, "SINGLE")
                signals = signals.to_frame("signal")

        logger.debug(f"Generated volatility breakout signals: shape={signals.shape}")
        return signals

    def calculate_positions(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        capital: float,
    ) -> pd.DataFrame:
        """
        Convert signals to position weights.

        Uses ATR-based position sizing.

        Args:
            signals: Signal DataFrame
            prices: Current prices
            capital: Available capital

        Returns:
            Position weights
        """
        positions = signals.copy().astype(float)

        for idx in positions.index:
            row = positions.loc[idx]
            n_positions = (row != 0).sum()

            if n_positions > 0:
                weight = 1.0 / max(n_positions, 5)  # Max 20% per position
                positions.loc[idx] = row * weight

        return positions

    def reset_state(self) -> None:
        """Reset position tracking state."""
        self._position_state = {}
