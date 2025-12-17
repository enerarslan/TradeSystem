"""
Base strategy class for AlphaTrade system.

This module provides the abstract base class for all trading strategies
with common functionality and interface definitions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class SignalType(Enum):
    """Trading signal types."""

    LONG = 1
    SHORT = -1
    NEUTRAL = 0


@dataclass
class Signal:
    """
    Trading signal with metadata.

    Attributes:
        symbol: Trading symbol
        signal_type: Signal direction
        strength: Signal strength/confidence (0-1)
        timestamp: Signal generation time
        price: Price at signal generation
        metadata: Additional signal information
    """

    symbol: str
    signal_type: SignalType
    strength: float
    timestamp: datetime | pd.Timestamp
    price: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def value(self) -> int:
        """Get numeric signal value."""
        return self.signal_type.value

    def __repr__(self) -> str:
        return (
            f"Signal({self.symbol}, {self.signal_type.name}, "
            f"strength={self.strength:.2f}, price={self.price:.2f})"
        )


@dataclass
class Position:
    """
    Trading position information.

    Attributes:
        symbol: Trading symbol
        quantity: Position quantity (negative for short)
        entry_price: Average entry price
        entry_time: Position entry time
        current_price: Current market price
        unrealized_pnl: Unrealized profit/loss
    """

    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime | pd.Timestamp
    current_price: float = 0.0
    unrealized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        """Calculate current market value."""
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        """Calculate cost basis."""
        return self.quantity * self.entry_price

    def update_price(self, price: float) -> None:
        """Update current price and P&L."""
        self.current_price = price
        self.unrealized_pnl = (price - self.entry_price) * self.quantity


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    All strategies must implement:
    - generate_signals(): Generate trading signals from data
    - calculate_positions(): Convert signals to target positions

    Provides common functionality:
    - Parameter management
    - Signal validation
    - Position sizing integration
    - Logging
    """

    def __init__(
        self,
        name: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the strategy.

        Args:
            name: Strategy name
            params: Strategy parameters
        """
        self.name = name
        self.params = params or {}
        self._signals: dict[str, Signal] = {}
        self._positions: dict[str, Position] = {}

        logger.info(f"Initialized strategy: {self.name}")

    @abstractmethod
    def generate_signals(
        self,
        data: pd.DataFrame | dict[str, pd.DataFrame],
        features: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """
        Generate trading signals from data.

        Args:
            data: OHLCV data (single stock or dict of stocks)
            features: Pre-computed features (optional)

        Returns:
            DataFrame with signals column(s)
            - Single stock: Series/DataFrame with 'signal' column
            - Multi-stock: DataFrame with symbols as columns
        """
        pass

    @abstractmethod
    def calculate_positions(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        capital: float,
    ) -> pd.DataFrame:
        """
        Convert signals to target positions.

        Args:
            signals: Signal DataFrame from generate_signals
            prices: Current prices
            capital: Available capital

        Returns:
            DataFrame with target position weights or quantities
        """
        pass

    def apply_risk_rules(
        self,
        positions: pd.DataFrame,
        risk_limits: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """
        Apply risk management rules to positions.

        Args:
            positions: Target positions
            risk_limits: Risk limit parameters

        Returns:
            Risk-adjusted positions
        """
        if risk_limits is None:
            risk_limits = {
                "max_position": 0.05,
                "max_sector": 0.25,
                "max_leverage": 1.0,
            }

        adjusted = positions.copy()

        # Apply maximum position limit
        max_pos = risk_limits.get("max_position", 0.05)
        adjusted = adjusted.clip(lower=-max_pos, upper=max_pos)

        # Apply leverage limit
        max_leverage = risk_limits.get("max_leverage", 1.0)
        total_exposure = adjusted.abs().sum(axis=1)
        scale = (max_leverage / total_exposure).clip(upper=1.0)
        adjusted = adjusted.multiply(scale, axis=0)

        return adjusted

    def validate_signals(self, signals: pd.DataFrame) -> bool:
        """
        Validate generated signals.

        Args:
            signals: Signal DataFrame

        Returns:
            True if signals are valid
        """
        # Check for NaN
        if signals.isnull().any().any():
            logger.warning("Signals contain NaN values")
            return False

        # Check signal range
        if (signals.abs() > 1).any().any():
            logger.warning("Signals outside [-1, 1] range")
            return False

        return True

    def get_params(self) -> dict[str, Any]:
        """Get strategy parameters."""
        return self.params.copy()

    def set_params(self, **params: Any) -> None:
        """Update strategy parameters."""
        self.params.update(params)
        logger.debug(f"Updated params for {self.name}: {params}")

    def get_signal_stats(self, signals: pd.DataFrame) -> dict[str, Any]:
        """
        Calculate signal statistics.

        Args:
            signals: Signal DataFrame

        Returns:
            Dictionary with signal statistics
        """
        flat = signals.values.flatten()
        flat = flat[~np.isnan(flat)]

        return {
            "total_signals": len(flat),
            "long_signals": (flat > 0).sum(),
            "short_signals": (flat < 0).sum(),
            "neutral_signals": (flat == 0).sum(),
            "mean_signal": flat.mean(),
            "signal_std": flat.std(),
            "turnover": np.abs(np.diff(flat)).mean() if len(flat) > 1 else 0,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class SignalCombiner:
    """
    Utility class for combining signals from multiple strategies.
    """

    @staticmethod
    def weighted_average(
        signals: list[pd.DataFrame],
        weights: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Combine signals using weighted average.

        Args:
            signals: List of signal DataFrames
            weights: Weights for each strategy (equal if None)

        Returns:
            Combined signal DataFrame
        """
        if weights is None:
            weights = [1.0 / len(signals)] * len(signals)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        combined = signals[0] * weights[0]
        for sig, weight in zip(signals[1:], weights[1:]):
            combined = combined.add(sig * weight, fill_value=0)

        return combined

    @staticmethod
    def voting(
        signals: list[pd.DataFrame],
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        Combine signals using majority voting.

        Args:
            signals: List of signal DataFrames
            threshold: Minimum agreement ratio for signal

        Returns:
            Combined signal DataFrame
        """
        # Convert to signs (-1, 0, 1)
        signs = [np.sign(sig) for sig in signals]

        # Sum signs
        vote_sum = signs[0].copy()
        for sign in signs[1:]:
            vote_sum = vote_sum.add(sign, fill_value=0)

        # Normalize by number of strategies
        n_strategies = len(signals)
        agreement = vote_sum / n_strategies

        # Apply threshold
        combined = pd.DataFrame(0, index=signals[0].index, columns=signals[0].columns)
        combined[agreement > threshold] = 1
        combined[agreement < -threshold] = -1

        return combined

    @staticmethod
    def rank_average(
        signals: list[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Combine signals using rank averaging.

        Args:
            signals: List of signal DataFrames

        Returns:
            Combined signal DataFrame (normalized ranks)
        """
        # Rank each signal cross-sectionally
        ranks = [sig.rank(axis=1, pct=True) - 0.5 for sig in signals]

        # Average ranks
        combined = ranks[0].copy()
        for rank in ranks[1:]:
            combined = combined.add(rank, fill_value=0)
        combined /= len(signals)

        return combined


class SignalFilter:
    """
    Utility class for filtering and smoothing signals.
    """

    @staticmethod
    def smooth(
        signals: pd.DataFrame,
        window: int = 3,
        method: str = "ewm",
    ) -> pd.DataFrame:
        """
        Smooth signals over time.

        Args:
            signals: Signal DataFrame
            window: Smoothing window
            method: 'sma' or 'ewm'

        Returns:
            Smoothed signals
        """
        if method == "sma":
            return signals.rolling(window=window).mean()
        elif method == "ewm":
            return signals.ewm(span=window, adjust=False).mean()
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

    @staticmethod
    def threshold(
        signals: pd.DataFrame,
        min_signal: float = 0.1,
    ) -> pd.DataFrame:
        """
        Apply threshold to filter weak signals.

        Args:
            signals: Signal DataFrame
            min_signal: Minimum absolute signal strength

        Returns:
            Filtered signals
        """
        filtered = signals.copy()
        filtered[signals.abs() < min_signal] = 0
        return filtered

    @staticmethod
    def holding_period_filter(
        signals: pd.DataFrame,
        min_holding: int = 4,
    ) -> pd.DataFrame:
        """
        Prevent signal changes within minimum holding period.

        Args:
            signals: Signal DataFrame
            min_holding: Minimum bars between signal changes

        Returns:
            Filtered signals
        """
        filtered = signals.copy()

        for col in filtered.columns:
            last_change_idx = 0
            current_signal = 0

            for i in range(len(filtered)):
                if signals[col].iloc[i] != current_signal:
                    if i - last_change_idx >= min_holding:
                        current_signal = signals[col].iloc[i]
                        last_change_idx = i
                    else:
                        filtered[col].iloc[i] = current_signal
                else:
                    filtered[col].iloc[i] = current_signal

        return filtered
