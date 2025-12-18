"""
Target Engineering Module - Triple-Barrier Labeling & Meta-Labeling.

JPMorgan Institutional-Level Target Generation for ML Trading Models.

Implements:
1. Triple-Barrier Labeling (de Prado, 2018)
2. Meta-Labeling for bet sizing
3. Continuous return targets
4. Event-based sampling

Reference:
    "Advances in Financial Machine Learning" by Marcos Lopez de Prado (2018)
    Chapters 3-4: Labeling
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numba import jit


logger = logging.getLogger(__name__)


class LabelType(str, Enum):
    """Types of labels for ML models."""
    BINARY = "binary"  # Up/Down classification
    TERNARY = "ternary"  # Up/Down/Neutral
    CONTINUOUS = "continuous"  # Actual returns
    META = "meta"  # Meta-label (correct/incorrect prediction)


@dataclass
class TripleBarrierConfig:
    """Configuration for triple-barrier labeling."""
    # Profit taking multiplier (in terms of daily volatility)
    profit_taking: float = 2.0
    # Stop loss multiplier (in terms of daily volatility)
    stop_loss: float = 2.0
    # Maximum holding period in bars
    max_holding_period: int = 10
    # Minimum return threshold for neutral zone
    min_return: float = 0.0
    # Volatility lookback for barrier calculation
    volatility_lookback: int = 20
    # Use dynamic barriers based on volatility
    dynamic_barriers: bool = True
    # Side labels: -1 (short), 0 (neutral), 1 (long)
    include_side: bool = True


@dataclass
class TripleBarrierResult:
    """Result of triple-barrier labeling."""
    labels: pd.Series  # Target labels
    returns: pd.Series  # Actual returns achieved
    barriers_hit: pd.Series  # Which barrier was hit (1=profit, -1=stop, 0=time)
    holding_periods: pd.Series  # Actual holding periods
    touch_times: pd.Series  # When barrier was touched


class TripleBarrierLabeler:
    """
    Triple-Barrier Method for labeling trading opportunities.

    The three barriers are:
    1. Profit-taking barrier (upper): Take profit at this return level
    2. Stop-loss barrier (lower): Cut loss at this return level
    3. Time barrier (vertical): Close position after max holding period

    The label is determined by which barrier is touched first:
    - +1: Profit barrier touched first (successful long trade)
    - -1: Stop loss barrier touched first (unsuccessful trade)
    - 0: Time barrier touched (indeterminate)

    This is far superior to fixed-horizon labeling because:
    1. Accounts for path dependency
    2. Incorporates risk management
    3. Creates more realistic trading signals
    """

    def __init__(self, config: Optional[TripleBarrierConfig] = None):
        """
        Initialize the labeler.

        Args:
            config: Triple-barrier configuration
        """
        self.config = config or TripleBarrierConfig()

    def fit(self, X: pd.DataFrame, y=None) -> "TripleBarrierLabeler":
        """Fit the labeler (stateless, just validates)."""
        return self

    def transform(
        self,
        df: pd.DataFrame,
        events: Optional[pd.DatetimeIndex] = None,
        side: Optional[pd.Series] = None,
    ) -> TripleBarrierResult:
        """
        Generate triple-barrier labels.

        Args:
            df: OHLCV DataFrame with DatetimeIndex
            events: Optional event timestamps to label (default: all rows)
            side: Optional side prediction (-1, 0, 1) from primary model

        Returns:
            TripleBarrierResult with labels and metadata
        """
        if events is None:
            events = df.index

        # Calculate daily volatility for barrier sizing
        close = df["close"]
        daily_vol = self._calculate_volatility(close)

        # Initialize result arrays
        n = len(events)
        labels = np.zeros(n)
        returns = np.zeros(n)
        barriers_hit = np.zeros(n)
        holding_periods = np.zeros(n, dtype=int)
        touch_times = [None] * n

        # Process each event
        for i, event_time in enumerate(events):
            try:
                event_idx = df.index.get_loc(event_time)
            except KeyError:
                continue

            # Get volatility at event time
            vol = daily_vol.iloc[event_idx] if self.config.dynamic_barriers else daily_vol.mean()

            # Calculate barriers
            entry_price = close.iloc[event_idx]
            profit_barrier = entry_price * (1 + self.config.profit_taking * vol)
            stop_barrier = entry_price * (1 - self.config.stop_loss * vol)

            # Get future prices up to max holding period
            end_idx = min(event_idx + self.config.max_holding_period + 1, len(df))
            future_prices = close.iloc[event_idx:end_idx]

            if len(future_prices) < 2:
                continue

            # Find first barrier touch
            label, ret, barrier, holding, touch_idx = self._find_first_barrier(
                future_prices.values,
                entry_price,
                profit_barrier,
                stop_barrier,
                side.iloc[event_idx] if side is not None else 1,
            )

            labels[i] = label
            returns[i] = ret
            barriers_hit[i] = barrier
            holding_periods[i] = holding
            if touch_idx < len(future_prices):
                touch_times[i] = future_prices.index[touch_idx]

        return TripleBarrierResult(
            labels=pd.Series(labels, index=events, name="label"),
            returns=pd.Series(returns, index=events, name="return"),
            barriers_hit=pd.Series(barriers_hit, index=events, name="barrier_hit"),
            holding_periods=pd.Series(holding_periods, index=events, name="holding_period"),
            touch_times=pd.Series(touch_times, index=events, name="touch_time"),
        )

    def _calculate_volatility(self, close: pd.Series) -> pd.Series:
        """Calculate rolling volatility."""
        returns = close.pct_change()
        vol = returns.rolling(self.config.volatility_lookback).std()
        # Annualize if needed, but keep it simple for barrier calculation
        return vol.fillna(method="bfill").fillna(0.02)  # Default 2% if no data

    def _find_first_barrier(
        self,
        prices: np.ndarray,
        entry_price: float,
        profit_barrier: float,
        stop_barrier: float,
        side: int = 1,
    ) -> Tuple[int, float, int, int, int]:
        """
        Find which barrier is touched first.

        Args:
            prices: Array of future prices
            entry_price: Entry price
            profit_barrier: Profit taking level
            stop_barrier: Stop loss level
            side: Trade side (1=long, -1=short)

        Returns:
            Tuple of (label, return, barrier_type, holding_period, touch_index)
        """
        # Adjust barriers for short trades
        if side == -1:
            profit_barrier, stop_barrier = 2 * entry_price - profit_barrier, 2 * entry_price - stop_barrier

        for i, price in enumerate(prices):
            # Check profit barrier
            if (side == 1 and price >= profit_barrier) or (side == -1 and price <= profit_barrier):
                ret = (price / entry_price - 1) * side
                return 1, ret, 1, i, i

            # Check stop barrier
            if (side == 1 and price <= stop_barrier) or (side == -1 and price >= stop_barrier):
                ret = (price / entry_price - 1) * side
                return -1, ret, -1, i, i

        # Time barrier (max holding period reached)
        final_price = prices[-1]
        ret = (final_price / entry_price - 1) * side

        if ret > self.config.min_return:
            label = 1
        elif ret < -self.config.min_return:
            label = -1
        else:
            label = 0

        return label, ret, 0, len(prices) - 1, len(prices) - 1


class MetaLabeler:
    """
    Meta-Labeling for ML-based position sizing.

    Meta-labeling is a two-stage approach:
    1. Primary model predicts direction (side)
    2. Meta-model predicts whether primary model will be correct

    This allows for sophisticated bet sizing based on
    confidence in the primary model's predictions.

    Benefits:
    - Reduces false positives from primary model
    - Provides probability for position sizing
    - Decouples direction from sizing decision
    """

    def __init__(
        self,
        triple_barrier_config: Optional[TripleBarrierConfig] = None,
        min_confidence: float = 0.5,
    ):
        """
        Initialize meta-labeler.

        Args:
            triple_barrier_config: Config for triple-barrier labeling
            min_confidence: Minimum confidence threshold for trading
        """
        self.triple_barrier_config = triple_barrier_config or TripleBarrierConfig()
        self.min_confidence = min_confidence
        self.labeler = TripleBarrierLabeler(self.triple_barrier_config)

    def generate_meta_labels(
        self,
        df: pd.DataFrame,
        primary_predictions: pd.Series,
        events: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        """
        Generate meta-labels from primary model predictions.

        Args:
            df: OHLCV DataFrame
            primary_predictions: Side predictions from primary model (-1, 0, 1)
            events: Optional event timestamps

        Returns:
            DataFrame with:
            - meta_label: 1 if primary model was correct, 0 otherwise
            - primary_side: Original side prediction
            - actual_return: Realized return
            - primary_correct_pct: Running accuracy of primary model
        """
        if events is None:
            events = df.index

        # Get triple-barrier results using primary predictions as side
        tb_result = self.labeler.transform(df, events=events, side=primary_predictions)

        # Meta-label: Did the primary model get the direction right?
        # Primary is correct if: (side * actual_return) > 0
        primary_correct = (
            primary_predictions.reindex(events) * tb_result.returns > 0
        ).astype(int)

        # Build result DataFrame
        meta_df = pd.DataFrame(index=events)
        meta_df["meta_label"] = primary_correct
        meta_df["primary_side"] = primary_predictions.reindex(events)
        meta_df["actual_return"] = tb_result.returns
        meta_df["barrier_hit"] = tb_result.barriers_hit
        meta_df["holding_period"] = tb_result.holding_periods

        # Rolling accuracy of primary model
        meta_df["primary_accuracy_20"] = primary_correct.rolling(20).mean()

        logger.info(
            f"Meta-labeling complete: "
            f"{primary_correct.sum()}/{len(primary_correct)} primary predictions correct "
            f"({primary_correct.mean()*100:.1f}%)"
        )

        return meta_df

    def calculate_bet_size(
        self,
        meta_probabilities: pd.Series,
        primary_side: pd.Series,
        max_leverage: float = 1.0,
        scaling: str = "linear",
    ) -> pd.Series:
        """
        Calculate position sizes based on meta-model probabilities.

        Args:
            meta_probabilities: Probability of primary model being correct
            primary_side: Side predictions from primary model
            max_leverage: Maximum position size
            scaling: "linear" or "kelly" scaling method

        Returns:
            Position sizes (-max_leverage to +max_leverage)
        """
        if scaling == "linear":
            # Linear: size = (prob - 0.5) * 2 * max_leverage * side
            confidence = (meta_probabilities - 0.5) * 2
            confidence = confidence.clip(0, 1)
            sizes = confidence * max_leverage * primary_side

        elif scaling == "kelly":
            # Kelly criterion: f = (p * b - q) / b
            # Where p = prob of win, q = 1-p, b = win/loss ratio (assume 1)
            p = meta_probabilities
            q = 1 - p
            kelly_fraction = (p - q).clip(0, 1)
            sizes = kelly_fraction * max_leverage * primary_side

        else:
            raise ValueError(f"Unknown scaling method: {scaling}")

        # Apply minimum confidence threshold
        sizes = sizes.where(meta_probabilities >= self.min_confidence, 0)

        return sizes


def create_triple_barrier_labels(
    df: pd.DataFrame,
    profit_taking: float = 2.0,
    stop_loss: float = 2.0,
    max_holding_period: int = 10,
    volatility_lookback: int = 20,
) -> pd.DataFrame:
    """
    Convenience function to create triple-barrier labels.

    Args:
        df: OHLCV DataFrame
        profit_taking: Profit barrier (volatility multiplier)
        stop_loss: Stop loss barrier (volatility multiplier)
        max_holding_period: Maximum bars to hold
        volatility_lookback: Lookback for volatility calculation

    Returns:
        DataFrame with labels and metadata
    """
    config = TripleBarrierConfig(
        profit_taking=profit_taking,
        stop_loss=stop_loss,
        max_holding_period=max_holding_period,
        volatility_lookback=volatility_lookback,
    )

    labeler = TripleBarrierLabeler(config)
    result = labeler.transform(df)

    return pd.DataFrame({
        "label": result.labels,
        "return": result.returns,
        "barrier_hit": result.barriers_hit,
        "holding_period": result.holding_periods,
    })


def create_binary_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.0,
) -> pd.Series:
    """
    Create simple binary labels (up/down) based on future returns.

    Args:
        df: OHLCV DataFrame
        horizon: Prediction horizon in bars
        threshold: Minimum return for positive label

    Returns:
        Binary labels (0=down, 1=up)
    """
    future_return = df["close"].pct_change(horizon).shift(-horizon)
    labels = (future_return > threshold).astype(int)
    return labels


def create_continuous_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    log_returns: bool = True,
) -> pd.Series:
    """
    Create continuous return labels for regression.

    Args:
        df: OHLCV DataFrame
        horizon: Prediction horizon in bars
        log_returns: Use log returns (more stable for ML)

    Returns:
        Continuous return labels
    """
    if log_returns:
        returns = np.log(df["close"] / df["close"].shift(horizon)).shift(-horizon)
    else:
        returns = df["close"].pct_change(horizon).shift(-horizon)

    return returns
