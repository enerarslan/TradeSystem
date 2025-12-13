"""
Meta-Labeling Pipeline
======================

Implements the Meta-Labeling approach from AFML Chapter 3.6.

The key insight: Instead of predicting price direction directly (which is noisy),
we use a two-stage approach:

1. PRIMARY MODEL (Signal Generator):
   - Simple rule-based or weak ML model
   - Generates trade SIDE: Long (+1), Short (-1), Flat (0)
   - Examples: Trend-following, Bollinger breakout, Moving average crossover

2. SECONDARY MODEL (Meta-Labeler / Filter):
   - ML model that predicts: "Should I take this trade?"
   - Binary target: 1 = primary model's trade was profitable, 0 = was not
   - Effectively learns BET SIZING / FILTERING

Benefits of Meta-Labeling:
1. Separates direction prediction from bet sizing
2. Reduces false positives from primary model
3. Secondary model answers easier question: "Is this a good setup?"
4. Can use sophisticated ML for the filter while keeping simple signal
5. More interpretable: you know why you're taking a position

Mathematical Framework:
- Let S_t be the primary signal at time t: S_t ∈ {-1, 0, +1}
- Let Y_t be the triple-barrier label: Y_t ∈ {-1, 0, +1}
- Meta-label M_t = 1 if sign(S_t * Y_t) > 0 else 0
- The secondary model learns P(M_t = 1 | X_t, S_t)
- Final position = S_t * P(M_t = 1 | X_t, S_t)

Author: AlphaTrade Institutional System
Based on: Marcos Lopez de Prado - Advances in Financial Machine Learning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from ..data.labeling import (
    TripleBarrierLabeler, TripleBarrierConfig,
    MetaLabeler, get_sample_weights, get_time_decay_weights, combine_weights
)
from ..utils.logger import get_logger
from ..utils.helpers import safe_divide

logger = get_logger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MetaLabelingConfig:
    """Configuration for meta-labeling pipeline."""

    # Triple Barrier settings
    pt_sl_ratio: Tuple[float, float] = (1.5, 1.0)  # Asymmetric: PT > SL
    max_holding_period: int = 20  # Maximum bars
    volatility_lookback: int = 20  # For dynamic barriers
    min_return: float = 0.001  # Minimum return threshold

    # Sample weighting
    use_sample_weights: bool = True
    time_decay_factor: float = 0.5  # 0 = no decay, 1 = linear decay
    uniqueness_weight: float = 0.7  # Weight for uniqueness vs time decay

    # Primary model
    primary_confidence_threshold: float = 0.5  # Min confidence to generate signal


class SignalType(Enum):
    """Types of primary signals."""
    LONG = 1
    FLAT = 0
    SHORT = -1


# =============================================================================
# PRIMARY SIGNAL GENERATORS
# =============================================================================

class PrimarySignal(ABC):
    """
    Abstract base class for primary signal generators.

    The primary model should be SIMPLE and INTERPRETABLE.
    Its job is to identify DIRECTION, not bet size.
    """

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signal.

        Args:
            df: OHLCV DataFrame

        Returns:
            Series with values in {-1, 0, +1}
        """
        pass

    @abstractmethod
    def get_confidence(self, df: pd.DataFrame) -> pd.Series:
        """
        Get signal confidence score.

        Args:
            df: OHLCV DataFrame

        Returns:
            Series with values in [0, 1]
        """
        pass


class TrendFollowingSignal(PrimarySignal):
    """
    Trend-following primary signal.

    Uses dual moving average crossover:
    - Long when fast MA > slow MA AND price > both MAs
    - Short when fast MA < slow MA AND price < both MAs
    - Flat otherwise
    """

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        atr_filter: bool = True,
        atr_period: int = 14
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_filter = atr_filter
        self.atr_period = atr_period

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        high = df['high']
        low = df['low']

        # Moving averages
        fast_ma = close.rolling(self.fast_period).mean()
        slow_ma = close.rolling(self.slow_period).mean()

        # Base signals
        signal = pd.Series(0, index=df.index)

        # Long: Fast > Slow AND Price > Fast
        long_condition = (fast_ma > slow_ma) & (close > fast_ma)
        signal[long_condition] = 1

        # Short: Fast < Slow AND Price < Fast
        short_condition = (fast_ma < slow_ma) & (close < fast_ma)
        signal[short_condition] = -1

        # Optional: ATR filter to avoid low volatility periods
        if self.atr_filter:
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)

            atr = tr.rolling(self.atr_period).mean()
            atr_pct = atr / close

            # Only trade when ATR is above median
            atr_median = atr_pct.rolling(100).median()
            low_vol_mask = atr_pct < atr_median
            signal[low_vol_mask] = 0

        return signal

    def get_confidence(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']

        fast_ma = close.rolling(self.fast_period).mean()
        slow_ma = close.rolling(self.slow_period).mean()

        # Confidence based on MA separation
        ma_diff = (fast_ma - slow_ma) / slow_ma
        confidence = ma_diff.abs().clip(0, 0.05) / 0.05  # Normalize to [0, 1]

        return confidence


class BollingerBreakoutSignal(PrimarySignal):
    """
    Bollinger Band breakout signal.

    - Long when price closes above upper band (breakout)
    - Short when price closes below lower band (breakdown)
    - Flat when price is within bands

    This is a MEAN REVERSION setup when combined with meta-labeling
    (filter will learn when breakouts fail).
    """

    def __init__(
        self,
        period: int = 20,
        num_std: float = 2.0,
        confirm_bars: int = 1
    ):
        self.period = period
        self.num_std = num_std
        self.confirm_bars = confirm_bars

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']

        # Bollinger Bands
        sma = close.rolling(self.period).mean()
        std = close.rolling(self.period).std()
        upper = sma + self.num_std * std
        lower = sma - self.num_std * std

        signal = pd.Series(0, index=df.index)

        # Long: Price breaks above upper band
        long_condition = close > upper
        if self.confirm_bars > 1:
            long_condition = long_condition.rolling(self.confirm_bars).sum() >= self.confirm_bars

        # Short: Price breaks below lower band
        short_condition = close < lower
        if self.confirm_bars > 1:
            short_condition = short_condition.rolling(self.confirm_bars).sum() >= self.confirm_bars

        signal[long_condition] = 1
        signal[short_condition] = -1

        return signal

    def get_confidence(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']

        sma = close.rolling(self.period).mean()
        std = close.rolling(self.period).std()

        # Confidence = how far outside the band
        z_score = (close - sma) / std
        confidence = (z_score.abs() - self.num_std).clip(0, 2) / 2

        return confidence


class MomentumSignal(PrimarySignal):
    """
    Momentum-based primary signal.

    Uses RSI with confirmation from price momentum.
    - Long: RSI oversold + positive momentum
    - Short: RSI overbought + negative momentum
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        momentum_period: int = 10
    ):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.momentum_period = momentum_period

    def _compute_rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']

        rsi = self._compute_rsi(close)
        momentum = close.pct_change(self.momentum_period)

        signal = pd.Series(0, index=df.index)

        # Long: RSI oversold AND momentum turning positive
        long_condition = (rsi < self.rsi_oversold) & (momentum > 0)
        signal[long_condition] = 1

        # Short: RSI overbought AND momentum turning negative
        short_condition = (rsi > self.rsi_overbought) & (momentum < 0)
        signal[short_condition] = -1

        return signal

    def get_confidence(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']

        rsi = self._compute_rsi(close)

        # Confidence based on RSI extremity
        confidence = pd.Series(0.5, index=df.index)

        # More extreme RSI = higher confidence
        confidence[rsi < 30] = (30 - rsi[rsi < 30]) / 30
        confidence[rsi > 70] = (rsi[rsi > 70] - 70) / 30

        return confidence.clip(0, 1)


class CompositeSignal(PrimarySignal):
    """
    Combine multiple primary signals.

    Aggregates signals using voting or averaging.
    """

    def __init__(
        self,
        signals: List[PrimarySignal],
        method: str = 'vote',  # 'vote' or 'average'
        threshold: float = 0.5  # For voting: fraction required
    ):
        self.signals = signals
        self.method = method
        self.threshold = threshold

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        all_signals = pd.DataFrame({
            f'signal_{i}': sig.generate_signal(df)
            for i, sig in enumerate(self.signals)
        })

        if self.method == 'vote':
            # Majority vote
            avg = all_signals.mean(axis=1)
            result = pd.Series(0, index=df.index)
            result[avg > self.threshold] = 1
            result[avg < -self.threshold] = -1

        else:  # average
            result = all_signals.mean(axis=1)
            result = result.apply(lambda x: 1 if x > 0.5 else (-1 if x < -0.5 else 0))

        return result

    def get_confidence(self, df: pd.DataFrame) -> pd.Series:
        all_confidence = pd.DataFrame({
            f'conf_{i}': sig.get_confidence(df)
            for i, sig in enumerate(self.signals)
        })

        return all_confidence.mean(axis=1)


# =============================================================================
# META-LABELING PIPELINE
# =============================================================================

class MetaLabelingPipeline:
    """
    Complete meta-labeling pipeline for institutional trading.

    Steps:
    1. Generate primary signals using rule-based model
    2. Apply triple-barrier method to get true labels
    3. Create meta-labels: was the primary signal correct?
    4. Compute sample weights based on label uniqueness
    5. Prepare training data for secondary (meta) model

    The secondary model learns to FILTER the primary model's signals.
    Output: probability that the primary model's trade will be profitable.
    """

    def __init__(
        self,
        config: MetaLabelingConfig = None,
        primary_signal: PrimarySignal = None
    ):
        self.config = config or MetaLabelingConfig()

        # Use trend-following as default primary
        self.primary_signal = primary_signal or TrendFollowingSignal()

        # Initialize triple barrier labeler
        self._barrier_config = TripleBarrierConfig(
            pt_sl_ratio=self.config.pt_sl_ratio,
            max_holding_period=self.config.max_holding_period,
            volatility_lookback=self.config.volatility_lookback,
            min_return=self.config.min_return
        )
        self._labeler = TripleBarrierLabeler(self._barrier_config)

        # Cache for computed labels
        self._events: Optional[pd.DataFrame] = None
        self._meta_labels: Optional[pd.Series] = None
        self._sample_weights: Optional[pd.Series] = None

    def generate_labels(
        self,
        df: pd.DataFrame,
        signal: pd.Series = None
    ) -> pd.DataFrame:
        """
        Generate meta-labels for the dataset.

        Args:
            df: OHLCV DataFrame
            signal: Optional pre-computed signal (else generate from primary)

        Returns:
            DataFrame with columns:
            - primary_signal: +1, 0, -1
            - triple_barrier_label: actual outcome
            - meta_label: 1 if signal was correct, 0 otherwise
            - t1: barrier touch time
            - ret: return at touch
            - sample_weight: uniqueness-based weight
        """
        logger.info("Generating meta-labels...")

        close = df['close']

        # 1. Generate primary signals
        if signal is None:
            signal = self.primary_signal.generate_signal(df)

        confidence = self.primary_signal.get_confidence(df)

        # Filter by confidence
        signal[confidence < self.config.primary_confidence_threshold] = 0

        # Get event timestamps (where we have a signal)
        t_events = signal[signal != 0].index

        if len(t_events) == 0:
            logger.warning("No trading signals generated")
            return pd.DataFrame()

        logger.info(f"Generated {len(t_events)} primary signals")

        # 2. Apply triple-barrier method
        events = self._labeler.get_events(
            close=close,
            t_events=t_events,
            pt_sl=self.config.pt_sl_ratio,
            side=signal.loc[t_events]  # Use signal as side
        )

        if len(events) == 0:
            logger.warning("No events generated from triple barrier")
            return pd.DataFrame()

        # 3. Create meta-labels
        # Meta-label = 1 if the primary signal was CORRECT
        # Correct = (signal * barrier_label) > 0
        primary_side = signal.loc[events.index]
        barrier_label = events['label']

        # Handle the case where barrier_label might be NaN
        barrier_label = barrier_label.fillna(0)

        # Meta-label: was the signal correct?
        meta_label = ((primary_side * barrier_label) > 0).astype(int)

        # 4. Compute sample weights
        if self.config.use_sample_weights:
            # Uniqueness weights
            uniqueness_weights = get_sample_weights(events, close)

            # Time decay weights
            time_decay_weights = get_time_decay_weights(
                events, c=self.config.time_decay_factor
            )

            # Combine
            sample_weights = combine_weights(
                uniqueness_weights,
                time_decay_weights,
                alpha=self.config.uniqueness_weight
            )
        else:
            sample_weights = pd.Series(1.0, index=events.index)

        # 5. Build result DataFrame
        result = pd.DataFrame(index=events.index)
        result['primary_signal'] = primary_side
        result['signal_confidence'] = confidence.loc[events.index]
        result['triple_barrier_label'] = barrier_label
        result['meta_label'] = meta_label
        result['t1'] = events['t1']
        result['ret'] = events['ret']
        result['sample_weight'] = sample_weights

        # Binary label for classification
        result['bin_label'] = events['bin_label']

        # Cache
        self._events = events
        self._meta_labels = meta_label
        self._sample_weights = sample_weights

        # Log statistics
        n_long = (result['primary_signal'] == 1).sum()
        n_short = (result['primary_signal'] == -1).sum()
        accuracy = meta_label.mean()

        logger.info(
            f"Meta-labeling complete: "
            f"{len(result)} events, "
            f"{n_long} long, {n_short} short, "
            f"accuracy={accuracy:.2%}"
        )

        return result

    def prepare_training_data(
        self,
        features: pd.DataFrame,
        labels_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare aligned training data for meta-model.

        Args:
            features: Feature DataFrame
            labels_df: Output from generate_labels()

        Returns:
            Tuple of (X, y, sample_weights)
        """
        # Align indices
        common_idx = features.index.intersection(labels_df.index)

        if len(common_idx) == 0:
            raise ValueError("No common indices between features and labels")

        X = features.loc[common_idx].copy()
        y = labels_df.loc[common_idx, 'meta_label'].copy()
        weights = labels_df.loc[common_idx, 'sample_weight'].copy()

        # Add primary signal and confidence as features
        X['primary_signal'] = labels_df.loc[common_idx, 'primary_signal']
        X['signal_confidence'] = labels_df.loc[common_idx, 'signal_confidence']

        # Drop NaN rows
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        weights = weights[valid_mask]

        # Normalize weights
        weights = weights / weights.sum() * len(weights)

        logger.info(
            f"Prepared {len(X)} training samples "
            f"(positive rate: {y.mean():.2%})"
        )

        return X, y, weights

    def get_events(self) -> pd.DataFrame:
        """Get cached triple barrier events."""
        return self._events

    def get_sample_weights(self) -> pd.Series:
        """Get cached sample weights."""
        return self._sample_weights


# =============================================================================
# BET SIZING FROM META-PROBABILITIES
# =============================================================================

class BetSizer:
    """
    Convert meta-model probabilities to bet sizes.

    The meta-model outputs P(profitable | features, signal).
    We need to convert this to a position size.

    Options:
    1. Binary: size = 1 if P > threshold, else 0
    2. Linear: size = P (or P - 0.5 scaled)
    3. Discretized: buckets like [0, 0.25, 0.5, 0.75, 1.0]
    4. Kelly: size = edge / odds (requires calibrated probabilities)
    """

    def __init__(
        self,
        method: str = 'linear',  # 'binary', 'linear', 'discretized', 'kelly'
        threshold: float = 0.5,
        max_size: float = 1.0,
        discretize_bins: int = 5
    ):
        self.method = method
        self.threshold = threshold
        self.max_size = max_size
        self.discretize_bins = discretize_bins

    def size_bet(
        self,
        probability: Union[float, pd.Series],
        signal: Union[int, pd.Series]
    ) -> Union[float, pd.Series]:
        """
        Compute bet size from probability and signal.

        Args:
            probability: P(profitable) from meta-model
            signal: Primary signal (+1, -1)

        Returns:
            Position size in [-max_size, +max_size]
        """
        if self.method == 'binary':
            size = (probability > self.threshold).astype(float)

        elif self.method == 'linear':
            # Scale probability to [0, max_size]
            # P=0.5 -> size=0, P=1.0 -> size=max
            size = (probability - 0.5) * 2 * self.max_size
            size = size.clip(0, self.max_size)

        elif self.method == 'discretized':
            # Discretize into buckets
            bins = np.linspace(0, 1, self.discretize_bins + 1)
            labels = np.linspace(0, self.max_size, self.discretize_bins)

            if isinstance(probability, pd.Series):
                size = pd.cut(probability, bins=bins, labels=labels, include_lowest=True)
                size = size.astype(float)
            else:
                idx = np.digitize(probability, bins) - 1
                idx = min(idx, len(labels) - 1)
                size = labels[idx]

        elif self.method == 'kelly':
            # Kelly criterion: f* = (p*b - q) / b
            # Where b = odds (assume 1:1), p = probability, q = 1-p
            # Simplified: f* = 2*p - 1 (for 1:1 odds)
            size = (2 * probability - 1) * self.max_size
            size = size.clip(0, self.max_size)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Apply signal direction
        position = size * signal

        return position


# =============================================================================
# COMPLETE META-LABELING WORKFLOW
# =============================================================================

def create_meta_labeling_dataset(
    df: pd.DataFrame,
    features: pd.DataFrame,
    primary_signal: PrimarySignal = None,
    config: MetaLabelingConfig = None
) -> Dict[str, Any]:
    """
    Create complete meta-labeling dataset for training.

    This is the main entry point for preparing data for meta-model training.

    Args:
        df: OHLCV DataFrame
        features: Pre-computed feature DataFrame
        primary_signal: Primary signal generator
        config: Meta-labeling configuration

    Returns:
        Dictionary with:
        - X: Feature matrix
        - y: Meta-labels (0/1)
        - sample_weights: Uniqueness-based weights
        - events: Triple barrier events
        - stats: Statistics dictionary
    """
    config = config or MetaLabelingConfig()
    primary_signal = primary_signal or TrendFollowingSignal()

    pipeline = MetaLabelingPipeline(config, primary_signal)

    # Generate labels
    labels_df = pipeline.generate_labels(df)

    if len(labels_df) == 0:
        raise ValueError("No meta-labels generated")

    # Prepare training data
    X, y, weights = pipeline.prepare_training_data(features, labels_df)

    # Compute statistics
    stats = {
        'n_samples': len(X),
        'n_features': len(X.columns),
        'positive_rate': y.mean(),
        'n_long_signals': (labels_df['primary_signal'] == 1).sum(),
        'n_short_signals': (labels_df['primary_signal'] == -1).sum(),
        'primary_accuracy': labels_df['meta_label'].mean(),
        'avg_return': labels_df['ret'].mean(),
        'avg_weight': weights.mean()
    }

    return {
        'X': X,
        'y': y,
        'sample_weights': weights,
        'events': labels_df,
        'stats': stats
    }
