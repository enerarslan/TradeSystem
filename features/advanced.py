"""
Advanced Features Module
========================

JPMorgan-level feature engineering for the algorithmic trading platform.
Implements institutional-grade target labeling and feature generation.

Key Innovations:
1. Triple Barrier Method (de Prado) - Dynamic labeling based on price barriers
2. Meta-Labeling - Two-stage classification for bet sizing
3. Fractional Differentiation - Memory preservation with stationarity
4. Microstructure Features - Order flow imbalance from OHLC
5. Cross-Sectional Features - Relative strength indicators

These techniques are from "Advances in Financial Machine Learning" by 
Marcos López de Prado and are used by top quant funds.

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from config.settings import get_logger

logger = get_logger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class BarrierType(str, Enum):
    """Which barrier was touched first."""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TIME_BARRIER = "time_barrier"
    NONE = "none"


class LabelType(str, Enum):
    """Label type for targets."""
    BINARY = "binary"  # 0/1
    TERNARY = "ternary"  # -1/0/1
    CONTINUOUS = "continuous"  # Return value
    META = "meta"  # Meta-label (side * size)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TripleBarrierConfig:
    """Configuration for Triple Barrier Method."""
    # Barrier parameters
    take_profit_multiplier: float = 2.0  # ATR multiplier for take profit
    stop_loss_multiplier: float = 1.0  # ATR multiplier for stop loss
    max_holding_period: int = 20  # Maximum bars to hold
    
    # ATR settings
    atr_period: int = 20
    
    # Volatility scaling
    use_volatility_scaling: bool = True
    volatility_lookback: int = 50
    
    # Minimum profit threshold
    min_profit_threshold: float = 0.0  # Minimum profit to be considered "win"
    
    # Symmetric barriers
    symmetric: bool = False  # If True, take_profit = stop_loss


@dataclass
class MetaLabelConfig:
    """Configuration for Meta-Labeling."""
    # Primary model threshold
    primary_threshold: float = 0.5
    
    # Bet sizing
    enable_bet_sizing: bool = True
    max_bet_size: float = 1.0
    
    # Side prediction
    side_from_primary: bool = True  # Use primary model for direction


@dataclass
class AdvancedFeatureConfig:
    """Configuration for advanced feature generation."""
    # Fractional differentiation
    enable_frac_diff: bool = True
    frac_diff_d: float = 0.4  # Differentiation order (0 < d < 1)
    frac_diff_threshold: float = 1e-5
    
    # Microstructure
    enable_microstructure: bool = True
    
    # Cross-sectional
    enable_cross_sectional: bool = False  # Requires multi-asset data
    
    # Feature interaction
    enable_interactions: bool = True
    interaction_pairs: list[tuple[str, str]] = field(default_factory=list)
    
    # Calendar features
    enable_calendar: bool = True


# =============================================================================
# TRIPLE BARRIER METHOD
# =============================================================================

class TripleBarrierLabeler:
    """
    Triple Barrier Method for dynamic target labeling.
    
    Creates labels based on which barrier (take profit, stop loss, or time)
    is touched first. This method captures the actual trading outcome better
    than simple forward returns.
    
    From: "Advances in Financial Machine Learning" by López de Prado
    
    Example:
        config = TripleBarrierConfig(
            take_profit_multiplier=2.0,
            stop_loss_multiplier=1.0,
            max_holding_period=20,
        )
        labeler = TripleBarrierLabeler(config)
        
        df = labeler.apply_labels(df)
        # Adds columns: tb_label, tb_return, tb_barrier, tb_holding_period
    """
    
    def __init__(self, config: TripleBarrierConfig | None = None):
        """Initialize the labeler."""
        self.config = config or TripleBarrierConfig()
    
    def calculate_atr(
        self,
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        close: NDArray[np.float64],
        period: int | None = None,
    ) -> NDArray[np.float64]:
        """Calculate Average True Range."""
        period = period or self.config.atr_period
        n = len(close)
        
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        
        for i in range(1, n):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr[i] = max(tr1, tr2, tr3)
        
        # EMA of true range
        atr = np.zeros(n)
        atr[:period] = np.nan
        atr[period-1] = np.mean(tr[:period])
        
        alpha = 2 / (period + 1)
        for i in range(period, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
        
        return atr
    
    def calculate_dynamic_volatility(
        self,
        close: NDArray[np.float64],
        lookback: int | None = None,
    ) -> NDArray[np.float64]:
        """Calculate rolling volatility for barrier scaling."""
        lookback = lookback or self.config.volatility_lookback
        n = len(close)
        
        returns = np.zeros(n)
        returns[1:] = np.diff(np.log(close))
        
        volatility = np.zeros(n)
        volatility[:lookback] = np.nan
        
        for i in range(lookback, n):
            volatility[i] = np.std(returns[i-lookback+1:i+1])
        
        # Annualize (assuming 15-min bars, ~26 bars per day, 252 trading days)
        volatility *= np.sqrt(26 * 252)
        
        return volatility
    
    def get_barrier_levels(
        self,
        entry_price: float,
        atr: float,
        volatility: float | None = None,
    ) -> tuple[float, float]:
        """Calculate take profit and stop loss levels."""
        if self.config.use_volatility_scaling and volatility is not None:
            # Scale barriers by volatility
            scale = volatility / 0.20  # Normalize to 20% annual vol
            scale = np.clip(scale, 0.5, 2.0)  # Limit scaling
            
            take_profit = entry_price * (1 + self.config.take_profit_multiplier * atr / entry_price * scale)
            stop_loss = entry_price * (1 - self.config.stop_loss_multiplier * atr / entry_price * scale)
        else:
            take_profit = entry_price + self.config.take_profit_multiplier * atr
            stop_loss = entry_price - self.config.stop_loss_multiplier * atr
        
        if self.config.symmetric:
            distance = self.config.take_profit_multiplier * atr
            take_profit = entry_price + distance
            stop_loss = entry_price - distance
        
        return take_profit, stop_loss
    
    def label_single_point(
        self,
        idx: int,
        close: NDArray[np.float64],
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        atr: NDArray[np.float64],
        volatility: NDArray[np.float64] | None = None,
    ) -> tuple[int, float, BarrierType, int]:
        """
        Generate label for a single point using triple barrier method.
        
        Returns:
            Tuple of (label, return, barrier_type, holding_period)
        """
        n = len(close)
        entry_price = close[idx]
        entry_atr = atr[idx]
        
        if np.isnan(entry_atr) or entry_atr <= 0:
            return 0, 0.0, BarrierType.NONE, 0
        
        vol = volatility[idx] if volatility is not None else None
        take_profit, stop_loss = self.get_barrier_levels(entry_price, entry_atr, vol)
        
        max_period = min(idx + self.config.max_holding_period + 1, n)
        
        # Check each subsequent bar
        for j in range(idx + 1, max_period):
            # Check if take profit hit (high touches barrier)
            if high[j] >= take_profit:
                ret = (take_profit - entry_price) / entry_price
                return 1, ret, BarrierType.TAKE_PROFIT, j - idx
            
            # Check if stop loss hit (low touches barrier)
            if low[j] <= stop_loss:
                ret = (stop_loss - entry_price) / entry_price
                return -1 if ret < -self.config.min_profit_threshold else 0, ret, BarrierType.STOP_LOSS, j - idx
        
        # Time barrier hit - use final close
        if max_period - 1 > idx:
            final_price = close[max_period - 1]
            ret = (final_price - entry_price) / entry_price
            
            # Label based on return
            if ret > self.config.min_profit_threshold:
                label = 1
            elif ret < -self.config.min_profit_threshold:
                label = -1
            else:
                label = 0
            
            return label, ret, BarrierType.TIME_BARRIER, max_period - 1 - idx
        
        return 0, 0.0, BarrierType.NONE, 0
    
    def apply_labels(
        self,
        df: pl.DataFrame,
        close_col: str = "close",
        high_col: str = "high",
        low_col: str = "low",
    ) -> pl.DataFrame:
        """
        Apply triple barrier labels to a DataFrame.
        
        Adds columns:
        - tb_label: -1 (loss), 0 (neutral), 1 (profit)
        - tb_return: Actual return achieved
        - tb_barrier: Which barrier was hit
        - tb_holding_period: Bars held
        
        Args:
            df: DataFrame with OHLCV data
            close_col: Name of close column
            high_col: Name of high column
            low_col: Name of low column
        
        Returns:
            DataFrame with label columns added
        """
        close = df[close_col].to_numpy()
        high = df[high_col].to_numpy()
        low = df[low_col].to_numpy()
        n = len(close)
        
        # Calculate ATR
        atr = self.calculate_atr(high, low, close)
        
        # Calculate volatility if needed
        volatility = None
        if self.config.use_volatility_scaling:
            volatility = self.calculate_dynamic_volatility(close)
        
        # Generate labels
        labels = np.zeros(n, dtype=np.int8)
        returns = np.zeros(n, dtype=np.float64)
        barriers = ["none"] * n
        holding_periods = np.zeros(n, dtype=np.int32)
        
        # Leave last max_holding_period bars unlabeled
        for i in range(n - self.config.max_holding_period):
            if np.isnan(atr[i]):
                continue
            
            label, ret, barrier, period = self.label_single_point(
                i, close, high, low, atr, volatility
            )
            
            labels[i] = label
            returns[i] = ret
            barriers[i] = barrier.value
            holding_periods[i] = period
        
        # Mark last bars as NaN (no valid label)
        labels[-self.config.max_holding_period:] = 0
        returns[-self.config.max_holding_period:] = np.nan
        
        # Add columns to DataFrame
        df = df.with_columns([
            pl.Series("tb_label", labels),
            pl.Series("tb_return", returns),
            pl.Series("tb_barrier", barriers),
            pl.Series("tb_holding_period", holding_periods),
        ])
        
        # Log statistics
        valid_labels = labels[:-self.config.max_holding_period]
        logger.info(f"Triple Barrier Labels - Profit: {(valid_labels == 1).sum()}, "
                   f"Loss: {(valid_labels == -1).sum()}, "
                   f"Neutral: {(valid_labels == 0).sum()}")
        
        return df
    
    def apply_binary_labels(
        self,
        df: pl.DataFrame,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """
        Apply binary (0/1) labels instead of ternary (-1/0/1).
        
        Positive labels (profit) -> 1
        Non-positive labels -> 0
        """
        df = self.apply_labels(df, **kwargs)
        
        df = df.with_columns([
            (pl.col("tb_label") > 0).cast(pl.Int8).alias("target"),
        ])
        
        return df


# =============================================================================
# META-LABELING
# =============================================================================

class MetaLabeler:
    """
    Meta-Labeling for bet sizing.
    
    Two-stage approach:
    1. Primary model predicts side (long/short)
    2. Meta-model predicts probability of success
    
    Final prediction = side * probability
    
    This separates the "what to trade" decision from the "how much to bet" decision.
    
    Example:
        labeler = MetaLabeler()
        
        # Stage 1: Generate primary signals
        primary_signals = primary_model.predict(X)
        
        # Stage 2: Create meta-labels
        df = labeler.create_meta_labels(df, primary_signals)
        
        # Train meta-model on meta-labels
        meta_model.fit(X, df["meta_label"])
    """
    
    def __init__(self, config: MetaLabelConfig | None = None):
        """Initialize the meta-labeler."""
        self.config = config or MetaLabelConfig()
    
    def create_meta_labels(
        self,
        df: pl.DataFrame,
        primary_signals: NDArray[np.int64],
        tb_labels: str = "tb_label",
        tb_returns: str = "tb_return",
    ) -> pl.DataFrame:
        """
        Create meta-labels from primary signals and actual outcomes.
        
        Meta-label = 1 if (primary_signal * actual_outcome) > 0
                   = 0 otherwise
        
        Args:
            df: DataFrame with triple barrier labels
            primary_signals: Array of primary model signals (-1, 0, 1)
            tb_labels: Column name for triple barrier labels
            tb_returns: Column name for triple barrier returns
        
        Returns:
            DataFrame with meta_label column added
        """
        if tb_labels not in df.columns:
            raise ValueError(f"Column {tb_labels} not found. Run TripleBarrierLabeler first.")
        
        actual_labels = df[tb_labels].to_numpy()
        actual_returns = df[tb_returns].to_numpy() if tb_returns in df.columns else None
        
        n = len(actual_labels)
        meta_labels = np.zeros(n, dtype=np.int8)
        bet_sizes = np.zeros(n, dtype=np.float64)
        
        for i in range(n):
            if primary_signals[i] == 0:
                # No signal from primary model
                meta_labels[i] = 0
                bet_sizes[i] = 0
                continue
            
            # Check if primary signal was correct
            signal_correct = (primary_signals[i] * actual_labels[i]) > 0
            meta_labels[i] = 1 if signal_correct else 0
            
            # Calculate bet size based on return magnitude
            if signal_correct and actual_returns is not None:
                ret = abs(actual_returns[i])
                bet_sizes[i] = min(ret / 0.02, self.config.max_bet_size)  # Normalize to 2% return
            else:
                bet_sizes[i] = 0
        
        # Add columns
        df = df.with_columns([
            pl.Series("primary_signal", primary_signals),
            pl.Series("meta_label", meta_labels),
            pl.Series("bet_size", bet_sizes),
        ])
        
        return df
    
    def get_final_signal(
        self,
        primary_signal: int,
        meta_probability: float,
        threshold: float | None = None,
    ) -> tuple[int, float]:
        """
        Get final trading signal from primary and meta predictions.
        
        Args:
            primary_signal: Primary model signal (-1, 0, 1)
            meta_probability: Meta-model probability of success
            threshold: Minimum probability threshold
        
        Returns:
            Tuple of (final_signal, bet_size)
        """
        threshold = threshold or self.config.primary_threshold
        
        if primary_signal == 0:
            return 0, 0.0
        
        if meta_probability < threshold:
            return 0, 0.0
        
        # Bet size scaled by probability
        bet_size = meta_probability if self.config.enable_bet_sizing else 1.0
        bet_size = min(bet_size, self.config.max_bet_size)
        
        return primary_signal, bet_size


# =============================================================================
# FRACTIONAL DIFFERENTIATION
# =============================================================================

class FractionalDifferentiation:
    """
    Fractional Differentiation for stationary features.
    
    Standard differentiation (d=1) makes series stationary but loses memory.
    Fractional differentiation (0 < d < 1) preserves memory while achieving stationarity.
    
    From: "Advances in Financial Machine Learning" by López de Prado
    
    Example:
        frac_diff = FractionalDifferentiation(d=0.4)
        df = frac_diff.add_features(df, columns=["close", "volume"])
    """
    
    def __init__(
        self,
        d: float = 0.4,
        threshold: float = 1e-5,
    ):
        """
        Initialize fractional differentiation.
        
        Args:
            d: Differentiation order (0 < d < 1)
            threshold: Weight threshold for truncation
        """
        if not 0 < d < 1:
            raise ValueError("d must be between 0 and 1")
        
        self.d = d
        self.threshold = threshold
        self._weights_cache: dict[int, NDArray[np.float64]] = {}
    
    def get_weights(self, length: int) -> NDArray[np.float64]:
        """Calculate fractional differentiation weights."""
        if length in self._weights_cache:
            return self._weights_cache[length]
        
        weights = np.zeros(length)
        weights[0] = 1.0
        
        for k in range(1, length):
            weights[k] = weights[k-1] * (self.d - k + 1) / k
        
        # Truncate small weights
        weights[np.abs(weights) < self.threshold] = 0
        
        self._weights_cache[length] = weights
        return weights
    
    def differentiate(
        self,
        series: NDArray[np.float64],
        min_weight: float | None = None,
    ) -> NDArray[np.float64]:
        """
        Apply fractional differentiation to a series.
        
        Args:
            series: Input series
            min_weight: Minimum weight threshold
        
        Returns:
            Fractionally differentiated series
        """
        n = len(series)
        
        # Get weights
        weights = self.get_weights(n)
        
        # Find truncation point
        min_weight = min_weight or self.threshold
        valid_weights = np.where(np.abs(weights) >= min_weight)[0]
        if len(valid_weights) == 0:
            return np.full(n, np.nan)
        
        truncation = valid_weights[-1] + 1
        weights = weights[:truncation]
        
        # Apply convolution
        result = np.zeros(n)
        result[:truncation-1] = np.nan
        
        for i in range(truncation - 1, n):
            result[i] = np.dot(weights, series[i-truncation+1:i+1][::-1])
        
        return result
    
    def add_features(
        self,
        df: pl.DataFrame,
        columns: list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Add fractionally differentiated features to DataFrame.
        
        Args:
            df: Input DataFrame
            columns: Columns to differentiate (default: close, volume)
        
        Returns:
            DataFrame with fractionally differentiated columns
        """
        columns = columns or ["close", "volume"]
        
        for col in columns:
            if col not in df.columns:
                continue
            
            series = df[col].to_numpy()
            diff_series = self.differentiate(series)
            
            df = df.with_columns([
                pl.Series(f"{col}_fracdiff", diff_series),
            ])
        
        return df


# =============================================================================
# MICROSTRUCTURE FEATURES
# =============================================================================

class MicrostructureFeatures:
    """
    Microstructure features from OHLC data.
    
    Estimates order flow and market microstructure without tick data.
    """
    
    @staticmethod
    def add_features(df: pl.DataFrame) -> pl.DataFrame:
        """
        Add microstructure features to DataFrame.
        
        Features:
        - kyle_lambda: Kyle's lambda (price impact)
        - roll_spread: Roll's spread estimate
        - amihud_illiq: Amihud illiquidity ratio
        - volume_imbalance: Buy/sell volume imbalance
        - close_location: Close location within bar
        """
        close = df["close"].to_numpy()
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        open_ = df["open"].to_numpy()
        volume = df["volume"].to_numpy()
        n = len(close)
        
        # Close location value (0 to 1, where 1 = close at high)
        range_hl = high - low
        range_hl[range_hl == 0] = 1e-10
        close_location = (close - low) / range_hl
        
        # Volume imbalance estimate (close > open suggests buying pressure)
        # Positive = buying pressure, Negative = selling pressure
        price_move = close - open_
        volume_imbalance = np.sign(price_move) * volume
        
        # Rolling volume imbalance ratio
        window = 20
        volume_imbalance_ratio = np.zeros(n)
        volume_imbalance_ratio[:window] = np.nan
        for i in range(window, n):
            window_data = volume_imbalance[i-window:i]
            buy_vol = np.sum(window_data[window_data > 0])
            sell_vol = np.abs(np.sum(window_data[window_data < 0]))
            total = buy_vol + sell_vol
            volume_imbalance_ratio[i] = (buy_vol - sell_vol) / total if total > 0 else 0
        
        # Amihud illiquidity ratio (|return| / volume)
        returns = np.zeros(n)
        returns[1:] = np.abs(np.diff(np.log(close)))
        amihud = np.zeros(n)
        amihud[volume > 0] = returns[volume > 0] / volume[volume > 0]
        
        # Rolling Amihud (20-day average)
        amihud_rolling = np.zeros(n)
        amihud_rolling[:window] = np.nan
        for i in range(window, n):
            amihud_rolling[i] = np.mean(amihud[i-window:i])
        
        # Normalize Amihud by scaling
        amihud_rolling = amihud_rolling * 1e6  # Scale for readability
        
        # Roll's spread estimate (based on serial covariance of returns)
        roll_spread = np.zeros(n)
        roll_spread[:window+1] = np.nan
        for i in range(window + 1, n):
            ret_window = returns[i-window:i]
            ret_window_lag = returns[i-window-1:i-1]
            cov = np.cov(ret_window, ret_window_lag)[0, 1]
            roll_spread[i] = 2 * np.sqrt(-cov) if cov < 0 else 0
        
        # VPIN (Volume-Synchronized Probability of Informed Trading) proxy
        # Using absolute volume imbalance as proxy
        vpin_proxy = np.abs(volume_imbalance_ratio)
        
        # Add features
        df = df.with_columns([
            pl.Series("micro_close_location", close_location),
            pl.Series("micro_volume_imbalance", volume_imbalance_ratio),
            pl.Series("micro_amihud", amihud_rolling),
            pl.Series("micro_roll_spread", roll_spread),
            pl.Series("micro_vpin_proxy", vpin_proxy),
        ])
        
        return df


# =============================================================================
# CALENDAR FEATURES
# =============================================================================

class CalendarFeatures:
    """Calendar and time-based features."""
    
    @staticmethod
    def add_features(df: pl.DataFrame, timestamp_col: str = "timestamp") -> pl.DataFrame:
        """
        Add calendar features to DataFrame.
        
        Features:
        - Hour of day (0-23)
        - Day of week (0-6)
        - Week of year
        - Month
        - Is month end
        - Is quarter end
        - Days to month end
        - Is first/last hour of session
        """
        if timestamp_col not in df.columns:
            logger.warning(f"Timestamp column {timestamp_col} not found")
            return df
        
        df = df.with_columns([
            # Time features
            pl.col(timestamp_col).dt.hour().alias("cal_hour"),
            pl.col(timestamp_col).dt.weekday().alias("cal_day_of_week"),
            pl.col(timestamp_col).dt.week().alias("cal_week_of_year"),
            pl.col(timestamp_col).dt.month().alias("cal_month"),
            pl.col(timestamp_col).dt.day().alias("cal_day_of_month"),
            
            # Session features (US market 9:30-16:00 ET)
            (pl.col(timestamp_col).dt.hour() == 9).alias("cal_first_hour"),
            (pl.col(timestamp_col).dt.hour() == 15).alias("cal_last_hour"),
            
            # Period features
            (pl.col(timestamp_col).dt.day() >= 25).alias("cal_month_end_week"),
            ((pl.col(timestamp_col).dt.day() <= 5)).alias("cal_month_start_week"),
        ])
        
        # Quarter end (March, June, September, December)
        df = df.with_columns([
            ((pl.col("cal_month").is_in([3, 6, 9, 12])) & (pl.col("cal_day_of_month") >= 25))
            .alias("cal_quarter_end"),
        ])
        
        # Normalize hour to trading hours (0-1 scale)
        # Assuming 6.5 trading hours per day
        df = df.with_columns([
            ((pl.col("cal_hour") - 9.5) / 6.5).clip(0, 1).alias("cal_trading_hour_pct"),
        ])
        
        return df


# =============================================================================
# FEATURE INTERACTIONS
# =============================================================================

class FeatureInteractions:
    """Generate feature interactions."""
    
    @staticmethod
    def add_interactions(
        df: pl.DataFrame,
        feature_pairs: list[tuple[str, str]] | None = None,
    ) -> pl.DataFrame:
        """
        Add interaction features.
        
        Default interactions if none specified:
        - RSI * ADX (momentum with trend strength)
        - MACD * Volume (signal with volume confirmation)
        - ATR * Close (volatility normalized by price)
        """
        default_pairs = [
            ("rsi_14", "adx_14"),
            ("macd_line", "volume"),
            ("atr_14", "close"),
            ("bb_width_20", "rsi_14"),
            ("return_1", "volume"),
        ]
        
        pairs = feature_pairs or default_pairs
        
        for col1, col2 in pairs:
            if col1 not in df.columns or col2 not in df.columns:
                continue
            
            # Multiplication interaction
            interaction_name = f"interact_{col1}_{col2}"
            df = df.with_columns([
                (pl.col(col1) * pl.col(col2)).alias(interaction_name),
            ])
            
            # Ratio interaction (if denominator not zero)
            ratio_name = f"ratio_{col1}_{col2}"
            df = df.with_columns([
                (pl.col(col1) / pl.col(col2).replace(0, None)).alias(ratio_name),
            ])
        
        return df


# =============================================================================
# ADVANCED FEATURE PIPELINE
# =============================================================================

class AdvancedFeaturePipeline:
    """
    Pipeline for generating advanced features.
    
    Combines:
    - Triple Barrier labels
    - Meta-labeling
    - Fractional differentiation
    - Microstructure features
    - Calendar features
    - Feature interactions
    """
    
    def __init__(self, config: AdvancedFeatureConfig | None = None):
        """Initialize the pipeline."""
        self.config = config or AdvancedFeatureConfig()
        
        # Initialize components
        self.triple_barrier = TripleBarrierLabeler()
        self.meta_labeler = MetaLabeler()
        self.frac_diff = FractionalDifferentiation(d=self.config.frac_diff_d)
        self.microstructure = MicrostructureFeatures()
        self.calendar = CalendarFeatures()
        self.interactions = FeatureInteractions()
    
    def generate_advanced_features(
        self,
        df: pl.DataFrame,
        include_labels: bool = True,
    ) -> pl.DataFrame:
        """
        Generate all advanced features.
        
        Args:
            df: Input DataFrame with OHLCV data
            include_labels: Whether to include triple barrier labels
        
        Returns:
            DataFrame with advanced features added
        """
        logger.info("Generating advanced features...")
        
        # Triple Barrier labels (if requested)
        if include_labels:
            df = self.triple_barrier.apply_labels(df)
        
        # Fractional differentiation
        if self.config.enable_frac_diff:
            df = self.frac_diff.add_features(df, columns=["close", "volume"])
        
        # Microstructure features
        if self.config.enable_microstructure:
            df = self.microstructure.add_features(df)
        
        # Calendar features
        if self.config.enable_calendar:
            df = self.calendar.add_features(df)
        
        # Feature interactions
        if self.config.enable_interactions:
            df = self.interactions.add_interactions(df, self.config.interaction_pairs)
        
        return df
    
    def create_target_with_triple_barrier(
        self,
        df: pl.DataFrame,
        binary: bool = True,
    ) -> pl.DataFrame:
        """
        Create target variable using triple barrier method.
        
        Args:
            df: Input DataFrame
            binary: If True, create binary target (0/1)
        
        Returns:
            DataFrame with target column
        """
        if "tb_label" not in df.columns:
            df = self.triple_barrier.apply_labels(df)
        
        if binary:
            df = df.with_columns([
                (pl.col("tb_label") > 0).cast(pl.Int8).alias("target"),
            ])
        else:
            df = df.with_columns([
                pl.col("tb_label").alias("target"),
            ])
        
        return df


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "BarrierType",
    "LabelType",
    # Configurations
    "TripleBarrierConfig",
    "MetaLabelConfig",
    "AdvancedFeatureConfig",
    # Classes
    "TripleBarrierLabeler",
    "MetaLabeler",
    "FractionalDifferentiation",
    "MicrostructureFeatures",
    "CalendarFeatures",
    "FeatureInteractions",
    "AdvancedFeaturePipeline",
]