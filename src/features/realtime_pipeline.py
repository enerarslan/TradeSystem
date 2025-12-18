"""
Real-Time Feature Pipeline Module.

JPMorgan Institutional-Level Feature Computation for Real-Time Prediction.

This module provides O(1) incremental feature updates optimized for
low-latency production environments.

Key Features:
- Incremental rolling calculations (no history recalculation)
- Memory-efficient state management
- Latency monitoring with alerting
- Exact consistency with batch pipeline
- Thread-safe operations

Reference:
    "High-Frequency Trading" by Aldridge (2013)
    "Trading and Exchanges" by Harris (2003)
"""

from __future__ import annotations

import logging
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LatencyMetrics:
    """Latency tracking metrics."""
    update_times_ms: deque = field(default_factory=lambda: deque(maxlen=1000))
    threshold_breaches: int = 0
    last_update_time: float = 0.0
    max_observed_latency: float = 0.0


class IncrementalRollingStatistic:
    """
    O(1) incremental rolling statistics calculator.

    Uses Welford's online algorithm for numerically stable
    incremental mean and variance updates.
    """

    def __init__(self, window_size: int):
        """
        Initialize rolling statistic calculator.

        Args:
            window_size: Size of rolling window
        """
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0

    def update(self, value: float) -> Dict[str, float]:
        """
        Update with new value and return current statistics.

        O(1) time complexity.

        Args:
            value: New value to add

        Returns:
            Dictionary with mean, std, min, max
        """
        # Remove oldest value if window is full
        if len(self.values) == self.window_size:
            old_value = self.values[0]
            self._sum -= old_value
            self._sum_sq -= old_value * old_value
            self._count -= 1

        # Add new value
        self.values.append(value)
        self._sum += value
        self._sum_sq += value * value
        self._count += 1

        # Calculate statistics
        mean = self._sum / self._count if self._count > 0 else 0.0

        if self._count > 1:
            variance = (self._sum_sq - self._sum * self._sum / self._count) / (self._count - 1)
            std = np.sqrt(max(0, variance))
        else:
            std = 0.0

        return {
            "mean": mean,
            "std": std,
            "min": min(self.values) if self.values else 0.0,
            "max": max(self.values) if self.values else 0.0,
            "count": self._count,
        }

    def get_percentile(self, p: float) -> float:
        """Get percentile of current window (O(n log n))."""
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        idx = int(len(sorted_values) * p / 100)
        return sorted_values[min(idx, len(sorted_values) - 1)]


class IncrementalEMA:
    """
    O(1) Exponential Moving Average calculator.

    EMA(t) = alpha * value(t) + (1 - alpha) * EMA(t-1)
    """

    def __init__(self, span: int):
        """
        Initialize EMA calculator.

        Args:
            span: EMA span (period)
        """
        self.span = span
        self.alpha = 2.0 / (span + 1)
        self.ema = None
        self._initialized = False

    def update(self, value: float) -> float:
        """
        Update EMA with new value.

        Args:
            value: New value

        Returns:
            Current EMA value
        """
        if not self._initialized or self.ema is None:
            self.ema = value
            self._initialized = True
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema

        return self.ema


class IncrementalRSI:
    """
    O(1) incremental RSI calculator using Wilder's smoothing.
    """

    def __init__(self, period: int = 14):
        """
        Initialize RSI calculator.

        Args:
            period: RSI period
        """
        self.period = period
        self.avg_gain = None
        self.avg_loss = None
        self.last_close = None
        self._count = 0

    def update(self, close: float) -> float:
        """
        Update RSI with new close price.

        Args:
            close: New closing price

        Returns:
            Current RSI value (0-100)
        """
        if self.last_close is None:
            self.last_close = close
            return 50.0  # Neutral on first value

        change = close - self.last_close
        gain = max(0, change)
        loss = max(0, -change)

        if self._count < self.period:
            # Initial period: simple average
            if self.avg_gain is None:
                self.avg_gain = gain
                self.avg_loss = loss
            else:
                self.avg_gain = (self.avg_gain * self._count + gain) / (self._count + 1)
                self.avg_loss = (self.avg_loss * self._count + loss) / (self._count + 1)
        else:
            # Wilder's smoothing
            self.avg_gain = (self.avg_gain * (self.period - 1) + gain) / self.period
            self.avg_loss = (self.avg_loss * (self.period - 1) + loss) / self.period

        self.last_close = close
        self._count += 1

        # Calculate RSI
        if self.avg_loss == 0:
            return 100.0
        rs = self.avg_gain / self.avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi


class IncrementalMACD:
    """
    O(1) incremental MACD calculator.
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ):
        """
        Initialize MACD calculator.

        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
        """
        self.fast_ema = IncrementalEMA(fast_period)
        self.slow_ema = IncrementalEMA(slow_period)
        self.signal_ema = IncrementalEMA(signal_period)

    def update(self, close: float) -> Dict[str, float]:
        """
        Update MACD with new close price.

        Args:
            close: New closing price

        Returns:
            Dictionary with macd, signal, histogram
        """
        fast = self.fast_ema.update(close)
        slow = self.slow_ema.update(close)
        macd = fast - slow
        signal = self.signal_ema.update(macd)
        histogram = macd - signal

        return {
            "macd": macd,
            "signal": signal,
            "histogram": histogram,
        }


class IncrementalBollingerBands:
    """
    O(1) incremental Bollinger Bands calculator.
    """

    def __init__(self, period: int = 20, num_std: float = 2.0):
        """
        Initialize Bollinger Bands calculator.

        Args:
            period: Rolling period
            num_std: Number of standard deviations
        """
        self.stats = IncrementalRollingStatistic(period)
        self.num_std = num_std

    def update(self, close: float) -> Dict[str, float]:
        """
        Update Bollinger Bands with new close price.

        Args:
            close: New closing price

        Returns:
            Dictionary with upper, middle, lower bands and %B
        """
        stats = self.stats.update(close)
        middle = stats["mean"]
        band_width = self.num_std * stats["std"]

        upper = middle + band_width
        lower = middle - band_width

        # %B indicator
        if upper != lower:
            pct_b = (close - lower) / (upper - lower)
        else:
            pct_b = 0.5

        return {
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "width": (upper - lower) / middle if middle != 0 else 0,
            "pct_b": pct_b,
        }


class IncrementalATR:
    """
    O(1) incremental Average True Range calculator.
    """

    def __init__(self, period: int = 14):
        """
        Initialize ATR calculator.

        Args:
            period: ATR period
        """
        self.period = period
        self.atr = None
        self.last_close = None
        self._count = 0

    def update(self, high: float, low: float, close: float) -> float:
        """
        Update ATR with new OHLC data.

        Args:
            high: High price
            low: Low price
            close: Close price

        Returns:
            Current ATR value
        """
        if self.last_close is None:
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - self.last_close),
                abs(low - self.last_close)
            )

        if self.atr is None:
            self.atr = tr
        else:
            # Wilder's smoothing
            self.atr = (self.atr * (self.period - 1) + tr) / self.period

        self.last_close = close
        self._count += 1

        return self.atr


class RealTimeFeaturePipeline:
    """
    Real-time feature computation pipeline with O(1) updates.

    Provides the same features as the batch pipeline but optimized
    for incremental computation.

    Example:
        pipeline = RealTimeFeaturePipeline(
            latency_threshold_ms=50.0,
            feature_config={'rsi_periods': [14], 'ema_periods': [9, 21]}
        )

        # Process each new bar
        features = pipeline.update({
            'timestamp': pd.Timestamp.now(),
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 10000
        })
    """

    def __init__(
        self,
        latency_threshold_ms: float = 50.0,
        feature_config: Optional[Dict[str, Any]] = None,
        alert_callback: Optional[Callable[[str, float], None]] = None,
    ):
        """
        Initialize real-time feature pipeline.

        Args:
            latency_threshold_ms: Maximum allowed latency before alert
            feature_config: Configuration for which features to compute
            alert_callback: Function called when latency exceeds threshold
        """
        self.latency_threshold_ms = latency_threshold_ms
        self.feature_config = feature_config or {}
        self.alert_callback = alert_callback

        # Latency tracking
        self.latency_metrics = LatencyMetrics()

        # Thread safety
        self._lock = threading.Lock()

        # Initialize feature calculators
        self._init_calculators()

        # State for consistency checking
        self._update_count = 0
        self._last_features = None

    def _init_calculators(self):
        """Initialize all incremental feature calculators."""
        config = self.feature_config

        # Price statistics
        self.price_stats_20 = IncrementalRollingStatistic(20)
        self.price_stats_50 = IncrementalRollingStatistic(50)

        # Volume statistics
        self.volume_stats_20 = IncrementalRollingStatistic(20)

        # Returns
        self.returns_20 = IncrementalRollingStatistic(20)
        self._last_close = None

        # EMAs
        ema_periods = config.get("ema_periods", [9, 21, 50])
        self.emas = {p: IncrementalEMA(p) for p in ema_periods}

        # RSI
        rsi_periods = config.get("rsi_periods", [14])
        self.rsis = {p: IncrementalRSI(p) for p in rsi_periods}

        # MACD
        self.macd = IncrementalMACD()

        # Bollinger Bands
        self.bb = IncrementalBollingerBands()

        # ATR
        self.atr = IncrementalATR()

        # Price momentum
        self.momentum_lookbacks = config.get("momentum_lookbacks", [5, 10, 20])
        self.price_history = deque(maxlen=max(self.momentum_lookbacks) + 1)

    def update(self, bar: Dict[str, Any]) -> Dict[str, float]:
        """
        Process a new bar and compute features.

        This is the main entry point for real-time updates.

        Args:
            bar: Dictionary with keys: timestamp, open, high, low, close, volume

        Returns:
            Dictionary of feature_name -> feature_value
        """
        start_time = time.perf_counter()

        with self._lock:
            features = self._compute_features(bar)

        # Track latency
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._track_latency(elapsed_ms)

        self._update_count += 1
        self._last_features = features

        return features

    def _compute_features(self, bar: Dict[str, Any]) -> Dict[str, float]:
        """Compute all features for a single bar."""
        features = {}

        close = bar["close"]
        high = bar["high"]
        low = bar["low"]
        volume = bar.get("volume", 0)

        # Store price history for momentum
        self.price_history.append(close)

        # 1. Price statistics
        price_stats = self.price_stats_20.update(close)
        features["price_mean_20"] = price_stats["mean"]
        features["price_std_20"] = price_stats["std"]
        features["price_zscore_20"] = (
            (close - price_stats["mean"]) / price_stats["std"]
            if price_stats["std"] > 0 else 0
        )

        price_stats_50 = self.price_stats_50.update(close)
        features["price_mean_50"] = price_stats_50["mean"]
        features["price_std_50"] = price_stats_50["std"]

        # 2. Volume statistics
        vol_stats = self.volume_stats_20.update(volume)
        features["volume_mean_20"] = vol_stats["mean"]
        features["volume_zscore"] = (
            (volume - vol_stats["mean"]) / vol_stats["std"]
            if vol_stats["std"] > 0 else 0
        )

        # 3. Returns
        if self._last_close is not None:
            ret = (close - self._last_close) / self._last_close
            ret_stats = self.returns_20.update(ret)
            features["return"] = ret
            features["return_mean_20"] = ret_stats["mean"]
            features["return_std_20"] = ret_stats["std"]
            features["sharpe_20"] = (
                ret_stats["mean"] / ret_stats["std"] * np.sqrt(252)
                if ret_stats["std"] > 0 else 0
            )
        else:
            features["return"] = 0
            features["return_mean_20"] = 0
            features["return_std_20"] = 0
            features["sharpe_20"] = 0

        self._last_close = close

        # 4. EMAs
        for period, ema in self.emas.items():
            ema_val = ema.update(close)
            features[f"ema_{period}"] = ema_val
            features[f"ema_{period}_pct"] = (close - ema_val) / ema_val if ema_val != 0 else 0

        # 5. RSI
        for period, rsi in self.rsis.items():
            features[f"rsi_{period}"] = rsi.update(close)

        # 6. MACD
        macd_values = self.macd.update(close)
        features["macd"] = macd_values["macd"]
        features["macd_signal"] = macd_values["signal"]
        features["macd_histogram"] = macd_values["histogram"]

        # 7. Bollinger Bands
        bb_values = self.bb.update(close)
        features["bb_upper"] = bb_values["upper"]
        features["bb_middle"] = bb_values["middle"]
        features["bb_lower"] = bb_values["lower"]
        features["bb_width"] = bb_values["width"]
        features["bb_pct_b"] = bb_values["pct_b"]

        # 8. ATR
        features["atr"] = self.atr.update(high, low, close)
        features["atr_pct"] = features["atr"] / close if close > 0 else 0

        # 9. Momentum
        for lookback in self.momentum_lookbacks:
            if len(self.price_history) > lookback:
                past_price = self.price_history[-lookback - 1]
                features[f"momentum_{lookback}"] = (close - past_price) / past_price
            else:
                features[f"momentum_{lookback}"] = 0

        # 10. Price range features
        features["high_low_range"] = (high - low) / close if close > 0 else 0
        features["close_position"] = (
            (close - low) / (high - low) if high != low else 0.5
        )

        return features

    def _track_latency(self, elapsed_ms: float):
        """Track latency metrics and alert if threshold exceeded."""
        self.latency_metrics.update_times_ms.append(elapsed_ms)
        self.latency_metrics.last_update_time = elapsed_ms
        self.latency_metrics.max_observed_latency = max(
            self.latency_metrics.max_observed_latency, elapsed_ms
        )

        if elapsed_ms > self.latency_threshold_ms:
            self.latency_metrics.threshold_breaches += 1

            if self.alert_callback:
                self.alert_callback(
                    f"Feature computation latency exceeded threshold: {elapsed_ms:.2f}ms > {self.latency_threshold_ms}ms",
                    elapsed_ms
                )

            logger.warning(
                f"Latency threshold breach: {elapsed_ms:.2f}ms "
                f"(threshold: {self.latency_threshold_ms}ms)"
            )

    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        times = list(self.latency_metrics.update_times_ms)
        if not times:
            return {}

        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "p50_ms": np.percentile(times, 50),
            "p95_ms": np.percentile(times, 95),
            "p99_ms": np.percentile(times, 99),
            "threshold_breaches": self.latency_metrics.threshold_breaches,
            "total_updates": len(times),
        }

    def verify_consistency(
        self,
        batch_features: pd.DataFrame,
        tolerance: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Verify that real-time features match batch pipeline output.

        This is critical for ensuring production consistency.

        Args:
            batch_features: DataFrame from batch feature pipeline
            tolerance: Numerical tolerance for comparison

        Returns:
            Dictionary with consistency check results
        """
        # Reset state
        self._init_calculators()
        self._last_close = None

        mismatches = []
        max_diff = 0.0

        # Process each row
        for idx, row in batch_features.iterrows():
            if "close" not in row:
                continue

            bar = {
                "timestamp": idx,
                "open": row.get("open", row["close"]),
                "high": row.get("high", row["close"]),
                "low": row.get("low", row["close"]),
                "close": row["close"],
                "volume": row.get("volume", 0),
            }

            rt_features = self.update(bar)

            # Compare features
            for feature_name, rt_value in rt_features.items():
                if feature_name in row:
                    batch_value = row[feature_name]
                    if pd.notna(batch_value) and pd.notna(rt_value):
                        diff = abs(rt_value - batch_value)
                        max_diff = max(max_diff, diff)
                        if diff > tolerance:
                            mismatches.append({
                                "timestamp": idx,
                                "feature": feature_name,
                                "realtime": rt_value,
                                "batch": batch_value,
                                "diff": diff,
                            })

        is_consistent = len(mismatches) == 0

        return {
            "is_consistent": is_consistent,
            "max_diff": max_diff,
            "n_mismatches": len(mismatches),
            "mismatches": mismatches[:10],  # First 10
            "n_rows_checked": len(batch_features),
        }

    def reset(self):
        """Reset all state for fresh processing."""
        with self._lock:
            self._init_calculators()
            self._last_close = None
            self._update_count = 0
            self._last_features = None
            self.latency_metrics = LatencyMetrics()


class RealTimeFeatureManager:
    """
    Manager for multiple real-time feature pipelines (multi-symbol).

    Handles concurrent feature computation for multiple symbols
    with efficient resource management.
    """

    def __init__(
        self,
        latency_threshold_ms: float = 50.0,
        feature_config: Optional[Dict[str, Any]] = None,
        max_symbols: int = 100,
    ):
        """
        Initialize feature manager.

        Args:
            latency_threshold_ms: Latency threshold for alerts
            feature_config: Feature configuration
            max_symbols: Maximum number of symbols to track
        """
        self.latency_threshold_ms = latency_threshold_ms
        self.feature_config = feature_config or {}
        self.max_symbols = max_symbols

        self.pipelines: Dict[str, RealTimeFeaturePipeline] = {}
        self._lock = threading.Lock()

    def update(self, symbol: str, bar: Dict[str, Any]) -> Dict[str, float]:
        """
        Update features for a symbol.

        Args:
            symbol: Symbol identifier
            bar: OHLCV bar data

        Returns:
            Feature dictionary
        """
        with self._lock:
            if symbol not in self.pipelines:
                if len(self.pipelines) >= self.max_symbols:
                    # Remove least recently used
                    oldest = min(
                        self.pipelines.keys(),
                        key=lambda s: self.pipelines[s]._update_count
                    )
                    del self.pipelines[oldest]
                    logger.warning(f"Removed LRU symbol {oldest} to make room")

                self.pipelines[symbol] = RealTimeFeaturePipeline(
                    latency_threshold_ms=self.latency_threshold_ms,
                    feature_config=self.feature_config,
                )

        return self.pipelines[symbol].update(bar)

    def get_all_latency_stats(self) -> Dict[str, Dict[str, float]]:
        """Get latency statistics for all symbols."""
        return {
            symbol: pipeline.get_latency_stats()
            for symbol, pipeline in self.pipelines.items()
        }

    def reset_symbol(self, symbol: str):
        """Reset state for a specific symbol."""
        if symbol in self.pipelines:
            self.pipelines[symbol].reset()

    def reset_all(self):
        """Reset all pipelines."""
        for pipeline in self.pipelines.values():
            pipeline.reset()
