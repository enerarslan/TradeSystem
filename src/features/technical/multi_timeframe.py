"""
Multi-Timeframe Features Module.

JPMorgan Institutional-Level Multi-Timeframe Analysis.

This module generates features from multiple timeframes to capture
different market dynamics and improve model robustness.

Reference:
    "Evidence-Based Technical Analysis" by David Aronson
    "Trading and Exchanges" by Larry Harris
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


logger = logging.getLogger(__name__)


class Timeframe(str, Enum):
    """Supported timeframes for multi-timeframe analysis."""
    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    MINUTE_30 = "30min"
    HOUR_1 = "1H"
    HOUR_4 = "4H"
    DAILY = "1D"


@dataclass
class TimeframeConfig:
    """Configuration for a specific timeframe."""
    timeframe: Timeframe
    ma_periods: List[int]
    momentum_periods: List[int]
    volatility_periods: List[int]
    weight: float = 1.0


# Default multi-timeframe configuration
DEFAULT_MTF_CONFIG = [
    TimeframeConfig(
        timeframe=Timeframe.MINUTE_15,
        ma_periods=[20, 50],
        momentum_periods=[14],
        volatility_periods=[20],
        weight=0.4,
    ),
    TimeframeConfig(
        timeframe=Timeframe.HOUR_1,
        ma_periods=[20, 50, 200],
        momentum_periods=[14, 28],
        volatility_periods=[20],
        weight=0.35,
    ),
    TimeframeConfig(
        timeframe=Timeframe.HOUR_4,
        ma_periods=[50, 200],
        momentum_periods=[14],
        volatility_periods=[20],
        weight=0.25,
    ),
]


class MultiTimeframeFeatures(BaseEstimator, TransformerMixin):
    """
    Generate features from multiple timeframes.

    This class resamples data to different timeframes and generates
    technical indicators at each level, then aligns them back to
    the base timeframe.

    Features Generated Per Timeframe:
    - Trend: MA, EMA, price position relative to MA
    - Momentum: RSI, momentum
    - Volatility: ATR, Bollinger %B
    - Cross-timeframe alignment: Trend agreement across timeframes

    Example:
        mtf = MultiTimeframeFeatures(base_timeframe="15min")
        features = mtf.fit_transform(df)
    """

    def __init__(
        self,
        base_timeframe: str = "15min",
        timeframe_configs: Optional[List[TimeframeConfig]] = None,
        include_alignment_features: bool = True,
        include_divergence_features: bool = True,
    ):
        """
        Initialize multi-timeframe feature generator.

        Args:
            base_timeframe: The base timeframe of input data
            timeframe_configs: List of TimeframeConfig for each timeframe
            include_alignment_features: Include cross-timeframe trend alignment
            include_divergence_features: Include momentum/price divergences
        """
        self.base_timeframe = base_timeframe
        self.timeframe_configs = timeframe_configs or DEFAULT_MTF_CONFIG
        self.include_alignment_features = include_alignment_features
        self.include_divergence_features = include_divergence_features

        self._feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "MultiTimeframeFeatures":
        """Fit the transformer (stateless, just validates data)."""
        self._validate_data(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate multi-timeframe features.

        Args:
            X: OHLCV DataFrame with DatetimeIndex

        Returns:
            DataFrame with multi-timeframe features added
        """
        self._validate_data(X)
        features = pd.DataFrame(index=X.index)

        # Generate features for each timeframe
        for config in self.timeframe_configs:
            tf_features = self._generate_timeframe_features(X, config)
            features = pd.concat([features, tf_features], axis=1)

        # Cross-timeframe alignment features
        if self.include_alignment_features:
            alignment_features = self._generate_alignment_features(X, features)
            features = pd.concat([features, alignment_features], axis=1)

        # Cross-timeframe divergence features
        if self.include_divergence_features:
            divergence_features = self._generate_divergence_features(X, features)
            features = pd.concat([features, divergence_features], axis=1)

        self._feature_names = features.columns.tolist()
        logger.info(f"Generated {len(features.columns)} multi-timeframe features")

        return features

    def _validate_data(self, X: pd.DataFrame) -> None:
        """Validate input data."""
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex for resampling")

        required_cols = ["open", "high", "low", "close"]
        missing = set(required_cols) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _generate_timeframe_features(
        self,
        X: pd.DataFrame,
        config: TimeframeConfig,
    ) -> pd.DataFrame:
        """Generate features for a specific timeframe."""
        features = pd.DataFrame(index=X.index)
        tf_str = config.timeframe.value
        prefix = f"mtf_{tf_str}_"

        # Resample to higher timeframe
        resampled = self._resample_ohlcv(X, tf_str)

        # Generate indicators on resampled data
        close = resampled["close"]
        high = resampled["high"]
        low = resampled["low"]

        # Moving Averages
        for period in config.ma_periods:
            ma = close.rolling(period).mean()
            ema = close.ewm(span=period, adjust=False).mean()

            # Forward fill to base timeframe
            features[f"{prefix}sma_{period}"] = ma.reindex(X.index, method="ffill")
            features[f"{prefix}ema_{period}"] = ema.reindex(X.index, method="ffill")

            # Price position relative to MA
            ma_aligned = ma.reindex(X.index, method="ffill")
            features[f"{prefix}price_above_sma_{period}"] = (
                X["close"] > ma_aligned
            ).astype(np.int8)
            features[f"{prefix}price_sma_{period}_pct"] = (
                (X["close"] - ma_aligned) / ma_aligned * 100
            )

        # Momentum indicators
        for period in config.momentum_periods:
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            features[f"{prefix}rsi_{period}"] = rsi.reindex(X.index, method="ffill")

            # Momentum
            mom = close / close.shift(period) - 1
            features[f"{prefix}momentum_{period}"] = mom.reindex(X.index, method="ffill")

        # Volatility indicators
        for period in config.volatility_periods:
            # ATR
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            features[f"{prefix}atr_{period}"] = atr.reindex(X.index, method="ffill")

            # Bollinger %B
            ma = close.rolling(period).mean()
            std = close.rolling(period).std()
            upper = ma + 2 * std
            lower = ma - 2 * std
            pct_b = (close - lower) / (upper - lower)
            features[f"{prefix}bb_pctb_{period}"] = pct_b.reindex(X.index, method="ffill")

        # Trend direction
        if len(config.ma_periods) >= 2:
            fast_ma = close.rolling(min(config.ma_periods)).mean()
            slow_ma = close.rolling(max(config.ma_periods)).mean()
            trend = (fast_ma > slow_ma).astype(np.int8)
            features[f"{prefix}trend"] = trend.reindex(X.index, method="ffill")

        return features

    def _resample_ohlcv(self, X: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample OHLCV data to a different timeframe."""
        return X.resample(timeframe).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum" if "volume" in X.columns else "first",
        }).dropna()

    def _generate_alignment_features(
        self,
        X: pd.DataFrame,
        tf_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate cross-timeframe trend alignment features."""
        features = pd.DataFrame(index=X.index)

        # Find trend columns across timeframes
        trend_cols = [c for c in tf_features.columns if c.endswith("_trend")]

        if len(trend_cols) >= 2:
            # Count how many timeframes agree on trend direction
            trend_matrix = tf_features[trend_cols]
            features["mtf_trend_alignment"] = trend_matrix.sum(axis=1)
            features["mtf_trend_unanimous"] = (
                (trend_matrix.sum(axis=1) == len(trend_cols)) |
                (trend_matrix.sum(axis=1) == 0)
            ).astype(np.int8)

        # Find RSI columns across timeframes
        rsi_cols = [c for c in tf_features.columns if "rsi" in c]
        if rsi_cols:
            features["mtf_rsi_mean"] = tf_features[rsi_cols].mean(axis=1)
            features["mtf_rsi_std"] = tf_features[rsi_cols].std(axis=1)

        return features

    def _generate_divergence_features(
        self,
        X: pd.DataFrame,
        tf_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate momentum/price divergence features across timeframes."""
        features = pd.DataFrame(index=X.index)

        # Look for RSI divergence across timeframes
        rsi_cols = sorted([c for c in tf_features.columns if "rsi" in c])

        if len(rsi_cols) >= 2:
            # Divergence: price making highs but RSI not
            for i in range(len(rsi_cols) - 1):
                for j in range(i + 1, len(rsi_cols)):
                    col_name = f"mtf_rsi_div_{i}_{j}"
                    features[col_name] = (
                        tf_features[rsi_cols[i]] - tf_features[rsi_cols[j]]
                    )

        return features

    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names."""
        return self._feature_names


def generate_multi_timeframe_features(
    df: pd.DataFrame,
    base_timeframe: str = "15min",
    higher_timeframes: List[str] = None,
) -> pd.DataFrame:
    """
    Convenience function to generate multi-timeframe features.

    Args:
        df: OHLCV DataFrame with DatetimeIndex
        base_timeframe: Base timeframe of the data
        higher_timeframes: List of higher timeframes to analyze

    Returns:
        DataFrame with multi-timeframe features
    """
    if higher_timeframes is None:
        higher_timeframes = ["1H", "4H", "1D"]

    configs = []
    for tf in higher_timeframes:
        configs.append(TimeframeConfig(
            timeframe=Timeframe(tf),
            ma_periods=[20, 50, 200],
            momentum_periods=[14],
            volatility_periods=[20],
        ))

    mtf = MultiTimeframeFeatures(
        base_timeframe=base_timeframe,
        timeframe_configs=configs,
    )

    return mtf.fit_transform(df)
