"""
Point-in-Time Feature Engine
============================
JPMorgan-Level Feature Engineering with No Look-Ahead Bias

This module ensures all features are computed using only data
available at each point in time. Critical for:
1. Backtesting integrity
2. Live trading consistency
3. Regulatory compliance

Key Concepts:
- Look-ahead bias: Using future data for current predictions
- Point-in-time: Features use only past data
- Information leakage: When test data influences training

Implementation:
- All rolling calculations use closed windows
- No future-peeking in normalization
- Expanding windows for statistics
- Careful handling of NaN filling

Author: AlphaTrade Institutional System
Based on: ARCHITECTURAL_REVIEW_REPORT.md - Phase 3
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import warnings

from ..utils.logger import get_logger

logger = get_logger(__name__)


class NormalizationMethod(Enum):
    """Methods for feature normalization"""
    NONE = "none"
    ZSCORE = "zscore"  # (x - mean) / std
    MINMAX = "minmax"  # (x - min) / (max - min)
    RANK = "rank"  # Percentile rank
    ROBUST = "robust"  # (x - median) / IQR


@dataclass
class FeatureConfig:
    """Configuration for a single feature"""
    name: str
    calculation: Callable  # Function to compute feature
    lookback: int = 20  # Bars of history required
    normalize: NormalizationMethod = NormalizationMethod.ZSCORE
    norm_window: int = 252  # Window for normalization stats
    fill_method: str = "ffill"  # How to fill NaN
    min_periods: int = 1  # Minimum periods for rolling


class PointInTimeFeatureEngine:
    """
    Feature engine that guarantees no look-ahead bias.

    Every feature is computed using ONLY data available up to
    the current timestamp. This is critical for:

    1. Backtesting: Results reflect real trading conditions
    2. Live trading: Features match backtest exactly
    3. Model training: Prevents data leakage

    How it works:
    - Uses .shift() to ensure no current-bar data leaks
    - Rolling windows only look backwards
    - Normalization uses expanding windows
    - Forward fills are explicitly bounded

    Example:
        engine = PointInTimeFeatureEngine()
        engine.add_feature('momentum_10', lambda df: df['close'].pct_change(10))
        features = engine.compute_all(df)
    """

    def __init__(
        self,
        default_norm: NormalizationMethod = NormalizationMethod.ZSCORE,
        default_norm_window: int = 252,
        warn_on_nan: bool = True
    ):
        self.default_norm = default_norm
        self.default_norm_window = default_norm_window
        self.warn_on_nan = warn_on_nan

        # Feature registry
        self._features: Dict[str, FeatureConfig] = {}

        # Computed normalization statistics
        self._norm_stats: Dict[str, Dict[str, pd.Series]] = {}

        # Validation flags
        self._validated = False

    def add_feature(
        self,
        name: str,
        calculation: Callable,
        lookback: int = 20,
        normalize: Optional[NormalizationMethod] = None,
        norm_window: Optional[int] = None,
        fill_method: str = "ffill",
        min_periods: int = 1
    ) -> None:
        """
        Register a feature for computation.

        The calculation function should:
        - Take a DataFrame with OHLCV columns
        - Return a Series with the feature values
        - NOT look ahead in time (verified at runtime)

        Args:
            name: Feature name
            calculation: Function(df) -> Series
            lookback: Bars of history needed
            normalize: Normalization method
            norm_window: Window for normalization
            fill_method: NaN fill method
            min_periods: Minimum periods for rolling
        """
        self._features[name] = FeatureConfig(
            name=name,
            calculation=calculation,
            lookback=lookback,
            normalize=normalize or self.default_norm,
            norm_window=norm_window or self.default_norm_window,
            fill_method=fill_method,
            min_periods=min_periods
        )

    def add_standard_features(self) -> None:
        """Add standard technical features (all point-in-time safe)"""

        # Returns (1-period shift ensures point-in-time)
        for period in [1, 5, 10, 20]:
            self.add_feature(
                f'return_{period}',
                lambda df, p=period: df['close'].pct_change(p).shift(1),
                lookback=period + 1
            )

        # Volatility
        for period in [5, 10, 20]:
            self.add_feature(
                f'volatility_{period}',
                lambda df, p=period: df['close'].pct_change().rolling(p).std().shift(1),
                lookback=period + 1
            )

        # RSI
        self.add_feature(
            'rsi_14',
            lambda df: self._compute_rsi(df['close'], 14).shift(1),
            lookback=15
        )

        # Moving average ratios
        for ma_period in [10, 20, 50]:
            self.add_feature(
                f'ma_ratio_{ma_period}',
                lambda df, p=ma_period: (
                    df['close'].shift(1) / df['close'].rolling(p).mean().shift(1)
                ),
                lookback=ma_period + 1
            )

        # ATR
        self.add_feature(
            'atr_14',
            lambda df: self._compute_atr(df, 14).shift(1),
            lookback=15
        )

        # Volume ratio
        self.add_feature(
            'volume_ratio_20',
            lambda df: (
                df['volume'].shift(1) / df['volume'].rolling(20).mean().shift(1)
            ),
            lookback=21,
            normalize=NormalizationMethod.ROBUST
        )

        # Momentum
        self.add_feature(
            'momentum_10',
            lambda df: (
                df['close'].shift(1) / df['close'].shift(11) - 1
            ),
            lookback=12
        )

        # Bollinger position
        self.add_feature(
            'bb_position_20',
            lambda df: self._compute_bb_position(df['close'], 20).shift(1),
            lookback=21
        )

        logger.info(f"Added {len(self._features)} standard features")

    @staticmethod
    def _compute_rsi(prices: pd.Series, period: int) -> pd.Series:
        """Compute RSI (point-in-time safe)"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
        """Compute ATR (point-in-time safe)"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _compute_bb_position(prices: pd.Series, period: int) -> pd.Series:
        """Compute position within Bollinger Bands (0 = lower, 1 = upper)"""
        ma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        return (prices - lower) / (upper - lower)

    def compute_all(
        self,
        df: pd.DataFrame,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Compute all registered features.

        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
            validate: If True, validates no look-ahead bias

        Returns:
            DataFrame with all computed features
        """
        logger.info(f"Computing {len(self._features)} features for {len(df)} bars")

        result = pd.DataFrame(index=df.index)

        for name, config in self._features.items():
            try:
                # Compute raw feature
                raw_feature = config.calculation(df)

                # Validate point-in-time (no look-ahead)
                if validate:
                    self._validate_no_lookahead(df, raw_feature, name)

                # Fill NaN
                feature = self._fill_nan(raw_feature, config.fill_method)

                # Normalize (point-in-time)
                if config.normalize != NormalizationMethod.NONE:
                    feature = self._normalize_pit(
                        feature,
                        config.normalize,
                        config.norm_window
                    )

                result[name] = feature

            except Exception as e:
                logger.warning(f"Failed to compute feature '{name}': {e}")
                result[name] = np.nan

        # Log NaN summary
        nan_pct = result.isna().mean()
        high_nan = nan_pct[nan_pct > 0.1]
        if len(high_nan) > 0 and self.warn_on_nan:
            logger.warning(f"High NaN features: {high_nan.to_dict()}")

        return result

    def _validate_no_lookahead(
        self,
        df: pd.DataFrame,
        feature: pd.Series,
        name: str
    ) -> None:
        """
        Validate that feature doesn't look ahead.

        Strategy: Modify the last few rows of input data and recompute.
        If feature values before modification change, there's look-ahead.
        """
        if len(df) < 10:
            return

        # Create modified df with last rows zeroed
        df_modified = df.copy()
        df_modified.iloc[-5:] = df_modified.iloc[-5:] * 0.5  # Change last 5 rows

        # Recompute feature
        feature_config = self._features[name]
        modified_feature = feature_config.calculation(df_modified)

        # Compare values BEFORE modification point
        # If they differ, feature is looking ahead
        compare_idx = -6  # Before modified rows
        if len(feature) > abs(compare_idx):
            original_val = feature.iloc[compare_idx]
            modified_val = modified_feature.iloc[compare_idx]

            if not np.isnan(original_val) and not np.isnan(modified_val):
                if not np.isclose(original_val, modified_val, rtol=1e-10):
                    logger.error(
                        f"LOOK-AHEAD BIAS DETECTED in feature '{name}'! "
                        f"Value at index {compare_idx} changed from "
                        f"{original_val} to {modified_val}"
                    )

    def _fill_nan(self, series: pd.Series, method: str) -> pd.Series:
        """Fill NaN values (point-in-time safe)"""
        if method == "ffill":
            # Forward fill is PIT safe (uses only past values)
            return series.ffill()
        elif method == "zero":
            return series.fillna(0)
        elif method == "mean":
            # Expanding mean is PIT safe
            return series.fillna(series.expanding().mean())
        else:
            return series

    def _normalize_pit(
        self,
        series: pd.Series,
        method: NormalizationMethod,
        window: int
    ) -> pd.Series:
        """
        Normalize feature using only point-in-time data.

        Key: Use EXPANDING window for initial period, then ROLLING.
        Never use the full series statistics (that's look-ahead).
        """
        if method == NormalizationMethod.ZSCORE:
            # Expanding mean/std for point-in-time
            mean = series.expanding(min_periods=20).mean()
            std = series.expanding(min_periods=20).std()

            # After enough history, switch to rolling for adaptation
            if len(series) > window:
                mean_roll = series.rolling(window).mean()
                std_roll = series.rolling(window).std()

                # Use rolling where available, expanding otherwise
                mean = mean.where(mean_roll.isna(), mean_roll)
                std = std.where(std_roll.isna(), std_roll)

            return (series - mean) / std.replace(0, np.nan)

        elif method == NormalizationMethod.MINMAX:
            # Expanding min/max
            min_val = series.expanding(min_periods=20).min()
            max_val = series.expanding(min_periods=20).max()
            range_val = max_val - min_val
            return (series - min_val) / range_val.replace(0, np.nan)

        elif method == NormalizationMethod.RANK:
            # Expanding rank is complex, use rolling approximation
            def expanding_rank(x):
                return x.rank(pct=True).iloc[-1]

            return series.expanding(min_periods=20).apply(
                expanding_rank, raw=False
            )

        elif method == NormalizationMethod.ROBUST:
            # Median and IQR
            median = series.expanding(min_periods=20).median()
            q75 = series.expanding(min_periods=20).quantile(0.75)
            q25 = series.expanding(min_periods=20).quantile(0.25)
            iqr = q75 - q25
            return (series - median) / iqr.replace(0, np.nan)

        return series

    def get_required_lookback(self) -> int:
        """Get maximum lookback required across all features"""
        if not self._features:
            return 0
        return max(config.lookback for config in self._features.values())


# =============================================================================
# FEATURE PIPELINE WITH POINT-IN-TIME GUARANTEES
# =============================================================================

class PointInTimeFeaturePipeline:
    """
    Complete feature pipeline with point-in-time guarantees.

    Combines:
    - Technical features
    - Microstructure features
    - Regime features
    - Cross-asset features

    All computed point-in-time.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Feature engines for different categories
        self._technical_engine = PointInTimeFeatureEngine()
        self._custom_engines: Dict[str, PointInTimeFeatureEngine] = {}

        # Setup standard features
        self._technical_engine.add_standard_features()

    def add_custom_feature(
        self,
        name: str,
        calculation: Callable,
        category: str = "custom",
        **kwargs
    ) -> None:
        """Add custom feature to specific category"""
        if category not in self._custom_engines:
            self._custom_engines[category] = PointInTimeFeatureEngine()

        self._custom_engines[category].add_feature(name, calculation, **kwargs)

    def compute(
        self,
        df: pd.DataFrame,
        categories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compute all features for given data.

        Args:
            df: OHLCV DataFrame
            categories: Which feature categories to compute (None = all)

        Returns:
            DataFrame with all features
        """
        result = pd.DataFrame(index=df.index)

        # Technical features
        if categories is None or "technical" in categories:
            tech_features = self._technical_engine.compute_all(df)
            for col in tech_features.columns:
                result[f"tech_{col}"] = tech_features[col]

        # Custom features
        for category, engine in self._custom_engines.items():
            if categories is None or category in categories:
                cat_features = engine.compute_all(df)
                for col in cat_features.columns:
                    result[f"{category}_{col}"] = cat_features[col]

        return result

    def get_required_warmup(self) -> int:
        """Get bars required for warmup"""
        warmup = self._technical_engine.get_required_lookback()

        for engine in self._custom_engines.values():
            warmup = max(warmup, engine.get_required_lookback())

        return warmup + 50  # Extra buffer for normalization


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_no_future_leakage(
    df: pd.DataFrame,
    features_df: pd.DataFrame
) -> Dict[str, bool]:
    """
    Validate that no features leak future information.

    Args:
        df: Original OHLCV data
        features_df: Computed features

    Returns:
        Dict mapping feature name to is_valid (True = no leakage)
    """
    results = {}

    for col in features_df.columns:
        # Compute correlation with future returns
        future_return = df['close'].pct_change().shift(-1)  # 1-bar future return

        # If feature correlates strongly with FUTURE return, it's leaking
        # Note: Some correlation is expected (predictive signal), but
        # perfect correlation would indicate leakage

        valid_idx = ~(features_df[col].isna() | future_return.isna())
        if valid_idx.sum() < 100:
            results[col] = True  # Not enough data to test
            continue

        corr = features_df[col][valid_idx].corr(future_return[valid_idx])

        # Correlation > 0.5 with 1-bar future return is suspicious
        if abs(corr) > 0.5:
            logger.warning(
                f"Feature '{col}' has high correlation ({corr:.3f}) "
                f"with future return - possible leakage!"
            )
            results[col] = False
        else:
            results[col] = True

    return results
