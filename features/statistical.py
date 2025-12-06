"""
Statistical Features Module
===========================

Statistical analysis and derived features for the algorithmic trading platform.
Provides features for machine learning models and quantitative analysis.

Feature Categories:
- Returns: Log returns, simple returns, cumulative returns
- Rolling Statistics: Mean, std, skewness, kurtosis
- Price Momentum: Rate of change, momentum scores
- Volatility: Realized volatility, Parkinson, Garman-Klass
- Correlation: Rolling correlation, beta
- Regime Detection: Volatility regimes, trend regimes
- Distribution: Percentiles, z-scores

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from core.enums import MarketRegime
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats as scipy_stats


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class StatisticalConfig:
    """Configuration for statistical features."""
    
    # Return periods
    RETURN_PERIODS: tuple[int, ...] = (1, 5, 10, 20, 60)
    
    # Rolling windows
    ROLLING_WINDOWS: tuple[int, ...] = (5, 10, 20, 60, 120, 252)
    
    # Volatility
    VOLATILITY_WINDOW: int = 20
    ANNUALIZATION_FACTOR: int = 252
    
    # Regime detection
    REGIME_LOOKBACK: int = 60
    REGIME_THRESHOLD: float = 1.5
    
    # Distribution
    PERCENTILE_WINDOW: int = 252


DEFAULT_STAT_CONFIG = StatisticalConfig()




# =============================================================================
# RETURNS
# =============================================================================

def log_returns(
    df: pl.DataFrame,
    periods: list[int] | None = None,
    column: str = "close",
) -> pl.DataFrame:
    """
    Calculate log returns for multiple periods.
    
    Formula:
        Log Return = ln(Price_t / Price_{t-n})
    
    Args:
        df: DataFrame with price data
        periods: List of return periods (default: [1, 5, 10, 20, 60])
        column: Price column to use
    
    Returns:
        DataFrame with log return columns
    """
    periods = periods or list(DEFAULT_STAT_CONFIG.RETURN_PERIODS)
    
    exprs = []
    for period in periods:
        col_name = f"log_return_{period}"
        exprs.append(
            (pl.col(column) / pl.col(column).shift(period)).log().alias(col_name)
        )
    
    return df.with_columns(exprs)


def simple_returns(
    df: pl.DataFrame,
    periods: list[int] | None = None,
    column: str = "close",
) -> pl.DataFrame:
    """
    Calculate simple (percentage) returns for multiple periods.
    
    Formula:
        Simple Return = (Price_t - Price_{t-n}) / Price_{t-n}
    
    Args:
        df: DataFrame with price data
        periods: List of return periods
        column: Price column to use
    
    Returns:
        DataFrame with simple return columns
    """
    periods = periods or list(DEFAULT_STAT_CONFIG.RETURN_PERIODS)
    
    exprs = []
    for period in periods:
        col_name = f"return_{period}"
        exprs.append(
            pl.col(column).pct_change(n=period).alias(col_name)
        )
    
    return df.with_columns(exprs)


def cumulative_returns(
    df: pl.DataFrame,
    windows: list[int] | None = None,
    column: str = "close",
) -> pl.DataFrame:
    """
    Calculate cumulative returns over rolling windows.
    
    Formula:
        Cumulative Return = (Price_t / Price_{t-n}) - 1
    
    Args:
        df: DataFrame with price data
        windows: List of rolling windows
        column: Price column to use
    
    Returns:
        DataFrame with cumulative return columns
    """
    windows = windows or [5, 10, 20, 60, 120, 252]
    
    exprs = []
    for window in windows:
        col_name = f"cum_return_{window}"
        exprs.append(
            (pl.col(column) / pl.col(column).shift(window) - 1.0).alias(col_name)
        )
    
    return df.with_columns(exprs)


def excess_returns(
    df: pl.DataFrame,
    benchmark_col: str = "benchmark",
    period: int = 1,
    column: str = "close",
) -> pl.DataFrame:
    """
    Calculate excess returns over a benchmark.
    
    Formula:
        Excess Return = Asset Return - Benchmark Return
    
    Args:
        df: DataFrame with price and benchmark data
        benchmark_col: Benchmark column name
        period: Return period
        column: Asset price column
    
    Returns:
        DataFrame with excess return column
    """
    return df.with_columns([
        (pl.col(column) / pl.col(column).shift(period) - 1.0).alias("_asset_ret"),
        (pl.col(benchmark_col) / pl.col(benchmark_col).shift(period) - 1.0).alias("_bench_ret"),
    ]).with_columns([
        (pl.col("_asset_ret") - pl.col("_bench_ret")).alias(f"excess_return_{period}"),
    ]).drop(["_asset_ret", "_bench_ret"])


# =============================================================================
# ROLLING STATISTICS
# =============================================================================

def rolling_stats(
    df: pl.DataFrame,
    windows: list[int] | None = None,
    column: str = "close",
) -> pl.DataFrame:
    """
    Calculate rolling statistical measures.
    
    Computes:
        - Rolling mean
        - Rolling standard deviation
        - Rolling min/max
        - Rolling range
    
    Args:
        df: DataFrame with price data
        windows: List of rolling windows
        column: Column to analyze
    
    Returns:
        DataFrame with rolling statistics
    """
    windows = windows or list(DEFAULT_STAT_CONFIG.ROLLING_WINDOWS)
    
    for window in windows:
        df = df.with_columns([
            pl.col(column).rolling_mean(window_size=window).alias(f"rolling_mean_{window}"),
            pl.col(column).rolling_std(window_size=window).alias(f"rolling_std_{window}"),
            pl.col(column).rolling_min(window_size=window).alias(f"rolling_min_{window}"),
            pl.col(column).rolling_max(window_size=window).alias(f"rolling_max_{window}"),
        ]).with_columns([
            (
                (pl.col(f"rolling_max_{window}") - pl.col(f"rolling_min_{window}")) /
                (pl.col(f"rolling_min_{window}") + 1e-10)
            ).alias(f"rolling_range_pct_{window}"),
        ])
    
    return df


def rolling_higher_moments(
    df: pl.DataFrame,
    window: int = 60,
    column: str = "close",
) -> pl.DataFrame:
    """
    Calculate rolling skewness and kurtosis.
    
    Args:
        df: DataFrame with price data
        window: Rolling window size
        column: Column to analyze
    
    Returns:
        DataFrame with skewness and kurtosis
    """
    # First calculate returns
    returns = df[column].pct_change().to_numpy()
    n = len(returns)
    
    skewness = np.full(n, np.nan)
    kurtosis = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        window_returns = returns[i - window + 1:i + 1]
        valid = window_returns[~np.isnan(window_returns)]
        if len(valid) >= 3:
            skewness[i] = scipy_stats.skew(valid)
            kurtosis[i] = scipy_stats.kurtosis(valid)
    
    return df.with_columns([
        pl.Series(f"rolling_skewness_{window}", skewness),
        pl.Series(f"rolling_kurtosis_{window}", kurtosis),
    ])


def rolling_quantiles(
    df: pl.DataFrame,
    window: int = 60,
    quantiles: list[float] | None = None,
    column: str = "close",
) -> pl.DataFrame:
    """
    Calculate rolling quantiles.
    
    Args:
        df: DataFrame with price data
        window: Rolling window size
        quantiles: Quantile levels (default: [0.1, 0.25, 0.5, 0.75, 0.9])
        column: Column to analyze
    
    Returns:
        DataFrame with rolling quantiles
    """
    quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]
    
    for q in quantiles:
        col_name = f"rolling_q{int(q*100)}_{window}"
        df = df.with_columns([
            pl.col(column)
            .rolling_quantile(quantile=q, window_size=window)
            .alias(col_name),
        ])
    
    return df


# =============================================================================
# PRICE MOMENTUM FEATURES
# =============================================================================

def price_momentum(
    df: pl.DataFrame,
    periods: list[int] | None = None,
    column: str = "close",
) -> pl.DataFrame:
    """
    Calculate price momentum features.
    
    Features:
        - Momentum (price change)
        - Rate of change
        - Price position relative to range
    
    Args:
        df: DataFrame with price data
        periods: Momentum periods
        column: Price column
    
    Returns:
        DataFrame with momentum features
    """
    periods = periods or [5, 10, 20, 60]
    
    for period in periods:
        df = df.with_columns([
            # Simple momentum
            (pl.col(column) - pl.col(column).shift(period)).alias(f"momentum_{period}"),
            # Rate of change
            (
                (pl.col(column) - pl.col(column).shift(period)) /
                (pl.col(column).shift(period) + 1e-10) * 100.0
            ).alias(f"roc_{period}"),
            # Price position (where in the range is current price)
            (
                (pl.col(column) - pl.col(column).rolling_min(window_size=period)) /
                (
                    pl.col(column).rolling_max(window_size=period) -
                    pl.col(column).rolling_min(window_size=period) + 1e-10
                )
            ).alias(f"price_position_{period}"),
        ])
    
    return df


def trend_strength(
    df: pl.DataFrame,
    windows: list[int] | None = None,
    column: str = "close",
) -> pl.DataFrame:
    """
    Calculate trend strength indicators.
    
    Features:
        - Linear regression slope
        - R-squared
        - Price vs MA ratio
    
    Args:
        df: DataFrame with price data
        windows: Analysis windows
        column: Price column
    
    Returns:
        DataFrame with trend strength features
    """
    windows = windows or [20, 60]
    
    for window in windows:
        # Price vs SMA ratio
        df = df.with_columns([
            (
                pl.col(column) /
                pl.col(column).rolling_mean(window_size=window)
            ).alias(f"price_sma_ratio_{window}"),
        ])
        
        # Linear regression slope (normalized)
        closes = df[column].to_numpy()
        n = len(closes)
        slopes = np.full(n, np.nan)
        r_squared = np.full(n, np.nan)
        
        x = np.arange(window)
        for i in range(window - 1, n):
            y = closes[i - window + 1:i + 1]
            if not np.any(np.isnan(y)):
                slope, intercept, r, p, se = scipy_stats.linregress(x, y)
                # Normalize slope by mean price
                slopes[i] = slope / (np.mean(y) + 1e-10) * window
                r_squared[i] = r ** 2
        
        df = df.with_columns([
            pl.Series(f"trend_slope_{window}", slopes),
            pl.Series(f"trend_r2_{window}", r_squared),
        ])
    
    return df


def price_acceleration(
    df: pl.DataFrame,
    period: int = 20,
    column: str = "close",
) -> pl.DataFrame:
    """
    Calculate price acceleration (second derivative).
    
    Formula:
        Velocity = Price Change
        Acceleration = Velocity Change
    
    Args:
        df: DataFrame with price data
        period: Calculation period
        column: Price column
    
    Returns:
        DataFrame with acceleration features
    """
    return df.with_columns([
        pl.col(column).diff().alias("_velocity"),
    ]).with_columns([
        pl.col("_velocity").diff().alias(f"price_acceleration_{period}"),
        # Smoothed acceleration
        pl.col("_velocity").diff().rolling_mean(window_size=period).alias(f"smooth_acceleration_{period}"),
    ]).drop(["_velocity"])


# =============================================================================
# VOLATILITY FEATURES
# =============================================================================

def volatility_features(
    df: pl.DataFrame,
    windows: list[int] | None = None,
    annualize: bool = True,
) -> pl.DataFrame:
    """
    Calculate comprehensive volatility features.
    
    Features:
        - Realized volatility (close-to-close)
        - Parkinson volatility (high-low)
        - Garman-Klass volatility
        - Yang-Zhang volatility
        - Volatility ratio
    
    Args:
        df: DataFrame with OHLC data
        windows: Volatility windows
        annualize: Annualize volatility (sqrt(252))
    
    Returns:
        DataFrame with volatility features
    """
    windows = windows or [5, 10, 20, 60]
    ann_factor = np.sqrt(252) if annualize else 1.0
    
    # Pre-calculate log returns
    df = df.with_columns([
        (pl.col("close") / pl.col("close").shift(1)).log().alias("_log_return"),
        (pl.col("high") / pl.col("low")).log().alias("_log_hl"),
        (pl.col("close") / pl.col("open")).log().alias("_log_co"),
        (pl.col("open") / pl.col("close").shift(1)).log().alias("_log_oc"),
    ])
    
    for window in windows:
        # Realized volatility (standard)
        df = df.with_columns([
            (
                pl.col("_log_return").rolling_std(window_size=window) * ann_factor
            ).alias(f"realized_vol_{window}"),
        ])
        
        # Parkinson volatility (using high-low range)
        # More efficient estimator than close-to-close
        df = df.with_columns([
            (
                (pl.col("_log_hl").pow(2) / (4 * np.log(2)))
                .rolling_mean(window_size=window)
                .sqrt() * ann_factor
            ).alias(f"parkinson_vol_{window}"),
        ])
        
        # Garman-Klass volatility
        # Uses open, high, low, close
        df = df.with_columns([
            (
                (
                    0.5 * pl.col("_log_hl").pow(2) -
                    (2 * np.log(2) - 1) * pl.col("_log_co").pow(2)
                )
                .rolling_mean(window_size=window)
                .sqrt() * ann_factor
            ).alias(f"garman_klass_vol_{window}"),
        ])
    
    # Clean up temporary columns
    df = df.drop(["_log_return", "_log_hl", "_log_co", "_log_oc"])
    
    return df


def volatility_regime(
    df: pl.DataFrame,
    lookback: int = 60,
    short_window: int = 10,
    long_window: int = 60,
) -> pl.DataFrame:
    """
    Detect volatility regime (high/low volatility).
    
    Args:
        df: DataFrame with price data
        lookback: Lookback for percentile calculation
        short_window: Short-term volatility window
        long_window: Long-term volatility window
    
    Returns:
        DataFrame with volatility regime features
    """
    # Calculate short and long-term volatility
    df = df.with_columns([
        (pl.col("close") / pl.col("close").shift(1)).log().alias("_ret"),
    ]).with_columns([
        pl.col("_ret").rolling_std(window_size=short_window).alias("_vol_short"),
        pl.col("_ret").rolling_std(window_size=long_window).alias("_vol_long"),
    ])
    
    # Volatility ratio
    df = df.with_columns([
        (pl.col("_vol_short") / (pl.col("_vol_long") + 1e-10)).alias("vol_ratio"),
    ])
    
    # Volatility percentile
    vol_values = df["_vol_short"].to_numpy()
    n = len(vol_values)
    vol_percentile = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        window = vol_values[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            vol_percentile[i] = scipy_stats.percentileofscore(valid, vol_values[i]) / 100.0
    
    df = df.with_columns([
        pl.Series("vol_percentile", vol_percentile),
    ])
    
    # Regime classification
    df = df.with_columns([
        pl.when(pl.col("vol_percentile") > 0.8)
        .then(pl.lit("high"))
        .when(pl.col("vol_percentile") < 0.2)
        .then(pl.lit("low"))
        .otherwise(pl.lit("normal"))
        .alias("vol_regime"),
    ]).drop(["_ret", "_vol_short", "_vol_long"])
    
    return df


def intraday_volatility(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Calculate intraday volatility measures.
    
    Features:
        - True range
        - Bar range
        - Body-to-range ratio
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        DataFrame with intraday volatility features
    """
    return df.with_columns([
        # True range
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs(),
        ).alias("true_range"),
        # Bar range
        (pl.col("high") - pl.col("low")).alias("bar_range"),
        # Body
        (pl.col("close") - pl.col("open")).abs().alias("bar_body"),
    ]).with_columns([
        # Body to range ratio (candle pattern indicator)
        (pl.col("bar_body") / (pl.col("bar_range") + 1e-10)).alias("body_range_ratio"),
        # Range as percentage of price
        (pl.col("bar_range") / pl.col("close") * 100).alias("range_pct"),
    ])


# =============================================================================
# CORRELATION & BETA
# =============================================================================

def rolling_correlation(
    df: pl.DataFrame,
    column1: str,
    column2: str,
    windows: list[int] | None = None,
) -> pl.DataFrame:
    """
    Calculate rolling correlation between two series.
    
    Args:
        df: DataFrame with both columns
        column1: First column name
        column2: Second column name
        windows: Rolling windows
    
    Returns:
        DataFrame with correlation columns
    """
    windows = windows or [20, 60, 120]
    
    # Calculate returns for correlation
    df = df.with_columns([
        pl.col(column1).pct_change().alias("_ret1"),
        pl.col(column2).pct_change().alias("_ret2"),
    ])
    
    for window in windows:
        col_name = f"corr_{column1}_{column2}_{window}"
        
        # Calculate rolling correlation using numpy
        ret1 = df["_ret1"].to_numpy()
        ret2 = df["_ret2"].to_numpy()
        n = len(ret1)
        correlations = np.full(n, np.nan)
        
        for i in range(window - 1, n):
            r1 = ret1[i - window + 1:i + 1]
            r2 = ret2[i - window + 1:i + 1]
            valid_mask = ~(np.isnan(r1) | np.isnan(r2))
            if valid_mask.sum() >= 3:
                correlations[i] = np.corrcoef(r1[valid_mask], r2[valid_mask])[0, 1]
        
        df = df.with_columns([
            pl.Series(col_name, correlations),
        ])
    
    return df.drop(["_ret1", "_ret2"])


def rolling_beta(
    df: pl.DataFrame,
    asset_col: str = "close",
    benchmark_col: str = "benchmark",
    windows: list[int] | None = None,
) -> pl.DataFrame:
    """
    Calculate rolling beta to a benchmark.
    
    Formula:
        Beta = Cov(Asset, Benchmark) / Var(Benchmark)
    
    Args:
        df: DataFrame with asset and benchmark data
        asset_col: Asset price column
        benchmark_col: Benchmark price column
        windows: Rolling windows
    
    Returns:
        DataFrame with beta columns
    """
    windows = windows or [60, 120, 252]
    
    # Check if benchmark column exists
    if benchmark_col not in df.columns:
        # Return with null beta columns
        for window in windows:
            df = df.with_columns([
                pl.lit(None).alias(f"beta_{window}"),
            ])
        return df
    
    # Calculate returns
    df = df.with_columns([
        pl.col(asset_col).pct_change().alias("_asset_ret"),
        pl.col(benchmark_col).pct_change().alias("_bench_ret"),
    ])
    
    asset_ret = df["_asset_ret"].to_numpy()
    bench_ret = df["_bench_ret"].to_numpy()
    n = len(asset_ret)
    
    for window in windows:
        betas = np.full(n, np.nan)
        
        for i in range(window - 1, n):
            ar = asset_ret[i - window + 1:i + 1]
            br = bench_ret[i - window + 1:i + 1]
            valid_mask = ~(np.isnan(ar) | np.isnan(br))
            
            if valid_mask.sum() >= 10:
                cov = np.cov(ar[valid_mask], br[valid_mask])[0, 1]
                var = np.var(br[valid_mask])
                if var > 1e-10:
                    betas[i] = cov / var
        
        df = df.with_columns([
            pl.Series(f"beta_{window}", betas),
        ])
    
    return df.drop(["_asset_ret", "_bench_ret"])


# =============================================================================
# REGIME DETECTION
# =============================================================================

def detect_market_regime(
    df: pl.DataFrame,
    trend_window: int = 20,
    vol_window: int = 20,
    vol_lookback: int = 252,
) -> pl.DataFrame:
    """
    Detect overall market regime.
    
    Regimes:
        - High volatility uptrend
        - High volatility downtrend
        - Low volatility uptrend
        - Low volatility downtrend
        - Ranging
    
    Args:
        df: DataFrame with OHLC data
        trend_window: Window for trend detection
        vol_window: Window for volatility calculation
        vol_lookback: Lookback for volatility percentile
    
    Returns:
        DataFrame with regime column
    """
    # Calculate trend indicator
    df = df.with_columns([
        (
            pl.col("close") / pl.col("close").rolling_mean(window_size=trend_window) - 1
        ).alias("_trend"),
    ])
    
    # Calculate volatility
    df = df.with_columns([
        (pl.col("close") / pl.col("close").shift(1)).log().alias("_ret"),
    ]).with_columns([
        pl.col("_ret").rolling_std(window_size=vol_window).alias("_vol"),
    ])
    
    # Calculate volatility percentile
    vol = df["_vol"].to_numpy()
    trend = df["_trend"].to_numpy()
    n = len(vol)
    
    vol_pct = np.full(n, np.nan)
    for i in range(vol_lookback - 1, n):
        window = vol[i - vol_lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            vol_pct[i] = scipy_stats.percentileofscore(valid, vol[i]) / 100.0
    
    # Classify regime
    regimes = []
    for i in range(n):
        if np.isnan(vol_pct[i]) or np.isnan(trend[i]):
            regimes.append("unknown")
        elif vol_pct[i] > 0.7:  # High volatility
            if trend[i] > 0.01:
                regimes.append(MarketRegime.HIGH_VOL_UPTREND.value)
            elif trend[i] < -0.01:
                regimes.append(MarketRegime.HIGH_VOL_DOWNTREND.value)
            else:
                regimes.append(MarketRegime.RANGING.value)
        elif vol_pct[i] < 0.3:  # Low volatility
            if trend[i] > 0.01:
                regimes.append(MarketRegime.LOW_VOL_UPTREND.value)
            elif trend[i] < -0.01:
                regimes.append(MarketRegime.LOW_VOL_DOWNTREND.value)
            else:
                regimes.append(MarketRegime.RANGING.value)
        else:  # Normal volatility
            if abs(trend[i]) < 0.005:
                regimes.append(MarketRegime.RANGING.value)
            elif trend[i] > 0:
                regimes.append(MarketRegime.LOW_VOL_UPTREND.value)
            else:
                regimes.append(MarketRegime.LOW_VOL_DOWNTREND.value)
    
    df = df.with_columns([
        pl.Series("market_regime", regimes),
        pl.Series("vol_percentile_regime", vol_pct),
    ]).drop(["_trend", "_ret", "_vol"])
    
    return df


def trend_regime(
    df: pl.DataFrame,
    short_window: int = 10,
    long_window: int = 50,
) -> pl.DataFrame:
    """
    Detect trend regime based on moving average alignment.
    
    Args:
        df: DataFrame with price data
        short_window: Short MA window
        long_window: Long MA window
    
    Returns:
        DataFrame with trend regime features
    """
    return df.with_columns([
        pl.col("close").rolling_mean(window_size=short_window).alias("_ma_short"),
        pl.col("close").rolling_mean(window_size=long_window).alias("_ma_long"),
    ]).with_columns([
        # Trend direction
        pl.when(pl.col("_ma_short") > pl.col("_ma_long"))
        .then(1)
        .when(pl.col("_ma_short") < pl.col("_ma_long"))
        .then(-1)
        .otherwise(0)
        .alias("trend_direction"),
        # Trend strength (distance from long MA)
        (
            (pl.col("_ma_short") - pl.col("_ma_long")) /
            (pl.col("_ma_long") + 1e-10)
        ).alias("trend_strength"),
        # Price vs MAs
        (pl.col("close") > pl.col("_ma_short")).cast(pl.Int8).alias("above_short_ma"),
        (pl.col("close") > pl.col("_ma_long")).cast(pl.Int8).alias("above_long_ma"),
    ]).drop(["_ma_short", "_ma_long"])


# =============================================================================
# DISTRIBUTION FEATURES
# =============================================================================

def zscore(
    df: pl.DataFrame,
    window: int = 60,
    column: str = "close",
) -> pl.DataFrame:
    """
    Calculate rolling z-score.
    
    Formula:
        Z-Score = (Value - Rolling Mean) / Rolling Std
    
    Args:
        df: DataFrame with data
        window: Rolling window
        column: Column to analyze
    
    Returns:
        DataFrame with z-score column
    """
    col_name = f"zscore_{window}"
    
    return df.with_columns([
        (
            (pl.col(column) - pl.col(column).rolling_mean(window_size=window)) /
            (pl.col(column).rolling_std(window_size=window) + 1e-10)
        ).alias(col_name),
    ])


def percentile_rank(
    df: pl.DataFrame,
    window: int = 252,
    column: str = "close",
) -> pl.DataFrame:
    """
    Calculate rolling percentile rank.
    
    Args:
        df: DataFrame with data
        window: Rolling window
        column: Column to analyze
    
    Returns:
        DataFrame with percentile rank column
    """
    col_name = f"percentile_{window}"
    
    values = df[column].to_numpy()
    n = len(values)
    percentiles = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        window_vals = values[i - window + 1:i + 1]
        valid = window_vals[~np.isnan(window_vals)]
        if len(valid) > 0:
            percentiles[i] = scipy_stats.percentileofscore(valid, values[i]) / 100.0
    
    return df.with_columns([
        pl.Series(col_name, percentiles),
    ])


def distance_from_extremes(
    df: pl.DataFrame,
    window: int = 252,
    column: str = "close",
) -> pl.DataFrame:
    """
    Calculate distance from rolling high/low.
    
    Args:
        df: DataFrame with data
        window: Rolling window
        column: Column to analyze
    
    Returns:
        DataFrame with distance features
    """
    return df.with_columns([
        pl.col(column).rolling_max(window_size=window).alias("_rolling_high"),
        pl.col(column).rolling_min(window_size=window).alias("_rolling_low"),
    ]).with_columns([
        # Distance from high (drawdown)
        (
            (pl.col(column) - pl.col("_rolling_high")) /
            (pl.col("_rolling_high") + 1e-10)
        ).alias(f"dist_from_high_{window}"),
        # Distance from low
        (
            (pl.col(column) - pl.col("_rolling_low")) /
            (pl.col("_rolling_low") + 1e-10)
        ).alias(f"dist_from_low_{window}"),
        # Position in range
        (
            (pl.col(column) - pl.col("_rolling_low")) /
            (pl.col("_rolling_high") - pl.col("_rolling_low") + 1e-10)
        ).alias(f"range_position_{window}"),
    ]).drop(["_rolling_high", "_rolling_low"])


# =============================================================================
# MEAN REVERSION FEATURES
# =============================================================================

def mean_reversion_features(
    df: pl.DataFrame,
    windows: list[int] | None = None,
    column: str = "close",
) -> pl.DataFrame:
    """
    Calculate mean reversion indicators.
    
    Features:
        - Distance from MA
        - Mean reversion score
        - Deviation percentile
    
    Args:
        df: DataFrame with price data
        windows: Analysis windows
        column: Price column
    
    Returns:
        DataFrame with mean reversion features
    """
    windows = windows or [20, 60, 120]
    
    for window in windows:
        df = df.with_columns([
            pl.col(column).rolling_mean(window_size=window).alias(f"_ma_{window}"),
            pl.col(column).rolling_std(window_size=window).alias(f"_std_{window}"),
        ]).with_columns([
            # Distance from MA
            (
                (pl.col(column) - pl.col(f"_ma_{window}")) /
                (pl.col(f"_ma_{window}") + 1e-10)
            ).alias(f"ma_distance_{window}"),
            # Z-score deviation
            (
                (pl.col(column) - pl.col(f"_ma_{window}")) /
                (pl.col(f"_std_{window}") + 1e-10)
            ).alias(f"ma_zscore_{window}"),
        ]).drop([f"_ma_{window}", f"_std_{window}"])
    
    return df


def half_life(
    df: pl.DataFrame,
    window: int = 60,
    column: str = "close",
) -> pl.DataFrame:
    """
    Estimate mean reversion half-life using OLS.
    
    Uses Ornstein-Uhlenbeck process assumption.
    
    Args:
        df: DataFrame with price data
        window: Rolling window for estimation
        column: Price column
    
    Returns:
        DataFrame with half-life estimate
    """
    prices = df[column].to_numpy()
    n = len(prices)
    half_lives = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        p = prices[i - window + 1:i + 1]
        if np.any(np.isnan(p)):
            continue
        
        # Prepare data for regression
        y = np.diff(p)  # Price changes
        x = p[:-1]  # Lagged prices
        
        # OLS regression: y = alpha + beta * x
        x_with_const = np.column_stack([np.ones(len(x)), x])
        try:
            beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0][1]
            if beta < 0:
                half_lives[i] = -np.log(2) / beta
        except Exception:
            pass
    
    return df.with_columns([
        pl.Series(f"half_life_{window}", half_lives),
    ])


# =============================================================================
# STATISTICAL FEATURES CLASS
# =============================================================================

class StatisticalFeatures:
    """
    Comprehensive statistical features generator.
    
    Example:
        sf = StatisticalFeatures()
        df = sf.add_all_features(df)
    """
    
    def __init__(self, config: StatisticalConfig | None = None):
        """
        Initialize with configuration.
        
        Args:
            config: Statistical feature configuration
        """
        self.config = config or DEFAULT_STAT_CONFIG
    
    def add_return_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add return-based features."""
        df = log_returns(df, list(self.config.RETURN_PERIODS))
        df = simple_returns(df, list(self.config.RETURN_PERIODS))
        df = cumulative_returns(df)
        return df
    
    def add_rolling_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add rolling statistical features."""
        df = rolling_stats(df, list(self.config.ROLLING_WINDOWS))
        df = rolling_higher_moments(df)
        return df
    
    def add_momentum_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add momentum features."""
        df = price_momentum(df)
        df = trend_strength(df)
        df = price_acceleration(df)
        return df
    
    def add_volatility_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add volatility features."""
        df = volatility_features(df)
        df = volatility_regime(df)
        df = intraday_volatility(df)
        return df
    
    def add_regime_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add regime detection features."""
        df = detect_market_regime(df)
        df = trend_regime(df)
        return df
    
    def add_distribution_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add distribution features."""
        df = zscore(df)
        df = percentile_rank(df)
        df = distance_from_extremes(df)
        return df
    
    def add_mean_reversion_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add mean reversion features."""
        df = mean_reversion_features(df)
        df = half_life(df)
        return df
    
    def add_all_features(
        self,
        df: pl.DataFrame,
        categories: list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Add all statistical features.
        
        Args:
            df: Input DataFrame
            categories: Feature categories to include
        
        Returns:
            DataFrame with all features
        """
        all_categories = [
            "returns", "rolling", "momentum",
            "volatility", "regime", "distribution", "mean_reversion"
        ]
        categories = categories or all_categories
        
        if "returns" in categories:
            df = self.add_return_features(df)
        if "rolling" in categories:
            df = self.add_rolling_features(df)
        if "momentum" in categories:
            df = self.add_momentum_features(df)
        if "volatility" in categories:
            df = self.add_volatility_features(df)
        if "regime" in categories:
            df = self.add_regime_features(df)
        if "distribution" in categories:
            df = self.add_distribution_features(df)
        if "mean_reversion" in categories:
            df = self.add_mean_reversion_features(df)
        
        return df
    
    def get_feature_names(self) -> list[str]:
        """Get list of all feature names."""
        features = []
        
        # Return features
        for p in self.config.RETURN_PERIODS:
            features.extend([f"log_return_{p}", f"return_{p}"])
        for w in [5, 10, 20, 60, 120, 252]:
            features.append(f"cum_return_{w}")
        
        # Rolling features
        for w in self.config.ROLLING_WINDOWS:
            features.extend([
                f"rolling_mean_{w}", f"rolling_std_{w}",
                f"rolling_min_{w}", f"rolling_max_{w}",
                f"rolling_range_pct_{w}"
            ])
        features.extend(["rolling_skewness_60", "rolling_kurtosis_60"])
        
        # Momentum features
        for p in [5, 10, 20, 60]:
            features.extend([
                f"momentum_{p}", f"roc_{p}", f"price_position_{p}"
            ])
        for w in [20, 60]:
            features.extend([
                f"price_sma_ratio_{w}", f"trend_slope_{w}", f"trend_r2_{w}"
            ])
        features.extend(["price_acceleration_20", "smooth_acceleration_20"])
        
        # Volatility features
        for w in [5, 10, 20, 60]:
            features.extend([
                f"realized_vol_{w}", f"parkinson_vol_{w}", f"garman_klass_vol_{w}"
            ])
        features.extend([
            "vol_ratio", "vol_percentile", "vol_regime",
            "true_range", "bar_range", "bar_body",
            "body_range_ratio", "range_pct"
        ])
        
        # Regime features
        features.extend([
            "market_regime", "vol_percentile_regime",
            "trend_direction", "trend_strength",
            "above_short_ma", "above_long_ma"
        ])
        
        # Distribution features
        features.extend([
            "zscore_60", "percentile_252",
            "dist_from_high_252", "dist_from_low_252", "range_position_252"
        ])
        
        # Mean reversion features
        for w in [20, 60, 120]:
            features.extend([f"ma_distance_{w}", f"ma_zscore_{w}"])
        features.append("half_life_60")
        
        return features


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def add_statistical_features(
    df: pl.DataFrame,
    config: StatisticalConfig | None = None,
) -> pl.DataFrame:
    """
    Convenience function to add all statistical features.
    
    Args:
        df: Input DataFrame with OHLCV data
        config: Feature configuration
    
    Returns:
        DataFrame with all statistical features
    """
    sf = StatisticalFeatures(config)
    return sf.add_all_features(df)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "StatisticalConfig",
    "DEFAULT_STAT_CONFIG",
    "MarketRegime",
    # Returns
    "log_returns",
    "simple_returns",
    "cumulative_returns",
    "excess_returns",
    # Rolling statistics
    "rolling_stats",
    "rolling_higher_moments",
    "rolling_quantiles",
    # Momentum
    "price_momentum",
    "trend_strength",
    "price_acceleration",
    # Volatility
    "volatility_features",
    "volatility_regime",
    "intraday_volatility",
    # Correlation
    "rolling_correlation",
    "rolling_beta",
    # Regime
    "detect_market_regime",
    "trend_regime",
    # Distribution
    "zscore",
    "percentile_rank",
    "distance_from_extremes",
    # Mean reversion
    "mean_reversion_features",
    "half_life",
    # Class
    "StatisticalFeatures",
    "add_statistical_features",
]