"""
Technical Indicators Module
===========================

Comprehensive technical analysis library with 50+ indicators.
Optimized for high-performance using Polars expressions.

Indicator Categories:
- Momentum: RSI, MACD, Stochastic, Williams %R, ROC, CCI, CMO, Ultimate Oscillator
- Trend: SMA, EMA, WMA, DEMA, TEMA, ADX, Supertrend, Parabolic SAR, Aroon
- Volatility: Bollinger Bands, ATR, Keltner Channels, Donchian Channels
- Volume: OBV, VWAP, MFI, A/D Line, CMF, Force Index, VWMA

Design Principles:
- All functions are pure (no side effects)
- Polars-native for vectorized performance
- Consistent naming conventions
- Comprehensive docstrings
- Type hints throughout

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class IndicatorConfig:
    """Default configuration for technical indicators."""
    
    # Momentum
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    STOCH_K_PERIOD: int = 14
    STOCH_D_PERIOD: int = 3
    WILLIAMS_PERIOD: int = 14
    ROC_PERIOD: int = 12
    CCI_PERIOD: int = 20
    CMO_PERIOD: int = 14
    
    # Trend
    SMA_PERIODS: tuple[int, ...] = (10, 20, 50, 100, 200)
    EMA_PERIODS: tuple[int, ...] = (12, 26, 50, 100, 200)
    ADX_PERIOD: int = 14
    SUPERTREND_PERIOD: int = 10
    SUPERTREND_MULTIPLIER: float = 3.0
    AROON_PERIOD: int = 25
    
    # Volatility
    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    ATR_PERIOD: int = 14
    KELTNER_PERIOD: int = 20
    KELTNER_MULTIPLIER: float = 2.0
    DONCHIAN_PERIOD: int = 20
    
    # Volume
    MFI_PERIOD: int = 14
    CMF_PERIOD: int = 20
    FORCE_PERIOD: int = 13


DEFAULT_CONFIG = IndicatorConfig()


# =============================================================================
# MOMENTUM INDICATORS
# =============================================================================

def rsi(
    df: pl.DataFrame,
    period: int = DEFAULT_CONFIG.RSI_PERIOD,
    column: str = "close",
) -> pl.DataFrame:
    """
    Relative Strength Index (RSI).
    
    Measures the speed and magnitude of recent price changes
    to evaluate overbought or oversold conditions.
    
    Formula:
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
    
    Args:
        df: DataFrame with price data
        period: RSI period (default: 14)
        column: Price column to use
    
    Returns:
        DataFrame with RSI column added
    
    Interpretation:
        - RSI > 70: Overbought (potential sell signal)
        - RSI < 30: Oversold (potential buy signal)
        - RSI = 50: Neutral
    """
    col_name = f"rsi_{period}"
    
    return df.with_columns([
        pl.col(column).diff().alias("_delta"),
    ]).with_columns([
        pl.when(pl.col("_delta") > 0)
        .then(pl.col("_delta"))
        .otherwise(0.0)
        .alias("_gain"),
        pl.when(pl.col("_delta") < 0)
        .then(pl.col("_delta").abs())
        .otherwise(0.0)
        .alias("_loss"),
    ]).with_columns([
        pl.col("_gain")
        .rolling_mean(window_size=period)
        .alias("_avg_gain"),
        pl.col("_loss")
        .rolling_mean(window_size=period)
        .alias("_avg_loss"),
    ]).with_columns([
        pl.when(pl.col("_avg_loss") == 0)
        .then(100.0)
        .otherwise(
            100.0 - (100.0 / (1.0 + pl.col("_avg_gain") / pl.col("_avg_loss")))
        )
        .alias(col_name),
    ]).drop(["_delta", "_gain", "_loss", "_avg_gain", "_avg_loss"])


def macd(
    df: pl.DataFrame,
    fast_period: int = DEFAULT_CONFIG.MACD_FAST,
    slow_period: int = DEFAULT_CONFIG.MACD_SLOW,
    signal_period: int = DEFAULT_CONFIG.MACD_SIGNAL,
    column: str = "close",
) -> pl.DataFrame:
    """
    Moving Average Convergence Divergence (MACD).
    
    Trend-following momentum indicator showing relationship
    between two exponential moving averages.
    
    Formula:
        MACD Line = EMA(fast) - EMA(slow)
        Signal Line = EMA(MACD Line, signal_period)
        Histogram = MACD Line - Signal Line
    
    Args:
        df: DataFrame with price data
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)
        column: Price column to use
    
    Returns:
        DataFrame with macd, macd_signal, macd_histogram columns
    
    Interpretation:
        - MACD crosses above signal: Bullish
        - MACD crosses below signal: Bearish
        - Histogram expansion: Trend strengthening
    """
    return df.with_columns([
        pl.col(column)
        .ewm_mean(span=fast_period, adjust=False)
        .alias("_ema_fast"),
        pl.col(column)
        .ewm_mean(span=slow_period, adjust=False)
        .alias("_ema_slow"),
    ]).with_columns([
        (pl.col("_ema_fast") - pl.col("_ema_slow")).alias("macd"),
    ]).with_columns([
        pl.col("macd")
        .ewm_mean(span=signal_period, adjust=False)
        .alias("macd_signal"),
    ]).with_columns([
        (pl.col("macd") - pl.col("macd_signal")).alias("macd_histogram"),
    ]).drop(["_ema_fast", "_ema_slow"])


def stochastic(
    df: pl.DataFrame,
    k_period: int = DEFAULT_CONFIG.STOCH_K_PERIOD,
    d_period: int = DEFAULT_CONFIG.STOCH_D_PERIOD,
    smooth_k: int = 3,
) -> pl.DataFrame:
    """
    Stochastic Oscillator.
    
    Momentum indicator comparing closing price to price range
    over a given period.
    
    Formula:
        %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
        %D = SMA(%K, d_period)
    
    Args:
        df: DataFrame with OHLC data
        k_period: %K lookback period (default: 14)
        d_period: %D smoothing period (default: 3)
        smooth_k: %K smoothing period (default: 3)
    
    Returns:
        DataFrame with stoch_k and stoch_d columns
    
    Interpretation:
        - %K > 80: Overbought
        - %K < 20: Oversold
        - %K crosses above %D: Bullish
    """
    return df.with_columns([
        pl.col("low").rolling_min(window_size=k_period).alias("_lowest_low"),
        pl.col("high").rolling_max(window_size=k_period).alias("_highest_high"),
    ]).with_columns([
        (
            100.0 * (pl.col("close") - pl.col("_lowest_low")) /
            (pl.col("_highest_high") - pl.col("_lowest_low") + 1e-10)
        )
        .rolling_mean(window_size=smooth_k)
        .alias("stoch_k"),
    ]).with_columns([
        pl.col("stoch_k")
        .rolling_mean(window_size=d_period)
        .alias("stoch_d"),
    ]).drop(["_lowest_low", "_highest_high"])


def williams_r(
    df: pl.DataFrame,
    period: int = DEFAULT_CONFIG.WILLIAMS_PERIOD,
) -> pl.DataFrame:
    """
    Williams %R.
    
    Momentum indicator measuring overbought/oversold levels,
    similar to Stochastic but inverted scale.
    
    Formula:
        %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    
    Args:
        df: DataFrame with OHLC data
        period: Lookback period (default: 14)
    
    Returns:
        DataFrame with williams_r column
    
    Interpretation:
        - %R > -20: Overbought
        - %R < -80: Oversold
    """
    col_name = f"williams_r_{period}"
    
    return df.with_columns([
        pl.col("high").rolling_max(window_size=period).alias("_hh"),
        pl.col("low").rolling_min(window_size=period).alias("_ll"),
    ]).with_columns([
        (
            (pl.col("_hh") - pl.col("close")) /
            (pl.col("_hh") - pl.col("_ll") + 1e-10) * -100.0
        ).alias(col_name),
    ]).drop(["_hh", "_ll"])


def roc(
    df: pl.DataFrame,
    period: int = DEFAULT_CONFIG.ROC_PERIOD,
    column: str = "close",
) -> pl.DataFrame:
    """
    Rate of Change (ROC).
    
    Measures percentage change in price over a specified period.
    
    Formula:
        ROC = ((Close - Close[n]) / Close[n]) * 100
    
    Args:
        df: DataFrame with price data
        period: Lookback period (default: 12)
        column: Price column to use
    
    Returns:
        DataFrame with roc column
    
    Interpretation:
        - ROC > 0: Upward momentum
        - ROC < 0: Downward momentum
        - Divergence from price: Potential reversal
    """
    col_name = f"roc_{period}"
    
    return df.with_columns([
        (
            (pl.col(column) - pl.col(column).shift(period)) /
            (pl.col(column).shift(period) + 1e-10) * 100.0
        ).alias(col_name),
    ])


def momentum(
    df: pl.DataFrame,
    period: int = 10,
    column: str = "close",
) -> pl.DataFrame:
    """
    Momentum Indicator.
    
    Measures the amount of price change over a period.
    
    Formula:
        Momentum = Close - Close[n]
    
    Args:
        df: DataFrame with price data
        period: Lookback period (default: 10)
        column: Price column to use
    
    Returns:
        DataFrame with momentum column
    """
    col_name = f"momentum_{period}"
    
    return df.with_columns([
        (pl.col(column) - pl.col(column).shift(period)).alias(col_name),
    ])


def cci(
    df: pl.DataFrame,
    period: int = DEFAULT_CONFIG.CCI_PERIOD,
) -> pl.DataFrame:
    """
    Commodity Channel Index (CCI).
    
    Measures the deviation of price from its statistical mean.
    
    Formula:
        TP = (High + Low + Close) / 3
        CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)
    
    Args:
        df: DataFrame with OHLC data
        period: Lookback period (default: 20)
    
    Returns:
        DataFrame with cci column
    
    Interpretation:
        - CCI > 100: Overbought
        - CCI < -100: Oversold
    """
    col_name = f"cci_{period}"
    
    return df.with_columns([
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("_tp"),
    ]).with_columns([
        pl.col("_tp").rolling_mean(window_size=period).alias("_tp_sma"),
        pl.col("_tp").rolling_std(window_size=period).alias("_tp_std"),
    ]).with_columns([
        (
            (pl.col("_tp") - pl.col("_tp_sma")) /
            (0.015 * pl.col("_tp_std") + 1e-10)
        ).alias(col_name),
    ]).drop(["_tp", "_tp_sma", "_tp_std"])


def cmo(
    df: pl.DataFrame,
    period: int = DEFAULT_CONFIG.CMO_PERIOD,
    column: str = "close",
) -> pl.DataFrame:
    """
    Chande Momentum Oscillator (CMO).
    
    Measures momentum on both up and down days.
    
    Formula:
        CMO = 100 * (Sum of Ups - Sum of Downs) / (Sum of Ups + Sum of Downs)
    
    Args:
        df: DataFrame with price data
        period: Lookback period (default: 14)
        column: Price column to use
    
    Returns:
        DataFrame with cmo column
    
    Interpretation:
        - CMO > 50: Strong upward momentum
        - CMO < -50: Strong downward momentum
    """
    col_name = f"cmo_{period}"
    
    return df.with_columns([
        pl.col(column).diff().alias("_delta"),
    ]).with_columns([
        pl.when(pl.col("_delta") > 0)
        .then(pl.col("_delta"))
        .otherwise(0.0)
        .rolling_sum(window_size=period)
        .alias("_sum_up"),
        pl.when(pl.col("_delta") < 0)
        .then(pl.col("_delta").abs())
        .otherwise(0.0)
        .rolling_sum(window_size=period)
        .alias("_sum_down"),
    ]).with_columns([
        (
            100.0 * (pl.col("_sum_up") - pl.col("_sum_down")) /
            (pl.col("_sum_up") + pl.col("_sum_down") + 1e-10)
        ).alias(col_name),
    ]).drop(["_delta", "_sum_up", "_sum_down"])


def ultimate_oscillator(
    df: pl.DataFrame,
    period1: int = 7,
    period2: int = 14,
    period3: int = 28,
) -> pl.DataFrame:
    """
    Ultimate Oscillator.
    
    Multi-timeframe momentum oscillator combining three periods.
    
    Formula:
        BP = Close - Min(Low, Previous Close)
        TR = Max(High, Previous Close) - Min(Low, Previous Close)
        UO = 100 * [(4 * Avg1) + (2 * Avg2) + Avg3] / 7
    
    Args:
        df: DataFrame with OHLC data
        period1: Short period (default: 7)
        period2: Medium period (default: 14)
        period3: Long period (default: 28)
    
    Returns:
        DataFrame with ultimate_oscillator column
    
    Interpretation:
        - UO > 70: Overbought
        - UO < 30: Oversold
    """
    return df.with_columns([
        pl.col("close").shift(1).alias("_prev_close"),
    ]).with_columns([
        (
            pl.col("close") -
            pl.min_horizontal("low", "_prev_close")
        ).alias("_bp"),
        (
            pl.max_horizontal("high", "_prev_close") -
            pl.min_horizontal("low", "_prev_close")
        ).alias("_tr"),
    ]).with_columns([
        (
            pl.col("_bp").rolling_sum(window_size=period1) /
            (pl.col("_tr").rolling_sum(window_size=period1) + 1e-10)
        ).alias("_avg1"),
        (
            pl.col("_bp").rolling_sum(window_size=period2) /
            (pl.col("_tr").rolling_sum(window_size=period2) + 1e-10)
        ).alias("_avg2"),
        (
            pl.col("_bp").rolling_sum(window_size=period3) /
            (pl.col("_tr").rolling_sum(window_size=period3) + 1e-10)
        ).alias("_avg3"),
    ]).with_columns([
        (
            100.0 * (4 * pl.col("_avg1") + 2 * pl.col("_avg2") + pl.col("_avg3")) / 7.0
        ).alias("ultimate_oscillator"),
    ]).drop(["_prev_close", "_bp", "_tr", "_avg1", "_avg2", "_avg3"])


def tsi(
    df: pl.DataFrame,
    long_period: int = 25,
    short_period: int = 13,
    signal_period: int = 13,
    column: str = "close",
) -> pl.DataFrame:
    """
    True Strength Index (TSI).
    
    Double-smoothed momentum oscillator.
    
    Formula:
        TSI = 100 * EMA(EMA(PC, long), short) / EMA(EMA(|PC|, long), short)
        PC = Price Change
    
    Args:
        df: DataFrame with price data
        long_period: Long smoothing period (default: 25)
        short_period: Short smoothing period (default: 13)
        signal_period: Signal line period (default: 13)
        column: Price column to use
    
    Returns:
        DataFrame with tsi and tsi_signal columns
    """
    return df.with_columns([
        pl.col(column).diff().alias("_pc"),
    ]).with_columns([
        pl.col("_pc")
        .ewm_mean(span=long_period, adjust=False)
        .ewm_mean(span=short_period, adjust=False)
        .alias("_double_smoothed_pc"),
        pl.col("_pc")
        .abs()
        .ewm_mean(span=long_period, adjust=False)
        .ewm_mean(span=short_period, adjust=False)
        .alias("_double_smoothed_abs_pc"),
    ]).with_columns([
        (
            100.0 * pl.col("_double_smoothed_pc") /
            (pl.col("_double_smoothed_abs_pc") + 1e-10)
        ).alias("tsi"),
    ]).with_columns([
        pl.col("tsi")
        .ewm_mean(span=signal_period, adjust=False)
        .alias("tsi_signal"),
    ]).drop(["_pc", "_double_smoothed_pc", "_double_smoothed_abs_pc"])


# =============================================================================
# TREND INDICATORS
# =============================================================================

def sma(
    df: pl.DataFrame,
    period: int,
    column: str = "close",
) -> pl.DataFrame:
    """
    Simple Moving Average (SMA).
    
    Unweighted mean of the previous n data points.
    
    Formula:
        SMA = Sum(Price, n) / n
    
    Args:
        df: DataFrame with price data
        period: Moving average period
        column: Price column to use
    
    Returns:
        DataFrame with sma column added
    """
    col_name = f"sma_{period}"
    
    return df.with_columns([
        pl.col(column)
        .rolling_mean(window_size=period)
        .alias(col_name),
    ])


def ema(
    df: pl.DataFrame,
    period: int,
    column: str = "close",
) -> pl.DataFrame:
    """
    Exponential Moving Average (EMA).
    
    Weighted moving average giving more weight to recent data.
    
    Formula:
        EMA = Price * k + EMA[1] * (1 - k)
        k = 2 / (period + 1)
    
    Args:
        df: DataFrame with price data
        period: Moving average period
        column: Price column to use
    
    Returns:
        DataFrame with ema column added
    """
    col_name = f"ema_{period}"
    
    return df.with_columns([
        pl.col(column)
        .ewm_mean(span=period, adjust=False)
        .alias(col_name),
    ])


def wma(
    df: pl.DataFrame,
    period: int,
    column: str = "close",
) -> pl.DataFrame:
    """
    Weighted Moving Average (WMA).
    
    Moving average with linearly decreasing weights.
    
    Formula:
        WMA = Sum(Price * Weight) / Sum(Weight)
        Weights = [1, 2, 3, ..., n]
    
    Args:
        df: DataFrame with price data
        period: Moving average period
        column: Price column to use
    
    Returns:
        DataFrame with wma column added
    """
    col_name = f"wma_{period}"
    
    # Create weights
    weights = np.arange(1, period + 1, dtype=np.float64)
    weight_sum = weights.sum()
    
    # Apply weighted moving average using rolling apply
    return df.with_columns([
        pl.col(column)
        .rolling_map(
            lambda x: np.sum(x * weights[-len(x):]) / weight_sum,
            window_size=period,
        )
        .alias(col_name),
    ])


def dema(
    df: pl.DataFrame,
    period: int,
    column: str = "close",
) -> pl.DataFrame:
    """
    Double Exponential Moving Average (DEMA).
    
    Reduces lag compared to traditional EMA.
    
    Formula:
        DEMA = 2 * EMA - EMA(EMA)
    
    Args:
        df: DataFrame with price data
        period: Moving average period
        column: Price column to use
    
    Returns:
        DataFrame with dema column added
    """
    col_name = f"dema_{period}"
    
    return df.with_columns([
        pl.col(column)
        .ewm_mean(span=period, adjust=False)
        .alias("_ema1"),
    ]).with_columns([
        pl.col("_ema1")
        .ewm_mean(span=period, adjust=False)
        .alias("_ema2"),
    ]).with_columns([
        (2 * pl.col("_ema1") - pl.col("_ema2")).alias(col_name),
    ]).drop(["_ema1", "_ema2"])


def tema(
    df: pl.DataFrame,
    period: int,
    column: str = "close",
) -> pl.DataFrame:
    """
    Triple Exponential Moving Average (TEMA).
    
    Further reduces lag compared to DEMA.
    
    Formula:
        TEMA = 3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))
    
    Args:
        df: DataFrame with price data
        period: Moving average period
        column: Price column to use
    
    Returns:
        DataFrame with tema column added
    """
    col_name = f"tema_{period}"
    
    return df.with_columns([
        pl.col(column)
        .ewm_mean(span=period, adjust=False)
        .alias("_ema1"),
    ]).with_columns([
        pl.col("_ema1")
        .ewm_mean(span=period, adjust=False)
        .alias("_ema2"),
    ]).with_columns([
        pl.col("_ema2")
        .ewm_mean(span=period, adjust=False)
        .alias("_ema3"),
    ]).with_columns([
        (
            3 * pl.col("_ema1") - 3 * pl.col("_ema2") + pl.col("_ema3")
        ).alias(col_name),
    ]).drop(["_ema1", "_ema2", "_ema3"])


def adx(
    df: pl.DataFrame,
    period: int = DEFAULT_CONFIG.ADX_PERIOD,
) -> pl.DataFrame:
    """
    Average Directional Index (ADX).
    
    Measures trend strength regardless of direction.
    
    Formula:
        +DM = High - Previous High (if positive and > -DM)
        -DM = Previous Low - Low (if positive and > +DM)
        TR = Max(High - Low, |High - Prev Close|, |Low - Prev Close|)
        +DI = 100 * EMA(+DM) / EMA(TR)
        -DI = 100 * EMA(-DM) / EMA(TR)
        DX = 100 * |+DI - -DI| / (+DI + -DI)
        ADX = EMA(DX)
    
    Args:
        df: DataFrame with OHLC data
        period: ADX period (default: 14)
    
    Returns:
        DataFrame with adx, plus_di, minus_di columns
    
    Interpretation:
        - ADX > 25: Strong trend
        - ADX < 20: Weak trend or ranging
        - +DI > -DI: Bullish
        - -DI > +DI: Bearish
    """
    return df.with_columns([
        pl.col("high").shift(1).alias("_prev_high"),
        pl.col("low").shift(1).alias("_prev_low"),
        pl.col("close").shift(1).alias("_prev_close"),
    ]).with_columns([
        # True Range
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("_prev_close")).abs(),
            (pl.col("low") - pl.col("_prev_close")).abs(),
        ).alias("_tr"),
        # +DM and -DM
        (pl.col("high") - pl.col("_prev_high")).alias("_up_move"),
        (pl.col("_prev_low") - pl.col("low")).alias("_down_move"),
    ]).with_columns([
        pl.when(
            (pl.col("_up_move") > pl.col("_down_move")) &
            (pl.col("_up_move") > 0)
        )
        .then(pl.col("_up_move"))
        .otherwise(0.0)
        .alias("_plus_dm"),
        pl.when(
            (pl.col("_down_move") > pl.col("_up_move")) &
            (pl.col("_down_move") > 0)
        )
        .then(pl.col("_down_move"))
        .otherwise(0.0)
        .alias("_minus_dm"),
    ]).with_columns([
        # Smooth TR, +DM, -DM
        pl.col("_tr").ewm_mean(span=period, adjust=False).alias("_atr"),
        pl.col("_plus_dm").ewm_mean(span=period, adjust=False).alias("_smooth_plus_dm"),
        pl.col("_minus_dm").ewm_mean(span=period, adjust=False).alias("_smooth_minus_dm"),
    ]).with_columns([
        # +DI and -DI
        (100.0 * pl.col("_smooth_plus_dm") / (pl.col("_atr") + 1e-10)).alias("plus_di"),
        (100.0 * pl.col("_smooth_minus_dm") / (pl.col("_atr") + 1e-10)).alias("minus_di"),
    ]).with_columns([
        # DX
        (
            100.0 * (pl.col("plus_di") - pl.col("minus_di")).abs() /
            (pl.col("plus_di") + pl.col("minus_di") + 1e-10)
        ).alias("_dx"),
    ]).with_columns([
        # ADX
        pl.col("_dx").ewm_mean(span=period, adjust=False).alias("adx"),
    ]).drop([
        "_prev_high", "_prev_low", "_prev_close", "_tr", "_up_move",
        "_down_move", "_plus_dm", "_minus_dm", "_atr",
        "_smooth_plus_dm", "_smooth_minus_dm", "_dx"
    ])


def supertrend(
    df: pl.DataFrame,
    period: int = DEFAULT_CONFIG.SUPERTREND_PERIOD,
    multiplier: float = DEFAULT_CONFIG.SUPERTREND_MULTIPLIER,
) -> pl.DataFrame:
    """
    Supertrend Indicator.
    
    Trend following indicator using ATR for volatility.
    
    Formula:
        Basic Upper Band = (High + Low) / 2 + Multiplier * ATR
        Basic Lower Band = (High + Low) / 2 - Multiplier * ATR
        Final bands adjusted based on previous close
    
    Args:
        df: DataFrame with OHLC data
        period: ATR period (default: 10)
        multiplier: ATR multiplier (default: 3.0)
    
    Returns:
        DataFrame with supertrend, supertrend_direction columns
    
    Interpretation:
        - Price above Supertrend: Bullish (direction = 1)
        - Price below Supertrend: Bearish (direction = -1)
    """
    # First calculate ATR
    df = df.with_columns([
        pl.col("close").shift(1).alias("_prev_close"),
    ]).with_columns([
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("_prev_close")).abs(),
            (pl.col("low") - pl.col("_prev_close")).abs(),
        ).alias("_tr"),
    ]).with_columns([
        pl.col("_tr").rolling_mean(window_size=period).alias("_atr"),
    ])
    
    # Calculate basic bands
    df = df.with_columns([
        ((pl.col("high") + pl.col("low")) / 2.0).alias("_hl2"),
    ]).with_columns([
        (pl.col("_hl2") + multiplier * pl.col("_atr")).alias("_basic_upper"),
        (pl.col("_hl2") - multiplier * pl.col("_atr")).alias("_basic_lower"),
    ])
    
    # Convert to numpy for iterative calculation
    closes = df["close"].to_numpy()
    basic_upper = df["_basic_upper"].to_numpy()
    basic_lower = df["_basic_lower"].to_numpy()
    
    n = len(closes)
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend_vals = np.zeros(n)
    direction = np.zeros(n)
    
    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    supertrend_vals[0] = basic_upper[0]
    direction[0] = 1
    
    for i in range(1, n):
        # Final Upper Band
        if basic_upper[i] < final_upper[i-1] or closes[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]
        
        # Final Lower Band
        if basic_lower[i] > final_lower[i-1] or closes[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]
        
        # Supertrend
        if supertrend_vals[i-1] == final_upper[i-1]:
            if closes[i] > final_upper[i]:
                supertrend_vals[i] = final_lower[i]
                direction[i] = 1
            else:
                supertrend_vals[i] = final_upper[i]
                direction[i] = -1
        else:
            if closes[i] < final_lower[i]:
                supertrend_vals[i] = final_upper[i]
                direction[i] = -1
            else:
                supertrend_vals[i] = final_lower[i]
                direction[i] = 1
    
    return df.with_columns([
        pl.Series("supertrend", supertrend_vals),
        pl.Series("supertrend_direction", direction).cast(pl.Int8),
    ]).drop(["_prev_close", "_tr", "_atr", "_hl2", "_basic_upper", "_basic_lower"])


def aroon(
    df: pl.DataFrame,
    period: int = DEFAULT_CONFIG.AROON_PERIOD,
) -> pl.DataFrame:
    """
    Aroon Indicator.
    
    Identifies trend changes and strength.
    
    Formula:
        Aroon Up = 100 * (period - bars since highest high) / period
        Aroon Down = 100 * (period - bars since lowest low) / period
        Aroon Oscillator = Aroon Up - Aroon Down
    
    Args:
        df: DataFrame with OHLC data
        period: Lookback period (default: 25)
    
    Returns:
        DataFrame with aroon_up, aroon_down, aroon_oscillator columns
    
    Interpretation:
        - Aroon Up > 70 & Aroon Down < 30: Strong uptrend
        - Aroon Down > 70 & Aroon Up < 30: Strong downtrend
    """
    # Rolling argmax/argmin for bars since high/low
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(highs)
    
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window_high = highs[i - period + 1:i + 1]
        window_low = lows[i - period + 1:i + 1]
        
        bars_since_high = period - 1 - np.argmax(window_high)
        bars_since_low = period - 1 - np.argmin(window_low)
        
        aroon_up[i] = 100.0 * (period - bars_since_high) / period
        aroon_down[i] = 100.0 * (period - bars_since_low) / period
    
    return df.with_columns([
        pl.Series("aroon_up", aroon_up),
        pl.Series("aroon_down", aroon_down),
    ]).with_columns([
        (pl.col("aroon_up") - pl.col("aroon_down")).alias("aroon_oscillator"),
    ])


def ichimoku(
    df: pl.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26,
) -> pl.DataFrame:
    """
    Ichimoku Cloud (Ichimoku Kinko Hyo).
    
    Comprehensive indicator showing support/resistance, trend, and momentum.
    
    Components:
        - Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        - Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        - Senkou Span A: (Tenkan + Kijun) / 2, displaced 26 periods forward
        - Senkou Span B: (52-period high + 52-period low) / 2, displaced forward
        - Chikou Span: Close displaced 26 periods backward
    
    Args:
        df: DataFrame with OHLC data
        tenkan_period: Tenkan-sen period (default: 9)
        kijun_period: Kijun-sen period (default: 26)
        senkou_b_period: Senkou Span B period (default: 52)
        displacement: Cloud displacement (default: 26)
    
    Returns:
        DataFrame with Ichimoku components
    """
    return df.with_columns([
        # Tenkan-sen
        (
            (
                pl.col("high").rolling_max(window_size=tenkan_period) +
                pl.col("low").rolling_min(window_size=tenkan_period)
            ) / 2.0
        ).alias("ichimoku_tenkan"),
        # Kijun-sen
        (
            (
                pl.col("high").rolling_max(window_size=kijun_period) +
                pl.col("low").rolling_min(window_size=kijun_period)
            ) / 2.0
        ).alias("ichimoku_kijun"),
        # Chikou Span
        pl.col("close").shift(displacement).alias("ichimoku_chikou"),
    ]).with_columns([
        # Senkou Span A
        (
            (pl.col("ichimoku_tenkan") + pl.col("ichimoku_kijun")) / 2.0
        ).shift(-displacement).alias("ichimoku_senkou_a"),
        # Senkou Span B
        (
            (
                pl.col("high").rolling_max(window_size=senkou_b_period) +
                pl.col("low").rolling_min(window_size=senkou_b_period)
            ) / 2.0
        ).shift(-displacement).alias("ichimoku_senkou_b"),
    ])


def parabolic_sar(
    df: pl.DataFrame,
    af_start: float = 0.02,
    af_increment: float = 0.02,
    af_max: float = 0.20,
) -> pl.DataFrame:
    """
    Parabolic SAR (Stop and Reverse).
    
    Trend following indicator providing potential entry/exit points.
    
    Formula:
        SAR = Prior SAR + AF * (EP - Prior SAR)
        AF increases by increment when new extreme point is made
        EP = Extreme Point (highest high in uptrend, lowest low in downtrend)
    
    Args:
        df: DataFrame with OHLC data
        af_start: Starting acceleration factor (default: 0.02)
        af_increment: AF increment (default: 0.02)
        af_max: Maximum AF (default: 0.20)
    
    Returns:
        DataFrame with psar and psar_direction columns
    
    Interpretation:
        - Price above SAR: Bullish
        - Price below SAR: Bearish
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    n = len(highs)
    
    psar = np.zeros(n)
    direction = np.zeros(n)  # 1 = uptrend, -1 = downtrend
    af = np.zeros(n)
    ep = np.zeros(n)
    
    # Initialize
    direction[0] = 1 if closes[0] > closes[min(1, n-1)] else -1
    psar[0] = lows[0] if direction[0] == 1 else highs[0]
    af[0] = af_start
    ep[0] = highs[0] if direction[0] == 1 else lows[0]
    
    for i in range(1, n):
        # Calculate SAR
        psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
        
        # Adjust SAR based on prior bars
        if direction[i-1] == 1:  # Uptrend
            psar[i] = min(psar[i], lows[i-1])
            if i >= 2:
                psar[i] = min(psar[i], lows[i-2])
        else:  # Downtrend
            psar[i] = max(psar[i], highs[i-1])
            if i >= 2:
                psar[i] = max(psar[i], highs[i-2])
        
        # Check for reversal
        reverse = False
        if direction[i-1] == 1:
            if lows[i] < psar[i]:
                direction[i] = -1
                reverse = True
                psar[i] = ep[i-1]
                ep[i] = lows[i]
                af[i] = af_start
            else:
                direction[i] = 1
        else:
            if highs[i] > psar[i]:
                direction[i] = 1
                reverse = True
                psar[i] = ep[i-1]
                ep[i] = highs[i]
                af[i] = af_start
            else:
                direction[i] = -1
        
        if not reverse:
            # Update EP and AF
            if direction[i] == 1:
                if highs[i] > ep[i-1]:
                    ep[i] = highs[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
            else:
                if lows[i] < ep[i-1]:
                    ep[i] = lows[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
    
    return df.with_columns([
        pl.Series("psar", psar),
        pl.Series("psar_direction", direction).cast(pl.Int8),
    ])


# =============================================================================
# VOLATILITY INDICATORS
# =============================================================================

def bollinger_bands(
    df: pl.DataFrame,
    period: int = DEFAULT_CONFIG.BB_PERIOD,
    std_dev: float = DEFAULT_CONFIG.BB_STD,
    column: str = "close",
) -> pl.DataFrame:
    """
    Bollinger Bands.
    
    Volatility bands placed above and below a moving average.
    
    Formula:
        Middle Band = SMA(close, period)
        Upper Band = Middle Band + (std_dev * Standard Deviation)
        Lower Band = Middle Band - (std_dev * Standard Deviation)
        %B = (Price - Lower Band) / (Upper Band - Lower Band)
        Bandwidth = (Upper Band - Lower Band) / Middle Band
    
    Args:
        df: DataFrame with price data
        period: SMA period (default: 20)
        std_dev: Standard deviation multiplier (default: 2.0)
        column: Price column to use
    
    Returns:
        DataFrame with bb_upper, bb_middle, bb_lower, bb_pctb, bb_width columns
    
    Interpretation:
        - Price near upper band: Overbought
        - Price near lower band: Oversold
        - Bandwidth squeeze: Low volatility, potential breakout
    """
    return df.with_columns([
        pl.col(column).rolling_mean(window_size=period).alias("bb_middle"),
        pl.col(column).rolling_std(window_size=period).alias("_bb_std"),
    ]).with_columns([
        (pl.col("bb_middle") + std_dev * pl.col("_bb_std")).alias("bb_upper"),
        (pl.col("bb_middle") - std_dev * pl.col("_bb_std")).alias("bb_lower"),
    ]).with_columns([
        # Percent B
        (
            (pl.col(column) - pl.col("bb_lower")) /
            (pl.col("bb_upper") - pl.col("bb_lower") + 1e-10)
        ).alias("bb_pctb"),
        # Bandwidth
        (
            (pl.col("bb_upper") - pl.col("bb_lower")) /
            (pl.col("bb_middle") + 1e-10)
        ).alias("bb_width"),
    ]).drop(["_bb_std"])


def atr(
    df: pl.DataFrame,
    period: int = DEFAULT_CONFIG.ATR_PERIOD,
) -> pl.DataFrame:
    """
    Average True Range (ATR).
    
    Measures market volatility by analyzing the range of price movement.
    
    Formula:
        TR = Max(High - Low, |High - Prev Close|, |Low - Prev Close|)
        ATR = EMA(TR, period)
    
    Args:
        df: DataFrame with OHLC data
        period: ATR period (default: 14)
    
    Returns:
        DataFrame with atr and atr_pct columns
    
    Interpretation:
        - High ATR: High volatility
        - Low ATR: Low volatility
        - Used for stop-loss placement and position sizing
    """
    col_name = f"atr_{period}"
    
    return df.with_columns([
        pl.col("close").shift(1).alias("_prev_close"),
    ]).with_columns([
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("_prev_close")).abs(),
            (pl.col("low") - pl.col("_prev_close")).abs(),
        ).alias("_tr"),
    ]).with_columns([
        pl.col("_tr").ewm_mean(span=period, adjust=False).alias(col_name),
    ]).with_columns([
        # ATR as percentage of close
        (pl.col(col_name) / pl.col("close") * 100.0).alias(f"{col_name}_pct"),
    ]).drop(["_prev_close", "_tr"])


def keltner_channels(
    df: pl.DataFrame,
    period: int = DEFAULT_CONFIG.KELTNER_PERIOD,
    multiplier: float = DEFAULT_CONFIG.KELTNER_MULTIPLIER,
) -> pl.DataFrame:
    """
    Keltner Channels.
    
    Volatility-based envelope using ATR instead of standard deviation.
    
    Formula:
        Middle = EMA(close, period)
        Upper = Middle + multiplier * ATR
        Lower = Middle - multiplier * ATR
    
    Args:
        df: DataFrame with OHLC data
        period: EMA/ATR period (default: 20)
        multiplier: ATR multiplier (default: 2.0)
    
    Returns:
        DataFrame with kc_upper, kc_middle, kc_lower columns
    """
    # First calculate ATR
    df = df.with_columns([
        pl.col("close").shift(1).alias("_prev_close"),
    ]).with_columns([
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("_prev_close")).abs(),
            (pl.col("low") - pl.col("_prev_close")).abs(),
        ).alias("_tr"),
    ]).with_columns([
        pl.col("_tr").ewm_mean(span=period, adjust=False).alias("_atr"),
        pl.col("close").ewm_mean(span=period, adjust=False).alias("kc_middle"),
    ]).with_columns([
        (pl.col("kc_middle") + multiplier * pl.col("_atr")).alias("kc_upper"),
        (pl.col("kc_middle") - multiplier * pl.col("_atr")).alias("kc_lower"),
    ]).drop(["_prev_close", "_tr", "_atr"])
    
    return df


def donchian_channels(
    df: pl.DataFrame,
    period: int = DEFAULT_CONFIG.DONCHIAN_PERIOD,
) -> pl.DataFrame:
    """
    Donchian Channels.
    
    Price channels based on highest high and lowest low.
    
    Formula:
        Upper = Highest High over period
        Lower = Lowest Low over period
        Middle = (Upper + Lower) / 2
    
    Args:
        df: DataFrame with OHLC data
        period: Lookback period (default: 20)
    
    Returns:
        DataFrame with dc_upper, dc_middle, dc_lower columns
    
    Interpretation:
        - Price breaks above upper: Potential long entry
        - Price breaks below lower: Potential short entry
        - Used in turtle trading system
    """
    return df.with_columns([
        pl.col("high").rolling_max(window_size=period).alias("dc_upper"),
        pl.col("low").rolling_min(window_size=period).alias("dc_lower"),
    ]).with_columns([
        ((pl.col("dc_upper") + pl.col("dc_lower")) / 2.0).alias("dc_middle"),
    ])


def natr(
    df: pl.DataFrame,
    period: int = DEFAULT_CONFIG.ATR_PERIOD,
) -> pl.DataFrame:
    """
    Normalized Average True Range (NATR).
    
    ATR expressed as percentage of closing price.
    
    Formula:
        NATR = (ATR / Close) * 100
    
    Args:
        df: DataFrame with OHLC data
        period: ATR period (default: 14)
    
    Returns:
        DataFrame with natr column
    """
    col_name = f"natr_{period}"
    
    return df.with_columns([
        pl.col("close").shift(1).alias("_prev_close"),
    ]).with_columns([
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("_prev_close")).abs(),
            (pl.col("low") - pl.col("_prev_close")).abs(),
        ).alias("_tr"),
    ]).with_columns([
        (
            pl.col("_tr").ewm_mean(span=period, adjust=False) /
            pl.col("close") * 100.0
        ).alias(col_name),
    ]).drop(["_prev_close", "_tr"])


def historical_volatility(
    df: pl.DataFrame,
    period: int = 20,
    annualize: bool = True,
    trading_days: int = 252,
    column: str = "close",
) -> pl.DataFrame:
    """
    Historical Volatility (HV).
    
    Standard deviation of log returns, optionally annualized.
    
    Formula:
        Log Return = ln(Close / Prev Close)
        HV = StdDev(Log Returns) * sqrt(trading_days)
    
    Args:
        df: DataFrame with price data
        period: Lookback period (default: 20)
        annualize: Annualize volatility (default: True)
        trading_days: Trading days per year (default: 252)
        column: Price column to use
    
    Returns:
        DataFrame with hv column
    """
    col_name = f"hv_{period}"
    multiplier = np.sqrt(trading_days) if annualize else 1.0
    
    return df.with_columns([
        (pl.col(column) / pl.col(column).shift(1)).log().alias("_log_return"),
    ]).with_columns([
        (pl.col("_log_return").rolling_std(window_size=period) * multiplier).alias(col_name),
    ]).drop(["_log_return"])


# =============================================================================
# VOLUME INDICATORS
# =============================================================================

def obv(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    On-Balance Volume (OBV).
    
    Cumulative volume indicator showing buying/selling pressure.
    
    Formula:
        If Close > Prev Close: OBV = Prev OBV + Volume
        If Close < Prev Close: OBV = Prev OBV - Volume
        If Close = Prev Close: OBV = Prev OBV
    
    Args:
        df: DataFrame with close and volume data
    
    Returns:
        DataFrame with obv column
    
    Interpretation:
        - Rising OBV: Buying pressure
        - Falling OBV: Selling pressure
        - Divergence from price: Potential reversal
    """
    return df.with_columns([
        pl.when(pl.col("close") > pl.col("close").shift(1))
        .then(pl.col("volume"))
        .when(pl.col("close") < pl.col("close").shift(1))
        .then(-pl.col("volume"))
        .otherwise(0.0)
        .cum_sum()
        .alias("obv"),
    ])


def vwap(
    df: pl.DataFrame,
    anchor: str = "session",
) -> pl.DataFrame:
    """
    Volume Weighted Average Price (VWAP).
    
    Average price weighted by volume.
    
    Formula:
        VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
        Typical Price = (High + Low + Close) / 3
    
    Args:
        df: DataFrame with OHLCV data
        anchor: Reset period ("session", "week", "month")
    
    Returns:
        DataFrame with vwap column
    
    Interpretation:
        - Price above VWAP: Bullish intraday
        - Price below VWAP: Bearish intraday
        - Institutional benchmark for execution quality
    """
    return df.with_columns([
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("_tp"),
    ]).with_columns([
        (pl.col("_tp") * pl.col("volume")).cum_sum().alias("_cum_tp_vol"),
        pl.col("volume").cum_sum().alias("_cum_vol"),
    ]).with_columns([
        (pl.col("_cum_tp_vol") / (pl.col("_cum_vol") + 1e-10)).alias("vwap"),
    ]).drop(["_tp", "_cum_tp_vol", "_cum_vol"])


def mfi(
    df: pl.DataFrame,
    period: int = DEFAULT_CONFIG.MFI_PERIOD,
) -> pl.DataFrame:
    """
    Money Flow Index (MFI).
    
    Volume-weighted RSI measuring buying/selling pressure.
    
    Formula:
        Typical Price = (High + Low + Close) / 3
        Raw Money Flow = Typical Price * Volume
        Money Flow Ratio = Positive Flow / Negative Flow
        MFI = 100 - (100 / (1 + Money Flow Ratio))
    
    Args:
        df: DataFrame with OHLCV data
        period: MFI period (default: 14)
    
    Returns:
        DataFrame with mfi column
    
    Interpretation:
        - MFI > 80: Overbought
        - MFI < 20: Oversold
    """
    col_name = f"mfi_{period}"
    
    return df.with_columns([
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("_tp"),
    ]).with_columns([
        (pl.col("_tp") * pl.col("volume")).alias("_raw_mf"),
        pl.col("_tp").diff().alias("_tp_change"),
    ]).with_columns([
        pl.when(pl.col("_tp_change") > 0)
        .then(pl.col("_raw_mf"))
        .otherwise(0.0)
        .rolling_sum(window_size=period)
        .alias("_positive_mf"),
        pl.when(pl.col("_tp_change") < 0)
        .then(pl.col("_raw_mf"))
        .otherwise(0.0)
        .rolling_sum(window_size=period)
        .alias("_negative_mf"),
    ]).with_columns([
        (
            100.0 - 100.0 / (1.0 + pl.col("_positive_mf") / (pl.col("_negative_mf") + 1e-10))
        ).alias(col_name),
    ]).drop(["_tp", "_raw_mf", "_tp_change", "_positive_mf", "_negative_mf"])


def ad_line(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Accumulation/Distribution Line.
    
    Volume-based indicator measuring money flow.
    
    Formula:
        MFM = ((Close - Low) - (High - Close)) / (High - Low)
        MFV = MFM * Volume
        A/D = Cumulative(MFV)
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with ad_line column
    
    Interpretation:
        - Rising A/D: Accumulation (buying pressure)
        - Falling A/D: Distribution (selling pressure)
    """
    return df.with_columns([
        (
            ((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close"))) /
            (pl.col("high") - pl.col("low") + 1e-10)
        ).alias("_mfm"),
    ]).with_columns([
        (pl.col("_mfm") * pl.col("volume")).cum_sum().alias("ad_line"),
    ]).drop(["_mfm"])


def cmf(
    df: pl.DataFrame,
    period: int = DEFAULT_CONFIG.CMF_PERIOD,
) -> pl.DataFrame:
    """
    Chaikin Money Flow (CMF).
    
    Oscillator based on A/D line.
    
    Formula:
        MFM = ((Close - Low) - (High - Close)) / (High - Low)
        MFV = MFM * Volume
        CMF = Sum(MFV, period) / Sum(Volume, period)
    
    Args:
        df: DataFrame with OHLCV data
        period: CMF period (default: 20)
    
    Returns:
        DataFrame with cmf column
    
    Interpretation:
        - CMF > 0: Buying pressure
        - CMF < 0: Selling pressure
    """
    col_name = f"cmf_{period}"
    
    return df.with_columns([
        (
            ((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close"))) /
            (pl.col("high") - pl.col("low") + 1e-10)
        ).alias("_mfm"),
    ]).with_columns([
        (pl.col("_mfm") * pl.col("volume")).alias("_mfv"),
    ]).with_columns([
        (
            pl.col("_mfv").rolling_sum(window_size=period) /
            (pl.col("volume").rolling_sum(window_size=period) + 1e-10)
        ).alias(col_name),
    ]).drop(["_mfm", "_mfv"])


def force_index(
    df: pl.DataFrame,
    period: int = DEFAULT_CONFIG.FORCE_PERIOD,
) -> pl.DataFrame:
    """
    Force Index.
    
    Oscillator measuring the force behind price movements.
    
    Formula:
        Force Index = Close Change * Volume
        Smoothed FI = EMA(Force Index, period)
    
    Args:
        df: DataFrame with close and volume data
        period: Smoothing period (default: 13)
    
    Returns:
        DataFrame with force_index column
    
    Interpretation:
        - Positive: Bulls in control
        - Negative: Bears in control
    """
    col_name = f"force_index_{period}"
    
    return df.with_columns([
        (pl.col("close").diff() * pl.col("volume")).alias("_raw_force"),
    ]).with_columns([
        pl.col("_raw_force").ewm_mean(span=period, adjust=False).alias(col_name),
    ]).drop(["_raw_force"])


def vwma(
    df: pl.DataFrame,
    period: int = 20,
    column: str = "close",
) -> pl.DataFrame:
    """
    Volume Weighted Moving Average (VWMA).
    
    Moving average weighted by volume.
    
    Formula:
        VWMA = Sum(Price * Volume, n) / Sum(Volume, n)
    
    Args:
        df: DataFrame with price and volume data
        period: Moving average period (default: 20)
        column: Price column to use
    
    Returns:
        DataFrame with vwma column
    """
    col_name = f"vwma_{period}"
    
    return df.with_columns([
        (pl.col(column) * pl.col("volume")).alias("_pv"),
    ]).with_columns([
        (
            pl.col("_pv").rolling_sum(window_size=period) /
            (pl.col("volume").rolling_sum(window_size=period) + 1e-10)
        ).alias(col_name),
    ]).drop(["_pv"])


def eom(
    df: pl.DataFrame,
    period: int = 14,
) -> pl.DataFrame:
    """
    Ease of Movement (EOM).
    
    Relates price movement to volume.
    
    Formula:
        Distance Moved = ((High + Low) / 2) - ((Prev High + Prev Low) / 2)
        Box Ratio = (Volume / 10000) / (High - Low)
        EOM = Distance Moved / Box Ratio
        Smoothed EOM = SMA(EOM, period)
    
    Args:
        df: DataFrame with OHLCV data
        period: Smoothing period (default: 14)
    
    Returns:
        DataFrame with eom column
    """
    col_name = f"eom_{period}"
    
    return df.with_columns([
        ((pl.col("high") + pl.col("low")) / 2.0).alias("_midpoint"),
    ]).with_columns([
        (pl.col("_midpoint") - pl.col("_midpoint").shift(1)).alias("_distance"),
        (
            (pl.col("volume") / 10000.0) /
            (pl.col("high") - pl.col("low") + 1e-10)
        ).alias("_box_ratio"),
    ]).with_columns([
        (pl.col("_distance") / (pl.col("_box_ratio") + 1e-10)).alias("_raw_eom"),
    ]).with_columns([
        pl.col("_raw_eom").rolling_mean(window_size=period).alias(col_name),
    ]).drop(["_midpoint", "_distance", "_box_ratio", "_raw_eom"])


def volume_profile(
    df: pl.DataFrame,
    num_bins: int = 24,
) -> pl.DataFrame:
    """
    Volume Profile.
    
    Shows volume distribution across price levels.
    
    Args:
        df: DataFrame with OHLCV data
        num_bins: Number of price bins (default: 24)
    
    Returns:
        DataFrame with volume_profile_bin column
    """
    # Get price range
    price_min = df["low"].min()
    price_max = df["high"].max()
    
    if price_min is None or price_max is None:
        return df.with_columns([
            pl.lit(None).alias("volume_profile_bin"),
        ])
    
    bin_size = (price_max - price_min) / num_bins
    
    return df.with_columns([
        (
            ((pl.col("close") - price_min) / (bin_size + 1e-10)).floor().cast(pl.Int32)
        ).alias("volume_profile_bin"),
    ])


# =============================================================================
# COMPREHENSIVE INDICATOR CLASS
# =============================================================================

class TechnicalIndicators:
    """
    Comprehensive technical indicators class.
    
    Provides a unified interface for applying multiple indicators
    to price data with configurable parameters.
    
    Example:
        ti = TechnicalIndicators()
        df = ti.add_momentum_indicators(df)
        df = ti.add_trend_indicators(df)
        df = ti.add_all(df)
    """
    
    def __init__(self, config: IndicatorConfig | None = None):
        """
        Initialize with configuration.
        
        Args:
            config: Indicator configuration (uses defaults if None)
        """
        self.config = config or DEFAULT_CONFIG
    
    def add_momentum_indicators(
        self,
        df: pl.DataFrame,
        include: list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Add momentum indicators.
        
        Args:
            df: Input DataFrame
            include: List of indicators to include (None = all)
        
        Returns:
            DataFrame with momentum indicators
        """
        all_momentum = {
            "rsi": lambda d: rsi(d, self.config.RSI_PERIOD),
            "macd": lambda d: macd(d, self.config.MACD_FAST, self.config.MACD_SLOW, self.config.MACD_SIGNAL),
            "stochastic": lambda d: stochastic(d, self.config.STOCH_K_PERIOD, self.config.STOCH_D_PERIOD),
            "williams_r": lambda d: williams_r(d, self.config.WILLIAMS_PERIOD),
            "roc": lambda d: roc(d, self.config.ROC_PERIOD),
            "cci": lambda d: cci(d, self.config.CCI_PERIOD),
            "cmo": lambda d: cmo(d, self.config.CMO_PERIOD),
            "ultimate_oscillator": ultimate_oscillator,
            "tsi": tsi,
        }
        
        indicators = include or list(all_momentum.keys())
        
        for name in indicators:
            if name in all_momentum:
                df = all_momentum[name](df)
        
        return df
    
    def add_trend_indicators(
        self,
        df: pl.DataFrame,
        include: list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Add trend indicators.
        
        Args:
            df: Input DataFrame
            include: List of indicators to include (None = all)
        
        Returns:
            DataFrame with trend indicators
        """
        all_trend = {
            "sma": lambda d: self._add_multi_sma(d),
            "ema": lambda d: self._add_multi_ema(d),
            "adx": lambda d: adx(d, self.config.ADX_PERIOD),
            "supertrend": lambda d: supertrend(d, self.config.SUPERTREND_PERIOD, self.config.SUPERTREND_MULTIPLIER),
            "aroon": lambda d: aroon(d, self.config.AROON_PERIOD),
            "ichimoku": ichimoku,
            "parabolic_sar": parabolic_sar,
        }
        
        indicators = include or list(all_trend.keys())
        
        for name in indicators:
            if name in all_trend:
                df = all_trend[name](df)
        
        return df
    
    def add_volatility_indicators(
        self,
        df: pl.DataFrame,
        include: list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Add volatility indicators.
        
        Args:
            df: Input DataFrame
            include: List of indicators to include (None = all)
        
        Returns:
            DataFrame with volatility indicators
        """
        all_volatility = {
            "bollinger_bands": lambda d: bollinger_bands(d, self.config.BB_PERIOD, self.config.BB_STD),
            "atr": lambda d: atr(d, self.config.ATR_PERIOD),
            "keltner_channels": lambda d: keltner_channels(d, self.config.KELTNER_PERIOD, self.config.KELTNER_MULTIPLIER),
            "donchian_channels": lambda d: donchian_channels(d, self.config.DONCHIAN_PERIOD),
            "natr": lambda d: natr(d, self.config.ATR_PERIOD),
            "historical_volatility": historical_volatility,
        }
        
        indicators = include or list(all_volatility.keys())
        
        for name in indicators:
            if name in all_volatility:
                df = all_volatility[name](df)
        
        return df
    
    def add_volume_indicators(
        self,
        df: pl.DataFrame,
        include: list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Add volume indicators.
        
        Args:
            df: Input DataFrame
            include: List of indicators to include (None = all)
        
        Returns:
            DataFrame with volume indicators
        """
        all_volume = {
            "obv": obv,
            "vwap": vwap,
            "mfi": lambda d: mfi(d, self.config.MFI_PERIOD),
            "ad_line": ad_line,
            "cmf": lambda d: cmf(d, self.config.CMF_PERIOD),
            "force_index": lambda d: force_index(d, self.config.FORCE_PERIOD),
            "vwma": vwma,
            "eom": eom,
        }
        
        indicators = include or list(all_volume.keys())
        
        for name in indicators:
            if name in all_volume:
                df = all_volume[name](df)
        
        return df
    
    def add_all(
        self,
        df: pl.DataFrame,
        categories: list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Add all indicators.
        
        Args:
            df: Input DataFrame
            categories: Categories to include (None = all)
        
        Returns:
            DataFrame with all indicators
        """
        all_categories = ["momentum", "trend", "volatility", "volume"]
        categories = categories or all_categories
        
        if "momentum" in categories:
            df = self.add_momentum_indicators(df)
        if "trend" in categories:
            df = self.add_trend_indicators(df)
        if "volatility" in categories:
            df = self.add_volatility_indicators(df)
        if "volume" in categories:
            df = self.add_volume_indicators(df)
        
        return df
    
    def _add_multi_sma(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add multiple SMAs."""
        for period in self.config.SMA_PERIODS:
            df = sma(df, period)
        return df
    
    def _add_multi_ema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add multiple EMAs."""
        for period in self.config.EMA_PERIODS:
            df = ema(df, period)
        return df
    
    def get_feature_names(self, categories: list[str] | None = None) -> list[str]:
        """
        Get list of all feature names that would be generated.
        
        Args:
            categories: Categories to include
        
        Returns:
            List of feature names
        """
        features = []
        all_categories = categories or ["momentum", "trend", "volatility", "volume"]
        
        if "momentum" in all_categories:
            features.extend([
                f"rsi_{self.config.RSI_PERIOD}",
                "macd", "macd_signal", "macd_histogram",
                "stoch_k", "stoch_d",
                f"williams_r_{self.config.WILLIAMS_PERIOD}",
                f"roc_{self.config.ROC_PERIOD}",
                f"cci_{self.config.CCI_PERIOD}",
                f"cmo_{self.config.CMO_PERIOD}",
                "ultimate_oscillator",
                "tsi", "tsi_signal",
            ])
        
        if "trend" in all_categories:
            features.extend([f"sma_{p}" for p in self.config.SMA_PERIODS])
            features.extend([f"ema_{p}" for p in self.config.EMA_PERIODS])
            features.extend([
                "adx", "plus_di", "minus_di",
                "supertrend", "supertrend_direction",
                "aroon_up", "aroon_down", "aroon_oscillator",
                "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_chikou",
                "ichimoku_senkou_a", "ichimoku_senkou_b",
                "psar", "psar_direction",
            ])
        
        if "volatility" in all_categories:
            features.extend([
                "bb_upper", "bb_middle", "bb_lower", "bb_pctb", "bb_width",
                f"atr_{self.config.ATR_PERIOD}", f"atr_{self.config.ATR_PERIOD}_pct",
                "kc_upper", "kc_middle", "kc_lower",
                "dc_upper", "dc_middle", "dc_lower",
                f"natr_{self.config.ATR_PERIOD}",
                "hv_20",
            ])
        
        if "volume" in all_categories:
            features.extend([
                "obv", "vwap",
                f"mfi_{self.config.MFI_PERIOD}",
                "ad_line",
                f"cmf_{self.config.CMF_PERIOD}",
                f"force_index_{self.config.FORCE_PERIOD}",
                "vwma_20",
                "eom_14",
            ])
        
        return features


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def add_all_indicators(
    df: pl.DataFrame,
    config: IndicatorConfig | None = None,
) -> pl.DataFrame:
    """
    Convenience function to add all technical indicators.
    
    Args:
        df: Input DataFrame with OHLCV data
        config: Indicator configuration
    
    Returns:
        DataFrame with all indicators added
    """
    ti = TechnicalIndicators(config)
    return ti.add_all(df)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "IndicatorConfig",
    "DEFAULT_CONFIG",
    # Momentum
    "rsi",
    "macd",
    "stochastic",
    "williams_r",
    "roc",
    "momentum",
    "cci",
    "cmo",
    "ultimate_oscillator",
    "tsi",
    # Trend
    "sma",
    "ema",
    "wma",
    "dema",
    "tema",
    "adx",
    "supertrend",
    "aroon",
    "ichimoku",
    "parabolic_sar",
    # Volatility
    "bollinger_bands",
    "atr",
    "keltner_channels",
    "donchian_channels",
    "natr",
    "historical_volatility",
    # Volume
    "obv",
    "vwap",
    "mfi",
    "ad_line",
    "cmf",
    "force_index",
    "vwma",
    "eom",
    "volume_profile",
    # Class
    "TechnicalIndicators",
    "add_all_indicators",
]