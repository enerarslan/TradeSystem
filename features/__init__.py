"""
Features Module
===============

Feature engineering for the algorithmic trading platform.
Includes technical indicators, statistical features, and ML features.

Components:
- technical: 50+ technical indicators (momentum, trend, volatility, volume)
- statistical: Statistical and derived features
- pipeline: Feature orchestration and management

Author: Algo Trading Platform
License: MIT
"""

# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

from features.technical import (
    # Configuration
    IndicatorConfig,
    DEFAULT_CONFIG,
    # Class
    TechnicalIndicators,
    add_all_indicators,
    # Momentum indicators
    rsi,
    macd,
    stochastic,
    williams_r,
    roc,
    momentum,
    cci,
    cmo,
    ultimate_oscillator,
    tsi,
    # Trend indicators
    sma,
    ema,
    wma,
    dema,
    tema,
    adx,
    supertrend,
    aroon,
    ichimoku,
    parabolic_sar,
    # Volatility indicators
    bollinger_bands,
    atr,
    keltner_channels,
    donchian_channels,
    natr,
    historical_volatility,
    # Volume indicators
    obv,
    vwap,
    mfi,
    ad_line,
    cmf,
    force_index,
    vwma,
    eom,
    volume_profile,
)

# =============================================================================
# STATISTICAL FEATURES
# =============================================================================

from features.statistical import (
    # Configuration
    StatisticalConfig,
    DEFAULT_STAT_CONFIG,
    MarketRegime,
    # Class
    StatisticalFeatures,
    add_statistical_features,
    # Returns
    log_returns,
    simple_returns,
    cumulative_returns,
    excess_returns,
    # Rolling statistics
    rolling_stats,
    rolling_higher_moments,
    rolling_quantiles,
    # Momentum features
    price_momentum,
    trend_strength,
    price_acceleration,
    # Volatility features
    volatility_features,
    volatility_regime,
    intraday_volatility,
    # Correlation
    rolling_correlation,
    rolling_beta,
    # Regime detection
    detect_market_regime,
    trend_regime,
    # Distribution
    zscore,
    percentile_rank,
    distance_from_extremes,
    # Mean reversion
    mean_reversion_features,
    half_life,
)

# =============================================================================
# FEATURE PIPELINE
# =============================================================================

from features.pipeline import (
    # Configuration
    FeatureCategory,
    FeatureConfig,
    create_default_config,
    # Pipeline
    FeaturePipeline,
    # Convenience functions
    create_default_pipeline,
    generate_all_features,
    create_ml_dataset,
)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Technical Configuration
    "IndicatorConfig",
    "DEFAULT_CONFIG",
    # Technical Class
    "TechnicalIndicators",
    "add_all_indicators",
    # Momentum Indicators
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
    # Trend Indicators
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
    # Volatility Indicators
    "bollinger_bands",
    "atr",
    "keltner_channels",
    "donchian_channels",
    "natr",
    "historical_volatility",
    # Volume Indicators
    "obv",
    "vwap",
    "mfi",
    "ad_line",
    "cmf",
    "force_index",
    "vwma",
    "eom",
    "volume_profile",
    # Statistical Configuration
    "StatisticalConfig",
    "DEFAULT_STAT_CONFIG",
    "MarketRegime",
    # Statistical Class
    "StatisticalFeatures",
    "add_statistical_features",
    # Returns
    "log_returns",
    "simple_returns",
    "cumulative_returns",
    "excess_returns",
    # Rolling Statistics
    "rolling_stats",
    "rolling_higher_moments",
    "rolling_quantiles",
    # Momentum Features
    "price_momentum",
    "trend_strength",
    "price_acceleration",
    # Volatility Features
    "volatility_features",
    "volatility_regime",
    "intraday_volatility",
    # Correlation
    "rolling_correlation",
    "rolling_beta",
    # Regime Detection
    "detect_market_regime",
    "trend_regime",
    # Distribution
    "zscore",
    "percentile_rank",
    "distance_from_extremes",
    # Mean Reversion
    "mean_reversion_features",
    "half_life",
    # Pipeline Configuration
    "FeatureCategory",
    "FeatureConfig",
    "create_default_config",
    # Pipeline
    "FeaturePipeline",
    "create_default_pipeline",
    "generate_all_features",
    "create_ml_dataset",
]