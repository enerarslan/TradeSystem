"""
Features Module
===============

Feature engineering for the algorithmic trading platform.
Includes technical indicators, statistical features, and ML features.

Components:
- technical: 50+ technical indicators
- statistical: Statistical and derived features
- pipeline: Feature orchestration

Author: Algo Trading Platform
License: MIT
"""

from features.technical import (
    TechnicalIndicators,
    add_all_indicators,
    # Momentum
    rsi,
    macd,
    stochastic,
    williams_r,
    roc,
    momentum,
    # Trend
    sma,
    ema,
    wma,
    dema,
    tema,
    adx,
    supertrend,
    # Volatility
    bollinger_bands,
    atr,
    keltner_channels,
    donchian_channels,
    # Volume
    obv,
    vwap,
    mfi,
    ad_line,
    cmf,
)
from features.statistical import (
    StatisticalFeatures,
    add_statistical_features,
    log_returns,
    rolling_stats,
    price_momentum,
    volatility_features,
)
from features.pipeline import (
    FeaturePipeline,
    FeatureConfig,
    create_default_pipeline,
)

__all__ = [
    # Technical
    "TechnicalIndicators",
    "add_all_indicators",
    "rsi", "macd", "stochastic", "williams_r", "roc", "momentum",
    "sma", "ema", "wma", "dema", "tema", "adx", "supertrend",
    "bollinger_bands", "atr", "keltner_channels", "donchian_channels",
    "obv", "vwap", "mfi", "ad_line", "cmf",
    # Statistical
    "StatisticalFeatures",
    "add_statistical_features",
    "log_returns", "rolling_stats", "price_momentum", "volatility_features",
    # Pipeline
    "FeaturePipeline",
    "FeatureConfig",
    "create_default_pipeline",
]