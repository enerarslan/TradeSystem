"""
FEATURES MODULE
JPMorgan-Style Feature Engineering Library

Bu modül 150+ profesyonel feature içerir:
- Technical Indicators (40+)
- Price Features (30+)
- Volume Features (25+)
- Time Features (30+)

Kullanım:
    from features import FeatureEngineeringPipeline, create_features
    
    # Hızlı kullanım
    features = create_features(df, symbol="AAPL")
    
    # Detaylı kullanım
    pipeline = FeatureEngineeringPipeline()
    features = pipeline.fit_transform(df, symbol="AAPL")
    pipeline.print_feature_report()
"""

# Technical Indicators
from features.technical import (
    TechnicalIndicators,
    IndicatorCategory,
    IndicatorResult
)

# Price Features
from features.price_features import (
    PriceFeatures,
    CandlePattern
)

# Time Features
from features.time_features import (
    TimeFeatures,
    MarketSession,
    MARKET_SESSIONS
)

# Volume Features
from features.volume_features import (
    VolumeFeatures
)

# Pipeline
from features.pipeline import (
    FeatureEngineeringPipeline,
    FeatureConfig,
    FeatureNormalizer,
    FeatureStats,
    create_features,
    get_default_config
)


# Convenience function for quick feature generation
def generate_all_features(df, symbol: str = "UNKNOWN", normalize: bool = True):
    """
    Tüm feature'ları tek satırda üret.
    
    Args:
        df: OHLCV DataFrame
        symbol: Sembol adı
        normalize: Normalization uygula
    
    Returns:
        pd.DataFrame: Feature matrix
    
    Example:
        features = generate_all_features(df, "AAPL")
    """
    return create_features(df, symbol, normalize)


__all__ = [
    # Technical
    'TechnicalIndicators',
    'IndicatorCategory',
    'IndicatorResult',
    
    # Price
    'PriceFeatures',
    'CandlePattern',
    
    # Time
    'TimeFeatures',
    'MarketSession',
    'MARKET_SESSIONS',
    
    # Volume
    'VolumeFeatures',
    
    # Pipeline
    'FeatureEngineeringPipeline',
    'FeatureConfig',
    'FeatureNormalizer',
    'FeatureStats',
    'create_features',
    'get_default_config',
    'generate_all_features'
]