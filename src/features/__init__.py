"""
Feature engineering module for AlphaTrade system.

This module provides comprehensive feature engineering:
- Technical indicators (50+ indicators)
- Statistical features
- Cross-sectional features
- Lagged features
- Feature pipeline for ML
- Fractional differentiation for stationarity
- Cointegration and statistical arbitrage features
- Macroeconomic features from FRED

Designed for institutional requirements:
- Point-in-time accuracy
- Memory-preserving transformations
- Regime detection
"""

from src.features.technical.indicators import TechnicalIndicators
from src.features.pipeline import (
    FeaturePipeline,
    FeatureProcessor,
    create_feature_matrix,
)
from src.features.fractional_diff import (
    frac_diff_ffd,
    find_min_d,
    test_stationarity_adf,
    FractionalDiffTransformer,
)
from src.features.cointegration import (
    CointegrationAnalyzer,
    OrnsteinUhlenbeckEstimator,
)
from src.features.macro_features import (
    FREDClient,
    FREDConfig,
    MacroIndicator,
    MacroFeatureGenerator,
    EconomicRegimeDetector,
    align_macro_to_price_data,
)

__all__ = [
    # Technical indicators
    "TechnicalIndicators",
    # Pipeline
    "FeaturePipeline",
    "FeatureProcessor",
    "create_feature_matrix",
    # Fractional differentiation
    "frac_diff_ffd",
    "find_min_d",
    "test_stationarity_adf",
    "FractionalDiffTransformer",
    # Statistical arbitrage
    "CointegrationAnalyzer",
    "OrnsteinUhlenbeckEstimator",
    # Macro features
    "FREDClient",
    "FREDConfig",
    "MacroIndicator",
    "MacroFeatureGenerator",
    "EconomicRegimeDetector",
    "align_macro_to_price_data",
]
