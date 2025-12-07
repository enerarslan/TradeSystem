"""
Features Module
===============

Feature engineering for the algorithmic trading platform.

Modules:
- technical: 50+ technical indicators (momentum, trend, volatility, volume)
- statistical: Statistical features (returns, correlations, regime)
- pipeline: Feature orchestration and management
- advanced: JPMorgan-level features (Triple Barrier, Meta-labeling, etc.)

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
    
    # Generator class
    TechnicalIndicators,
    
    # Convenience functions
    add_all_indicators,
    add_momentum_indicators,
    add_trend_indicators,
    add_volatility_indicators,
    add_volume_indicators,
)

# =============================================================================
# STATISTICAL FEATURES
# =============================================================================

from features.statistical import (
    # Configuration
    StatisticalConfig,
    DEFAULT_STAT_CONFIG,
    
    # Generator class
    StatisticalFeatures,
    
    # Convenience functions
    add_statistical_features,
    add_return_features,
    add_rolling_features,
)

# =============================================================================
# FEATURE PIPELINE
# =============================================================================

from features.pipeline import (
    # Configuration
    FeatureCategory,
    FeatureConfig,
    create_default_config,
    
    # Pipeline class
    FeaturePipeline,
    
    # Convenience functions
    create_default_pipeline,
    generate_all_features,
    create_ml_dataset,
)

# =============================================================================
# ADVANCED FEATURES (JPMorgan-level)
# =============================================================================

from features.advanced import (
    # Enums
    BarrierType,
    LabelType,
    
    # Configurations
    TripleBarrierConfig,
    MetaLabelConfig,
    AdvancedFeatureConfig,
    
    # Triple Barrier Method
    TripleBarrierLabeler,
    
    # Meta-Labeling
    MetaLabeler,
    
    # Fractional Differentiation
    FractionalDifferentiation,
    
    # Microstructure Features
    MicrostructureFeatures,
    
    # Calendar Features
    CalendarFeatures,
    
    # Feature Interactions
    FeatureInteractions,
    
    # Advanced Pipeline
    AdvancedFeaturePipeline,
)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # === Technical - Config ===
    "IndicatorConfig",
    "DEFAULT_CONFIG",
    
    # === Technical - Generator ===
    "TechnicalIndicators",
    
    # === Technical - Functions ===
    "add_all_indicators",
    "add_momentum_indicators",
    "add_trend_indicators",
    "add_volatility_indicators",
    "add_volume_indicators",
    
    # === Statistical - Config ===
    "StatisticalConfig",
    "DEFAULT_STAT_CONFIG",
    
    # === Statistical - Generator ===
    "StatisticalFeatures",
    
    # === Statistical - Functions ===
    "add_statistical_features",
    "add_return_features",
    "add_rolling_features",
    
    # === Pipeline - Config ===
    "FeatureCategory",
    "FeatureConfig",
    "create_default_config",
    
    # === Pipeline - Class ===
    "FeaturePipeline",
    
    # === Pipeline - Functions ===
    "create_default_pipeline",
    "generate_all_features",
    "create_ml_dataset",
    
    # === Advanced - Enums ===
    "BarrierType",
    "LabelType",
    
    # === Advanced - Configs ===
    "TripleBarrierConfig",
    "MetaLabelConfig",
    "AdvancedFeatureConfig",
    
    # === Advanced - Triple Barrier ===
    "TripleBarrierLabeler",
    
    # === Advanced - Meta-Labeling ===
    "MetaLabeler",
    
    # === Advanced - Fractional Differentiation ===
    "FractionalDifferentiation",
    
    # === Advanced - Microstructure ===
    "MicrostructureFeatures",
    
    # === Advanced - Calendar ===
    "CalendarFeatures",
    
    # === Advanced - Interactions ===
    "FeatureInteractions",
    
    # === Advanced - Pipeline ===
    "AdvancedFeaturePipeline",
]