"""
Feature engineering module for AlphaTrade system.

This module provides comprehensive feature engineering:
- Technical indicators (50+ indicators)
- Statistical features
- Cross-sectional features
- Lagged features
- Feature pipeline for ML
"""

from src.features.technical.indicators import TechnicalIndicators
from src.features.pipeline import (
    FeaturePipeline,
    FeatureProcessor,
    create_feature_matrix,
)

__all__ = [
    "TechnicalIndicators",
    "FeaturePipeline",
    "FeatureProcessor",
    "create_feature_matrix",
]
