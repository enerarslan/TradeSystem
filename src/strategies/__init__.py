"""
Trading strategies module for AlphaTrade system.

This module provides various trading strategy implementations:
- Base strategy class
- Multi-factor momentum strategy
- Mean reversion strategy
- ML-based alpha strategy
- Volatility breakout strategy
- Ensemble strategy
"""

from src.strategies.base import BaseStrategy, Signal, SignalType
from src.strategies.momentum.multi_factor_momentum import MultiFactorMomentumStrategy
from src.strategies.mean_reversion.mean_reversion import MeanReversionStrategy
from src.strategies.ml_based.ml_alpha import MLAlphaStrategy
from src.strategies.multi_factor.volatility_breakout import VolatilityBreakoutStrategy
from src.strategies.ensemble import EnsembleStrategy

__all__ = [
    "BaseStrategy",
    "Signal",
    "SignalType",
    "MultiFactorMomentumStrategy",
    "MeanReversionStrategy",
    "MLAlphaStrategy",
    "VolatilityBreakoutStrategy",
    "EnsembleStrategy",
]
