"""
Regime Detection Module for AlphaTrade System.

This module provides institutional-grade market regime detection using:
1. Hidden Markov Models (HMM) for state classification
2. Volatility regime detection (GARCH-based)
3. Correlation regime detection
4. Structural break detection (CUSUM, Bai-Perron)

Regime-aware strategies typically outperform by 0.5-1.0 Sharpe ratio
by adapting parameters to market conditions.

Reference:
    "Machine Learning for Asset Managers" by Lopez de Prado (2020)
"""

from .hmm_regime import HMMRegimeDetector, RegimeState
from .volatility_regime import VolatilityRegimeDetector, GARCHModel
from .correlation_regime import CorrelationRegimeDetector
from .structural_breaks import StructuralBreakDetector, CUSUMTest

__all__ = [
    "HMMRegimeDetector",
    "RegimeState",
    "VolatilityRegimeDetector",
    "GARCHModel",
    "CorrelationRegimeDetector",
    "StructuralBreakDetector",
    "CUSUMTest",
]
