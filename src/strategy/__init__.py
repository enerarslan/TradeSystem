"""Strategy modules for AlphaTrade System"""

from .base_strategy import BaseStrategy, Signal, SignalType
from .momentum import MomentumStrategy, TrendFollowingStrategy
from .mean_reversion import MeanReversionStrategy, StatArbStrategy
from .ml_strategy import MLStrategy, EnsembleMLStrategy

__all__ = [
    'BaseStrategy',
    'Signal',
    'SignalType',
    'MomentumStrategy',
    'TrendFollowingStrategy',
    'MeanReversionStrategy',
    'StatArbStrategy',
    'MLStrategy',
    'EnsembleMLStrategy'
]
