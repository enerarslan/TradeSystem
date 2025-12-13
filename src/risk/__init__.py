"""
Risk Management Module
JPMorgan-Level Risk Controls and Portfolio Management

Advanced Features:
- Hierarchical Risk Parity (HRP) Allocation
- Meta-Labeled Kelly Criterion Position Sizing
- Dynamic Bet Sizing with ML Probability
- Advanced Risk Parity Optimization
"""

from .position_sizer import (
    PositionSizer, KellyCriterion, VolatilityPositionSizer,
    RiskParityPositionSizer, OptimalFPositionSizer,
    MetaLabeledKelly, VolatilityInverseKelly, create_position_sizer
)
from .risk_manager import (
    RiskManager, RiskLimits, RiskMetrics, PreTradeRiskCheck
)
from .portfolio import (
    Portfolio, Position, PortfolioManager, PortfolioOptimizer,
    HierarchicalRiskParity
)
from .allocation import (
    HierarchicalRiskParityAllocator, HRPConfig, AllocationResult,
    CovarianceEstimator, HRPBacktestIntegration,
    DistanceMetric, LinkageMethod, RiskMeasure,
    compute_hrp_weights, compute_correlation_distance
)

__all__ = [
    # Position Sizers
    'PositionSizer',
    'KellyCriterion',
    'VolatilityPositionSizer',
    'RiskParityPositionSizer',
    'OptimalFPositionSizer',
    'MetaLabeledKelly',
    'VolatilityInverseKelly',
    'create_position_sizer',

    # Risk Management
    'RiskManager',
    'RiskLimits',
    'RiskMetrics',
    'PreTradeRiskCheck',

    # Portfolio Management
    'Portfolio',
    'Position',
    'PortfolioManager',
    'PortfolioOptimizer',
    'HierarchicalRiskParity',

    # HRP Allocation
    'HierarchicalRiskParityAllocator',
    'HRPConfig',
    'AllocationResult',
    'CovarianceEstimator',
    'HRPBacktestIntegration',
    'DistanceMetric',
    'LinkageMethod',
    'RiskMeasure',
    'compute_hrp_weights',
    'compute_correlation_distance'
]
