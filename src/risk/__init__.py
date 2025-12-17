"""
Risk management module for AlphaTrade system.

This module provides comprehensive risk management:
- Position sizing algorithms
- VaR calculations
- Drawdown controls
- Correlation analysis
"""

from src.risk.position_sizing import (
    PositionSizer,
    fixed_fraction,
    kelly_criterion,
    volatility_target,
    risk_parity_weights,
)
from src.risk.var_models import (
    VaRCalculator,
    calculate_var,
    calculate_cvar,
)
from src.risk.drawdown import (
    DrawdownController,
    calculate_drawdown,
    calculate_max_drawdown,
)
from src.risk.correlation import (
    CorrelationAnalyzer,
    calculate_correlation_matrix,
)

__all__ = [
    "PositionSizer",
    "fixed_fraction",
    "kelly_criterion",
    "volatility_target",
    "risk_parity_weights",
    "VaRCalculator",
    "calculate_var",
    "calculate_cvar",
    "DrawdownController",
    "calculate_drawdown",
    "calculate_max_drawdown",
    "CorrelationAnalyzer",
    "calculate_correlation_matrix",
]
