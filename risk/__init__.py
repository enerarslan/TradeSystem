"""
Risk Module
===========

Risk management for the algorithmic trading platform.
Provides position sizing, VaR calculations, risk limits, and drawdown monitoring.

Components:
- manager: Comprehensive risk management system

Author: Algo Trading Platform
License: MIT
"""

from risk.manager import (
    # Enums
    RiskLevel,
    PositionSizingMethod,
    # Classes
    RiskConfig,
    RiskMetrics,
    RiskManager,
    # Functions
    calculate_var,
    calculate_cvar,
    calculate_position_size,
    kelly_criterion,
    optimal_f,
)

__all__ = [
    # Enums
    "RiskLevel",
    "PositionSizingMethod",
    # Classes
    "RiskConfig",
    "RiskMetrics",
    "RiskManager",
    # Functions
    "calculate_var",
    "calculate_cvar",
    "calculate_position_size",
    "kelly_criterion",
    "optimal_f",
]