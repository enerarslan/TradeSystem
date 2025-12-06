"""
Portfolio Module
================

Portfolio management and optimization for the algorithmic trading platform.
Provides portfolio optimization, allocation, and rebalancing functionality.

Components:
- optimizer: Portfolio optimization algorithms

Author: Algo Trading Platform
License: MIT
"""

from portfolio.optimizer import (
    # Enums
    OptimizationMethod,
    RebalanceFrequency,
    # Classes
    OptimizationConfig,
    OptimizationResult,
    PortfolioOptimizer,
    # Functions
    mean_variance_optimize,
    min_variance_portfolio,
    max_sharpe_portfolio,
    risk_parity_portfolio,
    equal_weight_portfolio,
    inverse_volatility_portfolio,
    hrp_portfolio,
)

__all__ = [
    # Enums
    "OptimizationMethod",
    "RebalanceFrequency",
    # Classes
    "OptimizationConfig",
    "OptimizationResult",
    "PortfolioOptimizer",
    # Functions
    "mean_variance_optimize",
    "min_variance_portfolio",
    "max_sharpe_portfolio",
    "risk_parity_portfolio",
    "equal_weight_portfolio",
    "inverse_volatility_portfolio",
    "hrp_portfolio",
]