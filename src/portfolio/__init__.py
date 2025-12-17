"""
Portfolio management module for AlphaTrade system.

This module provides institutional-grade portfolio optimization:
- Mean-Variance Optimization (MVO, Risk Parity, HRP, etc.)
- Black-Litterman model for combining views with market equilibrium
- Rebalancing logic
- Asset allocation

Reference:
    - Black, F. and Litterman, R. (1992) - "Global Portfolio Optimization"
    - "Quantitative Portfolio Management" by Isichenko (2021)
"""

from src.portfolio.optimizer import (
    PortfolioOptimizer,
    mean_variance_optimize,
    minimum_variance_portfolio,
    maximum_sharpe_portfolio,
    risk_parity_portfolio,
    hierarchical_risk_parity,
)
from src.portfolio.rebalancer import (
    Rebalancer,
    RebalanceSignal,
)
from src.portfolio.allocation import (
    AssetAllocator,
    apply_constraints,
)
from src.portfolio.black_litterman import (
    BlackLittermanOptimizer,
    BlackLittermanResult,
    BlackLittermanWithML,
    ViewSpecification,
    create_absolute_view,
    create_relative_view,
)

__all__ = [
    # Traditional optimization
    "PortfolioOptimizer",
    "mean_variance_optimize",
    "minimum_variance_portfolio",
    "maximum_sharpe_portfolio",
    "risk_parity_portfolio",
    "hierarchical_risk_parity",
    # Black-Litterman
    "BlackLittermanOptimizer",
    "BlackLittermanResult",
    "BlackLittermanWithML",
    "ViewSpecification",
    "create_absolute_view",
    "create_relative_view",
    # Rebalancing
    "Rebalancer",
    "RebalanceSignal",
    # Allocation
    "AssetAllocator",
    "apply_constraints",
]
