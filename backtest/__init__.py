"""
BACKTEST MODULE
Professional Backtesting & Walk-Forward Analysis Suite

This module provides:
- ProfessionalBacktester: Single-symbol backtesting engine
- MultiAssetPortfolioBacktest: Multi-asset portfolio backtesting
- WalkForwardOptimizer: Walk-forward optimization and analysis
- Performance metrics and reporting

Usage:
    from backtest import ProfessionalBacktester, WalkForwardOptimizer
    
    # Single symbol backtest
    backtester = ProfessionalBacktester(symbol="AAPL")
    results = await backtester.run()
    
    # Walk-forward optimization
    optimizer = WalkForwardOptimizer(symbol="AAPL")
    wf_results = await optimizer.run(param_grid={...})
"""

# Import from main backtest module (backtest.py at project root)
# Note: The main backtest.py should be moved to backtest/engine.py
# For now, we import from the walk_forward module

from backtest.walk_forward import (
    WalkForwardOptimizer,
    WalkForwardSummary,
    WindowMetrics,
    WindowMode,
    OptimizationConfig,
    run_walk_forward
)

__all__ = [
    # Walk-Forward
    'WalkForwardOptimizer',
    'WalkForwardSummary',
    'WindowMetrics',
    'WindowMode',
    'OptimizationConfig',
    'run_walk_forward',
]