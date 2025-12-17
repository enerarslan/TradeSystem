"""
Backtesting module for AlphaTrade system.

This module provides:
- Core backtesting engine
- Performance metrics
- Result analysis
- Report generation
"""

from src.backtesting.engine import (
    BacktestEngine,
    BacktestResult,
    run_backtest,
)
from src.backtesting.metrics import (
    PerformanceMetrics,
    calculate_all_metrics,
)
from src.backtesting.analysis import (
    BacktestAnalyzer,
    analyze_trades,
    analyze_positions,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "run_backtest",
    "PerformanceMetrics",
    "calculate_all_metrics",
    "BacktestAnalyzer",
    "analyze_trades",
    "analyze_positions",
]
