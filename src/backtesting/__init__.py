"""
Backtesting module for AlphaTrade system.

This module provides:
- Core backtesting engine (vectorized)
- Event-driven backtesting engine (institutional-grade)
- Performance metrics
- Result analysis
- Report generation
- Market impact modeling (Almgren-Chriss)
- Monte Carlo analysis and statistical tests

Designed for institutional requirements:
- Microsecond-precision event simulation
- Full order lifecycle tracking
- Realistic execution simulation
- Comprehensive audit trails
"""

from src.backtesting.engine import (
    BacktestEngine,
    BacktestResult,
    run_backtest,
    VectorizedBacktest,
    TransactionCostModel,
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
from src.backtesting.event_engine import (
    EventDrivenEngine,
    EventEngineConfig,
    EventEngineResult,
    Position,
    Portfolio,
    ExecutionSimulator,
    SimpleExecutionSimulator,
    OrderBookExecutionSimulator,
    run_event_backtest,
)
from src.backtesting.market_impact import (
    AlmgrenChrissModel,
    DynamicSpreadModel,
    LatencySimulator,
)
from src.backtesting.monte_carlo import (
    MonteCarloAnalyzer,
    StatisticalTests,
)

__all__ = [
    # Vectorized backtesting
    "BacktestEngine",
    "BacktestResult",
    "run_backtest",
    "VectorizedBacktest",
    "TransactionCostModel",
    # Event-driven backtesting
    "EventDrivenEngine",
    "EventEngineConfig",
    "EventEngineResult",
    "Position",
    "Portfolio",
    "ExecutionSimulator",
    "SimpleExecutionSimulator",
    "OrderBookExecutionSimulator",
    "run_event_backtest",
    # Metrics and analysis
    "PerformanceMetrics",
    "calculate_all_metrics",
    "BacktestAnalyzer",
    "analyze_trades",
    "analyze_positions",
    # Market impact
    "AlmgrenChrissModel",
    "DynamicSpreadModel",
    "LatencySimulator",
    # Monte Carlo
    "MonteCarloAnalyzer",
    "StatisticalTests",
]
