"""
Backtesting Engine Module
JPMorgan-Level Historical Simulation and Performance Analysis

Advanced Features:
- Microstructure Simulation (stochastic latency, partial fills)
- Deflated Sharpe Ratio for overfitting detection
- Institutional-grade execution modeling
- Walk-forward optimization with CPCV
"""

from .engine import (
    BacktestEngine, BacktestConfig, BacktestResult,
    EventDrivenBacktester, VectorizedBacktester,
    MicrostructureSimulator, InstitutionalBacktestEngine,
    WalkForwardOptimizer, DynamicTransactionCostModel
)
from .metrics import (
    PerformanceMetrics, RiskMetrics, TradeMetrics,
    MetricsCalculator, ReportGenerator,
    DeflatedSharpeRatio, SharpeRatioStatistics,
    calculate_dsr, calculate_psr, calculate_dsr_with_trials_estimation
)

__all__ = [
    # Core Engine
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResult',
    'EventDrivenBacktester',
    'VectorizedBacktester',

    # Institutional Engine
    'InstitutionalBacktestEngine',
    'MicrostructureSimulator',
    'WalkForwardOptimizer',
    'DynamicTransactionCostModel',

    # Metrics
    'PerformanceMetrics',
    'RiskMetrics',
    'TradeMetrics',
    'MetricsCalculator',
    'ReportGenerator',

    # Statistical Analysis
    'DeflatedSharpeRatio',
    'SharpeRatioStatistics',
    'calculate_dsr',
    'calculate_psr',
    'calculate_dsr_with_trials_estimation'
]
