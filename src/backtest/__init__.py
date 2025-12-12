"""
Backtesting Engine Module
JPMorgan-Level Historical Simulation and Performance Analysis
"""

from .engine import (
    BacktestEngine, BacktestConfig, BacktestResult,
    EventDrivenBacktester, VectorizedBacktester
)
from .metrics import (
    PerformanceMetrics, RiskMetrics, TradeMetrics,
    MetricsCalculator, ReportGenerator
)

__all__ = [
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResult',
    'EventDrivenBacktester',
    'VectorizedBacktester',
    'PerformanceMetrics',
    'RiskMetrics',
    'TradeMetrics',
    'MetricsCalculator',
    'ReportGenerator'
]
