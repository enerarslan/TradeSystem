"""
Backtesting Module
==================

Comprehensive backtesting framework for the algorithmic trading platform.
Provides realistic simulation of trading strategies with proper execution
modeling, position tracking, and performance analysis.

Components:
- engine: Event-driven backtesting engine
- execution: Slippage, commission, and fill models  
- metrics: Performance metrics and analytics

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

# =============================================================================
# ENGINE IMPORTS
# =============================================================================

from backtesting.engine import (
    # Enums
    BacktestMode,
    OrderFillMode,
    # Configuration
    BacktestConfig,
    # Core classes
    PortfolioTracker,
    BacktestEngine,
    # Walk-forward
    WalkForwardResult,
    WalkForwardAnalyzer,
    # Reporting
    ReportGenerator,
    # Convenience functions
    run_backtest,
    quick_backtest,
)

# =============================================================================
# EXECUTION IMPORTS
# =============================================================================

from backtesting.execution import (
    # Slippage Models
    SlippageModel,
    NoSlippage,
    FixedSlippage,
    PercentageSlippage,
    VolumeSlippage,
    SpreadSlippage,
    MarketImpactSlippage,
    # Commission Models
    CommissionModel,
    NoCommission,
    FixedCommission,
    PerShareCommission,
    PercentageCommission,
    TieredCommission,
    IBKRCommission,
    # Fill Models
    FillModel,
    ImmediateFill,
    OHLCFill,
    PartialFill,
    ProbabilisticFill,
    # Simulator
    FillResult,
    ExecutionSimulator,
    # Factory
    create_realistic_simulator,
    create_zero_cost_simulator,
)

# =============================================================================
# METRICS IMPORTS
# =============================================================================

from backtesting.metrics import (
    # Constants
    TRADING_DAYS_PER_YEAR,
    RISK_FREE_RATE,
    PERIODS_PER_YEAR,
    # Return metrics
    calculate_returns,
    total_return,
    annualized_return,
    rolling_returns,
    monthly_returns,
    yearly_returns,
    # Volatility metrics
    volatility,
    downside_volatility,
    upside_volatility,
    # Drawdown metrics
    calculate_drawdown_series,
    max_drawdown,
    max_drawdown_duration,
    average_drawdown,
    drawdown_details,
    ulcer_index,
    pain_index,
    # VaR metrics
    var_historical,
    var_parametric,
    var_cornish_fisher,
    cvar,
    # Risk-adjusted metrics
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    omega_ratio,
    information_ratio,
    treynor_ratio,
    gain_to_pain_ratio,
    ulcer_performance_index,
    # Trade statistics
    TradeStats,
    calculate_trade_stats,
    # Distribution metrics
    skewness,
    kurtosis,
    tail_ratio,
    common_sense_ratio,
    outlier_ratio,
    # Classes
    PerformanceReport,
    MetricsCalculator,
)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # === Engine ===
    "BacktestMode",
    "OrderFillMode",
    "BacktestConfig",
    "PortfolioTracker",
    "BacktestEngine",
    "WalkForwardResult",
    "WalkForwardAnalyzer",
    "ReportGenerator",
    "run_backtest",
    "quick_backtest",
    
    # === Execution - Slippage ===
    "SlippageModel",
    "NoSlippage",
    "FixedSlippage",
    "PercentageSlippage",
    "VolumeSlippage",
    "SpreadSlippage",
    "MarketImpactSlippage",
    
    # === Execution - Commission ===
    "CommissionModel",
    "NoCommission",
    "FixedCommission",
    "PerShareCommission",
    "PercentageCommission",
    "TieredCommission",
    "IBKRCommission",
    
    # === Execution - Fill ===
    "FillModel",
    "ImmediateFill",
    "OHLCFill",
    "PartialFill",
    "ProbabilisticFill",
    "FillResult",
    "ExecutionSimulator",
    "create_realistic_simulator",
    "create_zero_cost_simulator",
    
    # === Metrics - Constants ===
    "TRADING_DAYS_PER_YEAR",
    "RISK_FREE_RATE",
    "PERIODS_PER_YEAR",
    
    # === Metrics - Returns ===
    "calculate_returns",
    "total_return",
    "annualized_return",
    "rolling_returns",
    "monthly_returns",
    "yearly_returns",
    
    # === Metrics - Volatility ===
    "volatility",
    "downside_volatility",
    "upside_volatility",
    
    # === Metrics - Drawdown ===
    "calculate_drawdown_series",
    "max_drawdown",
    "max_drawdown_duration",
    "average_drawdown",
    "drawdown_details",
    "ulcer_index",
    "pain_index",
    
    # === Metrics - VaR ===
    "var_historical",
    "var_parametric",
    "var_cornish_fisher",
    "cvar",
    
    # === Metrics - Risk-Adjusted ===
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "omega_ratio",
    "information_ratio",
    "treynor_ratio",
    "gain_to_pain_ratio",
    "ulcer_performance_index",
    
    # === Metrics - Trade Stats ===
    "TradeStats",
    "calculate_trade_stats",
    
    # === Metrics - Distribution ===
    "skewness",
    "kurtosis",
    "tail_ratio",
    "common_sense_ratio",
    "outlier_ratio",
    
    # === Metrics - Classes ===
    "PerformanceReport",
    "MetricsCalculator",
]