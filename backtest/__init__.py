"""
BACKTEST MODULE
Professional Backtesting & Walk-Forward Analysis Suite

Usage:
    from backtest import ProfessionalBacktester, MultiAssetPortfolioBacktest
    
    # Single symbol backtest
    backtester = ProfessionalBacktester(symbol="AAPL")
    results = await backtester.run()
    
    # Portfolio backtest
    portfolio_bt = MultiAssetPortfolioBacktest(initial_capital=100000)
    results = await portfolio_bt.run()
"""

# Import from backtest_engine.py (inside this folder)
try:
    from backtest.backtest_engine import (
        ProfessionalBacktester,
        BacktestMetrics,
        TradeRecord
    )
    _backtest_available = True
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import from backtest.backtest_engine: {e}")
    ProfessionalBacktester = None
    BacktestMetrics = None
    TradeRecord = None
    _backtest_available = False

# Import from portfolio.py (inside this folder)
try:
    from backtest.portfolio import (
        MultiAssetPortfolioBacktest,
        PortfolioBacktestResult
    )
    _portfolio_available = True
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import from backtest.portfolio: {e}")
    MultiAssetPortfolioBacktest = None
    PortfolioBacktestResult = None
    _portfolio_available = False

# Import from walk_forward.py (inside this folder)
try:
    from backtest.walk_forward import (
        WalkForwardOptimizer,
        WalkForwardSummary,
        WindowMetrics,
        WindowMode,
        OptimizationConfig,
        run_walk_forward
    )
    _walkforward_available = True
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import from backtest.walk_forward: {e}")
    WalkForwardOptimizer = None
    WalkForwardSummary = None
    WindowMetrics = None
    WindowMode = None
    OptimizationConfig = None
    run_walk_forward = None
    _walkforward_available = False

__all__ = [
    # Single Stock Backtest
    'ProfessionalBacktester',
    'BacktestMetrics',
    'TradeRecord',
    
    # Portfolio Backtest
    'MultiAssetPortfolioBacktest',
    'PortfolioBacktestResult',
    
    # Walk-Forward
    'WalkForwardOptimizer',
    'WalkForwardSummary',
    'WindowMetrics',
    'WindowMode',
    'OptimizationConfig',
    'run_walk_forward',
]


def check_availability():
    """Check which backtest components are available"""
    print("Backtest Module Status:")
    print(f"  - ProfessionalBacktester: {'✅' if _backtest_available else '❌'}")
    print(f"  - MultiAssetPortfolioBacktest: {'✅' if _portfolio_available else '❌'}")
    print(f"  - WalkForwardOptimizer: {'✅' if _walkforward_available else '❌'}")


if __name__ == "__main__":
    check_availability()