"""
BACKTEST MODULE
Professional Backtesting & Walk-Forward Analysis Suite

IMPORTANT: 
- Root-level backtest.py was renamed to backtest_engine.py
- This avoids Python naming conflict with backtest/ folder

Usage:
    from backtest import ProfessionalBacktester, WalkForwardOptimizer
    
    # Single symbol backtest
    backtester = ProfessionalBacktester(symbol="AAPL")
    results = await backtester.run()
    
    # Walk-forward optimization
    optimizer = WalkForwardOptimizer(symbol="AAPL")
    wf_results = await optimizer.run(param_grid={...})
"""

import sys
from pathlib import Path

# Add parent directory to path to import from root-level files
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import from root-level backtest_engine.py (renamed from backtest.py)
try:
    from backtest_engine import (
        ProfessionalBacktester,
        BacktestMetrics,
        TradeRecord
    )
    _backtest_available = True
except ImportError as e:
    import warnings
    warnings.warn(
        f"Could not import from backtest_engine.py: {e}\n"
        "Please rename 'backtest.py' to 'backtest_engine.py'"
    )
    ProfessionalBacktester = None
    BacktestMetrics = None
    TradeRecord = None
    _backtest_available = False

# Import from root-level portfolio_backtest.py
try:
    from portfolio_backtest import (
        MultiAssetPortfolioBacktest,
        PortfolioBacktestResult
    )
    _portfolio_available = True
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import from portfolio_backtest.py: {e}")
    MultiAssetPortfolioBacktest = None
    PortfolioBacktestResult = None
    _portfolio_available = False

# Import from walk_forward module
from backtest.walk_forward import (
    WalkForwardOptimizer,
    WalkForwardSummary,
    WindowMetrics,
    WindowMode,
    OptimizationConfig,
    run_walk_forward
)

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