#!/usr/bin/env python3
"""
Run Backtest Script
===================

Command-line interface for running backtests on the algo trading platform.

Usage:
    python scripts/run_backtest.py                              # Interactive
    python scripts/run_backtest.py --symbol AAPL MSFT           # Multiple symbols
    python scripts/run_backtest.py --strategy trend_following   # Specific strategy
    python scripts/run_backtest.py --all-symbols                # All available data

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl

from config.settings import get_settings, get_logger, configure_logging, TimeFrame
from core.types import BacktestError
from data.loader import CSVLoader, load_all_symbols
from data.processor import DataProcessor
from backtesting.engine import (
    BacktestEngine,
    BacktestConfig,
    ReportGenerator,
    run_backtest,
    quick_backtest,
)
from strategies import (
    StrategyRegistry,
    StrategyFactory,
    list_strategies,
    create_strategy,
    TrendFollowingStrategy,
    TrendFollowingConfig,
    MeanReversionStrategy,
    MeanReversionConfig,
    BreakoutStrategy,
    BreakoutConfig,
)


# Initialize
settings = get_settings()
configure_logging(settings)
logger = get_logger(__name__)


def discover_symbols(data_path: Path) -> list[str]:
    """Discover available symbols from CSV files."""
    csv_files = list(data_path.glob("*_15min.csv")) + list(data_path.glob("*_1h.csv"))
    symbols = []
    for f in csv_files:
        # Extract symbol from filename (e.g., AAPL_15min.csv -> AAPL)
        symbol = f.stem.split("_")[0]
        if symbol not in symbols:
            symbols.append(symbol)
    return sorted(symbols)


def load_data(
    symbols: list[str],
    data_path: Path,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> dict[str, pl.DataFrame]:
    """Load and process data for symbols."""
    processor = DataProcessor()
    
    data = {}
    for symbol in symbols:
        # Try different filename patterns
        patterns = [
            f"{symbol}_15min.csv",
            f"{symbol}_1h.csv",
            f"{symbol}.csv",
        ]
        
        loaded = False
        for pattern in patterns:
            file_path = data_path / pattern
            if file_path.exists():
                logger.info(f"Loading {symbol} from {file_path}")
                
                # Load directly with polars
                try:
                    df = pl.read_csv(file_path)
                    
                    # Ensure timestamp column is datetime
                    if "timestamp" in df.columns:
                        df = df.with_columns([
                            pl.col("timestamp").str.to_datetime().alias("timestamp")
                        ])
                    
                    # Process data
                    df = processor.process(df)
                    
                    # Filter by date range
                    if start_date:
                        df = df.filter(pl.col("timestamp") >= start_date)
                    if end_date:
                        df = df.filter(pl.col("timestamp") <= end_date)
                    
                    if len(df) > 0:
                        data[symbol] = df
                        logger.info(f"  Loaded {len(df)} bars for {symbol}")
                        loaded = True
                        break
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    continue
        
        if not loaded:
            logger.warning(f"Could not find data for {symbol}")
    
    return data


def create_strategy_instance(
    strategy_name: str,
    symbols: list[str],
    **kwargs: Any,
) -> Any:
    """Create strategy instance from name."""
    
    # Built-in strategy mapping
    strategy_map = {
        "trend_following": (TrendFollowingStrategy, TrendFollowingConfig),
        "mean_reversion": (MeanReversionStrategy, MeanReversionConfig),
        "breakout": (BreakoutStrategy, BreakoutConfig),
    }
    
    if strategy_name in strategy_map:
        strategy_cls, config_cls = strategy_map[strategy_name]
        config = config_cls(symbols=symbols, **kwargs)
        return strategy_cls(config)
    
    # Try registry
    try:
        return create_strategy(strategy_name, {"symbols": symbols, **kwargs})
    except Exception as e:
        logger.error(f"Failed to create strategy '{strategy_name}': {e}")
        raise


def run_single_backtest(
    symbols: list[str],
    strategy_name: str,
    data_path: Path,
    initial_capital: float = 100_000,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run a single backtest."""
    
    logger.info("=" * 60)
    logger.info("BACKTEST CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Strategy: {strategy_name}")
    logger.info(f"Initial Capital: ${initial_capital:,.2f}")
    logger.info(f"Start Date: {start_date or 'All available'}")
    logger.info(f"End Date: {end_date or 'All available'}")
    logger.info("=" * 60)
    
    # Load data
    data = load_data(symbols, data_path, start_date, end_date)
    
    if not data:
        raise BacktestError("No data loaded. Check your data path and symbols.")
    
    # Create strategy
    strategy = create_strategy_instance(strategy_name, list(data.keys()))
    
    # Configure backtest
    config = BacktestConfig(
        initial_capital=initial_capital,
        commission_pct=settings.backtest.commission_pct,
        slippage_pct=settings.backtest.slippage_pct,
        allow_shorting=settings.backtest.allow_shorting,
    )
    
    # Create and run engine
    engine = BacktestEngine(config)
    
    for symbol, df in data.items():
        engine.add_data(symbol, df)
    
    engine.add_strategy(strategy)
    
    logger.info("\nRunning backtest...")
    report = engine.run(start_date, end_date)
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    
    print_results(report)
    
    # Generate reports
    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get equity curve and trades
        equity_curve = engine.get_equity_curve()
        trades = engine.get_trades()
        
        # Generate report
        generator = ReportGenerator(report, equity_curve, trades)
        
        # Save JSON
        json_path = output_path / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        generator.to_json(json_path)
        logger.info(f"\nReport saved to: {json_path}")
        
        # Print text summary
        generator.print_summary()
    
    return {
        "report": report,
        "equity_curve": engine.get_equity_curve(),
        "trades": engine.get_trades(),
        "signals": engine.get_signals(),
    }


def print_results(report: Any) -> None:
    """Print backtest results to console."""
    print("\n" + "-" * 60)
    print("PERFORMANCE SUMMARY")
    print("-" * 60)
    
    print(f"\n  Returns:")
    print(f"    Total Return:      {report.total_return_pct:>10.2%}")
    print(f"    Annualized Return: {report.annualized_return:>10.2%}")
    
    print(f"\n  Risk:")
    print(f"    Volatility:        {report.annualized_volatility:>10.2%}")
    print(f"    Max Drawdown:      {report.max_drawdown:>10.2%}")
    print(f"    VaR (95%):         {report.var_95:>10.2%}")
    
    print(f"\n  Risk-Adjusted:")
    print(f"    Sharpe Ratio:      {report.sharpe_ratio:>10.2f}")
    print(f"    Sortino Ratio:     {report.sortino_ratio:>10.2f}")
    print(f"    Calmar Ratio:      {report.calmar_ratio:>10.2f}")
    
    print(f"\n  Trading:")
    print(f"    Total Trades:      {report.trade_stats.total_trades:>10}")
    print(f"    Win Rate:          {report.trade_stats.win_rate:>10.2%}")
    print(f"    Profit Factor:     {report.trade_stats.profit_factor:>10.2f}")
    print(f"    Avg Trade:         ${report.trade_stats.avg_trade:>9.2f}")
    
    print(f"\n  Capital:")
    print(f"    Initial:           ${report.initial_capital:>12,.2f}")
    print(f"    Final:             ${report.final_capital:>12,.2f}")
    print(f"    P&L:               ${report.total_return:>12,.2f}")
    
    print("-" * 60)


def interactive_mode() -> dict[str, Any]:
    """Run interactive backtest configuration."""
    data_path = Path(settings.data.storage_path)
    
    print("\n" + "=" * 60)
    print("  BACKTEST WIZARD")
    print("=" * 60)
    
    # Discover available symbols
    available_symbols = discover_symbols(data_path)
    
    if not available_symbols:
        print(f"\n  No data files found in: {data_path}")
        print("  Please add CSV files with format: SYMBOL_15min.csv")
        return {}
    
    print(f"\n  Available symbols ({len(available_symbols)}):")
    for i, symbol in enumerate(available_symbols, 1):
        print(f"    {i}. {symbol}")
    
    # Select symbols
    print("\n  Enter symbol numbers (comma-separated) or 'all':")
    selection = input("  > ").strip()
    
    if selection.lower() == "all":
        symbols = available_symbols
    else:
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(",")]
            symbols = [available_symbols[i] for i in indices if 0 <= i < len(available_symbols)]
        except (ValueError, IndexError):
            print("  Invalid selection. Using first symbol.")
            symbols = [available_symbols[0]]
    
    print(f"\n  Selected symbols: {symbols}")
    
    # Select strategy
    strategies = list_strategies()
    print(f"\n  Available strategies:")
    for i, strat in enumerate(strategies, 1):
        print(f"    {i}. {strat}")
    
    print("\n  Enter strategy number [default: 1]:")
    strat_selection = input("  > ").strip() or "1"
    
    try:
        strat_idx = int(strat_selection) - 1
        strategy_name = strategies[strat_idx] if 0 <= strat_idx < len(strategies) else "trend_following"
    except ValueError:
        strategy_name = "trend_following"
    
    print(f"\n  Selected strategy: {strategy_name}")
    
    # Capital
    print("\n  Enter initial capital [default: 100000]:")
    capital_input = input("  > ").strip() or "100000"
    try:
        capital = float(capital_input)
    except ValueError:
        capital = 100000
    
    print(f"\n  Initial capital: ${capital:,.2f}")
    
    # Confirm
    print("\n" + "-" * 60)
    print("  Press Enter to start backtest or 'q' to quit...")
    if input("  > ").strip().lower() == "q":
        return {}
    
    # Run backtest
    output_path = Path("backtesting/reports")
    
    return run_single_backtest(
        symbols=symbols,
        strategy_name=strategy_name,
        data_path=data_path,
        initial_capital=capital,
        output_path=output_path,
    )


def main(args: argparse.Namespace | None = None) -> int:
    """Main entry point."""
    
    if args is None:
        parser = argparse.ArgumentParser(description="Run Backtest")
        parser.add_argument("--symbol", "-s", type=str, nargs="+", help="Symbols to backtest")
        parser.add_argument("--strategy", "-t", type=str, default="trend_following", help="Strategy")
        parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
        parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
        parser.add_argument("--capital", "-c", type=float, default=100000, help="Initial capital")
        parser.add_argument("--output", "-o", type=str, help="Output directory")
        parser.add_argument("--all-symbols", action="store_true", help="Use all available symbols")
        parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
        parser.add_argument("--list-strategies", action="store_true", help="List available strategies")
        args = parser.parse_args()
    
    # List strategies
    if hasattr(args, 'list_strategies') and args.list_strategies:
        print("\nAvailable strategies:")
        for strat in list_strategies():
            print(f"  - {strat}")
        return 0
    
    # Interactive mode
    if (hasattr(args, 'interactive') and args.interactive) or \
       (not hasattr(args, 'symbol') or args.symbol is None):
        try:
            results = interactive_mode()
            return 0 if results else 1
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            return 1
    
    # Command-line mode
    data_path = Path(settings.data.storage_path)
    
    # Get symbols
    if hasattr(args, 'all_symbols') and args.all_symbols:
        symbols = discover_symbols(data_path)
    elif hasattr(args, 'symbol') and args.symbol:
        symbols = args.symbol
    else:
        symbols = discover_symbols(data_path)[:1]  # Default to first symbol
    
    if not symbols:
        logger.error("No symbols specified or found.")
        return 1
    
    # Parse dates
    start_date = None
    end_date = None
    
    if hasattr(args, 'start') and args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
    if hasattr(args, 'end') and args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    
    # Output path
    output_path = Path(args.output) if hasattr(args, 'output') and args.output else Path("backtesting/reports")
    
    try:
        run_single_backtest(
            symbols=symbols,
            strategy_name=args.strategy if hasattr(args, 'strategy') else "trend_following",
            data_path=data_path,
            initial_capital=args.capital if hasattr(args, 'capital') else 100000,
            start_date=start_date,
            end_date=end_date,
            output_path=output_path,
        )
        return 0
    except Exception as e:
        logger.exception(f"Backtest failed: {e}")
        return 1
def validate_backtest_metrics(report) -> bool:
    """
    Validate that metrics are mathematically possible.
    Call this after calculating metrics.
    """
    errors = []
    
    total_ret = report.total_return_pct
    sharpe = report.sharpe_ratio
    sortino = report.sortino_ratio
    max_dd = report.max_drawdown
    var_95 = report.var_95
    n_trades = report.trade_stats.total_trades
    
    # Check 1: Sharpe sign
    if total_ret > 0.10 and sharpe < 0:
        errors.append(
            f"IMPOSSIBLE: Sharpe={sharpe:.2f} is negative with "
            f"positive return={total_ret:.2%}"
        )
    
    # Check 2: Sortino sign  
    if total_ret > 0.10 and sortino < 0:
        errors.append(
            f"IMPOSSIBLE: Sortino={sortino:.2f} is negative with "
            f"positive return={total_ret:.2%}"
        )
    
    # Check 3: Zero risk metrics with trades
    if n_trades > 100:
        if var_95 == 0:
            errors.append(f"IMPOSSIBLE: VaR=0% with {n_trades} trades")
        if abs(max_dd) < 0.0001:
            errors.append(f"UNLIKELY: Max DD={max_dd:.4%} with {n_trades} trades")
    
    # Check 4: Annualized vs Total consistency
    # For 4.5 years, (1+total)^(1/4.5)-1 = annual
    if total_ret > 1.0:  # > 100%
        expected_annual = (1 + total_ret) ** (1/4.5) - 1
        if report.annualized_return < expected_annual * 0.5:  # Allow 50% margin
            errors.append(
                f"INCONSISTENT: Annual={report.annualized_return:.2%} "
                f"too low for total={total_ret:.2%} over 4.5 years"
            )
    
    if errors:
        print("\n" + "=" * 70)
        print("⚠️  METRICS VALIDATION FAILED - CHECK periods_per_year!")
        print("=" * 70)
        for e in errors:
            print(f"  ❌ {e}")
        print(f"\n  Your data needs: periods_per_year = 15794")
        print(f"  Current setting may be: 252 (wrong!)")
        print("=" * 70 + "\n")
        return False
    
    print("✅ Metrics validation passed")
    return True


if __name__ == "__main__":
    sys.exit(main())