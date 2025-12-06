#!/usr/bin/env python3
"""
============================================================================
ALPHATRADE - BACKTEST RUNNER
============================================================================
The main entry point for running backtests on your 46 stock dataset.

Usage:
    python run_backtest.py                    # Run with default settings
    python run_backtest.py --mode portfolio   # Run portfolio backtest
    python run_backtest.py --mode walkforward # Run walk-forward optimization
    python run_backtest.py --symbol AAPL      # Run single stock
    python run_backtest.py --help             # Show all options

============================================================================
"""

import asyncio
import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def print_banner():
    """Print startup banner"""
    print("\n" + "=" * 70)
    print("   🏦 ALPHATRADE BACKTEST SYSTEM")
    print("   JPMorgan-Style Quantitative Research")
    print("=" * 70)
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")


async def run_single_backtest(symbol: str, capital: float, start_date=None, end_date=None):
    """
    Run backtest on a single stock.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL")
        capital: Initial capital
        start_date: Optional start date
        end_date: Optional end date
    """
    # Import here to avoid circular imports
    from backtest import ProfessionalBacktester
    from strategies.momentum import AdvancedMomentum
    
    print(f"📊 Single Stock Backtest: {symbol}")
    print(f"💰 Capital: ${capital:,.2f}")
    print("-" * 50)
    
    backtester = ProfessionalBacktester(
        symbol=symbol,
        initial_capital=capital,
        commission_pct=0.001,    # 0.1% commission
        slippage_pct=0.0005,     # 0.05% slippage
        use_risk_management=True
    )
    
    results = await backtester.run(
        strategy_class=AdvancedMomentum,
        strategy_params={
            'fast_period': 10,
            'slow_period': 30,
            'rsi_period': 14,
            'min_confidence': 0.3,              # LOWERED from 0.6
            'use_regime_filter': False,          # ADDED - disable
            'use_volume_confirmation': False     # ADDED - disable
        },
        start_date=start_date,
        end_date=end_date
    )
    
    return results


async def run_portfolio_backtest(capital: float, max_positions: int, allocation: str):
    """
    Run backtest on all 46 stocks together.
    
    Args:
        capital: Initial capital
        max_positions: Maximum concurrent positions
        allocation: Allocation mode (equal_weight, risk_parity, etc.)
    """
    from backtest.portfolio import MultiAssetPortfolioBacktest
    
    print(f"📊 Portfolio Backtest: All Stocks")
    print(f"💰 Capital: ${capital:,.2f}")
    print(f"📈 Max Positions: {max_positions}")
    print(f"⚖️ Allocation: {allocation}")
    print("-" * 50)
    
    backtester = MultiAssetPortfolioBacktest(
        initial_capital=capital,
        allocation_mode=allocation,
        rebalance_frequency="monthly",
        max_positions=max_positions,
        commission_pct=0.001,
        use_risk_management=True
    )
    
    results = await backtester.run()
    
    return results


async def run_walk_forward(symbol: str, capital: float, train_days: int, test_days: int):
    """
    Run walk-forward optimization on a single stock.
    
    Args:
        symbol: Stock symbol
        capital: Initial capital
        train_days: Training period in days
        test_days: Testing period in days
    """
    from backtest.walk_forward import WalkForwardOptimizer
    from strategies.momentum import AdvancedMomentum
    
    print(f"📊 Walk-Forward Optimization: {symbol}")
    print(f"💰 Capital: ${capital:,.2f}")
    print(f"📅 Train Period: {train_days} days")
    print(f"📅 Test Period: {test_days} days")
    print("-" * 50)
    
    optimizer = WalkForwardOptimizer(
        symbol=symbol,
        initial_capital=capital,
        train_period_days=train_days,
        test_period_days=test_days,
        window_mode="rolling"
    )
    
    # Parameter grid to optimize
    param_grid = {
        'rsi_period': [10, 14, 20],
        'rsi_oversold': [25, 30, 35],
        'rsi_overbought': [65, 70, 75],
        'volume_threshold': [1.2, 1.5, 2.0]
    }
    
    results = await optimizer.run(
        strategy_class=AdvancedMomentum,
        param_grid=param_grid
    )
    
    # Export results
    optimizer.export_results(f"wf_{symbol}_{datetime.now().strftime('%Y%m%d')}")
    
    return results


async def run_all_stocks_sequential(capital: float):
    """
    Run backtest on each stock individually (sequential).
    Useful for comparing performance across stocks.
    """
    from data.csv_loader import LocalCSVLoader
    from backtest import ProfessionalBacktester
    from strategies.momentum import AdvancedMomentum
    
    print(f"📊 Sequential Backtest: All Stocks")
    print(f"💰 Capital per stock: ${capital:,.2f}")
    print("-" * 50)
    
    # Discover all symbols
    loader = LocalCSVLoader()
    storage_path = Path("data/storage")
    
    symbols = []
    for file in storage_path.glob("*_15min.csv"):
        symbol = file.stem.replace("_15min", "")
        symbols.append(symbol)
    
    print(f"📁 Found {len(symbols)} symbols: {', '.join(symbols[:5])}...")
    
    results_summary = []
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Testing {symbol}...")
        
        try:
            backtester = ProfessionalBacktester(
                symbol=symbol,
                initial_capital=capital,
                use_risk_management=True
            )
            
            result = await backtester.run(
                strategy_class=AdvancedMomentum,
                strategy_params={'rsi_period': 14}
            )
            
            if result:
                results_summary.append({
                    'symbol': symbol,
                    'return': result.total_return,
                    'sharpe': result.sharpe_ratio,
                    'max_dd': result.max_drawdown,
                    'trades': result.total_trades,
                    'win_rate': result.win_rate
                })
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("   📊 RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Symbol':<10} {'Return':>10} {'Sharpe':>10} {'MaxDD':>10} {'Trades':>8} {'WinRate':>10}")
    print("-" * 70)
    
    for r in sorted(results_summary, key=lambda x: x['return'], reverse=True):
        print(f"{r['symbol']:<10} {r['return']:>+9.2%} {r['sharpe']:>10.2f} "
              f"{r['max_dd']:>9.2%} {r['trades']:>8} {r['win_rate']:>9.1%}")
    
    return results_summary


def list_available_symbols():
    """List all available symbols in data/storage"""
    storage_path = Path("data/storage")
    
    if not storage_path.exists():
        print("❌ data/storage folder not found!")
        return []
    
    symbols = []
    for file in storage_path.glob("*.csv"):
        symbol = file.stem.replace("_15min", "")
        file_size = file.stat().st_size / (1024 * 1024)  # MB
        symbols.append((symbol, file_size))
    
    print("\n📁 Available Symbols in data/storage:")
    print("-" * 40)
    for symbol, size in sorted(symbols):
        print(f"   {symbol:<15} ({size:.1f} MB)")
    print("-" * 40)
    print(f"   Total: {len(symbols)} symbols")
    
    return [s[0] for s in symbols]


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="AlphaTrade Backtest Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_backtest.py --mode single --symbol AAPL
    python run_backtest.py --mode portfolio --capital 100000
    python run_backtest.py --mode walkforward --symbol MSFT
    python run_backtest.py --mode all
    python run_backtest.py --list-symbols
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["single", "portfolio", "walkforward", "all"],
        default="single",
        help="Backtest mode (default: single)"
    )
    
    parser.add_argument(
        "--symbol",
        default="AAPL",
        help="Stock symbol for single/walkforward mode (default: AAPL)"
    )
    
    parser.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="Initial capital (default: 100000)"
    )
    
    parser.add_argument(
        "--max-positions",
        type=int,
        default=20,
        help="Max positions for portfolio mode (default: 20)"
    )
    
    parser.add_argument(
        "--allocation",
        choices=["equal_weight", "risk_parity", "markowitz"],
        default="equal_weight",
        help="Portfolio allocation mode (default: equal_weight)"
    )
    
    parser.add_argument(
        "--train-days",
        type=int,
        default=180,
        help="Training period for walk-forward (default: 180)"
    )
    
    parser.add_argument(
        "--test-days",
        type=int,
        default=30,
        help="Test period for walk-forward (default: 30)"
    )
    
    parser.add_argument(
        "--list-symbols",
        action="store_true",
        help="List all available symbols and exit"
    )
    
    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_args()
    
    print_banner()
    
    # List symbols and exit
    if args.list_symbols:
        list_available_symbols()
        return
    
    # Run selected mode
    try:
        if args.mode == "single":
            results = await run_single_backtest(
                symbol=args.symbol,
                capital=args.capital
            )
            
        elif args.mode == "portfolio":
            results = await run_portfolio_backtest(
                capital=args.capital,
                max_positions=args.max_positions,
                allocation=args.allocation
            )
            
        elif args.mode == "walkforward":
            results = await run_walk_forward(
                symbol=args.symbol,
                capital=args.capital,
                train_days=args.train_days,
                test_days=args.test_days
            )
            
        elif args.mode == "all":
            results = await run_all_stocks_sequential(
                capital=args.capital
            )
        
        print("\n" + "=" * 70)
        print("   ✅ BACKTEST COMPLETE")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())