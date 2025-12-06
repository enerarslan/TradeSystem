#!/usr/bin/env python3
"""
================================================================================
ALPHATRADE - PROFESSIONAL BACKTEST RUNNER V3
================================================================================
JPMorgan Quantitative Research Division - Production Grade

Complete backtest runner with support for:
- Single stock backtesting
- Portfolio backtesting (46 stocks)
- Walk-forward optimization
- Multiple strategy types
- Comprehensive reporting

Usage:
    python run_backtest.py --symbol AAPL --capital 100000
    python run_backtest.py --symbol AAPL --capital 100000 --strategy ml
    python run_backtest.py --mode portfolio --capital 500000
    python run_backtest.py --mode walkforward --symbol AAPL
    python run_backtest.py --mode sequential --capital 100000

================================================================================
Author: AlphaTrade Quantitative Team
Version: 3.0.0
================================================================================
"""

import asyncio
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


# ============================================================================
# BANNER & DISPLAY
# ============================================================================

def print_banner():
    """Print startup banner"""
    print("\n" + "=" * 70)
    print("   🏦 ALPHATRADE BACKTEST SYSTEM V3")
    print("   JPMorgan-Style Quantitative Research")
    print("=" * 70)
    print(f"   📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")


def print_config(args):
    """Print configuration summary"""
    print("📋 CONFIGURATION")
    print("-" * 50)
    print(f"   Mode          : {args.mode.upper()}")
    print(f"   Strategy      : {args.strategy.upper()}")
    print(f"   Capital       : ${args.capital:,.2f}")
    
    if args.mode == "single":
        print(f"   Symbol        : {args.symbol}")
    elif args.mode == "portfolio":
        print(f"   Max Positions : {args.max_positions}")
        print(f"   Allocation    : {args.allocation}")
    elif args.mode == "walkforward":
        print(f"   Symbol        : {args.symbol}")
        print(f"   Train Days    : {args.train_days}")
        print(f"   Test Days     : {args.test_days}")
    
    print("-" * 50 + "\n")


def print_summary_table(results: Dict[str, Dict], title: str = "RESULTS SUMMARY"):
    """Print formatted summary table"""
    print("\n" + "=" * 80)
    print(f"   {title}")
    print("=" * 80)
    print(f"   {'Symbol':<8} {'Return':>10} {'Sharpe':>8} {'Win Rate':>10} {'Trades':>8} {'MaxDD':>10}")
    print("-" * 80)
    
    for symbol, data in sorted(results.items(), key=lambda x: x[1].get('return', 0), reverse=True):
        ret = data.get('return', 0)
        sharpe = data.get('sharpe', 0)
        win_rate = data.get('win_rate', 0)
        trades = data.get('trades', 0)
        max_dd = data.get('max_drawdown', 0)
        
        print(f"   {symbol:<8} {ret:>+9.2f}% {sharpe:>8.2f} {win_rate:>9.1f}% {trades:>8} {max_dd:>9.2f}%")
    
    print("=" * 80)


# ============================================================================
# STRATEGY FACTORY
# ============================================================================

def get_strategy_class(strategy_name: str):
    """Get strategy class by name"""
    from strategies.momentum import AdvancedMomentum, MLMomentumStrategy
    
    strategies = {
        'advanced': AdvancedMomentum,
        'momentum': AdvancedMomentum,
        'ml': MLMomentumStrategy,
        'mlmomentum': MLMomentumStrategy,
        'hybrid': MLMomentumStrategy,
    }
    
    return strategies.get(strategy_name.lower(), AdvancedMomentum)


def get_strategy_params(strategy_name: str) -> Dict[str, Any]:
    """Get strategy parameters based on strategy type"""
    
    base_params = {
        'fast_period': 8,
        'slow_period': 21,
        'rsi_period': 14,
        'min_confidence': 0.3,
        'signal_threshold': 0.05,
        'use_regime_filter': True,
        'use_volume_confirmation': False,
        'lookback': 200,
    }
    
    if strategy_name.lower() in ['ml', 'mlmomentum', 'hybrid']:
        base_params.update({
            'ml_model_type': 'xgboost',
            'ml_model_path': None,
            'ml_weight': 0.4,
            'online_learning': True,
            'min_agreement': 0.3,
        })
    
    return base_params


# ============================================================================
# SINGLE STOCK BACKTEST
# ============================================================================

async def run_single_backtest(
    symbol: str, 
    capital: float, 
    strategy_name: str = "advanced",
    start_date: Optional[datetime] = None, 
    end_date: Optional[datetime] = None,
    verbose: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Run backtest on a single stock.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL")
        capital: Initial capital
        strategy_name: Strategy to use
        start_date: Optional start date filter
        end_date: Optional end date filter
        verbose: Print detailed output
    
    Returns:
        Dict with backtest results or None on error
    """
    from backtest import ProfessionalBacktester
    from risk.optimized_configs import RiskProfiles
    
    if verbose:
        print(f"📊 Single Stock Backtest: {symbol}")
        print(f"💰 Capital: ${capital:,.2f}")
        print(f"📈 Strategy: {strategy_name}")
        print("-" * 50)
    
    try:
        # Create backtester
        backtester = ProfessionalBacktester(
            symbol=symbol,
            initial_capital=capital,
            commission_pct=0.001,    # 0.1% commission
            slippage_pct=0.0005,     # 0.05% slippage
            use_risk_management=True
        )
        
        # Use MODERATE risk profile for more trades
        backtester.risk_manager.config = RiskProfiles.MODERATE
        
        # Get strategy class and params
        strategy_class = get_strategy_class(strategy_name)
        strategy_params = get_strategy_params(strategy_name)
        
        if verbose:
            print(f"🎯 Using: {strategy_class.__name__}")
            print(f"⚙️  Params: signal_threshold={strategy_params['signal_threshold']}, min_confidence={strategy_params['min_confidence']}")
        
        # Run backtest
        results = await backtester.run(
            strategy_class=strategy_class,
            strategy_params=strategy_params,
            start_date=start_date,
            end_date=end_date
        )
        
        if results:
            return {
                'symbol': symbol,
                'return': results.total_return,
                'annual_return': results.annualized_return,
                'sharpe': results.sharpe_ratio,
                'sortino': results.sortino_ratio,
                'calmar': results.calmar_ratio,
                'max_drawdown': results.max_drawdown,
                'trades': results.total_trades,
                'win_rate': results.win_rate,
                'profit_factor': results.profit_factor,
                'avg_win': results.avg_win,
                'avg_loss': results.avg_loss,
                'final_balance': results.final_balance,
            }
        
        return None
        
    except Exception as e:
        if verbose:
            print(f"❌ Error backtesting {symbol}: {e}")
            import traceback
            traceback.print_exc()
        return None


# ============================================================================
# PORTFOLIO BACKTEST
# ============================================================================

async def run_portfolio_backtest(
    capital: float,
    max_positions: int = 10,
    allocation: str = "equal_weight",
    strategy_name: str = "advanced"
) -> Optional[Dict[str, Any]]:
    """
    Run backtest on full portfolio (all available stocks).
    """
    from backtest.portfolio import MultiAssetPortfolioBacktest
    
    print(f"📊 Portfolio Backtest: All Stocks")
    print(f"💰 Capital: ${capital:,.2f}")
    print(f"📈 Max Positions: {max_positions}")
    print(f"⚖️ Allocation: {allocation}")
    print(f"🎯 Strategy: {strategy_name}")
    print("-" * 50)
    
    try:
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
        
    except Exception as e:
        print(f"❌ Portfolio backtest error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# WALK-FORWARD OPTIMIZATION
# ============================================================================

async def run_walk_forward(
    symbol: str,
    capital: float,
    train_days: int = 180,
    test_days: int = 30,
    strategy_name: str = "advanced"
) -> Optional[Dict[str, Any]]:
    """
    Run walk-forward optimization on a single stock.
    """
    from backtest.walk_forward import WalkForwardOptimizer
    
    print(f"📊 Walk-Forward Optimization: {symbol}")
    print(f"💰 Capital: ${capital:,.2f}")
    print(f"📅 Train Period: {train_days} days")
    print(f"📅 Test Period: {test_days} days")
    print(f"🎯 Strategy: {strategy_name}")
    print("-" * 50)
    
    try:
        strategy_class = get_strategy_class(strategy_name)
        
        optimizer = WalkForwardOptimizer(
            symbol=symbol,
            initial_capital=capital,
            train_period_days=train_days,
            test_period_days=test_days,
            window_mode="rolling"
        )
        
        # Parameter grid to optimize
        param_grid = {
            'fast_period': [5, 8, 10, 13],
            'slow_period': [15, 21, 30],
            'rsi_period': [10, 14, 20],
            'min_confidence': [0.2, 0.3, 0.4],
            'signal_threshold': [0.03, 0.05, 0.08],
        }
        
        results = await optimizer.run(
            strategy_class=strategy_class,
            param_grid=param_grid
        )
        
        # Export results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        optimizer.export_results(f"wf_{symbol}_{timestamp}")
        
        return results
        
    except Exception as e:
        print(f"❌ Walk-forward error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# SEQUENTIAL BACKTEST (ALL STOCKS)
# ============================================================================

async def run_sequential_backtest(
    capital: float,
    strategy_name: str = "advanced",
    max_symbols: Optional[int] = None
) -> Dict[str, Dict]:
    """
    Run backtest on each stock individually.
    """
    storage_path = Path("data/storage")
    csv_files = list(storage_path.glob("*_15min.csv"))
    
    if not csv_files:
        print("❌ No data files found in data/storage/")
        return {}
    
    symbols = [f.stem.replace("_15min", "") for f in csv_files]
    
    if max_symbols:
        symbols = symbols[:max_symbols]
    
    print(f"📊 Sequential Backtest: {len(symbols)} stocks")
    print(f"💰 Capital per stock: ${capital:,.2f}")
    print(f"🎯 Strategy: {strategy_name}")
    print("-" * 50)
    
    results = {}
    start_time = time.time()
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\r⚡ Progress: [{i}/{len(symbols)}] {symbol}...", end="", flush=True)
        
        try:
            result = await run_single_backtest(
                symbol=symbol,
                capital=capital,
                strategy_name=strategy_name,
                verbose=False
            )
            
            if result:
                results[symbol] = result
                
        except Exception as e:
            pass  # Continue on error
    
    print()  # Newline after progress
    
    elapsed = time.time() - start_time
    print(f"\n✅ Completed in {elapsed:.1f}s")
    
    # Print summary
    if results:
        print_summary_table(results, "SEQUENTIAL BACKTEST RESULTS")
        
        # Calculate aggregate stats
        returns = [r['return'] for r in results.values()]
        win_rates = [r['win_rate'] for r in results.values() if r['trades'] > 0]
        
        print(f"\n📊 AGGREGATE STATISTICS:")
        print(f"   Stocks Tested   : {len(results)}")
        print(f"   Profitable      : {sum(1 for r in returns if r > 0)} ({sum(1 for r in returns if r > 0)/len(returns)*100:.1f}%)")
        print(f"   Avg Return      : {sum(returns)/len(returns):.2f}%")
        print(f"   Best Return     : {max(returns):.2f}%")
        print(f"   Worst Return    : {min(returns):.2f}%")
        if win_rates:
            print(f"   Avg Win Rate    : {sum(win_rates)/len(win_rates):.1f}%")
    
    return results


# ============================================================================
# COMPARISON MODE
# ============================================================================

async def run_strategy_comparison(
    symbol: str,
    capital: float
) -> Dict[str, Dict]:
    """
    Compare different strategies on the same symbol.
    """
    strategies = ['advanced', 'ml']
    
    print(f"📊 Strategy Comparison: {symbol}")
    print(f"💰 Capital: ${capital:,.2f}")
    print(f"🎯 Strategies: {', '.join(strategies)}")
    print("-" * 50)
    
    results = {}
    
    for strategy in strategies:
        print(f"\n🔄 Testing {strategy}...")
        
        result = await run_single_backtest(
            symbol=symbol,
            capital=capital,
            strategy_name=strategy,
            verbose=False
        )
        
        if result:
            results[strategy] = result
    
    # Print comparison
    if results:
        print("\n" + "=" * 60)
        print("   STRATEGY COMPARISON RESULTS")
        print("=" * 60)
        print(f"   {'Strategy':<15} {'Return':>10} {'Sharpe':>8} {'Win Rate':>10} {'Trades':>8}")
        print("-" * 60)
        
        for strategy, data in results.items():
            print(f"   {strategy:<15} {data['return']:>+9.2f}% {data['sharpe']:>8.2f} {data['win_rate']:>9.1f}% {data['trades']:>8}")
        
        print("=" * 60)
    
    return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="AlphaTrade Professional Backtest Runner V3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single stock backtest with default strategy
  python run_backtest.py --symbol AAPL --capital 100000

  # Single stock with ML strategy  
  python run_backtest.py --symbol AAPL --capital 100000 --strategy ml

  # Portfolio backtest
  python run_backtest.py --mode portfolio --capital 500000 --max-positions 15

  # Walk-forward optimization
  python run_backtest.py --mode walkforward --symbol AAPL --train-days 180 --test-days 30

  # Sequential test all stocks
  python run_backtest.py --mode sequential --capital 100000

  # Compare strategies
  python run_backtest.py --mode compare --symbol AAPL --capital 100000
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--mode", 
        type=str, 
        default="single",
        choices=["single", "portfolio", "walkforward", "sequential", "compare"],
        help="Backtest mode (default: single)"
    )
    
    # Common parameters
    parser.add_argument(
        "--symbol", 
        type=str, 
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
        "--strategy",
        type=str,
        default="advanced",
        choices=["advanced", "momentum", "ml", "mlmomentum", "hybrid"],
        help="Strategy to use (default: advanced)"
    )
    
    # Portfolio parameters
    parser.add_argument(
        "--max-positions", 
        type=int, 
        default=10,
        help="Max positions for portfolio mode (default: 10)"
    )
    
    parser.add_argument(
        "--allocation", 
        type=str, 
        default="equal_weight",
        choices=["equal_weight", "risk_parity", "momentum"],
        help="Allocation mode for portfolio (default: equal_weight)"
    )
    
    # Walk-forward parameters
    parser.add_argument(
        "--train-days", 
        type=int, 
        default=180,
        help="Training days for walk-forward (default: 180)"
    )
    
    parser.add_argument(
        "--test-days", 
        type=int, 
        default=30,
        help="Test days for walk-forward (default: 30)"
    )
    
    # Sequential parameters
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Limit number of symbols for sequential mode"
    )
    
    # Legacy compatibility
    parser.add_argument("--ml", action="store_true", help="Use ML strategy (shortcut)")
    parser.add_argument("--no-ml", action="store_true", help="Don't use ML (ignored)")
    
    args = parser.parse_args()
    
    # Handle legacy --ml flag
    if args.ml:
        args.strategy = "ml"
    
    # Print configuration
    print_config(args)
    
    # Run based on mode
    if args.mode == "single":
        asyncio.run(run_single_backtest(
            symbol=args.symbol,
            capital=args.capital,
            strategy_name=args.strategy
        ))
    
    elif args.mode == "portfolio":
        asyncio.run(run_portfolio_backtest(
            capital=args.capital,
            max_positions=args.max_positions,
            allocation=args.allocation,
            strategy_name=args.strategy
        ))
    
    elif args.mode == "walkforward":
        asyncio.run(run_walk_forward(
            symbol=args.symbol,
            capital=args.capital,
            train_days=args.train_days,
            test_days=args.test_days,
            strategy_name=args.strategy
        ))
    
    elif args.mode == "sequential":
        asyncio.run(run_sequential_backtest(
            capital=args.capital,
            strategy_name=args.strategy,
            max_symbols=args.max_symbols
        ))
    
    elif args.mode == "compare":
        asyncio.run(run_strategy_comparison(
            symbol=args.symbol,
            capital=args.capital
        ))
    
    # Final message
    print("\n" + "=" * 70)
    print("   ✅ BACKTEST COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()