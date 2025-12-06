#!/usr/bin/env python3
"""
============================================================================
ALPHATRADE - BACKTEST RUNNER (FIXED VERSION)
============================================================================
Replace your run_backtest.py with this file.

Changes made:
1. Uses MLMomentumStrategy when --ml flag is set
2. Uses MODERATE risk profile for more trades
3. Lowered min_confidence significantly
4. Added proper ML parameters

Usage:
    python run_backtest.py --symbol AAPL --capital 100000 --ml
    python run_backtest.py --symbol AAPL --capital 100000 --no-ml
    python run_backtest.py --mode portfolio --capital 100000

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


async def run_single_backtest(symbol: str, capital: float, start_date=None, end_date=None, use_ml: bool = True):
    """
    Run backtest on a single stock.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL")
        capital: Initial capital
        start_date: Optional start date
        end_date: Optional end date
        use_ml: Use ML-enhanced strategy (default: True)
    """
    # Import here to avoid circular imports
    from backtest import ProfessionalBacktester
    from strategies.momentum import AdvancedMomentum, MLMomentumStrategy
    from risk.optimized_configs import RiskProfiles
    
    print(f"📊 Single Stock Backtest: {symbol}")
    print(f"💰 Capital: ${capital:,.2f}")
    print(f"🤖 ML Strategy: {'YES - MLMomentumStrategy' if use_ml else 'NO - AdvancedMomentum'}")
    print("-" * 50)
    
    backtester = ProfessionalBacktester(
        symbol=symbol,
        initial_capital=capital,
        commission_pct=0.001,    # 0.1% commission
        slippage_pct=0.0005,     # 0.05% slippage
        use_risk_management=True
    )
    
    # IMPORTANT: Use MODERATE risk profile for more trades
    backtester.risk_manager.config = RiskProfiles.MODERATE
    
    if use_ml:
        # ============================================
        # ML + TECHNICAL FUSION STRATEGY
        # ============================================
        print("🤖 Using MLMomentumStrategy with online learning...")
        
        results = await backtester.run(
            strategy_class=MLMomentumStrategy,
            strategy_params={
                # Technical parameters
                'fast_period': 8,
                'slow_period': 21,
                'rsi_period': 14,
                
                # ML parameters
                'ml_model_type': 'xgboost',
                'ml_model_path': None,          # No pre-trained model
                'ml_weight': 0.4,               # 40% ML, 60% Technical
                'online_learning': True,        # Train during backtest!
                
                # Signal thresholds - CRITICAL FOR MORE TRADES
                'min_confidence': 0.25,         # VERY LOW - more signals
                'min_agreement': 0.3,           # LOW - less strict agreement
                'lookback': 100                 # Shorter lookback
            },
            start_date=start_date,
            end_date=end_date
        )
    else:
        # ============================================
        # PURE TECHNICAL STRATEGY
        # ============================================
        print("📈 Using AdvancedMomentum (pure technical)...")
        
        results = await backtester.run(
            strategy_class=AdvancedMomentum,
            strategy_params={
                'fast_period': 8,
                'slow_period': 21,
                'rsi_period': 14,
                'min_confidence': 0.2,          # VERY LOW
                'use_regime_filter': False,
                'use_volume_confirmation': False
            },
            start_date=start_date,
            end_date=end_date
        )
    
    return results


async def run_portfolio_backtest(capital: float, max_positions: int, allocation: str):
    """
    Run backtest on all 46 stocks together.
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
        'fast_period': [5, 8, 10],
        'slow_period': [15, 21, 30],
        'rsi_period': [10, 14, 20],
        'min_confidence': [0.2, 0.3, 0.4]
    }
    
    results = await optimizer.run(
        strategy_class=AdvancedMomentum,
        param_grid=param_grid
    )
    
    # Export results
    optimizer.export_results(f"wf_{symbol}_{datetime.now().strftime('%Y%m%d')}")
    
    return results


async def run_all_stocks_sequential(capital: float, use_ml: bool = True):
    """
    Run backtest on each stock individually (sequential).
    """
    from pathlib import Path
    
    storage_path = Path("data/storage")
    csv_files = list(storage_path.glob("*_15min.csv"))
    
    if not csv_files:
        print("❌ No data files found!")
        return None
    
    symbols = [f.stem.replace("_15min", "") for f in csv_files]
    print(f"📊 Running backtest on {len(symbols)} stocks...")
    
    results = {}
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Testing {symbol}...")
        try:
            result = await run_single_backtest(symbol, capital, use_ml=use_ml)
            if result:
                results[symbol] = {
                    'return': result.total_return,
                    'sharpe': result.sharpe_ratio,
                    'trades': result.total_trades
                }
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("   SEQUENTIAL BACKTEST SUMMARY")
    print("=" * 70)
    
    for symbol, data in sorted(results.items(), key=lambda x: x[1]['return'], reverse=True)[:10]:
        print(f"   {symbol:<6} | Return: {data['return']:>+8.2f}% | Sharpe: {data['sharpe']:>6.2f} | Trades: {data['trades']:>4}")
    
    return results


def main():
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="AlphaTrade Backtest Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_backtest.py --symbol AAPL --capital 100000 --ml
  python run_backtest.py --symbol AAPL --capital 100000 --no-ml
  python run_backtest.py --mode portfolio --capital 100000
  python run_backtest.py --mode walkforward --symbol AAPL
  python run_backtest.py --mode sequential --capital 100000
        """
    )
    
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "portfolio", "walkforward", "sequential"],
                        help="Backtest mode (default: single)")
    
    parser.add_argument("--symbol", type=str, default="AAPL",
                        help="Symbol for single backtest (default: AAPL)")
    
    parser.add_argument("--capital", type=float, default=100000,
                        help="Initial capital (default: 100000)")
    
    parser.add_argument("--ml", action="store_true", default=False,
                        help="Use ML-enhanced strategy (MLMomentumStrategy)")
    
    parser.add_argument("--no-ml", action="store_true", default=False,
                        help="Use pure technical strategy (AdvancedMomentum)")
    
    parser.add_argument("--train-days", type=int, default=180,
                        help="Training days for walk-forward (default: 180)")
    
    parser.add_argument("--test-days", type=int, default=30,
                        help="Test days for walk-forward (default: 30)")
    
    parser.add_argument("--max-positions", type=int, default=10,
                        help="Max positions for portfolio mode (default: 10)")
    
    parser.add_argument("--allocation", type=str, default="equal_weight",
                        choices=["equal_weight", "risk_parity", "momentum"],
                        help="Allocation mode for portfolio (default: equal_weight)")
    
    args = parser.parse_args()
    
    # Determine ML usage: default is NO ML unless --ml is specified
    use_ml = args.ml and not args.no_ml
    
    print(f"📋 Mode: {args.mode.upper()}")
    print(f"🤖 ML Strategy: {'ENABLED' if use_ml else 'DISABLED'}")
    print("-" * 50)
    
    # Run based on mode
    if args.mode == "single":
        asyncio.run(run_single_backtest(
            symbol=args.symbol,
            capital=args.capital,
            use_ml=use_ml
        ))
    
    elif args.mode == "portfolio":
        asyncio.run(run_portfolio_backtest(
            capital=args.capital,
            max_positions=args.max_positions,
            allocation=args.allocation
        ))
    
    elif args.mode == "walkforward":
        asyncio.run(run_walk_forward(
            symbol=args.symbol,
            capital=args.capital,
            train_days=args.train_days,
            test_days=args.test_days
        ))
    
    elif args.mode == "sequential":
        asyncio.run(run_all_stocks_sequential(
            capital=args.capital,
            use_ml=use_ml
        ))
    
    print("\n" + "=" * 70)
    print("   ✅ BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()