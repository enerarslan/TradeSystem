#!/usr/bin/env python3
"""
Paper Trading Script
====================

Run paper trading with configured strategies.

Usage:
    python scripts/paper_trade.py --symbols AAPL GOOGL MSFT
    python scripts/paper_trade.py --capital 50000 --strategy trend_following
    python scripts/paper_trade.py --symbols AAPL --duration 60

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings, get_logger, configure_logging, TradingMode
from execution import (
    LiveTradingEngine,
    LiveTradingConfig,
    TradingEngineState,
    run_paper_trading,
)
from strategies import (
    TrendFollowingStrategy,
    TrendFollowingConfig,
    MeanReversionStrategy,
    MeanReversionConfig,
    MACDStrategy,
    MACDStrategyConfig,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run paper trading with configured strategies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL"],
        help="Symbols to trade",
    )
    
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Initial capital",
    )
    
    parser.add_argument(
        "--strategy",
        choices=["trend_following", "mean_reversion", "macd", "all"],
        default="trend_following",
        help="Strategy to use",
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Trading duration in minutes (0 = indefinite)",
    )
    
    parser.add_argument(
        "--max-positions",
        type=int,
        default=5,
        help="Maximum concurrent positions",
    )
    
    parser.add_argument(
        "--position-size",
        type=float,
        default=0.10,
        help="Position size as fraction of portfolio",
    )
    
    parser.add_argument(
        "--max-daily-loss",
        type=float,
        default=0.02,
        help="Maximum daily loss before shutdown",
    )
    
    parser.add_argument(
        "--smart-routing",
        action="store_true",
        default=False,
        help="Use smart order routing",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    return parser.parse_args()


def create_strategies(strategy_name: str, symbols: list[str]) -> list:
    """Create strategy instances based on name."""
    strategies = []
    
    if strategy_name in ("trend_following", "all"):
        config = TrendFollowingConfig(
            name="TrendFollowing_Paper",
            symbols=symbols,
            fast_period=10,
            slow_period=30,
            atr_period=14,
            atr_multiplier=2.0,
            min_signal_strength=0.3,
        )
        strategies.append(TrendFollowingStrategy(config))
    
    if strategy_name in ("mean_reversion", "all"):
        config = MeanReversionConfig(
            name="MeanReversion_Paper",
            symbols=symbols,
            lookback_period=20,
            entry_threshold=2.0,
            exit_threshold=0.5,
            min_signal_strength=0.3,
        )
        strategies.append(MeanReversionStrategy(config))
    
    if strategy_name in ("macd", "all"):
        config = MACDStrategyConfig(
            name="MACD_Paper",
            symbols=symbols,
            fast_period=12,
            slow_period=26,
            signal_period=9,
            min_signal_strength=0.3,
        )
        strategies.append(MACDStrategy(config))
    
    return strategies


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    settings = get_settings()
    if args.debug:
        settings.log_level = "DEBUG"
    configure_logging(settings)
    
    logger = get_logger(__name__)
    
    print("\n" + "=" * 70)
    print("  PAPER TRADING SESSION")
    print("  Algo Trading Platform - JPMorgan-Level Trading System")
    print("=" * 70)
    print(f"\n  Symbols:        {', '.join(args.symbols)}")
    print(f"  Initial Capital: ${args.capital:,.2f}")
    print(f"  Strategy:        {args.strategy}")
    print(f"  Max Positions:   {args.max_positions}")
    print(f"  Position Size:   {args.position_size:.1%}")
    print(f"  Max Daily Loss:  {args.max_daily_loss:.1%}")
    print(f"  Smart Routing:   {'Enabled' if args.smart_routing else 'Disabled'}")
    if args.duration > 0:
        print(f"  Duration:        {args.duration} minutes")
    else:
        print(f"  Duration:        Indefinite (Ctrl+C to stop)")
    print("\n" + "-" * 70)
    
    # Create strategies
    strategies = create_strategies(args.strategy, args.symbols)
    
    if not strategies:
        logger.error("No strategies created")
        return 1
    
    logger.info(f"Created {len(strategies)} strategies")
    for strategy in strategies:
        logger.info(f"  - {strategy.name}")
    
    # Create trading config
    config = LiveTradingConfig(
        mode=TradingMode.PAPER,
        symbols=args.symbols,
        strategies=strategies,
        initial_capital=args.capital,
        max_positions=args.max_positions,
        position_size_pct=args.position_size,
        max_daily_loss=args.max_daily_loss,
        use_smart_routing=args.smart_routing,
    )
    
    # Create and start engine
    engine = LiveTradingEngine(config)
    
    try:
        print("\n  Starting paper trading...")
        print("  Press Ctrl+C to stop\n")
        print("-" * 70)
        
        # Start engine
        if args.duration > 0:
            # Time-limited trading
            import threading
            
            def stop_after_duration():
                import time
                time.sleep(args.duration * 60)
                if engine.is_running:
                    logger.info("Duration limit reached")
                    engine.stop("Duration limit reached")
            
            timer = threading.Thread(target=stop_after_duration, daemon=True)
            timer.start()
        
        engine.start(block=True)
        
    except KeyboardInterrupt:
        print("\n")
        logger.info("Interrupted by user")
        engine.stop("User interrupt")
        
    except Exception as e:
        logger.exception(f"Paper trading failed: {e}")
        if engine.state == TradingEngineState.RUNNING:
            engine.stop(f"Error: {e}")
        return 1
    
    # Print final statistics
    stats = engine.get_statistics()
    
    print("\n" + "=" * 70)
    print("  PAPER TRADING SESSION ENDED")
    print("=" * 70)
    print(f"\n  Total Trades:    {stats.get('total_trades', 0)}")
    print(f"  Win Rate:        {stats.get('win_rate', 0):.1%}")
    print(f"  Profit Factor:   {stats.get('profit_factor', 0):.2f}")
    print(f"  Realized P&L:    ${stats.get('realized_pnl', 0):,.2f}")
    print(f"  Final Equity:    ${stats.get('current_equity', 0):,.2f}")
    print(f"  Max Drawdown:    {stats.get('max_drawdown', 0):.1%}")
    print("\n" + "=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())