#!/usr/bin/env python3
"""
Backtest Diagnostic Script
==========================

Diagnoses why no trades are being executed.
Tests each component step by step.

Usage:
    python diagnose_backtest.py

Author: Algo Trading Platform
"""

import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl
import numpy as np

def main():
    print("=" * 70)
    print("BACKTEST DIAGNOSTIC")
    print("=" * 70)
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n[STEP 1] Loading data...")
    
    data_path = Path("data/storage/AAPL_15min.csv")
    if not data_path.exists():
        print(f"  ERROR: {data_path} not found!")
        return 1
    
    df = pl.read_csv(data_path)
    df = df.with_columns([
        pl.col("timestamp").str.to_datetime().alias("timestamp")
    ])
    
    print(f"  ✓ Loaded {len(df)} bars")
    print(f"  ✓ Columns: {df.columns}")
    print(f"  ✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Take a smaller sample for faster testing
    df_sample = df.head(2000)
    print(f"  ✓ Using sample of {len(df_sample)} bars for testing")
    
    # =========================================================================
    # STEP 2: Test Technical Indicators
    # =========================================================================
    print("\n[STEP 2] Testing technical indicators...")
    
    try:
        from features.technical import ema, sma, adx, atr
        
        df_test = df_sample.clone()
        df_test = ema(df_test, 10)
        df_test = ema(df_test, 30)
        df_test = atr(df_test, 14)
        
        # Check for NaN values
        ema_10_nulls = df_test["ema_10"].null_count()
        ema_30_nulls = df_test["ema_30"].null_count()
        
        print(f"  ✓ EMA 10 calculated (nulls: {ema_10_nulls})")
        print(f"  ✓ EMA 30 calculated (nulls: {ema_30_nulls})")
        print(f"  ✓ Indicator columns: {[c for c in df_test.columns if c not in df_sample.columns]}")
        
    except Exception as e:
        print(f"  ✗ Indicator error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # =========================================================================
    # STEP 3: Test Strategy Signal Generation
    # =========================================================================
    print("\n[STEP 3] Testing strategy signal generation...")
    
    try:
        from strategies.momentum import TrendFollowingStrategy, TrendFollowingConfig
        from core.events import MarketEvent
        from core.types import PortfolioState
        
        # Create strategy with very relaxed parameters
        config = TrendFollowingConfig(
            symbols=["AAPL"],
            fast_ma_period=10,
            slow_ma_period=30,
            trend_ma_period=50,  # Very short
            ma_type="ema",
            use_adx_filter=False,  # Disable ADX filter
            require_price_above_trend_ma=False,  # Disable trend filter
        )
        
        strategy = TrendFollowingStrategy(config)
        print(f"  ✓ Strategy created: {strategy.name}")
        
        # Initialize strategy
        strategy.initialize(
            symbols=["AAPL"],
            start_date=df_sample["timestamp"].min(),
            end_date=df_sample["timestamp"].max(),
        )
        print(f"  ✓ Strategy initialized, state: {strategy._state}")
        
        # Create mock portfolio state
        portfolio = PortfolioState(
            timestamp=datetime.now(),
            cash=100000.0,
            equity=100000.0,
            buying_power=100000.0,
            positions={},
        )
        
        # Test signal generation for each bar
        signals_generated = []
        
        for i in range(100, len(df_sample)):  # Start from 100 to have enough history
            # Get data up to this point
            data_slice = df_sample.head(i + 1)
            
            # Create market event
            market_event = MarketEvent(
                symbol="AAPL",
                data=data_slice,
                timeframe="15min",
            )
            
            # Generate signals
            signals = strategy.on_bar(market_event, portfolio)
            
            if signals:
                signals_generated.extend(signals)
                if len(signals_generated) <= 5:  # Print first 5
                    for sig in signals:
                        print(f"    Signal at bar {i}: {sig.signal_type}, direction={sig.direction}, price={sig.price:.2f}")
        
        print(f"  ✓ Total signals generated: {len(signals_generated)}")
        
        if not signals_generated:
            print("  ✗ NO SIGNALS GENERATED! This is the problem.")
            print("    Possible causes:")
            print("    - MA crossover conditions never met")
            print("    - Strategy filters too strict")
            print("    - Data not reaching strategy correctly")
            
            # Debug: manually check for MA crossovers
            print("\n  [DEBUG] Checking for MA crossovers manually...")
            df_debug = df_sample.clone()
            df_debug = ema(df_debug, 10)
            df_debug = ema(df_debug, 30)
            
            crossovers = 0
            prev_fast = None
            prev_slow = None
            
            for row in df_debug.iter_rows(named=True):
                fast = row.get("ema_10")
                slow = row.get("ema_30")
                
                if fast is None or slow is None:
                    continue
                    
                if prev_fast is not None and prev_slow is not None:
                    # Check for crossover
                    if prev_fast <= prev_slow and fast > slow:
                        crossovers += 1
                    elif prev_fast >= prev_slow and fast < slow:
                        crossovers += 1
                
                prev_fast = fast
                prev_slow = slow
            
            print(f"    Manual crossover count: {crossovers}")
            
            return 1  # Exit early, problem found
        
    except Exception as e:
        print(f"  ✗ Strategy error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # =========================================================================
    # STEP 4: Test Signal to Order Conversion
    # =========================================================================
    print("\n[STEP 4] Testing signal to order conversion...")
    
    try:
        from backtesting.engine import BacktestEngine, BacktestConfig
        
        config = BacktestConfig(
            initial_capital=100000,
            commission_pct=0.001,
            slippage_pct=0.0005,
            warmup_bars=50,
        )
        
        engine = BacktestEngine(config)
        
        # Test _create_order_from_signal
        if signals_generated:
            test_signal = signals_generated[0]
            print(f"  Testing with signal: {test_signal.signal_type}, direction={test_signal.direction}")
            
            order = engine._create_order_from_signal(test_signal)
            
            if order:
                print(f"  ✓ Order created: {order.side} {order.quantity:.2f} @ market")
            else:
                print(f"  ✗ Order NOT created! _create_order_from_signal returned None")
                print(f"    Signal is_entry: {test_signal.is_entry}")
                print(f"    Signal is_exit: {test_signal.is_exit}")
                print(f"    Signal is_long: {test_signal.is_long}")
                print(f"    Signal is_short: {test_signal.is_short}")
        
    except Exception as e:
        print(f"  ✗ Order conversion error: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # STEP 5: Full Backtest with Debug
    # =========================================================================
    print("\n[STEP 5] Running full backtest with debug...")
    
    try:
        from backtesting.engine import BacktestEngine, BacktestConfig
        from strategies.momentum import TrendFollowingStrategy, TrendFollowingConfig
        
        # Ultra-relaxed strategy
        strat_config = TrendFollowingConfig(
            symbols=["AAPL"],
            fast_ma_period=5,
            slow_ma_period=15,
            trend_ma_period=30,
            use_adx_filter=False,
            require_price_above_trend_ma=False,
        )
        
        strategy = TrendFollowingStrategy(strat_config)
        
        bt_config = BacktestConfig(
            initial_capital=100000,
            warmup_bars=30,
        )
        
        engine = BacktestEngine(bt_config)
        engine.add_data("AAPL", df_sample)
        engine.add_strategy(strategy)
        
        # Monkey-patch to add debug output
        original_handle_signal = engine._handle_signal
        signal_count = [0]
        order_count = [0]
        
        def debug_handle_signal(event):
            signal_count[0] += 1
            if signal_count[0] <= 5:
                print(f"    [DEBUG] Signal received: {event.signal_type}, direction={event.direction}")
            original_handle_signal(event)
            if engine._orders and len(engine._orders) > order_count[0]:
                order_count[0] = len(engine._orders)
                if order_count[0] <= 5:
                    print(f"    [DEBUG] Order created: {engine._orders[-1].side}")
        
        engine._handle_signal = debug_handle_signal
        
        print("  Running backtest...")
        report = engine.run(show_progress=False)
        
        print(f"\n  Results:")
        print(f"    Signals received by engine: {signal_count[0]}")
        print(f"    Orders created: {len(engine._orders)}")
        print(f"    Fills: {len(engine._fills)}")
        print(f"    Trades: {report.trade_stats.total_trades}")
        print(f"    Total Return: {report.total_return_pct:.2%}")
        
    except Exception as e:
        print(f"  ✗ Full backtest error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())