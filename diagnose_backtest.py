#!/usr/bin/env python3
"""
Backtest Diagnostic Script - FIXED VERSION
==========================================

Diagnoses why backtesting produces no signals.

Fix Applied: Added strategy.start() call after initialize()

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl


def main() -> int:
    """Run diagnostic tests."""
    print("=" * 70)
    print("BACKTEST DIAGNOSTIC - FIXED VERSION")
    print("=" * 70)
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n[STEP 1] Loading data...")
    
    try:
        from data.loader import CSVLoader
        
        loader = CSVLoader(Path("data/storage"))
        # IMPORTANT: Use keyword argument for timeframe!
        # Signature: load(symbol, start_date=None, end_date=None, timeframe="15min")
        df = loader.load(symbol="AAPL", timeframe="15min")
        
        if df is None or len(df) == 0:
            print("  ✗ Failed to load data")
            return 1
        
        print(f"  ✓ Loaded {len(df)} bars")
        print(f"  ✓ Columns: {df.columns}")
        print(f"  ✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Use sample for faster testing
        df_sample = df.head(2000)
        print(f"  ✓ Using sample of {len(df_sample)} bars for testing")
        
    except Exception as e:
        print(f"  ✗ Data loading error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # =========================================================================
    # STEP 2: Test Technical Indicators
    # =========================================================================
    print("\n[STEP 2] Testing technical indicators...")
    
    try:
        from features.technical import ema, atr
        
        df_test = df_sample.clone()
        df_test = ema(df_test, 10)
        df_test = ema(df_test, 30)
        df_test = atr(df_test, 14)
        
        ema_10_nulls = df_test["ema_10"].null_count()
        ema_30_nulls = df_test["ema_30"].null_count()
        
        print(f"  ✓ EMA 10 calculated (nulls: {ema_10_nulls})")
        print(f"  ✓ EMA 30 calculated (nulls: {ema_30_nulls})")
        
        indicator_cols = [c for c in df_test.columns if c not in df_sample.columns]
        print(f"  ✓ Indicator columns: {indicator_cols}")
        
    except Exception as e:
        print(f"  ✗ Indicator error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # =========================================================================
    # STEP 3: Test Strategy Signal Generation (FIXED)
    # =========================================================================
    print("\n[STEP 3] Testing strategy signal generation...")
    
    try:
        from strategies.momentum import TrendFollowingStrategy, TrendFollowingConfig
        from strategies.base import StrategyState
        from core.events import MarketEvent
        from core.types import PortfolioState
        
        # Create strategy with very relaxed parameters
        config = TrendFollowingConfig(
            symbols=["AAPL"],
            fast_ma_period=10,
            slow_ma_period=30,
            trend_ma_period=50,
            ma_type="ema",
            use_adx_filter=False,      # Disable ADX filter
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
        
        # =====================================================================
        # FIX: MUST CALL start() TO TRANSITION TO RUNNING STATE
        # =====================================================================
        strategy.start()
        print(f"  ✓ Strategy started, state: {strategy._state}")
        
        # Verify state is RUNNING
        if strategy._state != StrategyState.RUNNING:
            print(f"  ✗ ERROR: Strategy state is {strategy._state}, expected RUNNING")
            return 1
        
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
        
        for i in range(100, len(df_sample)):
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
            print("  ✗ NO SIGNALS GENERATED! Additional debugging needed.")
            print("\n  [DEBUG] Checking for MA crossovers manually...")
            
            from features.technical import ema as calc_ema
            
            df_debug = df_sample.clone()
            df_debug = calc_ema(df_debug, 10)
            df_debug = calc_ema(df_debug, 30)
            
            crossovers = 0
            prev_fast = None
            prev_slow = None
            
            for row in df_debug.iter_rows(named=True):
                fast = row.get("ema_10")
                slow = row.get("ema_30")
                
                if fast is None or slow is None:
                    continue
                    
                if prev_fast is not None and prev_slow is not None:
                    if prev_fast <= prev_slow and fast > slow:
                        crossovers += 1
                    elif prev_fast >= prev_slow and fast < slow:
                        crossovers += 1
                
                prev_fast = fast
                prev_slow = slow
            
            print(f"    Manual crossover count: {crossovers}")
            
            if crossovers > 0:
                print("    ⚠ Crossovers exist but strategy didn't generate signals.")
                print("    Possible issues:")
                print("    - min_signal_strength filter too strict")
                print("    - Data slice not providing enough history")
                print("    - Indicator column names mismatch")
            
            return 1
        else:
            print(f"  ✓ SUCCESS! Strategy is generating signals correctly.")
            
            # Show signal distribution
            entry_signals = [s for s in signals_generated if s.is_entry]
            exit_signals = [s for s in signals_generated if s.is_exit]
            long_signals = [s for s in signals_generated if s.direction > 0]
            short_signals = [s for s in signals_generated if s.direction < 0]
            
            print(f"\n  Signal Distribution:")
            print(f"    Entry signals: {len(entry_signals)}")
            print(f"    Exit signals: {len(exit_signals)}")
            print(f"    Long signals: {len(long_signals)}")
            print(f"    Short signals: {len(short_signals)}")
        
    except Exception as e:
        print(f"  ✗ Strategy error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # =========================================================================
    # STEP 4: Test Full Backtest
    # =========================================================================
    print("\n[STEP 4] Running full backtest...")
    
    try:
        from backtesting.engine import BacktestEngine, BacktestConfig
        from strategies.momentum import TrendFollowingStrategy, TrendFollowingConfig
        
        strat_config = TrendFollowingConfig(
            symbols=["AAPL"],
            fast_ma_period=10,
            slow_ma_period=30,
            trend_ma_period=50,
            use_adx_filter=False,
            require_price_above_trend_ma=False,
        )
        
        strategy = TrendFollowingStrategy(strat_config)
        
        bt_config = BacktestConfig(
            initial_capital=100000,
            warmup_bars=50,
            commission_pct=0.001,
            slippage_pct=0.0005,
        )
        
        engine = BacktestEngine(bt_config)
        engine.add_data("AAPL", df_sample)
        engine.add_strategy(strategy)
        
        print("  Running backtest...")
        report = engine.run(show_progress=False)
        
        print(f"\n  Results:")
        print(f"    Total Trades: {report.trade_stats.total_trades}")
        print(f"    Win Rate: {report.trade_stats.win_rate:.2%}")
        print(f"    Total Return: {report.total_return_pct:.2%}")
        print(f"    Sharpe Ratio: {report.sharpe_ratio:.2f}")
        print(f"    Max Drawdown: {report.max_drawdown:.2%}")
        
        if report.trade_stats.total_trades == 0:
            print("\n  ⚠ WARNING: No trades executed!")
            print("    This might indicate:")
            print("    - Signal to order conversion issue")
            print("    - Position sizing rejecting orders")
            print("    - Risk management blocking trades")
        
    except Exception as e:
        print(f"  ✗ Full backtest error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # =========================================================================
    # STEP 5: Test Other Strategies
    # =========================================================================
    print("\n[STEP 5] Testing other strategies...")
    
    strategies_to_test = [
        ("MeanReversion", "strategies.momentum", "MeanReversionStrategy", "MeanReversionConfig"),
        ("Breakout", "strategies.momentum", "BreakoutStrategy", "BreakoutConfig"),
    ]
    
    for strat_name, module_name, class_name, config_name in strategies_to_test:
        try:
            import importlib
            module = importlib.import_module(module_name)
            StrategyClass = getattr(module, class_name)
            ConfigClass = getattr(module, config_name)
            
            config = ConfigClass(symbols=["AAPL"])
            strategy = StrategyClass(config)
            strategy.initialize(["AAPL"], df_sample["timestamp"].min(), df_sample["timestamp"].max())
            strategy.start()
            
            signals = []
            portfolio = PortfolioState(
                timestamp=datetime.now(),
                cash=100000.0,
                equity=100000.0,
                buying_power=100000.0,
                positions={},
            )
            
            for i in range(100, min(500, len(df_sample))):
                data_slice = df_sample.head(i + 1)
                market_event = MarketEvent(symbol="AAPL", data=data_slice, timeframe="15min")
                sigs = strategy.on_bar(market_event, portfolio)
                signals.extend(sigs)
            
            print(f"  ✓ {strat_name}: {len(signals)} signals (first 400 bars)")
            
        except Exception as e:
            print(f"  ✗ {strat_name}: Error - {e}")
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())