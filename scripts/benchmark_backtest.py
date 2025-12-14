#!/usr/bin/env python3
"""
Backtest Benchmark Script
=========================

Measures and reports backtest performance against optimization targets.
Run after each optimization to verify improvements.

Usage:
    python scripts/benchmark_backtest.py
    python scripts/benchmark_backtest.py --iterations 5
    python scripts/benchmark_backtest.py --symbol AAPL --days 252
"""

import sys
import os
import time
import json
import pickle
import argparse
import tracemalloc
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_optimization_config() -> Dict:
    """Load backtest optimization configuration."""
    config_path = Path("config/backtest_optimization.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def get_test_data(symbol: str, days: int = 252) -> Optional[pd.DataFrame]:
    """
    Get test data for benchmarking.

    Tries multiple sources:
    1. Cached processed data
    2. Raw data files
    3. Download from API
    """
    # Try cached data first
    cache_path = Path(f"data/processed/{symbol}_test.pkl")
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    # Try raw data
    raw_paths = [
        Path(f"data/raw/{symbol}.parquet"),
        Path(f"data/raw/{symbol}.csv"),
        Path(f"data/raw/{symbol}_15min.csv"),
    ]

    for path in raw_paths:
        if path.exists():
            if path.suffix == '.parquet':
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path, parse_dates=['timestamp'])

            # Standardize columns
            df.columns = df.columns.str.lower()
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)

            # Take last N days of 15-min bars
            if len(df) > days * 26:  # 26 bars per day for 15-min data
                df = df.tail(days * 26)

            return df

    # Try downloading
    try:
        from src.data.loader import DataLoader
        loader = DataLoader()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        df = loader.load_data(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            timeframe='15min'
        )

        if df is not None and len(df) > 0:
            # Cache for future runs
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
            return df

    except Exception as e:
        print(f"Warning: Could not download data for {symbol}: {e}")

    return None


def benchmark_feature_generation(
    df: pd.DataFrame,
    iterations: int = 3
) -> Dict[str, Any]:
    """Benchmark feature generation speed."""
    from src.features.institutional import InstitutionalFeatureEngineer

    times = []
    feature_counts = []

    for i in range(iterations):
        # Clear any caches
        engineer = InstitutionalFeatureEngineer()

        start = time.perf_counter()
        features = engineer.build_features(df)
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        if features is not None:
            feature_counts.append(len(features.columns))

    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': min(times),
        'max_time': max(times),
        'feature_count': feature_counts[0] if feature_counts else 0,
        'bars_per_second': len(df) / np.mean(times) if np.mean(times) > 0 else 0
    }


def benchmark_vectorized_backtest(
    prices: pd.DataFrame,
    iterations: int = 3
) -> Dict[str, Any]:
    """Benchmark vectorized backtester speed."""
    from src.backtest.engine import VectorizedBacktester

    # Create random signals for testing
    np.random.seed(42)
    signals = pd.DataFrame(
        np.random.choice([-1, 0, 1], size=prices.shape, p=[0.2, 0.6, 0.2]),
        index=prices.index,
        columns=prices.columns
    )

    times = []

    for i in range(iterations):
        backtester = VectorizedBacktester(
            initial_capital=1000000,
            commission_pct=0.001,
            slippage_pct=0.0005
        )

        start = time.perf_counter()
        result = backtester.run(prices, signals)
        elapsed = time.perf_counter() - start

        times.append(elapsed)

    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': min(times),
        'max_time': max(times),
        'bars_per_second': len(prices) * len(prices.columns) / np.mean(times) if np.mean(times) > 0 else 0
    }


def measure_memory_usage(func, *args, **kwargs) -> Tuple[Any, float]:
    """Measure peak memory usage of a function."""
    tracemalloc.start()

    result = func(*args, **kwargs)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return result, peak / 1024 / 1024  # Convert to MB


def run_benchmark(
    symbol: str = 'AAPL',
    days: int = 252,
    iterations: int = 3
) -> Dict[str, Any]:
    """Run full benchmark suite."""
    print(f"\n{'='*60}")
    print("ALPHATRADE BACKTEST BENCHMARK")
    print(f"{'='*60}")
    print(f"Symbol: {symbol}")
    print(f"Days: {days}")
    print(f"Iterations: {iterations}")
    print(f"{'='*60}\n")

    results = {
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'days': days,
        'iterations': iterations,
        'benchmarks': {},
        'targets': {}
    }

    # Load targets from config
    opt_config = load_optimization_config()
    targets = opt_config.get('performance_targets', {})
    results['targets'] = targets

    # Get test data
    print("Loading test data...")
    df = get_test_data(symbol, days)

    if df is None:
        print(f"ERROR: Could not load data for {symbol}")
        return results

    print(f"  Loaded {len(df)} bars")
    results['data'] = {
        'bars': len(df),
        'date_range': f"{df.index.min()} to {df.index.max()}"
    }

    # Benchmark 1: Feature Generation
    print("\n1. Benchmarking FEATURE GENERATION...")
    try:
        feat_results, feat_memory = measure_memory_usage(
            benchmark_feature_generation, df, iterations
        )
        feat_results['peak_memory_mb'] = feat_memory
        results['benchmarks']['feature_generation'] = feat_results

        print(f"   Mean time: {feat_results['mean_time']:.2f}s")
        print(f"   Bars/second: {feat_results['bars_per_second']:.0f}")
        print(f"   Peak memory: {feat_memory:.1f} MB")
        print(f"   Features: {feat_results['feature_count']}")
    except Exception as e:
        print(f"   ERROR: {e}")
        results['benchmarks']['feature_generation'] = {'error': str(e)}

    # Benchmark 2: Vectorized Backtest
    print("\n2. Benchmarking VECTORIZED BACKTEST...")
    try:
        # Create price DataFrame
        prices_df = df[['close']].rename(columns={'close': symbol})

        vect_results, vect_memory = measure_memory_usage(
            benchmark_vectorized_backtest, prices_df, iterations
        )
        vect_results['peak_memory_mb'] = vect_memory
        results['benchmarks']['vectorized_backtest'] = vect_results

        print(f"   Mean time: {vect_results['mean_time']:.4f}s")
        print(f"   Bars/second: {vect_results['bars_per_second']:.0f}")
        print(f"   Peak memory: {vect_memory:.1f} MB")
    except Exception as e:
        print(f"   ERROR: {e}")
        results['benchmarks']['vectorized_backtest'] = {'error': str(e)}

    # Benchmark 3: Full Pipeline (Features + Backtest)
    print("\n3. Benchmarking FULL PIPELINE (features + backtest)...")
    try:
        start = time.perf_counter()

        # Feature generation
        from src.features.institutional import InstitutionalFeatureEngineer
        engineer = InstitutionalFeatureEngineer()
        features = engineer.build_features(df)

        # Vectorized backtest
        if features is not None and len(features) > 0:
            from src.backtest.engine import VectorizedBacktester

            # Generate signals from features
            signals = np.sign(features.get('return_momentum', pd.Series(0, index=features.index)))
            signals_df = pd.DataFrame({symbol: signals})
            prices_df = df['close'].loc[features.index].to_frame(name=symbol)

            backtester = VectorizedBacktester()
            result = backtester.run(prices_df, signals_df)

        total_time = time.perf_counter() - start

        results['benchmarks']['full_pipeline'] = {
            'total_time': total_time,
            'bars_per_second': len(df) / total_time if total_time > 0 else 0
        }

        print(f"   Total time: {total_time:.2f}s")
        print(f"   Bars/second: {len(df) / total_time:.0f}")

    except Exception as e:
        print(f"   ERROR: {e}")
        results['benchmarks']['full_pipeline'] = {'error': str(e)}

    # Compare against targets
    print("\n" + "="*60)
    print("PERFORMANCE VS TARGETS")
    print("="*60)

    max_time = targets.get('max_backtest_seconds', 300)
    min_bars_per_sec = targets.get('min_bars_per_second', 1000)
    memory_limit = targets.get('memory_limit_gb', 16) * 1024  # Convert to MB

    full_time = results['benchmarks'].get('full_pipeline', {}).get('total_time', float('inf'))
    full_speed = results['benchmarks'].get('full_pipeline', {}).get('bars_per_second', 0)
    peak_mem = max(
        results['benchmarks'].get('feature_generation', {}).get('peak_memory_mb', 0),
        results['benchmarks'].get('vectorized_backtest', {}).get('peak_memory_mb', 0)
    )

    time_pass = full_time <= max_time
    speed_pass = full_speed >= min_bars_per_sec
    memory_pass = peak_mem <= memory_limit

    print(f"Time Target ({max_time}s):     {'PASS' if time_pass else 'FAIL'} ({full_time:.1f}s)")
    print(f"Speed Target ({min_bars_per_sec}/s): {'PASS' if speed_pass else 'FAIL'} ({full_speed:.0f}/s)")
    print(f"Memory Target ({memory_limit:.0f}MB): {'PASS' if memory_pass else 'FAIL'} ({peak_mem:.1f}MB)")

    results['pass'] = time_pass and speed_pass and memory_pass

    # Save results
    output_dir = Path("results/benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    latest_file = output_dir / "benchmark_latest.json"
    with open(latest_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    print(f"Latest results: {latest_file}")

    print("\n" + "="*60)
    print(f"OVERALL: {'PASS' if results['pass'] else 'FAIL'}")
    print("="*60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark AlphaTrade backtest performance"
    )
    parser.add_argument("--symbol", type=str, default="AAPL",
                       help="Symbol to benchmark (default: AAPL)")
    parser.add_argument("--days", type=int, default=252,
                       help="Number of days of data (default: 252)")
    parser.add_argument("--iterations", type=int, default=3,
                       help="Number of iterations for timing (default: 3)")

    args = parser.parse_args()

    results = run_benchmark(
        symbol=args.symbol,
        days=args.days,
        iterations=args.iterations
    )

    sys.exit(0 if results.get('pass', False) else 1)


if __name__ == "__main__":
    main()
