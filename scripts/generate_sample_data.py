"""
Generate sample OHLCV data for testing the AlphaTrade system.

This script creates realistic-looking market data for multiple symbols
using geometric Brownian motion with mean reversion.

Usage:
    python scripts/generate_sample_data.py
    python scripts/generate_sample_data.py --symbols 10 --days 500
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def generate_ohlcv(
    symbol: str,
    start_date: datetime,
    num_days: int,
    initial_price: float = 100.0,
    volatility: float = 0.02,
    drift: float = 0.0001,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data using geometric Brownian motion.

    Args:
        symbol: Stock symbol name
        start_date: Start date for data
        num_days: Number of trading days
        initial_price: Starting price
        volatility: Daily volatility (std dev of returns)
        drift: Daily drift (expected return)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with OHLCV columns
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate daily returns using GBM
    returns = np.random.normal(drift, volatility, num_days)

    # Add some mean reversion
    prices = [initial_price]
    mean_price = initial_price
    for i, r in enumerate(returns):
        # Mean reversion factor
        mean_revert = 0.01 * (mean_price - prices[-1]) / mean_price
        new_price = prices[-1] * (1 + r + mean_revert)
        prices.append(max(new_price, 1.0))  # Price floor
        mean_price = 0.99 * mean_price + 0.01 * new_price  # Slow-moving average

    prices = np.array(prices[1:])  # Remove initial price

    # Generate OHLC from close prices
    data = []
    current_date = start_date

    for i, close in enumerate(prices):
        # Skip weekends
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)

        # Generate intraday range
        daily_range = close * np.random.uniform(0.01, 0.03)
        high_bias = np.random.uniform(0.3, 0.7)

        high = close + daily_range * high_bias
        low = close - daily_range * (1 - high_bias)

        # Open price - gap from previous close
        if i == 0:
            open_price = close * np.random.uniform(0.99, 1.01)
        else:
            gap = np.random.normal(0, volatility / 2)
            open_price = prices[i - 1] * (1 + gap)

        # Ensure OHLC consistency
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # Volume - random with trend correlation
        base_volume = np.random.lognormal(15, 1)
        volatility_factor = 1 + abs(close - open_price) / open_price * 10
        volume = int(base_volume * volatility_factor)

        data.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": volume,
        })

        current_date += timedelta(days=1)

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    return df


def generate_sample_symbols(
    num_symbols: int = 10,
    num_days: int = 500,
    output_dir: str = "data/raw",
) -> None:
    """Generate sample data for multiple symbols."""

    # Sample symbol names
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA", "TSLA", "JPM", "V", "JNJ",
        "WMT", "PG", "MA", "UNH", "HD",
        "DIS", "PYPL", "ADBE", "NFLX", "CRM",
    ][:num_symbols]

    # Different characteristics for each symbol
    symbol_params = {
        "AAPL": {"price": 150, "vol": 0.018, "drift": 0.0003},
        "MSFT": {"price": 300, "vol": 0.016, "drift": 0.0003},
        "GOOGL": {"price": 120, "vol": 0.020, "drift": 0.0002},
        "AMZN": {"price": 130, "vol": 0.022, "drift": 0.0002},
        "META": {"price": 300, "vol": 0.025, "drift": 0.0001},
        "NVDA": {"price": 400, "vol": 0.028, "drift": 0.0005},
        "TSLA": {"price": 250, "vol": 0.035, "drift": 0.0001},
        "JPM": {"price": 150, "vol": 0.015, "drift": 0.0002},
        "V": {"price": 250, "vol": 0.014, "drift": 0.0002},
        "JNJ": {"price": 160, "vol": 0.012, "drift": 0.0001},
        "WMT": {"price": 160, "vol": 0.013, "drift": 0.0001},
        "PG": {"price": 150, "vol": 0.011, "drift": 0.0001},
        "MA": {"price": 380, "vol": 0.015, "drift": 0.0002},
        "UNH": {"price": 500, "vol": 0.014, "drift": 0.0002},
        "HD": {"price": 320, "vol": 0.016, "drift": 0.0002},
        "DIS": {"price": 90, "vol": 0.020, "drift": 0.0000},
        "PYPL": {"price": 60, "vol": 0.028, "drift": -0.0001},
        "ADBE": {"price": 500, "vol": 0.020, "drift": 0.0002},
        "NFLX": {"price": 400, "vol": 0.025, "drift": 0.0002},
        "CRM": {"price": 250, "vol": 0.022, "drift": 0.0002},
    }

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Start date - go back from today
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(num_days * 1.5))  # Account for weekends

    print(f"Generating sample data for {len(symbols)} symbols...")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Output directory: {output_path.absolute()}")
    print()

    for i, symbol in enumerate(symbols):
        params = symbol_params.get(
            symbol,
            {"price": 100, "vol": 0.02, "drift": 0.0001}
        )

        df = generate_ohlcv(
            symbol=symbol,
            start_date=start_date,
            num_days=num_days,
            initial_price=params["price"],
            volatility=params["vol"],
            drift=params["drift"],
            seed=hash(symbol) % (2**32),  # Reproducible per symbol
        )

        # Save to CSV
        filepath = output_path / f"{symbol}.csv"
        df.to_csv(filepath)

        print(f"  [{i+1}/{len(symbols)}] {symbol}: {len(df)} bars, "
              f"${df['close'].iloc[0]:.2f} -> ${df['close'].iloc[-1]:.2f}")

    print()
    print(f"Sample data generated successfully!")
    print(f"Files saved to: {output_path.absolute()}")
    print()
    print("Next steps:")
    print("  1. Run backtest: python main.py")
    print("  2. View results: Open reports/tear_sheet_*.html")


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample OHLCV data for testing AlphaTrade"
    )
    parser.add_argument(
        "--symbols",
        type=int,
        default=10,
        help="Number of symbols to generate (default: 10)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=500,
        help="Number of trading days (default: 500)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory (default: data/raw)",
    )

    args = parser.parse_args()

    generate_sample_symbols(
        num_symbols=args.symbols,
        num_days=args.days,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
