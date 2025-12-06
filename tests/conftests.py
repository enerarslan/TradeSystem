"""
Test Fixtures
=============

Shared pytest fixtures for the algo trading platform tests.

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator
from uuid import uuid4

import numpy as np
import polars as pl
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_ohlcv_data() -> pl.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    
    n_bars = 1000
    base_price = 100.0
    
    # Generate random returns
    returns = np.random.normal(0.0001, 0.02, n_bars)
    
    # Generate prices
    close = base_price * np.cumprod(1 + returns)
    
    # Generate OHLCV
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n_bars)))
    open_price = low + (high - low) * np.random.random(n_bars)
    volume = np.random.uniform(100000, 1000000, n_bars)
    
    # Generate timestamps (15-min bars)
    start_time = datetime(2023, 1, 1, 9, 30)
    timestamps = [start_time + timedelta(minutes=15 * i) for i in range(n_bars)]
    
    return pl.DataFrame({
        "timestamp": timestamps,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def sample_multi_symbol_data(sample_ohlcv_data: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Generate sample data for multiple symbols."""
    np.random.seed(42)
    
    symbols = ["AAPL", "GOOGL", "MSFT"]
    data = {}
    
    for symbol in symbols:
        # Add some random variation to each symbol
        df = sample_ohlcv_data.clone()
        multiplier = np.random.uniform(0.8, 1.2)
        
        df = df.with_columns([
            (pl.col("open") * multiplier).alias("open"),
            (pl.col("high") * multiplier).alias("high"),
            (pl.col("low") * multiplier).alias("low"),
            (pl.col("close") * multiplier).alias("close"),
        ])
        
        data[symbol] = df
    
    return data


@pytest.fixture
def trending_data() -> pl.DataFrame:
    """Generate data with a clear uptrend."""
    np.random.seed(42)
    
    n_bars = 500
    base_price = 100.0
    
    # Generate uptrending prices
    trend = np.linspace(0, 0.5, n_bars)  # 50% gain over period
    noise = np.random.normal(0, 0.01, n_bars)
    close = base_price * (1 + trend + noise)
    
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n_bars)))
    open_price = low + (high - low) * np.random.random(n_bars)
    volume = np.random.uniform(100000, 1000000, n_bars)
    
    start_time = datetime(2023, 1, 1, 9, 30)
    timestamps = [start_time + timedelta(minutes=15 * i) for i in range(n_bars)]
    
    return pl.DataFrame({
        "timestamp": timestamps,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def mean_reverting_data() -> pl.DataFrame:
    """Generate mean-reverting data."""
    np.random.seed(42)
    
    n_bars = 500
    base_price = 100.0
    
    # Generate mean-reverting process (Ornstein-Uhlenbeck)
    theta = 0.1  # Mean reversion speed
    mu = 0.0  # Long-term mean
    sigma = 0.02  # Volatility
    
    x = np.zeros(n_bars)
    for i in range(1, n_bars):
        x[i] = x[i-1] + theta * (mu - x[i-1]) + sigma * np.random.normal()
    
    close = base_price * np.exp(x)
    
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n_bars)))
    open_price = low + (high - low) * np.random.random(n_bars)
    volume = np.random.uniform(100000, 1000000, n_bars)
    
    start_time = datetime(2023, 1, 1, 9, 30)
    timestamps = [start_time + timedelta(minutes=15 * i) for i in range(n_bars)]
    
    return pl.DataFrame({
        "timestamp": timestamps,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def backtest_config():
    """Create default backtest configuration."""
    from backtesting.engine import BacktestConfig
    
    return BacktestConfig(
        initial_capital=100_000,
        commission_pct=0.001,
        slippage_pct=0.0005,
        allow_shorting=True,
        margin_requirement=0.25,
    )


# =============================================================================
# STRATEGY FIXTURES
# =============================================================================

@pytest.fixture
def trend_following_strategy():
    """Create trend following strategy."""
    from strategies.momentum import TrendFollowingStrategy, TrendFollowingConfig
    
    config = TrendFollowingConfig(
        symbols=["TEST"],
        ma_fast_period=10,
        ma_slow_period=20,
        adx_period=14,
        adx_threshold=20,
    )
    return TrendFollowingStrategy(config)


@pytest.fixture
def mean_reversion_strategy():
    """Create mean reversion strategy."""
    from strategies.momentum import MeanReversionStrategy, MeanReversionConfig
    
    config = MeanReversionConfig(
        symbols=["TEST"],
        bb_period=20,
        bb_std=2.0,
        rsi_period=14,
        rsi_oversold=30,
        rsi_overbought=70,
    )
    return MeanReversionStrategy(config)


@pytest.fixture
def breakout_strategy():
    """Create breakout strategy."""
    from strategies.momentum import BreakoutStrategy, BreakoutConfig
    
    config = BreakoutConfig(
        symbols=["TEST"],
        lookback_period=20,
        volume_factor=1.5,
        atr_multiplier=2.0,
    )
    return BreakoutStrategy(config)


# =============================================================================
# ENGINE FIXTURES
# =============================================================================

@pytest.fixture
def backtest_engine(backtest_config):
    """Create backtest engine."""
    from backtesting.engine import BacktestEngine
    
    return BacktestEngine(backtest_config)


# =============================================================================
# UTILITY FIXTURES
# =============================================================================

@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create temporary data directory."""
    data_dir = tmp_path / "data" / "storage"
    data_dir.mkdir(parents=True)
    return data_dir


@pytest.fixture
def sample_csv_file(temp_data_dir: Path, sample_ohlcv_data: pl.DataFrame) -> Path:
    """Create sample CSV file."""
    file_path = temp_data_dir / "TEST_15min.csv"
    sample_ohlcv_data.write_csv(file_path)
    return file_path


# =============================================================================
# MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_broker():
    """Create mock broker for testing."""
    from unittest.mock import MagicMock
    
    broker = MagicMock()
    broker.is_market_open.return_value = True
    broker.get_account_info.return_value = {
        "cash": 100000,
        "equity": 100000,
        "buying_power": 200000,
    }
    broker.get_positions.return_value = []
    broker.submit_order.return_value = str(uuid4())
    
    return broker


# =============================================================================
# MARKERS
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "requires_data: marks tests that require real data")