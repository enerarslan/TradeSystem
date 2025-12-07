"""
Pytest Configuration and Fixtures
==================================

Shared fixtures for all tests in the algo trading platform.

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Generator

import numpy as np
import polars as pl
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# CONFIGURATION
# =============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get project root path."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_path(project_root: Path) -> Path:
    """Get data directory path."""
    return project_root / "data" / "storage"


@pytest.fixture(scope="session")
def test_data_path(project_root: Path) -> Path:
    """Get test data directory path."""
    path = project_root / "tests" / "test_data"
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_ohlcv_data() -> pl.DataFrame:
    """Generate sample OHLCV data for testing."""
    n_bars = 1000
    base_price = 100.0
    timestamps = [datetime(2023, 1, 1) + timedelta(minutes=15 * i) for i in range(n_bars)]
    
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.02, n_bars)
    prices = base_price * np.cumprod(1 + returns)
    
    df = pl.DataFrame({
        "timestamp": timestamps,
        "open": prices * (1 + np.random.uniform(-0.005, 0.005, n_bars)),
        "high": prices * (1 + np.random.uniform(0.001, 0.015, n_bars)),
        "low": prices * (1 - np.random.uniform(0.001, 0.015, n_bars)),
        "close": prices,
        "volume": np.random.uniform(1e6, 1e7, n_bars),
    })
    
    # Ensure high >= close >= low
    df = df.with_columns([
        pl.when(pl.col("high") < pl.col("close"))
        .then(pl.col("close"))
        .otherwise(pl.col("high"))
        .alias("high"),
        pl.when(pl.col("low") > pl.col("close"))
        .then(pl.col("close"))
        .otherwise(pl.col("low"))
        .alias("low"),
    ])
    
    return df


@pytest.fixture
def sample_multi_symbol_data(sample_ohlcv_data: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Generate sample data for multiple symbols."""
    symbols = ["AAPL", "GOOGL", "MSFT"]
    data = {}
    
    for i, symbol in enumerate(symbols):
        # Add some variation to each symbol
        np.random.seed(42 + i)
        multiplier = 1 + np.random.uniform(-0.1, 0.1)
        
        df = sample_ohlcv_data.with_columns([
            (pl.col("open") * multiplier).alias("open"),
            (pl.col("high") * multiplier).alias("high"),
            (pl.col("low") * multiplier).alias("low"),
            (pl.col("close") * multiplier).alias("close"),
            pl.lit(symbol).alias("symbol"),
        ])
        
        data[symbol] = df
    
    return data


@pytest.fixture
def sample_features_data(sample_ohlcv_data: pl.DataFrame) -> pl.DataFrame:
    """Generate sample data with features."""
    from features.pipeline import FeaturePipeline, create_default_config
    
    pipeline = FeaturePipeline(create_default_config())
    return pipeline.generate(sample_ohlcv_data)


@pytest.fixture
def sample_ml_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate sample ML training data."""
    np.random.seed(42)
    
    n_samples = 1000
    n_features = 50
    
    X = np.random.randn(n_samples, n_features)
    y = (np.random.rand(n_samples) > 0.5).astype(int)  # Binary classification
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    return X, y, feature_names


@pytest.fixture
def sample_multiclass_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate sample multi-class ML data."""
    np.random.seed(42)
    
    n_samples = 1000
    n_features = 50
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 3, n_samples)  # 3 classes: -1, 0, 1 (sell, hold, buy)
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    return X, y, feature_names


# =============================================================================
# STRATEGY FIXTURES
# =============================================================================

@pytest.fixture
def trend_following_strategy():
    """Create a trend following strategy for testing."""
    from strategies import TrendFollowingStrategy, TrendFollowingConfig
    
    config = TrendFollowingConfig(
        fast_period=10,
        slow_period=30,
    )
    return TrendFollowingStrategy(config)


@pytest.fixture
def mean_reversion_strategy():
    """Create a mean reversion strategy for testing."""
    from strategies import MeanReversionStrategy, MeanReversionConfig
    
    config = MeanReversionConfig(
        period=20,
        std_multiplier=2.0,
    )
    return MeanReversionStrategy(config)


# =============================================================================
# MODEL FIXTURES
# =============================================================================

@pytest.fixture
def lightgbm_model():
    """Create a LightGBM model for testing."""
    from models.classifiers import LightGBMClassifier, LightGBMClassifierConfig
    
    config = LightGBMClassifierConfig(
        n_estimators=100,
        max_depth=5,
    )
    return LightGBMClassifier(config)


@pytest.fixture
def xgboost_model():
    """Create an XGBoost model for testing."""
    from models.classifiers import XGBoostClassifier, XGBoostClassifierConfig
    
    config = XGBoostClassifierConfig(
        n_estimators=100,
        max_depth=5,
    )
    return XGBoostClassifier(config)


@pytest.fixture
def trained_lightgbm_model(sample_multiclass_data, lightgbm_model):
    """Create a trained LightGBM model."""
    X, y, feature_names = sample_multiclass_data
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    lightgbm_model.fit(X_train, y_train, feature_names=feature_names)
    
    return lightgbm_model


# =============================================================================
# BACKTESTING FIXTURES
# =============================================================================

@pytest.fixture
def backtest_config():
    """Create backtest configuration."""
    from backtesting.engine import BacktestConfig
    
    return BacktestConfig(
        initial_capital=100_000.0,
        commission_pct=0.001,
        slippage_pct=0.0005,
    )


@pytest.fixture
def backtest_engine(backtest_config):
    """Create backtest engine."""
    from backtesting.engine import BacktestEngine
    
    return BacktestEngine(backtest_config)


# =============================================================================
# UTILITY FIXTURES
# =============================================================================

@pytest.fixture
def temp_model_dir(tmp_path: Path) -> Path:
    """Create temporary directory for model artifacts."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


@pytest.fixture
def temp_report_dir(tmp_path: Path) -> Path:
    """Create temporary directory for reports."""
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


# =============================================================================
# ASYNC FIXTURES
# =============================================================================

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# CLEANUP FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Any cleanup code here


# =============================================================================
# MARKS
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "live: marks tests requiring live data")