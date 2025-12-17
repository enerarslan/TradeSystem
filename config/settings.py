"""
Global settings and configuration management for AlphaTrade system.

This module provides centralized configuration using Pydantic for validation
and environment variable support.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
for dir_path in [PROCESSED_DATA_DIR, CACHE_DIR, REPORTS_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


class DataSettings(BaseSettings):
    """Data-related configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="ALPHATRADE_DATA_",
        env_file=".env",
        extra="ignore"
    )

    raw_data_dir: Path = RAW_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR
    cache_dir: Path = CACHE_DIR

    # Data specifications
    timeframe: str = "15min"
    default_file_format: Literal["csv", "parquet", "feather"] = "csv"
    date_column: str = "timestamp"
    ohlcv_columns: list[str] = Field(
        default=["open", "high", "low", "close", "volume"]
    )

    # Date range
    start_date: str | None = None
    end_date: str | None = None

    # Data quality settings
    max_missing_pct: float = 0.05  # Maximum 5% missing values allowed
    min_price: float = 0.01
    max_price_change_pct: float = 50.0  # Max single-bar price change

    @field_validator("raw_data_dir", "processed_data_dir", "cache_dir", mode="before")
    @classmethod
    def convert_to_path(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects."""
        return Path(v) if isinstance(v, str) else v


class TradingSettings(BaseSettings):
    """Trading and strategy configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="ALPHATRADE_TRADING_",
        env_file=".env",
        extra="ignore"
    )

    # Capital and sizing
    initial_capital: float = 1_000_000.0
    max_position_size_pct: float = 0.05  # 5% max per position
    max_sector_exposure_pct: float = 0.25  # 25% max per sector
    max_leverage: float = 1.0  # No leverage by default

    # Transaction costs
    commission_pct: float = 0.001  # 0.1% commission
    slippage_pct: float = 0.0005  # 0.05% slippage
    min_commission: float = 1.0  # Minimum $1 commission

    # Risk limits
    max_drawdown_pct: float = 0.15  # 15% max drawdown
    daily_loss_limit_pct: float = 0.02  # 2% daily loss limit
    position_reduce_drawdown_pct: float = 0.10  # Reduce at 10% drawdown

    # Signal settings
    min_holding_periods: int = 4  # Minimum 4 bars (1 hour for 15min)
    signal_threshold: float = 0.0  # Threshold for signal generation

    # Trading hours (market hours)
    trading_start_time: str = "09:30"
    trading_end_time: str = "16:00"
    timezone: str = "America/New_York"


class BacktestSettings(BaseSettings):
    """Backtesting configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="ALPHATRADE_BACKTEST_",
        env_file=".env",
        extra="ignore"
    )

    # Walk-forward settings
    train_period_bars: int = 5040  # ~2 months of 15-min bars (26 bars/day * 21 days * 2)
    test_period_bars: int = 1260  # ~1 month
    min_train_samples: int = 1000

    # Execution settings
    execution_mode: Literal["next_open", "vwap", "close"] = "next_open"
    allow_partial_fills: bool = False

    # Analysis settings
    benchmark_symbol: str | None = None
    risk_free_rate: float = 0.05  # 5% annual risk-free rate
    trading_days_per_year: int = 252
    bars_per_day: int = 26  # 15-min bars from 9:30 to 16:00

    @property
    def bars_per_year(self) -> int:
        """Calculate bars per year for annualization."""
        return self.trading_days_per_year * self.bars_per_day


class FeatureSettings(BaseSettings):
    """Feature engineering configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="ALPHATRADE_FEATURE_",
        env_file=".env",
        extra="ignore"
    )

    # Moving average periods
    ma_periods: list[int] = Field(default=[5, 10, 20, 50, 100, 200])

    # RSI periods
    rsi_periods: list[int] = Field(default=[7, 14, 21])

    # Volatility windows
    volatility_windows: list[int] = Field(default=[10, 20, 50])

    # Return lookback periods
    return_periods: list[int] = Field(default=[1, 5, 10, 20, 60])

    # Lag periods for features
    lag_periods: list[int] = Field(default=[1, 2, 5, 10])

    # Feature scaling
    scaling_method: Literal["standard", "robust", "minmax"] = "robust"

    # Feature selection
    max_features: int | None = None
    min_importance: float = 0.001


class ModelSettings(BaseSettings):
    """Machine learning model configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="ALPHATRADE_MODEL_",
        env_file=".env",
        extra="ignore"
    )

    # Random seed for reproducibility
    random_seed: int = 42

    # Cross-validation
    n_splits: int = 5
    purge_gap: int = 10  # Gap between train and test to prevent leakage
    embargo_pct: float = 0.01  # Embargo period as fraction of test

    # XGBoost defaults
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8

    # LightGBM defaults
    lgb_n_estimators: int = 100
    lgb_max_depth: int = -1
    lgb_learning_rate: float = 0.1
    lgb_num_leaves: int = 31


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="ALPHATRADE_LOG_",
        env_file=".env",
        extra="ignore"
    )

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    log_file: Path | None = PROJECT_ROOT / "logs" / "alphatrade.log"
    rotation: str = "10 MB"
    retention: str = "30 days"
    serialize: bool = False  # JSON logging


class Settings(BaseSettings):
    """Main settings container combining all configuration sections."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

    # Sub-settings
    data: DataSettings = Field(default_factory=DataSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # Environment
    environment: Literal["development", "testing", "production"] = "development"
    debug: bool = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment and config files."""
    global settings
    settings = Settings()
    return settings
