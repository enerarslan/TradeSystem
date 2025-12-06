"""
Settings Module
===============

Comprehensive configuration management for the algorithmic trading platform.
Includes all settings, constants, enums, and logging configuration.

Features:
- Environment-based configuration with .env support
- Type-safe settings with Pydantic v2
- Structured logging with structlog
- Trading constants and enums
- Risk parameters and limits

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

import logging
import sys
from datetime import time, timezone
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import structlog
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# ENUMS
# =============================================================================

class TradingMode(str, Enum):
    """Trading execution mode."""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class LogLevel(str, Enum):
    """Logging level configuration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TimeFrame(str, Enum):
    """Supported trading timeframes."""
    M1 = "1min"
    M5 = "5min"
    M15 = "15min"
    M30 = "30min"
    H1 = "1hour"
    H4 = "4hour"
    D1 = "1day"
    W1 = "1week"
    
    @property
    def minutes(self) -> int:
        """Convert timeframe to minutes."""
        mapping = {
            "1min": 1, "5min": 5, "15min": 15, "30min": 30,
            "1hour": 60, "4hour": 240, "1day": 1440, "1week": 10080
        }
        return mapping[self.value]
    
    @property
    def pandas_freq(self) -> str:
        """Convert to pandas frequency string."""
        mapping = {
            "1min": "1T", "5min": "5T", "15min": "15T", "30min": "30T",
            "1hour": "1H", "4hour": "4H", "1day": "1D", "1week": "1W"
        }
        return mapping[self.value]


class OrderSide(str, Enum):
    """Order side (direction)."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type for execution."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class PositionSide(str, Enum):
    """Position direction."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class SignalType(str, Enum):
    """Trading signal types."""
    ENTRY_LONG = "entry_long"
    ENTRY_SHORT = "entry_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    HOLD = "hold"


class MarketRegime(str, Enum):
    """Market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


# =============================================================================
# CONSTANTS
# =============================================================================

class TradingConstants:
    """Trading-related constants."""
    
    # Market hours (US Eastern Time)
    MARKET_OPEN: time = time(9, 30)
    MARKET_CLOSE: time = time(16, 0)
    PRE_MARKET_OPEN: time = time(4, 0)
    AFTER_MARKET_CLOSE: time = time(20, 0)
    
    # Trading days per year
    TRADING_DAYS_PER_YEAR: int = 252
    TRADING_HOURS_PER_DAY: float = 6.5
    
    # Risk-free rate (annualized)
    RISK_FREE_RATE: float = 0.05  # 5%
    
    # Minimum values
    MIN_TRADE_VALUE: float = 1.0
    MIN_POSITION_SIZE: float = 0.001
    
    # Maximum values
    MAX_LEVERAGE: float = 4.0
    MAX_POSITION_PCT: float = 0.25  # 25% of portfolio
    MAX_SECTOR_EXPOSURE: float = 0.35  # 35% per sector
    
    # Slippage defaults (basis points)
    DEFAULT_SLIPPAGE_BPS: float = 5.0
    HIGH_VOLUME_SLIPPAGE_BPS: float = 2.0
    LOW_VOLUME_SLIPPAGE_BPS: float = 15.0
    
    # Commission defaults
    DEFAULT_COMMISSION_PCT: float = 0.001  # 0.1%
    MIN_COMMISSION: float = 0.0
    
    # Technical indicator defaults
    DEFAULT_SMA_PERIODS: list[int] = [10, 20, 50, 100, 200]
    DEFAULT_EMA_PERIODS: list[int] = [12, 26, 50, 100, 200]
    DEFAULT_RSI_PERIOD: int = 14
    DEFAULT_MACD_FAST: int = 12
    DEFAULT_MACD_SLOW: int = 26
    DEFAULT_MACD_SIGNAL: int = 9
    DEFAULT_BB_PERIOD: int = 20
    DEFAULT_BB_STD: float = 2.0
    DEFAULT_ATR_PERIOD: int = 14


class RiskConstants:
    """Risk management constants."""
    
    # Value at Risk
    VAR_CONFIDENCE_LEVELS: list[float] = [0.95, 0.99]
    VAR_LOOKBACK_DAYS: int = 252
    
    # Drawdown limits
    MAX_DRAWDOWN_PCT: float = 0.15  # 15%
    DAILY_LOSS_LIMIT_PCT: float = 0.02  # 2%
    WEEKLY_LOSS_LIMIT_PCT: float = 0.05  # 5%
    
    # Position limits
    MAX_POSITIONS: int = 20
    MAX_CORRELATED_POSITIONS: int = 5
    CORRELATION_THRESHOLD: float = 0.7
    
    # Circuit breakers
    CIRCUIT_BREAKER_THRESHOLD: float = 0.10  # 10% portfolio loss
    CIRCUIT_BREAKER_COOLDOWN_HOURS: int = 24
    
    # Kelly criterion
    KELLY_FRACTION: float = 0.25  # Quarter Kelly


class MLConstants:
    """Machine learning constants."""
    
    # Training
    DEFAULT_TEST_SIZE: float = 0.2
    DEFAULT_VAL_SIZE: float = 0.1
    CV_FOLDS: int = 5
    TIME_SERIES_CV_SPLITS: int = 5
    
    # Feature engineering
    LOOKBACK_PERIODS: list[int] = [5, 10, 20, 60, 120, 252]
    FORWARD_PERIODS: list[int] = [1, 5, 10, 20]
    
    # Model defaults
    RANDOM_STATE: int = 42
    N_JOBS: int = -1  # Use all cores
    
    # Early stopping
    EARLY_STOPPING_ROUNDS: int = 50
    MIN_IMPROVEMENT: float = 0.0001


# =============================================================================
# SETTINGS
# =============================================================================

class DatabaseSettings(BaseSettings):
    """Database configuration."""
    model_config = SettingsConfigDict(env_prefix="DB_")
    
    url: str = Field(
        default="sqlite:///data/trading.db",
        description="Database connection URL"
    )
    echo: bool = Field(default=False, description="Echo SQL queries")
    pool_size: int = Field(default=5, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Max overflow connections")


class AlpacaSettings(BaseSettings):
    """Alpaca broker configuration."""
    model_config = SettingsConfigDict(env_prefix="ALPACA_")
    
    api_key: str = Field(default="", description="Alpaca API key")
    secret_key: str = Field(default="", description="Alpaca secret key")
    base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        description="Alpaca API base URL"
    )
    data_url: str = Field(
        default="https://data.alpaca.markets",
        description="Alpaca data API URL"
    )
    paper_trading: bool = Field(
        default=True,
        description="Use paper trading mode"
    )
    
    @property
    def is_configured(self) -> bool:
        """Check if Alpaca credentials are configured."""
        return bool(self.api_key and self.secret_key)


class DataSettings(BaseSettings):
    """Data configuration."""
    model_config = SettingsConfigDict(env_prefix="DATA_")
    
    storage_path: Path = Field(
        default=Path("data/storage"),
        description="Raw data storage path"
    )
    processed_path: Path = Field(
        default=Path("data/processed"),
        description="Processed data path"
    )
    cache_path: Path = Field(
        default=Path("data/cache"),
        description="Cache directory path"
    )
    default_timeframe: TimeFrame = Field(
        default=TimeFrame.M15,
        description="Default data timeframe"
    )
    use_cache: bool = Field(default=True, description="Enable data caching")
    cache_ttl_hours: int = Field(default=24, description="Cache TTL in hours")


class BacktestSettings(BaseSettings):
    """Backtesting configuration."""
    model_config = SettingsConfigDict(env_prefix="BACKTEST_")
    
    initial_capital: float = Field(
        default=100000.0,
        description="Initial portfolio capital"
    )
    commission_pct: float = Field(
        default=0.001,
        description="Commission percentage per trade"
    )
    slippage_pct: float = Field(
        default=0.0005,
        description="Slippage percentage per trade"
    )
    margin_requirement: float = Field(
        default=0.25,
        description="Margin requirement for positions"
    )
    allow_shorting: bool = Field(
        default=True,
        description="Allow short positions"
    )
    fractional_shares: bool = Field(
        default=True,
        description="Allow fractional share trading"
    )


class RiskSettings(BaseSettings):
    """Risk management configuration."""
    model_config = SettingsConfigDict(env_prefix="RISK_")
    
    max_position_size: float = Field(
        default=0.10,
        description="Maximum position size as portfolio fraction"
    )
    max_portfolio_risk: float = Field(
        default=0.02,
        description="Maximum portfolio risk per trade"
    )
    max_drawdown: float = Field(
        default=0.15,
        description="Maximum allowed drawdown"
    )
    daily_loss_limit: float = Field(
        default=0.02,
        description="Daily loss limit as portfolio fraction"
    )
    use_stop_loss: bool = Field(
        default=True,
        description="Enable stop-loss orders"
    )
    default_stop_loss_pct: float = Field(
        default=0.02,
        description="Default stop-loss percentage"
    )
    use_take_profit: bool = Field(
        default=True,
        description="Enable take-profit orders"
    )
    default_take_profit_pct: float = Field(
        default=0.04,
        description="Default take-profit percentage"
    )
    var_confidence: float = Field(
        default=0.95,
        description="VaR confidence level"
    )


class MLSettings(BaseSettings):
    """Machine learning configuration."""
    model_config = SettingsConfigDict(env_prefix="ML_")
    
    models_path: Path = Field(
        default=Path("models/artifacts"),
        description="Trained models storage path"
    )
    experiment_tracking: bool = Field(
        default=True,
        description="Enable MLflow experiment tracking"
    )
    mlflow_uri: str = Field(
        default="sqlite:///mlflow.db",
        description="MLflow tracking URI"
    )
    auto_optimize: bool = Field(
        default=True,
        description="Enable hyperparameter optimization"
    )
    optimization_trials: int = Field(
        default=100,
        description="Number of Optuna trials"
    )
    use_gpu: bool = Field(
        default=False,
        description="Use GPU for training"
    )


class Settings(BaseSettings):
    """
    Main application settings.
    
    Loads configuration from environment variables and .env file.
    All sub-settings are nested for organization.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application
    app_name: str = Field(
        default="Algo Trading Platform",
        description="Application name"
    )
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    
    # Trading mode
    trading_mode: TradingMode = Field(
        default=TradingMode.BACKTEST,
        description="Trading execution mode"
    )
    
    # Timezone
    timezone: str = Field(default="America/New_York", description="Trading timezone")
    
    # Project paths
    project_root: Path = Field(
        default=Path("."),
        description="Project root directory"
    )
    
    # Sub-settings (nested)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    alpaca: AlpacaSettings = Field(default_factory=AlpacaSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    
    @model_validator(mode="after")
    def validate_paths(self) -> "Settings":
        """Ensure all paths exist."""
        paths = [
            self.data.storage_path,
            self.data.processed_path,
            self.data.cache_path,
            self.ml.models_path,
        ]
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)
        return self
    
    @field_validator("project_root", mode="before")
    @classmethod
    def resolve_project_root(cls, v: Any) -> Path:
        """Resolve project root to absolute path."""
        return Path(v).resolve()


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def configure_logging(settings: Settings) -> None:
    """
    Configure structured logging with structlog.
    
    Args:
        settings: Application settings
    """
    # Set log level
    log_level = getattr(logging, settings.log_level.value)
    
    # Configure structlog processors
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    if settings.debug:
        # Development: colored console output
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]
    else:
        # Production: JSON output
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ]
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )
    
    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (optional)
    
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


# =============================================================================
# SETTINGS SINGLETON
# =============================================================================

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Application settings singleton
    """
    settings = Settings()
    configure_logging(settings)
    return settings


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "TradingMode",
    "LogLevel",
    "TimeFrame",
    "OrderSide",
    "OrderType",
    "PositionSide",
    "SignalType",
    "MarketRegime",
    # Constants
    "TradingConstants",
    "RiskConstants",
    "MLConstants",
    # Settings
    "Settings",
    "DatabaseSettings",
    "AlpacaSettings",
    "DataSettings",
    "BacktestSettings",
    "RiskSettings",
    "MLSettings",
    # Functions
    "get_settings",
    "configure_logging",
    "get_logger",
]