"""
KURUMSAL KONFİGÜRASYON - LEGACY UYUMLULUK MODÜLÜ
JPMorgan Enterprise Trading System

Bu modül yaml_config.py ile tam uyumlu çalışır.
Eski kod tabanıyla geriye dönük uyumluluk sağlar.

Kullanım:
    from config.settings import settings
    
    print(settings.APP_MODE)
    print(settings.VERSION)
    print(settings.INITIAL_CAPITAL)
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class Settings:
    """
    Merkezi konfigürasyon sınıfı.
    
    Environment variables ile override edilebilir:
        APP_MODE=PRODUCTION python main.py
    """
    
    # Application
    APP_NAME: str = "AlphaTrade"
    VERSION: str = "2.1.0"
    APP_MODE: str = field(default_factory=lambda: os.getenv("APP_MODE", "DEVELOPMENT"))
    DEBUG: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    
    # Database
    DATABASE_URL: str = field(default_factory=lambda: os.getenv(
        "DATABASE_URL", 
        "sqlite+aiosqlite:///./trade_history.db"
    ))
    DATABASE_ECHO: bool = False
    
    # Trading
    INITIAL_CAPITAL: float = field(default_factory=lambda: float(os.getenv("INITIAL_CAPITAL", "100000")))
    COMMISSION_PCT: float = 0.001  # 0.1%
    SLIPPAGE_PCT: float = 0.0005  # 0.05%
    MAX_POSITION_SIZE: float = 50000
    DEFAULT_QUANTITY: int = 10
    
    # Risk Management
    MAX_POSITION_SIZE_PCT: float = 10.0
    MAX_DAILY_LOSS_PCT: float = 2.0
    MAX_TOTAL_DRAWDOWN_PCT: float = 10.0
    MAX_DAILY_TRADES: int = 50
    MAX_OPEN_POSITIONS: int = 10
    MAX_VAR_1D: float = 5000
    MIN_CASH_RESERVE_PCT: float = 20.0
    
    # Circuit Breaker
    CIRCUIT_BREAKER_ENABLED: bool = True
    CIRCUIT_BREAKER_THRESHOLD: float = 3.0
    CIRCUIT_BREAKER_COOLDOWN_MINUTES: int = 30
    
    # Strategy
    STRATEGY_NAME: str = "AdvancedMomentum"
    FAST_PERIOD: int = 10
    SLOW_PERIOD: int = 30
    RSI_PERIOD: int = 14
    RSI_OVERBOUGHT: float = 70.0
    RSI_OVERSOLD: float = 30.0
    MIN_CONFIDENCE: float = 0.6
    ATR_MULTIPLIER: float = 2.0
    
    # Machine Learning
    ML_ENABLED: bool = False
    ML_MODEL_TYPE: str = "xgboost"
    ML_LOOKBACK_PERIOD: int = 60
    ML_PREDICTION_HORIZON: int = 5
    ML_RETRAIN_FREQUENCY: str = "weekly"
    ML_MIN_TRAINING_SAMPLES: int = 1000
    
    # Data Feed
    DATA_SOURCE: str = "csv"
    DATA_STORAGE_PATH: str = "data/storage"
    SYMBOLS: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"
    ])
    TIMEFRAME: str = "15min"
    
    # WebSocket
    WEBSOCKET_ENABLED: bool = False
    WEBSOCKET_URL: Optional[str] = None
    RECONNECT_ATTEMPTS: int = 5
    RECONNECT_DELAY_SECONDS: int = 5
    
    # Backtest
    BACKTEST_START_DATE: Optional[str] = None
    BACKTEST_END_DATE: Optional[str] = None
    BACKTEST_USE_RISK: bool = True
    BACKTEST_EXPORT_TRADES: bool = True
    BACKTEST_EXPORT_PATH: str = "data/backtest_results"
    
    # API
    API_ENABLED: bool = False
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_KEY: Optional[str] = field(default_factory=lambda: os.getenv("API_KEY"))
    
    # Logging
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    LOG_FILE_ENABLED: bool = True
    LOG_FILE_PATH: str = "logs"
    LOG_MAX_SIZE_MB: int = 500
    LOG_RETENTION_DAYS: int = 10
    LOG_JSON_FORMAT: bool = False
    
    # Monitoring
    PROMETHEUS_ENABLED: bool = False
    PROMETHEUS_PORT: int = 9090
    HEARTBEAT_INTERVAL_SECONDS: int = 30
    METRICS_EXPORT_INTERVAL_SECONDS: int = 60
    
    # Paths
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Ensure directories exist
        Path(self.DATA_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
        Path(self.BACKTEST_EXPORT_PATH).mkdir(parents=True, exist_ok=True)
        Path(self.LOG_FILE_PATH).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            key: getattr(self, key) 
            for key in dir(self) 
            if not key.startswith('_') and key.isupper()
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get setting by key name"""
        return getattr(self, key, default)
    
    def update(self, **kwargs):
        """Update settings dynamically"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @classmethod
    def from_env(cls) -> 'Settings':
        """Create settings from environment variables"""
        return cls()
    
    @classmethod
    def from_yaml(cls, config_path: str = "config/settings.yaml") -> 'Settings':
        """Create settings from YAML file using yaml_config module"""
        try:
            from config.yaml_config import load_config
            yaml_config = load_config(config_path)
            
            instance = cls()
            
            # Map YAML config to settings
            if yaml_config.trading:
                instance.INITIAL_CAPITAL = yaml_config.trading.initial_capital
                instance.COMMISSION_PCT = yaml_config.trading.commission_pct
                instance.SLIPPAGE_PCT = yaml_config.trading.slippage_pct
            
            if yaml_config.risk:
                instance.MAX_POSITION_SIZE_PCT = yaml_config.risk.max_position_size_pct
                instance.MAX_DAILY_LOSS_PCT = yaml_config.risk.max_daily_loss_pct
                instance.MAX_DAILY_TRADES = yaml_config.risk.max_daily_trades
            
            if yaml_config.strategy:
                instance.STRATEGY_NAME = yaml_config.strategy.name
                instance.FAST_PERIOD = yaml_config.strategy.fast_period
                instance.SLOW_PERIOD = yaml_config.strategy.slow_period
            
            if yaml_config.ml:
                instance.ML_ENABLED = yaml_config.ml.enabled
                instance.ML_MODEL_TYPE = yaml_config.ml.model_type
            
            if yaml_config.data_feed:
                instance.DATA_SOURCE = yaml_config.data_feed.source
                instance.SYMBOLS = yaml_config.data_feed.symbols
            
            if yaml_config.logging:
                instance.LOG_LEVEL = yaml_config.logging.level
            
            return instance
            
        except Exception as e:
            print(f"Warning: Could not load YAML config: {e}")
            return cls()


# Global singleton instance
settings = Settings()


# Export
__all__ = ['Settings', 'settings']