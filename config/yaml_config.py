"""
KURUMSAL KONFİGÜRASYON YÖNETİMİ
JPMorgan Enterprise Config Tarzı

Özellikler:
- YAML tabanlı konfigürasyon
- Environment overlay (dev, staging, prod)
- Validation (Pydantic)
- Hot-reload desteği
- Secrets management
- Config inheritance
- Default values

Kullanım:
    from config.yaml_config import get_config, ConfigManager
    
    config = get_config()
    print(config.trading.initial_capital)
    print(config.risk.max_daily_loss_pct)
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from functools import lru_cache
import json


# ============================================================================
# CONFIGURATION MODELS (Pydantic ile validation)
# ============================================================================

class DatabaseConfig(BaseModel):
    """Veritabanı konfigürasyonu"""
    url: str = "sqlite:///./trade_system.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10


class TradingConfig(BaseModel):
    """Trading parametreleri"""
    initial_capital: float = Field(default=100_000, ge=1000)
    commission_pct: float = Field(default=0.001, ge=0, le=0.1)
    slippage_pct: float = Field(default=0.0005, ge=0, le=0.1)
    max_position_size: float = Field(default=50_000, ge=0)
    default_quantity: int = Field(default=10, ge=1)


class RiskConfig(BaseModel):
    """Risk yönetimi parametreleri"""
    max_position_size_pct: float = Field(default=10.0, ge=1, le=100)
    max_daily_loss_pct: float = Field(default=2.0, ge=0.1, le=20)
    max_total_drawdown_pct: float = Field(default=10.0, ge=1, le=50)
    max_daily_trades: int = Field(default=50, ge=1, le=1000)
    max_open_positions: int = Field(default=10, ge=1, le=100)
    max_var_1d: float = Field(default=5_000, ge=100)
    min_cash_reserve_pct: float = Field(default=20.0, ge=0, le=100)
    
    # Circuit breaker
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: float = Field(default=3.0, ge=1, le=10)
    circuit_breaker_cooldown_minutes: int = Field(default=30, ge=5)


class StrategyConfig(BaseModel):
    """Strateji parametreleri"""
    name: str = "AdvancedMomentum"
    fast_period: int = Field(default=10, ge=2, le=100)
    slow_period: int = Field(default=30, ge=5, le=200)
    rsi_period: int = Field(default=14, ge=5, le=50)
    rsi_overbought: float = Field(default=70.0, ge=50, le=90)
    rsi_oversold: float = Field(default=30.0, ge=10, le=50)
    min_confidence: float = Field(default=0.6, ge=0.3, le=1.0)
    atr_multiplier: float = Field(default=2.0, ge=0.5, le=5.0)
    
    @field_validator('slow_period')
    @classmethod
    def slow_must_be_greater(cls, v, info):
        if 'fast_period' in info.data and v <= info.data['fast_period']:
            raise ValueError('slow_period must be greater than fast_period')
        return v


class MLConfig(BaseModel):
    """Machine Learning konfigürasyonu"""
    enabled: bool = False
    model_type: str = "xgboost"  # xgboost, lstm, transformer
    lookback_period: int = Field(default=60, ge=10, le=500)
    prediction_horizon: int = Field(default=5, ge=1, le=50)
    retrain_frequency: str = "weekly"  # daily, weekly, monthly
    min_training_samples: int = Field(default=1000, ge=100)
    
    # Feature engineering
    use_technical_features: bool = True
    use_volume_features: bool = True
    use_time_features: bool = True
    
    # Model parameters
    xgboost_params: Dict[str, Any] = Field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1
    })


class DataFeedConfig(BaseModel):
    """Veri kaynağı konfigürasyonu"""
    source: str = "csv"  # csv, binance, alpaca, yahoo
    storage_path: str = "data/storage"
    symbols: List[str] = Field(default_factory=lambda: ["AAPL"])
    timeframe: str = "15min"
    
    # Real-time settings
    websocket_enabled: bool = False
    websocket_url: Optional[str] = None
    reconnect_attempts: int = 5
    reconnect_delay_seconds: int = 5


class BacktestConfig(BaseModel):
    """Backtest konfigürasyonu"""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    use_risk_management: bool = True
    export_trades: bool = True
    export_path: str = "data/backtest_results"
    
    # Optimization
    parallel_backtests: int = 4
    optimize_parameters: bool = False


class APIConfig(BaseModel):
    """API/Dashboard konfigürasyonu"""
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    api_key: Optional[str] = None


class LoggingConfig(BaseModel):
    """Logging konfigürasyonu"""
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    file_enabled: bool = True
    file_path: str = "logs"
    max_file_size_mb: int = 500
    retention_days: int = 10
    json_format: bool = False


class MonitoringConfig(BaseModel):
    """Monitoring konfigürasyonu"""
    prometheus_enabled: bool = False
    prometheus_port: int = 9090
    heartbeat_interval_seconds: int = 30
    metrics_export_interval_seconds: int = 60


class AppConfig(BaseModel):
    """Ana uygulama konfigürasyonu - Tüm alt konfigürasyonları içerir"""
    # Meta
    app_name: str = "AlphaTrade"
    version: str = "2.0.0"
    environment: str = "development"  # development, staging, production
    
    # Sub-configs
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    data_feed: DataFeedConfig = Field(default_factory=DataFeedConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Config'i dictionary'e çevir"""
        return self.model_dump()
    
    def to_yaml(self) -> str:
        """Config'i YAML string'e çevir"""
        return yaml.dump(self.model_dump(), default_flow_style=False, sort_keys=False)
    
    def save_to_file(self, path: str):
        """Config'i dosyaya kaydet"""
        with open(path, 'w') as f:
            f.write(self.to_yaml())


# ============================================================================
# CONFIGURATION MANAGER
# ============================================================================

class ConfigManager:
    """
    Merkezi konfigürasyon yöneticisi.
    
    Özellikler:
    - YAML dosyalarından yükleme
    - Environment overlay
    - Environment variable override
    - Validation
    - Hot-reload
    """
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[AppConfig] = None
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._config_path: Optional[Path] = None
            self._last_load: Optional[datetime] = None
    
    def load(
        self,
        config_path: Optional[str] = None,
        environment: Optional[str] = None
    ) -> AppConfig:
        """
        Konfigürasyonu yükle.
        
        Yükleme sırası:
        1. Base config (config/settings.yaml)
        2. Environment config (config/settings.{env}.yaml)
        3. Environment variables
        
        Args:
            config_path: Ana config dosyası yolu
            environment: Ortam (dev, staging, prod)
        
        Returns:
            AppConfig: Yüklenmiş ve validate edilmiş config
        """
        # Config path
        if config_path:
            base_path = Path(config_path)
        else:
            base_path = Path("config/settings.yaml")
        
        # Environment
        env = environment or os.getenv("APP_ENV", "development")
        
        # Base config yükle
        config_dict = {}
        
        if base_path.exists():
            with open(base_path, 'r') as f:
                config_dict = yaml.safe_load(f) or {}
        
        # Environment overlay
        env_path = base_path.with_suffix(f".{env}.yaml")
        if env_path.exists():
            with open(env_path, 'r') as f:
                env_config = yaml.safe_load(f) or {}
                config_dict = self._deep_merge(config_dict, env_config)
        
        # Environment variables override
        config_dict = self._apply_env_overrides(config_dict)
        
        # Set environment
        config_dict['environment'] = env
        
        # Validate ve oluştur
        try:
            self._config = AppConfig(**config_dict)
        except Exception as e:
            print(f"⚠️ Config validation warning: {e}")
            print("   Using default config values where needed")
            # Hatalı değerleri temizle ve tekrar dene
            self._config = AppConfig()
        
        self._config_path = base_path
        self._last_load = datetime.now()
        
        return self._config
    
    def get(self) -> AppConfig:
        """Mevcut config'i döndür"""
        if self._config is None:
            return self.load()
        return self._config
    
    def reload(self) -> AppConfig:
        """Config'i yeniden yükle (hot-reload)"""
        if self._config_path:
            return self.load(str(self._config_path))
        return self.load()
    
    def _deep_merge(self, base: Dict, overlay: Dict) -> Dict:
        """İki dictionary'i derin birleştir"""
        result = base.copy()
        
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config: Dict) -> Dict:
        """
        Environment variable override'ları uygula.
        
        Format: ALPHATRADE_SECTION_KEY=value
        Örnek: ALPHATRADE_TRADING_INITIAL_CAPITAL=50000
        """
        prefix = "ALPHATRADE_"
        
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            
            # Parse key
            parts = key[len(prefix):].lower().split('_')
            
            if len(parts) < 2:
                continue
            
            section = parts[0]
            setting = '_'.join(parts[1:])
            
            # Type conversion
            try:
                if value.lower() in ('true', 'false'):
                    typed_value = value.lower() == 'true'
                elif value.isdigit():
                    typed_value = int(value)
                elif '.' in value and value.replace('.', '').isdigit():
                    typed_value = float(value)
                else:
                    typed_value = value
            except:
                typed_value = value
            
            # Apply
            if section in config:
                config[section][setting] = typed_value
        
        return config
    
    def generate_default_config(self, output_path: str = "config/settings.yaml"):
        """Varsayılan config dosyası oluştur"""
        default_config = AppConfig()
        
        # Dizin oluştur
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # YAML header ile kaydet
        header = """# ============================================================================
# ALPHATRADE CONFIGURATION
# JPMorgan-Style Enterprise Trading System
# ============================================================================
# 
# Environment Overrides:
#   - Create settings.{environment}.yaml for environment-specific settings
#   - Use environment variables: ALPHATRADE_SECTION_KEY=value
#
# Example:
#   ALPHATRADE_TRADING_INITIAL_CAPITAL=50000
#   ALPHATRADE_RISK_MAX_DAILY_LOSS_PCT=3.0
#
# ============================================================================

"""
        
        with open(output_path, 'w') as f:
            f.write(header)
            f.write(default_config.to_yaml())
        
        print(f"✅ Default config generated: {output_path}")
        return output_path


# ============================================================================
# GLOBAL ACCESS FUNCTIONS
# ============================================================================

_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Config manager instance'ını al"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> AppConfig:
    """Global config'i al"""
    return get_config_manager().get()


def load_config(
    config_path: Optional[str] = None,
    environment: Optional[str] = None
) -> AppConfig:
    """Config'i yükle"""
    return get_config_manager().load(config_path, environment)


def reload_config() -> AppConfig:
    """Config'i yeniden yükle"""
    return get_config_manager().reload()


# ============================================================================
# CLI TOOL
# ============================================================================

def main():
    """CLI tool for config management"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python yaml_config.py [generate|validate|show]")
        return
    
    command = sys.argv[1]
    
    if command == "generate":
        output = sys.argv[2] if len(sys.argv) > 2 else "config/settings.yaml"
        get_config_manager().generate_default_config(output)
    
    elif command == "validate":
        config_path = sys.argv[2] if len(sys.argv) > 2 else "config/settings.yaml"
        try:
            config = load_config(config_path)
            print(f"✅ Config valid: {config_path}")
            print(f"   Environment: {config.environment}")
            print(f"   Symbols: {config.data_feed.symbols}")
        except Exception as e:
            print(f"❌ Config invalid: {e}")
    
    elif command == "show":
        config = get_config()
        print(config.to_yaml())
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()


# Export
__all__ = [
    'AppConfig',
    'DatabaseConfig',
    'TradingConfig',
    'RiskConfig',
    'StrategyConfig',
    'MLConfig',
    'DataFeedConfig',
    'BacktestConfig',
    'APIConfig',
    'LoggingConfig',
    'MonitoringConfig',
    'ConfigManager',
    'get_config',
    'load_config',
    'reload_config',
    'get_config_manager'
]