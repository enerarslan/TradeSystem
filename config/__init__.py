"""
Configuration Module
====================

Centralized configuration management for the algorithmic trading platform.

Modules:
- settings: Main settings and environment configuration
- symbols: Symbol management for 46 supported stocks

Author: Algo Trading Platform
License: MIT
"""

# =============================================================================
# SETTINGS - Core Configuration
# =============================================================================

from config.settings import (
    # Enums
    TradingMode,
    LogLevel,
    TimeFrame,
    OrderSide,
    OrderType,
    PositionSide,
    SignalType,
    
    # Settings classes
    Settings,
    DatabaseSettings,
    AlpacaSettings,
    DataSettings,
    BacktestSettings,
    RiskSettings,
    MLSettings,
    
    # Constants
    TradingConstants,
    RiskConstants,
    MLConstants,
    
    # Functions
    get_settings,
    configure_logging,
    get_logger,
)

# =============================================================================
# SYMBOLS - Stock Symbol Management
# =============================================================================

from config.symbols import (
    # Enums
    Sector,
    MarketCapTier,
    Index,
    
    # Data classes
    SymbolInfo,
    
    # Symbol data
    SYMBOL_INFO,
    ALL_SYMBOLS,
    DOW_JONES_SYMBOLS,
    NASDAQ100_SYMBOLS,
    TECH_SYMBOLS,
    HEALTHCARE_SYMBOLS,
    FINANCIAL_SYMBOLS,
    MEGA_CAP_SYMBOLS,
    CORE_SYMBOLS,
    
    # Functions
    get_symbol_info,
    get_symbols_by_sector,
    get_symbols_by_index,
    validate_symbol,
    validate_symbols,
    discover_symbols_from_data,
    get_sector_allocation,
    
    # Model naming utilities
    get_model_filename,
    parse_model_filename,
    get_model_directory,
)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # === Settings - Enums ===
    "TradingMode",
    "LogLevel",
    "TimeFrame",
    "OrderSide",
    "OrderType",
    "PositionSide",
    "SignalType",
    
    # === Settings - Classes ===
    "Settings",
    "DatabaseSettings",
    "AlpacaSettings",
    "DataSettings",
    "BacktestSettings",
    "RiskSettings",
    "MLSettings",
    
    # === Settings - Constants ===
    "TradingConstants",
    "RiskConstants",
    "MLConstants",
    
    # === Settings - Functions ===
    "get_settings",
    "configure_logging",
    "get_logger",
    
    # === Symbols - Enums ===
    "Sector",
    "MarketCapTier",
    "Index",
    
    # === Symbols - Data Classes ===
    "SymbolInfo",
    
    # === Symbols - Symbol Data ===
    "SYMBOL_INFO",
    "ALL_SYMBOLS",
    "DOW_JONES_SYMBOLS",
    "NASDAQ100_SYMBOLS",
    "TECH_SYMBOLS",
    "HEALTHCARE_SYMBOLS",
    "FINANCIAL_SYMBOLS",
    "MEGA_CAP_SYMBOLS",
    "CORE_SYMBOLS",
    
    # === Symbols - Functions ===
    "get_symbol_info",
    "get_symbols_by_sector",
    "get_symbols_by_index",
    "validate_symbol",
    "validate_symbols",
    "discover_symbols_from_data",
    "get_sector_allocation",
    
    # === Symbols - Model Naming ===
    "get_model_filename",
    "parse_model_filename",
    "get_model_directory",
]