"""
Core Module
===========

Core building blocks for the algorithmic trading platform.
Contains events, types, interfaces, and exceptions.

Author: Algo Trading Platform
License: MIT
"""

# =============================================================================
# TYPES - Data structures and domain objects
# =============================================================================

from core.types import (
    # Data structures
    OHLCV,
    Bar,
    Trade,
    Position,
    Order,
    Signal,
    SignalStrength,
    PortfolioState,
    PerformanceMetrics,
    
    # Enums
    OrderStatus,
    PositionStatus,
    
    # Base exceptions
    AlgoTradingError,
    
    # Data exceptions
    DataError,
    DataNotFoundError,
    DataValidationError,
    
    # Strategy exceptions
    StrategyError,
    StrategyInitializationError,
    SignalGenerationError,
    
    # Execution exceptions
    ExecutionError,
    OrderRejectedError,
    InsufficientFundsError,
    
    # Risk exceptions
    RiskError,
    RiskLimitExceededError,
    DrawdownLimitError,
    
    # Model exceptions
    ModelError,
    ModelNotTrainedError,
    PredictionError,
    
    # Other exceptions
    BacktestError,
    ConfigurationError,
)

# =============================================================================
# EVENTS - Event-driven architecture
# =============================================================================

from core.events import (
    # Enums
    EventType,
    EventPriority,
    
    # Base event
    Event,
    
    # Event types
    MarketEvent,
    MultiAssetMarketEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
    PortfolioEvent,
    RiskEvent,
    SystemEvent,
    
    # Event bus
    EventBus,
    EventHandler,
)

# =============================================================================
# INTERFACES - Abstract base classes and protocols
# =============================================================================

from core.interfaces import (
    DataProvider,
    Strategy,
    RiskManager,
    ExecutionHandler,
    PortfolioManager,
    Model,
    BacktestEngine,
    FeatureGenerator,
    Optimizer,
)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # === Types - Data Structures ===
    "OHLCV",
    "Bar",
    "Trade",
    "Position",
    "Order",
    "Signal",
    "SignalStrength",
    "PortfolioState",
    "PerformanceMetrics",
    
    # === Types - Enums ===
    "OrderStatus",
    "PositionStatus",
    
    # === Exceptions - Base ===
    "AlgoTradingError",
    
    # === Exceptions - Data ===
    "DataError",
    "DataNotFoundError",
    "DataValidationError",
    
    # === Exceptions - Strategy ===
    "StrategyError",
    "StrategyInitializationError",
    "SignalGenerationError",
    
    # === Exceptions - Execution ===
    "ExecutionError",
    "OrderRejectedError",
    "InsufficientFundsError",
    
    # === Exceptions - Risk ===
    "RiskError",
    "RiskLimitExceededError",
    "DrawdownLimitError",
    
    # === Exceptions - Model ===
    "ModelError",
    "ModelNotTrainedError",
    "PredictionError",
    
    # === Exceptions - Other ===
    "BacktestError",
    "ConfigurationError",
    
    # === Events - Enums ===
    "EventType",
    "EventPriority",
    
    # === Events - Classes ===
    "Event",
    "MarketEvent",
    "MultiAssetMarketEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    "PortfolioEvent",
    "RiskEvent",
    "SystemEvent",
    "EventBus",
    "EventHandler",
    
    # === Interfaces ===
    "DataProvider",
    "Strategy",
    "RiskManager",
    "ExecutionHandler",
    "PortfolioManager",
    "Model",
    "BacktestEngine",
    "FeatureGenerator",
    "Optimizer",
]