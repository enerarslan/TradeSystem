"""
Execution Module
================

Live trading execution for the algorithmic trading platform.
Provides broker integration, order management, and execution algorithms.

Components:
- broker: Alpaca integration with paper trading support
- algorithms: TWAP, VWAP, Iceberg, POV execution algorithms

Architecture:
- BrokerBase: Abstract base for all broker implementations
- AlpacaBroker: Production Alpaca integration
- PaperBroker: Paper trading simulator
- ExecutionAlgorithm: Smart order execution

Author: Algo Trading Platform
License: MIT
"""

from execution.broker import (
    # Enums
    BrokerType,
    ConnectionStatus,
    # Configuration
    BrokerConfig,
    # Base classes
    BrokerBase,
    # Implementations
    AlpacaBroker,
    PaperBroker,
    # Factory
    create_broker,
    # Order Manager
    OrderManager,
    # Position Reconciler
    PositionReconciler,
)

from execution.algorithms import (
    # Enums
    ExecutionAlgoType,
    ExecutionStatus,
    # Configuration
    TWAPConfig,
    VWAPConfig,
    IcebergConfig,
    POVConfig,
    # Base
    ExecutionAlgorithm,
    # Implementations
    TWAPAlgorithm,
    VWAPAlgorithm,
    IcebergAlgorithm,
    POVAlgorithm,
    # Factory
    create_execution_algorithm,
    # Smart Router
    SmartOrderRouter,
)

from execution.live_engine import (
    # Enums
    TradingEngineState,
    TradingSession,
    # Config
    LiveTradingConfig,
    # Stats
    TradingStatistics,
    # Engine
    LiveTradingEngine,
    # Functions
    run_paper_trading,
    run_live_trading,
)

__all__ = [
    # === Broker Enums ===
    "BrokerType",
    "ConnectionStatus",
    
    # === Broker Config ===
    "BrokerConfig",
    
    # === Broker Classes ===
    "BrokerBase",
    "AlpacaBroker",
    "PaperBroker",
    
    # === Broker Factory ===
    "create_broker",
    
    # === Order Management ===
    "OrderManager",
    "PositionReconciler",
    
    # === Algorithm Enums ===
    "ExecutionAlgoType",
    "ExecutionStatus",
    
    # === Algorithm Configs ===
    "TWAPConfig",
    "VWAPConfig",
    "IcebergConfig",
    "POVConfig",
    
    # === Algorithm Classes ===
    "ExecutionAlgorithm",
    "TWAPAlgorithm",
    "VWAPAlgorithm",
    "IcebergAlgorithm",
    "POVAlgorithm",
    
    # === Algorithm Factory ===
    "create_execution_algorithm",
    
    # === Smart Routing ===
    "SmartOrderRouter",
    
    # === Live Engine Enums ===
    "TradingEngineState",
    "TradingSession",
    
    # === Live Engine Config ===
    "LiveTradingConfig",
    
    # === Live Engine Stats ===
    "TradingStatistics",
    
    # === Live Engine ===
    "LiveTradingEngine",
    
    # === Live Engine Functions ===
    "run_paper_trading",
    "run_live_trading",
]