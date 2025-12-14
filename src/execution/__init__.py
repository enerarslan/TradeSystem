"""
Execution Layer Module
JPMorgan-Level Order Management and Trade Execution
"""

from .broker_api import (
    BrokerAPI, AlpacaBroker, IBKRBroker,
    BrokerFactory, ConnectionStatus
)
from .order_manager import (
    OrderManager, Order, OrderStatus, OrderType,
    OrderQueue, SmartOrderRouter
)
from .executor import (
    ExecutionEngine, ExecutionAlgo, TWAPExecutor,
    VWAPExecutor, POVExecutor, AdaptiveExecutor
)
from .async_pipeline import (
    AsyncTradingPipeline, PipelineBuilder, PipelineConfig,
    PipelineState, PipelineMetrics, run_async_pipeline
)
from .protected_positions import (
    ProtectedPositionManager, ProtectionConfig,
    ProtectedPosition, ProtectionStatus
)
from .reconciliation import (
    ReconciliationEngine, ReconciliationReport,
    Discrepancy, DiscrepancyType
)

__all__ = [
    # Broker API
    'BrokerAPI',
    'AlpacaBroker',
    'IBKRBroker',
    'BrokerFactory',
    'ConnectionStatus',
    # Order Management
    'OrderManager',
    'Order',
    'OrderStatus',
    'OrderType',
    'OrderQueue',
    'SmartOrderRouter',
    # Execution
    'ExecutionEngine',
    'ExecutionAlgo',
    'TWAPExecutor',
    'VWAPExecutor',
    'POVExecutor',
    'AdaptiveExecutor',
    # Async Pipeline
    'AsyncTradingPipeline',
    'PipelineBuilder',
    'PipelineConfig',
    'PipelineState',
    'PipelineMetrics',
    'run_async_pipeline',
    # Protected Positions (CRITICAL)
    'ProtectedPositionManager',
    'ProtectionConfig',
    'ProtectedPosition',
    'ProtectionStatus',
    # Reconciliation (CRITICAL)
    'ReconciliationEngine',
    'ReconciliationReport',
    'Discrepancy',
    'DiscrepancyType',
]
