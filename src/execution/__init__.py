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

__all__ = [
    'BrokerAPI',
    'AlpacaBroker',
    'IBKRBroker',
    'BrokerFactory',
    'ConnectionStatus',
    'OrderManager',
    'Order',
    'OrderStatus',
    'OrderType',
    'OrderQueue',
    'SmartOrderRouter',
    'ExecutionEngine',
    'ExecutionAlgo',
    'TWAPExecutor',
    'VWAPExecutor',
    'POVExecutor',
    'AdaptiveExecutor'
]
