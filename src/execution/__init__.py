"""
Execution module for AlphaTrade system.

This module provides:
- Order management
- Slippage modeling
- Transaction cost analysis
"""

from src.execution.order_manager import (
    OrderManager,
    Order,
    OrderStatus,
    OrderType,
)
from src.execution.slippage import (
    SlippageModel,
    calculate_slippage,
)
from src.execution.transaction_cost import (
    TransactionCostAnalyzer,
    estimate_costs,
)

__all__ = [
    "OrderManager",
    "Order",
    "OrderStatus",
    "OrderType",
    "SlippageModel",
    "calculate_slippage",
    "TransactionCostAnalyzer",
    "estimate_costs",
]
