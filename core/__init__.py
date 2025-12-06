"""
Core Module
===========

Core building blocks for the algorithmic trading platform.
Contains events, types, interfaces, and exceptions.

Author: Algo Trading Platform
License: MIT
"""

from core.types import (
    OHLCV,
    Bar,
    Trade,
    Position,
    Order,
    Signal,
    PortfolioState,
    PerformanceMetrics,
)
from core.events import (
    Event,
    MarketEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
    PortfolioEvent,
    RiskEvent,
)
from core.interfaces import (
    DataProvider,
    Strategy,
    RiskManager,
    ExecutionHandler,
    PortfolioManager,
    Model,
)

__all__ = [
    # Types
    "OHLCV",
    "Bar",
    "Trade",
    "Position",
    "Order",
    "Signal",
    "PortfolioState",
    "PerformanceMetrics",
    # Events
    "Event",
    "MarketEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    "PortfolioEvent",
    "RiskEvent",
    # Interfaces
    "DataProvider",
    "Strategy",
    "RiskManager",
    "ExecutionHandler",
    "PortfolioManager",
    "Model",
]