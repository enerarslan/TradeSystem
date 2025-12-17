"""
Event system for event-driven backtesting.

This module provides:
- Base event classes
- Market events (tick, bar, order book)
- Signal events (trading signals)
- Order events (new, fill, cancel)
- Event queue management

Designed for institutional-grade backtesting with:
- Microsecond-precision timestamps
- Full event audit trail
- Async event processing support
"""

from src.backtesting.events.base import (
    Event,
    EventType,
    EventPriority,
)
from src.backtesting.events.market import (
    MarketEvent,
    TickEvent,
    BarEvent,
    OrderBookEvent,
    OrderBookLevel,
)
from src.backtesting.events.signal import (
    SignalEvent,
    SignalType,
)
from src.backtesting.events.order import (
    OrderEvent,
    OrderType,
    OrderSide,
    TimeInForce,
)
from src.backtesting.events.fill import (
    FillEvent,
    FillType,
)
from src.backtesting.events.queue import (
    EventQueue,
    PriorityEventQueue,
)

__all__ = [
    # Base
    "Event",
    "EventType",
    "EventPriority",
    # Market
    "MarketEvent",
    "TickEvent",
    "BarEvent",
    "OrderBookEvent",
    "OrderBookLevel",
    # Signal
    "SignalEvent",
    "SignalType",
    # Order
    "OrderEvent",
    "OrderType",
    "OrderSide",
    "TimeInForce",
    # Fill
    "FillEvent",
    "FillType",
    # Queue
    "EventQueue",
    "PriorityEventQueue",
]
