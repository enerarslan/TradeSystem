"""
CORE MODULE
Event-Driven Trading System Core Components

Bu modül sistemin temel yapı taşlarını içerir:
- events.py: Tüm event tipleri ve factory
- bus.py: Async Event Bus (Pub/Sub)
- engine.py: Ana Trading Engine (Orchestrator)
"""

from core.events import (
    # Event Types
    EventType,
    EventPriority,
    
    # Base Event
    Event,
    
    # Market Data Events
    MarketDataEvent,
    BarEvent,
    
    # Signal Events
    SignalEvent,
    AlphaEvent,
    
    # Order Events
    OrderEvent,
    OrderSide,
    OrderType,
    OrderStatus,
    
    # Execution Events
    FillEvent,
    
    # Risk Events
    RiskEvent,
    RiskLevel,
    
    # Portfolio Events
    PositionUpdateEvent,
    PnLUpdateEvent,
    
    # System Events
    SystemEvent,
    HeartbeatEvent,
    
    # Factory
    EventFactory
)

from core.bus import (
    EventBus,
    EventBusMetrics,
    Subscription,
    get_event_bus,
    init_event_bus,
    shutdown_event_bus,
    on_event
)


__all__ = [
    # Events
    'EventType',
    'EventPriority',
    'Event',
    'MarketDataEvent',
    'BarEvent',
    'SignalEvent',
    'AlphaEvent',
    'OrderEvent',
    'OrderSide',
    'OrderType',
    'OrderStatus',
    'FillEvent',
    'RiskEvent',
    'RiskLevel',
    'PositionUpdateEvent',
    'PnLUpdateEvent',
    'SystemEvent',
    'HeartbeatEvent',
    'EventFactory',
    
    # Bus
    'EventBus',
    'EventBusMetrics',
    'Subscription',
    'get_event_bus',
    'init_event_bus',
    'shutdown_event_bus',
    'on_event'
]