"""
Events Module
=============

Event-driven architecture for the algorithmic trading platform.
All system components communicate through events for loose coupling.

Event Types:
- MarketEvent: New market data received
- SignalEvent: Strategy generated trading signal
- OrderEvent: Order created/updated
- FillEvent: Order filled
- PortfolioEvent: Portfolio state change
- RiskEvent: Risk limit triggered

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, TypeVar
from uuid import UUID, uuid4

import polars as pl


# =============================================================================
# EVENT TYPES
# =============================================================================

class EventType(str, Enum):
    """Event type enumeration."""
    MARKET = "market"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"
    PORTFOLIO = "portfolio"
    RISK = "risk"
    SYSTEM = "system"


class EventPriority(int, Enum):
    """Event processing priority."""
    CRITICAL = 0  # Process immediately
    HIGH = 1
    NORMAL = 2
    LOW = 3


# =============================================================================
# BASE EVENT
# =============================================================================

@dataclass
class Event(ABC):
    """
    Abstract base event class.
    
    All events in the system inherit from this class.
    
    Attributes:
        id: Unique event identifier
        timestamp: Event creation timestamp
        event_type: Type of event
        priority: Processing priority
    """
    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: EventPriority = EventPriority.NORMAL
    
    @property
    @abstractmethod
    def event_type(self) -> EventType:
        """Return the event type."""
        pass
    
    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "priority": self.priority.value,
        }


# =============================================================================
# MARKET EVENTS
# =============================================================================

@dataclass
class MarketEvent(Event):
    """
    Event triggered when new market data is received.
    
    Contains OHLCV data for one or more symbols.
    
    Attributes:
        symbol: Trading symbol
        data: Market data (Polars DataFrame with OHLCV)
        timeframe: Data timeframe
        is_realtime: Whether data is real-time
    """
    symbol: str = ""
    data: pl.DataFrame | None = None
    timeframe: str = "15min"
    is_realtime: bool = False
    
    @property
    def event_type(self) -> EventType:
        return EventType.MARKET
    
    @property
    def latest_bar(self) -> dict[str, Any] | None:
        """Get the latest bar from data."""
        if self.data is None or len(self.data) == 0:
            return None
        return self.data.tail(1).to_dicts()[0]
    
    @property
    def latest_close(self) -> float | None:
        """Get the latest close price."""
        bar = self.latest_bar
        return bar.get("close") if bar else None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update({
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "is_realtime": self.is_realtime,
            "num_bars": len(self.data) if self.data is not None else 0,
        })
        return base


@dataclass
class MultiAssetMarketEvent(Event):
    """
    Event for multiple assets' market data.
    
    Attributes:
        data: Dictionary mapping symbol to DataFrame
        timeframe: Data timeframe
        is_realtime: Whether data is real-time
    """
    data: dict[str, pl.DataFrame] = field(default_factory=dict)
    timeframe: str = "15min"
    is_realtime: bool = False
    
    @property
    def event_type(self) -> EventType:
        return EventType.MARKET
    
    @property
    def symbols(self) -> list[str]:
        """Get list of symbols in this event."""
        return list(self.data.keys())
    
    def get_symbol_data(self, symbol: str) -> pl.DataFrame | None:
        """Get data for a specific symbol."""
        return self.data.get(symbol)


# =============================================================================
# SIGNAL EVENTS
# =============================================================================

@dataclass
class SignalEvent(Event):
    """
    Event triggered when a strategy generates a trading signal.
    
    Attributes:
        symbol: Trading symbol
        signal_type: Type of signal (entry/exit)
        direction: Signal direction (1=long, -1=short, 0=neutral)
        strength: Signal strength (0.0 to 1.0)
        price: Current price at signal generation
        strategy_name: Name of the generating strategy
        target_price: Optional target price
        stop_loss: Optional stop loss price
        take_profit: Optional take profit price
        confidence: Model confidence (for ML strategies)
        features: Features used for signal generation
        metadata: Additional signal metadata
    """
    symbol: str = ""
    signal_type: str = "entry"  # entry, exit, scale_in, scale_out
    direction: int = 0  # 1=long, -1=short, 0=neutral
    strength: float = 0.0
    price: float = 0.0
    strategy_name: str = ""
    target_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    confidence: float | None = None
    features: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.HIGH
    
    @property
    def event_type(self) -> EventType:
        return EventType.SIGNAL
    
    @property
    def is_entry(self) -> bool:
        """Check if this is an entry signal."""
        return self.signal_type in ("entry", "scale_in")
    
    @property
    def is_exit(self) -> bool:
        """Check if this is an exit signal."""
        return self.signal_type in ("exit", "scale_out")
    
    @property
    def is_long(self) -> bool:
        """Check if this is a long signal."""
        return self.direction > 0
    
    @property
    def is_short(self) -> bool:
        """Check if this is a short signal."""
        return self.direction < 0
    
    @property
    def suggested_quantity(self) -> float | None:
        """Calculate suggested position size based on strength."""
        return self.metadata.get("suggested_quantity")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update({
            "symbol": self.symbol,
            "signal_type": self.signal_type,
            "direction": self.direction,
            "strength": self.strength,
            "price": self.price,
            "strategy_name": self.strategy_name,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "confidence": self.confidence,
            "metadata": self.metadata,
        })
        return base


# =============================================================================
# ORDER EVENTS
# =============================================================================

@dataclass
class OrderEvent(Event):
    """
    Event triggered when an order is created or updated.
    
    Attributes:
        order_id: Unique order identifier
        symbol: Trading symbol
        side: Order side (buy/sell)
        order_type: Order type (market/limit/stop)
        quantity: Order quantity
        price: Limit price (if applicable)
        stop_price: Stop price (if applicable)
        time_in_force: Order duration (day/gtc/ioc/fok)
        status: Order status
        signal_id: Related signal ID
    """
    order_id: UUID = field(default_factory=uuid4)
    symbol: str = ""
    side: str = "buy"
    order_type: str = "market"
    quantity: float = 0.0
    price: float | None = None
    stop_price: float | None = None
    time_in_force: str = "day"
    status: str = "pending"
    signal_id: UUID | None = None
    priority: EventPriority = EventPriority.HIGH
    
    @property
    def event_type(self) -> EventType:
        return EventType.ORDER
    
    @property
    def is_buy(self) -> bool:
        """Check if this is a buy order."""
        return self.side.lower() == "buy"
    
    @property
    def is_sell(self) -> bool:
        """Check if this is a sell order."""
        return self.side.lower() == "sell"
    
    @property
    def notional_value(self) -> float:
        """Calculate notional value."""
        price = self.price or 0.0
        return self.quantity * price
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update({
            "order_id": str(self.order_id),
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force,
            "status": self.status,
            "signal_id": str(self.signal_id) if self.signal_id else None,
        })
        return base


# =============================================================================
# FILL EVENTS
# =============================================================================

@dataclass
class FillEvent(Event):
    """
    Event triggered when an order is filled.
    
    Attributes:
        order_id: Original order ID
        symbol: Trading symbol
        side: Fill side (buy/sell)
        quantity: Filled quantity
        price: Fill price
        commission: Commission charged
        fill_time: Time of fill
        exchange: Exchange where filled
        is_partial: Whether this is a partial fill
    """
    order_id: UUID = field(default_factory=uuid4)
    symbol: str = ""
    side: str = "buy"
    quantity: float = 0.0
    price: float = 0.0
    commission: float = 0.0
    fill_time: datetime = field(default_factory=datetime.now)
    exchange: str = ""
    is_partial: bool = False
    priority: EventPriority = EventPriority.HIGH
    
    @property
    def event_type(self) -> EventType:
        return EventType.FILL
    
    @property
    def cost(self) -> float:
        """Calculate total cost including commission."""
        return (self.quantity * self.price) + self.commission
    
    @property
    def notional_value(self) -> float:
        """Calculate notional value."""
        return self.quantity * self.price
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update({
            "order_id": str(self.order_id),
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "commission": self.commission,
            "fill_time": self.fill_time.isoformat(),
            "exchange": self.exchange,
            "is_partial": self.is_partial,
            "cost": self.cost,
        })
        return base


# =============================================================================
# PORTFOLIO EVENTS
# =============================================================================

@dataclass
class PortfolioEvent(Event):
    """
    Event triggered on portfolio state changes.
    
    Attributes:
        action: Type of change (position_opened, position_closed, rebalance, etc.)
        symbol: Affected symbol (if applicable)
        equity: Current equity
        cash: Available cash
        positions_count: Number of open positions
        daily_pnl: Day's PnL
        details: Additional details
    """
    action: str = ""
    symbol: str = ""
    equity: float = 0.0
    cash: float = 0.0
    positions_count: int = 0
    daily_pnl: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    
    @property
    def event_type(self) -> EventType:
        return EventType.PORTFOLIO
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update({
            "action": self.action,
            "symbol": self.symbol,
            "equity": self.equity,
            "cash": self.cash,
            "positions_count": self.positions_count,
            "daily_pnl": self.daily_pnl,
            "details": self.details,
        })
        return base


# =============================================================================
# RISK EVENTS
# =============================================================================

@dataclass
class RiskEvent(Event):
    """
    Event triggered when risk limits are breached.
    
    Attributes:
        risk_type: Type of risk breach
        level: Severity level (warning/critical/emergency)
        current_value: Current metric value
        limit_value: Threshold that was breached
        message: Description of the risk event
        action_required: Suggested action
        affected_positions: List of affected position symbols
    """
    risk_type: str = ""  # drawdown, var, position_size, correlation, etc.
    level: str = "warning"  # warning, critical, emergency
    current_value: float = 0.0
    limit_value: float = 0.0
    message: str = ""
    action_required: str = ""
    affected_positions: list[str] = field(default_factory=list)
    priority: EventPriority = EventPriority.CRITICAL
    
    @property
    def event_type(self) -> EventType:
        return EventType.RISK
    
    @property
    def is_critical(self) -> bool:
        """Check if this is a critical risk event."""
        return self.level in ("critical", "emergency")
    
    @property
    def breach_pct(self) -> float:
        """Calculate breach percentage over limit."""
        if self.limit_value == 0:
            return 0.0
        return (self.current_value - self.limit_value) / abs(self.limit_value)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update({
            "risk_type": self.risk_type,
            "level": self.level,
            "current_value": self.current_value,
            "limit_value": self.limit_value,
            "message": self.message,
            "action_required": self.action_required,
            "affected_positions": self.affected_positions,
            "breach_pct": self.breach_pct,
        })
        return base


# =============================================================================
# SYSTEM EVENTS
# =============================================================================

@dataclass
class SystemEvent(Event):
    """
    System-level events (startup, shutdown, errors, etc.).
    
    Attributes:
        action: System action (startup, shutdown, error, etc.)
        component: Component that generated the event
        message: Event message
        details: Additional details
    """
    action: str = ""
    component: str = ""
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    
    @property
    def event_type(self) -> EventType:
        return EventType.SYSTEM
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update({
            "action": self.action,
            "component": self.component,
            "message": self.message,
            "details": self.details,
        })
        return base


# =============================================================================
# EVENT BUS
# =============================================================================

EventHandler = Callable[[Event], None]
T = TypeVar("T", bound=Event)


class EventBus:
    """
    Central event bus for publish-subscribe pattern.
    
    Components can subscribe to specific event types and
    publish events to notify other components.
    
    Example:
        bus = EventBus()
        
        def handle_signal(event: SignalEvent):
            print(f"Got signal: {event.symbol}")
        
        bus.subscribe(EventType.SIGNAL, handle_signal)
        bus.publish(SignalEvent(symbol="AAPL", direction=1))
    """
    
    def __init__(self) -> None:
        """Initialize event bus."""
        self._handlers: dict[EventType, list[EventHandler]] = {}
        self._global_handlers: list[EventHandler] = []
        self._event_history: list[Event] = []
        self._max_history: int = 10000
    
    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> None:
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Callback function to handle events
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    def subscribe_all(self, handler: EventHandler) -> None:
        """
        Subscribe to all event types.
        
        Args:
            handler: Callback function to handle all events
        """
        self._global_handlers.append(handler)
    
    def unsubscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> None:
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: Type of event to unsubscribe from
            handler: Handler to remove
        """
        if event_type in self._handlers:
            self._handlers[event_type].remove(handler)
    
    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event: Event to publish
        """
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]
        
        # Notify type-specific handlers
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            handler(event)
        
        # Notify global handlers
        for handler in self._global_handlers:
            handler(event)
    
    def get_history(
        self,
        event_type: EventType | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """
        Get event history.
        
        Args:
            event_type: Filter by event type (optional)
            limit: Maximum number of events to return
        
        Returns:
            List of historical events
        """
        if event_type is None:
            return self._event_history[-limit:]
        
        filtered = [e for e in self._event_history if e.event_type == event_type]
        return filtered[-limit:]
    
    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "EventType",
    "EventPriority",
    # Base
    "Event",
    # Event types
    "MarketEvent",
    "MultiAssetMarketEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    "PortfolioEvent",
    "RiskEvent",
    "SystemEvent",
    # Event bus
    "EventBus",
    "EventHandler",
]