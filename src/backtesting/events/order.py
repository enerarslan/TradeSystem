"""
Order events for event-driven backtesting.

This module provides order event types:
- OrderEvent: New order submissions
- Order types (market, limit, stop, etc.)
- Time in force specifications
- Order modification and cancellation

Designed for institutional requirements:
- FIX protocol compliance
- Full order lifecycle tracking
- Parent/child order relationships
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from src.backtesting.events.base import Event, EventType, EventPriority


class OrderType(str, Enum):
    """Types of orders."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"
    ICEBERG = "ICEBERG"
    TWAP = "TWAP"
    VWAP = "VWAP"
    MOC = "MOC"  # Market on Close
    MOO = "MOO"  # Market on Open
    LOC = "LOC"  # Limit on Close
    LOO = "LOO"  # Limit on Open


class OrderSide(str, Enum):
    """Order side/direction."""

    BUY = "BUY"
    SELL = "SELL"
    BUY_TO_COVER = "BUY_TO_COVER"
    SELL_SHORT = "SELL_SHORT"


class TimeInForce(str, Enum):
    """Time in force specifications."""

    DAY = "DAY"
    GTC = "GTC"           # Good Till Cancelled
    IOC = "IOC"           # Immediate or Cancel
    FOK = "FOK"           # Fill or Kill
    GTD = "GTD"           # Good Till Date
    OPG = "OPG"           # At the Opening
    CLS = "CLS"           # At the Close
    GTX = "GTX"           # Good Till Crossing


class OrderStatus(str, Enum):
    """Order status values."""

    PENDING_NEW = "PENDING_NEW"
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    PENDING_CANCEL = "PENDING_CANCEL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    SUSPENDED = "SUSPENDED"


@dataclass(frozen=True)
class OrderEvent(Event):
    """
    Order submission event.

    Represents a new order being submitted to the market.
    Supports all standard order types and configurations.

    Attributes:
        symbol: Trading symbol
        side: Order side (buy/sell)
        order_type: Type of order
        quantity: Order quantity
        limit_price: Limit price (for limit orders)
        stop_price: Stop trigger price (for stop orders)
        time_in_force: Time in force specification
        strategy_name: Strategy that generated the order
        signal_id: ID of signal that triggered this order
        client_order_id: Client-assigned order ID
        parent_order_id: Parent order ID (for child orders)
        account: Trading account
        exchange: Target exchange
        algo_params: Algorithm-specific parameters
    """

    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    strategy_name: str = ""
    signal_id: Optional[str] = None
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_order_id: Optional[str] = None
    account: str = "default"
    exchange: str = ""
    algo_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, 'event_type', EventType.ORDER_NEW)

    @property
    def is_buy(self) -> bool:
        """Check if order is a buy."""
        return self.side in (OrderSide.BUY, OrderSide.BUY_TO_COVER)

    @property
    def is_sell(self) -> bool:
        """Check if order is a sell."""
        return self.side in (OrderSide.SELL, OrderSide.SELL_SHORT)

    @property
    def is_market_order(self) -> bool:
        """Check if order is market order."""
        return self.order_type == OrderType.MARKET

    @property
    def is_limit_order(self) -> bool:
        """Check if order is limit order."""
        return self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT)

    @property
    def is_stop_order(self) -> bool:
        """Check if order is stop order."""
        return self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP)

    @property
    def is_algo_order(self) -> bool:
        """Check if order is algorithmic."""
        return self.order_type in (OrderType.TWAP, OrderType.VWAP, OrderType.ICEBERG)

    @property
    def has_parent(self) -> bool:
        """Check if order has a parent."""
        return self.parent_order_id is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": str(self.timestamp),
            "created_at": self.created_at.isoformat(),
            "priority": self.priority.value,
            "source": self.source,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "strategy_name": self.strategy_name,
            "signal_id": self.signal_id,
            "client_order_id": self.client_order_id,
            "parent_order_id": self.parent_order_id,
            "account": self.account,
            "exchange": self.exchange,
            "algo_params": self.algo_params,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrderEvent":
        return cls(
            event_type=EventType.ORDER_NEW,
            timestamp=pd.Timestamp(data["timestamp"]),
            event_id=data.get("event_id", str(uuid.uuid4())),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            priority=EventPriority(data.get("priority", EventPriority.NORMAL)),
            source=data.get("source", "strategy"),
            symbol=data.get("symbol", ""),
            side=OrderSide(data.get("side", "BUY")),
            order_type=OrderType(data.get("order_type", "MARKET")),
            quantity=data.get("quantity", 0.0),
            limit_price=data.get("limit_price"),
            stop_price=data.get("stop_price"),
            time_in_force=TimeInForce(data.get("time_in_force", "DAY")),
            strategy_name=data.get("strategy_name", ""),
            signal_id=data.get("signal_id"),
            client_order_id=data.get("client_order_id", str(uuid.uuid4())),
            parent_order_id=data.get("parent_order_id"),
            account=data.get("account", "default"),
            exchange=data.get("exchange", ""),
            algo_params=data.get("algo_params", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class OrderCancelEvent(Event):
    """
    Order cancellation event.

    Attributes:
        client_order_id: Client order ID to cancel
        order_id: Exchange order ID (if known)
        symbol: Trading symbol
        reason: Cancellation reason
    """

    client_order_id: str = ""
    order_id: Optional[str] = None
    symbol: str = ""
    reason: str = ""

    def __post_init__(self):
        object.__setattr__(self, 'event_type', EventType.ORDER_CANCEL)
        object.__setattr__(self, 'priority', EventPriority.HIGH)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": str(self.timestamp),
            "created_at": self.created_at.isoformat(),
            "priority": self.priority.value,
            "source": self.source,
            "client_order_id": self.client_order_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "reason": self.reason,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrderCancelEvent":
        return cls(
            event_type=EventType.ORDER_CANCEL,
            timestamp=pd.Timestamp(data["timestamp"]),
            event_id=data.get("event_id", str(uuid.uuid4())),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            priority=EventPriority.HIGH,
            source=data.get("source", "system"),
            client_order_id=data.get("client_order_id", ""),
            order_id=data.get("order_id"),
            symbol=data.get("symbol", ""),
            reason=data.get("reason", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class OrderModifyEvent(Event):
    """
    Order modification event.

    Attributes:
        client_order_id: Client order ID to modify
        order_id: Exchange order ID (if known)
        symbol: Trading symbol
        new_quantity: New quantity (None = no change)
        new_limit_price: New limit price (None = no change)
        new_stop_price: New stop price (None = no change)
    """

    client_order_id: str = ""
    order_id: Optional[str] = None
    symbol: str = ""
    new_quantity: Optional[float] = None
    new_limit_price: Optional[float] = None
    new_stop_price: Optional[float] = None

    def __post_init__(self):
        object.__setattr__(self, 'event_type', EventType.ORDER_MODIFY)
        object.__setattr__(self, 'priority', EventPriority.HIGH)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": str(self.timestamp),
            "created_at": self.created_at.isoformat(),
            "priority": self.priority.value,
            "source": self.source,
            "client_order_id": self.client_order_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "new_quantity": self.new_quantity,
            "new_limit_price": self.new_limit_price,
            "new_stop_price": self.new_stop_price,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrderModifyEvent":
        return cls(
            event_type=EventType.ORDER_MODIFY,
            timestamp=pd.Timestamp(data["timestamp"]),
            event_id=data.get("event_id", str(uuid.uuid4())),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            priority=EventPriority.HIGH,
            source=data.get("source", "system"),
            client_order_id=data.get("client_order_id", ""),
            order_id=data.get("order_id"),
            symbol=data.get("symbol", ""),
            new_quantity=data.get("new_quantity"),
            new_limit_price=data.get("new_limit_price"),
            new_stop_price=data.get("new_stop_price"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class BracketOrder:
    """
    Bracket order (entry + stop loss + take profit).

    Attributes:
        entry_order: Main entry order
        stop_loss_order: Stop loss order
        take_profit_order: Take profit order
    """

    entry_order: OrderEvent
    stop_loss_order: Optional[OrderEvent] = None
    take_profit_order: Optional[OrderEvent] = None

    @property
    def bracket_id(self) -> str:
        """Get bracket ID (same as entry order ID)."""
        return self.entry_order.client_order_id


def create_market_order(
    symbol: str,
    side: OrderSide,
    quantity: float,
    strategy_name: str,
    timestamp: Optional[pd.Timestamp] = None,
    **kwargs,
) -> OrderEvent:
    """Create a market order."""
    return OrderEvent(
        event_type=EventType.ORDER_NEW,
        timestamp=timestamp or pd.Timestamp.now(),
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=quantity,
        strategy_name=strategy_name,
        source=strategy_name,
        **kwargs,
    )


def create_limit_order(
    symbol: str,
    side: OrderSide,
    quantity: float,
    limit_price: float,
    strategy_name: str,
    timestamp: Optional[pd.Timestamp] = None,
    time_in_force: TimeInForce = TimeInForce.DAY,
    **kwargs,
) -> OrderEvent:
    """Create a limit order."""
    return OrderEvent(
        event_type=EventType.ORDER_NEW,
        timestamp=timestamp or pd.Timestamp.now(),
        symbol=symbol,
        side=side,
        order_type=OrderType.LIMIT,
        quantity=quantity,
        limit_price=limit_price,
        time_in_force=time_in_force,
        strategy_name=strategy_name,
        source=strategy_name,
        **kwargs,
    )


def create_stop_order(
    symbol: str,
    side: OrderSide,
    quantity: float,
    stop_price: float,
    strategy_name: str,
    timestamp: Optional[pd.Timestamp] = None,
    **kwargs,
) -> OrderEvent:
    """Create a stop order."""
    return OrderEvent(
        event_type=EventType.ORDER_NEW,
        timestamp=timestamp or pd.Timestamp.now(),
        symbol=symbol,
        side=side,
        order_type=OrderType.STOP,
        quantity=quantity,
        stop_price=stop_price,
        strategy_name=strategy_name,
        source=strategy_name,
        **kwargs,
    )


def create_bracket_order(
    symbol: str,
    side: OrderSide,
    quantity: float,
    entry_price: Optional[float],
    stop_loss_price: float,
    take_profit_price: float,
    strategy_name: str,
    timestamp: Optional[pd.Timestamp] = None,
    **kwargs,
) -> BracketOrder:
    """
    Create a bracket order (entry + SL + TP).

    Args:
        symbol: Trading symbol
        side: Order side
        quantity: Order quantity
        entry_price: Entry limit price (None for market)
        stop_loss_price: Stop loss price
        take_profit_price: Take profit price
        strategy_name: Strategy name
        timestamp: Order timestamp
        **kwargs: Additional order parameters

    Returns:
        BracketOrder with entry, stop loss, and take profit
    """
    ts = timestamp or pd.Timestamp.now()
    bracket_id = str(uuid.uuid4())

    # Entry order
    if entry_price is not None:
        entry_order = create_limit_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            limit_price=entry_price,
            strategy_name=strategy_name,
            timestamp=ts,
            client_order_id=bracket_id,
            **kwargs,
        )
    else:
        entry_order = create_market_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            strategy_name=strategy_name,
            timestamp=ts,
            client_order_id=bracket_id,
            **kwargs,
        )

    # Stop loss (opposite side)
    sl_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
    stop_loss = create_stop_order(
        symbol=symbol,
        side=sl_side,
        quantity=quantity,
        stop_price=stop_loss_price,
        strategy_name=strategy_name,
        timestamp=ts,
        parent_order_id=bracket_id,
        **kwargs,
    )

    # Take profit (opposite side)
    take_profit = create_limit_order(
        symbol=symbol,
        side=sl_side,
        quantity=quantity,
        limit_price=take_profit_price,
        strategy_name=strategy_name,
        timestamp=ts,
        parent_order_id=bracket_id,
        **kwargs,
    )

    return BracketOrder(
        entry_order=entry_order,
        stop_loss_order=stop_loss,
        take_profit_order=take_profit,
    )
