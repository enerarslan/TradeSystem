"""
Fill events for event-driven backtesting.

This module provides fill/execution event types:
- FillEvent: Order fills/executions
- Partial fills
- Execution reports

Designed for institutional requirements:
- Full execution attribution
- Commission and fee tracking
- Regulatory reporting fields
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

import pandas as pd

from src.backtesting.events.base import Event, EventType, EventPriority
from src.backtesting.events.order import OrderSide, OrderType


class FillType(str, Enum):
    """Types of fills."""

    FULL = "FULL"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class LiquidityIndicator(str, Enum):
    """Liquidity provision indicator."""

    ADDED = "ADDED"      # Maker (added liquidity)
    REMOVED = "REMOVED"  # Taker (removed liquidity)
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class FillEvent(Event):
    """
    Order fill/execution event.

    Represents an order being filled (fully or partially).
    Contains all information needed for P&L calculation
    and regulatory reporting.

    Attributes:
        symbol: Trading symbol
        client_order_id: Client order ID
        order_id: Exchange order ID
        side: Order side
        fill_type: Type of fill
        fill_quantity: Quantity filled
        fill_price: Execution price
        commission: Commission charged
        commission_currency: Commission currency
        total_quantity: Total order quantity
        cumulative_quantity: Cumulative quantity filled
        leaves_quantity: Remaining quantity
        average_price: Average fill price (for partials)
        exchange: Exchange where filled
        execution_id: Unique execution ID
        liquidity: Maker/taker indicator
        strategy_name: Strategy that placed the order
        slippage: Slippage from expected price
        market_impact: Estimated market impact
        fees: Additional fees breakdown
    """

    symbol: str = ""
    client_order_id: str = ""
    order_id: str = ""
    side: OrderSide = OrderSide.BUY
    fill_type: FillType = FillType.FULL
    fill_quantity: float = 0.0
    fill_price: float = 0.0
    commission: float = 0.0
    commission_currency: str = "USD"
    total_quantity: float = 0.0
    cumulative_quantity: float = 0.0
    leaves_quantity: float = 0.0
    average_price: Optional[float] = None
    exchange: str = ""
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    liquidity: LiquidityIndicator = LiquidityIndicator.UNKNOWN
    strategy_name: str = ""
    slippage: float = 0.0
    market_impact: float = 0.0
    fees: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, 'event_type', EventType.FILL)
        object.__setattr__(self, 'priority', EventPriority.HIGH)

    @property
    def is_complete(self) -> bool:
        """Check if order is completely filled."""
        return self.fill_type == FillType.FULL or self.leaves_quantity == 0

    @property
    def fill_value(self) -> float:
        """Calculate fill value (quantity * price)."""
        return self.fill_quantity * self.fill_price

    @property
    def total_cost(self) -> float:
        """Calculate total cost including fees."""
        base_cost = self.fill_value
        fees = self.commission + sum(self.fees.values())

        if self.side in (OrderSide.BUY, OrderSide.BUY_TO_COVER):
            return base_cost + fees
        else:
            return base_cost - fees

    @property
    def effective_price(self) -> float:
        """Calculate effective price including all costs."""
        if self.fill_quantity == 0:
            return 0.0

        total_fees = self.commission + sum(self.fees.values())

        if self.side in (OrderSide.BUY, OrderSide.BUY_TO_COVER):
            return (self.fill_value + total_fees) / self.fill_quantity
        else:
            return (self.fill_value - total_fees) / self.fill_quantity

    @property
    def fill_pct(self) -> float:
        """Percentage of order filled."""
        if self.total_quantity == 0:
            return 0.0
        return self.cumulative_quantity / self.total_quantity

    @property
    def is_partial(self) -> bool:
        """Check if this is a partial fill."""
        return self.fill_type == FillType.PARTIAL or self.leaves_quantity > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": str(self.timestamp),
            "created_at": self.created_at.isoformat(),
            "priority": self.priority.value,
            "source": self.source,
            "symbol": self.symbol,
            "client_order_id": self.client_order_id,
            "order_id": self.order_id,
            "side": self.side.value,
            "fill_type": self.fill_type.value,
            "fill_quantity": self.fill_quantity,
            "fill_price": self.fill_price,
            "commission": self.commission,
            "commission_currency": self.commission_currency,
            "total_quantity": self.total_quantity,
            "cumulative_quantity": self.cumulative_quantity,
            "leaves_quantity": self.leaves_quantity,
            "average_price": self.average_price,
            "exchange": self.exchange,
            "execution_id": self.execution_id,
            "liquidity": self.liquidity.value,
            "strategy_name": self.strategy_name,
            "slippage": self.slippage,
            "market_impact": self.market_impact,
            "fees": self.fees,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FillEvent":
        return cls(
            event_type=EventType.FILL,
            timestamp=pd.Timestamp(data["timestamp"]),
            event_id=data.get("event_id", str(uuid.uuid4())),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            priority=EventPriority.HIGH,
            source=data.get("source", "exchange"),
            symbol=data.get("symbol", ""),
            client_order_id=data.get("client_order_id", ""),
            order_id=data.get("order_id", ""),
            side=OrderSide(data.get("side", "BUY")),
            fill_type=FillType(data.get("fill_type", "FULL")),
            fill_quantity=data.get("fill_quantity", 0.0),
            fill_price=data.get("fill_price", 0.0),
            commission=data.get("commission", 0.0),
            commission_currency=data.get("commission_currency", "USD"),
            total_quantity=data.get("total_quantity", 0.0),
            cumulative_quantity=data.get("cumulative_quantity", 0.0),
            leaves_quantity=data.get("leaves_quantity", 0.0),
            average_price=data.get("average_price"),
            exchange=data.get("exchange", ""),
            execution_id=data.get("execution_id", str(uuid.uuid4())),
            liquidity=LiquidityIndicator(data.get("liquidity", "UNKNOWN")),
            strategy_name=data.get("strategy_name", ""),
            slippage=data.get("slippage", 0.0),
            market_impact=data.get("market_impact", 0.0),
            fees=data.get("fees", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class RejectionEvent(Event):
    """
    Order rejection event.

    Attributes:
        symbol: Trading symbol
        client_order_id: Client order ID
        reason: Rejection reason
        error_code: Error code (if applicable)
    """

    symbol: str = ""
    client_order_id: str = ""
    reason: str = ""
    error_code: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, 'event_type', EventType.FILL)
        object.__setattr__(self, 'priority', EventPriority.HIGH)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": str(self.timestamp),
            "created_at": self.created_at.isoformat(),
            "priority": self.priority.value,
            "source": self.source,
            "symbol": self.symbol,
            "client_order_id": self.client_order_id,
            "reason": self.reason,
            "error_code": self.error_code,
            "fill_type": FillType.REJECTED.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RejectionEvent":
        return cls(
            event_type=EventType.FILL,
            timestamp=pd.Timestamp(data["timestamp"]),
            event_id=data.get("event_id", str(uuid.uuid4())),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            priority=EventPriority.HIGH,
            source=data.get("source", "exchange"),
            symbol=data.get("symbol", ""),
            client_order_id=data.get("client_order_id", ""),
            reason=data.get("reason", ""),
            error_code=data.get("error_code"),
            metadata=data.get("metadata", {}),
        )


def create_fill(
    order_event: Any,  # OrderEvent
    fill_price: float,
    fill_quantity: Optional[float] = None,
    commission: float = 0.0,
    slippage: float = 0.0,
    timestamp: Optional[pd.Timestamp] = None,
    **kwargs,
) -> FillEvent:
    """
    Create a fill event from an order event.

    Args:
        order_event: Original order event
        fill_price: Execution price
        fill_quantity: Quantity filled (default: full order)
        commission: Commission charged
        slippage: Slippage from expected price
        timestamp: Fill timestamp (default: now)
        **kwargs: Additional fill attributes

    Returns:
        FillEvent instance
    """
    qty = fill_quantity if fill_quantity is not None else order_event.quantity
    leaves = order_event.quantity - qty
    fill_type = FillType.FULL if leaves == 0 else FillType.PARTIAL

    return FillEvent(
        event_type=EventType.FILL,
        timestamp=timestamp or pd.Timestamp.now(),
        symbol=order_event.symbol,
        client_order_id=order_event.client_order_id,
        order_id=kwargs.get("order_id", ""),
        side=order_event.side,
        fill_type=fill_type,
        fill_quantity=qty,
        fill_price=fill_price,
        commission=commission,
        total_quantity=order_event.quantity,
        cumulative_quantity=qty,
        leaves_quantity=leaves,
        strategy_name=order_event.strategy_name,
        slippage=slippage,
        source="execution",
        **kwargs,
    )


def create_rejection(
    order_event: Any,  # OrderEvent
    reason: str,
    error_code: Optional[str] = None,
    timestamp: Optional[pd.Timestamp] = None,
) -> RejectionEvent:
    """
    Create a rejection event for an order.

    Args:
        order_event: Original order event
        reason: Rejection reason
        error_code: Error code
        timestamp: Rejection timestamp

    Returns:
        RejectionEvent instance
    """
    return RejectionEvent(
        event_type=EventType.FILL,
        timestamp=timestamp or pd.Timestamp.now(),
        symbol=order_event.symbol,
        client_order_id=order_event.client_order_id,
        reason=reason,
        error_code=error_code,
        source="execution",
    )
