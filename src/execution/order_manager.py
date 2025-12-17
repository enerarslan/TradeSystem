"""
Order management for AlphaTrade system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import pandas as pd
from loguru import logger


class OrderType(Enum):
    """Order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Trading order."""

    symbol: str
    side: str  # BUY or SELL
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    order_id: str = field(default_factory=lambda: str(uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: datetime | None = None
    filled_price: float | None = None
    filled_quantity: float = 0.0
    commission: float = 0.0
    metadata: dict = field(default_factory=dict)


class OrderManager:
    """
    Order management system.

    Handles order lifecycle:
    - Order creation and validation
    - Order submission
    - Fill tracking
    - Order history
    """

    def __init__(self) -> None:
        """Initialize order manager."""
        self._orders: dict[str, Order] = {}
        self._order_history: list[Order] = []

    def create_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        **metadata,
    ) -> Order:
        """
        Create a new order.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Order quantity
            order_type: Order type
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            **metadata: Additional metadata

        Returns:
            Created order
        """
        order = Order(
            symbol=symbol,
            side=side.upper(),
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            metadata=metadata,
        )

        self._orders[order.order_id] = order
        logger.debug(f"Created order: {order.order_id} - {side} {quantity} {symbol}")

        return order

    def submit_order(self, order_id: str) -> bool:
        """
        Submit an order.

        Args:
            order_id: Order ID

        Returns:
            True if submitted successfully
        """
        if order_id not in self._orders:
            logger.error(f"Order not found: {order_id}")
            return False

        order = self._orders[order_id]
        if order.status != OrderStatus.PENDING:
            logger.warning(f"Order {order_id} is not pending")
            return False

        order.status = OrderStatus.SUBMITTED
        logger.debug(f"Submitted order: {order_id}")

        return True

    def fill_order(
        self,
        order_id: str,
        fill_price: float,
        fill_quantity: float | None = None,
        commission: float = 0.0,
    ) -> bool:
        """
        Fill an order.

        Args:
            order_id: Order ID
            fill_price: Execution price
            fill_quantity: Filled quantity (None for full fill)
            commission: Commission charged

        Returns:
            True if filled successfully
        """
        if order_id not in self._orders:
            logger.error(f"Order not found: {order_id}")
            return False

        order = self._orders[order_id]

        if fill_quantity is None:
            fill_quantity = order.quantity - order.filled_quantity

        order.filled_quantity += fill_quantity
        order.filled_price = fill_price
        order.commission += commission
        order.filled_at = datetime.now()

        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIAL

        logger.debug(
            f"Filled order {order_id}: {fill_quantity} @ {fill_price}"
        )

        return True

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if order_id not in self._orders:
            return False

        order = self._orders[order_id]
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return False

        order.status = OrderStatus.CANCELLED
        logger.debug(f"Cancelled order: {order_id}")

        return True

    def get_order(self, order_id: str) -> Order | None:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Get all open orders."""
        open_statuses = [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]
        orders = [o for o in self._orders.values() if o.status in open_statuses]

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        return orders

    def get_filled_orders(self) -> list[Order]:
        """Get all filled orders."""
        return [o for o in self._orders.values() if o.status == OrderStatus.FILLED]

    def get_order_summary(self) -> pd.DataFrame:
        """Get summary of all orders."""
        if not self._orders:
            return pd.DataFrame()

        data = []
        for order in self._orders.values():
            data.append({
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "filled_quantity": order.filled_quantity,
                "filled_price": order.filled_price,
                "status": order.status.value,
                "commission": order.commission,
                "created_at": order.created_at,
                "filled_at": order.filled_at,
            })

        return pd.DataFrame(data)
