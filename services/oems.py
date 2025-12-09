"""
Order Execution Management System (OEMS) Service
================================================

Handles order lifecycle management and broker connectivity.
Receives approved signals and executes them via broker API.

Responsibilities:
- Order creation and submission
- Order lifecycle tracking
- Fill processing
- Position reconciliation
- Emergency order cancellation

Author: AlphaTrade Platform
Version: 3.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from enum import Enum
import os

from config.settings import get_logger
from infrastructure.message_bus import (
    Message,
    MessageType,
    Channel,
    MessagePriority,
)
from infrastructure.state_store import OrderState
from infrastructure.service_registry import ServiceType
from services.base_service import BaseService, ServiceConfig

logger = get_logger(__name__)

# Try to import Alpaca
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
    from alpaca.trading.stream import TradingStream
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not installed, using paper trading simulation")


class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class OEMSConfig(ServiceConfig):
    """Configuration for OEMS service."""
    name: str = "oems"
    service_type: ServiceType = ServiceType.OEMS
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    paper_trading: bool = True
    default_time_in_force: str = "day"
    max_order_retries: int = 3
    order_timeout: float = 30.0  # seconds
    enable_smart_routing: bool = True


class OEMSService(BaseService):
    """
    Order Execution Management System Service.

    Handles the full order lifecycle from signal receipt to fill.
    Connects directly to broker API for order execution.

    Example:
        config = OEMSConfig(
            alpaca_api_key="your_key",
            alpaca_secret_key="your_secret",
            paper_trading=True
        )

        service = OEMSService(config)
        await service.run_forever()
    """

    def __init__(self, config: OEMSConfig | None = None):
        """Initialize OEMS service."""
        config = config or OEMSConfig()

        # Get API keys from environment
        if not config.alpaca_api_key:
            config.alpaca_api_key = os.getenv("ALPACA_API_KEY", "")
        if not config.alpaca_secret_key:
            config.alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY", "")

        super().__init__(config)

        self.config: OEMSConfig = config

        # Broker client
        self._trading_client: TradingClient | None = None
        self._trading_stream: TradingStream | None = None

        # Order tracking
        self._pending_orders: dict[str, OrderState] = {}
        self._order_history: list[OrderState] = []

        # Statistics
        self._orders_submitted = 0
        self._orders_filled = 0
        self._orders_cancelled = 0
        self._orders_rejected = 0

        # Paper trading simulation
        self._simulated_positions: dict[str, float] = {}
        self._simulated_cash: float = 100000.0

    async def _on_start(self) -> None:
        """Start OEMS service."""
        logger.info("Starting OEMS service")

        # Connect to broker
        await self._connect_broker()

        # Load pending orders from Redis
        await self._load_pending_orders()

        # Subscribe to approved signals
        await self.subscribe(Channel.ORDERS, self._handle_order_message)

        # Start order monitoring
        self.add_background_task(self._order_monitoring_loop())

        logger.info(
            f"OEMS started. Paper trading: {self.config.paper_trading}, "
            f"Broker connected: {self._trading_client is not None}"
        )

    async def _on_stop(self) -> None:
        """Stop OEMS service."""
        # Cancel all pending orders on graceful shutdown
        await self._cancel_all_orders(reason="Service shutdown")

        logger.info(
            f"OEMS stopped. "
            f"Orders: {self._orders_submitted} submitted, "
            f"{self._orders_filled} filled, "
            f"{self._orders_cancelled} cancelled"
        )

    async def _connect_broker(self) -> None:
        """Connect to broker API."""
        if not ALPACA_AVAILABLE:
            logger.warning("Alpaca not available, using simulation")
            return

        if not self.config.alpaca_api_key:
            logger.warning("No API key, using simulation")
            return

        try:
            self._trading_client = TradingClient(
                api_key=self.config.alpaca_api_key,
                secret_key=self.config.alpaca_secret_key,
                paper=self.config.paper_trading,
            )

            # Verify connection
            account = self._trading_client.get_account()
            logger.info(
                f"Connected to Alpaca. "
                f"Equity: ${float(account.equity):,.2f}, "
                f"Cash: ${float(account.cash):,.2f}"
            )

            # Start trade stream for fill updates
            self._trading_stream = TradingStream(
                api_key=self.config.alpaca_api_key,
                secret_key=self.config.alpaca_secret_key,
                paper=self.config.paper_trading,
            )

            self._trading_stream.subscribe_trade_updates(self._handle_trade_update)
            self.add_background_task(self._run_trade_stream())

        except Exception as e:
            logger.error(f"Failed to connect to broker: {e}")
            self._trading_client = None

    async def _run_trade_stream(self) -> None:
        """Run the trade update stream."""
        if not self._trading_stream:
            return

        try:
            await self._trading_stream._run_forever()
        except Exception as e:
            logger.error(f"Trade stream error: {e}")

    async def _load_pending_orders(self) -> None:
        """Load pending orders from Redis."""
        if not self._state_store:
            return

        try:
            orders = await self._state_store.get_pending_orders()
            for order in orders:
                self._pending_orders[order.order_id] = order

            logger.info(f"Loaded {len(self._pending_orders)} pending orders")

        except Exception as e:
            logger.error(f"Failed to load pending orders: {e}")

    async def _handle_order_message(self, message: Message) -> None:
        """Handle incoming order-related messages."""
        try:
            if message.type == MessageType.SIGNAL_APPROVED:
                await self._execute_signal(message)
            elif message.type == MessageType.ORDER_CANCEL:
                await self._cancel_order(message.payload.get("order_id"))

        except Exception as e:
            logger.error(f"Error handling order message: {e}")

    async def _execute_signal(self, message: Message) -> None:
        """Execute an approved trading signal."""
        payload = message.payload

        symbol = payload.get("symbol")
        direction = payload.get("direction")
        quantity = payload.get("quantity", 0)
        price = payload.get("price", 0)

        if quantity <= 0:
            logger.warning(f"Invalid quantity for {symbol}: {quantity}")
            return

        # Determine order side
        side = "buy" if direction > 0 else "sell"

        # Create order
        order_id = str(uuid.uuid4())[:8]

        order = OrderState(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type="market",
            status="pending",
            created_at=time.time(),
        )

        self._pending_orders[order_id] = order

        # Save to Redis
        if self._state_store:
            await self._state_store.set_order(order)

        # Submit order
        await self._submit_order(order)

    async def _submit_order(self, order: OrderState) -> bool:
        """Submit order to broker."""
        self._orders_submitted += 1

        try:
            if self._trading_client:
                # Real broker order
                return await self._submit_alpaca_order(order)
            else:
                # Simulated order
                return await self._simulate_order(order)

        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            await self._handle_order_rejected(order, str(e))
            return False

    async def _submit_alpaca_order(self, order: OrderState) -> bool:
        """Submit order to Alpaca."""
        try:
            side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL

            if order.order_type == "market":
                request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
            else:
                request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=order.limit_price,
                )

            alpaca_order = self._trading_client.submit_order(request)

            # Update order with broker ID
            order.status = "submitted"
            order.updated_at = time.time()

            # Store broker order ID in order state
            # Note: In production, you'd store this mapping

            logger.info(
                f"Order submitted to Alpaca: {order.symbol} {order.side} "
                f"{order.quantity} @ market"
            )

            # Publish order event
            await self._publish_order_event(order, MessageType.ORDER_NEW)

            return True

        except Exception as e:
            logger.error(f"Alpaca order failed: {e}")
            return False

    async def _simulate_order(self, order: OrderState) -> bool:
        """Simulate order execution for paper trading."""
        try:
            # Immediate fill simulation
            await asyncio.sleep(0.1)  # Simulate latency

            # Get simulated price (in real system, use last price)
            fill_price = order.limit_price or 150.0  # Mock price

            # Update order
            order.status = "filled"
            order.filled_qty = order.quantity
            order.avg_fill_price = fill_price
            order.updated_at = time.time()

            # Update simulated position
            if order.side == "buy":
                self._simulated_positions[order.symbol] = (
                    self._simulated_positions.get(order.symbol, 0) + order.quantity
                )
                self._simulated_cash -= order.quantity * fill_price
            else:
                self._simulated_positions[order.symbol] = (
                    self._simulated_positions.get(order.symbol, 0) - order.quantity
                )
                self._simulated_cash += order.quantity * fill_price

            logger.info(
                f"Order filled (simulated): {order.symbol} {order.side} "
                f"{order.quantity} @ ${fill_price:.2f}"
            )

            # Publish fill event
            await self._handle_order_filled(order, fill_price)

            return True

        except Exception as e:
            logger.error(f"Order simulation failed: {e}")
            return False

    async def _handle_trade_update(self, data) -> None:
        """Handle trade update from Alpaca stream."""
        try:
            event = data.event
            order = data.order

            logger.debug(f"Trade update: {event} for {order.symbol}")

            if event == "fill":
                # Find our order
                for oid, our_order in self._pending_orders.items():
                    if our_order.symbol == order.symbol:
                        await self._handle_order_filled(
                            our_order,
                            float(order.filled_avg_price),
                        )
                        break

            elif event in ("canceled", "cancelled"):
                for oid, our_order in list(self._pending_orders.items()):
                    if our_order.symbol == order.symbol:
                        await self._handle_order_cancelled(our_order)
                        break

            elif event == "rejected":
                for oid, our_order in list(self._pending_orders.items()):
                    if our_order.symbol == order.symbol:
                        await self._handle_order_rejected(our_order, "Rejected by broker")
                        break

        except Exception as e:
            logger.error(f"Error handling trade update: {e}")

    async def _handle_order_filled(self, order: OrderState, fill_price: float) -> None:
        """Handle order fill."""
        self._orders_filled += 1

        order.status = "filled"
        order.filled_qty = order.quantity
        order.avg_fill_price = fill_price
        order.updated_at = time.time()

        # Calculate PnL (simplified)
        pnl = 0.0  # Would calculate based on entry price

        # Remove from pending
        self._pending_orders.pop(order.order_id, None)
        self._order_history.append(order)

        # Update Redis
        if self._state_store:
            await self._state_store.set_order(order)

        # Publish fill event
        message = Message(
            type=MessageType.ORDER_FILLED,
            channel=Channel.FILLS,
            payload={
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.filled_qty,
                "price": fill_price,
                "pnl": pnl,
                "timestamp": datetime.now().isoformat(),
            },
            priority=MessagePriority.HIGH,
            source=self.name,
        )
        await self.publish(message)

        logger.info(
            f"Order FILLED: {order.symbol} {order.side} "
            f"{order.filled_qty:.2f} @ ${fill_price:.2f}"
        )

    async def _handle_order_cancelled(self, order: OrderState) -> None:
        """Handle order cancellation."""
        self._orders_cancelled += 1

        order.status = "cancelled"
        order.updated_at = time.time()

        self._pending_orders.pop(order.order_id, None)
        self._order_history.append(order)

        if self._state_store:
            await self._state_store.set_order(order)

        await self._publish_order_event(order, MessageType.ORDER_CANCELLED)

        logger.info(f"Order CANCELLED: {order.symbol} {order.order_id}")

    async def _handle_order_rejected(self, order: OrderState, reason: str) -> None:
        """Handle order rejection."""
        self._orders_rejected += 1

        order.status = "rejected"
        order.updated_at = time.time()

        self._pending_orders.pop(order.order_id, None)
        self._order_history.append(order)

        if self._state_store:
            await self._state_store.set_order(order)

        message = Message(
            type=MessageType.ORDER_REJECTED,
            channel=Channel.ORDERS,
            payload={
                "order_id": order.order_id,
                "symbol": order.symbol,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
            },
            priority=MessagePriority.HIGH,
            source=self.name,
        )
        await self.publish(message)

        logger.warning(f"Order REJECTED: {order.symbol} - {reason}")

    async def _publish_order_event(self, order: OrderState, msg_type: MessageType) -> None:
        """Publish order event message."""
        message = Message(
            type=msg_type,
            channel=Channel.ORDERS,
            payload=order.to_dict(),
            priority=MessagePriority.HIGH,
            source=self.name,
        )
        await self.publish(message)

    async def _cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order."""
        order = self._pending_orders.get(order_id)
        if not order:
            logger.warning(f"Order not found: {order_id}")
            return False

        try:
            if self._trading_client:
                # Cancel via broker
                # self._trading_client.cancel_order(broker_order_id)
                pass

            await self._handle_order_cancelled(order)
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False

    async def _cancel_all_orders(self, reason: str = "User request") -> int:
        """Cancel all pending orders."""
        cancelled = 0

        for order_id in list(self._pending_orders.keys()):
            if await self._cancel_order(order_id):
                cancelled += 1

        logger.info(f"Cancelled {cancelled} orders: {reason}")
        return cancelled

    async def _close_all_positions(self) -> None:
        """Close all positions - emergency procedure."""
        logger.critical("Closing all positions")

        if self._trading_client:
            try:
                self._trading_client.close_all_positions(cancel_orders=True)
                logger.critical("All positions closed via broker")
            except Exception as e:
                logger.error(f"Failed to close positions: {e}")
        else:
            # Simulation
            for symbol, qty in list(self._simulated_positions.items()):
                if qty != 0:
                    side = "sell" if qty > 0 else "buy"
                    order = OrderState(
                        order_id=str(uuid.uuid4())[:8],
                        symbol=symbol,
                        side=side,
                        quantity=abs(qty),
                        order_type="market",
                    )
                    await self._simulate_order(order)

    async def _order_monitoring_loop(self) -> None:
        """Monitor pending orders for timeouts."""
        while self._running:
            try:
                now = time.time()

                for order_id, order in list(self._pending_orders.items()):
                    # Check for timeout
                    if now - order.created_at > self.config.order_timeout:
                        logger.warning(f"Order timeout: {order_id}")
                        await self._cancel_order(order_id)

                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Order monitoring error: {e}")
                await asyncio.sleep(1)

    async def _handle_kill_switch(self, message: Message) -> None:
        """Handle kill switch - OEMS specific."""
        logger.critical("OEMS received kill switch")

        # Cancel all pending orders
        await self._cancel_all_orders(reason="Kill switch")

        # Close all positions
        await self._close_all_positions()

        # Parent handler will stop the service
        await super()._handle_kill_switch(message)

    def get_status(self) -> dict[str, Any]:
        """Get service status."""
        status = super().get_status()
        status.update({
            "broker_connected": self._trading_client is not None,
            "paper_trading": self.config.paper_trading,
            "pending_orders": len(self._pending_orders),
            "orders_submitted": self._orders_submitted,
            "orders_filled": self._orders_filled,
            "orders_cancelled": self._orders_cancelled,
            "orders_rejected": self._orders_rejected,
            "simulated_cash": self._simulated_cash if not self._trading_client else None,
            "simulated_positions": self._simulated_positions if not self._trading_client else None,
        })
        return status


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main():
    """Run OEMS service."""
    config = OEMSConfig(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        paper_trading=os.getenv("PAPER_TRADING", "true").lower() == "true",
    )

    service = OEMSService(config)
    await service.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
