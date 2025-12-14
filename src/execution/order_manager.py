"""
Order Management System
JPMorgan-Level Order Lifecycle Management

Features:
- Order state machine
- Order validation
- Smart order routing
- Order queuing
- Partial fill handling
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import threading
import uuid
import heapq

from .broker_api import (
    BrokerAPI, OrderRequest, OrderResponse,
    OrderSide, OrderType, TimeInForce
)
from .protected_positions import ProtectedPositionManager, ProtectionConfig
from ..risk.risk_manager import RiskManager, PreTradeRiskCheck
from ..utils.logger import get_logger, get_audit_logger

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class OrderStatus(Enum):
    """Order status states"""
    CREATED = "created"
    VALIDATED = "validated"
    PENDING_RISK = "pending_risk"
    RISK_APPROVED = "risk_approved"
    RISK_REJECTED = "risk_rejected"
    QUEUED = "queued"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


class OrderPriority(Enum):
    """Order priority levels"""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    URGENT = 0


@dataclass
class Order:
    """
    Internal order representation with full lifecycle tracking.
    """
    # Identifiers
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_order_id: str = ""
    broker_order_id: str = ""
    parent_order_id: Optional[str] = None

    # Order details
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: int = 0
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY

    # Execution details
    filled_quantity: int = 0
    remaining_quantity: int = 0
    avg_fill_price: float = 0
    fills: List[Dict] = field(default_factory=list)

    # Status
    status: OrderStatus = OrderStatus.CREATED
    priority: OrderPriority = OrderPriority.NORMAL

    # Risk
    risk_check: Optional[PreTradeRiskCheck] = None

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    # Strategy info
    strategy_name: str = ""
    signal_strength: float = 0
    signal_price: float = 0

    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.client_order_id:
            self.client_order_id = self.order_id[:12]
        self.remaining_quantity = self.quantity

    @property
    def is_active(self) -> bool:
        return self.status in [
            OrderStatus.CREATED, OrderStatus.VALIDATED,
            OrderStatus.QUEUED, OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED
        ]

    @property
    def is_complete(self) -> bool:
        return self.status in [
            OrderStatus.FILLED, OrderStatus.CANCELLED,
            OrderStatus.REJECTED, OrderStatus.EXPIRED,
            OrderStatus.FAILED
        ]

    @property
    def fill_pct(self) -> float:
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0

    @property
    def notional(self) -> float:
        price = self.avg_fill_price or self.limit_price or self.signal_price
        return self.quantity * price if price else 0

    def add_fill(
        self,
        quantity: int,
        price: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record a fill"""
        self.fills.append({
            'quantity': quantity,
            'price': price,
            'timestamp': timestamp or datetime.now()
        })

        # Update running totals
        total_qty = sum(f['quantity'] for f in self.fills)
        total_value = sum(f['quantity'] * f['price'] for f in self.fills)

        self.filled_quantity = total_qty
        self.remaining_quantity = self.quantity - total_qty
        self.avg_fill_price = total_value / total_qty if total_qty > 0 else 0

        # Update status
        if self.remaining_quantity <= 0:
            self.status = OrderStatus.FILLED
            self.filled_at = datetime.now()
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

        self.updated_at = datetime.now()

    def to_request(self) -> OrderRequest:
        """Convert to broker OrderRequest"""
        return OrderRequest(
            symbol=self.symbol,
            side=self.side,
            quantity=self.remaining_quantity,
            order_type=self.order_type,
            limit_price=self.limit_price,
            stop_price=self.stop_price,
            time_in_force=self.time_in_force,
            client_order_id=self.client_order_id
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'created_at': self.created_at.isoformat(),
            'strategy': self.strategy_name
        }


class OrderQueue:
    """
    Priority queue for order management.

    Orders are processed by priority and submission time.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._queue: List[tuple] = []  # heapq
        self._orders: Dict[str, Order] = {}
        self._lock = threading.Lock()

    def push(self, order: Order) -> bool:
        """Add order to queue"""
        with self._lock:
            if len(self._queue) >= self.max_size:
                return False

            # Priority tuple: (priority, timestamp, order_id)
            priority_tuple = (
                order.priority.value,
                order.created_at.timestamp(),
                order.order_id
            )

            heapq.heappush(self._queue, priority_tuple)
            self._orders[order.order_id] = order
            order.status = OrderStatus.QUEUED
            return True

    def pop(self) -> Optional[Order]:
        """Get highest priority order"""
        with self._lock:
            while self._queue:
                _, _, order_id = heapq.heappop(self._queue)
                if order_id in self._orders:
                    order = self._orders.pop(order_id)
                    return order
            return None

    def peek(self) -> Optional[Order]:
        """View highest priority order without removing"""
        with self._lock:
            while self._queue:
                _, _, order_id = self._queue[0]
                if order_id in self._orders:
                    return self._orders[order_id]
                heapq.heappop(self._queue)
            return None

    def remove(self, order_id: str) -> Optional[Order]:
        """Remove specific order"""
        with self._lock:
            if order_id in self._orders:
                return self._orders.pop(order_id)
            return None

    def get(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self._orders.get(order_id)

    @property
    def size(self) -> int:
        return len(self._orders)

    def clear(self) -> int:
        """Clear queue, return count"""
        with self._lock:
            count = len(self._orders)
            self._queue.clear()
            self._orders.clear()
            return count


class SmartOrderRouter:
    """
    Smart Order Router for optimal execution.

    Features:
    - Venue selection
    - Order splitting
    - Cost optimization
    """

    def __init__(
        self,
        brokers: Dict[str, BrokerAPI],
        default_broker: str = "alpaca"
    ):
        self.brokers = brokers
        self.default_broker = default_broker

        # Routing rules
        self._routing_rules: List[Callable[[Order], Optional[str]]] = []

    def add_routing_rule(self, rule: Callable[[Order], Optional[str]]) -> None:
        """Add routing rule"""
        self._routing_rules.append(rule)

    def route(self, order: Order) -> str:
        """
        Determine best broker for order.

        Returns broker name.
        """
        # Apply routing rules
        for rule in self._routing_rules:
            broker = rule(order)
            if broker and broker in self.brokers:
                return broker

        return self.default_broker

    def get_broker(self, broker_name: str) -> Optional[BrokerAPI]:
        """Get broker by name"""
        return self.brokers.get(broker_name)

    def split_order(
        self,
        order: Order,
        max_slice_size: int = 1000,
        max_slices: int = 10
    ) -> List[Order]:
        """
        Split large order into smaller slices.

        Returns list of child orders.
        """
        if order.quantity <= max_slice_size:
            return [order]

        slices = []
        remaining = order.quantity
        slice_num = 0

        while remaining > 0 and slice_num < max_slices:
            slice_qty = min(remaining, max_slice_size)

            child = Order(
                parent_order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=slice_qty,
                order_type=order.order_type,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                time_in_force=order.time_in_force,
                strategy_name=order.strategy_name,
                priority=order.priority
            )
            child.tags['slice'] = str(slice_num)
            slices.append(child)

            remaining -= slice_qty
            slice_num += 1

        return slices


class OrderManager:
    """
    Central order management system.

    Handles:
    - Order creation and validation
    - Risk checks
    - Order routing
    - Lifecycle management
    - Event handling
    - Protected position management (bracket orders with SL/TP)
    """

    def __init__(
        self,
        broker: BrokerAPI,
        risk_manager: Optional[RiskManager] = None,
        max_pending_orders: int = 100,
        protected_manager: Optional[ProtectedPositionManager] = None
    ):
        self.broker = broker
        self.risk_manager = risk_manager
        self.max_pending_orders = max_pending_orders
        self._protected_manager = protected_manager

        # Order tracking with thread-safe lock
        # CRITICAL FIX: Added threading.Lock to prevent race conditions
        # when broker callbacks (potentially from different thread) modify
        # order dictionaries while main thread is iterating
        self._orders: Dict[str, Order] = {}
        self._active_orders: Dict[str, Order] = {}
        self._orders_lock = threading.Lock()  # Thread safety for order dicts
        self._order_queue = OrderQueue()

        # Order routing
        self._router: Optional[SmartOrderRouter] = None

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            'order_created': [],
            'order_submitted': [],
            'order_filled': [],
            'order_cancelled': [],
            'order_rejected': [],
            'order_updated': []
        }

        # Processing
        self._processing = False
        self._process_task: Optional[asyncio.Task] = None

        # Register broker callbacks
        self.broker.register_callback('trade_update', self._on_broker_update)

    def set_router(self, router: SmartOrderRouter) -> None:
        """Set smart order router"""
        self._router = router

    def register_callback(
        self,
        event_type: str,
        callback: Callable
    ) -> None:
        """Register callback"""
        if event_type in self._callbacks:
            self._callbacks[event_type].append(callback)

    def _emit(self, event_type: str, order: Order) -> None:
        """Emit order event"""
        for callback in self._callbacks.get(event_type, []):
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    async def create_order(
        self,
        symbol: str,
        side: Union[str, OrderSide],
        quantity: int,
        order_type: Union[str, OrderType] = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: Union[str, TimeInForce] = TimeInForce.DAY,
        strategy_name: str = "",
        signal_strength: float = 0,
        signal_price: float = 0,
        priority: OrderPriority = OrderPriority.NORMAL,
        tags: Optional[Dict] = None
    ) -> Order:
        """
        Create new order.

        Returns Order object.
        """
        # Normalize enums
        if isinstance(side, str):
            side = OrderSide(side.lower())
        if isinstance(order_type, str):
            order_type = OrderType(order_type.lower())
        if isinstance(time_in_force, str):
            time_in_force = TimeInForce(time_in_force.lower())

        # Create order
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            strategy_name=strategy_name,
            signal_strength=signal_strength,
            signal_price=signal_price,
            priority=priority,
            tags=tags or {}
        )

        # Store order
        self._orders[order.order_id] = order

        logger.info(f"Order created: {order.order_id} - {side.value} {quantity} {symbol}")

        self._emit('order_created', order)

        return order

    async def validate_order(self, order: Order) -> bool:
        """Validate order parameters"""
        errors = []

        # Basic validation
        if order.quantity <= 0:
            errors.append("Quantity must be positive")

        if order.order_type == OrderType.LIMIT and not order.limit_price:
            errors.append("Limit price required for limit orders")

        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and not order.stop_price:
            errors.append("Stop price required for stop orders")

        if errors:
            order.notes.extend(errors)
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order validation failed: {errors}")
            return False

        order.status = OrderStatus.VALIDATED
        return True

    async def check_risk(self, order: Order) -> bool:
        """Run pre-trade risk checks"""
        if not self.risk_manager:
            order.status = OrderStatus.RISK_APPROVED
            return True

        order.status = OrderStatus.PENDING_RISK

        price = order.limit_price or order.signal_price or 0
        check = self.risk_manager.pre_trade_check(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            price=price
        )

        order.risk_check = check

        if check.passed:
            order.status = OrderStatus.RISK_APPROVED

            # Apply adjustments
            if check.adjusted_quantity:
                order.quantity = check.adjusted_quantity
                order.remaining_quantity = check.adjusted_quantity
                order.notes.append(f"Quantity adjusted: {check.adjustment_reason}")

            return True
        else:
            order.status = OrderStatus.RISK_REJECTED
            order.notes.extend(check.checks_failed)
            logger.warning(f"Risk check failed: {check.checks_failed}")
            return False

    async def submit_order(self, order: Order) -> bool:
        """Submit order to broker"""
        try:
            # Validate
            if not await self.validate_order(order):
                self._emit('order_rejected', order)
                return False

            # Risk check
            if not await self.check_risk(order):
                self._emit('order_rejected', order)
                return False

            # Route order
            broker = self.broker
            if self._router:
                broker_name = self._router.route(order)
                broker = self._router.get_broker(broker_name) or self.broker

            # Submit to broker
            request = order.to_request()
            response = await broker.submit_order(request)

            # Update order
            order.broker_order_id = response.order_id
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.now()

            # Track active order
            self._active_orders[order.order_id] = order

            logger.info(
                f"Order submitted: {order.order_id} -> broker: {response.order_id}"
            )

            audit_logger.log_order(
                order_id=order.order_id,
                symbol=order.symbol,
                action="SUBMITTED",
                details=order.to_dict()
            )

            self._emit('order_submitted', order)
            return True

        except Exception as e:
            order.status = OrderStatus.FAILED
            order.notes.append(f"Submission failed: {str(e)}")
            logger.error(f"Order submission failed: {e}")
            self._emit('order_rejected', order)
            return False

    async def submit_order_with_protection(
        self,
        order: Order,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04
    ) -> Tuple[Order, bool]:
        """
        Submit order with bracket protection (server-side SL/TP).

        Delegates to ProtectedPositionManager for proper SL/TP handling.
        This ensures every position has broker-side protection via bracket orders.

        Args:
            order: Order to submit
            stop_loss_pct: Stop loss percentage from entry (default 2%)
            take_profit_pct: Take profit percentage from entry (default 4%)

        Returns:
            Tuple of (Order, success_bool)
        """
        if self._protected_manager is None:
            # Fallback to basic submission without protection
            logger.warning("No protected manager available, submitting without bracket orders")
            success = await self.submit_order(order)
            return order, success

        # Use protected manager for bracket orders
        position, success = await self._protected_manager.open_position_with_protection(
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct
        )

        if success:
            order.status = OrderStatus.FILLED
            order.broker_order_id = position.position_id
            self._orders[order.order_id] = order
            logger.info(
                f"Protected order submitted: {order.symbol} "
                f"SL: {stop_loss_pct:.1%} TP: {take_profit_pct:.1%}"
            )
        else:
            order.status = OrderStatus.FAILED
            order.notes.append("Protected position failed")

        return order, success

    def set_protected_manager(self, manager: ProtectedPositionManager) -> None:
        """Set the protected position manager for bracket order support."""
        self._protected_manager = manager
        logger.info("Protected position manager attached to OrderManager")

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        order = self._orders.get(order_id)
        if not order:
            return False

        if not order.is_active:
            return False

        try:
            if order.broker_order_id:
                success = await self.broker.cancel_order(order.broker_order_id)
                if success:
                    order.status = OrderStatus.CANCELLED
                    order.cancelled_at = datetime.now()

                    if order_id in self._active_orders:
                        del self._active_orders[order_id]

                    audit_logger.log_order(
                        order_id=order_id,
                        symbol=order.symbol,
                        action="CANCELLED",
                        details=order.to_dict()
                    )

                    self._emit('order_cancelled', order)
                    return True

            # If not submitted yet, just cancel internally
            elif order.status in [OrderStatus.CREATED, OrderStatus.VALIDATED, OrderStatus.QUEUED]:
                order.status = OrderStatus.CANCELLED
                order.cancelled_at = datetime.now()
                self._order_queue.remove(order_id)
                self._emit('order_cancelled', order)
                return True

            return False

        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False

    async def cancel_all(self) -> int:
        """Cancel all active orders.

        CRITICAL FIX: Uses thread-safe snapshot to prevent dictionary
        mutation during iteration when broker callbacks fire.
        """
        count = 0

        # Take thread-safe snapshot of order IDs to cancel
        with self._orders_lock:
            order_ids_to_cancel = list(self._active_orders.keys())

        # Cancel orders outside the lock to avoid deadlock
        for order_id in order_ids_to_cancel:
            if await self.cancel_order(order_id):
                count += 1

        # Also cancel queued orders
        count += self._order_queue.clear()

        return count

    async def queue_order(self, order: Order) -> bool:
        """Add order to processing queue"""
        if len(self._active_orders) >= self.max_pending_orders:
            logger.warning("Max pending orders reached")
            return False

        return self._order_queue.push(order)

    async def start_processing(self) -> None:
        """Start order queue processing"""
        if self._processing:
            return

        self._processing = True
        self._process_task = asyncio.create_task(self._process_loop())

    async def stop_processing(self) -> None:
        """Stop order queue processing"""
        self._processing = False
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

    async def _process_loop(self) -> None:
        """Process orders from queue"""
        while self._processing:
            try:
                order = self._order_queue.pop()
                if order:
                    await self.submit_order(order)
                else:
                    await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Processing error: {e}")
                await asyncio.sleep(1)

    def _on_broker_update(self, data: Dict) -> None:
        """Handle broker order update.

        CRITICAL FIX: Uses thread-safe lock when accessing order dictionaries
        since this callback may fire from broker's thread while main thread
        is iterating.
        """
        event = data.get('event', '')
        order_data = data.get('order', {})

        broker_order_id = order_data.get('id', '')

        # Find matching order (thread-safe)
        order = None
        with self._orders_lock:
            for o in self._orders.values():
                if o.broker_order_id == broker_order_id:
                    order = o
                    break

        if not order:
            return

        # Update status based on event
        if event == 'fill':
            fill_qty = int(order_data.get('filled_qty', 0))
            fill_price = float(order_data.get('filled_avg_price', 0))

            if fill_qty > order.filled_quantity:
                new_fill = fill_qty - order.filled_quantity
                order.add_fill(new_fill, fill_price)

                if order.status == OrderStatus.FILLED:
                    with self._orders_lock:
                        if order.order_id in self._active_orders:
                            del self._active_orders[order.order_id]

                    audit_logger.log_order(
                        order_id=order.order_id,
                        symbol=order.symbol,
                        action="FILLED",
                        details=order.to_dict()
                    )

                    self._emit('order_filled', order)

        elif event == 'partial_fill':
            fill_qty = int(order_data.get('filled_qty', 0))
            fill_price = float(order_data.get('filled_avg_price', 0))

            if fill_qty > order.filled_quantity:
                new_fill = fill_qty - order.filled_quantity
                order.add_fill(new_fill, fill_price)
                self._emit('order_updated', order)

        elif event in ['canceled', 'cancelled']:
            order.status = OrderStatus.CANCELLED
            order.cancelled_at = datetime.now()

            if order.order_id in self._active_orders:
                del self._active_orders[order.order_id]

            self._emit('order_cancelled', order)

        elif event == 'rejected':
            order.status = OrderStatus.REJECTED
            order.notes.append(order_data.get('reject_reason', 'Unknown'))

            if order.order_id in self._active_orders:
                del self._active_orders[order.order_id]

            self._emit('order_rejected', order)

        order.updated_at = datetime.now()

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self._orders.get(order_id)

    def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        limit: int = 100
    ) -> List[Order]:
        """Get orders with optional filters"""
        orders = list(self._orders.values())

        if status:
            orders = [o for o in orders if o.status == status]

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        if strategy:
            orders = [o for o in orders if o.strategy_name == strategy]

        # Sort by creation time
        orders.sort(key=lambda o: o.created_at, reverse=True)

        return orders[:limit]

    def get_active_orders(self) -> List[Order]:
        """Get all active orders"""
        return list(self._active_orders.values())

    def get_fills_for_symbol(self, symbol: str) -> List[Dict]:
        """Get all fills for a symbol"""
        fills = []
        for order in self._orders.values():
            if order.symbol == symbol:
                for fill in order.fills:
                    fills.append({
                        'order_id': order.order_id,
                        'side': order.side.value,
                        **fill
                    })
        return fills

    def get_statistics(self) -> Dict[str, Any]:
        """Get order statistics"""
        total = len(self._orders)
        by_status = {}

        for order in self._orders.values():
            status = order.status.value
            by_status[status] = by_status.get(status, 0) + 1

        filled_orders = [o for o in self._orders.values() if o.status == OrderStatus.FILLED]
        total_filled_qty = sum(o.filled_quantity for o in filled_orders)
        total_notional = sum(o.filled_quantity * o.avg_fill_price for o in filled_orders)

        return {
            'total_orders': total,
            'active_orders': len(self._active_orders),
            'queued_orders': self._order_queue.size,
            'orders_by_status': by_status,
            'total_filled_quantity': total_filled_qty,
            'total_notional': total_notional
        }
