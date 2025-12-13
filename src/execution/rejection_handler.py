"""
Order Rejection Handler
=======================
JPMorgan-Level Order Rejection Management

Handles broker order rejections intelligently:
1. Categorizes rejection reasons
2. Determines if retry is appropriate
3. Adjusts order parameters for retry
4. Tracks rejection patterns
5. Alerts on systemic issues

Key Rejection Types:
- Insufficient funds → Reduce size or skip
- Market closed → Queue for open
- Symbol not tradeable → Exclude from universe
- Rate limit → Backoff and retry
- Invalid price → Adjust to market

Author: AlphaTrade Institutional System
Based on: ARCHITECTURAL_REVIEW_REPORT.md - Phase 2
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

from .order_manager import Order, OrderStatus
from ..utils.logger import get_logger, get_audit_logger

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class RejectionReason(Enum):
    """Categories of order rejections"""
    # Funds/margin
    INSUFFICIENT_FUNDS = "insufficient_funds"
    INSUFFICIENT_MARGIN = "insufficient_margin"
    EXCEEDS_BUYING_POWER = "exceeds_buying_power"

    # Market conditions
    MARKET_CLOSED = "market_closed"
    HALTED = "halted"
    NO_LIQUIDITY = "no_liquidity"

    # Symbol issues
    SYMBOL_NOT_TRADEABLE = "symbol_not_tradeable"
    SYMBOL_NOT_FOUND = "symbol_not_found"
    SYMBOL_DELISTED = "symbol_delisted"

    # Price issues
    PRICE_TOO_FAR = "price_too_far"
    LIMIT_PRICE_INVALID = "limit_price_invalid"
    STOP_PRICE_INVALID = "stop_price_invalid"

    # Quantity issues
    QUANTITY_TOO_SMALL = "quantity_too_small"
    QUANTITY_TOO_LARGE = "quantity_too_large"
    LOT_SIZE_VIOLATION = "lot_size_violation"

    # Rate limiting
    RATE_LIMITED = "rate_limited"
    TOO_MANY_ORDERS = "too_many_orders"

    # Account issues
    ACCOUNT_RESTRICTED = "account_restricted"
    PATTERN_DAY_TRADER = "pattern_day_trader"
    ACCOUNT_NOT_ENABLED = "account_not_enabled"

    # Order issues
    DUPLICATE_ORDER = "duplicate_order"
    ORDER_EXPIRED = "order_expired"
    INVALID_ORDER_TYPE = "invalid_order_type"

    # Technical issues
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    UNKNOWN = "unknown"


class RetryStrategy(Enum):
    """Strategies for handling rejections"""
    RETRY_IMMEDIATELY = "retry_immediately"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    RETRY_WITH_ADJUSTMENT = "retry_with_adjustment"
    QUEUE_FOR_LATER = "queue_for_later"
    CANCEL = "cancel"
    ALERT = "alert"


@dataclass
class RejectionInfo:
    """Detailed rejection information"""
    order_id: str
    symbol: str
    reason: RejectionReason
    raw_message: str
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    max_retries: int = 3
    last_retry: Optional[datetime] = None
    resolved: bool = False
    resolution: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'reason': self.reason.value,
            'raw_message': self.raw_message,
            'timestamp': self.timestamp.isoformat(),
            'retry_count': self.retry_count,
            'resolved': self.resolved,
            'resolution': self.resolution
        }


@dataclass
class RejectionPattern:
    """Pattern detection for rejections"""
    reason: RejectionReason
    count: int = 0
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    symbols: set = field(default_factory=set)

    @property
    def is_systemic(self) -> bool:
        """Check if this appears to be a systemic issue"""
        # Multiple symbols affected in short time = systemic
        time_window = (self.last_seen - self.first_seen).total_seconds()
        if time_window < 60 and self.count >= 5:
            return True
        if len(self.symbols) >= 3:
            return True
        return False


class OrderRejectionHandler:
    """
    Intelligent handling of order rejections.

    This is critical for production trading. Different rejections
    require different responses:

    - Insufficient funds → Reduce position size
    - Rate limited → Exponential backoff
    - Market closed → Queue for market open
    - Symbol halted → Remove from universe temporarily
    - Price too far → Convert to market order
    """

    def __init__(
        self,
        order_manager,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        alert_callback: Optional[Callable] = None
    ):
        self.order_manager = order_manager
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.alert_callback = alert_callback

        # Rejection tracking
        self._rejections: Dict[str, RejectionInfo] = {}  # order_id -> info
        self._patterns: Dict[RejectionReason, RejectionPattern] = {}

        # Queued orders (for market closed, etc.)
        self._queued_orders: List[Order] = []

        # Blocked symbols (halted, delisted, etc.)
        self._blocked_symbols: Dict[str, datetime] = {}  # symbol -> block_time

        # Statistics
        self._stats = {
            'total_rejections': 0,
            'retries_successful': 0,
            'retries_failed': 0,
            'orders_cancelled': 0,
            'orders_queued': 0,
            'by_reason': defaultdict(int)
        }

        # Reason to strategy mapping
        self._strategy_map: Dict[RejectionReason, Tuple[RetryStrategy, Dict]] = {
            # Retry immediately (transient issues)
            RejectionReason.TIMEOUT: (RetryStrategy.RETRY_IMMEDIATELY, {}),
            RejectionReason.CONNECTION_ERROR: (RetryStrategy.RETRY_WITH_BACKOFF, {}),

            # Retry with backoff
            RejectionReason.RATE_LIMITED: (RetryStrategy.RETRY_WITH_BACKOFF, {'multiplier': 2.0}),
            RejectionReason.TOO_MANY_ORDERS: (RetryStrategy.RETRY_WITH_BACKOFF, {'multiplier': 1.5}),

            # Retry with adjustment
            RejectionReason.INSUFFICIENT_FUNDS: (RetryStrategy.RETRY_WITH_ADJUSTMENT, {'reduce_pct': 0.5}),
            RejectionReason.EXCEEDS_BUYING_POWER: (RetryStrategy.RETRY_WITH_ADJUSTMENT, {'reduce_pct': 0.5}),
            RejectionReason.QUANTITY_TOO_LARGE: (RetryStrategy.RETRY_WITH_ADJUSTMENT, {'reduce_pct': 0.5}),
            RejectionReason.PRICE_TOO_FAR: (RetryStrategy.RETRY_WITH_ADJUSTMENT, {'use_market': True}),

            # Queue for later
            RejectionReason.MARKET_CLOSED: (RetryStrategy.QUEUE_FOR_LATER, {}),

            # Cancel (permanent issues)
            RejectionReason.SYMBOL_NOT_FOUND: (RetryStrategy.CANCEL, {}),
            RejectionReason.SYMBOL_DELISTED: (RetryStrategy.CANCEL, {}),
            RejectionReason.ACCOUNT_RESTRICTED: (RetryStrategy.ALERT, {}),
            RejectionReason.PATTERN_DAY_TRADER: (RetryStrategy.ALERT, {}),

            # Alert on unknown
            RejectionReason.UNKNOWN: (RetryStrategy.ALERT, {}),
        }

    async def handle_rejection(
        self,
        order: Order,
        raw_message: str
    ) -> Tuple[bool, str]:
        """
        Handle an order rejection.

        Args:
            order: The rejected order
            raw_message: Raw rejection message from broker

        Returns:
            Tuple of (retry_successful, resolution_message)
        """
        # Parse rejection reason
        reason = self._parse_rejection_reason(raw_message)

        # Create rejection info
        info = RejectionInfo(
            order_id=order.order_id,
            symbol=order.symbol,
            reason=reason,
            raw_message=raw_message,
            max_retries=self.max_retries
        )

        self._rejections[order.order_id] = info

        # Update statistics
        self._stats['total_rejections'] += 1
        self._stats['by_reason'][reason.value] += 1

        # Update pattern tracking
        self._update_pattern(reason, order.symbol)

        # Log rejection
        logger.warning(
            f"Order rejected: {order.order_id} [{order.symbol}] "
            f"Reason: {reason.value} - {raw_message}"
        )

        # Get handling strategy
        strategy, params = self._strategy_map.get(
            reason,
            (RetryStrategy.CANCEL, {})
        )

        # Execute strategy
        success, message = await self._execute_strategy(order, info, strategy, params)

        # Update resolution
        info.resolved = True
        info.resolution = message

        # Audit log
        audit_logger.log_order(
            order_id=order.order_id,
            symbol=order.symbol,
            action="REJECTION_HANDLED",
            details={
                'reason': reason.value,
                'strategy': strategy.value,
                'success': success,
                'message': message
            }
        )

        # Check for systemic issues
        pattern = self._patterns.get(reason)
        if pattern and pattern.is_systemic:
            await self._handle_systemic_issue(reason, pattern)

        return success, message

    def _parse_rejection_reason(self, message: str) -> RejectionReason:
        """Parse raw message to determine rejection reason"""
        message_lower = message.lower()

        # Funds/margin
        if any(k in message_lower for k in ['insufficient', 'not enough', 'buying power']):
            if 'margin' in message_lower:
                return RejectionReason.INSUFFICIENT_MARGIN
            return RejectionReason.INSUFFICIENT_FUNDS

        # Market conditions
        if 'market' in message_lower and ('closed' in message_lower or 'hours' in message_lower):
            return RejectionReason.MARKET_CLOSED

        if 'halt' in message_lower:
            return RejectionReason.HALTED

        # Symbol issues
        if 'not found' in message_lower or 'invalid symbol' in message_lower:
            return RejectionReason.SYMBOL_NOT_FOUND

        if 'not tradeable' in message_lower or 'cannot trade' in message_lower:
            return RejectionReason.SYMBOL_NOT_TRADEABLE

        if 'delist' in message_lower:
            return RejectionReason.SYMBOL_DELISTED

        # Price issues
        if 'price' in message_lower:
            if 'too far' in message_lower or 'away' in message_lower:
                return RejectionReason.PRICE_TOO_FAR
            if 'limit' in message_lower:
                return RejectionReason.LIMIT_PRICE_INVALID
            if 'stop' in message_lower:
                return RejectionReason.STOP_PRICE_INVALID

        # Quantity issues
        if 'quantity' in message_lower or 'size' in message_lower:
            if 'small' in message_lower or 'minimum' in message_lower:
                return RejectionReason.QUANTITY_TOO_SMALL
            if 'large' in message_lower or 'maximum' in message_lower:
                return RejectionReason.QUANTITY_TOO_LARGE
            if 'lot' in message_lower:
                return RejectionReason.LOT_SIZE_VIOLATION

        # Rate limiting
        if 'rate' in message_lower or 'throttl' in message_lower:
            return RejectionReason.RATE_LIMITED

        if 'too many' in message_lower:
            return RejectionReason.TOO_MANY_ORDERS

        # Account issues
        if 'restricted' in message_lower:
            return RejectionReason.ACCOUNT_RESTRICTED

        if 'pdt' in message_lower or 'pattern day' in message_lower:
            return RejectionReason.PATTERN_DAY_TRADER

        # Technical
        if 'timeout' in message_lower:
            return RejectionReason.TIMEOUT

        if 'connection' in message_lower or 'network' in message_lower:
            return RejectionReason.CONNECTION_ERROR

        if 'duplicate' in message_lower:
            return RejectionReason.DUPLICATE_ORDER

        return RejectionReason.UNKNOWN

    def _update_pattern(self, reason: RejectionReason, symbol: str) -> None:
        """Update rejection pattern tracking"""
        if reason not in self._patterns:
            self._patterns[reason] = RejectionPattern(reason=reason)

        pattern = self._patterns[reason]
        pattern.count += 1
        pattern.last_seen = datetime.now()
        pattern.symbols.add(symbol)

    async def _execute_strategy(
        self,
        order: Order,
        info: RejectionInfo,
        strategy: RetryStrategy,
        params: Dict
    ) -> Tuple[bool, str]:
        """Execute the handling strategy"""

        if strategy == RetryStrategy.RETRY_IMMEDIATELY:
            return await self._retry_order(order, info, delay=0)

        elif strategy == RetryStrategy.RETRY_WITH_BACKOFF:
            delay = self.backoff_base * (params.get('multiplier', 2.0) ** info.retry_count)
            return await self._retry_order(order, info, delay=delay)

        elif strategy == RetryStrategy.RETRY_WITH_ADJUSTMENT:
            return await self._retry_with_adjustment(order, info, params)

        elif strategy == RetryStrategy.QUEUE_FOR_LATER:
            return await self._queue_order(order, info)

        elif strategy == RetryStrategy.CANCEL:
            return await self._cancel_order(order, info)

        elif strategy == RetryStrategy.ALERT:
            return await self._alert_and_cancel(order, info)

        return False, "Unknown strategy"

    async def _retry_order(
        self,
        order: Order,
        info: RejectionInfo,
        delay: float
    ) -> Tuple[bool, str]:
        """Retry order with optional delay"""
        if info.retry_count >= info.max_retries:
            self._stats['retries_failed'] += 1
            return False, f"Max retries ({info.max_retries}) exceeded"

        if delay > 0:
            logger.info(f"Waiting {delay:.1f}s before retry")
            await asyncio.sleep(delay)

        info.retry_count += 1
        info.last_retry = datetime.now()

        # Create new order with same parameters
        new_order = await self.order_manager.create_order(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            time_in_force=order.time_in_force,
            strategy_name=order.strategy_name,
            signal_strength=order.signal_strength,
            signal_price=order.signal_price
        )

        success = await self.order_manager.submit_order(new_order)

        if success:
            self._stats['retries_successful'] += 1
            return True, f"Retry successful (attempt {info.retry_count})"
        else:
            return await self._retry_order(order, info, delay * 2)

    async def _retry_with_adjustment(
        self,
        order: Order,
        info: RejectionInfo,
        params: Dict
    ) -> Tuple[bool, str]:
        """Retry with adjusted parameters"""
        if info.retry_count >= info.max_retries:
            self._stats['retries_failed'] += 1
            return False, f"Max retries ({info.max_retries}) exceeded"

        info.retry_count += 1
        info.last_retry = datetime.now()

        # Adjust quantity
        new_quantity = order.quantity
        if 'reduce_pct' in params:
            new_quantity = int(order.quantity * params['reduce_pct'])
            if new_quantity < 1:
                self._stats['retries_failed'] += 1
                return False, "Quantity too small after reduction"

        # Adjust order type
        new_order_type = order.order_type
        new_limit_price = order.limit_price
        if params.get('use_market'):
            from .broker_api import OrderType
            new_order_type = OrderType.MARKET
            new_limit_price = None

        # Create adjusted order
        new_order = await self.order_manager.create_order(
            symbol=order.symbol,
            side=order.side,
            quantity=new_quantity,
            order_type=new_order_type,
            limit_price=new_limit_price,
            stop_price=order.stop_price,
            time_in_force=order.time_in_force,
            strategy_name=order.strategy_name,
            signal_strength=order.signal_strength,
            signal_price=order.signal_price
        )

        success = await self.order_manager.submit_order(new_order)

        if success:
            self._stats['retries_successful'] += 1
            adjustments = []
            if new_quantity != order.quantity:
                adjustments.append(f"qty: {order.quantity} → {new_quantity}")
            if new_order_type != order.order_type:
                adjustments.append(f"type: {order.order_type.value} → {new_order_type.value}")

            return True, f"Adjusted and resubmitted ({', '.join(adjustments)})"
        else:
            # Try again with more aggressive adjustment
            params['reduce_pct'] = params.get('reduce_pct', 1.0) * 0.5
            return await self._retry_with_adjustment(order, info, params)

    async def _queue_order(
        self,
        order: Order,
        info: RejectionInfo
    ) -> Tuple[bool, str]:
        """Queue order for later submission"""
        self._queued_orders.append(order)
        self._stats['orders_queued'] += 1

        return True, "Queued for later submission"

    async def _cancel_order(
        self,
        order: Order,
        info: RejectionInfo
    ) -> Tuple[bool, str]:
        """Cancel order (permanent rejection)"""
        self._stats['orders_cancelled'] += 1

        # Block symbol if needed
        if info.reason in [RejectionReason.SYMBOL_DELISTED, RejectionReason.SYMBOL_NOT_TRADEABLE]:
            self._blocked_symbols[order.symbol] = datetime.now()

        return False, f"Order cancelled: {info.reason.value}"

    async def _alert_and_cancel(
        self,
        order: Order,
        info: RejectionInfo
    ) -> Tuple[bool, str]:
        """Alert and cancel (serious issue)"""
        self._stats['orders_cancelled'] += 1

        message = (
            f"ALERT: Order {order.order_id} rejected\n"
            f"Symbol: {order.symbol}\n"
            f"Reason: {info.reason.value}\n"
            f"Message: {info.raw_message}"
        )

        logger.critical(message)

        if self.alert_callback:
            self.alert_callback(message)

        return False, f"Alert sent, order cancelled: {info.reason.value}"

    async def _handle_systemic_issue(
        self,
        reason: RejectionReason,
        pattern: RejectionPattern
    ) -> None:
        """Handle systemic rejection issue"""
        message = (
            f"SYSTEMIC REJECTION DETECTED\n"
            f"Reason: {reason.value}\n"
            f"Count: {pattern.count}\n"
            f"Symbols affected: {len(pattern.symbols)}\n"
            f"Time window: {(pattern.last_seen - pattern.first_seen).total_seconds():.0f}s"
        )

        logger.critical(message)

        if self.alert_callback:
            self.alert_callback(message)

        # If systemic, might need to pause trading
        if reason in [RejectionReason.ACCOUNT_RESTRICTED, RejectionReason.INSUFFICIENT_FUNDS]:
            logger.critical("TRADING SHOULD BE PAUSED - Systemic account issue")

    async def process_queue(self) -> int:
        """Process queued orders (call when market opens)"""
        processed = 0

        while self._queued_orders:
            order = self._queued_orders.pop(0)

            # Create fresh order
            new_order = await self.order_manager.create_order(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                order_type=order.order_type,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                time_in_force=order.time_in_force,
                strategy_name=order.strategy_name,
                signal_strength=order.signal_strength,
                signal_price=order.signal_price
            )

            success = await self.order_manager.submit_order(new_order)
            if success:
                processed += 1

        logger.info(f"Processed {processed} queued orders")
        return processed

    def is_symbol_blocked(self, symbol: str) -> bool:
        """Check if a symbol is blocked"""
        if symbol not in self._blocked_symbols:
            return False

        # Unblock after 24 hours
        block_time = self._blocked_symbols[symbol]
        if (datetime.now() - block_time) > timedelta(hours=24):
            del self._blocked_symbols[symbol]
            return False

        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get rejection statistics"""
        return {
            **self._stats,
            'by_reason': dict(self._stats['by_reason']),
            'queued_count': len(self._queued_orders),
            'blocked_symbols': list(self._blocked_symbols.keys()),
            'patterns': {
                reason.value: {
                    'count': pattern.count,
                    'symbols_affected': len(pattern.symbols),
                    'is_systemic': pattern.is_systemic
                }
                for reason, pattern in self._patterns.items()
            }
        }

    def clear_patterns(self) -> None:
        """Clear pattern tracking (call periodically)"""
        self._patterns.clear()
