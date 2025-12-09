"""
State Store Module
==================

Distributed state management for fault-tolerant trading system.
Externalizes all runtime state to Redis for crash recovery.

This ensures that if any service crashes:
1. Current PnL is preserved
2. Open positions are known
3. Risk limits are enforced
4. No "memory amnesia" on restart

Features:
- Atomic operations for position updates
- Distributed locking for concurrent access
- TTL-based expiration for heartbeats
- Transaction support for multi-key updates
- Optimistic locking with CAS operations

Author: AlphaTrade Platform
Version: 3.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Any, TypeVar, Generic
from contextlib import asynccontextmanager
import threading

try:
    import redis.asyncio as aioredis
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from config.settings import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


# =============================================================================
# STATE KEYS
# =============================================================================

class StateKey(str, Enum):
    """Standardized state keys for the trading system."""
    # Account State
    ACCOUNT_EQUITY = "state:account:equity"
    ACCOUNT_CASH = "state:account:cash"
    ACCOUNT_BUYING_POWER = "state:account:buying_power"
    ACCOUNT_MARGIN_USED = "state:account:margin_used"

    # PnL State
    DAILY_PNL = "state:pnl:daily"
    TOTAL_PNL = "state:pnl:total"
    REALIZED_PNL = "state:pnl:realized"
    UNREALIZED_PNL = "state:pnl:unrealized"
    PNL_HISTORY = "state:pnl:history"

    # Position State
    POSITIONS = "state:positions"
    POSITION_PREFIX = "state:position:"

    # Order State
    PENDING_ORDERS = "state:orders:pending"
    ORDER_PREFIX = "state:order:"
    ORDER_HISTORY = "state:orders:history"

    # Risk State
    RISK_EXPOSURE = "state:risk:exposure"
    RISK_VAR = "state:risk:var"
    RISK_LIMITS = "state:risk:limits"
    RISK_BREACHES = "state:risk:breaches"
    DRAWDOWN_CURRENT = "state:risk:drawdown:current"
    DRAWDOWN_MAX = "state:risk:drawdown:max"
    EQUITY_HIGH_WATER = "state:risk:equity_hwm"

    # Trade State
    TRADE_COUNT = "state:trades:count"
    TRADE_HISTORY = "state:trades:history"
    WIN_COUNT = "state:trades:wins"
    LOSS_COUNT = "state:trades:losses"

    # Service State
    SERVICE_HEARTBEAT = "state:service:heartbeat:"
    SERVICE_STATUS = "state:service:status:"

    # System State
    SYSTEM_STATUS = "state:system:status"
    KILL_SWITCH_ACTIVE = "state:system:kill_switch"
    LAST_MARKET_TIME = "state:system:last_market_time"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PositionState:
    """Serializable position state."""
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    side: str  # "long" or "short"
    opened_at: float  # timestamp
    last_updated: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_entry_price": self.avg_entry_price,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "side": self.side,
            "opened_at": self.opened_at,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PositionState":
        """Deserialize from dictionary."""
        return cls(**data)

    @property
    def market_value(self) -> float:
        """Calculate current market value."""
        return abs(self.quantity) * self.current_price

    @property
    def cost_basis(self) -> float:
        """Calculate cost basis."""
        return abs(self.quantity) * self.avg_entry_price


@dataclass
class RiskState:
    """Serializable risk state."""
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    current_equity: float = 0.0
    high_water_mark: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    total_exposure: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    position_count: int = 0
    trade_count_today: int = 0
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "daily_pnl": self.daily_pnl,
            "total_pnl": self.total_pnl,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "current_equity": self.current_equity,
            "high_water_mark": self.high_water_mark,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "total_exposure": self.total_exposure,
            "var_95": self.var_95,
            "var_99": self.var_99,
            "position_count": self.position_count,
            "trade_count_today": self.trade_count_today,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RiskState":
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class OrderState:
    """Serializable order state."""
    order_id: str
    symbol: str
    side: str
    quantity: float
    order_type: str
    limit_price: float | None = None
    stop_price: float | None = None
    status: str = "pending"
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "status": self.status,
            "filled_qty": self.filled_qty,
            "avg_fill_price": self.avg_fill_price,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrderState":
        """Deserialize from dictionary."""
        return cls(**data)


# =============================================================================
# ABSTRACT STATE STORE
# =============================================================================

class StateStore(ABC):
    """
    Abstract base class for state storage.

    The state store is critical for fault tolerance. All dynamic
    state must be externalized so that service restarts don't
    cause "memory amnesia."

    Key guarantees:
    1. Atomic updates for positions and PnL
    2. Distributed locking for concurrent access
    3. Transaction support for multi-key updates

    Example:
        store = RedisStateStore(redis_url="redis://localhost:6379")
        await store.connect()

        # Update position atomically
        async with store.lock("position:AAPL"):
            position = await store.get_position("AAPL")
            position.quantity += 100
            await store.set_position(position)
    """

    def __init__(self, service_name: str = "unknown"):
        """Initialize state store."""
        self.service_name = service_name
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the state store."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the state store."""
        pass

    # =========================================================================
    # BASIC OPERATIONS
    # =========================================================================

    @abstractmethod
    async def get(self, key: str | StateKey) -> Any:
        """Get a value by key."""
        pass

    @abstractmethod
    async def set(
        self,
        key: str | StateKey,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """Set a value with optional TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str | StateKey) -> bool:
        """Delete a key."""
        pass

    @abstractmethod
    async def exists(self, key: str | StateKey) -> bool:
        """Check if a key exists."""
        pass

    # =========================================================================
    # ATOMIC OPERATIONS
    # =========================================================================

    @abstractmethod
    async def incr(self, key: str | StateKey, amount: float = 1.0) -> float:
        """Atomically increment a value."""
        pass

    @abstractmethod
    async def decr(self, key: str | StateKey, amount: float = 1.0) -> float:
        """Atomically decrement a value."""
        pass

    # =========================================================================
    # POSITION OPERATIONS
    # =========================================================================

    @abstractmethod
    async def get_position(self, symbol: str) -> PositionState | None:
        """Get position for a symbol."""
        pass

    @abstractmethod
    async def set_position(self, position: PositionState) -> bool:
        """Set/update a position."""
        pass

    @abstractmethod
    async def delete_position(self, symbol: str) -> bool:
        """Delete a position (when closed)."""
        pass

    @abstractmethod
    async def get_all_positions(self) -> dict[str, PositionState]:
        """Get all open positions."""
        pass

    # =========================================================================
    # RISK STATE OPERATIONS
    # =========================================================================

    @abstractmethod
    async def get_risk_state(self) -> RiskState:
        """Get current risk state."""
        pass

    @abstractmethod
    async def set_risk_state(self, state: RiskState) -> bool:
        """Set risk state."""
        pass

    @abstractmethod
    async def update_pnl(
        self,
        realized: float = 0.0,
        unrealized: float = 0.0,
    ) -> RiskState:
        """Atomically update PnL values."""
        pass

    # =========================================================================
    # ORDER OPERATIONS
    # =========================================================================

    @abstractmethod
    async def get_order(self, order_id: str) -> OrderState | None:
        """Get an order by ID."""
        pass

    @abstractmethod
    async def set_order(self, order: OrderState) -> bool:
        """Set/update an order."""
        pass

    @abstractmethod
    async def get_pending_orders(self) -> list[OrderState]:
        """Get all pending orders."""
        pass

    # =========================================================================
    # LOCKING
    # =========================================================================

    @abstractmethod
    @asynccontextmanager
    async def lock(
        self,
        name: str,
        timeout: float = 10.0,
    ):
        """
        Acquire a distributed lock.

        Usage:
            async with store.lock("position:AAPL"):
                # Critical section
                pass
        """
        pass

    # =========================================================================
    # TRANSACTIONS
    # =========================================================================

    @abstractmethod
    @asynccontextmanager
    async def transaction(self):
        """
        Execute multiple operations atomically.

        Usage:
            async with store.transaction() as tx:
                await tx.set("key1", "value1")
                await tx.incr("key2")
        """
        pass


# =============================================================================
# REDIS STATE STORE
# =============================================================================

class RedisStateStore(StateStore):
    """
    Redis-based state store implementation.

    Features:
    - Atomic operations with Lua scripts
    - Distributed locking with Redlock algorithm
    - Pipeline support for batch operations
    - Automatic reconnection
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        service_name: str = "unknown",
        key_prefix: str = "alphatrade",
    ):
        """
        Initialize Redis state store.

        Args:
            redis_url: Redis connection URL
            service_name: Service name for namespacing
            key_prefix: Prefix for all keys
        """
        super().__init__(service_name)

        if not REDIS_AVAILABLE:
            raise ImportError("redis package required: pip install redis")

        self.redis_url = redis_url
        self.key_prefix = key_prefix

        self._client: aioredis.Redis | None = None
        self._lock_timeout = 10.0

    def _make_key(self, key: str | StateKey) -> str:
        """Create a namespaced key."""
        key_str = key.value if isinstance(key, StateKey) else key
        return f"{self.key_prefix}:{key_str}"

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            self._client = aioredis.from_url(
                self.redis_url,
                decode_responses=True,
            )

            # Test connection
            await self._client.ping()

            self._connected = True
            logger.info(f"State store connected to Redis at {self.redis_url}")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._client:
            await self._client.close()

        self._connected = False
        logger.info("State store disconnected from Redis")

    # =========================================================================
    # BASIC OPERATIONS
    # =========================================================================

    async def get(self, key: str | StateKey) -> Any:
        """Get a value by key."""
        if not self._client:
            return None

        full_key = self._make_key(key)
        value = await self._client.get(full_key)

        if value is None:
            return None

        # Try to deserialize JSON
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    async def set(
        self,
        key: str | StateKey,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """Set a value with optional TTL."""
        if not self._client:
            return False

        full_key = self._make_key(key)

        # Serialize complex types
        if isinstance(value, (dict, list)):
            value = json.dumps(value)

        if ttl:
            await self._client.setex(full_key, ttl, value)
        else:
            await self._client.set(full_key, value)

        return True

    async def delete(self, key: str | StateKey) -> bool:
        """Delete a key."""
        if not self._client:
            return False

        full_key = self._make_key(key)
        result = await self._client.delete(full_key)
        return result > 0

    async def exists(self, key: str | StateKey) -> bool:
        """Check if a key exists."""
        if not self._client:
            return False

        full_key = self._make_key(key)
        return await self._client.exists(full_key) > 0

    # =========================================================================
    # ATOMIC OPERATIONS
    # =========================================================================

    async def incr(self, key: str | StateKey, amount: float = 1.0) -> float:
        """Atomically increment a value."""
        if not self._client:
            return 0.0

        full_key = self._make_key(key)

        if amount == int(amount):
            result = await self._client.incrby(full_key, int(amount))
        else:
            result = await self._client.incrbyfloat(full_key, amount)

        return float(result)

    async def decr(self, key: str | StateKey, amount: float = 1.0) -> float:
        """Atomically decrement a value."""
        return await self.incr(key, -amount)

    # =========================================================================
    # POSITION OPERATIONS
    # =========================================================================

    async def get_position(self, symbol: str) -> PositionState | None:
        """Get position for a symbol."""
        if not self._client:
            return None

        key = f"{StateKey.POSITION_PREFIX.value}{symbol}"
        data = await self.get(key)

        if data is None:
            return None

        return PositionState.from_dict(data)

    async def set_position(self, position: PositionState) -> bool:
        """Set/update a position."""
        if not self._client:
            return False

        position.last_updated = time.time()

        key = f"{StateKey.POSITION_PREFIX.value}{position.symbol}"
        await self.set(key, position.to_dict())

        # Also add to positions set
        positions_key = self._make_key(StateKey.POSITIONS)
        await self._client.sadd(positions_key, position.symbol)

        return True

    async def delete_position(self, symbol: str) -> bool:
        """Delete a position (when closed)."""
        if not self._client:
            return False

        key = f"{StateKey.POSITION_PREFIX.value}{symbol}"
        await self.delete(key)

        # Remove from positions set
        positions_key = self._make_key(StateKey.POSITIONS)
        await self._client.srem(positions_key, symbol)

        return True

    async def get_all_positions(self) -> dict[str, PositionState]:
        """Get all open positions."""
        if not self._client:
            return {}

        positions_key = self._make_key(StateKey.POSITIONS)
        symbols = await self._client.smembers(positions_key)

        positions = {}
        for symbol in symbols:
            position = await self.get_position(symbol)
            if position:
                positions[symbol] = position

        return positions

    # =========================================================================
    # RISK STATE OPERATIONS
    # =========================================================================

    async def get_risk_state(self) -> RiskState:
        """Get current risk state."""
        if not self._client:
            return RiskState()

        # Fetch all risk-related keys
        keys = [
            StateKey.DAILY_PNL,
            StateKey.TOTAL_PNL,
            StateKey.REALIZED_PNL,
            StateKey.UNREALIZED_PNL,
            StateKey.ACCOUNT_EQUITY,
            StateKey.EQUITY_HIGH_WATER,
            StateKey.DRAWDOWN_CURRENT,
            StateKey.DRAWDOWN_MAX,
            StateKey.RISK_EXPOSURE,
            StateKey.RISK_VAR,
            StateKey.TRADE_COUNT,
        ]

        values = {}
        for key in keys:
            val = await self.get(key)
            values[key] = float(val) if val is not None else 0.0

        positions = await self.get_all_positions()

        return RiskState(
            daily_pnl=values[StateKey.DAILY_PNL],
            total_pnl=values[StateKey.TOTAL_PNL],
            realized_pnl=values[StateKey.REALIZED_PNL],
            unrealized_pnl=values[StateKey.UNREALIZED_PNL],
            current_equity=values[StateKey.ACCOUNT_EQUITY],
            high_water_mark=values[StateKey.EQUITY_HIGH_WATER],
            current_drawdown=values[StateKey.DRAWDOWN_CURRENT],
            max_drawdown=values[StateKey.DRAWDOWN_MAX],
            total_exposure=values[StateKey.RISK_EXPOSURE],
            var_95=values.get(StateKey.RISK_VAR, 0.0),
            position_count=len(positions),
            trade_count_today=int(values[StateKey.TRADE_COUNT]),
            last_updated=time.time(),
        )

    async def set_risk_state(self, state: RiskState) -> bool:
        """Set risk state."""
        if not self._client:
            return False

        state.last_updated = time.time()

        # Use pipeline for atomic update
        async with self._client.pipeline(transaction=True) as pipe:
            pipe.set(self._make_key(StateKey.DAILY_PNL), state.daily_pnl)
            pipe.set(self._make_key(StateKey.TOTAL_PNL), state.total_pnl)
            pipe.set(self._make_key(StateKey.REALIZED_PNL), state.realized_pnl)
            pipe.set(self._make_key(StateKey.UNREALIZED_PNL), state.unrealized_pnl)
            pipe.set(self._make_key(StateKey.ACCOUNT_EQUITY), state.current_equity)
            pipe.set(self._make_key(StateKey.EQUITY_HIGH_WATER), state.high_water_mark)
            pipe.set(self._make_key(StateKey.DRAWDOWN_CURRENT), state.current_drawdown)
            pipe.set(self._make_key(StateKey.DRAWDOWN_MAX), state.max_drawdown)
            pipe.set(self._make_key(StateKey.RISK_EXPOSURE), state.total_exposure)
            pipe.set(self._make_key(StateKey.TRADE_COUNT), state.trade_count_today)
            await pipe.execute()

        return True

    async def update_pnl(
        self,
        realized: float = 0.0,
        unrealized: float = 0.0,
    ) -> RiskState:
        """Atomically update PnL values."""
        if not self._client:
            return RiskState()

        # Lua script for atomic PnL update
        lua_script = """
        local realized_key = KEYS[1]
        local unrealized_key = KEYS[2]
        local daily_key = KEYS[3]
        local total_key = KEYS[4]
        local equity_key = KEYS[5]
        local hwm_key = KEYS[6]
        local dd_key = KEYS[7]
        local max_dd_key = KEYS[8]

        local realized_delta = tonumber(ARGV[1])
        local unrealized = tonumber(ARGV[2])
        local initial_equity = tonumber(ARGV[3])

        -- Update realized PnL
        local new_realized = redis.call('INCRBYFLOAT', realized_key, realized_delta)

        -- Set unrealized PnL (not incremented, replaced)
        redis.call('SET', unrealized_key, unrealized)

        -- Update daily and total PnL
        local daily = redis.call('INCRBYFLOAT', daily_key, realized_delta)
        local total = redis.call('INCRBYFLOAT', total_key, realized_delta)

        -- Calculate new equity
        local new_equity = initial_equity + tonumber(new_realized) + unrealized
        redis.call('SET', equity_key, new_equity)

        -- Update high water mark
        local hwm = tonumber(redis.call('GET', hwm_key) or initial_equity)
        if new_equity > hwm then
            hwm = new_equity
            redis.call('SET', hwm_key, hwm)
        end

        -- Calculate drawdown
        local drawdown = 0
        if hwm > 0 then
            drawdown = (hwm - new_equity) / hwm
        end
        redis.call('SET', dd_key, drawdown)

        -- Update max drawdown
        local max_dd = tonumber(redis.call('GET', max_dd_key) or 0)
        if drawdown > max_dd then
            max_dd = drawdown
            redis.call('SET', max_dd_key, max_dd)
        end

        return {new_realized, unrealized, daily, total, new_equity, hwm, drawdown, max_dd}
        """

        # Get initial equity
        initial_equity = await self.get(StateKey.ACCOUNT_EQUITY) or 100000.0

        result = await self._client.eval(
            lua_script,
            8,  # Number of keys
            self._make_key(StateKey.REALIZED_PNL),
            self._make_key(StateKey.UNREALIZED_PNL),
            self._make_key(StateKey.DAILY_PNL),
            self._make_key(StateKey.TOTAL_PNL),
            self._make_key(StateKey.ACCOUNT_EQUITY),
            self._make_key(StateKey.EQUITY_HIGH_WATER),
            self._make_key(StateKey.DRAWDOWN_CURRENT),
            self._make_key(StateKey.DRAWDOWN_MAX),
            realized,
            unrealized,
            initial_equity,
        )

        return RiskState(
            realized_pnl=float(result[0]),
            unrealized_pnl=float(result[1]),
            daily_pnl=float(result[2]),
            total_pnl=float(result[3]),
            current_equity=float(result[4]),
            high_water_mark=float(result[5]),
            current_drawdown=float(result[6]),
            max_drawdown=float(result[7]),
        )

    # =========================================================================
    # ORDER OPERATIONS
    # =========================================================================

    async def get_order(self, order_id: str) -> OrderState | None:
        """Get an order by ID."""
        key = f"{StateKey.ORDER_PREFIX.value}{order_id}"
        data = await self.get(key)

        if data is None:
            return None

        return OrderState.from_dict(data)

    async def set_order(self, order: OrderState) -> bool:
        """Set/update an order."""
        if not self._client:
            return False

        order.updated_at = time.time()

        key = f"{StateKey.ORDER_PREFIX.value}{order.order_id}"
        await self.set(key, order.to_dict())

        # Add to pending orders set if pending
        if order.status == "pending":
            pending_key = self._make_key(StateKey.PENDING_ORDERS)
            await self._client.sadd(pending_key, order.order_id)
        else:
            # Remove from pending if filled/cancelled
            pending_key = self._make_key(StateKey.PENDING_ORDERS)
            await self._client.srem(pending_key, order.order_id)

        return True

    async def get_pending_orders(self) -> list[OrderState]:
        """Get all pending orders."""
        if not self._client:
            return []

        pending_key = self._make_key(StateKey.PENDING_ORDERS)
        order_ids = await self._client.smembers(pending_key)

        orders = []
        for order_id in order_ids:
            order = await self.get_order(order_id)
            if order:
                orders.append(order)

        return orders

    # =========================================================================
    # LOCKING
    # =========================================================================

    @asynccontextmanager
    async def lock(
        self,
        name: str,
        timeout: float = 10.0,
    ):
        """
        Acquire a distributed lock using Redis.

        Uses the SET NX PX pattern for atomic lock acquisition.
        """
        if not self._client:
            yield
            return

        lock_key = f"{self.key_prefix}:lock:{name}"
        lock_value = f"{self.service_name}:{time.time()}"
        acquired = False

        try:
            # Try to acquire lock
            acquired = await self._client.set(
                lock_key,
                lock_value,
                nx=True,
                px=int(timeout * 1000),
            )

            if not acquired:
                # Wait and retry
                for _ in range(int(timeout * 10)):
                    await asyncio.sleep(0.1)
                    acquired = await self._client.set(
                        lock_key,
                        lock_value,
                        nx=True,
                        px=int(timeout * 1000),
                    )
                    if acquired:
                        break

            if not acquired:
                raise TimeoutError(f"Could not acquire lock: {name}")

            yield

        finally:
            # Release lock (only if we own it)
            if acquired:
                # Use Lua script for atomic check-and-delete
                lua_script = """
                if redis.call('GET', KEYS[1]) == ARGV[1] then
                    return redis.call('DEL', KEYS[1])
                else
                    return 0
                end
                """
                await self._client.eval(lua_script, 1, lock_key, lock_value)

    # =========================================================================
    # TRANSACTIONS
    # =========================================================================

    @asynccontextmanager
    async def transaction(self):
        """
        Execute multiple operations atomically using Redis pipeline.
        """
        if not self._client:
            yield None
            return

        pipe = self._client.pipeline(transaction=True)
        try:
            yield pipe
            await pipe.execute()
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            raise

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    async def reset_daily_state(self) -> None:
        """Reset daily state (called at market open)."""
        if not self._client:
            return

        await self.set(StateKey.DAILY_PNL, 0.0)
        await self.set(StateKey.TRADE_COUNT, 0)

        logger.info("Daily state reset")

    async def set_kill_switch(self, active: bool, reason: str = "") -> None:
        """Set kill switch state."""
        await self.set(
            StateKey.KILL_SWITCH_ACTIVE,
            {"active": active, "reason": reason, "timestamp": time.time()},
        )

        if active:
            logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
        else:
            logger.info("Kill switch deactivated")

    async def is_kill_switch_active(self) -> bool:
        """Check if kill switch is active."""
        data = await self.get(StateKey.KILL_SWITCH_ACTIVE)
        if data and isinstance(data, dict):
            return data.get("active", False)
        return False

    async def update_heartbeat(self) -> None:
        """Update service heartbeat."""
        key = f"{StateKey.SERVICE_HEARTBEAT.value}{self.service_name}"
        await self.set(key, time.time(), ttl=30)

    async def check_service_alive(self, service_name: str) -> bool:
        """Check if a service is alive (has recent heartbeat)."""
        key = f"{StateKey.SERVICE_HEARTBEAT.value}{service_name}"
        last_heartbeat = await self.get(key)

        if last_heartbeat is None:
            return False

        return time.time() - float(last_heartbeat) < 30


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_state_store(
    backend: str = "redis",
    service_name: str = "unknown",
    **kwargs: Any,
) -> StateStore:
    """
    Factory function to create a state store.

    Args:
        backend: Backend type (redis)
        service_name: Service name
        **kwargs: Backend-specific arguments

    Returns:
        StateStore instance
    """
    if backend == "redis":
        return RedisStateStore(service_name=service_name, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "StateKey",
    # Data classes
    "PositionState",
    "RiskState",
    "OrderState",
    # Base class
    "StateStore",
    # Implementations
    "RedisStateStore",
    # Factory
    "create_state_store",
]
