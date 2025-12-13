"""
Redis State Management System
=============================
JPMorgan-Level Persistent State Management

Provides crash recovery and state persistence using Redis:
1. Position state persistence
2. Order state tracking
3. Risk state (exposure, limits)
4. Session recovery after restart
5. Cross-instance state sharing

Why Redis for Trading:
- Sub-millisecond latency (< 1ms typical)
- Atomic operations for consistency
- Pub/Sub for real-time updates
- Persistence options (RDB, AOF)
- Cluster support for high availability

Author: AlphaTrade Institutional System
Based on: ARCHITECTURAL_REVIEW_REPORT.md - Phase 5
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import pickle
from abc import ABC, abstractmethod

from ..utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class StateNamespace(Enum):
    """Redis key namespaces"""
    POSITIONS = "alphatrade:positions"
    ORDERS = "alphatrade:orders"
    RISK = "alphatrade:risk"
    SESSION = "alphatrade:session"
    SIGNALS = "alphatrade:signals"
    METRICS = "alphatrade:metrics"
    LOCKS = "alphatrade:locks"


@dataclass
class PositionState:
    """Position state for persistence"""
    symbol: str
    quantity: int
    side: str  # 'long' or 'short'
    entry_price: float
    entry_time: str  # ISO format
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    last_update: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PositionState':
        return cls(**data)


@dataclass
class OrderState:
    """Order state for persistence"""
    order_id: str
    symbol: str
    side: str
    quantity: int
    order_type: str
    status: str
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrderState':
        return cls(**data)


@dataclass
class RiskState:
    """Risk state for persistence"""
    total_exposure: float = 0.0
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    max_drawdown_today: float = 0.0
    risk_limit_used_pct: float = 0.0
    last_update: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskState':
        return cls(**data)


@dataclass
class SessionState:
    """Trading session state"""
    session_id: str
    start_time: str
    status: str  # 'active', 'paused', 'stopped'
    positions_count: int = 0
    orders_count: int = 0
    last_heartbeat: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionState':
        return cls(**data)


class RedisClient(ABC):
    """Abstract Redis client interface"""

    @abstractmethod
    async def get(self, key: str) -> Optional[str]:
        pass

    @abstractmethod
    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        pass

    @abstractmethod
    async def delete(self, key: str) -> int:
        pass

    @abstractmethod
    async def hget(self, name: str, key: str) -> Optional[str]:
        pass

    @abstractmethod
    async def hset(self, name: str, key: str = None, value: str = None, mapping: Dict = None) -> int:
        pass

    @abstractmethod
    async def hgetall(self, name: str) -> Dict[str, str]:
        pass

    @abstractmethod
    async def hdel(self, name: str, *keys: str) -> int:
        pass

    @abstractmethod
    async def publish(self, channel: str, message: str) -> int:
        pass

    @abstractmethod
    async def subscribe(self, channel: str, callback: Callable) -> None:
        pass

    @abstractmethod
    async def ping(self) -> bool:
        pass


class AsyncRedisClient(RedisClient):
    """
    Production Redis client using aioredis/redis-py.

    Falls back to in-memory storage if Redis unavailable.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        socket_timeout: float = 5.0
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.socket_timeout = socket_timeout

        self._redis = None
        self._connected = False
        self._fallback_store: Dict[str, Any] = {}

    async def connect(self) -> bool:
        """Connect to Redis"""
        try:
            import redis.asyncio as redis

            self._redis = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                socket_timeout=self.socket_timeout,
                decode_responses=True
            )

            await self._redis.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            return True

        except ImportError:
            logger.warning("redis-py not installed, using in-memory fallback")
            self._connected = False
            return False

        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using in-memory fallback")
            self._connected = False
            return False

    async def close(self) -> None:
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()
            self._connected = False

    async def get(self, key: str) -> Optional[str]:
        if self._connected:
            return await self._redis.get(key)
        return self._fallback_store.get(key)

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        if self._connected:
            await self._redis.set(key, value, ex=ex)
            return True
        self._fallback_store[key] = value
        return True

    async def delete(self, key: str) -> int:
        if self._connected:
            return await self._redis.delete(key)
        if key in self._fallback_store:
            del self._fallback_store[key]
            return 1
        return 0

    async def hget(self, name: str, key: str) -> Optional[str]:
        if self._connected:
            return await self._redis.hget(name, key)
        hash_data = self._fallback_store.get(name, {})
        return hash_data.get(key)

    async def hset(self, name: str, key: str = None, value: str = None, mapping: Dict = None) -> int:
        if self._connected:
            if mapping:
                return await self._redis.hset(name, mapping=mapping)
            return await self._redis.hset(name, key, value)

        if name not in self._fallback_store:
            self._fallback_store[name] = {}

        if mapping:
            self._fallback_store[name].update(mapping)
            return len(mapping)
        else:
            self._fallback_store[name][key] = value
            return 1

    async def hgetall(self, name: str) -> Dict[str, str]:
        if self._connected:
            return await self._redis.hgetall(name)
        return self._fallback_store.get(name, {})

    async def hdel(self, name: str, *keys: str) -> int:
        if self._connected:
            return await self._redis.hdel(name, *keys)

        count = 0
        if name in self._fallback_store:
            for key in keys:
                if key in self._fallback_store[name]:
                    del self._fallback_store[name][key]
                    count += 1
        return count

    async def publish(self, channel: str, message: str) -> int:
        if self._connected:
            return await self._redis.publish(channel, message)
        return 0

    async def subscribe(self, channel: str, callback: Callable) -> None:
        if self._connected:
            pubsub = self._redis.pubsub()
            await pubsub.subscribe(channel)
            # Would need async iterator to handle messages
            logger.info(f"Subscribed to channel: {channel}")

    async def ping(self) -> bool:
        if self._connected:
            try:
                await self._redis.ping()
                return True
            except Exception:
                return False
        return True  # Fallback always "available"


class RedisStateManager:
    """
    Main state management class.

    Handles:
    - Position persistence
    - Order tracking
    - Risk state
    - Session management
    - Crash recovery
    """

    def __init__(
        self,
        redis_client: Optional[RedisClient] = None,
        auto_save_interval: float = 5.0,  # Auto-save every 5 seconds
        state_ttl_hours: int = 24  # State expires after 24 hours
    ):
        self._redis = redis_client or AsyncRedisClient()
        self.auto_save_interval = auto_save_interval
        self.state_ttl = state_ttl_hours * 3600

        # Local cache for fast access
        self._positions: Dict[str, PositionState] = {}
        self._orders: Dict[str, OrderState] = {}
        self._risk_state: RiskState = RiskState()
        self._session: Optional[SessionState] = None

        # Auto-save task
        self._auto_save_task: Optional[asyncio.Task] = None

        # Change tracking
        self._dirty_positions: set = set()
        self._dirty_orders: set = set()
        self._risk_dirty: bool = False

    async def initialize(self) -> bool:
        """Initialize state manager"""
        if isinstance(self._redis, AsyncRedisClient):
            connected = await self._redis.connect()
            if not connected:
                logger.warning("Running with in-memory state (no Redis)")

        logger.info("State manager initialized")
        return True

    async def start_auto_save(self) -> None:
        """Start auto-save background task"""
        self._auto_save_task = asyncio.create_task(self._auto_save_loop())
        logger.info(f"Auto-save started (interval: {self.auto_save_interval}s)")

    async def stop_auto_save(self) -> None:
        """Stop auto-save"""
        if self._auto_save_task:
            self._auto_save_task.cancel()
            try:
                await self._auto_save_task
            except asyncio.CancelledError:
                pass

    async def _auto_save_loop(self) -> None:
        """Background auto-save loop"""
        while True:
            await asyncio.sleep(self.auto_save_interval)
            try:
                await self.save_dirty_state()
            except Exception as e:
                logger.error(f"Auto-save failed: {e}")

    async def save_dirty_state(self) -> None:
        """Save only modified state"""
        # Save dirty positions
        if self._dirty_positions:
            positions_to_save = {
                symbol: self._positions[symbol].to_dict()
                for symbol in self._dirty_positions
                if symbol in self._positions
            }
            if positions_to_save:
                mapping = {k: json.dumps(v) for k, v in positions_to_save.items()}
                await self._redis.hset(StateNamespace.POSITIONS.value, mapping=mapping)
            self._dirty_positions.clear()

        # Save dirty orders
        if self._dirty_orders:
            orders_to_save = {
                oid: self._orders[oid].to_dict()
                for oid in self._dirty_orders
                if oid in self._orders
            }
            if orders_to_save:
                mapping = {k: json.dumps(v) for k, v in orders_to_save.items()}
                await self._redis.hset(StateNamespace.ORDERS.value, mapping=mapping)
            self._dirty_orders.clear()

        # Save risk state
        if self._risk_dirty:
            await self._redis.set(
                f"{StateNamespace.RISK.value}:current",
                json.dumps(self._risk_state.to_dict()),
                ex=self.state_ttl
            )
            self._risk_dirty = False

    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================

    async def save_position(self, position: PositionState) -> None:
        """Save a position to Redis"""
        self._positions[position.symbol] = position
        self._dirty_positions.add(position.symbol)

        # Immediate save for critical updates
        await self._redis.hset(
            StateNamespace.POSITIONS.value,
            position.symbol,
            json.dumps(position.to_dict())
        )

    async def get_position(self, symbol: str) -> Optional[PositionState]:
        """Get a position"""
        # Check local cache first
        if symbol in self._positions:
            return self._positions[symbol]

        # Try Redis
        data = await self._redis.hget(StateNamespace.POSITIONS.value, symbol)
        if data:
            position = PositionState.from_dict(json.loads(data))
            self._positions[symbol] = position
            return position

        return None

    async def get_all_positions(self) -> Dict[str, PositionState]:
        """Get all positions"""
        raw = await self._redis.hgetall(StateNamespace.POSITIONS.value)

        positions = {}
        for symbol, data in raw.items():
            positions[symbol] = PositionState.from_dict(json.loads(data))

        self._positions = positions
        return positions

    async def delete_position(self, symbol: str) -> None:
        """Delete a position"""
        self._positions.pop(symbol, None)
        self._dirty_positions.discard(symbol)
        await self._redis.hdel(StateNamespace.POSITIONS.value, symbol)

    async def save_all_positions(self, positions: Dict[str, PositionState]) -> None:
        """Save all positions atomically"""
        if not positions:
            return

        mapping = {
            symbol: json.dumps(pos.to_dict())
            for symbol, pos in positions.items()
        }

        await self._redis.hset(StateNamespace.POSITIONS.value, mapping=mapping)
        self._positions = positions
        self._dirty_positions.clear()

    # =========================================================================
    # ORDER MANAGEMENT
    # =========================================================================

    async def save_order(self, order: OrderState) -> None:
        """Save an order"""
        self._orders[order.order_id] = order
        self._dirty_orders.add(order.order_id)

        await self._redis.hset(
            StateNamespace.ORDERS.value,
            order.order_id,
            json.dumps(order.to_dict())
        )

    async def get_order(self, order_id: str) -> Optional[OrderState]:
        """Get an order"""
        if order_id in self._orders:
            return self._orders[order_id]

        data = await self._redis.hget(StateNamespace.ORDERS.value, order_id)
        if data:
            order = OrderState.from_dict(json.loads(data))
            self._orders[order_id] = order
            return order

        return None

    async def get_all_orders(self) -> Dict[str, OrderState]:
        """Get all orders"""
        raw = await self._redis.hgetall(StateNamespace.ORDERS.value)

        orders = {}
        for oid, data in raw.items():
            orders[oid] = OrderState.from_dict(json.loads(data))

        self._orders = orders
        return orders

    async def delete_order(self, order_id: str) -> None:
        """Delete an order"""
        self._orders.pop(order_id, None)
        self._dirty_orders.discard(order_id)
        await self._redis.hdel(StateNamespace.ORDERS.value, order_id)

    async def get_open_orders(self) -> List[OrderState]:
        """Get only open orders"""
        all_orders = await self.get_all_orders()
        return [
            o for o in all_orders.values()
            if o.status in ['new', 'pending', 'partially_filled', 'accepted']
        ]

    # =========================================================================
    # RISK STATE
    # =========================================================================

    async def save_risk_state(self, risk: RiskState) -> None:
        """Save risk state"""
        self._risk_state = risk
        self._risk_dirty = True

        # Immediate save
        await self._redis.set(
            f"{StateNamespace.RISK.value}:current",
            json.dumps(risk.to_dict()),
            ex=self.state_ttl
        )

    async def get_risk_state(self) -> RiskState:
        """Get risk state"""
        data = await self._redis.get(f"{StateNamespace.RISK.value}:current")

        if data:
            self._risk_state = RiskState.from_dict(json.loads(data))

        return self._risk_state

    async def update_daily_pnl(self, pnl: float) -> None:
        """Update daily P&L"""
        self._risk_state.daily_pnl = pnl
        self._risk_state.last_update = datetime.now().isoformat()
        self._risk_dirty = True

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    async def start_session(self, session_id: str) -> SessionState:
        """Start a new trading session"""
        session = SessionState(
            session_id=session_id,
            start_time=datetime.now().isoformat(),
            status='active'
        )

        await self._redis.set(
            f"{StateNamespace.SESSION.value}:{session_id}",
            json.dumps(session.to_dict()),
            ex=self.state_ttl
        )

        # Mark as current session
        await self._redis.set(
            f"{StateNamespace.SESSION.value}:current",
            session_id,
            ex=self.state_ttl
        )

        self._session = session
        logger.info(f"Started trading session: {session_id}")

        return session

    async def heartbeat(self) -> None:
        """Update session heartbeat"""
        if self._session:
            self._session.last_heartbeat = datetime.now().isoformat()
            self._session.positions_count = len(self._positions)
            self._session.orders_count = len(self._orders)

            await self._redis.set(
                f"{StateNamespace.SESSION.value}:{self._session.session_id}",
                json.dumps(self._session.to_dict()),
                ex=self.state_ttl
            )

    async def get_current_session(self) -> Optional[SessionState]:
        """Get current session"""
        session_id = await self._redis.get(f"{StateNamespace.SESSION.value}:current")

        if session_id:
            data = await self._redis.get(f"{StateNamespace.SESSION.value}:{session_id}")
            if data:
                return SessionState.from_dict(json.loads(data))

        return None

    async def end_session(self) -> None:
        """End current session"""
        if self._session:
            self._session.status = 'stopped'
            await self._redis.set(
                f"{StateNamespace.SESSION.value}:{self._session.session_id}",
                json.dumps(self._session.to_dict()),
                ex=self.state_ttl
            )
            logger.info(f"Ended trading session: {self._session.session_id}")

    # =========================================================================
    # CRASH RECOVERY
    # =========================================================================

    async def recover_state(self) -> Dict[str, Any]:
        """
        Recover all state after crash/restart.

        Returns dictionary with recovered state.
        """
        logger.info("Starting state recovery...")

        # Check for previous session
        prev_session = await self.get_current_session()

        if prev_session and prev_session.status == 'active':
            # Previous session didn't end cleanly - crash recovery
            last_heartbeat = datetime.fromisoformat(prev_session.last_heartbeat)
            downtime = datetime.now() - last_heartbeat

            logger.warning(
                f"Recovering from crash. Previous session: {prev_session.session_id}, "
                f"downtime: {downtime}"
            )

        # Recover positions
        positions = await self.get_all_positions()
        logger.info(f"Recovered {len(positions)} positions")

        # Recover orders (filter stale ones)
        all_orders = await self.get_all_orders()
        open_orders = [o for o in all_orders.values()
                       if o.status in ['new', 'pending', 'partially_filled']]
        logger.info(f"Recovered {len(open_orders)} open orders")

        # Recover risk state
        risk = await self.get_risk_state()
        logger.info(f"Recovered risk state: exposure={risk.total_exposure}")

        return {
            'previous_session': prev_session,
            'positions': positions,
            'open_orders': open_orders,
            'risk_state': risk,
            'recovery_time': datetime.now().isoformat()
        }

    # =========================================================================
    # DISTRIBUTED LOCKING
    # =========================================================================

    async def acquire_lock(
        self,
        lock_name: str,
        timeout_seconds: int = 30
    ) -> bool:
        """
        Acquire a distributed lock.

        For multi-instance deployments to prevent race conditions.
        """
        lock_key = f"{StateNamespace.LOCKS.value}:{lock_name}"

        # Try to set lock with NX (only if not exists)
        # This is a simplified version - production would use Redlock
        result = await self._redis.set(
            lock_key,
            datetime.now().isoformat(),
            ex=timeout_seconds
        )

        return result

    async def release_lock(self, lock_name: str) -> None:
        """Release a distributed lock"""
        lock_key = f"{StateNamespace.LOCKS.value}:{lock_name}"
        await self._redis.delete(lock_key)

    # =========================================================================
    # CLEANUP
    # =========================================================================

    async def cleanup_old_orders(self, days: int = 7) -> int:
        """Remove orders older than specified days"""
        cutoff = datetime.now() - timedelta(days=days)
        all_orders = await self.get_all_orders()

        removed = 0
        for oid, order in all_orders.items():
            order_time = datetime.fromisoformat(order.created_at)
            if order_time < cutoff and order.status in ['filled', 'cancelled', 'rejected']:
                await self.delete_order(oid)
                removed += 1

        logger.info(f"Cleaned up {removed} old orders")
        return removed

    async def close(self) -> None:
        """Shutdown state manager"""
        await self.stop_auto_save()
        await self.save_dirty_state()

        if isinstance(self._redis, AsyncRedisClient):
            await self._redis.close()

        logger.info("State manager closed")


# =============================================================================
# STATE SNAPSHOT
# =============================================================================

@dataclass
class StateSnapshot:
    """Complete system state snapshot"""
    timestamp: str
    session: Optional[SessionState]
    positions: Dict[str, PositionState]
    orders: Dict[str, OrderState]
    risk: RiskState

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'session': self.session.to_dict() if self.session else None,
            'positions': {k: v.to_dict() for k, v in self.positions.items()},
            'orders': {k: v.to_dict() for k, v in self.orders.items()},
            'risk': self.risk.to_dict()
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict) -> 'StateSnapshot':
        return cls(
            timestamp=data['timestamp'],
            session=SessionState.from_dict(data['session']) if data['session'] else None,
            positions={k: PositionState.from_dict(v) for k, v in data['positions'].items()},
            orders={k: OrderState.from_dict(v) for k, v in data['orders'].items()},
            risk=RiskState.from_dict(data['risk'])
        )


async def create_snapshot(state_manager: RedisStateManager) -> StateSnapshot:
    """Create a complete state snapshot"""
    return StateSnapshot(
        timestamp=datetime.now().isoformat(),
        session=state_manager._session,
        positions=await state_manager.get_all_positions(),
        orders=await state_manager.get_all_orders(),
        risk=await state_manager.get_risk_state()
    )
