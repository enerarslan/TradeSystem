"""
Async Connection Pool Manager
=============================

High-performance async connection pooling for ultra-low latency trading.
Optimizes database, Redis, and HTTP connections for sub-millisecond operations.

Features:
- Connection pooling with health checks
- Automatic reconnection with backoff
- Connection warming (pre-established connections)
- Latency tracking and metrics
- Circuit breaker pattern
- Graceful degradation

Architecture:
- AsyncPool: Generic async connection pool
- RedisPool: Optimized Redis connection pool
- DatabasePool: PostgreSQL/TimescaleDB pool
- HTTPPool: aiohttp session pool
- PoolManager: Centralized pool management

Author: AlphaTrade Platform
Version: 3.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generic, TypeVar
import statistics

from config.settings import get_logger

logger = get_logger(__name__)

# Type variable for connection types
T = TypeVar("T")


# =============================================================================
# ENUMS
# =============================================================================

class PoolState(str, Enum):
    """Connection pool state."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CLOSED = "closed"


class CircuitState(str, Enum):
    """Circuit breaker state."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PoolConfig:
    """
    Connection pool configuration.

    Attributes:
        min_size: Minimum pool size (pre-warmed connections)
        max_size: Maximum pool size
        max_idle_time: Maximum idle time before connection is closed
        health_check_interval: Interval between health checks
        acquire_timeout: Timeout for acquiring a connection
        connection_timeout: Timeout for establishing connection
        max_retries: Maximum connection retry attempts
        retry_delay: Base delay between retries
        enable_circuit_breaker: Enable circuit breaker pattern
        circuit_failure_threshold: Failures before opening circuit
        circuit_recovery_time: Time before testing recovery
    """
    min_size: int = 5
    max_size: int = 20
    max_idle_time: float = 300.0  # 5 minutes
    health_check_interval: float = 30.0
    acquire_timeout: float = 5.0
    connection_timeout: float = 10.0
    max_retries: int = 3
    retry_delay: float = 0.1
    enable_circuit_breaker: bool = True
    circuit_failure_threshold: int = 5
    circuit_recovery_time: float = 30.0


@dataclass
class PoolMetrics:
    """Pool performance metrics."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    total_acquires: int = 0
    total_releases: int = 0
    total_timeouts: int = 0
    total_errors: int = 0
    avg_acquire_time_ms: float = 0.0
    p99_acquire_time_ms: float = 0.0
    circuit_state: CircuitState = CircuitState.CLOSED
    _acquire_times: deque = field(default_factory=lambda: deque(maxlen=1000))

    def record_acquire_time(self, time_ms: float) -> None:
        """Record an acquire time."""
        self._acquire_times.append(time_ms)
        if self._acquire_times:
            self.avg_acquire_time_ms = statistics.mean(self._acquire_times)
            if len(self._acquire_times) >= 10:
                sorted_times = sorted(self._acquire_times)
                idx = int(len(sorted_times) * 0.99)
                self.p99_acquire_time_ms = sorted_times[min(idx, len(sorted_times) - 1)]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_connections": self.total_connections,
            "active_connections": self.active_connections,
            "idle_connections": self.idle_connections,
            "total_acquires": self.total_acquires,
            "total_releases": self.total_releases,
            "total_timeouts": self.total_timeouts,
            "total_errors": self.total_errors,
            "avg_acquire_time_ms": round(self.avg_acquire_time_ms, 3),
            "p99_acquire_time_ms": round(self.p99_acquire_time_ms, 3),
            "circuit_state": self.circuit_state.value,
        }


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.

    Prevents cascade failures by stopping requests to failing services.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_time: float = 30.0,
    ):
        """Initialize circuit breaker."""
        self._failure_threshold = failure_threshold
        self._recovery_time = recovery_time

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current state."""
        return self._state

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self._state == CircuitState.OPEN

    async def record_success(self) -> None:
        """Record a successful operation."""
        async with self._lock:
            self._failure_count = 0
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                logger.info("Circuit breaker closed - service recovered")

    async def record_failure(self) -> None:
        """Record a failed operation."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._failure_count >= self._failure_threshold:
                if self._state != CircuitState.OPEN:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit breaker opened - {self._failure_count} failures"
                    )

    async def allow_request(self) -> bool:
        """Check if a request should be allowed."""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if recovery time has passed
                if time.monotonic() - self._last_failure_time >= self._recovery_time:
                    self._state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker half-open - testing recovery")
                    return True
                return False

            # Half-open: allow limited requests
            return True

    def reset(self) -> None:
        """Reset circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0


# =============================================================================
# CONNECTION WRAPPER
# =============================================================================

@dataclass
class PooledConnection(Generic[T]):
    """
    Wrapper for pooled connections.

    Tracks connection metadata and health.
    """
    connection: T
    pool: "AsyncPool[T]"
    created_at: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)
    use_count: int = 0
    healthy: bool = True

    @property
    def age(self) -> float:
        """Get connection age in seconds."""
        return time.monotonic() - self.created_at

    @property
    def idle_time(self) -> float:
        """Get idle time in seconds."""
        return time.monotonic() - self.last_used

    def mark_used(self) -> None:
        """Mark connection as used."""
        self.last_used = time.monotonic()
        self.use_count += 1

    async def release(self) -> None:
        """Release connection back to pool."""
        await self.pool.release(self)

    async def __aenter__(self) -> T:
        """Context manager entry."""
        return self.connection

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if exc_type is not None:
            self.healthy = False
        await self.release()


# =============================================================================
# ABSTRACT POOL
# =============================================================================

class AsyncPool(ABC, Generic[T]):
    """
    Abstract async connection pool.

    Provides base implementation for connection pooling with:
    - Async connection management
    - Health checking
    - Circuit breaker integration
    - Metrics tracking
    """

    def __init__(self, config: PoolConfig | None = None):
        """Initialize pool."""
        self.config = config or PoolConfig()
        self._state = PoolState.INITIALIZING

        # Connection storage
        self._idle: asyncio.Queue[PooledConnection[T]] = asyncio.Queue()
        self._active: set[PooledConnection[T]] = set()
        self._all_connections: set[PooledConnection[T]] = set()

        # Synchronization
        self._lock = asyncio.Lock()
        self._connection_semaphore = asyncio.Semaphore(self.config.max_size)

        # Circuit breaker
        self._circuit_breaker: CircuitBreaker | None = None
        if self.config.enable_circuit_breaker:
            self._circuit_breaker = CircuitBreaker(
                failure_threshold=self.config.circuit_failure_threshold,
                recovery_time=self.config.circuit_recovery_time,
            )

        # Metrics
        self._metrics = PoolMetrics()

        # Background tasks
        self._health_check_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None

    @property
    def state(self) -> PoolState:
        """Get pool state."""
        return self._state

    @property
    def metrics(self) -> PoolMetrics:
        """Get pool metrics."""
        self._metrics.total_connections = len(self._all_connections)
        self._metrics.active_connections = len(self._active)
        self._metrics.idle_connections = self._idle.qsize()
        if self._circuit_breaker:
            self._metrics.circuit_state = self._circuit_breaker.state
        return self._metrics

    # =========================================================================
    # ABSTRACT METHODS
    # =========================================================================

    @abstractmethod
    async def _create_connection(self) -> T:
        """Create a new connection."""
        pass

    @abstractmethod
    async def _close_connection(self, conn: T) -> None:
        """Close a connection."""
        pass

    @abstractmethod
    async def _health_check(self, conn: T) -> bool:
        """Check if connection is healthy."""
        pass

    # =========================================================================
    # POOL OPERATIONS
    # =========================================================================

    async def initialize(self) -> None:
        """Initialize pool with minimum connections."""
        logger.info(f"Initializing pool with {self.config.min_size} connections")

        try:
            # Pre-warm connections
            tasks = [self._add_connection() for _ in range(self.config.min_size)]
            await asyncio.gather(*tasks)

            # Start background tasks
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            self._state = PoolState.HEALTHY
            logger.info(f"Pool initialized: {len(self._all_connections)} connections")

        except Exception as e:
            self._state = PoolState.UNHEALTHY
            logger.error(f"Pool initialization failed: {e}")
            raise

    async def close(self) -> None:
        """Close pool and all connections."""
        logger.info("Closing connection pool")
        self._state = PoolState.CLOSED

        # Cancel background tasks
        for task in [self._health_check_task, self._cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close all connections
        async with self._lock:
            for pc in self._all_connections:
                try:
                    await self._close_connection(pc.connection)
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")

            self._all_connections.clear()
            self._active.clear()

            # Drain idle queue
            while not self._idle.empty():
                try:
                    self._idle.get_nowait()
                except asyncio.QueueEmpty:
                    break

        logger.info("Pool closed")

    async def acquire(self) -> PooledConnection[T]:
        """
        Acquire a connection from the pool.

        Returns:
            PooledConnection wrapper

        Raises:
            TimeoutError: If acquire times out
            RuntimeError: If pool is closed or circuit is open
        """
        if self._state == PoolState.CLOSED:
            raise RuntimeError("Pool is closed")

        # Check circuit breaker
        if self._circuit_breaker and not await self._circuit_breaker.allow_request():
            raise RuntimeError("Circuit breaker is open")

        start_time = time.monotonic()

        try:
            # Try to get idle connection
            async with asyncio.timeout(self.config.acquire_timeout):
                while True:
                    try:
                        # Try to get from idle queue (non-blocking first)
                        try:
                            pc = self._idle.get_nowait()

                            # Check if connection is still healthy
                            if pc.healthy and pc.idle_time < self.config.max_idle_time:
                                pc.mark_used()
                                self._active.add(pc)

                                # Record metrics
                                acquire_time = (time.monotonic() - start_time) * 1000
                                self._metrics.record_acquire_time(acquire_time)
                                self._metrics.total_acquires += 1

                                if self._circuit_breaker:
                                    await self._circuit_breaker.record_success()

                                return pc
                            else:
                                # Connection is unhealthy or too old
                                await self._remove_connection(pc)

                        except asyncio.QueueEmpty:
                            pass

                        # Try to create new connection if below max
                        if len(self._all_connections) < self.config.max_size:
                            if await self._add_connection():
                                continue

                        # Wait for connection to be released
                        try:
                            pc = await asyncio.wait_for(
                                self._idle.get(),
                                timeout=0.1
                            )
                            if pc.healthy:
                                pc.mark_used()
                                self._active.add(pc)

                                acquire_time = (time.monotonic() - start_time) * 1000
                                self._metrics.record_acquire_time(acquire_time)
                                self._metrics.total_acquires += 1

                                if self._circuit_breaker:
                                    await self._circuit_breaker.record_success()

                                return pc
                            else:
                                await self._remove_connection(pc)

                        except asyncio.TimeoutError:
                            continue

                    except Exception as e:
                        self._metrics.total_errors += 1
                        if self._circuit_breaker:
                            await self._circuit_breaker.record_failure()
                        raise

        except asyncio.TimeoutError:
            self._metrics.total_timeouts += 1
            raise TimeoutError(
                f"Failed to acquire connection within {self.config.acquire_timeout}s"
            )

    async def release(self, pc: PooledConnection[T]) -> None:
        """Release a connection back to the pool."""
        self._metrics.total_releases += 1
        self._active.discard(pc)

        if pc.healthy and self._state != PoolState.CLOSED:
            await self._idle.put(pc)
        else:
            await self._remove_connection(pc)

    async def _add_connection(self) -> bool:
        """Add a new connection to the pool."""
        if len(self._all_connections) >= self.config.max_size:
            return False

        try:
            conn = await asyncio.wait_for(
                self._create_connection(),
                timeout=self.config.connection_timeout
            )

            pc = PooledConnection(connection=conn, pool=self)
            self._all_connections.add(pc)
            await self._idle.put(pc)

            return True

        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            return False

    async def _remove_connection(self, pc: PooledConnection[T]) -> None:
        """Remove a connection from the pool."""
        self._all_connections.discard(pc)
        self._active.discard(pc)

        try:
            await self._close_connection(pc.connection)
        except Exception as e:
            logger.error(f"Error closing connection: {e}")

    # =========================================================================
    # BACKGROUND TASKS
    # =========================================================================

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self._state != PoolState.CLOSED:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                # Check all idle connections
                connections_to_check = []
                while not self._idle.empty():
                    try:
                        pc = self._idle.get_nowait()
                        connections_to_check.append(pc)
                    except asyncio.QueueEmpty:
                        break

                healthy_count = 0
                for pc in connections_to_check:
                    try:
                        if await self._health_check(pc.connection):
                            pc.healthy = True
                            healthy_count += 1
                            await self._idle.put(pc)
                        else:
                            pc.healthy = False
                            await self._remove_connection(pc)
                    except Exception:
                        pc.healthy = False
                        await self._remove_connection(pc)

                # Update pool state
                total = len(self._all_connections)
                if total == 0:
                    self._state = PoolState.UNHEALTHY
                elif healthy_count < total * 0.5:
                    self._state = PoolState.DEGRADED
                else:
                    self._state = PoolState.HEALTHY

                # Ensure minimum connections
                while len(self._all_connections) < self.config.min_size:
                    if not await self._add_connection():
                        break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop for idle connections."""
        while self._state != PoolState.CLOSED:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Clean up excess idle connections
                while (
                    len(self._all_connections) > self.config.min_size
                    and not self._idle.empty()
                ):
                    try:
                        pc = self._idle.get_nowait()
                        if pc.idle_time > self.config.max_idle_time:
                            await self._remove_connection(pc)
                        else:
                            await self._idle.put(pc)
                            break
                    except asyncio.QueueEmpty:
                        break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")


# =============================================================================
# REDIS POOL
# =============================================================================

class RedisPool(AsyncPool):
    """
    Optimized Redis connection pool.

    Provides ultra-low latency Redis connections for:
    - State storage
    - Message bus
    - Feature caching
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        config: PoolConfig | None = None,
    ):
        """Initialize Redis pool."""
        super().__init__(config)
        self.url = url
        self._redis_module: Any = None

    async def _create_connection(self):
        """Create Redis connection."""
        if self._redis_module is None:
            try:
                import redis.asyncio as aioredis
                self._redis_module = aioredis
            except ImportError:
                raise ImportError("redis package required: pip install redis")

        return await self._redis_module.from_url(
            self.url,
            decode_responses=True,
            socket_connect_timeout=self.config.connection_timeout,
        )

    async def _close_connection(self, conn) -> None:
        """Close Redis connection."""
        await conn.close()

    async def _health_check(self, conn) -> bool:
        """Check Redis connection health."""
        try:
            await conn.ping()
            return True
        except Exception:
            return False


# =============================================================================
# DATABASE POOL
# =============================================================================

class DatabasePool(AsyncPool):
    """
    PostgreSQL/TimescaleDB connection pool.

    Optimized for:
    - Market data queries
    - Feature lookups
    - Order persistence
    """

    def __init__(
        self,
        dsn: str = "postgresql://localhost/market_data",
        config: PoolConfig | None = None,
    ):
        """Initialize database pool."""
        super().__init__(config)
        self.dsn = dsn
        self._asyncpg_module: Any = None

    async def _create_connection(self):
        """Create database connection."""
        if self._asyncpg_module is None:
            try:
                import asyncpg
                self._asyncpg_module = asyncpg
            except ImportError:
                raise ImportError("asyncpg package required: pip install asyncpg")

        return await self._asyncpg_module.connect(
            self.dsn,
            timeout=self.config.connection_timeout,
        )

    async def _close_connection(self, conn) -> None:
        """Close database connection."""
        await conn.close()

    async def _health_check(self, conn) -> bool:
        """Check database connection health."""
        try:
            await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False


# =============================================================================
# HTTP POOL
# =============================================================================

class HTTPPool(AsyncPool):
    """
    aiohttp session pool for external API calls.

    Optimized for:
    - Broker API calls
    - Market data APIs
    - Webhook notifications
    """

    def __init__(
        self,
        base_url: str | None = None,
        config: PoolConfig | None = None,
    ):
        """Initialize HTTP pool."""
        super().__init__(config)
        self.base_url = base_url
        self._aiohttp_module: Any = None
        self._connector: Any = None

    async def initialize(self) -> None:
        """Initialize HTTP pool with shared connector."""
        try:
            import aiohttp
            self._aiohttp_module = aiohttp

            # Create shared TCP connector
            self._connector = aiohttp.TCPConnector(
                limit=self.config.max_size,
                limit_per_host=self.config.max_size,
                keepalive_timeout=30,
                enable_cleanup_closed=True,
            )
        except ImportError:
            raise ImportError("aiohttp package required: pip install aiohttp")

        await super().initialize()

    async def _create_connection(self):
        """Create aiohttp session."""
        timeout = self._aiohttp_module.ClientTimeout(
            total=self.config.connection_timeout
        )

        return self._aiohttp_module.ClientSession(
            base_url=self.base_url,
            connector=self._connector,
            timeout=timeout,
            connector_owner=False,  # Don't close connector with session
        )

    async def _close_connection(self, conn) -> None:
        """Close aiohttp session."""
        await conn.close()

    async def _health_check(self, conn) -> bool:
        """Check HTTP session health."""
        return not conn.closed

    async def close(self) -> None:
        """Close pool and connector."""
        await super().close()
        if self._connector:
            await self._connector.close()


# =============================================================================
# POOL MANAGER
# =============================================================================

class PoolManager:
    """
    Centralized connection pool manager.

    Manages all connection pools for the trading system:
    - Redis pools (state, pubsub, cache)
    - Database pools (TimescaleDB)
    - HTTP pools (broker APIs)

    Example:
        manager = PoolManager()
        await manager.initialize()

        # Get connections
        async with manager.redis.acquire() as conn:
            await conn.get("key")

        async with manager.database.acquire() as conn:
            await conn.fetch("SELECT * FROM bars")
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        database_dsn: str = "postgresql://localhost/market_data",
        broker_url: str | None = None,
        pool_config: PoolConfig | None = None,
    ):
        """Initialize pool manager."""
        self._config = pool_config or PoolConfig()

        # Create pools
        self.redis = RedisPool(redis_url, self._config)
        self.database = DatabasePool(database_dsn, self._config)
        self.http = HTTPPool(broker_url, self._config) if broker_url else None

        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if pools are initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize all pools."""
        logger.info("Initializing pool manager")

        try:
            # Initialize pools in parallel
            tasks = [self.redis.initialize(), self.database.initialize()]
            if self.http:
                tasks.append(self.http.initialize())

            await asyncio.gather(*tasks)

            self._initialized = True
            logger.info("Pool manager initialized")

        except Exception as e:
            logger.error(f"Pool manager initialization failed: {e}")
            raise

    async def close(self) -> None:
        """Close all pools."""
        logger.info("Closing pool manager")

        tasks = [self.redis.close(), self.database.close()]
        if self.http:
            tasks.append(self.http.close())

        await asyncio.gather(*tasks, return_exceptions=True)

        self._initialized = False
        logger.info("Pool manager closed")

    def get_metrics(self) -> dict[str, Any]:
        """Get metrics from all pools."""
        metrics = {
            "redis": self.redis.metrics.to_dict(),
            "database": self.database.metrics.to_dict(),
        }
        if self.http:
            metrics["http"] = self.http.metrics.to_dict()
        return metrics

    async def health_check(self) -> dict[str, Any]:
        """Check health of all pools."""
        return {
            "redis": self.redis.state.value,
            "database": self.database.state.value,
            "http": self.http.state.value if self.http else "not_configured",
            "overall": (
                "healthy" if all(
                    p.state == PoolState.HEALTHY
                    for p in [self.redis, self.database]
                    if p is not None
                ) else "degraded"
            ),
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "PoolState",
    "CircuitState",
    "PoolConfig",
    "PoolMetrics",
    "CircuitBreaker",
    "PooledConnection",
    "AsyncPool",
    "RedisPool",
    "DatabasePool",
    "HTTPPool",
    "PoolManager",
]
