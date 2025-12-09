"""
Infrastructure Module
=====================

Core infrastructure components for distributed trading system.
Provides message bus, state management, and service coordination.

Components:
- MessageBus: Redis/ZeroMQ-based pub/sub messaging
- StateStore: Redis-based state persistence
- ServiceRegistry: Service discovery and health monitoring
- Heartbeat: Service health monitoring
- AsyncPool: High-performance connection pooling

Author: AlphaTrade Platform
Version: 3.0.0
License: MIT
"""

from infrastructure.message_bus import (
    MessageBus,
    RedisMessageBus,
    ZeroMQMessageBus,
    InMemoryMessageBus,
    Message,
    MessageType,
    MessagePriority,
)
from infrastructure.state_store import (
    StateStore,
    RedisStateStore,
    StateKey,
)
from infrastructure.service_registry import (
    ServiceRegistry,
    ServiceInfo,
    ServiceStatus,
)
from infrastructure.heartbeat import (
    HeartbeatMonitor,
    HeartbeatConfig,
)
from infrastructure.async_pool import (
    PoolManager,
    RedisPool,
    DatabasePool,
    HTTPPool,
    PoolConfig,
    PoolMetrics,
    CircuitBreaker,
)

__all__ = [
    # Message Bus
    "MessageBus",
    "RedisMessageBus",
    "ZeroMQMessageBus",
    "InMemoryMessageBus",
    "Message",
    "MessageType",
    "MessagePriority",
    # State Store
    "StateStore",
    "RedisStateStore",
    "StateKey",
    # Service Registry
    "ServiceRegistry",
    "ServiceInfo",
    "ServiceStatus",
    # Heartbeat
    "HeartbeatMonitor",
    "HeartbeatConfig",
    # Connection Pools
    "PoolManager",
    "RedisPool",
    "DatabasePool",
    "HTTPPool",
    "PoolConfig",
    "PoolMetrics",
    "CircuitBreaker",
]
