"""
Message Bus Module
==================

Distributed message bus for decoupled microservices architecture.
Supports Redis Pub/Sub and ZeroMQ for low-latency communication.

This is the backbone of the institutional trading system - if Strategy Engine
crashes, the OMS continues to manage open risk via the message bus.

Features:
- Multiple backend support (Redis, ZeroMQ, InMemory)
- Async-first design for high throughput
- Message serialization with msgpack for speed
- Dead letter queue for failed messages
- Message acknowledgment and retry
- Priority queues for critical messages

Author: AlphaTrade Platform
Version: 3.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Awaitable, Callable, TypeVar, Generic
from collections import defaultdict
import threading
import queue

try:
    import redis.asyncio as aioredis
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import zmq
    import zmq.asyncio
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

from config.settings import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class MessageType(str, Enum):
    """Message types for trading system."""
    # Market Data
    MARKET_DATA = "market_data"
    TICK_DATA = "tick_data"
    ORDERBOOK = "orderbook"

    # Trading Signals
    SIGNAL = "signal"
    SIGNAL_APPROVED = "signal_approved"
    SIGNAL_REJECTED = "signal_rejected"

    # Orders
    ORDER_NEW = "order_new"
    ORDER_CANCEL = "order_cancel"
    ORDER_MODIFY = "order_modify"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIAL = "order_partial"
    ORDER_REJECTED = "order_rejected"
    ORDER_CANCELLED = "order_cancelled"

    # Risk
    RISK_CHECK = "risk_check"
    RISK_BREACH = "risk_breach"
    RISK_UPDATE = "risk_update"

    # System
    HEARTBEAT = "heartbeat"
    SHUTDOWN = "shutdown"
    RESTART = "restart"
    KILL_SWITCH = "kill_switch"

    # State
    STATE_UPDATE = "state_update"
    POSITION_UPDATE = "position_update"
    PNL_UPDATE = "pnl_update"


class MessagePriority(IntEnum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3  # Kill switch, risk breach


class Channel(str, Enum):
    """Pub/Sub channels."""
    MARKET_DATA = "channel:market_data"
    SIGNALS = "channel:signals"
    ORDERS = "channel:orders"
    FILLS = "channel:fills"
    RISK = "channel:risk"
    SYSTEM = "channel:system"
    HEARTBEAT = "channel:heartbeat"
    DEAD_LETTER = "channel:dead_letter"


# =============================================================================
# MESSAGE DATA CLASS
# =============================================================================

@dataclass
class Message:
    """
    Standardized message format for inter-service communication.

    Attributes:
        id: Unique message identifier
        type: Message type enum
        channel: Target channel
        payload: Message data
        priority: Message priority
        source: Source service name
        target: Target service (optional, for direct messages)
        timestamp: Message creation time
        correlation_id: For request-response patterns
        retry_count: Number of delivery attempts
        ttl: Time to live in seconds
    """
    type: MessageType
    channel: Channel
    payload: dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: MessagePriority = MessagePriority.NORMAL
    source: str = "unknown"
    target: str | None = None
    timestamp: float = field(default_factory=time.time)
    correlation_id: str | None = None
    retry_count: int = 0
    ttl: int = 300  # 5 minutes default

    def to_dict(self) -> dict[str, Any]:
        """Serialize message to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "channel": self.channel.value,
            "payload": self.payload,
            "priority": int(self.priority),
            "source": self.source,
            "target": self.target,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "retry_count": self.retry_count,
            "ttl": self.ttl,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Deserialize message from dictionary."""
        return cls(
            id=data["id"],
            type=MessageType(data["type"]),
            channel=Channel(data["channel"]),
            payload=data["payload"],
            priority=MessagePriority(data["priority"]),
            source=data["source"],
            target=data.get("target"),
            timestamp=data["timestamp"],
            correlation_id=data.get("correlation_id"),
            retry_count=data.get("retry_count", 0),
            ttl=data.get("ttl", 300),
        )

    def serialize(self) -> bytes:
        """Serialize message to bytes (msgpack for speed)."""
        if MSGPACK_AVAILABLE:
            return msgpack.packb(self.to_dict(), use_bin_type=True)
        return json.dumps(self.to_dict()).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> "Message":
        """Deserialize message from bytes."""
        if MSGPACK_AVAILABLE:
            return cls.from_dict(msgpack.unpackb(data, raw=False))
        return cls.from_dict(json.loads(data.decode("utf-8")))

    def is_expired(self) -> bool:
        """Check if message has expired."""
        return time.time() - self.timestamp > self.ttl


# =============================================================================
# MESSAGE HANDLER TYPE
# =============================================================================

MessageHandler = Callable[[Message], Awaitable[None]]
SyncMessageHandler = Callable[[Message], None]


# =============================================================================
# ABSTRACT MESSAGE BUS
# =============================================================================

class MessageBus(ABC):
    """
    Abstract base class for message bus implementations.

    The message bus is the core communication layer that decouples
    services in the trading system. This allows:

    1. Strategy Engine to crash without affecting OMS
    2. Risk Engine to intercept all orders
    3. Independent scaling of components
    4. Easy testing with InMemory implementation

    Example:
        bus = RedisMessageBus(redis_url="redis://localhost:6379")
        await bus.connect()

        # Subscribe to signals
        await bus.subscribe(Channel.SIGNALS, handle_signal)

        # Publish a signal
        msg = Message(
            type=MessageType.SIGNAL,
            channel=Channel.SIGNALS,
            payload={"symbol": "AAPL", "direction": 1}
        )
        await bus.publish(msg)
    """

    def __init__(self, service_name: str = "unknown"):
        """
        Initialize message bus.

        Args:
            service_name: Name of the service using this bus
        """
        self.service_name = service_name
        self._handlers: dict[Channel, list[MessageHandler]] = defaultdict(list)
        self._sync_handlers: dict[Channel, list[SyncMessageHandler]] = defaultdict(list)
        self._running = False
        self._connected = False
        self._message_count = 0
        self._error_count = 0

    @property
    def is_connected(self) -> bool:
        """Check if bus is connected."""
        return self._connected

    @property
    def is_running(self) -> bool:
        """Check if bus is processing messages."""
        return self._running

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the message bus backend."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the message bus backend."""
        pass

    @abstractmethod
    async def publish(self, message: Message) -> bool:
        """
        Publish a message to a channel.

        Args:
            message: Message to publish

        Returns:
            True if published successfully
        """
        pass

    @abstractmethod
    async def subscribe(
        self,
        channel: Channel,
        handler: MessageHandler,
    ) -> None:
        """
        Subscribe to a channel with an async handler.

        Args:
            channel: Channel to subscribe to
            handler: Async callback function
        """
        pass

    @abstractmethod
    async def unsubscribe(self, channel: Channel) -> None:
        """
        Unsubscribe from a channel.

        Args:
            channel: Channel to unsubscribe from
        """
        pass

    async def publish_signal(
        self,
        symbol: str,
        direction: int,
        strength: float,
        price: float,
        strategy: str,
        **kwargs: Any,
    ) -> bool:
        """
        Convenience method to publish a trading signal.

        Args:
            symbol: Trading symbol
            direction: Signal direction (1=long, -1=short)
            strength: Signal strength (0-1)
            price: Current price
            strategy: Strategy name
            **kwargs: Additional signal data

        Returns:
            True if published successfully
        """
        payload = {
            "symbol": symbol,
            "direction": direction,
            "strength": strength,
            "price": price,
            "strategy": strategy,
            **kwargs,
        }

        message = Message(
            type=MessageType.SIGNAL,
            channel=Channel.SIGNALS,
            payload=payload,
            source=self.service_name,
            priority=MessagePriority.HIGH,
        )

        return await self.publish(message)

    async def publish_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: float | None = None,
        **kwargs: Any,
    ) -> bool:
        """
        Convenience method to publish an order.

        Args:
            order_id: Order identifier
            symbol: Trading symbol
            side: Order side (buy/sell)
            quantity: Order quantity
            order_type: Order type (market/limit)
            price: Limit price (optional)
            **kwargs: Additional order data

        Returns:
            True if published successfully
        """
        payload = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_type": order_type,
            "price": price,
            **kwargs,
        }

        message = Message(
            type=MessageType.ORDER_NEW,
            channel=Channel.ORDERS,
            payload=payload,
            source=self.service_name,
            priority=MessagePriority.CRITICAL,
        )

        return await self.publish(message)

    async def publish_heartbeat(self) -> bool:
        """Publish a heartbeat message."""
        message = Message(
            type=MessageType.HEARTBEAT,
            channel=Channel.HEARTBEAT,
            payload={
                "service": self.service_name,
                "timestamp": time.time(),
                "message_count": self._message_count,
                "error_count": self._error_count,
            },
            source=self.service_name,
            priority=MessagePriority.LOW,
            ttl=60,  # Heartbeats expire quickly
        )

        return await self.publish(message)

    async def publish_kill_switch(self, reason: str) -> bool:
        """
        Publish emergency kill switch message.

        This message has CRITICAL priority and should trigger
        immediate position closure and order cancellation.

        Args:
            reason: Reason for kill switch activation

        Returns:
            True if published successfully
        """
        message = Message(
            type=MessageType.KILL_SWITCH,
            channel=Channel.SYSTEM,
            payload={
                "reason": reason,
                "triggered_by": self.service_name,
                "timestamp": time.time(),
            },
            source=self.service_name,
            priority=MessagePriority.CRITICAL,
        )

        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
        return await self.publish(message)

    def get_stats(self) -> dict[str, Any]:
        """Get message bus statistics."""
        return {
            "service_name": self.service_name,
            "connected": self._connected,
            "running": self._running,
            "message_count": self._message_count,
            "error_count": self._error_count,
            "subscribed_channels": list(self._handlers.keys()),
        }


# =============================================================================
# REDIS MESSAGE BUS IMPLEMENTATION
# =============================================================================

class RedisMessageBus(MessageBus):
    """
    Redis Pub/Sub based message bus.

    Ideal for:
    - Medium latency (1-5ms)
    - High reliability
    - Easy setup and maintenance
    - Built-in persistence options

    Features:
    - Automatic reconnection
    - Dead letter queue for failed messages
    - Priority queue support via sorted sets
    - Message acknowledgment

    Example:
        bus = RedisMessageBus(
            redis_url="redis://localhost:6379",
            service_name="strategy_engine"
        )
        await bus.connect()
        await bus.subscribe(Channel.MARKET_DATA, handle_data)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        service_name: str = "unknown",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        pool_size: int = 10,
    ):
        """
        Initialize Redis message bus.

        Args:
            redis_url: Redis connection URL
            service_name: Name of the service
            max_retries: Maximum message delivery retries
            retry_delay: Delay between retries in seconds
            pool_size: Connection pool size
        """
        super().__init__(service_name)

        if not REDIS_AVAILABLE:
            raise ImportError("redis package required: pip install redis")

        self.redis_url = redis_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.pool_size = pool_size

        self._client: aioredis.Redis | None = None
        self._pubsub: aioredis.client.PubSub | None = None
        self._listener_task: asyncio.Task | None = None
        self._subscribed_channels: set[str] = set()

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            self._client = aioredis.from_url(
                self.redis_url,
                max_connections=self.pool_size,
                decode_responses=False,
            )

            # Test connection
            await self._client.ping()

            self._pubsub = self._client.pubsub()
            self._connected = True
            self._running = True

            logger.info(f"Connected to Redis at {self.redis_url}")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        self._running = False

        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        if self._pubsub:
            await self._pubsub.close()

        if self._client:
            await self._client.close()

        self._connected = False
        logger.info("Disconnected from Redis")

    async def publish(self, message: Message) -> bool:
        """Publish message to Redis channel."""
        if not self._connected or not self._client:
            logger.error("Cannot publish: not connected to Redis")
            return False

        try:
            message.source = self.service_name
            data = message.serialize()

            # For critical messages, also add to priority queue
            if message.priority >= MessagePriority.HIGH:
                queue_key = f"queue:{message.channel.value}:priority"
                await self._client.zadd(
                    queue_key,
                    {data: -int(message.priority)},
                )

            # Publish to channel
            await self._client.publish(message.channel.value, data)

            self._message_count += 1
            logger.debug(f"Published {message.type.value} to {message.channel.value}")

            return True

        except Exception as e:
            self._error_count += 1
            logger.error(f"Failed to publish message: {e}")

            # Send to dead letter queue
            await self._send_to_dlq(message, str(e))
            return False

    async def subscribe(
        self,
        channel: Channel,
        handler: MessageHandler,
    ) -> None:
        """Subscribe to a Redis channel."""
        if not self._connected or not self._pubsub:
            raise RuntimeError("Not connected to Redis")

        self._handlers[channel].append(handler)

        if channel.value not in self._subscribed_channels:
            await self._pubsub.subscribe(channel.value)
            self._subscribed_channels.add(channel.value)
            logger.info(f"Subscribed to {channel.value}")

        # Start listener if not running
        if not self._listener_task or self._listener_task.done():
            self._listener_task = asyncio.create_task(self._listen())

    async def unsubscribe(self, channel: Channel) -> None:
        """Unsubscribe from a Redis channel."""
        if channel.value in self._subscribed_channels:
            await self._pubsub.unsubscribe(channel.value)
            self._subscribed_channels.remove(channel.value)
            self._handlers.pop(channel, None)
            logger.info(f"Unsubscribed from {channel.value}")

    async def _listen(self) -> None:
        """Listen for messages on subscribed channels."""
        logger.info("Starting Redis message listener")

        while self._running and self._pubsub:
            try:
                async for msg in self._pubsub.listen():
                    if not self._running:
                        break

                    if msg["type"] == "message":
                        await self._handle_message(msg)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._error_count += 1
                logger.error(f"Error in message listener: {e}")
                await asyncio.sleep(1)

    async def _handle_message(self, raw_msg: dict) -> None:
        """Handle incoming message."""
        try:
            message = Message.deserialize(raw_msg["data"])

            # Check expiration
            if message.is_expired():
                logger.warning(f"Dropped expired message: {message.id}")
                return

            # Find channel enum from string
            channel = Channel(raw_msg["channel"].decode() if isinstance(raw_msg["channel"], bytes) else raw_msg["channel"])

            # Call handlers
            for handler in self._handlers.get(channel, []):
                try:
                    await handler(message)
                except Exception as e:
                    self._error_count += 1
                    logger.error(f"Handler error for {message.id}: {e}")

                    # Retry logic
                    if message.retry_count < self.max_retries:
                        message.retry_count += 1
                        await asyncio.sleep(self.retry_delay)
                        await self.publish(message)
                    else:
                        await self._send_to_dlq(message, str(e))

        except Exception as e:
            self._error_count += 1
            logger.error(f"Failed to handle message: {e}")

    async def _send_to_dlq(self, message: Message, error: str) -> None:
        """Send failed message to dead letter queue."""
        if not self._client:
            return

        dlq_key = f"dlq:{message.channel.value}"
        dlq_data = {
            "message": message.to_dict(),
            "error": error,
            "failed_at": time.time(),
        }

        await self._client.lpush(dlq_key, json.dumps(dlq_data))
        logger.warning(f"Message {message.id} sent to DLQ")

    async def get_dlq_messages(
        self,
        channel: Channel,
        count: int = 10,
    ) -> list[dict]:
        """Retrieve messages from dead letter queue."""
        if not self._client:
            return []

        dlq_key = f"dlq:{channel.value}"
        messages = await self._client.lrange(dlq_key, 0, count - 1)

        return [json.loads(m) for m in messages]

    async def reprocess_dlq(self, channel: Channel) -> int:
        """Reprocess messages from dead letter queue."""
        if not self._client:
            return 0

        dlq_key = f"dlq:{channel.value}"
        reprocessed = 0

        while True:
            data = await self._client.rpop(dlq_key)
            if not data:
                break

            dlq_data = json.loads(data)
            message = Message.from_dict(dlq_data["message"])
            message.retry_count = 0  # Reset retry count

            if await self.publish(message):
                reprocessed += 1

        logger.info(f"Reprocessed {reprocessed} messages from DLQ")
        return reprocessed


# =============================================================================
# ZEROMQ MESSAGE BUS IMPLEMENTATION
# =============================================================================

class ZeroMQMessageBus(MessageBus):
    """
    ZeroMQ based message bus for ultra-low latency.

    Ideal for:
    - Sub-millisecond latency requirements
    - High-frequency trading scenarios
    - Direct service-to-service communication

    Uses PUB/SUB pattern with XPUB/XSUB proxy for scalability.

    Example:
        bus = ZeroMQMessageBus(
            pub_address="tcp://localhost:5555",
            sub_address="tcp://localhost:5556",
            service_name="strategy_engine"
        )
        await bus.connect()
    """

    def __init__(
        self,
        pub_address: str = "tcp://localhost:5555",
        sub_address: str = "tcp://localhost:5556",
        service_name: str = "unknown",
        hwm: int = 10000,  # High water mark
    ):
        """
        Initialize ZeroMQ message bus.

        Args:
            pub_address: Publisher socket address
            sub_address: Subscriber socket address
            service_name: Name of the service
            hwm: High water mark (message queue size)
        """
        super().__init__(service_name)

        if not ZMQ_AVAILABLE:
            raise ImportError("pyzmq package required: pip install pyzmq")

        self.pub_address = pub_address
        self.sub_address = sub_address
        self.hwm = hwm

        self._context: zmq.asyncio.Context | None = None
        self._pub_socket: zmq.asyncio.Socket | None = None
        self._sub_socket: zmq.asyncio.Socket | None = None
        self._listener_task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Connect to ZeroMQ."""
        try:
            self._context = zmq.asyncio.Context()

            # Publisher socket
            self._pub_socket = self._context.socket(zmq.PUB)
            self._pub_socket.setsockopt(zmq.SNDHWM, self.hwm)
            self._pub_socket.connect(self.pub_address)

            # Subscriber socket
            self._sub_socket = self._context.socket(zmq.SUB)
            self._sub_socket.setsockopt(zmq.RCVHWM, self.hwm)
            self._sub_socket.connect(self.sub_address)

            self._connected = True
            self._running = True

            # Small delay for socket connection
            await asyncio.sleep(0.1)

            logger.info(f"Connected to ZeroMQ pub={self.pub_address} sub={self.sub_address}")

        except Exception as e:
            logger.error(f"Failed to connect to ZeroMQ: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from ZeroMQ."""
        self._running = False

        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        if self._pub_socket:
            self._pub_socket.close()

        if self._sub_socket:
            self._sub_socket.close()

        if self._context:
            self._context.term()

        self._connected = False
        logger.info("Disconnected from ZeroMQ")

    async def publish(self, message: Message) -> bool:
        """Publish message via ZeroMQ."""
        if not self._connected or not self._pub_socket:
            logger.error("Cannot publish: not connected to ZeroMQ")
            return False

        try:
            message.source = self.service_name

            # Topic is the channel name
            topic = message.channel.value.encode()
            data = message.serialize()

            # Send multipart message: [topic, data]
            await self._pub_socket.send_multipart([topic, data])

            self._message_count += 1
            logger.debug(f"Published {message.type.value} to {message.channel.value}")

            return True

        except Exception as e:
            self._error_count += 1
            logger.error(f"Failed to publish message: {e}")
            return False

    async def subscribe(
        self,
        channel: Channel,
        handler: MessageHandler,
    ) -> None:
        """Subscribe to a ZeroMQ topic."""
        if not self._connected or not self._sub_socket:
            raise RuntimeError("Not connected to ZeroMQ")

        self._handlers[channel].append(handler)

        # Subscribe to topic
        self._sub_socket.setsockopt(zmq.SUBSCRIBE, channel.value.encode())
        logger.info(f"Subscribed to {channel.value}")

        # Start listener if not running
        if not self._listener_task or self._listener_task.done():
            self._listener_task = asyncio.create_task(self._listen())

    async def unsubscribe(self, channel: Channel) -> None:
        """Unsubscribe from a ZeroMQ topic."""
        if self._sub_socket:
            self._sub_socket.setsockopt(zmq.UNSUBSCRIBE, channel.value.encode())
            self._handlers.pop(channel, None)
            logger.info(f"Unsubscribed from {channel.value}")

    async def _listen(self) -> None:
        """Listen for messages."""
        logger.info("Starting ZeroMQ message listener")

        while self._running and self._sub_socket:
            try:
                # Receive multipart message
                topic, data = await self._sub_socket.recv_multipart()

                message = Message.deserialize(data)
                channel = Channel(topic.decode())

                # Check expiration
                if message.is_expired():
                    continue

                # Call handlers
                for handler in self._handlers.get(channel, []):
                    try:
                        await handler(message)
                    except Exception as e:
                        self._error_count += 1
                        logger.error(f"Handler error: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._error_count += 1
                logger.error(f"Error in message listener: {e}")


# =============================================================================
# IN-MEMORY MESSAGE BUS (FOR TESTING)
# =============================================================================

class InMemoryMessageBus(MessageBus):
    """
    In-memory message bus for testing and development.

    Provides the same interface as Redis/ZeroMQ but runs entirely
    in memory. Useful for:
    - Unit testing
    - Local development
    - Backtesting without infrastructure

    Example:
        bus = InMemoryMessageBus(service_name="test_service")
        await bus.connect()
        await bus.subscribe(Channel.SIGNALS, mock_handler)
    """

    def __init__(self, service_name: str = "unknown"):
        """Initialize in-memory message bus."""
        super().__init__(service_name)
        self._queues: dict[Channel, asyncio.Queue] = {}
        self._listener_tasks: dict[Channel, asyncio.Task] = {}

    async def connect(self) -> None:
        """Connect (no-op for in-memory)."""
        self._connected = True
        self._running = True
        logger.info("In-memory message bus connected")

    async def disconnect(self) -> None:
        """Disconnect and cleanup."""
        self._running = False

        for task in self._listener_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._queues.clear()
        self._listener_tasks.clear()
        self._connected = False
        logger.info("In-memory message bus disconnected")

    async def publish(self, message: Message) -> bool:
        """Publish message to in-memory queue."""
        if not self._connected:
            return False

        message.source = self.service_name

        if message.channel in self._queues:
            await self._queues[message.channel].put(message)
            self._message_count += 1
            return True

        return False

    async def subscribe(
        self,
        channel: Channel,
        handler: MessageHandler,
    ) -> None:
        """Subscribe to channel."""
        self._handlers[channel].append(handler)

        if channel not in self._queues:
            self._queues[channel] = asyncio.Queue()
            self._listener_tasks[channel] = asyncio.create_task(
                self._listen(channel)
            )

    async def unsubscribe(self, channel: Channel) -> None:
        """Unsubscribe from channel."""
        if channel in self._listener_tasks:
            self._listener_tasks[channel].cancel()
            del self._listener_tasks[channel]

        if channel in self._queues:
            del self._queues[channel]

        self._handlers.pop(channel, None)

    async def _listen(self, channel: Channel) -> None:
        """Listen for messages on a channel."""
        queue = self._queues[channel]

        while self._running:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=1.0)

                for handler in self._handlers.get(channel, []):
                    try:
                        await handler(message)
                    except Exception as e:
                        self._error_count += 1
                        logger.error(f"Handler error: {e}")

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_message_bus(
    backend: str = "redis",
    service_name: str = "unknown",
    **kwargs: Any,
) -> MessageBus:
    """
    Factory function to create a message bus.

    Args:
        backend: Backend type (redis, zeromq, memory)
        service_name: Service name
        **kwargs: Backend-specific arguments

    Returns:
        MessageBus instance
    """
    if backend == "redis":
        return RedisMessageBus(service_name=service_name, **kwargs)
    elif backend == "zeromq":
        return ZeroMQMessageBus(service_name=service_name, **kwargs)
    elif backend == "memory":
        return InMemoryMessageBus(service_name=service_name)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "MessageType",
    "MessagePriority",
    "Channel",
    # Data classes
    "Message",
    # Base class
    "MessageBus",
    # Implementations
    "RedisMessageBus",
    "ZeroMQMessageBus",
    "InMemoryMessageBus",
    # Factory
    "create_message_bus",
]
