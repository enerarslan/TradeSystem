"""
Base Service Module
===================

Abstract base class for all microservices in the trading system.
Provides common functionality for service lifecycle, messaging, and health.

Author: AlphaTrade Platform
Version: 3.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import signal
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from config.settings import get_logger
from infrastructure.message_bus import (
    MessageBus,
    RedisMessageBus,
    Message,
    MessageType,
    Channel,
)
from infrastructure.state_store import StateStore, RedisStateStore
from infrastructure.service_registry import (
    ServiceRegistry,
    ServiceInfo,
    ServiceType,
    ServiceStatus,
)

logger = get_logger(__name__)


@dataclass
class ServiceConfig:
    """Base configuration for services."""
    name: str
    service_type: ServiceType
    redis_url: str = "redis://localhost:6379"
    host: str = "localhost"
    port: int = 0
    heartbeat_interval: float = 10.0
    log_level: str = "INFO"


class BaseService(ABC):
    """
    Abstract base class for all trading system services.

    Provides:
    - Lifecycle management (start, stop, restart)
    - Message bus integration
    - State store access
    - Service registry registration
    - Health monitoring
    - Graceful shutdown handling

    Example:
        class MyService(BaseService):
            async def _on_start(self):
                await self.subscribe(Channel.MARKET_DATA, self.handle_data)

            async def _on_stop(self):
                pass

            async def handle_data(self, message):
                # Process market data
                pass
    """

    def __init__(self, config: ServiceConfig):
        """
        Initialize the service.

        Args:
            config: Service configuration
        """
        self.config = config
        self.name = config.name

        # Infrastructure components
        self._message_bus: MessageBus | None = None
        self._state_store: StateStore | None = None
        self._registry: ServiceRegistry | None = None

        # State
        self._running = False
        self._started_at: float | None = None
        self._status = ServiceStatus.STARTING

        # Tasks
        self._heartbeat_task: asyncio.Task | None = None
        self._background_tasks: list[asyncio.Task] = []

        # Shutdown handling
        self._shutdown_event = asyncio.Event()

    @property
    def is_running(self) -> bool:
        """Check if service is running."""
        return self._running

    @property
    def uptime(self) -> float:
        """Get service uptime in seconds."""
        if self._started_at:
            return time.time() - self._started_at
        return 0.0

    async def start(self) -> None:
        """
        Start the service.

        This method:
        1. Connects to infrastructure (Redis, message bus)
        2. Registers with service registry
        3. Starts heartbeat
        4. Calls _on_start() for service-specific initialization
        5. Sets up signal handlers for graceful shutdown
        """
        logger.info(f"Starting service: {self.name}")

        try:
            # Connect to message bus
            self._message_bus = RedisMessageBus(
                redis_url=self.config.redis_url,
                service_name=self.name,
            )
            await self._message_bus.connect()

            # Connect to state store
            self._state_store = RedisStateStore(
                redis_url=self.config.redis_url,
                service_name=self.name,
            )
            await self._state_store.connect()

            # Connect to service registry
            self._registry = ServiceRegistry(
                redis_url=self.config.redis_url,
            )
            await self._registry.connect()

            # Register service
            service_info = ServiceInfo(
                name=self.name,
                service_type=self.config.service_type,
                host=self.config.host,
                port=self.config.port,
                status=ServiceStatus.STARTING,
            )
            await self._registry.register(service_info)

            # Subscribe to system channel for kill switch
            await self._message_bus.subscribe(
                Channel.SYSTEM,
                self._handle_system_message,
            )

            # Call service-specific startup
            await self._on_start()

            # Start heartbeat
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # Update status
            self._running = True
            self._started_at = time.time()
            self._status = ServiceStatus.HEALTHY
            await self._registry.update_status(self.name, ServiceStatus.HEALTHY)

            # Setup signal handlers
            self._setup_signal_handlers()

            logger.info(f"Service {self.name} started successfully")

        except Exception as e:
            logger.error(f"Failed to start service: {e}")
            self._status = ServiceStatus.UNHEALTHY
            raise

    async def stop(self) -> None:
        """
        Stop the service gracefully.

        This method:
        1. Sets shutdown event
        2. Calls _on_stop() for service-specific cleanup
        3. Cancels background tasks
        4. Deregisters from service registry
        5. Disconnects from infrastructure
        """
        logger.info(f"Stopping service: {self.name}")

        self._running = False
        self._status = ServiceStatus.STOPPING
        self._shutdown_event.set()

        try:
            # Update registry status
            if self._registry:
                await self._registry.update_status(self.name, ServiceStatus.STOPPING)

            # Call service-specific shutdown
            await self._on_stop()

            # Cancel heartbeat
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass

            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Deregister from registry
            if self._registry:
                await self._registry.deregister(self.name)
                await self._registry.disconnect()

            # Disconnect from state store
            if self._state_store:
                await self._state_store.disconnect()

            # Disconnect from message bus
            if self._message_bus:
                await self._message_bus.disconnect()

            logger.info(f"Service {self.name} stopped")

        except Exception as e:
            logger.error(f"Error stopping service: {e}")

    async def restart(self) -> None:
        """Restart the service."""
        logger.info(f"Restarting service: {self.name}")
        await self.stop()
        await asyncio.sleep(1)
        await self.start()

    @abstractmethod
    async def _on_start(self) -> None:
        """
        Service-specific startup logic.

        Override this method to:
        - Subscribe to channels
        - Initialize service state
        - Start background tasks
        """
        pass

    @abstractmethod
    async def _on_stop(self) -> None:
        """
        Service-specific shutdown logic.

        Override this method to:
        - Clean up resources
        - Flush pending data
        - Close connections
        """
        pass

    async def subscribe(
        self,
        channel: Channel,
        handler,
    ) -> None:
        """Subscribe to a message channel."""
        if self._message_bus:
            await self._message_bus.subscribe(channel, handler)

    async def publish(self, message: Message) -> bool:
        """Publish a message."""
        if self._message_bus:
            return await self._message_bus.publish(message)
        return False

    async def _handle_system_message(self, message: Message) -> None:
        """Handle system messages (kill switch, restart, etc.)."""
        if message.type == MessageType.KILL_SWITCH:
            logger.critical(f"Kill switch received: {message.payload}")
            await self._handle_kill_switch(message)

        elif message.type == MessageType.RESTART:
            if message.target is None or message.target == self.name:
                await self.restart()

        elif message.type == MessageType.SHUTDOWN:
            if message.target is None or message.target == self.name:
                await self.stop()

    async def _handle_kill_switch(self, message: Message) -> None:
        """
        Handle kill switch activation.

        Override this method in services that need special
        kill switch handling (e.g., OEMS closing positions).
        """
        logger.critical(f"Service {self.name} received kill switch")
        # Default: just stop the service
        await self.stop()

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats."""
        while self._running:
            try:
                # Update heartbeat in state store
                if self._state_store:
                    await self._state_store.update_heartbeat()

                # Update registry
                if self._registry:
                    await self._registry.heartbeat(self.name)

                # Publish heartbeat message
                if self._message_bus:
                    await self._message_bus.publish_heartbeat()

                await asyncio.sleep(self.config.heartbeat_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(1)

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(
                    sig,
                    lambda: asyncio.create_task(self.stop()),
                )
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

    def add_background_task(self, coro) -> asyncio.Task:
        """Add a background task that will be cancelled on shutdown."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        return task

    def get_status(self) -> dict[str, Any]:
        """Get service status."""
        return {
            "name": self.name,
            "type": self.config.service_type.value,
            "status": self._status.value,
            "running": self._running,
            "uptime": self.uptime,
            "started_at": self._started_at,
        }

    async def run_forever(self) -> None:
        """Run the service until shutdown."""
        await self.start()

        try:
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ServiceConfig",
    "BaseService",
]
