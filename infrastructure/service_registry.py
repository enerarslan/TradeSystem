"""
Service Registry Module
=======================

Service discovery and health monitoring for distributed trading system.
Enables dynamic service registration and health-based routing.

Features:
- Automatic service registration
- Health check monitoring
- Service discovery
- Load balancing support
- Failover handling

Author: AlphaTrade Platform
Version: 3.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from config.settings import get_logger

logger = get_logger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class ServiceStatus(str, Enum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    STARTING = "starting"
    STOPPING = "stopping"


class ServiceType(str, Enum):
    """Service types in the trading system."""
    DATA_INGESTION = "data_ingestion"
    STRATEGY_ENGINE = "strategy_engine"
    RISK_ENGINE = "risk_engine"
    OEMS = "oems"  # Order Execution Management System
    WATCHDOG = "watchdog"
    API_GATEWAY = "api_gateway"
    ML_INFERENCE = "ml_inference"
    FEATURE_STORE = "feature_store"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ServiceInfo:
    """
    Service registration information.

    Attributes:
        name: Unique service name
        service_type: Type of service
        host: Service host address
        port: Service port
        status: Current health status
        version: Service version
        metadata: Additional service metadata
        registered_at: Registration timestamp
        last_heartbeat: Last heartbeat timestamp
        health_check_url: URL for health checks
    """
    name: str
    service_type: ServiceType
    host: str = "localhost"
    port: int = 0
    status: ServiceStatus = ServiceStatus.STARTING
    version: str = "1.0.0"
    metadata: dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    health_check_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "service_type": self.service_type.value,
            "host": self.host,
            "port": self.port,
            "status": self.status.value,
            "version": self.version,
            "metadata": self.metadata,
            "registered_at": self.registered_at,
            "last_heartbeat": self.last_heartbeat,
            "health_check_url": self.health_check_url,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ServiceInfo":
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            service_type=ServiceType(data["service_type"]),
            host=data.get("host", "localhost"),
            port=data.get("port", 0),
            status=ServiceStatus(data.get("status", "unknown")),
            version=data.get("version", "1.0.0"),
            metadata=data.get("metadata", {}),
            registered_at=data.get("registered_at", time.time()),
            last_heartbeat=data.get("last_heartbeat", time.time()),
            health_check_url=data.get("health_check_url"),
        )

    @property
    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self.status == ServiceStatus.HEALTHY

    @property
    def is_alive(self) -> bool:
        """Check if service has recent heartbeat (within 30 seconds)."""
        return time.time() - self.last_heartbeat < 30

    @property
    def address(self) -> str:
        """Get service address."""
        return f"{self.host}:{self.port}" if self.port else self.host


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

class ServiceRegistry:
    """
    Distributed service registry for trading system.

    Enables:
    1. Dynamic service discovery
    2. Health monitoring
    3. Automatic failover
    4. Load balancing

    Example:
        registry = ServiceRegistry(redis_url="redis://localhost:6379")
        await registry.connect()

        # Register a service
        info = ServiceInfo(
            name="strategy_engine_1",
            service_type=ServiceType.STRATEGY_ENGINE,
            host="localhost",
            port=8080
        )
        await registry.register(info)

        # Discover services
        strategies = await registry.discover(ServiceType.STRATEGY_ENGINE)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "alphatrade:services",
        heartbeat_interval: float = 10.0,
        heartbeat_timeout: float = 30.0,
    ):
        """
        Initialize service registry.

        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for registry keys
            heartbeat_interval: Seconds between heartbeats
            heartbeat_timeout: Seconds before service considered dead
        """
        if not REDIS_AVAILABLE:
            raise ImportError("redis package required: pip install redis")

        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout

        self._client: aioredis.Redis | None = None
        self._connected = False
        self._local_services: dict[str, ServiceInfo] = {}
        self._heartbeat_task: asyncio.Task | None = None

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            self._client = aioredis.from_url(
                self.redis_url,
                decode_responses=True,
            )
            await self._client.ping()
            self._connected = True
            logger.info("Service registry connected")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect and deregister all local services."""
        # Stop heartbeat task
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Deregister all local services
        for name in list(self._local_services.keys()):
            await self.deregister(name)

        if self._client:
            await self._client.close()

        self._connected = False
        logger.info("Service registry disconnected")

    async def register(
        self,
        service: ServiceInfo,
        start_heartbeat: bool = True,
    ) -> bool:
        """
        Register a service.

        Args:
            service: Service information
            start_heartbeat: Start automatic heartbeat

        Returns:
            True if registered successfully
        """
        if not self._client:
            return False

        try:
            service.registered_at = time.time()
            service.last_heartbeat = time.time()
            service.status = ServiceStatus.HEALTHY

            # Store service info
            service_key = f"{self.key_prefix}:{service.name}"
            await self._client.hset(
                service_key,
                mapping={k: str(v) if not isinstance(v, str) else v
                         for k, v in service.to_dict().items()
                         if not isinstance(v, dict)},
            )

            # Store metadata separately
            if service.metadata:
                await self._client.hset(
                    f"{service_key}:metadata",
                    mapping=service.metadata,
                )

            # Add to service type set
            type_key = f"{self.key_prefix}:type:{service.service_type.value}"
            await self._client.sadd(type_key, service.name)

            # Add to all services set
            await self._client.sadd(f"{self.key_prefix}:all", service.name)

            # Set TTL for automatic expiration
            await self._client.expire(
                service_key,
                int(self.heartbeat_timeout * 2),
            )

            self._local_services[service.name] = service

            # Start heartbeat if requested
            if start_heartbeat and not self._heartbeat_task:
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            logger.info(f"Registered service: {service.name} ({service.service_type.value})")
            return True

        except Exception as e:
            logger.error(f"Failed to register service: {e}")
            return False

    async def deregister(self, name: str) -> bool:
        """
        Deregister a service.

        Args:
            name: Service name

        Returns:
            True if deregistered successfully
        """
        if not self._client:
            return False

        try:
            service = self._local_services.pop(name, None)

            # Remove from Redis
            service_key = f"{self.key_prefix}:{name}"
            await self._client.delete(service_key)
            await self._client.delete(f"{service_key}:metadata")

            # Remove from sets
            if service:
                type_key = f"{self.key_prefix}:type:{service.service_type.value}"
                await self._client.srem(type_key, name)

            await self._client.srem(f"{self.key_prefix}:all", name)

            logger.info(f"Deregistered service: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to deregister service: {e}")
            return False

    async def heartbeat(self, name: str) -> bool:
        """
        Send heartbeat for a service.

        Args:
            name: Service name

        Returns:
            True if heartbeat sent successfully
        """
        if not self._client:
            return False

        try:
            service_key = f"{self.key_prefix}:{name}"
            now = time.time()

            # Update heartbeat timestamp
            await self._client.hset(service_key, "last_heartbeat", str(now))

            # Refresh TTL
            await self._client.expire(
                service_key,
                int(self.heartbeat_timeout * 2),
            )

            # Update local service
            if name in self._local_services:
                self._local_services[name].last_heartbeat = now

            return True

        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")
            return False

    async def update_status(
        self,
        name: str,
        status: ServiceStatus,
    ) -> bool:
        """
        Update service status.

        Args:
            name: Service name
            status: New status

        Returns:
            True if updated successfully
        """
        if not self._client:
            return False

        try:
            service_key = f"{self.key_prefix}:{name}"
            await self._client.hset(service_key, "status", status.value)

            if name in self._local_services:
                self._local_services[name].status = status

            logger.info(f"Service {name} status: {status.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to update status: {e}")
            return False

    async def get_service(self, name: str) -> ServiceInfo | None:
        """
        Get service information.

        Args:
            name: Service name

        Returns:
            ServiceInfo or None if not found
        """
        if not self._client:
            return None

        try:
            service_key = f"{self.key_prefix}:{name}"
            data = await self._client.hgetall(service_key)

            if not data:
                return None

            # Parse data
            parsed = {
                "name": data.get("name", name),
                "service_type": data.get("service_type", "unknown"),
                "host": data.get("host", "localhost"),
                "port": int(data.get("port", 0)),
                "status": data.get("status", "unknown"),
                "version": data.get("version", "1.0.0"),
                "registered_at": float(data.get("registered_at", time.time())),
                "last_heartbeat": float(data.get("last_heartbeat", time.time())),
                "health_check_url": data.get("health_check_url"),
            }

            # Get metadata
            metadata = await self._client.hgetall(f"{service_key}:metadata")
            parsed["metadata"] = metadata or {}

            return ServiceInfo.from_dict(parsed)

        except Exception as e:
            logger.error(f"Failed to get service: {e}")
            return None

    async def discover(
        self,
        service_type: ServiceType | None = None,
        healthy_only: bool = True,
    ) -> list[ServiceInfo]:
        """
        Discover services.

        Args:
            service_type: Filter by service type (None = all)
            healthy_only: Only return healthy services

        Returns:
            List of ServiceInfo
        """
        if not self._client:
            return []

        try:
            # Get service names
            if service_type:
                type_key = f"{self.key_prefix}:type:{service_type.value}"
                names = await self._client.smembers(type_key)
            else:
                names = await self._client.smembers(f"{self.key_prefix}:all")

            # Get service info for each
            services = []
            for name in names:
                service = await self.get_service(name)
                if service:
                    # Check health filter
                    if healthy_only and not service.is_alive:
                        continue
                    services.append(service)

            return services

        except Exception as e:
            logger.error(f"Failed to discover services: {e}")
            return []

    async def get_healthy_service(
        self,
        service_type: ServiceType,
    ) -> ServiceInfo | None:
        """
        Get a healthy service of the specified type.

        Uses simple round-robin selection among healthy services.

        Args:
            service_type: Type of service needed

        Returns:
            ServiceInfo or None if no healthy service found
        """
        services = await self.discover(service_type, healthy_only=True)

        if not services:
            return None

        # Simple selection: return first healthy service
        # Could be enhanced with load balancing
        return services[0]

    async def _heartbeat_loop(self) -> None:
        """Background task to send heartbeats for local services."""
        logger.info("Starting heartbeat loop")

        while True:
            try:
                for name in self._local_services:
                    await self.heartbeat(name)

                await asyncio.sleep(self.heartbeat_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(1)

    async def cleanup_dead_services(self) -> int:
        """
        Remove services that haven't sent heartbeats.

        Returns:
            Number of services cleaned up
        """
        if not self._client:
            return 0

        cleaned = 0
        names = await self._client.smembers(f"{self.key_prefix}:all")

        for name in names:
            service = await self.get_service(name)
            if service and not service.is_alive:
                await self.deregister(name)
                cleaned += 1

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} dead services")

        return cleaned


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ServiceStatus",
    "ServiceType",
    "ServiceInfo",
    "ServiceRegistry",
]
