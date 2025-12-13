"""
Health Check System
===================
JPMorgan-Level System Health Monitoring

Comprehensive health monitoring for all system components:
- Liveness: Is the component alive?
- Readiness: Is the component ready to serve?
- Dependency: Are all dependencies healthy?

Key Features:
1. HTTP health endpoints (for container orchestration)
2. Component-level health checks
3. Dependency graph validation
4. Health history tracking
5. Alert generation

Author: AlphaTrade Institutional System
Based on: ARCHITECTURAL_REVIEW_REPORT.md - Phase 2
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import aiohttp
from aiohttp import web

from ..utils.logger import get_logger

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a single health check"""
    name: str
    status: HealthStatus
    message: str
    response_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'response_time_ms': self.response_time_ms,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details
        }


@dataclass
class SystemHealth:
    """Overall system health status"""
    status: HealthStatus
    timestamp: datetime
    checks: List[HealthCheckResult]
    uptime_seconds: float
    version: str

    @property
    def all_healthy(self) -> bool:
        return all(c.status == HealthStatus.HEALTHY for c in self.checks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'uptime_seconds': self.uptime_seconds,
            'version': self.version,
            'checks': [c.to_dict() for c in self.checks],
            'healthy_count': len([c for c in self.checks if c.status == HealthStatus.HEALTHY]),
            'unhealthy_count': len([c for c in self.checks if c.status == HealthStatus.UNHEALTHY])
        }


class HealthCheckRegistry:
    """
    Registry for health checks.

    Manages registration and execution of health checks for all components.
    """

    def __init__(
        self,
        check_timeout_seconds: float = 5.0,
        history_size: int = 100
    ):
        self.timeout = check_timeout_seconds
        self.history_size = history_size

        # Health checks
        self._checks: Dict[str, Callable] = {}
        self._dependencies: Dict[str, List[str]] = {}

        # Results
        self._last_results: Dict[str, HealthCheckResult] = {}
        self._history: Dict[str, deque] = {}

        # System info
        self._start_time = datetime.now()
        self._version = "1.0.0"

    def register(
        self,
        name: str,
        check_function: Callable,
        dependencies: Optional[List[str]] = None
    ) -> None:
        """
        Register a health check.

        Args:
            name: Unique name for the check
            check_function: Async callable returning Tuple[HealthStatus, str, Dict]
            dependencies: List of other checks this depends on
        """
        self._checks[name] = check_function
        self._dependencies[name] = dependencies or []
        self._history[name] = deque(maxlen=self.history_size)

        logger.info(f"Registered health check: {name}")

    def unregister(self, name: str) -> None:
        """Unregister a health check"""
        self._checks.pop(name, None)
        self._dependencies.pop(name, None)
        self._history.pop(name, None)
        self._last_results.pop(name, None)

    async def check(self, name: str) -> HealthCheckResult:
        """Run a single health check"""
        check_fn = self._checks.get(name)
        if not check_fn:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="Health check not found",
                response_time_ms=0
            )

        start = time.time()
        try:
            status, message, details = await asyncio.wait_for(
                check_fn(),
                timeout=self.timeout
            )
            elapsed = (time.time() - start) * 1000

            result = HealthCheckResult(
                name=name,
                status=status,
                message=message,
                response_time_ms=elapsed,
                details=details
            )

        except asyncio.TimeoutError:
            result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout}s",
                response_time_ms=self.timeout * 1000
            )

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                response_time_ms=elapsed
            )

        # Store result
        self._last_results[name] = result
        self._history[name].append(result)

        return result

    async def check_all(self) -> SystemHealth:
        """Run all registered health checks"""
        results = []

        # Run checks in dependency order
        checked = set()
        remaining = set(self._checks.keys())

        while remaining:
            # Find checks with satisfied dependencies
            ready = [
                name for name in remaining
                if all(dep in checked for dep in self._dependencies.get(name, []))
            ]

            if not ready:
                # Break circular dependencies
                ready = list(remaining)

            # Run ready checks in parallel
            tasks = [self.check(name) for name in ready]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for name, result in zip(ready, batch_results):
                if isinstance(result, Exception):
                    result = HealthCheckResult(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=str(result),
                        response_time_ms=0
                    )
                results.append(result)
                checked.add(name)
                remaining.discard(name)

        # Determine overall status
        if all(r.status == HealthStatus.HEALTHY for r in results):
            overall_status = HealthStatus.HEALTHY
        elif any(r.status == HealthStatus.UNHEALTHY for r in results):
            overall_status = HealthStatus.UNHEALTHY
        else:
            overall_status = HealthStatus.DEGRADED

        uptime = (datetime.now() - self._start_time).total_seconds()

        return SystemHealth(
            status=overall_status,
            timestamp=datetime.now(),
            checks=results,
            uptime_seconds=uptime,
            version=self._version
        )

    async def check_liveness(self) -> Tuple[bool, str]:
        """
        Liveness check - is the system alive?

        Returns True if the system is running, regardless of component health.
        """
        return True, "System is alive"

    async def check_readiness(self) -> Tuple[bool, str]:
        """
        Readiness check - is the system ready to serve?

        Returns True only if all critical components are healthy.
        """
        health = await self.check_all()
        if health.all_healthy:
            return True, "System is ready"
        else:
            unhealthy = [c.name for c in health.checks if c.status != HealthStatus.HEALTHY]
            return False, f"Unhealthy components: {unhealthy}"

    def get_last_result(self, name: str) -> Optional[HealthCheckResult]:
        """Get last result for a check"""
        return self._last_results.get(name)

    def get_history(self, name: str, limit: int = 10) -> List[HealthCheckResult]:
        """Get history for a check"""
        history = self._history.get(name, [])
        return list(history)[-limit:]

    def get_uptime(self) -> timedelta:
        """Get system uptime"""
        return datetime.now() - self._start_time


class HealthCheckServer:
    """
    HTTP server for health check endpoints.

    Provides standard endpoints for container orchestration:
    - /health - Full health status
    - /health/live - Liveness probe
    - /health/ready - Readiness probe
    """

    def __init__(
        self,
        registry: HealthCheckRegistry,
        host: str = "0.0.0.0",
        port: int = 8080
    ):
        self.registry = registry
        self.host = host
        self.port = port

        self._app = web.Application()
        self._runner: Optional[web.AppRunner] = None

        # Setup routes
        self._app.router.add_get('/health', self._handle_health)
        self._app.router.add_get('/health/live', self._handle_liveness)
        self._app.router.add_get('/health/ready', self._handle_readiness)
        self._app.router.add_get('/health/{name}', self._handle_single_check)

    async def start(self) -> None:
        """Start health check server"""
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()

        logger.info(f"Health check server started on {self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop health check server"""
        if self._runner:
            await self._runner.cleanup()
            logger.info("Health check server stopped")

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Handle /health endpoint"""
        health = await self.registry.check_all()

        status_code = 200 if health.all_healthy else 503

        return web.json_response(
            health.to_dict(),
            status=status_code
        )

    async def _handle_liveness(self, request: web.Request) -> web.Response:
        """Handle /health/live endpoint"""
        alive, message = await self.registry.check_liveness()

        return web.json_response(
            {'alive': alive, 'message': message},
            status=200 if alive else 503
        )

    async def _handle_readiness(self, request: web.Request) -> web.Response:
        """Handle /health/ready endpoint"""
        ready, message = await self.registry.check_readiness()

        return web.json_response(
            {'ready': ready, 'message': message},
            status=200 if ready else 503
        )

    async def _handle_single_check(self, request: web.Request) -> web.Response:
        """Handle /health/{name} endpoint"""
        name = request.match_info['name']
        result = await self.registry.check(name)

        status_code = 200 if result.status == HealthStatus.HEALTHY else 503

        return web.json_response(
            result.to_dict(),
            status=status_code
        )


# =============================================================================
# STANDARD HEALTH CHECK FUNCTIONS
# =============================================================================

def create_broker_health_check(broker) -> Callable:
    """Create health check for broker"""
    async def check():
        try:
            start = time.time()
            account = await broker.get_account()
            elapsed = (time.time() - start) * 1000

            if account and hasattr(account, 'portfolio_value'):
                return (
                    HealthStatus.HEALTHY,
                    f"Connected, portfolio value: ${account.portfolio_value:,.2f}",
                    {'response_time_ms': elapsed, 'buying_power': account.buying_power}
                )
            else:
                return (
                    HealthStatus.UNHEALTHY,
                    "Failed to get account info",
                    {}
                )
        except Exception as e:
            return (
                HealthStatus.UNHEALTHY,
                f"Broker error: {str(e)}",
                {}
            )

    return check


def create_websocket_health_check(ws_manager) -> Callable:
    """Create health check for WebSocket"""
    async def check():
        try:
            stats = ws_manager.get_stats()
            feeds = stats.get('feeds', {})
            connected_count = sum(1 for f in feeds.values() if f.get('connected'))

            if connected_count > 0:
                return (
                    HealthStatus.HEALTHY,
                    f"{connected_count} feeds connected",
                    {'feeds': feeds}
                )
            else:
                return (
                    HealthStatus.UNHEALTHY,
                    "No WebSocket connections",
                    {'feeds': feeds}
                )
        except Exception as e:
            return (
                HealthStatus.UNHEALTHY,
                f"WebSocket error: {str(e)}",
                {}
            )

    return check


def create_database_health_check(db_pool) -> Callable:
    """Create health check for database"""
    async def check():
        try:
            start = time.time()
            async with db_pool.acquire() as conn:
                await conn.execute("SELECT 1")
            elapsed = (time.time() - start) * 1000

            return (
                HealthStatus.HEALTHY,
                f"Database connected ({elapsed:.1f}ms)",
                {'response_time_ms': elapsed}
            )
        except Exception as e:
            return (
                HealthStatus.UNHEALTHY,
                f"Database error: {str(e)}",
                {}
            )

    return check


def create_redis_health_check(redis_client) -> Callable:
    """Create health check for Redis"""
    async def check():
        try:
            start = time.time()
            pong = await redis_client.ping()
            elapsed = (time.time() - start) * 1000

            if pong:
                return (
                    HealthStatus.HEALTHY,
                    f"Redis connected ({elapsed:.1f}ms)",
                    {'response_time_ms': elapsed}
                )
            else:
                return (
                    HealthStatus.UNHEALTHY,
                    "Redis ping failed",
                    {}
                )
        except Exception as e:
            return (
                HealthStatus.UNHEALTHY,
                f"Redis error: {str(e)}",
                {}
            )

    return check


def create_model_health_check(model, sample_input) -> Callable:
    """Create health check for ML model"""
    async def check():
        try:
            start = time.time()
            prediction = model.predict(sample_input)
            elapsed = (time.time() - start) * 1000

            return (
                HealthStatus.HEALTHY,
                f"Model responding ({elapsed:.1f}ms)",
                {
                    'response_time_ms': elapsed,
                    'prediction_shape': prediction.shape if hasattr(prediction, 'shape') else None
                }
            )
        except Exception as e:
            return (
                HealthStatus.UNHEALTHY,
                f"Model error: {str(e)}",
                {}
            )

    return check


def create_disk_space_health_check(path: str = "/", min_free_gb: float = 1.0) -> Callable:
    """Create health check for disk space"""
    async def check():
        import shutil
        try:
            total, used, free = shutil.disk_usage(path)
            free_gb = free / (1024 ** 3)

            if free_gb >= min_free_gb:
                return (
                    HealthStatus.HEALTHY,
                    f"Disk space OK ({free_gb:.1f} GB free)",
                    {'free_gb': free_gb, 'total_gb': total / (1024 ** 3)}
                )
            else:
                return (
                    HealthStatus.UNHEALTHY,
                    f"Low disk space ({free_gb:.1f} GB free, min: {min_free_gb} GB)",
                    {'free_gb': free_gb}
                )
        except Exception as e:
            return (
                HealthStatus.UNHEALTHY,
                f"Disk check error: {str(e)}",
                {}
            )

    return check


def create_memory_health_check(max_usage_pct: float = 90.0) -> Callable:
    """Create health check for memory usage"""
    async def check():
        try:
            import psutil
            memory = psutil.virtual_memory()
            usage_pct = memory.percent

            if usage_pct < max_usage_pct:
                return (
                    HealthStatus.HEALTHY,
                    f"Memory OK ({usage_pct:.1f}% used)",
                    {'usage_pct': usage_pct, 'available_mb': memory.available / (1024 ** 2)}
                )
            else:
                return (
                    HealthStatus.UNHEALTHY,
                    f"High memory usage ({usage_pct:.1f}%, max: {max_usage_pct}%)",
                    {'usage_pct': usage_pct}
                )
        except ImportError:
            return (
                HealthStatus.UNKNOWN,
                "psutil not installed",
                {}
            )
        except Exception as e:
            return (
                HealthStatus.UNHEALTHY,
                f"Memory check error: {str(e)}",
                {}
            )

    return check
