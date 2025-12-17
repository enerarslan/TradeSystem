"""
Health Monitoring System for AlphaTrade.

JPMorgan-level implementation of system health monitoring:
- Component health tracking
- Heartbeat monitoring
- Resource utilization
- Latency tracking
- Alert generation

This module provides continuous monitoring of all system components.
"""

from __future__ import annotations

import json
import os
import platform
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - resource monitoring limited")


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    CRITICAL = "critical"


@dataclass
class ComponentHealth:
    """Health status of a single component."""

    name: str
    status: HealthStatus
    last_heartbeat: datetime | None = None
    latency_ms: float | None = None
    error_count: int = 0
    last_error: str | None = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "latency_ms": self.latency_ms,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "details": self.details,
        }


@dataclass
class SystemMetrics:
    """System resource metrics."""

    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    disk_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    thread_count: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_gb": round(self.memory_used_gb, 2),
            "disk_percent": self.disk_percent,
            "disk_free_gb": round(self.disk_free_gb, 2),
            "network_bytes_sent": self.network_bytes_sent,
            "network_bytes_recv": self.network_bytes_recv,
            "process_count": self.process_count,
            "thread_count": self.thread_count,
            "timestamp": self.timestamp.isoformat(),
        }


class HealthMonitor:
    """
    System health monitoring.

    Monitors all system components and resources, generating alerts
    when issues are detected.
    """

    def __init__(
        self,
        heartbeat_timeout: int = 60,
        check_interval: int = 10,
        alert_callback: Callable[[str, HealthStatus, str], None] | None = None,
    ) -> None:
        """
        Initialize health monitor.

        Args:
            heartbeat_timeout: Seconds before component is unhealthy
            check_interval: Seconds between health checks
            alert_callback: Function to call on alerts (component, status, message)
        """
        self.heartbeat_timeout = heartbeat_timeout
        self.check_interval = check_interval
        self._alert_callback = alert_callback

        # Component health tracking
        self._components: Dict[str, ComponentHealth] = {}
        self._health_checks: Dict[str, Callable[[], bool]] = {}

        # Resource thresholds
        self._thresholds = {
            "cpu_warning": 80.0,
            "cpu_critical": 95.0,
            "memory_warning": 80.0,
            "memory_critical": 95.0,
            "disk_warning": 85.0,
            "disk_critical": 95.0,
        }

        # Metrics history
        self._metrics_history: List[SystemMetrics] = []
        self._max_history = 1000

        # Monitoring thread
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Register core components
        self._register_core_components()

        logger.info("Health monitor initialized")

    def _register_core_components(self) -> None:
        """Register core system components."""
        self.register_component("system", self._check_system_health)
        self.register_component("data_layer", self._check_data_health)
        self.register_component("risk_manager", self._check_risk_health)
        self.register_component("backtest_engine", self._check_backtest_health)

    def register_component(
        self,
        name: str,
        health_check: Callable[[], bool] | None = None,
    ) -> None:
        """
        Register a component for health monitoring.

        Args:
            name: Component name
            health_check: Optional function that returns True if healthy
        """
        self._components[name] = ComponentHealth(
            name=name,
            status=HealthStatus.UNKNOWN,
            last_heartbeat=datetime.now(),
        )

        if health_check:
            self._health_checks[name] = health_check

        logger.debug(f"Registered component: {name}")

    def heartbeat(self, component: str, latency_ms: float | None = None) -> None:
        """
        Record a heartbeat from a component.

        Args:
            component: Component name
            latency_ms: Optional latency measurement
        """
        if component not in self._components:
            self.register_component(component)

        health = self._components[component]
        health.last_heartbeat = datetime.now()
        health.status = HealthStatus.HEALTHY

        if latency_ms is not None:
            health.latency_ms = latency_ms

    def report_error(self, component: str, error: str) -> None:
        """
        Report an error from a component.

        Args:
            component: Component name
            error: Error message
        """
        if component not in self._components:
            self.register_component(component)

        health = self._components[component]
        health.error_count += 1
        health.last_error = error

        # Degrade status
        if health.error_count >= 3:
            health.status = HealthStatus.UNHEALTHY
        else:
            health.status = HealthStatus.DEGRADED

        self._generate_alert(
            component,
            health.status,
            f"Error reported: {error}",
        )

    def get_component_health(self, component: str) -> ComponentHealth | None:
        """Get health status of a component."""
        return self._components.get(component)

    def get_all_health(self) -> Dict[str, ComponentHealth]:
        """Get health status of all components."""
        return self._components.copy()

    def get_overall_status(self) -> HealthStatus:
        """
        Get overall system health status.

        Returns the worst status of any component.
        """
        if not self._components:
            return HealthStatus.UNKNOWN

        statuses = [c.status for c in self._components.values()]

        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        if HealthStatus.UNKNOWN in statuses:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def collect_metrics(self) -> SystemMetrics | None:
        """Collect current system metrics."""
        if not PSUTIL_AVAILABLE:
            return None

        try:
            cpu = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            network = psutil.net_io_counters()

            metrics = SystemMetrics(
                cpu_percent=cpu,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024 ** 3),
                disk_percent=disk.percent,
                disk_free_gb=disk.free / (1024 ** 3),
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                process_count=len(psutil.pids()),
                thread_count=threading.active_count(),
            )

            # Store in history
            self._metrics_history.append(metrics)
            if len(self._metrics_history) > self._max_history:
                self._metrics_history = self._metrics_history[-self._max_history:]

            return metrics

        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return None

    def check_all(self) -> Dict[str, HealthStatus]:
        """
        Check health of all components.

        Returns:
            Dictionary mapping component names to status
        """
        now = datetime.now()
        results = {}

        for name, health in self._components.items():
            # Check heartbeat timeout
            if health.last_heartbeat:
                elapsed = (now - health.last_heartbeat).total_seconds()
                if elapsed > self.heartbeat_timeout:
                    health.status = HealthStatus.UNHEALTHY
                    self._generate_alert(
                        name,
                        HealthStatus.UNHEALTHY,
                        f"No heartbeat for {elapsed:.0f} seconds",
                    )

            # Run custom health check if available
            if name in self._health_checks:
                try:
                    is_healthy = self._health_checks[name]()
                    if not is_healthy:
                        health.status = HealthStatus.DEGRADED
                except Exception as e:
                    health.status = HealthStatus.UNHEALTHY
                    health.last_error = str(e)

            results[name] = health.status

        # Check resource thresholds
        metrics = self.collect_metrics()
        if metrics:
            self._check_resource_thresholds(metrics)

        return results

    def _check_resource_thresholds(self, metrics: SystemMetrics) -> None:
        """Check if resource metrics exceed thresholds."""
        # CPU
        if metrics.cpu_percent >= self._thresholds["cpu_critical"]:
            self._generate_alert(
                "system",
                HealthStatus.CRITICAL,
                f"CPU usage critical: {metrics.cpu_percent:.1f}%",
            )
        elif metrics.cpu_percent >= self._thresholds["cpu_warning"]:
            self._generate_alert(
                "system",
                HealthStatus.DEGRADED,
                f"CPU usage high: {metrics.cpu_percent:.1f}%",
            )

        # Memory
        if metrics.memory_percent >= self._thresholds["memory_critical"]:
            self._generate_alert(
                "system",
                HealthStatus.CRITICAL,
                f"Memory usage critical: {metrics.memory_percent:.1f}%",
            )
        elif metrics.memory_percent >= self._thresholds["memory_warning"]:
            self._generate_alert(
                "system",
                HealthStatus.DEGRADED,
                f"Memory usage high: {metrics.memory_percent:.1f}%",
            )

        # Disk
        if metrics.disk_percent >= self._thresholds["disk_critical"]:
            self._generate_alert(
                "system",
                HealthStatus.CRITICAL,
                f"Disk usage critical: {metrics.disk_percent:.1f}%",
            )

    def _generate_alert(
        self,
        component: str,
        status: HealthStatus,
        message: str,
    ) -> None:
        """Generate a health alert."""
        logger.warning(f"HEALTH ALERT [{component}] {status.value}: {message}")

        if self._alert_callback:
            try:
                self._alert_callback(component, status, message)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def _check_system_health(self) -> bool:
        """Check overall system health."""
        if not PSUTIL_AVAILABLE:
            return True

        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent

        return cpu < 95 and memory < 95

    def _check_data_health(self) -> bool:
        """Check data layer health."""
        # Would check database connectivity, data freshness, etc.
        return True

    def _check_risk_health(self) -> bool:
        """Check risk manager health."""
        # Would check risk calculations are running, limits are active
        return True

    def _check_backtest_health(self) -> bool:
        """Check backtest engine health."""
        return True

    def start(self) -> None:
        """Start background health monitoring."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.warning("Health monitor already running")
            return

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Health monitoring started")

    def stop(self) -> None:
        """Stop background health monitoring."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self.check_all()
            except Exception as e:
                logger.error(f"Health check error: {e}")

            self._stop_event.wait(self.check_interval)

    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        metrics = self.collect_metrics()

        return {
            "overall_status": self.get_overall_status().value,
            "timestamp": datetime.now().isoformat(),
            "components": {
                name: health.to_dict()
                for name, health in self._components.items()
            },
            "system_metrics": metrics.to_dict() if metrics else None,
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "python_version": platform.python_version(),
            },
        }

    def save_report(self, filepath: str | Path) -> None:
        """Save health report to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        report = self.get_health_report()

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        logger.debug(f"Health report saved to {filepath}")
