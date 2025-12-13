"""
Graceful Degradation Manager
============================
JPMorgan-Level System Resilience and Fault Tolerance

This module manages system behavior during component failures.
Instead of crashing, the system gracefully degrades:

1. WebSocket fails → Fall back to REST polling
2. Model fails → Use rule-based strategy
3. Feature fails → Use reduced feature set
4. Broker fails → Pause trading, don't crash

Key Features:
- Component health monitoring
- Automatic fallback activation
- Recovery detection and restoration
- Degradation level tracking
- Alert generation

Author: AlphaTrade Institutional System
Based on: ARCHITECTURAL_REVIEW_REPORT.md - Phase 2
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import threading

from ..utils.logger import get_logger, get_audit_logger

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class ComponentStatus(Enum):
    """Status of a system component"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"
    UNKNOWN = "unknown"


class DegradationLevel(Enum):
    """System degradation levels"""
    NONE = 0  # Fully operational
    LOW = 1  # Minor issues, full functionality
    MEDIUM = 2  # Some components degraded, reduced functionality
    HIGH = 3  # Critical components degraded, minimal functionality
    CRITICAL = 4  # System barely functional, no trading
    EMERGENCY = 5  # Emergency shutdown required


class ComponentType(Enum):
    """Types of system components"""
    WEBSOCKET = "websocket"
    REST_API = "rest_api"
    BROKER = "broker"
    DATABASE = "database"
    REDIS = "redis"
    MODEL = "model"
    FEATURES = "features"
    RISK = "risk"
    EXECUTION = "execution"
    DATA_FEED = "data_feed"


@dataclass
class ComponentHealth:
    """Health status of a component"""
    component_type: ComponentType
    name: str
    status: ComponentStatus = ComponentStatus.UNKNOWN
    last_check: datetime = field(default_factory=datetime.now)
    last_healthy: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    error_message: str = ""
    fallback_active: bool = False
    response_time_ms: float = 0.0

    def update_healthy(self, response_time_ms: float = 0.0) -> None:
        """Mark component as healthy"""
        self.status = ComponentStatus.HEALTHY
        self.last_check = datetime.now()
        self.last_healthy = datetime.now()
        self.consecutive_failures = 0
        self.consecutive_successes += 1
        self.error_message = ""
        self.response_time_ms = response_time_ms

    def update_failed(self, error: str) -> None:
        """Mark component as failed"""
        self.status = ComponentStatus.FAILED
        self.last_check = datetime.now()
        self.last_failure = datetime.now()
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.error_message = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            'component_type': self.component_type.value,
            'name': self.name,
            'status': self.status.value,
            'last_check': self.last_check.isoformat(),
            'last_healthy': self.last_healthy.isoformat() if self.last_healthy else None,
            'consecutive_failures': self.consecutive_failures,
            'error_message': self.error_message,
            'fallback_active': self.fallback_active,
            'response_time_ms': self.response_time_ms
        }


@dataclass
class FallbackConfig:
    """Configuration for a fallback strategy"""
    component_type: ComponentType
    trigger_failures: int = 3  # Failures before fallback
    recovery_successes: int = 5  # Successes before restoration
    cooldown_seconds: float = 60.0  # Minimum time between transitions
    fallback_function: Optional[Callable] = None
    recovery_function: Optional[Callable] = None


class GracefulDegradationManager:
    """
    Manages system degradation and recovery.

    This is critical for production systems. When components fail:
    1. Detects failure via health checks
    2. Activates fallback strategies
    3. Reduces system functionality appropriately
    4. Monitors for recovery
    5. Restores full functionality when healthy

    Example Degradations:
    - WebSocket fails → REST polling fallback
    - ML model fails → Rule-based signals
    - Broker connection fails → Pause trading
    - Database fails → In-memory caching
    """

    def __init__(
        self,
        check_interval_seconds: float = 5.0,
        alert_callback: Optional[Callable] = None
    ):
        self.check_interval = check_interval_seconds
        self.alert_callback = alert_callback

        # Component tracking
        self._components: Dict[str, ComponentHealth] = {}
        self._fallback_configs: Dict[ComponentType, FallbackConfig] = {}

        # System state
        self._degradation_level = DegradationLevel.NONE
        self._active_fallbacks: Set[ComponentType] = set()
        self._last_transitions: Dict[str, datetime] = {}

        # Health check functions
        self._health_checks: Dict[str, Callable] = {}

        # Monitoring
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            'degradation_changed': [],
            'fallback_activated': [],
            'recovery_detected': [],
            'component_failed': [],
            'component_recovered': []
        }

        # Statistics
        self._stats = {
            'health_checks': 0,
            'failures_detected': 0,
            'fallbacks_activated': 0,
            'recoveries_detected': 0,
            'max_degradation_level': 0
        }

    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register callback for events"""
        if event_type in self._callbacks:
            self._callbacks[event_type].append(callback)

    def _emit(self, event_type: str, data: Any) -> None:
        """Emit event to callbacks"""
        for callback in self._callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def register_component(
        self,
        name: str,
        component_type: ComponentType,
        health_check: Callable,
        fallback_config: Optional[FallbackConfig] = None
    ) -> None:
        """
        Register a component for monitoring.

        Args:
            name: Unique component name
            component_type: Type of component
            health_check: Async callable returning (healthy: bool, response_time_ms: float, error: str)
            fallback_config: Configuration for fallback behavior
        """
        self._components[name] = ComponentHealth(
            component_type=component_type,
            name=name
        )
        self._health_checks[name] = health_check

        if fallback_config:
            self._fallback_configs[component_type] = fallback_config

        logger.info(f"Registered component: {name} ({component_type.value})")

    def register_fallback(
        self,
        component_type: ComponentType,
        trigger_failures: int = 3,
        recovery_successes: int = 5,
        cooldown_seconds: float = 60.0,
        fallback_function: Optional[Callable] = None,
        recovery_function: Optional[Callable] = None
    ) -> None:
        """
        Register fallback strategy for a component type.

        Args:
            component_type: Type to configure
            trigger_failures: Consecutive failures to trigger fallback
            recovery_successes: Consecutive successes to restore
            cooldown_seconds: Minimum time between transitions
            fallback_function: Called when fallback activates
            recovery_function: Called when recovering from fallback
        """
        self._fallback_configs[component_type] = FallbackConfig(
            component_type=component_type,
            trigger_failures=trigger_failures,
            recovery_successes=recovery_successes,
            cooldown_seconds=cooldown_seconds,
            fallback_function=fallback_function,
            recovery_function=recovery_function
        )

    async def start(self) -> None:
        """Start health monitoring"""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Graceful degradation manager started")

    async def stop(self) -> None:
        """Stop health monitoring"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Graceful degradation manager stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self._running:
            try:
                await self._check_all_components()
                await self._evaluate_degradation()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(self.check_interval)

    async def _check_all_components(self) -> None:
        """Run health checks on all components"""
        for name, health_check in self._health_checks.items():
            try:
                start = time.time()
                healthy, response_time, error = await health_check()
                elapsed = (time.time() - start) * 1000

                component = self._components[name]

                if healthy:
                    component.update_healthy(response_time or elapsed)
                    await self._check_recovery(component)
                else:
                    component.update_failed(error)
                    await self._check_failure(component)

                self._stats['health_checks'] += 1

            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                if name in self._components:
                    self._components[name].update_failed(str(e))
                    await self._check_failure(self._components[name])

    async def _check_failure(self, component: ComponentHealth) -> None:
        """Check if failure should trigger fallback"""
        config = self._fallback_configs.get(component.component_type)
        if not config:
            return

        # Check if already in fallback
        if component.fallback_active:
            return

        # Check if enough failures
        if component.consecutive_failures < config.trigger_failures:
            return

        # Check cooldown
        transition_key = f"{component.name}_to_fallback"
        if transition_key in self._last_transitions:
            elapsed = (datetime.now() - self._last_transitions[transition_key]).total_seconds()
            if elapsed < config.cooldown_seconds:
                return

        # Activate fallback
        await self._activate_fallback(component, config)

    async def _check_recovery(self, component: ComponentHealth) -> None:
        """Check if recovery should restore normal operation"""
        config = self._fallback_configs.get(component.component_type)
        if not config:
            return

        # Check if in fallback
        if not component.fallback_active:
            return

        # Check if enough successes
        if component.consecutive_successes < config.recovery_successes:
            return

        # Check cooldown
        transition_key = f"{component.name}_to_normal"
        if transition_key in self._last_transitions:
            elapsed = (datetime.now() - self._last_transitions[transition_key]).total_seconds()
            if elapsed < config.cooldown_seconds:
                return

        # Restore normal operation
        await self._restore_normal(component, config)

    async def _activate_fallback(
        self,
        component: ComponentHealth,
        config: FallbackConfig
    ) -> None:
        """Activate fallback for a component"""
        logger.warning(
            f"ACTIVATING FALLBACK for {component.name} "
            f"after {component.consecutive_failures} failures"
        )

        component.fallback_active = True
        component.status = ComponentStatus.DEGRADED
        self._active_fallbacks.add(component.component_type)
        self._last_transitions[f"{component.name}_to_fallback"] = datetime.now()

        # Call fallback function
        if config.fallback_function:
            try:
                await config.fallback_function() if asyncio.iscoroutinefunction(config.fallback_function) \
                    else config.fallback_function()
            except Exception as e:
                logger.error(f"Fallback function error: {e}")

        self._stats['fallbacks_activated'] += 1

        # Log and emit events
        self._emit('fallback_activated', component)
        self._emit('component_failed', component)

        audit_logger.log_risk_event(
            event_type="FALLBACK_ACTIVATED",
            details={
                'component': component.name,
                'type': component.component_type.value,
                'failures': component.consecutive_failures,
                'error': component.error_message
            }
        )

        if self.alert_callback:
            self.alert_callback(
                f"FALLBACK ACTIVATED: {component.name} - {component.error_message}"
            )

    async def _restore_normal(
        self,
        component: ComponentHealth,
        config: FallbackConfig
    ) -> None:
        """Restore normal operation for a component"""
        logger.info(
            f"RESTORING NORMAL operation for {component.name} "
            f"after {component.consecutive_successes} successes"
        )

        component.fallback_active = False
        component.status = ComponentStatus.HEALTHY
        self._active_fallbacks.discard(component.component_type)
        self._last_transitions[f"{component.name}_to_normal"] = datetime.now()

        # Call recovery function
        if config.recovery_function:
            try:
                await config.recovery_function() if asyncio.iscoroutinefunction(config.recovery_function) \
                    else config.recovery_function()
            except Exception as e:
                logger.error(f"Recovery function error: {e}")

        self._stats['recoveries_detected'] += 1

        # Log and emit events
        self._emit('recovery_detected', component)
        self._emit('component_recovered', component)

        audit_logger.log_risk_event(
            event_type="NORMAL_RESTORED",
            details={
                'component': component.name,
                'type': component.component_type.value
            }
        )

        if self.alert_callback:
            self.alert_callback(f"RESTORED: {component.name} back to normal")

    async def _evaluate_degradation(self) -> None:
        """Evaluate overall system degradation level"""
        failed_components = [c for c in self._components.values()
                           if c.status == ComponentStatus.FAILED]
        degraded_components = [c for c in self._components.values()
                              if c.fallback_active]

        # Determine degradation level
        new_level = DegradationLevel.NONE

        if len(failed_components) == 0 and len(degraded_components) == 0:
            new_level = DegradationLevel.NONE
        elif len(degraded_components) <= 1:
            new_level = DegradationLevel.LOW
        elif len(degraded_components) <= 2:
            new_level = DegradationLevel.MEDIUM
        else:
            new_level = DegradationLevel.HIGH

        # Check for critical components
        critical_types = {ComponentType.BROKER, ComponentType.RISK, ComponentType.EXECUTION}
        critical_failed = any(
            c.component_type in critical_types
            for c in failed_components
        )

        if critical_failed:
            new_level = DegradationLevel.CRITICAL

        # Update if changed
        if new_level != self._degradation_level:
            old_level = self._degradation_level
            self._degradation_level = new_level

            logger.warning(
                f"DEGRADATION LEVEL CHANGED: {old_level.name} → {new_level.name}"
            )

            self._stats['max_degradation_level'] = max(
                self._stats['max_degradation_level'],
                new_level.value
            )

            self._emit('degradation_changed', {
                'old_level': old_level,
                'new_level': new_level,
                'failed_components': [c.name for c in failed_components],
                'degraded_components': [c.name for c in degraded_components]
            })

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_degradation_level(self) -> DegradationLevel:
        """Get current system degradation level"""
        return self._degradation_level

    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed at current degradation level"""
        return self._degradation_level.value < DegradationLevel.CRITICAL.value

    def is_component_healthy(self, name: str) -> bool:
        """Check if a specific component is healthy"""
        component = self._components.get(name)
        return component is not None and component.status == ComponentStatus.HEALTHY

    def is_fallback_active(self, component_type: ComponentType) -> bool:
        """Check if fallback is active for component type"""
        return component_type in self._active_fallbacks

    def get_component_status(self, name: str) -> Optional[ComponentHealth]:
        """Get status of a specific component"""
        return self._components.get(name)

    def get_all_status(self) -> Dict[str, Dict]:
        """Get status of all components"""
        return {
            name: comp.to_dict()
            for name, comp in self._components.items()
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get degradation statistics"""
        return {
            **self._stats,
            'current_degradation_level': self._degradation_level.value,
            'degradation_level_name': self._degradation_level.name,
            'active_fallbacks': [c.value for c in self._active_fallbacks],
            'components_healthy': len([c for c in self._components.values()
                                       if c.status == ComponentStatus.HEALTHY]),
            'components_failed': len([c for c in self._components.values()
                                      if c.status == ComponentStatus.FAILED]),
            'components_degraded': len([c for c in self._components.values()
                                        if c.fallback_active])
        }

    def force_fallback(self, component_type: ComponentType) -> None:
        """Manually force fallback for testing"""
        for comp in self._components.values():
            if comp.component_type == component_type:
                comp.consecutive_failures = 999
                asyncio.create_task(self._check_failure(comp))

    def force_recovery(self, component_type: ComponentType) -> None:
        """Manually force recovery for testing"""
        for comp in self._components.values():
            if comp.component_type == component_type:
                comp.consecutive_successes = 999
                asyncio.create_task(self._check_recovery(comp))


# =============================================================================
# HEALTH CHECK FACTORIES
# =============================================================================

def create_websocket_health_check(ws_manager) -> Callable:
    """Create health check for WebSocket manager"""
    async def check():
        try:
            stats = ws_manager.get_stats()
            any_connected = any(
                feed.get('connected', False)
                for feed in stats.get('feeds', {}).values()
            )

            if any_connected:
                return True, 0, ""
            else:
                return False, 0, "No WebSocket connections active"
        except Exception as e:
            return False, 0, str(e)

    return check


def create_broker_health_check(broker) -> Callable:
    """Create health check for broker connection"""
    async def check():
        try:
            start = time.time()
            account = await broker.get_account()
            elapsed = (time.time() - start) * 1000

            if account:
                return True, elapsed, ""
            else:
                return False, elapsed, "Failed to get account"
        except Exception as e:
            return False, 0, str(e)

    return check


def create_model_health_check(model) -> Callable:
    """Create health check for ML model"""
    async def check():
        try:
            # Simple prediction test
            import numpy as np
            test_input = np.zeros((1, 10))  # Adjust shape as needed
            start = time.time()
            _ = model.predict(test_input)
            elapsed = (time.time() - start) * 1000

            return True, elapsed, ""
        except Exception as e:
            return False, 0, str(e)

    return check


def create_redis_health_check(redis_client) -> Callable:
    """Create health check for Redis"""
    async def check():
        try:
            start = time.time()
            pong = await redis_client.ping()
            elapsed = (time.time() - start) * 1000

            if pong:
                return True, elapsed, ""
            else:
                return False, elapsed, "Redis ping failed"
        except Exception as e:
            return False, 0, str(e)

    return check


def create_database_health_check(db_connection) -> Callable:
    """Create health check for database"""
    async def check():
        try:
            start = time.time()
            # Execute simple query
            result = await db_connection.execute("SELECT 1")
            elapsed = (time.time() - start) * 1000

            return True, elapsed, ""
        except Exception as e:
            return False, 0, str(e)

    return check
