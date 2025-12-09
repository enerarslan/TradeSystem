"""
Heartbeat Monitor Module
========================

Independent watchdog system for monitoring service health and
triggering emergency actions when services fail.

This is a CRITICAL safety component that runs independently
and can trigger kill switches even if main services crash.

Features:
- Independent process monitoring
- Automatic kill switch on failure
- PnL-based circuit breakers
- Configurable thresholds
- Alert notifications

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
from typing import Any, Callable, Awaitable

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from config.settings import get_logger
from infrastructure.state_store import StateStore, StateKey, RedisStateStore
from infrastructure.message_bus import MessageBus, Message, MessageType, Channel, MessagePriority

logger = get_logger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class HeartbeatConfig:
    """
    Configuration for heartbeat monitoring.

    Attributes:
        check_interval: Seconds between health checks
        heartbeat_timeout: Seconds before service considered dead
        max_pnl_loss_pct: Max PnL loss % before kill switch (e.g., 0.02 = 2%)
        max_pnl_loss_1min_pct: Max PnL loss % in 1 minute
        max_drawdown_pct: Max drawdown % before kill switch
        required_services: Services that must be healthy
        alert_callback: Optional callback for alerts
        enable_auto_kill: Enable automatic kill switch
    """
    check_interval: float = 5.0
    heartbeat_timeout: float = 30.0
    max_pnl_loss_pct: float = 0.05  # 5% max daily loss
    max_pnl_loss_1min_pct: float = 0.02  # 2% max loss in 1 minute
    max_drawdown_pct: float = 0.10  # 10% max drawdown
    required_services: list[str] = field(default_factory=lambda: [
        "strategy_engine",
        "risk_engine",
        "oems",
    ])
    alert_callback: Callable[[str, str], Awaitable[None]] | None = None
    enable_auto_kill: bool = True


class KillReason(str, Enum):
    """Reasons for kill switch activation."""
    SERVICE_FAILURE = "service_failure"
    PNL_BREACH = "pnl_breach"
    DRAWDOWN_BREACH = "drawdown_breach"
    RAPID_LOSS = "rapid_loss"
    MANUAL = "manual"
    RISK_BREACH = "risk_breach"
    CONNECTIVITY_LOSS = "connectivity_loss"


# =============================================================================
# HEARTBEAT MONITOR
# =============================================================================

class HeartbeatMonitor:
    """
    Independent watchdog for monitoring trading system health.

    This monitor runs as a separate process and:
    1. Monitors heartbeats from all critical services
    2. Tracks PnL and drawdown in real-time
    3. Triggers kill switch if thresholds breached
    4. Sends emergency orders directly to broker API

    CRITICAL: This should run on a SEPARATE server/process from the
    main trading system to ensure it can act even if main system crashes.

    Example:
        config = HeartbeatConfig(
            max_pnl_loss_pct=0.03,  # 3% max loss
            required_services=["strategy_engine", "oems"]
        )

        monitor = HeartbeatMonitor(
            config=config,
            redis_url="redis://localhost:6379"
        )

        await monitor.start()
    """

    def __init__(
        self,
        config: HeartbeatConfig | None = None,
        redis_url: str = "redis://localhost:6379",
        message_bus: MessageBus | None = None,
        state_store: StateStore | None = None,
    ):
        """
        Initialize heartbeat monitor.

        Args:
            config: Monitor configuration
            redis_url: Redis connection URL
            message_bus: Optional message bus for kill switch
            state_store: Optional state store for state access
        """
        self.config = config or HeartbeatConfig()
        self.redis_url = redis_url

        self._message_bus = message_bus
        self._state_store = state_store
        self._client: aioredis.Redis | None = None

        self._running = False
        self._monitor_task: asyncio.Task | None = None

        # Tracking state
        self._last_pnl: float = 0.0
        self._pnl_history: list[tuple[float, float]] = []  # (timestamp, pnl)
        self._kill_switch_active = False
        self._service_status: dict[str, float] = {}  # service -> last_heartbeat

    async def start(self) -> None:
        """Start the heartbeat monitor."""
        try:
            # Connect to Redis directly (independent of other services)
            self._client = aioredis.from_url(
                self.redis_url,
                decode_responses=True,
            )
            await self._client.ping()

            # Initialize state store if not provided
            if not self._state_store:
                self._state_store = RedisStateStore(
                    redis_url=self.redis_url,
                    service_name="watchdog",
                )
                await self._state_store.connect()

            self._running = True
            self._monitor_task = asyncio.create_task(self._monitor_loop())

            logger.info("Heartbeat monitor started")

        except Exception as e:
            logger.error(f"Failed to start heartbeat monitor: {e}")
            raise

    async def stop(self) -> None:
        """Stop the heartbeat monitor."""
        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        if self._client:
            await self._client.close()

        logger.info("Heartbeat monitor stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Starting monitoring loop")

        while self._running:
            try:
                # Check service heartbeats
                await self._check_service_health()

                # Check PnL thresholds
                await self._check_pnl_thresholds()

                # Check drawdown
                await self._check_drawdown()

                await asyncio.sleep(self.config.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(1)

    async def _check_service_health(self) -> None:
        """Check heartbeats of required services."""
        if not self._client:
            return

        now = time.time()
        failed_services = []

        for service_name in self.config.required_services:
            # Check heartbeat in Redis
            heartbeat_key = f"alphatrade:state:service:heartbeat:{service_name}"
            last_heartbeat = await self._client.get(heartbeat_key)

            if last_heartbeat is None:
                # Service never registered
                if service_name not in self._service_status:
                    logger.warning(f"Service {service_name} not found")
                    self._service_status[service_name] = 0
                continue

            last_beat = float(last_heartbeat)
            self._service_status[service_name] = last_beat

            # Check if heartbeat is stale
            if now - last_beat > self.config.heartbeat_timeout:
                failed_services.append(service_name)
                logger.error(
                    f"Service {service_name} heartbeat stale "
                    f"(last: {now - last_beat:.1f}s ago)"
                )

        # Trigger kill switch if critical services failed
        if failed_services and self.config.enable_auto_kill:
            await self._trigger_kill_switch(
                KillReason.SERVICE_FAILURE,
                f"Services failed: {', '.join(failed_services)}"
            )

    async def _check_pnl_thresholds(self) -> None:
        """Check PnL against configured thresholds."""
        if not self._state_store:
            return

        try:
            # Get current risk state
            risk_state = await self._state_store.get_risk_state()
            current_pnl = risk_state.daily_pnl
            initial_equity = risk_state.high_water_mark or 100000.0

            # Record PnL history
            now = time.time()
            self._pnl_history.append((now, current_pnl))

            # Keep only last 5 minutes of history
            cutoff = now - 300
            self._pnl_history = [
                (t, p) for t, p in self._pnl_history if t > cutoff
            ]

            # Check daily PnL threshold
            pnl_pct = current_pnl / initial_equity if initial_equity > 0 else 0

            if pnl_pct < -self.config.max_pnl_loss_pct:
                await self._trigger_kill_switch(
                    KillReason.PNL_BREACH,
                    f"Daily PnL breach: {pnl_pct:.2%} (threshold: -{self.config.max_pnl_loss_pct:.2%})"
                )
                return

            # Check 1-minute PnL change
            one_min_ago = now - 60
            old_pnl_entries = [p for t, p in self._pnl_history if t < one_min_ago]

            if old_pnl_entries:
                old_pnl = old_pnl_entries[-1]
                pnl_change = current_pnl - old_pnl
                pnl_change_pct = pnl_change / initial_equity if initial_equity > 0 else 0

                if pnl_change_pct < -self.config.max_pnl_loss_1min_pct:
                    await self._trigger_kill_switch(
                        KillReason.RAPID_LOSS,
                        f"Rapid loss: {pnl_change_pct:.2%} in 1 min "
                        f"(threshold: -{self.config.max_pnl_loss_1min_pct:.2%})"
                    )

            self._last_pnl = current_pnl

        except Exception as e:
            logger.error(f"PnL check error: {e}")

    async def _check_drawdown(self) -> None:
        """Check drawdown against threshold."""
        if not self._state_store:
            return

        try:
            risk_state = await self._state_store.get_risk_state()

            if risk_state.current_drawdown > self.config.max_drawdown_pct:
                await self._trigger_kill_switch(
                    KillReason.DRAWDOWN_BREACH,
                    f"Drawdown breach: {risk_state.current_drawdown:.2%} "
                    f"(threshold: {self.config.max_drawdown_pct:.2%})"
                )

        except Exception as e:
            logger.error(f"Drawdown check error: {e}")

    async def _trigger_kill_switch(
        self,
        reason: KillReason,
        details: str,
    ) -> None:
        """
        Trigger the emergency kill switch.

        This will:
        1. Set kill switch state in Redis
        2. Publish kill switch message
        3. Cancel all pending orders
        4. Close all positions
        5. Send alert notifications
        """
        if self._kill_switch_active:
            return  # Already triggered

        self._kill_switch_active = True

        logger.critical(f"KILL SWITCH TRIGGERED: {reason.value} - {details}")

        try:
            # Set kill switch in Redis
            if self._state_store:
                await self._state_store.set_kill_switch(
                    active=True,
                    reason=f"{reason.value}: {details}",
                )

            # Publish kill switch message
            if self._message_bus:
                message = Message(
                    type=MessageType.KILL_SWITCH,
                    channel=Channel.SYSTEM,
                    payload={
                        "reason": reason.value,
                        "details": details,
                        "timestamp": time.time(),
                        "triggered_by": "watchdog",
                    },
                    priority=MessagePriority.CRITICAL,
                    source="watchdog",
                )
                await self._message_bus.publish(message)

            # Send alert
            if self.config.alert_callback:
                await self.config.alert_callback(
                    f"KILL SWITCH: {reason.value}",
                    details,
                )

            # Execute emergency procedures
            await self._execute_emergency_procedures()

        except Exception as e:
            logger.error(f"Kill switch execution error: {e}")

    async def _execute_emergency_procedures(self) -> None:
        """
        Execute emergency procedures directly via broker API.

        This bypasses the normal trading system to ensure execution
        even if the main system is down.
        """
        logger.critical("Executing emergency procedures")

        # Import broker for direct API access
        try:
            from execution.broker import AlpacaBroker

            # Create direct broker connection
            broker = AlpacaBroker()
            await broker.connect()

            # Cancel all orders
            logger.critical("Cancelling all orders...")
            await broker.cancel_all_orders()

            # Close all positions
            logger.critical("Closing all positions...")
            await broker.close_all_positions()

            await broker.disconnect()

            logger.critical("Emergency procedures completed")

        except ImportError:
            logger.error("Could not import broker for emergency procedures")
        except Exception as e:
            logger.error(f"Emergency procedure error: {e}")

    async def manual_kill_switch(self, reason: str) -> None:
        """
        Manually trigger kill switch.

        Args:
            reason: Reason for manual kill switch
        """
        await self._trigger_kill_switch(
            KillReason.MANUAL,
            f"Manual trigger: {reason}",
        )

    async def reset_kill_switch(self) -> None:
        """Reset the kill switch (after manual review)."""
        self._kill_switch_active = False

        if self._state_store:
            await self._state_store.set_kill_switch(active=False)

        logger.info("Kill switch reset")

    def get_status(self) -> dict[str, Any]:
        """Get current monitor status."""
        return {
            "running": self._running,
            "kill_switch_active": self._kill_switch_active,
            "service_status": {
                svc: {
                    "last_heartbeat": ts,
                    "is_healthy": time.time() - ts < self.config.heartbeat_timeout
                    if ts > 0 else False,
                }
                for svc, ts in self._service_status.items()
            },
            "last_pnl": self._last_pnl,
            "config": {
                "check_interval": self.config.check_interval,
                "heartbeat_timeout": self.config.heartbeat_timeout,
                "max_pnl_loss_pct": self.config.max_pnl_loss_pct,
                "max_drawdown_pct": self.config.max_drawdown_pct,
            },
        }


# =============================================================================
# STANDALONE WATCHDOG SCRIPT
# =============================================================================

async def run_standalone_watchdog(
    redis_url: str = "redis://localhost:6379",
    config: HeartbeatConfig | None = None,
) -> None:
    """
    Run the watchdog as a standalone process.

    This should be run on a SEPARATE server from the main trading system.

    Example:
        # In a separate terminal/server:
        python -c "
        import asyncio
        from infrastructure.heartbeat import run_standalone_watchdog
        asyncio.run(run_standalone_watchdog())
        "
    """
    config = config or HeartbeatConfig()
    monitor = HeartbeatMonitor(config=config, redis_url=redis_url)

    try:
        await monitor.start()

        # Run forever
        while True:
            await asyncio.sleep(60)
            status = monitor.get_status()
            logger.info(f"Watchdog status: {status}")

    except KeyboardInterrupt:
        logger.info("Watchdog interrupted")
    finally:
        await monitor.stop()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "HeartbeatConfig",
    "KillReason",
    "HeartbeatMonitor",
    "run_standalone_watchdog",
]
