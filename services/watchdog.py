"""
Watchdog Service
================

Independent monitoring service that runs on a separate server/process.
Monitors system health and triggers emergency kill switch when needed.

This service is CRITICAL for risk management - it must survive
even if all other services crash.

Author: AlphaTrade Platform
Version: 3.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import os

from config.settings import get_logger
from infrastructure.heartbeat import HeartbeatMonitor, HeartbeatConfig
from infrastructure.message_bus import RedisMessageBus
from infrastructure.state_store import RedisStateStore

logger = get_logger(__name__)


class WatchdogService:
    """
    Standalone watchdog service for system monitoring.

    Should run on a SEPARATE server from the main trading system
    to ensure it can act even if the main system fails.

    Example:
        service = WatchdogService(
            redis_url="redis://localhost:6379",
            max_pnl_loss_pct=0.03  # 3% max loss
        )
        await service.run_forever()
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        max_pnl_loss_pct: float = 0.05,
        max_pnl_loss_1min_pct: float = 0.02,
        max_drawdown_pct: float = 0.10,
        check_interval: float = 5.0,
        heartbeat_timeout: float = 30.0,
    ):
        """
        Initialize watchdog service.

        Args:
            redis_url: Redis connection URL
            max_pnl_loss_pct: Max daily loss before kill switch
            max_pnl_loss_1min_pct: Max 1-minute loss before kill switch
            max_drawdown_pct: Max drawdown before kill switch
            check_interval: Seconds between health checks
            heartbeat_timeout: Seconds before service considered dead
        """
        self.redis_url = redis_url

        self._config = HeartbeatConfig(
            check_interval=check_interval,
            heartbeat_timeout=heartbeat_timeout,
            max_pnl_loss_pct=max_pnl_loss_pct,
            max_pnl_loss_1min_pct=max_pnl_loss_1min_pct,
            max_drawdown_pct=max_drawdown_pct,
            required_services=[
                "strategy_engine",
                "risk_engine",
                "oems",
            ],
            enable_auto_kill=True,
        )

        self._monitor: HeartbeatMonitor | None = None
        self._message_bus: RedisMessageBus | None = None
        self._state_store: RedisStateStore | None = None
        self._running = False

    async def start(self) -> None:
        """Start the watchdog service."""
        logger.info("Starting watchdog service")

        try:
            # Create message bus
            self._message_bus = RedisMessageBus(
                redis_url=self.redis_url,
                service_name="watchdog",
            )
            await self._message_bus.connect()

            # Create state store
            self._state_store = RedisStateStore(
                redis_url=self.redis_url,
                service_name="watchdog",
            )
            await self._state_store.connect()

            # Create heartbeat monitor
            self._monitor = HeartbeatMonitor(
                config=self._config,
                redis_url=self.redis_url,
                message_bus=self._message_bus,
                state_store=self._state_store,
            )
            await self._monitor.start()

            self._running = True
            logger.info("Watchdog service started")

        except Exception as e:
            logger.error(f"Failed to start watchdog: {e}")
            raise

    async def stop(self) -> None:
        """Stop the watchdog service."""
        logger.info("Stopping watchdog service")
        self._running = False

        if self._monitor:
            await self._monitor.stop()

        if self._message_bus:
            await self._message_bus.disconnect()

        if self._state_store:
            await self._state_store.disconnect()

        logger.info("Watchdog service stopped")

    async def run_forever(self) -> None:
        """Run the watchdog until interrupted."""
        await self.start()

        try:
            while self._running:
                # Log status periodically
                if self._monitor:
                    status = self._monitor.get_status()
                    logger.info(
                        f"Watchdog status: kill_switch={status['kill_switch_active']}, "
                        f"services={len(status['service_status'])}"
                    )

                await asyncio.sleep(60)

        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def trigger_kill_switch(self, reason: str) -> None:
        """Manually trigger the kill switch."""
        if self._monitor:
            await self._monitor.manual_kill_switch(reason)

    async def reset_kill_switch(self) -> None:
        """Reset the kill switch."""
        if self._monitor:
            await self._monitor.reset_kill_switch()

    def get_status(self) -> dict:
        """Get watchdog status."""
        if self._monitor:
            return self._monitor.get_status()
        return {"running": False}


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main():
    """Run watchdog service."""
    service = WatchdogService(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        max_pnl_loss_pct=float(os.getenv("MAX_PNL_LOSS_PCT", "0.05")),
        max_drawdown_pct=float(os.getenv("MAX_DRAWDOWN_PCT", "0.10")),
    )

    await service.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
