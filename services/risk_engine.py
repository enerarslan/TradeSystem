"""
Risk Engine Service
===================

Intercepts all trading signals, validates them against risk limits,
and either approves or rejects them before they reach the OEMS.

Responsibilities:
- Signal validation
- Position limit checking
- PnL limit enforcement
- Drawdown monitoring
- Exposure management
- Risk state persistence

Author: AlphaTrade Platform
Version: 3.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import os

from config.settings import get_logger
from infrastructure.message_bus import (
    Message,
    MessageType,
    Channel,
    MessagePriority,
)
from infrastructure.state_store import StateKey, RiskState, PositionState
from infrastructure.service_registry import ServiceType
from services.base_service import BaseService, ServiceConfig

logger = get_logger(__name__)


@dataclass
class RiskConfig(ServiceConfig):
    """Configuration for risk engine service."""
    name: str = "risk_engine"
    service_type: ServiceType = ServiceType.RISK_ENGINE

    # Position limits
    max_position_size: float = 10000.0  # Max $ per position
    max_position_pct: float = 0.10  # Max % of portfolio per position
    max_positions: int = 10  # Max concurrent positions
    max_sector_exposure: float = 0.30  # Max % in single sector

    # PnL limits
    max_daily_loss: float = 5000.0  # Max daily loss $
    max_daily_loss_pct: float = 0.05  # Max daily loss %
    max_drawdown_pct: float = 0.10  # Max drawdown %

    # Trade limits
    max_trades_per_day: int = 100
    max_trades_per_minute: int = 10
    min_time_between_trades: float = 1.0  # seconds

    # Order limits
    max_order_size: float = 5000.0  # Max $ per order
    max_order_pct: float = 0.05  # Max % of portfolio per order

    # Initial capital
    initial_capital: float = 100000.0


class RiskService(BaseService):
    """
    Risk Engine Service for signal validation and risk management.

    Intercepts all signals from strategy engine and validates them
    against configured risk limits before forwarding to OEMS.

    All risk state is persisted in Redis for fault tolerance.

    Example:
        config = RiskConfig(
            max_position_size=10000,
            max_daily_loss=5000
        )

        service = RiskService(config)
        await service.run_forever()
    """

    def __init__(self, config: RiskConfig | None = None):
        """Initialize risk engine."""
        config = config or RiskConfig()
        super().__init__(config)

        self.config: RiskConfig = config

        # Risk state (loaded from Redis on startup)
        self._risk_state: RiskState | None = None
        self._positions: dict[str, PositionState] = {}

        # Rate limiting
        self._trades_today = 0
        self._trades_this_minute = 0
        self._last_minute = 0
        self._last_trade_time: dict[str, float] = {}

        # Statistics
        self._signals_received = 0
        self._signals_approved = 0
        self._signals_rejected = 0

    async def _on_start(self) -> None:
        """Start risk engine."""
        logger.info("Starting risk engine")

        # Load risk state from Redis
        await self._load_risk_state()

        # Subscribe to signals
        await self.subscribe(Channel.SIGNALS, self._handle_signal)

        # Subscribe to fills for position updates
        await self.subscribe(Channel.FILLS, self._handle_fill)

        # Start risk monitoring loop
        self.add_background_task(self._risk_monitoring_loop())

        logger.info("Risk engine started")

    async def _on_stop(self) -> None:
        """Stop risk engine."""
        # Persist final state
        await self._save_risk_state()

        logger.info(
            f"Risk engine stopped. "
            f"Signals: {self._signals_received} received, "
            f"{self._signals_approved} approved, "
            f"{self._signals_rejected} rejected"
        )

    async def _load_risk_state(self) -> None:
        """Load risk state from Redis."""
        if not self._state_store:
            self._risk_state = RiskState(current_equity=self.config.initial_capital)
            return

        try:
            self._risk_state = await self._state_store.get_risk_state()
            self._positions = await self._state_store.get_all_positions()

            # Initialize if empty
            if self._risk_state.current_equity == 0:
                self._risk_state.current_equity = self.config.initial_capital
                self._risk_state.high_water_mark = self.config.initial_capital
                await self._save_risk_state()

            logger.info(
                f"Loaded risk state: equity=${self._risk_state.current_equity:,.2f}, "
                f"positions={len(self._positions)}"
            )

        except Exception as e:
            logger.error(f"Failed to load risk state: {e}")
            self._risk_state = RiskState(current_equity=self.config.initial_capital)

    async def _save_risk_state(self) -> None:
        """Save risk state to Redis."""
        if not self._state_store or not self._risk_state:
            return

        try:
            await self._state_store.set_risk_state(self._risk_state)
        except Exception as e:
            logger.error(f"Failed to save risk state: {e}")

    async def _handle_signal(self, message: Message) -> None:
        """Handle incoming trading signal."""
        self._signals_received += 1

        try:
            payload = message.payload
            symbol = payload.get("symbol")
            direction = payload.get("direction")
            strength = payload.get("strength", 0)
            price = payload.get("price", 0)

            logger.debug(
                f"Received signal: {symbol} direction={direction} "
                f"strength={strength:.2%} price=${price:.2f}"
            )

            # Validate signal
            approved, rejection_reason = await self._validate_signal(
                symbol=symbol,
                direction=direction,
                strength=strength,
                price=price,
            )

            if approved:
                await self._approve_signal(message)
            else:
                await self._reject_signal(message, rejection_reason)

        except Exception as e:
            logger.error(f"Error handling signal: {e}")
            await self._reject_signal(message, f"Internal error: {e}")

    async def _validate_signal(
        self,
        symbol: str,
        direction: int,
        strength: float,
        price: float,
    ) -> tuple[bool, str]:
        """
        Validate signal against all risk rules.

        Returns:
            Tuple of (approved, rejection_reason)
        """
        if not self._risk_state:
            return False, "Risk state not initialized"

        # Check kill switch
        if self._state_store:
            if await self._state_store.is_kill_switch_active():
                return False, "Kill switch active"

        # Check daily PnL limit
        if self._risk_state.daily_pnl <= -self.config.max_daily_loss:
            return False, f"Daily loss limit reached: ${-self._risk_state.daily_pnl:,.2f}"

        daily_loss_pct = abs(self._risk_state.daily_pnl) / self._risk_state.current_equity
        if daily_loss_pct >= self.config.max_daily_loss_pct:
            return False, f"Daily loss % limit reached: {daily_loss_pct:.2%}"

        # Check drawdown
        if self._risk_state.current_drawdown >= self.config.max_drawdown_pct:
            return False, f"Drawdown limit reached: {self._risk_state.current_drawdown:.2%}"

        # Check trade rate limits
        current_minute = int(time.time() / 60)
        if current_minute != self._last_minute:
            self._trades_this_minute = 0
            self._last_minute = current_minute

        if self._trades_this_minute >= self.config.max_trades_per_minute:
            return False, f"Trade rate limit: {self.config.max_trades_per_minute}/min"

        if self._trades_today >= self.config.max_trades_per_day:
            return False, f"Daily trade limit: {self.config.max_trades_per_day}/day"

        # Check time between trades for same symbol
        last_trade = self._last_trade_time.get(symbol, 0)
        if time.time() - last_trade < self.config.min_time_between_trades:
            return False, f"Min time between trades: {self.config.min_time_between_trades}s"

        # Check position limits
        current_position = self._positions.get(symbol)
        is_closing = (
            current_position and
            current_position.quantity != 0 and
            (current_position.quantity > 0) != (direction > 0)
        )

        if not is_closing:
            # Opening or adding to position
            # Check max positions
            open_positions = len([p for p in self._positions.values() if p.quantity != 0])
            if open_positions >= self.config.max_positions:
                return False, f"Max positions reached: {self.config.max_positions}"

            # Check position size limit
            position_size = self._calculate_position_size(price)
            if position_size > self.config.max_position_size:
                return False, f"Position size limit: ${self.config.max_position_size:,.0f}"

            # Check position % of portfolio
            position_pct = position_size / self._risk_state.current_equity
            if position_pct > self.config.max_position_pct:
                return False, f"Position % limit: {self.config.max_position_pct:.0%}"

        # All checks passed
        return True, ""

    def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on risk parameters."""
        # Use the smaller of fixed size or % of portfolio
        fixed_size = self.config.max_order_size
        pct_size = self._risk_state.current_equity * self.config.max_order_pct

        return min(fixed_size, pct_size)

    async def _approve_signal(self, message: Message) -> None:
        """Approve signal and forward to OEMS."""
        self._signals_approved += 1
        self._trades_this_minute += 1
        self._trades_today += 1

        payload = message.payload
        symbol = payload.get("symbol")
        self._last_trade_time[symbol] = time.time()

        # Calculate order size
        price = payload.get("price", 0)
        position_size = self._calculate_position_size(price)
        quantity = position_size / price if price > 0 else 0

        # Create approved signal message
        approved_message = Message(
            type=MessageType.SIGNAL_APPROVED,
            channel=Channel.ORDERS,
            payload={
                **payload,
                "quantity": quantity,
                "position_size": position_size,
                "approved_at": datetime.now().isoformat(),
                "approved_by": self.name,
            },
            priority=MessagePriority.CRITICAL,
            source=self.name,
            correlation_id=message.id,
        )

        await self.publish(approved_message)

        logger.info(
            f"Signal APPROVED: {symbol} qty={quantity:.2f} size=${position_size:,.0f}"
        )

    async def _reject_signal(self, message: Message, reason: str) -> None:
        """Reject signal and publish rejection."""
        self._signals_rejected += 1

        payload = message.payload
        symbol = payload.get("symbol", "UNKNOWN")

        # Create rejection message
        rejection = Message(
            type=MessageType.SIGNAL_REJECTED,
            channel=Channel.SIGNALS,
            payload={
                **payload,
                "rejection_reason": reason,
                "rejected_at": datetime.now().isoformat(),
                "rejected_by": self.name,
            },
            priority=MessagePriority.NORMAL,
            source=self.name,
            correlation_id=message.id,
        )

        await self.publish(rejection)

        logger.warning(f"Signal REJECTED: {symbol} - {reason}")

    async def _handle_fill(self, message: Message) -> None:
        """Handle fill events for position tracking."""
        try:
            payload = message.payload
            symbol = payload.get("symbol")
            side = payload.get("side")
            quantity = payload.get("quantity", 0)
            price = payload.get("price", 0)
            pnl = payload.get("pnl", 0)

            # Update position
            if symbol not in self._positions:
                self._positions[symbol] = PositionState(
                    symbol=symbol,
                    quantity=0,
                    avg_entry_price=0,
                    current_price=price,
                    unrealized_pnl=0,
                    realized_pnl=0,
                    side="flat",
                    opened_at=time.time(),
                    last_updated=time.time(),
                )

            position = self._positions[symbol]

            # Update position quantity
            if side == "buy":
                if position.quantity >= 0:
                    # Adding to long
                    total_cost = position.quantity * position.avg_entry_price + quantity * price
                    position.quantity += quantity
                    position.avg_entry_price = total_cost / position.quantity if position.quantity > 0 else 0
                else:
                    # Closing/reducing short
                    position.quantity += quantity
                    position.realized_pnl += pnl
            else:  # sell
                if position.quantity <= 0:
                    # Adding to short
                    total_cost = abs(position.quantity) * position.avg_entry_price + quantity * price
                    position.quantity -= quantity
                    position.avg_entry_price = total_cost / abs(position.quantity) if position.quantity != 0 else 0
                else:
                    # Closing/reducing long
                    position.quantity -= quantity
                    position.realized_pnl += pnl

            position.current_price = price
            position.side = "long" if position.quantity > 0 else "short" if position.quantity < 0 else "flat"
            position.last_updated = time.time()

            # Update unrealized PnL
            if position.quantity != 0:
                if position.quantity > 0:
                    position.unrealized_pnl = (price - position.avg_entry_price) * position.quantity
                else:
                    position.unrealized_pnl = (position.avg_entry_price - price) * abs(position.quantity)
            else:
                position.unrealized_pnl = 0

            # Save to Redis
            if self._state_store:
                if position.quantity != 0:
                    await self._state_store.set_position(position)
                else:
                    await self._state_store.delete_position(symbol)

            # Update risk state
            await self._update_risk_state(pnl)

            logger.debug(
                f"Position updated: {symbol} qty={position.quantity:.2f} "
                f"pnl={position.unrealized_pnl:+.2f}"
            )

        except Exception as e:
            logger.error(f"Error handling fill: {e}")

    async def _update_risk_state(self, realized_pnl: float = 0) -> None:
        """Update risk state after trade."""
        if not self._risk_state:
            return

        # Calculate total unrealized PnL
        total_unrealized = sum(p.unrealized_pnl for p in self._positions.values())

        # Update via state store for atomic operation
        if self._state_store:
            self._risk_state = await self._state_store.update_pnl(
                realized=realized_pnl,
                unrealized=total_unrealized,
            )
        else:
            self._risk_state.realized_pnl += realized_pnl
            self._risk_state.daily_pnl += realized_pnl
            self._risk_state.unrealized_pnl = total_unrealized

    async def _risk_monitoring_loop(self) -> None:
        """Periodic risk monitoring and state updates."""
        while self._running:
            try:
                # Reload state from Redis (in case other services updated it)
                if self._state_store:
                    self._risk_state = await self._state_store.get_risk_state()
                    self._positions = await self._state_store.get_all_positions()

                # Check for risk breaches
                await self._check_risk_breaches()

                # Publish risk update
                await self._publish_risk_update()

                await asyncio.sleep(5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Risk monitoring error: {e}")
                await asyncio.sleep(1)

    async def _check_risk_breaches(self) -> None:
        """Check for risk limit breaches."""
        if not self._risk_state:
            return

        breaches = []

        # Daily loss
        if self._risk_state.daily_pnl <= -self.config.max_daily_loss:
            breaches.append(f"Daily loss: ${-self._risk_state.daily_pnl:,.2f}")

        # Drawdown
        if self._risk_state.current_drawdown >= self.config.max_drawdown_pct:
            breaches.append(f"Drawdown: {self._risk_state.current_drawdown:.2%}")

        if breaches:
            logger.error(f"Risk breaches detected: {breaches}")

            # Publish risk breach message
            message = Message(
                type=MessageType.RISK_BREACH,
                channel=Channel.RISK,
                payload={
                    "breaches": breaches,
                    "risk_state": self._risk_state.to_dict(),
                    "timestamp": datetime.now().isoformat(),
                },
                priority=MessagePriority.CRITICAL,
                source=self.name,
            )
            await self.publish(message)

    async def _publish_risk_update(self) -> None:
        """Publish periodic risk state update."""
        if not self._risk_state:
            return

        message = Message(
            type=MessageType.RISK_UPDATE,
            channel=Channel.RISK,
            payload={
                "risk_state": self._risk_state.to_dict(),
                "position_count": len([p for p in self._positions.values() if p.quantity != 0]),
                "signals_today": self._signals_received,
                "approved_today": self._signals_approved,
                "rejected_today": self._signals_rejected,
            },
            priority=MessagePriority.LOW,
            source=self.name,
        )
        await self.publish(message)

    async def _handle_kill_switch(self, message: Message) -> None:
        """Handle kill switch - risk engine specific."""
        logger.critical("Risk engine received kill switch")

        # Reject all new signals
        self._signals_received = float("inf")  # Block new signals

        # Parent handler will stop the service
        await super()._handle_kill_switch(message)

    def get_status(self) -> dict[str, Any]:
        """Get service status."""
        status = super().get_status()
        status.update({
            "risk_state": self._risk_state.to_dict() if self._risk_state else None,
            "position_count": len([p for p in self._positions.values() if p.quantity != 0]),
            "signals_received": self._signals_received,
            "signals_approved": self._signals_approved,
            "signals_rejected": self._signals_rejected,
            "trades_today": self._trades_today,
        })
        return status


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main():
    """Run risk engine service."""
    config = RiskConfig(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        initial_capital=float(os.getenv("INITIAL_CAPITAL", "100000")),
    )

    service = RiskService(config)
    await service.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
