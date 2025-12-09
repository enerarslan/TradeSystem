"""
Smart Order Routing Module
==========================

Advanced execution algorithms for minimizing market impact and slippage.
JPMorgan-level execution quality.

Algorithms:
- TWAP: Time-Weighted Average Price
- VWAP: Volume-Weighted Average Price
- Iceberg: Hidden liquidity orders
- Pegged: Dynamic price adjustment
- POV: Percentage of Volume

Author: AlphaTrade Platform
Version: 3.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Awaitable
from uuid import uuid4

import numpy as np

from config.settings import get_logger

logger = get_logger(__name__)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class AlgoType(str, Enum):
    """Execution algorithm types."""
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"
    PEGGED = "pegged"
    POV = "pov"  # Percentage of Volume
    IMPLEMENTATION_SHORTFALL = "is"
    ADAPTIVE = "adaptive"


class OrderSide(str, Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class PegType(str, Enum):
    """Peg types for pegged orders."""
    MIDPOINT = "midpoint"
    BID = "bid"
    ASK = "ask"
    LAST = "last"
    VWAP = "vwap"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AlgoOrder:
    """
    Algorithmic order specification.

    Attributes:
        order_id: Unique order identifier
        symbol: Trading symbol
        side: Buy or sell
        total_quantity: Total quantity to execute
        algo_type: Algorithm type
        start_time: Algorithm start time
        end_time: Algorithm end time
        limit_price: Maximum/minimum price
        params: Algorithm-specific parameters
        filled_quantity: Quantity filled so far
        avg_fill_price: Average fill price
        status: Order status
        child_orders: List of child order IDs
    """
    order_id: str = field(default_factory=lambda: str(uuid4())[:8])
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    total_quantity: float = 0.0
    algo_type: AlgoType = AlgoType.TWAP
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    limit_price: float | None = None
    params: dict[str, Any] = field(default_factory=dict)
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    status: str = "pending"
    child_orders: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def remaining_quantity(self) -> float:
        """Get remaining quantity to fill."""
        return self.total_quantity - self.filled_quantity

    @property
    def fill_rate(self) -> float:
        """Get fill rate percentage."""
        if self.total_quantity <= 0:
            return 0.0
        return self.filled_quantity / self.total_quantity

    @property
    def is_complete(self) -> bool:
        """Check if order is complete."""
        return self.remaining_quantity <= 0 or self.status in ("filled", "cancelled")


@dataclass
class ChildOrder:
    """
    Child order for algo execution.

    Represents individual orders placed as part of an algorithm.
    """
    order_id: str = field(default_factory=lambda: str(uuid4())[:8])
    parent_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    order_type: str = "limit"
    limit_price: float | None = None
    status: str = "pending"
    filled_quantity: float = 0.0
    fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: datetime | None = None


@dataclass
class MarketState:
    """
    Current market state for execution decisions.

    Attributes:
        bid_price: Best bid price
        ask_price: Best ask price
        last_price: Last trade price
        bid_size: Best bid size
        ask_size: Best ask size
        volume: Recent volume
        vwap: Current VWAP
    """
    bid_price: float = 0.0
    ask_price: float = 0.0
    last_price: float = 0.0
    bid_size: int = 0
    ask_size: int = 0
    volume: int = 0
    vwap: float = 0.0

    @property
    def mid_price(self) -> float:
        """Get midpoint price."""
        return (self.bid_price + self.ask_price) / 2

    @property
    def spread(self) -> float:
        """Get bid-ask spread."""
        return self.ask_price - self.bid_price

    @property
    def spread_bps(self) -> float:
        """Get spread in basis points."""
        if self.mid_price <= 0:
            return 0.0
        return (self.spread / self.mid_price) * 10000


# =============================================================================
# BASE ALGORITHM
# =============================================================================

# Order submission callback type
OrderCallback = Callable[[ChildOrder], Awaitable[bool]]


class ExecutionAlgorithm(ABC):
    """
    Abstract base class for execution algorithms.

    Subclasses must implement:
    - generate_schedule(): Create execution schedule
    - get_next_slice(): Get next order slice
    - on_fill(): Handle fill events
    """

    def __init__(
        self,
        algo_order: AlgoOrder,
        submit_order: OrderCallback,
        get_market_state: Callable[[str], Awaitable[MarketState]],
    ):
        """
        Initialize algorithm.

        Args:
            algo_order: Parent algo order
            submit_order: Callback to submit child orders
            get_market_state: Callback to get market data
        """
        self.order = algo_order
        self._submit_order = submit_order
        self._get_market_state = get_market_state

        self._running = False
        self._paused = False
        self._child_orders: dict[str, ChildOrder] = {}
        self._execution_task: asyncio.Task | None = None

    @property
    def is_running(self) -> bool:
        """Check if algorithm is running."""
        return self._running

    @abstractmethod
    def generate_schedule(self) -> list[tuple[datetime, float]]:
        """
        Generate execution schedule.

        Returns:
            List of (time, quantity) tuples
        """
        pass

    @abstractmethod
    async def get_next_slice(self, market: MarketState) -> ChildOrder | None:
        """
        Get next order slice based on current market.

        Args:
            market: Current market state

        Returns:
            Child order to submit or None
        """
        pass

    @abstractmethod
    async def on_fill(self, child_order: ChildOrder, fill_price: float, fill_qty: float) -> None:
        """
        Handle fill event.

        Args:
            child_order: Child order that was filled
            fill_price: Fill price
            fill_qty: Fill quantity
        """
        pass

    async def start(self) -> None:
        """Start the algorithm."""
        self._running = True
        self.order.status = "running"
        self._execution_task = asyncio.create_task(self._run())
        logger.info(f"Started {self.order.algo_type.value} for {self.order.symbol}")

    async def stop(self) -> None:
        """Stop the algorithm."""
        self._running = False
        if self._execution_task:
            self._execution_task.cancel()
            try:
                await self._execution_task
            except asyncio.CancelledError:
                pass

        self.order.status = "stopped"
        logger.info(f"Stopped {self.order.algo_type.value} for {self.order.symbol}")

    async def pause(self) -> None:
        """Pause the algorithm."""
        self._paused = True
        self.order.status = "paused"

    async def resume(self) -> None:
        """Resume the algorithm."""
        self._paused = False
        self.order.status = "running"

    async def _run(self) -> None:
        """Main execution loop."""
        logger.info(f"Running {self.order.algo_type.value} algorithm")

        while self._running and not self.order.is_complete:
            try:
                if self._paused:
                    await asyncio.sleep(0.1)
                    continue

                # Get market state
                market = await self._get_market_state(self.order.symbol)

                # Check limit price
                if self.order.limit_price:
                    if self.order.side == OrderSide.BUY and market.ask_price > self.order.limit_price:
                        await asyncio.sleep(0.1)
                        continue
                    if self.order.side == OrderSide.SELL and market.bid_price < self.order.limit_price:
                        await asyncio.sleep(0.1)
                        continue

                # Get next slice
                child = await self.get_next_slice(market)

                if child:
                    # Submit order
                    success = await self._submit_order(child)
                    if success:
                        self._child_orders[child.order_id] = child
                        self.order.child_orders.append(child.order_id)

                await asyncio.sleep(0.1)  # Rate limit

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Algorithm error: {e}")
                await asyncio.sleep(1)

        # Update final status
        if self.order.remaining_quantity <= 0:
            self.order.status = "filled"
        elif not self._running:
            self.order.status = "cancelled"

        logger.info(
            f"Algorithm complete: {self.order.symbol} "
            f"filled={self.order.filled_quantity:.2f}/{self.order.total_quantity:.2f}"
        )


# =============================================================================
# TWAP ALGORITHM
# =============================================================================

class TWAPAlgorithm(ExecutionAlgorithm):
    """
    Time-Weighted Average Price Algorithm.

    Slices the order into equal parts over a time period.
    Ideal for minimizing market impact on large orders.

    Parameters:
        duration_minutes: Total execution duration
        slice_interval_seconds: Time between slices
        randomize: Add randomization to timing
    """

    def __init__(
        self,
        algo_order: AlgoOrder,
        submit_order: OrderCallback,
        get_market_state: Callable[[str], Awaitable[MarketState]],
    ):
        super().__init__(algo_order, submit_order, get_market_state)

        # Get parameters with defaults
        self.duration = self.order.params.get("duration_minutes", 10)
        self.slice_interval = self.order.params.get("slice_interval_seconds", 30)
        self.randomize = self.order.params.get("randomize", True)

        # Calculate schedule
        self._schedule = self.generate_schedule()
        self._current_slice = 0
        self._last_slice_time = None

    def generate_schedule(self) -> list[tuple[datetime, float]]:
        """Generate TWAP schedule."""
        schedule = []

        duration_seconds = self.duration * 60
        num_slices = max(1, int(duration_seconds / self.slice_interval))
        slice_qty = self.order.total_quantity / num_slices

        start = self.order.start_time or datetime.now()

        for i in range(num_slices):
            offset = i * self.slice_interval

            # Add randomization
            if self.randomize:
                offset += np.random.uniform(-5, 5)

            slice_time = start + timedelta(seconds=offset)
            schedule.append((slice_time, slice_qty))

        return schedule

    async def get_next_slice(self, market: MarketState) -> ChildOrder | None:
        """Get next TWAP slice."""
        if self._current_slice >= len(self._schedule):
            return None

        scheduled_time, qty = self._schedule[self._current_slice]

        # Check if it's time for next slice
        now = datetime.now()
        if now < scheduled_time:
            return None

        # Check minimum time between slices
        if self._last_slice_time:
            min_interval = self.slice_interval * 0.5
            if (now - self._last_slice_time).total_seconds() < min_interval:
                return None

        # Calculate price
        if self.order.side == OrderSide.BUY:
            price = market.ask_price * 1.001  # Small buffer
        else:
            price = market.bid_price * 0.999

        # Create child order
        child = ChildOrder(
            parent_id=self.order.order_id,
            symbol=self.order.symbol,
            side=self.order.side,
            quantity=min(qty, self.order.remaining_quantity),
            order_type="limit",
            limit_price=price,
        )

        self._current_slice += 1
        self._last_slice_time = now

        logger.debug(
            f"TWAP slice {self._current_slice}/{len(self._schedule)}: "
            f"{child.quantity:.2f} @ ${price:.2f}"
        )

        return child

    async def on_fill(self, child_order: ChildOrder, fill_price: float, fill_qty: float) -> None:
        """Handle fill."""
        # Update parent order
        old_qty = self.order.filled_quantity
        old_value = old_qty * self.order.avg_fill_price

        new_qty = old_qty + fill_qty
        new_value = old_value + (fill_qty * fill_price)

        self.order.filled_quantity = new_qty
        self.order.avg_fill_price = new_value / new_qty if new_qty > 0 else 0

        logger.debug(f"TWAP fill: {fill_qty:.2f} @ ${fill_price:.2f}")


# =============================================================================
# VWAP ALGORITHM
# =============================================================================

class VWAPAlgorithm(ExecutionAlgorithm):
    """
    Volume-Weighted Average Price Algorithm.

    Slices the order based on historical volume profile.
    Aims to achieve execution at or better than VWAP.

    Parameters:
        duration_minutes: Total execution duration
        volume_profile: Historical volume by time bucket
        participation_rate: Max % of volume to participate
    """

    def __init__(
        self,
        algo_order: AlgoOrder,
        submit_order: OrderCallback,
        get_market_state: Callable[[str], Awaitable[MarketState]],
    ):
        super().__init__(algo_order, submit_order, get_market_state)

        self.duration = self.order.params.get("duration_minutes", 30)
        self.participation_rate = self.order.params.get("participation_rate", 0.10)

        # Default volume profile (U-shaped intraday pattern)
        self.volume_profile = self.order.params.get("volume_profile", self._default_profile())

        self._schedule = self.generate_schedule()
        self._current_slice = 0
        self._cumulative_volume = 0

    def _default_profile(self) -> list[float]:
        """Generate default U-shaped volume profile."""
        # Simulate typical intraday volume (higher at open/close)
        hours = np.linspace(0, 1, 12)
        profile = 1 + 0.5 * np.cos(2 * np.pi * (hours - 0.5))
        return (profile / profile.sum()).tolist()

    def generate_schedule(self) -> list[tuple[datetime, float]]:
        """Generate VWAP schedule based on volume profile."""
        schedule = []

        total_profile = sum(self.volume_profile)
        start = self.order.start_time or datetime.now()
        interval = (self.duration * 60) / len(self.volume_profile)

        for i, vol_pct in enumerate(self.volume_profile):
            slice_time = start + timedelta(seconds=i * interval)
            slice_qty = self.order.total_quantity * (vol_pct / total_profile)
            schedule.append((slice_time, slice_qty))

        return schedule

    async def get_next_slice(self, market: MarketState) -> ChildOrder | None:
        """Get next VWAP slice."""
        if self._current_slice >= len(self._schedule):
            return None

        scheduled_time, target_qty = self._schedule[self._current_slice]

        now = datetime.now()
        if now < scheduled_time:
            return None

        # Adjust based on actual volume
        if market.volume > 0:
            max_qty = market.volume * self.participation_rate
            qty = min(target_qty, max_qty, self.order.remaining_quantity)
        else:
            qty = min(target_qty, self.order.remaining_quantity)

        if qty <= 0:
            self._current_slice += 1
            return None

        # Calculate price relative to VWAP
        if market.vwap > 0:
            if self.order.side == OrderSide.BUY:
                # Try to buy below VWAP
                price = min(market.vwap, market.ask_price)
            else:
                # Try to sell above VWAP
                price = max(market.vwap, market.bid_price)
        else:
            price = market.mid_price

        child = ChildOrder(
            parent_id=self.order.order_id,
            symbol=self.order.symbol,
            side=self.order.side,
            quantity=qty,
            order_type="limit",
            limit_price=price,
        )

        self._current_slice += 1

        logger.debug(
            f"VWAP slice {self._current_slice}/{len(self._schedule)}: "
            f"{qty:.2f} @ ${price:.2f} (VWAP: ${market.vwap:.2f})"
        )

        return child

    async def on_fill(self, child_order: ChildOrder, fill_price: float, fill_qty: float) -> None:
        """Handle fill."""
        old_qty = self.order.filled_quantity
        old_value = old_qty * self.order.avg_fill_price

        new_qty = old_qty + fill_qty
        new_value = old_value + (fill_qty * fill_price)

        self.order.filled_quantity = new_qty
        self.order.avg_fill_price = new_value / new_qty if new_qty > 0 else 0


# =============================================================================
# ICEBERG ALGORITHM
# =============================================================================

class IcebergAlgorithm(ExecutionAlgorithm):
    """
    Iceberg Order Algorithm.

    Shows only a small "visible" quantity while hiding the total size.
    Refills automatically when visible quantity is filled.

    Parameters:
        visible_quantity: Amount to show in market
        variance: Random variance in visible size (0.0-0.5)
        price_offset: Offset from best price in ticks
    """

    def __init__(
        self,
        algo_order: AlgoOrder,
        submit_order: OrderCallback,
        get_market_state: Callable[[str], Awaitable[MarketState]],
    ):
        super().__init__(algo_order, submit_order, get_market_state)

        self.visible_qty = self.order.params.get("visible_quantity", 100)
        self.variance = self.order.params.get("variance", 0.2)
        self.price_offset = self.order.params.get("price_offset", 0)

        self._current_visible: ChildOrder | None = None
        self._pending_fill = False

    def generate_schedule(self) -> list[tuple[datetime, float]]:
        """Iceberg doesn't use a fixed schedule."""
        return []

    def _get_random_visible_qty(self) -> float:
        """Get randomized visible quantity."""
        if self.variance > 0:
            min_qty = self.visible_qty * (1 - self.variance)
            max_qty = self.visible_qty * (1 + self.variance)
            return np.random.uniform(min_qty, max_qty)
        return self.visible_qty

    async def get_next_slice(self, market: MarketState) -> ChildOrder | None:
        """Get next iceberg slice."""
        # Don't create new slice if one is pending
        if self._current_visible and self._current_visible.status == "pending":
            return None

        # Check if we have remaining quantity
        if self.order.remaining_quantity <= 0:
            return None

        # Calculate visible quantity
        visible = min(
            self._get_random_visible_qty(),
            self.order.remaining_quantity,
        )

        # Calculate price
        if self.order.side == OrderSide.BUY:
            price = market.bid_price + (self.price_offset * 0.01)
            price = min(price, market.ask_price)  # Don't cross spread
        else:
            price = market.ask_price - (self.price_offset * 0.01)
            price = max(price, market.bid_price)  # Don't cross spread

        child = ChildOrder(
            parent_id=self.order.order_id,
            symbol=self.order.symbol,
            side=self.order.side,
            quantity=visible,
            order_type="limit",
            limit_price=price,
        )

        self._current_visible = child

        logger.debug(
            f"Iceberg slice: {visible:.2f} @ ${price:.2f} "
            f"(total remaining: {self.order.remaining_quantity:.2f})"
        )

        return child

    async def on_fill(self, child_order: ChildOrder, fill_price: float, fill_qty: float) -> None:
        """Handle fill - will trigger next slice."""
        old_qty = self.order.filled_quantity
        old_value = old_qty * self.order.avg_fill_price

        new_qty = old_qty + fill_qty
        new_value = old_value + (fill_qty * fill_price)

        self.order.filled_quantity = new_qty
        self.order.avg_fill_price = new_value / new_qty if new_qty > 0 else 0

        # Mark current visible as filled
        if self._current_visible and self._current_visible.order_id == child_order.order_id:
            self._current_visible.status = "filled"
            self._current_visible = None

        logger.debug(f"Iceberg fill: {fill_qty:.2f} @ ${fill_price:.2f}")


# =============================================================================
# PEGGED ALGORITHM
# =============================================================================

class PeggedAlgorithm(ExecutionAlgorithm):
    """
    Pegged Order Algorithm.

    Dynamically adjusts limit price to track a reference price.
    Useful for capturing spread while maintaining queue position.

    Parameters:
        peg_type: What to peg to (midpoint, bid, ask, vwap)
        offset: Offset from peg price in cents
        max_deviation: Max allowed deviation before repricing
        reprice_interval: Minimum seconds between reprices
    """

    def __init__(
        self,
        algo_order: AlgoOrder,
        submit_order: OrderCallback,
        get_market_state: Callable[[str], Awaitable[MarketState]],
    ):
        super().__init__(algo_order, submit_order, get_market_state)

        self.peg_type = PegType(self.order.params.get("peg_type", "midpoint"))
        self.offset = self.order.params.get("offset", 0.0)
        self.max_deviation = self.order.params.get("max_deviation", 0.02)
        self.reprice_interval = self.order.params.get("reprice_interval", 1.0)

        self._current_order: ChildOrder | None = None
        self._last_reprice: float = 0

    def generate_schedule(self) -> list[tuple[datetime, float]]:
        """Pegged doesn't use a fixed schedule."""
        return []

    def _get_peg_price(self, market: MarketState) -> float:
        """Calculate peg reference price."""
        if self.peg_type == PegType.MIDPOINT:
            return market.mid_price
        elif self.peg_type == PegType.BID:
            return market.bid_price
        elif self.peg_type == PegType.ASK:
            return market.ask_price
        elif self.peg_type == PegType.VWAP:
            return market.vwap if market.vwap > 0 else market.mid_price
        elif self.peg_type == PegType.LAST:
            return market.last_price
        else:
            return market.mid_price

    def _calculate_price(self, market: MarketState) -> float:
        """Calculate order price based on peg."""
        peg_price = self._get_peg_price(market)

        if self.order.side == OrderSide.BUY:
            price = peg_price - self.offset
            # Don't cross spread aggressively
            price = min(price, market.ask_price - 0.01)
        else:
            price = peg_price + self.offset
            # Don't cross spread aggressively
            price = max(price, market.bid_price + 0.01)

        return round(price, 2)

    async def get_next_slice(self, market: MarketState) -> ChildOrder | None:
        """Get pegged order or reprice existing."""
        now = time.time()

        # Check if we need to create initial order
        if not self._current_order or self._current_order.status != "pending":
            price = self._calculate_price(market)

            child = ChildOrder(
                parent_id=self.order.order_id,
                symbol=self.order.symbol,
                side=self.order.side,
                quantity=self.order.remaining_quantity,
                order_type="limit",
                limit_price=price,
            )

            self._current_order = child
            self._last_reprice = now

            logger.debug(
                f"Pegged order: {child.quantity:.2f} @ ${price:.2f} "
                f"(peg: {self.peg_type.value})"
            )

            return child

        # Check if we need to reprice
        if now - self._last_reprice < self.reprice_interval:
            return None

        current_price = self._current_order.limit_price
        target_price = self._calculate_price(market)

        deviation = abs(target_price - current_price) / current_price if current_price > 0 else 0

        if deviation > self.max_deviation:
            # Cancel and replace
            logger.debug(
                f"Pegged reprice: ${current_price:.2f} -> ${target_price:.2f} "
                f"(deviation: {deviation:.2%})"
            )

            # Mark old order as cancelled
            self._current_order.status = "cancelled"

            # Create new order
            child = ChildOrder(
                parent_id=self.order.order_id,
                symbol=self.order.symbol,
                side=self.order.side,
                quantity=self.order.remaining_quantity,
                order_type="limit",
                limit_price=target_price,
            )

            self._current_order = child
            self._last_reprice = now

            return child

        return None

    async def on_fill(self, child_order: ChildOrder, fill_price: float, fill_qty: float) -> None:
        """Handle fill."""
        old_qty = self.order.filled_quantity
        old_value = old_qty * self.order.avg_fill_price

        new_qty = old_qty + fill_qty
        new_value = old_value + (fill_qty * fill_price)

        self.order.filled_quantity = new_qty
        self.order.avg_fill_price = new_value / new_qty if new_qty > 0 else 0

        if self._current_order:
            self._current_order.filled_quantity += fill_qty
            if self._current_order.filled_quantity >= self._current_order.quantity:
                self._current_order.status = "filled"


# =============================================================================
# SMART ROUTER
# =============================================================================

class SmartRouter:
    """
    Smart Order Router.

    Manages execution algorithms and routes orders optimally.

    Example:
        router = SmartRouter(submit_func, market_func)

        # Execute TWAP
        order = AlgoOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10000,
            algo_type=AlgoType.TWAP,
            params={"duration_minutes": 30}
        )

        await router.execute(order)
    """

    def __init__(
        self,
        submit_order: OrderCallback,
        get_market_state: Callable[[str], Awaitable[MarketState]],
    ):
        """
        Initialize smart router.

        Args:
            submit_order: Callback to submit orders
            get_market_state: Callback to get market data
        """
        self._submit_order = submit_order
        self._get_market_state = get_market_state
        self._active_algos: dict[str, ExecutionAlgorithm] = {}

    def _create_algorithm(self, order: AlgoOrder) -> ExecutionAlgorithm:
        """Create algorithm instance based on order type."""
        algo_map = {
            AlgoType.TWAP: TWAPAlgorithm,
            AlgoType.VWAP: VWAPAlgorithm,
            AlgoType.ICEBERG: IcebergAlgorithm,
            AlgoType.PEGGED: PeggedAlgorithm,
        }

        algo_class = algo_map.get(order.algo_type)
        if not algo_class:
            raise ValueError(f"Unknown algorithm type: {order.algo_type}")

        return algo_class(order, self._submit_order, self._get_market_state)

    async def execute(self, order: AlgoOrder) -> str:
        """
        Execute an algorithmic order.

        Args:
            order: Algo order to execute

        Returns:
            Order ID
        """
        algo = self._create_algorithm(order)
        self._active_algos[order.order_id] = algo

        await algo.start()

        return order.order_id

    async def cancel(self, order_id: str) -> bool:
        """
        Cancel an algorithmic order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully
        """
        algo = self._active_algos.get(order_id)
        if not algo:
            return False

        await algo.stop()
        del self._active_algos[order_id]

        return True

    async def pause(self, order_id: str) -> bool:
        """Pause an algorithm."""
        algo = self._active_algos.get(order_id)
        if algo:
            await algo.pause()
            return True
        return False

    async def resume(self, order_id: str) -> bool:
        """Resume an algorithm."""
        algo = self._active_algos.get(order_id)
        if algo:
            await algo.resume()
            return True
        return False

    def get_status(self, order_id: str) -> dict[str, Any] | None:
        """Get algorithm status."""
        algo = self._active_algos.get(order_id)
        if not algo:
            return None

        return {
            "order_id": algo.order.order_id,
            "symbol": algo.order.symbol,
            "algo_type": algo.order.algo_type.value,
            "total_quantity": algo.order.total_quantity,
            "filled_quantity": algo.order.filled_quantity,
            "remaining_quantity": algo.order.remaining_quantity,
            "avg_fill_price": algo.order.avg_fill_price,
            "status": algo.order.status,
            "fill_rate": algo.order.fill_rate,
            "child_orders": len(algo.order.child_orders),
        }

    def get_all_active(self) -> list[str]:
        """Get all active algorithm order IDs."""
        return list(self._active_algos.keys())


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "AlgoType",
    "OrderSide",
    "PegType",
    # Data classes
    "AlgoOrder",
    "ChildOrder",
    "MarketState",
    # Algorithms
    "ExecutionAlgorithm",
    "TWAPAlgorithm",
    "VWAPAlgorithm",
    "IcebergAlgorithm",
    "PeggedAlgorithm",
    # Router
    "SmartRouter",
]
