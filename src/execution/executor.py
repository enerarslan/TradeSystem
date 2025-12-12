"""
Execution Algorithms
JPMorgan-Level Algorithmic Execution

Algorithms:
- TWAP (Time-Weighted Average Price)
- VWAP (Volume-Weighted Average Price)
- POV (Percentage of Volume)
- Adaptive execution
- Implementation shortfall minimization
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import threading

from .broker_api import BrokerAPI, OrderRequest, OrderSide, OrderType, TimeInForce
from .order_manager import OrderManager, Order, OrderStatus, OrderPriority
from ..utils.logger import get_logger, get_audit_logger

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class ExecutionStyle(Enum):
    """Execution urgency/style"""
    PASSIVE = "passive"
    NEUTRAL = "neutral"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"


@dataclass
class ExecutionParams:
    """Execution algorithm parameters"""
    # Time constraints
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: int = 3600  # 1 hour default

    # Size constraints
    min_slice_size: int = 100
    max_slice_size: int = 5000
    max_participation_rate: float = 0.10  # Max 10% of volume

    # Price constraints
    limit_price: Optional[float] = None
    price_tolerance_bps: float = 10  # 10 bps

    # Style
    execution_style: ExecutionStyle = ExecutionStyle.NEUTRAL
    allow_partial: bool = True

    # Scheduling
    randomize_timing: bool = True
    randomize_size: bool = True

    # Risk
    max_slippage_bps: float = 20
    stop_on_adverse_move: float = 0.02  # 2% adverse move


@dataclass
class ExecutionState:
    """Execution state tracking"""
    parent_order_id: str
    symbol: str
    side: OrderSide
    target_quantity: int
    executed_quantity: int = 0
    remaining_quantity: int = 0
    avg_price: float = 0
    total_value: float = 0
    child_orders: List[str] = field(default_factory=list)
    slices_sent: int = 0
    slices_filled: int = 0
    slices_cancelled: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    benchmark_price: float = 0
    vwap: float = 0
    slippage_bps: float = 0
    status: str = "running"
    errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.remaining_quantity = self.target_quantity

    def add_fill(self, quantity: int, price: float) -> None:
        """Record fill"""
        new_value = quantity * price
        self.total_value += new_value
        self.executed_quantity += quantity
        self.remaining_quantity = self.target_quantity - self.executed_quantity

        if self.executed_quantity > 0:
            self.avg_price = self.total_value / self.executed_quantity

        # Calculate slippage
        if self.benchmark_price > 0:
            if self.side == OrderSide.BUY:
                self.slippage_bps = (self.avg_price / self.benchmark_price - 1) * 10000
            else:
                self.slippage_bps = (1 - self.avg_price / self.benchmark_price) * 10000

        self.slices_filled += 1

    @property
    def fill_rate(self) -> float:
        return self.executed_quantity / self.target_quantity if self.target_quantity > 0 else 0

    @property
    def is_complete(self) -> bool:
        return self.remaining_quantity <= 0 or self.status in ['completed', 'cancelled', 'error']

    def to_dict(self) -> Dict[str, Any]:
        return {
            'parent_order_id': self.parent_order_id,
            'symbol': self.symbol,
            'target_quantity': self.target_quantity,
            'executed_quantity': self.executed_quantity,
            'remaining_quantity': self.remaining_quantity,
            'avg_price': self.avg_price,
            'fill_rate': self.fill_rate,
            'slippage_bps': self.slippage_bps,
            'slices_sent': self.slices_sent,
            'slices_filled': self.slices_filled,
            'status': self.status
        }


class ExecutionAlgo(ABC):
    """
    Abstract base class for execution algorithms.
    """

    def __init__(
        self,
        order_manager: OrderManager,
        params: Optional[ExecutionParams] = None
    ):
        self.order_manager = order_manager
        self.params = params or ExecutionParams()

        self._state: Optional[ExecutionState] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    @property
    def state(self) -> Optional[ExecutionState]:
        return self._state

    @abstractmethod
    async def execute(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        **kwargs
    ) -> ExecutionState:
        """Execute order with algorithm"""
        pass

    @abstractmethod
    async def calculate_slice(self) -> Optional[int]:
        """Calculate next slice size"""
        pass

    async def start(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        **kwargs
    ) -> str:
        """Start execution"""
        self._state = ExecutionState(
            parent_order_id=kwargs.get('order_id', ''),
            symbol=symbol,
            side=side,
            target_quantity=quantity,
            benchmark_price=kwargs.get('benchmark_price', 0)
        )

        self._running = True
        self._task = asyncio.create_task(self.execute(symbol, side, quantity, **kwargs))

        return self._state.parent_order_id

    async def stop(self) -> None:
        """Stop execution"""
        self._running = False
        if self._task:
            self._task.cancel()

        if self._state:
            self._state.status = 'cancelled'
            self._state.end_time = datetime.now()

    async def _send_slice(
        self,
        quantity: int,
        limit_price: Optional[float] = None
    ) -> Optional[Order]:
        """Send slice order"""
        if not self._state:
            return None

        try:
            order = await self.order_manager.create_order(
                symbol=self._state.symbol,
                side=self._state.side,
                quantity=quantity,
                order_type=OrderType.LIMIT if limit_price else OrderType.MARKET,
                limit_price=limit_price,
                time_in_force=TimeInForce.IOC,
                strategy_name=f"algo_{self.__class__.__name__}",
                priority=OrderPriority.HIGH
            )

            # Submit immediately
            success = await self.order_manager.submit_order(order)

            if success:
                self._state.child_orders.append(order.order_id)
                self._state.slices_sent += 1
                return order

        except Exception as e:
            logger.error(f"Slice error: {e}")
            if self._state:
                self._state.errors.append(str(e))

        return None


class TWAPExecutor(ExecutionAlgo):
    """
    Time-Weighted Average Price (TWAP) execution.

    Splits order into equal slices over time.
    """

    def __init__(
        self,
        order_manager: OrderManager,
        params: Optional[ExecutionParams] = None,
        num_slices: int = 20
    ):
        super().__init__(order_manager, params)
        self.num_slices = num_slices

    async def execute(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        **kwargs
    ) -> ExecutionState:
        """Execute TWAP"""
        logger.info(f"Starting TWAP: {side.value} {quantity} {symbol}")

        # Calculate slice timing
        duration = self.params.duration_seconds
        interval = duration / self.num_slices
        base_slice_size = quantity // self.num_slices

        while self._running and not self._state.is_complete:
            # Calculate slice size
            slice_size = await self.calculate_slice()

            if slice_size and slice_size > 0:
                # Get current price for limit
                limit_price = kwargs.get('current_price')
                if limit_price and self.params.price_tolerance_bps:
                    if side == OrderSide.BUY:
                        limit_price *= (1 + self.params.price_tolerance_bps / 10000)
                    else:
                        limit_price *= (1 - self.params.price_tolerance_bps / 10000)

                # Send slice
                order = await self._send_slice(slice_size, limit_price)

                if order:
                    # Wait for fill or timeout
                    await self._wait_for_fill(order, timeout=interval / 2)

                    # Record fill
                    if order.filled_quantity > 0:
                        self._state.add_fill(order.filled_quantity, order.avg_fill_price)

            # Wait for next interval
            if self.params.randomize_timing:
                jitter = np.random.uniform(0.8, 1.2)
                await asyncio.sleep(interval * jitter)
            else:
                await asyncio.sleep(interval)

        # Finalize
        self._state.status = 'completed' if self._state.remaining_quantity <= 0 else 'partial'
        self._state.end_time = datetime.now()

        logger.info(
            f"TWAP complete: filled {self._state.executed_quantity}/{quantity} "
            f"@ {self._state.avg_price:.2f}"
        )

        return self._state

    async def calculate_slice(self) -> Optional[int]:
        """Calculate TWAP slice size"""
        if not self._state:
            return None

        remaining = self._state.remaining_quantity
        remaining_slices = self.num_slices - self._state.slices_sent

        if remaining_slices <= 0:
            return remaining  # Last slice gets remainder

        base_size = remaining // remaining_slices

        # Apply randomization
        if self.params.randomize_size:
            jitter = np.random.uniform(0.8, 1.2)
            base_size = int(base_size * jitter)

        # Apply constraints
        base_size = max(base_size, self.params.min_slice_size)
        base_size = min(base_size, self.params.max_slice_size)
        base_size = min(base_size, remaining)

        return base_size

    async def _wait_for_fill(
        self,
        order: Order,
        timeout: float = 30
    ) -> None:
        """Wait for order fill"""
        start = datetime.now()
        while not order.is_complete:
            if (datetime.now() - start).total_seconds() > timeout:
                await self.order_manager.cancel_order(order.order_id)
                break
            await asyncio.sleep(0.5)


class VWAPExecutor(ExecutionAlgo):
    """
    Volume-Weighted Average Price (VWAP) execution.

    Follows historical volume profile.
    """

    def __init__(
        self,
        order_manager: OrderManager,
        params: Optional[ExecutionParams] = None,
        volume_profile: Optional[Dict[int, float]] = None
    ):
        super().__init__(order_manager, params)

        # Default volume profile (hourly, 9:30-16:00)
        self.volume_profile = volume_profile or {
            930: 0.12,   # 9:30
            1000: 0.08,  # 10:00
            1030: 0.06,  # 10:30
            1100: 0.06,  # 11:00
            1130: 0.05,  # 11:30
            1200: 0.05,  # 12:00
            1230: 0.05,  # 12:30
            1300: 0.06,  # 13:00
            1330: 0.07,  # 13:30
            1400: 0.08,  # 14:00
            1430: 0.09,  # 14:30
            1500: 0.10,  # 15:00
            1530: 0.13   # 15:30
        }

    async def execute(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        **kwargs
    ) -> ExecutionState:
        """Execute VWAP"""
        logger.info(f"Starting VWAP: {side.value} {quantity} {symbol}")

        # Calculate slice schedule
        schedule = self._build_schedule(quantity)

        for time_slot, slice_qty in schedule.items():
            if not self._running or self._state.is_complete:
                break

            # Wait for time slot
            await self._wait_until_slot(time_slot)

            if slice_qty > 0:
                # Get limit price
                limit_price = kwargs.get('current_price')

                # Send slice
                order = await self._send_slice(slice_qty, limit_price)

                if order:
                    await self._wait_for_fill(order, timeout=60)

                    if order.filled_quantity > 0:
                        self._state.add_fill(order.filled_quantity, order.avg_fill_price)

        self._state.status = 'completed' if self._state.remaining_quantity <= 0 else 'partial'
        self._state.end_time = datetime.now()

        return self._state

    def _build_schedule(self, quantity: int) -> Dict[int, int]:
        """Build VWAP schedule"""
        schedule = {}
        remaining = quantity

        for time_slot, pct in sorted(self.volume_profile.items()):
            slice_qty = int(quantity * pct)
            slice_qty = min(slice_qty, remaining)
            schedule[time_slot] = slice_qty
            remaining -= slice_qty

        # Add remainder to last slot
        if remaining > 0 and schedule:
            last_slot = max(schedule.keys())
            schedule[last_slot] += remaining

        return schedule

    async def _wait_until_slot(self, time_slot: int) -> None:
        """Wait until time slot"""
        now = datetime.now()
        target_hour = time_slot // 100
        target_minute = time_slot % 100

        target = now.replace(hour=target_hour, minute=target_minute, second=0)

        if target > now:
            wait_seconds = (target - now).total_seconds()
            await asyncio.sleep(wait_seconds)

    async def calculate_slice(self) -> Optional[int]:
        """Calculate VWAP slice from profile"""
        now = datetime.now()
        current_slot = now.hour * 100 + now.minute

        # Find nearest slot
        nearest = min(self.volume_profile.keys(), key=lambda x: abs(x - current_slot))
        pct = self.volume_profile.get(nearest, 0.05)

        slice_size = int(self._state.remaining_quantity * pct)
        return max(slice_size, self.params.min_slice_size)

    async def _wait_for_fill(self, order: Order, timeout: float = 30) -> None:
        """Wait for order fill"""
        start = datetime.now()
        while not order.is_complete:
            if (datetime.now() - start).total_seconds() > timeout:
                await self.order_manager.cancel_order(order.order_id)
                break
            await asyncio.sleep(0.5)


class POVExecutor(ExecutionAlgo):
    """
    Percentage of Volume (POV) execution.

    Participates as percentage of market volume.
    """

    def __init__(
        self,
        order_manager: OrderManager,
        params: Optional[ExecutionParams] = None,
        target_pov: float = 0.10,  # 10% of volume
        volume_callback: Optional[Callable] = None
    ):
        super().__init__(order_manager, params)
        self.target_pov = target_pov
        self.volume_callback = volume_callback

        self._volume_tracker: Dict[str, int] = {}

    async def execute(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        **kwargs
    ) -> ExecutionState:
        """Execute POV"""
        logger.info(f"Starting POV: {side.value} {quantity} {symbol} @ {self.target_pov:.1%}")

        check_interval = 30  # Check every 30 seconds

        while self._running and not self._state.is_complete:
            # Get recent volume
            recent_volume = await self._get_recent_volume(symbol, check_interval)

            if recent_volume > 0:
                # Calculate allowed participation
                allowed_qty = int(recent_volume * self.target_pov)
                allowed_qty = min(allowed_qty, self._state.remaining_quantity)
                allowed_qty = min(allowed_qty, self.params.max_slice_size)

                if allowed_qty >= self.params.min_slice_size:
                    order = await self._send_slice(allowed_qty)

                    if order:
                        await self._wait_for_fill(order, timeout=check_interval / 2)

                        if order.filled_quantity > 0:
                            self._state.add_fill(order.filled_quantity, order.avg_fill_price)

            await asyncio.sleep(check_interval)

        self._state.status = 'completed' if self._state.remaining_quantity <= 0 else 'partial'
        self._state.end_time = datetime.now()

        return self._state

    async def _get_recent_volume(
        self,
        symbol: str,
        interval_seconds: int
    ) -> int:
        """Get recent trading volume"""
        if self.volume_callback:
            return await self.volume_callback(symbol, interval_seconds)

        # Default: estimate from daily average
        # Assuming 6.5 hour trading day, average daily volume of 1M
        avg_daily = 1000000
        per_second = avg_daily / (6.5 * 3600)
        return int(per_second * interval_seconds)

    async def calculate_slice(self) -> Optional[int]:
        """Calculate POV slice"""
        recent_volume = await self._get_recent_volume(self._state.symbol, 60)
        return int(recent_volume * self.target_pov)

    async def _wait_for_fill(self, order: Order, timeout: float = 30) -> None:
        """Wait for order fill"""
        start = datetime.now()
        while not order.is_complete:
            if (datetime.now() - start).total_seconds() > timeout:
                await self.order_manager.cancel_order(order.order_id)
                break
            await asyncio.sleep(0.5)


class AdaptiveExecutor(ExecutionAlgo):
    """
    Adaptive execution algorithm.

    Dynamically adjusts between passive and aggressive
    based on market conditions and urgency.
    """

    def __init__(
        self,
        order_manager: OrderManager,
        params: Optional[ExecutionParams] = None,
        price_callback: Optional[Callable] = None,
        spread_callback: Optional[Callable] = None
    ):
        super().__init__(order_manager, params)
        self.price_callback = price_callback
        self.spread_callback = spread_callback

        self._urgency = 0.5  # 0 = passive, 1 = aggressive

    async def execute(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        **kwargs
    ) -> ExecutionState:
        """Execute adaptively"""
        logger.info(f"Starting Adaptive: {side.value} {quantity} {symbol}")

        self._state.benchmark_price = kwargs.get('benchmark_price', 0)

        while self._running and not self._state.is_complete:
            # Update urgency based on time remaining
            self._update_urgency()

            # Calculate slice
            slice_size = await self.calculate_slice()

            if slice_size and slice_size > 0:
                # Determine order type based on urgency
                if self._urgency > 0.7:
                    # Aggressive: market order
                    order = await self._send_slice(slice_size, None)
                else:
                    # Passive: limit order at favorable price
                    limit_price = await self._calculate_limit_price(symbol, side)
                    order = await self._send_slice(slice_size, limit_price)

                if order:
                    timeout = 30 if self._urgency < 0.5 else 10
                    await self._wait_for_fill(order, timeout)

                    if order.filled_quantity > 0:
                        self._state.add_fill(order.filled_quantity, order.avg_fill_price)

            # Adaptive sleep
            sleep_time = 30 * (1 - self._urgency * 0.8)
            await asyncio.sleep(max(5, sleep_time))

        self._state.status = 'completed' if self._state.remaining_quantity <= 0 else 'partial'
        self._state.end_time = datetime.now()

        return self._state

    def _update_urgency(self) -> None:
        """Update urgency based on time and fill progress"""
        if not self._state:
            return

        # Time urgency
        elapsed = (datetime.now() - self._state.start_time).total_seconds()
        time_pct = elapsed / self.params.duration_seconds
        time_urgency = min(1.0, time_pct * 1.5)  # Increase faster near end

        # Fill urgency (if behind schedule)
        expected_fill_pct = min(1.0, time_pct)
        actual_fill_pct = self._state.fill_rate
        fill_urgency = max(0, expected_fill_pct - actual_fill_pct)

        # Combine
        self._urgency = min(1.0, time_urgency * 0.6 + fill_urgency * 0.4)

    async def _calculate_limit_price(
        self,
        symbol: str,
        side: OrderSide
    ) -> Optional[float]:
        """Calculate limit price"""
        if self.price_callback:
            mid_price = await self.price_callback(symbol)
        else:
            mid_price = self._state.benchmark_price

        if not mid_price:
            return None

        # Adjust based on urgency
        offset_bps = self.params.price_tolerance_bps * (1 - self._urgency)

        if side == OrderSide.BUY:
            return mid_price * (1 - offset_bps / 10000)
        else:
            return mid_price * (1 + offset_bps / 10000)

    async def calculate_slice(self) -> Optional[int]:
        """Calculate adaptive slice size"""
        if not self._state:
            return None

        remaining = self._state.remaining_quantity
        time_left = self.params.duration_seconds - \
            (datetime.now() - self._state.start_time).total_seconds()

        if time_left <= 0:
            return remaining  # Final slice

        # Base slice on urgency
        num_remaining_slices = max(1, int(time_left / 30))
        base_size = remaining // num_remaining_slices

        # Adjust by urgency
        urgency_multiplier = 1 + self._urgency * 0.5
        slice_size = int(base_size * urgency_multiplier)

        # Apply constraints
        slice_size = max(slice_size, self.params.min_slice_size)
        slice_size = min(slice_size, self.params.max_slice_size)
        slice_size = min(slice_size, remaining)

        return slice_size

    async def _wait_for_fill(self, order: Order, timeout: float = 30) -> None:
        """Wait for order fill"""
        start = datetime.now()
        while not order.is_complete:
            if (datetime.now() - start).total_seconds() > timeout:
                await self.order_manager.cancel_order(order.order_id)
                break
            await asyncio.sleep(0.5)


class ExecutionEngine:
    """
    Central execution engine.

    Manages multiple execution algorithms and tracks performance.
    """

    def __init__(
        self,
        order_manager: OrderManager
    ):
        self.order_manager = order_manager

        self._active_executions: Dict[str, ExecutionAlgo] = {}
        self._completed_executions: List[ExecutionState] = []

    async def execute_twap(
        self,
        symbol: str,
        side: Union[str, OrderSide],
        quantity: int,
        duration_minutes: int = 60,
        num_slices: int = 20,
        **kwargs
    ) -> ExecutionState:
        """Execute using TWAP"""
        if isinstance(side, str):
            side = OrderSide(side.lower())

        params = ExecutionParams(duration_seconds=duration_minutes * 60)
        algo = TWAPExecutor(self.order_manager, params, num_slices)

        execution_id = await algo.start(symbol, side, quantity, **kwargs)
        self._active_executions[execution_id] = algo

        state = await algo._task
        del self._active_executions[execution_id]
        self._completed_executions.append(state)

        return state

    async def execute_vwap(
        self,
        symbol: str,
        side: Union[str, OrderSide],
        quantity: int,
        volume_profile: Optional[Dict] = None,
        **kwargs
    ) -> ExecutionState:
        """Execute using VWAP"""
        if isinstance(side, str):
            side = OrderSide(side.lower())

        params = ExecutionParams()
        algo = VWAPExecutor(self.order_manager, params, volume_profile)

        execution_id = await algo.start(symbol, side, quantity, **kwargs)
        self._active_executions[execution_id] = algo

        state = await algo._task
        del self._active_executions[execution_id]
        self._completed_executions.append(state)

        return state

    async def execute_pov(
        self,
        symbol: str,
        side: Union[str, OrderSide],
        quantity: int,
        target_pov: float = 0.10,
        **kwargs
    ) -> ExecutionState:
        """Execute using POV"""
        if isinstance(side, str):
            side = OrderSide(side.lower())

        params = ExecutionParams()
        algo = POVExecutor(self.order_manager, params, target_pov)

        execution_id = await algo.start(symbol, side, quantity, **kwargs)
        self._active_executions[execution_id] = algo

        state = await algo._task
        del self._active_executions[execution_id]
        self._completed_executions.append(state)

        return state

    async def execute_adaptive(
        self,
        symbol: str,
        side: Union[str, OrderSide],
        quantity: int,
        duration_minutes: int = 60,
        **kwargs
    ) -> ExecutionState:
        """Execute using adaptive algorithm"""
        if isinstance(side, str):
            side = OrderSide(side.lower())

        params = ExecutionParams(duration_seconds=duration_minutes * 60)
        algo = AdaptiveExecutor(self.order_manager, params)

        execution_id = await algo.start(symbol, side, quantity, **kwargs)
        self._active_executions[execution_id] = algo

        state = await algo._task
        del self._active_executions[execution_id]
        self._completed_executions.append(state)

        return state

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel active execution"""
        if execution_id in self._active_executions:
            await self._active_executions[execution_id].stop()
            return True
        return False

    async def cancel_all(self) -> int:
        """Cancel all executions"""
        count = 0
        for exec_id in list(self._active_executions.keys()):
            await self.cancel_execution(exec_id)
            count += 1
        return count

    def get_execution(self, execution_id: str) -> Optional[ExecutionState]:
        """Get execution state"""
        if execution_id in self._active_executions:
            return self._active_executions[execution_id].state

        for state in self._completed_executions:
            if state.parent_order_id == execution_id:
                return state

        return None

    def get_active_executions(self) -> List[ExecutionState]:
        """Get all active executions"""
        return [algo.state for algo in self._active_executions.values() if algo.state]

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        total = len(self._completed_executions)
        if total == 0:
            return {'total_executions': 0}

        completed = [s for s in self._completed_executions if s.fill_rate >= 0.99]
        avg_slippage = np.mean([s.slippage_bps for s in self._completed_executions])
        avg_fill_rate = np.mean([s.fill_rate for s in self._completed_executions])

        return {
            'total_executions': total,
            'active_executions': len(self._active_executions),
            'completion_rate': len(completed) / total,
            'avg_fill_rate': avg_fill_rate,
            'avg_slippage_bps': avg_slippage
        }
