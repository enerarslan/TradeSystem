"""
Algorithmic Execution Engine
Smart Order Router with VWAP/TWAP Algorithms

Institutional-grade execution algorithms designed to minimize market impact
and slippage by intelligently slicing large orders into smaller child orders.

Features:
- VWAP (Volume Weighted Average Price) execution
- TWAP (Time Weighted Average Price) execution
- Adaptive slicing based on market conditions
- Real-time execution monitoring
- Slippage tracking and reporting
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import heapq

from ..utils.logger import get_logger, get_audit_logger


logger = get_logger(__name__)
audit_logger = get_audit_logger()


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class ExecutionAlgorithm(Enum):
    VWAP = "VWAP"
    TWAP = "TWAP"
    POV = "POV"  # Percentage of Volume
    IMPLEMENTATION_SHORTFALL = "IS"


@dataclass
class ChildOrder:
    """
    Child order created by slicing a parent order.

    Attributes:
        child_id: Unique identifier for this child order
        parent_id: Reference to parent order
        symbol: Trading symbol
        side: Buy or Sell
        quantity: Order quantity
        limit_price: Optional limit price
        scheduled_time: When this order should be executed
        status: Current status
        fill_price: Actual fill price (if filled)
        fill_quantity: Actual filled quantity
        fill_time: Time of fill
        slippage_bps: Slippage in basis points
    """
    child_id: str
    parent_id: str
    symbol: str
    side: OrderSide
    quantity: float
    limit_price: Optional[float] = None
    scheduled_time: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    fill_quantity: float = 0.0
    fill_time: Optional[datetime] = None
    slippage_bps: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __lt__(self, other):
        """For priority queue ordering by scheduled_time"""
        if self.scheduled_time is None or other.scheduled_time is None:
            return False
        return self.scheduled_time < other.scheduled_time


@dataclass
class ExecutionReport:
    """
    Final report for an executed parent order.

    Contains metrics on execution quality including VWAP comparison,
    slippage, and implementation shortfall.
    """
    parent_id: str
    symbol: str
    side: OrderSide
    algorithm: ExecutionAlgorithm
    total_quantity: float
    filled_quantity: float
    average_fill_price: float
    vwap_benchmark: float
    market_vwap: float
    slippage_bps: float
    implementation_shortfall_bps: float
    num_child_orders: int
    num_fills: int
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    child_orders: List[ChildOrder] = field(default_factory=list)

    @property
    def fill_rate(self) -> float:
        """Percentage of order filled"""
        return self.filled_quantity / self.total_quantity if self.total_quantity > 0 else 0.0

    @property
    def execution_quality_score(self) -> float:
        """
        Score from 0-100 indicating execution quality.
        100 = Beat VWAP significantly
        50 = Matched VWAP
        0 = Significant underperformance
        """
        # Convert slippage to score (lower slippage = higher score)
        # Assume +/- 50 bps is the range
        score = 50 - self.slippage_bps
        return max(0, min(100, score))


class BaseExecutionAlgorithm(ABC):
    """
    Base class for execution algorithms.

    All execution algorithms must implement the slice_order method
    to divide a parent order into child orders.
    """

    def __init__(
        self,
        max_participation_rate: float = 0.10,
        min_order_size: float = 100,
        max_slices: int = 50
    ):
        """
        Initialize base execution algorithm.

        Args:
            max_participation_rate: Max % of volume to participate
            min_order_size: Minimum child order size (shares)
            max_slices: Maximum number of child orders
        """
        self.max_participation_rate = max_participation_rate
        self.min_order_size = min_order_size
        self.max_slices = max_slices

    @abstractmethod
    def slice_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        duration_minutes: int,
        volume_profile: Optional[pd.Series] = None,
        current_price: Optional[float] = None
    ) -> List[ChildOrder]:
        """
        Slice a parent order into child orders.

        Args:
            symbol: Trading symbol
            side: Buy or Sell
            quantity: Total quantity to execute
            duration_minutes: Time window for execution
            volume_profile: Historical volume profile by time
            current_price: Current market price

        Returns:
            List of child orders
        """
        pass

    def _generate_parent_id(self) -> str:
        """Generate unique parent order ID"""
        return f"PO-{uuid.uuid4().hex[:12].upper()}"

    def _generate_child_id(self, parent_id: str, index: int) -> str:
        """Generate unique child order ID"""
        return f"{parent_id}-C{index:03d}"


class VWAPExecutor(BaseExecutionAlgorithm):
    """
    Volume Weighted Average Price (VWAP) Execution Algorithm.

    Slices orders based on historical volume profile to achieve
    execution at or better than the market VWAP.

    Strategy:
    - Analyze historical intraday volume distribution
    - Distribute child orders proportionally to expected volume
    - Concentrate execution during high-volume periods
    - Reduce market impact by matching natural liquidity
    """

    def __init__(
        self,
        volume_lookback_days: int = 20,
        aggression_factor: float = 1.0,
        **kwargs
    ):
        """
        Initialize VWAP executor.

        Args:
            volume_lookback_days: Days of volume data to analyze
            aggression_factor: 1.0 = match volume profile exactly
                              >1.0 = front-load execution
                              <1.0 = back-load execution
        """
        super().__init__(**kwargs)
        self.volume_lookback_days = volume_lookback_days
        self.aggression_factor = aggression_factor

    def slice_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        duration_minutes: int,
        volume_profile: Optional[pd.Series] = None,
        current_price: Optional[float] = None
    ) -> List[ChildOrder]:
        """
        Slice order using VWAP algorithm.

        Distributes quantity across time buckets proportional to
        historical volume distribution.
        """
        parent_id = self._generate_parent_id()
        now = datetime.utcnow()

        # Calculate number of slices
        num_slices = min(
            self.max_slices,
            max(1, duration_minutes // 5)  # One slice per 5 minutes
        )

        # Get volume weights
        if volume_profile is not None and len(volume_profile) >= num_slices:
            # Use provided volume profile
            weights = self._normalize_profile(volume_profile, num_slices)
        else:
            # Generate synthetic U-shaped volume profile (market typical)
            weights = self._generate_u_shaped_profile(num_slices)

        # Apply aggression factor
        weights = self._apply_aggression(weights)

        # Calculate quantities for each slice
        quantities = self._distribute_quantity(quantity, weights)

        # Generate child orders
        child_orders = []
        interval_minutes = duration_minutes / num_slices

        for i, qty in enumerate(quantities):
            if qty < self.min_order_size:
                continue

            scheduled_time = now + timedelta(minutes=i * interval_minutes)

            child = ChildOrder(
                child_id=self._generate_child_id(parent_id, i),
                parent_id=parent_id,
                symbol=symbol,
                side=side,
                quantity=qty,
                scheduled_time=scheduled_time,
                limit_price=current_price if current_price else None
            )
            child_orders.append(child)

        logger.info(
            f"VWAP sliced {symbol} {side.value} {quantity} into "
            f"{len(child_orders)} child orders over {duration_minutes} min"
        )

        return child_orders

    def _normalize_profile(
        self,
        profile: pd.Series,
        num_slices: int
    ) -> np.ndarray:
        """Normalize and resample volume profile to target slices"""
        # Resample to target number of buckets
        profile_array = profile.values
        if len(profile_array) != num_slices:
            indices = np.linspace(0, len(profile_array) - 1, num_slices)
            profile_array = np.interp(indices, range(len(profile_array)), profile_array)

        # Normalize to sum to 1
        total = profile_array.sum()
        if total > 0:
            return profile_array / total
        return np.ones(num_slices) / num_slices

    def _generate_u_shaped_profile(self, num_slices: int) -> np.ndarray:
        """
        Generate typical U-shaped intraday volume profile.

        Markets typically have higher volume at open and close,
        lower volume mid-day.
        """
        x = np.linspace(-1, 1, num_slices)
        # U-shape: higher at ends, lower in middle
        profile = 1 + 0.5 * (x ** 2)
        return profile / profile.sum()

    def _apply_aggression(self, weights: np.ndarray) -> np.ndarray:
        """Apply aggression factor to weight distribution"""
        if self.aggression_factor == 1.0:
            return weights

        n = len(weights)
        # Create position array
        positions = np.linspace(0, 1, n)

        if self.aggression_factor > 1.0:
            # Front-load: increase early weights
            adjustment = (1 - positions) ** (self.aggression_factor - 1)
        else:
            # Back-load: increase later weights
            adjustment = positions ** (1 / self.aggression_factor)

        adjusted = weights * adjustment
        return adjusted / adjusted.sum()

    def _distribute_quantity(
        self,
        total_qty: float,
        weights: np.ndarray
    ) -> List[float]:
        """Distribute total quantity according to weights"""
        raw_quantities = total_qty * weights

        # Round to integers, preserving total
        quantities = np.floor(raw_quantities).astype(int)
        remainder = int(total_qty - quantities.sum())

        # Distribute remainder to largest fractional parts
        if remainder > 0:
            fractional = raw_quantities - quantities
            indices = np.argsort(fractional)[-remainder:]
            quantities[indices] += 1

        return quantities.tolist()


class TWAPExecutor(BaseExecutionAlgorithm):
    """
    Time Weighted Average Price (TWAP) Execution Algorithm.

    Slices orders evenly across time intervals for uniform execution.
    Simpler than VWAP but effective when volume profile is unknown
    or for instruments with uniform liquidity.

    Strategy:
    - Divide total quantity evenly across time buckets
    - Execute at regular intervals
    - Optional randomization to reduce predictability
    """

    def __init__(
        self,
        randomize: bool = True,
        randomization_pct: float = 0.20,
        **kwargs
    ):
        """
        Initialize TWAP executor.

        Args:
            randomize: Add randomization to slice timing/sizing
            randomization_pct: Max +/- variation percentage
        """
        super().__init__(**kwargs)
        self.randomize = randomize
        self.randomization_pct = randomization_pct

    def slice_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        duration_minutes: int,
        volume_profile: Optional[pd.Series] = None,
        current_price: Optional[float] = None
    ) -> List[ChildOrder]:
        """
        Slice order using TWAP algorithm.

        Distributes quantity evenly across time buckets.
        """
        parent_id = self._generate_parent_id()
        now = datetime.utcnow()

        # Calculate number of slices
        num_slices = min(
            self.max_slices,
            max(1, duration_minutes // 5)  # One slice per 5 minutes
        )

        # Base quantity per slice
        base_qty = quantity / num_slices

        # Generate quantities with optional randomization
        if self.randomize:
            quantities = self._randomize_quantities(base_qty, num_slices, quantity)
        else:
            quantities = [quantity // num_slices] * num_slices
            # Handle remainder
            remainder = quantity - sum(quantities)
            for i in range(int(remainder)):
                quantities[i] += 1

        # Generate child orders
        child_orders = []
        interval_minutes = duration_minutes / num_slices

        for i, qty in enumerate(quantities):
            if qty < self.min_order_size:
                continue

            # Base scheduled time
            base_time = now + timedelta(minutes=i * interval_minutes)

            # Add random jitter to timing if enabled
            if self.randomize:
                jitter = np.random.uniform(
                    -interval_minutes * 0.3,
                    interval_minutes * 0.3
                )
                scheduled_time = base_time + timedelta(minutes=jitter)
            else:
                scheduled_time = base_time

            child = ChildOrder(
                child_id=self._generate_child_id(parent_id, i),
                parent_id=parent_id,
                symbol=symbol,
                side=side,
                quantity=qty,
                scheduled_time=scheduled_time,
                limit_price=current_price if current_price else None
            )
            child_orders.append(child)

        # Sort by scheduled time
        child_orders.sort(key=lambda x: x.scheduled_time or datetime.max)

        logger.info(
            f"TWAP sliced {symbol} {side.value} {quantity} into "
            f"{len(child_orders)} child orders over {duration_minutes} min"
        )

        return child_orders

    def _randomize_quantities(
        self,
        base_qty: float,
        num_slices: int,
        total_qty: float
    ) -> List[float]:
        """Add random variation to quantities while preserving total"""
        # Generate random variations
        variations = np.random.uniform(
            1 - self.randomization_pct,
            1 + self.randomization_pct,
            num_slices
        )

        raw_quantities = base_qty * variations

        # Normalize to maintain total
        raw_quantities = raw_quantities * (total_qty / raw_quantities.sum())

        # Round and adjust
        quantities = np.floor(raw_quantities).astype(int)
        remainder = int(total_qty - quantities.sum())

        if remainder > 0:
            indices = np.random.choice(num_slices, remainder, replace=False)
            quantities[indices] += 1

        return quantities.tolist()


class SmartOrderRouter:
    """
    Smart Order Router that manages algorithmic execution.

    Features:
    - Automatic algorithm selection based on order characteristics
    - Real-time execution monitoring
    - Child order scheduling and dispatch
    - Execution quality tracking
    - Integration with broker API
    """

    def __init__(
        self,
        vwap_executor: Optional[VWAPExecutor] = None,
        twap_executor: Optional[TWAPExecutor] = None,
        default_algorithm: ExecutionAlgorithm = ExecutionAlgorithm.VWAP
    ):
        """
        Initialize Smart Order Router.

        Args:
            vwap_executor: VWAP execution algorithm instance
            twap_executor: TWAP execution algorithm instance
            default_algorithm: Default algorithm when not specified
        """
        self.vwap_executor = vwap_executor or VWAPExecutor()
        self.twap_executor = twap_executor or TWAPExecutor()
        self.default_algorithm = default_algorithm

        # Order tracking
        self._pending_orders: Dict[str, List[ChildOrder]] = {}
        self._active_orders: Dict[str, ChildOrder] = {}
        self._completed_orders: Dict[str, List[ChildOrder]] = {}
        self._execution_reports: Dict[str, ExecutionReport] = {}

        # Execution callback
        self._order_callback: Optional[Callable] = None

        logger.info("SmartOrderRouter initialized")

    def set_order_callback(self, callback: Callable[[ChildOrder], None]) -> None:
        """Set callback for when child orders should be executed"""
        self._order_callback = callback

    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        duration_minutes: int = 60,
        algorithm: Optional[ExecutionAlgorithm] = None,
        volume_profile: Optional[pd.Series] = None,
        current_price: Optional[float] = None
    ) -> str:
        """
        Submit a parent order for algorithmic execution.

        Args:
            symbol: Trading symbol
            side: Buy or Sell
            quantity: Total quantity to execute
            duration_minutes: Execution time window
            algorithm: Execution algorithm to use
            volume_profile: Historical volume profile
            current_price: Current market price

        Returns:
            Parent order ID
        """
        algo = algorithm or self.default_algorithm

        # Select executor
        if algo == ExecutionAlgorithm.VWAP:
            executor = self.vwap_executor
        elif algo == ExecutionAlgorithm.TWAP:
            executor = self.twap_executor
        else:
            executor = self.vwap_executor  # Default

        # Slice the order
        child_orders = executor.slice_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            duration_minutes=duration_minutes,
            volume_profile=volume_profile,
            current_price=current_price
        )

        if not child_orders:
            raise ValueError("Failed to generate child orders")

        parent_id = child_orders[0].parent_id

        # Store pending orders
        self._pending_orders[parent_id] = child_orders

        # Log to audit trail
        audit_logger.log_order(
            order_id=parent_id,
            symbol=symbol,
            action="CREATED",
            details={
                "algorithm": algo.value,
                "total_quantity": quantity,
                "num_child_orders": len(child_orders),
                "duration_minutes": duration_minutes
            }
        )

        logger.info(
            f"Submitted {algo.value} order {parent_id}: "
            f"{symbol} {side.value} {quantity}"
        )

        return parent_id

    async def execute_scheduled_orders(self) -> List[ChildOrder]:
        """
        Execute all child orders that are due.

        Returns:
            List of executed child orders
        """
        now = datetime.utcnow()
        executed = []

        for parent_id, children in list(self._pending_orders.items()):
            due_orders = [
                c for c in children
                if c.scheduled_time and c.scheduled_time <= now
                and c.status == OrderStatus.PENDING
            ]

            for child in due_orders:
                child.status = OrderStatus.ACTIVE
                self._active_orders[child.child_id] = child

                # Execute via callback
                if self._order_callback:
                    try:
                        await asyncio.to_thread(self._order_callback, child)
                    except Exception as e:
                        logger.error(f"Order execution failed: {e}")
                        child.status = OrderStatus.REJECTED

                executed.append(child)

        return executed

    def record_fill(
        self,
        child_id: str,
        fill_price: float,
        fill_quantity: float,
        fill_time: Optional[datetime] = None
    ) -> None:
        """
        Record a fill for a child order.

        Args:
            child_id: Child order ID
            fill_price: Execution price
            fill_quantity: Filled quantity
            fill_time: Time of fill
        """
        if child_id not in self._active_orders:
            logger.warning(f"Unknown child order: {child_id}")
            return

        child = self._active_orders[child_id]
        child.fill_price = fill_price
        child.fill_quantity = fill_quantity
        child.fill_time = fill_time or datetime.utcnow()

        if fill_quantity >= child.quantity:
            child.status = OrderStatus.FILLED
        else:
            child.status = OrderStatus.PARTIALLY_FILLED

        # Calculate slippage if limit price was set
        if child.limit_price:
            if child.side == OrderSide.BUY:
                child.slippage_bps = (
                    (fill_price - child.limit_price) / child.limit_price * 10000
                )
            else:
                child.slippage_bps = (
                    (child.limit_price - fill_price) / child.limit_price * 10000
                )

        # Move to completed
        parent_id = child.parent_id
        if parent_id not in self._completed_orders:
            self._completed_orders[parent_id] = []
        self._completed_orders[parent_id].append(child)
        del self._active_orders[child_id]

        # Check if all children are complete
        self._check_parent_completion(parent_id)

    def _check_parent_completion(self, parent_id: str) -> None:
        """Check if all child orders for a parent are complete"""
        if parent_id not in self._pending_orders:
            return

        pending = self._pending_orders.get(parent_id, [])
        completed = self._completed_orders.get(parent_id, [])

        # Check if all orders are either completed or cancelled
        all_done = all(
            o.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]
            for o in pending + completed
        )

        if all_done:
            self._generate_execution_report(parent_id)

    def _generate_execution_report(self, parent_id: str) -> ExecutionReport:
        """Generate final execution report for a parent order"""
        completed = self._completed_orders.get(parent_id, [])
        pending = self._pending_orders.get(parent_id, [])
        all_orders = completed + pending

        if not all_orders:
            return None

        # Calculate metrics
        first_order = all_orders[0]
        filled_orders = [o for o in all_orders if o.fill_price is not None]

        total_quantity = sum(o.quantity for o in all_orders)
        filled_quantity = sum(o.fill_quantity for o in filled_orders)

        if filled_quantity > 0:
            avg_fill_price = sum(
                o.fill_price * o.fill_quantity for o in filled_orders
            ) / filled_quantity
        else:
            avg_fill_price = 0

        # Calculate average slippage
        slippage = np.mean([o.slippage_bps for o in filled_orders]) if filled_orders else 0

        # Create report
        report = ExecutionReport(
            parent_id=parent_id,
            symbol=first_order.symbol,
            side=first_order.side,
            algorithm=ExecutionAlgorithm.VWAP,  # TODO: track this
            total_quantity=total_quantity,
            filled_quantity=filled_quantity,
            average_fill_price=avg_fill_price,
            vwap_benchmark=avg_fill_price,  # TODO: calculate market VWAP
            market_vwap=avg_fill_price,
            slippage_bps=slippage,
            implementation_shortfall_bps=slippage,
            num_child_orders=len(all_orders),
            num_fills=len(filled_orders),
            start_time=min(o.created_at for o in all_orders),
            end_time=max(o.fill_time or o.created_at for o in all_orders),
            duration_seconds=(
                max(o.fill_time or o.created_at for o in all_orders) -
                min(o.created_at for o in all_orders)
            ).total_seconds(),
            child_orders=all_orders
        )

        self._execution_reports[parent_id] = report

        # Cleanup
        if parent_id in self._pending_orders:
            del self._pending_orders[parent_id]

        # Log report
        logger.info(
            f"Execution complete for {parent_id}: "
            f"{filled_quantity}/{total_quantity} filled @ {avg_fill_price:.2f}, "
            f"slippage: {slippage:.1f} bps"
        )

        return report

    def get_execution_report(self, parent_id: str) -> Optional[ExecutionReport]:
        """Get execution report for a completed order"""
        return self._execution_reports.get(parent_id)

    def get_pending_orders(self, parent_id: str) -> List[ChildOrder]:
        """Get pending child orders for a parent order"""
        return self._pending_orders.get(parent_id, [])

    def cancel_order(self, parent_id: str) -> bool:
        """
        Cancel all pending child orders for a parent order.

        Args:
            parent_id: Parent order ID to cancel

        Returns:
            True if successfully cancelled
        """
        if parent_id not in self._pending_orders:
            logger.warning(f"Order {parent_id} not found")
            return False

        for child in self._pending_orders[parent_id]:
            if child.status == OrderStatus.PENDING:
                child.status = OrderStatus.CANCELLED

        # Also cancel any active orders
        for child_id, child in list(self._active_orders.items()):
            if child.parent_id == parent_id:
                child.status = OrderStatus.CANCELLED
                del self._active_orders[child_id]

        logger.info(f"Cancelled order {parent_id}")

        audit_logger.log_order(
            order_id=parent_id,
            symbol=self._pending_orders[parent_id][0].symbol if self._pending_orders.get(parent_id) else "UNKNOWN",
            action="CANCELLED",
            details={"reason": "User requested cancellation"}
        )

        return True

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get aggregate execution statistics"""
        reports = list(self._execution_reports.values())

        if not reports:
            return {}

        return {
            "total_orders": len(reports),
            "total_volume": sum(r.filled_quantity for r in reports),
            "avg_slippage_bps": np.mean([r.slippage_bps for r in reports]),
            "avg_fill_rate": np.mean([r.fill_rate for r in reports]),
            "avg_quality_score": np.mean([r.execution_quality_score for r in reports])
        }
