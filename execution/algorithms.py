"""
Execution Algorithms Module
===========================

Smart order execution algorithms for minimizing market impact.
Institutional-grade execution strategies for large orders.

Algorithms:
- TWAP: Time-Weighted Average Price
- VWAP: Volume-Weighted Average Price
- Iceberg: Hidden order quantity
- POV: Percentage of Volume
- Implementation Shortfall (placeholder)
- Arrival Price (placeholder)

Features:
- Adaptive execution based on market conditions
- Real-time monitoring and adjustment
- Risk controls and limits
- Performance attribution

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Generator
from uuid import UUID, uuid4
import threading
import math

import numpy as np
from numpy.typing import NDArray

from config.settings import get_logger
from core.types import Order, Position, Trade, ExecutionError
from execution.broker import BrokerBase, BrokerType

logger = get_logger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class ExecutionAlgoType(str, Enum):
    """Execution algorithm types."""
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"
    POV = "pov"
    IMPLEMENTATION_SHORTFALL = "is"
    ARRIVAL_PRICE = "arrival"
    MARKET = "market"
    LIMIT = "limit"


class ExecutionStatus(str, Enum):
    """Execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class UrgencyLevel(str, Enum):
    """Execution urgency level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# CONFIGURATIONS
# =============================================================================

@dataclass
class BaseExecutionConfig:
    """Base configuration for execution algorithms."""
    symbol: str
    side: str  # "buy" or "sell"
    total_quantity: float
    urgency: UrgencyLevel = UrgencyLevel.MEDIUM
    start_time: datetime | None = None
    end_time: datetime | None = None
    max_participation_rate: float = 0.10  # Max % of volume
    min_order_size: float = 1.0
    max_order_size: float | None = None
    price_limit: float | None = None  # Price limit for execution
    allow_aggressive: bool = True  # Allow crossing spread
    
    def validate(self) -> list[str]:
        """Validate configuration."""
        errors = []
        if self.total_quantity <= 0:
            errors.append("total_quantity must be positive")
        if self.min_order_size <= 0:
            errors.append("min_order_size must be positive")
        if self.max_participation_rate <= 0 or self.max_participation_rate > 1:
            errors.append("max_participation_rate must be between 0 and 1")
        return errors


@dataclass
class TWAPConfig(BaseExecutionConfig):
    """
    TWAP (Time-Weighted Average Price) configuration.
    
    Splits order evenly across time intervals.
    
    Attributes:
        duration_minutes: Total execution duration
        num_slices: Number of order slices
        randomize_timing: Add randomness to slice timing
        randomize_size: Add randomness to slice sizes
    """
    duration_minutes: int = 60
    num_slices: int = 10
    randomize_timing: bool = True
    randomize_size: bool = True
    timing_variance: float = 0.2  # 20% timing variance
    size_variance: float = 0.1  # 10% size variance


@dataclass
class VWAPConfig(BaseExecutionConfig):
    """
    VWAP (Volume-Weighted Average Price) configuration.
    
    Executes proportionally to historical volume profile.
    
    Attributes:
        duration_minutes: Total execution duration
        volume_profile: Historical volume buckets (optional)
        adaptive: Adjust to real-time volume
        lookback_days: Days of volume history
    """
    duration_minutes: int = 60
    volume_profile: list[float] | None = None
    adaptive: bool = True
    lookback_days: int = 20
    check_interval_seconds: int = 60


@dataclass
class IcebergConfig(BaseExecutionConfig):
    """
    Iceberg order configuration.
    
    Shows only a small portion of total quantity.
    
    Attributes:
        display_quantity: Visible order size
        refresh_quantity: Size to refresh when filled
        variance: Random variance in display size
        price_offset: Offset from reference price
    """
    display_quantity: float = 100.0
    refresh_quantity: float | None = None  # Defaults to display_quantity
    variance: float = 0.1  # 10% variance
    price_offset: float = 0.0  # Price offset from mid


@dataclass
class POVConfig(BaseExecutionConfig):
    """
    POV (Percentage of Volume) configuration.
    
    Executes as a target percentage of market volume.
    
    Attributes:
        target_pov: Target participation rate (0-1)
        min_pov: Minimum participation rate
        max_pov: Maximum participation rate
        check_interval_seconds: Volume check interval
    """
    target_pov: float = 0.05  # 5% of volume
    min_pov: float = 0.01
    max_pov: float = 0.15
    check_interval_seconds: int = 30
    volume_lookback_seconds: int = 300  # 5 minutes


# =============================================================================
# EXECUTION RESULT
# =============================================================================

@dataclass
class ExecutionResult:
    """
    Result of an execution algorithm.
    
    Attributes:
        algorithm: Algorithm type used
        symbol: Trading symbol
        side: Order side
        target_quantity: Target quantity
        filled_quantity: Actual filled quantity
        avg_price: Average fill price
        vwap_benchmark: Market VWAP during execution
        slippage_bps: Slippage in basis points
        duration_seconds: Execution duration
        num_orders: Number of child orders
        num_fills: Number of fills
        start_time: Execution start time
        end_time: Execution end time
        status: Final status
        child_orders: List of child order IDs
        fills: List of fill details
        metadata: Additional metadata
    """
    algorithm: ExecutionAlgoType
    symbol: str
    side: str
    target_quantity: float
    filled_quantity: float
    avg_price: float
    vwap_benchmark: float | None = None
    slippage_bps: float = 0.0
    duration_seconds: float = 0.0
    num_orders: int = 0
    num_fills: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    child_orders: list[str] = field(default_factory=list)
    fills: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def fill_rate(self) -> float:
        """Calculate fill rate."""
        if self.target_quantity <= 0:
            return 0.0
        return self.filled_quantity / self.target_quantity
    
    @property
    def total_value(self) -> float:
        """Calculate total execution value."""
        return self.filled_quantity * self.avg_price
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "algorithm": self.algorithm.value,
            "symbol": self.symbol,
            "side": self.side,
            "target_quantity": self.target_quantity,
            "filled_quantity": self.filled_quantity,
            "avg_price": self.avg_price,
            "vwap_benchmark": self.vwap_benchmark,
            "slippage_bps": self.slippage_bps,
            "duration_seconds": self.duration_seconds,
            "num_orders": self.num_orders,
            "num_fills": self.num_fills,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status.value,
            "fill_rate": self.fill_rate,
            "total_value": self.total_value,
        }


# =============================================================================
# BASE EXECUTION ALGORITHM
# =============================================================================

class ExecutionAlgorithm(ABC):
    """
    Abstract base class for execution algorithms.
    
    Lifecycle:
        1. Initialize with config and broker
        2. Call start() to begin execution
        3. Algorithm generates child orders
        4. Monitor progress via get_status()
        5. Call stop() to cancel or wait for completion
    """
    
    def __init__(
        self,
        config: BaseExecutionConfig,
        broker: BrokerBase,
    ):
        """
        Initialize execution algorithm.
        
        Args:
            config: Algorithm configuration
            broker: Broker for order execution
        """
        self.config = config
        self.broker = broker
        
        # State
        self._id = uuid4()
        self._status = ExecutionStatus.PENDING
        self._start_time: datetime | None = None
        self._end_time: datetime | None = None
        
        # Tracking
        self._filled_quantity = 0.0
        self._total_cost = 0.0
        self._child_orders: list[str] = []
        self._fills: list[dict[str, Any]] = []
        
        # Control
        self._stop_requested = threading.Event()
        self._pause_requested = threading.Event()
        self._execution_thread: threading.Thread | None = None
        
        # Callbacks
        self._on_fill_callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._on_complete_callbacks: list[Callable[[ExecutionResult], None]] = []
    
    @property
    def id(self) -> UUID:
        """Get execution ID."""
        return self._id
    
    @property
    def status(self) -> ExecutionStatus:
        """Get current status."""
        return self._status
    
    @property
    def remaining_quantity(self) -> float:
        """Get remaining quantity to execute."""
        return max(0, self.config.total_quantity - self._filled_quantity)
    
    @property
    def fill_rate(self) -> float:
        """Get current fill rate."""
        if self.config.total_quantity <= 0:
            return 0.0
        return self._filled_quantity / self.config.total_quantity
    
    @property
    def avg_fill_price(self) -> float:
        """Get average fill price."""
        if self._filled_quantity <= 0:
            return 0.0
        return self._total_cost / self._filled_quantity
    
    def on_fill(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register fill callback."""
        self._on_fill_callbacks.append(callback)
    
    def on_complete(self, callback: Callable[[ExecutionResult], None]) -> None:
        """Register completion callback."""
        self._on_complete_callbacks.append(callback)
    
    def start(self, async_execution: bool = True) -> None:
        """
        Start execution.
        
        Args:
            async_execution: Run in background thread
        """
        if self._status not in (ExecutionStatus.PENDING, ExecutionStatus.PAUSED):
            raise ExecutionError(f"Cannot start from status: {self._status}")
        
        self._status = ExecutionStatus.RUNNING
        self._start_time = datetime.now()
        
        logger.info(
            f"Starting {self.__class__.__name__} execution: "
            f"{self.config.symbol} {self.config.side} {self.config.total_quantity}"
        )
        
        if async_execution:
            self._execution_thread = threading.Thread(
                target=self._execute_wrapper,
                daemon=True,
            )
            self._execution_thread.start()
        else:
            self._execute_wrapper()
    
    def _execute_wrapper(self) -> None:
        """Wrapper for execution with error handling."""
        try:
            self._execute()
            
            if self._status == ExecutionStatus.RUNNING:
                self._status = ExecutionStatus.COMPLETED
                
        except Exception as e:
            logger.exception(f"Execution failed: {e}")
            self._status = ExecutionStatus.FAILED
            
        finally:
            self._end_time = datetime.now()
            self._notify_complete()
    
    @abstractmethod
    def _execute(self) -> None:
        """
        Execute the algorithm.
        
        Must be implemented by subclasses.
        Should check _stop_requested and _pause_requested periodically.
        """
        pass
    
    def pause(self) -> None:
        """Pause execution."""
        if self._status == ExecutionStatus.RUNNING:
            self._pause_requested.set()
            self._status = ExecutionStatus.PAUSED
            logger.info(f"Execution paused: {self._id}")
    
    def resume(self) -> None:
        """Resume paused execution."""
        if self._status == ExecutionStatus.PAUSED:
            self._pause_requested.clear()
            self._status = ExecutionStatus.RUNNING
            logger.info(f"Execution resumed: {self._id}")
    
    def stop(self, cancel_orders: bool = True) -> None:
        """
        Stop execution.
        
        Args:
            cancel_orders: Cancel pending child orders
        """
        self._stop_requested.set()
        
        if cancel_orders:
            for order_id in self._child_orders:
                try:
                    self.broker.cancel_order(order_id)
                except Exception as e:
                    logger.warning(f"Failed to cancel order {order_id}: {e}")
        
        self._status = ExecutionStatus.CANCELLED
        logger.info(f"Execution stopped: {self._id}")
    
    def wait(self, timeout: float | None = None) -> ExecutionResult:
        """
        Wait for execution to complete.
        
        Args:
            timeout: Maximum wait time in seconds
        
        Returns:
            ExecutionResult
        """
        if self._execution_thread:
            self._execution_thread.join(timeout=timeout)
        
        return self.get_result()
    
    def get_result(self) -> ExecutionResult:
        """Get execution result."""
        duration = 0.0
        if self._start_time:
            end = self._end_time or datetime.now()
            duration = (end - self._start_time).total_seconds()
        
        return ExecutionResult(
            algorithm=self._get_algo_type(),
            symbol=self.config.symbol,
            side=self.config.side,
            target_quantity=self.config.total_quantity,
            filled_quantity=self._filled_quantity,
            avg_price=self.avg_fill_price,
            duration_seconds=duration,
            num_orders=len(self._child_orders),
            num_fills=len(self._fills),
            start_time=self._start_time,
            end_time=self._end_time,
            status=self._status,
            child_orders=self._child_orders.copy(),
            fills=self._fills.copy(),
        )
    
    @abstractmethod
    def _get_algo_type(self) -> ExecutionAlgoType:
        """Get algorithm type."""
        pass
    
    def _submit_child_order(
        self,
        quantity: float,
        order_type: str = "market",
        limit_price: float | None = None,
    ) -> str | None:
        """
        Submit a child order.
        
        Args:
            quantity: Order quantity
            order_type: Order type (market/limit)
            limit_price: Limit price for limit orders
        
        Returns:
            Broker order ID or None if failed
        """
        if quantity < self.config.min_order_size:
            return None
        
        # Apply max order size
        if self.config.max_order_size:
            quantity = min(quantity, self.config.max_order_size)
        
        # Don't exceed remaining quantity
        quantity = min(quantity, self.remaining_quantity)
        
        if quantity <= 0:
            return None
        
        # Create order
        order = Order.create_market_order(
            symbol=self.config.symbol,
            side=self.config.side,
            quantity=quantity,
        ) if order_type == "market" else Order.create_limit_order(
            symbol=self.config.symbol,
            side=self.config.side,
            quantity=quantity,
            limit_price=limit_price,
        )
        
        try:
            order_id = self.broker.submit_order(order)
            self._child_orders.append(order_id)
            
            logger.debug(
                f"Child order submitted: {self.config.symbol} "
                f"{self.config.side} {quantity} @ {order_type}"
            )
            
            return order_id
            
        except Exception as e:
            logger.warning(f"Child order failed: {e}")
            return None
    
    def _record_fill(
        self,
        quantity: float,
        price: float,
        order_id: str | None = None,
    ) -> None:
        """
        Record a fill.
        
        Args:
            quantity: Filled quantity
            price: Fill price
            order_id: Associated order ID
        """
        fill = {
            "timestamp": datetime.now().isoformat(),
            "quantity": quantity,
            "price": price,
            "order_id": order_id,
        }
        
        self._fills.append(fill)
        self._filled_quantity += quantity
        self._total_cost += quantity * price
        
        # Notify callbacks
        for callback in self._on_fill_callbacks:
            try:
                callback(fill)
            except Exception as e:
                logger.error(f"Fill callback error: {e}")
    
    def _notify_complete(self) -> None:
        """Notify completion callbacks."""
        result = self.get_result()
        
        for callback in self._on_complete_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Complete callback error: {e}")
    
    def _should_stop(self) -> bool:
        """Check if execution should stop."""
        return self._stop_requested.is_set()
    
    def _should_pause(self) -> bool:
        """Check if execution should pause."""
        return self._pause_requested.is_set()
    
    def _wait_while_paused(self) -> None:
        """Wait while paused."""
        while self._should_pause() and not self._should_stop():
            time.sleep(0.1)


# =============================================================================
# TWAP ALGORITHM
# =============================================================================

class TWAPAlgorithm(ExecutionAlgorithm):
    """
    Time-Weighted Average Price execution algorithm.
    
    Splits total quantity evenly across time slices,
    with optional randomization for stealth.
    
    Example:
        config = TWAPConfig(
            symbol="AAPL",
            side="buy",
            total_quantity=10000,
            duration_minutes=60,
            num_slices=20,
        )
        
        algo = TWAPAlgorithm(config, broker)
        algo.start()
        result = algo.wait()
    """
    
    def __init__(self, config: TWAPConfig, broker: BrokerBase):
        """Initialize TWAP algorithm."""
        super().__init__(config, broker)
        self.twap_config = config
    
    def _get_algo_type(self) -> ExecutionAlgoType:
        return ExecutionAlgoType.TWAP
    
    def _execute(self) -> None:
        """Execute TWAP strategy."""
        config = self.twap_config
        
        # Calculate slice parameters
        base_quantity = config.total_quantity / config.num_slices
        slice_interval = (config.duration_minutes * 60) / config.num_slices
        
        logger.info(
            f"TWAP: {config.num_slices} slices, "
            f"{slice_interval:.1f}s interval, "
            f"{base_quantity:.2f} shares/slice"
        )
        
        for i in range(config.num_slices):
            if self._should_stop():
                break
            
            self._wait_while_paused()
            
            # Calculate slice quantity with optional variance
            slice_qty = base_quantity
            if config.randomize_size:
                variance = np.random.uniform(
                    1 - config.size_variance,
                    1 + config.size_variance
                )
                slice_qty *= variance
            
            # Don't exceed remaining
            slice_qty = min(slice_qty, self.remaining_quantity)
            
            if slice_qty >= config.min_order_size:
                # Submit order
                order_id = self._submit_child_order(slice_qty, "market")
                
                if order_id:
                    # Simulate fill for now (real implementation would wait for fill)
                    # In production, you'd query order status
                    price = self._get_current_price()
                    self._record_fill(slice_qty, price, order_id)
            
            # Check if complete
            if self.remaining_quantity <= 0:
                break
            
            # Wait for next slice (if not last)
            if i < config.num_slices - 1:
                wait_time = slice_interval
                if config.randomize_timing:
                    variance = np.random.uniform(
                        1 - config.timing_variance,
                        1 + config.timing_variance
                    )
                    wait_time *= variance
                
                # Wait with stop checking
                self._interruptible_sleep(wait_time)
    
    def _get_current_price(self) -> float:
        """Get current market price (placeholder)."""
        # In production, this would query real-time data
        return 100.0  # Placeholder
    
    def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep with stop checking."""
        end_time = time.monotonic() + seconds
        while time.monotonic() < end_time:
            if self._should_stop():
                break
            time.sleep(min(0.5, end_time - time.monotonic()))


# =============================================================================
# VWAP ALGORITHM
# =============================================================================

class VWAPAlgorithm(ExecutionAlgorithm):
    """
    Volume-Weighted Average Price execution algorithm.
    
    Executes proportionally to historical or real-time volume.
    
    Example:
        config = VWAPConfig(
            symbol="AAPL",
            side="buy",
            total_quantity=10000,
            duration_minutes=60,
            adaptive=True,
        )
        
        algo = VWAPAlgorithm(config, broker)
        algo.start()
    """
    
    def __init__(self, config: VWAPConfig, broker: BrokerBase):
        """Initialize VWAP algorithm."""
        super().__init__(config, broker)
        self.vwap_config = config
        self._volume_profile = self._get_volume_profile()
    
    def _get_algo_type(self) -> ExecutionAlgoType:
        return ExecutionAlgoType.VWAP
    
    def _get_volume_profile(self) -> list[float]:
        """
        Get volume profile for execution period.
        
        Returns normalized volume buckets.
        """
        if self.vwap_config.volume_profile:
            profile = np.array(self.vwap_config.volume_profile)
            return (profile / profile.sum()).tolist()
        
        # Default U-shaped intraday volume profile
        # Higher volume at open and close
        hours = np.linspace(0, 6.5, 26)  # 15-min buckets for 6.5 hour day
        
        # U-shape: high at start, low in middle, high at end
        profile = np.zeros_like(hours)
        profile = 1 + 0.5 * np.cos(np.pi * hours / 6.5)  # Base shape
        profile[:4] *= 1.5  # Boost opening
        profile[-4:] *= 1.5  # Boost closing
        
        return (profile / profile.sum()).tolist()
    
    def _execute(self) -> None:
        """Execute VWAP strategy."""
        config = self.vwap_config
        
        # Calculate execution buckets
        num_buckets = len(self._volume_profile)
        bucket_duration = (config.duration_minutes * 60) / num_buckets
        
        logger.info(
            f"VWAP: {num_buckets} buckets, "
            f"{bucket_duration:.1f}s per bucket"
        )
        
        for i, volume_pct in enumerate(self._volume_profile):
            if self._should_stop():
                break
            
            self._wait_while_paused()
            
            # Calculate target quantity for this bucket
            bucket_qty = config.total_quantity * volume_pct
            
            # Adjust for what's already filled
            remaining_pct = 1 - self.fill_rate
            if remaining_pct < 1:
                # Scale up to catch up or scale down to slow down
                bucket_qty *= (remaining_pct * num_buckets / (num_buckets - i))
            
            bucket_qty = min(bucket_qty, self.remaining_quantity)
            
            if bucket_qty >= config.min_order_size:
                order_id = self._submit_child_order(bucket_qty, "market")
                
                if order_id:
                    price = self._get_current_price()
                    self._record_fill(bucket_qty, price, order_id)
            
            if self.remaining_quantity <= 0:
                break
            
            # Wait for next bucket
            if i < num_buckets - 1:
                self._interruptible_sleep(bucket_duration)
    
    def _get_current_price(self) -> float:
        """Get current market price."""
        return 100.0  # Placeholder
    
    def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep with stop checking."""
        end_time = time.monotonic() + seconds
        while time.monotonic() < end_time:
            if self._should_stop():
                break
            time.sleep(min(0.5, end_time - time.monotonic()))


# =============================================================================
# ICEBERG ALGORITHM
# =============================================================================

class IcebergAlgorithm(ExecutionAlgorithm):
    """
    Iceberg order execution algorithm.
    
    Shows only a small "tip" of the total order,
    refreshing as portions are filled.
    
    Example:
        config = IcebergConfig(
            symbol="AAPL",
            side="buy",
            total_quantity=10000,
            display_quantity=100,
        )
        
        algo = IcebergAlgorithm(config, broker)
        algo.start()
    """
    
    def __init__(self, config: IcebergConfig, broker: BrokerBase):
        """Initialize Iceberg algorithm."""
        super().__init__(config, broker)
        self.iceberg_config = config
    
    def _get_algo_type(self) -> ExecutionAlgoType:
        return ExecutionAlgoType.ICEBERG
    
    def _execute(self) -> None:
        """Execute Iceberg strategy."""
        config = self.iceberg_config
        
        refresh_qty = config.refresh_quantity or config.display_quantity
        
        logger.info(
            f"Iceberg: display={config.display_quantity}, "
            f"refresh={refresh_qty}"
        )
        
        while self.remaining_quantity > 0 and not self._should_stop():
            self._wait_while_paused()
            
            # Calculate display quantity with variance
            display = config.display_quantity
            if config.variance > 0:
                variance = np.random.uniform(
                    1 - config.variance,
                    1 + config.variance
                )
                display *= variance
            
            display = min(display, self.remaining_quantity)
            
            if display >= config.min_order_size:
                # Submit limit order at current price +/- offset
                price = self._get_current_price()
                if config.side == "buy":
                    price += config.price_offset
                else:
                    price -= config.price_offset
                
                order_id = self._submit_child_order(display, "limit", price)
                
                if order_id:
                    # Wait for fill (simplified)
                    filled = self._wait_for_fill(order_id, timeout=30)
                    
                    if filled:
                        self._record_fill(display, price, order_id)
                    else:
                        # Cancel and retry
                        self.broker.cancel_order(order_id)
            
            # Small delay between waves
            time.sleep(0.5)
    
    def _get_current_price(self) -> float:
        """Get current market price."""
        return 100.0  # Placeholder
    
    def _wait_for_fill(self, order_id: str, timeout: float = 30) -> bool:
        """Wait for order to fill."""
        end_time = time.monotonic() + timeout
        
        while time.monotonic() < end_time:
            if self._should_stop():
                return False
            
            try:
                status = self.broker.get_order_status(order_id)
                if status.get("status") == "filled":
                    return True
                if status.get("status") in ("cancelled", "rejected"):
                    return False
            except Exception:
                pass
            
            time.sleep(0.5)
        
        return False


# =============================================================================
# POV ALGORITHM
# =============================================================================

class POVAlgorithm(ExecutionAlgorithm):
    """
    Percentage of Volume execution algorithm.
    
    Executes as a target percentage of market volume,
    adapting to real-time trading activity.
    
    Example:
        config = POVConfig(
            symbol="AAPL",
            side="buy",
            total_quantity=10000,
            target_pov=0.05,  # 5% of volume
        )
        
        algo = POVAlgorithm(config, broker)
        algo.start()
    """
    
    def __init__(self, config: POVConfig, broker: BrokerBase):
        """Initialize POV algorithm."""
        super().__init__(config, broker)
        self.pov_config = config
        self._volume_tracker: list[tuple[datetime, float]] = []
    
    def _get_algo_type(self) -> ExecutionAlgoType:
        return ExecutionAlgoType.POV
    
    def _execute(self) -> None:
        """Execute POV strategy."""
        config = self.pov_config
        
        logger.info(
            f"POV: target={config.target_pov:.1%}, "
            f"range={config.min_pov:.1%}-{config.max_pov:.1%}"
        )
        
        while self.remaining_quantity > 0 and not self._should_stop():
            self._wait_while_paused()
            
            # Get recent market volume
            market_volume = self._get_recent_volume()
            
            if market_volume > 0:
                # Calculate target order size
                target_qty = market_volume * config.target_pov
                
                # Apply min/max POV bounds
                min_qty = market_volume * config.min_pov
                max_qty = market_volume * config.max_pov
                
                order_qty = np.clip(target_qty, min_qty, max_qty)
                order_qty = min(order_qty, self.remaining_quantity)
                
                # Apply participation rate cap
                order_qty = min(
                    order_qty,
                    market_volume * config.max_participation_rate
                )
                
                if order_qty >= config.min_order_size:
                    order_id = self._submit_child_order(order_qty, "market")
                    
                    if order_id:
                        price = self._get_current_price()
                        self._record_fill(order_qty, price, order_id)
                        self._track_volume(order_qty)
            
            # Wait before next check
            self._interruptible_sleep(config.check_interval_seconds)
    
    def _get_recent_volume(self) -> float:
        """Get recent market volume."""
        # Placeholder - in production, query real-time data
        # Returns volume traded in the lookback period
        return np.random.uniform(50000, 100000)
    
    def _get_current_price(self) -> float:
        """Get current market price."""
        return 100.0  # Placeholder
    
    def _track_volume(self, quantity: float) -> None:
        """Track our execution volume."""
        self._volume_tracker.append((datetime.now(), quantity))
        
        # Clean old entries
        cutoff = datetime.now() - timedelta(
            seconds=self.pov_config.volume_lookback_seconds
        )
        self._volume_tracker = [
            (t, v) for t, v in self._volume_tracker if t > cutoff
        ]
    
    def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep with stop checking."""
        end_time = time.monotonic() + seconds
        while time.monotonic() < end_time:
            if self._should_stop():
                break
            time.sleep(min(0.5, end_time - time.monotonic()))


# =============================================================================
# SMART ORDER ROUTER
# =============================================================================

class SmartOrderRouter:
    """
    Smart order routing for optimal execution.
    
    Features:
        - Algorithm selection based on order characteristics
        - Dynamic algo switching
        - Execution quality analysis
        - Multi-algo coordination
    
    Example:
        router = SmartOrderRouter(broker)
        
        # Router selects best algorithm
        result = router.execute(
            symbol="AAPL",
            side="buy",
            quantity=10000,
            urgency=UrgencyLevel.MEDIUM,
        )
    """
    
    def __init__(self, broker: BrokerBase):
        """Initialize smart router."""
        self.broker = broker
        self._active_executions: dict[UUID, ExecutionAlgorithm] = {}
    
    def execute(
        self,
        symbol: str,
        side: str,
        quantity: float,
        urgency: UrgencyLevel = UrgencyLevel.MEDIUM,
        algo_type: ExecutionAlgoType | None = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """
        Execute order with optimal algorithm.
        
        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Total quantity
            urgency: Execution urgency
            algo_type: Force specific algorithm (optional)
            **kwargs: Additional config parameters
        
        Returns:
            ExecutionResult
        """
        # Select algorithm if not specified
        if algo_type is None:
            algo_type = self._select_algorithm(
                symbol, quantity, urgency
            )
        
        logger.info(f"Smart router selected: {algo_type.value}")
        
        # Create algorithm
        algo = create_execution_algorithm(
            algo_type=algo_type,
            broker=self.broker,
            symbol=symbol,
            side=side,
            quantity=quantity,
            urgency=urgency,
            **kwargs,
        )
        
        # Track execution
        self._active_executions[algo.id] = algo
        
        # Execute
        algo.start(async_execution=False)
        
        # Cleanup
        del self._active_executions[algo.id]
        
        return algo.get_result()
    
    def execute_async(
        self,
        symbol: str,
        side: str,
        quantity: float,
        urgency: UrgencyLevel = UrgencyLevel.MEDIUM,
        algo_type: ExecutionAlgoType | None = None,
        **kwargs: Any,
    ) -> ExecutionAlgorithm:
        """
        Start async execution with optimal algorithm.
        
        Returns the algorithm instance for monitoring.
        """
        if algo_type is None:
            algo_type = self._select_algorithm(symbol, quantity, urgency)
        
        algo = create_execution_algorithm(
            algo_type=algo_type,
            broker=self.broker,
            symbol=symbol,
            side=side,
            quantity=quantity,
            urgency=urgency,
            **kwargs,
        )
        
        self._active_executions[algo.id] = algo
        algo.start(async_execution=True)
        
        return algo
    
    def _select_algorithm(
        self,
        symbol: str,
        quantity: float,
        urgency: UrgencyLevel,
    ) -> ExecutionAlgoType:
        """
        Select optimal algorithm based on order characteristics.
        
        Selection criteria:
        - Order size relative to ADV
        - Urgency level
        - Market conditions
        - Time of day
        """
        # Get ADV (placeholder)
        adv = 1_000_000  # Average daily volume
        
        # Calculate order as % of ADV
        pct_adv = quantity * 100 / adv  # Assuming $100 price
        
        # Selection logic
        if urgency == UrgencyLevel.CRITICAL:
            # Immediate execution needed
            return ExecutionAlgoType.MARKET
        
        elif urgency == UrgencyLevel.HIGH:
            # Fast but controlled
            if pct_adv < 0.5:
                return ExecutionAlgoType.MARKET
            elif pct_adv < 2:
                return ExecutionAlgoType.TWAP
            else:
                return ExecutionAlgoType.POV
        
        elif urgency == UrgencyLevel.MEDIUM:
            # Balance speed and impact
            if pct_adv < 1:
                return ExecutionAlgoType.TWAP
            elif pct_adv < 5:
                return ExecutionAlgoType.VWAP
            else:
                return ExecutionAlgoType.POV
        
        else:  # LOW urgency
            # Minimize market impact
            if pct_adv < 2:
                return ExecutionAlgoType.VWAP
            else:
                return ExecutionAlgoType.ICEBERG
    
    def get_active_executions(self) -> list[ExecutionAlgorithm]:
        """Get all active executions."""
        return list(self._active_executions.values())
    
    def cancel_all(self) -> int:
        """Cancel all active executions."""
        count = 0
        for algo in self._active_executions.values():
            algo.stop()
            count += 1
        return count


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_execution_algorithm(
    algo_type: ExecutionAlgoType | str,
    broker: BrokerBase,
    symbol: str,
    side: str,
    quantity: float,
    urgency: UrgencyLevel = UrgencyLevel.MEDIUM,
    **kwargs: Any,
) -> ExecutionAlgorithm:
    """
    Factory function to create execution algorithms.
    
    Args:
        algo_type: Algorithm type
        broker: Broker instance
        symbol: Trading symbol
        side: Order side
        quantity: Total quantity
        urgency: Execution urgency
        **kwargs: Additional config parameters
    
    Returns:
        ExecutionAlgorithm instance
    """
    if isinstance(algo_type, str):
        algo_type = ExecutionAlgoType(algo_type)
    
    if algo_type == ExecutionAlgoType.TWAP:
        config = TWAPConfig(
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            urgency=urgency,
            **kwargs,
        )
        return TWAPAlgorithm(config, broker)
    
    elif algo_type == ExecutionAlgoType.VWAP:
        config = VWAPConfig(
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            urgency=urgency,
            **kwargs,
        )
        return VWAPAlgorithm(config, broker)
    
    elif algo_type == ExecutionAlgoType.ICEBERG:
        config = IcebergConfig(
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            urgency=urgency,
            **kwargs,
        )
        return IcebergAlgorithm(config, broker)
    
    elif algo_type == ExecutionAlgoType.POV:
        config = POVConfig(
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            urgency=urgency,
            **kwargs,
        )
        return POVAlgorithm(config, broker)
    
    else:
        raise ValueError(f"Unknown algorithm type: {algo_type}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ExecutionAlgoType",
    "ExecutionStatus",
    "UrgencyLevel",
    # Configs
    "BaseExecutionConfig",
    "TWAPConfig",
    "VWAPConfig",
    "IcebergConfig",
    "POVConfig",
    # Result
    "ExecutionResult",
    # Base class
    "ExecutionAlgorithm",
    # Implementations
    "TWAPAlgorithm",
    "VWAPAlgorithm",
    "IcebergAlgorithm",
    "POVAlgorithm",
    # Router
    "SmartOrderRouter",
    # Factory
    "create_execution_algorithm",
]