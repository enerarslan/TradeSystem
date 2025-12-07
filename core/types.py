"""
Core Types Module
=================

Core data structures, enums, and exceptions for the algorithmic trading platform.

CRITICAL FIXES APPLIED:
- Trade.create() requires entry_time parameter (no datetime.now() default)
- Trade.close() requires exit_time parameter (no datetime.now() default)
- Position.add() requires timestamp parameter
- Position.update_price() requires timestamp parameter
- Position.reduce() requires timestamp parameter

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4
from enum import Enum

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# EXCEPTIONS
# =============================================================================

class AlgoTradingError(Exception):
    """Base exception for all trading errors."""
    pass


class DataError(AlgoTradingError):
    """Data-related errors."""
    pass


class DataNotFoundError(DataError):
    """Data not found."""
    pass


class DataValidationError(DataError):
    """Data validation failed."""
    pass


class StrategyError(AlgoTradingError):
    """Strategy-related errors."""
    pass


class StrategyInitializationError(StrategyError):
    """Strategy initialization failed."""
    pass


class SignalGenerationError(StrategyError):
    """Signal generation failed."""
    pass


class ExecutionError(AlgoTradingError):
    """Execution-related errors."""
    pass


class OrderRejectedError(ExecutionError):
    """Order was rejected."""
    pass


class InsufficientFundsError(ExecutionError):
    """Insufficient funds for order."""
    pass


class RiskError(AlgoTradingError):
    """Risk-related errors."""
    pass


class RiskLimitExceededError(RiskError):
    """Risk limit exceeded."""
    pass


class DrawdownLimitError(RiskError):
    """Drawdown limit exceeded."""
    pass


class ModelError(AlgoTradingError):
    """Model-related errors."""
    pass


class ModelNotTrainedError(ModelError):
    """Model not trained."""
    pass


class PredictionError(ModelError):
    """Prediction failed."""
    pass


class BacktestError(AlgoTradingError):
    """Backtest-related errors."""
    pass


class ConfigurationError(AlgoTradingError):
    """Configuration error."""
    pass


# =============================================================================
# ENUMS
# =============================================================================

class OrderStatus(str, Enum):
    """Order lifecycle status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionStatus(str, Enum):
    """Position status."""
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"


class FillType(str, Enum):
    """Order fill type."""
    FULL = "full"
    PARTIAL = "partial"


class SignalStrength(str, Enum):
    """Signal confidence level."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True, slots=True)
class OHLCV:
    """Immutable OHLCV bar representation."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""
    timeframe: str = "15min"
    
    def __post_init__(self) -> None:
        """Validate OHLCV data."""
        if self.high < self.low:
            raise DataValidationError(
                f"High ({self.high}) cannot be less than low ({self.low})"
            )
        if self.high < self.open or self.high < self.close:
            raise DataValidationError("High must be >= open and close")
        if self.low > self.open or self.low > self.close:
            raise DataValidationError("Low must be <= open and close")
        if self.volume < 0:
            raise DataValidationError("Volume cannot be negative")
    
    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3
    
    @property
    def median_price(self) -> float:
        return (self.high + self.low) / 2
    
    @property
    def weighted_close(self) -> float:
        return (self.high + self.low + 2 * self.close) / 4
    
    @property
    def bar_range(self) -> float:
        return self.high - self.low
    
    @property
    def body(self) -> float:
        return self.close - self.open
    
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open


@dataclass(slots=True)
class Bar:
    """Mutable bar with additional metadata."""
    ohlcv: OHLCV
    vwap: float | None = None
    trades: int = 0
    
    @property
    def timestamp(self) -> datetime:
        return self.ohlcv.timestamp


# =============================================================================
# SIGNAL
# =============================================================================

@dataclass(frozen=True, slots=True)
class Signal:
    """Trading signal representation."""
    id: UUID
    timestamp: datetime
    symbol: str
    signal_type: str
    direction: str
    strength: SignalStrength
    price: float
    strategy_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        timestamp: datetime,
        symbol: str,
        signal_type: str,
        direction: str,
        strength: SignalStrength,
        price: float,
        strategy_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> "Signal":
        """Factory method to create a signal."""
        return cls(
            id=uuid4(),
            timestamp=timestamp,
            symbol=symbol,
            signal_type=signal_type,
            direction=direction,
            strength=strength,
            price=price,
            strategy_name=strategy_name,
            metadata=metadata or {},
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "signal_type": self.signal_type,
            "direction": self.direction,
            "strength": self.strength.value,
            "price": self.price,
            "strategy_name": self.strategy_name,
            "metadata": self.metadata,
        }


# =============================================================================
# ORDER
# =============================================================================

@dataclass(slots=True)
class Order:
    """Trading order representation."""
    id: UUID
    timestamp: datetime
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: float | None = None
    stop_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    signal_id: UUID | None = None
    broker_order_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create_market_order(
        cls,
        symbol: str,
        side: str,
        quantity: float,
        timestamp: datetime,
        signal_id: UUID | None = None,
    ) -> "Order":
        """Factory method to create market order."""
        return cls(
            id=uuid4(),
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            order_type="market",
            quantity=quantity,
            signal_id=signal_id,
        )
    
    @classmethod
    def create_limit_order(
        cls,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        signal_id: UUID | None = None,
    ) -> "Order":
        """Factory method to create limit order."""
        return cls(
            id=uuid4(),
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            order_type="limit",
            quantity=quantity,
            price=price,
            signal_id=signal_id,
        )
    
    def fill(
        self,
        quantity: float,
        price: float,
        commission: float = 0.0,
    ) -> None:
        """Record a fill for this order."""
        self.filled_quantity += quantity
        total_cost = (self.filled_price * (self.filled_quantity - quantity) +
                      price * quantity)
        if self.filled_quantity > 0:
            self.filled_price = total_cost / self.filled_quantity
        self.commission += commission
        
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIAL
    
    def cancel(self) -> None:
        """Cancel this order."""
        self.status = OrderStatus.CANCELLED
    
    def reject(self, reason: str = "") -> None:
        """Reject this order."""
        self.status = OrderStatus.REJECTED
        self.metadata["rejection_reason"] = reason
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in (
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIAL,
        )
    
    @property
    def remaining_quantity(self) -> float:
        """Calculate remaining quantity to fill."""
        return self.quantity - self.filled_quantity
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "filled_price": self.filled_price,
            "commission": self.commission,
            "signal_id": str(self.signal_id) if self.signal_id else None,
            "broker_order_id": self.broker_order_id,
            "metadata": self.metadata,
        }


# =============================================================================
# TRADE (FIXED - NO datetime.now()!)
# =============================================================================

@dataclass(slots=True)
class Trade:
    """
    Completed trade record.
    
    CRITICAL: entry_time and exit_time must be explicitly provided.
    They should be the simulated market timestamps, NOT datetime.now().
    """
    id: UUID
    entry_time: datetime
    symbol: str
    side: str
    entry_price: float
    quantity: float
    exit_time: datetime | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    entry_order_id: UUID | None = None
    exit_order_id: UUID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        entry_time: datetime,
        entry_order_id: UUID | None = None,
    ) -> "Trade":
        """
        Factory method to create a new trade.
        
        Args:
            symbol: Trading symbol
            side: Trade side ('long' or 'short')
            entry_price: Entry price
            quantity: Trade quantity
            entry_time: Entry timestamp (MUST be simulated market time!)
            entry_order_id: Optional entry order ID
        """
        return cls(
            id=uuid4(),
            entry_time=entry_time,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            entry_order_id=entry_order_id,
        )
    
    def close(
        self,
        exit_price: float,
        exit_time: datetime,
        commission: float = 0.0,
        exit_order_id: UUID | None = None,
    ) -> None:
        """
        Close the trade.
        
        Args:
            exit_price: Exit price
            exit_time: Exit timestamp (MUST be simulated market time!)
            commission: Commission for exit
            exit_order_id: Optional exit order ID
        """
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_order_id = exit_order_id
        self.commission += commission
        
        # Calculate PnL
        if self.side == "long":
            self.pnl = (exit_price - self.entry_price) * self.quantity - self.commission
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.quantity - self.commission
        
        # Calculate PnL percentage
        cost_basis = self.entry_price * self.quantity
        if cost_basis != 0:
            self.pnl_pct = self.pnl / cost_basis
        else:
            self.pnl_pct = 0.0
    
    @property
    def is_open(self) -> bool:
        """Check if trade is still open."""
        return self.exit_time is None
    
    @property
    def duration(self) -> float | None:
        """Calculate trade duration in seconds."""
        if self.exit_time is None:
            return None
        return (self.exit_time - self.entry_time).total_seconds()
    
    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "commission": self.commission,
            "entry_order_id": str(self.entry_order_id) if self.entry_order_id else None,
            "exit_order_id": str(self.exit_order_id) if self.exit_order_id else None,
            "metadata": self.metadata,
        }


# =============================================================================
# POSITION (FIXED - REQUIRES TIMESTAMP PARAMETER!)
# =============================================================================

@dataclass(slots=True)
class Position:
    """
    Current position in a symbol.
    
    CRITICAL: All methods that modify state require a timestamp parameter.
    This must be the simulated market time, NOT datetime.now().
    """
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    opened_at: datetime | None = None
    updated_at: datetime | None = None
    trades: list[UUID] = field(default_factory=list)
    
    @property
    def side(self) -> str:
        """Get position side."""
        if self.quantity > 0:
            return "long"
        elif self.quantity < 0:
            return "short"
        return "flat"
    
    @property
    def market_value(self) -> float:
        """Calculate current market value."""
        return abs(self.quantity) * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Calculate cost basis."""
        return abs(self.quantity) * self.avg_price
    
    @property
    def total_pnl(self) -> float:
        """Calculate total PnL (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def pnl_pct(self) -> float:
        """Calculate PnL percentage."""
        if self.cost_basis == 0:
            return 0.0
        return self.total_pnl / self.cost_basis
    
    @property
    def is_open(self) -> bool:
        """Check if position is open."""
        return self.quantity != 0
    
    def update_price(self, price: float, timestamp: datetime) -> None:
        """
        Update current price and unrealized PnL.
        
        Args:
            price: Current market price
            timestamp: Current simulated market time (NOT datetime.now()!)
        """
        self.current_price = price
        self.updated_at = timestamp
        
        if self.quantity > 0:  # Long
            self.unrealized_pnl = (price - self.avg_price) * self.quantity
        elif self.quantity < 0:  # Short
            self.unrealized_pnl = (self.avg_price - price) * abs(self.quantity)
        else:
            self.unrealized_pnl = 0.0
    
    def add(self, quantity: float, price: float, timestamp: datetime) -> None:
        """
        Add to position.
        
        Args:
            quantity: Quantity to add (positive for long, negative for short)
            price: Execution price
            timestamp: Current simulated market time (NOT datetime.now()!)
        """
        if self.quantity == 0:
            self.opened_at = timestamp
        
        # Calculate new average price
        total_cost = (self.avg_price * abs(self.quantity)) + (price * abs(quantity))
        self.quantity += quantity
        
        if self.quantity != 0:
            self.avg_price = total_cost / abs(self.quantity)
        
        self.update_price(price, timestamp)
    
    def reduce(self, quantity: float, price: float, timestamp: datetime) -> float:
        """
        Reduce position and return realized PnL.
        
        Args:
            quantity: Quantity to reduce (always positive)
            price: Execution price
            timestamp: Current simulated market time (NOT datetime.now()!)
            
        Returns:
            Realized PnL from this reduction
        """
        if abs(quantity) > abs(self.quantity):
            quantity = abs(self.quantity)
        
        # Calculate realized PnL for this portion
        if self.quantity > 0:  # Long position
            realized = (price - self.avg_price) * quantity
            self.quantity -= quantity
        else:  # Short position
            realized = (self.avg_price - price) * quantity
            self.quantity += quantity
        
        self.update_price(price, timestamp)
        
        return realized
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "side": self.side,
            "avg_price": self.avg_price,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "cost_basis": self.cost_basis,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_pnl": self.total_pnl,
            "pnl_pct": self.pnl_pct,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "trades": [str(t) for t in self.trades],
        }


# =============================================================================
# PORTFOLIO STATE
# =============================================================================

@dataclass
class PortfolioState:
    """Current portfolio state snapshot."""
    timestamp: datetime
    cash: float
    equity: float
    buying_power: float
    positions: dict[str, Position]
    open_orders: list[Order]
    total_pnl: float
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cash": self.cash,
            "equity": self.equity,
            "buying_power": self.buying_power,
            "total_pnl": self.total_pnl,
            "num_positions": len([p for p in self.positions.values() if p.is_open]),
            "num_open_orders": len(self.open_orders),
        }


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

@dataclass
class PerformanceMetrics:
    """All standard institutional-grade metrics for strategy evaluation."""
    total_return: float = 0.0
    annual_return: float = 0.0
    monthly_returns: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    daily_returns: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    
    volatility: float = 0.0
    annual_volatility: float = 0.0
    downside_volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade: float = 0.0
    avg_holding_period: float = 0.0
    
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    
    skewness: float = 0.0
    kurtosis: float = 0.0
    tail_ratio: float = 0.0
    
    @property
    def expectancy(self) -> float:
        """Calculate trade expectancy."""
        return (self.win_rate * self.avg_win) - ((1 - self.win_rate) * abs(self.avg_loss))
    
    @property
    def payoff_ratio(self) -> float:
        """Calculate payoff ratio (avg win / avg loss)."""
        if self.avg_loss == 0:
            return float('inf')
        return abs(self.avg_win / self.avg_loss)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excluding large arrays)."""
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "volatility": self.volatility,
            "annual_volatility": self.annual_volatility,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "var_95": self.var_95,
            "var_99": self.var_99,
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data structures
    "OHLCV",
    "Bar",
    "Trade",
    "Position",
    "Order",
    "Signal",
    "SignalStrength",
    "PortfolioState",
    "PerformanceMetrics",
    # Enums
    "OrderStatus",
    "PositionStatus",
    "FillType",
    # Exceptions
    "AlgoTradingError",
    "DataError",
    "DataNotFoundError",
    "DataValidationError",
    "StrategyError",
    "StrategyInitializationError",
    "SignalGenerationError",
    "ExecutionError",
    "OrderRejectedError",
    "InsufficientFundsError",
    "RiskError",
    "RiskLimitExceededError",
    "DrawdownLimitError",
    "ModelError",
    "ModelNotTrainedError",
    "PredictionError",
    "BacktestError",
    "ConfigurationError",
]