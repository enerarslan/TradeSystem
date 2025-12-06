"""
Core Types Module
=================

Domain objects and data structures for the algorithmic trading platform.
All types are immutable dataclasses for thread-safety and clarity.

Features:
- OHLCV bar representation
- Trade and position tracking
- Order management
- Signal generation
- Portfolio state
- Performance metrics
- Custom exceptions

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# EXCEPTIONS
# =============================================================================

class AlgoTradingError(Exception):
    """Base exception for all trading platform errors."""
    pass


class DataError(AlgoTradingError):
    """Errors related to data loading and processing."""
    pass


class DataNotFoundError(DataError):
    """Requested data not found."""
    pass


class DataValidationError(DataError):
    """Data failed validation checks."""
    pass


class StrategyError(AlgoTradingError):
    """Errors in strategy execution."""
    pass


class StrategyInitializationError(StrategyError):
    """Strategy failed to initialize."""
    pass


class SignalGenerationError(StrategyError):
    """Error generating trading signal."""
    pass


class ExecutionError(AlgoTradingError):
    """Errors in order execution."""
    pass


class OrderRejectedError(ExecutionError):
    """Order was rejected by broker."""
    pass


class InsufficientFundsError(ExecutionError):
    """Insufficient funds for order."""
    pass


class RiskError(AlgoTradingError):
    """Risk management related errors."""
    pass


class RiskLimitExceededError(RiskError):
    """Risk limit has been exceeded."""
    pass


class DrawdownLimitError(RiskError):
    """Maximum drawdown exceeded."""
    pass


class ModelError(AlgoTradingError):
    """Machine learning model errors."""
    pass


class ModelNotTrainedError(ModelError):
    """Model has not been trained."""
    pass


class PredictionError(ModelError):
    """Error during prediction."""
    pass


class BacktestError(AlgoTradingError):
    """Backtesting related errors."""
    pass


class ConfigurationError(AlgoTradingError):
    """Configuration related errors."""
    pass


# =============================================================================
# ENUMS (Additional domain-specific)
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
    """
    Immutable OHLCV bar representation.
    
    Attributes:
        timestamp: Bar timestamp
        open: Opening price
        high: Highest price
        low: Lowest price
        close: Closing price
        volume: Trading volume
        symbol: Trading symbol
        timeframe: Bar timeframe
    """
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
        """Calculate typical price (HLC/3)."""
        return (self.high + self.low + self.close) / 3
    
    @property
    def median_price(self) -> float:
        """Calculate median price (HL/2)."""
        return (self.high + self.low) / 2
    
    @property
    def weighted_close(self) -> float:
        """Calculate weighted close (HLCC/4)."""
        return (self.high + self.low + 2 * self.close) / 4
    
    @property
    def bar_range(self) -> float:
        """Calculate bar range (high - low)."""
        return self.high - self.low
    
    @property
    def body(self) -> float:
        """Calculate bar body (close - open)."""
        return self.close - self.open
    
    @property
    def is_bullish(self) -> bool:
        """Check if bar is bullish."""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """Check if bar is bearish."""
        return self.close < self.open
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
        }


# Alias for backward compatibility
Bar = OHLCV


@dataclass(frozen=True, slots=True)
class Signal:
    """
    Trading signal generated by a strategy.
    
    Attributes:
        id: Unique signal identifier
        timestamp: Signal generation time
        symbol: Trading symbol
        signal_type: Type of signal (entry/exit)
        direction: Signal direction (1=long, -1=short, 0=neutral)
        strength: Signal confidence level
        price: Current price at signal
        strategy_name: Name of generating strategy
        metadata: Additional signal metadata
    """
    id: UUID
    timestamp: datetime
    symbol: str
    signal_type: str
    direction: int  # 1=long, -1=short, 0=neutral
    strength: SignalStrength
    price: float
    strategy_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate signal."""
        if self.direction not in (-1, 0, 1):
            raise DataValidationError(
                f"Direction must be -1, 0, or 1, got {self.direction}"
            )
    
    @classmethod
    def create(
        cls,
        timestamp: datetime,
        symbol: str,
        signal_type: str,
        direction: int,
        strength: SignalStrength,
        price: float,
        strategy_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Signal:
        """Factory method to create a new signal."""
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


@dataclass(slots=True)
class Order:
    """
    Trading order representation.
    
    Mutable to track order lifecycle.
    
    Attributes:
        id: Unique order identifier
        timestamp: Order creation time
        symbol: Trading symbol
        side: Order side (buy/sell)
        order_type: Order type (market/limit/etc)
        quantity: Order quantity
        price: Limit price (if applicable)
        stop_price: Stop price (if applicable)
        status: Current order status
        filled_quantity: Quantity filled so far
        filled_price: Average fill price
        commission: Commission charged
        signal_id: Related signal ID
        metadata: Additional order metadata
    """
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
        signal_id: UUID | None = None,
    ) -> Order:
        """Factory method to create market order."""
        return cls(
            id=uuid4(),
            timestamp=datetime.now(),
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
        signal_id: UUID | None = None,
    ) -> Order:
        """Factory method to create limit order."""
        return cls(
            id=uuid4(),
            timestamp=datetime.now(),
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
        """Record an order fill."""
        if quantity <= 0:
            raise ExecutionError("Fill quantity must be positive")
        
        # Update average fill price
        total_value = (self.filled_price * self.filled_quantity) + (price * quantity)
        self.filled_quantity += quantity
        self.filled_price = total_value / self.filled_quantity
        self.commission += commission
        
        # Update status
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIAL
    
    def cancel(self) -> None:
        """Cancel the order."""
        if self.status == OrderStatus.FILLED:
            raise ExecutionError("Cannot cancel filled order")
        self.status = OrderStatus.CANCELLED
    
    def reject(self, reason: str = "") -> None:
        """Reject the order."""
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


@dataclass(slots=True)
class Trade:
    """
    Completed trade record.
    
    Attributes:
        id: Unique trade identifier
        entry_time: Trade entry timestamp
        exit_time: Trade exit timestamp (None if open)
        symbol: Trading symbol
        side: Trade side (long/short)
        entry_price: Entry price
        exit_price: Exit price (None if open)
        quantity: Trade quantity
        pnl: Realized profit/loss
        pnl_pct: PnL as percentage
        commission: Total commission paid
        entry_order_id: Entry order ID
        exit_order_id: Exit order ID
        metadata: Additional trade metadata
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
        entry_time: datetime | None = None,
        entry_order_id: UUID | None = None,
    ) -> Trade:
        """Factory method to create a new trade."""
        return cls(
            id=uuid4(),
            entry_time=entry_time or datetime.now(),
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            entry_order_id=entry_order_id,
        )
    
    def close(
        self,
        exit_price: float,
        exit_time: datetime | None = None,
        commission: float = 0.0,
        exit_order_id: UUID | None = None,
    ) -> None:
        """Close the trade."""
        self.exit_price = exit_price
        self.exit_time = exit_time or datetime.now()
        self.exit_order_id = exit_order_id
        self.commission += commission
        
        # Calculate PnL
        if self.side == "long":
            self.pnl = (exit_price - self.entry_price) * self.quantity - self.commission
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.quantity - self.commission
        
        self.pnl_pct = self.pnl / (self.entry_price * self.quantity)
    
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


@dataclass(slots=True)
class Position:
    """
    Current position in a symbol.
    
    Attributes:
        symbol: Trading symbol
        quantity: Position quantity (positive=long, negative=short)
        avg_price: Average entry price
        current_price: Current market price
        unrealized_pnl: Unrealized profit/loss
        realized_pnl: Realized profit/loss from partial closes
        opened_at: Position open timestamp
        updated_at: Last update timestamp
        trades: List of trade IDs in this position
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
    
    def update_price(self, price: float) -> None:
        """Update current price and unrealized PnL."""
        self.current_price = price
        self.updated_at = datetime.now()
        
        if self.quantity > 0:  # Long
            self.unrealized_pnl = (price - self.avg_price) * self.quantity
        elif self.quantity < 0:  # Short
            self.unrealized_pnl = (self.avg_price - price) * abs(self.quantity)
        else:
            self.unrealized_pnl = 0.0
    
    def add(self, quantity: float, price: float) -> None:
        """Add to position."""
        if self.quantity == 0:
            self.opened_at = datetime.now()
        
        # Calculate new average price
        total_cost = (self.avg_price * abs(self.quantity)) + (price * abs(quantity))
        self.quantity += quantity
        
        if self.quantity != 0:
            self.avg_price = total_cost / abs(self.quantity)
        
        self.update_price(price)
    
    def reduce(self, quantity: float, price: float) -> float:
        """Reduce position and return realized PnL."""
        if abs(quantity) > abs(self.quantity):
            raise ExecutionError("Cannot reduce by more than position size")
        
        # Calculate realized PnL for this portion
        if self.quantity > 0:  # Long
            realized = (price - self.avg_price) * quantity
        else:  # Short
            realized = (self.avg_price - price) * abs(quantity)
        
        self.realized_pnl += realized
        self.quantity -= quantity
        self.update_price(price)
        
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


@dataclass(slots=True)
class PortfolioState:
    """
    Current portfolio state snapshot.
    
    Attributes:
        timestamp: State timestamp
        cash: Available cash
        equity: Total equity value
        buying_power: Available buying power
        positions: Current positions by symbol
        open_orders: Active orders
        daily_pnl: Day's PnL
        total_pnl: Total PnL
        margin_used: Margin currently used
    """
    timestamp: datetime
    cash: float
    equity: float
    buying_power: float
    positions: dict[str, Position] = field(default_factory=dict)
    open_orders: list[Order] = field(default_factory=list)
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    margin_used: float = 0.0
    
    @property
    def position_value(self) -> float:
        """Calculate total position value."""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate total unrealized PnL."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def realized_pnl(self) -> float:
        """Calculate total realized PnL."""
        return sum(pos.realized_pnl for pos in self.positions.values())
    
    @property
    def num_positions(self) -> int:
        """Get number of open positions."""
        return len([p for p in self.positions.values() if p.is_open])
    
    @property
    def long_exposure(self) -> float:
        """Calculate total long exposure."""
        return sum(
            pos.market_value
            for pos in self.positions.values()
            if pos.quantity > 0
        )
    
    @property
    def short_exposure(self) -> float:
        """Calculate total short exposure."""
        return sum(
            pos.market_value
            for pos in self.positions.values()
            if pos.quantity < 0
        )
    
    @property
    def net_exposure(self) -> float:
        """Calculate net market exposure."""
        return self.long_exposure - self.short_exposure
    
    @property
    def gross_exposure(self) -> float:
        """Calculate gross market exposure."""
        return self.long_exposure + self.short_exposure
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cash": self.cash,
            "equity": self.equity,
            "buying_power": self.buying_power,
            "position_value": self.position_value,
            "num_positions": self.num_positions,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "daily_pnl": self.daily_pnl,
            "total_pnl": self.total_pnl,
            "long_exposure": self.long_exposure,
            "short_exposure": self.short_exposure,
            "net_exposure": self.net_exposure,
            "gross_exposure": self.gross_exposure,
            "margin_used": self.margin_used,
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
        }


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """
    Trading performance metrics.
    
    All standard institutional-grade metrics for strategy evaluation.
    """
    # Returns
    total_return: float
    annual_return: float
    monthly_returns: NDArray[np.float64]
    daily_returns: NDArray[np.float64]
    
    # Risk metrics
    volatility: float
    annual_volatility: float
    downside_volatility: float
    max_drawdown: float
    max_drawdown_duration: int  # in days
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade: float
    avg_holding_period: float  # in hours
    
    # Value at Risk
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    
    # Other
    skewness: float
    kurtosis: float
    tail_ratio: float
    
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
    
    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio."""
        return self.payoff_ratio
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excluding large arrays)."""
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "volatility": self.volatility,
            "annual_volatility": self.annual_volatility,
            "downside_volatility": self.downside_volatility,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "omega_ratio": self.omega_ratio,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "avg_trade": self.avg_trade,
            "expectancy": self.expectancy,
            "payoff_ratio": self.payoff_ratio,
            "avg_holding_period": self.avg_holding_period,
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "cvar_99": self.cvar_99,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "tail_ratio": self.tail_ratio,
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
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
    # Enums
    "OrderStatus",
    "PositionStatus",
    "FillType",
    "SignalStrength",
    # Data structures
    "OHLCV",
    "Bar",
    "Signal",
    "Order",
    "Trade",
    "Position",
    "PortfolioState",
    "PerformanceMetrics",
]