"""
Base Strategy Module
====================

Abstract base class and common utilities for all trading strategies.
Provides a standardized interface for strategy development.

Features:
- Abstract strategy interface
- Position tracking
- Signal generation utilities
- Risk management integration
- Performance tracking
- Parameter validation

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, TypeVar
from uuid import UUID, uuid4

import numpy as np
import polars as pl
from numpy.typing import NDArray

from config.settings import get_logger, TimeFrame, OrderSide, PositionSide
from core.events import (
    MarketEvent,
    SignalEvent,
    FillEvent,
    OrderEvent,
    EventPriority,
)
from core.types import (
    Signal,
    SignalStrength,
    Position,
    PortfolioState,
    StrategyError,
    StrategyInitializationError,
    SignalGenerationError,
)

logger = get_logger(__name__)

T = TypeVar("T", bound="BaseStrategy")


# =============================================================================
# ENUMS AND CONFIGURATION
# =============================================================================

class StrategyState(str, Enum):
    """Strategy lifecycle state."""
    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class SignalAction(str, Enum):
    """Signal action types."""
    ENTRY_LONG = "entry_long"
    ENTRY_SHORT = "entry_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"
    HOLD = "hold"
    FLATTEN = "flatten"


@dataclass
class StrategyConfig:
    """
    Base configuration for strategies.
    
    Attributes:
        name: Strategy name
        symbols: List of symbols to trade
        timeframe: Primary timeframe
        max_positions: Maximum number of concurrent positions
        position_size_pct: Default position size as % of portfolio
        use_stop_loss: Enable stop-loss orders
        stop_loss_pct: Default stop-loss percentage
        use_take_profit: Enable take-profit orders
        take_profit_pct: Default take-profit percentage
        max_holding_period: Maximum bars to hold a position
        min_signal_strength: Minimum signal strength to trade
        cooldown_bars: Bars to wait after a trade
        allow_pyramiding: Allow adding to existing positions
        max_pyramid_levels: Maximum pyramid levels
    """
    name: str = "BaseStrategy"
    symbols: list[str] = field(default_factory=list)
    timeframe: TimeFrame = TimeFrame.M15
    max_positions: int = 10
    position_size_pct: float = 0.10
    use_stop_loss: bool = True
    stop_loss_pct: float = 0.02
    use_take_profit: bool = True
    take_profit_pct: float = 0.04
    max_holding_period: int | None = None
    min_signal_strength: float = 0.5
    cooldown_bars: int = 0
    allow_pyramiding: bool = False
    max_pyramid_levels: int = 3
    
    def validate(self) -> list[str]:
        """Validate configuration, return list of errors."""
        errors = []
        
        if not self.name:
            errors.append("Strategy name is required")
        if self.max_positions < 1:
            errors.append("max_positions must be >= 1")
        if not 0 < self.position_size_pct <= 1:
            errors.append("position_size_pct must be between 0 and 1")
        if self.stop_loss_pct < 0:
            errors.append("stop_loss_pct must be >= 0")
        if self.take_profit_pct < 0:
            errors.append("take_profit_pct must be >= 0")
        if self.min_signal_strength < 0 or self.min_signal_strength > 1:
            errors.append("min_signal_strength must be between 0 and 1")
        
        return errors


@dataclass
class StrategyMetrics:
    """
    Real-time strategy performance metrics.
    
    Attributes:
        total_signals: Total signals generated
        entry_signals: Entry signals count
        exit_signals: Exit signals count
        trades_opened: Number of trades opened
        trades_closed: Number of trades closed
        winning_trades: Number of winning trades
        losing_trades: Number of losing trades
        total_pnl: Total profit/loss
        max_drawdown: Maximum drawdown experienced
        current_drawdown: Current drawdown
        avg_holding_bars: Average holding period in bars
        signal_accuracy: Percentage of profitable signals
    """
    total_signals: int = 0
    entry_signals: int = 0
    exit_signals: int = 0
    trades_opened: int = 0
    trades_closed: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    avg_holding_bars: float = 0.0
    signal_accuracy: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        total = self.winning_trades + self.losing_trades
        return self.winning_trades / total if total > 0 else 0.0
    
    def update(
        self,
        signal_generated: bool = False,
        is_entry: bool = False,
        trade_closed: bool = False,
        pnl: float = 0.0,
    ) -> None:
        """Update metrics."""
        if signal_generated:
            self.total_signals += 1
            if is_entry:
                self.entry_signals += 1
            else:
                self.exit_signals += 1
        
        if trade_closed:
            self.trades_closed += 1
            self.total_pnl += pnl
            if pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            self.signal_accuracy = self.win_rate
        
        self.last_updated = datetime.now()


# =============================================================================
# BASE STRATEGY CLASS
# =============================================================================

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Provides common functionality for:
    - Strategy lifecycle management
    - Signal generation
    - Position tracking
    - Risk management integration
    - Performance monitoring
    
    Subclasses must implement:
    - calculate_signals(): Core signal generation logic
    
    Lifecycle:
        1. __init__: Create strategy with parameters
        2. initialize(): Setup before trading (indicators, state)
        3. on_bar(): Called for each new bar (generates signals)
        4. on_fill(): Called when orders are filled
        5. shutdown(): Cleanup when strategy stops
    
    Example:
        class MyStrategy(BaseStrategy):
            def calculate_signals(self, data, portfolio):
                # Your signal logic here
                if some_condition:
                    return self.create_entry_signal("AAPL", 1, 0.8, price)
                return []
    """
    
    def __init__(
        self,
        config: StrategyConfig | None = None,
        parameters: dict[str, Any] | None = None,
    ):
        """
        Initialize strategy.
        
        Args:
            config: Strategy configuration
            parameters: Strategy-specific parameters
        """
        self.config = config or StrategyConfig()
        self.parameters = parameters or {}
        
        # Validate configuration
        errors = self.config.validate()
        if errors:
            raise StrategyInitializationError(f"Invalid config: {errors}")
        
        # State
        self._state = StrategyState.CREATED
        self._is_initialized = False
        
        # Trading state
        self._symbols: list[str] = []
        self._current_positions: dict[str, Position] = {}
        self._pending_orders: dict[UUID, dict[str, Any]] = {}
        self._cooldown_tracker: dict[str, int] = {}
        self._bar_count: int = 0
        self._entry_bars: dict[str, int] = {}
        
        # Data storage
        self._data_cache: dict[str, pl.DataFrame] = {}
        self._indicator_cache: dict[str, Any] = {}
        
        # Metrics
        self.metrics = StrategyMetrics()
        
        # Callbacks
        self._signal_callbacks: list[Callable[[SignalEvent], None]] = []
        
        logger.info(f"Strategy '{self.config.name}' created")
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def name(self) -> str:
        """Get strategy name."""
        return self.config.name
    
    @property
    def state(self) -> StrategyState:
        """Get current strategy state."""
        return self._state
    
    @property
    def is_initialized(self) -> bool:
        """Check if strategy is initialized."""
        return self._is_initialized
    
    @property
    def is_running(self) -> bool:
        """Check if strategy is running."""
        return self._state == StrategyState.RUNNING
    
    @property
    def symbols(self) -> list[str]:
        """Get list of trading symbols."""
        return self._symbols.copy()
    
    @property
    def bar_count(self) -> int:
        """Get current bar count."""
        return self._bar_count
    
    @property
    def current_positions(self) -> dict[str, Position]:
        """Get current positions."""
        return self._current_positions.copy()
    
    @property
    def position_count(self) -> int:
        """Get number of open positions."""
        return len([p for p in self._current_positions.values() if p.is_open])
    
    # =========================================================================
    # LIFECYCLE METHODS
    # =========================================================================
    
    def initialize(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        """
        Initialize strategy before trading.
        
        Override this method to setup indicators, load models, etc.
        
        Args:
            symbols: List of symbols to trade
            start_date: Trading start date
            end_date: Trading end date
        """
        logger.info(f"Initializing {self.name} for {len(symbols)} symbols")
        
        self._symbols = symbols if symbols else self.config.symbols
        self._state = StrategyState.INITIALIZED
        self._is_initialized = True
        
        # Initialize cooldown tracker
        for symbol in self._symbols:
            self._cooldown_tracker[symbol] = 0
        
        # Call subclass initialization
        self._on_initialize(symbols, start_date, end_date)
        
        logger.info(f"{self.name} initialized successfully")
    
    def _on_initialize(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        """
        Override for custom initialization logic.
        
        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
        """
        pass
    
    def start(self) -> None:
        """Start the strategy."""
        if not self._is_initialized:
            raise StrategyError("Strategy must be initialized before starting")
        
        self._state = StrategyState.RUNNING
        logger.info(f"{self.name} started")
    
    def pause(self) -> None:
        """Pause the strategy."""
        self._state = StrategyState.PAUSED
        logger.info(f"{self.name} paused")
    
    def resume(self) -> None:
        """Resume the strategy."""
        if self._state == StrategyState.PAUSED:
            self._state = StrategyState.RUNNING
            logger.info(f"{self.name} resumed")
    
    def shutdown(self) -> None:
        """
        Shutdown the strategy.
        
        Override to cleanup resources.
        """
        logger.info(f"Shutting down {self.name}")
        
        self._state = StrategyState.STOPPED
        self._is_initialized = False
        
        # Clear caches
        self._data_cache.clear()
        self._indicator_cache.clear()
        
        # Call subclass shutdown
        self._on_shutdown()
        
        logger.info(f"{self.name} shutdown complete")
    
    def _on_shutdown(self) -> None:
        """Override for custom shutdown logic."""
        pass
    
    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================
    
    def on_bar(
        self,
        event: MarketEvent,
        portfolio: PortfolioState,
    ) -> list[SignalEvent]:
        """
        Process a new bar and generate signals.
        
        This is the main entry point called by the backtester/live engine.
        
        Args:
            event: Market event with bar data
            portfolio: Current portfolio state
        
        Returns:
            List of signal events
        """
        if self._state != StrategyState.RUNNING:
            return []
        
        self._bar_count += 1
        signals: list[SignalEvent] = []
        
        try:
            # Update cooldown trackers
            self._update_cooldowns()
            
            # Update data cache
            symbol = event.symbol
            if event.data is not None:
                self._data_cache[symbol] = event.data
            
            # Check for max holding period exits
            if self.config.max_holding_period:
                exit_signals = self._check_holding_period_exits(symbol, portfolio)
                signals.extend(exit_signals)
            
            # Generate new signals
            new_signals = self.calculate_signals(event, portfolio)
            
            # Filter and validate signals
            valid_signals = self._filter_signals(new_signals, portfolio)
            signals.extend(valid_signals)
            
            # Update metrics
            for signal in signals:
                self.metrics.update(
                    signal_generated=True,
                    is_entry=signal.is_entry,
                )
            
            # Notify callbacks
            for callback in self._signal_callbacks:
                for signal in signals:
                    callback(signal)
            
        except Exception as e:
            logger.error(f"Error in {self.name}.on_bar: {e}")
            self._state = StrategyState.ERROR
            raise SignalGenerationError(f"Signal generation failed: {e}")
        
        return signals
    
    @abstractmethod
    def calculate_signals(
        self,
        event: MarketEvent,
        portfolio: PortfolioState,
    ) -> list[SignalEvent]:
        """
        Calculate trading signals.
        
        This is the core method that subclasses must implement.
        
        Args:
            event: Market event with bar data
            portfolio: Current portfolio state
        
        Returns:
            List of signal events
        """
        pass
    
    def on_fill(self, event: FillEvent) -> None:
        """
        Called when an order is filled.
    
        Updates position tracking and metrics.
    
        Properly handles all combinations of:
        - Buy on long/flat position (add to long)
        - Buy on short position (close/reduce short)
        - Sell on long position (close/reduce long)
        - Sell on short/flat position (add to short)
    
        Args:
            event: Fill event
        """
        symbol = event.symbol
    
        # Update position tracking
        if symbol not in self._current_positions:
            self._current_positions[symbol] = Position(symbol=symbol)
    
        position = self._current_positions[symbol]
        current_qty = position.quantity  # KEY: Check current state first!
    
        if event.side == "buy":
            if current_qty < 0:
                # Closing/reducing a SHORT position
                reduce_qty = min(event.quantity, abs(current_qty))
                pnl = position.reduce(reduce_qty, event.price)
            
                if not position.is_open:
                    self._cooldown_tracker[symbol] = self.config.cooldown_bars
                    if symbol in self._entry_bars:
                        del self._entry_bars[symbol]
                    self.metrics.update(trade_closed=True, pnl=pnl)
            else:
                # Opening/adding to a LONG position
                position.add(event.quantity, event.price)
                if symbol not in self._entry_bars:
                    self._entry_bars[symbol] = self._bar_count
                self.metrics.trades_opened += 1
    
        else:  # event.side == "sell"
            if current_qty > 0:
                # Closing/reducing a LONG position
                reduce_qty = min(event.quantity, current_qty)
                pnl = position.reduce(reduce_qty, event.price)
            
                if not position.is_open:
                    self._cooldown_tracker[symbol] = self.config.cooldown_bars
                    if symbol in self._entry_bars:
                        del self._entry_bars[symbol]
                    self.metrics.update(trade_closed=True, pnl=pnl)
            else:
                # Opening/adding to a SHORT position
                position.add(-event.quantity, event.price)
                if symbol not in self._entry_bars:
                    self._entry_bars[symbol] = self._bar_count
                self.metrics.trades_opened += 1
    
        # Call subclass handler
        self._on_fill(event)
    
    def _on_fill(self, event: FillEvent) -> None:
            """Override for custom fill handling."""
            pass
    
    def on_order(self, event: OrderEvent) -> None:
        """
        Called when an order is created.
        
        Args:
            event: Order event
        """
        self._pending_orders[event.order_id] = {
            "symbol": event.symbol,
            "side": event.side,
            "quantity": event.quantity,
            "timestamp": event.timestamp,
        }
        
        # Call subclass handler
        self._on_order(event)
    
    def _on_order(self, event: OrderEvent) -> None:
        """Override for custom order handling."""
        pass
    
    # =========================================================================
    # SIGNAL GENERATION HELPERS
    # =========================================================================
    
    def create_entry_signal(
        self,
        symbol: str,
        direction: int,
        strength: float,
        price: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SignalEvent:
        """
        Create an entry signal.
        
        Args:
            symbol: Trading symbol
            direction: Signal direction (1=long, -1=short)
            strength: Signal strength (0.0 to 1.0)
            price: Current price
            stop_loss: Stop-loss price (optional)
            take_profit: Take-profit price (optional)
            metadata: Additional metadata
        
        Returns:
            SignalEvent for entry
        """
        # Calculate default stop-loss and take-profit
        if stop_loss is None and self.config.use_stop_loss:
            if direction == 1:  # Long
                stop_loss = price * (1 - self.config.stop_loss_pct)
            else:  # Short
                stop_loss = price * (1 + self.config.stop_loss_pct)
        
        if take_profit is None and self.config.use_take_profit:
            if direction == 1:  # Long
                take_profit = price * (1 + self.config.take_profit_pct)
            else:  # Short
                take_profit = price * (1 - self.config.take_profit_pct)
        
        signal_type = "entry_long" if direction == 1 else "entry_short"
        
        return SignalEvent(
            symbol=symbol,
            signal_type=signal_type,
            direction=direction,
            strength=strength,
            price=price,
            strategy_name=self.name,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata or {},
            priority=EventPriority.HIGH,
        )
    
    def create_exit_signal(
        self,
        symbol: str,
        direction: int,
        strength: float,
        price: float,
        reason: str = "strategy",
        metadata: dict[str, Any] | None = None,
    ) -> SignalEvent:
        """
        Create an exit signal.
        
        Args:
            symbol: Trading symbol
            direction: Original position direction (1=long, -1=short)
            strength: Signal strength
            price: Current price
            reason: Exit reason
            metadata: Additional metadata
        
        Returns:
            SignalEvent for exit
        """
        signal_type = "exit_long" if direction == 1 else "exit_short"
        
        meta = metadata or {}
        meta["exit_reason"] = reason
        
        return SignalEvent(
            symbol=symbol,
            signal_type=signal_type,
            direction=-direction,  # Opposite direction to close
            strength=strength,
            price=price,
            strategy_name=self.name,
            metadata=meta,
            priority=EventPriority.HIGH,
        )
    
    def create_scale_signal(
        self,
        symbol: str,
        direction: int,
        strength: float,
        price: float,
        scale_pct: float = 0.5,
        is_scale_in: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> SignalEvent:
        """
        Create a scale-in or scale-out signal.
        
        Args:
            symbol: Trading symbol
            direction: Signal direction
            strength: Signal strength
            price: Current price
            scale_pct: Percentage to scale (0.0 to 1.0)
            is_scale_in: True for scale-in, False for scale-out
            metadata: Additional metadata
        
        Returns:
            SignalEvent for scaling
        """
        signal_type = "scale_in" if is_scale_in else "scale_out"
        
        meta = metadata or {}
        meta["scale_pct"] = scale_pct
        
        return SignalEvent(
            symbol=symbol,
            signal_type=signal_type,
            direction=direction if is_scale_in else -direction,
            strength=strength,
            price=price,
            strategy_name=self.name,
            metadata=meta,
        )
    
    # =========================================================================
    # POSITION AND DATA HELPERS
    # =========================================================================
    
    def get_position(self, symbol: str) -> Position | None:
        """Get current position for a symbol."""
        return self._current_positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """Check if there's an open position for symbol."""
        pos = self._current_positions.get(symbol)
        return pos is not None and pos.is_open
    
    def get_position_side(self, symbol: str) -> PositionSide:
        """Get current position side for symbol."""
        pos = self._current_positions.get(symbol)
        if pos is None or not pos.is_open:
            return PositionSide.FLAT
        return PositionSide.LONG if pos.quantity > 0 else PositionSide.SHORT
    
    def get_data(self, symbol: str) -> pl.DataFrame | None:
        """Get cached data for symbol."""
        return self._data_cache.get(symbol)
    
    def get_latest_price(self, symbol: str) -> float | None:
        """Get latest close price for symbol."""
        data = self._data_cache.get(symbol)
        if data is None or len(data) == 0:
            return None
        return data["close"].item(-1)
    
    def get_indicator(self, name: str) -> Any:
        """Get cached indicator value."""
        return self._indicator_cache.get(name)
    
    def set_indicator(self, name: str, value: Any) -> None:
        """Cache indicator value."""
        self._indicator_cache[name] = value
    
    # =========================================================================
    # VALIDATION AND FILTERING
    # =========================================================================
    
    def _filter_signals(
        self,
        signals: list[SignalEvent],
        portfolio: PortfolioState,
    ) -> list[SignalEvent]:
        """Filter signals based on strategy rules."""
        filtered = []
        
        for signal in signals:
            # Check signal strength
            if signal.strength < self.config.min_signal_strength:
                logger.debug(f"Signal filtered: strength {signal.strength} < {self.config.min_signal_strength}")
                continue
            
            # Check cooldown
            if self._cooldown_tracker.get(signal.symbol, 0) > 0:
                logger.debug(f"Signal filtered: {signal.symbol} in cooldown")
                continue
            
            # Check max positions for entries
            if signal.is_entry:
                if self.position_count >= self.config.max_positions:
                    logger.debug(f"Signal filtered: max positions reached")
                    continue
                
                # Check pyramiding rules
                if self.has_position(signal.symbol):
                    if not self.config.allow_pyramiding:
                        logger.debug(f"Signal filtered: pyramiding not allowed")
                        continue
            
            filtered.append(signal)
        
        return filtered
    
    def _update_cooldowns(self) -> None:
        """Decrement cooldown counters."""
        for symbol in self._cooldown_tracker:
            if self._cooldown_tracker[symbol] > 0:
                self._cooldown_tracker[symbol] -= 1

    def _check_holding_period_exits(
        self,
        symbol: str,
        portfolio: PortfolioState,
    ) -> list[SignalEvent]:
        """Check for max holding period exits."""
        signals = []
        
        if symbol in self._entry_bars:
            bars_held = self._bar_count - self._entry_bars[symbol]
            
            if bars_held >= self.config.max_holding_period:
                position = self._current_positions.get(symbol)
                if position and position.is_open:
                    price = self.get_latest_price(symbol)
                    if price:
                        direction = 1 if position.quantity > 0 else -1
                        signal = self.create_exit_signal(
                            symbol=symbol,
                            direction=direction,
                            strength=1.0,
                            price=price,
                            reason="max_holding_period",
                        )
                        signals.append(signal)
                        logger.info(f"Max holding period exit for {symbol}")
        
        return signals
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def add_signal_callback(
        self,
        callback: Callable[[SignalEvent], None],
    ) -> None:
        """Add a callback for signal generation."""
        self._signal_callbacks.append(callback)
    
    def get_parameters(self) -> dict[str, Any]:
        """Get strategy parameters."""
        return self.parameters.copy()
    
    def set_parameters(self, parameters: dict[str, Any]) -> None:
        """Update strategy parameters."""
        self.parameters.update(parameters)
        logger.info(f"{self.name} parameters updated")
    
    def get_metrics(self) -> StrategyMetrics:
        """Get strategy metrics."""
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset strategy metrics."""
        self.metrics = StrategyMetrics()
    
    def get_state_dict(self) -> dict[str, Any]:
        """Get strategy state for serialization."""
        return {
            "name": self.name,
            "state": self._state.value,
            "bar_count": self._bar_count,
            "parameters": self.parameters,
            "metrics": {
                "total_signals": self.metrics.total_signals,
                "trades_opened": self.metrics.trades_opened,
                "trades_closed": self.metrics.trades_closed,
                "win_rate": self.metrics.win_rate,
                "total_pnl": self.metrics.total_pnl,
            },
            "positions": {
                k: v.to_dict() for k, v in self._current_positions.items()
            },
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, state={self._state.value})"


# =============================================================================
# STRATEGY COMBINER
# =============================================================================

class StrategyCombiner:
    """
    Combines signals from multiple strategies.
    
    Methods:
    - Voting: Take action if majority agree
    - Weighted: Weight signals by strategy performance
    - Unanimous: Only act if all strategies agree
    - Any: Act on any signal (union)
    
    Example:
        combiner = StrategyCombiner(
            strategies=[strat1, strat2, strat3],
            method="voting",
        )
        combined_signals = combiner.combine(signals_list)
    """
    
    def __init__(
        self,
        strategies: list[BaseStrategy],
        method: str = "voting",
        weights: dict[str, float] | None = None,
        min_agreement: float = 0.5,
    ):
        """
        Initialize combiner.
        
        Args:
            strategies: List of strategies
            method: Combination method
            weights: Strategy weights (for weighted method)
            min_agreement: Minimum agreement threshold
        """
        self.strategies = strategies
        self.method = method
        self.weights = weights or {s.name: 1.0 for s in strategies}
        self.min_agreement = min_agreement
    
    def combine(
        self,
        signals_by_strategy: dict[str, list[SignalEvent]],
    ) -> list[SignalEvent]:
        """
        Combine signals from multiple strategies.
        
        Args:
            signals_by_strategy: Dict mapping strategy name to signals
        
        Returns:
            Combined list of signals
        """
        if self.method == "voting":
            return self._combine_voting(signals_by_strategy)
        elif self.method == "weighted":
            return self._combine_weighted(signals_by_strategy)
        elif self.method == "unanimous":
            return self._combine_unanimous(signals_by_strategy)
        elif self.method == "any":
            return self._combine_any(signals_by_strategy)
        else:
            raise ValueError(f"Unknown combination method: {self.method}")
    
    def _combine_voting(
        self,
        signals_by_strategy: dict[str, list[SignalEvent]],
    ) -> list[SignalEvent]:
        """Combine by majority voting."""
        # Group signals by symbol and direction
        votes: dict[tuple[str, int], list[SignalEvent]] = {}
        
        for strategy_name, signals in signals_by_strategy.items():
            for signal in signals:
                key = (signal.symbol, signal.direction)
                if key not in votes:
                    votes[key] = []
                votes[key].append(signal)
        
        # Keep signals with majority agreement
        combined = []
        n_strategies = len(self.strategies)
        min_votes = int(n_strategies * self.min_agreement)
        
        for (symbol, direction), signal_list in votes.items():
            if len(signal_list) >= min_votes:
                # Use signal with highest strength
                best_signal = max(signal_list, key=lambda s: s.strength)
                combined.append(best_signal)
        
        return combined
    
    def _combine_weighted(
        self,
        signals_by_strategy: dict[str, list[SignalEvent]],
    ) -> list[SignalEvent]:
        """Combine using weighted voting."""
        # Group signals by symbol
        symbol_signals: dict[str, list[tuple[SignalEvent, float]]] = {}
        
        for strategy_name, signals in signals_by_strategy.items():
            weight = self.weights.get(strategy_name, 1.0)
            for signal in signals:
                if signal.symbol not in symbol_signals:
                    symbol_signals[signal.symbol] = []
                symbol_signals[signal.symbol].append((signal, weight))
        
        # Calculate weighted direction and strength
        combined = []
        total_weight = sum(self.weights.values())
        
        for symbol, weighted_signals in symbol_signals.items():
            weighted_direction = sum(
                s.direction * w for s, w in weighted_signals
            ) / total_weight
            
            weighted_strength = sum(
                s.strength * w for s, w in weighted_signals
            ) / total_weight
            
            # Use the best signal as template
            best_signal = max(weighted_signals, key=lambda x: x[1])[0]
            
            if abs(weighted_direction) >= self.min_agreement:
                direction = 1 if weighted_direction > 0 else -1
                combined.append(SignalEvent(
                    symbol=symbol,
                    signal_type=best_signal.signal_type,
                    direction=direction,
                    strength=weighted_strength,
                    price=best_signal.price,
                    strategy_name="combined",
                    metadata={"source_strategies": list(signals_by_strategy.keys())},
                ))
        
        return combined
    
    def _combine_unanimous(
        self,
        signals_by_strategy: dict[str, list[SignalEvent]],
    ) -> list[SignalEvent]:
        """Only return signals where all strategies agree."""
        self.min_agreement = 1.0
        return self._combine_voting(signals_by_strategy)
    
    def _combine_any(
        self,
        signals_by_strategy: dict[str, list[SignalEvent]],
    ) -> list[SignalEvent]:
        """Return all signals (union)."""
        combined = []
        seen = set()
        
        for signals in signals_by_strategy.values():
            for signal in signals:
                key = (signal.symbol, signal.direction, signal.signal_type)
                if key not in seen:
                    combined.append(signal)
                    seen.add(key)
        
        return combined


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "StrategyState",
    "SignalAction",
    # Configuration
    "StrategyConfig",
    "StrategyMetrics",
    # Base class
    "BaseStrategy",
    # Combiner
    "StrategyCombiner",
]