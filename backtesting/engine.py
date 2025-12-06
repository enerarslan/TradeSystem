"""
Backtesting Engine Module
=========================

Event-driven backtesting engine for strategy evaluation.
Provides realistic simulation of trading strategies with proper
execution modeling, position tracking, and performance analysis.

Architecture:
- Event-driven design with event queue
- Support for multiple strategies and symbols
- Walk-forward validation
- Real-time metrics calculation
- Comprehensive reporting

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator
from uuid import UUID, uuid4
import json

import numpy as np
import polars as pl
from numpy.typing import NDArray

from config.settings import get_logger, get_settings, TradingMode
from core.types import (
    Order, OrderStatus, Trade, Position, PortfolioState,
    Signal, SignalStrength, OHLCV, PerformanceMetrics,
    ExecutionError, BacktestError,
)
from core.events import (
    Event, EventType, EventBus,
    MarketEvent, SignalEvent, OrderEvent, FillEvent, PortfolioEvent,
)
from core.interfaces import Strategy, DataProvider
from backtesting.execution import (
    ExecutionSimulator, FillResult,
    SlippageModel, CommissionModel, FillModel,
    PercentageSlippage, PerShareCommission, OHLCFill,
    create_realistic_simulator, create_zero_cost_simulator,
)
from backtesting.metrics import (
    MetricsCalculator, PerformanceReport, TradeStats,
    calculate_trade_stats, max_drawdown, sharpe_ratio,
)

logger = get_logger(__name__)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class BacktestMode(str, Enum):
    """Backtesting modes."""
    VECTORIZED = "vectorized"   # Fast, bulk processing
    EVENT_DRIVEN = "event_driven"  # Realistic, bar-by-bar
    TICK = "tick"  # Tick-level simulation


class OrderFillMode(str, Enum):
    """Order fill timing."""
    CURRENT_BAR = "current_bar"  # Fill at current bar close
    NEXT_BAR_OPEN = "next_bar_open"  # Fill at next bar open
    NEXT_BAR_CLOSE = "next_bar_close"  # Fill at next bar close


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BacktestConfig:
    """
    Backtesting configuration.
    
    Attributes:
        initial_capital: Starting capital
        commission_pct: Commission percentage
        slippage_pct: Slippage percentage
        margin_requirement: Margin requirement for positions
        allow_shorting: Allow short positions
        fractional_shares: Allow fractional shares
        fill_mode: Order fill timing
        max_positions: Maximum concurrent positions
        position_sizing: Default position sizing method
        risk_per_trade: Risk per trade (for position sizing)
        warmup_bars: Bars for indicator warmup
        benchmark_symbol: Benchmark for comparison
    """
    # Capital
    initial_capital: float = 100_000.0
    
    # Costs
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    
    # Trading rules
    margin_requirement: float = 1.0  # 1.0 = no margin
    allow_shorting: bool = True
    fractional_shares: bool = True
    
    # Execution
    fill_mode: OrderFillMode = OrderFillMode.NEXT_BAR_OPEN
    
    # Position management
    max_positions: int = 10
    position_sizing: str = "fixed"  # fixed, percent, kelly, volatility
    risk_per_trade: float = 0.02  # 2%
    
    # Other
    warmup_bars: int = 50
    benchmark_symbol: str | None = None
    
    # Data
    timeframe: str = "1day"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "initial_capital": self.initial_capital,
            "commission_pct": self.commission_pct,
            "slippage_pct": self.slippage_pct,
            "margin_requirement": self.margin_requirement,
            "allow_shorting": self.allow_shorting,
            "fractional_shares": self.fractional_shares,
            "fill_mode": self.fill_mode.value,
            "max_positions": self.max_positions,
            "position_sizing": self.position_sizing,
            "risk_per_trade": self.risk_per_trade,
            "warmup_bars": self.warmup_bars,
            "benchmark_symbol": self.benchmark_symbol,
            "timeframe": self.timeframe,
        }


# =============================================================================
# PORTFOLIO TRACKER
# =============================================================================

class PortfolioTracker:
    """
    Tracks portfolio state during backtesting.
    
    Manages positions, cash, and equity calculations.
    """
    
    def __init__(
        self,
        initial_capital: float,
        allow_shorting: bool = True,
        margin_requirement: float = 1.0,
    ):
        """
        Initialize portfolio tracker.
        
        Args:
            initial_capital: Starting capital
            allow_shorting: Allow short positions
            margin_requirement: Margin requirement
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.allow_shorting = allow_shorting
        self.margin_requirement = margin_requirement
        
        # Positions
        self.positions: dict[str, Position] = {}
        
        # History
        self.trades: list[Trade] = []
        self.equity_history: list[tuple[datetime, float]] = []
        self.cash_history: list[tuple[datetime, float]] = []
        
        # Open orders (pending)
        self.pending_orders: list[Order] = []
        
        # Current prices
        self._current_prices: dict[str, float] = {}
    
    @property
    def equity(self) -> float:
        """Calculate current equity."""
        position_value = sum(
            pos.market_value for pos in self.positions.values()
        )
        return self.cash + position_value
    
    @property
    def buying_power(self) -> float:
        """Calculate available buying power."""
        if self.margin_requirement <= 0:
            return float('inf')
        return self.cash / self.margin_requirement
    
    @property
    def num_positions(self) -> int:
        """Get number of open positions."""
        return len([p for p in self.positions.values() if p.is_open])
    
    def update_prices(self, prices: dict[str, float]) -> None:
        """
        Update current prices for all positions.
        
        Args:
            prices: Symbol to price mapping
        """
        self._current_prices.update(prices)
        
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.update_price(prices[symbol])
    
    def get_position(self, symbol: str) -> Position | None:
        """Get position for symbol."""
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """Check if position exists for symbol."""
        pos = self.positions.get(symbol)
        return pos is not None and pos.is_open
    
    def process_fill(
        self,
        fill: FillResult,
        timestamp: datetime,
    ) -> Trade | None:
        """
        Process an order fill.
        
        Args:
            fill: Fill result from execution
            timestamp: Fill timestamp
            
        Returns:
            Trade object if position was closed
        """
        if not fill.filled:
            return None
        
        symbol = fill.metadata.get("symbol", "UNKNOWN")
        side = fill.metadata.get("side", "buy")
        
        # Get or create position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        
        pos = self.positions[symbol]
        
        # Update position
        if side.lower() == "buy":
            # Buying
            cost = fill.fill_quantity * fill.fill_price + fill.commission
            
            if pos.quantity < 0:
                # Closing short position
                trade = self._close_position_partial(
                    pos, fill.fill_quantity, fill.fill_price,
                    fill.commission, timestamp
                )
                self.cash += fill.fill_quantity * fill.fill_price - fill.commission
                return trade
            else:
                # Opening/adding to long
                pos.add(fill.fill_quantity, fill.fill_price)
                self.cash -= cost
        else:
            # Selling
            if pos.quantity > 0:
                # Closing long position
                trade = self._close_position_partial(
                    pos, fill.fill_quantity, fill.fill_price,
                    fill.commission, timestamp
                )
                self.cash += fill.fill_quantity * fill.fill_price - fill.commission
                return trade
            else:
                # Opening/adding to short
                if not self.allow_shorting:
                    logger.warning(f"Short selling not allowed for {symbol}")
                    return None
                pos.add(-fill.fill_quantity, fill.fill_price)
                self.cash += fill.fill_quantity * fill.fill_price - fill.commission
        
        return None
    
    def _close_position_partial(
        self,
        pos: Position,
        quantity: float,
        price: float,
        commission: float,
        timestamp: datetime,
    ) -> Trade | None:
        """Close or reduce a position."""
        if quantity >= abs(pos.quantity):
            # Full close
            trade = Trade.create(
                symbol=pos.symbol,
                side="long" if pos.quantity > 0 else "short",
                entry_price=pos.avg_price,
                quantity=abs(pos.quantity),
                entry_time=pos.opened_at,
            )
            trade.close(price, timestamp, commission)
            
            # Reset position
            pos.quantity = 0
            pos.avg_price = 0
            pos.realized_pnl += trade.pnl
            
            self.trades.append(trade)
            return trade
        else:
            # Partial close
            realized = pos.reduce(quantity, price)
            pos.realized_pnl += realized
            
            # Create trade record
            trade = Trade.create(
                symbol=pos.symbol,
                side="long" if pos.quantity > 0 else "short",
                entry_price=pos.avg_price,
                quantity=quantity,
                entry_time=pos.opened_at,
            )
            trade.pnl = realized - commission
            trade.exit_price = price
            trade.exit_time = timestamp
            
            self.trades.append(trade)
            return trade
    
    def record_equity(self, timestamp: datetime) -> None:
        """Record current equity point."""
        self.equity_history.append((timestamp, self.equity))
        self.cash_history.append((timestamp, self.cash))
    
    def get_state(self, timestamp: datetime) -> PortfolioState:
        """Get current portfolio state."""
        return PortfolioState(
            timestamp=timestamp,
            cash=self.cash,
            equity=self.equity,
            buying_power=self.buying_power,
            positions=self.positions.copy(),
            open_orders=self.pending_orders.copy(),
            total_pnl=self.equity - self.initial_capital,
        )
    
    def get_equity_curve(self) -> NDArray[np.float64]:
        """Get equity curve as numpy array."""
        return np.array([e for _, e in self.equity_history])
    
    def get_timestamps(self) -> list[datetime]:
        """Get timestamps for equity curve."""
        return [t for t, _ in self.equity_history]
    
    def get_trade_list(self) -> list[dict[str, Any]]:
        """Get list of trades as dictionaries."""
        return [t.to_dict() for t in self.trades]
    
    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.equity_history.clear()
        self.cash_history.clear()
        self.pending_orders.clear()
        self._current_prices.clear()


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class BacktestEngine:
    """
    Event-driven backtesting engine.
    
    Simulates strategy execution with realistic market conditions,
    order execution, and position tracking.
    
    Example:
        engine = BacktestEngine(config)
        engine.add_data(data_provider)
        engine.add_strategy(my_strategy)
        
        results = engine.run(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 12, 31),
        )
        
        print(results.sharpe_ratio)
    """
    
    def __init__(self, config: BacktestConfig | None = None):
        """
        Initialize backtest engine.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config or BacktestConfig()
        
        # Components
        self.portfolio = PortfolioTracker(
            initial_capital=self.config.initial_capital,
            allow_shorting=self.config.allow_shorting,
            margin_requirement=self.config.margin_requirement,
        )
        
        # Execution simulator
        self.execution = ExecutionSimulator(
            slippage_model=PercentageSlippage(self.config.slippage_pct),
            commission_model=PerShareCommission(
                per_share=0.0,  # Use percentage
                min_commission=0.0,
            ),
        )
        
        # Event bus
        self.event_bus = EventBus()
        
        # Data and strategies
        self._data: dict[str, pl.DataFrame] = {}
        self._strategies: list[Strategy] = []
        self._benchmark_data: pl.DataFrame | None = None
        
        # State
        self._current_bar_idx: int = 0
        self._current_timestamp: datetime | None = None
        self._is_running: bool = False
        
        # Results
        self._signals: list[SignalEvent] = []
        self._orders: list[Order] = []
        self._fills: list[FillResult] = []
        
        # Register event handlers
        self._setup_event_handlers()
    
    def _setup_event_handlers(self) -> None:
        """Setup event bus handlers."""
        self.event_bus.subscribe(EventType.SIGNAL, self._handle_signal)
        self.event_bus.subscribe(EventType.ORDER, self._handle_order)
        self.event_bus.subscribe(EventType.FILL, self._handle_fill)
    
    def add_data(
        self,
        symbol: str,
        data: pl.DataFrame,
    ) -> None:
        """
        Add market data for a symbol.
        
        Args:
            symbol: Trading symbol
            data: OHLCV DataFrame
        """
        # Validate required columns
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = set(required) - set(data.columns)
        if missing:
            raise BacktestError(f"Missing columns: {missing}")
        
        # Sort by timestamp
        data = data.sort("timestamp")
        
        self._data[symbol] = data
        logger.info(f"Added data for {symbol}: {len(data)} bars")
    
    def add_data_provider(
        self,
        provider: DataProvider,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        """
        Add data from a data provider.
        
        Args:
            provider: Data provider instance
            symbols: List of symbols to load
            start_date: Start date
            end_date: End date
        """
        for symbol in symbols:
            data = provider.get_historical_data(
                symbol, start_date, end_date, self.config.timeframe
            )
            self.add_data(symbol, data)
    
    def add_strategy(self, strategy: Strategy) -> None:
        """
        Add a trading strategy.
        
        Args:
            strategy: Strategy instance
        """
        self._strategies.append(strategy)
        logger.info(f"Added strategy: {strategy.name}")
    
    def set_benchmark(self, data: pl.DataFrame) -> None:
        """
        Set benchmark data for comparison.
        
        Args:
            data: Benchmark OHLCV DataFrame
        """
        self._benchmark_data = data.sort("timestamp")
    
    def run(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        show_progress: bool = True,
    ) -> PerformanceReport:
        """
        Run the backtest.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            show_progress: Show progress bar
            
        Returns:
            PerformanceReport with results
        """
        if not self._data:
            raise BacktestError("No data loaded")
        
        if not self._strategies:
            raise BacktestError("No strategies added")
        
        logger.info("Starting backtest...")
        
        # Reset state
        self._reset()
        
        # Get common timestamps across all symbols
        timestamps = self._get_aligned_timestamps(start_date, end_date)
        
        if len(timestamps) == 0:
            raise BacktestError("No data in specified date range")
        
        # Initialize strategies
        symbols = list(self._data.keys())
        for strategy in self._strategies:
            strategy.initialize(symbols, timestamps[0], timestamps[-1])
            strategy.start()
        
        self._is_running = True
        n_bars = len(timestamps)
        
        logger.info(f"Processing {n_bars} bars from {timestamps[0]} to {timestamps[-1]}")
        
        # Main loop
        for i, timestamp in enumerate(timestamps):
            self._current_bar_idx = i
            self._current_timestamp = timestamp
            
            # Skip warmup period
            if i < self.config.warmup_bars:
                self._update_portfolio_prices(timestamp)
                self.portfolio.record_equity(timestamp)
                continue
            
            # Process bar
            self._process_bar(timestamp)
            
            # Update equity
            self.portfolio.record_equity(timestamp)
            
            # Progress
            if show_progress and (i + 1) % 100 == 0:
                logger.debug(f"Processed {i + 1}/{n_bars} bars")
        
        self._is_running = False
        
        # Shutdown strategies
        for strategy in self._strategies:
            strategy.shutdown()
        
        # Calculate results
        report = self._generate_report()
        
        logger.info(f"Backtest complete. Total return: {report.total_return_pct:.2%}")
        
        return report
    
    def _reset(self) -> None:
        """Reset engine state."""
        self.portfolio.reset()
        self._signals.clear()
        self._orders.clear()
        self._fills.clear()
        self._current_bar_idx = 0
        self._current_timestamp = None
    
    def _get_aligned_timestamps(
        self,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> list[datetime]:
        """Get timestamps present in all data."""
        if not self._data:
            return []
        
        # Get timestamps from first symbol
        first_symbol = next(iter(self._data.keys()))
        timestamps = set(self._data[first_symbol]["timestamp"].to_list())
        
        # Intersect with other symbols
        for symbol, data in self._data.items():
            symbol_timestamps = set(data["timestamp"].to_list())
            timestamps &= symbol_timestamps
        
        # Convert to sorted list
        timestamps_list = sorted(timestamps)
        
        # Apply date filters
        if start_date:
            timestamps_list = [t for t in timestamps_list if t >= start_date]
        if end_date:
            timestamps_list = [t for t in timestamps_list if t <= end_date]
        
        return timestamps_list
    
    def _process_bar(self, timestamp: datetime) -> None:
        """Process a single bar."""
        # 1. Get current bar data for all symbols
        bar_data = self._get_bar_data(timestamp)
        
        # 2. Process pending orders (fill at this bar if CURRENT_BAR mode)
        if self.config.fill_mode == OrderFillMode.CURRENT_BAR:
            self._process_pending_orders(bar_data, timestamp)
        
        # 3. Update portfolio prices
        self._update_portfolio_prices(timestamp)
        
        # 4. Generate signals from strategies
        portfolio_state = self.portfolio.get_state(timestamp)
        
        for strategy in self._strategies:
            # Create market event
            for symbol, data in self._data.items():
                # Get history up to current bar
                history = data.filter(pl.col("timestamp") <= timestamp)
                
                market_event = MarketEvent(
                    symbol=symbol,
                    data=history,
                    timeframe=self.config.timeframe,
                    is_realtime=False,
                )
                
                # Get signals from strategy
                signals = strategy.on_bar(market_event, portfolio_state)
                
                for signal in signals:
                    self._signals.append(signal)
                    self.event_bus.publish(signal)
        
        # 5. Process pending orders (fill at next bar if NEXT_BAR mode)
        if self.config.fill_mode in [OrderFillMode.NEXT_BAR_OPEN, OrderFillMode.NEXT_BAR_CLOSE]:
            self._process_pending_orders(bar_data, timestamp)
    
    def _get_bar_data(self, timestamp: datetime) -> dict[str, dict[str, Any]]:
        """Get bar data for all symbols at timestamp."""
        bar_data = {}
        
        for symbol, data in self._data.items():
            bar = data.filter(pl.col("timestamp") == timestamp)
            
            if len(bar) > 0:
                bar_dict = bar.to_dicts()[0]
                bar_data[symbol] = bar_dict
        
        return bar_data
    
    def _update_portfolio_prices(self, timestamp: datetime) -> None:
        """Update portfolio with current prices."""
        prices = {}
        
        for symbol, data in self._data.items():
            bar = data.filter(pl.col("timestamp") == timestamp)
            if len(bar) > 0:
                prices[symbol] = bar["close"].item()
        
        self.portfolio.update_prices(prices)
    
    def _handle_signal(self, event: SignalEvent) -> None:
        """Handle a signal event."""
        # Skip if not running
        if not self._is_running:
            return
        
        # Check if we can take the position
        if not self._validate_signal(event):
            return
        
        # Convert signal to order
        order = self._create_order_from_signal(event)
        
        if order:
            self._orders.append(order)
            self.portfolio.pending_orders.append(order)
            
            order_event = OrderEvent(
                order_id=order.id,
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                price=order.price,
                status="pending",
                signal_id=event.id,
            )
            self.event_bus.publish(order_event)
    
    def _validate_signal(self, signal: SignalEvent) -> bool:
        """Validate if signal can be executed."""
        # Reject entry signals when there's already a position
        if signal.is_entry:
            pos = self.portfolio.get_position(signal.symbol)
            if pos and pos.is_open:
                logger.debug(f"Signal rejected: already have position in {signal.symbol}")
                return False
        # Check position limits
        if signal.is_entry:
            if self.portfolio.num_positions >= self.config.max_positions:
                logger.debug(f"Signal rejected: max positions reached")
                return False
        
        # Check shorting
        if signal.is_short and not self.config.allow_shorting:
            logger.debug(f"Signal rejected: shorting not allowed")
            return False
        
        return True
    
    def _create_order_from_signal(self, signal: SignalEvent) -> Order | None:
        """Create order from signal."""
        # Calculate quantity
        quantity = self._calculate_position_size(signal)
        
        if quantity <= 0:
            return None
        
        # Determine side
        if signal.is_entry:
            side = "buy" if signal.is_long else "sell"
        else:
            # Exit - reverse of current position
            pos = self.portfolio.get_position(signal.symbol)
            if pos and pos.is_open:
                side = "sell" if pos.quantity > 0 else "buy"
                quantity = abs(pos.quantity)
            else:
                return None
        
        order = Order.create_market_order(
            symbol=signal.symbol,
            side=side,
            quantity=quantity,
            signal_id=signal.id,
        )
        
        # Add stop/take profit if specified
        if signal.stop_loss:
            order.stop_price = signal.stop_loss
        
        order.metadata["signal_strength"] = signal.strength
        order.metadata["strategy"] = signal.strategy_name
        
        return order
    
    def _calculate_position_size(self, signal: SignalEvent) -> float:
        """Calculate position size for signal."""
        if self.config.position_sizing == "fixed":
            # Fixed fraction of equity
            equity = self.portfolio.equity
            target_value = equity * self.config.risk_per_trade
            
            if signal.price > 0:
                quantity = target_value / signal.price
            else:
                quantity = 0
        
        elif self.config.position_sizing == "percent":
            # Percentage of portfolio
            equity = self.portfolio.equity
            target_value = equity * 0.1  # 10% per position
            
            if signal.price > 0:
                quantity = target_value / signal.price
            else:
                quantity = 0
        
        elif self.config.position_sizing == "volatility":
            # Volatility-based (ATR)
            target_risk = self.portfolio.equity * self.config.risk_per_trade
            
            # Use signal metadata for ATR if available
            atr = signal.metadata.get("atr", signal.price * 0.02)
            
            if atr > 0:
                quantity = target_risk / atr
            else:
                quantity = 0
        
        else:
            # Default: use signal strength
            base_quantity = self.portfolio.equity * 0.05 / signal.price if signal.price > 0 else 0
            quantity = base_quantity * signal.strength
        
        # Apply fractional shares setting
        if not self.config.fractional_shares:
            quantity = int(quantity)
        
        return max(0, quantity)
    
    def _process_pending_orders(
        self,
        bar_data: dict[str, dict[str, Any]],
        timestamp: datetime,
    ) -> None:
        """Process pending orders against current bar data."""
        orders_to_remove = []
        
        for order in self.portfolio.pending_orders:
            symbol = order.symbol
            
            if symbol not in bar_data:
                continue
            
            bar = bar_data[symbol]
            
            # Create OHLCV object for execution
            ohlcv = OHLCV(
                timestamp=timestamp,
                open=bar["open"],
                high=bar["high"],
                low=bar["low"],
                close=bar["close"],
                volume=bar["volume"],
                symbol=symbol,
            )
            
            # Simulate fill
            fill = self.execution.simulate_fill(order, ohlcv)
            
            if fill.filled:
                # Add metadata
                fill.metadata["symbol"] = symbol
                fill.metadata["side"] = order.side
                
                # Process fill
                trade = self.portfolio.process_fill(fill, timestamp)
                
                self._fills.append(fill)
                orders_to_remove.append(order)
                
                # Update order status
                order.status = OrderStatus.FILLED
                order.filled_quantity = fill.fill_quantity
                order.filled_price = fill.fill_price
                order.commission = fill.commission
                
                # Publish fill event
                fill_event = FillEvent(
                    order_id=order.id,
                    symbol=symbol,
                    side=order.side,
                    quantity=fill.fill_quantity,
                    price=fill.fill_price,
                    commission=fill.commission,
                    fill_time=timestamp,
                )
                self.event_bus.publish(fill_event)
                
                # Notify strategy
                for strategy in self._strategies:
                    strategy.on_fill(fill_event)
                    
                    # Sync strategy position state with portfolio
                    if hasattr(strategy, '_current_positions'):
                        portfolio_pos = self.portfolio.get_position(order.symbol)
                        if portfolio_pos:
                            strategy._current_positions[order.symbol] = portfolio_pos
                        elif order.symbol in strategy._current_positions:
                            # Position closed - remove from strategy tracking
                            del strategy._current_positions[order.symbol]
        
        # Remove filled orders
        for order in orders_to_remove:
            self.portfolio.pending_orders.remove(order)
    
    def _handle_order(self, event: OrderEvent) -> None:
        """Handle order event."""
        pass  # Orders are processed in _process_pending_orders
    
    def _handle_fill(self, event: FillEvent) -> None:
        """Handle fill event."""
        pass  # Fills are processed in _process_pending_orders
    
    def _generate_report(self) -> PerformanceReport:
        """Generate performance report."""
        equity_curve = self.portfolio.get_equity_curve()
        timestamps = self.portfolio.get_timestamps()
        trades = self.portfolio.get_trade_list()
        
        # Get benchmark returns if available
        benchmark_returns = None
        if self._benchmark_data is not None:
            bench_close = self._benchmark_data["close"].to_numpy()
            benchmark_returns = np.diff(bench_close) / bench_close[:-1]
        
        # Use MetricsCalculator
        calculator = MetricsCalculator(
            equity_curve=equity_curve,
            timestamps=timestamps,
            trades=trades,
            initial_capital=self.config.initial_capital,
            benchmark_returns=benchmark_returns,
            strategy_name=self._strategies[0].name if self._strategies else "Backtest",
        )
        
        return calculator.calculate_all()
    
    def get_equity_curve(self) -> pl.DataFrame:
        """Get equity curve as DataFrame."""
        timestamps = self.portfolio.get_timestamps()
        equity = self.portfolio.get_equity_curve()
        
        return pl.DataFrame({
            "timestamp": timestamps,
            "equity": equity,
        })
    
    def get_trades(self) -> list[Trade]:
        """Get list of trades."""
        return self.portfolio.trades.copy()
    
    def get_signals(self) -> list[SignalEvent]:
        """Get list of signals generated."""
        return self._signals.copy()
    
    def get_orders(self) -> list[Order]:
        """Get list of orders."""
        return self._orders.copy()


# =============================================================================
# WALK-FORWARD ANALYSIS
# =============================================================================

@dataclass
class WalkForwardResult:
    """Result of a single walk-forward fold."""
    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_metrics: PerformanceReport
    test_metrics: PerformanceReport
    optimized_params: dict[str, Any] = field(default_factory=dict)


class WalkForwardAnalyzer:
    """
    Walk-forward analysis for strategy validation.
    
    Implements rolling window optimization and testing
    to assess strategy robustness.
    
    Example:
        analyzer = WalkForwardAnalyzer(
            engine=BacktestEngine(config),
            n_splits=5,
            train_ratio=0.6,
        )
        
        results = analyzer.run(
            strategy_class=MyStrategy,
            data=data,
            param_grid=param_grid,
        )
    """
    
    def __init__(
        self,
        engine: BacktestEngine,
        n_splits: int = 5,
        train_ratio: float = 0.6,
        gap_bars: int = 0,
    ):
        """
        Initialize walk-forward analyzer.
        
        Args:
            engine: Backtest engine to use
            n_splits: Number of walk-forward folds
            train_ratio: Ratio of data for training
            gap_bars: Gap between train and test periods
        """
        self.engine = engine
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.gap_bars = gap_bars
        
        self.results: list[WalkForwardResult] = []
    
    def run(
        self,
        strategy_factory: Callable[..., Strategy],
        data: dict[str, pl.DataFrame],
        param_grid: dict[str, list[Any]] | None = None,
        optimize_metric: str = "sharpe_ratio",
    ) -> list[WalkForwardResult]:
        """
        Run walk-forward analysis.
        
        Args:
            strategy_factory: Factory function to create strategy
            data: Symbol to DataFrame mapping
            param_grid: Parameters to optimize
            optimize_metric: Metric to optimize
            
        Returns:
            List of WalkForwardResult
        """
        self.results.clear()
        
        # Get common timestamps
        all_timestamps = self._get_common_timestamps(data)
        n_bars = len(all_timestamps)
        
        if n_bars == 0:
            raise BacktestError("No common timestamps in data")
        
        # Calculate split sizes
        fold_size = n_bars // self.n_splits
        train_size = int(fold_size * self.train_ratio)
        test_size = fold_size - train_size - self.gap_bars
        
        logger.info(f"Walk-forward: {self.n_splits} folds, {train_size} train bars, {test_size} test bars")
        
        for fold in range(self.n_splits):
            # Calculate indices
            fold_start = fold * fold_size
            train_end_idx = fold_start + train_size
            test_start_idx = train_end_idx + self.gap_bars
            test_end_idx = min(test_start_idx + test_size, n_bars)
            
            if test_end_idx <= test_start_idx:
                continue
            
            train_start = all_timestamps[fold_start]
            train_end = all_timestamps[train_end_idx - 1]
            test_start = all_timestamps[test_start_idx]
            test_end = all_timestamps[test_end_idx - 1]
            
            logger.info(f"Fold {fold + 1}: Train {train_start} to {train_end}, Test {test_start} to {test_end}")
            
            # Optimize on training period (if param_grid provided)
            best_params = {}
            if param_grid:
                best_params = self._optimize_parameters(
                    strategy_factory, data, param_grid,
                    train_start, train_end, optimize_metric
                )
            
            # Create strategy with best params
            strategy = strategy_factory(**best_params) if best_params else strategy_factory()
            
            # Run on training period
            self.engine.portfolio.reset()
            for symbol, df in data.items():
                self.engine.add_data(symbol, df)
            self.engine._strategies = [strategy]
            
            train_report = self.engine.run(train_start, train_end, show_progress=False)
            
            # Run on test period
            self.engine.portfolio.reset()
            test_strategy = strategy_factory(**best_params) if best_params else strategy_factory()
            self.engine._strategies = [test_strategy]
            
            test_report = self.engine.run(test_start, test_end, show_progress=False)
            
            # Store result
            result = WalkForwardResult(
                fold=fold + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_metrics=train_report,
                test_metrics=test_report,
                optimized_params=best_params,
            )
            self.results.append(result)
        
        return self.results
    
    def _get_common_timestamps(
        self,
        data: dict[str, pl.DataFrame],
    ) -> list[datetime]:
        """Get timestamps present in all DataFrames."""
        if not data:
            return []
        
        timestamps = None
        for df in data.values():
            ts_set = set(df["timestamp"].to_list())
            if timestamps is None:
                timestamps = ts_set
            else:
                timestamps &= ts_set
        
        return sorted(timestamps) if timestamps else []
    
    def _optimize_parameters(
        self,
        strategy_factory: Callable[..., Strategy],
        data: dict[str, pl.DataFrame],
        param_grid: dict[str, list[Any]],
        start_date: datetime,
        end_date: datetime,
        optimize_metric: str,
    ) -> dict[str, Any]:
        """Optimize strategy parameters on training data."""
        best_metric = float('-inf')
        best_params = {}
        
        # Generate parameter combinations
        from itertools import product
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for values in product(*param_values):
            params = dict(zip(param_names, values))
            
            # Create strategy
            strategy = strategy_factory(**params)
            
            # Run backtest
            self.engine.portfolio.reset()
            self.engine._strategies = [strategy]
            
            try:
                report = self.engine.run(start_date, end_date, show_progress=False)
                metric_value = getattr(report, optimize_metric, 0)
                
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_params = params.copy()
            except Exception as e:
                logger.warning(f"Parameter combination failed: {params}, error: {e}")
        
        return best_params
    
    def get_summary(self) -> dict[str, Any]:
        """Get walk-forward analysis summary."""
        if not self.results:
            return {}
        
        train_returns = [r.train_metrics.annualized_return for r in self.results]
        test_returns = [r.test_metrics.annualized_return for r in self.results]
        train_sharpes = [r.train_metrics.sharpe_ratio for r in self.results]
        test_sharpes = [r.test_metrics.sharpe_ratio for r in self.results]
        
        return {
            "n_folds": len(self.results),
            "avg_train_return": np.mean(train_returns),
            "avg_test_return": np.mean(test_returns),
            "avg_train_sharpe": np.mean(train_sharpes),
            "avg_test_sharpe": np.mean(test_sharpes),
            "train_test_correlation": np.corrcoef(train_returns, test_returns)[0, 1],
            "robustness_ratio": np.mean(test_sharpes) / np.mean(train_sharpes) if np.mean(train_sharpes) != 0 else 0,
            "positive_test_folds": sum(1 for r in test_returns if r > 0),
            "consistent_folds": sum(1 for tr, te in zip(train_returns, test_returns) if (tr > 0) == (te > 0)),
        }


# =============================================================================
# REPORT GENERATOR
# =============================================================================

class ReportGenerator:
    """
    Generate backtest reports in various formats.
    
    Supports HTML, JSON, and text reports.
    """
    
    def __init__(
        self,
        report: PerformanceReport,
        equity_curve: pl.DataFrame | None = None,
        trades: list[Trade] | None = None,
    ):
        """
        Initialize report generator.
        
        Args:
            report: Performance report
            equity_curve: Equity curve DataFrame
            trades: List of trades
        """
        self.report = report
        self.equity_curve = equity_curve
        self.trades = trades or []
    
    def to_json(self, path: Path | str | None = None) -> str:
        """
        Generate JSON report.
        
        Args:
            path: Optional file path to save
            
        Returns:
            JSON string
        """
        data = self.report.to_dict()
        data["trades"] = [t.to_dict() for t in self.trades]
        
        json_str = json.dumps(data, indent=2, default=str)
        
        if path:
            Path(path).write_text(json_str)
        
        return json_str
    
    def to_text(self) -> str:
        """Generate text summary report."""
        r = self.report
        
        lines = [
            "=" * 60,
            f"BACKTEST REPORT: {r.strategy_name}",
            "=" * 60,
            "",
            "PERIOD",
            f"  Start Date:        {r.start_date}",
            f"  End Date:          {r.end_date}",
            f"  Initial Capital:   ${r.initial_capital:,.2f}",
            f"  Final Capital:     ${r.final_capital:,.2f}",
            "",
            "RETURNS",
            f"  Total Return:      {r.total_return_pct:.2%}",
            f"  Annualized Return: {r.annualized_return:.2%}",
            f"  Best Month:        {r.best_month:.2%}",
            f"  Worst Month:       {r.worst_month:.2%}",
            "",
            "RISK",
            f"  Volatility:        {r.annualized_volatility:.2%}",
            f"  Max Drawdown:      {r.max_drawdown:.2%}",
            f"  VaR (95%):         {r.var_95:.2%}",
            f"  CVaR (95%):        {r.cvar_95:.2%}",
            "",
            "RISK-ADJUSTED",
            f"  Sharpe Ratio:      {r.sharpe_ratio:.2f}",
            f"  Sortino Ratio:     {r.sortino_ratio:.2f}",
            f"  Calmar Ratio:      {r.calmar_ratio:.2f}",
            "",
            "TRADES",
            f"  Total Trades:      {r.trade_stats.total_trades}",
            f"  Win Rate:          {r.trade_stats.win_rate:.2%}",
            f"  Profit Factor:     {r.trade_stats.profit_factor:.2f}",
            f"  Avg Trade:         ${r.trade_stats.avg_trade:.2f}",
            f"  Expectancy:        ${r.trade_stats.expectancy:.2f}",
            "",
            "=" * 60,
        ]
        
        return "\n".join(lines)
    
    def print_summary(self) -> None:
        """Print text summary to console."""
        print(self.to_text())


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_backtest(
    strategy: Strategy,
    data: dict[str, pl.DataFrame],
    config: BacktestConfig | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> PerformanceReport:
    """
    Convenience function to run a backtest.
    
    Args:
        strategy: Strategy to test
        data: Symbol to DataFrame mapping
        config: Backtest configuration
        start_date: Start date
        end_date: End date
        
    Returns:
        PerformanceReport
    """
    engine = BacktestEngine(config)
    
    for symbol, df in data.items():
        engine.add_data(symbol, df)
    
    engine.add_strategy(strategy)
    
    return engine.run(start_date, end_date)


def quick_backtest(
    strategy: Strategy,
    symbol: str,
    data: pl.DataFrame,
    initial_capital: float = 100_000,
) -> PerformanceReport:
    """
    Quick backtest with minimal configuration.
    
    Args:
        strategy: Strategy to test
        symbol: Symbol name
        data: OHLCV DataFrame
        initial_capital: Starting capital
        
    Returns:
        PerformanceReport
    """
    config = BacktestConfig(initial_capital=initial_capital)
    return run_backtest(strategy, {symbol: data}, config)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "BacktestMode",
    "OrderFillMode",
    
    # Configuration
    "BacktestConfig",
    
    # Core classes
    "PortfolioTracker",
    "BacktestEngine",
    
    # Walk-forward
    "WalkForwardResult",
    "WalkForwardAnalyzer",
    
    # Reporting
    "ReportGenerator",
    
    # Convenience functions
    "run_backtest",
    "quick_backtest",
]