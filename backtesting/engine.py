"""
Backtesting Engine Module (FIXED)
=================================

Event-driven backtesting engine with CORRECT strategy interface.

CRITICAL FIXES:
1. Uses strategy.on_bar(event, portfolio) instead of generate_signal()
2. Properly creates MarketEvent for strategy consumption
3. Passes PortfolioState to strategy
4. Correct timestamp handling throughout

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

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
    Event, EventType, EventBus, EventPriority,
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
    VECTORIZED = "vectorized"
    EVENT_DRIVEN = "event_driven"
    TICK = "tick"


class OrderFillMode(str, Enum):
    """Order fill timing."""
    CURRENT_BAR = "current_bar"
    NEXT_BAR_OPEN = "next_bar_open"
    NEXT_BAR_CLOSE = "next_bar_close"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    # Capital
    initial_capital: float = 100_000.0
    
    # Costs
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    
    # Trading rules
    margin_requirement: float = 1.0
    allow_shorting: bool = True
    fractional_shares: bool = True
    
    # Execution
    fill_mode: OrderFillMode = OrderFillMode.NEXT_BAR_OPEN
    
    # Position management
    max_positions: int = 10
    position_sizing: str = "fixed"
    risk_per_trade: float = 0.02
    
    # Other
    warmup_bars: int = 50
    benchmark_symbol: str | None = None
    
    # Data frequency (for correct annualization)
    timeframe: str = "15min"
    periods_per_year: int = 15794  # 15-min bars with extended hours
    
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
            "periods_per_year": self.periods_per_year,
        }


# =============================================================================
# PORTFOLIO TRACKER
# =============================================================================

class PortfolioTracker:
    """
    Tracks portfolio state during backtesting.
    
    All methods that update positions require an explicit timestamp
    parameter (the simulated market time, NOT datetime.now()).
    """
    
    def __init__(
        self,
        initial_capital: float,
        allow_shorting: bool = True,
        margin_requirement: float = 1.0,
    ):
        """Initialize portfolio tracker."""
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.allow_shorting = allow_shorting
        self.margin_requirement = margin_requirement
        
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.equity_history: list[tuple[datetime, float]] = []
        self.cash_history: list[tuple[datetime, float]] = []
        self.pending_orders: list[Order] = []
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
    
    def update_prices(
        self,
        prices: dict[str, float],
        timestamp: datetime,
    ) -> None:
        """Update current prices for all positions."""
        self._current_prices.update(prices)
        
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.update_price(prices[symbol], timestamp)
    
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
        """Process an order fill."""
        if not fill.filled:
            return None
        
        symbol = fill.metadata.get("symbol", "UNKNOWN")
        side = fill.metadata.get("side", "buy")
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        
        pos = self.positions[symbol]
        
        if side.lower() == "buy":
            cost = fill.fill_quantity * fill.fill_price + fill.commission
            
            if pos.quantity < 0:
                # Closing short
                trade = self._close_position_partial(
                    pos, fill.fill_quantity, fill.fill_price,
                    fill.commission, timestamp
                )
                self.cash += fill.fill_quantity * fill.fill_price - fill.commission
                return trade
            else:
                # Opening or adding to long
                pos.add(fill.fill_quantity, fill.fill_price, timestamp)
                self.cash -= cost
        else:  # sell
            if pos.quantity > 0:
                # Closing long
                trade = self._close_position_partial(
                    pos, fill.fill_quantity, fill.fill_price,
                    fill.commission, timestamp
                )
                self.cash += fill.fill_quantity * fill.fill_price - fill.commission
                return trade
            else:
                # Opening or adding to short
                if not self.allow_shorting:
                    logger.warning(f"Short selling not allowed for {symbol}")
                    return None
                pos.add(-fill.fill_quantity, fill.fill_price, timestamp)
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
        """Close or reduce a position with correct timestamps."""
        if quantity >= abs(pos.quantity):
            # Full close
            trade = Trade.create(
                symbol=pos.symbol,
                side="long" if pos.quantity > 0 else "short",
                entry_price=pos.avg_price,
                quantity=abs(pos.quantity),
                entry_time=pos.opened_at or timestamp,
            )
            trade.close(price, timestamp, commission)
            
            pos.quantity = 0
            pos.avg_price = 0
            pos.realized_pnl += trade.pnl
            
            self.trades.append(trade)
            return trade
        else:
            # Partial close
            realized = pos.reduce(quantity, price, timestamp)
            pos.realized_pnl += realized
            
            trade = Trade.create(
                symbol=pos.symbol,
                side="long" if pos.quantity > 0 else "short",
                entry_price=pos.avg_price,
                quantity=quantity,
                entry_time=pos.opened_at or timestamp,
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
# BACKTEST ENGINE (FIXED!)
# =============================================================================

class BacktestEngine:
    """
    Event-driven backtesting engine.
    
    CRITICAL FIX: Now uses strategy.on_bar(event, portfolio) 
    instead of the non-existent generate_signal() method.
    """
    
    def __init__(self, config: BacktestConfig | None = None):
        """Initialize backtest engine."""
        self.config = config or BacktestConfig()
        
        self.portfolio = PortfolioTracker(
            initial_capital=self.config.initial_capital,
            allow_shorting=self.config.allow_shorting,
            margin_requirement=self.config.margin_requirement,
        )
        
        self.execution = ExecutionSimulator(
            slippage_model=PercentageSlippage(self.config.slippage_pct),
            commission_model=PerShareCommission(per_share=0.0, min_commission=0.0),
        )
        
        self.event_bus = EventBus()
        
        self._data: dict[str, pl.DataFrame] = {}
        self._strategies: list[Any] = []  # Use Any to avoid import issues
        self._benchmark_data: pl.DataFrame | None = None
        
        self._current_bar_idx: int = 0
        self._current_timestamp: datetime | None = None
        self._is_running: bool = False
        
        self._signals: list[SignalEvent] = []
        self._orders: list[Order] = []
        self._fills: list[FillResult] = []
        
        self._setup_event_handlers()
    
    def _setup_event_handlers(self) -> None:
        """Setup event bus handlers."""
        self.event_bus.subscribe(EventType.SIGNAL, self._handle_signal)
        self.event_bus.subscribe(EventType.ORDER, self._handle_order)
        self.event_bus.subscribe(EventType.FILL, self._handle_fill)
    
    def add_data(self, symbol: str, data: pl.DataFrame) -> None:
        """Add market data for a symbol."""
        if "timestamp" not in data.columns:
            raise BacktestError(f"Data for {symbol} missing 'timestamp' column")
        
        self._data[symbol] = data.sort("timestamp")
        logger.debug(f"Added {len(data)} bars for {symbol}")
    
    def add_strategy(self, strategy: Any) -> None:
        """Add a strategy to the engine."""
        self._strategies.append(strategy)
    
    def set_benchmark(self, data: pl.DataFrame) -> None:
        """Set benchmark data for comparison."""
        self._benchmark_data = data.sort("timestamp")
    
    def run(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        show_progress: bool = True,
    ) -> PerformanceReport:
        """Run the backtest."""
        if not self._data:
            raise BacktestError("No data loaded")
        
        if not self._strategies:
            raise BacktestError("No strategies added")
        
        logger.info("Starting backtest...")
        
        self._reset()
        
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
            
            # Progress logging
            if show_progress and (i + 1) % 5000 == 0:
                logger.info(f"Processed {i + 1}/{n_bars} bars ({100*(i+1)/n_bars:.1f}%)")
        
        self._is_running = False
        
        # Shutdown strategies
        for strategy in self._strategies:
            strategy.shutdown()
        
        # Generate report
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
        """Get timestamps common to all data sources."""
        if not self._data:
            return []
        
        timestamps = None
        for symbol, df in self._data.items():
            ts = set(df["timestamp"].to_list())
            if timestamps is None:
                timestamps = ts
            else:
                timestamps &= ts
        
        if not timestamps:
            return []
        
        sorted_ts = sorted(timestamps)
        
        if start_date:
            sorted_ts = [t for t in sorted_ts if t >= start_date]
        if end_date:
            sorted_ts = [t for t in sorted_ts if t <= end_date]
        
        return sorted_ts
    
    def _process_bar(self, timestamp: datetime) -> None:
        """
        Process a single bar.
        
        CRITICAL FIX: Uses strategy.on_bar(event, portfolio) with proper
        MarketEvent and PortfolioState objects.
        """
        bar_data = self._get_bar_data(timestamp)
        
        if not bar_data:
            return
        
        # Update portfolio prices FIRST
        self._update_portfolio_prices(timestamp)
        
        # Process pending orders if using current bar fill
        if self.config.fill_mode == OrderFillMode.CURRENT_BAR:
            self._process_pending_orders(bar_data, timestamp)
        
        # Get current portfolio state for strategies
        portfolio_state = self.portfolio.get_state(timestamp)
        
        # Process each strategy
        for strategy in self._strategies:
            for symbol, data in bar_data.items():
                # Get historical data up to current timestamp
                hist_data = self._data.get(symbol)
                if hist_data is None:
                    continue
                
                hist_to_now = hist_data.filter(pl.col("timestamp") <= timestamp)
                
                # Create MarketEvent for the strategy
                market_event = MarketEvent(
                    timestamp=timestamp,
                    symbol=symbol,
                    data=hist_to_now,
                    ohlcv=OHLCV(
                        timestamp=timestamp,
                        open=data.get("open", 0),
                        high=data.get("high", 0),
                        low=data.get("low", 0),
                        close=data.get("close", 0),
                        volume=data.get("volume", 0),
                        symbol=symbol,
                    ),
                )
                
                # CRITICAL FIX: Call on_bar() with correct signature!
                try:
                    signals = strategy.on_bar(market_event, portfolio_state)
                    
                    # Process returned signals
                    if signals:
                        for signal in signals:
                            self._signals.append(signal)
                            self.event_bus.publish(signal)
                            
                except Exception as e:
                    logger.error(f"Strategy error on {symbol}: {e}")
                    continue
        
        # Process pending orders after signals
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
        
        self.portfolio.update_prices(prices, timestamp)
    
    def _handle_signal(self, event: SignalEvent) -> None:
        """Handle a signal event."""
        if not self._is_running:
            return
        
        if not self._validate_signal(event):
            return
        
        order = self._create_order_from_signal(event)
        
        if order:
            self._orders.append(order)
            self.portfolio.pending_orders.append(order)
    
    def _validate_signal(self, signal: SignalEvent) -> bool:
        """Validate if signal can be executed."""
        # Check if it's an entry signal
        if signal.is_entry:
            pos = self.portfolio.get_position(signal.symbol)
            if pos and pos.is_open:
                return False  # Already have position
            
            if self.portfolio.num_positions >= self.config.max_positions:
                return False  # Max positions reached
        
        # Check shorting restriction
        if signal.is_short and not self.config.allow_shorting:
            return False
        
        return True
    
    def _create_order_from_signal(self, signal: SignalEvent) -> Order | None:
        """Create order from signal."""
        # Calculate position size
        position_value = self.portfolio.equity * self.config.risk_per_trade
        quantity = position_value / signal.price if signal.price > 0 else 0
        
        if not self.config.fractional_shares:
            quantity = int(quantity)
        
        if quantity <= 0:
            return None
        
        # Determine order side
        if signal.is_entry:
            side = "buy" if signal.is_long else "sell"
        else:
            # Exit signal
            pos = self.portfolio.get_position(signal.symbol)
            if pos:
                side = "sell" if pos.quantity > 0 else "buy"
                quantity = abs(pos.quantity)
            else:
                return None
        
        return Order.create_market_order(
            symbol=signal.symbol,
            side=side,
            quantity=quantity,
            timestamp=signal.timestamp,
            signal_id=signal.id if hasattr(signal, 'id') else None,
        )
    
    def _process_pending_orders(
        self,
        bar_data: dict[str, dict[str, Any]],
        timestamp: datetime,
    ) -> None:
        """Process pending orders."""
        for order in list(self.portfolio.pending_orders):
            if order.symbol not in bar_data:
                continue
            
            bar = bar_data[order.symbol]
            
            # Determine fill price
            if self.config.fill_mode == OrderFillMode.CURRENT_BAR:
                fill_price = bar.get("close", 0)
            elif self.config.fill_mode == OrderFillMode.NEXT_BAR_OPEN:
                fill_price = bar.get("open", 0)
            else:
                fill_price = bar.get("close", 0)
            
            # Execute order
            fill = self.execution.execute(
                order=order,
                price=fill_price,
                volume=bar.get("volume", 1000000),
                high=bar.get("high", fill_price),
                low=bar.get("low", fill_price),
            )
            
            fill.metadata["symbol"] = order.symbol
            fill.metadata["side"] = order.side
            
            # Process fill
            trade = self.portfolio.process_fill(fill, timestamp)
            
            if fill.filled:
                order.fill(fill.fill_quantity, fill.fill_price, fill.commission)
                self.portfolio.pending_orders.remove(order)
                self._fills.append(fill)
    
    def _handle_order(self, event: OrderEvent) -> None:
        """Handle order event."""
        pass
    
    def _handle_fill(self, event: FillEvent) -> None:
        """Handle fill event."""
        pass
    
    def _generate_report(self) -> PerformanceReport:
        """Generate performance report."""
        equity_curve = self.portfolio.get_equity_curve()
        timestamps = self.portfolio.get_timestamps()
        trades = self.portfolio.trades
        
        # Get benchmark returns if available
        benchmark_returns = None
        if self._benchmark_data is not None:
            bench_close = self._benchmark_data["close"].to_numpy()
            benchmark_returns = np.diff(bench_close) / bench_close[:-1]
        
        # Use correct periods_per_year from config
        calculator = MetricsCalculator(
            equity_curve=equity_curve,
            timestamps=timestamps,
            trades=trades,
            initial_capital=self.config.initial_capital,
            benchmark_returns=benchmark_returns,
            strategy_name=self._strategies[0].name if self._strategies else "Backtest",
            periods_per_year=self.config.periods_per_year,
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
    """Walk-forward analysis for strategy validation."""
    
    def __init__(
        self,
        engine: BacktestEngine,
        n_splits: int = 5,
        train_ratio: float = 0.6,
        gap_bars: int = 0,
    ):
        """Initialize walk-forward analyzer."""
        self.engine = engine
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.gap_bars = gap_bars
        self.results: list[WalkForwardResult] = []
    
    def run(
        self,
        strategy_factory: Callable[..., Any],
        data: dict[str, pl.DataFrame],
        param_grid: dict[str, list[Any]] | None = None,
        optimize_metric: str = "sharpe_ratio",
    ) -> list[WalkForwardResult]:
        """Run walk-forward analysis."""
        self.results.clear()
        
        all_timestamps = self._get_common_timestamps(data)
        n_bars = len(all_timestamps)
        
        if n_bars == 0:
            raise BacktestError("No common timestamps in data")
        
        fold_size = n_bars // self.n_splits
        train_size = int(fold_size * self.train_ratio)
        test_size = fold_size - train_size - self.gap_bars
        
        logger.info(f"Walk-forward: {self.n_splits} folds, {train_size} train bars, {test_size} test bars")
        
        for fold in range(self.n_splits):
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
            
            # Optimize parameters
            best_params = {}
            if param_grid:
                best_params = self._optimize_parameters(
                    strategy_factory, data, param_grid,
                    train_start, train_end, optimize_metric
                )
            
            # Create strategy
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
    
    def _get_common_timestamps(self, data: dict[str, pl.DataFrame]) -> list[datetime]:
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
        strategy_factory: Callable[..., Any],
        data: dict[str, pl.DataFrame],
        param_grid: dict[str, list[Any]],
        start_date: datetime,
        end_date: datetime,
        optimize_metric: str,
    ) -> dict[str, Any]:
        """Optimize strategy parameters on training data."""
        from itertools import product
        
        best_metric = float('-inf')
        best_params = {}
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for values in product(*param_values):
            params = dict(zip(param_names, values))
            strategy = strategy_factory(**params)
            
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
            "train_test_correlation": np.corrcoef(train_returns, test_returns)[0, 1] if len(train_returns) > 1 else 0,
            "robustness_ratio": np.mean(test_sharpes) / np.mean(train_sharpes) if np.mean(train_sharpes) != 0 else 0,
            "positive_test_folds": sum(1 for r in test_returns if r > 0),
            "consistent_folds": sum(1 for tr, te in zip(train_returns, test_returns) if (tr > 0) == (te > 0)),
        }


# =============================================================================
# REPORT GENERATOR
# =============================================================================

class ReportGenerator:
    """Generate backtest reports in various formats."""
    
    def __init__(
        self,
        report: PerformanceReport,
        equity_curve: pl.DataFrame | None = None,
        trades: list[Trade] | None = None,
    ):
        """Initialize report generator."""
        self.report = report
        self.equity_curve = equity_curve
        self.trades = trades or []
    
    def to_json(self, path: Path | str | None = None) -> str:
        """Generate JSON report."""
        data = self.report.to_dict()
        data["trades"] = [t.to_dict() if hasattr(t, 'to_dict') else t for t in self.trades]
        
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
    strategy: Any,
    data: dict[str, pl.DataFrame],
    config: BacktestConfig | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> PerformanceReport:
    """Convenience function to run a backtest."""
    engine = BacktestEngine(config)
    
    for symbol, df in data.items():
        engine.add_data(symbol, df)
    
    engine.add_strategy(strategy)
    
    return engine.run(start_date, end_date)


def quick_backtest(
    strategy: Any,
    symbol: str,
    data: pl.DataFrame,
    initial_capital: float = 100_000,
) -> PerformanceReport:
    """Quick backtest with minimal configuration."""
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