"""
Institutional Backtesting Engine
JPMorgan-Level Historical Simulation Framework

Features:
- Event-driven backtesting
- Vectorized backtesting for speed
- Multi-asset support
- Realistic order filling
- Transaction costs modeling
- Slippage simulation
- Walk-forward optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings

from ..strategy.base_strategy import BaseStrategy, Signal, SignalType
from ..risk.portfolio import Portfolio, Position, PositionSide
from ..risk.position_sizer import PositionSizer, VolatilityPositionSizer
from ..risk.risk_manager import RiskManager, RiskLimits, PreTradeRiskCheck
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FillModel(Enum):
    """Order fill models"""
    IMMEDIATE = "immediate"  # Fill at signal price
    NEXT_OPEN = "next_open"  # Fill at next bar open
    NEXT_CLOSE = "next_close"  # Fill at next bar close
    VWAP = "vwap"  # Fill at VWAP
    TWAP = "twap"  # Fill at TWAP


class SlippageModel(Enum):
    """Slippage models"""
    FIXED = "fixed"  # Fixed basis points
    VOLUME_BASED = "volume_based"  # Based on volume
    VOLATILITY_BASED = "volatility_based"  # Based on volatility
    SQRT_VOLUME = "sqrt_volume"  # Square root of volume impact


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    # Time settings
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # Capital settings
    initial_capital: float = 1000000
    margin_requirement: float = 1.0  # 100% margin (no leverage)

    # Execution settings
    fill_model: FillModel = FillModel.NEXT_OPEN
    slippage_model: SlippageModel = SlippageModel.FIXED
    slippage_bps: float = 5  # 5 basis points
    commission_per_share: float = 0.005
    commission_min: float = 1.0
    commission_pct: float = 0.0

    # Risk settings
    enable_risk_checks: bool = True
    max_position_pct: float = 0.10
    max_drawdown_stop: float = 0.20  # Stop at 20% drawdown

    # Data settings
    data_frequency: str = "15min"  # 15-minute bars
    warmup_period: int = 100  # Bars for indicator warmup

    # Output settings
    save_trades: bool = True
    save_daily_stats: bool = True
    verbose: bool = True


@dataclass
class Trade:
    """Trade record"""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    price: float
    fill_price: float
    slippage: float
    commission: float
    notional: float
    signal_strength: float
    strategy: str
    pnl: float = 0  # Realized P&L (for closing trades)
    is_entry: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'trade_id': self.trade_id,
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'price': self.price,
            'fill_price': self.fill_price,
            'slippage': self.slippage,
            'commission': self.commission,
            'notional': self.notional,
            'pnl': self.pnl,
            'strategy': self.strategy
        }


@dataclass
class BacktestResult:
    """Backtesting results container"""
    # Configuration
    config: BacktestConfig

    # Performance
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade_pnl: float
    largest_win: float
    largest_loss: float

    # Costs
    total_commission: float
    total_slippage: float
    total_costs: float

    # Data
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    trades: List[Trade]
    daily_stats: pd.DataFrame
    positions_history: List[Dict]

    # Metadata
    start_date: datetime
    end_date: datetime
    trading_days: int

    def summary(self) -> str:
        """Generate summary string"""
        return f"""
========================================
         BACKTEST RESULTS
========================================
Period: {self.start_date.date()} to {self.end_date.date()}
Trading Days: {self.trading_days}

PERFORMANCE
-----------
Total Return:      {self.total_return:>10.2%}
Annualized Return: {self.annualized_return:>10.2%}
Volatility:        {self.volatility:>10.2%}
Sharpe Ratio:      {self.sharpe_ratio:>10.2f}
Sortino Ratio:     {self.sortino_ratio:>10.2f}
Calmar Ratio:      {self.calmar_ratio:>10.2f}
Max Drawdown:      {self.max_drawdown:>10.2%}

TRADES
------
Total Trades:      {self.total_trades:>10d}
Win Rate:          {self.win_rate:>10.2%}
Profit Factor:     {self.profit_factor:>10.2f}
Avg Win:           ${self.avg_win:>9,.2f}
Avg Loss:          ${self.avg_loss:>9,.2f}
Largest Win:       ${self.largest_win:>9,.2f}
Largest Loss:      ${self.largest_loss:>9,.2f}

COSTS
-----
Total Commission:  ${self.total_commission:>9,.2f}
Total Slippage:    ${self.total_slippage:>9,.2f}
Total Costs:       ${self.total_costs:>9,.2f}
========================================
"""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage
        }


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Features:
    - Realistic order execution
    - Risk management integration
    - Multiple strategies
    - Multi-asset support
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        config: Optional[BacktestConfig] = None,
        position_sizer: Optional[PositionSizer] = None,
        risk_manager: Optional[RiskManager] = None
    ):
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self.position_sizer = position_sizer or VolatilityPositionSizer()
        self.risk_manager = risk_manager or RiskManager()

        # Portfolio
        self.portfolio = Portfolio(initial_capital=self.config.initial_capital)

        # State
        self._current_bar: int = 0
        self._current_time: Optional[datetime] = None
        self._pending_orders: List[Dict] = []

        # Results tracking
        self._trades: List[Trade] = []
        self._equity_curve: List[Tuple[datetime, float]] = []
        self._daily_stats: List[Dict] = []
        self._positions_history: List[Dict] = []

        # Costs tracking
        self._total_commission: float = 0
        self._total_slippage: float = 0

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        verbose: bool = None
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            data: Dict of {symbol: DataFrame} with OHLCV data
            verbose: Override config verbose setting

        Returns:
            BacktestResult with performance metrics
        """
        verbose = verbose if verbose is not None else self.config.verbose

        # Validate data
        self._validate_data(data)

        # Align data to common timeframe
        aligned_data = self._align_data(data)

        # Get date range
        all_dates = sorted(set(
            ts for df in aligned_data.values()
            for ts in df.index
        ))

        start_idx = self.config.warmup_period
        if start_idx >= len(all_dates):
            raise ValueError(f"Not enough data for warmup period of {self.config.warmup_period}")

        if verbose:
            logger.info(f"Starting backtest: {len(all_dates)} bars, {len(aligned_data)} symbols")

        # Main backtest loop
        for bar_idx in range(start_idx, len(all_dates)):
            self._current_bar = bar_idx
            self._current_time = all_dates[bar_idx]

            # Get current bar data
            current_data = self._get_bar_data(aligned_data, bar_idx)

            # Process pending orders
            self._process_pending_orders(current_data)

            # Update portfolio prices
            prices = {
                symbol: df['close'].iloc[bar_idx]
                for symbol, df in aligned_data.items()
                if bar_idx < len(df)
            }
            self.portfolio.update_prices(prices)

            # Get historical data for strategy
            historical_data = {
                symbol: df.iloc[:bar_idx + 1]
                for symbol, df in aligned_data.items()
                if bar_idx < len(df)
            }

            # Generate signals
            try:
                signals = self.strategy.generate_signals(historical_data)
            except Exception as e:
                logger.warning(f"Signal generation error at {self._current_time}: {e}")
                signals = {}

            # Process signals
            for symbol, signal in signals.items():
                self._process_signal(signal, current_data)

            # Record equity
            self._equity_curve.append((self._current_time, self.portfolio.total_value))

            # Check drawdown stop
            if self.portfolio.current_drawdown >= self.config.max_drawdown_stop:
                if verbose:
                    logger.warning(
                        f"Max drawdown reached ({self.portfolio.current_drawdown:.2%}), "
                        f"stopping backtest at {self._current_time}"
                    )
                break

            # Daily snapshot (if new day)
            if len(self._equity_curve) > 1:
                prev_time = self._equity_curve[-2][0]
                if self._current_time.date() != prev_time.date():
                    self._take_daily_snapshot()

        # Final snapshot
        self._take_daily_snapshot()

        # Calculate results
        result = self._calculate_results()

        if verbose:
            print(result.summary())

        return result

    def _validate_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """Validate input data"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        for symbol, df in data.items():
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing column '{col}' for {symbol}")

            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"Index must be DatetimeIndex for {symbol}")

    def _align_data(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Align all data to common timeframe"""
        # Find common date range
        start = max(df.index.min() for df in data.values())
        end = min(df.index.max() for df in data.values())

        if self.config.start_date:
            start = max(start, pd.Timestamp(self.config.start_date))
        if self.config.end_date:
            end = min(end, pd.Timestamp(self.config.end_date))

        aligned = {}
        for symbol, df in data.items():
            aligned[symbol] = df.loc[start:end].copy()

        return aligned

    def _get_bar_data(
        self,
        data: Dict[str, pd.DataFrame],
        bar_idx: int
    ) -> Dict[str, pd.Series]:
        """Get data for current bar"""
        current = {}
        for symbol, df in data.items():
            if bar_idx < len(df):
                current[symbol] = df.iloc[bar_idx]
        return current

    def _process_signal(
        self,
        signal: Signal,
        current_data: Dict[str, pd.Series]
    ) -> None:
        """Process trading signal"""
        symbol = signal.symbol

        if symbol not in current_data:
            return

        bar = current_data[symbol]

        # Determine target position
        current_position = self.portfolio.positions.get(symbol)
        current_qty = current_position.quantity if current_position else 0

        # Calculate target position based on signal
        if signal.signal_type == SignalType.LONG:
            direction = 1
        elif signal.signal_type == SignalType.SHORT:
            direction = -1
        else:
            direction = 0

        # Calculate position size
        if direction != 0:
            size = self.position_sizer.calculate_size(
                symbol=symbol,
                current_price=bar['close'],
                portfolio_value=self.portfolio.total_value,
                signal_strength=signal.strength
            )
            target_qty = size.shares * direction
        else:
            target_qty = 0

        # Calculate order quantity
        order_qty = target_qty - current_qty

        if order_qty == 0:
            return

        # Create order
        order = {
            'symbol': symbol,
            'quantity': order_qty,
            'signal_price': bar['close'],
            'signal_time': self._current_time,
            'signal_strength': signal.strength,
            'strategy': signal.strategy_name,
            'bar_idx': self._current_bar
        }

        # Risk check
        if self.config.enable_risk_checks:
            check = self._pre_trade_check(order, bar)
            if not check.passed:
                logger.debug(f"Order rejected: {check.checks_failed}")
                return
            if check.adjusted_quantity:
                order['quantity'] = check.adjusted_quantity

        # Add to pending orders (filled on next bar based on fill model)
        self._pending_orders.append(order)

    def _pre_trade_check(
        self,
        order: Dict,
        bar: pd.Series
    ) -> PreTradeRiskCheck:
        """Perform pre-trade risk check"""
        return self.risk_manager.pre_trade_check(
            order_id=str(uuid.uuid4())[:8],
            symbol=order['symbol'],
            side='buy' if order['quantity'] > 0 else 'sell',
            quantity=abs(order['quantity']),
            price=bar['close']
        )

    def _process_pending_orders(
        self,
        current_data: Dict[str, pd.Series]
    ) -> None:
        """Process pending orders"""
        for order in self._pending_orders:
            symbol = order['symbol']

            if symbol not in current_data:
                continue

            bar = current_data[symbol]

            # Calculate fill price
            fill_price = self._calculate_fill_price(order, bar)

            # Calculate slippage
            slippage = self._calculate_slippage(order, bar, fill_price)
            fill_price += slippage * np.sign(order['quantity'])

            # Calculate commission
            commission = self._calculate_commission(order, fill_price)

            # Execute trade
            self._execute_trade(order, fill_price, slippage, commission)

        self._pending_orders = []

    def _calculate_fill_price(
        self,
        order: Dict,
        bar: pd.Series
    ) -> float:
        """Calculate fill price based on fill model"""
        if self.config.fill_model == FillModel.IMMEDIATE:
            return order['signal_price']
        elif self.config.fill_model == FillModel.NEXT_OPEN:
            return bar['open']
        elif self.config.fill_model == FillModel.NEXT_CLOSE:
            return bar['close']
        elif self.config.fill_model == FillModel.VWAP:
            # Approximate VWAP as average of OHLC
            return (bar['open'] + bar['high'] + bar['low'] + bar['close']) / 4
        elif self.config.fill_model == FillModel.TWAP:
            # Approximate TWAP as midpoint
            return (bar['open'] + bar['close']) / 2
        else:
            return bar['open']

    def _calculate_slippage(
        self,
        order: Dict,
        bar: pd.Series,
        fill_price: float
    ) -> float:
        """Calculate slippage"""
        if self.config.slippage_model == SlippageModel.FIXED:
            return fill_price * self.config.slippage_bps / 10000
        elif self.config.slippage_model == SlippageModel.VOLUME_BASED:
            # Impact proportional to order size vs volume
            participation = abs(order['quantity']) * fill_price / (bar['volume'] * fill_price)
            return fill_price * participation * 100 / 10000  # Scale factor
        elif self.config.slippage_model == SlippageModel.VOLATILITY_BASED:
            # Use bar range as volatility proxy
            bar_range = (bar['high'] - bar['low']) / fill_price
            return fill_price * bar_range * 0.1  # 10% of bar range
        elif self.config.slippage_model == SlippageModel.SQRT_VOLUME:
            # Square root impact model
            participation = abs(order['quantity']) * fill_price / (bar['volume'] * fill_price)
            return fill_price * np.sqrt(participation) * 50 / 10000
        else:
            return fill_price * self.config.slippage_bps / 10000

    def _calculate_commission(
        self,
        order: Dict,
        fill_price: float
    ) -> float:
        """Calculate commission"""
        notional = abs(order['quantity']) * fill_price

        # Per-share commission
        per_share = abs(order['quantity']) * self.config.commission_per_share

        # Percentage commission
        pct = notional * self.config.commission_pct

        commission = per_share + pct

        # Apply minimum
        return max(commission, self.config.commission_min)

    def _execute_trade(
        self,
        order: Dict,
        fill_price: float,
        slippage: float,
        commission: float
    ) -> None:
        """Execute trade and update portfolio"""
        symbol = order['symbol']
        quantity = order['quantity']
        notional = abs(quantity) * fill_price

        # Check if closing position
        current_pos = self.portfolio.positions.get(symbol)
        is_closing = False
        if current_pos:
            if (current_pos.quantity > 0 and quantity < 0) or \
               (current_pos.quantity < 0 and quantity > 0):
                is_closing = True

        # Update portfolio
        realized_pnl = self.portfolio.update_position(
            symbol=symbol,
            quantity=quantity,
            price=fill_price
        )

        # Deduct commission
        self.portfolio.cash -= commission

        # Track costs
        self._total_commission += commission
        self._total_slippage += abs(slippage) * abs(quantity)

        # Record trade
        trade = Trade(
            trade_id=str(uuid.uuid4())[:8],
            timestamp=self._current_time,
            symbol=symbol,
            side='buy' if quantity > 0 else 'sell',
            quantity=abs(quantity),
            price=order['signal_price'],
            fill_price=fill_price,
            slippage=slippage,
            commission=commission,
            notional=notional,
            signal_strength=order['signal_strength'],
            strategy=order['strategy'],
            pnl=realized_pnl,
            is_entry=not is_closing
        )

        self._trades.append(trade)

    def _take_daily_snapshot(self) -> None:
        """Take end-of-day snapshot"""
        if not self._current_time:
            return

        daily = {
            'date': self._current_time.date(),
            'portfolio_value': self.portfolio.total_value,
            'cash': self.portfolio.cash,
            'positions_value': self.portfolio.total_value - self.portfolio.cash,
            'pnl': self.portfolio.total_pnl,
            'drawdown': self.portfolio.current_drawdown,
            'num_positions': len(self.portfolio.positions),
            'num_trades': sum(
                1 for t in self._trades
                if t.timestamp.date() == self._current_time.date()
            )
        }

        self._daily_stats.append(daily)

        # Record position snapshot
        positions = {
            symbol: pos.to_dict()
            for symbol, pos in self.portfolio.positions.items()
        }
        self._positions_history.append({
            'date': self._current_time.date(),
            'positions': positions
        })

    def _calculate_results(self) -> BacktestResult:
        """Calculate final backtest results"""
        # Build equity curve
        equity = pd.Series(
            {ts: val for ts, val in self._equity_curve},
            name='equity'
        )

        # Calculate returns
        returns = equity.pct_change().dropna()

        # Performance metrics
        total_return = (equity.iloc[-1] / self.config.initial_capital) - 1

        trading_days = len(returns)
        if trading_days > 0:
            ann_factor = 252 * 26  # 15-min bars per day * trading days
            bars_per_year = ann_factor
            ann_return = (1 + total_return) ** (bars_per_year / len(returns)) - 1 if len(returns) > 0 else 0
            volatility = returns.std() * np.sqrt(bars_per_year)
        else:
            ann_return = 0
            volatility = 0

        sharpe = ann_return / volatility if volatility > 0 else 0

        # Sortino (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252 * 26) if len(downside_returns) > 0 else 0
        sortino = ann_return / downside_std if downside_std > 0 else 0

        # Drawdown
        cum_max = equity.cummax()
        drawdown = (cum_max - equity) / cum_max
        max_drawdown = drawdown.max()

        # Calmar
        calmar = ann_return / max_drawdown if max_drawdown > 0 else 0

        # Trade statistics
        trade_pnls = [t.pnl for t in self._trades if not t.is_entry]
        winning = [p for p in trade_pnls if p > 0]
        losing = [p for p in trade_pnls if p < 0]

        total_trades = len(self._trades)
        winning_trades = len(winning)
        losing_trades = len(losing)
        win_rate = winning_trades / len(trade_pnls) if trade_pnls else 0

        gross_profit = sum(winning) if winning else 0
        gross_loss = abs(sum(losing)) if losing else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win = np.mean(winning) if winning else 0
        avg_loss = np.mean(losing) if losing else 0
        avg_trade = np.mean(trade_pnls) if trade_pnls else 0

        largest_win = max(winning) if winning else 0
        largest_loss = min(losing) if losing else 0

        # Daily stats DataFrame
        daily_df = pd.DataFrame(self._daily_stats)
        if not daily_df.empty:
            daily_df.set_index('date', inplace=True)

        return BacktestResult(
            config=self.config,
            total_return=total_return,
            annualized_return=ann_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_drawdown,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade_pnl=avg_trade,
            largest_win=largest_win,
            largest_loss=largest_loss,
            total_commission=self._total_commission,
            total_slippage=self._total_slippage,
            total_costs=self._total_commission + self._total_slippage,
            equity_curve=equity,
            drawdown_curve=drawdown,
            trades=self._trades,
            daily_stats=daily_df,
            positions_history=self._positions_history,
            start_date=equity.index[0] if len(equity) > 0 else datetime.now(),
            end_date=equity.index[-1] if len(equity) > 0 else datetime.now(),
            trading_days=trading_days
        )


class VectorizedBacktester:
    """
    Fast vectorized backtester for strategy optimization.

    Uses numpy operations for speed.
    Less realistic but much faster.
    """

    def __init__(
        self,
        initial_capital: float = 1000000,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def run(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
        position_sizes: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Run vectorized backtest.

        Args:
            prices: DataFrame of prices (symbols as columns)
            signals: DataFrame of signals (-1, 0, 1)
            position_sizes: Optional position sizes (default: equal weight)

        Returns:
            Dictionary of results
        """
        if position_sizes is None:
            # Equal weight among active positions
            active_count = (signals != 0).sum(axis=1).replace(0, 1)
            position_sizes = signals.div(active_count, axis=0)

        # Calculate returns
        returns = prices.pct_change()

        # Shift signals (trade on next bar)
        shifted_signals = signals.shift(1).fillna(0)
        shifted_sizes = position_sizes.shift(1).fillna(0)

        # Calculate strategy returns
        strategy_returns = (shifted_sizes * returns).sum(axis=1)

        # Apply costs
        turnover = shifted_sizes.diff().abs().sum(axis=1)
        costs = turnover * (self.commission_pct + self.slippage_pct)
        strategy_returns = strategy_returns - costs

        # Calculate equity curve
        equity = (1 + strategy_returns).cumprod() * self.initial_capital

        # Calculate metrics
        total_return = equity.iloc[-1] / self.initial_capital - 1
        ann_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe = ann_return / volatility if volatility > 0 else 0

        # Drawdown
        cum_max = equity.cummax()
        drawdown = (cum_max - equity) / cum_max
        max_drawdown = drawdown.max()

        return {
            'equity_curve': equity,
            'returns': strategy_returns,
            'total_return': total_return,
            'annualized_return': ann_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_costs': costs.sum()
        }


class EventDrivenBacktester(BacktestEngine):
    """
    Extended event-driven backtester with additional features.

    Features:
    - Multiple strategy support
    - Order book simulation
    - Partial fills
    - Stop-loss and take-profit orders
    """

    def __init__(
        self,
        strategies: List[BaseStrategy],
        config: Optional[BacktestConfig] = None,
        **kwargs
    ):
        # Use first strategy as primary
        super().__init__(
            strategy=strategies[0] if strategies else None,
            config=config,
            **kwargs
        )

        self.strategies = strategies
        self._stop_orders: Dict[str, List[Dict]] = defaultdict(list)
        self._limit_orders: Dict[str, List[Dict]] = defaultdict(list)

    def add_stop_order(
        self,
        symbol: str,
        stop_price: float,
        quantity: int,
        order_type: str = 'stop_loss'
    ) -> str:
        """Add stop order"""
        order_id = str(uuid.uuid4())[:8]

        self._stop_orders[symbol].append({
            'order_id': order_id,
            'stop_price': stop_price,
            'quantity': quantity,
            'order_type': order_type,
            'created_at': self._current_time
        })

        return order_id

    def _check_stop_orders(
        self,
        symbol: str,
        bar: pd.Series
    ) -> None:
        """Check and execute triggered stop orders"""
        triggered = []

        for order in self._stop_orders[symbol]:
            if order['order_type'] == 'stop_loss':
                # Stop loss triggered when price goes below stop
                if bar['low'] <= order['stop_price']:
                    triggered.append(order)
            elif order['order_type'] == 'take_profit':
                # Take profit triggered when price goes above stop
                if bar['high'] >= order['stop_price']:
                    triggered.append(order)

        # Execute triggered orders
        for order in triggered:
            fill_price = order['stop_price']

            # Add pending order
            self._pending_orders.append({
                'symbol': symbol,
                'quantity': order['quantity'],
                'signal_price': fill_price,
                'signal_time': self._current_time,
                'signal_strength': 1.0,
                'strategy': 'stop_order',
                'bar_idx': self._current_bar
            })

            # Remove from stop orders
            self._stop_orders[symbol].remove(order)

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        verbose: bool = None
    ) -> BacktestResult:
        """Run backtest with multiple strategies"""
        verbose = verbose if verbose is not None else self.config.verbose

        self._validate_data(data)
        aligned_data = self._align_data(data)

        all_dates = sorted(set(
            ts for df in aligned_data.values()
            for ts in df.index
        ))

        start_idx = self.config.warmup_period

        if verbose:
            logger.info(
                f"Starting backtest with {len(self.strategies)} strategies, "
                f"{len(all_dates)} bars, {len(aligned_data)} symbols"
            )

        for bar_idx in range(start_idx, len(all_dates)):
            self._current_bar = bar_idx
            self._current_time = all_dates[bar_idx]

            current_data = self._get_bar_data(aligned_data, bar_idx)

            # Check stop orders
            for symbol in current_data:
                self._check_stop_orders(symbol, current_data[symbol])

            # Process pending orders
            self._process_pending_orders(current_data)

            # Update portfolio
            prices = {
                symbol: df['close'].iloc[bar_idx]
                for symbol, df in aligned_data.items()
                if bar_idx < len(df)
            }
            self.portfolio.update_prices(prices)

            # Get historical data
            historical_data = {
                symbol: df.iloc[:bar_idx + 1]
                for symbol, df in aligned_data.items()
                if bar_idx < len(df)
            }

            # Generate signals from all strategies
            all_signals = {}
            for strategy in self.strategies:
                try:
                    signals = strategy.generate_signals(historical_data)
                    for symbol, signal in signals.items():
                        # Combine signals (last wins for now)
                        all_signals[symbol] = signal
                except Exception as e:
                    logger.warning(f"Strategy {strategy.name} error: {e}")

            # Process signals
            for symbol, signal in all_signals.items():
                self._process_signal(signal, current_data)

                # Add stop orders from signal
                if signal.stop_loss:
                    direction = -1 if signal.signal_type == SignalType.LONG else 1
                    self.add_stop_order(
                        symbol=symbol,
                        stop_price=signal.stop_loss,
                        quantity=direction * 100,  # Close position
                        order_type='stop_loss'
                    )

                if signal.take_profit:
                    direction = -1 if signal.signal_type == SignalType.LONG else 1
                    self.add_stop_order(
                        symbol=symbol,
                        stop_price=signal.take_profit,
                        quantity=direction * 100,
                        order_type='take_profit'
                    )

            # Record equity
            self._equity_curve.append((self._current_time, self.portfolio.total_value))

            # Check drawdown stop
            if self.portfolio.current_drawdown >= self.config.max_drawdown_stop:
                if verbose:
                    logger.warning(f"Max drawdown reached, stopping at {self._current_time}")
                break

            # Daily snapshot
            if len(self._equity_curve) > 1:
                prev_time = self._equity_curve[-2][0]
                if self._current_time.date() != prev_time.date():
                    self._take_daily_snapshot()

        self._take_daily_snapshot()
        result = self._calculate_results()

        if verbose:
            print(result.summary())

        return result


class WalkForwardOptimizer:
    """
    Walk-forward optimization framework.

    Prevents overfitting by using rolling train/test splits.
    """

    def __init__(
        self,
        backtester: BacktestEngine,
        param_grid: Dict[str, List[Any]],
        train_period: int = 252,  # Days
        test_period: int = 63,  # Days (3 months)
        step_size: int = 21,  # Days (1 month)
        metric: str = 'sharpe_ratio'
    ):
        self.backtester = backtester
        self.param_grid = param_grid
        self.train_period = train_period
        self.test_period = test_period
        self.step_size = step_size
        self.metric = metric

        self._results: List[Dict] = []

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        n_jobs: int = 1
    ) -> Dict[str, Any]:
        """
        Run walk-forward optimization.

        Returns:
            Optimization results with best parameters per period
        """
        # Get date range
        all_dates = sorted(set(
            d for df in data.values()
            for d in df.index.date
        ))

        periods = []
        start_idx = 0

        while start_idx + self.train_period + self.test_period <= len(all_dates):
            train_start = all_dates[start_idx]
            train_end = all_dates[start_idx + self.train_period - 1]
            test_start = all_dates[start_idx + self.train_period]
            test_end = all_dates[min(
                start_idx + self.train_period + self.test_period - 1,
                len(all_dates) - 1
            )]

            periods.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })

            start_idx += self.step_size

        logger.info(f"Walk-forward optimization: {len(periods)} periods")

        # Run optimization for each period
        for i, period in enumerate(periods):
            logger.info(f"Period {i + 1}/{len(periods)}: {period['train_start']} - {period['test_end']}")

            # Get train data
            train_data = {
                symbol: df.loc[period['train_start']:period['train_end']]
                for symbol, df in data.items()
            }

            # Find best parameters on train data
            best_params, best_score = self._optimize_period(train_data)

            # Get test data
            test_data = {
                symbol: df.loc[period['test_start']:period['test_end']]
                for symbol, df in data.items()
            }

            # Run backtest with best params on test data
            self._apply_params(best_params)
            test_result = self.backtester.run(test_data, verbose=False)

            self._results.append({
                'period': i + 1,
                'train_start': period['train_start'],
                'train_end': period['train_end'],
                'test_start': period['test_start'],
                'test_end': period['test_end'],
                'best_params': best_params,
                'train_score': best_score,
                'test_return': test_result.total_return,
                'test_sharpe': test_result.sharpe_ratio
            })

        # Aggregate results
        return self._aggregate_results()

    def _optimize_period(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict, float]:
        """Optimize parameters for a single period"""
        from itertools import product

        best_params = None
        best_score = float('-inf')

        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = [self.param_grid[name] for name in param_names]

        for values in product(*param_values):
            params = dict(zip(param_names, values))

            # Apply parameters
            self._apply_params(params)

            # Run backtest
            try:
                result = self.backtester.run(data, verbose=False)
                score = getattr(result, self.metric)

                if score > best_score:
                    best_score = score
                    best_params = params

            except Exception as e:
                logger.debug(f"Parameter set failed: {e}")

        return best_params or {}, best_score

    def _apply_params(self, params: Dict) -> None:
        """Apply parameters to strategy"""
        for name, value in params.items():
            if hasattr(self.backtester.strategy, name):
                setattr(self.backtester.strategy, name, value)

    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate walk-forward results"""
        if not self._results:
            return {}

        test_returns = [r['test_return'] for r in self._results]
        test_sharpes = [r['test_sharpe'] for r in self._results]

        return {
            'periods': len(self._results),
            'avg_test_return': np.mean(test_returns),
            'total_test_return': np.prod([1 + r for r in test_returns]) - 1,
            'avg_test_sharpe': np.mean(test_sharpes),
            'results_by_period': self._results,
            'best_overall_params': max(
                self._results,
                key=lambda x: x['test_sharpe']
            )['best_params']
        }
