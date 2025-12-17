"""
Core backtesting engine for AlphaTrade system.

This module provides:
- Vectorized backtesting engine
- Transaction cost modeling
- Fill simulation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from loguru import logger

from src.strategies.base import BaseStrategy
from src.risk.drawdown import DrawdownController
from src.portfolio.allocation import AssetAllocator


@dataclass
class Trade:
    """Individual trade record."""

    timestamp: pd.Timestamp
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: float
    price: float
    value: float
    commission: float
    slippage: float
    total_cost: float


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: list[Trade]
    metrics: dict[str, float]
    signals: pd.DataFrame
    drawdown: pd.Series
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float
    final_capital: float
    metadata: dict[str, Any] = field(default_factory=dict)


class TransactionCostModel:
    """
    Transaction cost modeling.

    Includes:
    - Commission
    - Slippage
    - Market impact
    """

    def __init__(
        self,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        min_commission: float = 1.0,
        market_impact_power: float = 0.5,
    ) -> None:
        """
        Initialize cost model.

        Args:
            commission_pct: Commission as percentage
            slippage_pct: Slippage as percentage
            min_commission: Minimum commission per trade
            market_impact_power: Power for market impact model
        """
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.min_commission = min_commission
        self.market_impact_power = market_impact_power

    def calculate_costs(
        self,
        trade_value: float,
        price: float,
        avg_volume: float | None = None,
    ) -> tuple[float, float, float]:
        """
        Calculate transaction costs.

        Args:
            trade_value: Absolute value of trade
            price: Execution price
            avg_volume: Average daily volume (for market impact)

        Returns:
            Tuple of (commission, slippage, total_cost)
        """
        # Commission
        commission = max(
            trade_value * self.commission_pct,
            self.min_commission,
        )

        # Slippage
        slippage = trade_value * self.slippage_pct

        # Market impact (if volume provided)
        if avg_volume and avg_volume > 0:
            participation = trade_value / (avg_volume * price)
            impact = trade_value * 0.001 * (participation ** self.market_impact_power)
            slippage += impact

        total = commission + slippage

        return commission, slippage, total


class BacktestEngine:
    """
    Vectorized backtesting engine.

    Provides high-performance backtesting with:
    - Realistic transaction costs
    - Position tracking
    - Risk management integration
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        execution_price: Literal["open", "close", "vwap"] = "open",
        risk_limits: dict | None = None,
        use_drawdown_control: bool = True,
    ) -> None:
        """
        Initialize the engine.

        Args:
            initial_capital: Starting capital
            commission_pct: Commission percentage
            slippage_pct: Slippage percentage
            execution_price: Price for trade execution
            risk_limits: Risk limit parameters
            use_drawdown_control: Enable drawdown-based controls
        """
        self.initial_capital = initial_capital
        self.execution_price = execution_price
        self.use_drawdown_control = use_drawdown_control

        self.cost_model = TransactionCostModel(
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
        )

        self.risk_limits = risk_limits or {
            "max_position": 0.05,
            "max_sector": 0.25,
            "max_leverage": 1.0,
        }

        self.allocator = AssetAllocator(
            max_position=self.risk_limits["max_position"],
            max_sector_exposure=self.risk_limits.get("max_sector", 0.25),
            max_leverage=self.risk_limits.get("max_leverage", 1.0),
        )

        self.drawdown_controller = DrawdownController() if use_drawdown_control else None

    def run(
        self,
        strategy: BaseStrategy,
        data: dict[str, pd.DataFrame],
        features: dict[str, pd.DataFrame] | None = None,
    ) -> BacktestResult:
        """
        Run backtest.

        Args:
            strategy: Trading strategy
            data: OHLCV data by symbol
            features: Pre-computed features by symbol

        Returns:
            BacktestResult with all results
        """
        logger.info(f"Starting backtest for {strategy.name}")

        # Generate signals
        signals = strategy.generate_signals(data, features)

        # Get execution prices
        prices = self._get_execution_prices(data)

        # Initialize tracking
        equity = [self.initial_capital]
        positions = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        cash = self.initial_capital
        holdings: dict[str, float] = {sym: 0.0 for sym in prices.columns}
        trades: list[Trade] = []

        # Main backtest loop
        prev_idx = None
        for i, idx in enumerate(prices.index[1:], 1):
            prev_idx = prices.index[i - 1]

            # Get current prices
            current_prices = prices.loc[idx]
            prev_prices = prices.loc[prev_idx]

            # Calculate portfolio value
            holdings_value = sum(
                holdings[sym] * current_prices.get(sym, 0)
                for sym in holdings
            )
            portfolio_value = cash + holdings_value

            # Apply drawdown control if enabled
            if self.drawdown_controller:
                dd_status = self.drawdown_controller.update(portfolio_value)
                scale_factor = dd_status["scale_factor"]
            else:
                scale_factor = 1.0

            # Get target positions from signals
            if idx in signals.index:
                target_weights = signals.loc[idx]

                # Apply constraints
                target_weights = self.allocator.apply_constraints(target_weights)

                # Apply drawdown scaling
                target_weights = target_weights * scale_factor

                # Convert to target positions
                target_positions = strategy.calculate_positions(
                    target_weights.to_frame().T,
                    current_prices.to_frame().T,
                    portfolio_value,
                ).iloc[0]

                # Execute trades
                for symbol in prices.columns:
                    target_value = target_positions.get(symbol, 0) * portfolio_value
                    current_value = holdings.get(symbol, 0) * current_prices.get(symbol, 0)
                    trade_value = target_value - current_value

                    if abs(trade_value) > 100:  # Minimum trade size
                        price = current_prices.get(symbol, 0)
                        if price > 0:
                            # Calculate costs
                            comm, slip, total_cost = self.cost_model.calculate_costs(
                                abs(trade_value), price
                            )

                            # Execute trade
                            if trade_value > 0:
                                # CRITICAL: Cash validation before buy (JPMorgan-level)
                                required_cash = trade_value + total_cost
                                if required_cash > cash:
                                    # Insufficient cash - adjust trade size or skip
                                    if cash > total_cost + 100:  # Can afford reduced position
                                        available_for_trade = cash * 0.95  # Use 95% max
                                        trade_value = available_for_trade - total_cost
                                        # Recalculate costs for adjusted trade
                                        comm, slip, total_cost = self.cost_model.calculate_costs(
                                            abs(trade_value), price
                                        )
                                        logger.debug(
                                            f"Reduced trade for {symbol}: ${trade_value:.2f} "
                                            f"(available cash: ${cash:.2f})"
                                        )
                                    else:
                                        logger.warning(
                                            f"Insufficient cash for {symbol}: need ${required_cash:.2f}, "
                                            f"have ${cash:.2f} - skipping trade"
                                        )
                                        continue  # Skip this trade

                                # Buy
                                shares = (trade_value - total_cost) / price
                                holdings[symbol] = holdings.get(symbol, 0) + shares
                                cash -= trade_value
                                side = "BUY"

                                # Validate no negative cash (safety check)
                                if cash < 0:
                                    logger.error(
                                        f"CRITICAL: Negative cash detected after {symbol} buy: ${cash:.2f}"
                                    )
                            else:
                                # Sell - validate sufficient holdings
                                shares_to_sell = abs(trade_value) / price
                                current_holdings = holdings.get(symbol, 0)

                                if shares_to_sell > current_holdings * 1.001:  # 0.1% tolerance for float
                                    # Cannot sell more than we own (no shorting without explicit flag)
                                    if current_holdings > 0:
                                        shares_to_sell = current_holdings
                                        trade_value = -shares_to_sell * price
                                        comm, slip, total_cost = self.cost_model.calculate_costs(
                                            abs(trade_value), price
                                        )
                                        logger.debug(
                                            f"Adjusted sell for {symbol}: selling all {shares_to_sell:.4f} shares"
                                        )
                                    else:
                                        continue  # Skip - nothing to sell

                                shares = shares_to_sell
                                holdings[symbol] = holdings.get(symbol, 0) - shares
                                cash += abs(trade_value) - total_cost
                                side = "SELL"

                            # Record trade
                            trades.append(Trade(
                                timestamp=idx,
                                symbol=symbol,
                                side=side,
                                quantity=shares,
                                price=price,
                                value=abs(trade_value),
                                commission=comm,
                                slippage=slip,
                                total_cost=total_cost,
                            ))

            # Record positions
            for symbol in prices.columns:
                positions.loc[idx, symbol] = holdings.get(symbol, 0)

            # Record equity
            holdings_value = sum(
                holdings[sym] * current_prices.get(sym, 0)
                for sym in holdings
            )
            equity.append(cash + holdings_value)

        # Create equity series
        equity_series = pd.Series(equity[1:], index=prices.index[1:])
        returns = equity_series.pct_change().fillna(0)

        # Calculate drawdown
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max

        # Calculate metrics
        from src.backtesting.metrics import calculate_all_metrics

        metrics = calculate_all_metrics(returns, equity_series)

        result = BacktestResult(
            equity_curve=equity_series,
            returns=returns,
            positions=positions,
            trades=trades,
            metrics=metrics,
            signals=signals,
            drawdown=drawdown,
            start_date=prices.index[0],
            end_date=prices.index[-1],
            initial_capital=self.initial_capital,
            final_capital=equity_series.iloc[-1],
            metadata={
                "strategy_name": strategy.name,
                "num_trades": len(trades),
                "symbols": list(prices.columns),
            },
        )

        logger.info(
            f"Backtest complete: Return={metrics.get('total_return', 0):.2%}, "
            f"Sharpe={metrics.get('sharpe_ratio', 0):.2f}, "
            f"MaxDD={metrics.get('max_drawdown', 0):.2%}"
        )

        return result

    def _get_execution_prices(
        self,
        data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Get execution prices based on configuration."""
        if self.execution_price == "open":
            col = "open"
        elif self.execution_price == "close":
            col = "close"
        else:
            col = "close"

        prices = pd.DataFrame({
            sym: df[col] for sym, df in data.items()
        })

        # Shift by 1 to avoid look-ahead bias (use next bar's open)
        if self.execution_price == "open":
            prices = prices.shift(-1)

        return prices.dropna(how="all")


class VectorizedBacktest:
    """
    Fully vectorized backtest for maximum speed.

    Uses numpy operations for fast backtesting.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
    ) -> None:
        """Initialize vectorized backtest."""
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def run(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> BacktestResult:
        """
        Run vectorized backtest.

        Args:
            signals: Signal DataFrame (same shape as prices)
            prices: Price DataFrame

        Returns:
            BacktestResult
        """
        # Align signals and prices
        common_idx = signals.index.intersection(prices.index)
        signals = signals.loc[common_idx]
        prices = prices.loc[common_idx]

        # Calculate returns
        asset_returns = prices.pct_change().fillna(0)

        # Position weights (shift signals by 1 for next-bar execution)
        weights = signals.shift(1).fillna(0)

        # Turnover for transaction costs
        turnover = weights.diff().abs().sum(axis=1) / 2
        costs = turnover * (self.commission_pct + self.slippage_pct)

        # Portfolio returns
        portfolio_returns = (weights * asset_returns).sum(axis=1) - costs

        # Equity curve
        equity = self.initial_capital * (1 + portfolio_returns).cumprod()

        # Drawdown
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max

        # Metrics
        from src.backtesting.metrics import calculate_all_metrics

        metrics = calculate_all_metrics(portfolio_returns, equity)

        return BacktestResult(
            equity_curve=equity,
            returns=portfolio_returns,
            positions=weights,
            trades=[],  # No individual trades in vectorized mode
            metrics=metrics,
            signals=signals,
            drawdown=drawdown,
            start_date=common_idx[0],
            end_date=common_idx[-1],
            initial_capital=self.initial_capital,
            final_capital=equity.iloc[-1],
            metadata={"mode": "vectorized"},
        )


def run_backtest(
    strategy: BaseStrategy,
    data: dict[str, pd.DataFrame],
    initial_capital: float = 1_000_000.0,
    **kwargs,
) -> BacktestResult:
    """
    Convenience function to run a backtest.

    Args:
        strategy: Trading strategy
        data: OHLCV data
        initial_capital: Starting capital
        **kwargs: Additional engine parameters

    Returns:
        BacktestResult
    """
    engine = BacktestEngine(initial_capital=initial_capital, **kwargs)
    return engine.run(strategy, data)
