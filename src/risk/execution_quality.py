"""
Execution Quality Metrics Module.

This module provides institutional-grade execution analysis including:
- Implementation Shortfall (IS)
- VWAP/TWAP deviation
- Market Impact measurement
- Timing analysis
- Fill rate analysis

JPMorgan-level requirements:
- Accurate cost attribution
- Transaction-by-transaction analysis
- Comparison to benchmarks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeExecution:
    """Single trade execution record."""
    timestamp: pd.Timestamp
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    execution_price: float
    arrival_price: float  # Price when order was placed
    benchmark_vwap: Optional[float] = None
    benchmark_twap: Optional[float] = None
    market_volume: Optional[float] = None


@dataclass
class ExecutionMetrics:
    """Execution quality metrics for a single trade."""
    symbol: str
    implementation_shortfall_bps: float
    vwap_deviation_bps: float
    market_impact_bps: float
    timing_cost_bps: float
    opportunity_cost_bps: float
    fill_rate: float


@dataclass
class AggregateExecutionReport:
    """Aggregate execution quality report."""
    period_start: str
    period_end: str
    n_trades: int
    total_volume: float
    avg_implementation_shortfall_bps: float
    avg_vwap_deviation_bps: float
    avg_market_impact_bps: float
    total_execution_cost_bps: float
    by_symbol: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_hour: Dict[int, Dict[str, float]] = field(default_factory=dict)


class ExecutionQualityAnalyzer:
    """
    Analyzes execution quality of trades.

    Implementation Shortfall decomposition:
    IS = Market Impact + Timing Cost + Opportunity Cost

    Where:
    - Market Impact: Difference between exec price and arrival price
    - Timing Cost: Slippage due to delay in execution
    - Opportunity Cost: Cost of unfilled orders
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        trading_days_per_year: int = 252,
    ):
        """
        Initialize analyzer.

        Args:
            risk_free_rate: Annual risk-free rate
            trading_days_per_year: Trading days per year
        """
        self.rf_rate = risk_free_rate
        self.trading_days = trading_days_per_year

    def calculate_implementation_shortfall(
        self,
        execution: TradeExecution,
    ) -> float:
        """
        Calculate Implementation Shortfall (IS) for a single trade.

        IS = (Execution Price - Decision Price) / Decision Price * Side

        Where Side = +1 for buy, -1 for sell

        Returns:
            IS in basis points
        """
        side_mult = 1 if execution.side == "buy" else -1

        is_raw = (execution.execution_price - execution.arrival_price) / execution.arrival_price
        is_bps = is_raw * side_mult * 10000

        return is_bps

    def calculate_vwap_deviation(
        self,
        execution: TradeExecution,
    ) -> float:
        """
        Calculate deviation from VWAP benchmark.

        Returns:
            Deviation in basis points (positive = worse than VWAP)
        """
        if execution.benchmark_vwap is None:
            return 0.0

        side_mult = 1 if execution.side == "buy" else -1
        deviation = (execution.execution_price - execution.benchmark_vwap) / execution.benchmark_vwap
        return deviation * side_mult * 10000

    def calculate_market_impact(
        self,
        execution: TradeExecution,
        market_data: Optional[pd.DataFrame] = None,
    ) -> float:
        """
        Estimate market impact of the trade.

        Uses the square-root impact model:
        Impact = sigma * sqrt(Q / ADV)

        Where:
        - sigma: Daily volatility
        - Q: Order quantity
        - ADV: Average daily volume
        """
        if market_data is None or len(market_data) < 20:
            # Fallback: use simple price impact
            side_mult = 1 if execution.side == "buy" else -1
            impact = (execution.execution_price - execution.arrival_price) / execution.arrival_price
            return impact * side_mult * 10000

        # Calculate volatility
        returns = market_data["close"].pct_change()
        sigma = returns.std() * np.sqrt(self.trading_days)

        # Calculate participation rate
        adv = market_data["volume"].mean()
        participation = execution.quantity / adv if adv > 0 else 0.01

        # Square-root impact model
        impact = sigma * np.sqrt(participation) * 10000

        return impact

    def analyze_trade(
        self,
        execution: TradeExecution,
        market_data: Optional[pd.DataFrame] = None,
    ) -> ExecutionMetrics:
        """
        Comprehensive analysis of a single trade.

        Returns:
            ExecutionMetrics with all quality measures
        """
        # Implementation Shortfall
        is_bps = self.calculate_implementation_shortfall(execution)

        # VWAP deviation
        vwap_dev = self.calculate_vwap_deviation(execution)

        # Market impact
        impact = self.calculate_market_impact(execution, market_data)

        # Timing cost (IS - Market Impact)
        timing = max(0, is_bps - impact)

        # Opportunity cost (assume fully filled for now)
        opportunity = 0.0
        fill_rate = 1.0

        return ExecutionMetrics(
            symbol=execution.symbol,
            implementation_shortfall_bps=is_bps,
            vwap_deviation_bps=vwap_dev,
            market_impact_bps=impact,
            timing_cost_bps=timing,
            opportunity_cost_bps=opportunity,
            fill_rate=fill_rate,
        )

    def generate_aggregate_report(
        self,
        executions: List[TradeExecution],
        market_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> AggregateExecutionReport:
        """
        Generate aggregate execution quality report.

        Args:
            executions: List of trade executions
            market_data: Optional market data by symbol

        Returns:
            Aggregate report
        """
        if not executions:
            return AggregateExecutionReport(
                period_start="",
                period_end="",
                n_trades=0,
                total_volume=0,
                avg_implementation_shortfall_bps=0,
                avg_vwap_deviation_bps=0,
                avg_market_impact_bps=0,
                total_execution_cost_bps=0,
            )

        # Analyze each trade
        metrics_list = []
        for exec in executions:
            mkt_data = market_data.get(exec.symbol) if market_data else None
            metrics = self.analyze_trade(exec, mkt_data)
            metrics_list.append(metrics)

        # Aggregate
        is_values = [m.implementation_shortfall_bps for m in metrics_list]
        vwap_values = [m.vwap_deviation_bps for m in metrics_list]
        impact_values = [m.market_impact_bps for m in metrics_list]

        # By symbol
        by_symbol = {}
        for m in metrics_list:
            if m.symbol not in by_symbol:
                by_symbol[m.symbol] = {
                    "count": 0,
                    "avg_is": 0,
                    "avg_vwap_dev": 0,
                }
            by_symbol[m.symbol]["count"] += 1
            by_symbol[m.symbol]["avg_is"] += m.implementation_shortfall_bps
            by_symbol[m.symbol]["avg_vwap_dev"] += m.vwap_deviation_bps

        for sym in by_symbol:
            count = by_symbol[sym]["count"]
            by_symbol[sym]["avg_is"] /= count
            by_symbol[sym]["avg_vwap_dev"] /= count

        # By hour
        by_hour = {}
        for exec, m in zip(executions, metrics_list):
            hour = exec.timestamp.hour
            if hour not in by_hour:
                by_hour[hour] = {"count": 0, "avg_is": 0}
            by_hour[hour]["count"] += 1
            by_hour[hour]["avg_is"] += m.implementation_shortfall_bps

        for hour in by_hour:
            by_hour[hour]["avg_is"] /= by_hour[hour]["count"]

        return AggregateExecutionReport(
            period_start=str(min(e.timestamp for e in executions)),
            period_end=str(max(e.timestamp for e in executions)),
            n_trades=len(executions),
            total_volume=sum(e.quantity * e.execution_price for e in executions),
            avg_implementation_shortfall_bps=np.mean(is_values),
            avg_vwap_deviation_bps=np.mean(vwap_values),
            avg_market_impact_bps=np.mean(impact_values),
            total_execution_cost_bps=np.sum(is_values),
            by_symbol=by_symbol,
            by_hour=by_hour,
        )


def analyze_execution_quality(
    trades: pd.DataFrame,
    prices: Dict[str, pd.DataFrame],
) -> AggregateExecutionReport:
    """
    Convenience function to analyze execution quality from trade DataFrame.

    Args:
        trades: DataFrame with columns [timestamp, symbol, side, quantity, price]
        prices: Market data by symbol

    Returns:
        Aggregate execution report
    """
    # Convert trades to TradeExecution objects
    executions = []
    for _, row in trades.iterrows():
        symbol = row["symbol"]
        timestamp = pd.to_datetime(row["timestamp"])

        # Get arrival price (price at order time)
        if symbol in prices:
            price_data = prices[symbol]
            arrival_idx = price_data.index.get_indexer([timestamp], method="ffill")[0]
            if arrival_idx >= 0:
                arrival_price = price_data.iloc[arrival_idx]["close"]
            else:
                arrival_price = row["price"]
        else:
            arrival_price = row["price"]

        executions.append(TradeExecution(
            timestamp=timestamp,
            symbol=symbol,
            side=row.get("side", "buy"),
            quantity=row.get("quantity", 0),
            execution_price=row["price"],
            arrival_price=arrival_price,
        ))

    analyzer = ExecutionQualityAnalyzer()
    return analyzer.generate_aggregate_report(executions, prices)
