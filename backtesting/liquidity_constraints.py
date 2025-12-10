"""
Liquidity-Constrained Execution Module
======================================

JPMorgan-level liquidity constraints for realistic backtest execution.

Critical Problem Solved:
Traditional backtests allow infinite size execution - you can "buy" 10,000 shares
even if only 100 shares traded in that bar. This creates "fake alpha" where
strategies appear profitable only because they assume impossible fills.

This module enforces:
1. Volume Participation Limits - Max % of bar volume you can execute
2. Remaining Order Carry-Over - Unfilled quantity carries to next bar
3. Dynamic Slippage - Larger orders have more market impact
4. ADV-Based Position Limits - Max position as % of average daily volume
5. Intraday Liquidity Patterns - Time-of-day liquidity variation

This ensures strategies only show realistic, achievable performance.

Reference: Almgren & Chriss (2000), Kissell & Malamut (2006)

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Any, Callable
import math

import numpy as np
import polars as pl
from numpy.typing import NDArray

from config.settings import get_logger
from core.types import ExecutionError

logger = get_logger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class ParticipationModel(str, Enum):
    """Volume participation constraint models."""
    FIXED = "fixed"              # Fixed percentage of bar volume
    DYNAMIC = "dynamic"          # Varies with market conditions
    ADAPTIVE = "adaptive"        # Learns from historical data


class ImpactModel(str, Enum):
    """Market impact models."""
    LINEAR = "linear"            # Impact proportional to size
    SQUARE_ROOT = "square_root"  # Almgren-Chriss sqrt model
    POWER_LAW = "power_law"      # General power law
    KYLE = "kyle"                # Kyle's lambda model


@dataclass
class LiquidityConfig:
    """Configuration for liquidity constraints."""
    # Volume participation limits
    max_participation_rate: float = 0.01    # Max 1% of bar volume
    target_participation_rate: float = 0.005  # Target 0.5% participation

    # ADV-based limits
    max_position_adv_pct: float = 0.05      # Max position = 5% of ADV
    max_order_adv_pct: float = 0.01         # Max single order = 1% of ADV
    adv_lookback_days: int = 20             # Days for ADV calculation

    # Market impact parameters
    impact_model: ImpactModel = ImpactModel.SQUARE_ROOT
    temporary_impact_coeff: float = 0.1     # Temporary impact coefficient
    permanent_impact_coeff: float = 0.05    # Permanent impact coefficient

    # Carry-over settings
    enable_order_carryover: bool = True     # Carry unfilled to next bar
    max_carryover_bars: int = 10            # Max bars to carry unfilled

    # Time-of-day adjustments
    enable_intraday_adjustment: bool = True
    opening_multiplier: float = 1.5         # Higher limits at open
    closing_multiplier: float = 1.5         # Higher limits at close
    midday_multiplier: float = 0.7          # Lower limits at lunch

    # Minimum fill requirements
    min_fill_size: float = 1.0              # Minimum share fill
    min_fill_value: float = 100.0           # Minimum dollar fill


@dataclass
class LiquidityMetrics:
    """Liquidity metrics for a symbol."""
    symbol: str
    adv: float                      # Average Daily Volume
    adv_dollars: float              # Average Daily Dollar Volume
    avg_spread_bps: float           # Average spread in bps
    avg_bar_volume: float           # Average volume per bar
    volatility: float               # Price volatility
    last_updated: datetime = field(default_factory=datetime.now)

    def get_max_order_size(self, config: LiquidityConfig) -> float:
        """Maximum order size based on ADV."""
        return self.adv * config.max_order_adv_pct

    def get_max_position_size(self, config: LiquidityConfig) -> float:
        """Maximum position size based on ADV."""
        return self.adv * config.max_position_adv_pct


@dataclass
class ExecutionResult:
    """Result of a liquidity-constrained execution."""
    order_id: str
    requested_quantity: float
    filled_quantity: float
    remaining_quantity: float
    fill_price: float
    market_impact: float           # Price impact in bps
    participation_rate: float      # Actual participation rate
    is_complete: bool
    carryover_bars: int           # Bars the order has been carried
    timestamp: datetime
    rejection_reason: str | None = None


# =============================================================================
# LIQUIDITY CALCULATOR
# =============================================================================

class LiquidityCalculator:
    """
    Calculates liquidity metrics for symbols.

    Uses historical data to estimate:
    - Average Daily Volume (ADV)
    - Average spread
    - Volume distribution across time of day
    - Volatility for impact calculations
    """

    def __init__(self, config: LiquidityConfig | None = None):
        """Initialize calculator."""
        self.config = config or LiquidityConfig()
        self._cache: dict[str, LiquidityMetrics] = {}

    def calculate_metrics(
        self,
        symbol: str,
        historical_data: pl.DataFrame,
    ) -> LiquidityMetrics:
        """
        Calculate liquidity metrics from historical data.

        Args:
            symbol: Trading symbol
            historical_data: OHLCV DataFrame

        Returns:
            LiquidityMetrics for the symbol
        """
        if len(historical_data) == 0:
            return self._default_metrics(symbol)

        # Calculate ADV
        if "timestamp" in historical_data.columns:
            daily_data = historical_data.with_columns(
                pl.col("timestamp").dt.date().alias("date")
            ).group_by("date").agg(
                pl.col("volume").sum().alias("daily_volume"),
                (pl.col("close") * pl.col("volume")).sum().alias("daily_dollar_volume"),
            )

            lookback = min(self.config.adv_lookback_days, len(daily_data))
            recent = daily_data.tail(lookback)

            adv = recent["daily_volume"].mean()
            adv_dollars = recent["daily_dollar_volume"].mean()
        else:
            adv = historical_data["volume"].mean() * 26  # Assume 15-min bars
            adv_dollars = (historical_data["close"] * historical_data["volume"]).mean() * 26

        # Average bar volume
        avg_bar_volume = historical_data["volume"].mean()

        # Estimate spread from high-low range (Roll's model approximation)
        ranges = (historical_data["high"] - historical_data["low"]) / historical_data["close"]
        avg_spread_bps = ranges.mean() * 1000  # Rough approximation

        # Volatility
        if len(historical_data) > 1:
            returns = historical_data["close"].pct_change().drop_nulls()
            volatility = returns.std() * np.sqrt(252 * 26)  # Annualized
        else:
            volatility = 0.20  # Default 20%

        metrics = LiquidityMetrics(
            symbol=symbol,
            adv=float(adv) if adv is not None else 1000000,
            adv_dollars=float(adv_dollars) if adv_dollars is not None else 100000000,
            avg_spread_bps=float(avg_spread_bps) if avg_spread_bps is not None else 5.0,
            avg_bar_volume=float(avg_bar_volume) if avg_bar_volume is not None else 10000,
            volatility=float(volatility) if volatility is not None else 0.20,
        )

        self._cache[symbol] = metrics
        return metrics

    def get_metrics(self, symbol: str) -> LiquidityMetrics | None:
        """Get cached metrics for a symbol."""
        return self._cache.get(symbol)

    def _default_metrics(self, symbol: str) -> LiquidityMetrics:
        """Return default metrics for unknown symbols."""
        return LiquidityMetrics(
            symbol=symbol,
            adv=1000000,
            adv_dollars=100000000,
            avg_spread_bps=5.0,
            avg_bar_volume=10000,
            volatility=0.20,
        )


# =============================================================================
# MARKET IMPACT CALCULATOR
# =============================================================================

class MarketImpactCalculator:
    """
    Calculates market impact of order execution.

    Implements multiple impact models:
    1. Linear: impact = k * (Q/V)
    2. Square Root (Almgren-Chriss): impact = σ * sqrt(Q/V) * f(urgency)
    3. Power Law: impact = k * (Q/V)^α
    4. Kyle: impact = λ * Q

    Market impact has two components:
    - Temporary: Price moves while executing, then reverts
    - Permanent: Lasting price change due to information content
    """

    def __init__(self, config: LiquidityConfig | None = None):
        """Initialize calculator."""
        self.config = config or LiquidityConfig()

    def calculate_impact(
        self,
        order_size: float,
        bar_volume: float,
        volatility: float,
        adv: float,
        side: str,
    ) -> tuple[float, float]:
        """
        Calculate market impact of an order.

        Args:
            order_size: Order size in shares
            bar_volume: Volume in this bar
            volatility: Annualized volatility
            adv: Average daily volume
            side: "buy" or "sell"

        Returns:
            Tuple of (temporary_impact_bps, permanent_impact_bps)
        """
        if bar_volume <= 0 or adv <= 0:
            return 0.0, 0.0

        participation = order_size / bar_volume
        daily_participation = order_size / adv

        model = self.config.impact_model

        if model == ImpactModel.LINEAR:
            temp_impact = self.config.temporary_impact_coeff * participation * volatility * 10000
            perm_impact = self.config.permanent_impact_coeff * daily_participation * volatility * 10000

        elif model == ImpactModel.SQUARE_ROOT:
            # Almgren-Chriss model
            # Temporary: η * σ * sqrt(X/V)
            # Permanent: γ * σ * X/V
            temp_impact = self.config.temporary_impact_coeff * volatility * np.sqrt(participation) * 10000
            perm_impact = self.config.permanent_impact_coeff * volatility * daily_participation * 10000

        elif model == ImpactModel.POWER_LAW:
            alpha = 0.6  # Typical power law exponent
            temp_impact = self.config.temporary_impact_coeff * (participation ** alpha) * volatility * 10000
            perm_impact = self.config.permanent_impact_coeff * (daily_participation ** alpha) * volatility * 10000

        elif model == ImpactModel.KYLE:
            # Kyle's lambda: impact = λ * Q
            # λ estimated from volatility and volume
            kyle_lambda = volatility / np.sqrt(adv)
            temp_impact = kyle_lambda * order_size / 100  # Scale appropriately
            perm_impact = temp_impact * 0.5  # Assume 50% is permanent

        else:
            temp_impact = 0.0
            perm_impact = 0.0

        return temp_impact, perm_impact

    def get_execution_price(
        self,
        base_price: float,
        order_size: float,
        bar_volume: float,
        volatility: float,
        adv: float,
        side: str,
    ) -> float:
        """
        Get execution price including market impact.

        Args:
            base_price: Mid-price before execution
            order_size: Order size
            bar_volume: Bar volume
            volatility: Volatility
            adv: Average daily volume
            side: "buy" or "sell"

        Returns:
            Execution price after impact
        """
        temp_impact, perm_impact = self.calculate_impact(
            order_size, bar_volume, volatility, adv, side
        )

        total_impact_pct = (temp_impact + perm_impact) / 10000

        if side == "buy":
            return base_price * (1 + total_impact_pct)
        else:
            return base_price * (1 - total_impact_pct)


# =============================================================================
# LIQUIDITY CONSTRAINED EXECUTOR
# =============================================================================

class LiquidityConstrainedExecutor:
    """
    Enforces liquidity constraints on order execution.

    Key constraints:
    1. Max participation rate (e.g., 1% of bar volume)
    2. ADV-based position limits
    3. Carry-over of unfilled orders
    4. Dynamic market impact
    """

    def __init__(self, config: LiquidityConfig | None = None):
        """Initialize executor."""
        self.config = config or LiquidityConfig()
        self.liquidity_calc = LiquidityCalculator(self.config)
        self.impact_calc = MarketImpactCalculator(self.config)

        # Pending orders (carry-over)
        self.pending_orders: dict[str, dict[str, Any]] = {}

    def initialize_symbol(
        self,
        symbol: str,
        historical_data: pl.DataFrame,
    ) -> LiquidityMetrics:
        """
        Initialize liquidity metrics for a symbol.

        Must be called before executing orders for a symbol.
        """
        return self.liquidity_calc.calculate_metrics(symbol, historical_data)

    def execute_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        bar_volume: float,
        bar_timestamp: datetime,
    ) -> ExecutionResult:
        """
        Execute an order with liquidity constraints.

        The order may be partially filled based on volume constraints.
        Unfilled quantity is carried over to subsequent bars.

        Args:
            order_id: Unique order ID
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Requested quantity
            price: Execution price (before impact)
            bar_volume: Volume in this bar
            bar_timestamp: Current bar timestamp

        Returns:
            ExecutionResult with fill details
        """
        metrics = self.liquidity_calc.get_metrics(symbol)
        if metrics is None:
            # Use default metrics
            metrics = self.liquidity_calc._default_metrics(symbol)

        # Check if this is a carry-over order
        if order_id in self.pending_orders:
            pending = self.pending_orders[order_id]
            quantity = pending["remaining"]
            carryover_bars = pending["bars"] + 1
        else:
            carryover_bars = 0

        # CONSTRAINT 1: Volume participation limit
        max_fill = bar_volume * self.config.max_participation_rate

        # CONSTRAINT 2: ADV-based order limit
        adv_limit = metrics.adv * self.config.max_order_adv_pct
        max_fill = min(max_fill, adv_limit)

        # Apply time-of-day adjustment
        max_fill = self._apply_intraday_adjustment(max_fill, bar_timestamp)

        # CONSTRAINT 3: Minimum fill size
        if max_fill < self.config.min_fill_size:
            max_fill = 0

        # Determine actual fill
        if max_fill <= 0:
            # Cannot fill anything this bar
            filled_quantity = 0
            fill_price = price
            market_impact = 0
        else:
            filled_quantity = min(quantity, max_fill)

            # Calculate market impact
            temp_impact, perm_impact = self.impact_calc.calculate_impact(
                filled_quantity, bar_volume, metrics.volatility, metrics.adv, side
            )
            market_impact = temp_impact + perm_impact

            # Get execution price
            fill_price = self.impact_calc.get_execution_price(
                price, filled_quantity, bar_volume, metrics.volatility, metrics.adv, side
            )

        remaining_quantity = quantity - filled_quantity
        is_complete = remaining_quantity <= 0

        # Handle carry-over
        if not is_complete and self.config.enable_order_carryover:
            if carryover_bars < self.config.max_carryover_bars:
                self.pending_orders[order_id] = {
                    "symbol": symbol,
                    "side": side,
                    "remaining": remaining_quantity,
                    "original_quantity": quantity if carryover_bars == 0 else pending.get("original_quantity", quantity),
                    "bars": carryover_bars,
                }
            else:
                # Max carry-over reached - cancel remaining
                logger.warning(f"Order {order_id} cancelled after {carryover_bars} bars of carry-over")
                if order_id in self.pending_orders:
                    del self.pending_orders[order_id]
        elif is_complete and order_id in self.pending_orders:
            del self.pending_orders[order_id]

        # Calculate participation rate
        participation_rate = filled_quantity / bar_volume if bar_volume > 0 else 0

        return ExecutionResult(
            order_id=order_id,
            requested_quantity=quantity,
            filled_quantity=filled_quantity,
            remaining_quantity=remaining_quantity,
            fill_price=fill_price,
            market_impact=market_impact,
            participation_rate=participation_rate,
            is_complete=is_complete,
            carryover_bars=carryover_bars,
            timestamp=bar_timestamp,
        )

    def _apply_intraday_adjustment(
        self,
        base_limit: float,
        timestamp: datetime,
    ) -> float:
        """Apply time-of-day liquidity adjustment."""
        if not self.config.enable_intraday_adjustment:
            return base_limit

        hour = timestamp.hour
        minute = timestamp.minute
        time_decimal = hour + minute / 60

        # Market hours: 9:30 - 16:00 ET
        if time_decimal < 9.5 or time_decimal >= 16:
            return 0  # Outside market hours

        # First 30 minutes: opening
        if time_decimal < 10:
            return base_limit * self.config.opening_multiplier

        # Last 30 minutes: closing
        if time_decimal >= 15.5:
            return base_limit * self.config.closing_multiplier

        # Lunch hours: 12-13
        if 12 <= time_decimal < 13:
            return base_limit * self.config.midday_multiplier

        # Normal hours
        return base_limit

    def check_position_limit(
        self,
        symbol: str,
        current_position: float,
        proposed_order_size: float,
    ) -> tuple[bool, float]:
        """
        Check if proposed order exceeds position limits.

        Args:
            symbol: Trading symbol
            current_position: Current position size
            proposed_order_size: Proposed order size (signed)

        Returns:
            Tuple of (is_allowed, max_allowed_size)
        """
        metrics = self.liquidity_calc.get_metrics(symbol)
        if metrics is None:
            return True, proposed_order_size

        max_position = metrics.get_max_position_size(self.config)
        new_position = current_position + proposed_order_size

        if abs(new_position) <= max_position:
            return True, proposed_order_size

        # Calculate maximum allowed order
        if proposed_order_size > 0:
            max_allowed = max_position - current_position
        else:
            max_allowed = -max_position - current_position

        return False, max_allowed

    def get_pending_orders(self) -> dict[str, dict[str, Any]]:
        """Get all pending (carry-over) orders."""
        return self.pending_orders.copy()

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
            return True
        return False


# =============================================================================
# BACKTEST ENGINE INTEGRATION
# =============================================================================

class LiquidityAwareBacktestEngine:
    """
    Backtest engine integration for liquidity-constrained execution.

    Wraps the standard backtest engine to enforce realistic execution.
    """

    def __init__(
        self,
        config: LiquidityConfig | None = None,
    ):
        """Initialize liquidity-aware engine."""
        self.config = config or LiquidityConfig()
        self.executor = LiquidityConstrainedExecutor(self.config)

        # Tracking
        self.execution_history: list[ExecutionResult] = []
        self.position_tracker: dict[str, float] = {}

        # Statistics
        self.stats = {
            "total_orders": 0,
            "fully_filled": 0,
            "partially_filled": 0,
            "cancelled_carryover": 0,
            "rejected_liquidity": 0,
            "total_market_impact_bps": 0.0,
            "avg_participation_rate": 0.0,
            "avg_carryover_bars": 0.0,
        }

    def initialize(
        self,
        symbols: list[str],
        historical_data: dict[str, pl.DataFrame],
    ) -> None:
        """
        Initialize the engine with historical data for liquidity calculation.

        Args:
            symbols: List of symbols to trade
            historical_data: Dict of symbol -> OHLCV DataFrame
        """
        for symbol in symbols:
            if symbol in historical_data:
                self.executor.initialize_symbol(symbol, historical_data[symbol])
            else:
                logger.warning(f"No historical data for {symbol}, using defaults")

    def execute(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        bar_data: dict[str, float],
        timestamp: datetime,
    ) -> ExecutionResult:
        """
        Execute an order with full liquidity constraints.

        Args:
            order_id: Unique order ID
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Requested quantity
            price: Base execution price
            bar_data: Current bar OHLCV
            timestamp: Current timestamp

        Returns:
            ExecutionResult with fill details
        """
        self.stats["total_orders"] += 1

        # Check position limits
        current_pos = self.position_tracker.get(symbol, 0)
        signed_qty = quantity if side == "buy" else -quantity

        allowed, max_qty = self.executor.check_position_limit(
            symbol, current_pos, signed_qty
        )

        if not allowed:
            logger.warning(f"Position limit reached for {symbol}")
            quantity = abs(max_qty)
            if quantity <= 0:
                self.stats["rejected_liquidity"] += 1
                return ExecutionResult(
                    order_id=order_id,
                    requested_quantity=quantity,
                    filled_quantity=0,
                    remaining_quantity=quantity,
                    fill_price=price,
                    market_impact=0,
                    participation_rate=0,
                    is_complete=False,
                    carryover_bars=0,
                    timestamp=timestamp,
                    rejection_reason="Position limit exceeded",
                )

        # Execute with liquidity constraints
        result = self.executor.execute_order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            bar_volume=bar_data.get("volume", 0),
            bar_timestamp=timestamp,
        )

        # Update position
        if result.filled_quantity > 0:
            fill_signed = result.filled_quantity if side == "buy" else -result.filled_quantity
            self.position_tracker[symbol] = current_pos + fill_signed

        # Update statistics
        if result.is_complete:
            self.stats["fully_filled"] += 1
        elif result.filled_quantity > 0:
            self.stats["partially_filled"] += 1

        self.stats["total_market_impact_bps"] += result.market_impact

        # Track execution history
        self.execution_history.append(result)

        return result

    def process_pending_orders(
        self,
        bar_data: dict[str, dict[str, float]],
        timestamp: datetime,
    ) -> list[ExecutionResult]:
        """
        Process all pending (carry-over) orders for current bar.

        Args:
            bar_data: Dict of symbol -> OHLCV dict
            timestamp: Current timestamp

        Returns:
            List of ExecutionResults for processed orders
        """
        results = []
        pending = self.executor.get_pending_orders()

        for order_id, order_info in pending.items():
            symbol = order_info["symbol"]
            if symbol not in bar_data:
                continue

            bar = bar_data[symbol]
            result = self.executor.execute_order(
                order_id=order_id,
                symbol=symbol,
                side=order_info["side"],
                quantity=order_info["remaining"],
                price=bar.get("close", 0),
                bar_volume=bar.get("volume", 0),
                bar_timestamp=timestamp,
            )

            if result.filled_quantity > 0:
                fill_signed = result.filled_quantity if order_info["side"] == "buy" else -result.filled_quantity
                current_pos = self.position_tracker.get(symbol, 0)
                self.position_tracker[symbol] = current_pos + fill_signed

            results.append(result)

        return results

    def get_statistics(self) -> dict[str, Any]:
        """Get execution statistics."""
        stats = self.stats.copy()

        if self.execution_history:
            participation_rates = [r.participation_rate for r in self.execution_history if r.filled_quantity > 0]
            carryover_bars = [r.carryover_bars for r in self.execution_history]

            if participation_rates:
                stats["avg_participation_rate"] = np.mean(participation_rates)
            if carryover_bars:
                stats["avg_carryover_bars"] = np.mean(carryover_bars)

        return stats

    def get_execution_summary(self) -> pl.DataFrame:
        """Get execution history as DataFrame."""
        if not self.execution_history:
            return pl.DataFrame()

        records = []
        for r in self.execution_history:
            records.append({
                "order_id": r.order_id,
                "timestamp": r.timestamp,
                "requested_qty": r.requested_quantity,
                "filled_qty": r.filled_quantity,
                "remaining_qty": r.remaining_quantity,
                "fill_price": r.fill_price,
                "market_impact_bps": r.market_impact,
                "participation_rate": r.participation_rate,
                "is_complete": r.is_complete,
                "carryover_bars": r.carryover_bars,
            })

        return pl.DataFrame(records)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ParticipationModel",
    "ImpactModel",
    # Configuration
    "LiquidityConfig",
    "LiquidityMetrics",
    "ExecutionResult",
    # Calculators
    "LiquidityCalculator",
    "MarketImpactCalculator",
    # Executor
    "LiquidityConstrainedExecutor",
    "LiquidityAwareBacktestEngine",
]
