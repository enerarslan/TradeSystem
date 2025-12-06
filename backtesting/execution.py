"""
Execution Models Module
=======================

Realistic execution modeling for backtesting.
Simulates slippage, commissions, and order fills.

Models:
- Slippage: Fixed, percentage, volume-based, market impact
- Commission: Fixed, percentage, tiered, per-share
- Fill: Immediate, partial, probabilistic, volume-weighted

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from config.settings import get_logger
from core.types import Order, OrderStatus, OHLCV

logger = get_logger(__name__)


# =============================================================================
# SLIPPAGE MODELS
# =============================================================================

class SlippageModel(ABC):
    """
    Abstract base class for slippage models.
    
    Slippage represents the difference between expected
    execution price and actual fill price.
    """
    
    @abstractmethod
    def calculate_slippage(
        self,
        order: Order,
        bar: OHLCV,
        **kwargs: Any,
    ) -> float:
        """
        Calculate slippage amount.
        
        Args:
            order: Order to execute
            bar: Current market bar
            **kwargs: Additional parameters
        
        Returns:
            Slippage amount (positive = worse price)
        """
        pass
    
    def get_fill_price(
        self,
        order: Order,
        bar: OHLCV,
        reference_price: float | None = None,
        **kwargs: Any,
    ) -> float:
        """
        Calculate fill price including slippage.
        
        Args:
            order: Order to execute
            bar: Current market bar
            reference_price: Base price (default: bar close)
            **kwargs: Additional parameters
        
        Returns:
            Fill price
        """
        base_price = reference_price or bar.close
        slippage = self.calculate_slippage(order, bar, **kwargs)
        
        # Slippage is adverse: buy orders get higher price, sell orders get lower
        if order.side.lower() == "buy":
            return base_price + slippage
        else:
            return base_price - slippage


class NoSlippage(SlippageModel):
    """No slippage model - fills at reference price."""
    
    def calculate_slippage(
        self,
        order: Order,
        bar: OHLCV,
        **kwargs: Any,
    ) -> float:
        """Zero slippage."""
        return 0.0


class FixedSlippage(SlippageModel):
    """
    Fixed slippage model.
    
    Applies a constant slippage amount per trade.
    """
    
    def __init__(self, slippage_amount: float = 0.01):
        """
        Initialize fixed slippage.
        
        Args:
            slippage_amount: Fixed slippage per share
        """
        self.slippage_amount = slippage_amount
    
    def calculate_slippage(
        self,
        order: Order,
        bar: OHLCV,
        **kwargs: Any,
    ) -> float:
        """Return fixed slippage amount."""
        return self.slippage_amount


class PercentageSlippage(SlippageModel):
    """
    Percentage-based slippage model.
    
    Slippage as percentage of price.
    """
    
    def __init__(self, slippage_pct: float = 0.001):
        """
        Initialize percentage slippage.
        
        Args:
            slippage_pct: Slippage as decimal (0.001 = 0.1%)
        """
        self.slippage_pct = slippage_pct
    
    def calculate_slippage(
        self,
        order: Order,
        bar: OHLCV,
        **kwargs: Any,
    ) -> float:
        """Calculate percentage-based slippage."""
        return bar.close * self.slippage_pct


class VolumeSlippage(SlippageModel):
    """
    Volume-based slippage model.
    
    Slippage increases with order size relative to volume.
    Models market impact based on participation rate.
    
    Formula:
        slippage = base_slippage * (1 + impact_factor * participation_rate^2)
        participation_rate = order_quantity / bar_volume
    """
    
    def __init__(
        self,
        base_slippage_pct: float = 0.0005,
        impact_factor: float = 10.0,
        max_participation: float = 0.10,
    ):
        """
        Initialize volume-based slippage.
        
        Args:
            base_slippage_pct: Base slippage percentage
            impact_factor: Market impact multiplier
            max_participation: Maximum volume participation (10% default)
        """
        self.base_slippage_pct = base_slippage_pct
        self.impact_factor = impact_factor
        self.max_participation = max_participation
    
    def calculate_slippage(
        self,
        order: Order,
        bar: OHLCV,
        **kwargs: Any,
    ) -> float:
        """Calculate volume-weighted slippage."""
        if bar.volume <= 0:
            # No volume, use high slippage
            return bar.close * self.base_slippage_pct * 5
        
        # Calculate participation rate
        participation = order.quantity / bar.volume
        participation = min(participation, self.max_participation)
        
        # Market impact increases with square of participation
        impact = 1 + self.impact_factor * (participation ** 2)
        
        return bar.close * self.base_slippage_pct * impact


class SpreadSlippage(SlippageModel):
    """
    Bid-ask spread based slippage.
    
    Models crossing the spread for market orders.
    Uses ATR as proxy for spread width.
    """
    
    def __init__(
        self,
        spread_pct: float = 0.001,
        use_atr_proxy: bool = True,
        atr_multiplier: float = 0.1,
    ):
        """
        Initialize spread-based slippage.
        
        Args:
            spread_pct: Fixed spread percentage
            use_atr_proxy: Use ATR as spread proxy
            atr_multiplier: ATR to spread multiplier
        """
        self.spread_pct = spread_pct
        self.use_atr_proxy = use_atr_proxy
        self.atr_multiplier = atr_multiplier
        self._atr_buffer: list[float] = []
    
    def calculate_slippage(
        self,
        order: Order,
        bar: OHLCV,
        **kwargs: Any,
    ) -> float:
        """Calculate spread-based slippage."""
        if self.use_atr_proxy:
            # Use bar range as volatility proxy
            bar_range = bar.high - bar.low
            spread = bar_range * self.atr_multiplier
        else:
            spread = bar.close * self.spread_pct
        
        # Cross half the spread (to mid-point assumption)
        return spread / 2


class MarketImpactSlippage(SlippageModel):
    """
    Advanced market impact model.
    
    Based on Kyle's lambda and Almgren-Chriss framework.
    Models both temporary and permanent impact.
    
    Temporary Impact: Affects execution price only
    Permanent Impact: Moves market price
    """
    
    def __init__(
        self,
        daily_volume: float = 1_000_000,
        volatility: float = 0.02,
        temporary_impact: float = 0.1,
        permanent_impact: float = 0.05,
    ):
        """
        Initialize market impact model.
        
        Args:
            daily_volume: Average daily volume
            volatility: Daily volatility
            temporary_impact: Temporary impact coefficient
            permanent_impact: Permanent impact coefficient
        """
        self.daily_volume = daily_volume
        self.volatility = volatility
        self.temporary_impact = temporary_impact
        self.permanent_impact = permanent_impact
    
    def calculate_slippage(
        self,
        order: Order,
        bar: OHLCV,
        **kwargs: Any,
    ) -> float:
        """
        Calculate market impact slippage.
        
        Uses square-root model for market impact.
        """
        # Participation rate
        volume = bar.volume if bar.volume > 0 else self.daily_volume / 390  # Minute volume
        participation = order.quantity / volume
        
        # Temporary impact: sigma * sqrt(participation)
        temp_impact = self.volatility * np.sqrt(participation) * self.temporary_impact
        
        # Total impact
        total_impact = temp_impact * bar.close
        
        return total_impact


# =============================================================================
# COMMISSION MODELS
# =============================================================================

class CommissionModel(ABC):
    """
    Abstract base class for commission models.
    
    Calculates trading costs and fees.
    """
    
    @abstractmethod
    def calculate_commission(
        self,
        order: Order,
        fill_price: float,
        fill_quantity: float,
    ) -> float:
        """
        Calculate commission for a trade.
        
        Args:
            order: Order being filled
            fill_price: Execution price
            fill_quantity: Quantity filled
        
        Returns:
            Commission amount
        """
        pass


class NoCommission(CommissionModel):
    """No commission model."""
    
    def calculate_commission(
        self,
        order: Order,
        fill_price: float,
        fill_quantity: float,
    ) -> float:
        """Zero commission."""
        return 0.0


class FixedCommission(CommissionModel):
    """
    Fixed commission per trade.
    """
    
    def __init__(self, commission_per_trade: float = 1.0):
        """
        Initialize fixed commission.
        
        Args:
            commission_per_trade: Fixed amount per trade
        """
        self.commission_per_trade = commission_per_trade
    
    def calculate_commission(
        self,
        order: Order,
        fill_price: float,
        fill_quantity: float,
    ) -> float:
        """Return fixed commission."""
        return self.commission_per_trade


class PerShareCommission(CommissionModel):
    """
    Per-share commission model.
    
    Common for equity trading (e.g., $0.005 per share).
    """
    
    def __init__(
        self,
        per_share: float = 0.005,
        min_commission: float = 1.0,
        max_commission_pct: float = 0.01,
    ):
        """
        Initialize per-share commission.
        
        Args:
            per_share: Commission per share
            min_commission: Minimum commission per trade
            max_commission_pct: Maximum as % of trade value
        """
        self.per_share = per_share
        self.min_commission = min_commission
        self.max_commission_pct = max_commission_pct
    
    def calculate_commission(
        self,
        order: Order,
        fill_price: float,
        fill_quantity: float,
    ) -> float:
        """Calculate per-share commission with min/max."""
        commission = fill_quantity * self.per_share
        
        # Apply minimum
        commission = max(commission, self.min_commission)
        
        # Apply maximum as % of trade value
        trade_value = fill_price * fill_quantity
        max_commission = trade_value * self.max_commission_pct
        commission = min(commission, max_commission)
        
        return commission


class PercentageCommission(CommissionModel):
    """
    Percentage-based commission model.
    
    Common for crypto and forex trading.
    """
    
    def __init__(
        self,
        commission_pct: float = 0.001,
        min_commission: float = 0.0,
    ):
        """
        Initialize percentage commission.
        
        Args:
            commission_pct: Commission as decimal (0.001 = 0.1%)
            min_commission: Minimum commission per trade
        """
        self.commission_pct = commission_pct
        self.min_commission = min_commission
    
    def calculate_commission(
        self,
        order: Order,
        fill_price: float,
        fill_quantity: float,
    ) -> float:
        """Calculate percentage commission."""
        trade_value = fill_price * fill_quantity
        commission = trade_value * self.commission_pct
        return max(commission, self.min_commission)


class TieredCommission(CommissionModel):
    """
    Tiered commission based on monthly volume.
    
    Lower rates for higher volume traders.
    """
    
    def __init__(
        self,
        tiers: list[tuple[float, float]] | None = None,
    ):
        """
        Initialize tiered commission.
        
        Args:
            tiers: List of (volume_threshold, commission_pct) tuples
                   Sorted by volume ascending
        """
        self.tiers = tiers or [
            (0, 0.0020),         # 0-$1M: 0.20%
            (1_000_000, 0.0015), # $1M-$10M: 0.15%
            (10_000_000, 0.0010),# $10M-$50M: 0.10%
            (50_000_000, 0.0005),# $50M+: 0.05%
        ]
        self.monthly_volume = 0.0
    
    def calculate_commission(
        self,
        order: Order,
        fill_price: float,
        fill_quantity: float,
    ) -> float:
        """Calculate tiered commission based on volume."""
        trade_value = fill_price * fill_quantity
        
        # Find applicable tier
        rate = self.tiers[0][1]
        for threshold, tier_rate in self.tiers:
            if self.monthly_volume >= threshold:
                rate = tier_rate
        
        # Update monthly volume
        self.monthly_volume += trade_value
        
        return trade_value * rate
    
    def reset_monthly_volume(self) -> None:
        """Reset monthly volume counter."""
        self.monthly_volume = 0.0


class IBKRCommission(CommissionModel):
    """
    Interactive Brokers tiered commission model.
    
    Realistic model based on IBKR pricing.
    """
    
    def __init__(self, pro: bool = True):
        """
        Initialize IBKR commission model.
        
        Args:
            pro: Use Pro pricing (vs Lite)
        """
        self.pro = pro
        self.monthly_volume = 0
    
    def calculate_commission(
        self,
        order: Order,
        fill_price: float,
        fill_quantity: float,
    ) -> float:
        """Calculate IBKR-style commission."""
        trade_value = fill_price * fill_quantity
        shares = fill_quantity
        
        if not self.pro:
            # IBKR Lite: Free for US stocks
            return 0.0
        
        # IBKR Pro tiered pricing
        if self.monthly_volume <= 300_000:
            # Tier 1: $0.0035 per share
            commission = shares * 0.0035
        elif self.monthly_volume <= 3_000_000:
            # Tier 2: $0.0020 per share
            commission = shares * 0.0020
        elif self.monthly_volume <= 20_000_000:
            # Tier 3: $0.0015 per share
            commission = shares * 0.0015
        else:
            # Tier 4: $0.0010 per share
            commission = shares * 0.0010
        
        # Minimum $0.35, maximum 1% of trade value
        commission = max(commission, 0.35)
        commission = min(commission, trade_value * 0.01)
        
        self.monthly_volume += int(shares)
        
        return commission


# =============================================================================
# FILL MODELS
# =============================================================================

class FillModel(ABC):
    """
    Abstract base class for order fill models.
    
    Determines how orders are filled during simulation.
    """
    
    @abstractmethod
    def get_fill(
        self,
        order: Order,
        bar: OHLCV,
        **kwargs: Any,
    ) -> tuple[float, float, bool]:
        """
        Determine fill for an order.
        
        Args:
            order: Order to fill
            bar: Current market bar
            **kwargs: Additional parameters
        
        Returns:
            Tuple of (fill_price, fill_quantity, is_complete)
        """
        pass


class ImmediateFill(FillModel):
    """
    Immediate full fill model.
    
    All orders fill completely at bar close.
    Simple but unrealistic for large orders.
    """
    
    def __init__(self, slippage_model: SlippageModel | None = None):
        """
        Initialize immediate fill model.
        
        Args:
            slippage_model: Slippage model to use
        """
        self.slippage_model = slippage_model or NoSlippage()
    
    def get_fill(
        self,
        order: Order,
        bar: OHLCV,
        **kwargs: Any,
    ) -> tuple[float, float, bool]:
        """Fill entire order at close with slippage."""
        fill_price = self.slippage_model.get_fill_price(order, bar)
        return fill_price, order.quantity, True


class OHLCFill(FillModel):
    """
    OHLC-based fill model.
    
    More realistic fill using bar OHLC data:
    - Market orders: Fill at open of next bar
    - Limit orders: Fill if price touches limit
    - Stop orders: Trigger if price crosses stop
    """
    
    def __init__(
        self,
        slippage_model: SlippageModel | None = None,
        use_next_open: bool = True,
    ):
        """
        Initialize OHLC fill model.
        
        Args:
            slippage_model: Slippage model to use
            use_next_open: Use next bar open for market orders
        """
        self.slippage_model = slippage_model or NoSlippage()
        self.use_next_open = use_next_open
    
    def get_fill(
        self,
        order: Order,
        bar: OHLCV,
        **kwargs: Any,
    ) -> tuple[float, float, bool]:
        """Determine fill based on order type and OHLC."""
        order_type = order.order_type.lower()
        
        if order_type == "market":
            return self._fill_market(order, bar)
        elif order_type == "limit":
            return self._fill_limit(order, bar)
        elif order_type == "stop":
            return self._fill_stop(order, bar)
        elif order_type == "stop_limit":
            return self._fill_stop_limit(order, bar)
        else:
            # Default to market
            return self._fill_market(order, bar)
    
    def _fill_market(
        self,
        order: Order,
        bar: OHLCV,
    ) -> tuple[float, float, bool]:
        """Fill market order."""
        # Use open if available, else close
        base_price = bar.open if self.use_next_open else bar.close
        fill_price = self.slippage_model.get_fill_price(
            order, bar, reference_price=base_price
        )
        return fill_price, order.quantity, True
    
    def _fill_limit(
        self,
        order: Order,
        bar: OHLCV,
    ) -> tuple[float, float, bool]:
        """Fill limit order if price is favorable."""
        if order.price is None:
            return self._fill_market(order, bar)
        
        limit_price = order.price
        
        if order.side.lower() == "buy":
            # Buy limit: fill if low <= limit
            if bar.low <= limit_price:
                # Fill at limit or better
                fill_price = min(limit_price, bar.open)
                return fill_price, order.quantity, True
        else:
            # Sell limit: fill if high >= limit
            if bar.high >= limit_price:
                # Fill at limit or better
                fill_price = max(limit_price, bar.open)
                return fill_price, order.quantity, True
        
        # No fill
        return 0.0, 0.0, False
    
    def _fill_stop(
        self,
        order: Order,
        bar: OHLCV,
    ) -> tuple[float, float, bool]:
        """Fill stop order if stop is triggered."""
        if order.stop_price is None:
            return self._fill_market(order, bar)
        
        stop_price = order.stop_price
        
        if order.side.lower() == "buy":
            # Buy stop: trigger if high >= stop
            if bar.high >= stop_price:
                # Fill at stop or worse (with slippage)
                fill_price = max(stop_price, bar.open)
                fill_price = self.slippage_model.get_fill_price(
                    order, bar, reference_price=fill_price
                )
                return fill_price, order.quantity, True
        else:
            # Sell stop: trigger if low <= stop
            if bar.low <= stop_price:
                # Fill at stop or worse
                fill_price = min(stop_price, bar.open)
                fill_price = self.slippage_model.get_fill_price(
                    order, bar, reference_price=fill_price
                )
                return fill_price, order.quantity, True
        
        # No fill
        return 0.0, 0.0, False
    
    def _fill_stop_limit(
        self,
        order: Order,
        bar: OHLCV,
    ) -> tuple[float, float, bool]:
        """Fill stop-limit order."""
        if order.stop_price is None:
            return self._fill_limit(order, bar)
        
        stop_price = order.stop_price
        triggered = False
        
        if order.side.lower() == "buy":
            triggered = bar.high >= stop_price
        else:
            triggered = bar.low <= stop_price
        
        if triggered:
            # Stop triggered, now check limit
            return self._fill_limit(order, bar)
        
        return 0.0, 0.0, False


class PartialFill(FillModel):
    """
    Partial fill model based on volume.
    
    Orders fill partially based on available volume.
    More realistic for large orders.
    """
    
    def __init__(
        self,
        slippage_model: SlippageModel | None = None,
        max_participation: float = 0.10,
        min_fill_pct: float = 0.20,
    ):
        """
        Initialize partial fill model.
        
        Args:
            slippage_model: Slippage model to use
            max_participation: Maximum volume participation
            min_fill_pct: Minimum fill percentage per bar
        """
        self.slippage_model = slippage_model or VolumeSlippage()
        self.max_participation = max_participation
        self.min_fill_pct = min_fill_pct
    
    def get_fill(
        self,
        order: Order,
        bar: OHLCV,
        **kwargs: Any,
    ) -> tuple[float, float, bool]:
        """Fill based on volume availability."""
        if bar.volume <= 0:
            # No volume, no fill
            return 0.0, 0.0, False
        
        # Calculate available fill quantity
        max_fill = bar.volume * self.max_participation
        remaining = order.quantity - order.filled_quantity
        
        # Ensure minimum fill
        min_fill = order.quantity * self.min_fill_pct
        fill_qty = max(min(remaining, max_fill), min(min_fill, remaining))
        
        # Get fill price with slippage
        fill_price = self.slippage_model.get_fill_price(order, bar)
        
        # Check if order is complete
        is_complete = (order.filled_quantity + fill_qty) >= order.quantity
        
        return fill_price, fill_qty, is_complete


class ProbabilisticFill(FillModel):
    """
    Probabilistic fill model.
    
    Orders have probability of filling based on
    order type, market conditions, and randomness.
    """
    
    def __init__(
        self,
        slippage_model: SlippageModel | None = None,
        market_fill_prob: float = 0.99,
        limit_fill_prob: float = 0.70,
        seed: int | None = None,
    ):
        """
        Initialize probabilistic fill model.
        
        Args:
            slippage_model: Slippage model to use
            market_fill_prob: Probability of market order fill
            limit_fill_prob: Base probability of limit order fill
            seed: Random seed for reproducibility
        """
        self.slippage_model = slippage_model or PercentageSlippage()
        self.market_fill_prob = market_fill_prob
        self.limit_fill_prob = limit_fill_prob
        self.rng = np.random.default_rng(seed)
    
    def get_fill(
        self,
        order: Order,
        bar: OHLCV,
        **kwargs: Any,
    ) -> tuple[float, float, bool]:
        """Fill with probability."""
        order_type = order.order_type.lower()
        
        if order_type == "market":
            fill_prob = self.market_fill_prob
        elif order_type == "limit":
            # Adjust probability based on how aggressive limit is
            fill_prob = self._calculate_limit_prob(order, bar)
        else:
            fill_prob = self.limit_fill_prob
        
        # Random fill determination
        if self.rng.random() < fill_prob:
            fill_price = self.slippage_model.get_fill_price(order, bar)
            return fill_price, order.quantity, True
        
        return 0.0, 0.0, False
    
    def _calculate_limit_prob(self, order: Order, bar: OHLCV) -> float:
        """Calculate limit order fill probability."""
        if order.price is None:
            return self.limit_fill_prob
        
        limit = order.price
        
        if order.side.lower() == "buy":
            # More aggressive (higher) limit = higher prob
            if limit >= bar.high:
                return 0.95
            elif limit >= bar.close:
                return 0.80
            elif limit >= bar.low:
                return 0.50
            else:
                return 0.10
        else:
            # More aggressive (lower) limit = higher prob
            if limit <= bar.low:
                return 0.95
            elif limit <= bar.close:
                return 0.80
            elif limit <= bar.high:
                return 0.50
            else:
                return 0.10


# =============================================================================
# EXECUTION SIMULATOR
# =============================================================================

@dataclass
class FillResult:
    """Result of order fill simulation."""
    order_id: UUID
    filled: bool
    fill_price: float
    fill_quantity: float
    commission: float
    slippage: float
    timestamp: datetime
    is_complete: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_cost(self) -> float:
        """Total cost including commission."""
        return (self.fill_price * self.fill_quantity) + self.commission


class ExecutionSimulator:
    """
    Execution simulation engine.
    
    Combines slippage, commission, and fill models
    for realistic order execution.
    
    Example:
        simulator = ExecutionSimulator(
            slippage_model=VolumeSlippage(),
            commission_model=PerShareCommission(),
            fill_model=PartialFill(),
        )
        
        result = simulator.simulate_fill(order, bar)
    """
    
    def __init__(
        self,
        slippage_model: SlippageModel | None = None,
        commission_model: CommissionModel | None = None,
        fill_model: FillModel | None = None,
    ):
        """
        Initialize execution simulator.
        
        Args:
            slippage_model: Model for price slippage
            commission_model: Model for commissions
            fill_model: Model for order fills
        """
        self.slippage_model = slippage_model or PercentageSlippage(0.0005)
        self.commission_model = commission_model or PerShareCommission()
        self.fill_model = fill_model or OHLCFill(self.slippage_model)
    
    def simulate_fill(
        self,
        order: Order,
        bar: OHLCV,
        **kwargs: Any,
    ) -> FillResult:
        """
        Simulate order execution.
        
        Args:
            order: Order to execute
            bar: Current market bar
            **kwargs: Additional parameters
        
        Returns:
            FillResult with execution details
        """
        # Get fill from model
        fill_price, fill_qty, is_complete = self.fill_model.get_fill(
            order, bar, **kwargs
        )
        
        if fill_qty <= 0:
            # No fill
            return FillResult(
                order_id=order.id,
                filled=False,
                fill_price=0.0,
                fill_quantity=0.0,
                commission=0.0,
                slippage=0.0,
                timestamp=bar.timestamp,
                is_complete=False,
            )
        
        # Calculate commission
        commission = self.commission_model.calculate_commission(
            order, fill_price, fill_qty
        )
        
        # Calculate slippage (for reporting)
        base_price = bar.close
        if order.side.lower() == "buy":
            slippage = fill_price - base_price
        else:
            slippage = base_price - fill_price
        
        return FillResult(
            order_id=order.id,
            filled=True,
            fill_price=fill_price,
            fill_quantity=fill_qty,
            commission=commission,
            slippage=slippage,
            timestamp=bar.timestamp,
            is_complete=is_complete,
            metadata={
                "bar_close": bar.close,
                "bar_volume": bar.volume,
                "order_type": order.order_type,
            },
        )
    
    def simulate_multiple(
        self,
        orders: list[Order],
        bar: OHLCV,
        **kwargs: Any,
    ) -> list[FillResult]:
        """
        Simulate multiple order executions.
        
        Args:
            orders: List of orders
            bar: Current market bar
            **kwargs: Additional parameters
        
        Returns:
            List of FillResults
        """
        results = []
        
        # Sort by priority: market orders first, then by timestamp
        sorted_orders = sorted(
            orders,
            key=lambda o: (
                0 if o.order_type.lower() == "market" else 1,
                o.timestamp,
            )
        )
        
        for order in sorted_orders:
            result = self.simulate_fill(order, bar, **kwargs)
            results.append(result)
        
        return results


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_realistic_simulator(
    trading_style: str = "default",
) -> ExecutionSimulator:
    """
    Create a realistic execution simulator.
    
    Args:
        trading_style: Trading style preset
            - "default": Balanced settings
            - "hft": Low slippage, volume-based
            - "institutional": Higher impact
            - "retail": Simple fixed costs
    
    Returns:
        Configured ExecutionSimulator
    """
    if trading_style == "hft":
        return ExecutionSimulator(
            slippage_model=SpreadSlippage(spread_pct=0.0002),
            commission_model=PerShareCommission(per_share=0.001, min_commission=0.0),
            fill_model=OHLCFill(),
        )
    
    elif trading_style == "institutional":
        return ExecutionSimulator(
            slippage_model=MarketImpactSlippage(),
            commission_model=TieredCommission(),
            fill_model=PartialFill(max_participation=0.05),
        )
    
    elif trading_style == "retail":
        return ExecutionSimulator(
            slippage_model=FixedSlippage(0.01),
            commission_model=FixedCommission(1.0),
            fill_model=ImmediateFill(),
        )
    
    else:  # default
        return ExecutionSimulator(
            slippage_model=VolumeSlippage(),
            commission_model=PerShareCommission(),
            fill_model=OHLCFill(),
        )


def create_zero_cost_simulator() -> ExecutionSimulator:
    """
    Create a zero-cost simulator for testing.
    
    Returns:
        ExecutionSimulator with no costs
    """
    return ExecutionSimulator(
        slippage_model=NoSlippage(),
        commission_model=NoCommission(),
        fill_model=ImmediateFill(),
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Slippage Models
    "SlippageModel",
    "NoSlippage",
    "FixedSlippage",
    "PercentageSlippage",
    "VolumeSlippage",
    "SpreadSlippage",
    "MarketImpactSlippage",
    # Commission Models
    "CommissionModel",
    "NoCommission",
    "FixedCommission",
    "PerShareCommission",
    "PercentageCommission",
    "TieredCommission",
    "IBKRCommission",
    # Fill Models
    "FillModel",
    "ImmediateFill",
    "OHLCFill",
    "PartialFill",
    "ProbabilisticFill",
    # Simulator
    "FillResult",
    "ExecutionSimulator",
    # Factory
    "create_realistic_simulator",
    "create_zero_cost_simulator",
]