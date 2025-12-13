"""
Realistic Fill Model
====================
JPMorgan-Level Execution Simulation for Backtesting

Standard backtests assume instant fills at quoted prices.
Reality is different:
1. Market impact: Large orders move prices
2. Slippage: Fill price differs from signal price
3. Partial fills: Orders may not fully execute
4. Queue position: Limit orders wait in queue

This module implements realistic fill simulation:
- Volume-based market impact (Kyle's Lambda)
- Spread crossing costs
- Probability of fill for limit orders
- Partial fill modeling

Author: AlphaTrade Institutional System
Based on: ARCHITECTURAL_REVIEW_REPORT.md - Phase 3
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger(__name__)


class FillModel(Enum):
    """Available fill models"""
    INSTANT = "instant"  # Fills at mid-price (unrealistic)
    SPREAD_CROSS = "spread_cross"  # Crosses bid-ask spread
    VWAP = "vwap"  # Volume-weighted execution
    IMPACT = "impact"  # Full market impact model


@dataclass
class MarketMicrostructure:
    """Market microstructure parameters for a symbol"""
    symbol: str
    avg_spread_bps: float = 10.0  # Average bid-ask spread in bps
    avg_daily_volume: float = 1_000_000  # Average daily volume
    volatility: float = 0.02  # Daily volatility
    kyle_lambda: float = 0.0001  # Market impact parameter
    min_tick: float = 0.01  # Minimum price increment


@dataclass
class FillResult:
    """Result of fill simulation"""
    symbol: str
    side: str  # 'buy' or 'sell'
    requested_quantity: int
    filled_quantity: int
    avg_fill_price: float
    slippage_bps: float
    market_impact_bps: float
    spread_cost_bps: float
    total_cost_bps: float
    partial_fill: bool
    fill_time_bars: int  # Bars to complete fill

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'side': self.side,
            'requested_quantity': self.requested_quantity,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'slippage_bps': self.slippage_bps,
            'market_impact_bps': self.market_impact_bps,
            'spread_cost_bps': self.spread_cost_bps,
            'total_cost_bps': self.total_cost_bps,
            'partial_fill': self.partial_fill,
            'fill_time_bars': self.fill_time_bars
        }


class RealisticFillSimulator:
    """
    Simulates realistic order execution in backtesting.

    Models three main costs:
    1. Spread cost: Cost of crossing bid-ask spread
    2. Market impact: Price movement from order flow
    3. Timing cost: Opportunity cost from delayed execution

    The model uses:
    - Square root impact model: Impact ∝ √(Volume)
    - Spread estimation from OHLC when bid-ask not available
    - Probability-based limit order fills
    """

    def __init__(
        self,
        model: FillModel = FillModel.IMPACT,
        default_spread_bps: float = 10.0,
        impact_coefficient: float = 0.1,  # Scales market impact
        participation_rate: float = 0.1,  # Max % of volume to trade
        execution_uncertainty: float = 0.2,  # Random fill variation
        partial_fill_probability: float = 0.05  # Chance of partial fill
    ):
        self.model = model
        self.default_spread_bps = default_spread_bps
        self.impact_coefficient = impact_coefficient
        self.participation_rate = participation_rate
        self.execution_uncertainty = execution_uncertainty
        self.partial_fill_probability = partial_fill_probability

        # Per-symbol microstructure
        self._microstructure: Dict[str, MarketMicrostructure] = {}

        # Statistics
        self._stats = {
            'total_orders': 0,
            'total_slippage_bps': 0.0,
            'total_impact_bps': 0.0,
            'partial_fills': 0
        }

    def set_microstructure(
        self,
        symbol: str,
        avg_spread_bps: float = 10.0,
        avg_daily_volume: float = 1_000_000,
        volatility: float = 0.02
    ) -> None:
        """Set market microstructure for a symbol"""
        # Estimate Kyle's Lambda from volatility and volume
        kyle_lambda = volatility / np.sqrt(avg_daily_volume)

        self._microstructure[symbol] = MarketMicrostructure(
            symbol=symbol,
            avg_spread_bps=avg_spread_bps,
            avg_daily_volume=avg_daily_volume,
            volatility=volatility,
            kyle_lambda=kyle_lambda
        )

    def estimate_microstructure(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> None:
        """Estimate microstructure from OHLCV data"""
        if len(df) < 20:
            return

        # Estimate spread from high-low
        # Corwin-Schultz estimator
        log_hl = np.log(df['high'] / df['low'])
        beta = log_hl ** 2

        # Two-period high-low
        high_2 = df['high'].rolling(2).max()
        low_2 = df['low'].rolling(2).min()
        gamma = np.log(high_2 / low_2) ** 2

        alpha = (np.sqrt(2 * beta.mean()) - np.sqrt(beta.mean())) / (3 - 2 * np.sqrt(2)) - \
                np.sqrt(gamma.mean() / (3 - 2 * np.sqrt(2)))

        spread_pct = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        spread_bps = max(spread_pct * 10000, 1.0)  # Minimum 1 bp

        # Volatility
        returns = df['close'].pct_change().dropna()
        volatility = returns.std()

        # Average volume
        avg_volume = df['volume'].mean()

        self.set_microstructure(
            symbol=symbol,
            avg_spread_bps=spread_bps,
            avg_daily_volume=avg_volume,
            volatility=volatility
        )

    def simulate_fill(
        self,
        symbol: str,
        side: str,
        quantity: int,
        signal_price: float,
        current_bar: pd.Series
    ) -> FillResult:
        """
        Simulate realistic order fill.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            signal_price: Price at signal generation
            current_bar: OHLCV bar at execution time

        Returns:
            FillResult with execution details
        """
        self._stats['total_orders'] += 1

        # Get microstructure
        micro = self._microstructure.get(symbol)
        if not micro:
            micro = MarketMicrostructure(
                symbol=symbol,
                avg_spread_bps=self.default_spread_bps
            )

        if self.model == FillModel.INSTANT:
            return self._simulate_instant(symbol, side, quantity, signal_price, current_bar, micro)
        elif self.model == FillModel.SPREAD_CROSS:
            return self._simulate_spread_cross(symbol, side, quantity, signal_price, current_bar, micro)
        elif self.model == FillModel.VWAP:
            return self._simulate_vwap(symbol, side, quantity, signal_price, current_bar, micro)
        else:  # IMPACT
            return self._simulate_full_impact(symbol, side, quantity, signal_price, current_bar, micro)

    def _simulate_instant(
        self,
        symbol: str,
        side: str,
        quantity: int,
        signal_price: float,
        current_bar: pd.Series,
        micro: MarketMicrostructure
    ) -> FillResult:
        """Instant fill at mid-price (baseline, unrealistic)"""
        fill_price = (current_bar['high'] + current_bar['low']) / 2

        return FillResult(
            symbol=symbol,
            side=side,
            requested_quantity=quantity,
            filled_quantity=quantity,
            avg_fill_price=fill_price,
            slippage_bps=0.0,
            market_impact_bps=0.0,
            spread_cost_bps=0.0,
            total_cost_bps=0.0,
            partial_fill=False,
            fill_time_bars=0
        )

    def _simulate_spread_cross(
        self,
        symbol: str,
        side: str,
        quantity: int,
        signal_price: float,
        current_bar: pd.Series,
        micro: MarketMicrostructure
    ) -> FillResult:
        """Fill accounting for bid-ask spread crossing"""
        mid_price = (current_bar['high'] + current_bar['low']) / 2

        # Half spread cost (we cross half the spread)
        half_spread_bps = micro.avg_spread_bps / 2

        # Fill price (buy at ask, sell at bid)
        if side == 'buy':
            fill_price = mid_price * (1 + half_spread_bps / 10000)
        else:
            fill_price = mid_price * (1 - half_spread_bps / 10000)

        # Slippage from signal price
        slippage_bps = abs(fill_price - signal_price) / signal_price * 10000

        return FillResult(
            symbol=symbol,
            side=side,
            requested_quantity=quantity,
            filled_quantity=quantity,
            avg_fill_price=fill_price,
            slippage_bps=slippage_bps,
            market_impact_bps=0.0,
            spread_cost_bps=half_spread_bps,
            total_cost_bps=slippage_bps + half_spread_bps,
            partial_fill=False,
            fill_time_bars=0
        )

    def _simulate_vwap(
        self,
        symbol: str,
        side: str,
        quantity: int,
        signal_price: float,
        current_bar: pd.Series,
        micro: MarketMicrostructure
    ) -> FillResult:
        """Simulate VWAP execution"""
        # Use VWAP if available, else estimate
        if 'vwap' in current_bar and not pd.isna(current_bar['vwap']):
            vwap = current_bar['vwap']
        else:
            # Estimate VWAP as weighted average of OHLC
            vwap = (current_bar['open'] + current_bar['high'] +
                   current_bar['low'] + current_bar['close']) / 4

        # Add spread crossing
        half_spread_bps = micro.avg_spread_bps / 2
        if side == 'buy':
            fill_price = vwap * (1 + half_spread_bps / 10000)
        else:
            fill_price = vwap * (1 - half_spread_bps / 10000)

        slippage_bps = abs(fill_price - signal_price) / signal_price * 10000

        return FillResult(
            symbol=symbol,
            side=side,
            requested_quantity=quantity,
            filled_quantity=quantity,
            avg_fill_price=fill_price,
            slippage_bps=slippage_bps,
            market_impact_bps=0.0,
            spread_cost_bps=half_spread_bps,
            total_cost_bps=slippage_bps + half_spread_bps,
            partial_fill=False,
            fill_time_bars=0
        )

    def _simulate_full_impact(
        self,
        symbol: str,
        side: str,
        quantity: int,
        signal_price: float,
        current_bar: pd.Series,
        micro: MarketMicrostructure
    ) -> FillResult:
        """
        Full market impact model.

        Uses square-root impact model:
        Impact = λ * σ * √(V/ADV)

        where:
        - λ = impact coefficient
        - σ = volatility
        - V = order volume
        - ADV = average daily volume
        """
        bar_volume = current_bar['volume']
        close_price = current_bar['close']

        # Participation rate check
        max_quantity = int(bar_volume * self.participation_rate)

        # Partial fill if order is too large
        if quantity > max_quantity and max_quantity > 0:
            fill_quantity = max_quantity
            partial_fill = True
            self._stats['partial_fills'] += 1
        else:
            fill_quantity = quantity
            partial_fill = False

        # Market impact (square root model)
        if micro.avg_daily_volume > 0 and fill_quantity > 0:
            volume_fraction = fill_quantity / micro.avg_daily_volume
            impact_bps = self.impact_coefficient * micro.volatility * \
                        np.sqrt(volume_fraction) * 10000
        else:
            impact_bps = 0.0

        # Spread cost
        half_spread_bps = micro.avg_spread_bps / 2

        # Timing/random uncertainty
        uncertainty = np.random.normal(0, self.execution_uncertainty * half_spread_bps)

        # Total execution cost
        if side == 'buy':
            total_cost_bps = half_spread_bps + impact_bps + uncertainty
            fill_price = close_price * (1 + total_cost_bps / 10000)
        else:
            total_cost_bps = half_spread_bps + impact_bps + uncertainty
            fill_price = close_price * (1 - total_cost_bps / 10000)

        # Ensure fill price is within bar range
        fill_price = np.clip(fill_price, current_bar['low'], current_bar['high'])

        # Slippage from signal price
        slippage_bps = abs(fill_price - signal_price) / signal_price * 10000

        # Update statistics
        self._stats['total_slippage_bps'] += slippage_bps
        self._stats['total_impact_bps'] += impact_bps

        return FillResult(
            symbol=symbol,
            side=side,
            requested_quantity=quantity,
            filled_quantity=fill_quantity,
            avg_fill_price=fill_price,
            slippage_bps=slippage_bps,
            market_impact_bps=impact_bps,
            spread_cost_bps=half_spread_bps,
            total_cost_bps=slippage_bps,
            partial_fill=partial_fill,
            fill_time_bars=0 if not partial_fill else 1
        )

    def simulate_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        limit_price: float,
        bars: List[pd.Series],
        max_bars: int = 10
    ) -> Tuple[FillResult, int]:
        """
        Simulate limit order fill probability.

        Limit orders don't always fill. This models:
        - Fill probability based on price level
        - Queue position effects
        - Partial fills

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            limit_price: Limit price
            bars: Future bars for fill simulation
            max_bars: Maximum bars to wait

        Returns:
            Tuple of (FillResult, bars_to_fill)
        """
        micro = self._microstructure.get(symbol, MarketMicrostructure(symbol=symbol))

        total_filled = 0
        fill_prices = []
        bars_elapsed = 0

        for bar in bars[:max_bars]:
            bars_elapsed += 1

            # Check if limit price is touched
            if side == 'buy':
                # Buy limit fills if price drops to limit
                if bar['low'] <= limit_price:
                    # Fill probability based on how far through the level we are
                    penetration = (limit_price - bar['low']) / (bar['high'] - bar['low'] + 0.0001)
                    fill_prob = min(0.3 + 0.7 * penetration, 1.0)

                    if np.random.random() < fill_prob:
                        # How much fills?
                        max_fill = int(bar['volume'] * self.participation_rate)
                        this_fill = min(quantity - total_filled, max_fill)

                        total_filled += this_fill
                        fill_prices.extend([limit_price] * this_fill)

            else:  # sell
                # Sell limit fills if price rises to limit
                if bar['high'] >= limit_price:
                    penetration = (bar['high'] - limit_price) / (bar['high'] - bar['low'] + 0.0001)
                    fill_prob = min(0.3 + 0.7 * penetration, 1.0)

                    if np.random.random() < fill_prob:
                        max_fill = int(bar['volume'] * self.participation_rate)
                        this_fill = min(quantity - total_filled, max_fill)

                        total_filled += this_fill
                        fill_prices.extend([limit_price] * this_fill)

            if total_filled >= quantity:
                break

        # Calculate average fill
        if total_filled > 0:
            avg_fill = np.mean(fill_prices)
        else:
            avg_fill = limit_price  # No fill

        return FillResult(
            symbol=symbol,
            side=side,
            requested_quantity=quantity,
            filled_quantity=total_filled,
            avg_fill_price=avg_fill,
            slippage_bps=0.0,  # Limit orders don't have slippage in same sense
            market_impact_bps=0.0,
            spread_cost_bps=0.0,
            total_cost_bps=0.0,
            partial_fill=total_filled < quantity,
            fill_time_bars=bars_elapsed
        ), bars_elapsed

    def get_statistics(self) -> Dict[str, Any]:
        """Get fill simulation statistics"""
        n = self._stats['total_orders']
        if n == 0:
            return self._stats

        return {
            **self._stats,
            'avg_slippage_bps': self._stats['total_slippage_bps'] / n,
            'avg_impact_bps': self._stats['total_impact_bps'] / n,
            'partial_fill_rate': self._stats['partial_fills'] / n
        }

    def reset_statistics(self) -> None:
        """Reset statistics"""
        self._stats = {
            'total_orders': 0,
            'total_slippage_bps': 0.0,
            'total_impact_bps': 0.0,
            'partial_fills': 0
        }


# =============================================================================
# INTEGRATION WITH BACKTEST ENGINE
# =============================================================================

class RealisticBacktestFillProvider:
    """
    Integration class for realistic fills in BacktestEngine.

    Replaces the simple fill model with realistic execution.
    """

    def __init__(
        self,
        model: FillModel = FillModel.IMPACT,
        impact_coefficient: float = 0.1
    ):
        self._simulator = RealisticFillSimulator(
            model=model,
            impact_coefficient=impact_coefficient
        )

    def setup_symbols(self, data: Dict[str, pd.DataFrame]) -> None:
        """Estimate microstructure for all symbols"""
        for symbol, df in data.items():
            self._simulator.estimate_microstructure(symbol, df)

    def execute(
        self,
        symbol: str,
        side: str,
        quantity: int,
        signal_price: float,
        bar: pd.Series
    ) -> Tuple[int, float, float]:
        """
        Execute order with realistic fill.

        Returns:
            Tuple of (filled_quantity, fill_price, cost_bps)
        """
        result = self._simulator.simulate_fill(
            symbol=symbol,
            side=side,
            quantity=quantity,
            signal_price=signal_price,
            current_bar=bar
        )

        return result.filled_quantity, result.avg_fill_price, result.total_cost_bps

    def get_fill_result(
        self,
        symbol: str,
        side: str,
        quantity: int,
        signal_price: float,
        bar: pd.Series
    ) -> FillResult:
        """Get full fill result"""
        return self._simulator.simulate_fill(
            symbol=symbol,
            side=side,
            quantity=quantity,
            signal_price=signal_price,
            current_bar=bar
        )

    def get_statistics(self) -> Dict[str, Any]:
        return self._simulator.get_statistics()
