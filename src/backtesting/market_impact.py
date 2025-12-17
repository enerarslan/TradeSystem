"""
Market Impact Models for realistic transaction cost estimation.

This module provides institutional-grade market impact models including:
- Almgren-Chriss model for optimal execution
- Dynamic spread models
- Temporary and permanent impact components
- Latency simulation

Based on:
- Almgren, R. and Chriss, N. (2000). "Optimal Execution of Portfolio Transactions"
- Gatheral, J. (2010). "No-Dynamic-Arbitrage and Market Impact"

Designed for JPMorgan-level requirements:
- Realistic execution cost estimation
- Optimal execution trajectory computation
- Market microstructure modeling
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class MarketImpactEstimate:
    """Result of market impact estimation."""
    temporary_impact_bps: float
    permanent_impact_bps: float
    total_impact_bps: float
    spread_cost_bps: float
    total_cost_bps: float
    execution_price: float
    mid_price: float
    quantity: float
    participation_rate: float


@dataclass
class ExecutionTrajectory:
    """Optimal execution trajectory."""
    times: np.ndarray
    quantities: np.ndarray
    cumulative_quantities: np.ndarray
    expected_cost: float
    variance: float
    execution_time_horizon: float


class AlmgrenChrissModel:
    """
    Almgren-Chriss market impact model for optimal execution.

    The model decomposes execution costs into:
    1. Temporary impact: Price impact that decays immediately
    2. Permanent impact: Lasting price impact from information/inventory

    Total execution cost = Permanent Impact + Temporary Impact + Risk Penalty

    Parameters:
    - sigma: Daily volatility (annualized / sqrt(252))
    - eta: Temporary impact coefficient
    - gamma: Permanent impact coefficient
    - lambda_: Risk aversion parameter
    - adv: Average Daily Volume

    Example:
        model = AlmgrenChrissModel(
            sigma=0.02,      # 2% daily vol
            eta=0.1,         # Temporary impact
            gamma=0.05,      # Permanent impact
            lambda_=1e-6,    # Risk aversion
            adv=1_000_000    # 1M shares ADV
        )

        # Calculate impact for an order
        impact = model.total_cost(quantity=50000, execution_time=0.5)

        # Get optimal execution trajectory
        trajectory = model.optimal_trajectory(quantity=50000, time_horizon=1.0)
    """

    def __init__(
        self,
        sigma: float,
        eta: float = 0.1,
        gamma: float = 0.05,
        lambda_: float = 1e-6,
        adv: float = 1_000_000,
        price: float = 100.0,
    ):
        """
        Initialize Almgren-Chriss model.

        Args:
            sigma: Daily volatility
            eta: Temporary impact coefficient (linear)
            gamma: Permanent impact coefficient (linear)
            lambda_: Risk aversion parameter
            adv: Average Daily Volume
            price: Current price (for dollar impact)
        """
        self.sigma = sigma
        self.eta = eta
        self.gamma = gamma
        self.lambda_ = lambda_
        self.adv = adv
        self.price = price

    def temporary_impact(self, rate: float) -> float:
        """
        Calculate temporary price impact.

        Temporary impact is proportional to execution rate.

        Args:
            rate: Execution rate (shares per unit time / ADV)

        Returns:
            Temporary impact as fraction of price
        """
        # Linear temporary impact: h(v) = eta * v
        participation_rate = rate / self.adv
        return self.eta * participation_rate

    def permanent_impact(self, quantity: float) -> float:
        """
        Calculate permanent price impact.

        Permanent impact is proportional to total quantity.

        Args:
            quantity: Total order size

        Returns:
            Permanent impact as fraction of price
        """
        # Linear permanent impact: g(Q) = gamma * Q / ADV
        normalized_quantity = quantity / self.adv
        return self.gamma * normalized_quantity

    def execution_cost(
        self,
        quantity: float,
        execution_time: float,
    ) -> float:
        """
        Calculate expected execution cost (no risk).

        Args:
            quantity: Total order size
            execution_time: Time to execute (days)

        Returns:
            Expected cost as fraction of position value
        """
        # Participation rate
        rate = quantity / execution_time if execution_time > 0 else quantity

        # Permanent impact (incurred regardless of trajectory)
        permanent = self.permanent_impact(quantity)

        # Temporary impact (depends on execution speed)
        temporary = self.temporary_impact(rate) * execution_time

        return permanent + temporary

    def execution_risk(
        self,
        quantity: float,
        execution_time: float,
    ) -> float:
        """
        Calculate execution risk (variance of execution cost).

        Risk comes from price volatility during execution.

        Args:
            quantity: Total order size
            execution_time: Time to execute (days)

        Returns:
            Variance of execution cost
        """
        # Risk is proportional to sigma^2 * X^2 * T
        # where X is remaining quantity and T is time
        return self.sigma ** 2 * quantity ** 2 * execution_time / 3

    def total_cost(
        self,
        quantity: float,
        execution_time: float,
        include_risk: bool = True,
    ) -> float:
        """
        Calculate total execution cost including risk penalty.

        Total = Expected Cost + lambda * Risk

        Args:
            quantity: Total order size
            execution_time: Time to execute (days)
            include_risk: Include risk penalty

        Returns:
            Total execution cost
        """
        expected = self.execution_cost(quantity, execution_time)

        if include_risk:
            risk = self.execution_risk(quantity, execution_time)
            return expected + self.lambda_ * risk

        return expected

    def optimal_execution_time(
        self,
        quantity: float,
    ) -> float:
        """
        Calculate optimal execution time that minimizes total cost.

        Balances market impact (faster is worse) vs. risk (slower is worse).

        Args:
            quantity: Total order size

        Returns:
            Optimal execution time in days
        """
        if not SCIPY_AVAILABLE:
            # Approximate solution
            return np.sqrt(quantity / self.adv)

        def objective(T):
            if T <= 0:
                return 1e10
            return self.total_cost(quantity, T[0])

        result = minimize(
            objective,
            x0=[1.0],
            method='L-BFGS-B',
            bounds=[(0.01, 10.0)],
        )

        return result.x[0]

    def optimal_trajectory(
        self,
        quantity: float,
        time_horizon: float,
        n_steps: int = 10,
    ) -> ExecutionTrajectory:
        """
        Calculate optimal execution trajectory.

        The optimal trajectory balances market impact with timing risk.

        For linear impact, the optimal trajectory is:
        x(t) = X * sinh(kappa * (T-t)) / sinh(kappa * T)

        where kappa = sqrt(lambda * sigma^2 / eta)

        Args:
            quantity: Total order size
            time_horizon: Time to complete execution (days)
            n_steps: Number of time steps

        Returns:
            ExecutionTrajectory with optimal schedule
        """
        # Calculate kappa (urgency parameter)
        if self.eta > 0 and self.lambda_ > 0:
            kappa = np.sqrt(self.lambda_ * self.sigma ** 2 / self.eta)
        else:
            kappa = 0

        times = np.linspace(0, time_horizon, n_steps + 1)
        dt = time_horizon / n_steps

        if kappa > 0 and kappa * time_horizon < 100:  # Avoid overflow
            # Optimal trajectory (TWAP modified by urgency)
            sinh_kT = np.sinh(kappa * time_horizon)
            remaining = np.array([
                quantity * np.sinh(kappa * (time_horizon - t)) / sinh_kT
                for t in times
            ])
            cumulative = quantity - remaining
        else:
            # TWAP (Time-Weighted Average Price)
            cumulative = np.linspace(0, quantity, n_steps + 1)
            remaining = quantity - cumulative

        # Quantities per step (derivative)
        quantities = np.zeros(n_steps + 1)
        quantities[:-1] = np.diff(cumulative)
        quantities[-1] = 0  # Last step complete

        # Expected cost and variance
        expected_cost = self.execution_cost(quantity, time_horizon)
        variance = self.execution_risk(quantity, time_horizon)

        return ExecutionTrajectory(
            times=times,
            quantities=quantities,
            cumulative_quantities=cumulative,
            expected_cost=expected_cost,
            variance=variance,
            execution_time_horizon=time_horizon,
        )

    def estimate_impact(
        self,
        quantity: float,
        side: str,
        mid_price: float,
        spread_bps: float = 10.0,
        execution_time: float = None,
    ) -> MarketImpactEstimate:
        """
        Estimate market impact for an order.

        Args:
            quantity: Order quantity
            side: 'buy' or 'sell'
            mid_price: Current mid price
            spread_bps: Current bid-ask spread in basis points
            execution_time: Execution time (None = instant)

        Returns:
            MarketImpactEstimate with detailed cost breakdown
        """
        if execution_time is None or execution_time == 0:
            execution_time = 0.01  # Small time for immediate execution

        participation_rate = quantity / (self.adv * execution_time)

        # Calculate impacts
        temp_impact = self.temporary_impact(quantity / execution_time)
        perm_impact = self.permanent_impact(quantity)

        # Convert to basis points
        temp_bps = temp_impact * 10000
        perm_bps = perm_impact * 10000
        total_impact_bps = temp_bps + perm_bps

        # Spread cost (half spread for each side)
        spread_cost_bps = spread_bps / 2

        # Total cost
        total_cost_bps = total_impact_bps + spread_cost_bps

        # Execution price
        direction = 1 if side.lower() == 'buy' else -1
        execution_price = mid_price * (1 + direction * total_cost_bps / 10000)

        return MarketImpactEstimate(
            temporary_impact_bps=temp_bps,
            permanent_impact_bps=perm_bps,
            total_impact_bps=total_impact_bps,
            spread_cost_bps=spread_cost_bps,
            total_cost_bps=total_cost_bps,
            execution_price=execution_price,
            mid_price=mid_price,
            quantity=quantity,
            participation_rate=participation_rate,
        )

    @classmethod
    def from_historical_data(
        cls,
        prices: pd.Series,
        volumes: pd.Series,
        trades: Optional[pd.DataFrame] = None,
    ) -> "AlmgrenChrissModel":
        """
        Calibrate model from historical data.

        Args:
            prices: Historical price series
            volumes: Historical volume series
            trades: Optional trade data for impact estimation

        Returns:
            Calibrated AlmgrenChrissModel
        """
        # Calculate volatility
        returns = prices.pct_change().dropna()
        sigma = returns.std()

        # Average daily volume
        adv = volumes.mean()

        # Current price
        price = prices.iloc[-1]

        # Default impact parameters (can be refined with trade data)
        eta = 0.1
        gamma = 0.05

        if trades is not None:
            # Calibrate from trade data (implementation depends on data format)
            pass

        return cls(
            sigma=sigma,
            eta=eta,
            gamma=gamma,
            adv=adv,
            price=price,
        )


class DynamicSpreadModel:
    """
    Dynamic bid-ask spread model.

    Models spread as function of:
    - Base spread
    - Volatility
    - Volume
    - Time of day

    Example:
        model = DynamicSpreadModel(base_spread_bps=5.0)

        spread = model.estimate_spread(
            volatility=0.02,
            volume_ratio=1.5,
            time=datetime.time(10, 30)
        )
    """

    def __init__(
        self,
        base_spread_bps: float = 5.0,
        volatility_sensitivity: float = 0.5,
        volume_sensitivity: float = 0.3,
        min_spread_bps: float = 1.0,
        max_spread_bps: float = 100.0,
    ):
        """
        Initialize spread model.

        Args:
            base_spread_bps: Base spread in basis points
            volatility_sensitivity: Spread sensitivity to volatility
            volume_sensitivity: Spread sensitivity to volume
            min_spread_bps: Minimum spread floor
            max_spread_bps: Maximum spread cap
        """
        self.base_spread_bps = base_spread_bps
        self.volatility_sensitivity = volatility_sensitivity
        self.volume_sensitivity = volume_sensitivity
        self.min_spread_bps = min_spread_bps
        self.max_spread_bps = max_spread_bps

        # Intraday pattern (U-shaped)
        self._intraday_multipliers = self._build_intraday_pattern()

    def _build_intraday_pattern(self) -> Dict[int, float]:
        """Build U-shaped intraday spread pattern."""
        # Higher spreads at open and close
        pattern = {}
        for hour in range(9, 17):
            if hour < 10:  # Opening
                pattern[hour] = 1.5
            elif hour < 11:
                pattern[hour] = 1.2
            elif hour < 15:  # Midday
                pattern[hour] = 1.0
            elif hour < 16:
                pattern[hour] = 1.2
            else:  # Closing
                pattern[hour] = 1.5
        return pattern

    def estimate_spread(
        self,
        volatility: float = 0.02,
        volume_ratio: float = 1.0,
        time_of_day: Optional[time] = None,
        current_price: float = 100.0,
    ) -> float:
        """
        Estimate current bid-ask spread.

        Args:
            volatility: Current volatility (daily)
            volume_ratio: Current volume / average volume
            time_of_day: Current time
            current_price: Current price

        Returns:
            Estimated spread in basis points
        """
        # Base spread
        spread = self.base_spread_bps

        # Volatility adjustment
        # Higher volatility -> wider spread
        vol_multiplier = 1 + self.volatility_sensitivity * (volatility / 0.02 - 1)
        spread *= max(0.5, vol_multiplier)

        # Volume adjustment
        # Higher volume -> tighter spread (more liquidity)
        # Lower volume -> wider spread
        if volume_ratio > 0:
            vol_adj = self.volume_sensitivity * (1 / volume_ratio - 1)
            spread *= max(0.5, 1 + vol_adj)

        # Intraday adjustment
        if time_of_day is not None:
            hour = time_of_day.hour
            multiplier = self._intraday_multipliers.get(hour, 1.0)
            spread *= multiplier

        # Apply bounds
        spread = np.clip(spread, self.min_spread_bps, self.max_spread_bps)

        return spread

    def get_effective_price(
        self,
        mid_price: float,
        side: str,
        size: float,
        adv: float = 1_000_000,
        volatility: float = 0.02,
    ) -> float:
        """
        Get effective execution price including spread and size impact.

        Args:
            mid_price: Current mid price
            side: 'buy' or 'sell'
            size: Order size
            adv: Average daily volume
            volatility: Current volatility

        Returns:
            Effective execution price
        """
        # Base spread
        spread_bps = self.estimate_spread(volatility=volatility)

        # Size impact (larger orders get worse prices)
        participation_rate = size / adv
        size_impact_bps = 5 * participation_rate * 10000  # 5 bps per 1% participation

        total_impact_bps = spread_bps / 2 + size_impact_bps

        direction = 1 if side.lower() == 'buy' else -1
        return mid_price * (1 + direction * total_impact_bps / 10000)


class LatencySimulator:
    """
    Simulates network latency and order processing delays.

    Models:
    - Signal to order latency
    - Order to exchange latency
    - Exchange processing time
    - Fill notification latency
    """

    def __init__(
        self,
        signal_to_order_ms: int = 10,
        order_to_exchange_ms: int = 5,
        exchange_processing_ms: int = 1,
        fill_notification_ms: int = 5,
        jitter_pct: float = 0.2,
    ):
        """
        Initialize latency simulator.

        Args:
            signal_to_order_ms: Time from signal to order submission
            order_to_exchange_ms: Network latency to exchange
            exchange_processing_ms: Exchange matching time
            fill_notification_ms: Fill notification latency
            jitter_pct: Random jitter percentage
        """
        self.signal_to_order_ms = signal_to_order_ms
        self.order_to_exchange_ms = order_to_exchange_ms
        self.exchange_processing_ms = exchange_processing_ms
        self.fill_notification_ms = fill_notification_ms
        self.jitter_pct = jitter_pct

    def _add_jitter(self, base_latency: float) -> float:
        """Add random jitter to latency."""
        jitter = np.random.uniform(
            -self.jitter_pct,
            self.jitter_pct
        ) * base_latency
        return max(0, base_latency + jitter)

    def get_total_latency_ms(self) -> float:
        """Get total round-trip latency with jitter."""
        total = (
            self._add_jitter(self.signal_to_order_ms) +
            self._add_jitter(self.order_to_exchange_ms) +
            self._add_jitter(self.exchange_processing_ms) +
            self._add_jitter(self.fill_notification_ms)
        )
        return total

    def add_latency_to_timestamp(
        self,
        timestamp: datetime,
        latency_type: str = 'full',
    ) -> datetime:
        """
        Add latency to a timestamp.

        Args:
            timestamp: Original timestamp
            latency_type: 'full', 'order', 'fill'

        Returns:
            Adjusted timestamp
        """
        if latency_type == 'full':
            latency_ms = self.get_total_latency_ms()
        elif latency_type == 'order':
            latency_ms = (
                self._add_jitter(self.signal_to_order_ms) +
                self._add_jitter(self.order_to_exchange_ms)
            )
        elif latency_type == 'fill':
            latency_ms = (
                self._add_jitter(self.exchange_processing_ms) +
                self._add_jitter(self.fill_notification_ms)
            )
        else:
            latency_ms = 0

        return timestamp + timedelta(milliseconds=latency_ms)

    def simulate_price_slippage(
        self,
        target_price: float,
        volatility: float,
        latency_ms: float = None,
    ) -> float:
        """
        Simulate price slippage due to latency.

        Args:
            target_price: Target execution price
            volatility: Price volatility (per millisecond)
            latency_ms: Latency in milliseconds (None = use model)

        Returns:
            Slipped price
        """
        if latency_ms is None:
            latency_ms = self.get_total_latency_ms()

        # Price can move during latency
        # Model as random walk
        price_change = np.random.normal(0, volatility * np.sqrt(latency_ms))

        return target_price * (1 + price_change)


# Convenience functions
def estimate_market_impact(
    quantity: float,
    adv: float,
    volatility: float,
    side: str = 'buy',
    mid_price: float = 100.0,
) -> MarketImpactEstimate:
    """
    Quick market impact estimation.

    Args:
        quantity: Order quantity
        adv: Average Daily Volume
        volatility: Daily volatility
        side: 'buy' or 'sell'
        mid_price: Current mid price

    Returns:
        MarketImpactEstimate
    """
    model = AlmgrenChrissModel(
        sigma=volatility,
        adv=adv,
        price=mid_price,
    )
    return model.estimate_impact(quantity, side, mid_price)


def calculate_transaction_costs(
    quantity: float,
    price: float,
    adv: float,
    volatility: float,
    commission_bps: float = 5.0,
    spread_bps: float = 10.0,
) -> Dict[str, float]:
    """
    Calculate all-in transaction costs.

    Returns breakdown of:
    - Commission
    - Spread cost
    - Market impact
    - Total cost
    """
    model = AlmgrenChrissModel(sigma=volatility, adv=adv, price=price)
    impact = model.estimate_impact(quantity, 'buy', price, spread_bps)

    notional = quantity * price

    return {
        'commission_bps': commission_bps,
        'spread_cost_bps': impact.spread_cost_bps,
        'market_impact_bps': impact.total_impact_bps,
        'total_cost_bps': commission_bps + impact.total_cost_bps,
        'commission_dollars': notional * commission_bps / 10000,
        'spread_cost_dollars': notional * impact.spread_cost_bps / 10000,
        'market_impact_dollars': notional * impact.total_impact_bps / 10000,
        'total_cost_dollars': notional * (commission_bps + impact.total_cost_bps) / 10000,
    }
