"""
Portfolio rebalancing for AlphaTrade system.

This module provides:
- Calendar-based rebalancing
- Threshold-based rebalancing
- Signal-based rebalancing
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger


class RebalanceType(Enum):
    """Types of rebalancing triggers."""

    CALENDAR = "calendar"
    THRESHOLD = "threshold"
    SIGNAL = "signal"


@dataclass
class RebalanceSignal:
    """Signal to rebalance the portfolio."""

    timestamp: datetime | pd.Timestamp
    trigger_type: RebalanceType
    current_weights: pd.Series
    target_weights: pd.Series
    turnover: float
    reason: str


class Rebalancer:
    """
    Portfolio rebalancer with multiple trigger types.

    Supports:
    - Calendar-based (daily, weekly, monthly)
    - Threshold-based (when drift exceeds threshold)
    - Signal-based (when strategy signals change)
    """

    def __init__(
        self,
        rebalance_type: Literal["calendar", "threshold", "signal"] = "threshold",
        calendar_frequency: Literal["daily", "weekly", "monthly"] = "weekly",
        drift_threshold: float = 0.05,
        min_trade_size: float = 0.01,
        max_turnover: float = 0.5,
    ) -> None:
        """
        Initialize the rebalancer.

        Args:
            rebalance_type: Type of rebalancing trigger
            calendar_frequency: Frequency for calendar rebalancing
            drift_threshold: Threshold for drift-based rebalancing
            min_trade_size: Minimum trade size (as fraction)
            max_turnover: Maximum allowed turnover
        """
        self.rebalance_type = rebalance_type
        self.calendar_frequency = calendar_frequency
        self.drift_threshold = drift_threshold
        self.min_trade_size = min_trade_size
        self.max_turnover = max_turnover

        self._last_rebalance_date: pd.Timestamp | None = None
        self._last_weights: pd.Series | None = None

    def check_rebalance(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        current_date: pd.Timestamp,
        new_signal: bool = False,
    ) -> RebalanceSignal | None:
        """
        Check if rebalancing is needed.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            current_date: Current timestamp
            new_signal: Whether there's a new trading signal

        Returns:
            RebalanceSignal if rebalancing needed, None otherwise
        """
        should_rebalance = False
        reason = ""

        if self.rebalance_type == "calendar":
            should_rebalance, reason = self._check_calendar(current_date)

        elif self.rebalance_type == "threshold":
            should_rebalance, reason = self._check_threshold(
                current_weights, target_weights
            )

        elif self.rebalance_type == "signal":
            should_rebalance, reason = self._check_signal(new_signal)

        if not should_rebalance:
            return None

        # Calculate turnover
        turnover = self._calculate_turnover(current_weights, target_weights)

        # Check turnover limit
        if turnover > self.max_turnover:
            # Scale down trades to max turnover
            target_weights = self._scale_trades(
                current_weights, target_weights, self.max_turnover
            )
            turnover = self.max_turnover

        # Filter small trades
        target_weights = self._filter_small_trades(
            current_weights, target_weights
        )

        # Update state
        self._last_rebalance_date = current_date
        self._last_weights = target_weights.copy()

        return RebalanceSignal(
            timestamp=current_date,
            trigger_type=RebalanceType(self.rebalance_type),
            current_weights=current_weights,
            target_weights=target_weights,
            turnover=turnover,
            reason=reason,
        )

    def _check_calendar(
        self,
        current_date: pd.Timestamp,
    ) -> tuple[bool, str]:
        """Check calendar-based rebalancing."""
        if self._last_rebalance_date is None:
            return True, "Initial rebalance"

        if self.calendar_frequency == "daily":
            if current_date.date() > self._last_rebalance_date.date():
                return True, "Daily rebalance"

        elif self.calendar_frequency == "weekly":
            # Rebalance on Monday
            if (current_date - self._last_rebalance_date).days >= 7:
                return True, "Weekly rebalance"

        elif self.calendar_frequency == "monthly":
            if current_date.month != self._last_rebalance_date.month:
                return True, "Monthly rebalance"

        return False, ""

    def _check_threshold(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
    ) -> tuple[bool, str]:
        """Check threshold-based rebalancing."""
        # Calculate drift
        drift = (current_weights - target_weights).abs()
        max_drift = drift.max()

        if max_drift > self.drift_threshold:
            return True, f"Drift threshold exceeded: {max_drift:.2%}"

        return False, ""

    def _check_signal(self, new_signal: bool) -> tuple[bool, str]:
        """Check signal-based rebalancing."""
        if new_signal:
            return True, "New trading signal"
        return False, ""

    def _calculate_turnover(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
    ) -> float:
        """Calculate portfolio turnover."""
        # Align indices
        all_assets = current_weights.index.union(target_weights.index)
        current = current_weights.reindex(all_assets, fill_value=0)
        target = target_weights.reindex(all_assets, fill_value=0)

        # One-way turnover
        return (current - target).abs().sum() / 2

    def _scale_trades(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        max_turnover: float,
    ) -> pd.Series:
        """Scale trades to stay within turnover limit."""
        current_turnover = self._calculate_turnover(current_weights, target_weights)

        if current_turnover <= max_turnover:
            return target_weights

        scale = max_turnover / current_turnover
        diff = target_weights - current_weights

        return current_weights + diff * scale

    def _filter_small_trades(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
    ) -> pd.Series:
        """Filter out small trades below minimum size."""
        diff = (target_weights - current_weights).abs()

        # Keep target weights only if trade is above minimum
        mask = diff >= self.min_trade_size
        filtered = current_weights.copy()
        filtered[mask] = target_weights[mask]

        # Renormalize
        filtered = filtered / filtered.abs().sum() if filtered.abs().sum() > 0 else filtered

        return filtered

    def get_trades(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        portfolio_value: float,
    ) -> pd.DataFrame:
        """
        Calculate trades needed for rebalancing.

        Args:
            current_weights: Current weights
            target_weights: Target weights
            portfolio_value: Total portfolio value

        Returns:
            DataFrame with trade details
        """
        # Align indices
        all_assets = current_weights.index.union(target_weights.index)
        current = current_weights.reindex(all_assets, fill_value=0)
        target = target_weights.reindex(all_assets, fill_value=0)

        trades = []
        for asset in all_assets:
            weight_change = target[asset] - current[asset]
            trade_value = weight_change * portfolio_value

            if abs(weight_change) >= self.min_trade_size:
                trades.append({
                    "asset": asset,
                    "current_weight": current[asset],
                    "target_weight": target[asset],
                    "weight_change": weight_change,
                    "trade_value": trade_value,
                    "side": "BUY" if weight_change > 0 else "SELL",
                })

        return pd.DataFrame(trades)

    def reset(self) -> None:
        """Reset rebalancer state."""
        self._last_rebalance_date = None
        self._last_weights = None


class AdaptiveRebalancer(Rebalancer):
    """
    Adaptive rebalancer that adjusts threshold based on conditions.

    Considers:
    - Market volatility
    - Transaction costs
    - Tax implications
    """

    def __init__(
        self,
        base_threshold: float = 0.05,
        vol_adjustment: bool = True,
        cost_adjustment: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize adaptive rebalancer.

        Args:
            base_threshold: Base drift threshold
            vol_adjustment: Adjust threshold for volatility
            cost_adjustment: Adjust threshold for costs
            **kwargs: Arguments for parent class
        """
        super().__init__(**kwargs)
        self.base_threshold = base_threshold
        self.vol_adjustment = vol_adjustment
        self.cost_adjustment = cost_adjustment

    def get_adaptive_threshold(
        self,
        market_vol: float,
        avg_spread: float = 0.001,
    ) -> float:
        """
        Calculate adaptive threshold.

        Args:
            market_vol: Current market volatility
            avg_spread: Average bid-ask spread

        Returns:
            Adjusted threshold
        """
        threshold = self.base_threshold

        # Increase threshold in high volatility
        if self.vol_adjustment:
            vol_factor = 1 + (market_vol - 0.15) / 0.15  # Baseline vol = 15%
            threshold *= max(0.5, min(2.0, vol_factor))

        # Increase threshold for higher costs
        if self.cost_adjustment:
            cost_factor = 1 + avg_spread * 100
            threshold *= cost_factor

        return threshold
