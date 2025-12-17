"""
Drawdown analysis and control for AlphaTrade system.

This module provides:
- Drawdown calculation
- Drawdown-based position reduction
- Recovery analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class DrawdownEvent:
    """Information about a drawdown event."""

    start_date: datetime | pd.Timestamp
    end_date: datetime | pd.Timestamp | None
    trough_date: datetime | pd.Timestamp
    peak_value: float
    trough_value: float
    current_value: float
    drawdown_pct: float
    duration_days: int
    recovery_days: int | None
    is_active: bool


class DrawdownController:
    """
    Drawdown monitoring and position control.

    Implements automatic position reduction based on
    drawdown thresholds.
    """

    def __init__(
        self,
        max_drawdown: float = 0.15,
        reduce_at_drawdown: float = 0.10,
        reduce_by_pct: float = 0.50,
        close_all_at_drawdown: float = 0.20,
        recovery_threshold: float = 0.05,
    ) -> None:
        """
        Initialize the drawdown controller.

        Args:
            max_drawdown: Maximum acceptable drawdown
            reduce_at_drawdown: Drawdown level to start reducing
            reduce_by_pct: How much to reduce positions
            close_all_at_drawdown: Drawdown level to close all
            recovery_threshold: Drawdown level to resume normal trading
        """
        self.max_drawdown = max_drawdown
        self.reduce_at_drawdown = reduce_at_drawdown
        self.reduce_by_pct = reduce_by_pct
        self.close_all_at_drawdown = close_all_at_drawdown
        self.recovery_threshold = recovery_threshold

        self._peak_value = 0.0
        self._current_drawdown = 0.0
        self._is_reduced = False
        self._is_closed = False

    def update(self, portfolio_value: float) -> dict:
        """
        Update drawdown state with new portfolio value.

        Args:
            portfolio_value: Current portfolio value

        Returns:
            Dictionary with drawdown status and actions
        """
        # Update peak
        if portfolio_value > self._peak_value:
            self._peak_value = portfolio_value
            self._is_reduced = False  # Reset on new peak

        # Calculate drawdown
        if self._peak_value > 0:
            self._current_drawdown = (self._peak_value - portfolio_value) / self._peak_value
        else:
            self._current_drawdown = 0.0

        # Determine action
        action = "none"
        scale_factor = 1.0

        if self._current_drawdown >= self.close_all_at_drawdown:
            action = "close_all"
            scale_factor = 0.0
            self._is_closed = True
            logger.warning(
                f"DRAWDOWN ALERT: {self._current_drawdown:.1%} - Closing all positions"
            )

        elif self._current_drawdown >= self.reduce_at_drawdown:
            if not self._is_reduced:
                action = "reduce"
                scale_factor = 1.0 - self.reduce_by_pct
                self._is_reduced = True
                logger.warning(
                    f"DRAWDOWN ALERT: {self._current_drawdown:.1%} - "
                    f"Reducing positions by {self.reduce_by_pct:.0%}"
                )
            else:
                action = "maintain_reduced"
                scale_factor = 1.0 - self.reduce_by_pct

        elif self._is_closed and self._current_drawdown < self.recovery_threshold:
            action = "resume"
            scale_factor = 1.0 - self.reduce_by_pct  # Gradual recovery
            self._is_closed = False
            logger.info("Drawdown recovery: Resuming trading with reduced positions")

        return {
            "drawdown": self._current_drawdown,
            "peak_value": self._peak_value,
            "action": action,
            "scale_factor": scale_factor,
            "is_reduced": self._is_reduced,
            "is_closed": self._is_closed,
        }

    def apply_to_positions(
        self,
        positions: pd.DataFrame | pd.Series,
        portfolio_value: float,
    ) -> pd.DataFrame | pd.Series:
        """
        Apply drawdown controls to positions.

        Args:
            positions: Target positions
            portfolio_value: Current portfolio value

        Returns:
            Adjusted positions
        """
        status = self.update(portfolio_value)

        adjusted = positions * status["scale_factor"]

        return adjusted

    def reset(self) -> None:
        """Reset the controller state."""
        self._peak_value = 0.0
        self._current_drawdown = 0.0
        self._is_reduced = False
        self._is_closed = False

    @property
    def current_drawdown(self) -> float:
        """Get current drawdown level."""
        return self._current_drawdown

    @property
    def peak_value(self) -> float:
        """Get peak portfolio value."""
        return self._peak_value


def calculate_drawdown(equity: pd.Series) -> pd.DataFrame:
    """
    Calculate drawdown series from equity curve.

    Args:
        equity: Equity curve series

    Returns:
        DataFrame with drawdown, peak, and underwater values
    """
    # Running maximum
    running_max = equity.expanding().max()

    # Drawdown
    drawdown = (equity - running_max) / running_max

    # Underwater (absolute)
    underwater = equity - running_max

    return pd.DataFrame({
        "equity": equity,
        "peak": running_max,
        "drawdown": drawdown,
        "underwater": underwater,
    })


def calculate_max_drawdown(equity: pd.Series) -> float:
    """
    Calculate maximum drawdown.

    Args:
        equity: Equity curve series

    Returns:
        Maximum drawdown (positive value)
    """
    dd = calculate_drawdown(equity)
    return abs(dd["drawdown"].min())


def find_drawdown_events(
    equity: pd.Series,
    min_drawdown: float = 0.05,
) -> list[DrawdownEvent]:
    """
    Find all drawdown events in equity curve.

    Args:
        equity: Equity curve series
        min_drawdown: Minimum drawdown to consider

    Returns:
        List of DrawdownEvent objects
    """
    dd_df = calculate_drawdown(equity)
    events = []

    in_drawdown = False
    event_start = None
    peak_value = 0.0
    trough_value = np.inf
    trough_date = None

    for idx, row in dd_df.iterrows():
        if not in_drawdown and row["drawdown"] < -min_drawdown:
            # Start of drawdown
            in_drawdown = True
            event_start = idx
            peak_value = row["peak"]
            trough_value = row["equity"]
            trough_date = idx

        elif in_drawdown:
            if row["equity"] < trough_value:
                # New trough
                trough_value = row["equity"]
                trough_date = idx

            if row["equity"] >= row["peak"]:
                # Recovery
                event = DrawdownEvent(
                    start_date=event_start,
                    end_date=idx,
                    trough_date=trough_date,
                    peak_value=peak_value,
                    trough_value=trough_value,
                    current_value=row["equity"],
                    drawdown_pct=(peak_value - trough_value) / peak_value,
                    duration_days=(idx - event_start).days if hasattr(idx - event_start, 'days') else 0,
                    recovery_days=(idx - trough_date).days if hasattr(idx - trough_date, 'days') else 0,
                    is_active=False,
                )
                events.append(event)

                in_drawdown = False
                event_start = None
                trough_value = np.inf

    # Check for active drawdown
    if in_drawdown:
        last_row = dd_df.iloc[-1]
        event = DrawdownEvent(
            start_date=event_start,
            end_date=None,
            trough_date=trough_date,
            peak_value=peak_value,
            trough_value=trough_value,
            current_value=last_row["equity"],
            drawdown_pct=(peak_value - trough_value) / peak_value,
            duration_days=(dd_df.index[-1] - event_start).days if hasattr(dd_df.index[-1] - event_start, 'days') else 0,
            recovery_days=None,
            is_active=True,
        )
        events.append(event)

    return events


def drawdown_statistics(equity: pd.Series) -> dict:
    """
    Calculate comprehensive drawdown statistics.

    Args:
        equity: Equity curve series

    Returns:
        Dictionary with drawdown statistics
    """
    dd_df = calculate_drawdown(equity)
    events = find_drawdown_events(equity, min_drawdown=0.02)

    if not events:
        return {
            "max_drawdown": 0.0,
            "avg_drawdown": 0.0,
            "max_duration_days": 0,
            "avg_duration_days": 0.0,
            "num_drawdowns": 0,
            "current_drawdown": abs(dd_df["drawdown"].iloc[-1]),
            "time_underwater_pct": 0.0,
        }

    drawdowns = [e.drawdown_pct for e in events]
    durations = [e.duration_days for e in events if e.duration_days]

    # Time spent underwater
    underwater_bars = (dd_df["drawdown"] < -0.01).sum()
    time_underwater = underwater_bars / len(dd_df) * 100

    return {
        "max_drawdown": max(drawdowns),
        "avg_drawdown": np.mean(drawdowns),
        "max_duration_days": max(durations) if durations else 0,
        "avg_duration_days": np.mean(durations) if durations else 0,
        "num_drawdowns": len(events),
        "current_drawdown": abs(dd_df["drawdown"].iloc[-1]),
        "time_underwater_pct": time_underwater,
        "longest_recovery_days": max([e.recovery_days for e in events if e.recovery_days], default=0),
    }


class DrawdownAnalyzer:
    """
    Comprehensive drawdown analysis.

    Provides detailed analysis of drawdown patterns and characteristics.
    """

    def __init__(self, equity: pd.Series) -> None:
        """
        Initialize the analyzer.

        Args:
            equity: Equity curve series
        """
        self.equity = equity
        self.dd_df = calculate_drawdown(equity)
        self.events = find_drawdown_events(equity)

    def get_worst_drawdowns(self, n: int = 5) -> list[DrawdownEvent]:
        """
        Get the N worst drawdowns.

        Args:
            n: Number of drawdowns to return

        Returns:
            List of worst drawdown events
        """
        sorted_events = sorted(
            self.events, key=lambda x: x.drawdown_pct, reverse=True
        )
        return sorted_events[:n]

    def get_longest_drawdowns(self, n: int = 5) -> list[DrawdownEvent]:
        """
        Get the N longest drawdowns.

        Args:
            n: Number of drawdowns to return

        Returns:
            List of longest drawdown events
        """
        sorted_events = sorted(
            self.events,
            key=lambda x: x.duration_days if x.duration_days else 0,
            reverse=True,
        )
        return sorted_events[:n]

    def get_drawdown_distribution(self) -> pd.DataFrame:
        """
        Get distribution of drawdown depths.

        Returns:
            DataFrame with drawdown distribution
        """
        if not self.events:
            return pd.DataFrame()

        depths = [e.drawdown_pct * 100 for e in self.events]

        bins = [0, 2, 5, 10, 15, 20, 30, 50, 100]
        labels = ["0-2%", "2-5%", "5-10%", "10-15%", "15-20%", "20-30%", "30-50%", "50%+"]

        dist = pd.cut(depths, bins=bins, labels=labels)
        counts = dist.value_counts().sort_index()

        return pd.DataFrame({
            "range": counts.index,
            "count": counts.values,
            "pct": counts.values / len(self.events) * 100,
        })

    def get_summary(self) -> dict:
        """
        Get summary statistics.

        Returns:
            Summary dictionary
        """
        return drawdown_statistics(self.equity)
