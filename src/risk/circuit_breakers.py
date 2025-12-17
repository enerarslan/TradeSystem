"""
Circuit Breakers for AlphaTrade System.

JPMorgan-level implementation of automated trading halts based on:
- Market-wide conditions (e.g., flash crashes)
- Portfolio-specific thresholds
- Rapid loss detection
- Volatility spikes

This module provides active monitoring and enforcement of circuit breaker rules
that were previously only defined in configuration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger


class CircuitBreakerAction(Enum):
    """Actions that can be taken when a circuit breaker is triggered."""

    NONE = "none"
    ALERT = "alert"
    PAUSE_15MIN = "pause_15min"
    PAUSE_30MIN = "pause_30min"
    PAUSE_1HR = "pause_1hr"
    HALT_DAY = "halt_day"
    HALT_REVIEW = "halt_review"
    REDUCE_50 = "reduce_50"
    REDUCE_75 = "reduce_75"
    FLATTEN = "flatten"


@dataclass
class CircuitBreakerEvent:
    """Record of a circuit breaker trigger event."""

    timestamp: datetime
    breaker_name: str
    trigger_value: float
    threshold: float
    action: CircuitBreakerAction
    message: str
    resolved: bool = False
    resolution_time: datetime | None = None


@dataclass
class CircuitBreakerState:
    """Current state of the circuit breaker system."""

    is_halted: bool = False
    halt_reason: str | None = None
    halt_until: datetime | None = None
    position_scale: float = 1.0  # 1.0 = full positions, 0.0 = closed
    events: list[CircuitBreakerEvent] = field(default_factory=list)
    last_check: datetime | None = None


class MarketCircuitBreaker:
    """
    Market-wide circuit breaker monitoring.

    Monitors market indices for significant drops and triggers
    trading halts similar to NYSE circuit breakers.
    """

    def __init__(
        self,
        thresholds: list[dict] | None = None,
    ) -> None:
        """
        Initialize market circuit breaker.

        Args:
            thresholds: List of threshold configs with 'threshold' and 'action' keys
        """
        self.thresholds = thresholds or [
            {"threshold": -0.07, "action": "pause_15min"},
            {"threshold": -0.13, "action": "pause_15min"},
            {"threshold": -0.20, "action": "halt_day"},
        ]

        self._reference_level: float | None = None
        self._reference_time: datetime | None = None
        self._triggered_levels: set[float] = set()

    def set_reference(self, level: float) -> None:
        """Set the reference level (e.g., previous close)."""
        self._reference_level = level
        self._reference_time = datetime.now()
        self._triggered_levels.clear()
        logger.debug(f"Market circuit breaker reference set: {level}")

    def check(self, current_level: float) -> CircuitBreakerAction | None:
        """
        Check if circuit breaker should be triggered.

        Args:
            current_level: Current market level

        Returns:
            Action to take, or None if no trigger
        """
        if self._reference_level is None or self._reference_level == 0:
            return None

        change_pct = (current_level - self._reference_level) / self._reference_level

        for threshold_config in sorted(self.thresholds, key=lambda x: x["threshold"]):
            threshold = threshold_config["threshold"]

            if change_pct <= threshold and threshold not in self._triggered_levels:
                self._triggered_levels.add(threshold)
                action_str = threshold_config["action"]
                action = CircuitBreakerAction(action_str)

                logger.warning(
                    f"MARKET CIRCUIT BREAKER TRIGGERED: "
                    f"Change {change_pct:.2%} breached threshold {threshold:.2%}. "
                    f"Action: {action.value}"
                )

                return action

        return None


class PortfolioCircuitBreaker:
    """
    Portfolio-specific circuit breaker for rapid loss detection.

    Monitors portfolio P&L and triggers halts on rapid losses.
    """

    def __init__(
        self,
        rapid_loss_pct: float = 0.05,
        rapid_loss_period_minutes: int = 30,
        daily_loss_limit: float = 0.03,
        weekly_loss_limit: float = 0.06,
    ) -> None:
        """
        Initialize portfolio circuit breaker.

        Args:
            rapid_loss_pct: Percentage loss to trigger halt
            rapid_loss_period_minutes: Time window for rapid loss detection
            daily_loss_limit: Maximum daily loss before halt
            weekly_loss_limit: Maximum weekly loss before halt
        """
        self.rapid_loss_pct = rapid_loss_pct
        self.rapid_loss_period_minutes = rapid_loss_period_minutes
        self.daily_loss_limit = daily_loss_limit
        self.weekly_loss_limit = weekly_loss_limit

        # P&L tracking
        self._pnl_history: list[tuple[datetime, float]] = []
        self._daily_start_value: float | None = None
        self._weekly_start_value: float | None = None
        self._last_daily_reset: datetime | None = None
        self._last_weekly_reset: datetime | None = None

    def update(self, portfolio_value: float) -> CircuitBreakerAction | None:
        """
        Update with current portfolio value and check for triggers.

        Args:
            portfolio_value: Current portfolio value

        Returns:
            Action to take, or None if no trigger
        """
        now = datetime.now()

        # Initialize reference values
        if self._daily_start_value is None:
            self._daily_start_value = portfolio_value
            self._last_daily_reset = now

        if self._weekly_start_value is None:
            self._weekly_start_value = portfolio_value
            self._last_weekly_reset = now

        # Reset daily/weekly as needed
        if self._last_daily_reset and now.date() > self._last_daily_reset.date():
            self._daily_start_value = portfolio_value
            self._last_daily_reset = now
            logger.debug("Daily P&L counter reset")

        if self._last_weekly_reset and (now - self._last_weekly_reset).days >= 7:
            self._weekly_start_value = portfolio_value
            self._last_weekly_reset = now
            logger.debug("Weekly P&L counter reset")

        # Record P&L history
        self._pnl_history.append((now, portfolio_value))

        # Cleanup old history
        cutoff = now - timedelta(hours=2)
        self._pnl_history = [(t, v) for t, v in self._pnl_history if t > cutoff]

        # Check rapid loss
        rapid_action = self._check_rapid_loss(portfolio_value, now)
        if rapid_action:
            return rapid_action

        # Check daily loss
        if self._daily_start_value and self._daily_start_value > 0:
            daily_loss = (self._daily_start_value - portfolio_value) / self._daily_start_value
            if daily_loss >= self.daily_loss_limit:
                logger.warning(
                    f"DAILY LOSS LIMIT BREACHED: {daily_loss:.2%} >= {self.daily_loss_limit:.2%}"
                )
                return CircuitBreakerAction.HALT_DAY

        # Check weekly loss
        if self._weekly_start_value and self._weekly_start_value > 0:
            weekly_loss = (self._weekly_start_value - portfolio_value) / self._weekly_start_value
            if weekly_loss >= self.weekly_loss_limit:
                logger.warning(
                    f"WEEKLY LOSS LIMIT BREACHED: {weekly_loss:.2%} >= {self.weekly_loss_limit:.2%}"
                )
                return CircuitBreakerAction.HALT_REVIEW

        return None

    def _check_rapid_loss(
        self,
        current_value: float,
        now: datetime,
    ) -> CircuitBreakerAction | None:
        """Check for rapid loss within the time window."""
        cutoff = now - timedelta(minutes=self.rapid_loss_period_minutes)

        # Find value at beginning of window
        window_history = [(t, v) for t, v in self._pnl_history if t >= cutoff]

        if len(window_history) < 2:
            return None

        start_value = window_history[0][1]

        if start_value > 0:
            loss = (start_value - current_value) / start_value

            if loss >= self.rapid_loss_pct:
                logger.warning(
                    f"RAPID LOSS DETECTED: {loss:.2%} in {self.rapid_loss_period_minutes} minutes"
                )
                return CircuitBreakerAction.HALT_REVIEW

        return None


class VolatilityCircuitBreaker:
    """
    Volatility-based circuit breaker.

    Reduces position sizes or halts trading when volatility exceeds thresholds.
    """

    def __init__(
        self,
        vol_spike_threshold: float = 3.0,  # Multiple of normal vol
        lookback_period: int = 20,
        reduce_threshold: float = 2.0,
    ) -> None:
        """
        Initialize volatility circuit breaker.

        Args:
            vol_spike_threshold: Multiplier for halt trigger
            lookback_period: Period for baseline volatility calculation
            reduce_threshold: Multiplier for position reduction
        """
        self.vol_spike_threshold = vol_spike_threshold
        self.lookback_period = lookback_period
        self.reduce_threshold = reduce_threshold

        self._returns_history: list[float] = []
        self._baseline_vol: float | None = None

    def update(self, returns: float) -> tuple[CircuitBreakerAction | None, float]:
        """
        Update with new return and check for triggers.

        Args:
            returns: Latest period return

        Returns:
            Tuple of (action, position_scale)
        """
        self._returns_history.append(returns)

        # Keep limited history
        if len(self._returns_history) > self.lookback_period * 3:
            self._returns_history = self._returns_history[-self.lookback_period * 3:]

        if len(self._returns_history) < self.lookback_period:
            return None, 1.0

        # Calculate baseline volatility from older data
        baseline_returns = self._returns_history[:-5]  # Exclude recent
        if len(baseline_returns) >= self.lookback_period:
            self._baseline_vol = np.std(baseline_returns)

        if self._baseline_vol is None or self._baseline_vol == 0:
            return None, 1.0

        # Calculate recent volatility
        recent_vol = np.std(self._returns_history[-5:])
        vol_ratio = recent_vol / self._baseline_vol

        # Check thresholds
        if vol_ratio >= self.vol_spike_threshold:
            logger.warning(
                f"VOLATILITY SPIKE: Current vol {recent_vol:.4f} is {vol_ratio:.1f}x baseline"
            )
            return CircuitBreakerAction.HALT_REVIEW, 0.0

        elif vol_ratio >= self.reduce_threshold:
            scale = 1.0 - (vol_ratio - self.reduce_threshold) / (self.vol_spike_threshold - self.reduce_threshold)
            scale = max(0.25, min(1.0, scale))
            logger.info(f"Volatility elevated: reducing positions to {scale:.0%}")
            return CircuitBreakerAction.REDUCE_50, scale

        return None, 1.0


class CircuitBreakerManager:
    """
    Central manager for all circuit breakers.

    Coordinates multiple circuit breaker types and manages overall trading state.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        alert_callback: Callable[[CircuitBreakerEvent], None] | None = None,
    ) -> None:
        """
        Initialize circuit breaker manager.

        Args:
            config: Configuration dictionary
            alert_callback: Function to call on circuit breaker events
        """
        config = config or {}

        # Initialize individual breakers
        market_config = config.get("market_drop", [])
        self.market_breaker = MarketCircuitBreaker(thresholds=market_config)

        portfolio_config = config.get("portfolio", {})
        self.portfolio_breaker = PortfolioCircuitBreaker(
            rapid_loss_pct=portfolio_config.get("rapid_loss_pct", 0.05),
            rapid_loss_period_minutes=portfolio_config.get("rapid_loss_period_minutes", 30),
            daily_loss_limit=config.get("loss_limits", {}).get("daily_loss_limit", 0.03),
            weekly_loss_limit=config.get("loss_limits", {}).get("weekly_loss_limit", 0.06),
        )

        vol_config = config.get("volatility", {})
        self.volatility_breaker = VolatilityCircuitBreaker(
            vol_spike_threshold=vol_config.get("spike_threshold", 3.0),
            reduce_threshold=vol_config.get("reduce_threshold", 2.0),
        )

        # State
        self.state = CircuitBreakerState()
        self._alert_callback = alert_callback

        # Pause tracking
        self._pause_end_time: datetime | None = None

        logger.info("Circuit breaker manager initialized")

    def check_all(
        self,
        market_level: float | None = None,
        portfolio_value: float | None = None,
        returns: float | None = None,
    ) -> CircuitBreakerState:
        """
        Check all circuit breakers.

        Args:
            market_level: Current market index level
            portfolio_value: Current portfolio value
            returns: Latest period returns

        Returns:
            Current circuit breaker state
        """
        now = datetime.now()
        self.state.last_check = now

        # Check if pause has ended
        if self._pause_end_time and now >= self._pause_end_time:
            self._resume_trading()

        # Skip checks if already halted
        if self.state.is_halted and self.state.halt_until and now < self.state.halt_until:
            return self.state

        # Check market circuit breaker
        if market_level is not None:
            market_action = self.market_breaker.check(market_level)
            if market_action:
                self._handle_action(market_action, "Market", market_level)

        # Check portfolio circuit breaker
        if portfolio_value is not None:
            portfolio_action = self.portfolio_breaker.update(portfolio_value)
            if portfolio_action:
                self._handle_action(portfolio_action, "Portfolio", portfolio_value)

        # Check volatility circuit breaker
        if returns is not None:
            vol_action, scale = self.volatility_breaker.update(returns)
            if vol_action:
                self._handle_action(vol_action, "Volatility", returns)
            self.state.position_scale = min(self.state.position_scale, scale)

        return self.state

    def _handle_action(
        self,
        action: CircuitBreakerAction,
        breaker_name: str,
        trigger_value: float,
    ) -> None:
        """Handle a circuit breaker trigger."""
        now = datetime.now()

        event = CircuitBreakerEvent(
            timestamp=now,
            breaker_name=breaker_name,
            trigger_value=trigger_value,
            threshold=0.0,  # Would need to pass this from breaker
            action=action,
            message=f"{breaker_name} circuit breaker triggered: {action.value}",
        )

        self.state.events.append(event)

        # Apply action
        if action == CircuitBreakerAction.PAUSE_15MIN:
            self._pause_trading(timedelta(minutes=15))
        elif action == CircuitBreakerAction.PAUSE_30MIN:
            self._pause_trading(timedelta(minutes=30))
        elif action == CircuitBreakerAction.PAUSE_1HR:
            self._pause_trading(timedelta(hours=1))
        elif action == CircuitBreakerAction.HALT_DAY:
            self._halt_trading("End of day halt", self._get_next_market_open())
        elif action == CircuitBreakerAction.HALT_REVIEW:
            self._halt_trading("Manual review required", None)
        elif action == CircuitBreakerAction.REDUCE_50:
            self.state.position_scale = min(self.state.position_scale, 0.5)
        elif action == CircuitBreakerAction.REDUCE_75:
            self.state.position_scale = min(self.state.position_scale, 0.25)
        elif action == CircuitBreakerAction.FLATTEN:
            self.state.position_scale = 0.0

        # Alert callback
        if self._alert_callback:
            try:
                self._alert_callback(event)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        logger.warning(f"Circuit breaker event: {event.message}")

    def _pause_trading(self, duration: timedelta) -> None:
        """Pause trading for a duration."""
        self.state.is_halted = True
        self._pause_end_time = datetime.now() + duration
        self.state.halt_until = self._pause_end_time
        self.state.halt_reason = f"Paused for {duration}"
        logger.warning(f"Trading paused until {self._pause_end_time}")

    def _halt_trading(self, reason: str, until: datetime | None) -> None:
        """Halt trading (requires manual intervention if until is None)."""
        self.state.is_halted = True
        self.state.halt_reason = reason
        self.state.halt_until = until
        logger.error(f"TRADING HALTED: {reason}")

    def _resume_trading(self) -> None:
        """Resume trading after pause."""
        self.state.is_halted = False
        self.state.halt_reason = None
        self.state.halt_until = None
        self._pause_end_time = None
        logger.info("Trading resumed")

    def manual_resume(self) -> None:
        """Manually resume trading after halt."""
        if self.state.is_halted:
            self._resume_trading()
            self.state.position_scale = 0.5  # Resume at reduced size
            logger.info("Manual trading resume - starting at 50% position size")

    def _get_next_market_open(self) -> datetime:
        """Get next market open time (simplified)."""
        now = datetime.now()
        # Simple implementation - next day at 9:30 AM
        next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        if now.hour >= 9:
            next_open += timedelta(days=1)
        return next_open

    def get_position_scale(self) -> float:
        """Get current position scaling factor."""
        if self.state.is_halted:
            return 0.0
        return self.state.position_scale

    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed."""
        return not self.state.is_halted

    def save_state(self, filepath: str | Path) -> None:
        """Save circuit breaker state to file."""
        filepath = Path(filepath)

        state_dict = {
            "is_halted": self.state.is_halted,
            "halt_reason": self.state.halt_reason,
            "halt_until": self.state.halt_until.isoformat() if self.state.halt_until else None,
            "position_scale": self.state.position_scale,
            "last_check": self.state.last_check.isoformat() if self.state.last_check else None,
            "events_count": len(self.state.events),
            "recent_events": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "breaker_name": e.breaker_name,
                    "action": e.action.value,
                    "message": e.message,
                }
                for e in self.state.events[-10:]  # Last 10 events
            ],
        }

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(state_dict, f, indent=2)

        logger.debug(f"Circuit breaker state saved to {filepath}")

    def load_state(self, filepath: str | Path) -> bool:
        """Load circuit breaker state from file."""
        filepath = Path(filepath)

        if not filepath.exists():
            return False

        try:
            with open(filepath, "r") as f:
                state_dict = json.load(f)

            self.state.is_halted = state_dict.get("is_halted", False)
            self.state.halt_reason = state_dict.get("halt_reason")
            self.state.position_scale = state_dict.get("position_scale", 1.0)

            halt_until_str = state_dict.get("halt_until")
            if halt_until_str:
                self.state.halt_until = datetime.fromisoformat(halt_until_str)

            logger.info(f"Circuit breaker state loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load circuit breaker state: {e}")
            return False

    def get_status_report(self) -> dict[str, Any]:
        """Get comprehensive status report."""
        return {
            "is_trading_allowed": self.is_trading_allowed(),
            "is_halted": self.state.is_halted,
            "halt_reason": self.state.halt_reason,
            "halt_until": self.state.halt_until.isoformat() if self.state.halt_until else None,
            "position_scale": self.state.position_scale,
            "total_events": len(self.state.events),
            "last_check": self.state.last_check.isoformat() if self.state.last_check else None,
        }
