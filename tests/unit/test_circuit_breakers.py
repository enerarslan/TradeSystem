"""
Unit tests for circuit breaker system.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.risk.circuit_breakers import (
    CircuitBreakerManager,
    CircuitBreakerAction,
    CircuitBreakerState,
    MarketCircuitBreaker,
    PortfolioCircuitBreaker,
    VolatilityCircuitBreaker,
)


class TestMarketCircuitBreaker:
    """Tests for market-wide circuit breaker."""

    def test_no_trigger_small_drop(self):
        """Test no trigger on small market drop."""
        breaker = MarketCircuitBreaker()
        breaker.set_reference(100.0)

        action = breaker.check(97.0)  # 3% drop

        assert action is None

    def test_trigger_at_7_percent(self):
        """Test trigger at 7% drop threshold."""
        breaker = MarketCircuitBreaker()
        breaker.set_reference(100.0)

        action = breaker.check(92.0)  # 8% drop

        assert action == CircuitBreakerAction.PAUSE_15MIN

    def test_trigger_at_13_percent(self):
        """Test trigger at 13% drop threshold."""
        breaker = MarketCircuitBreaker()
        breaker.set_reference(100.0)

        # First breach 7%
        breaker.check(92.0)

        # Then breach 13%
        action = breaker.check(86.0)

        assert action == CircuitBreakerAction.PAUSE_15MIN

    def test_trigger_at_20_percent(self):
        """Test trigger at 20% drop threshold (day halt)."""
        breaker = MarketCircuitBreaker()
        breaker.set_reference(100.0)

        # Breach all thresholds
        breaker.check(92.0)  # 7%
        breaker.check(86.0)  # 13%
        action = breaker.check(79.0)  # 21%

        assert action == CircuitBreakerAction.HALT_DAY

    def test_no_double_trigger(self):
        """Test that same threshold doesn't trigger twice."""
        breaker = MarketCircuitBreaker()
        breaker.set_reference(100.0)

        action1 = breaker.check(92.0)  # First trigger
        action2 = breaker.check(91.0)  # Same level again

        assert action1 == CircuitBreakerAction.PAUSE_15MIN
        assert action2 is None

    def test_reference_reset(self):
        """Test that setting new reference clears triggered levels."""
        breaker = MarketCircuitBreaker()

        # Day 1
        breaker.set_reference(100.0)
        breaker.check(92.0)  # Trigger

        # Day 2 - new reference
        breaker.set_reference(95.0)
        action = breaker.check(88.0)  # Should trigger again

        assert action == CircuitBreakerAction.PAUSE_15MIN


class TestPortfolioCircuitBreaker:
    """Tests for portfolio-specific circuit breaker."""

    def test_daily_loss_limit(self):
        """Test daily loss limit trigger."""
        breaker = PortfolioCircuitBreaker(daily_loss_limit=0.03)

        # Start of day
        action = breaker.update(1000000)
        assert action is None

        # Loss of 3.5%
        action = breaker.update(965000)

        assert action == CircuitBreakerAction.HALT_DAY

    def test_rapid_loss_detection(self):
        """Test rapid loss detection."""
        breaker = PortfolioCircuitBreaker(
            rapid_loss_pct=0.05,
            rapid_loss_period_minutes=30,
        )

        # Build up history
        for i in range(10):
            action = breaker.update(1000000)

        # Sudden drop
        action = breaker.update(940000)

        assert action == CircuitBreakerAction.HALT_REVIEW

    def test_weekly_loss_limit(self):
        """Test weekly loss limit."""
        breaker = PortfolioCircuitBreaker(weekly_loss_limit=0.06)

        # Start value
        breaker.update(1000000)

        # Loss of 7%
        action = breaker.update(930000)

        assert action == CircuitBreakerAction.HALT_REVIEW

    def test_no_trigger_small_loss(self):
        """Test no trigger on small loss."""
        breaker = PortfolioCircuitBreaker(daily_loss_limit=0.03)

        breaker.update(1000000)
        action = breaker.update(990000)  # 1% loss

        assert action is None


class TestVolatilityCircuitBreaker:
    """Tests for volatility circuit breaker."""

    def test_position_reduction_elevated_vol(self):
        """Test position reduction when volatility is elevated."""
        breaker = VolatilityCircuitBreaker(
            vol_spike_threshold=3.0,
            reduce_threshold=2.0,
        )

        # Build baseline with normal volatility
        np.random.seed(42)
        for _ in range(30):
            breaker.update(np.random.normal(0, 0.01))

        # Spike in volatility
        action, scale = breaker.update(0.05)  # Large return

        # Should reduce positions
        assert scale < 1.0

    def test_halt_extreme_volatility(self):
        """Test halt on extreme volatility spike."""
        breaker = VolatilityCircuitBreaker(
            vol_spike_threshold=3.0,
        )

        # Build baseline
        for _ in range(30):
            breaker.update(np.random.normal(0, 0.005))

        # Extreme moves
        for _ in range(5):
            action, scale = breaker.update(0.10)

        # Should trigger halt
        assert action == CircuitBreakerAction.HALT_REVIEW or scale == 0.0


class TestCircuitBreakerManager:
    """Tests for circuit breaker manager."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = CircuitBreakerManager()

        assert manager.is_trading_allowed()
        assert manager.get_position_scale() == 1.0

    def test_check_all_no_trigger(self):
        """Test check_all with no triggers."""
        manager = CircuitBreakerManager()
        manager.market_breaker.set_reference(100.0)

        state = manager.check_all(
            market_level=99.0,
            portfolio_value=1000000,
            returns=0.001,
        )

        assert not state.is_halted
        assert state.position_scale == 1.0

    def test_save_and_load_state(self):
        """Test state persistence."""
        manager = CircuitBreakerManager()
        manager.state.is_halted = True
        manager.state.halt_reason = "Test halt"
        manager.state.position_scale = 0.5

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            manager.save_state(filepath)

            # Create new manager and load
            new_manager = CircuitBreakerManager()
            loaded = new_manager.load_state(filepath)

            assert loaded
            assert new_manager.state.is_halted
            assert new_manager.state.halt_reason == "Test halt"
            assert new_manager.state.position_scale == 0.5
        finally:
            filepath.unlink()

    def test_manual_resume(self):
        """Test manual trading resume."""
        manager = CircuitBreakerManager()
        manager.state.is_halted = True

        manager.manual_resume()

        assert not manager.state.is_halted
        assert manager.state.position_scale == 0.5  # Resumes at 50%

    def test_status_report(self):
        """Test status report generation."""
        manager = CircuitBreakerManager()

        report = manager.get_status_report()

        assert "is_trading_allowed" in report
        assert "position_scale" in report
        assert "is_halted" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
