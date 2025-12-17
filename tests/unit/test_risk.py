"""
Unit tests for risk management modules.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.risk.position_sizing import (
    PositionSizer,
    fixed_fraction,
    kelly_criterion,
    volatility_target,
    risk_parity_weights,
)
from src.risk.var_models import VaRCalculator, calculate_var, calculate_cvar
from src.risk.drawdown import (
    DrawdownController,
    calculate_drawdown,
    calculate_max_drawdown,
)


@pytest.fixture
def sample_returns():
    """Create sample returns for testing."""
    np.random.seed(42)
    n = 500
    returns = pd.Series(
        np.random.normal(0.0005, 0.02, n),
        index=pd.date_range(start="2023-01-01", periods=n, freq="D"),
    )
    return returns


@pytest.fixture
def sample_returns_df():
    """Create sample returns DataFrame for multiple assets."""
    np.random.seed(42)
    n = 500

    dates = pd.date_range(start="2023-01-01", periods=n, freq="D")

    # Create correlated returns
    returns = pd.DataFrame({
        "AAPL": np.random.normal(0.001, 0.02, n),
        "GOOGL": np.random.normal(0.0008, 0.025, n),
        "MSFT": np.random.normal(0.0009, 0.018, n),
    }, index=dates)

    return returns


class TestPositionSizing:
    """Tests for position sizing."""

    def test_fixed_fraction(self):
        """Test fixed fractional sizing."""
        capital = 100000
        fraction = 0.02

        position = fixed_fraction(capital, fraction)

        assert position == 2000
        assert position == capital * fraction

    def test_kelly_criterion_positive(self):
        """Test Kelly criterion with positive edge."""
        win_rate = 0.55
        avg_win = 0.02
        avg_loss = 0.01

        kelly = kelly_criterion(win_rate, avg_win, avg_loss)

        assert kelly > 0
        assert kelly <= 0.5  # Capped

    def test_kelly_criterion_negative(self):
        """Test Kelly criterion with negative edge."""
        win_rate = 0.3
        avg_win = 0.01
        avg_loss = 0.02

        kelly = kelly_criterion(win_rate, avg_win, avg_loss)

        assert kelly == 0  # No betting with negative edge

    def test_volatility_target(self):
        """Test volatility targeting."""
        capital = 100000
        asset_vol = 0.30  # 30% volatility
        target_vol = 0.15  # 15% target

        position = volatility_target(capital, asset_vol, target_vol)

        # Should be half-sized due to double volatility
        assert position == capital * 0.5


class TestRiskParity:
    """Tests for risk parity weights."""

    def test_risk_parity_sum_to_one(self, sample_returns_df):
        """Test that risk parity weights sum to 1."""
        weights = risk_parity_weights(sample_returns_df)

        assert abs(weights.sum() - 1.0) < 1e-6

    def test_risk_parity_all_positive(self, sample_returns_df):
        """Test that risk parity weights are all positive."""
        weights = risk_parity_weights(sample_returns_df)

        assert (weights > 0).all()

    def test_risk_parity_equal_risk_contribution(self, sample_returns_df):
        """Test that risk contributions are approximately equal."""
        weights = risk_parity_weights(sample_returns_df)

        # Calculate risk contributions
        cov = sample_returns_df.cov()
        port_var = weights.values @ cov.values @ weights.values
        mrc = cov.values @ weights.values
        rc = weights.values * mrc / np.sqrt(port_var)
        rc_pct = rc / rc.sum()

        # Should be approximately equal (1/3 each)
        target = 1 / len(weights)
        assert all(abs(r - target) < 0.1 for r in rc_pct)


class TestVaRCalculator:
    """Tests for VaR calculations."""

    def test_historical_var(self, sample_returns):
        """Test historical VaR."""
        calculator = VaRCalculator(confidence_level=0.95, method="historical")
        var = calculator.calculate_var(sample_returns)

        assert var > 0
        # VaR should be roughly around 2 std devs
        assert var < sample_returns.std() * 5

    def test_parametric_var(self, sample_returns):
        """Test parametric VaR."""
        calculator = VaRCalculator(confidence_level=0.95, method="parametric")
        var = calculator.calculate_var(sample_returns)

        assert var > 0

    def test_monte_carlo_var(self, sample_returns):
        """Test Monte Carlo VaR."""
        calculator = VaRCalculator(confidence_level=0.95, method="monte_carlo")
        var = calculator.calculate_var(sample_returns)

        assert var > 0

    def test_cvar_greater_than_var(self, sample_returns):
        """Test that CVaR >= VaR."""
        calculator = VaRCalculator(confidence_level=0.95)
        var = calculator.calculate_var(sample_returns)
        cvar = calculator.calculate_cvar(sample_returns)

        assert cvar >= var

    def test_var_with_different_confidence(self, sample_returns):
        """Test VaR at different confidence levels."""
        calc_95 = VaRCalculator(confidence_level=0.95)
        calc_99 = VaRCalculator(confidence_level=0.99)

        var_95 = calc_95.calculate_var(sample_returns)
        var_99 = calc_99.calculate_var(sample_returns)

        # 99% VaR should be greater than 95% VaR
        assert var_99 > var_95


class TestDrawdownController:
    """Tests for drawdown controller."""

    def test_normal_operation(self):
        """Test normal operation without drawdown."""
        controller = DrawdownController(
            reduce_at_drawdown=0.10,
            close_all_at_drawdown=0.20,
        )

        # Initial update
        status = controller.update(1000000)
        assert status["action"] == "none"
        assert status["scale_factor"] == 1.0

        # Slight gain
        status = controller.update(1010000)
        assert status["action"] == "none"
        assert status["drawdown"] == 0.0

    def test_reduce_at_threshold(self):
        """Test position reduction at drawdown threshold."""
        controller = DrawdownController(
            reduce_at_drawdown=0.10,
            reduce_by_pct=0.50,
        )

        # Set peak
        controller.update(1000000)

        # Drop 11%
        status = controller.update(890000)

        assert status["action"] == "reduce"
        assert status["scale_factor"] == 0.5

    def test_close_all_at_threshold(self):
        """Test closing all positions at extreme drawdown."""
        controller = DrawdownController(
            close_all_at_drawdown=0.20,
        )

        # Set peak
        controller.update(1000000)

        # Drop 21%
        status = controller.update(790000)

        assert status["action"] == "close_all"
        assert status["scale_factor"] == 0.0


class TestDrawdownCalculation:
    """Tests for drawdown calculations."""

    def test_calculate_drawdown(self, sample_returns):
        """Test drawdown calculation."""
        equity = (1 + sample_returns).cumprod() * 100000

        dd_df = calculate_drawdown(equity)

        assert "drawdown" in dd_df.columns
        assert "peak" in dd_df.columns
        assert (dd_df["drawdown"] <= 0).all()

    def test_calculate_max_drawdown(self, sample_returns):
        """Test max drawdown calculation."""
        equity = (1 + sample_returns).cumprod() * 100000

        max_dd = calculate_max_drawdown(equity)

        assert max_dd >= 0
        assert max_dd <= 1

    def test_drawdown_zero_for_increasing_equity(self):
        """Test that drawdown is 0 for continuously increasing equity."""
        equity = pd.Series([100, 101, 102, 103, 104, 105])

        dd_df = calculate_drawdown(equity)

        assert (dd_df["drawdown"] == 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
