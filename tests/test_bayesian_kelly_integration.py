"""
Unit Tests for Bayesian Kelly Integration
==========================================

Tests the integration of BayesianKellySizer with the trading loop.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.risk.bayesian_kelly import (
    BayesianKellySizer,
    TradeOutcome,
    BayesianEstimate,
    KellyResult
)


# =============================================================================
# BAYESIAN KELLY SIZER TESTS
# =============================================================================

class TestBayesianKellySizer:
    """Tests for BayesianKellySizer"""

    def test_initialization_defaults(self):
        """Test default initialization"""
        sizer = BayesianKellySizer()

        assert sizer.prior_wins == 2.0
        assert sizer.prior_losses == 2.0
        assert sizer.kelly_fraction == 0.25
        assert sizer.max_position_pct == 0.20
        assert sizer.min_observations == 20

    def test_calculate_kelly_no_observations(self):
        """Test Kelly calculation with no historical data"""
        sizer = BayesianKellySizer()

        result = sizer.calculate_kelly(
            symbol="AAPL",
            strategy="momentum",
            signal_strength=0.8
        )

        # With no observations and uninformative prior (2,2),
        # win rate estimate should be 0.5
        assert isinstance(result, KellyResult)
        assert result.win_rate_estimate.mean == 0.5
        assert result.win_rate_estimate.n_observations == 0
        # Position should be small due to uncertainty
        assert result.fractional_kelly < result.max_position_pct

    def test_record_outcome_updates_statistics(self):
        """Test that recording outcomes updates internal state"""
        sizer = BayesianKellySizer()

        # Record some wins
        for _ in range(10):
            outcome = TradeOutcome(
                symbol="AAPL",
                strategy="momentum",
                win=True,
                profit_pct=0.02
            )
            sizer.record_outcome(outcome)

        # Record some losses
        for _ in range(5):
            outcome = TradeOutcome(
                symbol="AAPL",
                strategy="momentum",
                win=False,
                profit_pct=-0.01
            )
            sizer.record_outcome(outcome)

        stats = sizer.get_statistics("AAPL", "momentum")

        assert stats['n_trades'] == 15
        assert stats['wins'] == 10
        assert stats['losses'] == 5
        assert abs(stats['win_rate'] - 10/15) < 0.01

    def test_win_rate_estimate_with_observations(self):
        """Test that win rate estimate updates with observations"""
        sizer = BayesianKellySizer()

        # Record mostly wins
        for _ in range(18):
            sizer.record_outcome(TradeOutcome(
                symbol="AAPL", strategy="ml", win=True, profit_pct=0.02
            ))
        for _ in range(2):
            sizer.record_outcome(TradeOutcome(
                symbol="AAPL", strategy="ml", win=False, profit_pct=-0.01
            ))

        result = sizer.calculate_kelly(
            symbol="AAPL",
            strategy="ml",
            signal_strength=1.0
        )

        # With 18 wins and 2 losses, plus prior (2,2):
        # Posterior alpha = 2 + 18 = 20
        # Posterior beta = 2 + 2 = 4
        # Expected win rate = 20 / 24 ≈ 0.833
        assert result.win_rate_estimate.mean > 0.8
        assert result.win_rate_estimate.n_observations == 20

    def test_kelly_scales_with_signal_strength(self):
        """Test that Kelly fraction scales with signal strength"""
        sizer = BayesianKellySizer()

        # Add some observations to reduce uncertainty
        for _ in range(25):
            sizer.record_outcome(TradeOutcome(
                symbol="AAPL", strategy="test", win=True, profit_pct=0.03
            ))

        result_full = sizer.calculate_kelly("AAPL", "test", signal_strength=1.0)
        result_half = sizer.calculate_kelly("AAPL", "test", signal_strength=0.5)

        # Half signal strength should give roughly half position
        assert result_half.fractional_kelly < result_full.fractional_kelly

    def test_max_position_pct_enforced(self):
        """Test that maximum position percentage is enforced"""
        sizer = BayesianKellySizer(max_position_pct=0.10)

        # Record many wins to get high Kelly fraction
        for _ in range(100):
            sizer.record_outcome(TradeOutcome(
                symbol="AAPL", strategy="test", win=True, profit_pct=0.05
            ))

        result = sizer.calculate_kelly("AAPL", "test", signal_strength=1.0)

        assert result.fractional_kelly <= 0.10

    def test_min_observations_ramp(self):
        """Test that position size ramps up with observations"""
        sizer = BayesianKellySizer(min_observations=20)

        # With just 5 observations (less than min)
        for _ in range(4):
            sizer.record_outcome(TradeOutcome(
                symbol="AAPL", strategy="test", win=True, profit_pct=0.02
            ))
        sizer.record_outcome(TradeOutcome(
            symbol="AAPL", strategy="test", win=False, profit_pct=-0.01
        ))

        result_5 = sizer.calculate_kelly("AAPL", "test", signal_strength=1.0)

        # Add more to reach min_observations
        for _ in range(20):
            sizer.record_outcome(TradeOutcome(
                symbol="AAPL", strategy="test", win=True, profit_pct=0.02
            ))

        result_25 = sizer.calculate_kelly("AAPL", "test", signal_strength=1.0)

        # With more observations, position should be larger
        # (assuming both have positive expected Kelly)
        assert result_5.fractional_kelly < result_25.fractional_kelly

    def test_separate_tracking_per_symbol_strategy(self):
        """Test that statistics are tracked separately per symbol/strategy"""
        sizer = BayesianKellySizer()

        # Record for AAPL momentum
        for _ in range(10):
            sizer.record_outcome(TradeOutcome(
                symbol="AAPL", strategy="momentum", win=True, profit_pct=0.02
            ))

        # Record for MSFT momentum
        for _ in range(10):
            sizer.record_outcome(TradeOutcome(
                symbol="MSFT", strategy="momentum", win=False, profit_pct=-0.01
            ))

        aapl_stats = sizer.get_statistics("AAPL", "momentum")
        msft_stats = sizer.get_statistics("MSFT", "momentum")

        assert aapl_stats['win_rate'] == 1.0  # All wins
        assert msft_stats['win_rate'] == 0.0  # All losses

    def test_reset_beliefs(self):
        """Test that beliefs can be reset"""
        sizer = BayesianKellySizer()

        # Record some outcomes
        for _ in range(10):
            sizer.record_outcome(TradeOutcome(
                symbol="AAPL", strategy="test", win=True, profit_pct=0.02
            ))

        stats_before = sizer.get_statistics("AAPL", "test")
        assert stats_before['n_trades'] == 10

        # Reset
        sizer.reset_beliefs("AAPL", "test")

        stats_after = sizer.get_statistics("AAPL", "test")
        assert stats_after['n_trades'] == 0

    def test_uncertainty_penalty(self):
        """Test that high uncertainty leads to smaller positions"""
        sizer = BayesianKellySizer()

        # Very few observations = high uncertainty
        sizer.record_outcome(TradeOutcome(
            symbol="AAPL", strategy="uncertain", win=True, profit_pct=0.05
        ))

        result = sizer.calculate_kelly("AAPL", "uncertain", signal_strength=1.0)

        # Should have significant uncertainty penalty
        assert result.uncertainty_penalty > 0

    def test_get_position_size_in_shares(self):
        """Test calculating position size in shares"""
        sizer = BayesianKellySizer()

        # Add some observations
        for _ in range(30):
            sizer.record_outcome(TradeOutcome(
                symbol="AAPL", strategy="test", win=True, profit_pct=0.02
            ))

        shares, result = sizer.get_position_size(
            symbol="AAPL",
            strategy="test",
            portfolio_value=100000,
            current_price=150.0,
            signal_strength=1.0
        )

        assert shares >= 0
        assert isinstance(shares, int)
        assert result.fractional_kelly > 0


# =============================================================================
# TRADE OUTCOME TESTS
# =============================================================================

class TestTradeOutcome:
    """Tests for TradeOutcome dataclass"""

    def test_trade_outcome_creation(self):
        """Test creating a trade outcome"""
        outcome = TradeOutcome(
            symbol="AAPL",
            strategy="momentum",
            win=True,
            profit_pct=0.025
        )

        assert outcome.symbol == "AAPL"
        assert outcome.strategy == "momentum"
        assert outcome.win is True
        assert outcome.profit_pct == 0.025
        assert isinstance(outcome.timestamp, datetime)

    def test_trade_outcome_timestamp_default(self):
        """Test that timestamp defaults to now"""
        before = datetime.now()
        outcome = TradeOutcome(
            symbol="AAPL", strategy="test", win=True, profit_pct=0.01
        )
        after = datetime.now()

        assert before <= outcome.timestamp <= after


# =============================================================================
# KELLY RESULT TESTS
# =============================================================================

class TestKellyResult:
    """Tests for KellyResult dataclass"""

    def test_kelly_result_to_dict(self):
        """Test KellyResult.to_dict()"""
        win_rate_est = BayesianEstimate(
            mean=0.6, std=0.05, lower_95=0.5, upper_95=0.7, n_observations=100
        )
        payoff_est = BayesianEstimate(
            mean=1.5, std=0.1, lower_95=1.3, upper_95=1.7, n_observations=100
        )
        edge_est = BayesianEstimate(
            mean=0.05, std=0.02, lower_95=0.01, upper_95=0.09, n_observations=100
        )

        result = KellyResult(
            kelly_fraction=0.15,
            bayesian_fraction=0.12,
            fractional_kelly=0.05,
            win_rate_estimate=win_rate_est,
            payoff_ratio_estimate=payoff_est,
            edge_estimate=edge_est,
            uncertainty_penalty=0.2,
            max_position_pct=0.05
        )

        d = result.to_dict()

        assert d['kelly_fraction'] == 0.15
        assert d['bayesian_fraction'] == 0.12
        assert d['fractional_kelly'] == 0.05
        assert d['win_rate'] == 0.6
        assert d['uncertainty_penalty'] == 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
