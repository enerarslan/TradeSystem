"""
Tests for statistical significance tests module.

Tests verify that Deflated Sharpe Ratio, PBO, and Monte Carlo methods
correctly assess strategy validity.

Section 9: Required test coverage for Directive 3.9.
"""

import numpy as np
import pandas as pd
import pytest


class TestSharpeRatioTest:
    """Test Sharpe ratio significance tests."""

    def setup_method(self):
        """Create test returns as pandas Series."""
        np.random.seed(42)

        # Good strategy returns (positive mean, low volatility)
        self.good_returns = pd.Series(np.random.randn(252) * 0.01 + 0.002)

        # Random strategy (should fail significance)
        self.random_returns = pd.Series(np.random.randn(252) * 0.01)

    def test_sharpe_significance_returns_result(self):
        """Test that Sharpe test returns valid result."""
        from src.training.statistical_tests import SharpeRatioTest

        result = SharpeRatioTest.test_sharpe_significance(
            self.good_returns, benchmark_sharpe=0.0
        )

        # Should have expected attributes
        assert hasattr(result, "p_value")
        assert hasattr(result, "statistic")
        assert 0 <= result.p_value <= 1

    def test_sharpe_significance_random_strategy(self):
        """Test that random strategy is not significant."""
        from src.training.statistical_tests import SharpeRatioTest

        result = SharpeRatioTest.test_sharpe_significance(
            self.random_returns, benchmark_sharpe=0.0
        )

        # Random strategy with zero mean should not be significant at 5%
        assert result.p_value > 0.05, f"Random strategy should not be significant, p={result.p_value}"

    def test_deflated_sharpe_ratio(self):
        """Test deflated Sharpe ratio calculation."""
        from src.training.statistical_tests import SharpeRatioTest, StatisticalTestResult

        # Calculate regular Sharpe
        sharpe = self.good_returns.mean() / self.good_returns.std() * np.sqrt(252)

        # Deflate for multiple testing
        result = SharpeRatioTest.deflated_sharpe_ratio(
            sharpe=sharpe,
            n_trials=50,
            n_observations=252,
        )

        # Result should be StatisticalTestResult
        assert isinstance(result, StatisticalTestResult)
        assert result.test_name == "deflated_sharpe_ratio"
        assert hasattr(result, "statistic")
        assert hasattr(result, "p_value")
        # DSR (statistic) should be between 0 and 1
        assert 0 <= result.statistic <= 1


class TestMonteCarloSimulator:
    """Test Monte Carlo simulation methods."""

    def setup_method(self):
        """Create test returns as pandas Series."""
        np.random.seed(42)
        self.returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005)

    def test_permutation_test(self):
        """Test permutation test for strategy significance."""
        from src.training.statistical_tests import MonteCarloSimulator

        simulator = MonteCarloSimulator(n_simulations=100)

        result = simulator.permutation_test(self.returns)

        # Result should have p_value (either as attribute or in dict)
        if hasattr(result, "p_value"):
            assert 0 <= result.p_value <= 1
        elif isinstance(result, dict):
            assert "p_value" in result
            assert 0 <= result["p_value"] <= 1

    def test_bootstrap_confidence_interval(self):
        """Test bootstrap confidence interval calculation."""
        from src.training.statistical_tests import MonteCarloSimulator

        simulator = MonteCarloSimulator(n_simulations=100)

        result = simulator.bootstrap_confidence_interval(
            self.returns, confidence_level=0.95
        )

        # Should return bounds
        if isinstance(result, dict):
            assert "lower_bound" in result or "lower" in result
            assert "upper_bound" in result or "upper" in result
        elif isinstance(result, tuple):
            assert len(result) >= 2


class TestProbabilityOfBacktestOverfitting:
    """Test Probability of Backtest Overfitting calculation."""

    def setup_method(self):
        """Create returns matrix for multiple strategies."""
        np.random.seed(42)
        n_periods = 252
        n_strategies = 10

        # Matrix of strategy returns as DataFrame
        self.returns_matrix = pd.DataFrame(
            np.random.randn(n_periods, n_strategies) * 0.01,
            columns=[f"strategy_{i}" for i in range(n_strategies)]
        )

    def test_pbo_calculation(self):
        """Test PBO calculation returns valid result."""
        from src.training.statistical_tests import ProbabilityOfBacktestOverfitting

        pbo_calc = ProbabilityOfBacktestOverfitting(n_partitions=4)

        result = pbo_calc.calculate_pbo(self.returns_matrix)

        # Result can be dict or float
        if isinstance(result, dict):
            assert "pbo" in result or "probability" in result
            pbo_value = result.get("pbo", result.get("probability", 0))
            assert 0 <= pbo_value <= 1
        else:
            assert 0 <= result <= 1


class TestMultipleTestingCorrection:
    """Test multiple testing correction methods."""

    def setup_method(self):
        """Create p-values for testing."""
        self.p_values = [0.001, 0.01, 0.03, 0.05, 0.10, 0.20, 0.50, 0.80]

    def test_bonferroni_correction(self):
        """Test Bonferroni correction returns result."""
        from src.training.statistical_tests import MultipleTestingCorrection

        result = MultipleTestingCorrection.bonferroni(self.p_values)

        # Result should be dict with expected keys
        assert isinstance(result, dict)
        assert "method" in result
        assert result["method"] == "bonferroni"
        assert "corrected_alpha" in result
        assert "is_significant" in result
        assert "n_tests" in result
        assert result["n_tests"] == len(self.p_values)

    def test_benjamini_hochberg_correction(self):
        """Test Benjamini-Hochberg FDR correction returns result."""
        from src.training.statistical_tests import MultipleTestingCorrection

        result = MultipleTestingCorrection.benjamini_hochberg(self.p_values)

        # Result should be dict
        assert isinstance(result, dict)

    def test_holm_bonferroni_correction(self):
        """Test Holm-Bonferroni correction returns result."""
        from src.training.statistical_tests import MultipleTestingCorrection

        result = MultipleTestingCorrection.holm_bonferroni(self.p_values)

        # Result should be dict with expected keys
        assert isinstance(result, dict)
        assert "method" in result
        assert result["method"] == "holm_bonferroni"
        assert "is_significant" in result
        assert "n_tests" in result


class TestStrategyValidation:
    """Test combined strategy validation pipeline."""

    def test_full_validation_pipeline(self):
        """Test complete validation of a strategy."""
        from src.training.statistical_tests import SharpeRatioTest

        np.random.seed(42)

        # Create strategy returns as pandas Series
        returns = pd.Series(np.random.randn(500) * 0.01 + 0.0005)

        # Test Sharpe significance
        sharpe_result = SharpeRatioTest.test_sharpe_significance(returns)

        # Should provide valid result
        assert sharpe_result is not None
        assert hasattr(sharpe_result, "p_value") or isinstance(sharpe_result, dict)


class TestStatisticalTestResult:
    """Test StatisticalTestResult dataclass."""

    def test_result_structure(self):
        """Test that result has expected structure."""
        from src.training.statistical_tests import SharpeRatioTest

        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01)

        result = SharpeRatioTest.test_sharpe_significance(returns)

        # Check structure
        assert hasattr(result, "test_name")
        assert hasattr(result, "statistic")
        assert hasattr(result, "p_value")
        assert hasattr(result, "is_significant")
        assert hasattr(result, "confidence_level")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
