"""
Integration tests for hyperparameter optimization.

Tests the optimization components:
- Optuna optimization
- Walk-forward optimization
- Multi-objective optimization
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def sample_data():
    """Create sample data for optimization testing."""
    np.random.seed(42)
    n = 1000

    # Generate features
    X = np.random.randn(n, 20)

    # Target with some signal
    noise = np.random.randn(n) * 0.5
    y = 0.3 * X[:, 0] + 0.2 * X[:, 1] - 0.1 * X[:, 2] + noise

    # Returns for financial metrics
    returns = np.random.normal(0.0001, 0.02, n)

    return X, y, returns


class TestOptunaOptimization:
    """Tests for Optuna optimizer."""

    @pytest.mark.skipif(
        not pytest.importorskip("optuna", reason="Optuna not installed"),
        reason="Optuna not installed"
    )
    def test_basic_optimization(self, sample_data):
        """Test basic hyperparameter optimization."""
        try:
            from src.training.optimization import OptunaOptimizer
            from src.training.model_factory import ModelType
            from src.training.validation import PurgedKFoldCV
        except ImportError as e:
            pytest.skip(f"Import error: {e}")

        X, y, _ = sample_data

        optimizer = OptunaOptimizer(
            model_type=ModelType.LIGHTGBM_REGRESSOR,
            validation_strategy=PurgedKFoldCV(n_splits=3, purge_gap=10),
            objective_metric="r2",
            direction="maximize",
            n_trials=5,  # Small number for testing
            random_state=42,
        )

        result = optimizer.optimize(X, y)

        # Check result
        assert result is not None
        assert result.best_params is not None
        assert result.best_value is not None
        assert len(result.best_params) > 0

    @pytest.mark.skipif(
        not pytest.importorskip("optuna", reason="Optuna not installed"),
        reason="Optuna not installed"
    )
    def test_optimization_with_financial_metric(self, sample_data):
        """Test optimization with financial objective."""
        try:
            from src.training.optimization import OptunaOptimizer
            from src.training.model_factory import ModelType
            from src.training.validation import PurgedKFoldCV
        except ImportError as e:
            pytest.skip(f"Import error: {e}")

        X, y, returns = sample_data

        optimizer = OptunaOptimizer(
            model_type=ModelType.LIGHTGBM_REGRESSOR,
            validation_strategy=PurgedKFoldCV(n_splits=3, purge_gap=10),
            objective_metric="sharpe_ratio",
            direction="maximize",
            n_trials=5,
            random_state=42,
        )

        result = optimizer.optimize(X, y, returns=returns)

        assert result is not None
        assert result.best_params is not None


class TestWalkForwardOptimization:
    """Tests for walk-forward optimization."""

    @pytest.mark.skipif(
        not pytest.importorskip("optuna", reason="Optuna not installed"),
        reason="Optuna not installed"
    )
    def test_walk_forward_optimizer(self, sample_data):
        """Test walk-forward optimization."""
        try:
            from src.training.optimization import WalkForwardOptimizer
            from src.training.model_factory import ModelType
        except ImportError as e:
            pytest.skip(f"Import error: {e}")

        X, y, returns = sample_data

        optimizer = WalkForwardOptimizer(
            model_type=ModelType.LIGHTGBM_REGRESSOR,
            train_period=300,
            test_period=50,
            step_size=50,
            n_optimization_trials=3,  # Small for testing
            cv_splits=2,
            purge_gap=10,
            objective_metric="r2",
            random_state=42,
        )

        result = optimizer.optimize(X, y, returns=returns)

        # Check result
        assert result is not None
        assert result.n_steps > 0
        assert len(result.walk_forward_results) > 0

        # Check each step has results
        for step_result in result.walk_forward_results:
            assert 'best_params' in step_result
            assert 'test_metrics' in step_result

    @pytest.mark.skipif(
        not pytest.importorskip("optuna", reason="Optuna not installed"),
        reason="Optuna not installed"
    )
    def test_walk_forward_result_aggregation(self, sample_data):
        """Test walk-forward result aggregation."""
        try:
            from src.training.optimization import WalkForwardOptimizer
            from src.training.model_factory import ModelType
        except ImportError as e:
            pytest.skip(f"Import error: {e}")

        X, y, returns = sample_data

        optimizer = WalkForwardOptimizer(
            model_type=ModelType.LIGHTGBM_REGRESSOR,
            train_period=300,
            test_period=50,
            n_optimization_trials=3,
            random_state=42,
        )

        result = optimizer.optimize(X, y, returns=returns)

        # Check aggregated metrics
        assert hasattr(result, 'aggregated_sharpe')
        assert hasattr(result, 'aggregated_return')
        assert hasattr(result, 'parameter_stability')

        # Check predictions
        all_preds, all_actual = result.get_all_predictions()
        assert len(all_preds) > 0
        assert len(all_actual) == len(all_preds)


class TestMultiObjectiveOptimization:
    """Tests for multi-objective optimization."""

    @pytest.mark.skipif(
        not pytest.importorskip("optuna", reason="Optuna not installed"),
        reason="Optuna not installed"
    )
    def test_multi_objective_optimizer(self, sample_data):
        """Test multi-objective optimization."""
        try:
            from src.training.optimization import MultiObjectiveOptimizer
            from src.training.model_factory import ModelType
        except ImportError as e:
            pytest.skip(f"Import error: {e}")

        X, y, returns = sample_data

        optimizer = MultiObjectiveOptimizer(
            model_type=ModelType.LIGHTGBM_REGRESSOR,
            objectives=["r2", "mse"],
            directions=["maximize", "minimize"],
            n_trials=5,  # Small for testing
            random_state=42,
        )

        results = optimizer.optimize(X, y)

        # Should return Pareto front
        assert results is not None
        assert len(results) > 0

        # Each result should have params
        for result in results:
            assert result.best_params is not None

    @pytest.mark.skipif(
        not pytest.importorskip("optuna", reason="Optuna not installed"),
        reason="Optuna not installed"
    )
    def test_pareto_front(self, sample_data):
        """Test Pareto front extraction."""
        try:
            from src.training.optimization import MultiObjectiveOptimizer
            from src.training.model_factory import ModelType
        except ImportError as e:
            pytest.skip(f"Import error: {e}")

        X, y, _ = sample_data

        optimizer = MultiObjectiveOptimizer(
            model_type=ModelType.LIGHTGBM_REGRESSOR,
            objectives=["r2", "mse"],
            directions=["maximize", "minimize"],
            n_trials=5,
            random_state=42,
        )

        optimizer.optimize(X, y)

        # Get Pareto front
        front = optimizer.get_pareto_front()

        assert front is not None
        # Each solution should have params and values
        for solution in front:
            assert 'params' in solution
            assert 'values' in solution


class TestAdaptiveOptimization:
    """Tests for adaptive optimization."""

    @pytest.mark.skipif(
        not pytest.importorskip("optuna", reason="Optuna not installed"),
        reason="Optuna not installed"
    )
    def test_adaptive_optimizer(self, sample_data):
        """Test adaptive optimization."""
        try:
            from src.training.optimization import AdaptiveOptimizer
            from src.training.model_factory import ModelType
        except ImportError as e:
            pytest.skip(f"Import error: {e}")

        X, y, _ = sample_data

        optimizer = AdaptiveOptimizer(
            model_type=ModelType.LIGHTGBM_REGRESSOR,
            objective_metric="r2",
            direction="maximize",
            n_trials=10,
            adaptation_frequency=5,
            random_state=42,
        )

        result = optimizer.optimize(X, y)

        assert result is not None
        assert result.best_params is not None
        assert result.metadata.get('adaptive') is True


class TestOptimizationConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.skipif(
        not pytest.importorskip("optuna", reason="Optuna not installed"),
        reason="Optuna not installed"
    )
    def test_optimize_model_function(self, sample_data):
        """Test optimize_model convenience function."""
        try:
            from src.training.optimization import optimize_model
        except ImportError as e:
            pytest.skip(f"Import error: {e}")

        X, y, _ = sample_data

        result = optimize_model(
            model_type="lightgbm_regressor",
            X=X,
            y=y,
            objective="r2",
            n_trials=5,
            cv_splits=3,
        )

        assert result is not None
        assert result.best_params is not None

    @pytest.mark.skipif(
        not pytest.importorskip("optuna", reason="Optuna not installed"),
        reason="Optuna not installed"
    )
    def test_walk_forward_optimize_function(self, sample_data):
        """Test walk_forward_optimize convenience function."""
        try:
            from src.training.optimization import walk_forward_optimize
        except ImportError as e:
            pytest.skip(f"Import error: {e}")

        X, y, returns = sample_data

        result = walk_forward_optimize(
            model_type="lightgbm_regressor",
            X=X,
            y=y,
            returns=returns,
            train_period=300,
            test_period=50,
            n_trials=3,
            objective="r2",
        )

        assert result is not None
        assert result.n_steps > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
