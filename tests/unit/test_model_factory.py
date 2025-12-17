"""
Unit tests for ModelFactory.

Tests the critical fix:
- create_model method exists and works correctly
- Parameter passing via params= argument
"""

import pytest
import numpy as np


class TestModelFactoryMethods:
    """Test ModelFactory class methods."""

    def test_create_model_method_exists(self):
        """Verify create_model method exists (not just 'create')."""
        from src.training.model_factory import ModelFactory

        # The correct method name is create_model, not create
        assert hasattr(ModelFactory, 'create_model')
        assert callable(ModelFactory.create_model)

    def test_create_method_does_not_exist(self):
        """Verify the old 'create' method doesn't exist (it was a bug)."""
        from src.training.model_factory import ModelFactory

        # The bug was calling ModelFactory.create() which doesn't exist
        # The correct method is create_model()
        assert not hasattr(ModelFactory, 'create'), \
            "ModelFactory.create() should not exist - use create_model() instead"


class TestModelFactoryCreation:
    """Test model creation functionality."""

    def test_create_lightgbm_regressor(self):
        """Test creating a LightGBM regressor."""
        from src.training.model_factory import ModelFactory

        try:
            model = ModelFactory.create_model(
                model_type="lightgbm_regressor",
                params={"n_estimators": 10, "max_depth": 3},
            )
            assert model is not None
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict')
        except ImportError:
            pytest.skip("LightGBM not installed")

    def test_create_model_with_params_dict(self):
        """Test that params must be passed as a dict to params= argument."""
        from src.training.model_factory import ModelFactory

        try:
            # Correct way: params=dict
            model = ModelFactory.create_model(
                model_type="lightgbm_regressor",
                params={"n_estimators": 10},
            )
            assert model is not None

            # Also test with None params
            model2 = ModelFactory.create_model(
                model_type="lightgbm_regressor",
                params=None,
            )
            assert model2 is not None
        except ImportError:
            pytest.skip("LightGBM not installed")

    def test_create_random_forest_regressor(self):
        """Test creating a RandomForest regressor (sklearn, always available)."""
        from src.training.model_factory import ModelFactory

        model = ModelFactory.create_model(
            model_type="random_forest_regressor",
            params={"n_estimators": 10, "max_depth": 3},
        )

        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_model_can_fit_and_predict(self):
        """Test that created model can fit and predict."""
        from src.training.model_factory import ModelFactory

        model = ModelFactory.create_model(
            model_type="random_forest_regressor",
            params={"n_estimators": 10, "max_depth": 3, "random_state": 42},
        )

        # Create dummy data
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randn(100)

        # Fit
        model.fit(X, y)

        # Predict
        predictions = model.predict(X[:10])

        assert len(predictions) == 10
        assert all(np.isfinite(predictions))


class TestModelTypes:
    """Test different model types can be created."""

    @pytest.mark.parametrize("model_type", [
        "random_forest_regressor",
        "random_forest_classifier",
        "ridge",
        "lasso",
        "elastic_net",
        "logistic_regression",
    ])
    def test_sklearn_models(self, model_type):
        """Test sklearn models can be created."""
        from src.training.model_factory import ModelFactory

        model = ModelFactory.create_model(model_type=model_type)
        assert model is not None

    @pytest.mark.parametrize("model_type", [
        "lightgbm_regressor",
        "lightgbm_classifier",
    ])
    def test_lightgbm_models(self, model_type):
        """Test LightGBM models can be created."""
        from src.training.model_factory import ModelFactory

        try:
            model = ModelFactory.create_model(
                model_type=model_type,
                params={"n_estimators": 10, "verbose": -1},
            )
            assert model is not None
        except ImportError:
            pytest.skip("LightGBM not installed")

    @pytest.mark.parametrize("model_type", [
        "xgboost_regressor",
        "xgboost_classifier",
    ])
    def test_xgboost_models(self, model_type):
        """Test XGBoost models can be created."""
        from src.training.model_factory import ModelFactory

        try:
            model = ModelFactory.create_model(
                model_type=model_type,
                params={"n_estimators": 10, "verbosity": 0},
            )
            assert model is not None
        except ImportError:
            pytest.skip("XGBoost not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
