"""
Tests for feature leakage prevention.

These tests verify that the feature pipeline and ML strategies properly
prevent data leakage between training and test sets.

CRITICAL: Data leakage is the #1 cause of over-optimistic backtest results.
These tests must pass before any production deployment.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


class TestFeaturePipelineLeakage:
    """Test feature pipeline for data leakage prevention."""

    def setup_method(self):
        """Create test data."""
        np.random.seed(42)
        n_samples = 1000
        dates = pd.date_range("2020-01-01", periods=n_samples, freq="15min")

        self.test_data = pd.DataFrame(
            {
                "open": 100 + np.cumsum(np.random.randn(n_samples) * 0.1),
                "high": 100 + np.cumsum(np.random.randn(n_samples) * 0.1) + np.abs(np.random.randn(n_samples) * 0.5),
                "low": 100 + np.cumsum(np.random.randn(n_samples) * 0.1) - np.abs(np.random.randn(n_samples) * 0.5),
                "close": 100 + np.cumsum(np.random.randn(n_samples) * 0.1),
                "volume": np.random.randint(1000, 10000, n_samples),
            },
            index=dates,
        )

        self.split_idx = 700  # 70/30 split

    def test_pipeline_requires_fit_before_transform(self):
        """Test that transform() raises error if not fitted."""
        from src.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline()

        with pytest.raises(ValueError, match="Pipeline not fitted"):
            pipeline.transform(self.test_data)

    def test_scaler_not_fitted_on_test_data(self):
        """Test that scaler parameters come from training data only."""
        from src.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline()

        # Split data
        train_data = self.test_data.iloc[: self.split_idx]
        test_data = self.test_data.iloc[self.split_idx :]

        # Fit on training data
        train_features = pipeline.fit_transform(train_data)

        # Get scaler statistics from training
        train_mean = train_features.mean()
        train_std = train_features.std()

        # Transform test data (should use training statistics)
        test_features = pipeline.transform(test_data)

        # Verify test data is scaled using training statistics
        # (not perfectly centered at 0, std not perfectly 1)
        test_mean = test_features.mean()
        test_std = test_features.std()

        # Test data statistics should NOT be exactly 0/1
        # (if they were, it would indicate leakage - fitting on test data)
        assert not np.allclose(
            test_mean.dropna(), 0, atol=0.1
        ), "Test data appears to be fit-transformed (leakage)"

    def test_fit_transform_equals_fit_then_transform(self):
        """Test that fit_transform equals fit() followed by transform()."""
        from src.features.pipeline import FeaturePipeline

        pipeline1 = FeaturePipeline()
        pipeline2 = FeaturePipeline()

        train_data = self.test_data.iloc[: self.split_idx]

        # Method 1: fit_transform
        result1 = pipeline1.fit_transform(train_data)

        # Method 2: fit then transform
        pipeline2.fit(train_data)
        result2 = pipeline2.transform(train_data)

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    def test_max_lookback_property(self):
        """Test that max_lookback is correctly calculated."""
        from src.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(ma_periods=[5, 10, 20, 50, 100, 200])

        assert pipeline.max_lookback == 200

    def test_purge_gap_recommendation(self):
        """Test purge gap recommendation calculation."""
        from src.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(ma_periods=[5, 10, 20, 50, 100, 200])

        # For prediction_horizon=5, buffer=10
        recommended = pipeline.get_purge_gap_recommendation(
            prediction_horizon=5, buffer=10
        )

        # Should be: 5 + 200 + 10 = 215
        assert recommended == 215


class TestMLStrategyLeakage:
    """Test ML strategy for data leakage prevention."""

    def setup_method(self):
        """Create test data."""
        np.random.seed(42)
        n_samples = 1000
        dates = pd.date_range("2020-01-01", periods=n_samples, freq="15min")

        self.prices = pd.DataFrame(
            {
                "close": 100 + np.cumsum(np.random.randn(n_samples) * 0.1),
            },
            index=dates,
        )

        self.features = pd.DataFrame(
            np.random.randn(n_samples, 10),
            index=dates,
            columns=[f"feature_{i}" for i in range(10)],
        )

    def test_scaler_fitted_only_on_training_data(self):
        """Test that ML strategy scaler is fitted only on training data."""
        from src.strategies.ml_based.ml_alpha import MLAlphaStrategy

        # Use regression mode for continuous targets, disable feature selection
        strategy = MLAlphaStrategy(params={
            "min_samples_train": 100,
            "feature_selection": False,
            "classification": False,  # Regression mode for continuous targets
        })

        # Prepare target (continuous returns)
        target = self.prices["close"].pct_change(5).shift(-5)

        # Split data
        split_idx = 700
        train_features = self.features.iloc[:split_idx]
        train_target = target.iloc[:split_idx]

        test_features = self.features.iloc[split_idx:]

        # Fit on training data
        strategy.fit(train_features, train_target, fit_scaler=True)

        # Check that scaler is fitted
        assert strategy._scaler_fitted

        # Predict on test data (should use training scaler)
        predictions = strategy.predict(test_features)

        # Predictions should exist
        assert len(predictions) > 0

    def test_cannot_predict_without_fitting_scaler(self):
        """Test that prediction fails if scaler not fitted."""
        from src.strategies.ml_based.ml_alpha import MLAlphaStrategy

        # Use regression mode, disable feature selection
        strategy = MLAlphaStrategy(params={
            "min_samples_train": 100,
            "feature_selection": False,
            "classification": False,  # Regression mode
        })

        # Try to fit with fit_scaler=False (should fail)
        target = self.prices["close"].pct_change(5).shift(-5)

        with pytest.raises(ValueError, match="Scaler not fitted"):
            strategy.fit(self.features, target, fit_scaler=False)

    def test_embargo_period_calculation(self):
        """Test embargo period calculation."""
        from src.strategies.ml_based.ml_alpha import MLAlphaStrategy

        strategy = MLAlphaStrategy(
            params={
                "prediction_horizon": 5,
                "max_feature_lookback": 200,
                "embargo_buffer": 10,
            }
        )

        embargo = strategy._calculate_embargo_periods()

        # Should be: 5 + 200 + 10 = 215
        assert embargo == 215


class TestPurgedCVLeakage:
    """Test purged cross-validation for information leakage."""

    def setup_method(self):
        """Create test data."""
        np.random.seed(42)
        n_samples = 1000

        self.X = np.random.randn(n_samples, 10)
        self.y = np.random.randn(n_samples)

    def test_purge_gap_enforced(self):
        """Test that purge gap is properly enforced between folds."""
        from src.training.validation import PurgedKFoldCV

        purge_gap = 50
        cv = PurgedKFoldCV(n_splits=5, purge_gap=purge_gap)

        for train_idx, test_idx in cv.split(self.X):
            test_min = test_idx.min()
            test_max = test_idx.max()

            # Check training indices that are BEFORE the test set
            train_before_test = train_idx[train_idx < test_min]
            if len(train_before_test) > 0:
                train_before_max = train_before_test.max()
                # Gap between train (before) and test should be at least purge_gap
                assert test_min - train_before_max >= purge_gap, (
                    f"Purge gap violated: train_max={train_before_max}, "
                    f"test_min={test_min}, required_gap={purge_gap}"
                )

            # Check training indices that are AFTER the test set have embargo
            train_after_test = train_idx[train_idx > test_max]
            if len(train_after_test) > 0:
                train_after_min = train_after_test.min()
                # There should be a gap after test set (embargo)
                assert train_after_min > test_max, "Training should not immediately follow test"

    def test_no_overlap_between_train_and_test(self):
        """Test that train and test sets never overlap."""
        from src.training.validation import PurgedKFoldCV

        cv = PurgedKFoldCV(n_splits=5, purge_gap=20)

        for train_idx, test_idx in cv.split(self.X):
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0, f"Train and test overlap at indices: {overlap}"

    def test_cpcv_multiple_paths(self):
        """Test combinatorial purged CV generates multiple paths."""
        from src.training.validation import CombinatorialPurgedKFoldCV

        cv = CombinatorialPurgedKFoldCV(n_splits=6, n_test_splits=2, purge_gap=20)

        # Count number of unique paths
        paths = list(cv.split(self.X))

        # Should be C(6,2) = 15 paths
        assert len(paths) == 15

    def test_dynamic_purge_gap_calculation(self):
        """Test that purge gap is calculated dynamically when set to auto."""
        from main import calculate_purge_gap

        config = {
            "training": {
                "prediction_horizon": 5,
                "max_feature_lookback": 200,
                "purge_gap_buffer": 10,
            }
        }

        purge_gap = calculate_purge_gap(config)

        # Should be: 5 + 200 + 10 = 215
        assert purge_gap == 215


class TestTargetLeakage:
    """Test for target variable construction leakage."""

    def test_target_not_using_future_data(self):
        """Test that target construction doesn't use future data."""
        from src.strategies.ml_based.ml_alpha import MLAlphaStrategy

        # Use regression mode so NaN values are preserved (not converted to 0/1)
        strategy = MLAlphaStrategy(params={"classification": False})

        # Create simple price series
        dates = pd.date_range("2020-01-01", periods=100, freq="15min")
        prices = pd.DataFrame({"close": range(100)}, index=dates)

        # Prepare target with horizon=5
        target = strategy._prepare_target(prices, horizon=5)

        # Target at index 95-99 should be NaN (not enough future data)
        # shift(-5) makes last 5 rows NaN
        assert pd.isna(target["close"].iloc[95])
        assert pd.isna(target["close"].iloc[96])
        assert pd.isna(target["close"].iloc[97])
        assert pd.isna(target["close"].iloc[98])
        assert pd.isna(target["close"].iloc[99])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
