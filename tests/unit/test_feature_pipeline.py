"""
Unit tests for Feature Pipeline.

Tests the critical fixes:
1. generate_features method is properly inside FeaturePipeline class
2. fit/transform separation for leakage prevention
3. Feature generation produces valid output
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


class TestFeaturePipelineClass:
    """Test that FeaturePipeline class is correctly structured."""

    def test_generate_features_is_method(self):
        """Verify generate_features is a method of FeaturePipeline class."""
        from src.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline()

        # Check that generate_features is a callable method
        assert hasattr(pipeline, 'generate_features')
        assert callable(getattr(pipeline, 'generate_features'))

        # Check that it's actually bound to the instance (not a module-level function)
        method = getattr(pipeline, 'generate_features')
        assert hasattr(method, '__self__')
        assert method.__self__ is pipeline

    def test_fit_method_exists(self):
        """Verify fit method exists and is callable."""
        from src.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline()
        assert hasattr(pipeline, 'fit')
        assert callable(pipeline.fit)

    def test_transform_method_exists(self):
        """Verify transform method exists and is callable."""
        from src.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline()
        assert hasattr(pipeline, 'transform')
        assert callable(pipeline.transform)

    def test_fit_transform_method_exists(self):
        """Verify fit_transform method exists and is callable."""
        from src.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline()
        assert hasattr(pipeline, 'fit_transform')
        assert callable(pipeline.fit_transform)


class TestFeaturePipelineGeneration:
    """Test feature generation functionality."""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        n_bars = 500
        dates = pd.date_range(start='2020-01-01', periods=n_bars, freq='15min')

        np.random.seed(42)

        # Generate realistic-ish price data
        base_price = 100.0
        returns = np.random.normal(0, 0.001, n_bars)
        prices = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.001, 0.001, n_bars)),
            'high': prices * (1 + np.random.uniform(0, 0.005, n_bars)),
            'low': prices * (1 - np.random.uniform(0, 0.005, n_bars)),
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_bars),
        }, index=dates)

        return df

    def test_generate_features_produces_output(self, sample_ohlcv_data):
        """Test that generate_features produces a DataFrame with features."""
        from src.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(strict_leakage_check=False)
        features = pipeline.generate_features(sample_ohlcv_data)

        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        assert len(features.columns) > 0

    def test_generate_features_column_count(self, sample_ohlcv_data):
        """Test that generate_features produces expected number of features."""
        from src.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(strict_leakage_check=False)
        features = pipeline.generate_features(
            sample_ohlcv_data,
            include_technical=True,
            include_statistical=True,
            include_lagged=True,
        )

        # Should produce a reasonable number of features
        assert len(features.columns) >= 10, "Should generate at least 10 features"

    def test_generate_features_index_alignment(self, sample_ohlcv_data):
        """Test that generated features align with input index."""
        from src.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(strict_leakage_check=False)
        features = pipeline.generate_features(sample_ohlcv_data)

        # Index should match
        assert len(features.index) == len(sample_ohlcv_data.index)
        assert all(features.index == sample_ohlcv_data.index)


class TestFeaturePipelineFitTransform:
    """Test fit/transform separation for leakage prevention."""

    @pytest.fixture
    def train_test_data(self):
        """Create train/test split data."""
        n_bars = 500
        dates = pd.date_range(start='2020-01-01', periods=n_bars, freq='15min')

        np.random.seed(42)
        base_price = 100.0
        returns = np.random.normal(0, 0.001, n_bars)
        prices = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.001, 0.001, n_bars)),
            'high': prices * (1 + np.random.uniform(0, 0.005, n_bars)),
            'low': prices * (1 - np.random.uniform(0, 0.005, n_bars)),
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_bars),
        }, index=dates)

        # Split 80/20
        split_idx = int(len(df) * 0.8)
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]

        return train, test

    def test_fit_sets_is_fitted_flag(self, train_test_data):
        """Test that fit() sets the _is_fitted flag."""
        from src.features.pipeline import FeaturePipeline

        train, _ = train_test_data
        pipeline = FeaturePipeline(strict_leakage_check=False)

        assert not pipeline.is_fitted

        pipeline.fit(train)

        assert pipeline.is_fitted

    def test_transform_requires_fitting(self, train_test_data):
        """Test that transform() raises error if not fitted."""
        from src.features.pipeline import FeaturePipeline

        train, test = train_test_data
        pipeline = FeaturePipeline(strict_leakage_check=False)

        with pytest.raises(ValueError, match="Pipeline not fitted"):
            pipeline.transform(test)

    def test_fit_transform_workflow(self, train_test_data):
        """Test the correct fit/transform workflow."""
        from src.features.pipeline import FeaturePipeline

        train, test = train_test_data
        pipeline = FeaturePipeline(strict_leakage_check=False)

        # Fit on training data
        train_features = pipeline.fit_transform(train)

        # Transform test data using fitted pipeline
        test_features = pipeline.transform(test)

        assert isinstance(train_features, pd.DataFrame)
        assert isinstance(test_features, pd.DataFrame)
        assert len(train_features) == len(train)
        assert len(test_features) == len(test)

        # Both should have same columns
        assert list(train_features.columns) == list(test_features.columns)


class TestMaxLookback:
    """Test max lookback tracking for purge gap calculation."""

    def test_max_lookback_property(self):
        """Test that max_lookback property is accessible."""
        from src.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(ma_periods=[5, 10, 20, 50, 100, 200])

        assert hasattr(pipeline, 'max_lookback')
        assert pipeline.max_lookback == 200

    def test_purge_gap_recommendation(self):
        """Test purge gap recommendation calculation."""
        from src.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(ma_periods=[5, 10, 20, 50, 100, 200])

        # purge_gap should be at least: prediction_horizon + max_lookback + buffer
        prediction_horizon = 5
        buffer = 10

        recommended = pipeline.get_purge_gap_recommendation(prediction_horizon, buffer)

        expected = prediction_horizon + 200 + buffer  # 215
        assert recommended == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
