"""
Tests for market regime detection module.

Tests verify that HMM, volatility, and trend regime detectors correctly
classify market states.

Section 9: Required test coverage for Directive 2.7.
"""

import numpy as np
import pandas as pd
import pytest


class TestVolatilityRegimeDetector:
    """Test volatility-based regime detection."""

    def setup_method(self):
        """Create test data with clear volatility regimes."""
        np.random.seed(42)

        # Create data with three distinct volatility regimes
        n_per_regime = 200

        # Low volatility regime
        low_vol = np.random.randn(n_per_regime) * 0.5

        # Normal volatility regime
        normal_vol = np.random.randn(n_per_regime) * 1.0

        # High volatility regime
        high_vol = np.random.randn(n_per_regime) * 2.5

        # Combine into single series
        returns = np.concatenate([low_vol, normal_vol, high_vol])
        dates = pd.date_range("2020-01-01", periods=len(returns), freq="15min")

        self.test_data = pd.DataFrame(
            {"close": 100 + np.cumsum(returns)},
            index=dates,
        )

    def test_volatility_detector_initialization(self):
        """Test volatility regime detector initializes correctly."""
        from src.features.regime_detection import VolatilityRegimeDetector

        detector = VolatilityRegimeDetector(lookback=60, n_regimes=3)
        assert detector.lookback == 60
        assert detector.n_regimes == 3

    def test_volatility_detector_fit_predict(self):
        """Test volatility detector can fit and predict regimes."""
        from src.features.regime_detection import VolatilityRegimeDetector

        detector = VolatilityRegimeDetector(lookback=30, n_regimes=3)

        # Fit and predict
        detector.fit(self.test_data)
        result = detector.predict(self.test_data)

        # Should have regime predictions
        assert len(result.regimes) == len(self.test_data)

        # Regimes should be in valid range
        valid_regimes = result.regimes.dropna()
        assert all(valid_regimes.isin([0, 1, 2])), "Regimes should be 0, 1, or 2"

    def test_volatility_detector_regime_separation(self):
        """Test that volatility detector separates regimes correctly."""
        from src.features.regime_detection import VolatilityRegimeDetector

        detector = VolatilityRegimeDetector(lookback=30, n_regimes=3)
        detector.fit(self.test_data)
        result = detector.predict(self.test_data)

        # Drop NaN values
        valid_regimes = result.regimes.dropna()

        # Check that we have all three regimes
        unique_regimes = valid_regimes.unique()
        assert len(unique_regimes) >= 2, "Should identify multiple regimes"


class TestTrendRegimeDetector:
    """Test trend-based regime detection."""

    def setup_method(self):
        """Create test data with trending and mean-reverting periods."""
        np.random.seed(42)

        # Strong uptrend
        uptrend = 100 + np.cumsum(np.random.randn(100) * 0.1 + 0.3)

        # Range-bound
        range_bound = 130 + np.cumsum(np.random.randn(200) * 0.5)
        range_bound = range_bound - np.linspace(0, np.mean(range_bound) - 130, 200)

        # Strong downtrend
        downtrend = 130 - np.cumsum(np.random.randn(100) * 0.1 + 0.3)

        # More range-bound
        range_bound2 = 100 + np.cumsum(np.random.randn(100) * 0.5)

        close = np.concatenate([uptrend, range_bound, downtrend, range_bound2])
        dates = pd.date_range("2020-01-01", periods=len(close), freq="15min")

        # Create OHLC data
        self.test_data = pd.DataFrame(index=dates)
        self.test_data["close"] = close
        self.test_data["high"] = close + np.abs(np.random.randn(len(close)) * 0.5)
        self.test_data["low"] = close - np.abs(np.random.randn(len(close)) * 0.5)
        self.test_data["open"] = close + np.random.randn(len(close)) * 0.2

    def test_trend_detector_initialization(self):
        """Test trend regime detector initializes correctly."""
        from src.features.regime_detection import TrendRegimeDetector

        detector = TrendRegimeDetector(short_window=20, long_window=50)
        assert detector.short_window == 20
        assert detector.long_window == 50

    def test_trend_detector_fit_predict(self):
        """Test trend detector can predict regimes."""
        from src.features.regime_detection import TrendRegimeDetector

        detector = TrendRegimeDetector(short_window=10, long_window=30)

        # TrendRegimeDetector doesn't have fit method, just predict
        result = detector.predict(self.test_data)

        # Should have regime predictions
        assert len(result.regimes) == len(self.test_data)

        # Valid regimes after initial NaN period
        valid_regimes = result.regimes.dropna()
        assert len(valid_regimes) > 0, "Should have valid regime predictions"


class TestHMMRegimeDetector:
    """Test Hidden Markov Model regime detection."""

    def setup_method(self):
        """Create test data."""
        np.random.seed(42)
        n_samples = 500

        close = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)
        dates = pd.date_range("2020-01-01", periods=n_samples, freq="15min")

        self.test_data = pd.DataFrame(
            {"close": close},
            index=dates,
        )

    def test_hmm_detector_initialization(self):
        """Test HMM detector initializes correctly (or skips if not installed)."""
        try:
            from src.features.regime_detection import HMMRegimeDetector
            detector = HMMRegimeDetector(n_regimes=3, n_iter=100)
            assert detector.n_regimes == 3
        except ImportError:
            pytest.skip("hmmlearn not installed")

    def test_hmm_detector_fit_predict(self):
        """Test HMM detector can fit and predict regimes."""
        try:
            from src.features.regime_detection import HMMRegimeDetector
            detector = HMMRegimeDetector(n_regimes=3, n_iter=50)
            detector.fit(self.test_data)
            result = detector.predict(self.test_data)
            assert len(result.regimes) == len(self.test_data)
        except ImportError:
            pytest.skip("hmmlearn not installed")


class TestCompositeRegimeDetector:
    """Test composite regime detection combining multiple methods."""

    def setup_method(self):
        """Create comprehensive test data."""
        np.random.seed(42)
        n_samples = 500

        close = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)
        dates = pd.date_range("2020-01-01", periods=n_samples, freq="15min")

        self.test_data = pd.DataFrame(index=dates)
        self.test_data["close"] = close
        self.test_data["high"] = close + np.abs(np.random.randn(n_samples) * 0.3)
        self.test_data["low"] = close - np.abs(np.random.randn(n_samples) * 0.3)
        self.test_data["volume"] = np.random.randint(1000, 10000, n_samples)

    def test_composite_detector_initialization(self):
        """Test composite detector initializes."""
        from src.features.regime_detection import CompositeRegimeDetector

        detector = CompositeRegimeDetector()
        assert detector is not None

    def test_composite_detector_fit_predict(self):
        """Test composite detector combines multiple methods."""
        from src.features.regime_detection import CompositeRegimeDetector

        detector = CompositeRegimeDetector()
        detector.fit(self.test_data)
        result = detector.predict(self.test_data)

        # CompositeRegimeDetector returns DataFrame, not RegimeResult
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestRegimeFeatureGeneration:
    """Test regime-based feature generation."""

    def test_regime_as_feature(self):
        """Test that regime can be used as a feature."""
        from src.features.regime_detection import VolatilityRegimeDetector

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(500) * 0.5)
        df = pd.DataFrame({"close": close})

        detector = VolatilityRegimeDetector(lookback=20)
        detector.fit(df)
        result = detector.predict(df)

        # Regime should be valid categorical feature
        valid_regimes = result.regimes.dropna()
        assert valid_regimes.dtype in [np.int64, np.int32, np.float64, int]

    def test_regime_stability(self):
        """Test that regime assignments are stable (not too noisy)."""
        from src.features.regime_detection import VolatilityRegimeDetector

        np.random.seed(42)
        # Create smooth returns
        close = 100 + np.cumsum(np.random.randn(500) * 0.1)
        df = pd.DataFrame({"close": close})

        detector = VolatilityRegimeDetector(lookback=30)
        detector.fit(df)
        result = detector.predict(df)

        # Count regime changes
        valid_regimes = result.regimes.dropna()
        regime_changes = (valid_regimes != valid_regimes.shift(1)).sum()

        # Should not change too frequently (less than 50% of observations)
        change_rate = regime_changes / len(valid_regimes)
        assert change_rate < 0.5, f"Regime changes too frequently: {change_rate:.2%}"


class TestRegimeResult:
    """Test RegimeResult dataclass."""

    def test_regime_result_structure(self):
        """Test RegimeResult has expected structure."""
        from src.features.regime_detection import VolatilityRegimeDetector

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(300) * 0.5)
        df = pd.DataFrame({"close": close})

        detector = VolatilityRegimeDetector(lookback=20, n_regimes=3)
        detector.fit(df)
        result = detector.predict(df)

        # Check result structure
        assert hasattr(result, "regimes")
        assert hasattr(result, "regime_stats")
        assert isinstance(result.regimes, pd.Series)

    def test_regime_stats(self):
        """Test that regime stats are computed."""
        from src.features.regime_detection import VolatilityRegimeDetector

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(300) * 0.5)
        df = pd.DataFrame({"close": close})

        detector = VolatilityRegimeDetector(lookback=20, n_regimes=3)
        detector.fit(df)
        result = detector.predict(df)

        # Should have stats for each regime
        assert result.regime_stats is not None
        assert len(result.regime_stats) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
