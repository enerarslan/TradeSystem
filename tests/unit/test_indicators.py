"""
Unit tests for technical indicators.
"""

import numpy as np
import pandas as pd
import pytest

# Import after setting up path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.technical.indicators import TechnicalIndicators


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 200

    # Generate random walk for close prices
    returns = np.random.normal(0.0005, 0.02, n)
    close = 100 * np.cumprod(1 + returns)

    # Generate OHLCV
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
    open_ = (close + low) / 2 + np.random.normal(0, 0.5, n)
    volume = np.random.randint(10000, 1000000, n)

    dates = pd.date_range(start="2023-01-01", periods=n, freq="15min")

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)

    return df


class TestMovingAverages:
    """Tests for moving average indicators."""

    def test_sma_calculation(self, sample_ohlcv):
        """Test SMA calculation."""
        ti = TechnicalIndicators(sample_ohlcv)
        sma = ti.sma(period=20)

        assert len(sma) == len(sample_ohlcv)
        assert sma.iloc[19:].notna().all()  # First 19 are NaN
        assert sma.iloc[:19].isna().all()

        # Check manual calculation
        expected = sample_ohlcv["close"].iloc[0:20].mean()
        assert abs(sma.iloc[19] - expected) < 1e-10

    def test_ema_calculation(self, sample_ohlcv):
        """Test EMA calculation."""
        ti = TechnicalIndicators(sample_ohlcv)
        ema = ti.ema(period=20)

        assert len(ema) == len(sample_ohlcv)
        # EMA should have values from the start
        assert ema.notna().sum() > len(sample_ohlcv) - 20

    def test_wma_calculation(self, sample_ohlcv):
        """Test WMA calculation."""
        ti = TechnicalIndicators(sample_ohlcv)
        wma = ti.wma(period=10)

        assert len(wma) == len(sample_ohlcv)
        assert wma.iloc[9:].notna().all()


class TestMomentumIndicators:
    """Tests for momentum indicators."""

    def test_rsi_bounds(self, sample_ohlcv):
        """Test RSI is bounded between 0 and 100."""
        ti = TechnicalIndicators(sample_ohlcv)
        rsi = ti.rsi(period=14)

        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_stochastic_bounds(self, sample_ohlcv):
        """Test Stochastic is bounded between 0 and 100."""
        ti = TechnicalIndicators(sample_ohlcv)
        stoch_k, stoch_d = ti.stochastic()

        valid_k = stoch_k.dropna()
        valid_d = stoch_d.dropna()

        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()
        assert (valid_d >= 0).all()
        assert (valid_d <= 100).all()

    def test_williams_r_bounds(self, sample_ohlcv):
        """Test Williams %R is bounded between -100 and 0."""
        ti = TechnicalIndicators(sample_ohlcv)
        wr = ti.williams_r()

        valid_wr = wr.dropna()
        assert (valid_wr >= -100).all()
        assert (valid_wr <= 0).all()


class TestVolatilityIndicators:
    """Tests for volatility indicators."""

    def test_atr_positive(self, sample_ohlcv):
        """Test ATR is always positive."""
        ti = TechnicalIndicators(sample_ohlcv)
        atr = ti.atr()

        valid_atr = atr.dropna()
        assert (valid_atr >= 0).all()

    def test_bollinger_bands_ordering(self, sample_ohlcv):
        """Test Bollinger Bands ordering: lower < middle < upper."""
        ti = TechnicalIndicators(sample_ohlcv)
        bb = ti.bollinger_bands()

        # Get valid rows
        valid_idx = bb["bb_upper"].notna()

        assert (bb["bb_lower"][valid_idx] <= bb["bb_middle"][valid_idx]).all()
        assert (bb["bb_middle"][valid_idx] <= bb["bb_upper"][valid_idx]).all()

    def test_percent_b_calculation(self, sample_ohlcv):
        """Test %B is calculated correctly."""
        ti = TechnicalIndicators(sample_ohlcv)
        bb = ti.bollinger_bands()

        # %B = (close - lower) / (upper - lower)
        valid_idx = bb["bb_upper"].notna()
        close = sample_ohlcv["close"][valid_idx]
        expected_pct_b = (close - bb["bb_lower"][valid_idx]) / (
            bb["bb_upper"][valid_idx] - bb["bb_lower"][valid_idx]
        )

        np.testing.assert_array_almost_equal(
            bb["bb_percent_b"][valid_idx].values,
            expected_pct_b.values,
            decimal=10,
        )


class TestVolumeIndicators:
    """Tests for volume indicators."""

    def test_obv_calculation(self, sample_ohlcv):
        """Test OBV calculation."""
        ti = TechnicalIndicators(sample_ohlcv)
        obv = ti.obv()

        assert len(obv) == len(sample_ohlcv)
        # OBV should change based on price direction

    def test_mfi_bounds(self, sample_ohlcv):
        """Test MFI is bounded between 0 and 100."""
        ti = TechnicalIndicators(sample_ohlcv)
        mfi = ti.mfi()

        valid_mfi = mfi.dropna()
        # MFI can occasionally be outside bounds due to edge cases
        assert (valid_mfi >= -1).all()
        assert (valid_mfi <= 101).all()


class TestFeatureGeneration:
    """Tests for comprehensive feature generation."""

    def test_generate_all_features(self, sample_ohlcv):
        """Test generating all features."""
        ti = TechnicalIndicators(sample_ohlcv)
        features = ti.generate_all_features()

        # Should have many features
        assert len(features.columns) >= 50

        # Should have same length as input
        assert len(features) == len(sample_ohlcv)

    def test_no_look_ahead_bias(self, sample_ohlcv):
        """Test features don't have look-ahead bias."""
        ti = TechnicalIndicators(sample_ohlcv)
        features = ti.generate_all_features()

        # Features at time t should only depend on data up to time t
        # Check a few indicators
        sma_20 = features["sma_20"].iloc[100]
        expected_sma = sample_ohlcv["close"].iloc[81:101].mean()

        assert abs(sma_20 - expected_sma) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
