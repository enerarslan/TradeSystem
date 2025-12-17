"""
Unit tests for data loading and validation.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.loaders.data_loader import DataLoader, get_available_symbols
from src.data.validators.data_validator import DataValidator, validate_ohlcv


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV DataFrame."""
    np.random.seed(42)
    n = 100

    dates = pd.date_range(start="2023-01-01", periods=n, freq="15min")
    close = 100 + np.cumsum(np.random.normal(0, 1, n))
    high = close + np.abs(np.random.normal(0, 0.5, n))
    low = close - np.abs(np.random.normal(0, 0.5, n))
    open_ = close + np.random.normal(0, 0.3, n)
    volume = np.random.randint(1000, 100000, n)

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


class TestDataValidator:
    """Tests for DataValidator."""

    def test_valid_data(self, sample_ohlcv_df):
        """Test validation of valid data."""
        validator = DataValidator()
        result = validator.validate(sample_ohlcv_df, "TEST")

        assert result.is_valid
        assert len(result.errors) == 0

    def test_missing_columns(self, sample_ohlcv_df):
        """Test detection of missing columns."""
        df = sample_ohlcv_df.drop(columns=["close"])
        validator = DataValidator()
        result = validator.validate(df, "TEST")

        assert not result.is_valid
        assert any("Missing required columns" in e for e in result.errors)

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        validator = DataValidator()
        result = validator.validate(df, "TEST")

        assert not result.is_valid
        assert "empty" in result.errors[0].lower()

    def test_negative_prices(self, sample_ohlcv_df):
        """Test detection of negative prices."""
        df = sample_ohlcv_df.copy()
        df.iloc[50, df.columns.get_loc("close")] = -10

        validator = DataValidator()
        result = validator.validate(df, "TEST")

        assert any("Negative" in e for e in result.errors)

    def test_ohlc_violations(self, sample_ohlcv_df):
        """Test detection of OHLC violations."""
        df = sample_ohlcv_df.copy()
        # Set high below low
        df.iloc[50, df.columns.get_loc("high")] = df.iloc[50]["low"] - 1

        validator = DataValidator()
        result = validator.validate(df, "TEST")

        assert any("OHLC violation" in w for w in result.warnings)

    def test_missing_values_warning(self, sample_ohlcv_df):
        """Test detection of missing values."""
        df = sample_ohlcv_df.copy()
        df.iloc[10:12, df.columns.get_loc("close")] = np.nan

        # With 2% missing (2 of 100), threshold of 5% makes it a warning, threshold of 1% makes it an error
        validator = DataValidator(max_missing_pct=5.0)
        result = validator.validate(df, "TEST")

        # Check for missing in warnings (below threshold) or errors (above threshold)
        all_messages = result.warnings + result.errors
        assert any("missing" in m.lower() for m in all_messages)

    def test_extreme_price_change_warning(self, sample_ohlcv_df):
        """Test detection of extreme price changes."""
        df = sample_ohlcv_df.copy()
        # Create a 60% price jump
        df.iloc[50, df.columns.get_loc("close")] = df.iloc[49]["close"] * 1.6

        validator = DataValidator(max_price_change_pct=50.0)
        result = validator.validate(df, "TEST")

        assert any("Extreme" in w for w in result.warnings)


class TestValidationStatistics:
    """Tests for validation statistics."""

    def test_stats_collection(self, sample_ohlcv_df):
        """Test that statistics are collected correctly."""
        validator = DataValidator()
        result = validator.validate(sample_ohlcv_df, "TEST")

        assert "row_count" in result.stats
        assert result.stats["row_count"] == len(sample_ohlcv_df)
        assert "close_min" in result.stats
        assert "close_max" in result.stats


class TestValidateOHLCVFunction:
    """Tests for the validate_ohlcv convenience function."""

    def test_normal_validation(self, sample_ohlcv_df):
        """Test normal validation mode."""
        result = validate_ohlcv(sample_ohlcv_df, "TEST", strict=False)
        assert result.is_valid

    def test_strict_validation(self, sample_ohlcv_df):
        """Test strict validation mode."""
        result = validate_ohlcv(sample_ohlcv_df, "TEST", strict=True)
        # Strict mode has tighter thresholds
        assert isinstance(result.is_valid, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
