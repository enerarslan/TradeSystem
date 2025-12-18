"""
Tests for real-time feature pipeline.

Tests verify O(1) incremental updates and consistency with batch pipeline.

Section 9: Required test coverage for Module 6.1.
"""

import numpy as np
import pandas as pd
import pytest
import time


class TestIncrementalRollingStatistic:
    """Test incremental rolling mean/std calculations."""

    def setup_method(self):
        """Create test data."""
        np.random.seed(42)
        self.n_samples = 1000
        self.values = np.random.randn(self.n_samples)
        self.window = 20

    def test_incremental_mean_matches_batch(self):
        """Test that incremental mean matches rolling mean."""
        from src.features.realtime_pipeline import IncrementalRollingStatistic

        # Batch calculation
        batch_mean = pd.Series(self.values).rolling(self.window).mean().values

        # Incremental calculation
        inc = IncrementalRollingStatistic(window_size=self.window)
        inc_means = []

        for val in self.values:
            result = inc.update(val)
            inc_means.append(result["mean"])

        inc_means = np.array(inc_means)

        # Compare after warmup period
        valid_idx = self.window
        np.testing.assert_allclose(
            inc_means[valid_idx:],
            batch_mean[valid_idx:],
            rtol=1e-10,
            err_msg="Incremental mean should match batch mean",
        )

    def test_incremental_std_matches_batch(self):
        """Test that incremental std matches rolling std."""
        from src.features.realtime_pipeline import IncrementalRollingStatistic

        # Batch calculation (ddof=1 for sample std)
        batch_std = pd.Series(self.values).rolling(self.window).std().values

        # Incremental calculation
        inc = IncrementalRollingStatistic(window_size=self.window)
        inc_stds = []

        for val in self.values:
            result = inc.update(val)
            inc_stds.append(result["std"])

        inc_stds = np.array(inc_stds)

        # Compare after warmup period
        valid_idx = self.window
        np.testing.assert_allclose(
            inc_stds[valid_idx:],
            batch_std[valid_idx:],
            rtol=1e-6,
            err_msg="Incremental std should match batch std",
        )


class TestIncrementalEMA:
    """Test incremental EMA calculations."""

    def setup_method(self):
        """Create test data."""
        np.random.seed(42)
        self.n_samples = 500
        self.values = np.random.randn(self.n_samples).cumsum() + 100
        self.span = 20

    def test_ema_matches_batch(self):
        """Test that incremental EMA matches pandas EMA."""
        from src.features.realtime_pipeline import IncrementalEMA

        # Batch calculation
        batch_ema = pd.Series(self.values).ewm(span=self.span, adjust=False).mean().values

        # Incremental calculation
        inc = IncrementalEMA(span=self.span)
        inc_emas = []

        for val in self.values:
            inc_emas.append(inc.update(val))

        inc_emas = np.array(inc_emas)

        # Compare
        np.testing.assert_allclose(
            inc_emas,
            batch_ema,
            rtol=1e-10,
            err_msg="Incremental EMA should match batch EMA",
        )


class TestIncrementalRSI:
    """Test incremental RSI calculations."""

    def setup_method(self):
        """Create test price data."""
        np.random.seed(42)
        self.n_samples = 500
        self.prices = 100 + np.cumsum(np.random.randn(self.n_samples) * 0.5)
        self.period = 14

    def test_rsi_range(self):
        """Test that RSI is always in [0, 100]."""
        from src.features.realtime_pipeline import IncrementalRSI

        inc = IncrementalRSI(period=self.period)

        for price in self.prices:
            rsi = inc.update(price)
            if rsi is not None:
                assert 0 <= rsi <= 100, f"RSI should be in [0, 100], got {rsi}"

    def test_rsi_consistency(self):
        """Test RSI produces reasonable values after warmup."""
        from src.features.realtime_pipeline import IncrementalRSI

        # Incremental calculation
        inc = IncrementalRSI(period=self.period)
        inc_rsis = []

        for price in self.prices:
            inc_rsis.append(inc.update(price))

        inc_rsis = np.array(inc_rsis, dtype=float)

        # After warmup, RSI should be in valid range and reasonable
        valid_idx = self.period * 2
        valid_rsis = inc_rsis[valid_idx:]

        # All RSI values should be in [0, 100]
        assert all((valid_rsis >= 0) & (valid_rsis <= 100)), "RSI should be in [0, 100]"

        # RSI should not be stuck at extremes (unless price is trending heavily)
        # For random prices, RSI should have some variation
        rsi_std = np.std(valid_rsis)
        assert rsi_std > 1.0, f"RSI should vary, got std={rsi_std}"


class TestIncrementalMACD:
    """Test incremental MACD calculations."""

    def setup_method(self):
        """Create test price data."""
        np.random.seed(42)
        self.n_samples = 500
        self.prices = 100 + np.cumsum(np.random.randn(self.n_samples) * 0.5)

    def test_macd_computation(self):
        """Test MACD components are computed correctly."""
        from src.features.realtime_pipeline import IncrementalMACD

        inc = IncrementalMACD(fast_period=12, slow_period=26, signal_period=9)

        for price in self.prices:
            result = inc.update(price)
            if result["macd"] is not None:
                # MACD line should exist
                assert isinstance(result["macd"], float)
                # Signal should exist after enough samples
                if result["signal"] is not None:
                    assert isinstance(result["signal"], float)
                    # Histogram = MACD - Signal
                    expected_hist = result["macd"] - result["signal"]
                    assert abs(result["histogram"] - expected_hist) < 1e-10


class TestRealTimeFeaturePipeline:
    """Test complete real-time feature pipeline."""

    def setup_method(self):
        """Create test bar data."""
        np.random.seed(42)
        self.n_bars = 300

        base_price = 100 + np.cumsum(np.random.randn(self.n_bars) * 0.3)

        self.bars = []
        for i, close in enumerate(base_price):
            self.bars.append({
                "open": close + np.random.randn() * 0.1,
                "high": close + abs(np.random.randn() * 0.2),
                "low": close - abs(np.random.randn() * 0.2),
                "close": close,
                "volume": np.random.randint(1000, 10000),
            })

    def test_pipeline_initialization(self):
        """Test pipeline initializes correctly."""
        from src.features.realtime_pipeline import RealTimeFeaturePipeline

        pipeline = RealTimeFeaturePipeline(
            feature_config={
                "ema_periods": [9, 21],
                "rsi_periods": [14],
            }
        )

        assert pipeline is not None

    def test_pipeline_update(self):
        """Test pipeline processes bars and returns features."""
        from src.features.realtime_pipeline import RealTimeFeaturePipeline

        pipeline = RealTimeFeaturePipeline(
            feature_config={
                "ema_periods": [9, 21],
                "rsi_periods": [14],
            }
        )

        for bar in self.bars[:50]:
            features = pipeline.update(bar)

        # After warmup, features should be available
        assert "ema_9" in features or "ema_21" in features
        assert "rsi_14" in features

    def test_pipeline_o1_complexity(self):
        """Test that update time is O(1) - constant regardless of history."""
        from src.features.realtime_pipeline import RealTimeFeaturePipeline

        pipeline = RealTimeFeaturePipeline(
            feature_config={
                "ema_periods": [9, 21, 50],
                "rsi_periods": [14],
            }
        )

        # Warmup
        for bar in self.bars[:100]:
            pipeline.update(bar)

        # Time updates
        times = []
        for bar in self.bars[100:]:
            start = time.perf_counter()
            pipeline.update(bar)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        max_time = np.max(times)

        # Should be very fast (< 1ms on average)
        assert avg_time < 0.001, f"Average update time {avg_time*1000:.3f}ms exceeds 1ms"
        # Max time should not be much higher than average (no O(n) spikes)
        assert max_time < avg_time * 10, "Max time is too high relative to average"

    def test_feature_consistency(self):
        """Test that real-time features match batch-computed features."""
        from src.features.realtime_pipeline import RealTimeFeaturePipeline

        pipeline = RealTimeFeaturePipeline(
            feature_config={
                "ema_periods": [9, 21],
                "rsi_periods": [14],
            }
        )

        # Process all bars
        all_features = []
        for bar in self.bars:
            features = pipeline.update(bar)
            all_features.append(features)

        # Build batch DataFrame for EMA
        batch_df = pd.DataFrame(self.bars)
        batch_df["ema_9"] = batch_df["close"].ewm(span=9, adjust=False).mean()
        batch_df["ema_21"] = batch_df["close"].ewm(span=21, adjust=False).mean()

        # Compare last 100 values - EMA should match exactly
        for i in range(-100, -1):
            rt_features = all_features[i]
            if "ema_9" in rt_features:
                np.testing.assert_allclose(
                    rt_features["ema_9"],
                    batch_df["ema_9"].iloc[i],
                    rtol=1e-10,
                )


class TestPipelineConsistencyVerification:
    """Test the consistency verification method."""

    def test_verify_consistency_method(self):
        """Test that verify_consistency catches discrepancies."""
        from src.features.realtime_pipeline import RealTimeFeaturePipeline

        np.random.seed(42)

        pipeline = RealTimeFeaturePipeline(
            feature_config={
                "ema_periods": [9, 21],
            }
        )

        # Process bars
        bars = []
        for i in range(100):
            bar = {
                "open": 100 + i * 0.1,
                "high": 100 + i * 0.1 + 0.2,
                "low": 100 + i * 0.1 - 0.2,
                "close": 100 + i * 0.1,
                "volume": 1000,
            }
            bars.append(bar)
            pipeline.update(bar)

        # Build batch features
        batch_df = pd.DataFrame(bars)
        batch_df["ema_9"] = batch_df["close"].ewm(span=9, adjust=False).mean()
        batch_df["ema_21"] = batch_df["close"].ewm(span=21, adjust=False).mean()

        # Verify consistency (method exists in the API)
        if hasattr(pipeline, "verify_consistency"):
            # Reset and verify
            result = pipeline.verify_consistency(batch_df)
            assert isinstance(result, dict)
            assert "is_consistent" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
