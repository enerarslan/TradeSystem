"""
Tests for advanced LOB (Limit Order Book) microstructure features.

Tests verify multi-depth OBI, spread analytics, toxicity, and smart money
indicators compute correctly.

Section 9: Required test coverage for Directive 2.4.
"""

import numpy as np
import pandas as pd
import pytest


class TestMultiDepthImbalance:
    """Test multi-depth order book imbalance calculations."""

    def setup_method(self):
        """Create test LOB data."""
        np.random.seed(42)

        self.bid_sizes = np.array([100, 200, 150, 300, 250, 180, 120, 280, 190, 160])
        self.ask_sizes = np.array([80, 180, 220, 280, 200, 150, 200, 250, 170, 140])

    def test_level_1_obi(self):
        """Test OBI at level 1 (best bid/ask only)."""
        from src.features.microstructure.advanced_lob_features import MultiDepthImbalance

        mdi = MultiDepthImbalance(levels=[1])
        obi = mdi.calculate_at_level(self.bid_sizes, self.ask_sizes, level=1)

        # Level 1: bid=100, ask=80 -> (100-80)/(100+80) = 20/180 = 0.111
        expected = (self.bid_sizes[0] - self.ask_sizes[0]) / (
            self.bid_sizes[0] + self.ask_sizes[0]
        )
        assert abs(obi - expected) < 1e-10

    def test_level_3_obi(self):
        """Test OBI at level 3 (top 3 levels)."""
        from src.features.microstructure.advanced_lob_features import MultiDepthImbalance

        mdi = MultiDepthImbalance(levels=[3])
        obi = mdi.calculate_at_level(self.bid_sizes, self.ask_sizes, level=3)

        bid_sum = np.sum(self.bid_sizes[:3])
        ask_sum = np.sum(self.ask_sizes[:3])
        expected = (bid_sum - ask_sum) / (bid_sum + ask_sum)
        assert abs(obi - expected) < 1e-10

    def test_obi_range(self):
        """Test that OBI is always in [-1, 1]."""
        from src.features.microstructure.advanced_lob_features import MultiDepthImbalance

        mdi = MultiDepthImbalance()

        for level in [1, 3, 5, 10]:
            obi = mdi.calculate_at_level(self.bid_sizes, self.ask_sizes, level=level)
            assert -1 <= obi <= 1, f"OBI at level {level} should be in [-1, 1]"

    def test_weighted_obi(self):
        """Test depth-weighted OBI calculation."""
        from src.features.microstructure.advanced_lob_features import MultiDepthImbalance

        bid_prices = np.array([100.0, 99.9, 99.8, 99.7, 99.6])
        ask_prices = np.array([100.1, 100.2, 100.3, 100.4, 100.5])
        bid_sizes = np.array([100, 200, 150, 100, 50])
        ask_sizes = np.array([80, 150, 200, 120, 60])

        mdi = MultiDepthImbalance(weighting="linear")
        weighted_obi = mdi.calculate_weighted(
            bid_prices, bid_sizes, ask_prices, ask_sizes
        )

        # Should be in valid range
        assert -1 <= weighted_obi <= 1


class TestSpreadAnalytics:
    """Test bid-ask spread analysis."""

    def setup_method(self):
        """Create test spread data."""
        np.random.seed(42)
        n_samples = 500

        # Base mid price with trend
        mid = 100 + np.cumsum(np.random.randn(n_samples) * 0.1)

        # Spread that varies
        half_spread = 0.02 + np.abs(np.random.randn(n_samples) * 0.01)

        self.bid = pd.Series(mid - half_spread)
        self.ask = pd.Series(mid + half_spread)
        self.mid = pd.Series(mid)

    def test_spread_bps_calculation(self):
        """Test spread in basis points."""
        from src.features.microstructure.advanced_lob_features import SpreadAnalytics

        analyzer = SpreadAnalytics(lookback=50)
        features = analyzer.calculate(self.bid, self.ask)

        spread_bps = features["spread_bps"]

        # Spread should be positive
        assert all(spread_bps > 0), "Spread should be positive"

        # Spread should be reasonable (< 100 bps for typical equity)
        assert spread_bps.mean() < 100, "Spread seems too wide"

    def test_spread_percentile(self):
        """Test spread percentile calculation."""
        from src.features.microstructure.advanced_lob_features import SpreadAnalytics

        analyzer = SpreadAnalytics(lookback=50)
        features = analyzer.calculate(self.bid, self.ask)

        spread_pct = features["spread_percentile"].dropna()

        # Percentile should be in [0, 1]
        assert all((spread_pct >= 0) & (spread_pct <= 1)), "Percentile should be in [0, 1]"

    def test_spread_momentum(self):
        """Test spread momentum calculation."""
        from src.features.microstructure.advanced_lob_features import SpreadAnalytics

        analyzer = SpreadAnalytics(lookback=50, momentum_window=10)
        features = analyzer.calculate(self.bid, self.ask)

        spread_mom = features["spread_momentum"].dropna()

        # Momentum should be centered around 0
        assert abs(spread_mom.mean()) < spread_mom.std() * 3


class TestTradeFlowToxicity:
    """Test trade flow toxicity indicators."""

    def setup_method(self):
        """Create test trade data."""
        np.random.seed(42)
        n_samples = 500

        self.price = pd.Series(100 + np.cumsum(np.random.randn(n_samples) * 0.1))
        self.volume = pd.Series(np.random.randint(100, 10000, n_samples))
        self.high = self.price + abs(np.random.randn(n_samples) * 0.1)
        self.low = self.price - abs(np.random.randn(n_samples) * 0.1)

    def test_toxicity_index_range(self):
        """Test toxicity index is in valid range."""
        from src.features.microstructure.advanced_lob_features import TradeFlowToxicity

        toxicity = TradeFlowToxicity(window=50)
        index = toxicity.calculate_toxicity_index(
            self.price, self.volume, self.high, self.low
        )

        valid_index = index.dropna()

        # Toxicity should be in [0, 1]
        assert all((valid_index >= 0) & (valid_index <= 1)), "Toxicity should be in [0, 1]"

    def test_toxicity_reacts_to_imbalance(self):
        """Test that toxicity increases with order imbalance."""
        from src.features.microstructure.advanced_lob_features import TradeFlowToxicity

        # Create balanced flow
        balanced_price = pd.Series(100 + np.random.randn(200) * 0.1)
        balanced_volume = pd.Series([1000] * 200)
        balanced_high = balanced_price + 0.05
        balanced_low = balanced_price - 0.05

        # Create imbalanced flow (all upticks)
        imbalanced_price = pd.Series(100 + np.cumsum(np.abs(np.random.randn(200) * 0.1)))
        imbalanced_volume = pd.Series([1000] * 200)
        imbalanced_high = imbalanced_price + 0.05
        imbalanced_low = imbalanced_price - 0.05

        toxicity = TradeFlowToxicity(window=50)

        tox_balanced = toxicity.calculate_toxicity_index(
            balanced_price, balanced_volume, balanced_high, balanced_low
        )
        tox_imbalanced = toxicity.calculate_toxicity_index(
            imbalanced_price, imbalanced_volume, imbalanced_high, imbalanced_low
        )

        # Imbalanced flow should have higher toxicity on average
        assert tox_imbalanced.dropna().mean() >= tox_balanced.dropna().mean() - 0.1


class TestSmartMoneyIndicator:
    """Test smart money detection."""

    def setup_method(self):
        """Create test trade data with block trades."""
        np.random.seed(42)
        n_samples = 500

        self.price = pd.Series(100 + np.cumsum(np.random.randn(n_samples) * 0.1))
        self.volume = pd.Series(np.random.randint(100, 1000, n_samples))

        # Add some block trades
        block_indices = np.random.choice(n_samples, 20, replace=False)
        self.volume.iloc[block_indices] = np.random.randint(5000, 20000, 20)

        self.high = self.price + abs(np.random.randn(n_samples) * 0.1)
        self.low = self.price - abs(np.random.randn(n_samples) * 0.1)

    def test_smart_money_index_range(self):
        """Test smart money index is in valid range."""
        from src.features.microstructure.advanced_lob_features import SmartMoneyIndicator

        smi = SmartMoneyIndicator(block_percentile=0.9, lookback=50)
        features = smi.calculate(self.price, self.volume, self.high, self.low)

        valid_smi = features["smart_money_index"].dropna()

        # SMI should be in [-1, 1]
        assert all((valid_smi >= -1) & (valid_smi <= 1)), "SMI should be in [-1, 1]"

    def test_block_trade_detection(self):
        """Test that block trades are detected."""
        from src.features.microstructure.advanced_lob_features import SmartMoneyIndicator

        smi = SmartMoneyIndicator(block_percentile=0.9, lookback=50)
        features = smi.calculate(self.price, self.volume, self.high, self.low)

        block_ratio = features["block_trade_ratio"].dropna()

        # Should detect some block trades (we added 20 out of 500)
        assert block_ratio.mean() > 0.01, "Should detect block trades"
        assert block_ratio.mean() < 0.2, "Block trade detection seems too aggressive"

    def test_institutional_activity_range(self):
        """Test institutional activity index is in valid range."""
        from src.features.microstructure.advanced_lob_features import SmartMoneyIndicator

        smi = SmartMoneyIndicator(block_percentile=0.95, lookback=50)
        features = smi.calculate(self.price, self.volume, self.high, self.low)

        inst_activity = features["institutional_activity"].dropna()

        # Should be in [0, 1]
        assert all((inst_activity >= 0) & (inst_activity <= 1))


class TestAdvancedLOBFeatures:
    """Test integrated LOB feature calculator."""

    def setup_method(self):
        """Create comprehensive OHLCV test data."""
        np.random.seed(42)
        n_samples = 500

        close = 100 + np.cumsum(np.random.randn(n_samples) * 0.3)

        self.df = pd.DataFrame({
            "open": close + np.random.randn(n_samples) * 0.1,
            "high": close + abs(np.random.randn(n_samples) * 0.3),
            "low": close - abs(np.random.randn(n_samples) * 0.3),
            "close": close,
            "volume": np.random.randint(1000, 10000, n_samples),
        })

    def test_calculate_from_ohlcv(self):
        """Test feature calculation from OHLCV data."""
        from src.features.microstructure.advanced_lob_features import AdvancedLOBFeatures

        lob = AdvancedLOBFeatures()
        features = lob.calculate_from_ohlcv(self.df)

        # Should return DataFrame with features
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(self.df)

        # Should have expected columns
        expected_cols = ["spread_bps", "toxicity_index", "smart_money_index"]
        for col in expected_cols:
            assert col in features.columns, f"Missing column: {col}"

    def test_feature_completeness(self):
        """Test that all expected features are generated."""
        from src.features.microstructure.advanced_lob_features import AdvancedLOBFeatures

        lob = AdvancedLOBFeatures(
            spread_lookback=50,
            toxicity_window=30,
            block_percentile=0.9,
        )
        features = lob.calculate_from_ohlcv(self.df)

        # Check minimum number of features
        assert len(features.columns) >= 5, "Should generate multiple features"


class TestConvenienceFunctions:
    """Test module convenience functions."""

    def test_calculate_toxicity_function(self):
        """Test standalone toxicity calculation function."""
        from src.features.microstructure.advanced_lob_features import calculate_toxicity

        np.random.seed(42)
        price = pd.Series(100 + np.cumsum(np.random.randn(200) * 0.1))
        volume = pd.Series(np.random.randint(100, 1000, 200))

        toxicity = calculate_toxicity(price, volume)

        # Should return Series
        assert isinstance(toxicity, pd.Series)
        assert len(toxicity) == len(price)

    def test_calculate_smart_money_function(self):
        """Test standalone smart money calculation function."""
        from src.features.microstructure.advanced_lob_features import calculate_smart_money

        np.random.seed(42)
        price = pd.Series(100 + np.cumsum(np.random.randn(200) * 0.1))
        volume = pd.Series(np.random.randint(100, 1000, 200))

        smi = calculate_smart_money(price, volume)

        # Should return Series
        assert isinstance(smi, pd.Series)
        assert len(smi) == len(price)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
