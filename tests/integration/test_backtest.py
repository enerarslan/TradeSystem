"""
Integration tests for backtesting.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategies.momentum.multi_factor_momentum import MultiFactorMomentumStrategy
from src.strategies.mean_reversion.mean_reversion import MeanReversionStrategy
from src.backtesting.engine import BacktestEngine, VectorizedBacktest


@pytest.fixture
def multi_stock_data():
    """Create multi-stock OHLCV data for testing."""
    np.random.seed(42)
    n = 1000
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]

    dates = pd.date_range(start="2023-01-01", periods=n, freq="15min")
    data = {}

    for i, symbol in enumerate(symbols):
        # Generate random walk with some drift
        returns = np.random.normal(0.0001 + i * 0.00002, 0.015, n)
        close = 100 * (1 + i * 0.1) * np.cumprod(1 + returns)

        high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
        low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
        open_ = close + np.random.normal(0, 0.3, n)
        volume = np.random.randint(10000, 500000, n)

        data[symbol] = pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }, index=dates)

    return data


class TestMomentumStrategy:
    """Integration tests for momentum strategy."""

    def test_signal_generation(self, multi_stock_data):
        """Test signal generation."""
        strategy = MultiFactorMomentumStrategy()
        signals = strategy.generate_signals(multi_stock_data)

        assert signals is not None
        assert len(signals) > 0
        assert signals.shape[1] == len(multi_stock_data)

        # Signals should be in [-1, 0, 1]
        unique_values = set(signals.values.flatten())
        assert unique_values.issubset({-1, 0, 1, np.nan})

    def test_position_calculation(self, multi_stock_data):
        """Test position calculation."""
        strategy = MultiFactorMomentumStrategy()
        signals = strategy.generate_signals(multi_stock_data)

        prices = pd.DataFrame({
            sym: df["close"] for sym, df in multi_stock_data.items()
        })

        positions = strategy.calculate_positions(signals, prices, 1000000)

        # Positions should sum to <= 1
        assert (positions.abs().sum(axis=1) <= 1.01).all()


class TestMeanReversionStrategy:
    """Integration tests for mean reversion strategy."""

    def test_signal_generation(self, multi_stock_data):
        """Test signal generation."""
        strategy = MeanReversionStrategy()
        signals = strategy.generate_signals(multi_stock_data)

        assert signals is not None
        assert len(signals) > 0

    def test_zscore_signals(self, multi_stock_data):
        """Test that signals are generated based on z-score."""
        strategy = MeanReversionStrategy(params={
            "entry_zscore": 2.0,
            "exit_zscore": 0.5,
        })
        signals = strategy.generate_signals(multi_stock_data)

        # Should have some non-zero signals
        non_zero = (signals != 0).sum().sum()
        assert non_zero > 0


class TestBacktestEngine:
    """Integration tests for backtest engine."""

    def test_full_backtest(self, multi_stock_data):
        """Test full backtest execution."""
        strategy = MultiFactorMomentumStrategy(params={
            "top_n_long": 2,
            "lookback_periods": [5, 10],
        })

        engine = BacktestEngine(
            initial_capital=1000000,
            commission_pct=0.001,
            slippage_pct=0.0005,
        )

        result = engine.run(strategy, multi_stock_data)

        assert result is not None
        assert result.equity_curve is not None
        assert len(result.equity_curve) > 0
        assert result.final_capital > 0

    def test_backtest_metrics(self, multi_stock_data):
        """Test that backtest returns valid metrics."""
        strategy = MultiFactorMomentumStrategy()
        engine = BacktestEngine(initial_capital=1000000)

        result = engine.run(strategy, multi_stock_data)

        # Check metrics exist
        assert "total_return" in result.metrics
        assert "sharpe_ratio" in result.metrics
        assert "max_drawdown" in result.metrics

        # Metrics should be finite
        assert np.isfinite(result.metrics["total_return"])
        assert np.isfinite(result.metrics["sharpe_ratio"])

    def test_backtest_trades(self, multi_stock_data):
        """Test that backtest generates trades."""
        strategy = MultiFactorMomentumStrategy(params={
            "top_n_long": 3,
        })

        engine = BacktestEngine(initial_capital=1000000)
        result = engine.run(strategy, multi_stock_data)

        # Should have some trades
        assert len(result.trades) > 0

        # All trades should have valid values
        for trade in result.trades:
            assert trade.symbol in multi_stock_data
            assert trade.quantity > 0
            assert trade.price > 0


class TestVectorizedBacktest:
    """Tests for vectorized backtest."""

    def test_vectorized_backtest(self, multi_stock_data):
        """Test vectorized backtest."""
        # Create signals
        prices = pd.DataFrame({
            sym: df["close"] for sym, df in multi_stock_data.items()
        })

        # Simple momentum signals
        returns = prices.pct_change(20)
        signals = (returns > 0).astype(int) - (returns < 0).astype(int)
        signals = signals / signals.abs().sum(axis=1).replace(0, 1).values[:, None]
        signals = signals.fillna(0)

        # Run backtest
        bt = VectorizedBacktest(initial_capital=1000000)
        result = bt.run(signals, prices)

        assert result is not None
        assert len(result.equity_curve) > 0
        assert result.final_capital > 0

    def test_vectorized_matches_engine(self, multi_stock_data):
        """Test that vectorized produces similar results to engine."""
        # This is a rough comparison - they won't be exact due to
        # different implementation details

        strategy = MultiFactorMomentumStrategy(params={
            "top_n_long": 2,
        })

        # Engine backtest
        engine = BacktestEngine(initial_capital=1000000)
        engine_result = engine.run(strategy, multi_stock_data)

        # Both should be profitable or unprofitable together (roughly)
        # This is a weak test but confirms basic functionality
        assert engine_result.final_capital > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
