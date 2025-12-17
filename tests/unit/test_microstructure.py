"""
Unit tests for Microstructure module exports.

Tests the critical fix:
- OrderBookDynamics is properly exported in __all__
"""

import pytest
import numpy as np
import pandas as pd


class TestMicrostructureExports:
    """Test that all microstructure classes are properly exported."""

    def test_order_book_dynamics_import(self):
        """Test that OrderBookDynamics can be imported from microstructure module."""
        from src.features.microstructure import OrderBookDynamics

        assert OrderBookDynamics is not None

    def test_order_book_dynamics_in_all(self):
        """Test that OrderBookDynamics is in __all__ list."""
        from src.features import microstructure

        assert "OrderBookDynamics" in microstructure.__all__

    def test_all_microstructure_exports(self):
        """Test all expected classes are exported."""
        from src.features import microstructure

        expected_exports = [
            "OrderFlowImbalance",
            "calculate_ofi",
            "VPIN",
            "calculate_vpin",
            "KyleLambda",
            "calculate_kyle_lambda",
            "RollSpread",
            "AmihudIlliquidity",
            "OrderBookDynamics",
        ]

        for export in expected_exports:
            assert export in microstructure.__all__, f"{export} not in __all__"
            assert hasattr(microstructure, export), f"{export} not importable"


class TestOrderBookDynamics:
    """Test OrderBookDynamics functionality."""

    def test_calculate_obi(self):
        """Test Order Book Imbalance calculation."""
        from src.features.microstructure import OrderBookDynamics

        bid_size = pd.Series([100, 150, 200, 180, 120])
        ask_size = pd.Series([120, 100, 150, 200, 100])

        obi = OrderBookDynamics.calculate_obi(bid_size, ask_size)

        assert isinstance(obi, pd.Series)
        assert len(obi) == len(bid_size)
        # OBI should be between -1 and 1
        assert all(obi >= -1)
        assert all(obi <= 1)

    def test_calculate_wmp(self):
        """Test Weighted Mid Price calculation."""
        from src.features.microstructure import OrderBookDynamics

        bid_price = pd.Series([99.0, 99.5, 100.0, 100.5, 101.0])
        bid_size = pd.Series([100, 150, 200, 180, 120])
        ask_price = pd.Series([100.0, 100.5, 101.0, 101.5, 102.0])
        ask_size = pd.Series([120, 100, 150, 200, 100])

        wmp = OrderBookDynamics.calculate_wmp(bid_price, bid_size, ask_price, ask_size)

        assert isinstance(wmp, pd.Series)
        assert len(wmp) == len(bid_price)
        # WMP should be between bid and ask
        assert all(wmp >= bid_price)
        assert all(wmp <= ask_price)

    def test_calculate_mid_price_divergence(self):
        """Test Mid Price Divergence calculation."""
        from src.features.microstructure import OrderBookDynamics

        mid_price = pd.Series([100.0, 100.5, 101.0, 100.5, 100.0])
        wmp = pd.Series([100.1, 100.4, 101.2, 100.6, 99.9])

        divergence = OrderBookDynamics.calculate_mid_price_divergence(mid_price, wmp)

        assert isinstance(divergence, pd.Series)
        assert len(divergence) == len(mid_price)


class TestMainFeaturesModuleExports:
    """Test that microstructure exports are available from main features module."""

    def test_features_module_has_microstructure(self):
        """Test importing microstructure classes from main features module."""
        from src.features import (
            OrderFlowImbalance,
            calculate_ofi,
            VPIN,
            calculate_vpin,
            KyleLambda,
            calculate_kyle_lambda,
            RollSpread,
            AmihudIlliquidity,
            OrderBookDynamics,
        )

        # All should be importable
        assert OrderFlowImbalance is not None
        assert calculate_ofi is not None
        assert VPIN is not None
        assert calculate_vpin is not None
        assert KyleLambda is not None
        assert calculate_kyle_lambda is not None
        assert RollSpread is not None
        assert AmihudIlliquidity is not None
        assert OrderBookDynamics is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
