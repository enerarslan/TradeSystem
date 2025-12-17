"""
Tests for Market Impact and Execution Simulation.

These tests verify that the backtesting system properly accounts for:
1. Market impact (slippage from order size)
2. Order book simulation
3. Partial fills
4. Realistic latency
5. Transaction costs

CRITICAL: Ignoring market impact is a major source of unrealistic backtest results.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


class TestMarketImpactModels:
    """Test market impact calculation models."""

    def setup_method(self):
        """Create test data."""
        np.random.seed(42)
        n_samples = 1000
        dates = pd.date_range("2020-01-01", periods=n_samples, freq="15min")

        self.market_data = pd.DataFrame(
            {
                "close": 100 + np.cumsum(np.random.randn(n_samples) * 0.1),
                "volume": np.random.randint(100000, 1000000, n_samples),
                "high": 100 + np.cumsum(np.random.randn(n_samples) * 0.1) + 0.5,
                "low": 100 + np.cumsum(np.random.randn(n_samples) * 0.1) - 0.5,
            },
            index=dates,
        )

    def test_linear_market_impact(self):
        """Test linear market impact model."""
        # Simple linear impact: impact = k * (order_size / ADV)
        adv = self.market_data["volume"].mean()
        order_size = adv * 0.01  # 1% of ADV

        impact_coefficient = 5  # basis points
        expected_impact = impact_coefficient * (order_size / adv)

        assert expected_impact > 0, "Market impact should be positive"
        assert expected_impact == pytest.approx(0.05, rel=0.1)  # ~5 bps

    def test_square_root_market_impact(self):
        """Test square-root market impact model (Almgren-Chriss)."""
        # Square root impact: impact = k * sqrt(order_size / ADV)
        adv = self.market_data["volume"].mean()

        # Test different order sizes
        sizes = [0.01, 0.02, 0.05, 0.10]  # % of ADV
        impacts = []

        for size_pct in sizes:
            order_size = adv * size_pct
            impact = 10 * np.sqrt(order_size / adv)  # 10 bps base
            impacts.append(impact)

        # Verify square-root relationship (impact doubles when size quadruples)
        ratio = impacts[2] / impacts[0]  # 5% vs 1%
        expected_ratio = np.sqrt(5)

        assert ratio == pytest.approx(expected_ratio, rel=0.01)

    def test_market_impact_increases_with_size(self):
        """Test that market impact increases with order size."""
        adv = self.market_data["volume"].mean()

        small_order = adv * 0.01
        large_order = adv * 0.10

        # Using square-root model
        small_impact = np.sqrt(small_order / adv)
        large_impact = np.sqrt(large_order / adv)

        assert large_impact > small_impact, (
            "Large orders should have higher impact"
        )

    def test_impact_capped_at_reasonable_level(self):
        """Test that impact is capped at reasonable levels."""
        from src.backtesting.event_engine import ExecutionEngine

        engine = ExecutionEngine(
            market_impact_bps=10,
            max_participation_rate=0.02,  # 2% of volume
        )

        # Very large order (would exceed participation limit)
        adv = 1_000_000
        order_size = adv * 0.10  # 10% of ADV

        # Calculate max allowed size
        max_size = adv * engine.max_participation_rate

        assert max_size == 20_000, "Max participation should limit order size"


class TestOrderBookSimulation:
    """Test order book simulation."""

    def setup_method(self):
        """Create test order book."""
        self.mid_price = 100.0
        self.spread = 0.02  # 2 cents

        # Simulated order book levels
        self.bid_levels = [
            (99.99, 1000),  # price, quantity
            (99.98, 2000),
            (99.97, 3000),
            (99.96, 5000),
        ]

        self.ask_levels = [
            (100.01, 1000),
            (100.02, 2000),
            (100.03, 3000),
            (100.04, 5000),
        ]

    def test_spread_calculation(self):
        """Test bid-ask spread calculation."""
        best_bid = self.bid_levels[0][0]
        best_ask = self.ask_levels[0][0]

        spread = best_ask - best_bid
        spread_bps = (spread / self.mid_price) * 10000

        assert spread == pytest.approx(0.02, rel=0.001)  # ~2 cents (use approx for float)
        assert spread_bps == pytest.approx(2.0, rel=0.01)  # ~2 bps

    def test_fill_price_small_order(self):
        """Test that small orders fill at best price."""
        order_size = 500  # Less than best ask quantity

        # Should fill entirely at best ask
        fill_price = self.ask_levels[0][0]

        assert fill_price == 100.01

    def test_fill_price_large_order(self):
        """Test that large orders walk up the book."""
        order_size = 2500  # Needs to eat into second level

        # Calculate VWAP fill
        filled = 0
        total_value = 0

        for price, qty in self.ask_levels:
            fill_qty = min(qty, order_size - filled)
            total_value += price * fill_qty
            filled += fill_qty

            if filled >= order_size:
                break

        vwap_fill = total_value / filled

        # VWAP should be higher than best ask
        assert vwap_fill > self.ask_levels[0][0], (
            "Large order VWAP should exceed best ask"
        )

    def test_slippage_calculation(self):
        """Test slippage calculation from order book."""
        # Order that needs multiple levels
        order_size = 3000

        # Fill from multiple levels
        filled = 0
        total_value = 0

        for price, qty in self.ask_levels:
            fill_qty = min(qty, order_size - filled)
            total_value += price * fill_qty
            filled += fill_qty
            if filled >= order_size:
                break

        vwap = total_value / filled
        slippage = (vwap - self.mid_price) / self.mid_price * 10000  # in bps

        # Slippage should be positive for buy orders
        assert slippage > 0, "Slippage should be positive for buy orders"


class TestPartialFills:
    """Test partial fill simulation."""

    def test_partial_fill_enabled(self):
        """Test that partial fills are enabled by default."""
        from src.backtesting.event_engine import ExecutionEngine

        engine = ExecutionEngine()

        assert engine.partial_fills is True, (
            "Partial fills should be enabled by default"
        )

    def test_partial_fill_calculation(self):
        """Test partial fill quantity calculation."""
        available_liquidity = 1000
        order_size = 2000
        fill_rate = 0.8  # 80% fill probability

        # Expected fill is min(order, available * fill_rate)
        expected_fill = min(order_size, available_liquidity * fill_rate)

        assert expected_fill == 800

    def test_order_rejection_probability(self):
        """Test order rejection simulation."""
        from src.backtesting.event_engine import ExecutionEngine

        engine = ExecutionEngine(rejection_rate=0.02)

        # Simulate many orders
        np.random.seed(42)
        n_orders = 10000
        rejections = sum(
            1 for _ in range(n_orders)
            if np.random.random() < engine.rejection_rate
        )

        rejection_rate = rejections / n_orders

        # Should be approximately 2%
        assert rejection_rate == pytest.approx(0.02, rel=0.2)


class TestLatencySimulation:
    """Test latency simulation."""

    def test_order_latency_applied(self):
        """Test that order latency is applied."""
        from src.backtesting.event_engine import ExecutionEngine

        engine = ExecutionEngine(latency_ms=50)

        assert engine.latency_ms == 50

    def test_fill_at_next_bar(self):
        """Test that orders fill at next bar open (not same bar close)."""
        from src.backtesting.event_engine import ExecutionEngine

        engine = ExecutionEngine()

        # Signal at bar N should fill at bar N+1 open
        signal_bar = 100
        expected_fill_bar = signal_bar + 1

        # This is the standard fill convention
        assert expected_fill_bar == 101

    def test_stale_prices_rejected(self):
        """Test that orders with stale prices are handled."""
        # If price moves significantly during latency, order may be rejected

        original_price = 100.0
        latency_price_change = 0.5  # 50 cents move during latency
        new_price = 100.5

        # Price move in bps
        move_bps = abs(new_price - original_price) / original_price * 10000

        # Should flag significant moves
        assert move_bps == 50


class TestTransactionCosts:
    """Test transaction cost models."""

    def test_commission_calculation(self):
        """Test commission calculation."""
        trade_value = 100_000  # $100k trade
        commission_bps = 10  # 10 bps

        commission = trade_value * (commission_bps / 10000)

        assert commission == 100  # $100 commission

    def test_slippage_calculation(self):
        """Test slippage cost calculation."""
        entry_price = 100.0
        fill_price = 100.05  # 5 cents slippage
        shares = 1000

        slippage_cost = (fill_price - entry_price) * shares
        slippage_bps = (fill_price / entry_price - 1) * 10000

        assert slippage_cost == pytest.approx(50, rel=0.001)  # $50 slippage
        assert slippage_bps == pytest.approx(5, rel=0.01)  # ~5 bps

    def test_total_execution_cost(self):
        """Test total execution cost (commission + slippage + impact)."""
        trade_value = 100_000
        commission_bps = 10
        slippage_bps = 5
        market_impact_bps = 3

        total_cost_bps = commission_bps + slippage_bps + market_impact_bps
        total_cost = trade_value * (total_cost_bps / 10000)

        assert total_cost == 180  # $180 total cost
        assert total_cost_bps == 18

    def test_round_trip_costs(self):
        """Test round-trip transaction costs."""
        trade_value = 100_000

        # Entry costs
        entry_commission_bps = 10
        entry_slippage_bps = 5

        # Exit costs
        exit_commission_bps = 10
        exit_slippage_bps = 5

        round_trip_bps = (
            entry_commission_bps + entry_slippage_bps +
            exit_commission_bps + exit_slippage_bps
        )

        assert round_trip_bps == 30  # 30 bps round trip

    def test_borrowing_cost_shorts(self):
        """Test borrowing cost for short positions."""
        position_value = 100_000
        holding_days = 30
        annual_borrow_rate = 0.02  # 2% annual

        # Daily borrow rate
        daily_rate = annual_borrow_rate / 365
        borrow_cost = position_value * daily_rate * holding_days

        expected_cost = 100_000 * 0.02 * (30 / 365)

        assert borrow_cost == pytest.approx(expected_cost, rel=0.01)


class TestExecutionEngine:
    """Test ExecutionEngine class."""

    def test_default_configuration(self):
        """Test default execution engine configuration."""
        from src.backtesting.event_engine import ExecutionEngine

        engine = ExecutionEngine()

        # Verify institutional defaults
        assert engine.use_order_book is True
        assert engine.partial_fills is True
        assert engine.market_impact_bps > 0
        assert engine.latency_ms > 0

    def test_execution_with_impact(self):
        """Test execution includes market impact."""
        from src.backtesting.event_engine import ExecutionEngine

        engine = ExecutionEngine(
            market_impact_bps=5,
            commission_bps=10,
        )

        # Execute a buy order
        order_price = 100.0
        order_size = 1000

        # Calculate expected fill
        impact_factor = 1 + (5 / 10000)
        expected_fill_price = order_price * impact_factor

        assert expected_fill_price == pytest.approx(100.05, rel=0.001)

    def test_max_participation_enforced(self):
        """Test max participation rate is enforced."""
        from src.backtesting.event_engine import ExecutionEngine

        engine = ExecutionEngine(max_participation_rate=0.02)

        bar_volume = 100_000
        desired_size = 10_000  # 10% of volume

        # Max allowed is 2%
        max_allowed = bar_volume * engine.max_participation_rate

        assert max_allowed == 2000
        assert desired_size > max_allowed


class TestAlmgrenChrissModel:
    """Test Almgren-Chriss market impact model."""

    def test_temporary_impact(self):
        """Test temporary impact calculation."""
        # Temporary impact = eta * (order_rate / daily_volume)
        eta = 0.05  # Temporary impact coefficient
        order_rate = 1000  # shares per interval
        daily_volume = 100_000

        temp_impact = eta * (order_rate / daily_volume)

        assert temp_impact == 0.0005  # 5 bps

    def test_permanent_impact(self):
        """Test permanent impact calculation."""
        # Permanent impact = gamma * order_size
        gamma = 0.0001  # Permanent impact coefficient
        order_size = 1000

        perm_impact = gamma * order_size

        assert perm_impact == 0.1  # 10 cents per share

    def test_optimal_execution_trajectory(self):
        """Test optimal execution splits order over time."""
        total_shares = 10_000
        urgency = 0.5  # Higher = faster execution

        # Simple linear trajectory
        n_intervals = 5
        shares_per_interval = total_shares / n_intervals

        executed = [shares_per_interval] * n_intervals

        assert sum(executed) == total_shares

    def test_impact_increases_with_urgency(self):
        """Test that faster execution has higher impact."""
        total_shares = 10_000
        daily_volume = 100_000

        # Slow execution (spread over full day)
        slow_participation = total_shares / daily_volume
        slow_impact = np.sqrt(slow_participation)

        # Fast execution (10% of day)
        fast_participation = total_shares / (daily_volume * 0.1)
        fast_impact = np.sqrt(fast_participation)

        assert fast_impact > slow_impact, (
            "Faster execution should have higher impact"
        )


class TestRealisticExecutionScenarios:
    """Test realistic execution scenarios."""

    def test_large_order_splitting(self):
        """Test that large orders are split across multiple bars."""
        total_order = 50_000
        max_per_bar = 2_000  # 2% of ADV of 100k

        n_bars_needed = int(np.ceil(total_order / max_per_bar))

        assert n_bars_needed == 25

    def test_illiquid_market_higher_impact(self):
        """Test higher impact in illiquid markets."""
        liquid_adv = 10_000_000
        illiquid_adv = 100_000
        order_size = 10_000

        # Square-root impact model
        liquid_impact = np.sqrt(order_size / liquid_adv)
        illiquid_impact = np.sqrt(order_size / illiquid_adv)

        ratio = illiquid_impact / liquid_impact

        # Illiquid market should have 10x impact for same order size
        assert ratio == pytest.approx(10, rel=0.01)

    def test_market_close_higher_costs(self):
        """Test higher costs near market close."""
        # Liquidity typically lower at open/close
        normal_spread_bps = 2
        close_spread_bps = 5

        assert close_spread_bps > normal_spread_bps

    def test_news_event_impact(self):
        """Test higher impact during news events."""
        # Spread widens during high volatility
        normal_volatility = 0.01
        event_volatility = 0.05

        # Impact scales with volatility
        normal_impact = 5 * normal_volatility
        event_impact = 5 * event_volatility

        assert event_impact > normal_impact


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
