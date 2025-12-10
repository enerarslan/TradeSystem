"""
Information-Driven Bar Aggregation Module
==========================================

JPMorgan-level bar construction that moves beyond time-based aggregation.

Traditional Time Bars have a critical flaw: they sample uniformly in time,
but market activity varies dramatically (high volume at open/close, low at lunch).
This creates non-stationary, heteroscedastic returns that hurt ML models.

This module implements:
1. Volume Bars - New bar when X shares traded
2. Dollar Bars - New bar when $X traded (preferred)
3. Tick Bars - New bar every N ticks
4. Imbalance Bars - New bar when order flow imbalance threshold reached
5. Run Bars - New bar when sequence of buy/sell runs ends

Why Dollar Bars?
- Normalizes activity across different price stocks ($100 stock vs $10 stock)
- Recovers approximate normality (Gaussian) in returns
- Reduces heteroscedasticity
- Significantly improves ML model performance

Reference: "Advances in Financial Machine Learning" by López de Prado

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Iterator, Generator

import numpy as np
import polars as pl
from numpy.typing import NDArray

from config.settings import get_logger
from data.tick_data import TickDataLoader, QuoteDataLoader, TradeDirection

logger = get_logger(__name__)


# =============================================================================
# ENUMS & CONFIGURATION
# =============================================================================

class BarType(str, Enum):
    """Types of information-driven bars."""
    TIME = "time"             # Traditional time bars
    TICK = "tick"             # Fixed number of ticks per bar
    VOLUME = "volume"         # Fixed volume per bar
    DOLLAR = "dollar"         # Fixed dollar volume per bar
    IMBALANCE_TICK = "imbalance_tick"    # Tick imbalance bars
    IMBALANCE_VOLUME = "imbalance_volume"  # Volume imbalance bars
    IMBALANCE_DOLLAR = "imbalance_dollar"  # Dollar imbalance bars
    RUN_TICK = "run_tick"     # Tick run bars
    RUN_VOLUME = "run_volume" # Volume run bars
    RUN_DOLLAR = "run_dollar" # Dollar run bars


@dataclass
class BarConfig:
    """Configuration for bar generation."""
    bar_type: BarType = BarType.DOLLAR

    # Thresholds (only one is used based on bar_type)
    tick_threshold: int = 1000              # Ticks per bar
    volume_threshold: float = 50000         # Shares per bar
    dollar_threshold: float = 10_000_000    # Dollars per bar ($10M default)

    # Imbalance/Run bar parameters
    imbalance_window: int = 100             # Window for expected imbalance
    imbalance_sensitivity: float = 2.0      # Multiplier for threshold

    # Run bar parameters
    run_window: int = 100                   # Window for expected run length

    # Output options
    include_vwap: bool = True               # Include VWAP in output
    include_trade_count: bool = True        # Include trade count
    include_imbalance: bool = True          # Include order flow imbalance


@dataclass
class Bar:
    """Represents a single aggregated bar."""
    timestamp_open: datetime
    timestamp_close: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    dollar_volume: float
    trade_count: int
    vwap: float
    buy_volume: float = 0.0
    sell_volume: float = 0.0

    @property
    def imbalance(self) -> float:
        """Order flow imbalance: (buy - sell) / (buy + sell)."""
        total = self.buy_volume + self.sell_volume
        if total > 0:
            return (self.buy_volume - self.sell_volume) / total
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp_close,  # Use close time as bar timestamp
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "dollar_volume": self.dollar_volume,
            "vwap": self.vwap,
            "trade_count": self.trade_count,
            "buy_volume": self.buy_volume,
            "sell_volume": self.sell_volume,
            "imbalance": self.imbalance,
        }


# =============================================================================
# BAR AGGREGATOR BASE CLASS
# =============================================================================

class BarAggregator(ABC):
    """Abstract base class for bar aggregation."""

    def __init__(self, config: BarConfig | None = None):
        """Initialize aggregator."""
        self.config = config or BarConfig()

    @abstractmethod
    def aggregate(self, ticks: pl.DataFrame) -> list[Bar]:
        """
        Aggregate tick data into bars.

        Args:
            ticks: DataFrame with columns: timestamp, price, size, direction

        Returns:
            List of Bar objects
        """
        pass

    def to_dataframe(self, bars: list[Bar]) -> pl.DataFrame:
        """Convert list of bars to DataFrame."""
        if not bars:
            return pl.DataFrame()

        records = [bar.to_dict() for bar in bars]
        return pl.DataFrame(records)

    def _create_bar(
        self,
        ticks: list[dict[str, Any]],
    ) -> Bar:
        """Create a bar from a list of tick records."""
        if not ticks:
            raise ValueError("Cannot create bar from empty ticks")

        prices = [t["price"] for t in ticks]
        sizes = [t["size"] for t in ticks]
        timestamps = [t["timestamp"] for t in ticks]

        # Calculate dollar volumes
        dollar_volumes = [p * s for p, s in zip(prices, sizes)]

        # VWAP calculation
        total_dollar = sum(dollar_volumes)
        total_volume = sum(sizes)
        vwap = total_dollar / total_volume if total_volume > 0 else prices[-1]

        # Buy/Sell volume
        buy_volume = sum(
            s for t, s in zip(ticks, sizes)
            if t.get("direction") == TradeDirection.BUY.value
        )
        sell_volume = sum(
            s for t, s in zip(ticks, sizes)
            if t.get("direction") == TradeDirection.SELL.value
        )

        return Bar(
            timestamp_open=timestamps[0],
            timestamp_close=timestamps[-1],
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            volume=total_volume,
            dollar_volume=total_dollar,
            trade_count=len(ticks),
            vwap=vwap,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
        )


# =============================================================================
# DOLLAR BAR AGGREGATOR (PREFERRED)
# =============================================================================

class DollarBarAggregator(BarAggregator):
    """
    Dollar Bar Aggregator.

    Creates a new bar when cumulative dollar volume reaches threshold.

    Benefits:
    - Normalizes activity across different price levels
    - Approximately Gaussian returns (Central Limit Theorem)
    - Reduces autocorrelation in returns
    - Better for ML models

    Example:
        aggregator = DollarBarAggregator(BarConfig(dollar_threshold=10_000_000))
        bars = aggregator.aggregate(tick_data)
        # Each bar represents $10M of traded value

    Typical thresholds:
    - Mega-cap (AAPL, MSFT): $50M - $100M
    - Large-cap: $10M - $50M
    - Mid-cap: $1M - $10M
    - Small-cap: $100K - $1M
    """

    def aggregate(self, ticks: pl.DataFrame) -> list[Bar]:
        """Aggregate ticks into dollar bars."""
        if len(ticks) == 0:
            return []

        threshold = self.config.dollar_threshold
        bars: list[Bar] = []

        # Convert to records for iteration
        records = ticks.to_dicts()

        current_bar_ticks: list[dict] = []
        cumulative_dollar = 0.0

        for tick in records:
            price = tick["price"]
            size = tick["size"]
            dollar_value = price * size

            current_bar_ticks.append(tick)
            cumulative_dollar += dollar_value

            # Check if threshold reached
            if cumulative_dollar >= threshold:
                bars.append(self._create_bar(current_bar_ticks))
                current_bar_ticks = []
                cumulative_dollar = 0.0

        # Handle remaining ticks (partial bar)
        if current_bar_ticks and len(current_bar_ticks) >= 10:
            # Only create bar if enough ticks to be meaningful
            bars.append(self._create_bar(current_bar_ticks))

        logger.info(f"Created {len(bars)} dollar bars from {len(ticks)} ticks "
                   f"(threshold: ${threshold/1e6:.1f}M)")

        return bars


# =============================================================================
# VOLUME BAR AGGREGATOR
# =============================================================================

class VolumeBarAggregator(BarAggregator):
    """
    Volume Bar Aggregator.

    Creates a new bar when cumulative volume reaches threshold.

    Benefits:
    - Samples uniformly in volume space (not time)
    - Each bar represents same "information content"
    - More stable volatility estimates

    For most cases, Dollar Bars are preferred over Volume Bars because
    they normalize for price level differences.
    """

    def aggregate(self, ticks: pl.DataFrame) -> list[Bar]:
        """Aggregate ticks into volume bars."""
        if len(ticks) == 0:
            return []

        threshold = self.config.volume_threshold
        bars: list[Bar] = []

        records = ticks.to_dicts()

        current_bar_ticks: list[dict] = []
        cumulative_volume = 0.0

        for tick in records:
            size = tick["size"]

            current_bar_ticks.append(tick)
            cumulative_volume += size

            if cumulative_volume >= threshold:
                bars.append(self._create_bar(current_bar_ticks))
                current_bar_ticks = []
                cumulative_volume = 0.0

        if current_bar_ticks and len(current_bar_ticks) >= 10:
            bars.append(self._create_bar(current_bar_ticks))

        logger.info(f"Created {len(bars)} volume bars from {len(ticks)} ticks "
                   f"(threshold: {threshold:,.0f} shares)")

        return bars


# =============================================================================
# TICK BAR AGGREGATOR
# =============================================================================

class TickBarAggregator(BarAggregator):
    """
    Tick Bar Aggregator.

    Creates a new bar every N ticks.

    Simplest information-driven bar, but less useful than Dollar/Volume
    because tick sizes vary dramatically.
    """

    def aggregate(self, ticks: pl.DataFrame) -> list[Bar]:
        """Aggregate ticks into tick bars."""
        if len(ticks) == 0:
            return []

        threshold = self.config.tick_threshold
        bars: list[Bar] = []

        records = ticks.to_dicts()
        n = len(records)

        for i in range(0, n, threshold):
            batch = records[i:i + threshold]
            if len(batch) >= threshold // 2:  # At least half a bar
                bars.append(self._create_bar(batch))

        logger.info(f"Created {len(bars)} tick bars from {len(ticks)} ticks "
                   f"(threshold: {threshold} ticks)")

        return bars


# =============================================================================
# IMBALANCE BAR AGGREGATOR
# =============================================================================

class ImbalanceBarAggregator(BarAggregator):
    """
    Imbalance Bar Aggregator.

    Creates a new bar when order flow imbalance deviates significantly
    from its expected value. This captures sudden shifts in market sentiment.

    Imbalance = |Cumulative(Buy Ticks) - Cumulative(Sell Ticks)|

    When imbalance exceeds expected imbalance by threshold, create new bar.

    These bars have variable length and capture information events.

    Reference: López de Prado, "Advances in Financial Machine Learning", Ch. 2.4
    """

    def aggregate(self, ticks: pl.DataFrame) -> list[Bar]:
        """Aggregate ticks into imbalance bars."""
        if len(ticks) == 0:
            return []

        if "direction" not in ticks.columns:
            logger.warning("No direction column - using tick signs instead")
            ticks = self._infer_direction(ticks)

        records = ticks.to_dicts()
        n = len(records)

        bars: list[Bar] = []
        current_bar_ticks: list[dict] = []

        # Track imbalance
        cumulative_imbalance = 0.0

        # Expected imbalance (rolling average)
        window = self.config.imbalance_window
        recent_bar_imbalances: list[float] = []

        for i, tick in enumerate(records):
            direction = tick.get("direction", TradeDirection.UNKNOWN.value)

            # Add tick to current bar
            current_bar_ticks.append(tick)

            # Update imbalance based on bar type
            if self.config.bar_type == BarType.IMBALANCE_TICK:
                delta = 1 if direction == TradeDirection.BUY.value else -1
            elif self.config.bar_type == BarType.IMBALANCE_VOLUME:
                size = tick["size"]
                delta = size if direction == TradeDirection.BUY.value else -size
            else:  # IMBALANCE_DOLLAR
                dollar = tick["price"] * tick["size"]
                delta = dollar if direction == TradeDirection.BUY.value else -dollar

            cumulative_imbalance += delta

            # Calculate expected imbalance
            if recent_bar_imbalances:
                expected = np.mean(np.abs(recent_bar_imbalances))
            else:
                expected = abs(cumulative_imbalance) * 2  # Initial estimate

            # Threshold
            threshold = expected * self.config.imbalance_sensitivity

            # Check if should create bar
            if abs(cumulative_imbalance) >= threshold and len(current_bar_ticks) >= 10:
                bar = self._create_bar(current_bar_ticks)
                bars.append(bar)

                # Track for expected calculation
                recent_bar_imbalances.append(abs(cumulative_imbalance))
                if len(recent_bar_imbalances) > window:
                    recent_bar_imbalances.pop(0)

                # Reset
                current_bar_ticks = []
                cumulative_imbalance = 0.0

        # Handle remaining ticks
        if current_bar_ticks and len(current_bar_ticks) >= 10:
            bars.append(self._create_bar(current_bar_ticks))

        logger.info(f"Created {len(bars)} imbalance bars from {len(ticks)} ticks")

        return bars

    def _infer_direction(self, ticks: pl.DataFrame) -> pl.DataFrame:
        """Infer direction from price changes (tick test)."""
        prices = ticks["price"].to_numpy()
        n = len(prices)

        directions = [TradeDirection.UNKNOWN.value] * n

        for i in range(1, n):
            if prices[i] > prices[i-1]:
                directions[i] = TradeDirection.BUY.value
            elif prices[i] < prices[i-1]:
                directions[i] = TradeDirection.SELL.value
            else:
                directions[i] = directions[i-1]

        return ticks.with_columns(pl.Series("direction", directions))


# =============================================================================
# RUN BAR AGGREGATOR
# =============================================================================

class RunBarAggregator(BarAggregator):
    """
    Run Bar Aggregator.

    Creates a new bar when the length of a run (sequence of same-direction
    trades) exceeds its expected value.

    This detects when informed traders are executing large orders that
    move the market in one direction.

    Reference: López de Prado, "Advances in Financial Machine Learning", Ch. 2.4
    """

    def aggregate(self, ticks: pl.DataFrame) -> list[Bar]:
        """Aggregate ticks into run bars."""
        if len(ticks) == 0:
            return []

        if "direction" not in ticks.columns:
            logger.warning("No direction column - using tick signs")
            ticks = self._infer_direction(ticks)

        records = ticks.to_dicts()
        n = len(records)

        bars: list[Bar] = []
        current_bar_ticks: list[dict] = []

        # Track runs
        buy_run = 0.0
        sell_run = 0.0
        current_direction = None

        # Expected run lengths
        window = self.config.run_window
        recent_buy_runs: list[float] = []
        recent_sell_runs: list[float] = []

        for i, tick in enumerate(records):
            direction = tick.get("direction", TradeDirection.UNKNOWN.value)
            current_bar_ticks.append(tick)

            # Update runs based on bar type
            if self.config.bar_type == BarType.RUN_TICK:
                delta = 1
            elif self.config.bar_type == BarType.RUN_VOLUME:
                delta = tick["size"]
            else:  # RUN_DOLLAR
                delta = tick["price"] * tick["size"]

            if direction == TradeDirection.BUY.value:
                buy_run += delta
            elif direction == TradeDirection.SELL.value:
                sell_run += delta

            # Calculate expected run
            expected_buy = np.mean(recent_buy_runs) if recent_buy_runs else buy_run
            expected_sell = np.mean(recent_sell_runs) if recent_sell_runs else sell_run

            # Check threshold
            sensitivity = self.config.imbalance_sensitivity
            max_run = max(buy_run, sell_run)
            expected_max = max(expected_buy, expected_sell) * sensitivity

            if max_run >= expected_max and len(current_bar_ticks) >= 10:
                bar = self._create_bar(current_bar_ticks)
                bars.append(bar)

                # Track for expected
                recent_buy_runs.append(buy_run)
                recent_sell_runs.append(sell_run)

                if len(recent_buy_runs) > window:
                    recent_buy_runs.pop(0)
                if len(recent_sell_runs) > window:
                    recent_sell_runs.pop(0)

                # Reset
                current_bar_ticks = []
                buy_run = 0.0
                sell_run = 0.0

        # Handle remaining
        if current_bar_ticks and len(current_bar_ticks) >= 10:
            bars.append(self._create_bar(current_bar_ticks))

        logger.info(f"Created {len(bars)} run bars from {len(ticks)} ticks")

        return bars

    def _infer_direction(self, ticks: pl.DataFrame) -> pl.DataFrame:
        """Infer direction from price changes."""
        prices = ticks["price"].to_numpy()
        n = len(prices)

        directions = [TradeDirection.UNKNOWN.value] * n

        for i in range(1, n):
            if prices[i] > prices[i-1]:
                directions[i] = TradeDirection.BUY.value
            elif prices[i] < prices[i-1]:
                directions[i] = TradeDirection.SELL.value
            else:
                directions[i] = directions[i-1]

        return ticks.with_columns(pl.Series("direction", directions))


# =============================================================================
# BAR FACTORY
# =============================================================================

def create_bar_aggregator(
    bar_type: BarType | str,
    config: BarConfig | None = None,
) -> BarAggregator:
    """
    Factory function to create appropriate bar aggregator.

    Args:
        bar_type: Type of bars to create
        config: Bar configuration

    Returns:
        Appropriate BarAggregator instance

    Example:
        aggregator = create_bar_aggregator("dollar", BarConfig(dollar_threshold=10e6))
        bars = aggregator.aggregate(tick_data)
    """
    if isinstance(bar_type, str):
        bar_type = BarType(bar_type.lower())

    config = config or BarConfig(bar_type=bar_type)

    if bar_type == BarType.DOLLAR:
        return DollarBarAggregator(config)
    elif bar_type == BarType.VOLUME:
        return VolumeBarAggregator(config)
    elif bar_type == BarType.TICK:
        return TickBarAggregator(config)
    elif bar_type in (BarType.IMBALANCE_TICK, BarType.IMBALANCE_VOLUME, BarType.IMBALANCE_DOLLAR):
        return ImbalanceBarAggregator(config)
    elif bar_type in (BarType.RUN_TICK, BarType.RUN_VOLUME, BarType.RUN_DOLLAR):
        return RunBarAggregator(config)
    else:
        raise ValueError(f"Unknown bar type: {bar_type}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_dollar_bars(
    ticks: pl.DataFrame,
    threshold: float = 10_000_000,
) -> pl.DataFrame:
    """
    Create dollar bars from tick data.

    This is the recommended bar type for most ML applications.

    Args:
        ticks: Tick DataFrame with timestamp, price, size columns
        threshold: Dollar volume per bar (default: $10M)

    Returns:
        DataFrame with OHLCV + microstructure columns

    Example:
        # Load tick data
        ticks = tick_loader.load("AAPL", start, end)

        # Create $10M dollar bars
        bars = create_dollar_bars(ticks, threshold=10_000_000)

        # Use for feature generation and ML
        features = feature_pipeline.generate(bars)
    """
    config = BarConfig(bar_type=BarType.DOLLAR, dollar_threshold=threshold)
    aggregator = DollarBarAggregator(config)
    bars = aggregator.aggregate(ticks)
    return aggregator.to_dataframe(bars)


def create_volume_bars(
    ticks: pl.DataFrame,
    threshold: float = 50000,
) -> pl.DataFrame:
    """
    Create volume bars from tick data.

    Args:
        ticks: Tick DataFrame
        threshold: Volume (shares) per bar

    Returns:
        DataFrame with OHLCV columns
    """
    config = BarConfig(bar_type=BarType.VOLUME, volume_threshold=threshold)
    aggregator = VolumeBarAggregator(config)
    bars = aggregator.aggregate(ticks)
    return aggregator.to_dataframe(bars)


def create_imbalance_bars(
    ticks: pl.DataFrame,
    bar_type: BarType = BarType.IMBALANCE_DOLLAR,
    sensitivity: float = 2.0,
) -> pl.DataFrame:
    """
    Create imbalance bars from tick data.

    Args:
        ticks: Tick DataFrame with direction column
        bar_type: Type of imbalance bar (tick, volume, or dollar)
        sensitivity: Threshold multiplier

    Returns:
        DataFrame with OHLCV columns
    """
    config = BarConfig(bar_type=bar_type, imbalance_sensitivity=sensitivity)
    aggregator = ImbalanceBarAggregator(config)
    bars = aggregator.aggregate(ticks)
    return aggregator.to_dataframe(bars)


def estimate_dollar_threshold(
    ohlcv_data: pl.DataFrame,
    target_bars_per_day: int = 50,
) -> float:
    """
    Estimate appropriate dollar bar threshold from historical OHLCV data.

    Args:
        ohlcv_data: Historical OHLCV DataFrame
        target_bars_per_day: Desired number of bars per trading day

    Returns:
        Recommended dollar threshold

    Example:
        # Get historical data
        data = loader.load("AAPL")

        # Estimate threshold for ~50 bars/day
        threshold = estimate_dollar_threshold(data, target_bars_per_day=50)
        # Returns something like $15,000,000 for AAPL
    """
    # Calculate daily dollar volume
    if "dollar_volume" not in ohlcv_data.columns:
        ohlcv_data = ohlcv_data.with_columns(
            (pl.col("close") * pl.col("volume")).alias("dollar_volume")
        )

    # Extract date
    ohlcv_data = ohlcv_data.with_columns(
        pl.col("timestamp").dt.date().alias("date")
    )

    # Average daily dollar volume
    daily_volume = ohlcv_data.group_by("date").agg(
        pl.col("dollar_volume").sum()
    )["dollar_volume"].mean()

    # Threshold = daily volume / target bars
    threshold = daily_volume / target_bars_per_day

    logger.info(f"Estimated dollar threshold: ${threshold/1e6:.2f}M "
               f"(for {target_bars_per_day} bars/day)")

    return threshold


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "BarType",
    # Configuration
    "BarConfig",
    "Bar",
    # Aggregators
    "BarAggregator",
    "DollarBarAggregator",
    "VolumeBarAggregator",
    "TickBarAggregator",
    "ImbalanceBarAggregator",
    "RunBarAggregator",
    # Factory
    "create_bar_aggregator",
    # Convenience functions
    "create_dollar_bars",
    "create_volume_bars",
    "create_imbalance_bars",
    "estimate_dollar_threshold",
]
