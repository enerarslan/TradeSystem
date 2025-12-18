"""
Advanced Limit Order Book (LOB) Features.

Institutional-grade microstructure features for high-frequency trading:
- Multi-depth order book imbalance
- Spread analytics
- Trade flow toxicity
- Smart money detection
- Institutional activity indicators

Reference:
    Cartea, Jaimungal, Penalva (2015) - "Algorithmic and High-Frequency Trading"
    Gueant (2016) - "The Financial Mathematics of Market Liquidity"

Designed for JPMorgan-level institutional requirements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LOBSnapshot:
    """
    Limit Order Book snapshot at a single point in time.

    Attributes:
        timestamp: Snapshot timestamp
        bid_prices: Bid prices at each level (index 0 = best bid)
        bid_sizes: Bid sizes at each level
        ask_prices: Ask prices at each level (index 0 = best ask)
        ask_sizes: Ask sizes at each level
    """
    timestamp: pd.Timestamp
    bid_prices: np.ndarray
    bid_sizes: np.ndarray
    ask_prices: np.ndarray
    ask_sizes: np.ndarray

    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        if len(self.bid_prices) == 0 or len(self.ask_prices) == 0:
            return np.nan
        return (self.bid_prices[0] + self.ask_prices[0]) / 2

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        if len(self.bid_prices) == 0 or len(self.ask_prices) == 0:
            return np.nan
        return self.ask_prices[0] - self.bid_prices[0]

    @property
    def spread_bps(self) -> float:
        """Calculate spread in basis points."""
        mid = self.mid_price
        if np.isnan(mid) or mid == 0:
            return np.nan
        return (self.spread / mid) * 10000


@dataclass
class TradeRecord:
    """
    Individual trade record for flow analysis.

    Attributes:
        timestamp: Trade timestamp
        price: Execution price
        size: Trade size
        side: Trade side (1 = buy, -1 = sell, 0 = unknown)
        is_block: Whether trade is a block trade
    """
    timestamp: pd.Timestamp
    price: float
    size: float
    side: int = 0
    is_block: bool = False


@dataclass
class LOBFeatures:
    """Container for all LOB-derived features."""

    # Multi-depth imbalance
    obi_level_1: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    obi_level_3: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    obi_level_5: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    obi_level_10: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # Weighted imbalance
    weighted_obi: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    depth_weighted_obi: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # Spread features
    spread_bps: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    spread_percentile: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    spread_momentum: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    spread_zscore: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # Flow toxicity
    toxicity_index: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    adverse_selection: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # Smart money / institutional
    smart_money_index: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    institutional_activity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    block_trade_ratio: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


# =============================================================================
# MULTI-DEPTH ORDER BOOK IMBALANCE
# =============================================================================

class MultiDepthImbalance:
    """
    Calculate order book imbalance at multiple depth levels.

    OBI measures the relative pressure between bid and ask sides:
    - OBI > 0: More bid volume (buying pressure)
    - OBI < 0: More ask volume (selling pressure)

    Multi-depth analysis reveals:
    - Level 1: Immediate execution pressure
    - Level 3: Near-term order flow
    - Level 5-10: Broader liquidity structure

    Example:
        mdi = MultiDepthImbalance()
        features = mdi.calculate(lob_data)
    """

    def __init__(
        self,
        levels: List[int] = [1, 3, 5, 10],
        weighting: str = "linear",
        normalize: bool = True,
    ) -> None:
        """
        Initialize Multi-Depth Imbalance calculator.

        Args:
            levels: Depth levels to calculate OBI for
            weighting: Weight scheme for aggregated OBI ("linear", "exp", "inv_dist")
            normalize: Whether to normalize OBI to [-1, 1]
        """
        self.levels = levels
        self.weighting = weighting
        self.normalize = normalize

    def calculate_at_level(
        self,
        bid_sizes: np.ndarray,
        ask_sizes: np.ndarray,
        level: int,
    ) -> float:
        """
        Calculate OBI at a specific depth level.

        Args:
            bid_sizes: Array of bid sizes (index 0 = best bid)
            ask_sizes: Array of ask sizes (index 0 = best ask)
            level: Depth level (1, 3, 5, etc.)

        Returns:
            Order book imbalance at specified level
        """
        # Sum volume up to the specified level
        max_bid_level = min(level, len(bid_sizes))
        max_ask_level = min(level, len(ask_sizes))

        bid_vol = np.sum(bid_sizes[:max_bid_level]) if max_bid_level > 0 else 0
        ask_vol = np.sum(ask_sizes[:max_ask_level]) if max_ask_level > 0 else 0

        total = bid_vol + ask_vol

        if total == 0:
            return 0.0

        obi = (bid_vol - ask_vol) / total
        return obi

    def calculate_weighted(
        self,
        bid_prices: np.ndarray,
        bid_sizes: np.ndarray,
        ask_prices: np.ndarray,
        ask_sizes: np.ndarray,
    ) -> float:
        """
        Calculate depth-weighted OBI across all visible levels.

        Weights are assigned based on distance from mid-price.
        Closer levels receive higher weights.

        Args:
            bid_prices: Bid prices at each level
            bid_sizes: Bid sizes at each level
            ask_prices: Ask prices at each level
            ask_sizes: Ask sizes at each level

        Returns:
            Weighted order book imbalance
        """
        if len(bid_prices) == 0 or len(ask_prices) == 0:
            return 0.0

        mid_price = (bid_prices[0] + ask_prices[0]) / 2

        if mid_price == 0:
            return 0.0

        # Calculate weights based on distance from mid
        bid_distances = np.abs(bid_prices - mid_price)
        ask_distances = np.abs(ask_prices - mid_price)

        if self.weighting == "linear":
            # Linear decay
            max_dist = max(bid_distances.max() if len(bid_distances) > 0 else 1,
                          ask_distances.max() if len(ask_distances) > 0 else 1)
            if max_dist == 0:
                max_dist = 1
            bid_weights = 1 - (bid_distances / max_dist)
            ask_weights = 1 - (ask_distances / max_dist)
        elif self.weighting == "exp":
            # Exponential decay
            decay = 0.5
            bid_weights = np.exp(-decay * bid_distances / mid_price * 100)
            ask_weights = np.exp(-decay * ask_distances / mid_price * 100)
        else:  # inv_dist
            # Inverse distance
            bid_weights = 1 / (1 + bid_distances / mid_price * 100)
            ask_weights = 1 / (1 + ask_distances / mid_price * 100)

        weighted_bid = np.sum(bid_sizes * bid_weights)
        weighted_ask = np.sum(ask_sizes * ask_weights)

        total = weighted_bid + weighted_ask

        if total == 0:
            return 0.0

        return (weighted_bid - weighted_ask) / total

    def calculate_from_snapshots(
        self,
        snapshots: List[LOBSnapshot],
    ) -> Dict[str, pd.Series]:
        """
        Calculate multi-depth OBI from a series of LOB snapshots.

        Args:
            snapshots: List of LOBSnapshot objects

        Returns:
            Dictionary with OBI series for each level
        """
        results = {f"obi_level_{level}": [] for level in self.levels}
        results["weighted_obi"] = []
        timestamps = []

        for snapshot in snapshots:
            timestamps.append(snapshot.timestamp)

            # Calculate OBI at each level
            for level in self.levels:
                obi = self.calculate_at_level(
                    snapshot.bid_sizes, snapshot.ask_sizes, level
                )
                results[f"obi_level_{level}"].append(obi)

            # Calculate weighted OBI
            weighted_obi = self.calculate_weighted(
                snapshot.bid_prices, snapshot.bid_sizes,
                snapshot.ask_prices, snapshot.ask_sizes
            )
            results["weighted_obi"].append(weighted_obi)

        # Convert to pandas Series
        index = pd.DatetimeIndex(timestamps)
        for key in results:
            results[key] = pd.Series(results[key], index=index, name=key)

        return results

    def calculate_from_dataframe(
        self,
        df: pd.DataFrame,
        bid_price_cols: List[str],
        bid_size_cols: List[str],
        ask_price_cols: List[str],
        ask_size_cols: List[str],
    ) -> Dict[str, pd.Series]:
        """
        Calculate multi-depth OBI from DataFrame with LOB columns.

        Args:
            df: DataFrame with LOB data
            bid_price_cols: Column names for bid prices (best to worst)
            bid_size_cols: Column names for bid sizes
            ask_price_cols: Column names for ask prices (best to worst)
            ask_size_cols: Column names for ask sizes

        Returns:
            Dictionary with OBI series for each level
        """
        results = {f"obi_level_{level}": [] for level in self.levels}
        results["weighted_obi"] = []

        for idx in range(len(df)):
            row = df.iloc[idx]

            bid_prices = np.array([row[col] for col in bid_price_cols])
            bid_sizes = np.array([row[col] for col in bid_size_cols])
            ask_prices = np.array([row[col] for col in ask_price_cols])
            ask_sizes = np.array([row[col] for col in ask_size_cols])

            # Remove NaN values
            bid_mask = ~np.isnan(bid_prices) & ~np.isnan(bid_sizes)
            ask_mask = ~np.isnan(ask_prices) & ~np.isnan(ask_sizes)

            bid_prices = bid_prices[bid_mask]
            bid_sizes = bid_sizes[bid_mask]
            ask_prices = ask_prices[ask_mask]
            ask_sizes = ask_sizes[ask_mask]

            # Calculate OBI at each level
            for level in self.levels:
                obi = self.calculate_at_level(bid_sizes, ask_sizes, level)
                results[f"obi_level_{level}"].append(obi)

            # Calculate weighted OBI
            weighted_obi = self.calculate_weighted(
                bid_prices, bid_sizes, ask_prices, ask_sizes
            )
            results["weighted_obi"].append(weighted_obi)

        # Convert to pandas Series
        for key in results:
            results[key] = pd.Series(results[key], index=df.index, name=key)

        return results


# =============================================================================
# SPREAD ANALYTICS
# =============================================================================

class SpreadAnalytics:
    """
    Advanced bid-ask spread analysis for liquidity assessment.

    Features:
    - Spread percentile (vs historical distribution)
    - Spread momentum (rate of change)
    - Spread z-score (standardized spread)
    - Effective vs quoted spread

    High spreads indicate:
    - Low liquidity
    - High uncertainty
    - Potential adverse selection

    Example:
        analyzer = SpreadAnalytics(lookback=100)
        features = analyzer.calculate(bid, ask)
    """

    def __init__(
        self,
        lookback: int = 100,
        momentum_window: int = 10,
    ) -> None:
        """
        Initialize Spread Analytics.

        Args:
            lookback: Window for percentile/z-score calculation
            momentum_window: Window for momentum calculation
        """
        self.lookback = lookback
        self.momentum_window = momentum_window

    def calculate(
        self,
        bid: pd.Series,
        ask: pd.Series,
        price: Optional[pd.Series] = None,
    ) -> Dict[str, pd.Series]:
        """
        Calculate spread analytics.

        Args:
            bid: Best bid prices
            ask: Best ask prices
            price: Mid or close prices (for basis point calculation)

        Returns:
            Dictionary with spread features
        """
        # Calculate raw spread
        spread = ask - bid

        # Mid price for basis point conversion
        if price is None:
            price = (bid + ask) / 2

        # Spread in basis points
        spread_bps = (spread / price) * 10000

        # Rolling percentile
        spread_percentile = spread_bps.rolling(self.lookback).apply(
            lambda x: stats.percentileofscore(x[:-1], x.iloc[-1]) / 100
            if len(x) > 1 else 0.5,
            raw=False
        )

        # Spread momentum (rate of change)
        spread_momentum = spread_bps.diff(self.momentum_window) / self.momentum_window

        # Z-score
        rolling_mean = spread_bps.rolling(self.lookback).mean()
        rolling_std = spread_bps.rolling(self.lookback).std()
        spread_zscore = (spread_bps - rolling_mean) / rolling_std.replace(0, np.nan)

        return {
            "spread_bps": spread_bps,
            "spread_percentile": spread_percentile,
            "spread_momentum": spread_momentum,
            "spread_zscore": spread_zscore,
        }

    def calculate_effective_spread(
        self,
        trade_price: pd.Series,
        mid_price: pd.Series,
        trade_side: pd.Series,
    ) -> pd.Series:
        """
        Calculate effective spread from trade data.

        Effective spread = 2 * |trade_price - mid_price| * side

        Args:
            trade_price: Trade execution prices
            mid_price: Mid prices at trade time
            trade_side: Trade direction (1 = buy, -1 = sell)

        Returns:
            Effective spread series
        """
        return 2 * np.abs(trade_price - mid_price) * trade_side


# =============================================================================
# TRADE FLOW TOXICITY
# =============================================================================

class TradeFlowToxicity:
    """
    Advanced trade flow toxicity metrics beyond VPIN.

    Measures adverse selection and informed trading:
    - Order flow toxicity index
    - Adverse selection component
    - Trade informativeness

    Reference:
        Easley, Kiefer, O'Hara (1997) - "Liquidity, Information, and
        Infrequently Traded Stocks"

    Example:
        toxicity = TradeFlowToxicity()
        features = toxicity.calculate(trades_df)
    """

    def __init__(
        self,
        window: int = 50,
        volume_bucket_pct: float = 0.01,
    ) -> None:
        """
        Initialize Trade Flow Toxicity calculator.

        Args:
            window: Rolling window for calculations
            volume_bucket_pct: Percentage of daily volume per bucket
        """
        self.window = window
        self.volume_bucket_pct = volume_bucket_pct

    def calculate_toxicity_index(
        self,
        price: pd.Series,
        volume: pd.Series,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Calculate trade flow toxicity index.

        Combines multiple toxicity signals:
        1. Volume imbalance
        2. Price impact
        3. Return volatility clustering

        Args:
            price: Trade/close prices
            volume: Volume
            high: High prices
            low: Low prices

        Returns:
            Toxicity index (0-1, higher = more toxic)
        """
        # Volume imbalance component
        if high is not None and low is not None:
            bar_range = high - low
            normalized_pos = (price - low) / bar_range.replace(0, np.nan)
            buy_vol = volume * normalized_pos.fillna(0.5)
            sell_vol = volume * (1 - normalized_pos.fillna(0.5))
        else:
            price_change = price.diff()
            buy_mask = price_change > 0
            buy_vol = volume * buy_mask.astype(float)
            sell_vol = volume * (~buy_mask).astype(float)

        # Rolling imbalance
        rolling_buy = buy_vol.rolling(self.window).sum()
        rolling_sell = sell_vol.rolling(self.window).sum()
        rolling_total = rolling_buy + rolling_sell

        imbalance = np.abs(rolling_buy - rolling_sell) / rolling_total.replace(0, np.nan)

        # Price impact component
        returns = price.pct_change()
        volume_normalized = volume / volume.rolling(self.window).mean()

        # Amihud-style illiquidity
        illiquidity = np.abs(returns) / volume_normalized.replace(0, np.nan)
        illiquidity_percentile = illiquidity.rolling(self.window).apply(
            lambda x: stats.percentileofscore(x[:-1], x.iloc[-1]) / 100
            if len(x) > 1 else 0.5,
            raw=False
        )

        # Volatility clustering
        vol_clustering = returns.abs().rolling(self.window).std()
        vol_percentile = vol_clustering.rolling(self.window).apply(
            lambda x: stats.percentileofscore(x[:-1], x.iloc[-1]) / 100
            if len(x) > 1 else 0.5,
            raw=False
        )

        # Combine into toxicity index
        toxicity = (imbalance * 0.4 + illiquidity_percentile * 0.3 + vol_percentile * 0.3)

        return toxicity.fillna(0.5)

    def calculate_adverse_selection(
        self,
        trade_price: pd.Series,
        mid_price_before: pd.Series,
        mid_price_after: pd.Series,
        trade_side: pd.Series,
    ) -> pd.Series:
        """
        Calculate adverse selection component of trades.

        Adverse selection = (mid_after - mid_before) * trade_side

        Positive values indicate informed trading (price moves in trade direction).

        Args:
            trade_price: Trade execution prices
            mid_price_before: Mid price before trade
            mid_price_after: Mid price after trade
            trade_side: Trade direction (1 = buy, -1 = sell)

        Returns:
            Adverse selection series
        """
        price_impact = mid_price_after - mid_price_before
        adverse_selection = price_impact * trade_side

        return adverse_selection


# =============================================================================
# SMART MONEY INDICATOR
# =============================================================================

class SmartMoneyIndicator:
    """
    Detect institutional/smart money activity in trade flow.

    Signals:
    - Large trades at bid vs ask
    - Block trade detection
    - Institutional footprint

    Large trades that move price are more likely institutional.
    Large trades that don't move price are likely dark pool/VWAP.

    Example:
        smi = SmartMoneyIndicator(block_threshold=10000)
        smart_flow = smi.calculate(trades_df)
    """

    def __init__(
        self,
        block_threshold: Optional[float] = None,
        block_percentile: float = 0.95,
        lookback: int = 100,
    ) -> None:
        """
        Initialize Smart Money Indicator.

        Args:
            block_threshold: Fixed threshold for block trades
            block_percentile: Percentile-based threshold if no fixed value
            lookback: Rolling window for calculations
        """
        self.block_threshold = block_threshold
        self.block_percentile = block_percentile
        self.lookback = lookback

    def calculate(
        self,
        price: pd.Series,
        volume: pd.Series,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
    ) -> Dict[str, pd.Series]:
        """
        Calculate smart money indicators.

        Args:
            price: Trade/close prices
            volume: Volume
            high: High prices
            low: Low prices

        Returns:
            Dictionary with smart money features
        """
        # Determine block threshold
        if self.block_threshold is None:
            rolling_threshold = volume.rolling(self.lookback).quantile(
                self.block_percentile
            )
        else:
            rolling_threshold = pd.Series(
                self.block_threshold, index=volume.index
            )

        # Identify block trades
        is_block = volume > rolling_threshold

        # Classify trade direction
        if high is not None and low is not None:
            bar_range = high - low
            normalized_pos = (price - low) / bar_range.replace(0, np.nan)
            trade_side = (normalized_pos.fillna(0.5) - 0.5) * 2  # Scale to [-1, 1]
        else:
            price_change = price.diff()
            trade_side = np.sign(price_change).replace(0, np.nan).ffill().fillna(0)

        # Smart money index: large trades weighted by direction
        block_volume = volume * is_block
        smart_buy = block_volume * (trade_side > 0).astype(float)
        smart_sell = block_volume * (trade_side < 0).astype(float)

        rolling_smart_buy = smart_buy.rolling(self.lookback).sum()
        rolling_smart_sell = smart_sell.rolling(self.lookback).sum()
        total_smart = rolling_smart_buy + rolling_smart_sell

        smart_money_index = (
            (rolling_smart_buy - rolling_smart_sell) / total_smart.replace(0, np.nan)
        ).fillna(0)

        # Block trade ratio
        block_trade_ratio = is_block.rolling(self.lookback).mean()

        # Institutional activity: combines block ratio and price impact
        returns = price.pct_change().abs()
        block_returns = returns * is_block
        non_block_returns = returns * (~is_block)

        avg_block_impact = block_returns.rolling(self.lookback).mean()
        avg_non_block_impact = non_block_returns.rolling(self.lookback).mean()

        impact_ratio = avg_block_impact / avg_non_block_impact.replace(0, np.nan)

        # High impact ratio + high block ratio = institutional activity
        institutional_activity = (
            block_trade_ratio * 0.5 +
            impact_ratio.clip(0, 2) / 2 * 0.5
        ).fillna(0)

        return {
            "smart_money_index": smart_money_index,
            "block_trade_ratio": block_trade_ratio,
            "institutional_activity": institutional_activity,
            "is_block_trade": is_block,
        }


# =============================================================================
# INTEGRATED LOB FEATURE CALCULATOR
# =============================================================================

class AdvancedLOBFeatures:
    """
    Integrated calculator for all advanced LOB features.

    Combines:
    - Multi-depth order book imbalance
    - Spread analytics
    - Trade flow toxicity
    - Smart money indicators

    Example:
        lob_features = AdvancedLOBFeatures()

        # From OHLCV data (simulated LOB)
        features = lob_features.calculate_from_ohlcv(df)

        # From actual LOB data
        features = lob_features.calculate_from_lob(lob_df, column_map)
    """

    def __init__(
        self,
        imbalance_levels: List[int] = [1, 3, 5, 10],
        spread_lookback: int = 100,
        toxicity_window: int = 50,
        block_percentile: float = 0.95,
    ) -> None:
        """
        Initialize Advanced LOB Features calculator.

        Args:
            imbalance_levels: Depth levels for OBI calculation
            spread_lookback: Lookback for spread analytics
            toxicity_window: Window for toxicity calculation
            block_percentile: Percentile for block trade detection
        """
        self.multi_depth = MultiDepthImbalance(levels=imbalance_levels)
        self.spread_analyzer = SpreadAnalytics(lookback=spread_lookback)
        self.toxicity = TradeFlowToxicity(window=toxicity_window)
        self.smart_money = SmartMoneyIndicator(
            block_percentile=block_percentile,
            lookback=toxicity_window
        )

    def calculate_from_ohlcv(
        self,
        df: pd.DataFrame,
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
    ) -> pd.DataFrame:
        """
        Calculate LOB features from OHLCV data.

        Uses price range to simulate bid/ask when actual LOB not available.

        Args:
            df: DataFrame with OHLCV data
            open_col: Open price column
            high_col: High price column
            low_col: Low price column
            close_col: Close price column
            volume_col: Volume column

        Returns:
            DataFrame with LOB features
        """
        results = pd.DataFrame(index=df.index)

        # Simulate bid/ask from high/low
        mid_price = (df[high_col] + df[low_col]) / 2
        half_spread = (df[high_col] - df[low_col]) / 2

        simulated_bid = mid_price - half_spread * 0.5
        simulated_ask = mid_price + half_spread * 0.5

        # Spread analytics
        spread_features = self.spread_analyzer.calculate(
            simulated_bid, simulated_ask, df[close_col]
        )
        for key, series in spread_features.items():
            results[key] = series

        # Toxicity
        results["toxicity_index"] = self.toxicity.calculate_toxicity_index(
            df[close_col], df[volume_col], df[high_col], df[low_col]
        )

        # Smart money
        smart_features = self.smart_money.calculate(
            df[close_col], df[volume_col], df[high_col], df[low_col]
        )
        for key, series in smart_features.items():
            if key != "is_block_trade":  # Exclude boolean column
                results[key] = series

        # Simulated OBI (using volume and price position)
        bar_range = df[high_col] - df[low_col]
        price_position = (df[close_col] - df[low_col]) / bar_range.replace(0, np.nan)

        # Single-level OBI from price position
        results["obi_simulated"] = (price_position.fillna(0.5) - 0.5) * 2

        logger.info(f"Generated {len(results.columns)} LOB features from OHLCV data")

        return results

    def calculate_from_lob(
        self,
        df: pd.DataFrame,
        bid_price_cols: List[str],
        bid_size_cols: List[str],
        ask_price_cols: List[str],
        ask_size_cols: List[str],
        trade_price_col: Optional[str] = None,
        volume_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Calculate LOB features from actual LOB data.

        Args:
            df: DataFrame with LOB data
            bid_price_cols: Column names for bid prices (best to worst)
            bid_size_cols: Column names for bid sizes
            ask_price_cols: Column names for ask prices
            ask_size_cols: Column names for ask sizes
            trade_price_col: Column for trade/close price
            volume_col: Column for volume

        Returns:
            DataFrame with LOB features
        """
        results = pd.DataFrame(index=df.index)

        # Multi-depth OBI
        obi_features = self.multi_depth.calculate_from_dataframe(
            df, bid_price_cols, bid_size_cols, ask_price_cols, ask_size_cols
        )
        for key, series in obi_features.items():
            results[key] = series

        # Spread analytics (from best bid/ask)
        bid = df[bid_price_cols[0]]
        ask = df[ask_price_cols[0]]

        spread_features = self.spread_analyzer.calculate(bid, ask)
        for key, series in spread_features.items():
            results[key] = series

        # If trade data available, calculate toxicity and smart money
        if trade_price_col is not None and volume_col is not None:
            price = df[trade_price_col]
            volume = df[volume_col]

            results["toxicity_index"] = self.toxicity.calculate_toxicity_index(
                price, volume
            )

            smart_features = self.smart_money.calculate(price, volume)
            for key, series in smart_features.items():
                if key != "is_block_trade":
                    results[key] = series

        logger.info(f"Generated {len(results.columns)} LOB features from order book data")

        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_multi_depth_obi(
    bid_sizes: Union[pd.DataFrame, np.ndarray],
    ask_sizes: Union[pd.DataFrame, np.ndarray],
    levels: List[int] = [1, 3, 5, 10],
) -> Dict[str, pd.Series]:
    """
    Convenience function for multi-depth OBI calculation.

    Args:
        bid_sizes: Bid sizes at each level
        ask_sizes: Ask sizes at each level
        levels: Depth levels to calculate

    Returns:
        Dictionary with OBI at each level
    """
    calculator = MultiDepthImbalance(levels=levels)

    results = {}
    for level in levels:
        if isinstance(bid_sizes, pd.DataFrame):
            obi = []
            for i in range(len(bid_sizes)):
                bid = bid_sizes.iloc[i].values
                ask = ask_sizes.iloc[i].values
                obi.append(calculator.calculate_at_level(bid, ask, level))
            results[f"obi_level_{level}"] = pd.Series(obi, index=bid_sizes.index)
        else:
            obi = calculator.calculate_at_level(bid_sizes, ask_sizes, level)
            results[f"obi_level_{level}"] = obi

    return results


def calculate_spread_percentile(
    bid: pd.Series,
    ask: pd.Series,
    lookback: int = 100,
) -> pd.Series:
    """
    Calculate spread percentile relative to recent history.

    Args:
        bid: Best bid prices
        ask: Best ask prices
        lookback: Rolling window for percentile

    Returns:
        Spread percentile (0-1)
    """
    analyzer = SpreadAnalytics(lookback=lookback)
    features = analyzer.calculate(bid, ask)
    return features["spread_percentile"]


def calculate_toxicity(
    price: pd.Series,
    volume: pd.Series,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    window: int = 50,
) -> pd.Series:
    """
    Calculate trade flow toxicity index.

    Args:
        price: Trade/close prices
        volume: Volume
        high: High prices
        low: Low prices
        window: Rolling window

    Returns:
        Toxicity index (0-1)
    """
    calculator = TradeFlowToxicity(window=window)
    return calculator.calculate_toxicity_index(price, volume, high, low)


def calculate_smart_money(
    price: pd.Series,
    volume: pd.Series,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    block_percentile: float = 0.95,
) -> pd.Series:
    """
    Calculate smart money index.

    Args:
        price: Trade/close prices
        volume: Volume
        high: High prices
        low: Low prices
        block_percentile: Percentile for block detection

    Returns:
        Smart money index (-1 to 1)
    """
    calculator = SmartMoneyIndicator(block_percentile=block_percentile)
    features = calculator.calculate(price, volume, high, low)
    return features["smart_money_index"]


def calculate_institutional_activity(
    price: pd.Series,
    volume: pd.Series,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Calculate institutional activity index.

    Args:
        price: Trade/close prices
        volume: Volume
        high: High prices
        low: Low prices

    Returns:
        Institutional activity index (0-1)
    """
    calculator = SmartMoneyIndicator()
    features = calculator.calculate(price, volume, high, low)
    return features["institutional_activity"]
