"""
Order Book Feature Extraction.

This module extracts microstructure features from order book data:
1. Book imbalance measures
2. Weighted price levels
3. Depth profiles
4. Price impact estimates

Reference:
    - "Market Microstructure in Practice" by Lehalle & Laruelle (2018)
    - "Algorithmic and High-Frequency Trading" by Cartea et al. (2015)

Designed for JPMorgan-level institutional requirements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .order_book import OrderBook, OrderBookSnapshot, Side

logger = logging.getLogger(__name__)


@dataclass
class OrderBookFeatures:
    """Container for extracted order book features."""

    # Price features
    mid_price: float
    microprice: float                  # Volume-weighted mid
    weighted_mid: float                # Multi-level weighted mid

    # Spread features
    spread: float
    spread_bps: float
    relative_spread: float             # Spread / Mid

    # Imbalance features
    book_imbalance: float              # (bid_qty - ask_qty) / (bid + ask)
    top_imbalance: float               # Top-of-book imbalance
    depth_imbalance: float             # Deeper level imbalance

    # Depth features
    bid_depth_1: float                 # Top level
    ask_depth_1: float
    bid_depth_5: float                 # Top 5 levels
    ask_depth_5: float
    total_depth: float                 # All visible depth

    # Pressure features
    bid_pressure: float                # Quantity-weighted bid strength
    ask_pressure: float
    pressure_imbalance: float          # (bid_pressure - ask_pressure) / total

    # Impact estimates
    buy_impact_100: float              # Impact of 100 unit buy
    sell_impact_100: float
    buy_impact_1000: float             # Impact of 1000 unit buy
    sell_impact_1000: float


def calculate_book_imbalance(
    snapshot: OrderBookSnapshot,
    levels: int = 5,
    weighted: bool = True,
) -> float:
    """
    Calculate order book imbalance.

    Imbalance = (Bid_Qty - Ask_Qty) / (Bid_Qty + Ask_Qty)

    Positive values indicate buying pressure.
    Negative values indicate selling pressure.

    Args:
        snapshot: Order book snapshot
        levels: Number of levels to include
        weighted: Use distance-weighted quantities

    Returns:
        Imbalance value in [-1, 1]
    """
    bid_qty = 0.0
    ask_qty = 0.0

    for i, level in enumerate(snapshot.bids[:levels]):
        weight = 1.0 / (i + 1) if weighted else 1.0
        bid_qty += level.quantity * weight

    for i, level in enumerate(snapshot.asks[:levels]):
        weight = 1.0 / (i + 1) if weighted else 1.0
        ask_qty += level.quantity * weight

    total = bid_qty + ask_qty

    if total == 0:
        return 0.0

    return (bid_qty - ask_qty) / total


def calculate_weighted_mid(
    snapshot: OrderBookSnapshot,
    levels: int = 3,
) -> float:
    """
    Calculate volume-weighted mid price.

    Uses multiple levels weighted by inverse distance.

    Args:
        snapshot: Order book snapshot
        levels: Number of levels to include

    Returns:
        Weighted mid price
    """
    if not snapshot.bids or not snapshot.asks:
        return 0.0

    bid_prices = []
    bid_weights = []
    ask_prices = []
    ask_weights = []

    for i, level in enumerate(snapshot.bids[:levels]):
        weight = level.quantity / (i + 1)
        bid_prices.append(level.price)
        bid_weights.append(weight)

    for i, level in enumerate(snapshot.asks[:levels]):
        weight = level.quantity / (i + 1)
        ask_prices.append(level.price)
        ask_weights.append(weight)

    if not bid_weights or not ask_weights:
        return snapshot.mid_price or 0.0

    weighted_bid = np.average(bid_prices, weights=bid_weights)
    weighted_ask = np.average(ask_prices, weights=ask_weights)

    return (weighted_bid + weighted_ask) / 2


def calculate_microprice(snapshot: OrderBookSnapshot) -> float:
    """
    Calculate microprice (volume-weighted mid at top of book).

    Microprice = (Bid * Ask_Size + Ask * Bid_Size) / (Bid_Size + Ask_Size)

    Args:
        snapshot: Order book snapshot

    Returns:
        Microprice
    """
    if not snapshot.best_bid or not snapshot.best_ask:
        return 0.0

    bid = snapshot.best_bid.price
    ask = snapshot.best_ask.price
    bid_size = snapshot.best_bid.quantity
    ask_size = snapshot.best_ask.quantity

    total_size = bid_size + ask_size

    if total_size == 0:
        return (bid + ask) / 2

    return (bid * ask_size + ask * bid_size) / total_size


def calculate_depth_profile(
    snapshot: OrderBookSnapshot,
    levels: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate depth profile at each level.

    Args:
        snapshot: Order book snapshot
        levels: Number of levels

    Returns:
        Tuple of (bid_prices, bid_quantities, ask_prices, ask_quantities)
    """
    bid_prices = np.zeros(levels)
    bid_qtys = np.zeros(levels)
    ask_prices = np.zeros(levels)
    ask_qtys = np.zeros(levels)

    for i, level in enumerate(snapshot.bids[:levels]):
        bid_prices[i] = level.price
        bid_qtys[i] = level.quantity

    for i, level in enumerate(snapshot.asks[:levels]):
        ask_prices[i] = level.price
        ask_qtys[i] = level.quantity

    return bid_prices, bid_qtys, ask_prices, ask_qtys


def calculate_pressure(
    snapshot: OrderBookSnapshot,
    levels: int = 5,
    decay: float = 0.5,
) -> Tuple[float, float]:
    """
    Calculate buying/selling pressure from order book.

    Pressure is quantity-weighted by inverse distance from mid.

    Args:
        snapshot: Order book snapshot
        levels: Number of levels
        decay: Weight decay factor

    Returns:
        Tuple of (bid_pressure, ask_pressure)
    """
    mid = snapshot.mid_price or 0.0

    if mid == 0:
        return 0.0, 0.0

    bid_pressure = 0.0
    ask_pressure = 0.0

    for i, level in enumerate(snapshot.bids[:levels]):
        distance = (mid - level.price) / mid
        weight = np.exp(-decay * i) * (1 - distance)
        bid_pressure += level.quantity * max(0, weight)

    for i, level in enumerate(snapshot.asks[:levels]):
        distance = (level.price - mid) / mid
        weight = np.exp(-decay * i) * (1 - distance)
        ask_pressure += level.quantity * max(0, weight)

    return bid_pressure, ask_pressure


def extract_features(
    snapshot: OrderBookSnapshot,
    levels: int = 5,
    impact_sizes: List[float] = [100, 1000],
) -> OrderBookFeatures:
    """
    Extract comprehensive features from order book snapshot.

    Args:
        snapshot: Order book snapshot
        levels: Number of levels to analyze
        impact_sizes: Order sizes for impact calculation

    Returns:
        OrderBookFeatures dataclass
    """
    # Price features
    mid = snapshot.mid_price or 0.0
    microprice = calculate_microprice(snapshot)
    weighted_mid = calculate_weighted_mid(snapshot, levels=3)

    # Spread features
    spread = snapshot.spread or 0.0
    spread_bps = snapshot.spread_bps or 0.0
    relative_spread = spread / mid if mid > 0 else 0.0

    # Imbalance features
    book_imbalance = calculate_book_imbalance(snapshot, levels=levels)
    top_imbalance = calculate_book_imbalance(snapshot, levels=1, weighted=False)
    depth_imbalance = calculate_book_imbalance(
        snapshot, levels=levels, weighted=False
    )

    # Depth features
    bid_depth_1 = snapshot.bids[0].quantity if snapshot.bids else 0.0
    ask_depth_1 = snapshot.asks[0].quantity if snapshot.asks else 0.0
    bid_depth_5 = snapshot.get_depth(Side.BID, 5)
    ask_depth_5 = snapshot.get_depth(Side.ASK, 5)
    total_depth = bid_depth_5 + ask_depth_5

    # Pressure features
    bid_pressure, ask_pressure = calculate_pressure(snapshot, levels=levels)
    total_pressure = bid_pressure + ask_pressure
    pressure_imbalance = (
        (bid_pressure - ask_pressure) / total_pressure
        if total_pressure > 0
        else 0.0
    )

    # Impact estimates (need OrderBook for simulation)
    buy_impact_100 = 0.0
    sell_impact_100 = 0.0
    buy_impact_1000 = 0.0
    sell_impact_1000 = 0.0

    # Estimate impact from depth
    if bid_depth_1 > 0 and ask_depth_1 > 0 and mid > 0:
        # Simple impact model: impact ~ order_size / depth * spread
        for size in impact_sizes:
            buy_impact = (size / ask_depth_5) * spread_bps if ask_depth_5 > 0 else 0
            sell_impact = (size / bid_depth_5) * spread_bps if bid_depth_5 > 0 else 0

            if size == 100:
                buy_impact_100 = buy_impact
                sell_impact_100 = sell_impact
            elif size == 1000:
                buy_impact_1000 = buy_impact
                sell_impact_1000 = sell_impact

    return OrderBookFeatures(
        mid_price=mid,
        microprice=microprice,
        weighted_mid=weighted_mid,
        spread=spread,
        spread_bps=spread_bps,
        relative_spread=relative_spread,
        book_imbalance=book_imbalance,
        top_imbalance=top_imbalance,
        depth_imbalance=depth_imbalance,
        bid_depth_1=bid_depth_1,
        ask_depth_1=ask_depth_1,
        bid_depth_5=bid_depth_5,
        ask_depth_5=ask_depth_5,
        total_depth=total_depth,
        bid_pressure=bid_pressure,
        ask_pressure=ask_pressure,
        pressure_imbalance=pressure_imbalance,
        buy_impact_100=buy_impact_100,
        sell_impact_100=sell_impact_100,
        buy_impact_1000=buy_impact_1000,
        sell_impact_1000=sell_impact_1000,
    )


def extract_features_series(
    snapshots: List[OrderBookSnapshot],
    levels: int = 5,
) -> pd.DataFrame:
    """
    Extract features from a series of snapshots.

    Args:
        snapshots: List of order book snapshots
        levels: Number of levels to analyze

    Returns:
        DataFrame with features for each timestamp
    """
    records = []

    for snapshot in snapshots:
        features = extract_features(snapshot, levels=levels)

        record = {
            "timestamp": snapshot.timestamp,
            "symbol": snapshot.symbol,
        }
        record.update(features.__dict__)
        records.append(record)

    return pd.DataFrame(records)


def calculate_book_dynamics(
    snapshots: List[OrderBookSnapshot],
    window: int = 10,
) -> pd.DataFrame:
    """
    Calculate dynamic features from order book evolution.

    Args:
        snapshots: List of snapshots
        window: Rolling window size

    Returns:
        DataFrame with dynamic features
    """
    # Extract static features first
    df = extract_features_series(snapshots)

    if len(df) < window:
        return df

    # Add dynamic features
    df["mid_change"] = df["mid_price"].diff()
    df["mid_return"] = df["mid_price"].pct_change()
    df["spread_change"] = df["spread_bps"].diff()

    df["imbalance_change"] = df["book_imbalance"].diff()
    df["imbalance_ma"] = df["book_imbalance"].rolling(window).mean()
    df["imbalance_std"] = df["book_imbalance"].rolling(window).std()

    df["depth_change"] = df["total_depth"].diff()
    df["depth_ma"] = df["total_depth"].rolling(window).mean()

    # Imbalance momentum
    df["imbalance_momentum"] = (
        df["book_imbalance"] - df["imbalance_ma"]
    ) / df["imbalance_std"].replace(0, 1)

    # Pressure dynamics
    df["pressure_change"] = df["pressure_imbalance"].diff()

    return df


class OrderBookFeatureEngine:
    """
    Engine for real-time order book feature extraction.

    Maintains rolling statistics and provides streaming feature updates.

    Example usage:
        engine = OrderBookFeatureEngine(window=20)

        for snapshot in live_data:
            features = engine.update(snapshot)
            signals = my_strategy.process(features)
    """

    def __init__(
        self,
        window: int = 20,
        levels: int = 5,
    ) -> None:
        """
        Initialize feature engine.

        Args:
            window: Rolling window size
            levels: Number of book levels to analyze
        """
        self.window = window
        self.levels = levels

        # Rolling buffers
        self._mid_prices: List[float] = []
        self._spreads: List[float] = []
        self._imbalances: List[float] = []
        self._depths: List[float] = []

    def update(self, snapshot: OrderBookSnapshot) -> Dict:
        """
        Update with new snapshot and return features.

        Args:
            snapshot: New order book snapshot

        Returns:
            Dictionary of features
        """
        # Extract base features
        features = extract_features(snapshot, levels=self.levels)

        # Update buffers
        self._mid_prices.append(features.mid_price)
        self._spreads.append(features.spread_bps)
        self._imbalances.append(features.book_imbalance)
        self._depths.append(features.total_depth)

        # Trim buffers
        if len(self._mid_prices) > self.window:
            self._mid_prices = self._mid_prices[-self.window:]
            self._spreads = self._spreads[-self.window:]
            self._imbalances = self._imbalances[-self.window:]
            self._depths = self._depths[-self.window:]

        # Calculate dynamic features
        result = features.__dict__.copy()

        if len(self._mid_prices) >= 2:
            result["mid_change"] = self._mid_prices[-1] - self._mid_prices[-2]
            result["mid_return"] = (
                self._mid_prices[-1] / self._mid_prices[-2] - 1
                if self._mid_prices[-2] != 0
                else 0.0
            )

        if len(self._imbalances) >= self.window:
            imb_arr = np.array(self._imbalances)
            result["imbalance_ma"] = imb_arr.mean()
            result["imbalance_std"] = imb_arr.std()
            result["imbalance_momentum"] = (
                (self._imbalances[-1] - result["imbalance_ma"]) /
                max(result["imbalance_std"], 0.001)
            )

        if len(self._depths) >= self.window:
            depth_arr = np.array(self._depths)
            result["depth_ma"] = depth_arr.mean()
            result["depth_std"] = depth_arr.std()

        return result

    def reset(self) -> None:
        """Reset rolling buffers."""
        self._mid_prices.clear()
        self._spreads.clear()
        self._imbalances.clear()
        self._depths.clear()
