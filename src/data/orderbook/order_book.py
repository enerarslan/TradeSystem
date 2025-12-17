"""
Order Book Data Structures.

This module provides core data structures for order book representation:
1. L2 Order Book (aggregated price levels)
2. L3 Order Book (individual orders)

Reference:
    - "Algorithmic and High-Frequency Trading" by Cartea et al. (2015)

Designed for JPMorgan-level institutional requirements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Iterator, List, Optional, Tuple
from collections import OrderedDict
import heapq

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Side(Enum):
    """Order side enumeration."""
    BID = "bid"
    ASK = "ask"

    @property
    def opposite(self) -> "Side":
        """Get opposite side."""
        return Side.ASK if self == Side.BID else Side.BID


@dataclass
class OrderBookLevel:
    """
    Single price level in the order book.

    For L2: Contains aggregated quantity at this price
    For L3: Contains list of individual orders
    """

    price: float
    quantity: float
    side: Side
    order_count: int = 1               # Number of orders at this level
    orders: List[Dict] = field(default_factory=list)  # For L3: individual orders

    def __lt__(self, other: "OrderBookLevel") -> bool:
        """For heap ordering."""
        if self.side == Side.BID:
            return self.price > other.price  # Max heap for bids
        return self.price < other.price  # Min heap for asks


@dataclass
class OrderBookSnapshot:
    """
    Point-in-time snapshot of the order book.

    Contains full state at a specific timestamp.
    """

    timestamp: datetime
    symbol: str
    bids: List[OrderBookLevel]         # Best bid first
    asks: List[OrderBookLevel]         # Best ask first
    last_trade_price: Optional[float] = None
    last_trade_quantity: Optional[float] = None

    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        """Get best bid level."""
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        """Get best ask level."""
        return self.asks[0] if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return None

    @property
    def spread_bps(self) -> Optional[float]:
        """Calculate spread in basis points."""
        if self.spread and self.mid_price:
            return (self.spread / self.mid_price) * 10000
        return None

    def get_depth(self, side: Side, levels: int = 5) -> float:
        """Get total depth for first N levels."""
        book = self.bids if side == Side.BID else self.asks
        return sum(level.quantity for level in book[:levels])

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "best_bid": self.best_bid.price if self.best_bid else None,
            "best_bid_qty": self.best_bid.quantity if self.best_bid else None,
            "best_ask": self.best_ask.price if self.best_ask else None,
            "best_ask_qty": self.best_ask.quantity if self.best_ask else None,
            "mid_price": self.mid_price,
            "spread": self.spread,
            "spread_bps": self.spread_bps,
            "bid_depth_5": self.get_depth(Side.BID, 5),
            "ask_depth_5": self.get_depth(Side.ASK, 5),
        }


class OrderBook:
    """
    Order book with efficient updates and queries.

    Supports both L2 (aggregated) and L3 (order-by-order) modes.

    Example usage:
        # Create order book
        book = OrderBook(symbol="AAPL", levels=10)

        # Process market data
        for update in market_data_feed:
            book.process_update(update)

            # Get current state
            snapshot = book.get_snapshot()
            print(f"Mid: {snapshot.mid_price}, Spread: {snapshot.spread_bps:.1f}bps")

            # Simulate order execution
            fill_price, filled_qty = book.simulate_market_order(
                side=Side.BID,
                quantity=1000,
            )
    """

    def __init__(
        self,
        symbol: str,
        levels: int = 10,
        tick_size: float = 0.01,
    ) -> None:
        """
        Initialize order book.

        Args:
            symbol: Symbol identifier
            levels: Number of levels to track
            tick_size: Minimum price increment
        """
        self.symbol = symbol
        self.levels = levels
        self.tick_size = tick_size

        # Price level storage (price -> quantity)
        self._bids: OrderedDict[float, float] = OrderedDict()
        self._asks: OrderedDict[float, float] = OrderedDict()

        # Order storage for L3 (order_id -> order_info)
        self._orders: Dict[str, Dict] = {}

        # Last update timestamp
        self._timestamp: Optional[datetime] = None

        # Trade info
        self._last_trade_price: Optional[float] = None
        self._last_trade_quantity: Optional[float] = None

    def process_l2_update(
        self,
        timestamp: datetime,
        side: Side,
        price: float,
        quantity: float,
    ) -> None:
        """
        Process L2 (price level) update.

        Args:
            timestamp: Update timestamp
            side: Bid or ask
            price: Price level
            quantity: New quantity at this level (0 to remove)
        """
        self._timestamp = timestamp
        book = self._bids if side == Side.BID else self._asks

        if quantity <= 0:
            # Remove level
            book.pop(price, None)
        else:
            # Update level
            book[price] = quantity

        # Keep sorted
        self._sort_book(side)

    def process_l3_update(
        self,
        timestamp: datetime,
        order_id: str,
        side: Side,
        price: float,
        quantity: float,
        action: str,  # "add", "modify", "delete", "execute"
    ) -> None:
        """
        Process L3 (order-by-order) update.

        Args:
            timestamp: Update timestamp
            order_id: Unique order identifier
            side: Bid or ask
            price: Order price
            quantity: Order quantity
            action: Update type
        """
        self._timestamp = timestamp
        book = self._bids if side == Side.BID else self._asks

        if action == "add":
            # Add new order
            self._orders[order_id] = {
                "side": side,
                "price": price,
                "quantity": quantity,
            }
            # Aggregate into price level
            book[price] = book.get(price, 0) + quantity

        elif action == "modify":
            # Modify existing order
            if order_id in self._orders:
                old_order = self._orders[order_id]
                old_price = old_order["price"]
                old_qty = old_order["quantity"]

                # Remove old quantity
                book[old_price] = book.get(old_price, 0) - old_qty
                if book[old_price] <= 0:
                    book.pop(old_price, None)

                # Add new quantity
                book[price] = book.get(price, 0) + quantity

                # Update order
                self._orders[order_id] = {
                    "side": side,
                    "price": price,
                    "quantity": quantity,
                }

        elif action in ("delete", "execute"):
            # Remove order
            if order_id in self._orders:
                old_order = self._orders.pop(order_id)
                old_price = old_order["price"]
                old_qty = old_order["quantity"]

                # Remove from price level
                book[old_price] = book.get(old_price, 0) - old_qty
                if book[old_price] <= 0:
                    book.pop(old_price, None)

                # Track trade if execution
                if action == "execute":
                    self._last_trade_price = price
                    self._last_trade_quantity = quantity

        self._sort_book(side)

    def get_snapshot(self) -> OrderBookSnapshot:
        """
        Get current order book snapshot.

        Returns:
            OrderBookSnapshot with current state
        """
        # Convert to OrderBookLevel objects
        bid_levels = [
            OrderBookLevel(price=p, quantity=q, side=Side.BID)
            for p, q in list(self._bids.items())[:self.levels]
        ]

        ask_levels = [
            OrderBookLevel(price=p, quantity=q, side=Side.ASK)
            for p, q in list(self._asks.items())[:self.levels]
        ]

        return OrderBookSnapshot(
            timestamp=self._timestamp or datetime.now(),
            symbol=self.symbol,
            bids=bid_levels,
            asks=ask_levels,
            last_trade_price=self._last_trade_price,
            last_trade_quantity=self._last_trade_quantity,
        )

    def simulate_market_order(
        self,
        side: Side,
        quantity: float,
    ) -> Tuple[float, float]:
        """
        Simulate market order execution.

        Args:
            side: Order side (BID to buy, ASK to sell)
            quantity: Order quantity

        Returns:
            Tuple of (average_fill_price, filled_quantity)
        """
        # Buy orders execute against asks, sell against bids
        book = self._asks if side == Side.BID else self._bids

        if not book:
            return 0.0, 0.0

        remaining = quantity
        total_value = 0.0
        filled_qty = 0.0

        for price in list(book.keys()):
            if remaining <= 0:
                break

            available = book[price]
            fill_qty = min(remaining, available)

            total_value += price * fill_qty
            filled_qty += fill_qty
            remaining -= fill_qty

        if filled_qty == 0:
            return 0.0, 0.0

        avg_price = total_value / filled_qty
        return avg_price, filled_qty

    def simulate_limit_order(
        self,
        side: Side,
        price: float,
        quantity: float,
    ) -> Tuple[float, float, float]:
        """
        Simulate limit order execution.

        Args:
            side: Order side
            price: Limit price
            quantity: Order quantity

        Returns:
            Tuple of (avg_fill_price, filled_qty, unfilled_qty)
        """
        book = self._asks if side == Side.BID else self._bids

        if not book:
            return 0.0, 0.0, quantity

        remaining = quantity
        total_value = 0.0
        filled_qty = 0.0

        for level_price in list(book.keys()):
            if remaining <= 0:
                break

            # Check price constraint
            if side == Side.BID and level_price > price:
                break
            if side == Side.ASK and level_price < price:
                break

            available = book[level_price]
            fill_qty = min(remaining, available)

            total_value += level_price * fill_qty
            filled_qty += fill_qty
            remaining -= fill_qty

        avg_price = total_value / filled_qty if filled_qty > 0 else 0.0
        return avg_price, filled_qty, remaining

    def get_vwap_to_quantity(
        self,
        side: Side,
        quantity: float,
    ) -> Optional[float]:
        """
        Calculate VWAP to fill a given quantity.

        Args:
            side: Order side
            quantity: Target quantity

        Returns:
            VWAP or None if insufficient depth
        """
        avg_price, filled = self.simulate_market_order(side, quantity)

        if filled < quantity:
            return None

        return avg_price

    def get_market_impact(
        self,
        side: Side,
        quantity: float,
    ) -> Optional[float]:
        """
        Calculate market impact in basis points.

        Args:
            side: Order side
            quantity: Order quantity

        Returns:
            Market impact in bps or None
        """
        snapshot = self.get_snapshot()
        mid = snapshot.mid_price

        if mid is None:
            return None

        vwap = self.get_vwap_to_quantity(side, quantity)

        if vwap is None:
            return None

        # Impact = (VWAP - Mid) / Mid * 10000
        if side == Side.BID:
            impact = (vwap - mid) / mid * 10000
        else:
            impact = (mid - vwap) / mid * 10000

        return impact

    def _sort_book(self, side: Side) -> None:
        """Sort book by price."""
        book = self._bids if side == Side.BID else self._asks

        if side == Side.BID:
            # Bids: highest price first
            sorted_items = sorted(book.items(), key=lambda x: -x[0])
        else:
            # Asks: lowest price first
            sorted_items = sorted(book.items(), key=lambda x: x[0])

        if side == Side.BID:
            self._bids = OrderedDict(sorted_items)
        else:
            self._asks = OrderedDict(sorted_items)

    def clear(self) -> None:
        """Clear order book."""
        self._bids.clear()
        self._asks.clear()
        self._orders.clear()
        self._timestamp = None
        self._last_trade_price = None
        self._last_trade_quantity = None


class OrderBookSeries:
    """
    Time series of order book snapshots.

    Stores historical order book states for backtesting and analysis.

    Example usage:
        series = OrderBookSeries(symbol="AAPL")

        for snapshot in historical_data:
            series.add_snapshot(snapshot)

        # Get snapshot at specific time
        book = series.get_at_time(datetime(2024, 1, 15, 10, 30))

        # Get features over time
        features = series.extract_features()
    """

    def __init__(
        self,
        symbol: str,
        max_snapshots: int = 10000,
    ) -> None:
        """
        Initialize order book series.

        Args:
            symbol: Symbol identifier
            max_snapshots: Maximum number of snapshots to store
        """
        self.symbol = symbol
        self.max_snapshots = max_snapshots

        self._snapshots: List[OrderBookSnapshot] = []
        self._timestamps: List[datetime] = []

    def add_snapshot(self, snapshot: OrderBookSnapshot) -> None:
        """Add a snapshot to the series."""
        self._snapshots.append(snapshot)
        self._timestamps.append(snapshot.timestamp)

        # Trim if needed
        if len(self._snapshots) > self.max_snapshots:
            self._snapshots = self._snapshots[-self.max_snapshots:]
            self._timestamps = self._timestamps[-self.max_snapshots:]

    def get_at_time(
        self,
        timestamp: datetime,
        method: str = "ffill",
    ) -> Optional[OrderBookSnapshot]:
        """
        Get snapshot at or before a specific time.

        Args:
            timestamp: Target timestamp
            method: "ffill" for forward fill, "exact" for exact match

        Returns:
            OrderBookSnapshot or None
        """
        if not self._timestamps:
            return None

        # Binary search for closest timestamp
        import bisect
        idx = bisect.bisect_right(self._timestamps, timestamp)

        if method == "exact":
            if idx > 0 and self._timestamps[idx - 1] == timestamp:
                return self._snapshots[idx - 1]
            return None

        # Forward fill: use most recent snapshot
        if idx == 0:
            return None

        return self._snapshots[idx - 1]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert series to DataFrame."""
        data = [s.to_dict() for s in self._snapshots]
        return pd.DataFrame(data)

    def __len__(self) -> int:
        return len(self._snapshots)

    def __iter__(self) -> Iterator[OrderBookSnapshot]:
        return iter(self._snapshots)
