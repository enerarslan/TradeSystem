"""
Order Book Builder for reconstructing order books from market data.

This module provides utilities to build order books from:
1. L2 market data feeds (aggregated depth updates)
2. L3 market data feeds (order-by-order updates)
3. Trade and quote (TAQ) data

Reference:
    - "Market Microstructure in Practice" by Lehalle & Laruelle (2018)

Designed for JPMorgan-level institutional requirements.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .order_book import OrderBook, OrderBookSnapshot, Side

logger = logging.getLogger(__name__)


@dataclass
class MarketDataUpdate:
    """Generic market data update message."""

    timestamp: datetime
    symbol: str
    message_type: str              # "quote", "trade", "depth", "order"
    data: Dict[str, Any]           # Message-specific data


class OrderBookBuilder(ABC):
    """
    Abstract base class for order book builders.

    Subclasses implement specific protocols (L2, L3, etc.)
    """

    def __init__(
        self,
        symbol: str,
        levels: int = 10,
    ) -> None:
        """
        Initialize builder.

        Args:
            symbol: Symbol to build book for
            levels: Number of levels to track
        """
        self.symbol = symbol
        self.levels = levels
        self.book = OrderBook(symbol=symbol, levels=levels)

        self._update_count = 0
        self._error_count = 0

    @abstractmethod
    def process_message(self, message: MarketDataUpdate) -> Optional[OrderBookSnapshot]:
        """
        Process a market data message.

        Args:
            message: Market data update

        Returns:
            Updated snapshot or None
        """
        pass

    def get_snapshot(self) -> OrderBookSnapshot:
        """Get current order book snapshot."""
        return self.book.get_snapshot()

    def reset(self) -> None:
        """Reset order book."""
        self.book.clear()
        self._update_count = 0

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def error_count(self) -> int:
        return self._error_count


class L2BookBuilder(OrderBookBuilder):
    """
    Builder for L2 (top-of-book with depth) order books.

    L2 data provides aggregated quantities at each price level,
    typically for the top N levels (e.g., 5, 10, 20 levels).

    Example usage:
        builder = L2BookBuilder(symbol="AAPL", levels=10)

        for msg in market_data_feed:
            snapshot = builder.process_message(msg)
            if snapshot:
                print(f"Mid: {snapshot.mid_price}")
    """

    def __init__(
        self,
        symbol: str,
        levels: int = 10,
        validate_crossed: bool = True,
    ) -> None:
        """
        Initialize L2 builder.

        Args:
            symbol: Symbol
            levels: Number of levels
            validate_crossed: Check for crossed book conditions
        """
        super().__init__(symbol, levels)
        self.validate_crossed = validate_crossed

    def process_message(self, message: MarketDataUpdate) -> Optional[OrderBookSnapshot]:
        """Process L2 market data message."""
        if message.symbol != self.symbol:
            return None

        try:
            if message.message_type == "depth":
                return self._process_depth_update(message)
            elif message.message_type == "quote":
                return self._process_quote_update(message)
            else:
                logger.debug(f"Unknown message type: {message.message_type}")
                return None

        except Exception as e:
            self._error_count += 1
            logger.warning(f"Error processing message: {e}")
            return None

    def _process_depth_update(
        self,
        message: MarketDataUpdate,
    ) -> Optional[OrderBookSnapshot]:
        """Process depth update (multiple levels)."""
        data = message.data

        # Process bids
        if "bids" in data:
            for level in data["bids"]:
                self.book.process_l2_update(
                    timestamp=message.timestamp,
                    side=Side.BID,
                    price=float(level["price"]),
                    quantity=float(level["quantity"]),
                )

        # Process asks
        if "asks" in data:
            for level in data["asks"]:
                self.book.process_l2_update(
                    timestamp=message.timestamp,
                    side=Side.ASK,
                    price=float(level["price"]),
                    quantity=float(level["quantity"]),
                )

        self._update_count += 1

        snapshot = self.book.get_snapshot()

        # Validate
        if self.validate_crossed and self._is_crossed(snapshot):
            logger.warning(f"Crossed book detected at {message.timestamp}")
            self._error_count += 1

        return snapshot

    def _process_quote_update(
        self,
        message: MarketDataUpdate,
    ) -> Optional[OrderBookSnapshot]:
        """Process BBO (best bid/offer) quote update."""
        data = message.data

        # Update best bid
        if "bid_price" in data and "bid_size" in data:
            # Clear existing bids first for BBO-only data
            self.book._bids.clear()
            self.book.process_l2_update(
                timestamp=message.timestamp,
                side=Side.BID,
                price=float(data["bid_price"]),
                quantity=float(data["bid_size"]),
            )

        # Update best ask
        if "ask_price" in data and "ask_size" in data:
            self.book._asks.clear()
            self.book.process_l2_update(
                timestamp=message.timestamp,
                side=Side.ASK,
                price=float(data["ask_price"]),
                quantity=float(data["ask_size"]),
            )

        self._update_count += 1
        return self.book.get_snapshot()

    def process_snapshot(
        self,
        timestamp: datetime,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> OrderBookSnapshot:
        """
        Process full book snapshot.

        Args:
            timestamp: Snapshot timestamp
            bids: List of (price, quantity) tuples
            asks: List of (price, quantity) tuples

        Returns:
            OrderBookSnapshot
        """
        # Clear existing book
        self.book.clear()

        # Add bids
        for price, qty in bids:
            self.book.process_l2_update(timestamp, Side.BID, price, qty)

        # Add asks
        for price, qty in asks:
            self.book.process_l2_update(timestamp, Side.ASK, price, qty)

        self._update_count += 1
        return self.book.get_snapshot()

    def _is_crossed(self, snapshot: OrderBookSnapshot) -> bool:
        """Check if book is crossed (bid >= ask)."""
        if snapshot.best_bid and snapshot.best_ask:
            return snapshot.best_bid.price >= snapshot.best_ask.price
        return False


class L3BookBuilder(OrderBookBuilder):
    """
    Builder for L3 (order-by-order) order books.

    L3 data provides individual order information, allowing for
    precise book reconstruction and order queue position tracking.

    Example usage:
        builder = L3BookBuilder(symbol="AAPL")

        for msg in order_feed:
            snapshot = builder.process_message(msg)
    """

    def __init__(
        self,
        symbol: str,
        levels: int = 10,
        track_queue_position: bool = True,
    ) -> None:
        """
        Initialize L3 builder.

        Args:
            symbol: Symbol
            levels: Number of levels to show
            track_queue_position: Track position in order queue
        """
        super().__init__(symbol, levels)
        self.track_queue_position = track_queue_position

        # Queue tracking: price -> list of order_ids (FIFO)
        self._bid_queues: Dict[float, List[str]] = {}
        self._ask_queues: Dict[float, List[str]] = {}

    def process_message(self, message: MarketDataUpdate) -> Optional[OrderBookSnapshot]:
        """Process L3 order message."""
        if message.symbol != self.symbol:
            return None

        try:
            if message.message_type == "order":
                return self._process_order_update(message)
            elif message.message_type == "trade":
                return self._process_trade(message)
            else:
                return None

        except Exception as e:
            self._error_count += 1
            logger.warning(f"Error processing L3 message: {e}")
            return None

    def _process_order_update(
        self,
        message: MarketDataUpdate,
    ) -> Optional[OrderBookSnapshot]:
        """Process order add/modify/cancel."""
        data = message.data

        order_id = data["order_id"]
        side = Side.BID if data["side"].lower() == "bid" else Side.ASK
        price = float(data["price"])
        quantity = float(data["quantity"])
        action = data["action"]  # add, modify, delete

        # Update order book
        self.book.process_l3_update(
            timestamp=message.timestamp,
            order_id=order_id,
            side=side,
            price=price,
            quantity=quantity,
            action=action,
        )

        # Update queue tracking
        if self.track_queue_position:
            self._update_queue(order_id, side, price, action)

        self._update_count += 1
        return self.book.get_snapshot()

    def _process_trade(
        self,
        message: MarketDataUpdate,
    ) -> Optional[OrderBookSnapshot]:
        """Process trade execution."""
        data = message.data

        # Trade typically removes passive order
        if "order_id" in data:
            order_id = data["order_id"]
            side = Side.BID if data.get("side", "").lower() == "bid" else Side.ASK
            price = float(data["price"])
            quantity = float(data["quantity"])

            self.book.process_l3_update(
                timestamp=message.timestamp,
                order_id=order_id,
                side=side,
                price=price,
                quantity=quantity,
                action="execute",
            )

            # Remove from queue
            if self.track_queue_position:
                self._remove_from_queue(order_id, side, price)

        self._update_count += 1
        return self.book.get_snapshot()

    def _update_queue(
        self,
        order_id: str,
        side: Side,
        price: float,
        action: str,
    ) -> None:
        """Update order queue tracking."""
        queues = self._bid_queues if side == Side.BID else self._ask_queues

        if action == "add":
            if price not in queues:
                queues[price] = []
            queues[price].append(order_id)

        elif action == "delete":
            self._remove_from_queue(order_id, side, price)

    def _remove_from_queue(
        self,
        order_id: str,
        side: Side,
        price: float,
    ) -> None:
        """Remove order from queue."""
        queues = self._bid_queues if side == Side.BID else self._ask_queues

        if price in queues and order_id in queues[price]:
            queues[price].remove(order_id)
            if not queues[price]:
                del queues[price]

    def get_queue_position(
        self,
        order_id: str,
        side: Side,
        price: float,
    ) -> Optional[int]:
        """
        Get position in queue for an order.

        Args:
            order_id: Order identifier
            side: Order side
            price: Order price

        Returns:
            Position in queue (0 = first) or None if not found
        """
        queues = self._bid_queues if side == Side.BID else self._ask_queues

        if price not in queues:
            return None

        try:
            return queues[price].index(order_id)
        except ValueError:
            return None

    def get_queue_depth_ahead(
        self,
        order_id: str,
        side: Side,
        price: float,
    ) -> Optional[float]:
        """
        Get total quantity ahead of an order in queue.

        Args:
            order_id: Order identifier
            side: Order side
            price: Order price

        Returns:
            Total quantity ahead or None
        """
        position = self.get_queue_position(order_id, side, price)

        if position is None:
            return None

        queues = self._bid_queues if side == Side.BID else self._ask_queues
        orders_ahead = queues[price][:position]

        # Sum quantities (would need order quantity tracking)
        # For now, return position count
        return float(position)


def build_book_from_taq(
    taq_data: pd.DataFrame,
    symbol: str,
    levels: int = 10,
) -> Iterator[OrderBookSnapshot]:
    """
    Build order book snapshots from TAQ (Trade and Quote) data.

    Args:
        taq_data: DataFrame with columns: timestamp, type, bid, bid_size, ask, ask_size
        symbol: Symbol
        levels: Number of levels

    Yields:
        OrderBookSnapshot for each timestamp
    """
    builder = L2BookBuilder(symbol=symbol, levels=levels)

    for _, row in taq_data.iterrows():
        if row["type"] == "quote":
            message = MarketDataUpdate(
                timestamp=row["timestamp"],
                symbol=symbol,
                message_type="quote",
                data={
                    "bid_price": row["bid"],
                    "bid_size": row["bid_size"],
                    "ask_price": row["ask"],
                    "ask_size": row["ask_size"],
                },
            )

            snapshot = builder.process_message(message)
            if snapshot:
                yield snapshot


def build_book_from_depth_data(
    depth_data: pd.DataFrame,
    symbol: str,
    levels: int = 10,
) -> Iterator[OrderBookSnapshot]:
    """
    Build order book snapshots from depth data.

    Args:
        depth_data: DataFrame with bid/ask columns for each level
        symbol: Symbol
        levels: Number of levels

    Yields:
        OrderBookSnapshot for each row
    """
    builder = L2BookBuilder(symbol=symbol, levels=levels)

    for _, row in depth_data.iterrows():
        # Extract bids (bid_price_1, bid_qty_1, etc.)
        bids = []
        asks = []

        for i in range(1, levels + 1):
            bid_price_col = f"bid_price_{i}"
            bid_qty_col = f"bid_qty_{i}"
            ask_price_col = f"ask_price_{i}"
            ask_qty_col = f"ask_qty_{i}"

            if bid_price_col in row and pd.notna(row[bid_price_col]):
                bids.append((row[bid_price_col], row[bid_qty_col]))

            if ask_price_col in row and pd.notna(row[ask_price_col]):
                asks.append((row[ask_price_col], row[ask_qty_col]))

        snapshot = builder.process_snapshot(
            timestamp=row["timestamp"],
            bids=bids,
            asks=asks,
        )
        yield snapshot
