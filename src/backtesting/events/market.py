"""
Market data events for event-driven backtesting.

This module provides market data event types:
- TickEvent: Individual tick/trade data
- BarEvent: OHLCV bar data
- OrderBookEvent: Level 2 order book snapshots

Designed for institutional requirements:
- Microsecond precision timestamps
- Full L2/L3 order book support
- Exchange-specific metadata
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.backtesting.events.base import Event, EventType, EventPriority


@dataclass(frozen=True)
class OrderBookLevel:
    """
    Single level in the order book.

    Attributes:
        price: Price level
        size: Total size at this level
        num_orders: Number of orders at this level
    """
    price: float
    size: float
    num_orders: int = 1


@dataclass(frozen=True)
class MarketEvent(Event):
    """
    Base class for all market data events.

    Attributes:
        symbol: Trading symbol
        exchange: Exchange identifier
    """

    symbol: str = ""
    exchange: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": str(self.timestamp),
            "created_at": self.created_at.isoformat(),
            "priority": self.priority.value,
            "source": self.source,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketEvent":
        return cls(
            event_type=EventType(data["event_type"]),
            timestamp=pd.Timestamp(data["timestamp"]),
            event_id=data.get("event_id", str(uuid.uuid4())),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            priority=EventPriority(data.get("priority", EventPriority.NORMAL)),
            source=data.get("source", "market"),
            symbol=data.get("symbol", ""),
            exchange=data.get("exchange", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class TickEvent(Event):
    """
    Individual tick/trade event.

    Represents a single trade or quote update with
    full precision for HFT-grade backtesting.

    Attributes:
        symbol: Trading symbol
        price: Trade/quote price
        size: Trade size or quote quantity
        bid: Best bid price (if quote)
        ask: Best ask price (if quote)
        bid_size: Best bid size
        ask_size: Best ask size
        trade_id: Exchange trade ID
        exchange: Exchange identifier
        conditions: Trade condition codes
    """

    symbol: str = ""
    price: float = 0.0
    size: float = 0.0
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    trade_id: Optional[str] = None
    exchange: str = ""
    conditions: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        # Ensure event_type is correct
        object.__setattr__(self, 'event_type', EventType.TICK)

    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price from bid/ask."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    @property
    def spread_bps(self) -> Optional[float]:
        """Calculate spread in basis points."""
        if self.spread is not None and self.mid_price is not None and self.mid_price > 0:
            return (self.spread / self.mid_price) * 10000
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": str(self.timestamp),
            "created_at": self.created_at.isoformat(),
            "priority": self.priority.value,
            "source": self.source,
            "symbol": self.symbol,
            "price": self.price,
            "size": self.size,
            "bid": self.bid,
            "ask": self.ask,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "trade_id": self.trade_id,
            "exchange": self.exchange,
            "conditions": list(self.conditions),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TickEvent":
        return cls(
            event_type=EventType.TICK,
            timestamp=pd.Timestamp(data["timestamp"]),
            event_id=data.get("event_id", str(uuid.uuid4())),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            priority=EventPriority(data.get("priority", EventPriority.NORMAL)),
            source=data.get("source", "market"),
            symbol=data.get("symbol", ""),
            price=data.get("price", 0.0),
            size=data.get("size", 0.0),
            bid=data.get("bid"),
            ask=data.get("ask"),
            bid_size=data.get("bid_size"),
            ask_size=data.get("ask_size"),
            trade_id=data.get("trade_id"),
            exchange=data.get("exchange", ""),
            conditions=tuple(data.get("conditions", [])),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class BarEvent(Event):
    """
    OHLCV bar event.

    Represents an aggregated bar (candle) for any timeframe.

    Attributes:
        symbol: Trading symbol
        open: Opening price
        high: High price
        low: Low price
        close: Closing price
        volume: Total volume
        vwap: Volume-weighted average price
        num_trades: Number of trades in bar
        exchange: Exchange identifier
        timeframe: Bar timeframe (e.g., "1m", "5m", "1h", "1d")
    """

    symbol: str = ""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    vwap: Optional[float] = None
    num_trades: Optional[int] = None
    exchange: str = ""
    timeframe: str = "1d"

    def __post_init__(self):
        object.__setattr__(self, 'event_type', EventType.BAR)

    @property
    def typical_price(self) -> float:
        """Calculate typical price (HLC/3)."""
        return (self.high + self.low + self.close) / 3

    @property
    def range(self) -> float:
        """Calculate bar range (high - low)."""
        return self.high - self.low

    @property
    def body(self) -> float:
        """Calculate bar body (close - open)."""
        return self.close - self.open

    @property
    def is_bullish(self) -> bool:
        """Check if bar is bullish."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """Check if bar is bearish."""
        return self.close < self.open

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": str(self.timestamp),
            "created_at": self.created_at.isoformat(),
            "priority": self.priority.value,
            "source": self.source,
            "symbol": self.symbol,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
            "num_trades": self.num_trades,
            "exchange": self.exchange,
            "timeframe": self.timeframe,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BarEvent":
        return cls(
            event_type=EventType.BAR,
            timestamp=pd.Timestamp(data["timestamp"]),
            event_id=data.get("event_id", str(uuid.uuid4())),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            priority=EventPriority(data.get("priority", EventPriority.NORMAL)),
            source=data.get("source", "market"),
            symbol=data.get("symbol", ""),
            open=data.get("open", 0.0),
            high=data.get("high", 0.0),
            low=data.get("low", 0.0),
            close=data.get("close", 0.0),
            volume=data.get("volume", 0.0),
            vwap=data.get("vwap"),
            num_trades=data.get("num_trades"),
            exchange=data.get("exchange", ""),
            timeframe=data.get("timeframe", "1d"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_series(
        cls,
        symbol: str,
        series: pd.Series,
        timeframe: str = "1d",
        exchange: str = "",
    ) -> "BarEvent":
        """
        Create BarEvent from a pandas Series.

        Args:
            symbol: Trading symbol
            series: Series with OHLCV data (index is timestamp)
            timeframe: Bar timeframe
            exchange: Exchange identifier

        Returns:
            BarEvent instance
        """
        return cls(
            event_type=EventType.BAR,
            timestamp=pd.Timestamp(series.name) if hasattr(series, 'name') else pd.Timestamp.now(),
            symbol=symbol,
            open=float(series.get("open", 0)),
            high=float(series.get("high", 0)),
            low=float(series.get("low", 0)),
            close=float(series.get("close", 0)),
            volume=float(series.get("volume", 0)),
            vwap=series.get("vwap"),
            exchange=exchange,
            timeframe=timeframe,
        )


@dataclass(frozen=True)
class OrderBookEvent(Event):
    """
    Order book snapshot event.

    Represents a complete Level 2 order book snapshot
    or update for market microstructure analysis.

    Attributes:
        symbol: Trading symbol
        bids: List of bid levels (price, size, num_orders)
        asks: List of ask levels (price, size, num_orders)
        exchange: Exchange identifier
        sequence_id: Exchange sequence number for ordering
        is_snapshot: True if full snapshot, False if incremental update
    """

    symbol: str = ""
    bids: Tuple[OrderBookLevel, ...] = field(default_factory=tuple)
    asks: Tuple[OrderBookLevel, ...] = field(default_factory=tuple)
    exchange: str = ""
    sequence_id: Optional[int] = None
    is_snapshot: bool = True

    def __post_init__(self):
        object.__setattr__(self, 'event_type', EventType.ORDER_BOOK)

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
        if self.spread is not None and self.mid_price is not None and self.mid_price > 0:
            return (self.spread / self.mid_price) * 10000
        return None

    @property
    def total_bid_depth(self) -> float:
        """Calculate total bid depth (all levels)."""
        return sum(level.size for level in self.bids)

    @property
    def total_ask_depth(self) -> float:
        """Calculate total ask depth (all levels)."""
        return sum(level.size for level in self.asks)

    @property
    def imbalance(self) -> Optional[float]:
        """
        Calculate order book imbalance.

        Returns:
            Imbalance in range [-1, 1] where:
            - Positive values indicate more bid pressure
            - Negative values indicate more ask pressure
        """
        total_bid = self.total_bid_depth
        total_ask = self.total_ask_depth
        total = total_bid + total_ask
        if total > 0:
            return (total_bid - total_ask) / total
        return None

    def depth_at_price(self, price: float, side: str = "bid") -> float:
        """
        Get depth available at or better than price.

        Args:
            price: Price level
            side: "bid" or "ask"

        Returns:
            Total depth at or better than price
        """
        if side.lower() == "bid":
            return sum(level.size for level in self.bids if level.price >= price)
        else:
            return sum(level.size for level in self.asks if level.price <= price)

    def weighted_mid_price(self) -> Optional[float]:
        """
        Calculate size-weighted mid price.

        Returns:
            Weighted mid price based on best bid/ask sizes
        """
        if self.best_bid and self.best_ask:
            total_size = self.best_bid.size + self.best_ask.size
            if total_size > 0:
                return (
                    self.best_bid.price * self.best_ask.size +
                    self.best_ask.price * self.best_bid.size
                ) / total_size
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": str(self.timestamp),
            "created_at": self.created_at.isoformat(),
            "priority": self.priority.value,
            "source": self.source,
            "symbol": self.symbol,
            "bids": [
                {"price": l.price, "size": l.size, "num_orders": l.num_orders}
                for l in self.bids
            ],
            "asks": [
                {"price": l.price, "size": l.size, "num_orders": l.num_orders}
                for l in self.asks
            ],
            "exchange": self.exchange,
            "sequence_id": self.sequence_id,
            "is_snapshot": self.is_snapshot,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrderBookEvent":
        bids = tuple(
            OrderBookLevel(
                price=l["price"],
                size=l["size"],
                num_orders=l.get("num_orders", 1),
            )
            for l in data.get("bids", [])
        )
        asks = tuple(
            OrderBookLevel(
                price=l["price"],
                size=l["size"],
                num_orders=l.get("num_orders", 1),
            )
            for l in data.get("asks", [])
        )

        return cls(
            event_type=EventType.ORDER_BOOK,
            timestamp=pd.Timestamp(data["timestamp"]),
            event_id=data.get("event_id", str(uuid.uuid4())),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            priority=EventPriority(data.get("priority", EventPriority.NORMAL)),
            source=data.get("source", "market"),
            symbol=data.get("symbol", ""),
            bids=bids,
            asks=asks,
            exchange=data.get("exchange", ""),
            sequence_id=data.get("sequence_id"),
            is_snapshot=data.get("is_snapshot", True),
            metadata=data.get("metadata", {}),
        )
