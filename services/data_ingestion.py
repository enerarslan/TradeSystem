"""
Data Ingestion Service
======================

Connects to market data sources (Alpaca WebSocket) and pushes
normalized data to the message bus for consumption by other services.

Responsibilities:
- WebSocket connection to Alpaca
- Real-time bar/quote/trade data
- Data normalization
- Heartbeat monitoring
- Reconnection handling

Author: AlphaTrade Platform
Version: 3.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import os

from config.settings import get_logger
from infrastructure.message_bus import (
    Message,
    MessageType,
    Channel,
    MessagePriority,
)
from infrastructure.service_registry import ServiceType
from services.base_service import BaseService, ServiceConfig

logger = get_logger(__name__)

# Try to import Alpaca
try:
    from alpaca.data.live import StockDataStream
    from alpaca.data.enums import DataFeed
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not installed, using mock data")


@dataclass
class DataIngestionConfig(ServiceConfig):
    """Configuration for data ingestion service."""
    name: str = "data_ingestion"
    service_type: ServiceType = ServiceType.DATA_INGESTION
    symbols: list[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL"])
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    data_feed: str = "iex"  # "sip" for pro, "iex" for free
    subscribe_bars: bool = True
    subscribe_quotes: bool = True
    subscribe_trades: bool = False


class DataIngestionService(BaseService):
    """
    Data Ingestion Service for real-time market data.

    Connects to Alpaca WebSocket stream and publishes normalized
    market data events to the message bus.

    Example:
        config = DataIngestionConfig(
            symbols=["AAPL", "MSFT", "GOOGL"],
            alpaca_api_key="your_key",
            alpaca_secret_key="your_secret"
        )

        service = DataIngestionService(config)
        await service.run_forever()
    """

    def __init__(self, config: DataIngestionConfig | None = None):
        """Initialize data ingestion service."""
        config = config or DataIngestionConfig()

        # Get API keys from environment if not provided
        if not config.alpaca_api_key:
            config.alpaca_api_key = os.getenv("ALPACA_API_KEY", "")
        if not config.alpaca_secret_key:
            config.alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY", "")

        super().__init__(config)

        self.config: DataIngestionConfig = config
        self._stream: StockDataStream | None = None
        self._reconnect_task: asyncio.Task | None = None
        self._last_data_time: dict[str, float] = {}
        self._bar_count = 0
        self._quote_count = 0

    async def _on_start(self) -> None:
        """Start data ingestion."""
        logger.info(f"Starting data ingestion for {len(self.config.symbols)} symbols")

        if ALPACA_AVAILABLE and self.config.alpaca_api_key:
            await self._connect_alpaca()
        else:
            # Start mock data generator for testing
            logger.warning("Using mock data generator")
            self.add_background_task(self._mock_data_generator())

    async def _on_stop(self) -> None:
        """Stop data ingestion."""
        if self._stream:
            try:
                await self._stream.close()
            except Exception as e:
                logger.error(f"Error closing stream: {e}")

        logger.info(
            f"Data ingestion stopped. Bars: {self._bar_count}, Quotes: {self._quote_count}"
        )

    async def _connect_alpaca(self) -> None:
        """Connect to Alpaca WebSocket stream."""
        try:
            data_feed = DataFeed.IEX if self.config.data_feed == "iex" else DataFeed.SIP

            self._stream = StockDataStream(
                api_key=self.config.alpaca_api_key,
                secret_key=self.config.alpaca_secret_key,
                feed=data_feed,
            )

            # Subscribe to bars
            if self.config.subscribe_bars:
                self._stream.subscribe_bars(
                    self._handle_bar,
                    *self.config.symbols,
                )

            # Subscribe to quotes
            if self.config.subscribe_quotes:
                self._stream.subscribe_quotes(
                    self._handle_quote,
                    *self.config.symbols,
                )

            # Subscribe to trades
            if self.config.subscribe_trades:
                self._stream.subscribe_trades(
                    self._handle_trade,
                    *self.config.symbols,
                )

            # Run stream in background
            self.add_background_task(self._run_stream())

            logger.info(f"Connected to Alpaca {data_feed.value} feed")

        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            # Fall back to mock data
            self.add_background_task(self._mock_data_generator())

    async def _run_stream(self) -> None:
        """Run the Alpaca stream with reconnection logic."""
        while self._running:
            try:
                await self._stream.run()
            except Exception as e:
                logger.error(f"Stream error: {e}")
                if self._running:
                    logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)
                    await self._connect_alpaca()

    async def _handle_bar(self, bar) -> None:
        """Handle incoming bar data."""
        try:
            self._bar_count += 1
            self._last_data_time[bar.symbol] = time.time()

            # Create market data message
            message = Message(
                type=MessageType.MARKET_DATA,
                channel=Channel.MARKET_DATA,
                payload={
                    "symbol": bar.symbol,
                    "timestamp": bar.timestamp.isoformat(),
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume),
                    "vwap": float(bar.vwap) if hasattr(bar, "vwap") else None,
                    "trade_count": int(bar.trade_count) if hasattr(bar, "trade_count") else None,
                    "data_type": "bar",
                },
                priority=MessagePriority.HIGH,
                source=self.name,
            )

            await self.publish(message)

            if self._bar_count % 100 == 0:
                logger.debug(f"Published {self._bar_count} bars")

        except Exception as e:
            logger.error(f"Error handling bar: {e}")

    async def _handle_quote(self, quote) -> None:
        """Handle incoming quote data."""
        try:
            self._quote_count += 1

            message = Message(
                type=MessageType.TICK_DATA,
                channel=Channel.MARKET_DATA,
                payload={
                    "symbol": quote.symbol,
                    "timestamp": quote.timestamp.isoformat(),
                    "bid_price": float(quote.bid_price),
                    "bid_size": int(quote.bid_size),
                    "ask_price": float(quote.ask_price),
                    "ask_size": int(quote.ask_size),
                    "data_type": "quote",
                },
                priority=MessagePriority.NORMAL,
                source=self.name,
            )

            await self.publish(message)

        except Exception as e:
            logger.error(f"Error handling quote: {e}")

    async def _handle_trade(self, trade) -> None:
        """Handle incoming trade data."""
        try:
            message = Message(
                type=MessageType.TICK_DATA,
                channel=Channel.MARKET_DATA,
                payload={
                    "symbol": trade.symbol,
                    "timestamp": trade.timestamp.isoformat(),
                    "price": float(trade.price),
                    "size": int(trade.size),
                    "exchange": trade.exchange,
                    "data_type": "trade",
                },
                priority=MessagePriority.NORMAL,
                source=self.name,
            )

            await self.publish(message)

        except Exception as e:
            logger.error(f"Error handling trade: {e}")

    async def _mock_data_generator(self) -> None:
        """Generate mock market data for testing."""
        import random

        # Initial prices
        prices = {symbol: 150.0 + random.uniform(-50, 100) for symbol in self.config.symbols}

        while self._running:
            try:
                for symbol in self.config.symbols:
                    # Random walk
                    change_pct = random.gauss(0, 0.001)
                    prices[symbol] *= (1 + change_pct)

                    price = prices[symbol]
                    high = price * (1 + abs(random.gauss(0, 0.001)))
                    low = price * (1 - abs(random.gauss(0, 0.001)))

                    message = Message(
                        type=MessageType.MARKET_DATA,
                        channel=Channel.MARKET_DATA,
                        payload={
                            "symbol": symbol,
                            "timestamp": datetime.now().isoformat(),
                            "open": round(price * (1 + random.gauss(0, 0.0005)), 2),
                            "high": round(high, 2),
                            "low": round(low, 2),
                            "close": round(price, 2),
                            "volume": random.randint(1000, 100000),
                            "data_type": "bar",
                            "mock": True,
                        },
                        priority=MessagePriority.HIGH,
                        source=self.name,
                    )

                    await self.publish(message)
                    self._bar_count += 1

                # Generate bars every second for testing
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Mock data error: {e}")
                await asyncio.sleep(1)

    def get_status(self) -> dict[str, Any]:
        """Get service status."""
        status = super().get_status()
        status.update({
            "symbols": self.config.symbols,
            "bar_count": self._bar_count,
            "quote_count": self._quote_count,
            "last_data_time": self._last_data_time,
            "stream_connected": self._stream is not None,
        })
        return status


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main():
    """Run data ingestion service."""
    config = DataIngestionConfig(
        symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
    )

    service = DataIngestionService(config)
    await service.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
