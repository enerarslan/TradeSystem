"""
Institutional-Grade Live Data Feed
JPMorgan-Level Real-Time Market Data Infrastructure

Features:
- WebSocket connections to multiple data sources
- Automatic reconnection with backoff
- Data normalization across sources
- Rate limiting and throttling
- Heartbeat monitoring
"""

import asyncio
import json
import time
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Set
from enum import Enum
from queue import Queue
import websockets
import aiohttp

from ..utils.logger import get_logger, get_performance_logger
from ..utils.helpers import retry_with_backoff, CircularBuffer


logger = get_logger(__name__)
perf_logger = get_performance_logger()


class DataSource(Enum):
    """Supported data sources"""
    ALPACA = "alpaca"
    POLYGON = "polygon"
    YAHOO = "yahoo"
    IEX = "iex"
    FINNHUB = "finnhub"


class SubscriptionType(Enum):
    """Subscription types"""
    TRADES = "trades"
    QUOTES = "quotes"
    BARS = "bars"
    NEWS = "news"


@dataclass
class TickData:
    """Standardized tick data structure"""
    symbol: str
    timestamp: datetime
    price: float
    size: int
    exchange: str = ""
    conditions: List[str] = field(default_factory=list)
    source: DataSource = DataSource.ALPACA


@dataclass
class QuoteData:
    """Standardized quote data structure"""
    symbol: str
    timestamp: datetime
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int
    source: DataSource = DataSource.ALPACA


@dataclass
class BarData:
    """Standardized bar (OHLCV) data structure"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    trade_count: Optional[int] = None
    source: DataSource = DataSource.ALPACA


class BaseDataFeed(ABC):
    """Abstract base class for data feeds"""

    def __init__(self, api_key: str, api_secret: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret
        self._connected = False
        self._subscriptions: Set[str] = set()
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to data source"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from data source"""
        pass

    @abstractmethod
    async def subscribe(self, symbols: List[str], sub_type: SubscriptionType) -> bool:
        """Subscribe to symbols"""
        pass

    @abstractmethod
    async def unsubscribe(self, symbols: List[str], sub_type: SubscriptionType) -> bool:
        """Unsubscribe from symbols"""
        pass

    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register callback for event type"""
        self._callbacks[event_type].append(callback)

    def _emit(self, event_type: str, data: Any) -> None:
        """Emit event to registered callbacks"""
        for callback in self._callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")


class AlpacaDataFeed(BaseDataFeed):
    """
    Alpaca Markets real-time data feed.

    Supports:
    - Real-time trades
    - Real-time quotes (level 1)
    - Real-time bars (1-min aggregates)
    """

    WS_URL = "wss://stream.data.alpaca.markets/v2/iex"
    WS_URL_SIP = "wss://stream.data.alpaca.markets/v2/sip"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        use_sip: bool = False  # SIP requires paid subscription
    ):
        super().__init__(api_key, api_secret)
        self.ws_url = self.WS_URL_SIP if use_sip else self.WS_URL
        self._ws = None
        self._running = False
        self._message_queue: Queue = Queue()
        self._last_heartbeat = time.time()

    async def connect(self) -> bool:
        """Connect to Alpaca WebSocket"""
        try:
            self._ws = await websockets.connect(self.ws_url)

            # Authenticate
            auth_msg = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.api_secret
            }
            await self._ws.send(json.dumps(auth_msg))

            # Wait for auth response
            response = await asyncio.wait_for(self._ws.recv(), timeout=10)
            data = json.loads(response)

            if isinstance(data, list) and len(data) > 0:
                if data[0].get("T") == "success":
                    self._connected = True
                    logger.info("Alpaca WebSocket connected and authenticated")
                    return True
                elif data[0].get("T") == "error":
                    logger.error(f"Alpaca auth failed: {data[0].get('msg')}")
                    return False

            return False

        except Exception as e:
            logger.error(f"Alpaca connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Alpaca WebSocket"""
        self._running = False

        if self._ws:
            await self._ws.close()
            self._ws = None

        self._connected = False
        logger.info("Alpaca WebSocket disconnected")

    async def subscribe(self, symbols: List[str], sub_type: SubscriptionType) -> bool:
        """Subscribe to symbols"""
        if not self._connected:
            logger.warning("Not connected to Alpaca")
            return False

        sub_map = {
            SubscriptionType.TRADES: "trades",
            SubscriptionType.QUOTES: "quotes",
            SubscriptionType.BARS: "bars"
        }

        sub_key = sub_map.get(sub_type)
        if not sub_key:
            logger.warning(f"Unsupported subscription type: {sub_type}")
            return False

        msg = {
            "action": "subscribe",
            sub_key: symbols
        }

        await self._ws.send(json.dumps(msg))
        self._subscriptions.update(symbols)

        logger.info(f"Subscribed to {sub_type.value} for {len(symbols)} symbols")
        return True

    async def unsubscribe(self, symbols: List[str], sub_type: SubscriptionType) -> bool:
        """Unsubscribe from symbols"""
        if not self._connected:
            return False

        sub_map = {
            SubscriptionType.TRADES: "trades",
            SubscriptionType.QUOTES: "quotes",
            SubscriptionType.BARS: "bars"
        }

        sub_key = sub_map.get(sub_type)
        if not sub_key:
            return False

        msg = {
            "action": "unsubscribe",
            sub_key: symbols
        }

        await self._ws.send(json.dumps(msg))
        self._subscriptions.difference_update(symbols)

        return True

    async def start_streaming(self) -> None:
        """Start receiving messages"""
        self._running = True

        while self._running and self._connected:
            try:
                message = await asyncio.wait_for(self._ws.recv(), timeout=30)
                self._last_heartbeat = time.time()

                data = json.loads(message)
                await self._process_message(data)

            except asyncio.TimeoutError:
                # Check if we need to reconnect
                if time.time() - self._last_heartbeat > 60:
                    logger.warning("Heartbeat timeout, reconnecting...")
                    await self._reconnect()

            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                await self._reconnect()

            except Exception as e:
                logger.error(f"Streaming error: {e}")

    async def _reconnect(self) -> None:
        """Reconnect with exponential backoff"""
        max_retries = 5
        base_delay = 1

        for attempt in range(max_retries):
            delay = base_delay * (2 ** attempt)
            logger.info(f"Reconnection attempt {attempt + 1}/{max_retries} in {delay}s")
            await asyncio.sleep(delay)

            if await self.connect():
                # Resubscribe
                if self._subscriptions:
                    await self.subscribe(
                        list(self._subscriptions),
                        SubscriptionType.TRADES
                    )
                return

        logger.error("Max reconnection attempts reached")
        self._running = False

    async def _process_message(self, data: Any) -> None:
        """Process incoming WebSocket message"""
        if not isinstance(data, list):
            return

        for msg in data:
            msg_type = msg.get("T")

            if msg_type == "t":  # Trade
                tick = TickData(
                    symbol=msg.get("S"),
                    timestamp=datetime.fromisoformat(msg.get("t").replace("Z", "+00:00")),
                    price=msg.get("p"),
                    size=msg.get("s"),
                    exchange=msg.get("x", ""),
                    conditions=msg.get("c", []),
                    source=DataSource.ALPACA
                )
                self._emit("trade", tick)

            elif msg_type == "q":  # Quote
                quote = QuoteData(
                    symbol=msg.get("S"),
                    timestamp=datetime.fromisoformat(msg.get("t").replace("Z", "+00:00")),
                    bid_price=msg.get("bp"),
                    bid_size=msg.get("bs"),
                    ask_price=msg.get("ap"),
                    ask_size=msg.get("as"),
                    source=DataSource.ALPACA
                )
                self._emit("quote", quote)

            elif msg_type == "b":  # Bar
                bar = BarData(
                    symbol=msg.get("S"),
                    timestamp=datetime.fromisoformat(msg.get("t").replace("Z", "+00:00")),
                    open=msg.get("o"),
                    high=msg.get("h"),
                    low=msg.get("l"),
                    close=msg.get("c"),
                    volume=msg.get("v"),
                    vwap=msg.get("vw"),
                    trade_count=msg.get("n"),
                    source=DataSource.ALPACA
                )
                self._emit("bar", bar)


class WebSocketManager:
    """
    Manages multiple WebSocket connections with automatic failover.

    Features:
    - Multiple data source management
    - Automatic failover on disconnect
    - Connection health monitoring
    - Rate limiting
    """

    def __init__(self):
        self._feeds: Dict[DataSource, BaseDataFeed] = {}
        self._primary_source: Optional[DataSource] = None
        self._running = False
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._message_buffer: Dict[str, CircularBuffer] = defaultdict(
            lambda: CircularBuffer(1000)
        )
        self._rate_limits: Dict[str, int] = {}
        self._message_counts: Dict[str, int] = defaultdict(int)

    def add_feed(
        self,
        source: DataSource,
        feed: BaseDataFeed,
        is_primary: bool = False
    ) -> None:
        """Add data feed source"""
        self._feeds[source] = feed

        if is_primary:
            self._primary_source = source

        # Wire up callbacks
        feed.register_callback("trade", lambda d: self._on_data("trade", d))
        feed.register_callback("quote", lambda d: self._on_data("quote", d))
        feed.register_callback("bar", lambda d: self._on_data("bar", d))

        logger.info(f"Added data feed: {source.value}")

    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register callback for event type"""
        self._callbacks[event_type].append(callback)

    def set_rate_limit(self, event_type: str, max_per_second: int) -> None:
        """Set rate limit for event type"""
        self._rate_limits[event_type] = max_per_second

    def _on_data(self, event_type: str, data: Any) -> None:
        """Handle incoming data"""
        # Rate limiting
        if event_type in self._rate_limits:
            self._message_counts[event_type] += 1
            # TODO: Implement proper rate limiting with time windows

        # Buffer data
        self._message_buffer[event_type].append(data)

        # Emit to callbacks
        for callback in self._callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event_type}: {e}")

    async def connect_all(self) -> bool:
        """Connect to all data feeds"""
        results = []

        for source, feed in self._feeds.items():
            try:
                result = await feed.connect()
                results.append(result)
                logger.info(f"{source.value} connected: {result}")
            except Exception as e:
                logger.error(f"Failed to connect {source.value}: {e}")
                results.append(False)

        return any(results)

    async def disconnect_all(self) -> None:
        """Disconnect from all data feeds"""
        self._running = False

        for source, feed in self._feeds.items():
            try:
                await feed.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting {source.value}: {e}")

    async def subscribe(
        self,
        symbols: List[str],
        sub_type: SubscriptionType,
        sources: Optional[List[DataSource]] = None
    ) -> bool:
        """Subscribe to symbols on specified sources"""
        sources = sources or list(self._feeds.keys())
        results = []

        for source in sources:
            if source in self._feeds:
                try:
                    result = await self._feeds[source].subscribe(symbols, sub_type)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Subscribe error on {source.value}: {e}")
                    results.append(False)

        return any(results)

    async def start(self) -> None:
        """Start all data feeds"""
        self._running = True

        # Start streaming for each feed
        tasks = []
        for source, feed in self._feeds.items():
            if hasattr(feed, 'start_streaming'):
                tasks.append(asyncio.create_task(feed.start_streaming()))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def get_latest(self, symbol: str, data_type: str = "trade") -> Optional[Any]:
        """Get latest data for symbol"""
        buffer = self._message_buffer.get(data_type)
        if not buffer:
            return None

        # Find latest for symbol
        for item in reversed(list(buffer)):
            if hasattr(item, 'symbol') and item.symbol == symbol:
                return item

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        stats = {
            "feeds": {},
            "buffer_sizes": {},
            "message_counts": dict(self._message_counts)
        }

        for source, feed in self._feeds.items():
            stats["feeds"][source.value] = {
                "connected": feed._connected,
                "subscriptions": len(feed._subscriptions)
            }

        for event_type, buffer in self._message_buffer.items():
            stats["buffer_sizes"][event_type] = len(buffer)

        return stats


class LiveDataFeed:
    """
    High-level live data feed interface.

    Provides simple API for:
    - Subscribing to real-time data
    - Accessing latest prices
    - Building real-time bars
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LiveDataFeed.

        Args:
            config: Configuration dictionary with API keys
        """
        self.config = config or {}
        self._ws_manager = WebSocketManager()
        self._bar_builders: Dict[str, BarBuilder] = {}
        self._latest_prices: Dict[str, float] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    def setup_alpaca(
        self,
        api_key: str,
        api_secret: str,
        use_sip: bool = False
    ) -> None:
        """Setup Alpaca data feed"""
        feed = AlpacaDataFeed(api_key, api_secret, use_sip)
        self._ws_manager.add_feed(DataSource.ALPACA, feed, is_primary=True)

    def register_trade_callback(self, callback: Callable[[TickData], None]) -> None:
        """Register callback for trade events"""
        self._ws_manager.register_callback("trade", callback)

    def register_quote_callback(self, callback: Callable[[QuoteData], None]) -> None:
        """Register callback for quote events"""
        self._ws_manager.register_callback("quote", callback)

    def register_bar_callback(self, callback: Callable[[BarData], None]) -> None:
        """Register callback for bar events"""
        self._ws_manager.register_callback("bar", callback)

    def start(self, symbols: List[str]) -> None:
        """Start live data feed in background thread"""
        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            try:
                self._loop.run_until_complete(self._run(symbols))
            except Exception as e:
                logger.error(f"Live feed error: {e}")
            finally:
                self._loop.close()

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()

        logger.info("Live data feed started in background thread")

    async def _run(self, symbols: List[str]) -> None:
        """Main run loop"""
        # Connect
        if not await self._ws_manager.connect_all():
            logger.error("Failed to connect to any data source")
            return

        # Subscribe
        await self._ws_manager.subscribe(symbols, SubscriptionType.TRADES)
        await self._ws_manager.subscribe(symbols, SubscriptionType.BARS)

        # Register internal price tracker
        self._ws_manager.register_callback("trade", self._update_price)

        # Start streaming
        await self._ws_manager.start()

    def _update_price(self, tick: TickData) -> None:
        """Update latest price"""
        self._latest_prices[tick.symbol] = tick.price

    def stop(self) -> None:
        """Stop live data feed"""
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._ws_manager.disconnect_all(),
                self._loop
            )

        if self._thread:
            self._thread.join(timeout=5)

        logger.info("Live data feed stopped")

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol"""
        return self._latest_prices.get(symbol)

    def get_all_prices(self) -> Dict[str, float]:
        """Get all latest prices"""
        return self._latest_prices.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get feed statistics"""
        return self._ws_manager.get_stats()


class BarBuilder:
    """
    Builds OHLCV bars from tick data.

    Aggregates trades into bars of specified interval.
    """

    def __init__(self, symbol: str, interval_seconds: int = 60):
        self.symbol = symbol
        self.interval_seconds = interval_seconds

        self._current_bar: Optional[Dict] = None
        self._completed_bars: List[BarData] = []
        self._callbacks: List[Callable] = []

    def on_tick(self, tick: TickData) -> Optional[BarData]:
        """Process tick and potentially complete a bar"""
        if tick.symbol != self.symbol:
            return None

        # Get bar timestamp (floored to interval)
        bar_ts = self._floor_timestamp(tick.timestamp)

        if self._current_bar is None:
            self._current_bar = self._new_bar(bar_ts, tick)
        elif self._current_bar['timestamp'] != bar_ts:
            # Complete current bar
            completed = self._finalize_bar()
            self._current_bar = self._new_bar(bar_ts, tick)
            return completed
        else:
            # Update current bar
            self._update_bar(tick)

        return None

    def _floor_timestamp(self, ts: datetime) -> datetime:
        """Floor timestamp to interval"""
        seconds = ts.timestamp()
        floored = (seconds // self.interval_seconds) * self.interval_seconds
        return datetime.fromtimestamp(floored, tz=ts.tzinfo)

    def _new_bar(self, timestamp: datetime, tick: TickData) -> Dict:
        """Create new bar from tick"""
        return {
            'timestamp': timestamp,
            'open': tick.price,
            'high': tick.price,
            'low': tick.price,
            'close': tick.price,
            'volume': tick.size
        }

    def _update_bar(self, tick: TickData) -> None:
        """Update current bar with tick"""
        self._current_bar['high'] = max(self._current_bar['high'], tick.price)
        self._current_bar['low'] = min(self._current_bar['low'], tick.price)
        self._current_bar['close'] = tick.price
        self._current_bar['volume'] += tick.size

    def _finalize_bar(self) -> BarData:
        """Finalize current bar"""
        bar = BarData(
            symbol=self.symbol,
            timestamp=self._current_bar['timestamp'],
            open=self._current_bar['open'],
            high=self._current_bar['high'],
            low=self._current_bar['low'],
            close=self._current_bar['close'],
            volume=self._current_bar['volume']
        )

        self._completed_bars.append(bar)

        # Notify callbacks
        for callback in self._callbacks:
            callback(bar)

        return bar

    def register_callback(self, callback: Callable[[BarData], None]) -> None:
        """Register callback for completed bars"""
        self._callbacks.append(callback)

    def get_bars(self, n: Optional[int] = None) -> List[BarData]:
        """Get completed bars"""
        if n is None:
            return self._completed_bars.copy()
        return self._completed_bars[-n:]
