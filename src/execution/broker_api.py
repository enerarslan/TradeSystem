"""
Broker API Integration
JPMorgan-Level Multi-Broker Connectivity

Supported Brokers:
- Alpaca (Paper and Live)
- Interactive Brokers (TWS/Gateway)
- Abstract interface for additional brokers
"""

import asyncio
import aiohttp
import websockets
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import threading
import queue
import time
import json
import hmac
import hashlib
import uuid
import functools
import random

from ..utils.logger import get_logger, get_audit_logger

logger = get_logger(__name__)
audit_logger = get_audit_logger()


# =============================================================================
# RETRY DECORATOR WITH EXPONENTIAL BACKOFF
# =============================================================================

class RetryableError(Exception):
    """Exception that indicates the operation should be retried"""
    pass


class NonRetryableError(Exception):
    """Exception that indicates the operation should NOT be retried"""
    pass


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 16.0,
    exponential_base: float = 2.0,
    retryable_status_codes: tuple = (502, 503, 504, 429),
    retryable_exceptions: tuple = (
        aiohttp.ClientError,
        asyncio.TimeoutError,
        ConnectionError,
        OSError,
    ),
    jitter: bool = True
):
    """
    Decorator for async functions to retry with exponential backoff.

    Implements JPMorgan-level retry logic:
    - Retries on HTTP 502, 503, 504, 429 (rate limit) errors
    - Retries on connection errors and timeouts
    - Uses exponential backoff: 1s, 2s, 4s, 8s... (capped at max_delay)
    - Adds jitter to prevent thundering herd
    - Logs each retry attempt
    - Raises after max retries with clear error message

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap in seconds
        exponential_base: Multiplier for each retry
        retryable_status_codes: HTTP status codes that trigger retry
        retryable_exceptions: Exception types that trigger retry
        jitter: Add random jitter to delay (prevents thundering herd)

    Example:
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        async def submit_order(self, order: OrderRequest) -> OrderResponse:
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except aiohttp.ClientResponseError as e:
                    # Check if status code is retryable
                    if e.status not in retryable_status_codes:
                        logger.error(f"{func.__name__} failed with non-retryable status {e.status}: {e.message}")
                        raise

                    last_exception = e
                    error_msg = f"HTTP {e.status}: {e.message}"

                except retryable_exceptions as e:
                    last_exception = e
                    error_msg = f"{type(e).__name__}: {str(e)}"

                except NonRetryableError:
                    # Explicitly marked as non-retryable
                    raise

                except Exception as e:
                    # Unexpected exception - don't retry
                    logger.error(f"{func.__name__} failed with unexpected error: {e}")
                    raise

                # Check if we have retries left
                if attempt >= max_retries:
                    logger.error(
                        f"{func.__name__} failed after {max_retries + 1} attempts. "
                        f"Last error: {error_msg}"
                    )
                    raise last_exception

                # Calculate delay with exponential backoff
                delay = min(base_delay * (exponential_base ** attempt), max_delay)

                # Add jitter (0-50% of delay)
                if jitter:
                    delay = delay * (1 + random.uniform(0, 0.5))

                logger.warning(
                    f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} failed: {error_msg}. "
                    f"Retrying in {delay:.2f}s..."
                )

                audit_logger.log_order(
                    order_id="retry",
                    symbol="",
                    action="RETRY_ATTEMPT",
                    details={
                        'function': func.__name__,
                        'attempt': attempt + 1,
                        'max_retries': max_retries + 1,
                        'error': error_msg,
                        'delay': delay
                    }
                )

                await asyncio.sleep(delay)

            # Should never reach here, but just in case
            raise last_exception

        return wrapper
    return decorator


# =============================================================================
# WEBSOCKET RECONNECTION MANAGER
# =============================================================================

@dataclass
class WebSocketConfig:
    """Configuration for WebSocket connection"""
    url: str
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    max_reconnect_attempts: int = -1  # -1 = unlimited
    ping_interval: float = 30.0
    ping_timeout: float = 10.0
    jitter: bool = True


class WebSocketReconnectManager:
    """
    Manages WebSocket connections with automatic reconnection.

    Features:
    - Detects disconnection (ping timeout or connection error)
    - Attempts reconnection with exponential backoff (1s, 2s, 4s... max 60s)
    - Resubscribes to previous channels on reconnect
    - Emits on_reconnected event for state sync
    - Integrates with GracefulDegradationManager

    Usage:
        manager = WebSocketReconnectManager(
            config=WebSocketConfig(url="wss://api.example.com/stream"),
            auth_handler=lambda ws: ws.send(json.dumps({"auth": "key"})),
            message_handler=self._handle_message
        )
        await manager.connect()
    """

    def __init__(
        self,
        config: WebSocketConfig,
        auth_handler: Optional[Callable] = None,
        message_handler: Optional[Callable] = None,
        on_connect: Optional[Callable] = None,
        on_disconnect: Optional[Callable] = None,
        on_reconnect: Optional[Callable] = None,
        degradation_manager: Optional[Any] = None  # GracefulDegradationManager
    ):
        self.config = config
        self.auth_handler = auth_handler
        self.message_handler = message_handler
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.on_reconnect = on_reconnect
        self.degradation_manager = degradation_manager

        # State
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._connected = False
        self._reconnect_attempts = 0
        self._subscriptions: List[Dict[str, Any]] = []
        self._last_pong: Optional[datetime] = None

        # Tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        self._reconnect_lock = asyncio.Lock()

        # Statistics
        self._stats = {
            'connects': 0,
            'disconnects': 0,
            'reconnects': 0,
            'messages_received': 0,
            'messages_sent': 0,
            'ping_failures': 0,
            'last_connected': None,
            'last_disconnected': None,
            'total_uptime_seconds': 0
        }

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self._connected and self._ws is not None

    async def connect(self) -> bool:
        """
        Establish WebSocket connection with auto-reconnect enabled.

        Returns:
            True if initial connection successful
        """
        if self._running:
            return self._connected

        self._running = True

        try:
            await self._establish_connection()
            return True
        except Exception as e:
            logger.error(f"Initial WebSocket connection failed: {e}")
            # Start reconnection loop in background
            asyncio.create_task(self._reconnect_loop())
            return False

    async def disconnect(self) -> None:
        """Gracefully disconnect WebSocket"""
        self._running = False

        # Cancel tasks
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass

        # Close connection
        if self._ws:
            await self._ws.close()
            self._ws = None

        self._connected = False
        self._stats['disconnects'] += 1
        self._stats['last_disconnected'] = datetime.now().isoformat()

        logger.info("WebSocket disconnected gracefully")

    async def subscribe(self, subscription: Dict[str, Any]) -> bool:
        """
        Subscribe to a channel. Subscription will be replayed on reconnect.

        Args:
            subscription: Subscription message to send

        Returns:
            True if subscription sent successfully
        """
        # Store for replay on reconnect
        if subscription not in self._subscriptions:
            self._subscriptions.append(subscription)

        if self._connected and self._ws:
            try:
                await self._ws.send(json.dumps(subscription))
                self._stats['messages_sent'] += 1
                return True
            except Exception as e:
                logger.error(f"Failed to subscribe: {e}")
                return False
        return False

    async def unsubscribe(self, subscription: Dict[str, Any]) -> bool:
        """Remove subscription"""
        if subscription in self._subscriptions:
            self._subscriptions.remove(subscription)
        return True

    async def send(self, message: Dict[str, Any]) -> bool:
        """Send message through WebSocket"""
        if not self._connected or not self._ws:
            logger.warning("Cannot send: WebSocket not connected")
            return False

        try:
            await self._ws.send(json.dumps(message))
            self._stats['messages_sent'] += 1
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            await self._handle_disconnect()
            return False

    async def _establish_connection(self) -> None:
        """Establish WebSocket connection"""
        logger.info(f"Connecting to WebSocket: {self.config.url}")

        self._ws = await websockets.connect(
            self.config.url,
            ping_interval=None,  # We handle pings ourselves
            ping_timeout=None,
            close_timeout=5.0
        )

        self._connected = True
        self._reconnect_attempts = 0
        self._last_pong = datetime.now()
        self._stats['connects'] += 1
        self._stats['last_connected'] = datetime.now().isoformat()

        logger.info("WebSocket connected")

        # Authenticate if handler provided
        if self.auth_handler:
            await self.auth_handler(self._ws)

        # Replay subscriptions
        for sub in self._subscriptions:
            try:
                await self._ws.send(json.dumps(sub))
                self._stats['messages_sent'] += 1
            except Exception as e:
                logger.error(f"Failed to replay subscription: {e}")

        # Start receive and ping tasks
        self._receive_task = asyncio.create_task(self._receive_loop())
        self._ping_task = asyncio.create_task(self._ping_loop())

        # Notify callbacks
        if self.on_connect:
            try:
                await self.on_connect() if asyncio.iscoroutinefunction(self.on_connect) else self.on_connect()
            except Exception as e:
                logger.error(f"on_connect callback error: {e}")

    async def _receive_loop(self) -> None:
        """Receive messages from WebSocket"""
        while self._running and self._connected:
            try:
                message = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=self.config.ping_timeout * 2
                )

                self._stats['messages_received'] += 1

                # Handle pong
                if message == 'pong' or (isinstance(message, str) and '"type":"pong"' in message):
                    self._last_pong = datetime.now()
                    continue

                # Handle message
                if self.message_handler:
                    try:
                        await self.message_handler(message) if asyncio.iscoroutinefunction(self.message_handler) \
                            else self.message_handler(message)
                    except Exception as e:
                        logger.error(f"Message handler error: {e}")

            except asyncio.TimeoutError:
                # No message received - check if connection is still alive
                if self._last_pong:
                    elapsed = (datetime.now() - self._last_pong).total_seconds()
                    if elapsed > self.config.ping_timeout * 2:
                        logger.warning(f"No pong received for {elapsed:.1f}s - reconnecting")
                        await self._handle_disconnect()
                        break

            except websockets.ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
                await self._handle_disconnect()
                break

            except Exception as e:
                logger.error(f"WebSocket receive error: {e}")
                await self._handle_disconnect()
                break

    async def _ping_loop(self) -> None:
        """Send periodic pings to keep connection alive"""
        while self._running and self._connected:
            try:
                await asyncio.sleep(self.config.ping_interval)

                if self._ws and self._connected:
                    try:
                        await self._ws.ping()
                    except Exception as e:
                        logger.warning(f"Ping failed: {e}")
                        self._stats['ping_failures'] += 1
                        await self._handle_disconnect()
                        break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ping loop error: {e}")

    async def _handle_disconnect(self) -> None:
        """Handle disconnection and trigger reconnect"""
        async with self._reconnect_lock:
            if not self._connected:
                return

            self._connected = False
            self._stats['disconnects'] += 1
            self._stats['last_disconnected'] = datetime.now().isoformat()

            # Close WebSocket
            if self._ws:
                try:
                    await self._ws.close()
                except Exception:
                    pass
                self._ws = None

            # Notify callbacks
            if self.on_disconnect:
                try:
                    await self.on_disconnect() if asyncio.iscoroutinefunction(self.on_disconnect) \
                        else self.on_disconnect()
                except Exception as e:
                    logger.error(f"on_disconnect callback error: {e}")

            # Start reconnection if still running
            if self._running:
                asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        """Attempt reconnection with exponential backoff"""
        while self._running and not self._connected:
            self._reconnect_attempts += 1

            # Check max attempts
            if self.config.max_reconnect_attempts > 0:
                if self._reconnect_attempts > self.config.max_reconnect_attempts:
                    logger.error(f"Max reconnect attempts ({self.config.max_reconnect_attempts}) reached")
                    self._running = False
                    return

            # Calculate delay with exponential backoff
            delay = min(
                self.config.initial_delay * (self.config.exponential_base ** (self._reconnect_attempts - 1)),
                self.config.max_delay
            )

            # Add jitter
            if self.config.jitter:
                delay = delay * (1 + random.uniform(0, 0.5))

            logger.info(
                f"WebSocket reconnect attempt {self._reconnect_attempts} in {delay:.2f}s..."
            )

            await asyncio.sleep(delay)

            try:
                await self._establish_connection()

                self._stats['reconnects'] += 1
                logger.info(f"WebSocket reconnected after {self._reconnect_attempts} attempts")

                # Notify reconnect callback
                if self.on_reconnect:
                    try:
                        await self.on_reconnect() if asyncio.iscoroutinefunction(self.on_reconnect) \
                            else self.on_reconnect()
                    except Exception as e:
                        logger.error(f"on_reconnect callback error: {e}")

                # Notify degradation manager
                if self.degradation_manager:
                    try:
                        self.degradation_manager.force_recovery(ComponentType.WEBSOCKET)
                    except Exception as e:
                        logger.debug(f"Could not notify degradation manager: {e}")

                return

            except Exception as e:
                logger.warning(f"Reconnect attempt {self._reconnect_attempts} failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            **self._stats,
            'connected': self._connected,
            'reconnect_attempts': self._reconnect_attempts,
            'subscriptions': len(self._subscriptions),
            'uptime_current': (
                (datetime.now() - datetime.fromisoformat(self._stats['last_connected'])).total_seconds()
                if self._connected and self._stats['last_connected']
                else 0
            )
        }


# Import for type hints
try:
    from ..core.graceful_degradation import ComponentType
except ImportError:
    # Fallback if not available
    class ComponentType(Enum):
        WEBSOCKET = "websocket"


class ConnectionStatus(Enum):
    """Broker connection status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    ERROR = "error"
    RECONNECTING = "reconnecting"


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(Enum):
    """Time in force"""
    DAY = "day"
    GTC = "gtc"  # Good till cancelled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill
    OPG = "opg"  # At open
    CLS = "cls"  # At close


@dataclass
class AccountInfo:
    """Broker account information"""
    account_id: str
    buying_power: float
    cash: float
    portfolio_value: float
    equity: float
    margin_used: float
    margin_available: float
    day_trade_count: int
    pattern_day_trader: bool
    trading_blocked: bool
    account_blocked: bool
    currency: str = "USD"
    status: str = "active"


@dataclass
class OrderRequest:
    """Order request"""
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    extended_hours: bool = False
    client_order_id: Optional[str] = None

    def __post_init__(self):
        if self.client_order_id is None:
            self.client_order_id = str(uuid.uuid4())[:12]


@dataclass
class OrderResponse:
    """Order response from broker"""
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    quantity: int
    order_type: str
    status: str
    filled_quantity: int = 0
    filled_avg_price: float = 0
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    canceled_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    asset_class: str = "us_equity"
    legs: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'client_order_id': self.client_order_id,
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'order_type': self.order_type,
            'status': self.status,
            'filled_quantity': self.filled_quantity,
            'filled_avg_price': self.filled_avg_price,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class Position:
    """Position from broker"""
    symbol: str
    quantity: int
    avg_entry_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    current_price: float
    side: str  # 'long' or 'short'
    exchange: str = "NASDAQ"


class BrokerAPI(ABC):
    """
    Abstract broker API interface.

    All broker implementations must inherit from this class.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        paper: bool = True,
        **kwargs
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper

        self._status = ConnectionStatus.DISCONNECTED
        self._account: Optional[AccountInfo] = None
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, OrderResponse] = {}

        self._callbacks: Dict[str, List[Callable]] = {
            'order_update': [],
            'position_update': [],
            'account_update': [],
            'trade_update': [],
            'connection_status': []
        }

        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 5

    @property
    def status(self) -> ConnectionStatus:
        return self._status

    @property
    def is_connected(self) -> bool:
        return self._status in [ConnectionStatus.CONNECTED, ConnectionStatus.AUTHENTICATED]

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from broker"""
        pass

    @abstractmethod
    async def get_account(self) -> AccountInfo:
        """Get account information"""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get all positions"""
        pass

    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        pass

    @abstractmethod
    async def submit_order(self, order: OrderRequest) -> OrderResponse:
        """Submit order"""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        pass

    @abstractmethod
    async def cancel_all_orders(self) -> int:
        """Cancel all orders, return count"""
        pass

    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[OrderResponse]:
        """Get order status"""
        pass

    @abstractmethod
    async def get_orders(
        self,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[OrderResponse]:
        """Get orders"""
        pass

    @abstractmethod
    async def close_position(self, symbol: str) -> Optional[OrderResponse]:
        """Close position for symbol"""
        pass

    @abstractmethod
    async def close_all_positions(self) -> List[OrderResponse]:
        """Close all positions"""
        pass

    def register_callback(
        self,
        event_type: str,
        callback: Callable
    ) -> None:
        """Register callback for events"""
        if event_type in self._callbacks:
            self._callbacks[event_type].append(callback)

    def _emit(self, event_type: str, data: Any) -> None:
        """Emit event to callbacks"""
        for callback in self._callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _update_status(self, status: ConnectionStatus) -> None:
        """Update connection status"""
        old_status = self._status
        self._status = status
        if old_status != status:
            logger.info(f"Connection status: {old_status.value} -> {status.value}")
            self._emit('connection_status', status)


class AlpacaBroker(BrokerAPI):
    """
    Alpaca Markets broker implementation.

    Features:
    - REST API for orders
    - WebSocket for real-time updates
    - Paper and live trading
    - Fractional shares
    """

    PAPER_BASE_URL = "https://paper-api.alpaca.markets"
    LIVE_BASE_URL = "https://api.alpaca.markets"
    PAPER_WS_URL = "wss://paper-api.alpaca.markets/stream"
    LIVE_WS_URL = "wss://api.alpaca.markets/stream"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        paper: bool = True,
        **kwargs
    ):
        super().__init__(api_key, api_secret, paper, **kwargs)

        self.base_url = self.PAPER_BASE_URL if paper else self.LIVE_BASE_URL
        self.ws_url = self.PAPER_WS_URL if paper else self.LIVE_WS_URL

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._running = False

    def _get_headers(self) -> Dict[str, str]:
        """Get API headers"""
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json"
        }

    async def connect(self) -> bool:
        """Connect to Alpaca"""
        try:
            self._update_status(ConnectionStatus.CONNECTING)

            # Create session
            self._session = aiohttp.ClientSession(headers=self._get_headers())

            # Test connection
            account = await self.get_account()

            if account:
                self._account = account
                self._update_status(ConnectionStatus.AUTHENTICATED)

                # Start WebSocket
                self._running = True
                self._ws_task = asyncio.create_task(self._ws_loop())

                logger.info(f"Connected to Alpaca ({'Paper' if self.paper else 'Live'})")
                return True

            return False

        except Exception as e:
            logger.error(f"Connection error: {e}")
            self._update_status(ConnectionStatus.ERROR)
            return False

    async def disconnect(self) -> None:
        """Disconnect from Alpaca"""
        self._running = False

        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            await self._ws.close()

        if self._session:
            await self._session.close()

        self._update_status(ConnectionStatus.DISCONNECTED)

    async def _ws_loop(self) -> None:
        """WebSocket message loop"""
        while self._running:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    self._ws = ws

                    # Authenticate
                    auth_msg = {
                        "action": "auth",
                        "key": self.api_key,
                        "secret": self.api_secret
                    }
                    await ws.send(json.dumps(auth_msg))

                    # Subscribe to trade updates
                    sub_msg = {
                        "action": "listen",
                        "data": {
                            "streams": ["trade_updates"]
                        }
                    }
                    await ws.send(json.dumps(sub_msg))

                    # Process messages
                    async for message in ws:
                        await self._handle_ws_message(message)

            except websockets.ConnectionClosed:
                if self._running:
                    logger.warning("WebSocket closed, reconnecting...")
                    await asyncio.sleep(self._reconnect_delay)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(self._reconnect_delay)

    async def _handle_ws_message(self, message: str) -> None:
        """Handle WebSocket message"""
        try:
            data = json.loads(message)
            stream = data.get('stream')

            if stream == 'trade_updates':
                trade_data = data.get('data', {})
                event = trade_data.get('event')
                order_data = trade_data.get('order', {})

                # Update order cache
                order_id = order_data.get('id')
                if order_id:
                    self._orders[order_id] = self._parse_order(order_data)

                # Emit event
                self._emit('trade_update', {
                    'event': event,
                    'order': order_data
                })

                # Log
                audit_logger.log_order(
                    order_id=order_id,
                    symbol=order_data.get('symbol'),
                    action=event.upper(),
                    details=order_data
                )

        except Exception as e:
            logger.error(f"Error handling WS message: {e}")

    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=8.0)
    async def get_account(self) -> AccountInfo:
        """Get account information"""
        async with self._session.get(f"{self.base_url}/v2/account") as resp:
            if resp.status == 200:
                data = await resp.json()
                return AccountInfo(
                    account_id=data['id'],
                    buying_power=float(data['buying_power']),
                    cash=float(data['cash']),
                    portfolio_value=float(data['portfolio_value']),
                    equity=float(data['equity']),
                    margin_used=float(data.get('initial_margin', 0)),
                    margin_available=float(data.get('regt_buying_power', 0)),
                    day_trade_count=int(data.get('daytrade_count', 0)),
                    pattern_day_trader=data.get('pattern_day_trader', False),
                    trading_blocked=data.get('trading_blocked', False),
                    account_blocked=data.get('account_blocked', False),
                    currency=data.get('currency', 'USD'),
                    status=data.get('status', 'active')
                )
            elif resp.status in (502, 503, 504, 429):
                # Raise retryable error
                error = await resp.text()
                raise aiohttp.ClientResponseError(
                    resp.request_info, resp.history,
                    status=resp.status, message=error
                )
            else:
                error = await resp.text()
                raise Exception(f"Failed to get account: {error}")

    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=8.0)
    async def get_positions(self) -> List[Position]:
        """Get all positions"""
        async with self._session.get(f"{self.base_url}/v2/positions") as resp:
            if resp.status == 200:
                data = await resp.json()
                positions = [self._parse_position(p) for p in data]
                self._positions = {p.symbol: p for p in positions}
                return positions
            elif resp.status in (502, 503, 504, 429):
                error = await resp.text()
                raise aiohttp.ClientResponseError(
                    resp.request_info, resp.history,
                    status=resp.status, message=error
                )
            else:
                error = await resp.text()
                raise Exception(f"Failed to get positions: {error}")

    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=8.0)
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        async with self._session.get(f"{self.base_url}/v2/positions/{symbol}") as resp:
            if resp.status == 200:
                data = await resp.json()
                position = self._parse_position(data)
                self._positions[symbol] = position
                return position
            elif resp.status == 404:
                return None
            elif resp.status in (502, 503, 504, 429):
                error = await resp.text()
                raise aiohttp.ClientResponseError(
                    resp.request_info, resp.history,
                    status=resp.status, message=error
                )
            else:
                error = await resp.text()
                raise Exception(f"Failed to get position: {error}")

    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=8.0)
    async def submit_order(self, order: OrderRequest) -> OrderResponse:
        """Submit order to Alpaca"""
        payload = {
            "symbol": order.symbol,
            "qty": str(order.quantity),
            "side": order.side.value,
            "type": order.order_type.value,
            "time_in_force": order.time_in_force.value,
            "client_order_id": order.client_order_id
        }

        if order.limit_price:
            payload["limit_price"] = str(order.limit_price)
        if order.stop_price:
            payload["stop_price"] = str(order.stop_price)
        if order.extended_hours:
            payload["extended_hours"] = True

        async with self._session.post(
            f"{self.base_url}/v2/orders",
            json=payload
        ) as resp:
            if resp.status in [200, 201]:
                data = await resp.json()
                response = self._parse_order(data)
                self._orders[response.order_id] = response

                audit_logger.log_order(
                    order_id=response.order_id,
                    symbol=order.symbol,
                    action="SUBMITTED",
                    details=response.to_dict()
                )

                return response
            elif resp.status in (502, 503, 504, 429):
                error = await resp.text()
                logger.warning(f"Order submission got retryable error {resp.status}: {error}")
                raise aiohttp.ClientResponseError(
                    resp.request_info, resp.history,
                    status=resp.status, message=error
                )
            else:
                error = await resp.text()
                logger.error(f"Order submission failed: {error}")
                raise Exception(f"Order failed: {error}")

    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=8.0)
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        async with self._session.delete(f"{self.base_url}/v2/orders/{order_id}") as resp:
            if resp.status in [200, 204]:
                audit_logger.log_order(
                    order_id=order_id,
                    symbol="",
                    action="CANCELLED",
                    details={'order_id': order_id}
                )
                return True
            elif resp.status in (502, 503, 504, 429):
                error = await resp.text()
                raise aiohttp.ClientResponseError(
                    resp.request_info, resp.history,
                    status=resp.status, message=error
                )
            else:
                return False

    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=8.0)
    async def cancel_all_orders(self) -> int:
        """Cancel all orders"""
        async with self._session.delete(f"{self.base_url}/v2/orders") as resp:
            if resp.status == 200:
                data = await resp.json()
                return len(data)
            elif resp.status in (502, 503, 504, 429):
                error = await resp.text()
                raise aiohttp.ClientResponseError(
                    resp.request_info, resp.history,
                    status=resp.status, message=error
                )
            return 0

    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=8.0)
    async def get_order(self, order_id: str) -> Optional[OrderResponse]:
        """Get order by ID"""
        async with self._session.get(f"{self.base_url}/v2/orders/{order_id}") as resp:
            if resp.status == 200:
                data = await resp.json()
                return self._parse_order(data)
            elif resp.status in (502, 503, 504, 429):
                error = await resp.text()
                raise aiohttp.ClientResponseError(
                    resp.request_info, resp.history,
                    status=resp.status, message=error
                )
            return None

    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=8.0)
    async def get_orders(
        self,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[OrderResponse]:
        """Get orders"""
        params = {"limit": limit}
        if status:
            params["status"] = status

        async with self._session.get(
            f"{self.base_url}/v2/orders",
            params=params
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return [self._parse_order(o) for o in data]
            elif resp.status in (502, 503, 504, 429):
                error = await resp.text()
                raise aiohttp.ClientResponseError(
                    resp.request_info, resp.history,
                    status=resp.status, message=error
                )
            return []

    async def close_position(self, symbol: str) -> Optional[OrderResponse]:
        """Close position"""
        async with self._session.delete(f"{self.base_url}/v2/positions/{symbol}") as resp:
            if resp.status == 200:
                data = await resp.json()
                return self._parse_order(data)
            return None

    async def close_all_positions(self) -> List[OrderResponse]:
        """Close all positions"""
        async with self._session.delete(f"{self.base_url}/v2/positions") as resp:
            if resp.status == 200:
                data = await resp.json()
                return [self._parse_order(o) for o in data]
            return []

    def _parse_order(self, data: Dict) -> OrderResponse:
        """Parse order response"""
        return OrderResponse(
            order_id=data['id'],
            client_order_id=data.get('client_order_id', ''),
            symbol=data['symbol'],
            side=data['side'],
            quantity=int(float(data['qty'])),
            order_type=data['type'],
            status=data['status'],
            filled_quantity=int(float(data.get('filled_qty', 0))),
            filled_avg_price=float(data.get('filled_avg_price', 0) or 0),
            limit_price=float(data['limit_price']) if data.get('limit_price') else None,
            stop_price=float(data['stop_price']) if data.get('stop_price') else None,
            created_at=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')) if data.get('created_at') else None,
            submitted_at=datetime.fromisoformat(data['submitted_at'].replace('Z', '+00:00')) if data.get('submitted_at') else None,
            filled_at=datetime.fromisoformat(data['filled_at'].replace('Z', '+00:00')) if data.get('filled_at') else None,
            asset_class=data.get('asset_class', 'us_equity')
        )

    def _parse_position(self, data: Dict) -> Position:
        """Parse position response"""
        qty = int(float(data['qty']))
        return Position(
            symbol=data['symbol'],
            quantity=qty,
            avg_entry_price=float(data['avg_entry_price']),
            market_value=float(data['market_value']),
            cost_basis=float(data['cost_basis']),
            unrealized_pnl=float(data['unrealized_pl']),
            unrealized_pnl_pct=float(data['unrealized_plpc']),
            current_price=float(data['current_price']),
            side='long' if qty > 0 else 'short',
            exchange=data.get('exchange', 'NASDAQ')
        )


class IBKRBroker(BrokerAPI):
    """
    Interactive Brokers implementation.

    Connects via TWS API or IB Gateway.
    Note: Requires ib_insync library.
    """

    def __init__(
        self,
        api_key: str = "",  # Not used for IBKR
        api_secret: str = "",  # Not used for IBKR
        paper: bool = True,
        host: str = "127.0.0.1",
        port: int = 7497,  # 7497 for paper, 7496 for live
        client_id: int = 1,
        **kwargs
    ):
        super().__init__(api_key, api_secret, paper, **kwargs)

        self.host = host
        self.port = port if paper else 7496
        self.client_id = client_id

        self._ib = None
        self._connected = False

    async def connect(self) -> bool:
        """Connect to IBKR TWS/Gateway"""
        try:
            from ib_insync import IB

            self._update_status(ConnectionStatus.CONNECTING)

            self._ib = IB()
            await self._ib.connectAsync(
                host=self.host,
                port=self.port,
                clientId=self.client_id
            )

            self._connected = True
            self._update_status(ConnectionStatus.AUTHENTICATED)

            # Register callbacks
            self._ib.orderStatusEvent += self._on_order_status
            self._ib.execDetailsEvent += self._on_execution

            logger.info(f"Connected to IBKR ({'Paper' if self.paper else 'Live'})")
            return True

        except ImportError:
            logger.error("ib_insync not installed. Install with: pip install ib_insync")
            return False
        except Exception as e:
            logger.error(f"IBKR connection error: {e}")
            self._update_status(ConnectionStatus.ERROR)
            return False

    async def disconnect(self) -> None:
        """Disconnect from IBKR"""
        if self._ib:
            self._ib.disconnect()
        self._connected = False
        self._update_status(ConnectionStatus.DISCONNECTED)

    def _on_order_status(self, trade) -> None:
        """Handle order status update"""
        self._emit('order_update', {
            'order_id': trade.order.orderId,
            'status': trade.orderStatus.status,
            'filled': trade.orderStatus.filled,
            'remaining': trade.orderStatus.remaining
        })

    def _on_execution(self, trade, fill) -> None:
        """Handle execution"""
        self._emit('trade_update', {
            'order_id': trade.order.orderId,
            'symbol': trade.contract.symbol,
            'quantity': fill.execution.shares,
            'price': fill.execution.price
        })

    async def get_account(self) -> AccountInfo:
        """Get IBKR account"""
        if not self._ib:
            raise Exception("Not connected")

        summary = self._ib.accountSummary()

        values = {item.tag: float(item.value) for item in summary}

        return AccountInfo(
            account_id=summary[0].account if summary else "unknown",
            buying_power=values.get('BuyingPower', 0),
            cash=values.get('TotalCashValue', 0),
            portfolio_value=values.get('NetLiquidation', 0),
            equity=values.get('EquityWithLoanValue', 0),
            margin_used=values.get('InitMarginReq', 0),
            margin_available=values.get('AvailableFunds', 0),
            day_trade_count=0,
            pattern_day_trader=False,
            trading_blocked=False,
            account_blocked=False
        )

    async def get_positions(self) -> List[Position]:
        """Get IBKR positions"""
        if not self._ib:
            return []

        positions = []
        for pos in self._ib.positions():
            positions.append(Position(
                symbol=pos.contract.symbol,
                quantity=int(pos.position),
                avg_entry_price=pos.avgCost / abs(pos.position) if pos.position != 0 else 0,
                market_value=pos.marketValue if hasattr(pos, 'marketValue') else 0,
                cost_basis=abs(pos.position) * pos.avgCost / abs(pos.position) if pos.position != 0 else 0,
                unrealized_pnl=pos.unrealizedPNL if hasattr(pos, 'unrealizedPNL') else 0,
                unrealized_pnl_pct=0,
                current_price=0,
                side='long' if pos.position > 0 else 'short'
            ))

        return positions

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        positions = await self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    async def submit_order(self, order: OrderRequest) -> OrderResponse:
        """Submit order to IBKR"""
        if not self._ib:
            raise Exception("Not connected")

        from ib_insync import Stock, Order as IBOrder

        # Create contract
        contract = Stock(order.symbol, 'SMART', 'USD')

        # Create order
        ib_order = IBOrder()
        ib_order.action = 'BUY' if order.side == OrderSide.BUY else 'SELL'
        ib_order.totalQuantity = order.quantity
        ib_order.orderType = order.order_type.value.upper()

        if order.limit_price:
            ib_order.lmtPrice = order.limit_price
        if order.stop_price:
            ib_order.auxPrice = order.stop_price

        # Submit
        trade = self._ib.placeOrder(contract, ib_order)

        return OrderResponse(
            order_id=str(trade.order.orderId),
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            order_type=order.order_type.value,
            status='submitted',
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            created_at=datetime.now()
        )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel IBKR order"""
        if not self._ib:
            return False

        for trade in self._ib.openTrades():
            if str(trade.order.orderId) == order_id:
                self._ib.cancelOrder(trade.order)
                return True
        return False

    async def cancel_all_orders(self) -> int:
        """Cancel all IBKR orders"""
        if not self._ib:
            return 0

        count = 0
        for trade in self._ib.openTrades():
            self._ib.cancelOrder(trade.order)
            count += 1
        return count

    async def get_order(self, order_id: str) -> Optional[OrderResponse]:
        """Get IBKR order"""
        if not self._ib:
            return None

        for trade in self._ib.trades():
            if str(trade.order.orderId) == order_id:
                return OrderResponse(
                    order_id=str(trade.order.orderId),
                    client_order_id="",
                    symbol=trade.contract.symbol,
                    side='buy' if trade.order.action == 'BUY' else 'sell',
                    quantity=int(trade.order.totalQuantity),
                    order_type=trade.order.orderType.lower(),
                    status=trade.orderStatus.status.lower(),
                    filled_quantity=int(trade.orderStatus.filled),
                    filled_avg_price=trade.orderStatus.avgFillPrice
                )
        return None

    async def get_orders(
        self,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[OrderResponse]:
        """Get IBKR orders"""
        if not self._ib:
            return []

        orders = []
        trades = self._ib.openTrades() if status == 'open' else self._ib.trades()

        for trade in trades[:limit]:
            orders.append(OrderResponse(
                order_id=str(trade.order.orderId),
                client_order_id="",
                symbol=trade.contract.symbol,
                side='buy' if trade.order.action == 'BUY' else 'sell',
                quantity=int(trade.order.totalQuantity),
                order_type=trade.order.orderType.lower(),
                status=trade.orderStatus.status.lower(),
                filled_quantity=int(trade.orderStatus.filled),
                filled_avg_price=trade.orderStatus.avgFillPrice
            ))

        return orders

    async def close_position(self, symbol: str) -> Optional[OrderResponse]:
        """Close IBKR position"""
        position = await self.get_position(symbol)
        if not position:
            return None

        side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
        order = OrderRequest(
            symbol=symbol,
            side=side,
            quantity=abs(position.quantity),
            order_type=OrderType.MARKET
        )

        return await self.submit_order(order)

    async def close_all_positions(self) -> List[OrderResponse]:
        """Close all IBKR positions"""
        responses = []
        positions = await self.get_positions()

        for pos in positions:
            response = await self.close_position(pos.symbol)
            if response:
                responses.append(response)

        return responses


class BrokerFactory:
    """Factory for creating broker instances"""

    _brokers = {
        'alpaca': AlpacaBroker,
        'ibkr': IBKRBroker,
        'interactive_brokers': IBKRBroker
    }

    @classmethod
    def create(
        cls,
        broker_name: str,
        **kwargs
    ) -> BrokerAPI:
        """Create broker instance"""
        broker_name = broker_name.lower()

        if broker_name not in cls._brokers:
            raise ValueError(f"Unknown broker: {broker_name}. Available: {list(cls._brokers.keys())}")

        return cls._brokers[broker_name](**kwargs)

    @classmethod
    def register(cls, name: str, broker_class: type) -> None:
        """Register new broker"""
        cls._brokers[name.lower()] = broker_class
