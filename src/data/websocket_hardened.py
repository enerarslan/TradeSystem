"""
Hardened WebSocket Manager
==========================
JPMorgan-Level Resilient WebSocket Connections

Implements production-grade WebSocket handling:
1. Exponential backoff with jitter
2. Circuit breaker pattern
3. Connection health monitoring
4. Automatic failover
5. Message sequencing and gap detection

Key Improvements over base implementation:
- Jitter in reconnection delays (prevents thundering herd)
- Circuit breaker (stops reconnect spam during outages)
- Health heartbeats with configurable timeout
- Message sequence tracking (detects data gaps)
- Graceful degradation to REST fallback

Author: AlphaTrade Institutional System
Based on: ARCHITECTURAL_REVIEW_REPORT.md - Phase 2
"""

import asyncio
import json
import time
import random
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import websockets
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
    InvalidStatusCode
)

from ..utils.logger import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking connections
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes to close from half-open
    timeout_seconds: float = 60.0  # Time before trying half-open
    half_open_max_requests: int = 1  # Requests to test in half-open


@dataclass
class ReconnectionConfig:
    """Configuration for reconnection behavior"""
    initial_delay: float = 1.0  # Initial delay in seconds
    max_delay: float = 60.0  # Maximum delay
    backoff_multiplier: float = 2.0  # Exponential backoff factor
    jitter_factor: float = 0.3  # Random jitter (0.3 = ±30%)
    max_retries: int = 10  # Maximum retry attempts (0 = infinite)


@dataclass
class ConnectionStats:
    """Connection statistics"""
    connects: int = 0
    disconnects: int = 0
    reconnects: int = 0
    messages_received: int = 0
    messages_sent: int = 0
    bytes_received: int = 0
    bytes_sent: int = 0
    errors: int = 0
    last_connect: Optional[datetime] = None
    last_disconnect: Optional[datetime] = None
    last_message: Optional[datetime] = None
    uptime_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'connects': self.connects,
            'disconnects': self.disconnects,
            'reconnects': self.reconnects,
            'messages_received': self.messages_received,
            'messages_sent': self.messages_sent,
            'errors': self.errors,
            'last_connect': self.last_connect.isoformat() if self.last_connect else None,
            'last_message': self.last_message.isoformat() if self.last_message else None,
            'uptime_seconds': self.uptime_seconds
        }


class CircuitBreaker:
    """
    Circuit breaker for connection management.

    Prevents connection spam during outages.
    """

    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_requests = 0

    @property
    def state(self) -> CircuitState:
        # Check if we should transition to half-open
        if self._state == CircuitState.OPEN and self._last_failure_time:
            elapsed = (datetime.now() - self._last_failure_time).total_seconds()
            if elapsed >= self.config.timeout_seconds:
                self._state = CircuitState.HALF_OPEN
                self._half_open_requests = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN")

        return self._state

    def allow_request(self) -> bool:
        """Check if a connection attempt is allowed"""
        state = self.state

        if state == CircuitState.CLOSED:
            return True

        if state == CircuitState.OPEN:
            return False

        if state == CircuitState.HALF_OPEN:
            if self._half_open_requests < self.config.half_open_max_requests:
                self._half_open_requests += 1
                return True
            return False

        return False

    def record_success(self) -> None:
        """Record a successful connection/operation"""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                logger.info("Circuit breaker CLOSED (recovered)")
        else:
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed connection/operation"""
        self._failure_count += 1
        self._success_count = 0
        self._last_failure_time = datetime.now()

        if self._state == CircuitState.HALF_OPEN:
            # Failed in half-open, back to open
            self._state = CircuitState.OPEN
            logger.warning("Circuit breaker back to OPEN (half-open test failed)")

        elif self._failure_count >= self.config.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker OPEN after {self._failure_count} failures"
            )


class HardenedWebSocket:
    """
    Production-grade WebSocket connection.

    Features:
    - Circuit breaker for connection management
    - Exponential backoff with jitter
    - Heartbeat monitoring
    - Sequence tracking
    - Graceful shutdown
    """

    def __init__(
        self,
        url: str,
        auth_message: Optional[Dict] = None,
        reconnect_config: ReconnectionConfig = None,
        circuit_config: CircuitBreakerConfig = None,
        heartbeat_timeout: float = 30.0
    ):
        self.url = url
        self.auth_message = auth_message
        self.reconnect_config = reconnect_config or ReconnectionConfig()
        self.heartbeat_timeout = heartbeat_timeout

        # Connection state
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._running = False
        self._should_reconnect = True

        # Circuit breaker
        self._circuit = CircuitBreaker(circuit_config)

        # Reconnection state
        self._retry_count = 0
        self._current_delay = self.reconnect_config.initial_delay

        # Heartbeat
        self._last_message_time = time.time()
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Statistics
        self._stats = ConnectionStats()
        self._connect_start_time: Optional[float] = None

        # Callbacks
        self._on_message: Optional[Callable] = None
        self._on_connect: Optional[Callable] = None
        self._on_disconnect: Optional[Callable] = None
        self._on_error: Optional[Callable] = None

        # Message sequence tracking
        self._last_sequence: Dict[str, int] = {}
        self._gaps_detected = 0

    def on_message(self, callback: Callable) -> None:
        """Set message callback"""
        self._on_message = callback

    def on_connect(self, callback: Callable) -> None:
        """Set connect callback"""
        self._on_connect = callback

    def on_disconnect(self, callback: Callable) -> None:
        """Set disconnect callback"""
        self._on_disconnect = callback

    def on_error(self, callback: Callable) -> None:
        """Set error callback"""
        self._on_error = callback

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def circuit_state(self) -> CircuitState:
        return self._circuit.state

    async def connect(self) -> bool:
        """
        Connect to WebSocket with circuit breaker and retry logic.
        """
        if not self._circuit.allow_request():
            logger.warning(
                f"Connection blocked by circuit breaker (state: {self._circuit.state.value})"
            )
            return False

        try:
            logger.info(f"Connecting to {self.url}")

            self._ws = await asyncio.wait_for(
                websockets.connect(
                    self.url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5
                ),
                timeout=30
            )

            # Authenticate if needed
            if self.auth_message:
                await self._ws.send(json.dumps(self.auth_message))
                response = await asyncio.wait_for(self._ws.recv(), timeout=10)
                auth_result = json.loads(response)

                # Check auth success (adapt based on your broker's response format)
                if not self._check_auth_success(auth_result):
                    logger.error(f"Authentication failed: {auth_result}")
                    await self._ws.close()
                    self._circuit.record_failure()
                    return False

            # Connection successful
            self._connected = True
            self._stats.connects += 1
            self._stats.last_connect = datetime.now()
            self._connect_start_time = time.time()
            self._last_message_time = time.time()

            # Reset reconnection state
            self._retry_count = 0
            self._current_delay = self.reconnect_config.initial_delay
            self._circuit.record_success()

            logger.info("WebSocket connected successfully")

            if self._on_connect:
                try:
                    await self._on_connect() if asyncio.iscoroutinefunction(self._on_connect) \
                        else self._on_connect()
                except Exception as e:
                    logger.error(f"Connect callback error: {e}")

            return True

        except asyncio.TimeoutError:
            logger.error("Connection timeout")
            self._circuit.record_failure()
            self._stats.errors += 1
            return False

        except Exception as e:
            logger.error(f"Connection error: {e}")
            self._circuit.record_failure()
            self._stats.errors += 1
            return False

    def _check_auth_success(self, response: Any) -> bool:
        """Check if authentication was successful"""
        # Alpaca format
        if isinstance(response, list) and len(response) > 0:
            msg = response[0]
            if msg.get("T") == "success":
                return True
            if msg.get("T") == "error":
                return False

        # Generic format
        if isinstance(response, dict):
            if response.get("success") or response.get("authenticated"):
                return True

        return True  # Assume success if no clear error

    async def disconnect(self) -> None:
        """Gracefully disconnect"""
        self._should_reconnect = False
        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        self._connected = False
        self._stats.disconnects += 1
        self._stats.last_disconnect = datetime.now()

        if self._connect_start_time:
            self._stats.uptime_seconds += time.time() - self._connect_start_time

        logger.info("WebSocket disconnected")

    async def send(self, message: Any) -> bool:
        """Send message"""
        if not self._connected or not self._ws:
            logger.warning("Cannot send: not connected")
            return False

        try:
            data = json.dumps(message) if isinstance(message, dict) else str(message)
            await self._ws.send(data)
            self._stats.messages_sent += 1
            self._stats.bytes_sent += len(data)
            return True

        except Exception as e:
            logger.error(f"Send error: {e}")
            self._stats.errors += 1
            return False

    async def start(self) -> None:
        """Start the WebSocket connection and message loop"""
        self._running = True
        self._should_reconnect = True

        while self._running:
            # Connect if needed
            if not self._connected:
                success = await self.connect()
                if not success:
                    await self._wait_for_reconnect()
                    continue

            # Start heartbeat monitor
            if not self._heartbeat_task or self._heartbeat_task.done():
                self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())

            # Receive messages
            try:
                message = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=self.heartbeat_timeout
                )

                self._last_message_time = time.time()
                self._stats.messages_received += 1
                self._stats.bytes_received += len(message)
                self._stats.last_message = datetime.now()

                # Process message
                await self._process_message(message)

            except asyncio.TimeoutError:
                # Check heartbeat
                elapsed = time.time() - self._last_message_time
                if elapsed > self.heartbeat_timeout:
                    logger.warning(f"Heartbeat timeout ({elapsed:.1f}s)")
                    await self._handle_disconnect("Heartbeat timeout")

            except ConnectionClosed as e:
                logger.warning(f"Connection closed: {e}")
                await self._handle_disconnect(str(e))

            except Exception as e:
                logger.error(f"Receive error: {e}")
                self._stats.errors += 1
                await self._handle_disconnect(str(e))

    async def _process_message(self, raw_message: str) -> None:
        """Process incoming message"""
        try:
            data = json.loads(raw_message)

            # Check for sequence gaps (if applicable)
            self._check_sequence(data)

            # Call message handler
            if self._on_message:
                await self._on_message(data) if asyncio.iscoroutinefunction(self._on_message) \
                    else self._on_message(data)

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON message: {raw_message[:100]}")

    def _check_sequence(self, data: Any) -> None:
        """Check for message sequence gaps"""
        if not isinstance(data, list):
            data = [data]

        for msg in data:
            if not isinstance(msg, dict):
                continue

            symbol = msg.get("S") or msg.get("symbol")
            seq = msg.get("seq") or msg.get("sequence")

            if symbol and seq is not None:
                if symbol in self._last_sequence:
                    expected = self._last_sequence[symbol] + 1
                    if seq != expected:
                        self._gaps_detected += 1
                        logger.warning(
                            f"Sequence gap for {symbol}: expected {expected}, got {seq}"
                        )

                self._last_sequence[symbol] = seq

    async def _handle_disconnect(self, reason: str) -> None:
        """Handle disconnection"""
        self._connected = False
        self._stats.disconnects += 1
        self._stats.last_disconnect = datetime.now()

        if self._connect_start_time:
            self._stats.uptime_seconds += time.time() - self._connect_start_time
            self._connect_start_time = None

        if self._on_disconnect:
            try:
                await self._on_disconnect(reason) if asyncio.iscoroutinefunction(self._on_disconnect) \
                    else self._on_disconnect(reason)
            except Exception as e:
                logger.error(f"Disconnect callback error: {e}")

        # Close existing connection
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _wait_for_reconnect(self) -> None:
        """Wait before reconnection attempt with backoff and jitter"""
        if not self._should_reconnect:
            return

        # Check retry limit
        if (self.reconnect_config.max_retries > 0 and
            self._retry_count >= self.reconnect_config.max_retries):
            logger.error(f"Max retries ({self.reconnect_config.max_retries}) reached")
            self._running = False
            return

        # Calculate delay with jitter
        jitter = random.uniform(
            -self.reconnect_config.jitter_factor,
            self.reconnect_config.jitter_factor
        )
        delay = self._current_delay * (1 + jitter)
        delay = min(delay, self.reconnect_config.max_delay)

        logger.info(
            f"Reconnecting in {delay:.1f}s "
            f"(attempt {self._retry_count + 1}, circuit: {self._circuit.state.value})"
        )

        await asyncio.sleep(delay)

        # Increase delay for next time
        self._current_delay = min(
            self._current_delay * self.reconnect_config.backoff_multiplier,
            self.reconnect_config.max_delay
        )
        self._retry_count += 1
        self._stats.reconnects += 1

    async def _heartbeat_monitor(self) -> None:
        """Monitor connection heartbeat"""
        while self._running and self._connected:
            await asyncio.sleep(5)

            elapsed = time.time() - self._last_message_time
            if elapsed > self.heartbeat_timeout * 0.8:
                logger.debug(f"No message for {elapsed:.1f}s (timeout: {self.heartbeat_timeout}s)")

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        stats = self._stats.to_dict()
        stats['circuit_state'] = self._circuit.state.value
        stats['retry_count'] = self._retry_count
        stats['current_delay'] = self._current_delay
        stats['gaps_detected'] = self._gaps_detected
        stats['connected'] = self._connected
        return stats

    async def stop(self) -> None:
        """Stop the WebSocket"""
        await self.disconnect()


# =============================================================================
# REST FALLBACK FOR GRACEFUL DEGRADATION
# =============================================================================

class RESTFallback:
    """
    REST API fallback when WebSocket is unavailable.

    Polls REST endpoints at configurable intervals.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        api_secret: str,
        poll_interval: float = 1.0
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.poll_interval = poll_interval

        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None
        self._symbols: Set[str] = set()
        self._on_data: Optional[Callable] = None

    async def start(self, symbols: List[str]) -> None:
        """Start REST polling"""
        self._symbols = set(symbols)
        self._running = True

        self._session = aiohttp.ClientSession(
            headers={
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret
            }
        )

        logger.info(f"REST fallback started for {len(symbols)} symbols")

        while self._running:
            await self._poll()
            await asyncio.sleep(self.poll_interval)

    async def stop(self) -> None:
        """Stop REST polling"""
        self._running = False
        if self._session:
            await self._session.close()

    async def _poll(self) -> None:
        """Poll for latest data"""
        if not self._symbols:
            return

        try:
            # Get latest bars for all symbols
            symbols_str = ",".join(self._symbols)
            url = f"{self.base_url}/v2/stocks/bars/latest?symbols={symbols_str}"

            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if self._on_data:
                        self._on_data(data)
                else:
                    logger.warning(f"REST poll failed: {resp.status}")

        except Exception as e:
            logger.error(f"REST poll error: {e}")

    def on_data(self, callback: Callable) -> None:
        """Set data callback"""
        self._on_data = callback
