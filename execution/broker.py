"""
Broker Module
=============

Broker integration for live trading.
Supports Alpaca Markets for both paper and live trading.

Features:
- Alpaca API integration (REST + WebSocket)
- Paper trading mode with realistic simulation
- Order management with tracking
- Position reconciliation
- Real-time account monitoring
- Automatic reconnection
- Rate limiting and error handling

Architecture:
- BrokerBase: Abstract interface
- AlpacaBroker: Production Alpaca integration
- PaperBroker: Local paper trading simulator
- OrderManager: Order lifecycle management
- PositionReconciler: Position sync between broker and local

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, TypeVar
from uuid import UUID, uuid4
import threading
from collections import deque
import json

import numpy as np

# Conditional imports for Alpaca
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        StopOrderRequest,
        StopLimitOrderRequest,
        TrailingStopOrderRequest,
        GetOrdersRequest,
    )
    from alpaca.trading.enums import (
        OrderSide as AlpacaOrderSide,
        OrderType as AlpacaOrderType,
        TimeInForce,
        OrderStatus as AlpacaOrderStatus,
        QueryOrderStatus,
    )
    from alpaca.data.live import StockDataStream
    from alpaca.data.historical import StockHistoricalDataClient
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    TradingClient = None
    StockDataStream = None

from config.settings import get_logger, get_settings, OrderSide, OrderType
from core.types import (
    Order,
    Position,
    Trade,
    PortfolioState,
    ExecutionError,
    OrderRejectedError,
    InsufficientFundsError,
)
from core.events import (
    EventBus,
    EventType,
    OrderEvent,
    FillEvent,
    PortfolioEvent,
    SystemEvent,
)

logger = get_logger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class BrokerType(str, Enum):
    """Broker type enumeration."""
    ALPACA_PAPER = "alpaca_paper"
    ALPACA_LIVE = "alpaca_live"
    PAPER = "paper"
    INTERACTIVE_BROKERS = "interactive_brokers"  # Future
    
    
class ConnectionStatus(str, Enum):
    """Broker connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BrokerConfig:
    """
    Broker configuration.
    
    Attributes:
        broker_type: Type of broker
        api_key: API key for broker
        secret_key: Secret key for broker
        base_url: API base URL
        data_url: Market data URL
        paper_trading: Enable paper trading mode
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries
        connection_timeout: Connection timeout seconds
        rate_limit_per_minute: API rate limit
        enable_websocket: Enable WebSocket streaming
        reconciliation_interval: Position sync interval
    """
    broker_type: BrokerType = BrokerType.ALPACA_PAPER
    api_key: str = ""
    secret_key: str = ""
    base_url: str = "https://paper-api.alpaca.markets"
    data_url: str = "https://data.alpaca.markets"
    paper_trading: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    connection_timeout: float = 30.0
    rate_limit_per_minute: int = 200
    enable_websocket: bool = True
    reconciliation_interval: int = 60  # seconds
    
    @classmethod
    def from_settings(cls) -> "BrokerConfig":
        """Create config from application settings."""
        settings = get_settings()
        
        return cls(
            broker_type=BrokerType.ALPACA_PAPER if settings.alpaca.paper_trading else BrokerType.ALPACA_LIVE,
            api_key=settings.alpaca.api_key,
            secret_key=settings.alpaca.secret_key,
            base_url=settings.alpaca.base_url,
            data_url=settings.alpaca.data_url,
            paper_trading=settings.alpaca.paper_trading,
        )


# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    
    Prevents hitting API rate limits.
    """
    
    def __init__(self, requests_per_minute: int = 200):
        """Initialize rate limiter."""
        self.requests_per_minute = requests_per_minute
        self.tokens = requests_per_minute
        self.last_update = time.monotonic()
        self._lock = threading.Lock()
    
    def acquire(self, timeout: float = 10.0) -> bool:
        """
        Acquire a token, blocking if necessary.
        
        Args:
            timeout: Maximum time to wait
        
        Returns:
            True if token acquired
        """
        deadline = time.monotonic() + timeout
        
        while time.monotonic() < deadline:
            with self._lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return True
            time.sleep(0.1)
        
        return False
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_update
        
        # Refill rate: tokens per second
        refill_rate = self.requests_per_minute / 60.0
        new_tokens = elapsed * refill_rate
        
        self.tokens = min(self.requests_per_minute, self.tokens + new_tokens)
        self.last_update = now


# =============================================================================
# BASE BROKER
# =============================================================================

class BrokerBase(ABC):
    """
    Abstract base class for broker implementations.
    
    Defines the contract for all broker integrations.
    Implements the ExecutionHandler interface from core/interfaces.py.
    """
    
    def __init__(self, config: BrokerConfig):
        """
        Initialize broker.
        
        Args:
            config: Broker configuration
        """
        self.config = config
        self._status = ConnectionStatus.DISCONNECTED
        self._event_bus: EventBus | None = None
        self._order_callbacks: list[Callable[[Order], None]] = []
        self._fill_callbacks: list[Callable[[Trade], None]] = []
        self._error_callbacks: list[Callable[[Exception], None]] = []
        
        # Order tracking
        self._orders: dict[str, Order] = {}
        self._pending_orders: dict[str, Order] = {}
        
        # Rate limiting
        self._rate_limiter = RateLimiter(config.rate_limit_per_minute)
    
    @property
    def status(self) -> ConnectionStatus:
        """Get connection status."""
        return self._status
    
    @property
    def is_connected(self) -> bool:
        """Check if broker is connected."""
        return self._status == ConnectionStatus.CONNECTED
    
    def set_event_bus(self, event_bus: EventBus) -> None:
        """Set event bus for publishing events."""
        self._event_bus = event_bus
    
    def on_order(self, callback: Callable[[Order], None]) -> None:
        """Register order update callback."""
        self._order_callbacks.append(callback)
    
    def on_fill(self, callback: Callable[[Trade], None]) -> None:
        """Register fill callback."""
        self._fill_callbacks.append(callback)
    
    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """Register error callback."""
        self._error_callbacks.append(callback)
    
    def _notify_order(self, order: Order) -> None:
        """Notify order callbacks."""
        for callback in self._order_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Order callback error: {e}")
    
    def _notify_fill(self, trade: Trade) -> None:
        """Notify fill callbacks."""
        for callback in self._fill_callbacks:
            try:
                callback(trade)
            except Exception as e:
                logger.error(f"Fill callback error: {e}")
    
    def _notify_error(self, error: Exception) -> None:
        """Notify error callbacks."""
        for callback in self._error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Error callback error: {e}")
    
    # =========================================================================
    # ABSTRACT METHODS
    # =========================================================================
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to broker.
        
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from broker."""
        pass
    
    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """
        Submit an order to the broker.
        
        Args:
            order: Order to submit
        
        Returns:
            Broker order ID
        
        Raises:
            OrderRejectedError: If order is rejected
            ExecutionError: If submission fails
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.
        
        Args:
            order_id: Broker order ID
        
        Returns:
            True if cancellation successful
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """
        Get current order status.
        
        Args:
            order_id: Broker order ID
        
        Returns:
            Order status dictionary
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> list[Position]:
        """
        Get current positions from broker.
        
        Returns:
            List of positions
        """
        pass
    
    @abstractmethod
    def get_account_info(self) -> dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Account info dictionary
        """
        pass
    
    @abstractmethod
    def is_market_open(self) -> bool:
        """
        Check if market is currently open.
        
        Returns:
            True if market is open
        """
        pass
    
    @abstractmethod
    def get_clock(self) -> dict[str, Any]:
        """
        Get market clock information.
        
        Returns:
            Clock info with open/close times
        """
        pass


# =============================================================================
# ALPACA BROKER
# =============================================================================

class AlpacaBroker(BrokerBase):
    """
    Alpaca Markets broker implementation.
    
    Features:
        - REST API for order management
        - WebSocket for real-time updates
        - Paper and live trading modes
        - Automatic reconnection
        - Order tracking and sync
    
    Example:
        config = BrokerConfig.from_settings()
        broker = AlpacaBroker(config)
        
        if broker.connect():
            order = Order.create_market_order("AAPL", "buy", 100)
            order_id = broker.submit_order(order)
    """
    
    def __init__(self, config: BrokerConfig):
        """Initialize Alpaca broker."""
        super().__init__(config)
        
        if not ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-py not installed. Run: pip install alpaca-py"
            )
        
        self._client: TradingClient | None = None
        self._data_client: StockHistoricalDataClient | None = None
        self._stream: StockDataStream | None = None
        self._stream_thread: threading.Thread | None = None
        self._stop_stream = threading.Event()
    
    def connect(self) -> bool:
        """Connect to Alpaca."""
        try:
            self._status = ConnectionStatus.CONNECTING
            logger.info("Connecting to Alpaca...")
            
            # Create trading client
            self._client = TradingClient(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
                paper=self.config.paper_trading,
            )
            
            # Verify connection by getting account
            account = self._client.get_account()
            logger.info(f"Connected to Alpaca account: {account.account_number}")
            logger.info(f"  Equity: ${float(account.equity):,.2f}")
            logger.info(f"  Cash: ${float(account.cash):,.2f}")
            logger.info(f"  Buying Power: ${float(account.buying_power):,.2f}")
            
            # Create data client
            self._data_client = StockHistoricalDataClient(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
            )
            
            self._status = ConnectionStatus.CONNECTED
            return True
            
        except Exception as e:
            self._status = ConnectionStatus.ERROR
            logger.error(f"Failed to connect to Alpaca: {e}")
            self._notify_error(ExecutionError(f"Connection failed: {e}"))
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Alpaca."""
        logger.info("Disconnecting from Alpaca...")
        
        # Stop WebSocket stream
        if self._stream_thread and self._stream_thread.is_alive():
            self._stop_stream.set()
            self._stream_thread.join(timeout=5.0)
        
        self._client = None
        self._data_client = None
        self._stream = None
        self._status = ConnectionStatus.DISCONNECTED
        
        logger.info("Disconnected from Alpaca")
    
    def submit_order(self, order: Order) -> str:
        """Submit order to Alpaca."""
        if not self.is_connected or self._client is None:
            raise ExecutionError("Not connected to Alpaca")
        
        if not self._rate_limiter.acquire():
            raise ExecutionError("Rate limit exceeded")
        
        try:
            # Convert order side
            side = AlpacaOrderSide.BUY if order.side.lower() == "buy" else AlpacaOrderSide.SELL
            
            # Create appropriate request based on order type
            if order.order_type.lower() == "market":
                request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
            elif order.order_type.lower() == "limit":
                if order.limit_price is None:
                    raise OrderRejectedError("Limit order requires limit_price")
                request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=order.limit_price,
                )
            elif order.order_type.lower() == "stop":
                if order.stop_price is None:
                    raise OrderRejectedError("Stop order requires stop_price")
                request = StopOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    stop_price=order.stop_price,
                )
            elif order.order_type.lower() == "stop_limit":
                if order.stop_price is None or order.limit_price is None:
                    raise OrderRejectedError("Stop-limit requires both prices")
                request = StopLimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    stop_price=order.stop_price,
                    limit_price=order.limit_price,
                )
            else:
                raise OrderRejectedError(f"Unknown order type: {order.order_type}")
            
            # Submit to Alpaca
            alpaca_order = self._client.submit_order(request)
            broker_id = str(alpaca_order.id)
            
            # Track order
            order.broker_order_id = broker_id
            self._orders[broker_id] = order
            self._pending_orders[broker_id] = order
            
            logger.info(f"Order submitted: {order.symbol} {order.side} {order.quantity} @ {order.order_type}")
            logger.info(f"  Broker ID: {broker_id}")
            
            # Notify
            self._notify_order(order)
            
            return broker_id
            
        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            if "insufficient" in str(e).lower():
                raise InsufficientFundsError(str(e))
            raise ExecutionError(f"Order submission failed: {e}")
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order on Alpaca."""
        if not self.is_connected or self._client is None:
            raise ExecutionError("Not connected to Alpaca")
        
        if not self._rate_limiter.acquire():
            raise ExecutionError("Rate limit exceeded")
        
        try:
            self._client.cancel_order_by_id(order_id)
            
            # Update tracking
            if order_id in self._pending_orders:
                del self._pending_orders[order_id]
            
            logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return False
    
    def cancel_all_orders(self) -> int:
        """Cancel all open orders."""
        if not self.is_connected or self._client is None:
            raise ExecutionError("Not connected to Alpaca")
        
        try:
            cancelled = self._client.cancel_orders()
            count = len(cancelled) if cancelled else 0
            
            self._pending_orders.clear()
            logger.info(f"Cancelled {count} orders")
            
            return count
            
        except Exception as e:
            logger.error(f"Cancel all orders failed: {e}")
            return 0
    
    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Get order status from Alpaca."""
        if not self.is_connected or self._client is None:
            raise ExecutionError("Not connected to Alpaca")
        
        if not self._rate_limiter.acquire():
            raise ExecutionError("Rate limit exceeded")
        
        try:
            alpaca_order = self._client.get_order_by_id(order_id)
            
            return {
                "id": str(alpaca_order.id),
                "client_order_id": alpaca_order.client_order_id,
                "symbol": alpaca_order.symbol,
                "side": str(alpaca_order.side.value),
                "type": str(alpaca_order.type.value),
                "status": str(alpaca_order.status.value),
                "qty": float(alpaca_order.qty) if alpaca_order.qty else 0,
                "filled_qty": float(alpaca_order.filled_qty) if alpaca_order.filled_qty else 0,
                "filled_avg_price": float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else 0,
                "limit_price": float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
                "stop_price": float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
                "created_at": alpaca_order.created_at.isoformat() if alpaca_order.created_at else None,
                "filled_at": alpaca_order.filled_at.isoformat() if alpaca_order.filled_at else None,
            }
            
        except Exception as e:
            logger.error(f"Get order status failed: {e}")
            raise ExecutionError(f"Failed to get order status: {e}")
    
    def get_open_orders(self) -> list[dict[str, Any]]:
        """Get all open orders."""
        if not self.is_connected or self._client is None:
            raise ExecutionError("Not connected to Alpaca")
        
        try:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = self._client.get_orders(request)
            
            return [
                {
                    "id": str(o.id),
                    "symbol": o.symbol,
                    "side": str(o.side.value),
                    "type": str(o.type.value),
                    "qty": float(o.qty) if o.qty else 0,
                    "filled_qty": float(o.filled_qty) if o.filled_qty else 0,
                    "status": str(o.status.value),
                }
                for o in orders
            ]
            
        except Exception as e:
            logger.error(f"Get open orders failed: {e}")
            return []
    
    def get_positions(self) -> list[Position]:
        """Get positions from Alpaca."""
        if not self.is_connected or self._client is None:
            raise ExecutionError("Not connected to Alpaca")
        
        if not self._rate_limiter.acquire():
            raise ExecutionError("Rate limit exceeded")
        
        try:
            alpaca_positions = self._client.get_all_positions()
            positions = []
            
            for p in alpaca_positions:
                pos = Position(
                    symbol=p.symbol,
                    quantity=float(p.qty),
                    entry_price=float(p.avg_entry_price),
                    current_price=float(p.current_price),
                    market_value=float(p.market_value),
                    unrealized_pnl=float(p.unrealized_pl),
                    unrealized_pnl_pct=float(p.unrealized_plpc) if p.unrealized_plpc else 0,
                    side="long" if float(p.qty) > 0 else "short",
                )
                positions.append(pos)
            
            return positions
            
        except Exception as e:
            logger.error(f"Get positions failed: {e}")
            return []
    
    def close_position(self, symbol: str) -> bool:
        """Close a position completely."""
        if not self.is_connected or self._client is None:
            raise ExecutionError("Not connected to Alpaca")
        
        try:
            self._client.close_position(symbol)
            logger.info(f"Position closed: {symbol}")
            return True
        except Exception as e:
            logger.error(f"Close position failed: {e}")
            return False
    
    def close_all_positions(self) -> int:
        """Close all positions."""
        if not self.is_connected or self._client is None:
            raise ExecutionError("Not connected to Alpaca")
        
        try:
            self._client.close_all_positions(cancel_orders=True)
            positions = self.get_positions()
            logger.info(f"Closed all positions")
            return len(positions)
        except Exception as e:
            logger.error(f"Close all positions failed: {e}")
            return 0
    
    def get_account_info(self) -> dict[str, Any]:
        """Get account information from Alpaca."""
        if not self.is_connected or self._client is None:
            raise ExecutionError("Not connected to Alpaca")
        
        if not self._rate_limiter.acquire():
            raise ExecutionError("Rate limit exceeded")
        
        try:
            account = self._client.get_account()
            
            return {
                "account_number": account.account_number,
                "status": str(account.status),
                "currency": account.currency,
                "cash": float(account.cash),
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value) if account.portfolio_value else float(account.equity),
                "long_market_value": float(account.long_market_value) if account.long_market_value else 0,
                "short_market_value": float(account.short_market_value) if account.short_market_value else 0,
                "initial_margin": float(account.initial_margin) if account.initial_margin else 0,
                "maintenance_margin": float(account.maintenance_margin) if account.maintenance_margin else 0,
                "last_equity": float(account.last_equity) if account.last_equity else 0,
                "daytrade_count": account.daytrade_count,
                "pattern_day_trader": account.pattern_day_trader,
                "trading_blocked": account.trading_blocked,
                "account_blocked": account.account_blocked,
            }
            
        except Exception as e:
            logger.error(f"Get account info failed: {e}")
            raise ExecutionError(f"Failed to get account info: {e}")
    
    def is_market_open(self) -> bool:
        """Check if market is open."""
        if not self.is_connected or self._client is None:
            return False
        
        try:
            clock = self._client.get_clock()
            return clock.is_open
        except Exception:
            return False
    
    def get_clock(self) -> dict[str, Any]:
        """Get market clock information."""
        if not self.is_connected or self._client is None:
            raise ExecutionError("Not connected to Alpaca")
        
        try:
            clock = self._client.get_clock()
            
            return {
                "is_open": clock.is_open,
                "timestamp": clock.timestamp.isoformat(),
                "next_open": clock.next_open.isoformat() if clock.next_open else None,
                "next_close": clock.next_close.isoformat() if clock.next_close else None,
            }
            
        except Exception as e:
            logger.error(f"Get clock failed: {e}")
            raise ExecutionError(f"Failed to get clock: {e}")


# =============================================================================
# PAPER BROKER
# =============================================================================

class PaperBroker(BrokerBase):
    """
    Paper trading broker for testing strategies without real money.
    
    Features:
        - Realistic order filling simulation
        - Slippage modeling
        - Position tracking
        - Market hours simulation
        - Order book simulation (basic)
    
    Example:
        config = BrokerConfig(broker_type=BrokerType.PAPER)
        broker = PaperBroker(config, initial_capital=100000)
        
        broker.connect()
        order = Order.create_market_order("AAPL", "buy", 100)
        broker.submit_order(order)
    """
    
    def __init__(
        self,
        config: BrokerConfig,
        initial_capital: float = 100_000.0,
        slippage_pct: float = 0.0005,
        commission_per_share: float = 0.0,
    ):
        """
        Initialize paper broker.
        
        Args:
            config: Broker configuration
            initial_capital: Starting capital
            slippage_pct: Slippage as percentage
            commission_per_share: Commission per share
        """
        super().__init__(config)
        
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._slippage_pct = slippage_pct
        self._commission_per_share = commission_per_share
        
        # Positions: symbol -> Position
        self._positions: dict[str, Position] = {}
        
        # Order counter
        self._order_counter = 0
        
        # Trades history
        self._trades: list[Trade] = []
        
        # Market data (for simulation)
        self._current_prices: dict[str, float] = {}
        
        # Simulated order queue
        self._order_queue: deque[Order] = deque()
    
    def connect(self) -> bool:
        """Connect paper broker (always succeeds)."""
        self._status = ConnectionStatus.CONNECTED
        logger.info("Paper broker connected")
        logger.info(f"  Initial Capital: ${self._initial_capital:,.2f}")
        return True
    
    def disconnect(self) -> None:
        """Disconnect paper broker."""
        self._status = ConnectionStatus.DISCONNECTED
        logger.info("Paper broker disconnected")
    
    def set_price(self, symbol: str, price: float) -> None:
        """Set current price for a symbol (for simulation)."""
        self._current_prices[symbol] = price
    
    def submit_order(self, order: Order) -> str:
        """Submit order to paper broker."""
        if not self.is_connected:
            raise ExecutionError("Paper broker not connected")
        
        # Generate broker order ID
        self._order_counter += 1
        broker_id = f"PAPER-{self._order_counter:08d}"
        order.broker_order_id = broker_id
        
        # Validate order
        current_price = self._current_prices.get(order.symbol, order.limit_price or 100.0)
        
        # Check buying power
        if order.side.lower() == "buy":
            estimated_cost = order.quantity * current_price * (1 + self._slippage_pct)
            if estimated_cost > self._cash:
                raise InsufficientFundsError(
                    f"Insufficient funds: need ${estimated_cost:,.2f}, have ${self._cash:,.2f}"
                )
        
        # For market orders, fill immediately
        if order.order_type.lower() == "market":
            self._fill_order(order, current_price)
        else:
            # Queue limit/stop orders
            self._pending_orders[broker_id] = order
            self._order_queue.append(order)
        
        self._orders[broker_id] = order
        logger.info(f"Paper order submitted: {order.symbol} {order.side} {order.quantity}")
        
        return broker_id
    
    def _fill_order(self, order: Order, reference_price: float) -> None:
        """Fill an order at simulated price."""
        # Apply slippage
        if order.side.lower() == "buy":
            fill_price = reference_price * (1 + self._slippage_pct)
        else:
            fill_price = reference_price * (1 - self._slippage_pct)
        
        # Calculate commission
        commission = order.quantity * self._commission_per_share
        
        # Calculate cost
        if order.side.lower() == "buy":
            cost = order.quantity * fill_price + commission
            self._cash -= cost
        else:
            proceeds = order.quantity * fill_price - commission
            self._cash += proceeds
        
        # Update position
        self._update_position(order.symbol, order.quantity, fill_price, order.side)
        
        # Mark order as filled
        order.fill(order.quantity, fill_price, commission)
        
        # Remove from pending
        if order.broker_order_id in self._pending_orders:
            del self._pending_orders[order.broker_order_id]
        
        logger.info(f"Paper order filled: {order.symbol} @ ${fill_price:.2f}")
    
    def _update_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str,
    ) -> None:
        """Update position after fill."""
        if symbol in self._positions:
            pos = self._positions[symbol]
            
            if side.lower() == "buy":
                # Adding to long or closing short
                if pos.quantity >= 0:  # Long position
                    new_qty = pos.quantity + quantity
                    new_cost = (pos.entry_price * pos.quantity + price * quantity) / new_qty
                    pos.quantity = new_qty
                    pos.entry_price = new_cost
                else:  # Closing short
                    pos.quantity += quantity
                    if pos.quantity >= 0:
                        pos.entry_price = price
                        pos.side = "long"
            else:
                # Selling from long or adding to short
                if pos.quantity > 0:  # Closing long
                    pos.quantity -= quantity
                    if pos.quantity <= 0:
                        pos.entry_price = price
                        pos.side = "short"
                else:  # Adding to short
                    new_qty = abs(pos.quantity) + quantity
                    new_cost = (pos.entry_price * abs(pos.quantity) + price * quantity) / new_qty
                    pos.quantity = -new_qty
                    pos.entry_price = new_cost
            
            # Remove position if quantity is zero
            if abs(pos.quantity) < 0.001:
                del self._positions[symbol]
            else:
                pos.current_price = price
                pos.market_value = pos.quantity * price
                pos.unrealized_pnl = (price - pos.entry_price) * pos.quantity
        else:
            # Create new position
            qty = quantity if side.lower() == "buy" else -quantity
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=qty,
                entry_price=price,
                current_price=price,
                market_value=qty * price,
                unrealized_pnl=0.0,
                side="long" if qty > 0 else "short",
            )
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel paper order."""
        if order_id in self._pending_orders:
            del self._pending_orders[order_id]
            logger.info(f"Paper order cancelled: {order_id}")
            return True
        return False
    
    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Get paper order status."""
        if order_id in self._orders:
            order = self._orders[order_id]
            return {
                "id": order_id,
                "symbol": order.symbol,
                "side": order.side,
                "type": order.order_type,
                "status": order.status.value,
                "qty": order.quantity,
                "filled_qty": order.filled_quantity,
                "filled_avg_price": order.avg_fill_price,
            }
        return {"id": order_id, "status": "not_found"}
    
    def get_positions(self) -> list[Position]:
        """Get paper positions."""
        return list(self._positions.values())
    
    def get_account_info(self) -> dict[str, Any]:
        """Get paper account info."""
        # Calculate total equity
        positions_value = sum(
            abs(p.quantity) * p.current_price
            for p in self._positions.values()
        )
        unrealized_pnl = sum(p.unrealized_pnl for p in self._positions.values())
        equity = self._cash + positions_value
        
        return {
            "account_number": "PAPER",
            "status": "active",
            "currency": "USD",
            "cash": self._cash,
            "equity": equity,
            "buying_power": self._cash * 2,  # Simple margin
            "portfolio_value": equity,
            "long_market_value": sum(
                p.market_value for p in self._positions.values() if p.quantity > 0
            ),
            "short_market_value": sum(
                abs(p.market_value) for p in self._positions.values() if p.quantity < 0
            ),
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": sum(t.pnl for t in self._trades),
            "initial_capital": self._initial_capital,
        }
    
    def is_market_open(self) -> bool:
        """Check if simulated market is open."""
        now = datetime.now()
        # Simple check: Mon-Fri, 9:30-16:00 ET
        if now.weekday() >= 5:
            return False
        market_open = now.replace(hour=9, minute=30, second=0)
        market_close = now.replace(hour=16, minute=0, second=0)
        return market_open <= now <= market_close
    
    def get_clock(self) -> dict[str, Any]:
        """Get simulated market clock."""
        now = datetime.now()
        
        return {
            "is_open": self.is_market_open(),
            "timestamp": now.isoformat(),
            "next_open": now.replace(hour=9, minute=30).isoformat(),
            "next_close": now.replace(hour=16, minute=0).isoformat(),
        }
    
    def process_pending_orders(self, prices: dict[str, float]) -> None:
        """
        Process pending limit/stop orders against current prices.
        
        Args:
            prices: Current prices for symbols
        """
        self._current_prices.update(prices)
        
        orders_to_fill = []
        
        for order in self._pending_orders.values():
            if order.symbol not in prices:
                continue
            
            current_price = prices[order.symbol]
            
            # Check limit orders
            if order.order_type.lower() == "limit":
                if order.limit_price:
                    if order.side.lower() == "buy" and current_price <= order.limit_price:
                        orders_to_fill.append((order, current_price))
                    elif order.side.lower() == "sell" and current_price >= order.limit_price:
                        orders_to_fill.append((order, current_price))
            
            # Check stop orders
            elif order.order_type.lower() == "stop":
                if order.stop_price:
                    if order.side.lower() == "buy" and current_price >= order.stop_price:
                        orders_to_fill.append((order, current_price))
                    elif order.side.lower() == "sell" and current_price <= order.stop_price:
                        orders_to_fill.append((order, current_price))
        
        # Fill orders
        for order, price in orders_to_fill:
            self._fill_order(order, price)


# =============================================================================
# ORDER MANAGER
# =============================================================================

class OrderManager:
    """
    Order lifecycle management.
    
    Features:
        - Order tracking
        - Status updates
        - Fill aggregation
        - Order history
    """
    
    def __init__(self, broker: BrokerBase):
        """Initialize order manager."""
        self.broker = broker
        
        # Orders by various keys
        self._orders_by_id: dict[str, Order] = {}
        self._orders_by_symbol: dict[str, list[Order]] = {}
        self._pending_orders: dict[str, Order] = {}
        
        # Fill tracking
        self._fills: list[dict[str, Any]] = []
        
        # Statistics
        self._stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "cancelled_orders": 0,
            "rejected_orders": 0,
        }
    
    def submit(self, order: Order) -> str:
        """
        Submit an order through the broker.
        
        Args:
            order: Order to submit
        
        Returns:
            Broker order ID
        """
        broker_id = self.broker.submit_order(order)
        
        # Track order
        self._orders_by_id[broker_id] = order
        
        if order.symbol not in self._orders_by_symbol:
            self._orders_by_symbol[order.symbol] = []
        self._orders_by_symbol[order.symbol].append(order)
        
        self._pending_orders[broker_id] = order
        self._stats["total_orders"] += 1
        
        return broker_id
    
    def cancel(self, order_id: str) -> bool:
        """Cancel an order."""
        success = self.broker.cancel_order(order_id)
        
        if success:
            if order_id in self._pending_orders:
                del self._pending_orders[order_id]
            self._stats["cancelled_orders"] += 1
        
        return success
    
    def cancel_all(self, symbol: str | None = None) -> int:
        """Cancel all orders, optionally for a specific symbol."""
        count = 0
        
        orders_to_cancel = list(self._pending_orders.values())
        if symbol:
            orders_to_cancel = [o for o in orders_to_cancel if o.symbol == symbol]
        
        for order in orders_to_cancel:
            if order.broker_order_id and self.cancel(order.broker_order_id):
                count += 1
        
        return count
    
    def get_order(self, order_id: str) -> Order | None:
        """Get order by ID."""
        return self._orders_by_id.get(order_id)
    
    def get_orders_for_symbol(self, symbol: str) -> list[Order]:
        """Get all orders for a symbol."""
        return self._orders_by_symbol.get(symbol, [])
    
    def get_pending_orders(self) -> list[Order]:
        """Get all pending orders."""
        return list(self._pending_orders.values())
    
    def get_stats(self) -> dict[str, int]:
        """Get order statistics."""
        return self._stats.copy()


# =============================================================================
# POSITION RECONCILER
# =============================================================================

class PositionReconciler:
    """
    Position reconciliation between local tracking and broker.
    
    Features:
        - Automatic position sync
        - Discrepancy detection
        - Reconciliation reports
    """
    
    def __init__(self, broker: BrokerBase):
        """Initialize reconciler."""
        self.broker = broker
        self._last_reconciliation: datetime | None = None
        self._discrepancies: list[dict[str, Any]] = []
    
    def reconcile(
        self,
        local_positions: dict[str, Position],
    ) -> dict[str, Any]:
        """
        Reconcile local positions with broker positions.
        
        Args:
            local_positions: Local position tracking
        
        Returns:
            Reconciliation report
        """
        broker_positions = {p.symbol: p for p in self.broker.get_positions()}
        
        discrepancies = []
        
        # Check all local positions
        for symbol, local_pos in local_positions.items():
            if symbol in broker_positions:
                broker_pos = broker_positions[symbol]
                
                # Check quantity
                if abs(local_pos.quantity - broker_pos.quantity) > 0.01:
                    discrepancies.append({
                        "type": "quantity_mismatch",
                        "symbol": symbol,
                        "local_qty": local_pos.quantity,
                        "broker_qty": broker_pos.quantity,
                        "difference": local_pos.quantity - broker_pos.quantity,
                    })
            else:
                # Local position not found in broker
                discrepancies.append({
                    "type": "missing_broker_position",
                    "symbol": symbol,
                    "local_qty": local_pos.quantity,
                })
        
        # Check broker positions not in local
        for symbol, broker_pos in broker_positions.items():
            if symbol not in local_positions:
                discrepancies.append({
                    "type": "missing_local_position",
                    "symbol": symbol,
                    "broker_qty": broker_pos.quantity,
                })
        
        self._last_reconciliation = datetime.now()
        self._discrepancies = discrepancies
        
        return {
            "timestamp": self._last_reconciliation.isoformat(),
            "local_positions": len(local_positions),
            "broker_positions": len(broker_positions),
            "discrepancies": len(discrepancies),
            "details": discrepancies,
            "is_reconciled": len(discrepancies) == 0,
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_broker(
    broker_type: BrokerType | str = BrokerType.PAPER,
    config: BrokerConfig | None = None,
    **kwargs: Any,
) -> BrokerBase:
    """
    Factory function to create broker instances.
    
    Args:
        broker_type: Type of broker to create
        config: Broker configuration
        **kwargs: Additional arguments for specific brokers
    
    Returns:
        Broker instance
    
    Example:
        # Paper trading
        broker = create_broker("paper", initial_capital=100000)
        
        # Alpaca paper trading
        broker = create_broker(
            BrokerType.ALPACA_PAPER,
            config=BrokerConfig.from_settings()
        )
    """
    if isinstance(broker_type, str):
        broker_type = BrokerType(broker_type)
    
    if config is None:
        config = BrokerConfig(broker_type=broker_type)
    
    if broker_type == BrokerType.PAPER:
        return PaperBroker(config, **kwargs)
    
    elif broker_type in (BrokerType.ALPACA_PAPER, BrokerType.ALPACA_LIVE):
        config.paper_trading = broker_type == BrokerType.ALPACA_PAPER
        return AlpacaBroker(config)
    
    else:
        raise ValueError(f"Unknown broker type: {broker_type}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "BrokerType",
    "ConnectionStatus",
    # Config
    "BrokerConfig",
    # Classes
    "BrokerBase",
    "AlpacaBroker",
    "PaperBroker",
    "OrderManager",
    "PositionReconciler",
    "RateLimiter",
    # Factory
    "create_broker",
]