"""
Asynchronous Execution Pipeline
High-Performance Concurrent Trading System

Features:
- Async data ingestion from multiple sources
- Parallel feature computation
- Non-blocking order execution
- Event-driven architecture with message queues
- Graceful degradation and error recovery

Architecture:
    DataIngestion -> FeatureEngine -> SignalGenerator -> OrderManager
         |               |                |                  |
         v               v                v                  v
    [async queue]  [process pool]  [async queue]     [async execution]

Usage:
    pipeline = AsyncTradingPipeline(config)
    await pipeline.start()
    await pipeline.run_trading_loop()
    await pipeline.shutdown()
"""

import asyncio
import queue
from typing import Dict, List, Optional, Any, Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
import traceback
import signal
import warnings

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


class PipelineState(Enum):
    """Pipeline execution states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class MessageType(Enum):
    """Internal message types for async communication"""
    MARKET_DATA = "market_data"
    FEATURE_UPDATE = "feature_update"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"
    ERROR = "error"
    SHUTDOWN = "shutdown"
    HEARTBEAT = "heartbeat"


@dataclass
class PipelineMessage:
    """Message for inter-component communication"""
    msg_type: MessageType
    timestamp: datetime
    symbol: str
    payload: Any
    priority: int = 5  # 1 = highest, 10 = lowest
    correlation_id: Optional[str] = None

    def __lt__(self, other):
        """For priority queue ordering"""
        return self.priority < other.priority


@dataclass
class PipelineConfig:
    """Configuration for async pipeline"""
    # Threading/processing
    max_workers: int = 4
    use_process_pool: bool = False  # Use processes vs threads for CPU work

    # Queue settings
    max_queue_size: int = 10000
    queue_timeout: float = 1.0

    # Timing
    data_poll_interval: float = 0.1  # seconds
    feature_compute_interval: float = 1.0
    signal_check_interval: float = 1.0
    heartbeat_interval: float = 5.0

    # Error handling
    max_consecutive_errors: int = 10
    error_backoff_base: float = 1.0
    error_backoff_max: float = 60.0

    # Performance
    batch_size: int = 100
    enable_profiling: bool = False


@dataclass
class PipelineMetrics:
    """Runtime metrics for monitoring"""
    messages_processed: int = 0
    errors_count: int = 0
    avg_latency_ms: float = 0.0
    queue_depth: int = 0
    active_tasks: int = 0
    last_heartbeat: Optional[datetime] = None

    # Per-component metrics
    data_ingestion_count: int = 0
    features_computed: int = 0
    signals_generated: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0


class AsyncQueue:
    """
    Thread-safe async queue with priority support.

    Bridges sync producers with async consumers.
    """

    def __init__(self, maxsize: int = 10000):
        self._queue: asyncio.Queue = None
        self._maxsize = maxsize
        self._loop: asyncio.AbstractEventLoop = None

    async def initialize(self):
        """Initialize queue with event loop."""
        self._loop = asyncio.get_event_loop()
        self._queue = asyncio.PriorityQueue(maxsize=self._maxsize)

    async def put(self, item: PipelineMessage, timeout: float = None):
        """Put item in queue with optional timeout."""
        if timeout:
            await asyncio.wait_for(
                self._queue.put((item.priority, item)),
                timeout=timeout
            )
        else:
            await self._queue.put((item.priority, item))

    def put_nowait(self, item: PipelineMessage):
        """Non-blocking put (for sync callers)."""
        self._queue.put_nowait((item.priority, item))

    async def get(self, timeout: float = None) -> Optional[PipelineMessage]:
        """Get item from queue."""
        try:
            if timeout:
                priority, item = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=timeout
                )
            else:
                priority, item = await self._queue.get()
            return item
        except asyncio.TimeoutError:
            return None

    def qsize(self) -> int:
        """Get queue size."""
        return self._queue.qsize() if self._queue else 0

    def empty(self) -> bool:
        """Check if empty."""
        return self._queue.empty() if self._queue else True


class DataIngestionWorker:
    """
    Async worker for data ingestion from multiple sources.

    Supports:
    - REST API polling
    - WebSocket streaming
    - Database queries
    - File watching
    """

    def __init__(
        self,
        output_queue: AsyncQueue,
        config: PipelineConfig,
        symbols: List[str]
    ):
        self.output_queue = output_queue
        self.config = config
        self.symbols = symbols

        self._running = False
        self._websocket_connections: Dict[str, Any] = {}
        self._last_data: Dict[str, pd.DataFrame] = {}

    async def start(self):
        """Start data ingestion."""
        self._running = True
        logger.info(f"DataIngestionWorker started for {len(self.symbols)} symbols")

    async def stop(self):
        """Stop data ingestion."""
        self._running = False

        # Close websocket connections
        for symbol, ws in self._websocket_connections.items():
            try:
                await ws.close()
            except Exception:
                pass

        logger.info("DataIngestionWorker stopped")

    async def poll_data(self, data_source: Callable) -> None:
        """
        Poll data source at regular intervals.

        Args:
            data_source: Callable that returns market data
        """
        while self._running:
            try:
                for symbol in self.symbols:
                    # Get data (this should be the actual data fetch)
                    data = await asyncio.to_thread(data_source, symbol)

                    if data is not None:
                        msg = PipelineMessage(
                            msg_type=MessageType.MARKET_DATA,
                            timestamp=datetime.now(),
                            symbol=symbol,
                            payload=data,
                            priority=1  # Market data is high priority
                        )
                        await self.output_queue.put(msg)

                await asyncio.sleep(self.config.data_poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Data ingestion error: {e}")
                await asyncio.sleep(1.0)

    async def stream_websocket(
        self,
        url: str,
        symbol: str,
        message_handler: Callable
    ) -> None:
        """
        Stream data via WebSocket.

        Args:
            url: WebSocket URL
            symbol: Symbol to subscribe
            message_handler: Function to parse messages
        """
        try:
            import websockets
        except ImportError:
            logger.warning("websockets not installed, falling back to polling")
            return

        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    self._websocket_connections[symbol] = ws

                    # Subscribe to symbol
                    await ws.send(f'{{"action": "subscribe", "symbol": "{symbol}"}}')

                    async for message in ws:
                        if not self._running:
                            break

                        data = message_handler(message)

                        if data is not None:
                            msg = PipelineMessage(
                                msg_type=MessageType.MARKET_DATA,
                                timestamp=datetime.now(),
                                symbol=symbol,
                                payload=data,
                                priority=1
                            )
                            await self.output_queue.put(msg)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket error for {symbol}: {e}")
                await asyncio.sleep(5.0)  # Reconnect delay


class FeatureComputeWorker:
    """
    Async worker for parallel feature computation.

    Uses process pool for CPU-bound calculations.
    """

    def __init__(
        self,
        input_queue: AsyncQueue,
        output_queue: AsyncQueue,
        config: PipelineConfig,
        feature_builder: Any  # FeatureBuilder instance
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.config = config
        self.feature_builder = feature_builder

        self._running = False
        self._executor: Optional[ProcessPoolExecutor] = None
        self._data_buffer: Dict[str, pd.DataFrame] = {}

    async def start(self):
        """Start feature computation worker."""
        self._running = True

        if self.config.use_process_pool:
            self._executor = ProcessPoolExecutor(max_workers=self.config.max_workers)
        else:
            self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

        logger.info("FeatureComputeWorker started")

    async def stop(self):
        """Stop feature computation worker."""
        self._running = False

        if self._executor:
            self._executor.shutdown(wait=True)

        logger.info("FeatureComputeWorker stopped")

    async def run(self):
        """Main processing loop."""
        while self._running:
            try:
                msg = await self.input_queue.get(timeout=self.config.queue_timeout)

                if msg is None:
                    continue

                if msg.msg_type == MessageType.SHUTDOWN:
                    break

                if msg.msg_type == MessageType.MARKET_DATA:
                    # Buffer data
                    self._data_buffer[msg.symbol] = msg.payload

                    # Compute features in executor
                    loop = asyncio.get_event_loop()
                    features = await loop.run_in_executor(
                        self._executor,
                        self._compute_features,
                        msg.symbol,
                        msg.payload
                    )

                    if features is not None:
                        out_msg = PipelineMessage(
                            msg_type=MessageType.FEATURE_UPDATE,
                            timestamp=datetime.now(),
                            symbol=msg.symbol,
                            payload=features,
                            priority=2,
                            correlation_id=msg.correlation_id
                        )
                        await self.output_queue.put(out_msg)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Feature computation error: {e}\n{traceback.format_exc()}")

    def _compute_features(self, symbol: str, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Compute features for a symbol (runs in executor).

        Args:
            symbol: Symbol name
            data: OHLCV data

        Returns:
            Feature DataFrame
        """
        try:
            if self.feature_builder is not None:
                return self.feature_builder.build_features(data)
            else:
                # Simple placeholder features
                return data
        except Exception as e:
            logger.error(f"Feature computation failed for {symbol}: {e}")
            return None


class SignalGeneratorWorker:
    """
    Async worker for signal generation.
    """

    def __init__(
        self,
        input_queue: AsyncQueue,
        output_queue: AsyncQueue,
        config: PipelineConfig,
        strategy: Any  # Strategy instance
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.config = config
        self.strategy = strategy

        self._running = False
        self._feature_cache: Dict[str, pd.DataFrame] = {}

    async def start(self):
        """Start signal generator."""
        self._running = True
        logger.info("SignalGeneratorWorker started")

    async def stop(self):
        """Stop signal generator."""
        self._running = False
        logger.info("SignalGeneratorWorker stopped")

    async def run(self):
        """Main processing loop."""
        while self._running:
            try:
                msg = await self.input_queue.get(timeout=self.config.queue_timeout)

                if msg is None:
                    continue

                if msg.msg_type == MessageType.SHUTDOWN:
                    break

                if msg.msg_type == MessageType.FEATURE_UPDATE:
                    # Cache features
                    self._feature_cache[msg.symbol] = msg.payload

                    # Generate signals
                    signals = await self._generate_signals(msg.symbol)

                    for signal in signals:
                        out_msg = PipelineMessage(
                            msg_type=MessageType.SIGNAL,
                            timestamp=datetime.now(),
                            symbol=msg.symbol,
                            payload=signal,
                            priority=3,
                            correlation_id=msg.correlation_id
                        )
                        await self.output_queue.put(out_msg)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Signal generation error: {e}")

    async def _generate_signals(self, symbol: str) -> List[Any]:
        """
        Generate signals for a symbol.

        Args:
            symbol: Symbol name

        Returns:
            List of signals
        """
        try:
            if self.strategy is not None:
                data = {symbol: self._feature_cache.get(symbol)}
                signals = self.strategy.generate_signals(data)
                return list(signals.values())
            return []
        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}: {e}")
            return []


class OrderExecutionWorker:
    """
    Async worker for order execution.

    Handles:
    - Order submission
    - Fill monitoring
    - Position updates
    - Risk checks
    """

    def __init__(
        self,
        input_queue: AsyncQueue,
        config: PipelineConfig,
        order_manager: Any,  # OrderManager instance
        risk_manager: Any  # RiskManager instance
    ):
        self.input_queue = input_queue
        self.config = config
        self.order_manager = order_manager
        self.risk_manager = risk_manager

        self._running = False
        self._pending_orders: Dict[str, Any] = {}

    async def start(self):
        """Start order execution worker."""
        self._running = True
        logger.info("OrderExecutionWorker started")

    async def stop(self):
        """Stop order execution worker."""
        self._running = False

        # Cancel pending orders
        for order_id in list(self._pending_orders.keys()):
            try:
                await self._cancel_order(order_id)
            except Exception:
                pass

        logger.info("OrderExecutionWorker stopped")

    async def run(self):
        """Main processing loop."""
        while self._running:
            try:
                msg = await self.input_queue.get(timeout=self.config.queue_timeout)

                if msg is None:
                    continue

                if msg.msg_type == MessageType.SHUTDOWN:
                    break

                if msg.msg_type == MessageType.SIGNAL:
                    await self._process_signal(msg)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Order execution error: {e}")

    async def _process_signal(self, msg: PipelineMessage):
        """
        Process a trading signal.

        Args:
            msg: Signal message
        """
        signal = msg.payload
        symbol = msg.symbol

        # Risk check
        if self.risk_manager:
            risk_check = self.risk_manager.pre_trade_check(
                order_id=str(id(signal)),
                symbol=symbol,
                side='buy' if getattr(signal, 'signal_type', None) == 1 else 'sell',
                quantity=100,  # Placeholder
                price=100.0  # Placeholder
            )

            if not risk_check.passed:
                logger.warning(f"Signal rejected by risk manager: {risk_check.checks_failed}")
                return

        # Submit order
        await self._submit_order(signal, symbol)

    async def _submit_order(self, signal: Any, symbol: str):
        """
        Submit order for execution.

        Args:
            signal: Trading signal
            symbol: Symbol
        """
        if self.order_manager:
            try:
                # This would call the actual order manager
                order = await asyncio.to_thread(
                    self.order_manager.submit_order,
                    symbol=symbol,
                    side='buy' if getattr(signal, 'signal_type', None) == 1 else 'sell',
                    quantity=100,  # Calculate from position sizer
                    order_type='market'
                )

                if order:
                    self._pending_orders[order.order_id] = order
                    logger.info(f"Order submitted: {order.order_id} for {symbol}")

            except Exception as e:
                logger.error(f"Order submission failed: {e}")

    async def _cancel_order(self, order_id: str):
        """Cancel a pending order."""
        if self.order_manager and order_id in self._pending_orders:
            try:
                await asyncio.to_thread(
                    self.order_manager.cancel_order,
                    order_id
                )
                del self._pending_orders[order_id]
            except Exception as e:
                logger.error(f"Order cancellation failed: {e}")


class AsyncTradingPipeline:
    """
    Main async trading pipeline orchestrator.

    Coordinates all workers and manages lifecycle.
    """

    def __init__(
        self,
        config: PipelineConfig,
        symbols: List[str],
        feature_builder: Any = None,
        strategy: Any = None,
        order_manager: Any = None,
        risk_manager: Any = None
    ):
        self.config = config
        self.symbols = symbols

        # State
        self._state = PipelineState.STOPPED
        self._metrics = PipelineMetrics()

        # Queues
        self._data_queue = AsyncQueue(config.max_queue_size)
        self._feature_queue = AsyncQueue(config.max_queue_size)
        self._signal_queue = AsyncQueue(config.max_queue_size)

        # Workers
        self._data_worker = DataIngestionWorker(
            self._data_queue, config, symbols
        )
        self._feature_worker = FeatureComputeWorker(
            self._data_queue, self._feature_queue, config, feature_builder
        )
        self._signal_worker = SignalGeneratorWorker(
            self._feature_queue, self._signal_queue, config, strategy
        )
        self._order_worker = OrderExecutionWorker(
            self._signal_queue, config, order_manager, risk_manager
        )

        # Tasks
        self._tasks: List[asyncio.Task] = []

        # Shutdown event
        self._shutdown_event = asyncio.Event()

    async def initialize(self):
        """Initialize all queues and connections."""
        await self._data_queue.initialize()
        await self._feature_queue.initialize()
        await self._signal_queue.initialize()

        logger.info("Pipeline initialized")

    async def start(self):
        """Start the pipeline."""
        if self._state != PipelineState.STOPPED:
            logger.warning(f"Cannot start pipeline in state {self._state}")
            return

        self._state = PipelineState.STARTING

        try:
            await self.initialize()

            # Start workers
            await self._data_worker.start()
            await self._feature_worker.start()
            await self._signal_worker.start()
            await self._order_worker.start()

            self._state = PipelineState.RUNNING
            logger.info("Pipeline started")

        except Exception as e:
            self._state = PipelineState.ERROR
            logger.error(f"Pipeline start failed: {e}")
            raise

    async def run_trading_loop(
        self,
        data_source: Optional[Callable] = None,
        duration: Optional[float] = None
    ):
        """
        Run the main trading loop.

        Args:
            data_source: Function to fetch market data
            duration: Optional max duration in seconds
        """
        if self._state != PipelineState.RUNNING:
            raise RuntimeError(f"Pipeline not running (state: {self._state})")

        start_time = datetime.now()

        # Create worker tasks
        tasks = [
            asyncio.create_task(self._feature_worker.run()),
            asyncio.create_task(self._signal_worker.run()),
            asyncio.create_task(self._order_worker.run()),
            asyncio.create_task(self._heartbeat_loop()),
        ]

        if data_source:
            tasks.append(
                asyncio.create_task(self._data_worker.poll_data(data_source))
            )

        self._tasks = tasks

        try:
            if duration:
                # Run for specified duration
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=duration
                )
            else:
                # Run until shutdown
                await self._shutdown_event.wait()

        except asyncio.TimeoutError:
            logger.info("Trading loop duration reached")

        finally:
            await self.shutdown()

    async def shutdown(self):
        """Gracefully shutdown the pipeline."""
        if self._state == PipelineState.STOPPED:
            return

        self._state = PipelineState.STOPPING
        logger.info("Pipeline shutting down...")

        # Signal shutdown
        self._shutdown_event.set()

        # Send shutdown messages
        shutdown_msg = PipelineMessage(
            msg_type=MessageType.SHUTDOWN,
            timestamp=datetime.now(),
            symbol="",
            payload=None,
            priority=0
        )

        try:
            await self._data_queue.put(shutdown_msg, timeout=1.0)
            await self._feature_queue.put(shutdown_msg, timeout=1.0)
            await self._signal_queue.put(shutdown_msg, timeout=1.0)
        except asyncio.TimeoutError:
            pass

        # Cancel tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Stop workers
        await self._data_worker.stop()
        await self._feature_worker.stop()
        await self._signal_worker.stop()
        await self._order_worker.stop()

        self._state = PipelineState.STOPPED
        logger.info("Pipeline shutdown complete")

    async def _heartbeat_loop(self):
        """Send periodic heartbeat messages."""
        while self._state == PipelineState.RUNNING:
            try:
                self._metrics.last_heartbeat = datetime.now()
                self._metrics.queue_depth = (
                    self._data_queue.qsize() +
                    self._feature_queue.qsize() +
                    self._signal_queue.qsize()
                )

                await asyncio.sleep(self.config.heartbeat_interval)

            except asyncio.CancelledError:
                break

    def get_metrics(self) -> PipelineMetrics:
        """Get current pipeline metrics."""
        return self._metrics

    def get_state(self) -> PipelineState:
        """Get current pipeline state."""
        return self._state

    def request_shutdown(self):
        """Request graceful shutdown from another thread."""
        self._shutdown_event.set()


class PipelineBuilder:
    """
    Builder for constructing trading pipelines.

    Provides fluent API for configuration.
    """

    def __init__(self):
        self._config = PipelineConfig()
        self._symbols: List[str] = []
        self._feature_builder = None
        self._strategy = None
        self._order_manager = None
        self._risk_manager = None

    def with_symbols(self, symbols: List[str]) -> 'PipelineBuilder':
        """Set symbols to trade."""
        self._symbols = symbols
        return self

    def with_feature_builder(self, feature_builder: Any) -> 'PipelineBuilder':
        """Set feature builder."""
        self._feature_builder = feature_builder
        return self

    def with_strategy(self, strategy: Any) -> 'PipelineBuilder':
        """Set trading strategy."""
        self._strategy = strategy
        return self

    def with_order_manager(self, order_manager: Any) -> 'PipelineBuilder':
        """Set order manager."""
        self._order_manager = order_manager
        return self

    def with_risk_manager(self, risk_manager: Any) -> 'PipelineBuilder':
        """Set risk manager."""
        self._risk_manager = risk_manager
        return self

    def with_config(self, **kwargs) -> 'PipelineBuilder':
        """Update configuration."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        return self

    def build(self) -> AsyncTradingPipeline:
        """Build the pipeline."""
        if not self._symbols:
            raise ValueError("No symbols configured")

        return AsyncTradingPipeline(
            config=self._config,
            symbols=self._symbols,
            feature_builder=self._feature_builder,
            strategy=self._strategy,
            order_manager=self._order_manager,
            risk_manager=self._risk_manager
        )


async def run_async_pipeline(
    symbols: List[str],
    data_source: Callable,
    feature_builder: Any = None,
    strategy: Any = None,
    order_manager: Any = None,
    risk_manager: Any = None,
    duration: float = None
):
    """
    Convenience function to run an async trading pipeline.

    Args:
        symbols: List of symbols to trade
        data_source: Function to fetch market data
        feature_builder: Optional feature builder
        strategy: Optional strategy
        order_manager: Optional order manager
        risk_manager: Optional risk manager
        duration: Optional duration in seconds
    """
    pipeline = (
        PipelineBuilder()
        .with_symbols(symbols)
        .with_feature_builder(feature_builder)
        .with_strategy(strategy)
        .with_order_manager(order_manager)
        .with_risk_manager(risk_manager)
        .build()
    )

    # Handle shutdown signals
    def signal_handler():
        pipeline.request_shutdown()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    await pipeline.start()
    await pipeline.run_trading_loop(data_source, duration)


if __name__ == "__main__":
    # Example usage
    async def mock_data_source(symbol: str) -> pd.DataFrame:
        """Mock data source for testing."""
        return pd.DataFrame({
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.5],
            'volume': [1000000]
        })

    asyncio.run(run_async_pipeline(
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        data_source=mock_data_source,
        duration=10.0
    ))
