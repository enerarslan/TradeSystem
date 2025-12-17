"""
Custom logging configuration using loguru.

This module provides a centralized logging setup with:
- Console and file handlers
- Structured logging support
- Performance logging
- Trade-specific logging
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger

from config.settings import PROJECT_ROOT, settings


def setup_logging(
    level: str = "INFO",
    log_file: Path | str | None = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
    serialize: bool = False,
    console: bool = True,
) -> None:
    """
    Configure the global logger with specified settings.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None to disable file logging)
        rotation: When to rotate the log file
        retention: How long to keep old log files
        serialize: Whether to output JSON logs
        console: Whether to log to console
    """
    # Remove default handler
    logger.remove()

    # Console handler
    if console:
        logger.add(
            sys.stderr,
            level=level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True,
        )

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(log_path),
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=rotation,
            retention=retention,
            compression="gz",
            serialize=serialize,
        )

    # Error-only file handler
    error_log = PROJECT_ROOT / "logs" / "errors.log"
    error_log.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(error_log),
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="5 MB",
        retention="90 days",
        compression="gz",
    )


def get_logger(name: str | None = None) -> "logger":
    """
    Get a logger instance with optional name binding.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


class TradeLogger:
    """
    Specialized logger for trade-related events.

    Provides structured logging for:
    - Order submissions
    - Trade executions
    - Position changes
    - Risk events
    """

    def __init__(self) -> None:
        """Initialize trade logger with dedicated file handler."""
        self._logger = logger.bind(category="trades")

        # Add trade-specific file handler
        trade_log = PROJECT_ROOT / "logs" / "trades.log"
        trade_log.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(trade_log),
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
            rotation="5 MB",
            retention="90 days",
            filter=lambda record: record["extra"].get("category") == "trades",
        )

    def log_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: float | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Log an order submission.

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            order_type: Order type (MARKET/LIMIT)
            price: Limit price (if applicable)
            **kwargs: Additional order details
        """
        self._logger.info(
            f"ORDER | {symbol} | {side} | qty={quantity:.2f} | type={order_type} | price={price}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            **kwargs,
        )

    def log_fill(
        self,
        symbol: str,
        side: str,
        quantity: float,
        fill_price: float,
        commission: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """
        Log a trade fill/execution.

        Args:
            symbol: Trading symbol
            side: Trade side (BUY/SELL)
            quantity: Filled quantity
            fill_price: Execution price
            commission: Commission paid
            **kwargs: Additional fill details
        """
        self._logger.info(
            f"FILL | {symbol} | {side} | qty={quantity:.2f} | price={fill_price:.4f} | comm={commission:.2f}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            fill_price=fill_price,
            commission=commission,
            **kwargs,
        )

    def log_position(
        self,
        symbol: str,
        quantity: float,
        avg_price: float,
        market_value: float,
        unrealized_pnl: float,
        **kwargs: Any,
    ) -> None:
        """
        Log position update.

        Args:
            symbol: Trading symbol
            quantity: Current position quantity
            avg_price: Average entry price
            market_value: Current market value
            unrealized_pnl: Unrealized P&L
            **kwargs: Additional position details
        """
        self._logger.info(
            f"POSITION | {symbol} | qty={quantity:.2f} | avg={avg_price:.4f} | value={market_value:.2f} | pnl={unrealized_pnl:.2f}",
            symbol=symbol,
            quantity=quantity,
            avg_price=avg_price,
            market_value=market_value,
            unrealized_pnl=unrealized_pnl,
            **kwargs,
        )

    def log_risk_event(
        self,
        event_type: str,
        message: str,
        severity: str = "WARNING",
        **kwargs: Any,
    ) -> None:
        """
        Log a risk management event.

        Args:
            event_type: Type of risk event
            message: Event description
            severity: Event severity
            **kwargs: Additional event details
        """
        log_func = getattr(self._logger, severity.lower(), self._logger.warning)
        log_func(
            f"RISK | {event_type} | {message}",
            event_type=event_type,
            **kwargs,
        )


class PerformanceLogger:
    """
    Logger for performance metrics and timing.

    Tracks execution times and memory usage for performance monitoring.
    """

    def __init__(self, slow_threshold_ms: float = 1000.0) -> None:
        """
        Initialize performance logger.

        Args:
            slow_threshold_ms: Threshold for slow operation warning
        """
        self._logger = logger.bind(category="performance")
        self.slow_threshold_ms = slow_threshold_ms

    def log_timing(
        self,
        operation: str,
        duration_ms: float,
        **kwargs: Any,
    ) -> None:
        """
        Log operation timing.

        Args:
            operation: Name of the operation
            duration_ms: Duration in milliseconds
            **kwargs: Additional metrics
        """
        if duration_ms > self.slow_threshold_ms:
            self._logger.warning(
                f"SLOW | {operation} | {duration_ms:.2f}ms",
                operation=operation,
                duration_ms=duration_ms,
                **kwargs,
            )
        else:
            self._logger.debug(
                f"TIMING | {operation} | {duration_ms:.2f}ms",
                operation=operation,
                duration_ms=duration_ms,
                **kwargs,
            )

    def log_memory(
        self,
        operation: str,
        memory_mb: float,
        **kwargs: Any,
    ) -> None:
        """
        Log memory usage.

        Args:
            operation: Name of the operation
            memory_mb: Memory usage in MB
            **kwargs: Additional metrics
        """
        self._logger.debug(
            f"MEMORY | {operation} | {memory_mb:.2f}MB",
            operation=operation,
            memory_mb=memory_mb,
            **kwargs,
        )


# Initialize logging with default settings
setup_logging(
    level=settings.logging.level,
    log_file=settings.logging.log_file,
    rotation=settings.logging.rotation,
    retention=settings.logging.retention,
    serialize=settings.logging.serialize,
)

# Create global logger instances
trade_logger = TradeLogger()
perf_logger = PerformanceLogger()
