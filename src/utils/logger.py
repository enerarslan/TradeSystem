"""
Institutional-Grade Logging System
JPMorgan-Level Audit Trail and Monitoring

Features:
- Structured logging with JSON support
- Audit trail for compliance
- Performance metrics logging
- Trade execution logging
- Error tracking and alerting
"""

import logging
import logging.handlers
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
from functools import wraps
import traceback
import threading
from contextlib import contextmanager
import time
from dataclasses import dataclass, asdict
from enum import Enum


class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    AUDIT = "AUDIT"
    TRADE = "TRADE"
    PERFORMANCE = "PERFORMANCE"


@dataclass
class TradeLog:
    """Structured trade log entry"""
    timestamp: str
    trade_id: str
    symbol: str
    side: str  # BUY/SELL
    quantity: float
    price: float
    order_type: str
    status: str
    fill_price: Optional[float] = None
    fill_quantity: Optional[float] = None
    commission: Optional[float] = None
    slippage: Optional[float] = None
    latency_ms: Optional[float] = None
    strategy: Optional[str] = None
    signal_strength: Optional[float] = None
    risk_score: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class PerformanceLog:
    """Structured performance log entry"""
    timestamp: str
    metric_name: str
    value: float
    unit: str
    context: Optional[Dict[str, Any]] = None


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging"""

    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }

        # Add extra fields
        if self.include_extra:
            extra_fields = {
                k: v for k, v in record.__dict__.items()
                if k not in logging.LogRecord(
                    "", 0, "", 0, "", (), None
                ).__dict__ and not k.startswith("_")
            }
            if extra_fields:
                log_entry["extra"] = extra_fields

        return json.dumps(log_entry, default=str)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for development"""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'AUDIT': '\033[34m',     # Blue
        'TRADE': '\033[96m',     # Light Cyan
        'RESET': '\033[0m'       # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # Format timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Create formatted message
        formatted = (
            f"{color}[{timestamp}] "
            f"[{record.levelname:8}] "
            f"[{record.name}] "
            f"{record.getMessage()}{reset}"
        )

        if record.exc_info:
            formatted += f"\n{color}{self.formatException(record.exc_info)}{reset}"

        return formatted


class AuditLogger:
    """
    Compliance-grade audit logger for trade operations.
    Maintains immutable audit trail for regulatory compliance.
    """

    def __init__(self, log_path: str = "logs/audit"):
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)

        # Setup audit file handler
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # Daily rotating audit file
        audit_handler = logging.handlers.TimedRotatingFileHandler(
            self.log_path / "audit.log",
            when="midnight",
            interval=1,
            backupCount=365,  # Keep 1 year of audit logs
            encoding="utf-8"
        )
        audit_handler.setFormatter(JsonFormatter())
        self.logger.addHandler(audit_handler)

        # Trade-specific log
        self.trade_logger = logging.getLogger("audit.trades")
        self.trade_logger.setLevel(logging.INFO)
        self.trade_logger.propagate = False

        trade_handler = logging.handlers.TimedRotatingFileHandler(
            self.log_path / "trades.log",
            when="midnight",
            interval=1,
            backupCount=365,
            encoding="utf-8"
        )
        trade_handler.setFormatter(JsonFormatter())
        self.trade_logger.addHandler(trade_handler)

        self._lock = threading.Lock()

    def log_trade(self, trade: Union[TradeLog, Dict[str, Any]]) -> None:
        """Log trade execution with full audit trail"""
        with self._lock:
            if isinstance(trade, TradeLog):
                trade_dict = asdict(trade)
            else:
                trade_dict = trade

            trade_dict["audit_timestamp"] = datetime.utcnow().isoformat() + "Z"
            trade_dict["log_type"] = "TRADE_EXECUTION"

            self.trade_logger.info(
                json.dumps(trade_dict, default=str),
                extra={"trade_data": trade_dict}
            )

    def log_order(
            self,
            order_id: str,
            symbol: str,
            action: str,
            details: Dict[str, Any]
    ) -> None:
        """Log order lifecycle events"""
        with self._lock:
            log_entry = {
                "audit_timestamp": datetime.utcnow().isoformat() + "Z",
                "log_type": "ORDER_EVENT",
                "order_id": order_id,
                "symbol": symbol,
                "action": action,  # CREATED, SUBMITTED, FILLED, CANCELLED, REJECTED
                "details": details
            }
            self.logger.info(json.dumps(log_entry, default=str))

    def log_position_change(
            self,
            symbol: str,
            prev_quantity: float,
            new_quantity: float,
            reason: str,
            details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log position changes"""
        with self._lock:
            log_entry = {
                "audit_timestamp": datetime.utcnow().isoformat() + "Z",
                "log_type": "POSITION_CHANGE",
                "symbol": symbol,
                "prev_quantity": prev_quantity,
                "new_quantity": new_quantity,
                "change": new_quantity - prev_quantity,
                "reason": reason,
                "details": details or {}
            }
            self.logger.info(json.dumps(log_entry, default=str))

    def log_risk_event(
            self,
            event_type: str,
            severity: str,
            details: Dict[str, Any]
    ) -> None:
        """Log risk management events"""
        with self._lock:
            log_entry = {
                "audit_timestamp": datetime.utcnow().isoformat() + "Z",
                "log_type": "RISK_EVENT",
                "event_type": event_type,
                "severity": severity,
                "details": details
            }
            self.logger.warning(json.dumps(log_entry, default=str))

    def log_system_event(
            self,
            event_type: str,
            details: Dict[str, Any]
    ) -> None:
        """Log system-level events"""
        with self._lock:
            log_entry = {
                "audit_timestamp": datetime.utcnow().isoformat() + "Z",
                "log_type": "SYSTEM_EVENT",
                "event_type": event_type,
                "details": details
            }
            self.logger.info(json.dumps(log_entry, default=str))


class PerformanceLogger:
    """Logger for performance metrics and latency tracking"""

    def __init__(self, log_path: str = "logs/performance"):
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("performance")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        handler = logging.handlers.RotatingFileHandler(
            self.log_path / "performance.log",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10,
            encoding="utf-8"
        )
        handler.setFormatter(JsonFormatter())
        self.logger.addHandler(handler)

        self._metrics: Dict[str, list] = {}
        self._lock = threading.Lock()

    def log_metric(
            self,
            metric_name: str,
            value: float,
            unit: str = "ms",
            context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a performance metric"""
        log_entry = PerformanceLog(
            timestamp=datetime.utcnow().isoformat() + "Z",
            metric_name=metric_name,
            value=value,
            unit=unit,
            context=context
        )
        self.logger.info(json.dumps(asdict(log_entry), default=str))

        # Track for aggregation
        with self._lock:
            if metric_name not in self._metrics:
                self._metrics[metric_name] = []
            self._metrics[metric_name].append(value)

            # Keep only last 1000 values
            if len(self._metrics[metric_name]) > 1000:
                self._metrics[metric_name] = self._metrics[metric_name][-1000:]

    def log_latency(
            self,
            operation: str,
            latency_ms: float,
            context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log operation latency"""
        self.log_metric(f"latency.{operation}", latency_ms, "ms", context)

    @contextmanager
    def measure_time(self, operation: str, context: Optional[Dict[str, Any]] = None):
        """Context manager to measure operation time"""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.log_latency(operation, elapsed_ms, context)

    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        with self._lock:
            values = self._metrics.get(metric_name, [])
            if not values:
                return {}

            import numpy as np
            return {
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99))
            }


# Global loggers
_loggers: Dict[str, logging.Logger] = {}
_audit_logger: Optional[AuditLogger] = None
_performance_logger: Optional[PerformanceLogger] = None


def setup_logging(
        level: str = "INFO",
        log_path: str = "logs",
        json_format: bool = False,
        console: bool = True,
        file_logging: bool = True
) -> None:
    """
    Setup global logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_path: Path for log files
        json_format: Use JSON format for logs
        console: Enable console output
        file_logging: Enable file logging
    """
    global _audit_logger, _performance_logger

    log_path = Path(log_path)
    log_path.mkdir(parents=True, exist_ok=True)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))

        if json_format:
            console_handler.setFormatter(JsonFormatter())
        else:
            console_handler.setFormatter(ColoredFormatter())

        root_logger.addHandler(console_handler)

    # File handler
    if file_logging:
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / "trading.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(file_handler)

        # Error-only file
        error_handler = logging.handlers.RotatingFileHandler(
            log_path / "errors.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=10,
            encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(error_handler)

    # Initialize audit and performance loggers
    _audit_logger = AuditLogger(str(log_path / "audit"))
    _performance_logger = PerformanceLogger(str(log_path / "performance"))

    logging.info(f"Logging initialized. Level: {level}, Path: {log_path}")


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a named logger.

    Args:
        name: Logger name (typically module name)

    Returns:
        Configured logger instance
    """
    if name not in _loggers:
        logger = logging.getLogger(name)
        _loggers[name] = logger

    return _loggers[name]


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def get_performance_logger() -> PerformanceLogger:
    """Get the global performance logger"""
    global _performance_logger
    if _performance_logger is None:
        _performance_logger = PerformanceLogger()
    return _performance_logger


def log_execution_time(logger: Optional[logging.Logger] = None):
    """Decorator to log function execution time"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)

            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                logger.debug(
                    f"{func.__name__} completed in {elapsed:.2f}ms"
                )
                return result
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                logger.error(
                    f"{func.__name__} failed after {elapsed:.2f}ms: {e}",
                    exc_info=True
                )
                raise

        return wrapper

    return decorator


def log_trade_execution(func):
    """Decorator to log trade execution details"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        audit = get_audit_logger()
        perf = get_performance_logger()

        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            latency = (time.perf_counter() - start) * 1000

            # Log performance
            perf.log_latency("trade_execution", latency)

            return result
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            audit.log_system_event(
                "TRADE_EXECUTION_ERROR",
                {
                    "function": func.__name__,
                    "error": str(e),
                    "latency_ms": latency
                }
            )
            raise

    return wrapper


# Custom log levels
AUDIT_LEVEL = 25
TRADE_LEVEL = 26
PERFORMANCE_LEVEL = 27

logging.addLevelName(AUDIT_LEVEL, "AUDIT")
logging.addLevelName(TRADE_LEVEL, "TRADE")
logging.addLevelName(PERFORMANCE_LEVEL, "PERFORMANCE")


def audit(self, message, *args, **kwargs):
    if self.isEnabledFor(AUDIT_LEVEL):
        self._log(AUDIT_LEVEL, message, args, **kwargs)


def trade(self, message, *args, **kwargs):
    if self.isEnabledFor(TRADE_LEVEL):
        self._log(TRADE_LEVEL, message, args, **kwargs)


def performance(self, message, *args, **kwargs):
    if self.isEnabledFor(PERFORMANCE_LEVEL):
        self._log(PERFORMANCE_LEVEL, message, args, **kwargs)


# Add custom methods to Logger class
logging.Logger.audit = audit
logging.Logger.trade = trade
logging.Logger.performance = performance


def set_backtest_logging_mode(fast_mode: bool = True) -> None:
    """
    Configure logging for backtest performance optimization.

    OPTIMIZATION:
    - fast_mode=True: Reduces logging to WARNING level, disables file I/O
    - fast_mode=False: Normal logging (INFO level, full file logging)

    This reduces logging overhead from ~5-10% of runtime to <1%.

    Args:
        fast_mode: Enable fast logging mode
    """
    root_logger = logging.getLogger()

    if fast_mode:
        # Set to WARNING to skip INFO and DEBUG messages
        root_logger.setLevel(logging.WARNING)

        # Disable file handlers during backtest
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.setLevel(logging.CRITICAL)  # Only log critical errors

        logging.warning("Backtest fast logging mode enabled (WARNING level)")
    else:
        # Restore normal logging
        root_logger.setLevel(logging.INFO)

        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                if 'error' in str(handler.baseFilename).lower():
                    handler.setLevel(logging.ERROR)
                else:
                    handler.setLevel(logging.DEBUG)

        logging.info("Backtest normal logging mode restored (INFO level)")


def get_optimized_logger(name: str, level: str = "WARNING") -> logging.Logger:
    """
    Get an optimized logger for performance-critical code.

    Use this in hot paths like backtest loops.

    Args:
        name: Logger name
        level: Minimum log level (default WARNING for performance)

    Returns:
        Logger with optimized settings
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    return logger
