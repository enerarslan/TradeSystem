"""
Audit Trail System for AlphaTrade.

JPMorgan-level implementation of immutable audit logging:
- All order events (submit, modify, cancel, fill, reject)
- All risk events (limit breach, circuit breaker)
- All system events (startup, shutdown, config change)
- Compliance with regulatory requirements (MiFID II, SEC)

The audit trail is:
- Immutable (append-only)
- Timestamped with microsecond precision
- Sequentially numbered
- Cryptographically verifiable (hash chain)
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


class AuditEventType(Enum):
    """Types of events in the audit trail."""

    # Order events
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_ACKNOWLEDGED = "ORDER_ACKNOWLEDGED"
    ORDER_MODIFIED = "ORDER_MODIFIED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_PARTIALLY_FILLED = "ORDER_PARTIALLY_FILLED"
    ORDER_EXPIRED = "ORDER_EXPIRED"

    # Execution events
    EXECUTION_RECEIVED = "EXECUTION_RECEIVED"
    EXECUTION_CONFIRMED = "EXECUTION_CONFIRMED"
    EXECUTION_BREAK = "EXECUTION_BREAK"

    # Risk events
    RISK_LIMIT_WARNING = "RISK_LIMIT_WARNING"
    RISK_LIMIT_BREACH = "RISK_LIMIT_BREACH"
    CIRCUIT_BREAKER_TRIGGERED = "CIRCUIT_BREAKER_TRIGGERED"
    CIRCUIT_BREAKER_RESET = "CIRCUIT_BREAKER_RESET"
    POSITION_LIQUIDATION = "POSITION_LIQUIDATION"

    # Compliance events
    COMPLIANCE_CHECK_PASSED = "COMPLIANCE_CHECK_PASSED"
    COMPLIANCE_CHECK_FAILED = "COMPLIANCE_CHECK_FAILED"
    COMPLIANCE_OVERRIDE = "COMPLIANCE_OVERRIDE"
    RESTRICTED_SECURITY_ACCESSED = "RESTRICTED_SECURITY_ACCESSED"

    # System events
    SYSTEM_STARTUP = "SYSTEM_STARTUP"
    SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"
    CONFIG_CHANGE = "CONFIG_CHANGE"
    CONNECTION_ESTABLISHED = "CONNECTION_ESTABLISHED"
    CONNECTION_LOST = "CONNECTION_LOST"
    ERROR = "ERROR"

    # User events
    USER_LOGIN = "USER_LOGIN"
    USER_LOGOUT = "USER_LOGOUT"
    USER_ACTION = "USER_ACTION"
    MANUAL_INTERVENTION = "MANUAL_INTERVENTION"


@dataclass
class AuditEvent:
    """A single audit trail event."""

    # Core fields
    event_id: str
    sequence_number: int
    timestamp: datetime
    event_type: AuditEventType
    source: str  # System/module that generated the event

    # Event details
    description: str
    severity: str  # INFO, WARNING, ERROR, CRITICAL

    # Context
    account_id: str | None = None
    strategy_id: str | None = None
    order_id: str | None = None
    symbol: str | None = None
    side: str | None = None
    quantity: float | None = None
    price: float | None = None

    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Audit chain
    previous_hash: str = ""
    event_hash: str = ""

    def compute_hash(self) -> str:
        """Compute hash of this event for chain verification."""
        data = (
            f"{self.event_id}|{self.sequence_number}|{self.timestamp.isoformat()}|"
            f"{self.event_type.value}|{self.source}|{self.description}|"
            f"{self.order_id}|{self.symbol}|{self.quantity}|{self.price}|"
            f"{self.previous_hash}"
        )
        return hashlib.sha256(data.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "sequence_number": self.sequence_number,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "source": self.source,
            "description": self.description,
            "severity": self.severity,
            "account_id": self.account_id,
            "strategy_id": self.strategy_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "metadata": self.metadata,
            "previous_hash": self.previous_hash,
            "event_hash": self.event_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEvent":
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            sequence_number=data["sequence_number"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_type=AuditEventType(data["event_type"]),
            source=data["source"],
            description=data["description"],
            severity=data["severity"],
            account_id=data.get("account_id"),
            strategy_id=data.get("strategy_id"),
            order_id=data.get("order_id"),
            symbol=data.get("symbol"),
            side=data.get("side"),
            quantity=data.get("quantity"),
            price=data.get("price"),
            metadata=data.get("metadata", {}),
            previous_hash=data.get("previous_hash", ""),
            event_hash=data.get("event_hash", ""),
        )


class AuditTrail:
    """
    Immutable audit trail system.

    All events are:
    1. Assigned a unique ID and sequence number
    2. Timestamped with microsecond precision
    3. Hash-chained for tamper detection
    4. Persisted to disk immediately
    5. Never modified or deleted
    """

    def __init__(
        self,
        log_dir: str | Path = "logs/audit",
        rotate_size_mb: int = 100,
        compress_rotated: bool = True,
        sync_interval: int = 10,  # Sync to disk every N events
    ) -> None:
        """
        Initialize audit trail.

        Args:
            log_dir: Directory for audit log files
            rotate_size_mb: Rotate log when file reaches this size
            compress_rotated: Compress rotated log files
            sync_interval: Flush to disk after this many events
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.rotate_size_mb = rotate_size_mb
        self.compress_rotated = compress_rotated
        self.sync_interval = sync_interval

        # State
        self._sequence_number = 0
        self._last_hash = "GENESIS"
        self._events_since_sync = 0

        # Current log file
        self._current_log_path: Path | None = None
        self._log_file: Any = None

        # Thread safety
        self._lock = threading.Lock()

        # Event callbacks
        self._callbacks: List[Callable[[AuditEvent], None]] = []

        # Initialize log file
        self._initialize_log()

        logger.info(f"Audit trail initialized: {self.log_dir}")

    def _initialize_log(self) -> None:
        """Initialize or open the current log file."""
        today = datetime.now().strftime("%Y%m%d")
        self._current_log_path = self.log_dir / f"audit_{today}.jsonl"

        # Load last sequence number and hash if file exists
        if self._current_log_path.exists():
            self._load_last_state()

        # Open file for appending
        self._log_file = open(self._current_log_path, "a", buffering=1)

    def _load_last_state(self) -> None:
        """Load last sequence number and hash from existing log."""
        try:
            with open(self._current_log_path, "r") as f:
                last_line = None
                for line in f:
                    if line.strip():
                        last_line = line

                if last_line:
                    event_data = json.loads(last_line)
                    self._sequence_number = event_data.get("sequence_number", 0)
                    self._last_hash = event_data.get("event_hash", "GENESIS")
                    logger.debug(
                        f"Loaded audit state: seq={self._sequence_number}, "
                        f"hash={self._last_hash[:16]}..."
                    )
        except Exception as e:
            logger.error(f"Failed to load audit state: {e}")

    def log_event(
        self,
        event_type: AuditEventType,
        source: str,
        description: str,
        severity: str = "INFO",
        account_id: str | None = None,
        strategy_id: str | None = None,
        order_id: str | None = None,
        symbol: str | None = None,
        side: str | None = None,
        quantity: float | None = None,
        price: float | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> AuditEvent:
        """
        Log an audit event.

        This method is thread-safe and will:
        1. Assign sequence number and timestamp
        2. Compute hash chain
        3. Persist to disk
        4. Notify callbacks

        Args:
            event_type: Type of event
            source: System/module generating the event
            description: Human-readable description
            severity: INFO, WARNING, ERROR, or CRITICAL
            account_id: Associated account
            strategy_id: Associated strategy
            order_id: Associated order
            symbol: Associated symbol
            side: BUY or SELL
            quantity: Order/fill quantity
            price: Order/execution price
            metadata: Additional data

        Returns:
            The created AuditEvent
        """
        with self._lock:
            # Check for log rotation
            self._check_rotation()

            # Create event
            self._sequence_number += 1

            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                sequence_number=self._sequence_number,
                timestamp=datetime.now(),
                event_type=event_type,
                source=source,
                description=description,
                severity=severity,
                account_id=account_id,
                strategy_id=strategy_id,
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                metadata=metadata or {},
                previous_hash=self._last_hash,
            )

            # Compute hash
            event.event_hash = event.compute_hash()
            self._last_hash = event.event_hash

            # Persist
            self._write_event(event)

            # Sync if needed
            self._events_since_sync += 1
            if self._events_since_sync >= self.sync_interval:
                self._sync()

        # Notify callbacks (outside lock)
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Audit callback error: {e}")

        return event

    def _write_event(self, event: AuditEvent) -> None:
        """Write event to log file."""
        try:
            line = json.dumps(event.to_dict()) + "\n"
            self._log_file.write(line)
        except Exception as e:
            logger.error(f"Failed to write audit event: {e}")
            raise

    def _sync(self) -> None:
        """Sync log file to disk."""
        if self._log_file:
            self._log_file.flush()
            os.fsync(self._log_file.fileno())
            self._events_since_sync = 0

    def _check_rotation(self) -> None:
        """Check if log file needs rotation."""
        if self._current_log_path and self._current_log_path.exists():
            size_mb = self._current_log_path.stat().st_size / (1024 * 1024)

            # Also check for date change
            today = datetime.now().strftime("%Y%m%d")
            current_date = self._current_log_path.stem.split("_")[1]

            if size_mb >= self.rotate_size_mb or today != current_date:
                self._rotate_log()

    def _rotate_log(self) -> None:
        """Rotate the current log file."""
        if self._log_file:
            self._sync()
            self._log_file.close()

            # Compress if configured
            if self.compress_rotated and self._current_log_path:
                self._compress_log(self._current_log_path)

            # Start new log
            self._initialize_log()

    def _compress_log(self, filepath: Path) -> None:
        """Compress a log file."""
        try:
            compressed_path = filepath.with_suffix(filepath.suffix + ".gz")
            with open(filepath, "rb") as f_in:
                with gzip.open(compressed_path, "wb") as f_out:
                    f_out.writelines(f_in)
            # Don't delete original - keep for immediate access
            logger.info(f"Compressed audit log: {compressed_path}")
        except Exception as e:
            logger.error(f"Failed to compress audit log: {e}")

    def add_callback(self, callback: Callable[[AuditEvent], None]) -> None:
        """Add a callback to be notified of new events."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[AuditEvent], None]) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def verify_chain(self, filepath: str | Path | None = None) -> tuple[bool, str]:
        """
        Verify the integrity of the audit chain.

        Args:
            filepath: Specific file to verify (default: current log)

        Returns:
            Tuple of (is_valid, message)
        """
        filepath = Path(filepath) if filepath else self._current_log_path

        if not filepath or not filepath.exists():
            return False, "Log file not found"

        try:
            previous_hash = "GENESIS"
            line_number = 0

            with open(filepath, "r") as f:
                for line in f:
                    if not line.strip():
                        continue

                    line_number += 1
                    event_data = json.loads(line)
                    event = AuditEvent.from_dict(event_data)

                    # Verify previous hash
                    if event.previous_hash != previous_hash:
                        return False, f"Hash chain break at line {line_number}"

                    # Verify event hash
                    computed_hash = event.compute_hash()
                    if computed_hash != event.event_hash:
                        return False, f"Invalid hash at line {line_number}"

                    previous_hash = event.event_hash

            return True, f"Chain verified: {line_number} events"

        except Exception as e:
            return False, f"Verification error: {e}"

    def query_events(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: List[AuditEventType] | None = None,
        order_id: str | None = None,
        symbol: str | None = None,
        severity: str | None = None,
        limit: int = 1000,
    ) -> List[AuditEvent]:
        """
        Query audit events with filters.

        Args:
            start_time: Start of time range
            end_time: End of time range
            event_types: Filter by event types
            order_id: Filter by order ID
            symbol: Filter by symbol
            severity: Filter by severity
            limit: Maximum events to return

        Returns:
            List of matching events
        """
        events = []

        # Query all log files in date range
        log_files = sorted(self.log_dir.glob("audit_*.jsonl"))

        for log_file in log_files:
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue

                        event_data = json.loads(line)
                        event = AuditEvent.from_dict(event_data)

                        # Apply filters
                        if start_time and event.timestamp < start_time:
                            continue
                        if end_time and event.timestamp > end_time:
                            continue
                        if event_types and event.event_type not in event_types:
                            continue
                        if order_id and event.order_id != order_id:
                            continue
                        if symbol and event.symbol != symbol:
                            continue
                        if severity and event.severity != severity:
                            continue

                        events.append(event)

                        if len(events) >= limit:
                            break

                if len(events) >= limit:
                    break

            except Exception as e:
                logger.error(f"Error reading {log_file}: {e}")

        return events

    def export_csv(
        self,
        filepath: str | Path,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> int:
        """
        Export audit trail to CSV format.

        Args:
            filepath: Output CSV path
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Number of events exported
        """
        events = self.query_events(
            start_time=start_time,
            end_time=end_time,
            limit=1_000_000,  # High limit for export
        )

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "event_id", "sequence_number", "timestamp", "event_type",
            "source", "description", "severity", "account_id", "strategy_id",
            "order_id", "symbol", "side", "quantity", "price", "event_hash",
        ]

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for event in events:
                row = {
                    "event_id": event.event_id,
                    "sequence_number": event.sequence_number,
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type.value,
                    "source": event.source,
                    "description": event.description,
                    "severity": event.severity,
                    "account_id": event.account_id or "",
                    "strategy_id": event.strategy_id or "",
                    "order_id": event.order_id or "",
                    "symbol": event.symbol or "",
                    "side": event.side or "",
                    "quantity": event.quantity or "",
                    "price": event.price or "",
                    "event_hash": event.event_hash,
                }
                writer.writerow(row)

        logger.info(f"Exported {len(events)} events to {filepath}")
        return len(events)

    def get_statistics(self) -> Dict[str, Any]:
        """Get audit trail statistics."""
        return {
            "sequence_number": self._sequence_number,
            "log_directory": str(self.log_dir),
            "current_log": str(self._current_log_path) if self._current_log_path else None,
            "log_files": len(list(self.log_dir.glob("audit_*.jsonl"))),
            "last_hash": self._last_hash[:16] + "..." if self._last_hash != "GENESIS" else "GENESIS",
        }

    def close(self) -> None:
        """Close the audit trail properly."""
        with self._lock:
            if self._log_file:
                self._sync()
                self._log_file.close()
                self._log_file = None
        logger.info("Audit trail closed")

    # Convenience methods for common event types

    def log_order_submitted(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None,
        strategy_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Log an order submission."""
        return self.log_event(
            event_type=AuditEventType.ORDER_SUBMITTED,
            source="OrderManager",
            description=f"Order submitted: {side} {quantity} {symbol}",
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            strategy_id=strategy_id,
            metadata=metadata,
        )

    def log_order_filled(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        metadata: Dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Log an order fill."""
        return self.log_event(
            event_type=AuditEventType.ORDER_FILLED,
            source="ExecutionHandler",
            description=f"Order filled: {side} {quantity} {symbol} @ {price}",
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            metadata=metadata,
        )

    def log_risk_event(
        self,
        event_type: AuditEventType,
        description: str,
        severity: str = "WARNING",
        metadata: Dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Log a risk event."""
        return self.log_event(
            event_type=event_type,
            source="RiskManager",
            description=description,
            severity=severity,
            metadata=metadata,
        )

    def log_system_event(
        self,
        event_type: AuditEventType,
        description: str,
        metadata: Dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Log a system event."""
        return self.log_event(
            event_type=event_type,
            source="System",
            description=description,
            metadata=metadata,
        )
