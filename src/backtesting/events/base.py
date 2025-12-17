"""
Base event classes for event-driven backtesting.

This module defines the core event infrastructure including:
- Base Event class with microsecond precision timestamps
- Event types for categorization
- Event priorities for queue ordering
- Event serialization for audit trails

Designed for JPMorgan-level requirements:
- Immutable events for audit compliance
- Unique event IDs for tracking
- Nanosecond timestamp support where available
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Dict, Optional

import pandas as pd


class EventType(str, Enum):
    """Types of events in the system."""

    # Market events
    TICK = "TICK"
    BAR = "BAR"
    ORDER_BOOK = "ORDER_BOOK"
    TRADE = "TRADE"

    # Signal events
    SIGNAL = "SIGNAL"

    # Order events
    ORDER_NEW = "ORDER_NEW"
    ORDER_CANCEL = "ORDER_CANCEL"
    ORDER_MODIFY = "ORDER_MODIFY"

    # Fill events
    FILL = "FILL"
    PARTIAL_FILL = "PARTIAL_FILL"

    # System events
    HEARTBEAT = "HEARTBEAT"
    SESSION_START = "SESSION_START"
    SESSION_END = "SESSION_END"
    RISK_BREACH = "RISK_BREACH"

    # Portfolio events
    POSITION_UPDATE = "POSITION_UPDATE"
    PNL_UPDATE = "PNL_UPDATE"
    MARGIN_CALL = "MARGIN_CALL"


class EventPriority(IntEnum):
    """
    Event processing priority.

    Lower values are processed first. This ensures proper
    sequencing of events (e.g., fills before new signals).
    """
    CRITICAL = 0      # Risk breaches, margin calls
    HIGH = 10         # Fills, cancellations
    NORMAL = 50       # Market data, signals
    LOW = 100         # Heartbeats, logging


@dataclass(frozen=True)
class Event(ABC):
    """
    Base event class for all system events.

    Events are immutable (frozen) to ensure audit trail integrity.
    Each event has a unique ID and high-precision timestamp.

    Attributes:
        event_id: Unique identifier for this event
        event_type: Type classification of the event
        timestamp: When the event occurred
        created_at: When the event object was created
        priority: Processing priority
        source: Origin of the event (strategy, exchange, etc.)
        metadata: Additional event-specific data
    """

    event_type: EventType
    timestamp: pd.Timestamp
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    priority: EventPriority = EventPriority.NORMAL
    source: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert event to dictionary for serialization.

        Returns:
            Dictionary representation of the event
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """
        Create event from dictionary.

        Args:
            data: Dictionary with event data

        Returns:
            Event instance
        """
        pass

    def __lt__(self, other: "Event") -> bool:
        """
        Compare events for priority queue ordering.

        Events are ordered by:
        1. Priority (lower = higher priority)
        2. Timestamp (earlier first)
        3. Event ID (for deterministic ordering)
        """
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        return self.event_id < other.event_id

    def __le__(self, other: "Event") -> bool:
        return self == other or self < other

    def __gt__(self, other: "Event") -> bool:
        return not self <= other

    def __ge__(self, other: "Event") -> bool:
        return not self < other

    def __hash__(self) -> int:
        return hash(self.event_id)


@dataclass(frozen=True)
class SystemEvent(Event):
    """
    System-level events for session management.

    Used for:
    - Session start/end
    - Heartbeats
    - System status updates
    """

    message: str = ""
    level: str = "INFO"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": str(self.timestamp),
            "created_at": self.created_at.isoformat(),
            "priority": self.priority.value,
            "source": self.source,
            "message": self.message,
            "level": self.level,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemEvent":
        return cls(
            event_type=EventType(data["event_type"]),
            timestamp=pd.Timestamp(data["timestamp"]),
            event_id=data.get("event_id", str(uuid.uuid4())),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            priority=EventPriority(data.get("priority", EventPriority.NORMAL)),
            source=data.get("source", "system"),
            message=data.get("message", ""),
            level=data.get("level", "INFO"),
            metadata=data.get("metadata", {}),
        )


def create_heartbeat(source: str = "system") -> SystemEvent:
    """Create a heartbeat event."""
    return SystemEvent(
        event_type=EventType.HEARTBEAT,
        timestamp=pd.Timestamp.now(),
        priority=EventPriority.LOW,
        source=source,
        message="Heartbeat",
    )


def create_session_start(
    session_id: Optional[str] = None,
    source: str = "system",
) -> SystemEvent:
    """Create a session start event."""
    return SystemEvent(
        event_type=EventType.SESSION_START,
        timestamp=pd.Timestamp.now(),
        priority=EventPriority.HIGH,
        source=source,
        message="Session started",
        metadata={"session_id": session_id or str(uuid.uuid4())},
    )


def create_session_end(
    session_id: str,
    source: str = "system",
) -> SystemEvent:
    """Create a session end event."""
    return SystemEvent(
        event_type=EventType.SESSION_END,
        timestamp=pd.Timestamp.now(),
        priority=EventPriority.HIGH,
        source=source,
        message="Session ended",
        metadata={"session_id": session_id},
    )
