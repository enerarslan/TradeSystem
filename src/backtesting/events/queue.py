"""
Event queue implementations for event-driven backtesting.

This module provides queue implementations:
- EventQueue: Basic FIFO event queue
- PriorityEventQueue: Priority-based event queue
- AsyncEventQueue: Async-compatible queue

Designed for institutional requirements:
- Thread-safe operations
- Priority ordering
- High-throughput processing
"""

from __future__ import annotations

import heapq
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional, Set

import pandas as pd

from src.backtesting.events.base import Event, EventPriority, EventType

logger = logging.getLogger(__name__)


class EventQueue:
    """
    Basic FIFO event queue.

    Simple queue for events processed in arrival order.
    Thread-safe for producer/consumer patterns.
    """

    def __init__(self, maxsize: int = 0):
        """
        Initialize event queue.

        Args:
            maxsize: Maximum queue size (0 = unlimited)
        """
        self._queue: deque = deque()
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._processed_count = 0
        self._dropped_count = 0

    def put(self, event: Event) -> bool:
        """
        Add event to queue.

        Args:
            event: Event to add

        Returns:
            True if added, False if queue full
        """
        with self._lock:
            if self._maxsize > 0 and len(self._queue) >= self._maxsize:
                self._dropped_count += 1
                logger.warning(f"Queue full, dropping event: {event.event_id}")
                return False

            self._queue.append(event)
            return True

    def get(self) -> Optional[Event]:
        """
        Get next event from queue.

        Returns:
            Event or None if queue empty
        """
        with self._lock:
            if not self._queue:
                return None

            event = self._queue.popleft()
            self._processed_count += 1
            return event

    def peek(self) -> Optional[Event]:
        """
        Peek at next event without removing.

        Returns:
            Event or None if queue empty
        """
        with self._lock:
            if not self._queue:
                return None
            return self._queue[0]

    def empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return len(self._queue) == 0

    def size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return len(self._queue)

    def clear(self) -> int:
        """
        Clear all events from queue.

        Returns:
            Number of events cleared
        """
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            return count

    @property
    def processed_count(self) -> int:
        """Number of events processed."""
        return self._processed_count

    @property
    def dropped_count(self) -> int:
        """Number of events dropped."""
        return self._dropped_count

    def __len__(self) -> int:
        return self.size()

    def __iter__(self) -> Iterator[Event]:
        """Iterate over events (consumes queue)."""
        while True:
            event = self.get()
            if event is None:
                break
            yield event


class PriorityEventQueue:
    """
    Priority-based event queue.

    Events are processed in priority order:
    1. Priority (lower = higher priority)
    2. Timestamp (earlier first)
    3. Event ID (for determinism)

    Uses a min-heap for O(log n) operations.
    """

    def __init__(self, maxsize: int = 0):
        """
        Initialize priority queue.

        Args:
            maxsize: Maximum queue size (0 = unlimited)
        """
        self._heap: List[Event] = []
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._processed_count = 0
        self._dropped_count = 0
        self._event_ids: Set[str] = set()  # For duplicate detection

    def put(self, event: Event, allow_duplicates: bool = False) -> bool:
        """
        Add event to queue.

        Args:
            event: Event to add
            allow_duplicates: Allow duplicate event IDs

        Returns:
            True if added, False if queue full or duplicate
        """
        with self._lock:
            # Check for duplicates
            if not allow_duplicates and event.event_id in self._event_ids:
                logger.debug(f"Duplicate event ignored: {event.event_id}")
                return False

            # Check queue size
            if self._maxsize > 0 and len(self._heap) >= self._maxsize:
                self._dropped_count += 1
                logger.warning(f"Queue full, dropping event: {event.event_id}")
                return False

            heapq.heappush(self._heap, event)
            self._event_ids.add(event.event_id)
            return True

    def get(self) -> Optional[Event]:
        """
        Get highest priority event.

        Returns:
            Event or None if queue empty
        """
        with self._lock:
            if not self._heap:
                return None

            event = heapq.heappop(self._heap)
            self._event_ids.discard(event.event_id)
            self._processed_count += 1
            return event

    def peek(self) -> Optional[Event]:
        """
        Peek at highest priority event without removing.

        Returns:
            Event or None if queue empty
        """
        with self._lock:
            if not self._heap:
                return None
            return self._heap[0]

    def empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return len(self._heap) == 0

    def size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return len(self._heap)

    def clear(self) -> int:
        """
        Clear all events from queue.

        Returns:
            Number of events cleared
        """
        with self._lock:
            count = len(self._heap)
            self._heap.clear()
            self._event_ids.clear()
            return count

    def get_by_type(self, event_type: EventType) -> List[Event]:
        """
        Get all events of a specific type.

        Note: Does not remove events from queue.

        Args:
            event_type: Event type to filter

        Returns:
            List of matching events
        """
        with self._lock:
            return [e for e in self._heap if e.event_type == event_type]

    def remove(self, event_id: str) -> bool:
        """
        Remove specific event by ID.

        Args:
            event_id: Event ID to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            for i, event in enumerate(self._heap):
                if event.event_id == event_id:
                    self._heap.pop(i)
                    heapq.heapify(self._heap)
                    self._event_ids.discard(event_id)
                    return True
            return False

    @property
    def processed_count(self) -> int:
        """Number of events processed."""
        return self._processed_count

    @property
    def dropped_count(self) -> int:
        """Number of events dropped."""
        return self._dropped_count

    def __len__(self) -> int:
        return self.size()

    def __iter__(self) -> Iterator[Event]:
        """Iterate over events in priority order (consumes queue)."""
        while True:
            event = self.get()
            if event is None:
                break
            yield event


@dataclass
class EventHandler:
    """
    Event handler registration.

    Attributes:
        event_type: Type of events to handle
        callback: Handler function
        priority: Handler priority (for ordering)
        name: Handler name for logging
    """
    event_type: EventType
    callback: Callable[[Event], None]
    priority: int = 0
    name: str = ""


class EventDispatcher:
    """
    Event dispatcher with handler registration.

    Provides pub/sub pattern for event distribution
    to multiple handlers.
    """

    def __init__(self):
        """Initialize event dispatcher."""
        self._handlers: Dict[EventType, List[EventHandler]] = {}
        self._global_handlers: List[EventHandler] = []
        self._lock = threading.Lock()
        self._dispatch_count = 0
        self._error_count = 0

    def register(
        self,
        event_type: EventType,
        callback: Callable[[Event], None],
        priority: int = 0,
        name: str = "",
    ) -> None:
        """
        Register event handler.

        Args:
            event_type: Event type to handle
            callback: Handler function
            priority: Handler priority (lower = earlier)
            name: Handler name for logging
        """
        handler = EventHandler(
            event_type=event_type,
            callback=callback,
            priority=priority,
            name=name or callback.__name__,
        )

        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []

            self._handlers[event_type].append(handler)
            # Sort by priority
            self._handlers[event_type].sort(key=lambda h: h.priority)

        logger.debug(f"Registered handler '{handler.name}' for {event_type}")

    def register_global(
        self,
        callback: Callable[[Event], None],
        priority: int = 0,
        name: str = "",
    ) -> None:
        """
        Register global handler for all events.

        Args:
            callback: Handler function
            priority: Handler priority
            name: Handler name
        """
        handler = EventHandler(
            event_type=EventType.HEARTBEAT,  # Placeholder
            callback=callback,
            priority=priority,
            name=name or callback.__name__,
        )

        with self._lock:
            self._global_handlers.append(handler)
            self._global_handlers.sort(key=lambda h: h.priority)

        logger.debug(f"Registered global handler '{handler.name}'")

    def unregister(
        self,
        event_type: EventType,
        callback: Callable[[Event], None],
    ) -> bool:
        """
        Unregister event handler.

        Args:
            event_type: Event type
            callback: Handler function to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if event_type not in self._handlers:
                return False

            original_count = len(self._handlers[event_type])
            self._handlers[event_type] = [
                h for h in self._handlers[event_type]
                if h.callback != callback
            ]

            return len(self._handlers[event_type]) < original_count

    def dispatch(self, event: Event) -> int:
        """
        Dispatch event to all registered handlers.

        Args:
            event: Event to dispatch

        Returns:
            Number of handlers that processed the event
        """
        handlers_called = 0

        with self._lock:
            # Get handlers for this event type
            type_handlers = self._handlers.get(event.event_type, [])
            all_handlers = self._global_handlers + type_handlers

        for handler in all_handlers:
            try:
                handler.callback(event)
                handlers_called += 1
            except Exception as e:
                self._error_count += 1
                logger.error(
                    f"Handler '{handler.name}' error for {event.event_type}: {e}"
                )

        self._dispatch_count += 1
        return handlers_called

    def clear(self) -> None:
        """Clear all registered handlers."""
        with self._lock:
            self._handlers.clear()
            self._global_handlers.clear()

    @property
    def dispatch_count(self) -> int:
        """Number of events dispatched."""
        return self._dispatch_count

    @property
    def error_count(self) -> int:
        """Number of handler errors."""
        return self._error_count


class EventBus:
    """
    Complete event bus combining queue and dispatcher.

    Provides full event-driven architecture with:
    - Priority queue
    - Handler registration
    - Event dispatch
    - Processing loop
    """

    def __init__(self, maxsize: int = 0):
        """
        Initialize event bus.

        Args:
            maxsize: Maximum queue size
        """
        self._queue = PriorityEventQueue(maxsize=maxsize)
        self._dispatcher = EventDispatcher()
        self._running = False
        self._processing_thread: Optional[threading.Thread] = None

    def publish(self, event: Event) -> bool:
        """
        Publish event to bus.

        Args:
            event: Event to publish

        Returns:
            True if published successfully
        """
        return self._queue.put(event)

    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[Event], None],
        priority: int = 0,
        name: str = "",
    ) -> None:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Event type to subscribe to
            callback: Handler function
            priority: Handler priority
            name: Handler name
        """
        self._dispatcher.register(event_type, callback, priority, name)

    def subscribe_all(
        self,
        callback: Callable[[Event], None],
        priority: int = 0,
        name: str = "",
    ) -> None:
        """
        Subscribe to all events.

        Args:
            callback: Handler function
            priority: Handler priority
            name: Handler name
        """
        self._dispatcher.register_global(callback, priority, name)

    def unsubscribe(
        self,
        event_type: EventType,
        callback: Callable[[Event], None],
    ) -> bool:
        """
        Unsubscribe from events.

        Args:
            event_type: Event type
            callback: Handler to remove

        Returns:
            True if removed
        """
        return self._dispatcher.unregister(event_type, callback)

    def process_one(self) -> bool:
        """
        Process single event from queue.

        Returns:
            True if event was processed, False if queue empty
        """
        event = self._queue.get()
        if event is None:
            return False

        self._dispatcher.dispatch(event)
        return True

    def process_all(self) -> int:
        """
        Process all events in queue.

        Returns:
            Number of events processed
        """
        count = 0
        while self.process_one():
            count += 1
        return count

    def start(self) -> None:
        """Start background event processing."""
        if self._running:
            return

        self._running = True
        self._processing_thread = threading.Thread(
            target=self._process_loop,
            daemon=True,
        )
        self._processing_thread.start()
        logger.info("Event bus started")

    def stop(self) -> None:
        """Stop background event processing."""
        self._running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)
            self._processing_thread = None
        logger.info("Event bus stopped")

    def _process_loop(self) -> None:
        """Background processing loop."""
        while self._running:
            if not self.process_one():
                # Small sleep when queue is empty
                import time
                time.sleep(0.001)

    @property
    def queue_size(self) -> int:
        """Current queue size."""
        return self._queue.size()

    @property
    def is_running(self) -> bool:
        """Check if bus is running."""
        return self._running

    def clear(self) -> None:
        """Clear queue and handlers."""
        self._queue.clear()
        self._dispatcher.clear()

    def stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            "queue_size": self._queue.size(),
            "processed_count": self._queue.processed_count,
            "dropped_count": self._queue.dropped_count,
            "dispatch_count": self._dispatcher.dispatch_count,
            "error_count": self._dispatcher.error_count,
            "is_running": self._running,
        }
