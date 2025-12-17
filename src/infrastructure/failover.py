"""
Failover and Recovery System for AlphaTrade.

JPMorgan-level implementation of system recovery:
- State checkpointing
- Automatic recovery
- Graceful degradation
- Manual intervention support

This module ensures system continuity and data integrity during failures.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


class RecoveryAction(Enum):
    """Actions that can be taken during recovery."""

    NONE = "none"
    RESTART_COMPONENT = "restart_component"
    RELOAD_STATE = "reload_state"
    FAILOVER_BACKUP = "failover_backup"
    HALT_TRADING = "halt_trading"
    MANUAL_INTERVENTION = "manual_intervention"
    FULL_RESTART = "full_restart"


@dataclass
class Checkpoint:
    """A system state checkpoint."""

    checkpoint_id: str
    timestamp: datetime
    component: str
    state_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "state_data": self.state_data,
            "metadata": self.metadata,
        }


@dataclass
class RecoveryEvent:
    """Record of a recovery event."""

    timestamp: datetime
    component: str
    failure_type: str
    action_taken: RecoveryAction
    success: bool
    details: str
    recovery_time_ms: float | None = None


class StateManager:
    """
    Manages system state persistence and recovery.

    Provides checkpointing, state save/restore, and recovery coordination.
    """

    def __init__(
        self,
        state_dir: str | Path = "state",
        checkpoint_interval: int = 300,  # 5 minutes
        max_checkpoints: int = 10,
    ) -> None:
        """
        Initialize state manager.

        Args:
            state_dir: Directory for state files
            checkpoint_interval: Seconds between auto-checkpoints
            max_checkpoints: Maximum checkpoints to retain per component
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints

        # Registered components and their state getters
        self._components: Dict[str, Callable[[], Dict[str, Any]]] = {}
        self._restore_handlers: Dict[str, Callable[[Dict[str, Any]], bool]] = {}

        # Checkpoint tracking
        self._checkpoints: Dict[str, List[Checkpoint]] = {}
        self._last_checkpoint_time: Dict[str, datetime] = {}

        # Auto-checkpoint thread
        self._checkpoint_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        logger.info(f"State manager initialized: {self.state_dir}")

    def register_component(
        self,
        name: str,
        state_getter: Callable[[], Dict[str, Any]],
        restore_handler: Callable[[Dict[str, Any]], bool] | None = None,
    ) -> None:
        """
        Register a component for state management.

        Args:
            name: Component name
            state_getter: Function that returns current state dict
            restore_handler: Optional function to restore state (returns success)
        """
        self._components[name] = state_getter
        if restore_handler:
            self._restore_handlers[name] = restore_handler
        self._checkpoints[name] = []

        logger.debug(f"Registered component for state management: {name}")

    def create_checkpoint(self, component: str | None = None) -> List[Checkpoint]:
        """
        Create a checkpoint for one or all components.

        Args:
            component: Specific component, or None for all

        Returns:
            List of created checkpoints
        """
        components = [component] if component else list(self._components.keys())
        checkpoints = []

        for comp in components:
            if comp not in self._components:
                continue

            try:
                state_getter = self._components[comp]
                state_data = state_getter()

                checkpoint = Checkpoint(
                    checkpoint_id=f"{comp}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now(),
                    component=comp,
                    state_data=state_data,
                    metadata={
                        "version": "1.0",
                        "size_bytes": len(json.dumps(state_data, default=str)),
                    },
                )

                # Store in memory
                self._checkpoints[comp].append(checkpoint)

                # Limit stored checkpoints
                if len(self._checkpoints[comp]) > self.max_checkpoints:
                    self._checkpoints[comp] = self._checkpoints[comp][-self.max_checkpoints:]

                # Persist to disk
                self._save_checkpoint(checkpoint)

                self._last_checkpoint_time[comp] = datetime.now()
                checkpoints.append(checkpoint)

                logger.debug(f"Checkpoint created: {checkpoint.checkpoint_id}")

            except Exception as e:
                logger.error(f"Failed to create checkpoint for {comp}: {e}")

        return checkpoints

    def _save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to disk."""
        comp_dir = self.state_dir / checkpoint.component
        comp_dir.mkdir(parents=True, exist_ok=True)

        filepath = comp_dir / f"{checkpoint.checkpoint_id}.json"

        with open(filepath, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2, default=str)

        # Also save pickled version for complex objects
        pickle_path = comp_dir / f"{checkpoint.checkpoint_id}.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(checkpoint.state_data, f)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints(comp_dir)

    def _cleanup_old_checkpoints(self, comp_dir: Path) -> None:
        """Remove old checkpoint files."""
        json_files = sorted(comp_dir.glob("*.json"))

        if len(json_files) > self.max_checkpoints:
            for old_file in json_files[:-self.max_checkpoints]:
                old_file.unlink()
                pkl_file = old_file.with_suffix(".pkl")
                if pkl_file.exists():
                    pkl_file.unlink()

    def get_latest_checkpoint(self, component: str) -> Checkpoint | None:
        """Get the most recent checkpoint for a component."""
        if component in self._checkpoints and self._checkpoints[component]:
            return self._checkpoints[component][-1]

        # Try to load from disk
        comp_dir = self.state_dir / component
        if comp_dir.exists():
            json_files = sorted(comp_dir.glob("*.json"))
            if json_files:
                return self._load_checkpoint(json_files[-1])

        return None

    def _load_checkpoint(self, filepath: Path) -> Checkpoint | None:
        """Load checkpoint from disk."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Try to load pickled state for full fidelity
            pkl_path = filepath.with_suffix(".pkl")
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    data["state_data"] = pickle.load(f)

            return Checkpoint(
                checkpoint_id=data["checkpoint_id"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                component=data["component"],
                state_data=data["state_data"],
                metadata=data.get("metadata", {}),
            )
        except Exception as e:
            logger.error(f"Failed to load checkpoint {filepath}: {e}")
            return None

    def restore_state(self, component: str, checkpoint: Checkpoint | None = None) -> bool:
        """
        Restore component state from checkpoint.

        Args:
            component: Component to restore
            checkpoint: Specific checkpoint, or None for latest

        Returns:
            True if restoration successful
        """
        if component not in self._restore_handlers:
            logger.warning(f"No restore handler for {component}")
            return False

        if checkpoint is None:
            checkpoint = self.get_latest_checkpoint(component)

        if checkpoint is None:
            logger.error(f"No checkpoint available for {component}")
            return False

        try:
            restore_handler = self._restore_handlers[component]
            success = restore_handler(checkpoint.state_data)

            if success:
                logger.info(
                    f"State restored for {component} from checkpoint "
                    f"{checkpoint.checkpoint_id}"
                )
            else:
                logger.error(f"State restoration failed for {component}")

            return success

        except Exception as e:
            logger.error(f"Exception during state restoration for {component}: {e}")
            return False

    def start_auto_checkpoint(self) -> None:
        """Start automatic checkpointing."""
        if self._checkpoint_thread and self._checkpoint_thread.is_alive():
            return

        self._stop_event.clear()
        self._checkpoint_thread = threading.Thread(
            target=self._auto_checkpoint_loop,
            daemon=True,
        )
        self._checkpoint_thread.start()
        logger.info("Auto-checkpointing started")

    def stop_auto_checkpoint(self) -> None:
        """Stop automatic checkpointing."""
        self._stop_event.set()
        if self._checkpoint_thread:
            self._checkpoint_thread.join(timeout=5)

    def _auto_checkpoint_loop(self) -> None:
        """Background checkpointing loop."""
        while not self._stop_event.is_set():
            self.create_checkpoint()
            self._stop_event.wait(self.checkpoint_interval)


class FailoverManager:
    """
    Manages system failover and recovery.

    Coordinates recovery actions when failures are detected.
    """

    def __init__(
        self,
        state_manager: StateManager | None = None,
        recovery_log_path: str | Path = "logs/recovery.log",
    ) -> None:
        """
        Initialize failover manager.

        Args:
            state_manager: StateManager instance
            recovery_log_path: Path for recovery event logging
        """
        self.state_manager = state_manager or StateManager()
        self.recovery_log_path = Path(recovery_log_path)
        self.recovery_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Recovery history
        self._recovery_events: List[RecoveryEvent] = []

        # Component restart handlers
        self._restart_handlers: Dict[str, Callable[[], bool]] = {}

        # Failure counts for escalation
        self._failure_counts: Dict[str, int] = {}
        self._failure_windows: Dict[str, List[datetime]] = {}

        # Escalation thresholds
        self._escalation_config = {
            "window_minutes": 15,
            "restart_threshold": 2,
            "halt_threshold": 5,
        }

        logger.info("Failover manager initialized")

    def register_restart_handler(
        self,
        component: str,
        handler: Callable[[], bool],
    ) -> None:
        """
        Register a restart handler for a component.

        Args:
            component: Component name
            handler: Function to restart component (returns success)
        """
        self._restart_handlers[component] = handler
        self._failure_counts[component] = 0
        self._failure_windows[component] = []

    def handle_failure(
        self,
        component: str,
        failure_type: str,
        details: str = "",
    ) -> RecoveryAction:
        """
        Handle a component failure.

        Determines appropriate recovery action based on failure history.

        Args:
            component: Failed component
            failure_type: Type of failure
            details: Additional details

        Returns:
            Recovery action taken
        """
        start_time = time.time()

        # Update failure tracking
        now = datetime.now()
        self._failure_counts[component] = self._failure_counts.get(component, 0) + 1

        if component not in self._failure_windows:
            self._failure_windows[component] = []
        self._failure_windows[component].append(now)

        # Clean old failures from window
        window_cutoff = now - timedelta(minutes=self._escalation_config["window_minutes"])
        self._failure_windows[component] = [
            t for t in self._failure_windows[component]
            if t > window_cutoff
        ]

        failures_in_window = len(self._failure_windows[component])

        # Determine action based on escalation
        if failures_in_window >= self._escalation_config["halt_threshold"]:
            action = RecoveryAction.HALT_TRADING
            success = self._halt_trading(component)
        elif failures_in_window >= self._escalation_config["restart_threshold"]:
            action = RecoveryAction.RESTART_COMPONENT
            success = self._restart_component(component)
        else:
            action = RecoveryAction.RELOAD_STATE
            success = self._reload_state(component)

        recovery_time = (time.time() - start_time) * 1000

        # Record event
        event = RecoveryEvent(
            timestamp=now,
            component=component,
            failure_type=failure_type,
            action_taken=action,
            success=success,
            details=details,
            recovery_time_ms=recovery_time,
        )
        self._recovery_events.append(event)

        # Log event
        self._log_recovery_event(event)

        logger.warning(
            f"Recovery [{component}]: {failure_type} -> {action.value} "
            f"({'success' if success else 'failed'}) in {recovery_time:.1f}ms"
        )

        return action

    def _restart_component(self, component: str) -> bool:
        """Restart a component."""
        if component in self._restart_handlers:
            try:
                return self._restart_handlers[component]()
            except Exception as e:
                logger.error(f"Restart handler failed for {component}: {e}")
                return False
        return False

    def _reload_state(self, component: str) -> bool:
        """Reload component state from checkpoint."""
        return self.state_manager.restore_state(component)

    def _halt_trading(self, component: str) -> bool:
        """Halt trading due to repeated failures."""
        logger.critical(
            f"TRADING HALTED: {component} has exceeded failure threshold. "
            "Manual intervention required."
        )
        # Would trigger circuit breakers, notifications, etc.
        return True

    def _log_recovery_event(self, event: RecoveryEvent) -> None:
        """Log recovery event to file."""
        try:
            with open(self.recovery_log_path, "a") as f:
                line = json.dumps({
                    "timestamp": event.timestamp.isoformat(),
                    "component": event.component,
                    "failure_type": event.failure_type,
                    "action": event.action_taken.value,
                    "success": event.success,
                    "details": event.details,
                    "recovery_time_ms": event.recovery_time_ms,
                })
                f.write(line + "\n")
        except Exception as e:
            logger.error(f"Failed to log recovery event: {e}")

    def get_recovery_history(
        self,
        component: str | None = None,
        limit: int = 100,
    ) -> List[RecoveryEvent]:
        """Get recovery event history."""
        events = self._recovery_events
        if component:
            events = [e for e in events if e.component == component]
        return events[-limit:]

    def get_failure_stats(self) -> Dict[str, Any]:
        """Get failure statistics."""
        return {
            "failure_counts": self._failure_counts.copy(),
            "total_recoveries": len(self._recovery_events),
            "successful_recoveries": sum(
                1 for e in self._recovery_events if e.success
            ),
            "components_tracked": len(self._failure_counts),
        }

    def manual_recovery(
        self,
        component: str,
        action: RecoveryAction,
    ) -> bool:
        """
        Trigger manual recovery action.

        Args:
            component: Component to recover
            action: Recovery action to take

        Returns:
            True if successful
        """
        logger.info(f"Manual recovery requested: {component} -> {action.value}")

        success = False

        if action == RecoveryAction.RESTART_COMPONENT:
            success = self._restart_component(component)
        elif action == RecoveryAction.RELOAD_STATE:
            success = self._reload_state(component)
        elif action == RecoveryAction.HALT_TRADING:
            success = self._halt_trading(component)

        # Reset failure counts on successful manual recovery
        if success:
            self._failure_counts[component] = 0
            self._failure_windows[component] = []

        event = RecoveryEvent(
            timestamp=datetime.now(),
            component=component,
            failure_type="manual_intervention",
            action_taken=action,
            success=success,
            details="Manual recovery requested",
        )
        self._recovery_events.append(event)

        return success

    def create_backup(self, backup_dir: str | Path) -> bool:
        """
        Create a full backup of all state.

        Args:
            backup_dir: Directory for backup files

        Returns:
            True if successful
        """
        backup_dir = Path(backup_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"backup_{timestamp}"

        try:
            # Create checkpoints for all components
            self.state_manager.create_checkpoint()

            # Copy state directory
            shutil.copytree(self.state_manager.state_dir, backup_path)

            logger.info(f"Backup created: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
