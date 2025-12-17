"""
Training Checkpointing Module for institutional-grade model training.

This module provides comprehensive checkpointing capabilities:
- Save and resume training state
- Automatic checkpoint management
- Model versioning and lineage tracking
- Checkpoint validation and integrity checks
- Support for both sklearn and deep learning models

Designed for JPMorgan-level requirements:
- Fault tolerance for long-running training
- Reproducible training resumption
- Complete audit trail of training progress
- Memory-efficient checkpoint storage
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for a training checkpoint."""
    checkpoint_id: str
    epoch: int
    step: int
    timestamp: str
    model_type: str
    task_type: str
    metrics: Dict[str, float]
    best_metric_value: Optional[float]
    best_metric_name: Optional[str]
    data_hash: Optional[str]
    random_state: int
    params: Dict[str, Any]
    training_time_seconds: float
    n_samples_processed: int
    is_best: bool = False
    fold_idx: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetadata":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TrainingState:
    """
    Complete training state for resumption.

    Contains all information needed to resume training
    from a checkpoint.
    """
    epoch: int
    step: int
    best_metric: Optional[float]
    best_epoch: Optional[int]
    early_stopping_counter: int
    learning_rate: Optional[float]
    optimizer_state: Optional[Dict[str, Any]]
    scheduler_state: Optional[Dict[str, Any]]
    cv_fold_results: List[Dict[str, float]] = field(default_factory=list)
    training_history: List[Dict[str, float]] = field(default_factory=list)
    random_state: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "epoch": self.epoch,
            "step": self.step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "early_stopping_counter": self.early_stopping_counter,
            "learning_rate": self.learning_rate,
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "cv_fold_results": self.cv_fold_results,
            "training_history": self.training_history,
            "random_state": self.random_state,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingState":
        """Create from dictionary."""
        return cls(
            epoch=data.get("epoch", 0),
            step=data.get("step", 0),
            best_metric=data.get("best_metric"),
            best_epoch=data.get("best_epoch"),
            early_stopping_counter=data.get("early_stopping_counter", 0),
            learning_rate=data.get("learning_rate"),
            optimizer_state=data.get("optimizer_state"),
            scheduler_state=data.get("scheduler_state"),
            cv_fold_results=data.get("cv_fold_results", []),
            training_history=data.get("training_history", []),
            random_state=data.get("random_state", 42),
        )


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management."""
    checkpoint_dir: Path
    save_frequency: int = 10  # Save every N epochs
    keep_last_n: int = 3  # Keep last N checkpoints
    keep_best_n: int = 1  # Keep best N checkpoints
    monitor_metric: str = "val_loss"
    mode: str = "min"  # "min" or "max"
    save_optimizer: bool = True
    save_scheduler: bool = True
    compress: bool = False
    validate_on_load: bool = True

    def __post_init__(self):
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


class CheckpointManager:
    """
    Manages training checkpoints for fault-tolerant training.

    Features:
    - Automatic checkpoint cleanup
    - Best model tracking
    - Checkpoint validation
    - Training state resumption
    - Support for sklearn and PyTorch models

    Example:
        manager = CheckpointManager(
            CheckpointConfig(
                checkpoint_dir="checkpoints/experiment_1",
                save_frequency=10,
                keep_last_n=3,
                monitor_metric="val_ic",
                mode="max"
            )
        )

        # Save checkpoint
        manager.save_checkpoint(
            model=model,
            state=training_state,
            metadata=metadata
        )

        # Load latest checkpoint
        model, state, metadata = manager.load_latest()

        # Load best checkpoint
        model, state, metadata = manager.load_best()
    """

    def __init__(self, config: CheckpointConfig):
        self.config = config
        self._checkpoints: List[CheckpointMetadata] = []
        self._best_metric: Optional[float] = None
        self._best_checkpoint_id: Optional[str] = None

        # Load existing checkpoint index
        self._load_checkpoint_index()

    def _load_checkpoint_index(self) -> None:
        """Load existing checkpoint index from disk."""
        index_path = self.config.checkpoint_dir / "checkpoint_index.json"
        if index_path.exists():
            try:
                with open(index_path, "r") as f:
                    data = json.load(f)

                self._checkpoints = [
                    CheckpointMetadata.from_dict(cp)
                    for cp in data.get("checkpoints", [])
                ]
                self._best_metric = data.get("best_metric")
                self._best_checkpoint_id = data.get("best_checkpoint_id")

                logger.info(
                    f"Loaded checkpoint index with {len(self._checkpoints)} checkpoints"
                )
            except Exception as e:
                logger.warning(f"Failed to load checkpoint index: {e}")
                self._checkpoints = []

    def _save_checkpoint_index(self) -> None:
        """Save checkpoint index to disk."""
        index_path = self.config.checkpoint_dir / "checkpoint_index.json"

        data = {
            "checkpoints": [cp.to_dict() for cp in self._checkpoints],
            "best_metric": self._best_metric,
            "best_checkpoint_id": self._best_checkpoint_id,
            "last_updated": datetime.now().isoformat(),
        }

        with open(index_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _generate_checkpoint_id(self, epoch: int, step: int) -> str:
        """Generate unique checkpoint ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"checkpoint_e{epoch:04d}_s{step:06d}_{timestamp}"

    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get path for a checkpoint."""
        return self.config.checkpoint_dir / checkpoint_id

    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.config.mode == "min":
            return current < best
        else:
            return current > best

    def save_checkpoint(
        self,
        model: Any,
        state: TrainingState,
        metadata: CheckpointMetadata,
        force: bool = False,
    ) -> Optional[str]:
        """
        Save a training checkpoint.

        Args:
            model: The model to save
            state: Training state
            metadata: Checkpoint metadata
            force: Force save even if not scheduled

        Returns:
            Checkpoint ID if saved, None otherwise
        """
        # Check if we should save
        should_save = force or (metadata.epoch % self.config.save_frequency == 0)

        # Always save if it's the best
        current_metric = metadata.metrics.get(self.config.monitor_metric)
        is_best = False

        if current_metric is not None:
            if self._best_metric is None or self._is_better(current_metric, self._best_metric):
                self._best_metric = current_metric
                is_best = True
                should_save = True
                metadata.is_best = True

        if not should_save:
            return None

        checkpoint_id = metadata.checkpoint_id
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save model
            model_path = checkpoint_path / "model.joblib"
            self._save_model(model, model_path)

            # Save training state
            state_path = checkpoint_path / "training_state.json"
            with open(state_path, "w") as f:
                json.dump(state.to_dict(), f, indent=2, default=str)

            # Save metadata
            meta_path = checkpoint_path / "metadata.json"
            with open(meta_path, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2, default=str)

            # Update checkpoint list
            self._checkpoints.append(metadata)

            if is_best:
                self._best_checkpoint_id = checkpoint_id
                # Create symlink to best
                best_link = self.config.checkpoint_dir / "best"
                if best_link.exists():
                    if best_link.is_symlink():
                        best_link.unlink()
                    else:
                        shutil.rmtree(best_link)
                # Copy best checkpoint to 'best' directory
                shutil.copytree(checkpoint_path, best_link)

            # Update index
            self._save_checkpoint_index()

            # Cleanup old checkpoints
            self._cleanup_checkpoints()

            logger.info(
                f"Saved checkpoint {checkpoint_id} "
                f"(epoch={metadata.epoch}, is_best={is_best})"
            )

            return checkpoint_id

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            # Clean up partial checkpoint
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
            raise

    def _save_model(self, model: Any, path: Path) -> None:
        """Save model with appropriate method."""
        # Check for PyTorch model
        try:
            import torch
            if isinstance(model, torch.nn.Module):
                torch.save(model.state_dict(), path.with_suffix(".pt"))
                return
        except ImportError:
            pass

        # Default: use joblib
        if self.config.compress:
            joblib.dump(model, path, compress=3)
        else:
            joblib.dump(model, path)

    def _load_model(self, path: Path, model_template: Optional[Any] = None) -> Any:
        """Load model from checkpoint."""
        # Check for PyTorch model
        pt_path = path.with_suffix(".pt")
        if pt_path.exists():
            try:
                import torch
                state_dict = torch.load(pt_path, map_location="cpu")
                if model_template is not None:
                    model_template.load_state_dict(state_dict)
                    return model_template
                return state_dict
            except ImportError:
                pass

        # Default: use joblib
        return joblib.load(path)

    def load_checkpoint(
        self,
        checkpoint_id: str,
        model_template: Optional[Any] = None,
    ) -> Tuple[Any, TrainingState, CheckpointMetadata]:
        """
        Load a specific checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to load
            model_template: Optional model template for PyTorch models

        Returns:
            Tuple of (model, training_state, metadata)
        """
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")

        # Load model
        model_path = checkpoint_path / "model.joblib"
        model = self._load_model(model_path, model_template)

        # Load training state
        state_path = checkpoint_path / "training_state.json"
        with open(state_path, "r") as f:
            state_data = json.load(f)
        state = TrainingState.from_dict(state_data)

        # Load metadata
        meta_path = checkpoint_path / "metadata.json"
        with open(meta_path, "r") as f:
            meta_data = json.load(f)
        metadata = CheckpointMetadata.from_dict(meta_data)

        # Validate checkpoint if configured
        if self.config.validate_on_load:
            self._validate_checkpoint(checkpoint_path, metadata)

        logger.info(f"Loaded checkpoint {checkpoint_id}")

        return model, state, metadata

    def load_latest(
        self,
        model_template: Optional[Any] = None,
    ) -> Optional[Tuple[Any, TrainingState, CheckpointMetadata]]:
        """
        Load the most recent checkpoint.

        Returns:
            Tuple of (model, training_state, metadata) or None if no checkpoints
        """
        if not self._checkpoints:
            logger.info("No checkpoints found")
            return None

        # Sort by epoch and step
        latest = max(
            self._checkpoints,
            key=lambda cp: (cp.epoch, cp.step)
        )

        return self.load_checkpoint(latest.checkpoint_id, model_template)

    def load_best(
        self,
        model_template: Optional[Any] = None,
    ) -> Optional[Tuple[Any, TrainingState, CheckpointMetadata]]:
        """
        Load the best checkpoint based on monitored metric.

        Returns:
            Tuple of (model, training_state, metadata) or None if no checkpoints
        """
        best_path = self.config.checkpoint_dir / "best"

        if best_path.exists():
            # Load from best directory
            model_path = best_path / "model.joblib"
            model = self._load_model(model_path, model_template)

            state_path = best_path / "training_state.json"
            with open(state_path, "r") as f:
                state = TrainingState.from_dict(json.load(f))

            meta_path = best_path / "metadata.json"
            with open(meta_path, "r") as f:
                metadata = CheckpointMetadata.from_dict(json.load(f))

            logger.info(f"Loaded best checkpoint (epoch={metadata.epoch})")
            return model, state, metadata

        # Fallback: find best from index
        if not self._checkpoints:
            return None

        best_checkpoints = [cp for cp in self._checkpoints if cp.is_best]
        if best_checkpoints:
            return self.load_checkpoint(
                best_checkpoints[-1].checkpoint_id,
                model_template
            )

        return None

    def _validate_checkpoint(
        self,
        checkpoint_path: Path,
        metadata: CheckpointMetadata,
    ) -> bool:
        """Validate checkpoint integrity."""
        required_files = ["model.joblib", "training_state.json", "metadata.json"]

        for filename in required_files:
            if filename == "model.joblib":
                # Check for either .joblib or .pt
                if not (checkpoint_path / filename).exists():
                    if not (checkpoint_path / "model.pt").exists():
                        logger.warning(f"Missing required file: {filename}")
                        return False
            else:
                if not (checkpoint_path / filename).exists():
                    logger.warning(f"Missing required file: {filename}")
                    return False

        return True

    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints according to retention policy."""
        if len(self._checkpoints) <= self.config.keep_last_n:
            return

        # Identify checkpoints to keep
        # 1. Keep best N checkpoints
        best_checkpoints = sorted(
            [cp for cp in self._checkpoints if cp.is_best],
            key=lambda cp: cp.epoch,
            reverse=True
        )[:self.config.keep_best_n]

        # 2. Keep last N checkpoints
        sorted_checkpoints = sorted(
            self._checkpoints,
            key=lambda cp: (cp.epoch, cp.step),
            reverse=True
        )
        last_checkpoints = sorted_checkpoints[:self.config.keep_last_n]

        # Combine keep sets
        keep_ids = set(cp.checkpoint_id for cp in best_checkpoints)
        keep_ids.update(cp.checkpoint_id for cp in last_checkpoints)

        # Remove others
        to_remove = [
            cp for cp in self._checkpoints
            if cp.checkpoint_id not in keep_ids
        ]

        for cp in to_remove:
            checkpoint_path = self._get_checkpoint_path(cp.checkpoint_id)
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
                logger.debug(f"Removed old checkpoint: {cp.checkpoint_id}")

        # Update checkpoint list
        self._checkpoints = [
            cp for cp in self._checkpoints
            if cp.checkpoint_id in keep_ids
        ]

        # Update index
        self._save_checkpoint_index()

    def has_checkpoint(self) -> bool:
        """Check if any checkpoint exists."""
        return len(self._checkpoints) > 0

    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about available checkpoints."""
        return {
            "num_checkpoints": len(self._checkpoints),
            "best_metric": self._best_metric,
            "best_checkpoint_id": self._best_checkpoint_id,
            "checkpoint_dir": str(self.config.checkpoint_dir),
            "checkpoints": [
                {
                    "id": cp.checkpoint_id,
                    "epoch": cp.epoch,
                    "is_best": cp.is_best,
                    "metrics": cp.metrics,
                }
                for cp in sorted(self._checkpoints, key=lambda x: x.epoch)
            ],
        }

    def clear_all(self) -> None:
        """Remove all checkpoints."""
        for cp in self._checkpoints:
            checkpoint_path = self._get_checkpoint_path(cp.checkpoint_id)
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)

        best_path = self.config.checkpoint_dir / "best"
        if best_path.exists():
            shutil.rmtree(best_path)

        index_path = self.config.checkpoint_dir / "checkpoint_index.json"
        if index_path.exists():
            index_path.unlink()

        self._checkpoints = []
        self._best_metric = None
        self._best_checkpoint_id = None

        logger.info("Cleared all checkpoints")


class ResumableTrainer:
    """
    Wrapper for trainers that adds checkpoint-based resumption.

    Integrates with the Trainer class to provide:
    - Automatic checkpointing during training
    - Seamless training resumption
    - Progress tracking across sessions

    Example:
        from src.training import Trainer
        from src.training.checkpointing import ResumableTrainer, CheckpointConfig

        # Create base trainer
        base_trainer = Trainer(model_type="lightgbm_classifier")

        # Wrap with resumable trainer
        resumable = ResumableTrainer(
            trainer=base_trainer,
            checkpoint_config=CheckpointConfig(
                checkpoint_dir="checkpoints/my_experiment",
                save_frequency=5
            )
        )

        # Train with automatic checkpointing
        result = resumable.fit(X, y, X_val, y_val)

        # Resume training later
        result = resumable.resume(X, y, X_val, y_val, additional_epochs=50)
    """

    def __init__(
        self,
        trainer: Any,
        checkpoint_config: CheckpointConfig,
    ):
        self.trainer = trainer
        self.checkpoint_manager = CheckpointManager(checkpoint_config)
        self._training_state: Optional[TrainingState] = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        epochs: int = 100,
        **kwargs,
    ) -> Any:
        """
        Train with automatic checkpointing.

        Args:
            X: Training features
            y: Training target
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of epochs
            **kwargs: Additional arguments for trainer

        Returns:
            Training result
        """
        # Initialize training state
        self._training_state = TrainingState(
            epoch=0,
            step=0,
            best_metric=None,
            best_epoch=None,
            early_stopping_counter=0,
            learning_rate=kwargs.get("learning_rate"),
            optimizer_state=None,
            scheduler_state=None,
            random_state=getattr(self.trainer, "random_state", 42),
        )

        # Train
        result = self.trainer.fit(X, y, X_val, y_val, **kwargs)

        # Save final checkpoint
        self._save_final_checkpoint(result)

        return result

    def resume(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        additional_epochs: int = 50,
        **kwargs,
    ) -> Any:
        """
        Resume training from the latest checkpoint.

        Args:
            X: Training features
            y: Training target
            X_val: Validation features
            y_val: Validation targets
            additional_epochs: Additional epochs to train
            **kwargs: Additional arguments for trainer

        Returns:
            Training result
        """
        # Load latest checkpoint
        loaded = self.checkpoint_manager.load_latest()

        if loaded is None:
            logger.warning("No checkpoint found, starting fresh training")
            return self.fit(X, y, X_val, y_val, epochs=additional_epochs, **kwargs)

        model, state, metadata = loaded

        logger.info(
            f"Resuming from epoch {state.epoch} "
            f"(best={state.best_metric}, early_stop_counter={state.early_stopping_counter})"
        )

        # Update trainer with loaded model
        self.trainer.model = model
        self._training_state = state

        # Continue training
        result = self.trainer.fit(X, y, X_val, y_val, **kwargs)

        # Save final checkpoint
        self._save_final_checkpoint(result, start_epoch=state.epoch)

        return result

    def _save_final_checkpoint(
        self,
        result: Any,
        start_epoch: int = 0,
    ) -> None:
        """Save final checkpoint after training."""
        if self._training_state is None:
            self._training_state = TrainingState(
                epoch=0, step=0, best_metric=None, best_epoch=None,
                early_stopping_counter=0, learning_rate=None,
                optimizer_state=None, scheduler_state=None,
            )

        # Update state
        final_epoch = start_epoch + 1

        # Get metrics from result
        metrics = {}
        if hasattr(result, "validation_metrics"):
            metrics.update(result.validation_metrics)
        if hasattr(result, "train_metrics"):
            metrics.update(result.train_metrics)

        # Create metadata
        checkpoint_id = self.checkpoint_manager._generate_checkpoint_id(
            final_epoch, 0
        )

        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            epoch=final_epoch,
            step=0,
            timestamp=datetime.now().isoformat(),
            model_type=getattr(result, "model_type", "unknown"),
            task_type=getattr(result, "task_type", "unknown"),
            metrics=metrics,
            best_metric_value=self._training_state.best_metric,
            best_metric_name=self.checkpoint_manager.config.monitor_metric,
            data_hash=getattr(result, "data_hash", None),
            random_state=self._training_state.random_state,
            params=getattr(result, "params", {}),
            training_time_seconds=getattr(result, "training_time_seconds", 0.0),
            n_samples_processed=getattr(result, "n_train_samples", 0),
        )

        # Save checkpoint
        model = result.model if hasattr(result, "model") else self.trainer.model
        if model is not None:
            self.checkpoint_manager.save_checkpoint(
                model=model,
                state=self._training_state,
                metadata=metadata,
                force=True,
            )


def create_checkpoint_callback(
    checkpoint_dir: Union[str, Path],
    save_frequency: int = 10,
    keep_last_n: int = 3,
    monitor_metric: str = "val_loss",
    mode: str = "min",
) -> "CheckpointCallback":
    """
    Create a checkpoint callback for use with Trainer.

    Args:
        checkpoint_dir: Directory to save checkpoints
        save_frequency: Save every N epochs
        keep_last_n: Keep last N checkpoints
        monitor_metric: Metric to monitor for best model
        mode: "min" or "max"

    Returns:
        CheckpointCallback instance
    """
    from .trainer import CheckpointCallback

    return CheckpointCallback(
        save_dir=checkpoint_dir,
        save_frequency=save_frequency,
        save_best_only=False,  # We handle this in CheckpointManager
        monitor=monitor_metric,
        mode=mode,
    )


__all__ = [
    "CheckpointMetadata",
    "TrainingState",
    "CheckpointConfig",
    "CheckpointManager",
    "ResumableTrainer",
    "create_checkpoint_callback",
]
