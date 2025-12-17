"""
Main Trainer class for orchestrating ML model training.

This module provides a unified training interface with:
- Cross-validation strategies
- Early stopping
- Callback system
- Result tracking
- Checkpoint management

Designed for JPMorgan-level requirements:
- Reproducible training pipelines
- Comprehensive logging
- Memory-efficient processing
- Production deployment support
"""

from __future__ import annotations

import gc
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import joblib

from .model_factory import ModelFactory, ModelType, TaskType
from .experiment_tracker import ExperimentTracker, RunMetrics


logger = logging.getLogger(__name__)


def _compute_data_hash(X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> str:
    """
    Compute a deterministic hash of the training data.

    Used for reproducibility tracking - ensures we know exactly
    which data was used to train a model.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values

    # Create hash from data statistics (faster than hashing all data)
    data_repr = (
        f"shape={X.shape},"
        f"X_mean={np.nanmean(X):.6f},"
        f"X_std={np.nanstd(X):.6f},"
        f"X_min={np.nanmin(X):.6f},"
        f"X_max={np.nanmax(X):.6f},"
        f"y_mean={np.nanmean(y):.6f},"
        f"y_unique={len(np.unique(y[~np.isnan(y)]))},"
        f"first_row={X[0, :5].tobytes().hex()[:32]},"
        f"last_row={X[-1, :5].tobytes().hex()[:32]}"
    )
    return hashlib.sha256(data_repr.encode()).hexdigest()[:16]


def _validate_data(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    check_inf: bool = True,
    check_nan: bool = True,
) -> Dict[str, Any]:
    """
    Validate training data and return diagnostics.

    Checks for common data quality issues:
    - NaN values
    - Infinite values
    - Constant features
    - Highly correlated features

    Returns:
        Dictionary with validation results and warnings
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
        feature_names = list(X.columns)
    else:
        X_arr = X
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    if isinstance(y, (pd.Series, pd.DataFrame)):
        y_arr = y.values
    else:
        y_arr = y

    warnings = []
    stats = {}

    # Check for NaN
    if check_nan:
        nan_mask = np.isnan(X_arr)
        nan_count = nan_mask.sum()
        nan_features = nan_mask.any(axis=0)

        if nan_count > 0:
            nan_pct = nan_count / X_arr.size * 100
            warnings.append(
                f"Data contains {nan_count} NaN values ({nan_pct:.2f}%). "
                f"Affected features: {[feature_names[i] for i, v in enumerate(nan_features) if v][:5]}"
            )
        stats['nan_count'] = int(nan_count)
        stats['nan_features'] = int(nan_features.sum())

    # Check for Inf
    if check_inf:
        inf_mask = np.isinf(X_arr)
        inf_count = inf_mask.sum()

        if inf_count > 0:
            warnings.append(
                f"Data contains {inf_count} Inf values. "
                f"Use Pipeline with InfinityHandler."
            )
        stats['inf_count'] = int(inf_count)

    # Check for constant features
    stds = np.nanstd(X_arr, axis=0)
    constant_features = stds == 0
    if constant_features.any():
        warnings.append(
            f"{constant_features.sum()} constant features detected. "
            f"Consider removing: {[feature_names[i] for i, v in enumerate(constant_features) if v][:5]}"
        )
    stats['constant_features'] = int(constant_features.sum())

    # Check target variable
    if np.isnan(y_arr).any():
        warnings.append("Target variable contains NaN values!")

    unique_targets = len(np.unique(y_arr[~np.isnan(y_arr)]))
    stats['unique_targets'] = unique_targets

    # Log warnings
    for warning in warnings:
        logger.warning(warning)

    return {
        'valid': len(warnings) == 0,
        'warnings': warnings,
        'stats': stats,
    }


@dataclass
class TrainingResult:
    """Container for training results."""
    model: Any
    model_type: str
    task_type: str
    train_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    cv_scores: Optional[Dict[str, List[float]]] = None
    feature_importance: Optional[pd.DataFrame] = None
    training_time_seconds: float = 0.0
    n_train_samples: int = 0
    n_features: int = 0
    best_iteration: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Feature Importance Stability (across CV folds)
    cv_feature_importances: Optional[List[pd.DataFrame]] = None
    data_hash: Optional[str] = None
    data_validation: Optional[Dict[str, Any]] = None

    def get_cv_mean(self, metric: str) -> Optional[float]:
        """Get mean CV score for a metric."""
        if self.cv_scores and metric in self.cv_scores:
            return np.mean(self.cv_scores[metric])
        return None

    def get_cv_std(self, metric: str) -> Optional[float]:
        """Get std of CV scores for a metric."""
        if self.cv_scores and metric in self.cv_scores:
            return np.std(self.cv_scores[metric])
        return None

    def get_stable_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance with stability metrics across CV folds.

        Returns DataFrame with:
        - mean_importance: Average importance across folds
        - std_importance: Standard deviation across folds
        - cv_importance: Coefficient of variation (std/mean)
        - stability_rank: Rank based on CV (lower = more stable)

        Features with low CV are more stable/reliable.
        """
        if not self.cv_feature_importances or len(self.cv_feature_importances) == 0:
            return self.feature_importance

        # Stack all fold importances
        all_importances = []
        for fold_df in self.cv_feature_importances:
            if fold_df is not None and 'importance' in fold_df.columns:
                all_importances.append(
                    fold_df.set_index('feature')['importance']
                )

        if len(all_importances) == 0:
            return self.feature_importance

        # Combine into DataFrame
        importance_matrix = pd.concat(all_importances, axis=1)
        importance_matrix.columns = [f"fold_{i}" for i in range(len(all_importances))]

        # Calculate stability metrics
        result = pd.DataFrame({
            'feature': importance_matrix.index,
            'mean_importance': importance_matrix.mean(axis=1).values,
            'std_importance': importance_matrix.std(axis=1).values,
            'min_importance': importance_matrix.min(axis=1).values,
            'max_importance': importance_matrix.max(axis=1).values,
        })

        # Coefficient of variation (lower = more stable)
        result['cv_importance'] = result['std_importance'] / (result['mean_importance'] + 1e-10)

        # Stability rank (1 = most stable)
        result['stability_rank'] = result['cv_importance'].rank().astype(int)

        # Sort by mean importance
        result = result.sort_values('mean_importance', ascending=False)

        # Add normalized importance
        result['importance_normalized'] = (
            result['mean_importance'] / result['mean_importance'].sum()
        )

        return result

    def get_top_stable_features(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top features that are both important AND stable.

        Uses a combined score of importance and stability.
        """
        stable_importance = self.get_stable_feature_importance()
        if stable_importance is None:
            return None

        # Combined score: high importance + low CV
        stable_importance['stability_score'] = (
            stable_importance['importance_normalized'] *
            (1 / (stable_importance['cv_importance'] + 0.1))
        )

        return stable_importance.nlargest(top_n, 'stability_score')

    def summary(self) -> str:
        """Generate text summary of results."""
        lines = [
            "=" * 60,
            "TRAINING RESULTS",
            "=" * 60,
            f"Model: {self.model_type}",
            f"Task: {self.task_type}",
            f"Training time: {self.training_time_seconds:.2f}s",
            f"Train samples: {self.n_train_samples:,}",
            f"Features: {self.n_features}",
        ]

        if self.data_hash:
            lines.append(f"Data hash: {self.data_hash}")

        lines.append("")
        lines.append("Validation Metrics:")

        for metric, value in self.validation_metrics.items():
            cv_mean = self.get_cv_mean(metric)
            cv_std = self.get_cv_std(metric)
            if cv_mean is not None:
                lines.append(f"  {metric}: {value:.4f} (CV: {cv_mean:.4f} +/- {cv_std:.4f})")
            else:
                lines.append(f"  {metric}: {value:.4f}")

        # Add feature importance stability summary
        if self.cv_feature_importances:
            stable = self.get_stable_feature_importance()
            if stable is not None:
                lines.append("")
                lines.append("Top 5 Stable Features:")
                top5 = self.get_top_stable_features(5)
                if top5 is not None:
                    for _, row in top5.iterrows():
                        lines.append(
                            f"  {row['feature']}: {row['mean_importance']:.4f} "
                            f"(CV: {row['cv_importance']:.2f})"
                        )

        lines.append("=" * 60)
        return "\n".join(lines)


class Callback(ABC):
    """Base class for training callbacks."""

    def on_train_begin(self, trainer: "Trainer", **kwargs) -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self, trainer: "Trainer", result: TrainingResult, **kwargs) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, trainer: "Trainer", epoch: int, **kwargs) -> None:
        """Called at the start of each epoch/iteration."""
        pass

    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        metrics: Dict[str, float],
        **kwargs
    ) -> bool:
        """
        Called at the end of each epoch/iteration.

        Returns:
            True to continue training, False to stop
        """
        return True

    def on_fold_begin(self, trainer: "Trainer", fold: int, **kwargs) -> None:
        """Called at the start of each CV fold."""
        pass

    def on_fold_end(
        self,
        trainer: "Trainer",
        fold: int,
        metrics: Dict[str, float],
        **kwargs
    ) -> None:
        """Called at the end of each CV fold."""
        pass


class EarlyStoppingCallback(Callback):
    """
    Early stopping callback to prevent overfitting.

    Monitors a metric and stops training if no improvement
    is observed for a specified number of iterations.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
        restore_best_weights: bool = True,
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.best_value: Optional[float] = None
        self.best_weights: Optional[Any] = None
        self.counter = 0
        self.stopped_epoch = 0

    def on_train_begin(self, trainer: "Trainer", **kwargs) -> None:
        self.best_value = None
        self.best_weights = None
        self.counter = 0

    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        metrics: Dict[str, float],
        **kwargs
    ) -> bool:
        current = metrics.get(self.monitor)

        if current is None:
            return True

        if self.best_value is None:
            self.best_value = current
            return True

        if self.mode == "min":
            improved = current < self.best_value - self.min_delta
        else:
            improved = current > self.best_value + self.min_delta

        if improved:
            self.best_value = current
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = self._get_weights(trainer.model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stopped_epoch = epoch
            logger.info(f"Early stopping at epoch {epoch}")

            if self.restore_best_weights and self.best_weights is not None:
                self._set_weights(trainer.model, self.best_weights)

            return False

        return True

    def _get_weights(self, model: Any) -> Any:
        """Get model weights/state."""
        if hasattr(model, "get_params"):
            return model.get_params()
        return None

    def _set_weights(self, model: Any, weights: Any) -> None:
        """Set model weights/state."""
        if hasattr(model, "set_params"):
            model.set_params(**weights)


class CheckpointCallback(Callback):
    """Save model checkpoints during training."""

    def __init__(
        self,
        save_dir: Union[str, Path],
        save_frequency: int = 10,
        save_best_only: bool = True,
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        self.save_dir = Path(save_dir)
        self.save_frequency = save_frequency
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.best_value: Optional[float] = None

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        metrics: Dict[str, float],
        **kwargs
    ) -> bool:
        current = metrics.get(self.monitor)

        should_save = False

        if self.save_best_only:
            if current is not None:
                if self.best_value is None:
                    self.best_value = current
                    should_save = True
                elif self.mode == "min" and current < self.best_value:
                    self.best_value = current
                    should_save = True
                elif self.mode == "max" and current > self.best_value:
                    self.best_value = current
                    should_save = True
        elif epoch % self.save_frequency == 0:
            should_save = True

        if should_save:
            checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.joblib"
            joblib.dump(trainer.model, checkpoint_path)
            logger.debug(f"Saved checkpoint to {checkpoint_path}")

        return True


class LoggingCallback(Callback):
    """Log training progress."""

    def __init__(self, log_frequency: int = 10):
        self.log_frequency = log_frequency
        self.start_time: Optional[float] = None

    def on_train_begin(self, trainer: "Trainer", **kwargs) -> None:
        self.start_time = time.time()
        logger.info("Training started")

    def on_train_end(self, trainer: "Trainer", result: TrainingResult, **kwargs) -> None:
        elapsed = time.time() - self.start_time if self.start_time else 0
        logger.info(f"Training completed in {elapsed:.2f}s")

    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        metrics: Dict[str, float],
        **kwargs
    ) -> bool:
        if epoch % self.log_frequency == 0:
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            logger.info(f"Epoch {epoch}: {metric_str}")
        return True


class Trainer:
    """
    Main training orchestrator for ML models.

    Provides unified interface for:
    - Model training with various strategies
    - Cross-validation
    - Early stopping
    - Result tracking
    - Experiment logging

    Example:
        from src.training import Trainer, ModelFactory

        # Create trainer
        trainer = Trainer(
            model_type=ModelType.LIGHTGBM_CLASSIFIER,
            params={"n_estimators": 100},
            callbacks=[
                EarlyStoppingCallback(patience=10),
                CheckpointCallback("checkpoints/")
            ]
        )

        # Train with cross-validation
        result = trainer.fit(X_train, y_train, X_val, y_val)

        # Predict
        predictions = trainer.predict(X_test)
    """

    def __init__(
        self,
        model_type: Union[ModelType, str] = ModelType.LIGHTGBM_CLASSIFIER,
        params: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Callback]] = None,
        experiment_tracker: Optional[ExperimentTracker] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        gpu_enabled: bool = False,
    ):
        if isinstance(model_type, str):
            model_type = ModelType(model_type)

        self.model_type = model_type
        self.task_type = ModelFactory.get_task_type(model_type)
        self.params = params or {}
        self.callbacks = callbacks or [LoggingCallback()]
        self.tracker = experiment_tracker
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.gpu_enabled = gpu_enabled

        self.model: Optional[Any] = None
        self._feature_names: Optional[List[str]] = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        sample_weights: Optional[np.ndarray] = None,
        val_sample_weights: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        validate_data: bool = True,
    ) -> TrainingResult:
        """
        Train the model.

        Args:
            X: Training features
            y: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            sample_weights: Training sample weights
            val_sample_weights: Validation sample weights
            feature_names: Feature names
            categorical_features: Categorical feature names
            validate_data: Whether to validate data before training

        Returns:
            TrainingResult with model and metrics
        """
        start_time = time.time()

        # Compute data hash for reproducibility
        data_hash = _compute_data_hash(X, y)
        logger.info(f"Data hash: {data_hash}")

        # Validate data
        data_validation = None
        if validate_data:
            data_validation = _validate_data(X, y)
            if data_validation['warnings']:
                logger.warning(
                    f"Data validation warnings: {len(data_validation['warnings'])}. "
                    "Consider using Pipeline for preprocessing."
                )

        # Store feature names
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
        elif feature_names:
            self._feature_names = feature_names

        # Create model
        self.model = ModelFactory.create_model(
            self.model_type,
            params=self.params,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            gpu_enabled=self.gpu_enabled,
        )

        # Trigger callbacks
        for callback in self.callbacks:
            callback.on_train_begin(self)

        # Prepare fit kwargs
        fit_kwargs = {}

        if sample_weights is not None:
            fit_kwargs["sample_weight"] = sample_weights

        # Handle evaluation set for gradient boosting
        if X_val is not None and y_val is not None:
            fit_kwargs = self._prepare_eval_set(
                fit_kwargs, X_val, y_val, val_sample_weights
            )

        # Handle categorical features
        if categorical_features:
            fit_kwargs = self._prepare_categorical(
                fit_kwargs, X, categorical_features
            )

        # Fit the model
        logger.info(f"Training {self.model_type.value} on {len(X)} samples")

        try:
            self.model.fit(X, y, **fit_kwargs)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        # Calculate metrics
        train_metrics = self._calculate_metrics(X, y, prefix="train_")

        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_metrics = self._calculate_metrics(X_val, y_val, prefix="val_")

        # Get feature importance
        feature_importance = None
        if self._feature_names:
            try:
                feature_importance = ModelFactory.get_feature_importance(
                    self.model, self._feature_names
                )
            except Exception as e:
                logger.warning(f"Could not extract feature importance: {e}")

        # Get best iteration if available
        best_iteration = None
        if hasattr(self.model, "best_iteration_"):
            best_iteration = self.model.best_iteration_
        elif hasattr(self.model, "best_iteration"):
            best_iteration = self.model.best_iteration

        # Create result
        training_time = time.time() - start_time
        result = TrainingResult(
            model=self.model,
            model_type=self.model_type.value,
            task_type=self.task_type.value,
            train_metrics=train_metrics,
            validation_metrics=val_metrics,
            feature_importance=feature_importance,
            training_time_seconds=training_time,
            n_train_samples=len(X),
            n_features=X.shape[1] if hasattr(X, "shape") else len(X[0]),
            best_iteration=best_iteration,
            params=self.params,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "random_state": self.random_state,
            },
            data_hash=data_hash,
            data_validation=data_validation,
        )

        # Trigger callbacks
        for callback in self.callbacks:
            callback.on_train_end(self, result)

        # Log to experiment tracker
        if self.tracker:
            self._log_to_tracker(result)

        logger.info(f"Training completed in {training_time:.2f}s")

        return result

    def fit_cv(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv,
        sample_weights: Optional[np.ndarray] = None,
        return_estimator: bool = False,
        collect_feature_importance: bool = True,
        validate_data: bool = True,
    ) -> TrainingResult:
        """
        Train with cross-validation.

        Includes memory management (gc.collect after each fold) and
        feature importance stability analysis across folds.

        Args:
            X: Features
            y: Target
            cv: Cross-validation splitter
            sample_weights: Sample weights
            return_estimator: If True, train final model on all data
            collect_feature_importance: Collect importance from each fold
            validate_data: Whether to validate data before training

        Returns:
            TrainingResult with CV scores and feature importance stability
        """
        start_time = time.time()

        # Compute data hash for reproducibility
        data_hash = _compute_data_hash(X, y)
        logger.info(f"Data hash: {data_hash}")

        # Validate data
        data_validation = None
        if validate_data:
            data_validation = _validate_data(X, y)

        # Store feature names
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X_arr = X.values
        else:
            X_arr = X

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_arr = y.values
        else:
            y_arr = y

        # Determine scoring based on task type
        if self.task_type == TaskType.CLASSIFICATION:
            scoring_dict = {
                'accuracy': 'accuracy',
                'roc_auc': 'roc_auc',
                'f1': 'f1',
            }
        else:
            scoring_dict = {
                'r2': 'r2',
                'neg_mse': 'neg_mean_squared_error',
                'neg_mae': 'neg_mean_absolute_error',
            }

        # Manual CV loop for memory management and feature importance
        n_splits = cv.get_n_splits(X_arr)
        logger.info(f"Running {n_splits}-fold cross-validation with memory management")

        cv_scores = {metric: [] for metric in scoring_dict.keys()}
        cv_feature_importances = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_arr, y_arr)):
            logger.info(f"Training fold {fold_idx + 1}/{n_splits}")

            # Trigger fold callbacks
            for callback in self.callbacks:
                callback.on_fold_begin(self, fold_idx)

            # Create fold data
            X_train_fold = X_arr[train_idx]
            y_train_fold = y_arr[train_idx]
            X_val_fold = X_arr[val_idx]
            y_val_fold = y_arr[val_idx]

            # Create model for this fold
            fold_model = ModelFactory.create_model(
                self.model_type,
                params=self.params,
                random_state=self.random_state + fold_idx,  # Different seed per fold
                n_jobs=self.n_jobs,
                gpu_enabled=self.gpu_enabled,
            )

            # Fit
            fit_kwargs = {}
            if sample_weights is not None:
                fit_kwargs["sample_weight"] = sample_weights[train_idx]

            fold_model.fit(X_train_fold, y_train_fold, **fit_kwargs)

            # Evaluate
            fold_metrics = self._evaluate_fold(
                fold_model, X_val_fold, y_val_fold, scoring_dict
            )
            for metric, value in fold_metrics.items():
                cv_scores[metric].append(value)

            # Collect feature importance from this fold
            if collect_feature_importance and self._feature_names:
                try:
                    fold_importance = ModelFactory.get_feature_importance(
                        fold_model, self._feature_names
                    )
                    cv_feature_importances.append(fold_importance)
                except Exception as e:
                    logger.warning(f"Could not get feature importance for fold {fold_idx}: {e}")

            # Trigger fold end callbacks
            for callback in self.callbacks:
                callback.on_fold_end(self, fold_idx, fold_metrics)

            # Memory management: delete fold model and collect garbage
            del fold_model
            del X_train_fold, y_train_fold, X_val_fold, y_val_fold
            gc.collect()

            logger.debug(f"Fold {fold_idx + 1} complete, memory cleaned")

        # Calculate aggregated metrics
        val_metrics = {}
        for metric, scores in cv_scores.items():
            val_metrics[f"cv_{metric}_mean"] = np.mean(scores)
            val_metrics[f"cv_{metric}_std"] = np.std(scores)

        # Train final model if requested
        if return_estimator:
            logger.info("Training final model on all data")
            self.model = ModelFactory.create_model(
                self.model_type,
                params=self.params,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                gpu_enabled=self.gpu_enabled,
            )
            fit_kwargs = {}
            if sample_weights is not None:
                fit_kwargs["sample_weight"] = sample_weights
            self.model.fit(X_arr, y_arr, **fit_kwargs)
        else:
            self.model = None

        # Get feature importance from final model
        feature_importance = None
        if self.model and self._feature_names:
            try:
                feature_importance = ModelFactory.get_feature_importance(
                    self.model, self._feature_names
                )
            except Exception:
                pass

        training_time = time.time() - start_time

        result = TrainingResult(
            model=self.model,
            model_type=self.model_type.value,
            task_type=self.task_type.value,
            train_metrics={},
            validation_metrics=val_metrics,
            cv_scores=cv_scores,
            feature_importance=feature_importance,
            training_time_seconds=training_time,
            n_train_samples=len(X),
            n_features=X_arr.shape[1],
            params=self.params,
            metadata={
                "cv_folds": n_splits,
                "timestamp": datetime.now().isoformat(),
            },
            cv_feature_importances=cv_feature_importances if collect_feature_importance else None,
            data_hash=data_hash,
            data_validation=data_validation,
        )

        # Log to tracker
        if self.tracker:
            self._log_to_tracker(result)

        # Final garbage collection
        gc.collect()

        return result

    def _evaluate_fold(
        self,
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray,
        scoring_dict: Dict[str, str],
    ) -> Dict[str, float]:
        """Evaluate model on validation fold."""
        from sklearn import metrics as sklearn_metrics

        predictions = model.predict(X_val)
        result = {}

        for name, scorer in scoring_dict.items():
            try:
                if scorer == 'accuracy':
                    result[name] = sklearn_metrics.accuracy_score(y_val, predictions)
                elif scorer == 'roc_auc':
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_val)
                        if proba.shape[1] == 2:
                            result[name] = sklearn_metrics.roc_auc_score(y_val, proba[:, 1])
                        else:
                            result[name] = sklearn_metrics.roc_auc_score(
                                y_val, proba, multi_class='ovr'
                            )
                    else:
                        result[name] = 0.0
                elif scorer == 'f1':
                    result[name] = sklearn_metrics.f1_score(
                        y_val, predictions, average='weighted', zero_division=0
                    )
                elif scorer == 'r2':
                    result[name] = sklearn_metrics.r2_score(y_val, predictions)
                elif scorer == 'neg_mean_squared_error':
                    result[name] = -sklearn_metrics.mean_squared_error(y_val, predictions)
                elif scorer == 'neg_mean_absolute_error':
                    result[name] = -sklearn_metrics.mean_absolute_error(y_val, predictions)
                else:
                    result[name] = 0.0
            except Exception as e:
                logger.warning(f"Could not compute {name}: {e}")
                result[name] = 0.0

        return result

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Generate probability predictions (classification only)."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("Model does not support probability predictions")
        return self.model.predict_proba(X)

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> Dict[str, float]:
        """Evaluate model on data."""
        return self._calculate_metrics(X, y)

    def save(self, path: Union[str, Path]) -> None:
        """Save the trainer and model."""
        if self.model is None:
            raise ValueError("No model to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model": self.model,
            "model_type": self.model_type,
            "task_type": self.task_type,
            "params": self.params,
            "feature_names": self._feature_names,
            "random_state": self.random_state,
        }
        joblib.dump(state, path)
        logger.info(f"Trainer saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Trainer":
        """Load a trainer from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Trainer file not found: {path}")

        state = joblib.load(path)

        trainer = cls(
            model_type=state["model_type"],
            params=state["params"],
            random_state=state["random_state"],
        )
        trainer.model = state["model"]
        trainer.task_type = state["task_type"]
        trainer._feature_names = state["feature_names"]

        logger.info(f"Trainer loaded from {path}")
        return trainer

    def _prepare_eval_set(
        self,
        fit_kwargs: Dict,
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        val_sample_weights: Optional[np.ndarray],
    ) -> Dict:
        """Prepare evaluation set for gradient boosting models."""
        model_name = self.model_type.value

        if "lightgbm" in model_name:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ] if LIGHTGBM_AVAILABLE else []
            if val_sample_weights is not None:
                fit_kwargs["eval_sample_weight"] = [val_sample_weights]

        elif "xgboost" in model_name:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False
            if val_sample_weights is not None:
                fit_kwargs["sample_weight_eval_set"] = [val_sample_weights]

        elif "catboost" in model_name:
            fit_kwargs["eval_set"] = (X_val, y_val)
            fit_kwargs["early_stopping_rounds"] = 50

        return fit_kwargs

    def _prepare_categorical(
        self,
        fit_kwargs: Dict,
        X: Union[pd.DataFrame, np.ndarray],
        categorical_features: List[str],
    ) -> Dict:
        """Prepare categorical feature handling."""
        model_name = self.model_type.value

        if isinstance(X, pd.DataFrame):
            cat_indices = [
                X.columns.get_loc(c) for c in categorical_features
                if c in X.columns
            ]
        else:
            cat_indices = []

        if "catboost" in model_name and cat_indices:
            fit_kwargs["cat_features"] = cat_indices
        elif "lightgbm" in model_name and cat_indices:
            fit_kwargs["categorical_feature"] = cat_indices

        return fit_kwargs

    def _calculate_metrics(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        prefix: str = "",
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        from sklearn import metrics as sklearn_metrics

        predictions = self.predict(X)
        result = {}

        if self.task_type == TaskType.CLASSIFICATION:
            result[f"{prefix}accuracy"] = sklearn_metrics.accuracy_score(y, predictions)

            if hasattr(self.model, "predict_proba"):
                proba = self.predict_proba(X)
                if proba.shape[1] == 2:
                    result[f"{prefix}roc_auc"] = sklearn_metrics.roc_auc_score(y, proba[:, 1])
                else:
                    result[f"{prefix}roc_auc"] = sklearn_metrics.roc_auc_score(
                        y, proba, multi_class='ovr'
                    )

            result[f"{prefix}f1"] = sklearn_metrics.f1_score(y, predictions, average='weighted')
            result[f"{prefix}precision"] = sklearn_metrics.precision_score(
                y, predictions, average='weighted', zero_division=0
            )
            result[f"{prefix}recall"] = sklearn_metrics.recall_score(
                y, predictions, average='weighted', zero_division=0
            )

        else:  # Regression
            result[f"{prefix}mse"] = sklearn_metrics.mean_squared_error(y, predictions)
            result[f"{prefix}rmse"] = np.sqrt(result[f"{prefix}mse"])
            result[f"{prefix}mae"] = sklearn_metrics.mean_absolute_error(y, predictions)
            result[f"{prefix}r2"] = sklearn_metrics.r2_score(y, predictions)

        return result

    def _log_to_tracker(self, result: TrainingResult) -> None:
        """Log results to experiment tracker."""
        if not self.tracker:
            return

        # Log parameters
        self.tracker.log_params(result.params)

        # Log metrics
        all_metrics = {**result.train_metrics, **result.validation_metrics}
        self.tracker.log_metrics(all_metrics)

        # Log feature importance
        if result.feature_importance is not None and self._feature_names:
            self.tracker.log_feature_importance(
                self.model, self._feature_names
            )

        # Log model
        if result.model is not None:
            self.tracker.log_model(result.model, "model")


# Check for LightGBM for callback support
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
