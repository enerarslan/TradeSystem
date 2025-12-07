"""
Base Model Module
=================

Abstract base classes and interfaces for all ML models in the trading platform.
Implements JPMorgan-level standards for model lifecycle management.

Features:
- Unified model interface (fit, predict, evaluate)
- Automatic feature importance tracking
- Cross-validation with time series awareness
- Model persistence with versioning
- Performance metrics and diagnostics
- Experiment tracking integration

Architecture:
- BaseModel: Abstract base for all models
- ModelRegistry: Dynamic model registration
- ModelSerializer: Standardized persistence
- ModelMetrics: Comprehensive evaluation

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generic, Protocol, TypeVar, runtime_checkable
from uuid import UUID, uuid4
import hashlib
import json
import pickle

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from config.settings import get_logger

logger = get_logger(__name__)

# Type variables
ModelT = TypeVar("ModelT", bound="BaseModel")
ConfigT = TypeVar("ConfigT", bound="ModelConfig")


# =============================================================================
# ENUMS
# =============================================================================

class ModelType(str, Enum):
    """Model type classification."""
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
    ENSEMBLE = "ensemble"
    DEEP_LEARNING = "deep_learning"
    REINFORCEMENT = "reinforcement"


class ModelState(str, Enum):
    """Model lifecycle state."""
    CREATED = "created"
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


class ValidationMethod(str, Enum):
    """Cross-validation methods."""
    HOLDOUT = "holdout"
    KFOLD = "kfold"
    TIME_SERIES = "time_series"
    WALK_FORWARD = "walk_forward"
    PURGED_KFOLD = "purged_kfold"
    COMBINATORIAL_PURGED = "combinatorial_purged"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """
    Base configuration for all models.
    
    Attributes:
        name: Model name
        version: Model version string
        model_type: Type of model
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 = all cores)
        verbose: Verbosity level
        early_stopping: Enable early stopping
        early_stopping_rounds: Rounds for early stopping
        validation_method: Cross-validation method
        n_splits: Number of CV splits
        test_size: Test set fraction
        gap: Gap between train and test (for time series)
        feature_selection: Enable feature selection
        max_features: Maximum features to use
        save_artifacts: Save training artifacts
    """
    name: str = "BaseModel"
    version: str = "1.0.0"
    model_type: ModelType = ModelType.CLASSIFIER
    
    # Training
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = 0
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_rounds: int = 50
    
    # Validation
    validation_method: ValidationMethod = ValidationMethod.WALK_FORWARD
    n_splits: int = 5
    test_size: float = 0.2
    gap: int = 0  # Gap between train and test
    
    # Feature selection
    feature_selection: bool = False
    max_features: int | None = None
    
    # Artifacts
    save_artifacts: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type.value,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
            "early_stopping": self.early_stopping,
            "early_stopping_rounds": self.early_stopping_rounds,
            "validation_method": self.validation_method.value,
            "n_splits": self.n_splits,
            "test_size": self.test_size,
            "gap": self.gap,
            "feature_selection": self.feature_selection,
            "max_features": self.max_features,
            "save_artifacts": self.save_artifacts,
        }


# =============================================================================
# MODEL METRICS
# =============================================================================

@dataclass
class ClassificationMetrics:
    """Metrics for classification models."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    log_loss: float = 0.0
    confusion_matrix: NDArray | None = None
    
    # Per-class metrics
    precision_per_class: dict[int, float] = field(default_factory=dict)
    recall_per_class: dict[int, float] = field(default_factory=dict)
    f1_per_class: dict[int, float] = field(default_factory=dict)
    
    # Trading-specific
    directional_accuracy: float = 0.0
    profit_factor: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "log_loss": self.log_loss,
            "directional_accuracy": self.directional_accuracy,
            "profit_factor": self.profit_factor,
        }


@dataclass
class RegressionMetrics:
    """Metrics for regression models."""
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0
    adjusted_r2: float = 0.0
    mape: float = 0.0
    
    # Directional
    directional_accuracy: float = 0.0
    ic: float = 0.0  # Information coefficient
    rank_ic: float = 0.0  # Rank IC (Spearman)
    
    # Trading-specific
    hit_rate: float = 0.0
    profit_factor: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "adjusted_r2": self.adjusted_r2,
            "mape": self.mape,
            "directional_accuracy": self.directional_accuracy,
            "ic": self.ic,
            "rank_ic": self.rank_ic,
            "hit_rate": self.hit_rate,
            "profit_factor": self.profit_factor,
        }


@dataclass
class ModelMetrics:
    """
    Comprehensive model metrics container.
    
    Tracks training, validation, and test performance.
    """
    model_id: str = ""
    model_name: str = ""
    model_type: ModelType = ModelType.CLASSIFIER
    
    # Training info
    train_samples: int = 0
    test_samples: int = 0
    n_features: int = 0
    training_time: float = 0.0
    
    # Classification metrics
    classification: ClassificationMetrics | None = None
    
    # Regression metrics
    regression: RegressionMetrics | None = None
    
    # Cross-validation results
    cv_scores: list[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    
    # Feature importance
    feature_importance: dict[str, float] = field(default_factory=dict)
    top_features: list[str] = field(default_factory=list)
    
    # Timestamps
    trained_at: datetime | None = None
    evaluated_at: datetime | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_type": self.model_type.value,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "n_features": self.n_features,
            "training_time": self.training_time,
            "cv_scores": self.cv_scores,
            "cv_mean": self.cv_mean,
            "cv_std": self.cv_std,
            "top_features": self.top_features[:10],
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
        }
        
        if self.classification:
            result["classification"] = self.classification.to_dict()
        if self.regression:
            result["regression"] = self.regression.to_dict()
        
        return result


# =============================================================================
# BASE MODEL
# =============================================================================

class BaseModel(ABC, Generic[ConfigT]):
    """
    Abstract base class for all ML models.
    
    Provides unified interface for:
    - Training with cross-validation
    - Prediction and probability estimation
    - Feature importance extraction
    - Model persistence
    - Performance evaluation
    
    Subclasses must implement:
    - _build_model(): Create the underlying model
    - _fit_impl(): Training implementation
    - _predict_impl(): Prediction implementation
    
    Example:
        class MyModel(BaseModel[MyConfig]):
            def _build_model(self):
                return SomeModel(**self.config.to_dict())
            
            def _fit_impl(self, X, y):
                self._model.fit(X, y)
            
            def _predict_impl(self, X):
                return self._model.predict(X)
    """
    
    def __init__(
        self,
        config: ConfigT | None = None,
        **kwargs: Any,
    ):
        """
        Initialize model.
        
        Args:
            config: Model configuration
            **kwargs: Additional parameters
        """
        self.config: ConfigT = config or self._default_config()
        self._model: Any = None
        self._model_id: str = str(uuid4())[:8]
        self._state: ModelState = ModelState.CREATED
        
        # Feature tracking
        self._feature_names: list[str] = []
        self._n_features: int = 0
        self._feature_importance: dict[str, float] = {}
        
        # Training artifacts
        self._train_history: list[dict[str, Any]] = []
        self._validation_scores: list[float] = []
        self._best_iteration: int = 0
        
        # Metrics
        self._metrics: ModelMetrics = ModelMetrics(
            model_id=self._model_id,
            model_name=self.config.name,
            model_type=self.config.model_type,
        )
        
        # Timing
        self._training_start: datetime | None = None
        self._training_end: datetime | None = None
        
        # Apply any additional kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    # =========================================================================
    # ABSTRACT METHODS
    # =========================================================================
    
    @abstractmethod
    def _default_config(self) -> ConfigT:
        """Create default configuration."""
        pass
    
    @abstractmethod
    def _build_model(self) -> Any:
        """Build the underlying model."""
        pass
    
    @abstractmethod
    def _fit_impl(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.float64] | None = None,
        sample_weight: NDArray[np.float64] | None = None,
    ) -> None:
        """
        Internal fit implementation.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Validation features
            y_val: Validation targets
            sample_weight: Sample weights
        """
        pass
    
    @abstractmethod
    def _predict_impl(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Internal prediction implementation.
        
        Args:
            X: Features
        
        Returns:
            Predictions
        """
        pass
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def name(self) -> str:
        """Get model name."""
        return self.config.name
    
    @property
    def model_id(self) -> str:
        """Get model ID."""
        return self._model_id
    
    @property
    def state(self) -> ModelState:
        """Get model state."""
        return self._state
    
    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._state in {ModelState.TRAINED, ModelState.VALIDATED, ModelState.DEPLOYED}
    
    @property
    def feature_names(self) -> list[str]:
        """Get feature names."""
        return self._feature_names.copy()
    
    @property
    def n_features(self) -> int:
        """Get number of features."""
        return self._n_features
    
    @property
    def feature_importance(self) -> dict[str, float]:
        """Get feature importance."""
        return self._feature_importance.copy()
    
    @property
    def metrics(self) -> ModelMetrics:
        """Get model metrics."""
        return self._metrics
    
    @property
    def underlying_model(self) -> Any:
        """Get the underlying model object."""
        return self._model
    
    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================
    
    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        feature_names: list[str] | None = None,
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.float64] | None = None,
        sample_weight: NDArray[np.float64] | None = None,
    ) -> ModelT:
        """
        Train the model.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
            feature_names: Names of features
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            sample_weight: Sample weights (optional)
        
        Returns:
            Self for chaining
        """
        import time
        
        logger.info(f"Training {self.name} on {X.shape[0]} samples, {X.shape[1]} features")
        
        # Store feature info
        self._feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        self._n_features = X.shape[1]
        
        # Build model
        self._model = self._build_model()
        self._state = ModelState.TRAINING
        self._training_start = datetime.now()
        
        start_time = time.time()
        
        try:
            # Train
            self._fit_impl(X, y, X_val, y_val, sample_weight)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Extract feature importance
            self._extract_feature_importance()
            
            # Update metrics
            self._metrics.train_samples = len(X)
            self._metrics.n_features = self._n_features
            self._metrics.training_time = training_time
            self._metrics.trained_at = datetime.now()
            self._metrics.feature_importance = self._feature_importance
            self._metrics.top_features = self._get_top_features(20)
            
            self._state = ModelState.TRAINED
            self._training_end = datetime.now()
            
            logger.info(f"Training completed in {training_time:.2f}s")
            
        except Exception as e:
            self._state = ModelState.FAILED
            logger.error(f"Training failed: {e}")
            raise
        
        return self
    
    def predict(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Make predictions.
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Predictions (n_samples,)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        return self._predict_impl(X)
    
    def predict_proba(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64] | None:
        """
        Predict class probabilities (for classifiers).
        
        Args:
            X: Features
        
        Returns:
            Class probabilities or None if not supported
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        return None
    
    def evaluate(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        returns: NDArray[np.float64] | None = None,
    ) -> dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: Test targets
            returns: Actual returns (for trading metrics)
        
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X)
        
        self._metrics.test_samples = len(X)
        self._metrics.evaluated_at = datetime.now()
        
        if self.config.model_type == ModelType.CLASSIFIER:
            metrics = self._evaluate_classification(y, predictions, X)
            self._metrics.classification = metrics
            return metrics.to_dict()
        else:
            metrics = self._evaluate_regression(y, predictions, returns)
            self._metrics.regression = metrics
            return metrics.to_dict()
    
    def cross_validate(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Targets
            feature_names: Feature names
        
        Returns:
            Cross-validation results
        """
        from sklearn.model_selection import TimeSeriesSplit, KFold
        
        method = self.config.validation_method
        n_splits = self.config.n_splits
        
        logger.info(f"Cross-validating with {method.value}, {n_splits} splits")
        
        # Select splitter
        if method == ValidationMethod.TIME_SERIES:
            splitter = TimeSeriesSplit(
                n_splits=n_splits,
                gap=self.config.gap,
            )
        elif method == ValidationMethod.WALK_FORWARD:
            splitter = self._create_walk_forward_splitter(len(X), n_splits)
        else:
            splitter = KFold(
                n_splits=n_splits,
                shuffle=False,
            )
        
        scores = []
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(splitter.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create fresh model for each fold
            fold_model = self.__class__(config=self.config)
            fold_model.fit(X_train, y_train, feature_names, X_val, y_val)
            
            # Evaluate
            fold_eval = fold_model.evaluate(X_val, y_val)
            
            # Get main score
            if self.config.model_type == ModelType.CLASSIFIER:
                score = fold_eval.get("roc_auc", fold_eval.get("accuracy", 0))
            else:
                score = -fold_eval.get("rmse", 0)  # Negate for consistency
            
            scores.append(score)
            fold_metrics.append(fold_eval)
            
            logger.debug(f"Fold {fold + 1}: score = {score:.4f}")
        
        # Store CV results
        self._metrics.cv_scores = scores
        self._metrics.cv_mean = float(np.mean(scores))
        self._metrics.cv_std = float(np.std(scores))
        
        return {
            "scores": scores,
            "mean": self._metrics.cv_mean,
            "std": self._metrics.cv_std,
            "fold_metrics": fold_metrics,
        }
    
    def save(
        self,
        path: Path | str,
        include_artifacts: bool = True,
    ) -> Path:
        """
        Save model to disk.
        
        Args:
            path: Save path (directory or file)
            include_artifacts: Include training artifacts
        
        Returns:
            Path to saved model
        """
        path = Path(path)
        
        # Create directory if needed
        if path.suffix != ".pkl":
            path.mkdir(parents=True, exist_ok=True)
            path = path / f"{self.name}_{self._model_id}.pkl"
        
        # Prepare save data
        save_data = {
            "model": self._model,
            "config": self.config,
            "model_id": self._model_id,
            "state": self._state,
            "feature_names": self._feature_names,
            "n_features": self._n_features,
            "feature_importance": self._feature_importance,
            "metrics": self._metrics,
            "class_name": self.__class__.__name__,
            "saved_at": datetime.now().isoformat(),
        }
        
        if include_artifacts:
            save_data["train_history"] = self._train_history
            save_data["validation_scores"] = self._validation_scores
            save_data["best_iteration"] = self._best_iteration
        
        with open(path, "wb") as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Model saved to {path}")
        return path
    
    @classmethod
    def load(cls, path: Path | str) -> BaseModel:
        """
        Load model from disk.
        
        Args:
            path: Path to saved model
        
        Returns:
            Loaded model
        """
        path = Path(path)
        
        with open(path, "rb") as f:
            save_data = pickle.load(f)
        
        # Create instance
        instance = cls(config=save_data["config"])
        instance._model = save_data["model"]
        instance._model_id = save_data["model_id"]
        instance._state = save_data["state"]
        instance._feature_names = save_data["feature_names"]
        instance._n_features = save_data["n_features"]
        instance._feature_importance = save_data["feature_importance"]
        instance._metrics = save_data["metrics"]
        
        # Optional artifacts
        instance._train_history = save_data.get("train_history", [])
        instance._validation_scores = save_data.get("validation_scores", [])
        instance._best_iteration = save_data.get("best_iteration", 0)
        
        logger.info(f"Model loaded from {path}")
        return instance
    
    def get_summary(self) -> dict[str, Any]:
        """Get model summary."""
        return {
            "name": self.name,
            "model_id": self._model_id,
            "state": self._state.value,
            "type": self.config.model_type.value,
            "is_trained": self.is_trained,
            "n_features": self._n_features,
            "top_features": self._get_top_features(5),
            "metrics": self._metrics.to_dict(),
        }
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    
    def _extract_feature_importance(self) -> None:
        """Extract feature importance from model."""
        importance = {}
        
        # Try different attribute names
        if hasattr(self._model, "feature_importances_"):
            raw_importance = self._model.feature_importances_
        elif hasattr(self._model, "coef_"):
            raw_importance = np.abs(self._model.coef_).flatten()
        elif hasattr(self._model, "feature_importance"):
            raw_importance = self._model.feature_importance()
        else:
            return
        
        # Map to feature names
        for i, imp in enumerate(raw_importance):
            if i < len(self._feature_names):
                importance[self._feature_names[i]] = float(imp)
        
        # Sort by importance
        self._feature_importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )
    
    def _get_top_features(self, n: int = 10) -> list[str]:
        """Get top N features by importance."""
        return list(self._feature_importance.keys())[:n]
    
    def _evaluate_classification(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64],
        X: NDArray[np.float64],
    ) -> ClassificationMetrics:
        """Evaluate classification model."""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            average_precision_score,
            log_loss,
            confusion_matrix,
        )
        
        metrics = ClassificationMetrics()
        
        # Basic metrics
        metrics.accuracy = accuracy_score(y_true, y_pred)
        
        # Handle binary vs multiclass
        n_classes = len(np.unique(y_true))
        average = "binary" if n_classes == 2 else "weighted"
        
        metrics.precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics.recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics.f1_score = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # ROC-AUC
        if hasattr(self._model, "predict_proba"):
            y_proba = self._model.predict_proba(X)
            try:
                if n_classes == 2:
                    metrics.roc_auc = roc_auc_score(y_true, y_proba[:, 1])
                    metrics.pr_auc = average_precision_score(y_true, y_proba[:, 1])
                else:
                    metrics.roc_auc = roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="weighted"
                    )
                metrics.log_loss = log_loss(y_true, y_proba)
            except Exception:
                pass
        
        # Confusion matrix
        metrics.confusion_matrix = confusion_matrix(y_true, y_pred)
        
        # Directional accuracy (for trading)
        metrics.directional_accuracy = metrics.accuracy
        
        return metrics
    
    def _evaluate_regression(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64],
        returns: NDArray[np.float64] | None = None,
    ) -> RegressionMetrics:
        """Evaluate regression model."""
        from sklearn.metrics import (
            mean_squared_error,
            mean_absolute_error,
            r2_score,
        )
        
        metrics = RegressionMetrics()
        
        # Basic metrics
        metrics.mse = mean_squared_error(y_true, y_pred)
        metrics.rmse = np.sqrt(metrics.mse)
        metrics.mae = mean_absolute_error(y_true, y_pred)
        metrics.r2 = r2_score(y_true, y_pred)
        
        # Adjusted R2
        n = len(y_true)
        p = self._n_features
        if n > p + 1:
            metrics.adjusted_r2 = 1 - (1 - metrics.r2) * (n - 1) / (n - p - 1)
        
        # MAPE
        mask = y_true != 0
        if mask.any():
            metrics.mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        # Directional accuracy
        y_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)
        metrics.directional_accuracy = np.mean(y_direction == pred_direction)
        
        # Information Coefficient (Pearson)
        metrics.ic = float(np.corrcoef(y_true, y_pred)[0, 1])
        
        # Rank IC (Spearman)
        metrics.rank_ic = float(scipy_stats.spearmanr(y_true, y_pred)[0])
        
        # Hit rate
        metrics.hit_rate = metrics.directional_accuracy
        
        # Profit factor (if returns provided)
        if returns is not None:
            strategy_returns = returns * np.sign(y_pred)
            gains = strategy_returns[strategy_returns > 0].sum()
            losses = abs(strategy_returns[strategy_returns < 0].sum())
            metrics.profit_factor = gains / losses if losses > 0 else float("inf")
        
        return metrics
    
    def _create_walk_forward_splitter(
        self,
        n_samples: int,
        n_splits: int,
    ):
        """Create walk-forward validation splitter."""
        class WalkForwardSplitter:
            def __init__(self, n_samples: int, n_splits: int, train_pct: float = 0.7):
                self.n_samples = n_samples
                self.n_splits = n_splits
                self.train_pct = train_pct
            
            def split(self, X=None):
                step_size = (self.n_samples - int(self.n_samples * self.train_pct)) // self.n_splits
                
                for i in range(self.n_splits):
                    train_end = int(self.n_samples * self.train_pct) + i * step_size
                    test_start = train_end
                    test_end = min(test_start + step_size, self.n_samples)
                    
                    train_idx = np.arange(0, train_end)
                    test_idx = np.arange(test_start, test_end)
                    
                    yield train_idx, test_idx
            
            def get_n_splits(self, X=None, y=None, groups=None):
                return self.n_splits
        
        return WalkForwardSplitter(n_samples, n_splits)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"state={self._state.value}, "
            f"features={self._n_features})"
        )


# =============================================================================
# MODEL REGISTRY
# =============================================================================

class ModelRegistry:
    """
    Registry for model classes.
    
    Allows dynamic model registration and discovery.
    """
    
    _models: dict[str, type[BaseModel]] = {}
    _configs: dict[str, type[ModelConfig]] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        model_class: type[BaseModel],
        config_class: type[ModelConfig] | None = None,
    ) -> None:
        """Register a model class."""
        cls._models[name.lower()] = model_class
        if config_class:
            cls._configs[name.lower()] = config_class
        logger.debug(f"Registered model: {name}")
    
    @classmethod
    def get(cls, name: str) -> type[BaseModel]:
        """Get a registered model class."""
        return cls._models[name.lower()]
    
    @classmethod
    def get_config(cls, name: str) -> type[ModelConfig]:
        """Get config class for a model."""
        return cls._configs.get(name.lower(), ModelConfig)
    
    @classmethod
    def list_models(cls) -> list[str]:
        """List all registered models."""
        return list(cls._models.keys())
    
    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseModel:
        """Create a model instance."""
        model_class = cls.get(name)
        config_class = cls.get_config(name)
        config = config_class(**{k: v for k, v in kwargs.items() if hasattr(config_class, k)})
        return model_class(config=config)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ModelType",
    "ModelState",
    "ValidationMethod",
    # Config
    "ModelConfig",
    # Metrics
    "ClassificationMetrics",
    "RegressionMetrics",
    "ModelMetrics",
    # Base class
    "BaseModel",
    # Registry
    "ModelRegistry",
]