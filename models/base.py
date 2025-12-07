"""
Base Model Module
=================

Abstract base classes and interfaces for all ML models in the AlphaTrade platform.
Implements JPMorgan-level standards for model lifecycle management.

Features:
- Unified model interface (fit, predict, evaluate)
- Automatic feature importance tracking
- Cross-validation with time series awareness
- Model persistence with versioning
- Performance metrics and diagnostics
- Experiment tracking integration
- ROBUST handling of binary vs multiclass classification

Architecture:
- BaseModel: Abstract base for all models
- ModelRegistry: Dynamic model registration
- ModelSerializer: Standardized persistence
- ModelMetrics: Comprehensive evaluation

Author: AlphaTrade Platform
Version: 2.0.0
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
import time

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
    
    This provides common configuration options that all models share.
    Subclasses should extend this with model-specific parameters.
    
    Attributes:
        name: Model name for identification
        version: Model version string (semver)
        model_type: Type of model (classifier, regressor, etc.)
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 = all cores)
        verbose: Verbosity level (0 = silent)
        early_stopping: Enable early stopping during training
        early_stopping_rounds: Patience for early stopping
        validation_method: Cross-validation method to use
        n_splits: Number of CV splits
        test_size: Test set fraction for holdout
        gap: Gap between train and test (for time series)
        feature_selection: Enable automatic feature selection
        max_features: Maximum features to use (None = all)
        save_artifacts: Save training artifacts automatically
    """
    name: str = "BaseModel"
    version: str = "2.0.0"
    model_type: ModelType = ModelType.CLASSIFIER
    
    # Training parameters
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
    gap: int = 0  # Gap between train and test (prevents leakage)
    
    # Feature selection
    feature_selection: bool = False
    max_features: int | None = None
    
    # Artifacts
    save_artifacts: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        """Create configuration from dictionary."""
        # Convert enum strings back to enums
        if "model_type" in data and isinstance(data["model_type"], str):
            data["model_type"] = ModelType(data["model_type"])
        if "validation_method" in data and isinstance(data["validation_method"], str):
            data["validation_method"] = ValidationMethod(data["validation_method"])
        return cls(**data)


# =============================================================================
# MODEL METRICS
# =============================================================================

@dataclass
class ClassificationMetrics:
    """
    Comprehensive metrics for classification models.
    
    Includes standard ML metrics plus trading-specific metrics.
    """
    # Standard classification metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    log_loss: float = 0.0
    confusion_matrix: NDArray | None = None
    
    # Per-class metrics (for multiclass)
    precision_per_class: dict[int, float] = field(default_factory=dict)
    recall_per_class: dict[int, float] = field(default_factory=dict)
    f1_per_class: dict[int, float] = field(default_factory=dict)
    
    # Trading-specific metrics
    directional_accuracy: float = 0.0
    profit_factor: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (JSON-serializable)."""
        return {
            "accuracy": float(self.accuracy),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1_score": float(self.f1_score),
            "roc_auc": float(self.roc_auc),
            "pr_auc": float(self.pr_auc),
            "log_loss": float(self.log_loss),
            "directional_accuracy": float(self.directional_accuracy),
            "profit_factor": float(self.profit_factor),
        }


@dataclass
class RegressionMetrics:
    """
    Comprehensive metrics for regression models.
    
    Includes standard ML metrics plus trading-specific metrics.
    """
    # Standard regression metrics
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0
    adjusted_r2: float = 0.0
    mape: float = 0.0
    
    # Directional metrics (important for trading)
    directional_accuracy: float = 0.0
    ic: float = 0.0  # Information coefficient (Pearson)
    rank_ic: float = 0.0  # Rank IC (Spearman)
    
    # Trading-specific metrics
    hit_rate: float = 0.0
    profit_factor: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (JSON-serializable)."""
        return {
            "mse": float(self.mse),
            "rmse": float(self.rmse),
            "mae": float(self.mae),
            "r2": float(self.r2),
            "adjusted_r2": float(self.adjusted_r2),
            "mape": float(self.mape),
            "directional_accuracy": float(self.directional_accuracy),
            "ic": float(self.ic),
            "rank_ic": float(self.rank_ic),
            "hit_rate": float(self.hit_rate),
            "profit_factor": float(self.profit_factor),
        }


@dataclass
class ModelMetrics:
    """
    Comprehensive model metrics container.
    
    Tracks training, validation, and test performance along with
    metadata about the training process.
    """
    model_id: str = ""
    model_name: str = ""
    model_type: ModelType = ModelType.CLASSIFIER
    
    # Training info
    train_samples: int = 0
    test_samples: int = 0
    n_features: int = 0
    training_time: float = 0.0
    
    # Classification metrics (if applicable)
    classification: ClassificationMetrics | None = None
    
    # Regression metrics (if applicable)
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
        """Convert to dictionary (JSON-serializable)."""
        result = {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_type": self.model_type.value if isinstance(self.model_type, Enum) else self.model_type,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "n_features": self.n_features,
            "training_time": self.training_time,
            "cv_scores": self.cv_scores,
            "cv_mean": self.cv_mean,
            "cv_std": self.cv_std,
            "top_features": self.top_features[:10],
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "evaluated_at": self.evaluated_at.isoformat() if self.evaluated_at else None,
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
    Abstract base class for all ML models in the AlphaTrade platform.
    
    Provides unified interface for:
    - Training with cross-validation
    - Prediction and probability estimation
    - Feature importance extraction
    - Model persistence (save/load)
    - Performance evaluation
    - Automatic class detection (binary vs multiclass)
    
    Subclasses must implement:
    - _default_config(): Return default configuration
    - _build_model(): Create the underlying model object
    - _fit_impl(): Training implementation
    - _predict_impl(): Prediction implementation
    
    Example:
        class MyModel(BaseModel[MyConfig]):
            def _default_config(self) -> MyConfig:
                return MyConfig()
            
            def _build_model(self) -> Any:
                return SomeModel(**self.config.to_dict())
            
            def _fit_impl(self, X, y, X_val, y_val, sample_weight):
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
            config: Model configuration (uses defaults if None)
            **kwargs: Additional parameters to override config
        """
        self.config: ConfigT = config or self._default_config()
        self._model: Any = None
        self._model_id: str = str(uuid4())[:8]
        self._state: ModelState = ModelState.CREATED
        
        # Feature tracking
        self._feature_names: list[str] = []
        self._n_features: int = 0
        self._feature_importance: dict[str, float] = {}
        
        # Class tracking (for classification)
        self._n_classes: int | None = None
        self._classes: NDArray | None = None
        
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
        
        # Apply any additional kwargs to config
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    # =========================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # =========================================================================
    
    @abstractmethod
    def _default_config(self) -> ConfigT:
        """
        Create default configuration.
        
        Returns:
            Default configuration instance
        """
        pass
    
    @abstractmethod
    def _build_model(self) -> Any:
        """
        Build the underlying model object.
        
        Returns:
            Model instance (e.g., LGBMClassifier, XGBClassifier)
        """
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
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            sample_weight: Sample weights (optional)
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
            X: Features (n_samples, n_features)
        
        Returns:
            Predictions (n_samples,)
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
        """Get unique model ID."""
        return self._model_id
    
    @property
    def state(self) -> ModelState:
        """Get current model state."""
        return self._state
    
    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self._state in {
            ModelState.TRAINED,
            ModelState.VALIDATED,
            ModelState.DEPLOYED,
        }
    
    @property
    def feature_names(self) -> list[str]:
        """Get feature names used during training."""
        return self._feature_names.copy()
    
    @property
    def n_features(self) -> int:
        """Get number of features."""
        return self._n_features
    
    @property
    def n_classes(self) -> int | None:
        """Get number of classes (for classifiers)."""
        return self._n_classes
    
    @property
    def classes(self) -> NDArray | None:
        """Get unique class labels (for classifiers)."""
        return self._classes.copy() if self._classes is not None else None
    
    @property
    def feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
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
    ) -> "BaseModel":
        """
        Train the model.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
            feature_names: Names of features (optional)
            X_val: Validation features for early stopping (optional)
            y_val: Validation targets for early stopping (optional)
            sample_weight: Sample weights (optional)
        
        Returns:
            self (for method chaining)
        
        Example:
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        """
        self._training_start = datetime.now()
        self._state = ModelState.TRAINING
        
        try:
            # Store feature info
            self._n_features = X.shape[1]
            self._feature_names = feature_names or [f"feature_{i}" for i in range(self._n_features)]
            
            # Detect classes for classification
            if self.config.model_type in {ModelType.CLASSIFIER, ModelType.ENSEMBLE}:
                self._classes = np.unique(y)
                self._n_classes = len(self._classes)
                logger.debug(f"Detected {self._n_classes} classes: {self._classes}")
            
            # Build model if not already built
            if self._model is None:
                self._model = self._build_model()
            
            # Log training start
            logger.info(f"Training {self.name} on {len(X)} samples, {self._n_features} features")
            
            start_time = time.time()
            
            # Call implementation
            self._fit_impl(X, y, X_val, y_val, sample_weight)
            
            training_time = time.time() - start_time
            
            # Update metrics
            self._metrics.train_samples = len(X)
            self._metrics.n_features = self._n_features
            self._metrics.training_time = training_time
            self._metrics.trained_at = datetime.now()
            
            # Extract feature importance
            self._extract_feature_importance()
            
            # Update state
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
        
        Raises:
            RuntimeError: If model is not trained
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
            X: Features (n_samples, n_features)
        
        Returns:
            Class probabilities (n_samples, n_classes) or None if not supported
        
        Raises:
            RuntimeError: If model is not trained
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
        
        if self.config.model_type in {ModelType.CLASSIFIER, ModelType.ENSEMBLE}:
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
        n_splits: int | None = None,
    ) -> dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Targets
            feature_names: Feature names
            n_splits: Number of CV splits (overrides config)
        
        Returns:
            Dictionary with CV results
        """
        from sklearn.model_selection import TimeSeriesSplit, KFold
        
        n_splits = n_splits or self.config.n_splits
        
        # Choose CV strategy
        if self.config.validation_method in {
            ValidationMethod.TIME_SERIES,
            ValidationMethod.WALK_FORWARD,
        }:
            cv = TimeSeriesSplit(n_splits=n_splits, gap=self.config.gap)
        else:
            cv = KFold(n_splits=n_splits, shuffle=False)
        
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create fresh model for each fold
            fold_model = self.__class__(config=self.config)
            fold_model.fit(X_train, y_train, feature_names=feature_names)
            
            # Evaluate
            metrics = fold_model.evaluate(X_val, y_val)
            
            # Get primary metric
            if self.config.model_type == ModelType.CLASSIFIER:
                score = metrics.get("roc_auc", metrics.get("accuracy", 0))
            else:
                score = metrics.get("r2", 0)
            
            scores.append(score)
            logger.debug(f"Fold {fold + 1}/{n_splits}: {score:.4f}")
        
        # Update metrics
        self._metrics.cv_scores = scores
        self._metrics.cv_mean = float(np.mean(scores))
        self._metrics.cv_std = float(np.std(scores))
        
        return {
            "scores": scores,
            "mean": self._metrics.cv_mean,
            "std": self._metrics.cv_std,
            "n_splits": n_splits,
        }
    
    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores,
            sorted by importance (descending)
        """
        if not self._feature_importance:
            self._extract_feature_importance()
        
        return dict(
            sorted(
                self._feature_importance.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )
    
    def get_top_features(self, n: int = 10) -> list[str]:
        """
        Get top N most important features.
        
        Args:
            n: Number of features to return
        
        Returns:
            List of feature names
        """
        importance = self.get_feature_importance()
        return list(importance.keys())[:n]
    
    def save(
        self,
        path: Path | str,
        include_artifacts: bool = True,
    ) -> Path:
        """
        Save model to disk.
        
        Args:
            path: Save path (directory or .pkl file)
            include_artifacts: Include training history and artifacts
        
        Returns:
            Path to saved model file
        """
        path = Path(path)
        
        # Create directory if needed
        if path.suffix != ".pkl":
            path.mkdir(parents=True, exist_ok=True)
            path = path / f"{self.name}_{self._model_id}.pkl"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare save data
        save_data = {
            "model": self._model,
            "config": self.config,
            "model_id": self._model_id,
            "state": self._state,
            "feature_names": self._feature_names,
            "n_features": self._n_features,
            "n_classes": self._n_classes,
            "classes": self._classes,
            "feature_importance": self._feature_importance,
            "metrics": self._metrics,
            "class_name": self.__class__.__name__,
            "module_name": self.__class__.__module__,
            "saved_at": datetime.now().isoformat(),
            "version": self.config.version,
        }
        
        if include_artifacts:
            save_data["train_history"] = self._train_history
            save_data["validation_scores"] = self._validation_scores
            save_data["best_iteration"] = self._best_iteration
        
        with open(path, "wb") as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Model saved to {path}")
        return path
    
    @classmethod
    def load(cls, path: Path | str) -> "BaseModel":
        """
        Load model from disk.
        
        Args:
            path: Path to saved model file
        
        Returns:
            Loaded model instance
        """
        path = Path(path)
        
        with open(path, "rb") as f:
            save_data = pickle.load(f)
        
        # Create instance with saved config
        instance = cls(config=save_data["config"])
        
        # Restore state
        instance._model = save_data["model"]
        instance._model_id = save_data["model_id"]
        instance._state = save_data["state"]
        instance._feature_names = save_data["feature_names"]
        instance._n_features = save_data["n_features"]
        instance._n_classes = save_data.get("n_classes")
        instance._classes = save_data.get("classes")
        instance._feature_importance = save_data["feature_importance"]
        instance._metrics = save_data["metrics"]
        
        # Optional artifacts
        instance._train_history = save_data.get("train_history", [])
        instance._validation_scores = save_data.get("validation_scores", [])
        instance._best_iteration = save_data.get("best_iteration", 0)
        
        logger.info(f"Model loaded from {path}")
        return instance
    
    def get_summary(self) -> dict[str, Any]:
        """
        Get comprehensive model summary.
        
        Returns:
            Dictionary with model information
        """
        return {
            "name": self.name,
            "model_id": self._model_id,
            "state": self._state.value,
            "n_features": self._n_features,
            "n_classes": self._n_classes,
            "is_trained": self.is_trained,
            "config": self.config.to_dict(),
            "metrics": self._metrics.to_dict(),
            "top_features": self.get_top_features(10),
        }
    
    # =========================================================================
    # PROTECTED METHODS
    # =========================================================================
    
    def _extract_feature_importance(self) -> None:
        """Extract feature importance from underlying model."""
        if self._model is None:
            return
        
        importance = None
        
        # Try different attribute names
        if hasattr(self._model, "feature_importances_"):
            importance = self._model.feature_importances_
        elif hasattr(self._model, "coef_"):
            importance = np.abs(self._model.coef_).mean(axis=0) if self._model.coef_.ndim > 1 else np.abs(self._model.coef_)
        elif hasattr(self._model, "get_score"):
            # XGBoost native
            importance_dict = self._model.get_score(importance_type="gain")
            importance = np.zeros(self._n_features)
            for i, name in enumerate(self._feature_names):
                importance[i] = importance_dict.get(name, 0)
        
        if importance is not None and len(importance) > 0:
            # Map to feature names
            for i, imp in enumerate(importance):
                if i < len(self._feature_names):
                    self._feature_importance[self._feature_names[i]] = float(imp)
            
            # Sort by importance
            self._feature_importance = dict(
                sorted(
                    self._feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
            
            # Update metrics
            self._metrics.feature_importance = self._feature_importance
            self._metrics.top_features = list(self._feature_importance.keys())[:10]
    
    def _evaluate_classification(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64],
        X: NDArray[np.float64],
    ) -> ClassificationMetrics:
        """
        Evaluate classification model with ROBUST handling.
        
        This method handles:
        - Binary vs multiclass detection
        - Probability matrix dimension mismatches
        - Models trained with different num_class than actual data
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            X: Features (for probability prediction)
        
        Returns:
            ClassificationMetrics instance
        """
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
        
        # Basic accuracy
        metrics.accuracy = float(accuracy_score(y_true, y_pred))
        
        # Detect number of classes in evaluation data
        unique_true = np.unique(y_true)
        n_classes = len(unique_true)
        average = "binary" if n_classes == 2 else "weighted"
        
        # Precision, Recall, F1
        metrics.precision = float(precision_score(y_true, y_pred, average=average, zero_division=0))
        metrics.recall = float(recall_score(y_true, y_pred, average=average, zero_division=0))
        metrics.f1_score = float(f1_score(y_true, y_pred, average=average, zero_division=0))
        
        # =====================================================================
        # CRITICAL FIX: Robust ROC-AUC with dimension mismatch handling
        # =====================================================================
        if hasattr(self._model, "predict_proba"):
            try:
                y_proba = self._model.predict_proba(X)
                proba_n_cols = y_proba.shape[1] if y_proba.ndim > 1 else 1
                
                if n_classes == 2:
                    # Binary classification
                    if proba_n_cols == 2:
                        # Standard case: 2 probability columns
                        pos_proba = y_proba[:, 1]
                    elif proba_n_cols > 2:
                        # Model was trained multiclass but data is binary
                        # Try to find the probability for class 1
                        model_classes = getattr(self._model, "classes_", None)
                        if model_classes is not None:
                            try:
                                # Find index of class 1 in model's classes
                                pos_idx = np.where(model_classes == 1)[0]
                                if len(pos_idx) > 0:
                                    pos_proba = y_proba[:, pos_idx[0]]
                                else:
                                    # Class 1 not found, use column 1
                                    pos_proba = y_proba[:, 1] if proba_n_cols > 1 else y_proba[:, 0]
                            except Exception:
                                pos_proba = y_proba[:, 1] if proba_n_cols > 1 else y_proba[:, 0]
                        else:
                            # No classes_ attribute, use column 1
                            pos_proba = y_proba[:, 1] if proba_n_cols > 1 else y_proba[:, 0]
                    else:
                        # Single column (unusual)
                        pos_proba = y_proba.ravel()
                    
                    # Calculate binary metrics
                    metrics.roc_auc = float(roc_auc_score(y_true, pos_proba))
                    metrics.pr_auc = float(average_precision_score(y_true, pos_proba))
                    
                    # Log loss for binary
                    proba_2col = np.column_stack([1 - pos_proba, pos_proba])
                    metrics.log_loss = float(log_loss(y_true, proba_2col))
                    
                else:
                    # Multiclass classification
                    if proba_n_cols == n_classes:
                        # Dimensions match - standard calculation
                        metrics.roc_auc = float(roc_auc_score(
                            y_true, y_proba,
                            multi_class="ovr",
                            average="weighted",
                            labels=unique_true,
                        ))
                        metrics.log_loss = float(log_loss(y_true, y_proba, labels=unique_true))
                    elif proba_n_cols > n_classes:
                        # Model has more classes than data
                        # Extract relevant columns based on model's classes
                        model_classes = getattr(self._model, "classes_", None)
                        if model_classes is not None:
                            try:
                                indices = [
                                    np.where(model_classes == c)[0][0]
                                    for c in unique_true
                                    if c in model_classes
                                ]
                                if len(indices) == n_classes:
                                    adjusted_proba = y_proba[:, indices]
                                    # Renormalize
                                    adjusted_proba = adjusted_proba / adjusted_proba.sum(axis=1, keepdims=True)
                                    metrics.roc_auc = float(roc_auc_score(
                                        y_true, adjusted_proba,
                                        multi_class="ovr",
                                        average="weighted",
                                    ))
                            except Exception as e:
                                logger.warning(f"Could not adjust probability matrix: {e}")
                        else:
                            logger.warning(
                                f"Probability matrix has {proba_n_cols} columns but data has {n_classes} classes"
                            )
                    else:
                        # Model has fewer classes than data (unusual)
                        logger.warning(
                            f"Model probability matrix ({proba_n_cols} cols) smaller than data classes ({n_classes})"
                        )
                        
            except Exception as e:
                logger.warning(f"ROC-AUC calculation failed: {e}")
                metrics.roc_auc = 0.0
                metrics.pr_auc = 0.0
                metrics.log_loss = 0.0
        
        # Confusion matrix
        try:
            metrics.confusion_matrix = confusion_matrix(y_true, y_pred)
        except Exception:
            pass
        
        # Directional accuracy (same as accuracy for classification)
        metrics.directional_accuracy = metrics.accuracy
        
        return metrics
    
    def _evaluate_regression(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64],
        returns: NDArray[np.float64] | None = None,
    ) -> RegressionMetrics:
        """
        Evaluate regression model.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            returns: Actual returns (for trading metrics)
        
        Returns:
            RegressionMetrics instance
        """
        from sklearn.metrics import (
            mean_squared_error,
            mean_absolute_error,
            r2_score,
        )
        
        metrics = RegressionMetrics()
        
        # Standard metrics
        metrics.mse = float(mean_squared_error(y_true, y_pred))
        metrics.rmse = float(np.sqrt(metrics.mse))
        metrics.mae = float(mean_absolute_error(y_true, y_pred))
        metrics.r2 = float(r2_score(y_true, y_pred))
        
        # Adjusted R2
        n = len(y_true)
        p = self._n_features
        if n > p + 1:
            metrics.adjusted_r2 = 1 - (1 - metrics.r2) * (n - 1) / (n - p - 1)
        
        # MAPE (avoiding division by zero)
        mask = y_true != 0
        if mask.any():
            metrics.mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        
        # Directional accuracy
        if len(y_true) > 1:
            true_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            metrics.directional_accuracy = float(np.mean(true_direction == pred_direction))
        
        # Information coefficient (Pearson correlation)
        try:
            metrics.ic = float(np.corrcoef(y_true, y_pred)[0, 1])
        except Exception:
            metrics.ic = 0.0
        
        # Rank IC (Spearman correlation)
        try:
            metrics.rank_ic = float(scipy_stats.spearmanr(y_true, y_pred)[0])
        except Exception:
            metrics.rank_ic = 0.0
        
        # Trading metrics if returns provided
        if returns is not None:
            # Hit rate: % of correct direction predictions that were profitable
            pred_direction = np.sign(y_pred)
            actual_direction = np.sign(returns)
            metrics.hit_rate = float(np.mean(pred_direction == actual_direction))
            
            # Profit factor
            gains = returns[pred_direction * returns > 0].sum()
            losses = np.abs(returns[pred_direction * returns < 0].sum())
            metrics.profit_factor = float(gains / losses) if losses > 0 else 0.0
        
        return metrics
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"state={self._state.value}, "
            f"features={self._n_features}, "
            f"classes={self._n_classes})"
        )


# =============================================================================
# MODEL REGISTRY
# =============================================================================

class ModelRegistry:
    """
    Registry for model classes.
    
    Allows dynamic model registration and discovery.
    Supports factory pattern for model creation.
    
    Example:
        # Register a model
        ModelRegistry.register("my_model", MyModel, MyModelConfig)
        
        # Create model from registry
        model = ModelRegistry.create("my_model", learning_rate=0.01)
        
        # List available models
        print(ModelRegistry.list_models())
    """
    
    _registry: dict[str, type[BaseModel]] = {}
    _configs: dict[str, type[ModelConfig]] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        model_class: type[BaseModel],
        config_class: type[ModelConfig] | None = None,
    ) -> None:
        """
        Register a model class.
        
        Args:
            name: Registration name (case-insensitive)
            model_class: Model class to register
            config_class: Configuration class (optional)
        """
        key = name.lower()
        cls._registry[key] = model_class
        if config_class:
            cls._configs[key] = config_class
        logger.debug(f"Registered model: {name}")
    
    @classmethod
    def get(cls, name: str) -> type[BaseModel]:
        """
        Get a registered model class.
        
        Args:
            name: Model name
        
        Returns:
            Model class
        
        Raises:
            KeyError: If model not found
        """
        key = name.lower()
        if key not in cls._registry:
            raise KeyError(
                f"Model '{name}' not found. Available: {cls.list_models()}"
            )
        return cls._registry[key]
    
    @classmethod
    def get_config(cls, name: str) -> type[ModelConfig]:
        """
        Get config class for a model.
        
        Args:
            name: Model name
        
        Returns:
            Config class (ModelConfig if not registered)
        """
        return cls._configs.get(name.lower(), ModelConfig)
    
    @classmethod
    def list_models(cls) -> list[str]:
        """
        List all registered model names.
        
        Returns:
            List of model names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseModel:
        """
        Create a model instance from registry.
        
        Args:
            name: Model name
            **kwargs: Model configuration parameters
        
        Returns:
            Model instance
        """
        model_class = cls.get(name)
        config_class = cls.get_config(name)
        
        # Filter kwargs to only valid config parameters
        valid_kwargs = {
            k: v for k, v in kwargs.items()
            if hasattr(config_class, k)
        }
        
        config = config_class(**valid_kwargs)
        return model_class(config=config)
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registrations."""
        cls._registry.clear()
        cls._configs.clear()


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