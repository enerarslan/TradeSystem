"""
Base Model Framework
JPMorgan-Level Model Architecture

Features:
- Abstract model interface
- Model versioning and tracking
- Hyperparameter management
- Model persistence
- Performance monitoring
"""

import os
import json
import pickle
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd

from ..utils.logger import get_logger, get_audit_logger
from ..utils.helpers import ensure_dir


logger = get_logger(__name__)
audit_logger = get_audit_logger()


@dataclass
class ModelMetadata:
    """Model metadata for tracking and versioning"""
    model_id: str
    model_type: str
    version: str
    created_at: datetime
    trained_at: Optional[datetime] = None
    features: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    data_hash: Optional[str] = None
    notes: str = ""


@dataclass
class PredictionResult:
    """Standardized prediction result"""
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    feature_contributions: Optional[Dict[str, np.ndarray]] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseModel(ABC):
    """
    Abstract base class for all ML models.

    Provides common interface for:
    - Training
    - Prediction
    - Model persistence
    - Performance tracking
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        model_type: str = "base",
        version: str = "1.0.0",
        **hyperparameters
    ):
        """
        Initialize BaseModel.

        Args:
            model_id: Unique model identifier
            model_type: Type of model
            version: Model version
            **hyperparameters: Model hyperparameters
        """
        self.model_type = model_type  # Set first for _generate_model_id()
        self.model_id = model_id or self._generate_model_id()
        self.version = version
        self.hyperparameters = hyperparameters

        self._model = None
        self._is_trained = False
        self._feature_names: List[str] = []

        self.metadata = ModelMetadata(
            model_id=self.model_id,
            model_type=model_type,
            version=version,
            created_at=datetime.utcnow(),
            hyperparameters=hyperparameters
        )

        logger.info(f"Initialized {model_type} model: {self.model_id}")

    def _generate_model_id(self) -> str:
        """Generate unique model ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        random_part = hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:8]
        return f"{self.model_type}_{timestamp}_{random_part}"

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> 'BaseModel':
        """
        Train the model.

        Args:
            X: Feature DataFrame
            y: Target series
            validation_data: Optional validation data tuple
            **kwargs: Additional training parameters

        Returns:
            Self for chaining
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Predictions array
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature DataFrame

        Returns:
            Probability array
        """
        pass

    def predict_with_confidence(
        self,
        X: pd.DataFrame
    ) -> PredictionResult:
        """
        Make predictions with confidence scores.

        Args:
            X: Feature DataFrame

        Returns:
            PredictionResult with predictions and confidence
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        # Confidence is max probability
        if probabilities.ndim > 1:
            confidence = np.max(probabilities, axis=1)
        else:
            confidence = probabilities

        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            confidence=confidence
        )

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self._is_trained:
            logger.warning("Model not trained, no feature importance available")
            return {}

        return dict(zip(
            self._feature_names,
            self._get_importance_scores()
        ))

    @abstractmethod
    def _get_importance_scores(self) -> np.ndarray:
        """Get raw importance scores from model"""
        pass

    def save(self, path: Union[str, Path]) -> str:
        """
        Save model to disk.

        Args:
            path: Directory path for saving

        Returns:
            Full path to saved model
        """
        path = Path(path)
        ensure_dir(path)

        model_path = path / f"{self.model_id}.pkl"
        metadata_path = path / f"{self.model_id}_metadata.json"

        # Update metadata
        self.metadata.feature_importance = self.get_feature_importance()

        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self._model,
                'feature_names': self._feature_names,
                'is_trained': self._is_trained,
                'hyperparameters': self.hyperparameters
            }, f)

        # Save metadata - convert numpy types to native Python types
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                # Convert both keys and values - numpy int64 keys cause JSON serialization errors
                return {
                    (int(k) if isinstance(k, np.integer) else k): convert_to_serializable(v)
                    for k, v in obj.items()
                }
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        metadata_dict = asdict(self.metadata)
        metadata_dict['created_at'] = metadata_dict['created_at'].isoformat()
        if metadata_dict['trained_at']:
            metadata_dict['trained_at'] = metadata_dict['trained_at'].isoformat()

        # Convert any numpy types
        metadata_dict = convert_to_serializable(metadata_dict)

        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)

        logger.info(f"Model saved to {model_path}")

        # Audit log
        audit_logger.log_system_event(
            "MODEL_SAVED",
            {
                "model_id": self.model_id,
                "path": str(model_path),
                "metrics": self.metadata.performance_metrics
            }
        )

        return str(model_path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BaseModel':
        """
        Load model from disk.

        Args:
            path: Path to saved model file

        Returns:
            Loaded model instance
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, 'rb') as f:
            data = pickle.load(f)

        # Load metadata if exists
        metadata_path = path.parent / f"{path.stem}_metadata.json"
        metadata = None
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        # Create instance
        instance = cls.__new__(cls)
        instance._model = data['model']
        instance._feature_names = data['feature_names']
        instance._is_trained = data['is_trained']
        instance.hyperparameters = data['hyperparameters']

        if metadata:
            instance.model_id = metadata.get('model_id', path.stem)
            instance.model_type = metadata.get('model_type', 'unknown')
            instance.version = metadata.get('version', '1.0.0')

        logger.info(f"Model loaded from {path}")

        return instance

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Feature DataFrame
            y: True labels
            metrics: List of metrics to calculate

        Returns:
            Dictionary of metric scores
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, log_loss, mean_squared_error, mean_absolute_error
        )

        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        results = {}

        # Classification metrics
        if metrics is None or 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(y, predictions)

        if metrics is None or 'precision' in metrics:
            try:
                results['precision'] = precision_score(y, predictions, average='weighted', zero_division=0)
            except:
                pass

        if metrics is None or 'recall' in metrics:
            try:
                results['recall'] = recall_score(y, predictions, average='weighted', zero_division=0)
            except:
                pass

        if metrics is None or 'f1' in metrics:
            try:
                results['f1'] = f1_score(y, predictions, average='weighted', zero_division=0)
            except:
                pass

        if (metrics is None or 'auc' in metrics) and probabilities is not None:
            try:
                if len(np.unique(y)) == 2:
                    results['auc'] = roc_auc_score(y, probabilities[:, 1])
                else:
                    results['auc'] = roc_auc_score(y, probabilities, multi_class='ovr')
            except:
                pass

        if (metrics is None or 'log_loss' in metrics) and probabilities is not None:
            try:
                results['log_loss'] = log_loss(y, probabilities)
            except:
                pass

        return results

    def clone(self) -> 'BaseModel':
        """Create a clone of this model with same hyperparameters"""
        # Filter out model_type and version from hyperparameters to avoid duplicate kwargs
        # when subclasses pass these explicitly to super().__init__()
        filtered_params = {
            k: v for k, v in self.hyperparameters.items()
            if k not in ('model_type', 'version')
        }
        return self.__class__(**filtered_params)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_id={self.model_id}, "
            f"trained={self._is_trained}, "
            f"features={len(self._feature_names)})"
        )


class ModelRegistry:
    """
    Central registry for managing multiple models.

    Features:
    - Model versioning
    - Model comparison
    - A/B testing support
    - Champion/Challenger model management
    """

    def __init__(self, storage_path: str = "models"):
        """
        Initialize ModelRegistry.

        Args:
            storage_path: Path for model storage
        """
        self.storage_path = Path(storage_path)
        ensure_dir(self.storage_path)

        self._models: Dict[str, BaseModel] = {}
        self._champion: Optional[str] = None

        logger.info(f"ModelRegistry initialized at {storage_path}")

    def register(
        self,
        model: BaseModel,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Register a model in the registry.

        Args:
            model: Model to register
            tags: Optional tags for the model

        Returns:
            Model ID
        """
        self._models[model.model_id] = model

        # Save model
        model.save(self.storage_path)

        logger.info(f"Registered model: {model.model_id}")

        return model.model_id

    def get(self, model_id: str) -> Optional[BaseModel]:
        """Get model by ID"""
        if model_id in self._models:
            return self._models[model_id]

        # Try to load from disk
        model_path = self.storage_path / f"{model_id}.pkl"
        if model_path.exists():
            model = BaseModel.load(model_path)
            self._models[model_id] = model
            return model

        return None

    def set_champion(self, model_id: str) -> None:
        """Set a model as the champion (production) model"""
        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")

        self._champion = model_id
        logger.info(f"Champion model set: {model_id}")

        audit_logger.log_system_event(
            "CHAMPION_MODEL_SET",
            {"model_id": model_id}
        )

    def get_champion(self) -> Optional[BaseModel]:
        """Get the champion model"""
        if self._champion:
            return self.get(self._champion)
        return None

    def compare_models(
        self,
        model_ids: List[str],
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """
        Compare multiple models.

        Args:
            model_ids: List of model IDs to compare
            X: Test features
            y: Test labels

        Returns:
            DataFrame with comparison metrics
        """
        results = []

        for model_id in model_ids:
            model = self.get(model_id)
            if model:
                metrics = model.evaluate(X, y)
                metrics['model_id'] = model_id
                metrics['model_type'] = model.model_type
                results.append(metrics)

        return pd.DataFrame(results)

    def list_models(
        self,
        model_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all registered models"""
        models = []

        for model_id, model in self._models.items():
            if model_type is None or model.model_type == model_type:
                models.append({
                    'model_id': model_id,
                    'model_type': model.model_type,
                    'version': model.version,
                    'is_champion': model_id == self._champion,
                    'is_trained': model._is_trained
                })

        return models

    def delete(self, model_id: str) -> bool:
        """Delete a model from registry"""
        if model_id in self._models:
            del self._models[model_id]

            # Delete from disk
            model_path = self.storage_path / f"{model_id}.pkl"
            if model_path.exists():
                model_path.unlink()

            metadata_path = self.storage_path / f"{model_id}_metadata.json"
            if metadata_path.exists():
                metadata_path.unlink()

            logger.info(f"Deleted model: {model_id}")
            return True

        return False
