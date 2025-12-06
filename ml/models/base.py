"""
KURUMSAL ML MODEL BASE CLASS
JPMorgan Quantitative Research Division Tarzı

Base sınıf tüm ML modellerinin implement edeceği interface'i tanımlar.
- Fit/Predict interface
- Model persistence (save/load)
- Feature importance
- Performance metrics
- Backtesting integration
"""
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import pickle
import json


@dataclass
class ModelMetadata:
    """Model metadata"""
    model_name: str
    model_type: str  # xgboost, lstm, transformer, etc.
    version: str
    created_at: datetime
    trained_at: Optional[datetime] = None
    feature_names: List[str] = field(default_factory=list)
    target_name: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_samples: int = 0
    validation_samples: int = 0
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Model prediction sonucu"""
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None  # Classification için
    confidence: Optional[np.ndarray] = None
    timestamp: datetime = field(default_factory=datetime.now)
    feature_importance: Optional[Dict[str, float]] = None


@dataclass
class TrainingResult:
    """Training sonucu"""
    success: bool
    train_score: float
    val_score: float
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    training_time_seconds: float
    feature_importance: Dict[str, float]
    best_params: Optional[Dict[str, Any]] = None
    cv_scores: Optional[List[float]] = None


class BaseMLModel(ABC):
    """
    Base ML Model sınıfı - Tüm ML modelleri bundan türer.
    
    Zorunlu metodlar:
    - fit(): Model training
    - predict(): Prediction
    - save(): Model kaydet
    - load(): Model yükle
    
    Opsiyonel metodlar:
    - predict_proba(): Probability prediction
    - feature_importance(): Feature importance scores
    - cross_validate(): Cross-validation
    """
    
    def __init__(
        self,
        model_name: str,
        model_type: str,
        version: str = "1.0.0"
    ):
        """
        Args:
            model_name: Model adı
            model_type: Model tipi (xgboost, lstm, etc.)
            version: Model versiyonu
        """
        self.model_name = model_name
        self.model_type = model_type
        self.version = version
        
        # Metadata
        self.metadata = ModelMetadata(
            model_name=model_name,
            model_type=model_type,
            version=version,
            created_at=datetime.now()
        )
        
        # Model state
        self.model = None
        self.is_fitted = False
        
        # Feature tracking
        self.feature_names: List[str] = []
        self.target_name: str = ""
        
        # Performance tracking
        self.training_history: List[Dict] = []
    
    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> TrainingResult:
        """
        Model training.
        
        Args:
            X: Training features
            y: Training target
            X_val: Validation features
            y_val: Validation target
            **kwargs: Model-specific parameters
        
        Returns:
            TrainingResult: Training sonuçları
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
        return_confidence: bool = False
    ) -> PredictionResult:
        """
        Prediction.
        
        Args:
            X: Features
            return_confidence: Confidence score döndür
        
        Returns:
            PredictionResult: Tahminler ve metadata
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """
        Model'i kaydet.
        
        Args:
            path: Kayıt yolu
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> 'BaseMLModel':
        """
        Model'i yükle.
        
        Args:
            path: Model yolu
        
        Returns:
            self
        """
        pass
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Probability prediction (classification için).
        
        Default implementation - Alt sınıflar override edebilir.
        """
        raise NotImplementedError("predict_proba not implemented for this model type")
    
    def feature_importance(self) -> Dict[str, float]:
        """
        Feature importance scores.
        
        Default implementation - Alt sınıflar override etmeli.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return {}
    
    def get_metadata(self) -> ModelMetadata:
        """Model metadata döndür"""
        return self.metadata
    
    def update_metadata(self, **kwargs):
        """Metadata güncelle"""
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
    
    def validate_features(self, X: pd.DataFrame):
        """
        Feature validation.
        
        Args:
            X: Input features
        
        Raises:
            ValueError: Feature mismatch varsa
        """
        if not self.is_fitted:
            return
        
        if not self.feature_names:
            return
        
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        extra_features = set(X.columns) - set(self.feature_names)
        if extra_features:
            # Warning ama hata verme (extra feature'lar drop edilebilir)
            pass
    
    def preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Feature preprocessing (opsiyonel).
        
        Alt sınıflar override edebilir.
        """
        # Feature order'ı düzelt
        if self.feature_names:
            return X[self.feature_names]
        return X
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: str = "regression"
    ) -> Dict[str, float]:
        """
        Performance metrics hesapla.
        
        Args:
            y_true: Gerçek değerler
            y_pred: Tahminler
            task_type: "regression" veya "classification"
        
        Returns:
            Metrics dictionary
        """
        metrics = {}
        
        if task_type == "regression":
            # Regression metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics['mse'] = float(mean_squared_error(y_true, y_pred))
            metrics['rmse'] = float(np.sqrt(metrics['mse']))
            metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
            metrics['r2'] = float(r2_score(y_true, y_pred))
            
            # Direction accuracy (finance için önemli)
            direction_correct = np.sign(y_true) == np.sign(y_pred)
            metrics['direction_accuracy'] = float(direction_correct.mean())
            
        elif task_type == "classification":
            # Classification metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics['f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        
        return metrics
    
    def log_training(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        **kwargs
    ):
        """
        Training progress log.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            **kwargs: Ek metrikler
        """
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        
        self.training_history.append(log_entry)
    
    def get_training_history(self) -> pd.DataFrame:
        """Training history DataFrame olarak döndür"""
        if not self.training_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.training_history)
    
    def export_metadata(self, path: str):
        """Metadata'yı JSON olarak export et"""
        metadata_dict = {
            'model_name': self.metadata.model_name,
            'model_type': self.metadata.model_type,
            'version': self.metadata.version,
            'created_at': self.metadata.created_at.isoformat(),
            'trained_at': self.metadata.trained_at.isoformat() if self.metadata.trained_at else None,
            'feature_names': self.metadata.feature_names,
            'target_name': self.metadata.target_name,
            'hyperparameters': self.metadata.hyperparameters,
            'training_samples': self.metadata.training_samples,
            'validation_samples': self.metadata.validation_samples,
            'training_metrics': self.metadata.training_metrics,
            'validation_metrics': self.metadata.validation_metrics
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    def __repr__(self) -> str:
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        return f"{self.model_type}(name='{self.model_name}', {fitted_str})"


class EnsembleModel(BaseMLModel):
    """
    Ensemble model - Birden fazla model'in kombinasyonu.
    
    Stacking, voting, averaging gibi yöntemler.
    """
    
    def __init__(
        self,
        models: List[BaseMLModel],
        combination_method: str = "average",  # average, weighted, stacking
        weights: Optional[List[float]] = None
    ):
        """
        Args:
            models: Ensemble'a dahil modeller
            combination_method: Kombinasyon yöntemi
            weights: Model ağırlıkları (weighted için)
        """
        super().__init__(
            model_name="EnsembleModel",
            model_type="ensemble",
            version="1.0.0"
        )
        
        self.models = models
        self.combination_method = combination_method
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        if len(self.weights) != len(self.models):
            raise ValueError("Weights length must match models length")
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> TrainingResult:
        """Tüm modelleri train et"""
        results = []
        
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{len(self.models)}: {model.model_name}")
            result = model.fit(X, y, X_val, y_val, **kwargs)
            results.append(result)
        
        self.is_fitted = True
        
        # Aggregate results
        avg_train_score = np.mean([r.train_score for r in results])
        avg_val_score = np.mean([r.val_score for r in results])
        
        return TrainingResult(
            success=True,
            train_score=avg_train_score,
            val_score=avg_val_score,
            train_metrics={'avg_score': avg_train_score},
            val_metrics={'avg_score': avg_val_score},
            training_time_seconds=sum([r.training_time_seconds for r in results]),
            feature_importance={}
        )
    
    def predict(
        self,
        X: pd.DataFrame,
        return_confidence: bool = False
    ) -> PredictionResult:
        """Ensemble prediction"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted first")
        
        # Tüm modellerden tahmin al
        predictions = []
        for model in self.models:
            pred_result = model.predict(X, return_confidence=False)
            predictions.append(pred_result.predictions)
        
        # Combine predictions
        predictions = np.array(predictions)
        
        if self.combination_method == "average":
            final_pred = predictions.mean(axis=0)
        elif self.combination_method == "weighted":
            final_pred = np.average(predictions, axis=0, weights=self.weights)
        elif self.combination_method == "median":
            final_pred = np.median(predictions, axis=0)
        else:
            final_pred = predictions.mean(axis=0)
        
        # Confidence (variance of predictions)
        confidence = 1.0 - predictions.std(axis=0) if return_confidence else None
        
        return PredictionResult(
            predictions=final_pred,
            confidence=confidence,
            timestamp=datetime.now()
        )
    
    def save(self, directory: str):
        """Ensemble'ı kaydet"""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save each model
        for i, model in enumerate(self.models):
            model.save(str(dir_path / f"model_{i}.pkl"))
        
        # Save ensemble config
        config = {
            'combination_method': self.combination_method,
            'weights': self.weights,
            'num_models': len(self.models)
        }
        
        with open(dir_path / 'ensemble_config.pkl', 'wb') as f:
            pickle.dump(config, f)
    
    def load(self, directory: str) -> 'EnsembleModel':
        """Ensemble'ı yükle"""
        dir_path = Path(directory)
        
        # Load config
        with open(dir_path / 'ensemble_config.pkl', 'rb') as f:
            config = pickle.load(f)
        
        self.combination_method = config['combination_method']
        self.weights = config['weights']
        
        # Load models (bu kısım model tipine göre özelleştirilmeli)
        # Şimdilik placeholder
        
        self.is_fitted = True
        return self


# Export
__all__ = [
    'BaseMLModel',
    'EnsembleModel',
    'ModelMetadata',
    'PredictionResult',
    'TrainingResult'
]