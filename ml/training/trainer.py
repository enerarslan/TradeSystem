"""
KURUMSAL ML MODEL EĞİTİM SİSTEMİ
JPMorgan Quantitative Research Division Tarzı

Özellikler:
- Multi-model training pipeline
- Hyperparameter optimization (Optuna, GridSearch, RandomSearch)
- Cross-validation (Time-series aware)
- Feature selection (RFE, importance-based)
- Model ensembling
- Experiment tracking
- Model versioning & registry
- Training scheduling
- GPU acceleration support

Bu modül tüm ML modellerinin eğitimini yönetir.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import pickle
import json
import hashlib
import time
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings

warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import (
    TimeSeriesSplit, 
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.preprocessing import StandardScaler, RobustScaler

# XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Optuna for hyperparameter optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


class ModelType(Enum):
    """Desteklenen model tipleri"""
    XGBOOST_REGRESSOR = "xgboost_regressor"
    XGBOOST_CLASSIFIER = "xgboost_classifier"
    LIGHTGBM_REGRESSOR = "lightgbm_regressor"
    LIGHTGBM_CLASSIFIER = "lightgbm_classifier"
    RANDOM_FOREST = "random_forest"
    LINEAR_REGRESSION = "linear_regression"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"


class TaskType(Enum):
    """Model görev tipi"""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    RANKING = "ranking"


@dataclass
class TrainingConfig:
    """Eğitim konfigürasyonu"""
    # Model
    model_type: ModelType = ModelType.XGBOOST_REGRESSOR
    task_type: TaskType = TaskType.REGRESSION
    
    # Training
    test_size: float = 0.2
    validation_size: float = 0.1
    n_splits: int = 5  # Cross-validation splits
    shuffle: bool = False  # Time-series için False
    
    # Hyperparameter optimization
    optimize_hyperparams: bool = True
    optimization_method: str = "optuna"  # optuna, grid, random
    n_trials: int = 100  # Optuna trials
    optimization_timeout: int = 3600  # seconds
    
    # Feature selection
    feature_selection: bool = True
    feature_selection_method: str = "importance"  # importance, rfe, correlation
    max_features: Optional[int] = None
    feature_importance_threshold: float = 0.01
    
    # Early stopping
    early_stopping_rounds: int = 50
    
    # Scaling
    scale_features: bool = True
    scaler_type: str = "robust"  # standard, robust, minmax
    
    # Persistence
    model_dir: str = "models"
    save_model: bool = True
    save_training_data: bool = False
    
    # Logging
    verbose: int = 1  # 0: silent, 1: progress, 2: detailed


@dataclass
class TrainingResult:
    """Eğitim sonucu"""
    model_id: str
    model_type: str
    task_type: str
    
    # Metrics
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    cv_scores: Optional[List[float]] = None
    
    # Training info
    feature_names: List[str] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    best_params: Optional[Dict[str, Any]] = None
    
    # Meta
    training_time_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    n_samples_train: int = 0
    n_samples_val: int = 0
    n_samples_test: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'model_id': self.model_id,
            'model_type': self.model_type,
            'task_type': self.task_type,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'test_metrics': self.test_metrics,
            'cv_scores': self.cv_scores,
            'feature_importance': dict(list(self.feature_importance.items())[:20]),
            'best_params': self.best_params,
            'training_time_seconds': self.training_time_seconds,
            'timestamp': self.timestamp.isoformat(),
            'n_samples': {
                'train': self.n_samples_train,
                'val': self.n_samples_val,
                'test': self.n_samples_test
            }
        }


@dataclass
class ExperimentLog:
    """Experiment tracking log"""
    experiment_id: str
    name: str
    description: str
    config: TrainingConfig
    results: List[TrainingResult]
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed
    tags: List[str] = field(default_factory=list)


class ModelRegistry:
    """
    Model versiyonlama ve registry sistemi.
    
    Tüm eğitilmiş modeller burada saklanır ve yönetilir.
    """
    
    def __init__(self, registry_path: str = "models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = self.registry_path / "registry.json"
        self.models: Dict[str, Dict] = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Registry'yi yükle"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Registry'yi kaydet"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.models, f, indent=2, default=str)
    
    def register_model(
        self,
        model_id: str,
        model: Any,
        result: TrainingResult,
        tags: List[str] = None
    ) -> str:
        """Model'i registry'ye kaydet"""
        # Model path
        model_path = self.registry_path / f"{model_id}.pkl"
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Register
        self.models[model_id] = {
            'model_id': model_id,
            'model_type': result.model_type,
            'task_type': result.task_type,
            'model_path': str(model_path),
            'metrics': {
                'train': result.train_metrics,
                'val': result.val_metrics,
                'test': result.test_metrics
            },
            'feature_names': result.feature_names,
            'best_params': result.best_params,
            'registered_at': datetime.now().isoformat(),
            'tags': tags or [],
            'status': 'active'
        }
        
        self._save_registry()
        return model_id
    
    def get_model(self, model_id: str) -> Optional[Any]:
        """Model'i yükle"""
        if model_id not in self.models:
            return None
        
        model_path = self.models[model_id]['model_path']
        
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def get_best_model(
        self, 
        model_type: Optional[str] = None,
        metric: str = "val_r2"
    ) -> Optional[Tuple[str, Any]]:
        """En iyi performanslı model'i getir"""
        candidates = []
        
        for model_id, info in self.models.items():
            if info['status'] != 'active':
                continue
            
            if model_type and info['model_type'] != model_type:
                continue
            
            # Parse metric
            metric_parts = metric.split('_')
            split = metric_parts[0]  # train, val, test
            metric_name = '_'.join(metric_parts[1:])
            
            if split in info['metrics'] and metric_name in info['metrics'][split]:
                score = info['metrics'][split][metric_name]
                candidates.append((model_id, score))
        
        if not candidates:
            return None
        
        # Sort by score (descending for most metrics)
        best_id = max(candidates, key=lambda x: x[1])[0]
        
        return best_id, self.get_model(best_id)
    
    def list_models(
        self, 
        model_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict]:
        """Model listesini getir"""
        result = []
        
        for model_id, info in self.models.items():
            if model_type and info['model_type'] != model_type:
                continue
            
            if tags and not any(t in info['tags'] for t in tags):
                continue
            
            result.append(info)
        
        return result
    
    def deprecate_model(self, model_id: str):
        """Model'i deprecated olarak işaretle"""
        if model_id in self.models:
            self.models[model_id]['status'] = 'deprecated'
            self._save_registry()


class MLTrainer:
    """
    Kurumsal seviye ML model eğitim sistemi.
    
    Kullanım:
        trainer = MLTrainer(config=TrainingConfig())
        
        # Tek model eğit
        result = trainer.train(X_train, y_train, X_test, y_test)
        
        # Multi-model karşılaştırma
        results = trainer.train_multiple_models(X, y, models=[...])
        
        # Hyperparameter optimization
        best_params = trainer.optimize_hyperparameters(X, y)
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        
        # Model registry
        self.registry = ModelRegistry(
            registry_path=str(Path(self.config.model_dir) / "registry")
        )
        
        # Experiment tracking
        self.experiments: Dict[str, ExperimentLog] = {}
        self.current_experiment: Optional[str] = None
        
        # Scaler
        self.scaler = self._get_scaler()
        
        # Feature names
        self.feature_names: List[str] = []
        self.selected_features: List[str] = []
        
        print(f"🎯 MLTrainer initialized")
        print(f"   Model Type: {self.config.model_type.value}")
        print(f"   Task Type: {self.config.task_type.value}")
        print(f"   Hyperparameter Optimization: {self.config.optimize_hyperparams}")
    
    def _get_scaler(self):
        """Scaler oluştur"""
        if self.config.scaler_type == "standard":
            return StandardScaler()
        elif self.config.scaler_type == "robust":
            return RobustScaler()
        else:
            return StandardScaler()
    
    def _create_model(self, params: Optional[Dict] = None):
        """Model instance oluştur"""
        model_type = self.config.model_type
        default_params = params or {}
        
        if model_type == ModelType.XGBOOST_REGRESSOR:
            if not HAS_XGBOOST:
                raise ImportError("XGBoost not installed")
            return xgb.XGBRegressor(
                n_estimators=default_params.get('n_estimators', 100),
                max_depth=default_params.get('max_depth', 6),
                learning_rate=default_params.get('learning_rate', 0.1),
                subsample=default_params.get('subsample', 0.8),
                colsample_bytree=default_params.get('colsample_bytree', 0.8),
                random_state=42,
                n_jobs=-1
            )
        
        elif model_type == ModelType.XGBOOST_CLASSIFIER:
            if not HAS_XGBOOST:
                raise ImportError("XGBoost not installed")
            return xgb.XGBClassifier(
                n_estimators=default_params.get('n_estimators', 100),
                max_depth=default_params.get('max_depth', 6),
                learning_rate=default_params.get('learning_rate', 0.1),
                subsample=default_params.get('subsample', 0.8),
                colsample_bytree=default_params.get('colsample_bytree', 0.8),
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        
        elif model_type == ModelType.LIGHTGBM_REGRESSOR:
            if not HAS_LIGHTGBM:
                raise ImportError("LightGBM not installed")
            return lgb.LGBMRegressor(
                n_estimators=default_params.get('n_estimators', 100),
                max_depth=default_params.get('max_depth', 6),
                learning_rate=default_params.get('learning_rate', 0.1),
                subsample=default_params.get('subsample', 0.8),
                colsample_bytree=default_params.get('colsample_bytree', 0.8),
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        elif model_type == ModelType.LIGHTGBM_CLASSIFIER:
            if not HAS_LIGHTGBM:
                raise ImportError("LightGBM not installed")
            return lgb.LGBMClassifier(
                n_estimators=default_params.get('n_estimators', 100),
                max_depth=default_params.get('max_depth', 6),
                learning_rate=default_params.get('learning_rate', 0.1),
                subsample=default_params.get('subsample', 0.8),
                colsample_bytree=default_params.get('colsample_bytree', 0.8),
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        elif model_type == ModelType.RANDOM_FOREST:
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            if self.config.task_type == TaskType.REGRESSION:
                return RandomForestRegressor(
                    n_estimators=default_params.get('n_estimators', 100),
                    max_depth=default_params.get('max_depth', 10),
                    random_state=42,
                    n_jobs=-1
                )
            else:
                return RandomForestClassifier(
                    n_estimators=default_params.get('n_estimators', 100),
                    max_depth=default_params.get('max_depth', 10),
                    random_state=42,
                    n_jobs=-1
                )
        
        elif model_type == ModelType.LINEAR_REGRESSION:
            from sklearn.linear_model import LinearRegression
            return LinearRegression()
        
        elif model_type == ModelType.RIDGE:
            from sklearn.linear_model import Ridge
            return Ridge(alpha=default_params.get('alpha', 1.0))
        
        elif model_type == ModelType.LASSO:
            from sklearn.linear_model import Lasso
            return Lasso(alpha=default_params.get('alpha', 1.0))
        
        elif model_type == ModelType.ELASTIC_NET:
            from sklearn.linear_model import ElasticNet
            return ElasticNet(
                alpha=default_params.get('alpha', 1.0),
                l1_ratio=default_params.get('l1_ratio', 0.5)
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        params: Optional[Dict] = None
    ) -> Tuple[Any, TrainingResult]:
        """
        Model eğitimi ana fonksiyonu.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            X_test: Test features
            y_test: Test target
            params: Model parameters (if None, will optimize)
        
        Returns:
            Tuple of (trained_model, TrainingResult)
        """
        start_time = time.time()
        
        print("\n" + "="*70)
        print("   🚀 ML MODEL EĞİTİMİ BAŞLIYOR")
        print("="*70)
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # 1. Feature scaling
        if self.config.scale_features:
            print("📊 Feature scaling...")
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_val_scaled = pd.DataFrame(
                self.scaler.transform(X_val),
                columns=X_val.columns,
                index=X_val.index
            ) if X_val is not None else None
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            ) if X_test is not None else None
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
            X_test_scaled = X_test
        
        # 2. Feature selection
        if self.config.feature_selection:
            print("🔍 Feature selection...")
            X_train_scaled, selected_features = self._select_features(
                X_train_scaled, y_train
            )
            self.selected_features = selected_features
            
            if X_val_scaled is not None:
                X_val_scaled = X_val_scaled[selected_features]
            if X_test_scaled is not None:
                X_test_scaled = X_test_scaled[selected_features]
            
            print(f"   Selected {len(selected_features)}/{len(self.feature_names)} features")
        
        # 3. Hyperparameter optimization
        if self.config.optimize_hyperparams and params is None:
            print("⚡ Hyperparameter optimization...")
            best_params = self._optimize_hyperparameters(X_train_scaled, y_train)
            print(f"   Best params: {best_params}")
        else:
            best_params = params or {}
        
        # 4. Create and train model
        print("🎯 Training model...")
        model = self._create_model(best_params)
        
        # Training with early stopping (for tree-based models)
        if hasattr(model, 'fit') and X_val_scaled is not None:
            if isinstance(model, (xgb.XGBRegressor, xgb.XGBClassifier)):
                model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_val_scaled, y_val)],
                    early_stopping_rounds=self.config.early_stopping_rounds,
                    verbose=self.config.verbose > 1
                )
            elif HAS_LIGHTGBM and isinstance(model, (lgb.LGBMRegressor, lgb.LGBMClassifier)):
                model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_val_scaled, y_val)],
                    callbacks=[lgb.early_stopping(self.config.early_stopping_rounds)]
                )
            else:
                model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train_scaled, y_train)
        
        # 5. Calculate metrics
        print("📈 Calculating metrics...")
        train_metrics = self._calculate_metrics(y_train, model.predict(X_train_scaled))
        
        val_metrics = {}
        if X_val_scaled is not None and y_val is not None:
            val_metrics = self._calculate_metrics(y_val, model.predict(X_val_scaled))
        
        test_metrics = {}
        if X_test_scaled is not None and y_test is not None:
            test_metrics = self._calculate_metrics(y_test, model.predict(X_test_scaled))
        
        # 6. Feature importance
        feature_importance = self._get_feature_importance(model)
        
        # 7. Cross-validation scores
        cv_scores = None
        if self.config.n_splits > 1:
            print("🔄 Cross-validation...")
            tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
            scoring = 'neg_mean_squared_error' if self.config.task_type == TaskType.REGRESSION else 'accuracy'
            cv_scores = list(cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring=scoring))
        
        training_time = time.time() - start_time
        
        # 8. Create result
        model_id = self._generate_model_id()
        
        result = TrainingResult(
            model_id=model_id,
            model_type=self.config.model_type.value,
            task_type=self.config.task_type.value,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            cv_scores=cv_scores,
            feature_names=self.selected_features if self.config.feature_selection else self.feature_names,
            feature_importance=feature_importance,
            best_params=best_params,
            training_time_seconds=training_time,
            n_samples_train=len(X_train),
            n_samples_val=len(X_val) if X_val is not None else 0,
            n_samples_test=len(X_test) if X_test is not None else 0
        )
        
        # 9. Register model
        if self.config.save_model:
            self.registry.register_model(model_id, model, result)
            print(f"💾 Model saved: {model_id}")
        
        # 10. Print summary
        self._print_training_summary(result)
        
        return model, result
    
    def _select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Feature selection"""
        method = self.config.feature_selection_method
        
        if method == "importance":
            # Train a quick model for importance
            temp_model = self._create_model({'n_estimators': 50, 'max_depth': 4})
            temp_model.fit(X, y)
            
            importance = self._get_feature_importance(temp_model)
            
            # Filter by threshold
            selected = [
                f for f, imp in importance.items() 
                if imp >= self.config.feature_importance_threshold
            ]
            
            # Apply max features limit
            if self.config.max_features and len(selected) > self.config.max_features:
                selected = selected[:self.config.max_features]
            
            return X[selected], selected
        
        elif method == "rfe":
            # Recursive Feature Elimination
            temp_model = self._create_model({'n_estimators': 50})
            n_features = self.config.max_features or int(len(X.columns) * 0.5)
            
            rfe = RFE(temp_model, n_features_to_select=n_features)
            rfe.fit(X, y)
            
            selected = list(X.columns[rfe.support_])
            return X[selected], selected
        
        elif method == "correlation":
            # Remove highly correlated features
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
            selected = [col for col in X.columns if col not in to_drop]
            
            return X[selected], selected
        
        else:
            return X, list(X.columns)
    
    def _optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """Hyperparameter optimization"""
        method = self.config.optimization_method
        
        if method == "optuna" and HAS_OPTUNA:
            return self._optuna_optimization(X, y)
        elif method == "grid":
            return self._grid_search(X, y)
        elif method == "random":
            return self._random_search(X, y)
        else:
            return {}
    
    def _optuna_optimization(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """Optuna hyperparameter optimization"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }
            
            model = self._create_model(params)
            
            tscv = TimeSeriesSplit(n_splits=3)
            scoring = 'neg_mean_squared_error' if self.config.task_type == TaskType.REGRESSION else 'accuracy'
            
            scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring)
            
            return scores.mean()
        
        # Create study
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.optimization_timeout,
            show_progress_bar=self.config.verbose > 0
        )
        
        return study.best_params
    
    def _grid_search(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """Grid search optimization"""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2]
        }
        
        model = self._create_model()
        tscv = TimeSeriesSplit(n_splits=3)
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error' if self.config.task_type == TaskType.REGRESSION else 'accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        return grid_search.best_params_
    
    def _random_search(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """Random search optimization"""
        from scipy.stats import randint, uniform
        
        param_distributions = {
            'n_estimators': randint(50, 500),
            'max_depth': randint(3, 12),
            'learning_rate': uniform(0.01, 0.29)
        }
        
        model = self._create_model()
        tscv = TimeSeriesSplit(n_splits=3)
        
        random_search = RandomizedSearchCV(
            model,
            param_distributions,
            n_iter=self.config.n_trials,
            cv=tscv,
            scoring='neg_mean_squared_error' if self.config.task_type == TaskType.REGRESSION else 'accuracy',
            n_jobs=-1,
            random_state=42
        )
        
        random_search.fit(X, y)
        
        return random_search.best_params_
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate metrics"""
        metrics = {}
        
        if self.config.task_type == TaskType.REGRESSION:
            metrics['mse'] = float(mean_squared_error(y_true, y_pred))
            metrics['rmse'] = float(np.sqrt(metrics['mse']))
            metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
            metrics['r2'] = float(r2_score(y_true, y_pred))
            
            # Direction accuracy (financial metric)
            direction_true = np.sign(y_true)
            direction_pred = np.sign(y_pred)
            metrics['direction_accuracy'] = float((direction_true == direction_pred).mean())
            
            # Hit rate for positive predictions
            positive_mask = y_pred > 0
            if positive_mask.sum() > 0:
                metrics['positive_hit_rate'] = float(
                    (y_true[positive_mask] > 0).mean()
                )
            else:
                metrics['positive_hit_rate'] = 0.0
        
        else:  # Classification
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics['f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        
        return metrics
    
    def _get_feature_importance(self, model) -> Dict[str, float]:
        """Extract feature importance"""
        importance = {}
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            feature_names = self.selected_features if self.selected_features else self.feature_names
            
            for name, imp in zip(feature_names, importances):
                importance[name] = float(imp)
            
            # Sort by importance
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        elif hasattr(model, 'coef_'):
            coefs = np.abs(model.coef_).flatten()
            feature_names = self.selected_features if self.selected_features else self.feature_names
            
            for name, coef in zip(feature_names, coefs):
                importance[name] = float(coef)
            
            # Normalize
            total = sum(importance.values())
            if total > 0:
                importance = {k: v/total for k, v in importance.items()}
            
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return importance
    
    def _generate_model_id(self) -> str:
        """Generate unique model ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_hash = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"{self.config.model_type.value}_{timestamp}_{random_hash}"
    
    def _print_training_summary(self, result: TrainingResult):
        """Print training summary"""
        print("\n" + "="*70)
        print("   📊 EĞİTİM SONUÇLARI")
        print("="*70)
        print(f"   Model ID        : {result.model_id}")
        print(f"   Model Type      : {result.model_type}")
        print(f"   Training Time   : {result.training_time_seconds:.2f}s")
        print(f"   Features        : {len(result.feature_names)}")
        
        print("\n   📈 TRAINING METRICS:")
        for metric, value in result.train_metrics.items():
            print(f"      {metric:<20}: {value:.6f}")
        
        if result.val_metrics:
            print("\n   📊 VALIDATION METRICS:")
            for metric, value in result.val_metrics.items():
                print(f"      {metric:<20}: {value:.6f}")
        
        if result.test_metrics:
            print("\n   🎯 TEST METRICS:")
            for metric, value in result.test_metrics.items():
                print(f"      {metric:<20}: {value:.6f}")
        
        if result.cv_scores:
            print(f"\n   🔄 CV SCORES: {np.mean(result.cv_scores):.4f} (+/- {np.std(result.cv_scores):.4f})")
        
        if result.feature_importance:
            print("\n   🔝 TOP 10 FEATURES:")
            for i, (feature, importance) in enumerate(list(result.feature_importance.items())[:10], 1):
                print(f"      {i:2}. {feature:<30}: {importance:.6f}")
        
        print("="*70 + "\n")
    
    def predict(
        self,
        model: Any,
        X: pd.DataFrame,
        scale: bool = True
    ) -> np.ndarray:
        """Make predictions with trained model"""
        # Apply feature selection
        if self.selected_features:
            X = X[self.selected_features]
        
        # Scale
        if scale and self.config.scale_features:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        return model.predict(X_scaled)
    
    def save_trainer(self, path: str):
        """Save trainer state"""
        state = {
            'config': self.config,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'selected_features': self.selected_features
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load_trainer(self, path: str):
        """Load trainer state"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.config = state['config']
        self.scaler = state['scaler']
        self.feature_names = state['feature_names']
        self.selected_features = state['selected_features']


# Export
__all__ = [
    'MLTrainer',
    'TrainingConfig',
    'TrainingResult',
    'ModelRegistry',
    'ModelType',
    'TaskType',
    'ExperimentLog'
]