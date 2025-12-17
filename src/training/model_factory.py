"""
Model Factory for standardized model creation and configuration.

This module provides a factory pattern for creating ML models with:
- Consistent interface across model types
- Default hyperparameters per model type
- Optuna parameter space definitions
- GPU/device configuration
- Serialization support

Designed for JPMorgan-level requirements:
- Standardized model interfaces
- Configuration-driven model creation
- Support for ensemble methods
- Production-ready serialization
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import joblib

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Supported model types."""
    LIGHTGBM_CLASSIFIER = "lightgbm_classifier"
    LIGHTGBM_REGRESSOR = "lightgbm_regressor"
    XGBOOST_CLASSIFIER = "xgboost_classifier"
    XGBOOST_REGRESSOR = "xgboost_regressor"
    CATBOOST_CLASSIFIER = "catboost_classifier"
    CATBOOST_REGRESSOR = "catboost_regressor"
    RANDOM_FOREST_CLASSIFIER = "random_forest_classifier"
    RANDOM_FOREST_REGRESSOR = "random_forest_regressor"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
    LOGISTIC_REGRESSION = "logistic_regression"


class TaskType(str, Enum):
    """ML task types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass
class ParamSpace:
    """Parameter space definition for hyperparameter optimization."""
    name: str
    param_type: str  # int, float, float_log, categorical
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    step: Optional[float] = None
    log: bool = False

    def sample(self, trial: "optuna.Trial") -> Any:
        """Sample a value from this parameter space."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna required for hyperparameter sampling")

        if self.param_type == "int":
            return trial.suggest_int(
                self.name, int(self.low), int(self.high),
                step=int(self.step) if self.step else 1
            )
        elif self.param_type == "float":
            return trial.suggest_float(
                self.name, self.low, self.high,
                step=self.step, log=False
            )
        elif self.param_type == "float_log":
            return trial.suggest_float(
                self.name, self.low, self.high, log=True
            )
        elif self.param_type == "categorical":
            return trial.suggest_categorical(self.name, self.choices)
        else:
            raise ValueError(f"Unknown param type: {self.param_type}")


@dataclass
class ModelConfig:
    """Configuration for a model."""
    model_type: ModelType
    task_type: TaskType
    params: Dict[str, Any] = field(default_factory=dict)
    param_spaces: List[ParamSpace] = field(default_factory=list)
    random_state: int = 42
    n_jobs: int = -1
    gpu_enabled: bool = False


# Default parameters for each model type
DEFAULT_PARAMS: Dict[ModelType, Dict[str, Any]] = {
    ModelType.LIGHTGBM_CLASSIFIER: {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 100,
        "max_depth": -1,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "min_child_samples": 20,
        "verbose": -1,
    },
    ModelType.LIGHTGBM_REGRESSOR: {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": 100,
        "max_depth": -1,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "min_child_samples": 20,
        "verbose": -1,
    },
    ModelType.XGBOOST_CLASSIFIER: {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "min_child_weight": 1,
        "gamma": 0,
        "tree_method": "hist",
        "verbosity": 0,
    },
    ModelType.XGBOOST_REGRESSOR: {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "min_child_weight": 1,
        "gamma": 0,
        "tree_method": "hist",
        "verbosity": 0,
    },
    ModelType.CATBOOST_CLASSIFIER: {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 100,
        "depth": 6,
        "learning_rate": 0.1,
        "l2_leaf_reg": 3,
        "verbose": False,
    },
    ModelType.CATBOOST_REGRESSOR: {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 100,
        "depth": 6,
        "learning_rate": 0.1,
        "l2_leaf_reg": 3,
        "verbose": False,
    },
    ModelType.RANDOM_FOREST_CLASSIFIER: {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
    },
    ModelType.RANDOM_FOREST_REGRESSOR: {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
    },
    ModelType.RIDGE: {
        "alpha": 1.0,
        "fit_intercept": True,
        "solver": "auto",
    },
    ModelType.LASSO: {
        "alpha": 1.0,
        "fit_intercept": True,
        "max_iter": 1000,
    },
    ModelType.ELASTIC_NET: {
        "alpha": 1.0,
        "l1_ratio": 0.5,
        "fit_intercept": True,
        "max_iter": 1000,
    },
    ModelType.LOGISTIC_REGRESSION: {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
    },
}

# Default parameter spaces for Optuna optimization
DEFAULT_PARAM_SPACES: Dict[ModelType, List[ParamSpace]] = {
    ModelType.LIGHTGBM_CLASSIFIER: [
        ParamSpace("n_estimators", "int", 50, 500),
        ParamSpace("max_depth", "int", 3, 12),
        ParamSpace("num_leaves", "int", 20, 100),
        ParamSpace("learning_rate", "float_log", 0.01, 0.3),
        ParamSpace("subsample", "float", 0.6, 1.0),
        ParamSpace("colsample_bytree", "float", 0.6, 1.0),
        ParamSpace("reg_alpha", "float_log", 1e-8, 10.0),
        ParamSpace("reg_lambda", "float_log", 1e-8, 10.0),
        ParamSpace("min_child_samples", "int", 10, 100),
    ],
    ModelType.XGBOOST_CLASSIFIER: [
        ParamSpace("n_estimators", "int", 50, 500),
        ParamSpace("max_depth", "int", 3, 12),
        ParamSpace("learning_rate", "float_log", 0.01, 0.3),
        ParamSpace("subsample", "float", 0.6, 1.0),
        ParamSpace("colsample_bytree", "float", 0.6, 1.0),
        ParamSpace("reg_alpha", "float_log", 1e-8, 10.0),
        ParamSpace("reg_lambda", "float_log", 1e-8, 10.0),
        ParamSpace("min_child_weight", "int", 1, 20),
        ParamSpace("gamma", "float", 0, 5),
    ],
    ModelType.CATBOOST_CLASSIFIER: [
        ParamSpace("iterations", "int", 50, 500),
        ParamSpace("depth", "int", 3, 10),
        ParamSpace("learning_rate", "float_log", 0.01, 0.3),
        ParamSpace("l2_leaf_reg", "float_log", 1.0, 10.0),
        ParamSpace("bagging_temperature", "float", 0, 3),
    ],
    ModelType.RANDOM_FOREST_CLASSIFIER: [
        ParamSpace("n_estimators", "int", 50, 300),
        ParamSpace("max_depth", "int", 5, 30),
        ParamSpace("min_samples_split", "int", 2, 20),
        ParamSpace("min_samples_leaf", "int", 1, 10),
        ParamSpace("max_features", "categorical", choices=["sqrt", "log2", 0.5, 0.7]),
    ],
}

# Copy classifier spaces to regressors
DEFAULT_PARAM_SPACES[ModelType.LIGHTGBM_REGRESSOR] = DEFAULT_PARAM_SPACES[ModelType.LIGHTGBM_CLASSIFIER]
DEFAULT_PARAM_SPACES[ModelType.XGBOOST_REGRESSOR] = DEFAULT_PARAM_SPACES[ModelType.XGBOOST_CLASSIFIER]
DEFAULT_PARAM_SPACES[ModelType.CATBOOST_REGRESSOR] = DEFAULT_PARAM_SPACES[ModelType.CATBOOST_CLASSIFIER]
DEFAULT_PARAM_SPACES[ModelType.RANDOM_FOREST_REGRESSOR] = DEFAULT_PARAM_SPACES[ModelType.RANDOM_FOREST_CLASSIFIER]


class ModelFactory:
    """
    Factory for creating and configuring ML models.

    Provides standardized model creation with:
    - Consistent hyperparameter handling
    - GPU detection and configuration
    - Optuna parameter space generation
    - Model serialization/deserialization

    Example:
        # Create model with default params
        model = ModelFactory.create_model(ModelType.LIGHTGBM_CLASSIFIER)

        # Create model with custom params
        model = ModelFactory.create_model(
            ModelType.XGBOOST_CLASSIFIER,
            params={"n_estimators": 200, "max_depth": 8}
        )

        # Get Optuna parameter space
        space = ModelFactory.get_param_space(ModelType.LIGHTGBM_CLASSIFIER)

        # Create model from Optuna trial
        model = ModelFactory.create_model_from_trial(
            ModelType.LIGHTGBM_CLASSIFIER,
            trial
        )
    """

    @staticmethod
    def create_model(
        model_type: Union[ModelType, str],
        params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        gpu_enabled: bool = False,
    ) -> Any:
        """
        Create a model instance.

        Args:
            model_type: Type of model to create
            params: Model hyperparameters (merged with defaults)
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs
            gpu_enabled: Whether to enable GPU acceleration

        Returns:
            Instantiated model object
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type)

        # Get default params and merge with provided
        default_params = DEFAULT_PARAMS.get(model_type, {}).copy()
        if params:
            default_params.update(params)

        # Add random state and n_jobs where applicable
        if "random_state" not in default_params:
            default_params["random_state"] = random_state
        if "n_jobs" not in default_params and "n_jobs" in str(model_type):
            default_params["n_jobs"] = n_jobs

        # Create model based on type
        return ModelFactory._create_model_instance(
            model_type, default_params, gpu_enabled
        )

    @staticmethod
    def _create_model_instance(
        model_type: ModelType,
        params: Dict[str, Any],
        gpu_enabled: bool,
    ) -> Any:
        """Create the actual model instance."""

        # LightGBM models
        if model_type == ModelType.LIGHTGBM_CLASSIFIER:
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available")
            if gpu_enabled:
                params["device"] = "gpu"
            return lgb.LGBMClassifier(**params)

        elif model_type == ModelType.LIGHTGBM_REGRESSOR:
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available")
            if gpu_enabled:
                params["device"] = "gpu"
            return lgb.LGBMRegressor(**params)

        # XGBoost models
        elif model_type == ModelType.XGBOOST_CLASSIFIER:
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available")
            if gpu_enabled:
                params["tree_method"] = "gpu_hist"
                params["predictor"] = "gpu_predictor"
            return xgb.XGBClassifier(**params)

        elif model_type == ModelType.XGBOOST_REGRESSOR:
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available")
            if gpu_enabled:
                params["tree_method"] = "gpu_hist"
                params["predictor"] = "gpu_predictor"
            return xgb.XGBRegressor(**params)

        # CatBoost models
        elif model_type == ModelType.CATBOOST_CLASSIFIER:
            if not CATBOOST_AVAILABLE:
                raise ImportError("CatBoost not available")
            if gpu_enabled:
                params["task_type"] = "GPU"
            return cb.CatBoostClassifier(**params)

        elif model_type == ModelType.CATBOOST_REGRESSOR:
            if not CATBOOST_AVAILABLE:
                raise ImportError("CatBoost not available")
            if gpu_enabled:
                params["task_type"] = "GPU"
            return cb.CatBoostRegressor(**params)

        # Sklearn models
        elif model_type == ModelType.RANDOM_FOREST_CLASSIFIER:
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn not available")
            return RandomForestClassifier(**params)

        elif model_type == ModelType.RANDOM_FOREST_REGRESSOR:
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn not available")
            return RandomForestRegressor(**params)

        elif model_type == ModelType.RIDGE:
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn not available")
            return Ridge(**params)

        elif model_type == ModelType.LASSO:
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn not available")
            return Lasso(**params)

        elif model_type == ModelType.ELASTIC_NET:
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn not available")
            return ElasticNet(**params)

        elif model_type == ModelType.LOGISTIC_REGRESSION:
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn not available")
            return LogisticRegression(**params)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def get_default_params(model_type: Union[ModelType, str]) -> Dict[str, Any]:
        """
        Get default hyperparameters for a model type.

        Args:
            model_type: Type of model

        Returns:
            Dictionary of default parameters
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type)
        return DEFAULT_PARAMS.get(model_type, {}).copy()

    @staticmethod
    def get_param_space(
        model_type: Union[ModelType, str],
    ) -> List[ParamSpace]:
        """
        Get parameter space for hyperparameter optimization.

        Args:
            model_type: Type of model

        Returns:
            List of ParamSpace definitions
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type)
        return DEFAULT_PARAM_SPACES.get(model_type, [])

    @staticmethod
    def create_model_from_trial(
        model_type: Union[ModelType, str],
        trial: "optuna.Trial",
        fixed_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        gpu_enabled: bool = False,
    ) -> Any:
        """
        Create a model with parameters sampled from Optuna trial.

        Args:
            model_type: Type of model
            trial: Optuna trial object
            fixed_params: Parameters that should not be optimized
            random_state: Random seed
            gpu_enabled: Enable GPU acceleration

        Returns:
            Model with sampled parameters
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna required for trial-based model creation")

        if isinstance(model_type, str):
            model_type = ModelType(model_type)

        # Get param space
        param_spaces = ModelFactory.get_param_space(model_type)

        # Sample parameters
        sampled_params = {}
        for space in param_spaces:
            sampled_params[space.name] = space.sample(trial)

        # Merge with fixed params
        if fixed_params:
            sampled_params.update(fixed_params)

        return ModelFactory.create_model(
            model_type,
            params=sampled_params,
            random_state=random_state,
            gpu_enabled=gpu_enabled,
        )

    @staticmethod
    def get_task_type(model_type: Union[ModelType, str]) -> TaskType:
        """
        Get the task type (classification/regression) for a model type.

        Args:
            model_type: Type of model

        Returns:
            TaskType enum value
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type)

        classification_types = [
            ModelType.LIGHTGBM_CLASSIFIER,
            ModelType.XGBOOST_CLASSIFIER,
            ModelType.CATBOOST_CLASSIFIER,
            ModelType.RANDOM_FOREST_CLASSIFIER,
            ModelType.LOGISTIC_REGRESSION,
        ]

        if model_type in classification_types:
            return TaskType.CLASSIFICATION
        return TaskType.REGRESSION

    @staticmethod
    def save_model(
        model: Any,
        path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save a model to disk.

        Args:
            model: Model to save
            path: Path to save to
            metadata: Optional metadata to save alongside model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Determine save method based on model type
        model_type = type(model).__name__

        if "LGBM" in model_type:
            model.booster_.save_model(str(path.with_suffix(".txt")))
        elif "XGB" in model_type:
            model.save_model(str(path.with_suffix(".json")))
        elif "CatBoost" in model_type:
            model.save_model(str(path.with_suffix(".cbm")))
        else:
            # Default: use joblib
            joblib.dump(model, str(path.with_suffix(".joblib")))

        # Save metadata if provided
        if metadata:
            import json
            meta_path = path.with_suffix(".meta.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Model saved to {path}")

    @staticmethod
    def load_model(
        path: Union[str, Path],
        model_type: Optional[Union[ModelType, str]] = None,
    ) -> Any:
        """
        Load a model from disk.

        Args:
            path: Path to load from
            model_type: Type of model (optional, inferred from extension if not provided)

        Returns:
            Loaded model
        """
        path = Path(path)

        if not path.exists():
            # Try common extensions
            for ext in [".joblib", ".txt", ".json", ".cbm"]:
                if path.with_suffix(ext).exists():
                    path = path.with_suffix(ext)
                    break

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        suffix = path.suffix.lower()

        if suffix == ".txt" and LIGHTGBM_AVAILABLE:
            # LightGBM model
            return lgb.Booster(model_file=str(path))

        elif suffix == ".json" and XGBOOST_AVAILABLE:
            # XGBoost model
            model = xgb.Booster()
            model.load_model(str(path))
            return model

        elif suffix == ".cbm" and CATBOOST_AVAILABLE:
            # CatBoost model
            return cb.CatBoost().load_model(str(path))

        else:
            # Default: joblib
            return joblib.load(str(path))

    @staticmethod
    def get_feature_importance(
        model: Any,
        feature_names: Optional[List[str]] = None,
        importance_type: str = "gain",
    ) -> pd.DataFrame:
        """
        Extract feature importance from a model.

        Args:
            model: Trained model
            feature_names: Feature names (optional)
            importance_type: Type of importance (gain, weight, cover)

        Returns:
            DataFrame with feature importance
        """
        model_type = type(model).__name__

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "get_score"):
            # XGBoost Booster
            score = model.get_score(importance_type=importance_type)
            if feature_names:
                importance = [score.get(f, 0) for f in feature_names]
            else:
                feature_names = list(score.keys())
                importance = list(score.values())
        elif hasattr(model, "feature_importance"):
            # LightGBM Booster
            importance = model.feature_importance(importance_type=importance_type)
        else:
            raise ValueError(f"Cannot extract feature importance from {model_type}")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]

        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)

        # Normalize
        df["importance_normalized"] = df["importance"] / df["importance"].sum()

        return df

    @staticmethod
    def detect_gpu() -> bool:
        """Detect if GPU is available for training."""
        try:
            import torch
            return torch.cuda.is_available() or torch.backends.mps.is_available()
        except ImportError:
            pass

        # Try XGBoost GPU detection
        if XGBOOST_AVAILABLE:
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi"], capture_output=True, timeout=5
                )
                return result.returncode == 0
            except Exception:
                pass

        return False

    @staticmethod
    def list_available_models() -> List[ModelType]:
        """List all available model types based on installed packages."""
        available = []

        if LIGHTGBM_AVAILABLE:
            available.extend([
                ModelType.LIGHTGBM_CLASSIFIER,
                ModelType.LIGHTGBM_REGRESSOR,
            ])

        if XGBOOST_AVAILABLE:
            available.extend([
                ModelType.XGBOOST_CLASSIFIER,
                ModelType.XGBOOST_REGRESSOR,
            ])

        if CATBOOST_AVAILABLE:
            available.extend([
                ModelType.CATBOOST_CLASSIFIER,
                ModelType.CATBOOST_REGRESSOR,
            ])

        if SKLEARN_AVAILABLE:
            available.extend([
                ModelType.RANDOM_FOREST_CLASSIFIER,
                ModelType.RANDOM_FOREST_REGRESSOR,
                ModelType.RIDGE,
                ModelType.LASSO,
                ModelType.ELASTIC_NET,
                ModelType.LOGISTIC_REGRESSION,
            ])

        return available
