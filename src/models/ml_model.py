"""
Gradient Boosting Models
JPMorgan-Level ML Model Implementations

Features:
- XGBoost with GPU support
- LightGBM with advanced parameters
- CatBoost with categorical handling
- Hyperparameter optimization
- Feature importance analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import warnings

from .base_model import BaseModel, ModelMetadata, PredictionResult
from ..utils.logger import get_logger


logger = get_logger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost classifier/regressor with JPMorgan-level configuration.

    Features:
    - GPU acceleration
    - Early stopping
    - Custom objective functions
    - Monotonic constraints
    """

    DEFAULT_PARAMS = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'random_state': 42,
        'n_jobs': -1
    }

    def __init__(
        self,
        task: str = 'classification',
        use_gpu: bool = False,
        **hyperparameters
    ):
        """
        Initialize XGBoostModel.

        Args:
            task: 'classification' or 'regression'
            use_gpu: Whether to use GPU acceleration
            **hyperparameters: Model hyperparameters
        """
        # Merge with defaults
        params = {**self.DEFAULT_PARAMS, **hyperparameters}

        super().__init__(
            model_type='xgboost',
            **params
        )

        self.task = task
        self.use_gpu = use_gpu

        # Adjust params for task
        if task == 'regression':
            self.hyperparameters['objective'] = 'reg:squarederror'
            self.hyperparameters['eval_metric'] = 'rmse'

        # GPU settings
        if use_gpu:
            self.hyperparameters['tree_method'] = 'gpu_hist'
            self.hyperparameters['predictor'] = 'gpu_predictor'

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = True,
        **kwargs
    ) -> 'XGBoostModel':
        """
        Train XGBoost model.

        Args:
            X: Training features
            y: Training labels
            validation_data: Validation data tuple
            early_stopping_rounds: Early stopping patience
            verbose: Whether to print training progress
            **kwargs: Additional fit parameters

        Returns:
            Self for chaining
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost not installed. Run: pip install xgboost")

        self._feature_names = list(X.columns)

        # Handle class labels
        if self.task == 'classification':
            from sklearn.preprocessing import LabelEncoder
            self._label_encoder = LabelEncoder()
            y_encoded = self._label_encoder.fit_transform(y)
            num_classes = len(self._label_encoder.classes_)

            if num_classes == 2:
                self.hyperparameters['objective'] = 'binary:logistic'
            else:
                self.hyperparameters['num_class'] = num_classes
        else:
            y_encoded = y.values
            self._label_encoder = None

        # Create model
        if self.task == 'classification':
            self._model = xgb.XGBClassifier(**self.hyperparameters)
        else:
            self._model = xgb.XGBRegressor(**self.hyperparameters)

        # Prepare eval set
        eval_set = [(X, y_encoded)]
        if validation_data:
            X_val, y_val = validation_data
            if self._label_encoder:
                y_val_encoded = self._label_encoder.transform(y_val)
            else:
                y_val_encoded = y_val.values
            eval_set.append((X_val, y_val_encoded))

        # Train
        self._model.fit(
            X, y_encoded,
            eval_set=eval_set,
            verbose=verbose,
            **kwargs
        )

        self._is_trained = True
        self.metadata.trained_at = datetime.utcnow()
        self.metadata.features = self._feature_names

        logger.info(f"XGBoost model trained with {len(X)} samples")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self._is_trained:
            raise RuntimeError("Model not trained")

        predictions = self._model.predict(X)

        if self._label_encoder:
            predictions = self._label_encoder.inverse_transform(predictions)

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities"""
        if not self._is_trained:
            raise RuntimeError("Model not trained")

        if self.task == 'regression':
            return self._model.predict(X)

        return self._model.predict_proba(X)

    def _get_importance_scores(self) -> np.ndarray:
        """Get feature importance scores"""
        if self._model is None:
            return np.array([])

        importance = self._model.feature_importances_
        return importance

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate SHAP values for interpretability.

        Args:
            X: Feature DataFrame

        Returns:
            SHAP values array
        """
        try:
            import shap

            explainer = shap.TreeExplainer(self._model)
            shap_values = explainer.shap_values(X)

            return shap_values
        except ImportError:
            logger.warning("shap not installed")
            return np.array([])


class LightGBMModel(BaseModel):
    """
    LightGBM classifier/regressor.

    Features:
    - Leaf-wise tree growth (faster)
    - Categorical feature support
    - GPU acceleration
    - Feature bundling
    """

    DEFAULT_PARAMS = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    def __init__(
        self,
        task: str = 'classification',
        use_gpu: bool = False,
        categorical_features: Optional[List[str]] = None,
        **hyperparameters
    ):
        """
        Initialize LightGBMModel.

        Args:
            task: 'classification' or 'regression'
            use_gpu: Whether to use GPU
            categorical_features: List of categorical feature names
            **hyperparameters: Model hyperparameters
        """
        params = {**self.DEFAULT_PARAMS, **hyperparameters}

        super().__init__(
            model_type='lightgbm',
            **params
        )

        self.task = task
        self.use_gpu = use_gpu
        self.categorical_features = categorical_features or []

        if use_gpu:
            self.hyperparameters['device'] = 'gpu'

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = False,
        **kwargs
    ) -> 'LightGBMModel':
        """Train LightGBM model"""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm not installed. Run: pip install lightgbm")

        self._feature_names = list(X.columns)

        # Handle labels
        if self.task == 'classification':
            from sklearn.preprocessing import LabelEncoder
            self._label_encoder = LabelEncoder()
            y_encoded = self._label_encoder.fit_transform(y)
        else:
            y_encoded = y.values
            self._label_encoder = None

        # Create model
        if self.task == 'classification':
            self._model = lgb.LGBMClassifier(**self.hyperparameters)
        else:
            self._model = lgb.LGBMRegressor(**self.hyperparameters)

        # Prepare callbacks
        callbacks = []
        if early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))
        if verbose:
            callbacks.append(lgb.log_evaluation(period=100))

        # Prepare eval set
        eval_set = [(X, y_encoded)]
        if validation_data:
            X_val, y_val = validation_data
            if self._label_encoder:
                y_val_encoded = self._label_encoder.transform(y_val)
            else:
                y_val_encoded = y_val.values
            eval_set = [(X_val, y_val_encoded)]

        # Identify categorical columns
        cat_features = [
            col for col in self.categorical_features
            if col in X.columns
        ]

        # Train
        self._model.fit(
            X, y_encoded,
            eval_set=eval_set,
            callbacks=callbacks,
            categorical_feature=cat_features if cat_features else 'auto',
            **kwargs
        )

        self._is_trained = True
        self.metadata.trained_at = datetime.utcnow()
        self.metadata.features = self._feature_names

        logger.info(f"LightGBM model trained with {len(X)} samples")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self._is_trained:
            raise RuntimeError("Model not trained")

        predictions = self._model.predict(X)

        if self._label_encoder:
            predictions = self._label_encoder.inverse_transform(predictions)

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities"""
        if not self._is_trained:
            raise RuntimeError("Model not trained")

        if self.task == 'regression':
            return self._model.predict(X)

        return self._model.predict_proba(X)

    def _get_importance_scores(self) -> np.ndarray:
        """Get feature importance scores"""
        if self._model is None:
            return np.array([])

        return self._model.feature_importances_


class CatBoostModel(BaseModel):
    """
    CatBoost classifier/regressor.

    Features:
    - Native categorical feature handling
    - Ordered boosting (reduces overfitting)
    - GPU acceleration
    - Symmetric trees
    """

    DEFAULT_PARAMS = {
        'iterations': 500,
        'depth': 6,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3.0,
        'random_seed': 42,
        'verbose': False,
        'allow_writing_files': False
    }

    def __init__(
        self,
        task: str = 'classification',
        use_gpu: bool = False,
        categorical_features: Optional[List[str]] = None,
        **hyperparameters
    ):
        """
        Initialize CatBoostModel.

        Args:
            task: 'classification' or 'regression'
            use_gpu: Whether to use GPU
            categorical_features: List of categorical feature names
            **hyperparameters: Model hyperparameters
        """
        params = {**self.DEFAULT_PARAMS, **hyperparameters}

        super().__init__(
            model_type='catboost',
            **params
        )

        self.task = task
        self.use_gpu = use_gpu
        self.categorical_features = categorical_features or []

        if use_gpu:
            self.hyperparameters['task_type'] = 'GPU'

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = False,
        **kwargs
    ) -> 'CatBoostModel':
        """Train CatBoost model"""
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError:
            raise ImportError("catboost not installed. Run: pip install catboost")

        self._feature_names = list(X.columns)

        # Handle labels
        if self.task == 'classification':
            from sklearn.preprocessing import LabelEncoder
            self._label_encoder = LabelEncoder()
            y_encoded = self._label_encoder.fit_transform(y)
        else:
            y_encoded = y.values
            self._label_encoder = None

        # Create model
        if self.task == 'classification':
            self._model = CatBoostClassifier(**self.hyperparameters)
        else:
            self._model = CatBoostRegressor(**self.hyperparameters)

        # Identify categorical columns indices
        cat_features = [
            X.columns.get_loc(col)
            for col in self.categorical_features
            if col in X.columns
        ]

        # Prepare eval set
        eval_set = None
        if validation_data:
            X_val, y_val = validation_data
            if self._label_encoder:
                y_val_encoded = self._label_encoder.transform(y_val)
            else:
                y_val_encoded = y_val.values
            eval_set = (X_val, y_val_encoded)

        # Train
        self._model.fit(
            X, y_encoded,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            cat_features=cat_features if cat_features else None,
            verbose=verbose,
            **kwargs
        )

        self._is_trained = True
        self.metadata.trained_at = datetime.utcnow()
        self.metadata.features = self._feature_names

        logger.info(f"CatBoost model trained with {len(X)} samples")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self._is_trained:
            raise RuntimeError("Model not trained")

        predictions = self._model.predict(X)

        if self._label_encoder and self.task == 'classification':
            predictions = self._label_encoder.inverse_transform(predictions.flatten())

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities"""
        if not self._is_trained:
            raise RuntimeError("Model not trained")

        if self.task == 'regression':
            return self._model.predict(X)

        return self._model.predict_proba(X)

    def _get_importance_scores(self) -> np.ndarray:
        """Get feature importance scores"""
        if self._model is None:
            return np.array([])

        return self._model.feature_importances_


class RandomForestModel(BaseModel):
    """
    Random Forest classifier/regressor.

    Baseline model for comparison.
    """

    DEFAULT_PARAMS = {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    }

    def __init__(
        self,
        task: str = 'classification',
        **hyperparameters
    ):
        params = {**self.DEFAULT_PARAMS, **hyperparameters}

        super().__init__(
            model_type='random_forest',
            **params
        )

        self.task = task

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> 'RandomForestModel':
        """Train Random Forest model"""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        self._feature_names = list(X.columns)

        if self.task == 'classification':
            self._model = RandomForestClassifier(**self.hyperparameters)
        else:
            self._model = RandomForestRegressor(**self.hyperparameters)

        self._model.fit(X, y)

        self._is_trained = True
        self.metadata.trained_at = datetime.utcnow()

        logger.info(f"Random Forest trained with {len(X)} samples")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("Model not trained")
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("Model not trained")
        if self.task == 'regression':
            return self._model.predict(X)
        return self._model.predict_proba(X)

    def _get_importance_scores(self) -> np.ndarray:
        if self._model is None:
            return np.array([])
        return self._model.feature_importances_
