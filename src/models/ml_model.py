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
                self.hyperparameters['eval_metric'] = 'logloss'  # Binary log loss
                # Remove num_class if it was set for multiclass
                self.hyperparameters.pop('num_class', None)
            else:
                self.hyperparameters['num_class'] = num_classes
                self.hyperparameters['eval_metric'] = 'mlogloss'  # Multi-class log loss
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


class MetaLabelingModel:
    """
    Meta-Labeling Framework for bet sizing and filtering.

    Based on "Advances in Financial Machine Learning" by Marcos López de Prado.

    Meta-labeling works in two stages:
    1. Primary Model: Generates trading signals (side: long/short)
    2. Secondary Model: Predicts probability of primary model being correct

    Benefits:
    - Transforms problem to binary classification (easier to solve)
    - Provides bet sizing through prediction probabilities
    - Allows filtering of low-confidence trades
    - Can use different features for each model
    """

    def __init__(
        self,
        primary_model: BaseModel,
        secondary_model: Optional[BaseModel] = None,
        secondary_model_type: str = 'lightgbm',
        prob_threshold: float = 0.5,
        bet_sizing_method: str = 'linear',
        max_bet_size: float = 1.0
    ):
        """
        Initialize MetaLabelingModel.

        Args:
            primary_model: Model that generates trading signals (side)
            secondary_model: Model that predicts if primary is correct
                            (if None, creates default LightGBM)
            secondary_model_type: Type for auto-created secondary model
            prob_threshold: Minimum probability to take a trade
            bet_sizing_method: 'linear', 'sigmoid', or 'kelly'
            max_bet_size: Maximum position size (as fraction)
        """
        self.primary_model = primary_model
        self.prob_threshold = prob_threshold
        self.bet_sizing_method = bet_sizing_method
        self.max_bet_size = max_bet_size

        # Create secondary model if not provided
        if secondary_model is None:
            if secondary_model_type == 'lightgbm':
                self.secondary_model = LightGBMModel(task='classification')
            elif secondary_model_type == 'xgboost':
                self.secondary_model = XGBoostModel(task='classification')
            elif secondary_model_type == 'catboost':
                self.secondary_model = CatBoostModel(task='classification')
            else:
                from sklearn.ensemble import RandomForestClassifier
                self.secondary_model = RandomForestModel(task='classification')
        else:
            self.secondary_model = secondary_model

        self._is_fitted = False
        self._meta_labels = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        triple_barrier_events: Optional[pd.DataFrame] = None,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> 'MetaLabelingModel':
        """
        Fit the meta-labeling model.

        Args:
            X: Feature DataFrame
            y: Target labels (from triple barrier or other labeling)
            triple_barrier_events: Events DataFrame from triple barrier method
                                   containing 't1', 'ret', 'label', 'barrier'
            validation_data: Validation data tuple
            **kwargs: Additional fit parameters

        Returns:
            Self for chaining
        """
        # Step 1: Get primary model predictions (sides)
        if not self.primary_model._is_trained:
            raise RuntimeError("Primary model must be trained first")

        primary_pred = self.primary_model.predict(X)
        primary_side = pd.Series(primary_pred, index=X.index)

        # Convert to +1/-1 if needed
        if set(primary_side.unique()) == {0, 1}:
            primary_side = primary_side * 2 - 1  # Convert 0/1 to -1/+1

        # Step 2: Create meta-labels
        # Meta-label = 1 if primary prediction matches actual outcome
        if triple_barrier_events is not None:
            # Use triple barrier labels
            actual_labels = triple_barrier_events['label']
        else:
            actual_labels = y

        # Meta-label: 1 if primary was correct direction, 0 otherwise
        # For a long prediction (+1), correct if actual return > 0
        # For a short prediction (-1), correct if actual return < 0
        meta_labels = (primary_side * actual_labels > 0).astype(int)

        self._meta_labels = meta_labels

        # Step 3: Train secondary model on meta-labels
        # Prepare validation data for secondary model
        val_data = None
        if validation_data is not None:
            X_val, y_val = validation_data
            primary_pred_val = self.primary_model.predict(X_val)
            primary_side_val = pd.Series(primary_pred_val, index=X_val.index)
            if set(primary_side_val.unique()) == {0, 1}:
                primary_side_val = primary_side_val * 2 - 1

            if triple_barrier_events is not None and len(triple_barrier_events) == len(y_val):
                actual_val = triple_barrier_events.loc[X_val.index, 'label']
            else:
                actual_val = y_val

            meta_labels_val = (primary_side_val * actual_val > 0).astype(int)
            val_data = (X_val, meta_labels_val)

        # Train secondary model
        self.secondary_model.fit(
            X, meta_labels,
            validation_data=val_data,
            **kwargs
        )

        self._is_fitted = True
        logger.info(
            f"Meta-labeling model fitted. "
            f"Positive rate: {meta_labels.mean():.2%}"
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with meta-labeling.

        Returns the primary model's signal, filtered by secondary model.

        Args:
            X: Feature DataFrame

        Returns:
            Filtered predictions (-1, 0, or 1)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        # Get primary predictions (sides)
        primary_pred = self.primary_model.predict(X)
        primary_side = np.array(primary_pred)

        # Convert to +1/-1 if needed
        if set(np.unique(primary_side)) <= {0, 1}:
            primary_side = primary_side * 2 - 1

        # Get secondary model probability
        proba = self.secondary_model.predict_proba(X)

        # Handle binary vs multi-class probability output
        if proba.ndim == 2:
            # Probability of class 1 (primary being correct)
            confidence = proba[:, 1] if proba.shape[1] == 2 else proba[:, -1]
        else:
            confidence = proba

        # Filter trades based on probability threshold
        # Return 0 (no trade) if confidence is below threshold
        filtered_signal = np.where(
            confidence >= self.prob_threshold,
            primary_side,
            0
        )

        return filtered_signal

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities from secondary model.

        Args:
            X: Feature DataFrame

        Returns:
            Probability array
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        return self.secondary_model.predict_proba(X)

    def get_bet_sizes(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate bet sizes based on meta-model confidence.

        Args:
            X: Feature DataFrame

        Returns:
            Array of bet sizes (0 to max_bet_size)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        # Get probabilities
        proba = self.secondary_model.predict_proba(X)

        if proba.ndim == 2:
            confidence = proba[:, 1] if proba.shape[1] == 2 else proba[:, -1]
        else:
            confidence = proba

        # Calculate bet size based on method
        if self.bet_sizing_method == 'linear':
            # Linear scaling: 0.5 prob -> 0, 1.0 prob -> max_bet
            bet_sizes = np.maximum(0, (confidence - 0.5) * 2) * self.max_bet_size

        elif self.bet_sizing_method == 'sigmoid':
            # Sigmoid scaling for smoother bet sizing
            # Map probability to bet size with steeper curve
            z = (confidence - 0.5) * 10  # Scale and center
            bet_sizes = (1 / (1 + np.exp(-z))) * self.max_bet_size

        elif self.bet_sizing_method == 'kelly':
            # Simplified Kelly criterion
            # f = p - (1-p)/b where b=1 for equal payoffs
            # f = 2p - 1
            bet_sizes = np.maximum(0, 2 * confidence - 1) * self.max_bet_size

        else:
            bet_sizes = confidence * self.max_bet_size

        # Apply threshold filter
        bet_sizes = np.where(confidence >= self.prob_threshold, bet_sizes, 0)

        return bet_sizes

    def get_sized_positions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get full position recommendations with sides and sizes.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with 'signal', 'probability', 'bet_size', 'position'
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        signals = self.predict(X)
        bet_sizes = self.get_bet_sizes(X)

        proba = self.secondary_model.predict_proba(X)
        if proba.ndim == 2:
            confidence = proba[:, 1] if proba.shape[1] == 2 else proba[:, -1]
        else:
            confidence = proba

        positions = pd.DataFrame({
            'signal': signals,
            'probability': confidence,
            'bet_size': bet_sizes,
            'position': signals * bet_sizes  # Signed position size
        }, index=X.index)

        return positions

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate meta-labeling model.

        Args:
            X: Feature DataFrame
            y: True labels

        Returns:
            Dictionary with evaluation metrics
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        predictions = self.predict(X)

        # Calculate metrics
        # Accuracy on filtered trades only
        active_mask = predictions != 0
        if active_mask.sum() > 0:
            active_preds = predictions[active_mask]
            active_actual = y.values[active_mask]

            correct = (np.sign(active_preds) == np.sign(active_actual))
            filtered_accuracy = correct.mean()
        else:
            filtered_accuracy = 0.0

        # Coverage: percentage of samples with non-zero prediction
        coverage = active_mask.mean()

        # Secondary model accuracy on meta-labels
        meta_pred = self.secondary_model.predict(X)
        primary_pred = self.primary_model.predict(X)
        primary_side = np.array(primary_pred)
        if set(np.unique(primary_side)) <= {0, 1}:
            primary_side = primary_side * 2 - 1

        actual_meta = (primary_side * y.values > 0).astype(int)
        meta_accuracy = (meta_pred == actual_meta).mean()

        return {
            'filtered_accuracy': filtered_accuracy,
            'coverage': coverage,
            'meta_model_accuracy': meta_accuracy,
            'n_trades': int(active_mask.sum()),
            'n_samples': len(X)
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from secondary model."""
        return self.secondary_model.get_feature_importance()


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
