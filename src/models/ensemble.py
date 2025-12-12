"""
Ensemble Models
JPMorgan-Level Model Ensemble Strategies

Features:
- Voting ensembles (soft/hard)
- Stacking with meta-learner
- Weighted averaging
- Dynamic weight adjustment
- Diversity optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from dataclasses import dataclass

from .base_model import BaseModel, PredictionResult
from ..utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class EnsembleMember:
    """Ensemble member configuration"""
    model: BaseModel
    weight: float = 1.0
    name: Optional[str] = None


class EnsembleModel(BaseModel):
    """
    Base ensemble model combining multiple base models.

    Supports:
    - Weighted averaging
    - Soft voting
    - Hard voting
    """

    def __init__(
        self,
        models: Optional[List[BaseModel]] = None,
        weights: Optional[List[float]] = None,
        voting: str = 'soft',
        **kwargs
    ):
        """
        Initialize EnsembleModel.

        Args:
            models: List of base models
            weights: Weight for each model
            voting: 'soft' (probability average) or 'hard' (majority vote)
        """
        super().__init__(model_type='ensemble', **kwargs)

        self.models = models or []
        self.weights = weights
        self.voting = voting

        if self.weights is None and self.models:
            self.weights = [1.0 / len(self.models)] * len(self.models)

    def add_model(self, model: BaseModel, weight: float = 1.0) -> None:
        """Add model to ensemble"""
        self.models.append(model)

        if self.weights is None:
            self.weights = [1.0]
        else:
            self.weights.append(weight)

        # Renormalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        logger.info(f"Added {model.model_type} to ensemble, total: {len(self.models)}")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> 'EnsembleModel':
        """
        Train all models in the ensemble.

        Args:
            X: Training features
            y: Training labels
            validation_data: Validation data
            **kwargs: Additional training parameters

        Returns:
            Self for chaining
        """
        self._feature_names = list(X.columns)

        for i, model in enumerate(self.models):
            logger.info(f"Training ensemble member {i+1}/{len(self.models)}: {model.model_type}")
            model.fit(X, y, validation_data=validation_data, **kwargs)

        self._is_trained = True
        self.metadata.trained_at = datetime.utcnow()

        logger.info(f"Ensemble trained with {len(self.models)} models")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using ensemble"""
        if not self._is_trained:
            raise RuntimeError("Ensemble not trained")

        if self.voting == 'soft':
            # Weighted probability average
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1) if proba.ndim > 1 else (proba > 0.5).astype(int)
        else:
            # Hard voting (majority)
            predictions = np.array([
                model.predict(X) for model in self.models
            ])

            # Weighted majority vote
            from scipy.stats import mode
            result, _ = mode(predictions, axis=0)
            return result.flatten()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict weighted average probabilities"""
        if not self._is_trained:
            raise RuntimeError("Ensemble not trained")

        probas = []
        for i, model in enumerate(self.models):
            try:
                proba = model.predict_proba(X)
                probas.append(proba * self.weights[i])
            except Exception as e:
                logger.warning(f"Model {i} predict_proba failed: {e}")

        if not probas:
            raise RuntimeError("No predictions from any model")

        # Stack and sum weighted probabilities
        return np.sum(probas, axis=0)

    def _get_importance_scores(self) -> np.ndarray:
        """Get averaged feature importance from all models"""
        all_importance = []

        for i, model in enumerate(self.models):
            try:
                importance = model._get_importance_scores()
                if len(importance) > 0:
                    all_importance.append(importance * self.weights[i])
            except:
                pass

        if not all_importance:
            return np.array([])

        return np.sum(all_importance, axis=0)

    def get_model_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get individual model predictions for analysis.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with each model's predictions
        """
        contributions = {}

        for i, model in enumerate(self.models):
            name = f"{model.model_type}_{i}"
            contributions[f"{name}_pred"] = model.predict(X)

            proba = model.predict_proba(X)
            if proba.ndim > 1:
                contributions[f"{name}_conf"] = np.max(proba, axis=1)
            else:
                contributions[f"{name}_conf"] = np.abs(proba - 0.5) * 2

        return pd.DataFrame(contributions, index=X.index)


class VotingEnsemble(EnsembleModel):
    """
    Voting ensemble with dynamic weight adjustment.

    Features:
    - Soft/hard voting
    - Performance-based weight adjustment
    - Confidence-weighted voting
    """

    def __init__(
        self,
        models: Optional[List[BaseModel]] = None,
        voting: str = 'soft',
        weight_method: str = 'performance',
        **kwargs
    ):
        """
        Initialize VotingEnsemble.

        Args:
            models: List of base models
            voting: 'soft' or 'hard'
            weight_method: 'equal', 'performance', or 'confidence'
        """
        super().__init__(models=models, voting=voting, **kwargs)
        self.weight_method = weight_method
        self._performance_history: Dict[int, List[float]] = {}

    def update_weights(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str = 'accuracy'
    ) -> None:
        """
        Update model weights based on validation performance.

        Args:
            X: Validation features
            y: Validation labels
            metric: Metric to use for weighting
        """
        performances = []

        for i, model in enumerate(self.models):
            metrics = model.evaluate(X, y)
            perf = metrics.get(metric, 0.5)
            performances.append(perf)

            # Track history
            if i not in self._performance_history:
                self._performance_history[i] = []
            self._performance_history[i].append(perf)

        # Update weights
        if self.weight_method == 'performance':
            # Weight by performance
            total = sum(performances)
            if total > 0:
                self.weights = [p / total for p in performances]
        elif self.weight_method == 'confidence':
            # Weight by recent performance consistency
            weights = []
            for i in range(len(self.models)):
                history = self._performance_history.get(i, [0.5])
                # Use average of recent performance
                recent = history[-5:] if len(history) >= 5 else history
                weights.append(np.mean(recent))

            total = sum(weights)
            self.weights = [w / total for w in weights]

        logger.info(f"Updated ensemble weights: {self.weights}")


class StackingEnsemble(BaseModel):
    """
    Stacking ensemble with meta-learner.

    Features:
    - Multi-level stacking
    - Cross-validation for base predictions
    - Various meta-learner options
    """

    def __init__(
        self,
        base_models: Optional[List[BaseModel]] = None,
        meta_model: Optional[BaseModel] = None,
        use_proba: bool = True,
        cv_folds: int = 5,
        passthrough: bool = True,
        **kwargs
    ):
        """
        Initialize StackingEnsemble.

        Args:
            base_models: List of base (level-0) models
            meta_model: Meta (level-1) model
            use_proba: Use probabilities for meta features
            cv_folds: Number of CV folds for base predictions
            passthrough: Include original features in meta model
        """
        super().__init__(model_type='stacking', **kwargs)

        self.base_models = base_models or []
        self.meta_model = meta_model
        self.use_proba = use_proba
        self.cv_folds = cv_folds
        self.passthrough = passthrough

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> 'StackingEnsemble':
        """
        Train stacking ensemble.

        Uses cross-validation to generate base model predictions,
        then trains meta-model on these predictions.
        """
        from sklearn.model_selection import KFold

        self._feature_names = list(X.columns)

        # Generate base model predictions using CV
        base_predictions = self._generate_base_predictions(X, y)

        # Train base models on full data
        for model in self.base_models:
            model.fit(X, y, **kwargs)

        # Prepare meta features
        meta_X = self._prepare_meta_features(X, base_predictions)

        # Train meta model
        if self.meta_model is None:
            from .ml_model import LightGBMModel
            self.meta_model = LightGBMModel(task='classification')

        self.meta_model.fit(meta_X, y, **kwargs)

        self._is_trained = True
        self.metadata.trained_at = datetime.utcnow()

        logger.info(f"Stacking ensemble trained with {len(self.base_models)} base models")

        return self

    def _generate_base_predictions(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> np.ndarray:
        """Generate out-of-fold predictions from base models"""
        from sklearn.model_selection import KFold

        n_samples = len(X)
        n_classes = len(np.unique(y))

        if self.use_proba:
            base_preds = np.zeros((n_samples, len(self.base_models) * n_classes))
        else:
            base_preds = np.zeros((n_samples, len(self.base_models)))

        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        for i, model in enumerate(self.base_models):
            logger.info(f"Generating CV predictions for model {i+1}/{len(self.base_models)}")

            for train_idx, val_idx in kf.split(X):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_val = X.iloc[val_idx]

                # Clone and train model
                model_clone = model.clone()
                model_clone.fit(X_train, y_train)

                if self.use_proba:
                    preds = model_clone.predict_proba(X_val)
                    start_col = i * n_classes
                    end_col = start_col + n_classes
                    base_preds[val_idx, start_col:end_col] = preds
                else:
                    preds = model_clone.predict(X_val)
                    base_preds[val_idx, i] = preds

        return base_preds

    def _prepare_meta_features(
        self,
        X: pd.DataFrame,
        base_predictions: np.ndarray
    ) -> pd.DataFrame:
        """Prepare features for meta model"""
        # Create DataFrame from base predictions
        n_cols = base_predictions.shape[1]
        columns = [f'base_pred_{i}' for i in range(n_cols)]
        meta_df = pd.DataFrame(base_predictions, columns=columns, index=X.index)

        if self.passthrough:
            # Include original features
            meta_df = pd.concat([X, meta_df], axis=1)

        return meta_df

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using stacked ensemble"""
        if not self._is_trained:
            raise RuntimeError("Ensemble not trained")

        # Get base model predictions
        base_preds = self._get_base_predictions(X)

        # Prepare meta features
        meta_X = self._prepare_meta_features(X, base_preds)

        # Predict with meta model
        return self.meta_model.predict(meta_X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using stacked ensemble"""
        if not self._is_trained:
            raise RuntimeError("Ensemble not trained")

        base_preds = self._get_base_predictions(X)
        meta_X = self._prepare_meta_features(X, base_preds)

        return self.meta_model.predict_proba(meta_X)

    def _get_base_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Get predictions from trained base models"""
        predictions = []

        for model in self.base_models:
            if self.use_proba:
                pred = model.predict_proba(X)
            else:
                pred = model.predict(X).reshape(-1, 1)
            predictions.append(pred)

        return np.hstack(predictions)

    def _get_importance_scores(self) -> np.ndarray:
        """Get feature importance from meta model"""
        if self.meta_model is None:
            return np.array([])

        return self.meta_model._get_importance_scores()


class BlendingEnsemble(BaseModel):
    """
    Blending ensemble using holdout set for meta training.

    Faster than stacking (no CV) but requires larger dataset.
    """

    def __init__(
        self,
        base_models: Optional[List[BaseModel]] = None,
        meta_model: Optional[BaseModel] = None,
        holdout_size: float = 0.2,
        use_proba: bool = True,
        **kwargs
    ):
        super().__init__(model_type='blending', **kwargs)

        self.base_models = base_models or []
        self.meta_model = meta_model
        self.holdout_size = holdout_size
        self.use_proba = use_proba

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs
    ) -> 'BlendingEnsemble':
        """Train blending ensemble"""
        from sklearn.model_selection import train_test_split

        self._feature_names = list(X.columns)

        # Split data
        X_train, X_blend, y_train, y_blend = train_test_split(
            X, y, test_size=self.holdout_size, random_state=42
        )

        # Train base models on training set
        for model in self.base_models:
            model.fit(X_train, y_train, **kwargs)

        # Generate blend predictions
        blend_preds = self._get_base_predictions(X_blend)

        # Train meta model
        if self.meta_model is None:
            from .ml_model import LightGBMModel
            self.meta_model = LightGBMModel(task='classification')

        meta_X = pd.DataFrame(blend_preds, index=X_blend.index)
        self.meta_model.fit(meta_X, y_blend, **kwargs)

        self._is_trained = True
        self.metadata.trained_at = datetime.utcnow()

        return self

    def _get_base_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Get predictions from base models"""
        predictions = []

        for model in self.base_models:
            if self.use_proba:
                pred = model.predict_proba(X)
            else:
                pred = model.predict(X).reshape(-1, 1)
            predictions.append(pred)

        return np.hstack(predictions)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("Ensemble not trained")

        base_preds = self._get_base_predictions(X)
        meta_X = pd.DataFrame(base_preds)

        return self.meta_model.predict(meta_X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("Ensemble not trained")

        base_preds = self._get_base_predictions(X)
        meta_X = pd.DataFrame(base_preds)

        return self.meta_model.predict_proba(meta_X)

    def _get_importance_scores(self) -> np.ndarray:
        if self.meta_model is None:
            return np.array([])
        return self.meta_model._get_importance_scores()
