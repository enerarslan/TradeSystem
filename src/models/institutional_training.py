"""
Institutional-Grade Model Training Pipeline
===========================================

Implements AFML best practices for training ML models on financial data:

1. Sample Weighting (Chapter 4)
   - Concurrent labels have less information
   - Down-weight overlapping samples
   - Time decay for non-stationarity

2. Purged K-Fold Cross-Validation (Chapter 7)
   - Prevents information leakage
   - Embargo period after test sets
   - Critical for serial correlation

3. Bagging with Replacement (Chapter 6)
   - Reduces variance of predictions
   - Sequential bootstrap for dependent data
   - Multiple weak estimators

4. Combinatorial Purged CV (Chapter 12)
   - More realistic backtest
   - Multiple train/test paths

The key insight: Standard ML practices FAIL on financial data due to:
- Serial correlation
- Overlapping labels (from triple-barrier)
- Non-IID samples
- Information leakage

Author: AlphaTrade Institutional System
Based on: Marcos Lopez de Prado - Advances in Financial Machine Learning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
from itertools import combinations

from .training import (
    PurgedKFoldCV, CrossValidationTrainer,
    ClusteredFeatureImportance, feature_importance_with_clustering
)
from ..data.labeling import get_sample_weights, get_time_decay_weights, combine_weights
from ..utils.logger import get_logger, get_audit_logger

logger = get_logger(__name__)
audit_logger = get_audit_logger()


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class InstitutionalTrainingConfig:
    """Configuration for institutional training pipeline."""

    # Cross-validation
    n_splits: int = 5
    purge_gap: int = 10  # Samples to purge before test (label lookback)
    embargo_pct: float = 0.05  # 5% embargo - AFML minimum recommendation
    cv_method: str = 'purged_kfold'  # 'purged_kfold', 'combinatorial', 'walk_forward'

    # Sample weighting
    use_sample_weights: bool = True
    time_decay_factor: float = 0.5
    uniqueness_weight: float = 0.7
    min_weight: float = 0.1  # Minimum sample weight

    # Bagging ensemble
    n_estimators: int = 100
    max_samples: float = 0.5  # Fraction of samples per estimator
    max_features: float = 0.7  # Fraction of features per estimator
    bootstrap: bool = True
    use_sequential_bootstrap: bool = True

    # Training
    early_stopping_rounds: int = 50
    verbose: bool = True
    random_state: int = 42

    # Feature importance
    compute_importance: bool = True
    importance_method: str = 'mda'  # 'mda' or 'mdi'
    n_importance_iterations: int = 10


@dataclass
class TrainingResult:
    """Result of institutional training."""
    model: Any
    train_metrics: Dict[str, float]
    cv_metrics: Dict[str, float]
    feature_importance: Dict[str, float] = field(default_factory=dict)
    cv_fold_results: List[Dict] = field(default_factory=list)
    training_time_seconds: float = 0.0
    config: Dict = field(default_factory=dict)


# =============================================================================
# SEQUENTIAL BOOTSTRAP
# =============================================================================

class SequentialBootstrap:
    """
    Sequential Bootstrap for financial data.

    Standard bootstrap assumes IID samples, which is WRONG for financial data.
    Sequential bootstrap accounts for sample overlap by:
    1. Computing average uniqueness of each sample
    2. Sampling with probability proportional to uniqueness

    Mathematical formulation:
    - Let c_t,i = 1 if sample i spans time t, 0 otherwise
    - Uniqueness u_i = avg_t(1 / sum_j(c_t,j)) for t where c_t,i = 1
    - Sample with P(i) proportional to u_i

    This ensures unique samples are more likely to be selected.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def get_indicators(
        self,
        events: pd.DataFrame,
        close: pd.Series
    ) -> pd.DataFrame:
        """
        Compute indicator matrix for overlapping labels.

        Args:
            events: DataFrame with 't1' column (label end time)
            close: Close price series (for time index)

        Returns:
            Indicator matrix (time x samples)
        """
        # Get time index from close
        time_idx = close.index

        # Create indicator matrix
        indicator = pd.DataFrame(0, index=time_idx, columns=events.index)

        for sample_idx, row in events.iterrows():
            t0 = sample_idx
            t1 = row['t1']

            if pd.isna(t1):
                continue

            # Mark times spanned by this sample
            mask = (time_idx >= t0) & (time_idx <= t1)
            indicator.loc[mask, sample_idx] = 1

        return indicator

    def get_average_uniqueness(
        self,
        indicator: pd.DataFrame
    ) -> pd.Series:
        """
        Compute average uniqueness for each sample.

        Uniqueness = 1 / (number of concurrent samples)

        Args:
            indicator: Indicator matrix from get_indicators()

        Returns:
            Series of average uniqueness per sample
        """
        # Concurrency at each time
        concurrency = indicator.sum(axis=1)

        # Uniqueness at each time (1 / concurrency)
        uniqueness_t = 1 / concurrency.replace(0, np.nan)

        # Average uniqueness per sample
        uniqueness = pd.Series(index=indicator.columns, dtype=float)

        for col in indicator.columns:
            # Get times where this sample is active
            active_times = indicator.index[indicator[col] == 1]

            if len(active_times) > 0:
                uniqueness[col] = uniqueness_t.loc[active_times].mean()
            else:
                uniqueness[col] = 0

        return uniqueness

    def sample_indices(
        self,
        events: pd.DataFrame,
        close: pd.Series,
        n_samples: int = None
    ) -> np.ndarray:
        """
        Generate bootstrap sample indices using sequential bootstrap.

        Args:
            events: DataFrame with 't1' column
            close: Close price series
            n_samples: Number of samples to draw (default: len(events))

        Returns:
            Array of sample indices
        """
        n_samples = n_samples or len(events)

        # Compute indicator matrix
        indicator = self.get_indicators(events, close)

        # Compute average uniqueness
        uniqueness = self.get_average_uniqueness(indicator)

        # Sample with probability proportional to uniqueness
        probs = uniqueness / uniqueness.sum()
        probs = probs.fillna(0)

        # Ensure probabilities sum to 1
        if probs.sum() == 0:
            probs = pd.Series(1 / len(events), index=events.index)

        probs = probs / probs.sum()

        # Draw samples
        indices = np.random.choice(
            len(events),
            size=n_samples,
            replace=True,
            p=probs.values
        )

        if self.verbose:
            logger.info(
                f"Sequential bootstrap: {n_samples} samples, "
                f"avg uniqueness = {uniqueness.mean():.4f}"
            )

        return indices


# =============================================================================
# BAGGING ENSEMBLE
# =============================================================================

class BaggingEnsemble:
    """
    Bagging ensemble with sequential bootstrap for financial ML.

    Key differences from sklearn BaggingClassifier:
    1. Uses sequential bootstrap (not standard bootstrap)
    2. Preserves sample weights
    3. Designed for financial time series

    The ensemble reduces variance by:
    - Training 100+ small estimators
    - Each on different bootstrap sample
    - Each with different feature subset
    - Averaging predictions
    """

    def __init__(
        self,
        base_estimator: Callable,
        n_estimators: int = 100,
        max_samples: float = 0.5,
        max_features: float = 0.7,
        use_sequential_bootstrap: bool = True,
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize BaggingEnsemble.

        Args:
            base_estimator: Callable that returns a new estimator
            n_estimators: Number of estimators
            max_samples: Fraction of samples per estimator
            max_features: Fraction of features per estimator
            use_sequential_bootstrap: Use sequential bootstrap
            n_jobs: Number of parallel jobs
            random_state: Random seed
            verbose: Print progress
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.use_sequential_bootstrap = use_sequential_bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self._estimators: List[Any] = []
        self._feature_subsets: List[List[str]] = []
        self._is_fitted = False
        self._sequential_bootstrap = SequentialBootstrap(verbose=False)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series = None,
        events: pd.DataFrame = None,
        close: pd.Series = None
    ) -> 'BaggingEnsemble':
        """
        Fit the bagging ensemble.

        Args:
            X: Feature DataFrame
            y: Labels
            sample_weight: Sample weights
            events: Events DataFrame (for sequential bootstrap)
            close: Close prices (for sequential bootstrap)

        Returns:
            Self for chaining
        """
        np.random.seed(self.random_state)

        n_samples = int(len(X) * self.max_samples)
        n_features = int(len(X.columns) * self.max_features)

        self._estimators = []
        self._feature_subsets = []

        logger.info(
            f"Training bagging ensemble: "
            f"{self.n_estimators} estimators, "
            f"{n_samples} samples, "
            f"{n_features} features each"
        )

        for i in range(self.n_estimators):
            # Sample indices
            if self.use_sequential_bootstrap and events is not None and close is not None:
                sample_idx = self._sequential_bootstrap.sample_indices(
                    events.loc[X.index],
                    close,
                    n_samples
                )
            else:
                sample_idx = np.random.choice(
                    len(X),
                    size=n_samples,
                    replace=True
                )

            # Feature subset
            feature_idx = np.random.choice(
                len(X.columns),
                size=n_features,
                replace=False
            )
            features = X.columns[feature_idx].tolist()
            self._feature_subsets.append(features)

            # Get training data
            X_train = X.iloc[sample_idx][features]
            y_train = y.iloc[sample_idx]

            # Get weights if provided
            weights_train = None
            if sample_weight is not None:
                weights_train = sample_weight.iloc[sample_idx]

            # Train estimator
            estimator = self.base_estimator()

            if weights_train is not None and hasattr(estimator, 'fit'):
                # Try to pass sample weights
                try:
                    estimator.fit(X_train, y_train, sample_weight=weights_train)
                except TypeError:
                    # Estimator doesn't accept sample_weight
                    estimator.fit(X_train, y_train)
            else:
                estimator.fit(X_train, y_train)

            self._estimators.append(estimator)

            if self.verbose and (i + 1) % 10 == 0:
                logger.info(f"Trained {i + 1}/{self.n_estimators} estimators")

        self._is_fitted = True

        logger.info(f"Bagging ensemble training complete")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature DataFrame

        Returns:
            Array of shape (n_samples, n_classes) with probabilities
        """
        if not self._is_fitted:
            raise ValueError("Ensemble not fitted")

        all_probs = []

        for estimator, features in zip(self._estimators, self._feature_subsets):
            X_subset = X[features]

            if hasattr(estimator, 'predict_proba'):
                probs = estimator.predict_proba(X_subset)
            else:
                # Binary prediction -> probabilities
                preds = estimator.predict(X_subset)
                probs = np.column_stack([1 - preds, preds])

            all_probs.append(probs)

        # Average probabilities
        avg_probs = np.mean(all_probs, axis=0)

        return avg_probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)

    def get_feature_importance(self) -> pd.Series:
        """
        Get aggregated feature importance across all estimators.

        Returns:
            Series with feature importance scores
        """
        importance = {}

        for estimator, features in zip(self._estimators, self._feature_subsets):
            if hasattr(estimator, 'feature_importances_'):
                for feat, imp in zip(features, estimator.feature_importances_):
                    if feat not in importance:
                        importance[feat] = []
                    importance[feat].append(imp)

        # Average importance per feature
        avg_importance = {
            feat: np.mean(imps)
            for feat, imps in importance.items()
        }

        return pd.Series(avg_importance).sort_values(ascending=False)

    def __getstate__(self):
        """Custom pickle: exclude non-serializable base_estimator."""
        state = self.__dict__.copy()
        # Remove callable that can't be pickled
        state['base_estimator'] = None
        return state

    def __setstate__(self, state):
        """Custom unpickle: restore state without base_estimator."""
        self.__dict__.update(state)
        # base_estimator is None after unpickling, but that's OK
        # because the ensemble is already fitted


# =============================================================================
# INSTITUTIONAL TRAINING PIPELINE
# =============================================================================

class InstitutionalTrainingPipeline:
    """
    Complete institutional-grade training pipeline.

    Integrates:
    1. Sample weighting based on label uniqueness
    2. Purged K-fold cross-validation
    3. Bagging ensemble with sequential bootstrap
    4. Clustered feature importance
    5. Comprehensive validation

    This replaces naive train/test split with proper financial ML methodology.
    """

    def __init__(self, config: InstitutionalTrainingConfig = None):
        self.config = config or InstitutionalTrainingConfig()

        # Initialize components
        self._cv = PurgedKFoldCV(
            n_splits=self.config.n_splits,
            purge_gap=self.config.purge_gap,
            embargo_pct=self.config.embargo_pct
        )

        self._training_results: List[TrainingResult] = []

    def compute_sample_weights(
        self,
        events: pd.DataFrame,
        close: pd.Series
    ) -> pd.Series:
        """
        Compute sample weights based on label uniqueness and time decay.

        Args:
            events: DataFrame with 't1' column
            close: Close price series

        Returns:
            Series of sample weights
        """
        if not self.config.use_sample_weights:
            return pd.Series(1.0, index=events.index)

        # Uniqueness-based weights
        uniqueness_weights = get_sample_weights(events, close)

        # Time decay weights
        time_decay_weights = get_time_decay_weights(
            events,
            c=self.config.time_decay_factor
        )

        # Combine
        weights = combine_weights(
            uniqueness_weights,
            time_decay_weights,
            alpha=self.config.uniqueness_weight
        )

        # Apply minimum weight
        weights = weights.clip(lower=self.config.min_weight)

        # Normalize to sum to n
        weights = weights / weights.mean()

        logger.info(
            f"Sample weights: min={weights.min():.4f}, "
            f"max={weights.max():.4f}, mean={weights.mean():.4f}"
        )

        return weights

    def cross_validate(
        self,
        model_factory: Callable,
        X: pd.DataFrame,
        y: pd.Series,
        t1: pd.Series = None,
        sample_weight: pd.Series = None
    ) -> Dict[str, Any]:
        """
        Perform purged cross-validation.

        Args:
            model_factory: Callable that returns a new model instance
            X: Features
            y: Labels
            t1: Label end times for purging
            sample_weight: Sample weights

        Returns:
            Dictionary with CV results
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        logger.info(f"Starting purged {self.config.n_splits}-fold CV")

        fold_results = []
        all_predictions = []
        all_actuals = []

        for fold, (train_idx, test_idx) in enumerate(
            self._cv.split(X, y, t1)
        ):
            logger.info(
                f"Fold {fold + 1}: "
                f"Train size={len(train_idx)}, Test size={len(test_idx)}"
            )

            # Split data
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            # Split weights
            weights_train = None
            if sample_weight is not None:
                weights_train = sample_weight.iloc[train_idx]

            # Create new model instance and train
            model = model_factory()

            if weights_train is not None:
                try:
                    model.fit(X_train, y_train, sample_weight=weights_train)
                except TypeError:
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)

            # Predict and evaluate
            y_pred = model.predict(X_test)

            # Compute metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }

            fold_results.append({
                'fold': fold,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                **metrics
            })

            all_predictions.extend(y_pred)
            all_actuals.extend(y_test.values)

        # Aggregate results
        results_df = pd.DataFrame(fold_results)

        cv_results = {
            'mean_accuracy': results_df['accuracy'].mean(),
            'std_accuracy': results_df['accuracy'].std(),
            'mean_precision': results_df['precision'].mean(),
            'mean_recall': results_df['recall'].mean(),
            'mean_f1': results_df['f1'].mean(),
            'min_accuracy': results_df['accuracy'].min(),
            'max_accuracy': results_df['accuracy'].max(),
            'fold_results': fold_results,
            'all_predictions': all_predictions,
            'all_actuals': all_actuals
        }

        logger.info(
            f"CV complete: "
            f"Accuracy = {cv_results['mean_accuracy']:.4f} "
            f"(+/- {cv_results['std_accuracy']:.4f})"
        )

        return cv_results

    def train_ensemble(
        self,
        base_estimator: Callable,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series = None,
        events: pd.DataFrame = None,
        close: pd.Series = None
    ) -> BaggingEnsemble:
        """
        Train bagging ensemble with sequential bootstrap.

        Args:
            base_estimator: Function returning new estimator
            X: Features
            y: Labels
            sample_weight: Sample weights
            events: Events for sequential bootstrap
            close: Close prices for sequential bootstrap

        Returns:
            Fitted BaggingEnsemble
        """
        ensemble = BaggingEnsemble(
            base_estimator=base_estimator,
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            max_features=self.config.max_features,
            use_sequential_bootstrap=self.config.use_sequential_bootstrap,
            random_state=self.config.random_state,
            verbose=self.config.verbose
        )

        ensemble.fit(
            X, y,
            sample_weight=sample_weight,
            events=events,
            close=close
        )

        return ensemble

    def compute_feature_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.Series:
        """
        Compute feature importance from model.

        Args:
            model: Fitted model
            X: Features
            y: Labels

        Returns:
            Series of feature importance scores
        """
        importance = {}

        # Try to get feature importance from model
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(X.columns, model.feature_importances_))
        elif hasattr(model, 'get_feature_importance'):
            # CatBoost
            importance = dict(zip(X.columns, model.get_feature_importance()))
        else:
            # Fallback: permutation importance (simplified)
            logger.warning("Model doesn't have feature_importances_, using zeros")
            importance = {col: 0.0 for col in X.columns}

        feature_imp = pd.Series(importance).sort_values(ascending=False)

        logger.info(
            f"Feature importance computed, "
            f"top features: {list(feature_imp.head(5).index)}"
        )

        return feature_imp

    def train(
        self,
        model_factory: Callable,
        X: pd.DataFrame,
        y: pd.Series,
        events: pd.DataFrame = None,
        close: pd.Series = None,
        use_ensemble: bool = True
    ) -> TrainingResult:
        """
        Complete institutional training workflow.

        Args:
            model_factory: Function returning new model instance
            X: Features
            y: Labels
            events: Events DataFrame (for sample weights)
            close: Close prices (for sample weights)
            use_ensemble: Whether to use bagging ensemble

        Returns:
            TrainingResult with model and metrics
        """
        import time
        start_time = time.time()

        logger.info(
            f"Starting institutional training: "
            f"{len(X)} samples, {len(X.columns)} features"
        )

        # 1. Compute sample weights
        if events is not None and close is not None:
            sample_weight = self.compute_sample_weights(events, close)
        else:
            sample_weight = pd.Series(1.0, index=X.index)

        # 2. Cross-validation
        t1 = events['t1'] if events is not None and 't1' in events.columns else None

        cv_results = self.cross_validate(
            model_factory, X, y,
            t1=t1,
            sample_weight=sample_weight
        )

        # 3. Train final model (or ensemble)
        if use_ensemble:
            final_model = self.train_ensemble(
                model_factory, X, y,
                sample_weight=sample_weight,
                events=events,
                close=close
            )
        else:
            final_model = model_factory()
            final_model.fit(X, y, sample_weight=sample_weight)

        # 4. Compute feature importance
        feature_importance = {}
        if self.config.compute_importance:
            # Use a single model for importance (not ensemble)
            importance_model = model_factory()
            importance_model.fit(X, y)

            feature_imp = self.compute_feature_importance(
                importance_model, X, y
            )
            feature_importance = feature_imp.to_dict()

        # 5. Compute final metrics
        y_pred = final_model.predict(X)
        train_metrics = self._compute_metrics(y, y_pred)

        training_time = time.time() - start_time

        # Build result
        result = TrainingResult(
            model=final_model,
            train_metrics=train_metrics,
            cv_metrics={
                'accuracy': cv_results['mean_accuracy'],
                'accuracy_std': cv_results['std_accuracy'],
                'f1': cv_results.get('mean_f1', 0),
            },
            feature_importance=feature_importance,
            cv_fold_results=cv_results['fold_results'],
            training_time_seconds=training_time,
            config=vars(self.config)
        )

        self._training_results.append(result)

        logger.info(
            f"Training complete in {training_time:.1f}s: "
            f"Train accuracy={train_metrics['accuracy']:.4f}, "
            f"CV accuracy={cv_results['mean_accuracy']:.4f}"
        )

        # Audit log
        audit_logger.log_system_event(
            "INSTITUTIONAL_MODEL_TRAINED",
            {
                "train_metrics": train_metrics,
                "cv_metrics": cv_results,
                "n_features": len(X.columns),
                "n_samples": len(X),
                "training_time": training_time
            }
        )

        return result

    def _compute_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        try:
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score
            )

            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
        except ImportError:
            accuracy = (y_true == y_pred).mean()
            return {'accuracy': accuracy}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_model_factory(model_type: str = 'catboost', **kwargs) -> Callable:
    """
    Create a model factory function.

    Args:
        model_type: 'catboost', 'xgboost', 'lightgbm', or 'random_forest'
        **kwargs: Model parameters

    Returns:
        Callable that returns new model instance
    """
    def catboost_factory():
        try:
            from catboost import CatBoostClassifier
            return CatBoostClassifier(
                iterations=kwargs.get('iterations', 500),
                depth=kwargs.get('depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.03),
                l2_leaf_reg=kwargs.get('l2_leaf_reg', 3),
                random_state=kwargs.get('random_state', 42),
                verbose=kwargs.get('verbose', 0)
            )
        except ImportError:
            raise ImportError("catboost not installed")

    def xgboost_factory():
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 500),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.03),
                reg_lambda=kwargs.get('reg_lambda', 1),
                random_state=kwargs.get('random_state', 42),
                verbosity=kwargs.get('verbosity', 0)
            )
        except ImportError:
            raise ImportError("xgboost not installed")

    def lightgbm_factory():
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=kwargs.get('n_estimators', 500),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.03),
                reg_lambda=kwargs.get('reg_lambda', 1),
                random_state=kwargs.get('random_state', 42),
                verbose=kwargs.get('verbose', -1)
            )
        except ImportError:
            raise ImportError("lightgbm not installed")

    def rf_factory():
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 10),
            min_samples_split=kwargs.get('min_samples_split', 5),
            random_state=kwargs.get('random_state', 42),
            n_jobs=kwargs.get('n_jobs', -1)
        )

    factories = {
        'catboost': catboost_factory,
        'xgboost': xgboost_factory,
        'lightgbm': lightgbm_factory,
        'random_forest': rf_factory
    }

    if model_type not in factories:
        raise ValueError(f"Unknown model type: {model_type}")

    return factories[model_type]


def train_institutional_model(
    X: pd.DataFrame,
    y: pd.Series,
    events: pd.DataFrame = None,
    close: pd.Series = None,
    model_type: str = 'catboost',
    use_ensemble: bool = True,
    config: InstitutionalTrainingConfig = None,
    **model_kwargs
) -> TrainingResult:
    """
    Convenience function for training institutional model.

    Args:
        X: Features
        y: Labels
        events: Events DataFrame
        close: Close prices
        model_type: Type of base model
        use_ensemble: Use bagging ensemble
        config: Training configuration
        **model_kwargs: Model parameters

    Returns:
        TrainingResult
    """
    config = config or InstitutionalTrainingConfig()

    pipeline = InstitutionalTrainingPipeline(config)
    model_factory = create_model_factory(model_type, **model_kwargs)

    return pipeline.train(
        model_factory=model_factory,
        X=X,
        y=y,
        events=events,
        close=close,
        use_ensemble=use_ensemble
    )
