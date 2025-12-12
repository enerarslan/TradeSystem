"""
Model Training Pipeline
JPMorgan-Level Training Infrastructure

Features:
- Walk-forward validation
- Hyperparameter optimization
- Cross-validation strategies
- Model selection
- Performance tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import warnings

from .base_model import BaseModel, ModelMetadata
from ..utils.logger import get_logger, get_audit_logger


logger = get_logger(__name__)
audit_logger = get_audit_logger()


@dataclass
class TrainingConfig:
    """Training configuration"""
    test_size: float = 0.2
    validation_size: float = 0.1
    n_splits: int = 5
    shuffle: bool = False  # Don't shuffle time series
    random_state: int = 42
    early_stopping_rounds: int = 50
    verbose: bool = True


@dataclass
class WalkForwardConfig:
    """Walk-forward optimization configuration"""
    train_periods: int = 2520  # ~6 months of 15-min bars
    test_periods: int = 420   # ~1 month
    step_periods: int = 420   # Step by 1 month
    min_train_periods: int = 1260  # Minimum 3 months
    expanding: bool = False   # Use expanding or sliding window


@dataclass
class TrainingResult:
    """Result of model training"""
    model: BaseModel
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Optional[Dict[str, float]] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    training_time_seconds: float = 0.0
    fold_results: List[Dict[str, float]] = field(default_factory=list)


class ModelTrainer:
    """
    Comprehensive model training pipeline.

    Features:
    - Multiple validation strategies
    - Hyperparameter optimization
    - Ensemble training
    - Performance tracking
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None
    ):
        """
        Initialize ModelTrainer.

        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self._training_history: List[TrainingResult] = []

    def train(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> TrainingResult:
        """
        Train a single model.

        Args:
            model: Model to train
            X: Training features
            y: Training labels
            validation_data: Validation data tuple
            **kwargs: Additional training parameters

        Returns:
            TrainingResult with metrics
        """
        import time

        logger.info(f"Training {model.model_type} model...")
        start_time = time.time()

        # Split data if no validation provided
        if validation_data is None:
            X_train, X_val, y_train, y_val = self._time_series_split(X, y)
        else:
            X_train, y_train = X, y
            X_val, y_val = validation_data

        # Train model
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose=self.config.verbose,
            **kwargs
        )

        training_time = time.time() - start_time

        # Evaluate
        train_metrics = model.evaluate(X_train, y_train)
        val_metrics = model.evaluate(X_val, y_val)

        result = TrainingResult(
            model=model,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            feature_importance=model.get_feature_importance(),
            training_time_seconds=training_time
        )

        # Update model metadata
        model.metadata.performance_metrics = val_metrics
        model.metadata.feature_importance = result.feature_importance

        self._training_history.append(result)

        logger.info(
            f"Training complete. Val metrics: "
            f"Acc={val_metrics.get('accuracy', 0):.4f}, "
            f"F1={val_metrics.get('f1', 0):.4f}"
        )

        # Audit log
        audit_logger.log_system_event(
            "MODEL_TRAINED",
            {
                "model_id": model.model_id,
                "model_type": model.model_type,
                "val_metrics": val_metrics,
                "training_time": training_time
            }
        )

        return result

    def _time_series_split(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data preserving time order"""
        n = len(X)
        train_size = int(n * (1 - self.config.test_size - self.config.validation_size))
        val_size = int(n * self.config.validation_size)

        X_train = X.iloc[:train_size]
        X_val = X.iloc[train_size:train_size + val_size]
        y_train = y.iloc[:train_size]
        y_val = y.iloc[train_size:train_size + val_size]

        return X_train, X_val, y_train, y_val

    def cross_validate(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: Optional[int] = None,
        strategy: str = 'time_series'
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.

        Args:
            model: Model to validate
            X: Features
            y: Labels
            n_splits: Number of CV folds
            strategy: 'time_series', 'kfold', or 'stratified'

        Returns:
            Dictionary with CV results
        """
        n_splits = n_splits or self.config.n_splits

        if strategy == 'time_series':
            from sklearn.model_selection import TimeSeriesSplit
            cv = TimeSeriesSplit(n_splits=n_splits)
        elif strategy == 'kfold':
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=n_splits, shuffle=False)
        elif strategy == 'stratified':
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=n_splits, shuffle=False)
        else:
            raise ValueError(f"Unknown CV strategy: {strategy}")

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            logger.info(f"Training fold {fold + 1}/{n_splits}")

            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            # Clone and train model
            model_clone = model.clone()
            model_clone.fit(X_train, y_train)

            # Evaluate
            metrics = model_clone.evaluate(X_val, y_val)
            metrics['fold'] = fold
            fold_results.append(metrics)

        # Aggregate results
        results_df = pd.DataFrame(fold_results)

        summary = {
            'mean_accuracy': results_df['accuracy'].mean(),
            'std_accuracy': results_df['accuracy'].std(),
            'mean_f1': results_df['f1'].mean() if 'f1' in results_df else None,
            'std_f1': results_df['f1'].std() if 'f1' in results_df else None,
            'fold_results': fold_results,
            'n_splits': n_splits,
            'strategy': strategy
        }

        return summary

    def hyperparameter_search(
        self,
        model_class: type,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict[str, List[Any]],
        method: str = 'grid',
        n_iter: int = 50,
        cv_splits: int = 3
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        """
        Hyperparameter optimization.

        Args:
            model_class: Model class to instantiate
            X: Features
            y: Labels
            param_grid: Parameter grid
            method: 'grid', 'random', or 'optuna'
            n_iter: Number of iterations for random/optuna
            cv_splits: CV splits for evaluation

        Returns:
            Tuple of (best model, best parameters)
        """
        logger.info(f"Starting hyperparameter search ({method})...")

        if method == 'grid':
            return self._grid_search(model_class, X, y, param_grid, cv_splits)
        elif method == 'random':
            return self._random_search(model_class, X, y, param_grid, n_iter, cv_splits)
        elif method == 'optuna':
            return self._optuna_search(model_class, X, y, param_grid, n_iter, cv_splits)
        else:
            raise ValueError(f"Unknown search method: {method}")

    def _grid_search(
        self,
        model_class: type,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict[str, List[Any]],
        cv_splits: int
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        """Grid search hyperparameter optimization"""
        from itertools import product

        # Generate all parameter combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))

        best_score = -np.inf
        best_params = {}
        best_model = None

        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            logger.info(f"Grid search {i+1}/{len(combinations)}: {params}")

            model = model_class(**params)
            cv_results = self.cross_validate(model, X, y, n_splits=cv_splits)

            score = cv_results['mean_accuracy']

            if score > best_score:
                best_score = score
                best_params = params
                best_model = model

        # Train best model on full data
        best_model.fit(X, y)

        logger.info(f"Best params: {best_params}, Score: {best_score:.4f}")

        return best_model, best_params

    def _random_search(
        self,
        model_class: type,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict[str, List[Any]],
        n_iter: int,
        cv_splits: int
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        """Random search hyperparameter optimization"""
        best_score = -np.inf
        best_params = {}
        best_model = None

        for i in range(n_iter):
            # Random parameter selection
            params = {
                key: np.random.choice(values)
                for key, values in param_grid.items()
            }

            logger.info(f"Random search {i+1}/{n_iter}: {params}")

            model = model_class(**params)
            cv_results = self.cross_validate(model, X, y, n_splits=cv_splits)

            score = cv_results['mean_accuracy']

            if score > best_score:
                best_score = score
                best_params = params
                best_model = model

        # Train best model on full data
        best_model.fit(X, y)

        logger.info(f"Best params: {best_params}, Score: {best_score:.4f}")

        return best_model, best_params

    def _optuna_search(
        self,
        model_class: type,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict[str, List[Any]],
        n_iter: int,
        cv_splits: int
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        """Optuna-based hyperparameter optimization"""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not installed, falling back to random search")
            return self._random_search(model_class, X, y, param_grid, n_iter, cv_splits)

        def objective(trial):
            params = {}
            for key, values in param_grid.items():
                if isinstance(values[0], int):
                    params[key] = trial.suggest_int(key, min(values), max(values))
                elif isinstance(values[0], float):
                    params[key] = trial.suggest_float(key, min(values), max(values))
                else:
                    params[key] = trial.suggest_categorical(key, values)

            model = model_class(**params)
            cv_results = self.cross_validate(model, X, y, n_splits=cv_splits)

            return cv_results['mean_accuracy']

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_iter, show_progress_bar=True)

        best_params = study.best_params
        best_model = model_class(**best_params)
        best_model.fit(X, y)

        logger.info(f"Best params: {best_params}, Score: {study.best_value:.4f}")

        return best_model, best_params


class WalkForwardValidator:
    """
    Walk-forward validation for time series models.

    Simulates realistic trading scenario where model is
    periodically retrained on new data.
    """

    def __init__(
        self,
        config: Optional[WalkForwardConfig] = None
    ):
        """
        Initialize WalkForwardValidator.

        Args:
            config: Walk-forward configuration
        """
        self.config = config or WalkForwardConfig()
        self._results: List[Dict[str, Any]] = []

    def validate(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        retrain: bool = True
    ) -> Dict[str, Any]:
        """
        Perform walk-forward validation.

        Args:
            model: Model to validate
            X: Features
            y: Labels
            retrain: Whether to retrain at each step

        Returns:
            Dictionary with validation results
        """
        logger.info("Starting walk-forward validation...")

        train_periods = self.config.train_periods
        test_periods = self.config.test_periods
        step = self.config.step_periods
        expanding = self.config.expanding

        n = len(X)
        fold_results = []
        all_predictions = []
        all_actuals = []

        # Calculate number of folds
        start = train_periods
        fold = 0

        while start + test_periods <= n:
            fold += 1

            # Define train/test indices
            if expanding:
                train_start = 0
            else:
                train_start = max(0, start - train_periods)

            train_end = start
            test_start = start
            test_end = min(start + test_periods, n)

            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]

            logger.info(
                f"Fold {fold}: Train [{train_start}:{train_end}], "
                f"Test [{test_start}:{test_end}]"
            )

            # Train or reuse model
            if retrain or fold == 1:
                model_fold = model.clone()
                model_fold.fit(X_train, y_train)
            else:
                model_fold = model

            # Predict and evaluate
            predictions = model_fold.predict(X_test)
            metrics = model_fold.evaluate(X_test, y_test)

            fold_result = {
                'fold': fold,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_size': train_end - train_start,
                'test_size': test_end - test_start,
                **metrics
            }
            fold_results.append(fold_result)

            all_predictions.extend(predictions)
            all_actuals.extend(y_test.values)

            # Move to next fold
            start += step

        # Aggregate results
        results_df = pd.DataFrame(fold_results)

        summary = {
            'n_folds': len(fold_results),
            'mean_accuracy': results_df['accuracy'].mean(),
            'std_accuracy': results_df['accuracy'].std(),
            'mean_f1': results_df['f1'].mean() if 'f1' in results_df else None,
            'min_accuracy': results_df['accuracy'].min(),
            'max_accuracy': results_df['accuracy'].max(),
            'fold_results': fold_results,
            'all_predictions': all_predictions,
            'all_actuals': all_actuals,
            'config': {
                'train_periods': train_periods,
                'test_periods': test_periods,
                'step_periods': step,
                'expanding': expanding
            }
        }

        self._results.append(summary)

        logger.info(
            f"Walk-forward complete. {len(fold_results)} folds, "
            f"Mean accuracy: {summary['mean_accuracy']:.4f} "
            f"(+/- {summary['std_accuracy']:.4f})"
        )

        return summary

    def get_equity_curve(
        self,
        predictions: List[int],
        actuals: List[int],
        returns: pd.Series
    ) -> pd.Series:
        """
        Calculate equity curve from predictions.

        Args:
            predictions: Model predictions
            actuals: Actual labels
            returns: Actual returns

        Returns:
            Equity curve series
        """
        # Align predictions with returns
        signals = pd.Series(predictions, index=returns.index[-len(predictions):])

        # Calculate strategy returns
        strategy_returns = signals.shift(1) * returns

        # Calculate equity curve
        equity = (1 + strategy_returns).cumprod()

        return equity

    def analyze_results(self) -> pd.DataFrame:
        """Analyze all walk-forward results"""
        if not self._results:
            return pd.DataFrame()

        analysis = []

        for i, result in enumerate(self._results):
            analysis.append({
                'run': i + 1,
                'n_folds': result['n_folds'],
                'mean_accuracy': result['mean_accuracy'],
                'std_accuracy': result['std_accuracy'],
                'min_accuracy': result['min_accuracy'],
                'max_accuracy': result['max_accuracy'],
                'sharpe_ratio': self._calculate_information_ratio(result)
            })

        return pd.DataFrame(analysis)

    def _calculate_information_ratio(self, result: Dict[str, Any]) -> float:
        """Calculate information ratio of predictions"""
        if 'all_predictions' not in result or 'all_actuals' not in result:
            return 0.0

        predictions = np.array(result['all_predictions'])
        actuals = np.array(result['all_actuals'])

        correct = (predictions == actuals).astype(float)

        if correct.std() == 0:
            return 0.0

        return (correct.mean() - 0.5) / correct.std() * np.sqrt(252)


class PurgedKFoldCV:
    """
    Purged K-Fold Cross-Validation for Financial Time Series.

    Based on "Advances in Financial Machine Learning" by Marcos López de Prado.

    Key Features:
    1. Purging: Removes training samples that overlap with test samples
       to prevent information leakage from labels that span multiple periods
    2. Embargo: Adds a gap after each test set before training samples
       to account for serial correlation

    This is essential for financial ML where:
    - Labels often look forward (e.g., triple barrier method)
    - Features may have memory (e.g., rolling windows)
    - Serial correlation exists in returns

    Without purging and embargo, cross-validation scores are overly optimistic
    due to information leakage.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 0,
        embargo_pct: float = 0.05  # AFML recommends at least 5% embargo
    ):
        """
        Initialize PurgedKFoldCV.

        Args:
            n_splits: Number of CV folds
            purge_gap: Number of periods to purge before test set
                      (accounts for label forward-looking window)
            embargo_pct: Percentage of test set size to embargo after test set
                        (accounts for serial correlation)
                        IMPORTANT: Must be at least 0.05 (5%) to eliminate
                        serial correlation leakage. Default changed from 0.01
                        per AFML recommendations.
        """
        if embargo_pct < 0.05:
            logger.warning(
                f"embargo_pct={embargo_pct} is below recommended minimum of 0.05. "
                f"Setting to 0.05 to prevent serial correlation leakage."
            )
            embargo_pct = 0.05

        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        t1: pd.Series = None
    ):
        """
        Generate train/test indices for purged k-fold CV.

        Args:
            X: Feature DataFrame with DatetimeIndex
            y: Labels (optional)
            t1: Series mapping sample start time to end time
                (e.g., entry time to barrier touch time from triple barrier)
                If None, assumes samples don't overlap

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate embargo size
        test_size = n_samples // self.n_splits
        embargo_size = int(test_size * self.embargo_pct)

        for fold in range(self.n_splits):
            # Calculate test set boundaries
            test_start = fold * test_size
            test_end = (fold + 1) * test_size if fold < self.n_splits - 1 else n_samples

            test_indices = indices[test_start:test_end]

            # Build train indices with purging and embargo
            train_indices = []

            for i in indices:
                # Skip if in test set
                if test_start <= i < test_end:
                    continue

                # Apply purging: skip if sample overlaps with test set
                if t1 is not None:
                    sample_end_time = t1.iloc[i]
                    if pd.notna(sample_end_time):
                        test_start_time = X.index[test_start]
                        test_end_time = X.index[test_end - 1]

                        # Sample ends after test starts -> potential leakage
                        if sample_end_time >= test_start_time and i < test_start:
                            continue

                # Apply purge gap before test set
                if test_start - self.purge_gap <= i < test_start:
                    continue

                # Apply embargo after test set
                if test_end <= i < test_end + embargo_size:
                    continue

                train_indices.append(i)

            yield np.array(train_indices), test_indices

    def get_test_indices(self, X: pd.DataFrame) -> List[np.ndarray]:
        """Get list of test indices for each fold."""
        return [test_idx for _, test_idx in self.split(X)]


class PurgedGroupTimeSeriesSplit:
    """
    Purged Group Time Series Split.

    Combines group-aware splitting with purging/embargo for cases where
    samples are grouped (e.g., multiple assets with same timestamp).

    Based on AFML Chapter 7.
    """

    def __init__(
        self,
        n_splits: int = 5,
        max_train_group_size: int = np.inf,
        max_test_group_size: int = np.inf,
        group_gap: int = 1,
        embargo_pct: float = 0.01
    ):
        """
        Initialize PurgedGroupTimeSeriesSplit.

        Args:
            n_splits: Number of CV folds
            max_train_group_size: Maximum groups in training set
            max_test_group_size: Maximum groups in test set
            group_gap: Number of groups to gap between train/test
            embargo_pct: Percentage of data to embargo
        """
        self.n_splits = n_splits
        self.max_train_group_size = max_train_group_size
        self.max_test_group_size = max_test_group_size
        self.group_gap = group_gap
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        groups: pd.Series = None
    ):
        """
        Generate train/test indices with group-aware purging.

        Args:
            X: Feature DataFrame
            y: Labels
            groups: Group labels (e.g., dates for multi-asset data)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        if groups is None:
            # Use index as groups if not provided
            groups = pd.Series(X.index, index=X.index)

        unique_groups = groups.unique()
        n_groups = len(unique_groups)

        # Sort groups chronologically
        unique_groups = np.sort(unique_groups)

        # Calculate test set size in groups
        test_group_size = min(
            n_groups // self.n_splits,
            self.max_test_group_size
        )

        # Calculate embargo in groups
        embargo_groups = int(test_group_size * self.embargo_pct)

        for fold in range(self.n_splits):
            # Test group boundaries
            test_group_start = fold * test_group_size
            test_group_end = min(
                test_group_start + test_group_size,
                n_groups
            )

            # Train group boundaries (with gap and embargo)
            train_group_end = max(0, test_group_start - self.group_gap)
            train_group_start = max(
                0,
                train_group_end - self.max_train_group_size
            )

            # Get actual train groups
            train_groups = unique_groups[train_group_start:train_group_end]
            test_groups = unique_groups[test_group_start:test_group_end]

            # Convert to indices
            train_mask = groups.isin(train_groups)
            test_mask = groups.isin(test_groups)

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV).

    AFML Chapter 12 approach that generates more diverse train/test
    combinations while respecting temporal structure.

    This method:
    1. Divides data into N groups
    2. Uses combinatorial selection for test sets
    3. Applies purging between groups
    4. Generates more paths than standard k-fold
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        purge_gap: int = 0,
        embargo_pct: float = 0.01
    ):
        """
        Initialize CombinatorialPurgedCV.

        Args:
            n_groups: Total number of groups to divide data into
            n_test_groups: Number of groups to use for testing
            purge_gap: Gap between train and test
            embargo_pct: Embargo percentage
        """
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        t1: pd.Series = None
    ):
        """
        Generate combinatorial train/test splits.

        Args:
            X: Feature DataFrame
            y: Labels
            t1: Label end times for purging

        Yields:
            Tuple of (train_indices, test_indices)
        """
        from itertools import combinations

        n_samples = len(X)
        indices = np.arange(n_samples)

        # Divide into groups
        group_size = n_samples // self.n_groups
        groups = []

        for i in range(self.n_groups):
            start = i * group_size
            end = (i + 1) * group_size if i < self.n_groups - 1 else n_samples
            groups.append(indices[start:end])

        # Generate all combinations of test groups
        for test_group_indices in combinations(range(self.n_groups), self.n_test_groups):
            test_indices = np.concatenate([groups[i] for i in test_group_indices])

            # Determine train groups (non-test, with purging)
            train_indices_list = []

            for i in range(self.n_groups):
                if i in test_group_indices:
                    continue

                # Check for purge gap
                should_include = True
                for test_i in test_group_indices:
                    # Skip if adjacent to test group
                    if abs(i - test_i) <= 1:  # Adjacent group check
                        # Apply embargo
                        embargo_size = int(len(groups[i]) * self.embargo_pct)
                        if i < test_i:
                            # Before test: potentially trim end
                            train_indices_list.append(groups[i][:-embargo_size] if embargo_size > 0 else groups[i])
                        else:
                            # After test: potentially trim start
                            train_indices_list.append(groups[i][embargo_size:] if embargo_size > 0 else groups[i])
                        should_include = False
                        break

                if should_include:
                    train_indices_list.append(groups[i])

            if train_indices_list:
                train_indices = np.concatenate(train_indices_list)
                yield train_indices, test_indices

    def get_n_splits(self) -> int:
        """Get number of splits (combinations)."""
        from math import comb
        return comb(self.n_groups, self.n_test_groups)


class CrossValidationTrainer:
    """
    Advanced cross-validation trainer with purging support.

    Integrates PurgedKFoldCV with model training for proper
    financial time series validation.
    """

    def __init__(
        self,
        cv_method: str = 'purged_kfold',
        n_splits: int = 5,
        purge_gap: int = 0,
        embargo_pct: float = 0.01,
        score_metric: str = 'accuracy'
    ):
        """
        Initialize CrossValidationTrainer.

        Args:
            cv_method: 'purged_kfold', 'purged_group', 'combinatorial', or 'standard'
            n_splits: Number of CV folds
            purge_gap: Periods to purge
            embargo_pct: Embargo percentage
            score_metric: Metric to optimize
        """
        self.cv_method = cv_method
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
        self.score_metric = score_metric

        # Initialize CV splitter
        if cv_method == 'purged_kfold':
            self.cv = PurgedKFoldCV(n_splits, purge_gap, embargo_pct)
        elif cv_method == 'purged_group':
            self.cv = PurgedGroupTimeSeriesSplit(n_splits, embargo_pct=embargo_pct)
        elif cv_method == 'combinatorial':
            self.cv = CombinatorialPurgedCV(n_groups=n_splits * 2, n_test_groups=2)
        else:
            from sklearn.model_selection import TimeSeriesSplit
            self.cv = TimeSeriesSplit(n_splits=n_splits)

    def cross_validate(
        self,
        model: 'BaseModel',
        X: pd.DataFrame,
        y: pd.Series,
        t1: pd.Series = None,
        groups: pd.Series = None,
        return_models: bool = False
    ) -> Dict[str, Any]:
        """
        Perform cross-validation with purging.

        Args:
            model: Model to validate
            X: Features
            y: Labels
            t1: Label end times (for purged_kfold)
            groups: Group labels (for purged_group)
            return_models: Whether to return trained models

        Returns:
            Dictionary with CV results
        """
        fold_results = []
        trained_models = []

        # Get splits based on CV method
        if self.cv_method == 'purged_kfold':
            splits = self.cv.split(X, y, t1)
        elif self.cv_method == 'purged_group':
            splits = self.cv.split(X, y, groups)
        elif self.cv_method == 'combinatorial':
            splits = self.cv.split(X, y, t1)
        else:
            splits = self.cv.split(X, y)

        for fold, (train_idx, test_idx) in enumerate(splits):
            logger.info(
                f"Fold {fold + 1}: Train size={len(train_idx)}, Test size={len(test_idx)}"
            )

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            # Clone and train model
            model_clone = model.clone()
            model_clone.fit(X_train, y_train)

            # Evaluate
            metrics = model_clone.evaluate(X_test, y_test)
            metrics['fold'] = fold
            metrics['train_size'] = len(train_idx)
            metrics['test_size'] = len(test_idx)

            fold_results.append(metrics)

            if return_models:
                trained_models.append(model_clone)

        # Aggregate results
        results_df = pd.DataFrame(fold_results)

        summary = {
            'cv_method': self.cv_method,
            'n_splits': len(fold_results),
            'mean_score': results_df[self.score_metric].mean(),
            'std_score': results_df[self.score_metric].std(),
            'min_score': results_df[self.score_metric].min(),
            'max_score': results_df[self.score_metric].max(),
            'fold_results': fold_results
        }

        # Add all metric means
        for col in results_df.columns:
            if col not in ['fold', 'train_size', 'test_size']:
                summary[f'mean_{col}'] = results_df[col].mean()
                summary[f'std_{col}'] = results_df[col].std()

        if return_models:
            summary['trained_models'] = trained_models

        logger.info(
            f"CV complete ({self.cv_method}): "
            f"Mean {self.score_metric}={summary['mean_score']:.4f} "
            f"(+/- {summary['std_score']:.4f})"
        )

        return summary


class ClusteredFeatureImportance:
    """
    Clustered Feature Importance for Financial ML.

    Based on AFML Chapter 8: Feature Importance.

    Standard feature importance methods (MDI, MDA) are unreliable when
    features are correlated because:
    - Importance is diluted across correlated features
    - Substitution effects distort rankings
    - Random selection among correlated features adds noise

    Clustered Feature Importance addresses this by:
    1. Clustering features using hierarchical clustering
    2. Computing importance at the cluster level
    3. Distributing cluster importance to individual features

    Benefits:
    - More stable importance rankings
    - Better handles multicollinearity
    - Identifies redundant feature groups
    - More interpretable results
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        linkage_method: str = 'ward',
        distance_metric: str = 'correlation',
        cluster_method: str = 'silhouette'
    ):
        """
        Initialize ClusteredFeatureImportance.

        Args:
            n_clusters: Number of clusters (auto-determined if None)
            linkage_method: Hierarchical clustering linkage method
                           ('ward', 'complete', 'average', 'single')
            distance_metric: Distance metric for clustering
            cluster_method: Method for determining optimal clusters
                           ('silhouette', 'gap', 'elbow')
        """
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric
        self.cluster_method = cluster_method

        self._cluster_labels: Optional[np.ndarray] = None
        self._linkage_matrix: Optional[np.ndarray] = None
        self._feature_names: Optional[List[str]] = None

    def fit(
        self,
        X: pd.DataFrame,
        n_clusters: Optional[int] = None
    ) -> 'ClusteredFeatureImportance':
        """
        Fit feature clusters using hierarchical clustering.

        Args:
            X: Feature DataFrame
            n_clusters: Override for number of clusters

        Returns:
            Self for chaining
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist, squareform

        self._feature_names = X.columns.tolist()
        n_clusters = n_clusters or self.n_clusters

        # Compute correlation matrix
        corr_matrix = X.corr()

        # Convert correlation to distance
        # distance = sqrt(0.5 * (1 - correlation))
        distance_matrix = np.sqrt(0.5 * (1 - corr_matrix.fillna(0)))

        # Ensure distance matrix is valid (no NaN/Inf)
        distance_matrix = distance_matrix.replace([np.inf, -np.inf], 1.0).fillna(1.0)

        # Convert to condensed form for linkage
        condensed_dist = squareform(distance_matrix.values, checks=False)

        # Perform hierarchical clustering
        self._linkage_matrix = linkage(condensed_dist, method=self.linkage_method)

        # Determine optimal number of clusters if not specified
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(X, condensed_dist)

        # Get cluster labels
        self._cluster_labels = fcluster(self._linkage_matrix, n_clusters, criterion='maxclust')

        logger.info(
            f"Clustered {len(self._feature_names)} features into {n_clusters} clusters"
        )

        return self

    def _find_optimal_clusters(
        self,
        X: pd.DataFrame,
        condensed_dist: np.ndarray
    ) -> int:
        """
        Find optimal number of clusters using specified method.

        Args:
            X: Feature DataFrame
            condensed_dist: Condensed distance matrix

        Returns:
            Optimal number of clusters
        """
        from scipy.cluster.hierarchy import fcluster

        n_features = len(X.columns)
        max_clusters = min(n_features // 2, 20)  # Reasonable upper bound

        if self.cluster_method == 'silhouette':
            return self._silhouette_optimal(X, condensed_dist, max_clusters)
        elif self.cluster_method == 'gap':
            return self._gap_statistic_optimal(X, condensed_dist, max_clusters)
        else:  # elbow
            return self._elbow_optimal(condensed_dist, max_clusters)

    def _silhouette_optimal(
        self,
        X: pd.DataFrame,
        condensed_dist: np.ndarray,
        max_clusters: int
    ) -> int:
        """Find optimal clusters using silhouette score."""
        from scipy.cluster.hierarchy import fcluster
        from scipy.spatial.distance import squareform

        try:
            from sklearn.metrics import silhouette_score
        except ImportError:
            logger.warning("sklearn not available for silhouette, using default 5 clusters")
            return 5

        best_score = -1
        best_n = 2

        # Need square form for silhouette
        distance_matrix = squareform(condensed_dist)

        for n in range(2, max_clusters + 1):
            labels = fcluster(self._linkage_matrix, n, criterion='maxclust')

            # Need at least 2 unique clusters
            if len(np.unique(labels)) < 2:
                continue

            try:
                score = silhouette_score(distance_matrix, labels, metric='precomputed')
                if score > best_score:
                    best_score = score
                    best_n = n
            except Exception:
                continue

        logger.info(f"Optimal clusters (silhouette): {best_n} (score={best_score:.4f})")
        return best_n

    def _gap_statistic_optimal(
        self,
        X: pd.DataFrame,
        condensed_dist: np.ndarray,
        max_clusters: int
    ) -> int:
        """Find optimal clusters using gap statistic."""
        from scipy.cluster.hierarchy import fcluster

        n_references = 10
        gaps = []

        for n in range(1, max_clusters + 1):
            labels = fcluster(self._linkage_matrix, n, criterion='maxclust')
            wk = self._compute_within_cluster_dispersion(X.values, labels)

            # Reference distributions
            ref_wks = []
            for _ in range(n_references):
                # Generate random data with same shape
                random_data = np.random.uniform(
                    X.min().values, X.max().values, X.shape
                )
                ref_labels = fcluster(self._linkage_matrix, n, criterion='maxclust')
                ref_wk = self._compute_within_cluster_dispersion(random_data, ref_labels)
                ref_wks.append(np.log(ref_wk) if ref_wk > 0 else 0)

            gap = np.mean(ref_wks) - (np.log(wk) if wk > 0 else 0)
            gaps.append(gap)

        # Find first significant gap
        best_n = np.argmax(gaps) + 1
        logger.info(f"Optimal clusters (gap statistic): {best_n}")
        return best_n

    def _elbow_optimal(
        self,
        condensed_dist: np.ndarray,
        max_clusters: int
    ) -> int:
        """Find optimal clusters using elbow method."""
        from scipy.cluster.hierarchy import fcluster

        dispersions = []

        for n in range(1, max_clusters + 1):
            labels = fcluster(self._linkage_matrix, n, criterion='maxclust')
            # Use linkage matrix heights as proxy for dispersion
            dispersions.append(self._linkage_matrix[-(n-1), 2] if n > 1 else 0)

        # Find elbow using second derivative
        if len(dispersions) < 3:
            return 2

        second_deriv = np.diff(dispersions, n=2)
        if len(second_deriv) > 0:
            best_n = np.argmax(second_deriv) + 2
        else:
            best_n = 2

        logger.info(f"Optimal clusters (elbow): {best_n}")
        return max(2, min(best_n, max_clusters))

    def _compute_within_cluster_dispersion(
        self,
        X: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Compute total within-cluster dispersion."""
        wk = 0
        for cluster in np.unique(labels):
            cluster_points = X[labels == cluster]
            if len(cluster_points) > 1:
                centroid = cluster_points.mean(axis=0)
                wk += np.sum((cluster_points - centroid) ** 2)
        return wk

    def get_cluster_assignments(self) -> Dict[str, int]:
        """
        Get feature to cluster assignments.

        Returns:
            Dictionary mapping feature names to cluster IDs
        """
        if self._cluster_labels is None:
            raise ValueError("Must fit() first")

        return dict(zip(self._feature_names, self._cluster_labels))

    def get_clusters(self) -> Dict[int, List[str]]:
        """
        Get cluster to features mapping.

        Returns:
            Dictionary mapping cluster IDs to feature names
        """
        if self._cluster_labels is None:
            raise ValueError("Must fit() first")

        clusters = {}
        for feature, cluster in zip(self._feature_names, self._cluster_labels):
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(feature)

        return clusters

    def compute_clustered_importance(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'mda',
        n_iterations: int = 10
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute feature importance at cluster level then distribute.

        Args:
            model: Fitted model with feature_importances_ or predict method
            X: Feature DataFrame
            y: Labels
            method: 'mda' (Mean Decrease Accuracy) or 'mdi' (Mean Decrease Impurity)
            n_iterations: Iterations for MDA

        Returns:
            Tuple of (cluster_importance, feature_importance)
        """
        if self._cluster_labels is None:
            self.fit(X)

        clusters = self.get_clusters()
        n_clusters = len(clusters)

        if method == 'mdi':
            cluster_importance = self._compute_mdi_cluster_importance(model, clusters)
        else:  # mda
            cluster_importance = self._compute_mda_cluster_importance(
                model, X, y, clusters, n_iterations
            )

        # Distribute cluster importance to individual features
        feature_importance = self._distribute_importance(cluster_importance, clusters)

        return cluster_importance, feature_importance

    def _compute_mdi_cluster_importance(
        self,
        model,
        clusters: Dict[int, List[str]]
    ) -> pd.Series:
        """
        Compute MDI (Mean Decrease Impurity) at cluster level.

        For tree-based models, sums feature importances within clusters.
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model must have feature_importances_ for MDI")

        # Get feature importances from model
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
        else:
            feature_names = self._feature_names

        importances = dict(zip(feature_names, model.feature_importances_))

        # Sum importances within clusters
        cluster_importance = {}
        for cluster_id, features in clusters.items():
            cluster_importance[cluster_id] = sum(
                importances.get(f, 0) for f in features
            )

        # Normalize
        total = sum(cluster_importance.values())
        if total > 0:
            cluster_importance = {k: v / total for k, v in cluster_importance.items()}

        return pd.Series(cluster_importance).sort_values(ascending=False)

    def _compute_mda_cluster_importance(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        clusters: Dict[int, List[str]],
        n_iterations: int
    ) -> pd.Series:
        """
        Compute MDA (Mean Decrease Accuracy) at cluster level.

        Permutes all features in a cluster simultaneously to measure
        cluster-level importance.
        """
        try:
            from sklearn.metrics import accuracy_score
        except ImportError:
            logger.warning("sklearn not available for MDA")
            return pd.Series()

        # Get baseline score
        y_pred = model.predict(X)
        baseline_score = accuracy_score(y, y_pred)

        cluster_importance = {}

        for cluster_id, features in clusters.items():
            importance_scores = []

            for _ in range(n_iterations):
                X_permuted = X.copy()

                # Permute all features in cluster simultaneously
                for feature in features:
                    if feature in X_permuted.columns:
                        X_permuted[feature] = np.random.permutation(X_permuted[feature].values)

                # Score with permuted features
                y_pred_permuted = model.predict(X_permuted)
                permuted_score = accuracy_score(y, y_pred_permuted)

                # Importance = decrease in accuracy
                importance_scores.append(baseline_score - permuted_score)

            cluster_importance[cluster_id] = np.mean(importance_scores)

        # Normalize (only positive importances)
        min_imp = min(cluster_importance.values())
        if min_imp < 0:
            cluster_importance = {k: v - min_imp for k, v in cluster_importance.items()}

        total = sum(cluster_importance.values())
        if total > 0:
            cluster_importance = {k: v / total for k, v in cluster_importance.items()}

        return pd.Series(cluster_importance).sort_values(ascending=False)

    def _distribute_importance(
        self,
        cluster_importance: pd.Series,
        clusters: Dict[int, List[str]]
    ) -> pd.Series:
        """
        Distribute cluster importance to individual features.

        Each feature gets its cluster's importance divided by cluster size.
        """
        feature_importance = {}

        for cluster_id, features in clusters.items():
            cluster_imp = cluster_importance.get(cluster_id, 0)
            feature_imp = cluster_imp / len(features)

            for feature in features:
                feature_importance[feature] = feature_imp

        return pd.Series(feature_importance).sort_values(ascending=False)

    def select_features(
        self,
        feature_importance: pd.Series,
        method: str = 'top_n',
        n: int = 20,
        threshold: float = 0.01
    ) -> List[str]:
        """
        Select features based on clustered importance.

        Args:
            feature_importance: Series of feature importances
            method: 'top_n', 'threshold', or 'top_per_cluster'
            n: Number of features for 'top_n'
            threshold: Minimum importance for 'threshold'

        Returns:
            List of selected feature names
        """
        if method == 'top_n':
            return feature_importance.head(n).index.tolist()

        elif method == 'threshold':
            return feature_importance[feature_importance >= threshold].index.tolist()

        elif method == 'top_per_cluster':
            # Select top feature from each cluster
            clusters = self.get_clusters()
            selected = []

            for cluster_id, features in clusters.items():
                cluster_features = feature_importance[features].sort_values(ascending=False)
                if len(cluster_features) > 0:
                    selected.append(cluster_features.index[0])

            return selected

        else:
            raise ValueError(f"Unknown selection method: {method}")

    def drop_redundant_clusters(
        self,
        cluster_importance: pd.Series,
        threshold: float = 0.01
    ) -> List[int]:
        """
        Identify redundant clusters with low importance.

        Args:
            cluster_importance: Series of cluster importances
            threshold: Minimum importance to keep cluster

        Returns:
            List of cluster IDs to drop
        """
        return cluster_importance[cluster_importance < threshold].index.tolist()


def feature_importance_with_clustering(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_clusters: Optional[int] = None,
    method: str = 'mda',
    n_iterations: int = 10
) -> Dict[str, Any]:
    """
    Convenience function for clustered feature importance analysis.

    Args:
        model: Fitted model
        X: Feature DataFrame
        y: Labels
        n_clusters: Number of clusters (auto if None)
        method: 'mda' or 'mdi'
        n_iterations: Iterations for MDA

    Returns:
        Dictionary with importance results and cluster info
    """
    cfi = ClusteredFeatureImportance(n_clusters=n_clusters)
    cfi.fit(X)

    cluster_imp, feature_imp = cfi.compute_clustered_importance(
        model, X, y, method=method, n_iterations=n_iterations
    )

    clusters = cfi.get_clusters()

    # Identify redundant clusters
    redundant = cfi.drop_redundant_clusters(cluster_imp, threshold=0.01)

    # Get recommended features
    selected = cfi.select_features(feature_imp, method='top_per_cluster')

    return {
        'cluster_importance': cluster_imp,
        'feature_importance': feature_imp,
        'clusters': clusters,
        'redundant_clusters': redundant,
        'selected_features': selected,
        'n_clusters': len(clusters),
        'n_features': len(X.columns)
    }
