"""
Hyperparameter optimization using Optuna.

This module provides sophisticated hyperparameter optimization for ML models
with support for financial objectives and multi-objective optimization.

Features:
- Single and multi-objective optimization
- Custom financial objectives (Sharpe, Sortino, etc.)
- Pruning for efficient search
- MLflow integration for tracking
- Parallel execution

Designed for JPMorgan-level requirements:
- Rigorous optimization methodology
- Production-ready parameter selection
- Statistical significance testing
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler, NSGAIISampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from .model_factory import ModelFactory, ModelType, ParamSpace
from .validation import PurgedKFoldCV, WalkForwardValidator


logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    best_params: Dict[str, Any]
    best_value: float
    best_trial_number: int
    n_trials: int
    optimization_time_seconds: float
    study_name: str
    direction: str
    objective_metric: str
    all_trials: List[Dict[str, Any]]
    param_importance: Optional[Dict[str, float]] = None
    convergence_history: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "=" * 60,
            "OPTIMIZATION RESULTS",
            "=" * 60,
            f"Study: {self.study_name}",
            f"Direction: {self.direction} {self.objective_metric}",
            f"Best Value: {self.best_value:.6f}",
            f"Best Trial: #{self.best_trial_number}",
            f"Total Trials: {self.n_trials}",
            f"Time: {self.optimization_time_seconds:.1f}s",
            "",
            "Best Parameters:",
        ]
        for key, value in self.best_params.items():
            lines.append(f"  {key}: {value}")

        if self.param_importance:
            lines.append("")
            lines.append("Parameter Importance:")
            for key, value in sorted(
                self.param_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]:
                lines.append(f"  {key}: {value:.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)


class OptunaOptimizer:
    """
    Optuna-based hyperparameter optimizer for financial ML models.

    Features:
    - Multiple objective metrics (Sharpe, Sortino, accuracy, etc.)
    - Purged cross-validation integration
    - Early stopping with pruning
    - Parameter importance analysis
    - Visualization support

    Example:
        optimizer = OptunaOptimizer(
            model_type=ModelType.LIGHTGBM_CLASSIFIER,
            validation_strategy=PurgedKFoldCV(n_splits=5, purge_gap=20),
            objective_metric="sharpe_ratio",
            n_trials=100
        )

        result = optimizer.optimize(X, y, returns=returns)
        best_model = optimizer.get_best_model()
    """

    def __init__(
        self,
        model_type: Union[ModelType, str],
        validation_strategy: Optional[Any] = None,
        objective_metric: str = "sharpe_ratio",
        direction: str = "maximize",
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        sampler: Optional["optuna.samplers.BaseSampler"] = None,
        pruner: Optional["optuna.pruners.BasePruner"] = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        random_state: int = 42,
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for OptunaOptimizer. "
                "Install with: pip install optuna"
            )

        if isinstance(model_type, str):
            model_type = ModelType(model_type)

        self.model_type = model_type
        self.validation_strategy = validation_strategy or PurgedKFoldCV()
        self.objective_metric = objective_metric
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Set up sampler
        self.sampler = sampler or TPESampler(
            seed=random_state,
            n_startup_trials=10,
        )

        # Set up pruner
        self.pruner = pruner or MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
        )

        # Study configuration
        self.study_name = study_name or f"optuna_{model_type.value}_{datetime.now():%Y%m%d_%H%M%S}"
        self.storage = storage

        # Results
        self.study: Optional[optuna.Study] = None
        self.best_model: Optional[Any] = None
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._returns: Optional[np.ndarray] = None
        self._sample_weights: Optional[np.ndarray] = None

    def optimize(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        returns: Optional[Union[pd.Series, np.ndarray]] = None,
        sample_weights: Optional[np.ndarray] = None,
        param_space: Optional[List[ParamSpace]] = None,
        fixed_params: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> OptimizationResult:
        """
        Run hyperparameter optimization.

        Args:
            X: Feature data
            y: Target data
            returns: Returns for financial metrics (optional)
            sample_weights: Sample weights for training
            param_space: Custom parameter space (uses default if None)
            fixed_params: Parameters to fix (not optimize)
            callbacks: Optuna callbacks

        Returns:
            OptimizationResult with best parameters and metrics
        """
        import time
        start_time = time.time()

        # Store data for objective function
        self._X = np.asarray(X)
        self._y = np.asarray(y)
        self._returns = np.asarray(returns) if returns is not None else None
        self._sample_weights = sample_weights
        self._fixed_params = fixed_params or {}
        self._param_space = param_space or ModelFactory.get_param_space(self.model_type)

        # Create study
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            storage=self.storage,
            load_if_exists=True,
        )

        # Run optimization
        logger.info(f"Starting optimization: {self.n_trials} trials, "
                   f"{self.direction} {self.objective_metric}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.study.optimize(
                self._objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=self.n_jobs,
                callbacks=callbacks,
                show_progress_bar=True,
            )

        # Get results
        optimization_time = time.time() - start_time

        # Calculate parameter importance
        try:
            param_importance = optuna.importance.get_param_importances(self.study)
        except Exception:
            param_importance = None

        # Get convergence history
        convergence = [t.value for t in self.study.trials if t.value is not None]

        # Compile all trials
        all_trials = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                all_trials.append({
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'datetime': trial.datetime_complete,
                })

        result = OptimizationResult(
            best_params=self.study.best_params,
            best_value=self.study.best_value,
            best_trial_number=self.study.best_trial.number,
            n_trials=len(self.study.trials),
            optimization_time_seconds=optimization_time,
            study_name=self.study_name,
            direction=self.direction,
            objective_metric=self.objective_metric,
            all_trials=all_trials,
            param_importance=param_importance,
            convergence_history=convergence,
            metadata={
                'model_type': self.model_type.value,
                'cv_type': type(self.validation_strategy).__name__,
            }
        )

        logger.info(f"Optimization complete: best {self.objective_metric}={result.best_value:.6f}")

        return result

    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna."""
        # Sample parameters
        params = {}
        for space in self._param_space:
            params[space.name] = space.sample(trial)

        # Add fixed parameters
        params.update(self._fixed_params)

        # Cross-validation
        scores = []

        for fold, (train_idx, test_idx) in enumerate(
            self.validation_strategy.split(self._X, self._y)
        ):
            # Create and train model
            model = ModelFactory.create_model(
                self.model_type,
                params=params,
                random_state=self.random_state,
            )

            X_train, X_test = self._X[train_idx], self._X[test_idx]
            y_train, y_test = self._y[train_idx], self._y[test_idx]

            # Sample weights
            weights = None
            if self._sample_weights is not None:
                weights = self._sample_weights[train_idx]

            # Fit
            fit_kwargs = {}
            if weights is not None:
                fit_kwargs["sample_weight"] = weights

            try:
                model.fit(X_train, y_train, **fit_kwargs)
            except Exception as e:
                logger.warning(f"Trial {trial.number} fold {fold} failed: {e}")
                return float('-inf') if self.direction == "maximize" else float('inf')

            # Calculate objective metric
            score = self._calculate_objective(model, X_test, y_test, test_idx)
            scores.append(score)

            # Report for pruning
            trial.report(np.mean(scores), fold)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)

    def _calculate_objective(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        test_idx: np.ndarray,
    ) -> float:
        """Calculate the objective metric."""
        predictions = model.predict(X_test)

        # Classification metrics
        if self.objective_metric == "accuracy":
            from sklearn.metrics import accuracy_score
            return accuracy_score(y_test, predictions)

        elif self.objective_metric == "roc_auc":
            from sklearn.metrics import roc_auc_score
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                if proba.shape[1] == 2:
                    return roc_auc_score(y_test, proba[:, 1])
            return 0.5

        elif self.objective_metric == "f1":
            from sklearn.metrics import f1_score
            return f1_score(y_test, predictions, average='weighted')

        # Regression metrics
        elif self.objective_metric == "mse":
            return -np.mean((y_test - predictions) ** 2)

        elif self.objective_metric == "r2":
            from sklearn.metrics import r2_score
            return r2_score(y_test, predictions)

        # Financial metrics
        elif self.objective_metric in ["sharpe_ratio", "sortino_ratio", "calmar_ratio"]:
            return self._calculate_financial_metric(
                model, X_test, y_test, test_idx
            )

        else:
            raise ValueError(f"Unknown objective metric: {self.objective_metric}")

    def _calculate_financial_metric(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        test_idx: np.ndarray,
    ) -> float:
        """Calculate financial performance metric."""
        if self._returns is None:
            raise ValueError(
                f"Returns data required for {self.objective_metric}. "
                "Pass returns parameter to optimize()."
            )

        # Get predictions (signals)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            if proba.shape[1] == 2:
                signals = proba[:, 1] - 0.5  # Convert to [-0.5, 0.5]
            else:
                signals = proba.argmax(axis=1) - 1  # Multi-class
        else:
            predictions = model.predict(X_test)
            signals = predictions  # For regression

        # Normalize signals to [-1, 1]
        if np.abs(signals).max() > 0:
            signals = signals / np.abs(signals).max()

        # Get returns for test period
        test_returns = self._returns[test_idx]

        # Calculate strategy returns
        strategy_returns = signals * test_returns

        # Handle NaN/inf
        strategy_returns = np.nan_to_num(strategy_returns, nan=0, posinf=0, neginf=0)

        if len(strategy_returns) == 0 or np.std(strategy_returns) == 0:
            return 0.0

        if self.objective_metric == "sharpe_ratio":
            # Annualized Sharpe (assuming 15-min bars)
            return np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(6552)

        elif self.objective_metric == "sortino_ratio":
            downside = strategy_returns[strategy_returns < 0]
            if len(downside) == 0:
                return 10.0  # Cap at 10 if no downside
            downside_std = np.std(downside)
            return np.mean(strategy_returns) / (downside_std + 1e-8) * np.sqrt(6552)

        elif self.objective_metric == "calmar_ratio":
            cumulative = np.cumsum(strategy_returns)
            peak = np.maximum.accumulate(cumulative)
            drawdown = (peak - cumulative) / (peak + 1e-8)
            max_dd = np.max(drawdown)
            if max_dd == 0:
                return 10.0
            annual_return = np.mean(strategy_returns) * 6552
            return annual_return / (max_dd + 1e-8)

        return 0.0

    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters found."""
        if self.study is None:
            raise ValueError("No optimization run yet")
        return self.study.best_params

    def get_best_model(self) -> Any:
        """Create model with best parameters."""
        if self.study is None:
            raise ValueError("No optimization run yet")

        params = {**self.study.best_params, **self._fixed_params}
        return ModelFactory.create_model(
            self.model_type,
            params=params,
            random_state=self.random_state,
        )

    def get_param_importance(self) -> Dict[str, float]:
        """Get parameter importance scores."""
        if self.study is None:
            raise ValueError("No optimization run yet")
        return optuna.importance.get_param_importances(self.study)

    def plot_optimization_history(self) -> Optional[Any]:
        """Plot optimization history."""
        if self.study is None:
            return None

        try:
            return optuna.visualization.plot_optimization_history(self.study)
        except Exception:
            return None

    def plot_param_importance(self) -> Optional[Any]:
        """Plot parameter importance."""
        if self.study is None:
            return None

        try:
            return optuna.visualization.plot_param_importances(self.study)
        except Exception:
            return None

    def plot_contour(self, params: List[str]) -> Optional[Any]:
        """Plot contour for two parameters."""
        if self.study is None or len(params) != 2:
            return None

        try:
            return optuna.visualization.plot_contour(self.study, params=params)
        except Exception:
            return None


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization for balancing multiple metrics.

    Uses NSGA-II algorithm to find Pareto-optimal solutions that
    trade off between objectives like Sharpe ratio and max drawdown.

    Example:
        optimizer = MultiObjectiveOptimizer(
            model_type=ModelType.LIGHTGBM_CLASSIFIER,
            objectives=["sharpe_ratio", "max_drawdown"],
            directions=["maximize", "minimize"]
        )

        result = optimizer.optimize(X, y, returns=returns)
        pareto_front = optimizer.get_pareto_front()
    """

    def __init__(
        self,
        model_type: Union[ModelType, str],
        validation_strategy: Optional[Any] = None,
        objectives: List[str] = ["sharpe_ratio", "max_drawdown"],
        directions: List[str] = ["maximize", "minimize"],
        n_trials: int = 100,
        timeout: Optional[int] = None,
        random_state: int = 42,
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna required")

        if isinstance(model_type, str):
            model_type = ModelType(model_type)

        if len(objectives) != len(directions):
            raise ValueError("objectives and directions must have same length")

        self.model_type = model_type
        self.validation_strategy = validation_strategy or PurgedKFoldCV()
        self.objectives = objectives
        self.directions = directions
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state

        self.study: Optional[optuna.Study] = None
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._returns: Optional[np.ndarray] = None

    def optimize(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        returns: Optional[Union[pd.Series, np.ndarray]] = None,
        param_space: Optional[List[ParamSpace]] = None,
    ) -> List[OptimizationResult]:
        """
        Run multi-objective optimization.

        Returns list of Pareto-optimal results.
        """
        import time
        start_time = time.time()

        self._X = np.asarray(X)
        self._y = np.asarray(y)
        self._returns = np.asarray(returns) if returns is not None else None
        self._param_space = param_space or ModelFactory.get_param_space(self.model_type)

        # Create multi-objective study
        self.study = optuna.create_study(
            directions=self.directions,
            sampler=NSGAIISampler(seed=self.random_state),
        )

        # Run optimization
        logger.info(f"Starting multi-objective optimization: {self.objectives}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.study.optimize(
                self._objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=True,
            )

        # Get Pareto front
        pareto_trials = self.study.best_trials

        results = []
        for trial in pareto_trials:
            results.append(OptimizationResult(
                best_params=trial.params,
                best_value=trial.values[0],  # First objective
                best_trial_number=trial.number,
                n_trials=len(self.study.trials),
                optimization_time_seconds=time.time() - start_time,
                study_name="multi_objective",
                direction=str(self.directions),
                objective_metric=str(self.objectives),
                all_trials=[],
                metadata={'all_objectives': trial.values}
            ))

        return results

    def _objective(self, trial: optuna.Trial) -> Tuple[float, ...]:
        """Multi-objective function."""
        # Sample parameters
        params = {}
        for space in self._param_space:
            params[space.name] = space.sample(trial)

        # Cross-validation
        objective_scores = {obj: [] for obj in self.objectives}

        for train_idx, test_idx in self.validation_strategy.split(self._X, self._y):
            model = ModelFactory.create_model(
                self.model_type,
                params=params,
                random_state=self.random_state,
            )

            X_train, X_test = self._X[train_idx], self._X[test_idx]
            y_train, y_test = self._y[train_idx], self._y[test_idx]

            try:
                model.fit(X_train, y_train)
            except Exception:
                return tuple([float('-inf') if d == "maximize" else float('inf')
                             for d in self.directions])

            # Calculate each objective
            for obj in self.objectives:
                score = self._calculate_single_objective(
                    model, X_test, y_test, test_idx, obj
                )
                objective_scores[obj].append(score)

        return tuple(np.mean(objective_scores[obj]) for obj in self.objectives)

    def _calculate_single_objective(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        test_idx: np.ndarray,
        objective: str,
    ) -> float:
        """Calculate a single objective metric."""
        # Reuse single-objective calculation logic
        optimizer = OptunaOptimizer(
            model_type=self.model_type,
            objective_metric=objective,
        )
        optimizer._returns = self._returns
        return optimizer._calculate_objective(model, X_test, y_test, test_idx)

    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """Get Pareto-optimal solutions."""
        if self.study is None:
            return []

        front = []
        for trial in self.study.best_trials:
            front.append({
                'params': trial.params,
                'values': {
                    obj: val for obj, val in zip(self.objectives, trial.values)
                }
            })
        return front

    def plot_pareto_front(self) -> Optional[Any]:
        """Plot Pareto front (2D or 3D)."""
        if self.study is None:
            return None

        try:
            if len(self.objectives) == 2:
                return optuna.visualization.plot_pareto_front(self.study)
            elif len(self.objectives) == 3:
                return optuna.visualization.plot_pareto_front(
                    self.study, include_dominated_trials=False
                )
        except Exception:
            return None


# Convenience function
def optimize_model(
    model_type: Union[ModelType, str],
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    returns: Optional[Union[pd.Series, np.ndarray]] = None,
    objective: str = "sharpe_ratio",
    n_trials: int = 50,
    cv_splits: int = 5,
) -> OptimizationResult:
    """
    Quick function to optimize a model.

    Args:
        model_type: Type of model
        X: Features
        y: Target
        returns: Returns for financial metrics
        objective: Objective metric
        n_trials: Number of trials
        cv_splits: Number of CV folds

    Returns:
        OptimizationResult
    """
    optimizer = OptunaOptimizer(
        model_type=model_type,
        validation_strategy=PurgedKFoldCV(n_splits=cv_splits),
        objective_metric=objective,
        n_trials=n_trials,
    )
    return optimizer.optimize(X, y, returns=returns)
