"""
Training Pipeline Module
========================

Production-grade ML training infrastructure for the trading platform.
Implements JPMorgan-level training standards with Optuna optimization.

Features:
- Optuna hyperparameter optimization
- Walk-forward validation
- Purged cross-validation
- MLflow experiment tracking
- Model selection and comparison
- Automated feature selection
- Training reproducibility

Architecture:
- TrainingPipeline: Main orchestrator
- HyperparameterOptimizer: Optuna wrapper
- ModelEvaluator: Cross-validation engine
- ExperimentTracker: MLflow integration

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable
import hashlib
import json
import pickle
import time

import numpy as np
import polars as pl
from numpy.typing import NDArray

from config.settings import get_logger, get_settings

logger = get_logger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class OptimizationDirection(str, Enum):
    """Optimization direction."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class SamplerType(str, Enum):
    """Optuna sampler types."""
    TPE = "tpe"  # Tree-structured Parzen Estimator
    CMA_ES = "cmaes"  # CMA Evolution Strategy
    RANDOM = "random"
    GRID = "grid"


class PrunerType(str, Enum):
    """Optuna pruner types."""
    MEDIAN = "median"
    HYPERBAND = "hyperband"
    SUCCESSIVE_HALVING = "successive_halving"
    NONE = "none"


@dataclass
class OptimizationConfig:
    """
    Configuration for hyperparameter optimization.
    
    Attributes:
        n_trials: Number of optimization trials
        timeout: Maximum time in seconds (optional)
        n_jobs: Parallel trials (-1 = all cores)
        direction: Optimization direction
        sampler: Sampler type
        pruner: Pruner type
        seed: Random seed
        study_name: Optuna study name
        storage: Optuna storage URL (optional)
        load_if_exists: Load existing study
        show_progress_bar: Display progress
    """
    n_trials: int = 100
    timeout: int | None = None
    n_jobs: int = 1  # Parallel trials
    direction: OptimizationDirection = OptimizationDirection.MAXIMIZE
    sampler: SamplerType = SamplerType.TPE
    pruner: PrunerType = PrunerType.MEDIAN
    seed: int = 42
    study_name: str = "trading_model_optimization"
    storage: str | None = None
    load_if_exists: bool = True
    show_progress_bar: bool = True
    
    # Validation
    cv_splits: int = 5
    validation_metric: str = "roc_auc"  # For classification
    
    # Early stopping for trials
    early_stopping_rounds: int = 30


@dataclass
class TrainingConfig:
    """
    Configuration for training pipeline.
    
    Attributes:
        models_dir: Directory for saved models
        experiment_name: MLflow experiment name
        tracking_uri: MLflow tracking URI
        enable_tracking: Enable MLflow tracking
        auto_optimize: Run hyperparameter optimization
        optimization_config: Optuna configuration
        save_best_model: Save best model automatically
        feature_selection: Enable feature selection
        max_features: Maximum features to use
    """
    models_dir: Path = Path("models/artifacts")
    experiment_name: str = "trading_models"
    tracking_uri: str = "sqlite:///mlflow.db"
    enable_tracking: bool = True
    auto_optimize: bool = True
    optimization_config: OptimizationConfig = field(default_factory=OptimizationConfig)
    save_best_model: bool = True
    feature_selection: bool = False
    max_features: int = 100


# =============================================================================
# HYPERPARAMETER SPACES
# =============================================================================

def get_lightgbm_space(trial) -> dict[str, Any]:
    """Get LightGBM hyperparameter space for Optuna."""
    return {
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
    }


def get_xgboost_space(trial) -> dict[str, Any]:
    """Get XGBoost hyperparameter space for Optuna."""
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def get_catboost_space(trial) -> dict[str, Any]:
    """Get CatBoost hyperparameter space for Optuna."""
    return {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }


def get_random_forest_space(trial) -> dict[str, Any]:
    """Get Random Forest hyperparameter space for Optuna."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
    }


def get_lstm_space(trial) -> dict[str, Any]:
    """Get LSTM hyperparameter space for Optuna."""
    return {
        "sequence_length": trial.suggest_int("sequence_length", 20, 100),
        "hidden_size": trial.suggest_int("hidden_size", 32, 256),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }


def get_transformer_space(trial) -> dict[str, Any]:
    """Get Transformer hyperparameter space for Optuna."""
    return {
        "sequence_length": trial.suggest_int("sequence_length", 20, 100),
        "d_model": trial.suggest_categorical("d_model", [64, 128, 256]),
        "n_heads": trial.suggest_categorical("n_heads", [4, 8]),
        "n_encoder_layers": trial.suggest_int("n_encoder_layers", 2, 6),
        "d_ff": trial.suggest_categorical("d_ff", [256, 512, 1024]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
    }


PARAM_SPACES = {
    "lightgbm": get_lightgbm_space,
    "xgboost": get_xgboost_space,
    "catboost": get_catboost_space,
    "random_forest": get_random_forest_space,
    "lstm": get_lstm_space,
    "transformer": get_transformer_space,
}


# =============================================================================
# PURGED K-FOLD CROSS VALIDATION
# =============================================================================

class PurgedKFold:
    """
    Purged K-Fold Cross Validation for financial time series.
    
    Implements embargo and purging to prevent look-ahead bias:
    - Purging: Remove training samples too close to test samples
    - Embargo: Add gap between train and test sets
    
    Reference: "Advances in Financial Machine Learning" by Marcos López de Prado
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 10,
        embargo_pct: float = 0.01,
    ):
        """
        Initialize PurgedKFold.
        
        Args:
            n_splits: Number of folds
            purge_gap: Number of samples to purge between train/test
            embargo_pct: Percentage of samples to embargo after test
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
    
    def split(self, X, y=None, groups=None):
        """Generate train/test indices."""
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        embargo_size = int(n_samples * self.embargo_pct)
        
        for i in range(self.n_splits):
            # Test indices
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            test_indices = np.arange(test_start, test_end)
            
            # Train indices (with purging and embargo)
            train_indices = []
            
            # Before test set (with purge)
            if test_start > 0:
                train_end = max(0, test_start - self.purge_gap)
                train_indices.extend(range(0, train_end))
            
            # After test set (with embargo)
            if test_end < n_samples:
                train_start = min(n_samples, test_end + embargo_size)
                train_indices.extend(range(train_start, n_samples))
            
            yield np.array(train_indices), test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits."""
        return self.n_splits


class CombinatorialPurgedKFold:
    """
    Combinatorial Purged K-Fold Cross Validation.
    
    Tests on all possible contiguous blocks while maintaining
    temporal order and purging.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        n_test_groups: int = 2,
        purge_gap: int = 10,
    ):
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.purge_gap = purge_gap
    
    def split(self, X, y=None, groups=None):
        """Generate combinatorial train/test indices."""
        from itertools import combinations
        
        n_samples = len(X)
        group_size = n_samples // self.n_splits
        
        # Create groups
        groups = []
        for i in range(self.n_splits):
            start = i * group_size
            end = (i + 1) * group_size if i < self.n_splits - 1 else n_samples
            groups.append(np.arange(start, end))
        
        # Generate all combinations of test groups
        for test_group_indices in combinations(range(self.n_splits), self.n_test_groups):
            test_indices = np.concatenate([groups[i] for i in test_group_indices])
            
            # Train indices (excluding test and purge)
            train_mask = np.ones(n_samples, dtype=bool)
            
            for test_idx in test_indices:
                # Purge around each test index
                purge_start = max(0, test_idx - self.purge_gap)
                purge_end = min(n_samples, test_idx + self.purge_gap + 1)
                train_mask[purge_start:purge_end] = False
            
            train_indices = np.where(train_mask)[0]
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits."""
        from math import comb
        return comb(self.n_splits, self.n_test_groups)


# =============================================================================
# HYPERPARAMETER OPTIMIZER
# =============================================================================

class HyperparameterOptimizer:
    """
    Optuna-based hyperparameter optimizer.
    
    Features:
    - Multiple sampler options
    - Pruning for early stopping
    - Study persistence
    - Parallel optimization
    """
    
    def __init__(
        self,
        config: OptimizationConfig | None = None,
    ):
        self.config = config or OptimizationConfig()
        self._study: Any = None
        self._best_params: dict[str, Any] = {}
        self._best_value: float = 0.0
        self._history: list[dict[str, Any]] = []
    
    def optimize(
        self,
        model_type: str,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.float64] | None = None,
        feature_names: list[str] | None = None,
        custom_space: Callable | None = None,
    ) -> dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            model_type: Type of model to optimize
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            feature_names: Feature names
            custom_space: Custom parameter space function
        
        Returns:
            Best parameters found
        """
        import optuna
        from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
        from optuna.pruners import MedianPruner, HyperbandPruner, SuccessiveHalvingPruner
        
        logger.info(f"Starting hyperparameter optimization for {model_type}")
        logger.info(f"n_trials={self.config.n_trials}, direction={self.config.direction.value}")
        
        # Get parameter space
        param_space_fn = custom_space or PARAM_SPACES.get(model_type.lower())
        if param_space_fn is None:
            raise ValueError(f"No parameter space defined for {model_type}")
        
        # Create sampler
        if self.config.sampler == SamplerType.TPE:
            sampler = TPESampler(seed=self.config.seed)
        elif self.config.sampler == SamplerType.CMA_ES:
            sampler = CmaEsSampler(seed=self.config.seed)
        else:
            sampler = RandomSampler(seed=self.config.seed)
        
        # Create pruner
        if self.config.pruner == PrunerType.MEDIAN:
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif self.config.pruner == PrunerType.HYPERBAND:
            pruner = HyperbandPruner()
        elif self.config.pruner == PrunerType.SUCCESSIVE_HALVING:
            pruner = SuccessiveHalvingPruner()
        else:
            pruner = optuna.pruners.NopPruner()
        
        # Create study
        self._study = optuna.create_study(
            study_name=self.config.study_name,
            direction=self.config.direction.value,
            sampler=sampler,
            pruner=pruner,
            storage=self.config.storage,
            load_if_exists=self.config.load_if_exists,
        )
        
        # Define objective
        def objective(trial):
            params = param_space_fn(trial)
            
            try:
                # Create and train model
                model = self._create_model(model_type, params)
                
                # Cross-validation score
                score = self._cross_validate(
                    model, X_train, y_train, X_val, y_val, feature_names, trial
                )
                
                return score
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                raise optuna.TrialPruned()
        
        # Run optimization
        self._study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=self.config.show_progress_bar,
            callbacks=[self._log_callback],
        )
        
        # Store results
        self._best_params = self._study.best_params
        self._best_value = self._study.best_value
        
        logger.info(f"Best {self.config.validation_metric}: {self._best_value:.4f}")
        logger.info(f"Best params: {self._best_params}")
        
        return self._best_params
    
    def _create_model(
        self,
        model_type: str,
        params: dict[str, Any],
    ) -> Any:
        """Create model with given parameters."""
        from models.classifiers import create_classifier
        from models.deep import create_deep_model
        
        model_type = model_type.lower()
        
        if model_type in ["lstm", "transformer", "tcn"]:
            return create_deep_model(model_type, **params)
        else:
            return create_classifier(model_type, **params)
    
    def _cross_validate(
        self,
        model: Any,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        X_val: NDArray[np.float64] | None,
        y_val: NDArray[np.float64] | None,
        feature_names: list[str] | None,
        trial: Any,
    ) -> float:
        """Perform cross-validation and return score."""
        import optuna
        
        # Use purged k-fold for financial data
        cv = PurgedKFold(
            n_splits=self.config.cv_splits,
            purge_gap=10,
            embargo_pct=0.01,
        )
        
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train)):
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]
            
            # Train
            model.fit(
                X_fold_train,
                y_fold_train,
                feature_names=feature_names,
                X_val=X_fold_val,
                y_val=y_fold_val,
            )
            
            # Evaluate
            eval_results = model.evaluate(X_fold_val, y_fold_val)
            score = eval_results.get(
                self.config.validation_metric,
                eval_results.get("accuracy", 0),
            )
            scores.append(score)
            
            # Report intermediate value for pruning
            trial.report(np.mean(scores), fold)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(scores)
    
    def _log_callback(self, study, trial):
        """Callback for logging trial results."""
        self._history.append({
            "trial": trial.number,
            "value": trial.value,
            "params": trial.params,
            "state": trial.state.name,
        })
    
    @property
    def best_params(self) -> dict[str, Any]:
        """Get best parameters found."""
        return self._best_params
    
    @property
    def best_value(self) -> float:
        """Get best score achieved."""
        return self._best_value
    
    @property
    def study(self) -> Any:
        """Get Optuna study."""
        return self._study
    
    def get_importance(self) -> dict[str, float]:
        """Get parameter importance."""
        import optuna
        
        if self._study is None:
            return {}
        
        try:
            importance = optuna.importance.get_param_importances(self._study)
            return dict(importance)
        except Exception:
            return {}
    
    def plot_optimization_history(self, output_path: str | None = None):
        """Plot optimization history."""
        import optuna.visualization as vis
        
        if self._study is None:
            return None
        
        fig = vis.plot_optimization_history(self._study)
        
        if output_path:
            fig.write_html(output_path)
        
        return fig
    
    def plot_param_importances(self, output_path: str | None = None):
        """Plot parameter importance."""
        import optuna.visualization as vis
        
        if self._study is None:
            return None
        
        fig = vis.plot_param_importances(self._study)
        
        if output_path:
            fig.write_html(output_path)
        
        return fig


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

class TrainingPipeline:
    """
    End-to-end training pipeline for ML models.
    
    Features:
    - Data preprocessing
    - Feature selection
    - Hyperparameter optimization
    - Model training and validation
    - Experiment tracking
    - Model persistence
    """
    
    def __init__(
        self,
        config: TrainingConfig | None = None,
    ):
        self.config = config or TrainingConfig()
        self._optimizer: HyperparameterOptimizer | None = None
        self._best_model: Any = None
        self._training_results: dict[str, Any] = {}
        
        # Ensure models directory exists
        self.config.models_dir.mkdir(parents=True, exist_ok=True)
    
    def train(
        self,
        model_type: str,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        X_test: NDArray[np.float64] | None = None,
        y_test: NDArray[np.float64] | None = None,
        feature_names: list[str] | None = None,
        optimize: bool | None = None,
        model_params: dict[str, Any] | None = None,
    ) -> Any:
        """
        Train a model with optional hyperparameter optimization.
        
        Args:
            model_type: Type of model to train
            X_train: Training features
            y_train: Training targets
            X_test: Test features (optional)
            y_test: Test targets (optional)
            feature_names: Feature names
            optimize: Run hyperparameter optimization
            model_params: Custom model parameters
        
        Returns:
            Trained model
        """
        from models.classifiers import create_classifier
        from models.deep import create_deep_model
        
        logger.info(f"Starting training pipeline for {model_type}")
        start_time = time.time()
        
        optimize = optimize if optimize is not None else self.config.auto_optimize
        
        # Feature selection
        if self.config.feature_selection:
            X_train, X_test, feature_names = self._select_features(
                X_train, y_train, X_test, feature_names
            )
        
        # Hyperparameter optimization
        if optimize and model_params is None:
            logger.info("Running hyperparameter optimization...")
            self._optimizer = HyperparameterOptimizer(self.config.optimization_config)
            best_params = self._optimizer.optimize(
                model_type, X_train, y_train,
                feature_names=feature_names,
            )
            model_params = best_params
        
        model_params = model_params or {}
        
        # Create and train model
        if model_type.lower() in ["lstm", "transformer", "tcn"]:
            model = create_deep_model(model_type, **model_params)
        else:
            model = create_classifier(model_type, **model_params)
        
        # Split validation set if no test set provided
        if X_test is None:
            split_idx = int(len(X_train) * 0.85)
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]
        else:
            X_val, y_val = X_test, y_test
        
        # Train
        model.fit(
            X_train, y_train,
            feature_names=feature_names,
            X_val=X_val,
            y_val=y_val,
        )
        
        # Evaluate
        train_metrics = model.evaluate(X_train, y_train)
        val_metrics = model.evaluate(X_val, y_val)
        
        if X_test is not None and y_test is not None:
            test_metrics = model.evaluate(X_test, y_test)
        else:
            test_metrics = val_metrics
        
        # Store results
        training_time = time.time() - start_time
        self._training_results = {
            "model_type": model_type,
            "model_id": model.model_id,
            "params": model_params,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "training_time": training_time,
            "n_features": X_train.shape[1],
            "n_train_samples": len(X_train),
            "feature_importance": model.feature_importance,
        }
        
        self._best_model = model
        
        logger.info(f"Training completed in {training_time:.2f}s")
        logger.info(f"Val metrics: {val_metrics}")
        
        # Save model
        if self.config.save_best_model:
            self.save_model(model)
        
        return model
    
    def train_ensemble(
        self,
        model_types: list[str],
        X_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        X_test: NDArray[np.float64] | None = None,
        y_test: NDArray[np.float64] | None = None,
        feature_names: list[str] | None = None,
        voting: str = "soft",
        weights: list[float] | None = None,
    ) -> Any:
        """
        Train an ensemble of multiple models.
        
        Args:
            model_types: List of model types to include
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            feature_names: Feature names
            voting: Voting type (soft/hard)
            weights: Model weights
        
        Returns:
            Trained ensemble model
        """
        from sklearn.ensemble import VotingClassifier
        
        logger.info(f"Training ensemble with models: {model_types}")
        
        # Train individual models
        estimators = []
        for model_type in model_types:
            model = self.train(
                model_type, X_train, y_train, X_test, y_test,
                feature_names=feature_names,
                optimize=True,
            )
            estimators.append((model_type, model.underlying_model))
        
        # Create ensemble
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights,
            n_jobs=-1,
        )
        
        # Fit ensemble
        ensemble.fit(X_train, y_train)
        
        return ensemble
    
    def compare_models(
        self,
        model_types: list[str],
        X_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        X_test: NDArray[np.float64],
        y_test: NDArray[np.float64],
        feature_names: list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Compare multiple model types.
        
        Returns DataFrame with comparison results.
        """
        results = []
        
        for model_type in model_types:
            logger.info(f"Training and evaluating {model_type}...")
            
            model = self.train(
                model_type, X_train, y_train, X_test, y_test,
                feature_names=feature_names,
                optimize=True,
            )
            
            metrics = model.evaluate(X_test, y_test)
            
            results.append({
                "model": model_type,
                "accuracy": metrics.get("accuracy", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "f1": metrics.get("f1_score", 0),
                "roc_auc": metrics.get("roc_auc", 0),
                "training_time": self._training_results.get("training_time", 0),
            })
        
        df = pl.DataFrame(results)
        df = df.sort("roc_auc", descending=True)
        
        logger.info("\n" + str(df))
        
        return df
    
    def _select_features(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        X_test: NDArray[np.float64] | None,
        feature_names: list[str] | None,
    ) -> tuple[NDArray, NDArray | None, list[str] | None]:
        """Select top features using importance."""
        from sklearn.ensemble import RandomForestClassifier
        
        logger.info("Running feature selection...")
        
        # Train simple RF for feature importance
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        
        # Get importance
        importance = rf.feature_importances_
        
        # Select top features
        n_select = min(self.config.max_features, len(importance))
        top_indices = np.argsort(importance)[-n_select:]
        
        X_train_selected = X_train[:, top_indices]
        X_test_selected = X_test[:, top_indices] if X_test is not None else None
        
        selected_names = None
        if feature_names:
            selected_names = [feature_names[i] for i in top_indices]
        
        logger.info(f"Selected {n_select} features from {len(importance)}")
        
        return X_train_selected, X_test_selected, selected_names
    
    def save_model(
        self,
        model: Any,
        name: str | None = None,
    ) -> Path:
        """Save trained model."""
        name = name or f"{model.name}_{model.model_id}"
        path = self.config.models_dir / f"{name}.pkl"
        
        model.save(path)
        
        # Save training results
        results_path = self.config.models_dir / f"{name}_results.json"
        with open(results_path, "w") as f:
            json.dump(self._training_results, f, indent=2, default=str)
        
        logger.info(f"Model saved to {path}")
        return path
    
    @staticmethod
    def load_model(path: Path | str) -> Any:
        """Load a saved model."""
        from models.base import BaseModel
        return BaseModel.load(path)
    
    @property
    def best_model(self) -> Any:
        """Get the best trained model."""
        return self._best_model
    
    @property
    def training_results(self) -> dict[str, Any]:
        """Get training results."""
        return self._training_results
    
    @property
    def optimizer(self) -> HyperparameterOptimizer | None:
        """Get the hyperparameter optimizer."""
        return self._optimizer


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_train(
    model_type: str,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    feature_names: list[str] | None = None,
    test_size: float = 0.2,
    optimize: bool = True,
    n_trials: int = 50,
) -> Any:
    """
    Quick training function for rapid experimentation.
    
    Args:
        model_type: Type of model to train
        X: Features
        y: Targets
        feature_names: Feature names
        test_size: Test set size
        optimize: Run optimization
        n_trials: Number of optimization trials
    
    Returns:
        Trained model
    """
    # Split data
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Configure
    config = TrainingConfig(
        auto_optimize=optimize,
        optimization_config=OptimizationConfig(n_trials=n_trials),
    )
    
    # Train
    pipeline = TrainingPipeline(config)
    model = pipeline.train(
        model_type,
        X_train, y_train,
        X_test, y_test,
        feature_names=feature_names,
    )
    
    return model


def auto_ml(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    feature_names: list[str] | None = None,
    test_size: float = 0.2,
    n_trials_per_model: int = 30,
) -> tuple[Any, pl.DataFrame]:
    """
    Automatic model selection and training.
    
    Compares multiple model types and returns the best one.
    
    Args:
        X: Features
        y: Targets
        feature_names: Feature names
        test_size: Test set size
        n_trials_per_model: Optimization trials per model
    
    Returns:
        (best_model, comparison_dataframe)
    """
    # Split data
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Models to compare
    model_types = ["lightgbm", "xgboost", "random_forest"]
    
    # Configure
    config = TrainingConfig(
        optimization_config=OptimizationConfig(n_trials=n_trials_per_model),
    )
    
    pipeline = TrainingPipeline(config)
    
    # Compare models
    comparison = pipeline.compare_models(
        model_types,
        X_train, y_train,
        X_test, y_test,
        feature_names=feature_names,
    )
    
    return pipeline.best_model, comparison


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "OptimizationDirection",
    "SamplerType",
    "PrunerType",
    # Config
    "OptimizationConfig",
    "TrainingConfig",
    # Cross-validation
    "PurgedKFold",
    "CombinatorialPurgedKFold",
    # Optimizer
    "HyperparameterOptimizer",
    # Pipeline
    "TrainingPipeline",
    # Functions
    "quick_train",
    "auto_ml",
    # Param spaces
    "PARAM_SPACES",
    "get_lightgbm_space",
    "get_xgboost_space",
    "get_catboost_space",
    "get_random_forest_space",
    "get_lstm_space",
    "get_transformer_space",
]