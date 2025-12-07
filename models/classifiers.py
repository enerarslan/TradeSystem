"""
Classifier Models Module
========================

Production-grade classification models for the AlphaTrade trading platform.
Implements JPMorgan-level ML standards with proper validation.

Models:
- LightGBMClassifier: Gradient boosting (fastest, handles missing values)
- XGBoostClassifier: Gradient boosting (accurate, GPU support)
- CatBoostClassifier: Handles categoricals natively
- RandomForestClassifier: Bagging ensemble (robust)
- ExtraTreesClassifier: Extremely randomized trees
- StackingClassifier: Stacked ensemble (maximum accuracy)
- VotingClassifier: Voting ensemble

Features:
- AUTOMATIC binary/multiclass detection (fixes the multilabel error)
- Hyperparameter optimization ready (Optuna compatible)
- Class imbalance handling
- Early stopping with validation
- Feature importance extraction
- Calibrated probabilities
- XGBoost 2.0+ compatibility

CRITICAL FIX (v2.0.0):
- LightGBM and XGBoost now auto-detect number of classes
- objective="auto" triggers automatic detection
- num_class=None triggers automatic detection
- Prevents "multilabel-indicator targets" errors

Author: AlphaTrade Platform
Version: 2.0.0
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from config.settings import get_logger
from models.base import (
    BaseModel,
    ModelConfig,
    ModelType,
    ModelRegistry,
    ValidationMethod,
)

logger = get_logger(__name__)


# =============================================================================
# LIGHTGBM CLASSIFIER
# =============================================================================

@dataclass
class LightGBMClassifierConfig(ModelConfig):
    """
    Configuration for LightGBM Classifier.
    
    IMPORTANT: objective="auto" and num_class=None enable automatic
    detection of binary vs multiclass classification.
    """
    name: str = "LightGBMClassifier"
    model_type: ModelType = ModelType.CLASSIFIER
    
    # Core parameters - AUTO-DETECT BY DEFAULT
    objective: str = "auto"  # FIXED: "auto" = detect from data, or "binary"/"multiclass"
    num_class: int | None = None  # FIXED: None = detect from data
    boosting_type: str = "gbdt"  # gbdt, dart, goss
    
    # Tree parameters
    num_leaves: int = 31
    max_depth: int = -1  # -1 = no limit
    min_child_samples: int = 20
    min_child_weight: float = 1e-3
    
    # Learning parameters
    learning_rate: float = 0.05
    n_estimators: int = 1000
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    
    # Regularization
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    min_split_gain: float = 0.0
    
    # Class imbalance
    class_weight: str | dict | None = "balanced"
    scale_pos_weight: float = 1.0
    
    # Feature importance
    importance_type: str = "gain"  # gain, split


class LightGBMClassifier(BaseModel[LightGBMClassifierConfig]):
    """
    LightGBM Classifier for trading signal prediction.
    
    Optimal for:
    - Large datasets (millions of samples)
    - High-dimensional features (1000s)
    - Fast training and inference
    - Handling missing values natively
    
    AUTOMATIC CLASS DETECTION:
    - Binary: Uses objective="binary"
    - Multiclass: Uses objective="multiclass" with correct num_class
    
    Example:
        config = LightGBMClassifierConfig(
            learning_rate=0.03,
            num_leaves=63,
            max_depth=7,
        )
        model = LightGBMClassifier(config)
        model.fit(X_train, y_train)  # Auto-detects 2 or 3+ classes
        predictions = model.predict(X_test)
    """
    
    def _default_config(self) -> LightGBMClassifierConfig:
        """Create default configuration."""
        return LightGBMClassifierConfig()
    
    def _build_model(self) -> Any:
        """
        Build LightGBM classifier.
        
        Note: If objective="auto", actual objective is set in _fit_impl()
        after class detection.
        """
        import lightgbm as lgb
        
        params = {
            "boosting_type": self.config.boosting_type,
            "num_leaves": self.config.num_leaves,
            "max_depth": self.config.max_depth,
            "min_child_samples": self.config.min_child_samples,
            "min_child_weight": self.config.min_child_weight,
            "learning_rate": self.config.learning_rate,
            "n_estimators": self.config.n_estimators,
            "subsample": self.config.subsample,
            "subsample_freq": self.config.subsample_freq,
            "colsample_bytree": self.config.colsample_bytree,
            "reg_alpha": self.config.reg_alpha,
            "reg_lambda": self.config.reg_lambda,
            "min_split_gain": self.config.min_split_gain,
            "class_weight": self.config.class_weight,
            "importance_type": self.config.importance_type,
            "random_state": self.config.random_state,
            "n_jobs": self.config.n_jobs,
            "verbose": -1,
        }
        
        # Only set objective if not auto-detecting
        if self.config.objective != "auto":
            params["objective"] = self.config.objective
            if self.config.objective == "multiclass" and self.config.num_class:
                params["num_class"] = self.config.num_class
        
        return lgb.LGBMClassifier(**params)
    
    def _fit_impl(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.float64] | None = None,
        sample_weight: NDArray[np.float64] | None = None,
    ) -> None:
        """
        Train LightGBM classifier with automatic class detection.
        
        CRITICAL FIX: This method auto-detects binary vs multiclass
        and rebuilds the model with correct parameters.
        """
        import lightgbm as lgb
        
        # =====================================================================
        # AUTO-DETECT NUMBER OF CLASSES
        # =====================================================================
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        
        # Store for evaluation
        self._n_classes = n_classes
        self._classes = unique_classes
        
        # Rebuild model with correct objective if auto-detecting
        if self.config.objective == "auto":
            params = {
                "boosting_type": self.config.boosting_type,
                "num_leaves": self.config.num_leaves,
                "max_depth": self.config.max_depth,
                "min_child_samples": self.config.min_child_samples,
                "min_child_weight": self.config.min_child_weight,
                "learning_rate": self.config.learning_rate,
                "n_estimators": self.config.n_estimators,
                "subsample": self.config.subsample,
                "subsample_freq": self.config.subsample_freq,
                "colsample_bytree": self.config.colsample_bytree,
                "reg_alpha": self.config.reg_alpha,
                "reg_lambda": self.config.reg_lambda,
                "min_split_gain": self.config.min_split_gain,
                "class_weight": self.config.class_weight,
                "importance_type": self.config.importance_type,
                "random_state": self.config.random_state,
                "n_jobs": self.config.n_jobs,
                "verbose": -1,
            }
            
            # Set objective based on detected classes
            if n_classes == 2:
                params["objective"] = "binary"
                logger.debug(f"LightGBM: Auto-detected binary classification ({n_classes} classes)")
            else:
                params["objective"] = "multiclass"
                params["num_class"] = n_classes
                logger.debug(f"LightGBM: Auto-detected multiclass classification ({n_classes} classes)")
            
            self._model = lgb.LGBMClassifier(**params)
        
        # Prepare callbacks
        callbacks = []
        
        if self.config.early_stopping and X_val is not None:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self.config.early_stopping_rounds,
                    verbose=False,
                )
            )
            callbacks.append(lgb.log_evaluation(period=0))  # Suppress output
        
        # Fit with validation
        if X_val is not None and y_val is not None:
            self._model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks if callbacks else None,
                sample_weight=sample_weight,
            )
            self._best_iteration = self._model.best_iteration_
        else:
            self._model.fit(X, y, sample_weight=sample_weight)
            self._best_iteration = self.config.n_estimators
    
    def _predict_impl(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict with LightGBM."""
        return self._model.predict(X)
    
    def get_params_space(self) -> dict[str, Any]:
        """
        Get Optuna parameter space for hyperparameter optimization.
        
        Note: objective and num_class are NOT included - they are auto-detected.
        """
        return {
            "num_leaves": ("int", 16, 128),
            "max_depth": ("int", 3, 12),
            "min_child_samples": ("int", 5, 100),
            "learning_rate": ("float", 0.01, 0.3, "log"),
            "n_estimators": ("int", 100, 1000),
            "subsample": ("float", 0.5, 1.0),
            "colsample_bytree": ("float", 0.5, 1.0),
            "reg_alpha": ("float", 1e-8, 10.0, "log"),
            "reg_lambda": ("float", 1e-8, 10.0, "log"),
            "min_split_gain": ("float", 0.0, 1.0),
        }


# =============================================================================
# XGBOOST CLASSIFIER
# =============================================================================

@dataclass
class XGBoostClassifierConfig(ModelConfig):
    """
    Configuration for XGBoost Classifier.
    
    IMPORTANT: objective="auto" and num_class=None enable automatic
    detection of binary vs multiclass classification.
    
    XGBoost 2.0+ Compatibility:
    - early_stopping_rounds is set in constructor, not fit()
    - use_label_encoder is deprecated
    """
    name: str = "XGBoostClassifier"
    model_type: ModelType = ModelType.CLASSIFIER
    
    # Core parameters - AUTO-DETECT BY DEFAULT
    objective: str = "auto"  # FIXED: "auto" = detect from data
    num_class: int | None = None  # FIXED: None = detect from data
    booster: str = "gbtree"  # gbtree, gblinear, dart
    
    # Tree parameters
    max_depth: int = 6
    max_leaves: int = 0  # 0 = no limit
    min_child_weight: float = 1.0
    
    # Learning parameters
    learning_rate: float = 0.1
    n_estimators: int = 1000
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    colsample_bylevel: float = 1.0
    
    # Regularization
    gamma: float = 0.0  # Min loss reduction for split
    reg_alpha: float = 0.0  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization
    
    # Class imbalance
    scale_pos_weight: float = 1.0
    
    # GPU acceleration
    tree_method: str = "hist"  # hist, gpu_hist, auto
    
    # Feature importance
    importance_type: str = "gain"  # gain, weight, cover, total_gain, total_cover


class XGBoostClassifier(BaseModel[XGBoostClassifierConfig]):
    """
    XGBoost Classifier for trading signal prediction.
    
    Optimal for:
    - High accuracy requirements
    - Competition-winning performance
    - GPU acceleration (with tree_method="gpu_hist")
    - Handling structured/tabular data
    
    AUTOMATIC CLASS DETECTION:
    - Binary: Uses objective="binary:logistic"
    - Multiclass: Uses objective="multi:softprob" with correct num_class
    
    XGBoost 2.0+ Compatibility:
    - early_stopping_rounds is in constructor, not fit()
    - Proper handling of eval_set
    
    Example:
        config = XGBoostClassifierConfig(
            max_depth=7,
            learning_rate=0.05,
        )
        model = XGBoostClassifier(config)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    """
    
    def _default_config(self) -> XGBoostClassifierConfig:
        """Create default configuration."""
        return XGBoostClassifierConfig()
    
    def _build_model(self) -> Any:
        """
        Build XGBoost classifier.
        
        Note: If objective="auto", actual objective is set in _fit_impl()
        after class detection.
        """
        import xgboost as xgb
        
        params = {
            "booster": self.config.booster,
            "max_depth": self.config.max_depth,
            "max_leaves": self.config.max_leaves,
            "min_child_weight": self.config.min_child_weight,
            "learning_rate": self.config.learning_rate,
            "n_estimators": self.config.n_estimators,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "colsample_bylevel": self.config.colsample_bylevel,
            "gamma": self.config.gamma,
            "reg_alpha": self.config.reg_alpha,
            "reg_lambda": self.config.reg_lambda,
            "scale_pos_weight": self.config.scale_pos_weight,
            "tree_method": self.config.tree_method,
            "importance_type": self.config.importance_type,
            "random_state": self.config.random_state,
            "n_jobs": self.config.n_jobs,
            "verbosity": 0,
            "use_label_encoder": False,  # Suppress deprecation warning
        }
        
        # Only set objective if not auto-detecting
        if self.config.objective != "auto":
            params["objective"] = self.config.objective
            if "multi" in self.config.objective and self.config.num_class:
                params["num_class"] = self.config.num_class
        
        # XGBoost 2.0+: early_stopping_rounds in constructor
        if self.config.early_stopping:
            params["early_stopping_rounds"] = self.config.early_stopping_rounds
        
        return xgb.XGBClassifier(**params)
    
    def _fit_impl(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.float64] | None = None,
        sample_weight: NDArray[np.float64] | None = None,
    ) -> None:
        """
        Train XGBoost classifier with automatic class detection.
        
        CRITICAL FIX: This method auto-detects binary vs multiclass
        and rebuilds the model with correct parameters.
        """
        import xgboost as xgb
        
        # =====================================================================
        # AUTO-DETECT NUMBER OF CLASSES
        # =====================================================================
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        
        # Store for evaluation
        self._n_classes = n_classes
        self._classes = unique_classes
        
        # Rebuild model with correct objective if auto-detecting
        if self.config.objective == "auto":
            params = {
                "booster": self.config.booster,
                "max_depth": self.config.max_depth,
                "max_leaves": self.config.max_leaves,
                "min_child_weight": self.config.min_child_weight,
                "learning_rate": self.config.learning_rate,
                "n_estimators": self.config.n_estimators,
                "subsample": self.config.subsample,
                "colsample_bytree": self.config.colsample_bytree,
                "colsample_bylevel": self.config.colsample_bylevel,
                "gamma": self.config.gamma,
                "reg_alpha": self.config.reg_alpha,
                "reg_lambda": self.config.reg_lambda,
                "scale_pos_weight": self.config.scale_pos_weight,
                "tree_method": self.config.tree_method,
                "importance_type": self.config.importance_type,
                "random_state": self.config.random_state,
                "n_jobs": self.config.n_jobs,
                "verbosity": 0,
                "use_label_encoder": False,
            }
            
            # Set objective based on detected classes
            if n_classes == 2:
                params["objective"] = "binary:logistic"
                logger.debug(f"XGBoost: Auto-detected binary classification ({n_classes} classes)")
            else:
                params["objective"] = "multi:softprob"
                params["num_class"] = n_classes
                logger.debug(f"XGBoost: Auto-detected multiclass classification ({n_classes} classes)")
            
            # XGBoost 2.0+: early_stopping_rounds in constructor
            if self.config.early_stopping:
                params["early_stopping_rounds"] = self.config.early_stopping_rounds
            
            self._model = xgb.XGBClassifier(**params)
        
        # Prepare fit parameters
        fit_params = {
            "sample_weight": sample_weight,
            "verbose": False,
        }
        
        # XGBoost 2.0+: Only pass eval_set, not early_stopping_rounds
        if self.config.early_stopping and X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
        
        # Train
        self._model.fit(X, y, **fit_params)
        
        # Store best iteration
        if hasattr(self._model, "best_iteration"):
            self._best_iteration = self._model.best_iteration
        else:
            self._best_iteration = self.config.n_estimators
    
    def _predict_impl(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict with XGBoost."""
        return self._model.predict(X)
    
    def get_params_space(self) -> dict[str, Any]:
        """
        Get Optuna parameter space for hyperparameter optimization.
        
        Note: objective and num_class are NOT included - they are auto-detected.
        """
        return {
            "max_depth": ("int", 3, 12),
            "min_child_weight": ("float", 0.1, 10.0, "log"),
            "learning_rate": ("float", 0.01, 0.3, "log"),
            "n_estimators": ("int", 100, 1000),
            "subsample": ("float", 0.5, 1.0),
            "colsample_bytree": ("float", 0.5, 1.0),
            "gamma": ("float", 0.0, 5.0),
            "reg_alpha": ("float", 1e-8, 10.0, "log"),
            "reg_lambda": ("float", 1e-8, 10.0, "log"),
        }


# =============================================================================
# CATBOOST CLASSIFIER
# =============================================================================

@dataclass
class CatBoostClassifierConfig(ModelConfig):
    """Configuration for CatBoost Classifier."""
    name: str = "CatBoostClassifier"
    model_type: ModelType = ModelType.CLASSIFIER
    
    # Core parameters - AUTO-DETECT
    loss_function: str = "auto"  # auto, Logloss, MultiClass
    iterations: int = 1000
    depth: int = 6
    
    # Learning parameters
    learning_rate: float = 0.1
    l2_leaf_reg: float = 3.0
    
    # Sampling
    bagging_temperature: float = 1.0
    subsample: float = 0.8
    colsample_bylevel: float = 1.0
    
    # Features
    border_count: int = 254
    grow_policy: str = "SymmetricTree"  # SymmetricTree, Depthwise, Lossguide
    
    # Class imbalance
    auto_class_weights: str | None = "Balanced"
    
    # Device
    task_type: str = "CPU"  # CPU or GPU


class CatBoostClassifier(BaseModel[CatBoostClassifierConfig]):
    """
    CatBoost Classifier for trading signal prediction.
    
    Optimal for:
    - Datasets with categorical features
    - Missing value handling
    - Reduced overfitting
    - GPU acceleration
    
    Example:
        config = CatBoostClassifierConfig(
            depth=8,
            learning_rate=0.05,
        )
        model = CatBoostClassifier(config)
        model.fit(X_train, y_train)
    """
    
    def _default_config(self) -> CatBoostClassifierConfig:
        return CatBoostClassifierConfig()
    
    def _build_model(self) -> Any:
        """Build CatBoost classifier."""
        from catboost import CatBoostClassifier as CBClassifier
        
        params = {
            "iterations": self.config.iterations,
            "depth": self.config.depth,
            "learning_rate": self.config.learning_rate,
            "l2_leaf_reg": self.config.l2_leaf_reg,
            "bagging_temperature": self.config.bagging_temperature,
            "subsample": self.config.subsample,
            "colsample_bylevel": self.config.colsample_bylevel,
            "border_count": self.config.border_count,
            "grow_policy": self.config.grow_policy,
            "auto_class_weights": self.config.auto_class_weights,
            "task_type": self.config.task_type,
            "random_state": self.config.random_state,
            "verbose": False,
        }
        
        # Only set loss_function if not auto
        if self.config.loss_function != "auto":
            params["loss_function"] = self.config.loss_function
        
        return CBClassifier(**params)
    
    def _fit_impl(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.float64] | None = None,
        sample_weight: NDArray[np.float64] | None = None,
    ) -> None:
        """Train CatBoost classifier with auto class detection."""
        from catboost import CatBoostClassifier as CBClassifier
        
        # Auto-detect classes
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        self._n_classes = n_classes
        self._classes = unique_classes
        
        # Rebuild with correct loss function if auto
        if self.config.loss_function == "auto":
            params = {
                "iterations": self.config.iterations,
                "depth": self.config.depth,
                "learning_rate": self.config.learning_rate,
                "l2_leaf_reg": self.config.l2_leaf_reg,
                "bagging_temperature": self.config.bagging_temperature,
                "subsample": self.config.subsample,
                "colsample_bylevel": self.config.colsample_bylevel,
                "border_count": self.config.border_count,
                "grow_policy": self.config.grow_policy,
                "auto_class_weights": self.config.auto_class_weights,
                "task_type": self.config.task_type,
                "random_state": self.config.random_state,
                "verbose": False,
            }
            
            if n_classes == 2:
                params["loss_function"] = "Logloss"
                logger.debug(f"CatBoost: Auto-detected binary classification")
            else:
                params["loss_function"] = "MultiClass"
                logger.debug(f"CatBoost: Auto-detected multiclass classification ({n_classes} classes)")
            
            self._model = CBClassifier(**params)
        
        # Fit
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
        
        self._model.fit(
            X, y,
            eval_set=eval_set,
            sample_weight=sample_weight,
            early_stopping_rounds=self.config.early_stopping_rounds if self.config.early_stopping else None,
            verbose=False,
        )
        
        if hasattr(self._model, "best_iteration_"):
            self._best_iteration = self._model.best_iteration_
    
    def _predict_impl(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict with CatBoost."""
        return self._model.predict(X).ravel()
    
    def get_params_space(self) -> dict[str, Any]:
        """Get Optuna parameter space."""
        return {
            "depth": ("int", 4, 10),
            "learning_rate": ("float", 0.01, 0.3, "log"),
            "iterations": ("int", 100, 1000),
            "l2_leaf_reg": ("float", 1.0, 10.0),
            "bagging_temperature": ("float", 0.0, 2.0),
            "subsample": ("float", 0.5, 1.0),
        }


# =============================================================================
# RANDOM FOREST CLASSIFIER
# =============================================================================

@dataclass
class RandomForestClassifierConfig(ModelConfig):
    """Configuration for Random Forest Classifier."""
    name: str = "RandomForestClassifier"
    model_type: ModelType = ModelType.CLASSIFIER
    
    # Core parameters
    n_estimators: int = 500
    max_depth: int | None = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    max_features: str | float = "sqrt"  # sqrt, log2, or float
    
    # Bootstrap
    bootstrap: bool = True
    oob_score: bool = True
    
    # Class imbalance
    class_weight: str | dict | None = "balanced"
    
    # Performance
    max_samples: float | None = None  # Bootstrap sample size


class RandomForestClassifier(BaseModel[RandomForestClassifierConfig]):
    """
    Random Forest Classifier for trading signal prediction.
    
    Optimal for:
    - Robust baseline model
    - Feature importance analysis
    - Handling noisy data
    - Out-of-bag error estimation
    
    Example:
        config = RandomForestClassifierConfig(
            n_estimators=500,
            max_depth=12,
        )
        model = RandomForestClassifier(config)
        model.fit(X_train, y_train)
    """
    
    def _default_config(self) -> RandomForestClassifierConfig:
        return RandomForestClassifierConfig()
    
    def _build_model(self) -> Any:
        """Build Random Forest classifier."""
        from sklearn.ensemble import RandomForestClassifier as SKRandomForest
        
        return SKRandomForest(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            bootstrap=self.config.bootstrap,
            oob_score=self.config.oob_score,
            class_weight=self.config.class_weight,
            max_samples=self.config.max_samples,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbose=self.config.verbose,
        )
    
    def _fit_impl(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.float64] | None = None,
        sample_weight: NDArray[np.float64] | None = None,
    ) -> None:
        """Train Random Forest classifier."""
        # Auto-detect classes
        self._n_classes = len(np.unique(y))
        self._classes = np.unique(y)
        
        self._model.fit(X, y, sample_weight=sample_weight)
        
        # Log OOB score if available
        if self.config.oob_score and hasattr(self._model, "oob_score_"):
            logger.debug(f"Random Forest OOB score: {self._model.oob_score_:.4f}")
    
    def _predict_impl(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict with Random Forest."""
        return self._model.predict(X)
    
    def get_params_space(self) -> dict[str, Any]:
        """Get Optuna parameter space."""
        return {
            "n_estimators": ("int", 100, 1000),
            "max_depth": ("int", 3, 20),
            "min_samples_split": ("int", 2, 20),
            "min_samples_leaf": ("int", 1, 10),
            "max_features": ("categorical", ["sqrt", "log2", 0.3, 0.5]),
        }


# =============================================================================
# EXTRA TREES CLASSIFIER
# =============================================================================

@dataclass
class ExtraTreesClassifierConfig(ModelConfig):
    """Configuration for Extra Trees Classifier."""
    name: str = "ExtraTreesClassifier"
    model_type: ModelType = ModelType.CLASSIFIER
    
    # Core parameters
    n_estimators: int = 500
    max_depth: int | None = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    max_features: str | float = "sqrt"
    
    # Bootstrap
    bootstrap: bool = False  # Extra Trees typically don't use bootstrap
    
    # Class imbalance
    class_weight: str | dict | None = "balanced"


class ExtraTreesClassifier(BaseModel[ExtraTreesClassifierConfig]):
    """
    Extra Trees Classifier for trading signal prediction.
    
    Extra Trees (Extremely Randomized Trees) differ from Random Forest:
    - Uses all samples (no bootstrap by default)
    - Random split points (not best split)
    - Often faster and more random
    
    Example:
        config = ExtraTreesClassifierConfig(
            n_estimators=500,
            max_depth=12,
        )
        model = ExtraTreesClassifier(config)
        model.fit(X_train, y_train)
    """
    
    def _default_config(self) -> ExtraTreesClassifierConfig:
        return ExtraTreesClassifierConfig()
    
    def _build_model(self) -> Any:
        """Build Extra Trees classifier."""
        from sklearn.ensemble import ExtraTreesClassifier as SKExtraTrees
        
        return SKExtraTrees(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            bootstrap=self.config.bootstrap,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbose=self.config.verbose,
        )
    
    def _fit_impl(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.float64] | None = None,
        sample_weight: NDArray[np.float64] | None = None,
    ) -> None:
        """Train Extra Trees classifier."""
        self._n_classes = len(np.unique(y))
        self._classes = np.unique(y)
        self._model.fit(X, y, sample_weight=sample_weight)
    
    def _predict_impl(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict with Extra Trees."""
        return self._model.predict(X)
    
    def get_params_space(self) -> dict[str, Any]:
        """Get Optuna parameter space."""
        return {
            "n_estimators": ("int", 100, 1000),
            "max_depth": ("int", 3, 15),
            "min_samples_split": ("int", 2, 20),
            "min_samples_leaf": ("int", 1, 10),
            "max_features": ("categorical", ["sqrt", "log2", 0.3, 0.5]),
        }


# =============================================================================
# STACKING CLASSIFIER
# =============================================================================

@dataclass
class StackingClassifierConfig(ModelConfig):
    """Configuration for Stacking Classifier."""
    name: str = "StackingClassifier"
    model_type: ModelType = ModelType.ENSEMBLE
    
    # Base models to use
    use_lightgbm: bool = True
    use_xgboost: bool = True
    use_catboost: bool = False
    use_rf: bool = True
    
    # Meta-learner
    meta_learner: str = "logistic"  # logistic, lightgbm
    
    # Stacking configuration
    cv: int = 5  # Cross-validation folds for stacking
    passthrough: bool = True  # Include original features in meta-learner


class StackingClassifier(BaseModel[StackingClassifierConfig]):
    """
    Stacking Ensemble Classifier for trading signal prediction.
    
    Combines multiple base learners with a meta-learner for
    maximum predictive performance.
    
    Architecture:
    - Level 0: Base learners (LightGBM, XGBoost, RF, etc.)
    - Level 1: Meta-learner combines base predictions
    
    Optimal for:
    - Maximum accuracy requirements
    - Diverse model ensemble
    - Production deployment
    
    Example:
        config = StackingClassifierConfig(
            use_lightgbm=True,
            use_xgboost=True,
            use_rf=True,
            meta_learner="lightgbm",
        )
        model = StackingClassifier(config)
        model.fit(X_train, y_train)
    """
    
    def _default_config(self) -> StackingClassifierConfig:
        return StackingClassifierConfig()
    
    def _build_model(self) -> Any:
        """Build Stacking classifier."""
        from sklearn.ensemble import StackingClassifier as SKStacking
        from sklearn.linear_model import LogisticRegression
        
        estimators = []
        
        # Add base estimators
        if self.config.use_lightgbm:
            import lightgbm as lgb
            estimators.append((
                "lgb",
                lgb.LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    num_leaves=31,
                    random_state=self.config.random_state,
                    verbose=-1,
                )
            ))
        
        if self.config.use_xgboost:
            import xgboost as xgb
            estimators.append((
                "xgb",
                xgb.XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=self.config.random_state,
                    verbosity=0,
                    use_label_encoder=False,
                )
            ))
        
        if self.config.use_catboost:
            from catboost import CatBoostClassifier
            estimators.append((
                "cb",
                CatBoostClassifier(
                    iterations=200,
                    learning_rate=0.1,
                    depth=6,
                    random_state=self.config.random_state,
                    verbose=False,
                )
            ))
        
        if self.config.use_rf:
            from sklearn.ensemble import RandomForestClassifier
            estimators.append((
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                )
            ))
        
        # Meta-learner
        if self.config.meta_learner == "logistic":
            final_estimator = LogisticRegression(
                max_iter=1000,
                random_state=self.config.random_state,
            )
        else:
            import lightgbm as lgb
            final_estimator = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=self.config.random_state,
                verbose=-1,
            )
        
        return SKStacking(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=self.config.cv,
            passthrough=self.config.passthrough,
            n_jobs=self.config.n_jobs,
            verbose=self.config.verbose,
        )
    
    def _fit_impl(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.float64] | None = None,
        sample_weight: NDArray[np.float64] | None = None,
    ) -> None:
        """Train Stacking classifier."""
        self._n_classes = len(np.unique(y))
        self._classes = np.unique(y)
        self._model.fit(X, y, sample_weight=sample_weight)
    
    def _predict_impl(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict with Stacking."""
        return self._model.predict(X)
    
    def _extract_feature_importance(self) -> None:
        """Extract feature importance from meta-learner."""
        if hasattr(self._model, "final_estimator_"):
            if hasattr(self._model.final_estimator_, "feature_importances_"):
                raw_importance = self._model.final_estimator_.feature_importances_
                
                for i, imp in enumerate(raw_importance):
                    if i < len(self._feature_names):
                        self._feature_importance[self._feature_names[i]] = float(imp)


# =============================================================================
# VOTING CLASSIFIER
# =============================================================================

@dataclass
class VotingClassifierConfig(ModelConfig):
    """Configuration for Voting Classifier."""
    name: str = "VotingClassifier"
    model_type: ModelType = ModelType.ENSEMBLE
    
    # Voting strategy
    voting: str = "soft"  # soft (probability), hard (majority)
    
    # Models to use
    use_lightgbm: bool = True
    use_xgboost: bool = True
    use_rf: bool = True
    
    # Weights for each model (None = equal weights)
    weights: list[float] | None = None


class VotingClassifier(BaseModel[VotingClassifierConfig]):
    """
    Voting Ensemble Classifier for trading signal prediction.
    
    Combines predictions from multiple models via voting:
    - Soft voting: Weighted average of probabilities
    - Hard voting: Majority vote of predictions
    
    Example:
        config = VotingClassifierConfig(
            voting="soft",
            weights=[0.4, 0.4, 0.2],  # LGB, XGB, RF
        )
        model = VotingClassifier(config)
        model.fit(X_train, y_train)
    """
    
    def _default_config(self) -> VotingClassifierConfig:
        return VotingClassifierConfig()
    
    def _build_model(self) -> Any:
        """Build Voting classifier."""
        from sklearn.ensemble import VotingClassifier as SKVoting
        
        estimators = []
        
        if self.config.use_lightgbm:
            import lightgbm as lgb
            estimators.append((
                "lgb",
                lgb.LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.1,
                    random_state=self.config.random_state,
                    verbose=-1,
                )
            ))
        
        if self.config.use_xgboost:
            import xgboost as xgb
            estimators.append((
                "xgb",
                xgb.XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.1,
                    random_state=self.config.random_state,
                    verbosity=0,
                    use_label_encoder=False,
                )
            ))
        
        if self.config.use_rf:
            from sklearn.ensemble import RandomForestClassifier
            estimators.append((
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                )
            ))
        
        return SKVoting(
            estimators=estimators,
            voting=self.config.voting,
            weights=self.config.weights,
            n_jobs=self.config.n_jobs,
            verbose=self.config.verbose > 0,
        )
    
    def _fit_impl(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.float64] | None = None,
        sample_weight: NDArray[np.float64] | None = None,
    ) -> None:
        """Train Voting classifier."""
        self._n_classes = len(np.unique(y))
        self._classes = np.unique(y)
        self._model.fit(X, y, sample_weight=sample_weight)
    
    def _predict_impl(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict with Voting."""
        return self._model.predict(X)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_classifier(
    model_type: str,
    **kwargs: Any,
) -> BaseModel:
    """
    Factory function to create classifier models.
    
    Args:
        model_type: Type of classifier (lightgbm, xgboost, catboost, rf, etc.)
        **kwargs: Model configuration parameters
    
    Returns:
        Configured classifier model instance
    
    Example:
        # Create LightGBM with custom parameters
        model = create_classifier("lightgbm", learning_rate=0.05, num_leaves=63)
        
        # Create XGBoost with defaults
        model = create_classifier("xgboost")
        
        # Create Random Forest
        model = create_classifier("rf", n_estimators=500)
    """
    model_map = {
        "lightgbm": (LightGBMClassifier, LightGBMClassifierConfig),
        "lgb": (LightGBMClassifier, LightGBMClassifierConfig),
        "xgboost": (XGBoostClassifier, XGBoostClassifierConfig),
        "xgb": (XGBoostClassifier, XGBoostClassifierConfig),
        "catboost": (CatBoostClassifier, CatBoostClassifierConfig),
        "cb": (CatBoostClassifier, CatBoostClassifierConfig),
        "random_forest": (RandomForestClassifier, RandomForestClassifierConfig),
        "rf": (RandomForestClassifier, RandomForestClassifierConfig),
        "extra_trees": (ExtraTreesClassifier, ExtraTreesClassifierConfig),
        "et": (ExtraTreesClassifier, ExtraTreesClassifierConfig),
        "stacking": (StackingClassifier, StackingClassifierConfig),
        "voting": (VotingClassifier, VotingClassifierConfig),
    }
    
    model_type = model_type.lower()
    
    if model_type not in model_map:
        available = list(model_map.keys())
        raise ValueError(
            f"Unknown classifier type: {model_type}. "
            f"Available: {available}"
        )
    
    model_class, config_class = model_map[model_type]
    
    # Filter kwargs to valid config parameters
    valid_kwargs = {
        k: v for k, v in kwargs.items()
        if hasattr(config_class, k)
    }
    
    config = config_class(**valid_kwargs)
    return model_class(config=config)


# =============================================================================
# REGISTER MODELS
# =============================================================================

ModelRegistry.register("lightgbm_classifier", LightGBMClassifier, LightGBMClassifierConfig)
ModelRegistry.register("xgboost_classifier", XGBoostClassifier, XGBoostClassifierConfig)
ModelRegistry.register("catboost_classifier", CatBoostClassifier, CatBoostClassifierConfig)
ModelRegistry.register("random_forest_classifier", RandomForestClassifier, RandomForestClassifierConfig)
ModelRegistry.register("extra_trees_classifier", ExtraTreesClassifier, ExtraTreesClassifierConfig)
ModelRegistry.register("stacking_classifier", StackingClassifier, StackingClassifierConfig)
ModelRegistry.register("voting_classifier", VotingClassifier, VotingClassifierConfig)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # LightGBM
    "LightGBMClassifierConfig",
    "LightGBMClassifier",
    # XGBoost
    "XGBoostClassifierConfig",
    "XGBoostClassifier",
    # CatBoost
    "CatBoostClassifierConfig",
    "CatBoostClassifier",
    # Random Forest
    "RandomForestClassifierConfig",
    "RandomForestClassifier",
    # Extra Trees
    "ExtraTreesClassifierConfig",
    "ExtraTreesClassifier",
    # Stacking
    "StackingClassifierConfig",
    "StackingClassifier",
    # Voting
    "VotingClassifierConfig",
    "VotingClassifier",
    # Factory
    "create_classifier",
]