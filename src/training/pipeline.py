"""
Scikit-Learn Pipeline Factory for standardized preprocessing + model pipelines.

This module provides integrated preprocessing pipelines that:
- Ensure consistent data transformation between training and inference
- Prevent data leakage in cross-validation
- Save all preprocessing parameters with the model
- Support financial-specific transformations (Winsorization, RobustScaling)

Designed for JPMorgan-level requirements:
- Production-safe inference with embedded preprocessing
- Audit trail for all transformations
- Serialization of complete pipeline (Scaler + Model)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    QuantileTransformer,
    PowerTransformer,
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin, clone


logger = logging.getLogger(__name__)


class ScalerType(str, Enum):
    """Supported scaler types."""
    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"
    QUANTILE = "quantile"
    YEOHJOHNSON = "yeojohnson"
    NONE = "none"


class ImputerType(str, Enum):
    """Supported imputer types."""
    CONSTANT = "constant"
    MEAN = "mean"
    MEDIAN = "median"
    MOST_FREQUENT = "most_frequent"
    KNN = "knn"
    NONE = "none"


@dataclass
class PipelineConfig:
    """Configuration for preprocessing pipeline."""
    scaler_type: ScalerType = ScalerType.ROBUST
    imputer_type: ImputerType = ImputerType.CONSTANT
    imputer_fill_value: float = 0.0
    winsorize: bool = True
    winsorize_limits: Tuple[float, float] = (0.01, 0.99)
    clip_outliers: bool = True
    clip_std: float = 5.0
    handle_infinity: bool = True
    feature_names: Optional[List[str]] = None


class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Winsorize extreme values to specified percentiles.

    Financial data often contains outliers that can destabilize models.
    Winsorization clips values to percentile bounds rather than removing them.

    This transformer fits percentile bounds on training data and applies
    them consistently to new data, preventing data leakage.
    """

    def __init__(
        self,
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99,
    ):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_bounds_: Optional[np.ndarray] = None
        self.upper_bounds_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y=None) -> "Winsorizer":
        """Fit percentile bounds on training data."""
        X = np.asarray(X)
        self.lower_bounds_ = np.nanpercentile(
            X, self.lower_percentile * 100, axis=0
        )
        self.upper_bounds_ = np.nanpercentile(
            X, self.upper_percentile * 100, axis=0
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply winsorization using fitted bounds."""
        X = np.asarray(X).copy()

        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise ValueError("Winsorizer not fitted. Call fit() first.")

        for i in range(X.shape[1]):
            X[:, i] = np.clip(
                X[:, i],
                self.lower_bounds_[i],
                self.upper_bounds_[i]
            )

        return X

    def get_feature_names_out(self, input_features=None):
        """Return feature names (passthrough)."""
        return input_features


class InfinityHandler(BaseEstimator, TransformerMixin):
    """
    Handle infinite values in financial data.

    Replaces +inf and -inf with large finite values or column max/min
    to prevent model training failures.
    """

    def __init__(self, strategy: str = "clip"):
        """
        Args:
            strategy: "clip" (use large values) or "nan" (convert to NaN for imputer)
        """
        self.strategy = strategy
        self.max_vals_: Optional[np.ndarray] = None
        self.min_vals_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y=None) -> "InfinityHandler":
        """Fit max/min values for clipping strategy."""
        X = np.asarray(X)
        # Get finite max/min values
        finite_mask = np.isfinite(X)
        self.max_vals_ = np.where(
            finite_mask, X, np.nan
        ).max(axis=0)
        self.min_vals_ = np.where(
            finite_mask, X, np.nan
        ).min(axis=0)
        # Handle case where entire column is inf
        self.max_vals_ = np.nan_to_num(self.max_vals_, nan=1e10)
        self.min_vals_ = np.nan_to_num(self.min_vals_, nan=-1e10)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Replace infinite values."""
        X = np.asarray(X).copy()

        if self.strategy == "clip":
            for i in range(X.shape[1]):
                X[:, i] = np.where(
                    np.isposinf(X[:, i]),
                    self.max_vals_[i] * 1.5,
                    X[:, i]
                )
                X[:, i] = np.where(
                    np.isneginf(X[:, i]),
                    self.min_vals_[i] * 1.5,
                    X[:, i]
                )
        else:  # nan strategy
            X = np.where(np.isinf(X), np.nan, X)

        return X

    def get_feature_names_out(self, input_features=None):
        """Return feature names (passthrough)."""
        return input_features


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clip outliers based on standard deviations from mean.

    Provides an alternative to Winsorization based on
    parametric assumptions (mean + n*std).
    """

    def __init__(self, n_std: float = 5.0):
        self.n_std = n_std
        self.means_: Optional[np.ndarray] = None
        self.stds_: Optional[np.ndarray] = None
        self.lower_bounds_: Optional[np.ndarray] = None
        self.upper_bounds_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y=None) -> "OutlierClipper":
        """Fit mean and std on training data."""
        X = np.asarray(X)
        self.means_ = np.nanmean(X, axis=0)
        self.stds_ = np.nanstd(X, axis=0)
        # Prevent zero std
        self.stds_ = np.where(self.stds_ == 0, 1, self.stds_)

        self.lower_bounds_ = self.means_ - self.n_std * self.stds_
        self.upper_bounds_ = self.means_ + self.n_std * self.stds_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Clip values beyond n standard deviations."""
        X = np.asarray(X).copy()

        if self.lower_bounds_ is None:
            raise ValueError("OutlierClipper not fitted.")

        for i in range(X.shape[1]):
            X[:, i] = np.clip(
                X[:, i],
                self.lower_bounds_[i],
                self.upper_bounds_[i]
            )

        return X

    def get_feature_names_out(self, input_features=None):
        """Return feature names (passthrough)."""
        return input_features


class ModelPipelineFactory:
    """
    Factory for creating Scikit-Learn pipelines with preprocessing + model.

    Creates end-to-end pipelines that:
    1. Handle missing values (Imputation)
    2. Handle infinite values
    3. Winsorize or clip outliers
    4. Scale features
    5. Train/predict with the model

    The entire pipeline is saved as a single object, ensuring
    consistent preprocessing between training and inference.

    Example:
        factory = ModelPipelineFactory()

        # Create pipeline with model
        pipeline = factory.create_pipeline(
            model=lgb.LGBMClassifier(),
            config=PipelineConfig(
                scaler_type=ScalerType.ROBUST,
                winsorize=True
            )
        )

        # Train - preprocessing is applied within CV folds
        pipeline.fit(X_train, y_train)

        # Predict - same preprocessing automatically applied
        predictions = pipeline.predict(X_test)

        # Save entire pipeline
        factory.save_pipeline(pipeline, "model_pipeline.joblib")
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    def create_pipeline(
        self,
        model: Any,
        config: Optional[PipelineConfig] = None,
    ) -> Pipeline:
        """
        Create a complete preprocessing + model pipeline.

        Args:
            model: Scikit-learn compatible model/estimator
            config: Pipeline configuration (uses default if not provided)

        Returns:
            Sklearn Pipeline with preprocessing and model
        """
        config = config or self.config
        steps = []

        # Step 1: Handle infinite values
        if config.handle_infinity:
            steps.append(('infinity_handler', InfinityHandler(strategy='clip')))

        # Step 2: Imputation
        imputer = self._create_imputer(config)
        if imputer is not None:
            steps.append(('imputer', imputer))

        # Step 3: Outlier handling (Winsorization or Clipping)
        if config.winsorize:
            steps.append((
                'winsorizer',
                Winsorizer(
                    lower_percentile=config.winsorize_limits[0],
                    upper_percentile=config.winsorize_limits[1]
                )
            ))
        elif config.clip_outliers:
            steps.append((
                'outlier_clipper',
                OutlierClipper(n_std=config.clip_std)
            ))

        # Step 4: Scaling
        scaler = self._create_scaler(config)
        if scaler is not None:
            steps.append(('scaler', scaler))

        # Step 5: Model
        steps.append(('model', model))

        pipeline = Pipeline(steps)

        logger.info(
            f"Created pipeline with {len(steps)} steps: "
            f"{[s[0] for s in steps]}"
        )

        return pipeline

    def create_preprocessing_pipeline(
        self,
        config: Optional[PipelineConfig] = None,
    ) -> Pipeline:
        """
        Create preprocessing-only pipeline (without model).

        Useful when you want to transform data separately
        or use with custom training loops.

        Args:
            config: Pipeline configuration

        Returns:
            Sklearn Pipeline for preprocessing only
        """
        config = config or self.config
        steps = []

        if config.handle_infinity:
            steps.append(('infinity_handler', InfinityHandler(strategy='clip')))

        imputer = self._create_imputer(config)
        if imputer is not None:
            steps.append(('imputer', imputer))

        if config.winsorize:
            steps.append((
                'winsorizer',
                Winsorizer(
                    lower_percentile=config.winsorize_limits[0],
                    upper_percentile=config.winsorize_limits[1]
                )
            ))
        elif config.clip_outliers:
            steps.append((
                'outlier_clipper',
                OutlierClipper(n_std=config.clip_std)
            ))

        scaler = self._create_scaler(config)
        if scaler is not None:
            steps.append(('scaler', scaler))

        return Pipeline(steps)

    def _create_imputer(self, config: PipelineConfig) -> Optional[BaseEstimator]:
        """Create imputer based on configuration."""
        if config.imputer_type == ImputerType.NONE:
            return None
        elif config.imputer_type == ImputerType.CONSTANT:
            return SimpleImputer(
                strategy='constant',
                fill_value=config.imputer_fill_value
            )
        elif config.imputer_type == ImputerType.MEAN:
            return SimpleImputer(strategy='mean')
        elif config.imputer_type == ImputerType.MEDIAN:
            return SimpleImputer(strategy='median')
        elif config.imputer_type == ImputerType.MOST_FREQUENT:
            return SimpleImputer(strategy='most_frequent')
        elif config.imputer_type == ImputerType.KNN:
            return KNNImputer(n_neighbors=5)
        else:
            return None

    def _create_scaler(self, config: PipelineConfig) -> Optional[BaseEstimator]:
        """Create scaler based on configuration."""
        if config.scaler_type == ScalerType.NONE:
            return None
        elif config.scaler_type == ScalerType.STANDARD:
            return StandardScaler()
        elif config.scaler_type == ScalerType.ROBUST:
            return RobustScaler()
        elif config.scaler_type == ScalerType.MINMAX:
            return MinMaxScaler()
        elif config.scaler_type == ScalerType.QUANTILE:
            return QuantileTransformer(output_distribution='normal')
        elif config.scaler_type == ScalerType.YEOHJOHNSON:
            return PowerTransformer(method='yeo-johnson')
        else:
            return None

    @staticmethod
    def save_pipeline(
        pipeline: Pipeline,
        path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save pipeline to disk.

        Saves the complete pipeline including all fitted
        preprocessing parameters and model weights.

        Args:
            pipeline: Fitted sklearn Pipeline
            path: Path to save
            metadata: Optional metadata to save alongside
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_obj = {
            'pipeline': pipeline,
            'metadata': metadata or {},
        }

        joblib.dump(save_obj, path)
        logger.info(f"Pipeline saved to {path}")

    @staticmethod
    def load_pipeline(path: Union[str, Path]) -> Tuple[Pipeline, Dict[str, Any]]:
        """
        Load pipeline from disk.

        Args:
            path: Path to load from

        Returns:
            Tuple of (pipeline, metadata)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {path}")

        save_obj = joblib.load(path)

        pipeline = save_obj['pipeline']
        metadata = save_obj.get('metadata', {})

        logger.info(f"Pipeline loaded from {path}")
        return pipeline, metadata

    @staticmethod
    def get_preprocessing_params(pipeline: Pipeline) -> Dict[str, Any]:
        """
        Extract preprocessing parameters from fitted pipeline.

        Useful for auditing and documentation.

        Args:
            pipeline: Fitted sklearn Pipeline

        Returns:
            Dictionary of all fitted parameters
        """
        params = {}

        for name, step in pipeline.named_steps.items():
            if name == 'model':
                continue

            step_params = {}

            if hasattr(step, 'lower_bounds_'):
                step_params['lower_bounds'] = step.lower_bounds_.tolist()
            if hasattr(step, 'upper_bounds_'):
                step_params['upper_bounds'] = step.upper_bounds_.tolist()
            if hasattr(step, 'mean_'):
                step_params['mean'] = step.mean_.tolist()
            if hasattr(step, 'scale_'):
                step_params['scale'] = step.scale_.tolist()
            if hasattr(step, 'center_'):
                step_params['center'] = step.center_.tolist()
            if hasattr(step, 'statistics_'):
                step_params['statistics'] = step.statistics_.tolist()

            if step_params:
                params[name] = step_params

        return params


class FinancialPreprocessor(BaseEstimator, TransformerMixin):
    """
    All-in-one financial data preprocessor.

    Combines all preprocessing steps into a single transformer
    optimized for financial time-series data.

    Features:
    - Handles NaN, Inf values
    - Winsorization for outliers
    - Robust scaling
    - Feature-wise statistics tracking

    Example:
        preprocessor = FinancialPreprocessor(
            winsorize_limits=(0.01, 0.99),
            scaler_type='robust'
        )

        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
    """

    def __init__(
        self,
        handle_nan: bool = True,
        fill_value: float = 0.0,
        handle_inf: bool = True,
        winsorize: bool = True,
        winsorize_limits: Tuple[float, float] = (0.01, 0.99),
        scaler_type: str = "robust",
    ):
        self.handle_nan = handle_nan
        self.fill_value = fill_value
        self.handle_inf = handle_inf
        self.winsorize = winsorize
        self.winsorize_limits = winsorize_limits
        self.scaler_type = scaler_type

        # Fitted parameters
        self.lower_bounds_: Optional[np.ndarray] = None
        self.upper_bounds_: Optional[np.ndarray] = None
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.max_vals_: Optional[np.ndarray] = None
        self.min_vals_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None

    def fit(self, X: np.ndarray, y=None) -> "FinancialPreprocessor":
        """Fit preprocessor on training data."""
        X = np.asarray(X).copy()
        self.n_features_ = X.shape[1]

        # Handle infinity first
        if self.handle_inf:
            finite_mask = np.isfinite(X)
            self.max_vals_ = np.nanmax(
                np.where(finite_mask, X, np.nan), axis=0
            )
            self.min_vals_ = np.nanmin(
                np.where(finite_mask, X, np.nan), axis=0
            )
            self.max_vals_ = np.nan_to_num(self.max_vals_, nan=1e10)
            self.min_vals_ = np.nan_to_num(self.min_vals_, nan=-1e10)

            # Replace inf for fitting
            X = np.where(np.isposinf(X), self.max_vals_ * 1.5, X)
            X = np.where(np.isneginf(X), self.min_vals_ * 1.5, X)

        # Handle NaN for fitting
        if self.handle_nan:
            X = np.nan_to_num(X, nan=self.fill_value)

        # Fit winsorization bounds
        if self.winsorize:
            self.lower_bounds_ = np.percentile(
                X, self.winsorize_limits[0] * 100, axis=0
            )
            self.upper_bounds_ = np.percentile(
                X, self.winsorize_limits[1] * 100, axis=0
            )
            # Apply winsorization for scaler fitting
            X = np.clip(X, self.lower_bounds_, self.upper_bounds_)

        # Fit scaler
        if self.scaler_type == "robust":
            self.center_ = np.median(X, axis=0)
            q75 = np.percentile(X, 75, axis=0)
            q25 = np.percentile(X, 25, axis=0)
            self.scale_ = q75 - q25
            self.scale_ = np.where(self.scale_ == 0, 1, self.scale_)
        elif self.scaler_type == "standard":
            self.center_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0)
            self.scale_ = np.where(self.scale_ == 0, 1, self.scale_)
        elif self.scaler_type == "minmax":
            self.center_ = np.min(X, axis=0)
            self.scale_ = np.max(X, axis=0) - np.min(X, axis=0)
            self.scale_ = np.where(self.scale_ == 0, 1, self.scale_)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters."""
        X = np.asarray(X).copy()

        if self.n_features_ is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X.shape[1]}"
            )

        # Handle infinity
        if self.handle_inf and self.max_vals_ is not None:
            X = np.where(np.isposinf(X), self.max_vals_ * 1.5, X)
            X = np.where(np.isneginf(X), self.min_vals_ * 1.5, X)

        # Handle NaN
        if self.handle_nan:
            X = np.nan_to_num(X, nan=self.fill_value)

        # Apply winsorization
        if self.winsorize and self.lower_bounds_ is not None:
            X = np.clip(X, self.lower_bounds_, self.upper_bounds_)

        # Apply scaling
        if self.center_ is not None and self.scale_ is not None:
            if self.scaler_type == "minmax":
                X = (X - self.center_) / self.scale_
            else:
                X = (X - self.center_) / self.scale_

        return X

    def get_params_dict(self) -> Dict[str, Any]:
        """Get fitted parameters as dictionary."""
        return {
            'lower_bounds': self.lower_bounds_.tolist() if self.lower_bounds_ is not None else None,
            'upper_bounds': self.upper_bounds_.tolist() if self.upper_bounds_ is not None else None,
            'center': self.center_.tolist() if self.center_ is not None else None,
            'scale': self.scale_.tolist() if self.scale_ is not None else None,
            'n_features': self.n_features_,
        }


def create_financial_pipeline(
    model: Any,
    scaler_type: str = "robust",
    winsorize: bool = True,
    winsorize_limits: Tuple[float, float] = (0.01, 0.99),
) -> Pipeline:
    """
    Convenience function to create a financial data pipeline.

    Args:
        model: ML model to use
        scaler_type: Type of scaler ("robust", "standard", "minmax")
        winsorize: Whether to apply winsorization
        winsorize_limits: Percentile limits for winsorization

    Returns:
        Sklearn Pipeline ready for training

    Example:
        from lightgbm import LGBMClassifier

        pipeline = create_financial_pipeline(
            model=LGBMClassifier(n_estimators=100),
            scaler_type="robust",
            winsorize=True
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
    """
    config = PipelineConfig(
        scaler_type=ScalerType(scaler_type),
        imputer_type=ImputerType.CONSTANT,
        winsorize=winsorize,
        winsorize_limits=winsorize_limits,
    )

    factory = ModelPipelineFactory(config)
    return factory.create_pipeline(model, config)
