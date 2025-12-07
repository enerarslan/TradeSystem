"""
Feature Pipeline Module
=======================

Feature engineering orchestration for the algorithmic trading platform.
Provides a unified interface for generating, selecting, and managing features.

Capabilities:
- Configurable feature generation
- Feature selection methods
- Feature importance analysis
- Feature storage and caching
- Train/test feature alignment

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl
from numpy.typing import NDArray

from config.settings import get_logger
from features.technical import (
    TechnicalIndicators,
    IndicatorConfig,
    DEFAULT_CONFIG as DEFAULT_TECH_CONFIG,
    add_all_indicators,
)
from features.statistical import (
    StatisticalFeatures,
    StatisticalConfig,
    DEFAULT_STAT_CONFIG,
    add_statistical_features,
)


logger = get_logger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class FeatureCategory(str, Enum):
    """Feature categories."""
    TECHNICAL_MOMENTUM = "technical_momentum"
    TECHNICAL_TREND = "technical_trend"
    TECHNICAL_VOLATILITY = "technical_volatility"
    TECHNICAL_VOLUME = "technical_volume"
    STATISTICAL_RETURNS = "statistical_returns"
    STATISTICAL_ROLLING = "statistical_rolling"
    STATISTICAL_MOMENTUM = "statistical_momentum"
    STATISTICAL_VOLATILITY = "statistical_volatility"
    STATISTICAL_REGIME = "statistical_regime"
    STATISTICAL_DISTRIBUTION = "statistical_distribution"
    STATISTICAL_MEAN_REVERSION = "statistical_mean_reversion"
    CUSTOM = "custom"


@dataclass
class FeatureConfig:
    """
    Configuration for feature generation.
    
    Attributes:
        technical_config: Technical indicator configuration
        statistical_config: Statistical feature configuration
        enabled_categories: List of enabled feature categories
        custom_features: Custom feature functions
        drop_na: Drop rows with NaN values
        fill_method: Method for filling NaN values
        max_lookback: Maximum lookback period (for alignment)
        cache_features: Enable feature caching
        feature_prefix: Prefix for generated feature names
    """
    technical_config: IndicatorConfig = field(default_factory=lambda: DEFAULT_TECH_CONFIG)
    statistical_config: StatisticalConfig = field(default_factory=lambda: DEFAULT_STAT_CONFIG)
    enabled_categories: list[FeatureCategory] = field(default_factory=lambda: list(FeatureCategory))
    custom_features: dict[str, Callable[[pl.DataFrame], pl.DataFrame]] = field(default_factory=dict)
    drop_na: bool = False
    fill_method: str | None = "forward"
    normalize: bool = False
    max_lookback: int = 252
    cache_features: bool = True
    feature_prefix: str = ""


def create_default_config() -> FeatureConfig:
    """Create default feature configuration."""
    return FeatureConfig()


# =============================================================================
# FEATURE PIPELINE
# =============================================================================

class FeaturePipeline:
    """
    Main feature engineering pipeline.
    
    Orchestrates the generation of technical and statistical features,
    handles missing values, and provides feature management utilities.
    
    Example:
        config = FeatureConfig()
        pipeline = FeaturePipeline(config)
        
        # Generate features
        df_features = pipeline.generate(df)
        
        # Get feature matrix
        X, feature_names = pipeline.get_feature_matrix(df_features)
        
        # Get feature importance
        importance = pipeline.get_feature_importance(model)
    """
    
    def __init__(self, config: FeatureConfig | None = None):
        """
        Initialize the feature pipeline.
        
        Args:
            config: Feature configuration
        """
        self.config = config or create_default_config()
        
        # Initialize generators
        self._technical = TechnicalIndicators(self.config.technical_config)
        self._statistical = StatisticalFeatures(self.config.statistical_config)
        
        # Feature tracking
        self._feature_names: list[str] = []
        self._feature_stats: dict[str, dict[str, float]] = {}
        self._generation_time: float = 0.0
        
        # Cache
        self._cache: dict[str, pl.DataFrame] = {}
    
    @property
    def feature_names(self) -> list[str]:
        """Get list of generated feature names."""
        return self._feature_names.copy()
    
    @property
    def num_features(self) -> int:
        """Get number of features."""
        return len(self._feature_names)
    
    def generate(
        self,
        df: pl.DataFrame,
        categories: list[FeatureCategory] | None = None,
    ) -> pl.DataFrame:
        """
        Generate all configured features.
        
        Args:
            df: Input DataFrame with OHLCV data
            categories: Categories to generate (None = all enabled)
        
        Returns:
            DataFrame with all features added
        """
        import time
        start_time = time.time()
        
        categories = categories or self.config.enabled_categories
        original_columns = set(df.columns)
        
        logger.info(f"Generating features for {len(df)} rows, categories: {len(categories)}")
        
        # Technical features
        tech_categories = {
            FeatureCategory.TECHNICAL_MOMENTUM: "momentum",
            FeatureCategory.TECHNICAL_TREND: "trend",
            FeatureCategory.TECHNICAL_VOLATILITY: "volatility",
            FeatureCategory.TECHNICAL_VOLUME: "volume",
        }
        
        for cat, tech_cat in tech_categories.items():
            if cat in categories:
                logger.debug(f"Adding {tech_cat} indicators")
                if tech_cat == "momentum":
                    df = self._technical.add_momentum_indicators(df)
                elif tech_cat == "trend":
                    df = self._technical.add_trend_indicators(df)
                elif tech_cat == "volatility":
                    df = self._technical.add_volatility_indicators(df)
                elif tech_cat == "volume":
                    df = self._technical.add_volume_indicators(df)
        
        # Statistical features
        stat_categories = {
            FeatureCategory.STATISTICAL_RETURNS: "returns",
            FeatureCategory.STATISTICAL_ROLLING: "rolling",
            FeatureCategory.STATISTICAL_MOMENTUM: "momentum",
            FeatureCategory.STATISTICAL_VOLATILITY: "volatility",
            FeatureCategory.STATISTICAL_REGIME: "regime",
            FeatureCategory.STATISTICAL_DISTRIBUTION: "distribution",
            FeatureCategory.STATISTICAL_MEAN_REVERSION: "mean_reversion",
        }
        
        for cat, stat_cat in stat_categories.items():
            if cat in categories:
                logger.debug(f"Adding {stat_cat} features")
                if stat_cat == "returns":
                    df = self._statistical.add_return_features(df)
                elif stat_cat == "rolling":
                    df = self._statistical.add_rolling_features(df)
                elif stat_cat == "momentum":
                    df = self._statistical.add_momentum_features(df)
                elif stat_cat == "volatility":
                    df = self._statistical.add_volatility_features(df)
                elif stat_cat == "regime":
                    df = self._statistical.add_regime_features(df)
                elif stat_cat == "distribution":
                    df = self._statistical.add_distribution_features(df)
                elif stat_cat == "mean_reversion":
                    df = self._statistical.add_mean_reversion_features(df)
        
        # Custom features
        if FeatureCategory.CUSTOM in categories:
            for name, func in self.config.custom_features.items():
                logger.debug(f"Adding custom feature: {name}")
                df = func(df)
        
        # Track feature names
        new_columns = set(df.columns) - original_columns
        self._feature_names = sorted(new_columns)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Calculate feature statistics
        self._calculate_feature_stats(df)
        
        self._generation_time = time.time() - start_time
        logger.info(
            f"Generated {len(self._feature_names)} features in {self._generation_time:.2f}s"
        )
        
        return df
    
    def _handle_missing_values(self, df: pl.DataFrame) -> pl.DataFrame:
        """Handle missing values in features."""
        if self.config.drop_na:
            original_len = len(df)
            df = df.drop_nulls()
            dropped = original_len - len(df)
            if dropped > 0:
                logger.info(f"Dropped {dropped} rows with NaN values")
        
        elif self.config.fill_method == "forward":
            # Forward fill for numeric columns
            numeric_cols = [
                col for col in self._feature_names
                if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            ]
            df = df.with_columns([
                pl.col(col).forward_fill() for col in numeric_cols
            ])
        
        elif self.config.fill_method == "zero":
            numeric_cols = [
                col for col in self._feature_names
                if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            ]
            df = df.with_columns([
                pl.col(col).fill_null(0.0) for col in numeric_cols
            ])
        
        elif self.config.fill_method == "mean":
            numeric_cols = [
                col for col in self._feature_names
                if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            ]
            for col in numeric_cols:
                mean_val = df[col].mean()
                df = df.with_columns([pl.col(col).fill_null(mean_val)])
        
        return df
    
    def _calculate_feature_stats(self, df: pl.DataFrame) -> None:
        """Calculate statistics for each feature."""
        self._feature_stats = {}
        
        for col in self._feature_names:
            if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                series = df[col]
                self._feature_stats[col] = {
                    "mean": series.mean() or 0.0,
                    "std": series.std() or 0.0,
                    "min": series.min() or 0.0,
                    "max": series.max() or 0.0,
                    "null_count": series.null_count(),
                    "null_pct": series.null_count() / len(series) * 100,
                }
    
    def get_feature_matrix(
        self,
        df: pl.DataFrame,
        features: list[str] | None = None,
        as_numpy: bool = True,
    ) -> tuple[NDArray[np.float64] | pl.DataFrame, list[str]]:
        """
        Extract feature matrix from DataFrame.
        
        Args:
            df: DataFrame with features
            features: List of features to include (None = all)
            as_numpy: Return numpy array (vs DataFrame)
        
        Returns:
            Tuple of (feature matrix, feature names)
        """
        features = features or self._feature_names
        
        # Filter to numeric features only
        numeric_features = [
            f for f in features
            if f in df.columns and df[f].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]
        
        if as_numpy:
            X = df.select(numeric_features).to_numpy()
            return X, numeric_features
        else:
            return df.select(numeric_features), numeric_features
    
    def get_feature_stats(self) -> dict[str, dict[str, float]]:
        """Get feature statistics."""
        return self._feature_stats.copy()
    
    def get_feature_importance(
        self,
        model: Any,
        feature_names: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Get feature importance from a trained model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: Feature names (uses stored names if None)
        
        Returns:
            Dictionary of feature name to importance
        """
        feature_names = feature_names or self._feature_names
        
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).ravel()
        else:
            return {}
        
        if len(importances) != len(feature_names):
            logger.warning("Feature importance length mismatch")
            return {}
        
        return dict(sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        ))
    
    def select_features(
        self,
        df: pl.DataFrame,
        method: str = "variance",
        threshold: float = 0.01,
        top_k: int | None = None,
        model: Any | None = None,
    ) -> list[str]:
        """
        Select features based on various criteria.
        
        Methods:
            - variance: Remove low variance features
            - correlation: Remove highly correlated features
            - importance: Use model feature importance
        
        Args:
            df: DataFrame with features
            method: Selection method
            threshold: Selection threshold
            top_k: Select top k features (for importance method)
            model: Trained model (for importance method)
        
        Returns:
            List of selected feature names
        """
        if method == "variance":
            return self._select_by_variance(df, threshold)
        elif method == "correlation":
            return self._select_by_correlation(df, threshold)
        elif method == "importance":
            return self._select_by_importance(model, top_k or 50)
        else:
            return self._feature_names
    
    def _select_by_variance(
        self,
        df: pl.DataFrame,
        threshold: float,
    ) -> list[str]:
        """Select features with variance above threshold."""
        selected = []
        
        for col in self._feature_names:
            if col in df.columns and df[col].dtype in [pl.Float64, pl.Float32]:
                variance = df[col].var()
                if variance is not None and variance > threshold:
                    selected.append(col)
        
        logger.info(f"Variance selection: {len(selected)}/{len(self._feature_names)} features")
        return selected
    
    def _select_by_correlation(
        self,
        df: pl.DataFrame,
        threshold: float,
    ) -> list[str]:
        """Remove highly correlated features."""
        numeric_features = [
            f for f in self._feature_names
            if f in df.columns and df[f].dtype in [pl.Float64, pl.Float32]
        ]
        
        if not numeric_features:
            return []
        
        # Calculate correlation matrix
        X = df.select(numeric_features).to_numpy()
        corr_matrix = np.corrcoef(X.T)
        
        # Find highly correlated pairs
        to_remove = set()
        n_features = len(numeric_features)
        
        for i in range(n_features):
            if numeric_features[i] in to_remove:
                continue
            for j in range(i + 1, n_features):
                if numeric_features[j] in to_remove:
                    continue
                if abs(corr_matrix[i, j]) > threshold:
                    # Remove the feature with lower variance
                    var_i = np.nanvar(X[:, i])
                    var_j = np.nanvar(X[:, j])
                    if var_i < var_j:
                        to_remove.add(numeric_features[i])
                    else:
                        to_remove.add(numeric_features[j])
        
        selected = [f for f in numeric_features if f not in to_remove]
        logger.info(f"Correlation selection: {len(selected)}/{len(numeric_features)} features")
        return selected
    
    def _select_by_importance(
        self,
        model: Any,
        top_k: int,
    ) -> list[str]:
        """Select top k features by importance."""
        importance = self.get_feature_importance(model)
        
        if not importance:
            return self._feature_names
        
        selected = list(importance.keys())[:top_k]
        logger.info(f"Importance selection: top {top_k} features")
        return selected
    
    def create_target(
        self,
        df: pl.DataFrame,
        target_type: str = "return",
        horizon: int = 1,
        threshold: float | None = None,
        column: str = "close",
    ) -> pl.DataFrame:
        """
        Create target variable for supervised learning.
        
        Target Types:
            - return: Future return (regression)
            - direction: Future direction (classification)
            - threshold: Above/below threshold (classification)
        
        Args:
            df: DataFrame with price data
            target_type: Type of target variable
            horizon: Prediction horizon (bars forward)
            threshold: Threshold for classification targets
            column: Price column to use
        
        Returns:
            DataFrame with target column added
        """
        if target_type == "return":
            # Future log return
            df = df.with_columns([
                (pl.col(column).shift(-horizon) / pl.col(column)).log().alias("target"),
            ])
        
        elif target_type == "direction":
            # Future direction (1 = up, 0 = down)
            df = df.with_columns([
                (pl.col(column).shift(-horizon) > pl.col(column)).cast(pl.Int8).alias("target"),
            ])
        
        elif target_type == "threshold":
            # Above threshold return
            threshold = threshold or 0.0
            df = df.with_columns([
                (
                    (pl.col(column).shift(-horizon) / pl.col(column) - 1) > threshold
                ).cast(pl.Int8).alias("target"),
            ])
        
        elif target_type == "multi_class":
            # Multi-class: strong down, down, neutral, up, strong up
            threshold = threshold or 0.01
            future_return = (df[column].shift(-horizon) / df[column] - 1).to_numpy()
            
            classes = np.zeros(len(future_return), dtype=np.int8)
            classes[future_return < -2 * threshold] = 0  # Strong down
            classes[(future_return >= -2 * threshold) & (future_return < -threshold)] = 1  # Down
            classes[(future_return >= -threshold) & (future_return <= threshold)] = 2  # Neutral
            classes[(future_return > threshold) & (future_return <= 2 * threshold)] = 3  # Up
            classes[future_return > 2 * threshold] = 4  # Strong up
            
            df = df.with_columns([
                pl.Series("target", classes),
            ])
        
        return df
    
    def prepare_train_test(
        self,
        df: pl.DataFrame,
        test_size: float = 0.2,
        features: list[str] | None = None,
        target_col: str = "target",
        drop_na: bool = True,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, list[str]]:
        """
        Prepare train/test splits.
        
        Uses time-based split (no shuffling) to avoid look-ahead bias.
        
        Args:
            df: DataFrame with features and target
            test_size: Fraction for test set
            features: Features to include
            target_col: Target column name
            drop_na: Drop rows with NaN values
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names)
        """
        features = features or self._feature_names
        
        # Filter to valid features
        valid_features = [f for f in features if f in df.columns]
        
        # Drop rows with NaN in target
        if drop_na:
            df = df.drop_nulls(subset=[target_col] + valid_features)
        
        # Extract arrays
        X = df.select(valid_features).to_numpy()
        y = df[target_col].to_numpy()
        
        # Time-based split
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        logger.info(
            f"Train/test split: {len(X_train)} train, {len(X_test)} test samples"
        )
        
        return X_train, X_test, y_train, y_test, valid_features
    
    def prepare_walk_forward(
        self,
        df: pl.DataFrame,
        n_splits: int = 5,
        train_size: float = 0.6,
        features: list[str] | None = None,
        target_col: str = "target",
    ) -> list[tuple[NDArray, NDArray, NDArray, NDArray]]:
        """
        Prepare walk-forward validation splits.
        
        Creates expanding or rolling window splits for time series validation.
        
        Args:
            df: DataFrame with features and target
            n_splits: Number of splits
            train_size: Initial training size fraction
            features: Features to include
            target_col: Target column name
        
        Returns:
            List of (X_train, X_test, y_train, y_test) tuples
        """
        features = features or self._feature_names
        valid_features = [f for f in features if f in df.columns]
        
        df = df.drop_nulls(subset=[target_col] + valid_features)
        
        X = df.select(valid_features).to_numpy()
        y = df[target_col].to_numpy()
        
        n_samples = len(X)
        initial_train = int(n_samples * train_size)
        test_size = (n_samples - initial_train) // n_splits
        
        splits = []
        for i in range(n_splits):
            train_end = initial_train + i * test_size
            test_end = train_end + test_size
            
            X_train = X[:train_end]
            y_train = y[:train_end]
            X_test = X[train_end:test_end]
            y_test = y[train_end:test_end]
            
            splits.append((X_train, X_test, y_train, y_test))
        
        logger.info(f"Created {n_splits} walk-forward splits")
        return splits
    
    def save_features(
        self,
        df: pl.DataFrame,
        path: Path | str,
        features: list[str] | None = None,
    ) -> None:
        """
        Save features to parquet file.
        
        Args:
            df: DataFrame with features
            path: Output file path
            features: Features to save (None = all)
        """
        features = features or self._feature_names
        
        # Include timestamp and OHLCV columns
        base_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        all_cols = [c for c in base_cols if c in df.columns] + features
        
        df.select(all_cols).write_parquet(path)
        logger.info(f"Saved {len(features)} features to {path}")
    
    def load_features(
        self,
        path: Path | str,
    ) -> pl.DataFrame:
        """
        Load features from parquet file.
        
        Args:
            path: Input file path
        
        Returns:
            DataFrame with features
        """
        df = pl.read_parquet(path)
        
        # Update feature names
        base_cols = {"timestamp", "open", "high", "low", "close", "volume", "symbol"}
        self._feature_names = [c for c in df.columns if c not in base_cols]
        
        logger.info(f"Loaded {len(self._feature_names)} features from {path}")
        return df
    
    def get_summary(self) -> dict[str, Any]:
        """
        Get pipeline summary.
        
        Returns:
            Dictionary with pipeline statistics
        """
        return {
            "num_features": self.num_features,
            "feature_names": self._feature_names,
            "generation_time": self._generation_time,
            "enabled_categories": [c.value for c in self.config.enabled_categories],
            "feature_stats": self._feature_stats,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_default_pipeline() -> FeaturePipeline:
    """
    Create a feature pipeline with default configuration.
    
    Returns:
        Configured FeaturePipeline
    """
    return FeaturePipeline(create_default_config())


def generate_all_features(
    df: pl.DataFrame,
    config: FeatureConfig | None = None,
) -> pl.DataFrame:
    """
    Convenience function to generate all features.
    
    Args:
        df: Input DataFrame with OHLCV data
        config: Feature configuration
    
    Returns:
        DataFrame with all features
    """
    pipeline = FeaturePipeline(config)
    return pipeline.generate(df)


def create_ml_dataset(
    df: pl.DataFrame,
    target_type: str = "direction",
    horizon: int = 1,
    test_size: float = 0.2,
    config: FeatureConfig | None = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray, list[str]]:
    """
    Create ML-ready dataset from OHLCV data.
    
    Args:
        df: Input DataFrame with OHLCV data
        target_type: Type of target variable
        horizon: Prediction horizon
        test_size: Test set fraction
        config: Feature configuration
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    pipeline = FeaturePipeline(config)
    
    # Generate features
    df = pipeline.generate(df)
    
    # Create target
    df = pipeline.create_target(df, target_type=target_type, horizon=horizon)
    
    # Prepare train/test
    return pipeline.prepare_train_test(df, test_size=test_size)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "FeatureCategory",
    "FeatureConfig",
    "create_default_config",
    # Pipeline
    "FeaturePipeline",
    # Convenience functions
    "create_default_pipeline",
    "generate_all_features",
    "create_ml_dataset",
]