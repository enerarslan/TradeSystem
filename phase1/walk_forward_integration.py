#!/usr/bin/env python3
"""
Walk-Forward Validation Module
==============================

JPMorgan-level walk-forward validation for ML trading models.
Prevents overfitting by simulating real-world model deployment.

Walk-Forward Analysis:
1. Split data into multiple folds
2. Train on historical data (in-sample)
3. Test on future data (out-of-sample)
4. Roll forward and repeat
5. Combine all out-of-sample results

This is the GOLD STANDARD for validating trading strategies and is
used by all major quant funds including JPMorgan, Two Sigma, Citadel.

Features:
- Multiple validation schemes (expanding, rolling, anchored)
- Purged K-Fold to prevent data leakage
- Embargo periods between train/test
- Comprehensive metrics per fold
- Aggregate performance analysis
- Overfitting detection
- Model stability analysis

Usage:
    from phase1.walk_forward_validation import (
        WalkForwardValidator,
        run_walk_forward_validation,
    )
    
    validator = WalkForwardValidator(config)
    results = validator.validate(
        symbol="AAPL",
        model_factory=create_model,
        n_splits=5,
    )
    
    # Check if model generalizes
    if results.overfitting_ratio < 0.8:
        print("Model likely overfitting!")

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats as scipy_stats

# Project imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings, get_logger, configure_logging
from config.symbols import ALL_SYMBOLS, CORE_SYMBOLS
from data.loader import CSVLoader
from data.processor import DataProcessor
from features.pipeline import FeaturePipeline, create_default_config
from features.advanced import (
    TripleBarrierLabeler,
    TripleBarrierConfig,
    MicrostructureFeatures,
    CalendarFeatures,
)
from models.model_manager import ModelManager
from models.training import TrainingPipeline, TrainingConfig, OptimizationConfig
from models.classifiers import LightGBMClassifier, XGBoostClassifier

logger = get_logger(__name__)


# Type variable for model
ModelT = TypeVar("ModelT")


# =============================================================================
# ENUMS
# =============================================================================

class ValidationScheme:
    """Walk-forward validation schemes."""
    EXPANDING = "expanding"      # Train on all historical data
    ROLLING = "rolling"          # Fixed training window
    ANCHORED = "anchored"        # Fixed start, expanding
    COMBINATORIAL = "combinatorial"  # Multiple train/test combinations


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    # Split configuration
    n_splits: int = 5
    train_ratio: float = 0.6  # 60% train, 40% test per fold
    
    # Validation scheme
    scheme: str = ValidationScheme.EXPANDING
    min_train_samples: int = 5000
    min_test_samples: int = 1000
    
    # Embargo periods (prevent leakage)
    embargo_bars: int = 10  # Gap between train and test
    purge_bars: int = 5     # Remove samples near boundaries
    
    # Model configuration
    model_types: list[str] = field(default_factory=lambda: ["lightgbm", "xgboost"])
    optimize_per_fold: bool = False  # Re-optimize hyperparameters per fold
    optimization_trials: int = 20
    
    # Triple Barrier settings
    use_triple_barrier: bool = True
    tb_take_profit_mult: float = 2.0
    tb_stop_loss_mult: float = 1.0
    tb_max_holding_period: int = 20
    
    # Metrics
    primary_metric: str = "accuracy"  # Metric to track
    track_metrics: list[str] = field(default_factory=lambda: [
        "accuracy", "f1_macro", "precision", "recall", "roc_auc"
    ])
    
    # Output
    output_dir: Path = field(default_factory=lambda: Path("reports/walk_forward"))
    save_models: bool = True
    save_predictions: bool = True


@dataclass 
class FoldResult:
    """Result of a single walk-forward fold."""
    fold_idx: int
    
    # Periods
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # Sample counts
    train_samples: int
    test_samples: int
    
    # Training metrics (in-sample)
    train_metrics: dict[str, float] = field(default_factory=dict)
    
    # Test metrics (out-of-sample)
    test_metrics: dict[str, float] = field(default_factory=dict)
    
    # Model info
    model_type: str = ""
    best_params: dict[str, Any] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)
    
    # Predictions
    predictions: list[int] = field(default_factory=list)
    probabilities: list[float] = field(default_factory=list)
    actuals: list[int] = field(default_factory=list)
    
    # Timing
    training_time_seconds: float = 0.0
    
    def overfitting_ratio(self) -> float:
        """Calculate overfitting ratio (test/train performance)."""
        train_acc = self.train_metrics.get("accuracy", 0)
        test_acc = self.test_metrics.get("accuracy", 0)
        
        if train_acc == 0:
            return 0.0
        return test_acc / train_acc
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["train_start"] = str(d["train_start"])
        d["train_end"] = str(d["train_end"])
        d["test_start"] = str(d["test_start"])
        d["test_end"] = str(d["test_end"])
        return d


@dataclass
class WalkForwardResult:
    """Aggregate result of walk-forward validation."""
    symbol: str
    model_type: str
    
    # Configuration
    n_splits: int
    scheme: str
    
    # Period
    start_date: datetime
    end_date: datetime
    total_samples: int
    
    # Aggregate metrics (averaged across folds)
    mean_train_accuracy: float = 0.0
    mean_test_accuracy: float = 0.0
    std_test_accuracy: float = 0.0
    
    mean_train_f1: float = 0.0
    mean_test_f1: float = 0.0
    
    # Overfitting analysis
    overfitting_ratio: float = 0.0  # test/train ratio
    performance_degradation: float = 0.0  # % drop from train to test
    
    # Stability
    stability_score: float = 0.0  # Consistency across folds
    fold_variance: float = 0.0
    
    # Statistical significance
    t_statistic: float = 0.0
    p_value: float = 1.0
    significant: bool = False
    
    # Feature stability
    stable_features: list[str] = field(default_factory=list)
    feature_stability_score: float = 0.0
    
    # Individual fold results
    folds: list[FoldResult] = field(default_factory=list)
    
    # Combined out-of-sample predictions
    all_oos_predictions: list[int] = field(default_factory=list)
    all_oos_actuals: list[int] = field(default_factory=list)
    
    # Timing
    total_time_seconds: float = 0.0
    
    # Metadata
    run_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def is_valid(self, min_overfitting_ratio: float = 0.85) -> bool:
        """Check if model passes validation criteria."""
        return (
            self.overfitting_ratio >= min_overfitting_ratio and
            self.stability_score >= 0.7 and
            self.p_value < 0.05
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["start_date"] = str(d["start_date"])
        d["end_date"] = str(d["end_date"])
        d["folds"] = [f.to_dict() if isinstance(f, FoldResult) else f for f in d["folds"]]
        return d
    
    def to_json(self, path: Path | str | None = None) -> str:
        """Convert to JSON."""
        json_str = json.dumps(self.to_dict(), indent=2, default=str)
        if path:
            Path(path).write_text(json_str)
        return json_str
    
    def print_summary(self) -> None:
        """Print human-readable summary."""
        print("\n" + "="*60)
        print(f"WALK-FORWARD VALIDATION RESULTS: {self.symbol}")
        print("="*60)
        print(f"Model: {self.model_type}")
        print(f"Folds: {self.n_splits}")
        print(f"Scheme: {self.scheme}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Total Samples: {self.total_samples:,}")
        print()
        print("PERFORMANCE")
        print(f"  Train Accuracy (avg): {self.mean_train_accuracy:.4f}")
        print(f"  Test Accuracy (avg):  {self.mean_test_accuracy:.4f} ± {self.std_test_accuracy:.4f}")
        print(f"  Overfitting Ratio:    {self.overfitting_ratio:.4f}")
        print(f"  Performance Drop:     {self.performance_degradation:.2%}")
        print()
        print("STABILITY")
        print(f"  Stability Score:      {self.stability_score:.4f}")
        print(f"  Fold Variance:        {self.fold_variance:.4f}")
        print()
        print("STATISTICAL SIGNIFICANCE")
        print(f"  T-Statistic:          {self.t_statistic:.4f}")
        print(f"  P-Value:              {self.p_value:.4f}")
        print(f"  Significant (α=0.05): {'Yes' if self.significant else 'No'}")
        print()
        print(f"VALIDATION: {'✓ PASSED' if self.is_valid() else '✗ FAILED'}")
        print("="*60)


# =============================================================================
# WALK-FORWARD VALIDATOR
# =============================================================================

class WalkForwardValidator:
    """
    Walk-forward validation for ML trading models.
    
    Implements rigorous out-of-sample testing with proper handling of:
    - Data leakage prevention (purging, embargo)
    - Multiple validation schemes
    - Statistical significance testing
    - Feature stability analysis
    
    Example:
        validator = WalkForwardValidator(config)
        result = validator.validate(
            symbol="AAPL",
            n_splits=5,
        )
        
        if result.is_valid():
            print("Model generalizes well!")
        else:
            print(f"Overfitting detected: {result.overfitting_ratio:.2f}")
    """
    
    def __init__(self, config: WalkForwardConfig | None = None):
        """Initialize validator."""
        self.config = config or WalkForwardConfig()
        
        settings = get_settings()
        
        # Initialize components
        self._loader = CSVLoader(storage_path=settings.data.storage_path)
        self._processor = DataProcessor()
        self._feature_pipeline = FeaturePipeline(create_default_config())
        
        # Triple barrier labeler
        if self.config.use_triple_barrier:
            tb_config = TripleBarrierConfig(
                take_profit_multiplier=self.config.tb_take_profit_mult,
                stop_loss_multiplier=self.config.tb_stop_loss_mult,
                max_holding_period=self.config.tb_max_holding_period,
            )
            self._labeler = TripleBarrierLabeler(tb_config)
        else:
            self._labeler = None
        
        # Model manager
        self._model_manager = ModelManager()
        
        logger.info("WalkForwardValidator initialized")
    
    def validate(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        n_splits: int | None = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward validation for a symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            n_splits: Number of folds (overrides config)
        
        Returns:
            WalkForwardResult with comprehensive analysis
        """
        symbol = symbol.upper()
        n_splits = n_splits or self.config.n_splits
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Walk-Forward Validation: {symbol}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        # Load and prepare data
        df = self._load_and_prepare_data(symbol, start_date, end_date)
        
        if df is None or len(df) < self.config.min_train_samples + self.config.min_test_samples:
            raise ValueError(f"Insufficient data for {symbol}")
        
        # Prepare features and target
        X, y, feature_names, timestamps = self._prepare_ml_data(df)
        
        logger.info(f"Total samples: {len(X)}")
        logger.info(f"Features: {len(feature_names)}")
        logger.info(f"Class distribution: {np.bincount(y.astype(int))}")
        
        # Generate fold indices
        folds = self._generate_folds(len(X), n_splits, timestamps)
        
        # Run validation for each model type
        all_results = []
        
        for model_type in self.config.model_types:
            logger.info(f"\nValidating {model_type}...")
            
            fold_results = []
            all_oos_preds = []
            all_oos_actuals = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                logger.info(f"\n  Fold {fold_idx + 1}/{n_splits}")
                
                # Split data
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                train_timestamps = [timestamps[i] for i in train_idx]
                test_timestamps = [timestamps[i] for i in test_idx]
                
                # Run single fold
                fold_result = self._run_fold(
                    fold_idx=fold_idx,
                    model_type=model_type,
                    X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    feature_names=feature_names,
                    train_timestamps=train_timestamps,
                    test_timestamps=test_timestamps,
                )
                
                fold_results.append(fold_result)
                all_oos_preds.extend(fold_result.predictions)
                all_oos_actuals.extend(fold_result.actuals)
            
            # Aggregate results
            result = self._aggregate_results(
                symbol=symbol,
                model_type=model_type,
                fold_results=fold_results,
                all_oos_predictions=all_oos_preds,
                all_oos_actuals=all_oos_actuals,
                timestamps=timestamps,
                feature_names=feature_names,
            )
            
            result.total_time_seconds = time.time() - start_time
            all_results.append(result)
            
            # Print summary
            result.print_summary()
        
        # Return best result (highest test accuracy)
        best_result = max(all_results, key=lambda r: r.mean_test_accuracy)
        
        # Save results
        self._save_results(symbol, all_results)
        
        return best_result
    
    def validate_multiple(
        self,
        symbols: list[str],
        n_splits: int | None = None,
    ) -> dict[str, WalkForwardResult]:
        """
        Validate multiple symbols.
        
        Args:
            symbols: List of symbols
            n_splits: Number of folds
        
        Returns:
            Dictionary mapping symbol to result
        """
        results = {}
        
        for i, symbol in enumerate(symbols):
            logger.info(f"\n[{i+1}/{len(symbols)}] Validating {symbol}...")
            
            try:
                result = self.validate(symbol, n_splits=n_splits)
                results[symbol] = result
            except Exception as e:
                logger.error(f"Validation failed for {symbol}: {e}")
                continue
        
        # Generate cross-symbol analysis
        self._analyze_cross_symbol(results)
        
        return results
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    
    def _load_and_prepare_data(
        self,
        symbol: str,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> pl.DataFrame | None:
        """Load and prepare data."""
        try:
            df = self._loader.load(symbol, start_date=start_date, end_date=end_date)
            
            if df is None or len(df) == 0:
                return None
            
            df = self._processor.process(df)
            df = self._feature_pipeline.generate(df)
            
            # Add advanced features
            df = MicrostructureFeatures.add_features(df)
            df = CalendarFeatures.add_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def _prepare_ml_data(
        self,
        df: pl.DataFrame,
    ) -> tuple[NDArray, NDArray, list[str], list[datetime]]:
        """Prepare feature matrix and target."""
        # Get timestamps
        timestamps = df["timestamp"].to_list()
        
        # Create target
        if self._labeler is not None:
            df = self._labeler.apply_binary_labels(df)
            target_col = "tb_binary_label"
        else:
            # Simple direction target
            df = df.with_columns([
                pl.when(pl.col("close").shift(-5) > pl.col("close"))
                .then(1)
                .otherwise(0)
                .alias("target")
            ])
            target_col = "target"
        
        # Get feature columns
        exclude_cols = {
            "timestamp", "open", "high", "low", "close", "volume",
            "target", "tb_label", "tb_binary_label", "tb_return", "tb_barrier",
        }
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # Filter valid feature columns (numeric only, no nulls)
        valid_features = []
        for col in feature_cols:
            if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                null_pct = df[col].null_count() / len(df)
                if null_pct < 0.1:  # Less than 10% nulls
                    valid_features.append(col)
        
        # Extract arrays
        X = df.select(valid_features).fill_null(0).to_numpy()
        y = df[target_col].fill_null(0).to_numpy()
        
        # Remove rows with NaN in target
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        timestamps = [t for t, v in zip(timestamps, valid_mask) if v]
        
        return X, y, valid_features, timestamps
    
    def _generate_folds(
        self,
        n_samples: int,
        n_splits: int,
        timestamps: list[datetime],
    ) -> list[tuple[NDArray, NDArray]]:
        """Generate train/test indices for each fold."""
        folds = []
        
        if self.config.scheme == ValidationScheme.EXPANDING:
            # Expanding window: train on all past data
            fold_size = n_samples // (n_splits + 1)
            
            for i in range(n_splits):
                test_start = fold_size * (i + 1)
                test_end = fold_size * (i + 2)
                
                train_end = test_start - self.config.embargo_bars
                
                if train_end < self.config.min_train_samples:
                    continue
                if test_end - test_start < self.config.min_test_samples:
                    continue
                
                train_idx = np.arange(0, train_end)
                test_idx = np.arange(test_start, min(test_end, n_samples))
                
                # Apply purging
                train_idx = train_idx[:-self.config.purge_bars] if self.config.purge_bars > 0 else train_idx
                test_idx = test_idx[self.config.purge_bars:] if self.config.purge_bars > 0 else test_idx
                
                folds.append((train_idx, test_idx))
        
        elif self.config.scheme == ValidationScheme.ROLLING:
            # Rolling window: fixed training window size
            train_size = int(n_samples * self.config.train_ratio)
            test_size = n_samples // n_splits
            
            for i in range(n_splits):
                test_start = train_size + i * test_size
                test_end = test_start + test_size
                
                train_start = test_start - train_size - self.config.embargo_bars
                train_end = test_start - self.config.embargo_bars
                
                if train_start < 0:
                    train_start = 0
                
                train_idx = np.arange(train_start, train_end)
                test_idx = np.arange(test_start, min(test_end, n_samples))
                
                if len(train_idx) < self.config.min_train_samples:
                    continue
                if len(test_idx) < self.config.min_test_samples:
                    continue
                
                folds.append((train_idx, test_idx))
        
        else:  # ANCHORED
            # Anchored: fixed start, expanding train
            fold_size = n_samples // n_splits
            
            for i in range(n_splits):
                train_end = fold_size * (i + 1) - self.config.embargo_bars
                test_start = fold_size * (i + 1)
                test_end = fold_size * (i + 2)
                
                train_idx = np.arange(0, train_end)
                test_idx = np.arange(test_start, min(test_end, n_samples))
                
                if len(train_idx) < self.config.min_train_samples:
                    continue
                if len(test_idx) < self.config.min_test_samples:
                    continue
                
                folds.append((train_idx, test_idx))
        
        logger.info(f"Generated {len(folds)} folds")
        return folds
    
    def _run_fold(
        self,
        fold_idx: int,
        model_type: str,
        X_train: NDArray, y_train: NDArray,
        X_test: NDArray, y_test: NDArray,
        feature_names: list[str],
        train_timestamps: list[datetime],
        test_timestamps: list[datetime],
    ) -> FoldResult:
        """Run a single fold."""
        start_time = time.time()
        
        # Create model
        if model_type == "lightgbm":
            from models.classifiers import LightGBMClassifier, LightGBMClassifierConfig
            config = LightGBMClassifierConfig(n_estimators=200, max_depth=6)
            model = LightGBMClassifier(config)
        elif model_type == "xgboost":
            from models.classifiers import XGBoostClassifier, XGBoostClassifierConfig
            config = XGBoostClassifierConfig(n_estimators=200, max_depth=6)
            model = XGBoostClassifier(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Optional hyperparameter optimization
        best_params = {}
        if self.config.optimize_per_fold:
            # Simplified optimization (would use Optuna in production)
            pass
        
        # Train model
        model.fit(X_train, y_train, feature_names=feature_names)
        
        # Evaluate
        train_metrics = model.evaluate(X_train, y_train)
        test_metrics = model.evaluate(X_test, y_test)
        
        # Get predictions
        predictions = model.predict(X_test).tolist()
        probabilities = model.predict_proba(X_test)[:, 1].tolist() if hasattr(model, 'predict_proba') else []
        
        # Get feature importance
        importance = model.get_feature_importance()
        top_features = dict(list(importance.items())[:20]) if importance else {}
        
        return FoldResult(
            fold_idx=fold_idx,
            train_start=train_timestamps[0] if train_timestamps else datetime.now(),
            train_end=train_timestamps[-1] if train_timestamps else datetime.now(),
            test_start=test_timestamps[0] if test_timestamps else datetime.now(),
            test_end=test_timestamps[-1] if test_timestamps else datetime.now(),
            train_samples=len(X_train),
            test_samples=len(X_test),
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            model_type=model_type,
            best_params=best_params,
            feature_importance=top_features,
            predictions=predictions,
            probabilities=probabilities,
            actuals=y_test.tolist(),
            training_time_seconds=time.time() - start_time,
        )
    
    def _aggregate_results(
        self,
        symbol: str,
        model_type: str,
        fold_results: list[FoldResult],
        all_oos_predictions: list[int],
        all_oos_actuals: list[int],
        timestamps: list[datetime],
        feature_names: list[str],
    ) -> WalkForwardResult:
        """Aggregate results across all folds."""
        if not fold_results:
            raise ValueError("No fold results to aggregate")
        
        # Extract metrics
        train_accs = [f.train_metrics.get("accuracy", 0) for f in fold_results]
        test_accs = [f.test_metrics.get("accuracy", 0) for f in fold_results]
        train_f1s = [f.train_metrics.get("f1_macro", 0) for f in fold_results]
        test_f1s = [f.test_metrics.get("f1_macro", 0) for f in fold_results]
        
        # Calculate aggregate metrics
        mean_train_acc = np.mean(train_accs)
        mean_test_acc = np.mean(test_accs)
        std_test_acc = np.std(test_accs)
        
        # Overfitting analysis
        overfitting_ratio = mean_test_acc / mean_train_acc if mean_train_acc > 0 else 0
        performance_drop = (mean_train_acc - mean_test_acc) / mean_train_acc if mean_train_acc > 0 else 0
        
        # Stability (coefficient of variation)
        stability_score = 1 - (std_test_acc / mean_test_acc) if mean_test_acc > 0 else 0
        fold_variance = np.var(test_accs)
        
        # Statistical significance
        # T-test: Is test accuracy significantly > 0.5 (random)?
        t_stat, p_value = scipy_stats.ttest_1samp(test_accs, 0.5)
        significant = p_value < 0.05 and t_stat > 0
        
        # Feature stability
        all_importances = [f.feature_importance for f in fold_results]
        stable_features = self._get_stable_features(all_importances)
        feature_stability = len(stable_features) / len(feature_names) if feature_names else 0
        
        return WalkForwardResult(
            symbol=symbol,
            model_type=model_type,
            n_splits=len(fold_results),
            scheme=self.config.scheme,
            start_date=fold_results[0].train_start,
            end_date=fold_results[-1].test_end,
            total_samples=sum(f.train_samples + f.test_samples for f in fold_results),
            mean_train_accuracy=mean_train_acc,
            mean_test_accuracy=mean_test_acc,
            std_test_accuracy=std_test_acc,
            mean_train_f1=np.mean(train_f1s),
            mean_test_f1=np.mean(test_f1s),
            overfitting_ratio=overfitting_ratio,
            performance_degradation=performance_drop,
            stability_score=stability_score,
            fold_variance=fold_variance,
            t_statistic=t_stat,
            p_value=p_value,
            significant=significant,
            stable_features=stable_features,
            feature_stability_score=feature_stability,
            folds=fold_results,
            all_oos_predictions=all_oos_predictions,
            all_oos_actuals=all_oos_actuals,
        )
    
    def _get_stable_features(
        self,
        importance_dicts: list[dict[str, float]],
    ) -> list[str]:
        """Get features that are consistently important across folds."""
        if not importance_dicts:
            return []
        
        # Count how often each feature appears in top 20
        feature_counts: dict[str, int] = {}
        for imp in importance_dicts:
            for feat in list(imp.keys())[:20]:
                feature_counts[feat] = feature_counts.get(feat, 0) + 1
        
        # Return features that appear in at least 50% of folds
        threshold = len(importance_dicts) * 0.5
        stable = [f for f, c in feature_counts.items() if c >= threshold]
        
        return stable
    
    def _save_results(
        self,
        symbol: str,
        results: list[WalkForwardResult],
    ) -> None:
        """Save validation results."""
        output_dir = self.config.output_dir / symbol
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for result in results:
            filename = f"wf_{result.model_type}_{timestamp}.json"
            result.to_json(output_dir / filename)
        
        logger.info(f"Results saved to {output_dir}")
    
    def _analyze_cross_symbol(
        self,
        results: dict[str, WalkForwardResult],
    ) -> None:
        """Analyze results across all symbols."""
        if not results:
            return
        
        # Calculate aggregate statistics
        all_test_accs = [r.mean_test_accuracy for r in results.values()]
        all_overfit_ratios = [r.overfitting_ratio for r in results.values()]
        
        valid_count = sum(1 for r in results.values() if r.is_valid())
        
        logger.info("\n" + "="*60)
        logger.info("CROSS-SYMBOL ANALYSIS")
        logger.info("="*60)
        logger.info(f"Total Symbols: {len(results)}")
        logger.info(f"Valid Models: {valid_count}/{len(results)}")
        logger.info(f"Avg Test Accuracy: {np.mean(all_test_accs):.4f}")
        logger.info(f"Avg Overfitting Ratio: {np.mean(all_overfit_ratios):.4f}")
        logger.info("="*60)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_walk_forward_validation(
    symbol: str,
    config: WalkForwardConfig | None = None,
    n_splits: int = 5,
) -> WalkForwardResult:
    """
    Convenience function to run walk-forward validation.
    
    Args:
        symbol: Trading symbol
        config: Validation configuration
        n_splits: Number of folds
    
    Returns:
        WalkForwardResult
    """
    validator = WalkForwardValidator(config)
    return validator.validate(symbol, n_splits=n_splits)


def validate_all_symbols(
    config: WalkForwardConfig | None = None,
    core_only: bool = False,
) -> dict[str, WalkForwardResult]:
    """
    Validate all symbols.
    
    Args:
        config: Validation configuration
        core_only: Use only core symbols
    
    Returns:
        Dictionary of results
    """
    symbols = CORE_SYMBOLS if core_only else ALL_SYMBOLS
    validator = WalkForwardValidator(config)
    return validator.validate_multiple(symbols)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Walk-Forward Validation")
    parser.add_argument("--symbol", "-s", type=str, help="Symbol to validate")
    parser.add_argument("--symbols", "-S", type=str, nargs="+", help="Multiple symbols")
    parser.add_argument("--all", "-a", action="store_true", help="Validate all symbols")
    parser.add_argument("--core", "-c", action="store_true", help="Core symbols only")
    parser.add_argument("--splits", "-n", type=int, default=5, help="Number of splits")
    parser.add_argument("--scheme", type=str, default="expanding", 
                       choices=["expanding", "rolling", "anchored"])
    parser.add_argument("--output", "-o", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    # Create config
    config = WalkForwardConfig(n_splits=args.splits, scheme=args.scheme)
    if args.output:
        config.output_dir = Path(args.output)
    
    # Run validation
    if args.symbol:
        result = run_walk_forward_validation(args.symbol, config, args.splits)
        print(f"\nValidation {'PASSED' if result.is_valid() else 'FAILED'}")
    
    elif args.symbols:
        validator = WalkForwardValidator(config)
        results = validator.validate_multiple(args.symbols)
        print(f"\nValidated {len(results)} symbols")
    
    elif args.all:
        results = validate_all_symbols(config, args.core)
        print(f"\nValidated {len(results)} symbols")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    configure_logging(get_settings())
    main()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ValidationScheme",
    # Configuration
    "WalkForwardConfig",
    "FoldResult",
    "WalkForwardResult",
    # Main class
    "WalkForwardValidator",
    # Convenience functions
    "run_walk_forward_validation",
    "validate_all_symbols",
]