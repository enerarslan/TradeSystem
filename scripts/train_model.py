#!/usr/bin/env python3
"""
Train Model Script
==================

Comprehensive CLI for training machine learning models for algorithmic trading.

Features:
- Multiple model types (LightGBM, XGBoost, LSTM, Transformer)
- Hyperparameter optimization with Optuna
- Walk-forward validation
- Model comparison and selection
- Feature importance analysis
- Model persistence and versioning

Usage:
    python scripts/train_model.py --symbol AAPL --model lightgbm
    python scripts/train_model.py --symbols AAPL GOOGL MSFT --model xgboost --optimize
    python scripts/train_model.py --symbol AAPL --model lstm --epochs 100
    python scripts/train_model.py --all-symbols --compare-models

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl

from config.settings import get_settings, get_logger, configure_logging
from core.types import DataError


# =============================================================================
# INITIALIZATION
# =============================================================================

settings = get_settings()
configure_logging(settings)
logger = get_logger(__name__)


# =============================================================================
# DATA LOADING
# =============================================================================

def discover_symbols(data_path: Path) -> list[str]:
    """Discover available symbols from CSV files."""
    csv_files = list(data_path.glob("*_15min.csv")) + list(data_path.glob("*_1h.csv"))
    symbols = []
    for f in csv_files:
        symbol = f.stem.split("_")[0]
        if symbol not in symbols:
            symbols.append(symbol)
    return sorted(symbols)


def load_data(
    symbols: list[str],
    data_path: Path,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> dict[str, pl.DataFrame]:
    """Load and process data for symbols."""
    from data.processor import DataProcessor
    
    processor = DataProcessor()
    data = {}
    
    for symbol in symbols:
        patterns = [
            f"{symbol}_15min.csv",
            f"{symbol}_1h.csv",
            f"{symbol}.csv",
        ]
        
        loaded = False
        for pattern in patterns:
            file_path = data_path / pattern
            if file_path.exists():
                logger.info(f"Loading {symbol} from {file_path}")
                
                try:
                    df = pl.read_csv(file_path)
                    
                    if "timestamp" in df.columns:
                        df = df.with_columns([
                            pl.col("timestamp").str.to_datetime().alias("timestamp")
                        ])
                    
                    df = processor.process(df)
                    
                    if start_date:
                        df = df.filter(pl.col("timestamp") >= start_date)
                    if end_date:
                        df = df.filter(pl.col("timestamp") <= end_date)
                    
                    if len(df) > 0:
                        data[symbol] = df
                        logger.info(f"  Loaded {len(df)} bars for {symbol}")
                        loaded = True
                        break
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    continue
        
        if not loaded:
            logger.warning(f"Could not find data for {symbol}")
    
    return data


# =============================================================================
# FEATURE GENERATION
# =============================================================================

def generate_features(
    data: dict[str, pl.DataFrame],
    feature_config: dict[str, Any] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    """Generate features for all symbols."""
    from features.pipeline import (
        FeaturePipeline,
        FeatureConfig,
        FeatureCategory,
        create_default_config,
    )
    
    logger.info("Generating features...")
    
    # Default feature configuration
    if feature_config is None:
        config = create_default_config()
    else:
        config = FeatureConfig(**feature_config)
    
    pipeline = FeaturePipeline(config)
    results = {}
    
    for symbol, df in data.items():
        logger.info(f"  Processing {symbol}...")
        
        try:
            # Generate features
            df_features = pipeline.generate(df)
            
            # Create target (direction classification)
            df_features = pipeline.create_target(
                df_features,
                target_type="direction",
                horizon=5,  # Predict 5 bars ahead
            )
            
            # IMPORTANT: Filter out non-numeric columns before ML training
            # This removes categorical columns like 'regime' that contain strings
            numeric_types = [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8]
            
            # Get only numeric feature columns (exclude timestamp, symbol, target)
            exclude_cols = {"timestamp", "symbol", "target", "open", "high", "low", "close", "volume"}
            feature_cols = []
            
            for col in df_features.columns:
                if col not in exclude_cols:
                    col_dtype = df_features[col].dtype
                    # Check if numeric type
                    if col_dtype in numeric_types or col_dtype == pl.Float64 or "float" in str(col_dtype).lower() or "int" in str(col_dtype).lower():
                        # Additional check: ensure no string values
                        if col_dtype not in [pl.Utf8, pl.String, pl.Categorical]:
                            feature_cols.append(col)
            
            logger.info(f"    Filtered to {len(feature_cols)} numeric features")
            
            # Drop rows with null target
            df_clean = df_features.drop_nulls(subset=["target"])
            
            # Also drop nulls in feature columns
            df_clean = df_clean.drop_nulls(subset=feature_cols)
            
            # Extract features and target
            X = df_clean.select(feature_cols).to_numpy().astype(np.float64)
            y = df_clean["target"].to_numpy().astype(np.int64)
            
            # Handle any remaining NaN/Inf values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Train/test split (time-based, not random)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Combine for storage
            X_combined = np.vstack([X_train, X_test])
            y_combined = np.concatenate([y_train, y_test])
            
            results[symbol] = (X_combined, y_combined, feature_cols)
            
            logger.info(f"    Generated {len(feature_cols)} features, {len(X_combined)} samples")
            
        except Exception as e:
            logger.error(f"    Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_single_model(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    optimize: bool = True,
    n_trials: int = 50,
    model_params: dict[str, Any] | None = None,
) -> Any:
    """Train a single model."""
    from models.training import TrainingPipeline, TrainingConfig, OptimizationConfig
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Configure training
    opt_config = OptimizationConfig(
        n_trials=n_trials if optimize else 0,
        cv_splits=5,
        validation_metric="accuracy",
    )
    
    config = TrainingConfig(
        models_dir=Path("models/artifacts"),
        auto_optimize=optimize,
        optimization_config=opt_config,
    )
    
    pipeline = TrainingPipeline(config)
    
    # Train model
    model = pipeline.train(
        model_type=model_type,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        optimize=optimize,
        model_params=model_params,
    )
    
    return model, pipeline.training_results


def train_deep_model(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    epochs: int = 100,
    batch_size: int = 64,
    sequence_length: int = 60,
) -> Any:
    """Train a deep learning model."""
    from models.deep import create_deep_model
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create model
    model = create_deep_model(
        model_type,
        sequence_length=sequence_length,
        n_features=X.shape[1],
        epochs=epochs,
        batch_size=batch_size,
        num_classes=len(np.unique(y)),
    )
    
    # Train
    model.fit(
        X_train, y_train,
        feature_names=feature_names,
        X_val=X_test,
        y_val=y_test,
    )
    
    return model


def compare_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    model_types: list[str] | None = None,
    optimize: bool = True,
    n_trials: int = 30,
) -> tuple[Any, pl.DataFrame]:
    """Compare multiple model types and return the best one."""
    from models.training import TrainingPipeline, TrainingConfig, OptimizationConfig
    
    if model_types is None:
        model_types = ["lightgbm", "xgboost", "random_forest"]
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Configure
    opt_config = OptimizationConfig(
        n_trials=n_trials if optimize else 0,
        cv_splits=5,
    )
    
    config = TrainingConfig(
        models_dir=Path("models/artifacts"),
        auto_optimize=optimize,
        optimization_config=opt_config,
    )
    
    pipeline = TrainingPipeline(config)
    
    # Compare models
    comparison = pipeline.compare_models(
        model_types=model_types,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
    )
    
    return pipeline.best_model, comparison


# =============================================================================
# WALK-FORWARD TRAINING
# =============================================================================

def walk_forward_train(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_splits: int = 5,
    train_ratio: float = 0.6,
    optimize: bool = False,
) -> list[dict[str, Any]]:
    """Perform walk-forward training and validation."""
    from models.training import TrainingPipeline, TrainingConfig, OptimizationConfig
    from models.classifiers import create_classifier
    
    results = []
    n_samples = len(X)
    initial_train = int(n_samples * train_ratio)
    test_size = (n_samples - initial_train) // n_splits
    
    logger.info(f"Walk-forward training with {n_splits} splits")
    logger.info(f"  Initial training size: {initial_train}")
    logger.info(f"  Test size per split: {test_size}")
    
    for i in range(n_splits):
        train_end = initial_train + i * test_size
        test_end = train_end + test_size
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[train_end:test_end]
        y_test = y[train_end:test_end]
        
        logger.info(f"\nFold {i + 1}/{n_splits}:")
        logger.info(f"  Train: 0 to {train_end} ({len(X_train)} samples)")
        logger.info(f"  Test: {train_end} to {test_end} ({len(X_test)} samples)")
        
        # Train model
        model = create_classifier(model_type)
        model.fit(X_train, y_train, feature_names=feature_names)
        
        # Evaluate
        train_metrics = model.evaluate(X_train, y_train)
        test_metrics = model.evaluate(X_test, y_test)
        
        fold_result = {
            "fold": i + 1,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_accuracy": train_metrics.get("accuracy", 0),
            "test_accuracy": test_metrics.get("accuracy", 0),
            "train_f1": train_metrics.get("f1_macro", 0),
            "test_f1": test_metrics.get("f1_macro", 0),
        }
        
        results.append(fold_result)
        
        logger.info(f"  Train Accuracy: {fold_result['train_accuracy']:.4f}")
        logger.info(f"  Test Accuracy: {fold_result['test_accuracy']:.4f}")
        logger.info(f"  Test F1: {fold_result['test_f1']:.4f}")
    
    # Summary
    avg_test_acc = np.mean([r["test_accuracy"] for r in results])
    avg_test_f1 = np.mean([r["test_f1"] for r in results])
    std_test_acc = np.std([r["test_accuracy"] for r in results])
    
    logger.info(f"\nWalk-Forward Summary:")
    logger.info(f"  Average Test Accuracy: {avg_test_acc:.4f} (+/- {std_test_acc:.4f})")
    logger.info(f"  Average Test F1: {avg_test_f1:.4f}")
    
    return results


# =============================================================================
# MODEL SAVING AND REPORTING
# =============================================================================

def save_model(
    model: Any,
    symbol: str,
    model_type: str,
    output_dir: Path,
    results: dict[str, Any] | None = None,
) -> Path:
    """Save trained model and results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{symbol}_{model_type}_{timestamp}"
    
    # Save model
    model_path = output_dir / f"{model_name}.pkl"
    model.save(model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Save results
    if results:
        results_path = output_dir / f"{model_name}_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to: {results_path}")
    
    # Save feature importance
    if hasattr(model, "get_feature_importance"):
        importance = model.get_feature_importance()
        if importance:
            importance_path = output_dir / f"{model_name}_importance.json"
            with open(importance_path, "w") as f:
                json.dump(importance, f, indent=2)
            logger.info(f"Feature importance saved to: {importance_path}")
    
    return model_path


def print_training_report(
    model: Any,
    results: dict[str, Any],
    symbol: str,
    model_type: str,
) -> None:
    """Print training report to console."""
    print("\n" + "=" * 70)
    print("TRAINING REPORT")
    print("=" * 70)
    print(f"Symbol: {symbol}")
    print(f"Model Type: {model_type}")
    print(f"Model Name: {model.name if hasattr(model, 'name') else 'N/A'}")
    print("-" * 70)
    
    # Training metrics
    if "train_metrics" in results:
        print("\nTraining Metrics:")
        for key, value in results["train_metrics"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    # Validation metrics
    if "val_metrics" in results:
        print("\nValidation Metrics:")
        for key, value in results["val_metrics"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    # Test metrics
    if "test_metrics" in results:
        print("\nTest Metrics:")
        for key, value in results["test_metrics"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    # Best parameters
    if "best_params" in results:
        print("\nBest Hyperparameters:")
        for key, value in results["best_params"].items():
            print(f"  {key}: {value}")
    
    # Feature importance
    if hasattr(model, "get_feature_importance"):
        importance = model.get_feature_importance()
        if importance:
            print("\nTop 10 Features:")
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (feat, imp) in enumerate(sorted_importance, 1):
                print(f"  {i}. {feat}: {imp:.4f}")
    
    print("=" * 70)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train machine learning models for algorithmic trading",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data arguments
    parser.add_argument(
        "--symbol", "-s",
        type=str,
        help="Symbol to train on",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Multiple symbols to train on",
    )
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Train on all available symbols",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(settings.data.storage_path),
        help="Path to data directory",
    )
    
    # Model arguments
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="lightgbm",
        choices=["lightgbm", "xgboost", "catboost", "random_forest", "lstm", "transformer", "tcn"],
        help="Model type to train",
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Compare multiple model types",
    )
    parser.add_argument(
        "--models-to-compare",
        nargs="+",
        default=["lightgbm", "xgboost", "random_forest"],
        help="Models to compare when using --compare-models",
    )
    
    # Training arguments
    parser.add_argument(
        "--optimize",
        action="store_true",
        default=True,
        help="Run hyperparameter optimization",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip hyperparameter optimization",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Use walk-forward validation",
    )
    parser.add_argument(
        "--wf-splits",
        type=int,
        default=5,
        help="Number of walk-forward splits",
    )
    
    # Deep learning arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (deep learning)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (deep learning)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=60,
        help="Sequence length (deep learning)",
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="models/artifacts",
        help="Output directory for saved models",
    )
    parser.add_argument(
        "--save-features",
        action="store_true",
        help="Save generated features to parquet",
    )
    
    # Date range
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD)",
    )
    
    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> int:
    """Main entry point."""
    if args is None:
        args = parse_args()
    
    # Determine symbols
    data_path = Path(args.data_path)
    
    if args.all_symbols:
        symbols = discover_symbols(data_path)
        if not symbols:
            logger.error(f"No data files found in {data_path}")
            return 1
        logger.info(f"Found {len(symbols)} symbols: {symbols}")
    elif args.symbols:
        symbols = args.symbols
    elif args.symbol:
        symbols = [args.symbol]
    else:
        # Interactive mode - ask for symbol
        available = discover_symbols(data_path)
        if not available:
            logger.error(f"No data files found in {data_path}")
            return 1
        
        print("\nAvailable symbols:")
        for i, sym in enumerate(available, 1):
            print(f"  {i}. {sym}")
        
        choice = input("\nEnter symbol name or number: ").strip()
        try:
            idx = int(choice) - 1
            symbols = [available[idx]]
        except (ValueError, IndexError):
            symbols = [choice.upper()]
    
    # Parse dates
    start_date = None
    end_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    # Determine optimization setting
    optimize = args.optimize and not args.no_optimize
    
    print("\n" + "=" * 70)
    print("ML MODEL TRAINING PIPELINE")
    print("=" * 70)
    print(f"Symbols: {symbols}")
    print(f"Model: {args.model}")
    print(f"Optimize: {optimize}")
    print(f"Walk-Forward: {args.walk_forward}")
    print(f"Output: {args.output_dir}")
    print("=" * 70 + "\n")
    
    # Load data
    logger.info("Loading data...")
    data = load_data(symbols, data_path, start_date, end_date)
    
    if not data:
        logger.error("No data loaded!")
        return 1
    
    # Generate features
    feature_data = generate_features(data)
    
    if not feature_data:
        logger.error("No features generated!")
        return 1
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train for each symbol
    for symbol in symbols:
        if symbol not in feature_data:
            logger.warning(f"Skipping {symbol} - no features generated")
            continue
        
        X, y, feature_names = feature_data[symbol]
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Training model for {symbol}")
        logger.info(f"{'='*50}")
        logger.info(f"Features: {len(feature_names)}")
        logger.info(f"Samples: {len(X)}")
        logger.info(f"Classes: {np.unique(y)}")
        
        try:
            if args.compare_models:
                # Compare multiple models
                logger.info(f"Comparing models: {args.models_to_compare}")
                best_model, comparison = compare_models(
                    X, y, feature_names,
                    model_types=args.models_to_compare,
                    optimize=optimize,
                    n_trials=args.n_trials,
                )
                
                print("\nModel Comparison Results:")
                print(comparison)
                
                # Save best model
                model_type = best_model.name if hasattr(best_model, 'name') else 'best'
                save_model(best_model, symbol, model_type, output_dir)
                
            elif args.walk_forward:
                # Walk-forward validation
                wf_results = walk_forward_train(
                    args.model,
                    X, y, feature_names,
                    n_splits=args.wf_splits,
                    optimize=optimize,
                )
                
                # Save walk-forward results
                wf_path = output_dir / f"{symbol}_{args.model}_walkforward.json"
                with open(wf_path, "w") as f:
                    json.dump(wf_results, f, indent=2)
                logger.info(f"Walk-forward results saved to: {wf_path}")
                
            elif args.model in ["lstm", "transformer", "tcn"]:
                # Deep learning model
                model = train_deep_model(
                    args.model,
                    X, y, feature_names,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    sequence_length=args.sequence_length,
                )
                
                # Evaluate
                split_idx = int(len(X) * 0.8)
                test_metrics = model.evaluate(X[split_idx:], y[split_idx:])
                
                results = {
                    "model_type": args.model,
                    "symbol": symbol,
                    "test_metrics": test_metrics,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "sequence_length": args.sequence_length,
                }
                
                print_training_report(model, results, symbol, args.model)
                save_model(model, symbol, args.model, output_dir, results)
                
            else:
                # Standard ML model
                model, results = train_single_model(
                    args.model,
                    X, y, feature_names,
                    optimize=optimize,
                    n_trials=args.n_trials,
                )
                
                print_training_report(model, results, symbol, args.model)
                save_model(model, symbol, args.model, output_dir, results)
            
        except Exception as e:
            logger.exception(f"Error training model for {symbol}: {e}")
            continue
    
    logger.info("\nTraining pipeline completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())