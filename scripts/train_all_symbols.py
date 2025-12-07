#!/usr/bin/env python3
"""
Train All Symbols Script
========================

Batch training script for all 46 symbols with JPMorgan-level ML pipeline.

Features:
- Trains LightGBM + XGBoost ensemble for each symbol
- Uses Triple Barrier Method for target labeling
- Applies feature correlation filtering
- Saves models with proper naming convention
- Generates comprehensive training report

Usage:
    # Train all symbols
    python scripts/train_all_symbols.py
    
    # Train specific symbols
    python scripts/train_all_symbols.py --symbols AAPL GOOGL MSFT
    
    # Train with optimization
    python scripts/train_all_symbols.py --optimize --n-trials 50
    
    # Train core symbols only (most liquid)
    python scripts/train_all_symbols.py --core-only

Model Output Convention:
    models/artifacts/{SYMBOL}/{SYMBOL}_{model_type}_{version}.pkl
    
Example:
    models/artifacts/AAPL/AAPL_lightgbm_v1.pkl
    models/artifacts/AAPL/AAPL_xgboost_v1.pkl
    models/artifacts/AAPL/metadata.json

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
from config.symbols import (
    ALL_SYMBOLS, 
    CORE_SYMBOLS, 
    discover_symbols_from_data,
    get_model_filename,
    get_model_directory,
    validate_symbols,
    get_symbol_info,
)


# =============================================================================
# INITIALIZATION
# =============================================================================

settings = get_settings()
configure_logging(settings)
logger = get_logger(__name__)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_symbol_data(
    symbol: str,
    data_path: Path,
) -> pl.DataFrame | None:
    """Load data for a single symbol."""
    patterns = [
        f"{symbol}_15min.csv",
        f"{symbol}_1h.csv",
        f"{symbol.upper()}_15min.csv",
        f"{symbol.lower()}_15min.csv",
    ]
    
    for pattern in patterns:
        file_path = data_path / pattern
        if file_path.exists():
            try:
                df = pl.read_csv(file_path)
                
                # Ensure timestamp column
                if "timestamp" in df.columns:
                    df = df.with_columns([
                        pl.col("timestamp").str.to_datetime().alias("timestamp")
                    ])
                
                return df
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
    
    return None


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_symbol(
    symbol: str,
    data: pl.DataFrame,
    output_dir: Path,
    model_types: list[str],
    version: str = "v1",
    optimize: bool = True,
    n_trials: int = 30,
    use_triple_barrier: bool = True,
) -> dict[str, Any]:
    """
    Train models for a single symbol.
    
    Returns:
        Training results dictionary
    """
    from data.processor import DataProcessor
    from features.pipeline import FeaturePipeline, create_default_config
    from features.advanced import TripleBarrierLabeler, TripleBarrierConfig
    from models.training import TrainingPipeline, TrainingConfig, OptimizationConfig
    from models.model_manager import ModelManager
    
    results = {
        "symbol": symbol,
        "status": "pending",
        "models": {},
        "metrics": {},
        "training_time": 0,
        "error": None,
    }
    
    start_time = time.time()
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {symbol}")
        logger.info(f"{'='*60}")
        
        # Process data
        processor = DataProcessor()
        df = processor.process(data)
        logger.info(f"  Data: {len(df)} bars")
        
        # Generate features
        feature_pipeline = FeaturePipeline(create_default_config())
        df_features = feature_pipeline.generate(df)
        
        # Add advanced features
        from features.advanced import (
            MicrostructureFeatures,
            CalendarFeatures,
            FeatureInteractions,
        )
        df_features = MicrostructureFeatures.add_features(df_features)
        df_features = CalendarFeatures.add_features(df_features)
        df_features = FeatureInteractions.add_interactions(df_features)
        
        # Create target
        if use_triple_barrier:
            tb_config = TripleBarrierConfig(
                take_profit_multiplier=2.0,
                stop_loss_multiplier=1.0,
                max_holding_period=20,
            )
            labeler = TripleBarrierLabeler(tb_config)
            df_features = labeler.apply_binary_labels(df_features)
            logger.info("  Using Triple Barrier labels")
        else:
            df_features = feature_pipeline.create_target(
                df_features,
                target_type="direction",
                horizon=5,
            )
            logger.info("  Using direction labels")
        
        # Get numeric feature columns
        exclude_cols = {
            "timestamp", "symbol", "target", "open", "high", "low", "close", "volume",
            "tb_label", "tb_return", "tb_barrier", "tb_holding_period",
        }
        numeric_types = [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]
        
        feature_cols = []
        for col in df_features.columns:
            if col not in exclude_cols:
                if df_features[col].dtype in numeric_types:
                    feature_cols.append(col)
        
        # Correlation filtering
        feature_cols = filter_correlated_features(df_features, feature_cols, threshold=0.95)
        logger.info(f"  Features: {len(feature_cols)} (after correlation filter)")
        
        # Clean data
        df_clean = df_features.drop_nulls(subset=["target"])
        df_clean = df_clean.drop_nulls(subset=feature_cols)
        
        if len(df_clean) < 5000:
            raise ValueError(f"Insufficient data: {len(df_clean)} samples (need 5000+)")
        
        # Extract arrays
        X = df_clean.select(feature_cols).to_numpy().astype(np.float64)
        y = df_clean["target"].to_numpy().astype(np.int64)
        
        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Time-based split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        # Class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        class_dist = dict(zip(unique.tolist(), counts.tolist()))
        logger.info(f"  Class distribution: {class_dist}")
        
        results["data_info"] = {
            "total_samples": len(X),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_count": len(feature_cols),
            "class_distribution": class_dist,
        }
        
        # Initialize model manager
        model_manager = ModelManager(output_dir)
        
        # Configure training
        opt_config = OptimizationConfig(
            n_trials=n_trials if optimize else 0,
            cv_splits=3,
        )
        
        train_config = TrainingConfig(
            models_dir=output_dir,
            auto_optimize=optimize,
            optimization_config=opt_config,
        )
        
        training_pipeline = TrainingPipeline(train_config)
        
        # Train each model type
        for model_type in model_types:
            logger.info(f"\n  Training {model_type}...")
            
            try:
                model = training_pipeline.train(
                    model_type,
                    X_train, y_train,
                    X_test, y_test,
                    feature_names=feature_cols,
                )
                
                # Evaluate
                train_metrics = model.evaluate(X_train, y_train)
                test_metrics = model.evaluate(X_test, y_test)
                
                # Save with proper naming
                model_path = model_manager.save_model(
                    model=model,
                    symbol=symbol,
                    model_type=model_type,
                    version=version,
                    metrics={
                        "train_accuracy": train_metrics.get("accuracy", 0),
                        "test_accuracy": test_metrics.get("accuracy", 0),
                        "train_f1": train_metrics.get("f1_macro", 0),
                        "test_f1": test_metrics.get("f1_macro", 0),
                        "train_auc": train_metrics.get("roc_auc", 0),
                        "test_auc": test_metrics.get("roc_auc", 0),
                    },
                    feature_names=feature_cols,
                    training_samples=len(X_train),
                )
                
                results["models"][model_type] = {
                    "path": str(model_path),
                    "train_accuracy": train_metrics.get("accuracy", 0),
                    "test_accuracy": test_metrics.get("accuracy", 0),
                    "train_f1": train_metrics.get("f1_macro", 0),
                    "test_f1": test_metrics.get("f1_macro", 0),
                }
                
                logger.info(f"    Train Accuracy: {train_metrics.get('accuracy', 0):.4f}")
                logger.info(f"    Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")
                logger.info(f"    Test F1: {test_metrics.get('f1_macro', 0):.4f}")
                
            except Exception as e:
                logger.error(f"    Failed: {e}")
                results["models"][model_type] = {"error": str(e)}
        
        results["status"] = "success"
        
    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        logger.error(f"  Training failed: {e}")
    
    results["training_time"] = time.time() - start_time
    logger.info(f"  Training time: {results['training_time']:.1f}s")
    
    return results


def filter_correlated_features(
    df: pl.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.95,
) -> list[str]:
    """Remove highly correlated features."""
    if len(feature_cols) < 2:
        return feature_cols
    
    try:
        X = df.select(feature_cols).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Handle NaN correlations
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # Find features to remove
        to_remove = set()
        n = len(feature_cols)
        
        for i in range(n):
            if feature_cols[i] in to_remove:
                continue
            for j in range(i + 1, n):
                if feature_cols[j] in to_remove:
                    continue
                if abs(corr_matrix[i, j]) > threshold:
                    # Remove feature with higher average correlation
                    avg_i = np.mean(np.abs(corr_matrix[i, :]))
                    avg_j = np.mean(np.abs(corr_matrix[j, :]))
                    
                    if avg_i > avg_j:
                        to_remove.add(feature_cols[i])
                    else:
                        to_remove.add(feature_cols[j])
        
        filtered = [f for f in feature_cols if f not in to_remove]
        
        if len(to_remove) > 0:
            logger.debug(f"  Removed {len(to_remove)} correlated features")
        
        return filtered
        
    except Exception as e:
        logger.warning(f"  Correlation filtering failed: {e}")
        return feature_cols


# =============================================================================
# REPORTING
# =============================================================================

def generate_training_report(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate comprehensive training report."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_symbols": len(results),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "symbols": results,
    }
    
    # Calculate summary statistics
    successful_results = [r for r in results if r["status"] == "success"]
    
    if successful_results:
        accuracies = []
        for r in successful_results:
            for model_info in r.get("models", {}).values():
                if isinstance(model_info, dict) and "test_accuracy" in model_info:
                    accuracies.append(model_info["test_accuracy"])
        
        if accuracies:
            report["summary"] = {
                "avg_test_accuracy": float(np.mean(accuracies)),
                "std_test_accuracy": float(np.std(accuracies)),
                "min_test_accuracy": float(np.min(accuracies)),
                "max_test_accuracy": float(np.max(accuracies)),
            }
    
    # Save JSON report
    json_path = output_path / "training_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {json_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Total Symbols: {report['total_symbols']}")
    print(f"Successful: {report['successful']}")
    print(f"Failed: {report['failed']}")
    
    if "summary" in report:
        print(f"\nAccuracy Statistics:")
        print(f"  Average: {report['summary']['avg_test_accuracy']:.4f}")
        print(f"  Std Dev: {report['summary']['std_test_accuracy']:.4f}")
        print(f"  Min: {report['summary']['min_test_accuracy']:.4f}")
        print(f"  Max: {report['summary']['max_test_accuracy']:.4f}")
    
    # Top performers
    print("\nTop Performing Models:")
    all_models = []
    for r in successful_results:
        symbol = r["symbol"]
        for model_type, model_info in r.get("models", {}).items():
            if isinstance(model_info, dict) and "test_accuracy" in model_info:
                all_models.append({
                    "symbol": symbol,
                    "model": model_type,
                    "accuracy": model_info["test_accuracy"],
                })
    
    all_models.sort(key=lambda x: x["accuracy"], reverse=True)
    
    for i, m in enumerate(all_models[:10], 1):
        print(f"  {i}. {m['symbol']} ({m['model']}): {m['accuracy']:.4f}")
    
    # Failed symbols
    failed = [r for r in results if r["status"] == "failed"]
    if failed:
        print(f"\nFailed Symbols ({len(failed)}):")
        for r in failed:
            print(f"  - {r['symbol']}: {r.get('error', 'Unknown error')}")
    
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ML models for all symbols",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--symbols", "-s",
        nargs="+",
        help="Specific symbols to train (default: all available)",
    )
    
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Train only core (most liquid) symbols",
    )
    
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=["lightgbm", "xgboost"],
        help="Model types to train (default: lightgbm xgboost)",
    )
    
    parser.add_argument(
        "--version", "-v",
        default="v1",
        help="Model version string (default: v1)",
    )
    
    parser.add_argument(
        "--optimize",
        action="store_true",
        default=True,
        help="Enable hyperparameter optimization (default: True)",
    )
    
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable hyperparameter optimization",
    )
    
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of Optuna trials (default: 30)",
    )
    
    parser.add_argument(
        "--use-triple-barrier",
        action="store_true",
        default=True,
        help="Use Triple Barrier Method for targets (default: True)",
    )
    
    parser.add_argument(
        "--use-direction",
        action="store_true",
        help="Use simple direction targets instead of Triple Barrier",
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/storage",
        help="Path to data directory",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/artifacts",
        help="Output directory for models",
    )
    
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Maximum number of symbols to train (for testing)",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine symbols to train
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
        valid, invalid = validate_symbols(symbols)
        if invalid:
            logger.warning(f"Unknown symbols (will still attempt): {invalid}")
        symbols = valid + invalid
    elif args.core_only:
        symbols = CORE_SYMBOLS
        logger.info(f"Training core symbols only: {symbols}")
    else:
        # Discover from data directory
        symbols = discover_symbols_from_data(data_path)
        if not symbols:
            logger.error(f"No data files found in {data_path}")
            return 1
        logger.info(f"Discovered {len(symbols)} symbols from data")
    
    # Limit symbols if specified
    if args.max_symbols:
        symbols = symbols[:args.max_symbols]
    
    # Determine optimization settings
    optimize = args.optimize and not args.no_optimize
    
    # Determine target type
    use_triple_barrier = args.use_triple_barrier and not args.use_direction
    
    print("\n" + "=" * 70)
    print("BATCH MODEL TRAINING")
    print("=" * 70)
    print(f"Symbols: {len(symbols)}")
    print(f"Models: {args.models}")
    print(f"Version: {args.version}")
    print(f"Optimize: {optimize} ({args.n_trials} trials)")
    print(f"Target: {'Triple Barrier' if use_triple_barrier else 'Direction'}")
    print(f"Output: {output_dir}")
    print("=" * 70 + "\n")
    
    # Train each symbol
    all_results = []
    total_start = time.time()
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")
        
        # Load data
        df = load_symbol_data(symbol, data_path)
        
        if df is None:
            logger.warning(f"No data found for {symbol}")
            all_results.append({
                "symbol": symbol,
                "status": "failed",
                "error": "No data found",
            })
            continue
        
        # Train
        result = train_symbol(
            symbol=symbol,
            data=df,
            output_dir=output_dir,
            model_types=args.models,
            version=args.version,
            optimize=optimize,
            n_trials=args.n_trials,
            use_triple_barrier=use_triple_barrier,
        )
        
        all_results.append(result)
    
    total_time = time.time() - total_start
    
    # Generate report
    print(f"\n\nTotal training time: {total_time/60:.1f} minutes")
    generate_training_report(all_results, output_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())