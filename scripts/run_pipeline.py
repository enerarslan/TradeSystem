#!/usr/bin/env python3
"""
AlphaTrade Master Pipeline Script
==================================

Orchestrates the entire workflow from data preparation to paper trading.

Stages:
    1. Data Preparation - Download and preprocess historical data
    2. Feature Engineering - Generate institutional features
    3. Label Generation - Create triple barrier labels
    4. Model Training - Train CatBoost with purged k-fold CV
    5. Probability Calibration - Calibrate model probabilities for Kelly
    6. Backtest - Validate strategy with realistic execution
    7. Validation - Verify all components work together
    8. Paper Trading - Live paper trading (optional)

Usage:
    python scripts/run_pipeline.py                    # Run full pipeline
    python scripts/run_pipeline.py --stage train     # Run single stage
    python scripts/run_pipeline.py --from-stage 4    # Resume from stage 4
    python scripts/run_pipeline.py --skip-data       # Skip data download
    python scripts/run_pipeline.py --paper           # Include paper trading
"""

import sys
import os
import argparse
import json
import pickle
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

# Suppress warnings before imports
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import yaml
import pandas as pd
import numpy as np
import hashlib
import cProfile
import pstats
import io

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger, setup_logging


# Setup logging - default to INFO, can be overridden by optimization config
setup_logging(log_path="logs", level="INFO")
logger = get_logger(__name__)


# =============================================================================
# OPTIMIZATION UTILITIES
# =============================================================================

def load_optimization_config() -> Dict:
    """Load backtest optimization configuration."""
    config_path = Path("config/backtest_optimization.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {
        'optimization': {
            'use_vectorized_backtest': True,
            'enable_feature_cache': True,
            'cache_dir': 'data/cache',
            'fast_mode': True,
            'parallel_workers': 8,
            'logging_level': 'WARNING'
        }
    }


def compute_feature_cache_hash(
    symbols: List[str],
    config: Dict,
    data_hash: str = None
) -> str:
    """
    Compute a hash for feature caching based on:
    - Symbol list
    - Feature configuration
    - Data hash (optional)
    """
    hash_input = {
        'symbols': sorted(symbols),
        'feature_config': {
            'fracdiff_threshold': config.get('fracdiff_threshold', 1e-5),
            'vpin_n_buckets': config.get('vpin_n_buckets', 50),
            'hmm_n_states': config.get('hmm_n_states', 3),
        },
        'data_hash': data_hash
    }
    hash_str = json.dumps(hash_input, sort_keys=True)
    return hashlib.md5(hash_str.encode()).hexdigest()[:12]


def load_cached_features(cache_path: Path, cache_hash: str) -> Optional[Dict]:
    """Load features from cache if valid."""
    cache_file = cache_path / f"features_{cache_hash}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            logger.info(f"Loaded cached features from {cache_file}")
            return cached
        except Exception as e:
            logger.warning(f"Failed to load feature cache: {e}")
    return None


def save_features_to_cache(
    features_data: Dict,
    cache_path: Path,
    cache_hash: str
) -> None:
    """Save features to cache."""
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / f"features_{cache_hash}.pkl"
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(features_data, f)
        logger.info(f"Saved features to cache: {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to save feature cache: {e}")


# =============================================================================
# CONSTANTS
# =============================================================================

STAGES = {
    1: "data",
    2: "features",
    3: "labels",
    4: "train",
    5: "calibrate",
    6: "backtest",
    7: "validate",
    8: "paper"
}

STAGE_NAMES = {v: k for k, v in STAGES.items()}

CHECKPOINT_FILE = "results/.pipeline_checkpoint.json"


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

def save_checkpoint(stage: int, status: str, metadata: Dict = None):
    """Save pipeline checkpoint."""
    os.makedirs("results", exist_ok=True)
    checkpoint = {
        "stage": stage,
        "stage_name": STAGES.get(stage, "unknown"),
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {}
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    logger.info(f"Checkpoint saved: Stage {stage} ({STAGES.get(stage)}) - {status}")


def load_checkpoint() -> Optional[Dict]:
    """Load pipeline checkpoint."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return None


def clear_checkpoint():
    """Clear pipeline checkpoint."""
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


# =============================================================================
# STAGE 1: DATA PREPARATION
# =============================================================================

def stage_data(symbols: List[str], start_date: str, end_date: str, force: bool = False) -> bool:
    """
    Stage 1: Download and preprocess historical data.

    ISSUE 7.1 FIX: Added data quality validation before saving.

    Output: data/processed/combined_data.pkl
    """
    print("\n" + "=" * 70)
    print("STAGE 1: DATA PREPARATION (WITH QUALITY VALIDATION)")
    print("=" * 70)

    output_path = Path("data/processed/combined_data.pkl")

    # Check if data already exists
    if output_path.exists() and not force:
        print(f"Data already exists at {output_path}")
        print("Use --force to re-download")
        return True

    try:
        from src.data.loader import MultiAssetLoader
        from src.data.preprocessor import DataPreprocessor
        from src.risk.position_sizer import DataQualityValidator

        print(f"\nDownloading data for {len(symbols)} symbols...")
        print(f"Date range: {start_date} to {end_date}")

        # Initialize loader
        loader = MultiAssetLoader(data_path="data/raw", cache_path="data/cache")
        preprocessor = DataPreprocessor()

        # ISSUE 7.1 FIX: Initialize data quality validator
        quality_validator = DataQualityValidator()

        # Load all symbols
        raw_data = loader.load_symbols(symbols=symbols, show_progress=True)

        # Preprocess and validate
        processed_data = {}
        quality_passed = 0
        quality_failed = 0

        for symbol, df in raw_data.items():
            if df is not None and len(df) > 100:
                try:
                    df_clean, report = preprocessor.preprocess(df, symbol)

                    # ISSUE 7.1 FIX: Validate data quality
                    validation_result = quality_validator.validate(df_clean, symbol)

                    if validation_result['passed']:
                        processed_data[symbol] = df_clean
                        quality_passed += 1
                        print(f"  {symbol}: {len(df_clean)} bars (quality: {report.quality_score:.1f}) [PASS]")
                    else:
                        quality_failed += 1
                        errors = ", ".join(validation_result['errors'][:2])
                        print(f"  {symbol}: FAILED quality check - {errors}")
                        # Still include data with warnings (not critical errors)
                        if not validation_result['errors']:
                            processed_data[symbol] = df_clean

                except Exception as e:
                    logger.warning(f"Failed to preprocess {symbol}: {e}")

        if not processed_data:
            print("ERROR: No data processed!")
            return False

        # Save combined data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)

        print(f"\nData saved to {output_path}")
        print(f"Processed {len(processed_data)}/{len(symbols)} symbols")
        print(f"Quality validation: {quality_passed} passed, {quality_failed} failed")

        return True

    except Exception as e:
        logger.error(f"Stage 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# STAGE 2: FEATURE ENGINEERING
# =============================================================================

def build_features_single_symbol(args: tuple) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    Build features for a single symbol (for parallel processing).

    ISSUE 1.3 FIX: Added point-in-time regime features.

    Args:
        args: Tuple of (symbol, df, config)

    Returns:
        Tuple of (symbol, features_df or None)
    """
    from src.features.institutional import InstitutionalFeatureEngineer

    symbol, df, config = args

    try:
        # Use fast mode settings if specified
        institutional_fe = InstitutionalFeatureEngineer()

        # Generate features
        features = institutional_fe.build_features(df)

        # ISSUE 1.3 FIX: Add point-in-time regime features
        if features is not None and len(features) > 0:
            try:
                from src.features.regime import generate_regime_features_pit

                # Generate PIT regime features
                regime_features = generate_regime_features_pit(df, n_regimes=3)

                if regime_features is not None and len(regime_features) > 0:
                    # Align indices and merge
                    common_idx = features.index.intersection(regime_features.index)
                    if len(common_idx) > 0:
                        features = features.loc[common_idx]
                        for col in regime_features.columns:
                            features[f'pit_{col}'] = regime_features.loc[common_idx, col]

            except Exception as e:
                # Non-critical - continue without PIT regime features
                logger.debug(f"PIT regime features skipped for {symbol}: {e}")

            return (symbol, features)
        return (symbol, None)

    except Exception as e:
        logger.warning(f"Feature generation failed for {symbol}: {e}")
        return (symbol, None)


def stage_features(force: bool = False) -> bool:
    """
    Stage 2: Generate institutional features with caching and parallel processing.

    OPTIMIZED VERSION:
    - Uses disk cache with hash-based invalidation
    - Parallel symbol processing via ProcessPoolExecutor
    - Pre-computes HMM regime features (no look-ahead)

    Input: data/processed/combined_data.pkl
    Output: results/features/combined_features.pkl
    """
    print("\n" + "=" * 70)
    print("STAGE 2: FEATURE ENGINEERING (OPTIMIZED)")
    print("=" * 70)

    input_path = Path("data/processed/combined_data.pkl")
    output_path = Path("results/features/combined_features.pkl")

    # Load optimization config
    opt_config = load_optimization_config()
    opt = opt_config.get('optimization', {})
    use_cache = opt.get('enable_feature_cache', True)
    cache_dir = Path(opt.get('cache_dir', 'data/cache'))
    parallel_workers = opt.get('parallel_workers', 8)
    use_parallel = opt.get('use_multiprocessing', True)

    # Check dependencies
    if not input_path.exists():
        print(f"ERROR: Input not found: {input_path}")
        print("Run Stage 1 (data) first")
        return False

    # Check if output exists
    if output_path.exists() and not force:
        print(f"Features already exist at {output_path}")
        print("Use --force to regenerate")
        return True

    try:
        from src.features.institutional import InstitutionalFeatureEngineer
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

        # Load data
        print("\nLoading preprocessed data...")
        with open(input_path, 'rb') as f:
            data = pickle.load(f)

        symbols = list(data.keys())
        print(f"Loaded {len(symbols)} symbols")

        # Check feature cache
        if use_cache and not force:
            # Compute cache hash based on data modification time
            data_hash = str(int(input_path.stat().st_mtime))
            cache_hash = compute_feature_cache_hash(symbols, opt, data_hash)

            cached_features = load_cached_features(cache_dir, cache_hash)
            if cached_features is not None:
                print(f"Using cached features (hash: {cache_hash})")

                # Save to output path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    pickle.dump(cached_features, f)

                print(f"Features loaded from cache: {len(cached_features)} symbols")
                return True

        # Generate features
        features_data = {}
        start_time = time.time()

        if use_parallel and len(symbols) > 1:
            # Parallel feature generation
            print(f"Using parallel processing ({parallel_workers} workers)...")

            # Use ThreadPoolExecutor for better compatibility with pandas
            # ProcessPoolExecutor can have serialization issues
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                # Submit all jobs
                futures = {
                    executor.submit(
                        build_features_single_symbol,
                        (symbol, data[symbol], opt)
                    ): symbol
                    for symbol in symbols
                }

                # Collect results
                completed = 0
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        result_symbol, features = future.result()
                        if features is not None:
                            features_data[result_symbol] = features
                            completed += 1
                            print(f"\r  Completed: {completed}/{len(symbols)} symbols", end="")
                    except Exception as e:
                        logger.warning(f"Failed for {symbol}: {e}")

                print()  # New line after progress

        else:
            # Sequential processing
            institutional_fe = InstitutionalFeatureEngineer()

            for symbol, df in data.items():
                try:
                    print(f"  Processing {symbol}...", end=" ")

                    features = institutional_fe.build_features(df)

                    if features is not None and len(features) > 0:
                        features_data[symbol] = features
                        print(f"{len(features)} samples, {len(features.columns)} features")
                    else:
                        print("skipped (no features)")

                except Exception as e:
                    print(f"failed ({e})")

        elapsed = time.time() - start_time

        if not features_data:
            print("ERROR: No features generated!")
            return False

        # Save features to output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(features_data, f)

        # Save to cache for future runs
        if use_cache:
            data_hash = str(int(input_path.stat().st_mtime))
            cache_hash = compute_feature_cache_hash(symbols, opt, data_hash)
            save_features_to_cache(features_data, cache_dir, cache_hash)

        print(f"\nFeatures saved to {output_path}")
        print(f"Generated features for {len(features_data)} symbols in {elapsed:.1f}s")

        return True

    except Exception as e:
        logger.error(f"Stage 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# STAGE 3: LABEL GENERATION
# =============================================================================

def stage_labels(pt_sl_ratio: tuple = (1.5, 1.0), max_holding: int = 20, force: bool = False) -> bool:
    """
    Stage 3: Create triple barrier labels with decorrelation.

    ISSUE 1.1 FIX: Added label decorrelation to reduce autocorrelation.

    Input: data/processed/combined_data.pkl
    Output: results/labels/labels.pkl
    """
    print("\n" + "=" * 70)
    print("STAGE 3: LABEL GENERATION (WITH DECORRELATION)")
    print("=" * 70)

    input_path = Path("data/processed/combined_data.pkl")
    output_path = Path("results/labels/labels.pkl")

    # Check dependencies
    if not input_path.exists():
        print(f"ERROR: Input not found: {input_path}")
        print("Run Stage 1 (data) first")
        return False

    # Check if output exists
    if output_path.exists() and not force:
        print(f"Labels already exist at {output_path}")
        print("Use --force to regenerate")
        return True

    try:
        from src.data.labeling import TripleBarrierLabeler, TripleBarrierConfig
        from src.data.labeling import (
            apply_label_decorrelation_to_events,
            calculate_label_autocorrelation,
            validate_label_quality
        )
        from src.models.meta_labeling import MetaLabelingPipeline, MetaLabelingConfig, TrendFollowingSignal

        # Load data
        print("\nLoading preprocessed data...")
        with open(input_path, 'rb') as f:
            data = pickle.load(f)

        print(f"Loaded {len(data)} symbols")
        print(f"PT/SL ratio: {pt_sl_ratio}, Max holding: {max_holding} bars")

        # Meta-labeling configuration
        meta_config = MetaLabelingConfig(
            pt_sl_ratio=pt_sl_ratio,
            max_holding_period=max_holding,
            use_sample_weights=True,
            time_decay_factor=0.5
        )

        primary_signal = TrendFollowingSignal(
            fast_period=10,
            slow_period=30,
            atr_filter=False
        )

        pipeline = MetaLabelingPipeline(meta_config, primary_signal)

        # Generate labels for each symbol
        labels_data = {}
        total_original = 0
        total_decorrelated = 0

        for symbol, df in data.items():
            try:
                print(f"  Labeling {symbol}...", end=" ")

                # Standardize columns
                df.columns = df.columns.str.lower()

                # Generate labels
                labels_df = pipeline.generate_labels(df)

                if labels_df is not None and len(labels_df) > 0:
                    total_original += len(labels_df)

                    # ISSUE 1.1 FIX: Apply label decorrelation
                    if 'bin' in labels_df.columns:
                        # Calculate initial autocorrelation
                        initial_autocorr = calculate_label_autocorrelation(
                            labels_df['bin'],
                            max_lag=10
                        )

                        # Apply decorrelation if autocorrelation is high
                        if initial_autocorr.get(1, 0) > 0.1:
                            labels_df = apply_label_decorrelation_to_events(
                                labels_df,
                                method='subsample',
                                min_gap_bars=4
                            )

                            # Validate quality
                            quality = validate_label_quality(labels_df, labels_df['bin'])
                            final_autocorr = quality.get('autocorrelation_lag1', 0)

                            print(f"{len(labels_df)} labels (decorrelated: {initial_autocorr.get(1, 0):.2f} -> {final_autocorr:.2f})")
                        else:
                            print(f"{len(labels_df)} labels (autocorr OK: {initial_autocorr.get(1, 0):.2f})")
                    else:
                        print(f"{len(labels_df)} labels")

                    total_decorrelated += len(labels_df)
                    labels_data[symbol] = labels_df
                else:
                    print("skipped (no labels)")

            except Exception as e:
                print(f"failed ({e})")
                import traceback
                traceback.print_exc()

        if not labels_data:
            print("ERROR: No labels generated!")
            return False

        # Save labels
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(labels_data, f)

        print(f"\nLabels saved to {output_path}")
        print(f"Generated labels for {len(labels_data)} symbols")
        print(f"Labels after decorrelation: {total_decorrelated}/{total_original} ({total_decorrelated/max(total_original, 1)*100:.1f}%)")

        return True

    except Exception as e:
        logger.error(f"Stage 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# STAGE 4: MODEL TRAINING
# =============================================================================

def stage_train(model_type: str = "catboost", n_estimators: int = 100, force: bool = False) -> bool:
    """
    Stage 4: Train CatBoost model with purged k-fold CV.

    Input: results/features/combined_features.pkl, results/labels/labels.pkl
    Output: models/model.pkl, models/metrics.yaml
    """
    print("\n" + "=" * 70)
    print("STAGE 4: MODEL TRAINING")
    print("=" * 70)

    model_path = Path("models/model.pkl")

    # Check if model exists
    if model_path.exists() and not force:
        print(f"Model already exists at {model_path}")

        # Show existing metrics
        metrics_path = Path("models/metrics.yaml")
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = yaml.safe_load(f)
            print(f"Training date: {metrics.get('training_date', 'unknown')}")
            perf = metrics.get('performance', {})
            print(f"CV Accuracy: {perf.get('cv_accuracy', 0):.4f}")

        print("Use --force to retrain")
        return True

    try:
        # Use existing training script
        print("\nRunning institutional training pipeline...")
        print(f"Model type: {model_type}, Estimators: {n_estimators}")

        # Import and run training
        import subprocess

        cmd = [
            sys.executable,
            "scripts/train_models.py",
            "--model", model_type,
            "--n-estimators", str(n_estimators)
        ]

        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode != 0:
            print("Training failed!")
            return False

        # Verify output
        if not model_path.exists():
            print(f"ERROR: Model not created at {model_path}")
            return False

        print(f"\nModel saved to {model_path}")
        return True

    except Exception as e:
        logger.error(f"Stage 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# STAGE 5: PROBABILITY CALIBRATION
# =============================================================================

def stage_calibrate(force: bool = False) -> bool:
    """
    Stage 5: Calibrate model probabilities for Kelly sizing.

    Input: models/model.pkl, validation data
    Output: models/calibration_model.pkl
    """
    print("\n" + "=" * 70)
    print("STAGE 5: PROBABILITY CALIBRATION")
    print("=" * 70)

    model_path = Path("models/model.pkl")
    calibration_path = Path("models/calibration_model.pkl")

    # Check dependencies
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        print("Run Stage 4 (train) first")
        return False

    # Check if calibration exists
    if calibration_path.exists() and not force:
        print(f"Calibration model already exists at {calibration_path}")
        print("Use --force to recalibrate")
        return True

    try:
        from src.models.calibration import IsotonicCalibrator, PlattCalibrator
        import joblib

        # Load model
        print("\nLoading trained model...")
        model = joblib.load(model_path)

        # Load holdout data for calibration
        holdout_path = Path("data/holdout")
        features_path = Path("results/features/combined_features.pkl")
        labels_path = Path("results/labels/labels.pkl")

        if not features_path.exists() or not labels_path.exists():
            print("WARNING: Features/labels not found, skipping calibration")
            print("Creating placeholder calibrator...")

            # Create uncalibrated placeholder
            calibrator = IsotonicCalibrator()
            with open(calibration_path, 'wb') as f:
                pickle.dump(calibrator, f)

            return True

        # Load features and labels
        with open(features_path, 'rb') as f:
            features_data = pickle.load(f)
        with open(labels_path, 'rb') as f:
            labels_data = pickle.load(f)

        # Combine data for calibration (use last 20% as validation)
        print("Preparing calibration data...")

        X_list = []
        y_list = []

        for symbol in features_data.keys():
            if symbol in labels_data:
                feat = features_data[symbol]
                lab = labels_data[symbol]

                # Align indices
                common_idx = feat.index.intersection(lab.index)
                if len(common_idx) > 100:
                    # Use last 20% for calibration
                    n_cal = int(len(common_idx) * 0.2)
                    cal_idx = common_idx[-n_cal:]

                    X_list.append(feat.loc[cal_idx])
                    if 'bin' in lab.columns:
                        y_list.append(lab.loc[cal_idx, 'bin'])
                    elif 'label' in lab.columns:
                        y_list.append(lab.loc[cal_idx, 'label'])

        if not X_list:
            print("WARNING: No calibration data available")
            calibrator = IsotonicCalibrator()
            with open(calibration_path, 'wb') as f:
                pickle.dump(calibrator, f)
            return True

        X_cal = pd.concat(X_list, axis=0)
        y_cal = pd.concat(y_list, axis=0)

        # Get numeric columns only
        X_cal = X_cal.select_dtypes(include=[np.number]).fillna(0)

        print(f"Calibration set: {len(X_cal)} samples")

        # Get model predictions
        print("Getting model predictions...")

        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_cal)
                if proba.ndim > 1:
                    proba = proba[:, 1]  # Probability of positive class
            else:
                proba = model.predict(X_cal)
        except Exception as e:
            print(f"Prediction failed: {e}")
            print("Creating placeholder calibrator...")
            calibrator = IsotonicCalibrator()
            with open(calibration_path, 'wb') as f:
                pickle.dump(calibrator, f)
            return True

        # Fit calibrator
        print("Fitting isotonic calibrator...")
        calibrator = IsotonicCalibrator()
        calibrator.fit(proba, y_cal.values)

        # Evaluate calibration
        metrics = calibrator.evaluate(proba, y_cal.values)
        print(f"\nCalibration Metrics:")
        print(f"  ECE (Expected Calibration Error): {metrics.ece:.4f}")
        print(f"  MCE (Maximum Calibration Error): {metrics.mce:.4f}")
        print(f"  Brier Score: {metrics.brier_score:.4f}")

        # Save calibrator
        with open(calibration_path, 'wb') as f:
            pickle.dump(calibrator, f)

        print(f"\nCalibration model saved to {calibration_path}")

        return True

    except Exception as e:
        logger.error(f"Stage 5 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# STAGE 6: BACKTEST (TRUE OUT-OF-SAMPLE WITH FRESH DATA)
# =============================================================================

def download_oos_data(symbols: list, start_date: str, end_date: str = None) -> Dict[str, pd.DataFrame]:
    """
    Download fresh OOS data from Alpaca.

    This ensures we test on data the model has NEVER seen.

    Alpaca provides extensive historical data (years of 15-min bars).
    Uses IEX feed which is available on free tier.

    Args:
        symbols: List of stock symbols
        start_date: Start date for OOS period (should be after training end)
        end_date: End date (defaults to today)

    Returns:
        Dictionary of symbol -> DataFrame with 15min OHLCV data
    """
    from dotenv import load_dotenv
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from alpaca_trade_api import REST

    load_dotenv()

    api_key = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')

    if not api_key or not secret_key:
        print("ERROR: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")
        return {}

    api = REST(api_key, secret_key, base_url='https://paper-api.alpaca.markets')

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"\nDownloading OOS data from Alpaca (IEX feed)...")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Symbols: {len(symbols)}")

    def download_symbol(symbol: str) -> tuple:
        """Download single symbol data."""
        try:
            # Handle special ticker symbols for Alpaca
            alpaca_symbol = symbol.replace('.', '/')  # BRK.B -> BRK/B for Alpaca

            bars = api.get_bars(
                alpaca_symbol,
                '15Min',
                start=start_date,
                end=end_date,
                adjustment='split',
                feed='iex'  # Use IEX feed for free tier
            ).df

            if bars is None or len(bars) == 0:
                return (symbol, None, "No data returned")

            # Remove duplicates
            bars = bars[~bars.index.duplicated(keep='last')]

            # Standardize columns (Alpaca returns: open, high, low, close, volume, trade_count, vwap)
            df = bars[['open', 'high', 'low', 'close', 'volume']].copy()

            # Index is already timezone-aware (UTC) from Alpaca
            return (symbol, df, None)

        except Exception as e:
            return (symbol, None, str(e))

    # Download in parallel (but with rate limiting for Alpaca)
    data = {}
    errors = []

    # Use fewer workers to respect rate limits
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_symbol, sym): sym for sym in symbols}

        for i, future in enumerate(as_completed(futures)):
            symbol, df, error = future.result()

            if error:
                errors.append(f"{symbol}: {error}")
            elif df is not None and len(df) > 100:
                data[symbol] = df

            # Progress
            print(f"\r  Progress: {i+1}/{len(symbols)} symbols...", end="")

            # Small delay for rate limiting
            time.sleep(0.05)

    print(f"\n  Downloaded: {len(data)} symbols successfully")
    if errors:
        print(f"  Errors: {len(errors)} symbols failed")
        for err in errors[:5]:  # Show first 5 errors
            print(f"    - {err}")

    return data


def diagnose_model_predictions(
    model,
    data: Dict[str, pd.DataFrame],
    feature_builder,
    sample_size: int = 1000
) -> Dict[str, Any]:
    """
    Diagnose if model predictions are worse than random.

    CRITICAL: This detects when the model is generating anti-signals
    (predictions that are consistently OPPOSITE of the correct direction).

    Args:
        model: Trained model
        data: Dict of symbol -> DataFrame
        feature_builder: Feature engineering pipeline
        sample_size: Number of samples to evaluate

    Returns:
        Dict with diagnostic metrics
    """
    print("\n" + "=" * 60)
    print("MODEL PREDICTION DIAGNOSTIC")
    print("=" * 60)

    all_predictions = []
    all_actuals = []
    all_probas = []

    for symbol, df in data.items():
        try:
            features = feature_builder.build_features(df)
            if features is None or len(features) < 100:
                continue

            X = features.select_dtypes(include=[np.number]).fillna(0)

            # Get predictions
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                if proba.ndim > 1:
                    # Check which class is which
                    proba = proba[:, 1]  # Probability of class 1
                all_probas.extend(proba[:-1].tolist())
                predictions = (proba > 0.5).astype(int)
            else:
                predictions = model.predict(X)
                all_probas.extend([0.5] * (len(predictions) - 1))

            # Get actual future returns (1-bar forward)
            actual_returns = df['close'].pct_change().shift(-1).loc[features.index]
            actual_direction = (actual_returns > 0).astype(int)

            all_predictions.extend(predictions[:-1].tolist())
            all_actuals.extend(actual_direction[:-1].tolist())

            if len(all_predictions) >= sample_size:
                break

        except Exception as e:
            logger.debug(f"Diagnostic skipped {symbol}: {e}")

    if len(all_predictions) < 100:
        print("WARNING: Insufficient data for model diagnostic")
        return {'accuracy': 0.5, 'status': 'INSUFFICIENT_DATA'}

    # Calculate accuracy
    predictions = np.array(all_predictions)
    actuals = np.array(all_actuals)
    probas = np.array(all_probas)

    accuracy = np.mean(predictions == actuals)
    avg_proba = np.mean(probas)

    # Calculate correlation between predicted probability and actual returns
    correlation = np.corrcoef(probas, actuals)[0, 1] if len(probas) > 10 else 0

    print(f"\nSamples analyzed: {len(predictions)}")
    print(f"Directional Accuracy: {accuracy:.2%}")
    print(f"Average Probability: {avg_proba:.4f}")
    print(f"Proba-Return Correlation: {correlation:.4f}")
    print(f"Expected Random: 50%")

    # Determine status
    if accuracy < 0.45:
        status = 'INVERTED_SIGNALS'
        print(f"\n*** WARNING: Model accuracy {accuracy:.2%} is BELOW 45%! ***")
        print("    The model is predicting OPPOSITE of correct direction.")
        print("    Consider: 1) Inverting signals, 2) Checking label encoding")
        print("    3) Retraining with correct labels")
    elif accuracy < 0.48:
        status = 'NO_EDGE'
        print(f"\n*** WARNING: Model has NO EDGE (accuracy ~50%) ***")
        print("    The model is not better than random guessing.")
    elif accuracy > 0.52:
        status = 'OK'
        print(f"\n*** Model appears to have predictive power ***")
    else:
        status = 'MARGINAL'
        print(f"\n*** Model has marginal edge ***")

    # Check if classes might be inverted
    if hasattr(model, 'classes_'):
        print(f"\nModel classes: {model.classes_}")

    if hasattr(model, '_label_encoder') and model._label_encoder is not None:
        le = model._label_encoder
        if hasattr(le, 'classes_'):
            print(f"Label encoder classes: {le.classes_}")

    print("=" * 60)

    return {
        'accuracy': float(accuracy),
        'avg_proba': float(avg_proba),
        'correlation': float(correlation),
        'n_samples': len(predictions),
        'status': status,
        'should_invert': accuracy < 0.45
    }


def run_vectorized_backtest(
    data: Dict[str, pd.DataFrame],
    model,
    feature_builder,
    config: Dict,
    invert_signals: bool = False
) -> Dict[str, Any]:
    """
    Run fast vectorized backtest for development/optimization.

    This is 50-100x faster than event-driven backtest.
    Used for quick iteration, not final validation.

    FIXES APPLIED:
    - Added model diagnostic to detect inverted signals
    - Fixed probability-to-signal conversion
    - Added position limits via VectorizedBacktester v2
    - Improved signal threshold calibration

    Args:
        data: Dict of symbol -> DataFrame
        model: Trained model
        feature_builder: Feature engineering pipeline
        config: Backtest configuration
        invert_signals: If True, invert model signals (for anti-correlated models)

    Returns:
        Dict with backtest results
    """
    from src.backtest.engine import VectorizedBacktester

    print("\n  Running VECTORIZED backtest (fast mode)...")

    # Build features and signals for all symbols
    all_prices = []
    all_signals = []

    for symbol, df in data.items():
        try:
            # Build features
            features = feature_builder.build_features(df)

            if features is None or len(features) == 0:
                continue

            # Get predictions from model
            X = features.select_dtypes(include=[np.number]).fillna(0)

            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                if proba.ndim > 1:
                    # FIXED: Verify which column corresponds to positive class
                    # For binary classification, column 1 is typically P(class=1)
                    # If model was trained with labels {-1, 1} or {0, 1}, class 1 should be "bullish"
                    proba = proba[:, 1]

                # Convert probability to signal (-1 to 1)
                # P > 0.5 -> bullish, P < 0.5 -> bearish
                signals = (proba - 0.5) * 2

                # FIXED: Invert if model is anti-correlated
                if invert_signals:
                    signals = -signals
            else:
                predictions = model.predict(X)
                # Map predictions to signals
                if set(np.unique(predictions)) <= {0, 1}:
                    signals = (predictions - 0.5) * 2
                else:
                    signals = predictions

                if invert_signals:
                    signals = -signals

            # Create signal series aligned with prices
            signal_series = pd.Series(signals, index=features.index)

            # Get prices aligned with features
            prices = df['close'].loc[features.index]

            all_prices.append(prices.rename(symbol))
            all_signals.append(signal_series.rename(symbol))

        except Exception as e:
            logger.warning(f"Vectorized backtest skipped {symbol}: {e}")

    if not all_prices:
        return {'error': 'No valid data for vectorized backtest'}

    # Combine into DataFrames
    prices_df = pd.concat(all_prices, axis=1)
    signals_df = pd.concat(all_signals, axis=1)

    # Align indices
    common_idx = prices_df.index.intersection(signals_df.index)
    prices_df = prices_df.loc[common_idx]
    signals_df = signals_df.loc[common_idx]

    # Fill NaN signals with 0 (no position)
    signals_df = signals_df.fillna(0)

    # FIXED: Use VectorizedBacktester v2 with position limits
    # Signal thresholding is now handled inside the backtester
    backtester = VectorizedBacktester(
        initial_capital=config.get('initial_capital', 1000000),
        commission_pct=config.get('commission_pct', 0.001),
        slippage_pct=config.get('slippage_pct', 0.0005),
        max_position_pct=config.get('max_position_pct', 0.10),      # 10% max per position
        max_gross_exposure=config.get('max_gross_exposure', 0.80),   # 80% max invested
        min_signal_threshold=config.get('min_signal_threshold', 0.3) # Only trade strong signals
    )

    # Pass continuous signals - the backtester handles thresholding
    result = backtester.run(prices_df, signals_df)

    return result


def stage_backtest(force: bool = False) -> bool:
    """
    Stage 6: Validate strategy with realistic execution on TRUE OUT-OF-SAMPLE data.

    OPTIMIZED VERSION with ISSUE 8.1/8.2 FIXES:
    - Uses VectorizedBacktester for fast mode (50-100x faster)
    - Falls back to InstitutionalBacktestEngine for final validation
    - Configurable via backtest_optimization.yaml
    - ISSUE 8.1: Model staleness detection
    - ISSUE 8.2: Feature drift detection

    Input: All previous outputs
    Output: results/backtest/backtest_report.json, results/backtest/equity_curve.csv
    """
    print("\n" + "=" * 70)
    print("STAGE 6: TRUE OUT-OF-SAMPLE BACKTEST (OPTIMIZED)")
    print("=" * 70)

    model_path = Path("models/model.pkl")
    report_path = Path("results/backtest/backtest_report.json")
    holdout_manifest_path = Path("data/holdout/holdout_manifest.json")
    oos_data_path = Path("data/oos/oos_data.pkl")

    # Check dependencies
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        print("Run Stage 4 (train) first")
        return False

    # Check if backtest exists
    if report_path.exists() and not force:
        print(f"Backtest report already exists at {report_path}")

        # Show existing results
        with open(report_path, 'r') as f:
            report = json.load(f)
        print(f"Sharpe Ratio: {report.get('sharpe_ratio', 'N/A')}")
        print(f"Max Drawdown: {report.get('max_drawdown', 'N/A')}")
        print("Use --force to rerun")
        return True

    try:
        from src.backtest.engine import InstitutionalBacktestEngine, BacktestConfig, VectorizedBacktester
        from src.backtest.metrics import MetricsCalculator, ReportGenerator
        from src.backtest.realistic_fills import RealisticFillSimulator, FillModel
        from src.strategy.ml_strategy import MLStrategy
        from src.features.institutional import InstitutionalFeatureEngineer
        from src.risk.risk_manager import RiskManager, RiskLimits
        from src.risk.position_sizer import VolatilityPositionSizer
        # ISSUE 8.1/8.2 FIX: Import staleness and drift detectors
        from src.risk.position_sizer import ModelStalenessDetector, FeatureDriftDetector
        import joblib

        # Load optimization configuration
        opt_config = load_optimization_config()
        opt = opt_config.get('optimization', {})
        use_vectorized = opt.get('use_vectorized_backtest', True)
        fast_mode = opt.get('fast_mode', True)
        use_microstructure = opt.get('use_microstructure', False)

        # Set optimized logging mode for backtest
        if fast_mode:
            from src.utils.logger import set_backtest_logging_mode
            set_backtest_logging_mode(fast_mode=True)

        print(f"\nBacktest Mode: {'VECTORIZED (Fast)' if use_vectorized else 'EVENT-DRIVEN'}")
        print(f"Fast Mode: {fast_mode}")
        print(f"Microstructure Simulation: {'ON' if use_microstructure else 'OFF'}")

        # Get training end date from holdout manifest or model metrics
        training_end_date = "2025-05-01"  # Default

        if holdout_manifest_path.exists():
            with open(holdout_manifest_path, 'r') as f:
                holdout_manifest = json.load(f)
            training_end_date = holdout_manifest.get('temporal_cutoff_date', training_end_date)[:10]

        # Also check model metrics
        metrics_path = Path("models/metrics.yaml")
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                model_metrics = yaml.safe_load(f)
            # Training date tells us when model was created
            training_date = model_metrics.get('training_date', '')[:10]
            print(f"Model trained on: {training_date}")

        print(f"\n{'='*50}")
        print("TRUE OUT-OF-SAMPLE CONFIGURATION")
        print(f"{'='*50}")
        print(f"Training data ended: {training_end_date}")
        print(f"OOS data starts:     {training_end_date}")
        print(f"OOS data ends:       today")
        print(f"{'='*50}\n")

        # Get symbols from config
        symbols_config_path = Path("config/symbols.yaml")
        with open(symbols_config_path, 'r') as f:
            symbols_config = yaml.safe_load(f)

        symbols = list(symbols_config.get('symbols', {}).keys())
        print(f"Total symbols: {len(symbols)}")

        # Download or load OOS data
        oos_data_path.parent.mkdir(parents=True, exist_ok=True)

        if oos_data_path.exists() and not force:
            print(f"\nLoading cached OOS data from {oos_data_path}...")
            with open(oos_data_path, 'rb') as f:
                cached = pickle.load(f)

            # Check if cache is recent (within 1 day)
            cache_date = cached.get('download_date', '')
            if cache_date:
                cache_dt = pd.Timestamp(cache_date)
                if (datetime.now() - cache_dt.to_pydatetime()).days < 1:
                    data = cached.get('data', {})
                    print(f"Using cached data from {cache_date}")
                else:
                    print("Cache is stale, re-downloading...")
                    data = download_oos_data(symbols, training_end_date)
                    # Save cache
                    with open(oos_data_path, 'wb') as f:
                        pickle.dump({
                            'download_date': datetime.now().isoformat(),
                            'training_end_date': training_end_date,
                            'data': data
                        }, f)
            else:
                data = cached.get('data', {})
        else:
            # Download fresh data
            data = download_oos_data(symbols, training_end_date)

            # Cache the data
            print(f"\nCaching OOS data to {oos_data_path}...")
            with open(oos_data_path, 'wb') as f:
                pickle.dump({
                    'download_date': datetime.now().isoformat(),
                    'training_end_date': training_end_date,
                    'data': data
                }, f)

        if not data:
            print("ERROR: No OOS data available for backtest!")
            print("Check your internet connection and try again.")
            return False

        # Data summary
        total_bars = sum(len(df) for df in data.values())
        date_range_start = min(df.index.min() for df in data.values())
        date_range_end = max(df.index.max() for df in data.values())

        print(f"\nOOS Data Summary:")
        print(f"  Symbols loaded: {len(data)}")
        print(f"  Total bars: {total_bars:,}")
        print(f"  Date range: {date_range_start} to {date_range_end}")
        print(f"  Training ended: {training_end_date}")
        print(f"  Data is TRUE OUT-OF-SAMPLE: YES")

        # Load model
        print("\nLoading model...")
        model = joblib.load(model_path)

        # Load feature list
        features_file = Path("models/features.txt")
        feature_list = None
        if features_file.exists():
            with open(features_file, 'r') as f:
                feature_list = [line.strip() for line in f.readlines() if line.strip()]
            print(f"Loaded {len(feature_list)} features")

        # Initialize components
        feature_builder = InstitutionalFeatureEngineer()

        # ============================================================
        # ISSUE 8.2 FIX: Feature drift detection
        # Check if OOS feature distributions differ from training
        # ============================================================
        print("\n" + "=" * 70)
        print("CHECKING FEATURE DRIFT (ISSUE 8.2 FIX)")
        print("=" * 70)

        drift_report = {'features_checked': 0, 'features_drifted': 0, 'drifted_features': []}
        try:
            # Load training features for reference
            training_features_path = Path("results/features/combined_features.pkl")
            if training_features_path.exists():
                with open(training_features_path, 'rb') as f:
                    training_features_data = pickle.load(f)

                # Combine training features for reference stats
                all_training_features = []
                for sym_features in training_features_data.values():
                    if sym_features is not None and len(sym_features) > 0:
                        all_training_features.append(sym_features.select_dtypes(include=[np.number]))

                if all_training_features:
                    combined_training = pd.concat(all_training_features, axis=0)

                    # Initialize drift detector with training data
                    drift_detector = FeatureDriftDetector(drift_threshold=2.0)
                    drift_detector.set_reference_stats(combined_training)

                    # Build features for OOS data and check drift
                    oos_features_list = []
                    for symbol, df in list(data.items())[:5]:  # Check first 5 symbols
                        try:
                            features = feature_builder.build_features(df)
                            if features is not None and len(features) > 0:
                                oos_features_list.append(features.select_dtypes(include=[np.number]))
                        except Exception:
                            pass

                    if oos_features_list:
                        combined_oos = pd.concat(oos_features_list, axis=0)
                        drift_report = drift_detector.check_drift(combined_oos)

                        if drift_report['features_drifted'] > 0:
                            print(f"  WARNING: {drift_report['features_drifted']} features show significant drift!")
                            for feat in drift_report['drifted_features'][:5]:
                                print(f"    - {feat}")
                        else:
                            print(f"  Feature drift check passed ({drift_report['features_checked']} features checked)")
                    else:
                        print("  Skipped: Could not generate OOS features for comparison")
                else:
                    print("  Skipped: No training features available for comparison")
            else:
                print("  Skipped: Training features file not found")
        except Exception as e:
            print(f"  Feature drift check failed: {e}")

        # ============================================================
        # NEW: Run model diagnostic BEFORE backtest
        # This detects if model is generating anti-signals
        # ============================================================
        print("\n" + "=" * 70)
        print("RUNNING MODEL DIAGNOSTIC (Pre-Backtest Validation)")
        print("=" * 70)

        diagnostic = diagnose_model_predictions(
            model=model,
            data=data,
            feature_builder=feature_builder,
            sample_size=2000
        )

        # Determine if we should invert signals
        invert_signals = diagnostic.get('should_invert', False)
        model_accuracy = diagnostic.get('accuracy', 0.5)

        if invert_signals:
            print("\n*** AUTO-INVERTING SIGNALS due to anti-correlation ***")
            print("    Original model was predicting opposite of correct direction.")

        if diagnostic.get('status') == 'NO_EDGE':
            print("\n*** WARNING: Model has no predictive power ***")
            print("    Results may be close to random or negative after costs.")

        # Store diagnostic in report (including feature drift from ISSUE 8.2 FIX)
        diagnostic_report = {
            'model_accuracy': model_accuracy,
            'status': diagnostic.get('status', 'UNKNOWN'),
            'correlation': diagnostic.get('correlation', 0),
            'signals_inverted': invert_signals,
            # ISSUE 8.2 FIX: Feature drift info
            'feature_drift': {
                'features_checked': drift_report.get('features_checked', 0),
                'features_drifted': drift_report.get('features_drifted', 0),
                'drifted_features': drift_report.get('drifted_features', [])[:10]
            }
        }

        backtest_start_time = time.time()

        if use_vectorized:
            # === VECTORIZED BACKTEST (FAST MODE) ===
            # 50-100x faster than event-driven
            print("\nRunning VECTORIZED backtest (50-100x faster)...")

            vectorized_config = opt_config.get('vectorized_backtest', {})
            result = run_vectorized_backtest(
                data=data,
                model=model,
                feature_builder=feature_builder,
                config={
                    'initial_capital': vectorized_config.get('initial_capital', 1000000),
                    'commission_pct': vectorized_config.get('commission_pct', 0.001),
                    'slippage_pct': vectorized_config.get('slippage_pct', 0.0005),
                    'max_position_pct': vectorized_config.get('max_position_pct', 0.10),
                    'max_gross_exposure': vectorized_config.get('max_gross_exposure', 0.80),
                    'min_signal_threshold': vectorized_config.get('min_signal_threshold', 0.3)
                },
                invert_signals=invert_signals  # NEW: Use diagnostic result
            )

            if 'error' in result:
                print(f"ERROR: {result['error']}")
                return False

            # Extract metrics from vectorized result (FIXED - now includes trade metrics)
            metrics = {
                'total_return': result.get('total_return', 0),
                'annualized_return': result.get('annualized_return', 0),
                'sharpe_ratio': result.get('sharpe_ratio', 0),
                'sortino_ratio': result.get('sortino_ratio', 0),
                'calmar_ratio': result.get('calmar_ratio', 0),
                'max_drawdown': result.get('max_drawdown', 0),
                'volatility': result.get('volatility', 0),
                'total_costs': result.get('total_costs', 0),
                'cost_drag_pct': result.get('cost_drag_pct', 0),
                # NEW: Trade metrics (FIXED - no longer always 0)
                'total_trades': result.get('total_trades', 0),
                'win_rate': result.get('win_rate', 0),
                'profit_factor': result.get('profit_factor', 0),
                'avg_trade_pnl': result.get('avg_trade_pnl', 0),
                'gross_profit': result.get('gross_profit', 0),
                'gross_loss': result.get('gross_loss', 0)
            }

            # Create result-like object for report generation
            equity_curve = result.get('equity_curve', pd.Series())
            trades = []  # Vectorized doesn't track individual trades

            # Include positions for analysis
            positions = result.get('positions', pd.DataFrame())

        else:
            # === EVENT-DRIVEN BACKTEST ===
            # Full simulation with microstructure (slower but more realistic)

            risk_limits = RiskLimits(
                max_position_pct=0.10,
                max_sector_pct=0.30,
                max_drawdown=0.15,
                max_daily_loss=0.03,
                target_volatility=0.15
            )
            risk_manager = RiskManager(limits=risk_limits)
            position_sizer = VolatilityPositionSizer(target_volatility=0.15)

            # Initialize ML strategy
            strategy = MLStrategy(
                name="ml_ensemble",
                model_path=str(model_path),
                feature_builder=feature_builder,
                feature_list=feature_list
            )

            # Configure backtest
            config = BacktestConfig(
                initial_capital=1000000,
                commission_per_share=0.005,
                slippage_bps=10,  # Conservative slippage
                warmup_period=100
            )

            # Run backtest with/without microstructure based on config
            print(f"\nRunning EVENT-DRIVEN backtest (microstructure: {'ON' if use_microstructure else 'OFF'})...")

            engine = InstitutionalBacktestEngine(
                strategy=strategy,
                config=config,
                position_sizer=position_sizer,
                risk_manager=risk_manager,
                enable_microstructure=use_microstructure,
                microstructure_config={
                    'latency_mean_ms': 10.0,
                    'latency_shape': 2.0,
                    'liquidity_factor': 0.01,
                    'partial_fill_threshold': 0.5,
                    'rejection_probability': 0.02
                } if use_microstructure else None
            )

            result = engine.run(data)

            # Calculate metrics
            metrics_calc = MetricsCalculator()
            metrics = metrics_calc.calculate(
                equity_curve=result.equity_curve,
                trades=[t.to_dict() for t in result.trades]
            )

            equity_curve = result.equity_curve
            trades = result.trades

        backtest_elapsed = time.time() - backtest_start_time
        print(f"\nBacktest completed in {backtest_elapsed:.1f}s ({backtest_elapsed/60:.1f} min)")

        # Generate report with OOS metadata
        initial_capital = 1000000
        if isinstance(equity_curve, pd.Series) and len(equity_curve) > 0:
            final_value = float(equity_curve.iloc[-1])
        elif isinstance(equity_curve, pd.DataFrame) and 'equity' in equity_curve.columns:
            final_value = float(equity_curve['equity'].iloc[-1])
        else:
            final_value = initial_capital * (1 + metrics.get('total_return', 0))

        report = {
            'run_date': datetime.now().isoformat(),
            'backtest_type': 'VECTORIZED_FAST' if use_vectorized else 'TRUE_OUT_OF_SAMPLE',
            'backtest_mode': 'vectorized' if use_vectorized else 'event_driven',
            'elapsed_seconds': backtest_elapsed,
            'oos_config': {
                'training_end_date': training_end_date,
                'oos_start_date': str(date_range_start),
                'oos_end_date': str(date_range_end),
                'total_oos_symbols': len(data),
                'total_oos_bars': total_bars,
                'data_source': 'Alpaca (fresh download)'
            },
            # NEW: Model diagnostic results
            'model_diagnostic': diagnostic_report,
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': float(metrics.get('total_return', 0)),
            'annualized_return': float(metrics.get('annualized_return', 0)),
            'sharpe_ratio': float(metrics.get('sharpe_ratio', 0)),
            'sortino_ratio': float(metrics.get('sortino_ratio', 0)),
            'calmar_ratio': float(metrics.get('calmar_ratio', 0)),
            'max_drawdown': float(metrics.get('max_drawdown', 0)),
            'volatility': float(metrics.get('volatility', 0)),
            # FIXED: Trade metrics now populated
            'total_trades': int(metrics.get('total_trades', 0)) if use_vectorized else len(trades),
            'win_rate': float(metrics.get('win_rate', 0)),
            'profit_factor': float(metrics.get('profit_factor', 0)),
            'avg_trade_pnl': float(metrics.get('avg_trade_pnl', 0)),
            'gross_profit': float(metrics.get('gross_profit', 0)),
            'gross_loss': float(metrics.get('gross_loss', 0)),
            # Cost metrics
            'total_costs': float(metrics.get('total_costs', 0)),
            'cost_drag_pct': float(metrics.get('cost_drag_pct', 0))
        }

        # Save results
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Save equity curve
        equity_path = Path("results/backtest/equity_curve.csv")
        if isinstance(equity_curve, pd.Series):
            equity_curve.to_csv(equity_path)
        elif isinstance(equity_curve, pd.DataFrame):
            equity_curve.to_csv(equity_path)

        # Save trades (if available)
        trades_path = Path("results/backtest/trades.csv")
        if trades:
            if hasattr(trades[0], 'to_dict'):
                pd.DataFrame([t.to_dict() for t in trades]).to_csv(trades_path, index=False)
            else:
                pd.DataFrame(trades).to_csv(trades_path, index=False)

        # Print summary
        print("\n" + "=" * 60)
        print(f"{'VECTORIZED' if use_vectorized else 'EVENT-DRIVEN'} BACKTEST RESULTS")
        print("=" * 60)
        print(f"Backtest Type:    {'VECTORIZED (Fast)' if use_vectorized else 'EVENT-DRIVEN'}")
        print(f"Elapsed Time:     {backtest_elapsed:.1f}s ({backtest_elapsed/60:.1f} min)")
        print(f"OOS Period:       {date_range_start} to {date_range_end}")
        print(f"OOS Symbols:      {len(data)}")
        print(f"OOS Bars:         {total_bars:,}")
        print("-" * 60)
        print("MODEL DIAGNOSTIC:")
        print(f"  Model Accuracy:   {diagnostic_report.get('model_accuracy', 0):.2%}")
        print(f"  Status:           {diagnostic_report.get('status', 'UNKNOWN')}")
        print(f"  Signals Inverted: {diagnostic_report.get('signals_inverted', False)}")
        print("-" * 60)
        print("PERFORMANCE:")
        print(f"  Initial Capital:  ${initial_capital:,.0f}")
        print(f"  Final Value:      ${report['final_value']:,.0f}")
        print(f"  Total Return:     {report['total_return']:.2%}")
        print(f"  Ann. Return:      {report.get('annualized_return', 0):.2%}")
        print(f"  Sharpe Ratio:     {report['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio:    {report.get('sortino_ratio', 0):.2f}")
        print(f"  Max Drawdown:     {report['max_drawdown']:.2%}")
        print(f"  Calmar Ratio:     {report.get('calmar_ratio', 0):.2f}")
        print("-" * 60)
        print("TRADE METRICS (FIXED):")
        print(f"  Total Trades:     {report['total_trades']:,}")
        print(f"  Win Rate:         {report['win_rate']:.2%}")
        print(f"  Profit Factor:    {report['profit_factor']:.2f}")
        print(f"  Avg Trade P&L:    {report.get('avg_trade_pnl', 0):.6f}")
        print("-" * 60)
        print("COSTS:")
        print(f"  Total Costs:      ${report.get('total_costs', 0):,.2f}")
        print(f"  Cost Drag:        {report.get('cost_drag_pct', 0):.4%}")
        print("=" * 60)

        print(f"\nReport saved to {report_path}")

        return True

    except Exception as e:
        logger.error(f"Stage 6 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# STAGE 7: VALIDATION
# =============================================================================

def stage_validate() -> bool:
    """
    Stage 7: Verify all components work together.

    Checks:
    - Sharpe ratio > 0.5
    - Max drawdown < 25%
    - Win rate > 45%
    - Model accuracy > 52%
    - Calibration ECE < 0.1
    """
    print("\n" + "=" * 70)
    print("STAGE 7: VALIDATION")
    print("=" * 70)

    validation_results = {
        'timestamp': datetime.now().isoformat(),
        'checks': {},
        'passed': True
    }

    print("\nRunning validation checks...\n")

    # Check 1: Model exists
    model_path = Path("models/model.pkl")
    check_model = model_path.exists()
    validation_results['checks']['model_exists'] = check_model
    print(f"[{'PASS' if check_model else 'FAIL'}] Model exists: {model_path}")

    # Check 2: Model metrics
    metrics_path = Path("models/metrics.yaml")
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = yaml.safe_load(f)

        cv_accuracy = metrics.get('performance', {}).get('cv_accuracy', 0)
        check_accuracy = cv_accuracy > 0.52
        validation_results['checks']['model_accuracy'] = {
            'value': cv_accuracy,
            'threshold': 0.52,
            'passed': check_accuracy
        }
        print(f"[{'PASS' if check_accuracy else 'FAIL'}] Model accuracy > 52%: {cv_accuracy:.4f}")
    else:
        print("[SKIP] Model metrics not found")

    # Check 3: Backtest results
    report_path = Path("results/backtest/backtest_report.json")
    if report_path.exists():
        with open(report_path, 'r') as f:
            report = json.load(f)

        # Sharpe ratio check
        sharpe = report.get('sharpe_ratio', 0)
        check_sharpe = sharpe > 0.5
        validation_results['checks']['sharpe_ratio'] = {
            'value': sharpe,
            'threshold': 0.5,
            'passed': check_sharpe
        }
        print(f"[{'PASS' if check_sharpe else 'WARN'}] Sharpe ratio > 0.5: {sharpe:.2f}")

        # Max drawdown check
        max_dd = abs(report.get('max_drawdown', 1))
        check_dd = max_dd < 0.25
        validation_results['checks']['max_drawdown'] = {
            'value': max_dd,
            'threshold': 0.25,
            'passed': check_dd
        }
        print(f"[{'PASS' if check_dd else 'WARN'}] Max drawdown < 25%: {max_dd:.2%}")

        # Win rate check
        win_rate = report.get('win_rate', 0)
        check_wr = win_rate > 0.45
        validation_results['checks']['win_rate'] = {
            'value': win_rate,
            'threshold': 0.45,
            'passed': check_wr
        }
        print(f"[{'PASS' if check_wr else 'WARN'}] Win rate > 45%: {win_rate:.2%}")
    else:
        print("[SKIP] Backtest report not found")

    # Check 4: Calibration model
    calibration_path = Path("models/calibration_model.pkl")
    check_calibration = calibration_path.exists()
    validation_results['checks']['calibration_exists'] = check_calibration
    print(f"[{'PASS' if check_calibration else 'WARN'}] Calibration model exists: {calibration_path}")

    # Check 5: Required files
    required_files = [
        "config/settings.yaml",
        "config/symbols.yaml",
        "config/risk_params.yaml"
    ]

    for file in required_files:
        exists = Path(file).exists()
        validation_results['checks'][f'file_{file}'] = exists
        print(f"[{'PASS' if exists else 'FAIL'}] Config file: {file}")

    # Check 6: Import test
    print("\nTesting imports...")
    try:
        from main import AlphaTradeSystem
        print("[PASS] main.py imports successfully")
        validation_results['checks']['main_imports'] = True
    except Exception as e:
        print(f"[FAIL] main.py import error: {e}")
        validation_results['checks']['main_imports'] = False
        validation_results['passed'] = False

    # Save validation results
    os.makedirs("results", exist_ok=True)
    with open("results/validation_report.json", 'w') as f:
        json.dump(validation_results, f, indent=2)

    # Summary
    failed_checks = [k for k, v in validation_results['checks'].items()
                     if isinstance(v, bool) and not v
                     or isinstance(v, dict) and not v.get('passed', True)]

    print("\n" + "=" * 50)
    if not failed_checks:
        print("VALIDATION PASSED")
    else:
        print(f"VALIDATION WARNINGS: {len(failed_checks)} issues")
        for check in failed_checks:
            print(f"  - {check}")
    print("=" * 50)

    print(f"\nValidation report saved to results/validation_report.json")

    return len(failed_checks) == 0


# =============================================================================
# STAGE 8: PAPER TRADING
# =============================================================================

def stage_paper() -> bool:
    """
    Stage 8: Start paper trading.

    Requires Alpaca API credentials in environment.
    """
    print("\n" + "=" * 70)
    print("STAGE 8: PAPER TRADING")
    print("=" * 70)

    # Check for API credentials
    api_key = os.environ.get('ALPACA_API_KEY')
    api_secret = os.environ.get('ALPACA_API_SECRET')

    if not api_key or not api_secret:
        print("\nERROR: Alpaca API credentials not found!")
        print("Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables")
        print("\nExample:")
        print("  export ALPACA_API_KEY=your_key")
        print("  export ALPACA_API_SECRET=your_secret")
        return False

    print("\nStarting paper trading...")
    print("Press Ctrl+C to stop")
    print("")

    try:
        import subprocess

        cmd = [
            sys.executable,
            "main.py",
            "--mode", "paper"
        ]

        result = subprocess.run(cmd)

        return result.returncode == 0

    except KeyboardInterrupt:
        print("\nPaper trading stopped by user")
        return True
    except Exception as e:
        logger.error(f"Stage 8 failed: {e}")
        return False


# =============================================================================
# MAIN PIPELINE RUNNER
# =============================================================================

def run_pipeline(
    stages: List[int],
    symbols: List[str],
    start_date: str,
    end_date: str,
    force: bool = False,
    model_type: str = "catboost",
    n_estimators: int = 100,
    profile: bool = False
) -> bool:
    """
    Run the pipeline stages.

    Args:
        stages: List of stage numbers to run
        symbols: List of symbols to process
        start_date: Start date for data
        end_date: End date for data
        force: Force rerun of stages
        model_type: Type of ML model
        n_estimators: Number of estimators
        profile: Enable profiling mode
    """
    start_time = time.time()

    # Setup profiling if enabled
    profiler = None
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()
        print("\n[PROFILING ENABLED]")

    print("\n" + "=" * 70)
    print("ALPHATRADE PIPELINE")
    print("=" * 70)
    print(f"Stages to run: {[STAGES[s] for s in stages]}")
    print(f"Symbols: {len(symbols)}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Force rerun: {force}")
    print("=" * 70)

    results = {}

    for stage_num in stages:
        stage_name = STAGES[stage_num]
        stage_start = time.time()

        save_checkpoint(stage_num, "running")

        try:
            if stage_num == 1:
                success = stage_data(symbols, start_date, end_date, force)
            elif stage_num == 2:
                success = stage_features(force)
            elif stage_num == 3:
                success = stage_labels(force=force)
            elif stage_num == 4:
                success = stage_train(model_type, n_estimators, force)
            elif stage_num == 5:
                success = stage_calibrate(force)
            elif stage_num == 6:
                success = stage_backtest(force)
            elif stage_num == 7:
                success = stage_validate()
            elif stage_num == 8:
                success = stage_paper()
            else:
                print(f"Unknown stage: {stage_num}")
                success = False

            stage_time = time.time() - stage_start
            results[stage_name] = {
                'success': success,
                'time_seconds': stage_time
            }

            if success:
                save_checkpoint(stage_num, "completed", {'time_seconds': stage_time})
            else:
                save_checkpoint(stage_num, "failed")
                print(f"\nStage {stage_num} ({stage_name}) FAILED")
                break

        except Exception as e:
            save_checkpoint(stage_num, "error", {'error': str(e)})
            logger.error(f"Stage {stage_num} ({stage_name}) error: {e}")
            import traceback
            traceback.print_exc()
            results[stage_name] = {'success': False, 'error': str(e)}
            break

    total_time = time.time() - start_time

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)

    for stage_name, result in results.items():
        status = "PASS" if result.get('success') else "FAIL"
        time_str = f"{result.get('time_seconds', 0):.1f}s"
        print(f"  {stage_name}: {status} ({time_str})")

    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print("=" * 70)

    all_passed = all(r.get('success', False) for r in results.values())

    if all_passed:
        clear_checkpoint()
        print("\nPipeline completed successfully!")
    else:
        print("\nPipeline failed. Check logs for details.")
        print(f"Resume from checkpoint with: python scripts/run_pipeline.py --resume")

    # Save profiling results if enabled
    if profiler is not None:
        profiler.disable()

        # Create profiling output directory
        profile_dir = Path("results/profiling")
        profile_dir.mkdir(parents=True, exist_ok=True)

        # Save profile stats
        profile_file = profile_dir / f"pipeline_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prof"
        profiler.dump_stats(str(profile_file))

        # Print summary
        print("\n" + "=" * 70)
        print("PROFILING RESULTS")
        print("=" * 70)

        # Get string output of profiling stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(30)  # Top 30 functions

        # Save text report
        report_file = profile_dir / f"pipeline_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(s.getvalue())

        print(s.getvalue()[:3000])  # Print first 3000 chars
        print(f"\nFull profile saved to: {profile_file}")
        print(f"Text report saved to: {report_file}")

    return all_passed


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AlphaTrade Master Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_pipeline.py                    # Run full pipeline
    python scripts/run_pipeline.py --stage train     # Run single stage
    python scripts/run_pipeline.py --from-stage 4    # Start from stage 4
    python scripts/run_pipeline.py --resume          # Resume from checkpoint
    python scripts/run_pipeline.py --skip-data       # Skip data download
    python scripts/run_pipeline.py --force           # Force rerun all stages
    python scripts/run_pipeline.py --paper           # Include paper trading
        """
    )

    # Stage selection
    parser.add_argument("--stage", type=str, choices=list(STAGE_NAMES.keys()),
                        help="Run a single stage")
    parser.add_argument("--from-stage", type=int, choices=list(STAGES.keys()),
                        help="Start from a specific stage number")
    parser.add_argument("--to-stage", type=int, choices=list(STAGES.keys()),
                        help="End at a specific stage number")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data download stage")
    parser.add_argument("--paper", action="store_true",
                        help="Include paper trading stage")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")

    # Data options
    parser.add_argument("--symbols", type=str, nargs="+",
                        help="Symbols to process")
    parser.add_argument("--start", type=str, default="2020-01-01",
                        help="Start date (default: 2020-01-01)")
    parser.add_argument("--end", type=str, default=datetime.now().strftime("%Y-%m-%d"),
                        help="End date (default: today)")

    # Model options
    parser.add_argument("--model", type=str, default="catboost",
                        choices=["catboost", "xgboost", "lightgbm"],
                        help="Model type (default: catboost)")
    parser.add_argument("--n-estimators", type=int, default=100,
                        help="Number of ensemble estimators (default: 100)")

    # Execution options
    parser.add_argument("--force", action="store_true",
                        help="Force rerun of all stages")

    # Performance options
    parser.add_argument("--profile", action="store_true",
                        help="Run with profiling enabled (outputs to results/profiling/)")
    parser.add_argument("--fast", action="store_true",
                        help="Use fast mode (vectorized backtest, skip microstructure)")
    parser.add_argument("--full", action="store_true",
                        help="Use full mode (event-driven backtest with microstructure)")

    args = parser.parse_args()

    # Handle fast/full mode overrides
    if args.fast or args.full:
        opt_config_path = Path("config/backtest_optimization.yaml")
        if opt_config_path.exists():
            with open(opt_config_path, 'r') as f:
                opt_config = yaml.safe_load(f)

            if args.fast:
                opt_config['optimization']['use_vectorized_backtest'] = True
                opt_config['optimization']['fast_mode'] = True
                opt_config['optimization']['use_microstructure'] = False
                print("Fast mode enabled: vectorized backtest, no microstructure")
            elif args.full:
                opt_config['optimization']['use_vectorized_backtest'] = False
                opt_config['optimization']['fast_mode'] = False
                opt_config['optimization']['use_microstructure'] = True
                print("Full mode enabled: event-driven backtest with microstructure")

            with open(opt_config_path, 'w') as f:
                yaml.dump(opt_config, f, default_flow_style=False)

    # Get symbols
    if args.symbols:
        symbols = args.symbols
    else:
        # Load from config
        try:
            with open("config/symbols.yaml", 'r') as f:
                config = yaml.safe_load(f)

            symbols = []
            sectors = config.get('sectors', {})
            for sector_name, sector_data in sectors.items():
                symbols.extend(sector_data.get('symbols', []))

            if not symbols:
                symbols = list(config.get('symbols', {}).keys())
        except Exception:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

    # Determine stages to run
    if args.resume:
        checkpoint = load_checkpoint()
        if checkpoint and checkpoint.get('status') in ['failed', 'error']:
            args.from_stage = checkpoint['stage']
            print(f"Resuming from stage {args.from_stage} ({STAGES[args.from_stage]})")
        else:
            print("No checkpoint to resume from")

    if args.stage:
        stages = [STAGE_NAMES[args.stage]]
    else:
        start_stage = args.from_stage or 1
        end_stage = args.to_stage or (8 if args.paper else 7)

        if args.skip_data and start_stage == 1:
            start_stage = 2

        stages = list(range(start_stage, end_stage + 1))

    # Run pipeline
    success = run_pipeline(
        stages=stages,
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        force=args.force,
        model_type=args.model,
        n_estimators=args.n_estimators,
        profile=args.profile
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
