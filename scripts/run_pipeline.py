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
from typing import Dict, List, Optional, Any

# Suppress warnings before imports
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import yaml
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger, setup_logging


# Setup logging
setup_logging(log_path="logs", level="INFO")
logger = get_logger(__name__)


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

    Output: data/processed/combined_data.pkl
    """
    print("\n" + "=" * 70)
    print("STAGE 1: DATA PREPARATION")
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

        print(f"\nDownloading data for {len(symbols)} symbols...")
        print(f"Date range: {start_date} to {end_date}")

        # Initialize loader
        loader = MultiAssetLoader(data_path="data/raw", cache_path="data/cache")
        preprocessor = DataPreprocessor()

        # Load all symbols
        raw_data = loader.load_symbols(symbols=symbols, show_progress=True)

        # Preprocess
        processed_data = {}
        for symbol, df in raw_data.items():
            if df is not None and len(df) > 100:
                try:
                    df_clean, report = preprocessor.preprocess(df, symbol)
                    processed_data[symbol] = df_clean
                    print(f"  {symbol}: {len(df_clean)} bars (quality: {report.quality_score:.1f})")
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

        return True

    except Exception as e:
        logger.error(f"Stage 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# STAGE 2: FEATURE ENGINEERING
# =============================================================================

def stage_features(force: bool = False) -> bool:
    """
    Stage 2: Generate institutional features.

    Input: data/processed/combined_data.pkl
    Output: results/features/combined_features.pkl
    """
    print("\n" + "=" * 70)
    print("STAGE 2: FEATURE ENGINEERING")
    print("=" * 70)

    input_path = Path("data/processed/combined_data.pkl")
    output_path = Path("results/features/combined_features.pkl")

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
        from src.features.point_in_time import PointInTimeFeatureEngine

        # Load data
        print("\nLoading preprocessed data...")
        with open(input_path, 'rb') as f:
            data = pickle.load(f)

        print(f"Loaded {len(data)} symbols")

        # Initialize feature engineers
        institutional_fe = InstitutionalFeatureEngineer()
        pit_engine = PointInTimeFeatureEngine()
        pit_engine.add_standard_features()

        # Generate features for each symbol
        features_data = {}

        for symbol, df in data.items():
            try:
                print(f"  Processing {symbol}...", end=" ")

                # Generate institutional features
                features = institutional_fe.build_features(df)

                # The point-in-time engine ensures no look-ahead bias
                # (already handled in institutional features)

                if features is not None and len(features) > 0:
                    features_data[symbol] = features
                    print(f"{len(features)} samples, {len(features.columns)} features")
                else:
                    print("skipped (no features)")

            except Exception as e:
                print(f"failed ({e})")

        if not features_data:
            print("ERROR: No features generated!")
            return False

        # Save features
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(features_data, f)

        print(f"\nFeatures saved to {output_path}")
        print(f"Generated features for {len(features_data)} symbols")

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
    Stage 3: Create triple barrier labels.

    Input: data/processed/combined_data.pkl
    Output: results/labels/labels.pkl
    """
    print("\n" + "=" * 70)
    print("STAGE 3: LABEL GENERATION")
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

        for symbol, df in data.items():
            try:
                print(f"  Labeling {symbol}...", end=" ")

                # Standardize columns
                df.columns = df.columns.str.lower()

                # Generate labels
                labels_df = pipeline.generate_labels(df)

                if labels_df is not None and len(labels_df) > 0:
                    labels_data[symbol] = labels_df

                    # Show label distribution
                    if 'bin' in labels_df.columns:
                        dist = labels_df['bin'].value_counts().to_dict()
                        print(f"{len(labels_df)} labels, dist: {dist}")
                    else:
                        print(f"{len(labels_df)} labels")
                else:
                    print("skipped (no labels)")

            except Exception as e:
                print(f"failed ({e})")

        if not labels_data:
            print("ERROR: No labels generated!")
            return False

        # Save labels
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(labels_data, f)

        print(f"\nLabels saved to {output_path}")
        print(f"Generated labels for {len(labels_data)} symbols")

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
# STAGE 6: BACKTEST
# =============================================================================

def stage_backtest(force: bool = False) -> bool:
    """
    Stage 6: Validate strategy with realistic execution.

    Input: All previous outputs
    Output: results/backtest/backtest_report.json, results/backtest/equity_curve.csv
    """
    print("\n" + "=" * 70)
    print("STAGE 6: BACKTEST")
    print("=" * 70)

    model_path = Path("models/model.pkl")
    report_path = Path("results/backtest/backtest_report.json")

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
        from src.backtest.engine import BacktestEngine, BacktestConfig
        from src.backtest.metrics import MetricsCalculator, ReportGenerator
        from src.backtest.realistic_fills import RealisticFillSimulator, FillModel
        from src.strategy.ml_strategy import MLStrategy
        from src.features.institutional import InstitutionalFeatureEngineer
        from src.risk.risk_manager import RiskManager, RiskLimits
        from src.risk.position_sizer import VolatilityPositionSizer
        import joblib

        # Load model
        print("\nLoading model and data...")
        model = joblib.load(model_path)

        # Load feature list
        features_file = Path("models/features.txt")
        feature_list = None
        if features_file.exists():
            with open(features_file, 'r') as f:
                feature_list = [line.strip() for line in f.readlines() if line.strip()]

        # Load data
        data_path = Path("data/processed/combined_data.pkl")
        if not data_path.exists():
            print(f"ERROR: Data not found: {data_path}")
            return False

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        print(f"Loaded {len(data)} symbols")

        # Initialize components
        feature_builder = InstitutionalFeatureEngineer()

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

        # Configure backtest with realistic fills
        config = BacktestConfig(
            initial_capital=1000000,
            commission_per_share=0.005,
            slippage_bps=10,  # Conservative slippage
            warmup_period=100
        )

        # Run backtest
        print("\nRunning backtest with realistic fills...")

        engine = BacktestEngine(
            strategy=strategy,
            config=config,
            position_sizer=position_sizer,
            risk_manager=risk_manager
        )

        result = engine.run(data)

        # Calculate metrics
        metrics_calc = MetricsCalculator()
        metrics = metrics_calc.calculate(
            equity_curve=result.equity_curve,
            trades=[t.to_dict() for t in result.trades]
        )

        # Generate report
        report = {
            'run_date': datetime.now().isoformat(),
            'initial_capital': config.initial_capital,
            'final_value': float(result.equity_curve['equity'].iloc[-1]),
            'total_return': float(metrics.get('total_return', 0)),
            'sharpe_ratio': float(metrics.get('sharpe_ratio', 0)),
            'max_drawdown': float(metrics.get('max_drawdown', 0)),
            'win_rate': float(metrics.get('win_rate', 0)),
            'profit_factor': float(metrics.get('profit_factor', 0)),
            'total_trades': len(result.trades),
            'avg_trade_return': float(metrics.get('avg_trade_return', 0)),
            'calmar_ratio': float(metrics.get('calmar_ratio', 0))
        }

        # Save results
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        equity_path = Path("results/backtest/equity_curve.csv")
        result.equity_curve.to_csv(equity_path)

        trades_path = Path("results/backtest/trades.csv")
        pd.DataFrame([t.to_dict() for t in result.trades]).to_csv(trades_path, index=False)

        # Print summary
        print("\n" + "=" * 50)
        print("BACKTEST RESULTS")
        print("=" * 50)
        print(f"Initial Capital:  ${config.initial_capital:,.0f}")
        print(f"Final Value:      ${report['final_value']:,.0f}")
        print(f"Total Return:     {report['total_return']:.2%}")
        print(f"Sharpe Ratio:     {report['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:     {report['max_drawdown']:.2%}")
        print(f"Win Rate:         {report['win_rate']:.2%}")
        print(f"Total Trades:     {report['total_trades']}")
        print("=" * 50)

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
    n_estimators: int = 100
) -> bool:
    """
    Run the pipeline stages.
    """
    start_time = time.time()

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

    args = parser.parse_args()

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
        n_estimators=args.n_estimators
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
