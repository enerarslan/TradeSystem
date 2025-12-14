"""
Institutional-Grade Model Training Pipeline
============================================

Based on "Advances in Financial Machine Learning" by Marcos Lopez de Prado.

This is the SINGLE, BEST training script that implements:

1. INSTITUTIONAL FEATURES
   - Fractional Differentiation (preserves memory while achieving stationarity)
   - Microstructure (VPIN, Kyle's Lambda, Amihud, OFI)
   - HMM Regime Detection (bull/bear/neutral)

2. META-LABELING
   - Primary Model: Simple trend-following signal
   - Secondary Model: ML filter that learns "should I take this trade?"
   - Much easier than predicting direction directly

3. SAMPLE WEIGHTING
   - Uniqueness-based weights (down-weight overlapping labels)
   - Time decay weights (recent data more important)

4. PURGED K-FOLD CV
   - Prevents information leakage
   - Embargo period after test sets

5. BAGGING ENSEMBLE
   - 100+ small CatBoost/XGBoost estimators
   - Sequential bootstrap for dependent data
   - Reduces variance significantly

Usage:
    python scripts/train_models.py                    # Default: all symbols
    python scripts/train_models.py --symbol AAPL     # Single symbol
    python scripts/train_models.py --no-meta-label   # Skip meta-labeling
    python scripts/train_models.py --n-estimators 200  # More estimators
"""

import sys
import os
from pathlib import Path

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import argparse
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger, setup_logging
from src.data.loader import MultiAssetLoader
from src.data.labeling import (
    TripleBarrierLabeler, TripleBarrierConfig,
    get_sample_weights, get_time_decay_weights, combine_weights
)
from src.features.institutional import (
    InstitutionalFeatureEngineer, InstitutionalFeatureConfig
)
from src.models.meta_labeling import (
    MetaLabelingPipeline, MetaLabelingConfig,
    TrendFollowingSignal, BollingerBreakoutSignal, CompositeSignal
)
from src.models.institutional_training import (
    InstitutionalTrainingPipeline, InstitutionalTrainingConfig,
    BaggingEnsemble, SequentialBootstrap,
    train_institutional_model, create_model_factory
)

logger = get_logger(__name__)


def load_triple_barrier_params(config_path: str = "config/triple_barrier_params.yaml") -> dict:
    """Load symbol-specific triple barrier parameters."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Triple barrier params not found at {config_path}, using defaults")
        return {}


def get_symbol_params(symbol: str, tb_params: dict) -> Tuple[float, float, int]:
    """
    Get symbol-specific PT/SL ratios and max holding period from calibrated params.

    Returns:
        (profit_target_atr_mult, stop_loss_atr_mult, max_holding_period)
    """
    symbols_config = tb_params.get('symbols', {})
    default_params = tb_params.get('default_params', {})

    if symbol in symbols_config:
        sym_params = symbols_config[symbol]
        pt = sym_params.get('profit_target_atr_mult', default_params.get('profit_target_atr_mult', 1.5))
        sl = sym_params.get('stop_loss_atr_mult', default_params.get('stop_loss_atr_mult', 1.0))
        mhp = sym_params.get('max_holding_period', default_params.get('max_holding_period', 10))
        return (pt, sl, mhp)
    else:
        # Use defaults
        return (
            default_params.get('profit_target_atr_mult', 1.5),
            default_params.get('stop_loss_atr_mult', 1.0),
            default_params.get('max_holding_period', 10)
        )


def load_config(path: str) -> dict:
    """Load YAML configuration."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_symbols_from_config(config_path: str = "config/symbols.yaml") -> List[str]:
    """Extract all symbols from config file."""
    config = load_config(config_path)
    symbols = []

    # Extract from sectors
    sectors = config.get('sectors', {})
    for sector_name, sector_data in sectors.items():
        sector_symbols = sector_data.get('symbols', [])
        symbols.extend(sector_symbols)

    # Fallback to symbols dict
    if not symbols:
        symbols_dict = config.get('symbols', {})
        symbols = list(symbols_dict.keys())

    return symbols


def process_symbol(
    symbol: str,
    loader: MultiAssetLoader,
    feature_engineer: InstitutionalFeatureEngineer,
    use_meta_labeling: bool = True,
    pt_sl_ratio: Tuple[float, float] = (1.5, 1.0),
    max_holding_period: int = 20,
    tb_params: Optional[dict] = None
) -> Optional[Dict]:
    """
    Process a single symbol: load data, generate features, create labels.

    Uses symbol-specific parameters from triple_barrier_params.yaml if available.

    Returns dict with X, y, weights, events or None if failed.
    """
    try:
        # Get symbol-specific parameters from calibrated config
        if tb_params:
            pt, sl, mhp = get_symbol_params(symbol, tb_params)
            pt_sl_ratio = (pt, sl)
            max_holding_period = mhp
            logger.debug(f"{symbol}: Using calibrated params PT={pt}, SL={sl}, MHP={mhp}")

        # Load data
        df = loader.loader.load(symbol)
        if df is None or len(df) < 500:
            logger.warning(f"{symbol}: Insufficient data ({len(df) if df is not None else 0} bars)")
            return None

        # Standardize columns
        df.columns = df.columns.str.lower()

        # Verify required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(f"{symbol}: Missing columns: {missing}")
            return None

        # Generate institutional features
        features = feature_engineer.build_features(df)

        if use_meta_labeling:
            # META-LABELING APPROACH
            # Primary signal: trend-following
            meta_config = MetaLabelingConfig(
                pt_sl_ratio=pt_sl_ratio,
                max_holding_period=max_holding_period,
                use_sample_weights=True,
                time_decay_factor=0.5,
                primary_confidence_threshold=0.0  # Don't filter by confidence
            )

            # Use trend-following signal without ATR filter for more signals
            primary_signal = TrendFollowingSignal(
                fast_period=10,  # Faster MA for more signals
                slow_period=30,
                atr_filter=False  # Disable ATR filter to get more signals
            )

            pipeline = MetaLabelingPipeline(meta_config, primary_signal)
            labels_df = pipeline.generate_labels(df)

            if len(labels_df) == 0:
                logger.warning(f"{symbol}: No meta-labels generated")
                return None

            # Prepare training data
            X, y, weights = pipeline.prepare_training_data(features, labels_df)
            events = labels_df[['t1']].copy()

        else:
            # STANDARD TRIPLE BARRIER (without meta-labeling)
            tb_config = TripleBarrierConfig(
                pt_sl_ratio=pt_sl_ratio,
                max_holding_period=max_holding_period,
                volatility_lookback=20
            )
            labeler = TripleBarrierLabeler(tb_config)

            # Skip warmup period
            t_events = df.index[200:]

            events = labeler.get_events(
                close=df['close'],
                t_events=t_events,
                pt_sl=pt_sl_ratio
            )

            if len(events) == 0:
                logger.warning(f"{symbol}: No triple barrier events")
                return None

            # Align features with events
            common_idx = features.index.intersection(events.index)
            X = features.loc[common_idx]
            y = events.loc[common_idx, 'bin_label']

            # Sample weights
            weights = get_sample_weights(events.loc[common_idx], df['close'])

        # Drop NaN
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        weights = weights.reindex(X.index).fillna(1.0)
        events = events.reindex(X.index)

        if len(X) < 100:
            logger.warning(f"{symbol}: Too few samples after cleaning ({len(X)})")
            return None

        return {
            'X': X,
            'y': y,
            'weights': weights,
            'events': events,
            'close': df['close'],
            'n_samples': len(X)
        }

    except Exception as e:
        logger.error(f"{symbol}: Error - {e}")
        return None


def train_model(
    symbols: List[str],
    output_dir: Path,
    use_meta_labeling: bool = True,
    model_type: str = 'catboost',
    n_estimators: int = 100,
    n_splits: int = 3,  # Changed from 5 to 3 for faster training
    embargo_pct: float = 0.05,
    pt_sl_ratio: Tuple[float, float] = (1.5, 1.0),
    max_holding_period: int = 20
):
    """
    Main training function.

    Args:
        symbols: List of symbols to train on
        output_dir: Directory to save models
        use_meta_labeling: Use meta-labeling (recommended)
        model_type: 'catboost', 'xgboost', or 'lightgbm'
        n_estimators: Number of bagging estimators
        n_splits: Number of CV folds (default: 3)
        embargo_pct: Embargo percentage for purged CV
        pt_sl_ratio: (profit_take, stop_loss) ATR multipliers (overridden by triple_barrier_params.yaml)
        max_holding_period: Maximum bars to hold position (overridden by triple_barrier_params.yaml)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load symbol-specific triple barrier parameters
    tb_params = load_triple_barrier_params()
    if tb_params:
        logger.info(f"Loaded triple barrier params for {len(tb_params.get('symbols', {}))} symbols")

    # Initialize
    loader = MultiAssetLoader(data_path="data/processed")
    feature_config = InstitutionalFeatureConfig()
    feature_engineer = InstitutionalFeatureEngineer(feature_config)

    # =========================================================================
    # STEP 1: Process all symbols
    # =========================================================================
    print("\n" + "=" * 70)
    print("INSTITUTIONAL TRAINING PIPELINE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Meta-labeling: {'Yes' if use_meta_labeling else 'No'}")
    print(f"  Model type: {model_type}")
    print(f"  Bagging estimators: {n_estimators}")
    print(f"  CV folds: {n_splits}, Embargo: {embargo_pct:.0%}")
    print(f"  Default PT/SL ratio: {pt_sl_ratio}")
    print(f"  Default max holding: {max_holding_period} bars")
    if tb_params and tb_params.get('symbols'):
        print(f"  Using calibrated params: config/triple_barrier_params.yaml ({len(tb_params['symbols'])} symbols)")

    print(f"\n[1/4] Processing {len(symbols)} symbols...")

    # Use parallel processing for feature engineering (biggest speedup)
    from joblib import Parallel, delayed
    import multiprocessing

    n_jobs = min(multiprocessing.cpu_count() - 1, 8)  # Leave 1 core free, max 8

    def _process_symbol_wrapper(symbol):
        """Wrapper for parallel processing."""
        try:
            result = process_symbol(
                symbol=symbol,
                loader=loader,
                feature_engineer=feature_engineer,
                use_meta_labeling=use_meta_labeling,
                pt_sl_ratio=pt_sl_ratio,
                max_holding_period=max_holding_period,
                tb_params=tb_params  # Use calibrated symbol-specific params
            )
            if result is not None:
                result['X']['_symbol'] = symbol
                return result
        except Exception as e:
            logger.warning(f"{symbol} failed: {e}")
        return None

    # Parallel processing - much faster for 46 symbols
    all_data = Parallel(n_jobs=n_jobs, prefer="threads", verbose=1)(
        delayed(_process_symbol_wrapper)(symbol) for symbol in symbols
    )

    # Filter out None results
    all_data = [d for d in all_data if d is not None]

    for d in all_data:
        symbol = d['X']['_symbol'].iloc[0] if '_symbol' in d['X'].columns else 'unknown'
        logger.info(f"{symbol}: {d['n_samples']} samples")

    if not all_data:
        raise ValueError("No data processed for any symbol!")

    print(f"      Successfully processed {len(all_data)}/{len(symbols)} symbols")

    # =========================================================================
    # STEP 2: Combine all data
    # =========================================================================
    print(f"\n[2/4] Combining data...")

    X_combined = pd.concat([d['X'] for d in all_data], axis=0)
    y_combined = pd.concat([d['y'] for d in all_data], axis=0)
    weights_combined = pd.concat([d['weights'] for d in all_data], axis=0)

    # Remove symbol column and non-numeric
    symbol_col = X_combined.pop('_symbol')
    X = X_combined.select_dtypes(include=[np.number])

    # Drop columns with too many NaN
    nan_pct = X.isna().mean()
    good_cols = nan_pct[nan_pct < 0.3].index.tolist()
    X = X[good_cols]

    # Fill remaining NaN
    X = X.fillna(method='ffill').fillna(0)

    # Normalize weights
    weights_combined = weights_combined / weights_combined.mean()
    weights_combined = weights_combined.clip(0.1, 10.0)

    print(f"      Total samples: {len(X):,}")
    print(f"      Features: {len(X.columns)}")
    print(f"      Label distribution: {y_combined.value_counts().to_dict()}")

    # =========================================================================
    # STEP 3: Train model
    # =========================================================================
    print(f"\n[3/4] Training {model_type} ensemble...")

    training_config = InstitutionalTrainingConfig(
        n_splits=n_splits,
        embargo_pct=embargo_pct,
        n_estimators=n_estimators,
        max_samples=0.5,
        max_features=0.7,
        use_sequential_bootstrap=True,
        use_sample_weights=True,
        compute_importance=True
    )

    # Model-specific parameters
    model_params = {
        'catboost': {
            'iterations': 1500,
            'depth': 5,
            'learning_rate': 0.03,
            'l2_leaf_reg': 3.0,
            'verbose': 0,
            'task_type': 'GPU',
            'devices': '0'
        },
        'xgboost': {
            'n_estimators': 500,
            'max_depth': 5,
            'learning_rate': 0.03,
            'reg_lambda': 1.0,
            'verbosity': 0
        },
        'lightgbm': {
            'n_estimators': 500,
            'max_depth': 5,
            'learning_rate': 0.03,
            'reg_lambda': 1.0,
            'verbose': -1
        }
    }

    result = train_institutional_model(
        X=X,
        y=y_combined,
        events=None,
        close=None,
        model_type=model_type,
        use_ensemble=True,
        config=training_config,
        **model_params.get(model_type, {})
    )

    # =========================================================================
    # STEP 4: Save results
    # =========================================================================
    print(f"\n[4/4] Saving results...")

    # Save model
    model_path = output_dir / "model.pkl"
    joblib.dump(result.model, model_path)

    # Save feature names
    feature_path = output_dir / "features.txt"
    with open(feature_path, 'w') as f:
        f.write('\n'.join(X.columns.tolist()))

    # Get top features
    top_features = {}
    if result.feature_importance:
        sorted_imp = sorted(
            result.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        top_features = dict(sorted_imp[:20])

    # Save metrics
    metrics = {
        'training_date': datetime.now().isoformat(),
        'configuration': {
            'model_type': model_type,
            'meta_labeling': use_meta_labeling,
            'n_estimators': n_estimators,
            'n_splits': n_splits,
            'embargo_pct': embargo_pct,
            'pt_sl_ratio': list(pt_sl_ratio),
            'max_holding_period': max_holding_period
        },
        'data': {
            'symbols': symbols,
            'n_symbols_processed': len(all_data),
            'n_samples': len(X),
            'n_features': len(X.columns)
        },
        'performance': {
            'train_accuracy': float(result.train_metrics.get('accuracy', 0)),
            'train_precision': float(result.train_metrics.get('precision', 0)),
            'train_recall': float(result.train_metrics.get('recall', 0)),
            'train_f1': float(result.train_metrics.get('f1', 0)),
            'cv_accuracy': float(result.cv_metrics.get('accuracy', 0)),
            'cv_accuracy_std': float(result.cv_metrics.get('accuracy_std', 0))
        },
        'top_features': top_features,
        'training_time_seconds': result.training_time_seconds
    }

    metrics_path = output_dir / "metrics.yaml"
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False, sort_keys=False)

    # =========================================================================
    # Print Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(f"\nMethodology:")
    print(f"  - Institutional Features (FracDiff, VPIN, Kyle's Lambda, HMM Regime)")
    print(f"  - {'Meta-labeling (Primary Signal + ML Filter)' if use_meta_labeling else 'Triple Barrier Labeling'}")
    print(f"  - Purged K-Fold CV (embargo={embargo_pct:.0%})")
    print(f"  - Bagging Ensemble ({n_estimators} {model_type} estimators)")

    print(f"\nData:")
    print(f"  - Symbols: {len(all_data)}/{len(symbols)}")
    print(f"  - Samples: {len(X):,}")
    print(f"  - Features: {len(X.columns)}")

    print(f"\nPerformance:")
    print(f"  - Train Accuracy:  {result.train_metrics.get('accuracy', 0):.4f}")
    print(f"  - Train F1:        {result.train_metrics.get('f1', 0):.4f}")
    print(f"  - CV Accuracy:     {result.cv_metrics.get('accuracy', 0):.4f} (+/- {result.cv_metrics.get('accuracy_std', 0):.4f})")

    if top_features:
        print(f"\nTop 10 Features:")
        for i, (feat, imp) in enumerate(list(top_features.items())[:10], 1):
            print(f"  {i:2}. {feat}: {imp:.4f}")

    print(f"\nTraining time: {result.training_time_seconds:.1f}s")
    print(f"Model saved to: {model_path}")
    print("=" * 70)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Train Institutional-Grade ML Model (AFML Best Practices)"
    )

    # Data options
    parser.add_argument("--symbol", type=str, default=None,
                        help="Single symbol to train (default: all symbols)")
    parser.add_argument("--symbols", type=str, nargs="+", default=None,
                        help="List of symbols to train")

    # Model options
    parser.add_argument("--model", type=str, default="catboost",
                        choices=["catboost", "xgboost", "lightgbm"],
                        help="Base model type (default: catboost)")
    parser.add_argument("--n-estimators", type=int, default=100,
                        help="Number of bagging estimators (default: 100)")

    # Labeling options
    parser.add_argument("--no-meta-label", action="store_true",
                        help="Disable meta-labeling (use direct prediction)")
    parser.add_argument("--pt-ratio", type=float, default=1.5,
                        help="Profit take ATR multiplier (default: 1.5)")
    parser.add_argument("--sl-ratio", type=float, default=1.0,
                        help="Stop loss ATR multiplier (default: 1.0)")
    parser.add_argument("--max-hold", type=int, default=20,
                        help="Maximum holding period in bars (default: 20)")

    # CV options
    parser.add_argument("--n-splits", type=int, default=3,
                        help="Number of CV folds (default: 3)")
    parser.add_argument("--embargo", type=float, default=0.05,
                        help="Embargo percentage (default: 0.05)")

    # Output
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Output directory (default: models)")

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_path="logs", level="INFO")

    # Get symbols
    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = get_symbols_from_config()

    logger.info(f"Training on {len(symbols)} symbols")

    # Train
    train_model(
        symbols=symbols,
        output_dir=Path(args.output_dir),
        use_meta_labeling=not args.no_meta_label,
        model_type=args.model,
        n_estimators=args.n_estimators,
        n_splits=args.n_splits,
        embargo_pct=args.embargo,
        pt_sl_ratio=(args.pt_ratio, args.sl_ratio),
        max_holding_period=args.max_hold
    )


if __name__ == "__main__":
    main()
