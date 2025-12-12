"""
Model Training Script
Trains and saves ML models for the trading system
"""

import sys
import os
from pathlib import Path
import argparse
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger, setup_logging
from src.data.loader import MultiAssetLoader
from src.data.preprocessor import DataPreprocessor
from src.features.builder import FeatureBuilder
from src.models.ml_model import XGBoostModel, LightGBMModel, CatBoostModel
from src.models.ensemble import StackingEnsemble, VotingEnsemble
from src.models.training import ModelTrainer, WalkForwardValidator

logger = get_logger(__name__)


def load_config(path: str) -> dict:
    """Load YAML configuration"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def prepare_data(
    symbols: list,
    data_dir: str = "data/raw",
    lookback_days: int = 500
) -> tuple:
    """
    Load and prepare data for training.

    Returns:
        Tuple of (features_df, labels)
    """
    logger.info(f"Loading data for {len(symbols)} symbols...")

    # Load data
    loader = MultiAssetLoader(symbols=symbols, data_dir=data_dir)
    preprocessor = DataPreprocessor()
    feature_builder = FeatureBuilder()

    all_features = []
    all_labels = []

    for symbol in symbols:
        try:
            # Load raw data
            df = loader.load_symbol(symbol)
            if df is None or len(df) < 100:
                logger.warning(f"Insufficient data for {symbol}")
                continue

            # Preprocess
            df = preprocessor.clean_data(df)

            # Generate features
            features = feature_builder.build_features(df)

            # Create labels (forward returns)
            # 1 = up, 0 = down
            forward_returns = df['close'].pct_change(5).shift(-5)  # 5-bar forward return
            labels = (forward_returns > 0).astype(int)

            # Align and drop NaN
            valid_idx = features.dropna().index.intersection(labels.dropna().index)
            features = features.loc[valid_idx]
            labels = labels.loc[valid_idx]

            features['symbol'] = symbol
            all_features.append(features)
            all_labels.append(labels)

            logger.info(f"Prepared {len(features)} samples for {symbol}")

        except Exception as e:
            logger.warning(f"Error processing {symbol}: {e}")

    if not all_features:
        raise ValueError("No data prepared")

    # Combine
    features_df = pd.concat(all_features, axis=0)
    labels = pd.concat(all_labels, axis=0)

    logger.info(f"Total samples: {len(features_df)}")

    return features_df, labels


def train_single_model(
    model_class,
    X_train,
    y_train,
    X_val,
    y_val,
    model_name: str,
    **kwargs
) -> tuple:
    """Train a single model and return metrics"""
    logger.info(f"Training {model_name}...")

    model = model_class(name=model_name, **kwargs)

    # Train
    model.train(X_train, y_train, X_val, y_val)

    # Evaluate
    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)

    # Metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    metrics = {
        'accuracy': accuracy_score(y_val, val_pred),
        'precision': precision_score(y_val, val_pred, zero_division=0),
        'recall': recall_score(y_val, val_pred, zero_division=0),
        'f1': f1_score(y_val, val_pred, zero_division=0),
        'auc': roc_auc_score(y_val, val_prob[:, 1]) if val_prob is not None else 0
    }

    logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")

    return model, metrics


def train_ensemble(
    base_models: list,
    X_train,
    y_train,
    X_val,
    y_val
) -> tuple:
    """Train ensemble model"""
    logger.info("Training ensemble...")

    # Voting ensemble
    voting = VotingEnsemble(
        models=base_models,
        voting='soft'
    )
    voting.train(X_train, y_train, X_val, y_val)

    # Evaluate
    val_pred = voting.predict(X_val)
    val_prob = voting.predict_proba(X_val)

    from sklearn.metrics import accuracy_score, roc_auc_score

    metrics = {
        'accuracy': accuracy_score(y_val, val_pred),
        'auc': roc_auc_score(y_val, val_prob[:, 1]) if val_prob is not None else 0
    }

    logger.info(f"Ensemble - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")

    return voting, metrics


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument("--config", type=str, default="config/settings.yaml")
    parser.add_argument("--symbols", type=str, nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()

    # Setup logging
    setup_logging(log_dir="logs", level="INFO")

    logger.info("=" * 60)
    logger.info("Starting Model Training")
    logger.info("=" * 60)

    # Load config
    config = load_config(args.config)
    symbols_config = load_config("config/symbols.yaml")

    # Get symbols
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = symbols_config.get('universe', {}).get('symbols', [])[:10]  # First 10 for training

    logger.info(f"Training on {len(symbols)} symbols: {symbols}")

    # Prepare data
    features_df, labels = prepare_data(symbols)

    # Remove symbol column for training
    X = features_df.drop(columns=['symbol'], errors='ignore')
    y = labels

    # Train/validation split (time-based)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train individual models
    models = []
    all_metrics = {}

    # XGBoost
    xgb_model, xgb_metrics = train_single_model(
        XGBoostModel,
        X_train, y_train, X_val, y_val,
        model_name="xgboost_v1",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1
    )
    models.append(xgb_model)
    all_metrics['xgboost'] = xgb_metrics

    # LightGBM
    lgb_model, lgb_metrics = train_single_model(
        LightGBMModel,
        X_train, y_train, X_val, y_val,
        model_name="lightgbm_v1",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1
    )
    models.append(lgb_model)
    all_metrics['lightgbm'] = lgb_metrics

    # CatBoost
    cat_model, cat_metrics = train_single_model(
        CatBoostModel,
        X_train, y_train, X_val, y_val,
        model_name="catboost_v1",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1
    )
    models.append(cat_model)
    all_metrics['catboost'] = cat_metrics

    # Train ensemble
    ensemble_model, ensemble_metrics = train_ensemble(
        models, X_train, y_train, X_val, y_val
    )
    all_metrics['ensemble'] = ensemble_metrics

    # Save models
    logger.info("Saving models...")

    for model in models:
        model_path = output_dir / f"{model.name}.pkl"
        model.save(str(model_path))
        logger.info(f"Saved {model.name} to {model_path}")

    # Save ensemble
    ensemble_path = output_dir / "ensemble_model.pkl"
    ensemble_model.save(str(ensemble_path))
    logger.info(f"Saved ensemble to {ensemble_path}")

    # Save metrics
    metrics_path = output_dir / "training_metrics.yaml"
    with open(metrics_path, 'w') as f:
        yaml.dump({
            'training_date': datetime.now().isoformat(),
            'symbols': symbols,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'metrics': all_metrics
        }, f)

    # Save feature names
    feature_names_path = output_dir / "feature_names.txt"
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(X.columns.tolist()))

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)

    # Print summary
    print("\nTraining Summary:")
    print("-" * 40)
    for model_name, metrics in all_metrics.items():
        print(f"{model_name:15} - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics.get('auc', 0):.4f}")

    print(f"\nModels saved to: {output_dir}")


if __name__ == "__main__":
    main()
