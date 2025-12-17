"""
Integration tests for the full training pipeline.

Tests the complete flow from data to trained model:
- Feature engineering
- Data validation
- Cross-validation with purge gap
- Model training
- Evaluation
- Model saving

These tests ensure all components work together correctly.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def sample_ohlcv_data():
    """Create realistic OHLCV data for testing."""
    np.random.seed(42)
    n = 2000  # Enough for walk-forward testing

    dates = pd.date_range(start="2023-01-01", periods=n, freq="15min")

    # Generate random walk
    returns = np.random.normal(0.0001, 0.015, n)
    close = 100 * np.cumprod(1 + returns)

    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_price = close + np.random.normal(0, 0.3, n)
    volume = np.random.randint(10000, 500000, n).astype(float)

    df = pd.DataFrame({
        "timestamp": dates,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })

    # Add returns for target
    df["returns"] = df["close"].pct_change()
    df["future_returns"] = df["returns"].shift(-5)  # 5-period ahead returns

    return df.dropna()


@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


class TestFeaturePipelineIntegration:
    """Integration tests for feature pipeline."""

    def test_feature_generation_flow(self, sample_ohlcv_data):
        """Test complete feature generation flow."""
        from src.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(
            ma_periods=[5, 10, 20, 50],
            rsi_period=14,
            include_time_features=True,
            include_volatility=True,
        )

        # Generate features
        features_df = pipeline.generate_features(
            sample_ohlcv_data,
            include_technical=True,
            include_statistical=True,
            include_lagged=True,
        )

        # Should have features
        assert features_df is not None
        assert len(features_df) > 0
        assert len(features_df.columns) > 10  # Should have multiple features

        # No infinite values
        assert not np.isinf(features_df.select_dtypes(include=[np.number]).values).any()

    def test_purge_gap_recommendation(self, sample_ohlcv_data):
        """Test purge gap recommendation."""
        from src.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(ma_periods=[5, 10, 20, 50, 100, 200])

        # Get recommendation
        recommended = pipeline.get_purge_gap_recommendation(
            prediction_horizon=5,
            buffer=10
        )

        # Should be >= max lookback + horizon + buffer
        assert recommended >= 200 + 5 + 10


class TestCrossValidationIntegration:
    """Integration tests for cross-validation."""

    def test_purged_kfold_with_features(self, sample_ohlcv_data):
        """Test PurgedKFold CV with real features."""
        from src.features.pipeline import FeaturePipeline
        from src.training.validation import PurgedKFoldCV

        # Generate features
        pipeline = FeaturePipeline(ma_periods=[5, 10, 20])
        features_df = pipeline.generate_features(sample_ohlcv_data)

        # Prepare data
        feature_cols = [c for c in features_df.columns
                       if c not in ['timestamp', 'future_returns', 'returns']]
        X = features_df[feature_cols].values
        y = features_df['future_returns'].values

        # Remove NaN rows
        valid_mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        # Create CV
        cv = PurgedKFoldCV(n_splits=5, purge_gap=50, embargo_pct=0.01)

        # Verify splits
        fold_count = 0
        for train_idx, test_idx in cv.split(X, y):
            fold_count += 1

            # No overlap
            overlap = set(train_idx).intersection(set(test_idx))
            assert len(overlap) == 0, f"Found {len(overlap)} overlapping indices"

            # Proper gap
            if len(train_idx) > 0 and len(test_idx) > 0:
                gap = min(test_idx) - max(train_idx)
                assert gap >= 50, f"Gap {gap} < purge_gap 50"

        assert fold_count == 5

    def test_walk_forward_validation(self, sample_ohlcv_data):
        """Test walk-forward validation."""
        from src.features.pipeline import FeaturePipeline
        from src.training.validation import WalkForwardValidator

        # Generate features
        pipeline = FeaturePipeline(ma_periods=[5, 10, 20])
        features_df = pipeline.generate_features(sample_ohlcv_data)

        # Prepare data
        feature_cols = [c for c in features_df.columns
                       if c not in ['timestamp', 'future_returns', 'returns']]
        X = features_df[feature_cols].values
        y = features_df['future_returns'].values

        valid_mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        # Create walk-forward validator
        wf = WalkForwardValidator(
            train_period=500,
            test_period=100,
            step_size=100,
            purge_gap=20,
        )

        # Verify walk-forward steps
        prev_test_end = -1
        step_count = 0

        for train_idx, test_idx in wf.split(X, y):
            step_count += 1

            # Test should be after train
            assert min(test_idx) > max(train_idx)

            # Test should move forward
            if prev_test_end >= 0:
                assert min(test_idx) > prev_test_end

            prev_test_end = max(test_idx)

        assert step_count > 0


class TestModelTrainingIntegration:
    """Integration tests for model training."""

    def test_lightgbm_training(self, sample_ohlcv_data, temp_dir):
        """Test LightGBM training flow."""
        from src.features.pipeline import FeaturePipeline
        from src.training.model_factory import ModelFactory, ModelType
        from src.training.validation import PurgedKFoldCV

        # Generate features
        pipeline = FeaturePipeline(ma_periods=[5, 10, 20])
        features_df = pipeline.generate_features(sample_ohlcv_data)

        # Prepare data
        feature_cols = [c for c in features_df.columns
                       if c not in ['timestamp', 'future_returns', 'returns']]
        X = features_df[feature_cols].values
        y = features_df['future_returns'].values

        valid_mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        # Create model
        model = ModelFactory.create_model(
            ModelType.LIGHTGBM_REGRESSOR,
            params={
                "n_estimators": 50,
                "max_depth": 5,
                "learning_rate": 0.1,
            }
        )

        # Train
        model.fit(X, y)

        # Predict
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert not np.isnan(predictions).any()

    def test_trainer_with_cv(self, sample_ohlcv_data):
        """Test Trainer with cross-validation."""
        from src.features.pipeline import FeaturePipeline
        from src.training.trainer import Trainer
        from src.training.model_factory import ModelType
        from src.training.validation import PurgedKFoldCV

        # Generate features
        pipeline = FeaturePipeline(ma_periods=[5, 10, 20])
        features_df = pipeline.generate_features(sample_ohlcv_data)

        # Prepare data
        feature_cols = [c for c in features_df.columns
                       if c not in ['timestamp', 'future_returns', 'returns']]
        X = features_df[feature_cols]
        y = features_df['future_returns']

        valid_mask = ~y.isna() & ~X.isna().any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        # Create trainer
        trainer = Trainer(
            model_type=ModelType.LIGHTGBM_REGRESSOR,
            params={
                "n_estimators": 30,
                "max_depth": 4,
            }
        )

        # Train with CV
        cv = PurgedKFoldCV(n_splits=3, purge_gap=30)
        result = trainer.fit_cv(X, y, cv=cv, return_estimator=True)

        # Check result
        assert result is not None
        assert result.model is not None
        assert result.cv_scores is not None
        assert len(result.cv_scores) > 0


class TestLeakagePreventionIntegration:
    """Integration tests for leakage prevention."""

    def test_leakage_checker_integration(self, sample_ohlcv_data):
        """Test leakage checker with real features."""
        from src.features.pipeline import FeaturePipeline
        from src.features.leakage_prevention import LeakageChecker

        # Generate features
        pipeline = FeaturePipeline(ma_periods=[5, 10, 20])
        features_df = pipeline.generate_features(sample_ohlcv_data)

        # Prepare data
        feature_cols = [c for c in features_df.columns
                       if c not in ['timestamp', 'future_returns', 'returns']]
        X = features_df[feature_cols]
        y = features_df['future_returns']

        valid_mask = ~y.isna() & ~X.isna().any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        # Run leakage check
        checker = LeakageChecker(
            future_correlation_threshold=0.95,
            timestamp_column='timestamp' if 'timestamp' in X.columns else None,
        )

        report = checker.check_all(X, y)

        # Report should be generated
        assert report is not None
        assert hasattr(report, 'results')


class TestDriftDetectionIntegration:
    """Integration tests for drift detection."""

    def test_drift_detection_with_features(self, sample_ohlcv_data):
        """Test drift detection on feature distributions."""
        from src.features.pipeline import FeaturePipeline
        from src.training.drift_detection import DriftDetector

        # Generate features
        pipeline = FeaturePipeline(ma_periods=[5, 10, 20])
        features_df = pipeline.generate_features(sample_ohlcv_data)

        # Prepare data
        feature_cols = [c for c in features_df.columns
                       if c not in ['timestamp', 'future_returns', 'returns']]
        X = features_df[feature_cols].dropna()

        # Split into reference and current
        split_point = len(X) // 2
        X_ref = X.iloc[:split_point]
        X_curr = X.iloc[split_point:]

        # Create detector
        detector = DriftDetector(reference_data=X_ref)

        # Detect drift
        result = detector.detect_feature_drift(X_curr)

        # Should return result
        assert result is not None
        assert hasattr(result, 'has_drift')
        assert hasattr(result, 'psi_scores')


class TestCheckpointingIntegration:
    """Integration tests for checkpointing."""

    def test_checkpoint_save_load(self, sample_ohlcv_data, temp_dir):
        """Test checkpoint save and load."""
        from src.features.pipeline import FeaturePipeline
        from src.training.model_factory import ModelFactory, ModelType
        from src.training.checkpointing import (
            CheckpointManager,
            CheckpointConfig,
            CheckpointMetadata,
            TrainingState,
        )
        from datetime import datetime

        # Generate features
        pipeline = FeaturePipeline(ma_periods=[5, 10, 20])
        features_df = pipeline.generate_features(sample_ohlcv_data)

        feature_cols = [c for c in features_df.columns
                       if c not in ['timestamp', 'future_returns', 'returns']]
        X = features_df[feature_cols].values
        y = features_df['future_returns'].values

        valid_mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        # Train model
        model = ModelFactory.create_model(
            ModelType.LIGHTGBM_REGRESSOR,
            params={"n_estimators": 30}
        )
        model.fit(X, y)

        # Create checkpoint manager
        config = CheckpointConfig(
            checkpoint_dir=Path(temp_dir) / "checkpoints",
            save_frequency=1,
        )
        manager = CheckpointManager(config)

        # Create training state and metadata
        state = TrainingState(
            epoch=1,
            step=100,
            best_metric=0.5,
            best_epoch=1,
            early_stopping_counter=0,
            learning_rate=0.01,
            optimizer_state=None,
            scheduler_state=None,
        )

        metadata = CheckpointMetadata(
            checkpoint_id=manager._generate_checkpoint_id(1, 100),
            epoch=1,
            step=100,
            timestamp=datetime.now().isoformat(),
            model_type="lightgbm_regressor",
            task_type="regression",
            metrics={"val_mse": 0.01},
            best_metric_value=0.5,
            best_metric_name="val_mse",
            data_hash="abc123",
            random_state=42,
            params={"n_estimators": 30},
            training_time_seconds=10.0,
            n_samples_processed=1000,
        )

        # Save checkpoint
        checkpoint_id = manager.save_checkpoint(model, state, metadata, force=True)
        assert checkpoint_id is not None

        # Load checkpoint
        loaded_model, loaded_state, loaded_metadata = manager.load_latest()
        assert loaded_model is not None
        assert loaded_state.epoch == 1
        assert loaded_metadata.model_type == "lightgbm_regressor"


class TestFeatureSelectionIntegration:
    """Integration tests for feature selection."""

    def test_feature_selection_flow(self, sample_ohlcv_data):
        """Test feature selection with real features."""
        from src.features.pipeline import FeaturePipeline
        from src.features.feature_selection import FeatureSelector

        # Generate features
        pipeline = FeaturePipeline(ma_periods=[5, 10, 20, 50])
        features_df = pipeline.generate_features(sample_ohlcv_data)

        # Prepare data
        feature_cols = [c for c in features_df.columns
                       if c not in ['timestamp', 'future_returns', 'returns']]
        X = features_df[feature_cols]
        y = features_df['future_returns']

        valid_mask = ~y.isna() & ~X.isna().any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        # Create selector
        selector = FeatureSelector(
            n_features_to_select=10,
            methods=['variance', 'correlation', 'importance'],
        )

        # Select features
        result = selector.fit(X, y)

        # Should select features
        assert result is not None
        assert len(result.selected_features) <= 10
        assert len(result.selected_features) > 0


class TestEndToEndPipeline:
    """End-to-end integration tests."""

    def test_complete_pipeline(self, sample_ohlcv_data, temp_dir):
        """Test complete pipeline from data to saved model."""
        from src.features.pipeline import FeaturePipeline
        from src.training.model_factory import ModelFactory, ModelType
        from src.training.trainer import Trainer
        from src.training.validation import PurgedKFoldCV
        import joblib

        # Step 1: Generate features
        pipeline = FeaturePipeline(ma_periods=[5, 10, 20])
        features_df = pipeline.generate_features(sample_ohlcv_data)

        # Step 2: Prepare data
        feature_cols = [c for c in features_df.columns
                       if c not in ['timestamp', 'future_returns', 'returns']]
        X = features_df[feature_cols]
        y = features_df['future_returns']

        valid_mask = ~y.isna() & ~X.isna().any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        # Step 3: Create CV
        cv = PurgedKFoldCV(n_splits=3, purge_gap=30)

        # Step 4: Train with CV
        trainer = Trainer(
            model_type=ModelType.LIGHTGBM_REGRESSOR,
            params={"n_estimators": 30, "max_depth": 4}
        )

        result = trainer.fit_cv(X, y, cv=cv, return_estimator=True)

        # Step 5: Verify result
        assert result.model is not None
        assert result.cv_scores is not None

        # Step 6: Save model
        model_path = Path(temp_dir) / "model.joblib"
        joblib.dump(result.model, model_path)

        # Step 7: Load and verify
        loaded_model = joblib.load(model_path)
        predictions = loaded_model.predict(X.values)

        assert len(predictions) == len(y)
        assert not np.isnan(predictions).any()

        # Step 8: Verify predictions are reasonable
        ic = np.corrcoef(predictions, y.values)[0, 1]
        # IC should be defined (not NaN)
        assert np.isfinite(ic)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
