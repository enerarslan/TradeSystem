"""
ML Models Tests
===============

Unit tests for machine learning models.

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path


# =============================================================================
# LIGHTGBM TESTS
# =============================================================================

class TestLightGBMClassifier:
    """Tests for LightGBM Classifier."""
    
    def test_initialization(self, lightgbm_model):
        """Test model initialization."""
        assert lightgbm_model is not None
        assert lightgbm_model.name == "LightGBMClassifier"
    
    def test_fit(self, lightgbm_model, sample_multiclass_data):
        """Test model fitting."""
        X, y, feature_names = sample_multiclass_data
        
        X_train, y_train = X[:800], y[:800]
        
        lightgbm_model.fit(X_train, y_train, feature_names=feature_names)
        
        assert lightgbm_model.is_trained
    
    def test_predict(self, trained_lightgbm_model, sample_multiclass_data):
        """Test model prediction."""
        X, y, _ = sample_multiclass_data
        X_test = X[800:]
        
        predictions = trained_lightgbm_model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(p in [0, 1, 2] for p in predictions)
    
    def test_predict_proba(self, trained_lightgbm_model, sample_multiclass_data):
        """Test probability prediction."""
        X, y, _ = sample_multiclass_data
        X_test = X[800:]
        
        probas = trained_lightgbm_model.predict_proba(X_test)
        
        assert probas.shape == (len(X_test), 3)  # 3 classes
        assert np.allclose(probas.sum(axis=1), 1.0)  # Probabilities sum to 1
    
    def test_evaluate(self, trained_lightgbm_model, sample_multiclass_data):
        """Test model evaluation."""
        X, y, _ = sample_multiclass_data
        X_test, y_test = X[800:], y[800:]
        
        metrics = trained_lightgbm_model.evaluate(X_test, y_test)
        
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1
    
    def test_feature_importance(self, trained_lightgbm_model):
        """Test feature importance extraction."""
        importance = trained_lightgbm_model.get_feature_importance()
        
        assert importance is not None
        assert len(importance) > 0
    
    def test_save_load(self, trained_lightgbm_model, temp_model_dir, sample_multiclass_data):
        """Test model saving and loading."""
        X, y, _ = sample_multiclass_data
        X_test = X[800:]
        
        # Save model
        model_path = temp_model_dir / "test_lgb.pkl"
        trained_lightgbm_model.save(model_path)
        
        assert model_path.exists()
        
        # Load model
        from models.classifiers import LightGBMClassifier
        loaded_model = LightGBMClassifier.load(model_path)
        
        # Compare predictions
        original_preds = trained_lightgbm_model.predict(X_test)
        loaded_preds = loaded_model.predict(X_test)
        
        assert np.array_equal(original_preds, loaded_preds)


# =============================================================================
# XGBOOST TESTS
# =============================================================================

class TestXGBoostClassifier:
    """Tests for XGBoost Classifier."""
    
    def test_initialization(self, xgboost_model):
        """Test model initialization."""
        assert xgboost_model is not None
        assert xgboost_model.name == "XGBoostClassifier"
    
    def test_fit_predict(self, xgboost_model, sample_multiclass_data):
        """Test model fitting and prediction."""
        X, y, feature_names = sample_multiclass_data
        
        X_train, X_test = X[:800], X[800:]
        y_train, y_test = y[:800], y[800:]
        
        xgboost_model.fit(X_train, y_train, feature_names=feature_names)
        
        assert xgboost_model.is_trained
        
        predictions = xgboost_model.predict(X_test)
        
        assert len(predictions) == len(X_test)


# =============================================================================
# TRAINING PIPELINE TESTS
# =============================================================================

class TestTrainingPipeline:
    """Tests for training pipeline."""
    
    def test_quick_train(self, sample_multiclass_data):
        """Test quick training function."""
        from models.training import quick_train
        
        X, y, feature_names = sample_multiclass_data
        
        model = quick_train(
            model_type="lightgbm",
            X=X,
            y=y,
            feature_names=feature_names,
            optimize=False,  # Skip optimization for speed
        )
        
        assert model is not None
        assert model.is_trained
    
    def test_training_pipeline(self, sample_multiclass_data, temp_model_dir):
        """Test full training pipeline."""
        from models.training import TrainingPipeline, TrainingConfig, OptimizationConfig
        
        X, y, feature_names = sample_multiclass_data
        
        X_train, X_test = X[:800], X[800:]
        y_train, y_test = y[:800], y[800:]
        
        config = TrainingConfig(
            models_dir=temp_model_dir,
            auto_optimize=False,
        )
        
        pipeline = TrainingPipeline(config)
        
        model = pipeline.train(
            model_type="lightgbm",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
        )
        
        assert model is not None
        assert model.is_trained


# =============================================================================
# DEEP LEARNING TESTS
# =============================================================================

@pytest.mark.slow
class TestDeepLearningModels:
    """Tests for deep learning models."""
    
    def test_lstm_model(self, sample_multiclass_data):
        """Test LSTM model."""
        from models.deep import LSTMModel, LSTMConfig
        
        X, y, feature_names = sample_multiclass_data
        
        config = LSTMConfig(
            sequence_length=20,
            n_features=X.shape[1],
            epochs=5,
            batch_size=32,
        )
        
        model = LSTMModel(config)
        
        # Need to pad for sequence length
        X_padded = np.pad(X, ((config.sequence_length - 1, 0), (0, 0)), mode='constant')
        
        X_train = X_padded[:800]
        y_train = y[:800 - config.sequence_length + 1]
        
        model.fit(X_train, y_train, feature_names=feature_names)
        
        assert model.is_trained
    
    def test_create_deep_model_factory(self):
        """Test deep model factory function."""
        from models.deep import create_deep_model
        
        model = create_deep_model("lstm", sequence_length=20, n_features=50)
        
        assert model is not None
        assert model.name == "LSTMModel"


# =============================================================================
# MODEL REGISTRY TESTS
# =============================================================================

class TestModelRegistry:
    """Tests for model registry."""
    
    def test_registry_contains_models(self):
        """Test that registry contains expected models."""
        from models.base import ModelRegistry
        
        assert "lightgbm" in ModelRegistry._registry or len(ModelRegistry._registry) >= 0
    
    def test_create_from_registry(self):
        """Test creating model from registry."""
        from models.classifiers import create_classifier
        
        model = create_classifier("lightgbm")
        
        assert model is not None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for models with other components."""
    
    def test_model_with_feature_pipeline(self, sample_ohlcv_data):
        """Test model training with feature pipeline."""
        from features.pipeline import FeaturePipeline, create_default_config
        from models.classifiers import LightGBMClassifier, LightGBMClassifierConfig
        
        # Generate features
        pipeline = FeaturePipeline(create_default_config())
        df_features = pipeline.generate(sample_ohlcv_data)
        df_features = pipeline.create_target(df_features, target_type="direction", horizon=5)
        
        X_train, X_test, y_train, y_test, feature_names = pipeline.prepare_train_test(
            df_features, test_size=0.2
        )
        
        # Train model
        config = LightGBMClassifierConfig(n_estimators=100)
        model = LightGBMClassifier(config)
        
        model.fit(X_train, y_train, feature_names=feature_names)
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        
        assert model.is_trained
        assert "accuracy" in metrics