"""
ML Strategy Module
==================

Machine learning-based trading strategies for the algorithmic trading platform.

Strategies:
- MLClassifierStrategy: Classification-based (buy/sell/hold)
- MLRegressorStrategy: Regression-based (return prediction)
- EnsembleMLStrategy: Ensemble of multiple ML models
- RLStrategy: Reinforcement learning strategy (placeholder)

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar

import numpy as np
import polars as pl
from numpy.typing import NDArray

from config.settings import get_logger, TimeFrame
from core.events import MarketEvent, SignalEvent
from core.types import PortfolioState, ModelNotTrainedError
from strategies.base import BaseStrategy, StrategyConfig
from features.pipeline import FeaturePipeline, FeatureConfig

logger = get_logger(__name__)


# =============================================================================
# PROTOCOLS AND TYPES
# =============================================================================

class MLModel(Protocol):
    """Protocol for ML models."""
    
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Make predictions."""
        ...
    
    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict probabilities (for classifiers)."""
        ...


class PredictionType(str, Enum):
    """Type of ML prediction."""
    CLASSIFICATION = "classification"  # Buy/Sell/Hold classes
    REGRESSION = "regression"  # Return prediction
    PROBABILITY = "probability"  # Probability of positive return


# =============================================================================
# BASE ML STRATEGY
# =============================================================================

@dataclass
class MLStrategyConfig(StrategyConfig):
    """Configuration for ML-based strategies."""
    name: str = "MLStrategy"
    
    # Model configuration
    model_path: str | None = None
    prediction_type: PredictionType = PredictionType.CLASSIFICATION
    
    # Feature configuration
    feature_config: FeatureConfig | None = None
    required_features: list[str] = field(default_factory=list)
    lookback_bars: int = 100
    
    # Prediction thresholds
    long_threshold: float = 0.6  # Prob > 0.6 for long
    short_threshold: float = 0.4  # Prob < 0.4 for short
    return_threshold: float = 0.01  # Min predicted return
    
    # Signal generation
    min_confidence: float = 0.6
    use_probability_as_strength: bool = True
    
    # Risk parameters
    prediction_horizon: int = 5  # Bars forward
    retrain_frequency: int = 0  # Bars between retraining (0 = no retrain)
    
    # Feature update
    update_features_every_bar: bool = True


class BaseMLStrategy(BaseStrategy):
    """
    Base class for ML-based trading strategies.
    
    Provides common functionality for:
    - Feature generation
    - Model loading/prediction
    - Signal generation from predictions
    
    Subclasses must implement:
    - _load_model(): Load the ML model
    - _generate_prediction(): Generate prediction from features
    """
    
    def __init__(
        self,
        config: MLStrategyConfig | None = None,
        parameters: dict[str, Any] | None = None,
        model: Any | None = None,
    ):
        """
        Initialize ML strategy.
        
        Args:
            config: Strategy configuration
            parameters: Strategy parameters
            model: Pre-loaded model (optional)
        """
        super().__init__(config or MLStrategyConfig(), parameters)
        self.config: MLStrategyConfig = self.config
        
        # Model
        self._model = model
        self._model_loaded = model is not None
        
        # Feature pipeline
        self._feature_pipeline = FeaturePipeline(self.config.feature_config)
        
        # Feature storage
        self._feature_data: dict[str, pl.DataFrame] = {}
        self._last_features: dict[str, NDArray[np.float64]] = {}
        
        # Prediction cache
        self._last_prediction: dict[str, dict[str, Any]] = {}
        
        # Training state
        self._last_retrain_bar: int = 0
    
    @property
    def model(self) -> Any:
        """Get the ML model."""
        return self._model
    
    @property
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded
    
    def _on_initialize(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        """Initialize ML strategy."""
        # Load model if path provided
        if self.config.model_path and not self._model_loaded:
            self._load_model(self.config.model_path)
        
        # Initialize feature data for each symbol
        for symbol in symbols:
            self._feature_data[symbol] = pl.DataFrame()
            self._last_prediction[symbol] = {}
    
    @abstractmethod
    def _load_model(self, path: str) -> None:
        """
        Load ML model from path.
        
        Args:
            path: Path to saved model
        """
        pass
    
    def set_model(self, model: Any) -> None:
        """
        Set the ML model.
        
        Args:
            model: Trained ML model
        """
        self._model = model
        self._model_loaded = True
        logger.info(f"Model set for {self.name}")
    
    def calculate_signals(
        self,
        event: MarketEvent,
        portfolio: PortfolioState,
    ) -> list[SignalEvent]:
        """Generate signals from ML predictions."""
        signals = []
        symbol = event.symbol
        data = event.data
        
        if data is None or not self._model_loaded:
            return signals
        
        if len(data) < self.config.lookback_bars:
            return signals
        
        # Generate features
        features = self._generate_features(symbol, data)
        if features is None or len(features) == 0:
            return signals
        
        # Get prediction
        prediction = self._generate_prediction(symbol, features)
        if prediction is None:
            return signals
        
        # Cache prediction
        self._last_prediction[symbol] = prediction
        
        # Generate signal from prediction
        signal = self._prediction_to_signal(symbol, prediction, data)
        if signal is not None:
            signals.append(signal)
        
        # Check for exit signals
        exit_signal = self._check_exit(symbol, prediction, portfolio)
        if exit_signal is not None:
            signals.append(exit_signal)
        
        return signals
    
    def _generate_features(
        self,
        symbol: str,
        data: pl.DataFrame,
    ) -> NDArray[np.float64] | None:
        """
        Generate features from price data.
        
        Args:
            symbol: Trading symbol
            data: OHLCV DataFrame
        
        Returns:
            Feature array or None
        """
        try:
            # Generate all features
            df_features = self._feature_pipeline.generate(data)
            
            # Get feature matrix
            X, feature_names = self._feature_pipeline.get_feature_matrix(df_features)
            
            # Use only required features if specified
            if self.config.required_features:
                feature_indices = [
                    i for i, name in enumerate(feature_names)
                    if name in self.config.required_features
                ]
                if feature_indices:
                    X = X[:, feature_indices]
            
            # Get latest features (last row)
            if len(X) > 0:
                latest_features = X[-1:, :]
                
                # Handle NaN
                if np.any(np.isnan(latest_features)):
                    # Forward fill or use mean
                    latest_features = np.nan_to_num(latest_features, nan=0.0)
                
                self._last_features[symbol] = latest_features
                return latest_features
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating features for {symbol}: {e}")
            return None
    
    @abstractmethod
    def _generate_prediction(
        self,
        symbol: str,
        features: NDArray[np.float64],
    ) -> dict[str, Any] | None:
        """
        Generate prediction from features.
        
        Args:
            symbol: Trading symbol
            features: Feature array
        
        Returns:
            Prediction dictionary or None
        """
        pass
    
    def _prediction_to_signal(
        self,
        symbol: str,
        prediction: dict[str, Any],
        data: pl.DataFrame,
    ) -> SignalEvent | None:
        """
        Convert ML prediction to trading signal.
        
        Args:
            symbol: Trading symbol
            prediction: Prediction dictionary
            data: Price data
        
        Returns:
            Signal event or None
        """
        close = data["close"].item(-1)
        position = self.get_position(symbol)
        has_position = position and position.is_open
        
        pred_type = self.config.prediction_type
        
        if pred_type == PredictionType.CLASSIFICATION:
            return self._classification_to_signal(symbol, prediction, close, has_position)
        elif pred_type == PredictionType.REGRESSION:
            return self._regression_to_signal(symbol, prediction, close, has_position)
        elif pred_type == PredictionType.PROBABILITY:
            return self._probability_to_signal(symbol, prediction, close, has_position)
        
        return None
    
    def _classification_to_signal(
        self,
        symbol: str,
        prediction: dict[str, Any],
        price: float,
        has_position: bool,
    ) -> SignalEvent | None:
        """Convert classification prediction to signal."""
        pred_class = prediction.get("class", 0)
        confidence = prediction.get("confidence", 0.5)
        
        if confidence < self.config.min_confidence:
            return None
        
        if not has_position:
            # Entry signals
            if pred_class == 1:  # Long
                return self.create_entry_signal(
                    symbol=symbol,
                    direction=1,
                    strength=confidence if self.config.use_probability_as_strength else 0.7,
                    price=price,
                    metadata={"ml_prediction": prediction}
                )
            elif pred_class == -1:  # Short
                return self.create_entry_signal(
                    symbol=symbol,
                    direction=-1,
                    strength=confidence if self.config.use_probability_as_strength else 0.7,
                    price=price,
                    metadata={"ml_prediction": prediction}
                )
        
        return None
    
    def _regression_to_signal(
        self,
        symbol: str,
        prediction: dict[str, Any],
        price: float,
        has_position: bool,
    ) -> SignalEvent | None:
        """Convert regression prediction to signal."""
        pred_return = prediction.get("return", 0.0)
        
        if not has_position:
            if pred_return > self.config.return_threshold:
                strength = min(abs(pred_return) * 10, 1.0)
                return self.create_entry_signal(
                    symbol=symbol,
                    direction=1,
                    strength=strength,
                    price=price,
                    metadata={"ml_prediction": prediction}
                )
            elif pred_return < -self.config.return_threshold:
                strength = min(abs(pred_return) * 10, 1.0)
                return self.create_entry_signal(
                    symbol=symbol,
                    direction=-1,
                    strength=strength,
                    price=price,
                    metadata={"ml_prediction": prediction}
                )
        
        return None
    
    def _probability_to_signal(
        self,
        symbol: str,
        prediction: dict[str, Any],
        price: float,
        has_position: bool,
    ) -> SignalEvent | None:
        """Convert probability prediction to signal."""
        prob = prediction.get("probability", 0.5)
        
        if not has_position:
            if prob > self.config.long_threshold:
                return self.create_entry_signal(
                    symbol=symbol,
                    direction=1,
                    strength=prob if self.config.use_probability_as_strength else 0.7,
                    price=price,
                    metadata={"ml_prediction": prediction}
                )
            elif prob < self.config.short_threshold:
                strength = 1 - prob if self.config.use_probability_as_strength else 0.7
                return self.create_entry_signal(
                    symbol=symbol,
                    direction=-1,
                    strength=strength,
                    price=price,
                    metadata={"ml_prediction": prediction}
                )
        
        return None
    
    def _check_exit(
        self,
        symbol: str,
        prediction: dict[str, Any],
        portfolio: PortfolioState,
    ) -> SignalEvent | None:
        """Check for ML-based exit signals."""
        position = self.get_position(symbol)
        if not position or not position.is_open:
            return None
        
        close = position.current_price
        direction = 1 if position.quantity > 0 else -1
        
        pred_type = self.config.prediction_type
        
        if pred_type == PredictionType.CLASSIFICATION:
            pred_class = prediction.get("class", 0)
            # Exit if prediction reverses
            if direction == 1 and pred_class == -1:
                return self.create_exit_signal(
                    symbol=symbol,
                    direction=direction,
                    strength=0.8,
                    price=close,
                    reason="ml_reversal",
                )
            elif direction == -1 and pred_class == 1:
                return self.create_exit_signal(
                    symbol=symbol,
                    direction=direction,
                    strength=0.8,
                    price=close,
                    reason="ml_reversal",
                )
        
        elif pred_type == PredictionType.REGRESSION:
            pred_return = prediction.get("return", 0.0)
            # Exit if predicted return is adverse
            if direction == 1 and pred_return < -self.config.return_threshold / 2:
                return self.create_exit_signal(
                    symbol=symbol,
                    direction=direction,
                    strength=0.8,
                    price=close,
                    reason="ml_negative_return",
                )
            elif direction == -1 and pred_return > self.config.return_threshold / 2:
                return self.create_exit_signal(
                    symbol=symbol,
                    direction=direction,
                    strength=0.8,
                    price=close,
                    reason="ml_positive_return",
                )
        
        elif pred_type == PredictionType.PROBABILITY:
            prob = prediction.get("probability", 0.5)
            # Exit if probability moves to neutral
            if direction == 1 and prob < 0.5:
                return self.create_exit_signal(
                    symbol=symbol,
                    direction=direction,
                    strength=0.8,
                    price=close,
                    reason="ml_probability_drop",
                )
            elif direction == -1 and prob > 0.5:
                return self.create_exit_signal(
                    symbol=symbol,
                    direction=direction,
                    strength=0.8,
                    price=close,
                    reason="ml_probability_rise",
                )
        
        return None


# =============================================================================
# CLASSIFIER STRATEGY
# =============================================================================

@dataclass
class MLClassifierConfig(MLStrategyConfig):
    """Configuration for ML classifier strategy."""
    name: str = "MLClassifier"
    prediction_type: PredictionType = PredictionType.CLASSIFICATION
    
    # Class mapping
    num_classes: int = 3  # -1 (sell), 0 (hold), 1 (buy)
    class_labels: list[int] = field(default_factory=lambda: [-1, 0, 1])


class MLClassifierStrategy(BaseMLStrategy):
    """
    ML Classifier Strategy.
    
    Uses a classification model to predict buy/sell/hold signals.
    
    Model Requirements:
        - predict(): Returns class labels
        - predict_proba(): Returns class probabilities
    
    Supported Models:
        - scikit-learn classifiers
        - LightGBM Classifier
        - XGBoost Classifier
        - CatBoost Classifier
    
    Example:
        from lightgbm import LGBMClassifier
        
        model = LGBMClassifier()
        model.fit(X_train, y_train)
        
        config = MLClassifierConfig()
        strategy = MLClassifierStrategy(config, model=model)
    """
    
    def __init__(
        self,
        config: MLClassifierConfig | None = None,
        parameters: dict[str, Any] | None = None,
        model: Any | None = None,
    ):
        """Initialize classifier strategy."""
        super().__init__(config or MLClassifierConfig(), parameters, model)
        self.config: MLClassifierConfig = self.config
    
    def _load_model(self, path: str) -> None:
        """Load classifier from path."""
        import pickle
        
        try:
            with open(path, "rb") as f:
                self._model = pickle.load(f)
            self._model_loaded = True
            logger.info(f"Classifier loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            raise ModelNotTrainedError(f"Failed to load model: {e}")
    
    def _generate_prediction(
        self,
        symbol: str,
        features: NDArray[np.float64],
    ) -> dict[str, Any] | None:
        """Generate classification prediction."""
        try:
            # Get class prediction
            pred_class = self._model.predict(features)[0]
            
            # Get probabilities
            probas = None
            confidence = 0.5
            
            if hasattr(self._model, "predict_proba"):
                probas = self._model.predict_proba(features)[0]
                # Confidence is the max probability
                confidence = np.max(probas)
            
            return {
                "class": int(pred_class),
                "confidence": float(confidence),
                "probabilities": probas.tolist() if probas is not None else None,
            }
            
        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            return None


# =============================================================================
# REGRESSOR STRATEGY
# =============================================================================

@dataclass
class MLRegressorConfig(MLStrategyConfig):
    """Configuration for ML regressor strategy."""
    name: str = "MLRegressor"
    prediction_type: PredictionType = PredictionType.REGRESSION
    
    # Prediction scaling
    scale_prediction: bool = True
    prediction_std: float = 0.01  # Expected std of predictions


class MLRegressorStrategy(BaseMLStrategy):
    """
    ML Regressor Strategy.
    
    Uses a regression model to predict future returns.
    
    Model Requirements:
        - predict(): Returns continuous values (predicted returns)
    
    Supported Models:
        - scikit-learn regressors
        - LightGBM Regressor
        - XGBoost Regressor
        - Neural networks
    
    Example:
        from xgboost import XGBRegressor
        
        model = XGBRegressor()
        model.fit(X_train, y_train)
        
        config = MLRegressorConfig()
        strategy = MLRegressorStrategy(config, model=model)
    """
    
    def __init__(
        self,
        config: MLRegressorConfig | None = None,
        parameters: dict[str, Any] | None = None,
        model: Any | None = None,
    ):
        """Initialize regressor strategy."""
        super().__init__(config or MLRegressorConfig(), parameters, model)
        self.config: MLRegressorConfig = self.config
    
    def _load_model(self, path: str) -> None:
        """Load regressor from path."""
        import pickle
        
        try:
            with open(path, "rb") as f:
                self._model = pickle.load(f)
            self._model_loaded = True
            logger.info(f"Regressor loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load regressor: {e}")
            raise ModelNotTrainedError(f"Failed to load model: {e}")
    
    def _generate_prediction(
        self,
        symbol: str,
        features: NDArray[np.float64],
    ) -> dict[str, Any] | None:
        """Generate regression prediction."""
        try:
            pred_return = self._model.predict(features)[0]
            
            return {
                "return": float(pred_return),
                "scaled_return": float(pred_return / self.config.prediction_std) if self.config.scale_prediction else float(pred_return),
            }
            
        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            return None


# =============================================================================
# ENSEMBLE ML STRATEGY
# =============================================================================

@dataclass
class EnsembleMLConfig(MLStrategyConfig):
    """Configuration for ensemble ML strategy."""
    name: str = "EnsembleML"
    
    # Ensemble configuration
    combination_method: str = "voting"  # voting, average, weighted
    model_weights: list[float] = field(default_factory=list)
    min_agreement: float = 0.6  # For voting method


class EnsembleMLStrategy(BaseMLStrategy):
    """
    Ensemble ML Strategy.
    
    Combines predictions from multiple ML models.
    
    Combination Methods:
        - voting: Majority voting for classification
        - average: Average predictions for regression
        - weighted: Weighted combination
    
    Example:
        models = [lgbm_model, xgb_model, rf_model]
        config = EnsembleMLConfig(
            combination_method="voting",
        )
        strategy = EnsembleMLStrategy(config, models=models)
    """
    
    def __init__(
        self,
        config: EnsembleMLConfig | None = None,
        parameters: dict[str, Any] | None = None,
        models: list[Any] | None = None,
    ):
        """Initialize ensemble strategy."""
        super().__init__(config or EnsembleMLConfig(), parameters)
        self.config: EnsembleMLConfig = self.config
        
        self._models: list[Any] = models or []
        self._model_loaded = len(self._models) > 0
        
        # Set default weights
        if not self.config.model_weights and self._models:
            self.config.model_weights = [1.0 / len(self._models)] * len(self._models)
    
    def add_model(self, model: Any, weight: float = 1.0) -> None:
        """Add a model to the ensemble."""
        self._models.append(model)
        self.config.model_weights.append(weight)
        
        # Normalize weights
        total = sum(self.config.model_weights)
        self.config.model_weights = [w / total for w in self.config.model_weights]
        
        self._model_loaded = True
    
    def _load_model(self, path: str) -> None:
        """Load ensemble from path."""
        import pickle
        
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            
            self._models = data.get("models", [])
            self.config.model_weights = data.get("weights", [])
            
            self._model_loaded = len(self._models) > 0
            logger.info(f"Ensemble loaded with {len(self._models)} models")
        except Exception as e:
            logger.error(f"Failed to load ensemble: {e}")
            raise ModelNotTrainedError(f"Failed to load ensemble: {e}")
    
    def _generate_prediction(
        self,
        symbol: str,
        features: NDArray[np.float64],
    ) -> dict[str, Any] | None:
        """Generate ensemble prediction."""
        if not self._models:
            return None
        
        try:
            predictions = []
            probabilities = []
            
            for model in self._models:
                pred = model.predict(features)[0]
                predictions.append(pred)
                
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(features)[0]
                    probabilities.append(prob)
            
            # Combine predictions
            if self.config.combination_method == "voting":
                return self._voting_combination(predictions, probabilities)
            elif self.config.combination_method == "average":
                return self._average_combination(predictions)
            elif self.config.combination_method == "weighted":
                return self._weighted_combination(predictions)
            
            return None
            
        except Exception as e:
            logger.error(f"Ensemble prediction error for {symbol}: {e}")
            return None
    
    def _voting_combination(
        self,
        predictions: list[Any],
        probabilities: list[NDArray],
    ) -> dict[str, Any]:
        """Combine by majority voting."""
        # Count votes for each class
        votes = {}
        for pred in predictions:
            pred_int = int(pred)
            votes[pred_int] = votes.get(pred_int, 0) + 1
        
        # Find majority
        max_votes = max(votes.values())
        total_models = len(predictions)
        agreement = max_votes / total_models
        
        # Get winning class
        winning_class = max(votes, key=votes.get)
        
        # Average probabilities if available
        avg_prob = None
        if probabilities:
            avg_prob = np.mean(probabilities, axis=0)
        
        return {
            "class": winning_class,
            "confidence": agreement,
            "votes": votes,
            "probabilities": avg_prob.tolist() if avg_prob is not None else None,
        }
    
    def _average_combination(
        self,
        predictions: list[Any],
    ) -> dict[str, Any]:
        """Combine by averaging."""
        avg_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        return {
            "return": float(avg_pred),
            "std": float(std_pred),
            "all_predictions": [float(p) for p in predictions],
        }
    
    def _weighted_combination(
        self,
        predictions: list[Any],
    ) -> dict[str, Any]:
        """Combine by weighted average."""
        weights = self.config.model_weights
        weighted_pred = sum(p * w for p, w in zip(predictions, weights))
        
        return {
            "return": float(weighted_pred),
            "all_predictions": [float(p) for p in predictions],
            "weights": weights,
        }


# =============================================================================
# NEURAL NETWORK STRATEGY
# =============================================================================

@dataclass
class NeuralNetConfig(MLStrategyConfig):
    """Configuration for neural network strategy."""
    name: str = "NeuralNet"
    
    # Model architecture (for reference)
    input_dim: int = 0
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    output_dim: int = 1
    
    # Inference
    use_gpu: bool = False
    batch_inference: bool = False


class NeuralNetStrategy(BaseMLStrategy):
    """
    Neural Network Strategy.
    
    Uses deep learning models (LSTM, Transformer, MLP) for prediction.
    
    Supports:
        - PyTorch models
        - TensorFlow/Keras models
    
    Example:
        import torch
        
        model = MyLSTMModel()
        model.load_state_dict(torch.load("model.pt"))
        
        config = NeuralNetConfig()
        strategy = NeuralNetStrategy(config, model=model)
    """
    
    def __init__(
        self,
        config: NeuralNetConfig | None = None,
        parameters: dict[str, Any] | None = None,
        model: Any | None = None,
    ):
        """Initialize neural network strategy."""
        super().__init__(config or NeuralNetConfig(), parameters, model)
        self.config: NeuralNetConfig = self.config
        
        self._device = "cpu"
        if self.config.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self._device = "cuda"
            except ImportError:
                pass
    
    def _load_model(self, path: str) -> None:
        """Load neural network from path."""
        try:
            import torch
            
            self._model = torch.load(path, map_location=self._device)
            self._model.eval()
            self._model_loaded = True
            logger.info(f"Neural network loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load neural network: {e}")
            raise ModelNotTrainedError(f"Failed to load model: {e}")
    
    def _generate_prediction(
        self,
        symbol: str,
        features: NDArray[np.float64],
    ) -> dict[str, Any] | None:
        """Generate neural network prediction."""
        try:
            import torch
            
            # Convert to tensor
            x = torch.FloatTensor(features).to(self._device)
            
            # Inference
            with torch.no_grad():
                output = self._model(x)
                
                if isinstance(output, tuple):
                    output = output[0]
                
                pred = output.cpu().numpy().flatten()
            
            # Interpret output based on prediction type
            if self.config.prediction_type == PredictionType.CLASSIFICATION:
                if len(pred) > 1:
                    # Softmax output
                    pred_class = np.argmax(pred) - 1  # Map to -1, 0, 1
                    confidence = np.max(pred)
                else:
                    # Sigmoid output
                    pred_class = 1 if pred[0] > 0.5 else -1
                    confidence = abs(pred[0] - 0.5) * 2
                
                return {
                    "class": int(pred_class),
                    "confidence": float(confidence),
                    "raw_output": pred.tolist(),
                }
            
            else:  # Regression
                return {
                    "return": float(pred[0]),
                    "raw_output": pred.tolist(),
                }
            
        except Exception as e:
            logger.error(f"Neural network prediction error for {symbol}: {e}")
            return None


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_ml_strategy(
    strategy_type: str,
    model: Any | None = None,
    config: dict[str, Any] | None = None,
) -> BaseMLStrategy:
    """
    Factory function to create ML strategies.
    
    Args:
        strategy_type: Type of ML strategy
        model: Pre-trained model
        config: Configuration dictionary
    
    Returns:
        ML strategy instance
    """
    config_dict = config or {}
    
    if strategy_type == "classifier":
        cfg = MLClassifierConfig(**config_dict)
        return MLClassifierStrategy(cfg, model=model)
    
    elif strategy_type == "regressor":
        cfg = MLRegressorConfig(**config_dict)
        return MLRegressorStrategy(cfg, model=model)
    
    elif strategy_type == "ensemble":
        cfg = EnsembleMLConfig(**config_dict)
        return EnsembleMLStrategy(cfg, models=model if isinstance(model, list) else [model])
    
    elif strategy_type == "neural_net":
        cfg = NeuralNetConfig(**config_dict)
        return NeuralNetStrategy(cfg, model=model)
    
    else:
        raise ValueError(f"Unknown ML strategy type: {strategy_type}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Types
    "MLModel",
    "PredictionType",
    # Configurations
    "MLStrategyConfig",
    "MLClassifierConfig",
    "MLRegressorConfig",
    "EnsembleMLConfig",
    "NeuralNetConfig",
    # Strategies
    "BaseMLStrategy",
    "MLClassifierStrategy",
    "MLRegressorStrategy",
    "EnsembleMLStrategy",
    "NeuralNetStrategy",
    # Factory
    "create_ml_strategy",
]