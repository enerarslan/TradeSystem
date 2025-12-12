"""
ML-Based Trading Strategies
JPMorgan-Level Machine Learning Trading

Features:
- ML model signal generation
- Ensemble model strategies
- Feature importance integration
- Adaptive signal thresholds
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from .base_strategy import (
    BaseStrategy, Signal, SignalType, StrategyConfig
)
from ..models.base_model import BaseModel
from ..features.builder import FeatureBuilder
from ..features.technical import TechnicalIndicators
from ..utils.logger import get_logger


logger = get_logger(__name__)


class MLStrategy(BaseStrategy):
    """
    Machine learning based trading strategy.

    Uses trained ML model to generate trading signals
    based on computed features.
    """

    def __init__(
        self,
        model: Optional[BaseModel] = None,
        feature_builder: Optional[FeatureBuilder] = None,
        min_confidence: float = 0.6,
        signal_threshold: float = 0.5,
        use_probability: bool = True,
        **kwargs
    ):
        """
        Initialize MLStrategy.

        Args:
            model: Trained ML model
            feature_builder: Feature engineering builder
            min_confidence: Minimum prediction confidence
            signal_threshold: Threshold for signal generation
            use_probability: Use probability output for strength
        """
        config = StrategyConfig(name="ml_strategy", **kwargs)
        super().__init__(config)

        self.model = model
        self.feature_builder = feature_builder or FeatureBuilder()
        self.min_confidence = min_confidence
        self.signal_threshold = signal_threshold
        self.use_probability = use_probability

        self._feature_cache: Dict[str, pd.DataFrame] = {}
        self._prediction_history: List[Dict[str, Any]] = []

    def set_model(self, model: BaseModel) -> None:
        """Set the ML model"""
        self.model = model
        logger.info(f"Model set: {model.model_type}")

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Signal]:
        """Generate ML-based trading signals"""
        if self.model is None:
            logger.warning("No model set for ML strategy")
            return {}

        signals = {}

        for symbol, df in data.items():
            try:
                signal_type, strength = self.calculate_signal_strength(df, symbol)

                if signal_type != SignalType.FLAT:
                    signal = self._create_signal(df, symbol, signal_type, strength)

                    if self.validate_signal(signal):
                        signals[symbol] = signal
                        self.record_signal(signal)

            except Exception as e:
                logger.warning(f"Error in ML strategy for {symbol}: {e}")

        return signals

    def calculate_signal_strength(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> Tuple[SignalType, float]:
        """Calculate ML-based signal"""
        if len(df) < 50:
            return SignalType.FLAT, 0.0

        # Build features
        features = self.feature_builder.build_features(df)

        # Get latest feature row
        latest_features = features.iloc[[-1]].dropna(axis=1)

        # Ensure features match model's expected features
        if hasattr(self.model, '_feature_names'):
            expected_features = self.model._feature_names
            available = [f for f in expected_features if f in latest_features.columns]

            if len(available) < len(expected_features) * 0.8:
                logger.warning(f"Insufficient features for {symbol}")
                return SignalType.FLAT, 0.0

            latest_features = latest_features[available]

        # Make prediction
        prediction = self.model.predict(latest_features)[0]

        # Get probability/confidence
        if self.use_probability:
            proba = self.model.predict_proba(latest_features)
            if proba.ndim > 1:
                confidence = np.max(proba[0])
            else:
                confidence = abs(proba[0] - 0.5) * 2
        else:
            confidence = 0.7  # Default confidence

        # Record prediction
        self._prediction_history.append({
            'symbol': symbol,
            'timestamp': df.index[-1],
            'prediction': prediction,
            'confidence': confidence
        })

        # Convert prediction to signal
        if confidence < self.min_confidence:
            return SignalType.FLAT, 0.0

        # Map prediction to signal type
        # Assuming prediction is -1, 0, 1 or similar
        if prediction == 1 or prediction > self.signal_threshold:
            signal_type = SignalType.LONG
        elif prediction == -1 or prediction < -self.signal_threshold:
            signal_type = SignalType.SHORT
        else:
            signal_type = SignalType.FLAT

        strength = confidence

        return signal_type, strength

    def _create_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        signal_type: SignalType,
        strength: float
    ) -> Signal:
        """Create ML-based signal"""
        current_price = df['close'].iloc[-1]
        direction = 1 if signal_type == SignalType.LONG else -1

        atr = TechnicalIndicators.atr(
            df['high'], df['low'], df['close'], 14
        ).iloc[-1]

        stop_loss = self.calculate_stop_loss(current_price, direction, atr)
        take_profit = self.calculate_take_profit(current_price, direction, stop_loss)

        # Get feature importance contribution
        feature_contributions = self._get_feature_contributions(df, symbol)

        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            timestamp=df.index[-1],
            price=current_price,
            strategy_name=self.name,
            confidence=strength,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'model_type': self.model.model_type,
                'top_features': feature_contributions[:5] if feature_contributions else []
            }
        )

    def _get_feature_contributions(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> List[Tuple[str, float]]:
        """Get top feature contributions for this prediction"""
        try:
            importance = self.model.get_feature_importance()

            if not importance:
                return []

            # Sort by importance
            sorted_features = sorted(
                importance.items(),
                key=lambda x: x[1],
                reverse=True
            )

            return sorted_features[:10]

        except:
            return []

    def get_prediction_accuracy(self) -> Dict[str, float]:
        """Calculate prediction accuracy from history"""
        if len(self._prediction_history) < 2:
            return {}

        # This would require actual returns data to calculate
        return {
            'total_predictions': len(self._prediction_history),
            'avg_confidence': np.mean([p['confidence'] for p in self._prediction_history])
        }


class EnsembleMLStrategy(BaseStrategy):
    """
    Ensemble ML strategy combining multiple models.

    Features:
    - Multiple model voting
    - Confidence-weighted signals
    - Model disagreement filtering
    """

    def __init__(
        self,
        models: List[BaseModel] = None,
        weights: List[float] = None,
        min_agreement: float = 0.6,
        min_confidence: float = 0.55,
        **kwargs
    ):
        """
        Initialize EnsembleMLStrategy.

        Args:
            models: List of trained models
            weights: Weight for each model
            min_agreement: Minimum model agreement ratio
            min_confidence: Minimum confidence threshold
        """
        config = StrategyConfig(name="ensemble_ml", **kwargs)
        super().__init__(config)

        self.models = models or []
        self.weights = weights
        self.min_agreement = min_agreement
        self.min_confidence = min_confidence

        self.feature_builder = FeatureBuilder()

        if self.weights is None and self.models:
            self.weights = [1.0 / len(self.models)] * len(self.models)

    def add_model(self, model: BaseModel, weight: float = 1.0) -> None:
        """Add model to ensemble"""
        self.models.append(model)

        if self.weights is None:
            self.weights = [weight]
        else:
            self.weights.append(weight)

        # Renormalize
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Signal]:
        """Generate ensemble signals"""
        if not self.models:
            logger.warning("No models in ensemble")
            return {}

        signals = {}

        for symbol, df in data.items():
            try:
                signal_type, strength = self.calculate_signal_strength(df, symbol)

                if signal_type != SignalType.FLAT:
                    signal = self._create_signal(df, symbol, signal_type, strength)

                    if self.validate_signal(signal):
                        signals[symbol] = signal
                        self.record_signal(signal)

            except Exception as e:
                logger.warning(f"Error in ensemble ML for {symbol}: {e}")

        return signals

    def calculate_signal_strength(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> Tuple[SignalType, float]:
        """Calculate ensemble signal"""
        if len(df) < 50:
            return SignalType.FLAT, 0.0

        # Build features
        features = self.feature_builder.build_features(df)
        latest_features = features.iloc[[-1]].dropna(axis=1)

        # Collect predictions from all models
        predictions = []
        confidences = []

        for i, model in enumerate(self.models):
            try:
                # Align features with model
                if hasattr(model, '_feature_names'):
                    available = [f for f in model._feature_names if f in latest_features.columns]
                    model_features = latest_features[available]
                else:
                    model_features = latest_features

                pred = model.predict(model_features)[0]
                proba = model.predict_proba(model_features)

                if proba.ndim > 1:
                    conf = np.max(proba[0])
                else:
                    conf = abs(proba[0] - 0.5) * 2

                predictions.append((pred, self.weights[i]))
                confidences.append(conf * self.weights[i])

            except Exception as e:
                logger.debug(f"Model {i} prediction failed: {e}")

        if not predictions:
            return SignalType.FLAT, 0.0

        # Weighted voting
        weighted_pred = sum(p * w for p, w in predictions)
        avg_confidence = sum(confidences)

        # Check agreement
        directions = [1 if p > 0 else (-1 if p < 0 else 0) for p, _ in predictions]
        agreement = max(directions.count(1), directions.count(-1)) / len(directions)

        if agreement < self.min_agreement:
            return SignalType.FLAT, 0.0

        if avg_confidence < self.min_confidence:
            return SignalType.FLAT, 0.0

        # Determine signal
        if weighted_pred > 0.3:
            signal_type = SignalType.LONG
        elif weighted_pred < -0.3:
            signal_type = SignalType.SHORT
        else:
            signal_type = SignalType.FLAT

        strength = min(abs(weighted_pred) * avg_confidence, 1.0)

        return signal_type, strength

    def _create_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        signal_type: SignalType,
        strength: float
    ) -> Signal:
        """Create ensemble signal"""
        current_price = df['close'].iloc[-1]
        direction = 1 if signal_type == SignalType.LONG else -1

        atr = TechnicalIndicators.atr(
            df['high'], df['low'], df['close'], 14
        ).iloc[-1]

        stop_loss = self.calculate_stop_loss(current_price, direction, atr)
        take_profit = self.calculate_take_profit(current_price, direction, stop_loss)

        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            timestamp=df.index[-1],
            price=current_price,
            strategy_name=self.name,
            confidence=strength,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'n_models': len(self.models),
                'model_types': [m.model_type for m in self.models]
            }
        )

    def get_model_contributions(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """Get individual model predictions"""
        features = self.feature_builder.build_features(df)
        latest_features = features.iloc[[-1]].dropna(axis=1)

        contributions = {}

        for i, model in enumerate(self.models):
            try:
                if hasattr(model, '_feature_names'):
                    available = [f for f in model._feature_names if f in latest_features.columns]
                    model_features = latest_features[available]
                else:
                    model_features = latest_features

                pred = model.predict(model_features)[0]
                proba = model.predict_proba(model_features)

                contributions[f"{model.model_type}_{i}"] = {
                    'prediction': pred,
                    'probability': proba[0].tolist() if proba.ndim > 1 else proba[0],
                    'weight': self.weights[i]
                }

            except Exception as e:
                contributions[f"{model.model_type}_{i}"] = {
                    'error': str(e)
                }

        return contributions
