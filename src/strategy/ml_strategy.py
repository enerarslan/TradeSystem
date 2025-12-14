"""
ML-Based Trading Strategies
JPMorgan-Level Machine Learning Trading

Features:
- ML model signal generation
- Ensemble model strategies
- Feature importance integration
- Adaptive signal thresholds
- CatBoost model wrapper for direct pkl loading
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime

from .base_strategy import (
    BaseStrategy, Signal, SignalType, StrategyConfig
)
from ..models.base_model import BaseModel
from ..features.builder import FeatureBuilder
from ..features.technical import TechnicalIndicators
from ..utils.logger import get_logger


logger = get_logger(__name__)


class CatBoostModelWrapper:
    """
    Wrapper for raw CatBoost models loaded from pickle files.

    This wrapper provides a consistent interface compatible with BaseModel
    for CatBoost models that were saved directly (not through our BaseModel framework).
    """

    def __init__(self, model, feature_names: Optional[List[str]] = None):
        """
        Initialize wrapper.

        Args:
            model: Raw CatBoost model object
            feature_names: List of feature names the model was trained on
        """
        self._model = model
        self._feature_names = feature_names or []
        self.model_type = 'catboost'
        self._is_trained = True

        # Try to get feature names from model if not provided
        if not self._feature_names:
            try:
                self._feature_names = list(model.feature_names_)
            except (AttributeError, TypeError):
                pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        # Align features if we know the expected feature names
        if self._feature_names:
            available = [f for f in self._feature_names if f in X.columns]
            if available:
                X = X[available]

        predictions = self._model.predict(X)

        # Flatten if needed
        if hasattr(predictions, 'flatten'):
            predictions = predictions.flatten()

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities"""
        # Align features if we know the expected feature names
        if self._feature_names:
            available = [f for f in self._feature_names if f in X.columns]
            if available:
                X = X[available]

        try:
            return self._model.predict_proba(X)
        except AttributeError:
            # Regressor - return predictions as probabilities
            preds = self._model.predict(X)
            return np.column_stack([1 - preds, preds])

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        try:
            importance = self._model.feature_importances_
            if self._feature_names and len(self._feature_names) == len(importance):
                return dict(zip(self._feature_names, importance))
            return {f'feature_{i}': v for i, v in enumerate(importance)}
        except (AttributeError, TypeError):
            return {}


def load_model_from_path(model_path: str, feature_list: Optional[List[str]] = None) -> Union[BaseModel, CatBoostModelWrapper]:
    """
    Load a model from a pickle file path.

    Handles both BaseModel subclasses and raw sklearn/catboost models.

    Args:
        model_path: Path to the pickle file
        feature_list: Optional list of feature names

    Returns:
        Model object (BaseModel or CatBoostModelWrapper)
    """
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(path, 'rb') as f:
        loaded = pickle.load(f)

    # Check if it's our BaseModel format (dict with 'model' key)
    if isinstance(loaded, dict) and 'model' in loaded:
        # This is our standard format
        model = loaded['model']
        feature_names = loaded.get('feature_names', feature_list or [])

        # Return wrapped model
        return CatBoostModelWrapper(model, feature_names)

    # Check if it's a BaseModel subclass
    if isinstance(loaded, BaseModel):
        return loaded

    # It's a raw model (CatBoost, XGBoost, etc.)
    return CatBoostModelWrapper(loaded, feature_list)


class MLStrategy(BaseStrategy):
    """
    Machine learning based trading strategy.

    Uses trained ML model to generate trading signals
    based on computed features.
    """

    def __init__(
        self,
        model: Optional[BaseModel] = None,
        model_path: Optional[str] = None,
        feature_builder: Optional[FeatureBuilder] = None,
        feature_list: Optional[List[str]] = None,
        min_confidence: float = 0.6,
        signal_threshold: float = 0.5,
        use_probability: bool = True,
        **kwargs
    ):
        """
        Initialize MLStrategy.

        Args:
            model: Trained ML model (optional if model_path provided)
            model_path: Path to model pickle file (optional if model provided)
            feature_builder: Feature engineering builder
            feature_list: List of feature names for consistency (from features.txt)
            min_confidence: Minimum prediction confidence
            signal_threshold: Threshold for signal generation
            use_probability: Use probability output for strength
        """
        config = StrategyConfig(name=kwargs.pop('name', 'ml_strategy'), **kwargs)
        super().__init__(config)

        # Load model from path if provided
        if model is None and model_path is not None:
            self.model = load_model_from_path(model_path, feature_list)
            logger.info(f"Loaded model from {model_path}")
        else:
            self.model = model

        self.feature_builder = feature_builder or FeatureBuilder()
        self.feature_list = feature_list or []
        self.min_confidence = min_confidence
        self.signal_threshold = signal_threshold
        self.use_probability = use_probability

        self._feature_cache: Dict[str, pd.DataFrame] = {}
        self._prediction_history: List[Dict[str, Any]] = []

        # Log feature list info
        if self.feature_list:
            logger.info(f"MLStrategy initialized with {len(self.feature_list)} required features")

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

        # FIXED: Use feature_list from features.txt if available for feature alignment
        expected_features = None
        if self.feature_list:
            expected_features = self.feature_list
        elif hasattr(self.model, '_feature_names') and self.model._feature_names:
            expected_features = self.model._feature_names

        if expected_features:
            available = [f for f in expected_features if f in latest_features.columns]

            if len(available) < len(expected_features) * 0.5:
                logger.warning(
                    f"Insufficient features for {symbol}: "
                    f"{len(available)}/{len(expected_features)} available"
                )
                return SignalType.FLAT, 0.0

            # Reorder to match expected feature order
            latest_features = latest_features[available]

            # Fill missing features with 0 (for features that couldn't be computed)
            missing = [f for f in expected_features if f not in available]
            if missing:
                logger.debug(f"Missing {len(missing)} features for {symbol}, filling with 0")
                for f in missing:
                    latest_features[f] = 0.0
                # Reorder to expected order
                latest_features = latest_features[[f for f in expected_features if f in latest_features.columns]]

        # Make prediction
        prediction = self.model.predict(latest_features)[0]

        # Get probability/confidence
        if self.use_probability:
            try:
                proba = self.model.predict_proba(latest_features)
                if proba.ndim > 1:
                    confidence = np.max(proba[0])
                else:
                    confidence = abs(proba[0] - 0.5) * 2
            except Exception as e:
                logger.debug(f"Could not get probabilities: {e}")
                confidence = 0.7  # Default confidence
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


# =============================================================================
# CATEGORY 4 FIXES: Signal Generation Improvements
# =============================================================================

class AdaptiveSignalThreshold:
    """
    ISSUE 4.1 FIX: Adaptive signal threshold based on recent accuracy.

    Fixed min_confidence=0.6 and signal_threshold=0.5 don't adapt to market
    conditions. This class adjusts thresholds based on recent performance.
    """

    def __init__(
        self,
        base_threshold: float = 0.6,
        lookback: int = 100,
        target_accuracy: float = 0.55,
        min_threshold: float = 0.5,
        max_threshold: float = 0.8
    ):
        """
        Initialize AdaptiveSignalThreshold.

        Args:
            base_threshold: Base probability threshold
            lookback: Number of trades to look back
            target_accuracy: Target accuracy for threshold adjustment
            min_threshold: Minimum allowed threshold
            max_threshold: Maximum allowed threshold
        """
        self.base_threshold = base_threshold
        self.lookback = lookback
        self.target_accuracy = target_accuracy
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        self.predictions_history: List[float] = []
        self.outcomes_history: List[int] = []  # 1 = correct, 0 = incorrect

    def update(self, prediction: float, outcome: int) -> None:
        """
        Update history with new prediction and outcome.

        Args:
            prediction: Predicted probability
            outcome: 1 if prediction was correct, 0 otherwise
        """
        self.predictions_history.append(prediction)
        self.outcomes_history.append(outcome)

        # Keep only lookback window
        if len(self.predictions_history) > self.lookback:
            self.predictions_history.pop(0)
            self.outcomes_history.pop(0)

    def get_threshold(self) -> float:
        """
        Get current adaptive threshold.

        Returns:
            Adjusted threshold based on recent accuracy
        """
        if len(self.predictions_history) < self.lookback:
            return self.base_threshold

        # Calculate recent accuracy
        recent_accuracy = np.mean(self.outcomes_history)

        # Adjust threshold based on accuracy gap
        # If accuracy is low, increase threshold (be more selective)
        # If accuracy is high, can decrease threshold (more trades)
        accuracy_gap = self.target_accuracy - recent_accuracy
        adjustment = accuracy_gap * 2  # Scale factor

        new_threshold = self.base_threshold + adjustment
        new_threshold = np.clip(new_threshold, self.min_threshold, self.max_threshold)

        return new_threshold

    def get_stats(self) -> Dict[str, Any]:
        """Get threshold statistics."""
        return {
            'current_threshold': self.get_threshold(),
            'base_threshold': self.base_threshold,
            'recent_accuracy': np.mean(self.outcomes_history) if self.outcomes_history else None,
            'n_samples': len(self.predictions_history)
        }


class SignalCooldown:
    """
    ISSUE 4.2 FIX: Signal cooldown to prevent rapid signal switching.

    Without cooldown, rapid signal switching increases transaction costs.
    This class enforces minimum time between signals.
    """

    def __init__(
        self,
        min_bars_between_signals: int = 4,
        min_bars_same_direction: int = 2
    ):
        """
        Initialize SignalCooldown.

        Args:
            min_bars_between_signals: Minimum bars between any signals
            min_bars_same_direction: Minimum bars before same-direction signal
        """
        self.min_bars_between_signals = min_bars_between_signals
        self.min_bars_same_direction = min_bars_same_direction

        # Track last signal for each symbol
        self._last_signal_time: Dict[str, datetime] = {}
        self._last_signal_direction: Dict[str, str] = {}

    def can_signal(
        self,
        symbol: str,
        signal_direction: str,
        current_time: datetime,
        bar_duration_minutes: int = 15
    ) -> Tuple[bool, str]:
        """
        Check if signal is allowed based on cooldown.

        Args:
            symbol: Trading symbol
            signal_direction: 'long', 'short', or 'flat'
            current_time: Current timestamp
            bar_duration_minutes: Duration of one bar in minutes

        Returns:
            Tuple of (can_signal, reason)
        """
        last_time = self._last_signal_time.get(symbol)
        last_direction = self._last_signal_direction.get(symbol)

        if last_time is None:
            return True, "no_previous_signal"

        # Calculate bars since last signal
        time_diff = (current_time - last_time).total_seconds() / 60
        bars_since = time_diff / bar_duration_minutes

        # Check minimum bars between signals
        if bars_since < self.min_bars_between_signals:
            return False, f"cooldown: {self.min_bars_between_signals - bars_since:.1f} bars remaining"

        # Check same direction constraint
        if signal_direction == last_direction and bars_since < self.min_bars_same_direction:
            return False, f"same_direction_cooldown: {self.min_bars_same_direction - bars_since:.1f} bars"

        return True, "allowed"

    def record_signal(
        self,
        symbol: str,
        signal_direction: str,
        signal_time: datetime
    ) -> None:
        """Record that a signal was generated."""
        self._last_signal_time[symbol] = signal_time
        self._last_signal_direction[symbol] = signal_direction

    def get_cooldown_status(self, symbol: str) -> Dict[str, Any]:
        """Get cooldown status for a symbol."""
        return {
            'last_signal_time': self._last_signal_time.get(symbol),
            'last_signal_direction': self._last_signal_direction.get(symbol)
        }


class SignalDecay:
    """
    ISSUE 4.3 FIX: Signal decay over time.

    Old signals should decay over time, not remain at full strength.
    This class applies exponential decay to signal strength.
    """

    def __init__(
        self,
        decay_rate: float = 0.1,
        min_strength: float = 0.1
    ):
        """
        Initialize SignalDecay.

        Args:
            decay_rate: Decay rate per bar (0.1 = 10% decay per bar)
            min_strength: Minimum signal strength before signal is cancelled
        """
        self.decay_rate = decay_rate
        self.min_strength = min_strength

        # Track active signals
        self._active_signals: Dict[str, Dict[str, Any]] = {}

    def add_signal(
        self,
        symbol: str,
        signal_strength: float,
        signal_direction: str,
        entry_time: datetime
    ) -> None:
        """Add a new signal to track."""
        self._active_signals[symbol] = {
            'initial_strength': signal_strength,
            'current_strength': signal_strength,
            'direction': signal_direction,
            'entry_time': entry_time,
            'bars_held': 0
        }

    def update(self, symbol: str, bars_elapsed: int = 1) -> float:
        """
        Update signal strength after bars have passed.

        Args:
            symbol: Trading symbol
            bars_elapsed: Number of bars since last update

        Returns:
            Updated signal strength
        """
        if symbol not in self._active_signals:
            return 0.0

        signal = self._active_signals[symbol]
        signal['bars_held'] += bars_elapsed

        # Apply exponential decay
        decay_factor = np.exp(-self.decay_rate * signal['bars_held'])
        signal['current_strength'] = signal['initial_strength'] * decay_factor

        return signal['current_strength']

    def get_decayed_strength(self, symbol: str, bars_held: int) -> float:
        """
        Get decayed signal strength.

        Args:
            symbol: Trading symbol
            bars_held: Bars since signal was generated

        Returns:
            Decayed signal strength
        """
        if symbol not in self._active_signals:
            return 0.0

        initial = self._active_signals[symbol]['initial_strength']
        decay_factor = np.exp(-self.decay_rate * bars_held)
        return initial * decay_factor

    def is_signal_expired(self, symbol: str) -> bool:
        """Check if signal has decayed below minimum threshold."""
        if symbol not in self._active_signals:
            return True

        return self._active_signals[symbol]['current_strength'] < self.min_strength

    def remove_signal(self, symbol: str) -> None:
        """Remove a signal from tracking."""
        self._active_signals.pop(symbol, None)


class DynamicWeightedEnsemble:
    """
    ISSUE 4.4 FIX: Dynamic ensemble weighting based on recent performance.

    Fixed weights don't adapt to changing market conditions.
    This class adjusts strategy weights based on recent performance.
    """

    def __init__(
        self,
        strategy_names: List[str],
        initial_weights: Optional[List[float]] = None,
        lookback: int = 100,
        method: str = 'inverse_volatility'
    ):
        """
        Initialize DynamicWeightedEnsemble.

        Args:
            strategy_names: Names of strategies in ensemble
            initial_weights: Initial weights (equal if not provided)
            lookback: Lookback period for performance calculation
            method: Weighting method ('inverse_volatility', 'sharpe', 'equal')
        """
        self.strategy_names = strategy_names
        self.n_strategies = len(strategy_names)
        self.lookback = lookback
        self.method = method

        if initial_weights is None:
            initial_weights = [1.0 / self.n_strategies] * self.n_strategies

        self.current_weights = dict(zip(strategy_names, initial_weights))
        self._returns_history: Dict[str, List[float]] = {s: [] for s in strategy_names}

    def update_returns(self, strategy_returns: Dict[str, float]) -> None:
        """
        Update returns history with new returns.

        Args:
            strategy_returns: Dict mapping strategy name to return
        """
        for strategy, ret in strategy_returns.items():
            if strategy in self._returns_history:
                self._returns_history[strategy].append(ret)

                # Keep only lookback window
                if len(self._returns_history[strategy]) > self.lookback:
                    self._returns_history[strategy].pop(0)

    def update_weights(self) -> Dict[str, float]:
        """
        Update weights based on recent performance.

        Returns:
            Updated weights dictionary
        """
        if not all(len(r) >= 20 for r in self._returns_history.values()):
            # Not enough history, keep current weights
            return self.current_weights

        if self.method == 'inverse_volatility':
            self.current_weights = self._inverse_volatility_weights()
        elif self.method == 'sharpe':
            self.current_weights = self._sharpe_weights()
        elif self.method == 'equal':
            self.current_weights = {s: 1.0 / self.n_strategies for s in self.strategy_names}
        else:
            raise ValueError(f"Unknown weighting method: {self.method}")

        return self.current_weights

    def _inverse_volatility_weights(self) -> Dict[str, float]:
        """Calculate inverse-volatility weights."""
        volatilities = {}
        for strategy, returns in self._returns_history.items():
            vol = np.std(returns) if len(returns) > 1 else 1.0
            volatilities[strategy] = max(vol, 0.001)  # Prevent division by zero

        total_inv_vol = sum(1 / v for v in volatilities.values())

        weights = {}
        for strategy, vol in volatilities.items():
            weights[strategy] = (1 / vol) / total_inv_vol

        return weights

    def _sharpe_weights(self) -> Dict[str, float]:
        """Calculate Sharpe-ratio based weights."""
        sharpes = {}
        for strategy, returns in self._returns_history.items():
            if len(returns) < 2:
                sharpes[strategy] = 0
            else:
                mean_ret = np.mean(returns)
                std_ret = np.std(returns)
                sharpes[strategy] = mean_ret / std_ret if std_ret > 0 else 0

        # Convert to positive weights
        min_sharpe = min(sharpes.values())
        if min_sharpe < 0:
            sharpes = {s: sh - min_sharpe + 0.1 for s, sh in sharpes.items()}

        total_sharpe = sum(sharpes.values())
        if total_sharpe <= 0:
            return {s: 1.0 / self.n_strategies for s in self.strategy_names}

        return {s: sh / total_sharpe for s, sh in sharpes.items()}

    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        return self.current_weights

    def get_weighted_signal(
        self,
        strategy_signals: Dict[str, float]
    ) -> float:
        """
        Get weighted ensemble signal.

        Args:
            strategy_signals: Dict mapping strategy name to signal value

        Returns:
            Weighted average signal
        """
        weighted_sum = 0.0
        weight_sum = 0.0

        for strategy, signal in strategy_signals.items():
            if strategy in self.current_weights:
                weight = self.current_weights[strategy]
                weighted_sum += signal * weight
                weight_sum += weight

        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
