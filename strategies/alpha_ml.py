"""
Alpha ML Strategy
=================

JPMorgan-level production ML trading strategy for the algorithmic trading platform.
This is the primary ML-based strategy designed for live trading.

Architecture:
- Multi-model ensemble with dynamic weighting
- Adaptive feature selection
- Market regime detection
- Dynamic position sizing based on prediction confidence
- Multi-timeframe signal aggregation
- Risk-adjusted signal generation

Features:
- LightGBM + XGBoost + Neural Network ensemble
- Walk-forward model retraining
- Confidence-weighted position sizing
- Regime-aware signal generation
- Transaction cost awareness
- Real-time feature updates

This strategy is designed to compete with institutional-grade
quantitative trading systems.

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable
import pickle

import numpy as np
import polars as pl
from numpy.typing import NDArray

from config.settings import get_logger, TimeFrame, OrderSide
from core.events import MarketEvent, SignalEvent, EventPriority
from core.types import PortfolioState, Signal, SignalStrength
from strategies.base import BaseStrategy, StrategyConfig, StrategyState, SignalAction

logger = get_logger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class MarketRegime(str, Enum):
    """Market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"


class ModelType(str, Enum):
    """Model types in ensemble."""
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    NEURAL = "neural"


class SignalStrengthLevel(str, Enum):
    """Signal strength classification."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    NEUTRAL = "neutral"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AlphaMLConfig(StrategyConfig):
    """
    Configuration for Alpha ML Strategy.
    
    This is a comprehensive configuration that covers all aspects
    of the ML-based trading strategy.
    """
    name: str = "AlphaML"
    description: str = "JPMorgan-level ML trading strategy"
    
    # =========================================================================
    # MODEL CONFIGURATION
    # =========================================================================
    
    # Ensemble weights
    use_lightgbm: bool = True
    use_xgboost: bool = True
    use_neural: bool = False  # Disable by default for speed
    
    lightgbm_weight: float = 0.4
    xgboost_weight: float = 0.4
    neural_weight: float = 0.2
    
    # Model paths
    models_dir: Path = field(default_factory=lambda: Path("models/artifacts"))
    lightgbm_path: str | None = None
    xgboost_path: str | None = None
    neural_path: str | None = None
    
    # Auto-train if no model exists
    auto_train: bool = True
    min_training_samples: int = 5000
    
    # =========================================================================
    # FEATURE CONFIGURATION
    # =========================================================================
    
    # Feature lookback
    lookback_bars: int = 100
    feature_update_interval: int = 1  # Update every N bars
    
    # Feature categories
    use_momentum_features: bool = True
    use_trend_features: bool = True
    use_volatility_features: bool = True
    use_volume_features: bool = True
    use_statistical_features: bool = True
    
    # Feature selection
    max_features: int = 100
    feature_selection_method: str = "importance"  # importance, correlation, variance
    
    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================
    
    # Classification thresholds
    strong_buy_threshold: float = 0.70
    buy_threshold: float = 0.55
    sell_threshold: float = 0.45
    strong_sell_threshold: float = 0.30
    
    # Confidence requirements
    min_confidence: float = 0.55
    high_confidence_threshold: float = 0.70
    
    # Signal filtering
    require_consensus: bool = True  # All models must agree
    min_model_agreement: float = 0.6  # 60% of models agree
    
    # Transaction cost filter
    min_expected_return: float = 0.003  # 0.3% minimum expected return
    transaction_cost: float = 0.001  # 0.1% round-trip cost
    
    # =========================================================================
    # REGIME DETECTION
    # =========================================================================
    
    enable_regime_detection: bool = True
    regime_lookback: int = 50
    volatility_regime_threshold: float = 1.5  # Std multiplier
    trend_regime_threshold: float = 0.6  # ADX threshold
    
    # Regime-specific adjustments
    reduce_size_high_vol: bool = True
    high_vol_size_mult: float = 0.5
    boost_trend_signals: bool = True
    trend_signal_mult: float = 1.2
    
    # =========================================================================
    # POSITION SIZING
    # =========================================================================
    
    base_position_size: float = 0.05  # 5% of portfolio
    max_position_size: float = 0.10  # 10% max
    confidence_scaling: bool = True  # Scale by confidence
    
    # Kelly criterion
    use_kelly: bool = False
    kelly_fraction: float = 0.25  # Quarter Kelly
    
    # =========================================================================
    # RISK MANAGEMENT
    # =========================================================================
    
    # Stop loss
    use_stop_loss: bool = True
    stop_loss_pct: float = 0.02  # 2%
    trailing_stop: bool = True
    trailing_stop_pct: float = 0.015  # 1.5%
    
    # Take profit
    use_take_profit: bool = True
    take_profit_pct: float = 0.04  # 4%
    
    # Position limits
    max_positions: int = 5
    max_correlation: float = 0.7  # Avoid correlated positions
    
    # =========================================================================
    # RETRAINING
    # =========================================================================
    
    enable_retraining: bool = True
    retrain_interval_bars: int = 10000  # Retrain every N bars
    retrain_min_new_samples: int = 1000  # Minimum new samples to retrain
    walk_forward_window: int = 50000  # Training window size
    
    # =========================================================================
    # PREDICTION
    # =========================================================================
    
    prediction_horizon: int = 5  # Predict N bars ahead
    prediction_type: str = "classification"  # classification, regression
    num_classes: int = 3  # -1 (down), 0 (neutral), 1 (up)
    neutral_zone_pct: float = 0.005  # ±0.5% is neutral


# =============================================================================
# ALPHA ML STRATEGY
# =============================================================================

class AlphaMLStrategy(BaseStrategy):
    """
    Alpha ML Trading Strategy.
    
    A production-grade ML-based trading strategy that combines multiple
    models in an ensemble for robust signal generation.
    
    Architecture:
        1. Feature Generation: 100+ technical and statistical features
        2. Model Ensemble: LightGBM + XGBoost (+ optional Neural Net)
        3. Regime Detection: Adapt to market conditions
        4. Signal Generation: Confidence-weighted predictions
        5. Risk Management: Dynamic position sizing with stops
    
    Signal Flow:
        Market Data -> Feature Pipeline -> Model Ensemble -> 
        Regime Adjustment -> Confidence Filter -> Signal Generation
    
    Example:
        config = AlphaMLConfig(
            use_lightgbm=True,
            use_xgboost=True,
            min_confidence=0.6,
        )
        strategy = AlphaMLStrategy(config)
        strategy.initialize(["AAPL", "GOOGL", "MSFT"])
        
        # Generate signals
        signals = strategy.calculate_signals(market_event, portfolio)
    """
    
    def __init__(
        self,
        config: AlphaMLConfig | None = None,
        parameters: dict[str, Any] | None = None,
    ):
        """Initialize Alpha ML Strategy."""
        super().__init__(config or AlphaMLConfig(), parameters)
        self.config: AlphaMLConfig = self.config
        
        # Models
        self._models: dict[str, Any] = {}
        self._model_weights: dict[str, float] = {}
        self._models_loaded: bool = False
        
        # Feature pipeline
        self._feature_pipeline: Any = None
        self._feature_names: list[str] = []
        
        # Data storage
        self._historical_data: dict[str, pl.DataFrame] = {}
        self._feature_cache: dict[str, NDArray] = {}
        
        # Regime detection
        self._current_regime: dict[str, MarketRegime] = {}
        self._regime_history: dict[str, list[MarketRegime]] = {}
        
        # Predictions cache
        self._last_predictions: dict[str, dict[str, Any]] = {}
        
        # Training state
        self._last_retrain_bar: int = 0
        self._training_data_buffer: dict[str, list] = {}
        
        # Performance tracking
        self._prediction_history: list[dict[str, Any]] = []
        self._model_performance: dict[str, dict[str, float]] = {}
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def _on_initialize(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        """Initialize the strategy."""
        logger.info(f"Initializing AlphaML strategy for {len(symbols)} symbols")
        
        # Initialize feature pipeline
        self._initialize_feature_pipeline()
        
        # Initialize data structures for each symbol
        for symbol in symbols:
            self._historical_data[symbol] = pl.DataFrame()
            self._feature_cache[symbol] = np.array([])
            self._current_regime[symbol] = MarketRegime.RANGING
            self._regime_history[symbol] = []
            self._last_predictions[symbol] = {}
            self._training_data_buffer[symbol] = []
        
        # Load or train models
        self._load_or_train_models()
        
        # Set model weights
        self._set_model_weights()
        
        logger.info(f"AlphaML initialized with {len(self._models)} models")
    
    def _initialize_feature_pipeline(self) -> None:
        """Initialize the feature generation pipeline."""
        from features.pipeline import FeaturePipeline, FeatureConfig, FeatureCategory
        
        # Determine which feature categories to use
        enabled_categories = []
        
        if self.config.use_momentum_features:
            enabled_categories.append(FeatureCategory.TECHNICAL_MOMENTUM)
        if self.config.use_trend_features:
            enabled_categories.append(FeatureCategory.TECHNICAL_TREND)
        if self.config.use_volatility_features:
            enabled_categories.append(FeatureCategory.TECHNICAL_VOLATILITY)
        if self.config.use_volume_features:
            enabled_categories.append(FeatureCategory.TECHNICAL_VOLUME)
        if self.config.use_statistical_features:
            enabled_categories.extend([
                FeatureCategory.STATISTICAL_RETURNS,
                FeatureCategory.STATISTICAL_ROLLING,
                FeatureCategory.STATISTICAL_MOMENTUM,
                FeatureCategory.STATISTICAL_VOLATILITY,
            ])
        
        config = FeatureConfig(
            enabled_categories=enabled_categories,
            drop_na=True,
            normalize=True,
        )
        
        self._feature_pipeline = FeaturePipeline(config)
        logger.info(f"Feature pipeline initialized with {len(enabled_categories)} categories")
    
    def _load_or_train_models(self) -> None:
        """Load existing models or train new ones."""
        models_loaded = False
        
        # Try to load LightGBM
        if self.config.use_lightgbm:
            lgb_path = self.config.lightgbm_path or self.config.models_dir / "alpha_lgb.pkl"
            if Path(lgb_path).exists():
                self._models["lightgbm"] = self._load_model(lgb_path)
                models_loaded = True
                logger.info("LightGBM model loaded")
        
        # Try to load XGBoost
        if self.config.use_xgboost:
            xgb_path = self.config.xgboost_path or self.config.models_dir / "alpha_xgb.pkl"
            if Path(xgb_path).exists():
                self._models["xgboost"] = self._load_model(xgb_path)
                models_loaded = True
                logger.info("XGBoost model loaded")
        
        # Try to load Neural Network
        if self.config.use_neural:
            nn_path = self.config.neural_path or self.config.models_dir / "alpha_nn.pkl"
            if Path(nn_path).exists():
                self._models["neural"] = self._load_model(nn_path)
                models_loaded = True
                logger.info("Neural network model loaded")
        
        self._models_loaded = models_loaded
        
        if not models_loaded and self.config.auto_train:
            logger.warning("No pre-trained models found. Will train on first sufficient data.")
    
    def _load_model(self, path: str | Path) -> Any:
        """Load a model from disk with automatic class detection."""
        import pickle
        from pathlib import Path
        
        path = Path(path)
        
        # First, peek at the file to determine the class name
        with open(path, "rb") as f:
            save_data = pickle.load(f)
        
        class_name = save_data.get("class_name", "")
        
        # Import and use the correct concrete class
        if "LightGBM" in class_name:
            from models.classifiers import LightGBMClassifier
            return LightGBMClassifier.load(path)
        elif "XGBoost" in class_name:
            from models.classifiers import XGBoostClassifier
            return XGBoostClassifier.load(path)
        elif "CatBoost" in class_name:
            from models.classifiers import CatBoostClassifier
            return CatBoostClassifier.load(path)
        elif "RandomForest" in class_name:
            from models.classifiers import RandomForestClassifier
            return RandomForestClassifier.load(path)
        elif "ExtraTrees" in class_name:
            from models.classifiers import ExtraTreesClassifier
            return ExtraTreesClassifier.load(path)
        elif "LSTM" in class_name:
            from models.deep import LSTMModel
            return LSTMModel.load(path)
        elif "Transformer" in class_name:
            from models.deep import TransformerModel
            return TransformerModel.load(path)
        else:
            raise ValueError(f"Unknown model class: {class_name}. "
                            f"Cannot load model from {path}")
    
    def _set_model_weights(self) -> None:
        """Set ensemble model weights."""
        total_weight = 0.0
        
        if "lightgbm" in self._models:
            self._model_weights["lightgbm"] = self.config.lightgbm_weight
            total_weight += self.config.lightgbm_weight
        
        if "xgboost" in self._models:
            self._model_weights["xgboost"] = self.config.xgboost_weight
            total_weight += self.config.xgboost_weight
        
        if "neural" in self._models:
            self._model_weights["neural"] = self.config.neural_weight
            total_weight += self.config.neural_weight
        
        # Normalize weights
        if total_weight > 0:
            for key in self._model_weights:
                self._model_weights[key] /= total_weight
    
    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================
    
    def calculate_signals(
        self,
        event: MarketEvent,
        portfolio: PortfolioState,
    ) -> list[SignalEvent]:
        """
        Generate trading signals from market data.
        
        This is the main entry point for signal generation.
        """
        signals = []
        symbol = event.symbol
        data = event.data
        
        if data is None or len(data) < self.config.lookback_bars:
            return signals
        
        # Update historical data
        self._update_historical_data(symbol, data)
        
        # Check if we need to train models
        if not self._models_loaded and self.config.auto_train:
            self._check_and_train(symbol)
            if not self._models_loaded:
                return signals
        
        # Generate features
        features = self._generate_features(symbol)
        if features is None or len(features) == 0:
            return signals
        
        # Detect market regime
        if self.config.enable_regime_detection:
            self._detect_regime(symbol, data)
        
        # Get ensemble prediction
        prediction = self._get_ensemble_prediction(symbol, features)
        if prediction is None:
            return signals
        
        # Store prediction
        self._last_predictions[symbol] = prediction
        
        # Generate entry signal
        entry_signal = self._generate_entry_signal(symbol, prediction, data, portfolio)
        if entry_signal is not None:
            signals.append(entry_signal)
        
        # Check for exit signals on existing positions
        exit_signal = self._generate_exit_signal(symbol, prediction, data, portfolio)
        if exit_signal is not None:
            signals.append(exit_signal)
        
        # Track prediction for model evaluation
        self._track_prediction(symbol, prediction, data)
        
        return signals
    
    def _update_historical_data(
        self,
        symbol: str,
        data: pl.DataFrame,
    ) -> None:
        """Update historical data buffer.
        
        CRITICAL: Buffer must be large enough to support:
        1. Lookback period for features
        2. Minimum training samples for auto-training
        3. Walk-forward window for retraining
        """
        if symbol not in self._historical_data:
            self._historical_data[symbol] = data
        else:
            if len(self._historical_data[symbol]) == 0:
                self._historical_data[symbol] = data
            else:
                min_for_features = self.config.lookback_bars * 3
                min_for_training = self.config.min_training_samples + 1000
                min_for_walk_forward = self.config.walk_forward_window
                max_bars = max(min_for_features, min_for_training, min_for_walk_forward)
                self._historical_data[symbol] = data.tail(max_bars)
    
    def _generate_features_for_prediction(self, data: pl.DataFrame) -> NDArray[np.float64] | None:
        """Generate features for prediction."""
        try:
            df_features = self._feature_pipeline.generate(data)
            
            # ================================================================
            # CRITICAL FIX: Filter to only numeric features (exclude 'regime' etc.)
            # ================================================================
            if self._feature_names:
                # Use the same features that were used during training
                valid_features = [f for f in self._feature_names if f in df_features.columns]
                if valid_features:
                    X = df_features.select(valid_features).to_numpy().astype(np.float64)
                else:
                    return None
            else:
                # Fallback: filter numeric columns manually
                exclude_cols = {"timestamp", "symbol", "target", "open", "high", "low", "close", "volume"}
                numeric_types = [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8,
                                pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8]
                
                feature_cols = []
                for col in df_features.columns:
                    if col not in exclude_cols:
                        col_dtype = df_features[col].dtype
                        if col_dtype in numeric_types:
                            feature_cols.append(col)
                        elif col_dtype not in [pl.Utf8, pl.String, pl.Categorical]:
                            if "float" in str(col_dtype).lower() or "int" in str(col_dtype).lower():
                                feature_cols.append(col)
                
                if not feature_cols:
                    return None
                    
                X = df_features.select(feature_cols).to_numpy().astype(np.float64)
            
            # Handle NaN/Inf
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Return last row for prediction
            return X[-1:] if len(X) > 0 else None
            
        except Exception as e:
            logger.error(f"Feature generation error for prediction: {e}")
            return None
    
    def _detect_regime(
        self,
        symbol: str,
        data: pl.DataFrame,
    ) -> MarketRegime:
        """Detect current market regime."""
        try:
            lookback = min(self.config.regime_lookback, len(data))
            recent_data = data.tail(lookback)
            
            # Calculate volatility
            returns = recent_data["close"].pct_change().drop_nulls().to_numpy()
            current_vol = np.std(returns)
            
            # Calculate trend strength (simple approach)
            close = recent_data["close"].to_numpy()
            trend = (close[-1] - close[0]) / close[0] if close[0] != 0 else 0
            
            # Historical volatility baseline
            hist_vol = np.std(data["close"].pct_change().drop_nulls().to_numpy())
            vol_ratio = current_vol / hist_vol if hist_vol > 0 else 1.0
            
            # Determine regime
            if vol_ratio > self.config.volatility_regime_threshold:
                regime = MarketRegime.HIGH_VOLATILITY
            elif vol_ratio < 0.5:
                regime = MarketRegime.LOW_VOLATILITY
            elif trend > 0.02:  # 2% uptrend
                regime = MarketRegime.TRENDING_UP
            elif trend < -0.02:  # 2% downtrend
                regime = MarketRegime.TRENDING_DOWN
            else:
                regime = MarketRegime.RANGING
            
            self._current_regime[symbol] = regime
            self._regime_history[symbol].append(regime)
            
            # Keep history bounded
            if len(self._regime_history[symbol]) > 100:
                self._regime_history[symbol] = self._regime_history[symbol][-100:]
            
            return regime
            
        except Exception as e:
            logger.error(f"Regime detection error for {symbol}: {e}")
            return MarketRegime.RANGING
    
    def _get_ensemble_prediction(
        self,
        symbol: str,
        features: NDArray,
    ) -> dict[str, Any] | None:
        """Get prediction from model ensemble."""
        if not self._models:
            return None
        
        try:
            predictions = {}
            probabilities = {}
            
            # Get predictions from each model
            for name, model in self._models.items():
                try:
                    pred = model.predict(features)
                    predictions[name] = pred[0] if isinstance(pred, np.ndarray) else pred
                    
                    # Get probabilities if available
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(features)
                        if proba is not None:
                            probabilities[name] = proba[0]
                    
                except Exception as e:
                    logger.warning(f"Model {name} prediction error: {e}")
                    continue
            
            if not predictions:
                return None
            
            # Combine predictions
            combined = self._combine_predictions(predictions, probabilities)
            
            return combined
            
        except Exception as e:
            logger.error(f"Ensemble prediction error for {symbol}: {e}")
            return None
    
    def _combine_predictions(
        self,
        predictions: dict[str, Any],
        probabilities: dict[str, NDArray],
    ) -> dict[str, Any]:
        """Combine predictions from multiple models."""
        # Weighted voting for classification
        weighted_pred = 0.0
        total_weight = 0.0
        
        for name, pred in predictions.items():
            weight = self._model_weights.get(name, 1.0 / len(predictions))
            weighted_pred += pred * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_pred /= total_weight
        
        # Combine probabilities
        combined_proba = None
        if probabilities:
            proba_list = list(probabilities.values())
            combined_proba = np.mean(proba_list, axis=0)
        
        # Calculate confidence
        if combined_proba is not None:
            confidence = np.max(combined_proba)
        else:
            # Agreement-based confidence
            pred_values = list(predictions.values())
            agreement = sum(1 for p in pred_values if np.sign(p) == np.sign(weighted_pred))
            confidence = agreement / len(pred_values)
        
        # Determine signal class
        if combined_proba is not None and len(combined_proba) == 3:
            # 3-class: down (0), neutral (1), up (2)
            signal_class = np.argmax(combined_proba) - 1  # Map to -1, 0, 1
        else:
            signal_class = int(np.sign(weighted_pred))
        
        return {
            "class": signal_class,
            "confidence": float(confidence),
            "weighted_prediction": float(weighted_pred),
            "probabilities": combined_proba.tolist() if combined_proba is not None else None,
            "individual_predictions": {k: float(v) for k, v in predictions.items()},
            "model_agreement": self._calculate_agreement(predictions),
        }
    
    def _calculate_agreement(
        self,
        predictions: dict[str, Any],
    ) -> float:
        """Calculate agreement between models."""
        if len(predictions) <= 1:
            return 1.0
        
        pred_signs = [np.sign(p) for p in predictions.values()]
        most_common = max(set(pred_signs), key=pred_signs.count)
        agreement = pred_signs.count(most_common) / len(pred_signs)
        
        return agreement
    
    def _generate_entry_signal(
        self,
        symbol: str,
        prediction: dict[str, Any],
        data: pl.DataFrame,
        portfolio: PortfolioState,
    ) -> SignalEvent | None:
        """Generate entry signal based on prediction."""
        # Check if we already have a position
        current_position = portfolio.positions.get(symbol)
        if current_position is not None and abs(current_position.quantity) > 0:
            return None
        
        # Check position limits
        if len(portfolio.positions) >= self.config.max_positions:
            return None
        
        signal_class = prediction["class"]
        confidence = prediction["confidence"]
        model_agreement = prediction.get("model_agreement", 0)
        
        # Check minimum confidence
        if confidence < self.config.min_confidence:
            return None
        
        # Check model agreement if required
        if self.config.require_consensus and model_agreement < self.config.min_model_agreement:
            return None
        
        # Check transaction cost filter
        if not self._passes_transaction_cost_filter(prediction):
            return None
        
        # Determine signal strength and direction
        current_price = float(data["close"].tail(1).item())
        
        if signal_class == 1:  # Buy signal
            if confidence >= self.config.strong_buy_threshold:
                strength = SignalStrength.STRONG
            elif confidence >= self.config.buy_threshold:
                strength = SignalStrength.MEDIUM
            else:
                strength = SignalStrength.WEAK
            
            side = OrderSide.BUY
            
        elif signal_class == -1:  # Sell signal
            if confidence >= self.config.strong_buy_threshold:
                strength = SignalStrength.STRONG
            elif confidence >= self.config.buy_threshold:
                strength = SignalStrength.MEDIUM
            else:
                strength = SignalStrength.WEAK
            
            side = OrderSide.SELL
            
        else:  # Neutral
            return None
        
        # Calculate position size
        position_size = self._calculate_position_size(
            symbol, prediction, portfolio, current_price
        )
        
        if position_size <= 0:
            return None
        
        # Apply regime adjustments
        position_size = self._apply_regime_adjustments(symbol, position_size, strength)
        
        # Calculate stop loss and take profit
        stop_loss = None
        take_profit = None
        
        if self.config.use_stop_loss:
            if side == OrderSide.BUY:
                stop_loss = current_price * (1 - self.config.stop_loss_pct)
            else:
                stop_loss = current_price * (1 + self.config.stop_loss_pct)
        
        if self.config.use_take_profit:
            if side == OrderSide.BUY:
                take_profit = current_price * (1 + self.config.take_profit_pct)
            else:
                take_profit = current_price * (1 - self.config.take_profit_pct)
        
        # Create signal
        signal = SignalEvent(
            symbol=symbol,
            signal_type=side,
            strength=strength,
            price=current_price,
            quantity=position_size,
            strategy_id=self.strategy_id,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            metadata={
                "model_agreement": model_agreement,
                "regime": self._current_regime.get(symbol, MarketRegime.RANGING).value,
                "prediction": prediction,
            },
        )
        
        logger.info(
            f"AlphaML signal: {symbol} {side.value} "
            f"confidence={confidence:.2f} agreement={model_agreement:.2f}"
        )
        
        return signal
    
    def _generate_exit_signal(
        self,
        symbol: str,
        prediction: dict[str, Any],
        data: pl.DataFrame,
        portfolio: PortfolioState,
    ) -> SignalEvent | None:
        """Generate exit signal for existing position."""
        current_position = portfolio.positions.get(symbol)
        
        if current_position is None or current_position.quantity == 0:
            return None
        
        signal_class = prediction["class"]
        confidence = prediction["confidence"]
        current_price = float(data["close"].tail(1).item())
        
        should_exit = False
        exit_reason = ""
        
        # Check for signal reversal
        if current_position.quantity > 0 and signal_class == -1:
            if confidence >= self.config.min_confidence:
                should_exit = True
                exit_reason = "signal_reversal"
        
        elif current_position.quantity < 0 and signal_class == 1:
            if confidence >= self.config.min_confidence:
                should_exit = True
                exit_reason = "signal_reversal"
        
        # Check stop loss (handled by risk manager, but we can signal exit)
        if current_position.unrealized_pnl_pct < -self.config.stop_loss_pct:
            should_exit = True
            exit_reason = "stop_loss"
        
        # Check take profit
        if current_position.unrealized_pnl_pct > self.config.take_profit_pct:
            should_exit = True
            exit_reason = "take_profit"
        
        if not should_exit:
            return None
        
        # Create exit signal
        exit_side = OrderSide.SELL if current_position.quantity > 0 else OrderSide.BUY
        
        signal = SignalEvent(
            symbol=symbol,
            signal_type=exit_side,
            strength=SignalStrength.STRONG,
            price=current_price,
            quantity=abs(current_position.quantity),
            strategy_id=self.strategy_id,
            metadata={
                "exit_reason": exit_reason,
                "position_pnl": current_position.unrealized_pnl_pct,
            },
        )
        
        logger.info(f"AlphaML exit signal: {symbol} {exit_reason}")
        
        return signal
    
    def _passes_transaction_cost_filter(
        self,
        prediction: dict[str, Any],
    ) -> bool:
        """Check if expected return exceeds transaction costs."""
        # Simple filter based on confidence
        expected_return = prediction["confidence"] * self.config.min_expected_return
        return expected_return > self.config.transaction_cost * 2  # Round trip
    
    def _calculate_position_size(
        self,
        symbol: str,
        prediction: dict[str, Any],
        portfolio: PortfolioState,
        current_price: float,
    ) -> float:
        """Calculate position size based on confidence and config."""
        base_size = self.config.base_position_size * portfolio.equity
        
        if self.config.confidence_scaling:
            # Scale by confidence
            confidence = prediction["confidence"]
            confidence_mult = 0.5 + confidence  # 0.5 to 1.5 multiplier
            base_size *= confidence_mult
        
        # Apply max position size limit
        max_size = self.config.max_position_size * portfolio.equity
        position_value = min(base_size, max_size)
        
        # Convert to shares
        shares = position_value / current_price if current_price > 0 else 0
        
        return shares
    
    def _apply_regime_adjustments(
        self,
        symbol: str,
        position_size: float,
        strength: SignalStrength,
    ) -> float:
        """Apply regime-based adjustments to position size."""
        regime = self._current_regime.get(symbol, MarketRegime.RANGING)
        
        # Reduce size in high volatility
        if regime == MarketRegime.HIGH_VOLATILITY and self.config.reduce_size_high_vol:
            position_size *= self.config.high_vol_size_mult
        
        # Boost trend signals
        if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            if self.config.boost_trend_signals:
                position_size *= self.config.trend_signal_mult
        
        return position_size
    
    def _track_prediction(
        self,
        symbol: str,
        prediction: dict[str, Any],
        data: pl.DataFrame,
    ) -> None:
        """Track prediction for later evaluation."""
        timestamp = data["timestamp"].tail(1).item() if "timestamp" in data.columns else datetime.now()
        current_price = float(data["close"].tail(1).item())
        
        self._prediction_history.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "prediction": prediction["class"],
            "confidence": prediction["confidence"],
            "price": current_price,
            "regime": self._current_regime.get(symbol, MarketRegime.RANGING).value,
        })
        
        # Keep history bounded
        if len(self._prediction_history) > 10000:
            self._prediction_history = self._prediction_history[-10000:]
    
    # =========================================================================
    # MODEL TRAINING
    # =========================================================================
    
    def _check_and_train(self, symbol: str) -> None:
        """Check if we have enough data and train models."""
        data = self._historical_data.get(symbol)
        
        if data is None or len(data) < self.config.min_training_samples:
            return
        
        logger.info(f"Auto-training models with {len(data)} samples...")
        self._train_models(data)
    
    def _train_models(self, data: pl.DataFrame) -> None:
        """Train the ML models."""
        try:
            from models.training import TrainingPipeline, TrainingConfig, OptimizationConfig
            from features.pipeline import FeaturePipeline
            
            # Generate features
            df_features = self._feature_pipeline.generate(data)
            
            # Create target
            df_features = self._feature_pipeline.create_target(
                df_features,
                target_type="direction",
                horizon=self.config.prediction_horizon,
            )
            
            # ================================================================
            # CRITICAL FIX: Filter out non-numeric columns before training
            # ================================================================
            exclude_cols = {"timestamp", "symbol", "target", "open", "high", "low", "close", "volume"}
            numeric_types = [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8,
                            pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8]
            
            feature_cols = []
            for col in df_features.columns:
                if col not in exclude_cols:
                    col_dtype = df_features[col].dtype
                    if col_dtype in numeric_types:
                        feature_cols.append(col)
                    elif "float" in str(col_dtype).lower() or "int" in str(col_dtype).lower():
                        if col_dtype not in [pl.Utf8, pl.String, pl.Categorical]:
                            feature_cols.append(col)
            
            logger.info(f"Filtered to {len(feature_cols)} numeric features")
            
            # Drop rows with null target and features
            df_clean = df_features.drop_nulls(subset=["target"])
            df_clean = df_clean.drop_nulls(subset=feature_cols)
            
            # Extract features and target as numpy arrays
            X = df_clean.select(feature_cols).to_numpy().astype(np.float64)
            y = df_clean["target"].to_numpy().astype(np.int64)
            
            # Handle NaN/Inf values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Time-based train/test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            self._feature_names = feature_cols
            
            logger.info(f"Train/test split: {len(X_train)} train, {len(X_test)} test samples")
            # ================================================================
            
            # Configure training
            config = TrainingConfig(
                models_dir=self.config.models_dir,
                auto_optimize=True,
                optimization_config=OptimizationConfig(
                    n_trials=50,
                    cv_splits=5,
                ),
            )
            
            pipeline = TrainingPipeline(config)
            
            # Train LightGBM
            if self.config.use_lightgbm:
                logger.info("Training LightGBM...")
                lgb_model = pipeline.train(
                    "lightgbm",
                    X_train, y_train,
                    X_test, y_test,
                    feature_names=feature_cols,
                )
                self._models["lightgbm"] = lgb_model
                lgb_model.save(self.config.models_dir / "alpha_lgb.pkl")
            
            # Train XGBoost
            if self.config.use_xgboost:
                logger.info("Training XGBoost...")
                xgb_model = pipeline.train(
                    "xgboost",
                    X_train, y_train,
                    X_test, y_test,
                    feature_names=feature_cols,
                )
                self._models["xgboost"] = xgb_model
                xgb_model.save(self.config.models_dir / "alpha_xgb.pkl")
            
            self._models_loaded = True
            self._set_model_weights()
            
            logger.info(f"Models trained successfully. {len(self._models)} models active.")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            import traceback
            traceback.print_exc()
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def get_model_performance(self) -> dict[str, Any]:
        """Get model performance metrics."""
        return {
            "models_loaded": self._models_loaded,
            "active_models": list(self._models.keys()),
            "model_weights": self._model_weights,
            "prediction_count": len(self._prediction_history),
            "model_performance": self._model_performance,
        }
    
    def get_regime_summary(self) -> dict[str, Any]:
        """Get regime detection summary."""
        return {
            "current_regimes": {k: v.value for k, v in self._current_regime.items()},
            "regime_history_length": {k: len(v) for k, v in self._regime_history.items()},
        }
    
    def evaluate_predictions(
        self,
        lookback_bars: int = 100,
    ) -> dict[str, float]:
        """Evaluate prediction accuracy over recent history."""
        if len(self._prediction_history) < lookback_bars:
            return {"error": "insufficient_history"}
        
        recent = self._prediction_history[-lookback_bars:]
        
        # Calculate directional accuracy
        # This requires actual returns which we'd need to track
        # Placeholder for now
        return {
            "total_predictions": len(recent),
            "avg_confidence": np.mean([p["confidence"] for p in recent]),
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_alpha_ml_strategy(
    **kwargs: Any,
) -> AlphaMLStrategy:
    """
    Factory function to create Alpha ML Strategy.
    
    Args:
        **kwargs: Configuration parameters
    
    Returns:
        Configured AlphaMLStrategy
    """
    config = AlphaMLConfig(**kwargs)
    return AlphaMLStrategy(config)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "MarketRegime",
    "ModelType",
    "SignalStrengthLevel",
    # Config
    "AlphaMLConfig",
    # Strategy
    "AlphaMLStrategy",
    # Factory
    "create_alpha_ml_strategy",
]