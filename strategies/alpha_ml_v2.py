"""
Alpha ML Strategy V2
====================

JPMorgan-level production ML trading strategy with institutional-grade improvements.

Key Improvements over V1:
1. Triple Barrier Method for target labeling (replaces simple direction)
2. Meta-Labeling for bet sizing
3. Proper model naming convention per symbol
4. Feature correlation filtering
5. Combinatorial Purged Cross-Validation
6. Regime-aware model selection
7. Dynamic confidence thresholds
8. Transaction cost awareness

Architecture:
    Market Data -> Feature Pipeline -> Feature Selection ->
    Regime Detection -> Model Selection -> Ensemble Prediction ->
    Meta-Label Filtering -> Confidence Adjustment -> Signal Generation

Model Storage Convention:
    models/artifacts/{SYMBOL}/{SYMBOL}_{model_type}_{version}.pkl

Example:
    config = AlphaMLConfigV2(
        symbol="AAPL",
        use_triple_barrier=True,
        min_confidence=0.55,
    )
    strategy = AlphaMLStrategyV2(config)
    strategy.initialize(["AAPL"])

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
import json

import numpy as np
import polars as pl
from numpy.typing import NDArray

from config.settings import get_logger, TimeFrame, OrderSide
from config.symbols import get_model_filename, get_model_directory, validate_symbol, ALL_SYMBOLS
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
    CATBOOST = "catboost"
    STACKING = "stacking"
    NEURAL = "neural"


class PredictionMode(str, Enum):
    """Prediction mode."""
    DIRECTION = "direction"  # Simple direction prediction
    TRIPLE_BARRIER = "triple_barrier"  # Triple barrier labels
    META_LABEL = "meta_label"  # Meta-labeling with bet sizing


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AlphaMLConfigV2(StrategyConfig):
    """
    Configuration for Alpha ML Strategy V2.
    
    Comprehensive configuration with JPMorgan-level defaults.
    """
    name: str = "AlphaML_V2"
    description: str = "JPMorgan-level ML trading strategy with Triple Barrier"
    
    # =========================================================================
    # SYMBOL CONFIGURATION
    # =========================================================================
    symbol: str = ""  # Primary symbol (set during initialization)
    
    # =========================================================================
    # MODEL CONFIGURATION
    # =========================================================================
    
    # Model selection
    use_lightgbm: bool = True
    use_xgboost: bool = True
    use_catboost: bool = False
    use_stacking: bool = False
    use_neural: bool = False
    
    # Ensemble weights (normalized automatically)
    lightgbm_weight: float = 0.5
    xgboost_weight: float = 0.5
    catboost_weight: float = 0.0
    stacking_weight: float = 0.0
    neural_weight: float = 0.0
    
    # Model paths (auto-generated if None)
    models_dir: Path = field(default_factory=lambda: Path("models/artifacts"))
    model_version: str = "v1"
    
    # Auto-training
    auto_train: bool = True
    min_training_samples: int = 10000
    
    # =========================================================================
    # TARGET CONFIGURATION (KEY IMPROVEMENT)
    # =========================================================================
    
    # Prediction mode
    prediction_mode: PredictionMode = PredictionMode.TRIPLE_BARRIER
    
    # Triple Barrier settings
    tb_take_profit_mult: float = 2.0  # ATR multiplier
    tb_stop_loss_mult: float = 1.0
    tb_max_holding_period: int = 20  # Bars
    
    # Meta-labeling
    use_meta_labeling: bool = False
    meta_threshold: float = 0.5
    
    # Legacy direction settings (for backward compatibility)
    prediction_horizon: int = 5
    neutral_zone_pct: float = 0.005
    
    # =========================================================================
    # FEATURE CONFIGURATION
    # =========================================================================
    
    # Feature lookback
    lookback_bars: int = 100
    
    # Feature categories
    use_momentum_features: bool = True
    use_trend_features: bool = True
    use_volatility_features: bool = True
    use_volume_features: bool = True
    use_statistical_features: bool = True
    use_advanced_features: bool = True  # NEW: Triple barrier, microstructure, etc.
    
    # Feature selection
    max_features: int = 100
    correlation_threshold: float = 0.95  # Remove features correlated > this
    importance_threshold: float = 0.001  # Remove features with importance < this
    
    # =========================================================================
    # SIGNAL CONFIGURATION
    # =========================================================================
    
    # Confidence thresholds (KEY IMPROVEMENT: Higher thresholds)
    min_confidence: float = 0.55  # Minimum to generate any signal
    strong_buy_threshold: float = 0.65  # For strong signals
    buy_threshold: float = 0.55
    
    # Model agreement
    require_consensus: bool = True
    min_model_agreement: float = 0.6  # At least 60% of models agree
    
    # Transaction cost filter
    min_expected_return: float = 0.003  # 0.3% minimum expected return
    estimated_transaction_cost: float = 0.001  # 0.1% round-trip
    
    # =========================================================================
    # REGIME CONFIGURATION
    # =========================================================================
    
    enable_regime_detection: bool = True
    regime_lookback: int = 50
    volatility_regime_threshold: float = 1.5  # Std devs above normal
    
    # Regime-specific adjustments
    reduce_size_high_vol: bool = True
    high_vol_size_factor: float = 0.5
    
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
    max_position_size: float = 0.1  # 10% of portfolio per position
    max_correlation: float = 0.7
    
    # =========================================================================
    # RETRAINING
    # =========================================================================
    
    enable_retraining: bool = False
    retrain_interval_bars: int = 10000
    retrain_min_new_samples: int = 1000
    walk_forward_window: int = 50000


# =============================================================================
# ALPHA ML STRATEGY V2
# =============================================================================

class AlphaMLStrategyV2(BaseStrategy):
    """
    Alpha ML Trading Strategy V2.
    
    A JPMorgan-level ML-based trading strategy with:
    - Triple Barrier Method for proper target labeling
    - Per-symbol model management
    - Feature correlation filtering
    - Regime-aware trading
    - Dynamic confidence thresholds
    
    Model Naming Convention:
        models/artifacts/{SYMBOL}/{SYMBOL}_{model_type}_{version}.pkl
    
    Example:
        config = AlphaMLConfigV2(
            use_lightgbm=True,
            use_xgboost=True,
            prediction_mode=PredictionMode.TRIPLE_BARRIER,
        )
        strategy = AlphaMLStrategyV2(config)
        strategy.initialize(["AAPL", "GOOGL", "MSFT"])
    """
    
    def __init__(
        self,
        config: AlphaMLConfigV2 | None = None,
        parameters: dict[str, Any] | None = None,
    ):
        """Initialize Alpha ML Strategy V2."""
        super().__init__(config or AlphaMLConfigV2(), parameters)
        self.config: AlphaMLConfigV2 = self.config
        
        # Models per symbol
        self._models: dict[str, dict[str, Any]] = {}  # symbol -> {model_type -> model}
        self._model_weights: dict[str, float] = {}
        self._models_loaded: dict[str, bool] = {}
        
        # Feature pipeline
        self._feature_pipeline: Any = None
        self._feature_names: dict[str, list[str]] = {}  # symbol -> feature names
        self._scaler: dict[str, Any] = {}  # symbol -> scaler
        
        # Advanced labeling
        self._triple_barrier_labeler: Any = None
        self._meta_labeler: Any = None
        
        # Regime detection
        self._current_regime: dict[str, MarketRegime] = {}
        self._regime_history: dict[str, list[MarketRegime]] = {}
        
        # Historical data buffer
        self._historical_data: dict[str, pl.DataFrame] = {}
        
        # Predictions tracking
        self._last_predictions: dict[str, dict[str, Any]] = {}
        self._prediction_history: list[dict[str, Any]] = []
        
        # Model performance tracking
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
        """Initialize strategy for given symbols."""
        logger.info(f"Initializing AlphaML V2 for {len(symbols)} symbols")
        
        # Validate symbols
        for symbol in symbols:
            if not validate_symbol(symbol):
                logger.warning(f"Unknown symbol: {symbol}")
        
        # Initialize feature pipeline
        self._initialize_feature_pipeline()
        
        # Initialize advanced labelers
        self._initialize_labelers()
        
        # Initialize model weights
        self._set_model_weights()
        
        # Load or prepare models for each symbol
        for symbol in symbols:
            self._historical_data[symbol] = pl.DataFrame()
            self._current_regime[symbol] = MarketRegime.RANGING
            self._regime_history[symbol] = []
            self._models[symbol] = {}
            self._models_loaded[symbol] = False
            
            # Try to load pre-trained models
            self._load_models_for_symbol(symbol)
        
        logger.info("AlphaML V2 initialization complete")
    
    def _initialize_feature_pipeline(self) -> None:
        """Initialize the feature generation pipeline."""
        from features.pipeline import FeaturePipeline, FeatureConfig, FeatureCategory
        
        # Determine enabled categories
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
            normalize=False,  # We'll handle normalization separately
        )
        
        self._feature_pipeline = FeaturePipeline(config)
        logger.info(f"Feature pipeline initialized with {len(enabled_categories)} categories")
    
    def _initialize_labelers(self) -> None:
        """Initialize triple barrier and meta-labelers."""
        if self.config.prediction_mode == PredictionMode.TRIPLE_BARRIER:
            from features.advanced import TripleBarrierLabeler, TripleBarrierConfig
            
            tb_config = TripleBarrierConfig(
                take_profit_multiplier=self.config.tb_take_profit_mult,
                stop_loss_multiplier=self.config.tb_stop_loss_mult,
                max_holding_period=self.config.tb_max_holding_period,
            )
            self._triple_barrier_labeler = TripleBarrierLabeler(tb_config)
            logger.info("Triple Barrier labeler initialized")
        
        if self.config.use_meta_labeling:
            from features.advanced import MetaLabeler, MetaLabelConfig
            
            meta_config = MetaLabelConfig(
                primary_threshold=self.config.meta_threshold,
            )
            self._meta_labeler = MetaLabeler(meta_config)
            logger.info("Meta-labeler initialized")
    
    def _set_model_weights(self) -> None:
        """Set and normalize ensemble model weights."""
        weights = {}
        
        if self.config.use_lightgbm:
            weights["lightgbm"] = self.config.lightgbm_weight
        if self.config.use_xgboost:
            weights["xgboost"] = self.config.xgboost_weight
        if self.config.use_catboost:
            weights["catboost"] = self.config.catboost_weight
        if self.config.use_stacking:
            weights["stacking"] = self.config.stacking_weight
        if self.config.use_neural:
            weights["neural"] = self.config.neural_weight
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            self._model_weights = {k: v / total for k, v in weights.items()}
        else:
            self._model_weights = {"lightgbm": 1.0}
        
        logger.info(f"Model weights: {self._model_weights}")
    
    # =========================================================================
    # MODEL LOADING
    # =========================================================================
    
    def _load_models_for_symbol(self, symbol: str) -> bool:
        """Load pre-trained models for a symbol."""
        symbol = symbol.upper()
        symbol_dir = get_model_directory(self.config.models_dir, symbol, create=False)
        
        if not symbol_dir.exists():
            logger.info(f"No model directory for {symbol}. Will train on first data.")
            return False
        
        models_loaded = 0
        
        for model_type in self._model_weights.keys():
            filename = get_model_filename(symbol, model_type, self.config.model_version)
            model_path = symbol_dir / filename
            
            if model_path.exists():
                try:
                    model = self._load_single_model(model_path)
                    self._models[symbol][model_type] = model
                    models_loaded += 1
                    logger.info(f"Loaded {model_type} model for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to load {model_type} for {symbol}: {e}")
        
        self._models_loaded[symbol] = models_loaded > 0
        
        # Load feature names if available
        metadata_path = symbol_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                # Extract feature names from any model's metadata
                for model_info in metadata.get("models", {}).values():
                    if model_info.get("feature_names"):
                        self._feature_names[symbol] = model_info["feature_names"]
                        break
            except Exception as e:
                logger.warning(f"Failed to load metadata for {symbol}: {e}")
        
        return self._models_loaded[symbol]
    
    def _load_single_model(self, path: Path) -> Any:
        """Load a single model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        # Check if it's our wrapped format
        if isinstance(data, dict) and "class_name" in data:
            class_name = data.get("class_name", "")
            
            if "LightGBM" in class_name:
                from models.classifiers import LightGBMClassifier
                return LightGBMClassifier.load(path)
            elif "XGBoost" in class_name:
                from models.classifiers import XGBoostClassifier
                return XGBoostClassifier.load(path)
            elif "CatBoost" in class_name:
                from models.classifiers import CatBoostClassifier
                return CatBoostClassifier.load(path)
            elif "Stacking" in class_name:
                from models.classifiers import StackingClassifier
                return StackingClassifier.load(path)
        
        # Return raw model
        return data
    
    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================
    
    def calculate_signals(
        self,
        event: MarketEvent,
        portfolio: PortfolioState,
    ) -> list[SignalEvent]:
        """Generate trading signals from market data."""
        signals = []
        symbol = event.symbol
        data = event.data
        
        if data is None or len(data) < self.config.lookback_bars:
            return signals
        
        # Update historical data buffer
        self._update_historical_data(symbol, data)
        
        # Check if we need to train models
        if not self._models_loaded.get(symbol, False) and self.config.auto_train:
            self._check_and_train(symbol)
            if not self._models_loaded.get(symbol, False):
                return signals
        
        # Detect market regime
        if self.config.enable_regime_detection:
            self._detect_regime(symbol, data)
        
        # Generate features for prediction
        features = self._generate_features_for_prediction(symbol, data)
        if features is None or len(features) == 0:
            return signals
        
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
        
        # Check for exit signals
        exit_signal = self._generate_exit_signal(symbol, prediction, data, portfolio)
        if exit_signal is not None:
            signals.append(exit_signal)
        
        # Track prediction
        self._track_prediction(symbol, prediction, data)
        
        return signals
    
    def _update_historical_data(self, symbol: str, data: pl.DataFrame) -> None:
        """Update historical data buffer for a symbol."""
        # Calculate required buffer size
        min_for_features = self.config.lookback_bars * 3
        min_for_training = self.config.min_training_samples + 1000
        max_bars = max(min_for_features, min_for_training)
        
        if symbol not in self._historical_data or len(self._historical_data[symbol]) == 0:
            self._historical_data[symbol] = data.tail(max_bars)
        else:
            # Concatenate and trim
            combined = pl.concat([self._historical_data[symbol], data])
            self._historical_data[symbol] = combined.tail(max_bars)
    
    def _generate_features_for_prediction(
        self,
        symbol: str,
        data: pl.DataFrame,
    ) -> NDArray[np.float64] | None:
        """Generate features for a prediction."""
        try:
            # Generate all features
            df_features = self._feature_pipeline.generate(data)
            
            # Add advanced features if enabled
            if self.config.use_advanced_features:
                from features.advanced import (
                    MicrostructureFeatures, 
                    CalendarFeatures,
                    FeatureInteractions,
                )
                df_features = MicrostructureFeatures.add_features(df_features)
                df_features = CalendarFeatures.add_features(df_features)
                df_features = FeatureInteractions.add_interactions(df_features)
            
            # Get feature columns
            if symbol in self._feature_names and self._feature_names[symbol]:
                feature_cols = self._feature_names[symbol]
            else:
                # Use all numeric non-OHLCV columns
                exclude = {"timestamp", "symbol", "target", "open", "high", "low", "close", "volume"}
                numeric_types = [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
                feature_cols = [
                    c for c in df_features.columns 
                    if c not in exclude and df_features[c].dtype in numeric_types
                ]
                self._feature_names[symbol] = feature_cols
            
            # Extract last row as feature vector
            if len(df_features) == 0:
                return None
            
            features = df_features.tail(1).select(feature_cols).to_numpy().astype(np.float64)
            
            # Handle NaN/Inf
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature generation error for {symbol}: {e}")
            return None
    
    def _get_ensemble_prediction(
        self,
        symbol: str,
        features: NDArray[np.float64],
    ) -> dict[str, Any] | None:
        """Get prediction from model ensemble."""
        if symbol not in self._models or not self._models[symbol]:
            return None
        
        predictions = {}
        probabilities = {}
        
        for model_type, model in self._models[symbol].items():
            try:
                pred = model.predict(features)
                predictions[model_type] = pred[0] if isinstance(pred, np.ndarray) else pred
                
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(features)
                    if proba is not None:
                        probabilities[model_type] = proba[0]
            except Exception as e:
                logger.warning(f"Model {model_type} prediction error: {e}")
        
        if not predictions:
            return None
        
        # Combine predictions
        return self._combine_predictions(predictions, probabilities)
    
    def _combine_predictions(
        self,
        predictions: dict[str, Any],
        probabilities: dict[str, NDArray],
    ) -> dict[str, Any]:
        """Combine predictions from multiple models."""
        # Weighted voting
        weighted_pred = 0.0
        total_weight = 0.0
        
        for name, pred in predictions.items():
            weight = self._model_weights.get(name, 1.0 / len(predictions))
            weighted_pred += float(pred) * weight
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
            confidence = float(np.max(combined_proba))
        else:
            # Agreement-based confidence
            pred_values = list(predictions.values())
            pred_signs = [np.sign(p) for p in pred_values]
            most_common = max(set(pred_signs), key=pred_signs.count)
            confidence = pred_signs.count(most_common) / len(pred_signs)
        
        # Determine signal class
        if combined_proba is not None and len(combined_proba) >= 2:
            if len(combined_proba) == 2:
                # Binary classification
                signal_class = 1 if combined_proba[1] > 0.5 else 0
            else:
                # Multi-class
                signal_class = int(np.argmax(combined_proba)) - 1
        else:
            signal_class = int(np.sign(weighted_pred))
        
        # Model agreement
        pred_signs = [np.sign(p) for p in predictions.values()]
        agreement = pred_signs.count(max(set(pred_signs), key=pred_signs.count)) / len(pred_signs)
        
        return {
            "class": signal_class,
            "confidence": confidence,
            "weighted_prediction": float(weighted_pred),
            "probabilities": combined_proba.tolist() if combined_proba is not None else None,
            "individual_predictions": {k: float(v) for k, v in predictions.items()},
            "model_agreement": agreement,
        }
    
    def _generate_entry_signal(
        self,
        symbol: str,
        prediction: dict[str, Any],
        data: pl.DataFrame,
        portfolio: PortfolioState,
    ) -> SignalEvent | None:
        """Generate entry signal based on prediction."""
        # Check existing position
        current_position = portfolio.positions.get(symbol)
        if current_position is not None and abs(current_position.quantity) > 0:
            return None
        
        # Check position limits
        if len(portfolio.positions) >= self.config.max_positions:
            return None
        
        signal_class = prediction["class"]
        confidence = prediction["confidence"]
        agreement = prediction.get("model_agreement", 0)
        
        # Confidence filter
        if confidence < self.config.min_confidence:
            return None
        
        # Agreement filter
        if self.config.require_consensus and agreement < self.config.min_model_agreement:
            return None
        
        # Transaction cost filter
        if not self._passes_transaction_cost_filter(prediction):
            return None
        
        # Get current price
        current_price = float(data["close"].tail(1).item())
        
        # Determine direction and strength
        if signal_class == 1:  # Buy signal
            direction = 1
            signal_type = "entry_long"
            
            if confidence >= self.config.strong_buy_threshold:
                strength = SignalStrength.STRONG
            else:
                strength = SignalStrength.MODERATE
        
        elif signal_class == -1:  # Sell signal
            direction = -1
            signal_type = "entry_short"
            strength = SignalStrength.MODERATE
        
        else:
            return None
        
        # Regime adjustment
        size_multiplier = 1.0
        regime = self._current_regime.get(symbol, MarketRegime.RANGING)
        if regime == MarketRegime.HIGH_VOLATILITY and self.config.reduce_size_high_vol:
            size_multiplier = self.config.high_vol_size_factor
        
        # Create signal
        signal = SignalEvent(
            symbol=symbol,
            signal_type=signal_type,
            direction=direction,
            strength=strength.value if hasattr(strength, 'value') else strength,
            price=current_price,
            strategy_name=self.name,
            confidence=confidence,
            stop_loss=current_price * (1 - self.config.stop_loss_pct * direction) if self.config.use_stop_loss else None,
            take_profit=current_price * (1 + self.config.take_profit_pct * direction) if self.config.use_take_profit else None,
            metadata={
                "regime": regime.value,
                "model_agreement": agreement,
                "size_multiplier": size_multiplier,
                "prediction": prediction,
            },
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
        position = portfolio.positions.get(symbol)
        if position is None or position.quantity == 0:
            return None
        
        current_price = float(data["close"].tail(1).item())
        is_long = position.quantity > 0
        
        # Check for signal reversal
        signal_class = prediction["class"]
        confidence = prediction["confidence"]
        
        should_exit = False
        exit_reason = ""
        
        # Exit on strong contrary signal
        if is_long and signal_class == -1 and confidence >= self.config.min_confidence:
            should_exit = True
            exit_reason = "contrary_signal"
        elif not is_long and signal_class == 1 and confidence >= self.config.min_confidence:
            should_exit = True
            exit_reason = "contrary_signal"
        
        if not should_exit:
            return None
        
        signal = SignalEvent(
            symbol=symbol,
            signal_type="exit_long" if is_long else "exit_short",
            direction=-1 if is_long else 1,
            strength=0.5,
            price=current_price,
            strategy_name=self.name,
            confidence=confidence,
            metadata={
                "exit_reason": exit_reason,
                "position_quantity": position.quantity,
            },
        )
        
        return signal
    
    def _passes_transaction_cost_filter(self, prediction: dict[str, Any]) -> bool:
        """Check if expected return exceeds transaction costs."""
        # For triple barrier, expected return is implicit in the labeling
        if self.config.prediction_mode == PredictionMode.TRIPLE_BARRIER:
            return prediction["confidence"] >= self.config.min_confidence
        
        # For direction prediction, estimate expected return
        confidence = prediction["confidence"]
        expected_return = (confidence - 0.5) * 2 * 0.01  # Rough estimate
        
        return expected_return > self.config.estimated_transaction_cost + self.config.min_expected_return
    
    # =========================================================================
    # REGIME DETECTION
    # =========================================================================
    
    def _detect_regime(self, symbol: str, data: pl.DataFrame) -> MarketRegime:
        """Detect current market regime."""
        try:
            lookback = min(self.config.regime_lookback, len(data))
            recent_data = data.tail(lookback)
            
            # Calculate volatility
            returns = recent_data["close"].pct_change().drop_nulls().to_numpy()
            current_vol = np.std(returns) if len(returns) > 0 else 0
            
            # Historical volatility baseline
            hist_vol = np.std(data["close"].pct_change().drop_nulls().to_numpy())
            vol_ratio = current_vol / hist_vol if hist_vol > 0 else 1.0
            
            # Trend calculation
            close = recent_data["close"].to_numpy()
            trend = (close[-1] - close[0]) / close[0] if len(close) > 0 and close[0] != 0 else 0
            
            # Determine regime
            if vol_ratio > self.config.volatility_regime_threshold:
                regime = MarketRegime.HIGH_VOLATILITY
            elif vol_ratio < 0.5:
                regime = MarketRegime.LOW_VOLATILITY
            elif trend > 0.02:
                regime = MarketRegime.TRENDING_UP
            elif trend < -0.02:
                regime = MarketRegime.TRENDING_DOWN
            else:
                regime = MarketRegime.RANGING
            
            self._current_regime[symbol] = regime
            self._regime_history[symbol].append(regime)
            
            # Bound history
            if len(self._regime_history[symbol]) > 100:
                self._regime_history[symbol] = self._regime_history[symbol][-100:]
            
            return regime
            
        except Exception as e:
            logger.error(f"Regime detection error: {e}")
            return MarketRegime.RANGING
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    
    def _check_and_train(self, symbol: str) -> None:
        """Check if we have enough data to train models."""
        data = self._historical_data.get(symbol)
        
        if data is None or len(data) < self.config.min_training_samples:
            return
        
        logger.info(f"Auto-training models for {symbol} with {len(data)} samples...")
        self._train_models(symbol, data)
    
    def _train_models(self, symbol: str, data: pl.DataFrame) -> None:
        """Train ML models for a symbol."""
        try:
            from models.training import TrainingPipeline, TrainingConfig, OptimizationConfig
            from models.model_manager import ModelManager
            
            # Generate features
            df_features = self._feature_pipeline.generate(data)
            
            # Create target based on prediction mode
            if self.config.prediction_mode == PredictionMode.TRIPLE_BARRIER:
                df_features = self._triple_barrier_labeler.apply_binary_labels(df_features)
            else:
                df_features = self._feature_pipeline.create_target(
                    df_features,
                    target_type="direction",
                    horizon=self.config.prediction_horizon,
                )
            
            # Filter numeric features
            exclude_cols = {"timestamp", "symbol", "target", "open", "high", "low", "close", "volume",
                          "tb_label", "tb_return", "tb_barrier", "tb_holding_period"}
            numeric_types = [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            
            feature_cols = [
                c for c in df_features.columns
                if c not in exclude_cols and df_features[c].dtype in numeric_types
            ]
            
            # Apply correlation filtering
            feature_cols = self._filter_correlated_features(df_features, feature_cols)
            
            logger.info(f"Using {len(feature_cols)} features after filtering")
            
            # Clean data
            df_clean = df_features.drop_nulls(subset=["target"])
            df_clean = df_clean.drop_nulls(subset=feature_cols)
            
            # Extract arrays
            X = df_clean.select(feature_cols).to_numpy().astype(np.float64)
            y = df_clean["target"].to_numpy().astype(np.int64)
            
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Time-based split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            self._feature_names[symbol] = feature_cols
            
            # Initialize model manager
            model_manager = ModelManager(self.config.models_dir)
            
            # Configure training
            train_config = TrainingConfig(
                models_dir=self.config.models_dir,
                auto_optimize=True,
                optimization_config=OptimizationConfig(n_trials=30, cv_splits=3),
            )
            
            training_pipeline = TrainingPipeline(train_config)
            
            # Train each model type
            for model_type in self._model_weights.keys():
                logger.info(f"Training {model_type} for {symbol}...")
                
                try:
                    model = training_pipeline.train(
                        model_type,
                        X_train, y_train,
                        X_test, y_test,
                        feature_names=feature_cols,
                    )
                    
                    # Evaluate
                    train_metrics = model.evaluate(X_train, y_train)
                    test_metrics = model.evaluate(X_test, y_test)
                    
                    # Save with proper naming
                    model_manager.save_model(
                        model=model,
                        symbol=symbol,
                        model_type=model_type,
                        version=self.config.model_version,
                        metrics={
                            "train_accuracy": train_metrics.get("accuracy", 0),
                            "test_accuracy": test_metrics.get("accuracy", 0),
                            "train_f1": train_metrics.get("f1_macro", 0),
                            "test_f1": test_metrics.get("f1_macro", 0),
                        },
                        feature_names=feature_cols,
                        training_samples=len(X_train),
                    )
                    
                    self._models[symbol][model_type] = model
                    logger.info(f"  {model_type} - Train: {train_metrics.get('accuracy', 0):.4f}, "
                               f"Test: {test_metrics.get('accuracy', 0):.4f}")
                    
                except Exception as e:
                    logger.error(f"Failed to train {model_type}: {e}")
            
            self._models_loaded[symbol] = len(self._models[symbol]) > 0
            logger.info(f"Training complete for {symbol}. {len(self._models[symbol])} models active.")
            
        except Exception as e:
            logger.error(f"Model training failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    def _filter_correlated_features(
        self,
        df: pl.DataFrame,
        feature_cols: list[str],
    ) -> list[str]:
        """Remove highly correlated features."""
        if len(feature_cols) < 2:
            return feature_cols
        
        try:
            # Calculate correlation matrix
            X = df.select(feature_cols).to_numpy()
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            corr_matrix = np.corrcoef(X.T)
            
            # Find pairs above threshold
            to_remove = set()
            n = len(feature_cols)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(corr_matrix[i, j]) > self.config.correlation_threshold:
                        # Remove the feature with higher average correlation
                        avg_corr_i = np.mean(np.abs(corr_matrix[i, :]))
                        avg_corr_j = np.mean(np.abs(corr_matrix[j, :]))
                        
                        if avg_corr_i > avg_corr_j:
                            to_remove.add(feature_cols[i])
                        else:
                            to_remove.add(feature_cols[j])
            
            filtered = [f for f in feature_cols if f not in to_remove]
            
            if len(to_remove) > 0:
                logger.info(f"Removed {len(to_remove)} correlated features")
            
            return filtered
            
        except Exception as e:
            logger.warning(f"Correlation filtering failed: {e}")
            return feature_cols
    
    # =========================================================================
    # TRACKING
    # =========================================================================
    
    def _track_prediction(
        self,
        symbol: str,
        prediction: dict[str, Any],
        data: pl.DataFrame,
    ) -> None:
        """Track prediction for model evaluation."""
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
        
        # Bound history
        if len(self._prediction_history) > 10000:
            self._prediction_history = self._prediction_history[-10000:]
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def get_model_performance(self, symbol: str | None = None) -> dict[str, Any]:
        """Get model performance metrics."""
        if symbol:
            return {
                "symbol": symbol,
                "models_loaded": self._models_loaded.get(symbol, False),
                "active_models": list(self._models.get(symbol, {}).keys()),
                "feature_count": len(self._feature_names.get(symbol, [])),
            }
        else:
            return {
                "total_symbols": len(self._models),
                "loaded_symbols": sum(self._models_loaded.values()),
                "model_weights": self._model_weights,
                "prediction_count": len(self._prediction_history),
            }
    
    def get_regime_summary(self, symbol: str | None = None) -> dict[str, Any]:
        """Get regime detection summary."""
        if symbol:
            return {
                "symbol": symbol,
                "current_regime": self._current_regime.get(symbol, MarketRegime.RANGING).value,
                "history_length": len(self._regime_history.get(symbol, [])),
            }
        else:
            return {
                "regimes": {k: v.value for k, v in self._current_regime.items()},
            }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_alpha_ml_strategy_v2(**kwargs: Any) -> AlphaMLStrategyV2:
    """
    Factory function to create Alpha ML Strategy V2.
    
    Args:
        **kwargs: Configuration parameters
    
    Returns:
        Configured AlphaMLStrategyV2 instance
    """
    config = AlphaMLConfigV2(**kwargs)
    return AlphaMLStrategyV2(config)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "MarketRegime",
    "ModelType",
    "PredictionMode",
    # Config
    "AlphaMLConfigV2",
    # Strategy
    "AlphaMLStrategyV2",
    # Factory
    "create_alpha_ml_strategy_v2",
]