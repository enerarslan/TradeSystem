"""
ML-ENHANCED MOMENTUM STRATEJİSİ
JPMorgan Quantitative Research Division Tarzı

Bu strateji geleneksel momentum sinyallerini ML tahminleriyle birleştirir.
Alpha üretimi için çok katmanlı sinyal füzyonu kullanır.

Özellikler:
- XGBoost/LightGBM tahminleri
- Ensemble sinyal füzyonu
- Adaptive threshold adjustment
- Confidence-weighted position sizing
- Feature importance tracking
- Online learning desteği
- Regime-aware signal generation
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings('ignore')

from strategies.base import BaseStrategy
from strategies.momentum import AdvancedMomentum, TechnicalIndicators, MarketRegime
from data.models import MarketTick, TradeSignal, Side
from utils.logger import log

# Optional ML imports
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    log.warning("XGBoost not installed. ML features disabled.")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class MLModelType(Enum):
    """Desteklenen ML model tipleri"""
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    ENSEMBLE = "ensemble"


class SignalType(Enum):
    """Sinyal tipleri"""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class MLPrediction:
    """ML tahmin sonucu"""
    signal: SignalType
    probability: float
    confidence: float
    features_used: int
    model_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    feature_importance: Dict[str, float] = field(default_factory=dict)


@dataclass
class FusedSignal:
    """Füzyon sonucu birleşik sinyal"""
    side: Side
    strength: float  # -1 to 1
    confidence: float  # 0 to 1
    ml_contribution: float  # ML'in sinyale katkısı
    technical_contribution: float  # Teknik analizin katkısı
    regime_adjustment: float  # Rejime göre düzeltme
    components: Dict[str, float] = field(default_factory=dict)


class FeatureGenerator:
    """
    ML için feature üretici.
    
    Raw market data'dan zengin feature seti üretir.
    """
    
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.feature_names: List[str] = []
    
    def generate(self, prices: np.ndarray, volumes: np.ndarray) -> Optional[np.ndarray]:
        """
        Feature vektörü üretir.
        
        Returns:
            Feature array (1D) or None if insufficient data
        """
        if len(prices) < self.lookback:
            return None
        
        features = []
        self.feature_names = []
        
        # === RETURN FEATURES ===
        returns_1 = np.diff(prices[-2:]) / prices[-2] if prices[-2] != 0 else 0
        returns_5 = (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 and prices[-6] != 0 else 0
        returns_10 = (prices[-1] - prices[-11]) / prices[-11] if len(prices) >= 11 and prices[-11] != 0 else 0
        returns_20 = (prices[-1] - prices[-21]) / prices[-21] if len(prices) >= 21 and prices[-21] != 0 else 0
        
        features.extend([returns_1, returns_5, returns_10, returns_20])
        self.feature_names.extend(['return_1', 'return_5', 'return_10', 'return_20'])
        
        # === MOVING AVERAGES ===
        sma_5 = np.mean(prices[-5:])
        sma_10 = np.mean(prices[-10:])
        sma_20 = np.mean(prices[-20:])
        sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else sma_20
        
        # Price relative to MAs
        price_vs_sma5 = (prices[-1] / sma_5 - 1) if sma_5 != 0 else 0
        price_vs_sma10 = (prices[-1] / sma_10 - 1) if sma_10 != 0 else 0
        price_vs_sma20 = (prices[-1] / sma_20 - 1) if sma_20 != 0 else 0
        price_vs_sma50 = (prices[-1] / sma_50 - 1) if sma_50 != 0 else 0
        
        features.extend([price_vs_sma5, price_vs_sma10, price_vs_sma20, price_vs_sma50])
        self.feature_names.extend(['price_vs_sma5', 'price_vs_sma10', 'price_vs_sma20', 'price_vs_sma50'])
        
        # MA crossovers
        sma_5_10_cross = (sma_5 / sma_10 - 1) if sma_10 != 0 else 0
        sma_10_20_cross = (sma_10 / sma_20 - 1) if sma_20 != 0 else 0
        
        features.extend([sma_5_10_cross, sma_10_20_cross])
        self.feature_names.extend(['sma_5_10_cross', 'sma_10_20_cross'])
        
        # === RSI ===
        rsi = self._calculate_rsi(prices[-15:])
        rsi_normalized = (rsi - 50) / 50  # -1 to 1
        features.append(rsi_normalized)
        self.feature_names.append('rsi_normalized')
        
        # === VOLATILITY ===
        volatility_5 = np.std(np.diff(prices[-6:])) / prices[-1] if prices[-1] != 0 else 0
        volatility_20 = np.std(np.diff(prices[-21:])) / prices[-1] if prices[-1] != 0 else 0
        vol_ratio = volatility_5 / volatility_20 if volatility_20 != 0 else 1
        
        features.extend([volatility_5, volatility_20, vol_ratio])
        self.feature_names.extend(['volatility_5', 'volatility_20', 'vol_ratio'])
        
        # === MOMENTUM ===
        momentum_5 = prices[-1] - prices[-6] if len(prices) >= 6 else 0
        momentum_10 = prices[-1] - prices[-11] if len(prices) >= 11 else 0
        acceleration = momentum_5 - (prices[-6] - prices[-11]) if len(prices) >= 11 else 0
        
        features.extend([momentum_5, momentum_10, acceleration])
        self.feature_names.extend(['momentum_5', 'momentum_10', 'acceleration'])
        
        # === VOLUME FEATURES ===
        if len(volumes) >= 20:
            vol_sma_20 = np.mean(volumes[-20:])
            vol_ratio_current = volumes[-1] / vol_sma_20 if vol_sma_20 != 0 else 1
            vol_trend = (np.mean(volumes[-5:]) / np.mean(volumes[-20:-5])) - 1 if np.mean(volumes[-20:-5]) != 0 else 0
            
            features.extend([vol_ratio_current, vol_trend])
            self.feature_names.extend(['vol_ratio_current', 'vol_trend'])
        else:
            features.extend([1.0, 0.0])
            self.feature_names.extend(['vol_ratio_current', 'vol_trend'])
        
        # === PRICE PATTERNS ===
        # Higher highs / Lower lows
        highs_5 = prices[-5:]
        lows_5 = prices[-5:]
        higher_high = 1 if highs_5[-1] > np.max(highs_5[:-1]) else 0
        lower_low = 1 if lows_5[-1] < np.min(lows_5[:-1]) else 0
        
        features.extend([higher_high, lower_low])
        self.feature_names.extend(['higher_high', 'lower_low'])
        
        # === BOLLINGER BAND FEATURES ===
        bb_middle = sma_20
        bb_std = np.std(prices[-20:])
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle != 0 else 0
        bb_position = (prices[-1] - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) != 0 else 0.5
        
        features.extend([bb_width, bb_position])
        self.feature_names.extend(['bb_width', 'bb_position'])
        
        # === MACD FEATURES ===
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        macd = ema_12 - ema_26
        macd_normalized = macd / prices[-1] if prices[-1] != 0 else 0
        
        features.append(macd_normalized)
        self.feature_names.append('macd_normalized')
        
        return np.array(features)
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """RSI hesapla"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_ema(self, data: np.ndarray, period: int) -> float:
        """EMA hesapla"""
        if len(data) < period:
            return data[-1] if len(data) > 0 else 0
        
        multiplier = 2 / (period + 1)
        ema = data[-period]
        for price in data[-period+1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema


class MLSignalGenerator:
    """
    ML tabanlı sinyal üretici.
    
    Pre-trained model veya online learning ile çalışır.
    """
    
    def __init__(
        self,
        model_type: MLModelType = MLModelType.XGBOOST,
        model_path: Optional[str] = None,
        online_learning: bool = False
    ):
        self.model_type = model_type
        self.model_path = model_path
        self.online_learning = online_learning
        
        self.model = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.is_fitted = False
        
        self.feature_generator = FeatureGenerator()
        
        # Training buffer for online learning
        self.training_buffer_X: List[np.ndarray] = []
        self.training_buffer_y: List[int] = []
        self.buffer_size = 1000
        self.min_samples_to_train = 200
        
        # Performance tracking
        self.predictions_made = 0
        self.correct_predictions = 0
        
        # Load pre-trained model if provided
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
    
    def _initialize_model(self):
        """Model'i başlat"""
        if self.model_type == MLModelType.XGBOOST and HAS_XGBOOST:
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        elif self.model_type == MLModelType.LIGHTGBM and HAS_LIGHTGBM:
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
        elif self.model_type == MLModelType.RANDOM_FOREST and HAS_SKLEARN:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == MLModelType.GRADIENT_BOOSTING and HAS_SKLEARN:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            log.warning(f"Model type {self.model_type} not available. Using fallback.")
            self.model = None
    
    def predict(
        self, 
        prices: np.ndarray, 
        volumes: np.ndarray
    ) -> Optional[MLPrediction]:
        """
        ML tahmini üret.
        
        Returns:
            MLPrediction or None if prediction not possible
        """
        # Generate features
        features = self.feature_generator.generate(prices, volumes)
        
        if features is None:
            return None
        
        # Model yoksa veya fitted değilse, default prediction döndür
        if self.model is None or not self.is_fitted:
            return MLPrediction(
                signal=SignalType.HOLD,
                probability=0.5,
                confidence=0.0,
                features_used=len(features),
                model_type="none"
            )
        
        try:
            # Scale features
            features_scaled = features.reshape(1, -1)
            if self.scaler is not None and hasattr(self.scaler, 'mean_'):
                features_scaled = self.scaler.transform(features_scaled)
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Convert to SignalType
            signal = SignalType(int(prediction))
            
            # Calculate confidence from probability distribution
            max_prob = np.max(probabilities)
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            max_entropy = np.log(len(probabilities))
            confidence = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
            
            # Feature importance
            feature_importance = {}
            if hasattr(self.model, 'feature_importances_'):
                for name, imp in zip(self.feature_generator.feature_names, self.model.feature_importances_):
                    feature_importance[name] = float(imp)
            
            self.predictions_made += 1
            
            return MLPrediction(
                signal=signal,
                probability=float(max_prob),
                confidence=float(confidence),
                features_used=len(features),
                model_type=self.model_type.value,
                feature_importance=feature_importance
            )
            
        except Exception as e:
            log.error(f"ML prediction error: {e}")
            return None
    
    def add_training_sample(self, features: np.ndarray, label: int):
        """Online learning için training sample ekle"""
        if not self.online_learning:
            return
        
        self.training_buffer_X.append(features)
        self.training_buffer_y.append(label)
        
        # Buffer doluysa eski verileri çıkar
        if len(self.training_buffer_X) > self.buffer_size:
            self.training_buffer_X.pop(0)
            self.training_buffer_y.pop(0)
        
        # Yeterli veri varsa retrain
        if len(self.training_buffer_X) >= self.min_samples_to_train:
            if len(self.training_buffer_X) % 100 == 0:  # Her 100 sample'da bir
                self._retrain()
    
    def _retrain(self):
        """Model'i yeniden eğit"""
        if self.model is None:
            self._initialize_model()
        
        if self.model is None:
            return
        
        try:
            X = np.array(self.training_buffer_X)
            y = np.array(self.training_buffer_y)
            
            # Scale
            if self.scaler is not None:
                X = self.scaler.fit_transform(X)
            
            # Train
            self.model.fit(X, y)
            self.is_fitted = True
            
            log.debug(f"ML model retrained with {len(X)} samples")
            
        except Exception as e:
            log.error(f"ML retrain error: {e}")
    
    def _load_model(self, path: str):
        """Model'i yükle"""
        try:
            with open(path, 'rb') as f:
                saved = pickle.load(f)
            
            self.model = saved['model']
            self.scaler = saved.get('scaler', self.scaler)
            self.is_fitted = True
            
            log.info(f"ML model loaded from {path}")
            
        except Exception as e:
            log.error(f"Failed to load ML model: {e}")
    
    def save_model(self, path: str):
        """Model'i kaydet"""
        if self.model is None:
            return
        
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'model_type': self.model_type.value
                }, f)
            
            log.info(f"ML model saved to {path}")
            
        except Exception as e:
            log.error(f"Failed to save ML model: {e}")


class MLMomentumStrategy(BaseStrategy):
    """
    ML-Enhanced Momentum Stratejisi.
    
    Geleneksel momentum sinyalleriyle ML tahminlerini birleştirir.
    
    Sinyal Füzyon Yöntemi:
    1. Technical Analysis (AdvancedMomentum)
    2. ML Prediction (XGBoost/LightGBM)
    3. Regime Filter (Market condition adjustment)
    4. Confidence Weighting
    
    Position Sizing:
    - ATR-based base sizing
    - Confidence multiplier
    - Regime adjustment
    - Max position limits
    """
    
    def __init__(
        self,
        symbol: str,
        # Technical parameters
        fast_period: int = 10,
        slow_period: int = 30,
        rsi_period: int = 14,
        # ML parameters
        ml_model_type: str = "xgboost",
        ml_model_path: Optional[str] = None,
        ml_weight: float = 0.4,  # ML sinyalinin ağırlığı
        online_learning: bool = False,
        # Signal fusion
        min_confidence: float = 0.55,
        min_agreement: float = 0.6,  # Technical ve ML arası min uyum
        # Lookback
        lookback: int = 200
    ):
        super().__init__(name="MLMomentum_V1")
        
        self.symbol = symbol
        self.ml_weight = ml_weight
        self.min_confidence = min_confidence
        self.min_agreement = min_agreement
        self.lookback = lookback
        
        # Data buffers
        self.prices = deque(maxlen=lookback)
        self.volumes = deque(maxlen=lookback)
        self.timestamps = deque(maxlen=lookback)
        
        # Technical strategy component
        self.technical_strategy = AdvancedMomentum(
            symbol=symbol,
            fast_period=fast_period,
            slow_period=slow_period,
            rsi_period=rsi_period,
            min_confidence=0.5,  # Daha düşük threshold (fusion'da kullanılacak)
            lookback=lookback
        )
        
        # ML component
        model_type = MLModelType(ml_model_type) if ml_model_type in [e.value for e in MLModelType] else MLModelType.XGBOOST
        self.ml_generator = MLSignalGenerator(
            model_type=model_type,
            model_path=ml_model_path,
            online_learning=online_learning
        )
        
        # Feature generator
        self.feature_generator = FeatureGenerator(lookback=lookback)
        
        # Position tracking
        self.position_open = False
        self.entry_price = None
        self.entry_time = None
        self.position_side = None
        
        # Performance tracking
        self.signals_generated = 0
        self.ml_predictions_used = 0
        self.fusion_history: List[FusedSignal] = []
        
        # Last signals for debugging
        self.last_technical_signal: Optional[Side] = None
        self.last_ml_prediction: Optional[MLPrediction] = None
        
        log.info(f"🤖 ML Momentum Strategy başlatıldı: {symbol}")
        log.info(f"   ML Weight: {ml_weight}, Min Confidence: {min_confidence}")
    
    async def on_tick(self, tick: MarketTick) -> Optional[TradeSignal]:
        """
        Her tick'te çağrılır.
        Technical analysis + ML fusion yapar.
        """
        if tick.symbol != self.symbol:
            return None
        
        # Store data
        self.prices.append(tick.price)
        self.volumes.append(tick.volume)
        self.timestamps.append(tick.timestamp)
        
        # Technical strategy'nin kendi buffer'larını da güncelle
        self.technical_strategy.prices.append(tick.price)
        self.technical_strategy.volumes.append(tick.volume)
        self.technical_strategy.high_prices.append(tick.price)
        self.technical_strategy.low_prices.append(tick.price)
        
        # Yeterli veri yoksa bekle
        if len(self.prices) < self.lookback // 2:
            return None
        
        # 1. Technical Analysis Signal
        tech_indicators = self.technical_strategy._calculate_indicators()
        if tech_indicators is None:
            return None
        
        regime = self.technical_strategy._detect_market_regime(tech_indicators)
        tech_side, tech_confidence = self.technical_strategy._generate_signal(
            tech_indicators, regime, tick
        )
        self.last_technical_signal = tech_side
        
        # 2. ML Prediction
        prices_arr = np.array(self.prices)
        volumes_arr = np.array(self.volumes)
        ml_pred = self.ml_generator.predict(prices_arr, volumes_arr)
        self.last_ml_prediction = ml_pred
        
        # 3. Signal Fusion
        fused_signal = self._fuse_signals(
            tech_side, tech_confidence,
            ml_pred,
            regime,
            tick
        )
        
        if fused_signal is None:
            return None
        
        # 4. Store fusion history
        self.fusion_history.append(fused_signal)
        if len(self.fusion_history) > 1000:
            self.fusion_history.pop(0)
        
        # 5. Generate trade signal if actionable
        if fused_signal.side == Side.HOLD:
            return None
        
        if fused_signal.confidence < self.min_confidence:
            return None
        
        # 6. Position management
        if self._should_skip_signal(fused_signal):
            return None
        
        # 7. Calculate position size
        quantity = self._calculate_position_size(fused_signal, tech_indicators)
        
        # 8. Create TradeSignal
        signal = TradeSignal(
            symbol=self.symbol,
            side=fused_signal.side,
            price=tick.price,
            quantity=quantity,
            strategy_name=self.name,
            timestamp=datetime.now()
        )
        
        # 9. Update tracking
        self.signals_generated += 1
        if ml_pred and ml_pred.confidence > 0:
            self.ml_predictions_used += 1
        
        # 10. Update position state
        self._update_position_state(signal, tick)
        
        # 11. Online learning feedback
        if self.ml_generator.online_learning and ml_pred:
            self._provide_learning_feedback(tick)
        
        return signal
    
    def _fuse_signals(
        self,
        tech_side: Side,
        tech_confidence: float,
        ml_pred: Optional[MLPrediction],
        regime: MarketRegime,
        tick: MarketTick
    ) -> Optional[FusedSignal]:
        """
        Technical ve ML sinyallerini birleştirir.
        """
        # Initialize components
        components = {
            'technical': 0.0,
            'ml': 0.0,
            'regime': 0.0
        }
        
        # 1. Technical contribution
        tech_weight = 1 - self.ml_weight
        if tech_side == Side.BUY:
            components['technical'] = tech_confidence * tech_weight
        elif tech_side == Side.SELL:
            components['technical'] = -tech_confidence * tech_weight
        
        # 2. ML contribution
        if ml_pred and ml_pred.confidence > 0.3:
            ml_signal_value = ml_pred.signal.value / 2  # Normalize to -1 to 1
            components['ml'] = ml_signal_value * ml_pred.confidence * self.ml_weight
        
        # 3. Regime adjustment
        regime_mult = 1.0
        if regime.trending and regime.strength > 50:
            if regime.trend_direction == "BULLISH":
                regime_mult = 1.2
            elif regime.trend_direction == "BEARISH":
                regime_mult = 0.8
        elif not regime.trending:
            regime_mult = 0.7  # Sideways'de daha az agresif
        
        components['regime'] = regime_mult
        
        # 4. Calculate combined signal
        raw_signal = components['technical'] + components['ml']
        adjusted_signal = raw_signal * regime_mult
        
        # 5. Determine side and confidence
        if adjusted_signal > 0.1:
            side = Side.BUY
            strength = min(1.0, adjusted_signal)
        elif adjusted_signal < -0.1:
            side = Side.SELL
            strength = min(1.0, abs(adjusted_signal))
        else:
            side = Side.HOLD
            strength = 0.0
        
        # 6. Calculate final confidence
        agreement = self._calculate_agreement(tech_side, ml_pred)
        confidence = strength * (0.5 + 0.5 * agreement)  # Agreement bonus
        
        return FusedSignal(
            side=side,
            strength=strength,
            confidence=confidence,
            ml_contribution=abs(components['ml']),
            technical_contribution=abs(components['technical']),
            regime_adjustment=regime_mult,
            components=components
        )
    
    def _calculate_agreement(
        self, 
        tech_side: Side, 
        ml_pred: Optional[MLPrediction]
    ) -> float:
        """Technical ve ML sinyalleri arası uyum hesapla"""
        if ml_pred is None or ml_pred.confidence < 0.3:
            return 0.5  # Neutral
        
        tech_direction = 0
        if tech_side == Side.BUY:
            tech_direction = 1
        elif tech_side == Side.SELL:
            tech_direction = -1
        
        ml_direction = 0
        if ml_pred.signal in [SignalType.BUY, SignalType.STRONG_BUY]:
            ml_direction = 1
        elif ml_pred.signal in [SignalType.SELL, SignalType.STRONG_SELL]:
            ml_direction = -1
        
        if tech_direction == ml_direction and tech_direction != 0:
            return 1.0  # Full agreement
        elif tech_direction == -ml_direction and tech_direction != 0:
            return 0.0  # Disagreement
        else:
            return 0.5  # Partial/neutral
    
    def _should_skip_signal(self, fused: FusedSignal) -> bool:
        """Sinyal atlanmalı mı?"""
        # Zaten aynı yönde pozisyon varsa
        if self.position_open:
            if fused.side == Side.BUY and self.position_side == "LONG":
                return True  # Zaten long'dayız
            if fused.side == Side.SELL and not self.position_side:
                return True  # Kapatılacak pozisyon yok
        
        return False
    
    def _calculate_position_size(
        self, 
        fused: FusedSignal, 
        indicators: TechnicalIndicators
    ) -> float:
        """Pozisyon büyüklüğü hesapla"""
        base_size = 10.0
        
        # Confidence multiplier
        confidence_mult = 0.5 + fused.confidence
        
        # ML contribution bonus
        ml_bonus = 1.0 + (fused.ml_contribution * 0.5)
        
        # Volatility adjustment
        vol_mult = 1.0
        if indicators.atr / self.prices[-1] > 0.02:
            vol_mult = 0.7
        
        size = base_size * confidence_mult * ml_bonus * vol_mult
        
        return max(1.0, min(100.0, size))
    
    def _update_position_state(self, signal: TradeSignal, tick: MarketTick):
        """Pozisyon durumunu güncelle"""
        if signal.side == Side.BUY and not self.position_open:
            self.position_open = True
            self.entry_price = tick.price
            self.entry_time = datetime.now()
            self.position_side = "LONG"
            log.info(f"🟢 ML LONG: {self.symbol} @ ${tick.price:.2f} "
                    f"(Tech: {self.last_technical_signal}, ML: {self.last_ml_prediction.signal if self.last_ml_prediction else 'N/A'})")
        
        elif signal.side == Side.SELL and self.position_open:
            pnl_pct = ((tick.price - self.entry_price) / self.entry_price * 100) if self.entry_price else 0
            log.info(f"🔴 ML CLOSE: {self.symbol} @ ${tick.price:.2f} (PnL: {pnl_pct:+.2f}%)")
            self.position_open = False
            self.entry_price = None
            self.entry_time = None
            self.position_side = None
    
    def _provide_learning_feedback(self, tick: MarketTick):
        """Online learning için feedback sağla"""
        if len(self.prices) < 10:
            return
        
        # 5 bar sonraki sonucu label olarak kullan (delayed feedback)
        # Bu basitleştirilmiş bir yaklaşım
        features = self.feature_generator.generate(
            np.array(self.prices)[:-5],
            np.array(self.volumes)[:-5]
        )
        
        if features is None:
            return
        
        # Label: Gelecekteki fiyat hareketi
        future_return = (self.prices[-1] - self.prices[-6]) / self.prices[-6]
        
        if future_return > 0.005:
            label = 1  # BUY was correct
        elif future_return < -0.005:
            label = -1  # SELL was correct
        else:
            label = 0  # HOLD was correct
        
        self.ml_generator.add_training_sample(features, label)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Performans istatistikleri"""
        ml_usage_ratio = (
            self.ml_predictions_used / self.signals_generated 
            if self.signals_generated > 0 else 0
        )
        
        return {
            'total_signals': self.signals_generated,
            'ml_predictions_used': self.ml_predictions_used,
            'ml_usage_ratio': ml_usage_ratio,
            'position_open': self.position_open,
            'entry_price': self.entry_price,
            'ml_model_fitted': self.ml_generator.is_fitted,
            'fusion_history_size': len(self.fusion_history)
        }
    
    def get_last_fusion_details(self) -> Optional[Dict[str, Any]]:
        """Son füzyon detayları"""
        if not self.fusion_history:
            return None
        
        last = self.fusion_history[-1]
        return {
            'side': last.side.value if hasattr(last.side, 'value') else str(last.side),
            'strength': last.strength,
            'confidence': last.confidence,
            'ml_contribution': last.ml_contribution,
            'technical_contribution': last.technical_contribution,
            'regime_adjustment': last.regime_adjustment,
            'components': last.components
        }


# Export
__all__ = [
    'MLMomentumStrategy',
    'MLSignalGenerator',
    'FeatureGenerator',
    'MLPrediction',
    'FusedSignal',
    'MLModelType',
    'SignalType'
]