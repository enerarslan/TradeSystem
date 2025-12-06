"""
KURUMSAL MOMENTUM STRATEJİ SİSTEMİ
JPMorgan Quantitative Research Division Tarzı

Bu modül iki ana strateji içerir:
1. AdvancedMomentum - Geleneksel teknik analiz tabanlı
2. MLMomentumStrategy - ML-enhanced hibrit strateji

Özellikler:
- Multi-timeframe analiz
- Regime detection (Trending/Sideways/Volatile)
- Adaptive thresholds
- Risk-adjusted position sizing
- Real-time performance tracking
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
from data.models import MarketTick, TradeSignal, Side
from utils.logger import log

# Optional ML imports
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

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


# ============================================================================
# DATA CLASSES & ENUMS
# ============================================================================

@dataclass
class TechnicalIndicators:
    """Teknik gösterge değerleri"""
    # Moving Averages
    sma_fast: float = 0.0
    sma_slow: float = 0.0
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    
    # Momentum
    rsi: float = 50.0
    rsi_sma: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    
    # Volatility
    atr: float = 0.0
    atr_percent: float = 0.0
    bollinger_upper: float = 0.0
    bollinger_lower: float = 0.0
    bollinger_middle: float = 0.0
    bollinger_width: float = 0.0
    
    # Volume
    volume_sma: float = 0.0
    volume_ratio: float = 1.0
    obv: float = 0.0
    
    # Trend
    adx: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0
    
    # Price Action
    current_price: float = 0.0
    price_change: float = 0.0
    price_change_pct: float = 0.0
    high_20: float = 0.0
    low_20: float = 0.0
    
    # Stochastic
    stoch_k: float = 50.0
    stoch_d: float = 50.0
    
    # Williams %R
    williams_r: float = -50.0


@dataclass
class MarketRegime:
    """Piyasa rejimi analizi"""
    trending: bool = False
    trend_direction: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    strength: float = 0.0  # 0-100
    volatility_regime: str = "NORMAL"  # LOW, NORMAL, HIGH, EXTREME
    volume_regime: str = "NORMAL"  # LOW, NORMAL, HIGH
    regime_confidence: float = 0.0


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
    ml_contribution: float
    technical_contribution: float
    regime_adjustment: float
    components: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# ADVANCED MOMENTUM STRATEGY
# ============================================================================

class AdvancedMomentum(BaseStrategy):
    """
    Gelişmiş Momentum Stratejisi - JPMorgan Quant Tarzı
    
    Multi-factor momentum stratejisi:
    - Trend Following (MA crossovers)
    - Mean Reversion (RSI extremes)
    - Volatility Breakout
    - Volume Confirmation
    - Regime Adaptation
    
    Sinyal Üretim Süreci:
    1. Teknik göstergeleri hesapla
    2. Market regime tespit et
    3. Multi-factor scoring
    4. Confidence calculation
    5. Position sizing
    """
    
    def __init__(
        self,
        symbol: str,
        fast_period: int = 10,
        slow_period: int = 30,
        rsi_period: int = 14,
        atr_period: int = 14,
        adx_period: int = 14,
        volume_period: int = 20,
        min_confidence: float = 0.6,
        lookback: int = 200,
        use_regime_filter: bool = True,
        use_volume_confirmation: bool = True
    ):
        """
        Args:
            symbol: Trading sembolü
            fast_period: Hızlı MA periyodu
            slow_period: Yavaş MA periyodu
            rsi_period: RSI periyodu
            atr_period: ATR periyodu
            adx_period: ADX periyodu
            volume_period: Volume MA periyodu
            min_confidence: Minimum sinyal güveni
            lookback: Veri geçmişi boyutu
            use_regime_filter: Rejim filtresi kullan
            use_volume_confirmation: Hacim onayı kullan
        """
        super().__init__(name="AdvancedMomentum_V2")
        
        self.symbol = symbol
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.adx_period = adx_period
        self.volume_period = volume_period
        self.min_confidence = min_confidence
        self.lookback = lookback
        self.use_regime_filter = use_regime_filter
        self.use_volume_confirmation = use_volume_confirmation
        
        # Data buffers
        self.prices = deque(maxlen=lookback)
        self.high_prices = deque(maxlen=lookback)
        self.low_prices = deque(maxlen=lookback)
        self.volumes = deque(maxlen=lookback)
        self.timestamps = deque(maxlen=lookback)
        
        # Indicator caches
        self._ema_fast_cache = None
        self._ema_slow_cache = None
        self._rsi_cache = None
        
        # Position tracking
        self.position_open = False
        self.entry_price = None
        self.entry_time = None
        self.position_side = None
        
        # Performance tracking
        self.signals_generated = 0
        self.winning_signals = 0
        self.losing_signals = 0
        
        # Last known state
        self.last_indicators: Optional[TechnicalIndicators] = None
        self.last_regime: Optional[MarketRegime] = None
        
        log.info(f"🎯 Advanced Momentum başlatıldı: {symbol}")
        log.info(f"   Fast/Slow: {fast_period}/{slow_period}, RSI: {rsi_period}")
        log.info(f"   Min Confidence: {min_confidence}")
    
    async def on_tick(self, tick: MarketTick) -> Optional[TradeSignal]:
        """
        Her tick'te çağrılır, sinyal üretir.
        """
        if tick.symbol != self.symbol:
            return None
        
        # Store data
        self.prices.append(tick.price)
        self.high_prices.append(tick.price)  # Tick-level için aynı
        self.low_prices.append(tick.price)
        self.volumes.append(tick.volume)
        self.timestamps.append(tick.timestamp)
        
        # Yeterli veri yoksa bekle
        if len(self.prices) < self.slow_period + 10:
            return None
        
        # 1. Teknik göstergeleri hesapla
        indicators = self._calculate_indicators()
        if indicators is None:
            return None
        
        self.last_indicators = indicators
        
        # 2. Market regime tespit et
        regime = self._detect_market_regime(indicators)
        self.last_regime = regime
        
        # 3. Sinyal üret
        signal_side, confidence = self._generate_signal(indicators, regime, tick)
        
        # 4. Minimum güven kontrolü
        if signal_side == Side.HOLD or confidence < self.min_confidence:
            return None
        
        # 5. Regime filter
        if self.use_regime_filter:
            if not self._passes_regime_filter(signal_side, regime):
                return None
        
        # 6. Volume confirmation
        if self.use_volume_confirmation:
            if not self._check_volume_confirmation(indicators):
                confidence *= 0.8  # Düşük hacimde güveni azalt
        
        # 7. Position management
        if self._should_skip_signal(signal_side):
            return None
        
        # 8. Calculate position size
        quantity = self._calculate_position_size(indicators, confidence)
        
        # 9. Create TradeSignal
        signal = TradeSignal(
            symbol=self.symbol,
            side=signal_side,
            price=tick.price,
            quantity=quantity,
            strategy_name=self.name,
            timestamp=datetime.now()
        )
        
        # 10. Update position state
        self._update_position_state(signal, tick)
        
        self.signals_generated += 1
        
        return signal
    
    def _calculate_indicators(self) -> Optional[TechnicalIndicators]:
        """Tüm teknik göstergeleri hesaplar"""
        if len(self.prices) < self.slow_period:
            return None
        
        prices = np.array(self.prices)
        highs = np.array(self.high_prices)
        lows = np.array(self.low_prices)
        volumes = np.array(self.volumes)
        
        indicators = TechnicalIndicators()
        
        # Current price info
        indicators.current_price = prices[-1]
        indicators.price_change = prices[-1] - prices[-2] if len(prices) > 1 else 0
        indicators.price_change_pct = (indicators.price_change / prices[-2] * 100) if prices[-2] != 0 else 0
        
        # Moving Averages
        indicators.sma_fast = np.mean(prices[-self.fast_period:])
        indicators.sma_slow = np.mean(prices[-self.slow_period:])
        indicators.ema_fast = self._calculate_ema(prices, self.fast_period)
        indicators.ema_slow = self._calculate_ema(prices, self.slow_period)
        
        # RSI
        indicators.rsi = self._calculate_rsi(prices, self.rsi_period)
        indicators.rsi_sma = self._calculate_rsi_sma(prices, self.rsi_period, 5)
        
        # MACD
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        indicators.macd = ema_12 - ema_26
        indicators.macd_signal = self._calculate_ema(
            np.array([indicators.macd]), 9
        ) if len(prices) > 35 else indicators.macd
        indicators.macd_histogram = indicators.macd - indicators.macd_signal
        
        # ATR
        indicators.atr = self._calculate_atr(highs, lows, prices, self.atr_period)
        indicators.atr_percent = (indicators.atr / prices[-1] * 100) if prices[-1] != 0 else 0
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2.0
        indicators.bollinger_middle = np.mean(prices[-bb_period:])
        bb_std_val = np.std(prices[-bb_period:])
        indicators.bollinger_upper = indicators.bollinger_middle + bb_std * bb_std_val
        indicators.bollinger_lower = indicators.bollinger_middle - bb_std * bb_std_val
        indicators.bollinger_width = (
            (indicators.bollinger_upper - indicators.bollinger_lower) / indicators.bollinger_middle * 100
        ) if indicators.bollinger_middle != 0 else 0
        
        # Volume
        indicators.volume_sma = np.mean(volumes[-self.volume_period:])
        indicators.volume_ratio = volumes[-1] / indicators.volume_sma if indicators.volume_sma != 0 else 1
        indicators.obv = self._calculate_obv(prices, volumes)
        
        # ADX
        adx_result = self._calculate_adx(highs, lows, prices, self.adx_period)
        indicators.adx = adx_result[0]
        indicators.plus_di = adx_result[1]
        indicators.minus_di = adx_result[2]
        
        # Price Range
        indicators.high_20 = np.max(highs[-20:])
        indicators.low_20 = np.min(lows[-20:])
        
        # Stochastic
        stoch = self._calculate_stochastic(highs, lows, prices, 14, 3)
        indicators.stoch_k = stoch[0]
        indicators.stoch_d = stoch[1]
        
        # Williams %R
        indicators.williams_r = self._calculate_williams_r(highs, lows, prices, 14)
        
        return indicators
    
    def _detect_market_regime(self, indicators: TechnicalIndicators) -> MarketRegime:
        """Piyasa rejimini tespit eder"""
        regime = MarketRegime()
        
        # Trend detection using ADX
        if indicators.adx > 25:
            regime.trending = True
            if indicators.plus_di > indicators.minus_di:
                regime.trend_direction = "BULLISH"
            else:
                regime.trend_direction = "BEARISH"
            regime.strength = min(100, indicators.adx * 2)
        else:
            regime.trending = False
            regime.trend_direction = "NEUTRAL"
            regime.strength = indicators.adx * 2
        
        # Volatility regime
        if indicators.atr_percent < 1.0:
            regime.volatility_regime = "LOW"
        elif indicators.atr_percent < 2.5:
            regime.volatility_regime = "NORMAL"
        elif indicators.atr_percent < 4.0:
            regime.volatility_regime = "HIGH"
        else:
            regime.volatility_regime = "EXTREME"
        
        # Volume regime
        if indicators.volume_ratio < 0.5:
            regime.volume_regime = "LOW"
        elif indicators.volume_ratio < 1.5:
            regime.volume_regime = "NORMAL"
        else:
            regime.volume_regime = "HIGH"
        
        # Confidence calculation
        confidence_factors = []
        
        if regime.trending:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)
        
        if regime.volatility_regime in ["NORMAL", "HIGH"]:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        if regime.volume_regime != "LOW":
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        regime.regime_confidence = np.mean(confidence_factors)
        
        return regime
    
    def _generate_signal(
        self, 
        indicators: TechnicalIndicators, 
        regime: MarketRegime,
        tick: MarketTick
    ) -> Tuple[Side, float]:
        """
        Multi-factor sinyal üretimi.
        
        Faktörler:
        1. Trend (MA crossover)
        2. Momentum (RSI, MACD)
        3. Mean Reversion (Oversold/Overbought)
        4. Breakout (Price vs Bollinger)
        5. Volume (Confirmation)
        """
        scores = {
            'trend': 0.0,
            'momentum': 0.0,
            'mean_reversion': 0.0,
            'breakout': 0.0,
            'volume': 0.0
        }
        
        weights = {
            'trend': 0.30,
            'momentum': 0.25,
            'mean_reversion': 0.20,
            'breakout': 0.15,
            'volume': 0.10
        }
        
        # 1. Trend Score
        if indicators.ema_fast > indicators.ema_slow:
            trend_strength = (indicators.ema_fast - indicators.ema_slow) / indicators.ema_slow * 100
            scores['trend'] = min(1.0, trend_strength / 2)
        else:
            trend_strength = (indicators.ema_slow - indicators.ema_fast) / indicators.ema_slow * 100
            scores['trend'] = -min(1.0, trend_strength / 2)
        
        # 2. Momentum Score
        # RSI component
        rsi_score = 0.0
        if indicators.rsi > 50:
            rsi_score = (indicators.rsi - 50) / 50  # 0 to 1
        else:
            rsi_score = (indicators.rsi - 50) / 50  # -1 to 0
        
        # MACD component
        macd_score = 0.0
        if indicators.macd > 0 and indicators.macd_histogram > 0:
            macd_score = 0.5
        elif indicators.macd < 0 and indicators.macd_histogram < 0:
            macd_score = -0.5
        
        scores['momentum'] = (rsi_score * 0.6 + macd_score * 0.4)
        
        # 3. Mean Reversion Score
        if indicators.rsi < 30:
            scores['mean_reversion'] = (30 - indicators.rsi) / 30  # Oversold = Buy
        elif indicators.rsi > 70:
            scores['mean_reversion'] = -(indicators.rsi - 70) / 30  # Overbought = Sell
        else:
            scores['mean_reversion'] = 0.0
        
        # 4. Breakout Score
        price = indicators.current_price
        bb_range = indicators.bollinger_upper - indicators.bollinger_lower
        
        if bb_range > 0:
            bb_position = (price - indicators.bollinger_lower) / bb_range
            
            if bb_position > 0.9:  # Near upper band
                scores['breakout'] = 0.5 if regime.trending else -0.3
            elif bb_position < 0.1:  # Near lower band
                scores['breakout'] = -0.5 if regime.trending else 0.3
        
        # 5. Volume Score
        if indicators.volume_ratio > 1.5:
            # High volume confirms trend
            if scores['trend'] > 0:
                scores['volume'] = 0.5
            elif scores['trend'] < 0:
                scores['volume'] = -0.5
        
        # Calculate weighted score
        final_score = sum(scores[k] * weights[k] for k in scores.keys())
        
        # Regime adjustment
        if regime.trending:
            if regime.trend_direction == "BULLISH" and final_score > 0:
                final_score *= 1.2
            elif regime.trend_direction == "BEARISH" and final_score < 0:
                final_score *= 1.2
        else:
            # Sideways market - reduce signal strength
            final_score *= 0.7
        
        # Determine side and confidence
        if final_score > 0.15:
            side = Side.BUY
            confidence = min(1.0, abs(final_score) * regime.regime_confidence)
        elif final_score < -0.15:
            side = Side.SELL
            confidence = min(1.0, abs(final_score) * regime.regime_confidence)
        else:
            side = Side.HOLD
            confidence = 0.0
        
        return side, confidence
    
    def _passes_regime_filter(self, side: Side, regime: MarketRegime) -> bool:
        """Rejim filtresi"""
        # Extreme volatility'de işlem yapma
        if regime.volatility_regime == "EXTREME":
            log.debug("Rejim filtresi: EXTREME volatilite - sinyal reddedildi")
            return False
        
        # Trending market'te trend yönünde işlem
        if regime.trending and regime.strength > 50:
            if side == Side.BUY and regime.trend_direction == "BEARISH":
                log.debug("Rejim filtresi: BEARISH trend'de BUY reddedildi")
                return False
            if side == Side.SELL and regime.trend_direction == "BULLISH":
                log.debug("Rejim filtresi: BULLISH trend'de SELL reddedildi")
                return False
        
        return True
    
    def _check_volume_confirmation(self, indicators: TechnicalIndicators) -> bool:
        """Hacim onayı kontrolü"""
        return indicators.volume_ratio >= 0.7
    
    def _should_skip_signal(self, side: Side) -> bool:
        """Sinyal atlanmalı mı?"""
        # Aynı yönde pozisyon varsa
        if self.position_open:
            if side == Side.BUY and self.position_side == "LONG":
                return True
            if side == Side.SELL and not self.position_open:
                return True
        return False
    
    def _calculate_position_size(
        self, 
        indicators: TechnicalIndicators, 
        confidence: float
    ) -> float:
        """ATR-based position sizing"""
        base_size = 10.0
        
        # Confidence multiplier
        confidence_mult = 0.5 + confidence
        
        # Volatility adjustment (düşük volatilitede büyük pozisyon)
        vol_mult = 1.0
        if indicators.atr_percent > 3.0:
            vol_mult = 0.5
        elif indicators.atr_percent > 2.0:
            vol_mult = 0.75
        elif indicators.atr_percent < 1.0:
            vol_mult = 1.25
        
        size = base_size * confidence_mult * vol_mult
        
        return max(1.0, min(100.0, size))
    
    def _update_position_state(self, signal: TradeSignal, tick: MarketTick):
        """Pozisyon durumunu güncelle"""
        if signal.side == Side.BUY and not self.position_open:
            self.position_open = True
            self.entry_price = tick.price
            self.entry_time = datetime.now()
            self.position_side = "LONG"
            log.info(f"🟢 LONG: {self.symbol} @ ${tick.price:.2f}")
        
        elif signal.side == Side.SELL and self.position_open:
            if self.entry_price:
                pnl_pct = (tick.price - self.entry_price) / self.entry_price * 100
                if pnl_pct > 0:
                    self.winning_signals += 1
                else:
                    self.losing_signals += 1
                log.info(f"🔴 CLOSE: {self.symbol} @ ${tick.price:.2f} (PnL: {pnl_pct:+.2f}%)")
            
            self.position_open = False
            self.entry_price = None
            self.entry_time = None
            self.position_side = None
    
    # =========================================================================
    # HELPER METHODS - Technical Calculations
    # =========================================================================
    
    def _calculate_ema(self, data: np.ndarray, period: int) -> float:
        """Exponential Moving Average"""
        if len(data) < period:
            return float(np.mean(data))
        
        multiplier = 2 / (period + 1)
        ema = float(np.mean(data[:period]))
        
        for price in data[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-(period+1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))
    
    def _calculate_rsi_sma(self, prices: np.ndarray, rsi_period: int, sma_period: int) -> float:
        """RSI'ın hareketli ortalaması"""
        if len(prices) < rsi_period + sma_period:
            return self._calculate_rsi(prices, rsi_period)
        
        rsi_values = []
        for i in range(sma_period):
            idx = len(prices) - sma_period + i
            rsi_val = self._calculate_rsi(prices[:idx+1], rsi_period)
            rsi_values.append(rsi_val)
        
        return float(np.mean(rsi_values))
    
    def _calculate_atr(
        self, 
        highs: np.ndarray, 
        lows: np.ndarray, 
        closes: np.ndarray, 
        period: int = 14
    ) -> float:
        """Average True Range"""
        if len(closes) < period + 1:
            return float(np.mean(highs[-period:] - lows[-period:]))
        
        tr_values = []
        for i in range(1, min(period + 1, len(closes))):
            high = highs[-period + i - 1]
            low = lows[-period + i - 1]
            prev_close = closes[-period + i - 2]
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_values.append(tr)
        
        return float(np.mean(tr_values)) if tr_values else 0.0
    
    def _calculate_obv(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """On-Balance Volume"""
        if len(prices) < 2:
            return 0.0
        
        obv = 0.0
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                obv += volumes[i]
            elif prices[i] < prices[i-1]:
                obv -= volumes[i]
        
        return float(obv)
    
    def _calculate_adx(
        self, 
        highs: np.ndarray, 
        lows: np.ndarray, 
        closes: np.ndarray, 
        period: int = 14
    ) -> Tuple[float, float, float]:
        """Average Directional Index"""
        if len(closes) < period + 1:
            return (20.0, 20.0, 20.0)
        
        plus_dm_list = []
        minus_dm_list = []
        tr_list = []
        
        for i in range(1, len(closes)):
            high = highs[i]
            low = lows[i]
            prev_high = highs[i-1]
            prev_low = lows[i-1]
            prev_close = closes[i-1]
            
            # True Range
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)
            
            # Directional Movement
            plus_dm = max(0, high - prev_high)
            minus_dm = max(0, prev_low - low)
            
            if plus_dm > minus_dm:
                minus_dm = 0
            elif minus_dm > plus_dm:
                plus_dm = 0
            else:
                plus_dm = minus_dm = 0
            
            plus_dm_list.append(plus_dm)
            minus_dm_list.append(minus_dm)
        
        # Smoothed values
        atr = np.mean(tr_list[-period:])
        plus_dm_smooth = np.mean(plus_dm_list[-period:])
        minus_dm_smooth = np.mean(minus_dm_list[-period:])
        
        # DI calculations
        plus_di = (plus_dm_smooth / atr * 100) if atr != 0 else 0
        minus_di = (minus_dm_smooth / atr * 100) if atr != 0 else 0
        
        # DX and ADX
        di_sum = plus_di + minus_di
        dx = (abs(plus_di - minus_di) / di_sum * 100) if di_sum != 0 else 0
        
        return (float(dx), float(plus_di), float(minus_di))
    
    def _calculate_stochastic(
        self, 
        highs: np.ndarray, 
        lows: np.ndarray, 
        closes: np.ndarray,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[float, float]:
        """Stochastic Oscillator"""
        if len(closes) < k_period:
            return (50.0, 50.0)
        
        highest_high = np.max(highs[-k_period:])
        lowest_low = np.min(lows[-k_period:])
        
        if highest_high == lowest_low:
            stoch_k = 50.0
        else:
            stoch_k = ((closes[-1] - lowest_low) / (highest_high - lowest_low)) * 100
        
        # %D is SMA of %K
        stoch_d = stoch_k  # Simplified
        
        return (float(stoch_k), float(stoch_d))
    
    def _calculate_williams_r(
        self, 
        highs: np.ndarray, 
        lows: np.ndarray, 
        closes: np.ndarray,
        period: int = 14
    ) -> float:
        """Williams %R"""
        if len(closes) < period:
            return -50.0
        
        highest_high = np.max(highs[-period:])
        lowest_low = np.min(lows[-period:])
        
        if highest_high == lowest_low:
            return -50.0
        
        williams_r = ((highest_high - closes[-1]) / (highest_high - lowest_low)) * -100
        return float(williams_r)
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Strateji istatistikleri"""
        total_closed = self.winning_signals + self.losing_signals
        win_rate = (self.winning_signals / total_closed * 100) if total_closed > 0 else 0
        
        return {
            'symbol': self.symbol,
            'strategy': self.name,
            'signals_generated': self.signals_generated,
            'winning_signals': self.winning_signals,
            'losing_signals': self.losing_signals,
            'win_rate': win_rate,
            'position_open': self.position_open,
            'entry_price': self.entry_price,
            'last_regime': self.last_regime.trend_direction if self.last_regime else None,
            'last_rsi': self.last_indicators.rsi if self.last_indicators else None
        }


# ============================================================================
# ML COMPONENTS
# ============================================================================

class FeatureGenerator:
    """ML için feature üretici"""
    
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.feature_names: List[str] = []
    
    def generate(self, prices: np.ndarray, volumes: np.ndarray) -> Optional[np.ndarray]:
        """Feature vektörü üretir"""
        if len(prices) < self.lookback:
            return None
        
        features = []
        self.feature_names = []
        
        # === RETURN FEATURES ===
        returns_1 = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] != 0 else 0
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
        rsi_normalized = (rsi - 50) / 50
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
        highs_5 = prices[-5:]
        higher_high = 1 if prices[-1] > np.max(highs_5[:-1]) else 0
        lower_low = 1 if prices[-1] < np.min(highs_5[:-1]) else 0
        
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
    """ML tabanlı sinyal üretici"""
    
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
        
        # Training buffer
        self.training_buffer_X: List[np.ndarray] = []
        self.training_buffer_y: List[int] = []
        self.buffer_size = 1000
        self.min_samples_to_train = 200
        
        # Performance tracking
        self.predictions_made = 0
        
        # Load pre-trained model
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
            self.model = None
    
    def predict(
        self, 
        prices: np.ndarray, 
        volumes: np.ndarray
    ) -> Optional[MLPrediction]:
        """ML tahmini üret"""
        features = self.feature_generator.generate(prices, volumes)
        
        if features is None:
            return None
        
        if self.model is None or not self.is_fitted:
            return MLPrediction(
                signal=SignalType.HOLD,
                probability=0.5,
                confidence=0.0,
                features_used=len(features),
                model_type="none"
            )
        
        try:
            features_scaled = features.reshape(1, -1)
            if self.scaler is not None and hasattr(self.scaler, 'mean_'):
                features_scaled = self.scaler.transform(features_scaled)
            
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            signal = SignalType(int(prediction))
            max_prob = np.max(probabilities)
            
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            max_entropy = np.log(len(probabilities))
            confidence = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
            
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
        """Online learning sample ekle"""
        if not self.online_learning:
            return
        
        self.training_buffer_X.append(features)
        self.training_buffer_y.append(label)
        
        if len(self.training_buffer_X) > self.buffer_size:
            self.training_buffer_X.pop(0)
            self.training_buffer_y.pop(0)
        
        if len(self.training_buffer_X) >= self.min_samples_to_train:
            if len(self.training_buffer_X) % 100 == 0:
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
            
            if self.scaler is not None:
                X = self.scaler.fit_transform(X)
            
            self.model.fit(X, y)
            self.is_fitted = True
            
        except Exception as e:
            log.error(f"ML retrain error: {e}")
    
    def _load_model(self, path: str):
        """Model yükle"""
        try:
            with open(path, 'rb') as f:
                saved = pickle.load(f)
            
            self.model = saved['model']
            self.scaler = saved.get('scaler', self.scaler)
            self.is_fitted = True
            
        except Exception as e:
            log.error(f"Failed to load ML model: {e}")
    
    def save_model(self, path: str):
        """Model kaydet"""
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
            
        except Exception as e:
            log.error(f"Failed to save ML model: {e}")


# ============================================================================
# ML MOMENTUM STRATEGY
# ============================================================================

class MLMomentumStrategy(BaseStrategy):
    """
    ML-Enhanced Momentum Stratejisi.
    
    Technical analysis + ML tahminlerini birleştirir.
    """
    
    def __init__(
        self,
        symbol: str,
        fast_period: int = 10,
        slow_period: int = 30,
        rsi_period: int = 14,
        ml_model_type: str = "xgboost",
        ml_model_path: Optional[str] = None,
        ml_weight: float = 0.4,
        online_learning: bool = False,
        min_confidence: float = 0.55,
        min_agreement: float = 0.6,
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
            min_confidence=0.5,
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
        
        self.last_technical_signal: Optional[Side] = None
        self.last_ml_prediction: Optional[MLPrediction] = None
        
        log.info(f"🤖 ML Momentum Strategy başlatıldı: {symbol}")
    
    async def on_tick(self, tick: MarketTick) -> Optional[TradeSignal]:
        """Her tick'te sinyal üret"""
        if tick.symbol != self.symbol:
            return None
        
        # Store data
        self.prices.append(tick.price)
        self.volumes.append(tick.volume)
        self.timestamps.append(tick.timestamp)
        
        # Technical strategy buffer update
        self.technical_strategy.prices.append(tick.price)
        self.technical_strategy.volumes.append(tick.volume)
        self.technical_strategy.high_prices.append(tick.price)
        self.technical_strategy.low_prices.append(tick.price)
        
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
            ml_pred, regime, tick
        )
        
        if fused_signal is None:
            return None
        
        self.fusion_history.append(fused_signal)
        if len(self.fusion_history) > 1000:
            self.fusion_history.pop(0)
        
        if fused_signal.side == Side.HOLD:
            return None
        
        if fused_signal.confidence < self.min_confidence:
            return None
        
        if self._should_skip_signal(fused_signal):
            return None
        
        quantity = self._calculate_position_size(fused_signal, tech_indicators)
        
        signal = TradeSignal(
            symbol=self.symbol,
            side=fused_signal.side,
            price=tick.price,
            quantity=quantity,
            strategy_name=self.name,
            timestamp=datetime.now()
        )
        
        self.signals_generated += 1
        if ml_pred and ml_pred.confidence > 0:
            self.ml_predictions_used += 1
        
        self._update_position_state(signal, tick)
        
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
        """Technical ve ML sinyallerini birleştir"""
        components = {
            'technical': 0.0,
            'ml': 0.0,
            'regime': 0.0
        }
        
        tech_weight = 1 - self.ml_weight
        if tech_side == Side.BUY:
            components['technical'] = tech_confidence * tech_weight
        elif tech_side == Side.SELL:
            components['technical'] = -tech_confidence * tech_weight
        
        if ml_pred and ml_pred.confidence > 0.3:
            ml_signal_value = ml_pred.signal.value / 2
            components['ml'] = ml_signal_value * ml_pred.confidence * self.ml_weight
        
        regime_mult = 1.0
        if regime.trending and regime.strength > 50:
            if regime.trend_direction == "BULLISH":
                regime_mult = 1.2
            elif regime.trend_direction == "BEARISH":
                regime_mult = 0.8
        elif not regime.trending:
            regime_mult = 0.7
        
        components['regime'] = regime_mult
        
        raw_signal = components['technical'] + components['ml']
        adjusted_signal = raw_signal * regime_mult
        
        if adjusted_signal > 0.1:
            side = Side.BUY
            strength = min(1.0, adjusted_signal)
        elif adjusted_signal < -0.1:
            side = Side.SELL
            strength = min(1.0, abs(adjusted_signal))
        else:
            side = Side.HOLD
            strength = 0.0
        
        agreement = self._calculate_agreement(tech_side, ml_pred)
        confidence = strength * (0.5 + 0.5 * agreement)
        
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
        """Technical ve ML uyumu"""
        if ml_pred is None or ml_pred.confidence < 0.3:
            return 0.5
        
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
            return 1.0
        elif tech_direction == -ml_direction and tech_direction != 0:
            return 0.0
        else:
            return 0.5
    
    def _should_skip_signal(self, fused: FusedSignal) -> bool:
        """Sinyal atlanmalı mı?"""
        if self.position_open:
            if fused.side == Side.BUY and self.position_side == "LONG":
                return True
            if fused.side == Side.SELL and not self.position_side:
                return True
        return False
    
    def _calculate_position_size(
        self, 
        fused: FusedSignal, 
        indicators: TechnicalIndicators
    ) -> float:
        """Pozisyon büyüklüğü"""
        base_size = 10.0
        confidence_mult = 0.5 + fused.confidence
        ml_bonus = 1.0 + (fused.ml_contribution * 0.5)
        
        vol_mult = 1.0
        if indicators.atr / self.prices[-1] > 0.02:
            vol_mult = 0.7
        
        size = base_size * confidence_mult * ml_bonus * vol_mult
        return max(1.0, min(100.0, size))
    
    def _update_position_state(self, signal: TradeSignal, tick: MarketTick):
        """Pozisyon durumu güncelle"""
        if signal.side == Side.BUY and not self.position_open:
            self.position_open = True
            self.entry_price = tick.price
            self.entry_time = datetime.now()
            self.position_side = "LONG"
        elif signal.side == Side.SELL and self.position_open:
            self.position_open = False
            self.entry_price = None
            self.entry_time = None
            self.position_side = None
    
    def _provide_learning_feedback(self, tick: MarketTick):
        """Online learning feedback"""
        if len(self.prices) < 10:
            return
        
        features = self.feature_generator.generate(
            np.array(self.prices)[:-5],
            np.array(self.volumes)[:-5]
        )
        
        if features is None:
            return
        
        future_return = (self.prices[-1] - self.prices[-6]) / self.prices[-6]
        
        if future_return > 0.005:
            label = 1
        elif future_return < -0.005:
            label = -1
        else:
            label = 0
        
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


# Export
__all__ = [
    'AdvancedMomentum',
    'MLMomentumStrategy',
    'TechnicalIndicators',
    'MarketRegime',
    'MLSignalGenerator',
    'FeatureGenerator',
    'MLPrediction',
    'FusedSignal',
    'MLModelType',
    'SignalType'
]