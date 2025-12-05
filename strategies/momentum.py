"""
KURUMSAL SEVİYE MOMENTUM STRATEJİSİ
JPMorgan Quantitative Research tarzı

Çoklu teknik gösterge kombinasyonu:
- Dual Moving Average Crossover
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Volume Confirmation
- ATR-based Position Sizing
- Dynamic Threshold Adjustment
"""

import numpy as np
from collections import deque
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from dataclasses import dataclass

from strategies.base import BaseStrategy
from data.models import MarketTick, TradeSignal, Side
from utils.logger import log


@dataclass
class TechnicalIndicators:
    """Teknik göstergelerin snapshot'ı"""
    sma_fast: float
    sma_slow: float
    ema_fast: float
    ema_slow: float
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    atr: float
    volume_sma: float
    volume_ratio: float


@dataclass
class MarketRegime:
    """Piyasa rejimi tespiti"""
    trending: bool  # Trend var mı?
    trend_direction: str  # "BULLISH", "BEARISH", "SIDEWAYS"
    volatility: str  # "HIGH", "MEDIUM", "LOW"
    volume_trend: str  # "INCREASING", "DECREASING", "STABLE"
    strength: float  # Trend gücü (0-100)


class AdvancedMomentum(BaseStrategy):
    """
    Gelişmiş çok katmanlı momentum stratejisi.
    
    Sinyal Üretme Mantığı:
    1. Dual MA Crossover (Hızlı EMA, Yavaş SMA kesişimi)
    2. RSI Confirmation (Aşırı alım/satım kontrolü)
    3. MACD Confirmation (Momentum doğrulama)
    4. Volume Confirmation (Hacim artışı)
    5. Bollinger Bands (Volatilite filtreleme)
    6. Market Regime Filter (Trend veya sideways)
    
    Çıkış Stratejisi:
    - Trailing Stop (ATR tabanlı)
    - Profit Target (2x ATR)
    - Time-based Exit (Max hold period)
    - Reverse Signal
    """
    
    def __init__(
        self,
        symbol: str,
        fast_period: int = 10,
        slow_period: int = 30,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        atr_period: int = 14,
        lookback: int = 200,
        min_confidence: float = 0.6
    ):
        super().__init__(name="AdvancedMomentum_V2")
        
        self.symbol = symbol
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.atr_period = atr_period
        self.lookback = lookback
        self.min_confidence = min_confidence
        
        # Veri buffer'ları
        self.prices = deque(maxlen=lookback)
        self.volumes = deque(maxlen=lookback)
        self.high_prices = deque(maxlen=lookback)
        self.low_prices = deque(maxlen=lookback)
        
        # EMA state (exponential moving average için)
        self.ema_fast = None
        self.ema_slow = None
        self.macd_ema = None
        
        # Pozisyon tracking
        self.position_open = False
        self.entry_price = None
        self.entry_time = None
        self.position_side = None
        
        # Performance tracking
        self.signals_generated = 0
        self.signals_by_type = {
            'BUY': 0,
            'SELL': 0,
            'HOLD': 0
        }
        
        log.info(f"🎯 Advanced Momentum Strategy başlatıldı: {symbol}")
        log.debug(f"   Fast Period: {fast_period}, Slow Period: {slow_period}")
        log.debug(f"   RSI: {rsi_period}, MACD: ({macd_fast},{macd_slow},{macd_signal})")
    
    async def on_tick(self, tick: MarketTick) -> Optional[TradeSignal]:
        """
        Her tick'te çağrılır. Çoklu gösterge analizi yapar.
        """
        if tick.symbol != self.symbol:
            return None
        
        # Veriyi kaydet
        self.prices.append(tick.price)
        self.volumes.append(tick.volume)
        self.high_prices.append(tick.price)  # Gerçek OHLC verisi yoksa close kullan
        self.low_prices.append(tick.price)
        
        # Yeterli veri yoksa bekle
        if len(self.prices) < max(self.slow_period, self.rsi_period, self.atr_period):
            return None
        
        # 1. Teknik göstergeleri hesapla
        indicators = self._calculate_indicators()
        
        if indicators is None:
            return None
        
        # 2. Piyasa rejimini tespit et
        regime = self._detect_market_regime(indicators)
        
        # 3. Sinyal üret (Çoklu koşul kontrolü)
        signal_side, confidence = self._generate_signal(indicators, regime, tick)
        
        # 4. Minimum güven eşiğini kontrol et
        if confidence < self.min_confidence:
            return None
        
        # 5. Pozisyon yönetimi (Zaten pozisyondaysa çıkış sinyali mi?)
        if self.position_open:
            exit_signal = self._check_exit_conditions(tick, indicators)
            if exit_signal:
                signal_side = exit_signal
                confidence = 0.9  # Çıkış sinyalleri yüksek öncelikli
        
        # 6. Hold ise sinyal döndürme
        if signal_side == Side.HOLD:
            return None
        
        # 7. Pozisyon büyüklüğünü hesapla (ATR-based)
        quantity = self._calculate_position_size(indicators, confidence)
        
        # 8. TradeSignal oluştur
        signal = TradeSignal(
            symbol=self.symbol,
            side=signal_side,
            price=tick.price,
            quantity=quantity,
            strategy_name=self.name,
            timestamp=datetime.now()
        )
        
        # 9. İstatistikleri güncelle
        self.signals_generated += 1
        self.signals_by_type[signal_side] += 1
        
        # 10. Pozisyon durumunu güncelle
        if signal_side == Side.BUY:
            self.position_open = True
            self.entry_price = tick.price
            self.entry_time = datetime.now()
            self.position_side = "LONG"
            log.info(f"🟢 LONG POZİSYON AÇILDI: {self.symbol} @ ${tick.price:.2f} (Confidence: {confidence:.2%})")
        
        elif signal_side == Side.SELL and self.position_open:
            pnl_pct = ((tick.price - self.entry_price) / self.entry_price) * 100 if self.entry_price else 0
            log.info(f"🔴 POZİSYON KAPATILDI: {self.symbol} @ ${tick.price:.2f} (PnL: {pnl_pct:+.2f}%)")
            self.position_open = False
            self.entry_price = None
            self.entry_time = None
            self.position_side = None
        
        return signal
    
    def _calculate_indicators(self) -> Optional[TechnicalIndicators]:
        """Tüm teknik göstergeleri hesaplar"""
        try:
            prices_arr = np.array(self.prices)
            volumes_arr = np.array(self.volumes)
            
            # Moving Averages
            sma_fast = np.mean(prices_arr[-self.fast_period:])
            sma_slow = np.mean(prices_arr[-self.slow_period:])
            
            # EMA (Exponential Moving Average)
            ema_fast = self._calculate_ema(prices_arr, self.fast_period)
            ema_slow = self._calculate_ema(prices_arr, self.slow_period)
            
            # RSI
            rsi = self._calculate_rsi(prices_arr, self.rsi_period)
            
            # MACD
            macd, macd_signal, macd_hist = self._calculate_macd(prices_arr)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower, bb_width = self._calculate_bollinger_bands(prices_arr)
            
            # ATR (Average True Range)
            atr = self._calculate_atr()
            
            # Volume Analysis
            volume_sma = np.mean(volumes_arr[-20:])
            volume_ratio = volumes_arr[-1] / volume_sma if volume_sma > 0 else 1.0
            
            return TechnicalIndicators(
                sma_fast=sma_fast,
                sma_slow=sma_slow,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                rsi=rsi,
                macd=macd,
                macd_signal=macd_signal,
                macd_histogram=macd_hist,
                bb_upper=bb_upper,
                bb_middle=bb_middle,
                bb_lower=bb_lower,
                bb_width=bb_width,
                atr=atr,
                volume_sma=volume_sma,
                volume_ratio=volume_ratio
            )
            
        except Exception as e:
            log.error(f"Indicator calculation error: {e}")
            return None
    
    def _calculate_ema(self, data: np.ndarray, period: int) -> float:
        """Exponential Moving Average hesaplar (verimli)"""
        multiplier = 2 / (period + 1)
        
        # İlk değer SMA
        if self.ema_fast is None:  # İlk hesaplama
            return np.mean(data[-period:])
        
        # EMA = (Close - EMA_prev) * multiplier + EMA_prev
        if period == self.fast_period:
            self.ema_fast = (data[-1] - self.ema_fast) * multiplier + self.ema_fast
            return self.ema_fast
        else:
            self.ema_slow = (data[-1] - self.ema_slow) * multiplier + self.ema_slow
            return self.ema_slow
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """RSI (Relative Strength Index) hesaplar"""
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray) -> Tuple[float, float, float]:
        """MACD (Moving Average Convergence Divergence) hesaplar"""
        ema_fast = self._ema_array(prices, self.macd_fast)
        ema_slow = self._ema_array(prices, self.macd_slow)
        
        macd = ema_fast - ema_slow
        macd_signal = self._ema_array(
            np.append(prices[-self.macd_signal:], [macd]), 
            self.macd_signal
        )
        macd_histogram = macd - macd_signal
        
        return macd, macd_signal, macd_histogram
    
    def _ema_array(self, data: np.ndarray, period: int) -> float:
        """Array için EMA hesaplar (helper)"""
        multiplier = 2 / (period + 1)
        ema = data[0]
        for price in data[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema
    
    def _calculate_bollinger_bands(
        self, 
        prices: np.ndarray, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> Tuple[float, float, float, float]:
        """Bollinger Bands hesaplar"""
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        width = (upper - lower) / sma if sma > 0 else 0
        
        return upper, sma, lower, width
    
    def _calculate_atr(self, period: int = 14) -> float:
        """Average True Range hesaplar (volatility ölçüsü)"""
        if len(self.high_prices) < period:
            return 0.0
        
        highs = np.array(self.high_prices)[-period:]
        lows = np.array(self.low_prices)[-period:]
        closes = np.array(self.prices)[-period-1:-1]
        
        tr1 = highs - lows
        tr2 = np.abs(highs - closes)
        tr3 = np.abs(lows - closes)
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.mean(true_range)
        
        return atr
    
    def _detect_market_regime(self, ind: TechnicalIndicators) -> MarketRegime:
        """
        Piyasa rejimini tespit eder (Trending vs Sideways)
        """
        current_price = self.prices[-1]
        
        # 1. Trend Direction
        if ind.ema_fast > ind.ema_slow and current_price > ind.sma_slow:
            trend_direction = "BULLISH"
            trending = True
        elif ind.ema_fast < ind.ema_slow and current_price < ind.sma_slow:
            trend_direction = "BEARISH"
            trending = True
        else:
            trend_direction = "SIDEWAYS"
            trending = False
        
        # 2. Volatility (ATR ve BB width ile)
        if ind.atr / current_price > 0.02 or ind.bb_width > 0.04:
            volatility = "HIGH"
        elif ind.atr / current_price < 0.01:
            volatility = "LOW"
        else:
            volatility = "MEDIUM"
        
        # 3. Volume Trend
        if ind.volume_ratio > 1.2:
            volume_trend = "INCREASING"
        elif ind.volume_ratio < 0.8:
            volume_trend = "DECREASING"
        else:
            volume_trend = "STABLE"
        
        # 4. Trend Strength (ADX benzeri basit hesaplama)
        price_momentum = abs(ind.ema_fast - ind.ema_slow) / ind.ema_slow
        strength = min(100, price_momentum * 1000)
        
        return MarketRegime(
            trending=trending,
            trend_direction=trend_direction,
            volatility=volatility,
            volume_trend=volume_trend,
            strength=strength
        )
    
    def _generate_signal(
        self, 
        ind: TechnicalIndicators, 
        regime: MarketRegime,
        tick: MarketTick
    ) -> Tuple[Side, float]:
        """
        Çoklu koşul kontrolleriyle sinyal üretir.
        Returns: (Side, confidence_score)
        """
        current_price = tick.price
        confidence = 0.0
        signal_side = Side.HOLD
        
        # ===== LONG SİNYALİ KOŞULLARI =====
        buy_conditions = []
        
        # 1. MA Crossover (EMA fast > EMA slow ve fiyat SMA üstünde)
        if ind.ema_fast > ind.ema_slow and current_price > ind.sma_slow:
            buy_conditions.append(("MA_CROSSOVER", 0.25))
        
        # 2. RSI (30-70 arası, aşırı satımdan çıkış)
        if 30 < ind.rsi < 70 and ind.rsi > 40:
            buy_conditions.append(("RSI_BULLISH", 0.15))
        
        # 3. MACD Pozitif ve histogram yükseliyor
        if ind.macd > ind.macd_signal and ind.macd_histogram > 0:
            buy_conditions.append(("MACD_BULLISH", 0.20))
        
        # 4. Bollinger Bands (Alt banddan yukarı çıkış)
        bb_position = (current_price - ind.bb_lower) / (ind.bb_upper - ind.bb_lower) if ind.bb_upper != ind.bb_lower else 0.5
        if 0.2 < bb_position < 0.6:  # Alt bant yakınından yukarı
            buy_conditions.append(("BB_POSITION", 0.15))
        
        # 5. Volume Confirmation (Hacim artışı)
        if ind.volume_ratio > 1.1:
            buy_conditions.append(("VOLUME_SURGE", 0.15))
        
        # 6. Trend Confirmation
        if regime.trend_direction == "BULLISH" and regime.strength > 30:
            buy_conditions.append(("TREND_BULLISH", 0.10))
        
        # ===== SHORT/SELL SİNYALİ KOŞULLARI =====
        sell_conditions = []
        
        # 1. MA Crossover (Negatif)
        if ind.ema_fast < ind.ema_slow and current_price < ind.sma_slow:
            sell_conditions.append(("MA_CROSSOVER_NEG", 0.25))
        
        # 2. RSI Aşırı alım
        if ind.rsi > 70:
            sell_conditions.append(("RSI_OVERBOUGHT", 0.15))
        
        # 3. MACD Negatif
        if ind.macd < ind.macd_signal and ind.macd_histogram < 0:
            sell_conditions.append(("MACD_BEARISH", 0.20))
        
        # 4. Bollinger Bands üst band yakını
        if bb_position > 0.8:
            sell_conditions.append(("BB_UPPER", 0.15))
        
        # 5. Trend Bearish
        if regime.trend_direction == "BEARISH" and regime.strength > 30:
            sell_conditions.append(("TREND_BEARISH", 0.10))
        
        # Confidence hesaplama
        buy_confidence = sum(score for _, score in buy_conditions)
        sell_confidence = sum(score for _, score in sell_conditions)
        
        # Pozisyon yoksa ve buy sinyali varsa
        if not self.position_open and buy_confidence > sell_confidence:
            signal_side = Side.BUY
            confidence = buy_confidence
            log.debug(f"   🟢 BUY Signal: {', '.join([name for name, _ in buy_conditions])}")
        
        # Pozisyon varsa ve sell sinyali varsa
        elif self.position_open and sell_confidence > buy_confidence:
            signal_side = Side.SELL
            confidence = sell_confidence
            log.debug(f"   🔴 SELL Signal: {', '.join([name for name, _ in sell_conditions])}")
        
        return signal_side, confidence
    
    def _check_exit_conditions(
        self, 
        tick: MarketTick, 
        ind: TechnicalIndicators
    ) -> Optional[Side]:
        """
        Açık pozisyon için çıkış koşullarını kontrol eder.
        """
        if not self.position_open or not self.entry_price:
            return None
        
        current_price = tick.price
        pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
        
        # 1. Stop Loss (ATR tabanlı)
        stop_loss_distance = ind.atr * 2
        if current_price < (self.entry_price - stop_loss_distance):
            log.warning(f"🛑 STOP LOSS: PnL: {pnl_pct:.2f}%")
            return Side.SELL
        
        # 2. Take Profit (ATR tabanlı)
        take_profit_distance = ind.atr * 3
        if current_price > (self.entry_price + take_profit_distance):
            log.success(f"🎯 TAKE PROFIT: PnL: {pnl_pct:.2f}%")
            return Side.SELL
        
        # 3. Time-based exit (24 saatlik max hold)
        if self.entry_time:
            holding_period = (datetime.now() - self.entry_time).total_seconds() / 3600
            if holding_period > 24:
                log.info(f"⏱️ TIME EXIT: Holding {holding_period:.1f}h, PnL: {pnl_pct:.2f}%")
                return Side.SELL
        
        # 4. Trailing Stop (Basit: %5 geri düşme)
        if pnl_pct > 3:  # Kârda isek
            if pnl_pct < (pnl_pct * 0.95):  # %5 geri düşerse kapat
                log.info(f"📉 TRAILING STOP: PnL: {pnl_pct:.2f}%")
                return Side.SELL
        
        return None
    
    def _calculate_position_size(
        self, 
        ind: TechnicalIndicators, 
        confidence: float
    ) -> float:
        """
        ATR ve confidence'a göre pozisyon büyüklüğü hesaplar.
        """
        # Base size
        base_size = 10.0
        
        # Confidence multiplier (0.6-1.0 confidence -> 0.5x-2x size)
        confidence_mult = 0.5 + (confidence * 1.5)
        
        # Volatility adjustment (Yüksek volatilitede daha küçük pozisyon)
        volatility_mult = 1.0
        if ind.atr / self.prices[-1] > 0.02:  # %2'den fazla volatilite
            volatility_mult = 0.7
        
        position_size = base_size * confidence_mult * volatility_mult
        
        return max(1.0, position_size)  # Minimum 1 hisse
    
    def get_performance_stats(self) -> Dict:
        """Strateji performans istatistikleri"""
        return {
            'total_signals': self.signals_generated,
            'buy_signals': self.signals_by_type['BUY'],
            'sell_signals': self.signals_by_type['SELL'],
            'position_open': self.position_open,
            'entry_price': self.entry_price,
            'holding_period_hours': (datetime.now() - self.entry_time).total_seconds() / 3600 if self.entry_time else 0
        }


# KULLANIM
"""
strategy = AdvancedMomentum(
    symbol="AAPL",
    fast_period=10,
    slow_period=30,
    min_confidence=0.6
)

# Her tick'te
signal = await strategy.on_tick(tick)
if signal:
    print(f"Signal: {signal.side} {signal.quantity} @ ${signal.price}")
"""