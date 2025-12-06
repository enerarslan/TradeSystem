"""
KURUMSAL TEKNİK GÖSTERGE KÜTÜPHANESİ
JPMorgan Quantitative Research Division Tarzı

40+ Profesyonel Teknik Gösterge:
- Trend Göstergeleri (MA, EMA, DEMA, TEMA, WMA, HMA)
- Momentum Göstergeleri (RSI, MACD, Stochastic, Williams %R, CCI, ROC)
- Volatilite Göstergeleri (ATR, Bollinger, Keltner, Donchian)
- Hacim Göstergeleri (OBV, MFI, VWAP, AD Line)
- Trend Gücü (ADX, Aroon, Parabolic SAR)
- Custom Göstergeler (Squeeze, Supertrend, Ichimoku)

Özellikler:
- Vectorized hesaplama (NumPy/Pandas)
- NaN handling
- Configurable parameters
- Caching support
- Type hints
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import warnings

warnings.filterwarnings('ignore')


class IndicatorCategory(Enum):
    """Gösterge kategorileri"""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    STRENGTH = "strength"
    CUSTOM = "custom"


@dataclass
class IndicatorResult:
    """Gösterge sonucu"""
    name: str
    value: Union[float, np.ndarray, pd.Series]
    category: IndicatorCategory
    signal: Optional[str] = None  # BUY, SELL, NEUTRAL
    strength: float = 0.0  # -1 to 1
    metadata: Dict = field(default_factory=dict)


class TechnicalIndicators:
    """
    Profesyonel teknik gösterge hesaplayıcı.
    
    Tüm hesaplamalar vectorized ve optimize edilmiştir.
    NaN değerler otomatik olarak handle edilir.
    
    Kullanım:
        ti = TechnicalIndicators()
        
        # Tek gösterge
        rsi = ti.rsi(close_prices, period=14)
        
        # Tüm göstergeler
        features = ti.calculate_all(df)
    """
    
    def __init__(self, fillna: bool = True):
        """
        Args:
            fillna: NaN değerleri doldur (forward fill)
        """
        self.fillna = fillna
    
    # =========================================================================
    # TREND GÖSTERGELERİ
    # =========================================================================
    
    def sma(self, data: pd.Series, period: int = 20) -> pd.Series:
        """
        Simple Moving Average (SMA)
        
        Basit hareketli ortalama - Son N dönemin ortalaması.
        """
        result = data.rolling(window=period, min_periods=1).mean()
        return self._handle_nan(result, f"SMA_{period}")
    
    def ema(self, data: pd.Series, period: int = 20) -> pd.Series:
        """
        Exponential Moving Average (EMA)
        
        Üssel hareketli ortalama - Son verilere daha fazla ağırlık verir.
        """
        result = data.ewm(span=period, adjust=False, min_periods=1).mean()
        return self._handle_nan(result, f"EMA_{period}")
    
    def dema(self, data: pd.Series, period: int = 20) -> pd.Series:
        """
        Double Exponential Moving Average (DEMA)
        
        DEMA = 2 * EMA - EMA(EMA)
        Daha az gecikme ile trendi takip eder.
        """
        ema1 = self.ema(data, period)
        ema2 = self.ema(ema1, period)
        result = 2 * ema1 - ema2
        return self._handle_nan(result, f"DEMA_{period}")
    
    def tema(self, data: pd.Series, period: int = 20) -> pd.Series:
        """
        Triple Exponential Moving Average (TEMA)
        
        TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
        En az gecikme ile trend takibi.
        """
        ema1 = self.ema(data, period)
        ema2 = self.ema(ema1, period)
        ema3 = self.ema(ema2, period)
        result = 3 * ema1 - 3 * ema2 + ema3
        return self._handle_nan(result, f"TEMA_{period}")
    
    def wma(self, data: pd.Series, period: int = 20) -> pd.Series:
        """
        Weighted Moving Average (WMA)
        
        Son verilere lineer artan ağırlık verir.
        """
        weights = np.arange(1, period + 1)
        result = data.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), 
            raw=True
        )
        return self._handle_nan(result, f"WMA_{period}")
    
    def hma(self, data: pd.Series, period: int = 20) -> pd.Series:
        """
        Hull Moving Average (HMA)
        
        HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        Çok düşük gecikme, yüksek smoothing.
        """
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))
        
        wma_half = self.wma(data, half_period)
        wma_full = self.wma(data, period)
        
        raw_hma = 2 * wma_half - wma_full
        result = self.wma(raw_hma, sqrt_period)
        return self._handle_nan(result, f"HMA_{period}")
    
    def vwma(self, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Volume Weighted Moving Average (VWMA)
        
        Hacim ağırlıklı hareketli ortalama.
        """
        pv = close * volume
        result = pv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        return self._handle_nan(result, f"VWMA_{period}")
    
    def kama(self, data: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
        """
        Kaufman Adaptive Moving Average (KAMA)
        
        Piyasa volatilitesine göre adaptif hareketli ortalama.
        Trend dönemlerinde hızlı, sideways dönemlerinde yavaş.
        """
        change = abs(data - data.shift(period))
        volatility = abs(data - data.shift(1)).rolling(window=period).sum()
        
        er = change / volatility  # Efficiency Ratio
        er = er.fillna(0)
        
        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2  # Smoothing Constant
        
        kama = pd.Series(index=data.index, dtype=float)
        kama.iloc[period-1] = data.iloc[period-1]
        
        for i in range(period, len(data)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (data.iloc[i] - kama.iloc[i-1])
        
        return self._handle_nan(kama, f"KAMA_{period}")
    
    # =========================================================================
    # MOMENTUM GÖSTERGELERİ
    # =========================================================================
    
    def rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index (RSI)
        
        0-100 arasında momentum göstergesi.
        >70: Aşırı alım, <30: Aşırı satım
        """
        delta = data.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        result = 100 - (100 / (1 + rs))
        
        return self._handle_nan(result, f"RSI_{period}")
    
    def stochastic_rsi(self, data: pd.Series, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic RSI
        
        RSI'ın stochastic versiyonu - daha hassas sinyaller.
        Returns: (stoch_rsi_k, stoch_rsi_d)
        """
        rsi = self.rsi(data, period)
        
        rsi_min = rsi.rolling(window=period).min()
        rsi_max = rsi.rolling(window=period).max()
        
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100
        stoch_rsi_k = stoch_rsi.rolling(window=smooth_k).mean()
        stoch_rsi_d = stoch_rsi_k.rolling(window=smooth_d).mean()
        
        return (
            self._handle_nan(stoch_rsi_k, "StochRSI_K"),
            self._handle_nan(stoch_rsi_d, "StochRSI_D")
        )
    
    def macd(
        self, 
        data: pd.Series, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence (MACD)
        
        Trend takibi ve momentum göstergesi.
        Returns: (macd_line, signal_line, histogram)
        """
        ema_fast = self.ema(data, fast)
        ema_slow = self.ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return (
            self._handle_nan(macd_line, "MACD"),
            self._handle_nan(signal_line, "MACD_Signal"),
            self._handle_nan(histogram, "MACD_Hist")
        )
    
    def stochastic(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator
        
        %K ve %D çizgileri ile momentum ölçümü.
        Returns: (%K, %D)
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        stoch_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        stoch_d = stoch_k.rolling(window=d_period).mean()
        
        return (
            self._handle_nan(stoch_k, f"Stoch_K_{k_period}"),
            self._handle_nan(stoch_d, f"Stoch_D_{d_period}")
        )
    
    def williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Williams %R
        
        -100 ile 0 arasında momentum göstergesi.
        <-80: Aşırı satım, >-20: Aşırı alım
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        result = ((highest_high - close) / (highest_high - lowest_low)) * -100
        return self._handle_nan(result, f"WilliamsR_{period}")
    
    def cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """
        Commodity Channel Index (CCI)
        
        Fiyatın istatistiksel ortalamadan sapması.
        >100: Aşırı alım, <-100: Aşırı satım
        """
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        
        result = (typical_price - sma) / (0.015 * mad)
        return self._handle_nan(result, f"CCI_{period}")
    
    def roc(self, data: pd.Series, period: int = 12) -> pd.Series:
        """
        Rate of Change (ROC)
        
        Yüzde değişim momentum göstergesi.
        """
        result = ((data - data.shift(period)) / data.shift(period)) * 100
        return self._handle_nan(result, f"ROC_{period}")
    
    def momentum(self, data: pd.Series, period: int = 10) -> pd.Series:
        """
        Momentum
        
        Basit fiyat farkı momentum göstergesi.
        """
        result = data - data.shift(period)
        return self._handle_nan(result, f"MOM_{period}")
    
    def tsi(self, data: pd.Series, long_period: int = 25, short_period: int = 13) -> pd.Series:
        """
        True Strength Index (TSI)
        
        Çift smoothed momentum göstergesi.
        """
        diff = data.diff()
        
        double_smoothed = self.ema(self.ema(diff, long_period), short_period)
        double_smoothed_abs = self.ema(self.ema(diff.abs(), long_period), short_period)
        
        result = 100 * (double_smoothed / double_smoothed_abs)
        return self._handle_nan(result, "TSI")
    
    def ultimate_oscillator(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series,
        period1: int = 7,
        period2: int = 14,
        period3: int = 28
    ) -> pd.Series:
        """
        Ultimate Oscillator
        
        Çoklu zaman dilimi momentum göstergesi.
        """
        prev_close = close.shift(1)
        
        bp = close - np.minimum(low, prev_close)  # Buying Pressure
        tr = np.maximum(high, prev_close) - np.minimum(low, prev_close)  # True Range
        
        avg1 = bp.rolling(window=period1).sum() / tr.rolling(window=period1).sum()
        avg2 = bp.rolling(window=period2).sum() / tr.rolling(window=period2).sum()
        avg3 = bp.rolling(window=period3).sum() / tr.rolling(window=period3).sum()
        
        result = 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7
        return self._handle_nan(result, "UO")
    
    # =========================================================================
    # VOLATİLİTE GÖSTERGELERİ
    # =========================================================================
    
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Average True Range (ATR)
        
        Volatilite ölçümü - Position sizing için kritik.
        """
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        result = true_range.ewm(alpha=1/period, min_periods=period).mean()
        
        return self._handle_nan(result, f"ATR_{period}")
    
    def atr_percent(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        ATR as percentage of price
        
        Normalize edilmiş volatilite.
        """
        atr = self.atr(high, low, close, period)
        result = (atr / close) * 100
        return self._handle_nan(result, f"ATR_PCT_{period}")
    
    def bollinger_bands(
        self, 
        data: pd.Series, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands
        
        Volatilite bazlı bantlar.
        Returns: (upper, middle, lower, bandwidth)
        """
        middle = self.sma(data, period)
        std = data.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        bandwidth = ((upper - lower) / middle) * 100
        
        return (
            self._handle_nan(upper, "BB_Upper"),
            self._handle_nan(middle, "BB_Middle"),
            self._handle_nan(lower, "BB_Lower"),
            self._handle_nan(bandwidth, "BB_Width")
        )
    
    def bollinger_percent_b(self, data: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """
        Bollinger %B
        
        Fiyatın bantlar içindeki konumu (0-1).
        """
        upper, middle, lower, _ = self.bollinger_bands(data, period, std_dev)
        result = (data - lower) / (upper - lower)
        return self._handle_nan(result, "BB_PctB")
    
    def keltner_channels(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series,
        period: int = 20,
        atr_mult: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Keltner Channels
        
        ATR bazlı volatilite bantları.
        Returns: (upper, middle, lower)
        """
        middle = self.ema(close, period)
        atr = self.atr(high, low, close, period)
        
        upper = middle + (atr * atr_mult)
        lower = middle - (atr * atr_mult)
        
        return (
            self._handle_nan(upper, "KC_Upper"),
            self._handle_nan(middle, "KC_Middle"),
            self._handle_nan(lower, "KC_Lower")
        )
    
    def donchian_channels(
        self, 
        high: pd.Series, 
        low: pd.Series,
        period: int = 20
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Donchian Channels
        
        Breakout stratejileri için kullanılır.
        Returns: (upper, middle, lower)
        """
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        middle = (upper + lower) / 2
        
        return (
            self._handle_nan(upper, f"DC_Upper_{period}"),
            self._handle_nan(middle, f"DC_Middle_{period}"),
            self._handle_nan(lower, f"DC_Lower_{period}")
        )
    
    def historical_volatility(self, data: pd.Series, period: int = 20, annualize: bool = True) -> pd.Series:
        """
        Historical Volatility
        
        Log return'lerin standart sapması.
        """
        log_returns = np.log(data / data.shift(1))
        result = log_returns.rolling(window=period).std()
        
        if annualize:
            result = result * np.sqrt(252)  # Yıllık
        
        return self._handle_nan(result, f"HV_{period}")
    
    # =========================================================================
    # TREND GÜCÜ GÖSTERGELERİ
    # =========================================================================
    
    def adx(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series,
        period: int = 14
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Average Directional Index (ADX)
        
        Trend gücü ölçümü.
        >25: Güçlü trend, <20: Zayıf trend/sideways
        Returns: (ADX, +DI, -DI)
        """
        # True Range
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        ], axis=1).max(axis=1)
        
        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # Smoothed values
        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period, min_periods=period).mean()
        
        return (
            self._handle_nan(adx, f"ADX_{period}"),
            self._handle_nan(plus_di, f"DI_Plus_{period}"),
            self._handle_nan(minus_di, f"DI_Minus_{period}")
        )
    
    def aroon(self, high: pd.Series, low: pd.Series, period: int = 25) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Aroon Indicator
        
        Trend başlangıcı ve gücü.
        Returns: (aroon_up, aroon_down, aroon_oscillator)
        """
        aroon_up = 100 * high.rolling(window=period + 1).apply(
            lambda x: (period - x.argmax()) / period, raw=True
        )
        aroon_down = 100 * low.rolling(window=period + 1).apply(
            lambda x: (period - x.argmin()) / period, raw=True
        )
        aroon_osc = aroon_up - aroon_down
        
        return (
            self._handle_nan(aroon_up, f"Aroon_Up_{period}"),
            self._handle_nan(aroon_down, f"Aroon_Down_{period}"),
            self._handle_nan(aroon_osc, f"Aroon_Osc_{period}")
        )
    
    def psar(
        self, 
        high: pd.Series, 
        low: pd.Series,
        af_start: float = 0.02,
        af_max: float = 0.2
    ) -> pd.Series:
        """
        Parabolic SAR
        
        Trend takibi ve stop-loss seviyeleri.
        """
        length = len(high)
        psar = pd.Series(index=high.index, dtype=float)
        af = af_start
        uptrend = True
        
        psar.iloc[0] = low.iloc[0]
        ep = high.iloc[0]  # Extreme Point
        
        for i in range(1, length):
            if uptrend:
                psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
                psar.iloc[i] = min(psar.iloc[i], low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1])
                
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + af_start, af_max)
                
                if low.iloc[i] < psar.iloc[i]:
                    uptrend = False
                    psar.iloc[i] = ep
                    ep = low.iloc[i]
                    af = af_start
            else:
                psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
                psar.iloc[i] = max(psar.iloc[i], high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1])
                
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + af_start, af_max)
                
                if high.iloc[i] > psar.iloc[i]:
                    uptrend = True
                    psar.iloc[i] = ep
                    ep = high.iloc[i]
                    af = af_start
        
        return self._handle_nan(psar, "PSAR")
    
    # =========================================================================
    # HACİM GÖSTERGELERİ
    # =========================================================================
    
    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume (OBV)
        
        Kümülatif hacim akışı.
        """
        direction = np.where(close > close.shift(1), 1, 
                    np.where(close < close.shift(1), -1, 0))
        result = (volume * direction).cumsum()
        return self._handle_nan(pd.Series(result, index=close.index), "OBV")
    
    def mfi(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series,
        volume: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Money Flow Index (MFI)
        
        Hacim ağırlıklı RSI - Para akışı göstergesi.
        """
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        money_ratio = positive_mf / negative_mf
        result = 100 - (100 / (1 + money_ratio))
        
        return self._handle_nan(result, f"MFI_{period}")
    
    def vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Volume Weighted Average Price (VWAP)
        
        Kurumsal yatırımcıların referans fiyatı.
        """
        typical_price = (high + low + close) / 3
        cum_vol = volume.cumsum()
        cum_tp_vol = (typical_price * volume).cumsum()
        
        result = cum_tp_vol / cum_vol
        return self._handle_nan(result, "VWAP")
    
    def ad_line(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Accumulation/Distribution Line
        
        Alım/satım baskısı göstergesi.
        """
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        ad = (clv * volume).cumsum()
        
        return self._handle_nan(ad, "AD_Line")
    
    def cmf(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series,
        volume: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """
        Chaikin Money Flow (CMF)
        
        Belirli dönem için para akışı.
        """
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        
        result = (clv * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
        return self._handle_nan(result, f"CMF_{period}")
    
    def volume_sma(self, volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Volume SMA
        
        Hacim ortalaması.
        """
        return self._handle_nan(volume.rolling(window=period).mean(), f"Vol_SMA_{period}")
    
    def volume_ratio(self, volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Volume Ratio
        
        Mevcut hacim / Ortalama hacim.
        """
        vol_sma = self.volume_sma(volume, period)
        result = volume / vol_sma
        return self._handle_nan(result, f"Vol_Ratio_{period}")
    
    # =========================================================================
    # CUSTOM / ADVANCED GÖSTERGELERİ
    # =========================================================================
    
    def squeeze_momentum(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        bb_period: int = 20,
        bb_mult: float = 2.0,
        kc_period: int = 20,
        kc_mult: float = 1.5
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Squeeze Momentum Indicator (TTM Squeeze)
        
        BB ve KC kombinasyonu - Volatilite sıkışması tespiti.
        Returns: (squeeze_on, momentum)
        """
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower, _ = self.bollinger_bands(close, bb_period, bb_mult)
        
        # Keltner Channels
        kc_upper, kc_middle, kc_lower = self.keltner_channels(high, low, close, kc_period, kc_mult)
        
        # Squeeze detection
        squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        
        # Momentum (Linear Regression)
        highest = high.rolling(window=kc_period).max()
        lowest = low.rolling(window=kc_period).min()
        
        m1 = (highest + lowest) / 2
        m2 = self.sma(close, kc_period)
        
        momentum = close - (m1 + m2) / 2
        
        return (
            self._handle_nan(squeeze_on.astype(int), "Squeeze_On"),
            self._handle_nan(momentum, "Squeeze_Mom")
        )
    
    def supertrend(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 10,
        multiplier: float = 3.0
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Supertrend Indicator
        
        Trend takibi ve dinamik stop-loss.
        Returns: (supertrend, direction)
        """
        atr = self.atr(high, low, close, period)
        hl2 = (high + low) / 2
        
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=int)
        
        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = -1
        
        for i in range(1, len(close)):
            if close.iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            elif close.iloc[i] < supertrend.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]
                
                if direction.iloc[i] == 1 and lower_band.iloc[i] > supertrend.iloc[i]:
                    supertrend.iloc[i] = lower_band.iloc[i]
                elif direction.iloc[i] == -1 and upper_band.iloc[i] < supertrend.iloc[i]:
                    supertrend.iloc[i] = upper_band.iloc[i]
        
        return (
            self._handle_nan(supertrend, f"Supertrend_{period}"),
            self._handle_nan(direction, f"Supertrend_Dir_{period}")
        )
    
    def ichimoku(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        tenkan: int = 9,
        kijun: int = 26,
        senkou_b: int = 52
    ) -> Dict[str, pd.Series]:
        """
        Ichimoku Cloud
        
        Kapsamlı trend analizi sistemi.
        Returns: dict with all components
        """
        # Tenkan-sen (Conversion Line)
        tenkan_sen = (high.rolling(window=tenkan).max() + low.rolling(window=tenkan).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun_sen = (high.rolling(window=kijun).max() + low.rolling(window=kijun).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        
        # Senkou Span B (Leading Span B)
        senkou_span_b = ((high.rolling(window=senkou_b).max() + low.rolling(window=senkou_b).min()) / 2).shift(kijun)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-kijun)
        
        return {
            'tenkan_sen': self._handle_nan(tenkan_sen, "Ichimoku_Tenkan"),
            'kijun_sen': self._handle_nan(kijun_sen, "Ichimoku_Kijun"),
            'senkou_span_a': self._handle_nan(senkou_span_a, "Ichimoku_SpanA"),
            'senkou_span_b': self._handle_nan(senkou_span_b, "Ichimoku_SpanB"),
            'chikou_span': self._handle_nan(chikou_span, "Ichimoku_Chikou")
        }
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _handle_nan(self, data: pd.Series, name: str) -> pd.Series:
        """NaN değerleri handle et"""
        result = data.copy()
        result.name = name
        
        if self.fillna:
            result = result.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return result
    
    def calculate_all(
        self,
        df: pd.DataFrame,
        include_advanced: bool = True
    ) -> pd.DataFrame:
        """
        Tüm göstergeleri hesapla ve DataFrame olarak döndür.
        
        Args:
            df: OHLCV DataFrame (columns: open, high, low, close, volume)
            include_advanced: Gelişmiş göstergeleri dahil et
        
        Returns:
            pd.DataFrame: Tüm göstergeler
        """
        result = pd.DataFrame(index=df.index)
        
        # Extract columns
        high = df['high'] if 'high' in df.columns else df['close']
        low = df['low'] if 'low' in df.columns else df['close']
        close = df['close']
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
        
        # Trend indicators
        for period in [10, 20, 50, 200]:
            result[f'SMA_{period}'] = self.sma(close, period)
            result[f'EMA_{period}'] = self.ema(close, period)
        
        result['DEMA_20'] = self.dema(close, 20)
        result['TEMA_20'] = self.tema(close, 20)
        result['HMA_20'] = self.hma(close, 20)
        result['VWMA_20'] = self.vwma(close, volume, 20)
        
        # Momentum indicators
        result['RSI_14'] = self.rsi(close, 14)
        result['RSI_7'] = self.rsi(close, 7)
        
        stoch_k, stoch_d = self.stochastic(high, low, close)
        result['Stoch_K'] = stoch_k
        result['Stoch_D'] = stoch_d
        
        stoch_rsi_k, stoch_rsi_d = self.stochastic_rsi(close)
        result['StochRSI_K'] = stoch_rsi_k
        result['StochRSI_D'] = stoch_rsi_d
        
        macd, macd_signal, macd_hist = self.macd(close)
        result['MACD'] = macd
        result['MACD_Signal'] = macd_signal
        result['MACD_Hist'] = macd_hist
        
        result['Williams_R'] = self.williams_r(high, low, close)
        result['CCI_20'] = self.cci(high, low, close, 20)
        result['ROC_12'] = self.roc(close, 12)
        result['Momentum_10'] = self.momentum(close, 10)
        result['TSI'] = self.tsi(close)
        result['UO'] = self.ultimate_oscillator(high, low, close)
        
        # Volatility indicators
        result['ATR_14'] = self.atr(high, low, close, 14)
        result['ATR_PCT'] = self.atr_percent(high, low, close, 14)
        
        bb_upper, bb_middle, bb_lower, bb_width = self.bollinger_bands(close)
        result['BB_Upper'] = bb_upper
        result['BB_Middle'] = bb_middle
        result['BB_Lower'] = bb_lower
        result['BB_Width'] = bb_width
        result['BB_PctB'] = self.bollinger_percent_b(close)
        
        kc_upper, kc_middle, kc_lower = self.keltner_channels(high, low, close)
        result['KC_Upper'] = kc_upper
        result['KC_Middle'] = kc_middle
        result['KC_Lower'] = kc_lower
        
        result['HV_20'] = self.historical_volatility(close, 20)
        
        # Trend strength
        adx, di_plus, di_minus = self.adx(high, low, close)
        result['ADX'] = adx
        result['DI_Plus'] = di_plus
        result['DI_Minus'] = di_minus
        
        aroon_up, aroon_down, aroon_osc = self.aroon(high, low)
        result['Aroon_Up'] = aroon_up
        result['Aroon_Down'] = aroon_down
        result['Aroon_Osc'] = aroon_osc
        
        result['PSAR'] = self.psar(high, low)
        
        # Volume indicators
        result['OBV'] = self.obv(close, volume)
        result['MFI'] = self.mfi(high, low, close, volume)
        result['VWAP'] = self.vwap(high, low, close, volume)
        result['AD_Line'] = self.ad_line(high, low, close, volume)
        result['CMF'] = self.cmf(high, low, close, volume)
        result['Vol_SMA_20'] = self.volume_sma(volume, 20)
        result['Vol_Ratio'] = self.volume_ratio(volume, 20)
        
        # Advanced indicators
        if include_advanced:
            squeeze_on, squeeze_mom = self.squeeze_momentum(high, low, close)
            result['Squeeze_On'] = squeeze_on
            result['Squeeze_Mom'] = squeeze_mom
            
            supertrend, supertrend_dir = self.supertrend(high, low, close)
            result['Supertrend'] = supertrend
            result['Supertrend_Dir'] = supertrend_dir
            
            ichimoku = self.ichimoku(high, low, close)
            for key, value in ichimoku.items():
                result[f'Ichimoku_{key}'] = value
        
        # Price-based features
        result['Price_vs_SMA20'] = (close / result['SMA_20'] - 1) * 100
        result['Price_vs_SMA50'] = (close / result['SMA_50'] - 1) * 100
        result['Price_vs_VWAP'] = (close / result['VWAP'] - 1) * 100
        
        return result


# Export
__all__ = [
    'TechnicalIndicators',
    'IndicatorCategory',
    'IndicatorResult'
]