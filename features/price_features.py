"""
KURUMSAL FİYAT ÖZELLİKLERİ
JPMorgan Alpha Research Tarzı

Price-based Features:
- Return calculations (simple, log, excess)
- Price patterns (candlestick patterns)
- Statistical features (skewness, kurtosis)
- Momentum features
- Mean reversion features
- Support/Resistance levels

Bu modül fiyat hareketlerinden anlamlı özellikler çıkarır.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


class CandlePattern(Enum):
    """Mum formasyonları"""
    DOJI = "doji"
    HAMMER = "hammer"
    INVERTED_HAMMER = "inverted_hammer"
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    HARAMI_BULLISH = "harami_bullish"
    HARAMI_BEARISH = "harami_bearish"


class PriceFeatures:
    """
    Fiyat bazlı özellik mühendisliği.
    
    Kategoriler:
    1. Return hesaplamaları
    2. Mum formasyonları
    3. İstatistiksel özellikler
    4. Momentum özellikleri
    5. Mean reversion özellikleri
    6. Destek/Direnç seviyeleri
    """
    
    def __init__(self, fillna: bool = True):
        self.fillna = fillna
    
    # =========================================================================
    # RETURN HESAPLAMALARI
    # =========================================================================
    
    def simple_return(self, data: pd.Series, period: int = 1) -> pd.Series:
        """
        Simple Return (Basit Getiri)
        
        R = (P_t - P_{t-1}) / P_{t-1}
        """
        result = data.pct_change(periods=period)
        return self._handle_nan(result, f"Return_{period}")
    
    def log_return(self, data: pd.Series, period: int = 1) -> pd.Series:
        """
        Log Return (Logaritmik Getiri)
        
        r = ln(P_t / P_{t-1})
        Daha iyi istatistiksel özellikler için tercih edilir.
        """
        result = np.log(data / data.shift(period))
        return self._handle_nan(result, f"LogReturn_{period}")
    
    def cumulative_return(self, data: pd.Series, period: int = 20) -> pd.Series:
        """
        Cumulative Return (Kümülatif Getiri)
        
        Son N dönemdeki toplam getiri.
        """
        result = (data / data.shift(period)) - 1
        return self._handle_nan(result, f"CumReturn_{period}")
    
    def rolling_return(self, data: pd.Series, period: int = 20) -> pd.Series:
        """
        Rolling Return (Yuvarlanmalı Getiri)
        
        Hareketli pencere içindeki ortalama getiri.
        """
        returns = self.simple_return(data, 1)
        result = returns.rolling(window=period).mean()
        return self._handle_nan(result, f"RollingReturn_{period}")
    
    def excess_return(self, data: pd.Series, benchmark: pd.Series, period: int = 1) -> pd.Series:
        """
        Excess Return (Fazla Getiri)
        
        Benchmark'a göre fazla getiri (Alpha proxy).
        """
        asset_return = self.simple_return(data, period)
        bench_return = self.simple_return(benchmark, period)
        result = asset_return - bench_return
        return self._handle_nan(result, f"ExcessReturn_{period}")
    
    def realized_volatility(self, data: pd.Series, period: int = 20) -> pd.Series:
        """
        Realized Volatility
        
        Log return'lerin standart sapması.
        """
        log_returns = self.log_return(data, 1)
        result = log_returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
        return self._handle_nan(result, f"RealizedVol_{period}")
    
    def return_dispersion(self, data: pd.Series, period: int = 20) -> pd.Series:
        """
        Return Dispersion
        
        Return'lerin max-min aralığı.
        """
        returns = self.simple_return(data, 1)
        result = returns.rolling(window=period).max() - returns.rolling(window=period).min()
        return self._handle_nan(result, f"ReturnDispersion_{period}")
    
    # =========================================================================
    # İSTATİSTİKSEL ÖZELLİKLER
    # =========================================================================
    
    def rolling_skewness(self, data: pd.Series, period: int = 20) -> pd.Series:
        """
        Rolling Skewness (Çarpıklık)
        
        Dağılımın simetri ölçüsü.
        >0: Sağa çarpık (pozitif kuyruk)
        <0: Sola çarpık (negatif kuyruk)
        """
        returns = self.log_return(data, 1)
        result = returns.rolling(window=period).skew()
        return self._handle_nan(result, f"Skewness_{period}")
    
    def rolling_kurtosis(self, data: pd.Series, period: int = 20) -> pd.Series:
        """
        Rolling Kurtosis (Basıklık)
        
        Dağılımın kuyruk kalınlığı.
        >3: Leptokurtic (kalın kuyruk)
        <3: Platykurtic (ince kuyruk)
        """
        returns = self.log_return(data, 1)
        result = returns.rolling(window=period).kurt()
        return self._handle_nan(result, f"Kurtosis_{period}")
    
    def rolling_zscore(self, data: pd.Series, period: int = 20) -> pd.Series:
        """
        Rolling Z-Score
        
        Fiyatın ortalamadan sapması (standart sapma cinsinden).
        Mean reversion stratejileri için kritik.
        """
        mean = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        result = (data - mean) / std
        return self._handle_nan(result, f"ZScore_{period}")
    
    def rolling_percentile(self, data: pd.Series, period: int = 100) -> pd.Series:
        """
        Rolling Percentile
        
        Mevcut fiyatın son N dönem içindeki percentile'ı.
        """
        def percentile_rank(x):
            return (x.argsort().argsort()[-1] + 1) / len(x) * 100
        
        result = data.rolling(window=period).apply(percentile_rank, raw=True)
        return self._handle_nan(result, f"Percentile_{period}")
    
    def rolling_autocorrelation(self, data: pd.Series, period: int = 20, lag: int = 1) -> pd.Series:
        """
        Rolling Autocorrelation
        
        Geçmiş return'lerle korelasyon.
        Momentum vs Mean Reversion sinyali.
        """
        returns = self.simple_return(data, 1)
        
        def autocorr(x):
            if len(x) < lag + 1:
                return np.nan
            return np.corrcoef(x[:-lag], x[lag:])[0, 1]
        
        result = returns.rolling(window=period).apply(autocorr, raw=True)
        return self._handle_nan(result, f"Autocorr_{period}_Lag{lag}")
    
    # =========================================================================
    # MOMENTUM ÖZELLİKLERİ
    # =========================================================================
    
    def price_momentum(self, data: pd.Series, period: int = 20) -> pd.Series:
        """
        Price Momentum
        
        Basit fiyat momentumu.
        """
        result = data - data.shift(period)
        return self._handle_nan(result, f"Momentum_{period}")
    
    def momentum_score(self, data: pd.Series, periods: List[int] = [5, 10, 20, 60]) -> pd.Series:
        """
        Composite Momentum Score
        
        Birden fazla dönemin momentum ortalaması.
        """
        momentums = []
        for p in periods:
            mom = self.cumulative_return(data, p)
            momentums.append(mom)
        
        result = pd.concat(momentums, axis=1).mean(axis=1)
        return self._handle_nan(result, "MomentumScore")
    
    def acceleration(self, data: pd.Series, period: int = 20) -> pd.Series:
        """
        Price Acceleration
        
        Momentumun değişim hızı (momentum'un momentumu).
        """
        momentum = self.price_momentum(data, period)
        result = momentum - momentum.shift(period)
        return self._handle_nan(result, f"Acceleration_{period}")
    
    def velocity(self, data: pd.Series, period: int = 5) -> pd.Series:
        """
        Price Velocity
        
        Fiyat değişim hızı.
        """
        result = data.diff(period) / period
        return self._handle_nan(result, f"Velocity_{period}")
    
    # =========================================================================
    # MEAN REVERSION ÖZELLİKLERİ
    # =========================================================================
    
    def distance_from_mean(self, data: pd.Series, period: int = 20) -> pd.Series:
        """
        Distance from Rolling Mean
        
        Fiyatın hareketli ortalamadan uzaklığı.
        """
        mean = data.rolling(window=period).mean()
        result = (data - mean) / mean * 100  # Yüzde olarak
        return self._handle_nan(result, f"DistFromMean_{period}")
    
    def distance_from_high(self, data: pd.Series, period: int = 20) -> pd.Series:
        """
        Distance from Rolling High
        
        Fiyatın dönem içi yüksek seviyeden uzaklığı.
        Drawdown proxy.
        """
        rolling_high = data.rolling(window=period).max()
        result = (data / rolling_high - 1) * 100
        return self._handle_nan(result, f"DistFromHigh_{period}")
    
    def distance_from_low(self, data: pd.Series, period: int = 20) -> pd.Series:
        """
        Distance from Rolling Low
        
        Fiyatın dönem içi düşük seviyeden uzaklığı.
        """
        rolling_low = data.rolling(window=period).min()
        result = (data / rolling_low - 1) * 100
        return self._handle_nan(result, f"DistFromLow_{period}")
    
    def mean_reversion_signal(self, data: pd.Series, period: int = 20, threshold: float = 2.0) -> pd.Series:
        """
        Mean Reversion Signal
        
        Z-score bazlı mean reversion sinyali.
        """
        zscore = self.rolling_zscore(data, period)
        result = pd.Series(0, index=data.index)
        result[zscore < -threshold] = 1  # Oversold - BUY signal
        result[zscore > threshold] = -1   # Overbought - SELL signal
        return self._handle_nan(result, f"MeanRevSignal_{period}")
    
    # =========================================================================
    # FİYAT PATTERN ÖZELLİKLERİ
    # =========================================================================
    
    def higher_high(self, high: pd.Series, period: int = 5) -> pd.Series:
        """
        Higher High Pattern
        
        Son N bar'da artan yüksek seviyeleri tespit et.
        """
        rolling_max = high.rolling(window=period).max()
        result = (high == rolling_max) & (high > high.shift(1))
        return self._handle_nan(result.astype(int), f"HigherHigh_{period}")
    
    def lower_low(self, low: pd.Series, period: int = 5) -> pd.Series:
        """
        Lower Low Pattern
        
        Son N bar'da azalan düşük seviyeleri tespit et.
        """
        rolling_min = low.rolling(window=period).min()
        result = (low == rolling_min) & (low < low.shift(1))
        return self._handle_nan(result.astype(int), f"LowerLow_{period}")
    
    def price_range(self, high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
        """
        Price Range
        
        Dönem içi fiyat aralığı.
        """
        highest = high.rolling(window=period).max()
        lowest = low.rolling(window=period).min()
        result = (highest - lowest) / lowest * 100
        return self._handle_nan(result, f"PriceRange_{period}")
    
    def range_position(self, close: pd.Series, high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
        """
        Range Position
        
        Fiyatın dönem içi aralıktaki konumu (0-1).
        """
        highest = high.rolling(window=period).max()
        lowest = low.rolling(window=period).min()
        result = (close - lowest) / (highest - lowest)
        return self._handle_nan(result, f"RangePos_{period}")
    
    # =========================================================================
    # MUM FORMASYONLARI
    # =========================================================================
    
    def candle_body(self, open_price: pd.Series, close: pd.Series) -> pd.Series:
        """
        Candle Body Size
        
        Mum gövde büyüklüğü.
        """
        result = abs(close - open_price)
        return self._handle_nan(result, "CandleBody")
    
    def candle_body_ratio(self, open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Candle Body Ratio
        
        Gövde / Toplam mum boyutu.
        """
        body = abs(close - open_price)
        total = high - low
        result = body / total
        result = result.replace([np.inf, -np.inf], 0)
        return self._handle_nan(result, "CandleBodyRatio")
    
    def upper_shadow(self, open_price: pd.Series, high: pd.Series, close: pd.Series) -> pd.Series:
        """
        Upper Shadow
        
        Üst fitil uzunluğu.
        """
        result = high - np.maximum(open_price, close)
        return self._handle_nan(result, "UpperShadow")
    
    def lower_shadow(self, open_price: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Lower Shadow
        
        Alt fitil uzunluğu.
        """
        result = np.minimum(open_price, close) - low
        return self._handle_nan(result, "LowerShadow")
    
    def is_bullish(self, open_price: pd.Series, close: pd.Series) -> pd.Series:
        """
        Is Bullish Candle
        
        Yeşil mum mu?
        """
        result = (close > open_price).astype(int)
        return self._handle_nan(result, "IsBullish")
    
    def doji_pattern(self, open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, threshold: float = 0.1) -> pd.Series:
        """
        Doji Pattern Detection
        
        Kararsızlık mumu - Açılış ve kapanış yakın.
        """
        body = abs(close - open_price)
        total_range = high - low
        body_ratio = body / total_range
        body_ratio = body_ratio.replace([np.inf, -np.inf], 0)
        
        result = (body_ratio < threshold).astype(int)
        return self._handle_nan(result, "Doji")
    
    def hammer_pattern(self, open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Hammer Pattern Detection
        
        Çekiç formasyonu - Potansiyel dip sinyali.
        """
        body = abs(close - open_price)
        total_range = high - low
        lower_shadow = np.minimum(open_price, close) - low
        upper_shadow = high - np.maximum(open_price, close)
        
        # Hammer conditions
        condition1 = lower_shadow > (2 * body)  # Uzun alt fitil
        condition2 = upper_shadow < (body * 0.3)  # Kısa üst fitil
        condition3 = body > 0  # Gövde var
        
        result = (condition1 & condition2 & condition3).astype(int)
        return self._handle_nan(result, "Hammer")
    
    def engulfing_pattern(self, open_price: pd.Series, close: pd.Series) -> pd.Series:
        """
        Engulfing Pattern Detection
        
        Yutan formasyon - Güçlü dönüş sinyali.
        Returns: 1 (Bullish), -1 (Bearish), 0 (None)
        """
        prev_body = abs(close.shift(1) - open_price.shift(1))
        curr_body = abs(close - open_price)
        
        prev_bullish = close.shift(1) > open_price.shift(1)
        curr_bullish = close > open_price
        
        # Bullish engulfing
        bullish_eng = (~prev_bullish) & curr_bullish & (curr_body > prev_body) & (close > open_price.shift(1)) & (open_price < close.shift(1))
        
        # Bearish engulfing
        bearish_eng = prev_bullish & (~curr_bullish) & (curr_body > prev_body) & (close < open_price.shift(1)) & (open_price > close.shift(1))
        
        result = pd.Series(0, index=close.index)
        result[bullish_eng] = 1
        result[bearish_eng] = -1
        
        return self._handle_nan(result, "Engulfing")
    
    def consecutive_candles(self, open_price: pd.Series, close: pd.Series, count: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Consecutive Bullish/Bearish Candles
        
        Ardışık yeşil/kırmızı mumları say.
        """
        is_bull = (close > open_price).astype(int)
        is_bear = (close < open_price).astype(int)
        
        # Count consecutive
        def count_consecutive(x):
            groups = (x != x.shift()).cumsum()
            return x.groupby(groups).cumsum()
        
        consecutive_bull = count_consecutive(is_bull)
        consecutive_bear = count_consecutive(is_bear)
        
        return (
            self._handle_nan(consecutive_bull, "ConsecBullish"),
            self._handle_nan(consecutive_bear, "ConsecBearish")
        )
    
    # =========================================================================
    # DESTEK / DİRENÇ
    # =========================================================================
    
    def pivot_points(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """
        Pivot Points (Daily)
        
        Klasik pivot noktaları - Destek ve direnç seviyeleri.
        """
        pivot = (high + low + close) / 3
        
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': self._handle_nan(pivot, "Pivot"),
            'r1': self._handle_nan(r1, "R1"),
            's1': self._handle_nan(s1, "S1"),
            'r2': self._handle_nan(r2, "R2"),
            's2': self._handle_nan(s2, "S2"),
            'r3': self._handle_nan(r3, "R3"),
            's3': self._handle_nan(s3, "S3")
        }
    
    def support_resistance_levels(self, high: pd.Series, low: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        """
        Dynamic Support/Resistance
        
        Dinamik destek ve direnç seviyeleri.
        """
        resistance = high.rolling(window=period).max()
        support = low.rolling(window=period).min()
        
        return (
            self._handle_nan(support, f"Support_{period}"),
            self._handle_nan(resistance, f"Resistance_{period}")
        )
    
    def distance_to_support(self, close: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
        """
        Distance to Support
        
        Fiyatın destek seviyesine uzaklığı.
        """
        support = low.rolling(window=period).min()
        result = (close - support) / support * 100
        return self._handle_nan(result, f"DistToSupport_{period}")
    
    def distance_to_resistance(self, close: pd.Series, high: pd.Series, period: int = 20) -> pd.Series:
        """
        Distance to Resistance
        
        Fiyatın direnç seviyesine uzaklığı.
        """
        resistance = high.rolling(window=period).max()
        result = (resistance - close) / close * 100
        return self._handle_nan(result, f"DistToResistance_{period}")
    
    # =========================================================================
    # GAP ANALİZİ
    # =========================================================================
    
    def gap_up(self, open_price: pd.Series, prev_high: pd.Series) -> pd.Series:
        """
        Gap Up Detection
        
        Yukarı boşluk.
        """
        result = (open_price > prev_high.shift(1)).astype(int)
        return self._handle_nan(result, "GapUp")
    
    def gap_down(self, open_price: pd.Series, prev_low: pd.Series) -> pd.Series:
        """
        Gap Down Detection
        
        Aşağı boşluk.
        """
        result = (open_price < prev_low.shift(1)).astype(int)
        return self._handle_nan(result, "GapDown")
    
    def gap_size(self, open_price: pd.Series, prev_close: pd.Series) -> pd.Series:
        """
        Gap Size
        
        Boşluk büyüklüğü (yüzde).
        """
        result = (open_price / prev_close.shift(1) - 1) * 100
        return self._handle_nan(result, "GapSize")
    
    # =========================================================================
    # TÜM ÖZELLİKLERİ HESAPLA
    # =========================================================================
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tüm fiyat özelliklerini hesapla.
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            pd.DataFrame: Tüm fiyat özellikleri
        """
        result = pd.DataFrame(index=df.index)
        
        # Extract columns
        open_price = df['open'] if 'open' in df.columns else df['close']
        high = df['high'] if 'high' in df.columns else df['close']
        low = df['low'] if 'low' in df.columns else df['close']
        close = df['close']
        
        # Returns
        for period in [1, 5, 10, 20]:
            result[f'Return_{period}'] = self.simple_return(close, period)
            result[f'LogReturn_{period}'] = self.log_return(close, period)
        
        for period in [5, 10, 20, 60]:
            result[f'CumReturn_{period}'] = self.cumulative_return(close, period)
        
        result['RealizedVol_20'] = self.realized_volatility(close, 20)
        result['ReturnDispersion_20'] = self.return_dispersion(close, 20)
        
        # Statistical features
        result['Skewness_20'] = self.rolling_skewness(close, 20)
        result['Kurtosis_20'] = self.rolling_kurtosis(close, 20)
        result['ZScore_20'] = self.rolling_zscore(close, 20)
        result['Percentile_100'] = self.rolling_percentile(close, 100)
        result['Autocorr_20'] = self.rolling_autocorrelation(close, 20, 1)
        
        # Momentum features
        for period in [5, 10, 20]:
            result[f'Momentum_{period}'] = self.price_momentum(close, period)
        result['MomentumScore'] = self.momentum_score(close)
        result['Acceleration_20'] = self.acceleration(close, 20)
        result['Velocity_5'] = self.velocity(close, 5)
        
        # Mean reversion features
        for period in [10, 20, 50]:
            result[f'DistFromMean_{period}'] = self.distance_from_mean(close, period)
        result['DistFromHigh_20'] = self.distance_from_high(close, 20)
        result['DistFromLow_20'] = self.distance_from_low(close, 20)
        result['MeanRevSignal_20'] = self.mean_reversion_signal(close, 20)
        
        # Pattern features
        result['PriceRange_20'] = self.price_range(high, low, 20)
        result['RangePosition_20'] = self.range_position(close, high, low, 20)
        result['HigherHigh_5'] = self.higher_high(high, 5)
        result['LowerLow_5'] = self.lower_low(low, 5)
        
        # Candle features
        result['CandleBody'] = self.candle_body(open_price, close)
        result['CandleBodyRatio'] = self.candle_body_ratio(open_price, high, low, close)
        result['UpperShadow'] = self.upper_shadow(open_price, high, close)
        result['LowerShadow'] = self.lower_shadow(open_price, low, close)
        result['IsBullish'] = self.is_bullish(open_price, close)
        result['Doji'] = self.doji_pattern(open_price, high, low, close)
        result['Hammer'] = self.hammer_pattern(open_price, high, low, close)
        result['Engulfing'] = self.engulfing_pattern(open_price, close)
        
        consec_bull, consec_bear = self.consecutive_candles(open_price, close)
        result['ConsecBullish'] = consec_bull
        result['ConsecBearish'] = consec_bear
        
        # Support/Resistance
        support, resistance = self.support_resistance_levels(high, low, 20)
        result['Support_20'] = support
        result['Resistance_20'] = resistance
        result['DistToSupport_20'] = self.distance_to_support(close, low, 20)
        result['DistToResistance_20'] = self.distance_to_resistance(close, high, 20)
        
        # Gap analysis
        result['GapUp'] = self.gap_up(open_price, high)
        result['GapDown'] = self.gap_down(open_price, low)
        result['GapSize'] = self.gap_size(open_price, close)
        
        return result
    
    def _handle_nan(self, data: pd.Series, name: str) -> pd.Series:
        """NaN handling"""
        result = data.copy()
        result.name = name
        
        if self.fillna:
            result = result.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Replace inf
        result = result.replace([np.inf, -np.inf], 0)
        
        return result


# Export
__all__ = ['PriceFeatures', 'CandlePattern']