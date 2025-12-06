"""
KURUMSAL HACİM ANALİZ ÖZELLİKLERİ
JPMorgan Market Microstructure Research Tarzı

Volume-based Features:
- Volume analysis (relative volume, unusual volume)
- Order flow analysis
- Accumulation/Distribution
- Institutional activity signals
- Volume-Price relationship
- Liquidity metrics

Bu modül hacim pattern'lerinden kurumsal aktivite sinyalleri çıkarır.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


class VolumeFeatures:
    """
    Hacim bazlı özellik mühendisliği.
    
    Kategoriler:
    1. Temel hacim analizi
    2. Relative volume (RVOL)
    3. Volume-Price analizi
    4. Accumulation/Distribution
    5. Order flow metrics
    6. Liquidity indicators
    """
    
    def __init__(self, fillna: bool = True):
        self.fillna = fillna
    
    # =========================================================================
    # TEMEL HACİM ANALİZİ
    # =========================================================================
    
    def volume_sma(self, volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Volume Simple Moving Average
        """
        result = volume.rolling(window=period, min_periods=1).mean()
        return self._handle_nan(result, f"VolSMA_{period}")
    
    def volume_ema(self, volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Volume Exponential Moving Average
        """
        result = volume.ewm(span=period, adjust=False, min_periods=1).mean()
        return self._handle_nan(result, f"VolEMA_{period}")
    
    def volume_std(self, volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Volume Standard Deviation
        """
        result = volume.rolling(window=period, min_periods=1).std()
        return self._handle_nan(result, f"VolSTD_{period}")
    
    def relative_volume(self, volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Relative Volume (RVOL)
        
        Mevcut hacim / Ortalama hacim.
        >1.5: Unusual volume (dikkat!)
        """
        avg_volume = self.volume_sma(volume, period)
        result = volume / avg_volume
        return self._handle_nan(result, f"RVOL_{period}")
    
    def volume_zscore(self, volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Volume Z-Score
        
        Hacmin ortalamadan sapması.
        """
        mean = volume.rolling(window=period).mean()
        std = volume.rolling(window=period).std()
        result = (volume - mean) / std
        return self._handle_nan(result, f"VolZScore_{period}")
    
    def unusual_volume(self, volume: pd.Series, period: int = 20, threshold: float = 2.0) -> pd.Series:
        """
        Unusual Volume Detection
        
        Z-score > threshold ise unusual.
        """
        zscore = self.volume_zscore(volume, period)
        result = (zscore > threshold).astype(int)
        return self._handle_nan(result, f"UnusualVol_{period}")
    
    def volume_trend(self, volume: pd.Series, fast: int = 5, slow: int = 20) -> pd.Series:
        """
        Volume Trend
        
        Hacim artıyor mu azalıyor mu?
        """
        fast_sma = self.volume_sma(volume, fast)
        slow_sma = self.volume_sma(volume, slow)
        result = (fast_sma / slow_sma) - 1
        return self._handle_nan(result, "VolTrend")
    
    def volume_momentum(self, volume: pd.Series, period: int = 10) -> pd.Series:
        """
        Volume Momentum
        
        Hacim değişim hızı.
        """
        result = volume.pct_change(periods=period)
        return self._handle_nan(result, f"VolMom_{period}")
    
    def volume_percentile(self, volume: pd.Series, period: int = 100) -> pd.Series:
        """
        Volume Percentile
        
        Mevcut hacmin dönem içindeki yüzdelik dilimi.
        """
        def percentile_rank(x):
            return (x.argsort().argsort()[-1] + 1) / len(x) * 100
        
        result = volume.rolling(window=period).apply(percentile_rank, raw=True)
        return self._handle_nan(result, f"VolPercentile_{period}")
    
    # =========================================================================
    # HACİM-FİYAT İLİŞKİSİ
    # =========================================================================
    
    def price_volume_trend(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Price Volume Trend (PVT)
        
        Hacim ağırlıklı fiyat trend.
        OBV'ye benzer ama gradual.
        """
        pct_change = close.pct_change()
        pvt = (pct_change * volume).cumsum()
        return self._handle_nan(pvt, "PVT")
    
    def volume_price_confirmation(self, close: pd.Series, volume: pd.Series, period: int = 5) -> pd.Series:
        """
        Volume-Price Confirmation
        
        Fiyat ve hacim aynı yönde mi hareket ediyor?
        1: Confirmed up
        -1: Confirmed down
        0: Divergence
        """
        price_up = close.diff(period) > 0
        vol_up = volume.diff(period) > 0
        
        result = pd.Series(0, index=close.index)
        result[(price_up) & (vol_up)] = 1  # Bullish confirmation
        result[(~price_up) & (vol_up)] = -1  # Bearish confirmation (selling pressure)
        
        return self._handle_nan(result, "VolPriceConfirm")
    
    def volume_weighted_price(self, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Volume Weighted Price (Rolling VWAP)
        """
        pv = close * volume
        result = pv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        return self._handle_nan(result, f"VWPrice_{period}")
    
    def price_volume_divergence(self, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """
        Price-Volume Divergence
        
        Fiyat yükselirken hacim düşüyor mu?
        Zayıflık sinyali olabilir.
        """
        # Normalize for comparison
        price_norm = (close - close.rolling(period).min()) / (close.rolling(period).max() - close.rolling(period).min())
        vol_norm = (volume - volume.rolling(period).min()) / (volume.rolling(period).max() - volume.rolling(period).min())
        
        divergence = price_norm - vol_norm
        return self._handle_nan(divergence, "PriceVolDivergence")
    
    def up_down_volume_ratio(self, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """
        Up Volume / Down Volume Ratio
        
        Alım hacmi / Satım hacmi.
        >1: Bullish pressure
        <1: Bearish pressure
        """
        price_up = close > close.shift(1)
        
        up_volume = volume.where(price_up, 0).rolling(window=period).sum()
        down_volume = volume.where(~price_up, 0).rolling(window=period).sum()
        
        result = up_volume / down_volume
        result = result.replace([np.inf, -np.inf], 10)  # Cap extreme values
        return self._handle_nan(result, f"UpDownVolRatio_{period}")
    
    # =========================================================================
    # ACCUMULATION / DISTRIBUTION
    # =========================================================================
    
    def accumulation_distribution(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Accumulation/Distribution Line
        
        Money flow yönünü gösterir.
        """
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        ad = (clv * volume).cumsum()
        return self._handle_nan(ad, "AD")
    
    def ad_oscillator(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, fast: int = 3, slow: int = 10) -> pd.Series:
        """
        A/D Oscillator
        
        A/D line'ın EMA crossover'ı.
        """
        ad = self.accumulation_distribution(high, low, close, volume)
        fast_ema = ad.ewm(span=fast, adjust=False).mean()
        slow_ema = ad.ewm(span=slow, adjust=False).mean()
        result = fast_ema - slow_ema
        return self._handle_nan(result, "ADOsc")
    
    def chaikin_money_flow(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Chaikin Money Flow (CMF)
        
        Para akış göstergesi (-1 to 1).
        >0: Buying pressure
        <0: Selling pressure
        """
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        
        result = (clv * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
        return self._handle_nan(result, f"CMF_{period}")
    
    def force_index(self, close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
        """
        Force Index
        
        Fiyat değişimi * Hacim.
        Trend gücünü ölçer.
        """
        fi = close.diff() * volume
        result = fi.ewm(span=period, adjust=False).mean()
        return self._handle_nan(result, f"ForceIndex_{period}")
    
    def ease_of_movement(self, high: pd.Series, low: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """
        Ease of Movement (EMV)
        
        Fiyatın hareket kolaylığı.
        Yüksek: Fiyat kolayca hareket ediyor
        Düşük: Direnç var
        """
        distance = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
        box_ratio = (volume / 100000000) / (high - low)
        
        emv = distance / box_ratio
        result = emv.rolling(window=period).mean()
        return self._handle_nan(result, f"EMV_{period}")
    
    # =========================================================================
    # ORDER FLOW METRİKLERİ
    # =========================================================================
    
    def buying_pressure(self, open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Buying Pressure (BP)
        
        Close'un bar range'deki konumu.
        """
        result = close - low
        return self._handle_nan(result, "BuyingPressure")
    
    def selling_pressure(self, open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Selling Pressure (SP)
        """
        result = high - close
        return self._handle_nan(result, "SellingPressure")
    
    def buy_sell_ratio(self, open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Buy/Sell Ratio
        
        Dönem içi alım/satım baskısı oranı.
        """
        bp = self.buying_pressure(open_price, high, low, close)
        sp = self.selling_pressure(open_price, high, low, close)
        
        bp_sum = bp.rolling(window=period).sum()
        sp_sum = sp.rolling(window=period).sum()
        
        result = bp_sum / sp_sum
        result = result.replace([np.inf, -np.inf], 1)
        return self._handle_nan(result, f"BuySellRatio_{period}")
    
    def volume_weighted_buying_pressure(self, open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Volume Weighted Buying Pressure
        
        Hacim ağırlıklı alım baskısı.
        """
        bp = self.buying_pressure(open_price, high, low, close)
        result = bp * volume
        return self._handle_nan(result, "VWBuyingPressure")
    
    def trade_intensity(self, volume: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """
        Trade Intensity
        
        Hacim / Fiyat aralığı.
        Yüksek değer = Agresif trading.
        """
        price_range = high - low
        price_range = price_range.replace(0, 0.01)  # Avoid division by zero
        result = volume / price_range
        return self._handle_nan(result, "TradeIntensity")
    
    def volume_efficiency(self, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """
        Volume Efficiency
        
        Fiyat değişimi / Hacim.
        Düşük hacimle büyük hareket = Efficient
        """
        price_change = abs(close.diff(period))
        total_volume = volume.rolling(window=period).sum()
        
        result = price_change / total_volume * 1000000  # Scale
        return self._handle_nan(result, f"VolEfficiency_{period}")
    
    # =========================================================================
    # LİKİDİTE METRİKLERİ
    # =========================================================================
    
    def amihud_illiquidity(self, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Amihud Illiquidity Ratio
        
        |Return| / Dollar Volume.
        Yüksek değer = Düşük likidite.
        """
        abs_return = close.pct_change().abs()
        dollar_volume = close * volume
        
        daily_illiq = abs_return / dollar_volume * 1e6
        result = daily_illiq.rolling(window=period).mean()
        return self._handle_nan(result, f"AmihudIlliq_{period}")
    
    def volume_clock(self, volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Volume Clock
        
        Kümülatif hacim / Ortalama hacim.
        Trading day progress by volume.
        """
        cum_vol = volume.cumsum()
        avg_daily_vol = volume.rolling(window=period).sum()
        
        result = cum_vol / avg_daily_vol
        return self._handle_nan(result, "VolClock")
    
    def dollar_volume(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Dollar Volume
        
        Toplam işlem değeri.
        """
        result = close * volume
        return self._handle_nan(result, "DollarVolume")
    
    def dollar_volume_sma(self, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Dollar Volume SMA
        """
        dollar_vol = self.dollar_volume(close, volume)
        result = dollar_vol.rolling(window=period).mean()
        return self._handle_nan(result, f"DollarVolSMA_{period}")
    
    # =========================================================================
    # TÜM ÖZELLİKLERİ HESAPLA
    # =========================================================================
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tüm hacim özelliklerini hesapla.
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            pd.DataFrame: Tüm hacim özellikleri
        """
        result = pd.DataFrame(index=df.index)
        
        # Extract columns
        open_price = df['open'] if 'open' in df.columns else df['close']
        high = df['high'] if 'high' in df.columns else df['close']
        low = df['low'] if 'low' in df.columns else df['close']
        close = df['close']
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
        
        # Basic volume analysis
        for period in [5, 10, 20, 50]:
            result[f'VolSMA_{period}'] = self.volume_sma(volume, period)
        
        result['VolSTD_20'] = self.volume_std(volume, 20)
        result['RVOL_20'] = self.relative_volume(volume, 20)
        result['VolZScore_20'] = self.volume_zscore(volume, 20)
        result['UnusualVol'] = self.unusual_volume(volume, 20)
        result['VolTrend'] = self.volume_trend(volume)
        result['VolMom_10'] = self.volume_momentum(volume, 10)
        result['VolPercentile'] = self.volume_percentile(volume, 100)
        
        # Volume-Price relationship
        result['PVT'] = self.price_volume_trend(close, volume)
        result['VolPriceConfirm'] = self.volume_price_confirmation(close, volume)
        result['VWPrice_20'] = self.volume_weighted_price(close, volume, 20)
        result['PriceVolDivergence'] = self.price_volume_divergence(close, volume)
        result['UpDownVolRatio'] = self.up_down_volume_ratio(close, volume)
        
        # Accumulation/Distribution
        result['AD'] = self.accumulation_distribution(high, low, close, volume)
        result['ADOsc'] = self.ad_oscillator(high, low, close, volume)
        result['CMF_20'] = self.chaikin_money_flow(high, low, close, volume, 20)
        result['ForceIndex_13'] = self.force_index(close, volume, 13)
        result['EMV_14'] = self.ease_of_movement(high, low, volume, 14)
        
        # Order flow
        result['BuyingPressure'] = self.buying_pressure(open_price, high, low, close)
        result['SellingPressure'] = self.selling_pressure(open_price, high, low, close)
        result['BuySellRatio'] = self.buy_sell_ratio(open_price, high, low, close)
        result['VWBuyingPressure'] = self.volume_weighted_buying_pressure(open_price, high, low, close, volume)
        result['TradeIntensity'] = self.trade_intensity(volume, high, low)
        result['VolEfficiency'] = self.volume_efficiency(close, volume)
        
        # Liquidity
        result['AmihudIlliq'] = self.amihud_illiquidity(close, volume)
        result['DollarVolume'] = self.dollar_volume(close, volume)
        result['DollarVolSMA_20'] = self.dollar_volume_sma(close, volume, 20)
        
        return result
    
    def _handle_nan(self, data: pd.Series, name: str) -> pd.Series:
        """NaN handling"""
        result = data.copy()
        result.name = name
        
        if self.fillna:
            result = result.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        result = result.replace([np.inf, -np.inf], 0)
        
        return result


# Export
__all__ = ['VolumeFeatures']