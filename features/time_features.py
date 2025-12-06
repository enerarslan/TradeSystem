"""
KURUMSAL ZAMAN ÖZELLİKLERİ
JPMorgan Quantitative Research Tarzı

Zaman Bazlı Features:
- Calendar features (hour, day, month, quarter)
- Market session features (US, Europe, Asia)
- Seasonality features (day of week effects, month effects)
- Time decay features
- Holiday effects
- Trading session characteristics

Bu modül zamansal pattern'leri yakalar.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time, timedelta
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class MarketSession:
    """Piyasa seansı tanımı"""
    name: str
    start_hour: int
    end_hour: int
    timezone: str


# Global market sessions
MARKET_SESSIONS = {
    'asia': MarketSession('Asia', 0, 8, 'Asia/Tokyo'),
    'europe': MarketSession('Europe', 8, 14, 'Europe/London'),
    'us': MarketSession('US', 14, 21, 'America/New_York'),
    'us_premarket': MarketSession('US_Premarket', 9, 13, 'America/New_York'),
    'us_market': MarketSession('US_Market', 13, 20, 'America/New_York'),
    'us_afterhours': MarketSession('US_Afterhours', 20, 24, 'America/New_York'),
}


class TimeFeatures:
    """
    Zaman bazlı özellik mühendisliği.
    
    Kategoriler:
    1. Takvim özellikleri (saat, gün, ay, çeyrek)
    2. Cyclical encoding (sin/cos)
    3. Piyasa seansı özellikleri
    4. Mevsimsellik (day of week, month effects)
    5. Tatil etkileri
    6. Trading özellikleri (volatility patterns)
    """
    
    def __init__(self, fillna: bool = True):
        self.fillna = fillna
        
        # US Market holidays (Basitleştirilmiş)
        self.us_holidays = [
            # 2024-2025 major holidays
            '2024-01-01', '2024-01-15', '2024-02-19', '2024-03-29',
            '2024-05-27', '2024-06-19', '2024-07-04', '2024-09-02',
            '2024-11-28', '2024-12-25',
            '2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18',
            '2025-05-26', '2025-06-19', '2025-07-04', '2025-09-01',
            '2025-11-27', '2025-12-25'
        ]
    
    # =========================================================================
    # TAKVİM ÖZELLİKLERİ
    # =========================================================================
    
    def hour_of_day(self, timestamp: pd.Series) -> pd.Series:
        """
        Hour of Day (0-23)
        """
        result = timestamp.dt.hour
        return self._handle_nan(result, "Hour")
    
    def minute_of_hour(self, timestamp: pd.Series) -> pd.Series:
        """
        Minute of Hour (0-59)
        """
        result = timestamp.dt.minute
        return self._handle_nan(result, "Minute")
    
    def day_of_week(self, timestamp: pd.Series) -> pd.Series:
        """
        Day of Week (0=Monday, 6=Sunday)
        """
        result = timestamp.dt.dayofweek
        return self._handle_nan(result, "DayOfWeek")
    
    def day_of_month(self, timestamp: pd.Series) -> pd.Series:
        """
        Day of Month (1-31)
        """
        result = timestamp.dt.day
        return self._handle_nan(result, "DayOfMonth")
    
    def day_of_year(self, timestamp: pd.Series) -> pd.Series:
        """
        Day of Year (1-366)
        """
        result = timestamp.dt.dayofyear
        return self._handle_nan(result, "DayOfYear")
    
    def week_of_year(self, timestamp: pd.Series) -> pd.Series:
        """
        Week of Year (1-53)
        """
        result = timestamp.dt.isocalendar().week
        return self._handle_nan(result, "WeekOfYear")
    
    def month(self, timestamp: pd.Series) -> pd.Series:
        """
        Month (1-12)
        """
        result = timestamp.dt.month
        return self._handle_nan(result, "Month")
    
    def quarter(self, timestamp: pd.Series) -> pd.Series:
        """
        Quarter (1-4)
        """
        result = timestamp.dt.quarter
        return self._handle_nan(result, "Quarter")
    
    def year(self, timestamp: pd.Series) -> pd.Series:
        """
        Year
        """
        result = timestamp.dt.year
        return self._handle_nan(result, "Year")
    
    def is_weekend(self, timestamp: pd.Series) -> pd.Series:
        """
        Is Weekend (Saturday or Sunday)
        """
        result = (timestamp.dt.dayofweek >= 5).astype(int)
        return self._handle_nan(result, "IsWeekend")
    
    def is_month_start(self, timestamp: pd.Series) -> pd.Series:
        """
        Is Month Start (First trading day of month)
        """
        result = timestamp.dt.is_month_start.astype(int)
        return self._handle_nan(result, "IsMonthStart")
    
    def is_month_end(self, timestamp: pd.Series) -> pd.Series:
        """
        Is Month End (Last trading day of month)
        """
        result = timestamp.dt.is_month_end.astype(int)
        return self._handle_nan(result, "IsMonthEnd")
    
    def is_quarter_start(self, timestamp: pd.Series) -> pd.Series:
        """
        Is Quarter Start
        """
        result = timestamp.dt.is_quarter_start.astype(int)
        return self._handle_nan(result, "IsQuarterStart")
    
    def is_quarter_end(self, timestamp: pd.Series) -> pd.Series:
        """
        Is Quarter End
        """
        result = timestamp.dt.is_quarter_end.astype(int)
        return self._handle_nan(result, "IsQuarterEnd")
    
    def is_year_start(self, timestamp: pd.Series) -> pd.Series:
        """
        Is Year Start
        """
        result = timestamp.dt.is_year_start.astype(int)
        return self._handle_nan(result, "IsYearStart")
    
    def is_year_end(self, timestamp: pd.Series) -> pd.Series:
        """
        Is Year End
        """
        result = timestamp.dt.is_year_end.astype(int)
        return self._handle_nan(result, "IsYearEnd")
    
    # =========================================================================
    # CYCLICAL ENCODING (Sin/Cos)
    # =========================================================================
    
    def cyclical_hour(self, timestamp: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Cyclical encoding of hour.
        
        Sin/Cos encoding ensures continuity (23:59 is close to 00:00).
        """
        hour = timestamp.dt.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        return (
            self._handle_nan(hour_sin, "Hour_Sin"),
            self._handle_nan(hour_cos, "Hour_Cos")
        )
    
    def cyclical_day_of_week(self, timestamp: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Cyclical encoding of day of week.
        """
        dow = timestamp.dt.dayofweek
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)
        
        return (
            self._handle_nan(dow_sin, "DayOfWeek_Sin"),
            self._handle_nan(dow_cos, "DayOfWeek_Cos")
        )
    
    def cyclical_day_of_month(self, timestamp: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Cyclical encoding of day of month.
        """
        dom = timestamp.dt.day
        dom_sin = np.sin(2 * np.pi * dom / 31)
        dom_cos = np.cos(2 * np.pi * dom / 31)
        
        return (
            self._handle_nan(dom_sin, "DayOfMonth_Sin"),
            self._handle_nan(dom_cos, "DayOfMonth_Cos")
        )
    
    def cyclical_month(self, timestamp: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Cyclical encoding of month.
        """
        month = timestamp.dt.month
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        return (
            self._handle_nan(month_sin, "Month_Sin"),
            self._handle_nan(month_cos, "Month_Cos")
        )
    
    def cyclical_minute(self, timestamp: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Cyclical encoding of minute within the trading day.
        """
        minutes = timestamp.dt.hour * 60 + timestamp.dt.minute
        max_minutes = 24 * 60
        
        min_sin = np.sin(2 * np.pi * minutes / max_minutes)
        min_cos = np.cos(2 * np.pi * minutes / max_minutes)
        
        return (
            self._handle_nan(min_sin, "Minute_Sin"),
            self._handle_nan(min_cos, "Minute_Cos")
        )
    
    # =========================================================================
    # PİYASA SEANS ÖZELLİKLERİ
    # =========================================================================
    
    def market_session(self, timestamp: pd.Series) -> pd.Series:
        """
        Current Market Session
        
        Returns: 'asia', 'europe', 'us', 'overnight'
        """
        hour = timestamp.dt.hour
        
        conditions = [
            (hour >= 0) & (hour < 8),    # Asia
            (hour >= 8) & (hour < 14),   # Europe
            (hour >= 14) & (hour < 21),  # US
            (hour >= 21) | (hour < 0)    # Overnight
        ]
        
        choices = ['asia', 'europe', 'us', 'overnight']
        
        result = pd.Series(np.select(conditions, choices, default='unknown'), index=timestamp.index)
        return result
    
    def is_us_session(self, timestamp: pd.Series) -> pd.Series:
        """
        Is US Market Session (9:30 AM - 4:00 PM ET)
        
        For 15-min bars: ~13:30 - 20:00 UTC
        """
        hour = timestamp.dt.hour
        result = ((hour >= 14) & (hour < 21)).astype(int)
        return self._handle_nan(result, "IsUSSession")
    
    def is_europe_session(self, timestamp: pd.Series) -> pd.Series:
        """
        Is European Session (8:00 AM - 4:30 PM GMT)
        """
        hour = timestamp.dt.hour
        result = ((hour >= 8) & (hour < 17)).astype(int)
        return self._handle_nan(result, "IsEuropeSession")
    
    def is_asia_session(self, timestamp: pd.Series) -> pd.Series:
        """
        Is Asian Session
        """
        hour = timestamp.dt.hour
        result = ((hour >= 0) & (hour < 8)).astype(int)
        return self._handle_nan(result, "IsAsiaSession")
    
    def session_overlap(self, timestamp: pd.Series) -> pd.Series:
        """
        Session Overlap Detection
        
        Overlapping sessions typically have higher volatility.
        Returns: Number of overlapping sessions
        """
        hour = timestamp.dt.hour
        
        asia = (hour >= 0) & (hour < 8)
        europe = (hour >= 8) & (hour < 17)
        us = (hour >= 14) & (hour < 21)
        
        overlap_count = asia.astype(int) + europe.astype(int) + us.astype(int)
        return self._handle_nan(overlap_count, "SessionOverlap")
    
    def time_to_market_open(self, timestamp: pd.Series) -> pd.Series:
        """
        Minutes to US Market Open
        
        US Market opens at 9:30 AM ET (13:30 UTC typically)
        """
        current_minutes = timestamp.dt.hour * 60 + timestamp.dt.minute
        market_open_minutes = 14 * 60 + 30  # 14:30 UTC
        
        diff = market_open_minutes - current_minutes
        diff = diff.where(diff >= 0, diff + 24 * 60)  # Wrap around
        
        return self._handle_nan(diff, "MinutesToOpen")
    
    def time_to_market_close(self, timestamp: pd.Series) -> pd.Series:
        """
        Minutes to US Market Close
        
        US Market closes at 4:00 PM ET (21:00 UTC typically)
        """
        current_minutes = timestamp.dt.hour * 60 + timestamp.dt.minute
        market_close_minutes = 21 * 60  # 21:00 UTC
        
        diff = market_close_minutes - current_minutes
        diff = diff.where(diff >= 0, diff + 24 * 60)
        
        return self._handle_nan(diff, "MinutesToClose")
    
    def trading_day_progress(self, timestamp: pd.Series) -> pd.Series:
        """
        Trading Day Progress (0-1)
        
        Ne kadar trading günü geçti?
        """
        current_minutes = timestamp.dt.hour * 60 + timestamp.dt.minute
        
        # US trading hours: 14:30 - 21:00 UTC (390 minutes)
        market_open = 14 * 60 + 30
        market_close = 21 * 60
        total_minutes = market_close - market_open
        
        progress = (current_minutes - market_open) / total_minutes
        progress = progress.clip(0, 1)
        
        return self._handle_nan(progress, "TradingDayProgress")
    
    # =========================================================================
    # MEVSİMSELLİK ÖZELLİKLERİ
    # =========================================================================
    
    def is_monday(self, timestamp: pd.Series) -> pd.Series:
        """Is Monday"""
        result = (timestamp.dt.dayofweek == 0).astype(int)
        return self._handle_nan(result, "IsMonday")
    
    def is_friday(self, timestamp: pd.Series) -> pd.Series:
        """Is Friday"""
        result = (timestamp.dt.dayofweek == 4).astype(int)
        return self._handle_nan(result, "IsFriday")
    
    def is_first_hour(self, timestamp: pd.Series) -> pd.Series:
        """
        Is First Hour of US Session
        
        Genellikle yüksek volatilite.
        """
        hour = timestamp.dt.hour
        result = (hour == 14).astype(int)  # 14:00-15:00 UTC
        return self._handle_nan(result, "IsFirstHour")
    
    def is_last_hour(self, timestamp: pd.Series) -> pd.Series:
        """
        Is Last Hour of US Session
        
        Genellikle yüksek hacim.
        """
        hour = timestamp.dt.hour
        result = (hour == 20).astype(int)  # 20:00-21:00 UTC
        return self._handle_nan(result, "IsLastHour")
    
    def is_lunch_hour(self, timestamp: pd.Series) -> pd.Series:
        """
        Is US Lunch Hour
        
        Genellikle düşük hacim.
        """
        hour = timestamp.dt.hour
        result = ((hour >= 16) & (hour < 18)).astype(int)
        return self._handle_nan(result, "IsLunchHour")
    
    def is_witching_day(self, timestamp: pd.Series) -> pd.Series:
        """
        Triple Witching Day
        
        Options/Futures expiration - yüksek volatilite.
        3. Cuma of March, June, September, December
        """
        is_friday = timestamp.dt.dayofweek == 4
        day = timestamp.dt.day
        month = timestamp.dt.month
        
        is_third_friday = (day >= 15) & (day <= 21) & is_friday
        is_witching_month = month.isin([3, 6, 9, 12])
        
        result = (is_third_friday & is_witching_month).astype(int)
        return self._handle_nan(result, "IsWitchingDay")
    
    def is_turn_of_month(self, timestamp: pd.Series) -> pd.Series:
        """
        Turn of Month Effect
        
        Son 3 gün ve ilk 3 gün - window dressing.
        """
        day = timestamp.dt.day
        days_in_month = timestamp.dt.days_in_month
        
        result = ((day <= 3) | (day >= days_in_month - 2)).astype(int)
        return self._handle_nan(result, "IsTurnOfMonth")
    
    def is_january(self, timestamp: pd.Series) -> pd.Series:
        """
        January Effect
        
        Ocak ayı genellikle pozitif.
        """
        result = (timestamp.dt.month == 1).astype(int)
        return self._handle_nan(result, "IsJanuary")
    
    def sell_in_may(self, timestamp: pd.Series) -> pd.Series:
        """
        Sell in May Effect
        
        May-October genellikle zayıf.
        """
        month = timestamp.dt.month
        result = ((month >= 5) & (month <= 10)).astype(int)
        return self._handle_nan(result, "SellInMay")
    
    # =========================================================================
    # TATİL ETKİLERİ
    # =========================================================================
    
    def is_holiday(self, timestamp: pd.Series) -> pd.Series:
        """
        Is US Market Holiday
        """
        date_str = timestamp.dt.strftime('%Y-%m-%d')
        result = date_str.isin(self.us_holidays).astype(int)
        return self._handle_nan(result, "IsHoliday")
    
    def days_to_holiday(self, timestamp: pd.Series) -> pd.Series:
        """
        Days to Next Holiday
        
        Tatil öncesi ve sonrası volatilite pattern'leri.
        """
        holidays = pd.to_datetime(self.us_holidays)
        
        def calc_days_to_holiday(ts):
            future_holidays = holidays[holidays > ts]
            if len(future_holidays) > 0:
                return (future_holidays[0] - ts).days
            return 365  # Fallback
        
        result = timestamp.apply(calc_days_to_holiday)
        return self._handle_nan(result, "DaysToHoliday")
    
    def days_since_holiday(self, timestamp: pd.Series) -> pd.Series:
        """
        Days Since Last Holiday
        """
        holidays = pd.to_datetime(self.us_holidays)
        
        def calc_days_since_holiday(ts):
            past_holidays = holidays[holidays < ts]
            if len(past_holidays) > 0:
                return (ts - past_holidays[-1]).days
            return 365
        
        result = timestamp.apply(calc_days_since_holiday)
        return self._handle_nan(result, "DaysSinceHoliday")
    
    def is_pre_holiday(self, timestamp: pd.Series) -> pd.Series:
        """
        Is Day Before Holiday
        """
        days_to = self.days_to_holiday(timestamp)
        result = (days_to == 1).astype(int)
        return self._handle_nan(result, "IsPreHoliday")
    
    def is_post_holiday(self, timestamp: pd.Series) -> pd.Series:
        """
        Is Day After Holiday
        """
        days_since = self.days_since_holiday(timestamp)
        result = (days_since == 1).astype(int)
        return self._handle_nan(result, "IsPostHoliday")
    
    # =========================================================================
    # TİME DECAY ÖZELLİKLERİ
    # =========================================================================
    
    def bars_since_start(self, timestamp: pd.Series) -> pd.Series:
        """
        Bars Since Dataset Start
        
        Trend analizi için zaman indexi.
        """
        result = pd.Series(range(len(timestamp)), index=timestamp.index)
        return self._handle_nan(result, "BarsSinceStart")
    
    def days_since_start(self, timestamp: pd.Series) -> pd.Series:
        """
        Trading Days Since Dataset Start
        """
        start_date = timestamp.min()
        result = (timestamp - start_date).dt.days
        return self._handle_nan(result, "DaysSinceStart")
    
    def time_weight(self, timestamp: pd.Series, decay: float = 0.99) -> pd.Series:
        """
        Time Decay Weight
        
        Son verilere daha yüksek ağırlık.
        """
        bars = self.bars_since_start(timestamp)
        max_bars = bars.max()
        result = decay ** (max_bars - bars)
        return self._handle_nan(result, "TimeWeight")
    
    # =========================================================================
    # TÜM ÖZELLİKLERİ HESAPLA
    # =========================================================================
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tüm zaman özelliklerini hesapla.
        
        Args:
            df: DataFrame with DatetimeIndex or 'timestamp' column
        
        Returns:
            pd.DataFrame: Tüm zaman özellikleri
        """
        result = pd.DataFrame(index=df.index)
        
        # Get timestamp
        if isinstance(df.index, pd.DatetimeIndex):
            timestamp = pd.Series(df.index, index=df.index)
        elif 'timestamp' in df.columns:
            timestamp = pd.to_datetime(df['timestamp'])
        else:
            raise ValueError("DataFrame must have DatetimeIndex or 'timestamp' column")
        
        # Basic calendar features
        result['Hour'] = self.hour_of_day(timestamp)
        result['Minute'] = self.minute_of_hour(timestamp)
        result['DayOfWeek'] = self.day_of_week(timestamp)
        result['DayOfMonth'] = self.day_of_month(timestamp)
        result['DayOfYear'] = self.day_of_year(timestamp)
        result['WeekOfYear'] = self.week_of_year(timestamp)
        result['Month'] = self.month(timestamp)
        result['Quarter'] = self.quarter(timestamp)
        
        # Boolean calendar features
        result['IsWeekend'] = self.is_weekend(timestamp)
        result['IsMonthStart'] = self.is_month_start(timestamp)
        result['IsMonthEnd'] = self.is_month_end(timestamp)
        result['IsQuarterStart'] = self.is_quarter_start(timestamp)
        result['IsQuarterEnd'] = self.is_quarter_end(timestamp)
        
        # Cyclical encoding
        hour_sin, hour_cos = self.cyclical_hour(timestamp)
        result['Hour_Sin'] = hour_sin
        result['Hour_Cos'] = hour_cos
        
        dow_sin, dow_cos = self.cyclical_day_of_week(timestamp)
        result['DayOfWeek_Sin'] = dow_sin
        result['DayOfWeek_Cos'] = dow_cos
        
        dom_sin, dom_cos = self.cyclical_day_of_month(timestamp)
        result['DayOfMonth_Sin'] = dom_sin
        result['DayOfMonth_Cos'] = dom_cos
        
        month_sin, month_cos = self.cyclical_month(timestamp)
        result['Month_Sin'] = month_sin
        result['Month_Cos'] = month_cos
        
        # Market session features
        result['IsUSSession'] = self.is_us_session(timestamp)
        result['IsEuropeSession'] = self.is_europe_session(timestamp)
        result['IsAsiaSession'] = self.is_asia_session(timestamp)
        result['SessionOverlap'] = self.session_overlap(timestamp)
        result['TradingDayProgress'] = self.trading_day_progress(timestamp)
        
        # Seasonality features
        result['IsMonday'] = self.is_monday(timestamp)
        result['IsFriday'] = self.is_friday(timestamp)
        result['IsFirstHour'] = self.is_first_hour(timestamp)
        result['IsLastHour'] = self.is_last_hour(timestamp)
        result['IsLunchHour'] = self.is_lunch_hour(timestamp)
        result['IsTurnOfMonth'] = self.is_turn_of_month(timestamp)
        result['IsJanuary'] = self.is_january(timestamp)
        result['SellInMay'] = self.sell_in_may(timestamp)
        
        # Holiday features
        result['IsHoliday'] = self.is_holiday(timestamp)
        result['IsPreHoliday'] = self.is_pre_holiday(timestamp)
        result['IsPostHoliday'] = self.is_post_holiday(timestamp)
        
        # Time decay
        result['BarsSinceStart'] = self.bars_since_start(timestamp)
        result['TimeWeight'] = self.time_weight(timestamp)
        
        return result
    
    def _handle_nan(self, data: pd.Series, name: str) -> pd.Series:
        """NaN handling"""
        result = data.copy()
        result.name = name
        
        if self.fillna:
            result = result.fillna(0)
        
        return result


# Export
__all__ = ['TimeFeatures', 'MarketSession', 'MARKET_SESSIONS']