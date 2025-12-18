"""
Feature Transformers for Preprocessing.

JPMorgan Institutional-Level Time Feature Engineering.

Includes:
- Cyclical Time Encoding (Sin/Cos transformation)
- Market Session Indicators
- Time to Market Events
- Special Calendar Flags (Options Expiration, FOMC)
"""

from typing import Optional, List, Dict, Any
from datetime import time, datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from loguru import logger


# US Market Hours (Eastern Time)
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
PRE_MARKET_START = time(4, 0)
AFTER_HOURS_END = time(20, 0)


class TimeCyclicalEncoder(BaseEstimator, TransformerMixin):
    """
    Institutional-grade time feature encoder.

    Encodes time features into cyclical sin/cos components and generates
    market-specific timing features for financial ML models.

    Features Generated:
    - Cyclical encodings: hour, minute, day_of_week, quarter (sin/cos)
    - Market session: pre-market, regular, after-hours indicator
    - Time to events: minutes to open/close
    - Session flags: first/last 30 minutes
    - Calendar flags: options expiration week, FOMC week

    Example:
        encoder = TimeCyclicalEncoder()
        encoded_df = encoder.fit_transform(df)
    """

    def __init__(
        self,
        time_col: str = 'timestamp',
        market_open: time = MARKET_OPEN,
        market_close: time = MARKET_CLOSE,
        include_fomc: bool = True,
        include_options_expiration: bool = True,
        timezone: str = 'America/New_York',
    ):
        """
        Initialize the time encoder.

        Args:
            time_col: Column name containing timestamps
            market_open: Market open time (default: 9:30 ET)
            market_close: Market close time (default: 16:00 ET)
            include_fomc: Include FOMC announcement week flag
            include_options_expiration: Include options expiration week flag
            timezone: Timezone for market hours calculation
        """
        self.time_col = time_col
        self.market_open = market_open
        self.market_close = market_close
        self.include_fomc = include_fomc
        self.include_options_expiration = include_options_expiration
        self.timezone = timezone

        self.periods = {
            'hour': 24.0,
            'minute': 60.0,
            'day_of_week': 7.0,
            'day_of_month': 31.0,
            'month': 12.0,
            'quarter': 4.0,
        }

        # FOMC meeting dates (approximate - typically 8 meetings per year)
        # These should be updated annually for production use
        self._fomc_weeks: List[datetime] = []
        self._options_expiration_weeks: List[datetime] = []

    def fit(self, X: pd.DataFrame, y=None) -> "TimeCyclicalEncoder":
        """Fit the encoder (no-op, stateless transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data with comprehensive time features.

        Args:
            X: DataFrame with OHLCV data

        Returns:
            DataFrame with all time features added
        """
        X_encoded = X.copy()

        # Get datetime series
        dt_series = self._get_datetime_series(X_encoded)
        if dt_series is None:
            logger.warning("No datetime data found, returning original DataFrame")
            return X_encoded

        # 1. Cyclical Time Encodings
        X_encoded = self._add_cyclical_features(X_encoded, dt_series)

        # 2. Market Session Indicators
        X_encoded = self._add_market_session_features(X_encoded, dt_series)

        # 3. Time to Market Events
        X_encoded = self._add_time_to_events(X_encoded, dt_series)

        # 4. Session Boundary Flags
        X_encoded = self._add_session_boundary_flags(X_encoded, dt_series)

        # 5. Calendar Event Flags
        if self.include_fomc or self.include_options_expiration:
            X_encoded = self._add_calendar_flags(X_encoded, dt_series)

        return X_encoded

    def _get_datetime_series(self, X: pd.DataFrame) -> Optional[pd.DatetimeIndex]:
        """Extract datetime series from DataFrame."""
        if self.time_col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[self.time_col]):
                return pd.DatetimeIndex(X[self.time_col])
            try:
                return pd.DatetimeIndex(pd.to_datetime(X[self.time_col]))
            except Exception:
                pass

        if isinstance(X.index, pd.DatetimeIndex):
            return X.index

        return None

    def _add_cyclical_features(
        self,
        X: pd.DataFrame,
        dt_series: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Add sin/cos cyclical encodings for time components."""
        # Hour of day (sin/cos)
        X['hour_sin'] = np.sin(2 * np.pi * dt_series.hour / self.periods['hour'])
        X['hour_cos'] = np.cos(2 * np.pi * dt_series.hour / self.periods['hour'])

        # Minute of hour (sin/cos) - NEW
        X['minute_sin'] = np.sin(2 * np.pi * dt_series.minute / self.periods['minute'])
        X['minute_cos'] = np.cos(2 * np.pi * dt_series.minute / self.periods['minute'])

        # Day of week (sin/cos)
        X['dow_sin'] = np.sin(2 * np.pi * dt_series.dayofweek / self.periods['day_of_week'])
        X['dow_cos'] = np.cos(2 * np.pi * dt_series.dayofweek / self.periods['day_of_week'])

        # Quarter of year (sin/cos) - NEW
        X['quarter_sin'] = np.sin(2 * np.pi * dt_series.quarter / self.periods['quarter'])
        X['quarter_cos'] = np.cos(2 * np.pi * dt_series.quarter / self.periods['quarter'])

        # Month of year (sin/cos)
        X['month_sin'] = np.sin(2 * np.pi * dt_series.month / self.periods['month'])
        X['month_cos'] = np.cos(2 * np.pi * dt_series.month / self.periods['month'])

        return X

    def _add_market_session_features(
        self,
        X: pd.DataFrame,
        dt_series: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Add market session indicator.

        Values:
        - 0: Pre-market (4:00 - 9:30 ET)
        - 1: Regular trading (9:30 - 16:00 ET)
        - 2: After-hours (16:00 - 20:00 ET)
        - -1: Closed (other times)
        """
        # Convert to time only for comparison
        times = pd.Series([t.time() for t in dt_series], index=X.index)

        # Create session indicator
        session = pd.Series(-1, index=X.index, dtype=np.int8)

        # Pre-market: 4:00 AM - 9:30 AM
        pre_market_mask = (times >= PRE_MARKET_START) & (times < self.market_open)
        session[pre_market_mask] = 0

        # Regular hours: 9:30 AM - 4:00 PM
        regular_mask = (times >= self.market_open) & (times < self.market_close)
        session[regular_mask] = 1

        # After-hours: 4:00 PM - 8:00 PM
        after_hours_mask = (times >= self.market_close) & (times < AFTER_HOURS_END)
        session[after_hours_mask] = 2

        X['market_session'] = session

        # Binary flag for regular trading hours
        X['is_regular_hours'] = (session == 1).astype(np.int8)

        return X

    def _add_time_to_events(
        self,
        X: pd.DataFrame,
        dt_series: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Add time-to-event features in minutes.

        Features:
        - time_to_open: Minutes until market open (negative after open)
        - time_to_close: Minutes until market close
        """
        # Convert time objects to minutes from midnight
        open_minutes = self.market_open.hour * 60 + self.market_open.minute
        close_minutes = self.market_close.hour * 60 + self.market_close.minute

        # Current time in minutes from midnight
        current_minutes = dt_series.hour * 60 + dt_series.minute

        # Time to open (negative after open)
        time_to_open = open_minutes - current_minutes
        X['time_to_open'] = time_to_open

        # Time to close
        time_to_close = close_minutes - current_minutes
        X['time_to_close'] = time_to_close

        return X

    def _add_session_boundary_flags(
        self,
        X: pd.DataFrame,
        dt_series: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Add flags for session boundaries where volatility is typically higher.

        Features:
        - first_30_minutes: 1 if within first 30 min of regular session
        - last_30_minutes: 1 if within last 30 min of regular session
        """
        # Convert to time for comparison
        times = pd.Series([t.time() for t in dt_series], index=X.index)

        # First 30 minutes of trading: 9:30 - 10:00
        first_30_end = time(10, 0)
        X['first_30_minutes'] = (
            (times >= self.market_open) & (times < first_30_end)
        ).astype(np.int8)

        # Last 30 minutes of trading: 15:30 - 16:00
        last_30_start = time(15, 30)
        X['last_30_minutes'] = (
            (times >= last_30_start) & (times < self.market_close)
        ).astype(np.int8)

        return X

    def _add_calendar_flags(
        self,
        X: pd.DataFrame,
        dt_series: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Add calendar event flags.

        Features:
        - options_expiration_week: 1 if within monthly options expiration week
        - fomc_week: 1 if within FOMC announcement week
        """
        # Get unique dates
        dates = pd.Series(dt_series.date, index=X.index)

        # Options expiration: Third Friday of each month
        if self.include_options_expiration:
            X['options_expiration_week'] = dates.apply(
                self._is_options_expiration_week
            ).astype(np.int8)

        # FOMC week: Approximate based on typical FOMC schedule
        if self.include_fomc:
            X['fomc_week'] = dates.apply(
                self._is_fomc_week
            ).astype(np.int8)

        return X

    def _is_options_expiration_week(self, date: datetime.date) -> bool:
        """
        Check if date is in the options expiration week.

        Options expiration is typically the third Friday of each month.
        We flag the entire week (Monday-Friday) of expiration.
        """
        # Find third Friday of the month
        first_day = date.replace(day=1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(weeks=2)

        # Get the week containing third Friday
        week_start = third_friday - timedelta(days=third_friday.weekday())
        week_end = week_start + timedelta(days=4)

        return week_start <= date <= week_end

    def _is_fomc_week(self, date: datetime.date) -> bool:
        """
        Check if date is in an FOMC announcement week.

        FOMC typically meets 8 times per year. This is an approximation
        based on typical meeting patterns. For production, use actual
        FOMC calendar data.

        Approximate schedule (2024-2025 pattern):
        - Late January
        - Mid-March
        - Early May
        - Mid-June
        - Late July
        - Mid-September
        - Early November
        - Mid-December
        """
        month = date.month
        day = date.day

        # Approximate FOMC meeting weeks by month and day range
        fomc_weeks = {
            1: (24, 31),   # Late January
            3: (13, 20),   # Mid-March
            5: (1, 8),     # Early May
            6: (11, 18),   # Mid-June
            7: (24, 31),   # Late July
            9: (17, 24),   # Mid-September
            11: (1, 8),    # Early November
            12: (11, 18),  # Mid-December
        }

        if month in fomc_weeks:
            start_day, end_day = fomc_weeks[month]
            return start_day <= day <= end_day

        return False

    def get_feature_names(self) -> List[str]:
        """Get list of all generated feature names."""
        features = [
            # Cyclical features
            'hour_sin', 'hour_cos',
            'minute_sin', 'minute_cos',
            'dow_sin', 'dow_cos',
            'quarter_sin', 'quarter_cos',
            'month_sin', 'month_cos',
            # Market session
            'market_session',
            'is_regular_hours',
            # Time to events
            'time_to_open',
            'time_to_close',
            # Session boundaries
            'first_30_minutes',
            'last_30_minutes',
        ]

        if self.include_options_expiration:
            features.append('options_expiration_week')
        if self.include_fomc:
            features.append('fomc_week')

        return features