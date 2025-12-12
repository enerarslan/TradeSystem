"""
Alternative Data Features
JPMorgan-Level Alternative Data Integration

Features:
- Sentiment analysis features
- News-based features
- Economic calendar features
- Social media metrics (simulated)
- Satellite/web traffic features (simulated)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class NewsEvent:
    """News event data"""
    timestamp: datetime
    symbol: str
    headline: str
    sentiment: float  # -1 to 1
    relevance: float  # 0 to 1
    source: str


@dataclass
class EconomicEvent:
    """Economic calendar event"""
    timestamp: datetime
    event_name: str
    country: str
    importance: int  # 1-3
    actual: Optional[float]
    forecast: Optional[float]
    previous: Optional[float]


class AlternativeDataFeatures:
    """
    Alternative data feature generator.

    In production, these would integrate with:
    - News APIs (Bloomberg, Reuters, RavenPack)
    - Social media APIs (Twitter, StockTwits)
    - Economic data APIs (FRED, Trading Economics)
    - Alternative data providers (Quandl, Orbital Insight)
    """

    def __init__(self):
        """Initialize AlternativeDataFeatures"""
        self._news_cache: Dict[str, List[NewsEvent]] = {}
        self._economic_calendar: List[EconomicEvent] = []

    # =========================================================================
    # SENTIMENT FEATURES
    # =========================================================================

    def generate_sentiment_features(
        self,
        df: pd.DataFrame,
        news_sentiment: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Generate sentiment-based features.

        In production, would use real sentiment data from:
        - News sentiment (RavenPack, Bloomberg)
        - Social sentiment (Twitter, StockTwits, Reddit)
        - Analyst sentiment (earnings call analysis)
        """
        features = pd.DataFrame(index=df.index)

        # Simulate sentiment if not provided
        if news_sentiment is None:
            # Create synthetic sentiment based on price action
            returns = df['close'].pct_change()

            # Sentiment tends to follow price with some lag
            news_sentiment = returns.rolling(5).mean() * 10  # Scale to -1 to 1 range
            news_sentiment = news_sentiment.clip(-1, 1)

        features['news_sentiment'] = news_sentiment

        # Sentiment moving averages
        for period in [5, 10, 20]:
            features[f'sentiment_ma_{period}'] = news_sentiment.rolling(period).mean()

        # Sentiment momentum
        features['sentiment_momentum'] = (
            features['sentiment_ma_5'] - features['sentiment_ma_20']
        )

        # Sentiment volatility
        features['sentiment_vol'] = news_sentiment.rolling(20).std()

        # Extreme sentiment indicators
        features['sentiment_extreme_pos'] = (news_sentiment > 0.7).astype(int)
        features['sentiment_extreme_neg'] = (news_sentiment < -0.7).astype(int)

        # Sentiment divergence from price
        price_direction = np.sign(df['close'].pct_change())
        sentiment_direction = np.sign(news_sentiment)
        features['sentiment_divergence'] = (price_direction != sentiment_direction).astype(int)

        return features

    def generate_social_features(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """
        Generate social media-based features.

        In production, would use:
        - Twitter API for tweet volume/sentiment
        - StockTwits API for trader sentiment
        - Reddit API for discussion metrics
        """
        features = pd.DataFrame(index=df.index)

        # Simulate social metrics based on volume and volatility
        volume_zscore = (df['volume'] - df['volume'].rolling(50).mean()) / df['volume'].rolling(50).std()
        volatility = df['close'].pct_change().rolling(20).std()

        # Social volume (mentions, tweets)
        features['social_volume'] = abs(volume_zscore) * (1 + volatility * 10)
        features['social_volume_ma'] = features['social_volume'].rolling(10).mean()

        # Social sentiment (simulated)
        price_momentum = df['close'].pct_change(5)
        features['social_sentiment'] = np.tanh(price_momentum * 20)  # Bounded -1 to 1

        # Bullish/Bearish ratio
        features['bull_bear_ratio'] = (features['social_sentiment'] + 1) / 2

        # Message velocity (rate of change)
        features['social_velocity'] = features['social_volume'].pct_change(5)

        # Unusual activity indicator
        features['unusual_social'] = (
            features['social_volume'] > features['social_volume_ma'] * 2
        ).astype(int)

        return features

    # =========================================================================
    # ECONOMIC CALENDAR FEATURES
    # =========================================================================

    def generate_economic_features(
        self,
        df: pd.DataFrame,
        economic_events: Optional[List[EconomicEvent]] = None
    ) -> pd.DataFrame:
        """
        Generate economic calendar-based features.

        Important events:
        - FOMC meetings
        - NFP reports
        - CPI/PPI releases
        - GDP announcements
        - Earnings releases
        """
        features = pd.DataFrame(index=df.index)

        # Days to next major event (simulated)
        # In production, would use actual calendar
        features['days_to_fomc'] = self._days_to_periodic_event(df.index, 45)  # ~6 weeks
        features['days_to_nfp'] = self._days_to_periodic_event(df.index, 30)  # Monthly
        features['days_to_cpi'] = self._days_to_periodic_event(df.index, 30)  # Monthly

        # Event proximity indicators
        features['fomc_week'] = (features['days_to_fomc'] <= 3).astype(int)
        features['nfp_day'] = (features['days_to_nfp'] <= 1).astype(int)

        # Economic surprise (simulated)
        # In production, would use actual vs forecast
        returns = df['close'].pct_change()
        features['economic_surprise'] = returns.rolling(5).mean() * 100

        # Economic uncertainty index (simulated VIX-like)
        volatility = returns.rolling(20).std() * np.sqrt(252 * 26)
        features['uncertainty_index'] = volatility.rolling(10).mean() * 100

        return features

    def _days_to_periodic_event(
        self,
        index: pd.DatetimeIndex,
        period_days: int
    ) -> pd.Series:
        """Calculate days until next periodic event"""
        result = pd.Series(index=index, dtype=float)

        for i, dt in enumerate(index):
            day_of_period = i % (period_days * 26)  # Convert to bars
            days_remaining = (period_days * 26 - day_of_period) / 26
            result.iloc[i] = days_remaining

        return result

    # =========================================================================
    # EARNINGS FEATURES
    # =========================================================================

    def generate_earnings_features(
        self,
        df: pd.DataFrame,
        earnings_dates: Optional[List[datetime]] = None
    ) -> pd.DataFrame:
        """
        Generate earnings-related features.

        Features around earnings:
        - Days to/from earnings
        - Pre-earnings drift
        - Post-earnings announcement drift
        - Earnings volatility regime
        """
        features = pd.DataFrame(index=df.index)

        # Simulate earnings dates if not provided (quarterly)
        if earnings_dates is None:
            earnings_dates = self._generate_quarterly_dates(df.index)

        # Days to next earnings
        features['days_to_earnings'] = self._days_to_events(df.index, earnings_dates)

        # Days since last earnings
        features['days_since_earnings'] = self._days_since_events(df.index, earnings_dates)

        # Earnings week indicator
        features['earnings_week'] = (
            (features['days_to_earnings'] <= 5) |
            (features['days_since_earnings'] <= 5)
        ).astype(int)

        # Pre-earnings drift (5-day return before earnings)
        returns = df['close'].pct_change()
        features['pre_earnings_drift'] = returns.rolling(5 * 26).sum()  # 5 days of 15-min bars
        features['pre_earnings_drift'] = features['pre_earnings_drift'].where(
            features['days_to_earnings'] <= 5, 0
        )

        # Post-earnings volatility
        volatility = returns.rolling(20).std()
        features['post_earnings_vol'] = volatility.where(
            features['days_since_earnings'] <= 10, np.nan
        ).ffill(limit=10 * 26)

        # Earnings surprise proxy (price jump)
        daily_return = returns.rolling(26).sum()  # Approximate daily return
        features['earnings_surprise'] = daily_return.where(
            features['days_since_earnings'] == 0, 0
        )

        return features

    def _generate_quarterly_dates(
        self,
        index: pd.DatetimeIndex
    ) -> List[datetime]:
        """Generate quarterly earnings dates"""
        dates = []
        start = index.min()
        end = index.max()

        # Approximate quarterly dates
        current = start + timedelta(days=45)  # ~6 weeks into quarter
        while current <= end:
            dates.append(current)
            current += timedelta(days=90)  # Quarterly

        return dates

    def _days_to_events(
        self,
        index: pd.DatetimeIndex,
        events: List[datetime]
    ) -> pd.Series:
        """Calculate days to next event"""
        result = pd.Series(index=index, dtype=float)

        for i, dt in enumerate(index):
            future_events = [e for e in events if e > dt]
            if future_events:
                next_event = min(future_events)
                days = (next_event - dt).days
            else:
                days = 90  # Default to ~1 quarter

            result.iloc[i] = days

        return result

    def _days_since_events(
        self,
        index: pd.DatetimeIndex,
        events: List[datetime]
    ) -> pd.Series:
        """Calculate days since last event"""
        result = pd.Series(index=index, dtype=float)

        for i, dt in enumerate(index):
            past_events = [e for e in events if e <= dt]
            if past_events:
                last_event = max(past_events)
                days = (dt - last_event).days
            else:
                days = 90  # Default

            result.iloc[i] = days

        return result

    # =========================================================================
    # OPTIONS-DERIVED FEATURES
    # =========================================================================

    def generate_options_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate options-derived features (simulated).

        In production, would use:
        - Put/Call ratio
        - Implied volatility surface
        - Options flow data
        - Gamma exposure
        """
        features = pd.DataFrame(index=df.index)

        # Simulated implied volatility based on realized vol
        realized_vol = df['close'].pct_change().rolling(20).std() * np.sqrt(252 * 26)

        # IV typically trades at premium to realized
        iv_premium = 0.1 + 0.1 * np.random.randn(len(df))  # 10% avg premium
        features['implied_vol'] = realized_vol * (1 + iv_premium)

        # Put/Call ratio (simulated)
        # High P/C ratio often indicates fear/hedging
        features['put_call_ratio'] = 0.7 + 0.3 * features['implied_vol'] / realized_vol

        # IV percentile rank
        features['iv_percentile'] = features['implied_vol'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1],
            raw=False
        )

        # IV skew (simulated)
        # Negative skew = puts more expensive (fear)
        features['iv_skew'] = -0.1 * features['put_call_ratio']

        # Gamma exposure proxy
        features['gamma_proxy'] = features['implied_vol'] * df['volume'] / df['volume'].rolling(20).mean()

        # Max pain estimate (simulated)
        features['max_pain_distance'] = (
            df['close'].rolling(20).mean() - df['close']
        ) / df['close']

        return features

    # =========================================================================
    # COMPOSITE ALTERNATIVE DATA
    # =========================================================================

    def generate_all_features(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN"
    ) -> pd.DataFrame:
        """
        Generate all alternative data features.

        Args:
            df: OHLCV DataFrame
            symbol: Stock symbol

        Returns:
            DataFrame with all alternative data features
        """
        features = pd.DataFrame(index=df.index)

        # Sentiment features
        sentiment = self.generate_sentiment_features(df)
        features = pd.concat([features, sentiment], axis=1)

        # Social features
        social = self.generate_social_features(df, symbol)
        features = pd.concat([features, social], axis=1)

        # Economic features
        economic = self.generate_economic_features(df)
        features = pd.concat([features, economic], axis=1)

        # Earnings features
        earnings = self.generate_earnings_features(df)
        features = pd.concat([features, earnings], axis=1)

        # Options features
        options = self.generate_options_features(df)
        features = pd.concat([features, options], axis=1)

        # Composite indicators
        features['alt_data_signal'] = self._calculate_composite_signal(features)

        return features

    def _calculate_composite_signal(self, features: pd.DataFrame) -> pd.Series:
        """Calculate composite alternative data signal"""
        signal = pd.Series(0.0, index=features.index)

        # Weight different components
        weights = {
            'news_sentiment': 0.25,
            'social_sentiment': 0.15,
            'put_call_ratio': -0.15,  # Negative = contrarian
            'iv_percentile': -0.10,  # High IV = caution
            'uncertainty_index': -0.10,
            'earnings_week': -0.05,  # Extra caution around earnings
        }

        for col, weight in weights.items():
            if col in features.columns:
                # Normalize to -1 to 1 range
                normalized = features[col]
                if normalized.std() > 0:
                    normalized = (normalized - normalized.mean()) / normalized.std()
                    normalized = np.tanh(normalized)  # Bound to -1 to 1

                signal += weight * normalized

        return signal
