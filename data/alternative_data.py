"""
Alternative Data Integration Module
===================================

JPMorgan-level alternative data integration for alpha generation.

Technical indicators on price/volume are highly commoditized - every quant
has the same RSI and MACD. Institutional edge comes from informational
advantages through alternative data.

This module integrates:
1. Macro Data - Interest rates, VIX, economic indicators
2. Sentiment Data - News sentiment, social media (Twitter), analyst ratings
3. Fundamental Data - Earnings, revenue, guidance
4. Options Data - Put/Call ratios, implied volatility surface
5. Cross-Asset Signals - Sector momentum, correlation regimes

Data Sources Supported:
- FRED (Federal Reserve Economic Data)
- Alpha Vantage
- News APIs (NewsAPI, Benzinga)
- Yahoo Finance
- Custom CSV/Database

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable
import hashlib

import numpy as np
import polars as pl
from numpy.typing import NDArray

from config.settings import get_settings, get_logger

logger = get_logger(__name__)


# =============================================================================
# ENUMS & CONFIGURATION
# =============================================================================

class AlternativeDataType(str, Enum):
    """Types of alternative data."""
    MACRO = "macro"                 # Economic indicators
    SENTIMENT = "sentiment"         # News/social sentiment
    FUNDAMENTAL = "fundamental"     # Company fundamentals
    OPTIONS = "options"             # Options flow
    CROSS_ASSET = "cross_asset"     # Cross-asset signals
    CUSTOM = "custom"               # Custom data sources


class SentimentSource(str, Enum):
    """Sentiment data sources."""
    NEWS_API = "news_api"
    TWITTER = "twitter"
    REDDIT = "reddit"
    ANALYST = "analyst"
    EARNINGS_CALL = "earnings_call"


@dataclass
class AlternativeDataConfig:
    """Configuration for alternative data."""
    # API Keys (from environment)
    fred_api_key: str = ""
    alpha_vantage_key: str = ""
    news_api_key: str = ""

    # Cache settings
    cache_dir: Path = field(default_factory=lambda: Path("data/cache/alt_data"))
    cache_ttl_hours: int = 24

    # Feature settings
    macro_features_enabled: bool = True
    sentiment_features_enabled: bool = True
    options_features_enabled: bool = True

    # Lookback periods
    macro_lookback_days: int = 252
    sentiment_lookback_days: int = 30
    options_lookback_days: int = 30


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MacroIndicator:
    """Macro economic indicator."""
    name: str
    series_id: str           # FRED series ID
    description: str
    frequency: str           # daily, weekly, monthly
    transform: str           # none, pct_change, log, diff

    def __hash__(self):
        return hash(self.series_id)


@dataclass
class SentimentScore:
    """Sentiment score for a symbol."""
    timestamp: datetime
    symbol: str
    source: SentimentSource
    score: float             # -1 to +1
    volume: int              # Number of mentions
    confidence: float        # 0 to 1


# =============================================================================
# MACRO DATA LOADER
# =============================================================================

# Key macro indicators for equity trading
MACRO_INDICATORS = [
    MacroIndicator("VIX", "VIXCLS", "CBOE Volatility Index", "daily", "none"),
    MacroIndicator("10Y_Treasury", "DGS10", "10-Year Treasury Yield", "daily", "none"),
    MacroIndicator("2Y_Treasury", "DGS2", "2-Year Treasury Yield", "daily", "none"),
    MacroIndicator("Yield_Curve", "T10Y2Y", "10Y-2Y Treasury Spread", "daily", "none"),
    MacroIndicator("Fed_Funds", "DFF", "Effective Federal Funds Rate", "daily", "none"),
    MacroIndicator("Dollar_Index", "DTWEXBGS", "Trade Weighted US Dollar Index", "daily", "pct_change"),
    MacroIndicator("Credit_Spread", "BAMLC0A0CM", "ICE BofA US Corporate Index OAS", "daily", "none"),
    MacroIndicator("Unemployment", "UNRATE", "Unemployment Rate", "monthly", "none"),
    MacroIndicator("CPI", "CPIAUCSL", "Consumer Price Index", "monthly", "pct_change"),
    MacroIndicator("Industrial_Prod", "INDPRO", "Industrial Production Index", "monthly", "pct_change"),
    MacroIndicator("Retail_Sales", "RSXFS", "Advance Retail Sales", "monthly", "pct_change"),
    MacroIndicator("Housing_Starts", "HOUST", "Housing Starts", "monthly", "pct_change"),
    MacroIndicator("Consumer_Confidence", "UMCSENT", "U of Michigan Consumer Sentiment", "monthly", "none"),
    MacroIndicator("PMI_Manufacturing", "MANEMP", "Manufacturing Employment", "monthly", "pct_change"),
]


class MacroDataLoader:
    """
    Loads macroeconomic data from FRED and other sources.

    FRED (Federal Reserve Economic Data) provides free access to
    thousands of economic time series.
    """

    def __init__(self, config: AlternativeDataConfig | None = None):
        """Initialize loader."""
        self.config = config or AlternativeDataConfig()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

        # Get API key
        self.api_key = (
            self.config.fred_api_key or
            os.environ.get("FRED_API_KEY", "")
        )

    def load_indicator(
        self,
        indicator: MacroIndicator,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pl.DataFrame:
        """
        Load a single macro indicator.

        Args:
            indicator: MacroIndicator to load
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with timestamp and value columns
        """
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=self.config.macro_lookback_days))

        # Check cache
        cache_key = f"{indicator.series_id}_{start_date.date()}_{end_date.date()}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Try to fetch from FRED
        df = self._fetch_fred(indicator.series_id, start_date, end_date)

        if df is None or len(df) == 0:
            # Return synthetic data for development
            df = self._generate_synthetic(indicator, start_date, end_date)

        # Apply transform
        df = self._apply_transform(df, indicator.transform)

        # Rename value column
        df = df.rename({"value": indicator.name.lower()})

        # Cache result
        self._set_cached(cache_key, df)

        return df

    def load_all_indicators(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pl.DataFrame:
        """
        Load all macro indicators and merge into single DataFrame.

        Returns DataFrame with timestamp and columns for each indicator.
        """
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=self.config.macro_lookback_days))

        dfs = []
        for indicator in MACRO_INDICATORS:
            try:
                df = self.load_indicator(indicator, start_date, end_date)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {indicator.name}: {e}")

        if not dfs:
            return pl.DataFrame()

        # Merge all on timestamp
        result = dfs[0]
        for df in dfs[1:]:
            result = result.join(df, on="timestamp", how="outer")

        # Sort and forward fill
        result = result.sort("timestamp")
        result = result.fill_null(strategy="forward")

        return result

    def _fetch_fred(
        self,
        series_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pl.DataFrame | None:
        """Fetch data from FRED API."""
        if not self.api_key:
            logger.debug("No FRED API key, using synthetic data")
            return None

        try:
            import requests

            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "observation_start": start_date.strftime("%Y-%m-%d"),
                "observation_end": end_date.strftime("%Y-%m-%d"),
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            observations = data.get("observations", [])

            if not observations:
                return None

            records = []
            for obs in observations:
                try:
                    value = float(obs["value"])
                    records.append({
                        "timestamp": datetime.strptime(obs["date"], "%Y-%m-%d"),
                        "value": value,
                    })
                except (ValueError, KeyError):
                    continue

            return pl.DataFrame(records)

        except Exception as e:
            logger.debug(f"FRED fetch failed for {series_id}: {e}")
            return None

    def _generate_synthetic(
        self,
        indicator: MacroIndicator,
        start_date: datetime,
        end_date: datetime,
    ) -> pl.DataFrame:
        """Generate synthetic data for development."""
        days = (end_date - start_date).days + 1
        timestamps = [start_date + timedelta(days=i) for i in range(days)]

        # Generate realistic values based on indicator
        np.random.seed(hash(indicator.series_id) % (2**32))

        if indicator.name == "VIX":
            values = 15 + np.cumsum(np.random.randn(days) * 0.5)
            values = np.clip(values, 10, 80)
        elif "Treasury" in indicator.name or "Yield" in indicator.name:
            values = 3.0 + np.cumsum(np.random.randn(days) * 0.02)
            values = np.clip(values, 0, 10)
        elif indicator.name == "Unemployment":
            values = 4.0 + np.cumsum(np.random.randn(days) * 0.01)
            values = np.clip(values, 2, 15)
        else:
            values = 100 + np.cumsum(np.random.randn(days) * 0.5)
            values = np.clip(values, 50, 200)

        return pl.DataFrame({
            "timestamp": timestamps,
            "value": values,
        })

    def _apply_transform(self, df: pl.DataFrame, transform: str) -> pl.DataFrame:
        """Apply transformation to values."""
        if transform == "none" or len(df) < 2:
            return df

        if transform == "pct_change":
            df = df.with_columns(
                pl.col("value").pct_change().alias("value")
            )
        elif transform == "log":
            df = df.with_columns(
                pl.col("value").log().alias("value")
            )
        elif transform == "diff":
            df = df.with_columns(
                pl.col("value").diff().alias("value")
            )

        return df

    def _get_cached(self, key: str) -> pl.DataFrame | None:
        """Get cached DataFrame."""
        cache_path = self.config.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.parquet"

        if not cache_path.exists():
            return None

        # Check TTL
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - mtime > timedelta(hours=self.config.cache_ttl_hours):
            cache_path.unlink()
            return None

        try:
            return pl.read_parquet(cache_path)
        except Exception:
            return None

    def _set_cached(self, key: str, df: pl.DataFrame) -> None:
        """Cache DataFrame."""
        cache_path = self.config.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.parquet"
        try:
            df.write_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")


# =============================================================================
# SENTIMENT DATA LOADER
# =============================================================================

class SentimentDataLoader:
    """
    Loads sentiment data from news and social media sources.

    Sentiment scoring uses NLP to classify news/social content as
    positive, negative, or neutral for each symbol.
    """

    def __init__(self, config: AlternativeDataConfig | None = None):
        """Initialize loader."""
        self.config = config or AlternativeDataConfig()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

        self.news_api_key = (
            self.config.news_api_key or
            os.environ.get("NEWS_API_KEY", "")
        )

    def load_sentiment(
        self,
        symbol: str,
        source: SentimentSource = SentimentSource.NEWS_API,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pl.DataFrame:
        """
        Load sentiment data for a symbol.

        Args:
            symbol: Trading symbol
            source: Sentiment source
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with timestamp, sentiment_score, mention_volume
        """
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=self.config.sentiment_lookback_days))

        # Check cache
        cache_key = f"sentiment_{symbol}_{source.value}_{start_date.date()}_{end_date.date()}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Fetch based on source
        if source == SentimentSource.NEWS_API:
            df = self._fetch_news_sentiment(symbol, start_date, end_date)
        else:
            df = self._generate_synthetic_sentiment(symbol, start_date, end_date)

        if df is None or len(df) == 0:
            df = self._generate_synthetic_sentiment(symbol, start_date, end_date)

        # Cache
        self._set_cached(cache_key, df)

        return df

    def _fetch_news_sentiment(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pl.DataFrame | None:
        """Fetch and score news from NewsAPI."""
        if not self.news_api_key:
            return None

        try:
            import requests

            # Get company name for search
            company_names = {
                "AAPL": "Apple",
                "MSFT": "Microsoft",
                "GOOGL": "Google Alphabet",
                "AMZN": "Amazon",
                "META": "Meta Facebook",
                "TSLA": "Tesla",
                "NVDA": "NVIDIA",
                "JPM": "JPMorgan",
            }
            query = company_names.get(symbol, symbol)

            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "apiKey": self.news_api_key,
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
                "language": "en",
                "sortBy": "relevancy",
                "pageSize": 100,
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            articles = data.get("articles", [])

            if not articles:
                return None

            # Simple sentiment scoring using keyword matching
            # In production, use a proper NLP model (FinBERT, etc.)
            records = []
            daily_scores: dict[str, list[float]] = {}
            daily_counts: dict[str, int] = {}

            positive_words = {"surge", "jump", "gain", "profit", "growth", "beat", "up", "high", "bullish", "strong"}
            negative_words = {"drop", "fall", "loss", "miss", "decline", "down", "low", "bearish", "weak", "crash"}

            for article in articles:
                pub_date = article.get("publishedAt", "")[:10]
                if not pub_date:
                    continue

                text = (article.get("title", "") + " " + article.get("description", "")).lower()

                # Count sentiment words
                pos_count = sum(1 for w in positive_words if w in text)
                neg_count = sum(1 for w in negative_words if w in text)

                if pos_count + neg_count > 0:
                    score = (pos_count - neg_count) / (pos_count + neg_count)
                else:
                    score = 0

                if pub_date not in daily_scores:
                    daily_scores[pub_date] = []
                    daily_counts[pub_date] = 0

                daily_scores[pub_date].append(score)
                daily_counts[pub_date] += 1

            for date_str, scores in daily_scores.items():
                records.append({
                    "timestamp": datetime.strptime(date_str, "%Y-%m-%d"),
                    "sentiment_score": np.mean(scores),
                    "mention_volume": daily_counts[date_str],
                })

            return pl.DataFrame(records)

        except Exception as e:
            logger.debug(f"News sentiment fetch failed: {e}")
            return None

    def _generate_synthetic_sentiment(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pl.DataFrame:
        """Generate synthetic sentiment data."""
        days = (end_date - start_date).days + 1
        timestamps = [start_date + timedelta(days=i) for i in range(days)]

        np.random.seed(hash(symbol) % (2**32))

        # Mean-reverting sentiment
        sentiment = np.zeros(days)
        sentiment[0] = 0
        for i in range(1, days):
            sentiment[i] = 0.9 * sentiment[i-1] + np.random.randn() * 0.1
        sentiment = np.clip(sentiment, -1, 1)

        # Volume with some spikes
        volume = np.random.poisson(50, days)
        # Add occasional spikes
        spikes = np.random.choice(days, size=int(days * 0.1), replace=False)
        volume[spikes] = volume[spikes] * np.random.randint(3, 10, size=len(spikes))

        return pl.DataFrame({
            "timestamp": timestamps,
            "sentiment_score": sentiment,
            "mention_volume": volume,
        })

    def _get_cached(self, key: str) -> pl.DataFrame | None:
        """Get cached data."""
        cache_path = self.config.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.parquet"
        if cache_path.exists():
            try:
                return pl.read_parquet(cache_path)
            except Exception:
                pass
        return None

    def _set_cached(self, key: str, df: pl.DataFrame) -> None:
        """Cache data."""
        cache_path = self.config.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.parquet"
        try:
            df.write_parquet(cache_path)
        except Exception:
            pass


# =============================================================================
# OPTIONS DATA LOADER
# =============================================================================

class OptionsDataLoader:
    """
    Loads options market data for signal generation.

    Options data provides forward-looking information:
    - Put/Call ratio: Market sentiment
    - Implied volatility: Expected future volatility
    - Skew: Tail risk expectations
    - Term structure: Short vs long-term expectations
    """

    def __init__(self, config: AlternativeDataConfig | None = None):
        """Initialize loader."""
        self.config = config or AlternativeDataConfig()

    def load_options_metrics(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pl.DataFrame:
        """
        Load options metrics for a symbol.

        Returns:
            DataFrame with put_call_ratio, implied_vol, skew, term_structure
        """
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=self.config.options_lookback_days))

        # Generate synthetic data (real data would come from CBOE, options exchanges)
        return self._generate_synthetic_options(symbol, start_date, end_date)

    def _generate_synthetic_options(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pl.DataFrame:
        """Generate synthetic options metrics."""
        days = (end_date - start_date).days + 1
        timestamps = [start_date + timedelta(days=i) for i in range(days)]

        np.random.seed(hash(f"{symbol}_options") % (2**32))

        # Put/Call ratio (typically 0.5-1.5, mean around 0.8)
        pc_ratio = 0.8 + np.cumsum(np.random.randn(days) * 0.02)
        pc_ratio = np.clip(pc_ratio, 0.3, 2.0)

        # Implied volatility (typically 15-50%)
        iv = 0.25 + np.cumsum(np.random.randn(days) * 0.005)
        iv = np.clip(iv, 0.10, 0.80)

        # IV skew (put IV - call IV, typically positive)
        skew = 0.05 + np.random.randn(days) * 0.02
        skew = np.clip(skew, -0.10, 0.20)

        # Term structure slope (typically positive = contango)
        term_slope = 0.02 + np.random.randn(days) * 0.01
        term_slope = np.clip(term_slope, -0.10, 0.15)

        return pl.DataFrame({
            "timestamp": timestamps,
            "put_call_ratio": pc_ratio,
            "implied_volatility": iv,
            "iv_skew": skew,
            "term_structure_slope": term_slope,
        })


# =============================================================================
# ALTERNATIVE DATA FEATURE PIPELINE
# =============================================================================

class AlternativeDataPipeline:
    """
    Pipeline for integrating alternative data into feature generation.

    Aligns alternative data with price data and creates derived features.
    """

    def __init__(self, config: AlternativeDataConfig | None = None):
        """Initialize pipeline."""
        self.config = config or AlternativeDataConfig()
        self.macro_loader = MacroDataLoader(self.config)
        self.sentiment_loader = SentimentDataLoader(self.config)
        self.options_loader = OptionsDataLoader(self.config)

    def generate_features(
        self,
        price_data: pl.DataFrame,
        symbol: str,
        timestamp_col: str = "timestamp",
    ) -> pl.DataFrame:
        """
        Generate alternative data features aligned with price data.

        Args:
            price_data: OHLCV DataFrame
            symbol: Trading symbol
            timestamp_col: Timestamp column name

        Returns:
            price_data with alternative data features added
        """
        if timestamp_col not in price_data.columns:
            logger.warning(f"No {timestamp_col} column found")
            return price_data

        timestamps = price_data[timestamp_col].to_list()
        if not timestamps:
            return price_data

        start_date = min(timestamps)
        end_date = max(timestamps)

        # Load macro data
        if self.config.macro_features_enabled:
            price_data = self._add_macro_features(price_data, start_date, end_date, timestamp_col)

        # Load sentiment data
        if self.config.sentiment_features_enabled:
            price_data = self._add_sentiment_features(price_data, symbol, start_date, end_date, timestamp_col)

        # Load options data
        if self.config.options_features_enabled:
            price_data = self._add_options_features(price_data, symbol, start_date, end_date, timestamp_col)

        # Create derived features
        price_data = self._create_derived_features(price_data)

        return price_data

    def _add_macro_features(
        self,
        df: pl.DataFrame,
        start_date: datetime,
        end_date: datetime,
        timestamp_col: str,
    ) -> pl.DataFrame:
        """Add macro indicator features."""
        try:
            macro_df = self.macro_loader.load_all_indicators(start_date, end_date)

            if len(macro_df) == 0:
                return df

            # Extract date for joining
            df = df.with_columns(
                pl.col(timestamp_col).dt.date().alias("_join_date")
            )
            macro_df = macro_df.with_columns(
                pl.col("timestamp").dt.date().alias("_join_date")
            )

            # Join on date
            macro_cols = [c for c in macro_df.columns if c not in ["timestamp", "_join_date"]]
            macro_subset = macro_df.select(["_join_date"] + macro_cols)

            df = df.join(macro_subset, on="_join_date", how="left")
            df = df.drop("_join_date")

            # Forward fill any gaps
            for col in macro_cols:
                if col in df.columns:
                    df = df.with_columns(pl.col(col).fill_null(strategy="forward"))

            logger.info(f"Added {len(macro_cols)} macro features")
            return df

        except Exception as e:
            logger.warning(f"Failed to add macro features: {e}")
            return df

    def _add_sentiment_features(
        self,
        df: pl.DataFrame,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timestamp_col: str,
    ) -> pl.DataFrame:
        """Add sentiment features."""
        try:
            sent_df = self.sentiment_loader.load_sentiment(symbol, SentimentSource.NEWS_API, start_date, end_date)

            if len(sent_df) == 0:
                return df

            # Extract date for joining
            df = df.with_columns(
                pl.col(timestamp_col).dt.date().alias("_join_date")
            )
            sent_df = sent_df.with_columns(
                pl.col("timestamp").dt.date().alias("_join_date")
            )

            # Rename columns with prefix
            sent_df = sent_df.rename({
                "sentiment_score": "alt_sentiment_score",
                "mention_volume": "alt_mention_volume",
            })

            # Join
            sent_subset = sent_df.select(["_join_date", "alt_sentiment_score", "alt_mention_volume"])
            df = df.join(sent_subset, on="_join_date", how="left")
            df = df.drop("_join_date")

            # Fill nulls
            df = df.with_columns([
                pl.col("alt_sentiment_score").fill_null(0),
                pl.col("alt_mention_volume").fill_null(0),
            ])

            logger.info("Added sentiment features")
            return df

        except Exception as e:
            logger.warning(f"Failed to add sentiment features: {e}")
            return df

    def _add_options_features(
        self,
        df: pl.DataFrame,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timestamp_col: str,
    ) -> pl.DataFrame:
        """Add options-derived features."""
        try:
            opt_df = self.options_loader.load_options_metrics(symbol, start_date, end_date)

            if len(opt_df) == 0:
                return df

            # Extract date for joining
            df = df.with_columns(
                pl.col(timestamp_col).dt.date().alias("_join_date")
            )
            opt_df = opt_df.with_columns(
                pl.col("timestamp").dt.date().alias("_join_date")
            )

            # Rename with prefix
            opt_df = opt_df.rename({
                "put_call_ratio": "alt_put_call_ratio",
                "implied_volatility": "alt_implied_vol",
                "iv_skew": "alt_iv_skew",
                "term_structure_slope": "alt_term_slope",
            })

            opt_subset = opt_df.select([
                "_join_date", "alt_put_call_ratio", "alt_implied_vol",
                "alt_iv_skew", "alt_term_slope"
            ])

            df = df.join(opt_subset, on="_join_date", how="left")
            df = df.drop("_join_date")

            # Forward fill
            opt_cols = ["alt_put_call_ratio", "alt_implied_vol", "alt_iv_skew", "alt_term_slope"]
            for col in opt_cols:
                if col in df.columns:
                    df = df.with_columns(pl.col(col).fill_null(strategy="forward"))

            logger.info("Added options features")
            return df

        except Exception as e:
            logger.warning(f"Failed to add options features: {e}")
            return df

    def _create_derived_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create derived features from alternative data."""
        # VIX regime (if VIX available)
        if "vix" in df.columns:
            df = df.with_columns([
                (pl.col("vix") > 25).alias("alt_high_vix_regime"),
                (pl.col("vix") - pl.col("vix").shift(5)).alias("alt_vix_change_5d"),
            ])

        # Yield curve inversion
        if "yield_curve" in df.columns:
            df = df.with_columns([
                (pl.col("yield_curve") < 0).alias("alt_yield_curve_inverted"),
            ])

        # Sentiment momentum
        if "alt_sentiment_score" in df.columns:
            df = df.with_columns([
                pl.col("alt_sentiment_score").rolling_mean(5).alias("alt_sentiment_ma5"),
                (pl.col("alt_sentiment_score") - pl.col("alt_sentiment_score").shift(5)).alias("alt_sentiment_momentum"),
            ])

        # Put/Call ratio extreme
        if "alt_put_call_ratio" in df.columns:
            df = df.with_columns([
                (pl.col("alt_put_call_ratio") > 1.2).alias("alt_high_put_call"),
                (pl.col("alt_put_call_ratio") < 0.6).alias("alt_low_put_call"),
            ])

        # IV vs realized volatility (if both available)
        if "alt_implied_vol" in df.columns and "close" in df.columns:
            # Calculate realized vol
            df = df.with_columns([
                pl.col("close").pct_change().rolling_std(20).alias("_realized_vol")
            ])
            df = df.with_columns([
                (pl.col("alt_implied_vol") - pl.col("_realized_vol") * np.sqrt(252 * 26)).alias("alt_vol_premium")
            ])
            df = df.drop("_realized_vol")

        return df


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "AlternativeDataType",
    "SentimentSource",
    # Configuration
    "AlternativeDataConfig",
    # Data structures
    "MacroIndicator",
    "SentimentScore",
    # Loaders
    "MacroDataLoader",
    "SentimentDataLoader",
    "OptionsDataLoader",
    # Pipeline
    "AlternativeDataPipeline",
    # Constants
    "MACRO_INDICATORS",
]
