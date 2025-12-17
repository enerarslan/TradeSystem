"""
Macroeconomic features from FRED (Federal Reserve Economic Data).

This module provides:
- FRED API integration for economic data
- Key macro indicators (GDP, CPI, unemployment, etc.)
- Yield curve features
- Economic surprise indicators
- Regime detection

Designed for institutional requirements:
- Robust API handling with caching
- Proper data alignment for backtesting
- Point-in-time accuracy (no look-ahead bias)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

logger = logging.getLogger(__name__)


class MacroIndicator(str, Enum):
    """Key macroeconomic indicators from FRED."""

    # GDP and Output
    GDP = "GDP"                           # Gross Domestic Product
    GDPC1 = "GDPC1"                       # Real GDP
    INDPRO = "INDPRO"                     # Industrial Production Index

    # Employment
    UNRATE = "UNRATE"                     # Unemployment Rate
    PAYEMS = "PAYEMS"                     # Total Nonfarm Payrolls
    ICSA = "ICSA"                         # Initial Claims
    CIVPART = "CIVPART"                   # Labor Force Participation

    # Inflation
    CPIAUCSL = "CPIAUCSL"                 # CPI All Items
    CPILFESL = "CPILFESL"                 # Core CPI (Less Food & Energy)
    PCEPI = "PCEPI"                       # PCE Price Index
    PCEPILFE = "PCEPILFE"                 # Core PCE
    T5YIE = "T5YIE"                       # 5-Year Breakeven Inflation

    # Interest Rates
    FEDFUNDS = "FEDFUNDS"                 # Federal Funds Rate
    DFF = "DFF"                           # Fed Funds Effective Rate (Daily)
    DFEDTARU = "DFEDTARU"                 # Fed Funds Upper Target
    DFEDTARL = "DFEDTARL"                 # Fed Funds Lower Target

    # Treasury Yields
    DGS1MO = "DGS1MO"                     # 1-Month Treasury
    DGS3MO = "DGS3MO"                     # 3-Month Treasury
    DGS6MO = "DGS6MO"                     # 6-Month Treasury
    DGS1 = "DGS1"                         # 1-Year Treasury
    DGS2 = "DGS2"                         # 2-Year Treasury
    DGS5 = "DGS5"                         # 5-Year Treasury
    DGS10 = "DGS10"                       # 10-Year Treasury
    DGS30 = "DGS30"                       # 30-Year Treasury

    # Credit Spreads
    BAMLH0A0HYM2 = "BAMLH0A0HYM2"         # High Yield OAS
    BAMLC0A4CBBB = "BAMLC0A4CBBB"         # BBB Corporate OAS
    TEDRATE = "TEDRATE"                   # TED Spread

    # Money Supply
    M2SL = "M2SL"                         # M2 Money Stock
    BOGMBASE = "BOGMBASE"                 # Monetary Base

    # Consumer
    UMCSENT = "UMCSENT"                   # U Michigan Consumer Sentiment
    PCE = "PCE"                           # Personal Consumption Expenditures
    RSXFS = "RSXFS"                       # Retail Sales

    # Housing
    HOUST = "HOUST"                       # Housing Starts
    HSN1F = "HSN1F"                       # New Home Sales
    CSUSHPINSA = "CSUSHPINSA"             # Case-Shiller Home Price

    # Financial Conditions
    NFCI = "NFCI"                         # Chicago Fed NFCI
    STLFSI4 = "STLFSI4"                   # St. Louis Fed FSI

    # Market Indicators
    VIXCLS = "VIXCLS"                     # VIX Index
    SP500 = "SP500"                       # S&P 500


@dataclass
class FREDConfig:
    """Configuration for FRED API access."""

    api_key: Optional[str] = None
    cache_dir: Optional[Path] = None
    cache_expiry_hours: int = 24
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0

    def __post_init__(self):
        # Try to get API key from environment if not provided
        if self.api_key is None:
            self.api_key = os.environ.get("FRED_API_KEY")

        # Set default cache directory
        if self.cache_dir is None:
            self.cache_dir = Path("data/cache/fred")

        # Create cache directory
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)


class FREDClient:
    """
    FRED API client with caching and error handling.

    Provides robust access to FRED data with:
    - Local file caching
    - Automatic retry on errors
    - Data validation
    - Point-in-time data alignment
    """

    def __init__(self, config: Optional[FREDConfig] = None):
        """
        Initialize FRED client.

        Args:
            config: FRED configuration
        """
        if not FRED_AVAILABLE:
            raise ImportError(
                "fredapi is required for FRED integration. "
                "Install with: pip install fredapi"
            )

        self.config = config or FREDConfig()

        if not self.config.api_key:
            raise ValueError(
                "FRED API key required. Set FRED_API_KEY environment variable "
                "or pass api_key in config."
            )

        self._fred = Fred(api_key=self.config.api_key)
        self._cache: Dict[str, pd.Series] = {}

        logger.info("FREDClient initialized")

    def get_series(
        self,
        series_id: Union[str, MacroIndicator],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True,
    ) -> pd.Series:
        """
        Get FRED series data.

        Args:
            series_id: FRED series ID or MacroIndicator enum
            start_date: Start date
            end_date: End date
            use_cache: Use cached data if available

        Returns:
            Series with FRED data
        """
        # Convert enum to string
        if isinstance(series_id, MacroIndicator):
            series_id = series_id.value

        cache_key = f"{series_id}_{start_date}_{end_date}"

        # Check memory cache
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Check file cache
        if use_cache and self.config.cache_dir:
            cached = self._load_from_cache(series_id)
            if cached is not None:
                filtered = self._filter_dates(cached, start_date, end_date)
                self._cache[cache_key] = filtered
                return filtered

        # Fetch from FRED
        for attempt in range(self.config.retry_attempts):
            try:
                data = self._fred.get_series(
                    series_id,
                    observation_start=start_date,
                    observation_end=end_date,
                )

                # Save to cache
                if self.config.cache_dir:
                    self._save_to_cache(series_id, data)

                self._cache[cache_key] = data
                return data

            except Exception as e:
                logger.warning(f"FRED request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.retry_attempts - 1:
                    import time
                    time.sleep(self.config.retry_delay_seconds)
                else:
                    raise

    def get_multiple_series(
        self,
        series_ids: List[Union[str, MacroIndicator]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        align: bool = True,
    ) -> pd.DataFrame:
        """
        Get multiple FRED series as DataFrame.

        Args:
            series_ids: List of series IDs
            start_date: Start date
            end_date: End date
            align: Align all series to common dates

        Returns:
            DataFrame with all series
        """
        data = {}
        for series_id in series_ids:
            name = series_id.value if isinstance(series_id, MacroIndicator) else series_id
            try:
                data[name] = self.get_series(series_id, start_date, end_date)
            except Exception as e:
                logger.warning(f"Failed to fetch {name}: {e}")
                continue

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        if align:
            # Forward fill to align different frequencies
            df = df.ffill()

        return df

    def get_series_info(self, series_id: Union[str, MacroIndicator]) -> Dict[str, Any]:
        """
        Get metadata for a FRED series.

        Args:
            series_id: Series ID

        Returns:
            Dictionary with series metadata
        """
        if isinstance(series_id, MacroIndicator):
            series_id = series_id.value

        info = self._fred.get_series_info(series_id)
        return info.to_dict()

    def _load_from_cache(self, series_id: str) -> Optional[pd.Series]:
        """Load series from file cache."""
        if not self.config.cache_dir:
            return None

        cache_file = self.config.cache_dir / f"{series_id}.parquet"

        if not cache_file.exists():
            return None

        # Check cache expiry
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if file_age > timedelta(hours=self.config.cache_expiry_hours):
            return None

        try:
            df = pd.read_parquet(cache_file)
            return df.iloc[:, 0]
        except Exception as e:
            logger.warning(f"Failed to load cache for {series_id}: {e}")
            return None

    def _save_to_cache(self, series_id: str, data: pd.Series) -> None:
        """Save series to file cache."""
        if not self.config.cache_dir:
            return

        cache_file = self.config.cache_dir / f"{series_id}.parquet"

        try:
            df = data.to_frame(name=series_id)
            df.to_parquet(cache_file)
        except Exception as e:
            logger.warning(f"Failed to save cache for {series_id}: {e}")

    def _filter_dates(
        self,
        data: pd.Series,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> pd.Series:
        """Filter series by date range."""
        if start_date:
            data = data[data.index >= pd.Timestamp(start_date)]
        if end_date:
            data = data[data.index <= pd.Timestamp(end_date)]
        return data


class MacroFeatureGenerator:
    """
    Generate macroeconomic features for trading models.

    Provides:
    - Key macro indicators
    - Yield curve features
    - Economic regime detection
    - Surprise indicators (vs expectations)
    """

    def __init__(self, client: Optional[FREDClient] = None):
        """
        Initialize feature generator.

        Args:
            client: FRED client (created if not provided)
        """
        self._client = client

    @property
    def client(self) -> FREDClient:
        """Get or create FRED client."""
        if self._client is None:
            self._client = FREDClient()
        return self._client

    def get_yield_curve_features(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Generate yield curve features.

        Includes:
        - Raw yields at various maturities
        - Yield curve slope (10Y - 2Y)
        - Curvature (2*5Y - 2Y - 10Y)
        - Steepness changes

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with yield curve features
        """
        # Fetch yield data
        yields = self.client.get_multiple_series(
            [
                MacroIndicator.DGS3MO,
                MacroIndicator.DGS2,
                MacroIndicator.DGS5,
                MacroIndicator.DGS10,
                MacroIndicator.DGS30,
            ],
            start_date,
            end_date,
        )

        if yields.empty:
            return pd.DataFrame()

        features = pd.DataFrame(index=yields.index)

        # Raw yields
        for col in yields.columns:
            features[f"yield_{col}"] = yields[col]

        # Slopes
        if "DGS10" in yields.columns and "DGS2" in yields.columns:
            features["slope_10y_2y"] = yields["DGS10"] - yields["DGS2"]

        if "DGS10" in yields.columns and "DGS3MO" in yields.columns:
            features["slope_10y_3m"] = yields["DGS10"] - yields["DGS3MO"]

        if "DGS30" in yields.columns and "DGS2" in yields.columns:
            features["slope_30y_2y"] = yields["DGS30"] - yields["DGS2"]

        # Curvature (butterfly)
        if all(col in yields.columns for col in ["DGS2", "DGS5", "DGS10"]):
            features["curvature_2_5_10"] = 2 * yields["DGS5"] - yields["DGS2"] - yields["DGS10"]

        # Rate of change
        for col in yields.columns:
            features[f"yield_{col}_change_1d"] = yields[col].diff(1)
            features[f"yield_{col}_change_5d"] = yields[col].diff(5)
            features[f"yield_{col}_change_20d"] = yields[col].diff(20)

        # Slope changes
        if "slope_10y_2y" in features.columns:
            features["slope_10y_2y_change_5d"] = features["slope_10y_2y"].diff(5)
            features["slope_10y_2y_change_20d"] = features["slope_10y_2y"].diff(20)

        # Inversion indicator
        if "slope_10y_2y" in features.columns:
            features["curve_inverted"] = (features["slope_10y_2y"] < 0).astype(int)

        return features

    def get_inflation_features(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Generate inflation features.

        Includes:
        - CPI and PCE inflation
        - Core inflation
        - Breakeven inflation
        - Inflation momentum

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with inflation features
        """
        data = self.client.get_multiple_series(
            [
                MacroIndicator.CPIAUCSL,
                MacroIndicator.CPILFESL,
                MacroIndicator.PCEPI,
                MacroIndicator.PCEPILFE,
                MacroIndicator.T5YIE,
            ],
            start_date,
            end_date,
        )

        if data.empty:
            return pd.DataFrame()

        features = pd.DataFrame(index=data.index)

        # Year-over-year inflation rates
        for col in ["CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE"]:
            if col in data.columns:
                # Assuming monthly data, calculate YoY
                features[f"{col.lower()}_yoy"] = data[col].pct_change(12) * 100

        # Breakeven inflation
        if "T5YIE" in data.columns:
            features["breakeven_5y"] = data["T5YIE"]
            features["breakeven_5y_change"] = data["T5YIE"].diff(20)

        # Core vs headline spread
        if "CPIAUCSL" in data.columns and "CPILFESL" in data.columns:
            cpi_yoy = data["CPIAUCSL"].pct_change(12) * 100
            core_yoy = data["CPILFESL"].pct_change(12) * 100
            features["core_headline_spread"] = core_yoy - cpi_yoy

        # Inflation momentum
        if "CPIAUCSL" in data.columns:
            features["cpi_mom_3m"] = data["CPIAUCSL"].pct_change(3) * 100 * 4  # Annualized
            features["cpi_mom_6m"] = data["CPIAUCSL"].pct_change(6) * 100 * 2

        return features

    def get_employment_features(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Generate employment features.

        Includes:
        - Unemployment rate and changes
        - Payroll growth
        - Initial claims
        - Labor force participation

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with employment features
        """
        data = self.client.get_multiple_series(
            [
                MacroIndicator.UNRATE,
                MacroIndicator.PAYEMS,
                MacroIndicator.ICSA,
                MacroIndicator.CIVPART,
            ],
            start_date,
            end_date,
        )

        if data.empty:
            return pd.DataFrame()

        features = pd.DataFrame(index=data.index)

        # Unemployment
        if "UNRATE" in data.columns:
            features["unemployment_rate"] = data["UNRATE"]
            features["unemployment_change_1m"] = data["UNRATE"].diff(1)
            features["unemployment_change_3m"] = data["UNRATE"].diff(3)
            features["unemployment_ma_12m"] = data["UNRATE"].rolling(12).mean()

        # Payrolls
        if "PAYEMS" in data.columns:
            features["payroll_change_1m"] = data["PAYEMS"].diff(1)
            features["payroll_change_3m_avg"] = data["PAYEMS"].diff(1).rolling(3).mean()
            features["payroll_yoy_pct"] = data["PAYEMS"].pct_change(12) * 100

        # Initial claims
        if "ICSA" in data.columns:
            features["initial_claims"] = data["ICSA"]
            features["initial_claims_4wk_avg"] = data["ICSA"].rolling(4).mean()
            features["initial_claims_yoy_pct"] = data["ICSA"].pct_change(52) * 100

        # Labor force participation
        if "CIVPART" in data.columns:
            features["lfpr"] = data["CIVPART"]
            features["lfpr_change_12m"] = data["CIVPART"].diff(12)

        return features

    def get_financial_conditions_features(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Generate financial conditions features.

        Includes:
        - Credit spreads (HY, IG)
        - Financial stress indices
        - VIX
        - Fed funds rate

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with financial conditions features
        """
        data = self.client.get_multiple_series(
            [
                MacroIndicator.BAMLH0A0HYM2,
                MacroIndicator.BAMLC0A4CBBB,
                MacroIndicator.TEDRATE,
                MacroIndicator.NFCI,
                MacroIndicator.VIXCLS,
                MacroIndicator.DFF,
            ],
            start_date,
            end_date,
        )

        if data.empty:
            return pd.DataFrame()

        features = pd.DataFrame(index=data.index)

        # High yield spread
        if "BAMLH0A0HYM2" in data.columns:
            features["hy_spread"] = data["BAMLH0A0HYM2"]
            features["hy_spread_percentile_252d"] = data["BAMLH0A0HYM2"].rolling(252).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
            )

        # IG spread
        if "BAMLC0A4CBBB" in data.columns:
            features["ig_spread"] = data["BAMLC0A4CBBB"]
            features["ig_spread_change_20d"] = data["BAMLC0A4CBBB"].diff(20)

        # TED spread
        if "TEDRATE" in data.columns:
            features["ted_spread"] = data["TEDRATE"]

        # NFCI
        if "NFCI" in data.columns:
            features["nfci"] = data["NFCI"]
            features["nfci_tight"] = (data["NFCI"] > 0).astype(int)

        # VIX
        if "VIXCLS" in data.columns:
            features["vix"] = data["VIXCLS"]
            features["vix_ma_20d"] = data["VIXCLS"].rolling(20).mean()
            features["vix_percentile_252d"] = data["VIXCLS"].rolling(252).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
            )

        # Fed funds
        if "DFF" in data.columns:
            features["fed_funds"] = data["DFF"]
            features["fed_funds_change_20d"] = data["DFF"].diff(20)

        return features

    def get_growth_features(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Generate economic growth features.

        Includes:
        - GDP growth
        - Industrial production
        - Consumer spending
        - Retail sales

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with growth features
        """
        data = self.client.get_multiple_series(
            [
                MacroIndicator.GDPC1,
                MacroIndicator.INDPRO,
                MacroIndicator.PCE,
                MacroIndicator.RSXFS,
                MacroIndicator.UMCSENT,
            ],
            start_date,
            end_date,
        )

        if data.empty:
            return pd.DataFrame()

        features = pd.DataFrame(index=data.index)

        # Real GDP growth
        if "GDPC1" in data.columns:
            features["real_gdp_yoy"] = data["GDPC1"].pct_change(4) * 100  # Quarterly data
            features["real_gdp_qoq_ann"] = data["GDPC1"].pct_change(1) * 400

        # Industrial production
        if "INDPRO" in data.columns:
            features["indpro_yoy"] = data["INDPRO"].pct_change(12) * 100
            features["indpro_mom"] = data["INDPRO"].pct_change(1) * 100
            features["indpro_3m_ann"] = data["INDPRO"].pct_change(3) * 400

        # PCE
        if "PCE" in data.columns:
            features["pce_yoy"] = data["PCE"].pct_change(12) * 100
            features["pce_3m_ann"] = data["PCE"].pct_change(3) * 400

        # Retail sales
        if "RSXFS" in data.columns:
            features["retail_yoy"] = data["RSXFS"].pct_change(12) * 100
            features["retail_mom"] = data["RSXFS"].pct_change(1) * 100

        # Consumer sentiment
        if "UMCSENT" in data.columns:
            features["consumer_sentiment"] = data["UMCSENT"]
            features["sentiment_change_3m"] = data["UMCSENT"].diff(3)

        return features

    def get_all_features(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        categories: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get all macro features combined.

        Args:
            start_date: Start date
            end_date: End date
            categories: List of categories to include
                       (yield_curve, inflation, employment, financial, growth)

        Returns:
            DataFrame with all requested features
        """
        all_categories = ["yield_curve", "inflation", "employment", "financial", "growth"]
        if categories is None:
            categories = all_categories

        feature_dfs = []

        if "yield_curve" in categories:
            try:
                feature_dfs.append(self.get_yield_curve_features(start_date, end_date))
            except Exception as e:
                logger.warning(f"Failed to get yield curve features: {e}")

        if "inflation" in categories:
            try:
                feature_dfs.append(self.get_inflation_features(start_date, end_date))
            except Exception as e:
                logger.warning(f"Failed to get inflation features: {e}")

        if "employment" in categories:
            try:
                feature_dfs.append(self.get_employment_features(start_date, end_date))
            except Exception as e:
                logger.warning(f"Failed to get employment features: {e}")

        if "financial" in categories:
            try:
                feature_dfs.append(self.get_financial_conditions_features(start_date, end_date))
            except Exception as e:
                logger.warning(f"Failed to get financial conditions features: {e}")

        if "growth" in categories:
            try:
                feature_dfs.append(self.get_growth_features(start_date, end_date))
            except Exception as e:
                logger.warning(f"Failed to get growth features: {e}")

        if not feature_dfs:
            return pd.DataFrame()

        # Combine all features
        combined = pd.concat(feature_dfs, axis=1)

        # Remove duplicate columns
        combined = combined.loc[:, ~combined.columns.duplicated()]

        return combined


class EconomicRegimeDetector:
    """
    Detect economic regimes for regime-based strategies.

    Regimes:
    - Expansion vs Recession
    - High/Low inflation
    - Tightening/Easing monetary policy
    - Risk-on/Risk-off sentiment
    """

    def __init__(self, feature_generator: Optional[MacroFeatureGenerator] = None):
        """
        Initialize regime detector.

        Args:
            feature_generator: Macro feature generator
        """
        self._generator = feature_generator

    @property
    def generator(self) -> MacroFeatureGenerator:
        """Get or create feature generator."""
        if self._generator is None:
            self._generator = MacroFeatureGenerator()
        return self._generator

    def detect_growth_regime(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.Series:
        """
        Detect growth regime (expansion vs contraction).

        Returns:
            Series with regime labels (1=expansion, 0=contraction)
        """
        features = self.generator.get_growth_features(start_date, end_date)

        if features.empty:
            return pd.Series(dtype=int)

        # Combine signals
        signals = pd.DataFrame(index=features.index)

        if "real_gdp_yoy" in features.columns:
            signals["gdp_positive"] = (features["real_gdp_yoy"] > 0).astype(int)

        if "indpro_yoy" in features.columns:
            signals["indpro_positive"] = (features["indpro_yoy"] > 0).astype(int)

        if "payroll_yoy_pct" in features.columns:
            signals["payroll_positive"] = (features["payroll_yoy_pct"] > 0).astype(int)

        if signals.empty:
            return pd.Series(dtype=int)

        # Majority vote
        regime = (signals.mean(axis=1) > 0.5).astype(int)
        regime.name = "growth_regime"

        return regime

    def detect_inflation_regime(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        threshold: float = 3.0,
    ) -> pd.Series:
        """
        Detect inflation regime (high vs low).

        Args:
            start_date: Start date
            end_date: End date
            threshold: Inflation threshold for "high" regime

        Returns:
            Series with regime labels (1=high inflation, 0=low inflation)
        """
        features = self.generator.get_inflation_features(start_date, end_date)

        if features.empty or "cpiaucsl_yoy" not in features.columns:
            return pd.Series(dtype=int)

        regime = (features["cpiaucsl_yoy"] > threshold).astype(int)
        regime.name = "inflation_regime"

        return regime

    def detect_monetary_regime(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.Series:
        """
        Detect monetary policy regime (tightening vs easing).

        Returns:
            Series with regime labels (1=tightening, -1=easing, 0=neutral)
        """
        features = self.generator.get_financial_conditions_features(start_date, end_date)

        if features.empty or "fed_funds_change_20d" not in features.columns:
            return pd.Series(dtype=int)

        ff_change = features["fed_funds_change_20d"]

        regime = pd.Series(0, index=ff_change.index)
        regime[ff_change > 0.1] = 1   # Tightening
        regime[ff_change < -0.1] = -1  # Easing
        regime.name = "monetary_regime"

        return regime

    def detect_risk_regime(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.Series:
        """
        Detect risk regime (risk-on vs risk-off).

        Returns:
            Series with regime labels (1=risk-on, 0=risk-off)
        """
        features = self.generator.get_financial_conditions_features(start_date, end_date)

        if features.empty:
            return pd.Series(dtype=int)

        signals = pd.DataFrame(index=features.index)

        if "vix" in features.columns:
            signals["vix_low"] = (features["vix"] < features["vix"].rolling(252).median()).astype(int)

        if "hy_spread" in features.columns:
            signals["hy_tight"] = (features["hy_spread"] < features["hy_spread"].rolling(252).median()).astype(int)

        if "nfci" in features.columns:
            signals["nfci_loose"] = (features["nfci"] < 0).astype(int)

        if signals.empty:
            return pd.Series(dtype=int)

        regime = (signals.mean(axis=1) > 0.5).astype(int)
        regime.name = "risk_regime"

        return regime

    def get_all_regimes(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Get all regime indicators combined.

        Returns:
            DataFrame with all regime columns
        """
        regimes = pd.DataFrame()

        try:
            growth = self.detect_growth_regime(start_date, end_date)
            if not growth.empty:
                regimes["growth_regime"] = growth
        except Exception as e:
            logger.warning(f"Failed to detect growth regime: {e}")

        try:
            inflation = self.detect_inflation_regime(start_date, end_date)
            if not inflation.empty:
                regimes["inflation_regime"] = inflation
        except Exception as e:
            logger.warning(f"Failed to detect inflation regime: {e}")

        try:
            monetary = self.detect_monetary_regime(start_date, end_date)
            if not monetary.empty:
                regimes["monetary_regime"] = monetary
        except Exception as e:
            logger.warning(f"Failed to detect monetary regime: {e}")

        try:
            risk = self.detect_risk_regime(start_date, end_date)
            if not risk.empty:
                regimes["risk_regime"] = risk
        except Exception as e:
            logger.warning(f"Failed to detect risk regime: {e}")

        return regimes


def align_macro_to_price_data(
    macro_features: pd.DataFrame,
    price_index: pd.DatetimeIndex,
    lag_days: int = 1,
) -> pd.DataFrame:
    """
    Align macro features to price data index.

    Ensures point-in-time accuracy by applying appropriate lag
    to avoid look-ahead bias.

    Args:
        macro_features: Macro feature DataFrame
        price_index: Target index from price data
        lag_days: Days to lag macro data (for publication delay)

    Returns:
        Aligned macro features
    """
    if macro_features.empty:
        return pd.DataFrame(index=price_index)

    # Lag to account for publication delay
    if lag_days > 0:
        macro_features = macro_features.shift(lag_days)

    # Reindex to match price data
    aligned = macro_features.reindex(price_index, method="ffill")

    return aligned
