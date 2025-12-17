"""
Feature engineering pipeline for AlphaTrade system.

This module provides:
- Statistical features (returns, volatility, skewness, etc.)
- Cross-sectional features (rankings, z-scores)
- Lagged features
- Feature processing and scaling
- sklearn-compatible pipeline
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from loguru import logger

from src.features.technical.indicators import TechnicalIndicators
from src.features.microstructure import OrderBookDynamics
from src.features.transformers import TimeCyclicalEncoder

class StatisticalFeatures:
    """Generator for statistical features."""

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize with OHLCV DataFrame.

        Args:
            df: DataFrame with OHLCV data
        """
        self.df = df

    def returns(
        self,
        periods: list[int] | None = None,
        method: Literal["simple", "log"] = "simple",
    ) -> pd.DataFrame:
        """
        Calculate returns over multiple periods.

        Args:
            periods: Return periods
            method: Return calculation method

        Returns:
            DataFrame with return features
        """
        if periods is None:
            periods = [1, 5, 10, 20, 60]

        features = pd.DataFrame(index=self.df.index)

        for period in periods:
            if method == "simple":
                features[f"return_{period}"] = self.df["close"].pct_change(period)
            else:
                features[f"log_return_{period}"] = np.log(
                    self.df["close"] / self.df["close"].shift(period)
                )

        return features

    def volatility(
        self,
        windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate volatility over multiple windows.

        Args:
            windows: Rolling window sizes

        Returns:
            DataFrame with volatility features
        """
        if windows is None:
            windows = [10, 20, 50]

        features = pd.DataFrame(index=self.df.index)
        log_ret = np.log(self.df["close"] / self.df["close"].shift(1))

        for window in windows:
            # Realized volatility
            features[f"realized_vol_{window}"] = (
                log_ret.rolling(window=window).std() * np.sqrt(252 * 26)
            )

            # Parkinson volatility
            log_hl = np.log(self.df["high"] / self.df["low"]) ** 2
            features[f"parkinson_vol_{window}"] = (
                log_hl.rolling(window=window).mean().apply(np.sqrt)
                / (2 * np.sqrt(np.log(2)))
                * np.sqrt(252 * 26)
            )

        return features

    def skewness_kurtosis(self, window: int = 20) -> pd.DataFrame:
        """
        Calculate rolling skewness and kurtosis.

        Args:
            window: Rolling window size

        Returns:
            DataFrame with skewness and kurtosis
        """
        features = pd.DataFrame(index=self.df.index)
        log_ret = np.log(self.df["close"] / self.df["close"].shift(1))

        features["skewness"] = log_ret.rolling(window=window).skew()
        features["kurtosis"] = log_ret.rolling(window=window).kurt()

        return features

    def zscore(self, window: int = 20) -> pd.DataFrame:
        """
        Calculate z-scores of price and volume.

        Args:
            window: Rolling window for z-score

        Returns:
            DataFrame with z-scores
        """
        features = pd.DataFrame(index=self.df.index)

        # Price z-score
        price_mean = self.df["close"].rolling(window=window).mean()
        price_std = self.df["close"].rolling(window=window).std()
        features["price_zscore"] = (self.df["close"] - price_mean) / price_std

        # Volume z-score
        if "volume" in self.df.columns:
            vol_mean = self.df["volume"].rolling(window=window).mean()
            vol_std = self.df["volume"].rolling(window=window).std()
            features["volume_zscore"] = (self.df["volume"] - vol_mean) / vol_std

        return features

    def autocorrelation(self, lags: list[int] | None = None) -> pd.DataFrame:
        """
        Calculate autocorrelation of returns.

        Args:
            lags: Lag periods for autocorrelation

        Returns:
            DataFrame with autocorrelation features
        """
        if lags is None:
            lags = [1, 5, 10]

        features = pd.DataFrame(index=self.df.index)
        returns = self.df["close"].pct_change()

        for lag in lags:
            features[f"autocorr_{lag}"] = returns.rolling(window=50).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan,
                raw=False,
            )

        return features

    def hurst_exponent(self, window: int = 100) -> pd.Series:
        """
        Calculate rolling Hurst exponent (trend persistence).

        Args:
            window: Rolling window size

        Returns:
            Series with Hurst exponent
        """

        def calc_hurst(ts: np.ndarray) -> float:
            """Calculate Hurst exponent using R/S method."""
            if len(ts) < 20:
                return np.nan

            ts = np.array(ts)
            n = len(ts)

            # Mean
            mean = np.mean(ts)

            # Cumulative deviations
            y = np.cumsum(ts - mean)

            # Range
            r = np.max(y) - np.min(y)

            # Standard deviation
            s = np.std(ts, ddof=1)

            if s == 0:
                return np.nan

            # R/S statistic
            rs = r / s

            # Hurst exponent
            if rs > 0:
                return np.log(rs) / np.log(n)
            return np.nan

        log_ret = np.log(self.df["close"] / self.df["close"].shift(1))
        return log_ret.rolling(window=window).apply(calc_hurst, raw=True)

    def generate_all(self) -> pd.DataFrame:
        """Generate all statistical features."""
        features = pd.DataFrame(index=self.df.index)

        # Returns
        features = pd.concat([features, self.returns()], axis=1)
        features = pd.concat([features, self.returns(method="log")], axis=1)

        # Volatility
        features = pd.concat([features, self.volatility()], axis=1)

        # Higher moments
        features = pd.concat([features, self.skewness_kurtosis()], axis=1)

        # Z-scores
        features = pd.concat([features, self.zscore()], axis=1)

        # Autocorrelation
        features = pd.concat([features, self.autocorrelation()], axis=1)

        # Hurst exponent
        features["hurst"] = self.hurst_exponent()

        return features


class CrossSectionalFeatures:
    """Generator for cross-sectional features across multiple stocks."""

    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        sectors: dict[str, list[str]] | None = None,
    ) -> None:
        """
        Initialize with multi-stock data.

        Args:
            data: Dictionary mapping symbols to DataFrames
            sectors: Dictionary mapping sector names to symbol lists
        """
        self.data = data
        self.sectors = sectors or {}
        self.symbols = list(data.keys())

        # Create price panel
        self.prices = pd.DataFrame(
            {sym: df["close"] for sym, df in data.items()}
        )
        self.returns = self.prices.pct_change()

    def relative_strength(self, period: int = 20) -> pd.DataFrame:
        """
        Calculate relative strength vs market average.

        Args:
            period: Return period for comparison

        Returns:
            DataFrame with relative strength for each symbol
        """
        period_returns = self.prices.pct_change(period)
        market_return = period_returns.mean(axis=1)

        relative = pd.DataFrame(index=self.prices.index)
        for symbol in self.symbols:
            relative[f"{symbol}_rel_strength"] = (
                period_returns[symbol] - market_return
            )

        return relative

    def rank_features(self) -> dict[str, pd.DataFrame]:
        """
        Calculate percentile rank among all stocks.

        Returns:
            Dictionary with rank features for each symbol
        """
        features = {}

        for symbol in self.symbols:
            symbol_features = pd.DataFrame(index=self.prices.index)

            # Return rank
            for period in [5, 20]:
                period_ret = self.prices.pct_change(period)
                symbol_features[f"return_{period}_rank"] = period_ret[symbol].rolling(1).apply(
                    lambda x: stats.percentileofscore(
                        period_ret.loc[x.index[0]].dropna(), x.iloc[0]
                    ) / 100 if not np.isnan(x.iloc[0]) else np.nan,
                    raw=False,
                )

            features[symbol] = symbol_features

        return features

    def zscore_cross_sectional(self) -> dict[str, pd.DataFrame]:
        """
        Calculate cross-sectional z-scores.

        Returns:
            Dictionary with z-score features for each symbol
        """
        features = {}

        # Calculate cross-sectional stats
        for period in [5, 20]:
            period_ret = self.prices.pct_change(period)
            cs_mean = period_ret.mean(axis=1)
            cs_std = period_ret.std(axis=1)

            for symbol in self.symbols:
                if symbol not in features:
                    features[symbol] = pd.DataFrame(index=self.prices.index)

                features[symbol][f"cs_zscore_{period}"] = (
                    (period_ret[symbol] - cs_mean) / cs_std
                )

        return features

    def sector_momentum(self) -> dict[str, pd.DataFrame]:
        """
        Calculate sector momentum relative to stock.

        Returns:
            Dictionary with sector features for each symbol
        """
        if not self.sectors:
            return {}

        features = {}

        # Calculate sector returns
        sector_returns = {}
        for sector, symbols in self.sectors.items():
            valid_symbols = [s for s in symbols if s in self.returns.columns]
            if valid_symbols:
                sector_returns[sector] = self.returns[valid_symbols].mean(axis=1)

        # Calculate features for each symbol
        for symbol in self.symbols:
            symbol_features = pd.DataFrame(index=self.prices.index)

            # Find symbol's sector
            symbol_sector = None
            for sector, symbols in self.sectors.items():
                if symbol in symbols:
                    symbol_sector = sector
                    break

            if symbol_sector and symbol_sector in sector_returns:
                # Relative to own sector
                for period in [5, 20]:
                    stock_ret = self.returns[symbol].rolling(period).sum()
                    sector_ret = sector_returns[symbol_sector].rolling(period).sum()
                    symbol_features[f"vs_sector_{period}"] = stock_ret - sector_ret

            features[symbol] = symbol_features

        return features

    def correlation_features(self, window: int = 60) -> dict[str, pd.DataFrame]:
        """
        Calculate correlation with market.

        Args:
            window: Rolling window for correlation

        Returns:
            Dictionary with correlation features for each symbol
        """
        features = {}
        market_returns = self.returns.mean(axis=1)

        for symbol in self.symbols:
            symbol_features = pd.DataFrame(index=self.prices.index)

            # Rolling correlation with market
            symbol_features["market_corr"] = (
                self.returns[symbol].rolling(window=window).corr(market_returns)
            )

            # Rolling beta
            cov = self.returns[symbol].rolling(window=window).cov(market_returns)
            var = market_returns.rolling(window=window).var()
            symbol_features["beta"] = cov / var

            features[symbol] = symbol_features

        return features


class LaggedFeatures:
    """Generator for lagged features."""

    def __init__(
        self,
        df: pd.DataFrame,
        lag_periods: list[int] | None = None,
    ) -> None:
        """
        Initialize with DataFrame.

        Args:
            df: DataFrame with features
            lag_periods: Periods for lagging
        """
        self.df = df
        self.lag_periods = lag_periods or [1, 2, 5, 10]

    def create_lags(
        self,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Create lagged versions of columns.

        Args:
            columns: Columns to lag (None for all)

        Returns:
            DataFrame with lagged features
        """
        if columns is None:
            columns = self.df.columns.tolist()

        features = pd.DataFrame(index=self.df.index)

        for col in columns:
            if col in self.df.columns:
                for lag in self.lag_periods:
                    features[f"{col}_lag_{lag}"] = self.df[col].shift(lag)

        return features

    def create_changes(
        self,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Create change features (current - lagged).

        Args:
            columns: Columns to compute changes for

        Returns:
            DataFrame with change features
        """
        if columns is None:
            columns = self.df.columns.tolist()

        features = pd.DataFrame(index=self.df.index)

        for col in columns:
            if col in self.df.columns:
                for lag in self.lag_periods:
                    features[f"{col}_change_{lag}"] = (
                        self.df[col] - self.df[col].shift(lag)
                    )

        return features


class FeatureProcessor(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible feature processor.

    Handles scaling, missing values, and outliers.
    """

    def __init__(
        self,
        scaling: Literal["standard", "robust", "minmax", "none"] = "robust",
        handle_inf: bool = True,
        clip_outliers: float | None = 5.0,
        fill_method: str = "ffill",
    ) -> None:
        """
        Initialize the processor.

        Args:
            scaling: Scaling method
            handle_inf: Replace infinite values with NaN
            clip_outliers: Std deviations for outlier clipping
            fill_method: Method for filling NaN
        """
        self.scaling = scaling
        self.handle_inf = handle_inf
        self.clip_outliers = clip_outliers
        self.fill_method = fill_method

        self._scaler = None
        self._feature_names = None
        self._fitted = False

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureProcessor":
        """Fit the processor."""
        X = self._preprocess(X.copy())
        self._feature_names = X.columns.tolist()

        if self.scaling == "standard":
            self._scaler = StandardScaler()
        elif self.scaling == "robust":
            self._scaler = RobustScaler()
        elif self.scaling == "minmax":
            self._scaler = MinMaxScaler()
        else:
            self._scaler = None

        if self._scaler is not None:
            self._scaler.fit(X)

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features."""
        if not self._fitted:
            raise ValueError("Processor not fitted. Call fit() first.")

        X = self._preprocess(X.copy())

        # Ensure same columns
        X = X.reindex(columns=self._feature_names, fill_value=0)

        if self._scaler is not None:
            X_scaled = self._scaler.transform(X)
            X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing steps."""
        # Handle infinite values
        if self.handle_inf:
            X = X.replace([np.inf, -np.inf], np.nan)

        # Clip outliers
        if self.clip_outliers is not None:
            for col in X.columns:
                mean = X[col].mean()
                std = X[col].std()
                lower = mean - self.clip_outliers * std
                upper = mean + self.clip_outliers * std
                X[col] = X[col].clip(lower=lower, upper=upper)

        # Fill missing values
        if self.fill_method == "ffill":
            X = X.ffill().bfill()
        elif self.fill_method == "mean":
            X = X.fillna(X.mean())
        elif self.fill_method == "zero":
            X = X.fillna(0)

        return X

    def get_feature_names_out(self) -> list[str]:
        """Get output feature names."""
        return self._feature_names or []


class FeaturePipeline:
    """
    Complete feature engineering pipeline with proper fit/transform separation
    to prevent data leakage.

    CRITICAL: This pipeline enforces strict separation between fit() and transform()
    to prevent look-ahead bias in backtesting. The processor is ONLY fitted on
    training data, and the fitted parameters are applied to test data.

    Combines technical, statistical, and cross-sectional features
    with preprocessing.
    """

    def __init__(
        self,
        ma_periods: list[int] | None = None,
        rsi_periods: list[int] | None = None,
        return_periods: list[int] | None = None,
        lag_periods: list[int] | None = None,
        scaling: str = "robust",
        strict_leakage_check: bool = True,
    ) -> None:
        """
        Initialize the pipeline.

        Args:
            ma_periods: Moving average periods
            rsi_periods: RSI periods
            return_periods: Return periods
            lag_periods: Lag periods
            scaling: Feature scaling method
            strict_leakage_check: If True, warns on direct generate_features() calls
        """
        self.ma_periods = ma_periods or [5, 10, 20, 50, 100, 200]
        self.rsi_periods = rsi_periods or [7, 14, 21]
        self.return_periods = return_periods or [1, 5, 10, 20, 60]
        self.lag_periods = lag_periods or [1, 2, 5, 10]
        self.scaling = scaling
        self.strict_leakage_check = strict_leakage_check

        self.processor = FeatureProcessor(scaling=scaling)
        self._feature_names: list[str] = []
        self._is_fitted: bool = False
        self._internal_call: bool = False  # Flag for internal method calls
        self._direct_call_count: int = 0  # Track direct calls for monitoring

        # Track max lookback for purge gap calculation
        self._max_lookback: int = max(self.ma_periods) if self.ma_periods else 200

    @property
    def max_lookback(self) -> int:
        """
        Get maximum lookback period used in feature calculation.

        This is critical for calculating proper purge_gap in cross-validation:
            purge_gap = prediction_horizon + max_lookback + buffer
        """
        return self._max_lookback

    @property
    def is_fitted(self) -> bool:
        """Check if pipeline has been fitted."""
        return self._is_fitted

    def generate_features(
        self,
        df: pd.DataFrame,
        include_technical: bool = True,
        include_statistical: bool = True,
        include_lagged: bool = True,
        include_microstructure: bool = True,
        include_time: bool = True,
    ) -> pd.DataFrame:
        """
        Generate all features for a single stock.

        This method generates technical, statistical, time-based, and microstructure
        features from OHLCV data.

        IMPORTANT: For proper leakage prevention, use fit() and transform()
        methods instead of calling generate_features() directly.

        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
            include_technical: Include technical indicators
            include_statistical: Include statistical features
            include_lagged: Include lagged features
            include_microstructure: Include microstructure features (requires order book data)
            include_time: Include time-based cyclical features

        Returns:
            DataFrame with all generated features
        """
        # Leakage prevention warning for direct calls
        if self.strict_leakage_check and not self._internal_call:
            self._direct_call_count += 1
            if self._direct_call_count <= 3:
                logger.warning(
                    "Direct call to generate_features() detected. "
                    "For proper leakage prevention, use fit() on training data "
                    "and transform() on test data instead."
                )

        features = pd.DataFrame(index=df.index)

        # 1. Time-based cyclical features
        if include_time:
            try:
                time_encoder = TimeCyclicalEncoder()
                time_features = time_encoder.transform(df)

                # Only add newly generated sin/cos columns
                new_time_cols = [c for c in time_features.columns if c not in df.columns]
                if new_time_cols:
                    features = pd.concat([features, time_features[new_time_cols]], axis=1)
            except Exception as e:
                logger.warning(f"Failed to generate time features: {e}")

        # 2. Technical indicators
        if include_technical:
            try:
                tech = TechnicalIndicators(df)
                tech_features = tech.generate_all_features(
                    ma_periods=self.ma_periods,
                    rsi_periods=self.rsi_periods,
                )
                features = pd.concat([features, tech_features], axis=1)
            except Exception as e:
                logger.warning(f"Failed to generate technical features: {e}")

        # 3. Microstructure / Order Book features
        if include_microstructure:
            required_cols = ['bid_size', 'ask_size', 'bid_price', 'ask_price']
            if all(col in df.columns for col in required_cols):
                try:
                    # Order Book Imbalance (OBI)
                    features['obi'] = OrderBookDynamics.calculate_obi(
                        df['bid_size'], df['ask_size']
                    )

                    # Weighted Mid Price
                    wmp = OrderBookDynamics.calculate_wmp(
                        df['bid_price'], df['bid_size'],
                        df['ask_price'], df['ask_size']
                    )

                    # Mid Price Divergence
                    mid_price = (df['bid_price'] + df['ask_price']) / 2
                    features['wmp_divergence'] = OrderBookDynamics.calculate_mid_price_divergence(
                        mid_price, wmp
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate microstructure features: {e}")

        # 4. Statistical features
        if include_statistical:
            try:
                stat = StatisticalFeatures(df)
                stat_features = stat.generate_all()
                features = pd.concat([features, stat_features], axis=1)
            except Exception as e:
                logger.warning(f"Failed to generate statistical features: {e}")

        # 5. Lagged features
        if include_lagged:
            try:
                # Select key features for lagging
                key_cols = [
                    c for c in features.columns
                    if any(kw in c for kw in ["return", "rsi", "macd", "volume", "obi", "wmp"])
                ][:15]

                if key_cols:
                    lag_gen = LaggedFeatures(features[key_cols], self.lag_periods)
                    lag_features = lag_gen.create_lags()
                    features = pd.concat([features, lag_features], axis=1)
            except Exception as e:
                logger.warning(f"Failed to generate lagged features: {e}")

        self._feature_names = features.columns.tolist()
        logger.info(f"Generated {len(features.columns)} total features")

        return features

    def fit(
        self,
        df: pd.DataFrame,
        **kwargs,
    ) -> "FeaturePipeline":
        """
        Fit the pipeline on TRAINING data only.

        CRITICAL: This method should ONLY be called with training data.
        The processor learns scaling parameters (mean, std, etc.) from
        this data, which will then be applied to test data via transform().

        Args:
            df: OHLCV DataFrame (TRAINING DATA ONLY)
            **kwargs: Arguments for generate_features

        Returns:
            self (for method chaining)
        """
        self._internal_call = True
        try:
            features = self.generate_features(df, **kwargs)
            self.processor.fit(features)
            self._is_fitted = True
            logger.info(f"Pipeline fitted on {len(features)} training samples")
        finally:
            self._internal_call = False
        return self

    def fit_transform(
        self,
        df: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fit and transform TRAINING data.

        WARNING: This method should ONLY be used with training data.
        For test/validation data, use transform() after fitting on training data.

        DEPRECATED PATTERN: Do not use fit_transform() on the entire dataset
        before train/test split. This causes data leakage.

        CORRECT USAGE:
            # Split data first
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]

            # Fit on training, transform both
            train_features = pipeline.fit_transform(train_df)  # OK
            test_features = pipeline.transform(test_df)        # Uses fitted params

        Args:
            df: OHLCV DataFrame (TRAINING DATA ONLY)
            **kwargs: Arguments for generate_features

        Returns:
            Processed feature DataFrame
        """
        self._internal_call = True
        try:
            features = self.generate_features(df, **kwargs)
            self._is_fitted = True
            return self.processor.fit_transform(features)
        finally:
            self._internal_call = False

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform new data using fitted pipeline.

        CRITICAL: Pipeline must be fitted first via fit() or fit_transform()
        on training data. This method applies the fitted scaling parameters
        to new data WITHOUT refitting (preventing data leakage).

        Args:
            df: OHLCV DataFrame (can be test/validation data)
            **kwargs: Arguments for generate_features

        Returns:
            Processed feature DataFrame

        Raises:
            ValueError: If pipeline has not been fitted
        """
        if not self._is_fitted:
            raise ValueError(
                "Pipeline not fitted. Call fit() or fit_transform() on training data first. "
                "This error prevents data leakage by ensuring scaling parameters are "
                "learned only from training data."
            )
        self._internal_call = True
        try:
            features = self.generate_features(df, **kwargs)
            return self.processor.transform(features)
        finally:
            self._internal_call = False

    @property
    def feature_names(self) -> list[str]:
        """Get list of feature names."""
        return self._feature_names

    def get_purge_gap_recommendation(self, prediction_horizon: int, buffer: int = 10) -> int:
        """
        Calculate recommended purge gap for cross-validation.

        The purge gap should be at least: prediction_horizon + max_lookback + buffer
        to prevent information leakage from features that use historical data.

        Args:
            prediction_horizon: Number of bars ahead being predicted
            buffer: Safety buffer (default: 10)

        Returns:
            Recommended minimum purge gap in bars
        """
        recommended = prediction_horizon + self._max_lookback + buffer
        logger.info(
            f"Recommended purge_gap={recommended} "
            f"(horizon={prediction_horizon} + lookback={self._max_lookback} + buffer={buffer})"
        )
        return recommended


def create_feature_matrix(
    data: dict[str, pd.DataFrame],
    pipeline: FeaturePipeline | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Create feature matrices for multiple stocks.

    Args:
        data: Dictionary mapping symbols to OHLCV DataFrames
        pipeline: FeaturePipeline instance (creates new if None)

    Returns:
        Dictionary mapping symbols to feature DataFrames
    """
    if pipeline is None:
        pipeline = FeaturePipeline()

    features = {}
    for symbol, df in data.items():
        logger.info(f"Generating features for {symbol}")
        features[symbol] = pipeline.generate_features(df)

    return features
