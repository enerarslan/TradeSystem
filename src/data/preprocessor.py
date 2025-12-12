"""
Institutional-Grade Data Preprocessor
JPMorgan-Level Data Cleaning and Transformation

Features:
- Missing data handling
- Outlier detection and treatment (Z-score, IQR, MAD, Winsorization)
- Corporate actions adjustment
- Time alignment and resampling
- Data quality scoring
- Feature Neutralization (beta neutralization via OLS regression)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import warnings

from ..utils.logger import get_logger, get_audit_logger
from ..utils.helpers import validate_ohlcv, safe_divide


logger = get_logger(__name__)
audit_logger = get_audit_logger()


class FillMethod(Enum):
    """Methods for filling missing data"""
    FFILL = "ffill"
    BFILL = "bfill"
    INTERPOLATE = "interpolate"
    MEAN = "mean"
    MEDIAN = "median"
    ZERO = "zero"
    DROP = "drop"


class OutlierMethod(Enum):
    """Methods for outlier detection"""
    ZSCORE = "zscore"
    IQR = "iqr"
    MAD = "mad"
    PERCENTILE = "percentile"
    ISOLATION_FOREST = "isolation_forest"
    WINSORIZE = "winsorize"  # Winsorization - cap at percentiles instead of removing


@dataclass
class DataQualityReport:
    """Data quality assessment report"""
    symbol: str
    total_rows: int
    date_range: Tuple[datetime, datetime]
    missing_values: Dict[str, int]
    missing_pct: float
    outliers_detected: int
    gaps_detected: int
    duplicate_rows: int
    quality_score: float  # 0-100
    issues: List[str]
    timestamp: datetime


class DataCleaner:
    """
    Professional data cleaning utilities.

    Handles:
    - Missing value detection and imputation
    - Outlier detection and treatment
    - Duplicate removal
    - Type conversion and validation
    """

    def __init__(
        self,
        max_missing_pct: float = 0.05,
        outlier_std_threshold: float = 5.0,
        min_data_points: int = 100
    ):
        """
        Initialize DataCleaner.

        Args:
            max_missing_pct: Maximum allowed missing percentage
            outlier_std_threshold: Standard deviations for outlier detection
            min_data_points: Minimum required data points
        """
        self.max_missing_pct = max_missing_pct
        self.outlier_std_threshold = outlier_std_threshold
        self.min_data_points = min_data_points

    def clean(
        self,
        df: pd.DataFrame,
        fill_method: FillMethod = FillMethod.FFILL,
        handle_outliers: bool = True,
        outlier_method: OutlierMethod = OutlierMethod.WINSORIZE  # Changed from ZSCORE per AFML
    ) -> pd.DataFrame:
        """
        Clean DataFrame with standard pipeline.

        Args:
            df: Input DataFrame
            fill_method: Method for filling missing values
            handle_outliers: Whether to handle outliers
            outlier_method: Method for outlier detection
                           POLICY: Default is WINSORIZE (not DROP) per AFML recommendations.
                           Financial tail events contain critical information and should
                           be capped, not removed. Dropping outliers destroys information
                           limits and biases models toward normal market conditions.

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # 1. Remove duplicates
        df = self._remove_duplicates(df)

        # 2. Handle missing values
        df = self._handle_missing(df, fill_method)

        # 3. Handle outliers
        if handle_outliers:
            df = self._handle_outliers(df, outlier_method)

        # 4. Ensure correct dtypes
        df = self._enforce_dtypes(df)

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        initial_len = len(df)

        # Remove exact duplicates
        df = df[~df.index.duplicated(keep='last')]

        removed = initial_len - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")

        return df

    def _handle_missing(
        self,
        df: pd.DataFrame,
        method: FillMethod
    ) -> pd.DataFrame:
        """Handle missing values"""
        missing_count = df.isnull().sum().sum()

        if missing_count == 0:
            return df

        logger.debug(f"Handling {missing_count} missing values with {method.value}")

        if method == FillMethod.FFILL:
            df = df.ffill(limit=10)  # Limit forward fill to 10 periods
        elif method == FillMethod.BFILL:
            df = df.bfill(limit=10)
        elif method == FillMethod.INTERPOLATE:
            df = df.interpolate(method='time', limit=10)
        elif method == FillMethod.MEAN:
            df = df.fillna(df.mean())
        elif method == FillMethod.MEDIAN:
            df = df.fillna(df.median())
        elif method == FillMethod.ZERO:
            df = df.fillna(0)
        elif method == FillMethod.DROP:
            df = df.dropna()

        return df

    def _handle_outliers(
        self,
        df: pd.DataFrame,
        method: OutlierMethod
    ) -> pd.DataFrame:
        """Detect and handle outliers"""
        price_cols = ['open', 'high', 'low', 'close']

        for col in price_cols:
            if col not in df.columns:
                continue

            if method == OutlierMethod.WINSORIZE:
                # Winsorization - cap values at percentiles instead of removing
                df[col] = self._winsorize(df[col])
            else:
                if method == OutlierMethod.ZSCORE:
                    outlier_mask = self._detect_zscore_outliers(df[col])
                elif method == OutlierMethod.IQR:
                    outlier_mask = self._detect_iqr_outliers(df[col])
                elif method == OutlierMethod.MAD:
                    outlier_mask = self._detect_mad_outliers(df[col])
                else:
                    outlier_mask = pd.Series(False, index=df.index)

                # Replace outliers with interpolated values
                if outlier_mask.any():
                    outlier_count = outlier_mask.sum()
                    logger.debug(f"Found {outlier_count} outliers in {col}")

                    df.loc[outlier_mask, col] = np.nan
                    df[col] = df[col].interpolate(method='linear')

        return df

    def _winsorize(
        self,
        series: pd.Series,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0
    ) -> pd.Series:
        """
        Winsorize series by capping values at percentiles.

        Unlike deletion, Winsorization preserves temporal continuity by
        capping extreme values to the percentile bounds.

        Args:
            series: Input series
            lower_percentile: Lower percentile bound (default 1st percentile)
            upper_percentile: Upper percentile bound (default 99th percentile)

        Returns:
            Winsorized series
        """
        lower_bound = series.quantile(lower_percentile / 100)
        upper_bound = series.quantile(upper_percentile / 100)

        return series.clip(lower=lower_bound, upper=upper_bound)

    def _detect_zscore_outliers(
        self,
        series: pd.Series,
        threshold: Optional[float] = None
    ) -> pd.Series:
        """Detect outliers using Z-score method"""
        threshold = threshold or self.outlier_std_threshold

        mean = series.mean()
        std = series.std()

        if std == 0:
            return pd.Series(False, index=series.index)

        z_scores = np.abs((series - mean) / std)
        return z_scores > threshold

    def _detect_iqr_outliers(
        self,
        series: pd.Series,
        k: float = 1.5
    ) -> pd.Series:
        """Detect outliers using IQR method"""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr

        return (series < lower_bound) | (series > upper_bound)

    def _detect_mad_outliers(
        self,
        series: pd.Series,
        threshold: float = 3.5
    ) -> pd.Series:
        """Detect outliers using Median Absolute Deviation"""
        median = series.median()
        mad = np.median(np.abs(series - median))

        if mad == 0:
            return pd.Series(False, index=series.index)

        modified_z_scores = 0.6745 * (series - median) / mad
        return np.abs(modified_z_scores) > threshold

    def _enforce_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure correct data types"""
        dtype_map = {
            'open': np.float64,
            'high': np.float64,
            'low': np.float64,
            'close': np.float64,
            'volume': np.float64
        }

        for col, dtype in dtype_map.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

        return df


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline.

    Features:
    - Multi-symbol preprocessing
    - Time alignment across symbols
    - Gap detection and handling
    - Corporate actions adjustment
    - Data quality reporting
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize DataPreprocessor.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Extract config values
        self.max_missing_pct = self.config.get('max_missing_pct', 0.05)
        self.outlier_std = self.config.get('outlier_std_threshold', 5.0)
        self.gap_fill_method = self.config.get('gap_fill_method', 'ffill')
        self.min_data_points = self.config.get('min_data_points', 1000)

        self.cleaner = DataCleaner(
            max_missing_pct=self.max_missing_pct,
            outlier_std_threshold=self.outlier_std
        )

        logger.info("DataPreprocessor initialized")

    def preprocess(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN"
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Preprocess single symbol data.

        Args:
            df: Raw DataFrame
            symbol: Symbol name for reporting

        Returns:
            Tuple of (processed DataFrame, quality report)
        """
        logger.debug(f"Preprocessing {symbol}...")

        # Generate quality report before cleaning
        report = self._generate_quality_report(df, symbol)

        # Clean data
        df_clean = self.cleaner.clean(
            df,
            fill_method=FillMethod(self.gap_fill_method),
            handle_outliers=True
        )

        # Additional preprocessing
        df_clean = self._add_derived_columns(df_clean)

        # Validate OHLC relationships
        df_clean = self._fix_ohlc_relationships(df_clean)

        # Update report quality score
        report.quality_score = self._calculate_quality_score(df_clean, report)

        logger.info(
            f"Preprocessed {symbol}: {len(df_clean)} rows, "
            f"quality score: {report.quality_score:.1f}"
        )

        return df_clean, report

    def preprocess_multi(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, DataQualityReport]]:
        """
        Preprocess multiple symbols.

        Args:
            data: Dictionary of symbol DataFrames

        Returns:
            Tuple of (processed data dict, reports dict)
        """
        processed = {}
        reports = {}

        for symbol, df in data.items():
            try:
                df_clean, report = self.preprocess(df, symbol)
                processed[symbol] = df_clean
                reports[symbol] = report
            except Exception as e:
                logger.error(f"Failed to preprocess {symbol}: {e}")

        return processed, reports

    def align_timestamps(
        self,
        data: Dict[str, pd.DataFrame],
        method: str = 'intersection'
    ) -> Dict[str, pd.DataFrame]:
        """
        Align timestamps across all symbols.

        Args:
            data: Dictionary of symbol DataFrames
            method: 'intersection' (common timestamps) or 'union' (all timestamps)

        Returns:
            Aligned DataFrames
        """
        if not data:
            return {}

        # Get all indices
        indices = [df.index for df in data.values()]

        if method == 'intersection':
            # Find common timestamps
            common_idx = indices[0]
            for idx in indices[1:]:
                common_idx = common_idx.intersection(idx)

            logger.info(f"Aligned to {len(common_idx)} common timestamps")

            return {
                symbol: df.loc[common_idx]
                for symbol, df in data.items()
            }

        elif method == 'union':
            # Union of all timestamps
            union_idx = indices[0]
            for idx in indices[1:]:
                union_idx = union_idx.union(idx)

            union_idx = union_idx.sort_values()

            aligned = {}
            for symbol, df in data.items():
                aligned[symbol] = df.reindex(union_idx).ffill(limit=3)

            return aligned

        else:
            raise ValueError(f"Unknown alignment method: {method}")

    def resample(
        self,
        df: pd.DataFrame,
        target_timeframe: str
    ) -> pd.DataFrame:
        """
        Resample OHLCV data to different timeframe.

        Args:
            df: Source DataFrame
            target_timeframe: Target timeframe (e.g., '1H', '4H', '1D')

        Returns:
            Resampled DataFrame
        """
        # Map common timeframe strings to pandas offset
        tf_map = {
            '1min': '1min',
            '5min': '5min',
            '15min': '15min',
            '30min': '30min',
            '1H': '1H',
            '4H': '4H',
            '1D': '1D',
            '1W': '1W'
        }

        freq = tf_map.get(target_timeframe, target_timeframe)

        resampled = df.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return resampled

    def detect_gaps(
        self,
        df: pd.DataFrame,
        expected_freq: str = '15min'
    ) -> pd.DataFrame:
        """
        Detect data gaps in time series.

        Args:
            df: Input DataFrame
            expected_freq: Expected data frequency

        Returns:
            DataFrame with gap information
        """
        # Calculate time differences
        time_diffs = df.index.to_series().diff()

        # Expected gap (accounting for market hours)
        expected_gap = pd.Timedelta(expected_freq)

        # Find gaps larger than expected
        # Allow 2x expected for market close/open
        gap_threshold = expected_gap * 2

        gaps = time_diffs[time_diffs > gap_threshold]

        gap_info = []
        for gap_end, gap_duration in gaps.items():
            gap_start = gap_end - gap_duration
            gap_info.append({
                'gap_start': gap_start,
                'gap_end': gap_end,
                'duration': gap_duration,
                'expected_bars': int(gap_duration / expected_gap) - 1
            })

        return pd.DataFrame(gap_info)

    def _generate_quality_report(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> DataQualityReport:
        """Generate data quality report"""
        issues = []

        # Missing values
        missing_values = df.isnull().sum().to_dict()
        total_cells = len(df) * len(df.columns)
        missing_pct = sum(missing_values.values()) / total_cells if total_cells > 0 else 0

        if missing_pct > self.max_missing_pct:
            issues.append(f"High missing value percentage: {missing_pct:.2%}")

        # Duplicates
        duplicate_rows = df.index.duplicated().sum()
        if duplicate_rows > 0:
            issues.append(f"Found {duplicate_rows} duplicate timestamps")

        # Data gaps
        gaps = self.detect_gaps(df)
        gaps_detected = len(gaps)
        if gaps_detected > 0:
            issues.append(f"Found {gaps_detected} data gaps")

        # Outliers (rough estimate)
        outliers_detected = 0
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                outlier_mask = self.cleaner._detect_zscore_outliers(df[col])
                outliers_detected += outlier_mask.sum()

        if outliers_detected > 0:
            issues.append(f"Detected {outliers_detected} potential outliers")

        # Date range
        date_range = (df.index.min(), df.index.max())

        # Minimum data check
        if len(df) < self.min_data_points:
            issues.append(f"Insufficient data: {len(df)} < {self.min_data_points}")

        return DataQualityReport(
            symbol=symbol,
            total_rows=len(df),
            date_range=date_range,
            missing_values=missing_values,
            missing_pct=missing_pct,
            outliers_detected=outliers_detected,
            gaps_detected=gaps_detected,
            duplicate_rows=duplicate_rows,
            quality_score=0.0,  # Calculated later
            issues=issues,
            timestamp=datetime.utcnow()
        )

    def _calculate_quality_score(
        self,
        df: pd.DataFrame,
        report: DataQualityReport
    ) -> float:
        """Calculate data quality score (0-100)"""
        score = 100.0

        # Penalize missing values
        score -= report.missing_pct * 200  # Max -20 for 10% missing

        # Penalize gaps
        gap_penalty = min(report.gaps_detected * 2, 20)
        score -= gap_penalty

        # Penalize outliers
        outlier_pct = report.outliers_detected / (report.total_rows * 4)  # 4 price columns
        score -= min(outlier_pct * 100, 15)

        # Penalize duplicates
        dup_pct = report.duplicate_rows / report.total_rows if report.total_rows > 0 else 0
        score -= min(dup_pct * 50, 10)

        # Penalize insufficient data
        if report.total_rows < self.min_data_points:
            score -= 20

        return max(0, min(100, score))

    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add commonly used derived columns"""
        # Typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

        # Price range
        df['range'] = df['high'] - df['low']

        # Body size (for candlestick analysis)
        df['body'] = df['close'] - df['open']
        df['body_pct'] = safe_divide(df['body'], df['open'])

        # Upper/Lower shadows
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']

        return df

    def _fix_ohlc_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix any OHLC relationship violations"""
        # Ensure high >= open, close
        df['high'] = df[['high', 'open', 'close']].max(axis=1)

        # Ensure low <= open, close
        df['low'] = df[['low', 'open', 'close']].min(axis=1)

        return df


class CorporateActionsAdjuster:
    """
    Handle corporate actions adjustments.

    Adjusts for:
    - Stock splits
    - Dividends
    - Mergers/Spinoffs
    """

    def __init__(self):
        self._adjustments_cache: Dict[str, pd.DataFrame] = {}

    def adjust_for_splits(
        self,
        df: pd.DataFrame,
        splits: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Adjust prices for stock splits.

        Args:
            df: Price DataFrame
            splits: List of split events with 'date' and 'ratio'

        Returns:
            Adjusted DataFrame
        """
        df = df.copy()

        for split in sorted(splits, key=lambda x: x['date'], reverse=True):
            split_date = pd.to_datetime(split['date'])
            ratio = split['ratio']

            mask = df.index < split_date

            # Adjust prices
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df.loc[mask, col] = df.loc[mask, col] / ratio

            # Adjust volume (inverse)
            if 'volume' in df.columns:
                df.loc[mask, 'volume'] = df.loc[mask, 'volume'] * ratio

        return df

    def adjust_for_dividends(
        self,
        df: pd.DataFrame,
        dividends: List[Dict[str, Any]],
        method: str = 'proportional'
    ) -> pd.DataFrame:
        """
        Adjust prices for dividend payments.

        Args:
            df: Price DataFrame
            dividends: List of dividend events with 'date' and 'amount'
            method: 'proportional' or 'absolute'

        Returns:
            Adjusted DataFrame
        """
        df = df.copy()

        for div in sorted(dividends, key=lambda x: x['date'], reverse=True):
            div_date = pd.to_datetime(div['date'])
            amount = div['amount']

            mask = df.index < div_date

            if method == 'proportional':
                # Get price on div date
                if div_date in df.index:
                    price_on_date = df.loc[div_date, 'close']
                    adj_factor = 1 - (amount / price_on_date)
                else:
                    # Estimate using nearest price
                    nearest_idx = df.index.get_indexer([div_date], method='nearest')[0]
                    price_on_date = df.iloc[nearest_idx]['close']
                    adj_factor = 1 - (amount / price_on_date)

                for col in ['open', 'high', 'low', 'close']:
                    if col in df.columns:
                        df.loc[mask, col] = df.loc[mask, col] * adj_factor

            elif method == 'absolute':
                for col in ['open', 'high', 'low', 'close']:
                    if col in df.columns:
                        df.loc[mask, col] = df.loc[mask, col] - amount

        return df


class FeatureNeutralizer:
    """
    Feature Neutralization using OLS regression.

    Orthogonalizes features against market factors (e.g., SPY returns)
    to remove systematic market exposure.

    Formula: Feature_Neutral = Feature_Raw - (Beta * Market_Return)

    This helps create alpha factors that are independent of market direction.
    """

    def __init__(
        self,
        market_benchmark: str = 'SPY',
        window: Optional[int] = None,
        min_periods: int = 20
    ):
        """
        Initialize FeatureNeutralizer.

        Args:
            market_benchmark: Symbol for market benchmark (default: SPY)
            window: Rolling window for beta calculation (None = expanding)
            min_periods: Minimum periods for calculation
        """
        self.market_benchmark = market_benchmark
        self.window = window
        self.min_periods = min_periods
        self._market_returns: Optional[pd.Series] = None
        self._fitted_betas: Dict[str, float] = {}

        logger.info(f"FeatureNeutralizer initialized with benchmark: {market_benchmark}")

    def set_market_returns(
        self,
        market_data: Union[pd.Series, pd.DataFrame]
    ) -> 'FeatureNeutralizer':
        """
        Set market returns for neutralization.

        Args:
            market_data: Either returns series or DataFrame with 'close' column

        Returns:
            Self for chaining
        """
        if isinstance(market_data, pd.DataFrame):
            if 'close' in market_data.columns:
                self._market_returns = market_data['close'].pct_change()
            else:
                raise ValueError("DataFrame must have 'close' column")
        else:
            self._market_returns = market_data

        logger.info(f"Market returns set: {len(self._market_returns)} observations")
        return self

    def calculate_beta(
        self,
        feature: pd.Series,
        market_returns: Optional[pd.Series] = None
    ) -> float:
        """
        Calculate beta of feature against market using OLS regression.

        Args:
            feature: Feature series
            market_returns: Market returns (uses stored if not provided)

        Returns:
            Beta coefficient
        """
        market_returns = market_returns if market_returns is not None else self._market_returns

        if market_returns is None:
            raise ValueError("Market returns not set. Call set_market_returns() first.")

        # Align indices
        common_idx = feature.index.intersection(market_returns.index)
        f = feature.loc[common_idx].dropna()
        m = market_returns.loc[common_idx].dropna()

        # Use intersection after dropna
        common_idx = f.index.intersection(m.index)
        f = f.loc[common_idx]
        m = m.loc[common_idx]

        if len(f) < self.min_periods:
            logger.warning(f"Insufficient data for beta calculation: {len(f)} < {self.min_periods}")
            return 0.0

        # OLS regression: feature = alpha + beta * market + epsilon
        # Beta = Cov(feature, market) / Var(market)
        covariance = np.cov(f.values, m.values)[0, 1]
        variance = np.var(m.values)

        if variance == 0:
            return 0.0

        beta = covariance / variance
        return beta

    def calculate_rolling_beta(
        self,
        feature: pd.Series,
        market_returns: Optional[pd.Series] = None,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate rolling beta.

        Args:
            feature: Feature series
            market_returns: Market returns
            window: Rolling window (uses instance default if not provided)

        Returns:
            Series of rolling betas
        """
        market_returns = market_returns if market_returns is not None else self._market_returns
        window = window or self.window

        if market_returns is None:
            raise ValueError("Market returns not set.")

        # Align data
        common_idx = feature.index.intersection(market_returns.index)
        f = feature.loc[common_idx]
        m = market_returns.loc[common_idx]

        if window is None:
            # Expanding beta
            betas = pd.Series(index=f.index, dtype=float)
            for i in range(self.min_periods, len(f) + 1):
                f_window = f.iloc[:i]
                m_window = m.iloc[:i]
                cov = np.cov(f_window.dropna(), m_window.dropna())[0, 1]
                var = m_window.var()
                betas.iloc[i - 1] = cov / var if var > 0 else 0
        else:
            # Rolling beta
            def calc_beta(x, y):
                if len(x) < self.min_periods:
                    return np.nan
                cov = np.cov(x, y)[0, 1]
                var = np.var(y)
                return cov / var if var > 0 else 0

            betas = pd.Series(index=f.index, dtype=float)
            for i in range(window, len(f) + 1):
                f_window = f.iloc[i - window:i]
                m_window = m.iloc[i - window:i]
                valid_mask = f_window.notna() & m_window.notna()
                if valid_mask.sum() >= self.min_periods:
                    betas.iloc[i - 1] = calc_beta(
                        f_window[valid_mask].values,
                        m_window[valid_mask].values
                    )

        return betas

    def neutralize(
        self,
        feature: pd.Series,
        market_returns: Optional[pd.Series] = None,
        rolling: bool = False
    ) -> pd.Series:
        """
        Neutralize feature against market.

        Feature_Neutral = Feature_Raw - (Beta * Market_Return)

        Args:
            feature: Feature series to neutralize
            market_returns: Market returns (uses stored if not provided)
            rolling: Use rolling beta (True) or static beta (False)

        Returns:
            Neutralized feature series
        """
        market_returns = market_returns if market_returns is not None else self._market_returns

        if market_returns is None:
            raise ValueError("Market returns not set.")

        # Align data
        common_idx = feature.index.intersection(market_returns.index)
        f = feature.loc[common_idx]
        m = market_returns.loc[common_idx]

        if rolling:
            betas = self.calculate_rolling_beta(f, m)
            neutralized = f - (betas * m)
        else:
            beta = self.calculate_beta(f, m)
            neutralized = f - (beta * m)
            self._fitted_betas[feature.name or 'feature'] = beta
            logger.debug(f"Neutralized feature with beta={beta:.4f}")

        return neutralized

    def neutralize_dataframe(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        market_returns: Optional[pd.Series] = None,
        rolling: bool = False,
        suffix: str = '_neutral'
    ) -> pd.DataFrame:
        """
        Neutralize multiple features in a DataFrame.

        Args:
            df: DataFrame with features
            columns: Columns to neutralize (default: all numeric)
            market_returns: Market returns
            rolling: Use rolling beta
            suffix: Suffix for neutralized column names

        Returns:
            DataFrame with neutralized features
        """
        market_returns = market_returns if market_returns is not None else self._market_returns

        if market_returns is None:
            raise ValueError("Market returns not set.")

        result = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                continue

            try:
                neutralized = self.neutralize(df[col], market_returns, rolling)
                result[f'{col}{suffix}'] = neutralized
            except Exception as e:
                logger.warning(f"Failed to neutralize {col}: {e}")

        return result

    def fit(
        self,
        features_df: pd.DataFrame,
        market_returns: pd.Series,
        columns: Optional[List[str]] = None
    ) -> 'FeatureNeutralizer':
        """
        Fit neutralizer by calculating betas for all features.

        Args:
            features_df: DataFrame with features
            market_returns: Market returns series
            columns: Columns to fit (default: all numeric)

        Returns:
            Self for chaining
        """
        self.set_market_returns(market_returns)

        if columns is None:
            columns = features_df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in features_df.columns:
                continue

            beta = self.calculate_beta(features_df[col], market_returns)
            self._fitted_betas[col] = beta

        logger.info(f"Fitted betas for {len(self._fitted_betas)} features")
        return self

    def transform(
        self,
        features_df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        suffix: str = '_neutral'
    ) -> pd.DataFrame:
        """
        Transform features using fitted betas.

        Args:
            features_df: DataFrame with features
            columns: Columns to transform
            suffix: Suffix for neutralized columns

        Returns:
            Transformed DataFrame
        """
        if self._market_returns is None:
            raise ValueError("Must fit neutralizer first")

        result = features_df.copy()

        if columns is None:
            columns = list(self._fitted_betas.keys())

        for col in columns:
            if col not in features_df.columns:
                continue
            if col not in self._fitted_betas:
                continue

            beta = self._fitted_betas[col]
            common_idx = features_df.index.intersection(self._market_returns.index)

            neutralized = features_df.loc[common_idx, col] - (beta * self._market_returns.loc[common_idx])
            result.loc[common_idx, f'{col}{suffix}'] = neutralized

        return result

    def get_betas(self) -> Dict[str, float]:
        """Get fitted betas"""
        return self._fitted_betas.copy()


class RobustOutlierHandler:
    """
    Robust outlier detection and handling.

    Provides multiple methods for outlier treatment that preserve
    temporal continuity:
    - Winsorization (1st and 99th percentile capping)
    - MAD-based detection
    - Rolling statistics
    """

    def __init__(
        self,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
        mad_threshold: float = 3.5,
        rolling_window: int = 20
    ):
        """
        Initialize RobustOutlierHandler.

        Args:
            lower_percentile: Lower percentile for winsorization
            upper_percentile: Upper percentile for winsorization
            mad_threshold: Threshold for MAD-based detection
            rolling_window: Window for rolling statistics
        """
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.mad_threshold = mad_threshold
        self.rolling_window = rolling_window

    def winsorize(
        self,
        series: pd.Series,
        lower_percentile: Optional[float] = None,
        upper_percentile: Optional[float] = None
    ) -> pd.Series:
        """
        Winsorize series by capping at percentiles.

        Args:
            series: Input series
            lower_percentile: Lower bound percentile
            upper_percentile: Upper bound percentile

        Returns:
            Winsorized series
        """
        lower = lower_percentile or self.lower_percentile
        upper = upper_percentile or self.upper_percentile

        lower_bound = series.quantile(lower / 100)
        upper_bound = series.quantile(upper / 100)

        return series.clip(lower=lower_bound, upper=upper_bound)

    def mad_score(self, series: pd.Series) -> pd.Series:
        """
        Calculate Median Absolute Deviation score.

        MAD is more robust to outliers than standard deviation.

        Args:
            series: Input series

        Returns:
            MAD scores
        """
        median = series.median()
        mad = np.median(np.abs(series - median))

        if mad == 0:
            return pd.Series(0, index=series.index)

        # Modified Z-score using MAD
        return 0.6745 * (series - median) / mad

    def detect_mad_outliers(
        self,
        series: pd.Series,
        threshold: Optional[float] = None
    ) -> pd.Series:
        """
        Detect outliers using MAD.

        Args:
            series: Input series
            threshold: MAD threshold (default: instance setting)

        Returns:
            Boolean mask of outliers
        """
        threshold = threshold or self.mad_threshold
        mad_scores = self.mad_score(series)
        return np.abs(mad_scores) > threshold

    def rolling_winsorize(
        self,
        series: pd.Series,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Winsorize using rolling percentiles for time-series data.

        This adapts to changing market regimes.

        Args:
            series: Input series
            window: Rolling window size

        Returns:
            Rolling-winsorized series
        """
        window = window or self.rolling_window

        rolling_lower = series.rolling(window).quantile(self.lower_percentile / 100)
        rolling_upper = series.rolling(window).quantile(self.upper_percentile / 100)

        result = series.copy()
        result = result.clip(lower=rolling_lower, upper=rolling_upper)

        return result

    def handle_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'winsorize',
        rolling: bool = False
    ) -> pd.DataFrame:
        """
        Handle outliers in DataFrame.

        Args:
            df: Input DataFrame
            columns: Columns to process (default: numeric columns)
            method: 'winsorize', 'mad', or 'clip'
            rolling: Use rolling statistics

        Returns:
            DataFrame with handled outliers
        """
        result = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in result.columns:
                continue

            if method == 'winsorize':
                if rolling:
                    result[col] = self.rolling_winsorize(result[col])
                else:
                    result[col] = self.winsorize(result[col])

            elif method == 'mad':
                outliers = self.detect_mad_outliers(result[col])
                if outliers.any():
                    logger.debug(f"MAD detected {outliers.sum()} outliers in {col}")
                    # Replace with winsorized values
                    winsorized = self.winsorize(result[col])
                    result.loc[outliers, col] = winsorized.loc[outliers]

            elif method == 'clip':
                # Simple percentile clipping
                result[col] = self.winsorize(result[col])

        return result

    def detect_and_report(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Detect and report outliers without modifying data.

        Args:
            df: Input DataFrame
            columns: Columns to analyze

        Returns:
            Report dictionary with outlier statistics
        """
        report = {}

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                continue

            series = df[col].dropna()

            # MAD detection
            mad_outliers = self.detect_mad_outliers(series)

            # Percentile bounds
            lower_bound = series.quantile(self.lower_percentile / 100)
            upper_bound = series.quantile(self.upper_percentile / 100)
            percentile_outliers = (series < lower_bound) | (series > upper_bound)

            report[col] = {
                'mad_outliers': mad_outliers.sum(),
                'mad_outlier_pct': mad_outliers.mean() * 100,
                'percentile_outliers': percentile_outliers.sum(),
                'percentile_outlier_pct': percentile_outliers.mean() * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'median': series.median(),
                'mad': np.median(np.abs(series - series.median()))
            }

        return report


class InformationDrivenBars:
    """
    Information-Driven Bar Generation.

    Based on "Advances in Financial Machine Learning" by Marcos Lopez de Prado.

    Time bars have poor statistical properties:
    - Serial correlation
    - Non-normality
    - Heteroskedasticity

    Information-driven bars (Volume, Dollar, Tick) address these issues by
    sampling based on market activity rather than time:
    - Volume Bars: Sample every N shares traded
    - Dollar Bars: Sample every $X exchanged
    - Tick Bars: Sample every N transactions

    Benefits:
    - Returns closer to IID normal distribution
    - Better convergence for ML models
    - Adapts to market activity naturally
    - More events during high-information periods
    """

    def __init__(
        self,
        bar_type: str = "dollar",
        threshold: float = None,
        initial_estimate_bars: int = 100
    ):
        """
        Initialize InformationDrivenBars.

        Args:
            bar_type: "volume", "dollar", or "tick"
            threshold: Sampling threshold (auto-estimated if None)
            initial_estimate_bars: Bars to use for auto-estimation
        """
        self.bar_type = bar_type.lower()
        self.threshold = threshold
        self.initial_estimate_bars = initial_estimate_bars

        if self.bar_type not in ["volume", "dollar", "tick"]:
            raise ValueError(f"Unknown bar type: {bar_type}")

        logger.info(f"InformationDrivenBars initialized: type={bar_type}, threshold={threshold}")

    def estimate_threshold(
        self,
        data: pd.DataFrame,
        target_bars_per_day: int = 50
    ) -> float:
        """
        Estimate appropriate threshold based on data.

        Args:
            data: OHLCV DataFrame with tick or minute data
            target_bars_per_day: Desired number of bars per trading day

        Returns:
            Estimated threshold
        """
        if self.bar_type == "volume":
            # Total volume / (number of days * target bars per day)
            total_volume = data['volume'].sum()
            n_days = (data.index[-1] - data.index[0]).days or 1
            threshold = total_volume / (n_days * target_bars_per_day)

        elif self.bar_type == "dollar":
            # Dollar volume = price * volume
            data['dollar_volume'] = data['close'] * data['volume']
            total_dollar = data['dollar_volume'].sum()
            n_days = (data.index[-1] - data.index[0]).days or 1
            threshold = total_dollar / (n_days * target_bars_per_day)

        elif self.bar_type == "tick":
            # For tick bars, threshold is number of ticks
            n_ticks = len(data)
            n_days = (data.index[-1] - data.index[0]).days or 1
            threshold = n_ticks / (n_days * target_bars_per_day)

        logger.info(f"Estimated {self.bar_type} threshold: {threshold:,.2f}")

        return threshold

    def generate_bars(
        self,
        data: pd.DataFrame,
        threshold: float = None
    ) -> pd.DataFrame:
        """
        Generate information-driven bars from tick/minute data.

        Args:
            data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                  Index should be DatetimeIndex
            threshold: Sampling threshold (uses instance value if None)

        Returns:
            DataFrame with OHLCV bars sampled by information
        """
        threshold = threshold or self.threshold

        if threshold is None:
            threshold = self.estimate_threshold(data)
            self.threshold = threshold

        if self.bar_type == "volume":
            return self._generate_volume_bars(data, threshold)
        elif self.bar_type == "dollar":
            return self._generate_dollar_bars(data, threshold)
        elif self.bar_type == "tick":
            return self._generate_tick_bars(data, threshold)

    def _generate_volume_bars(
        self,
        data: pd.DataFrame,
        threshold: float
    ) -> pd.DataFrame:
        """
        Generate Volume Bars.

        Sample a new bar every time cumulative volume exceeds threshold.

        Args:
            data: OHLCV DataFrame
            threshold: Volume threshold per bar

        Returns:
            DataFrame with volume bars
        """
        bars = []
        cumulative_volume = 0

        bar_open = data['open'].iloc[0]
        bar_high = data['high'].iloc[0]
        bar_low = data['low'].iloc[0]
        bar_close = data['close'].iloc[0]
        bar_volume = 0
        bar_start_time = data.index[0]

        for idx in range(len(data)):
            row = data.iloc[idx]
            timestamp = data.index[idx]

            # Update bar statistics
            bar_high = max(bar_high, row['high'])
            bar_low = min(bar_low, row['low'])
            bar_close = row['close']
            bar_volume += row['volume']
            cumulative_volume += row['volume']

            # Check if threshold reached
            if cumulative_volume >= threshold:
                bars.append({
                    'timestamp': bar_start_time,
                    'open': bar_open,
                    'high': bar_high,
                    'low': bar_low,
                    'close': bar_close,
                    'volume': bar_volume,
                    'vwap': (bar_close + bar_high + bar_low) / 3,  # Approximation
                    'bar_duration': (timestamp - bar_start_time).total_seconds()
                })

                # Reset for next bar
                cumulative_volume = 0
                bar_volume = 0

                if idx + 1 < len(data):
                    next_row = data.iloc[idx + 1]
                    bar_start_time = data.index[idx + 1]
                    bar_open = next_row['open']
                    bar_high = next_row['high']
                    bar_low = next_row['low']

        # Handle remaining data
        if bar_volume > 0:
            bars.append({
                'timestamp': bar_start_time,
                'open': bar_open,
                'high': bar_high,
                'low': bar_low,
                'close': bar_close,
                'volume': bar_volume,
                'vwap': (bar_close + bar_high + bar_low) / 3,
                'bar_duration': (data.index[-1] - bar_start_time).total_seconds()
            })

        result = pd.DataFrame(bars)
        if len(result) > 0:
            result.set_index('timestamp', inplace=True)

        logger.info(f"Generated {len(result)} volume bars from {len(data)} samples")

        return result

    def _generate_dollar_bars(
        self,
        data: pd.DataFrame,
        threshold: float
    ) -> pd.DataFrame:
        """
        Generate Dollar Bars.

        Sample a new bar every time cumulative dollar volume exceeds threshold.
        Dollar volume = price * volume (represents actual money exchanged)

        This is the most robust sampling method as it:
        - Accounts for price changes over time
        - Better represents actual market activity
        - More stable across different price levels

        Args:
            data: OHLCV DataFrame
            threshold: Dollar volume threshold per bar

        Returns:
            DataFrame with dollar bars
        """
        bars = []
        cumulative_dollar = 0

        bar_open = data['open'].iloc[0]
        bar_high = data['high'].iloc[0]
        bar_low = data['low'].iloc[0]
        bar_close = data['close'].iloc[0]
        bar_volume = 0
        bar_dollar_volume = 0
        bar_start_time = data.index[0]

        for idx in range(len(data)):
            row = data.iloc[idx]
            timestamp = data.index[idx]

            # Calculate dollar volume for this period
            # Use VWAP approximation for better accuracy
            vwap = (row['high'] + row['low'] + row['close']) / 3
            dollar_volume = vwap * row['volume']

            # Update bar statistics
            bar_high = max(bar_high, row['high'])
            bar_low = min(bar_low, row['low'])
            bar_close = row['close']
            bar_volume += row['volume']
            bar_dollar_volume += dollar_volume
            cumulative_dollar += dollar_volume

            # Check if threshold reached
            if cumulative_dollar >= threshold:
                bars.append({
                    'timestamp': bar_start_time,
                    'open': bar_open,
                    'high': bar_high,
                    'low': bar_low,
                    'close': bar_close,
                    'volume': bar_volume,
                    'dollar_volume': bar_dollar_volume,
                    'vwap': bar_dollar_volume / bar_volume if bar_volume > 0 else bar_close,
                    'bar_duration': (timestamp - bar_start_time).total_seconds()
                })

                # Reset for next bar
                cumulative_dollar = 0
                bar_volume = 0
                bar_dollar_volume = 0

                if idx + 1 < len(data):
                    next_row = data.iloc[idx + 1]
                    bar_start_time = data.index[idx + 1]
                    bar_open = next_row['open']
                    bar_high = next_row['high']
                    bar_low = next_row['low']

        # Handle remaining data
        if bar_volume > 0:
            bars.append({
                'timestamp': bar_start_time,
                'open': bar_open,
                'high': bar_high,
                'low': bar_low,
                'close': bar_close,
                'volume': bar_volume,
                'dollar_volume': bar_dollar_volume,
                'vwap': bar_dollar_volume / bar_volume if bar_volume > 0 else bar_close,
                'bar_duration': (data.index[-1] - bar_start_time).total_seconds()
            })

        result = pd.DataFrame(bars)
        if len(result) > 0:
            result.set_index('timestamp', inplace=True)

        logger.info(f"Generated {len(result)} dollar bars from {len(data)} samples")

        return result

    def _generate_tick_bars(
        self,
        data: pd.DataFrame,
        threshold: float
    ) -> pd.DataFrame:
        """
        Generate Tick Bars.

        Sample a new bar every N ticks (transactions).

        Args:
            data: OHLCV DataFrame (each row represents one or more ticks)
            threshold: Number of ticks per bar

        Returns:
            DataFrame with tick bars
        """
        threshold = int(threshold)
        bars = []
        tick_count = 0

        bar_open = data['open'].iloc[0]
        bar_high = data['high'].iloc[0]
        bar_low = data['low'].iloc[0]
        bar_close = data['close'].iloc[0]
        bar_volume = 0
        bar_start_time = data.index[0]

        for idx in range(len(data)):
            row = data.iloc[idx]
            timestamp = data.index[idx]

            # Update bar statistics
            bar_high = max(bar_high, row['high'])
            bar_low = min(bar_low, row['low'])
            bar_close = row['close']
            bar_volume += row['volume']
            tick_count += 1

            # Check if threshold reached
            if tick_count >= threshold:
                bars.append({
                    'timestamp': bar_start_time,
                    'open': bar_open,
                    'high': bar_high,
                    'low': bar_low,
                    'close': bar_close,
                    'volume': bar_volume,
                    'tick_count': tick_count,
                    'bar_duration': (timestamp - bar_start_time).total_seconds()
                })

                # Reset for next bar
                tick_count = 0
                bar_volume = 0

                if idx + 1 < len(data):
                    next_row = data.iloc[idx + 1]
                    bar_start_time = data.index[idx + 1]
                    bar_open = next_row['open']
                    bar_high = next_row['high']
                    bar_low = next_row['low']

        # Handle remaining data
        if tick_count > 0:
            bars.append({
                'timestamp': bar_start_time,
                'open': bar_open,
                'high': bar_high,
                'low': bar_low,
                'close': bar_close,
                'volume': bar_volume,
                'tick_count': tick_count,
                'bar_duration': (data.index[-1] - bar_start_time).total_seconds()
            })

        result = pd.DataFrame(bars)
        if len(result) > 0:
            result.set_index('timestamp', inplace=True)

        logger.info(f"Generated {len(result)} tick bars from {len(data)} samples")

        return result

    def compute_bar_statistics(
        self,
        bars: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compute statistics about generated bars.

        Helps validate that bars have better statistical properties
        than time bars.

        Args:
            bars: Generated bars DataFrame

        Returns:
            Dictionary with statistics
        """
        if len(bars) == 0:
            return {}

        returns = bars['close'].pct_change().dropna()

        from scipy import stats as scipy_stats

        # Test for normality
        if len(returns) > 8:
            _, normality_pvalue = scipy_stats.normaltest(returns)
        else:
            normality_pvalue = np.nan

        # Test for serial correlation
        if len(returns) > 10:
            autocorr_lag1 = returns.autocorr(lag=1)
        else:
            autocorr_lag1 = np.nan

        statistics = {
            'n_bars': len(bars),
            'returns_mean': returns.mean(),
            'returns_std': returns.std(),
            'returns_skew': returns.skew(),
            'returns_kurtosis': returns.kurtosis(),
            'normality_pvalue': normality_pvalue,
            'autocorr_lag1': autocorr_lag1,
            'avg_bar_duration_seconds': bars['bar_duration'].mean() if 'bar_duration' in bars else np.nan,
            'bar_duration_std': bars['bar_duration'].std() if 'bar_duration' in bars else np.nan
        }

        return statistics


def convert_time_bars_to_information_bars(
    time_bars: pd.DataFrame,
    bar_type: str = "dollar",
    threshold: float = None,
    target_bars_per_day: int = 50
) -> pd.DataFrame:
    """
    Convenience function to convert time bars to information-driven bars.

    Args:
        time_bars: OHLCV DataFrame with time-based bars
        bar_type: "volume", "dollar", or "tick"
        threshold: Sampling threshold (auto-estimated if None)
        target_bars_per_day: Target bars per day for auto-estimation

    Returns:
        DataFrame with information-driven bars
    """
    generator = InformationDrivenBars(bar_type=bar_type, threshold=threshold)

    if threshold is None:
        threshold = generator.estimate_threshold(time_bars, target_bars_per_day)

    bars = generator.generate_bars(time_bars, threshold)

    # Compute and log statistics
    stats = generator.compute_bar_statistics(bars)

    logger.info(
        f"Bar conversion complete: {len(time_bars)} time bars -> {len(bars)} {bar_type} bars"
    )

    if 'normality_pvalue' in stats and not np.isnan(stats['normality_pvalue']):
        logger.info(
            f"Bar statistics: normality p={stats['normality_pvalue']:.4f}, "
            f"autocorr={stats['autocorr_lag1']:.4f}"
        )

    return bars
