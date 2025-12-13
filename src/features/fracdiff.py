"""
Fractional Differentiation Module
Implements fixed-window fractional differentiation for time-series stationarity.

Based on "Advances in Financial Machine Learning" by Marcos López de Prado.

Features:
- Fixed-window fractional differentiation
- Automatic minimum d estimation via ADF test
- Memory preservation while achieving stationarity
- Integration with FeatureBuilder pipeline
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Union, List
from dataclasses import dataclass
from functools import lru_cache
import warnings

from ..utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class FracDiffConfig:
    """Configuration for fractional differentiation"""
    d_min: float = 0.0
    d_max: float = 1.0
    d_step: float = 0.05
    threshold: float = 1e-5  # Weight threshold for window cutoff
    p_value_threshold: float = 0.05  # ADF test p-value threshold
    max_lag: int = 1  # Max lag for ADF test
    regression: str = 'c'  # ADF regression type


@dataclass
class FracDiffResult:
    """Result of fractional differentiation"""
    series: pd.Series
    d: float
    adf_statistic: float
    adf_pvalue: float
    is_stationary: bool
    weights: np.ndarray


class FractionalDifferentiation:
    """
    Fractional differentiation for achieving stationarity while preserving memory.

    Standard differentiation (d=1) makes series stationary but destroys all memory.
    Fractional differentiation allows finding minimum d where series becomes
    stationary while preserving maximum memory (correlation with original).

    Methods:
    - get_weights_ffd: Get weights for fixed-window fractional differentiation
    - frac_diff_ffd: Apply fixed-window fractional differentiation
    - find_min_ffd: Find minimum d for stationarity via ADF test
    - auto_frac_diff: Automatically differentiate with optimal d
    """

    def __init__(self, config: Optional[FracDiffConfig] = None):
        """
        Initialize FractionalDifferentiation.

        Args:
            config: Configuration for fractional differentiation
        """
        self.config = config or FracDiffConfig()
        self._cache: Dict[Tuple[float, float], np.ndarray] = {}

    @staticmethod
    @lru_cache(maxsize=128)
    def _compute_weights(d: float, threshold: float, max_size: int = 10000) -> np.ndarray:
        """
        Compute weights for fractional differentiation.

        The weights are computed using the formula:
        w_k = -w_{k-1} * (d - k + 1) / k

        Starting with w_0 = 1

        Args:
            d: Differentiation order (0 < d < 1)
            threshold: Minimum weight to include (convergence cutoff)
            max_size: Maximum number of weights

        Returns:
            Array of weights
        """
        weights = [1.0]
        k = 1

        while k < max_size:
            w_k = -weights[-1] * (d - k + 1) / k
            if abs(w_k) < threshold:
                break
            weights.append(w_k)
            k += 1

        return np.array(weights, dtype=np.float64)

    def get_weights_ffd(
        self,
        d: float,
        threshold: float = None,
        max_size: int = 10000
    ) -> np.ndarray:
        """
        Get weights for Fixed-window Fractional Differentiation (FFD).

        FFD uses a fixed window size determined by the weight threshold,
        which ensures consistent feature across different samples.

        Args:
            d: Differentiation order (0 < d < 1)
            threshold: Weight cutoff threshold
            max_size: Maximum window size

        Returns:
            Array of weights
        """
        threshold = threshold or self.config.threshold

        # Use cached version
        return self._compute_weights(d, threshold, max_size)

    def frac_diff_ffd(
        self,
        series: pd.Series,
        d: float,
        threshold: float = None
    ) -> pd.Series:
        """
        Apply Fixed-window Fractional Differentiation (FFD).

        This is the preferred method as it:
        1. Produces consistent features (same window for all samples)
        2. Avoids look-ahead bias
        3. More suitable for real-time applications

        Args:
            series: Price series to differentiate
            d: Differentiation order (0 < d < 1)
            threshold: Weight cutoff threshold

        Returns:
            Fractionally differentiated series
        """
        threshold = threshold or self.config.threshold
        weights = self.get_weights_ffd(d, threshold)
        width = len(weights)

        # Apply fractional differentiation using convolution
        result = pd.Series(index=series.index, dtype=np.float64)

        # Use numpy for efficient calculation
        values = series.values.astype(np.float64)

        for i in range(width - 1, len(values)):
            # Weighted sum of past values
            result.iloc[i] = np.dot(weights, values[i - width + 1:i + 1][::-1])

        return result

    def frac_diff_ffd_vectorized(
        self,
        series: pd.Series,
        d: float,
        threshold: float = None
    ) -> pd.Series:
        """
        Vectorized version of FFD for better performance.

        Uses NumPy convolution for efficient computation.

        Args:
            series: Price series to differentiate
            d: Differentiation order
            threshold: Weight cutoff threshold

        Returns:
            Fractionally differentiated series
        """
        threshold = threshold or self.config.threshold
        weights = self.get_weights_ffd(d, threshold)

        # Reverse weights for convolution
        weights_reversed = weights[::-1]

        # Apply convolution
        values = series.values.astype(np.float64)
        result_values = np.convolve(values, weights_reversed, mode='valid')

        # Align with original index
        result = pd.Series(
            np.nan,
            index=series.index,
            dtype=np.float64
        )

        # Safe assignment: ensure we only assign as many values as we have
        start_idx = len(weights) - 1
        end_idx = start_idx + len(result_values)
        if len(result_values) > 0 and end_idx <= len(result):
            result.iloc[start_idx:end_idx] = result_values

        return result

    @staticmethod
    def adf_test(
        series: pd.Series,
        max_lag: int = 1,
        regression: str = 'c'
    ) -> Tuple[float, float, bool]:
        """
        Augmented Dickey-Fuller test for stationarity.

        Args:
            series: Time series to test
            max_lag: Maximum lag for test
            regression: Type of regression ('c', 'ct', 'ctt', 'n')

        Returns:
            Tuple of (ADF statistic, p-value, is_stationary)
        """
        try:
            from statsmodels.tsa.stattools import adfuller

            # Remove NaN values
            clean_series = series.dropna()

            if len(clean_series) < 20:
                return 0.0, 1.0, False

            result = adfuller(
                clean_series,
                maxlag=max_lag,
                regression=regression,
                autolag=None
            )

            adf_stat = result[0]
            p_value = result[1]

            # Series is stationary if p-value < 0.05
            is_stationary = p_value < 0.05

            return adf_stat, p_value, is_stationary

        except ImportError:
            logger.warning("statsmodels not installed. ADF test unavailable.")
            return 0.0, 1.0, False
        except Exception as e:
            logger.warning(f"ADF test failed: {e}")
            return 0.0, 1.0, False

    def find_min_ffd(
        self,
        series: pd.Series,
        d_min: float = None,
        d_max: float = None,
        d_step: float = None,
        p_value_threshold: float = None
    ) -> Tuple[float, Dict[float, Tuple[float, float]]]:
        """
        Find minimum d that achieves stationarity.

        Iterates through d values from d_min to d_max and finds the
        minimum d where the ADF test p-value falls below threshold.

        Args:
            series: Price series to analyze
            d_min: Minimum d to test
            d_max: Maximum d to test
            d_step: Step size for d values
            p_value_threshold: P-value threshold for stationarity

        Returns:
            Tuple of (optimal_d, dict of {d: (adf_stat, p_value)})
        """
        d_min = d_min if d_min is not None else self.config.d_min
        d_max = d_max if d_max is not None else self.config.d_max
        d_step = d_step if d_step is not None else self.config.d_step
        p_value_threshold = p_value_threshold or self.config.p_value_threshold

        results = {}
        optimal_d = d_max  # Default to full differentiation if nothing works

        # Test original series first
        adf_stat, p_value, is_stationary = self.adf_test(
            series,
            self.config.max_lag,
            self.config.regression
        )
        results[0.0] = (adf_stat, p_value)

        if is_stationary:
            logger.info("Series is already stationary (d=0)")
            return 0.0, results

        # Iterate through d values
        d_values = np.arange(d_min + d_step, d_max + d_step, d_step)

        for d in d_values:
            # Apply fractional differentiation
            ffd_series = self.frac_diff_ffd_vectorized(series, d)

            # Test stationarity
            adf_stat, p_value, is_stationary = self.adf_test(
                ffd_series,
                self.config.max_lag,
                self.config.regression
            )

            results[round(d, 4)] = (adf_stat, p_value)

            if is_stationary and d < optimal_d:
                optimal_d = d
                logger.info(f"Found stationary at d={d:.4f}, p-value={p_value:.4f}")
                break

        return optimal_d, results

    def auto_frac_diff(
        self,
        series: pd.Series,
        find_optimal: bool = True
    ) -> FracDiffResult:
        """
        Automatically apply fractional differentiation with optimal d.

        Args:
            series: Price series to differentiate
            find_optimal: Whether to find optimal d (vs using default)

        Returns:
            FracDiffResult with differentiated series and metadata
        """
        if find_optimal:
            d, results = self.find_min_ffd(series)
        else:
            d = 0.5  # Default middle ground

        # Apply differentiation
        ffd_series = self.frac_diff_ffd_vectorized(series, d)
        weights = self.get_weights_ffd(d)

        # Test final series
        adf_stat, p_value, is_stationary = self.adf_test(ffd_series)

        return FracDiffResult(
            series=ffd_series,
            d=d,
            adf_statistic=adf_stat,
            adf_pvalue=p_value,
            is_stationary=is_stationary,
            weights=weights
        )

    def differentiate_dataframe(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        find_optimal: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Apply fractional differentiation to multiple columns.

        Args:
            df: DataFrame with price data
            columns: Columns to differentiate (default: ['close'])
            find_optimal: Whether to find optimal d for each column

        Returns:
            Tuple of (differentiated DataFrame, dict of optimal d values)
        """
        columns = columns or ['close']
        result_df = df.copy()
        optimal_ds = {}

        for col in columns:
            if col not in df.columns:
                continue

            result = self.auto_frac_diff(df[col], find_optimal)

            # Add fractionally differentiated column
            result_df[f'{col}_ffd'] = result.series
            optimal_ds[col] = result.d

            logger.info(
                f"Column '{col}': d={result.d:.4f}, "
                f"stationary={result.is_stationary}, "
                f"p-value={result.adf_pvalue:.4f}"
            )

        return result_df, optimal_ds


class FracDiffFeatureTransformer:
    """
    Feature transformer that applies fractional differentiation to price data.

    Designed to integrate with FeatureBuilder pipeline.
    """

    def __init__(
        self,
        d: Optional[float] = None,
        auto_optimize: bool = True,
        threshold: float = 1e-5
    ):
        """
        Initialize transformer.

        Args:
            d: Fixed differentiation order (if not auto-optimizing)
            auto_optimize: Whether to find optimal d automatically
            threshold: Weight threshold for FFD
        """
        self.d = d
        self.auto_optimize = auto_optimize
        self.threshold = threshold

        self._frac_diff = FractionalDifferentiation(
            FracDiffConfig(threshold=threshold)
        )
        self._fitted_ds: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, columns: List[str] = None) -> 'FracDiffFeatureTransformer':
        """
        Fit transformer by finding optimal d values.

        Args:
            df: Training data
            columns: Columns to fit

        Returns:
            Self for chaining
        """
        columns = columns or ['close', 'open', 'high', 'low']

        for col in columns:
            if col not in df.columns:
                continue

            if self.auto_optimize:
                d, _ = self._frac_diff.find_min_ffd(df[col])
            else:
                d = self.d or 0.5

            self._fitted_ds[col] = d
            logger.info(f"Fitted d={d:.4f} for column '{col}'")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted d values.

        Args:
            df: Data to transform

        Returns:
            DataFrame with fractionally differentiated columns
        """
        result = df.copy()

        for col, d in self._fitted_ds.items():
            if col not in df.columns:
                continue

            ffd_series = self._frac_diff.frac_diff_ffd_vectorized(df[col], d)
            result[f'{col}_ffd'] = ffd_series

        return result

    def fit_transform(
        self,
        df: pd.DataFrame,
        columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            df: Data to fit and transform
            columns: Columns to process

        Returns:
            Transformed DataFrame
        """
        return self.fit(df, columns).transform(df)

    def get_fitted_d(self) -> Dict[str, float]:
        """Get fitted d values for each column"""
        return self._fitted_ds.copy()


def test_stationarity_adf(
    series: pd.Series,
    significance: float = 0.05
) -> Dict[str, Union[float, bool, str]]:
    """
    Comprehensive ADF test with detailed output.

    Args:
        series: Time series to test
        significance: Significance level for hypothesis test

    Returns:
        Dictionary with test results
    """
    try:
        from statsmodels.tsa.stattools import adfuller

        clean_series = series.dropna()

        if len(clean_series) < 20:
            return {
                'test_statistic': None,
                'p_value': None,
                'critical_values': {},
                'is_stationary': False,
                'conclusion': 'Insufficient data for ADF test'
            }

        result = adfuller(clean_series, autolag='AIC')

        adf_stat = result[0]
        p_value = result[1]
        used_lag = result[2]
        n_obs = result[3]
        critical_values = result[4]

        is_stationary = p_value < significance

        if is_stationary:
            conclusion = f"Series is stationary (p-value {p_value:.4f} < {significance})"
        else:
            conclusion = f"Series is non-stationary (p-value {p_value:.4f} >= {significance})"

        return {
            'test_statistic': adf_stat,
            'p_value': p_value,
            'used_lag': used_lag,
            'n_observations': n_obs,
            'critical_values': critical_values,
            'is_stationary': is_stationary,
            'conclusion': conclusion
        }

    except ImportError:
        return {
            'test_statistic': None,
            'p_value': None,
            'critical_values': {},
            'is_stationary': False,
            'conclusion': 'statsmodels not installed'
        }
