"""
Fractional Differentiation for achieving stationarity while preserving memory.

This module implements the Fixed-Width Window Fractional Differentiation method
from "Advances in Financial Machine Learning" by Marcos Lopez de Prado.

Key concept:
- Standard differentiation (d=1) makes series stationary but destroys memory
- No differentiation (d=0) preserves memory but series is non-stationary
- Fractional differentiation (0<d<1) balances stationarity and memory preservation

The minimum d that achieves stationarity (via ADF test) while maximizing
correlation with the original series is optimal.

Reference:
    Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
    Chapter 5: Fractionally Differentiated Features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller, kpss
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from sklearn.base import BaseEstimator, TransformerMixin
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class StationarityTestResult:
    """Result of stationarity tests."""
    test_name: str
    statistic: float
    p_value: float
    is_stationary: bool
    critical_values: dict
    n_lags: int


def get_weights_ffd(
    d: float,
    threshold: float = 1e-5,
    max_window: int = 1000,
) -> np.ndarray:
    """
    Calculate weights for Fixed-Width Window Fractional Differentiation.

    The weights are derived from the binomial series expansion:
    (1-B)^d = sum_{k=0}^{inf} (-1)^k * C(d,k) * B^k

    where B is the backshift operator and C(d,k) is the binomial coefficient.

    Args:
        d: Differentiation order (0 < d < 1 typically)
        threshold: Drop weights below this threshold
        max_window: Maximum number of weights to compute

    Returns:
        Array of weights
    """
    weights = [1.0]

    for k in range(1, max_window):
        # Binomial coefficient: w_k = w_{k-1} * (k-d-1) / k
        w = -weights[-1] * (d - k + 1) / k
        weights.append(w)

        # Stop if weight falls below threshold
        if abs(w) < threshold:
            break

    return np.array(weights[::-1])  # Reverse for convolution


def frac_diff_ffd(
    series: Union[pd.Series, np.ndarray],
    d: float,
    threshold: float = 1e-5,
) -> pd.Series:
    """
    Apply Fixed-Width Window Fractional Differentiation.

    FFD uses a fixed window of weights truncated at a threshold,
    making it suitable for real-time applications.

    Args:
        series: Input time series
        d: Differentiation order
        threshold: Weight threshold for window truncation

    Returns:
        Fractionally differentiated series

    Example:
        prices = pd.Series([100, 101, 99, 102, 103])
        frac_prices = frac_diff_ffd(prices, d=0.4)
    """
    # Get weights
    weights = get_weights_ffd(d, threshold)
    window_size = len(weights)

    # Convert to numpy if needed
    if isinstance(series, pd.Series):
        values = series.values
        index = series.index
    else:
        values = series
        index = None

    # Apply convolution
    result = np.full(len(values), np.nan)

    for i in range(window_size - 1, len(values)):
        window = values[i - window_size + 1:i + 1]
        if not np.isnan(window).any():
            result[i] = np.dot(weights, window)

    # Return as Series if input was Series
    if index is not None:
        return pd.Series(result, index=index, name=f"ffd_{d:.3f}")
    return result


def frac_diff_expanding(
    series: Union[pd.Series, np.ndarray],
    d: float,
    threshold: float = 1e-5,
) -> pd.Series:
    """
    Apply Expanding Window Fractional Differentiation.

    Uses all available historical data (expanding window).
    More accurate but computationally expensive.

    Args:
        series: Input time series
        d: Differentiation order
        threshold: Weight threshold

    Returns:
        Fractionally differentiated series
    """
    # Get maximum weights
    weights = get_weights_ffd(d, threshold, max_window=len(series))

    if isinstance(series, pd.Series):
        values = series.values
        index = series.index
    else:
        values = series
        index = None

    result = np.full(len(values), np.nan)

    for i in range(len(values)):
        # Use expanding window
        window_size = min(i + 1, len(weights))
        if window_size > 0:
            w = weights[-window_size:]
            window = values[i - window_size + 1:i + 1]
            if not np.isnan(window).any():
                result[i] = np.dot(w, window)

    if index is not None:
        return pd.Series(result, index=index, name=f"ffd_exp_{d:.3f}")
    return result


def test_stationarity_adf(
    series: Union[pd.Series, np.ndarray],
    max_lag: Optional[int] = None,
    regression: str = 'c',
    significance: float = 0.05,
) -> StationarityTestResult:
    """
    Augmented Dickey-Fuller test for stationarity.

    Tests null hypothesis that series has a unit root (non-stationary).
    Reject null (p < significance) means series is stationary.

    Args:
        series: Time series to test
        max_lag: Maximum lag for test (None = auto)
        regression: 'c' (constant), 'ct' (constant+trend), 'ctt', 'n'
        significance: Significance level for stationarity

    Returns:
        StationarityTestResult
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels required for ADF test")

    # Remove NaN
    clean_series = pd.Series(series).dropna()

    if len(clean_series) < 20:
        raise ValueError("Series too short for ADF test")

    result = adfuller(clean_series, maxlag=max_lag, regression=regression)

    return StationarityTestResult(
        test_name="ADF",
        statistic=result[0],
        p_value=result[1],
        is_stationary=result[1] < significance,
        critical_values={f"{k}%": v for k, v in result[4].items()},
        n_lags=result[2],
    )


def test_stationarity_kpss(
    series: Union[pd.Series, np.ndarray],
    regression: str = 'c',
    significance: float = 0.05,
) -> StationarityTestResult:
    """
    KPSS test for stationarity.

    Tests null hypothesis that series is stationary.
    Reject null (p < significance) means series is non-stationary.

    Note: KPSS and ADF have opposite null hypotheses.
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels required for KPSS test")

    clean_series = pd.Series(series).dropna()

    result = kpss(clean_series, regression=regression)

    return StationarityTestResult(
        test_name="KPSS",
        statistic=result[0],
        p_value=result[1],
        is_stationary=result[1] > significance,  # Note: opposite of ADF
        critical_values={f"{k}%": v for k, v in result[3].items()},
        n_lags=result[2],
    )


def find_min_d(
    series: Union[pd.Series, np.ndarray],
    d_range: Tuple[float, float] = (0, 1),
    p_value_threshold: float = 0.05,
    n_steps: int = 20,
    threshold: float = 1e-5,
) -> Tuple[float, float, pd.DataFrame]:
    """
    Find minimum d that achieves stationarity while maximizing memory.

    Searches over d values to find the smallest d where the ADF test
    rejects the null hypothesis of non-stationarity.

    Args:
        series: Input time series
        d_range: Range of d values to search (min, max)
        p_value_threshold: P-value threshold for stationarity
        n_steps: Number of d values to test
        threshold: Weight threshold for FFD

    Returns:
        Tuple of (optimal_d, correlation_with_original, results_df)

    Example:
        prices = df['close']
        opt_d, corr, results = find_min_d(prices)
        print(f"Optimal d={opt_d:.3f}, correlation={corr:.3f}")
    """
    d_values = np.linspace(d_range[0], d_range[1], n_steps)
    results = []

    # Ensure series is pandas Series
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    original_clean = series.dropna()

    for d in d_values:
        # Apply fractional differentiation
        ffd_series = frac_diff_ffd(series, d, threshold)
        ffd_clean = ffd_series.dropna()

        if len(ffd_clean) < 20:
            continue

        try:
            # ADF test
            adf_result = test_stationarity_adf(ffd_clean)

            # Correlation with original
            # Align indices
            common_idx = original_clean.index.intersection(ffd_clean.index)
            if len(common_idx) > 0:
                corr = original_clean.loc[common_idx].corr(ffd_clean.loc[common_idx])
            else:
                corr = np.nan

            results.append({
                'd': d,
                'adf_stat': adf_result.statistic,
                'adf_pvalue': adf_result.p_value,
                'is_stationary': adf_result.is_stationary,
                'correlation': corr,
                'n_valid': len(ffd_clean),
            })

        except Exception as e:
            logger.warning(f"Error testing d={d:.3f}: {e}")
            continue

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    if results_df.empty:
        raise ValueError("No valid results found")

    # Find minimum d that achieves stationarity
    stationary_results = results_df[results_df['is_stationary']]

    if stationary_results.empty:
        # If none stationary, use d=1
        optimal_d = 1.0
        optimal_corr = 0.0
        logger.warning("No d value achieved stationarity, using d=1.0")
    else:
        # Find minimum d with maximum correlation among stationary
        optimal_row = stationary_results.loc[stationary_results['d'].idxmin()]
        optimal_d = optimal_row['d']
        optimal_corr = optimal_row['correlation']

    return optimal_d, optimal_corr, results_df


def plot_d_analysis(
    results_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 8),
) -> Optional["plt.Figure"]:
    """
    Visualize the relationship between d, stationarity, and correlation.

    Creates a dual-axis plot showing:
    - ADF p-value vs d
    - Correlation with original vs d

    Args:
        results_df: Results from find_min_d
        figsize: Figure size

    Returns:
        matplotlib Figure or None if matplotlib unavailable
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Plot ADF p-value
    ax1.plot(results_df['d'], results_df['adf_pvalue'], 'b-o', label='ADF p-value')
    ax1.axhline(y=0.05, color='r', linestyle='--', label='5% threshold')
    ax1.fill_between(
        results_df['d'],
        0,
        results_df['adf_pvalue'],
        where=results_df['adf_pvalue'] < 0.05,
        alpha=0.3,
        color='green',
        label='Stationary region'
    )
    ax1.set_ylabel('ADF p-value')
    ax1.set_title('Stationarity vs Differentiation Order')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot correlation
    ax2.plot(results_df['d'], results_df['correlation'], 'g-o', label='Correlation')
    ax2.set_xlabel('Differentiation Order (d)')
    ax2.set_ylabel('Correlation with Original')
    ax2.set_title('Memory Preservation vs Differentiation Order')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if SKLEARN_AVAILABLE:

    class FractionalDiffTransformer(BaseEstimator, TransformerMixin):
        """
        Sklearn-compatible transformer for fractional differentiation.

        Can automatically find optimal d for each column or use specified values.

        Example:
            # Auto-find optimal d
            transformer = FractionalDiffTransformer(d=None, auto_find=True)
            X_transformed = transformer.fit_transform(X)

            # Use specified d
            transformer = FractionalDiffTransformer(d=0.4)
            X_transformed = transformer.fit_transform(X)
        """

        def __init__(
            self,
            d: Optional[Union[float, Dict[str, float]]] = None,
            columns: Optional[List[str]] = None,
            threshold: float = 1e-5,
            auto_find: bool = True,
            p_value_threshold: float = 0.05,
        ):
            """
            Initialize transformer.

            Args:
                d: Differentiation order. Can be:
                   - float: Apply same d to all columns
                   - dict: {column_name: d_value} for column-specific d
                   - None: Auto-find optimal d for each column
                columns: Columns to transform (None = all numeric columns)
                threshold: Weight threshold for FFD
                auto_find: If d is None, auto-find optimal d
                p_value_threshold: P-value for stationarity in auto-find
            """
            self.d = d
            self.columns = columns
            self.threshold = threshold
            self.auto_find = auto_find
            self.p_value_threshold = p_value_threshold

            # Fitted parameters
            self.d_values_: Dict[str, float] = {}
            self.correlations_: Dict[str, float] = {}
            self.feature_names_in_: Optional[List[str]] = None

        def fit(
            self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Optional[np.ndarray] = None,
        ) -> "FractionalDiffTransformer":
            """
            Fit the transformer.

            If d is None and auto_find is True, finds optimal d for each column.
            """
            # Convert to DataFrame
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)

            self.feature_names_in_ = list(X.columns)

            # Determine columns to transform
            if self.columns is not None:
                columns = [c for c in self.columns if c in X.columns]
            else:
                columns = X.select_dtypes(include=[np.number]).columns.tolist()

            # Determine d values
            if self.d is None and self.auto_find:
                # Auto-find optimal d for each column
                for col in columns:
                    try:
                        opt_d, corr, _ = find_min_d(
                            X[col],
                            p_value_threshold=self.p_value_threshold,
                            threshold=self.threshold,
                        )
                        self.d_values_[col] = opt_d
                        self.correlations_[col] = corr
                        logger.info(f"Column {col}: optimal d={opt_d:.3f}, corr={corr:.3f}")
                    except Exception as e:
                        logger.warning(f"Could not find optimal d for {col}: {e}")
                        self.d_values_[col] = 1.0  # Default to full diff
                        self.correlations_[col] = 0.0

            elif isinstance(self.d, dict):
                # Use specified d values
                self.d_values_ = {
                    col: self.d.get(col, 0.5) for col in columns
                }

            else:
                # Use single d value for all
                d_val = self.d if self.d is not None else 0.5
                self.d_values_ = {col: d_val for col in columns}

            return self

        def transform(
            self,
            X: Union[pd.DataFrame, np.ndarray],
        ) -> pd.DataFrame:
            """
            Apply fractional differentiation.

            Returns DataFrame with transformed columns.
            """
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                X = X.copy()

            for col, d in self.d_values_.items():
                if col in X.columns:
                    X[f"{col}_ffd"] = frac_diff_ffd(X[col], d, self.threshold)

            return X

        def fit_transform(
            self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Optional[np.ndarray] = None,
        ) -> pd.DataFrame:
            """Fit and transform."""
            return self.fit(X, y).transform(X)

        def get_feature_names_out(
            self,
            input_features: Optional[List[str]] = None,
        ) -> List[str]:
            """Get output feature names."""
            if input_features is None:
                input_features = self.feature_names_in_

            output_features = list(input_features)
            for col in self.d_values_:
                output_features.append(f"{col}_ffd")

            return output_features


# Convenience functions
def fractional_diff(
    series: Union[pd.Series, np.ndarray],
    d: float = 0.4,
) -> pd.Series:
    """Quick function to apply fractional differentiation."""
    return frac_diff_ffd(series, d)


def auto_fractional_diff(
    series: Union[pd.Series, np.ndarray],
) -> Tuple[pd.Series, float]:
    """
    Automatically find optimal d and apply fractional differentiation.

    Returns:
        Tuple of (transformed_series, optimal_d)
    """
    opt_d, _, _ = find_min_d(series)
    transformed = frac_diff_ffd(series, opt_d)
    return transformed, opt_d
