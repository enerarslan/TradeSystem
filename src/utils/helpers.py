"""
Helper functions for AlphaTrade system.

This module provides common utility functions for:
- DataFrame operations
- Numerical computations
- Parallel processing
- Data transformations
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Literal, Sequence, TypeVar

import numpy as np
import pandas as pd
from numba import jit

T = TypeVar("T")


def ensure_datetime_index(
    df: pd.DataFrame,
    datetime_col: str | None = None,
    timezone: str | None = None,
) -> pd.DataFrame:
    """
    Ensure DataFrame has a DatetimeIndex.

    Args:
        df: Input DataFrame
        datetime_col: Column to convert to index (None if already indexed)
        timezone: Target timezone for localization/conversion

    Returns:
        DataFrame with DatetimeIndex

    Raises:
        ValueError: If datetime conversion fails
    """
    df = df.copy()

    if datetime_col is not None:
        if datetime_col not in df.columns:
            raise ValueError(f"Column '{datetime_col}' not found in DataFrame")
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.set_index(datetime_col)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if timezone is not None:
        if df.index.tz is None:
            df.index = df.index.tz_localize(timezone)
        else:
            df.index = df.index.tz_convert(timezone)

    return df


def safe_divide(
    numerator: np.ndarray | pd.Series | float,
    denominator: np.ndarray | pd.Series | float,
    fill_value: float = 0.0,
) -> np.ndarray | pd.Series | float:
    """
    Safely divide arrays/series, handling division by zero.

    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        fill_value: Value to use when denominator is zero

    Returns:
        Result of division with zeros filled
    """
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        result = numerator / denominator
        return result.fillna(fill_value).replace([np.inf, -np.inf], fill_value)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(numerator, denominator)
        if isinstance(result, np.ndarray):
            result = np.where(
                np.isfinite(result),
                result,
                fill_value,
            )
        elif not np.isfinite(result):
            result = fill_value
        return result


def clip_outliers(
    data: pd.Series | np.ndarray,
    n_std: float = 3.0,
    method: Literal["clip", "nan", "winsorize"] = "clip",
) -> pd.Series | np.ndarray:
    """
    Clip or handle outliers in data.

    Args:
        data: Input data
        n_std: Number of standard deviations for outlier threshold
        method: How to handle outliers
            - "clip": Clip to threshold values
            - "nan": Replace with NaN
            - "winsorize": Winsorize to percentiles

    Returns:
        Data with outliers handled
    """
    if method == "winsorize":
        lower = np.percentile(data[~np.isnan(data)], 1)
        upper = np.percentile(data[~np.isnan(data)], 99)
    else:
        mean = np.nanmean(data)
        std = np.nanstd(data)
        lower = mean - n_std * std
        upper = mean + n_std * std

    if isinstance(data, pd.Series):
        if method == "nan":
            return data.where((data >= lower) & (data <= upper), np.nan)
        return data.clip(lower=lower, upper=upper)
    else:
        if method == "nan":
            result = data.copy()
            result[(data < lower) | (data > upper)] = np.nan
            return result
        return np.clip(data, lower, upper)


@jit(nopython=True, cache=True)
def _rolling_sum_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """Numba-optimized rolling sum."""
    n = len(arr)
    result = np.empty(n)
    result[:window - 1] = np.nan

    # Initial sum
    current_sum = 0.0
    for i in range(window):
        current_sum += arr[i]
    result[window - 1] = current_sum

    # Rolling
    for i in range(window, n):
        current_sum = current_sum - arr[i - window] + arr[i]
        result[i] = current_sum

    return result


@jit(nopython=True, cache=True)
def _rolling_mean_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """Numba-optimized rolling mean."""
    return _rolling_sum_numba(arr, window) / window


@jit(nopython=True, cache=True)
def _rolling_std_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """Numba-optimized rolling standard deviation."""
    n = len(arr)
    result = np.empty(n)
    result[:window - 1] = np.nan

    for i in range(window - 1, n):
        window_data = arr[i - window + 1:i + 1]
        result[i] = np.std(window_data)

    return result


def rolling_apply(
    data: pd.Series | np.ndarray,
    window: int,
    func: Callable[[np.ndarray], float],
    min_periods: int | None = None,
) -> pd.Series | np.ndarray:
    """
    Apply a rolling window function to data.

    Args:
        data: Input data
        window: Window size
        func: Function to apply to each window
        min_periods: Minimum observations required (default: window)

    Returns:
        Rolling computation result
    """
    if min_periods is None:
        min_periods = window

    is_series = isinstance(data, pd.Series)
    arr = data.values if is_series else data
    n = len(arr)

    result = np.full(n, np.nan)

    for i in range(min_periods - 1, n):
        start_idx = max(0, i - window + 1)
        window_data = arr[start_idx:i + 1]
        if len(window_data) >= min_periods:
            result[i] = func(window_data)

    if is_series:
        return pd.Series(result, index=data.index, name=data.name)
    return result


def parallel_apply(
    func: Callable[[T], Any],
    items: Sequence[T],
    n_workers: int | None = None,
    use_threads: bool = False,
    show_progress: bool = False,
) -> list[Any]:
    """
    Apply a function to items in parallel.

    Args:
        func: Function to apply
        items: Sequence of items to process
        n_workers: Number of worker processes/threads
        use_threads: Use threads instead of processes
        show_progress: Show progress bar (requires tqdm)

    Returns:
        List of results
    """
    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    with executor_class(max_workers=n_workers) as executor:
        if show_progress:
            try:
                from tqdm import tqdm

                results = list(tqdm(executor.map(func, items), total=len(items)))
            except ImportError:
                results = list(executor.map(func, items))
        else:
            results = list(executor.map(func, items))

    return results


def resample_ohlcv(
    df: pd.DataFrame,
    target_timeframe: str,
    ohlcv_cols: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Resample OHLCV data to a different timeframe.

    Args:
        df: Input DataFrame with OHLCV data
        target_timeframe: Target timeframe (e.g., '1H', '1D')
        ohlcv_cols: Column name mapping (default: standard names)

    Returns:
        Resampled DataFrame
    """
    if ohlcv_cols is None:
        ohlcv_cols = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }

    agg_dict = {
        ohlcv_cols["open"]: "first",
        ohlcv_cols["high"]: "max",
        ohlcv_cols["low"]: "min",
        ohlcv_cols["close"]: "last",
        ohlcv_cols["volume"]: "sum",
    }

    # Add other columns (if any) with last value
    for col in df.columns:
        if col not in agg_dict:
            agg_dict[col] = "last"

    return df.resample(target_timeframe).agg(agg_dict).dropna()


def forward_fill_gaps(
    df: pd.DataFrame,
    max_gap: int = 10,
    method: Literal["ffill", "interpolate"] = "ffill",
) -> pd.DataFrame:
    """
    Fill gaps in time series data.

    Args:
        df: Input DataFrame
        max_gap: Maximum consecutive gaps to fill
        method: Fill method ('ffill' or 'interpolate')

    Returns:
        DataFrame with gaps filled
    """
    df = df.copy()

    if method == "ffill":
        df = df.ffill(limit=max_gap)
    elif method == "interpolate":
        df = df.interpolate(method="time", limit=max_gap)

    return df


def calculate_returns(
    prices: pd.Series | pd.DataFrame,
    periods: int = 1,
    method: Literal["simple", "log"] = "simple",
) -> pd.Series | pd.DataFrame:
    """
    Calculate returns from price series.

    Args:
        prices: Price series or DataFrame
        periods: Number of periods for return calculation
        method: Return calculation method

    Returns:
        Returns series or DataFrame
    """
    if method == "simple":
        return prices.pct_change(periods=periods)
    elif method == "log":
        return np.log(prices / prices.shift(periods))
    else:
        raise ValueError(f"Unknown method: {method}")


def annualize_returns(
    returns: float | pd.Series,
    periods_per_year: int = 252,
) -> float | pd.Series:
    """
    Annualize returns.

    Args:
        returns: Period returns (daily, monthly, etc.)
        periods_per_year: Number of periods in a year

    Returns:
        Annualized returns
    """
    if isinstance(returns, pd.Series):
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)
    else:
        total_return = returns
        n_periods = 1

    years = n_periods / periods_per_year
    return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0


def annualize_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Annualize volatility.

    Args:
        returns: Period returns
        periods_per_year: Number of periods in a year

    Returns:
        Annualized volatility
    """
    return returns.std() * np.sqrt(periods_per_year)


def create_multiindex_df(
    data: dict[str, pd.DataFrame],
    names: tuple[str, str] = ("timestamp", "symbol"),
) -> pd.DataFrame:
    """
    Create a MultiIndex DataFrame from dict of symbol DataFrames.

    Args:
        data: Dictionary mapping symbols to DataFrames
        names: Names for the MultiIndex levels

    Returns:
        MultiIndex DataFrame
    """
    dfs = []
    for symbol, df in data.items():
        df = df.copy()
        df["symbol"] = symbol
        dfs.append(df)

    combined = pd.concat(dfs)
    combined = combined.reset_index()
    combined = combined.set_index([combined.columns[0], "symbol"])
    combined.index.names = list(names)

    return combined.sort_index()


def split_multiindex_df(
    df: pd.DataFrame,
    level: int | str = 1,
) -> dict[str, pd.DataFrame]:
    """
    Split a MultiIndex DataFrame into dict of DataFrames.

    Args:
        df: MultiIndex DataFrame
        level: Level to split on (symbol level)

    Returns:
        Dictionary mapping index values to DataFrames
    """
    result = {}
    for key in df.index.get_level_values(level).unique():
        result[key] = df.xs(key, level=level)
    return result


def get_trading_calendar(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    freq: str = "15min",
    market_open: str = "09:30",
    market_close: str = "16:00",
    timezone: str = "America/New_York",
) -> pd.DatetimeIndex:
    """
    Generate a trading calendar with market hours.

    Args:
        start_date: Start date
        end_date: End date
        freq: Bar frequency
        market_open: Market open time
        market_close: Market close time
        timezone: Market timezone

    Returns:
        DatetimeIndex of valid trading timestamps
    """
    # Generate all business days
    dates = pd.date_range(start=start_date, end=end_date, freq="B")

    # Generate intraday timestamps for each day
    timestamps = []
    for date in dates:
        day_start = pd.Timestamp(f"{date.date()} {market_open}", tz=timezone)
        day_end = pd.Timestamp(f"{date.date()} {market_close}", tz=timezone)

        day_timestamps = pd.date_range(start=day_start, end=day_end, freq=freq)
        timestamps.extend(day_timestamps)

    return pd.DatetimeIndex(timestamps)
