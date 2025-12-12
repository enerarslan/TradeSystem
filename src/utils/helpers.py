"""
Utility Functions for Institutional Trading System
JPMorgan-Level Helper Functions and Common Utilities

Features:
- Configuration management
- Data validation utilities
- Performance helpers
- Mathematical utilities
- Retry mechanisms
"""

import os
import yaml
import json
import time
import hashlib
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import (
    Any, Callable, Dict, List, Optional,
    Tuple, TypeVar, Union, Generator
)
from functools import wraps, lru_cache
import threading
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


T = TypeVar('T')


# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================

def load_config(config_path: str, env_override: bool = True) -> Dict[str, Any]:
    """
    Load YAML configuration with environment variable substitution.

    Args:
        config_path: Path to YAML config file
        env_override: Whether to substitute ${VAR} with environment variables

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Environment variable substitution
    if env_override:
        import re
        pattern = r'\$\{([^}]+)\}'

        def replace_env(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        content = re.sub(pattern, replace_env, content)

    config = yaml.safe_load(content)
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge multiple configuration dictionaries.
    Later configs override earlier ones.
    """
    result = {}

    for config in configs:
        result = _deep_merge(result, config)

    return result


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries"""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def save_config(config: Dict[str, Any], path: str) -> None:
    """Save configuration to YAML file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# =============================================================================
# FILE AND DIRECTORY UTILITIES
# =============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """Get project root directory"""
    current = Path(__file__).resolve()

    # Navigate up to find project root (contains config/)
    for parent in current.parents:
        if (parent / "config").exists():
            return parent

    return Path.cwd()


def file_hash(file_path: Union[str, Path], algorithm: str = "md5") -> str:
    """Calculate file hash for integrity checking"""
    hasher = hashlib.new(algorithm)

    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


def atomic_write(path: Union[str, Path], content: Union[str, bytes]) -> None:
    """Atomic file write to prevent corruption"""
    path = Path(path)
    temp_path = path.with_suffix(path.suffix + '.tmp')

    mode = 'wb' if isinstance(content, bytes) else 'w'
    encoding = None if isinstance(content, bytes) else 'utf-8'

    try:
        with open(temp_path, mode, encoding=encoding) as f:
            f.write(content)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


# =============================================================================
# TIMING AND PERFORMANCE UTILITIES
# =============================================================================

@contextmanager
def timer(name: str = "Operation", logger=None):
    """
    Context manager for timing code blocks.

    Usage:
        with timer("Data loading"):
            load_data()
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        message = f"{name} completed in {elapsed:.4f}s ({elapsed*1000:.2f}ms)"

        if logger:
            logger.debug(message)
        else:
            print(message)


class Timer:
    """
    Timer class for performance measurement.

    Usage:
        t = Timer()
        # do something
        print(t.elapsed)  # seconds
        print(t.elapsed_ms)  # milliseconds
    """

    def __init__(self, auto_start: bool = True):
        self._start: Optional[float] = None
        self._stop: Optional[float] = None

        if auto_start:
            self.start()

    def start(self) -> 'Timer':
        self._start = time.perf_counter()
        self._stop = None
        return self

    def stop(self) -> float:
        self._stop = time.perf_counter()
        return self.elapsed

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds"""
        end = self._stop if self._stop else time.perf_counter()
        return end - (self._start or end)

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds"""
        return self.elapsed * 1000


def rate_limiter(max_calls: int, period: float):
    """
    Decorator for rate limiting function calls.

    Args:
        max_calls: Maximum number of calls allowed
        period: Time period in seconds
    """
    calls = []
    lock = threading.Lock()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            nonlocal calls

            with lock:
                now = time.time()
                # Remove old calls
                calls = [c for c in calls if now - c < period]

                if len(calls) >= max_calls:
                    sleep_time = period - (now - calls[0])
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    calls = calls[1:]

                calls.append(time.time())

            return func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# RETRY MECHANISMS
# =============================================================================

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[type, ...] = (Exception,),
    logger=None
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        exceptions: Tuple of exceptions to catch
        logger: Optional logger instance
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        if logger:
                            logger.error(
                                f"{func.__name__} failed after {max_retries} retries: {e}"
                            )
                        raise

                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )

                    if logger:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}), "
                            f"retrying in {delay:.2f}s: {e}"
                        )

                    time.sleep(delay)

            raise last_exception

        return wrapper

    return decorator


# =============================================================================
# DATA VALIDATION UTILITIES
# =============================================================================

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 0,
    max_null_pct: float = 1.0,
    numeric_columns: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate a DataFrame against requirements.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        max_null_pct: Maximum allowed percentage of null values (0-1)
        numeric_columns: Columns that must be numeric

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    # Check if DataFrame is empty
    if df is None or len(df) == 0:
        errors.append("DataFrame is empty")
        return False, errors

    # Check minimum rows
    if len(df) < min_rows:
        errors.append(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")

    # Check required columns
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            errors.append(f"Missing required columns: {missing}")

    # Check null percentage
    null_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
    if null_pct > max_null_pct:
        errors.append(
            f"Null percentage {null_pct:.2%} exceeds maximum {max_null_pct:.2%}"
        )

    # Check numeric columns
    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column '{col}' is not numeric")

    return len(errors) == 0, errors


def validate_ohlcv(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate OHLCV DataFrame"""
    required = ['open', 'high', 'low', 'close', 'volume']

    # Check columns (case-insensitive)
    df_cols_lower = [c.lower() for c in df.columns]
    missing = [c for c in required if c not in df_cols_lower]

    if missing:
        return False, [f"Missing OHLCV columns: {missing}"]

    errors = []

    # Standardize column names
    df.columns = df.columns.str.lower()

    # Check OHLC relationships
    if (df['high'] < df['low']).any():
        errors.append("Found rows where high < low")

    if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
        errors.append("Found rows where high < open or high < close")

    if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
        errors.append("Found rows where low > open or low > close")

    # Check for negative values
    if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
        errors.append("Found non-positive price values")

    if (df['volume'] < 0).any():
        errors.append("Found negative volume values")

    return len(errors) == 0, errors


# =============================================================================
# MATHEMATICAL UTILITIES
# =============================================================================

def safe_divide(
    numerator: Union[float, np.ndarray, pd.Series],
    denominator: Union[float, np.ndarray, pd.Series],
    fill_value: float = 0.0
) -> Union[float, np.ndarray, pd.Series]:
    """
    Safe division handling zero denominators.

    Args:
        numerator: Dividend
        denominator: Divisor
        fill_value: Value to use when denominator is zero

    Returns:
        Division result with zeros replaced by fill_value
    """
    if isinstance(denominator, (pd.Series, np.ndarray)):
        result = np.where(
            denominator != 0,
            numerator / denominator,
            fill_value
        )
        if isinstance(denominator, pd.Series):
            return pd.Series(result, index=denominator.index)
        return result
    else:
        return numerator / denominator if denominator != 0 else fill_value


def calculate_returns(
    prices: pd.Series,
    method: str = 'simple',
    periods: int = 1
) -> pd.Series:
    """
    Calculate returns from price series.

    Args:
        prices: Price series
        method: 'simple' or 'log'
        periods: Number of periods for return calculation

    Returns:
        Returns series
    """
    if method == 'simple':
        return prices.pct_change(periods=periods)
    elif method == 'log':
        return np.log(prices / prices.shift(periods))
    else:
        raise ValueError(f"Unknown method: {method}")


def rolling_window(
    data: Union[np.ndarray, pd.Series],
    window: int,
    step: int = 1
) -> Generator[np.ndarray, None, None]:
    """
    Generate rolling windows over data.

    Args:
        data: Input data
        window: Window size
        step: Step size between windows

    Yields:
        Window arrays
    """
    data = np.asarray(data)

    for i in range(0, len(data) - window + 1, step):
        yield data[i:i + window]


def ewm_weights(span: int, adjust: bool = True) -> np.ndarray:
    """Calculate exponentially weighted moving average weights"""
    alpha = 2.0 / (span + 1)

    if adjust:
        weights = (1 - alpha) ** np.arange(span)[::-1]
        weights /= weights.sum()
    else:
        weights = alpha * (1 - alpha) ** np.arange(span)[::-1]

    return weights


def normalize(
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    method: str = 'zscore',
    axis: int = 0
) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
    """
    Normalize data using various methods.

    Args:
        data: Input data
        method: 'zscore', 'minmax', 'robust'
        axis: Axis along which to normalize

    Returns:
        Normalized data
    """
    if method == 'zscore':
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        return safe_divide(data - mean, std)

    elif method == 'minmax':
        min_val = np.min(data, axis=axis, keepdims=True)
        max_val = np.max(data, axis=axis, keepdims=True)
        return safe_divide(data - min_val, max_val - min_val)

    elif method == 'robust':
        median = np.median(data, axis=axis, keepdims=True)
        q75, q25 = np.percentile(data, [75, 25], axis=axis, keepdims=True)
        iqr = q75 - q25
        return safe_divide(data - median, iqr)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


# =============================================================================
# CACHING UTILITIES
# =============================================================================

def disk_cache(cache_dir: str = ".cache", ttl_seconds: int = 3600):
    """
    Decorator for disk-based function result caching.

    Args:
        cache_dir: Directory for cache files
        ttl_seconds: Time-to-live for cache entries
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Create cache key from function name and arguments
            key_data = {
                'func': func.__name__,
                'args': args,
                'kwargs': kwargs
            }
            key = hashlib.md5(
                json.dumps(key_data, sort_keys=True, default=str).encode()
            ).hexdigest()

            cache_file = cache_path / f"{key}.pkl"

            # Check if cache exists and is valid
            if cache_file.exists():
                mtime = cache_file.stat().st_mtime
                if time.time() - mtime < ttl_seconds:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)

            # Execute function and cache result
            result = func(*args, **kwargs)

            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)

            return result

        return wrapper

    return decorator


def clear_disk_cache(cache_dir: str = ".cache") -> int:
    """Clear all disk cache files, return count of deleted files"""
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        return 0

    count = 0
    for file in cache_path.glob("*.pkl"):
        file.unlink()
        count += 1

    return count


# =============================================================================
# PARALLEL PROCESSING UTILITIES
# =============================================================================

def parallel_map(
    func: Callable[[T], Any],
    items: List[T],
    max_workers: Optional[int] = None,
    show_progress: bool = False
) -> List[Any]:
    """
    Apply function to items in parallel.

    Args:
        func: Function to apply
        items: List of items
        max_workers: Maximum number of workers
        show_progress: Whether to show progress

    Returns:
        List of results
    """
    results = [None] * len(items)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(func, item): i
            for i, item in enumerate(items)
        }

        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = e

            completed += 1
            if show_progress:
                print(f"\rProgress: {completed}/{len(items)}", end="")

        if show_progress:
            print()

    return results


# =============================================================================
# DATE/TIME UTILITIES
# =============================================================================

def market_hours_filter(
    df: pd.DataFrame,
    market_open: str = "09:30",
    market_close: str = "16:00",
    timezone: str = "America/New_York"
) -> pd.DataFrame:
    """
    Filter DataFrame to market hours only.

    Args:
        df: DataFrame with datetime index
        market_open: Market open time (HH:MM)
        market_close: Market close time (HH:MM)
        timezone: Timezone for market hours

    Returns:
        Filtered DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    # Localize if needed
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize(timezone)
    else:
        df = df.copy()
        df.index = df.index.tz_convert(timezone)

    # Filter by time
    open_time = pd.to_datetime(market_open).time()
    close_time = pd.to_datetime(market_close).time()

    mask = (df.index.time >= open_time) & (df.index.time <= close_time)

    return df[mask]


def is_trading_day(date: datetime, calendar: str = 'NYSE') -> bool:
    """Check if date is a trading day"""
    # Simplified check - exclude weekends
    # In production, use exchange_calendars package
    if date.weekday() >= 5:  # Saturday or Sunday
        return False

    # TODO: Add holiday check using exchange_calendars
    return True


def next_trading_day(date: datetime, calendar: str = 'NYSE') -> datetime:
    """Get next trading day"""
    next_day = date + timedelta(days=1)

    while not is_trading_day(next_day, calendar):
        next_day += timedelta(days=1)

    return next_day


# =============================================================================
# DATA STRUCTURE UTILITIES
# =============================================================================

@dataclass
class Result:
    """Generic result container with success/failure status"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def ok(cls, data: Any, metadata: Optional[Dict] = None) -> 'Result':
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def fail(cls, error: str, metadata: Optional[Dict] = None) -> 'Result':
        return cls(success=False, error=error, metadata=metadata)


class CircularBuffer:
    """Fixed-size circular buffer for streaming data"""

    def __init__(self, size: int):
        self.size = size
        self.buffer = [None] * size
        self.index = 0
        self.count = 0

    def append(self, item: Any) -> None:
        self.buffer[self.index] = item
        self.index = (self.index + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def get_all(self) -> List[Any]:
        if self.count < self.size:
            return self.buffer[:self.count]
        return self.buffer[self.index:] + self.buffer[:self.index]

    def __len__(self) -> int:
        return self.count

    def __iter__(self):
        return iter(self.get_all())
