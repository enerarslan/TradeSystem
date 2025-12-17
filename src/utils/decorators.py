"""
Utility decorators for AlphaTrade system.

This module provides decorators for:
- Timing and performance measurement
- Retry logic with exponential backoff
- Memoization/caching
- Input validation
- Execution logging
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, ParamSpec, TypeVar

from loguru import logger

P = ParamSpec("P")
T = TypeVar("T")


def timer(func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator to measure and log function execution time.

    Args:
        func: Function to time

    Returns:
        Wrapped function that logs execution time

    Example:
        @timer
        def slow_function():
            time.sleep(1)
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.debug(f"{func.__module__}.{func.__name__} executed in {elapsed:.2f}ms")

    return wrapper


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to retry a function with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Decorator function

    Example:
        @retry(max_attempts=3, delay=1.0, exceptions=(ConnectionError,))
        def fetch_data():
            ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            current_delay = delay
            last_exception: Exception | None = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for "
                            f"{func.__name__}: {e}. Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}"
                        )

            if last_exception:
                raise last_exception
            raise RuntimeError(f"Unexpected error in retry decorator for {func.__name__}")

        return wrapper

    return decorator


def memoize(
    maxsize: int = 128,
    typed: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to cache function results.

    Uses functools.lru_cache under the hood with additional logging.

    Args:
        maxsize: Maximum cache size (None for unlimited)
        typed: Whether to cache different types separately

    Returns:
        Decorator function

    Example:
        @memoize(maxsize=256)
        def expensive_calculation(x, y):
            ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        cached_func = functools.lru_cache(maxsize=maxsize, typed=typed)(func)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Convert kwargs to hashable format for caching
            key = args + tuple(sorted(kwargs.items()))
            return cached_func(*args)

        # Expose cache info
        wrapper.cache_info = cached_func.cache_info  # type: ignore
        wrapper.cache_clear = cached_func.cache_clear  # type: ignore

        return wrapper

    return decorator


def validate_input(
    validators: dict[str, Callable[[Any], bool]] | None = None,
    **kwarg_validators: Callable[[Any], bool],
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to validate function inputs.

    Args:
        validators: Dictionary mapping parameter names to validator functions
        **kwarg_validators: Validator functions as keyword arguments

    Returns:
        Decorator function

    Raises:
        ValueError: If validation fails

    Example:
        @validate_input(x=lambda v: v > 0, y=lambda v: isinstance(v, str))
        def process(x, y):
            ...
    """
    all_validators = validators or {}
    all_validators.update(kwarg_validators)

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Get function signature to map args to parameter names
            import inspect

            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Validate each parameter
            for param_name, validator in all_validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if not validator(value):
                        raise ValueError(
                            f"Validation failed for parameter '{param_name}' "
                            f"with value {value!r} in {func.__name__}"
                        )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def log_execution(
    level: str = "DEBUG",
    log_args: bool = True,
    log_result: bool = False,
    max_str_len: int = 100,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to log function execution details.

    Args:
        level: Logging level
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        max_str_len: Maximum string length for logged values

    Returns:
        Decorator function

    Example:
        @log_execution(level="INFO", log_args=True)
        def important_function(data):
            ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            log_func = getattr(logger, level.lower(), logger.debug)
            func_name = f"{func.__module__}.{func.__name__}"

            # Log function entry
            if log_args:
                # Truncate long arguments
                def truncate(v: Any) -> str:
                    s = repr(v)
                    if len(s) > max_str_len:
                        return s[: max_str_len - 3] + "..."
                    return s

                args_str = ", ".join(truncate(a) for a in args)
                kwargs_str = ", ".join(f"{k}={truncate(v)}" for k, v in kwargs.items())
                all_args = ", ".join(filter(None, [args_str, kwargs_str]))
                log_func(f"CALL {func_name}({all_args})")
            else:
                log_func(f"CALL {func_name}")

            # Execute function
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)

                # Log success
                elapsed = (time.perf_counter() - start_time) * 1000
                if log_result:
                    result_str = repr(result)
                    if len(result_str) > max_str_len:
                        result_str = result_str[: max_str_len - 3] + "..."
                    log_func(f"DONE {func_name} -> {result_str} ({elapsed:.2f}ms)")
                else:
                    log_func(f"DONE {func_name} ({elapsed:.2f}ms)")

                return result

            except Exception as e:
                elapsed = (time.perf_counter() - start_time) * 1000
                logger.error(f"FAIL {func_name}: {type(e).__name__}: {e} ({elapsed:.2f}ms)")
                raise

        return wrapper

    return decorator


def singleton(cls: type[T]) -> type[T]:
    """
    Decorator to make a class a singleton.

    Args:
        cls: Class to make singleton

    Returns:
        Singleton class

    Example:
        @singleton
        class Database:
            ...
    """
    instances: dict[type, Any] = {}

    @functools.wraps(cls)
    def get_instance(*args: Any, **kwargs: Any) -> T:
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance  # type: ignore


def deprecated(
    reason: str = "",
    version: str | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to mark a function as deprecated.

    Args:
        reason: Reason for deprecation or suggested alternative
        version: Version when function was deprecated

    Returns:
        Decorator function

    Example:
        @deprecated(reason="Use new_function() instead", version="2.0")
        def old_function():
            ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            message = f"{func.__name__} is deprecated"
            if version:
                message += f" since version {version}"
            if reason:
                message += f". {reason}"

            logger.warning(message)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def require_columns(
    *columns: str,
    df_param: str = "df",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to ensure DataFrame has required columns.

    Args:
        *columns: Required column names
        df_param: Name of the DataFrame parameter

    Returns:
        Decorator function

    Raises:
        ValueError: If required columns are missing

    Example:
        @require_columns("open", "high", "low", "close", df_param="data")
        def calculate_indicator(data):
            ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            import inspect
            import pandas as pd

            # Get the DataFrame argument
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            df = bound.arguments.get(df_param)

            if df is None:
                raise ValueError(f"Parameter '{df_param}' not found in {func.__name__}")

            if not isinstance(df, pd.DataFrame):
                raise TypeError(
                    f"Parameter '{df_param}' must be a DataFrame, got {type(df).__name__}"
                )

            # Check for required columns
            missing = set(columns) - set(df.columns)
            if missing:
                raise ValueError(
                    f"Missing required columns in {func.__name__}: {missing}"
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator
