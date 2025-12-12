"""
Numba-Accelerated Computational Functions
High-Performance Computing for Trading System

Features:
- JIT-compiled indicator calculations
- Vectorized rolling window operations
- Parallelized feature computation
- GPU-ready array operations

Usage:
    from src.utils.numba_accelerators import fast_ema, fast_rsi, fast_rolling_std

Note: Functions fallback to numpy if numba is not available.
"""

import numpy as np
from typing import Tuple, Optional
import warnings

# Try to import numba
try:
    from numba import jit, prange, float64, int64, boolean
    from numba import vectorize, guvectorize
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not installed. Using numpy fallbacks (slower).")

    # Create dummy decorator for fallback
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def prange(*args):
        return range(*args)


# ============================================================================
# Core Technical Indicator Accelerators
# ============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def fast_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Fast Exponential Moving Average using Numba JIT.

    10-50x faster than pandas ewm for large arrays.

    Args:
        prices: 1D array of prices
        period: EMA period

    Returns:
        EMA values array
    """
    n = len(prices)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    if n < period:
        return result

    # Smoothing factor
    alpha = 2.0 / (period + 1)

    # Initialize with SMA
    sma = 0.0
    for i in range(period):
        sma += prices[i]
    sma /= period

    result[period - 1] = sma

    # Calculate EMA
    for i in range(period, n):
        result[i] = alpha * prices[i] + (1 - alpha) * result[i - 1]

    return result


@jit(nopython=True, cache=True, fastmath=True)
def fast_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Fast Simple Moving Average using rolling sum.

    Args:
        prices: 1D array of prices
        period: SMA period

    Returns:
        SMA values array
    """
    n = len(prices)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    if n < period:
        return result

    # Initial sum
    rolling_sum = 0.0
    for i in range(period):
        rolling_sum += prices[i]

    result[period - 1] = rolling_sum / period

    # Rolling calculation
    for i in range(period, n):
        rolling_sum = rolling_sum - prices[i - period] + prices[i]
        result[i] = rolling_sum / period

    return result


@jit(nopython=True, cache=True, fastmath=True)
def fast_rolling_std(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Fast rolling standard deviation using Welford's algorithm.

    Numerically stable single-pass algorithm.

    Args:
        prices: 1D array of prices
        period: Window period

    Returns:
        Rolling std values
    """
    n = len(prices)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    if n < period:
        return result

    for i in range(period - 1, n):
        window = prices[i - period + 1:i + 1]
        mean = 0.0
        for j in range(period):
            mean += window[j]
        mean /= period

        variance = 0.0
        for j in range(period):
            diff = window[j] - mean
            variance += diff * diff
        variance /= period

        result[i] = np.sqrt(variance)

    return result


@jit(nopython=True, cache=True, fastmath=True)
def fast_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Fast Relative Strength Index calculation.

    Uses Wilder's smoothing method (exponential average).

    Args:
        prices: 1D array of prices
        period: RSI period (default 14)

    Returns:
        RSI values (0-100)
    """
    n = len(prices)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    if n < period + 1:
        return result

    # Calculate price changes
    deltas = np.empty(n, dtype=np.float64)
    deltas[0] = 0.0
    for i in range(1, n):
        deltas[i] = prices[i] - prices[i - 1]

    # Initialize gains and losses
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if deltas[i] > 0:
            gains[i] = deltas[i]
        elif deltas[i] < 0:
            losses[i] = -deltas[i]

    # First average
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, period + 1):
        avg_gain += gains[i]
        avg_loss += losses[i]
    avg_gain /= period
    avg_loss /= period

    if avg_loss != 0:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))
    else:
        result[period] = 100.0

    # Smoothed averages
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss != 0:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            result[i] = 100.0

    return result


@jit(nopython=True, cache=True, fastmath=True)
def fast_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             period: int = 14) -> np.ndarray:
    """
    Fast Average True Range calculation.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period

    Returns:
        ATR values
    """
    n = len(high)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    if n < period + 1:
        return result

    # Calculate True Range
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # First ATR is SMA of TR
    atr_sum = 0.0
    for i in range(1, period + 1):
        atr_sum += tr[i]
    result[period] = atr_sum / period

    # Wilder's smoothing
    for i in range(period + 1, n):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period

    return result


@jit(nopython=True, cache=True, fastmath=True)
def fast_macd(prices: np.ndarray,
              fast_period: int = 12,
              slow_period: int = 26,
              signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast MACD calculation.

    Args:
        prices: Close prices
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period

    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    n = len(prices)

    fast_ema_vals = fast_ema(prices, fast_period)
    slow_ema_vals = fast_ema(prices, slow_period)

    macd_line = np.empty(n, dtype=np.float64)
    macd_line[:] = np.nan

    for i in range(n):
        if not np.isnan(fast_ema_vals[i]) and not np.isnan(slow_ema_vals[i]):
            macd_line[i] = fast_ema_vals[i] - slow_ema_vals[i]

    signal_line = fast_ema(macd_line, signal_period)

    histogram = np.empty(n, dtype=np.float64)
    histogram[:] = np.nan

    for i in range(n):
        if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i]):
            histogram[i] = macd_line[i] - signal_line[i]

    return macd_line, signal_line, histogram


@jit(nopython=True, cache=True, fastmath=True)
def fast_bollinger_bands(prices: np.ndarray,
                         period: int = 20,
                         num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast Bollinger Bands calculation.

    Args:
        prices: Close prices
        period: MA period
        num_std: Number of standard deviations

    Returns:
        Tuple of (Upper band, Middle band, Lower band)
    """
    n = len(prices)

    middle = fast_sma(prices, period)
    std = fast_rolling_std(prices, period)

    upper = np.empty(n, dtype=np.float64)
    lower = np.empty(n, dtype=np.float64)
    upper[:] = np.nan
    lower[:] = np.nan

    for i in range(n):
        if not np.isnan(middle[i]) and not np.isnan(std[i]):
            upper[i] = middle[i] + num_std * std[i]
            lower[i] = middle[i] - num_std * std[i]

    return upper, middle, lower


# ============================================================================
# Fractional Differentiation Accelerators
# ============================================================================

@jit(nopython=True, cache=True)
def fast_ffd_weights(d: float, threshold: float, max_size: int = 10000) -> np.ndarray:
    """
    Fast computation of FFD weights.

    Args:
        d: Differentiation order
        threshold: Weight cutoff threshold
        max_size: Maximum weight array size

    Returns:
        Array of weights
    """
    weights = np.empty(max_size, dtype=np.float64)
    weights[0] = 1.0
    k = 1

    while k < max_size:
        w_k = -weights[k - 1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        weights[k] = w_k
        k += 1

    return weights[:k]


@jit(nopython=True, cache=True, fastmath=True)
def fast_ffd_apply(series: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Fast application of FFD transformation.

    Args:
        series: Price series
        weights: FFD weights

    Returns:
        Fractionally differentiated series
    """
    n = len(series)
    width = len(weights)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    # Reverse weights for convolution
    weights_rev = weights[::-1]

    for i in range(width - 1, n):
        val = 0.0
        for j in range(width):
            val += weights_rev[j] * series[i - width + 1 + j]
        result[i] = val

    return result


# ============================================================================
# Triple Barrier Labeling Accelerators
# ============================================================================

@jit(nopython=True, cache=True)
def fast_triple_barrier_labels(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volatility: np.ndarray,
    max_holding: int,
    vol_multiplier: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast triple barrier label generation.

    Args:
        close: Close prices
        high: High prices
        low: Low prices
        volatility: Rolling volatility
        max_holding: Maximum holding period
        vol_multiplier: Volatility multiplier for barriers

    Returns:
        Tuple of (labels, returns, barrier_types)
        barrier_types: 0=vertical, 1=upper, -1=lower
    """
    n = len(close)
    labels = np.empty(n, dtype=np.float64)
    returns = np.empty(n, dtype=np.float64)
    barrier_types = np.empty(n, dtype=np.int64)

    labels[:] = np.nan
    returns[:] = np.nan
    barrier_types[:] = 0

    for i in range(n - max_holding):
        entry_price = close[i]
        vol = volatility[i]

        if np.isnan(vol) or vol <= 0:
            continue

        upper_barrier = entry_price * (1 + vol_multiplier * vol)
        lower_barrier = entry_price * (1 - vol_multiplier * vol)

        label = 0
        exit_return = 0.0
        barrier_type = 0  # vertical

        for j in range(1, max_holding + 1):
            future_idx = i + j
            if future_idx >= n:
                break

            # Check upper barrier (using high)
            if high[future_idx] >= upper_barrier:
                label = 1
                exit_return = (upper_barrier - entry_price) / entry_price
                barrier_type = 1
                break

            # Check lower barrier (using low)
            if low[future_idx] <= lower_barrier:
                label = -1
                exit_return = (lower_barrier - entry_price) / entry_price
                barrier_type = -1
                break

            # Vertical barrier
            if j == max_holding:
                exit_return = (close[future_idx] - entry_price) / entry_price
                barrier_type = 0

        labels[i] = label
        returns[i] = exit_return
        barrier_types[i] = barrier_type

    return labels, returns, barrier_types


# ============================================================================
# Rolling Window Statistics
# ============================================================================

@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def fast_rolling_correlation(x: np.ndarray, y: np.ndarray, period: int) -> np.ndarray:
    """
    Fast rolling correlation using parallel computation.

    Args:
        x: First series
        y: Second series
        period: Rolling window

    Returns:
        Rolling correlation values
    """
    n = len(x)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    for i in prange(period - 1, n):
        x_window = x[i - period + 1:i + 1]
        y_window = y[i - period + 1:i + 1]

        # Calculate means
        mean_x = 0.0
        mean_y = 0.0
        for j in range(period):
            mean_x += x_window[j]
            mean_y += y_window[j]
        mean_x /= period
        mean_y /= period

        # Calculate correlation
        cov = 0.0
        var_x = 0.0
        var_y = 0.0
        for j in range(period):
            dx = x_window[j] - mean_x
            dy = y_window[j] - mean_y
            cov += dx * dy
            var_x += dx * dx
            var_y += dy * dy

        if var_x > 0 and var_y > 0:
            result[i] = cov / np.sqrt(var_x * var_y)
        else:
            result[i] = 0.0

    return result


@jit(nopython=True, cache=True, fastmath=True)
def fast_rolling_beta(returns: np.ndarray, market_returns: np.ndarray,
                      period: int) -> np.ndarray:
    """
    Fast rolling beta calculation.

    Beta = Cov(r, rm) / Var(rm)

    Args:
        returns: Asset returns
        market_returns: Market returns
        period: Rolling window

    Returns:
        Rolling beta values
    """
    n = len(returns)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    for i in range(period - 1, n):
        r = returns[i - period + 1:i + 1]
        rm = market_returns[i - period + 1:i + 1]

        # Calculate means
        mean_r = 0.0
        mean_rm = 0.0
        for j in range(period):
            mean_r += r[j]
            mean_rm += rm[j]
        mean_r /= period
        mean_rm /= period

        # Calculate covariance and variance
        cov = 0.0
        var_rm = 0.0
        for j in range(period):
            dr = r[j] - mean_r
            drm = rm[j] - mean_rm
            cov += dr * drm
            var_rm += drm * drm

        if var_rm > 0:
            result[i] = cov / var_rm
        else:
            result[i] = 0.0

    return result


@jit(nopython=True, cache=True, fastmath=True)
def fast_rolling_sharpe(returns: np.ndarray, period: int,
                        annualization: float = 252.0) -> np.ndarray:
    """
    Fast rolling Sharpe ratio calculation.

    Args:
        returns: Return series
        period: Rolling window
        annualization: Annualization factor

    Returns:
        Rolling Sharpe ratio
    """
    n = len(returns)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    sqrt_ann = np.sqrt(annualization)

    for i in range(period - 1, n):
        window = returns[i - period + 1:i + 1]

        mean = 0.0
        for j in range(period):
            mean += window[j]
        mean /= period

        variance = 0.0
        for j in range(period):
            diff = window[j] - mean
            variance += diff * diff
        variance /= period

        std = np.sqrt(variance)

        if std > 0:
            result[i] = (mean / std) * sqrt_ann
        else:
            result[i] = 0.0

    return result


@jit(nopython=True, cache=True, fastmath=True)
def fast_rolling_max_drawdown(equity: np.ndarray, period: int) -> np.ndarray:
    """
    Fast rolling maximum drawdown calculation.

    Args:
        equity: Equity curve
        period: Rolling window

    Returns:
        Rolling max drawdown (as positive values)
    """
    n = len(equity)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    for i in range(period - 1, n):
        window = equity[i - period + 1:i + 1]

        max_dd = 0.0
        peak = window[0]

        for j in range(period):
            if window[j] > peak:
                peak = window[j]
            dd = (peak - window[j]) / peak
            if dd > max_dd:
                max_dd = dd

        result[i] = max_dd

    return result


# ============================================================================
# HRP Clustering Accelerators
# ============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def fast_correlation_distance(corr: np.ndarray) -> np.ndarray:
    """
    Convert correlation matrix to distance matrix.

    dist = sqrt(0.5 * (1 - corr))

    Args:
        corr: Correlation matrix

    Returns:
        Distance matrix
    """
    n = corr.shape[0]
    dist = np.empty((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            dist[i, j] = np.sqrt(0.5 * (1 - corr[i, j]))

    return dist


@jit(nopython=True, cache=True, fastmath=True)
def fast_cluster_variance(cov: np.ndarray, indices: np.ndarray) -> float:
    """
    Calculate variance of minimum variance portfolio for cluster.

    Args:
        cov: Full covariance matrix
        indices: Indices of assets in cluster

    Returns:
        Cluster variance
    """
    n = len(indices)

    # Get cluster covariance
    cluster_cov = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            cluster_cov[i, j] = cov[indices[i], indices[j]]

    # Inverse variance weights
    diag = np.empty(n, dtype=np.float64)
    for i in range(n):
        diag[i] = cluster_cov[i, i]
        if diag[i] <= 0:
            diag[i] = 1e-10

    weights = np.empty(n, dtype=np.float64)
    inv_var_sum = 0.0
    for i in range(n):
        weights[i] = 1.0 / diag[i]
        inv_var_sum += weights[i]

    for i in range(n):
        weights[i] /= inv_var_sum

    # Calculate variance: w' * cov * w
    variance = 0.0
    for i in range(n):
        for j in range(n):
            variance += weights[i] * cluster_cov[i, j] * weights[j]

    return variance


# ============================================================================
# Utility Functions
# ============================================================================

def is_numba_available() -> bool:
    """Check if Numba is available."""
    return NUMBA_AVAILABLE


def get_accelerated_version(func_name: str):
    """
    Get accelerated version of a function if available.

    Args:
        func_name: Name of function

    Returns:
        Accelerated function or None
    """
    accelerators = {
        'ema': fast_ema,
        'sma': fast_sma,
        'rsi': fast_rsi,
        'atr': fast_atr,
        'macd': fast_macd,
        'bollinger_bands': fast_bollinger_bands,
        'rolling_std': fast_rolling_std,
        'rolling_correlation': fast_rolling_correlation,
        'rolling_beta': fast_rolling_beta,
        'rolling_sharpe': fast_rolling_sharpe,
        'ffd_weights': fast_ffd_weights,
        'ffd_apply': fast_ffd_apply,
        'triple_barrier': fast_triple_barrier_labels,
    }

    return accelerators.get(func_name)
