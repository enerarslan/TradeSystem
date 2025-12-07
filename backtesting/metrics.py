"""
Backtesting Metrics Module (FIXED)
==================================

Comprehensive performance metrics with CORRECT annualization.

CRITICAL FIX: 
- Auto-detection of periods_per_year from data frequency
- Validation of unrealistic metric values
- Proper handling of 15-minute data (6552 or 15794 periods/year)

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from config.settings import TradingConstants, get_logger

logger = get_logger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

TRADING_DAYS_PER_YEAR = TradingConstants.TRADING_DAYS_PER_YEAR  # 252
RISK_FREE_RATE = TradingConstants.RISK_FREE_RATE  # 0.05 (5%)

# CRITICAL: Periods per year for different timeframes
PERIODS_PER_YEAR = {
    "1min": 252 * 390,       # 98,280 (regular hours)
    "5min": 252 * 78,        # 19,656
    "15min": 252 * 26,       # 6,552 (regular hours)
    "15min_extended": 15794,  # Extended hours
    "30min": 252 * 13,       # 3,276
    "1hour": 252 * 6.5,      # 1,638
    "4hour": 252 * 1.625,    # 409.5
    "1day": 252,
    "1week": 52,
    "1month": 12,
}

# Default for 15-minute data with extended hours
DEFAULT_PERIODS_PER_YEAR_EXTENDED = 15794


# =============================================================================
# PERIOD DETECTION
# =============================================================================

def detect_periods_per_year(timestamps: list[datetime]) -> float:
    """
    Auto-detect periods per year from timestamp frequency.
    
    This is CRITICAL for accurate Sharpe/Sortino/Calmar ratios!
    
    Args:
        timestamps: Sorted list of timestamps
        
    Returns:
        Estimated periods per year
    
    Example:
        >>> timestamps = [datetime(2021,1,1,9,30), datetime(2021,1,1,9,45), ...]
        >>> detect_periods_per_year(timestamps)
        6552.0  # 15-minute data, regular hours
    """
    if len(timestamps) < 2:
        logger.warning("Not enough timestamps to detect frequency, defaulting to 252 (daily)")
        return 252.0
    
    # Calculate time differences
    diffs = []
    sample_size = min(1000, len(timestamps) - 1)
    
    for i in range(sample_size):
        diff = (timestamps[i + 1] - timestamps[i]).total_seconds()
        if diff > 0:  # Ignore duplicates
            diffs.append(diff)
    
    if not diffs:
        logger.warning("Could not calculate time differences, defaulting to 252")
        return 252.0
    
    # Use median to be robust to gaps (weekends, holidays)
    median_diff_seconds = np.median(diffs)
    
    # Map to approximate frequency
    freq_map = {
        60: "1min",
        300: "5min",
        900: "15min",
        1800: "30min",
        3600: "1hour",
        14400: "4hour",
        86400: "1day",
    }
    
    # Find closest frequency
    closest_freq = min(freq_map.keys(), key=lambda x: abs(x - median_diff_seconds))
    detected_freq = freq_map[closest_freq]
    
    logger.info(f"Detected data frequency: {detected_freq} (median interval: {median_diff_seconds:.0f}s)")
    
    # Check for extended hours
    hours = [t.hour for t in timestamps[:sample_size]]
    has_pre_market = any(h < 9 for h in hours)  # Before 9:30 AM
    has_after_hours = any(h >= 16 for h in hours)  # After 4 PM
    has_extended = has_pre_market or has_after_hours
    
    if has_extended and detected_freq == "15min":
        logger.info("Extended hours detected, using 15794 periods/year")
        return 15794.0
    
    return float(PERIODS_PER_YEAR.get(detected_freq, 252))


def validate_periods_per_year(
    periods_per_year: float,
    n_observations: int,
    date_range_days: float,
) -> float:
    """
    Validate and potentially correct periods_per_year based on actual data.
    
    Args:
        periods_per_year: Current setting
        n_observations: Number of data points
        date_range_days: Date range in days
        
    Returns:
        Validated periods_per_year
    """
    if date_range_days <= 0:
        return periods_per_year
    
    # Calculate implied periods per year from data
    implied_annual_periods = (n_observations / date_range_days) * 365
    
    # Check if current setting is way off
    ratio = periods_per_year / implied_annual_periods if implied_annual_periods > 0 else 1
    
    if ratio > 5 or ratio < 0.2:
        logger.warning(
            f"periods_per_year={periods_per_year:.0f} seems inconsistent with data "
            f"(implied: {implied_annual_periods:.0f}). Consider using auto-detection."
        )
    
    return periods_per_year


# =============================================================================
# RETURN METRICS
# =============================================================================

def calculate_returns(
    equity_curve: NDArray[np.float64],
    method: str = "simple",
) -> NDArray[np.float64]:
    """
    Calculate returns from equity curve.
    
    Args:
        equity_curve: Array of equity values
        method: 'simple' or 'log' returns
        
    Returns:
        Array of returns
    """
    if len(equity_curve) < 2:
        return np.array([])
    
    # Handle zeros and negatives
    equity_curve = np.maximum(equity_curve, 1e-10)
    
    if method == "log":
        returns = np.diff(np.log(equity_curve))
    else:  # simple
        returns = np.diff(equity_curve) / equity_curve[:-1]
    
    return returns


def total_return(equity_curve: NDArray[np.float64]) -> float:
    """
    Calculate total return.
    
    Args:
        equity_curve: Array of equity values
        
    Returns:
        Total return as decimal (e.g., 0.5 = 50%)
    """
    if len(equity_curve) < 2:
        return 0.0
    
    if equity_curve[0] == 0:
        return 0.0
    
    return float((equity_curve[-1] - equity_curve[0]) / equity_curve[0])


def annualized_return(
    equity_curve: NDArray[np.float64],
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate annualized return using CAGR formula.
    
    Formula:
        CAGR = (Final / Initial)^(periods_per_year / n_periods) - 1
    
    Args:
        equity_curve: Array of equity values
        periods_per_year: Trading periods per year (CRITICAL!)
        
    Returns:
        Annualized return as decimal
    """
    if len(equity_curve) < 2:
        return 0.0
    
    if equity_curve[0] <= 0:
        return 0.0
    
    n_periods = len(equity_curve) - 1
    total = equity_curve[-1] / equity_curve[0]
    
    if total <= 0:
        return -1.0  # Total loss
    
    # CAGR formula
    ann_return = total ** (periods_per_year / n_periods) - 1
    
    return float(ann_return)


def rolling_returns(
    equity_curve: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Calculate rolling returns."""
    if len(equity_curve) < window + 1:
        return np.array([])
    
    returns = []
    for i in range(window, len(equity_curve)):
        ret = (equity_curve[i] - equity_curve[i - window]) / equity_curve[i - window]
        returns.append(ret)
    
    return np.array(returns)


def monthly_returns(
    equity_curve: NDArray[np.float64],
    timestamps: list[datetime],
) -> dict[str, float]:
    """Calculate monthly returns."""
    if len(equity_curve) != len(timestamps):
        return {}
    
    monthly = {}
    current_month = None
    month_start_equity = None
    
    for i, (ts, equity) in enumerate(zip(timestamps, equity_curve)):
        month_key = ts.strftime("%Y-%m")
        
        if month_key != current_month:
            if current_month and month_start_equity:
                monthly[current_month] = (equity_curve[i-1] - month_start_equity) / month_start_equity
            current_month = month_key
            month_start_equity = equity
    
    # Last month
    if current_month and month_start_equity:
        monthly[current_month] = (equity_curve[-1] - month_start_equity) / month_start_equity
    
    return monthly


def yearly_returns(
    equity_curve: NDArray[np.float64],
    timestamps: list[datetime],
) -> dict[int, float]:
    """Calculate yearly returns."""
    if len(equity_curve) != len(timestamps):
        return {}
    
    yearly = {}
    current_year = None
    year_start_equity = None
    
    for i, (ts, equity) in enumerate(zip(timestamps, equity_curve)):
        year = ts.year
        
        if year != current_year:
            if current_year and year_start_equity:
                yearly[current_year] = (equity_curve[i-1] - year_start_equity) / year_start_equity
            current_year = year
            year_start_equity = equity
    
    # Last year
    if current_year and year_start_equity:
        yearly[current_year] = (equity_curve[-1] - year_start_equity) / year_start_equity
    
    return yearly


# =============================================================================
# VOLATILITY METRICS
# =============================================================================

def volatility(
    returns: NDArray[np.float64],
    annualize: bool = True,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate volatility (standard deviation of returns).
    
    Args:
        returns: Array of returns
        annualize: Whether to annualize
        periods_per_year: Trading periods per year (CRITICAL!)
        
    Returns:
        Volatility (annualized if requested)
    """
    if len(returns) < 2:
        return 0.0
    
    vol = float(np.std(returns, ddof=1))
    
    if annualize:
        vol *= np.sqrt(periods_per_year)
    
    return vol


def downside_volatility(
    returns: NDArray[np.float64],
    threshold: float = 0.0,
    annualize: bool = True,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate downside volatility (semi-deviation).
    
    Args:
        returns: Array of returns
        threshold: Minimum acceptable return (MAR)
        annualize: Whether to annualize
        periods_per_year: Trading periods per year
        
    Returns:
        Downside volatility
    """
    if len(returns) < 2:
        return 0.0
    
    downside_returns = returns[returns < threshold]
    
    if len(downside_returns) < 2:
        return 0.0
    
    downside_dev = float(np.std(downside_returns, ddof=1))
    
    if annualize:
        downside_dev *= np.sqrt(periods_per_year)
    
    return downside_dev


def upside_volatility(
    returns: NDArray[np.float64],
    threshold: float = 0.0,
    annualize: bool = True,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate upside volatility."""
    if len(returns) < 2:
        return 0.0
    
    upside_returns = returns[returns > threshold]
    
    if len(upside_returns) < 2:
        return 0.0
    
    upside_dev = float(np.std(upside_returns, ddof=1))
    
    if annualize:
        upside_dev *= np.sqrt(periods_per_year)
    
    return upside_dev


# =============================================================================
# DRAWDOWN METRICS
# =============================================================================

def calculate_drawdown_series(equity_curve: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate drawdown series."""
    if len(equity_curve) < 1:
        return np.array([])
    
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    
    return drawdowns


def max_drawdown(equity_curve: NDArray[np.float64]) -> float:
    """
    Calculate maximum drawdown.
    
    Returns:
        Maximum drawdown as negative decimal (e.g., -0.20 = -20%)
    """
    if len(equity_curve) < 2:
        return 0.0
    
    drawdowns = calculate_drawdown_series(equity_curve)
    return float(np.min(drawdowns))


def max_drawdown_duration(
    equity_curve: NDArray[np.float64],
    timestamps: list[datetime] | None = None,
) -> int:
    """
    Calculate maximum drawdown duration in periods.
    """
    if len(equity_curve) < 2:
        return 0
    
    running_max = np.maximum.accumulate(equity_curve)
    
    # Track drawdown periods
    in_drawdown = equity_curve < running_max
    max_duration = 0
    current_duration = 0
    
    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    
    return max_duration


def average_drawdown(equity_curve: NDArray[np.float64]) -> float:
    """Calculate average drawdown."""
    drawdowns = calculate_drawdown_series(equity_curve)
    if len(drawdowns) == 0:
        return 0.0
    
    return float(np.mean(drawdowns[drawdowns < 0])) if np.any(drawdowns < 0) else 0.0


def drawdown_details(
    equity_curve: NDArray[np.float64],
    timestamps: list[datetime] | None = None,
) -> list[dict[str, Any]]:
    """Get details of all drawdown periods."""
    if len(equity_curve) < 2:
        return []
    
    drawdowns = calculate_drawdown_series(equity_curve)
    details = []
    
    in_drawdown = False
    start_idx = 0
    
    for i, dd in enumerate(drawdowns):
        if dd < 0 and not in_drawdown:
            in_drawdown = True
            start_idx = i
        elif dd >= 0 and in_drawdown:
            in_drawdown = False
            end_idx = i
            
            dd_slice = drawdowns[start_idx:end_idx]
            details.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "trough_idx": start_idx + np.argmin(dd_slice),
                "max_drawdown": float(np.min(dd_slice)),
                "duration": end_idx - start_idx,
                "start_date": timestamps[start_idx] if timestamps else None,
                "end_date": timestamps[end_idx] if timestamps else None,
            })
    
    return details


def ulcer_index(equity_curve: NDArray[np.float64]) -> float:
    """Calculate Ulcer Index."""
    if len(equity_curve) < 2:
        return 0.0
    
    drawdowns = calculate_drawdown_series(equity_curve)
    return float(np.sqrt(np.mean(drawdowns ** 2)))


def pain_index(equity_curve: NDArray[np.float64]) -> float:
    """Calculate Pain Index (mean of absolute drawdowns)."""
    if len(equity_curve) < 2:
        return 0.0
    
    drawdowns = calculate_drawdown_series(equity_curve)
    return float(np.mean(np.abs(drawdowns)))


# =============================================================================
# VALUE AT RISK METRICS
# =============================================================================

def var_historical(
    returns: NDArray[np.float64],
    confidence: float = 0.95,
) -> float:
    """
    Calculate Historical VaR.
    
    Returns:
        VaR as positive decimal (loss amount)
    """
    if len(returns) < 10:
        return 0.0
    
    return float(-np.percentile(returns, (1 - confidence) * 100))


def var_parametric(
    returns: NDArray[np.float64],
    confidence: float = 0.95,
) -> float:
    """Calculate Parametric (Gaussian) VaR."""
    if len(returns) < 10:
        return 0.0
    
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    z_score = scipy_stats.norm.ppf(1 - confidence)
    
    return float(-(mean + z_score * std))


def var_cornish_fisher(
    returns: NDArray[np.float64],
    confidence: float = 0.95,
) -> float:
    """Calculate Cornish-Fisher VaR (adjusted for skewness/kurtosis)."""
    if len(returns) < 30:
        return var_parametric(returns, confidence)
    
    z = scipy_stats.norm.ppf(1 - confidence)
    s = scipy_stats.skew(returns)
    k = scipy_stats.kurtosis(returns)
    
    # Cornish-Fisher expansion
    z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * (k - 3) / 24 - (2*z**3 - 5*z) * s**2 / 36
    
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    
    return float(-(mean + z_cf * std))


def cvar(
    returns: NDArray[np.float64],
    confidence: float = 0.95,
) -> float:
    """
    Calculate Conditional VaR (Expected Shortfall).
    
    Returns:
        CVaR as positive decimal
    """
    if len(returns) < 10:
        return 0.0
    
    var = var_historical(returns, confidence)
    tail_losses = returns[returns < -var]
    
    if len(tail_losses) == 0:
        return var
    
    return float(-np.mean(tail_losses))


# =============================================================================
# RISK-ADJUSTED RETURN METRICS
# =============================================================================

def sharpe_ratio(
    returns: NDArray[np.float64],
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate Sharpe Ratio.
    
    Formula:
        Sharpe = (Return - Risk-Free) / Volatility * sqrt(periods_per_year)
    
    CRITICAL: periods_per_year must match data frequency!
    - Daily data: 252
    - 15-min data (regular): 6552
    - 15-min data (extended): 15794
    
    A realistic Sharpe ratio is typically between -2 and +3.
    Values above 5 should trigger investigation.
    """
    if len(returns) < 2:
        return 0.0
    
    # Convert annual risk-free to per-period
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    excess_returns = returns - rf_per_period
    std = np.std(excess_returns, ddof=1)
    
    if std == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / std
    
    # Annualize
    sharpe_annual = float(sharpe * np.sqrt(periods_per_year))
    
    # Sanity check
    if abs(sharpe_annual) > 10:
        logger.warning(
            f"Sharpe ratio {sharpe_annual:.2f} is unusually high. "
            f"Verify periods_per_year={periods_per_year:.0f} is correct for your data."
        )
    
    return sharpe_annual


def sortino_ratio(
    returns: NDArray[np.float64],
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate Sortino Ratio.
    
    Like Sharpe but uses downside volatility only.
    """
    if len(returns) < 2:
        return 0.0
    
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns - rf_per_period
    
    downside_vol = downside_volatility(returns, rf_per_period, annualize=False)
    
    if downside_vol == 0:
        return 0.0 if np.mean(excess_returns) <= 0 else float('inf')
    
    sortino = np.mean(excess_returns) / downside_vol
    sortino_annual = float(sortino * np.sqrt(periods_per_year))
    
    # Sanity check
    if abs(sortino_annual) > 20:
        logger.warning(
            f"Sortino ratio {sortino_annual:.2f} is unusually high. "
            f"Verify periods_per_year={periods_per_year:.0f} is correct."
        )
    
    return sortino_annual


def calmar_ratio(
    equity_curve: NDArray[np.float64],
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate Calmar Ratio.
    
    Formula:
        Calmar = Annual Return / |Max Drawdown|
    """
    if len(equity_curve) < 2:
        return 0.0
    
    ann_return = annualized_return(equity_curve, periods_per_year)
    mdd = abs(max_drawdown(equity_curve))
    
    if mdd == 0:
        return 0.0 if ann_return <= 0 else float('inf')
    
    calmar = float(ann_return / mdd)
    
    # Sanity check
    if abs(calmar) > 50:
        logger.warning(
            f"Calmar ratio {calmar:.2f} is unusually high. "
            f"Max drawdown: {mdd:.2%}, Annual return: {ann_return:.2%}"
        )
    
    return calmar


def omega_ratio(
    returns: NDArray[np.float64],
    threshold: float = 0.0,
) -> float:
    """
    Calculate Omega Ratio.
    
    Formula:
        Omega = Sum(Gains above threshold) / Sum(Losses below threshold)
    """
    if len(returns) < 2:
        return 0.0
    
    excess = returns - threshold
    gains = np.sum(excess[excess > 0])
    losses = -np.sum(excess[excess < 0])
    
    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    
    return float(gains / losses)


def information_ratio(
    returns: NDArray[np.float64],
    benchmark_returns: NDArray[np.float64],
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate Information Ratio."""
    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        return 0.0
    
    active_returns = returns - benchmark_returns
    tracking_error = np.std(active_returns, ddof=1)
    
    if tracking_error == 0:
        return 0.0
    
    ir = np.mean(active_returns) / tracking_error
    
    return float(ir * np.sqrt(periods_per_year))


def treynor_ratio(
    returns: NDArray[np.float64],
    benchmark_returns: NDArray[np.float64],
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate Treynor Ratio."""
    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        return 0.0
    
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    covariance = np.cov(returns, benchmark_returns)[0, 1]
    benchmark_var = np.var(benchmark_returns, ddof=1)
    
    if benchmark_var == 0:
        return 0.0
    
    beta = covariance / benchmark_var
    
    if beta == 0:
        return 0.0
    
    excess_return = np.mean(returns) - rf_per_period
    
    return float(excess_return / beta * periods_per_year)


def gain_to_pain_ratio(equity_curve: NDArray[np.float64]) -> float:
    """Calculate Gain to Pain Ratio."""
    if len(equity_curve) < 2:
        return 0.0
    
    total_ret = total_return(equity_curve)
    pain = pain_index(equity_curve)
    
    if pain == 0:
        return float('inf') if total_ret > 0 else 0.0
    
    return float(total_ret / pain)


def ulcer_performance_index(
    equity_curve: NDArray[np.float64],
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate Ulcer Performance Index (Martin Ratio)."""
    if len(equity_curve) < 2:
        return 0.0
    
    ann_return = annualized_return(equity_curve, periods_per_year)
    ui = ulcer_index(equity_curve)
    
    if ui == 0:
        return 0.0
    
    return float((ann_return - risk_free_rate) / ui)


# =============================================================================
# TRADE STATISTICS
# =============================================================================

@dataclass
class TradeStats:
    """Statistics calculated from a list of trades."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0
    
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    expectancy: float = 0.0
    
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    avg_holding_bars: float = 0.0
    avg_winner_bars: float = 0.0
    avg_loser_bars: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "net_profit": self.net_profit,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "avg_trade": self.avg_trade,
            "expectancy": self.expectancy,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
        }


def calculate_trade_stats(trades: list) -> TradeStats:
    """Calculate comprehensive trade statistics."""
    stats = TradeStats()
    
    if not trades:
        return stats
    
    stats.total_trades = len(trades)
    
    # Separate wins and losses
    pnls = []
    win_pnls = []
    loss_pnls = []
    
    for trade in trades:
        pnl = trade.pnl if hasattr(trade, 'pnl') else trade.get('pnl', 0)
        pnls.append(pnl)
        
        if pnl > 0:
            win_pnls.append(pnl)
        elif pnl < 0:
            loss_pnls.append(pnl)
    
    stats.winning_trades = len(win_pnls)
    stats.losing_trades = len(loss_pnls)
    
    if stats.total_trades > 0:
        stats.win_rate = stats.winning_trades / stats.total_trades
    
    # Profit/Loss metrics
    stats.gross_profit = sum(win_pnls) if win_pnls else 0
    stats.gross_loss = abs(sum(loss_pnls)) if loss_pnls else 0
    stats.net_profit = stats.gross_profit - stats.gross_loss
    
    if stats.gross_loss > 0:
        stats.profit_factor = stats.gross_profit / stats.gross_loss
    else:
        stats.profit_factor = float('inf') if stats.gross_profit > 0 else 0
    
    # Average trade metrics
    if win_pnls:
        stats.avg_win = np.mean(win_pnls)
        stats.largest_win = max(win_pnls)
    
    if loss_pnls:
        stats.avg_loss = np.mean(loss_pnls)
        stats.largest_loss = min(loss_pnls)
    
    if pnls:
        stats.avg_trade = np.mean(pnls)
    
    # Expectancy
    stats.expectancy = (stats.win_rate * stats.avg_win) + ((1 - stats.win_rate) * stats.avg_loss)
    
    # Consecutive wins/losses
    current_wins = 0
    current_losses = 0
    max_wins = 0
    max_losses = 0
    
    for pnl in pnls:
        if pnl > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        elif pnl < 0:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)
        else:
            current_wins = 0
            current_losses = 0
    
    stats.max_consecutive_wins = max_wins
    stats.max_consecutive_losses = max_losses
    
    return stats


# =============================================================================
# DISTRIBUTION METRICS
# =============================================================================

def skewness(returns: NDArray[np.float64]) -> float:
    """Calculate skewness of returns."""
    if len(returns) < 3:
        return 0.0
    return float(scipy_stats.skew(returns))


def kurtosis(returns: NDArray[np.float64]) -> float:
    """Calculate excess kurtosis."""
    if len(returns) < 4:
        return 0.0
    return float(scipy_stats.kurtosis(returns))


def tail_ratio(returns: NDArray[np.float64]) -> float:
    """Calculate tail ratio (right tail / left tail)."""
    if len(returns) < 10:
        return 0.0
    
    right_tail = np.percentile(returns, 95)
    left_tail = np.percentile(returns, 5)
    
    if left_tail == 0:
        return float('inf') if right_tail > 0 else 0.0
    
    return float(abs(right_tail / left_tail))


def common_sense_ratio(
    returns: NDArray[np.float64],
    equity_curve: NDArray[np.float64],
) -> float:
    """Calculate Common Sense Ratio."""
    tr = tail_ratio(returns)
    gp = gain_to_pain_ratio(equity_curve)
    
    return float(tr * gp)


def outlier_ratio(
    returns: NDArray[np.float64],
    threshold: float = 3.0,
) -> float:
    """Calculate fraction of returns beyond N standard deviations."""
    if len(returns) < 2:
        return 0.0
    
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    
    if std == 0:
        return 0.0
    
    z_scores = np.abs((returns - mean) / std)
    outliers = np.sum(z_scores > threshold)
    
    return float(outliers / len(returns))


# =============================================================================
# COMPREHENSIVE METRICS CLASS
# =============================================================================

@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    
    # Identification
    strategy_name: str = ""
    start_date: datetime | None = None
    end_date: datetime | None = None
    initial_capital: float = 0.0
    final_capital: float = 0.0
    
    # Return metrics
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    monthly_returns: dict[str, float] = field(default_factory=dict)
    yearly_returns: dict[int, float] = field(default_factory=dict)
    
    # Risk metrics
    volatility: float = 0.0
    annualized_volatility: float = 0.0
    downside_volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    average_drawdown: float = 0.0
    
    # VaR metrics
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    
    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    information_ratio: float = 0.0
    
    # Trade statistics
    trade_stats: TradeStats = field(default_factory=TradeStats)
    
    # Distribution metrics
    skewness: float = 0.0
    kurtosis: float = 0.0
    tail_ratio: float = 0.0
    
    # Additional
    best_month: float = 0.0
    worst_month: float = 0.0
    positive_months: int = 0
    negative_months: int = 0
    
    # Benchmark
    benchmark_return: float | None = None
    alpha: float | None = None
    beta: float | None = None
    
    # Metadata
    periods_per_year_used: float = 252.0  # Track what was used!
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy_name": self.strategy_name,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "annualized_volatility": self.annualized_volatility,
            "downside_volatility": self.downside_volatility,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "cvar_99": self.cvar_99,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "omega_ratio": self.omega_ratio,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "tail_ratio": self.tail_ratio,
            "trade_stats": {
                "total_trades": self.trade_stats.total_trades,
                "win_rate": self.trade_stats.win_rate,
                "profit_factor": self.trade_stats.profit_factor,
                "avg_trade": self.trade_stats.avg_trade,
                "expectancy": self.trade_stats.expectancy,
            },
            "periods_per_year_used": self.periods_per_year_used,
        }


class MetricsCalculator:
    """
    Calculator for comprehensive performance metrics.
    
    CRITICAL: Ensure periods_per_year matches your data frequency!
    """
    
    def __init__(
        self,
        equity_curve: NDArray[np.float64],
        timestamps: list[datetime] | None = None,
        trades: list | None = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = RISK_FREE_RATE,
        periods_per_year: float | None = None,  # Auto-detect if None
        benchmark_returns: NDArray[np.float64] | None = None,
        strategy_name: str = "",
    ):
        """
        Initialize calculator.
        
        Args:
            equity_curve: Array of equity values
            timestamps: List of timestamps (for auto-detection)
            trades: List of trade objects
            initial_capital: Starting capital
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year (auto-detect if None)
            benchmark_returns: Optional benchmark returns
            strategy_name: Strategy identifier
        """
        self.equity_curve = np.asarray(equity_curve)
        self.timestamps = timestamps or []
        self.trades = trades or []
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.benchmark_returns = benchmark_returns
        self.strategy_name = strategy_name
        
        # Auto-detect periods_per_year if not provided
        if periods_per_year is None and self.timestamps:
            self.periods_per_year = detect_periods_per_year(self.timestamps)
            logger.info(f"Auto-detected periods_per_year: {self.periods_per_year:.0f}")
        else:
            self.periods_per_year = periods_per_year or TRADING_DAYS_PER_YEAR
        
        # Calculate returns
        self.returns = calculate_returns(self.equity_curve)
    
    def calculate_all(self) -> PerformanceReport:
        """Calculate all performance metrics."""
        report = PerformanceReport()
        report.strategy_name = self.strategy_name
        report.initial_capital = self.initial_capital
        report.final_capital = float(self.equity_curve[-1]) if len(self.equity_curve) > 0 else 0.0
        report.periods_per_year_used = self.periods_per_year
        
        if self.timestamps:
            report.start_date = self.timestamps[0]
            report.end_date = self.timestamps[-1]
        
        # Return metrics
        report.total_return = report.final_capital - report.initial_capital
        report.total_return_pct = total_return(self.equity_curve)
        report.annualized_return = annualized_return(self.equity_curve, self.periods_per_year)
        
        if self.timestamps:
            report.monthly_returns = monthly_returns(self.equity_curve, self.timestamps)
            report.yearly_returns = yearly_returns(self.equity_curve, self.timestamps)
            
            if report.monthly_returns:
                monthly_vals = list(report.monthly_returns.values())
                report.best_month = max(monthly_vals)
                report.worst_month = min(monthly_vals)
                report.positive_months = sum(1 for v in monthly_vals if v > 0)
                report.negative_months = sum(1 for v in monthly_vals if v < 0)
        
        # Risk metrics
        report.volatility = volatility(self.returns, annualize=False)
        report.annualized_volatility = volatility(self.returns, True, self.periods_per_year)
        report.downside_volatility = downside_volatility(self.returns, 0, True, self.periods_per_year)
        report.max_drawdown = max_drawdown(self.equity_curve)
        report.max_drawdown_duration = max_drawdown_duration(self.equity_curve, self.timestamps)
        report.average_drawdown = average_drawdown(self.equity_curve)
        
        # VaR metrics
        report.var_95 = var_historical(self.returns, 0.95)
        report.var_99 = var_historical(self.returns, 0.99)
        report.cvar_95 = cvar(self.returns, 0.95)
        report.cvar_99 = cvar(self.returns, 0.99)
        
        # Risk-adjusted metrics - CRITICAL: use correct periods_per_year!
        report.sharpe_ratio = sharpe_ratio(self.returns, self.risk_free_rate, self.periods_per_year)
        report.sortino_ratio = sortino_ratio(self.returns, self.risk_free_rate, self.periods_per_year)
        report.calmar_ratio = calmar_ratio(self.equity_curve, self.periods_per_year)
        report.omega_ratio = omega_ratio(self.returns, 0)
        
        if self.benchmark_returns is not None and len(self.benchmark_returns) == len(self.returns):
            report.information_ratio = information_ratio(
                self.returns, self.benchmark_returns, self.periods_per_year
            )
            
            # Calculate alpha and beta
            cov_matrix = np.cov(self.returns, self.benchmark_returns)
            benchmark_var = np.var(self.benchmark_returns, ddof=1)
            
            if benchmark_var > 0:
                report.beta = float(cov_matrix[0, 1] / benchmark_var)
                rf_period = (1 + self.risk_free_rate) ** (1 / self.periods_per_year) - 1
                expected_return = rf_period + report.beta * (np.mean(self.benchmark_returns) - rf_period)
                report.alpha = float((np.mean(self.returns) - expected_return) * self.periods_per_year)
            
            report.benchmark_return = float(np.prod(1 + self.benchmark_returns) - 1)
        
        # Trade statistics
        if self.trades:
            report.trade_stats = calculate_trade_stats(self.trades)
        
        # Distribution metrics
        report.skewness = skewness(self.returns)
        report.kurtosis = kurtosis(self.returns)
        report.tail_ratio = tail_ratio(self.returns)
        
        return report


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "TRADING_DAYS_PER_YEAR",
    "RISK_FREE_RATE",
    "PERIODS_PER_YEAR",
    
    # Period detection
    "detect_periods_per_year",
    "validate_periods_per_year",
    
    # Return metrics
    "calculate_returns",
    "total_return",
    "annualized_return",
    "rolling_returns",
    "monthly_returns",
    "yearly_returns",
    
    # Volatility metrics
    "volatility",
    "downside_volatility",
    "upside_volatility",
    
    # Drawdown metrics
    "calculate_drawdown_series",
    "max_drawdown",
    "max_drawdown_duration",
    "average_drawdown",
    "drawdown_details",
    "ulcer_index",
    "pain_index",
    
    # VaR metrics
    "var_historical",
    "var_parametric",
    "var_cornish_fisher",
    "cvar",
    
    # Risk-adjusted metrics
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "omega_ratio",
    "information_ratio",
    "treynor_ratio",
    "gain_to_pain_ratio",
    "ulcer_performance_index",
    
    # Trade statistics
    "TradeStats",
    "calculate_trade_stats",
    
    # Distribution metrics
    "skewness",
    "kurtosis",
    "tail_ratio",
    "common_sense_ratio",
    "outlier_ratio",
    
    # Classes
    "PerformanceReport",
    "MetricsCalculator",
]