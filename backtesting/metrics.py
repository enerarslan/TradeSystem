"""
Backtesting Metrics Module
==========================

Comprehensive performance metrics for strategy evaluation.
Implements institutional-grade analytics used by quantitative funds.

Metric Categories:
- Return Metrics: Total, annual, monthly, daily returns
- Risk Metrics: Volatility, VaR, CVaR, drawdown
- Risk-Adjusted: Sharpe, Sortino, Calmar, Omega, Information ratio
- Trade Statistics: Win rate, profit factor, expectancy
- Distribution: Skewness, kurtosis, tail ratio

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
PERIODS_PER_YEAR = {
    "1min": 252 * 390,      # 390 minutes per trading day
    "5min": 252 * 78,
    "15min": 252 * 26,
    "30min": 252 * 13,
    "1hour": 252 * 6.5,
    "4hour": 252 * 1.625,
    "1day": 252,
    "1week": 52,
    "1month": 12,
}


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
    
    if method == "log":
        returns = np.diff(np.log(equity_curve))
    else:  # simple
        returns = np.diff(equity_curve) / equity_curve[:-1]
    
    return returns


def total_return(equity_curve: NDArray[np.float64]) -> float:
    """
    Calculate total return.
    
    Formula:
        Total Return = (Final Value - Initial Value) / Initial Value
    
    Args:
        equity_curve: Array of equity values
        
    Returns:
        Total return as decimal
    """
    if len(equity_curve) < 2:
        return 0.0
    
    return (equity_curve[-1] - equity_curve[0]) / equity_curve[0]


def annualized_return(
    equity_curve: NDArray[np.float64],
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate annualized return using CAGR formula.
    
    Formula:
        CAGR = (Final / Initial) ^ (periods_per_year / n_periods) - 1
    
    Args:
        equity_curve: Array of equity values
        periods_per_year: Number of periods in a year
        
    Returns:
        Annualized return as decimal
    """
    if len(equity_curve) < 2:
        return 0.0
    
    n_periods = len(equity_curve) - 1
    total = total_return(equity_curve)
    
    if total <= -1:
        return -1.0
    
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    
    return (1 + total) ** (1 / years) - 1


def rolling_returns(
    returns: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """
    Calculate rolling cumulative returns.
    
    Args:
        returns: Array of period returns
        window: Rolling window size
        
    Returns:
        Array of rolling returns
    """
    if len(returns) < window:
        return np.array([])
    
    rolling = np.zeros(len(returns) - window + 1)
    for i in range(len(rolling)):
        rolling[i] = np.prod(1 + returns[i:i + window]) - 1
    
    return rolling


def monthly_returns(
    equity_curve: NDArray[np.float64],
    timestamps: list[datetime],
) -> dict[str, float]:
    """
    Calculate monthly returns.
    
    Args:
        equity_curve: Array of equity values
        timestamps: List of timestamps
        
    Returns:
        Dictionary of 'YYYY-MM' to return
    """
    if len(equity_curve) != len(timestamps):
        raise ValueError("Equity curve and timestamps must have same length")
    
    monthly = {}
    current_month = None
    month_start_equity = equity_curve[0]
    
    for i, ts in enumerate(timestamps):
        month_key = ts.strftime("%Y-%m")
        
        if current_month is None:
            current_month = month_key
        elif month_key != current_month:
            # Calculate return for completed month
            monthly[current_month] = (equity_curve[i - 1] - month_start_equity) / month_start_equity
            month_start_equity = equity_curve[i - 1]
            current_month = month_key
    
    # Final month
    if current_month and len(equity_curve) > 0:
        monthly[current_month] = (equity_curve[-1] - month_start_equity) / month_start_equity
    
    return monthly


def yearly_returns(
    equity_curve: NDArray[np.float64],
    timestamps: list[datetime],
) -> dict[int, float]:
    """
    Calculate yearly returns.
    
    Args:
        equity_curve: Array of equity values
        timestamps: List of timestamps
        
    Returns:
        Dictionary of year to return
    """
    if len(equity_curve) != len(timestamps):
        raise ValueError("Equity curve and timestamps must have same length")
    
    yearly = {}
    current_year = None
    year_start_equity = equity_curve[0]
    
    for i, ts in enumerate(timestamps):
        year = ts.year
        
        if current_year is None:
            current_year = year
        elif year != current_year:
            yearly[current_year] = (equity_curve[i - 1] - year_start_equity) / year_start_equity
            year_start_equity = equity_curve[i - 1]
            current_year = year
    
    # Final year
    if current_year and len(equity_curve) > 0:
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
        periods_per_year: Periods per year for annualization
        
    Returns:
        Volatility as decimal
    """
    if len(returns) < 2:
        return 0.0
    
    vol = np.std(returns, ddof=1)
    
    if annualize:
        vol *= np.sqrt(periods_per_year)
    
    return float(vol)


def downside_volatility(
    returns: NDArray[np.float64],
    threshold: float = 0.0,
    annualize: bool = True,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate downside volatility (semi-deviation).
    
    Only considers returns below threshold.
    
    Args:
        returns: Array of returns
        threshold: Minimum acceptable return (MAR)
        annualize: Whether to annualize
        periods_per_year: Periods per year
        
    Returns:
        Downside volatility as decimal
    """
    if len(returns) < 2:
        return 0.0
    
    downside_returns = returns[returns < threshold]
    
    if len(downside_returns) < 2:
        return 0.0
    
    downside_vol = np.std(downside_returns, ddof=1)
    
    if annualize:
        downside_vol *= np.sqrt(periods_per_year)
    
    return float(downside_vol)


def upside_volatility(
    returns: NDArray[np.float64],
    threshold: float = 0.0,
    annualize: bool = True,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate upside volatility.
    
    Args:
        returns: Array of returns
        threshold: Minimum acceptable return
        annualize: Whether to annualize
        periods_per_year: Periods per year
        
    Returns:
        Upside volatility as decimal
    """
    if len(returns) < 2:
        return 0.0
    
    upside_returns = returns[returns > threshold]
    
    if len(upside_returns) < 2:
        return 0.0
    
    upside_vol = np.std(upside_returns, ddof=1)
    
    if annualize:
        upside_vol *= np.sqrt(periods_per_year)
    
    return float(upside_vol)


# =============================================================================
# DRAWDOWN METRICS
# =============================================================================

def calculate_drawdown_series(
    equity_curve: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate drawdown series.
    
    Args:
        equity_curve: Array of equity values
        
    Returns:
        Tuple of (drawdown series, running max series)
    """
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    return drawdown, running_max


def max_drawdown(equity_curve: NDArray[np.float64]) -> float:
    """
    Calculate maximum drawdown.
    
    Formula:
        MDD = min((Equity - Peak) / Peak)
    
    Args:
        equity_curve: Array of equity values
        
    Returns:
        Maximum drawdown as negative decimal
    """
    if len(equity_curve) < 2:
        return 0.0
    
    drawdown, _ = calculate_drawdown_series(equity_curve)
    return float(np.min(drawdown))


def max_drawdown_duration(
    equity_curve: NDArray[np.float64],
    timestamps: list[datetime] | None = None,
) -> int:
    """
    Calculate maximum drawdown duration in periods.
    
    Args:
        equity_curve: Array of equity values
        timestamps: Optional timestamps for duration in days
        
    Returns:
        Duration in periods (or days if timestamps provided)
    """
    if len(equity_curve) < 2:
        return 0
    
    drawdown, running_max = calculate_drawdown_series(equity_curve)
    
    max_duration = 0
    current_duration = 0
    
    for i in range(len(equity_curve)):
        if equity_curve[i] < running_max[i]:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    
    return max_duration


def average_drawdown(equity_curve: NDArray[np.float64]) -> float:
    """
    Calculate average drawdown.
    
    Args:
        equity_curve: Array of equity values
        
    Returns:
        Average drawdown as negative decimal
    """
    if len(equity_curve) < 2:
        return 0.0
    
    drawdown, _ = calculate_drawdown_series(equity_curve)
    negative_dd = drawdown[drawdown < 0]
    
    if len(negative_dd) == 0:
        return 0.0
    
    return float(np.mean(negative_dd))


def drawdown_details(
    equity_curve: NDArray[np.float64],
    timestamps: list[datetime] | None = None,
) -> list[dict[str, Any]]:
    """
    Get detailed information about each drawdown period.
    
    Args:
        equity_curve: Array of equity values
        timestamps: Optional timestamps
        
    Returns:
        List of drawdown detail dictionaries
    """
    if len(equity_curve) < 2:
        return []
    
    drawdown, running_max = calculate_drawdown_series(equity_curve)
    
    drawdowns = []
    in_drawdown = False
    start_idx = 0
    peak_value = equity_curve[0]
    
    for i in range(len(equity_curve)):
        if not in_drawdown and drawdown[i] < 0:
            # Start of drawdown
            in_drawdown = True
            start_idx = i - 1 if i > 0 else 0
            peak_value = running_max[i]
        elif in_drawdown and (drawdown[i] >= 0 or i == len(equity_curve) - 1):
            # End of drawdown (or end of series)
            end_idx = i
            trough_idx = start_idx + np.argmin(equity_curve[start_idx:end_idx + 1])
            
            dd_info = {
                "start_idx": start_idx,
                "trough_idx": int(trough_idx),
                "end_idx": end_idx,
                "peak_value": float(peak_value),
                "trough_value": float(equity_curve[trough_idx]),
                "drawdown": float((equity_curve[trough_idx] - peak_value) / peak_value),
                "duration": end_idx - start_idx,
                "recovery_duration": end_idx - trough_idx,
            }
            
            if timestamps:
                dd_info["start_date"] = timestamps[start_idx]
                dd_info["trough_date"] = timestamps[trough_idx]
                dd_info["end_date"] = timestamps[end_idx]
            
            drawdowns.append(dd_info)
            in_drawdown = False
    
    return sorted(drawdowns, key=lambda x: x["drawdown"])


def ulcer_index(equity_curve: NDArray[np.float64]) -> float:
    """
    Calculate Ulcer Index (measure of downside risk).
    
    Formula:
        UI = sqrt(mean(drawdown^2))
    
    Args:
        equity_curve: Array of equity values
        
    Returns:
        Ulcer index value
    """
    if len(equity_curve) < 2:
        return 0.0
    
    drawdown, _ = calculate_drawdown_series(equity_curve)
    return float(np.sqrt(np.mean(drawdown ** 2)))


def pain_index(equity_curve: NDArray[np.float64]) -> float:
    """
    Calculate Pain Index (average drawdown).
    
    Args:
        equity_curve: Array of equity values
        
    Returns:
        Pain index (positive value)
    """
    if len(equity_curve) < 2:
        return 0.0
    
    drawdown, _ = calculate_drawdown_series(equity_curve)
    return float(-np.mean(drawdown))


# =============================================================================
# VALUE AT RISK (VaR) METRICS
# =============================================================================

def var_historical(
    returns: NDArray[np.float64],
    confidence: float = 0.95,
) -> float:
    """
    Calculate Value at Risk using historical simulation.
    
    Args:
        returns: Array of returns
        confidence: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        VaR as positive decimal (loss)
    """
    if len(returns) < 10:
        return 0.0
    
    return float(-np.percentile(returns, (1 - confidence) * 100))


def var_parametric(
    returns: NDArray[np.float64],
    confidence: float = 0.95,
) -> float:
    """
    Calculate Value at Risk using parametric (normal) method.
    
    Formula:
        VaR = -(mean + z * std)
    
    Args:
        returns: Array of returns
        confidence: Confidence level
        
    Returns:
        VaR as positive decimal
    """
    if len(returns) < 2:
        return 0.0
    
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    z_score = scipy_stats.norm.ppf(1 - confidence)
    
    return float(-(mean + z_score * std))


def var_cornish_fisher(
    returns: NDArray[np.float64],
    confidence: float = 0.95,
) -> float:
    """
    Calculate VaR using Cornish-Fisher expansion.
    
    Adjusts for skewness and kurtosis.
    
    Args:
        returns: Array of returns
        confidence: Confidence level
        
    Returns:
        VaR as positive decimal
    """
    if len(returns) < 10:
        return 0.0
    
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    skew = scipy_stats.skew(returns)
    kurt = scipy_stats.kurtosis(returns)
    
    z = scipy_stats.norm.ppf(1 - confidence)
    
    # Cornish-Fisher adjustment
    z_cf = (
        z + 
        (z ** 2 - 1) * skew / 6 +
        (z ** 3 - 3 * z) * kurt / 24 -
        (2 * z ** 3 - 5 * z) * skew ** 2 / 36
    )
    
    return float(-(mean + z_cf * std))


def cvar(
    returns: NDArray[np.float64],
    confidence: float = 0.95,
) -> float:
    """
    Calculate Conditional Value at Risk (Expected Shortfall).
    
    Average of losses beyond VaR.
    
    Args:
        returns: Array of returns
        confidence: Confidence level
        
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
        Sharpe = (Return - Risk-Free) / Volatility
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Periods per year
        
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Convert annual risk-free to per-period
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    excess_returns = returns - rf_per_period
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns, ddof=1)
    
    # Annualize
    return float(sharpe * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: NDArray[np.float64],
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate Sortino Ratio.
    
    Like Sharpe but uses downside volatility.
    
    Formula:
        Sortino = (Return - MAR) / Downside Volatility
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate (MAR)
        periods_per_year: Periods per year
        
    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns - rf_per_period
    
    downside_vol = downside_volatility(returns, rf_per_period, annualize=False)
    
    if downside_vol == 0:
        return 0.0 if np.mean(excess_returns) <= 0 else float('inf')
    
    sortino = np.mean(excess_returns) / downside_vol
    
    return float(sortino * np.sqrt(periods_per_year))


def calmar_ratio(
    equity_curve: NDArray[np.float64],
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate Calmar Ratio.
    
    Formula:
        Calmar = Annual Return / |Max Drawdown|
    
    Args:
        equity_curve: Array of equity values
        periods_per_year: Periods per year
        
    Returns:
        Calmar ratio
    """
    if len(equity_curve) < 2:
        return 0.0
    
    ann_return = annualized_return(equity_curve, periods_per_year)
    mdd = abs(max_drawdown(equity_curve))
    
    if mdd == 0:
        return 0.0 if ann_return <= 0 else float('inf')
    
    return float(ann_return / mdd)


def omega_ratio(
    returns: NDArray[np.float64],
    threshold: float = 0.0,
) -> float:
    """
    Calculate Omega Ratio.
    
    Formula:
        Omega = Sum(Excess Gains) / Sum(Excess Losses)
    
    Args:
        returns: Array of returns
        threshold: Minimum acceptable return
        
    Returns:
        Omega ratio
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
    """
    Calculate Information Ratio.
    
    Formula:
        IR = Mean(Active Return) / Std(Active Return)
    
    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        periods_per_year: Periods per year
        
    Returns:
        Annualized information ratio
    """
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
    """
    Calculate Treynor Ratio.
    
    Formula:
        Treynor = (Return - Risk-Free) / Beta
    
    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Periods per year
        
    Returns:
        Treynor ratio
    """
    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        return 0.0
    
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Calculate beta
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
    """
    Calculate Gain to Pain Ratio.
    
    Formula:
        GtP = Total Return / Pain Index
    
    Args:
        equity_curve: Array of equity values
        
    Returns:
        Gain to pain ratio
    """
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
    """
    Calculate Ulcer Performance Index (Martin Ratio).
    
    Formula:
        UPI = (Return - Risk-Free) / Ulcer Index
    
    Args:
        equity_curve: Array of equity values
        risk_free_rate: Annual risk-free rate
        periods_per_year: Periods per year
        
    Returns:
        UPI value
    """
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
    breakeven_trades: int = 0
    
    win_rate: float = 0.0
    loss_rate: float = 0.0
    
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    
    profit_factor: float = 0.0
    
    avg_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    avg_holding_period: float = 0.0  # in hours
    avg_bars_in_trade: float = 0.0
    
    expectancy: float = 0.0
    expectancy_ratio: float = 0.0
    
    # Per-symbol statistics
    trades_per_symbol: dict[str, int] = field(default_factory=dict)
    pnl_per_symbol: dict[str, float] = field(default_factory=dict)
    
    # Time-based statistics
    trades_per_day: dict[str, int] = field(default_factory=dict)
    trades_per_hour: dict[int, int] = field(default_factory=dict)


def calculate_trade_stats(
    trades: list[dict[str, Any]],
) -> TradeStats:
    """
    Calculate comprehensive trade statistics.
    
    Args:
        trades: List of trade dictionaries with keys:
            - pnl: Profit/loss
            - pnl_pct: Profit/loss percentage
            - entry_time: Entry timestamp
            - exit_time: Exit timestamp
            - symbol: Trading symbol
            - side: 'long' or 'short'
    
    Returns:
        TradeStats object
    """
    stats = TradeStats()
    
    if not trades:
        return stats
    
    stats.total_trades = len(trades)
    
    # Extract PnL values
    pnls = np.array([t.get("pnl", 0) for t in trades])
    
    # Win/Loss counts
    stats.winning_trades = int(np.sum(pnls > 0))
    stats.losing_trades = int(np.sum(pnls < 0))
    stats.breakeven_trades = int(np.sum(pnls == 0))
    
    # Win/Loss rates
    stats.win_rate = stats.winning_trades / stats.total_trades if stats.total_trades > 0 else 0
    stats.loss_rate = stats.losing_trades / stats.total_trades if stats.total_trades > 0 else 0
    
    # Profit/Loss totals
    stats.gross_profit = float(np.sum(pnls[pnls > 0]))
    stats.gross_loss = float(np.sum(pnls[pnls < 0]))
    stats.net_profit = float(np.sum(pnls))
    
    # Profit factor
    if stats.gross_loss != 0:
        stats.profit_factor = abs(stats.gross_profit / stats.gross_loss)
    else:
        stats.profit_factor = float('inf') if stats.gross_profit > 0 else 0.0
    
    # Averages
    stats.avg_trade = float(np.mean(pnls))
    
    winning_pnls = pnls[pnls > 0]
    losing_pnls = pnls[pnls < 0]
    
    stats.avg_win = float(np.mean(winning_pnls)) if len(winning_pnls) > 0 else 0.0
    stats.avg_loss = float(np.mean(losing_pnls)) if len(losing_pnls) > 0 else 0.0
    
    # Largest trades
    stats.largest_win = float(np.max(pnls)) if len(pnls) > 0 else 0.0
    stats.largest_loss = float(np.min(pnls)) if len(pnls) > 0 else 0.0
    
    # Consecutive wins/losses
    stats.max_consecutive_wins = _max_consecutive(pnls > 0)
    stats.max_consecutive_losses = _max_consecutive(pnls < 0)
    
    # Holding period
    holding_periods = []
    for t in trades:
        entry = t.get("entry_time")
        exit_time = t.get("exit_time")
        if entry and exit_time:
            duration = (exit_time - entry).total_seconds() / 3600  # hours
            holding_periods.append(duration)
    
    stats.avg_holding_period = float(np.mean(holding_periods)) if holding_periods else 0.0
    
    # Bars in trade
    bars = [t.get("bars_held", 0) for t in trades]
    stats.avg_bars_in_trade = float(np.mean(bars)) if bars else 0.0
    
    # Expectancy
    stats.expectancy = (stats.win_rate * stats.avg_win) + ((1 - stats.win_rate) * stats.avg_loss)
    
    if stats.avg_loss != 0:
        stats.expectancy_ratio = abs(stats.avg_win / stats.avg_loss)
    else:
        stats.expectancy_ratio = float('inf') if stats.avg_win > 0 else 0.0
    
    # Per-symbol statistics
    for t in trades:
        symbol = t.get("symbol", "UNKNOWN")
        pnl = t.get("pnl", 0)
        
        stats.trades_per_symbol[symbol] = stats.trades_per_symbol.get(symbol, 0) + 1
        stats.pnl_per_symbol[symbol] = stats.pnl_per_symbol.get(symbol, 0) + pnl
    
    # Time-based statistics
    for t in trades:
        entry = t.get("entry_time")
        if entry:
            day_key = entry.strftime("%A")
            hour = entry.hour
            
            stats.trades_per_day[day_key] = stats.trades_per_day.get(day_key, 0) + 1
            stats.trades_per_hour[hour] = stats.trades_per_hour.get(hour, 0) + 1
    
    return stats


def _max_consecutive(mask: NDArray[np.bool_]) -> int:
    """Calculate maximum consecutive True values."""
    if len(mask) == 0:
        return 0
    
    max_count = 0
    current_count = 0
    
    for val in mask:
        if val:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0
    
    return max_count


# =============================================================================
# DISTRIBUTION METRICS
# =============================================================================

def skewness(returns: NDArray[np.float64]) -> float:
    """
    Calculate return distribution skewness.
    
    Positive: Right tail longer (good)
    Negative: Left tail longer (bad)
    
    Args:
        returns: Array of returns
        
    Returns:
        Skewness value
    """
    if len(returns) < 3:
        return 0.0
    
    return float(scipy_stats.skew(returns))


def kurtosis(returns: NDArray[np.float64]) -> float:
    """
    Calculate return distribution kurtosis.
    
    >3: Fat tails (more extreme events)
    <3: Thin tails
    
    Args:
        returns: Array of returns
        
    Returns:
        Kurtosis value (excess kurtosis, normal = 0)
    """
    if len(returns) < 4:
        return 0.0
    
    return float(scipy_stats.kurtosis(returns))


def tail_ratio(returns: NDArray[np.float64]) -> float:
    """
    Calculate tail ratio.
    
    Formula:
        Tail Ratio = |95th percentile| / |5th percentile|
    
    Args:
        returns: Array of returns
        
    Returns:
        Tail ratio (>1 means positive skew)
    """
    if len(returns) < 20:
        return 1.0
    
    p95 = np.percentile(returns, 95)
    p5 = np.percentile(returns, 5)
    
    if p5 == 0:
        return float('inf') if p95 > 0 else 0.0
    
    return float(abs(p95 / p5))


def common_sense_ratio(returns: NDArray[np.float64]) -> float:
    """
    Calculate Common Sense Ratio.
    
    Formula:
        CSR = Tail Ratio * Gain-to-Pain Ratio
    
    Args:
        returns: Array of returns
        
    Returns:
        Common sense ratio
    """
    tr = tail_ratio(returns)
    
    gains = np.sum(returns[returns > 0])
    losses = -np.sum(returns[returns < 0])
    
    if losses == 0:
        gtp = float('inf') if gains > 0 else 0.0
    else:
        gtp = gains / losses
    
    if tr == float('inf') or gtp == float('inf'):
        return float('inf')
    
    return float(tr * gtp)


def outlier_ratio(
    returns: NDArray[np.float64],
    threshold: float = 2.0,
) -> float:
    """
    Calculate ratio of outlier returns.
    
    Args:
        returns: Array of returns
        threshold: Number of standard deviations
        
    Returns:
        Fraction of returns beyond threshold
    """
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
    """
    Comprehensive performance report.
    
    Contains all metrics for strategy evaluation.
    """
    
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
    
    # Additional metrics
    best_month: float = 0.0
    worst_month: float = 0.0
    positive_months: int = 0
    negative_months: int = 0
    
    # Benchmark comparison (if available)
    benchmark_return: float | None = None
    alpha: float | None = None
    beta: float | None = None
    
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
        }


class MetricsCalculator:
    """
    Calculator for comprehensive performance metrics.
    
    Example:
        calc = MetricsCalculator(
            equity_curve=equity,
            timestamps=dates,
            trades=trade_list,
            initial_capital=100000,
        )
        report = calc.calculate_all()
    """
    
    def __init__(
        self,
        equity_curve: NDArray[np.float64],
        timestamps: list[datetime] | None = None,
        trades: list[dict[str, Any]] | None = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = RISK_FREE_RATE,
        periods_per_year: float = TRADING_DAYS_PER_YEAR,
        benchmark_returns: NDArray[np.float64] | None = None,
        strategy_name: str = "",
    ):
        """
        Initialize calculator.
        
        Args:
            equity_curve: Array of equity values
            timestamps: List of timestamps
            trades: List of trade dictionaries
            initial_capital: Starting capital
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
            benchmark_returns: Optional benchmark returns
            strategy_name: Strategy identifier
        """
        self.equity_curve = np.asarray(equity_curve)
        self.timestamps = timestamps or []
        self.trades = trades or []
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.benchmark_returns = benchmark_returns
        self.strategy_name = strategy_name
        
        # Calculate returns
        self.returns = calculate_returns(self.equity_curve)
    
    def calculate_all(self) -> PerformanceReport:
        """
        Calculate all performance metrics.
        
        Returns:
            PerformanceReport with all metrics
        """
        report = PerformanceReport()
        report.strategy_name = self.strategy_name
        report.initial_capital = self.initial_capital
        report.final_capital = float(self.equity_curve[-1]) if len(self.equity_curve) > 0 else 0.0
        
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
        
        # Risk-adjusted metrics
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
    
    def calculate_returns_metrics(self) -> dict[str, float]:
        """Calculate only return metrics."""
        return {
            "total_return": total_return(self.equity_curve),
            "annualized_return": annualized_return(self.equity_curve, self.periods_per_year),
        }
    
    def calculate_risk_metrics(self) -> dict[str, float]:
        """Calculate only risk metrics."""
        return {
            "volatility": volatility(self.returns, True, self.periods_per_year),
            "downside_volatility": downside_volatility(self.returns, 0, True, self.periods_per_year),
            "max_drawdown": max_drawdown(self.equity_curve),
            "var_95": var_historical(self.returns, 0.95),
            "cvar_95": cvar(self.returns, 0.95),
        }
    
    def calculate_risk_adjusted_metrics(self) -> dict[str, float]:
        """Calculate only risk-adjusted metrics."""
        return {
            "sharpe_ratio": sharpe_ratio(self.returns, self.risk_free_rate, self.periods_per_year),
            "sortino_ratio": sortino_ratio(self.returns, self.risk_free_rate, self.periods_per_year),
            "calmar_ratio": calmar_ratio(self.equity_curve, self.periods_per_year),
            "omega_ratio": omega_ratio(self.returns, 0),
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "TRADING_DAYS_PER_YEAR",
    "RISK_FREE_RATE",
    "PERIODS_PER_YEAR",
    
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