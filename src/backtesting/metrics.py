"""
Performance metrics for AlphaTrade system.

This module provides comprehensive performance calculations:
- Return metrics
- Risk-adjusted metrics
- Risk metrics
- Trading metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""

    # Return metrics
    total_return: float
    cagr: float
    avg_daily_return: float
    best_day: float
    worst_day: float
    best_month: float
    worst_month: float

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float | None
    treynor_ratio: float | None

    # Risk metrics
    volatility: float
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int
    var_95: float
    cvar_95: float
    beta: float | None
    alpha: float | None

    # Trading metrics
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    payoff_ratio: float
    expectancy: float

    # Efficiency metrics
    turnover: float
    trading_days: int
    avg_holding_period: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "volatility": self.volatility,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
        }


def calculate_returns_metrics(
    returns: pd.Series,
    periods_per_year: int = 252 * 26,
) -> dict[str, float]:
    """
    Calculate return-related metrics.

    Args:
        returns: Return series
        periods_per_year: Number of periods per year

    Returns:
        Dictionary of return metrics
    """
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year

    # CAGR
    if years > 0 and total_return > -1:
        cagr = (1 + total_return) ** (1 / years) - 1
    else:
        cagr = 0.0

    # Monthly returns (approximate)
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "avg_daily_return": returns.mean() * 26,  # Per trading day
        "best_day": returns.max() * 26,
        "worst_day": returns.min() * 26,
        "best_month": monthly.max() if len(monthly) > 0 else 0,
        "worst_month": monthly.min() if len(monthly) > 0 else 0,
    }


def calculate_risk_adjusted_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252 * 26,
) -> dict[str, float]:
    """
    Calculate risk-adjusted performance metrics.

    Args:
        returns: Return series
        benchmark_returns: Benchmark return series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Dictionary of risk-adjusted metrics
    """
    rf_per_period = risk_free_rate / periods_per_year

    # Volatility
    volatility = returns.std() * np.sqrt(periods_per_year)

    # Sharpe Ratio
    excess_returns = returns - rf_per_period
    if returns.std() > 0:
        sharpe = excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(periods_per_year)
    else:
        sortino = sharpe

    # Max Drawdown for Calmar
    equity = (1 + returns).cumprod()
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    max_dd = abs(drawdown.min())

    # Calmar Ratio
    cagr = calculate_returns_metrics(returns, periods_per_year)["cagr"]
    calmar = cagr / max_dd if max_dd > 0 else 0.0

    # Information Ratio and Beta/Alpha (if benchmark provided)
    information_ratio = None
    treynor_ratio = None
    beta = None
    alpha = None

    if benchmark_returns is not None:
        # Align
        common_idx = returns.index.intersection(benchmark_returns.index)
        ret = returns.loc[common_idx]
        bench = benchmark_returns.loc[common_idx]

        # Tracking error
        active_returns = ret - bench
        tracking_error = active_returns.std() * np.sqrt(periods_per_year)

        if tracking_error > 0:
            information_ratio = active_returns.mean() * periods_per_year / tracking_error

        # Beta
        cov = np.cov(ret, bench)[0, 1]
        var = bench.var()
        if var > 0:
            beta = cov / var
            alpha = (ret.mean() - rf_per_period - beta * (bench.mean() - rf_per_period)) * periods_per_year

            # Treynor Ratio
            if beta != 0:
                treynor_ratio = (ret.mean() - rf_per_period) * periods_per_year / beta

    return {
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "information_ratio": information_ratio,
        "treynor_ratio": treynor_ratio,
        "volatility": volatility,
        "beta": beta,
        "alpha": alpha,
    }


def calculate_risk_metrics(
    returns: pd.Series,
    equity: pd.Series,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """
    Calculate risk metrics.

    Args:
        returns: Return series
        equity: Equity curve
        confidence_level: VaR confidence level

    Returns:
        Dictionary of risk metrics
    """
    # Drawdown analysis
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max

    max_dd = abs(drawdown.min())

    # Average drawdown
    avg_dd = abs(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0

    # Max drawdown duration
    underwater = drawdown < 0
    if underwater.any():
        # Find consecutive underwater periods
        groups = (~underwater).cumsum()
        underwater_lengths = underwater.groupby(groups).sum()
        max_dd_duration = int(underwater_lengths.max())
    else:
        max_dd_duration = 0

    # VaR and CVaR
    quantile = 1 - confidence_level
    var_95 = abs(returns.quantile(quantile))

    tail_returns = returns[returns <= returns.quantile(quantile)]
    cvar_95 = abs(tail_returns.mean()) if len(tail_returns) > 0 else var_95

    return {
        "max_drawdown": max_dd,
        "avg_drawdown": avg_dd,
        "max_drawdown_duration": max_dd_duration,
        "var_95": var_95,
        "cvar_95": cvar_95,
    }


def calculate_trading_metrics(
    returns: pd.Series,
    positions: pd.DataFrame | None = None,
) -> dict[str, float]:
    """
    Calculate trading metrics.

    Args:
        returns: Return series
        positions: Position DataFrame

    Returns:
        Dictionary of trading metrics
    """
    # Win/Loss analysis
    winning = returns[returns > 0]
    losing = returns[returns < 0]

    total_trades = len(returns[returns != 0])
    win_rate = len(winning) / total_trades if total_trades > 0 else 0

    avg_win = winning.mean() if len(winning) > 0 else 0
    avg_loss = abs(losing.mean()) if len(losing) > 0 else 0

    # Profit factor
    gross_profit = winning.sum() if len(winning) > 0 else 0
    gross_loss = abs(losing.sum()) if len(losing) > 0 else 1e-10
    profit_factor = gross_profit / gross_loss

    # Payoff ratio
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    # Expectancy
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # Consecutive wins/losses
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_wins = 0
    current_losses = 0

    for ret in returns:
        if ret > 0:
            current_wins += 1
            current_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, current_wins)
        elif ret < 0:
            current_losses += 1
            current_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
        else:
            current_wins = 0
            current_losses = 0

    # Turnover
    turnover = 0.0
    avg_holding = 0.0
    if positions is not None:
        weight_changes = positions.diff().abs()
        turnover = weight_changes.sum(axis=1).mean() / 2 * 252 * 26  # Annualized

        # Average holding period (approximate)
        holding_periods = []
        for col in positions.columns:
            pos = positions[col]
            changes = pos.diff().abs()
            trade_dates = changes[changes > 0].index
            if len(trade_dates) > 1:
                diffs = pd.Series(trade_dates).diff().dropna()
                if len(diffs) > 0:
                    holding_periods.extend(diffs.dt.total_seconds() / 3600)

        avg_holding = np.mean(holding_periods) if holding_periods else 0

    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        "payoff_ratio": payoff_ratio,
        "expectancy": expectancy,
        "turnover": turnover,
        "avg_holding_period": avg_holding,
        "trading_days": len(returns),
    }


def calculate_all_metrics(
    returns: pd.Series,
    equity: pd.Series,
    benchmark_returns: pd.Series | None = None,
    positions: pd.DataFrame | None = None,
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252 * 26,
) -> dict[str, float]:
    """
    Calculate all performance metrics.

    Args:
        returns: Return series
        equity: Equity curve
        benchmark_returns: Optional benchmark returns
        positions: Optional position DataFrame
        risk_free_rate: Annual risk-free rate
        periods_per_year: Periods per year

    Returns:
        Dictionary of all metrics
    """
    metrics = {}

    # Return metrics
    metrics.update(calculate_returns_metrics(returns, periods_per_year))

    # Risk-adjusted metrics
    metrics.update(
        calculate_risk_adjusted_metrics(
            returns, benchmark_returns, risk_free_rate, periods_per_year
        )
    )

    # Risk metrics
    metrics.update(calculate_risk_metrics(returns, equity))

    # Trading metrics
    metrics.update(calculate_trading_metrics(returns, positions))

    return metrics


def compare_strategies(
    results: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """
    Compare metrics across multiple strategies.

    Args:
        results: Dictionary mapping strategy names to metric dicts

    Returns:
        Comparison DataFrame
    """
    return pd.DataFrame(results).T


def statistical_significance(
    returns1: pd.Series,
    returns2: pd.Series,
) -> dict[str, float]:
    """
    Test statistical significance between two return series.

    Args:
        returns1: First return series
        returns2: Second return series

    Returns:
        Dictionary with test results
    """
    # T-test
    t_stat, t_pvalue = stats.ttest_ind(returns1, returns2)

    # Mann-Whitney U test
    u_stat, u_pvalue = stats.mannwhitneyu(returns1, returns2)

    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = stats.ks_2samp(returns1, returns2)

    return {
        "t_statistic": t_stat,
        "t_pvalue": t_pvalue,
        "u_statistic": u_stat,
        "u_pvalue": u_pvalue,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pvalue,
        "mean_diff": returns1.mean() - returns2.mean(),
        "sharpe_diff": (
            returns1.mean() / returns1.std() - returns2.mean() / returns2.std()
        ) * np.sqrt(252 * 26),
    }
