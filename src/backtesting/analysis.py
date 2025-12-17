"""
Backtest analysis for AlphaTrade system.

This module provides:
- Trade analysis
- Position analysis
- Performance attribution
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.backtesting.engine import BacktestResult, Trade


class BacktestAnalyzer:
    """
    Comprehensive backtest analyzer.

    Provides detailed analysis of:
    - Trade performance
    - Position characteristics
    - Return attribution
    - Strategy behavior
    """

    def __init__(self, result: BacktestResult) -> None:
        """
        Initialize analyzer with backtest result.

        Args:
            result: BacktestResult object
        """
        self.result = result

    def trade_analysis(self) -> pd.DataFrame:
        """
        Analyze individual trades.

        Returns:
            DataFrame with trade analysis
        """
        if not self.result.trades:
            return pd.DataFrame()

        trades_data = []
        for trade in self.result.trades:
            trades_data.append({
                "timestamp": trade.timestamp,
                "symbol": trade.symbol,
                "side": trade.side,
                "quantity": trade.quantity,
                "price": trade.price,
                "value": trade.value,
                "commission": trade.commission,
                "slippage": trade.slippage,
                "total_cost": trade.total_cost,
            })

        df = pd.DataFrame(trades_data)

        # Add derived metrics
        df["cost_pct"] = df["total_cost"] / df["value"] * 100

        return df

    def position_analysis(self) -> dict[str, Any]:
        """
        Analyze position characteristics.

        Returns:
            Dictionary with position analysis
        """
        positions = self.result.positions

        # Average position count
        non_zero = (positions != 0).sum(axis=1)
        avg_positions = non_zero.mean()
        max_positions = non_zero.max()

        # Position concentration
        abs_weights = positions.abs()
        total_weight = abs_weights.sum(axis=1)
        concentration = (abs_weights.max(axis=1) / total_weight.replace(0, 1)).mean()

        # Long/Short breakdown
        long_exposure = positions.clip(lower=0).sum(axis=1).mean()
        short_exposure = positions.clip(upper=0).sum(axis=1).mean()

        return {
            "avg_positions": avg_positions,
            "max_positions": max_positions,
            "avg_concentration": concentration,
            "avg_long_exposure": long_exposure,
            "avg_short_exposure": abs(short_exposure),
            "avg_net_exposure": long_exposure + short_exposure,
            "avg_gross_exposure": long_exposure + abs(short_exposure),
        }

    def monthly_returns(self) -> pd.DataFrame:
        """
        Calculate monthly returns.

        Returns:
            DataFrame with monthly returns
        """
        returns = self.result.returns

        # Resample to monthly
        monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)

        # Pivot for heatmap
        monthly_df = monthly.to_frame("return")
        monthly_df["year"] = monthly_df.index.year
        monthly_df["month"] = monthly_df.index.month

        pivot = monthly_df.pivot(index="year", columns="month", values="return")
        pivot.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ][:len(pivot.columns)]

        return pivot

    def drawdown_analysis(self) -> pd.DataFrame:
        """
        Analyze drawdown events.

        Returns:
            DataFrame with drawdown events
        """
        from src.risk.drawdown import find_drawdown_events

        events = find_drawdown_events(self.result.equity_curve, min_drawdown=0.02)

        if not events:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "start_date": e.start_date,
                "trough_date": e.trough_date,
                "end_date": e.end_date,
                "drawdown": e.drawdown_pct,
                "duration_days": e.duration_days,
                "recovery_days": e.recovery_days,
                "is_active": e.is_active,
            }
            for e in events
        ])

    def rolling_metrics(
        self,
        window: int = 252 * 26,
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.

        Args:
            window: Rolling window size

        Returns:
            DataFrame with rolling metrics
        """
        returns = self.result.returns

        rolling = pd.DataFrame(index=returns.index)

        # Rolling returns
        rolling["return_ann"] = returns.rolling(window).mean() * 252 * 26

        # Rolling volatility
        rolling["volatility_ann"] = returns.rolling(window).std() * np.sqrt(252 * 26)

        # Rolling Sharpe
        rolling["sharpe"] = rolling["return_ann"] / rolling["volatility_ann"]

        # Rolling max drawdown
        equity = self.result.equity_curve
        rolling["max_dd"] = equity.rolling(window).apply(
            lambda x: (x / x.expanding().max() - 1).min(),
            raw=False,
        ).abs()

        return rolling.dropna()

    def symbol_attribution(self) -> pd.DataFrame:
        """
        Calculate return attribution by symbol.

        Returns:
            DataFrame with symbol contributions
        """
        positions = self.result.positions

        # We need price data to calculate contributions
        # This is a simplified version
        attribution = []

        for symbol in positions.columns:
            symbol_pos = positions[symbol]

            # Count of periods with position
            active_periods = (symbol_pos != 0).sum()
            avg_weight = symbol_pos[symbol_pos != 0].mean() if active_periods > 0 else 0

            attribution.append({
                "symbol": symbol,
                "active_periods": active_periods,
                "avg_weight": avg_weight,
                "max_weight": symbol_pos.max(),
                "min_weight": symbol_pos.min(),
            })

        return pd.DataFrame(attribution).set_index("symbol")

    def get_summary(self) -> dict[str, Any]:
        """
        Get comprehensive summary.

        Returns:
            Summary dictionary
        """
        return {
            "metrics": self.result.metrics,
            "position_analysis": self.position_analysis(),
            "num_trades": len(self.result.trades),
            "total_return": self.result.metrics.get("total_return", 0),
            "sharpe_ratio": self.result.metrics.get("sharpe_ratio", 0),
            "max_drawdown": self.result.metrics.get("max_drawdown", 0),
            "win_rate": self.result.metrics.get("win_rate", 0),
        }


def analyze_trades(trades: list[Trade]) -> pd.DataFrame:
    """
    Convenience function to analyze trades.

    Args:
        trades: List of Trade objects

    Returns:
        Trade analysis DataFrame
    """
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame([
        {
            "timestamp": t.timestamp,
            "symbol": t.symbol,
            "side": t.side,
            "quantity": t.quantity,
            "price": t.price,
            "value": t.value,
            "total_cost": t.total_cost,
        }
        for t in trades
    ])

    # Summary stats
    summary = {
        "total_trades": len(trades),
        "total_value": df["value"].sum(),
        "total_costs": df["total_cost"].sum(),
        "avg_trade_size": df["value"].mean(),
        "buys": len(df[df["side"] == "BUY"]),
        "sells": len(df[df["side"] == "SELL"]),
    }

    return df, summary


def analyze_positions(positions: pd.DataFrame) -> dict[str, Any]:
    """
    Convenience function to analyze positions.

    Args:
        positions: Position DataFrame

    Returns:
        Position analysis dictionary
    """
    # Non-zero position count
    active = (positions != 0).sum(axis=1)

    # Concentration (HHI)
    weights = positions.abs()
    total = weights.sum(axis=1).replace(0, 1)
    normalized = weights.div(total, axis=0)
    hhi = (normalized ** 2).sum(axis=1)

    return {
        "avg_num_positions": active.mean(),
        "max_num_positions": active.max(),
        "avg_hhi": hhi.mean(),  # Lower is more diversified
        "effective_positions": (1 / hhi).mean(),  # Effective number of positions
        "turnover": positions.diff().abs().sum(axis=1).mean() / 2,
    }


class WalkForwardAnalyzer:
    """
    Walk-forward analysis results analyzer.
    """

    def __init__(
        self,
        results: list[BacktestResult],
        split_info: list | None = None,
    ) -> None:
        """
        Initialize with walk-forward results.

        Args:
            results: List of BacktestResult for each fold
            split_info: Optional split information
        """
        self.results = results
        self.split_info = split_info

    def aggregate_metrics(self) -> pd.DataFrame:
        """
        Aggregate metrics across folds.

        Returns:
            DataFrame with aggregated metrics
        """
        metrics_list = [r.metrics for r in self.results]
        df = pd.DataFrame(metrics_list)

        summary = pd.DataFrame({
            "mean": df.mean(),
            "std": df.std(),
            "min": df.min(),
            "max": df.max(),
            "median": df.median(),
        })

        return summary

    def stability_analysis(self) -> dict[str, float]:
        """
        Analyze strategy stability across folds.

        Returns:
            Stability metrics
        """
        sharpes = [r.metrics.get("sharpe_ratio", 0) for r in self.results]
        returns = [r.metrics.get("total_return", 0) for r in self.results]

        return {
            "sharpe_mean": np.mean(sharpes),
            "sharpe_std": np.std(sharpes),
            "sharpe_stability": np.mean(sharpes) / (np.std(sharpes) + 1e-10),
            "return_mean": np.mean(returns),
            "return_std": np.std(returns),
            "positive_folds": sum(1 for r in returns if r > 0) / len(returns),
            "profitable_sharpe_folds": sum(1 for s in sharpes if s > 0) / len(sharpes),
        }

    def combine_equity_curves(self) -> pd.Series:
        """
        Combine equity curves from all folds.

        Returns:
            Combined equity curve
        """
        curves = []
        for result in self.results:
            # Normalize to start at 1
            normalized = result.equity_curve / result.equity_curve.iloc[0]
            curves.append(normalized)

        # Concatenate (assuming non-overlapping periods)
        combined = pd.concat(curves)
        combined = combined[~combined.index.duplicated(keep="first")]
        combined = combined.sort_index()

        return combined
