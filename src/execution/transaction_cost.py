"""
Transaction cost analysis for AlphaTrade system.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class TransactionCostAnalyzer:
    """
    Transaction cost analysis.

    Analyzes and reports on trading costs including:
    - Commission
    - Slippage
    - Market impact
    """

    def __init__(
        self,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        min_commission: float = 1.0,
    ) -> None:
        """
        Initialize analyzer.

        Args:
            commission_pct: Commission percentage
            slippage_pct: Slippage percentage
            min_commission: Minimum commission per trade
        """
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.min_commission = min_commission

    def estimate_round_trip_cost(
        self,
        trade_value: float,
    ) -> dict[str, float]:
        """
        Estimate round-trip trading cost.

        Args:
            trade_value: Value of the trade

        Returns:
            Dictionary with cost breakdown
        """
        # Entry costs
        entry_commission = max(trade_value * self.commission_pct, self.min_commission)
        entry_slippage = trade_value * self.slippage_pct

        # Exit costs (similar)
        exit_commission = max(trade_value * self.commission_pct, self.min_commission)
        exit_slippage = trade_value * self.slippage_pct

        total = entry_commission + entry_slippage + exit_commission + exit_slippage

        return {
            "entry_commission": entry_commission,
            "entry_slippage": entry_slippage,
            "exit_commission": exit_commission,
            "exit_slippage": exit_slippage,
            "total_cost": total,
            "cost_pct": total / trade_value * 100,
        }

    def analyze_trades(
        self,
        trades: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Analyze costs for a set of trades.

        Args:
            trades: DataFrame with trades

        Returns:
            DataFrame with cost analysis
        """
        if trades.empty:
            return pd.DataFrame()

        analysis = trades.copy()

        # Calculate costs
        analysis["commission"] = trades["value"].apply(
            lambda v: max(v * self.commission_pct, self.min_commission)
        )
        analysis["slippage"] = trades["value"] * self.slippage_pct
        analysis["total_cost"] = analysis["commission"] + analysis["slippage"]
        analysis["cost_pct"] = analysis["total_cost"] / analysis["value"] * 100

        return analysis

    def get_summary(
        self,
        trades: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Get cost summary.

        Args:
            trades: DataFrame with trades

        Returns:
            Summary dictionary
        """
        if trades.empty:
            return {}

        analysis = self.analyze_trades(trades)

        return {
            "total_trades": len(analysis),
            "total_volume": analysis["value"].sum(),
            "total_commission": analysis["commission"].sum(),
            "total_slippage": analysis["slippage"].sum(),
            "total_costs": analysis["total_cost"].sum(),
            "avg_cost_per_trade": analysis["total_cost"].mean(),
            "avg_cost_pct": analysis["cost_pct"].mean(),
            "cost_as_pct_of_volume": analysis["total_cost"].sum() / analysis["value"].sum() * 100,
        }


def estimate_costs(
    trade_value: float,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
) -> float:
    """
    Convenience function to estimate trading costs.

    Args:
        trade_value: Trade value
        commission_pct: Commission percentage
        slippage_pct: Slippage percentage

    Returns:
        Total estimated cost
    """
    analyzer = TransactionCostAnalyzer(
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
    )
    costs = analyzer.estimate_round_trip_cost(trade_value)
    return costs["total_cost"]
