"""
Asset allocation for AlphaTrade system.

This module provides:
- Constraint application
- Sector allocation
- Position scaling
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class AssetAllocator:
    """
    Asset allocation manager.

    Handles:
    - Weight constraints
    - Sector exposure limits
    - Position scaling
    """

    def __init__(
        self,
        max_position: float = 0.05,
        min_position: float = 0.0,
        max_sector_exposure: float = 0.25,
        max_leverage: float = 1.0,
        sectors: dict[str, list[str]] | None = None,
    ) -> None:
        """
        Initialize the allocator.

        Args:
            max_position: Maximum weight per position
            min_position: Minimum weight per position
            max_sector_exposure: Maximum sector exposure
            max_leverage: Maximum total leverage
            sectors: Sector mapping (sector -> symbols)
        """
        self.max_position = max_position
        self.min_position = min_position
        self.max_sector_exposure = max_sector_exposure
        self.max_leverage = max_leverage
        self.sectors = sectors or {}

        # Create reverse mapping
        self._symbol_to_sector = {}
        for sector, symbols in self.sectors.items():
            for symbol in symbols:
                self._symbol_to_sector[symbol] = sector

    def apply_constraints(
        self,
        weights: pd.Series,
    ) -> pd.Series:
        """
        Apply all constraints to weights.

        Args:
            weights: Raw portfolio weights

        Returns:
            Constrained weights
        """
        weights = weights.copy()

        # Apply position limits
        weights = self._apply_position_limits(weights)

        # Apply sector limits
        weights = self._apply_sector_limits(weights)

        # Apply leverage limit
        weights = self._apply_leverage_limit(weights)

        return weights

    def _apply_position_limits(
        self,
        weights: pd.Series,
    ) -> pd.Series:
        """Apply per-position limits."""
        # Clip to max/min
        weights = weights.clip(
            lower=-self.max_position if self.min_position == 0 else self.min_position,
            upper=self.max_position,
        )

        return weights

    def _apply_sector_limits(
        self,
        weights: pd.Series,
    ) -> pd.Series:
        """Apply sector exposure limits."""
        if not self.sectors:
            return weights

        adjusted = weights.copy()

        for sector, symbols in self.sectors.items():
            sector_weights = adjusted[[s for s in symbols if s in adjusted.index]]

            if sector_weights.empty:
                continue

            sector_exposure = sector_weights.abs().sum()

            if sector_exposure > self.max_sector_exposure:
                # Scale down sector weights
                scale = self.max_sector_exposure / sector_exposure
                for symbol in sector_weights.index:
                    adjusted[symbol] *= scale

        return adjusted

    def _apply_leverage_limit(
        self,
        weights: pd.Series,
    ) -> pd.Series:
        """Apply total leverage limit."""
        total_exposure = weights.abs().sum()

        if total_exposure > self.max_leverage:
            scale = self.max_leverage / total_exposure
            weights = weights * scale

        return weights

    def get_sector_exposure(
        self,
        weights: pd.Series,
    ) -> pd.Series:
        """
        Calculate sector exposures.

        Args:
            weights: Portfolio weights

        Returns:
            Series of sector exposures
        """
        exposures = {}

        for sector, symbols in self.sectors.items():
            sector_weights = weights[[s for s in symbols if s in weights.index]]
            exposures[sector] = sector_weights.sum()

        return pd.Series(exposures)

    def check_constraints(
        self,
        weights: pd.Series,
    ) -> dict[str, Any]:
        """
        Check constraint violations.

        Args:
            weights: Portfolio weights

        Returns:
            Dictionary with violation information
        """
        violations = []

        # Position limits
        over_max = weights[weights.abs() > self.max_position]
        if not over_max.empty:
            violations.append({
                "type": "position_limit",
                "assets": over_max.index.tolist(),
                "values": over_max.tolist(),
            })

        # Sector limits
        sector_exp = self.get_sector_exposure(weights)
        over_sector = sector_exp[sector_exp.abs() > self.max_sector_exposure]
        if not over_sector.empty:
            violations.append({
                "type": "sector_limit",
                "sectors": over_sector.index.tolist(),
                "values": over_sector.tolist(),
            })

        # Leverage
        total_leverage = weights.abs().sum()
        if total_leverage > self.max_leverage:
            violations.append({
                "type": "leverage_limit",
                "current": total_leverage,
                "limit": self.max_leverage,
            })

        return {
            "is_valid": len(violations) == 0,
            "violations": violations,
            "total_exposure": total_leverage,
            "sector_exposures": sector_exp.to_dict(),
        }

    def optimize_with_constraints(
        self,
        target_weights: pd.Series,
        current_weights: pd.Series | None = None,
        max_turnover: float | None = None,
    ) -> pd.Series:
        """
        Optimize weights while respecting constraints.

        Args:
            target_weights: Desired target weights
            current_weights: Current weights (for turnover)
            max_turnover: Maximum allowed turnover

        Returns:
            Optimized weights
        """
        from scipy.optimize import minimize

        n = len(target_weights)
        assets = target_weights.index.tolist()

        # Objective: minimize deviation from target
        def objective(w):
            return np.sum((w - target_weights.values) ** 2)

        # Constraints
        constraints = []

        # Sum to 1 (or less for partial investment)
        constraints.append({
            "type": "ineq",
            "fun": lambda w: self.max_leverage - np.sum(np.abs(w)),
        })

        # Sector constraints
        for sector, symbols in self.sectors.items():
            indices = [i for i, a in enumerate(assets) if a in symbols]
            if indices:
                constraints.append({
                    "type": "ineq",
                    "fun": lambda w, idx=indices: self.max_sector_exposure - np.sum(np.abs(w[idx])),
                })

        # Turnover constraint
        if max_turnover is not None and current_weights is not None:
            current = current_weights.reindex(assets, fill_value=0).values
            constraints.append({
                "type": "ineq",
                "fun": lambda w: max_turnover - np.sum(np.abs(w - current)) / 2,
            })

        # Bounds
        bounds = [(-self.max_position, self.max_position) for _ in range(n)]

        # Optimize
        x0 = target_weights.values
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return pd.Series(result.x, index=assets)


def apply_constraints(
    weights: pd.Series,
    max_position: float = 0.05,
    max_sector: float = 0.25,
    max_leverage: float = 1.0,
    sectors: dict | None = None,
) -> pd.Series:
    """
    Convenience function to apply constraints.

    Args:
        weights: Raw weights
        max_position: Max position weight
        max_sector: Max sector exposure
        max_leverage: Max leverage
        sectors: Sector mapping

    Returns:
        Constrained weights
    """
    allocator = AssetAllocator(
        max_position=max_position,
        max_sector_exposure=max_sector,
        max_leverage=max_leverage,
        sectors=sectors,
    )
    return allocator.apply_constraints(weights)


class TargetPortfolio:
    """
    Target portfolio management.

    Maintains target allocation and provides methods
    for calculating required trades.
    """

    def __init__(
        self,
        target_weights: pd.Series,
        rebalance_threshold: float = 0.05,
    ) -> None:
        """
        Initialize target portfolio.

        Args:
            target_weights: Target allocation weights
            rebalance_threshold: Threshold for rebalancing
        """
        self.target_weights = target_weights
        self.rebalance_threshold = rebalance_threshold

    def get_drift(self, current_weights: pd.Series) -> pd.Series:
        """
        Calculate drift from target.

        Args:
            current_weights: Current allocation

        Returns:
            Drift series
        """
        # Align indices
        all_assets = self.target_weights.index.union(current_weights.index)
        target = self.target_weights.reindex(all_assets, fill_value=0)
        current = current_weights.reindex(all_assets, fill_value=0)

        return current - target

    def needs_rebalance(self, current_weights: pd.Series) -> bool:
        """
        Check if rebalancing is needed.

        Args:
            current_weights: Current allocation

        Returns:
            True if rebalancing needed
        """
        drift = self.get_drift(current_weights)
        return drift.abs().max() > self.rebalance_threshold

    def get_trades(
        self,
        current_weights: pd.Series,
        portfolio_value: float,
    ) -> pd.DataFrame:
        """
        Calculate trades to reach target.

        Args:
            current_weights: Current allocation
            portfolio_value: Total portfolio value

        Returns:
            DataFrame with trades
        """
        drift = self.get_drift(current_weights)

        trades = []
        for asset in drift.index:
            if abs(drift[asset]) > 0.001:  # Minimum trade size
                trades.append({
                    "asset": asset,
                    "weight_change": -drift[asset],
                    "trade_value": -drift[asset] * portfolio_value,
                    "side": "BUY" if drift[asset] < 0 else "SELL",
                })

        return pd.DataFrame(trades)

    def update_targets(self, new_weights: pd.Series) -> None:
        """Update target weights."""
        self.target_weights = new_weights
