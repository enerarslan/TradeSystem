"""
Position sizing algorithms for AlphaTrade system.

This module provides various position sizing methods:
- Fixed fractional
- Kelly criterion
- Volatility targeting
- Risk parity
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger


class PositionSizer:
    """
    Position sizing calculator with multiple methods.

    Provides institutional-grade position sizing based on:
    - Fixed fraction of capital
    - Kelly criterion
    - Volatility targeting
    - Risk parity
    """

    def __init__(
        self,
        method: Literal["fixed_fraction", "kelly", "volatility_target", "risk_parity"] = "volatility_target",
        params: dict | None = None,
    ) -> None:
        """
        Initialize the position sizer.

        Args:
            method: Sizing method
            params: Method parameters
        """
        self.method = method
        self.params = params or {}

        # Default parameters
        self._defaults = {
            "fixed_fraction": {"fraction": 0.02},
            "kelly": {"kelly_fraction": 0.25, "min_win_rate": 0.4},
            "volatility_target": {"target_vol": 0.15, "lookback": 60},
            "risk_parity": {"lookback": 60},
        }

        # Merge with defaults
        if method in self._defaults:
            for key, value in self._defaults[method].items():
                if key not in self.params:
                    self.params[key] = value

    def calculate_size(
        self,
        capital: float,
        price: float,
        volatility: float | None = None,
        win_rate: float | None = None,
        avg_win: float | None = None,
        avg_loss: float | None = None,
        **kwargs,
    ) -> float:
        """
        Calculate position size.

        Args:
            capital: Available capital
            price: Current price
            volatility: Asset volatility (annualized)
            win_rate: Historical win rate (for Kelly)
            avg_win: Average win size (for Kelly)
            avg_loss: Average loss size (for Kelly)
            **kwargs: Additional parameters

        Returns:
            Position size in shares/units
        """
        if self.method == "fixed_fraction":
            position_value = fixed_fraction(
                capital,
                self.params["fraction"],
            )

        elif self.method == "kelly":
            if win_rate is None or avg_win is None or avg_loss is None:
                logger.warning("Kelly requires win_rate, avg_win, avg_loss. Using fixed fraction.")
                position_value = fixed_fraction(capital, 0.02)
            else:
                kelly_f = kelly_criterion(win_rate, avg_win, avg_loss)
                # Apply Kelly fraction (partial Kelly)
                kelly_f *= self.params["kelly_fraction"]
                position_value = capital * kelly_f

        elif self.method == "volatility_target":
            if volatility is None:
                logger.warning("Volatility target requires volatility. Using fixed fraction.")
                position_value = fixed_fraction(capital, 0.02)
            else:
                position_value = volatility_target(
                    capital,
                    volatility,
                    self.params["target_vol"],
                )

        elif self.method == "risk_parity":
            # Risk parity handled at portfolio level
            position_value = fixed_fraction(capital, 0.02)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Convert to shares
        shares = position_value / price

        return shares

    def calculate_portfolio_weights(
        self,
        returns: pd.DataFrame,
        target_vol: float | None = None,
    ) -> pd.Series:
        """
        Calculate portfolio weights using selected method.

        Args:
            returns: Asset returns DataFrame
            target_vol: Target portfolio volatility

        Returns:
            Weights series
        """
        if self.method == "risk_parity":
            return risk_parity_weights(returns)

        elif self.method == "volatility_target":
            target_vol = target_vol or self.params["target_vol"]
            vols = returns.std() * np.sqrt(252 * 26)

            # Inverse volatility weights
            inv_vol = 1 / (vols + 1e-10)
            weights = inv_vol / inv_vol.sum()

            # Scale to target volatility
            portfolio_vol = np.sqrt(
                weights.values @ returns.cov().values @ weights.values
            ) * np.sqrt(252 * 26)

            if portfolio_vol > 0:
                scale = target_vol / portfolio_vol
                weights *= scale

            return weights

        else:
            # Equal weight fallback
            n = len(returns.columns)
            return pd.Series(1 / n, index=returns.columns)


def fixed_fraction(
    capital: float,
    fraction: float = 0.02,
) -> float:
    """
    Calculate position size using fixed fraction.

    Args:
        capital: Available capital
        fraction: Fraction of capital to risk

    Returns:
        Position value
    """
    return capital * fraction


def kelly_criterion(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """
    Calculate optimal fraction using Kelly criterion.

    Args:
        win_rate: Probability of winning
        avg_win: Average win amount
        avg_loss: Average loss amount

    Returns:
        Kelly fraction (0-1)
    """
    if avg_loss == 0:
        return 0

    # Kelly formula: f* = (p * b - q) / b
    # where p = win rate, q = 1 - p, b = win/loss ratio
    b = avg_win / abs(avg_loss)
    q = 1 - win_rate

    kelly = (win_rate * b - q) / b

    # Clip to reasonable range
    kelly = max(0, min(kelly, 0.5))

    return kelly


def volatility_target(
    capital: float,
    asset_vol: float,
    target_vol: float = 0.15,
) -> float:
    """
    Calculate position size to target specific volatility.

    Args:
        capital: Available capital
        asset_vol: Asset volatility (annualized)
        target_vol: Target position volatility

    Returns:
        Position value
    """
    if asset_vol <= 0:
        return 0

    # Position that targets specific volatility contribution
    position_value = capital * (target_vol / asset_vol)

    return position_value


def risk_parity_weights(
    returns: pd.DataFrame,
    lookback: int = 60,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    verbose: bool = False,
) -> pd.Series:
    """
    Calculate risk parity weights.

    Each asset contributes equal risk to portfolio.

    Args:
        returns: Asset returns DataFrame
        lookback: Lookback period for covariance
        max_iterations: Maximum optimization iterations
        tolerance: Convergence tolerance
        verbose: Log iteration details

    Returns:
        Risk parity weights
    """
    # Use recent returns
    recent_returns = returns.tail(lookback)

    # Covariance matrix
    cov = recent_returns.cov()

    # Check for degenerate covariance matrix
    if cov.isnull().any().any():
        logger.warning("Risk parity: Covariance matrix contains NaN values, using equal weights")
        return pd.Series(1.0 / len(returns.columns), index=returns.columns)

    # Initial equal weights
    n = len(returns.columns)
    weights = np.ones(n) / n

    # Track convergence metrics
    converged = False
    final_iteration = 0
    max_weight_change = 0.0

    # Iterative optimization for risk parity
    for iteration in range(max_iterations):
        # Portfolio variance
        port_var = weights @ cov.values @ weights

        if port_var <= 0:
            logger.warning(f"Risk parity: Non-positive portfolio variance at iteration {iteration}")
            break

        # Marginal risk contribution
        mrc = cov.values @ weights

        # Risk contribution
        rc = weights * mrc / np.sqrt(port_var)

        # Target: equal risk contribution
        target_rc = port_var / n

        # Adjustment
        adjustment = target_rc / (rc + 1e-10)
        adjustment = np.clip(adjustment, 0.5, 2.0)

        # Update weights
        new_weights = weights * adjustment
        new_weights = new_weights / new_weights.sum()

        # Calculate weight change for convergence check
        max_weight_change = np.max(np.abs(new_weights - weights))

        if verbose and iteration % 10 == 0:
            logger.debug(
                f"Risk parity iteration {iteration}: max_weight_change={max_weight_change:.8f}"
            )

        # Check convergence
        if max_weight_change < tolerance:
            converged = True
            final_iteration = iteration
            logger.debug(f"Risk parity converged in {iteration + 1} iterations (tolerance: {tolerance})")
            break

        weights = new_weights
        final_iteration = iteration

    # Log warning if not converged (JPMorgan-level monitoring)
    if not converged:
        logger.warning(
            f"Risk parity optimization did NOT converge after {max_iterations} iterations. "
            f"Final max weight change: {max_weight_change:.8f} (tolerance: {tolerance}). "
            "Consider increasing max_iterations or relaxing tolerance. "
            "Returning best weights found."
        )

        # Calculate final risk contributions for diagnostics
        port_var = weights @ cov.values @ weights
        if port_var > 0:
            mrc = cov.values @ weights
            rc = weights * mrc / np.sqrt(port_var)
            rc_pct = rc / rc.sum() * 100
            logger.debug(f"Final risk contributions (%): {dict(zip(returns.columns, rc_pct.round(2)))}")

    return pd.Series(weights, index=returns.columns)


class RiskBudget:
    """
    Risk budgeting for portfolio allocation.

    Allocates risk budget across assets based on:
    - Equal risk contribution
    - Custom risk budgets
    - Factor-based allocation
    """

    def __init__(
        self,
        total_risk: float = 0.15,
        risk_budgets: dict[str, float] | None = None,
    ) -> None:
        """
        Initialize the risk budget.

        Args:
            total_risk: Total portfolio risk budget
            risk_budgets: Risk budgets per asset (sum to 1)
        """
        self.total_risk = total_risk
        self.risk_budgets = risk_budgets

    def allocate(
        self,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """
        Allocate risk budget to portfolio weights.

        Args:
            returns: Asset returns DataFrame

        Returns:
            Portfolio weights
        """
        n = len(returns.columns)
        cov = returns.cov() * 252 * 26  # Annualize

        if self.risk_budgets is None:
            # Equal risk budget
            budgets = {col: 1 / n for col in returns.columns}
        else:
            budgets = self.risk_budgets

        budget_array = np.array([budgets.get(col, 1 / n) for col in returns.columns])
        budget_array = budget_array / budget_array.sum()

        # Optimize for risk budget
        weights = self._optimize_risk_budget(cov.values, budget_array)

        # Scale to target volatility
        port_vol = np.sqrt(weights @ cov.values @ weights)
        if port_vol > 0:
            weights = weights * (self.total_risk / port_vol)

        return pd.Series(weights, index=returns.columns)

    def _optimize_risk_budget(
        self,
        cov: np.ndarray,
        budgets: np.ndarray,
        max_iter: int = 100,
    ) -> np.ndarray:
        """
        Optimize weights for target risk budget.

        Args:
            cov: Covariance matrix
            budgets: Target risk budgets
            max_iter: Maximum iterations

        Returns:
            Optimal weights
        """
        n = len(budgets)
        weights = np.ones(n) / n

        for _ in range(max_iter):
            # Portfolio variance
            port_var = weights @ cov @ weights

            # Marginal risk contribution
            mrc = cov @ weights

            # Risk contribution
            rc = weights * mrc / np.sqrt(port_var + 1e-10)
            rc = rc / rc.sum()  # Normalize to percentages

            # Adjustment based on budget difference
            adjustment = np.sqrt(budgets / (rc + 1e-10))
            adjustment = np.clip(adjustment, 0.5, 2.0)

            # Update weights
            new_weights = weights * adjustment
            new_weights = new_weights / new_weights.sum()

            if np.max(np.abs(new_weights - weights)) < 1e-6:
                break

            weights = new_weights

        return weights

    def get_risk_contributions(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """
        Calculate risk contribution of each asset.

        Args:
            weights: Portfolio weights
            returns: Asset returns

        Returns:
            Risk contributions
        """
        cov = returns.cov() * 252 * 26

        weights_array = weights.values
        port_var = weights_array @ cov.values @ weights_array

        # Marginal risk contribution
        mrc = cov.values @ weights_array

        # Risk contribution
        rc = weights_array * mrc / np.sqrt(port_var + 1e-10)

        return pd.Series(rc / rc.sum(), index=weights.index)
