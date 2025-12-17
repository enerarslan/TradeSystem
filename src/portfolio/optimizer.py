"""
Portfolio optimization for AlphaTrade system.

This module provides various optimization methods:
- Mean-Variance Optimization (Markowitz)
- Minimum Variance Portfolio
- Maximum Sharpe Ratio
- Risk Parity
- Black-Litterman
- Hierarchical Risk Parity (HRP)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from loguru import logger


class PortfolioOptimizer:
    """
    Portfolio optimizer with multiple methods.

    Provides institutional-grade portfolio optimization
    with various objectives and constraints.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.05,
        lookback: int | None = None,
    ) -> None:
        """
        Initialize the optimizer.

        Args:
            returns: Asset returns DataFrame
            risk_free_rate: Annual risk-free rate
            lookback: Lookback period (None for all data)
        """
        self.returns = returns.tail(lookback) if lookback else returns
        self.risk_free_rate = risk_free_rate / (252 * 26)  # Convert to per-bar
        self.n_assets = len(returns.columns)
        self.asset_names = returns.columns.tolist()

        # Pre-compute statistics
        self._mean_returns: pd.Series | None = None
        self._cov_matrix: pd.DataFrame | None = None

    @property
    def mean_returns(self) -> pd.Series:
        """Get mean returns."""
        if self._mean_returns is None:
            self._mean_returns = self.returns.mean()
        return self._mean_returns

    @property
    def cov_matrix(self) -> pd.DataFrame:
        """Get covariance matrix."""
        if self._cov_matrix is None:
            self._cov_matrix = self.returns.cov()
        return self._cov_matrix

    def optimize(
        self,
        method: Literal[
            "min_variance",
            "max_sharpe",
            "risk_parity",
            "hrp",
            "equal_weight",
        ] = "max_sharpe",
        constraints: dict | None = None,
    ) -> pd.Series:
        """
        Optimize portfolio weights.

        Args:
            method: Optimization method
            constraints: Constraint parameters

        Returns:
            Optimal weights
        """
        if method == "min_variance":
            weights = self._minimum_variance(constraints)
        elif method == "max_sharpe":
            weights = self._maximum_sharpe(constraints)
        elif method == "risk_parity":
            weights = self._risk_parity()
        elif method == "hrp":
            weights = self._hierarchical_risk_parity()
        elif method == "equal_weight":
            weights = pd.Series(
                1 / self.n_assets,
                index=self.asset_names,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        return weights

    def _minimum_variance(
        self,
        constraints: dict | None = None,
    ) -> pd.Series:
        """Calculate minimum variance portfolio."""
        constraints = constraints or {}
        min_weight = constraints.get("min_weight", 0.0)
        max_weight = constraints.get("max_weight", 1.0)
        long_only = constraints.get("long_only", True)

        # Objective: minimize portfolio variance
        def objective(weights):
            return weights @ self.cov_matrix.values @ weights

        # Constraints
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        # Bounds
        if long_only:
            bounds = [(min_weight, max_weight) for _ in range(self.n_assets)]
        else:
            bounds = [(-max_weight, max_weight) for _ in range(self.n_assets)]

        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets

        # Optimize
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )

        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")

        return pd.Series(result.x, index=self.asset_names)

    def _maximum_sharpe(
        self,
        constraints: dict | None = None,
    ) -> pd.Series:
        """Calculate maximum Sharpe ratio portfolio."""
        constraints = constraints or {}
        min_weight = constraints.get("min_weight", 0.0)
        max_weight = constraints.get("max_weight", 1.0)
        long_only = constraints.get("long_only", True)

        # Objective: minimize negative Sharpe ratio
        def objective(weights):
            port_return = weights @ self.mean_returns.values
            port_vol = np.sqrt(weights @ self.cov_matrix.values @ weights)
            if port_vol == 0:
                return 0
            return -(port_return - self.risk_free_rate) / port_vol

        # Constraints
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        # Bounds
        if long_only:
            bounds = [(min_weight, max_weight) for _ in range(self.n_assets)]
        else:
            bounds = [(-max_weight, max_weight) for _ in range(self.n_assets)]

        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets

        # Optimize
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )

        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")

        return pd.Series(result.x, index=self.asset_names)

    def _risk_parity(self) -> pd.Series:
        """Calculate risk parity portfolio."""
        # Objective: equal risk contribution
        def objective(weights):
            port_var = weights @ self.cov_matrix.values @ weights
            mrc = self.cov_matrix.values @ weights
            rc = weights * mrc / np.sqrt(port_var + 1e-10)
            rc = rc / rc.sum()

            # Target: 1/n for each asset
            target = 1 / self.n_assets
            return np.sum((rc - target) ** 2)

        # Constraints
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        # Bounds (long only)
        bounds = [(0.01, 0.5) for _ in range(self.n_assets)]

        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets

        # Optimize
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )

        return pd.Series(result.x, index=self.asset_names)

    def _hierarchical_risk_parity(self) -> pd.Series:
        """Calculate HRP portfolio weights."""
        # 1. Tree clustering
        corr = self.returns.corr()
        dist = np.sqrt((1 - corr) / 2)

        # Convert to condensed distance matrix
        dist_condensed = squareform(dist.values)
        link = linkage(dist_condensed, method="single")

        # Get sorted indices
        sorted_idx = leaves_list(link)
        sorted_assets = [self.asset_names[i] for i in sorted_idx]

        # 2. Quasi-diagonalization (reorder covariance matrix)
        cov_sorted = self.cov_matrix.loc[sorted_assets, sorted_assets]

        # 3. Recursive bisection
        weights = self._recursive_bisection(cov_sorted)

        # Reorder to original asset order
        return weights.reindex(self.asset_names)

    def _recursive_bisection(self, cov: pd.DataFrame) -> pd.Series:
        """Recursive bisection for HRP."""
        weights = pd.Series(1.0, index=cov.index)
        clusters = [cov.index.tolist()]

        while clusters:
            cluster = clusters.pop(0)

            if len(cluster) == 1:
                continue

            # Split cluster
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]

            # Calculate cluster variances
            left_cov = cov.loc[left, left]
            right_cov = cov.loc[right, right]

            # Inverse variance allocation
            left_var = self._get_cluster_variance(left_cov)
            right_var = self._get_cluster_variance(right_cov)

            alpha = 1 - left_var / (left_var + right_var)

            # Update weights
            weights[left] *= alpha
            weights[right] *= (1 - alpha)

            # Add sub-clusters if needed
            if len(left) > 1:
                clusters.append(left)
            if len(right) > 1:
                clusters.append(right)

        return weights / weights.sum()

    def _get_cluster_variance(self, cov: pd.DataFrame) -> float:
        """Calculate variance of equal-weight cluster."""
        n = len(cov)
        w = np.ones(n) / n
        return w @ cov.values @ w

    def efficient_frontier(
        self,
        n_points: int = 50,
        constraints: dict | None = None,
    ) -> pd.DataFrame:
        """
        Calculate efficient frontier.

        Args:
            n_points: Number of points on frontier
            constraints: Constraint parameters

        Returns:
            DataFrame with frontier portfolios
        """
        constraints = constraints or {}

        # Get min and max return portfolios
        min_var = self._minimum_variance(constraints)
        min_ret = min_var @ self.mean_returns

        max_sharpe = self._maximum_sharpe(constraints)
        max_ret = max_sharpe @ self.mean_returns

        # Target returns
        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier = []
        for target in target_returns:
            weights = self._target_return_portfolio(target, constraints)
            port_return = weights @ self.mean_returns.values
            port_vol = np.sqrt(weights @ self.cov_matrix.values @ weights)
            sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

            frontier.append({
                "return": port_return * 252 * 26,  # Annualize
                "volatility": port_vol * np.sqrt(252 * 26),
                "sharpe": sharpe * np.sqrt(252 * 26),
                "weights": weights.tolist(),
            })

        return pd.DataFrame(frontier)

    def _target_return_portfolio(
        self,
        target_return: float,
        constraints: dict | None = None,
    ) -> np.ndarray:
        """Find minimum variance portfolio for target return."""
        constraints = constraints or {}
        min_weight = constraints.get("min_weight", 0.0)
        max_weight = constraints.get("max_weight", 1.0)

        def objective(weights):
            return weights @ self.cov_matrix.values @ weights

        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w: w @ self.mean_returns.values - target_return},
        ]

        bounds = [(min_weight, max_weight) for _ in range(self.n_assets)]
        x0 = np.ones(self.n_assets) / self.n_assets

        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons)

        return result.x

    def black_litterman(
        self,
        views: dict[str, float],
        view_confidence: dict[str, float] | None = None,
        tau: float = 0.05,
    ) -> pd.Series:
        """
        Black-Litterman model optimization.

        Args:
            views: Dictionary of asset -> expected return views
            view_confidence: Confidence in each view (0-1)
            tau: Uncertainty in prior

        Returns:
            Optimal weights
        """
        # Market equilibrium returns (reverse optimization from market cap weights)
        # Assuming equal weights as proxy for market cap weights
        market_weights = np.ones(self.n_assets) / self.n_assets
        delta = 2.5  # Risk aversion coefficient

        # Equilibrium returns
        pi = delta * self.cov_matrix.values @ market_weights

        # Create view matrices
        n_views = len(views)
        P = np.zeros((n_views, self.n_assets))
        Q = np.zeros(n_views)

        for i, (asset, view_return) in enumerate(views.items()):
            if asset in self.asset_names:
                idx = self.asset_names.index(asset)
                P[i, idx] = 1
                Q[i] = view_return

        # View uncertainty
        if view_confidence:
            omega_diag = [
                (1 - view_confidence.get(asset, 0.5)) * 0.1
                for asset in views.keys()
            ]
        else:
            omega_diag = [0.05] * n_views

        omega = np.diag(omega_diag)

        # Black-Litterman formula
        tau_sigma = tau * self.cov_matrix.values

        # Posterior expected returns
        inv1 = np.linalg.inv(tau_sigma)
        inv2 = np.linalg.inv(P @ tau_sigma @ P.T + omega)

        posterior_mean = np.linalg.inv(inv1 + P.T @ inv2 @ P) @ (
            inv1 @ pi + P.T @ inv2 @ Q
        )

        # Optimize using posterior returns
        def objective(weights):
            port_return = weights @ posterior_mean
            port_vol = np.sqrt(weights @ self.cov_matrix.values @ weights)
            return -(port_return / port_vol)

        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, 0.2) for _ in range(self.n_assets)]
        x0 = np.ones(self.n_assets) / self.n_assets

        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons)

        return pd.Series(result.x, index=self.asset_names)


# Convenience functions
def mean_variance_optimize(
    returns: pd.DataFrame,
    target_return: float | None = None,
) -> pd.Series:
    """Convenience function for mean-variance optimization."""
    optimizer = PortfolioOptimizer(returns)
    if target_return:
        weights = optimizer._target_return_portfolio(target_return)
        return pd.Series(weights, index=returns.columns)
    return optimizer.optimize(method="max_sharpe")


def minimum_variance_portfolio(returns: pd.DataFrame) -> pd.Series:
    """Convenience function for minimum variance portfolio."""
    optimizer = PortfolioOptimizer(returns)
    return optimizer.optimize(method="min_variance")


def maximum_sharpe_portfolio(returns: pd.DataFrame) -> pd.Series:
    """Convenience function for maximum Sharpe portfolio."""
    optimizer = PortfolioOptimizer(returns)
    return optimizer.optimize(method="max_sharpe")


def risk_parity_portfolio(returns: pd.DataFrame) -> pd.Series:
    """Convenience function for risk parity portfolio."""
    optimizer = PortfolioOptimizer(returns)
    return optimizer.optimize(method="risk_parity")


def hierarchical_risk_parity(returns: pd.DataFrame) -> pd.Series:
    """Convenience function for HRP portfolio."""
    optimizer = PortfolioOptimizer(returns)
    return optimizer.optimize(method="hrp")
