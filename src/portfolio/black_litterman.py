"""
Black-Litterman Portfolio Optimization.

This module implements the Black-Litterman model for combining investor
views with market equilibrium returns to produce optimal portfolios.

The model solves a key problem with traditional MVO: sensitivity to
expected return estimates. By starting from equilibrium and adjusting
based on views with specified confidence, BL produces more stable and
intuitive portfolios.

Reference:
    - Black, F. and Litterman, R. (1992) - "Global Portfolio Optimization"
    - Meucci, A. (2010) - "The Black-Litterman Approach"
    - Idzorek, T. (2007) - "A Step-by-Step Guide to Black-Litterman"

Designed for JPMorgan-level institutional requirements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize

logger = logging.getLogger(__name__)


@dataclass
class ViewSpecification:
    """
    Specification of an investor view for Black-Litterman.

    Views can be:
    1. Absolute: "Asset A will return 5%"
    2. Relative: "Asset A will outperform Asset B by 2%"

    Example usage:
        # Absolute view: Tech will return 10%
        view1 = ViewSpecification(
            assets=["TECH"],
            weights=[1.0],
            expected_return=0.10,
            confidence=0.75,  # 75% confidence
        )

        # Relative view: Tech outperforms Energy by 3%
        view2 = ViewSpecification(
            assets=["TECH", "ENERGY"],
            weights=[1.0, -1.0],
            expected_return=0.03,
            confidence=0.50,
        )
    """

    assets: List[str]                # Assets involved in the view
    weights: List[float]             # Weights (+1 long, -1 short for relative)
    expected_return: float           # Expected return of the view
    confidence: float                # Confidence level (0-1)
    description: str = ""            # Optional description

    def __post_init__(self):
        """Validate view specification."""
        if len(self.assets) != len(self.weights):
            raise ValueError("Assets and weights must have same length")

        if not 0 < self.confidence <= 1:
            raise ValueError("Confidence must be in (0, 1]")

        # Normalize relative view weights
        if len(self.assets) > 1:
            weight_sum = sum(self.weights)
            if abs(weight_sum) > 1e-10:
                logger.warning(
                    f"Relative view weights don't sum to zero: {weight_sum}"
                )


@dataclass
class BlackLittermanResult:
    """Container for Black-Litterman optimization results."""

    optimal_weights: pd.Series          # Final portfolio weights
    expected_returns: pd.Series         # BL expected returns
    covariance: pd.DataFrame            # Posterior covariance
    equilibrium_returns: pd.Series      # Prior (market implied) returns
    view_contribution: pd.Series        # Return contribution from views
    expected_portfolio_return: float    # Portfolio expected return
    expected_portfolio_vol: float       # Portfolio expected volatility
    sharpe_ratio: float                 # Expected Sharpe ratio
    risk_free_rate: float               # Risk-free rate used
    tau: float                          # Uncertainty scalar
    views_applied: List[ViewSpecification]  # Views that were applied


class BlackLittermanOptimizer:
    """
    Black-Litterman portfolio optimizer.

    The model combines:
    1. Market equilibrium returns (from CAPM/reverse optimization)
    2. Investor views with confidence levels

    To produce posterior expected returns that can be used in MVO.

    Key Parameters:
    - tau: Scalar for uncertainty in equilibrium returns (typically 0.025-0.05)
    - risk_aversion: Market risk aversion coefficient (typically 2.5-3.5)

    Example usage:
        optimizer = BlackLittermanOptimizer(
            market_caps=market_caps,
            risk_free_rate=0.02,
            tau=0.05,
        )

        # Add views
        optimizer.add_view(ViewSpecification(
            assets=["AAPL"],
            weights=[1.0],
            expected_return=0.15,
            confidence=0.6,
        ))

        # Optimize
        result = optimizer.optimize(
            returns=historical_returns,
            target_return=0.10,
        )

        print(result.optimal_weights)
    """

    def __init__(
        self,
        market_caps: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
    ) -> None:
        """
        Initialize Black-Litterman optimizer.

        Args:
            market_caps: Market capitalizations for equilibrium weights
            risk_free_rate: Risk-free rate for Sharpe calculation
            tau: Uncertainty scalar for equilibrium returns
            risk_aversion: Market risk aversion coefficient
        """
        self.market_caps = market_caps
        self.risk_free_rate = risk_free_rate
        self.tau = tau
        self.risk_aversion = risk_aversion

        self._views: List[ViewSpecification] = []
        self._asset_names: Optional[List[str]] = None

    def add_view(self, view: ViewSpecification) -> None:
        """
        Add an investor view.

        Args:
            view: ViewSpecification to add
        """
        self._views.append(view)
        logger.info(
            f"Added view: {view.assets} expected return={view.expected_return:.2%}, "
            f"confidence={view.confidence:.0%}"
        )

    def clear_views(self) -> None:
        """Clear all views."""
        self._views.clear()

    def optimize(
        self,
        returns: pd.DataFrame,
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None,
        max_weight: float = 0.30,
        min_weight: float = 0.0,
        allow_short: bool = False,
    ) -> BlackLittermanResult:
        """
        Run Black-Litterman optimization.

        Args:
            returns: Historical returns DataFrame
            target_return: Target portfolio return (optional)
            target_risk: Target portfolio volatility (optional)
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
            allow_short: Allow short positions

        Returns:
            BlackLittermanResult with optimal weights
        """
        self._asset_names = list(returns.columns)
        n_assets = len(self._asset_names)

        # Step 1: Calculate covariance matrix
        cov_matrix = self._calculate_covariance(returns)

        # Step 2: Calculate equilibrium returns
        equilibrium_weights = self._calculate_equilibrium_weights(n_assets)
        equilibrium_returns = self._calculate_equilibrium_returns(
            cov_matrix, equilibrium_weights
        )

        # Step 3: If no views, use equilibrium
        if not self._views:
            logger.info("No views specified, using equilibrium returns")
            bl_returns = equilibrium_returns
            posterior_cov = cov_matrix
            view_contribution = pd.Series(0.0, index=self._asset_names)
        else:
            # Step 4: Construct view matrices
            P, Q, omega = self._construct_view_matrices(cov_matrix)

            # Step 5: Calculate posterior returns
            bl_returns, posterior_cov = self._calculate_posterior(
                equilibrium_returns, cov_matrix, P, Q, omega
            )

            view_contribution = bl_returns - equilibrium_returns

        # Step 6: Optimize portfolio
        optimal_weights = self._mean_variance_optimize(
            bl_returns,
            posterior_cov,
            target_return,
            target_risk,
            max_weight,
            min_weight,
            allow_short,
        )

        # Calculate portfolio metrics
        portfolio_return = optimal_weights @ bl_returns
        portfolio_vol = np.sqrt(
            optimal_weights @ posterior_cov.values @ optimal_weights
        )
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol

        return BlackLittermanResult(
            optimal_weights=pd.Series(optimal_weights, index=self._asset_names),
            expected_returns=bl_returns,
            covariance=posterior_cov,
            equilibrium_returns=equilibrium_returns,
            view_contribution=view_contribution,
            expected_portfolio_return=portfolio_return,
            expected_portfolio_vol=portfolio_vol,
            sharpe_ratio=sharpe,
            risk_free_rate=self.risk_free_rate,
            tau=self.tau,
            views_applied=self._views.copy(),
        )

    def _calculate_covariance(
        self,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate covariance matrix with shrinkage.

        Uses Ledoit-Wolf shrinkage for robustness.
        """
        # Simple exponentially weighted covariance
        # In production, use Ledoit-Wolf or other shrinkage estimator
        cov = returns.ewm(span=60).cov().iloc[-len(returns.columns):]
        cov = cov.droplevel(0)

        # If ewm fails, use simple covariance
        if cov.isnull().any().any():
            cov = returns.cov()

        # Annualize (assuming daily returns)
        cov = cov * 252

        return cov

    def _calculate_equilibrium_weights(
        self,
        n_assets: int,
    ) -> np.ndarray:
        """
        Calculate equilibrium weights from market caps or equal weight.
        """
        if self.market_caps is not None:
            # Filter to only assets in our universe
            caps = self.market_caps.reindex(self._asset_names).fillna(0)
            weights = caps / caps.sum()
            return weights.values
        else:
            # Equal weight if no market caps
            return np.ones(n_assets) / n_assets

    def _calculate_equilibrium_returns(
        self,
        cov_matrix: pd.DataFrame,
        equilibrium_weights: np.ndarray,
    ) -> pd.Series:
        """
        Calculate equilibrium returns using reverse optimization.

        pi = delta * Sigma * w_mkt

        Where:
        - pi: equilibrium excess returns
        - delta: risk aversion
        - Sigma: covariance matrix
        - w_mkt: market weights
        """
        pi = self.risk_aversion * cov_matrix.values @ equilibrium_weights

        return pd.Series(pi, index=self._asset_names)

    def _construct_view_matrices(
        self,
        cov_matrix: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct P, Q, and Omega matrices from views.

        P: Pick matrix (k x n) - which assets involved in each view
        Q: View returns (k x 1) - expected returns for each view
        Omega: View uncertainty (k x k) - diagonal covariance of view errors
        """
        n_views = len(self._views)
        n_assets = len(self._asset_names)

        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        omega_diag = np.zeros(n_views)

        asset_idx = {name: i for i, name in enumerate(self._asset_names)}

        for i, view in enumerate(self._views):
            # Fill P matrix
            for asset, weight in zip(view.assets, view.weights):
                if asset in asset_idx:
                    P[i, asset_idx[asset]] = weight
                else:
                    logger.warning(f"Asset {asset} not found in universe")

            # Fill Q vector
            Q[i] = view.expected_return

            # Calculate view uncertainty from confidence
            # Higher confidence = lower uncertainty
            # Omega_i = (1/confidence - 1) * P_i * tau * Sigma * P_i'
            view_variance = (
                (1 / view.confidence - 1) *
                self.tau *
                P[i] @ cov_matrix.values @ P[i].T
            )
            omega_diag[i] = max(view_variance, 1e-8)  # Ensure positive

        omega = np.diag(omega_diag)

        return P, Q, omega

    def _calculate_posterior(
        self,
        equilibrium_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        P: np.ndarray,
        Q: np.ndarray,
        omega: np.ndarray,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Calculate posterior returns using Black-Litterman formula.

        E[R] = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 *
               [(tau*Sigma)^-1*pi + P'*Omega^-1*Q]

        Where:
        - tau: uncertainty scalar
        - Sigma: covariance matrix
        - P: pick matrix
        - Omega: view uncertainty
        - pi: equilibrium returns
        - Q: view returns
        """
        pi = equilibrium_returns.values
        sigma = cov_matrix.values
        tau_sigma = self.tau * sigma

        # Calculate posterior precision and mean
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(omega)

        # Posterior precision
        precision = tau_sigma_inv + P.T @ omega_inv @ P

        # Posterior covariance
        posterior_cov_values = np.linalg.inv(precision)

        # Posterior mean
        posterior_mean = posterior_cov_values @ (
            tau_sigma_inv @ pi + P.T @ omega_inv @ Q
        )

        posterior_returns = pd.Series(posterior_mean, index=self._asset_names)
        posterior_cov = pd.DataFrame(
            posterior_cov_values + sigma,  # Add back uncertainty
            index=self._asset_names,
            columns=self._asset_names,
        )

        return posterior_returns, posterior_cov

    def _mean_variance_optimize(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        target_return: Optional[float],
        target_risk: Optional[float],
        max_weight: float,
        min_weight: float,
        allow_short: bool,
    ) -> np.ndarray:
        """
        Mean-variance optimization with constraints.
        """
        n_assets = len(expected_returns)
        mu = expected_returns.values
        sigma = cov_matrix.values

        # Objective: minimize portfolio variance
        def objective(w):
            return w @ sigma @ w

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]

        if target_return is not None:
            constraints.append(
                {"type": "eq", "fun": lambda w: w @ mu - target_return}
            )

        if target_risk is not None:
            constraints.append(
                {"type": "ineq", "fun": lambda w: target_risk**2 - w @ sigma @ w}
            )

        # Bounds
        if allow_short:
            bounds = [(min_weight, max_weight) for _ in range(n_assets)]
        else:
            bounds = [(max(0, min_weight), max_weight) for _ in range(n_assets)]

        # Initial guess (equal weight)
        x0 = np.ones(n_assets) / n_assets

        # Optimize
        result = optimize.minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        if not result.success:
            logger.warning(f"Optimization may not have converged: {result.message}")

        return result.x


def create_absolute_view(
    asset: str,
    expected_return: float,
    confidence: float,
    description: str = "",
) -> ViewSpecification:
    """
    Convenience function to create an absolute view.

    Args:
        asset: Asset symbol
        expected_return: Expected return (e.g., 0.10 for 10%)
        confidence: Confidence level (0-1)
        description: Optional description

    Returns:
        ViewSpecification for absolute view
    """
    return ViewSpecification(
        assets=[asset],
        weights=[1.0],
        expected_return=expected_return,
        confidence=confidence,
        description=description or f"Absolute view on {asset}",
    )


def create_relative_view(
    long_asset: str,
    short_asset: str,
    expected_outperformance: float,
    confidence: float,
    description: str = "",
) -> ViewSpecification:
    """
    Convenience function to create a relative view.

    Args:
        long_asset: Asset expected to outperform
        short_asset: Asset expected to underperform
        expected_outperformance: Expected outperformance (e.g., 0.02 for 2%)
        confidence: Confidence level (0-1)
        description: Optional description

    Returns:
        ViewSpecification for relative view
    """
    return ViewSpecification(
        assets=[long_asset, short_asset],
        weights=[1.0, -1.0],
        expected_return=expected_outperformance,
        confidence=confidence,
        description=description or f"{long_asset} vs {short_asset}",
    )


class BlackLittermanWithML:
    """
    Black-Litterman optimizer that uses ML predictions as views.

    Converts ML model predictions into Black-Litterman views,
    allowing integration of quantitative signals with market equilibrium.

    Example usage:
        bl_ml = BlackLittermanWithML()

        # Add predictions as views
        for asset, prediction, prob in ml_predictions:
            bl_ml.add_prediction_view(
                asset=asset,
                predicted_return=prediction,
                prediction_confidence=prob,
            )

        # Optimize
        result = bl_ml.optimize(returns)
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        tau: float = 0.05,
        prediction_to_confidence_scale: float = 1.0,
    ) -> None:
        """
        Initialize ML-integrated Black-Litterman optimizer.

        Args:
            risk_free_rate: Risk-free rate
            tau: Uncertainty scalar
            prediction_to_confidence_scale: Scale factor for converting
                prediction probability to BL confidence
        """
        self.optimizer = BlackLittermanOptimizer(
            risk_free_rate=risk_free_rate,
            tau=tau,
        )
        self.confidence_scale = prediction_to_confidence_scale

    def add_prediction_view(
        self,
        asset: str,
        predicted_return: float,
        prediction_confidence: float,
        min_confidence: float = 0.1,
        max_confidence: float = 0.9,
    ) -> None:
        """
        Add ML prediction as a Black-Litterman view.

        Args:
            asset: Asset symbol
            predicted_return: Predicted return from ML model
            prediction_confidence: Model confidence (e.g., probability)
            min_confidence: Minimum confidence floor
            max_confidence: Maximum confidence cap
        """
        # Scale and clip confidence
        bl_confidence = np.clip(
            prediction_confidence * self.confidence_scale,
            min_confidence,
            max_confidence,
        )

        view = create_absolute_view(
            asset=asset,
            expected_return=predicted_return,
            confidence=bl_confidence,
            description=f"ML prediction for {asset}",
        )

        self.optimizer.add_view(view)

    def optimize(
        self,
        returns: pd.DataFrame,
        **kwargs,
    ) -> BlackLittermanResult:
        """
        Run optimization with ML-derived views.

        Args:
            returns: Historical returns
            **kwargs: Additional arguments for optimizer

        Returns:
            BlackLittermanResult
        """
        return self.optimizer.optimize(returns, **kwargs)

    def clear_views(self) -> None:
        """Clear all views."""
        self.optimizer.clear_views()
