"""
Value at Risk (VaR) models for AlphaTrade system.

This module provides VaR calculations:
- Historical VaR
- Parametric VaR
- Monte Carlo VaR
- Conditional VaR (Expected Shortfall)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


class VaRCalculator:
    """
    Value at Risk calculator with multiple methods.

    Provides:
    - Historical simulation
    - Parametric (variance-covariance)
    - Monte Carlo simulation
    - Conditional VaR (Expected Shortfall)
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        horizon_days: int = 1,
        method: Literal["historical", "parametric", "monte_carlo"] = "historical",
    ) -> None:
        """
        Initialize the VaR calculator.

        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            horizon_days: Time horizon in days
            method: VaR calculation method
        """
        self.confidence_level = confidence_level
        self.horizon_days = horizon_days
        self.method = method

    def calculate_var(
        self,
        returns: pd.Series | pd.DataFrame,
        weights: pd.Series | None = None,
        position_value: float = 1.0,
    ) -> float:
        """
        Calculate Value at Risk.

        Args:
            returns: Historical returns (daily)
            weights: Portfolio weights (if DataFrame)
            position_value: Total position value

        Returns:
            VaR as a positive value
        """
        # Convert to portfolio returns if needed
        if isinstance(returns, pd.DataFrame):
            if weights is None:
                weights = pd.Series(1 / len(returns.columns), index=returns.columns)
            portfolio_returns = (returns * weights).sum(axis=1)
        else:
            portfolio_returns = returns

        # Scale to horizon
        portfolio_returns = portfolio_returns * np.sqrt(self.horizon_days)

        if self.method == "historical":
            var = self._historical_var(portfolio_returns)
        elif self.method == "parametric":
            var = self._parametric_var(portfolio_returns)
        elif self.method == "monte_carlo":
            var = self._monte_carlo_var(portfolio_returns)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return abs(var) * position_value

    def _historical_var(self, returns: pd.Series) -> float:
        """
        Calculate historical VaR.

        Args:
            returns: Historical returns

        Returns:
            VaR value
        """
        quantile = 1 - self.confidence_level
        return returns.quantile(quantile)

    def _parametric_var(self, returns: pd.Series) -> float:
        """
        Calculate parametric (Gaussian) VaR.

        Args:
            returns: Historical returns

        Returns:
            VaR value
        """
        mean = returns.mean()
        std = returns.std()

        # Z-score for confidence level
        z = stats.norm.ppf(1 - self.confidence_level)

        return mean + z * std

    def _monte_carlo_var(
        self,
        returns: pd.Series,
        n_simulations: int = 10000,
    ) -> float:
        """
        Calculate Monte Carlo VaR.

        Args:
            returns: Historical returns
            n_simulations: Number of simulations

        Returns:
            VaR value
        """
        mean = returns.mean()
        std = returns.std()

        # Generate simulated returns
        simulated = np.random.normal(mean, std, n_simulations)

        # Calculate VaR from simulation
        quantile = 1 - self.confidence_level
        return np.percentile(simulated, quantile * 100)

    def calculate_cvar(
        self,
        returns: pd.Series | pd.DataFrame,
        weights: pd.Series | None = None,
        position_value: float = 1.0,
    ) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).

        Args:
            returns: Historical returns
            weights: Portfolio weights (if DataFrame)
            position_value: Total position value

        Returns:
            CVaR as a positive value
        """
        # Convert to portfolio returns if needed
        if isinstance(returns, pd.DataFrame):
            if weights is None:
                weights = pd.Series(1 / len(returns.columns), index=returns.columns)
            portfolio_returns = (returns * weights).sum(axis=1)
        else:
            portfolio_returns = returns

        # Scale to horizon
        portfolio_returns = portfolio_returns * np.sqrt(self.horizon_days)

        # Calculate VaR threshold
        var_threshold = self._historical_var(portfolio_returns)

        # CVaR is mean of returns below VaR
        tail_returns = portfolio_returns[portfolio_returns <= var_threshold]

        if len(tail_returns) == 0:
            return abs(var_threshold) * position_value

        cvar = tail_returns.mean()
        return abs(cvar) * position_value

    def calculate_component_var(
        self,
        returns: pd.DataFrame,
        weights: pd.Series,
        position_value: float = 1.0,
    ) -> pd.Series:
        """
        Calculate component VaR for each asset.

        Args:
            returns: Asset returns DataFrame
            weights: Portfolio weights
            position_value: Total position value

        Returns:
            Component VaR series
        """
        portfolio_returns = (returns * weights).sum(axis=1)

        # Portfolio VaR
        portfolio_var = self.calculate_var(portfolio_returns, position_value=1.0)

        # Covariance
        cov = returns.cov()

        # Portfolio variance
        port_var = weights.values @ cov.values @ weights.values

        # Marginal VaR
        marginal_var = cov.values @ weights.values / np.sqrt(port_var + 1e-10)
        marginal_var = marginal_var * portfolio_var

        # Component VaR
        component_var = weights.values * marginal_var

        return pd.Series(component_var * position_value, index=weights.index)

    def rolling_var(
        self,
        returns: pd.Series,
        window: int = 252,
    ) -> pd.Series:
        """
        Calculate rolling VaR.

        Args:
            returns: Return series
            window: Rolling window size

        Returns:
            Rolling VaR series
        """
        if self.method == "historical":
            quantile = 1 - self.confidence_level
            rolling_var = returns.rolling(window=window).quantile(quantile)
        else:
            rolling_mean = returns.rolling(window=window).mean()
            rolling_std = returns.rolling(window=window).std()
            z = stats.norm.ppf(1 - self.confidence_level)
            rolling_var = rolling_mean + z * rolling_std

        return rolling_var.abs() * np.sqrt(self.horizon_days)


def calculate_var(
    returns: pd.Series | pd.DataFrame,
    confidence_level: float = 0.95,
    method: str = "historical",
    weights: pd.Series | None = None,
) -> float:
    """
    Convenience function to calculate VaR.

    Args:
        returns: Historical returns
        confidence_level: Confidence level
        method: VaR method
        weights: Portfolio weights

    Returns:
        VaR value
    """
    calculator = VaRCalculator(
        confidence_level=confidence_level,
        method=method,
    )
    return calculator.calculate_var(returns, weights)


def calculate_cvar(
    returns: pd.Series | pd.DataFrame,
    confidence_level: float = 0.95,
    weights: pd.Series | None = None,
) -> float:
    """
    Convenience function to calculate CVaR.

    Args:
        returns: Historical returns
        confidence_level: Confidence level
        weights: Portfolio weights

    Returns:
        CVaR value
    """
    calculator = VaRCalculator(confidence_level=confidence_level)
    return calculator.calculate_cvar(returns, weights)


class StressTest:
    """
    Stress testing framework for portfolio risk.

    Provides scenario analysis for extreme market conditions.
    """

    def __init__(
        self,
        scenarios: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """
        Initialize stress test.

        Args:
            scenarios: Predefined stress scenarios
        """
        self.scenarios = scenarios or self._default_scenarios()

    def _default_scenarios(self) -> dict[str, dict[str, float]]:
        """Default stress scenarios."""
        return {
            "market_crash_2008": {"market_return": -0.40, "vol_multiplier": 3.0},
            "flash_crash_2010": {"market_return": -0.10, "vol_multiplier": 5.0},
            "covid_2020": {"market_return": -0.34, "vol_multiplier": 4.0},
            "rate_shock": {"market_return": -0.15, "vol_multiplier": 2.0},
            "moderate_correction": {"market_return": -0.10, "vol_multiplier": 1.5},
        }

    def run_scenario(
        self,
        portfolio_value: float,
        weights: pd.Series,
        betas: pd.Series,
        scenario: str,
    ) -> float:
        """
        Run a stress scenario.

        Args:
            portfolio_value: Current portfolio value
            weights: Portfolio weights
            betas: Asset betas
            scenario: Scenario name

        Returns:
            Portfolio loss under scenario
        """
        if scenario not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario}")

        params = self.scenarios[scenario]
        market_return = params["market_return"]

        # Calculate asset returns using beta
        asset_returns = betas * market_return

        # Portfolio return
        portfolio_return = (weights * asset_returns).sum()

        # Loss
        loss = portfolio_value * abs(portfolio_return)

        return loss

    def run_all_scenarios(
        self,
        portfolio_value: float,
        weights: pd.Series,
        betas: pd.Series,
    ) -> pd.DataFrame:
        """
        Run all stress scenarios.

        Args:
            portfolio_value: Current portfolio value
            weights: Portfolio weights
            betas: Asset betas

        Returns:
            DataFrame with scenario results
        """
        results = []

        for scenario_name in self.scenarios:
            loss = self.run_scenario(
                portfolio_value, weights, betas, scenario_name
            )
            params = self.scenarios[scenario_name]

            results.append({
                "scenario": scenario_name,
                "market_return": params["market_return"],
                "portfolio_loss": loss,
                "loss_pct": loss / portfolio_value * 100,
            })

        return pd.DataFrame(results)

    def add_scenario(
        self,
        name: str,
        market_return: float,
        vol_multiplier: float = 1.0,
    ) -> None:
        """
        Add a custom scenario.

        Args:
            name: Scenario name
            market_return: Market return in scenario
            vol_multiplier: Volatility multiplier
        """
        self.scenarios[name] = {
            "market_return": market_return,
            "vol_multiplier": vol_multiplier,
        }
