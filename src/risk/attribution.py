"""
Risk Attribution Analysis Module.

This module provides institutional-grade risk and return attribution:
- Factor attribution (market, size, value, momentum)
- Sector attribution
- Active vs passive decomposition
- Tracking error analysis

JPMorgan-level requirements:
- Brinson attribution model
- Multi-factor risk model integration
- Daily attribution granularity
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class FactorExposure:
    """Exposure to a single factor."""
    factor_name: str
    beta: float
    t_statistic: float
    p_value: float
    contribution_to_return: float
    contribution_to_risk: float


@dataclass
class SectorAttribution:
    """Attribution for a single sector."""
    sector: str
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total_effect: float
    portfolio_weight: float
    benchmark_weight: float


@dataclass
class RiskAttribution:
    """Complete risk attribution breakdown."""
    total_return: float
    benchmark_return: float
    active_return: float

    # Factor attribution
    factor_exposures: List[FactorExposure] = field(default_factory=list)
    systematic_return: float = 0.0
    idiosyncratic_return: float = 0.0
    r_squared: float = 0.0

    # Sector attribution
    sector_attribution: List[SectorAttribution] = field(default_factory=list)
    allocation_effect_total: float = 0.0
    selection_effect_total: float = 0.0

    # Risk decomposition
    total_risk: float = 0.0
    systematic_risk: float = 0.0
    specific_risk: float = 0.0
    tracking_error: float = 0.0


class FactorModel:
    """
    Multi-factor risk model.

    Implements a basic factor model with:
    - Market factor (beta)
    - Size factor (SMB proxy)
    - Value factor (HML proxy)
    - Momentum factor

    In production, this would connect to a proper risk model
    (Barra, Axioma, Bloomberg).
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        periods_per_year: int = 252 * 26,
    ):
        """
        Initialize factor model.

        Args:
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
        """
        self.rf = risk_free_rate
        self.periods = periods_per_year

    def estimate_exposures(
        self,
        portfolio_returns: pd.Series,
        factor_returns: Dict[str, pd.Series],
    ) -> List[FactorExposure]:
        """
        Estimate factor exposures using OLS regression.

        Args:
            portfolio_returns: Portfolio return series
            factor_returns: Dict of factor name -> return series

        Returns:
            List of factor exposures
        """
        # Align all series
        common_idx = portfolio_returns.dropna().index
        for factor in factor_returns.values():
            common_idx = common_idx.intersection(factor.dropna().index)

        if len(common_idx) < 30:
            logger.warning("Insufficient data for factor regression")
            return []

        y = portfolio_returns.loc[common_idx].values
        X = pd.DataFrame({k: v.loc[common_idx] for k, v in factor_returns.items()})

        # Add constant
        X_with_const = np.column_stack([np.ones(len(X)), X.values])

        # OLS regression
        try:
            # (X'X)^-1 X'y
            beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]

            # Calculate residuals and statistics
            y_pred = X_with_const @ beta
            residuals = y - y_pred
            n = len(y)
            k = X_with_const.shape[1]

            # Standard errors
            mse = np.sum(residuals**2) / (n - k)
            var_beta = mse * np.linalg.inv(X_with_const.T @ X_with_const)
            se = np.sqrt(np.diag(var_beta))

            # R-squared
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - y.mean())**2)
            r_squared = 1 - ss_res / ss_tot

            # Create factor exposures
            exposures = []
            factor_names = list(factor_returns.keys())

            for i, factor_name in enumerate(factor_names):
                factor_idx = i + 1  # Skip intercept
                factor_beta = beta[factor_idx]
                factor_se = se[factor_idx]
                t_stat = factor_beta / factor_se if factor_se > 0 else 0
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - k))

                # Contribution to return
                factor_contrib = factor_beta * factor_returns[factor_name].loc[common_idx].mean() * self.periods

                # Contribution to risk
                factor_var = factor_returns[factor_name].loc[common_idx].var()
                risk_contrib = abs(factor_beta) * np.sqrt(factor_var) * np.sqrt(self.periods)

                exposures.append(FactorExposure(
                    factor_name=factor_name,
                    beta=factor_beta,
                    t_statistic=t_stat,
                    p_value=p_val,
                    contribution_to_return=factor_contrib,
                    contribution_to_risk=risk_contrib,
                ))

            return exposures

        except Exception as e:
            logger.error(f"Factor regression failed: {e}")
            return []

    def decompose_risk(
        self,
        portfolio_returns: pd.Series,
        factor_returns: Dict[str, pd.Series],
    ) -> Tuple[float, float, float]:
        """
        Decompose total risk into systematic and specific components.

        Returns:
            Tuple of (total_risk, systematic_risk, specific_risk)
        """
        total_var = portfolio_returns.var() * self.periods
        total_risk = np.sqrt(total_var)

        exposures = self.estimate_exposures(portfolio_returns, factor_returns)
        if not exposures:
            return total_risk, 0.0, total_risk

        # Systematic variance from factors
        systematic_var = sum(
            e.contribution_to_risk**2 for e in exposures
        )
        systematic_risk = np.sqrt(systematic_var)

        # Specific risk is the residual
        specific_var = max(0, total_var - systematic_var)
        specific_risk = np.sqrt(specific_var)

        return total_risk, systematic_risk, specific_risk


class BrinsonAttribution:
    """
    Brinson Attribution Model for sector/asset allocation analysis.

    Decomposes active return into:
    - Allocation Effect: Return from over/underweighting sectors
    - Selection Effect: Return from stock selection within sectors
    - Interaction Effect: Combined effect
    """

    def calculate_attribution(
        self,
        portfolio_weights: pd.Series,  # Symbol -> weight
        benchmark_weights: pd.Series,  # Symbol -> weight
        portfolio_returns: pd.Series,  # Symbol -> return
        benchmark_returns: pd.Series,  # Symbol -> return
        symbol_sectors: Dict[str, str],  # Symbol -> sector
    ) -> List[SectorAttribution]:
        """
        Calculate Brinson attribution by sector.

        Args:
            portfolio_weights: Portfolio weights by symbol
            benchmark_weights: Benchmark weights by symbol
            portfolio_returns: Portfolio returns by symbol
            benchmark_returns: Benchmark returns by symbol
            symbol_sectors: Mapping of symbols to sectors

        Returns:
            List of sector attributions
        """
        # Get unique sectors
        sectors = set(symbol_sectors.values())
        attributions = []

        # Calculate overall benchmark return
        bm_total_return = (benchmark_weights * benchmark_returns).sum()

        for sector in sectors:
            # Get symbols in this sector
            sector_symbols = [s for s, sec in symbol_sectors.items() if sec == sector]

            # Portfolio sector weight and return
            port_weight = portfolio_weights.loc[sector_symbols].sum() if any(
                s in portfolio_weights.index for s in sector_symbols
            ) else 0

            port_symbols_in_sector = [s for s in sector_symbols if s in portfolio_returns.index]
            if port_symbols_in_sector and port_weight > 0:
                port_return = (
                    portfolio_weights.loc[port_symbols_in_sector] *
                    portfolio_returns.loc[port_symbols_in_sector]
                ).sum() / port_weight
            else:
                port_return = 0

            # Benchmark sector weight and return
            bm_weight = benchmark_weights.loc[sector_symbols].sum() if any(
                s in benchmark_weights.index for s in sector_symbols
            ) else 0

            bm_symbols_in_sector = [s for s in sector_symbols if s in benchmark_returns.index]
            if bm_symbols_in_sector and bm_weight > 0:
                bm_return = (
                    benchmark_weights.loc[bm_symbols_in_sector] *
                    benchmark_returns.loc[bm_symbols_in_sector]
                ).sum() / bm_weight
            else:
                bm_return = 0

            # Brinson attribution
            allocation_effect = (port_weight - bm_weight) * (bm_return - bm_total_return)
            selection_effect = bm_weight * (port_return - bm_return)
            interaction_effect = (port_weight - bm_weight) * (port_return - bm_return)
            total_effect = allocation_effect + selection_effect + interaction_effect

            attributions.append(SectorAttribution(
                sector=sector,
                allocation_effect=allocation_effect,
                selection_effect=selection_effect,
                interaction_effect=interaction_effect,
                total_effect=total_effect,
                portfolio_weight=port_weight,
                benchmark_weight=bm_weight,
            ))

        return attributions


def run_risk_attribution(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    factor_returns: Optional[Dict[str, pd.Series]] = None,
    portfolio_positions: Optional[pd.DataFrame] = None,
    benchmark_positions: Optional[pd.DataFrame] = None,
    symbol_sectors: Optional[Dict[str, str]] = None,
) -> RiskAttribution:
    """
    Run comprehensive risk attribution analysis.

    Args:
        portfolio_returns: Portfolio return series
        benchmark_returns: Benchmark return series
        factor_returns: Optional factor return series
        portfolio_positions: Optional position weights over time
        benchmark_positions: Optional benchmark weights
        symbol_sectors: Symbol to sector mapping

    Returns:
        Complete RiskAttribution
    """
    logger.info("Running risk attribution analysis...")

    # Basic metrics
    total_return = (1 + portfolio_returns).prod() - 1
    bm_return = (1 + benchmark_returns).prod() - 1
    active_return = total_return - bm_return

    result = RiskAttribution(
        total_return=total_return,
        benchmark_return=bm_return,
        active_return=active_return,
    )

    # Factor attribution
    if factor_returns:
        model = FactorModel()
        result.factor_exposures = model.estimate_exposures(portfolio_returns, factor_returns)

        total_risk, sys_risk, spec_risk = model.decompose_risk(
            portfolio_returns, factor_returns
        )
        result.total_risk = total_risk
        result.systematic_risk = sys_risk
        result.specific_risk = spec_risk

        # Calculate R-squared
        if total_risk > 0:
            result.r_squared = sys_risk**2 / total_risk**2

        # Systematic vs idiosyncratic return
        result.systematic_return = sum(e.contribution_to_return for e in result.factor_exposures)
        result.idiosyncratic_return = total_return - result.systematic_return

    # Tracking error
    active_returns = portfolio_returns - benchmark_returns
    result.tracking_error = active_returns.std() * np.sqrt(252 * 26)

    # Sector attribution (if data available)
    if portfolio_positions is not None and symbol_sectors:
        brinson = BrinsonAttribution()
        # Use average weights for simplicity
        port_weights = portfolio_positions.mean()
        if benchmark_positions is not None:
            bm_weights = benchmark_positions.mean()
        else:
            bm_weights = pd.Series(1.0 / len(port_weights), index=port_weights.index)

        # Get returns by symbol (would need actual data)
        # Placeholder implementation
        result.sector_attribution = []

    return result
