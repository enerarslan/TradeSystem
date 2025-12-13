"""
Factor Risk Model (Barra-Lite)
Institutional-Grade Factor-Based Risk Decomposition

Implements a simplified Barra-style factor model to decompose portfolio risk
into systematic factor exposures (Momentum, Volatility, Liquidity, etc.)
and idiosyncratic risk.

Features:
- Multi-factor risk decomposition
- Factor exposure calculation
- Dollar/Beta neutrality constraints
- Regime-aware factor weights
- Portfolio optimization integration
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings

from ..utils.logger import get_logger


logger = get_logger(__name__)


class FactorType(Enum):
    """Standard risk factors"""
    MOMENTUM = "MOMENTUM"
    VOLATILITY = "VOLATILITY"
    LIQUIDITY = "LIQUIDITY"
    SIZE = "SIZE"
    VALUE = "VALUE"
    QUALITY = "QUALITY"
    MARKET = "MARKET"


class NeutralityConstraint(Enum):
    """Portfolio neutrality constraints"""
    DOLLAR_NEUTRAL = "DOLLAR_NEUTRAL"
    BETA_NEUTRAL = "BETA_NEUTRAL"
    FACTOR_NEUTRAL = "FACTOR_NEUTRAL"
    SECTOR_NEUTRAL = "SECTOR_NEUTRAL"


@dataclass
class FactorExposure:
    """
    Factor exposure for a single asset or portfolio.

    Attributes:
        factor: Factor type
        exposure: Raw exposure value
        z_score: Standardized exposure (z-score)
        contribution: Contribution to total risk
    """
    factor: FactorType
    exposure: float
    z_score: float
    contribution: float


@dataclass
class RiskDecomposition:
    """
    Complete risk decomposition of a portfolio.

    Attributes:
        total_risk: Total portfolio risk (annualized volatility)
        systematic_risk: Risk from factor exposures
        idiosyncratic_risk: Residual/specific risk
        factor_contributions: Risk contribution by factor
        factor_exposures: Exposure to each factor
        active_risk: Tracking error vs benchmark
    """
    total_risk: float
    systematic_risk: float
    idiosyncratic_risk: float
    factor_contributions: Dict[FactorType, float]
    factor_exposures: Dict[FactorType, FactorExposure]
    active_risk: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def systematic_ratio(self) -> float:
        """Ratio of systematic to total risk"""
        return self.systematic_risk / self.total_risk if self.total_risk > 0 else 0.0

    @property
    def diversification_ratio(self) -> float:
        """Measure of portfolio diversification"""
        if self.total_risk > 0:
            return 1 - (self.idiosyncratic_risk / self.total_risk)
        return 0.0


@dataclass
class PortfolioConstraint:
    """
    Constraint for portfolio optimization.

    Attributes:
        constraint_type: Type of neutrality constraint
        target_value: Target exposure (typically 0 for neutrality)
        tolerance: Allowed deviation from target
    """
    constraint_type: NeutralityConstraint
    target_value: float = 0.0
    tolerance: float = 0.05
    is_active: bool = True


class FactorCalculator:
    """
    Calculates factor exposures from price/fundamental data.

    Implements standard factor definitions:
    - Momentum: Past returns (12M-1M)
    - Volatility: Historical volatility
    - Liquidity: Average daily volume
    - Size: Market cap (or price proxy)
    - Value: Book-to-Market (simplified)
    - Quality: Profitability metrics (simplified)
    """

    def __init__(
        self,
        momentum_lookback: int = 252,
        volatility_lookback: int = 60,
        volume_lookback: int = 20
    ):
        """
        Initialize Factor Calculator.

        Args:
            momentum_lookback: Days for momentum calculation
            volatility_lookback: Days for volatility calculation
            volume_lookback: Days for volume averaging
        """
        self.momentum_lookback = momentum_lookback
        self.volatility_lookback = volatility_lookback
        self.volume_lookback = volume_lookback

    def calculate_momentum(self, prices: pd.Series) -> float:
        """
        Calculate momentum factor.

        12-month return minus 1-month return (avoiding reversal effect).
        """
        if len(prices) < self.momentum_lookback:
            return 0.0

        # Skip most recent month for momentum
        ret_12m = prices.iloc[-22] / prices.iloc[-self.momentum_lookback] - 1
        ret_1m = prices.iloc[-1] / prices.iloc[-22] - 1

        return ret_12m - ret_1m

    def calculate_volatility(self, prices: pd.Series) -> float:
        """
        Calculate volatility factor.

        Annualized historical volatility.
        """
        if len(prices) < self.volatility_lookback:
            return 0.0

        returns = prices.pct_change().dropna()
        vol = returns.iloc[-self.volatility_lookback:].std() * np.sqrt(252)

        return vol

    def calculate_liquidity(
        self,
        prices: pd.Series,
        volumes: pd.Series
    ) -> float:
        """
        Calculate liquidity factor.

        Average dollar volume (higher = more liquid).
        """
        if len(prices) < self.volume_lookback or len(volumes) < self.volume_lookback:
            return 0.0

        dollar_volume = prices * volumes
        avg_dollar_volume = dollar_volume.iloc[-self.volume_lookback:].mean()

        return np.log(avg_dollar_volume + 1)  # Log transform

    def calculate_size(self, prices: pd.Series) -> float:
        """
        Calculate size factor.

        Using price as proxy for market cap (simplified).
        """
        if len(prices) < 1:
            return 0.0

        return np.log(prices.iloc[-1] + 1)

    def calculate_all_factors(
        self,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None
    ) -> Dict[FactorType, float]:
        """
        Calculate all factor exposures for an asset.

        Args:
            prices: Price series
            volumes: Volume series (optional)

        Returns:
            Dictionary of factor exposures
        """
        factors = {
            FactorType.MOMENTUM: self.calculate_momentum(prices),
            FactorType.VOLATILITY: self.calculate_volatility(prices),
            FactorType.SIZE: self.calculate_size(prices),
        }

        if volumes is not None:
            factors[FactorType.LIQUIDITY] = self.calculate_liquidity(prices, volumes)
        else:
            factors[FactorType.LIQUIDITY] = 0.0

        return factors


class BarraLiteModel:
    """
    Simplified Barra-style Factor Risk Model.

    Decomposes portfolio risk into systematic factor risk and
    idiosyncratic risk. Supports portfolio optimization with
    neutrality constraints.

    Usage:
        model = BarraLiteModel()
        model.estimate_factor_covariance(returns_df)
        decomposition = model.decompose_risk(portfolio_weights, asset_factors)
    """

    def __init__(
        self,
        factors: Optional[List[FactorType]] = None,
        half_life_days: int = 60,
        min_history_days: int = 252
    ):
        """
        Initialize Barra-Lite Model.

        Args:
            factors: List of factors to use
            half_life_days: Half-life for exponential weighting
            min_history_days: Minimum days of history required
        """
        self.factors = factors or [
            FactorType.MOMENTUM,
            FactorType.VOLATILITY,
            FactorType.LIQUIDITY,
            FactorType.SIZE
        ]
        self.half_life_days = half_life_days
        self.min_history_days = min_history_days

        self.factor_calculator = FactorCalculator()

        # Model state
        self._factor_covariance: Optional[np.ndarray] = None
        self._factor_means: Optional[np.ndarray] = None
        self._factor_stds: Optional[np.ndarray] = None
        self._residual_variances: Dict[str, float] = {}

        logger.info(
            f"BarraLiteModel initialized with factors: "
            f"{[f.value for f in self.factors]}"
        )

    def compute_asset_factors(
        self,
        prices: Dict[str, pd.Series],
        volumes: Optional[Dict[str, pd.Series]] = None
    ) -> pd.DataFrame:
        """
        Compute factor exposures for all assets.

        Args:
            prices: Dictionary of price series by symbol
            volumes: Dictionary of volume series by symbol

        Returns:
            DataFrame with factor exposures (assets x factors)
        """
        factor_data = []

        for symbol, price_series in prices.items():
            vol_series = volumes.get(symbol) if volumes else None

            factors = self.factor_calculator.calculate_all_factors(
                price_series, vol_series
            )

            row = {"symbol": symbol}
            row.update({f.value: factors[f] for f in self.factors if f in factors})
            factor_data.append(row)

        df = pd.DataFrame(factor_data).set_index("symbol")

        # Standardize to z-scores
        df = (df - df.mean()) / (df.std() + 1e-10)

        return df

    def estimate_factor_covariance(
        self,
        returns: pd.DataFrame,
        factor_exposures: pd.DataFrame
    ) -> np.ndarray:
        """
        Estimate factor covariance matrix using cross-sectional regression.

        Args:
            returns: Asset returns DataFrame (time x assets)
            factor_exposures: Factor exposures DataFrame (assets x factors)

        Returns:
            Factor covariance matrix
        """
        if len(returns) < self.min_history_days:
            logger.warning("Insufficient history for factor covariance estimation")
            # Return identity matrix scaled by average variance
            n_factors = len(self.factors)
            avg_var = returns.var().mean()
            return np.eye(n_factors) * avg_var

        # Align data
        common_assets = list(set(returns.columns) & set(factor_exposures.index))
        returns = returns[common_assets]
        factors = factor_exposures.loc[common_assets]

        # Exponential weights for recent observations
        decay = 0.5 ** (1 / self.half_life_days)
        weights = np.array([decay ** i for i in range(len(returns))][::-1])
        weights = weights / weights.sum()

        # Cross-sectional regression for each time period
        factor_returns = []

        for t in range(len(returns)):
            y = returns.iloc[t].values
            X = factors.values

            # Skip if too many missing values
            valid = ~np.isnan(y)
            if valid.sum() < len(self.factors) + 1:
                continue

            # Weighted least squares
            try:
                W = np.diag(np.sqrt(np.full(valid.sum(), weights[t])))
                X_valid = X[valid]
                y_valid = y[valid]

                # Add intercept
                X_aug = np.column_stack([np.ones(len(X_valid)), X_valid])

                # Solve: (X'WX)^-1 X'Wy
                XtWX = X_aug.T @ W @ W @ X_aug
                XtWy = X_aug.T @ W @ W @ y_valid

                betas = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]
                factor_returns.append(betas[1:])  # Exclude intercept

            except Exception as e:
                logger.debug(f"Regression failed at t={t}: {e}")
                continue

        if len(factor_returns) < 30:
            logger.warning("Insufficient factor returns for covariance estimation")
            n_factors = len(self.factors)
            return np.eye(n_factors) * 0.01

        factor_returns = np.array(factor_returns)

        # Compute covariance with exponential weighting
        weights_cov = weights[-len(factor_returns):]
        weights_cov = weights_cov / weights_cov.sum()

        # Weighted covariance
        factor_means = np.average(factor_returns, axis=0, weights=weights_cov)
        centered = factor_returns - factor_means
        factor_cov = np.average(
            centered[:, :, np.newaxis] * centered[:, np.newaxis, :],
            axis=0,
            weights=weights_cov
        )

        # Annualize
        factor_cov = factor_cov * 252

        self._factor_covariance = factor_cov
        self._factor_means = factor_means * 252
        self._factor_stds = np.sqrt(np.diag(factor_cov))

        logger.info(
            f"Factor covariance estimated. "
            f"Avg correlation: {self._get_avg_correlation():.2f}"
        )

        return factor_cov

    def _get_avg_correlation(self) -> float:
        """Get average off-diagonal correlation"""
        if self._factor_covariance is None:
            return 0.0

        cov = self._factor_covariance
        n = len(cov)

        # Convert to correlation
        stds = np.sqrt(np.diag(cov))
        corr = cov / (np.outer(stds, stds) + 1e-10)

        # Average off-diagonal
        mask = ~np.eye(n, dtype=bool)
        return np.mean(np.abs(corr[mask]))

    def decompose_risk(
        self,
        weights: Dict[str, float],
        factor_exposures: pd.DataFrame,
        residual_variances: Optional[Dict[str, float]] = None
    ) -> RiskDecomposition:
        """
        Decompose portfolio risk into factor and idiosyncratic components.

        Args:
            weights: Portfolio weights by symbol
            factor_exposures: Factor exposures DataFrame
            residual_variances: Asset-specific variances (optional)

        Returns:
            RiskDecomposition object
        """
        if self._factor_covariance is None:
            raise ValueError("Factor covariance not estimated. Call estimate_factor_covariance first.")

        # Align weights with factor exposures
        symbols = [s for s in weights if s in factor_exposures.index]
        w = np.array([weights[s] for s in symbols])
        F = factor_exposures.loc[symbols].values

        # Portfolio factor exposures
        portfolio_factors = w @ F

        # Systematic risk: f' * Σ_F * f
        systematic_var = portfolio_factors @ self._factor_covariance @ portfolio_factors

        # Idiosyncratic risk
        if residual_variances:
            specific_vars = np.array([
                residual_variances.get(s, 0.01) for s in symbols
            ])
        else:
            specific_vars = np.full(len(symbols), 0.01)  # Default 10% annual vol

        idiosyncratic_var = np.sum((w ** 2) * specific_vars)

        # Total risk
        total_var = systematic_var + idiosyncratic_var
        total_risk = np.sqrt(total_var)
        systematic_risk = np.sqrt(systematic_var)
        idiosyncratic_risk = np.sqrt(idiosyncratic_var)

        # Factor contributions
        factor_contributions = {}
        factor_exposure_objects = {}

        for i, factor in enumerate(self.factors):
            exposure = portfolio_factors[i]

            # Marginal contribution to risk
            if total_risk > 0:
                marginal = (
                    exposure * (self._factor_covariance[i, :] @ portfolio_factors)
                ) / total_var
                contribution = marginal * total_risk
            else:
                contribution = 0.0

            z_score = exposure  # Already standardized

            factor_contributions[factor] = contribution
            factor_exposure_objects[factor] = FactorExposure(
                factor=factor,
                exposure=exposure,
                z_score=z_score,
                contribution=contribution
            )

        return RiskDecomposition(
            total_risk=total_risk,
            systematic_risk=systematic_risk,
            idiosyncratic_risk=idiosyncratic_risk,
            factor_contributions=factor_contributions,
            factor_exposures=factor_exposure_objects
        )

    def check_neutrality(
        self,
        weights: Dict[str, float],
        factor_exposures: pd.DataFrame,
        constraint: PortfolioConstraint
    ) -> Tuple[bool, float]:
        """
        Check if portfolio satisfies a neutrality constraint.

        Args:
            weights: Portfolio weights
            factor_exposures: Factor exposures
            constraint: Neutrality constraint to check

        Returns:
            Tuple of (satisfies_constraint, actual_exposure)
        """
        symbols = [s for s in weights if s in factor_exposures.index]
        w = np.array([weights[s] for s in symbols])

        if constraint.constraint_type == NeutralityConstraint.DOLLAR_NEUTRAL:
            # Sum of weights should be ~0
            exposure = np.sum(w)

        elif constraint.constraint_type == NeutralityConstraint.BETA_NEUTRAL:
            # Beta-weighted sum should be ~0
            # Using market factor as beta proxy
            if FactorType.MARKET in self.factors:
                market_idx = self.factors.index(FactorType.MARKET)
            else:
                # Use first factor as proxy
                market_idx = 0

            F = factor_exposures.loc[symbols].values
            exposure = w @ F[:, market_idx]

        elif constraint.constraint_type == NeutralityConstraint.FACTOR_NEUTRAL:
            # All factor exposures should be ~0
            F = factor_exposures.loc[symbols].values
            portfolio_factors = w @ F
            exposure = np.max(np.abs(portfolio_factors))

        else:
            exposure = 0.0

        satisfies = abs(exposure - constraint.target_value) <= constraint.tolerance

        return satisfies, exposure

    def suggest_hedge(
        self,
        weights: Dict[str, float],
        factor_exposures: pd.DataFrame,
        target_factor: FactorType,
        target_exposure: float = 0.0
    ) -> Dict[str, float]:
        """
        Suggest weight adjustments to achieve target factor exposure.

        Args:
            weights: Current portfolio weights
            factor_exposures: Factor exposures
            target_factor: Factor to hedge
            target_exposure: Target exposure level

        Returns:
            Suggested weight adjustments
        """
        symbols = [s for s in weights if s in factor_exposures.index]
        w = np.array([weights[s] for s in symbols])
        F = factor_exposures.loc[symbols].values

        factor_idx = self.factors.index(target_factor)
        current_exposure = w @ F[:, factor_idx]

        exposure_gap = target_exposure - current_exposure

        if abs(exposure_gap) < 0.01:
            return {}

        # Find assets with highest/lowest factor exposure
        asset_exposures = F[:, factor_idx]

        adjustments = {}

        if exposure_gap > 0:
            # Need to increase exposure - buy high-factor assets
            high_exposure_idx = np.argsort(asset_exposures)[-5:]
            for idx in high_exposure_idx:
                symbol = symbols[idx]
                adjustments[symbol] = exposure_gap * 0.1  # Scale down

        else:
            # Need to decrease exposure - sell high-factor or buy low-factor assets
            low_exposure_idx = np.argsort(asset_exposures)[:5]
            for idx in low_exposure_idx:
                symbol = symbols[idx]
                adjustments[symbol] = abs(exposure_gap) * 0.1

        return adjustments

    def get_factor_summary(self) -> pd.DataFrame:
        """Get summary statistics for all factors"""
        if self._factor_covariance is None:
            return pd.DataFrame()

        data = []
        for i, factor in enumerate(self.factors):
            data.append({
                "factor": factor.value,
                "mean_return": self._factor_means[i] if self._factor_means is not None else 0,
                "volatility": self._factor_stds[i] if self._factor_stds is not None else 0,
                "sharpe": (
                    self._factor_means[i] / self._factor_stds[i]
                    if self._factor_stds is not None and self._factor_stds[i] > 0
                    else 0
                )
            })

        return pd.DataFrame(data)
