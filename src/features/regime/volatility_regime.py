"""
Volatility Regime Detection using GARCH Models.

This module implements GARCH-based volatility regime detection, which is
superior to simple rolling volatility for:
1. Capturing volatility clustering
2. Asymmetric response to positive/negative shocks
3. Better forecasting of future volatility

Reference:
    Bollerslev, T. (1986) - "Generalized Autoregressive Conditional Heteroskedasticity"

Designed for JPMorgan-level institutional requirements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from arch import arch_model
    from arch.univariate import GARCH, EGARCH, GJR
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class VolatilityRegime(Enum):
    """Volatility regime classification."""

    VERY_LOW = "very_low"      # < 10th percentile
    LOW = "low"                # 10-25th percentile
    NORMAL = "normal"          # 25-75th percentile
    HIGH = "high"              # 75-90th percentile
    EXTREME = "extreme"        # > 90th percentile


@dataclass
class GARCHResult:
    """Container for GARCH model results."""

    conditional_volatility: pd.Series  # Time-varying volatility
    standardized_residuals: pd.Series  # Residuals / volatility
    forecast: pd.Series                # Forward volatility forecast
    params: Dict[str, float]           # Model parameters
    aic: float                         # Akaike Information Criterion
    bic: float                         # Bayesian Information Criterion
    log_likelihood: float
    converged: bool


@dataclass
class VolatilityRegimeResult:
    """Container for volatility regime results."""

    regimes: pd.Series                 # Regime at each timestamp
    volatility: pd.Series              # Conditional volatility
    percentile_ranks: pd.Series        # Volatility percentile ranks
    regime_thresholds: Dict[str, float]  # Threshold for each regime
    garch_params: Dict[str, float]


class GARCHModel:
    """
    GARCH volatility model with multiple specifications.

    Supports:
    - Standard GARCH(p,q)
    - EGARCH (exponential GARCH for asymmetry)
    - GJR-GARCH (threshold GARCH for leverage effect)

    Example usage:
        model = GARCHModel(model_type="gjr", p=1, q=1)
        result = model.fit(returns)

        # Get volatility forecast
        vol_forecast = model.forecast(horizon=5)
    """

    def __init__(
        self,
        model_type: str = "garch",
        p: int = 1,
        q: int = 1,
        dist: str = "t",
        rescale: bool = True,
    ) -> None:
        """
        Initialize GARCH model.

        Args:
            model_type: Model type ("garch", "egarch", "gjr")
            p: ARCH order
            q: GARCH order
            dist: Distribution ("normal", "t", "skewt")
            rescale: Whether to rescale returns for numerical stability
        """
        if not ARCH_AVAILABLE:
            raise ImportError(
                "arch package is required for GARCHModel. "
                "Install with: pip install arch"
            )

        self.model_type = model_type.lower()
        self.p = p
        self.q = q
        self.dist = dist
        self.rescale = rescale

        self._model = None
        self._result = None
        self._returns_scale = 1.0
        self._is_fitted = False

    def fit(
        self,
        returns: pd.Series,
        update_freq: int = 0,
    ) -> GARCHResult:
        """
        Fit GARCH model to returns.

        Args:
            returns: Return series (should be in percentage form)
            update_freq: Update frequency for online fitting (0 = batch)

        Returns:
            GARCHResult with conditional volatility and parameters
        """
        # Scale returns if needed
        if self.rescale:
            self._returns_scale = returns.std()
            scaled_returns = returns / self._returns_scale
        else:
            scaled_returns = returns

        # Specify volatility model
        if self.model_type == "garch":
            vol_model = "GARCH"
        elif self.model_type == "egarch":
            vol_model = "EGARCH"
        elif self.model_type == "gjr":
            vol_model = "GARCH"  # GJR is a GARCH variant
        else:
            vol_model = "GARCH"

        # Create and fit model
        self._model = arch_model(
            scaled_returns,
            mean="Constant",
            vol=vol_model,
            p=self.p,
            q=self.q,
            dist=self.dist,
        )

        if self.model_type == "gjr":
            self._model = arch_model(
                scaled_returns,
                mean="Constant",
                vol="GARCH",
                p=self.p,
                o=1,  # GJR term
                q=self.q,
                dist=self.dist,
            )

        self._result = self._model.fit(
            disp="off",
            update_freq=update_freq,
        )

        self._is_fitted = True

        # Extract results
        cond_vol = self._result.conditional_volatility * self._returns_scale
        std_resid = self._result.std_resid

        # Get parameters
        params = dict(self._result.params)

        logger.info(
            f"GARCH({self.p},{self.q}) fitted, AIC={self._result.aic:.2f}, "
            f"BIC={self._result.bic:.2f}"
        )

        return GARCHResult(
            conditional_volatility=pd.Series(cond_vol, index=returns.index),
            standardized_residuals=pd.Series(std_resid, index=returns.index),
            forecast=pd.Series(dtype=float),
            params=params,
            aic=self._result.aic,
            bic=self._result.bic,
            log_likelihood=self._result.loglikelihood,
            converged=self._result.convergence_flag == 0,
        )

    def forecast(
        self,
        horizon: int = 5,
        reindex: Optional[pd.Index] = None,
    ) -> pd.Series:
        """
        Forecast future volatility.

        Args:
            horizon: Forecast horizon in periods
            reindex: Optional index for forecast

        Returns:
            Volatility forecast series
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        forecast_result = self._result.forecast(horizon=horizon)
        var_forecast = forecast_result.variance.iloc[-1].values

        # Convert variance to volatility and rescale
        vol_forecast = np.sqrt(var_forecast) * self._returns_scale

        if reindex is not None:
            return pd.Series(vol_forecast, index=reindex[:horizon])

        return pd.Series(vol_forecast, name="volatility_forecast")

    def get_persistence(self) -> float:
        """
        Calculate volatility persistence (alpha + beta).

        Persistence < 1 indicates mean-reverting volatility.
        Persistence ≈ 1 indicates highly persistent volatility (IGARCH).
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted.")

        params = self._result.params
        alpha = params.get("alpha[1]", 0)
        beta = params.get("beta[1]", 0)
        gamma = params.get("gamma[1]", 0)  # GJR term

        return alpha + beta + 0.5 * gamma

    def get_half_life(self) -> float:
        """
        Calculate volatility half-life (time to mean-revert 50%).

        Returns:
            Half-life in periods
        """
        persistence = self.get_persistence()
        if persistence >= 1:
            return float('inf')
        return np.log(0.5) / np.log(persistence)


class VolatilityRegimeDetector:
    """
    Detects volatility regimes using GARCH conditional volatility.

    Classifies market conditions based on volatility percentile:
    - Very Low: < 10th percentile (complacency)
    - Low: 10-25th percentile
    - Normal: 25-75th percentile
    - High: 75-90th percentile
    - Extreme: > 90th percentile (crisis)

    Example usage:
        detector = VolatilityRegimeDetector()
        result = detector.detect(returns)

        # Adjust strategy based on regime
        if result.regimes.iloc[-1] == VolatilityRegime.EXTREME:
            reduce_leverage()
    """

    def __init__(
        self,
        garch_p: int = 1,
        garch_q: int = 1,
        lookback_percentile: int = 252,
        percentile_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize detector.

        Args:
            garch_p: GARCH p order
            garch_q: GARCH q order
            lookback_percentile: Lookback for percentile calculation
            percentile_thresholds: Custom percentile thresholds
        """
        self.garch_p = garch_p
        self.garch_q = garch_q
        self.lookback_percentile = lookback_percentile

        # Default thresholds
        self.percentile_thresholds = percentile_thresholds or {
            "very_low": 10,
            "low": 25,
            "high": 75,
            "extreme": 90,
        }

        self._garch_model = GARCHModel(p=garch_p, q=garch_q)
        self._is_fitted = False

    def detect(
        self,
        returns: pd.Series,
        use_rolling_percentile: bool = True,
    ) -> VolatilityRegimeResult:
        """
        Detect volatility regimes.

        Args:
            returns: Return series
            use_rolling_percentile: Use rolling window for percentiles

        Returns:
            VolatilityRegimeResult with regime classifications
        """
        # Fit GARCH model
        garch_result = self._garch_model.fit(returns)
        volatility = garch_result.conditional_volatility

        # Calculate percentile ranks
        if use_rolling_percentile:
            percentile_ranks = self._rolling_percentile(volatility)
        else:
            # Expanding percentile (more stable but slower to adapt)
            percentile_ranks = volatility.expanding().apply(
                lambda x: (x < x.iloc[-1]).sum() / len(x) * 100
            )

        # Classify regimes
        regimes = self._classify_regimes(percentile_ranks)

        self._is_fitted = True

        # Calculate threshold values
        thresholds = {
            name: volatility.quantile(pct / 100)
            for name, pct in self.percentile_thresholds.items()
        }

        return VolatilityRegimeResult(
            regimes=regimes,
            volatility=volatility,
            percentile_ranks=percentile_ranks,
            regime_thresholds=thresholds,
            garch_params=garch_result.params,
        )

    def _rolling_percentile(self, series: pd.Series) -> pd.Series:
        """Calculate rolling percentile rank."""

        def percentile_rank(x):
            return (x < x.iloc[-1]).sum() / (len(x) - 1) * 100 if len(x) > 1 else 50

        return series.rolling(
            window=self.lookback_percentile,
            min_periods=20,
        ).apply(percentile_rank)

    def _classify_regimes(self, percentile_ranks: pd.Series) -> pd.Series:
        """Classify into regimes based on percentile ranks."""
        thresholds = self.percentile_thresholds

        def classify(pct):
            if pd.isna(pct):
                return VolatilityRegime.NORMAL
            if pct < thresholds["very_low"]:
                return VolatilityRegime.VERY_LOW
            elif pct < thresholds["low"]:
                return VolatilityRegime.LOW
            elif pct < thresholds["high"]:
                return VolatilityRegime.NORMAL
            elif pct < thresholds["extreme"]:
                return VolatilityRegime.HIGH
            else:
                return VolatilityRegime.EXTREME

        return percentile_ranks.apply(classify)

    def get_regime_statistics(
        self,
        returns: pd.Series,
        result: VolatilityRegimeResult,
    ) -> pd.DataFrame:
        """
        Calculate statistics for each regime.

        Args:
            returns: Return series
            result: Regime detection result

        Returns:
            DataFrame with statistics per regime
        """
        stats = []

        for regime in VolatilityRegime:
            mask = result.regimes == regime
            if mask.sum() > 0:
                regime_returns = returns[mask]
                stats.append({
                    "regime": regime.value,
                    "frequency": mask.mean(),
                    "mean_return": regime_returns.mean(),
                    "std_return": regime_returns.std(),
                    "sharpe": regime_returns.mean() / regime_returns.std() * np.sqrt(252)
                              if regime_returns.std() > 0 else 0,
                    "mean_volatility": result.volatility[mask].mean(),
                    "count": mask.sum(),
                })

        return pd.DataFrame(stats).set_index("regime")


def fit_garch(
    returns: pd.Series,
    model_type: str = "garch",
    p: int = 1,
    q: int = 1,
) -> GARCHResult:
    """
    Convenience function to fit GARCH model.

    Args:
        returns: Return series
        model_type: Model type
        p: ARCH order
        q: GARCH order

    Returns:
        GARCHResult
    """
    model = GARCHModel(model_type=model_type, p=p, q=q)
    return model.fit(returns)


def detect_volatility_regime(returns: pd.Series) -> VolatilityRegimeResult:
    """
    Convenience function for volatility regime detection.

    Args:
        returns: Return series

    Returns:
        VolatilityRegimeResult
    """
    detector = VolatilityRegimeDetector()
    return detector.detect(returns)
