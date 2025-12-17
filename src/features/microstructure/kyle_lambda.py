"""
Kyle's Lambda (Price Impact) Estimation.

Kyle's Lambda measures the price impact of order flow, representing
how much prices move per unit of signed order flow.

High lambda = illiquid market (high price impact)
Low lambda = liquid market (low price impact)

Reference:
    Kyle, A.S. (1985) - "Continuous Auctions and Insider Trading"

Designed for JPMorgan-level institutional requirements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class KyleLambdaResult:
    """Container for Kyle's Lambda results."""

    lambda_value: float            # Estimated lambda
    lambda_series: pd.Series       # Rolling lambda
    r_squared: float               # Regression R-squared
    t_statistic: float             # T-statistic for lambda
    p_value: float                 # P-value
    price_impact_bps: pd.Series    # Price impact in basis points


class KyleLambda:
    """
    Kyle's Lambda estimator.

    Estimates price impact by regressing price changes on signed order flow:
        ΔP = λ * OrderFlow + ε

    where:
    - ΔP is the price change
    - OrderFlow is signed volume (buy - sell)
    - λ (lambda) is the price impact coefficient

    Example usage:
        estimator = KyleLambda(lookback=60)
        result = estimator.estimate(returns, volume, close)

        # Lambda represents price impact per unit volume
        print(f"Lambda: {result.lambda_value:.6f}")

        # High lambda indicates illiquid conditions
        if result.lambda_series.iloc[-1] > threshold:
            reduce_order_size()
    """

    def __init__(
        self,
        lookback: int = 60,
        min_periods: int = 30,
        volume_normalization: str = "adv",
    ) -> None:
        """
        Initialize Kyle Lambda estimator.

        Args:
            lookback: Rolling window for estimation
            min_periods: Minimum periods for regression
            volume_normalization: Normalize volume by "adv" or "none"
        """
        self.lookback = lookback
        self.min_periods = min_periods
        self.volume_normalization = volume_normalization

    def estimate(
        self,
        returns: pd.Series,
        volume: pd.Series,
        close: pd.Series,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
    ) -> KyleLambdaResult:
        """
        Estimate Kyle's Lambda.

        Args:
            returns: Return series
            volume: Volume series
            close: Close prices
            high: High prices (for signed volume)
            low: Low prices (for signed volume)

        Returns:
            KyleLambdaResult with lambda estimates
        """
        # Calculate signed order flow
        signed_flow = self._calculate_signed_flow(volume, close, high, low)

        # Normalize volume if requested
        if self.volume_normalization == "adv":
            adv = volume.rolling(20).mean()
            signed_flow = signed_flow / adv.replace(0, np.nan)

        # Full sample estimation
        full_result = self._estimate_lambda(returns, signed_flow)

        # Rolling estimation
        lambda_series = self._rolling_lambda(returns, signed_flow)

        # Calculate price impact in bps
        price_impact_bps = lambda_series * signed_flow.abs() * 10000

        return KyleLambdaResult(
            lambda_value=full_result["lambda"],
            lambda_series=lambda_series,
            r_squared=full_result["r_squared"],
            t_statistic=full_result["t_stat"],
            p_value=full_result["p_value"],
            price_impact_bps=price_impact_bps,
        )

    def _calculate_signed_flow(
        self,
        volume: pd.Series,
        close: pd.Series,
        high: Optional[pd.Series],
        low: Optional[pd.Series],
    ) -> pd.Series:
        """Calculate signed order flow."""
        if high is not None and low is not None:
            # Use bar position for sign
            bar_range = high - low
            normalized_position = (close - low) / bar_range.replace(0, np.nan)
            sign = 2 * normalized_position.fillna(0.5) - 1  # [-1, 1]
        else:
            # Use price change sign
            sign = np.sign(close.diff()).fillna(0)

        return volume * sign

    def _estimate_lambda(
        self,
        returns: pd.Series,
        signed_flow: pd.Series,
    ) -> Dict:
        """Estimate lambda using OLS regression."""
        # Align and clean data
        data = pd.DataFrame({"returns": returns, "flow": signed_flow}).dropna()

        if len(data) < self.min_periods:
            return {
                "lambda": np.nan,
                "r_squared": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
            }

        X = data["flow"].values
        y = data["returns"].values

        # Add constant for intercept
        X_with_const = np.column_stack([np.ones(len(X)), X])

        # OLS estimation
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)

            intercept = coeffs[0]
            lambda_val = coeffs[1]

            # Calculate R-squared
            y_pred = X_with_const @ coeffs
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            # Calculate standard errors and t-statistic
            n = len(y)
            k = 2  # Number of parameters
            mse = ss_res / (n - k) if n > k else np.nan
            var_coef = mse * np.linalg.inv(X_with_const.T @ X_with_const)
            se_lambda = np.sqrt(var_coef[1, 1]) if var_coef[1, 1] > 0 else np.nan
            t_stat = lambda_val / se_lambda if se_lambda and se_lambda > 0 else np.nan
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - k)) if not np.isnan(t_stat) else np.nan

            return {
                "lambda": lambda_val,
                "r_squared": r_squared,
                "t_stat": t_stat,
                "p_value": p_value,
            }

        except np.linalg.LinAlgError:
            return {
                "lambda": np.nan,
                "r_squared": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
            }

    def _rolling_lambda(
        self,
        returns: pd.Series,
        signed_flow: pd.Series,
    ) -> pd.Series:
        """Calculate rolling lambda estimates."""
        lambda_values = []
        index = []

        for i in range(self.lookback, len(returns)):
            window_returns = returns.iloc[i - self.lookback:i]
            window_flow = signed_flow.iloc[i - self.lookback:i]

            result = self._estimate_lambda(window_returns, window_flow)
            lambda_values.append(result["lambda"])
            index.append(returns.index[i])

        return pd.Series(lambda_values, index=index, name="kyle_lambda")


def calculate_kyle_lambda(
    returns: pd.Series,
    volume: pd.Series,
    close: pd.Series,
    lookback: int = 60,
) -> pd.Series:
    """
    Convenience function to calculate rolling Kyle's Lambda.

    Args:
        returns: Return series
        volume: Volume series
        close: Close prices
        lookback: Rolling window

    Returns:
        Rolling lambda series
    """
    estimator = KyleLambda(lookback=lookback)
    result = estimator.estimate(returns, volume, close)
    return result.lambda_series


def calculate_price_impact(
    lambda_series: pd.Series,
    order_size: float,
    adv: float,
) -> float:
    """
    Calculate expected price impact for an order.

    Args:
        lambda_series: Kyle's Lambda series
        order_size: Order size in shares
        adv: Average daily volume

    Returns:
        Expected price impact in basis points
    """
    current_lambda = lambda_series.iloc[-1]
    normalized_size = order_size / adv
    return current_lambda * normalized_size * 10000
