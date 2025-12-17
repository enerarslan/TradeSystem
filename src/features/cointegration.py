"""
Statistical Arbitrage Features - Cointegration and Mean Reversion.

This module provides tools for:
- Engle-Granger and Johansen cointegration tests
- Optimal hedge ratio estimation
- Spread construction and z-score calculation
- Ornstein-Uhlenbeck parameter estimation
- Half-life computation

These features are essential for pairs trading and statistical arbitrage strategies.

Reference:
    - Engle, R.F. and Granger, C.W.J. (1987). "Co-Integration and Error Correction"
    - Hamilton, J.D. (1994). "Time Series Analysis"
    - Avellaneda, M. and Lee, J.H. (2010). "Statistical Arbitrage in the US Equities Market"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller, coint
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from scipy import stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class CointegrationResult:
    """Result of cointegration test."""
    series1_name: str
    series2_name: str
    test_statistic: float
    p_value: float
    is_cointegrated: bool
    hedge_ratio: float
    half_life: Optional[float] = None
    critical_values: Optional[Dict[str, float]] = None


@dataclass
class OUParameters:
    """Ornstein-Uhlenbeck process parameters."""
    theta: float      # Mean reversion speed
    mu: float         # Long-term mean
    sigma: float      # Volatility
    half_life: float  # Mean reversion half-life in bars


@dataclass
class JohansenResult:
    """Result of Johansen cointegration test."""
    n_cointegrating_relations: int
    trace_statistics: np.ndarray
    trace_critical_values: np.ndarray
    max_eigen_statistics: np.ndarray
    max_eigen_critical_values: np.ndarray
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray


class CointegrationAnalyzer:
    """
    Analyzer for cointegration testing and pairs trading.

    Provides methods for:
    - Testing cointegration between pairs
    - Finding cointegrated pairs in a universe
    - Calculating optimal hedge ratios
    - Estimating mean reversion parameters

    Example:
        analyzer = CointegrationAnalyzer()

        # Test single pair
        result = analyzer.engle_granger_test(price1, price2)

        # Find all cointegrated pairs
        pairs = analyzer.find_cointegrated_pairs(returns_df)

        # Calculate spread
        spread = analyzer.calculate_spread(price1, price2, result.hedge_ratio)
    """

    def __init__(
        self,
        significance_level: float = 0.05,
    ):
        if not STATSMODELS_AVAILABLE:
            raise ImportError(
                "statsmodels required for CointegrationAnalyzer. "
                "Install with: pip install statsmodels"
            )

        self.significance_level = significance_level

    def engle_granger_test(
        self,
        series1: Union[pd.Series, np.ndarray],
        series2: Union[pd.Series, np.ndarray],
        trend: str = 'c',
    ) -> CointegrationResult:
        """
        Engle-Granger two-step cointegration test.

        Step 1: Regress series1 on series2 to get residuals
        Step 2: Test residuals for stationarity (unit root)

        Args:
            series1: First price series (y in regression)
            series2: Second price series (x in regression)
            trend: 'c' (constant), 'ct' (constant+trend), 'n' (none)

        Returns:
            CointegrationResult with test statistics and hedge ratio
        """
        # Get names
        name1 = series1.name if hasattr(series1, 'name') else 'series1'
        name2 = series2.name if hasattr(series2, 'name') else 'series2'

        # Convert to arrays
        y = np.asarray(series1)
        x = np.asarray(series2)

        # Remove NaN
        valid = ~(np.isnan(y) | np.isnan(x))
        y = y[valid]
        x = x[valid]

        # Use statsmodels coint function
        test_stat, p_value, critical_values = coint(y, x, trend=trend)

        # Calculate hedge ratio using OLS
        X = np.column_stack([np.ones(len(x)), x])
        model = OLS(y, X).fit()
        hedge_ratio = model.params[1]

        # Calculate spread and half-life
        spread = y - hedge_ratio * x
        try:
            half_life = self.calculate_half_life(spread)
        except Exception:
            half_life = None

        return CointegrationResult(
            series1_name=str(name1),
            series2_name=str(name2),
            test_statistic=test_stat,
            p_value=p_value,
            is_cointegrated=p_value < self.significance_level,
            hedge_ratio=hedge_ratio,
            half_life=half_life,
            critical_values={
                '1%': critical_values[0],
                '5%': critical_values[1],
                '10%': critical_values[2],
            }
        )

    def johansen_test(
        self,
        data: pd.DataFrame,
        det_order: int = 0,
        k_ar_diff: int = 1,
    ) -> JohansenResult:
        """
        Johansen cointegration test for multiple time series.

        Tests for multiple cointegrating relationships between
        more than two time series.

        Args:
            data: DataFrame with multiple price series
            det_order: Deterministic order (-1, 0, 1)
                -1: no deterministic terms
                 0: constant in cointegrating relation
                 1: linear trend in cointegrating relation
            k_ar_diff: Number of lagged differences in the model

        Returns:
            JohansenResult with test statistics and eigenvectors
        """
        # Remove NaN
        clean_data = data.dropna()

        # Johansen test
        result = coint_johansen(clean_data.values, det_order, k_ar_diff)

        # Count cointegrating relations using trace statistic
        n_coint = 0
        for i, (stat, cv) in enumerate(zip(
            result.trace_stat,
            result.trace_stat_crit_vals[:, 1]  # 5% critical value
        )):
            if stat > cv:
                n_coint += 1
            else:
                break

        return JohansenResult(
            n_cointegrating_relations=n_coint,
            trace_statistics=result.trace_stat,
            trace_critical_values=result.trace_stat_crit_vals,
            max_eigen_statistics=result.max_eig_stat,
            max_eigen_critical_values=result.max_eig_stat_crit_vals,
            eigenvectors=result.evec,
            eigenvalues=result.eig,
        )

    def find_cointegrated_pairs(
        self,
        price_data: pd.DataFrame,
        p_threshold: float = None,
        min_half_life: float = 1,
        max_half_life: float = 252,
    ) -> List[CointegrationResult]:
        """
        Find all cointegrated pairs in a universe of assets.

        Args:
            price_data: DataFrame with price series (columns = assets)
            p_threshold: P-value threshold (uses instance default if None)
            min_half_life: Minimum acceptable half-life
            max_half_life: Maximum acceptable half-life

        Returns:
            List of CointegrationResult for cointegrated pairs
        """
        if p_threshold is None:
            p_threshold = self.significance_level

        symbols = price_data.columns.tolist()
        n_symbols = len(symbols)
        results = []

        logger.info(f"Testing {n_symbols * (n_symbols - 1) // 2} pairs...")

        for i in range(n_symbols):
            for j in range(i + 1, n_symbols):
                sym1, sym2 = symbols[i], symbols[j]

                try:
                    result = self.engle_granger_test(
                        price_data[sym1],
                        price_data[sym2]
                    )

                    if result.is_cointegrated:
                        # Check half-life bounds
                        if result.half_life is not None:
                            if min_half_life <= result.half_life <= max_half_life:
                                results.append(result)
                        else:
                            results.append(result)

                except Exception as e:
                    logger.debug(f"Error testing {sym1}-{sym2}: {e}")
                    continue

        # Sort by p-value
        results.sort(key=lambda x: x.p_value)

        logger.info(f"Found {len(results)} cointegrated pairs")

        return results

    def calculate_spread(
        self,
        series1: Union[pd.Series, np.ndarray],
        series2: Union[pd.Series, np.ndarray],
        hedge_ratio: Optional[float] = None,
        method: str = 'ols',
    ) -> pd.Series:
        """
        Calculate spread between two cointegrated series.

        spread = series1 - hedge_ratio * series2

        Args:
            series1: First price series
            series2: Second price series
            hedge_ratio: Pre-computed hedge ratio (None = estimate)
            method: Hedge ratio method ('ols', 'tls', 'kalman')

        Returns:
            Spread series
        """
        y = np.asarray(series1)
        x = np.asarray(series2)

        # Estimate hedge ratio if not provided
        if hedge_ratio is None:
            hedge_ratio = self._estimate_hedge_ratio(y, x, method)

        spread = y - hedge_ratio * x

        if isinstance(series1, pd.Series):
            return pd.Series(spread, index=series1.index, name='spread')
        return spread

    def calculate_zscore(
        self,
        spread: Union[pd.Series, np.ndarray],
        window: int = 20,
    ) -> pd.Series:
        """
        Calculate rolling z-score of spread.

        z-score = (spread - rolling_mean) / rolling_std

        Args:
            spread: Spread series
            window: Rolling window size

        Returns:
            Z-score series
        """
        spread = pd.Series(spread)

        rolling_mean = spread.rolling(window=window, min_periods=window).mean()
        rolling_std = spread.rolling(window=window, min_periods=window).std()

        zscore = (spread - rolling_mean) / (rolling_std + 1e-8)

        return zscore

    def _estimate_hedge_ratio(
        self,
        y: np.ndarray,
        x: np.ndarray,
        method: str = 'ols',
    ) -> float:
        """
        Estimate hedge ratio using various methods.

        Methods:
        - ols: Ordinary Least Squares (standard)
        - tls: Total Least Squares (errors in both variables)
        """
        # Remove NaN
        valid = ~(np.isnan(y) | np.isnan(x))
        y = y[valid]
        x = x[valid]

        if method == 'ols':
            X = np.column_stack([np.ones(len(x)), x])
            model = OLS(y, X).fit()
            return model.params[1]

        elif method == 'tls':
            # Total Least Squares (orthogonal regression)
            data = np.column_stack([x, y])
            mean = data.mean(axis=0)
            data_centered = data - mean

            _, _, Vt = np.linalg.svd(data_centered)
            v = Vt[-1]

            return -v[0] / v[1]

        else:
            raise ValueError(f"Unknown method: {method}")

    def calculate_half_life(
        self,
        spread: Union[pd.Series, np.ndarray],
    ) -> float:
        """
        Estimate mean reversion half-life.

        Fits an AR(1) model: spread_t = alpha + beta * spread_{t-1} + epsilon
        Half-life = -ln(2) / ln(beta)

        Args:
            spread: Spread series

        Returns:
            Half-life in bars
        """
        spread = np.asarray(spread)

        # Remove NaN
        spread = spread[~np.isnan(spread)]

        # AR(1) regression
        y = spread[1:]
        x = spread[:-1]

        X = np.column_stack([np.ones(len(x)), x])
        model = OLS(y, X).fit()
        beta = model.params[1]

        # Half-life = -ln(2) / ln(beta)
        if 0 < beta < 1:
            half_life = -np.log(2) / np.log(beta)
        else:
            half_life = np.inf

        return half_life


class OrnsteinUhlenbeckEstimator:
    """
    Ornstein-Uhlenbeck process parameter estimation.

    The OU process models mean reversion:
    dX = theta * (mu - X) * dt + sigma * dW

    where:
    - theta: Mean reversion speed
    - mu: Long-term mean
    - sigma: Volatility
    - dW: Wiener process increment

    Example:
        estimator = OrnsteinUhlenbeckEstimator()
        params = estimator.fit(spread_series)

        print(f"Half-life: {params.half_life:.1f} bars")
        print(f"Mean: {params.mu:.4f}")
    """

    def __init__(self, dt: float = 1.0):
        """
        Initialize estimator.

        Args:
            dt: Time step (1.0 for discrete observations)
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required for OU estimation")

        self.dt = dt
        self.params: Optional[OUParameters] = None

    def fit(
        self,
        series: Union[pd.Series, np.ndarray],
        method: str = 'ols',
    ) -> OUParameters:
        """
        Estimate OU parameters from time series.

        Args:
            series: Mean-reverting time series
            method: 'ols' (simple) or 'mle' (maximum likelihood)

        Returns:
            OUParameters with estimated values
        """
        series = np.asarray(series)
        series = series[~np.isnan(series)]

        if method == 'ols':
            params = self._fit_ols(series)
        elif method == 'mle':
            params = self._fit_mle(series)
        else:
            raise ValueError(f"Unknown method: {method}")

        self.params = params
        return params

    def _fit_ols(self, series: np.ndarray) -> OUParameters:
        """
        OLS estimation of OU parameters.

        Uses AR(1) regression: X_t = a + b * X_{t-1} + epsilon
        Then converts to continuous OU parameters.
        """
        y = series[1:]
        x = series[:-1]

        X = np.column_stack([np.ones(len(x)), x])
        model = OLS(y, X).fit()

        a = model.params[0]  # Intercept
        b = model.params[1]  # AR coefficient
        residual_std = np.std(model.resid)

        # Convert to OU parameters
        # theta = -ln(b) / dt
        # mu = a / (1 - b)
        # sigma = residual_std * sqrt(-2*ln(b) / (dt * (1-b^2)))

        if 0 < b < 1:
            theta = -np.log(b) / self.dt
            mu = a / (1 - b)
            sigma = residual_std * np.sqrt(-2 * np.log(b) / (self.dt * (1 - b**2)))
            half_life = np.log(2) / theta
        else:
            # Not mean-reverting
            theta = 0.0
            mu = np.mean(series)
            sigma = np.std(series)
            half_life = np.inf

        return OUParameters(
            theta=theta,
            mu=mu,
            sigma=sigma,
            half_life=half_life,
        )

    def _fit_mle(self, series: np.ndarray) -> OUParameters:
        """Maximum likelihood estimation of OU parameters."""

        def negative_log_likelihood(params, data, dt):
            theta, mu, sigma = params

            if theta <= 0 or sigma <= 0:
                return 1e10

            n = len(data) - 1
            x = data[:-1]
            y = data[1:]

            # Expected value and variance
            exp_theta_dt = np.exp(-theta * dt)
            mean = mu + (x - mu) * exp_theta_dt
            var = sigma**2 / (2 * theta) * (1 - exp_theta_dt**2)

            if np.any(var <= 0):
                return 1e10

            # Log-likelihood
            ll = -0.5 * n * np.log(2 * np.pi)
            ll -= 0.5 * np.sum(np.log(var))
            ll -= 0.5 * np.sum((y - mean)**2 / var)

            return -ll

        # Initial guess from OLS
        ols_params = self._fit_ols(series)
        x0 = [ols_params.theta, ols_params.mu, ols_params.sigma]

        # Bounds
        bounds = [(1e-6, None), (None, None), (1e-6, None)]

        # Optimize
        result = minimize(
            negative_log_likelihood,
            x0,
            args=(series, self.dt),
            method='L-BFGS-B',
            bounds=bounds,
        )

        theta, mu, sigma = result.x
        half_life = np.log(2) / theta if theta > 0 else np.inf

        return OUParameters(
            theta=theta,
            mu=mu,
            sigma=sigma,
            half_life=half_life,
        )

    def expected_value(
        self,
        x0: float,
        t: float,
    ) -> float:
        """
        Expected value at time t given X_0 = x0.

        E[X_t | X_0] = mu + (x0 - mu) * exp(-theta * t)
        """
        if self.params is None:
            raise ValueError("Model not fitted")

        return self.params.mu + (x0 - self.params.mu) * np.exp(-self.params.theta * t)

    def variance(self, t: float) -> float:
        """
        Variance at time t.

        Var[X_t] = sigma^2 / (2*theta) * (1 - exp(-2*theta*t))
        """
        if self.params is None:
            raise ValueError("Model not fitted")

        return (self.params.sigma**2 / (2 * self.params.theta) *
                (1 - np.exp(-2 * self.params.theta * t)))

    def simulate(
        self,
        x0: float,
        n_steps: int,
        dt: float = None,
        random_state: int = None,
    ) -> np.ndarray:
        """
        Simulate OU process path.

        Args:
            x0: Initial value
            n_steps: Number of steps
            dt: Time step (uses fitted dt if None)
            random_state: Random seed

        Returns:
            Simulated path
        """
        if self.params is None:
            raise ValueError("Model not fitted")

        dt = dt or self.dt

        if random_state is not None:
            np.random.seed(random_state)

        path = np.zeros(n_steps)
        path[0] = x0

        for t in range(1, n_steps):
            dW = np.random.normal(0, np.sqrt(dt))
            path[t] = (path[t-1] +
                      self.params.theta * (self.params.mu - path[t-1]) * dt +
                      self.params.sigma * dW)

        return path


# Convenience functions
def test_cointegration(
    series1: pd.Series,
    series2: pd.Series,
    significance: float = 0.05,
) -> CointegrationResult:
    """Quick cointegration test."""
    analyzer = CointegrationAnalyzer(significance)
    return analyzer.engle_granger_test(series1, series2)


def calculate_spread_zscore(
    series1: pd.Series,
    series2: pd.Series,
    window: int = 20,
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate spread and z-score for a pair.

    Returns:
        Tuple of (spread, zscore)
    """
    analyzer = CointegrationAnalyzer()
    result = analyzer.engle_granger_test(series1, series2)
    spread = analyzer.calculate_spread(series1, series2, result.hedge_ratio)
    zscore = analyzer.calculate_zscore(spread, window)
    return spread, zscore
