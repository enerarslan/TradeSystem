"""
Monte Carlo Analysis and Statistical Tests for Backtesting.

This module provides institutional-grade statistical analysis including:
- Block bootstrap for autocorrelated returns
- Monte Carlo simulation of equity paths
- Confidence intervals for performance metrics
- Probabilistic Sharpe Ratio
- Deflated Sharpe Ratio (multiple testing adjustment)
- Minimum Track Record Length

Based on:
- Bailey, D.H. and López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier"
- Bailey, D.H. and López de Prado, M. (2014). "The Deflated Sharpe Ratio"

Designed for JPMorgan-level requirements:
- Statistical rigor in performance evaluation
- Multiple testing adjustment
- Proper handling of autocorrelation
- Confidence interval estimation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from scipy import stats
    from scipy.special import erfc
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class ConfidenceInterval:
    """Confidence interval for a metric."""
    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    std_error: float


@dataclass
class MonteCarloResult:
    """Result of Monte Carlo simulation."""
    point_estimate: float
    mean: float
    std: float
    confidence_intervals: Dict[float, ConfidenceInterval]
    simulated_values: np.ndarray
    n_simulations: int


@dataclass
class ProbabilisticSharpeResult:
    """Result of Probabilistic Sharpe Ratio calculation."""
    observed_sharpe: float
    benchmark_sharpe: float
    probability: float  # P(true Sharpe > benchmark)
    is_significant: bool
    min_track_record: int  # Minimum observations needed


@dataclass
class DeflatedSharpeResult:
    """Result of Deflated Sharpe Ratio calculation."""
    observed_sharpe: float
    deflated_sharpe: float
    expected_max_sharpe: float
    haircut: float
    n_trials: int
    is_significant: bool


class MonteCarloAnalyzer:
    """
    Monte Carlo analysis for performance evaluation.

    Provides:
    - Bootstrap simulation for confidence intervals
    - Path simulation for risk analysis
    - Worst-case scenario analysis
    - Distribution fitting

    Example:
        analyzer = MonteCarloAnalyzer(n_simulations=10000)

        # Get Sharpe ratio confidence interval
        ci = analyzer.confidence_intervals(sharpe_ratio, returns)

        # Simulate equity paths
        paths = analyzer.simulate_paths(returns, n_paths=1000)

        # Worst-case analysis
        worst = analyzer.worst_case_analysis(returns, percentile=5)
    """

    def __init__(
        self,
        n_simulations: int = 10000,
        confidence_levels: List[float] = None,
        random_state: int = None,
    ):
        """
        Initialize Monte Carlo analyzer.

        Args:
            n_simulations: Number of simulations to run
            confidence_levels: Confidence levels for intervals
            random_state: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.confidence_levels = confidence_levels or [0.90, 0.95, 0.99]
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

    def bootstrap_returns(
        self,
        returns: Union[pd.Series, np.ndarray],
        block_size: int = 20,
        n_simulations: int = None,
    ) -> np.ndarray:
        """
        Block bootstrap for autocorrelated returns.

        Standard bootstrap assumes i.i.d. data, which financial returns
        typically violate. Block bootstrap preserves autocorrelation
        by sampling contiguous blocks.

        Args:
            returns: Return series
            block_size: Size of blocks to sample
            n_simulations: Number of bootstrap samples

        Returns:
            Array of bootstrapped return series (n_simulations, n_observations)
        """
        n_simulations = n_simulations or self.n_simulations
        returns = np.asarray(returns)
        n = len(returns)

        # Number of blocks needed
        n_blocks = int(np.ceil(n / block_size))

        # Generate bootstrap samples
        bootstrapped = np.zeros((n_simulations, n))

        for i in range(n_simulations):
            # Randomly select block starting positions
            block_starts = np.random.randint(0, n - block_size + 1, n_blocks)

            # Concatenate blocks
            sample = []
            for start in block_starts:
                sample.extend(returns[start:start + block_size])

            bootstrapped[i, :] = sample[:n]

        return bootstrapped

    def simulate_paths(
        self,
        returns: Union[pd.Series, np.ndarray],
        n_periods: int = None,
        method: str = 'block_bootstrap',
    ) -> np.ndarray:
        """
        Simulate equity paths using bootstrap.

        Args:
            returns: Historical returns
            n_periods: Number of periods to simulate (None = same as input)
            method: 'block_bootstrap', 'parametric', or 'empirical'

        Returns:
            Array of simulated cumulative returns (n_simulations, n_periods)
        """
        returns = np.asarray(returns)
        n_periods = n_periods or len(returns)

        if method == 'block_bootstrap':
            # Block bootstrap
            sim_returns = self.bootstrap_returns(
                returns,
                block_size=min(20, n_periods // 5),
            )

        elif method == 'parametric':
            # Fit normal distribution
            mu = np.mean(returns)
            sigma = np.std(returns)
            sim_returns = np.random.normal(
                mu, sigma, (self.n_simulations, n_periods)
            )

        elif method == 'empirical':
            # Random sampling from empirical distribution
            sim_returns = np.random.choice(
                returns,
                size=(self.n_simulations, n_periods),
                replace=True,
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        # Calculate cumulative returns (equity paths)
        paths = np.cumprod(1 + sim_returns, axis=1)

        return paths

    def confidence_intervals(
        self,
        metric_func: Callable[[np.ndarray], float],
        returns: Union[pd.Series, np.ndarray],
        block_size: int = 20,
    ) -> MonteCarloResult:
        """
        Calculate confidence intervals for any metric using bootstrap.

        Args:
            metric_func: Function that takes returns and returns a scalar
            returns: Return series
            block_size: Block size for block bootstrap

        Returns:
            MonteCarloResult with point estimate and confidence intervals
        """
        returns = np.asarray(returns)

        # Point estimate
        point_estimate = metric_func(returns)

        # Bootstrap samples
        boot_returns = self.bootstrap_returns(returns, block_size)

        # Calculate metric for each bootstrap sample
        boot_metrics = np.array([metric_func(r) for r in boot_returns])

        # Remove any NaN or infinite values
        boot_metrics = boot_metrics[np.isfinite(boot_metrics)]

        # Calculate statistics
        mean = np.mean(boot_metrics)
        std = np.std(boot_metrics)

        # Confidence intervals
        confidence_intervals = {}
        for level in self.confidence_levels:
            alpha = 1 - level
            lower = np.percentile(boot_metrics, alpha / 2 * 100)
            upper = np.percentile(boot_metrics, (1 - alpha / 2) * 100)

            confidence_intervals[level] = ConfidenceInterval(
                point_estimate=point_estimate,
                lower=lower,
                upper=upper,
                confidence_level=level,
                std_error=std,
            )

        return MonteCarloResult(
            point_estimate=point_estimate,
            mean=mean,
            std=std,
            confidence_intervals=confidence_intervals,
            simulated_values=boot_metrics,
            n_simulations=len(boot_metrics),
        )

    def worst_case_analysis(
        self,
        returns: Union[pd.Series, np.ndarray],
        percentile: float = 5,
    ) -> Dict[str, float]:
        """
        Analyze worst-case scenarios using Monte Carlo.

        Args:
            returns: Return series
            percentile: Percentile for worst-case (e.g., 5 = worst 5%)

        Returns:
            Dictionary with worst-case metrics
        """
        # Simulate paths
        paths = self.simulate_paths(returns)

        # Calculate metrics for each path
        final_values = paths[:, -1]
        max_drawdowns = self._calculate_path_drawdowns(paths)

        # Get percentiles
        worst_return = np.percentile(final_values, percentile) - 1
        worst_drawdown = np.percentile(max_drawdowns, 100 - percentile)

        # Probability of loss
        prob_loss = np.mean(final_values < 1)

        # Probability of severe drawdown (>20%)
        prob_severe_dd = np.mean(max_drawdowns > 0.20)

        # VaR and CVaR from simulations
        sorted_returns = np.sort(final_values - 1)
        var_index = int(percentile / 100 * len(sorted_returns))
        var = -sorted_returns[var_index]
        cvar = -np.mean(sorted_returns[:var_index])

        return {
            'worst_case_return': worst_return,
            'worst_case_drawdown': worst_drawdown,
            'probability_of_loss': prob_loss,
            'probability_severe_drawdown': prob_severe_dd,
            'var': var,
            'cvar': cvar,
            'median_return': np.median(final_values) - 1,
            'mean_return': np.mean(final_values) - 1,
        }

    def _calculate_path_drawdowns(self, paths: np.ndarray) -> np.ndarray:
        """Calculate maximum drawdown for each simulated path."""
        max_drawdowns = np.zeros(len(paths))

        for i, path in enumerate(paths):
            peak = np.maximum.accumulate(path)
            drawdown = (peak - path) / peak
            max_drawdowns[i] = np.max(drawdown)

        return max_drawdowns

    def drawdown_distribution(
        self,
        returns: Union[pd.Series, np.ndarray],
    ) -> pd.DataFrame:
        """
        Get distribution of maximum drawdowns.

        Returns DataFrame with percentiles of max drawdown distribution.
        """
        paths = self.simulate_paths(returns)
        max_drawdowns = self._calculate_path_drawdowns(paths)

        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

        data = {
            'percentile': percentiles,
            'max_drawdown': [np.percentile(max_drawdowns, p) for p in percentiles],
        }

        return pd.DataFrame(data)


class StatisticalTests:
    """
    Statistical tests for backtesting validation.

    Provides:
    - Probabilistic Sharpe Ratio
    - Deflated Sharpe Ratio
    - Minimum Track Record Length
    - Strategy comparison tests
    """

    def __init__(self):
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required for statistical tests")

    def probabilistic_sharpe_ratio(
        self,
        observed_sharpe: float,
        benchmark_sharpe: float,
        n_observations: int,
        skewness: float = 0,
        kurtosis: float = 3,  # Excess kurtosis (normal = 0)
    ) -> ProbabilisticSharpeResult:
        """
        Calculate Probabilistic Sharpe Ratio.

        Gives the probability that the true Sharpe ratio exceeds
        a benchmark, accounting for estimation error.

        PSR = Phi((SR - SR*) / sigma_SR)

        where sigma_SR accounts for non-normal returns.

        Args:
            observed_sharpe: Observed Sharpe ratio
            benchmark_sharpe: Benchmark Sharpe to beat
            n_observations: Number of observations
            skewness: Sample skewness (0 for normal)
            kurtosis: Excess kurtosis (0 for normal)

        Returns:
            ProbabilisticSharpeResult
        """
        # Adjust kurtosis if given as full kurtosis
        if kurtosis > 2:
            kurtosis = kurtosis - 3  # Convert to excess kurtosis

        # Standard error of Sharpe ratio
        # From Lo (2002) and Bailey & López de Prado (2012)
        sigma_sr = np.sqrt(
            (1 + 0.5 * observed_sharpe**2 -
             skewness * observed_sharpe +
             (kurtosis / 4) * observed_sharpe**2) /
            (n_observations - 1)
        )

        # Z-score
        z = (observed_sharpe - benchmark_sharpe) / sigma_sr

        # Probability (one-sided test)
        probability = stats.norm.cdf(z)

        # Minimum track record length
        min_track_record = self.minimum_track_record_length(
            observed_sharpe,
            benchmark_sharpe,
            skewness,
            kurtosis,
        )

        return ProbabilisticSharpeResult(
            observed_sharpe=observed_sharpe,
            benchmark_sharpe=benchmark_sharpe,
            probability=probability,
            is_significant=probability > 0.95,  # 95% confidence
            min_track_record=min_track_record,
        )

    def deflated_sharpe_ratio(
        self,
        observed_sharpe: float,
        n_trials: int,
        n_observations: int,
        variance_of_trials: float = None,
        skewness: float = 0,
        kurtosis: float = 0,
    ) -> DeflatedSharpeResult:
        """
        Calculate Deflated Sharpe Ratio.

        Adjusts Sharpe ratio for multiple testing (strategy selection bias).
        When you test many strategies, some will have high Sharpe by chance.

        Args:
            observed_sharpe: Observed Sharpe ratio
            n_trials: Number of strategies tested
            n_observations: Number of observations
            variance_of_trials: Variance of Sharpe ratios across trials
                               (if None, assumes all independent)
            skewness: Sample skewness
            kurtosis: Excess kurtosis

        Returns:
            DeflatedSharpeResult
        """
        # Expected maximum Sharpe under null (all strategies have zero alpha)
        # E[max(Z_1, ..., Z_N)] for standard normal Z
        if variance_of_trials is None:
            variance_of_trials = 1.0

        # Approximate expected maximum using Euler-Mascheroni constant
        euler_mascheroni = 0.5772156649

        # For large n_trials
        expected_max = np.sqrt(2 * np.log(n_trials)) - \
                       (np.log(np.pi) + euler_mascheroni) / \
                       (2 * np.sqrt(2 * np.log(n_trials)))

        # Adjust for observation count
        sigma_sr = np.sqrt(
            (1 + 0.5 * observed_sharpe**2 -
             skewness * observed_sharpe +
             (kurtosis / 4) * observed_sharpe**2) /
            (n_observations - 1)
        )

        expected_max_sharpe = expected_max * sigma_sr

        # Deflated Sharpe (haircut for multiple testing)
        haircut = expected_max_sharpe / max(observed_sharpe, 1e-6)
        deflated_sharpe = max(0, observed_sharpe - expected_max_sharpe)

        # Is it still significant after deflation?
        psr = self.probabilistic_sharpe_ratio(
            deflated_sharpe, 0, n_observations, skewness, kurtosis
        )

        return DeflatedSharpeResult(
            observed_sharpe=observed_sharpe,
            deflated_sharpe=deflated_sharpe,
            expected_max_sharpe=expected_max_sharpe,
            haircut=haircut,
            n_trials=n_trials,
            is_significant=psr.probability > 0.95,
        )

    def minimum_track_record_length(
        self,
        observed_sharpe: float,
        benchmark_sharpe: float = 0,
        skewness: float = 0,
        kurtosis: float = 0,
        confidence: float = 0.95,
    ) -> int:
        """
        Calculate minimum track record length for statistical significance.

        How many observations are needed to be confident that
        the true Sharpe exceeds the benchmark?

        Args:
            observed_sharpe: Observed Sharpe ratio
            benchmark_sharpe: Benchmark to beat
            skewness: Sample skewness
            kurtosis: Excess kurtosis
            confidence: Required confidence level

        Returns:
            Minimum number of observations needed
        """
        if observed_sharpe <= benchmark_sharpe:
            return np.inf  # Can never be significant

        z_alpha = stats.norm.ppf(confidence)

        # From Bailey & López de Prado (2012)
        numerator = z_alpha**2 * (
            1 + 0.5 * observed_sharpe**2 -
            skewness * observed_sharpe +
            (kurtosis / 4) * observed_sharpe**2
        )
        denominator = (observed_sharpe - benchmark_sharpe)**2

        min_trl = numerator / denominator + 1

        return int(np.ceil(min_trl))

    def compare_strategies(
        self,
        returns1: np.ndarray,
        returns2: np.ndarray,
        alpha: float = 0.05,
    ) -> Dict[str, float]:
        """
        Compare two strategies statistically.

        Tests whether the difference in Sharpe ratios is significant.

        Args:
            returns1: Returns of strategy 1
            returns2: Returns of strategy 2
            alpha: Significance level

        Returns:
            Dictionary with test results
        """
        # Calculate Sharpe ratios
        sr1 = np.mean(returns1) / (np.std(returns1) + 1e-8) * np.sqrt(252)
        sr2 = np.mean(returns2) / (np.std(returns2) + 1e-8) * np.sqrt(252)

        # Paired difference
        diff = returns1 - returns2
        diff_mean = np.mean(diff)
        diff_std = np.std(diff)
        n = len(diff)

        # T-statistic
        t_stat = diff_mean / (diff_std / np.sqrt(n))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))

        return {
            'sharpe_1': sr1,
            'sharpe_2': sr2,
            'sharpe_diff': sr1 - sr2,
            'mean_diff': diff_mean,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < alpha,
            'strategy_1_better': sr1 > sr2 and p_value < alpha,
        }


# Convenience functions
def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Calculate annualized Sharpe ratio."""
    return np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(periods_per_year)


def get_sharpe_confidence_interval(
    returns: Union[pd.Series, np.ndarray],
    confidence: float = 0.95,
    n_simulations: int = 10000,
) -> ConfidenceInterval:
    """
    Get confidence interval for Sharpe ratio.

    Args:
        returns: Return series
        confidence: Confidence level
        n_simulations: Number of bootstrap samples

    Returns:
        ConfidenceInterval
    """
    analyzer = MonteCarloAnalyzer(n_simulations=n_simulations)
    result = analyzer.confidence_intervals(
        lambda r: sharpe_ratio(r),
        returns,
    )
    return result.confidence_intervals.get(confidence)


def is_sharpe_significant(
    returns: Union[pd.Series, np.ndarray],
    benchmark: float = 0,
    confidence: float = 0.95,
) -> Tuple[bool, float]:
    """
    Test if Sharpe ratio is significantly different from benchmark.

    Returns:
        Tuple of (is_significant, probability)
    """
    returns = np.asarray(returns)
    observed = sharpe_ratio(returns)

    # Calculate higher moments
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)

    tests = StatisticalTests()
    result = tests.probabilistic_sharpe_ratio(
        observed,
        benchmark,
        len(returns),
        skewness,
        kurtosis,
    )

    return result.probability > confidence, result.probability
